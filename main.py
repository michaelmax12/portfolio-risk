from src import Helper, Custom
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage
import plotly.express as px
import plotly.figure_factory as ff
import yfinance as yf

st.set_page_config(layout="wide", page_title="Risk Dashboard")

PRIMARY_COLOR  = "#0072B5"
SECONDARY_COLOR = "#B54300"

helper_instance = Helper.HELPER()
custom_instance = Custom.Custom_optimize()

# ── Cached helpers ─────────────────────────────────────────────────────────────

@st.cache_data
def Bl_input_data(edited_df):
    views = dict(
        zip(
            edited_df.loc[edited_df["Use_View"], "Asset"],
            edited_df.loc[edited_df["Use_View"], "Expected_Return"],
        )
    )
    return views


@st.cache_data
def get_data(tickers, period, max_stale_days: int = 61, max_flat_days: int = 61):
    df = helper_instance.data_downloader(tickers, period, "1d")
    df = df.dropna(axis=1, thresh=int(len(df) * 0.9))
    df = df.dropna()
    close = df["Close"].astype(int)
    last_date = close.index[-1]
    today = pd.Timestamp.now(tz=last_date.tzinfo)

    stale_cols = [
        col for col in close.columns
        if (today - close[col].last_valid_index()).days > max_stale_days
    ]
    flat_cols = [
        col for col in close.columns
        if (close[col].tail(max_flat_days) == close[col].tail(max_flat_days).iloc[0]).all()
    ]
    drop_cols = set(stale_cols) | set(flat_cols)
    if drop_cols:
        st.warning(f"Dropping stale/flat tickers: {drop_cols}")
        close = close.drop(columns=drop_cols)
    return close


# FIX: cache market caps with TTL so yfinance is not hammered on every re-render
@st.cache_data(ttl=3600)
def get_market_caps_cached(tickers_tuple, prices_tuple):
    """
    Wraps helper_instance.get_market_caps but accepts hashable args for caching.
    tickers_tuple : tuple of ticker strings
    prices_tuple  : tuple of last prices matching tickers_tuple order
    """
    import time
    import numpy as np

    mcaps = {}
    tickers = list(tickers_tuple)

    # ── 1. Batch fast_info (lightweight, no crumb/cookie) ──────────────────
    try:
        batch = yf.Tickers(" ".join(tickers))
        for c in tickers:
            try:
                info  = batch.tickers[c].fast_info
                shares = getattr(info, "shares", None)
                mcap   = getattr(info, "market_cap", None)
                last_price = dict(zip(tickers_tuple, prices_tuple))[c]
                if shares and last_price:
                    mcaps[c] = shares * last_price
                elif mcap:
                    mcaps[c] = mcap
                else:
                    mcaps[c] = None
            except Exception:
                mcaps[c] = None
        if all(v is not None for v in mcaps.values()):
            return mcaps
    except Exception as e:
        st.warning(f"Batch fast_info failed ({e}). Falling back to per-ticker.")

    # ── 2. Per-ticker fallback with exponential backoff ────────────────────
    price_map = dict(zip(tickers_tuple, prices_tuple))
    for c in tickers:
        if mcaps.get(c) is not None:
            continue
        for attempt in range(3):
            try:
                info = yf.Ticker(c).fast_info
                shares = getattr(info, "shares", None)
                mcap   = getattr(info, "market_cap", None)
                last_price = price_map[c]
                if shares and last_price:
                    mcaps[c] = shares * last_price
                elif mcap:
                    mcaps[c] = mcap
                else:
                    mcaps[c] = None
                time.sleep(0.5)
                break
            except Exception as e:
                wait = 2 ** (attempt + 1)
                st.warning(f"Rate limit for {c} (attempt {attempt+1}): waiting {wait}s…")
                time.sleep(wait)
        else:
            mcaps[c] = None

    # ── 3. Equal-weight fallback for any still-missing tickers ────────────
    missing = [c for c, v in mcaps.items() if not v]
    if missing:
        valid = [v for v in mcaps.values() if v]
        avg   = float(np.mean(valid)) if valid else 1.0
        for c in missing:
            mcaps[c] = avg
        st.info(f"Used equal-weight fallback for: {missing}")

    return mcaps


@st.cache_data
def run_monte_carlo(data_series, days=252, sims=1000):
    return helper_instance.monte_carlo_gbm(data_series, days, sims)


@st.cache_data
def running_ef(data_series, days=252, method_ef="Basic", target="constant_variance"):
    return helper_instance.run_ef(data_series, days, method_ef, target)


@st.cache_data
def running_bl(data_series, market_prices, mcaps, views, risk_free_rate):
    return helper_instance.get_max_sharpe_bl(
        data_series, market_prices, mcaps, views, risk_free_rate
    )


@st.cache_data
def running_hrp(df):
    return helper_instance.get_rec_bipart(df)


# ── Layout ─────────────────────────────────────────────────────────────────────

col1, col2 = st.columns([1.2, 4])
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["Data", "Monte Carlo", "EF", "Black-Litterman", "HRP", "Portofolio", "Stock Analysis", "PPP"]
)

with st.sidebar:
    st.markdown("### Risk Parameters")
    text_input = st.text_area(
        "Enter Tickers (comma or newline separated)",
        value="BBRI.JK, BBCA.JK, ADRO.JK, BBNI.JK, ^JKSE",
        height=150,
    )
    text_input4 = st.text_input(
        "Enter Historical Period  e.g. 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max",
        value="5y",
    )
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        text_input2 = st.text_input("Enter Days", value="252")
    with sub_col2:
        text_input3 = st.text_input("Enter Simulations", value="1000")
    ticker_list = [
        item.strip().upper()
        for item in text_input.replace("\n", ",").split(",")
        if item.strip()
    ]

with col1:
    st.markdown("### Data Parameters")
    capital       = st.number_input("Capital (IDR)", value=1_000_000_000)
    risk_free_rate = st.number_input("Risk Free Rate", value=0.06)
    options_data  = ["In-Sample", "Full"]
    selection     = st.segmented_control("Data", options_data, selection_mode="single", default="In-Sample")
    p_data        = st.slider("IS Split:", 0.5, 0.9, 0.7)

# ── Main content ───────────────────────────────────────────────────────────────

with col2:
    if not ticker_list:
        st.markdown("### Please enter at least one ticker symbol.")
    else:
        df = get_data(ticker_list, text_input4)

        has_jkse = "^JKSE" in df.columns
        if has_jkse:
            IHSG = df["^JKSE"]
            df   = df.drop("^JKSE", axis=1)
        else:
            IHSG = None

        split_index = int(len(df) * p_data)
        train_df    = df.iloc[:split_index].copy()
        test_df     = df.iloc[split_index:].copy()
        data_new    = train_df if selection == "In-Sample" else df.copy()

        if has_jkse:
            IHSG_train = IHSG.iloc[:split_index].copy() if selection == "In-Sample" else IHSG.copy()
        else:
            IHSG_train = None

        # ── Helper: get market caps via cache ──────────────────────────────
        def _get_mcaps(price_df):
            tickers_t = tuple(price_df.columns.tolist())
            prices_t  = tuple(float(price_df[c].iloc[-1]) for c in tickers_t)
            return get_market_caps_cached(tickers_t, prices_t)

        # ── Tab 1: Historical Data ─────────────────────────────────────────
        with tab1:
            st.markdown("### Historical Data")
            st.dataframe(df, width="stretch")

        # ── Tab 2: Monte Carlo (single asset) ─────────────────────────────
        with tab2:
            option = st.selectbox("Choose Monte Carlo Symbol", df.columns)
            monte_t = run_monte_carlo(df[option], int(text_input2), int(text_input3))

            p5  = np.percentile(monte_t, 5,  axis=1)
            p50 = np.median(monte_t,        axis=1)
            p95 = np.percentile(monte_t, 95, axis=1)
            summary_df = pd.DataFrame({"5%": p5, "Median": p50, "95%": p95})

            st.markdown(f"### Monte Carlo Simulation ({option})")
            st.line_chart(summary_df)

            fig_mc = go.Figure()
            for i in range(monte_t.shape[1]):
                fig_mc.add_trace(go.Scattergl(
                    y=monte_t[:, i], mode="lines",
                    line=dict(width=1, color="rgba(0,191,255,0.15)")
                ))
            fig_mc.update_layout(showlegend=False, margin=dict(l=0,r=0,t=30,b=0), title="Monte Carlo Paths")
            st.plotly_chart(fig_mc, width="stretch", key="mc_chart")

        # ── Tab 3: Efficient Frontier ──────────────────────────────────────
        with tab3:
            option2 = st.selectbox("Choose Method", ("Basic", "Ledoit Wolf"))
            option_wolf = (
                st.selectbox("Choose Target", ("constant_variance", "single_factor", "constant_correlation"))
                if option2 == "Ledoit Wolf"
                else "constant_variance"
            )
            sub_col3, sub_col4 = st.columns([3, 1])

            ef_vols, ef_returns = running_ef(data_new, int(text_input2), method_ef=option2, target=option_wolf)
            ms_vol, ms_return, ms_weights = helper_instance.get_max_sharpe(
                data_new, int(text_input2), method_ef=option2,
                risk_free_rate=risk_free_rate, target=option_wolf
            )

            with sub_col3:
                fig_ef = go.Figure()
                fig_ef.add_trace(go.Scatter(
                    x=ef_vols, y=ef_returns, mode="lines",
                    line=dict(color="rgba(0,191,255,1)", width=3), name="Optimal Frontier"
                ))
                fig_ef.add_trace(go.Scatter(
                    x=[ms_vol], y=[ms_return], mode="markers",
                    marker=dict(size=18, color="red", symbol="star"), name="Max Sharpe"
                ))
                fig_ef.update_layout(
                    title="Optimized Efficient Frontier Curve",
                    xaxis_title="Expected Volatility (Risk)",
                    yaxis_title="Target Return",
                    template="plotly_dark",
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                st.plotly_chart(fig_ef, width="stretch", key="ef_curve_chart")

            with sub_col4:
                st.markdown("### Max Sharpe Allocation")
                weight_df = (
                    pd.DataFrame({"Allocation": ms_weights}, index=data_new.columns)
                    .sort_values("Allocation", ascending=False)
                )
                st.dataframe(weight_df.style.format("{:.2%}"), width="stretch")

            st.markdown("### Backtest on allocation")
            if selection == "In-Sample":
                split_date = df.index[split_index]
                cum_is     = df.iloc[:split_index] / df.iloc[:split_index].iloc[0]
                is_values  = (cum_is * (ms_weights * capital)).sum(axis=1)
                cum_oos    = df.iloc[split_index:] / df.iloc[split_index:].iloc[0]
                oos_values = (cum_oos * (ms_weights * is_values.iloc[-1])).sum(axis=1)
                total_val  = pd.concat([is_values.iloc[:-1], oos_values])
                bt_df      = pd.DataFrame({"Total_Value": total_val})
                bt_df["Type"] = np.where(bt_df.index < split_date, "IS", "OOS")
                fig_port = px.line(bt_df, y="Total_Value", color="Type", template="plotly_dark")
                fig_port.add_vline(x=split_date, line_dash="dot", line_color="yellow")
                st.plotly_chart(fig_port, width="stretch", key="ef_backtest_chart")
            else:
                st.info("Switch to In-Sample mode to view the backtest.")

        # ── Tab 4: Black-Litterman ─────────────────────────────────────────
        with tab4:
            st.markdown("### Black-Litterman Views")

            # FIX: always initialise views so tab6 never hits NameError
            views = {}

            if not has_jkse:
                st.warning("Please add ^JKSE to the ticker list to enable Black-Litterman.")
            else:
                view_df = pd.DataFrame({
                    "Asset":           df.columns.tolist(),
                    "Use_View":        [False] * len(df.columns),
                    "Expected_Return": [0.05]  * len(df.columns),
                })
                edited_df = st.data_editor(
                    view_df, hide_index=True, width="stretch",
                    column_config={
                        "Expected_Return": st.column_config.NumberColumn(format="%.2f", step=0.01)
                    },
                )
                views = Bl_input_data(edited_df)

                if st.button("Run Black-Litterman Analysis"):
                    with st.spinner("Calculating BL Posterior and running backtests…"):
                        m_cap      = _get_mcaps(train_df)
                        total_mcap = sum(m_cap.values())
                        prior_weights = {t: c / total_mcap for t, c in m_cap.items()}

                        bl_vol, bl_return, bl_weights = running_bl(
                            train_df, IHSG_train, m_cap, views, risk_free_rate
                        )
                        _, _, w_basic = helper_instance.get_max_sharpe(
                            train_df, int(text_input2), risk_free_rate=risk_free_rate, method_ef="Basic"
                        )
                        _, _, w_lw = helper_instance.get_max_sharpe(
                            train_df, int(text_input2), risk_free_rate=risk_free_rate, method_ef="Ledoit Wolf"
                        )
                        weights_bl_tab = helper_instance.get_rec_bipart(train_df)

                        comparison_df = (
                            pd.DataFrame({
                                "Market Prior": prior_weights,
                                "BL Posterior": dict(zip(df.columns, bl_weights)),
                            })
                            .reset_index()
                            .rename(columns={"index": "Asset"})
                        )
                        fig_weights = px.bar(
                            comparison_df, x="Asset", y=["Market Prior", "BL Posterior"],
                            barmode="group",
                            title="Weight Shift: Market Equilibrium vs. Your Views",
                            template="plotly_dark",
                            color_discrete_map={"Market Prior": "#7F7F7F", "BL Posterior": PRIMARY_COLOR},
                        )
                        st.plotly_chart(fig_weights, width="stretch", key="bl_weights_chart")

                        st.markdown("### Strategy Backtest (Forward-Walking PnL)")
                        cum_return_test = test_df / test_df.iloc[0]
                        pnl_bl = pd.DataFrame(index=test_df.index)
                        pnl_bl["Basic EF"]         = (cum_return_test * (w_basic       * capital)).sum(axis=1)
                        pnl_bl["Ledoit Wolf EF"]    = (cum_return_test * (w_lw          * capital)).sum(axis=1)
                        pnl_bl["Black-Litterman"]   = (cum_return_test * (bl_weights    * capital)).sum(axis=1)
                        pnl_bl["HRP"]               = (cum_return_test * (weights_bl_tab * capital)).sum(axis=1)

                        sharpe_bl = helper_instance.compute_sharpe(pnl_bl, risk_free_rate=risk_free_rate)
                        fig_pnl_bl = px.line(
                            pnl_bl, title="OOS Portfolio Growth (Nominal IDR)",
                            template="plotly_dark",
                            color_discrete_sequence=["#FF4B4B", "#FFA500", PRIMARY_COLOR, "#00FF00"],
                        )
                        st.plotly_chart(fig_pnl_bl, width="stretch", key="bl_pnl_chart")

                        st.markdown("### Optimized Allocation Table & OOS Sharpe")
                        st.dataframe(
                            comparison_df.set_index("Asset").style.format("{:.2%}"),
                            width="stretch",
                        )
                        st.dataframe(sharpe_bl.to_frame("Sharpe Ratio").style.format("{:.2f}"))

        # ── Tab 5: HRP ────────────────────────────────────────────────────
        with tab5:
            st.markdown("### HRP")
            daily_returns_hrp = data_new.pct_change().dropna()
            st.markdown("### Asset Clustering Dendrogram")
            fig_dend = ff.create_dendrogram(
                daily_returns_hrp.T.values,
                labels=daily_returns_hrp.columns.tolist(),
                distfun=helper_instance.get_hrp_distance,
                linkagefun=lambda x: linkage(x, method="single"),
                orientation="bottom",
            )
            fig_dend.update_layout(height=600, showlegend=False, xaxis_title="Tickers", yaxis_title="Distance")
            st.plotly_chart(fig_dend, width="stretch", key="hrp_dendrogram_chart")

            weights_hrp_tab = running_hrp(data_new)
            weight_HRP = (
                pd.DataFrame({"Allocation": weights_hrp_tab}, index=data_new.columns)
                .sort_values("Allocation", ascending=False)
            )
            st.markdown("### Weight Allocation")
            st.dataframe(weight_HRP.style.format("{:.2%}"), width="stretch")

            st.markdown("### Backtest")
            if selection == "In-Sample":
                weight_HRP["Nominal_IDR"] = weight_HRP["Allocation"] * capital
                close_HRP = df.copy()
                cum_ret_hrp = close_HRP / close_HRP.iloc[0]
                for i in close_HRP.columns:
                    close_HRP[f"Return_{i}"] = cum_ret_hrp[i] * weight_HRP.loc[i, "Nominal_IDR"]
                val_cols = [c for c in close_HRP.columns if c.startswith("Return_")]
                close_HRP["Total_Value"] = close_HRP[val_cols].sum(axis=1)
                close_HRP["PNL_Pct"]     = ((close_HRP["Total_Value"] / capital) - 1) * 100
                split_date = df.index[split_index]
                close_HRP["Type"] = np.where(close_HRP.index < split_date, "IS", "OOS")
                fig_hrp_bt = px.line(close_HRP, y="Total_Value", color="Type", template="plotly_dark")
                fig_hrp_bt.add_vline(x=split_date, line_dash="dot", line_color="yellow")
                st.plotly_chart(fig_hrp_bt, width="stretch", key="hrp_backtest_chart")
            else:
                st.info("Switch to In-Sample mode to view the backtest.")

        # ── Tab 6: Portfolio ───────────────────────────────────────────────
        with tab6:
            st.markdown("### Portfolio Analysis")

            m_cap      = _get_mcaps(train_df)
            total_mcap = sum(m_cap.values())

            bl_vol, bl_return, bl_weights = running_bl(
                train_df, IHSG_train, m_cap, views, risk_free_rate
            )
            _, _, w_basic = helper_instance.get_max_sharpe(
                train_df, int(text_input2), risk_free_rate=risk_free_rate, method_ef="Basic"
            )
            _, _, w_lw = helper_instance.get_max_sharpe(
                train_df, int(text_input2), risk_free_rate=risk_free_rate, method_ef="Ledoit Wolf"
            )
            weights_port = helper_instance.get_rec_bipart(train_df)

            cum_return = test_df / test_df.iloc[0]
            pnl_df = pd.DataFrame(index=test_df.index)
            pnl_df["Basic EF"]       = (cum_return * (w_basic      * capital)).sum(axis=1)
            pnl_df["Ledoit Wolf EF"] = (cum_return * (w_lw          * capital)).sum(axis=1)
            pnl_df["Black-Litterman"]= (cum_return * (bl_weights    * capital)).sum(axis=1)
            pnl_df["HRP"]            = (cum_return * (weights_port  * capital)).sum(axis=1)

            fig_pnl = px.line(
                pnl_df, title="Out-Of-Sample Portfolio Growth (Nominal IDR)",
                template="plotly_dark",
                color_discrete_sequence=["#FF4B4B", "#FFA500", PRIMARY_COLOR, "#00FF00"],
            )
            st.plotly_chart(fig_pnl, width="stretch", key="Every_pnl_chart")

            # ── Rebalancing backtest ───────────────────────────────────────
            sliding = st.number_input("Sliding (Warmup)", value=252)
            step    = st.number_input("Rebalance",        value=120)

            if st.button("Run Portfolio Backtest (rebalance)"):
                sliding_int = int(sliding)
                step_int    = int(step)
                loop_range  = list(range(sliding_int, len(df) - step_int, step_int))

                # FIX: guard against empty loop → pd.concat([]) crash
                if not loop_range:
                    st.warning(
                        f"Not enough data for a rebalancing backtest. "
                        f"Dataset has {len(df)} rows but needs at least "
                        f"Sliding ({sliding_int}) + Rebalance ({step_int}) = {sliding_int + step_int}. "
                        f"Please reduce Sliding/Rebalance or extend the date range."
                    )
                else:
                    results = []
                    portfolio_value = {
                        "Basic EF":        capital,
                        "Black-Litterman": capital,
                        "HRP":             capital,
                        "Ledoit Wolf EF":  capital,
                        "Equal":           capital,
                        "IHSG":            capital,
                        "PPP":             capital,
                    }

                    progress = st.progress(0, text="Running rebalancing backtest…")
                    for idx, a in enumerate(loop_range):
                        data       = df.iloc[a - sliding_int : a]
                        data_ihsg  = IHSG.iloc[a - sliding_int : a] if has_jkse else None
                        mc         = _get_mcaps(data)

                        _, _, wb   = helper_instance.get_max_sharpe(data, int(text_input2), risk_free_rate=risk_free_rate, method_ef="Basic")
                        _, _, wlw  = helper_instance.get_max_sharpe(data, int(text_input2), risk_free_rate=risk_free_rate, method_ef="Ledoit Wolf")
                        _, _, wbl  = helper_instance.get_max_sharpe_bl(data, data_ihsg, mc, views, risk_free_rate)
                        whrp       = helper_instance.get_rec_bipart(data)
                        weq        = pd.Series(1 / data.shape[1], index=data.columns)

                        X_w, ret_w, clean_cols = custom_instance.data_preparation(data, window=90)
                        theta_w    = custom_instance.optimize_theta(X_w, ret_w, gamma=5)
                        wmat       = custom_instance.getting_weight(X_w, theta_w)
                        wppp       = pd.Series(wmat[-1], index=clean_cols)
                        wppp       = wppp / wppp.sum()

                        oos_period = df.iloc[a : a + step_int]
                        oos_ihsg   = IHSG.iloc[a : a + step_int] if has_jkse else None
                        cum_ret    = oos_period / oos_period.iloc[0]
                        common_c   = [c for c in data.columns if c in clean_cols]

                        pw = pd.DataFrame(index=oos_period.index)
                        pw["Basic EF"]        = (cum_ret * (wb   * portfolio_value["Basic EF"])).sum(axis=1)
                        pw["Ledoit Wolf EF"]  = (cum_ret * (wlw  * portfolio_value["Ledoit Wolf EF"])).sum(axis=1)
                        pw["Black-Litterman"] = (cum_ret * (wbl  * portfolio_value["Black-Litterman"])).sum(axis=1)
                        pw["HRP"]             = (cum_ret * (whrp * portfolio_value["HRP"])).sum(axis=1)
                        pw["Equal"]           = (cum_ret * (weq  * portfolio_value["Equal"])).sum(axis=1)
                        pw["PPP"]             = (cum_ret[common_c] * (wppp * portfolio_value["PPP"])).sum(axis=1)
                        if has_jkse and oos_ihsg is not None:
                            cum_ihsg  = oos_ihsg / oos_ihsg.iloc[0]
                            pw["IHSG"] = cum_ihsg * portfolio_value["IHSG"]

                        results.append(pw)
                        for col in portfolio_value:
                            if col in pw.columns:
                                portfolio_value[col] = pw[col].iloc[-1]

                        progress.progress((idx + 1) / len(loop_range),
                                          text=f"Window {idx+1}/{len(loop_range)}")

                    progress.empty()
                    final_pnl = pd.concat(results)
                    fig_reb = px.line(final_pnl, x=final_pnl.index, y=final_pnl.columns,
                                     title="Rebalance Backtest", template="plotly_dark")
                    st.plotly_chart(fig_reb, width="stretch", key="Porto_rebalance_backtest")
                    sharpe_reb = helper_instance.compute_sharpe(final_pnl, risk_free_rate=risk_free_rate)
                    st.dataframe(sharpe_reb.to_frame("Sharpe Ratio").style.format("{:.2f}"))

            # ── MC for Portfolio ───────────────────────────────────────────
            option_mc = st.selectbox(
                "Choose Method for MC",
                ("Basic EF", "Ledoit Wolf EF", "Black-Litterman", "HRP"),
            )
            if st.button("Run MC for Portfolio"):
                # FIX: use .iloc[-1] — DatetimeIndex does not support [-1] label lookup
                initial_porto = pnl_df[option_mc].iloc[-1]
                monte_t_port  = run_monte_carlo(pnl_df[option_mc], int(text_input2), int(text_input3))
                statistic_porto = helper_instance.calculated_risk_metric(monte_t_port, initial_porto)

                st.markdown(f"### Monte Carlo Simulation ({option_mc})")
                st.line_chart(statistic_porto)

                fig_mc_port = go.Figure()
                for i in range(monte_t_port.shape[1]):
                    fig_mc_port.add_trace(go.Scattergl(
                        y=monte_t_port[:, i], mode="lines",
                        line=dict(width=1, color="rgba(0,191,255,0.15)")
                    ))
                fig_mc_port.update_layout(showlegend=False, margin=dict(l=0,r=0,t=30,b=0), title="Monte Carlo Paths")
                st.plotly_chart(fig_mc_port, width="stretch", key="Porto_mc_chart")

                final_sim_values = monte_t_port[-1, :]
                var_threshold    = statistic_porto["5%_Val"].iloc[-1]
                cvar_loss        = statistic_porto["CVaR_95"].iloc[-1]
                cvar_threshold   = initial_porto - cvar_loss

                fig_CVAR = px.histogram(x=final_sim_values, nbins=100,
                                        title="Distribution of Final Simulated Values")
                fig_CVAR.add_vline(x=var_threshold,  line_dash="dash", line_color="orange", annotation_text="VaR 95%")
                fig_CVAR.add_vline(x=cvar_threshold, line_dash="dash", line_color="red",    annotation_text="CVaR 95%")
                st.plotly_chart(fig_CVAR, key="CVAR_distribution")

                fig_risk_time = px.line(statistic_porto, y=["Median_Val", "5%_Val"])
                cvar_vals = initial_porto - statistic_porto["CVaR_95"]
                fig_risk_time.add_scatter(y=cvar_vals, mode="lines", name="CVaR_95",
                                          line=dict(dash="dot", color="red"))
                st.plotly_chart(fig_risk_time, key="CVAR_timeseries")

        # ── Tab 7: Stock Analysis ──────────────────────────────────────────
        with tab7:
            st.markdown("### Stock Analysis")
            if not has_jkse:
                st.warning("Please add ^JKSE to enable stock analysis vs. market.")
            else:
                analysis_data = {}
                for q in df.columns:
                    asset_r  = df[q].pct_change().dropna()
                    market_r = IHSG.pct_change().dropna()
                    alpha, idio_vol, ir = helper_instance.idio_vol_alpha(
                        asset_r, market_r, annualize=True, trading_days=int(text_input2)
                    )
                    analysis_data[q] = {
                        "Alpha": alpha,
                        "Idiosyncratic Volatility": idio_vol,
                        "IR": ir,
                    }
                idio = pd.DataFrame(analysis_data).T
                st.dataframe(
                    idio.style.format({
                        "Alpha":                    "{:.2%}",
                        "Idiosyncratic Volatility": "{:.2%}",
                        "IR":                       "{:.2f}",
                    })
                )

        # ── Tab 8: PPP ────────────────────────────────────────────────────
        with tab8:
            st.markdown("### PPP")
            X_ppp, ret_ppp, clean_cols_ppp = custom_instance.data_preparation(data_new, window=60)
            theta_ppp   = custom_instance.optimize_theta(X_ppp, ret_ppp, gamma=5)
            wmat_ppp    = custom_instance.getting_weight(X_ppp, theta_ppp)
            w_ppp_latest = pd.Series(wmat_ppp[-1], index=clean_cols_ppp)
            w_ppp_latest = w_ppp_latest / w_ppp_latest.sum()

            weight_PPP = pd.DataFrame({"Allocation": w_ppp_latest}, index=w_ppp_latest.index)
            st.dataframe(weight_PPP.style.format("{:.2%}"), width="stretch")

            if selection == "In-Sample":
                # FIX: use weight_PPP (not weight_HRP) and guard for tickers not in PPP universe
                weight_PPP["Nominal_IDR"] = weight_PPP["Allocation"] * capital
                close_ppp   = df.copy()
                cum_ret_ppp = close_ppp / close_ppp.iloc[0]
                for i in close_ppp.columns:
                    if i in weight_PPP.index:
                        close_ppp[f"Return_{i}"] = cum_ret_ppp[i] * weight_PPP.loc[i, "Nominal_IDR"]
                    else:
                        close_ppp[f"Return_{i}"] = 0.0
                val_cols_ppp = [c for c in close_ppp.columns if c.startswith("Return_")]
                close_ppp["Total_Value"] = close_ppp[val_cols_ppp].sum(axis=1)
                close_ppp["PNL_Pct"]     = ((close_ppp["Total_Value"] / capital) - 1) * 100
                split_date = df.index[split_index]
                close_ppp["Type"] = np.where(close_ppp.index < split_date, "IS", "OOS")
                fig_ppp = px.line(close_ppp, y="Total_Value", color="Type", template="plotly_dark",
                                  title="PPP Strategy Backtest")
                fig_ppp.add_vline(x=split_date, line_dash="dot", line_color="yellow")
                st.plotly_chart(fig_ppp, width="stretch", key="PPP_backtest_chart")
            else:
                st.info("Switch to In-Sample mode to view the backtest.")

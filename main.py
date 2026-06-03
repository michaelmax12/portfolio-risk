# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR MODULE  —  drop into your Risk Dashboard
#
# Integration (3 steps):
#   1. Put the CONSTANTS block near the top of your app (after imports).
#   2. Replace your existing `with st.sidebar:` block with the one below.
#   3. Everything downstream still works because `ticker_list` is built exactly
#      as before (a flat UPPER-cased list that includes ^JKSE), so the
#      `has_jkse = "^JKSE" in df.columns` checks keep functioning.
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st

# ── CONSTANTS ───────────────────────────────────────────────────────────────

# Full Indonesia Stock Exchange (IDX) universe. dict.fromkeys() de-dupes while
# preserving the original (roughly liquidity/market-cap) ordering so the most
# relevant tickers sit at the top of the dropdown.
_RAW_UNIVERSE = [
    "BBCA","DCII","BBRI","BREN","BMRI","MORA","BYAN","TLKM","AMMN","ASII",
    "BRPT","SRAJ","DNET","TPIA","BBNI","PANI","EMAS","SMMA","BNLI","CDIA",
    "IMPC","DSSA","BRIS","MPRO","UNTR","BRMS","ICBP","HMSP","CASA","CUAN",
    "CPIN","ANTM","ISAT","ADRO","AADI","UNVR","MDKA","ADMR","BUMI","INDF",
    "GOTO","NCKL","TCPI","MBMA","EXCL","SUPR","INCO","AMRT","PTRO","BELI",
    "BDMN","GEMS","PGAS","MTEL","PGUN","MEGA","INKP","MYOR","BNGA","CMRY",
    "EMTK","PGEO","KLBF","MLPT","ENRG","TBIG","GGRM","PTBA","VKTR","SILO",
    "ARCI","MEDC","TAPG","NISP","JPFA","SUPA","BINA","FAPA","ITMG","BTPN",
    "MSIN","MAPI","AKRA","TINS","GIAA","FILM","PNBN","MIKA","TOWR","BNBR",
    "MDIY","JSMR","BBHI","AVIA","SRTG","MKPI","SOHO","BUVA","MAPA","RISE",
    "ARKO","BBTN","TKIM","INTP","FASW","SCMA","ARTO","BBSI","BSIM","RAJA",
    "ULTJ","JARR","BNII","HEAL","RATU","JRPT","SMAR","PWON","CMNT","CARE",
    "DEWA","STTP","BSDE","ALII","RMKE","ELPI","MCOL","MGLV","MLBI","DSNG",
    "LIFE","PSAB","BUKA","BKSL","AALI","CITA","ADES","GOOD","YUPI","AUTO",
    "CTRA","COIN","INDY","ESSA","APIC","SIDO","HRTA","IBST","POWR","SMGR",
    "BIPI","TRIO","NSSS","STAA","WIFI","TSPC","HRUM","BFIN","BSSR","ADMF",
    "BBKP","PRAY","PACK","SMSM","EDGE","KPIG","CMNP","CLEO","MIDI","BMAS",
    "PLIN","INPP","RLCO","LSIP","SIMP","CNMA","AGII","WIKA","BJBR","SSIA",
    "CYBR","POLU","BBMD","BJTM","PNLF","DUTI","DMAS","CLAY","BTPS","SMCB",
    "TLDN","TMAS","SSMS","GMFI","ABMM","SMMT","MTDL","FORE","OMED","EPMT",
    "NICL","YULE","BULL","ERAA","ACES","SINI","WBSA","WSKT","DMND","ANJT",
    "SAME","PALM","UNIC","INET","BSWD","SGRO","BOGA","BHAT","MPMX","SMDR",
    "MAYA","LPKR","SHIP","BALI","SMRA","NATO","MASB","SGER","CTBN","DRMA",
    "BPII","RDTX","ELSA","JTPE","MBSS","MSTI","TGKA","GJTL","META","KRAS",
    "AGRO","EURO","DKFT","TUGU","SURE","MTLA","BIRD","TBLA","CBRE","PBID",
    "IMAS","HEXA","CASS","VICI","NIRO","LPPF","PSGO","TRIM","TOBA","AMAR",
    "BANK","TOTL","PKPK","GGRP","MAPB","BESS","MKAP","BBYB","SDRA","NOBU",
    "GOTOM","WIIM","MDIA","APLN","KEJU","FISH","MINA","ROTI","ARNA","LINK",
    "UANG","MNCN","KEEN","GOLF","PNIN","AGRS","SKRN","IFSH","OMRE","DAAZ",
    "ISSP","MARK","SRIL","CPRO","SAMF","JSPT","HATM","TFCO","MDLA","INPC",
    "SOCI","BGTG","ARGO","NETV","PNGO","RALS","KAEF","SFAN","HUMI","PORT",
    "BWPT","MCOR","PBSA","PYFA","DAYA","KIJA","BHIT","MYOH","LPCK","TOTO",
    "CENT","BLTZ","GTSI","ROCK","TRAM","SMDM","BACA","MAHA","MSJA","PRDA",
    "ASSA","DATA","IRSX","DWGL","PSKT","UDNG","VISI","ASRI","WINS","SMIL",
    "RIMO","BNBA","MGRO","OASA","BABP","BISI","IPCC","ABDA","MMLP","VICO",
    "JAWA","BCAP","TRIN","IFII","IATA","STAR","BUKK","BCIC","LPGI","NEST",
    "BMTR","FPNI","DNAR","MMIX","BRAM","AMAG","SCCO","DGWG","ASGR","MTMH",
    "PNBS","INDS","BKSW","CBUT","BOLT","MAIN","MBAP","BTEL","ALDO","BVIC",
    "KMTR","POLI","DVLA","GMTD","ITMA","MERK","NICE","SUNI","WOOD","IPCM",
    "CSRA","TPMA","DEPO","IIKP","KETR","IMJS","SMBR","UCID","KINO","MLIA",
    "PSSI","HERO","BMHS","DOID","SIPD","DLTA","PMJS","RONY","LUCY","SPTO",
    "CSAP","MKTR","ERAL","ACST","IPTV","ASLI","KING","PTSN","KKGI","INDR",
    "GSMF","TRGU","FMII","ADHI","SONA","ELTY","BLUE","CARS","JGLE","SKLT",
    "BEEF","AMFG","CFIN","DMMX","CEKA","MLPL","RAAM","BSBK","TEBE","KBLI",
    "TRST","PTPP","PANS","TDPM","DILD","SUGI","AYAM","NRCA","LTLS","BLES",
    "FUTR","PSAT","BLOG","KOTA","ATIC","JKON","BEKS","TIFA","NFCX","FAST",
    "TCID","CAMP","ADCP","HOME","DOOH","KDTN","WMPP","BBLD","ARTA","PADI",
    "PPRE","BUAH","JIHD","GRIA","BBRM","ARII","GWSA","AISA","GHON","WOMF",
    "MINE","MFMI","TBMS","BEST","TALF","LIVE","BKDP","BOLA","STRK","POLL",
    "BLTA","GZCO","PPRO","BUDI","RSGK","VOKS","SKBM","RELI","COCO","CITY",
    "FOLK","HGII","SPMA","WSBP","PDPP","ASLC","DPUM","ADMG","GDST","RODA",
    "PTPW","HITS","RANC","INRU","SHID","SOSS","UNSP","TAMU","CHIP","MTWI",
    "BELL","AMOR","MABA","PJAA","IPOL","BSML","ZINC","TRUE","KONI","IBOS",
    "CMPP","DGIK","SOTS","WIRG","KDSI","INDO","BPFI","KSIX","PTMR","WMUU",
    "NICK","WTON","MITI","RSCH","PGJO","FORU","SQMI","BDKR","PANR","SOFA",
    "BIKE","VIVA","ENAK","AXIO","URBN","LFLO","PBRX","LCGP","MDLN","SMGA",
    "NIKL","SMRU","MPPA","BBSS","IKBI","MOLI","BRNA","IRRA","EKAD","IDPR",
    "HOKI","BAIK","LEAD","PEVE","HDFA","GULA","ZATA","VERN","SSTM","PZZA",
    "AWAN","BTEK","NELY","MTFN","LABS","TRIS","AREA","SULI","ATAP","HBAT",
    "MREI","UNTD","CHEK","ESTA","SMKL","GLVA","APEX","SWID","FITT","TGUK",
    "INPS","MHKI","CLPI","ASPR","MAXI","WEGE","UVCR","MDKI","TIRT","VRNA",
    "GTBO","AMMS","YPAS","HAIS","GUNA","ARMY","MAGP","KMDS","GDYR","KOCI",
    "RMKO","SURI","BABY","PADA","KBRI","GPRA","SQBI","HALO","WINE","POSA",
    "PMUI","BOAT","AHAP","UFOE","IGAR","PNSE","CRAB","PIPA","NASA","CBPE",
    "CNKO","GRPM","RIGS","BAYU","DART","SCNP","GTRA","JECC","SRSN","INAF",
    "VTNY","NUSA","TYRE","ELIT","SOLA","MYTX","DADA","EAST","PJHB","VAST",
    "HYGN","KOBX","PDES","APLI","TFAS","FUJI","SATU","TIRA","JATI","ASRM",
    "BOBA","MGNA","BMSR","REAL","GPSO","CCSI","PEGE","KBLM","DYAN","UNIQ",
    "GOLD","BEER","MUTU","ETWA","HOMI","KOKA","AKPI","ASHA","KIAS","COAL",
    "IKPM","ATLA","BIPP","MPXL","CGAS","CASH","MICE","NAIK","TARA","IOTF",
    "CRSN","DGNS","PTPS","ZONE","WICO","KBAG","TRON","KREN","NZIA","AMIN",
    "HRME","ALMI","JMAS","LPLI","IKAI","ALKA","HAJJ","PART","SAGE","CSIS",
    "LAPD","SAFE","BTON","HOPE","BAJA","WGSH","ASPI","KLAS","BPTR","BATR",
    "ESTI","YOII","MDRN","INTA","OKAS","BINO","MEJA","AKKU","COWL","TOSK",
    "SDPC","PURA","AGAR","ASJT","TAMA","HILL","SMLE","EPAC","DEWI","PUDP",
    "PEHA","VINS","INOV","DUCK","BEBS","TRUS","ACRO","DOSS","MCAS","MSIE",
    "NINE","EMDE","RELF","CTTH","ERTX","POLA","ECII","DFAM","ITIC","NTBK",
    "APII","ASDM","NPGF","LION","PTSP","PTMP","KAQI","PPGL","MRAT","MTRA",
    "OBAT","ANDI","PSDN","TRUK","SLIS","DIVA","TRJA","CAKK","GOLL","MANG",
    "ZYRX","WEHA","GAMA","LPPS","HOTL","AKSI","LAND","GLOB","CINT","SDMU",
    "LPIN","RBMS","MTPS","SAPX","OBMD","PTIS","TAXI","PAMG","HKMU","DKHH",
    "PPRI","FWCT","KARW","ENZO","FIRE","BAPA","KBLV","LMPI","WOWS","ASMI",
    "KLIN","WAPO","YELO","RGAS","LAJU","MARI","TOOL","DSFI","INTD","ESIP",
    "ABBA","MEDS","RUIS","HELI","ASBI","BAUT","SEMA","INCI","WINR","GEMA",
    "PURI","MPIX","IPAC","NAYZ","PMMP","BAPI","MERI","KOPI","MSKY","ZBRA",
    "CPRI","MBTO","AYLS","FLMC","CANI","KUAS","CHEM","POOL","FOOD","MTSM",
    "ICON","TMPO","RAFI","NANO","OPMS","SNLK","LOPI","INAI","IBFN","KJEN",
    "LCKM","JSKY","PTDU","ISAP","SCPI","LABA","SBMA","KRYA","DPNS","OLIV",
    "JAYA","SICO","NASI","SMKM","ISEA","JAST","CBMF","MPOW","KIOS","BCIP",
    "DEFI","ENVY","PGLI","OILS","AIMS","PICO","RUNS","LUCK","RCCC","BATA",
    "HDIT","MIRA","MENN","LMAX","KOIN","TGRA","POLY","INCF","CSMI","PURE",
    "IDEA","BOSS","IKAN","TNCA","KKES","LRNA","TELE","IPPE","ARKA","GRPH",
    "TECH","SPRE","TRIL","BRRC","PLAS","DIGI","TAYS","KICI","BIMA","RICY",
    "SWAT","AEGS","PCAR","WIDI","OCAP","SOUL","ALTO","LMAS","INDX","HADE",
    "PLAN","BMBL","TOPS","BIKA","CNTB","SKYB","FIMP","LMSH","UNIT","SIMA",
    "ARTI","KAYU","TOYS","CNTX","DEAL","MKNT","SBAT","SUDI","ATPK","BRAU",
    "DMAD","FORZ","HDTX","JASS","JKSW","KPAL","KPAS","KRAH","MAMI","MAMIP",
    "MYRX","MYRXP","NIPS","PAFI","PRAS","SFTCASH","SING",
]
IDX_UNIVERSE = list(dict.fromkeys(_RAW_UNIVERSE))  # de-dupe, keep order

# Pre-selected tickers (matches your screenshot). ^JKSE intentionally excluded
# here — it now lives in the dedicated Benchmark picker below.
DEFAULT_TICKERS = [
    "AADI","AALI","ACES","ADMR","ADRO","AKRA","AMRT","ANTM","ARCI","ASII",
    "BBCA","BBNI","BBRI","BBTN","BFIN","BKSL","BMRI","BNGA","BRIS","BRMS",
    "BUKA","BUMI","CPIN","CTRA","DEWA","DMAS","DSNG","ELSA","EMAS","ENRG",
    "ERAA","ESSA","EXCL","GGRM","GOTO","HEAL","HMSP","HRTA","ICBP","INCO",
    "INDF","INKP","ISAT","ITMG","JPFA","JSMR","KLBF","LSIP","MAPA","MAPI",
    "MBMA","MDKA","MEDC","MYOR","NCKL","PGAS","PGEO","PTBA","PWON","SCMA",
    "SMDR","SMGR","TAPG","TINS","TKIM","TLKM","TOBA","TOWR","UNTR","UNVR",
    "WIFI",
]

# Benchmark indices available on Yahoo Finance. ^JKSE (IDX Composite / IHSG) is
# the only one your downstream code consumes, but you can add more here later.
BENCHMARK_OPTIONS = ["^JKSE"]
DEFAULT_BENCHMARKS = ["^JKSE"]

# Friendly label -> yfinance period code. Single-select.
PERIOD_MAP = {
    "1 Day":     "1d",
    "5 Days":    "5d",
    "1 Month":   "1mo",
    "3 Months":  "3mo",
    "6 Months":  "6mo",
    "1 Year":    "1y",
    "2 Years":   "2y",
    "5 Years":   "5y",
    "10 Years":  "10y",
    "YTD":       "ytd",
    "Max":       "max",
}
DEFAULT_PERIOD_LABEL = "5 Years"


# ── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Risk Parameters")

    # Stocks: pill/tag multiselect over the full IDX universe.
    selected_stocks = st.multiselect(
        "Stocks (IDX Universe)",
        options=IDX_UNIVERSE,
        default=DEFAULT_TICKERS,
        help="Search and add any ticker listed on the IDX. Click an X on a pill to remove it.",
    )

    # Benchmark: its own picker, defaulting to ^JKSE.
    selected_benchmarks = st.multiselect(
        "Benchmark",
        options=BENCHMARK_OPTIONS,
        default=DEFAULT_BENCHMARKS,
        help="Index used as the market benchmark (Black-Litterman, Stock Analysis, etc.).",
    )

    # Historical period: single-select button group with human-readable labels.
    period_label = st.segmented_control(
        "Historical Period",
        options=list(PERIOD_MAP.keys()),
        selection_mode="single",
        default=DEFAULT_PERIOD_LABEL,
    )
    # segmented_control can return None if the user de-selects; fall back to 5y.
    text_input4 = PERIOD_MAP.get(period_label, "5y")

    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        text_input2 = st.text_input("Enter Days", value="252")
    with sub_col2:
        text_input3 = st.text_input("Enter Simulations", value="1000")

    # Build ticker_list exactly like before: flat, UPPER-cased, de-duplicated,
    # benchmark appended last so `has_jkse = "^JKSE" in df.columns` keeps working.
    ticker_list = []
    for item in selected_stocks + selected_benchmarks:
        t = item.strip().upper()
        if t and t not in ticker_list:
            ticker_list.append(t)

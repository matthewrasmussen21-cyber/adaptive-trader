"""Documentation for every base feature. These are generic measurements, not named
methodologies. The research agent may combine them into constructed features, which are
documented in state/registry/constructed_features.jsonl when first created."""

XS = {  # cross-sectional (per ticker)
    "ret_5": "Total return over the last 5 trading days",
    "ret_21": "Total return over 21 days",
    "ret_63": "Total return over 63 days",
    "ret_126": "Total return over 126 days",
    "ret_252": "Total return over 252 days",
    "vol_21": "Annualized std of daily returns, 21 days",
    "vol_63": "Annualized std of daily returns, 63 days",
    "vol_ratio": "vol_21 / vol_63",
    "dvol_21": "log of 21-day average dollar volume",
    "volu_chg_5_63": "log(avg volume 5d / avg volume 63d)",
    "volu_chg_21_126": "log(avg volume 21d / avg volume 126d)",
    "dist_high_252": "Close / 252-day high - 1 (total-return series)",
    "dist_low_252": "Close / 252-day low - 1",
    "dist_high_63": "Close / 63-day high - 1",
    "dist_low_63": "Close / 63-day low - 1",
    "gap_1": "Today's open / yesterday's close - 1",
    "range_21": "21-day average of (high - low) / close",
    "rs_spy_63": "ret_63 minus SPY 63-day return",
    "rs_spy_126": "ret_126 minus SPY 126-day return",
    "rs_sector_21": "ret_21 minus the stock's sector ETF 21-day return (SPY if unmapped)",
    "rs_sector_63": "ret_63 minus sector ETF 63-day return",
    "beta_252": "252-day beta of daily returns to SPY",
    "idio_vol_63": "Annualized std of 63-day residual returns after removing beta_252 * SPY",
    "skew_63": "Skewness of daily returns over 63 days",
    "up_frac_21": "Fraction of positive-return days over 21 days",
    "max_ret_21": "Largest single-day return over 21 days",
    "rev_yoy_q": "Latest filed quarter revenue vs same quarter a year earlier",
    "rev_yoy_ttm": "Trailing-12-month revenue growth vs prior 12 months",
    "eps_yoy_q": "Latest filed quarter diluted EPS change vs a year earlier, / |prior|",
    "ni_yoy_ttm": "TTM net income growth, / |prior|",
    "gross_margin": "TTM gross profit / TTM revenue",
    "op_margin": "TTM operating income / TTM revenue",
    "net_margin": "TTM net income / TTM revenue",
    "op_margin_chg": "TTM operating margin minus the prior-year TTM operating margin",
    "roe": "TTM net income / latest equity",
    "accruals": "(TTM net income - TTM operating cash flow) / assets",
    "share_chg_yoy": "Shares outstanding change vs four filings earlier",
    "ey": "TTM net income / market cap",
    "sy": "TTM revenue / market cap",
    "by": "Book equity / market cap",
    "cfy": "TTM operating cash flow / market cap",
    "log_mcap": "log market cap (actual traded price x reported shares)",
    "days_since_filing": "Trading days since the latest SEC filing became available",
}

MKT = {  # market context (one value per day, same for every ticker)
    "vix": "VIX close (FRED VIXCLS), lagged 1 day",
    "vix_z_252": "VIX z-score vs its trailing 252 days",
    "yc_10y2y": "10y minus 2y Treasury spread (FRED T10Y2Y), lagged 1 day",
    "hy_spread": "High-yield option-adjusted spread (FRED BAMLH0A0HYM2), lagged 1 day",
    "hy_chg_63": "63-day change in hy_spread",
    "dgs10_chg_63": "63-day change in the 10y Treasury yield",
    "tbill": "3-month T-bill rate (FRED DTB3), lagged 1 day",
    "breadth_21": "Share of eligible tickers with positive 21-day return",
    "breadth_63": "Share of eligible tickers with positive 63-day return",
    "spy_ret_21": "SPY 21-day total return",
    "spy_ret_63": "SPY 63-day total return",
    "spy_ret_252": "SPY 252-day total return",
    "spy_vol_21": "SPY annualized 21-day volatility",
    "spy_dist_high_252": "SPY / its 252-day high - 1",
}

CONSTRUCT_OPS = {
    "diff": "a - b on raw values (both features must share units, e.g. two returns)",
    "ratio": "a / b on raw values where b > 0",
    "prod": "rank(a) * rank(b) — an interaction of two cross-sectional ranks",
    "neg": "-a",
}

FUNDAMENTAL = {"rev_yoy_q", "rev_yoy_ttm", "eps_yoy_q", "ni_yoy_ttm", "gross_margin", "op_margin",
               "net_margin", "op_margin_chg", "roe", "accruals", "share_chg_yoy", "ey", "sy", "by",
               "cfy", "log_mcap", "days_since_filing"}
RETURN_LIKE = {"ret_5", "ret_21", "ret_63", "ret_126", "ret_252", "rs_spy_63", "rs_spy_126",
               "rs_sector_21", "rs_sector_63", "dist_high_252", "dist_low_252", "dist_high_63",
               "dist_low_63", "gap_1", "max_ret_21"}


def parse(name: str) -> tuple[str, list[str]] | None:
    """'diff(ret_252,ret_21)' -> ('diff', ['ret_252', 'ret_21']); base names -> None."""
    if "(" not in name:
        return None
    op, rest = name.split("(", 1)
    args = [a.strip() for a in rest.rstrip(")").split(",")]
    return op, args


def valid(name: str) -> bool:
    p = parse(name)
    if p is None:
        return name in XS
    op, args = p
    if op not in CONSTRUCT_OPS or not all(a in XS for a in args):
        return False
    if op == "neg":
        return len(args) == 1
    if len(args) != 2 or args[0] == args[1]:
        return False
    if op == "diff":
        return args[0] in RETURN_LIKE and args[1] in RETURN_LIKE
    return True


def describe(name: str) -> str:
    p = parse(name)
    if p is None:
        return XS.get(name) or MKT.get(name, "")
    op, args = p
    return f"{CONSTRUCT_OPS[op]} with a={args[0]}" + (f", b={args[1]}" if len(args) > 1 else "")

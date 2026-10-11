"""
Optional market/sector context features (NIFTY 50, BANK NIFTY, India VIX, ...).

Disabled by default so the base system works without it. Enable via:

    ENABLE_MARKET_CONTEXT = True          # settings.py
    MARKET_INDEX_TICKERS = {...}          # optional override, see below

Kept as its own module — separate from features.py's core pipeline — so it
can be turned on, or extended with sector indices, without touching
indicators.py, data.py, trainer.py, or predictor.py. When enabled, every
configured index gets a `mkt_<name>_return_1` column merged onto the
feature frame via nearest-preceding timestamp match (no lookahead).
"""
import pandas as pd
import yfinance as yf
from django.conf import settings

DEFAULT_MARKET_INDEX_TICKERS = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "INDIA_VIX": "^INDIAVIX",
}


def get_market_context_frame(interval: str = "5m", period: str = "1d") -> pd.DataFrame:
    """Fetch today's intraday levels for configured indices. In-memory only."""
    tickers = getattr(settings, "MARKET_INDEX_TICKERS", DEFAULT_MARKET_INDEX_TICKERS)
    frames = {}
    for name, ticker in tickers.items():
        try:
            raw = yf.Ticker(ticker).history(period=period, interval=interval)
            if raw is None or raw.empty:
                continue
            raw = raw.reset_index()
            ts_col = "Datetime" if "Datetime" in raw.columns else "Date"
            s = raw.set_index(pd.to_datetime(raw[ts_col]))["Close"].rename(name)
            frames[name] = s
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames.values(), axis=1)


def merge_market_context(df: pd.DataFrame) -> pd.DataFrame:
    """No-op unless settings.ENABLE_MARKET_CONTEXT is truthy."""
    if not getattr(settings, "ENABLE_MARKET_CONTEXT", False):
        return df

    ctx = get_market_context_frame()
    if ctx.empty:
        return df

    tickers = getattr(settings, "MARKET_INDEX_TICKERS", DEFAULT_MARKET_INDEX_TICKERS)
    ctx = ctx.sort_index().reset_index().rename(columns={"index": "timestamp"})
    df = df.sort_values("timestamp")
    merged = pd.merge_asof(df, ctx, on="timestamp", direction="backward")

    for name in tickers:
        if name in merged.columns:
            merged[f"mkt_{name.lower()}_return_1"] = merged[name].pct_change(1)
    return merged

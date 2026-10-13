"""
Feature / target datasets for the model zoo, for two timeframes:

  intraday  5-minute candles (database, plus Yahoo for live prediction)
            targets: close return 3 / 6 / 12 candles ahead (15 / 30 / 60 minutes),
            only within the same session (no overnight gap in a target)
  daily     daily candles (database, plus Yahoo for days the database is missing)
            targets: close return 1 / 5 trading days ahead

Every feature uses only information available at the close of its own row
(rolling windows, shifts into the past, running day high / low), so a model
never sees the future.  Only `fwd_*` target columns look ahead, and they are
used as labels only.

Market context comes from the Nifty 50 ETF (NIFTYBEES.NS) in the same tables.
Nothing in this module writes to the database.
"""
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from django.core.cache import cache

from market.models import Company, DailyData, FiveMinuteData

from . import features as base_features
from . import indicators as ind
from .data import load_historical_candles
from .market_time import IST
from .tickers import get_yahoo_ticker

logger = logging.getLogger(__name__)

MARKET_SYMBOL = "NIFTYBEES.NS"
INTRADAY_HORIZONS = (3, 6, 12)          # 5-minute candles ahead
DAILY_HORIZONS = (1, 5)                 # trading days ahead
DAILY_MAX_ROWS = 2500                   # ~10 years: recent enough to be relevant, long enough to learn
TIMEFRAMES = ("intraday", "daily")


def horizon_label(timeframe, h):
    return f"{h * 5} min" if timeframe == "intraday" else (f"{h} day" if h == 1 else f"{h} days")


# ==========================================================================
# Yahoo helpers (in memory only, cached briefly)
# ==========================================================================
def _yahoo(symbol, cache_seconds=60, **kwargs):
    key = "prediction:yf:" + symbol + ":" + ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    cached = cache.get(key)
    if cached is not None:
        return cached
    import yfinance as yf
    try:
        raw = yf.Ticker(symbol).history(**kwargs)
    except Exception as exc:  # network, rate limit, delisted ...
        logger.warning("yfinance %s %s failed: %s", symbol, kwargs, exc)
        raw = None
    out = pd.DataFrame() if raw is None else raw
    cache.set(key, out, cache_seconds if not out.empty else 30)
    return out


def _yahoo_frame(raw, daily=False):
    """yfinance history -> timestamp/open/high/low/close/volume frame in IST."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    raw = raw.reset_index()
    ts_col = "Datetime" if "Datetime" in raw.columns else "Date"
    ts = pd.to_datetime(raw[ts_col])
    ts = ts.dt.tz_localize("UTC").dt.tz_convert(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)
    df = pd.DataFrame({"timestamp": ts, "open": raw["Open"].astype(float), "high": raw["High"].astype(float),
                       "low": raw["Low"].astype(float), "close": raw["Close"].astype(float),
                       "volume": raw["Volume"].fillna(0).astype(float)})
    return df.dropna(subset=["open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)


# ==========================================================================
# INTRADAY
# ==========================================================================
def load_intraday(company, live=False):
    """5-minute candles: the database, plus (live) the last 5 days from Yahoo incl. today."""
    df = load_historical_candles(company)[["timestamp", "open", "high", "low", "close", "volume"]]
    if live:
        df = df.tail(75 * 60)                  # the latest ~60 sessions in the database, however old
        recent = _yahoo_frame(_yahoo(get_yahoo_ticker(company.code), period="5d", interval="5m", prepost=False))
        if not recent.empty:
            df = pd.concat([df, recent], ignore_index=True)
            df = df.drop_duplicates("timestamp", keep="last")
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # keep NSE session candles only
    t = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    return df[(t >= 9 * 60 + 15) & (t < 15 * 60 + 30)].reset_index(drop=True)


def _market_intraday(live=False):
    company = Company.objects.filter(code=MARKET_SYMBOL).first()
    if company is None:
        return None
    m = load_intraday(company, live=live)
    if m.empty:
        return None
    day = m["timestamp"].dt.date
    out = pd.DataFrame({"timestamp": m["timestamp"]})
    out["mkt_ret_1"] = m["close"].pct_change(1)
    out["mkt_ret_6"] = m["close"].pct_change(6)
    out["mkt_day_ret"] = m["close"] / m.groupby(day)["open"].transform("first") - 1
    return out


def intraday_features(df, market=None):
    """Feature frame (index aligned with df). `df` must be sorted 5-minute candles."""
    df = base_features.build_features(df.copy(), company=None)
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    F = pd.DataFrame(index=df.index)
    for n in (1, 2, 3, 6, 12, 24):
        F[f"ret_{n}"] = c.pct_change(n)
    for col in ("sma_5", "sma_10", "sma_20", "sma_50", "ema_5", "ema_10", "ema_20"):
        F[f"c_vs_{col}"] = c / df[col] - 1
    F["sma5_vs_sma20"] = df["sma_5"] / df["sma_20"] - 1
    F["body"] = (c - o) / c
    F["range"] = (h - l) / c
    F["upper_wick"] = df["upper_wick"] / c
    F["lower_wick"] = df["lower_wick"] / c
    F["body_range_ratio"] = df["body_range_ratio"]
    for col in ("vol_5", "vol_10", "vol_20", "rsi_14", "bb_width", "minutes_from_open", "day_of_week"):
        F[col] = df[col]
    F["atr_pct"] = df["atr_14"] / c
    F["macd_pct"] = df["macd"] / c
    F["macd_hist_pct"] = df["macd_hist"] / c
    F["bb_pos"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    F["c_vs_vwap"] = df["close_over_vwap"] - 1
    F["volume_ratio"] = df["volume_ratio"].clip(0, 20)
    F["relative_volume"] = df["relative_volume"].clip(0, 20)
    F["log_volume_chg"] = np.log1p(df["volume"]).diff()

    # day context: today so far, previous day's levels, gap
    day = df["timestamp"].dt.date
    day_open = df.groupby(day)["open"].transform("first")
    run_high = df.groupby(day)["high"].cummax()
    run_low = df.groupby(day)["low"].cummin()
    F["day_ret"] = c / day_open - 1
    F["day_range_pos"] = (c - run_low) / (run_high - run_low).replace(0, np.nan)
    daily = df.groupby(day).agg(dh=("high", "max"), dl=("low", "min"), dc=("close", "last"))
    prev = daily.shift(1)
    prev2_close = daily["dc"].shift(2)
    pdh, pdl, pdc = day.map(prev["dh"]), day.map(prev["dl"]), day.map(prev["dc"])
    F["gap"] = day_open / pdc - 1
    F["c_vs_pdh"] = c / pdh - 1
    F["c_vs_pdl"] = c / pdl - 1
    F["c_vs_pdc"] = c / pdc - 1
    F["prev_day_ret"] = pdc / day.map(prev2_close) - 1

    # market (Nifty ETF) at the same candle, and the stock's strength vs it
    if market is not None:
        m = df[["timestamp"]].merge(market, on="timestamp", how="left")
        for col in ("mkt_ret_1", "mkt_ret_6", "mkt_day_ret"):
            F[col] = m[col].values
        F["rs_6"] = F["ret_6"] - F["mkt_ret_6"]
        F["rs_day"] = F["day_ret"] - F["mkt_day_ret"]
    else:
        for col in ("mkt_ret_1", "mkt_ret_6", "mkt_day_ret", "rs_6", "rs_day"):
            F[col] = np.nan
    return F.replace([np.inf, -np.inf], np.nan).astype("float32")


def intraday_targets(df, horizons=INTRADAY_HORIZONS):
    """Close return h candles ahead - only when that candle is in the same session."""
    c = df["close"]
    day = df["timestamp"].dt.date
    out = pd.DataFrame(index=df.index)
    for h in horizons:
        same_day = day.shift(-h) == day
        out[f"fwd_{h}"] = (c.shift(-h) / c - 1).where(same_day)
    return out


def intraday_dataset(company):
    """(timestamps, X, Y) for training, database only."""
    df = load_intraday(company)
    if len(df) < 2000:
        raise ValueError(f"Not enough 5-minute history for {company.code}: {len(df)} candles (need 2,000+).")
    X = intraday_features(df, _market_intraday())
    Y = intraday_targets(df)
    return df, X, Y


def intraday_live(company):
    """(candles, features) up to the latest candle, database + Yahoo (incl. today)."""
    df = load_intraday(company, live=True)
    if len(df) < 100:
        raise ValueError(f"Not enough recent 5-minute candles for {company.code}.")
    return df, intraday_features(df, _market_intraday(live=True))


# ==========================================================================
# DAILY
# ==========================================================================
def _clean_daily(df):
    """Drop one-day bad prices and history before a lasting >50 % price break (unadjusted splits)."""
    if df.empty:
        return df
    c = df["close"].values
    keep = np.ones(len(df), dtype=bool)
    for i in range(1, len(df) - 1):
        up, back = c[i] / c[i - 1] - 1, c[i + 1] / c[i] - 1
        if abs(up) > 0.4 and abs(back) > 0.25 and (up > 0) != (back > 0):
            keep[i] = False
    df = df[keep].reset_index(drop=True)
    jumps = (df["close"] / df["close"].shift(1) - 1).abs() > 0.5
    if jumps.any():
        df = df.iloc[int(np.flatnonzero(jumps.values)[-1]):].reset_index(drop=True)
    return df


def load_daily(company, live=False):
    rows = (DailyData.objects.filter(company=company).order_by("date")
            .values_list("date", "open", "high", "low", "close", "volume"))
    df = pd.DataFrame([(d, float(o), float(h), float(l), float(c), float(v or 0)) for d, o, h, l, c, v in rows
                       if None not in (o, h, l, c) and float(l) > 0],
                      columns=["date", "open", "high", "low", "close", "volume"])
    if live:
        recent = _yahoo_frame(_yahoo(get_yahoo_ticker(company.code), cache_seconds=600, period="3mo", interval="1d"))
        if not recent.empty:
            recent["date"] = recent["timestamp"].dt.date
            today = pd.Timestamp.now(tz=IST).date()
            recent = recent[recent["date"] < today]            # completed days only
            df = pd.concat([df, recent[df.columns]], ignore_index=True).drop_duplicates("date", keep="last")
        # Yahoo's daily feed often lags a session; complete days from its 5-minute candles instead
        five = _yahoo_frame(_yahoo(get_yahoo_ticker(company.code), period="5d", interval="5m", prepost=False))
        if not five.empty:
            five["date"] = five["timestamp"].dt.date
            today = pd.Timestamp.now(tz=IST).date()
            have = set(df["date"])
            agg = (five[(five["date"] < today) & ~five["date"].isin(have)].groupby("date")
                   .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                        volume=("volume", "sum")).reset_index())
            if not agg.empty:
                df = pd.concat([df, agg[df.columns]], ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    return _clean_daily(df)


def _market_daily(live=False):
    company = Company.objects.filter(code=MARKET_SYMBOL).first()
    if company is None:
        return None
    m = load_daily(company, live=live)
    if m.empty:
        return None
    out = pd.DataFrame({"date": m["date"]})
    out["mkt_ret_1"] = m["close"].pct_change(1)
    out["mkt_ret_5"] = m["close"].pct_change(5)
    out["mkt_ret_20"] = m["close"].pct_change(20)
    out["mkt_c_vs_sma50"] = m["close"] / m["close"].rolling(50).mean() - 1
    return out


def daily_features(df, market=None):
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    F = pd.DataFrame(index=df.index)
    r1 = c.pct_change(1)
    for n in (1, 2, 3, 5, 10, 20, 60):
        F[f"ret_{n}"] = c.pct_change(n)
    F["vol_10"] = r1.rolling(10).std()
    F["vol_20"] = r1.rolling(20).std()
    F["vol_ratio_10_60"] = F["vol_10"] / r1.rolling(60).std()
    F["atr_pct"] = ind.atr(h, l, c, 14) / c
    F["rsi_14"] = ind.rsi(c, 14)
    F["rsi_2"] = ind.rsi(c, 2)
    _, _, hist = ind.macd(c)
    F["macd_hist_pct"] = hist / c
    for n in (10, 20, 50, 100, 200):
        F[f"c_vs_sma{n}"] = c / c.rolling(n).mean() - 1
    F["sma20_vs_sma50"] = c.rolling(20).mean() / c.rolling(50).mean() - 1
    F["sma50_vs_sma200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1
    F["dist_hi20"] = c / h.rolling(20).max() - 1
    F["dist_lo20"] = c / l.rolling(20).min() - 1
    F["dist_hi252"] = c / h.rolling(252).max() - 1
    F["dist_lo252"] = c / l.rolling(252).min() - 1
    up, bb_mid, lo, width = ind.bollinger_bands(c)
    F["bb_pos"] = (c - lo) / (up - lo).replace(0, np.nan)
    F["bb_width"] = width
    F["vol_ratio_20"] = (v / v.rolling(20).mean()).clip(0, 20)
    F["vol_ratio_5_20"] = (v.rolling(5).mean() / v.rolling(20).mean()).clip(0, 20)
    rng = (h - l).replace(0, np.nan)
    F["body"] = (c - o) / c
    F["range"] = (h - l) / c
    F["close_pos"] = (c - l) / rng
    F["upper_wick"] = (h - np.maximum(o, c)) / c
    F["lower_wick"] = (np.minimum(o, c) - l) / c
    F["gap"] = o / c.shift(1) - 1
    dates = pd.to_datetime(df["date"])
    F["day_of_week"] = dates.dt.dayofweek
    F["month"] = dates.dt.month
    if market is not None:
        m = df[["date"]].merge(market, on="date", how="left")
        for col in ("mkt_ret_1", "mkt_ret_5", "mkt_ret_20", "mkt_c_vs_sma50"):
            F[col] = m[col].values
        F["rs_5"] = F["ret_5"] - F["mkt_ret_5"]
        F["rs_20"] = F["ret_20"] - F["mkt_ret_20"]
    else:
        for col in ("mkt_ret_1", "mkt_ret_5", "mkt_ret_20", "mkt_c_vs_sma50", "rs_5", "rs_20"):
            F[col] = np.nan
    return F.replace([np.inf, -np.inf], np.nan).astype("float32")


def daily_targets(df, horizons=DAILY_HORIZONS):
    out = pd.DataFrame(index=df.index)
    for h in horizons:
        out[f"fwd_{h}"] = df["close"].shift(-h) / df["close"] - 1
    return out


def daily_dataset(company):
    df = load_daily(company)
    if len(df) < 600:
        raise ValueError(f"Not enough daily history for {company.code}: {len(df)} days (need 600+).")
    X = daily_features(df, _market_daily())
    Y = daily_targets(df)
    df, X, Y = df.iloc[-DAILY_MAX_ROWS:], X.iloc[-DAILY_MAX_ROWS:], Y.iloc[-DAILY_MAX_ROWS:]
    return df.reset_index(drop=True), X.reset_index(drop=True), Y.reset_index(drop=True)


def daily_live(company):
    df = load_daily(company, live=True)
    if len(df) < 260:
        raise ValueError(f"Not enough daily history for {company.code}.")
    return df, daily_features(df, _market_daily(live=True))


def dataset(company, timeframe):
    return intraday_dataset(company) if timeframe == "intraday" else daily_dataset(company)


def live(company, timeframe):
    return intraday_live(company) if timeframe == "intraday" else daily_live(company)


def horizons(timeframe):
    return INTRADAY_HORIZONS if timeframe == "intraday" else DAILY_HORIZONS


def next_trading_days(after, n):
    """The n-th weekday after `after` (exchange holidays are not known here)."""
    d = after
    count = 0
    while count < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d

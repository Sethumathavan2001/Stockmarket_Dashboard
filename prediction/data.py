"""
Data loading pipeline.

Two independent sources, kept strictly separate:

  Historical (training):   FiveMinuteData (DB)  -> pandas DataFrame
  Live (prediction only):  yfinance 5m intraday  -> pandas DataFrame, IN MEMORY ONLY

`combine_and_clean` merges the two purely in memory for feature generation.
Nothing in this module ever calls FiveMinuteData.objects.create(...) or
.update(...) — today's yfinance candles are never written to the database.
"""
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from market.models import Company, FiveMinuteData  # adjust if your app is named differently
from .tickers import get_yahoo_ticker
from .market_time import IST

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def load_historical_candles(company: Company, lookback_days: int = None) -> pd.DataFrame:
    """Historical candles from the database. Used for TRAINING."""
    qs = FiveMinuteData.objects.filter(company=company).order_by("timestamp")
    if lookback_days:
        cutoff = datetime.now(IST) - timedelta(days=lookback_days)
        qs = qs.filter(timestamp__gte=cutoff)

    rows = list(qs.values("timestamp", "open", "high", "low", "close", "volume"))
    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source"])

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].fillna(0).astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = (
        df["timestamp"].dt.tz_localize(IST)
        if df["timestamp"].dt.tz is None
        else df["timestamp"].dt.tz_convert(IST)
    )
    df["source"] = "database"
    return df.sort_values("timestamp").reset_index(drop=True)


def load_today_intraday_yfinance(company: Company) -> pd.DataFrame:
    """
    Today's 5-minute candles from yfinance. NEVER persisted anywhere —
    returned as an in-memory DataFrame only.
    """
    ticker_symbol = get_yahoo_ticker(company.code)
    try:
        raw = yf.Ticker(ticker_symbol).history(period="1d", interval="5m", prepost=False)
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s (%s): %s", company.code, ticker_symbol, exc)
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source"])

    if raw is None or raw.empty:
        logger.info("yfinance returned no intraday data for %s (%s)", company.code, ticker_symbol)
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source"])

    raw = raw.reset_index()
    ts_col = "Datetime" if "Datetime" in raw.columns else "Date"
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[ts_col]),
            "open": raw["Open"].astype(float),
            "high": raw["High"].astype(float),
            "low": raw["Low"].astype(float),
            "close": raw["Close"].astype(float),
            "volume": raw["Volume"].fillna(0).astype(float),
        }
    )
    df["timestamp"] = (
        df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(IST)
        if df["timestamp"].dt.tz is None
        else df["timestamp"].dt.tz_convert(IST)
    )
    df["source"] = "yfinance"
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df.sort_values("timestamp").reset_index(drop=True)


def combine_and_clean(hist_df: pd.DataFrame, today_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge historical + live frames, sort by timestamp, drop duplicate
    timestamps (keeping the later/live value if both sources overlap).
    Purely in memory — nothing here touches the database.
    """
    combined = pd.concat([hist_df, today_df], ignore_index=True)
    if combined.empty:
        return combined
    combined = combined.sort_values("timestamp")
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    combined = combined.replace([np.inf, -np.inf], np.nan)
    return combined.reset_index(drop=True)


def get_training_dataframe(company: Company, lookback_days: int = None) -> pd.DataFrame:
    """Historical-only dataframe. Used for TRAINING — never includes yfinance data."""
    return load_historical_candles(company, lookback_days=lookback_days)


def get_live_dataframe(company: Company, lookback_days: int = 60) -> pd.DataFrame:
    """
    Historical (DB) + today's yfinance candles, combined in memory.
    Used for PREDICTION only. Today's rows are never saved anywhere.
    """
    hist_df = load_historical_candles(company, lookback_days=lookback_days)
    today_df = load_today_intraday_yfinance(company)
    return combine_and_clean(hist_df, today_df)

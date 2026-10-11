"""
Reusable feature-engineering module.

`build_features` is called identically by the training pipeline (on
historical DB data) and the live prediction pipeline (on historical + today's
in-memory yfinance data) — this is what guarantees train/predict parity.

Every feature only uses information available AT or BEFORE its own row's
timestamp (rolling windows, ewm, pct_change, shift(positive), cumulative
groupby). `build_targets` is the one function that looks into the future
(shift(-h)) and its output must only ever be used as a training label, never
fed back in as a feature.
"""
import numpy as np
import pandas as pd
from datetime import time

from . import indicators as ind
from . import market_context

MARKET_OPEN_TIME = time(9, 15)

BASE_FEATURE_COLUMNS = [
    "return_1", "return_2", "return_3", "return_6", "return_12",
    "momentum_10",
    "candle_body", "candle_range", "upper_wick", "lower_wick", "body_range_ratio",
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_5", "ema_10", "ema_20",
    "close_over_sma_20", "close_over_ema_20",
    "vol_5", "vol_10", "vol_20", "atr_14",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_width",
    "vwap", "close_over_vwap",
    "volume_sma_20", "volume_ratio", "volume_change", "relative_volume",
    "hour", "minute", "day_of_week", "minutes_from_open",
]

COMPANY_FEATURE_COLUMNS = [
    "dist_from_yearly_high", "dist_from_yearly_low",
    "dist_from_monthly_high", "dist_from_monthly_low",
    "dist_from_weekly_high", "dist_from_weekly_low",
]

ALL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + COMPANY_FEATURE_COLUMNS


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def build_price_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df["return_1"] = df["close"].pct_change(1)
    df["return_2"] = df["close"].pct_change(2)
    df["return_3"] = df["close"].pct_change(3)
    df["return_6"] = df["close"].pct_change(6)
    df["return_12"] = df["close"].pct_change(12)
    df["momentum_10"] = df["close"] - df["close"].shift(10)
    df["candle_body"] = df["close"] - df["open"]
    df["candle_range"] = df["high"] - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["body_range_ratio"] = _safe_div(df["candle_body"], df["candle_range"]).fillna(0)
    return df


def build_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["close_over_sma_20"] = _safe_div(df["close"], df["sma_20"])
    df["close_over_ema_20"] = _safe_div(df["close"], df["ema_20"])
    return df


def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df["vol_5"] = df["return_1"].rolling(5).std()
    df["vol_10"] = df["return_1"].rolling(10).std()
    df["vol_20"] = df["return_1"].rolling(20).std()
    df["atr_14"] = ind.atr(df["high"], df["low"], df["close"], period=14)
    return df


def build_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi_14"] = ind.rsi(df["close"], period=14)
    macd_line, signal_line, hist = ind.macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    upper, mid, lower, width = ind.bollinger_bands(df["close"])
    df["bb_upper"] = upper
    df["bb_mid"] = mid
    df["bb_lower"] = lower
    df["bb_width"] = width
    df["vwap"] = ind.vwap(df)
    df["close_over_vwap"] = _safe_div(df["close"], df["vwap"])
    return df


def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = _safe_div(df["volume"], df["volume_sma_20"]).fillna(1)
    df["volume_change"] = df["volume"].pct_change(1)
    same_time_of_day = df["timestamp"].dt.time
    df["relative_volume"] = (
        df.groupby(same_time_of_day)["volume"]
        .transform(lambda s: _safe_div(s, s.rolling(10, min_periods=1).mean()))
        .fillna(1)
    )
    return df


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    minutes_of_day = ts.dt.hour * 60 + ts.dt.minute
    open_minutes = MARKET_OPEN_TIME.hour * 60 + MARKET_OPEN_TIME.minute
    df["minutes_from_open"] = (minutes_of_day - open_minutes).clip(lower=0)
    return df


def build_company_features(df: pd.DataFrame, company) -> pd.DataFrame:
    close = df["close"]

    def dist(level):
        if not level:
            return pd.Series(np.nan, index=df.index)
        return (close - level) / level

    df["dist_from_yearly_high"] = dist(company.high_yearly)
    df["dist_from_yearly_low"] = dist(company.low_yearly)
    df["dist_from_monthly_high"] = dist(company.high_monthly)
    df["dist_from_monthly_low"] = dist(company.low_monthly)
    df["dist_from_weekly_high"] = dist(company.high_weekly)
    df["dist_from_weekly_low"] = dist(company.low_weekly)
    return df


def build_features(df: pd.DataFrame, company=None) -> pd.DataFrame:
    """
    Build the full feature set on a chronologically sorted OHLCV dataframe
    (columns: timestamp, open, high, low, close, volume).

    Safe to call on the combined historical(DB) + live(yfinance) frame —
    nothing here writes to the database.
    """
    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = build_price_return_features(df)
    df = build_moving_averages(df)
    df = build_volatility_features(df)
    df = build_technical_indicators(df)
    df = build_volume_features(df)
    df = build_time_features(df)

    if company is not None:
        df = build_company_features(df, company)
    else:
        for col in COMPANY_FEATURE_COLUMNS:
            df[col] = np.nan

    # Optional market/sector context (NIFTY, BANKNIFTY, India VIX...) — no-op
    # unless settings.ENABLE_MARKET_CONTEXT is True. See market_context.py.
    df = market_context.merge_market_context(df)

    return df


def build_targets(df: pd.DataFrame, horizons=(1, 2, 3, 6)) -> pd.DataFrame:
    """
    Future OHLC return targets, relative to the CURRENT row's close, using
    shift(-h). This is strictly future information — use only as training
    labels, never as a model input feature.
    """
    for h in horizons:
        df[f"target_open_return_{h}"] = df["open"].shift(-h) / df["close"] - 1
        df[f"target_high_return_{h}"] = df["high"].shift(-h) / df["close"] - 1
        df[f"target_low_return_{h}"] = df["low"].shift(-h) / df["close"] - 1
        df[f"target_close_return_{h}"] = df["close"].shift(-h) / df["close"] - 1
    return df


def get_feature_matrix(df: pd.DataFrame, feature_columns=None) -> pd.DataFrame:
    feature_columns = feature_columns or ALL_FEATURE_COLUMNS
    return df[feature_columns].replace([np.inf, -np.inf], np.nan)

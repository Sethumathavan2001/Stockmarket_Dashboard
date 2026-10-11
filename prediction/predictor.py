"""
Live prediction service.

Company -> load historical FiveMinuteData -> load today's yfinance 5m data
(NEVER SAVED) -> combine in memory -> build features -> load trained model
-> predict future returns -> reconstruct prices -> return a plain dict
(JSON-serializable) for the API layer.
"""
import logging
from datetime import datetime

import numpy as np

from . import features as feat
from .data import get_live_dataframe
from .trainer import load_model_bundle, HORIZON_MINUTES
from .market_time import get_market_status, is_stale, IST

logger = logging.getLogger(__name__)

MIN_LIVE_ROWS = 60  # enough history for the longest rolling window (SMA 50) to be valid


class PredictionError(Exception):
    """Raised for expected, user-facing prediction failures (no model, no data, etc.)."""


def _confidence_from_metrics(horizon_metrics: dict) -> float:
    """
    Heuristic confidence score in [0, 1]:

        confidence = clip(2 * directional_accuracy - 1, 0, 1)

    directional_accuracy comes from this horizon's own walk-forward
    validation. 50% (a coin flip) maps to 0 confidence; 100% maps to 1.

    This is NOT a calibrated probability that any individual prediction is
    correct — it only reflects how reliable this horizon's UP/DOWN call has
    historically been on held-out, chronologically-future data. Treat it as
    a relative reliability signal, not a guarantee.
    """
    dir_acc = horizon_metrics.get("directional_accuracy", 0.5)
    return float(np.clip(2 * dir_acc - 1, 0.0, 1.0))


def predict_company(company, n_history_days: int = 60) -> dict:
    bundle = load_model_bundle(company.code)
    if bundle is None:
        raise PredictionError(
            f"No trained model found for {company.code}. Train it first via the "
            f"Train button, or `python manage.py train_models --company {company.code}`."
        )

    live_df = get_live_dataframe(company, lookback_days=n_history_days)
    if len(live_df) < MIN_LIVE_ROWS:
        raise PredictionError(
            f"Not enough recent candles for {company.code} to build features "
            f"({len(live_df)} rows, need at least {MIN_LIVE_ROWS})."
        )

    feature_df = feat.build_features(live_df, company=company)
    feature_columns = bundle["feature_columns"]

    missing = [c for c in feature_columns if c not in feature_df.columns]
    if missing:
        raise PredictionError(f"Missing engineered features for {company.code}: {missing}")

    latest_row = feature_df.iloc[[-1]]
    X_latest = latest_row[feature_columns].replace([np.inf, -np.inf], np.nan)
    if X_latest.isna().any(axis=None):
        X_latest = X_latest.fillna(0)

    latest_timestamp = latest_row["timestamp"].iloc[0]
    current_price = float(latest_row["close"].iloc[0])
    data_source = latest_row["source"].iloc[0]

    market_status = get_market_status()
    stale = is_stale(latest_timestamp.to_pydatetime())

    predictions = []
    for h, minutes in HORIZON_MINUTES.items():
        model = bundle["models"][h]
        metrics = bundle["metrics"][h]
        pred = model.predict(X_latest)[0]  # order: [open_ret, high_ret, low_ret, close_ret]
        open_ret, high_ret, low_ret, close_ret = pred

        predictions.append(
            {
                "horizon_minutes": minutes,
                "predicted_return_percent": round(float(close_ret) * 100, 4),
                "predicted_open": round(current_price * (1 + open_ret), 2),
                "predicted_high": round(current_price * (1 + high_ret), 2),
                "predicted_low": round(current_price * (1 + low_ret), 2),
                "predicted_close": round(current_price * (1 + close_ret), 2),
                "direction": "UP" if close_ret > 0 else ("DOWN" if close_ret < 0 else "FLAT"),
                "confidence": round(_confidence_from_metrics(metrics), 3),
                "validation_mae": round(metrics["mae"], 5),
                "validation_directional_accuracy": round(metrics["directional_accuracy"], 4),
            }
        )

    return {
        "company": company.code,
        "timestamp": datetime.now(IST).isoformat(),
        "latest_candle_timestamp": latest_timestamp.isoformat(),
        "current_price": round(current_price, 2),
        "data_source": data_source,
        "market_status": market_status,
        "data_stale": bool(stale),
        "historical_candles_used": int(len(feature_df)),
        "model_version": bundle["model_version"],
        "model_trained_at": bundle["trained_at"],
        "predictions": predictions,
        "disclaimer": (
            "Research/forecasting output only. Not investment advice. "
            "Confidence scores are heuristic, not calibrated probabilities."
        ),
    }

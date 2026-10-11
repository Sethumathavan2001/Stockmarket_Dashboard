"""
Training pipeline.

For each company, trains one MultiOutputRegressor per horizon (predicting
open/high/low/close FUTURE RETURNS simultaneously — not absolute prices),
validates it with expanding-window walk-forward validation (never a random
shuffle), then refits on the full chronological history and saves everything
(models + feature list + metadata + metrics) to a single pickle per company.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:  # pragma: no cover - fallback path
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_XGB = False

from . import features as feat
from .data import get_training_dataframe

logger = logging.getLogger(__name__)

HORIZONS = (1, 2, 3, 6)  # in 5-minute candles
HORIZON_MINUTES = {1: 5, 2: 10, 3: 15, 6: 30}
TARGET_KINDS = ("open", "high", "low", "close")
MODEL_VERSION = "1.0.0"
MIN_ROWS_REQUIRED = 300  # roughly a few weeks of 5-minute candles


def _make_base_estimator():
    if _HAS_XGB:
        return XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )
    logger.warning("xgboost not installed — falling back to sklearn GradientBoostingRegressor")
    return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)


def _model_path(company_code: str) -> Path:
    model_dir = Path(getattr(settings, "ML_MODELS_DIR"))
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{company_code}.pkl"


def prepare_training_frame(company) -> pd.DataFrame:
    """Historical-only feature+target frame, fully NaN-cleaned, chronological."""
    df = get_training_dataframe(company)
    if len(df) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"Not enough historical data for {company.code}: {len(df)} rows "
            f"(need at least {MIN_ROWS_REQUIRED})."
        )

    df = feat.build_features(df, company=company)
    df = feat.build_targets(df, horizons=HORIZONS)

    target_cols = [f"target_{kind}_return_{h}" for h in HORIZONS for kind in TARGET_KINDS]
    needed_cols = feat.ALL_FEATURE_COLUMNS + target_cols
    df = df.dropna(subset=needed_cols).reset_index(drop=True)

    if len(df) < MIN_ROWS_REQUIRED // 2:
        raise ValueError(
            f"Not enough usable rows for {company.code} after feature/target "
            f"generation: {len(df)} rows."
        )
    return df


def _directional_accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def _simulate_strategy_return(y_true, y_pred, cost_bps: float = 2.0) -> float:
    """Naive long/flat simulation: go long whenever the predicted return > 0."""
    position = (y_pred > 0).astype(float)
    gross = position * y_true
    turnover = np.abs(np.diff(position, prepend=0))
    cost = turnover * (cost_bps / 10000.0)
    return float(np.sum(gross - cost))


def walk_forward_validate(X: pd.DataFrame, Y: pd.DataFrame, n_splits: int = 5):
    """
    Expanding-window walk-forward validation via TimeSeriesSplit: each fold
    trains only on data strictly before its test window and evaluates only
    on that future window. No shuffling — chronological order is preserved
    throughout, so the model never sees data from its own future.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    close_col = [c for c in Y.columns if c.startswith("target_close_return")][0]

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

        model = MultiOutputRegressor(_make_base_estimator())
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        preds_df = pd.DataFrame(preds, columns=Y.columns, index=Y_test.index)

        y_true = Y_test[close_col].values
        y_pred = preds_df[close_col].values

        fold_metrics.append(
            {
                "fold": fold_idx,
                "test_size": int(len(test_idx)),
                "mae": float(np.mean(np.abs(y_true - y_pred))),
                "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                "directional_accuracy": _directional_accuracy(y_true, y_pred),
                "bias": float(np.mean(y_pred - y_true)),
                "simulated_strategy_return": _simulate_strategy_return(y_true, y_pred),
            }
        )
    return fold_metrics


def train_company_model(company, n_splits: int = 5) -> dict:
    """
    Trains + validates + saves the model bundle for one company. Returns the
    in-memory bundle dict (same content that was written to disk).
    """
    df = prepare_training_frame(company)
    X_all = feat.get_feature_matrix(df)

    horizon_models = {}
    horizon_metrics = {}

    for h in HORIZONS:
        target_cols = [f"target_{kind}_return_{h}" for kind in TARGET_KINDS]
        Y_h = df[target_cols]

        fold_metrics = walk_forward_validate(X_all, Y_h, n_splits=n_splits)
        agg = {
            "mae": float(np.mean([m["mae"] for m in fold_metrics])),
            "rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
            "directional_accuracy": float(np.mean([m["directional_accuracy"] for m in fold_metrics])),
            "bias": float(np.mean([m["bias"] for m in fold_metrics])),
            "simulated_strategy_return": float(np.sum([m["simulated_strategy_return"] for m in fold_metrics])),
            "folds": fold_metrics,
        }
        horizon_metrics[h] = agg

        # Final fit on ALL available history. This is still chronologically
        # sound — there's no "future" beyond the data we have to leak from.
        final_model = MultiOutputRegressor(_make_base_estimator())
        final_model.fit(X_all, Y_h)
        horizon_models[h] = final_model

        logger.info(
            "%s horizon=%dmin MAE=%.5f RMSE=%.5f DirAcc=%.2f%%",
            company.code, HORIZON_MINUTES[h], agg["mae"], agg["rmse"], agg["directional_accuracy"] * 100,
        )

    bundle = {
        "company_code": company.code,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feat.ALL_FEATURE_COLUMNS,
        "target_kinds": TARGET_KINDS,
        "horizons": HORIZONS,
        "horizon_minutes": HORIZON_MINUTES,
        "models": horizon_models,
        "metrics": horizon_metrics,
        "training_rows": int(len(df)),
        "training_data_start": df["timestamp"].min().isoformat(),
        "training_data_end": df["timestamp"].max().isoformat(),
    }

    path = _model_path(company.code)
    joblib.dump(bundle, path)
    logger.info("Saved model bundle for %s -> %s", company.code, path)
    return bundle


def load_model_bundle(company_code: str):
    path = _model_path(company_code)
    if not path.exists():
        return None
    return joblib.load(path)

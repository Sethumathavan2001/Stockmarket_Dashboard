"""
Standalone backtesting utility — runs walk-forward validation for a company
without training/saving a production model bundle. Useful for quickly
checking whether a company has enough/clean-enough data before committing to
a full training run, or for reporting on model quality independently.
"""
from . import features as feat
from .trainer import (
    HORIZONS,
    HORIZON_MINUTES,
    TARGET_KINDS,
    prepare_training_frame,
    walk_forward_validate,
)


def backtest_company(company, n_splits: int = 5) -> dict:
    df = prepare_training_frame(company)
    X = feat.get_feature_matrix(df)

    report = {
        "company": company.code,
        "rows": int(len(df)),
        "data_start": df["timestamp"].min().isoformat(),
        "data_end": df["timestamp"].max().isoformat(),
        "horizons": {},
    }

    for h in HORIZONS:
        target_cols = [f"target_{kind}_return_{h}" for kind in TARGET_KINDS]
        Y_h = df[target_cols]
        folds = walk_forward_validate(X, Y_h, n_splits=n_splits)
        report["horizons"][HORIZON_MINUTES[h]] = folds

    return report

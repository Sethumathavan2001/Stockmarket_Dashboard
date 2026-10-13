"""
Model zoo: train many ML models per stock, test them honestly, keep the best.

For every stock, timeframe (intraday / daily) and horizon:

  1. Build the feature / target dataset (prediction.datasets).
  2. Walk-forward test: the last 40 % of the history is split into 4 test windows;
     each model is trained only on data BEFORE a window (with a gap of `horizon`
     rows, so no training label overlaps the test period) and scored on it.
  3. Score every model on the out-of-sample predictions: direction accuracy (vs. the
     naive "always the more common direction"), correlation with the real move (IC),
     error vs. predicting zero, and a cost-aware long/short test.
  4. Two ensembles are scored too: the average of all models, and the average of the
     3 models that were best on EARLIER windows (chosen without seeing the window).
  5. The best candidate (highest IC) is refit on all data and saved with its
     out-of-sample error range and a probability calibration.

Storage: ml_models/<CODE>/manifest.json (every model's scores) and
ml_models/<CODE>/<timeframe>_h<h>.joblib (the fitted best model(s)).
"""
import json
import logging
import math
import os
import time as _time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.conf import settings

from . import datasets as ds

logger = logging.getLogger(__name__)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))   # stops joblib probing the CPU via wmic

ENGINE_VERSION = "2.0"
N_FOLDS = 4
TEST_SHARE = 0.4
COST = {"intraday": 0.0008, "daily": 0.0025}      # round-trip cost as a return (0.08 % / 0.25 %)
EDGE_MIN_IC, EDGE_MIN_TRADES = 0.03, 30


# ==========================================================================
# the zoo
# ==========================================================================
def _models():
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    imp = lambda: SimpleImputer(strategy="median")
    return {
        "ridge": ("Ridge regression", lambda: make_pipeline(imp(), StandardScaler(), Ridge(alpha=10.0))),
        "knn": ("k-Nearest Neighbours", lambda: make_pipeline(
            imp(), StandardScaler(), KNeighborsRegressor(n_neighbors=75, weights="distance"))),
        "rf": ("Random Forest", lambda: make_pipeline(imp(), RandomForestRegressor(
            n_estimators=150, max_depth=8, min_samples_leaf=50, max_features=0.5, n_jobs=-1, random_state=1))),
        "et": ("Extra Trees", lambda: make_pipeline(imp(), ExtraTreesRegressor(
            n_estimators=150, max_depth=8, min_samples_leaf=50, max_features=0.5, n_jobs=-1, random_state=2))),
        "hgb": ("Histogram Gradient Boosting", lambda: HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=60, l2_regularization=1.0,
            random_state=3)),
        "xgb": ("XGBoost", lambda: XGBRegressor(
            n_estimators=350, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
            min_child_weight=30, reg_lambda=2.0, tree_method="hist", n_jobs=-1, random_state=4, verbosity=0)),
        "mlp": ("Neural network (MLP)", lambda: make_pipeline(imp(), StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=120, early_stopping=True,
            n_iter_no_change=8, random_state=5))),
    }


ENSEMBLES = {"ens_all": "Ensemble - average of all models",
             "ens_top3": "Ensemble - top 3 (chosen on earlier windows)"}
SCALE = 10_000.0          # models learn returns in basis points (better conditioned than raw returns)


def model_names():
    return {k: v[0] for k, v in _models().items()} | ENSEMBLES


# ==========================================================================
# storage
# ==========================================================================
def models_dir():
    return Path(getattr(settings, "ML_MODELS_DIR"))


def stock_dir(code):
    return models_dir() / code


def load_manifest(code):
    path = stock_dir(code) / "manifest.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def list_manifests():
    out = []
    if not models_dir().exists():
        return out
    for p in sorted(models_dir().glob("*/manifest.json")):
        try:
            with open(p, encoding="utf-8") as fh:
                out.append(json.load(fh))
        except (OSError, ValueError):
            continue
    return out


# ==========================================================================
# metrics
# ==========================================================================
def _metrics(y, pred, fold_ids, cost):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    n = len(y)
    nz = y != 0
    p_up = float(np.mean(y[nz] > 0)) if nz.any() else 0.5
    dir_acc = float(np.mean(np.sign(pred[nz]) == np.sign(y[nz]))) if nz.any() else None
    ic = float(np.corrcoef(pred, y)[0, 1]) if n > 2 and np.std(pred) > 0 and np.std(y) > 0 else 0.0
    rank_ic = float(pd.Series(pred).rank().corr(pd.Series(y).rank())) if n > 2 else 0.0
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    rmse0 = float(np.sqrt(np.mean(y ** 2)))
    trade = np.abs(pred) >= cost
    pnl = np.sign(pred[trade]) * y[trade] - cost
    # the 10 % most confident predictions (largest |predicted move|): is the model useful when it is sure?
    top = np.abs(pred) >= np.quantile(np.abs(pred), 0.9) if n >= 20 else np.zeros(n, bool)
    top_pnl = np.sign(pred[top]) * y[top] - cost
    fold_ic = []
    for f in sorted(set(fold_ids)):
        m = fold_ids == f
        if m.sum() > 2 and np.std(pred[m]) > 0 and np.std(y[m]) > 0:
            fold_ic.append(float(np.corrcoef(pred[m], y[m])[0, 1]))
        else:
            fold_ic.append(0.0)
    return {
        "samples": int(n),
        "dir_acc": dir_acc,
        "baseline_acc": max(p_up, 1 - p_up),
        "ic": 0.0 if math.isnan(ic) else ic,
        "rank_ic": 0.0 if math.isnan(rank_ic) else rank_ic,
        "skill": 1 - rmse / rmse0 if rmse0 else 0.0,       # > 0: beats predicting "no move"
        "trades": int(trade.sum()),
        "win_rate": float(np.mean(pnl > 0)) if trade.any() else None,
        "avg_net_pct": float(np.mean(pnl) * 100) if trade.any() else None,
        "top10_dir_acc": float(np.mean(np.sign(pred[top]) == np.sign(y[top]))) if top.any() else None,
        "top10_avg_net_pct": float(np.mean(top_pnl) * 100) if top.any() else None,
        "fold_ic": fold_ic,
        "folds_positive": int(sum(1 for v in fold_ic if v > 0)),
    }


def _has_edge(m):
    return bool(m["ic"] >= EDGE_MIN_IC and m["folds_positive"] >= N_FOLDS - 1
                and m["dir_acc"] is not None and m["dir_acc"] > m["baseline_acc"]
                and m["trades"] >= EDGE_MIN_TRADES and (m["avg_net_pct"] or 0) > 0)


def _calibrate(pred, y):
    """
    Logistic map from a predicted return to P(the move is up), fitted on out-of-sample predictions.
    No intercept: P(up) > 50 % exactly when the model predicts a rise; how far above 50 % reflects
    how reliable such calls were in testing (a test period's general drift is not baked in).
    """
    from sklearn.linear_model import LogisticRegression
    up = (np.asarray(y) > 0).astype(int)
    if up.min() == up.max():
        return None
    lr = LogisticRegression(C=1.0, fit_intercept=False)
    lr.fit((np.asarray(pred) * SCALE).reshape(-1, 1), up)
    return [float(lr.coef_[0][0]), 0.0]


# ==========================================================================
# training
# ==========================================================================
def _fit(factory, X, y):
    model = factory()
    lo, hi = np.nanpercentile(y, [1, 99])                     # tame extreme moves (training labels only)
    model.fit(X, np.clip(y, lo, hi) * SCALE)
    return model


def _predict(model, X):
    return np.asarray(model.predict(X), float) / SCALE


def train_horizon(X, y, timeframe, h, progress=None, only=None):
    """Walk-forward test of the whole zoo for one horizon -> (leaderboard, best_key, best_members, oos, X, y)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # convergence / empty-slice chatter from individual models
        return _train_horizon(X, y, timeframe, h, progress, only)


def _train_horizon(X, y, timeframe, h, progress, only):
    zoo = _models()
    if only:
        zoo = {k: v for k, v in zoo.items() if k in only}
    ok = y.notna().values
    X, y = X[ok].reset_index(drop=True), y[ok].reset_index(drop=True).values
    n = len(y)
    test_start = int(n * (1 - TEST_SHARE))
    edges = np.linspace(test_start, n, N_FOLDS + 1).astype(int)
    gap = h + 1

    preds = {k: np.full(n, np.nan) for k in zoo}
    timing = {k: 0.0 for k in zoo}
    for f in range(N_FOLDS):
        a, b = edges[f], edges[f + 1]
        train_end = max(0, a - gap)
        for k, (label, factory) in zoo.items():
            if progress:
                progress(f"{label} - window {f + 1}/{N_FOLDS}")
            t0 = _time.time()
            model = _fit(factory, X.iloc[:train_end], y[:train_end])
            preds[k][a:b] = _predict(model, X.iloc[a:b])
            timing[k] += _time.time() - t0

    test = np.arange(test_start, n)
    fold_ids = np.searchsorted(edges[1:], test, side="right")
    yt = y[test]
    board = {}
    for k in zoo:
        board[k] = _metrics(yt, preds[k][test], fold_ids, COST[timeframe])
        board[k]["seconds"] = round(timing[k], 1)

    # ensembles
    stack = np.vstack([preds[k] for k in zoo])
    ens_all = np.nanmean(stack, axis=0)
    ens_top = np.full(n, np.nan)
    keys = list(zoo)
    for f in range(N_FOLDS):
        a, b = edges[f], edges[f + 1]
        if f == 0:
            chosen = keys
        else:
            seen = np.arange(edges[0], a)                       # earlier windows only
            sc = {k: (np.corrcoef(preds[k][seen], y[seen])[0, 1] if np.std(preds[k][seen]) > 0 else -1) for k in keys}
            chosen = sorted(keys, key=lambda k: -np.nan_to_num(sc[k], nan=-1))[:3]
        ens_top[a:b] = np.mean([preds[k][a:b] for k in chosen], axis=0)
    board["ens_all"] = _metrics(yt, ens_all[test], fold_ids, COST[timeframe])
    board["ens_top3"] = _metrics(yt, ens_top[test], fold_ids, COST[timeframe])
    preds["ens_all"], preds["ens_top3"] = ens_all, ens_top

    for k, m in board.items():
        m["edge"] = _has_edge(m)
    best = max(board, key=lambda k: (board[k]["ic"], board[k]["dir_acc"] or 0))
    if best == "ens_all":
        members = keys
    elif best == "ens_top3":
        members = sorted(keys, key=lambda k: -board[k]["ic"])[:3]
    else:
        members = [best]
    oos = {"pred": preds[best][test], "y": yt}
    return board, best, members, oos, X, y


def train_stock(company, timeframes=ds.TIMEFRAMES, progress=None, only=None):
    """Train + test the zoo for every horizon of the given timeframes, save, and return the manifest."""
    code = company.code
    out_dir = stock_dir(code)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(code) or {"company": code, "name": company.name, "timeframes": {}}
    manifest.update(engine_version=ENGINE_VERSION, name=company.name, sector=company.sector,
                    industry=company.industry)
    names = model_names()
    steps = sum(len(ds.horizons(tf)) for tf in timeframes)
    step = 0

    for tf in timeframes:
        try:
            frame, X, Y = ds.dataset(company, tf)
        except ValueError as exc:
            manifest["timeframes"][tf] = {"error": str(exc), "trained_at": datetime.now(timezone.utc).isoformat()}
            step += len(ds.horizons(tf))
            continue
        tcol = "timestamp" if tf == "intraday" else "date"
        tf_info = {"trained_at": datetime.now(timezone.utc).isoformat(), "rows": int(len(X)),
                   "features": list(X.columns), "data_from": str(frame[tcol].iloc[0])[:19],
                   "data_to": str(frame[tcol].iloc[-1])[:19], "horizons": {}}
        for h in ds.horizons(tf):
            label = ds.horizon_label(tf, h)

            def prog(msg, _s=step):
                if progress:
                    progress(f"{tf.title()} {label}: {msg}", _s / steps)

            board, best, members, oos, Xh, yh = train_horizon(X, Y[f"fwd_{h}"], tf, h, prog, only)
            prog("fitting the best model on all data")
            zoo = _models()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = [_fit(zoo[k][1], Xh, yh) for k in members]
            resid = oos["y"] - oos["pred"]
            bundle = {"members": fitted, "member_keys": members, "features": list(X.columns),
                      "calibration": _calibrate(oos["pred"], oos["y"]),
                      "resid_q": [float(q) for q in np.nanpercentile(resid, [10, 25, 75, 90])],
                      "best": best, "timeframe": tf, "horizon": h}
            joblib.dump(bundle, out_dir / f"{tf}_h{h}.joblib", compress=3)
            tf_info["horizons"][str(h)] = {
                "label": label, "best": best, "best_name": names[best], "members": members,
                "edge": board[best]["edge"], "train_rows": int(len(yh)),
                "leaderboard": [dict(key=k, name=names[k], **board[k]) for k in
                                sorted(board, key=lambda k: -board[k]["ic"])],
            }
            step += 1
            logger.info("%s %s %s: best=%s IC=%.3f acc=%s", code, tf, label, best, board[best]["ic"],
                        board[best]["dir_acc"])
        manifest["timeframes"][tf] = tf_info

    manifest["trained_at"] = datetime.now(timezone.utc).isoformat()
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    if progress:
        progress("done", 1.0)
    return manifest


# ==========================================================================
# prediction
# ==========================================================================
class NotTrained(Exception):
    pass


def predict_stock(company, timeframe):
    manifest = load_manifest(company.code)
    info = (manifest or {}).get("timeframes", {}).get(timeframe)
    if not info or "horizons" not in info:
        raise NotTrained(info.get("error") if info else f"No {timeframe} models for {company.code} yet - train it first.")
    frame, F = ds.live(company, timeframe)
    tcol = "timestamp" if timeframe == "intraday" else "date"
    last = frame.iloc[-1]
    price = float(last["close"])
    last_time = last[tcol]

    forecasts = []
    for h_str, hinfo in info["horizons"].items():
        h = int(h_str)
        path = stock_dir(company.code) / f"{timeframe}_h{h}.joblib"
        if not path.exists():
            continue
        b = joblib.load(path)
        x = F[b["features"]].iloc[[-1]]
        ret = float(np.mean([_predict(m, x)[0] for m in b["members"]]))
        p_up = None
        if b["calibration"]:
            a, c0 = b["calibration"]
            p_up = 1 / (1 + math.exp(-(a * ret * SCALE + c0)))
        q10, q25, q75, q90 = b["resid_q"]
        if timeframe == "intraday":
            target_time = (last_time + timedelta(minutes=5 * h)).isoformat()
        else:
            target_time = ds.next_trading_days(last_time, h).isoformat()
        best = next(m for m in hinfo["leaderboard"] if m["key"] == hinfo["best"])
        forecasts.append({
            "horizon": h, "label": hinfo["label"], "target_time": target_time,
            "predicted_return_pct": ret * 100, "predicted_price": price * (1 + ret),
            "direction": "UP" if ret > 0 else "DOWN" if ret < 0 else "FLAT",
            "p_up": p_up,
            "range_80": [price * (1 + ret + q10), price * (1 + ret + q90)],
            "range_50": [price * (1 + ret + q25), price * (1 + ret + q75)],
            "model": hinfo["best_name"], "edge": hinfo["edge"],
            "dir_acc": best["dir_acc"], "baseline_acc": best["baseline_acc"], "ic": best["ic"],
            "cost_pct": COST[timeframe] * 100,
        })

    if timeframe == "intraday":
        days = frame["timestamp"].dt.date
        keep = days >= sorted(days.unique())[-3] if days.nunique() >= 3 else days == days
        chart = frame[keep]
        candles = [{"x": t.isoformat(), "o": o, "h": hh, "l": ll, "c": cc} for t, o, hh, ll, cc in
                   zip(chart["timestamp"], chart["open"], chart["high"], chart["low"], chart["close"])]
    else:
        chart = frame.iloc[-150:]
        candles = [{"x": str(d), "o": o, "h": hh, "l": ll, "c": cc} for d, o, hh, ll, cc in
                   zip(chart["date"], chart["open"], chart["high"], chart["low"], chart["close"])]
    return {
        "company": company.code, "name": company.name, "timeframe": timeframe,
        "price": price, "as_of": str(last_time), "candles": candles, "forecasts": forecasts,
        "trained_at": info["trained_at"], "train_rows": info["rows"], "data_from": info["data_from"],
        "data_to": info["data_to"], "feature_count": len(info["features"]),
        "leaderboards": {v["label"]: v["leaderboard"] for v in info["horizons"].values()},
        "cost_pct": COST[timeframe] * 100,
    }


def overview():
    """One row per trained stock (from the manifests - no data loading)."""
    rows = []
    for m in list_manifests():
        row = {"company": m["company"], "name": m.get("name"), "sector": m.get("sector"),
               "industry": m.get("industry"), "trained_at": m.get("trained_at"), "horizons": []}
        for tf, info in m.get("timeframes", {}).items():
            for h, hinfo in (info.get("horizons") or {}).items():
                best = next(x for x in hinfo["leaderboard"] if x["key"] == hinfo["best"])
                row["horizons"].append({"timeframe": tf, "label": hinfo["label"], "model": hinfo["best_name"],
                                        "ic": best["ic"], "dir_acc": best["dir_acc"],
                                        "baseline_acc": best["baseline_acc"], "edge": hinfo["edge"]})
        row["edges"] = sum(1 for x in row["horizons"] if x["edge"])
        row["best_ic"] = max((x["ic"] for x in row["horizons"]), default=None)
        rows.append(row)
    rows.sort(key=lambda r: (-r["edges"], -(r["best_ic"] or -1)))
    return rows

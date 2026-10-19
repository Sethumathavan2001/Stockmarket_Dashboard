"""
Sector-wise and weekday study: which of the Stock Prediction page's ML models predicts best, for every stock.

For each company with 600+ days of daily history in the database, the exact model zoo and
walk-forward test of prediction/engine.py (7 models + 2 ensembles, last 40 % of history in 4 windows,
0.25 % round-trip cost) is run for the next-day and next-5-day return.  Results are then grouped by
sector and industry, and by weekday (every model's test predictions are scored per weekday).  A
descriptive market analysis (returns, volatility, drawdown, trend) of every stock and sector, and a
weekday analysis (return, up-days, gap, open-to-close, range by sector and weekday) are added from the
same database.

Runs in parallel (one stock per CPU core, models single-threaded) and saves every finished stock to
logs/sector_ml_results.jsonl, so a stopped run resumes where it left off.

  python script/sector_ml_research.py                 # all stocks, then the report
  python script/sector_ml_research.py --limit 16      # quick test
  python script/sector_ml_research.py --report-only   # rebuild the Excel / JSON from saved results

Output: script/output/sector_ml_analysis.xlsx and logs/sector_ml_summary.json (for the web page).
"""
import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sethu_project.settings")

RESULTS = BASE / "logs" / "sector_ml_results.jsonl"
SUMMARY = BASE / "logs" / "sector_ml_summary.json"
OUT = BASE / "script" / "output" / "sector_ml_analysis.xlsx"
METRICS = ["dir_acc", "baseline_acc", "ic", "rank_ic", "skill", "trades", "win_rate", "avg_net_pct",
           "top10_dir_acc", "top10_avg_net_pct", "folds_positive", "samples", "edge"]


# ==========================================================================
# worker
# ==========================================================================
def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    import django
    django.setup()
    from prediction import engine
    original = engine._models

    def single_threaded():
        out = {}
        for key, (label, factory) in original().items():
            def make(f=factory):
                m = f()
                params = {k: 1 for k in m.get_params() if k.endswith("n_jobs")}
                if params:
                    m.set_params(**params)
                return m
            out[key] = (label, make)
        return out

    engine._models = single_threaded
    # record every model's test predictions (the zoo scores them through _metrics, in board order)
    original_metrics = engine._metrics

    def capturing_metrics(y, pred, fold_ids, cost):
        CAPTURE.append(np.asarray(pred, float))
        return original_metrics(y, pred, fold_ids, cost)

    engine._metrics = capturing_metrics
    warnings.simplefilter("ignore")


CAPTURE = []
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def run_stock(code):
    import pandas as pd
    from market.models import Company
    from prediction import datasets as ds, engine

    c = Company.objects.get(code=code)
    t0 = time.time()
    try:
        df, X, Y = ds.daily_dataset(c)
    except ValueError as exc:
        return {"code": code, "skipped": str(exc)}
    rows, weekday = [], []
    n = len(df)
    test_from = str(df["date"].iloc[int(n * (1 - engine.TEST_SHARE))])
    dates = pd.to_datetime(df["date"])
    for col in Y.columns:
        h = int(col.split("_")[1])
        CAPTURE.clear()
        board, best, *_ = engine.train_horizon(X, Y[col], "daily", h)
        # the same rows the zoo tested on: drop rows without a target, then the last 40 %
        ok = Y[col].notna().values
        y_ok = Y[col].values[ok]
        n_ok = len(y_ok)
        test = np.arange(int(n_ok * (1 - engine.TEST_SHARE)), n_ok)
        # weekday: of the day being predicted (next session) for 1 day; of the decision day for 5 days
        when = (dates.shift(-1) if h == 1 else dates).values[ok][test]
        wd = pd.DatetimeIndex(when).weekday
        yt = y_ok[test]
        horizon = f"{h} day" + ("s" if h > 1 else "")
        for model, pred in zip(board.keys(), CAPTURE):
            if len(pred) != len(yt):
                continue
            nz = yt != 0
            hit = (np.sign(pred) == np.sign(yt)) & nz
            for d in range(5):
                m_ = (wd == d) & nz
                if m_.sum():
                    weekday.append({"code": code, "sector": c.sector or "Unknown", "horizon": horizon, "model": model,
                                    "weekday": WEEKDAYS[d], "n": int(m_.sum()), "correct": int(hit[m_].sum()),
                                    "up": int((yt[m_] > 0).sum())})
        for model, m in board.items():
            rows.append({"code": code, "name": c.name, "sector": c.sector or "Unknown", "industry": c.industry or "Unknown",
                         "horizon": f"{h} day" + ("s" if h > 1 else ""), "model": model, "best": model == best,
                         **{k: m.get(k) for k in METRICS}})
    return {"code": code, "rows": rows, "weekday": weekday, "days": n, "test_from": test_from, "to": str(df["date"].iloc[-1]),
            "seconds": round(time.time() - t0, 1)}


# ==========================================================================
# main run
# ==========================================================================
def train_all(limit=None, workers=8):
    import django
    django.setup()
    from django.db.models import Count
    from market.models import Company, DailyData

    counts = dict(DailyData.objects.values_list("company").annotate(n=Count("id")).values_list("company", "n"))
    companies = [c for c in Company.objects.exclude(sector="ETF").order_by("code") if counts.get(c.pk, 0) >= 600]
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["code"])
            except (ValueError, KeyError):
                pass
    todo = [c.code for c in companies if c.code not in done][:limit] if limit else [c.code for c in companies if c.code not in done]
    print(f"{len(companies)} stocks with 600+ days; {len(done)} already done; {len(todo)} to run on {workers} workers", flush=True)
    RESULTS.parent.mkdir(exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool, open(RESULTS, "a", encoding="utf-8") as fh:
        futures = {pool.submit(run_stock, code): code for code in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            code = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:                       # one bad stock must not stop the run
                res = {"code": code, "skipped": f"{type(exc).__name__}: {exc}"}
            fh.write(json.dumps(res, default=str) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{el / 60:6.1f} min] {i}/{len(todo)} done · {el / i:.1f} s/stock · "
                      f"~{(len(todo) - i) * el / i / 60:.0f} min left", flush=True)


# ==========================================================================
# market analysis (descriptive, from the database)
# ==========================================================================
def market_analysis():
    import numpy as np
    import pandas as pd
    from market.models import Company, DailyData

    meta = pd.DataFrame(list(Company.objects.exclude(sector="ETF").values("id", "code", "name", "sector", "industry")))
    last = DailyData.objects.order_by("-date").values_list("date", flat=True).first()
    rows = DailyData.objects.filter(date__gte=last - pd.Timedelta(days=1150)).values_list(
        "company_id", "date", "close", "volume")
    df = pd.DataFrame(list(rows), columns=["cid", "date", "close", "volume"]).dropna(subset=["close"])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].fillna(0).astype(float)
    df = df[df["close"] > 0].sort_values(["cid", "date"])
    out = []
    for cid, g in df.groupby("cid"):
        if len(g) < 60 or g["date"].iloc[-1] < last - pd.Timedelta(days=10):
            continue
        c = g["close"].to_numpy()
        r = np.diff(np.log(c))

        def ret(days):
            return (c[-1] / c[-1 - days] - 1) * 100 if len(c) > days else np.nan
        y1 = c[-250:]
        peak = np.maximum.accumulate(y1)
        out.append({"cid": cid, "close": c[-1], "ret_1m": ret(21), "ret_3m": ret(63), "ret_6m": ret(126),
                    "ret_1y": ret(250), "ret_3y": ret(750),
                    "vol_1y": float(np.std(r[-250:]) * np.sqrt(250) * 100) if len(r) > 60 else np.nan,
                    "max_dd_1y": float(((y1 / peak) - 1).min() * 100),
                    "from_52w_high": (c[-1] / y1.max() - 1) * 100,
                    "above_200d": bool(len(c) >= 200 and c[-1] > c[-200:].mean()),
                    "turnover_cr": float((g["close"] * g["volume"]).tail(20).mean() / 1e7)})
    m = pd.DataFrame(out).merge(meta, left_on="cid", right_on="id").drop(columns=["cid", "id"])
    return m, str(last)


def weekday_market(years=3):
    """Every stock-day of the last `years` years: return, up-day, gap, open-to-close and range by sector x weekday."""
    import pandas as pd
    from market.models import Company, DailyData

    last = DailyData.objects.order_by("-date").values_list("date", flat=True).first()
    meta = dict(Company.objects.exclude(sector="ETF").values_list("id", "sector"))
    rows = DailyData.objects.filter(date__gte=last - pd.Timedelta(days=365 * years + 10),
                                    company_id__in=list(meta)).values_list("company_id", "date", "open", "high", "low", "close")
    df = pd.DataFrame(list(rows), columns=["cid", "date", "open", "high", "low", "close"]).dropna()
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df = df[(df["low"] > 0) & (df["high"] >= df["low"])].sort_values(["cid", "date"])
    prev = df.groupby("cid")["close"].shift()
    gapdays = (pd.to_datetime(df["date"]) - pd.to_datetime(df.groupby("cid")["date"].shift())).dt.days
    df["ret"] = (df["close"] / prev - 1) * 100
    df["gap"] = (df["open"] / prev - 1) * 100
    df["oc"] = (df["close"] / df["open"] - 1) * 100
    df["range"] = (df["high"] - df["low"]) / prev * 100
    df = df[prev.notna() & (gapdays <= 5)]                               # skip data holes
    df = df[df["ret"].abs() < 25]                                        # splits / bad prints
    df["sector"] = df["cid"].map(meta).fillna("Unknown")
    df["weekday"] = pd.to_datetime(df["date"]).dt.weekday.map(dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
    df = df[df["weekday"].isin(["Mon", "Tue", "Wed", "Thu", "Fri"])]

    def agg(g):
        r = g["ret"]
        return pd.Series({"stock_days": len(g), "avg_return_pct": r.mean(), "median_return_pct": r.median(),
                          "up_days_pct": (r > 0).mean() * 100, "avg_gap_pct": g["gap"].mean(),
                          "gap_up_pct": (g["gap"] > 0).mean() * 100, "avg_open_to_close_pct": g["oc"].mean(),
                          "avg_range_pct": g["range"].mean(), "t_stat": r.mean() / (r.std() / np.sqrt(len(r))) if len(r) > 1 else 0})
    by_sector = df.groupby(["sector", "weekday"]).apply(agg).reset_index()
    # t-stat over trading days (stocks in a sector move together, so stock-days overstate certainty)
    sd = df.groupby(["sector", "date", "weekday"])["ret"].mean().reset_index()
    t_days = sd.groupby(["sector", "weekday"])["ret"].agg(
        days="count", t_stat=lambda s: s.mean() / (s.std() / np.sqrt(len(s))) if len(s) > 1 and s.std() > 0 else 0.0)
    by_sector = by_sector.drop(columns=["t_stat"]).merge(t_days.reset_index(), on=["sector", "weekday"])
    overall = df.groupby("weekday").apply(agg).reset_index()
    # per-day market average across stocks (each day counts once) - the honest unit for a weekday effect
    daily = df.groupby(["date", "weekday"])["ret"].mean().reset_index()
    day_level = daily.groupby("weekday")["ret"].agg(days="count", avg_market_return_pct="mean",
                                                    up_days_pct=lambda s: (s > 0).mean() * 100).reset_index()
    day_level["t_stat"] = daily.groupby("weekday")["ret"].apply(lambda s: s.mean() / (s.std() / np.sqrt(len(s)))).values
    order = {d: i for i, d in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"])}
    for t_ in (by_sector, overall, day_level):
        t_.sort_values([c for c in ("sector",) if c in t_] + ["weekday"], key=lambda s: s.map(order) if s.name == "weekday" else s,
                       inplace=True)
    return by_sector, overall, day_level, str(df["date"].min()), str(last)


# ==========================================================================
# report
# ==========================================================================
NAMES = {"ridge": "Ridge regression", "knn": "k-Nearest Neighbours", "rf": "Random Forest", "et": "Extra Trees",
         "hgb": "Histogram Gradient Boosting", "xgb": "XGBoost", "mlp": "Neural network (MLP)",
         "ens_all": "Ensemble - all models", "ens_top3": "Ensemble - top 3"}


def report():
    import django
    django.setup()
    import numpy as np
    import pandas as pd

    recs, skipped, meta, wrecs = [], [], [], []
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "rows" in r:
            recs.extend(r["rows"])
            wrecs.extend(r.get("weekday", []))
            meta.append({k: r[k] for k in ("code", "days", "test_from", "to", "seconds")})
        else:
            skipped.append(r)
    R = pd.DataFrame(recs)
    R["model_name"] = R["model"].map(NAMES)
    R["acc_vs_base"] = (R["dir_acc"] - R["baseline_acc"]) * 100
    R["dir_acc_pct"] = R["dir_acc"] * 100
    R["edge"] = R["edge"].astype(bool)

    def agg(g):
        return pd.Series({
            "stocks": g["code"].nunique(),
            "avg_dir_acc_pct": g["dir_acc_pct"].mean(),
            "avg_vs_baseline_pts": g["acc_vs_base"].mean(),
            "beats_baseline_pct": (g["acc_vs_base"] > 0).mean() * 100,
            "avg_ic": g["ic"].mean(),
            "positive_ic_pct": (g["ic"] > 0).mean() * 100,
            "avg_net_per_trade_pct": g["avg_net_pct"].mean(),
            "top10_dir_acc_pct": g["top10_dir_acc"].mean() * 100,
            "edge_stocks_pct": g["edge"].mean() * 100,
            "best_model_pct": g["best"].mean() * 100,
        })

    overall = R.groupby(["horizon", "model_name"]).apply(agg).reset_index()
    sector_model = R.groupby(["horizon", "sector", "model_name"]).apply(agg).reset_index()
    industry_model = R.groupby(["horizon", "sector", "industry", "model_name"]).apply(agg).reset_index()

    # winner per sector: highest average accuracy over baseline (ties -> IC)
    def winners(table, keys):
        t = table.sort_values(["avg_vs_baseline_pts", "avg_ic"], ascending=False)
        first = t.groupby(keys).head(1).set_index(keys)
        second = t.groupby(keys).nth(1).set_index(keys)["model_name"].rename("runner_up")
        out = first.join(second).reset_index()
        return out[keys + ["model_name", "runner_up", "stocks", "avg_dir_acc_pct", "avg_vs_baseline_pts",
                           "beats_baseline_pct", "avg_ic", "avg_net_per_trade_pct", "edge_stocks_pct"]]

    sector_win = winners(sector_model, ["horizon", "sector"])
    industry_win = winners(industry_model[industry_model["stocks"] >= 5], ["horizon", "sector", "industry"])

    # best model per stock (by the page's own rule: highest IC)
    per_stock_best = R[R["best"]][["code", "name", "sector", "industry", "horizon", "model_name", "dir_acc_pct",
                                   "baseline_acc", "acc_vs_base", "ic", "avg_net_pct", "edge"]]
    per_stock_best = per_stock_best.assign(baseline_acc=per_stock_best["baseline_acc"] * 100)
    per_stock_all = R[["code", "name", "sector", "industry", "horizon", "model_name", "dir_acc_pct", "acc_vs_base",
                       "ic", "rank_ic", "skill", "trades", "win_rate", "avg_net_pct", "top10_dir_acc",
                       "top10_avg_net_pct", "folds_positive", "edge", "best"]]

    # ---- model accuracy by weekday (pooled over test days; baseline = the more common direction that weekday)
    W = pd.DataFrame(wrecs)
    W["model_name"] = W["model"].map(NAMES)
    W["base_hits"] = np.maximum(W["up"], W["n"] - W["up"])

    def wagg(g):
        n = g["n"].sum()
        return pd.Series({"stocks": g["code"].nunique(), "test_days": n, "dir_acc_pct": g["correct"].sum() / n * 100,
                          "baseline_pct": g["base_hits"].sum() / n * 100,
                          "vs_baseline_pts": (g["correct"].sum() - g["base_hits"].sum()) / n * 100,
                          "up_share_pct": g["up"].sum() / n * 100})
    wd_overall = W.groupby(["horizon", "weekday", "model_name"]).apply(wagg).reset_index()
    wd_sector = W.groupby(["horizon", "sector", "weekday", "model_name"]).apply(wagg).reset_index()
    wd_best = (wd_sector.sort_values("dir_acc_pct", ascending=False).groupby(["horizon", "sector", "weekday"]).head(1)
               .sort_values(["horizon", "sector", "weekday"]))
    wk_sector, wk_overall, wk_days, wk_from, wk_to = weekday_market()

    M, last_day = market_analysis()
    sector_mkt = M.groupby("sector").agg(
        stocks=("code", "count"), median_ret_1m=("ret_1m", "median"), median_ret_3m=("ret_3m", "median"),
        median_ret_6m=("ret_6m", "median"), median_ret_1y=("ret_1y", "median"), median_ret_3y=("ret_3y", "median"),
        median_vol_1y=("vol_1y", "median"), median_max_dd_1y=("max_dd_1y", "median"),
        above_200d_pct=("above_200d", lambda s: s.mean() * 100), positive_1m_pct=("ret_1m", lambda s: (s > 0).mean() * 100),
        median_from_52w_high=("from_52w_high", "median"), total_turnover_cr=("turnover_cr", "sum"),
    ).reset_index().sort_values("median_ret_3m", ascending=False)
    leaders = (M[M["turnover_cr"] >= 5].sort_values("ret_3m", ascending=False).groupby("sector")
               .head(3)[["sector", "code", "name", "ret_3m", "ret_1y", "close"]])
    laggards = (M[M["turnover_cr"] >= 5].sort_values("ret_3m").groupby("sector")
                .head(3)[["sector", "code", "name", "ret_3m", "ret_1y", "close"]])

    info = pd.DataFrame(meta)
    method = pd.DataFrame([
        ["Stocks studied", f"{R['code'].nunique()} (every company with 600+ days of daily history; ETFs excluded)"],
        ["Skipped", f"{len(skipped)} (not enough clean history)"],
        ["Data", f"Daily candles in the database, up to {info['to'].max()} (latest 2,500 days per stock)"],
        ["Models", "The Stock Prediction page's zoo: Ridge, k-NN, Random Forest, Extra Trees, Histogram Gradient "
                   "Boosting, XGBoost, Neural network (MLP), Ensemble of all, Ensemble of top 3"],
        ["Targets", "Next-day return (1 day) and next-5-day return (5 days)"],
        ["Testing", "Walk-forward: trained on older data, tested on the last 40 % of each stock's history in 4 windows; "
                    "a model never sees the period it is scored on"],
        ["Direction accuracy", "Share of test days where the predicted direction was right (flat days excluded)"],
        ["Baseline", "Always predicting the stock's more common direction in the test period (hindsight - generous)"],
        ["vs baseline", "Direction accuracy minus baseline, in percentage points. Above 0 = better than the simple rule"],
        ["IC", "Correlation of predicted and actual return. 0 = no skill; 0.03+ is useful in practice"],
        ["Net per trade", "Average return per trade when the model predicts a move larger than 0.25 % costs, after costs"],
        ["Edge", "The page's rule: IC >= 0.03, positive in 3 of 4 windows, beats baseline, 30+ trades, profitable after costs"],
        ["Best model share", "How often the model was the page's pick (highest IC) for a stock"],
        ["Market analysis", f"Returns, volatility, drawdown and trend from daily closes up to {last_day}"],
        ["Weekday - models", "Each model's test predictions scored per weekday. Next-day: weekday of the day predicted; "
                             "5-day: weekday the prediction was made. Baseline = the more common direction on that weekday"],
        ["Weekday - market", f"Every stock-day from {wk_from} to {wk_to}: close-to-close return, up-days, gap at the open, "
                             "open-to-close move and day range, by sector and weekday. t-stat > 2 suggests the average is "
                             "unlikely to be zero by chance (but 5 weekdays x many sectors will throw up some by luck)"],
    ], columns=["Item", "Detail"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUT
    try:
        with open(out, "ab"):
            pass
    except PermissionError:                    # the file is open in Excel - save next to it instead
        out = OUT.with_name(f"{OUT.stem}_{time.strftime('%Y%m%d_%H%M')}{OUT.suffix}")
        print(f"{OUT.name} is open in another program - saving as {out.name}")
    with pd.ExcelWriter(out) as xl:
        method.to_excel(xl, sheet_name="Method", index=False)
        overall.sort_values(["horizon", "avg_vs_baseline_pts"], ascending=[True, False]).to_excel(xl, sheet_name="Models overall", index=False)
        sector_win.to_excel(xl, sheet_name="Best model by sector", index=False)
        sector_model.sort_values(["horizon", "sector", "avg_vs_baseline_pts"], ascending=[True, True, False]).to_excel(
            xl, sheet_name="Sector x model", index=False)
        industry_win.to_excel(xl, sheet_name="Best model by industry", index=False)
        per_stock_best.sort_values(["horizon", "sector", "code"]).to_excel(xl, sheet_name="Best model per stock", index=False)
        per_stock_all.sort_values(["horizon", "code", "acc_vs_base"], ascending=[True, True, False]).to_excel(
            xl, sheet_name="All stocks x models", index=False)
        sector_mkt.to_excel(xl, sheet_name="Sector market analysis", index=False)
        pd.concat([leaders.assign(group="Leader"), laggards.assign(group="Laggard")]).to_excel(
            xl, sheet_name="Sector leaders & laggards", index=False)
        M.sort_values(["sector", "code"]).to_excel(xl, sheet_name="Stock market analysis", index=False)
        wk_days.to_excel(xl, sheet_name="Weekday market overall", index=False)
        wk_overall.to_excel(xl, sheet_name="Weekday stock-days overall", index=False)
        wk_sector.to_excel(xl, sheet_name="Weekday market by sector", index=False)
        wd_overall.sort_values(["horizon", "weekday", "dir_acc_pct"], ascending=[True, True, False]).to_excel(
            xl, sheet_name="Weekday model accuracy", index=False)
        wd_best.to_excel(xl, sheet_name="Best model by sector+day", index=False)
        wd_sector.to_excel(xl, sheet_name="Sector x weekday x model", index=False)
        if skipped:
            pd.DataFrame(skipped).to_excel(xl, sheet_name="Skipped", index=False)
    print(f"saved {out}")

    def records(df):
        return json.loads(df.replace({np.nan: None}).to_json(orient="records"))
    SUMMARY.write_text(json.dumps({
        "stocks": int(R["code"].nunique()), "skipped": len(skipped), "data_to": str(info["to"].max()),
        "market_to": last_day, "overall": records(overall), "sector_model": records(sector_model),
        "sector_win": records(sector_win), "sector_mkt": records(sector_mkt),
        "leaders": records(leaders), "laggards": records(laggards),
        "edge_count": records(R[R["edge"]].groupby(["horizon", "model_name"])["code"].nunique().reset_index()),
        "wd_overall": records(wd_overall), "wd_sector": records(wd_sector), "wd_best": records(wd_best),
        "wk_sector": records(wk_sector), "wk_overall": records(wk_overall), "wk_days": records(wk_days),
        "wk_from": wk_from, "wk_to": wk_to,
    }, default=str), encoding="utf-8")
    print(f"saved {SUMMARY}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        train_all(a.limit, a.workers)
    report()

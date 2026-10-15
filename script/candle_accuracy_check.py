"""
How accurate can a 5-minute candle direction forecast really be?  A large walk-forward check.

Universe : the N most-traded stocks in the database (default 150), all 5-minute candles.
Windows  : two unseen test periods - A = sessions [-80, -40), B = the last 40 sessions.
           Every model is trained only on the sessions before its test window.

Tested
  1. "page"   - the Candle Predictor's current analog (nearest-neighbour) forecaster, per stock,
                exactly as the page scores itself (engine.backtest_analogs).
  2. "pooled" - gradient boosting trained across all the stocks at once, with NIFTY context.
                Scored on every candle, by confidence, by time of day, and only on moves bigger
                than intraday costs (0.08 %).
  3. "gate"   - does picking the stocks that looked accurate in window A still work in window B?

Output: script/output/candle_accuracy_check.xlsx  (+ a summary printed to the console)

Run:  python script/candle_accuracy_check.py [N_STOCKS]
"""
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sethu_project.settings")
import django  # noqa: E402

django.setup()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from ai_prediction.tools import build_screen  # noqa: E402
from market.models import Company  # noqa: E402
from patterns import engine  # noqa: E402
from prediction.data import load_historical_candles  # noqa: E402

N_STOCKS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
WINDOW = 40
TRAIN_DAYS = 150
HORIZONS = {1: "5 min", 3: "15 min", 6: "30 min", 12: "60 min"}
COST = 0.0008
OUT = BASE / "script" / "output" / "candle_accuracy_check.xlsx"
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)


# ---------------------------------------------------------------- data
screen, _ = build_screen()
codes = screen.sort_values("turnover_cr_20d", ascending=False)["code"].head(N_STOCKS).tolist()

mk = load_historical_candles(Company.objects.get(code="NIFTYBEES.NS"))[["timestamp", "close"]]
mk = engine._session(mk.assign(open=mk.close, high=mk.close, low=mk.close, volume=0))
mk["day"] = mk.timestamp.dt.date
mk["m_ret1"] = mk.groupby("day").close.pct_change(1)
mk["m_ret3"] = mk.groupby("day").close.pct_change(3)
mk["m_day"] = mk.close / mk.groupby("day").close.transform("first") - 1
mk = mk[["timestamp", "m_ret1", "m_ret3", "m_day"]]

frames, page_rows = [], []
for i, code in enumerate(codes, 1):
    c = Company.objects.get(code=code)
    df = engine._session(load_historical_candles(c)[["timestamp", "open", "high", "low", "close", "volume"]])
    if len(df) < 6000:
        continue
    b = engine.Bars(df, "5m")
    X = engine.features(b)

    # 1. the page's own analog forecaster, scored twice (window A, then window B)
    ndays = len(b.days)
    for name, upto in (("A", ndays - WINDOW), ("B", ndays)):
        hist = b.day_id < upto
        # backtest_analogs tests on the last fifth of the history days - force that to be exactly WINDOW days
        keep = b.day_id >= upto - WINDOW * 5
        bt = engine.backtest_analogs(b, X, hist & keep)
        for h, label in HORIZONS.items():
            r = (bt or {}).get("horizons", {}).get(h)
            if r:
                page_rows.append({"stock": code, "window": name, "horizon": label, "accuracy": r["accuracy"],
                                  "baseline_majority": r["baseline"], "beats_baseline": r["accuracy"] > r["baseline"],
                                  "confident_n": r["confident_n"], "confident_acc": r["confident_acc"]})

    f = pd.DataFrame(X.astype(np.float32), columns=[f"f{j}" for j in range(X.shape[1])])
    vol = np.where(b.vol > 0, b.vol, np.nan)
    f["rsi"] = b.rsi.astype(np.float32)
    f["ema_gap"] = ((b.ema9 / b.ema21 - 1) / vol).astype(np.float32)
    f["gap"] = (b.day_open / b.pdc - 1).astype(np.float32)
    f["or_pos"] = np.where(b.or_ready, (b.c - b.or_low) / np.maximum(b.or_high - b.or_low, 1e-9), np.nan).astype(np.float32)
    f["vol"] = b.vol.astype(np.float32)
    f["pos"] = b.pos.astype(np.int16)
    f["timestamp"] = df.timestamp.reset_index(drop=True)
    f["stock"] = code
    f["day"] = b.day
    for h in HORIZONS:
        f[f"y{h}"] = b.fwd[h].astype(np.float32)
    frames.append(f)
    if i % 10 == 0:
        log(f"loaded {i}/{len(codes)} stocks")

D = pd.concat(frames, ignore_index=True).merge(mk, on="timestamp", how="left")
del frames
days = sorted(D.day.unique())
FEAT = [c for c in D.columns if c.startswith("f")] + ["rsi", "ema_gap", "gap", "or_pos", "vol", "pos",
                                                       "m_ret1", "m_ret3", "m_day"]
log(f"{D.stock.nunique()} stocks, {len(D):,} candles, {days[0]} .. {days[-1]}")

# ---------------------------------------------------------------- 2. pooled model
windows = {"A": (days[-2 * WINDOW], days[-WINDOW]), "B": (days[-WINDOW], None)}
pooled, deciles, tod, per_stock = [], [], [], []
for wname, (start, end) in windows.items():
    si = days.index(start)
    train_from = days[max(0, si - TRAIN_DAYS)]
    in_test = (D.day >= start) & ((D.day < end) if end else True)
    in_train = (D.day >= train_from) & (D.day < start)
    for h, label in HORIZONS.items():
        y = D[f"y{h}"]
        ok = y.notna() & (y != 0)
        tr, te = ok & in_train, ok & in_test
        m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300,
                                           l2_regularization=1.0, random_state=0)
        m.fit(D.loc[tr, FEAT], (y[tr] > 0).astype(np.int8))
        p = m.predict_proba(D.loc[te, FEAT])[:, 1]
        yt = (y[te] > 0).to_numpy()
        mv = np.abs(y[te].to_numpy())
        hit = (p > 0.5) == yt
        up = yt.mean()
        conf = np.abs(p - 0.5)
        row = {"window": wname, "horizon": label, "test_candles": int(te.sum()),
               "accuracy_all": hit.mean(), "coin_flip": 0.5, "always_majority": max(up, 1 - up),
               "acc_moves_gt_cost": hit[mv > COST].mean(), "share_moves_gt_cost": (mv > COST).mean()}
        for q, tag in ((0.8, "top20"), (0.9, "top10"), (0.95, "top5"), (0.99, "top1")):
            sel = conf >= np.quantile(conf, q)
            row[f"acc_{tag}_confident"] = hit[sel].mean()
        pooled.append(row)
        dec = pd.qcut(conf, 10, labels=False, duplicates="drop")
        for d_ in sorted(set(dec)):
            s = dec == d_
            deciles.append({"window": wname, "horizon": label, "confidence_decile": int(d_) + 1,
                            "p_up_distance_from_50": f"{conf[s].min():.3f}-{conf[s].max():.3f}",
                            "n": int(s.sum()), "accuracy": hit[s].mean()})
        sub = D.loc[te, ["stock", "timestamp"]].assign(hit=hit, conf=conf)
        sub["slot"] = sub.timestamp.dt.strftime("%H") + ":00"
        for slot, g in sub.groupby("slot"):
            tod.append({"window": wname, "horizon": label, "hour": slot, "n": len(g), "accuracy": g.hit.mean(),
                        "acc_top10_confident": g.hit[g.conf >= np.quantile(conf, 0.9)].mean()})
        if label == "15 min":
            thr = np.quantile(conf, 0.9)
            for st, g in sub.groupby("stock"):
                gc = g[g.conf >= thr]
                per_stock.append({"window": wname, "stock": st, "n": len(g), "accuracy": g.hit.mean(),
                                  "confident_n": len(gc), "confident_acc": gc.hit.mean() if len(gc) else np.nan})
        log(f"pooled window {wname} {label}: acc {hit.mean():.3f}, top10% {row['acc_top10_confident']:.3f}")

pooled = pd.DataFrame(pooled)
page = pd.DataFrame(page_rows)
per_stock = pd.DataFrame(per_stock)

# ---------------------------------------------------------------- 3. does choosing "good" stocks carry forward?
gate = []
for h in HORIZONS.values():
    a = page[(page.window == "A") & (page.horizon == h)].set_index("stock")
    bb = page[(page.window == "B") & (page.horizon == h)].set_index("stock")
    j = a.join(bb, lsuffix="_A", rsuffix="_B", how="inner")
    for tag, sel in (("page: beat baseline in A", j.beats_baseline_A),
                     ("page: accuracy >= 55% in A", j.accuracy_A >= 0.55),
                     ("page: all stocks", j.accuracy_A > -1)):
        g = j[sel]
        gate.append({"method": tag, "horizon": h, "stocks": len(g),
                     "accuracy_in_A": g.accuracy_A.mean(), "accuracy_in_B": g.accuracy_B.mean(),
                     "beat_baseline_in_B_share": g.beats_baseline_B.mean() if len(g) else np.nan})
ps = per_stock.pivot(index="stock", columns="window", values="confident_acc").dropna()
for thr in (0.55, 0.60):
    g = ps[ps["A"] >= thr]
    gate.append({"method": f"pooled top10%: confident acc >= {thr:.0%} in A", "horizon": "15 min", "stocks": len(g),
                 "accuracy_in_A": g["A"].mean(), "accuracy_in_B": g["B"].mean(), "beat_baseline_in_B_share": np.nan})
gate = pd.DataFrame(gate)

# ---------------------------------------------------------------- summary
page_sum = (page.groupby(["window", "horizon"])
            .agg(stocks=("stock", "nunique"), avg_accuracy=("accuracy", "mean"),
                 avg_baseline=("baseline_majority", "mean"), share_beating_baseline=("beats_baseline", "mean"),
                 best_stock_accuracy=("accuracy", "max"), worst_stock_accuracy=("accuracy", "min"),
                 avg_confident_acc=("confident_acc", "mean"))
            .reset_index())
best_any = max([pooled[[c for c in pooled.columns if c.startswith("acc_")]].max().max(), page.accuracy.max()])
summary = pd.DataFrame([
    ["Stocks tested", D.stock.nunique()],
    ["5-minute candles", len(D)],
    ["Data", f"{days[0]} to {days[-1]}"],
    ["Test window A", f"{windows['A'][0]} to {days[-WINDOW - 1]}"],
    ["Test window B", f"{windows['B'][0]} to {days[-1]}"],
    ["Page engine - average accuracy, 15 min (A / B)",
     " / ".join(f"{page_sum[(page_sum.window == w) & (page_sum.horizon == '15 min')].avg_accuracy.iloc[0]:.1%}" for w in "AB")],
    ["Page engine - average baseline, 15 min (A / B)",
     " / ".join(f"{page_sum[(page_sum.window == w) & (page_sum.horizon == '15 min')].avg_baseline.iloc[0]:.1%}" for w in "AB")],
    ["Pooled model - all candles, 15 min (A / B)",
     " / ".join(f"{pooled[(pooled.window == w) & (pooled.horizon == '15 min')].accuracy_all.iloc[0]:.1%}" for w in "AB")],
    ["Pooled model - top 1% confident, 15 min (A / B)",
     " / ".join(f"{pooled[(pooled.window == w) & (pooled.horizon == '15 min')].acc_top1_confident.iloc[0]:.1%}" for w in "AB")],
    ["Highest accuracy seen anywhere in this test", f"{best_any:.1%}"],
], columns=["item", "value"])

OUT.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT) as xl:
    summary.to_excel(xl, "Summary", index=False)
    page_sum.to_excel(xl, "Page engine summary", index=False)
    page.to_excel(xl, "Page engine per stock", index=False)
    pooled.to_excel(xl, "Pooled model", index=False)
    pd.DataFrame(deciles).to_excel(xl, "By confidence", index=False)
    pd.DataFrame(tod).to_excel(xl, "By time of day", index=False)
    per_stock.to_excel(xl, "Pooled per stock 15m", index=False)
    gate.to_excel(xl, "Pick good stocks", index=False)

pd.set_option("display.width", 200, "display.max_columns", 20)
print("\n=== SUMMARY ===\n", summary.to_string(index=False))
print("\n=== PAGE ENGINE (current Candle Predictor) ===\n", page_sum.round(3).to_string(index=False))
print("\n=== POOLED MODEL ===\n", pooled.round(3).to_string(index=False))
print("\n=== PICKING 'GOOD' STOCKS IN A, CHECKED IN B ===\n", gate.round(3).to_string(index=False))
log(f"saved {OUT}")

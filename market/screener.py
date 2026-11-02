"""
Analysis page: market-wide indicators for every stock, from the daily table.

build_snapshot() loads the last LOOKBACK_DAYS of DailyData once (with volume), works out per stock

  returns      1D, 1W, 1M, 3M, 6M, 1Y
  trend        20 / 50 / 200-day averages, price vs them, golden / death cross (and how many days ago)
  momentum     RSI 14 (Wilder), MACD 12/26/9 and a fresh cross, relative strength vs NIFTY over 3M
  range        52-week high / low, distance from each, new 52-week high / low today
  activity     volume vs its 20-day average, 20-day average turnover (Rs crore), ATR 14 as % of price,
               up / down streak, today's gap from the previous close
  score        0-100 trend score (simple, explained on the page)

plus market breadth and a per-sector summary, and saves it all to logs/analysis_snapshot.json - the page
reads that file, so it opens instantly.  Run by the Analysis page's Refresh button (analysis_engine).
"""
import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings

from .models import Company, DailyData

logger = logging.getLogger(__name__)

SNAPSHOT = Path(settings.BASE_DIR) / "logs" / "analysis_snapshot.json"
LOOKBACK_DAYS = 420          # calendar days loaded: a full 52 weeks + room for the 200-day average
SPARK_POINTS = 60            # sparkline: last ~6 months, thinned to this many points
PERIODS = {"r1d": 1, "r1w": 5, "r1m": 21, "r3m": 63, "r6m": 126, "r1y": 252}


def _num(v, nd=2):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, nd)


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + up / dn)
    rsi[(dn == 0) & (up > 0)] = 100.0                      # only gains in the window
    rsi[(dn == 0) & (up == 0)] = 50.0                      # flat
    return rsi


def _cross_age(a, b, lookback=30):
    """When a last crossed b: +k crossed above, -k crossed below, k trading days ago counting today as 1;
    None if no cross within lookback."""
    diff = (a - b).dropna()
    if len(diff) < 2:
        return None
    sign = np.sign(diff.values)
    for k in range(1, min(lookback, len(sign) - 1) + 1):
        now, before = sign[-k], sign[-k - 1]
        if now != before and now != 0:
            return k if now > 0 else -k
    return None


def _benchmark_r3m():
    """NIFTY 50's 3-month return (for relative strength); None when Yahoo is unavailable."""
    try:
        from prediction.datasets import _yahoo, _yahoo_frame
        df = _yahoo_frame(_yahoo("^NSEI", cache_seconds=3600, period="1y", interval="1d", prepost=False))
        c = df["close"].astype(float)
        return float(c.iloc[-1] / c.iloc[-1 - PERIODS["r3m"]] - 1) * 100 if len(c) > PERIODS["r3m"] else None
    except Exception as exc:
        logger.warning("analysis: NIFTY benchmark unavailable: %s", exc)
        return None


def stock_metrics(g, nifty_r3m):
    """Indicators for one stock's daily rows g (date-sorted DataFrame: date, open, high, low, close, volume)."""
    c, h, l, v = g["close"], g["high"], g["low"], g["volume"].fillna(0)
    n = len(c)
    last = float(c.iloc[-1])
    out = {"price": _num(last), "date": g["date"].iloc[-1].isoformat(), "bars": n}
    for k, p in PERIODS.items():
        out[k] = _num((last / float(c.iloc[-1 - p]) - 1) * 100) if n > p and c.iloc[-1 - p] else None
    sma = {w: c.rolling(w).mean() for w in (20, 50, 200)}
    for w, s in sma.items():
        val = s.iloc[-1]
        out[f"sma{w}"] = _num(val)
        out[f"vs{w}"] = _num((last / val - 1) * 100) if val and not math.isnan(val) else None
    out["cross"] = _cross_age(sma[50], sma[200]) if n >= 200 else None        # +k golden, -k death cross, k days ago
    rsi = _rsi(c)
    out["rsi"] = _num(rsi.iloc[-1], 1)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    out["macd"], out["macd_sig"], out["macd_hist"] = _num(macd.iloc[-1], 3), _num(sig.iloc[-1], 3), _num((macd - sig).iloc[-1], 3)
    out["macd_cross"] = _cross_age(macd, sig, lookback=5) if n >= 35 else None
    yr = g.tail(252)
    hi, lo = float(yr["high"].max()), float(yr["low"].min())
    out["hi52"], out["lo52"] = _num(hi), _num(lo)
    out["from_hi"] = _num((last / hi - 1) * 100) if hi else None
    out["from_lo"] = _num((last / lo - 1) * 100) if lo else None
    out["pos52"] = _num((last - lo) / (hi - lo) * 100, 1) if hi > lo else None
    prev = g.iloc[:-1].tail(251)
    out["new_hi"] = bool(len(prev) >= 120 and float(h.iloc[-1]) > float(prev["high"].max()))
    out["new_lo"] = bool(len(prev) >= 120 and float(l.iloc[-1]) < float(prev["low"].min()))
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    out["atr_pct"] = _num(atr / last * 100) if last else None
    avg_vol = v.iloc[-21:-1].mean() if n > 21 else None
    out["vol"] = int(v.iloc[-1]) if not math.isnan(v.iloc[-1]) else None
    out["vol_x"] = _num(v.iloc[-1] / avg_vol) if avg_vol else None
    out["turnover_cr"] = _num((c * v).tail(20).mean() / 1e7)
    d = np.sign(c.diff().dropna().values)
    streak = 0
    for s in d[::-1]:
        if s == 0 or (streak and np.sign(streak) != s):
            break
        streak += int(s)
    out["streak"] = streak
    out["gap"] = _num((float(g["open"].iloc[-1]) / float(pc.iloc[-1]) - 1) * 100) if n > 1 and pc.iloc[-1] else None
    out["rs3m"] = _num(out["r3m"] - nifty_r3m) if out["r3m"] is not None and nifty_r3m is not None else None
    # trend score 0-100: price above 50 / 200-day, 50 above 200, MACD above signal, RSI 50-70, 3M return up, near 52w high
    pts = [last > (out["sma50"] or math.inf), last > (out["sma200"] or math.inf),
           (out["sma50"] or 0) > (out["sma200"] or math.inf), (out["macd_hist"] or 0) > 0,
           50 <= (out["rsi"] or 0) <= 70, (out["r3m"] or 0) > 0, (out["from_hi"] or -100) > -10]
    out["score"] = round(sum(pts) / len(pts) * 100)
    spark = c.tail(126).values
    step = max(1, len(spark) // SPARK_POINTS)
    out["spark"] = [round(float(x), 2) for x in spark[::step]]
    return out


def build_snapshot(progress=None):
    latest = DailyData.objects.order_by("-date").values_list("date", flat=True).first()
    if latest is None:
        return None
    start = latest - timedelta(days=LOOKBACK_DAYS)
    rows = list(DailyData.objects.filter(date__gte=start, close__isnull=False)
                .values_list("company_id", "date", "open", "high", "low", "close", "volume"))
    df = pd.DataFrame(rows, columns=["cid", "date", "open", "high", "low", "close", "volume"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values(["cid", "date"])
    companies = {c["id"]: c for c in Company.objects.values("id", "code", "name", "sector", "industry")}
    nifty_r3m = _benchmark_r3m()
    stale_before = latest - timedelta(days=10)                 # stocks that stopped trading are left out
    stocks, groups = [], list(df.groupby("cid", sort=False))
    for i, (cid, g) in enumerate(groups, 1):
        if progress and (i % 100 == 0 or i == len(groups)):
            progress(i, len(groups))
        co = companies.get(cid)
        if not co or len(g) < 30 or g["date"].iloc[-1] < stale_before:
            continue
        try:
            m = stock_metrics(g.reset_index(drop=True), nifty_r3m)
        except Exception:
            logger.exception("analysis: metrics failed for %s", co["code"])
            continue
        stocks.append({"id": cid, "code": co["code"], "name": co["name"], "sector": co["sector"] or "Other",
                       "industry": co["industry"] or "", "etf": co["sector"] == "ETF", **m})
    eq = [s for s in stocks if not s["etf"]]

    def share(cond):
        return round(sum(1 for s in eq if cond(s)) / len(eq) * 100, 1) if eq else None
    breadth = {
        "stocks": len(eq), "advancers": sum(1 for s in eq if (s["r1d"] or 0) > 0),
        "decliners": sum(1 for s in eq if (s["r1d"] or 0) < 0), "unchanged": sum(1 for s in eq if not s["r1d"]),
        "above20": share(lambda s: (s["vs20"] or -1) > 0), "above50": share(lambda s: (s["vs50"] or -1) > 0),
        "above200": share(lambda s: (s["vs200"] or -1) > 0),
        "new_hi": sum(1 for s in eq if s["new_hi"]), "new_lo": sum(1 for s in eq if s["new_lo"]),
        "rsi_avg": _num(np.nanmean([s["rsi"] for s in eq if s["rsi"] is not None]), 1),
        "oversold": sum(1 for s in eq if (s["rsi"] or 50) < 30), "overbought": sum(1 for s in eq if (s["rsi"] or 50) > 70),
        "golden": sum(1 for s in eq if s["cross"] is not None and 1 <= s["cross"] <= 10),
        "death": sum(1 for s in eq if s["cross"] is not None and -10 <= s["cross"] <= -1),
        "nifty_r3m": _num(nifty_r3m),
    }
    sectors = []
    for name, grp in pd.DataFrame(eq).groupby("sector") if eq else []:
        def med(col):
            return _num(grp[col].median())
        sectors.append({"sector": name, "stocks": len(grp), "r1d": med("r1d"), "r1w": med("r1w"), "r1m": med("r1m"),
                        "r3m": med("r3m"), "r1y": med("r1y"), "above50": _num((grp["vs50"] > 0).mean() * 100, 1),
                        "rsi": med("rsi"), "advancers": int((grp["r1d"] > 0).sum()), "decliners": int((grp["r1d"] < 0).sum())})
    sectors.sort(key=lambda s: -(s["r1d"] or 0))
    snap = {"as_of": latest.isoformat(), "built": datetime.now().isoformat(timespec="seconds"),
            "breadth": breadth, "sectors": sectors, "stocks": stocks}
    SNAPSHOT.parent.mkdir(exist_ok=True)
    tmp = SNAPSHOT.with_suffix(".tmp")
    tmp.write_text(json.dumps(snap, separators=(",", ":")), encoding="utf-8")
    tmp.replace(SNAPSHOT)
    return snap


def load_snapshot():
    try:
        return json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None

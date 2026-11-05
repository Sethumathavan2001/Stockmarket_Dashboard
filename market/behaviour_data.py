"""
Data behind the Stock Behaviour page: one compact stock-by-day matrix, built from the stored candles so that any mix of
filters (index, index trend, weekday, expiry, session, sector) can be answered in milliseconds.

The trading days are NIFTY's full sessions since START (Muhurat and other short sessions are left out). For every stock:
  - from DailyData (10 years):  open -> close %, previous close -> close %, day range %
  - from FiveMinuteData (the stored months only): the same move and range for each session in SESSIONS
and the same for every index, plus the index's expiry days (market/index_data.py).

The result is logs/stock_behaviour.npz (rebuildable, not in git). Build it with `python manage.py build_behaviour` or the
page's Rebuild button.
"""
import logging
import os
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.db import connection

from prediction.market_time import IST

from .models import Company, IndexDaily, IndexExpiry, IndexFiveMinute, MarketIndex

logger = logging.getLogger(__name__)

START = date(2016, 1, 1)
MIN_BARS = 40                                   # a full session has 75 five-minute bars; below this it's a special session
ACTIVE_DAYS = 30                                # a stock with no daily candle this recently is left out
TURNOVER_DAYS = 60                              # median close x volume over this many sessions
MAX_MOVE = 40.0                                 # bigger one-day moves are bad ticks or unadjusted corporate actions
GAP_DAYS = 7                                    # no previous-close move across a longer hole in a stock's history
# name, label, first minute (IST, inclusive), end minute (exclusive)
SESSIONS = [("morning", "Morning", 9 * 60 + 15, 11 * 60 + 30),
            ("afternoon", "Afternoon", 11 * 60 + 30, 13 * 60 + 30),
            ("evening", "Evening", 13 * 60 + 30, 15 * 60 + 30)]
WEEKLY, MONTHLY = 1, 2


def path():
    return Path(settings.BASE_DIR) / "logs" / "stock_behaviour.npz"


def trading_days():
    """NIFTY's full sessions since START, oldest first."""
    nifty = MarketIndex.objects.filter(key="NIFTY").first()
    if nifty is None:
        return []
    counts = {}
    for d in IndexFiveMinute.objects.filter(index=nifty, date__gte=START).order_by().values_list("date", flat=True):
        counts[d] = counts.get(d, 0) + 1
    return sorted(d for d, n in counts.items() if n >= MIN_BARS)


def _session_of(minute):
    """The index into SESSIONS for each minute of the day, -1 outside them."""
    out = np.full(len(minute), -1, dtype=np.int8)
    for i, (_, _, a, b) in enumerate(SESSIONS):
        out[(minute >= a) & (minute < b)] = i
    return out


def _sessions(frame):
    """frame: day, minute, open, high, low, close in time order -> per (session, day) move % and range %."""
    frame = frame.assign(session=_session_of(frame["minute"].to_numpy()))
    frame = frame[frame["session"] >= 0]
    g = frame.groupby(["session", "day"], sort=False).agg(
        open=("open", "first"), close=("close", "last"), high=("high", "max"), low=("low", "min"), bars=("open", "size"))
    g = g[(g["open"] > 0) & (g["bars"] >= 6)]
    g["ret"] = (g["close"] / g["open"] - 1) * 100
    g["rng"] = (g["high"] - g["low"]) / g["open"] * 100
    return g[(g["ret"].abs() <= MAX_MOVE) & (g["rng"] <= MAX_MOVE)].reset_index()


def _five_min_frame(rows):
    """rows: (UTC timestamp, open, high, low, close) in any order."""
    f = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
    f["ts"] = pd.to_datetime(f["ts"], utc=True).dt.tz_localize(None) + pd.Timedelta(minutes=330)
    f = f.sort_values("ts")
    f["day"] = f["ts"].dt.normalize()
    f["minute"] = f["ts"].dt.hour * 60 + f["ts"].dt.minute
    return f[["day", "minute", "open", "high", "low", "close"]].astype(
        {"open": float, "high": float, "low": float, "close": float})


def _daily_moves(rows, pos):
    """rows: (date, open, high, low, close, volume) -> axis rows, o2c, c2c, range (all %), median turnover in crore."""
    f = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    f["date"] = pd.to_datetime(f["date"])
    f = f.sort_values("date")
    for c in ("open", "high", "low", "close", "volume"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f[(f["close"] > 0) & (f["open"] > 0)]
    recent = f.tail(TURNOVER_DAYS)
    turnover = float((recent["close"] * recent["volume"]).median() / 1e7) if len(recent) else 0.0
    prev, gap = f["close"].shift(), f["date"].diff().dt.days
    f = f.assign(o2c=(f["close"] / f["open"] - 1) * 100, rng=(f["high"] - f["low"]) / f["open"] * 100,
                 c2c=((f["close"] / prev - 1) * 100).where(gap <= GAP_DAYS))
    f = f[(f["volume"].fillna(0) > 0) & (f["o2c"].abs() <= MAX_MOVE) & (f["rng"] <= MAX_MOVE)]
    f = f.assign(row=f["date"].map(pos)).dropna(subset=["row"])
    f.loc[f["c2c"].abs() > MAX_MOVE, "c2c"] = np.nan
    return f["row"].to_numpy(dtype=np.int64), f["o2c"], f["c2c"], f["rng"], turnover


def _indices(days, pos):
    """Every index on the day axis: moves, range, session moves and ranges, expiry kind."""
    from .index_analysis import expiry_kinds
    indices = list(MarketIndex.objects.order_by("sort"))
    n, count = len(days), len(indices)
    o2c, c2c, rng = (np.full((count, n), np.nan, dtype=np.float32) for _ in range(3))
    sret, srng = (np.full((len(SESSIONS), count, n), np.nan, dtype=np.float32) for _ in range(2))
    expiry = np.zeros((count, n), dtype=np.int8)
    for i, ix in enumerate(indices):
        bars = list(IndexFiveMinute.objects.filter(index=ix, date__gte=START).order_by()
                    .values_list("timestamp", "open", "high", "low", "close"))
        if bars:
            s = _sessions(_five_min_frame(bars))
            rows = s["day"].map(pos)
            keep = rows.notna().to_numpy()
            r, k = rows[keep].to_numpy(dtype=np.int64), s["session"].to_numpy()[keep]
            sret[k, i, r], srng[k, i, r] = s["ret"].to_numpy()[keep], s["rng"].to_numpy()[keep]
        daily = list(IndexDaily.objects.filter(index=ix, date__gte=START - timedelta(days=15)).order_by("date")
                     .values_list("date", "open", "high", "low", "close"))
        prev = None
        for d, o, h, lo, c in daily:
            row = pos.get(pd.Timestamp(d))
            if row is not None and o:
                o2c[i, row], rng[i, row] = (c / o - 1) * 100, (h - lo) / o * 100
                if prev:
                    c2c[i, row] = (c / prev - 1) * 100
            prev = c
        dates = list(IndexExpiry.objects.filter(index=ix).values_list("date", flat=True))
        for d, kind in expiry_kinds(dates).items():
            row = pos.get(pd.Timestamp(d))
            if row is not None:
                expiry[i, row] = MONTHLY if kind == "monthly" else WEEKLY
    return indices, o2c, c2c, rng, sret, srng, expiry


def _fetch(sql, *params):
    with connection.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def build(step=lambda message: None):
    """Rebuild the matrix file. No ORDER BY in the per-company reads: it would make SQLite walk the whole date index."""
    days = trading_days()
    if not days:
        raise ValueError("No NIFTY candles stored - run Update data on Index Analysis first")
    axis = pd.to_datetime(days)
    pos = {d: i for i, d in enumerate(axis)}
    n = len(days)
    step("Reading indices")
    indices, ix_o2c, ix_c2c, ix_rng, ix_sret, ix_srng, ix_expiry = _indices(days, pos)

    newest = _fetch("SELECT MAX(date) FROM market_dailydata")[0][0]
    newest = date.fromisoformat(str(newest)[:10]) if newest else days[-1]
    active = {r[0] for r in _fetch("SELECT DISTINCT company_id FROM market_dailydata WHERE date >= %s",
                                   str(newest - timedelta(days=ACTIVE_DAYS)))}
    companies = [c for c in Company.objects.order_by("code").only("id", "code", "name", "sector") if c.id in active]
    m = len(companies)
    o2c, c2c, rng = (np.full((n, m), np.nan, dtype=np.float32) for _ in range(3))
    turnover = np.zeros(m, dtype=np.float32)
    sessions = []                                                      # (column, frame)
    for col, company in enumerate(companies):
        if col % 100 == 0:
            step(f"Reading stocks {col} of {m}")
        rows = _fetch("SELECT date, open, high, low, close, volume FROM market_dailydata "
                      "WHERE company_id = %s AND date >= %s", company.id, str(START - timedelta(days=15)))
        if rows:
            r, a, b, c, turnover[col] = _daily_moves(rows, pos)
            o2c[r, col], c2c[r, col], rng[r, col] = a.to_numpy(), b.to_numpy(), c.to_numpy()
        bars = _fetch("SELECT timestamp, open, high, low, close FROM market_fiveminutedata WHERE company_id = %s",
                      company.id)
        if bars:
            s = _sessions(_five_min_frame(bars))
            sessions.append((col, s[["session", "day", "ret", "rng"]]))

    seen = sorted({d for _, s in sessions for d in s["day"].unique()})
    rows5 = np.array(sorted(pos[pd.Timestamp(d)] for d in seen if pd.Timestamp(d) in pos), dtype=np.int64)
    pos5 = {axis[r]: k for k, r in enumerate(rows5)}
    sret, srng = (np.full((len(SESSIONS), len(rows5), m), np.nan, dtype=np.float32) for _ in range(2))
    for col, s in sessions:
        rows = s["day"].map(pos5)
        keep = rows.notna().to_numpy()
        r, k = rows[keep].to_numpy(dtype=np.int64), s["session"].to_numpy()[keep]
        sret[k, r, col], srng[k, r, col] = s["ret"].to_numpy()[keep], s["rng"].to_numpy()[keep]

    step("Saving")
    target = path()
    target.parent.mkdir(exist_ok=True)
    tmp = target.with_name(f"{target.stem}.tmp.npz")
    np.savez(tmp, days=axis.to_numpy().astype("datetime64[D]"), rows5=rows5,
             codes=np.array([c.code for c in companies], dtype=str), names=np.array([c.name for c in companies], dtype=str),
             sectors=np.array([c.sector or "" for c in companies], dtype=str), turnover=turnover,
             o2c=o2c, c2c=c2c, rng=rng, sret=sret, srng=srng,
             ix_keys=np.array([i.key for i in indices], dtype=str), ix_names=np.array([i.name for i in indices], dtype=str),
             ix_o2c=ix_o2c, ix_c2c=ix_c2c, ix_rng=ix_rng, ix_sret=ix_sret, ix_srng=ix_srng, ix_expiry=ix_expiry,
             built=np.array(datetime.now(IST).isoformat()))
    os.replace(tmp, target)
    return {"days": n, "stocks": m, "session_days": len(rows5)}


_cache = {"mtime": None, "data": None}
_lock = threading.Lock()
_progress = {"running": False, "step": "", "error": None}


def load():
    """The matrices as a dict, or None before the first build. Reloaded when the file changes."""
    target = path()
    try:
        mtime = target.stat().st_mtime
    except OSError:
        return None
    with _lock:
        if _cache["mtime"] != mtime:
            with np.load(target) as z:
                _cache.update(mtime=mtime, data={k: z[k] for k in z.files})
        return _cache["data"]


def progress():
    with _lock:
        return dict(_progress)


def _set(**kw):
    with _lock:
        _progress.update(kw)


def _run():
    try:
        build(step=lambda message: _set(step=message))
        _set(step="Done")
    except Exception as exc:
        logger.exception("stock behaviour build failed")
        _set(error=f"{type(exc).__name__}: {exc}", step="Failed")
    finally:
        connection.close()
        _set(running=False)


def start_build():
    with _lock:
        if _progress["running"]:
            return False
        _progress.update(running=True, step="Starting", error=None)
    threading.Thread(target=_run, name="stock-behaviour", daemon=True).start()
    return True

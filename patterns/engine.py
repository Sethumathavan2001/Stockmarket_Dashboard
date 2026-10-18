"""
Intraday candle predictor engine (1-minute and 5-minute candles).

  1. Data      5m: the database (FiveMinuteData) + Yahoo for missing recent days and today.
               1m: Yahoo only (it keeps ~30 days of 1-minute candles).
  2. Patterns  ~40 candlestick and intraday-level patterns, detected on completed candles
               (each uses only candles up to its own).
  3. Evidence  every past occurrence of each pattern in THIS stock / interval is measured: the
               move 1..N candles later in the pattern's direction, win rate, edge over the
               stock's normal drift and a t-statistic.  "Reliable" = enough samples, t >= 2 and
               positive in both halves of the history.
  4. Forecast  analog (nearest-neighbour) forecasting: the ~60 past moments whose recent
               candles, VWAP / day / previous-day position, volume and time of day are most
               like "now"; what happened after them gives the predicted candles, ranges and P(up).
  5. Test      the analog forecaster is scored on the most recent unseen days.

Only data BEFORE the selected day is used for evidence and analogs, so a past day can be
replayed honestly from any "predict from" time.  Nothing here writes to the database.
"""
import logging
import math
import os
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
from django.core.cache import cache

from prediction.data import load_historical_candles
from prediction.datasets import _yahoo, _yahoo_frame
from prediction.market_time import IST, get_market_status
from prediction.tickers import get_yahoo_ticker

logger = logging.getLogger(__name__)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))  # skip joblib's wmic CPU probe

SESSION_OPEN, SESSION_END = time(9, 15), time(15, 30)
INTERVALS = {
    "5m": {"minutes": 5, "horizons": (1, 3, 6, 12), "main": 3, "max_ahead": 24, "or_bars": 3, "k": 60},
    "1m": {"minutes": 1, "horizons": (1, 5, 15, 30), "main": 15, "max_ahead": 60, "or_bars": 15, "k": 60},
}
LOOKBACK = 8                      # candles of recent shape in the analog features
MIN_SAMPLES, MIN_T = 30, 2.0      # a "Reliable" pattern


class PredictorError(Exception):
    pass


# ==========================================================================
# data
# ==========================================================================
def _session(df):
    if df.empty:
        return df
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    t = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    keep = (t >= 9 * 60 + 15) & (t < 15 * 60 + 30) & (df["high"] >= df["low"]) & (df["low"] > 0)
    return df[keep].reset_index(drop=True)


def load_candles(company, interval):
    """All available candles for the interval: timestamp (IST) / open / high / low / close / volume."""
    ticker = get_yahoo_ticker(company.code)
    if interval == "5m":
        db = load_historical_candles(company)[["timestamp", "open", "high", "low", "close", "volume"]]
        recent = _yahoo_frame(_yahoo(ticker, cache_seconds=30, period="5d", interval="5m", prepost=False))
        df = pd.concat([db, recent], ignore_index=True) if not recent.empty else db
    else:
        today = datetime.now(IST).date()
        parts = []
        for i in range(4):                            # Yahoo serves 1m history in <= 8-day windows
            end = today + timedelta(days=1) - timedelta(days=7 * i)
            start = end - timedelta(days=7)
            parts.append(_yahoo_frame(_yahoo(ticker, cache_seconds=1800 if i else 30, start=start.isoformat(),
                                             end=end.isoformat(), interval="1m", prepost=False)))
        parts.append(_yahoo_frame(_yahoo(ticker, cache_seconds=20, period="1d", interval="1m", prepost=False)))
        parts = [p for p in parts if not p.empty]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return _session(df)


# ==========================================================================
# per-candle context (all causal)
# ==========================================================================
class Bars:
    def __init__(self, df, interval):
        cfg = INTERVALS[interval]
        self.df = df
        self.interval = interval
        self.o, self.h, self.l, self.c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.v = df["volume"].fillna(0).to_numpy(float)
        self.ts = df["timestamp"].tolist()
        n = self.n = len(df)
        days = df["timestamp"].dt.date
        self.day = days.to_numpy()
        codes, uniq = pd.factorize(days, sort=True)
        self.day_id = codes
        self.days = [d for d in uniq]
        first = pd.Series(np.arange(n)).groupby(codes).transform("min").to_numpy()
        last = pd.Series(np.arange(n)).groupby(codes).transform("max").to_numpy()
        self.pos = np.arange(n) - first
        self.remaining = last - np.arange(n)                     # candles left in the session after this one
        o, h, l, c, v = self.o, self.h, self.l, self.c, self.v
        rng = h - l
        self.rng = rng
        self.body = np.abs(c - o)
        self.avg_rng = pd.Series(rng).rolling(20, min_periods=5).mean().shift(1).to_numpy()
        tp = (h + l + c) / 3
        g = pd.Series(codes)
        cum_pv = pd.Series(tp * v).groupby(g).cumsum().to_numpy()
        cum_v = pd.Series(v).groupby(g).cumsum().to_numpy()
        cum_tp = pd.Series(tp).groupby(g).cumsum().to_numpy()
        self.vwap = np.where(cum_v > 0, cum_pv / np.where(cum_v > 0, cum_v, 1), cum_tp / (self.pos + 1))
        cs = pd.Series(c)
        self.ema9 = cs.ewm(span=9, adjust=False).mean().to_numpy()
        self.ema21 = cs.ewm(span=21, adjust=False).mean().to_numpy()
        d = cs.diff()
        ag = d.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        al = (-d.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        self.rsi = (100 - 100 / (1 + ag / al.replace(0, np.nan))).fillna(50).to_numpy()
        self.ret1 = cs.pct_change().to_numpy()
        self.vol = pd.Series(self.ret1).rolling(60, min_periods=20).std().to_numpy()
        self.day_open = pd.Series(o).groupby(g).transform("first").to_numpy()
        self.hi_before = pd.Series(h).groupby(g).cummax().groupby(g).shift(1).to_numpy()
        self.lo_before = pd.Series(l).groupby(g).cummin().groupby(g).shift(1).to_numpy()
        orb = cfg["or_bars"]
        early = self.pos < orb
        self.or_high = pd.Series(np.where(early, h, np.nan)).groupby(g).transform("max").to_numpy()
        self.or_low = pd.Series(np.where(early, l, np.nan)).groupby(g).transform("min").to_numpy()
        self.or_ready = self.pos >= orb
        daily = pd.DataFrame({"g": codes, "h": h, "l": l, "c": c}).groupby("g").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
        prev = daily.shift(1)
        self.pdh, self.pdl, self.pdc = (prev[k].reindex(codes).to_numpy() for k in ("h", "l", "c"))
        # usual volume of this time slot over earlier sessions (0 = not reported)
        slot = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute).to_numpy()
        vs = pd.Series(np.where(v > 0, v, np.nan))
        slot_avg = vs.groupby(slot).transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean()).to_numpy()
        self.vol_ratio = np.where((v > 0) & (slot_avg > 0), v / np.where(slot_avg > 0, slot_avg, 1), np.nan)
        # forward close-to-close returns inside the session
        self.fwd = {}
        for k in cfg["horizons"]:
            fr = np.full(n, np.nan)
            if n > k:
                same = codes[k:] == codes[:-k]
                fr[:-k] = np.where(same, c[k:] / c[:-k] - 1, np.nan)
            self.fwd[k] = fr


# ==========================================================================
# patterns
# ==========================================================================
PATTERNS = {
    # candlestick - single
    "hammer": ("Hammer", 1, "Candlestick"),
    "hanging_man": ("Hanging man", -1, "Candlestick"),
    "inv_hammer": ("Inverted hammer", 1, "Candlestick"),
    "shooting_star": ("Shooting star", -1, "Candlestick"),
    "dragonfly": ("Dragonfly doji", 1, "Candlestick"),
    "gravestone": ("Gravestone doji", -1, "Candlestick"),
    "bull_marubozu": ("Bullish marubozu", 1, "Candlestick"),
    "bear_marubozu": ("Bearish marubozu", -1, "Candlestick"),
    # candlestick - multi
    "bull_engulfing": ("Bullish engulfing", 1, "Candlestick"),
    "bear_engulfing": ("Bearish engulfing", -1, "Candlestick"),
    "bull_harami": ("Bullish harami", 1, "Candlestick"),
    "bear_harami": ("Bearish harami", -1, "Candlestick"),
    "piercing": ("Piercing line", 1, "Candlestick"),
    "dark_cloud": ("Dark cloud cover", -1, "Candlestick"),
    "tweezer_bottom": ("Tweezer bottom", 1, "Candlestick"),
    "tweezer_top": ("Tweezer top", -1, "Candlestick"),
    "morning_star": ("Morning star", 1, "Candlestick"),
    "evening_star": ("Evening star", -1, "Candlestick"),
    "three_soldiers": ("Three white soldiers", 1, "Candlestick"),
    "three_crows": ("Three black crows", -1, "Candlestick"),
    "three_inside_up": ("Three inside up", 1, "Candlestick"),
    "three_inside_down": ("Three inside down", -1, "Candlestick"),
    "ib_break_up": ("Inside-bar breakout up", 1, "Candlestick"),
    "ib_break_down": ("Inside-bar breakout down", -1, "Candlestick"),
    "outside_up": ("Bullish outside bar", 1, "Candlestick"),
    "outside_down": ("Bearish outside bar", -1, "Candlestick"),
    # intraday levels
    "orb_up": ("Opening-range breakout up", 1, "Intraday level"),
    "orb_down": ("Opening-range breakdown", -1, "Intraday level"),
    "vwap_reclaim": ("VWAP reclaim", 1, "Intraday level"),
    "vwap_reject": ("VWAP rejection", -1, "Intraday level"),
    "pdh_break": ("Previous-day high breakout", 1, "Intraday level"),
    "pdh_reject": ("Previous-day high rejection", -1, "Intraday level"),
    "pdl_break": ("Previous-day low breakdown", -1, "Intraday level"),
    "pdl_reject": ("Previous-day low rejection", 1, "Intraday level"),
    "gap_fade": ("Gap-up fade", -1, "Intraday level"),
    "gap_recover": ("Gap-down recovery", 1, "Intraday level"),
    # volume / momentum
    "new_high_vol": ("New day high on volume", 1, "Volume"),
    "new_low_vol": ("New day low on volume", -1, "Volume"),
    "climax_up": ("Volume climax up", 1, "Volume"),
    "climax_down": ("Volume climax down", -1, "Volume"),
    "nr7_up": ("NR7 squeeze breakout up", 1, "Momentum"),
    "nr7_down": ("NR7 squeeze breakout down", -1, "Momentum"),
    "ema_cross_up": ("EMA 9/21 bullish cross", 1, "Momentum"),
    "ema_cross_down": ("EMA 9/21 bearish cross", -1, "Momentum"),
    "rsi_bounce": ("RSI oversold bounce", 1, "Momentum"),
    "rsi_drop": ("RSI overbought drop", -1, "Momentum"),
}


def detect(b):
    """List of (bar index, pattern key) for every completed candle."""
    out = []
    o, h, l, c, rng, body = b.o, b.h, b.l, b.c, b.rng, b.body
    for k in range(b.n):
        p = b.pos[k]
        r = rng[k]
        if r <= 0:
            continue
        ar = b.avg_rng[k]
        up_w, lo_w = h[k] - max(o[k], c[k]), min(o[k], c[k]) - l[k]
        bull, bear = c[k] > o[k], c[k] < o[k]
        hits = []
        fell = p >= 3 and c[k - 1] < c[k - 3]
        rose = p >= 3 and c[k - 1] > c[k - 3]
        small = body[k] <= 0.35 * r
        if small and lo_w >= 2 * body[k] and up_w <= 0.25 * r:
            hits.append("hammer" if fell else "hanging_man" if rose else None)
        if small and up_w >= 2 * body[k] and lo_w <= 0.25 * r:
            hits.append("inv_hammer" if fell else "shooting_star" if rose else None)
        if body[k] <= 0.1 * r and ar and r >= 0.8 * ar:
            if lo_w >= 0.7 * r and fell:
                hits.append("dragonfly")
            if up_w >= 0.7 * r and rose:
                hits.append("gravestone")
        if ar and body[k] >= 0.85 * r and r >= 1.3 * ar:
            hits.append("bull_marubozu" if bull else "bear_marubozu" if bear else None)
        if p >= 1:
            po, pc, ph, pl = o[k - 1], c[k - 1], h[k - 1], l[k - 1]
            pbull, pbear = pc > po, pc < po
            pbody = abs(pc - po)
            if pbear and bull and c[k] >= po and o[k] <= pc and body[k] > pbody:
                hits.append("bull_engulfing")
            if pbull and bear and c[k] <= po and o[k] >= pc and body[k] > pbody:
                hits.append("bear_engulfing")
            if pbear and bull and pbody > 0 and max(o[k], c[k]) < po and min(o[k], c[k]) > pc and body[k] < 0.6 * pbody and fell:
                hits.append("bull_harami")
            if pbull and bear and pbody > 0 and max(o[k], c[k]) < pc and min(o[k], c[k]) > po and body[k] < 0.6 * pbody and rose:
                hits.append("bear_harami")
            if pbear and bull and o[k] <= pc and c[k] > (po + pc) / 2 and c[k] < po:
                hits.append("piercing")
            if pbull and bear and o[k] >= pc and c[k] < (po + pc) / 2 and c[k] > po:
                hits.append("dark_cloud")
            tol = 0.1 * (ar or r)
            if abs(l[k] - pl) <= tol and pbear and bull and fell:
                hits.append("tweezer_bottom")
            if abs(h[k] - ph) <= tol and pbull and bear and rose:
                hits.append("tweezer_top")
            if h[k] > ph and l[k] < pl and ar and r >= 1.2 * ar:
                hits.append("outside_up" if bull and c[k] > ph else "outside_down" if bear and c[k] < pl else None)
            # intraday: VWAP crosses, EMA cross, RSI
            vw, pvw = b.vwap[k], b.vwap[k - 1]
            if pc <= pvw and c[k] > vw and bull:
                hits.append("vwap_reclaim")
            if pc >= pvw and c[k] < vw and bear:
                hits.append("vwap_reject")
            if b.ema9[k - 1] <= b.ema21[k - 1] and b.ema9[k] > b.ema21[k]:
                hits.append("ema_cross_up")
            if b.ema9[k - 1] >= b.ema21[k - 1] and b.ema9[k] < b.ema21[k]:
                hits.append("ema_cross_down")
            if b.rsi[k - 1] < 30 <= b.rsi[k]:
                hits.append("rsi_bounce")
            if b.rsi[k - 1] > 70 >= b.rsi[k]:
                hits.append("rsi_drop")
        if p >= 2:
            o2, c2, o1, c1 = o[k - 2], c[k - 2], o[k - 1], c[k - 1]
            b2, b1 = abs(c2 - o2), abs(c1 - o1)
            if c2 < o2 and b1 <= 0.3 * max(b2, 1e-12) and bull and c[k] > (o2 + c2) / 2 and b2 >= 0.5 * (h[k - 2] - l[k - 2]):
                hits.append("morning_star")
            if c2 > o2 and b1 <= 0.3 * max(b2, 1e-12) and bear and c[k] < (o2 + c2) / 2 and b2 >= 0.5 * (h[k - 2] - l[k - 2]):
                hits.append("evening_star")
            if c2 > o2 and c1 > o1 and bull and c[k] > c1 > c2 and o1 > o2 and o[k] > o1 and body[k] >= 0.5 * r:
                hits.append("three_soldiers")
            if c2 < o2 and c1 < o1 and bear and c[k] < c1 < c2 and o1 < o2 and o[k] < o1 and body[k] >= 0.5 * r:
                hits.append("three_crows")
            if c2 < o2 and c1 > o1 and max(o1, c1) <= o2 and min(o1, c1) >= c2 and bull and c[k] > o2:
                hits.append("three_inside_up")
            if c2 > o2 and c1 < o1 and max(o1, c1) <= c2 and min(o1, c1) >= o2 and bear and c[k] < o2:
                hits.append("three_inside_down")
            if h[k - 1] <= h[k - 2] and l[k - 1] >= l[k - 2]:            # k-1 is an inside bar
                if c[k] > h[k - 1] and bull:
                    hits.append("ib_break_up")
                elif c[k] < l[k - 1] and bear:
                    hits.append("ib_break_down")
        if p >= 7:
            prev_r = rng[k - 7:k]
            if rng[k - 1] == prev_r.min():                                 # k-1 was the narrowest of 7
                if c[k] > h[k - 1]:
                    hits.append("nr7_up")
                elif c[k] < l[k - 1]:
                    hits.append("nr7_down")
        # opening range (first 15 minutes)
        if b.or_ready[k] and p >= 1:
            if c[k] > b.or_high[k] >= c[k - 1]:
                hits.append("orb_up")
            if c[k] < b.or_low[k] <= c[k - 1]:
                hits.append("orb_down")
        # previous day's levels
        pdh, pdl, pdc = b.pdh[k], b.pdl[k], b.pdc[k]
        if not math.isnan(pdh) and p >= 1:
            if c[k] > pdh >= c[k - 1]:
                hits.append("pdh_break")
            if h[k] > pdh and c[k] < pdh and bear:
                hits.append("pdh_reject")
            if c[k] < pdl <= c[k - 1]:
                hits.append("pdl_break")
            if l[k] < pdl and c[k] > pdl and bull:
                hits.append("pdl_reject")
        if p == 0 and not math.isnan(pdc):
            gap = o[k] / pdc - 1
            if gap >= 0.005 and bear:
                hits.append("gap_fade")
            if gap <= -0.005 and bull:
                hits.append("gap_recover")
        # day high / low and volume
        vr = b.vol_ratio[k]
        if p >= 1 and not math.isnan(vr):
            if vr >= 1.5 and c[k] > b.hi_before[k] and bull:
                hits.append("new_high_vol")
            if vr >= 1.5 and c[k] < b.lo_before[k] and bear:
                hits.append("new_low_vol")
            if vr >= 3 and ar and r >= 1.5 * ar:
                hits.append("climax_up" if bull else "climax_down" if bear else None)
        for key in hits:
            if key:
                out.append((k, key))
    return out


def pattern_stats(b, detections, hist_mask):
    """Per pattern: the move 1..N candles later in its direction, over the history bars only."""
    cfg = INTERVALS[b.interval]
    by = {}
    for k, key in detections:
        if hist_mask[k]:
            by.setdefault(key, []).append(k)
    mid_day = np.median(b.day_id[hist_mask]) if hist_mask.any() else 0
    out = {}
    for key, (label, side, family) in PATTERNS.items():
        ks = np.array(by.get(key, []), dtype=int)
        row = {"key": key, "name": label, "direction": "Bullish" if side > 0 else "Bearish", "family": family,
               "samples": int(len(ks)), "horizons": {}}
        for hz in cfg["horizons"]:
            drift = np.nanmean(b.fwd[hz][hist_mask]) if hist_mask.any() else 0.0
            vals = b.fwd[hz][ks] * side if len(ks) else np.array([])
            ok = ~np.isnan(vals)
            vals, kk = vals[ok], ks[ok]
            n = len(vals)
            if n:
                edge = vals - side * drift
                sd = edge.std(ddof=1) if n > 1 else 0.0
                first = vals[b.day_id[kk] < mid_day]
                second = vals[b.day_id[kk] >= mid_day]
                row["horizons"][hz] = {
                    "n": n, "avg_pct": float(vals.mean() * 100), "win_pct": float((vals > 0).mean() * 100),
                    "edge_pct": float(edge.mean() * 100), "t": float(edge.mean() / (sd / math.sqrt(n))) if sd > 0 else 0.0,
                    "h1_pct": float(first.mean() * 100) if len(first) else None,
                    "h2_pct": float(second.mean() * 100) if len(second) else None,
                }
        m = row["horizons"].get(cfg["main"])
        row["reliable"] = bool(m and m["n"] >= MIN_SAMPLES and m["t"] >= MIN_T and (m["h1_pct"] or 0) > 0
                               and (m["h2_pct"] or 0) > 0)
        out[key] = row
    return out


# ==========================================================================
# analog (nearest-neighbour) forecasting
# ==========================================================================
def features(b):
    """One row per candle: recent shape (volatility-scaled), VWAP / day / previous-day position, volume, time."""
    n = b.n
    vol = np.where(np.isnan(b.vol) | (b.vol <= 0), np.nanmedian(b.vol[b.vol > 0]) if (b.vol > 0).any() else 1e-3, b.vol)
    cols = []
    for i in range(LOOKBACK):
        r = np.full(n, 0.0)
        if n > i:
            sh = np.roll(b.ret1, i)
            valid = b.pos >= i + 1
            r = np.where(valid, sh, 0.0)
        cols.append(np.clip(np.nan_to_num(r) / vol, -6, 6))
    c = b.c
    day_ret = c / b.day_open - 1
    cols.append(np.clip(day_ret / (vol * np.sqrt(b.pos + 1)), -6, 6))
    cols.append(np.clip((c / b.vwap - 1) / vol, -10, 10))
    cols.append(np.clip(np.nan_to_num((c / b.pdh - 1) / vol), -20, 20))
    cols.append(np.clip(np.nan_to_num((c / b.pdl - 1) / vol), -20, 20))
    run_hi = np.fmax(np.nan_to_num(b.hi_before, nan=b.h[0]), b.h)
    run_lo = np.fmin(np.nan_to_num(b.lo_before, nan=b.l[0]), b.l)
    cols.append(np.nan_to_num((c - run_lo) / np.where(run_hi - run_lo > 0, run_hi - run_lo, 1), nan=0.5))
    vr = pd.Series(np.nan_to_num(b.vol_ratio, nan=1.0)).rolling(3, min_periods=1).mean().to_numpy()
    cols.append(np.log(np.clip(vr, 0.05, 20)))
    cols.append(b.pos * INTERVALS[b.interval]["minutes"] / 375.0 * 3)       # time of day (weighted up)
    cols.append(np.where(b.rng > 0, (c - b.o) / np.where(b.rng > 0, b.rng, 1), 0))
    return np.column_stack(cols)


def _neighbors(X, cand, query, k):
    from sklearn.neighbors import NearestNeighbors
    mu, sd = X[cand].mean(axis=0), X[cand].std(axis=0)
    sd[sd == 0] = 1
    nn = NearestNeighbors(n_neighbors=min(k, len(cand))).fit((X[cand] - mu) / sd)
    _, idx = nn.kneighbors((np.atleast_2d(query) - mu) / sd)
    return cand[idx]


def analog_paths(b, nb, ahead):
    """Relative future candles of the neighbour bars: arrays (k, ahead) for open/high/low/close."""
    base = b.c[nb][:, None]
    steps = nb[:, None] + np.arange(1, ahead + 1)[None, :]
    return (b.o[steps] / base - 1, b.h[steps] / base - 1, b.l[steps] / base - 1, b.c[steps] / base - 1)


def backtest_analogs(b, X, hist_mask):
    """Score the analog forecast on the most recent history days it never saw."""
    cfg = INTERVALS[b.interval]
    hz_max = max(cfg["horizons"])
    hist_days = sorted(set(b.day_id[hist_mask]))
    if len(hist_days) < 8:
        return None
    n_test = max(3, len(hist_days) // 5)
    test_days = set(hist_days[-n_test:])
    train = np.flatnonzero(hist_mask & ~np.isin(b.day_id, list(test_days)) & (b.remaining >= hz_max) & (b.pos >= 1))
    tests = np.flatnonzero(np.isin(b.day_id, list(test_days)) & (b.remaining >= hz_max) & (b.pos >= 1))
    step = max(1, len(tests) // 600)
    tests = tests[::step]
    if len(train) < 500 or not len(tests):
        return None
    from sklearn.neighbors import NearestNeighbors
    mu, sd = X[train].mean(axis=0), X[train].std(axis=0)
    sd[sd == 0] = 1
    nn = NearestNeighbors(n_neighbors=cfg["k"]).fit((X[train] - mu) / sd)
    _, idx = nn.kneighbors((X[tests] - mu) / sd)
    nb = train[idx]                                              # (tests, k)
    out = {"test_days": n_test, "points": int(len(tests)), "horizons": {}}
    for hz in cfg["horizons"]:
        fut = b.c[np.minimum(nb + hz, b.n - 1)] / b.c[nb] - 1
        pred = np.median(fut, axis=1)
        p_up = (fut > 0).mean(axis=1)
        # the call = more neighbours went up than down (the median is often exactly 0 with price ticks);
        # a tie or an unchanged price is no call / no outcome, not a miss
        call = np.sign(p_up - (fut < 0).mean(axis=1))
        actual = b.c[tests + hz] / b.c[tests] - 1
        nz = (actual != 0) & (call != 0)
        acc = float((call[nz] == np.sign(actual[nz])).mean()) if nz.any() else None
        up_rate = float((actual[nz] > 0).mean()) if nz.any() else 0.5
        conf = (p_up >= 0.6) | (p_up <= 0.4)
        conf_hit = (np.sign(p_up[conf & nz] - 0.5) == np.sign(actual[conf & nz]))
        ic = float(np.corrcoef(pred, actual)[0, 1]) if np.std(pred) > 0 and np.std(actual) > 0 else 0.0
        out["horizons"][hz] = {"accuracy": acc, "baseline": max(up_rate, 1 - up_rate), "ic": ic,
                               "confident_n": int(conf_hit.size),
                               "confident_acc": float(conf_hit.mean()) if conf_hit.size else None}
    return out


# ==========================================================================
# public entry point
# ==========================================================================
NEWS_TILT = 0.35        # strongest news moves the median by 35 % of the half-width of the 10-90 % range
NEWS_WIDEN = 0.5        # and widens the range by up to 50 % (news days move more)


def apply_news(forecast, signal, pc, ph, pl, price, interval, backtest):
    """
    Tilt the analog forecast by the news signal (bias -1..1) and widen it by its intensity (0..1).
    Every analog's path is shifted by bias * NEWS_TILT * (that step's 10-90 % half-width), so the shift
    is sized to how much this stock normally moves over that many candles.  Rule of thumb, not tested.
    """
    cfg = INTERVALS[interval]
    bias, widen = float(signal["bias"]), 1 + NEWS_WIDEN * float(signal["intensity"])
    ahead = pc.shape[1]
    q10, q50, q90 = (np.percentile(pc, q, axis=0) for q in (10, 50, 90))
    tilt = bias * NEWS_TILT * (q90 - q10) / 2                    # per step, as a return
    adj = q50 + tilt
    hi = np.maximum(np.median(ph, axis=0), q50)
    lo = np.minimum(np.median(pl, axis=0), q50)
    candles, band, prev = [], [], 0.0
    for i in range(ahead):
        c = adj[i]
        h = max(prev, c, c + (hi[i] - q50[i]) * widen)
        l_ = min(prev, c, c + (lo[i] - q50[i]) * widen)
        x = forecast["candles"][i]["x"]
        candles.append({"x": x, "o": price * (1 + prev), "h": price * (1 + h), "l": price * (1 + l_), "c": price * (1 + c)})
        band.append({"x": x, "p10": price * (1 + c - (q50[i] - q10[i]) * widen),
                     "p25": price * (1 + c - (q50[i] - np.percentile(pc[:, i], 25)) * widen),
                     "p75": price * (1 + c + (np.percentile(pc[:, i], 75) - q50[i]) * widen),
                     "p90": price * (1 + c + (q90[i] - q50[i]) * widen)})
        prev = c
    horizons = []
    for h0 in forecast["horizons"]:
        i = h0["bars"] - 1
        col = pc[:, i] + tilt[i]
        med = float(np.median(col))
        horizons.append({**h0, "price": price * (1 + med), "ret_pct": med * 100, "p_up": float((col > 0).mean()),
                         "low": price * (1 + med - (q50[i] - q10[i]) * widen),
                         "high": price * (1 + med + (q90[i] - q50[i]) * widen),
                         "base_ret_pct": h0["ret_pct"], "base_p_up": h0["p_up"], "news_tilt_pct": float(tilt[i] * 100)})
    main = next((x for x in horizons if x["bars"] == cfg["main"]), horizons[-1] if horizons else None)
    return {**forecast, "candles": candles, "band": band, "horizons": horizons, "main": main,
            "base": {"candles": [{"x": c["x"], "c": c["c"]} for c in forecast["candles"]], "main": forecast["main"]},
            "news": {**signal, "applied": True, "tilt_main_pct": main["news_tilt_pct"] if main else 0.0,
                     "widen": round(widen, 2)}}


def _label(interval, bars):
    return f"{bars * INTERVALS[interval]['minutes']} min"


def predict(company, interval="5m", day=None, asof=None, news=None):
    """news: optional callable(company) -> news signal; used only for today's live session."""
    if interval not in INTERVALS:
        raise PredictorError("Interval must be 1m or 5m.")
    cfg = INTERVALS[interval]
    df = load_candles(company, interval)
    if df.empty or len(df) < 500:
        raise PredictorError(f"Not enough {interval} candles for {company.code}"
                             + (" - Yahoo keeps 1-minute data for ~30 days only; try 5m." if interval == "1m" else "."))
    b = Bars(df, interval)
    days_available = b.days[::-1][:40]
    day = day or b.days[-1]
    if day not in b.days:
        raise PredictorError(f"No {interval} candles for {company.code} on {day.isoformat()}.")
    di = b.days.index(day)
    day_idx = np.flatnonzero(b.day_id == di)
    hist_mask = b.day_id < di                                    # evidence and analogs: earlier days only
    if hist_mask.sum() < 300:
        raise PredictorError(f"Not enough history before {day.isoformat()} for {interval}.")

    # "predict from": the last candle at or before `asof` (default: the day's latest candle)
    k_now = day_idx[-1]
    if asof is not None:
        before = [k for k in day_idx if b.ts[k].time() <= asof]
        if before:
            k_now = before[-1]
    is_latest = k_now == day_idx[-1]
    news_status = None
    if news is not None and not (day == datetime.now(IST).date() and is_latest):
        news_status = "News is used only for today's live session (latest candle) - not for past sessions or earlier times."
    detections = detect(b)

    key = f"candlepred:{company.pk}:{interval}:{day.isoformat()}:{int(hist_mask.sum())}"
    cached = cache.get(key)
    X = features(b)
    if cached is None:
        cached = {"stats": pattern_stats(b, detections, hist_mask), "backtest": backtest_analogs(b, X, hist_mask)}
        cache.set(key, cached, 1800)
    stats, bt = cached["stats"], cached["backtest"]

    # ---- forecast from k_now
    # candles left until 15:30 by the clock (today's session is still being written)
    session_end = datetime.combine(day, SESSION_END, tzinfo=IST)
    left = int((session_end - (b.ts[k_now] + timedelta(minutes=cfg["minutes"]))).total_seconds() // (cfg["minutes"] * 60))
    ahead = int(min(cfg["max_ahead"], max(left, 0)))
    forecast = None
    if ahead >= 1:
        cand = np.flatnonzero(hist_mask & (b.remaining >= ahead) & (b.pos >= 1))
        if len(cand) >= cfg["k"]:
            nb = _neighbors(X, cand, X[k_now], cfg["k"])[0]
            po, ph, pl, pc = analog_paths(b, nb, ahead)
            price = b.c[k_now]
            step = timedelta(minutes=cfg["minutes"])
            t0 = b.ts[k_now]
            candles, band = [], []
            prev = 0.0
            for i in range(ahead):
                cr = float(np.median(pc[:, i]))
                hr = max(prev, cr, float(np.median(ph[:, i])))
                lr = min(prev, cr, float(np.median(pl[:, i])))
                t = t0 + step * (i + 1)
                candles.append({"x": t.isoformat(), "o": price * (1 + prev), "h": price * (1 + hr),
                                "l": price * (1 + lr), "c": price * (1 + cr)})
                q10, q25, q75, q90 = np.percentile(pc[:, i], [10, 25, 75, 90])
                band.append({"x": t.isoformat(), "p10": price * (1 + q10), "p25": price * (1 + q25),
                             "p75": price * (1 + q75), "p90": price * (1 + q90)})
                prev = cr
            horizons = []
            for hz in cfg["horizons"]:
                if hz > ahead:
                    continue
                col = pc[:, hz - 1]
                med = float(np.median(col))
                q10, q90 = np.percentile(col, [10, 90])
                tested = (bt or {}).get("horizons", {}).get(hz, {})
                horizons.append({"bars": hz, "label": _label(interval, hz), "time": (t0 + step * hz).isoformat(),
                                 "price": price * (1 + med), "ret_pct": med * 100, "p_up": float((col > 0).mean()),
                                 "low": price * (1 + q10), "high": price * (1 + q90),
                                 "accuracy": tested.get("accuracy"), "baseline": tested.get("baseline"),
                                 "ic": tested.get("ic")})
            main = next((x for x in horizons if x["bars"] == cfg["main"]), horizons[-1] if horizons else None)
            forecast = {"ahead": ahead, "neighbors": int(len(nb)), "candles": candles, "band": band,
                        "horizons": horizons, "main": main,
                        "analogs": [{"day": b.days[b.day_id[j]].isoformat(), "time": b.ts[j].strftime("%H:%M"),
                                     "move_pct": float(pc[i_, min(cfg["main"], ahead) - 1] * 100)}
                                    for i_, j in enumerate(nb[:8])]}
            live_session = day == datetime.now(IST).date() and is_latest
            if news is not None and live_session:
                try:
                    forecast = apply_news(forecast, news(company), pc, ph, pl, price, interval, bt)
                except Exception as exc:            # news sources down: keep the plain forecast
                    logger.warning("News signal failed for %s: %s", company.code, exc)
                    news_status = f"News unavailable ({exc}) - showing the forecast without news."

    # ---- patterns on the selected day (up to "now") with what happened next
    det_day = [(k, key_) for k, key_ in detections if b.day_id[k] == di and k <= k_now]
    markers, log = [], []
    for k, key_ in det_day:
        label, side, family = PATTERNS[key_]
        st = stats[key_]
        m = st["horizons"].get(cfg["main"], {})
        outcome = None
        hz = cfg["main"]
        if k + hz <= k_now and b.day_id[k + hz] == di:              # already known by "now"
            outcome = float((b.c[k + hz] / b.c[k] - 1) * side * 100)
        elif not is_latest and k + hz < b.n and b.day_id[k + hz] == di:
            outcome = float((b.c[k + hz] / b.c[k] - 1) * side * 100)
        markers.append({"x": b.ts[k].isoformat(), "key": key_, "name": label, "side": side, "reliable": st["reliable"]})
        log.append({"x": b.ts[k].isoformat(), "name": label, "direction": "Bullish" if side > 0 else "Bearish",
                    "family": family, "price": float(b.c[k]), "reliable": st["reliable"],
                    "hist_win_pct": m.get("win_pct"), "hist_edge_pct": m.get("edge_pct"), "samples": m.get("n", 0),
                    "outcome_pct": outcome})
    recent_from = max(day_idx[0], k_now - 2)
    active = [dict(stats[key_], x=b.ts[k].isoformat()) for k, key_ in det_day if k >= recent_from]

    # pattern verdict now: reliable active patterns vote with their historical edge
    vote = sum((1 if s["direction"] == "Bullish" else -1) * max(s["horizons"].get(cfg["main"], {}).get("edge_pct", 0), 0)
               for s in active if s["reliable"])

    day_rows = [{"x": b.ts[k].isoformat(), "o": b.o[k], "h": b.h[k], "l": b.l[k], "c": b.c[k], "v": b.v[k],
                 "future": bool(k > k_now)} for k in day_idx]
    vw = [{"x": b.ts[k].isoformat(), "v": float(b.vwap[k])} for k in day_idx if k <= k_now]
    k0 = day_idx[0]
    levels = {n_: (None if math.isnan(v_) else float(v_)) for n_, v_ in
              (("pdh", b.pdh[k0]), ("pdl", b.pdl[k0]), ("pdc", b.pdc[k0]),
               ("or_high", b.or_high[k0]), ("or_low", b.or_low[k0]))}
    hist_days = sorted(set(b.day_id[hist_mask]))
    library = sorted(stats.values(), key=lambda s: (not s["reliable"], -(s["horizons"].get(cfg["main"], {}).get("t", -99))))
    today = datetime.now(IST).date()
    return {
        "company": company.code, "name": company.name, "interval": interval, "day": day.isoformat(),
        "days_available": [d.isoformat() for d in days_available],
        "asof": b.ts[k_now].isoformat(), "is_latest": bool(is_latest),
        "is_live": bool(day == today and is_latest and get_market_status() == "OPEN"),
        "price": float(b.c[k_now]), "candles": day_rows, "vwap": vw, "levels": levels,
        "markers": markers, "log": log[::-1], "active": active, "pattern_vote": vote,
        "library": library, "forecast": forecast, "backtest": bt,
        "main_label": _label(interval, cfg["main"]),
        "news_status": news_status,
        "horizon_labels": {str(hz): _label(interval, hz) for hz in cfg["horizons"]},
        "history": {"days": len(hist_days), "bars": int(hist_mask.sum()),
                    "from": b.days[hist_days[0]].isoformat() if hist_days else None,
                    "to": b.days[hist_days[-1]].isoformat() if hist_days else None},
    }

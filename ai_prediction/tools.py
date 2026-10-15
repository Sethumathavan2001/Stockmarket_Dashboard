"""
Tools Claude can call while it analyses a stock or picks intraday trades.

Every tool reads this app's own data (the database, topped up from Yahoo where the
database lags) and returns compact JSON.  They run on the Django server, one at a time,
behind DATA_LOCK, because yfinance must never be called in parallel.

  get_stock_profile        name / sector / industry / 52-week numbers from the Company row
  get_daily_history        daily OHLCV + indicators, pivots, CPR, NR7 / inside-day flags
  get_intraday_candles     5-minute candles of the last 1-5 sessions incl. today (VWAP, opening range)
  get_live_quote           today's price, range, change and VWAP so far
  get_candle_pattern_signals  the Candle Predictor engine: active patterns tested on this stock + analog forecast
  get_ml_forecast          the Stock Prediction model zoo forecast, if this stock has trained models
  get_market_context       NIFTY 50 / BANK NIFTY / INDIA VIX daily + today's NIFTY intraday
  screen_stocks            rank the whole database for intraday candidates (picks only)

Two "submit" tools carry the final answer; their input is validated here before it is saved.
"""
import math
import threading
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from django.core.cache import cache
from django.db.models import Max

from market.models import Company, DailyData
from prediction.market_time import IST, get_market_status

DATA_LOCK = threading.Lock()          # yfinance: strictly one call at a time across all AI jobs
MAX_DAILY_ROWS = 250
MAX_INTRADAY_DAYS = 5
SCREEN_CACHE_SECONDS = 1800


class ToolError(Exception):
    """A tool could not answer; the message goes back to Claude as an error result."""


# ==========================================================================
# helpers
# ==========================================================================
def _num(x, nd=2):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


def find_company(code):
    code = (code or "").strip().upper()
    if not code:
        raise ToolError("company_code is required.")
    for c in (code, f"{code}.NS", code.removesuffix(".NS")):
        company = Company.objects.filter(code__iexact=c).first()
        if company:
            return company
    raise ToolError(f"{code} is not in this app's company list. Use codes exactly as listed, e.g. RELIANCE.NS.")


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _atr(df, n=14):
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def daily_indicators(df):
    """Indicators on completed daily candles (date/open/high/low/close/volume, oldest first)."""
    if len(df) < 2:
        return {}
    c, last = df["close"], df.iloc[-1]
    atr = _atr(df)
    rng = df["high"] - df["low"]
    hi, lo, cl = float(last["high"]), float(last["low"]), float(last["close"])
    p = (hi + lo + cl) / 3
    bc = (hi + lo) / 2
    tc = 2 * p - bc
    vol20 = df["volume"].tail(21).iloc[:-1].mean() if len(df) > 1 else np.nan
    year = df.tail(250)

    def ret(n):
        return _num((c.iloc[-1] / c.iloc[-1 - n] - 1) * 100) if len(c) > n else None

    def sma(n):
        return _num(c.tail(n).mean()) if len(c) >= n else None

    return {
        "last_day": str(last["date"]), "close": _num(cl),
        "change_1d_pct": ret(1), "change_5d_pct": ret(5), "change_20d_pct": ret(20), "change_60d_pct": ret(60),
        "sma20": sma(20), "sma50": sma(50), "sma200": sma(200),
        "ema9": _num(c.ewm(span=9, adjust=False).mean().iloc[-1]), "ema21": _num(c.ewm(span=21, adjust=False).mean().iloc[-1]),
        "rsi14": _num(_rsi(c).iloc[-1], 1),
        "atr14": _num(atr.iloc[-1]), "atr14_pct": _num(atr.iloc[-1] / cl * 100),
        "avg_volume_20d": _num(vol20, 0), "last_volume_vs_20d": _num(last["volume"] / vol20 if vol20 else None),
        "high_20d": _num(df["high"].tail(20).max()), "low_20d": _num(df["low"].tail(20).min()),
        "high_52w": _num(year["high"].max()), "low_52w": _num(year["low"].min()),
        "from_52w_high_pct": _num((cl / year["high"].max() - 1) * 100),
        "inside_day": bool(hi <= df["high"].iloc[-2] and lo >= df["low"].iloc[-2]),
        "nr7": bool(len(rng) >= 7 and rng.iloc[-1] <= rng.tail(7).min()),
        "pivots_for_next_session": {
            "P": _num(p), "R1": _num(2 * p - lo), "R2": _num(p + hi - lo), "R3": _num(hi + 2 * (p - lo)),
            "S1": _num(2 * p - hi), "S2": _num(p - (hi - lo)), "S3": _num(lo - 2 * (hi - p)),
            "CPR_top": _num(max(bc, tc)), "CPR_bottom": _num(min(bc, tc)),
            "CPR_width_pct": _num(abs(tc - bc) / p * 100, 3),
            "PDH": _num(hi), "PDL": _num(lo), "PDC": _num(cl),
        },
    }


def _session_summary(day_df):
    tp = (day_df["high"] + day_df["low"] + day_df["close"]) / 3
    vol = day_df["volume"].fillna(0)
    vwap = (tp * vol).cumsum() / vol.cumsum().replace(0, np.nan)
    first3 = day_df.head(3)
    return {
        "open": _num(day_df["open"].iloc[0]), "high": _num(day_df["high"].max()), "low": _num(day_df["low"].min()),
        "close_or_last": _num(day_df["close"].iloc[-1]), "volume": int(vol.sum()),
        "vwap": _num(vwap.iloc[-1]) if vol.sum() > 0 else None,
        "opening_range_15m": {"high": _num(first3["high"].max()), "low": _num(first3["low"].min())},
        "candles": len(day_df), "last_candle": day_df["timestamp"].iloc[-1].strftime("%H:%M"),
    }


# ==========================================================================
# data tools
# ==========================================================================
def get_stock_profile(company_code):
    c = find_company(company_code)
    last = DailyData.objects.filter(company=c).aggregate(Max("date"))["date__max"]
    return {
        "company_code": c.code, "name": c.name, "exchange": c.exchange, "sector": c.sector, "industry": c.industry,
        "latest_price_in_db": _num(c.latest_price), "db_daily_data_until": str(last) if last else None,
        "note": "52-week and other price levels: use get_daily_history.",
    }


def get_daily_history(company_code, days=60):
    from prediction.datasets import load_daily

    c = find_company(company_code)
    days = max(5, min(int(days or 60), MAX_DAILY_ROWS))
    df = load_daily(c, live=True)                    # database + Yahoo for the days the database lacks
    if df.empty:
        raise ToolError(f"No daily candles for {c.code}.")
    tail = df.tail(days)
    return {
        "company_code": c.code, "completed_days_only": True, "rows_total_available": len(df),
        "indicators": daily_indicators(df),
        "columns": ["date", "open", "high", "low", "close", "volume"],
        "rows": [[str(r.date), _num(r.open), _num(r.high), _num(r.low), _num(r.close), int(r.volume)]
                 for r in tail.itertuples(index=False)],
    }


def get_intraday_candles(company_code, sessions=2):
    from patterns.engine import load_candles

    c = find_company(company_code)
    sessions = max(1, min(int(sessions or 2), MAX_INTRADAY_DAYS))
    df = load_candles(c, "5m")                       # database + Yahoo (last 5 days incl. today)
    if df.empty:
        raise ToolError(f"No 5-minute candles for {c.code}.")
    df = df.copy()
    df["day"] = df["timestamp"].dt.date
    days = sorted(df["day"].unique())[-sessions:]
    today = datetime.now(IST).date()
    out = []
    for d in days:
        part = df[df["day"] == d]
        out.append({
            "date": str(d), "is_today": d == today, "summary": _session_summary(part),
            "columns": ["time", "open", "high", "low", "close", "volume"],
            "rows": [[t.strftime("%H:%M"), _num(o), _num(h), _num(l), _num(cl), int(v or 0)]
                     for t, o, h, l, cl, v in zip(part["timestamp"], part["open"], part["high"], part["low"],
                                                  part["close"], part["volume"])],
        })
    return {"company_code": c.code, "interval": "5m", "timezone": "IST", "market_status": get_market_status(),
            "note": "Candles of today are still forming while the market is open; the last row may be partial.",
            "sessions": out}


def get_live_quote(company_code):
    from paper_trading.services import fetch_live_today, live_previous_day

    c = find_company(company_code)
    df = fetch_live_today(c)
    prev = live_previous_day(c) or {}
    pdc = prev.get("close")
    out = {"company_code": c.code, "market_status": get_market_status(), "previous_day": {
        k: (_num(v) if k != "date" else str(v)) for k, v in prev.items() if k in ("date", "open", "high", "low", "close")}}
    if df.empty:
        out["note"] = "No candles today yet (market closed, holiday or Yahoo has nothing)."
        return out
    vol = df["volume"].fillna(0) if "volume" in df else pd.Series([0] * len(df))
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ltp = float(df["close"].iloc[-1])
    out.update({
        "ltp": _num(ltp), "last_candle_time": df["timestamp"].iloc[-1].strftime("%H:%M"),
        "day_open": _num(df["open"].iloc[0]), "day_high": _num(df["high"].max()), "day_low": _num(df["low"].min()),
        "change_pct": _num((ltp / pdc - 1) * 100) if pdc else None,
        "gap_pct": _num((df["open"].iloc[0] / pdc - 1) * 100) if pdc else None,
        "vwap": _num((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else None,
        "volume_so_far": int(vol.sum()),
    })
    return out


def get_candle_pattern_signals(company_code):
    from patterns import engine

    c = find_company(company_code)
    try:
        r = engine.predict(c, "5m")
    except engine.PredictorError as exc:
        raise ToolError(str(exc))
    main = engine.INTERVALS["5m"]["main"]

    def stat(s):
        m = s["horizons"].get(main, {})
        return {"pattern": s["name"], "direction": s["direction"], "reliable": s["reliable"],
                "samples": m.get("n", 0), "win_pct": _num(m.get("win_pct"), 1), "edge_pct": _num(m.get("edge_pct"), 3),
                "t_stat": _num(m.get("t")), "seen_at": (s.get("x") or "")[11:16] or None}

    f = r.get("forecast") or {}
    bt = (r.get("backtest") or {}).get("horizons", {})
    return {
        "company_code": c.code, "as_of": r["asof"], "price": _num(r["price"]), "is_live": r["is_live"],
        "explain": ("Patterns are detected on 5-minute candles and tested on this stock's own history: win_pct / "
                    "edge_pct = what happened 15 minutes later. reliable = >= 30 samples, t >= 2 and positive in both "
                    "halves of history. The analog forecast uses the ~60 most similar past moments; its tested "
                    "accuracy vs the always-guess-the-majority baseline is reported - treat a forecast whose accuracy "
                    "is not above the baseline as noise."),
        "levels": {k: _num(v) for k, v in r["levels"].items()},
        "active_patterns_now": [stat(s) for s in r["active"]],
        "patterns_today": [{"time": x["x"][11:16], "pattern": x["name"], "direction": x["direction"],
                            "reliable": x["reliable"], "hist_win_pct": _num(x["hist_win_pct"], 1),
                            "outcome_15m_pct": _num(x["outcome_pct"], 3)} for x in r["log"][:12]],
        "reliable_patterns_for_this_stock": [stat(s) for s in r["library"] if s["reliable"]][:8],
        "pattern_vote": _num(r["pattern_vote"], 4),
        "analog_forecast": None if not f else {
            "neighbors": f["neighbors"],
            "horizons": [{"in": h["label"], "price": _num(h["price"]), "ret_pct": _num(h["ret_pct"], 3),
                          "p_up": _num(h["p_up"]), "range_10_90": [_num(h["low"]), _num(h["high"])],
                          "tested_accuracy": _num(bt.get(h["bars"], {}).get("accuracy"), 3),
                          "baseline_accuracy": _num(bt.get(h["bars"], {}).get("baseline"), 3)} for h in f["horizons"]],
        },
    }


def get_ml_forecast(company_code):
    from prediction import engine as ml

    c = find_company(company_code)
    out = {"company_code": c.code, "explain": (
        "Forecasts from this app's per-stock model zoo (walk-forward tested). edge=false means the best model did "
        "not beat the baseline after costs on unseen data - treat it as noise.")}
    for tf in ("intraday", "daily"):
        try:
            p = ml.predict_stock(c, tf)
        except ml.NotTrained as exc:
            out[tf] = {"available": False, "reason": str(exc)}
            continue
        except Exception as exc:  # stale model files, missing data ...
            out[tf] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
            continue
        out[tf] = {"available": True, "as_of": p["as_of"], "price": _num(p["price"]), "forecasts": [
            {"in": f["label"], "direction": f["direction"], "ret_pct": _num(f["predicted_return_pct"], 3),
             "price": _num(f["predicted_price"]), "p_up": _num(f["p_up"]), "edge": bool(f["edge"]),
             "dir_acc": _num(f["dir_acc"], 3), "baseline_acc": _num(f["baseline_acc"], 3), "model": f["model"]}
            for f in p["forecasts"]]}
    return out


INDICES = {"NIFTY 50": "^NSEI", "NIFTY BANK": "^NSEBANK", "INDIA VIX": "^INDIAVIX"}


def get_market_context():
    from prediction.datasets import _yahoo, _yahoo_frame

    today = datetime.now(IST).date()
    out = {"market_status": get_market_status(), "now_ist": datetime.now(IST).strftime("%Y-%m-%d %H:%M"), "indices": {}}
    for name, sym in INDICES.items():
        d = _yahoo_frame(_yahoo(sym, cache_seconds=600, period="3mo", interval="1d"))
        info = {}
        if not d.empty:
            d["date"] = d["timestamp"].dt.date
            done = d[d["date"] < today].rename(columns={})
            done = done[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
            ind = daily_indicators(done) if len(done) > 2 else {}
            info["daily"] = {k: ind.get(k) for k in ("last_day", "close", "change_1d_pct", "change_5d_pct",
                                                     "change_20d_pct", "sma20", "sma50", "rsi14", "atr14_pct")}
            info["pivots_for_next_session"] = ind.get("pivots_for_next_session")
            info["last_10_closes"] = [[str(r.date), _num(r.close)] for r in done.tail(10).itertuples(index=False)]
        i5 = _yahoo_frame(_yahoo(sym, cache_seconds=60, period="1d", interval="5m", prepost=False))
        if not i5.empty and i5["timestamp"].dt.date.iloc[-1] == today:
            prev = info.get("daily", {}).get("close")
            last = float(i5["close"].iloc[-1])
            info["today"] = {"open": _num(i5["open"].iloc[0]), "high": _num(i5["high"].max()), "low": _num(i5["low"].min()),
                             "last": _num(last), "last_candle": i5["timestamp"].iloc[-1].strftime("%H:%M"),
                             "change_pct": _num((last / prev - 1) * 100) if prev else None,
                             "closes_every_30m": [[t.strftime("%H:%M"), _num(v)] for t, v in
                                                  zip(i5["timestamp"].iloc[5::6], i5["close"].iloc[5::6])]}
        out["indices"][name] = info
    return out


# ==========================================================================
# screener over the whole database (for intraday picks)
# ==========================================================================
SCREEN_SORTS = {
    "volume_surge": ("last_volume_vs_20d", False),          # vs the 20-day median
    "momentum_up": ("change_5d_pct", False),
    "momentum_down": ("change_5d_pct", True),
    "top_gainers_last_day": ("change_1d_pct", False),
    "top_losers_last_day": ("change_1d_pct", True),
    "high_volatility": ("atr14_pct", False),
    "near_20d_high": ("from_20d_high_pct", False),
    "near_20d_low": ("from_20d_low_pct", True),
    "narrow_range_compression": ("range_vs_atr", True),
    "most_liquid": ("turnover_cr_20d", False),
}


def build_screen():
    """Per-company metrics from the last ~2 months of daily candles in the database (cached)."""
    cached = cache.get("ai_prediction:screen")
    if cached is not None:
        return cached
    last = DailyData.objects.aggregate(Max("date"))["date__max"]
    if last is None:
        return pd.DataFrame(), None
    rows = DailyData.objects.filter(date__gte=last - timedelta(days=75)).exclude(company__sector="ETF").values_list(
        "company_id", "date", "open", "high", "low", "close", "volume")
    df = pd.DataFrame(list(rows), columns=["cid", "date", "open", "high", "low", "close", "volume"]).dropna(
        subset=["open", "high", "low", "close"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df = df[df["low"] > 0].sort_values(["cid", "date"])
    recs = []
    for cid, g in df.groupby("cid", sort=False):
        if len(g) < 22 or g["date"].iloc[-1] < last - timedelta(days=7):      # too short or stale
            continue
        c, last_row = g["close"], g.iloc[-1]
        atr = _atr(g).iloc[-1]
        vol20 = g["volume"].iloc[-21:-1].median()
        rng = g["high"] - g["low"]
        recs.append({
            "cid": cid, "last_day": last_row["date"], "close": last_row["close"],
            "change_1d_pct": (c.iloc[-1] / c.iloc[-2] - 1) * 100, "change_5d_pct": (c.iloc[-1] / c.iloc[-6] - 1) * 100,
            "change_20d_pct": (c.iloc[-1] / c.iloc[-21] - 1) * 100,
            "atr14_pct": atr / last_row["close"] * 100,
            "turnover_cr_20d": (g["close"] * g["volume"]).iloc[-20:].mean() / 1e7,
            "last_volume_vs_20d": last_row["volume"] / vol20 if vol20 else np.nan,
            "from_20d_high_pct": (last_row["close"] / g["high"].iloc[-20:].max() - 1) * 100,
            "from_20d_low_pct": (last_row["close"] / g["low"].iloc[-20:].min() - 1) * 100,
            "range_vs_atr": rng.iloc[-1] / atr if atr else np.nan,
            "nr7": bool(rng.iloc[-1] <= rng.iloc[-7:].min()),
            "inside_day": bool(last_row["high"] <= g["high"].iloc[-2] and last_row["low"] >= g["low"].iloc[-2]),
            "rsi14": _rsi(c).iloc[-1],
        })
    screen = pd.DataFrame(recs)
    if not screen.empty:
        meta = pd.DataFrame(list(Company.objects.filter(pk__in=screen["cid"]).values("id", "code", "name", "sector",
                                                                                      "industry")))
        screen = screen.merge(meta, left_on="cid", right_on="id").drop(columns=["id"])
    cache.set("ai_prediction:screen", (screen, last), SCREEN_CACHE_SECONDS)
    return screen, last


def screen_stocks(sort_by="volume_surge", sector=None, min_turnover_cr=10, limit=15):
    if sort_by not in SCREEN_SORTS:
        raise ToolError(f"sort_by must be one of {', '.join(SCREEN_SORTS)}.")
    screen, last = build_screen()
    if screen.empty:
        raise ToolError("The database has no recent daily candles to screen.")
    limit = max(5, min(int(limit or 15), 30))
    s = screen[screen["turnover_cr_20d"] >= float(min_turnover_cr or 0)]
    if sector:
        s = s[s["sector"].str.lower() == sector.strip().lower()]
    col, asc = SCREEN_SORTS[sort_by]
    s = s.dropna(subset=[col]).sort_values(col, ascending=asc).head(limit)
    sectors = (screen[screen["turnover_cr_20d"] >= 10].groupby("sector")
               .agg(stocks=("code", "size"), avg_change_1d_pct=("change_1d_pct", "mean"),
                    avg_change_5d_pct=("change_5d_pct", "mean"),
                    advancing_pct=("change_1d_pct", lambda x: (x > 0).mean() * 100)).reset_index())
    cols = ["code", "name", "sector", "industry", "close", "change_1d_pct", "change_5d_pct", "change_20d_pct",
            "atr14_pct", "turnover_cr_20d", "last_volume_vs_20d", "from_20d_high_pct", "from_20d_low_pct",
            "range_vs_atr", "nr7", "inside_day", "rsi14"]
    return {
        "database_data_until": str(last),
        "note": ("Metrics are from completed daily candles in the database, which can lag today by a few sessions. "
                 "Check today's price with get_live_quote before recommending a stock."),
        "sorted_by": sort_by, "min_turnover_cr": min_turnover_cr, "matches": len(s),
        "rows": [{k: (_num(v) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, (bool, np.bool_)) else v))
                  for k, v in zip(cols, r)} for r in s[cols].itertuples(index=False)],
        "sector_breadth_last_day": [{k: (_num(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
                                    for r in sectors.sort_values("avg_change_1d_pct", ascending=False).to_dict("records")],
    }


# ==========================================================================
# tool definitions for the Messages API
# ==========================================================================
_CODE = {"type": "string", "description": "Company code exactly as in this app, e.g. RELIANCE.NS"}


def _tool(name, description, props=None, required=()):
    return {"name": name, "description": description, "eager_input_streaming": True,
            "input_schema": {"type": "object", "properties": props or {}, "required": list(required),
                             "additionalProperties": False}}


DATA_TOOLS = {
    "get_stock_profile": (get_stock_profile, _tool(
        "get_stock_profile", "Company name, sector, industry and 52-week / monthly / weekly high-low from this app's "
        "database.", {"company_code": _CODE}, ["company_code"])),
    "get_daily_history": (get_daily_history, _tool(
        "get_daily_history", "Completed daily OHLCV candles (database topped up from Yahoo) plus indicators: SMA/EMA, "
        "RSI14, ATR14, 20-day and 52-week range, volume vs 20-day average, NR7 / inside-day flags and the classic "
        "pivots + CPR for the next session.",
        {"company_code": _CODE, "days": {"type": "integer", "description": "Rows to return, 5-250 (default 60)."}},
        ["company_code"])),
    "get_intraday_candles": (get_intraday_candles, _tool(
        "get_intraday_candles", "5-minute candles of the last 1-5 sessions, including today's live candles while the "
        "market is open, each with a session summary (open/high/low/last, VWAP, 15-minute opening range).",
        {"company_code": _CODE, "sessions": {"type": "integer", "description": "Sessions to return, 1-5 (default 2)."}},
        ["company_code"])),
    "get_live_quote": (get_live_quote, _tool(
        "get_live_quote", "Today's last price, open/high/low, gap and change vs previous close, VWAP and volume so far, "
        "plus the previous day's OHLC.", {"company_code": _CODE}, ["company_code"])),
    "get_candle_pattern_signals": (get_candle_pattern_signals, _tool(
        "get_candle_pattern_signals", "This app's Candle Predictor engine on 5-minute candles: intraday patterns active "
        "now and today, each tested on this stock's own history (win rate, edge, t-stat, reliable flag), key levels "
        "(PDH/PDL/opening range) and an analog forecast with its tested accuracy.", {"company_code": _CODE},
        ["company_code"])),
    "get_ml_forecast": (get_ml_forecast, _tool(
        "get_ml_forecast", "Forecast from this app's trained per-stock machine-learning models (intraday and daily "
        "horizons) with each model's walk-forward accuracy vs baseline. Not every stock is trained.",
        {"company_code": _CODE}, ["company_code"])),
    "get_market_context": (get_market_context, _tool(
        "get_market_context", "NIFTY 50, NIFTY BANK and INDIA VIX: recent daily trend, indicators, next-session pivots "
        "and today's intraday path.")),
    "screen_stocks": (screen_stocks, _tool(
        "screen_stocks", "Rank every stock in this app's database (ETFs excluded) by a daily-candle metric to find "
        "intraday candidates. Also returns sector breadth for the last session.",
        {"sort_by": {"type": "string", "enum": list(SCREEN_SORTS)},
         "sector": {"type": "string", "description": "Optional sector filter, e.g. Technology, Financial Services."},
         "min_turnover_cr": {"type": "number", "description": "Minimum 20-day average daily turnover in rupee crore "
                                                              "(default 10). Keep >= 10 for intraday liquidity."},
         "limit": {"type": "integer", "description": "Rows to return, 5-30 (default 15)."}},
        ["sort_by"])),
}

ANALYSIS_DATA_TOOLS = ["get_stock_profile", "get_daily_history", "get_intraday_candles", "get_live_quote",
                       "get_candle_pattern_signals", "get_ml_forecast", "get_market_context"]
PICKS_DATA_TOOLS = ["screen_stocks", "get_market_context", "get_live_quote", "get_daily_history",
                    "get_intraday_candles", "get_candle_pattern_signals"]

TOOL_LABELS = {
    "get_stock_profile": "Company profile", "get_daily_history": "Daily history + indicators",
    "get_intraday_candles": "5-minute candles", "get_live_quote": "Live quote",
    "get_candle_pattern_signals": "Candle Predictor patterns", "get_ml_forecast": "ML model forecast",
    "get_market_context": "NIFTY / BANK NIFTY / VIX", "screen_stocks": "Database screener",
}


def run_data_tool(name, args):
    """Run one data tool (serialised by DATA_LOCK). Returns (payload, is_error)."""
    fn = DATA_TOOLS[name][0]
    if not isinstance(args, dict):
        return {"error": "Tool input must be a JSON object."}, True
    try:
        with DATA_LOCK:
            return fn(**args), False
    except ToolError as exc:
        return {"error": str(exc)}, True
    except TypeError as exc:                        # unexpected / missing argument
        return {"error": f"Bad arguments: {exc}"}, True


# ==========================================================================
# the final answers
# ==========================================================================
_NUM_OR_NULL = {"type": ["number", "null"]}
_LEVEL = {"type": "object", "additionalProperties": False, "required": ["label", "price", "kind"], "properties": {
    "label": {"type": "string"}, "price": {"type": "number"},
    "kind": {"type": "string", "enum": ["support", "resistance", "pivot", "trigger"]}}}

ANALYSIS_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["company_code", "headline", "bias", "confidence", "price_at_analysis", "trade_plan", "price_outlook",
                 "key_levels", "scenarios", "technical_view", "news_view", "market_view", "catalysts", "risks",
                 "sources"],
    "properties": {
        "company_code": {"type": "string"},
        "headline": {"type": "string", "description": "One sentence verdict."},
        "bias": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
        "confidence": {"type": "integer", "description": "0-100, honest - 50 means no edge."},
        "price_at_analysis": {"type": "number"},
        "trade_plan": {"type": "object", "additionalProperties": False,
                       "required": ["action", "entry_type", "entry", "stop_loss", "target_1", "target_2", "valid_until",
                                    "invalidation", "position_note"],
                       "properties": {
                           "action": {"type": "string", "enum": ["BUY", "SELL", "NO_TRADE"]},
                           "entry_type": {"type": "string", "enum": ["MARKET", "BREAKOUT", "PULLBACK", "NONE"]},
                           "entry": _NUM_OR_NULL, "stop_loss": _NUM_OR_NULL,
                           "target_1": _NUM_OR_NULL, "target_2": _NUM_OR_NULL,
                           "valid_until": {"type": "string", "description": "IST time HH:MM after which the plan lapses."},
                           "invalidation": {"type": "string"},
                           "position_note": {"type": "string", "description": "Sizing / risk note."}}},
        "price_outlook": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["horizon", "expected_price", "low", "high", "direction", "probability"],
            "properties": {"horizon": {"type": "string", "description": "e.g. 30 min, 1 hour, Close"},
                           "expected_price": {"type": "number"}, "low": {"type": "number"}, "high": {"type": "number"},
                           "direction": {"type": "string", "enum": ["UP", "DOWN", "FLAT"]},
                           "probability": {"type": "integer", "description": "0-100 chance the direction is right."}}}},
        "key_levels": {"type": "array", "items": _LEVEL},
        "scenarios": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "probability", "trigger", "path"],
            "properties": {"name": {"type": "string"}, "probability": {"type": "integer"},
                           "trigger": {"type": "string"}, "path": {"type": "string"}}}},
        "technical_view": {"type": "string"}, "news_view": {"type": "string"}, "market_view": {"type": "string"},
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "sources": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                               "required": ["title", "url"],
                                               "properties": {"title": {"type": "string"}, "url": {"type": "string"}}}},
    },
}

PICKS_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["market_bias", "market_summary", "picks", "avoid"],
    "properties": {
        "market_bias": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL", "VOLATILE"]},
        "market_summary": {"type": "string"},
        "picks": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["company_code", "direction", "setup", "entry", "stop_loss", "target", "confidence", "reason",
                         "risk"],
            "properties": {"company_code": {"type": "string"}, "direction": {"type": "string", "enum": ["LONG", "SHORT"]},
                           "setup": {"type": "string", "description": "e.g. ORB breakout, VWAP pullback"},
                           "entry": {"type": "number"}, "stop_loss": {"type": "number"}, "target": {"type": "number"},
                           "confidence": {"type": "integer"}, "reason": {"type": "string"},
                           "catalyst": {"type": "string"}, "risk": {"type": "string"}}}},
        "avoid": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                             "required": ["company_code", "reason"],
                                             "properties": {"company_code": {"type": "string"},
                                                            "reason": {"type": "string"}}}},
    },
}

SUBMIT_TOOLS = {
    "submit_analysis": _tool("submit_analysis", "Submit the final intraday analysis of the stock. Call it exactly once, "
                                                "when the research is complete.", ANALYSIS_SCHEMA["properties"],
                             ANALYSIS_SCHEMA["required"]),
    "submit_picks": _tool("submit_picks", "Submit the final intraday trade ideas. Call it exactly once, when the "
                                          "research is complete.", PICKS_SCHEMA["properties"], PICKS_SCHEMA["required"]),
}


def _check(value, schema, path, errors):
    """Minimal JSON-schema check (types, enums, required, unknown keys) - enough for these two schemas."""
    t = schema.get("type")
    types = t if isinstance(t, list) else [t] if t else []
    ok = {"object": dict, "array": list, "string": str, "boolean": bool}

    def is_type(v, name):
        if name == "null":
            return v is None
        if name == "number":
            return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        if name == "integer":
            return isinstance(v, int) and not isinstance(v, bool)
        return isinstance(v, ok[name])

    if types and not any(is_type(value, n) for n in types):
        errors.append(f"{path or 'input'}: expected {' or '.join(types)}")
        return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']}")
    if isinstance(value, dict) and "properties" in schema:
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: missing" if path else f"{key}: missing")
        for key, v in value.items():
            if key not in schema["properties"]:
                errors.append(f"{path}.{key}: unknown field" if path else f"{key}: unknown field")
            else:
                _check(v, schema["properties"][key], f"{path}.{key}" if path else key, errors)
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            _check(v, schema["items"], f"{path}[{i}]", errors)


def validate_analysis(data, company):
    errors = []
    _check(data, ANALYSIS_SCHEMA, "", errors)
    if errors:
        return errors
    plan = data["trade_plan"]
    if plan["action"] != "NO_TRADE":
        need = [k for k in ("entry", "stop_loss", "target_1") if plan.get(k) is None]
        if need:
            return [f"trade_plan.{k}: required when action is {plan['action']}" for k in need]
        e, sl, t1 = plan["entry"], plan["stop_loss"], plan["target_1"]
        if plan["action"] == "BUY" and not sl < e < t1:
            errors.append("trade_plan: a BUY needs stop_loss < entry < target_1")
        if plan["action"] == "SELL" and not t1 < e < sl:
            errors.append("trade_plan: a SELL needs target_1 < entry < stop_loss")
    if not 0 <= data["confidence"] <= 100:
        errors.append("confidence: 0-100")
    if find_company(data["company_code"]).pk != company.pk:
        errors.append(f"company_code: must be {company.code}")
    return errors


def validate_picks(data):
    errors = []
    _check(data, PICKS_SCHEMA, "", errors)
    if errors:
        return errors
    for i, p in enumerate(data["picks"]):
        try:
            p["company_code"] = find_company(p["company_code"]).code
        except ToolError as exc:
            errors.append(f"picks[{i}]: {exc}")
            continue
        e, sl, t = p["entry"], p["stop_loss"], p["target"]
        if p["direction"] == "LONG" and not sl < e < t:
            errors.append(f"picks[{i}] {p['company_code']}: a LONG needs stop_loss < entry < target")
        if p["direction"] == "SHORT" and not t < e < sl:
            errors.append(f"picks[{i}] {p['company_code']}: a SHORT needs target < entry < stop_loss")
    if not data["picks"]:
        errors.append("picks: give at least one idea (or explain in market_summary why only low-confidence ones exist)")
    return errors

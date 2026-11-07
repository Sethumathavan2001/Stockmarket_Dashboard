"""
Claude Orders 2.0: the signal board of claude_board.py, widened to everything the dashboard knows.  Same order format and
same page; only what Refresh orders builds differs.

  1. today        weekday, session (Morning / Afternoon / Evening), every index on NSE's feed plus SENSEX, India VIX, the
                  world strip (GIFT Nifty, NASDAQ, S&P 500, Nikkei, Hang Seng), and which index contracts expire today
  2. behaviour    Stock Behaviour with today's filters: NIFTY's trend, the stock's own index, this session, this weekday,
                  this expiry.  A filter votes only if its list repeated when tested at refresh time (Engine.recheck), and
                  a stock only if it beat the typical stock on those days.  So a filter that doesn't repeat counts for nothing
  3. candidates   claude_board.candidates (Live page, News, trained models) + the behaviour lists + F&O open interest spurts
  4. evidence     claude_board.gather and score (Candle Predictor, patterns, ML, News, price action, NIFTY), then the ML daily
                  forecast, open interest, the stock's sector index, all indices, global markets and Stock Behaviour
  5. decision     see rescore(): stricter than the first version, and a stock can't be carried by the market alone
  6. levels       the first version's stop, widened to a quarter of the stock's usual range in this session (still 0.4%-1.2%)

The result replaces claude_orders2.json.  Stocks are analysed by WORKERS threads; start_refresh() runs the whole thing in
the background and progress() reports how far it is.
"""
import json
import logging
import threading
import time as _time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from prediction.market_time import IST

from . import board_pool
from . import claude_board as board

log = logging.getLogger(__name__)

FLAT = 0.2                     # an index or world market within this % of its previous close has no direction
BREADTH_SHARE = 0.6            # this share of the indices one way makes a direction
BREADTH_GROUPS = ("INDICES ELIGIBLE IN DERIVATIVES", "BROAD MARKET INDICES", "SECTORAL INDICES")
VIX = "INDIA VIX"
OI_MIN_PCT = 10                # an open interest rise below this % isn't a build-up
MIN_SCORE, MIN_AGREE, MIN_STOCK, AGAINST_MARKET = 0.4, 3, 2, 4
REPEATS, REPEATS_A_LITTLE = 5.0, 2.0            # same edges as the Stock Behaviour page's repeat label
TOP_BEHAVIOUR = 10
RANGE_SHARE = 0.25             # the stop stays outside this share of the stock's usual session range
STOP_MIN, STOP_MAX = 0.004, 0.012
SESSION_MIN_BARS = 3           # fewer 5-minute candles than this into a session says nothing about its trend
WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

MARKET, BEHAVIOUR, SECTOR, ALL_INDICES, GLOBAL = "Market (NIFTY)", "Stock Behaviour", "Sector index", "All indices", "Global markets"
ML_DAILY, OPEN_INTEREST = "ML daily", "Open interest"
BACKDROP = (MARKET, ALL_INDICES, SECTOR, GLOBAL)    # the market around the stock, not the stock itself

METHOD = ("BUY / SELL needs a score of 0.4+, three agreeing signals, two of them about the stock itself and one of those a "
          "tested intraday model (Candle Predictor, patterns, ML intraday). Stock Behaviour and the stock's sector index must not disagree, and "
          "going against NIFTY needs four stock signals. A Stock Behaviour filter votes only if it repeated when tested today, "
          "and a stock only if it beat the typical stock. Stop from the forecast's 80% range about an hour ahead, widened to a "
          "quarter of the stock's usual session range (0.4%-1.2%); target 1.5x the stop distance.")

# index name in the constituent lists -> the same index on NSE's feed
FEED_NAMES = {"NIFTY FIN SERVICE": "NIFTY FINANCIAL SERVICES", "NIFTY HEALTHCARE": "NIFTY HEALTHCARE INDEX"}
# stocks in no sector index: the closest one for their sector
SECTOR_INDEX = {"Technology": "NIFTY IT", "Financial Services": "NIFTY FINANCIAL SERVICES", "Healthcare": "NIFTY PHARMA",
                "Consumer Defensive": "NIFTY FMCG", "Consumer Cyclical": "NIFTY AUTO", "Basic Materials": "NIFTY METAL",
                "Energy": "NIFTY ENERGY", "Real Estate": "NIFTY REALTY", "Communication Services": "NIFTY MEDIA",
                "Utilities": "NIFTY ENERGY", "Industrials": "NIFTY INFRASTRUCTURE"}
# Stock Behaviour's index key -> its name on NSE's feed (BANKEX has no live feed here)
BEHAVIOUR_INDEX = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK", "FINNIFTY": "NIFTY FINANCIAL SERVICES",
                   "MIDCPNIFTY": "NIFTY MIDCAP SELECT", "NIFTYNXT50": "NIFTY NEXT 50", "NIFTYIT": "NIFTY IT",
                   "NIFTYMID100": "NIFTY MIDCAP 100", "NIFTYSML100": "NIFTY SMALLCAP 100", "NIFTY500": "NIFTY 500",
                   "SENSEX": "SENSEX"}
# a stock's own index for Stock Behaviour, narrowest first (NIFTY itself is always checked)
HOME_INDEX = [("NIFTY BANK", "BANKNIFTY"), ("NIFTY IT", "NIFTYIT"), ("NIFTY FIN SERVICE", "FINNIFTY"),
              ("NIFTY NEXT 50", "NIFTYNXT50"), ("NIFTY MIDCAP 100", "NIFTYMID100"), ("NIFTY SMALLCAP 100", "NIFTYSML100")]

_lock = threading.Lock()
_progress = {"running": False, "step": "", "done": 0, "total": 0, "started": None, "finished": None,
             "error": None, "orders": None}
_expiry_day = {}


def progress():
    with _lock:
        return dict(_progress)


def _set(**kw):
    with _lock:
        _progress.update(kw)


def start_refresh(orders_file):
    """Start a background refresh; False when one is already running."""
    with _lock:
        if _progress["running"]:
            return False
        _progress.update(running=True, step="Fetching Live page data", done=0, total=0, error=None, orders=None,
                         started=datetime.now(IST).isoformat(), finished=None)
    threading.Thread(target=_run, args=(Path(orders_file),), name="claude-board-2", daemon=True).start()
    return True


def _run(orders_file):
    try:
        payload = build()
        tmp = orders_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(orders_file)
        try:
            board.archive(payload)
        except Exception:                                         # the list itself is written; history is best effort
            log.exception("Claude Orders 2.0: could not save the list to the history")
        _set(orders=len(payload["orders"]), step="Done")
    except Exception as exc:                                       # shown on the page; the old list is kept
        log.exception("Claude Orders 2.0 refresh failed")
        _set(error=f"{type(exc).__name__}: {exc}", step="Failed")
    finally:
        from django.db import connection
        connection.close()
        _set(running=False, finished=datetime.now(IST).isoformat())


# ---------------------------------------------------------------- 1. today
def _sgn(v):
    return 0 if not v else (1 if v > 0 else -1)


def _word(direction):
    return {1: "up", -1: "down", 0: "flat"}[direction]


def current_session(now):
    """Stock Behaviour's session for this time: the first before the open, the last after the close."""
    from market.behaviour_data import SESSIONS

    minute = now.hour * 60 + now.minute
    for name, _, _, end in SESSIONS:
        if minute < end:
            return name
    return SESSIONS[-1][0]


def _index_row(name, last, opn, pct, group=""):
    o2c = (last / opn - 1) * 100 if last and opn else None
    direction = 0
    if pct is not None and o2c is not None and abs(pct) >= FLAT and _sgn(pct) == _sgn(o2c):
        direction = _sgn(pct)                                     # gaining on both the previous close and the open
    return {"name": name, "pct": pct, "o2c": o2c, "dir": direction, "group": group}


def read_indices():
    """{name: row} for every index on NSE's feed, and SENSEX."""
    from market.indices import _num, live_indices
    from news.fetch import _nse_json

    out = {}
    for r in _nse_json("/api/allIndices").get("data", []):
        out[r.get("index")] = dict(_index_row(r.get("index"), _num(r.get("last")), _num(r.get("open")),
                                              _num(r.get("percentChange")), r.get("key") or ""),
                                   advances=_num(r.get("advances")), declines=_num(r.get("declines")))
    for i in live_indices().get("indices", []):
        if i.get("short") == "SENSEX" and not i.get("stale"):
            out["SENSEX"] = _index_row("SENSEX", i.get("last"), i.get("open"), i.get("pct"))
    return out


def breadth(indices):
    """All broad and sector indices, with India VIX read the other way round -> direction + how many went each way."""
    rows = [r for r in indices.values() if r["group"] in BREADTH_GROUPS and r["name"] != VIX and r["pct"] is not None]
    up, down = sum(r["pct"] >= FLAT for r in rows), sum(r["pct"] <= -FLAT for r in rows)
    direction = 0
    if rows and up / len(rows) >= BREADTH_SHARE:
        direction = 1
    elif rows and down / len(rows) >= BREADTH_SHARE:
        direction = -1
    vix = (indices.get(VIX) or {}).get("pct")
    return {"dir": direction, "up": int(up), "down": int(down), "total": len(rows), "vix_pct": vix}


def world(rows):
    """The world strip -> direction + how many went each way."""
    rows = [r for r in rows if r.get("pct") is not None and not r.get("stale")]
    up, down = sum(r["pct"] >= FLAT for r in rows), sum(r["pct"] <= -FLAT for r in rows)
    return {"dir": 1 if up >= 3 and up > down else -1 if down >= 3 and down > up else 0, "up": up, "down": down,
            "total": len(rows), "detail": ", ".join(f"{r['short']} {r['pct']:+.2f}%" for r in rows)}


def session_move(chart, session):
    """% move of an index so far in this session, from its 5-minute candles; None too early to say."""
    from market.behaviour_data import SESSIONS

    start = next(s[2] for s in SESSIONS if s[0] == session)
    bars = []
    for c in chart.get("candles") or []:
        t = datetime.fromisoformat(c[0]).astimezone(IST)
        if t.hour * 60 + t.minute >= start:
            bars.append(c)
    if len(bars) < SESSION_MIN_BARS or not bars[0][1]:
        return None
    return (bars[-1][4] / bars[0][1] - 1) * 100


def today_expiries(today):
    """{index key: "weekly" | "monthly"} for index contracts expiring today.  The last trading day's F&O files list every
    live contract, and a month's final expiry is the monthly one.  Read once a day."""
    from market.index_data import CATALOG, bse_expiries, nse_expiries

    if _expiry_day.get("day") == today:
        return _expiry_day["expiring"]
    out = {}
    for reader, exchange in ((nse_expiries, "NSE"), (bse_expiries, "BSE")):
        symbols = {c[4]: c[0] for c in CATALOG if c[2] == exchange and c[4]}
        day, found = today, None
        for _ in range(7):                                         # back over weekends and holidays
            day -= timedelta(days=1)
            found = reader(day, set(symbols)) if day.weekday() < 5 else None
            if found:
                break
        for symbol, dates in (found or {}).items():
            if today in dates:
                later = any(d > today and (d.year, d.month) == (today.year, today.month) for d in dates)
                out[symbols[symbol]] = "weekly" if later else "monthly"
    _expiry_day.update(day=today, expiring=out)
    return out


def _attempt(what, read, default=None):
    """One source failing leaves its signal out; the list is still made from the rest."""
    try:
        return read()
    except Exception:
        log.warning("Claude Orders 2.0: %s unavailable", what, exc_info=True)
        return default


def read_context(market, now):
    from market.indices import global_indices, index_chart

    session_day = datetime.fromisoformat((market or {}).get("session") or now.date().isoformat()).date()
    session = current_session(now)
    indices = _attempt("NSE indices", read_indices, {})
    chart = _attempt("NIFTY candles", lambda: index_chart("nifty50"), {})
    return {"day": session_day, "weekday": session_day.weekday(), "session": session, "indices": indices,
            "breadth": breadth(indices), "session_move": session_move(chart, session),
            "world": world(_attempt("global indices", lambda: global_indices()["indices"], [])),
            "expiry": _attempt("expiry dates", lambda: today_expiries(session_day), {})}


def summary(ctx, market):
    """The day as this list read it, for the top of the page."""
    from market.behaviour_data import SESSIONS

    label = next(s[1] for s in SESSIONS if s[0] == ctx["session"])
    parts = [f"Read for {WEEKDAY_NAMES[ctx['weekday']]}, {label} session."]
    if market:
        parts.append(f"NIFTY {market['day_pct']:+.2f}% from its open.")
    b = ctx["breadth"]
    if b["total"]:
        parts.append(f"{b['up']} of {b['total']} indices up, {b['down']} down.")
    if b["vix_pct"] is not None:
        parts.append(f"India VIX {b['vix_pct']:+.2f}%.")
    if ctx["world"]["total"]:
        parts.append(f"World: {ctx['world']['detail']}.")
    if ctx["expiry"]:
        parts.append("Expiring today: " + ", ".join(f"{k} ({v})" for k, v in ctx["expiry"].items()) + ".")
    return " ".join(parts)


# ---------------------------------------------------------------- 2. behaviour
def _trend(pct, limit):
    return "any" if pct is None or abs(pct) < limit else ("up" if pct > 0 else "down")


class Condition:
    """One Stock Behaviour filter set with today's values: every stock's record on those days, each way."""

    def __init__(self, key, label, engine):
        from market.stock_behaviour import MIN_DAYS, _column_stats, wilson_lower

        self.key, self.label, self.engine = key, label, engine
        self.mask = engine.mask()
        self.days = int(self.mask.sum())
        self.sides = {}
        if self.days < MIN_DAYS or not len(engine.cols):
            return
        self.position = {str(engine.d["codes"][c]): j for j, c in enumerate(engine.cols)}
        moves = engine.moves(self.mask, engine.session, engine.cols)
        need = engine.need(self.days)
        for sign in (1, -1):
            n, hits, avg, _, _ = _column_stats(moves, sign)
            rate = hits / np.maximum(n, 1)
            ok = n >= need
            check = engine.recheck(self.mask, sign)
            edge = None if check is None else check["newer"] - check["typical"]
            self.sides[sign] = {"n": n, "rate": rate, "avg": avg, "ok": ok, "low": wilson_lower(hits, n), "edge": edge,
                                "typical": float(np.median(rate[ok])) if ok.any() else None,
                                "strength": 0.0 if edge is None or edge < REPEATS_A_LITTLE else 0.5 if edge < REPEATS else 1.0}

    def repeats(self):
        return any(s["strength"] for s in self.sides.values())

    def vote(self, code):
        """(+1 / -1, strength, words), or None: the filter must have repeated that way and the stock must have gone that
        way surely more often than the typical stock and more often than not."""
        if not self.sides or code not in self.position:
            return None
        j = self.position[code]
        for sign, side in self.sides.items():
            if (side["strength"] and side["ok"][j] and side["avg"][j] * sign > 0
                    and side["low"][j] > max(0.5, side["typical"])):
                return sign, side["strength"], (f"{self.label}: {_word(sign)} {side['rate'][j] * 100:.0f}% of "
                                                f"{int(side['n'][j])} days, typical stock {side['typical'] * 100:.0f}%")
        return None

    def candidates(self):
        return [(row["code"], sign) for sign, side in self.sides.items() if side["strength"]
                for row in self.engine.top(self.mask, sign, TOP_BEHAVIOUR)]


def home_index(code, sector_lookup=None):
    from market.index_membership import indices_for

    names = {i["name"] for i in (sector_lookup or indices_for)(code)["indices"]}
    return next((key for name, key in HOME_INDEX if name in names), None)


def conditions(data, ctx):
    """Today's filter sets for all stocks, and those for each index a stock can call home."""
    from market.behaviour_data import SESSIONS
    from market.stock_behaviour import DAY_MOVE, SESSION_MOVE, Engine, Filters

    def build(key, label, **filters):
        return Condition(key, label, Engine(data, Filters(**filters)))

    def day_trend(index):
        return _trend((ctx["indices"].get(BEHAVIOUR_INDEX.get(index)) or {}).get("o2c"), DAY_MOVE)

    def days(index, trend):
        return f"{index} {trend} days" if trend != "any" else f"All {index} days"

    session, weekday = ctx["session"], ctx["weekday"]
    label = next(s[1] for s in SESSIONS if s[0] == session)
    name = WEEKDAY_NAMES[weekday]
    trend, trend_now = day_trend("NIFTY"), _trend(ctx["session_move"], SESSION_MOVE)
    common = [build("index", days("NIFTY", trend), trend=trend),
              build("session", f"{label} with NIFTY {trend_now}" if trend_now != "any" else f"{label} sessions",
                    trend=trend_now, session=session)]
    if weekday < 5:
        common.append(build("weekday", f"{name}s with NIFTY {trend}" if trend != "any" else f"{name}s", trend=trend, weekday=weekday))
        common.append(build("weekday_session", f"{name} {label.lower()}s", weekday=weekday, session=session))
    common += [build(f"expiry:{index}", f"{index} {kind} expiries", index=index, expiry=kind)
               for index, kind in ctx["expiry"].items()]
    home = {index: build(f"home:{index}", days(index, day_trend(index)), index=index, trend=day_trend(index))
            for _, index in HOME_INDEX if index in data["ix_keys"]}
    return common, home


def session_ranges(data, session):
    """{code: the stock's average range % in this session}."""
    from market.stock_behaviour import Engine, Filters

    e = Engine(data, Filters(session=session))
    values = e.ranges(e.mask(), e.session, e.cols)
    n = np.isfinite(values).sum(axis=0)
    mean = np.nansum(values, axis=0) / np.maximum(n, 1)
    return {str(data["codes"][c]): float(mean[j]) for j, c in enumerate(e.cols) if n[j]}


# ---------------------------------------------------------------- 4. evidence + 5. decision
def behaviour_signal(code, conds):
    votes = [(c, c.vote(code)) for c in conds]
    votes = [(c, v) for c, v in votes if v]
    total = sum(sign * strength for _, (sign, strength, _) in votes)
    direction = _sgn(total)
    weight = 1.0 if abs(total) >= 1 else 0.5 if total else 0.0
    used = [v[2] for _, v in votes if v[0] == direction]
    skipped = [c.label for c in conds if c.sides and not c.repeats()]
    detail = "; ".join(used) or "nothing that repeated picks this stock out today"
    if skipped:
        detail += f" (no vote, didn't repeat: {', '.join(skipped)})"
    return {"name": BEHAVIOUR, "dir": direction, "w": weight, "cred": bool(weight), "detail": detail}


def sector_signal(s, ctx, sector_lookup=None):
    from market.index_membership import indices_for

    names = [FEED_NAMES.get(i["name"], i["name"]) for i in (sector_lookup or indices_for)(s["code"])["indices"]
             if i["group"] == "sector"]
    rows = [ctx["indices"][n] for n in names if n in ctx["indices"]]
    fallback = SECTOR_INDEX.get(s.get("sector") or "")
    if not rows and fallback in ctx["indices"]:
        rows = [ctx["indices"][fallback]]
    if not rows:
        return None
    direction = _sgn(sum(r["dir"] for r in rows))
    return {"name": SECTOR, "dir": direction, "w": 1.0 if direction else 0.0, "cred": bool(direction),
            "detail": ", ".join(f"{r['name']} {r['pct']:+.2f}%" for r in rows if r["pct"] is not None)}


def ml_daily_signal(s):
    forecasts = s.get("ml_daily") or []
    edge = [x for x in forecasts if x.get("edge")]
    if not edge:
        return None
    direction = _sgn(sum(x["ret_pct"] for x in edge))             # a next-days forecast only leans on an intraday trade
    return {"name": ML_DAILY, "dir": direction, "w": 0.5 if direction else 0.0, "cred": bool(direction),
            "detail": "; ".join(f"{x['in']} {x['direction']} ({x['ret_pct']:+.3f}%)" for x in edge)}


def oi_signal(s, oi):
    """Open interest rising with the price is new longs, with a falling price new shorts."""
    row = (oi or {}).get(s["code"].removesuffix(".NS"))
    pdc = (s.get("levels") or {}).get("pdc")
    if not row or not pdc or (row.get("avgInOI") or 0) < OI_MIN_PCT:
        return None
    direction = _sgn(s["price"] - pdc)
    label = "long build-up" if direction > 0 else "short build-up"
    return {"name": OPEN_INTEREST, "dir": direction, "w": 0.5 if direction else 0.0, "cred": bool(direction),
            "detail": f"OI {row['avgInOI']:+.0f}% with the price {(s['price'] / pdc - 1) * 100:+.2f}%: {label}"}


def backdrop_signals(ctx):
    out = []
    b, w = ctx["breadth"], ctx["world"]
    if b["total"]:
        vix = "" if b["vix_pct"] is None else f", India VIX {b['vix_pct']:+.2f}%"
        out.append({"name": ALL_INDICES, "dir": b["dir"], "w": 0.5 if b["dir"] else 0.0, "cred": bool(b["dir"]),
                    "detail": f"{b['up']} of {b['total']} indices up, {b['down']} down{vix}"})
    if w["total"]:
        out.append({"name": GLOBAL, "dir": w["dir"], "w": 0.5 if w["dir"] else 0.0, "cred": bool(w["dir"]), "detail": w["detail"]})
    return out


def plan_for(s, direction, usual_range):
    band = (s.get("forecast") or {}).get("band")
    if not band:
        return None
    price = s["price"]
    k = min(11, len(band) - 1)
    r = ((price - band[k][1]) if direction > 0 else (band[k][2] - price)) / price
    if usual_range:
        r = max(r, usual_range / 100 * RANGE_SHARE)
    r = min(max(r, STOP_MIN), STOP_MAX)
    return {"stop": price * (1 - direction * r), "target": price * (1 + direction * 1.5 * r), "risk_pct": r * 100}


def decide(sig):
    """-> (decision, score, conviction) from all of a stock's signals."""
    weight = sum(x["w"] for x in sig)
    score = sum(x["w"] * x["dir"] for x in sig) / weight if weight else 0.0
    d = _sgn(score)
    agree = [x for x in sig if x["cred"] and x["dir"] == d]
    stock = [x for x in agree if x["name"] not in BACKDROP]
    models = [x for x in stock if x["name"] in board.MODEL_SIGNALS]      # intraday: a next-days forecast is no reason alone
    opposed = lambda name: any(x["name"] == name and x["cred"] and x["dir"] == -d for x in sig)
    if (not d or abs(score) < MIN_SCORE or len(agree) < MIN_AGREE or len(stock) < MIN_STOCK or not models
            or opposed(BEHAVIOUR) or opposed(SECTOR) or (opposed(MARKET) and len(stock) < AGAINST_MARKET)):
        return "WATCH", score, None
    higher = len(models) >= 2 and any(x["name"] == BEHAVIOUR for x in agree) or len(stock) >= 5
    return ("BUY" if d > 0 else "SELL"), score, "Higher" if higher else "Medium"


def rescore(s, ctx, conds, usual_range=None, oi=None):
    """Adds the 2.0 signals to a stock board.score() has read, and decides again."""
    if "signals" not in s:
        return s
    extra = [ml_daily_signal(s), oi_signal(s, oi), _attempt("sector index", lambda: sector_signal(s, ctx))]
    if conds:
        extra.append(behaviour_signal(s["code"], conds))
    sig = s["signals"] + [x for x in extra if x] + backdrop_signals(ctx)
    decision, score, conviction = decide(sig)
    plan = None if decision == "WATCH" else plan_for(s, 1 if decision == "BUY" else -1, usual_range)
    s.update(signals=sig, score=round(score, 2), decision=decision, conviction=conviction, plan=plan)
    return s


# ---------------------------------------------------------------- 3. candidates + the run
def more_candidates(cands, why, common, home, oi):
    """The behaviour lists and open interest spurts, on top of the first version's candidates."""
    from market.models import Company

    extra = {}
    for c in common + list(home.values()):
        for code, sign in c.candidates():
            extra.setdefault(code, []).append(f"Behaviour: {c.label}, {_word(sign)}")
    for symbol, row in oi.items():
        if (row.get("avgInOI") or 0) >= OI_MIN_PCT:
            extra.setdefault(f"{symbol}.NS", []).append(f"Live: open interest {row['avgInOI']:+.0f}%")
    known = set(Company.objects.exclude(sector="ETF").filter(code__in=extra).values_list("code", flat=True))
    for code, reasons in extra.items():
        if code in known:
            if code not in why:
                cands.append(code)
                why[code] = []
            why[code] += reasons
    return cands, why


def build():
    from market import behaviour_data

    started = _time.monotonic()
    with board_pool.Workers() as workers:
        _set(step="Reading the Live page, NIFTY and the news")
        first = board.opening_reads()
        live, market = first["live"], first["market"]
        pdh = {r["nsecode"] for r in live.get("pdh_data", [])}
        pdl = {r["nsecode"] for r in live.get("pdl_data", [])}
        oi = {r["symbol"]: r for r in live.get("open_intrest") or [] if r.get("symbol")}
        _set(step="Reading all indices and global markets")
        gen = datetime.now(IST).replace(microsecond=0)
        ctx = read_context(market, gen)

        _set(step="Reading Stock Behaviour for today")
        data = behaviour_data.load()
        common, home, ranges = [], {}, {}
        if data is not None:
            common, home = conditions(data, ctx)
            ranges = session_ranges(data, ctx["session"])

        _set(step="Choosing candidate stocks")
        cands, why = more_candidates(*board.candidates(live, fresh_news=False), common, home, oi)
        _set(total=len(cands), done=0)
        done = {}

        def score_one(code, stock):
            try:
                if stock:
                    s = board.score(stock, market, pdh, pdl)
                    key = _attempt("home index", lambda: home_index(code))
                    done[code] = rescore(s, ctx, common + ([home[key]] if key in home else []), ranges.get(code), oi)
            except Exception:
                log.warning("Claude Orders 2.0 refresh: %s failed", code, exc_info=True)

        workers.gather(cands, why, lambda step, n: _set(step=step, done=n), score_one)

    scored = [done[code] for code in cands if code in done]       # the same order every time
    _set(step="Writing orders")
    picks = sorted((s for s in scored if s.get("decision") in ("BUY", "SELL") and s.get("plan")),
                   key=lambda s: (s["decision"] != "SELL", -abs(s["score"])))     # SELL first, then BUY
    counts = {k: sum(1 for s in scored if s.get("decision") == k) for k in ("BUY", "SELL", "WATCH", "SKIP")}
    orders = [board.to_order(s, gen) for s in picks]
    for o in orders:
        o["id"] = f"v2-{o['id']}"                                 # a list's ids never meet the first version's
    note = "" if data is not None else " Stock Behaviour data is not built, so it had no say in this list."
    return {
        "generated": gen.isoformat(),
        "session": (market or {}).get("session") or gen.date().isoformat(),
        "source": "Signal board 2.0", "source_url": None, "method": f"{summary(ctx, market)} {METHOD}{note}",
        "checked": len(cands), "counts": counts,
        "market": market, "live_ok": bool(live.get("gainers") or live.get("loosers")),
        "orders": orders,
        "seconds": round(_time.monotonic() - started), "angel_prices": workers.live_prices,
    }

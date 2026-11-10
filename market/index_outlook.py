"""
Home page: is each index Bullish or Bearish?

  GET /api/market/outlook/   ->  {"indices": [...], "bullish", "bearish", "mode", "day", "global", "as_of"}

Five readings, each scored from -1 (bearish) to +1 (bullish); the verdict is the sign of their average, equal weights:

  Previous day   the candle's colour and where it closed in its range
  Today          last price against the open, where it sits in today's range, and whether it is outside yesterday's range
  Gap            yesterday's close to today's open
  History        on past days with the same setup (yesterday's colour and close, today's gap), how often the index closed
                 above its open.  Only days before this session are counted.
  Global         the average move of the markets on the Global strip (market/indices.py).  A market whose latest candle
                 is older than the last weekday is left out: that move is already in the Indian session.

Before the 09:15 open Today and Gap don't exist yet and are left out.  After the close, and on holidays, the page reads the
last session.  India VIX reads the other way round: its falling is Bullish for the market.

Today's numbers come from Angel One's quotes, past candles from IndexDaily, which is topped up here once a day when behind.
"""
import logging
import threading
from collections import defaultdict
from datetime import date, datetime, timedelta

from django.core.cache import cache
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from prediction.market_time import IST, MARKET_OPEN, get_market_status

from . import index_data
from .models import IndexDaily, MarketIndex

logger = logging.getLogger(__name__)

CACHE_KEY, GOOD_KEY, TOPPED_KEY = "market:outlook", "market:outlook:last_good", "market:outlook:topped:{}"
CACHE_SECONDS = 30
HISTORY_FROM = date(2016, 1, 1)
FLAT_GAP = 0.1                  # % - a smaller gap counts as flat when looking for similar days
GAP_FULL = 0.5                  # % - a gap this big scores the full +1 / -1
GLOBAL_FULL = 0.75              # % - so does this average world move
HIST_FULL = 15                  # percentage points away from 50-50
MIN_SIMILAR = 30                # fewer similar days than this say nothing
INVERTED = {"INDIAVIX"}
ANY_GAP = "any"
_lock = threading.Lock()


def _sign(x):
    return (x > 0) - (x < 0)


def _clip(x):
    return max(-1.0, min(1.0, x))


def where_in_range(close, high, low):
    """-1 at the day's low, +1 at its high."""
    return ((close - low) - (high - close)) / (high - low) if high > low else 0.0


def _place(position):
    return "top" if position > 1 / 3 else "bottom" if position < -1 / 3 else "middle"


def _factor(key, score, short, text):
    return {"key": key, "score": None if score is None else round(score, 3), "short": short, "text": text}


# ---------------------------------------------------------------- the five readings
def previous_day(o, h, l, c, day):
    position, colour = where_in_range(c, h, l), "Green" if c >= o else "Red"
    place = _place(position)
    return _factor("previous", (_sign(c - o) + position) / 2, f"{colour}, closed {place}",
                   f"{day:%a %d %b}: O {o:,.2f}  H {h:,.2f}  L {l:,.2f}  C {c:,.2f}. {colour} candle, closed in the {place} of its range.")


def today(o, h, l, last, prev_high=None, prev_low=None):
    position = where_in_range(last, h, l)
    outside = 0
    if prev_high is not None and last > prev_high:
        outside = 1
    elif prev_low is not None and last < prev_low:
        outside = -1
    words = "Above open" if last > o else "Below open" if last < o else "At open"
    place = _place(position)
    beyond = {1: " Above yesterday's high.", -1: " Below yesterday's low.", 0: " Inside yesterday's range."}[outside]
    return _factor("today", (_sign(last - o) + position + outside) / 3, f"{words}, {place}",
                   f"O {o:,.2f}  H {h:,.2f}  L {l:,.2f}  Last {last:,.2f}. {words}, in the {place} of today's range.{beyond}")


def gap(opening, previous_close):
    pct = (opening / previous_close - 1) * 100
    word = "Gap up" if pct >= FLAT_GAP else "Gap down" if pct <= -FLAT_GAP else "Flat open"
    return _factor("gap", _clip(pct / GAP_FULL), f"{pct:+.2f}%",
                   f"{word}: previous close {previous_close:,.2f} to open {opening:,.2f} ({pct:+.2f}%).")


def setup(prev, gap_pct):
    """A day's description for finding similar days.  prev: (date, o, h, l, c)."""
    _, o, h, l, c = prev
    kind = ANY_GAP if gap_pct is None else "up" if gap_pct >= FLAT_GAP else "down" if gap_pct <= -FLAT_GAP else "flat"
    return ("green" if c >= o else "red", _place(where_in_range(c, h, l)), kind)


def add_day(table, prev, day):
    """Count one past day in the table of {setup: [days that closed above their open, days]}."""
    if not prev[4] or not day[1]:
        return
    gap_pct = (day[1] / prev[4] - 1) * 100
    for key in (setup(prev, gap_pct), setup(prev, None)):
        table[key][0] += day[4] > day[1]
        table[key][1] += 1


def similar_days(rows):
    table = defaultdict(lambda: [0, 0])
    for prev, day in zip(rows, rows[1:]):
        add_day(table, prev, day)
    return table


def history(table, prev, gap_pct):
    colour, place, kind = key = setup(prev, gap_pct)
    ups, n = table.get(key, (0, 0))
    described = f"previous day {colour}, closed {place}" + ("" if kind == ANY_GAP else f", {'flat open' if kind == 'flat' else 'gap ' + kind}")
    if n < MIN_SIMILAR:
        return _factor("history", None, "Too few days", f"Only {n} past days like this ({described}), too few to say.")
    rate = ups / n * 100
    return _factor("history", _clip((rate - 50) / HIST_FULL), f"{rate:.0f}% of {n}",
                   f"On {n} past days like this ({described}), the index closed above its open {rate:.0f}% of the time.")


def world(rows, oldest=None):
    """rows: the Global strip.  oldest: leave out a market whose last session is before this date."""
    rows = [r for r in rows if r.get("pct") is not None and not r.get("stale")]
    old = [r for r in rows if oldest and r.get("session") and date.fromisoformat(r["session"]) < oldest]
    rows = [r for r in rows if r not in old]
    left_out = "".join(f" Left out, last session too old: {r['short']} ({r['session']})." for r in old)
    if not rows:
        return _factor("global", None, "No data", "Global markets are unavailable right now." + left_out)
    mean = sum(r["pct"] for r in rows) / len(rows)
    up = sum(r["pct"] > 0 for r in rows)
    detail = ", ".join(f"{r['short']} {r['pct']:+.2f}%" for r in rows)
    return _factor("global", _clip(mean / GLOBAL_FULL), f"{up} of {len(rows)} up, {mean:+.2f}%", f"Average {mean:+.2f}%: {detail}.{left_out}")


def verdict(factors):
    known = [f["score"] for f in factors if f["score"] is not None]
    total = sum(known) / len(known) if known else 0.0
    side = _sign(total)
    return {"verdict": "Bullish" if side > 0 else "Bearish" if side < 0 else "Neutral", "score": round(total, 3),
            "agree": sum(_sign(s) == side for s in known), "known": len(known)}


# ---------------------------------------------------------------- reading the data
def before_open(now):
    return now.time() < MARKET_OPEN and get_market_status(now) not in ("WEEKEND", "HOLIDAY")


def _previous_weekday(day):
    day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def top_up(indices, now):
    """Add the daily candles the table is missing, once a day."""
    flag = TOPPED_KEY.format(now.date())
    if cache.get(flag):
        return
    wanted = _previous_weekday(now.date())
    ok = True
    for index in indices:
        last = IndexDaily.objects.filter(index=index).order_by("-date").values_list("date", flat=True).first()
        if last is not None and last >= wanted:
            continue
        try:
            index_data.update_daily(index, now)
        except Exception as exc:                          # the page still works from what is stored
            logger.warning("Index outlook: could not update %s: %s", index.key, exc)
            ok = False
            break
    cache.set(flag, True, 24 * 3600 if ok else 600)


def _feed_day(quote, fallback):
    try:
        return datetime.strptime(quote.get("exchFeedTime") or "", "%d-%b-%Y %H:%M:%S").date()
    except ValueError:
        return fallback


def read_quotes(indices):
    from paper_trading import angel

    out = {}
    for exchange in sorted({i.exchange for i in indices}):
        got = angel.quotes([i.angel_token for i in indices if i.exchange == exchange], exchange)
        out.update({i.key: got[i.angel_token] for i in indices if i.exchange == exchange and i.angel_token in got})
    return out


def _past_rows(index, before):
    return list(IndexDaily.objects.filter(index=index, date__gte=HISTORY_FROM, date__lt=before).order_by("date")
                .values_list("date", "open", "high", "low", "close"))


def read_index(index, quote, world_factor, now):
    pre = before_open(now)
    day = now.date() if pre else _feed_day(quote, now.date())
    rows = _past_rows(index, day)
    o, h, l, last, prev_close = (float(quote.get(k) or 0) for k in ("open", "high", "low", "ltp", "close"))
    prev = rows[-1] if rows else None
    started = not pre and o > 0
    prev_close = prev_close or (prev[4] if prev else 0)
    gap_pct = (o / prev_close - 1) * 100 if started and prev_close else None

    pending = "Before the 09:15 open." if pre else "No trade yet."
    own = [
        previous_day(*prev[1:], prev[0]) if prev else _factor("previous", None, "No data", "No stored candle before this session."),
        today(o, h, l, last, prev[2] if prev else None, prev[3] if prev else None) if started else _factor("today", None, "Waiting", pending),
        gap(o, prev_close) if gap_pct is not None else _factor("gap", None, "Waiting", pending),
        history(similar_days(rows), prev, gap_pct) if prev else _factor("history", None, "No data", "No stored history."),
    ]
    if index.key in INVERTED:
        for f in own:
            if f["score"] is not None:
                f["score"] = -f["score"]
    factors = own + [world_factor]
    return {"key": index.key, "name": index.name, "last": last, "pct": quote.get("percentChange"), "day": day.isoformat(),
            "inverted": index.key in INVERTED, "factors": factors, **verdict(factors)}


def build(now=None):
    from .indices import global_indices

    now = now or datetime.now(IST)
    indices = list(MarketIndex.objects.order_by("sort"))
    top_up(indices, now)
    quotes = read_quotes(indices)
    world_factor = world(global_indices()["indices"], _previous_weekday(now.date()))
    rows = [read_index(i, quotes[i.key], world_factor, now) for i in indices if i.key in quotes]
    status = get_market_status(now)
    return {"indices": rows, "bullish": sum(r["verdict"] == "Bullish" for r in rows), "bearish": sum(r["verdict"] == "Bearish" for r in rows),
            "mode": "pre_open" if before_open(now) else "live" if status == "OPEN" else "closed",
            "day": rows[0]["day"] if rows else None, "global": world_factor["text"], "missing": [i.name for i in indices if i.key not in quotes],
            "as_of": now.isoformat()}


def outlook():
    hit = cache.get(CACHE_KEY)
    if hit is not None:
        return hit
    with _lock:
        hit = cache.get(CACHE_KEY)
        if hit is not None:
            return hit
        try:
            out = build()
        except Exception as exc:
            logger.warning("Index outlook failed: %s", exc)
            good = cache.get(GOOD_KEY)
            if good is None:
                raise
            return {**good, "stale": True}
        cache.set(CACHE_KEY, out, CACHE_SECONDS)
        if out["indices"]:
            cache.set(GOOD_KEY, out, 24 * 3600)
        return out


@require_GET
def outlook_api(request):
    try:
        return JsonResponse(outlook())
    except Exception:
        return JsonResponse({"error": "The index outlook is unavailable right now."}, status=502)

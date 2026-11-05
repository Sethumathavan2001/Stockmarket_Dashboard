"""
Index Analysis page: every stored index (market/index_data.py), day by day from its 5-minute candles, and its
past expiry days compared with ordinary days.

  GET  /index-analysis/                         the page
  GET  /api/index/meta/                         indices, their stored coverage, next listed expiries, update progress
  GET  /api/index/day/?date=&index=             one day: the index's 5-minute candles + every index's stats that day
  GET  /api/index/expiries/?index=              the index's expiry days with stats, expiry vs other days, by weekday
  GET/POST /api/index/update/                   progress / start a background update of candles + expiries

Day stats (IST, from the 5-minute candles; open / close and the previous close from the daily candles when stored):
change % (close vs previous close), gap % (open vs previous close), range % ((high - low) / open), body %
(close vs open), time of the day's high and low, first 15 minutes (09:15 -> 09:30) and last hour
(14:30 -> close) moves, where the close sits in the day's range, and a day type:
Trend up / Trend down (moved one way and closed in the top / bottom quarter), Range (body under a quarter of
the range), Up but faded / Down but recovered (the rest).
"""
from collections import defaultdict
from datetime import date, datetime, timedelta

from django.core.cache import cache
from django.db.models import Count, Max, Min
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET

from prediction.market_time import IST

from . import index_data
from .models import IndexDaily, IndexExpiry, IndexFiveMinute, MarketIndex

IST_MIN = 330                                   # IST = UTC + 5:30
FIRST15_END, LAST_HOUR = 9 * 60 + 25, 14 * 60 + 30
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _hm(m):
    return f"{m // 60:02d}:{m % 60:02d}"


def _pct(a, b):
    return None if a is None or not b else round((a / b - 1) * 100, 3)


def day_stats(bars, prev_close=None, day_open=None, day_close=None):
    """bars: [(minute of day IST, o, h, l, c)] in time order."""
    if not bars:
        return None
    o = day_open or bars[0][1]
    c = day_close or bars[-1][4]
    h = max(b[2] for b in bars)
    lo = min(b[3] for b in bars)
    hi_t = next(b[0] for b in bars if b[2] == h)
    lo_t = next(b[0] for b in bars if b[3] == lo)
    first = next((b[4] for b in bars if b[0] == FIRST15_END), None)
    last = next((b for b in bars if b[0] >= LAST_HOUR), None)
    rng, body = (h - lo) / o * 100 if o else 0, (c / o - 1) * 100 if o else 0
    pos = (c - lo) / (h - lo) if h > lo else 0.5
    if h <= lo:
        kind = "Flat"
    elif abs(body) < 0.25 * rng:
        kind = "Range"
    elif body > 0:
        kind = "Trend up" if pos >= 0.75 else "Up but faded"
    else:
        kind = "Trend down" if pos <= 0.25 else "Down but recovered"
    return {"open": round(o, 2), "high": round(h, 2), "low": round(lo, 2), "close": round(c, 2),
            "prev_close": None if prev_close is None else round(prev_close, 2),
            "change": _pct(c, prev_close), "gap": _pct(o, prev_close), "range": round(rng, 3), "body": round(body, 3),
            "high_time": _hm(hi_t), "low_time": _hm(lo_t), "first15": _pct(first, o),
            "last_hour": _pct(c, last[1]) if last else None, "close_pos": round(pos, 3), "type": kind,
            "bars": len(bars)}


def _minute(ts):
    return (ts.hour * 60 + ts.minute + IST_MIN) % 1440


def _daily_map(index):
    return {d: (o, c) for d, o, c in IndexDaily.objects.filter(index=index).values_list("date", "open", "close")}


def day_table(index):
    """{date: stats} for every day with 5-minute candles - cached until new candles arrive."""
    agg = IndexFiveMinute.objects.filter(index=index).aggregate(n=Count("id"), last=Max("timestamp"))
    key = f"idx:days:{index.key}:{agg['n']}:{agg['last']}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    daily = _daily_map(index)
    days = defaultdict(list)
    for d, ts, o, h, lo, c in (IndexFiveMinute.objects.filter(index=index).order_by("timestamp")
                               .values_list("date", "timestamp", "open", "high", "low", "close").iterator(chunk_size=5000)):
        days[d].append((_minute(ts), o, h, lo, c))
    out, prev = {}, None
    for d in sorted(days):
        o, c = daily.get(d, (None, None))
        prev_close = prev
        out[d] = day_stats(days[d], prev_close, o, c)
        prev = c or days[d][-1][4]
    cache.set(key, out, 6 * 3600)
    return out


def expiry_kinds(dates):
    """{date: "monthly" | "weekly"} - monthly = the last expiry of its calendar month."""
    last = {}
    for d in dates:
        k = (d.year, d.month)
        last[k] = max(last.get(k, d), d)
    return {d: "monthly" if last[(d.year, d.month)] == d else "weekly" for d in dates}


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(rows):
    rows = [r for r in rows if r]
    if not rows:
        return None
    ch = [r["change"] for r in rows if r["change"] is not None]
    return {"days": len(rows), "avg_range": _avg([r["range"] for r in rows]),
            "avg_abs_change": _avg([abs(x) for x in ch]),
            "up_pct": round(100 * sum(1 for x in ch if x > 0) / len(ch), 1) if ch else None,
            "avg_abs_last_hour": _avg([abs(r["last_hour"]) for r in rows if r["last_hour"] is not None]),
            "avg_abs_first15": _avg([abs(r["first15"]) for r in rows if r["first15"] is not None]),
            "trend_pct": round(100 * sum(1 for r in rows if r["type"] in ("Trend up", "Trend down")) / len(rows), 1),
            "range_pct": round(100 * sum(1 for r in rows if r["type"] == "Range") / len(rows), 1)}


def _index(request):
    key = (request.GET.get("index") or "NIFTY").upper()
    idx = MarketIndex.objects.filter(key=key).first()
    if idx is None:
        index_data.sync_catalog()
        idx = MarketIndex.objects.filter(key=key).first()
    return idx


# ---------------------------------------------------------------- views
@require_GET
@ensure_csrf_cookie
def page(request):
    index_data.sync_catalog()
    return render(request, "market/index_analysis.html", {})


def _meta():
    today = datetime.now(IST).date()
    upcoming = index_data._scan_state().get("upcoming", {})
    five = {r["index"]: r for r in IndexFiveMinute.objects.values("index").annotate(first=Min("date"), last=Max("date"))}
    daily = {r["index"]: r for r in IndexDaily.objects.values("index").annotate(first=Min("date"), last=Max("date"))}
    exp = {r["index"]: r for r in IndexExpiry.objects.filter(date__lte=today).values("index")
           .annotate(n=Count("id"), first=Min("date"), last=Max("date"))}
    out = []
    for i in MarketIndex.objects.all():
        f, d, e = five.get(i.pk, {}), daily.get(i.pk, {}), exp.get(i.pk, {})
        nxt = [x for x in upcoming.get(i.option_symbol, []) if x >= today.isoformat()] if i.option_symbol else []
        out.append({"key": i.key, "name": i.name, "exchange": i.exchange, "options": bool(i.option_symbol),
                    "five_from": f.get("first"), "five_to": f.get("last"), "daily_from": d.get("first"),
                    "daily_to": d.get("last"), "expiries": e.get("n", 0), "expiry_from": e.get("first"),
                    "expiry_to": e.get("last"), "next_expiries": nxt[:3]})
    return out


@require_GET
def meta(request):
    return JsonResponse({"indices": _meta(), "update": index_data.progress(),
                         "accounts": bool(index_data._accounts())})


@require_GET
def day(request):
    idx = _index(request)
    if idx is None:
        return JsonResponse({"error": "Unknown index."}, status=404)
    nifty = MarketIndex.objects.filter(key="NIFTY").first() or idx
    days = IndexFiveMinute.objects.filter(index=nifty)
    try:
        d = date.fromisoformat(request.GET.get("date") or "")
    except ValueError:
        d = days.aggregate(m=Max("date"))["m"]
    if d is None:
        return JsonResponse({"error": "No 5-minute index data stored yet - press Update data."}, status=404)
    if not IndexFiveMinute.objects.filter(index=nifty, date=d).exists():   # holiday / weekend: the trading day before
        before = days.filter(date__lt=d).aggregate(m=Max("date"))["m"]
        d = before or days.filter(date__gt=d).aggregate(m=Min("date"))["m"] or d
    prev_day = days.filter(date__lt=d).aggregate(m=Max("date"))["m"]
    next_day = days.filter(date__gt=d).aggregate(m=Min("date"))["m"]

    exp_today = {}
    for e in IndexExpiry.objects.filter(date=d).select_related("index"):
        later = IndexExpiry.objects.filter(index=e.index, date__gt=d, date__year=d.year, date__month=d.month).exists()
        exp_today[e.index.key] = "weekly" if later else "monthly"
    sel_exp = IndexExpiry.objects.filter(index=idx, date__lte=datetime.now(IST).date())

    rows, series, candles, sel, sel_prev = [], {}, [], None, None
    for i in MarketIndex.objects.all():
        bars = list(IndexFiveMinute.objects.filter(index=i, date=d).order_by("timestamp")
                    .values_list("timestamp", "open", "high", "low", "close"))
        if not bars:
            rows.append({"key": i.key, "name": i.name, "stats": None, "expiry": exp_today.get(i.key)})
            continue
        prev = IndexDaily.objects.filter(index=i, date__lt=d).order_by("-date").first()
        today_d = IndexDaily.objects.filter(index=i, date=d).first()
        prev_close = prev.close if prev else None
        st = day_stats([(_minute(t), o, h, lo, c) for t, o, h, lo, c in bars], prev_close,
                       today_d.open if today_d else None, today_d.close if today_d else None)
        rows.append({"key": i.key, "name": i.name, "stats": st, "expiry": exp_today.get(i.key)})
        base = prev_close or bars[0][1]
        series[i.key] = [[int(t.timestamp()), round((c / base - 1) * 100, 3)] for t, _, _, _, c in bars]
        if i.pk == idx.pk:
            sel = st
            candles = [[int(t.timestamp()), o, h, lo, c] for t, o, h, lo, c in bars]
            sel_prev = {"high": prev.high, "low": prev.low, "close": prev.close, "date": prev.date} if prev else None
    return JsonResponse({
        "date": d, "weekday": WEEKDAYS[d.weekday()], "index": idx.key, "name": idx.name, "prev_day": prev_day,
        "next_day": next_day, "candles": candles, "stats": sel, "prev": sel_prev,
        "expiry": exp_today.get(idx.key),
        "prev_expiry": sel_exp.filter(date__lt=d).aggregate(m=Max("date"))["m"],
        "next_expiry": sel_exp.filter(date__gt=d).aggregate(m=Min("date"))["m"],
        "indices": rows, "series": series, "expiries_today": exp_today,
    })


@require_GET
def expiries(request):
    idx = _index(request)
    if idx is None:
        return JsonResponse({"error": "Unknown index."}, status=404)
    if not idx.option_symbol:
        return JsonResponse({"index": idx.key, "name": idx.name, "rows": [], "options": False})
    today = datetime.now(IST).date()
    exp = list(IndexExpiry.objects.filter(index=idx, date__lte=today).values_list("date", "source"))
    kinds = expiry_kinds([d for d, _ in exp])
    table = day_table(idx)
    exp_days = {d for d, _ in exp}
    rows = [{"date": d, "weekday": WEEKDAYS[d.weekday()], "kind": kinds[d], "source": src, "stats": table.get(d)}
            for d, src in sorted(exp, reverse=True)]
    first_exp = min(exp_days) if exp_days else None
    others = [s for d, s in table.items() if d not in exp_days and first_exp and d >= first_exp]
    weekday = []
    for w in range(5):
        weekday.append({"weekday": WEEKDAYS[w],
                        "expiry": summarize([table[d] for d in exp_days if d in table and d.weekday() == w]),
                        "other": summarize([s for d, s in table.items() if d.weekday() == w and d not in exp_days
                                            and first_exp and d >= first_exp])})
    return JsonResponse({
        "index": idx.key, "name": idx.name, "options": True, "rows": rows,
        "summary": {"expiry": summarize([table.get(d) for d in exp_days]),
                    "weekly": summarize([table.get(d) for d in exp_days if kinds[d] == "weekly"]),
                    "monthly": summarize([table.get(d) for d in exp_days if kinds[d] == "monthly"]),
                    "other": summarize(others)},
        "weekday": weekday,
        "next_expiries": [x for x in index_data._scan_state().get("upcoming", {}).get(idx.option_symbol, [])
                          if x >= today.isoformat()][:3],
    })


def update(request):
    if request.method == "POST":
        started = index_data.start_update()
        return JsonResponse({"started": started, "update": index_data.progress()}, status=202 if started else 409)
    return JsonResponse({"update": index_data.progress()})

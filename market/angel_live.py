"""
Angel One Live page: one stock's live price and candle chart, straight from Angel One (read only, no orders).

  GET  /angel-live/?company=INFY.NS                  the page
  GET  /api/angel-live/candles/?code=&interval=      candles for the chart, the forming one included
  GET  /api/angel-live/quote/?code=                  last price, day's open / high / low, volume, market status

The page asks for the quote every 2 seconds and moves the forming candle with it, and reloads the candles every
10 seconds so they stay exactly Angel One's.  Both are cached for a moment, so extra tabs add no extra calls.
Requests go through the first Angel One account in .env that answers.  Candle times are sent as seconds with
IST added, because the chart library shows them as UTC.
"""
from datetime import datetime, timedelta

from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from paper_trading import angel
from prediction.market_time import IST, get_market_status

from .models import Company

IST_SECONDS = 19800
QUOTE_SECONDS = 1.5
CANDLE_SECONDS = 8
ERROR_SECONDS = 4
# Angel One interval: label, seconds per candle, days of history to load (well inside Angel One's limits)
INTERVALS = {
    "ONE_MINUTE": ("1m", 60, 5),
    "FIVE_MINUTE": ("5m", 300, 20),
    "FIFTEEN_MINUTE": ("15m", 900, 45),
    "THIRTY_MINUTE": ("30m", 1800, 60),
    "ONE_HOUR": ("1h", 3600, 120),
    "ONE_DAY": ("1D", 86400, 730),
}
STATUS = {"OPEN": "Market open", "PRE_MARKET": "Pre-open", "POST_MARKET": "Market closed", "CLOSED": "Market closed",
          "WEEKEND": "Market closed (weekend)", "HOLIDAY": "Market closed (holiday)"}


def _cached(key, seconds, fetch):
    """fetch() kept for a moment.  A failure is kept briefly too, so a polling page doesn't hammer a bad connection."""
    hit = cache.get(key)
    if hit is None:
        try:
            hit = fetch()
            cache.set(key, hit, seconds)
        except angel.AngelError as exc:
            hit = {"error": str(exc)}
            cache.set(key, hit, ERROR_SECONDS)
    return JsonResponse(hit, status=502 if "error" in hit else 200)


def _shifted(moment):
    return int(moment.timestamp()) + IST_SECONDS


def _candles(inst, interval):
    days = INTERVALS[interval][2]
    now = datetime.now(IST)
    params = {"exchange": "NSE", "symboltoken": inst["token"], "interval": interval,
              "fromdate": (now - timedelta(days=days)).strftime("%Y-%m-%d %H:%M"), "todate": now.strftime("%Y-%m-%d %H:%M")}
    acc, res = angel.read_any(lambda sc: sc.getCandleData(dict(params)))
    rows = [{"time": _shifted(datetime.fromisoformat(r[0])), "open": r[1], "high": r[2], "low": r[3], "close": r[4],
             "volume": r[5]} for r in (res or {}).get("data") or []]
    return {"symbol": inst["symbol"], "interval": interval, "seconds": INTERVALS[interval][1], "candles": rows,
            "account": angel.ACCOUNTS[acc]}


def _quote(inst):
    acc, res = angel.read_any(lambda sc: sc.getMarketData("FULL", {"NSE": [inst["token"]]}))
    fetched = ((res or {}).get("data") or {}).get("fetched") or []
    if not fetched:
        raise angel.AngelError("Angel One gave no quote for this stock.")
    q = fetched[0]
    traded = angel.trade_time(q)
    status = get_market_status()
    return {"symbol": inst["symbol"], "ltp": q.get("ltp"), "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
            "prev_close": q.get("close"), "change": q.get("netChange"), "pct": q.get("percentChange"),
            "volume": q.get("tradeVolume"), "avg": q.get("avgPrice"), "lower": q.get("lowerCircuit"),
            "upper": q.get("upperCircuit"), "traded": traded.strftime("%d %b, %H:%M:%S") if traded else None,
            "ts": _shifted(traded) if traded else None, "live": status == "OPEN", "status": STATUS.get(status, status),
            "account": angel.ACCOUNTS[acc]}


def _instrument(request):
    """(instrument, None) for ?code=, or (None, the error response)."""
    code = (request.GET.get("code") or "").strip().upper()
    if not code:
        return None, JsonResponse({"error": "Choose a stock."}, status=400)
    try:
        return angel.instrument(code), None
    except angel.AngelError as exc:
        return None, JsonResponse({"error": str(exc)}, status=400)


@require_GET
def page(request):
    return render(request, "market/angel_live.html", {
        "companies": Company.objects.order_by("code").values("code", "name"),
        "intervals": [(k, v[0]) for k, v in INTERVALS.items()],
        "company": (request.GET.get("company") or "").strip().upper()})


@require_GET
def candles(request):
    interval = request.GET.get("interval") or "ONE_MINUTE"
    if interval not in INTERVALS:
        return JsonResponse({"error": "Unknown interval."}, status=400)
    inst, bad = _instrument(request)
    if bad:
        return bad
    return _cached(f"angel_live:c:{inst['token']}:{interval}", CANDLE_SECONDS, lambda: _candles(inst, interval))


@require_GET
def quote(request):
    inst, bad = _instrument(request)
    if bad:
        return bad
    return _cached(f"angel_live:q:{inst['token']}", QUOTE_SECONDS, lambda: _quote(inst))

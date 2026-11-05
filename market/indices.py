"""
Live index strip shown above the navbar on every page.

  GET /api/market/indices/              ->  {"indices": [{name, short, key, last, change, pct, ...}], "market_status", "as_of"}
  GET /api/market/indices/<key>/chart/  ->  latest session's 5-minute candles + the previous session's OHLC (popup chart)

NSE indices come from NSE's all-indices feed (one request, real time); SENSEX is a BSE index and
comes from Yahoo (about 15 minutes delayed - marked "delayed").  Cached for 20 seconds so many
open tabs share one fetch; if a source fails, the last good values are served and marked stale.

  GET /api/market/global/               ->  {"indices": [...], "as_of"}   the world strip above the NSE one

GIFT Nifty comes from NSE's market status feed (the near-month future at NSE IX, trading almost round the clock);
NASDAQ, S&P 500, Nikkei 225 and Hang Seng from Yahoo daily candles.  A closed market shows its last session's
close and move - what matters before the Indian open.  Cached for 2 minutes.
"""
import logging
import threading
from datetime import datetime, time
from zoneinfo import ZoneInfo

from django.core.cache import cache
from django.http import Http404, JsonResponse
from django.views.decorators.http import require_GET

from prediction.market_time import IST, get_market_status

logger = logging.getLogger(__name__)

CACHE_KEY, LAST_GOOD_KEY = "market:indices", "market:indices:last_good"
CACHE_SECONDS = 20
NSE_INDICES = [                      # (name in NSE's feed, label on the strip)
    ("NIFTY 50", "NIFTY 50"),
    ("NIFTY BANK", "BANK NIFTY"),
    ("NIFTY FINANCIAL SERVICES", "FIN NIFTY"),
    ("NIFTY IT", "NIFTY IT"),
    ("NIFTY MIDCAP 100", "MIDCAP 100"),
    ("NIFTY SMALLCAP 100", "SMALLCAP 100"),
    ("INDIA VIX", "INDIA VIX"),
]
_lock = threading.Lock()
# strip label -> (url key, Yahoo symbol) for the popup charts
CHART_SYMBOLS = {
    "NIFTY 50": ("nifty50", "^NSEI"), "SENSEX": ("sensex", "^BSESN"), "BANK NIFTY": ("banknifty", "^NSEBANK"),
    "FIN NIFTY": ("finnifty", "NIFTY_FIN_SERVICE.NS"), "NIFTY IT": ("niftyit", "^CNXIT"),
    "MIDCAP 100": ("midcap100", "NIFTY_MIDCAP_100.NS"), "SMALLCAP 100": ("smallcap100", "^CNXSC"),
    "INDIA VIX": ("indiavix", "^INDIAVIX"),
    "US NASDAQ": ("nasdaq", "^IXIC"), "US S&P 500": ("sp500", "^GSPC"), "Japan Nikkei 225": ("nikkei", "^N225"),
    "Hong Kong Hang Seng": ("hangseng", "^HSI"),
}
KEY_TO = {key: (label, sym) for label, (key, sym) in CHART_SYMBOLS.items()}
GLOBAL_KEY, GLOBAL_GOOD_KEY = "market:global", "market:global:last_good"
GLOBAL_SECONDS = 120
GIFT = "GIFT Nifty"
WORLD = [                            # (label, exchange's time zone, session open, close)
    ("US NASDAQ", "America/New_York", time(9, 30), time(16, 0)),
    ("US S&P 500", "America/New_York", time(9, 30), time(16, 0)),
    ("Japan Nikkei 225", "Asia/Tokyo", time(9, 0), time(15, 30)),
    ("Hong Kong Hang Seng", "Asia/Hong_Kong", time(9, 30), time(16, 0)),
]
SYMBOL_TZ = {CHART_SYMBOLS[label][1]: zone for label, zone, _, _ in WORLD}
_global_lock = threading.Lock()


def _num(v):
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _nse():
    from news.fetch import _nse_json

    data = _nse_json("/api/allIndices")
    rows = {r.get("index"): r for r in data.get("data", [])}
    out = []
    for key, label in NSE_INDICES:
        r = rows.get(key)
        if not r:
            continue
        out.append({"name": key, "short": label, "last": _num(r.get("last")), "change": _num(r.get("variation")),
                    "pct": _num(r.get("percentChange")), "open": _num(r.get("open")), "high": _num(r.get("high")),
                    "low": _num(r.get("low")), "prev_close": _num(r.get("previousClose")),
                    "advances": _num(r.get("advances")), "declines": _num(r.get("declines")), "source": "NSE",
                    "delayed": False, "time": data.get("timestamp")})
    return out


def _sensex():
    from prediction.datasets import _yahoo

    raw = _yahoo("^BSESN", cache_seconds=60, period="5d", interval="1d")
    if raw is None or len(raw) < 2:
        return None
    last, prev = float(raw["Close"].iloc[-1]), float(raw["Close"].iloc[-2])
    day = raw.index[-1]
    if day.date() != datetime.now(IST).date():           # no row for today yet: show the last close
        prev = last
    return {"name": "S&P BSE SENSEX", "short": "SENSEX", "last": last, "change": last - prev,
            "pct": (last / prev - 1) * 100 if prev else 0.0, "open": float(raw["Open"].iloc[-1]),
            "high": float(raw["High"].iloc[-1]), "low": float(raw["Low"].iloc[-1]), "prev_close": prev,
            "source": "Yahoo", "delayed": True, "time": None}


def _gift():
    from news.fetch import _nse_json

    g = _nse_json("/api/marketStatus").get("giftnifty") or {}
    last, change = _num(g.get("LASTPRICE")), _num(g.get("DAYCHANGE"))
    if last is None or change is None:
        raise ValueError("NSE sent no GIFT Nifty price")
    return {"name": f"GIFT Nifty future, expiry {g.get('EXPIRYDATE')}", "short": GIFT, "last": last, "change": change,
            "pct": _num(g.get("PERCHANGE")) or 0.0, "prev_close": last - change, "source": "NSE", "delayed": False,
            "time": g.get("TIMESTMP"), "key": None}


def _world(label, zone, opens, closes):
    from prediction.datasets import _yahoo

    key, sym = CHART_SYMBOLS[label]
    raw = _yahoo(sym, cache_seconds=GLOBAL_SECONDS, period="5d", interval="1d")
    raw = None if raw is None else raw.dropna(subset=["Close"])
    if raw is None or len(raw) < 2:
        raise ValueError(f"Yahoo sent no {sym} candles")
    last, prev = float(raw["Close"].iloc[-1]), float(raw["Close"].iloc[-2])
    now, session = datetime.now(ZoneInfo(zone)), raw.index[-1]
    is_open = session.date() == now.date() and opens <= now.time() < closes
    return {"name": label, "short": label, "last": last, "change": last - prev, "pct": (last / prev - 1) * 100 if prev else 0.0,
            "open": float(raw["Open"].iloc[-1]), "high": float(raw["High"].iloc[-1]), "low": float(raw["Low"].iloc[-1]),
            "prev_close": prev, "source": "Yahoo", "delayed": is_open, "time": None, "key": key,
            "tag": None if is_open else f"{session.day} {session:%b} close"}


def global_indices():
    hit = cache.get(GLOBAL_KEY)
    if hit is not None:
        return hit
    with _global_lock:
        hit = cache.get(GLOBAL_KEY)
        if hit is not None:
            return hit
        good = {i["short"]: i for i in (cache.get(GLOBAL_GOOD_KEY) or {}).get("indices", [])}
        indices, errors = [], []
        for label, fetch in [(GIFT, _gift)] + [(w[0], lambda w=w: _world(*w)) for w in WORLD]:
            try:
                indices.append(fetch())
            except Exception as exc:                     # keep the last good value for this one
                logger.warning("%s failed: %s", label, exc)
                errors.append(label)
                if label in good:
                    indices.append(dict(good[label], stale=True))
        out = {"indices": indices, "as_of": datetime.now(IST).isoformat(), "errors": errors}
        cache.set(GLOBAL_KEY, out, GLOBAL_SECONDS)
        cache.set(GLOBAL_GOOD_KEY, {"indices": [i for i in indices if not i.get("stale")] + [
            i for i in good.values() if i["short"] in errors]}, 24 * 3600)
        return out


@require_GET
def global_api(request):
    return JsonResponse(global_indices())


def live_indices():
    hit = cache.get(CACHE_KEY)
    if hit is not None:
        return hit
    with _lock:                                          # one fetch at a time; others wait for its result
        hit = cache.get(CACHE_KEY)
        if hit is not None:
            return hit
        last_good = cache.get(LAST_GOOD_KEY) or {}
        indices, errors = [], []
        try:
            indices = _nse()
        except Exception as exc:                         # NSE refused / network - keep the last good values
            logger.warning("NSE indices failed: %s", exc)
            errors.append("NSE")
            indices = [dict(i, stale=True) for i in last_good.get("indices", []) if i.get("source") == "NSE"]
        try:
            sx = _sensex()
            if sx:
                indices.insert(1, sx)
        except Exception as exc:
            logger.warning("SENSEX failed: %s", exc)
            errors.append("SENSEX")
            old = next((i for i in last_good.get("indices", []) if i.get("short") == "SENSEX"), None)
            if old:
                indices.insert(1, dict(old, stale=True))
        for i in indices:
            i["key"] = CHART_SYMBOLS.get(i["short"], (None,))[0]
        out = {"indices": indices, "market_status": get_market_status(), "as_of": datetime.now(IST).isoformat(),
               "errors": errors}
        cache.set(CACHE_KEY, out, CACHE_SECONDS)
        if not errors:
            cache.set(LAST_GOOD_KEY, out, 24 * 3600)
        return out


@require_GET
def indices_api(request):
    return JsonResponse(live_indices())


def index_chart(key):
    """Latest session's 5-minute candles and the previous session's OHLC, from Yahoo (5 days of 5-minute candles)."""
    from prediction.datasets import _yahoo, _yahoo_frame

    label, sym = KEY_TO[key]
    ck = f"market:indexchart:{key}"
    hit = cache.get(ck)
    if hit is not None:
        return hit
    df = _yahoo_frame(_yahoo(sym, cache_seconds=60, period="5d", interval="5m", prepost=False))
    if df.empty:
        return {"key": key, "name": label, "candles": [], "previous": None}
    zone = ZoneInfo(SYMBOL_TZ.get(sym, "Asia/Kolkata"))             # a US session crosses midnight IST
    df["day"] = df["timestamp"].dt.tz_convert(zone).dt.date
    days = sorted(df["day"].unique())
    cur = df[df["day"] == days[-1]]
    prev = None
    if len(days) > 1:
        p = df[df["day"] == days[-2]]
        prev = {"date": days[-2].isoformat(), "open": float(p["open"].iloc[0]), "high": float(p["high"].max()),
                "low": float(p["low"].min()), "close": float(p["close"].iloc[-1])}
    out = {"key": key, "name": label, "symbol": sym, "session": days[-1].isoformat(),
           "is_today": days[-1] == datetime.now(zone).date(), "delayed": sym == "^BSESN" or sym in SYMBOL_TZ,
           "today": {"open": float(cur["open"].iloc[0]), "high": float(cur["high"].max()), "low": float(cur["low"].min()),
                     "last": float(cur["close"].iloc[-1]), "time": cur["timestamp"].iloc[-1].isoformat()},
           "previous": prev,
           "candles": [[ts.isoformat(), o, h, l, c] for ts, o, h, l, c in
                       zip(cur["timestamp"], cur["open"], cur["high"], cur["low"], cur["close"])]}
    cache.set(ck, out, 60)
    return out


@require_GET
def index_chart_api(request, key):
    if key not in KEY_TO:
        raise Http404("Unknown index")
    try:
        return JsonResponse(index_chart(key))
    except Exception as exc:                              # Yahoo down - the popup shows the message
        logger.warning("index chart %s failed: %s", key, exc)
        return JsonResponse({"error": "Chart data is unavailable right now."}, status=502)

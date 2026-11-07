"""
GIFT Nifty popup chart.

Nobody we can read gives GIFT Nifty candles: Yahoo and Angel One don't carry NSE IX, and TradingView's free embed refuses
the symbol.  NSE's market status feed does give its price, so while runserver is up a thread reads it once a minute and
keeps each new price in logs/gift_nifty.jsonl, and the chart builds 5-minute candles from those.  The feed only moves
every few minutes, so a candle holds one to three prices: its high and low miss whatever happened between them, and
there are gaps while the server is off.

Until a session has MIN_CANDLES recorded candles the chart shows the NSE NIFTY future of the same expiry from Angel One.
That is a different contract which trades in the Indian day session only, so the popup names it as a stand-in.

A GIFT day runs from DAY_START_HOUR IST to the same hour next morning: its evening session ends about 02:45.
"""
import json
import logging
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
from django.conf import settings
from django.core.cache import cache

from prediction.market_time import IST

from . import indices

logger = logging.getLogger(__name__)

FILE = settings.BASE_DIR / "logs" / "gift_nifty.jsonl"
POLL_SECONDS = 60
KEEP_DAYS = 10
MIN_CANDLES = 12
DAY_START_HOUR = 3
FUTURE_DAYS = 5
CHART_SECONDS = 60
RECORDED_NOTE = "built from NSE's GIFT price, which updates every few minutes, so highs and lows are approximate"

_started = threading.Event()


# ---------------------------------------------------------------- recording
def _samples():
    """Recorded prices, oldest first."""
    if not FILE.exists():
        return []
    out = []
    for line in FILE.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:                                   # half a line, from a stop in mid-write
            logger.info("GIFT Nifty: skipped an unreadable line")
    return out


def record(quote, last_ts=None):
    """Append a quote from indices._gift() unless the exchange's time hasn't moved.  Returns the time it stands at."""
    ts = datetime.strptime(quote["time"], "%d-%b-%Y %H:%M").replace(tzinfo=IST).isoformat()
    if ts == last_ts:
        return ts
    FILE.parent.mkdir(exist_ok=True)
    with FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": ts, "last": quote["last"], "ref": quote["prev_close"]}) + "\n")
    return ts


def _prune():
    cutoff = (datetime.now(IST) - timedelta(days=KEEP_DAYS)).isoformat()
    rows = _samples()
    kept = [r for r in rows if r["ts"] >= cutoff]
    if len(kept) != len(rows):
        FILE.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    return kept


def _loop():
    rows = _prune()
    last_ts = rows[-1]["ts"] if rows else None
    while True:
        try:
            last_ts = record(indices._gift(), last_ts)
        except Exception as exc:                             # NSE refused or the network dropped: try again next minute
            logger.info("GIFT Nifty price not recorded: %s", exc)
        time.sleep(POLL_SECONDS)


def start():
    if not _started.is_set():
        _started.set()
        threading.Thread(target=_loop, name="gift-nifty", daemon=True).start()


# ---------------------------------------------------------------- chart
def _gift_day(ts):
    return (ts - pd.Timedelta(hours=DAY_START_HOUR)).dt.date


def _recorded_frame(rows):
    """Five-minute candles from the recorded prices, with their GIFT day."""
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["ts"]).dt.tz_convert(IST).dt.floor("5min")
    df = df.groupby("timestamp")["last"].agg(open="first", high="max", low="min", close="last").reset_index()
    df["day"] = _gift_day(df["timestamp"])
    return df


def _future_token(symbol):
    from paper_trading import angel

    key = f"market:gift:token:{symbol}"
    token = cache.get(key)
    if token is None:
        rows = angel.read_any(lambda sc: sc.searchScrip("NFO", symbol))[1].get("data") or []
        token = next((r["symboltoken"] for r in rows if r.get("tradingsymbol") == symbol), None)
        if token is None:
            raise ValueError(f"Angel One has no {symbol}")
        cache.set(key, token, 24 * 3600)
    return token


def _future_frame(symbol):
    """Angel One's 5-minute candles of the NSE future, with their date."""
    from paper_trading import angel

    now = datetime.now(IST)
    params = {"exchange": "NFO", "symboltoken": _future_token(symbol), "interval": "FIVE_MINUTE",
              "fromdate": (now - timedelta(days=FUTURE_DAYS)).strftime("%Y-%m-%d %H:%M"), "todate": now.strftime("%Y-%m-%d %H:%M")}
    rows = (angel.read_any(lambda sc: sc.getCandleData(dict(params)))[1] or {}).get("data") or []
    df = pd.DataFrame([r[:5] for r in rows], columns=["timestamp", "open", "high", "low", "close"])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(IST)
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
        df["day"] = df["timestamp"].dt.date
    return df


def _stand_in():
    expiry = indices._gift()["name"].rsplit(" ", 1)[-1]                  # "GIFT Nifty future, expiry 23-Nov-2026"
    when = datetime.strptime(expiry, "%d-%b-%Y")
    symbol = f"NIFTY{when:%d%b%y}FUT".upper()
    df = _future_frame(symbol)
    if df.empty:
        return None
    label = f"Stand-in, not GIFT: NSE NIFTY {when:%b} future, day session only"
    return {**indices.session_chart(df, datetime.now(IST).date()), "symbol": symbol, "stand_in": True, "note": label}


def _recorded(rows):
    today = (datetime.now(IST) - timedelta(hours=DAY_START_HOUR)).date()
    return {**indices.session_chart(_recorded_frame(rows), today), "ref": rows[-1]["ref"], "note": RECORDED_NOTE}


def chart():
    """The popup's data: recorded GIFT candles, or the stand-in while the latest session has too few."""
    hit = cache.get("market:gift:chart")
    if hit is not None:
        return hit
    rows = _samples()
    own = _recorded(rows) if rows else None
    out = own
    if own is None or len(own["candles"]) < MIN_CANDLES:
        try:
            out = _stand_in() or own
        except Exception as exc:                             # Angel One unavailable: what was recorded is still real
            logger.warning("GIFT Nifty stand-in failed: %s", exc)
    if out is None:
        return {"key": indices.GIFT_KEY, "name": indices.GIFT, "candles": [], "previous": None,
                "error": "No GIFT Nifty candles yet. Prices are recorded while the server runs."}
    out = {"key": indices.GIFT_KEY, "name": indices.GIFT, "delayed": False, **out}
    cache.set("market:gift:chart", out, CHART_SECONDS)
    return out

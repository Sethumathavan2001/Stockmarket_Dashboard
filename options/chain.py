"""
Live option data from NSE.

  underlyings()                 every F&O underlying with its lot size  (NSE fo_mktlots.csv, refreshed daily)
  expiries(symbol)              open expiries                           (/api/option-chain-contract-info)
  chain(symbol, expiry)         the full chain: LTP, bid / ask, OI, change in OI, volume, IV per strike and side
  quote(symbol, expiry, strike, type)

The chain is cached a few seconds (several tabs and the watcher share one download); when NSE refuses,
the last good copy is returned for a while, marked stale.
"""
import csv
import io
import json
import logging
import threading
from datetime import date, datetime, timedelta
from pathlib import Path

from django.conf import settings
from django.core.cache import cache

from prediction.market_time import IST

from . import greeks

logger = logging.getLogger(__name__)

LOTS_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
LOTS_FILE = Path(settings.BASE_DIR) / "logs" / "fo_lots.json"
INDICES = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50")
INDEX_NAMES = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK", "FINNIFTY": "NIFTY FIN SERVICE",
               "MIDCPNIFTY": "NIFTY MIDCAP SELECT", "NIFTYNXT50": "NIFTY NEXT 50"}
YAHOO = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
         "MIDCPNIFTY": "NIFTY_MID_SELECT.NS", "NIFTYNXT50": "^NSMIDCP"}
CHAIN_SECONDS = 3            # fresh enough for a page polling every few seconds
STALE_SECONDS = 600          # last good chain kept this long when NSE refuses
EXPIRY_SECONDS = 600
EXPIRY_TIME = (15, 30)

_lots_lock = threading.Lock()
_lots_mem = {"data": None}


class ChainError(Exception):
    pass


def _nse(path):
    from news.fetch import SourceError, _nse_json
    try:
        return _nse_json(path)
    except SourceError as exc:
        raise ChainError(str(exc)) from exc
    except Exception as exc:                                  # network
        raise ChainError(f"NSE unreachable ({type(exc).__name__})") from exc


def is_index(symbol):
    return symbol in INDICES


def expiry_dt(expiry):
    """Contracts expire at 15:30 IST on the expiry date."""
    return datetime(expiry.year, expiry.month, expiry.day, *EXPIRY_TIME, tzinfo=IST)


# ---------------------------------------------------------------- lot sizes / underlyings
def _download_lots():
    from curl_cffi import requests as cr
    r = cr.get(LOTS_URL, impersonate="chrome", timeout=20)
    if r.status_code != 200:
        raise ChainError(f"lot sizes: HTTP {r.status_code}")
    rows = list(csv.reader(io.StringIO(r.text)))
    head = [h.strip() for h in rows[0]]
    months = head[2:]
    out, section = {}, "index"
    for row in rows[1:]:
        row = [c.strip() for c in row]
        if len(row) < 3 or not row[1]:
            continue
        if row[1].upper() == "SYMBOL":                       # "Derivatives on Individual Securities" header
            section = "stock"
            continue
        lots = {m: int(v) for m, v in zip(months, row[2:]) if m and v.isdigit()}
        if lots:
            out[row[1]] = {"symbol": row[1], "name": INDEX_NAMES.get(row[1], row[0].title()), "kind": section, "lots": lots}
    return {"as_of": datetime.now(IST).isoformat(), "underlyings": out}


def _lots():
    with _lots_lock:
        data = _lots_mem["data"]
        if data is None and LOTS_FILE.exists():
            try:
                data = json.loads(LOTS_FILE.read_text(encoding="utf-8"))
            except ValueError:
                data = None
        stale = not data or datetime.fromisoformat(data["as_of"]).date() < datetime.now(IST).date()
        if stale:
            try:
                data = _download_lots()
                LOTS_FILE.parent.mkdir(exist_ok=True)
                LOTS_FILE.write_text(json.dumps(data), encoding="utf-8")
            except Exception as exc:
                logger.warning("options: lot sizes not refreshed: %s", exc)
                if not data:
                    raise ChainError("Lot sizes unavailable (NSE archive not reachable).") from exc
        _lots_mem["data"] = data
        return data["underlyings"]


def underlyings():
    """[{symbol, name, kind: index|stock, lot}] - indices first (in INDICES order), then stocks A-Z."""
    u = _lots()
    out = []
    for sym, v in u.items():
        out.append({"symbol": sym, "name": v["name"], "kind": v["kind"], "lot": next(iter(v["lots"].values()))})
    order = {s: i for i, s in enumerate(INDICES)}
    out.sort(key=lambda x: (x["kind"] != "index", order.get(x["symbol"], 99), x["symbol"]))
    return [x for x in out if x["kind"] == "stock" or x["symbol"] in INDICES]


def lot_size(symbol, expiry):
    u = _lots().get(symbol)
    if not u:
        raise ChainError(f"{symbol} has no options on NSE.")
    return u["lots"].get(expiry.strftime("%b-%y").upper()) or next(iter(u["lots"].values()))


# ---------------------------------------------------------------- expiries / chain
def expiries(symbol):
    key = f"opt:exp:{symbol}"
    hit = cache.get(key)
    if hit is None:
        d = _nse(f"/api/option-chain-contract-info?symbol={symbol}")
        hit = sorted(datetime.strptime(x, "%d-%b-%Y").date() for x in d.get("expiryDates", []))
        cache.set(key, hit, EXPIRY_SECONDS)
    now = datetime.now(IST)
    return [e for e in hit if expiry_dt(e) > now]


def _side(x):
    if not x:
        return None
    f = lambda k: float(x.get(k) or 0)
    return {"ltp": f("lastPrice"), "chg": f("change"), "pchg": f("pChange"), "bid": f("buyPrice1"),
            "bid_qty": int(x.get("buyQuantity1") or 0), "ask": f("sellPrice1"), "ask_qty": int(x.get("sellQuantity1") or 0),
            "oi": int(x.get("openInterest") or 0), "chg_oi": int(x.get("changeinOpenInterest") or 0),
            "vol": int(x.get("totalTradedVolume") or 0), "iv": f("impliedVolatility")}


def _download_chain(symbol, expiry):
    kind = "Indices" if is_index(symbol) else "Equity"
    d = _nse(f"/api/option-chain-v3?type={kind}&symbol={symbol}&expiry={expiry.strftime('%d-%b-%Y')}")
    rec = d.get("records") or {}
    rows = []
    for r in rec.get("data", []):
        rows.append({"strike": float(r["strikePrice"]), "CE": _side(r.get("CE")), "PE": _side(r.get("PE"))})
    if not rows:
        raise ChainError(f"NSE returned no {symbol} contracts for {expiry:%d %b %Y}.")
    rows.sort(key=lambda r: r["strike"])
    try:
        ts = datetime.strptime(rec.get("timestamp", ""), "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST).isoformat()
    except ValueError:
        ts = None
    return {"symbol": symbol, "expiry": expiry.isoformat(), "spot": float(rec.get("underlyingValue") or 0),
            "timestamp": ts, "rows": rows, "fetched_at": datetime.now(IST).isoformat(), "stale": False}


def chain(symbol, expiry, max_age=CHAIN_SECONDS):
    """The chain (cached max_age seconds). Falls back to the last good copy (stale=True) when NSE refuses."""
    key, good = f"opt:chain:{symbol}:{expiry.isoformat()}", f"opt:chain-good:{symbol}:{expiry.isoformat()}"
    hit = cache.get(key)
    if hit is not None and (datetime.now(IST) - datetime.fromisoformat(hit["fetched_at"])).total_seconds() <= max_age:
        return hit
    try:
        data = _download_chain(symbol, expiry)
    except ChainError as exc:
        last = cache.get(good)
        if last is None:
            raise
        logger.warning("options: chain %s %s from cache: %s", symbol, expiry, exc)
        return {**last, "stale": True, "error": str(exc)}
    cache.set(key, data, 60)
    cache.set(good, data, STALE_SECONDS)
    return data


def quote(symbol, expiry, strike, option_type, max_age=CHAIN_SECONDS):
    """(side dict or None, spot, chain) for one contract."""
    c = chain(symbol, expiry, max_age=max_age)
    for r in c["rows"]:
        if abs(r["strike"] - float(strike)) < 1e-6:
            return r.get(option_type), c["spot"], c
    return None, c["spot"], c


# ---------------------------------------------------------------- analytics for the page
def summary(c, now=None):
    """ATM strike, PCR, max pain and Greeks for every row (Black-Scholes on NSE's IV)."""
    now = now or datetime.now(IST)
    rows, spot = c["rows"], c["spot"]
    expiry = date.fromisoformat(c["expiry"])
    t = greeks.years_to(expiry_dt(expiry), now)
    atm = min(rows, key=lambda r: abs(r["strike"] - spot))["strike"] if rows and spot else None
    ce_oi = sum((r["CE"] or {}).get("oi", 0) for r in rows)
    pe_oi = sum((r["PE"] or {}).get("oi", 0) for r in rows)
    pain = None
    if ce_oi + pe_oi:
        def loss(k):
            return sum(max(0, k - r["strike"]) * (r["CE"] or {}).get("oi", 0) +
                       max(0, r["strike"] - k) * (r["PE"] or {}).get("oi", 0) for r in rows)
        pain = min((r["strike"] for r in rows), key=loss)
    out = []
    for r in rows:
        row = {"strike": r["strike"]}
        for side in ("CE", "PE"):
            x = r[side]
            if x is not None:
                x = {**x, **greeks.greeks(spot, r["strike"], t, x["iv"], side)}
            row[side] = x
        out.append(row)
    return {**c, "rows": out, "atm": atm, "pcr": round(pe_oi / ce_oi, 2) if ce_oi else None, "max_pain": pain,
            "ce_oi": ce_oi, "pe_oi": pe_oi, "days_to_expiry": round(t * 365, 2)}


def settlement_price(symbol, day):
    """Closing value of the underlying on `day` (Yahoo daily) - what expiring contracts settle against."""
    from prediction.datasets import _yahoo, _yahoo_frame
    ysym = YAHOO.get(symbol, f"{symbol}.NS")
    df = _yahoo_frame(_yahoo(ysym, cache_seconds=300, period="1mo", interval="1d", prepost=False))
    if df.empty:
        return None
    df = df[df["timestamp"].dt.date == day]
    return float(df["close"].iloc[-1]) if not df.empty else None

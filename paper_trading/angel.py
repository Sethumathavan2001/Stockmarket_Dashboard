"""
Angel One (SmartAPI) for the Claude Orders page: two real broker accounts, "vivek" and "sethu".

Credentials come from the environment / .env (never from the page):

    ANGEL_VIVEK_API_KEY   ANGEL_VIVEK_CLIENT_ID   ANGEL_VIVEK_PIN   ANGEL_VIVEK_TOTP_SECRET
    ANGEL_SETHU_API_KEY   ANGEL_SETHU_CLIENT_ID   ANGEL_SETHU_PIN   ANGEL_SETHU_TOTP_SECRET

    ANGEL_VIVEK_INTERFACE=Wi-Fi   which internet connection each account uses (netroute.py) - its IP must
    ANGEL_SETHU_INTERFACE=...     be the one registered in that account's SmartAPI app
    ANGEL_LIVE_TRADING=1          real orders are only sent when this is set (wallets show without it)
    ANGEL_MAX_ORDER_VALUE=200000  refuse a single order worth more than this (rupees); 0 / unlimited = no limit

Login uses the TOTP secret (pyotp) - the session is kept in memory and renewed every few hours or when
Angel One says it expired.  Instruments (symbol token, tick size) come from Angel One's scrip master,
kept in logs/angel_scrips.json and refreshed daily.
"""
import json
import logging
import os
import threading
import time as _time
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path

from django.conf import settings

from prediction.market_time import IST

from . import netroute

logger = logging.getLogger(__name__)

ACCOUNTS = {"vivek": "Vivek", "sethu": "Sethu"}
KEYS = ("API_KEY", "CLIENT_ID", "PIN", "TOTP_SECRET")
SCRIP_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
SCRIP_FILE = Path(settings.BASE_DIR) / "logs" / "angel_scrips.json"
SESSION_HOURS = 6
FUNDS_SECONDS = 30
TIMEOUT = 20              # seconds per request (the SDK's own default, 7 s, fails on a slow moment)
NET_TRIES = 3             # attempts on a network error (see _retry)
MARGIN_QTY = 10           # shares asked about in intraday_leverage
RATE_TRIES = 3            # attempts when Angel One says "exceeding access rate"
QUOTE_BATCH = 50          # stocks in one market quote call (Angel One's limit)
QUOTE_GAP = 1.1           # seconds between those calls

_lock = threading.RLock()
_sessions = {}            # account -> (SmartConnect, logged in at)
_funds = {}               # account -> (fetched at, data)
_scrips = {"data": None}


class AngelError(Exception):
    pass


# ---------------------------------------------------------------- config
def env_name(account, key):
    return f"ANGEL_{account.upper()}_{key}"


def config(account):
    """(credentials dict, missing env names)."""
    vals = {k: (os.environ.get(env_name(account, k)) or "").strip() for k in KEYS}
    return vals, [env_name(account, k) for k, v in vals.items() if not v]


def configured(account):
    return account in ACCOUNTS and not config(account)[1]


def live_enabled():
    return (os.environ.get("ANGEL_LIVE_TRADING") or "").strip().lower() in ("1", "true", "yes", "on")


def max_order_value():
    """Largest single order in rupees, or None for no limit (ANGEL_MAX_ORDER_VALUE=0 / unlimited / none)."""
    raw = (os.environ.get("ANGEL_MAX_ORDER_VALUE") or "").strip().lower()
    if raw in ("0", "unlimited", "none", "no", "off"):
        return None
    try:
        return float(raw) if raw else 200000.0
    except ValueError:
        return 200000.0


# ---------------------------------------------------------------- network retries
def _never_sent(exc):
    """True when the request certainly did not reach Angel One (the connection was never made)."""
    import requests
    return isinstance(exc, requests.exceptions.ConnectTimeout) or (
        isinstance(exc, requests.exceptions.ConnectionError) and "NewConnectionError" in str(exc))


def _network(exc):
    import requests
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


def _retry(fn, safe_any):
    """Run fn() up to NET_TRIES times on network errors.  safe_any=True (logins, reads): retry any network
    error.  False (placing / cancelling orders): retry only when the request was never sent - otherwise
    Angel One may already have the order, and sending it again could double it."""
    for attempt in range(NET_TRIES):
        try:
            return fn()
        except Exception as exc:
            again = _network(exc) if safe_any else _never_sent(exc)
            if not again or attempt == NET_TRIES - 1:
                raise
            logger.warning("angel: network error (%s), retrying", type(exc).__name__)
            _time.sleep(1.5 * (attempt + 1))


# ---------------------------------------------------------------- session
def _login(account):
    import pyotp
    from SmartApi import SmartConnect
    cfg, missing = config(account)
    if missing:
        raise AngelError(f"Not configured - add {', '.join(missing)} to .env")

    try:
        public = netroute.public_ip(account)                 # also checks the account's connection is there
    except netroute.RouteError as exc:
        raise AngelError(str(exc)) from exc
    except Exception:
        public = None

    def attempt():
        with netroute.via(account):                          # out through this account's own connection
            sc = SmartConnect(api_key=cfg["API_KEY"], timeout=TIMEOUT)
            if public:
                sc.clientPublicIp = public                   # the SDK otherwise sends a fixed made-up IP
            return sc, sc.generateSession(cfg["CLIENT_ID"], cfg["PIN"], pyotp.TOTP(cfg["TOTP_SECRET"]).now())
    try:
        sc, res = _retry(attempt, safe_any=True)
    except netroute.RouteError as exc:
        raise AngelError(str(exc)) from exc
    except Exception as exc:
        hint = " - Angel One's server didn't answer in time; check the internet connection and try again." if _network(exc) else ""
        raise AngelError(f"Angel One login failed: {type(exc).__name__}{hint}") from exc
    if not res or not res.get("status"):
        raise AngelError(f"Angel One login failed: {(res or {}).get('message') or 'no response'}")
    return sc


def client(account, fresh=False):
    with _lock:
        hit = _sessions.get(account)
        if fresh or not hit or _time.time() - hit[1] > SESSION_HOURS * 3600:
            _sessions[account] = (_login(account), _time.time())
        return _sessions[account][0]


SESSION_CODES = {"AG8001", "AG8002", "AG8003", "AB1010"}   # invalid / expired / missing token, session expired
SESSION_TEXTS = ("invalid token", "token expired", "token is expired", "session expired", "invalid session")


def _session_expired(text="", code=""):
    """Only Angel One's real login-session errors - NOT any message containing "token": "the token is
    categorised under cautionary listings" (AB4036) means the stock, and must never cause a re-send."""
    return str(code).upper() in SESSION_CODES or any(t in str(text).lower() for t in SESSION_TEXTS)


def _via(account, fn, sc):
    with netroute.via(account):                              # every request leaves through the account's connection
        return fn(sc)


def _call(account, fn, order=False):
    """Run fn(SmartConnect) with network retries; log in again once if the session expired.
    order=True for calls that change orders: those are retried only when certainly not sent."""
    for attempt in range(2):
        sc = client(account, fresh=bool(attempt))
        try:
            res = _retry(lambda: _via(account, fn, sc), safe_any=not order)
        except netroute.RouteError as exc:                   # connection missing: nothing was sent
            raise AngelError(str(exc)) from exc
        except Exception as exc:
            if attempt == 0 and _session_expired(exc, getattr(exc, "code", "")):
                continue
            if order and _network(exc) and not _never_sent(exc):
                raise AngelError("Angel One didn't confirm in time - the order MAY have gone through. Check the order "
                                 "book in the Angel One app before trying again.") from exc
            raise AngelError(f"Angel One: {type(exc).__name__}: {exc}") from exc
        if isinstance(res, dict) and res.get("status") is False:
            msg = str(res.get("message") or res.get("errorcode") or "request refused")
            if attempt == 0 and _session_expired(msg, res.get("errorcode", "")):
                continue
            raise AngelError(f"Angel One: {msg}" + (f" ({res['errorcode']})" if res.get("errorcode") else ""))
        return res
    raise AngelError("Angel One session could not be renewed.")


# ---------------------------------------------------------------- data
def read_any(fn):
    """fn(SmartConnect) through the first account that answers: (account, response).  For market data only."""
    accounts = [a for a in ACCOUNTS if configured(a)]
    if not accounts:
        raise AngelError("No Angel One account in .env - market data from Angel One needs one.")
    error = None
    for acc in accounts:
        for attempt in range(RATE_TRIES):
            try:
                return acc, _call(acc, fn)
            except AngelError as exc:
                text = str(exc).lower()
                if "no data" in text:                     # nothing for that window: an empty answer, not a failure
                    return acc, {}
                error = error or exc                      # the first account's reason is the one worth showing
                if "access rate" not in text:
                    break                                 # this account can't: try the next
                _time.sleep(1 + attempt)                  # too fast: Angel One allows only a few calls a second
    if "access rate" in str(error).lower():
        raise AngelError("Angel One is limiting requests for a moment - trying again.")
    raise error


def quotes(tokens, exchange="NSE"):
    """{token: full quote}, QUOTE_BATCH tokens a call at about one call a second (Angel One's limit)."""
    out = {}
    for i in range(0, len(tokens), QUOTE_BATCH):
        if i:
            _time.sleep(QUOTE_GAP)
        part = tokens[i:i + QUOTE_BATCH]
        res = read_any(lambda sc: sc.getMarketData("FULL", {exchange: part}))[1]
        for q in ((res or {}).get("data") or {}).get("fetched") or []:
            out[str(q.get("symbolToken"))] = q
    return out


def trade_time(quote):
    """When a quote's last trade happened (IST), or None."""
    try:
        return datetime.strptime(quote.get("exchTradeTime") or "", "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST)
    except ValueError:
        return None


def connect(account):
    """The Connect IP button: check this account's connection and public IP from scratch, then log in again."""
    if not configured(account):
        return {"name": ACCOUNTS[account], "ok": False, "route": {}, "error": f"Not set up - add {', '.join(config(account)[1])} to .env"}
    route = netroute.describe(account, fresh=True)
    out = {"name": ACCOUNTS[account], "ok": False, "route": route, "error": route.get("error")}
    if out["error"]:
        return out
    try:
        client(account, fresh=True)
    except AngelError as exc:
        out["error"] = str(exc)
        return out
    wallet = funds(account, use_cache=False)
    out.update(ok=not wallet.get("error"), error=wallet.get("error"), available=wallet.get("available"))
    return out


def funds(account, use_cache=True):
    """{"configured", "available", "net", "used", "error"} - the RMS limits (cash available to trade)."""
    if not configured(account):
        return {"configured": False, "missing": config(account)[1]}
    hit = _funds.get(account)
    if use_cache and hit and _time.time() - hit[0] < FUNDS_SECONDS:
        return hit[1]
    try:
        d = (_call(account, lambda sc: sc.rmsLimit()) or {}).get("data") or {}
        num = lambda k: float(d.get(k) or 0)
        out = {"configured": True, "available": num("availablecash"), "net": num("net"),
               "used": num("utiliseddebits"), "m2m": num("m2munrealized") + num("m2mrealized"), "error": None}
    except AngelError as exc:
        out = {"configured": True, "error": str(exc)}
    _funds[account] = (_time.time(), out)
    return out


def _load_scrips():
    import requests
    r = requests.get(SCRIP_URL, timeout=90)
    r.raise_for_status()
    out = {}
    for x in r.json():
        if x.get("exch_seg") == "NSE" and str(x.get("symbol", "")).endswith("-EQ"):
            tick = Decimal(str(x.get("tick_size") or "5")) / 100            # given in paise
            out[x["name"]] = {"symbol": x["symbol"], "token": x["token"], "tick": str(tick if tick > 0 else Decimal("0.05"))}
    return {"as_of": datetime.now(IST).date().isoformat(), "scrips": out}


def instrument(code):
    """NSE equity for a company code like "HDFCBANK.NS": {symbol, token, tick}."""
    name = code.removesuffix(".NS")
    with _lock:
        data = _scrips["data"]
        if data is None and SCRIP_FILE.exists():
            try:
                data = json.loads(SCRIP_FILE.read_text(encoding="utf-8"))
            except ValueError:
                data = None
        if not data or data.get("as_of") != datetime.now(IST).date().isoformat():
            try:
                data = _load_scrips()
                SCRIP_FILE.parent.mkdir(exist_ok=True)
                SCRIP_FILE.write_text(json.dumps(data), encoding="utf-8")
            except Exception as exc:
                logger.warning("angel: scrip master not refreshed: %s", exc)
                if not data:
                    raise AngelError("Angel One instrument list unavailable - try again in a minute.") from exc
        _scrips["data"] = data
    inst = data["scrips"].get(name)
    if not inst:
        raise AngelError(f"{name} is not an NSE equity on Angel One.")
    return {**inst, "tick": Decimal(inst["tick"])}


def intraday_leverage(inst, price):
    """How many times its margin an intraday position in this stock can be worth, from Angel One's margin calculator.
    Selling is used: buying adds about 0.5% to the margin (4.98x).  0 when Angel One gives no margin.  Places no order."""
    position = {"exchange": "NSE", "qty": MARGIN_QTY, "price": price, "productType": "INTRADAY", "orderType": "MARKET",
                "token": inst["token"], "tradeType": "SELL"}
    margin = (read_any(lambda sc: sc.getMarginApi({"positions": [position]}))[1].get("data") or {}).get("totalMarginRequired")
    return round(MARGIN_QTY * price / margin, 2) if margin else 0.0


def to_tick(price, tick, rounding=ROUND_HALF_UP):
    tick = Decimal(str(tick)).normalize()                    # "0.050000" -> "0.05": no trailing zeros in prices
    return ((Decimal(str(price)) / tick).quantize(Decimal("1"), rounding=rounding) * tick).quantize(Decimal("0.01"))


def ltp(account, inst):
    d = _call(account, lambda sc: sc.ltpData("NSE", inst["symbol"], inst["token"]))
    try:
        return float(d["data"]["ltp"])
    except (KeyError, TypeError, ValueError):
        raise AngelError("Angel One gave no last price.")


# ---------------------------------------------------------------- orders
def _order(account, params):
    res = _call(account, lambda sc: sc.placeOrderFullResponse(dict(params)), order=True)
    oid = ((res or {}).get("data") or {}).get("orderid")
    if not oid:
        raise AngelError(f"Angel One did not accept the order: {(res or {}).get('message') or res}")
    return str(oid)


CAUTION_CODE = "AB4036"   # "the token is categorised under cautionary listings by the exchange"


def _base(inst, side, qty, consent=False):
    """consent=True adds "scripconsent": "yes" - you accepted the exchange's cautionary warning for this stock,
    which Angel One requires before it takes an API order in a cautionary-listed stock (AB4036)."""
    p = {"tradingsymbol": inst["symbol"], "symboltoken": inst["token"], "transactiontype": side, "exchange": "NSE",
         "producttype": "INTRADAY", "duration": "DAY", "quantity": str(int(qty)), "squareoff": "0", "stoploss": "0"}
    if consent:
        p["scripconsent"] = "yes"
    return p


def place_market(account, inst, side, qty, consent=False):
    """Intraday market order -> Angel One order id."""
    return _order(account, {**_base(inst, side, qty, consent), "variety": "NORMAL", "ordertype": "MARKET", "price": "0"})


def place_stop(account, inst, side, qty, trigger, consent=False):
    """Stop-loss LIMIT order that sits at Angel One: triggers at `trigger`, limit 1 % beyond it so it fills."""
    tick = inst["tick"]
    trig = to_tick(trigger, tick)
    if side == "SELL":                                       # protecting a long
        limit = to_tick(trig * Decimal("0.99"), tick, ROUND_FLOOR)
    else:                                                    # protecting a short
        limit = to_tick(trig * Decimal("1.01"), tick, ROUND_CEILING)
    return _order(account, {**_base(inst, side, qty, consent), "variety": "STOPLOSS", "ordertype": "STOPLOSS_LIMIT",
                            "triggerprice": str(trig), "price": str(limit)}), trig


def cancel(account, order_id, variety):
    return _call(account, lambda sc: sc.cancelOrder(order_id, variety), order=True)


def order_status(account, order_id):
    """(status, average price, quantity filled, text) from the order book - status is Angel One's, lower case:
    complete / rejected / cancelled / open / trigger pending / ..."""
    book = (_call(account, lambda sc: sc.orderBook()) or {}).get("data") or []
    for o in book:
        if str(o.get("orderid")) == str(order_id):
            return (str(o.get("orderstatus") or o.get("status") or "").lower(), float(o.get("averageprice") or 0),
                    int(float(o.get("filledshares") or 0)), str(o.get("text") or ""))
    return ("unknown", 0.0, 0, "order not found in today's order book")


def wait_for(account, order_id, seconds=6.0, step=0.7):
    """Poll until the order is no longer pending (or time runs out)."""
    end = _time.time() + seconds
    st = order_status(account, order_id)
    while st[0] not in ("complete", "rejected", "cancelled") and _time.time() < end:
        _time.sleep(step)
        st = order_status(account, order_id)
    return st

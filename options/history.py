"""
Past option prices for Replay (old-date option trading).

Real data
  NSE's F&O bhavcopy (end-of-day file, from 8 Jul 2024): every option contract's open / high / low /
  close, previous close, OI, change in OI, contracts traded, lot size, and the underlying's close.
  Downloaded once per day and kept in logs/fo_bhav/.
  The underlying's intraday path: Yahoo 5-minute candles (last ~60 days), the app's 5-minute table
  (stocks), else Yahoo hourly candles (~2 years), else only its daily open and close.

Estimated
  NSE publishes no intraday option prices for past days.  At 09:15 a contract is priced at its real
  open and at 15:30 at its real close; in between, Black-Scholes on the underlying's real price at that
  moment, with the implied volatility moving linearly from the one implied by the open to the one implied
  by the close - and never outside the contract's real high / low of the day.  Contracts that did not
  trade that day have no price (can't be traded), as in the market.
"""
import gzip
import io
import json
import logging
import math
import threading
import zipfile
from collections import OrderedDict
from datetime import date, datetime, time, timedelta
from pathlib import Path

from django.conf import settings

from prediction.market_time import IST

from . import chain as ch
from . import greeks

logger = logging.getLogger(__name__)

BHAV_URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
FIRST_DAY = date(2024, 7, 8)                 # first day of NSE's current bhavcopy format
CACHE_DIR = Path(settings.BASE_DIR) / "logs" / "fo_bhav"
OPEN_T, CLOSE_T = time(9, 15), time(15, 30)
SESSION_MIN = 375
STEP_MIN = 5

_lock = threading.Lock()
_mem = OrderedDict()                          # day -> parsed bhavcopy (a few days)
_paths = {}                                   # (symbol, day) -> underlying path
_ivs = {}                                     # (day, symbol, expiry) -> {(strike, type): (iv_open, iv_close)}
_bar_cache = {}                               # (symbol, day) -> (interval minutes, bars)


class HistoryError(Exception):
    pass


# ---------------------------------------------------------------- bhavcopy
def _download(day):
    from curl_cffi import requests as cr
    r = cr.get(BHAV_URL.format(d=day), impersonate="chrome", timeout=40)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise HistoryError(f"NSE archive refused the {day:%d %b %Y} file (HTTP {r.status_code}).")
    import csv
    z = zipfile.ZipFile(io.BytesIO(r.content))
    rows = csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8"))
    contracts, under, lots = {}, {}, {}
    num = lambda v: float(v or 0)
    for x in rows:
        if x["FinInstrmTp"] not in ("IDO", "STO"):
            continue
        sym, exp = x["TckrSymb"], x["XpryDt"]
        under[sym] = num(x["UndrlygPric"])
        lots.setdefault(sym, {})[exp] = int(num(x["NewBrdLotQty"]))
        contracts.setdefault(sym, {}).setdefault(exp, []).append([
            num(x["StrkPric"]), x["OptnTp"], num(x["OpnPric"]), num(x["HghPric"]), num(x["LwPric"]), num(x["ClsPric"]),
            num(x["PrvsClsgPric"]), num(x["SttlmPric"]), int(num(x["OpnIntrst"])), int(num(x["ChngInOpnIntrst"])),
            int(num(x["TtlTradgVol"])), x["FinInstrmTp"] == "IDO"])
    return {"day": day.isoformat(), "contracts": contracts, "underlying": under, "lots": lots}


def bhavcopy(day):
    """The day's option contracts, or HistoryError when NSE has no file (holiday / weekend / not yet out)."""
    if day < FIRST_DAY:
        raise HistoryError(f"Option history starts on {FIRST_DAY:%d %b %Y}.")
    if day > datetime.now(IST).date():
        raise HistoryError("That date is in the future.")
    with _lock:
        if day in _mem:
            _mem.move_to_end(day)
            return _mem[day]
        path, none = CACHE_DIR / f"{day:%Y%m%d}.json.gz", CACHE_DIR / f"{day:%Y%m%d}.none"
        data = None
        if path.exists():
            data = json.loads(gzip.decompress(path.read_bytes()))
        elif none.exists():
            pass
        else:
            data = _download(day)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            if data is not None:
                path.write_bytes(gzip.compress(json.dumps(data).encode()))
            elif day < datetime.now(IST).date():                      # today's file may still come out tonight
                none.write_text("no file")
        if data is None:
            raise HistoryError(f"No F&O trading on {day:%a %d %b %Y} (holiday, weekend, or NSE's file isn't out yet).")
        _mem[day] = data
        while len(_mem) > 4:
            _mem.popitem(last=False)
        return data


def is_trading_day(day):
    if day.weekday() >= 5:
        return False
    try:
        bhavcopy(day)
        return True
    except HistoryError:
        return False


def next_trading_day(day, limit=10):
    d = day
    for _ in range(limit):
        d += timedelta(days=1)
        if d > datetime.now(IST).date():
            return None
        if is_trading_day(d):
            return d
    return None


def latest_day():
    d = datetime.now(IST).date()
    for _ in range(10):
        if is_trading_day(d):
            return d
        d -= timedelta(days=1)
    return None


def underlyings(day):
    b = bhavcopy(day)
    names = {}
    try:
        names = {u["symbol"]: u["name"] for u in ch.underlyings()}
    except ch.ChainError:
        pass
    out = []
    for sym, by_exp in b["lots"].items():
        idx = any(r[11] for rows in b["contracts"][sym].values() for r in rows[:1])
        out.append({"symbol": sym, "name": names.get(sym) or ch.INDEX_NAMES.get(sym, sym), "kind": "index" if idx else "stock",
                    "lot": by_exp[min(by_exp)]})
    order = {s: i for i, s in enumerate(ch.INDICES)}
    out.sort(key=lambda x: (x["kind"] != "index", order.get(x["symbol"], 99), x["symbol"]))
    return out


def expiries(symbol, day):
    b = bhavcopy(day)
    if symbol not in b["contracts"]:
        raise HistoryError(f"{symbol} had no options on {day:%d %b %Y}.")
    return sorted(date.fromisoformat(e) for e in b["contracts"][symbol] if date.fromisoformat(e) >= day)


def lot_size(symbol, expiry, day):
    try:
        return bhavcopy(day)["lots"][symbol][expiry.isoformat()]
    except KeyError:
        raise HistoryError(f"{symbol} {expiry:%d %b %Y} contracts didn't exist on {day:%d %b %Y}.")


def underlying_close(symbol, day):
    return bhavcopy(day)["underlying"].get(symbol)


# ---------------------------------------------------------------- underlying intraday path
def _ohlc(df):
    return [(t.time(), float(o), float(h), float(l), float(c))
            for t, o, h, l, c in zip(df["timestamp"], df["open"], df["high"], df["low"], df["close"])]


def _bars(symbol, day):
    """(interval minutes, [(start time, open, high, low, close)]) for the day, best source first."""
    from prediction.datasets import _yahoo, _yahoo_frame
    ysym = ch.YAHOO.get(symbol, f"{symbol}.NS")
    tries = [("5m", "60d", 5)]
    for interval, period, mins in tries:
        try:
            df = _yahoo_frame(_yahoo(ysym, cache_seconds=3600, period=period, interval=interval, prepost=False))
            df = df[df["timestamp"].dt.date == day]
        except Exception:
            df = None
        if df is not None and len(df) >= 10:
            return mins, _ohlc(df)
    if not ch.is_index(symbol):                                    # the app's own 5-minute table
        from market.models import Company, FiveMinuteData
        c = Company.objects.filter(code=f"{symbol}.NS").first()
        if c:
            start = datetime.combine(day, time(0), tzinfo=IST)
            qs = FiveMinuteData.objects.filter(company=c, timestamp__gte=start, timestamp__lt=start + timedelta(days=1)).order_by("timestamp")
            rows = [(r.timestamp.astimezone(IST).time(), float(r.open), float(r.high), float(r.low), float(r.close)) for r in qs]
            if len(rows) >= 10:
                return 5, rows
    for interval, period, mins in (("1h", "730d", 60), ("1d", "5y", None)):
        try:
            df = _yahoo_frame(_yahoo(ysym, cache_seconds=3600, period=period, interval=interval, prepost=False))
            df = df[df["timestamp"].dt.date == day]
        except Exception:
            continue
        if len(df):
            if mins is None:
                return None, [(OPEN_T, float(df["open"].iloc[0]), float(df["high"].max()), float(df["low"].min()),
                               float(df["close"].iloc[-1]))]
            return mins, _ohlc(df)
    return None, []


def _minutes(t):
    return (t.hour * 60 + t.minute) - (OPEN_T.hour * 60 + OPEN_T.minute)


def bars(symbol, day):
    """The underlying's real candles of the day: (interval minutes or None for daily only, bars)."""
    key = (symbol, day)
    if key not in _bar_cache:
        _bar_cache[key] = _bars(symbol, day)
        if len(_bar_cache) > 200:
            _bar_cache.pop(next(iter(_bar_cache)))
    return _bar_cache[key]


def path(symbol, day):
    """Known underlying prices through the day: [(minute from 09:15, price)] and the data resolution."""
    key = (symbol, day)
    if key in _paths:
        return _paths[key]
    mins, bars_ = bars(symbol, day)
    close = underlying_close(symbol, day)
    pts = []
    if bars_:
        pts.append((0, bars_[0][1]))
        for t0, _o, _h, _l, c in bars_:
            end = _minutes(t0) + (mins or SESSION_MIN)
            if 0 < end < SESSION_MIN:
                pts.append((end, c))
    if close:
        pts.append((SESSION_MIN, close))
    elif bars_:
        pts.append((SESSION_MIN, bars_[-1][4]))
    if not pts:
        raise HistoryError(f"No price history for {symbol} on {day:%d %b %Y}.")
    if pts[0][0] != 0:
        pts.insert(0, (0, pts[0][1]))
    res = {5: "5-minute", 60: "hourly"}.get(mins, "daily open / close only")
    _paths[key] = (sorted(dict(pts).items()), res)
    if len(_paths) > 200:
        _paths.pop(next(iter(_paths)))
    return _paths[key]


def spot_at(symbol, day, t):
    """The underlying at clock time t: real at each known point, straight line in between."""
    pts, _ = path(symbol, day)
    m = min(max(_minutes(t), 0), SESSION_MIN)
    for (m0, p0), (m1, p1) in zip(pts, pts[1:]):
        if m0 <= m <= m1:
            return p0 if m1 == m0 else p0 + (p1 - p0) * (m - m0) / (m1 - m0)
    return pts[-1][1] if m >= pts[-1][0] else pts[0][1]


# ---------------------------------------------------------------- option prices at the clock
def implied_vol(price, spot, strike, t, option_type):
    """IV in % that reproduces `price` (bisection); None when the price is below intrinsic or time is up."""
    if price <= 0 or spot <= 0 or t <= 0:
        return None
    intrinsic = max(0.0, spot - strike) if option_type == "CE" else max(0.0, strike - spot)
    if price <= intrinsic + 1e-9:
        return 0.5
    lo, hi = 0.1, 400.0
    if greeks.price(spot, strike, t, hi, option_type) < price:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        if greeks.price(spot, strike, t, mid, option_type) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _at(day, t):
    return datetime.combine(day, t, tzinfo=IST)


def _iv_table(day, symbol, expiry):
    key = (day, symbol, expiry)
    if key in _ivs:
        return _ivs[key]
    b = bhavcopy(day)
    rows = b["contracts"].get(symbol, {}).get(expiry.isoformat(), [])
    u_open, u_close = spot_at(symbol, day, OPEN_T), underlying_close(symbol, day) or spot_at(symbol, day, CLOSE_T)
    exp_dt = ch.expiry_dt(expiry)
    t_open, t_close = greeks.years_to(exp_dt, _at(day, OPEN_T)), greeks.years_to(exp_dt, _at(day, CLOSE_T))
    table = {}
    for k, typ, o, h, l, c, *_ in rows:
        table[(k, typ)] = (implied_vol(o, u_open, k, t_open, typ), implied_vol(c, u_close, k, t_close, typ))
    _ivs[key] = table
    if len(_ivs) > 100:
        _ivs.pop(next(iter(_ivs)))
    return table


def _tick(v):
    return round(round(v / 0.05) * 0.05, 2)


def price_at(row, ivs, symbol, day, expiry, now):
    """(estimated premium, IV %) of one bhavcopy row at clock `now`; (None, None) if it didn't trade."""
    k, typ, o, h, l, c, prev, settle, oi, chg_oi, vol, _ = row
    if vol <= 0 or c <= 0:
        return None, None
    t = now.time()
    iv_o, iv_c = ivs.get((k, typ), (None, None))
    m = min(max(_minutes(t), 0), SESSION_MIN)
    if m <= 0 and o > 0:
        return o, iv_o
    if m >= SESSION_MIN:
        return c, iv_c
    iv_o = iv_o if iv_o is not None else iv_c
    iv_c = iv_c if iv_c is not None else iv_o
    if iv_o is None:
        frac = m / SESSION_MIN                                       # no IV at all: straight line open -> close
        return _tick((o or c) + (c - (o or c)) * frac), None
    iv = iv_o + (iv_c - iv_o) * m / SESSION_MIN
    p = greeks.price(spot_at(symbol, day, t), k, greeks.years_to(ch.expiry_dt(expiry), now), iv, typ)
    lo_, hi_ = (l if l > 0 else min(o or c, c)), (h if h > 0 else max(o or c, c))
    return _tick(min(max(p, lo_), hi_)), iv


def chain(symbol, expiry, now):
    """The chain at replay clock `now`, shaped like chain.chain() (bid = ask = the estimated price)."""
    day = now.date()
    b = bhavcopy(day)
    rows = b["contracts"].get(symbol, {}).get(expiry.isoformat())
    if not rows:
        raise HistoryError(f"No {symbol} contracts expiring {expiry:%d %b %Y} on {day:%d %b %Y}.")
    lot = b["lots"][symbol][expiry.isoformat()]
    ivs = _iv_table(day, symbol, expiry)
    by_k = {}
    for r in rows:
        p, iv = price_at(r, ivs, symbol, day, expiry, now)
        k, typ, o, h, l, c, prev, settle, oi, chg_oi, vol, _ = r
        side = None if p is None else {
            "ltp": p, "chg": round(p - prev, 2) if prev else 0.0, "pchg": (p / prev - 1) * 100 if prev else 0.0,
            # the bhavcopy counts OI in shares; the live chain (and this page) in contracts - volume matches as is
            "bid": p, "ask": p, "bid_qty": 0, "ask_qty": 0, "oi": oi // lot, "chg_oi": int(chg_oi / lot), "vol": vol,
            "iv": round(iv, 2) if iv else 0.0, "day": {"open": o, "high": h, "low": l, "close": c}}
        by_k.setdefault(k, {"strike": k, "CE": None, "PE": None})[typ] = side
    spot = spot_at(symbol, day, now.time())
    _, res = path(symbol, day)
    return {"symbol": symbol, "expiry": expiry.isoformat(), "spot": round(spot, 2), "timestamp": now.isoformat(),
            "rows": [by_k[k] for k in sorted(by_k)], "fetched_at": now.isoformat(), "stale": False,
            "replay": True, "resolution": res}


def quote(symbol, expiry, strike, option_type, now):
    c = chain(symbol, expiry, now)
    for r in c["rows"]:
        if abs(r["strike"] - float(strike)) < 1e-6:
            return r.get(option_type), c["spot"]
    return None, c["spot"]


def clock_points(day, after, until):
    """Replay clock steps (every 5 minutes) after `after` up to and including `until` on `day`."""
    out, m = [], _minutes(after) + STEP_MIN
    while m <= min(_minutes(until), SESSION_MIN):
        out.append((datetime.combine(day, OPEN_T) + timedelta(minutes=m)).time())
        m += STEP_MIN
    return out


def series(symbol, expiry, strike, option_type, now):
    """For the replay chart, up to clock `now`: the underlying's real candles and the contract's premium
    every 5 minutes (real at 09:15 and 15:30, estimated in between)."""
    day = now.date()
    mins, bars_ = bars(symbol, day)
    m_now = _minutes(now.time())
    candles = [{"x": _at(day, t).isoformat(), "o": o, "h": h, "l": l, "c": c} for t, o, h, l, c in bars_
               if _minutes(t) < m_now or (mins is None and m_now >= SESSION_MIN)]
    row = next((r for r in bhavcopy(day)["contracts"].get(symbol, {}).get(expiry.isoformat(), [])
                if abs(r[0] - float(strike)) < 1e-6 and r[1] == option_type), None)
    prem = []
    if row is not None:
        ivs = _iv_table(day, symbol, expiry)
        for m in range(0, min(m_now, SESSION_MIN) + 1, STEP_MIN):
            t_dt = _at(day, OPEN_T) + timedelta(minutes=m)
            p, _ = price_at(row, ivs, symbol, day, expiry, t_dt)
            if p is not None:
                prem.append({"x": t_dt.isoformat(), "v": p})
    return {"interval": mins, "resolution": path(symbol, day)[1], "underlying": candles, "premium": prem,
            "traded": bool(row and row[10] > 0),
            "day": None if row is None else {"open": row[2], "high": row[3], "low": row[4], "close": row[5]}}

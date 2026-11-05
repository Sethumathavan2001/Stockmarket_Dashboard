"""
Index history: daily and 5-minute candles of NIFTY, SENSEX and the other main indices, and their past
option / future expiry dates.  Used by the Index Analysis page (market/index_analysis.py).

Candles   Angel One historical candles (getCandleData, through the first Angel One account in .env that
          works - a read-only data call, never an order): daily from 2008, 5-minute from Jan 2016.
          Updates continue from the last stored candle; a candle still forming is never stored.
Expiries  read from the exchanges' own F&O bhavcopy files, which list the expiry date of every live
          contract; a date counts only when that day's own file has contracts expiring on it (so holiday
          shifts and the many schedule changes are exact):
            NSE  from 2016, hopping from one expiry day's file to the next (old format until 5 Jul 2024, UDiFF after);
            BSE  every weekday from Jan 2024 (UDiFF);
          SENSEX / BANKEX before 2024 (relaunched May / Oct 2023) follow the published schedule - SENSEX
          weekly on Friday, BANKEX weekly on Monday - moved to the trading day before when that is a
          holiday (source "schedule").

  python manage.py update_indices          everything (first run: ~20 minutes)
  start_update() / progress()             the same from the page, in a background thread
"""
import csv
import io
import json
import logging
import threading
import time as _time
import zipfile
from datetime import date, datetime, time, timedelta
from pathlib import Path

from django.conf import settings
from django.db import connection

from prediction.market_time import IST

from .models import IndexDaily, IndexExpiry, IndexFiveMinute, MarketIndex

logger = logging.getLogger(__name__)

# key, name, exchange, Angel One token, option symbol ("" = no index options)
CATALOG = [
    ("NIFTY", "NIFTY 50", "NSE", "99926000", "NIFTY"),
    ("BANKNIFTY", "NIFTY BANK", "NSE", "99926009", "BANKNIFTY"),
    ("FINNIFTY", "NIFTY FIN SERVICE", "NSE", "99926037", "FINNIFTY"),
    ("MIDCPNIFTY", "NIFTY MIDCAP SELECT", "NSE", "99926074", "MIDCPNIFTY"),
    ("NIFTYNXT50", "NIFTY NEXT 50", "NSE", "99926013", "NIFTYNXT50"),
    ("SENSEX", "SENSEX", "BSE", "99919000", "SENSEX"),
    ("BANKEX", "BANKEX", "BSE", "99919012", "BANKEX"),
    ("INDIAVIX", "INDIA VIX", "NSE", "99926017", ""),
    ("NIFTYIT", "NIFTY IT", "NSE", "99926008", ""),
    ("NIFTYMID100", "NIFTY MIDCAP 100", "NSE", "99926011", ""),
    ("NIFTYSML100", "NIFTY SMALLCAP 100", "NSE", "99926032", ""),
    ("NIFTY500", "NIFTY 500", "NSE", "99926004", ""),
]
DAILY_FROM, FIVE_FROM = date(2008, 1, 1), date(2016, 1, 1)
DAILY_CHUNK, FIVE_CHUNK = 1800, 90                  # days per Angel One request (its limits: 2000 / 100)
GAP = 0.45                                          # seconds between candle requests (Angel One: 3 a second)
NSE_OLD_URL = "https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip"
NSE_NEW_URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
BSE_URL = "https://www.bseindia.com/download/Bhavcopy/Derivative/BhavCopy_BSE_FO_0_0_0_{d:%Y%m%d}_F_0000.CSV"
NSE_NEW_FROM, BSE_FROM = date(2024, 7, 8), date(2024, 1, 1)
BSE_SCHEDULE = {"SENSEX": (date(2023, 5, 15), 4), "BANKEX": (date(2023, 10, 17), 0)}   # (from, weekday) until BSE_FROM
SCAN_FILE = Path(settings.BASE_DIR) / "logs" / "index_expiry_scan.json"

_lock = threading.Lock()
_rate = threading.Lock()
_last_call = [0.0]
_progress = {"running": False, "step": "", "done": 0, "total": 0, "error": None, "started": None, "finished": None,
             "log": []}


class IndexDataError(Exception):
    pass


def sync_catalog():
    """MarketIndex rows for CATALOG (created / renamed as needed)."""
    out = {}
    for i, (key, name, exch, token, opt) in enumerate(CATALOG):
        obj, _ = MarketIndex.objects.update_or_create(
            key=key, defaults={"name": name, "exchange": exch, "angel_token": token, "option_symbol": opt, "sort": i})
        out[key] = obj
    return out


def progress():
    with _lock:
        return {**_progress, "log": list(_progress["log"][-12:])}


def _set(**kw):
    with _lock:
        msg = kw.pop("note", None)
        _progress.update(kw)
        if msg:
            _progress["log"].append(f"{datetime.now(IST):%H:%M:%S} {msg}")
            logger.info("index data: %s", msg)


# ---------------------------------------------------------------- Angel One candles
def _accounts():
    from paper_trading import angel
    return [a for a in angel.ACCOUNTS if angel.configured(a)]


def _candles(index, interval, start, end):
    """[(datetime IST, o, h, l, c)] from Angel One for start..end (datetimes)."""
    from paper_trading import angel

    params = {"exchange": index.exchange, "symboltoken": index.angel_token, "interval": interval,
              "fromdate": f"{start:%Y-%m-%d %H:%M}", "todate": f"{end:%Y-%m-%d %H:%M}"}
    accounts = _accounts()
    if not accounts:
        raise IndexDataError("No Angel One account in .env - index history comes from Angel One's candle data.")
    last = None
    for acc in accounts:
        for attempt in range(5):
            with _rate:
                wait = GAP - (_time.monotonic() - _last_call[0])
                if wait > 0:
                    _time.sleep(wait)
                _last_call[0] = _time.monotonic()
            try:
                res = angel._call(acc, lambda sc: sc.getCandleData(params))
                return [(datetime.fromisoformat(t).astimezone(IST), float(o), float(h), float(lo), float(c))
                        for t, o, h, lo, c, *_ in (res.get("data") or [])]
            except angel.AngelError as exc:
                last = exc
                if "no data" in str(exc).lower():                      # holiday / weekend window
                    return []
                if "access rate" in str(exc).lower():                  # too fast: back off and retry
                    _time.sleep(2 + attempt * 2)
                    continue
                break                                                   # this account can't: try the next
    raise IndexDataError(f"Angel One candles for {index.key}: {last}")


def _complete(rows, interval_min, now):
    """Drop a candle that is still forming."""
    return [r for r in rows if r[0] + timedelta(minutes=interval_min) <= now]


def update_five_min(index, now=None):
    now = now or datetime.now(IST)
    last = IndexFiveMinute.objects.filter(index=index).order_by("-timestamp").values_list("timestamp", flat=True).first()
    start = last.astimezone(IST) if last else datetime.combine(FIVE_FROM, time(9, 15), tzinfo=IST)
    added = 0
    while start < now:
        end = min(start + timedelta(days=FIVE_CHUNK), now)
        rows = _complete(_candles(index, "FIVE_MINUTE", start, end), 5, now)
        objs = [IndexFiveMinute(index=index, timestamp=t, date=t.date(), open=o, high=h, low=lo, close=c)
                for t, o, h, lo, c in rows]
        if objs:
            IndexFiveMinute.objects.bulk_create(objs, batch_size=2000, update_conflicts=True,
                                                unique_fields=["index", "timestamp"],
                                                update_fields=["open", "high", "low", "close", "date"])
            added += len(objs)
        start = end
    return added


def update_daily(index, now=None):
    now = now or datetime.now(IST)
    last = IndexDaily.objects.filter(index=index).order_by("-date").values_list("date", flat=True).first()
    start = datetime.combine(last or DAILY_FROM, time(0, 0), tzinfo=IST)
    session_over = now.time() >= time(15, 30)
    added = 0
    while start < now:
        end = min(start + timedelta(days=DAILY_CHUNK), now)
        rows = [r for r in _candles(index, "ONE_DAY", start, end) if r[0].date() < now.date() or session_over]
        objs = [IndexDaily(index=index, date=t.date(), open=o, high=h, low=lo, close=c) for t, o, h, lo, c in rows]
        if objs:
            IndexDaily.objects.bulk_create(objs, batch_size=2000, update_conflicts=True,
                                           unique_fields=["index", "date"],
                                           update_fields=["open", "high", "low", "close"])
            added += len(objs)
        start = end + timedelta(days=1)
    return added


# ---------------------------------------------------------------- expiries
def _get(url, referer=None):
    from curl_cffi import requests as cr
    r = cr.get(url, impersonate="chrome", timeout=60, headers={"Referer": referer} if referer else None)
    return r


def nse_expiries(day, symbols):
    """{symbol: {expiry dates}} from NSE's F&O bhavcopy of `day`; None when there is no file (holiday)."""
    if day < NSE_NEW_FROM:
        r = _get(NSE_OLD_URL.format(d=day, mon=day.strftime("%b").upper()))
    else:
        r = _get(NSE_NEW_URL.format(d=day))
    if r.status_code == 404 or (r.status_code == 200 and not r.content.startswith(b"PK")):
        return None
    if r.status_code != 200:
        raise IndexDataError(f"NSE archive: HTTP {r.status_code} for {day}")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    out = {}
    for x in csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]), encoding="utf-8")):
        if day < NSE_NEW_FROM:
            x = {k.strip(): (v or "").strip() for k, v in x.items() if k}
            if x.get("INSTRUMENT") not in ("OPTIDX", "FUTIDX") or x.get("SYMBOL") not in symbols:
                continue
            out.setdefault(x["SYMBOL"], set()).add(datetime.strptime(x["EXPIRY_DT"], "%d-%b-%Y").date())
        else:
            if x.get("FinInstrmTp") not in ("IDO", "IDF") or x.get("TckrSymb") not in symbols:
                continue
            out.setdefault(x["TckrSymb"], set()).add(date.fromisoformat(x["XpryDt"]))
    return out


def bse_expiries(day, symbols):
    r = _get(BSE_URL.format(d=day), referer="https://www.bseindia.com/")
    if r.status_code != 200 or not r.content.startswith(b"TradDt"):
        return None
    out = {}
    for x in csv.DictReader(io.StringIO(r.text)):
        if x.get("FinInstrmTp") in ("IDO", "IDF") and x.get("TckrSymb") in symbols:
            out.setdefault(x["TckrSymb"], set()).add(date.fromisoformat(x["XpryDt"]))
    return out


def _scan_state():
    try:
        return json.loads(SCAN_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _save_scan(state):
    SCAN_FILE.parent.mkdir(exist_ok=True)
    SCAN_FILE.write_text(json.dumps(state, indent=1), encoding="utf-8")


def _store(by_symbol, indices, source):
    sym_to_index = {i.option_symbol: i for i in indices.values() if i.option_symbol}
    objs = [IndexExpiry(index=sym_to_index[s], date=d, source=source)
            for s, ds in by_symbol.items() if s in sym_to_index for d in ds]
    IndexExpiry.objects.bulk_create(objs, ignore_conflicts=True)
    return len(objs)


def trading_days():
    """Every trading day in the stored history (NIFTY's daily candles)."""
    return set(IndexDaily.objects.filter(index__key="NIFTY").values_list("date", flat=True))


def update_expiries(indices, today=None, step=None):
    """Scan the bhavcopies not scanned yet; returns the number of expiry rows offered.

    An expiry counts only when the exchange's file OF THAT DAY has contracts expiring that day - dates listed
    weeks ahead are sometimes moved later (holidays, the 2024-25 expiry-day changes), so they only say which
    day's file to read next: NSE hops from one expiry to the next; BSE (small files) is read every weekday."""
    today = today or datetime.now(IST).date()
    step = step or (lambda msg: None)
    state = _scan_state()
    nse_syms = {i.option_symbol for i in indices.values() if i.option_symbol and i.exchange == "NSE"}
    bse_syms = {i.option_symbol for i in indices.values() if i.option_symbol and i.exchange == "BSE"}
    n = 0
    upcoming = state.get("upcoming", {})

    day = date.fromisoformat(state["nse_next"]) if state.get("nse_next") else FIVE_FROM
    while day <= today:
        got = nse_expiries(day, nse_syms)
        if got is None:                                                  # holiday / weekend / not out yet
            if day >= today - timedelta(days=1):
                break                                                    # today's file comes out in the evening
            day += timedelta(days=1)
            continue
        n += _store({s: {day} for s, ds in got.items() if day in ds}, indices, "nse_bhav")
        later = sorted({e for ds in got.values() for e in ds if e > day})
        upcoming.update({s: sorted(e.isoformat() for e in ds if e > day)[:4] for s, ds in got.items()})
        day = min(later[0] if later else day + timedelta(days=7), day + timedelta(days=7))
        state["nse_next"], state["upcoming"] = day.isoformat(), upcoming
        _save_scan(state)
        if day.month != (day - timedelta(days=7)).month:
            step(f"NSE expiries checked to {day:%b %Y}")

    day = date.fromisoformat(state["bse_next"]) if state.get("bse_next") else BSE_FROM
    while day <= today:
        if day.weekday() < 5:
            got = bse_expiries(day, bse_syms)
            if got is None and day >= today - timedelta(days=1):
                break
            if got:
                n += _store({s: {day} for s, ds in got.items() if day in ds}, indices, "bse_bhav")
                upcoming.update({s: sorted(e.isoformat() for e in ds if e > day)[:4] for s, ds in got.items()})
            if day.day == 1:
                step(f"BSE expiries checked to {day:%b %Y}")
        day += timedelta(days=1)
        state["bse_next"], state["upcoming"] = day.isoformat(), upcoming
        _save_scan(state)

    # SENSEX / BANKEX before BSE's files: the published weekly schedule
    days = trading_days()
    if days:
        for key, (start, weekday) in BSE_SCHEDULE.items():
            idx = indices.get(key)
            if not idx:
                continue
            d = start + timedelta(days=(weekday - start.weekday()) % 7)
            sched = set()
            while d < BSE_FROM:
                e = d
                while e not in days and e > d - timedelta(days=6):             # holiday: the trading day before
                    e -= timedelta(days=1)
                if e in days:
                    sched.add(e)
                d += timedelta(days=7)
            n += _store({idx.option_symbol: sched}, indices, "schedule")
    return n


# ---------------------------------------------------------------- the whole update
def update_all(keys=None, step=None, expiries=True):
    step = step or (lambda msg: None)
    indices = sync_catalog()
    todo = [indices[k] for k in (keys or indices)]
    summary = {}
    for i, idx in enumerate(todo, 1):
        step(f"{idx.key}: daily candles ({i}/{len(todo)})")
        d = update_daily(idx)
        step(f"{idx.key}: 5-minute candles ({i}/{len(todo)})")
        f = update_five_min(idx)
        summary[idx.key] = {"daily": d, "five_min": f}
        step(f"{idx.key}: +{d} daily, +{f} five-minute candles")
    if expiries:
        step("Expiry dates from the exchanges' bhavcopies")
        summary["expiries"] = update_expiries(indices, step=step)
    return summary


def start_update(keys=None):
    with _lock:
        if _progress["running"]:
            return False
        _progress.update(running=True, step="Starting", error=None, started=datetime.now(IST).isoformat(),
                         finished=None, log=[])
    threading.Thread(target=_run, args=(keys,), name="index-data", daemon=True).start()
    return True


def _run(keys):
    try:
        update_all(keys, step=lambda m: _set(step=m, note=m))
        _set(step="Done", note="Update finished")
    except Exception as exc:
        logger.exception("index data update failed")
        _set(error=f"{type(exc).__name__}: {exc}", step="Failed")
    finally:
        connection.close()
        _set(running=False, finished=datetime.now(IST).isoformat())

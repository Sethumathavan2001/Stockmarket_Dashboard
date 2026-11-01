"""
Stocks Angel One won't take orders in: "the token is categorised under cautionary listings by the exchange".

Two sources:
  * NSE's surveillance lists - ASM (long and short term) and GSM - downloaded once a day into
    logs/nse_surveillance.json.
  * Angel One's own refusals: a stock it rejected as "cautionary" in the last REMEMBER_DAYS days is
    treated as blocked too (Angel One's list is wider than what NSE publishes, e.g. HEXT on 28 Oct 2026).

caution(code) gives the reason or None; Claude Orders flags such stocks and asks for your consent to the
exchange's cautionary warning before placing them (sent as "scripconsent": "yes" - see broker.place).
"""
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

from django.conf import settings

from prediction.market_time import IST

logger = logging.getLogger(__name__)

FILE = Path(settings.BASE_DIR) / "logs" / "nse_surveillance.json"
REMEMBER_DAYS = 7
REFUSAL = "cautionary"

_lock = threading.Lock()
_mem = {"data": None}


def _download():
    from news.fetch import _nse_json
    out = {}
    asm = _nse_json("/api/reportASM") or {}
    for part in ("longterm", "shortterm"):
        for r in (asm.get(part) or {}).get("data", []):
            if r.get("symbol"):
                out[r["symbol"]] = r.get("survDesc") or r.get("survCode") or "NSE ASM"
    for r in _nse_json("/api/reportGSM") or []:
        if isinstance(r, dict) and r.get("symbol"):
            out.setdefault(r["symbol"], r.get("survDesc") or "NSE GSM")
    if not out:
        raise ValueError("NSE returned empty surveillance lists")
    return {"as_of": datetime.now(IST).date().isoformat(), "stocks": out}


def nse_lists():
    """{SYMBOL: description} of today's NSE ASM / GSM stocks (last good copy if NSE refuses)."""
    with _lock:
        data = _mem["data"]
        if data is None and FILE.exists():
            try:
                data = json.loads(FILE.read_text(encoding="utf-8"))
            except ValueError:
                data = None
        if not data or data.get("as_of") != datetime.now(IST).date().isoformat():
            try:
                data = _download()
                FILE.parent.mkdir(exist_ok=True)
                FILE.write_text(json.dumps(data), encoding="utf-8")
            except Exception as exc:
                logger.warning("surveillance lists not refreshed: %s", exc)
                data = data or {"as_of": None, "stocks": {}}
        _mem["data"] = data
        return data["stocks"]


def refused_recently(now=None):
    """{code: date} of stocks Angel One refused as cautionary in the last REMEMBER_DAYS days."""
    from .models import BrokerOrder
    now = now or datetime.now(IST)
    qs = BrokerOrder.objects.filter(status=BrokerOrder.REJECTED, message__icontains=REFUSAL,
                                    session__gte=now.date() - timedelta(days=REMEMBER_DAYS))
    return {code: day for code, day in qs.values_list("code", "session").order_by("session")}


def caution(code, nse=None, refused=None):
    """Why Angel One won't take orders in this stock, or None."""
    sym = code.removesuffix(".NS")
    nse = nse_lists() if nse is None else nse
    if sym in nse:
        return f"On NSE's surveillance list ({nse[sym]})."
    refused = refused_recently() if refused is None else refused
    if code in refused:
        return f"Angel One flagged it on {refused[code]:%d %b} as an exchange 'cautionary listing' (AB4036)."
    return None

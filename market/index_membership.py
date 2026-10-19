"""
Which NSE indices a stock belongs to (shown on the Paper Trading page next to the stock's name).

  GET /api/market/stock-indices/<code>/  ->  {"code", "symbol", "indices": [{name, group, key}], "as_of", "source"}

Constituent lists are the official CSV files on niftyindices.com (NSE Indices Ltd.), downloaded once a
day and kept in logs/index_constituents.json, so a restart doesn't re-download them.  `key` links an
index to the live strip / popup chart above the navbar (indices.CHART_SYMBOLS) when there is one.
SENSEX (a BSE index) is not covered - BSE publishes no comparable list.
"""
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from prediction.market_time import IST

logger = logging.getLogger(__name__)

STORE = Path(settings.BASE_DIR) / "logs" / "index_constituents.json"
MAX_AGE = timedelta(days=1)
BASE_URL = "https://www.niftyindices.com/IndexConstituent/{}.csv"
# (label, niftyindices file, group, key of the live strip / popup chart or None)
INDICES = [
    ("NIFTY 50", "ind_nifty50list", "broad", "nifty50"),
    ("NIFTY NEXT 50", "ind_niftynext50list", "broad", None),
    ("NIFTY MIDCAP 100", "ind_niftymidcap100list", "broad", "midcap100"),
    ("NIFTY SMALLCAP 100", "ind_niftysmallcap100list", "broad", "smallcap100"),
    ("NIFTY 500", "ind_nifty500list", "broad", None),
    ("NIFTY BANK", "ind_niftybanklist", "sector", "banknifty"),
    ("NIFTY FIN SERVICE", "ind_niftyfinancelist", "sector", "finnifty"),
    ("NIFTY IT", "ind_niftyitlist", "sector", "niftyit"),
    ("NIFTY AUTO", "ind_niftyautolist", "sector", None),
    ("NIFTY PHARMA", "ind_niftypharmalist", "sector", None),
    ("NIFTY HEALTHCARE", "ind_niftyhealthcarelist", "sector", None),
    ("NIFTY FMCG", "ind_niftyfmcglist", "sector", None),
    ("NIFTY METAL", "ind_niftymetallist", "sector", None),
    ("NIFTY ENERGY", "ind_niftyenergylist", "sector", None),
    ("NIFTY OIL & GAS", "ind_niftyoilgaslist", "sector", None),
    ("NIFTY REALTY", "ind_niftyrealtylist", "sector", None),
    ("NIFTY PSU BANK", "ind_niftypsubanklist", "sector", None),
    ("NIFTY MEDIA", "ind_niftymedialist", "sector", None),
    ("NIFTY CONSUMER DURABLES", "ind_niftyconsumerdurableslist", "sector", None),
]
_lock = threading.Lock()
_mem = {"data": None}


def _download():
    from curl_cffi import requests as cr

    s = cr.Session(impersonate="chrome")
    members, failed = {}, []
    for label, f, _group, _key in INDICES:
        try:
            r = s.get(BASE_URL.format(f), timeout=20)
            lines = r.text.strip().splitlines()
            if r.status_code != 200 or not lines or not lines[0].startswith("Company Name"):
                raise ValueError(f"HTTP {r.status_code}")
            head = [h.strip() for h in lines[0].split(",")]
            col = head.index("Symbol")
            members[label] = sorted({ln.split(",")[col].strip() for ln in lines[1:] if ln.count(",") >= col})
        except Exception as exc:                      # one missing list must not stop the rest
            logger.warning("index constituents %s failed: %s", label, exc)
            failed.append(label)
    return {"as_of": datetime.now(IST).isoformat(), "members": members, "failed": failed}


def constituents():
    """{index label: [symbols]} - from memory, the file on disk, or a fresh download (at most once a day)."""
    with _lock:
        data = _mem["data"]
        if data is None and STORE.exists():
            try:
                data = json.loads(STORE.read_text(encoding="utf-8"))
            except ValueError:
                data = None
        fresh = data and datetime.now(IST) - datetime.fromisoformat(data["as_of"]) < MAX_AGE
        if not fresh:
            new = _download()
            if new["members"]:
                if data and new["failed"]:            # keep yesterday's list for an index that failed today
                    for label in new["failed"]:
                        if label in data.get("members", {}):
                            new["members"][label] = data["members"][label]
                data = new
                STORE.parent.mkdir(exist_ok=True)
                STORE.write_text(json.dumps(data), encoding="utf-8")
        _mem["data"] = data
        return data or {"as_of": None, "members": {}, "failed": [i[0] for i in INDICES]}


def indices_for(code):
    symbol = code.upper().removesuffix(".NS").removesuffix(".BO")
    data = constituents()
    out = [{"name": label, "group": group, "key": key}
           for label, _f, group, key in INDICES if symbol in data["members"].get(label, ())]
    return {"code": code, "symbol": symbol, "indices": out, "as_of": data["as_of"], "source": "niftyindices.com"}


@require_GET
def stock_indices_api(request, code):
    return JsonResponse(indices_for(code))

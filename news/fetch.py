"""
News sources for the News page.  Everything is fetched on demand, cached briefly, never stored.

  Market news       RSS feeds of Economic Times, Moneycontrol, Mint and Business Standard
  Announcements     NSE corporate announcements (the official exchange filings), market-wide or per stock
  Stock news        Google News RSS search for one company

Requests use curl_cffi with a Chrome fingerprint (NSE rejects plain HTTP clients); NSE also needs
the cookies its home page sets, so one session is kept and renewed when NSE refuses it.
"""
import html
import logging
import re
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

from django.core.cache import cache

from market.models import Company
from prediction.market_time import IST

logger = logging.getLogger(__name__)

TIMEOUT = 15
MARKET_CACHE = 120          # seconds
ANN_CACHE = 120
STOCK_CACHE = 300

FEEDS = [
    ("Economic Times", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("ET Stocks", "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms"),
    ("Moneycontrol", "https://www.moneycontrol.com/rss/latestnews.xml"),
    ("Mint", "https://www.livemint.com/rss/markets"),
    ("Business Standard", "https://www.business-standard.com/rss/markets-106.rss"),
]

# NSE announcement categories worth a trader's attention (matched on the category only - the
# free text mentions "order", "award" etc. in routine filings far too often)
IMPORTANT = re.compile(
    r"financial result|integrated filing- financial|outcome of board|press release|orders/contracts|acquisition|"
    r"amalgamation|merger|scheme of arrangement|dividend|buy ?back|bonus|split|credit rating|change in management|"
    r"insolvency|action\(s\) taken|litigation|default|spurt in volume|price movement|news verification|clarification|"
    r"fund rais|qualified institutions|preferential|investor presentation|material issue|takeover|key licenses|"
    r"agreements|product launch|resignation of (managing|md|ceo|chief)|offer for sale|delisting|block deal|"
    r"bulk deal|pledge", re.I)


class SourceError(Exception):
    pass


# ==========================================================================
# HTTP
# ==========================================================================
def _get(url, **kw):
    from curl_cffi import requests as cr

    r = cr.get(url, impersonate="chrome", timeout=TIMEOUT, **kw)
    if r.status_code != 200:
        raise SourceError(f"HTTP {r.status_code}")
    return r


_nse = {"session": None, "made": None}
_nse_lock = threading.Lock()


def _nse_json(path):
    """GET an NSE API path with a cookie session (renewed every 10 minutes or when refused)."""
    from curl_cffi import requests as cr

    headers = {"Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
               "Accept": "application/json"}
    with _nse_lock:
        for attempt in range(2):
            s = _nse["session"]
            if s is None or attempt or datetime.now() - _nse["made"] > timedelta(minutes=10):
                s = cr.Session(impersonate="chrome")
                s.get("https://www.nseindia.com/", timeout=TIMEOUT)
                _nse.update(session=s, made=datetime.now())
            r = s.get("https://www.nseindia.com" + path, timeout=TIMEOUT, headers=headers)
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError:
                    pass
        raise SourceError(f"NSE refused the request (HTTP {r.status_code})")


# ==========================================================================
# parsing helpers
# ==========================================================================
_TAG = re.compile(r"<[^>]+>")


def clean_text(s, limit=300):
    s = html.unescape(_TAG.sub(" ", html.unescape(s or "")))
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0] + "..."


def _safe_url(u):
    u = (u or "").strip()
    return u if u.startswith(("http://", "https://")) else ""


def _rss_time(text):
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def parse_rss(xml_text, source):
    # bytes: let the XML declaration decide the encoding (requests' text guess garbles some feeds)
    root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text.strip())
    items = []
    for it in root.iter("item"):
        title = clean_text(it.findtext("title"), 400)
        link = _safe_url(it.findtext("link"))
        if not title or not link:
            continue
        src = source
        src_el = it.find("source")
        if src_el is not None and src_el.text:            # Google News: the real publisher
            src = src_el.text.strip()
            title = re.sub(r"\s+-\s+" + re.escape(src) + r"$", "", title)
        t = _rss_time(it.findtext("pubDate"))
        items.append({"title": title, "url": link, "source": src,
                      "summary": clean_text(it.findtext("description")) if source != "Google News" else "",
                      "time": t.isoformat() if t else None})
    return items


# ==========================================================================
# which of the user's companies a headline mentions
# ==========================================================================
_STOP = {"US", "UK", "RBI", "IPO", "GDP", "FII", "DII", "SEBI", "NSE", "BSE", "IT", "AI", "CEO", "MD", "EV", "GST",
         "FY", "PSU", "LIC", "MF", "ETF", "NIFTY", "SIP", "Q1", "Q2", "Q3", "Q4", "EPS", "OFS", "QIP", "MSCI", "FPI",
         "OIL", "GOLD", "POWER", "BANK", "INDIA", "JUST", "BEST", "GREEN", "TRENT", "IDEA", "SUN"}
_SUFFIX = re.compile(r"(\blimited|\bltd\.?|\bcorporation|\bcorp\.?|\bcompany|\bco\.|\(india\)|& co)\s*$", re.I)


def _short_name(name):
    n = re.sub(r"^the\s+", "", (name or "").strip(), flags=re.I)
    for _ in range(3):
        n2 = _SUFFIX.sub("", n).strip(" ,.-")
        if n2 == n:
            break
        n = n2
    return n


def company_matcher():
    """(regex over names, regex over symbols, lookup) built from the Company table, cached."""
    m = cache.get("news:matcher")
    if m is not None:
        return m
    by_name, by_symbol = {}, {}
    for code, name in Company.objects.exclude(sector="ETF").values_list("code", "name"):
        short = _short_name(name)
        if len(short) >= 5 and len(short.split()) <= 5:
            by_name.setdefault(short.lower(), code)
        sym = code.removesuffix(".NS").removesuffix(".BO")
        if len(sym) >= 3 and sym.isalpha() and sym not in _STOP:
            by_symbol.setdefault(sym, code)
    names = sorted(by_name, key=len, reverse=True)
    name_re = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.I) if names else None
    sym_re = re.compile(r"\b(" + "|".join(sorted(by_symbol, key=len, reverse=True)) + r")\b") if by_symbol else None
    m = (name_re, sym_re, by_name, by_symbol)
    cache.set("news:matcher", m, 3600)
    return m


def mentions(text):
    name_re, sym_re, by_name, by_symbol = company_matcher()
    found = []
    if name_re:
        for hit in name_re.findall(text):
            code = by_name.get(hit.lower())
            if code and code not in found:
                found.append(code)
    if sym_re:
        for hit in sym_re.findall(text):
            code = by_symbol.get(hit)
            if code and code not in found:
                found.append(code)
    return found[:4]


# ==========================================================================
# public
# ==========================================================================
def market_news(refresh=False):
    key = "news:market"
    if not refresh:
        hit = cache.get(key)
        if hit is not None:
            return hit
    items, sources = [], []
    for name, url in FEEDS:
        try:
            got = parse_rss(_get(url).content, name)
            sources.append({"name": name, "ok": True, "count": len(got)})
            items.extend(got)
        except Exception as exc:                  # one feed down must not break the page
            logger.warning("News feed %s failed: %s", name, exc)
            sources.append({"name": name, "ok": False, "count": 0, "error": str(exc)[:120]})
    seen, unique = set(), []
    for it in sorted(items, key=lambda x: x["time"] or "", reverse=True):
        k = re.sub(r"\W+", "", it["title"].lower())[:80]
        if k in seen:
            continue
        seen.add(k)
        it["mentions"] = mentions(it["title"] + " " + it["summary"])
        unique.append(it)
    out = {"items": unique[:250], "sources": sources, "fetched_at": datetime.now(IST).isoformat()}
    cache.set(key, out, MARKET_CACHE)
    return out


def _parse_nse_time(s):
    try:
        return datetime.strptime(s, "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST)
    except (TypeError, ValueError):
        return None


def _announcement(a, codes):
    t = _parse_nse_time(a.get("an_dt") or a.get("exchdisstime"))
    subject = (a.get("desc") or "").strip()
    text = clean_text(a.get("attchmntText"), 600)
    sym = (a.get("symbol") or "").strip()
    code = f"{sym}.NS"
    important = bool(IMPORTANT.search(subject))
    return {"symbol": sym, "code": code if code in codes else None, "name": a.get("sm_name") or sym,
            "industry": a.get("smIndustry") or "", "subject": subject, "text": text,
            "time": t.isoformat() if t else None, "pdf": _safe_url(a.get("attchmntFile")), "important": important}


def _codes():
    codes = cache.get("news:codes")
    if codes is None:
        codes = set(Company.objects.values_list("code", flat=True))
        cache.set("news:codes", codes, 3600)
    return codes


def announcements(days=1, refresh=False):
    """Market-wide NSE announcements for the last `days` calendar days (incl. today)."""
    days = max(1, min(int(days), 7))
    key = f"news:ann:{days}"
    if not refresh:
        hit = cache.get(key)
        if hit is not None:
            return hit
    today = datetime.now(IST).date()
    frm = today - timedelta(days=days - 1)
    data = _nse_json(f"/api/corporate-announcements?index=equities&from_date={frm:%d-%m-%Y}&to_date={today:%d-%m-%Y}")
    codes = _codes()
    items = [_announcement(a, codes) for a in (data if isinstance(data, list) else [])]
    items.sort(key=lambda x: x["time"] or "", reverse=True)
    out = {"items": items, "from": frm.isoformat(), "to": today.isoformat(), "fetched_at": datetime.now(IST).isoformat()}
    cache.set(key, out, ANN_CACHE)
    return out


def stock_news(company, refresh=False):
    """NSE filings (30 days) and Google News stories for one company."""
    key = f"news:stock:{company.pk}"
    if not refresh:
        hit = cache.get(key)
        if hit is not None:
            return hit
    out = {"company": company.code, "name": company.name, "sector": company.sector, "industry": company.industry,
           "announcements": [], "news": [], "errors": [], "fetched_at": datetime.now(IST).isoformat()}
    sym = company.code.removesuffix(".NS")
    if company.code.endswith(".NS"):
        today = datetime.now(IST).date()
        try:
            data = _nse_json(f"/api/corporate-announcements?index=equities&symbol={quote_plus(sym)}"
                             f"&from_date={today - timedelta(days=30):%d-%m-%Y}&to_date={today:%d-%m-%Y}")
            codes = _codes()
            out["announcements"] = sorted((_announcement(a, codes) for a in (data if isinstance(data, list) else [])),
                                          key=lambda x: x["time"] or "", reverse=True)
        except Exception as exc:
            logger.warning("NSE announcements for %s failed: %s", sym, exc)
            out["errors"].append(f"NSE announcements: {exc}")
    query = f'"{_short_name(company.name)}" (share OR stock OR shares)'
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}+when:30d&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        items = parse_rss(_get(url).content, "Google News")
        items.sort(key=lambda x: x["time"] or "", reverse=True)
        out["news"] = items[:40]
    except Exception as exc:
        logger.warning("Google News for %s failed: %s", company.code, exc)
        out["errors"].append(f"Google News: {exc}")
    cache.set(key, out, STOCK_CACHE)
    return out

"""
Refresh the Claude Orders list from current data: the Intraday Signal Board, run inside the app.

  1. candidates   Live page (top gainers / losers with Rs 20 crore+ turnover, PDH / PDL scans), stocks in the
                  headlines since the last close, and stocks with trained ML models
  2. evidence     per stock: Candle Predictor forecast (5m), reliable patterns, ML intraday forecast, news,
                  price vs VWAP / previous close; plus the NIFTY 50 direction today
  3. decision     BUY / SELL only when at least two credible signals agree, one of them a tested model signal
                  (forecast that beat its test, reliable pattern, ML model with an edge), and not against NIFTY
                  unless 3+ signals agree
  4. levels       stop from the forecast's 80 % range about an hour ahead (0.4 %-1.2 %), target 1.5x the stop

The result replaces claude_orders.json. A refresh takes a few minutes, so it runs in a background thread;
start_refresh() starts it (once at a time) and progress() reports how far it is.
"""
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from django.conf import settings

from prediction.market_time import IST

log = logging.getLogger(__name__)

TURNOVER_MIN = 2000          # NSE turnover field is in Rs lakh -> Rs 20 crore+
TOP_MOVERS = 12
TOP_NEWS = 15
METHOD = ("BUY / SELL only when at least two credible signals agree, one of them a tested model signal (Candle Predictor "
          "forecast that beat its test, reliable patterns, ML models with an edge), and not against the NIFTY direction "
          "unless 3+ signals agree. Stop sized from the forecast's 80% range about an hour ahead (0.4%-1.2%); target 1.5x "
          "the stop distance.")
MODEL_SIGNALS = ("Candle forecast", "Reliable patterns", "ML model")

_lock = threading.Lock()
_progress = {"running": False, "step": "", "done": 0, "total": 0, "started": None, "finished": None,
             "error": None, "orders": None}


def progress():
    with _lock:
        return dict(_progress)


def _set(**kw):
    with _lock:
        _progress.update(kw)


def start_refresh(orders_file):
    """Start a background refresh; False when one is already running."""
    with _lock:
        if _progress["running"]:
            return False
        _progress.update(running=True, step="Fetching Live page data", done=0, total=0, error=None, orders=None,
                         started=datetime.now(IST).isoformat(), finished=None)
    threading.Thread(target=_run, args=(Path(orders_file),), name="claude-board", daemon=True).start()
    return True


def _run(orders_file):
    try:
        payload = build()
        tmp = orders_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(orders_file)
        _set(orders=len(payload["orders"]), step="Done")
    except Exception as exc:                                       # shown on the page; the old list is kept
        log.exception("Claude Orders refresh failed")
        _set(error=f"{type(exc).__name__}: {exc}", step="Failed")
    finally:
        from django.db import connection
        connection.close()
        _set(running=False, finished=datetime.now(IST).isoformat())


# ---------------------------------------------------------------- 1. candidates

def candidates(live):
    from market.models import Company
    from news import fetch
    from news.signal import last_close

    codes = set(Company.objects.exclude(sector="ETF").values_list("code", flat=True))
    order, why = [], {}

    def add(sym, reason):
        c = f"{sym}.NS"
        if c not in codes:
            return
        if c not in why:
            order.append(c)
            why[c] = []
        why[c].append(reason)

    liquid = lambda r: (r.get("turnover") or 0) >= TURNOVER_MIN
    for r in sorted(filter(liquid, live.get("gainers", [])), key=lambda r: -r["net_price"])[:TOP_MOVERS]:
        add(r["symbol"], f"Live: top gainer {r['net_price']:+.2f}%")
    for r in sorted(filter(liquid, live.get("loosers", [])), key=lambda r: r["net_price"])[:TOP_MOVERS]:
        add(r["symbol"], f"Live: top loser {r['net_price']:+.2f}%")
    for r in live.get("pdh_data", []):
        add(r["nsecode"], "Live: above previous-day high & VWAP (PDH scan)")
    for r in live.get("pdl_data", []):
        add(r["nsecode"], "Live: below previous-day low & VWAP (PDL scan)")

    try:
        since, cnt = last_close().isoformat(), {}
        for it in fetch.market_news(refresh=True)["items"]:
            if (it.get("time") or "") >= since and it["sentiment"]["label"] != "neutral":
                for c in it["mentions"]:
                    cnt[c] = cnt.get(c, 0) + 1
        for c, n in sorted(cnt.items(), key=lambda kv: -kv[1])[:TOP_NEWS]:
            add(c.removesuffix(".NS"), f"News: {n} headline{'s' if n > 1 else ''} since the last close")
    except Exception:
        log.warning("Claude Orders refresh: news unavailable", exc_info=True)

    for p in (Path(settings.BASE_DIR) / "ml_models").glob("*/manifest.json"):
        add(p.parent.name.removesuffix(".NS"), "Has trained ML models")
    return order, why


# ---------------------------------------------------------------- 2. evidence

def gather(code, why):
    from ai_prediction import tools
    from market.models import Company
    from news.signal import news_signal
    from patterns import engine

    c = Company.objects.get(code=code)
    rec = {"code": code, "name": c.name, "sector": c.sector, "why": why}
    try:
        p = engine.predict(c, "5m", news=news_signal)
        f = p["forecast"]
        main = f["main"] if f else None
        keys = ("label", "price", "ret_pct", "p_up", "low", "high", "accuracy", "baseline", "base_ret_pct", "base_p_up",
                "news_tilt_pct")
        rec.update({
            "asof": p["asof"], "price": p["price"], "levels": p["levels"],
            "has_candles": any(not x["future"] for x in p["candles"]),
            "vwap_last": p["vwap"][-1]["v"] if p["vwap"] else None,
            "forecast": None if not f else {
                "band": [[x["x"], x["p10"], x["p90"]] for x in f["band"]],
                "main": {k: main.get(k) for k in keys} if main else None,
                "news": {k: f["news"].get(k) for k in ("label", "stock_bias", "fresh_count")} if f.get("news") else None,
                "news_items": [{k: it.get(k) for k in ("title", "url", "source")} | {"label": it["sentiment"]["label"]}
                               for it in (f.get("news") or {}).get("items", [])[:5]],
            },
            "active": [{"name": a["name"], "direction": a["direction"], "reliable": a["reliable"]} for a in p["active"]],
            "pattern_vote": p["pattern_vote"],
        })
    except Exception as exc:
        rec["error"] = f"candle: {type(exc).__name__}: {exc}"
    ml, err = tools.run_data_tool("get_ml_forecast", {"company_code": code})
    if not err:
        intr = ml.get("intraday", {})
        rec["ml"] = intr.get("forecasts") if intr.get("available") else None
    return rec


def market_direction():
    """NIFTY 50 today: +1 above its open and its level 30 min ago, -1 below both, else 0."""
    from prediction.datasets import _yahoo, _yahoo_frame

    nf = _yahoo_frame(_yahoo("^NSEI", cache_seconds=60, period="5d", interval="5m", prepost=False))
    if nf.empty:
        return None
    nf = nf[nf["timestamp"].dt.date == nf["timestamp"].iloc[-1].date()]       # latest session
    if len(nf) < 7:
        return None
    last, opn, ago30 = float(nf["close"].iloc[-1]), float(nf["open"].iloc[0]), float(nf["close"].iloc[-7])
    d = 1 if last > opn and last > ago30 else -1 if last < opn and last < ago30 else 0
    return {"dir": d, "day_pct": (last / opn - 1) * 100, "last30_pct": (last / ago30 - 1) * 100,
            "session": nf["timestamp"].iloc[-1].date().isoformat()}


# ---------------------------------------------------------------- 3. decision + 4. levels

def _sgn(v, dead=0.0):
    return 0 if v is None or abs(v) <= dead else (1 if v > 0 else -1)


def score(s, market, pdh, pdl):
    """Adds signals / score / decision / conviction / plan to one gathered stock."""
    if "error" in s or not s.get("has_candles"):
        s["decision"] = "SKIP"
        return s
    sym = s["code"].removesuffix(".NS")
    f = s.get("forecast") or {}
    m = f.get("main") or {}
    sig = []

    # candle forecast WITHOUT the news tilt (news is its own signal) - counts only if it beat the simple rule
    if m:
        acc, base = m.get("accuracy"), m.get("baseline")
        cred = acc is not None and base is not None and acc > base
        pu = m.get("base_p_up") if m.get("base_p_up") is not None else m.get("p_up")
        rp = m.get("base_ret_pct") if m.get("base_ret_pct") is not None else m.get("ret_pct")
        sig.append({"name": "Candle forecast", "dir": _sgn((pu or 0.5) - 0.5, 0.05), "w": 1.0 if cred else 0.0, "cred": cred,
                    "detail": f"next {m.get('label')}: P(up) {(pu or 0.5) * 100:.0f}%, {rp or 0:+.3f}% · tested "
                              f"{(acc or 0) * 100:.1f}% vs simple rule {(base or 0) * 100:.1f}%"})
    rel = [a for a in s.get("active", []) if a["reliable"]]
    sig.append({"name": "Reliable patterns", "dir": _sgn(s.get("pattern_vote")), "w": 1.0 if rel else 0.0, "cred": bool(rel),
                "detail": ", ".join(f"{a['name']} ({a['direction']})" for a in rel) or "none with a proven record active now"})
    ml = s.get("ml") or []
    edge = [x for x in ml if x.get("edge")]
    sig.append({"name": "ML model", "dir": _sgn(sum(x["ret_pct"] for x in edge)) if edge else 0, "w": 1.0 if edge else 0.0,
                "cred": bool(edge),
                "detail": ("; ".join(f"{x['in']} {x['direction']} ({x['ret_pct']:+.3f}%)" for x in edge) if edge
                           else "no horizon passed the edge test - ignored" if ml else "not trained for this stock")})
    nw = f.get("news") or {}
    nb = nw.get("stock_bias")
    d = _sgn(nb, 0.15)
    sig.append({"name": "News", "dir": d, "w": 0.5 if d else 0.0, "cred": bool(d),           # untested -> half weight
                "detail": f"{nw.get('label')} ({nb or 0:+.2f}) · {nw.get('fresh_count', 0)} items since the last close"
                          if nw else "no news signal"})
    price, vwap, pdc = s["price"], s.get("vwap_last"), (s.get("levels") or {}).get("pdc")
    d = 0
    if vwap and pdc:
        d = 1 if price > vwap and price > pdc else -1 if price < vwap and price < pdc else 0
    if sym in pdh and d >= 0:
        d = 1
    if sym in pdl and d <= 0:
        d = -1
    scan = " · PDH breakout scan" if sym in pdh else " · PDL breakdown scan" if sym in pdl else ""
    chg = (price / pdc - 1) * 100 if pdc else None
    sig.append({"name": "Price action", "dir": d, "w": 1.0 if d else 0.0, "cred": bool(d),
                "detail": f"₹{price:.2f} vs VWAP {vwap:.2f}, {chg:+.2f}% vs previous close{scan}" if vwap and pdc else "n/a"})
    mdir = (market or {}).get("dir", 0)
    if market:
        sig.append({"name": "Market (NIFTY)", "dir": mdir, "w": 1.0 if mdir else 0.0, "cred": bool(mdir),
                    "detail": f"NIFTY {market['day_pct']:+.2f}% today, {market['last30_pct']:+.2f}% in the last 30 min"})

    model_agree = lambda d: any(x["cred"] and x["dir"] == d and x["name"] in MODEL_SIGNALS for x in sig)
    wsum = sum(x["w"] for x in sig)
    sc = sum(x["w"] * x["dir"] for x in sig) / wsum if wsum else 0.0
    up = sum(1 for x in sig if x["cred"] and x["dir"] > 0)
    dn = sum(1 for x in sig if x["cred"] and x["dir"] < 0)
    stock_up, stock_dn = up - (mdir > 0), dn - (mdir < 0)        # agreeing signals other than the market itself
    # a call needs a MODEL signal on its side - price action and the market alone just describe what already happened
    if sc >= 0.4 and up >= 2 and model_agree(1) and (mdir >= 0 or stock_up >= 3):
        dec = "BUY"
    elif sc <= -0.4 and dn >= 2 and model_agree(-1) and (mdir <= 0 or stock_dn >= 3):
        dec = "SELL"
    else:
        dec = "WATCH"
    conviction = plan = None
    if dec != "WATCH":
        d = 1 if dec == "BUY" else -1
        n_models = sum(1 for x in sig if x["cred"] and x["dir"] == d and x["name"] in MODEL_SIGNALS)
        conviction = "Higher" if n_models >= 2 or max(up, dn) >= 4 else "Medium"
        if f.get("band"):
            k = min(11, len(f["band"]) - 1)
            p10, p90 = f["band"][k][1], f["band"][k][2]
            r = min(max(((price - p10) if d > 0 else (p90 - price)) / price, 0.004), 0.012)
            plan = {"stop": price * (1 - d * r), "target": price * (1 + d * 1.5 * r), "risk_pct": r * 100}
    s.update(signals=sig, score=round(sc, 2), decision=dec, conviction=conviction, plan=plan)
    return s


def to_order(s, gen):
    side, p = s["decision"], s["plan"]
    sym = s["code"].removesuffix(".NS")
    agree = [x for x in s["signals"] if x["cred"] and x["dir"] == (1 if side == "BUY" else -1)]
    news = [n for n in (s.get("forecast") or {}).get("news_items", []) if n.get("label") != "neutral"][:3]
    return {
        "id": f"{gen:%Y%m%d-%H%M}-{sym}-{side}",
        "code": s["code"], "name": s["name"], "sector": s.get("sector") or "",
        "side": side, "product": "INTRADAY", "conviction": s["conviction"], "score": s["score"],
        "price_at": round(s["price"], 2), "price_time": s["asof"],
        "stop": round(p["stop"], 2), "target": round(p["target"], 2),
        "stop_pct": round(p["risk_pct"], 3), "target_pct": round(p["risk_pct"] * 1.5, 3),
        "reasons": [{"signal": x["name"], "detail": x["detail"]} for x in agree],
        "news": [{"title": n["title"], "url": n.get("url"), "source": n.get("source"), "label": n["label"]} for n in news],
        "square_off_by": "15:20",
    }


def build():
    from market.views import live_snapshot

    live = live_snapshot(tolerant=True)
    pdh = {r["nsecode"] for r in live.get("pdh_data", [])}
    pdl = {r["nsecode"] for r in live.get("pdl_data", [])}
    _set(step="Choosing candidate stocks")
    cands, why = candidates(live)
    _set(step="Reading NIFTY direction", total=len(cands))
    try:
        market = market_direction()
    except Exception:
        log.warning("Claude Orders refresh: NIFTY unavailable", exc_info=True)
        market = None
    scored = []
    for i, code in enumerate(cands, 1):
        _set(step=f"Analysing {code.removesuffix('.NS')}", done=i - 1)
        try:
            scored.append(score(gather(code, why[code]), market, pdh, pdl))
        except Exception:
            log.warning("Claude Orders refresh: %s failed", code, exc_info=True)
    _set(done=len(cands), step="Writing orders")
    gen = datetime.now(IST).replace(microsecond=0)
    picks = sorted((s for s in scored if s.get("decision") in ("BUY", "SELL") and s.get("plan")),
                   key=lambda s: (s["decision"] != "BUY", -abs(s["score"])))
    counts = {k: sum(1 for s in scored if s.get("decision") == k) for k in ("BUY", "SELL", "WATCH", "SKIP")}
    return {
        "generated": gen.isoformat(),
        "session": (market or {}).get("session") or gen.date().isoformat(),
        "source": "Signal board", "source_url": None, "method": METHOD,
        "checked": len(cands), "counts": counts,
        "market": market, "live_ok": bool(live.get("gainers") or live.get("loosers")),
        "orders": [to_order(s, gen) for s in picks],
    }

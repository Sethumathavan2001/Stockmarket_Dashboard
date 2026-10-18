"""
News signal for one stock right now: the fresh news since the previous session's close, scored
with news/sentiment.py and turned into

  bias       -1..+1   which way the news leans (stock news ~80 %, today's market mood ~20 %)
  intensity   0..1    how much news there is (strong, recent items) - news days move more

Each item counts less the older it is (half-life ~6 hours).  Used by the Candle Predictor's
"Include news" option for the live session only.  There is no news history in this app, so the
effect of this signal on prices cannot be back-tested - it is a rule of thumb, shown as such.
"""
import math
from datetime import datetime, time, timedelta

from prediction.market_time import IST

from . import fetch

HALF_LIFE_H = 6.0
STOCK_SHARE, MARKET_SHARE = 0.8, 0.2


def last_close(now=None):
    """15:30 IST of the previous weekday session."""
    now = now or datetime.now(IST)
    d = now.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return datetime.combine(d, time(15, 30), tzinfo=IST)


def _weight(iso, now):
    try:
        age_h = max(0.0, (now - datetime.fromisoformat(iso)).total_seconds() / 3600)
    except (TypeError, ValueError):
        return 0.0
    return 0.5 ** (age_h / HALF_LIFE_H)


def news_signal(company, now=None):
    now = now or datetime.now(IST)
    since = last_close(now)
    stock = fetch.stock_news(company)
    fresh = []
    for it in stock["news"]:
        if it.get("time") and it["time"] >= since.isoformat():
            fresh.append({**it, "kind": "news", "w": _weight(it["time"], now)})
    for a in stock["announcements"]:
        if a.get("time") and a["time"] >= since.isoformat():
            fresh.append({"title": f"{a['subject']}: {a['text']}", "url": a["pdf"], "source": "NSE filing",
                          "time": a["time"], "sentiment": a["sentiment"], "kind": "filing",
                          "w": _weight(a["time"], now) * (1.5 if a["important"] else 0.6)})

    raw = sum(max(-3.0, min(3.0, it["sentiment"]["score"])) / 3 * it["w"] for it in fresh)
    stock_bias = math.tanh(raw / 1.2)
    # only items with a view count ("stocks to watch" lists are neutral and say nothing)
    opinionated = sum(it["w"] for it in fresh if it["sentiment"]["label"] != "neutral")
    strong = sum(it["w"] for it in fresh if it["sentiment"]["strong"])
    intensity = min(1.0, 0.2 * opinionated + 0.25 * strong)

    market_mood = None
    try:
        mk = fetch.market_news()
        today = [it for it in mk["items"] if it.get("time") and it["time"] >= since.isoformat()]
        from .sentiment import summarize
        market_mood = summarize(today) if today else None
    except Exception:                               # market feeds down: use the stock news alone
        pass
    market_bias = (market_mood["mood"] / 100) if market_mood else 0.0

    bias = max(-1.0, min(1.0, STOCK_SHARE * stock_bias + MARKET_SHARE * market_bias))
    label = "Positive" if bias >= 0.15 else "Negative" if bias <= -0.15 else "Neutral"
    fresh.sort(key=lambda x: x["time"], reverse=True)
    return {
        "since": since.isoformat(), "bias": round(bias, 3), "stock_bias": round(stock_bias, 3),
        "market_bias": round(market_bias, 3), "intensity": round(intensity, 3), "label": label,
        "fresh_count": len(fresh),
        "positive": sum(1 for x in fresh if x["sentiment"]["label"] == "positive"),
        "negative": sum(1 for x in fresh if x["sentiment"]["label"] == "negative"),
        "market_mood": market_mood,
        "items": [{k: x[k] for k in ("title", "url", "source", "time", "sentiment", "kind")} | {"weight": round(x["w"], 2)}
                  for x in fresh[:12]],
        "errors": stock.get("errors", []),
    }

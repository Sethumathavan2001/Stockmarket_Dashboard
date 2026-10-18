"""
Financial news sentiment for headlines and exchange filings (rule-based, instant, no API key).

Generic sentiment tools misread market language ("loss narrows" is good news, "debt rises" is
bad, "Sell rating" is not about shopping), so this scorer works on finance phrases:

  1. Direction rules   a metric next to a move:  good metric + up  (profit rises)   -> positive
                                                  good metric + down (sales fall)     -> negative
                                                  bad metric  + up   (loss widens)    -> negative
                                                  bad metric  + down (debt falls)     -> positive
  2. Event words       upgrade, order win, record high, buyback ... / downgrade, fraud, default, probe ...
  3. Negation          "not", "no", "fails to", "without" flip the next cue.

score > 0 positive, < 0 negative; |score| >= 2 counts as strong.  Every result lists the cues it
used, so the page can show why a headline got its colour.
"""
import re

GOOD_METRICS = {
    "profit", "profits", "pat", "revenue", "revenues", "sales", "income", "margin", "margins", "earnings", "ebitda",
    "aum", "orders", "order", "orderbook", "demand", "growth", "dividend", "volume", "volumes", "offtake", "exports",
    "share", "shares", "stock", "stocks", "nifty", "sensex", "market", "markets", "index", "indices", "rupee",
    "guidance", "outlook", "target", "rating", "eps", "deposits", "loans", "advances", "disbursements", "bookings",
    "production", "output", "subscription", "valuation", "mcap", "gdp", "inflows", "fii", "fpi", "dii", "nim",
    "realisation", "realization", "throughput", "traffic", "occupancy", "arpu", "collections", "premium", "premiums",
}
BAD_METRICS = {
    "loss", "losses", "debt", "costs", "cost", "npa", "npas", "inflation", "deficit", "slippages", "provisions",
    "attrition", "outflows", "volatility", "vix", "pressure", "borrowing", "borrowings", "expenses", "unemployment",
    "default", "defaults", "delinquencies", "pledge", "pledged", "yields", "crude", "rates",
    "selloff", "sell-off", "worries", "worry", "fears", "fear", "concerns", "concern", "turmoil", "tensions",
    "uncertainty", "slowdown", "headwinds", "volatility",
}
UP = {
    "rise", "rises", "rose", "rising", "jump", "jumps", "jumped", "surge", "surges", "surged", "soar", "soars",
    "soared", "climb", "climbs", "climbed", "gain", "gains", "gained", "up", "higher", "increase", "increases",
    "increased", "double", "doubles", "doubled", "triple", "triples", "grow", "grows", "grew", "expand", "expands",
    "expanded", "improve", "improves", "improved", "rally", "rallies", "rallied", "advance", "advances", "advanced",
    "widen", "widens", "widened", "spike", "spikes", "spiked", "zoom", "zooms", "zoomed", "skyrocket", "skyrockets",
    "accelerate", "accelerates", "rebound", "rebounds", "rebounded", "recover", "recovers", "recovered",
    "beat", "beats", "outperform", "outperforms", "highest", "bounce", "bounces", "mount", "mounts", "intensify",
    "intensifies", "deepen", "deepens", "escalate", "escalates", "hike", "hikes", "hiked",
}
DOWN = {
    "fall", "falls", "fell", "falling", "drop", "drops", "dropped", "decline", "declines", "declined", "slump",
    "slumps", "slumped", "plunge", "plunges", "plunged", "tumble", "tumbles", "tumbled", "slide", "slides", "slid",
    "dip", "dips", "dipped", "down", "lower", "shrink", "shrinks", "shrank", "contract", "contracts", "contracted",
    "narrow", "narrows", "narrowed", "cut", "cuts", "slash", "slashes", "slashed", "sink", "sinks", "sank", "crash",
    "crashes", "crashed", "slip", "slips", "slipped", "ease", "eases", "eased", "decrease", "decreases", "decreased",
    "lowest", "weaken", "weakens", "weakened", "erode", "erodes", "eroded", "halve", "halves", "halved", "miss",
    "misses", "missed", "reduce", "reduces", "reduced", "lose", "loses", "lost", "shed", "sheds", "tank", "tanks",
    "tanked", "retreat", "retreats", "retreated", "moderate", "moderates", "moderated", "trim", "trims", "trimmed",
    "fade", "fades", "faded", "recede", "recedes", "receded", "subside", "subsides", "subsided", "cool", "cools",
    "cooled", "abate", "abates", "calm", "calms",
}
# phrases first (longest wins), then single words: weight > 0 positive, < 0 negative
PHRASES = {
    "all-time high": 2, "record high": 2, "52-week high": 1.5, "lifetime high": 2, "order win": 2, "bags order": 2,
    "wins order": 2, "secures order": 2, "receives order": 2, "bagging of orders": 2, "receiving of orders": 1.5,
    "new orders": 1.5, "orders worth": 1.5, "order worth": 1.5, "buy rating": 2, "upgrade to buy": 2.5,
    "target price raised": 2, "raises target": 2, "hikes target": 2, "strong buy": 2.5, "beats estimates": 2,
    "above estimates": 1.5, "better-than-expected": 2, "better than expected": 2, "multibagger": 1.5,
    "stake buy": 1, "share buyback": 2, "buyback": 1.5, "bonus issue": 1.5, "special dividend": 2,
    "interim dividend": 1, "final dividend": 1, "debt-free": 1.5, "turns profitable": 2.5, "returns to profit": 2.5,
    "back in black": 2, "rate cut": 1, "fresh high": 1.5, "upper circuit": 2, "green zone": 1, "in the green": 1,
    "52-week low": -1.5, "record low": -2, "all-time low": -2, "lower circuit": -2, "sell rating": -2,
    "downgrade to sell": -2.5, "cuts target": -2, "target price cut": -2, "lowers target": -2, "below estimates": -1.5,
    "misses estimates": -2, "worse-than-expected": -2, "worse than expected": -2, "profit warning": -2.5,
    "net loss": -1.5, "swings to loss": -2.5, "slips into loss": -2.5, "turns loss": -2, "in the red": -1,
    "red zone": -1, "show cause": -1.5, "show-cause": -1.5, "income tax raid": -2.5, "it raid": -2, "ed raid": -2.5,
    "sebi order": -1.5, "sebi bans": -2.5, "rate hike": -1, "profit booking": -1, "profit-taking": -1,
    "profit taking": -1, "block deal": 0.5, "record date": 0, "stocks to watch": 0, "stock to watch": 0,
    "top stocks": 0, "top stock": 0, "top gainers": 0, "top losers": 0, "heavy blow": -2, "rules against": -1,
    "write-off": -1, "write off": -1, "slow growth": -1.5, "loses steam": -1, "flex muscle": 1, "promoter sells": -1.5,
    "promoter selling": -1.5, "promoters sell": -1.5, "stake sale": -1, "fii selling": -1.5, "fpi selling": -1.5,
    "fii outflows": -1.5, "fpi outflows": -1.5, "fii buying": 1.5, "fpi buying": 1.5, "rating upgrade": 2,
    "rating downgrade": -2, "credit rating upgrade": 2, "credit rating downgrade": -2, "outlook positive": 1.5,
    "outlook negative": -1.5, "outlook stable": 0.3, "resignation of": -1, "corporate insolvency": -3,
    "action(s) taken or orders passed": -1, "pendency of litigation": -1.5, "delay/default": -2.5,
    "spurt in volume": 0, "news verification": 0, "price movement": 0, "bagging/receiving of orders": 2,
    "change in management": -0.5, "cessation": -0.5,
}
WORDS = {
    # positive
    "upgrade": 2, "upgrades": 2, "upgraded": 2, "outperform": 1.5, "overweight": 1.5, "accumulate": 1,
    "bullish": 1.5, "rally": 1, "rallies": 1, "surge": 1, "surges": 1, "soars": 1.5, "skyrockets": 2, "zooms": 1.5,
    "boost": 1, "boosts": 1, "robust": 1, "strong": 1, "stellar": 1.5, "upbeat": 1, "optimism": 1, "optimistic": 1,
    "breakout": 1, "approval": 1, "approves": 1, "approved": 1, "wins": 1, "bags": 1.5, "secures": 1.5, "awarded": 1.5,
    "expansion": 1, "expands": 1, "launch": 0.5, "launches": 0.5, "partnership": 0.8, "tie-up": 0.8, "acquire": 0.5,
    "acquires": 0.5, "acquisition": 0.5, "merger": 0.3, "recovery": 1, "rebound": 1, "inflows": 1, "dividend": 1,
    "bonus": 1, "outperforms": 1.5, "beat": 1, "beats": 1.5, "record": 1, "jackpot": 1.5, "gains": 0.8,
    "gainers": 0.8, "positive": 0.8, "improves": 1, "improvement": 1, "strengthens": 1, "milestone": 1,
    "subscribed": 0.8, "oversubscribed": 1.5, "listing gains": 1.5,
    # negative
    "downgrade": -2, "downgrades": -2, "downgraded": -2, "underperform": -1.5, "underweight": -1.5,
    "reduce": -0.5, "bearish": -1.5, "fraud": -3, "scam": -3, "probe": -2, "probes": -2, "raid": -2.5,
    "raids": -2.5, "penalty": -1.5, "penalised": -1.5, "penalized": -1.5, "fine": -0.8, "fined": -1.5, "ban": -2,
    "bans": -2, "banned": -2, "default": -2.5, "defaults": -2.5, "insolvency": -3, "bankruptcy": -3,
    "bankrupt": -3, "resigns": -1.5, "resignation": -1, "quits": -1.5, "lawsuit": -1.5, "litigation": -1,
    "weak": -1, "weakness": -1, "slump": -1.5, "crash": -2, "crashes": -2, "plunge": -1.5, "plunges": -1.5,
    "tumbles": -1.5, "tanks": -2, "selloff": -1.5, "concern": -1, "concerns": -1, "worries": -1, "worry": -1,
    "fears": -1, "fear": -1, "warning": -1.5, "warns": -1.5, "layoffs": -1.5, "layoff": -1.5, "strike": -1,
    "outflows": -1, "delay": -1, "delays": -1, "delayed": -1, "recall": -1.5, "halt": -1.5, "halts": -1.5,
    "suspended": -2, "suspension": -2, "disappoints": -1.5, "disappointing": -1.5, "losses": -1, "loss": -0.8,
    "losers": -0.8, "negative": -0.8, "volatile": -0.5, "turmoil": -1.5, "slowdown": -1, "headwinds": -1,
    "pressure": -0.8, "cautious": -0.5, "uncertainty": -1, "tariff": -0.5, "tariffs": -0.5, "sanctions": -1,
    "war": -1, "downturn": -1.5, "recession": -2, "crisis": -2, "glitch": -1, "outage": -1, "ransomware": -2,
    "hack": -1.5, "hacked": -1.5, "breach": -1.5, "suspends": -1, "blow": -1.5,
    "slow": -0.8, "sluggish": -1, "muted": -0.8, "tepid": -1, "subdued": -0.8, "selloff": -1.5, "sell-off": -1.5, "misses": -1.5, "miss": -1, "cut": -0.5, "cuts": -0.5,
    "slashes": -1, "dumps": -1.5, "exits": -0.5, "pledges": -1, "selling": -0.5,
}
NEGATORS = {"not", "no", "never", "without", "fails", "failed", "unable", "nor", "despite"}
_TOKEN = re.compile(r"[a-z][a-z0-9&'\-]*|\d+(?:\.\d+)?%?")
_PHRASES = sorted(PHRASES, key=len, reverse=True)
WINDOW = 4


def _cues(text):
    """[(weight, cue text)] for one piece of text."""
    low = " " + (text or "").lower().replace("’", "'") + " "
    cues, taken = [], []
    for ph in _PHRASES:
        start = low.find(ph)
        while start != -1:
            end = start + len(ph)
            if not any(s < end and start < e for s, e in taken) and (not low[start - 1].isalnum()) \
                    and (end >= len(low) or not low[end].isalnum()):
                taken.append((start, end))
                if PHRASES[ph]:
                    cues.append((PHRASES[ph], ph))
            start = low.find(ph, end)
    masked = list(low)
    for s, e in taken:
        masked[s:e] = " " * (e - s)
    tokens = _TOKEN.findall("".join(masked))

    used = set()
    # 1. metric + direction within a few words ("net profit rises 33%", "jump in revenue")
    for i, tok in enumerate(tokens):
        if tok not in UP and tok not in DOWN:
            continue
        lo, hi = max(0, i - WINDOW), min(len(tokens), i + WINDOW + 1)
        near = [j for j in range(lo, hi) if j != i and (tokens[j] in GOOD_METRICS or tokens[j] in BAD_METRICS)
                and j not in used]
        if not near:
            continue
        j = min(near, key=lambda k: (abs(k - i), k > i))          # closest metric, preferring the one before the verb
        metric_sign = 1 if tokens[j] in GOOD_METRICS else -1
        move_sign = 1 if tok in UP else -1
        neg = any(t in NEGATORS for t in tokens[max(0, min(i, j) - 2):min(i, j)])
        w = 1.5 * metric_sign * move_sign * (-1 if neg else 1)
        cues.append((w, f"{tokens[j]} {tok}" if j < i else f"{tok} {tokens[j]}"))
        used.update({i, j})
    # 2. event words not already part of a rule
    for i, tok in enumerate(tokens):
        if i in used or tok not in WORDS:
            continue
        w = WORDS[tok]
        if any(t in NEGATORS for t in tokens[max(0, i - 2):i]):
            w = -w * 0.7
        cues.append((w, tok))
    return cues


def score(title, summary=""):
    """{"label": positive|negative|neutral, "score": float, "strong": bool, "cues": [str]}"""
    cues = [(w, c) for w, c in _cues(title)] + [(w * 0.5, c) for w, c in _cues(summary)]
    s = round(sum(w for w, _ in cues), 2)
    label = "positive" if s >= 0.8 else "negative" if s <= -0.8 else "neutral"
    ordered = sorted(cues, key=lambda x: -abs(x[0]))
    shown = []
    for w, c in ordered:
        tag = ("+" if w > 0 else "-") + " " + c
        if tag not in shown:
            shown.append(tag)
    return {"label": label, "score": s, "strong": abs(s) >= 2, "cues": shown[:5]}


def summarize(items):
    """Counts and a -100..100 mood for a list of scored items."""
    n = len(items)
    pos = sum(1 for it in items if it["sentiment"]["label"] == "positive")
    neg = sum(1 for it in items if it["sentiment"]["label"] == "negative")
    mood = round((pos - neg) / (pos + neg) * 100) if pos + neg else 0
    return {"total": n, "positive": pos, "negative": neg, "neutral": n - pos - neg, "mood": mood,
            "label": "Positive" if mood >= 20 else "Negative" if mood <= -20 else "Mixed"}

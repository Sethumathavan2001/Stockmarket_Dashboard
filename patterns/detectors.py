"""
Candlestick pattern detection.

Covers the 12 patterns explicitly called out as most useful for intraday
trading: Doji, Hammer, Shooting Star, (Bullish/Bearish) Marubozu, Bullish
Engulfing, Bearish Engulfing, Piercing Line, Dark Cloud Cover, Morning Star,
Evening Star, Three White Soldiers, Three Black Crows.

Detection is a straightforward, transparent rule-based shape classifier
using body/wick ratios — it does NOT check broader trend context (e.g. a
textbook "Hammer" is only meaningful after a decline; this only checks the
candle's own shape). Treat matches as descriptive labels for what a candle
(or short candle sequence) looks like, not as validated trading signals.
"""


def _body(c):
    return abs(c["close"] - c["open"])


def _range(c):
    return c["high"] - c["low"]


def _upper_wick(c):
    return c["high"] - max(c["open"], c["close"])


def _lower_wick(c):
    return min(c["open"], c["close"]) - c["low"]


def _is_bullish(c):
    return c["close"] > c["open"]


def _is_bearish(c):
    return c["close"] < c["open"]


# ---------------------------------------------------------------- single --

def match_doji(c):
    rng = _range(c)
    if rng <= 0:
        return False
    return _body(c) <= rng * 0.1


def match_hammer(c):
    rng = _range(c)
    if rng <= 0:
        return False
    return (
        _body(c) <= rng * 0.35
        and _lower_wick(c) >= rng * 0.5
        and _upper_wick(c) <= rng * 0.15
    )


def match_shooting_star(c):
    rng = _range(c)
    if rng <= 0:
        return False
    return (
        _body(c) <= rng * 0.35
        and _upper_wick(c) >= rng * 0.5
        and _lower_wick(c) <= rng * 0.15
    )


def match_marubozu(c):
    rng = _range(c)
    if rng <= 0:
        return None
    if _upper_wick(c) <= rng * 0.05 and _lower_wick(c) <= rng * 0.05 and _body(c) >= rng * 0.9:
        return "Bullish Marubozu" if _is_bullish(c) else "Bearish Marubozu"
    return None


# ------------------------------------------------------------------- two --

def match_bullish_engulfing(c1, c2):
    return (
        _is_bearish(c1) and _is_bullish(c2)
        and c2["open"] <= c1["close"] and c2["close"] >= c1["open"]
    )


def match_bearish_engulfing(c1, c2):
    return (
        _is_bullish(c1) and _is_bearish(c2)
        and c2["open"] >= c1["close"] and c2["close"] <= c1["open"]
    )


def match_piercing_line(c1, c2):
    if not (_is_bearish(c1) and _is_bullish(c2)):
        return False
    mid1 = (c1["open"] + c1["close"]) / 2
    return c2["open"] < c1["close"] and mid1 < c2["close"] < c1["open"]


def match_dark_cloud_cover(c1, c2):
    if not (_is_bullish(c1) and _is_bearish(c2)):
        return False
    mid1 = (c1["open"] + c1["close"]) / 2
    return c2["open"] > c1["close"] and c1["open"] < c2["close"] < mid1


# ----------------------------------------------------------------- three --

def match_morning_star(c1, c2, c3):
    if not (_is_bearish(c1) and _is_bullish(c3)):
        return False
    r1 = _range(c1)
    if r1 <= 0:
        return False
    small_middle = _body(c2) <= r1 * 0.3
    closes_into_c1 = c3["close"] > (c1["open"] + c1["close"]) / 2
    return small_middle and closes_into_c1


def match_evening_star(c1, c2, c3):
    if not (_is_bullish(c1) and _is_bearish(c3)):
        return False
    r1 = _range(c1)
    if r1 <= 0:
        return False
    small_middle = _body(c2) <= r1 * 0.3
    closes_into_c1 = c3["close"] < (c1["open"] + c1["close"]) / 2
    return small_middle and closes_into_c1


def match_three_white_soldiers(c1, c2, c3):
    return (
        _is_bullish(c1) and _is_bullish(c2) and _is_bullish(c3)
        and c2["close"] > c1["close"] and c3["close"] > c2["close"]
        and c2["open"] > c1["open"] and c3["open"] > c2["open"]
    )


def match_three_black_crows(c1, c2, c3):
    return (
        _is_bearish(c1) and _is_bearish(c2) and _is_bearish(c3)
        and c2["close"] < c1["close"] and c3["close"] < c2["close"]
        and c2["open"] < c1["open"] and c3["open"] < c2["open"]
    )


# Pattern name -> (direction, strength). Strength is only a relative weight
# used to combine simultaneous matches into one bias score in forecast.py —
# it is not a probability or a backtested statistical edge.
PATTERN_WEIGHTS = {
    "Doji": ("neutral", 0.0),
    "Hammer": ("bullish", 1.0),
    "Shooting Star": ("bearish", 1.0),
    "Bullish Marubozu": ("bullish", 1.2),
    "Bearish Marubozu": ("bearish", 1.2),
    "Bullish Engulfing": ("bullish", 1.5),
    "Bearish Engulfing": ("bearish", 1.5),
    "Piercing Line": ("bullish", 1.2),
    "Dark Cloud Cover": ("bearish", 1.2),
    "Morning Star": ("bullish", 2.0),
    "Evening Star": ("bearish", 2.0),
    "Three White Soldiers": ("bullish", 2.0),
    "Three Black Crows": ("bearish", 2.0),
}


def detect_patterns_at(candles, i):
    """Every pattern whose LAST candle is candles[i] (checks 3-candle, then 2, then 1)."""
    hits = []
    c = candles[i]

    if i >= 2:
        c1, c2, c3 = candles[i - 2], candles[i - 1], candles[i]
        if match_morning_star(c1, c2, c3):
            hits.append("Morning Star")
        if match_evening_star(c1, c2, c3):
            hits.append("Evening Star")
        if match_three_white_soldiers(c1, c2, c3):
            hits.append("Three White Soldiers")
        if match_three_black_crows(c1, c2, c3):
            hits.append("Three Black Crows")

    if i >= 1:
        c1, c2 = candles[i - 1], candles[i]
        if match_bullish_engulfing(c1, c2):
            hits.append("Bullish Engulfing")
        if match_bearish_engulfing(c1, c2):
            hits.append("Bearish Engulfing")
        if match_piercing_line(c1, c2):
            hits.append("Piercing Line")
        if match_dark_cloud_cover(c1, c2):
            hits.append("Dark Cloud Cover")

    if match_doji(c):
        hits.append("Doji")
    if match_hammer(c):
        hits.append("Hammer")
    if match_shooting_star(c):
        hits.append("Shooting Star")
    marubozu = match_marubozu(c)
    if marubozu:
        hits.append(marubozu)

    return hits


def detect_patterns(candles):
    """
    Annotate every candle with the patterns ending on it.
    `candles` must be a chronologically ordered list of dicts with
    timestamp/open/high/low/close keys.
    """
    annotated = []
    for i, c in enumerate(candles):
        names = detect_patterns_at(candles, i)
        annotated.append(
            {
                **c,
                "patterns": [
                    {"name": n, "direction": PATTERN_WEIGHTS[n][0], "strength": PATTERN_WEIGHTS[n][1]}
                    for n in names
                ],
            }
        )
    return annotated

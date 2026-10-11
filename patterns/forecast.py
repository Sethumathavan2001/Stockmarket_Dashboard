"""
Heuristic next-candle projection, from now until the prediction cutoff
(default 3:15 PM IST).

This is intentionally a simple, transparent, RULE-BASED extrapolation — NOT
a trained statistical or machine-learning model. It:

  1. Detects candlestick patterns on today's actual candles (detectors.py).
  2. Turns the most recent pattern(s) into a signed bias score (recency-
     weighted — the latest candle's patterns count most).
  3. Sizes each projected move from recent realized volatility (average
     candle range over the last 12 candles).
  4. Decays the bias toward flat the further into the future it projects
     (BIAS_DAMPING), rather than drawing a straight line to the close.

For a rigorously validated statistical forecast (walk-forward tested
XGBoost, confidence scores tied to historical directional accuracy), see
the separate `prediction` app built earlier. This module is independent by
design — swap `generate_forecast` to call `prediction.predictor.predict_company`
instead if you'd rather drive this page's "predicted" candles off that
trained model; see README.
"""
import datetime

from .detectors import detect_patterns

DEFAULT_INTERVAL_MINUTES = 5
DEFAULT_CUTOFF_TIME = datetime.time(15, 15)  # 3:15 PM IST, as specified
LOOKBACK_FOR_VOLATILITY = 12
BIAS_LOOKBACK_CANDLES = 3   # how many recent candles' patterns feed the bias
BIAS_DAMPING = 0.85         # bias shrinks with each further-out predicted candle


def _bias_from_recent_patterns(annotated_candles):
    """Combine pattern hits from the last BIAS_LOOKBACK_CANDLES candles into
    one signed bias score; more recent candles are weighted more heavily."""
    recent = annotated_candles[-BIAS_LOOKBACK_CANDLES:]
    score = 0.0
    matched = []
    for age, candle in enumerate(reversed(recent)):
        recency_weight = 1.0 / (1 + age)
        for p in candle["patterns"]:
            sign = 1 if p["direction"] == "bullish" else (-1 if p["direction"] == "bearish" else 0)
            score += sign * p["strength"] * recency_weight
            if p["direction"] != "neutral":
                matched.append(
                    {"name": p["name"], "timestamp": candle["timestamp"].isoformat(), "direction": p["direction"]}
                )
    return score, matched


def _volatility_step(actual_candles):
    recent = actual_candles[-LOOKBACK_FOR_VOLATILITY:] or actual_candles
    ranges = [c["high"] - c["low"] for c in recent if c["high"] is not None and c["low"] is not None]
    if ranges:
        avg_range = sum(ranges) / len(ranges)
        if avg_range > 0:
            return avg_range * 0.5
    last_close = actual_candles[-1]["close"] if actual_candles else 0
    return max(last_close * 0.0008, 0.01)


def generate_forecast(actual_candles, cutoff_time=DEFAULT_CUTOFF_TIME, interval_minutes=DEFAULT_INTERVAL_MINUTES):
    """
    actual_candles: chronologically ordered list of dicts with tz-aware
    `timestamp` plus open/high/low/close, for TODAY only (9:15 -> now).

    Returns (annotated_actual_candles, predicted_candles, bias_summary).
    """
    if not actual_candles:
        return [], [], {"bias_score": 0.0, "direction": "neutral", "matched_patterns": []}

    annotated = detect_patterns(actual_candles)
    bias_score, matched_patterns = _bias_from_recent_patterns(annotated)
    step = _volatility_step(actual_candles)

    direction = "neutral"
    if bias_score > 0.15:
        direction = "bullish"
    elif bias_score < -0.15:
        direction = "bearish"

    sign = 1 if direction == "bullish" else (-1 if direction == "bearish" else 0)
    magnitude_scale = min(abs(bias_score), 4.0)  # cap runaway magnitude

    predicted = []
    last_actual = actual_candles[-1]
    current_open = last_actual["close"]
    ts = last_actual["timestamp"] + datetime.timedelta(minutes=interval_minutes)
    step_index = 0

    while ts.time() <= cutoff_time:
        move = sign * step * magnitude_scale * (BIAS_DAMPING ** step_index)
        close = current_open + move
        wick = step * 0.15
        high = max(current_open, close) + wick
        low = min(current_open, close) - wick
        predicted.append(
            {
                "timestamp": ts,
                "open": round(current_open, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "is_predicted": True,
            }
        )
        current_open = close
        ts += datetime.timedelta(minutes=interval_minutes)
        step_index += 1

    bias_summary = {
        "bias_score": round(bias_score, 3),
        "direction": direction,
        "matched_patterns": matched_patterns,
    }
    return annotated, predicted, bias_summary

"""Black-Scholes price and Greeks (European, no dividends) - what brokers' option chains show."""
import math

RATE = 0.065                  # risk-free rate (about the 91-day T-bill yield)
MIN_T = 1 / (365 * 24 * 60)   # one minute - avoids division by zero at expiry


def years_to(expiry_dt, now):
    return max((expiry_dt - now).total_seconds() / (365 * 24 * 3600), 0.0)


def _n(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def _cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def price(spot, strike, t, iv_pct, option_type, r=RATE):
    """Theoretical premium; at expiry (t=0) or with no IV it is the intrinsic value."""
    intrinsic = max(0.0, spot - strike) if option_type == "CE" else max(0.0, strike - spot)
    sigma = (iv_pct or 0) / 100
    if t <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return intrinsic
    sq = sigma * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + sigma * sigma / 2) * t) / sq
    d2 = d1 - sq
    if option_type == "CE":
        return spot * _cdf(d1) - strike * math.exp(-r * t) * _cdf(d2)
    return strike * math.exp(-r * t) * _cdf(-d2) - spot * _cdf(-d1)


def greeks(spot, strike, t, iv_pct, option_type, r=RATE):
    """delta, gamma, theta (per day), vega (per 1 IV point). None values when IV / spot are missing."""
    sigma = (iv_pct or 0) / 100
    if spot <= 0 or strike <= 0 or sigma <= 0:
        return {"delta": None, "gamma": None, "theta": None, "vega": None}
    t = max(t, MIN_T)
    sq = sigma * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + sigma * sigma / 2) * t) / sq
    d2 = d1 - sq
    disc = math.exp(-r * t)
    gamma = _n(d1) / (spot * sq)
    vega = spot * _n(d1) * math.sqrt(t) / 100
    if option_type == "CE":
        delta = _cdf(d1)
        theta = (-spot * _n(d1) * sigma / (2 * math.sqrt(t)) - r * strike * disc * _cdf(d2)) / 365
    else:
        delta = _cdf(d1) - 1
        theta = (-spot * _n(d1) * sigma / (2 * math.sqrt(t)) + r * strike * disc * _cdf(-d2)) / 365
    return {"delta": round(delta, 4), "gamma": round(gamma, 6), "theta": round(theta, 3), "vega": round(vega, 3)}

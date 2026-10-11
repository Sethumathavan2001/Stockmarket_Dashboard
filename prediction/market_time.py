"""
NSE (India) market session state and data-staleness helpers.

All times are handled in Asia/Kolkata regardless of server timezone, since
mixing naive/UTC timestamps with IST market hours is a common source of bugs.
"""
import datetime
from zoneinfo import ZoneInfo

from django.conf import settings

IST = ZoneInfo("Asia/Kolkata")

MARKET_OPEN = datetime.time(9, 15)
MARKET_CLOSE = datetime.time(15, 30)
PRE_MARKET_START = datetime.time(9, 0)
POST_MARKET_END = datetime.time(16, 0)


def _to_ist(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def get_market_status(now: datetime.datetime = None) -> str:
    """
    Returns one of: WEEKEND, HOLIDAY, PRE_MARKET, OPEN, POST_MARKET, CLOSED.

    Holidays are configured via settings.MARKET_HOLIDAYS, a list of
    "YYYY-MM-DD" strings (exchange holiday calendar changes yearly, so this
    is intentionally left as plain config rather than hard-coded).
    """
    now = _to_ist(now or datetime.datetime.now(IST))

    if now.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return "WEEKEND"

    holidays = getattr(settings, "MARKET_HOLIDAYS", [])
    if now.date().isoformat() in holidays:
        return "HOLIDAY"

    t = now.time()
    if PRE_MARKET_START <= t < MARKET_OPEN:
        return "PRE_MARKET"
    if MARKET_OPEN <= t <= MARKET_CLOSE:
        return "OPEN"
    if MARKET_CLOSE < t <= POST_MARKET_END:
        return "POST_MARKET"
    return "CLOSED"


def is_stale(latest_ts: datetime.datetime, now: datetime.datetime = None,
             max_staleness_minutes: int = 15) -> bool:
    """True if the latest candle we have is older than max_staleness_minutes."""
    now = _to_ist(now or datetime.datetime.now(IST))
    latest_ts = _to_ist(latest_ts)
    return (now - latest_ts).total_seconds() / 60.0 > max_staleness_minutes

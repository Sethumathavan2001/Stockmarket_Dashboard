"""
Paper-trading engine.

Accounting model (all Decimal, 4 dp, exact — no float drift)
-------------------------------------------------------------
Opening or adding to a position BLOCKS its notional (qty x price) from the
wallet. Closing releases the blocked amount for the closed units plus/minus
realized P&L:

    long  close:  wallet += released + (proceeds - released)   = proceeds
    short close:  wallet += released + (released - proceeds)

so "wallet + blocked basis" only ever changes by realized P&L. Short selling
therefore needs the same wallet balance as buying (100% margin, no leverage).
A single order that crosses zero (e.g. sell 15 while long 10) closes the old
side first and opens the remainder on the new side; the remainder must be
affordable with the wallet *after* the close.

Products
--------
* INTRADAY : may be sold short; closed automatically at the end of the day
             (Old mode: when the replay reaches the close; Live: at the
             square-off time, 15:20 IST by default - see squareoff_live_intraday).
* DELIVERY : long only (a SELL must be covered by a holding); never
             auto-closed.  Live holdings stay in the "live" book.  Old-mode
             holdings live in ONE shared "replay" book so they carry from one
             replay date to the next - but can't be traded at a replay time
             earlier than their own last trade.

Price rules (the server never trusts a client-supplied price blindly)
---------------------------------------------------------------------
* live   : price = last yfinance 5-min close, fetched server-side; only while
           the market is OPEN and the data is fresh. Never written to
           FiveMinuteData / DailyData.  The "previous day" O/H/L/C lines in
           Live mode also come from yfinance (daily candles), NOT the database.
* replay : the client sends the candle timestamp it is "at" plus its current
           simulated tick price. The server looks the candle up in
           FiveMinuteData and rejects any price outside that candle's
           [low, high] range.
"""
import logging
from datetime import date as date_cls, datetime, time, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from zoneinfo import ZoneInfo

from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from django.db.models import Max, Q, Sum

import pandas as pd

from market.models import DailyData, FiveMinuteData  # adjust if your app is named differently
from prediction.data import load_today_intraday_yfinance
from prediction.market_time import get_market_status, is_stale
from prediction.tickers import get_yahoo_ticker

from .models import PaperAccount, PaperOrder, PaperPosition

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
LIVE = PaperPosition.MODE_LIVE
REPLAY = PaperPosition.MODE_REPLAY
INTRADAY = PaperPosition.PRODUCT_INTRADAY
DELIVERY = PaperPosition.PRODUCT_DELIVERY
PRODUCTS = (INTRADAY, DELIVERY)
REPLAY_HOLDINGS = "replay"            # shared context for Old-mode DELIVERY holdings

TWO = Decimal("0.01")
FOUR = Decimal("0.0001")
ZERO = Decimal("0")
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)
MAX_QUANTITY = 1_000_000
MAX_WALLET_ADD = Decimal("100000000")  # ₹10 crore per top-up


class TradeError(Exception):
    """A user-facing failure (bad input, insufficient funds, market closed...)."""


# --------------------------------------------------------------------------
# settings helpers
# --------------------------------------------------------------------------
def live_cache_seconds():
    return getattr(settings, "PAPER_LIVE_CACHE_SECONDS", 10)


def live_max_staleness_minutes():
    # A 5-minute candle is stamped with its START time and Yahoo can lag, so
    # anything under ~10 minutes would reject perfectly healthy data.
    return getattr(settings, "PAPER_LIVE_MAX_STALENESS_MINUTES", 20)


def intraday_squareoff_time():
    """Live INTRADAY positions are closed from this IST time on (default 15:20)."""
    raw = getattr(settings, "PAPER_INTRADAY_SQUAREOFF_TIME", "15:20")
    try:
        hh, mm = str(raw).split(":")
        return time(int(hh), int(mm))
    except (ValueError, TypeError):
        return time(15, 20)


MAX_QUOTE_COMPANIES = 40


# --------------------------------------------------------------------------
# time helpers
# --------------------------------------------------------------------------
def to_ist(dt):
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def to_db_dt(dt):
    """Datetime shaped the way the DB layer expects (aware if USE_TZ else naive IST)."""
    dt = to_ist(dt)
    return dt if settings.USE_TZ else dt.replace(tzinfo=None)


def today_ist():
    return datetime.now(IST).date()


def session_bounds(day):
    return (
        datetime.combine(day, SESSION_OPEN, tzinfo=IST),
        datetime.combine(day, SESSION_CLOSE, tzinfo=IST),
    )


def _day_bounds_db(day):
    start = datetime.combine(day, time.min, tzinfo=IST)
    return to_db_dt(start), to_db_dt(start + timedelta(days=1))


# --------------------------------------------------------------------------
# accounts & context
# --------------------------------------------------------------------------
def get_account(request):
    if getattr(settings, "PAPER_TRADING_SHARED_ACCOUNT", False):
        key = "shared"
    elif getattr(request, "user", None) is not None and request.user.is_authenticated:
        key = f"user:{request.user.pk}"
    else:
        if not request.session.session_key:
            request.session.create()
        key = f"session:{request.session.session_key}"
    account, _ = PaperAccount.objects.get_or_create(owner_key=key)
    return account


def make_context(mode, replay_date):
    """The day (or "live") an ORDER belongs to - used for the Orders tab and replay resume."""
    return LIVE if mode == LIVE else replay_date.isoformat()


def position_context(mode, replay_date, product):
    """Where a POSITION is kept (see PaperPosition.context)."""
    if mode == LIVE:
        return LIVE
    return REPLAY_HOLDINGS if product == DELIVERY else replay_date.isoformat()


def parse_product(raw):
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return INTRADAY
    product = str(raw).strip().upper()
    if product not in PRODUCTS:
        raise TradeError("Product must be INTRADAY or DELIVERY.")
    return product


def parse_replay_date(raw):
    try:
        day = date_cls.fromisoformat(str(raw))
    except (TypeError, ValueError):
        raise TradeError("Enter a valid date (YYYY-MM-DD).")
    if day >= today_ist():
        raise TradeError("Old mode needs a previous trading day (before today). Use Live for today.")
    return day


def parse_quantity(raw):
    try:
        if isinstance(raw, bool):
            raise ValueError
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError
        qty = int(raw)
    except (TypeError, ValueError):
        raise TradeError("Quantity must be a whole number.")
    if not 1 <= qty <= MAX_QUANTITY:
        raise TradeError(f"Quantity must be between 1 and {MAX_QUANTITY:,}.")
    return qty


def _parse_price(raw):
    try:
        price = Decimal(str(raw)).quantize(TWO, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError, TypeError):
        raise TradeError("Invalid price.")
    if not price.is_finite() or price <= 0:
        raise TradeError("Invalid price.")
    return price


def parse_optional_price(raw, label):
    """Blank / None -> None.  Otherwise a positive, finite Decimal (2 dp)."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        value = Decimal(str(raw)).quantize(TWO, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError, TypeError):
        raise TradeError(f"{label}: enter a valid price.")
    if not value.is_finite() or value <= 0:
        raise TradeError(f"{label}: price must be greater than zero.")
    return value


def parse_optional_percent(raw, label, max_pct):
    """Blank / None -> None.  Otherwise a Decimal strictly between 0 and max_pct."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError, TypeError):
        raise TradeError(f"{label}: enter a valid percentage.")
    if not value.is_finite() or value <= 0 or value >= max_pct:
        raise TradeError(f"{label}: percentage must be above 0 and below {max_pct}.")
    return value


MAX_SL_PERCENT = Decimal("100")       # a long's stop can't be 100% away (price would hit 0)
MAX_TARGET_PERCENT = Decimal("1000")


def parse_bracket(body):
    """Pull the optional SL / target inputs out of a request body."""
    return {
        "sl_percent": parse_optional_percent(body.get("stop_loss_percent"), "Stop loss", MAX_SL_PERCENT),
        "sl_price": parse_optional_price(body.get("stop_loss_price"), "Stop loss"),
        "target_percent": parse_optional_percent(body.get("target_percent"), "Target", MAX_TARGET_PERCENT),
        "target_price": parse_optional_price(body.get("target_price"), "Target"),
    }


def resolve_levels(sign, anchor, *, sl_percent=None, sl_price=None, target_percent=None, target_price=None):
    """
    Turn the user's SL / target inputs into absolute prices for a position
    whose direction is `sign` (+1 long, -1 short) and whose average entry
    price is `anchor`.  A percentage wins over a typed price (it is measured
    from the average entry price); the result is validated so the stop is on
    the losing side and the target on the winning side.
    Returns (stop_loss, target) - each a Decimal or None.
    """
    sl = tgt = None
    if sl_percent is not None:
        sl = anchor * (1 - sign * sl_percent / 100)
    elif sl_price is not None:
        sl = sl_price
    if target_percent is not None:
        tgt = anchor * (1 + sign * target_percent / 100)
    elif target_price is not None:
        tgt = target_price

    if sl is not None:
        sl = sl.quantize(TWO, rounding=ROUND_HALF_UP)
        if sl <= 0:
            raise TradeError("Stop loss: that works out to a price of zero or less.")
        if sign > 0 and sl >= anchor:
            raise TradeError(f"Stop loss must be below your entry price (₹{anchor:,.2f}) for a long position.")
        if sign < 0 and sl <= anchor:
            raise TradeError(f"Stop loss must be above your entry price (₹{anchor:,.2f}) for a short position.")
    if tgt is not None:
        tgt = tgt.quantize(TWO, rounding=ROUND_HALF_UP)
        if tgt <= 0:
            raise TradeError("Target: that works out to a price of zero or less.")
        if sign > 0 and tgt <= anchor:
            raise TradeError(f"Target must be above your entry price (₹{anchor:,.2f}) for a long position.")
        if sign < 0 and tgt >= anchor:
            raise TradeError(f"Target must be below your entry price (₹{anchor:,.2f}) for a short position.")
    return sl, tgt


def check_levels_hit(quantity, stop_loss, target_price, price):
    """'stop_loss' / 'target' / None for a position at `price` (informational helper)."""
    if not quantity:
        return None
    long_ = quantity > 0
    if stop_loss is not None and ((long_ and price <= stop_loss) or (not long_ and price >= stop_loss)):
        return "stop_loss"
    if target_price is not None and ((long_ and price >= target_price) or (not long_ and price <= target_price)):
        return "target"
    return None


# --------------------------------------------------------------------------
# wallet
# --------------------------------------------------------------------------
def add_to_wallet(account, raw_amount):
    try:
        amount = Decimal(str(raw_amount)).quantize(TWO, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError, TypeError):
        raise TradeError("Enter a valid amount.")
    if not amount.is_finite() or amount <= 0:
        raise TradeError("Amount must be greater than zero.")
    if amount > MAX_WALLET_ADD:
        raise TradeError(f"You can add at most ₹{MAX_WALLET_ADD:,.0f} at a time.")

    with transaction.atomic():
        locked = PaperAccount.objects.select_for_update().get(pk=account.pk)
        locked.balance += amount
        locked.save(update_fields=["balance"])
    return locked


# --------------------------------------------------------------------------
# market data: replay (DB) and live (yfinance, in memory only)
# --------------------------------------------------------------------------
def _candle(ts, o, h, l, c):
    return {
        "x": to_ist(ts).isoformat(),
        "o": float(o),
        "h": float(h),
        "l": float(l),
        "c": float(c),
    }


def previous_day_ohlc(company, reference_date):
    """
    Open / High / Low / Close of the last trading day BEFORE `reference_date`.
    DailyData is the source of truth; if it has no complete row for that day we
    aggregate that day's FiveMinuteData instead.  None if nothing is stored.
    """
    prev = DailyData.objects.filter(company=company, date__lt=reference_date).order_by("-date").first()
    if prev and None not in (prev.open, prev.high, prev.low, prev.close):
        return {
            "date": prev.date.isoformat(),
            "open": float(prev.open),
            "high": float(prev.high),
            "low": float(prev.low),
            "close": float(prev.close),
        }

    start_of_ref, _ = _day_bounds_db(reference_date)
    last = FiveMinuteData.objects.filter(company=company, timestamp__lt=start_of_ref).order_by("-timestamp").first()
    if last is None:
        return None
    prev_day = to_ist(last.timestamp).date()
    start, end = _day_bounds_db(prev_day)
    rows = list(
        FiveMinuteData.objects.filter(company=company, timestamp__gte=start, timestamp__lt=end).order_by("timestamp")
    )
    highs = [r.high for r in rows if r.high is not None]
    lows = [r.low for r in rows if r.low is not None]
    opens = [r.open for r in rows if r.open is not None]
    closes = [r.close for r in rows if r.close is not None]
    if not (highs and lows and opens and closes):
        return None
    return {
        "date": prev_day.isoformat(),
        "open": float(opens[0]),
        "high": float(max(highs)),
        "low": float(min(lows)),
        "close": float(closes[-1]),
    }


def load_replay_candles(company, day):
    start, end = _day_bounds_db(day)
    rows = (
        FiveMinuteData.objects.filter(company=company, timestamp__gte=start, timestamp__lt=end)
        .order_by("timestamp")
        .values("timestamp", "open", "high", "low", "close")
    )
    return [
        _candle(r["timestamp"], r["open"], r["high"], r["low"], r["close"])
        for r in rows
        if None not in (r["open"], r["high"], r["low"], r["close"])
    ]


def fetch_live_today(company, use_cache=True):
    """Today's 5-minute candles from yfinance. In memory only — never persisted."""
    key = f"paper_trading:live:{company.pk}"
    if use_cache:
        cached = cache.get(key)
        if cached is not None:
            return cached

    df = load_today_intraday_yfinance(company)
    if not df.empty:
        # yfinance returns the last session on weekends/holidays; keep only today.
        df = df[df["timestamp"].dt.date == today_ist()].reset_index(drop=True)
    cache.set(key, df, live_cache_seconds())
    return df


def _yf_history(symbol, **kwargs):
    """Thin wrapper around yfinance so tests can replace it."""
    import yfinance as yf

    return yf.Ticker(symbol).history(**kwargs)


DAILY_CACHE_OK_SECONDS = 3600      # previous-day numbers don't change intraday
DAILY_CACHE_FAIL_SECONDS = 300     # retry a failed Yahoo call after 5 minutes


def live_daily_candles(company, use_cache=True):
    """
    Completed daily candles from Yahoo (about a month), oldest first, as
    [{"date": date, "open": f, "high": f, "low": f, "close": f}].  Today's
    (possibly partial) row is excluded.  [] if Yahoo has nothing / fails.
    In memory only - never stored in the database.
    """
    key = f"paper_trading:daily:{company.pk}"
    if use_cache:
        cached = cache.get(key)
        if cached is not None:
            return cached

    rows = []
    try:
        raw = _yf_history(get_yahoo_ticker(company.code), period="1mo", interval="1d", auto_adjust=False)
    except Exception as exc:  # network, rate limiting, delisted ticker ...
        logger.warning("yfinance daily fetch failed for %s: %s", company.code, exc)
        raw = None

    if raw is not None and not raw.empty:
        today = today_ist()
        for idx, r in raw.iterrows():
            ts = pd.Timestamp(idx)
            day = ts.tz_convert(IST).date() if ts.tzinfo is not None else ts.date()
            vals = [r.get("Open"), r.get("High"), r.get("Low"), r.get("Close")]
            if day >= today or any(v is None or pd.isna(v) for v in vals):
                continue
            rows.append({"date": day, "open": float(vals[0]), "high": float(vals[1]),
                         "low": float(vals[2]), "close": float(vals[3])})
        rows.sort(key=lambda x: x["date"])

    cache.set(key, rows, DAILY_CACHE_OK_SECONDS if rows else DAILY_CACHE_FAIL_SECONDS)
    return rows


def yahoo_5m_previous_day(company, use_cache=True):
    """
    OHLC of the last COMPLETE day before today built from Yahoo's 5-minute candles, as
    {"date": date, "open", "high", "low", "close"} or None.  Yahoo's daily feed often publishes
    a session's candle a day or more late, while the 5-minute data has it right after the close.
    In memory only - never stored in the database.
    """
    key = f"paper_trading:prev5m:{company.pk}:{today_ist().isoformat()}"
    if use_cache:
        cached = cache.get(key)
        if cached is not None:
            return cached or None                   # {} = looked up, nothing found

    found = {}
    try:
        raw = _yf_history(get_yahoo_ticker(company.code), period="5d", interval="5m", prepost=False)
    except Exception as exc:  # network, rate limiting, delisted ticker ...
        logger.warning("yfinance 5m previous-day fetch failed for %s: %s", company.code, exc)
        raw = None
    if raw is not None and not raw.empty:
        today = today_ist()
        by_day = {}
        for idx, r in raw.iterrows():
            ts = pd.Timestamp(idx)
            day = ts.tz_convert(IST).date() if ts.tzinfo is not None else ts.date()
            vals = [r.get("Open"), r.get("High"), r.get("Low"), r.get("Close")]
            if day >= today or any(v is None or pd.isna(v) for v in vals):
                continue
            o, h, l, c = (float(v) for v in vals)
            d = by_day.get(day)
            by_day[day] = ({"date": day, "open": o, "high": h, "low": l, "close": c} if d is None else
                           {**d, "high": max(d["high"], h), "low": min(d["low"], l), "close": c})
        if by_day:
            found = by_day[max(by_day)]
    cache.set(key, found, DAILY_CACHE_OK_SECONDS if found else DAILY_CACHE_FAIL_SECONDS)
    return found or None


def live_previous_day(company):
    """
    Previous trading day's OHLC for Live mode - from Yahoo, deliberately NOT the database.
    Uses Yahoo's daily candles, or its 5-minute candles when they already have a later day
    (the daily feed often lags a day).
    """
    rows = [r for r in live_daily_candles(company) if r["date"] < today_ist()]
    best = rows[-1] if rows else None
    intraday = yahoo_5m_previous_day(company)
    if intraday and (best is None or intraday["date"] > best["date"]):
        best = intraday
    if best is None:
        return None
    return {"date": best["date"].isoformat(), "open": best["open"], "high": best["high"], "low": best["low"],
            "close": best["close"]}


def daily_close_on(company, day):
    for r in live_daily_candles(company):
        if r["date"] == day:
            return r["close"]
    return None


def _live_status(df):
    """ltp / staleness / tradability from today's in-memory candles."""
    status = get_market_status()
    ltp, stale, last_ts = None, True, None
    if not df.empty:
        last_ts = df.iloc[-1]["timestamp"].to_pydatetime()
        ltp = round(float(df.iloc[-1]["close"]), 2)
        stale = is_stale(last_ts, max_staleness_minutes=live_max_staleness_minutes())
    return {
        "ltp": ltp,
        "market_status": status,
        "stale": stale,
        "tradable": bool(status == "OPEN" and not stale and ltp is not None),
        "last_candle_time": to_ist(last_ts).isoformat() if last_ts else None,
    }


def live_quote(company):
    """Light-weight live price for one company (no candle list)."""
    quote = _live_status(fetch_live_today(company))
    quote["company"] = company.code
    return quote


def live_snapshot(company):
    df = fetch_live_today(company)
    candles = []
    if not df.empty:
        candles = [_candle(r.timestamp, r.open, r.high, r.low, r.close) for r in df.itertuples(index=False)]

    start, end = session_bounds(today_ist())
    snap = _live_status(df)
    snap.update({
        "mode": LIVE,
        "candles": candles,
        "as_of": datetime.now(IST).isoformat(),
        "session_start": start.isoformat(),
        "session_end": end.isoformat(),
        "previous_day": live_previous_day(company),
    })
    return snap


def replay_snapshot(company, day):
    candles = load_replay_candles(company, day)
    if not candles:
        raise TradeError(
            f"No 5-minute candles found for {company.code} on {day.isoformat()}. "
            "It may not have been a trading day, or that date isn't in your database."
        )
    start, end = session_bounds(day)
    return {
        "mode": REPLAY,
        "date": day.isoformat(),
        "candles": candles,
        "session_start": start.isoformat(),
        "session_end": end.isoformat(),
        "previous_day": previous_day_ohlc(company, day),
    }


# --------------------------------------------------------------------------
# execution price resolution
# --------------------------------------------------------------------------
def resolve_live_execution(company):
    status = get_market_status()
    if status != "OPEN":
        raise TradeError(
            f"The market is {status.replace('_', ' ').lower()}. Live orders are only accepted "
            "while it is open — switch to Old mode to practise on a previous day."
        )
    df = fetch_live_today(company, use_cache=False)
    if df.empty:
        raise TradeError("Live price is unavailable right now. Please try again shortly.")

    last = df.iloc[-1]
    ts = last["timestamp"].to_pydatetime()
    if is_stale(ts, max_staleness_minutes=live_max_staleness_minutes()):
        raise TradeError("Live data looks stale, so the order was not placed.")
    return _parse_price(round(float(last["close"]), 2)), ts


def resolve_replay_execution(company, replay_date, market_time_raw, price_raw):
    try:
        mt = to_ist(datetime.fromisoformat(str(market_time_raw)))
    except (TypeError, ValueError):
        raise TradeError("Invalid replay time.")
    if mt.date() != replay_date:
        raise TradeError("That replay time doesn't belong to the selected date.")

    candle = FiveMinuteData.objects.filter(company=company, timestamp=to_db_dt(mt)).first()
    if candle is None or candle.low is None or candle.high is None:
        raise TradeError("No candle exists at that replay time.")

    price = _parse_price(price_raw)
    if price < candle.low - TWO or price > candle.high + TWO:
        raise TradeError("Price is outside that candle's range, so the order was rejected.")
    return price, mt


# --------------------------------------------------------------------------
# order execution
# --------------------------------------------------------------------------
@transaction.atomic
def execute_order(*, account, company, side, quantity, price, mode, replay_date, market_time,
                  levels=None, reason="", product=INTRADAY):
    account = PaperAccount.objects.select_for_update().get(pk=account.pk)
    context = make_context(mode, replay_date)                    # the order's day / "live"
    pos, _ = PaperPosition.objects.select_for_update().get_or_create(
        account=account,
        company=company,
        context=position_context(mode, replay_date, product),     # where the holding is kept
        product=product,
        defaults={"mode": mode, "quantity": 0, "cost_basis": ZERO},
    )

    q, basis, balance = pos.quantity, pos.cost_basis, account.balance
    old_q = q
    direction = 1 if side == PaperOrder.SIDE_BUY else -1
    price = price.quantize(FOUR)

    if product == DELIVERY:
        # Delivery is long-only: a sell must be covered by what you hold.
        if side == PaperOrder.SIDE_SELL:
            if q <= 0:
                raise TradeError(
                    f"You don't hold {company.code} in Delivery, so there is nothing to sell. "
                    "Switch to Intraday if you want to short sell."
                )
            if quantity > q:
                raise TradeError(
                    f"You hold only {q} of {company.code} in Delivery, so you can't sell {quantity}. "
                    "Delivery has no short selling."
                )
        # A replay holding can't be traded at a time before its own last trade.
        if mode == REPLAY and q != 0 and pos.last_entry_at and market_time is not None:
            if to_ist(market_time) < to_ist(pos.last_entry_at):
                raise TradeError(
                    f"This replay time is before your last Delivery trade in {company.code} "
                    f"({to_ist(pos.last_entry_at):%d %b %Y %H:%M}). Move forward in time or pick a later date."
                )

    # Units that reduce an existing opposite-side position vs. units that open/add.
    if q != 0 and (q > 0) != (direction > 0):
        closing = min(quantity, abs(q))
    else:
        closing = 0
    opening = quantity - closing

    realized = ZERO
    if closing:
        released = basis if closing == abs(q) else (basis * closing / abs(q)).quantize(FOUR, ROUND_HALF_UP)
        proceeds = price * closing
        realized = (proceeds - released) if q > 0 else (released - proceeds)
        balance += released + realized
        basis -= released
        q += -closing if q > 0 else closing

    if opening:
        cost = price * opening
        if cost > balance:
            raise TradeError(
                f"Insufficient wallet balance: this order needs ₹{cost:,.2f} "
                f"but only ₹{balance:,.2f} is available. Add money to your wallet."
            )
        balance -= cost
        basis += cost
        q += direction * opening

    if q == 0:
        basis = ZERO

    entered_at = market_time or datetime.now(IST)
    if q == 0:
        pos.last_entry_at = None
    elif opening:                       # opened, added to, or flipped into a new position
        pos.last_entry_at = to_db_dt(entered_at)

    # ---- stop loss / target -------------------------------------------------
    # * flat after this order          -> levels are cleared
    # * entry-like (opened, added to, or flipped into this order's direction)
    #                                  -> new levels apply if the user supplied any
    # * plain partial exit (still on the old side) -> existing levels stay as they are
    levels = levels or {}
    has_levels = any(levels.get(k) is not None for k in ("sl_percent", "sl_price", "target_percent", "target_price"))
    update_fields = ["quantity", "cost_basis", "last_entry_at", "updated_at"]
    if q == 0:
        pos.stop_loss = pos.target_price = None
        update_fields += ["stop_loss", "target_price"]
    elif has_levels and ((q > 0 and direction > 0) or (q < 0 and direction < 0)):
        anchor = basis / abs(q)
        sl, tgt = resolve_levels(1 if q > 0 else -1, anchor, **levels)
        if sl is not None:
            pos.stop_loss = sl
            update_fields.append("stop_loss")
        if tgt is not None:
            pos.target_price = tgt
            update_fields.append("target_price")
    elif q != 0 and (q > 0) != (pos.quantity > 0) and pos.quantity != 0:
        # flipped into the opposite side without new levels: old levels no longer make sense
        pos.stop_loss = pos.target_price = None
        update_fields += ["stop_loss", "target_price"]

    pos.quantity, pos.cost_basis = q, basis
    pos.save(update_fields=list(dict.fromkeys(update_fields)))
    account.balance = balance
    account.save(update_fields=["balance"])

    order = PaperOrder.objects.create(
        account=account,
        company=company,
        side=side,
        product=product,
        quantity=quantity,
        price=price,
        mode=mode,
        context=context,
        market_time=to_db_dt(market_time) if market_time else None,
        realized_pnl=realized,
        balance_after=balance,
        reason=reason or "",
    )
    return order, pos, account


@transaction.atomic
def update_bracket(*, account, company, mode, replay_date, levels, clear_sl=False, clear_target=False,
                   product=INTRADAY):
    """
    Change (or clear) the stop loss / target of an OPEN position without
    trading.  Percentages are measured from the position's average entry price.
    Fields left blank are not touched; `clear_*` removes that level.
    """
    pos = (
        PaperPosition.objects.select_for_update()
        .filter(account=account, company=company, context=position_context(mode, replay_date, product),
                product=product)
        .exclude(quantity=0)
        .first()
    )
    if pos is None:
        raise TradeError(f"You have no open {product.title()} position in this company to attach a stop loss or target to.")

    sign = 1 if pos.quantity > 0 else -1
    sl, tgt = resolve_levels(sign, pos.avg_price, **(levels or {}))
    if clear_sl:
        pos.stop_loss = None
    elif sl is not None:
        pos.stop_loss = sl
    if clear_target:
        pos.target_price = None
    elif tgt is not None:
        pos.target_price = tgt
    pos.save(update_fields=["stop_loss", "target_price", "updated_at"])
    return pos


# --------------------------------------------------------------------------
# live intraday square-off
# --------------------------------------------------------------------------
def squareoff_live_intraday(account, now=None):
    """
    Close Live INTRADAY positions that must not stay open:

    * opened on an EARLIER day  -> settled at that day's closing price (Yahoo daily
      candle), as a broker's auto square-off would have done at that day's end;
    * opened TODAY and it is at/after the square-off time (15:20 IST by default)
      -> closed at the latest live price (even if the data is no longer "fresh",
      since the market has closed).

    Delivery positions are never touched.  Returns {"closed": [orders], "skipped": [str]}.
    The page calls this when a Live session starts and on each live refresh after the cutoff.
    """
    now = to_ist(now or datetime.now(IST))
    cutoff = intraday_squareoff_time()
    positions = list(
        PaperPosition.objects.filter(account=account, context=LIVE, product=INTRADAY)
        .exclude(quantity=0)
        .select_related("company")
    )
    closed, skipped = [], []
    for pos in positions:
        entry_day = to_ist(pos.last_entry_at).date() if pos.last_entry_at else now.date()
        if entry_day < now.date():
            close = daily_close_on(pos.company, entry_day)
            if close is None:
                skipped.append(f"{pos.company.code}: no closing price available for {entry_day.isoformat()} yet")
                continue
            price, when = close, datetime.combine(entry_day, SESSION_CLOSE, tzinfo=IST)
        elif now.time() >= cutoff:
            df = fetch_live_today(pos.company, use_cache=False)
            if df.empty:
                skipped.append(f"{pos.company.code}: no live price available to square off at")
                continue
            price = round(float(df.iloc[-1]["close"]), 2)
            when = df.iloc[-1]["timestamp"].to_pydatetime()
        else:
            continue

        order, _, _ = execute_order(
            account=account,
            company=pos.company,
            side=PaperOrder.SIDE_SELL if pos.quantity > 0 else PaperOrder.SIDE_BUY,
            quantity=abs(pos.quantity),
            price=_parse_price(price),
            mode=LIVE,
            replay_date=None,
            market_time=when,
            reason="session_end",
            product=INTRADAY,
        )
        order.company = pos.company
        closed.append(order)
    return {"closed": closed, "skipped": skipped}


# --------------------------------------------------------------------------
# serialisation / snapshots
# --------------------------------------------------------------------------
def serialize_position(pos):
    q = pos.quantity
    return {
        "company": pos.company.code,
        "name": pos.company.name,
        "product": pos.product,
        "quantity": q,
        "side": "LONG" if q > 0 else "SHORT",
        "avg_price": float(pos.avg_price),
        "cost_basis": float(pos.cost_basis),
        "stop_loss": float(pos.stop_loss) if pos.stop_loss is not None else None,
        "target_price": float(pos.target_price) if pos.target_price is not None else None,
        "last_entry_at": to_ist(pos.last_entry_at).isoformat() if pos.last_entry_at else None,
        "mode": pos.mode,
        "context": pos.context,
    }


def serialize_order(o):
    return {
        "id": o.id,
        "company": o.company.code,
        "side": o.side,
        "product": o.product,
        "quantity": o.quantity,
        "price": float(o.price),
        "mode": o.mode,
        "context": o.context,
        "market_time": to_ist(o.market_time).isoformat() if o.market_time else None,
        "realized_pnl": float(o.realized_pnl),
        "balance_after": float(o.balance_after),
        "reason": o.reason,
        "created_at": to_ist(o.created_at).isoformat(),
    }


def visible_positions(account, mode, replay_date=None):
    """
    Every open position (ALL companies, both products) that belongs on screen for this
    session: Live -> the live book; Old -> that day's intraday positions plus Delivery
    holdings bought on or before that date.
    """
    qs = PaperPosition.objects.filter(account=account).exclude(quantity=0).select_related("company")
    if mode == LIVE:
        qs = qs.filter(context=LIVE)
    else:
        _, next_day = _day_bounds_db(replay_date)
        qs = qs.filter(
            Q(context=replay_date.isoformat(), product=INTRADAY)
            | Q(context=REPLAY_HOLDINGS, product=DELIVERY, last_entry_at__lt=next_day)
            | Q(context=REPLAY_HOLDINGS, product=DELIVERY, last_entry_at__isnull=True)
        )
    return list(qs.order_by("company__code", "product"))


def account_snapshot(account, mode, replay_date=None, company=None):
    context = make_context(mode, replay_date)
    positions = visible_positions(account, mode, replay_date)
    blocked = PaperPosition.objects.filter(account=account).aggregate(s=Sum("cost_basis"))["s"] or ZERO
    realized = PaperOrder.objects.filter(account=account).aggregate(s=Sum("realized_pnl"))["s"] or ZERO
    orders = list(PaperOrder.objects.filter(account=account, context=context).select_related("company")[:50])

    # Resume a replay where the user left off - the clock is shared by every stock on that
    # date, so use the latest order time across all of them - but only while something
    # from THIS day is still open (otherwise a fresh run from 9:15 makes more sense).
    resume = None
    if mode == REPLAY:
        day_iso = replay_date.isoformat()
        anchored = [
            p for p in positions
            if p.context == day_iso
            or (p.product == DELIVERY and p.last_entry_at and to_ist(p.last_entry_at).date() == replay_date)
        ]
        if anchored:
            latest = (
                PaperOrder.objects.filter(account=account, context=context, market_time__isnull=False)
                .aggregate(m=Max("market_time"))["m"]
            )
            if latest:
                resume = to_ist(latest).isoformat()

    return {
        "balance": float(account.balance),
        "blocked": float(blocked),
        "realized_pnl": float(realized),
        "positions": [serialize_position(p) for p in positions],
        "orders": [serialize_order(o) for o in orders],
        "resume_market_time": resume,
    }

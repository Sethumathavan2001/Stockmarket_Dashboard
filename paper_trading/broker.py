"""
Real Angel One trades from the Claude Orders page (accounts in angel.ACCOUNTS).

  place()   stop loss / target are PERCENTAGES from the entry; checks (live trading switched on, configured,
            market hours 09:15-15:00, not already placed today, order value cap, percentages 0.05-20 %, the
            price not already past the recommendation's stop / target), then
            1. MARKET INTRADAY entry
            2. once filled, the stop / target prices are worked out from the actual fill price, and a
               STOPLOSS-LIMIT order is placed for the stop - it sits at Angel One, so the stop works even
               with this server off
  watch()   every monitor round (paper_trading/monitor.py): finishes entries that were still pending,
            notices the stop-loss order filling, and exits at the target or at 15:15 - always cancelling
            the stop-loss order FIRST and exiting only once Angel One confirms it is cancelled, so the two
            exits can never both happen.
Angel One also squares off intraday positions on its own near 15:20 (with a fee); exiting at 15:15 avoids it.
"""
import logging
import threading
from datetime import datetime, time
from decimal import Decimal, InvalidOperation

from prediction.market_time import IST, get_market_status

from . import angel, surveillance
from .models import BrokerOrder as BO

logger = logging.getLogger(__name__)

ENTRY_FROM, ENTRY_UNTIL, SQUAREOFF = time(9, 15), time(15, 0), time(15, 15)
MAX_QTY = 100000
MIN_PCT, MAX_PCT = Decimal("0.05"), Decimal("20")
PAISA = Decimal("0.01")
LOCK = threading.RLock()


class BrokerError(Exception):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


def now_ist():
    return datetime.now(IST)


def opposite(side):
    return "SELL" if side == "BUY" else "BUY"


def name(account):
    return angel.ACCOUNTS.get(account, account)


def placed_today(day):
    """{(account, code, side): BrokerOrder} - today's trades that count (rejected ones don't)."""
    qs = BO.objects.filter(session=day).exclude(status=BO.REJECTED).order_by("created_at")      # ERROR = maybe sent
    return {(o.account, o.code, o.side): o for o in qs}


def _pct(raw, label):
    try:
        v = Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError):
        raise BrokerError(f"{label} must be a percentage.")
    if not MIN_PCT <= v <= MAX_PCT:
        raise BrokerError(f"{label} must be between {MIN_PCT}% and {MAX_PCT}%.")
    return v


def levels(side, price, stop_pct, target_pct):
    """(stop, target) prices for an entry at `price`."""
    sign = 1 if side == "BUY" else -1
    price = Decimal(str(price))
    return ((price * (1 - sign * stop_pct / 100)).quantize(PAISA), (price * (1 + sign * target_pct / 100)).quantize(PAISA))


def place(account, *, rec_id, code, side, quantity, stop_pct, target_pct, guard=None, consent=False, now=None):
    """guard: the recommendation's (stop, target) prices - refuse when the live price is already past them.
    consent: you accepted the exchange's cautionary warning for this stock - sent to Angel One as
    "scripconsent": "yes" on this trade's entry, stop-loss and exit orders."""
    now = now or now_ist()
    if account not in angel.ACCOUNTS:
        raise BrokerError("Unknown Angel One account.")
    _, missing = angel.config(account)
    if missing:
        raise BrokerError(f"{name(account)}'s Angel One account is not set up - add {', '.join(missing)} to .env "
                          "and restart the server.")
    if not angel.live_enabled():
        raise BrokerError("Real orders are switched off. Add ANGEL_LIVE_TRADING=1 to .env and restart the server "
                          "to allow them.")
    if get_market_status(now) != "OPEN" or not ENTRY_FROM <= now.time() < ENTRY_UNTIL:
        raise BrokerError("Real intraday entries are allowed only between 09:15 and 15:00 on market days.")
    side = str(side).upper()
    if side not in ("BUY", "SELL"):
        raise BrokerError("Side must be BUY or SELL.")
    try:
        quantity = int(quantity)
    except (TypeError, ValueError):
        raise BrokerError("Quantity must be a whole number.")
    if not 1 <= quantity <= MAX_QTY:
        raise BrokerError(f"Quantity must be between 1 and {MAX_QTY:,}.")
    stop_pct, target_pct = _pct(stop_pct, "Stop loss"), _pct(target_pct, "Target")

    why = surveillance.caution(code)
    if why and not consent:                                      # cautionary stock: ask before sending
        raise BrokerError(f"{why} Placing it needs your consent to the exchange's cautionary warning.",
                          code="consent_required")

    with LOCK:                                                   # check + record as one step (double clicks)
        done = placed_today(now.date()).get((account, code, side))
        if done:
            raise BrokerError(f"Already placed today on {name(account)}: {side} {done.quantity} {code} "
                              f"({done.status}).", code="already_placed")
        try:
            inst = angel.instrument(code)
            price = Decimal(str(angel.ltp(account, inst)))
        except angel.AngelError as exc:
            raise BrokerError(str(exc))
        value = price * quantity
        if value > Decimal(str(angel.max_order_value())):
            raise BrokerError(f"Order value ₹{value:,.0f} is above the limit ₹{angel.max_order_value():,.0f} "
                              "(ANGEL_MAX_ORDER_VALUE in .env).")
        long_ = side == "BUY"
        if guard:
            g_stop, g_target = (Decimal(str(x)) for x in guard)
            if (price <= g_stop if long_ else price >= g_stop):
                raise BrokerError(f"The price (₹{price}) has already passed the recommendation's stop (₹{g_stop}) - "
                                  "skip this one.", code="missed")
            if (price >= g_target if long_ else price <= g_target):
                raise BrokerError(f"The price (₹{price}) has already reached the recommendation's target "
                                  f"(₹{g_target}) - the move is gone.", code="missed")
        stop, target = levels(side, price, stop_pct, target_pct)            # until the fill is known
        bo = BO.objects.create(account=account, session=now.date(), rec_id=rec_id or "", code=code, side=side,
                               quantity=quantity, stop_pct=stop_pct, target_pct=target_pct, stop=stop, target=target,
                               status=BO.PLACING, scrip_consent=bool(consent), message=f"Sending at about ₹{price}")
    try:
        bo.entry_order_id = angel.place_market(account, inst, side, quantity, consent=bo.scrip_consent)
    except angel.AngelError as exc:
        # "MAY have gone through": keep it as ERROR, which blocks sending it again today
        bo.status, bo.message = (BO.ERROR if "MAY have gone through" in str(exc) else BO.REJECTED), str(exc)[:400]
        bo.save()
        if angel.CAUTION_CODE in str(exc) or "cautionary" in str(exc).lower():
            if not consent:                                      # first time: let the page ask for consent
                raise BrokerError(f"Angel One says {code.removesuffix('.NS')} is on the exchange's cautionary list. "
                                  "Placing it needs your consent to the cautionary warning.", code="consent_required")
            raise BrokerError("Angel One refused it even with your consent (scripconsent) - place this one in the "
                              "Angel One app.")
        raise BrokerError(bo.message)
    bo.save()
    _settle_entry(bo, inst, wait=True)
    return bo


def _settle_entry(bo, inst, wait):
    st, avg, filled, text = (angel.wait_for if wait else angel.order_status)(bo.account, bo.entry_order_id)
    if st == "complete":
        bo.entry_price = Decimal(str(avg)).quantize(PAISA)
        if bo.stop_pct is not None and bo.target_pct is not None:          # levels from the actual fill
            bo.stop, bo.target = levels(bo.side, bo.entry_price, bo.stop_pct, bo.target_pct)
        _protect(bo, inst)
    elif st in ("rejected", "cancelled"):
        bo.status, bo.message = BO.REJECTED, f"Angel One {st} the entry: {text or 'no reason given'}"[:400]
    else:
        bo.status, bo.message = BO.PENDING, f"Entry order is '{st}' at Angel One - the server keeps checking."
    bo.save()


def _protect(bo, inst):
    verb = "Bought" if bo.side == "BUY" else "Sold"
    try:
        bo.sl_order_id, trig = angel.place_stop(bo.account, inst, opposite(bo.side), bo.quantity, bo.stop,
                                                consent=bo.scrip_consent)
        bo.sl_trigger = trig
        bo.status = BO.OPEN
        bo.message = (f"{verb} {bo.quantity} @ ₹{bo.entry_price}. Stop-loss order at Angel One (trigger ₹{trig}, "
                      f"{bo.stop_pct}%); target ₹{bo.target} ({bo.target_pct}%) and the 15:15 exit are watched by "
                      "this server.")
    except angel.AngelError as exc:
        bo.status = BO.OPEN
        bo.message = (f"{verb} {bo.quantity} @ ₹{bo.entry_price} but the STOP-LOSS ORDER FAILED ({exc}) - protect this "
                      "position in the Angel One app.")[:400]


def _close(bo, price, reason, order_id=""):
    bo.status, bo.exit_reason = BO.CLOSED, reason
    bo.exit_price = Decimal(str(price)).quantize(Decimal("0.01")) if price else None
    bo.exit_order_id = bo.exit_order_id or order_id
    label = {"stop_loss": "Stop loss hit", "target": "Target hit", "squareoff": "15:15 exit"}.get(reason, reason)
    bo.message = f"{label} @ ₹{bo.exit_price}" + (f" · P&L ₹{bo.pnl:,.2f}" if bo.pnl is not None else "")
    bo.save()


def _exit(bo, inst, reason):
    """Cancel the stop-loss order, and only once it is confirmed cancelled, exit at market."""
    if bo.sl_order_id:
        st = angel.order_status(bo.account, bo.sl_order_id)
        if st[0] == "complete":
            return _close(bo, st[1], "stop_loss", bo.sl_order_id)
        if st[0] not in ("cancelled", "rejected"):
            try:
                angel.cancel(bo.account, bo.sl_order_id, "STOPLOSS")
            except angel.AngelError as exc:
                logger.warning("broker: cancel of stop %s failed: %s", bo.sl_order_id, exc)
            st = angel.wait_for(bo.account, bo.sl_order_id, seconds=4)
            if st[0] == "complete":
                return _close(bo, st[1], "stop_loss", bo.sl_order_id)
            if st[0] not in ("cancelled", "rejected"):
                bo.message = f"Wanted to exit ({reason}) but the stop-loss order is still '{st[0]}' - retrying."
                bo.save()
                return None
    bo.exit_order_id = angel.place_market(bo.account, inst, opposite(bo.side), bo.quantity, consent=bo.scrip_consent)
    bo.exit_reason = reason
    bo.save()
    st = angel.wait_for(bo.account, bo.exit_order_id)
    if st[0] == "complete":
        return _close(bo, st[1], reason)
    bo.message = f"Exit order ({reason}) is '{st[0]}' at Angel One - checking again shortly."
    bo.save()
    return None


def watch(now=None):
    """One watcher round over today's live Angel One trades."""
    now = now or now_ist()
    stale = BO.objects.filter(status__in=[BO.PLACING, BO.PENDING, BO.OPEN], session__lt=now.date())
    for bo in stale:                                             # server was off at the close
        bo.status = BO.CLOSED
        bo.exit_reason = bo.exit_reason or "broker_squareoff"
        bo.message = "Not watched at the close - Angel One squares off intraday positions itself; check the app."
        bo.save()
    for bo in BO.objects.filter(status__in=[BO.PENDING, BO.OPEN], session=now.date()):
        with LOCK:
            try:
                _watch_one(bo, now)
            except angel.AngelError as exc:
                logger.warning("broker: %s %s: %s", bo.account, bo.code, exc)


def _watch_one(bo, now):
    inst = angel.instrument(bo.code)
    if bo.status == BO.PENDING:
        return _settle_entry(bo, inst, wait=False)
    if bo.exit_order_id:                                         # an exit was sent earlier - see how it went
        st = angel.order_status(bo.account, bo.exit_order_id)
        if st[0] == "complete":
            return _close(bo, st[1], bo.exit_reason or "exit")
        if st[0] in ("rejected", "cancelled"):
            bo.exit_order_id, bo.message = "", f"Exit order {st[0]}: {st[3]} - will try again."[:400]
            bo.save()
        return None
    if bo.sl_order_id:
        st = angel.order_status(bo.account, bo.sl_order_id)
        if st[0] == "complete":
            return _close(bo, st[1], "stop_loss", bo.sl_order_id)
    if now.time() >= SQUAREOFF:
        return _exit(bo, inst, "squareoff")
    price = angel.ltp(bo.account, inst)
    if (price >= float(bo.target)) if bo.side == "BUY" else (price <= float(bo.target)):
        return _exit(bo, inst, "target")
    return None


def has_active(now=None):
    now = now or now_ist()
    return BO.objects.filter(status__in=[BO.PLACING, BO.PENDING, BO.OPEN], session__lte=now.date()).exists()


def serialize(bo):
    f = lambda v: None if v is None else float(v)
    return {"id": bo.pk, "account": bo.account, "code": bo.code, "side": bo.side, "quantity": bo.quantity,
            "status": bo.status, "entry_price": f(bo.entry_price), "stop": f(bo.stop), "target": f(bo.target),
            "stop_pct": f(bo.stop_pct), "target_pct": f(bo.target_pct),
            "sl_trigger": f(bo.sl_trigger), "exit_price": f(bo.exit_price), "exit_reason": bo.exit_reason,
            "pnl": f(bo.pnl), "message": bo.message, "entry_order_id": bo.entry_order_id,
            "time": bo.created_at.astimezone(IST).isoformat()}

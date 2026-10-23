"""
Option paper trading, the way a broker does it.

Orders  BUY or SELL in whole lots, NRML (carry to expiry) or MIS (intraday, squared off at 15:20).
        MARKET fills at the live best ask (buy) / best bid (sell) - not the LTP - and only while the market
        is open.  LIMIT fills when the ask / bid reaches the price; one placed outside market hours waits
        for the open (like an AMO).
Money   Buying pays the premium.  Selling (writing) credits the premium and blocks margin - an estimate
        of NSE's SPAN + exposure margin (see short_margin), released as the short is bought back.
        Every executed order pays Zerodha's option charges (charges()).
Netting one position per contract + product; an opposite order closes it first (realized P&L) and any
        rest opens the other way.
Watcher (monitor.py, every 15 s in market hours) fills limit orders, exits at stop loss / target
        (checked on the premium's LTP, exit at the bid / ask), squares off MIS at 15:20, and settles
        expired contracts at their intrinsic value against the underlying's closing price.
"""
import logging
import threading
from datetime import date, datetime, time
from decimal import ROUND_HALF_UP, Decimal
from functools import wraps

from django.db import transaction

from prediction.market_time import IST, get_market_status

from . import chain as ch
from . import greeks
from .models import CE, MIS, NRML, PE, OptionAccount, OptionOrder, OptionPosition, contract_name

logger = logging.getLogger(__name__)

D = Decimal
PAISE, TICK, CRORE = D("0.01"), D("0.05"), D("10000000")
OWNER = "options:1"
START_BALANCE = D("500000")
MAX_LOTS = 100
SQUAREOFF = time(15, 20)

# Zerodha, Equity F&O options (zerodha.com/charges, checked 23 Oct 2026)
BROKERAGE = D("20")                 # flat per executed order
STT_SELL_PCT = D("0.15")            # % of premium, sell side
STT_EXERCISE_PCT = D("0.15")        # % of intrinsic value, options bought and exercised
EXCHANGE_PCT = D("0.03553")         # NSE transaction charges, % of premium
SEBI_PER_CRORE, IPFT_PER_CRORE = D("10"), D("0.01")
STAMP_BUY_PCT = D("0.003")
GST = D("0.18")

# short-option margin estimate (NSE SPAN + exposure, simplified)
MARGIN = {"index": {"span": 0.065, "min": 0.025, "exposure": 0.02},
          "stock": {"span": 0.15, "min": 0.05, "exposure": 0.035}}

LOCK = threading.RLock()


class TradeError(Exception):
    pass


class AlreadyClosed(TradeError):
    pass


def serialized(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        with LOCK:
            return fn(*a, **kw)
    return wrapper


def now_ist():
    return datetime.now(IST)


def to_ist(dt):
    return dt.astimezone(IST) if dt.tzinfo else dt.replace(tzinfo=IST)


def market_open(now=None):
    return get_market_status(now or now_ist()) == "OPEN"


def q2(v):
    return D(v).quantize(PAISE, ROUND_HALF_UP)


def to_tick(v):
    return (D(str(v)) / TICK).quantize(D("1"), ROUND_HALF_UP) * TICK


# ---------------------------------------------------------------- account
def get_account():
    acct, created = OptionAccount.objects.get_or_create(owner_key=OWNER, defaults={"balance": START_BALANCE})
    return acct


def blocked_margin(account):
    return sum((p.blocked_margin for p in account.positions.exclude(quantity=0)), D("0"))


def available(account):
    return account.balance - blocked_margin(account)


@serialized
def add_funds(account, raw):
    try:
        amt = q2(D(str(raw)))
    except Exception:
        raise TradeError("Enter an amount in rupees.")
    if not D("1") <= amt <= D("100000000"):
        raise TradeError("Amount must be between ₹1 and ₹10 crore.")
    account.balance += amt
    account.save(update_fields=["balance"])
    return account


# ---------------------------------------------------------------- charges / margin
def charges(side, premium_value, exercise_value=0, brokerage=True):
    v, ex = D(str(premium_value)), D(str(exercise_value))
    p = {
        "brokerage": BROKERAGE if brokerage else D("0"),
        "stt": (v * STT_SELL_PCT / 100 if side == "SELL" else D("0")) + ex * STT_EXERCISE_PCT / 100,
        "exchange": v * EXCHANGE_PCT / 100,
        "sebi": v * (SEBI_PER_CRORE + IPFT_PER_CRORE) / CRORE,
        "stamp": v * STAMP_BUY_PCT / 100 if side == "BUY" else D("0"),
    }
    p["gst"] = (p["brokerage"] + p["exchange"] + p["sebi"]) * GST
    p = {k: q2(x) for k, x in p.items()}
    p["total"] = sum(p.values(), D("0"))
    return p


def short_margin(symbol, spot, strike, option_type, qty):
    """SPAN-like: max(span% x spot - half the out-of-the-money amount, min% x spot) + exposure% x spot, per unit."""
    if not spot:
        raise TradeError("No underlying price - can't work out the margin right now.")
    m = MARGIN["index" if ch.is_index(symbol) else "stock"]
    spot, strike = float(spot), float(strike)
    otm = max(0.0, strike - spot) if option_type == CE else max(0.0, spot - strike)
    per_unit = max(m["span"] * spot - 0.5 * otm, m["min"] * spot) + m["exposure"] * spot
    return q2(D(str(per_unit)) * qty)


# ---------------------------------------------------------------- execution
def _position(account, o):
    pos, _ = OptionPosition.objects.get_or_create(
        account=account, symbol=o.symbol, expiry=o.expiry, strike=o.strike, option_type=o.option_type,
        product=o.product, defaults={"lot_size": o.lot_size})
    return pos


def plan_fill(account, pos, side, qty, price, spot):
    """What this fill does to the wallet: (closing, opening, cash, margin_add, margin_release, realized, charges)."""
    sign = 1 if side == "BUY" else -1
    q0 = pos.quantity
    closing = min(qty, abs(q0)) if q0 and (q0 > 0) != (sign > 0) else 0
    opening = qty - closing
    price = D(str(price))
    cash = realized = margin_add = margin_release = D("0")
    if closing:
        realized = (price - pos.avg_price) * closing * (1 if q0 > 0 else -1)
        cash += price * closing if q0 > 0 else -price * closing
        if q0 < 0:
            margin_release = pos.blocked_margin * closing / abs(q0)
    if opening:
        if sign > 0:
            cash -= price * opening
        else:
            cash += price * opening
            margin_add = short_margin(pos.symbol, spot, pos.strike, pos.option_type, opening)
    chg = charges(side, price * qty)
    return closing, opening, cash, margin_add, margin_release, realized, chg


def _check_funds(account, cash, margin_add, chg):
    need = margin_add + chg["total"] + max(D("0"), -cash)
    have = available(account)
    if need > have:
        raise TradeError(f"Insufficient funds: this order needs ₹{need:,.2f} "
                         f"({'margin ₹%s, ' % f'{margin_add:,.2f}' if margin_add else ''}charges ₹{chg['total']:,.2f}) "
                         f"but ₹{have:,.2f} is available.")


def _apply(account, o, price, spot, when, reason=""):
    """Fill order `o` at `price`: position, wallet, margin, charges. Caller holds LOCK, inside a transaction."""
    pos = _position(account, o)
    closing, opening, cash, m_add, m_rel, realized, chg = plan_fill(account, pos, o.side, o.quantity, price, spot)
    if opening:                                             # closing an existing position is never refused
        _check_funds(account, cash, m_add, chg)
    price = D(str(price))
    sign = 1 if o.side == "BUY" else -1
    q0 = pos.quantity
    if closing:
        q0 += sign * closing
        pos.blocked_margin -= m_rel
        pos.realized_pnl += realized
        if q0 == 0:
            pos.avg_price, pos.blocked_margin, pos.stop_loss, pos.target = D("0"), D("0"), None, None
    if opening:
        if q0 == 0:
            pos.avg_price, pos.opened_at, pos.stop_loss, pos.target = price, when, None, None
        else:                                               # adding to the same side
            pos.avg_price = (pos.avg_price * abs(q0) + price * opening) / (abs(q0) + opening)
        q0 += sign * opening
        pos.blocked_margin += m_add
    pos.quantity = q0
    pos.lot_size = o.lot_size
    if q0 and (o.stop_loss is not None or o.target is not None):
        pos.stop_loss = o.stop_loss if o.stop_loss is not None else pos.stop_loss
        pos.target = o.target if o.target is not None else pos.target
    pos.save()
    account.balance += cash - chg["total"]
    account.save(update_fields=["balance"])
    o.status, o.price, o.filled_at = OptionOrder.COMPLETE, price, when
    o.underlying_price = q2(spot) if spot else None
    o.realized_pnl, o.charges = q2(realized), chg["total"]
    o.charges_detail = {k: float(v) for k, v in chg.items()}
    o.balance_after = account.balance
    o.reason = o.reason or reason
    o.save()
    return o, pos


def fill_price(side, q):
    """Market order fill: best ask to buy, best bid to sell (LTP only when that side of the book is empty)."""
    if not q:
        return None
    px = q["ask"] if side == "BUY" else q["bid"]
    return px if px and px > 0 else (q["ltp"] or None)


def _validate_contract(symbol, expiry, strike, option_type, now):
    if option_type not in (CE, PE):
        raise TradeError("Option type must be CE or PE.")
    if ch.expiry_dt(expiry) <= now:
        raise TradeError(f"The {expiry:%d %b %Y} contracts have expired.")
    return ch.lot_size(symbol, expiry)


def _parse_level(raw, label):
    if raw in (None, ""):
        return None
    try:
        v = to_tick(raw)
    except Exception:
        raise TradeError(f"{label} must be a price.")
    if v <= 0:
        raise TradeError(f"{label} must be above zero.")
    return v


@serialized
def place_order(account, *, symbol, expiry, strike, option_type, side, lots, product=NRML, order_type="MARKET",
                limit_price=None, stop_loss=None, target=None, now=None):
    now = now or now_ist()
    side, product, order_type = str(side).upper(), str(product).upper(), str(order_type).upper()
    if side not in ("BUY", "SELL"):
        raise TradeError("Side must be BUY or SELL.")
    if product not in (NRML, MIS):
        raise TradeError("Product must be NRML or MIS.")
    if order_type not in (OptionOrder.MARKET, OptionOrder.LIMIT):
        raise TradeError("Order type must be MARKET or LIMIT.")
    try:
        lots = int(lots)
    except (TypeError, ValueError):
        raise TradeError("Lots must be a whole number.")
    if not 1 <= lots <= MAX_LOTS:
        raise TradeError(f"Lots must be between 1 and {MAX_LOTS}.")
    try:
        lot = _validate_contract(symbol, expiry, strike, option_type, now)
    except ch.ChainError as exc:
        raise TradeError(str(exc))
    if product == MIS and now.time() >= SQUAREOFF and market_open(now):
        raise TradeError("MIS orders are closed after 15:20 - use NRML.")
    sl, tgt = _parse_level(stop_loss, "Stop loss"), _parse_level(target, "Target")
    o = OptionOrder(account=account, symbol=symbol, expiry=expiry, strike=D(str(strike)), option_type=option_type,
                    product=product, side=side, quantity=lots * lot, lot_size=lot, order_type=order_type,
                    stop_loss=sl, target=tgt)
    is_open = market_open(now)
    if order_type == OptionOrder.MARKET and not is_open:
        raise TradeError("The market is closed - market orders can't be placed. Place a LIMIT order instead; it "
                         "waits for the market to open (like an AMO).")
    try:
        q, spot, _ = ch.quote(symbol, expiry, strike, option_type)
    except ch.ChainError as exc:
        raise TradeError(f"No live option prices: {exc}")
    if q is None:
        raise TradeError(f"{o.name} is not listed on NSE.")
    if order_type == OptionOrder.LIMIT:
        o.limit_price = _parse_level(limit_price, "Limit price")
        if o.limit_price is None:
            raise TradeError("Enter a limit price.")
    _check_levels_side(side, sl, tgt, o.limit_price or D(str(fill_price(side, q) or q["ltp"] or 0)))

    pos = OptionPosition.objects.filter(account=account, symbol=symbol, expiry=expiry, strike=o.strike,
                                        option_type=option_type, product=product).first()
    est = o.limit_price or D(str(fill_price(side, q) or 0))
    if est > 0:                                             # reject now what could never fill
        closing, opening, cash, m_add, _, _, chg = plan_fill(account, pos or _virtual(o), side, o.quantity, est, spot)
        if opening:
            _check_funds(account, cash, m_add, chg)

    with transaction.atomic():
        o.save()
        if order_type == OptionOrder.MARKET:
            px = fill_price(side, q)
            if not px:
                o.status, o.message = OptionOrder.REJECTED, "No buyers / sellers for this contract right now."
                o.save()
                raise TradeError(o.message)
            return _apply(account, o, px, spot, now)[0]
        if is_open:                                         # marketable limit: fills now at the bid / ask
            px = _marketable(side, q, o.limit_price)
            if px:
                return _apply(account, o, px, spot, now, reason="limit")[0]
        o.message = "Waiting for the market to open" if not is_open else "Open - waiting for the price"
        o.save()
        return o


def _virtual(o):
    return OptionPosition(symbol=o.symbol, expiry=o.expiry, strike=o.strike, option_type=o.option_type,
                          product=o.product, lot_size=o.lot_size, quantity=0, avg_price=D("0"), blocked_margin=D("0"))


def _marketable(side, q, limit):
    if not q:
        return None
    if side == "BUY" and q["ask"] and 0 < q["ask"] <= float(limit):
        return q["ask"]
    if side == "SELL" and q["bid"] and q["bid"] >= float(limit):
        return q["bid"]
    return None


def _check_levels_side(side, sl, tgt, ref):
    """Levels are on the premium: for a BUY the stop is below and the target above the price; reversed for a SELL."""
    if not ref:
        return
    long_ = side == "BUY"
    if sl is not None and (sl >= ref if long_ else sl <= ref):
        raise TradeError(f"Stop loss must be {'below' if long_ else 'above'} the premium (₹{ref:.2f}) for a "
                         f"{'buy' if long_ else 'sell'}.")
    if tgt is not None and (tgt <= ref if long_ else tgt >= ref):
        raise TradeError(f"Target must be {'above' if long_ else 'below'} the premium (₹{ref:.2f}) for a "
                         f"{'buy' if long_ else 'sell'}.")


@serialized
def cancel_order(account, order_id):
    o = account.orders.filter(pk=order_id).first()
    if o is None:
        raise TradeError("Unknown order.")
    if o.status != OptionOrder.PENDING:
        raise TradeError(f"Order is already {o.status.lower()}.")
    o.status, o.message = OptionOrder.CANCELLED, "Cancelled by you"
    o.save(update_fields=["status", "message"])
    return o


@serialized
def exit_position(account, position_id, reason="", now=None, quote=None):
    """Close the whole position at market (bid for a long, ask for a short)."""
    now = now or now_ist()
    pos = account.positions.filter(pk=position_id).first()
    if pos is None:
        raise TradeError("Unknown position.")
    if pos.quantity == 0:
        raise AlreadyClosed("This position is already closed.")
    if not market_open(now):
        raise TradeError("The market is closed - positions can be exited from 09:15.")
    side = "SELL" if pos.quantity > 0 else "BUY"
    if quote is None:
        try:
            quote, spot, _ = ch.quote(pos.symbol, pos.expiry, pos.strike, pos.option_type)
        except ch.ChainError as exc:
            raise TradeError(f"No live option prices: {exc}")
    else:
        quote, spot = quote
    px = fill_price(side, quote)
    if not px:
        raise TradeError("No buyers / sellers for this contract right now.")
    o = OptionOrder(account=account, symbol=pos.symbol, expiry=pos.expiry, strike=pos.strike,
                    option_type=pos.option_type, product=pos.product, side=side, quantity=abs(pos.quantity),
                    lot_size=pos.lot_size, reason=reason)
    with transaction.atomic():
        o.save()
        return _apply(account, o, px, spot, now, reason=reason)[0]


@serialized
def set_levels(account, position_id, stop_loss=None, target=None, now=None):
    pos = account.positions.filter(pk=position_id).exclude(quantity=0).first()
    if pos is None:
        raise TradeError("No open position.")
    sl, tgt = _parse_level(stop_loss, "Stop loss"), _parse_level(target, "Target")
    ref = None
    try:
        q, _, _ = ch.quote(pos.symbol, pos.expiry, pos.strike, pos.option_type)
        ref = D(str(q["ltp"])) if q and q["ltp"] else None
    except ch.ChainError:
        pass
    _check_levels_side("BUY" if pos.quantity > 0 else "SELL", sl, tgt, ref or pos.avg_price)
    pos.stop_loss, pos.target = sl, tgt
    pos.save(update_fields=["stop_loss", "target", "updated_at"])
    return pos


# ---------------------------------------------------------------- watcher jobs
def _quote_or_none(p):
    try:
        q, spot, _ = ch.quote(p.symbol, p.expiry, p.strike, p.option_type, max_age=10)
        return q, spot
    except ch.ChainError as exc:
        logger.warning("options: no quote for %s: %s", contract_name(p.symbol, p.expiry, p.strike, p.option_type), exc)
        return None


def process_pending(account, now=None):
    """Fill limit orders whose price was reached; cancel ones whose contract expired."""
    now = now or now_ist()
    done = []
    for o in account.orders.filter(status=OptionOrder.PENDING).order_by("created_at"):
        if ch.expiry_dt(o.expiry) <= now:
            o.status, o.message = OptionOrder.CANCELLED, "Contract expired"
            o.save(update_fields=["status", "message"])
            continue
        if o.product == MIS and now.time() >= SQUAREOFF:
            o.status, o.message = OptionOrder.CANCELLED, "MIS order cancelled at 15:20"
            o.save(update_fields=["status", "message"])
            continue
        if not market_open(now):
            continue
        qs = _quote_or_none(o)
        if not qs:
            continue
        px = _marketable(o.side, qs[0], o.limit_price)
        if not px:
            continue
        with LOCK, transaction.atomic():
            o.refresh_from_db()
            if o.status != OptionOrder.PENDING:
                continue
            try:
                done.append(_apply(account, o, px, qs[1], now, reason="limit")[0])
            except TradeError as exc:
                o.status, o.message = OptionOrder.REJECTED, str(exc)[:300]
                o.save(update_fields=["status", "message"])
    return done


def level_reason(pos, ltp):
    if not ltp:
        return None
    ltp, long_ = D(str(ltp)), pos.quantity > 0
    if pos.stop_loss is not None and (ltp <= pos.stop_loss if long_ else ltp >= pos.stop_loss):
        return "stop_loss"
    if pos.target is not None and (ltp >= pos.target if long_ else ltp <= pos.target):
        return "target"
    return None


def check_levels(account, now=None):
    now = now or now_ist()
    exits = []
    if not market_open(now):
        return exits
    from django.db.models import Q
    for pos in account.positions.exclude(quantity=0).filter(Q(stop_loss__isnull=False) | Q(target__isnull=False)):
        qs = _quote_or_none(pos)
        reason = level_reason(pos, qs[0]["ltp"]) if qs and qs[0] else None
        if not reason:
            continue
        try:
            exits.append(exit_position(account, pos.pk, reason=reason, now=now, quote=qs))
        except TradeError as exc:
            logger.warning("options: %s exit failed: %s", pos.name, exc)
    return exits


def squareoff_mis(account, now=None):
    now = now or now_ist()
    out = []
    if not market_open(now) or now.time() < SQUAREOFF:
        return out
    for pos in account.positions.filter(product=MIS).exclude(quantity=0):
        try:
            out.append(exit_position(account, pos.pk, reason="squareoff", now=now))
        except TradeError as exc:
            logger.warning("options: MIS square-off of %s failed: %s", pos.name, exc)
    return out


@serialized
def settle_expired(account, now=None, price_of=None):
    """Settle open positions whose contracts expired, at intrinsic value against the underlying's close."""
    now = now or now_ist()
    price_of = price_of or ch.settlement_price
    out = []
    for pos in account.positions.exclude(quantity=0):
        if ch.expiry_dt(pos.expiry) > now:
            continue
        try:
            spot = price_of(pos.symbol, pos.expiry)
        except Exception as exc:
            logger.warning("options: settlement price for %s unavailable: %s", pos.symbol, exc)
            spot = None
        if not spot:
            continue                                        # try again next round
        k = float(pos.strike)
        intrinsic = q2(D(str(max(0.0, spot - k) if pos.option_type == CE else max(0.0, k - spot))))
        long_, qty = pos.quantity > 0, abs(pos.quantity)
        value = intrinsic * qty
        if intrinsic > 0:                                   # exercised (long) / assigned (short)
            chg = charges("SELL" if long_ else "BUY", 0, exercise_value=value if long_ else 0)
        else:
            chg = {k_: D("0") for k_ in ("brokerage", "stt", "exchange", "sebi", "stamp", "gst", "total")}
        realized = (intrinsic - pos.avg_price) * qty * (1 if long_ else -1)
        cash = value if long_ else -value
        with transaction.atomic():
            o = OptionOrder.objects.create(
                account=account, symbol=pos.symbol, expiry=pos.expiry, strike=pos.strike, option_type=pos.option_type,
                product=pos.product, side="SELL" if long_ else "BUY", quantity=qty, lot_size=pos.lot_size,
                status=OptionOrder.COMPLETE, price=intrinsic, underlying_price=q2(D(str(spot))),
                realized_pnl=q2(realized), charges=chg["total"], charges_detail={k_: float(v) for k_, v in chg.items()},
                reason="expiry", filled_at=ch.expiry_dt(pos.expiry),
                message=(f"{'Exercised' if long_ else 'Assigned'} at ₹{intrinsic} (underlying closed ₹{spot:,.2f})"
                         if intrinsic > 0 else f"Expired worthless (underlying closed ₹{spot:,.2f})"))
            account.balance += cash - chg["total"]
            account.save(update_fields=["balance"])
            o.balance_after = account.balance
            o.save(update_fields=["balance_after"])
            pos.realized_pnl += realized
            pos.quantity, pos.avg_price, pos.blocked_margin, pos.stop_loss, pos.target = 0, D("0"), D("0"), None, None
            pos.save()
        out.append(o)
    return out


def run_once(now=None):
    """One watcher round (called by the paper-trading monitor loop)."""
    now = now or now_ist()
    account = get_account()
    return {"settled": settle_expired(account, now), "limits": process_pending(account, now),
            "levels": check_levels(account, now), "squareoff": squareoff_mis(account, now)}


# ---------------------------------------------------------------- page data
def _f(v):
    return None if v is None else float(v)


def serialize_order(o):
    return {"id": o.pk, "name": o.name, "symbol": o.symbol, "expiry": o.expiry.isoformat(), "strike": float(o.strike),
            "option_type": o.option_type, "product": o.product, "side": o.side, "quantity": o.quantity,
            "lots": o.quantity // o.lot_size if o.lot_size else None, "lot_size": o.lot_size,
            "order_type": o.order_type, "limit_price": _f(o.limit_price), "status": o.status, "price": _f(o.price),
            "underlying_price": _f(o.underlying_price), "realized_pnl": _f(o.realized_pnl), "charges": _f(o.charges),
            "charges_detail": o.charges_detail, "reason": o.reason, "message": o.message,
            "stop_loss": _f(o.stop_loss), "target": _f(o.target),
            "created_at": to_ist(o.created_at).isoformat(), "filled_at": to_ist(o.filled_at).isoformat() if o.filled_at else None}


def snapshot(account, now=None):
    now = now or now_ist()
    try:
        settle_expired(account, now)
    except Exception:
        logger.exception("options: settlement failed")
    day_start = datetime.combine(now.date(), time(0), tzinfo=IST)
    positions = []
    chains = {}
    for p in account.positions.all().order_by("expiry", "symbol", "strike"):
        touched_today = p.updated_at and to_ist(p.updated_at) >= day_start
        if p.quantity == 0 and not touched_today:
            continue
        key = (p.symbol, p.expiry)
        if key not in chains:
            try:
                chains[key] = ch.chain(p.symbol, p.expiry, max_age=5) if ch.expiry_dt(p.expiry) > now else None
            except ch.ChainError:
                chains[key] = None
        c = chains[key]
        q, spot = None, (c or {}).get("spot")
        if c:
            q = next((r.get(p.option_type) for r in c["rows"] if abs(r["strike"] - float(p.strike)) < 1e-6), None)
        ltp = q["ltp"] if q else None
        t = greeks.years_to(ch.expiry_dt(p.expiry), now)
        g = greeks.greeks(spot or 0, float(p.strike), t, (q or {}).get("iv"), p.option_type)
        qty = p.quantity
        mtm = (ltp - float(p.avg_price)) * qty if ltp is not None and qty else 0.0
        positions.append({
            "id": p.pk, "name": p.name, "symbol": p.symbol, "expiry": p.expiry.isoformat(), "strike": float(p.strike),
            "option_type": p.option_type, "product": p.product, "quantity": qty, "lot_size": p.lot_size,
            "lots": qty / p.lot_size if p.lot_size else None, "avg_price": float(p.avg_price), "ltp": ltp,
            "bid": (q or {}).get("bid"), "ask": (q or {}).get("ask"), "iv": (q or {}).get("iv"), "spot": spot,
            "chg_pct": (q or {}).get("pchg"), "mtm": mtm, "realized_pnl": float(p.realized_pnl),
            "blocked_margin": float(p.blocked_margin), "stop_loss": _f(p.stop_loss), "target": _f(p.target),
            "days_to_expiry": round(t * 365, 2), "greeks": g,
            "position_delta": g["delta"] * qty if g["delta"] is not None else None,
            "position_theta": g["theta"] * qty if g["theta"] is not None else None,
            "stale": bool(c and c.get("stale")),
        })
    today = account.orders.filter(created_at__gte=day_start)
    filled_today = today.filter(status=OptionOrder.COMPLETE)
    pending = account.orders.filter(status=OptionOrder.PENDING)
    blocked = blocked_margin(account)
    unrealized = sum(p["mtm"] for p in positions if p["quantity"])
    history = account.orders.filter(status=OptionOrder.COMPLETE).order_by("-filled_at", "-id")[:300]
    return {
        "funds": {"balance": float(account.balance), "blocked_margin": float(blocked),
                  "available": float(account.balance - blocked), "unrealized": unrealized,
                  "realized_today": float(sum((o.realized_pnl for o in filled_today), D("0"))),
                  "charges_today": float(sum((o.charges for o in filled_today), D("0"))),
                  "start_balance": float(START_BALANCE)},
        "positions": positions,
        "pending": [serialize_order(o) for o in pending],
        "orders_today": [serialize_order(o) for o in today[:200]],
        "history": [serialize_order(o) for o in history],
        "market": {"status": get_market_status(now), "open": market_open(now), "now": now.isoformat(),
                   "squareoff": SQUAREOFF.strftime("%H:%M")},
    }


def preview(account, *, symbol, expiry, strike, option_type, side, lots, product=NRML, price=None):
    """Margin / premium / charges for the order ticket, before placing."""
    lot = ch.lot_size(symbol, expiry)
    qty = int(lots) * lot
    q, spot, _ = ch.quote(symbol, expiry, strike, option_type)
    px = D(str(price)) if price else D(str(fill_price(side, q) or 0))
    pos = OptionPosition.objects.filter(account=account, symbol=symbol, expiry=expiry, strike=D(str(strike)),
                                        option_type=option_type, product=product).first()
    v = _virtual(OptionOrder(symbol=symbol, expiry=expiry, strike=D(str(strike)), option_type=option_type,
                             product=product, lot_size=lot))
    closing, opening, cash, m_add, m_rel, realized, chg = plan_fill(account, pos or v, side, qty, px, spot)
    need = m_add + chg["total"] + max(D("0"), -cash)
    return {"quantity": qty, "lot_size": lot, "price": float(px), "premium": float(px * qty), "spot": spot,
            "margin": float(m_add), "charges": {k: float(x) for k, x in chg.items()}, "closing": closing,
            "opening": opening, "required": float(need), "available": float(available(account)),
            "ok": need <= available(account) or not opening}

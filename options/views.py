"""
Option Trading page and API.  Every API call takes mode=live (default) or mode=replay (query string for GET,
body for POST): replay uses the separate replay wallet and prices on its replay clock.

  GET  /options/                                   the page (?symbol=NIFTY&mode=replay)
  GET  /api/options/underlyings/                   F&O underlyings with lot sizes (replay: on the replay date)
  GET  /api/options/expiries/<symbol>/             open expiries
  GET  /api/options/chain/<symbol>/?expiry=YYYY-MM-DD   chain + Greeks, ATM, PCR, max pain
  GET  /api/options/state/                         funds, positions (MTM + Greeks), orders, history, replay clock
  POST /api/options/preview/                       premium / margin / charges for the order ticket
  POST /api/options/order/                         place {symbol, expiry, strike, option_type, side, lots, product,
                                                   order_type, limit_price?, stop_loss?, target?, client_order_id?}
  POST /api/options/order/<id>/cancel/
  POST /api/options/position/<id>/exit/
  POST /api/options/position/<id>/levels/          {stop_loss, target}  (on the premium; empty clears)
  POST /api/options/funds/                         {amount}
  POST /api/options/replay/start/                  {date}  start replaying a past trading day at 09:15
  POST /api/options/replay/advance/                {to: 5 | 15 | 60 | close | next_day}
  POST /api/options/replay/reset/                  empty the replay account (wallet back to the start)
  GET  /api/options/replay/series/?symbol&expiry&strike&option_type
                                                   replay chart up to the clock: underlying candles, the contract's
                                                   premium every 5 minutes, and your fills in it that day
"""
import json
import logging
from datetime import date

from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from . import chain as ch
from . import history as hist
from . import services as svc

logger = logging.getLogger(__name__)


def _err(msg, status=400, **extra):
    return JsonResponse({"error": msg, **extra}, status=status)


def _body(request):
    try:
        return json.loads(request.body or b"{}")
    except ValueError:
        return {}


def _date(raw):
    try:
        return date.fromisoformat(str(raw))
    except (TypeError, ValueError):
        raise svc.TradeError("Dates must be YYYY-MM-DD.")


def _symbol(raw):
    s = str(raw or "").strip().upper()
    if not s or len(s) > 20:
        raise svc.TradeError("Choose an underlying.")
    return s


def _ctx(request, body=None):
    """(account, market) for mode=live / mode=replay."""
    mode = (body or {}).get("mode") or request.GET.get("mode")
    acct = svc.get_account(replay=mode == "replay")
    return acct, svc.market_for(acct)


def _replay_day(acct):
    if not acct.replay_date:
        raise svc.TradeError("Pick a date to replay first.")
    return acct.replay_date


@require_GET
@ensure_csrf_cookie
def page(request):
    return render(request, "options/options.html", {"initial": {
        "symbol": _symbol(request.GET.get("symbol") or "NIFTY"),
        "mode": "replay" if request.GET.get("mode") == "replay" else "live",
        "first_day": hist.FIRST_DAY.isoformat()}})


@require_GET
def underlyings(request):
    acct, mkt = _ctx(request)
    try:
        if mkt.replay:
            return JsonResponse({"underlyings": hist.underlyings(_replay_day(acct))})
        return JsonResponse({"underlyings": ch.underlyings()})
    except svc.TradeError as exc:
        return _err(str(exc))
    except svc.DATA_ERRORS as exc:
        return _err(str(exc), 503)


@require_GET
def expiries(request, symbol):
    acct, mkt = _ctx(request)
    try:
        sym = _symbol(symbol)
        if mkt.replay:
            day = _replay_day(acct)
            exps = hist.expiries(sym, day)
            return JsonResponse({"symbol": sym, "expiries": [e.isoformat() for e in exps],
                                 "lot_size": hist.lot_size(sym, exps[0], day) if exps else None})
        exps = ch.expiries(sym)
        return JsonResponse({"symbol": sym, "expiries": [e.isoformat() for e in exps],
                             "lot_size": ch.lot_size(sym, exps[0]) if exps else None})
    except svc.TradeError as exc:
        return _err(str(exc))
    except svc.DATA_ERRORS as exc:
        return _err(str(exc), 503)


@require_GET
def chain_view(request, symbol):
    acct, mkt = _ctx(request)
    try:
        sym, exp = _symbol(symbol), _date(request.GET.get("expiry"))
        now = mkt.now()
        data = ch.summary(mkt.chain(sym, exp, now=now), now=now)
        data["lot_size"] = mkt.lot_size(sym, exp, now)
        data["market"] = {"status": mkt.status(now), "open": mkt.is_open(now)}
        return JsonResponse(data)
    except svc.TradeError as exc:
        return _err(str(exc))
    except svc.DATA_ERRORS as exc:
        return _err(str(exc), 503)


@require_GET
def state(request):
    acct, mkt = _ctx(request)
    return JsonResponse(svc.snapshot(acct, mkt=mkt))


def _order_args(b):
    return {"symbol": _symbol(b.get("symbol")), "expiry": _date(b.get("expiry")), "strike": b.get("strike"),
            "option_type": str(b.get("option_type", "")).upper(), "side": b.get("side"), "lots": b.get("lots"),
            "product": b.get("product") or svc.NRML}


@require_POST
def preview(request):
    b = _body(request)
    acct, mkt = _ctx(request, b)
    try:
        return JsonResponse(svc.preview(acct, **_order_args(b), price=b.get("price") or None, mkt=mkt))
    except svc.TradeError as exc:
        return _err(str(exc))
    except svc.DATA_ERRORS as exc:
        return _err(str(exc))
    except (TypeError, ValueError, ArithmeticError):
        return _err("Check the order details.")


@require_POST
def place(request):
    b = _body(request)
    acct, mkt = _ctx(request, b)
    coid = str(b.get("client_order_id") or "")[:64]
    if coid and not cache.add(f"opt:coid:{coid}", 1, 600):          # the same click sent twice
        return _err("This order was already sent.", 409, code="duplicate")
    try:
        args = _order_args(b)
        if args["strike"] in (None, ""):
            raise svc.TradeError("Choose a strike.")
        o = svc.place_order(acct, **args, order_type=b.get("order_type") or "MARKET", limit_price=b.get("limit_price"),
                            stop_loss=b.get("stop_loss"), target=b.get("target"), mkt=mkt)
    except svc.TradeError as exc:
        if coid:
            cache.delete(f"opt:coid:{coid}")                       # a refused order may be sent again
        return _err(str(exc))
    except (TypeError, ValueError, ArithmeticError):
        return _err("Check the order details.")
    return JsonResponse({"ok": True, "order": svc.serialize_order(o)})


@require_POST
def cancel(request, order_id):
    acct, _ = _ctx(request, _body(request))
    try:
        o = svc.cancel_order(acct, order_id)
    except svc.TradeError as exc:
        return _err(str(exc))
    return JsonResponse({"ok": True, "order": svc.serialize_order(o)})


@require_POST
def exit_position(request, position_id):
    acct, mkt = _ctx(request, _body(request))
    try:
        o = svc.exit_position(acct, position_id, mkt=mkt)
    except svc.AlreadyClosed as exc:
        return _err(str(exc), 409, code="already_closed")
    except svc.TradeError as exc:
        return _err(str(exc))
    return JsonResponse({"ok": True, "order": svc.serialize_order(o)})


@require_POST
def levels(request, position_id):
    b = _body(request)
    acct, mkt = _ctx(request, b)
    try:
        p = svc.set_levels(acct, position_id, b.get("stop_loss"), b.get("target"), mkt=mkt)
    except svc.TradeError as exc:
        return _err(str(exc))
    return JsonResponse({"ok": True, "stop_loss": svc._f(p.stop_loss), "target": svc._f(p.target)})


@require_POST
def funds(request):
    b = _body(request)
    acct, _ = _ctx(request, b)
    try:
        a = svc.add_funds(acct, b.get("amount"))
    except svc.TradeError as exc:
        return _err(str(exc))
    return JsonResponse({"ok": True, "balance": float(a.balance)})


# ---------------------------------------------------------------- replay clock
def _events(ev):
    if not ev:
        return {}
    return {k: ([svc.serialize_order(o) for o in v] if isinstance(v, list) else v) for k, v in ev.items()}


@require_POST
def replay_start(request):
    b = _body(request)
    acct = svc.get_account(replay=True)
    try:
        svc.replay_start(acct, _date(b.get("date")))
    except svc.TradeError as exc:
        return _err(str(exc))
    return JsonResponse({"ok": True, "date": acct.replay_date.isoformat(), "time": acct.replay_time.strftime("%H:%M")})


@require_POST
def replay_advance(request):
    b = _body(request)
    acct = svc.get_account(replay=True)
    try:
        ev = svc.replay_advance(acct, str(b.get("to", "")))
    except svc.TradeError as exc:
        return _err(str(exc))
    except svc.DATA_ERRORS as exc:
        return _err(str(exc), 503)
    return JsonResponse({"ok": True, "date": acct.replay_date.isoformat(), "time": acct.replay_time.strftime("%H:%M"),
                         "events": _events(ev)})


@require_GET
def replay_series(request):
    acct = svc.get_account(replay=True)
    try:
        now = svc.ReplayMarket(acct).now()
        sym, exp = _symbol(request.GET.get("symbol")), _date(request.GET.get("expiry"))
        strike, typ = float(request.GET.get("strike")), str(request.GET.get("option_type", "")).upper()
        data = hist.series(sym, exp, strike, typ, now)
    except (TypeError, ValueError):
        return _err("Choose a contract.")
    except svc.TradeError as exc:
        return _err(str(exc))
    except svc.DATA_ERRORS as exc:
        return _err(str(exc), 503)
    day = now.date()
    fills = acct.orders.filter(status=svc.OptionOrder.COMPLETE, symbol=sym, expiry=exp, strike=svc.D(str(strike)),
                              option_type=typ)
    data["trades"] = [{"x": svc.to_ist(o.filled_at).isoformat(), "side": o.side, "price": float(o.price),
                       "quantity": o.quantity, "reason": o.reason}
                      for o in fills if o.filled_at and svc.to_ist(o.filled_at).date() == day]
    data["clock"] = now.isoformat()
    return JsonResponse(data)


@require_POST
def replay_reset(request):
    svc.replay_reset(svc.get_account(replay=True))
    return JsonResponse({"ok": True})

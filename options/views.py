"""
Option Trading page and API.

  GET  /options/                                   the page (?symbol=NIFTY)
  GET  /api/options/underlyings/                   F&O underlyings with lot sizes
  GET  /api/options/expiries/<symbol>/             open expiries
  GET  /api/options/chain/<symbol>/?expiry=YYYY-MM-DD   live chain + Greeks, ATM, PCR, max pain
  GET  /api/options/state/                         funds, positions (live MTM + Greeks), orders, history
  POST /api/options/preview/                       premium / margin / charges for the order ticket
  POST /api/options/order/                         place {symbol, expiry, strike, option_type, side, lots, product,
                                                   order_type, limit_price?, stop_loss?, target?, client_order_id?}
  POST /api/options/order/<id>/cancel/
  POST /api/options/position/<id>/exit/
  POST /api/options/position/<id>/levels/          {stop_loss, target}  (on the premium; empty clears)
  POST /api/options/funds/                         {amount}
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
        raise svc.TradeError("Expiry must be a date (YYYY-MM-DD).")


def _symbol(raw):
    s = str(raw or "").strip().upper()
    if not s or len(s) > 20:
        raise svc.TradeError("Choose an underlying.")
    return s


@require_GET
@ensure_csrf_cookie
def page(request):
    return render(request, "options/options.html", {"initial": {"symbol": _symbol(request.GET.get("symbol") or "NIFTY")}})


@require_GET
def underlyings(request):
    try:
        return JsonResponse({"underlyings": ch.underlyings()})
    except ch.ChainError as exc:
        return _err(str(exc), 503)


@require_GET
def expiries(request, symbol):
    try:
        sym = _symbol(symbol)
        return JsonResponse({"symbol": sym, "expiries": [e.isoformat() for e in ch.expiries(sym)],
                             "lot_size": ch.lot_size(sym, (ch.expiries(sym) or [date.today()])[0])})
    except (ch.ChainError, svc.TradeError) as exc:
        return _err(str(exc), 503)


@require_GET
def chain_view(request, symbol):
    try:
        sym, exp = _symbol(symbol), _date(request.GET.get("expiry"))
        data = ch.summary(ch.chain(sym, exp))
        data["lot_size"] = ch.lot_size(sym, exp)
        data["market"] = {"status": svc.get_market_status(), "open": svc.market_open()}
        return JsonResponse(data)
    except svc.TradeError as exc:
        return _err(str(exc))
    except ch.ChainError as exc:
        return _err(str(exc), 503)


@require_GET
def state(request):
    return JsonResponse(svc.snapshot(svc.get_account()))


def _order_args(b):
    return {"symbol": _symbol(b.get("symbol")), "expiry": _date(b.get("expiry")), "strike": b.get("strike"),
            "option_type": str(b.get("option_type", "")).upper(), "side": b.get("side"), "lots": b.get("lots"),
            "product": b.get("product") or svc.NRML}


@require_POST
def preview(request):
    b = _body(request)
    try:
        args = _order_args(b)
        return JsonResponse(svc.preview(svc.get_account(), **args, price=b.get("price") or None))
    except (svc.TradeError, ch.ChainError) as exc:
        return _err(str(exc))
    except (TypeError, ValueError, ArithmeticError):
        return _err("Check the order details.")


@require_POST
def place(request):
    b = _body(request)
    coid = str(b.get("client_order_id") or "")[:64]
    if coid and not cache.add(f"opt:coid:{coid}", 1, 600):          # the same click sent twice
        return _err("This order was already sent.", 409, code="duplicate")
    try:
        args = _order_args(b)
        if args["strike"] in (None, ""):
            raise svc.TradeError("Choose a strike.")
        o = svc.place_order(svc.get_account(), **args, order_type=b.get("order_type") or "MARKET",
                            limit_price=b.get("limit_price"), stop_loss=b.get("stop_loss"), target=b.get("target"))
    except svc.TradeError as exc:
        if coid:
            cache.delete(f"opt:coid:{coid}")                       # a refused order may be sent again
        return _err(str(exc))
    except (TypeError, ValueError, ArithmeticError):
        return _err("Check the order details.")
    return JsonResponse({"ok": True, "order": svc.serialize_order(o)})


@require_POST
def cancel(request, order_id):
    try:
        o = svc.cancel_order(svc.get_account(), order_id)
    except svc.TradeError as exc:
        return _err(str(exc))
    return JsonResponse({"ok": True, "order": svc.serialize_order(o)})


@require_POST
def exit_position(request, position_id):
    try:
        o = svc.exit_position(svc.get_account(), position_id)
    except svc.AlreadyClosed as exc:
        return _err(str(exc), 409, code="already_closed")
    except svc.TradeError as exc:
        return _err(str(exc))
    return JsonResponse({"ok": True, "order": svc.serialize_order(o)})


@require_POST
def levels(request, position_id):
    b = _body(request)
    try:
        p = svc.set_levels(svc.get_account(), position_id, b.get("stop_loss"), b.get("target"))
    except svc.TradeError as exc:
        return _err(str(exc))
    return JsonResponse({"ok": True, "stop_loss": svc._f(p.stop_loss), "target": svc._f(p.target)})


@require_POST
def funds(request):
    try:
        a = svc.add_funds(svc.get_account(), _body(request).get("amount"))
    except svc.TradeError as exc:
        return _err(str(exc))
    return JsonResponse({"ok": True, "balance": float(a.balance)})

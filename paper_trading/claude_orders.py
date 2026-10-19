"""
Claude recommended orders: the BUY / SELL picks of the Intraday Signal Board, saved by Claude in
claude_orders.json (next to this file), and placed on request into Paper Trading 2 (book 2).

  GET  /claude-orders/                       the page
  GET  /api/paper/claude-orders/             the orders + which were already placed today
  POST /api/paper/claude-orders/place/       {"id": "...", "quantity": 100} -> a live market order in book 2

An order is placed at the LIVE price with its stop loss / target as percentages from the actual fill
(so the reward-to-risk of the recommendation is kept even if the price moved), tagged reason="claude".
Refused when: the recommendations are from an earlier day, it was already placed today, or the live
price has already passed its stop or target since the recommendation.
"""
import json
from decimal import Decimal
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from market.models import Company

from . import services as svc
from .models import PaperOrder

ORDERS_FILE = Path(__file__).resolve().parent / "claude_orders.json"
BOOK = 2
REASON = "claude"


def load_orders():
    try:
        data = json.loads(ORDERS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        data = {}
    data.setdefault("orders", [])
    data.setdefault("session", None)
    data.setdefault("generated", None)
    return data


def placed_today(account):
    """{(code, side): order} for today's Claude entries in this book."""
    start, end = svc._day_bounds_db(svc.today_ist())
    qs = (PaperOrder.objects.filter(account=account, reason=REASON, mode=svc.LIVE, created_at__gte=start, created_at__lt=end)
          .select_related("company").order_by("created_at"))
    return {(o.company.code, o.side): o for o in qs}


def _state(account):
    data = load_orders()
    placed = placed_today(account)
    for o in data["orders"]:
        hit = placed.get((o["code"], o["side"]))
        o["placed"] = None if not hit else {
            "order_id": hit.pk, "price": float(hit.price), "quantity": hit.quantity,
            "time": svc.to_ist(hit.created_at).isoformat()}
    data["today"] = svc.today_ist().isoformat()
    data["is_today"] = data["session"] == data["today"]
    data["balance"] = float(account.balance)
    return data


@require_GET
@ensure_csrf_cookie
def page(request):
    return render(request, "paper_trading/claude_orders.html", {"state": _state(svc.get_account(BOOK))})


@require_GET
def state(request):
    return JsonResponse(_state(svc.get_account(BOOK)))


def _error(msg, status=400, code=None):
    return JsonResponse({"error": msg, **({"code": code} if code else {})}, status=status)


@require_POST
def place(request):
    try:
        body = json.loads(request.body or b"{}")
    except ValueError:
        body = {}
    data = load_orders()
    rec = next((o for o in data["orders"] if o.get("id") == body.get("id")), None)
    if rec is None:
        return _error("Unknown recommendation.", 404)
    if data["session"] != svc.today_ist().isoformat():
        return _error(f"These recommendations are from {data['session']}. Only today's can be placed - ask Claude for a "
                      "fresh list.")
    company = Company.objects.filter(code=rec["code"]).first()
    if company is None:
        return _error(f"{rec['code']} is not in the company list.", 404)
    try:
        quantity = svc.parse_quantity(body.get("quantity"))
        account = svc.get_account(BOOK)
        with svc.ORDER_LOCK:                                     # check + place as one step (double clicks, two tabs)
            done = placed_today(account).get((rec["code"], rec["side"]))
            if done:
                return _error(f"Already placed today: {done.side} {done.quantity} {rec['code']} @ ₹{done.price:.2f}.",
                              409, "already_placed")
            price, market_time = svc.resolve_live_execution(company)
            p, stop, target = float(price), rec["stop"], rec["target"]
            if rec["side"] == "BUY" and p <= stop or rec["side"] == "SELL" and p >= stop:
                return _error(f"The price (₹{p:.2f}) has already passed the stop loss (₹{stop:.2f}) since the "
                              "recommendation - skip this one.", 409, "missed")
            if rec["side"] == "BUY" and p >= target or rec["side"] == "SELL" and p <= target:
                return _error(f"The price (₹{p:.2f}) has already reached the target (₹{target:.2f}) since the "
                              "recommendation - the move is gone.", 409, "missed")
            order, position, account = svc.execute_order(
                account=account, company=company, side=rec["side"], quantity=quantity, price=price,
                mode=svc.LIVE, replay_date=None, market_time=market_time, product=svc.INTRADAY,
                levels={"sl_percent": Decimal(str(rec["stop_pct"])), "target_percent": Decimal(str(rec["target_pct"]))},
                reason=REASON,
            )
    except svc.TradeError as exc:
        return _error(str(exc))
    order.company = company
    position.company = company
    return JsonResponse({"ok": True, "order": svc.serialize_order(order), "position": svc.serialize_position(position),
                         "balance": float(account.balance)})

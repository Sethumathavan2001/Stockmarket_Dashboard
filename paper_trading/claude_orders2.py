"""
Claude Orders 2.0: the Claude Orders page and its order handling (claude_orders.py) on a list of its own, built by
claude_board2.py and kept in claude_orders2.json.

  GET  /claude-orders-2/                      the page
  GET  /api/paper/claude-orders-2/            the orders + what was already placed today, per account
  POST /api/paper/claude-orders-2/place/      as /api/paper/claude-orders/place/
  POST /api/paper/claude-orders-2/refresh/    rebuild the list (background, see claude_board2)

Wallets, exits and the history are shared with the first version, and so is the rule of one order per stock
and side a day in each account: a stock placed from either page shows as placed on both.
"""
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from . import claude_board2, claude_orders
from . import services as svc

ORDERS_FILE = Path(__file__).resolve().parent / "claude_orders2.json"


def _state():
    return claude_orders._state(svc.get_account(claude_orders.BOOK), ORDERS_FILE, claude_board2)


@require_GET
@ensure_csrf_cookie
def page(request):
    return render(request, "paper_trading/claude_orders.html", {"state": _state(), "v2": True})


@require_GET
def state(request):
    return JsonResponse(_state())


@require_POST
def refresh(request):
    started = claude_board2.start_refresh(ORDERS_FILE)
    return JsonResponse({"started": started, "refresh": claude_board2.progress()}, status=202 if started else 409)


@require_POST
def place(request):
    return claude_orders.place_from(request, claude_orders.load_orders(ORDERS_FILE))

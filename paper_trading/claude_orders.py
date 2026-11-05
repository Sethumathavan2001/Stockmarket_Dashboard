"""
Claude recommended orders: the BUY / SELL picks of the Intraday Signal Board (claude_orders.json next to
this file), placed on request into Paper Trading 2 (book 2) and / or the real Angel One accounts in
angel.ACCOUNTS (Vivek, Sethu - see broker.py).

  GET  /claude-orders/                       the page
  GET  /api/paper/claude-orders/             the orders + what was already placed today, per account
  GET  /api/paper/claude-orders/wallets/     Paper Trading 2 balance and each Angel One account's funds
  POST /api/paper/claude-orders/place/       {"id", "account": "paper" | "vivek" | "sethu", "quantity",
                                              "stop_pct"?, "target_pct"?}
  POST /api/paper/claude-orders/refresh/     rebuild the list from current data (background, see claude_board)
  GET  /claude-orders/history/               every list made (ClaudeOrderList) and every Vivek / Sethu trade

Stop loss / target are percentages from the actual fill (editable on the page; the recommendation's by
default), so the reward-to-risk holds even if the price moved.  Paper: a live market order in book 2
tagged reason="claude".  Refused when: the recommendations are from an earlier day, the stock + side was
already placed today in that account, or the live price has already passed the recommendation's stop or
target.
"""
import json
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from market.models import Company

from . import angel, broker, claude_board, netroute, surveillance
from . import services as svc
from .models import BrokerOrder, ClaudeOrderList, PaperOrder

ORDERS_FILE = Path(__file__).resolve().parent / "claude_orders.json"
BOOK = 2
REASON = "claude"


def load_orders(path=None):
    try:
        data = json.loads((path or ORDERS_FILE).read_text(encoding="utf-8"))
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


def _state(account, path=None, board=claude_board):
    data = load_orders(path)
    try:
        claude_board.archive(data)                               # lists made before the history existed
    except Exception:
        pass
    placed = placed_today(account)
    real = broker.placed_today(svc.today_ist())
    nse, refused = surveillance.nse_lists(), surveillance.refused_recently()
    for o in data["orders"]:
        o["caution"] = surveillance.caution(o["code"], nse, refused)
        hit = placed.get((o["code"], o["side"]))
        o["placed"] = None if not hit else {
            "order_id": hit.pk, "price": float(hit.price), "quantity": hit.quantity,
            "time": svc.to_ist(hit.created_at).isoformat()}
        o["broker"] = {acc: (broker.serialize(real[(acc, o["code"], o["side"])]) if (acc, o["code"], o["side"]) in real
                             else None) for acc in angel.ACCOUNTS}
    data["accounts"] = [{"key": k, "name": v, "missing": angel.config(k)[1]} for k, v in angel.ACCOUNTS.items()]
    data["live_trading"] = angel.live_enabled()
    data["max_order_value"] = angel.max_order_value()
    data["today"] = svc.today_ist().isoformat()
    data["is_today"] = data["session"] == data["today"]
    data["balance"] = float(account.balance)
    data["refresh"] = board.progress()
    return data


@require_GET
def history(request):
    """Every saved Claude Orders list and every real trade in Vivek's / Sethu's account, newest first."""
    try:
        days = max(1, min(365, int(request.GET.get("days") or 30)))
    except ValueError:
        days = 30
    since = svc.today_ist() - timedelta(days=days - 1)
    acc = request.GET.get("account") or ""
    trades = BrokerOrder.objects.filter(session__gte=since).order_by("-created_at")
    if acc in angel.ACCOUNTS:
        trades = trades.filter(account=acc)
    trades = list(trades)
    by_rec = {}
    for t in trades:
        if t.status != BrokerOrder.REJECTED or t.account not in by_rec.get(t.rec_id, {}):
            by_rec.setdefault(t.rec_id, {})[t.account] = t

    totals = {a: {"name": n, "trades": 0, "closed": 0, "wins": 0, "losses": 0, "rejected": 0, "open": 0,
                  "pnl": 0.0} for a, n in angel.ACCOUNTS.items()}
    daily = {}
    for t in trades:
        tot = totals.setdefault(t.account, {"name": t.account, "trades": 0, "closed": 0, "wins": 0, "losses": 0,
                                            "rejected": 0, "open": 0, "pnl": 0.0})
        d = daily.setdefault(t.session, {a: {"trades": 0, "pnl": 0.0} for a in angel.ACCOUNTS})
        d.setdefault(t.account, {"trades": 0, "pnl": 0.0})
        if t.status in (BrokerOrder.REJECTED, BrokerOrder.ERROR):
            tot["rejected"] += 1
            continue
        tot["trades"] += 1
        d[t.account]["trades"] += 1
        if t.status == BrokerOrder.OPEN or t.status == BrokerOrder.PENDING:
            tot["open"] += 1
        if t.pnl is not None:
            p = float(t.pnl)
            tot["closed"] += 1
            tot["pnl"] += p
            d[t.account]["pnl"] += p
            tot["wins" if p > 0 else "losses"] += 1

    lists = []
    for row in ClaudeOrderList.objects.filter(session__gte=since):
        orders = []
        for o in row.payload.get("orders") or []:
            placed = by_rec.get(o.get("id"), {})
            orders.append({**o, "trades": {a: placed[a] for a in angel.ACCOUNTS if a in placed}})
        lists.append({"row": row, "orders": orders, "counts": row.payload.get("counts") or {},
                      "checked": row.payload.get("checked"),
                      "placed": sum(1 for o in orders if o["trades"])})
    return render(request, "paper_trading/claude_orders_history.html", {
        "lists": lists, "trades": trades, "totals": totals, "days": days, "account": acc,
        "accounts": angel.ACCOUNTS, "daily": sorted(daily.items(), reverse=True),
        "day_choices": (1, 7, 30, 90, 365)})


@require_GET
@ensure_csrf_cookie
def page(request):
    return render(request, "paper_trading/claude_orders.html", {"state": _state(svc.get_account(BOOK))})


@require_GET
def state(request):
    return JsonResponse(_state(svc.get_account(BOOK)))


@require_GET
def wallets(request):
    """Funds of each Angel One account (cached 30 s - the page polls this) and the paper wallet."""
    out = {"paper": {"configured": True, "available": float(svc.get_account(BOOK).balance), "error": None}}
    for acc in angel.ACCOUNTS:
        out[acc] = angel.funds(acc, use_cache=request.GET.get("fresh") != "1")
        out[acc]["route"] = netroute.describe(acc)            # which connection / public IP Angel One sees
    return JsonResponse(out)


@require_POST
def exit_trades(request):
    """Exit one real Angel One trade now: {"id": <trade>}."""
    try:
        body = json.loads(request.body or b"{}")
    except ValueError:
        body = {}
    try:
        bo = broker.exit_trade(int(body.get("id")))
    except (broker.BrokerError, TypeError, ValueError) as exc:
        return _error(str(exc))
    angel._funds.pop(bo.account, None)
    return JsonResponse({"ok": bo.status == "closed", "results": [{**broker.serialize(bo), "error": None}]})


@require_POST
def refresh(request):
    started = claude_board.start_refresh(ORDERS_FILE)
    return JsonResponse({"started": started, "refresh": claude_board.progress()}, status=202 if started else 409)


def _given(body, key, default):
    v = body.get(key)
    return default if v in (None, "") else v


def _error(msg, status=400, code=None):
    return JsonResponse({"error": msg, **({"code": code} if code else {})}, status=status)


@require_POST
def place(request):
    return place_from(request, load_orders())


def place_from(request, data):
    try:
        body = json.loads(request.body or b"{}")
    except ValueError:
        body = {}
    rec = next((o for o in data["orders"] if o.get("id") == body.get("id")), None)
    if rec is None:
        return _error("Unknown recommendation.", 404)
    if data["session"] != svc.today_ist().isoformat():
        return _error(f"These recommendations are from {data['session']}. Only today's can be placed - press Refresh "
                      "for a fresh list.")
    acc = str(body.get("account") or "paper").lower()
    if acc in angel.ACCOUNTS:
        try:
            bo = broker.place(acc, rec_id=rec["id"], code=rec["code"], side=rec["side"], quantity=body.get("quantity"),
                              stop_pct=_given(body, "stop_pct", rec["stop_pct"]),
                              target_pct=_given(body, "target_pct", rec["target_pct"]),
                              guard=(rec["stop"], rec["target"]), consent=bool(body.get("consent")))
        except broker.BrokerError as exc:
            return _error(str(exc), 409 if exc.code else 400, exc.code)
        angel._funds.pop(acc, None)                              # the wallet changed
        return JsonResponse({"ok": bo.status not in ("rejected", "error"), "broker": broker.serialize(bo)})
    if acc != "paper":
        return _error("Unknown account.")
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
            try:
                sl_pct = Decimal(str(_given(body, "stop_pct", rec["stop_pct"])))
                tp_pct = Decimal(str(_given(body, "target_pct", rec["target_pct"])))
            except ArithmeticError:
                return _error("Stop loss and target must be percentages.")
            if not (broker.MIN_PCT <= sl_pct <= broker.MAX_PCT and broker.MIN_PCT <= tp_pct <= broker.MAX_PCT):
                return _error(f"Stop loss and target must be between {broker.MIN_PCT}% and {broker.MAX_PCT}%.")
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
                levels={"sl_percent": sl_pct, "target_percent": tp_pct},
                reason=REASON,
            )
    except svc.TradeError as exc:
        return _error(str(exc))
    order.company = company
    position.company = company
    return JsonResponse({"ok": True, "order": svc.serialize_order(order), "position": svc.serialize_position(position),
                         "balance": float(account.balance)})

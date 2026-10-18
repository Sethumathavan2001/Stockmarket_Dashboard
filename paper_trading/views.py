import json
from datetime import timedelta
from functools import wraps

from django.conf import settings
from django.core.cache import cache
from django.db.models import Max, Min
from django.http import Http404, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST, require_safe

from market.models import Company, FiveMinuteData  # adjust if your app is named differently

from . import charges
from . import services as svc
from . import signals as sig
from .models import PaperOrder


def api(view):
    """Turn TradeError into a clean 400 JSON response."""

    @wraps(view)
    def wrapper(request, *args, **kwargs):
        try:
            return view(request, *args, **kwargs)
        except svc.AlreadyClosed as exc:
            return JsonResponse({"error": str(exc), "code": "already_closed"}, status=409)
        except svc.TradeError as exc:
            return JsonResponse({"error": str(exc)}, status=400)

    return wrapper


def _json_body(request):
    try:
        body = json.loads(request.body or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise svc.TradeError("Invalid request body.")
    if not isinstance(body, dict):
        raise svc.TradeError("Invalid request body.")
    return body


def _account_get(request):
    """The account of the book named in the query string (?book=1|2)."""
    return svc.get_account(svc.parse_book(request.GET.get("book")))


def _account_post(body):
    """The account of the book named in a JSON body ({"book": 1|2})."""
    return svc.get_account(svc.parse_book(body.get("book")))


def _get_company(code):
    company = Company.objects.filter(code=code).first() if code else None
    if company is None:
        raise svc.TradeError("Unknown company.")
    return company


def _mode_and_date(source):
    """Read ?mode=&date= (or JSON body fields) into (mode, replay_date|None)."""
    mode = source.get("mode", svc.LIVE)
    if mode == svc.LIVE:
        return svc.LIVE, None
    if mode == svc.REPLAY:
        return svc.REPLAY, svc.parse_replay_date(source.get("date"))
    raise svc.TradeError("Mode must be 'live' or 'replay'.")


# ---------------------------------------------------------------- page ----
@require_safe
@ensure_csrf_cookie
def paper_trading_page(request, company_code=None, mode=None, date=None, book=1):
    """
    /paper-trading/                              the page, nothing selected
    /paper-trading/<code>/live/                  opens straight onto that stock, Live
    /paper-trading/<code>/old/<YYYY-MM-DD>/      ... Old mode on that date
    /paper-trading/<code>/old/?date=YYYY-MM-DD   same thing, date as a query string

    Direct links are plain GETs; the page starts the session by itself.  A bad
    date still renders the page (HTTP 400) with the reason shown, nothing started.
    """
    companies = Company.objects.all().order_by("code")
    initial = {"company": "", "mode": svc.LIVE, "date": None, "error": "", "autostart": False}
    status = 200

    if company_code is not None:
        company = Company.objects.filter(code=company_code).first()
        if company is None:
            raise Http404("Unknown company")
        initial["company"] = company.code
        if mode == "old":
            initial["mode"] = svc.REPLAY
            raw = date or request.GET.get("date")
            initial["date"] = raw or None
            try:
                initial["date"] = svc.parse_replay_date(raw).isoformat()
            except svc.TradeError as exc:
                initial["error"] = str(exc)
                status = 400
        initial["autostart"] = not initial["error"]

    return render(
        request,
        "paper_trading/paper_trading.html",
        {
            "companies": companies,
            "live_poll_seconds": getattr(settings, "PAPER_LIVE_POLL_SECONDS", 15),
            "squareoff_label": svc.intraday_squareoff_time().strftime("%I:%M %p").lstrip("0"),
            "initial": initial,
            "book": book,
            "brokers": charges.broker_info(),
            "page_url": reverse("paper_trading:page2" if book == 2 else "paper_trading:page"),
        },
        status=status,
    )


# ----------------------------------------------------------------- API ----
@require_GET
@api
def account_state(request):
    account = _account_get(request)
    mode, replay_date = _mode_and_date(request.GET)
    filled = svc.process_live_limits(account) if mode == svc.LIVE else []
    if filled:
        account.refresh_from_db()
    company = None
    if request.GET.get("company"):
        company = _get_company(request.GET["company"])
    snap = svc.account_snapshot(account, mode, replay_date, company)
    snap["limit_fills"] = [svc.serialize_order(o) for o in filled]
    return JsonResponse(snap)


@require_POST
@api
def wallet_add(request):
    body = _json_body(request)
    account = svc.add_to_wallet(_account_post(body), body.get("amount"))
    return JsonResponse({"balance": float(account.balance)})


@require_POST
@api
def wallet_withdraw(request):
    body = _json_body(request)
    account = svc.withdraw_from_wallet(_account_post(body), body.get("amount"))
    return JsonResponse({"balance": float(account.balance)})


@require_GET
@api
def date_range(request, company_code):
    company = _get_company(company_code)
    agg = FiveMinuteData.objects.filter(company=company).aggregate(lo=Min("timestamp"), hi=Max("timestamp"))
    if not agg["lo"]:
        return JsonResponse({"min": None, "max": None})

    yesterday = svc.today_ist() - timedelta(days=1)
    lo = svc.to_ist(agg["lo"]).date()
    hi = min(svc.to_ist(agg["hi"]).date(), yesterday)
    if hi < lo:
        return JsonResponse({"min": None, "max": None})
    return JsonResponse({"min": lo.isoformat(), "max": hi.isoformat()})


@require_GET
@api
def candles(request, company_code):
    company = _get_company(company_code)
    mode, replay_date = _mode_and_date(request.GET)
    if request.GET.get("interval") == "1m":          # 1-minute candles: Yahoo only (for display)
        rows = svc.live_candles_1m(company) if mode == svc.LIVE else svc.replay_candles_1m(company, replay_date)
        return JsonResponse({"interval": "1m", "candles": rows})
    if mode == svc.LIVE:
        return JsonResponse(svc.live_snapshot(company))
    return JsonResponse(svc.replay_snapshot(company, replay_date))


@require_GET
@api
def strategy_signals(request, company_code):
    """
    The six strategies' historical results for this stock, plus the session day's
    entries / exits.  Old mode returns the whole day (the page reveals each signal
    only when the replay reaches it); Live mode uses completed candles only.
    """
    company = _get_company(company_code)
    mode, replay_date = _mode_and_date(request.GET)
    return JsonResponse(sig.session_signals(company, mode, replay_date))


@require_POST
@api
def place_order(request):
    body = _json_body(request)
    company = _get_company(body.get("company"))

    side = str(body.get("side", "")).upper()
    if side not in (PaperOrder.SIDE_BUY, PaperOrder.SIDE_SELL):
        raise svc.TradeError("Side must be BUY or SELL.")
    quantity = svc.parse_quantity(body.get("quantity"))
    mode, replay_date = _mode_and_date(body)
    product = svc.parse_product(body.get("product"))
    levels = svc.parse_bracket(body)
    reason = str(body.get("reason") or "")
    if reason not in ("", "stop_loss", "target", "session_end"):
        reason = ""

    # stop loss / target / square-off and the Exit button only ever close what is open
    close_only = bool(body.get("close_only")) or reason in ("stop_loss", "target", "session_end")

    account = _account_post(body)
    # the same order sent twice (double click, network retry) runs once
    coid = str(body.get("client_order_id") or "")[:64]
    if coid and not cache.add(f"paper:coid:{account.pk}:{coid}", 1, 3600):
        raise svc.AlreadyClosed("This order was already received - the duplicate was ignored.")

    if str(body.get("order_type", "MARKET")).upper() == "LIMIT":
        if close_only:
            raise svc.TradeError("Exits are market orders.")
        limit_price = svc.parse_optional_price(body.get("limit_price"), "Limit price")
        current, market_time = (None, None) if mode == svc.LIVE else svc.resolve_replay_execution(
            company, replay_date, body.get("market_time"), body.get("price"))
        lo, order = svc.place_limit_order(
            account=account, company=company, side=side, quantity=quantity, limit_price=limit_price,
            mode=mode, replay_date=replay_date, product=product, levels=levels,
            current_price=current, market_time=market_time,
        )
        account.refresh_from_db()
        if order:
            order.company = company
        return JsonResponse({"ok": True, "limit_order": svc.serialize_limit(lo),
                             "order": svc.serialize_order(order) if order else None,
                             "balance": float(account.balance)})

    if mode == svc.LIVE:
        price, market_time = svc.resolve_live_execution(company)
    else:
        price, market_time = svc.resolve_replay_execution(
            company, replay_date, body.get("market_time"), body.get("price")
        )

    order, position, account = svc.execute_order(
        account=account,
        company=company,
        side=side,
        quantity=quantity,
        price=price,
        mode=mode,
        replay_date=replay_date,
        market_time=market_time,
        levels=levels,
        reason=reason,
        product=product,
        close_only=close_only,
    )
    order.company = company  # avoid an extra query when serialising
    position.company = company
    return JsonResponse(
        {
            "ok": True,
            "order": svc.serialize_order(order),
            "position": svc.serialize_position(position) if position.quantity else None,
            "balance": float(account.balance),
        }
    )


@require_POST
@api
def bracket_update(request):
    """Set / change / clear the stop loss and target of an open position (no trade)."""
    body = _json_body(request)
    company = _get_company(body.get("company"))
    mode, replay_date = _mode_and_date(body)
    product = svc.parse_product(body.get("product"))
    clear_sl = bool(body.get("clear_stop_loss"))
    clear_target = bool(body.get("clear_target"))
    levels = svc.parse_bracket(body)
    if not clear_sl and not clear_target and not any(v is not None for v in levels.values()):
        raise svc.TradeError("Enter a stop loss and/or target first.")

    position = svc.update_bracket(
        account=_account_post(body),
        company=company,
        mode=mode,
        replay_date=replay_date,
        levels=levels,
        clear_sl=clear_sl,
        clear_target=clear_target,
        product=product,
    )
    position.company = company
    return JsonResponse({"ok": True, "position": svc.serialize_position(position)})


@require_GET
@api
def quotes(request):
    """
    Live prices for several companies at once (the page uses it to value and
    monitor EVERY open position, not just the stock on the chart).
    GET /api/paper/quotes/?companies=TCS,INFY
    """
    codes = [c.strip() for c in request.GET.get("companies", "").split(",") if c.strip()]
    codes = list(dict.fromkeys(codes))[: svc.MAX_QUOTE_COMPANIES]
    found = {c.code: c for c in Company.objects.filter(code__in=codes)}
    return JsonResponse(
        {
            "quotes": {code: svc.live_quote(found[code]) for code in codes if code in found},
            "unknown": [c for c in codes if c not in found],
        }
    )


@require_POST
@api
def squareoff(request):
    """Close live INTRADAY positions that are due (older than today, or past the cutoff)."""
    account = _account_post(_json_body(request))
    result = svc.squareoff_live_intraday(account)
    account.refresh_from_db()
    return JsonResponse(
        {
            "closed": [svc.serialize_order(o) for o in result["closed"]],
            "skipped": result["skipped"],
            "balance": float(account.balance),
        }
    )


@require_POST
@api
def limit_cancel(request, limit_id):
    body = _json_body(request)
    lo = svc.cancel_limit_order(_account_post(body), limit_id)
    return JsonResponse({"ok": True, "limit_order": {"id": lo.id, "status": lo.status}})


@require_POST
@api
def limit_fill(request, limit_id):
    """Old mode: the replay price crossed this limit order inside the candle at market_time."""
    body = _json_body(request)
    account = _account_post(body)
    lo, order = svc.fill_replay_limit(account, limit_id, body.get("market_time"))
    account.refresh_from_db()
    return JsonResponse({"ok": True, "limit_order": svc.serialize_limit(lo),
                         "order": svc.serialize_order(order) if order else None,
                         "balance": float(account.balance)})


@require_POST
@api
def limit_expire(request):
    """Old mode: the replay day ended - its open limit orders expire."""
    body = _json_body(request)
    count = svc.expire_replay_limits(_account_post(body), svc.parse_replay_date(body.get("date")))
    return JsonResponse({"ok": True, "expired": count})

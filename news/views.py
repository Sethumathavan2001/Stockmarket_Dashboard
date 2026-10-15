"""
News page and its API.

  GET /news/                                   the page
  GET /api/news/market/?refresh=1              market headlines (ET, Moneycontrol, Mint, Business Standard)
  GET /api/news/announcements/?days=1          NSE corporate announcements, market-wide
  GET /api/news/stock/<code>/                  one company: NSE filings (30 days) + Google News stories
"""
import logging

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from market.models import Company

from . import fetch

logger = logging.getLogger(__name__)


def _find(code):
    code = (code or "").strip().upper()
    for c in (code, f"{code}.NS"):
        company = Company.objects.filter(code__iexact=c).first()
        if company:
            return company
    return None


@require_GET
def page(request):
    companies = Company.objects.exclude(sector="ETF").order_by("code").values("code", "name")
    return render(request, "news/news.html", {"companies": list(companies),
                                              "initial_company": request.GET.get("company", "")})


@require_GET
def market(request):
    return JsonResponse(fetch.market_news(refresh=request.GET.get("refresh") == "1"))


@require_GET
def announcements(request):
    try:
        days = int(request.GET.get("days", 1))
    except ValueError:
        days = 1
    try:
        return JsonResponse(fetch.announcements(days, refresh=request.GET.get("refresh") == "1"))
    except Exception as exc:
        logger.warning("NSE announcements failed: %s", exc)
        return JsonResponse({"error": f"NSE announcements are unavailable right now ({exc})."}, status=502)


@require_GET
def stock(request, company_code):
    company = _find(company_code)
    if company is None:
        return JsonResponse({"error": f"{company_code} is not in the company list."}, status=404)
    return JsonResponse(fetch.stock_news(company, refresh=request.GET.get("refresh") == "1"))

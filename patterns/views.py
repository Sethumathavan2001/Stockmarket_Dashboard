"""
Intraday Candle Predictor page and API.

  GET /candle-predictor/                                       the page
  GET /api/candle-predictor/<code>/?interval=5m|1m&day=YYYY-MM-DD&asof=HH:MM&news=1
        candles of the day, detected patterns (with this stock's history for each), the
        analog forecast from "asof" (default: the latest candle) and its tested accuracy.
"""
import logging
import math
from datetime import date, datetime

from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_GET

from market.models import Company

from . import engine

logger = logging.getLogger(__name__)


def _clean(obj):
    """JSON-safe: NaN / inf -> None, numpy numbers -> Python numbers."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):                      # numpy scalar
        return _clean(obj.item())
    return obj


@require_GET
def candle_predictor_view(request):
    companies = Company.objects.exclude(sector="ETF").order_by("code").values("code", "name")
    return render(request, "patterns/candle_predictor.html", {
        "companies": list(companies),
        "initial": {"company": request.GET.get("company", ""), "interval": request.GET.get("interval", "5m"),
                    "news": request.GET.get("news") == "1"},
    })


@require_GET
def candle_prediction_data(request, company_code):
    company = get_object_or_404(Company, code=company_code)
    interval = request.GET.get("interval", "5m")
    try:
        day = date.fromisoformat(request.GET["day"]) if request.GET.get("day") else None
        asof = datetime.strptime(request.GET["asof"], "%H:%M").time() if request.GET.get("asof") else None
    except ValueError:
        return JsonResponse({"error": "Invalid day or time."}, status=400)
    try:
        news = None
        if request.GET.get("news") == "1":
            from news.signal import news_signal
            news = news_signal
        data = engine.predict(company, interval=interval, day=day, asof=asof, news=news)
    except engine.PredictorError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Candle predictor failed for %s", company_code)
        return JsonResponse({"error": f"Prediction failed: {exc}"}, status=500)
    return JsonResponse(_clean(data))

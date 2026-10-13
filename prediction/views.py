"""
Stock Prediction page and API.

  GET  /stock_prediction/                      the page (chart first, then tables)
  GET  /api/predict/<code>/?timeframe=...      live prediction + chart data + model leaderboard
  POST /api/train/<code>/                      train the model zoo for a stock (background)
  GET  /api/train/<code>/                      training progress
  GET  /api/predict/                           every trained stock (from the saved manifests)
"""
import logging
import math
import threading

from django.core.cache import cache
from django.db import connection
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods

from market.models import Company

from . import datasets as ds
from . import engine

logger = logging.getLogger(__name__)

TRAIN_KEY = "prediction:train:{code}"
RUNNING_KEY = "prediction:train:running"      # one stock trains at a time - training uses every CPU core
_START_LOCK = threading.Lock()


def _clean(obj):
    """JSON-safe: NaN / inf -> None (browsers reject NaN in JSON)."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


@ensure_csrf_cookie
@require_GET
def stock_prediction_view(request):
    companies = Company.objects.exclude(sector="ETF").order_by("code").values("code", "name")
    trained = {m["company"] for m in engine.list_manifests()}
    return render(request, "prediction/stock_prediction.html", {
        "companies": [dict(c, trained=c["code"] in trained) for c in companies],
        "initial": {"company": request.GET.get("company", ""), "timeframe": request.GET.get("timeframe", "intraday")},
        "model_names": engine.model_names(),
    })


def _train_worker(company_id, code):
    key = TRAIN_KEY.format(code=code)

    def progress(message, fraction):
        cache.set(key, {"state": "training", "message": message, "percent": round(fraction * 100, 1)}, 3600)

    try:
        company = Company.objects.get(id=company_id)
        manifest = engine.train_stock(company, progress=progress)
        errors = {tf: v["error"] for tf, v in manifest["timeframes"].items() if "error" in v}
        cache.set(key, {"state": "completed", "percent": 100, "trained_at": manifest["trained_at"],
                        "message": "Training finished.", "errors": errors}, 3600)
    except Exception as exc:
        logger.exception("Training failed for %s", code)
        cache.set(key, {"state": "failed", "message": f"Training failed: {exc}"}, 3600)
    finally:
        cache.delete(RUNNING_KEY)
        connection.close()


@require_http_methods(["GET", "POST"])
def train_company(request, company_code):
    company = get_object_or_404(Company, code=company_code)
    key = TRAIN_KEY.format(code=company.code)
    if request.method == "GET":
        return JsonResponse(cache.get(key) or {"state": "idle"})

    with _START_LOCK:
        running = cache.get(RUNNING_KEY)
        if running and (cache.get(TRAIN_KEY.format(code=running)) or {}).get("state") == "training":
            if running == company.code:
                return JsonResponse(cache.get(key), status=202)
            return JsonResponse({"state": "busy", "message": f"{running} is training right now - one stock at a time. "
                                                             "Try again when it finishes."}, status=409)
        cache.set(RUNNING_KEY, company.code, 3600)
        cache.set(key, {"state": "training", "message": "Starting...", "percent": 0}, 3600)
        threading.Thread(target=_train_worker, args=(company.id, company.code), daemon=True).start()
    return JsonResponse(cache.get(key), status=202)


@require_GET
def predict_company(request, company_code):
    company = get_object_or_404(Company, code=company_code)
    timeframe = request.GET.get("timeframe", "intraday")
    if timeframe not in ds.TIMEFRAMES:
        return JsonResponse({"error": "timeframe must be 'intraday' or 'daily'."}, status=400)
    try:
        data = engine.predict_stock(company, timeframe)
    except engine.NotTrained as exc:
        return JsonResponse({"error": str(exc), "not_trained": True}, status=404)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Prediction failed for %s", company_code)
        return JsonResponse({"error": f"Prediction failed: {exc}"}, status=500)
    return JsonResponse(_clean(data))


@require_GET
def models_overview(request):
    return JsonResponse(_clean({"stocks": engine.overview()}))

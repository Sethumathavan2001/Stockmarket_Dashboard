import logging
import threading

from django.core.cache import cache
from django.shortcuts import render, get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from market.models import Company  # adjust if your app is named differently
from .trainer import train_company_model
from .predictor import predict_company, PredictionError
from .serializers import CompanyPredictionSerializer

logger = logging.getLogger(__name__)

TRAIN_STATUS_KEY = "prediction:train_status:{code}"
TRAIN_STATUS_TTL = 60 * 60  # 1 hour


def stock_prediction_view(request):
    """Renders the /stock_prediction/ page: company select, Train + Predict buttons."""
    companies = Company.objects.all().order_by("name")
    return render(request, "prediction/stock_prediction.html", {"companies": companies})


def _run_training(company_id, company_code):
    """Runs in a background thread so the Train button's request returns immediately.

    For heavier production workloads, swap this thread for a proper task
    queue (Celery/RQ) — the training call itself (`train_company_model`) is
    already a plain, synchronous function so that swap is a thin wrapper.
    """
    cache.set(TRAIN_STATUS_KEY.format(code=company_code), {"state": "training"}, TRAIN_STATUS_TTL)
    try:
        company = Company.objects.get(id=company_id)
        bundle = train_company_model(company)
        summary = {
            str(h): {"mae": m["mae"], "directional_accuracy": m["directional_accuracy"]}
            for h, m in bundle["metrics"].items()
        }
        cache.set(
            TRAIN_STATUS_KEY.format(code=company_code),
            {"state": "completed", "metrics": summary, "trained_at": bundle["trained_at"]},
            TRAIN_STATUS_TTL,
        )
    except Exception as exc:
        logger.exception("Training failed for %s", company_code)
        cache.set(
            TRAIN_STATUS_KEY.format(code=company_code),
            {"state": "failed", "error": str(exc)},
            TRAIN_STATUS_TTL,
        )


class TrainCompanyView(APIView):
    """
    POST /api/train/<company_code>/  -> starts training in the background, returns 202
    GET  /api/train/<company_code>/  -> polls current training status
    """

    def post(self, request, company_code):
        company = get_object_or_404(Company, code=company_code)
        current = cache.get(TRAIN_STATUS_KEY.format(code=company_code))
        if current and current.get("state") == "training":
            return Response(
                {"state": "training", "detail": "Training already in progress."},
                status=status.HTTP_202_ACCEPTED,
            )

        thread = threading.Thread(target=_run_training, args=(company.id, company.code), daemon=True)
        thread.start()
        return Response({"state": "training_started", "company": company.code}, status=status.HTTP_202_ACCEPTED)

    def get(self, request, company_code):
        state = cache.get(TRAIN_STATUS_KEY.format(code=company_code), {"state": "idle"})
        return Response(state)


class PredictCompanyView(APIView):
    """GET /api/predict/<company_code>/"""

    def get(self, request, company_code):
        company = get_object_or_404(Company, code=company_code)
        try:
            result = predict_company(company)
        except PredictionError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:
            logger.exception("Prediction failed for %s", company_code)
            return Response(
                {"error": "Prediction failed.", "detail": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        serializer = CompanyPredictionSerializer(result)
        return Response(serializer.data)


class PredictAllCompaniesView(APIView):
    """GET /api/predict/ — predictions for every company. Reuses in-memory
    yfinance fetches per company (no cross-company request caching is added
    here since each company needs its own ticker); see README for notes on
    reducing yfinance calls at scale.
    """

    def get(self, request):
        companies = Company.objects.all().order_by("code")
        results = []
        errors = []
        for company in companies:
            try:
                results.append(predict_company(company))
            except PredictionError as exc:
                errors.append({"company": company.code, "error": str(exc)})
            except Exception as exc:
                logger.exception("Prediction failed for %s", company.code)
                errors.append({"company": company.code, "error": "Prediction failed.", "detail": str(exc)})

        serializer = CompanyPredictionSerializer(results, many=True)
        return Response({"predictions": serializer.data, "errors": errors})

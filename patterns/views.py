"""
Views for the candlestick-pattern + next-candle projection page.

Reuses the existing `prediction` app's data loader (`get_live_dataframe`:
historical DB candles + today's in-memory yfinance candles, never written
back to the database) so this page follows the exact same data-handling
rules as the trained-model prediction system built earlier.
"""
import datetime

from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404

from market.models import Company  # adjust if your app is named differently
from prediction.data import get_live_dataframe
from prediction.market_time import get_market_status, IST

from .forecast import generate_forecast, DEFAULT_CUTOFF_TIME


def candle_predictor_view(request):
    """Renders the page: company select + Show button."""
    companies = Company.objects.all().order_by("name")
    return render(request, "patterns/candle_predictor.html", {"companies": companies})


def _row_to_candle(row):
    return {
        "timestamp": row["timestamp"],
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
    }


def candle_prediction_data(request, company_code):
    """
    GET /api/candle-predictor/<company_code>/

    Returns today's actual 5-minute candles (9:15 -> now) with detected
    patterns, plus heuristic predicted candles from now through the
    prediction cutoff (default 3:15 PM IST).
    """
    company = get_object_or_404(Company, code=company_code)

    live_df = get_live_dataframe(company, lookback_days=5)
    if live_df.empty:
        return JsonResponse({"error": f"No candle data available for {company.code}."}, status=400)

    today = datetime.datetime.now(IST).date()
    today_df = live_df[live_df["timestamp"].dt.date == today].sort_values("timestamp")

    if today_df.empty:
        return JsonResponse(
            {
                "error": (
                    f"No candles for {company.code} yet today (market status: "
                    f"{get_market_status()}). Try again once the market opens at 9:15."
                )
            },
            status=400,
        )

    actual_candles = [_row_to_candle(row) for _, row in today_df.iterrows()]
    annotated_actual, predicted, bias_summary = generate_forecast(actual_candles)

    def serialize(c, is_predicted=False):
        return {
            "x": c["timestamp"].isoformat(),
            "o": c["open"],
            "h": c["high"],
            "l": c["low"],
            "c": c["close"],
            "is_predicted": is_predicted,
            "patterns": c.get("patterns", []),
        }

    return JsonResponse(
        {
            "company": company.code,
            "market_status": get_market_status(),
            "as_of": datetime.datetime.now(IST).isoformat(),
            "prediction_cutoff": DEFAULT_CUTOFF_TIME.strftime("%H:%M"),
            "actual_candles": [serialize(c) for c in annotated_actual],
            "predicted_candles": [serialize(c, is_predicted=True) for c in predicted],
            "bias_summary": bias_summary,
            "disclaimer": (
                "Predicted candles are a simple rule-based extrapolation from "
                "recent candlestick patterns and recent volatility — not a "
                "statistical or machine-learning forecast, and not trading advice."
            ),
        }
    )

"""
analysis_engine.py

Production-quality re-implementation of the Company analysis refresh pipeline.

Design goals (per spec):
    - Zero N+1 queries. DailyData is loaded exactly once.
    - All grouping / windowing / statistics is done in memory using
      Python's `collections` and `statistics` modules.
    - A single `bulk_update()` call persists every company at the end.
    - A failure on one company never aborts the batch - that company's
      analysis fields are reset to None and processing continues.
    - The "current market date" is derived exclusively from the latest
      row present in DailyData - `timezone.now()` is never used.

NOTE ON IMPORTS:
    Adjust the model import below (`from .models import Company, DailyData`)
    to match wherever your app actually lives. Everything else is
    self-contained and does not depend on app layout.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

from django.http import JsonResponse
from django.db import transaction

from .models import Company, DailyData  # noqa: adjust to your app path

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Every analysis field on Company. Used both for bulk_update() and for
# resetting a company to a "no data / error" state.
ANALYSIS_FIELDS: List[str] = [
    "latest_price",
    "low_yearly",
    "low_monthly",
    "low_weekly",
    "low_monthly_avg",
    "low_monthly_min",
    "low_monthly_max",
    "low_weekly_avg",
    "low_weekly_min",
    "low_weekly_max",
    "high_yearly",
    "high_monthly",
    "high_weekly",
    "high_monthly_avg",
    "high_monthly_min",
    "high_monthly_max",
    "high_weekly_avg",
    "high_weekly_min",
    "high_weekly_max",
]

YEARLY_WINDOW = timedelta(weeks=52)
MONTHLY_WINDOW = timedelta(weeks=4)
WEEKLY_WINDOW = timedelta(days=7)


# =============================================================================
# LIGHTWEIGHT IN-MEMORY RECORD
# =============================================================================

@dataclass(slots=True)
class Bar:
    """A minimal, memory-light stand-in for a DailyData row."""

    date: date
    open: float
    low: float
    high: float
    close: float


# =============================================================================
# CORE FORMULAS
# =============================================================================

def near_low_percentage(low_price: Optional[float], current_price: Optional[float]) -> Optional[float]:
    """
    Nearness-to-low formula: (low_price / current_price) * 100.

    A low price of 10 with a current price of 100 means the stock is
    trading 10% "above" its low reference point -> result = 10.0
    """
    if low_price is None or current_price in (None, 0):
        return None
    return (low_price / current_price) * 100.0


def near_high_percentage(current_price: Optional[float], high_price: Optional[float]) -> Optional[float]:
    """
    Nearness-to-high formula: (current_price / high_price) * 100.
    """
    if high_price in (None, 0) or current_price is None:
        return None
    return (current_price / high_price) * 100.0


def fall_from_open_percentage(open_price: Optional[float], low_price: Optional[float]) -> Optional[float]:
    """(open - low) / open * 100 -- used for monthly/weekly LOW statistics."""
    if open_price in (None, 0) or low_price is None:
        return None
    return ((open_price - low_price) / open_price) * 100.0


def rise_from_open_percentage(open_price: Optional[float], high_price: Optional[float]) -> Optional[float]:
    """(high - open) / open * 100 -- used for monthly/weekly HIGH statistics."""
    if open_price in (None, 0) or high_price is None:
        return None
    return ((high_price - open_price) / open_price) * 100.0


# =============================================================================
# DATA LOADING (single query, grouped in memory)
# =============================================================================

def load_all_daily_data() -> Dict[int, List[Bar]]:
    """
    Load ALL DailyData rows in a single query and group them by company_id.

    Rows are streamed in ascending date order so that, per company, the
    resulting list is already sorted by date - no per-company sorting
    needed later.
    """
    grouped: Dict[int, List[Bar]] = defaultdict(list)

    queryset = (
        DailyData.objects
        .order_by("date")
        .only("company_id", "date", "open", "low", "high", "close")
        .iterator(chunk_size=5000)
    )

    for row in queryset:
        # DailyData's numeric columns are typically DecimalField. Cast to
        # float once, here, so every downstream calculation (which mixes
        # in plain float literals like 100.0) works without TypeErrors.
        grouped[row.company_id].append(
            Bar(
                date=row.date,
                open=float(row.open),
                low=float(row.low),
                high=float(row.high),
                close=float(row.close),
            )
        )

    return grouped


def get_latest_trading_date() -> Optional[date]:
    """
    The single source of truth for "today" in this engine: the most
    recent date present in DailyData. Returns None if the table is empty.
    """
    latest_row = DailyData.objects.order_by("-date").first()
    return latest_row.date if latest_row else None


# =============================================================================
# WINDOW HELPERS (last N days relative to latest_date)
# =============================================================================

def bars_in_window(bars: List[Bar], start_date: date, end_date: date) -> List[Bar]:
    """Return bars with start_date <= date <= end_date. Bars are pre-sorted."""
    return [b for b in bars if start_date <= b.date <= end_date]


def get_current_bar(bars: List[Bar], latest_date: date) -> Bar:
    """
    Return the bar for the current market date.

    Bars are sorted ascending, so the current bar - if present - is the
    last element. Raises ValueError if the company has no row on the
    latest trading date (caller treats this as "no data").
    """
    if not bars or bars[-1].date != latest_date:
        raise ValueError("No DailyData row for the latest trading date")
    return bars[-1]


# =============================================================================
# CALENDAR GROUPING (for monthly / weekly aggregate statistics)
# =============================================================================

def group_by_month(bars: List[Bar]) -> Dict[Tuple[int, int], List[Bar]]:
    """Group bars by (calendar year, calendar month)."""
    groups: Dict[Tuple[int, int], List[Bar]] = defaultdict(list)
    for b in bars:
        groups[(b.date.year, b.date.month)].append(b)
    return groups


def group_by_iso_week(bars: List[Bar]) -> Dict[Tuple[int, int], List[Bar]]:
    """Group bars by (ISO year, ISO week number)."""
    groups: Dict[Tuple[int, int], List[Bar]] = defaultdict(list)
    for b in bars:
        iso_year, iso_week, _ = b.date.isocalendar()
        groups[(iso_year, iso_week)].append(b)
    return groups


# =============================================================================
# AGGREGATE STATISTICS (avg / min / max across every month or week)
# =============================================================================

def _avg_min_max(values: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Shared reducer: returns (average, minimum, maximum) or (None, None, None)."""
    if not values:
        return None, None, None
    return statistics.mean(values), min(values), max(values)


def monthly_low_statistics(bars: List[Bar]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    For every calendar month present in `bars`:
        open  = opening price of that month's first trading day
        low   = lowest LOW of the whole month
        pct   = (open - low) / open * 100

    Returns (avg, min, max) of all monthly percentages.
    """
    percentages: List[float] = []
    for month_bars in group_by_month(bars).values():
        first_day_open = month_bars[0].open  # month_bars is already date-sorted
        month_low = min(b.low for b in month_bars)
        pct = fall_from_open_percentage(first_day_open, month_low)
        if pct is not None:
            percentages.append(pct)
    return _avg_min_max(percentages)


def weekly_low_statistics(bars: List[Bar]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Same as monthly_low_statistics but grouped by ISO week."""
    percentages: List[float] = []
    for week_bars in group_by_iso_week(bars).values():
        first_day_open = week_bars[0].open
        week_low = min(b.low for b in week_bars)
        pct = fall_from_open_percentage(first_day_open, week_low)
        if pct is not None:
            percentages.append(pct)
    return _avg_min_max(percentages)


def monthly_high_statistics(bars: List[Bar]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    For every calendar month present in `bars`:
        open  = opening price of that month's first trading day
        high  = highest HIGH of the whole month
        pct   = (high - open) / open * 100

    Returns (avg, min, max) of all monthly percentages.
    """
    percentages: List[float] = []
    for month_bars in group_by_month(bars).values():
        first_day_open = month_bars[0].open
        month_high = max(b.high for b in month_bars)
        pct = rise_from_open_percentage(first_day_open, month_high)
        if pct is not None:
            percentages.append(pct)
    return _avg_min_max(percentages)


def weekly_high_statistics(bars: List[Bar]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Same as monthly_high_statistics but grouped by ISO week."""
    percentages: List[float] = []
    for week_bars in group_by_iso_week(bars).values():
        first_day_open = week_bars[0].open
        week_high = max(b.high for b in week_bars)
        pct = rise_from_open_percentage(first_day_open, week_high)
        if pct is not None:
            percentages.append(pct)
    return _avg_min_max(percentages)


# =============================================================================
# RESET / ERROR HANDLING
# =============================================================================

def reset_analysis(company: Company) -> None:
    """Blank out every analysis field on `company` (no save - caller bulk_updates)."""
    for field_name in ANALYSIS_FIELDS:
        setattr(company, field_name, None)


# =============================================================================
# PER-COMPANY ANALYSIS
# =============================================================================

def analyze_company(company: Company, bars: List[Bar], latest_date: date) -> None:
    """
    Compute and set every analysis field on `company` using only the
    in-memory `bars` list for that company. Raises on any data problem;
    the caller is responsible for catching and resetting on failure.
    """
    current_bar = get_current_bar(bars, latest_date)
    current_price = current_bar.close
    print(current_price)
    company.latest_price = current_price
    # ---- Recent windows relative to the latest trading date ----
    yearly_window = bars_in_window(bars, latest_date - YEARLY_WINDOW, latest_date)
    monthly_window = bars_in_window(bars, latest_date - MONTHLY_WINDOW, latest_date)
    weekly_window = bars_in_window(bars, latest_date - WEEKLY_WINDOW, latest_date)

    low_52 = min((b.low for b in yearly_window), default=None)
    low_4w = min((b.low for b in monthly_window), default=None)
    low_7d = min((b.low for b in weekly_window), default=None)

    high_52 = max((b.high for b in yearly_window), default=None)
    high_4w = max((b.high for b in monthly_window), default=None)
    high_7d = max((b.high for b in weekly_window), default=None)

    company.low_yearly = near_low_percentage(low_52, current_price)
    company.low_monthly = near_low_percentage(low_4w, current_price)
    company.low_weekly = near_low_percentage(low_7d, current_price)

    company.high_yearly = near_high_percentage(current_price, high_52)
    company.high_monthly = near_high_percentage(current_price, high_4w)
    company.high_weekly = near_high_percentage(current_price, high_7d)

    # ---- Full-history monthly / weekly aggregate statistics ----
    low_m_avg, low_m_min, low_m_max = monthly_low_statistics(bars)
    company.low_monthly_avg = low_m_avg
    company.low_monthly_min = low_m_min
    company.low_monthly_max = low_m_max

    low_w_avg, low_w_min, low_w_max = weekly_low_statistics(bars)
    company.low_weekly_avg = low_w_avg
    company.low_weekly_min = low_w_min
    company.low_weekly_max = low_w_max

    high_m_avg, high_m_min, high_m_max = monthly_high_statistics(bars)
    company.high_monthly_avg = high_m_avg
    company.high_monthly_min = high_m_min
    company.high_monthly_max = high_m_max

    high_w_avg, high_w_min, high_w_max = weekly_high_statistics(bars)
    company.high_weekly_avg = high_w_avg
    company.high_weekly_min = high_w_min
    company.high_weekly_max = high_w_max


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_refresh() -> Dict[str, int]:
    """
    Run the full refresh over every Company and persist results with a
    single bulk_update(). Returns a small summary dict.
    """
    latest_date = get_latest_trading_date()
    if latest_date is None:
        logger.warning("refresh_analysis: DailyData is empty, nothing to do.")
        return {"total": 0, "succeeded": 0, "failed": 0}

    # One query for all historical data, grouped in memory by company_id.
    daily_by_company = load_all_daily_data()

    companies = list(Company.objects.only("id", *ANALYSIS_FIELDS))
    total_len = len(companies)
    succeeded = 0
    failed = 0
    count = 1
    for company in companies:
        print(company.id,count,total_len)
        count+=1
        bars = daily_by_company.get(company.id)

        try:
            if not bars:
                # No history at all for this company.
                reset_analysis(company)
                continue

            analyze_company(company, bars, latest_date)
            succeeded += 1

        except Exception:
            # Never let one bad company stop the batch.
            logger.exception(
                "refresh_analysis: failed to analyze company id=%s code=%s",
                company.id, getattr(company, "code", "?"),
            )
            reset_analysis(company)
            failed += 1
        
    # Single bulk_update for the entire batch.
    with transaction.atomic():
        Company.objects.bulk_update(companies, ANALYSIS_FIELDS, batch_size=500)

    return {"total": len(companies), "succeeded": succeeded, "failed": failed}


# =============================================================================
# DJANGO VIEW ENTRY POINT
# =============================================================================

def refresh_analysis(request):
    """
    View wrapper: triggers the full analysis refresh and returns a JSON
    summary. Wire this up in urls.py exactly as the old placeholder was.
    """
    summary = run_refresh()
    return JsonResponse({"status": "ok", **summary})
import re
import threading
import traceback
import uuid
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
from django.conf import settings
from django.db import connection
from django.http import FileResponse, Http404
from django.shortcuts import render, get_object_or_404
from django.db.models import Count
from .models import Company, DailyData, FiveMinuteData
from collections import defaultdict

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.db.models import Max
from datetime import date, timedelta, datetime,time
import yfinance as yf
from django.core.cache import cache
from .models import Company, DailyData
import warnings
warnings.filterwarnings("ignore")
def home(request):
    # ---- Sector data (keep all, small count) ----
    sectors = (
        Company.objects
        .values('sector')
        .annotate(total=Count('id'))
        .order_by('-total')
    )

    # ---- Industry data (Top 20 + Other) ----
    industries_qs = (
        Company.objects
        .values('industry')
        .annotate(total=Count('id'))
        .order_by('-total')
    )

    TOP_N = HOME_TOP_ALL
    top_industries = list(industries_qs[:TOP_N])

    other_total = sum(i['total'] for i in industries_qs[TOP_N:])
    if other_total:
        top_industries.append({
            'industry': 'Other',
            'total': other_total
        })

    context = {
        'sectors': sectors,
        'industries': top_industries
    }

    return render(request, 'market/home.html', context)
def industries_by_sector(request):
    sector = request.GET.get('sector')

    qs = (
        Company.objects
        .filter(sector=sector)
        .values('industry')
        .annotate(total=Count('id'))
        .order_by('-total')
    )

    TOP_N = HOME_TOP_SECTOR
    qs_list = list(qs[:TOP_N])

    other_total = sum(i['total'] for i in qs[TOP_N:])
    if other_total:
        qs_list.append({
            'industry': 'Other',
            'total': other_total
        })

    return JsonResponse(qs_list, safe=False)


HOME_TOP_ALL = 50        # industries shown on the home chart before "Other" (all sectors)
HOME_TOP_SECTOR = 20     # ... and inside one sector


def companies_by_industry(request):
    """
    Companies behind one bar of the home page chart.
    ?industry=<name>[&sector=<name>]   industry "Other" = every industry outside the chart's top N.
    """
    industry = request.GET.get("industry", "")
    sector = request.GET.get("sector") or None
    qs = Company.objects.all()
    if sector:
        qs = qs.filter(sector=sector)
    if industry == "Other":
        top_n = HOME_TOP_SECTOR if sector else HOME_TOP_ALL
        shown = [r["industry"] for r in
                 qs.values("industry").annotate(total=Count("id")).order_by("-total")[:top_n]]
        qs = qs.exclude(industry__in=shown)
    else:
        qs = qs.filter(industry=industry)
    rows = list(qs.order_by("code").values("code", "name", "sector", "industry", "latest_price"))
    return JsonResponse({"industry": industry, "sector": sector, "count": len(rows), "companies": rows})


def daily_data_view(request):
    company_code = request.GET.get('company')
    company = None
    data = None
    companies = Company.objects.filter(exchange="NSE").order_by('code')
    if company_code:
        company = get_object_or_404(Company, code=company_code, exchange="NSE")
        data = DailyData.objects.filter(company=company).order_by('-date')[:200]
    return render(request, 'market/daily_data.html', {'company': company, 'data': data, 'companies': companies})


def five_min_view(request):
    company_code = request.GET.get('company')
    company = None
    data = None
    if company_code:
        company = get_object_or_404(Company, code=company_code)
        data = FiveMinuteData.objects.filter(company=company).order_by('-timestamp')[:200]
    return render(request, 'market/five_min.html', {'company': company, 'data': data, 'companies': Company.objects.filter(exchange="NSE").order_by('code')})


def live_view(request):
    company_code = request.GET.get('company')
    company = None
    live = None
    if company_code:
        company = get_object_or_404(Company, code=company_code)
        live = {'note': 'Live data not implemented; this is a placeholder'}
    return render(request, 'market/live.html', {'company': company, 'live': live, 'companies': Company.objects.all()})


# views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.utils import timezone
from django.db.models import Min, Max, Avg
from datetime import timedelta
from .models import Company, DailyData

def analysis_view(request):
    """
    Analysis page with:
    - Company contains filter
    - Range filters for all numeric columns
    - Sorting
    - Pagination
    """

    page_number = int(request.GET.get("page", 1))
    mode = request.GET.get("mode", "low")
    sort = request.GET.get("sort", "low_weekly")
    order = request.GET.get("order", "asc")

    per_page = 50

    sortable_fields = [
        "code",

        "low_yearly",
        "low_monthly",
        "low_weekly",
        "low_monthly_min",
        "low_monthly_max",
        "low_monthly_avg",
        "low_weekly_min",
        "low_weekly_max",
        "low_weekly_avg",

        "high_yearly",
        "high_monthly",
        "high_weekly",
        "high_monthly_min",
        "high_monthly_max",
        "high_monthly_avg",
        "high_weekly_min",
        "high_weekly_max",
        "high_weekly_avg",
    ]

    if sort not in sortable_fields:
        sort = "low_weekly"

    ordering = f"-{sort}" if order == "desc" else sort

    companies = Company.objects.filter(
        exchange="NSE",
        low_yearly__isnull=False
    )

    # -------------------------
    # Company Filter
    # -------------------------

    company = request.GET.get("company", "").strip()

    if company:
        companies = companies.filter(code__icontains=company)

    # -------------------------
    # Numeric Range Filters
    # -------------------------

    # (field_name, human readable label)
    low_field_defs = [
        ("low_yearly", "Yearly"),
        ("low_monthly", "Monthly"),
        ("low_weekly", "Weekly"),
        ("low_monthly_min", "Monthly Min"),
        ("low_monthly_max", "Monthly Max"),
        ("low_monthly_avg", "Monthly Avg"),
        ("low_weekly_min", "Weekly Min"),
        ("low_weekly_max", "Weekly Max"),
        ("low_weekly_avg", "Weekly Avg"),
    ]

    high_field_defs = [
        ("high_yearly", "Yearly"),
        ("high_monthly", "Monthly"),
        ("high_weekly", "Weekly"),
        ("high_monthly_min", "Monthly Min"),
        ("high_monthly_max", "Monthly Max"),
        ("high_monthly_avg", "Monthly Avg"),
        ("high_weekly_min", "Weekly Min"),
        ("high_weekly_max", "Weekly Max"),
        ("high_weekly_avg", "Weekly Avg"),
    ]

    numeric_fields = [f for f, _ in low_field_defs] + [f for f, _ in high_field_defs]

    active_filter_count = 1 if company else 0

    for field in numeric_fields:

        min_value = request.GET.get(f"{field}_from", "").strip()
        max_value = request.GET.get(f"{field}_to", "").strip()

        if min_value:
            try:
                companies = companies.filter(
                    **{f"{field}__gte": float(min_value)}
                )
                active_filter_count += 1
            except ValueError:
                pass

        if max_value:
            try:
                companies = companies.filter(
                    **{f"{field}__lte": float(max_value)}
                )
                active_filter_count += 1
            except ValueError:
                pass

    # Build the field definitions the template needs, pre-loaded with
    # whatever from/to values are currently in the querystring.
    def build_filter_fields(defs):
        result = []
        for field, label in defs:
            result.append({
                "field": field,
                "label": label,
                "from_value": request.GET.get(f"{field}_from", ""),
                "to_value": request.GET.get(f"{field}_to", ""),
            })
        return result

    low_filter_fields = build_filter_fields(low_field_defs)
    high_filter_fields = build_filter_fields(high_field_defs)

    # -------------------------
    # Sorting
    # -------------------------

    companies = companies.order_by(ordering)

    # -------------------------
    # Pagination
    # -------------------------

    paginator = Paginator(companies, per_page)
    page_obj = paginator.get_page(page_number)

    context = {
        "companies": page_obj,
        "page_obj": page_obj,
        "mode": mode,
        "current_sort": request.GET.get("sort", ""),
        "current_order": request.GET.get("order", "asc"),
        "filters": request.GET,
        "low_filter_fields": low_filter_fields,
        "high_filter_fields": high_filter_fields,
        "active_filter_count": active_filter_count,
    }

    return render(request, "market/analysis.html", context)


def daily_data_page(request):
    return render(request, "market/daily_data_run.html")

def _daily_worker(task_id):
    companies = Company.objects.filter(exchange="NSE")
    total_companies = companies.count()
    _progress(task_id, total=total_companies)

    inserted_rows = 0
    errors = 0
    no_data = 0

    # Get the latest NSE trading day.
    # Saturday (5) and Sunday (6) -> Friday
    today = date.today()

    if today.weekday() == 5:       # Saturday
        market_date = today - timedelta(days=1)
    elif today.weekday() == 6:     # Sunday
        market_date = today - timedelta(days=2)
    else:
        market_date = today

    weekday = {
        "Friday": 6,
        "Monday": 2,
        "Tuesday": 3,
        "Wednesday": 4,
        "Thursday": 5,
        "Saturday": 7,
        "Sunday": 1,
    }

    for index, company in enumerate(companies, start=1):
        _progress(task_id, current=company.code)
        try:
            # print(f"Processing {company.code}")

            latest = DailyData.objects.filter(
                company=company
            ).aggregate(
                max_date=Max("date")
            )["max_date"]

            # Compare against the latest market/trading day,
            # NOT date.today().
            if latest == market_date:
                # print(
                #     f"Data for {company.code} is already up to date "
                #     f"for {market_date}."
                # )

                _progress(task_id, done=index, rows=inserted_rows, current=company.code)
                continue

            start_date = (
                latest + timedelta(days=1)
                if latest
                else date(2000, 1, 1)
            )

            # yfinance end date is exclusive, so add one day.
            end_date = market_date + timedelta(days=1)

            df = yf.download(
                company.code,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if df.empty:
                # print(f"No new data for {company.code}.")
                no_data += _record_no_data(task_id, company.code)
                _progress(task_id, done=index, rows=inserted_rows, current=company.code, no_data=no_data)
                continue
            print(f"Processing {company.code}")
            df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.date

            # Calculations
            df["High_Low"] = df["High"] - df["Low"]
            df["Close_Open"] = df["Close"] - df["Open"]
            df["High_Open"] = df["High"] - df["Open"]
            df["Open_Low"] = df["Open"] - df["Low"]
            df["Status"] = df["Close_Open"].apply(
                lambda x: 1 if x > 0 else 0
            )

            df["Weekday"] = (
                pd.to_datetime(df["Date"])
                .dt.day_name()
                .map(weekday)
            )

            df["WeekNumber"] = (
                pd.to_datetime(df["Date"])
                .dt.isocalendar()
                .week
            )

            # Get last DB close for PreviousClose
            last_record = (
                DailyData.objects
                .filter(company=company)
                .order_by("-date")
                .first()
            )

            last_close = (
                last_record.close
                if last_record
                else None
            )

            df["PreviousClose"] = df["Close"].shift(1)

            if not df.empty:
                if last_close is not None:
                    df.loc[df.index[0], "PreviousClose"] = last_close
                else:
                    df.loc[df.index[0], "PreviousClose"] = 0

            daily_objects = []

            for row in df.itertuples(index=False):
                obj = DailyData(
                    date=row.Date,
                    company=company,
                    previous=row.PreviousClose,
                    open=row.Open,
                    low=row.Low,
                    high=row.High,
                    close=row.Close,
                    high_low=row.High_Low,
                    close_open=row.Close_Open,
                    high_open=row.High_Open,
                    open_low=row.Open_Low,
                    status=row.Status,
                    weekday=row.Weekday,
                    week_number=row.WeekNumber,
                    volume=row.Volume,
                )

                daily_objects.append(obj)
            for obj in daily_objects:
                for field in obj._meta.fields:
                    value = getattr(obj, field.name)

                    # if pd.isna(value):
                    #     print(f"NULL/NaN found: {field.name} = {value}")
            DailyData.objects.bulk_create(
                daily_objects,
                ignore_conflicts=True,
            )

            inserted_rows += len(daily_objects)

            _progress(task_id, done=index, rows=inserted_rows, current=company.code)
        except Exception as exc:                      # one bad ticker must not stop the run
            errors += 1
            _record_exception(task_id, company.code, exc)
            _progress(task_id, done=index, rows=inserted_rows, errors=errors)

    _progress(task_id, done=total_companies, rows=inserted_rows, status="completed", current="")

def _five_minute_worker(task_id):
    companies = Company.objects.filter(exchange="NSE")
    total_companies = companies.count()
    _progress(task_id, total=total_companies)
    inserted_rows = 0
    errors = 0
    no_data = 0

    weekday = {
        "Monday": 2,
        "Tuesday": 3,
        "Wednesday": 4,
        "Thursday": 5,
        "Friday": 6,
        "Saturday": 7,
        "Sunday": 1,
    }

    # Current time
    now = timezone.localtime()

    # ---------------------------------------------------------
    # Determine the latest NSE market time
    # ---------------------------------------------------------
    if now.weekday() == 5:  # Saturday
        # Friday 15:30
        market_date = now.date() - timedelta(days=1)
        latest_market_dt = timezone.make_aware(
            datetime.combine(market_date, time(15, 30)),
            timezone.get_current_timezone(),
        )

    elif now.weekday() == 6:  # Sunday
        # Friday 15:30
        market_date = now.date() - timedelta(days=2)
        latest_market_dt = timezone.make_aware(
            datetime.combine(market_date, time(15, 30)),
            timezone.get_current_timezone(),
        )

    else:
        # Weekday
        latest_market_dt = now

    print(f"Latest expected NSE market time: {latest_market_dt}")

    for index, company in enumerate(companies, start=1):
        _progress(task_id, done=index - 1, rows=inserted_rows, current=company.code)
        try:
            print(f"Processing 5m data for {company.code}")

            latest_ts = (
                FiveMinuteData.objects
                .filter(company=company)
                .aggregate(max_ts=Max("timestamp"))
            )["max_ts"]

            # ---------------------------------------------------------
            # If database already has data up to the latest market time,
            # don't call Yahoo again.
            # ---------------------------------------------------------
            if latest_ts:
                # Make sure both datetimes are timezone-aware
                if timezone.is_naive(latest_ts):
                    latest_ts = timezone.make_aware(
                        latest_ts,
                        timezone.get_current_timezone(),
                    )

                if latest_ts >= latest_market_dt:
                    print(
                        f"{company.code} is already up to date. "
                        f"Latest DB timestamp: {latest_ts}"
                    )
                    continue

            # ---------------------------------------------------------
            # Yahoo Finance 5-minute data limitation
            # Approximately 60 days.
            # ---------------------------------------------------------
            yahoo_min_date = now - timedelta(days=59)

            if latest_ts and latest_ts > yahoo_min_date:
                start_dt = latest_ts + timedelta(minutes=5)
            else:
                start_dt = yahoo_min_date

            # Don't request beyond the latest available market time.
            end_dt = latest_market_dt + timedelta(minutes=5)

            print(
                f"{company.code}: downloading from "
                f"{start_dt} to {end_dt}"
            )

            df = yf.download(
                company.code,
                start=start_dt,
                end=end_dt,
                interval="5m",
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if df.empty:
                print(f"No new 5m data for {company.code}")
                no_data += _record_no_data(task_id, company.code)
                _progress(task_id, no_data=no_data)
                continue

            # Handle MultiIndex columns returned by yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()

            # yfinance may return Datetime or datetime depending on version
            if "Datetime" in df.columns:
                df.rename(
                    columns={"Datetime": "Timestamp"},
                    inplace=True,
                )

            df["Timestamp"] = pd.to_datetime(
                df["Timestamp"]
            )

            # ---------------------------------------------------------
            # Remove data beyond the latest expected market time
            # ---------------------------------------------------------
            df = df[
                df["Timestamp"] <= latest_market_dt
            ].copy()

            if df.empty:
                print(
                    f"No new market-time data for {company.code}"
                )
                continue

            # ---------------------------------------------------------
            # Calculations
            # ---------------------------------------------------------
            df["High_Low"] = (
                df["High"] - df["Low"]
            )

            df["Close_Open"] = (
                df["Close"] - df["Open"]
            )

            df["High_Open"] = (
                df["High"] - df["Open"]
            )

            df["Open_Low"] = (
                df["Open"] - df["Low"]
            )

            df["Status"] = df["Close_Open"].apply(
                lambda x: 1 if x > 0 else 0
            )

            df["Weekday"] = (
                df["Timestamp"]
                .dt.day_name()
                .map(weekday)
            )

            df["WeekNumber"] = (
                df["Timestamp"]
                .dt.isocalendar()
                .week
            )

            # ---------------------------------------------------------
            # Previous close logic
            # ---------------------------------------------------------
            last_record = (
                FiveMinuteData.objects
                .filter(company=company)
                .order_by("-timestamp")
                .first()
            )

            last_close = (
                last_record.close
                if last_record
                else None
            )

            df["PreviousClose"] = (
                df["Close"].shift(1)
            )

            if not df.empty:
                if last_close is not None:
                    df.loc[
                        df.index[0],
                        "PreviousClose"
                    ] = last_close
                else:
                    df.loc[
                        df.index[0],
                        "PreviousClose"
                    ] = 0

            # ---------------------------------------------------------
            # Create objects
            # ---------------------------------------------------------
            objects = []

            for row in df.itertuples(index=False):
                objects.append(
                    FiveMinuteData(
                        timestamp=row.Timestamp,
                        company=company,
                        previous=row.PreviousClose,
                        open=row.Open,
                        low=row.Low,
                        high=row.High,
                        close=row.Close,
                        high_low=row.High_Low,
                        close_open=row.Close_Open,
                        high_open=row.High_Open,
                        open_low=row.Open_Low,
                        status=row.Status,
                        weekday=row.Weekday,
                        week_number=row.WeekNumber,
                        volume=row.Volume,
                    )
                )

            FiveMinuteData.objects.bulk_create(
                objects,
                ignore_conflicts=True,
            )

            inserted_rows += len(objects)

            print(
                f"Inserted {len(objects)} rows for {company.code}"
            )
        except Exception as exc:                      # one bad ticker must not stop the run
            errors += 1
            _record_exception(task_id, company.code, exc)
            _progress(task_id, done=index, rows=inserted_rows, errors=errors)

    # return JsonResponse({
    #     "status": "success",
    #     "rows_inserted": inserted_rows,
    # })

    _progress(task_id, done=total_companies, rows=inserted_rows, status="completed", current="")


# ---------------------------------------------------------------------------
# Background fetch jobs.  The request only starts the job and returns its
# task_id at once; the page polls /progress/<task_id>/ while it runs.
# (Progress lives in the cache: with LocMemCache that means one server
# process, which is what runserver uses.)
#
# Strictly sequential: yfinance fails when called in parallel, so only ONE
# fetch job (Daily or 5 Minute) may run at a time, and it downloads one
# company after another.
# ---------------------------------------------------------------------------
FETCH_JOB_KEY = "fetch_job:current"          # task_id of the job that is running, if any
_FETCH_START_LOCK = threading.Lock()         # two clicks at once must not start two jobs


FETCH_ERROR_DIR = Path(settings.BASE_DIR) / "logs" / "fetch_errors"
FETCH_ERROR_NAME = re.compile(r"^[\w-]+\.log$")


def _progress(task_id, **fields):
    data = cache.get(task_id) or {"total": 0, "done": 0, "rows": 0, "errors": 0, "no_data": 0,
                                  "status": "running", "current": ""}
    data.update(fields)
    cache.set(task_id, data, 24 * 3600)


def _log_symbol_problem(task_id, code, kind, message, tb=None):
    """
    Append one symbol's problem to this run's log file, logs/fetch_errors/<job>_<time>.log.
    The file is only created when the first problem happens.  `kind` is ERROR (an exception)
    or NO DATA (Yahoo returned nothing and said why, e.g. "possibly delisted").
    """
    data = cache.get(task_id) or {}
    name = data.get("error_log")
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    path_ = FETCH_ERROR_DIR / (name or "")
    if not name:
        name = f"{data.get('kind', 'fetch')}_{now:%Y%m%d_%H%M%S}.log"
        path_ = FETCH_ERROR_DIR / name
        FETCH_ERROR_DIR.mkdir(parents=True, exist_ok=True)
        with open(path_, "w", encoding="utf-8") as fh:
            fh.write(f"# {data.get('kind', 'fetch')} fetch started {data.get('started', '')} - task {task_id}\n")
            fh.write("# time (IST) | symbol | type | message\n")
        _progress(task_id, error_log=name)
    with open(path_, "a", encoding="utf-8") as fh:
        fh.write(f"{now:%Y-%m-%d %H:%M:%S} | {code} | {kind} | {message}\n")
        if tb:
            fh.write("".join("    " + line + "\n" for line in tb.rstrip().splitlines()))


def _yahoo_error(code):
    """The reason Yahoo gave for an empty yf.download of `code` (yfinance keeps it in shared._ERRORS)."""
    errors = getattr(getattr(yf, "shared", None), "_ERRORS", None) or {}
    return errors.get(code)


def _record_no_data(task_id, code):
    """Log an empty download only when Yahoo said why (delisted / bad symbol / no data)."""
    reason = _yahoo_error(code)
    if not reason:
        return 0
    _log_symbol_problem(task_id, code, "NO DATA", reason)
    return 1


def _record_exception(task_id, code, exc):
    print(f"Fetch failed for {code}: {exc}")
    _log_symbol_problem(task_id, code, "ERROR", f"{type(exc).__name__}: {exc}", traceback.format_exc())


def fetch_error_log(request, name):
    """Download a fetch run's symbol error log (logs/fetch_errors/<name>)."""
    if not FETCH_ERROR_NAME.match(name) or not (FETCH_ERROR_DIR / name).is_file():
        raise Http404("No such log")
    return FileResponse(open(FETCH_ERROR_DIR / name, "rb"), content_type="text/plain; charset=utf-8",
                        as_attachment=request.GET.get("download") == "1", filename=name)


def _run_job(kind, task_id, worker):
    try:
        worker(task_id)
    except Exception as exc:
        _log_symbol_problem(task_id, "-", "JOB FAILED", f"{type(exc).__name__}: {exc}", traceback.format_exc())
        _progress(task_id, status="failed", error=str(exc))
    finally:
        cache.delete(FETCH_JOB_KEY)
        connection.close()                  # this thread's DB connection


def _start_job(kind, worker):
    """
    Start `worker` in a background thread - unless ANY fetch job is still running (Daily or
    5 Minute), in which case nothing new starts and the running job's task_id is returned.
    """
    with _FETCH_START_LOCK:
        running = cache.get(FETCH_JOB_KEY)
        state = cache.get(running) if running else None
        if state and state.get("status") == "running":
            return JsonResponse({"status": "running", "task_id": running, "already_running": True,
                                 "running_kind": state.get("kind")})
        task_id = str(uuid.uuid4())
        _progress(task_id, kind=kind, started=timezone.now().isoformat())
        cache.set(FETCH_JOB_KEY, task_id, 24 * 3600)
        threading.Thread(target=_run_job, args=(kind, task_id, worker), daemon=True).start()
    return JsonResponse({"status": "started", "task_id": task_id})


@require_POST
def fetch_daily_data(request):
    return _start_job("daily", _daily_worker)


@require_POST
def fetch_five_minute_data(request):
    return _start_job("5min", _five_minute_worker)
def fetch_progress(request, task_id):
    data = cache.get(task_id)
    if not data:
        return JsonResponse({"status": "not_found"})
    return JsonResponse(data)
@require_POST
def delete_data_by_date(request):
    date_str = request.POST.get("date")
    interval = request.POST.get("interval")

    if not date_str or not interval:
        return JsonResponse({"error": "Date and interval are required"}, status=400)

    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return JsonResponse({"error": "Invalid date format (YYYY-MM-DD)"}, status=400)

    if interval == "daily":
        deleted_count, _ = DailyData.objects.filter(date=target_date).delete()

    elif interval == "5min":
        deleted_count, _ = FiveMinuteData.objects.filter(
            timestamp__date=target_date
        ).delete()
    else:
        return JsonResponse({"error": "Invalid interval"}, status=400)

    return JsonResponse({
        "status": "success",
        "deleted_rows": deleted_count,
        "date": date_str,
        "interval": interval
    })

def retrive_5min(request, code=None):
    records = FiveMinuteData.objects.all()

    if code:
        records = records.filter(company__code=code)

    records = records.values(
        'id',
        'timestamp',
        'company_id',
        'company__code',
        'previous',
        'open',
        'low',
        'high',
        'close',
        'high_low',
        'close_open',
        'high_open',
        'open_low',
        'status',
        'weekday',
        'week_number',
        'volume'
    )

    return JsonResponse(list(records), safe=False)

def retrive_daily(request, code=None):
    records = DailyData.objects.all()

    if code:
        records = records.filter(company__code=code)

    records = records.values(
        'id',
        'date',
        'company_id',
        'company__code',
        'previous',
        'open',
        'low',
        'high',
        'close',
        'high_low',
        'close_open',
        'high_open',
        'open_low',
        'status',
        'weekday',
        'week_number',
        'volume'
    )

    return JsonResponse(list(records), safe=False)

from .analysis_engine import refresh_analysis, refresh_analysis_progress

import requests
def live_snapshot(tolerant=False):
    """Live page data: NSE top gainers / losers, OI spurts and the Chartink PDH / PDL scans.
    tolerant=True returns an empty list for a source that fails instead of raising."""
    gain_loss_url = "https://www.nseindia.com/api/live-analysis-variations"
    open_intrest_url = "https://www.nseindia.com/api/live-analysis-oi-spurts-underlyings"
    chartlink = "https://chartink.com/screener/pdl-breakdown-stocks"
    pdl_pdh_url = "https://chartink.com/screener/process"

    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9,ta;q=0.8",
        "DNT": "1",
        "Priority": "u=1, i",
        "Referer": "https://www.nseindia.com/market-data/top-gainers-losers",
        "Sec-CH-UA": '"Not;A=Brand";v="8", "Chromium";v="150", "Google Chrome";v="150"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/150.0.0.0 Safari/537.36"
        ),
    }
    def top_stocks(index):
        pg = requests.get(
            gain_loss_url,
            headers=headers,
            params={"index": index},
            timeout=30
        )
        response = pg.json()
        l = []
        exist = []
        for i in response:
            if 'data' in response[i]:
                for j in response[i]['data']:
                    if j["symbol"] in exist:
                        continue
                    l.append(j)
                    exist.append(j['symbol'])

        return l

    def fetch_open_intrest():
        pg = requests.get(open_intrest_url,headers=headers,timeout=30)
        response = pg.json()['data']
        return response

    
    pdl_payload = {
        "scan_clause": "( {33489} (  [0] 5 minute close <  1 day ago low and  [0] 5 minute close <  2 days ago low and  [0] 5 minute close <  daily \"( (1 candle ago high + 1 candle ago low + 1 candle ago close / 3 ) * 2 - 1 candle ago high)\" and  [0] 5 minute close <  [0] 5 minute vwap and  [0] 5 minute close >  400 ) )",
        "debug_clause": "groupcount( 1 where  [0] 5 minute close <  1 day ago low),groupcount( 1 where  [0] 5 minute close <  2 days ago low),groupcount( 1 where  [0] 5 minute close <  daily \"( (1 candle ago high + 1 candle ago low + 1 candle ago close / 3 ) * 2 - 1 candle ago high)\"),groupcount( 1 where  [0] 5 minute close <  [0] 5 minute vwap),groupcount( 1 where  [0] 5 minute close >  400)",
        "column_clause": " Daily Close as 'scan-column-default-close',  Daily \"close - 1 candle ago close / 1 candle ago close * 100\" as 'scan-column-default-percent-change', filternumber( daily close >  1 day ago close,1) as 'default-percent-change-conditional-filters-color',  Daily Volume as 'scan-column-default-volume'"
    }
    pdh_payload = {
        "scan_clause": "( {33489} (  [0] 5 minute close >  1 day ago high and  [0] 5 minute close >  2 days ago high and  [0] 5 minute close >  daily \"( (1 candle ago high + 1 candle ago low + 1 candle ago close / 3 ) * 2 - 1 candle ago low )\" and  [0] 5 minute close >  [0] 5 minute vwap and  [0] 5 minute close >  400 ) )",
        "debug_clause": "groupcount( 1 where  [0] 5 minute close >  1 day ago high),groupcount( 1 where  [0] 5 minute close >  2 days ago high),groupcount( 1 where  [0] 5 minute close >  daily \"( (1 candle ago high + 1 candle ago low + 1 candle ago close / 3 ) * 2 - 1 candle ago low )\"),groupcount( 1 where  [0] 5 minute close >  [0] 5 minute vwap),groupcount( 1 where  [0] 5 minute close >  400)",
        "column_clause": " Daily Close as 'scan-column-default-close',  Daily \"close - 1 candle ago close / 1 candle ago close * 100\" as 'scan-column-default-percent-change', filternumber( daily close >  1 day ago close,1) as 'default-percent-change-conditional-filters-color',  Daily Volume as 'scan-column-default-volume'"
    }
    from urllib.parse import unquote
    def fetch_pdh_pdl(payload):
        session = requests.Session()
        pg = session.get(chartlink)
        xsrf = dict(pg.cookies)["XSRF-TOKEN"]
        
        chartlink_headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9,ta;q=0.8",
            "Content-Type": "application/json",
            "DNT": "1",
            "Origin": "https://chartink.com",
            "Priority": "u=1, i",
            "Referer": "https://chartink.com/screener/pdl-breakdown-stocks",
            "Sec-CH-UA": '"Not;A=Brand";v="8", "Chromium";v="150", "Google Chrome";v="150"',
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "X-XSRF-TOKEN": unquote(xsrf)
        }
        pg = session.post(pdl_pdh_url,headers = chartlink_headers, json = payload, timeout = 30)
        data = pg.json()['data']
        return data
    def get(fn, *args):
        if not tolerant:
            return fn(*args)
        try:
            return fn(*args)
        except Exception:
            return []
    gainers = get(top_stocks, "gainers")
    loosers = get(top_stocks, "loosers")
    open_intrest = get(fetch_open_intrest)
    pdh_data = get(fetch_pdh_pdl, pdh_payload)
    pdl_data = get(fetch_pdh_pdl, pdl_payload)
    return_data = {
        "gainers":gainers,
        "loosers":loosers,
        "open_intrest":open_intrest,
        "pdh_data":pdh_data,
        "pdl_data":pdl_data
        }
    return return_data


def fetch_live(request):
    return JsonResponse(live_snapshot(), safe=False)


WEEKDAY_CHOICES = [
    ("", "All"),
    ("2", "Monday"),
    ("3", "Tuesday"),
    ("4", "Wednesday"),
    ("5", "Thursday"),
    ("6", "Friday"),
]
 
 
def chart_view(request):
    """Renders the filter form + chart shell. Data itself is fetched via AJAX."""
    companies = Company.objects.all().order_by("name")
    context = {
        "companies": companies,
        "weekday_choices": WEEKDAY_CHOICES,
    }
    return render(request, "market/chart.html", context)
 
 
def _get_previous_day_ohlc(company_id, current_date):
    """
    Previous trading day's O/H/L/C for the reference lines, sourced from
    DailyData (the source of truth for daily OHLC). Falls back to
    aggregating FiveMinuteData for that prior day if no DailyData row
    exists yet for it.
    """
    prev_daily = (
        DailyData.objects.filter(company_id=company_id, date__lt=current_date)
        .order_by("-date")
        .first()
    )
    if prev_daily:
        if all(
            getattr(prev_daily, f) is not None
            for f in ("open", "high", "low", "close")
        ):
            return {
                "date": prev_daily.date.isoformat(),
                "open": float(prev_daily.open),
                "high": float(prev_daily.high),
                "low": float(prev_daily.low),
                "close": float(prev_daily.close),
            }
 
    # Fallback: aggregate FiveMinuteData for the most recent day before
    # current_date that actually has candles.
    prev_row = (
        FiveMinuteData.objects.filter(company_id=company_id, timestamp__date__lt=current_date)
        .order_by("-timestamp")
        .first()
    )
    if not prev_row:
        return None
 
    prev_date = prev_row.timestamp.date()
    prev_candles = list(
        FiveMinuteData.objects.filter(company_id=company_id, timestamp__date=prev_date).order_by("timestamp")
    )
    if not prev_candles:
        return None
 
    highs = [float(c.high) for c in prev_candles if c.high is not None]
    lows = [float(c.low) for c in prev_candles if c.low is not None]
    if prev_candles[0].open is None or prev_candles[-1].close is None or not highs or not lows:
        return None
 
    return {
        "date": prev_date.isoformat(),
        "open": float(prev_candles[0].open),
        "high": max(highs),
        "low": min(lows),
        "close": float(prev_candles[-1].close),
    }
 
 
def chart_data(request):
    """
    JSON endpoint that returns candlestick data grouped by trading day,
    newest day first, paginated by number-of-days-per-page.
 
    Query params:
      company   (required) - Company.id
      weekday   (optional) - '2'..'6', blank/omitted = all weekdays
      page      (optional, default 1)
      page_size (optional, default 1) - how many trading days per page
    """
    company_id = request.GET.get("company")
    weekday = request.GET.get("weekday", "").strip()
    page_number = request.GET.get("page", 1)
    page_size = request.GET.get("page_size", 1)
 
    if not company_id:
        return JsonResponse({"error": "company is required"}, status=400)
 
    try:
        page_size = max(1, int(page_size))
    except (TypeError, ValueError):
        page_size = 1
 
    qs = (
        FiveMinuteData.objects.filter(company_id=company_id)
        .order_by("-timestamp")
    )
    if weekday:
        qs = qs.filter(weekday=weekday)
 
    # Group rows by calendar day (newest day first).
    days_map = defaultdict(list)
    day_order = []  # preserves first-seen (i.e. newest-first) order
    for row in qs.iterator():
        day_key = row.timestamp.date().isoformat()
        if day_key not in days_map:
            day_order.append(day_key)
        days_map[day_key].append(row)
 
    paginator = Paginator(day_order, page_size)
    page_obj = paginator.get_page(page_number)
 
    result_days = []
    for day_key in page_obj:
        # rows for a single day should render left-to-right chronologically
        rows = sorted(days_map[day_key], key=lambda r: r.timestamp)
        candles = []
        day_open = float(rows[0].open) if rows and rows[0].open is not None else None
        day_close = float(rows[-1].close) if rows and rows[-1].close is not None else None
 
        for row in rows:
            open_price = float(row.open) if row.open is not None else None
            close_price = float(row.close) if row.close is not None else None
            pct_change = None
            if open_price:
                pct_change = round(((close_price - open_price) / open_price) * 100, 2)
 
            candles.append(
                {
                    "x": row.timestamp.isoformat(),
                    "o": open_price,
                    "h": float(row.high) if row.high is not None else None,
                    "l": float(row.low) if row.low is not None else None,
                    "c": close_price,
                    "volume": row.volume,
                    "pct_change": pct_change,
                }
            )
 
        day_pct_change = None
        if day_open:
            day_pct_change = round(((day_close - day_open) / day_open) * 100, 2)
 
        current_date = date.fromisoformat(day_key)
        previous_day = _get_previous_day_ohlc(company_id, current_date)
 
        result_days.append(
            {
                "date": day_key,
                "weekday": rows[0].weekday if rows else None,
                "day_open": day_open,
                "day_close": day_close,
                "day_pct_change": day_pct_change,
                "candles": candles,
                "previous_day": previous_day,
            }
        )
 
    return JsonResponse(
        {
            "days": result_days,
            "page": page_obj.number,
            "num_pages": paginator.num_pages,
            "page_size": page_size,
            "has_next": page_obj.has_next(),
            "has_previous": page_obj.has_previous(),
            "total_days": paginator.count,
        }
    )
import uuid
import pandas as pd
from django.shortcuts import render, get_object_or_404
from django.db.models import Count
from .models import Company, DailyData, FiveMinuteData

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.db.models import Max
from datetime import date, timedelta, datetime
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

    TOP_N = 20
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

    TOP_N = 20
    qs_list = list(qs[:TOP_N])

    other_total = sum(i['total'] for i in qs[TOP_N:])
    if other_total:
        qs_list.append({
            'industry': 'Other',
            'total': other_total
        })

    return JsonResponse(qs_list, safe=False)

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
    # Placeholder: live data can be implemented with websockets or polling
    company_code = request.GET.get('company')
    company = None
    live = None
    if company_code:
        company = get_object_or_404(Company, code=company_code)
        live = {'note': 'Live data not implemented; this is a placeholder'}
    return render(request, 'market/live.html', {'company': company, 'live': live, 'companies': Company.objects.all()})


def analysis_view(request):
    # Placeholder for analysis page
    return render(request, 'market/analysis.html')
def daily_data_page(request):
    return render(request, "market/daily_data_run.html")

@require_POST
def fetch_daily_data(request):
    task_id = str(uuid.uuid4())
    
    companies = Company.objects.filter(exchange="NSE")
    cache.set(task_id, {"total": companies.count(), "done": 0, "rows": 0, "status": "running"}, None)
    inserted_rows = 0

    weekday = {
    'Friday': 6, 'Monday': 2, 'Tuesday': 3,
    'Wednesday': 4, 'Thursday': 5, 'Saturday': 7, 'Sunday': 1
    }

    for company in companies:
        print(f"Processing {company.code}")

        latest = DailyData.objects.filter(company=company).aggregate(
            max_date=Max("date")
        )["max_date"]
        if latest==date.today():
            print(f"Data for {company.code} is already up to date.")
            continue
        start_date = (latest + timedelta(days=1)) if latest else date(2000, 1, 1)
        end_date = date.today() + timedelta(days=1)

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
            continue
        df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])

        # Calculations
        df['High_Low'] = df['High'] - df['Low']
        df['Close_Open'] = df['Close'] - df['Open']
        df['High_Open'] = df['High'] - df['Open']
        df['Open_Low'] = df['Open'] - df['Low']
        df['Status'] = df['Close_Open'].apply(lambda x: 1 if x > 0 else 0)

        df['Weekday'] = df['Date'].dt.day_name().map(weekday)
        df['WeekNumber'] = df['Date'].dt.isocalendar().week

        # ✅ NEW: Use last DB close as PreviousClose
        last_record = DailyData.objects.filter(company=company).order_by('-date').first()
        last_close = last_record.close if last_record else None

        df['PreviousClose'] = df['Close'].shift(1)

        if last_close is not None:
            df['PreviousClose'].iloc[0] = last_close
        else:
            df['PreviousClose'].iloc[0] = 0

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

        DailyData.objects.bulk_create(daily_objects, ignore_conflicts=True)
        inserted_rows += len(daily_objects)
        cache.set(task_id, {
                    "total": companies.count(),
                    "done": companies.count(),
                    "rows": inserted_rows,
                    "status": "running"
                }, None)
    cache.set(task_id, {
                "total": companies.count(),
                "done": companies.count(),
                "rows": inserted_rows,
                "status": "completed"
            }, None)
    return JsonResponse({"status": "success", "rows_inserted": inserted_rows, "task_id": task_id})

@require_POST
def fetch_five_minute_data(request):
    companies = Company.objects.filter(exchange="NSE")
    inserted_rows = 0

    weekday = {
        'Monday': 2, 'Tuesday': 3, 'Wednesday': 4,
        'Thursday': 5, 'Friday': 6, 'Saturday': 7, 'Sunday': 1
    }

    for company in companies:
        print(f"Processing 5m data for {company.code}")

        latest_ts = FiveMinuteData.objects.filter(company=company).aggregate(
            max_ts=Max("timestamp")
        )["max_ts"]

        # Yahoo limit → max ~60 days for 5m
        if latest_ts:
            start_dt = latest_ts + timedelta(minutes=5)
        else:
            start_dt = datetime.now() - timedelta(days=59)

        end_dt = datetime.now()

        df = yf.download(
            company.code,
            start=start_dt,
            end=end_dt,
            interval="5m",
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            continue

        df.columns = df.columns.get_level_values(0)
        df = df.reset_index()

        df.rename(columns={"Datetime": "Timestamp"}, inplace=True)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        # Calculations
        df["High_Low"] = df["High"] - df["Low"]
        df["Close_Open"] = df["Close"] - df["Open"]
        df["High_Open"] = df["High"] - df["Open"]
        df["Open_Low"] = df["Open"] - df["Low"]
        df["Status"] = df["Close_Open"].apply(lambda x: 1 if x > 0 else 0)

        df["Weekday"] = df["Timestamp"].dt.day_name().map(weekday)
        df["WeekNumber"] = df["Timestamp"].dt.isocalendar().week

        # Previous close logic
        last_record = (
            FiveMinuteData.objects
            .filter(company=company)
            .order_by("-timestamp")
            .first()
        )
        last_close = last_record.close if last_record else None

        df["PreviousClose"] = df["Close"].shift(1)

        if last_close is not None:
            df.loc[df.index[0], "PreviousClose"] = last_close
        else:
            df.loc[df.index[0], "PreviousClose"] = 0

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

        FiveMinuteData.objects.bulk_create(objects, ignore_conflicts=True)
        inserted_rows += len(objects)

    return JsonResponse({
        "status": "success",
        "rows_inserted": inserted_rows
    })
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
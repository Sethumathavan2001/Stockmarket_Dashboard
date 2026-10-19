from django.urls import path
from . import index_membership, indices, views

app_name = 'market'

urlpatterns = [
    path('', views.home, name='home'),
    path('api/market/indices/', indices.indices_api, name='indices'),
    path('api/market/indices/<str:key>/chart/', indices.index_chart_api, name='index_chart'),
    path('api/market/stock-indices/<str:code>/', index_membership.stock_indices_api, name='stock_indices'),
    path('industries-by-sector/', views.industries_by_sector, name='industries_by_sector'),
    path('daily/', views.daily_data_view, name='daily_data'),
    path('5min/', views.five_min_view, name='five_min'),
    path('retrive_5min/', views.retrive_5min, name='retrive_5min_all'),
    path('retrive_5min/<str:code>',views.retrive_5min,name='retrive_5min'),
    path('retrive_daily/<str:code>',views.retrive_daily,name='retrive_daily'),
    path('live/', views.live_view, name='live'),
    path('fetch_live/', views.fetch_live, name='fetch_live'),
    path("analysis/", views.analysis_view, name="analysis"),
    path("analysis/refresh/", views.refresh_analysis, name="refresh_analysis"),
    path("analysis/refresh/progress/<str:task_id>/", views.refresh_analysis_progress, name="refresh_analysis_progress"),
    path("companies-by-industry/", views.companies_by_industry, name="companies_by_industry"),
    path("daily-data/", views.daily_data_page, name="daily_data_page"),   # UI
    path("api/fetch-daily-data/", views.fetch_daily_data, name="fetch_daily_data"),
    path("api/fetch-five-minute-data/", views.fetch_five_minute_data, name="fetch_five_minute_data"),
    path("progress/<str:task_id>/", views.fetch_progress, name="fetch_progress"),
    path("fetch-error-log/<str:name>/", views.fetch_error_log, name="fetch_error_log"),
    path("delete-data/", views.delete_data_by_date, name="delete_data"),
    path("chart/", views.chart_view, name="chart_view"),
    path("chart/data/", views.chart_data, name="chart_data"),
]

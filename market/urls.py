from django.urls import path
from . import views

app_name = 'market'

urlpatterns = [
    path('', views.home, name='home'),
    path('industries-by-sector/', views.industries_by_sector, name='industries_by_sector'),
    path('daily/', views.daily_data_view, name='daily_data'),
    path('5min/', views.five_min_view, name='five_min'),
    path('live/', views.live_view, name='live'),
    path('analysis/', views.analysis_view, name='analysis'),
    path("daily-data/", views.daily_data_page, name="daily_data_page"),   # UI
    path("api/fetch-daily-data/", views.fetch_daily_data, name="fetch_daily_data"),
    path("api/fetch-five-minute-data/", views.fetch_five_minute_data, name="fetch_five_minute_data"),
    path("progress/<str:task_id>/", views.fetch_progress, name="fetch_progress"),
    path("delete-data/", views.delete_data_by_date, name="delete_data"),

]

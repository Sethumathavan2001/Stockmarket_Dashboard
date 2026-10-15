from django.urls import path

from . import views

urlpatterns = [
    path("ai-prediction/", views.page, name="ai_prediction_page"),
    path("api/ai-prediction/analyze/", views.start_analysis, name="ai_prediction_analyze"),
    path("api/ai-prediction/picks/", views.start_picks, name="ai_prediction_picks"),
    path("api/ai-prediction/job/<str:job_id>/", views.job_status, name="ai_prediction_job"),
    path("api/ai-prediction/running/", views.running, name="ai_prediction_running"),
    path("api/ai-prediction/reports/", views.reports, name="ai_prediction_reports"),
    path("api/ai-prediction/report/<int:report_id>/", views.report_detail, name="ai_prediction_report"),
    path("api/ai-prediction/candles/<str:company_code>/", views.candles, name="ai_prediction_candles"),
]

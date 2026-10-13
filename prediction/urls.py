from django.urls import path

from . import views

# Included in the project urls.py without a prefix.
urlpatterns = [
    path("stock_prediction/", views.stock_prediction_view, name="stock_prediction_page"),
    path("api/predict/", views.models_overview, name="predict_all"),                 # every trained stock
    path("api/predict/<str:company_code>/", views.predict_company, name="predict_company"),
    path("api/train/<str:company_code>/", views.train_company, name="train_company"),
]

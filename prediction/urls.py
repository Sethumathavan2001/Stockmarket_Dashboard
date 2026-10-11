from django.urls import path

from . import views

urlpatterns = [
    # Frontend page
    path("stock_prediction/", views.stock_prediction_view, name="stock_prediction_page"),

    # REST API
    path("api/predict/", views.PredictAllCompaniesView.as_view(), name="predict_all"),
    path("api/predict/<str:company_code>/", views.PredictCompanyView.as_view(), name="predict_company"),
    path("api/train/<str:company_code>/", views.TrainCompanyView.as_view(), name="train_company"),
]

# In your project-level urls.py, include this WITHOUT a prefix so the routes
# land exactly on /stock_prediction/, /api/predict/, /api/predict/<code>/,
# and /api/train/<code>/ as specified:
#
#   from django.urls import path, include
#
#   urlpatterns = [
#       ...
#       path("", include("prediction.urls")),
#   ]

from django.urls import path

from . import views

urlpatterns = [
    path("candle-predictor/", views.candle_predictor_view, name="candle_predictor_page"),
    path("api/candle-predictor/<str:company_code>/", views.candle_prediction_data, name="candle_predictor_data"),
]

# In your project-level urls.py:
#
#   from django.urls import path, include
#
#   urlpatterns = [
#       ...
#       path("", include("patterns.urls")),
#   ]
#
# Page: /candle-predictor/    Data: /api/candle-predictor/<company_code>/

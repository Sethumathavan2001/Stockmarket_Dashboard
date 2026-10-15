from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('market.urls')),
    path("", include("prediction.urls")),
    path("", include("patterns.urls")),
    path("", include("paper_trading.urls")),
    path("", include("ai_prediction.urls")),
    path("", include("news.urls")),
]

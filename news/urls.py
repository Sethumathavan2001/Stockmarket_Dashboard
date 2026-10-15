from django.urls import path

from . import views

urlpatterns = [
    path("news/", views.page, name="news_page"),
    path("api/news/market/", views.market, name="news_market"),
    path("api/news/announcements/", views.announcements, name="news_announcements"),
    path("api/news/stock/<str:company_code>/", views.stock, name="news_stock"),
]

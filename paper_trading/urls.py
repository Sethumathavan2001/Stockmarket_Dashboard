from django.urls import path

from . import views

app_name = "paper_trading"

urlpatterns = [
    path("paper-trading/", views.paper_trading_page, name="page"),
    # Direct links: the same page, opened straight onto a stock (GET only)
    path("paper-trading/<str:company_code>/live/", views.paper_trading_page, {"mode": "live"}, name="page_live"),
    path("paper-trading/<str:company_code>/old/<str:date>/", views.paper_trading_page, {"mode": "old"}, name="page_old"),
    path("paper-trading/<str:company_code>/old/", views.paper_trading_page, {"mode": "old"}, name="page_old_query"),
    path("api/paper/account/", views.account_state, name="account"),
    path("api/paper/wallet/add/", views.wallet_add, name="wallet_add"),
    path("api/paper/date-range/<str:company_code>/", views.date_range, name="date_range"),
    path("api/paper/candles/<str:company_code>/", views.candles, name="candles"),
    path("api/paper/order/", views.place_order, name="order"),
    path("api/paper/bracket/", views.bracket_update, name="bracket"),
    path("api/paper/quotes/", views.quotes, name="quotes"),
    path("api/paper/squareoff/", views.squareoff, name="squareoff"),
]

# Project-level urls.py:
#
#   urlpatterns = [
#       ...
#       path("", include("paper_trading.urls")),
#   ]
#
# Page: /paper-trading/   Direct: /paper-trading/<code>/live/   /paper-trading/<code>/old/<YYYY-MM-DD>/

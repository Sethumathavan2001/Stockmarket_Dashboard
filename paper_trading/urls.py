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
    # Paper Trading 2: the same page with its own account (book 2)
    path("paper-trading-2/", views.paper_trading_page, {"book": 2}, name="page2"),
    path("paper-trading-2/<str:company_code>/live/", views.paper_trading_page, {"mode": "live", "book": 2}, name="page2_live"),
    path("paper-trading-2/<str:company_code>/old/<str:date>/", views.paper_trading_page, {"mode": "old", "book": 2}, name="page2_old"),
    path("paper-trading-2/<str:company_code>/old/", views.paper_trading_page, {"mode": "old", "book": 2}, name="page2_old_query"),
    path("api/paper/wallet/add/", views.wallet_add, name="wallet_add"),
    path("api/paper/wallet/withdraw/", views.wallet_withdraw, name="wallet_withdraw"),
    path("api/paper/limit/<int:limit_id>/cancel/", views.limit_cancel, name="limit_cancel"),
    path("api/paper/limit/<int:limit_id>/fill/", views.limit_fill, name="limit_fill"),
    path("api/paper/limit/expire/", views.limit_expire, name="limit_expire"),
    path("api/paper/date-range/<str:company_code>/", views.date_range, name="date_range"),
    path("api/paper/candles/<str:company_code>/", views.candles, name="candles"),
    path("api/paper/signals/<str:company_code>/", views.strategy_signals, name="signals"),
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

from django.urls import path

from . import views

app_name = "options"

urlpatterns = [
    path("options/", views.page, name="page"),
    path("api/options/underlyings/", views.underlyings, name="underlyings"),
    path("api/options/expiries/<str:symbol>/", views.expiries, name="expiries"),
    path("api/options/chain/<str:symbol>/", views.chain_view, name="chain"),
    path("api/options/state/", views.state, name="state"),
    path("api/options/preview/", views.preview, name="preview"),
    path("api/options/order/", views.place, name="place"),
    path("api/options/order/<int:order_id>/cancel/", views.cancel, name="cancel"),
    path("api/options/position/<int:position_id>/exit/", views.exit_position, name="exit"),
    path("api/options/position/<int:position_id>/levels/", views.levels, name="levels"),
    path("api/options/funds/", views.funds, name="funds"),
]

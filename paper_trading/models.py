from decimal import Decimal

from django.db import models


class PaperAccount(models.Model):
    """
    One virtual wallet. `owner_key` is "user:<id>" for logged-in users,
    "session:<key>" for anonymous browsers, or "shared" when
    settings.PAPER_TRADING_SHARED_ACCOUNT is True.
    """

    owner_key = models.CharField(max_length=120, unique=True)
    balance = models.DecimalField(max_digits=20, decimal_places=4, default=Decimal("0"))
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.owner_key} (₹{self.balance:.2f})"


class PaperPosition(models.Model):
    """
    Net position for one company inside one trading *context*:

      context == "live"        -> the live market
      context == "YYYY-MM-DD"  -> an "Old" (replay) session for that date

    quantity is signed: > 0 long, < 0 short.
    cost_basis is the money currently blocked from the wallet for this
    position (sum of entry notional of the open units). Average price is
    cost_basis / abs(quantity). Storing the basis (instead of a rounded
    average) keeps wallet accounting exact.
    """

    MODE_LIVE = "live"
    MODE_REPLAY = "replay"
    MODE_CHOICES = [(MODE_LIVE, "Live"), (MODE_REPLAY, "Old (replay)")]

    PRODUCT_INTRADAY = "INTRADAY"
    PRODUCT_DELIVERY = "DELIVERY"
    PRODUCT_CHOICES = [(PRODUCT_INTRADAY, "Intraday"), (PRODUCT_DELIVERY, "Delivery")]

    account = models.ForeignKey(PaperAccount, on_delete=models.CASCADE, related_name="positions")
    company = models.ForeignKey("market.Company", on_delete=models.CASCADE)
    mode = models.CharField(max_length=6, choices=MODE_CHOICES)
    # Where the position lives:
    #   live                      -> "live" (both products)
    #   old/replay, INTRADAY      -> "YYYY-MM-DD" (a practice day; closed at that day's end)
    #   old/replay, DELIVERY      -> "replay" (one shared book, so holdings carry from
    #                                 one replay date to the next)
    context = models.CharField(max_length=10)
    product = models.CharField(max_length=8, choices=PRODUCT_CHOICES, default=PRODUCT_INTRADAY)
    # Market time of the last order that opened / added to / flipped this position.
    # Used to settle stale live intraday positions and to stop a Delivery holding
    # being traded at a replay time earlier than its own last trade.
    last_entry_at = models.DateTimeField(null=True, blank=True)
    quantity = models.IntegerField(default=0)
    cost_basis = models.DecimalField(max_digits=20, decimal_places=4, default=Decimal("0"))
    # Optional exit levels for the open position (absolute prices). The UI also
    # accepts percentages, but they are resolved to prices against the position's
    # average entry price, so price is the single source of truth.
    stop_loss = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    target_price = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["account", "company", "context", "product"],
                name="uniq_paper_position_per_product",
            )
        ]

    @property
    def avg_price(self):
        if not self.quantity:
            return Decimal("0")
        return self.cost_basis / abs(self.quantity)

    def __str__(self):
        return f"{self.company_id} {self.quantity} [{self.context}]"


class PaperOrder(models.Model):
    """Immutable record of an executed order (a market order, or a limit order when it filled)."""

    SIDE_BUY = "BUY"
    SIDE_SELL = "SELL"
    SIDE_CHOICES = [(SIDE_BUY, "Buy"), (SIDE_SELL, "Sell")]

    account = models.ForeignKey(PaperAccount, on_delete=models.CASCADE, related_name="orders")
    company = models.ForeignKey("market.Company", on_delete=models.CASCADE)
    side = models.CharField(max_length=4, choices=SIDE_CHOICES)
    product = models.CharField(max_length=8, choices=PaperPosition.PRODUCT_CHOICES, default=PaperPosition.PRODUCT_INTRADAY)
    quantity = models.PositiveIntegerField()
    price = models.DecimalField(max_digits=20, decimal_places=4)
    mode = models.CharField(max_length=6, choices=PaperPosition.MODE_CHOICES)
    context = models.CharField(max_length=10)
    # Candle timestamp the order was filled against (simulated time in replay).
    market_time = models.DateTimeField(null=True, blank=True)
    realized_pnl = models.DecimalField(max_digits=20, decimal_places=4, default=Decimal("0"))
    balance_after = models.DecimalField(max_digits=20, decimal_places=4)
    ORDER_MARKET, ORDER_LIMIT = "MARKET", "LIMIT"
    order_type = models.CharField(max_length=6, choices=[(ORDER_MARKET, "Market"), (ORDER_LIMIT, "Limit")],
                                  default=ORDER_MARKET)
    # Why the order happened when it wasn't a manual click:
    # "stop_loss", "target" or "session_end" (auto square-off). Blank = manual.
    # (Order.context is always the day / "live" the order was placed in, whatever
    #  the product; Position.context is where the holding is kept.)
    reason = models.CharField(max_length=20, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-id"]
        indexes = [models.Index(fields=["account", "context", "-id"], name="paper_order_ctx_idx")]

    def __str__(self):
        return f"{self.side} {self.quantity} {self.company_id} @ {self.price}"


class PaperLimitOrder(models.Model):
    """
    A limit order. It fills only at its limit price or better:
    BUY when the price falls to the limit, SELL when it rises to it.
    A limit that can fill when placed fills at once (at the current price); otherwise it
    rests here as OPEN until it fills, is cancelled, or expires at the end of its day.
    """

    STATUS_OPEN, STATUS_FILLED = "OPEN", "FILLED"
    STATUS_CANCELLED, STATUS_REJECTED, STATUS_EXPIRED = "CANCELLED", "REJECTED", "EXPIRED"
    STATUS_CHOICES = [(s, s.title()) for s in (STATUS_OPEN, STATUS_FILLED, STATUS_CANCELLED, STATUS_REJECTED,
                                               STATUS_EXPIRED)]

    account = models.ForeignKey(PaperAccount, on_delete=models.CASCADE, related_name="limit_orders")
    company = models.ForeignKey("market.Company", on_delete=models.CASCADE)
    side = models.CharField(max_length=4, choices=PaperOrder.SIDE_CHOICES)
    product = models.CharField(max_length=8, choices=PaperPosition.PRODUCT_CHOICES, default=PaperPosition.PRODUCT_INTRADAY)
    quantity = models.PositiveIntegerField()
    limit_price = models.DecimalField(max_digits=20, decimal_places=4)
    mode = models.CharField(max_length=6, choices=PaperPosition.MODE_CHOICES)
    context = models.CharField(max_length=10)               # "live" or the replay date (the order's day)
    placed_at = models.DateTimeField()                       # market time it was placed (replay: simulated)
    levels = models.JSONField(default=dict, blank=True)      # stop loss / target to apply when it fills
    status = models.CharField(max_length=9, choices=STATUS_CHOICES, default=STATUS_OPEN)
    status_note = models.CharField(max_length=200, blank=True, default="")
    filled_order = models.OneToOneField(PaperOrder, null=True, blank=True, on_delete=models.SET_NULL,
                                        related_name="limit_order")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-id"]
        indexes = [models.Index(fields=["account", "status", "context"], name="paper_limit_open_idx")]

    def __str__(self):
        return f"LIMIT {self.side} {self.quantity} {self.company_id} @ {self.limit_price} [{self.status}]"

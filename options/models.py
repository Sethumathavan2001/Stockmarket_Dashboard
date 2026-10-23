from decimal import Decimal

from django.db import models

CE, PE = "CE", "PE"
OPTION_TYPES = [(CE, "Call"), (PE, "Put")]
NRML, MIS = "NRML", "MIS"
PRODUCTS = [(NRML, "NRML - carry to expiry"), (MIS, "MIS - intraday, squared off at 15:20")]


class OptionAccount(models.Model):
    """The option trading wallet. `balance` is cash; margin blocked by short positions sits on the positions."""

    owner_key = models.CharField(max_length=120, unique=True)
    balance = models.DecimalField(max_digits=20, decimal_places=4, default=Decimal("0"))
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.owner_key} (₹{self.balance:.2f})"


class OptionPosition(models.Model):
    """Net position in one contract + product. quantity > 0 long, < 0 short (units, a multiple of the lot)."""

    account = models.ForeignKey(OptionAccount, on_delete=models.CASCADE, related_name="positions")
    symbol = models.CharField(max_length=20)                 # underlying: NIFTY, BANKNIFTY, RELIANCE ...
    expiry = models.DateField()
    strike = models.DecimalField(max_digits=12, decimal_places=2)
    option_type = models.CharField(max_length=2, choices=OPTION_TYPES)
    product = models.CharField(max_length=4, choices=PRODUCTS, default=NRML)
    lot_size = models.PositiveIntegerField()
    quantity = models.IntegerField(default=0)
    avg_price = models.DecimalField(max_digits=14, decimal_places=4, default=Decimal("0"))
    blocked_margin = models.DecimalField(max_digits=20, decimal_places=4, default=Decimal("0"))
    realized_pnl = models.DecimalField(max_digits=20, decimal_places=4, default=Decimal("0"))
    stop_loss = models.DecimalField(max_digits=14, decimal_places=2, null=True, blank=True)     # on the premium
    target = models.DecimalField(max_digits=14, decimal_places=2, null=True, blank=True)
    opened_at = models.DateTimeField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["account", "symbol", "expiry", "strike", "option_type", "product"],
                                               name="uniq_option_position")]

    @property
    def name(self):
        return contract_name(self.symbol, self.expiry, self.strike, self.option_type)


class OptionOrder(models.Model):
    MARKET, LIMIT = "MARKET", "LIMIT"
    PENDING, COMPLETE, CANCELLED, REJECTED = "PENDING", "COMPLETE", "CANCELLED", "REJECTED"
    STATUSES = [(s, s.title()) for s in (PENDING, COMPLETE, CANCELLED, REJECTED)]

    account = models.ForeignKey(OptionAccount, on_delete=models.CASCADE, related_name="orders")
    symbol = models.CharField(max_length=20)
    expiry = models.DateField()
    strike = models.DecimalField(max_digits=12, decimal_places=2)
    option_type = models.CharField(max_length=2, choices=OPTION_TYPES)
    product = models.CharField(max_length=4, choices=PRODUCTS, default=NRML)
    side = models.CharField(max_length=4)                     # BUY / SELL
    quantity = models.PositiveIntegerField()
    lot_size = models.PositiveIntegerField()
    order_type = models.CharField(max_length=6, default=MARKET)
    limit_price = models.DecimalField(max_digits=14, decimal_places=2, null=True, blank=True)
    status = models.CharField(max_length=10, choices=STATUSES, default=PENDING)
    price = models.DecimalField(max_digits=14, decimal_places=4, null=True, blank=True)        # fill price
    underlying_price = models.DecimalField(max_digits=14, decimal_places=2, null=True, blank=True)
    realized_pnl = models.DecimalField(max_digits=20, decimal_places=4, default=Decimal("0"))
    charges = models.DecimalField(max_digits=14, decimal_places=4, default=Decimal("0"))
    charges_detail = models.JSONField(default=dict, blank=True)
    balance_after = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    reason = models.CharField(max_length=20, blank=True, default="")   # "", limit, stop_loss, target, squareoff, expiry
    message = models.CharField(max_length=300, blank=True, default="")
    stop_loss = models.DecimalField(max_digits=14, decimal_places=2, null=True, blank=True)    # levels to set on fill
    target = models.DecimalField(max_digits=14, decimal_places=2, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    filled_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        indexes = [models.Index(fields=["account", "status"])]

    @property
    def name(self):
        return contract_name(self.symbol, self.expiry, self.strike, self.option_type)


def contract_name(symbol, expiry, strike, option_type):
    """NIFTY 27 OCT 22600 CE (the way brokers show it)."""
    s = f"{Decimal(strike):f}".rstrip("0").rstrip(".") if "." in f"{Decimal(strike):f}" else f"{Decimal(strike):f}"
    return f"{symbol} {expiry.strftime('%d %b').upper()} {s} {option_type}"

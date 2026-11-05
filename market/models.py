from django.db import models


class Company(models.Model):
    name = models.CharField(max_length=255)
    code = models.CharField(max_length=50, unique=True)
    exchange = models.CharField(max_length=50, blank=True)
    sector = models.CharField(max_length=100, blank=True)
    industry = models.CharField(max_length=100, blank=True)
    latest_price = models.FloatField(null=True, blank=True)
    # 🔥 LOW ANALYSIS
    low_yearly = models.FloatField(null=True, blank=True)
    low_monthly = models.FloatField(null=True, blank=True)
    low_monthly_avg = models.FloatField(null=True, blank=True)
    low_monthly_min = models.FloatField(null=True, blank=True)
    low_monthly_max = models.FloatField(null=True, blank=True)
    low_weekly = models.FloatField(null=True, blank=True)
    low_weekly_avg = models.FloatField(null=True, blank=True)
    low_weekly_min = models.FloatField(null=True, blank=True)
    low_weekly_max = models.FloatField(null=True, blank=True)

    # 🔥 HIGH ANALYSIS
    high_yearly = models.FloatField(null=True, blank=True)
    high_monthly = models.FloatField(null=True, blank=True)
    high_monthly_avg = models.FloatField(null=True, blank=True)
    high_monthly_min = models.FloatField(null=True, blank=True)
    high_monthly_max = models.FloatField(null=True, blank=True)
    high_weekly = models.FloatField(null=True, blank=True)
    high_weekly_avg = models.FloatField(null=True, blank=True)
    high_weekly_min = models.FloatField(null=True, blank=True)
    high_weekly_max = models.FloatField(null=True, blank=True)
    def __str__(self):
        return f"{self.code} - {self.name}"


class DailyData(models.Model):
    date = models.DateField()
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    previous = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    open = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    low = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    high = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    close = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    high_low = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    close_open = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    high_open = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    open_low = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    status = models.CharField(max_length=50, blank=True)
    weekday = models.IntegerField(null=True, blank=True)
    week_number = models.IntegerField(null=True, blank=True)
    volume = models.BigIntegerField(null=True, blank=True)

    class Meta:
        unique_together = ('date', 'company')
        ordering = ['-date']

    def __str__(self):
        return f"{self.company.code} @ {self.date}"


class FiveMinuteData(models.Model):
    timestamp = models.DateTimeField()
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    previous = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    open = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    low = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    high = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    close = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    high_low = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    close_open = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    high_open = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    open_low = models.DecimalField(max_digits=20, decimal_places=4, null=True, blank=True)
    status = models.CharField(max_length=50, blank=True)
    weekday = models.IntegerField(null=True, blank=True)
    week_number = models.IntegerField(null=True, blank=True)
    volume = models.BigIntegerField(null=True, blank=True)

    class Meta:
        unique_together = ('timestamp', 'company')
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.company.code} @ {self.timestamp}"


class MarketIndex(models.Model):
    """An index (NIFTY 50, SENSEX, ...) whose history is kept - catalogue in market/index_data.py."""
    key = models.CharField(max_length=20, unique=True)                 # NIFTY, BANKNIFTY, SENSEX ...
    name = models.CharField(max_length=60)
    exchange = models.CharField(max_length=5)                           # NSE / BSE
    angel_token = models.CharField(max_length=20)
    option_symbol = models.CharField(max_length=20, blank=True)         # has index options (expiries) when set
    sort = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["sort"]

    def __str__(self):
        return self.key


class IndexDaily(models.Model):
    index = models.ForeignKey(MarketIndex, on_delete=models.CASCADE, related_name="daily")
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()

    class Meta:
        unique_together = ("index", "date")
        ordering = ["index", "date"]


class IndexFiveMinute(models.Model):
    index = models.ForeignKey(MarketIndex, on_delete=models.CASCADE, related_name="five_min")
    timestamp = models.DateTimeField()
    date = models.DateField()                                           # IST trading day (fast day lookups)
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()

    class Meta:
        unique_together = ("index", "timestamp")
        indexes = [models.Index(fields=["index", "date"])]
        ordering = ["index", "timestamp"]


class IndexExpiry(models.Model):
    """A past (or listed upcoming) expiry date of an index's options / futures."""
    index = models.ForeignKey(MarketIndex, on_delete=models.CASCADE, related_name="expiries")
    date = models.DateField()
    source = models.CharField(max_length=12)                            # nse_bhav / bse_bhav / schedule

    class Meta:
        unique_together = ("index", "date")
        ordering = ["index", "date"]

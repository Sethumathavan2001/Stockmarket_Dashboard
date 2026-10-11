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

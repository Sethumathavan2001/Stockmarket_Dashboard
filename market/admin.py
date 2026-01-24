from django.contrib import admin
from .models import Company, DailyData, FiveMinuteData


@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ('code', 'name', 'exchange', 'sector', 'industry')
    search_fields = ('code', 'name')


@admin.register(DailyData)
class DailyDataAdmin(admin.ModelAdmin):
    list_display = ('date', 'company', 'open', 'high', 'low', 'close', 'volume')
    list_filter = ('date', 'company__exchange', 'company__sector')
    search_fields = ('company__code',)


@admin.register(FiveMinuteData)
class FiveMinuteDataAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'company', 'open', 'high', 'low', 'close', 'volume')
    list_filter = ('company__exchange',)
    search_fields = ('company__code',)

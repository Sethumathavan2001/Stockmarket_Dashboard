from django.db import models

from market.models import Company


class AIReport(models.Model):
    """One finished Claude run: a single-stock analysis or a list of intraday picks."""

    KIND_ANALYSIS = "analysis"
    KIND_PICKS = "picks"
    KINDS = [(KIND_ANALYSIS, "Stock analysis"), (KIND_PICKS, "Intraday picks")]

    kind = models.CharField(max_length=16, choices=KINDS)
    company = models.ForeignKey(Company, null=True, blank=True, on_delete=models.CASCADE, related_name="ai_reports")
    created_at = models.DateTimeField(auto_now_add=True)
    question = models.TextField(blank=True)            # the user's optional extra instruction
    model = models.CharField(max_length=64)            # model that served the answer
    price_at = models.FloatField(null=True, blank=True)  # last price Claude saw (analysis only)
    result = models.JSONField(default=dict)            # the validated submit_* payload
    sources = models.JSONField(default=list)           # web pages the searches returned
    steps = models.JSONField(default=list)             # the run log shown on the page
    usage = models.JSONField(default=dict)
    cost_usd = models.FloatField(default=0)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["kind", "company", "-created_at"])]

    def __str__(self):
        who = self.company.code if self.company_id else "market"
        return f"{self.get_kind_display()} {who} {self.created_at:%Y-%m-%d %H:%M}"

import json

from django.core.management.base import BaseCommand, CommandError

from market.models import Company
from prediction import datasets as ds
from prediction import engine


class Command(BaseCommand):
    help = "Print a live prediction for a trained company (nothing is written to the database)."

    def add_arguments(self, parser):
        parser.add_argument("--company", required=True, help="company code, e.g. TCS.NS")
        parser.add_argument("--timeframe", choices=ds.TIMEFRAMES, default="intraday")

    def handle(self, *args, **opts):
        company = Company.objects.filter(code=opts["company"]).first()
        if company is None:
            raise CommandError(f"No company '{opts['company']}'")
        try:
            data = engine.predict_stock(company, opts["timeframe"])
        except (engine.NotTrained, ValueError) as exc:
            raise CommandError(str(exc))
        data.pop("candles")
        data.pop("leaderboards")
        self.stdout.write(json.dumps(data, indent=2, default=str))

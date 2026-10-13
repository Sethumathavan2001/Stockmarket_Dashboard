"""
Train the prediction model zoo for many stocks.

    python manage.py train_models --company TCS.NS
    python manage.py train_models --top 50                 # the 50 most traded stocks
    python manage.py train_models --top 50 --skip-trained  # only those without models yet
    python manage.py train_models --all --timeframe daily

Each stock: 7 models + 2 ensembles, walk-forward tested, for intraday (15 / 30 / 60 min)
and daily (1 / 5 days).  Models are saved to ml_models/<CODE>/.  About 3 minutes per stock.
"""
import re
import time
from datetime import timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db.models import Avg, ExpressionWrapper, F, FloatField, Max

from market.models import Company, DailyData
from prediction import datasets as ds
from prediction import engine

SPECIAL = re.compile(r"-(BL|T0|RE|P1|IV|RR)\.NS$")


def most_traded(n):
    last = DailyData.objects.aggregate(d=Max("date"))["d"]
    ranked = (DailyData.objects.filter(date__gte=last - timedelta(days=90)).exclude(company__sector="ETF")
              .values("company__code")
              .annotate(t=Avg(ExpressionWrapper(F("close") * F("volume"), output_field=FloatField())))
              .order_by("-t"))
    codes = [r["company__code"] for r in ranked if not SPECIAL.search(r["company__code"])]
    return codes[:n]


class Command(BaseCommand):
    help = "Train the model zoo (7 models + ensembles, walk-forward tested) for one, the top N, or all stocks."

    def add_arguments(self, parser):
        who = parser.add_mutually_exclusive_group(required=True)
        who.add_argument("--company", help="one company code, e.g. TCS.NS")
        who.add_argument("--top", type=int, help="the N most traded stocks (average daily turnover, last 3 months)")
        who.add_argument("--all", action="store_true", help="every company (excluding ETFs)")
        parser.add_argument("--timeframe", choices=ds.TIMEFRAMES, help="only intraday or only daily (default both)")
        parser.add_argument("--skip-trained", action="store_true", help="skip stocks that already have models")

    def handle(self, *args, **opts):
        if opts["company"]:
            codes = [opts["company"]]
            if not Company.objects.filter(code=codes[0]).exists():
                raise CommandError(f"No company '{codes[0]}'")
        elif opts["top"]:
            codes = most_traded(opts["top"])
        else:
            codes = list(Company.objects.exclude(sector="ETF").order_by("code").values_list("code", flat=True))
        if opts["skip_trained"]:
            done = {m["company"] for m in engine.list_manifests()}
            codes = [c for c in codes if c not in done]
        timeframes = (opts["timeframe"],) if opts["timeframe"] else ds.TIMEFRAMES

        self.stdout.write(f"Training {len(codes)} stock(s), timeframes: {', '.join(timeframes)}")
        started = time.time()
        for i, code in enumerate(codes, 1):
            t0 = time.time()
            company = Company.objects.get(code=code)
            try:
                m = engine.train_stock(company, timeframes=timeframes)
            except Exception as exc:
                self.stderr.write(self.style.ERROR(f"[{i}/{len(codes)}] {code}: failed - {exc}"))
                continue
            parts = []
            for tf, info in m["timeframes"].items():
                if tf not in timeframes:
                    continue
                if "error" in info:
                    parts.append(f"{tf}: {info['error']}")
                    continue
                for h in info["horizons"].values():
                    best = next(x for x in h["leaderboard"] if x["key"] == h["best"])
                    parts.append(f"{h['label']} {h['best']} IC {best['ic']:+.3f}" + (" EDGE" if h["edge"] else ""))
            self.stdout.write(f"[{i}/{len(codes)}] {code} ({time.time() - t0:.0f}s): " + " | ".join(parts))
        self.stdout.write(self.style.SUCCESS(f"Done in {(time.time() - started) / 60:.1f} min. Models in ml_models/."))

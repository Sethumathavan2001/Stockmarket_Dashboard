from django.core.management.base import BaseCommand, CommandError

from market.models import Company  # adjust if your app is named differently
from prediction.trainer import train_company_model


class Command(BaseCommand):
    help = "Train stock prediction models for one company (--company CODE) or all companies."

    def add_arguments(self, parser):
        parser.add_argument("--company", type=str, help="Company code to train. Omit to train all companies.")
        parser.add_argument("--n-splits", type=int, default=5, help="Number of walk-forward validation folds.")

    def handle(self, *args, **options):
        code = options.get("company")
        n_splits = options["n_splits"]

        companies = Company.objects.filter(code=code) if code else Company.objects.all()
        if code and not companies.exists():
            raise CommandError(f"No company found with code '{code}'")

        for company in companies:
            self.stdout.write(f"Training {company.code}...")
            try:
                bundle = train_company_model(company, n_splits=n_splits)
            except ValueError as exc:
                self.stderr.write(self.style.WARNING(f"  skipped: {exc}"))
                continue
            except Exception as exc:
                self.stderr.write(self.style.ERROR(f"  failed: {exc}"))
                continue

            for h, m in bundle["metrics"].items():
                self.stdout.write(
                    f"  horizon={bundle['horizon_minutes'][h]}m  "
                    f"MAE={m['mae']:.5f}  RMSE={m['rmse']:.5f}  "
                    f"DirAcc={m['directional_accuracy'] * 100:.1f}%  Bias={m['bias']:.5f}  "
                    f"SimStrategyReturn={m['simulated_strategy_return']:.5f}"
                )
            self.stdout.write(self.style.SUCCESS(f"  saved model for {company.code}"))

import json

from django.core.management.base import BaseCommand, CommandError

from market.models import Company  # adjust if your app is named differently
from prediction.predictor import predict_company, PredictionError


class Command(BaseCommand):
    help = "Run a live prediction for a company and print the JSON result (does not write to the DB)."

    def add_arguments(self, parser):
        parser.add_argument("--company", type=str, required=True, help="Company code to predict.")

    def handle(self, *args, **options):
        code = options["company"]
        try:
            company = Company.objects.get(code=code)
        except Company.DoesNotExist:
            raise CommandError(f"No company found with code '{code}'")

        try:
            result = predict_company(company)
        except PredictionError as exc:
            raise CommandError(str(exc))

        self.stdout.write(json.dumps(result, indent=2))

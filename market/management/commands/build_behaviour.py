from django.core.management.base import BaseCommand

from market import behaviour_data


class Command(BaseCommand):
    help = "Rebuild the stock-by-day matrix behind the Stock Behaviour page (logs/stock_behaviour.npz)."

    def handle(self, *args, **options):
        result = behaviour_data.build(step=lambda message: self.stdout.write(message))
        self.stdout.write(self.style.SUCCESS(str(result)))

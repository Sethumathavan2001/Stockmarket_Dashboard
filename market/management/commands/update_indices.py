from django.core.management.base import BaseCommand

from market import index_data


class Command(BaseCommand):
    help = "Download / update index history (daily + 5-minute candles from Angel One) and past expiry dates."

    def add_arguments(self, parser):
        parser.add_argument("--index", action="append", help="only this index key (repeatable), e.g. NIFTY")
        parser.add_argument("--no-expiries", action="store_true", help="skip the expiry scan")

    def handle(self, *args, **opts):
        summary = index_data.update_all(opts.get("index"), step=lambda m: self.stdout.write(m, ending="\n"),
                                        expiries=not opts["no_expiries"])
        self.stdout.write(self.style.SUCCESS(f"done: {summary}"))

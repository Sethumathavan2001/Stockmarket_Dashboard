"""
Fill Company.leverage from Angel One's margin calculator: how many times its margin a position can be worth, intraday.

  python manage.py update_leverage                 every company (about 10 minutes)
  python manage.py update_leverage --company TCS.NS
  python manage.py update_leverage --missing        only companies with no figure yet

0 means Angel One gives no intraday margin: it isn't listed as an NSE equity there, or the calculator returned none, which
is how stocks it refuses for INTRADAY answer.  A stock whose price or margin can't be read keeps its old figure.
Places no orders.  Claude Orders' Refresh uses only the stocks at 5x (claude_board.five_x_stocks).
"""
import time

from django.core.management.base import BaseCommand

from market.models import LEVERAGE_MIN, Company
from paper_trading import angel
from paper_trading.services import today_ist

CALL_GAP = 0.12
SAVE_EVERY = 100
PASSES = 4
PASS_WAIT = 20


class Command(BaseCommand):
    help = "Fill Company.leverage from Angel One's intraday margin calculator."

    def add_arguments(self, parser):
        parser.add_argument("--company", help="One company code, like TCS.NS")
        parser.add_argument("--missing", action="store_true", help="Only companies with no figure yet")

    def handle(self, *args, **options):
        companies = Company.objects.exclude(sector="ETF").order_by("code")
        if options["company"]:
            companies = companies.filter(code=options["company"])
        if options["missing"]:
            companies = companies.filter(leverage__isnull=True)
        today, found, unlisted = today_ist(), [], []
        for company in companies:
            try:
                found.append((company, angel.instrument(company.code)))
            except angel.AngelError:
                unlisted.append(company)
        prices = angel.quotes([inst["token"] for _, inst in found], "NSE") if found else {}

        counts, batch = {"5x": 0, "lower": 0, "none": len(unlisted)}, []
        for company in unlisted:
            self._mark(company, 0.0, today, batch)
        unread = found
        for attempt in range(PASSES):                            # Angel One's rate limit passes in a few seconds
            if attempt and unread:
                self.stdout.write(f"Trying {len(unread)} again")
                time.sleep(PASS_WAIT)
            unread = self._read(unread, prices, today, counts, batch)
        self._save(batch)
        for company, _ in unread:
            self.stderr.write(f"{company.code}: not read")
        self.stdout.write(f"5x: {counts['5x']} | lower: {counts['lower']} | no intraday margin: {counts['none']} | "
                          f"not read, kept as before: {len(unread)}")

    def _read(self, stocks, prices, today, counts, batch):
        """Ask for each stock's leverage and return the ones Angel One didn't answer for."""
        unread = []
        for n, (company, inst) in enumerate(stocks, 1):
            price = float((prices.get(inst["token"]) or {}).get("ltp") or 0)
            try:
                if price <= 0:
                    raise angel.AngelError("no last price")
                leverage = angel.intraday_leverage(inst, price)
            except angel.AngelError:
                unread.append((company, inst))
                continue
            counts["none" if not leverage else "5x" if leverage >= LEVERAGE_MIN else "lower"] += 1
            self._mark(company, leverage, today, batch)
            if n % 200 == 0:
                self.stdout.write(f"{n} of {len(stocks)}")
            time.sleep(CALL_GAP)
        return unread

    def _mark(self, company, leverage, today, batch):
        company.leverage, company.leverage_checked = leverage, today
        batch.append(company)
        if len(batch) >= SAVE_EVERY:
            self._save(batch)

    @staticmethod
    def _save(batch):
        Company.objects.bulk_update(batch, ["leverage", "leverage_checked"])
        batch.clear()

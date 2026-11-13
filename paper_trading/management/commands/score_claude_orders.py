"""
Replay every finished Claude Orders recommendation on its day's 5-minute candles (paper_trading/claude_record.py).

  python manage.py score_claude_orders

Reads candles from Angel One; places no orders.  The Claude Orders pages also do this after each Refresh.
"""
from django.core.management.base import BaseCommand

from paper_trading import claude_record


class Command(BaseCommand):
    help = "Score the finished Claude Orders recommendations for the profit / loss counts on the Claude Orders pages."

    def handle(self, *args, **options):
        done = claude_record.update()
        records = claude_record.records()
        profit = sum(r["profit"] for r in records.values())
        loss = sum(r["loss"] for r in records.values())
        self.stdout.write(f"Scored now: {done['scored']} | left for later (Angel One busy): {done['kept_for_later']} | "
                          f"stocks with a record: {len(records)} | profit: {profit} | loss: {loss}")

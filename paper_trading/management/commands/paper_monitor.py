"""Run the paper-trading stop loss / target / square-off watcher without runserver.

    python manage.py paper_monitor          # keep watching (Ctrl+C to stop)
    python manage.py paper_monitor --once   # one check now
"""
from django.core.management.base import BaseCommand

from paper_trading import monitor


class Command(BaseCommand):
    help = "Close Live paper positions that reached their stop loss / target, and square off Intraday at the cutoff."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true", help="Run one check and exit.")

    def handle(self, *args, **opts):
        if opts["once"]:
            exits, closed = monitor.run_once()
            self.stdout.write(f"{len(exits)} stop loss / target exits, {len(closed)} square-offs")
            return
        self.stdout.write("Watching Live paper positions every %s s during market hours (Ctrl+C to stop)..." % monitor.CHECK_SECONDS)
        monitor._loop()

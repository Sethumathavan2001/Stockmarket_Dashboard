import multiprocessing
import os
import sys

from django.apps import AppConfig


class PaperTradingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "paper_trading"
    verbose_name = "Paper Trading"

    def ready(self):
        # Stop loss / target / square-off watcher (paper_trading/monitor.py) and the GIFT Nifty price recorder
        # (market/gift_nifty.py): start them in the process that
        # serves `runserver` (the auto-reloader's child, or the only process with --noreload) - not in
        # tests, migrations, shells or scripts, and never in a worker process (board_pool.py): those inherit
        # runserver's arguments and RUN_MAIN, and a second watcher could exit a real trade twice.
        argv = sys.argv
        if multiprocessing.parent_process() is not None:
            return
        if "runserver" in argv and (os.environ.get("RUN_MAIN") == "true" or "--noreload" in argv):
            from market import gift_nifty

            from . import monitor
            monitor.start()
            gift_nifty.start()

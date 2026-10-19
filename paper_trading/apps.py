import os
import sys

from django.apps import AppConfig


class PaperTradingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "paper_trading"
    verbose_name = "Paper Trading"

    def ready(self):
        # Stop loss / target / square-off watcher (paper_trading/monitor.py): start it in the process that
        # serves `runserver` (the auto-reloader's child, or the only process with --noreload) - not in
        # tests, migrations, shells or scripts.
        argv = sys.argv
        if "runserver" in argv and (os.environ.get("RUN_MAIN") == "true" or "--noreload" in argv):
            from . import monitor
            monitor.start()

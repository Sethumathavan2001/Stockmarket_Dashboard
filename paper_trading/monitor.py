"""
Server-side watcher for Live paper positions: stop loss / target exits and the Intraday square-off
run here, so they happen with the Paper Trading page closed or while you are on another page.

Every CHECK_SECONDS during market hours (both books):
  * each open Live position with a stop loss or target is checked against today's 5-minute candles
    that started after the position (or its levels) last changed - so a level crossed while the
    server was off is still caught, at the right price and time.  Fills: at the level, or at the
    candle's open if price gapped past it; a candle touching both levels counts as the stop.
  * Intraday positions are squared off at the cutoff (15:20 IST) - squareoff_live_intraday().
  * the same round runs Option Trading's watcher (options/services.run_once).
Exits are close-only (services.execute_order(close_only=True)) and serialised by ORDER_LOCK, so the
watcher and an open page can never close the same position twice.

Started automatically by `manage.py runserver` (see apps.py); several server processes elect one
watcher through a lock file.  Without runserver:  python manage.py paper_monitor  [--once]
"""
import logging
import os
import threading
import time as _time
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

from django.conf import settings
from django.db import close_old_connections

from . import services as svc
from .models import PaperPosition

logger = logging.getLogger(__name__)

CHECK_SECONDS = 15
LOCK_FILE = Path(settings.BASE_DIR) / "logs" / "paper_monitor.lock"
LOCK_STALE_SECONDS = 90
BOOK_OWNERS = [f"book:{b}" for b in svc.BOOKS]
MARKET_OPEN, MARKET_END = time(9, 15), time(15, 35)


def _floor5(dt):
    return dt.replace(minute=dt.minute - dt.minute % 5, second=0, microsecond=0)


def level_hit(pos, df, since):
    """First stop / target touch in candles that STARTED after `since`: (reason, price, candle time) or None."""
    if df is None or df.empty:
        return None
    long_ = pos.quantity > 0
    sl = float(pos.stop_loss) if pos.stop_loss is not None else None
    tp = float(pos.target_price) if pos.target_price is not None else None
    start = _floor5(svc.to_ist(since)) + timedelta(minutes=5)
    for r in df.itertuples(index=False):
        ts = svc.to_ist(r.timestamp.to_pydatetime())
        if ts < start:
            continue
        o, h, l = float(r.open), float(r.high), float(r.low)
        if sl is not None and (l <= sl if long_ else h >= sl):
            return "stop_loss", (min(sl, o) if long_ else max(sl, o)), ts      # gap through the stop -> worse fill
        if tp is not None and (h >= tp if long_ else l <= tp):
            return "target", (max(tp, o) if long_ else min(tp, o)), ts         # gap through the target -> better fill
    return None


def check_levels(now=None):
    """Close every Live position (both books) whose stop loss or target was reached. Returns the exit orders."""
    exits = []
    positions = (PaperPosition.objects.filter(account__owner_key__in=BOOK_OWNERS, context=svc.LIVE)
                 .exclude(quantity=0).filter(models_q_levels()).select_related("company", "account"))
    for pos in positions:
        try:
            df = svc.fetch_live_today(pos.company)
        except Exception as exc:                          # network / Yahoo - try again next round
            logger.warning("monitor: no live data for %s: %s", pos.company.code, exc)
            continue
        since = max(d for d in (pos.updated_at, pos.last_entry_at) if d is not None)
        hit = level_hit(pos, df, since)
        if not hit:
            continue
        reason, price, when = hit
        try:
            order, _, _ = svc.execute_order(
                account=pos.account, company=pos.company,
                side="SELL" if pos.quantity > 0 else "BUY", quantity=abs(pos.quantity),
                price=Decimal(str(round(price, 2))), mode=svc.LIVE, replay_date=None, market_time=when,
                reason=reason, product=pos.product, close_only=True,
            )
            exits.append(order)
            logger.info("monitor: %s %s %s %s @ %.2f (%s)", pos.account.owner_key, reason, order.side,
                        pos.company.code, price, when.strftime("%H:%M"))
        except svc.AlreadyClosed:
            pass                                          # an open page closed it a moment ago
        except svc.TradeError as exc:
            logger.warning("monitor: could not exit %s: %s", pos.company.code, exc)
    return exits


def models_q_levels():
    from django.db.models import Q
    return Q(stop_loss__isnull=False) | Q(target_price__isnull=False)


def run_once(now=None):
    now = svc.to_ist(now or datetime.now(svc.IST))
    exits = check_levels(now)
    closed = []
    for owner in BOOK_OWNERS:
        acct = svc.get_account(int(owner.split(":")[1]))
        if acct.positions.filter(context=svc.LIVE, product=svc.INTRADAY).exclude(quantity=0).exists():
            closed += svc.squareoff_live_intraday(acct, now=now)["closed"]
    return exits, closed


def _run_options(now):
    """Option Trading's limit fills, stop loss / target, MIS square-off and expiry settlement (options/services.py)."""
    try:
        from options import services as opt
        opt.run_once(now)
    except Exception:
        logger.exception("options monitor round failed")


# ---------------------------------------------------------------- one watcher across server processes
def _is_leader():
    try:
        LOCK_FILE.parent.mkdir(exist_ok=True)
        if LOCK_FILE.exists():
            age = _time.time() - LOCK_FILE.stat().st_mtime
            owner = LOCK_FILE.read_text().strip()
            if owner != str(os.getpid()) and age < LOCK_STALE_SECONDS:
                return False
        LOCK_FILE.write_text(str(os.getpid()))
        return True
    except OSError:
        return False


def _loop():
    logger.info("paper monitor started (pid %s)", os.getpid())
    while True:
        now = datetime.now(svc.IST)
        trading = now.weekday() < 5 and MARKET_OPEN <= now.time() <= MARKET_END
        try:
            if _is_leader() and trading:
                run_once(now)
                _run_options(now)
        except Exception:                                 # never let the watcher die
            logger.exception("paper monitor round failed")
        finally:
            close_old_connections()
        _time.sleep(CHECK_SECONDS if trading else 60)


_started = False


def start():
    global _started
    if _started or not getattr(settings, "PAPER_MONITOR_ENABLED", True):
        return
    _started = True
    threading.Thread(target=_loop, name="paper-monitor", daemon=True).start()

"""
Track record of Claude Orders recommendations: how each stock's past recommendations turned out, for the profit / loss
counts beside each stock on the Claude Orders pages, which also put the best records first.

Every saved list (ClaudeOrderList) is replayed on that day's 5-minute candles from Angel One, as the page would have
placed it: entry at the open of the first candle after the list was made, stop loss and target the recommendation's
percentages from that entry, out at 15:20.

  profit   the target was reached first, or the price was on the right side at 15:20
  loss     the stop loss was hit first (also when one candle reached both - it can't say which came first), or the price
           was on the wrong side at 15:20
  flat     no change at 15:20

A stock recommended on the same side in several lists in one day counts once, from its first list.  Charges aren't
included.  A day is scored once it has closed; results are kept in logs/claude_outcomes.json.

  update()          score every finished recommendation not scored yet (Angel One candles, a read-only data call)
  start_update()    the same in a background thread - after each Refresh
  records()         {code: {"profit", "loss", "flat", "from", "to"}}
  python manage.py score_claude_orders
"""
import json
import logging
import threading
import time as _time
from datetime import datetime, time, timedelta
from pathlib import Path

from django.conf import settings
from django.core.cache import cache

from prediction.market_time import IST

log = logging.getLogger(__name__)

FILE = Path(settings.BASE_DIR) / "logs" / "claude_outcomes.json"
OPEN, SQUARE_OFF, DAY_OVER = time(9, 15), time(15, 20), time(15, 30)
CALL_GAP = 0.35                  # seconds between candle requests (Angel One allows about three a second)
RECORDS_KEY = "claude_record:records"
_lock = threading.Lock()
_running = threading.Lock()


# ---------------------------------------------------------------- one recommendation
def after(candles, made):
    """The 5-minute candles a recommendation made at `made` could trade: starting after it, before the square-off."""
    return [c for c in candles if c[0] >= made and c[0].time() < SQUARE_OFF]


def outcome(side, bars, stop_pct, target_pct):
    """bars: [(start, open, high, low, close)] in time order -> {"result", "entry", "exit", "how", "pct"} or None."""
    if not bars or not bars[0][1]:
        return None
    sign = 1 if side == "BUY" else -1
    entry = bars[0][1]
    stop, target = entry * (1 - sign * stop_pct / 100), entry * (1 + sign * target_pct / 100)
    exit_price, how = bars[-1][4], "square-off"
    for _, _, high, low, _ in bars:
        stopped = low <= stop if sign > 0 else high >= stop
        reached = high >= target if sign > 0 else low <= target
        if stopped or reached:
            exit_price, how = (stop, "stop loss") if stopped else (target, "target")
            break
    pct = sign * (exit_price / entry - 1) * 100
    result = "profit" if pct > 1e-9 else "loss" if pct < -1e-9 else "flat"
    return {"result": result, "entry": round(entry, 2), "exit": round(exit_price, 2), "how": how, "pct": round(pct, 3)}


# ---------------------------------------------------------------- the saved lists
def _made(row):
    return row.generated.astimezone(IST)


def recommendations():
    """[(made, rec)] from every saved list, oldest first."""
    from .models import ClaudeOrderList

    out = []
    for row in ClaudeOrderList.objects.order_by("generated"):
        for rec in row.payload.get("orders") or []:
            if rec.get("id") and rec.get("code") and rec.get("side") in ("BUY", "SELL"):
                out.append((_made(row), rec))
    return out


def finished(made, now):
    return made.date() < now.date() or now.time() >= DAY_OVER


def load():
    try:
        return json.loads(FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _save(scored):
    FILE.parent.mkdir(exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(scored, indent=0), encoding="utf-8")
    tmp.replace(FILE)


def day_candles(code, day):
    """[(start IST, o, h, l, c)] of one stock's 5-minute candles on one day, from Angel One."""
    from paper_trading import angel

    inst = angel.instrument(code)
    params = {"exchange": "NSE", "symboltoken": inst["token"], "interval": "FIVE_MINUTE",
              "fromdate": f"{day:%Y-%m-%d} 09:15", "todate": f"{day:%Y-%m-%d} 15:30"}
    res = angel.read_any(lambda sc: sc.getCandleData(params))[1]
    return [(datetime.fromisoformat(r[0]).astimezone(IST), float(r[1]), float(r[2]), float(r[3]), float(r[4]))
            for r in (res or {}).get("data") or []]


def update(now=None, candles=day_candles):
    """Score the finished recommendations not scored yet.  Returns {"scored", "kept_for_later"}."""
    from paper_trading import angel

    now = now or datetime.now(IST)
    with _lock:
        scored = load()
    todo = [(made, rec) for made, rec in recommendations() if rec["id"] not in scored and finished(made, now)]
    fetched, new, later = {}, {}, 0
    for made, rec in todo:
        key = (rec["code"], made.date())
        if key not in fetched:
            if fetched:
                _time.sleep(CALL_GAP)
            try:
                fetched[key] = candles(rec["code"], made.date())
            except angel.AngelError as exc:
                text = str(exc)
                if "is not an NSE equity" in text:
                    fetched[key] = []                         # not an NSE equity on Angel One: nothing to replay
                else:                                         # Angel One busy or down: try again next time
                    fetched[key] = None
                    log.warning("Claude record: %s on %s not read: %s", rec["code"], made.date(), exc)
        bars = fetched[key]
        if bars is None:
            later += 1
            continue
        try:
            stop_pct, target_pct = float(rec["stop_pct"]), float(rec["target_pct"])
        except (KeyError, TypeError, ValueError):
            new[rec["id"]] = None
            continue
        new[rec["id"]] = outcome(rec["side"], after(bars, made), stop_pct, target_pct)
    if new:
        with _lock:
            scored = {**load(), **new}
            _save(scored)
        cache.delete(RECORDS_KEY)
    return {"scored": len(new), "kept_for_later": later}


def start_update():
    """update() in the background; False when one is already running."""
    if not _running.acquire(blocking=False):
        return False

    def run():
        try:
            update()
        except Exception:
            log.exception("Claude record: scoring failed")
        finally:
            from django.db import connection
            connection.close()
            _running.release()

    threading.Thread(target=run, name="claude-record", daemon=True).start()
    return True


def records():
    """{code: {"profit", "loss", "flat", "from", "to"}}, each stock + side + day counted once (its first list)."""
    hit = cache.get(RECORDS_KEY)
    if hit is not None:
        return hit
    with _lock:
        scored = load()
    seen, out = set(), {}
    for made, rec in recommendations():
        result = scored.get(rec["id"])
        key = (rec["code"], rec["side"], made.date())
        if not result or key in seen:
            continue
        seen.add(key)
        r = out.setdefault(rec["code"], {"profit": 0, "loss": 0, "flat": 0, "from": made.date().isoformat()})
        r[result["result"]] += 1
        r["to"] = made.date().isoformat()
    cache.set(RECORDS_KEY, out, 60)
    return out

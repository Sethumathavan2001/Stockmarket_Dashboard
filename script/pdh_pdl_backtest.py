#!/usr/bin/env python3
"""
pdh_pdl_backtest.py - find the Previous-Day-High / Previous-Day-Low pattern in
ALL your historical 5-minute data and write the trades to an Excel file.

THE PATTERN
-----------
  PDH (long)   A candle crosses ABOVE the previous day's high, and the NEXT
               candle is RED  ->  BUY.
  PDL (short)  A candle crosses BELOW the previous day's low, and the NEXT
               candle is GREEN ->  SELL (short).

  Quantity 10.  Stop loss 1 %, target 2 % of the entry price.  Intraday: a
  trade that hits neither is closed at the end of that day.

HOW IT IS SIMULATED  (every choice below has a flag - see --help)
------------------------------------------------------------------
  * Previous day = the previous trading day PRESENT IN YOUR DATA for that stock,
    and PDH / PDL are the highest high / lowest low of its 5-minute candles.
  * "Cross" (default --cross-on close): the candle CLOSES beyond the level while
    the candle before it closed on the other side (or at the level).  The first
    candle of a day is compared with the previous day's last close, so a gap
    that closes beyond the level counts.  --cross-on high counts a wick: the
    first candle whose high (PDH) / low (PDL) goes beyond the level.
  * Confirmation candle: the very next candle.  Red = close < open,
    green = close > open (a doji is neither, so no trade).
  * Entry (default --entry close): at the CLOSE of the confirmation candle,
    because that is when you know it was red / green.  --entry next-open fills
    at the open of the candle after it instead (a stricter, more realistic fill).
  * Exits are checked on the candles AFTER entry:
      - stop loss / target touched inside a candle -> filled exactly at the level
      - a candle that OPENS beyond a level (gap)   -> filled at that open
      - BOTH levels inside one candle              -> the STOP LOSS is assumed
        to have happened first (the pessimistic choice; counted in the Summary)
      - neither by the end of the day              -> closed at the last close
        (or at --squareoff HH:MM if you give one)
  * One open position per stock at a time; after an exit the scan carries on,
    so a stock can trade more than once in a day.
  * Prices are rounded to 2 decimals, levels likewise (same as your paper
    trading app).  No brokerage, taxes or slippage.

TIME COLUMN
-----------
  Time (IST) is when the order would have been filled: entries at a candle's
  close are stamped with the END of that candle; exits that touch a level
  inside a candle are stamped with the START of that candle (5-minute data
  cannot say when inside the candle); end-of-day exits with the END of the last
  candle.

USAGE  (from the project folder; the Excel file goes to script/output/)
-----
    pip install openpyxl
    python script/pdh_pdl_backtest.py                       # all stocks, all dates
    python script/pdh_pdl_backtest.py --symbols TCS,INFY --from 2025-01-01 --to 2025-12-31
    python script/pdh_pdl_backtest.py --settings myproject.settings --output research.xlsx

  The settings module is found automatically from manage.py; use --settings if
  yours lives elsewhere.  Models are read from the app named by --app
  (default "stocks": Company, FiveMinuteData).  Nothing is written to the database.

OUTPUT (Excel)
--------------
  Trades        exactly: Date | Time (IST) | Symbol | Product | Side | Qty | Price |
                Realized P&L - an order log, each trade as an entry row followed
                by its exit row; Realized P&L is filled on the exit row only.
  Trade details one row per trade: the cross / confirmation candles, the level,
                stop loss, target, exit reason, P&L %, minutes held ...
  Summary       parameters, data coverage, signal funnel, win rate, profit factor,
                drawdown, results by setup / exit reason / month
  By symbol     which stocks the pattern worked on
  Daily P&L     per-day and cumulative P&L, with a chart
"""
import argparse
import datetime as dt
import math
import os
import re
import sys
import time as _time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# This script lives in <project>/script/; Excel output goes to <project>/script/output/.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def output_path(name):
    """A bare file name goes into script/output/; a path with a folder is used as given."""
    path = name if os.path.dirname(name) else os.path.join(OUTPUT_DIR, name)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path
CANDLE = dt.timedelta(minutes=5)
CENT = Decimal("0.01")
BUY, SELL = "BUY", "SELL"
PRODUCT = "INTRADAY"          # shorts are only possible intraday, and every trade is closed the same day
MAX_SHEET_ROWS = 1_000_000    # Excel's limit is 1,048,576


# ==========================================================================
# parameters, result objects
# ==========================================================================
@dataclass(frozen=True)
class Params:
    qty: int = 10
    sl_pct: Decimal = Decimal("1")
    target_pct: Decimal = Decimal("2")
    cross_on: str = "close"                  # "close" | "high"
    entry: str = "close"                     # "close" | "next-open"
    squareoff: Optional[dt.time] = None      # e.g. 15:15; None = hold to the last candle
    max_gap_days: int = 7                    # previous day must be at most this many calendar days back
    min_prev_candles: int = 0                # ignore days whose previous day has fewer candles


@dataclass
class Trade:
    symbol: str
    day: dt.date
    side: str                  # BUY (PDH long) / SELL (PDL short)
    level_name: str            # "PDH" / "PDL"
    level: float
    cross_time: dt.datetime    # start of the candle that crossed
    confirm_time: dt.datetime  # start of the confirming red / green candle
    entry_time: dt.datetime
    entry_price: Decimal
    stop_loss: Decimal
    target: Decimal
    exit_time: dt.datetime
    exit_price: Decimal
    exit_reason: str
    qty: int
    pnl: Decimal

    @property
    def setup(self):
        return ("PDH cross, then red candle -> BUY" if self.side == BUY
                else "PDL cross, then green candle -> SELL (short)")

    @property
    def minutes_held(self):
        return (self.exit_time - self.entry_time).total_seconds() / 60.0

    @property
    def pnl_pct(self):
        base = self.entry_price * self.qty
        return float(self.pnl / base * 100) if base else 0.0


@dataclass
class Stats:
    symbols: int = 0
    symbols_with_data: int = 0
    candles: int = 0
    days_scanned: int = 0
    days_no_previous: int = 0
    days_gap: int = 0
    days_prev_short: int = 0
    pdh_crosses: int = 0
    pdh_confirmed: int = 0
    pdl_crosses: int = 0
    pdl_confirmed: int = 0
    ambiguous_cross: int = 0
    no_exit_candle: int = 0
    after_squareoff: int = 0
    bad_levels: int = 0
    both_levels_same_candle: int = 0
    first_day: Optional[dt.date] = None
    last_day: Optional[dt.date] = None

    def note_day(self, d):
        self.first_day = d if self.first_day is None or d < self.first_day else self.first_day
        self.last_day = d if self.last_day is None or d > self.last_day else self.last_day


@dataclass
class DayInfo:
    day: dt.date
    high: float
    low: float
    close: float
    n: int


# candle tuple: (start datetime IST, open, high, low, close)
def day_info(day, candles):
    return DayInfo(day, max(c[2] for c in candles), min(c[3] for c in candles), candles[-1][4], len(candles))


def q2(x):
    """Round to 2 decimals, half up (what the paper-trading app does)."""
    return Decimal(repr(float(x))).quantize(CENT, rounding=ROUND_HALF_UP)


# ==========================================================================
# the pattern
# ==========================================================================
def find_exit(candles, start, side, sl, tp, squareoff, stats):
    """
    Walk forward from candle `start` until the stop loss / target / square-off.
    Returns (candle index, exit datetime, exit price Decimal, reason).
    """
    n = len(candles)
    long_ = side == BUY
    slf, tpf = float(sl), float(tp)
    for j in range(start, n):
        ts, o, h, l, c = candles[j]
        if squareoff is not None and ts.time() >= squareoff:
            return j, ts, q2(o), "Square-off"
        if long_:
            if o <= slf:
                return j, ts, q2(o), "Stop loss (gap)"
            if o >= tpf:
                return j, ts, q2(o), "Target (gap)"
            hit_sl, hit_tp = l <= slf, h >= tpf
        else:
            if o >= slf:
                return j, ts, q2(o), "Stop loss (gap)"
            if o <= tpf:
                return j, ts, q2(o), "Target (gap)"
            hit_sl, hit_tp = h >= slf, l <= tpf
        if hit_sl and hit_tp:
            stats.both_levels_same_candle += 1      # unknowable from 5-min data: assume the worst
            return j, ts, sl, "Stop loss"
        if hit_sl:
            return j, ts, sl, "Stop loss"
        if hit_tp:
            return j, ts, tp, "Target"
    ts, _o, _h, _l, c = candles[-1]
    return n - 1, ts + CANDLE, q2(c), "Square-off (end of day)"


def scan_day(symbol, day, candles, prev, P, stats):
    """All pattern trades for one stock on one day. `candles` is that day's list, oldest first."""
    n = len(candles)
    pdh, pdl = prev.high, prev.low
    trades = []
    i = 0
    while i < n - 1:
        ts, o, h, l, c = candles[i]
        if P.cross_on == "close":
            ref = candles[i - 1][4] if i else prev.close
            up = c > pdh and ref <= pdh
            down = c < pdl and ref >= pdl
        else:  # wick: first candle whose high / low goes beyond the level
            up = h > pdh and (i == 0 or candles[i - 1][2] <= pdh)
            down = l < pdl and (i == 0 or candles[i - 1][3] >= pdl)
        if up and down:                       # one huge candle through both levels: can't tell the order
            stats.ambiguous_cross += 1
            i += 1
            continue
        if not (up or down):
            i += 1
            continue

        cts, co, ch, cl, cc = candles[i + 1]  # the confirmation candle
        if up:
            stats.pdh_crosses += 1
            if not cc < co:                   # must be red
                i += 1
                continue
            stats.pdh_confirmed += 1
            side, level_name, level = BUY, "PDH", pdh
        else:
            stats.pdl_crosses += 1
            if not cc > co:                   # must be green
                i += 1
                continue
            stats.pdl_confirmed += 1
            side, level_name, level = SELL, "PDL", pdl

        if P.entry == "close":
            entry_time, entry_raw, first_exit = cts + CANDLE, cc, i + 2
        else:                                 # next-open
            first_exit = i + 2
            if first_exit > n - 1:
                stats.no_exit_candle += 1
                i += 1
                continue
            entry_time, entry_raw = candles[first_exit][0], candles[first_exit][1]
        if first_exit > n - 1:                # nothing after the entry to exit on
            stats.no_exit_candle += 1
            i += 1
            continue
        if P.squareoff is not None and entry_time.time() >= P.squareoff:
            stats.after_squareoff += 1
            i += 1
            continue

        entry = q2(entry_raw)
        slp, tpp = P.sl_pct / 100, P.target_pct / 100
        if side == BUY:
            sl, tp = q2(entry * (1 - slp)), q2(entry * (1 + tpp))
            valid = sl < entry < tp
        else:
            sl, tp = q2(entry * (1 + slp)), q2(entry * (1 - tpp))
            valid = tp < entry < sl
        if not valid:                         # penny-stock rounding made a level equal the entry
            stats.bad_levels += 1
            i += 1
            continue

        exit_idx, exit_time, exit_price, reason = find_exit(candles, first_exit, side, sl, tp, P.squareoff, stats)
        pnl = (exit_price - entry) * P.qty if side == BUY else (entry - exit_price) * P.qty
        trades.append(Trade(
            symbol=symbol, day=day, side=side, level_name=level_name, level=level,
            cross_time=ts, confirm_time=cts, entry_time=entry_time, entry_price=entry,
            stop_loss=sl, target=tp, exit_time=exit_time, exit_price=exit_price,
            exit_reason=reason, qty=P.qty, pnl=pnl,
        ))
        i = exit_idx                          # no new entry until this position is closed
    return trades


def run_symbol(symbol, days, P, stats, date_from=None, date_to=None):
    """`days` yields (date, candles) in date order. Returns the symbol's trades."""
    trades = []
    prev = None
    for day, candles in days:
        stats.candles += len(candles)
        in_range = (date_from is None or day >= date_from) and (date_to is None or day <= date_to)
        if in_range:
            if prev is None:
                stats.days_no_previous += 1
            elif (day - prev.day).days > P.max_gap_days:
                stats.days_gap += 1
            elif prev.n < P.min_prev_candles:
                stats.days_prev_short += 1
            else:
                stats.days_scanned += 1
                stats.note_day(day)
                trades.extend(scan_day(symbol, day, candles, prev, P, stats))
        prev = day_info(day, candles)
    return trades


# ==========================================================================
# reading your Django models
# ==========================================================================
def find_settings_module(explicit):
    if explicit:
        return explicit
    if os.environ.get("DJANGO_SETTINGS_MODULE"):
        return os.environ["DJANGO_SETTINGS_MODULE"]
    manage = os.path.join(PROJECT_ROOT, "manage.py")
    if os.path.exists(manage):
        m = re.search(r"DJANGO_SETTINGS_MODULE['\"]\s*,\s*['\"]([\w.]+)['\"]", open(manage, encoding="utf-8").read())
        if m:
            return m.group(1)
    return None


def to_ist(ts):
    return ts.replace(tzinfo=IST) if ts.tzinfo is None else ts.astimezone(IST)


def db_bound(settings, d, end_of_day=False):
    """A datetime in the form the DB layer expects (aware if USE_TZ, else naive IST)."""
    base = dt.datetime.combine(d + dt.timedelta(days=1 if end_of_day else 0), dt.time.min, tzinfo=IST)
    return base if settings.USE_TZ else base.replace(tzinfo=None)


def days_from_rows(rows):
    """Group time-ordered (timestamp, o, h, l, c) rows into (date, candles) days, IST."""
    day, bucket = None, []
    for ts, o, h, l, c in rows:
        if o is None or h is None or l is None or c is None:
            continue
        ts = to_ist(ts)
        d = ts.date()
        if d != day:
            if bucket:
                yield day, bucket
            day, bucket = d, []
        bucket.append((ts, float(o), float(h), float(l), float(c)))
    if bucket:
        yield day, bucket


def load_and_run(args, P):
    import django
    settings_module = find_settings_module(args.settings)
    if not settings_module:
        sys.exit("Could not find your Django settings. Run this next to manage.py, or pass --settings myproject.settings")
    sys.path.insert(0, PROJECT_ROOT)
    os.environ["DJANGO_SETTINGS_MODULE"] = settings_module
    django.setup()
    from django.apps import apps
    from django.conf import settings

    try:
        Company = apps.get_model(args.app, "Company")
        Candle = apps.get_model(args.app, "FiveMinuteData")
    except LookupError as exc:
        sys.exit(f"Could not find the models in app '{args.app}' ({exc}). Use --app <your app name>.")

    companies = Company.objects.order_by("code")
    if args.symbols:
        wanted = [s.strip() for s in args.symbols.split(",") if s.strip()]
        companies = companies.filter(code__in=wanted)
        missing = sorted(set(wanted) - set(companies.values_list("code", flat=True)))
        if missing:
            print(f"  note: not in your database: {', '.join(missing)}")
    companies = list(companies.values_list("id", "code"))

    stats = Stats(symbols=len(companies))
    all_trades: List[Trade] = []
    started = _time.time()
    for k, (cid, code) in enumerate(companies, 1):
        qs = Candle.objects.filter(company_id=cid)
        if args.date_from:                                            # a few days of lead-in for the previous day
            qs = qs.filter(timestamp__gte=db_bound(settings, args.date_from - dt.timedelta(days=15)))
        if args.date_to:
            qs = qs.filter(timestamp__lt=db_bound(settings, args.date_to, end_of_day=True))
        rows = qs.order_by("timestamp").values_list("timestamp", "open", "high", "low", "close").iterator(chunk_size=20000)
        before = stats.candles
        trades = run_symbol(code, days_from_rows(rows), P, stats, args.date_from, args.date_to)
        if stats.candles > before:
            stats.symbols_with_data += 1
        all_trades.extend(trades)
        print(f"  [{k}/{len(companies)}] {code:<14} {len(trades):>5} trades", flush=True)
    print(f"  scanned {stats.candles:,} candles in {_time.time() - started:.1f}s")
    return all_trades, stats


# ==========================================================================
# statistics
# ==========================================================================
def summarize(trades):
    n = len(trades)
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win, gross_loss = sum(wins, Decimal(0)), -sum(losses, Decimal(0))
    total = sum(pnls, Decimal(0))
    # max drawdown on the equity curve ordered by exit time
    peak = cum = worst = Decimal(0)
    for t in sorted(trades, key=lambda t: (t.exit_time, t.symbol)):
        cum += t.pnl
        peak = max(peak, cum)
        worst = max(worst, peak - cum)
    return {
        "Trades": n,
        "Winners": len(wins),
        "Losers": len(losses),
        "Break-even": n - len(wins) - len(losses),
        "Win rate %": (len(wins) / n * 100) if n else None,
        "Total P&L": float(total),
        "Average P&L per trade": float(total / n) if n else None,
        "Average winner": float(gross_win / len(wins)) if wins else None,
        "Average loser": float(-gross_loss / len(losses)) if losses else None,
        "Profit factor": float(gross_win / gross_loss) if gross_loss else None,
        "Best trade": float(max(pnls)) if n else None,
        "Worst trade": float(min(pnls)) if n else None,
        "Max drawdown": float(worst),
        "Average minutes held": (sum(t.minutes_held for t in trades) / n) if n else None,
    }


def group_rows(trades, keyfunc):
    groups = defaultdict(list)
    for t in trades:
        groups[keyfunc(t)].append(t)
    return groups


# ==========================================================================
# Excel
# ==========================================================================
def build_workbook(trades, stats, P, args):
    try:
        from openpyxl import Workbook
        from openpyxl.chart import LineChart, Reference
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        sys.exit("This script writes Excel files with openpyxl. Install it:  pip install openpyxl")

    NAVY = "123B7A"
    head_font = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor=NAVY)
    band_fill = PatternFill("solid", fgColor="EEF3FF")
    thin = Side(style="thin", color="D5DCE8")
    MONEY = "#,##0.00;[Red]-#,##0.00"
    PCT = '0.00"%"'

    def style_header(ws, row=1):
        for cell in ws[row]:
            if cell.value is not None:
                cell.font, cell.fill = head_font, head_fill
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    def widths(ws, spec):
        for idx, w in enumerate(spec, 1):
            ws.column_dimensions[get_column_letter(idx)].width = w

    trades = sorted(trades, key=lambda t: (t.day, t.symbol, t.entry_time))
    wb = Workbook()

    # ------------------------------------------------------------ Trades (the requested layout)
    cols = ["Date", "Time (IST)", "Symbol", "Product", "Side", "Qty", "Price", "Realized P&L"]
    order_rows = []
    for t in trades:
        other = SELL if t.side == BUY else BUY
        order_rows.append([t.entry_time.date(), t.entry_time.time().replace(second=0, microsecond=0), t.symbol,
                           PRODUCT, t.side, t.qty, float(t.entry_price), None])
        order_rows.append([t.exit_time.date(), t.exit_time.time().replace(second=0, microsecond=0), t.symbol,
                           PRODUCT, other, t.qty, float(t.exit_price), float(t.pnl)])

    def write_order_sheet(ws, rows):
        ws.append(cols)
        style_header(ws)
        for r in rows:
            ws.append(r)
        for row in ws.iter_rows(min_row=2):
            row[0].number_format = "yyyy-mm-dd"
            row[1].number_format = "hh:mm"
            row[6].number_format = "#,##0.00"
            row[7].number_format = MONEY
        widths(ws, [12, 11, 14, 11, 8, 7, 12, 14])
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

    ws = wb.active
    ws.title = "Trades"
    chunk = MAX_SHEET_ROWS - 1
    chunk -= chunk % 2                         # never split a trade across sheets
    write_order_sheet(ws, order_rows[:chunk])
    for part, start in enumerate(range(chunk, len(order_rows), chunk), 2):
        write_order_sheet(wb.create_sheet(f"Trades ({part})"), order_rows[start:start + chunk])

    # ------------------------------------------------------------ Trade details
    wd = wb.create_sheet("Trade details")
    wd.append(["Trade #", "Date", "Symbol", "Setup", "Side", "Qty", "Level", "Level price",
               "Cross candle", "Confirm candle", "Entry time", "Entry price", "Stop loss", "Target",
               "Exit time", "Exit price", "Exit reason", "P&L", "P&L %", "Minutes held"])
    style_header(wd)
    for no, t in enumerate(trades, 1):
        wd.append([no, t.day, t.symbol, t.setup, t.side, t.qty, t.level_name, t.level,
                   t.cross_time.time().replace(second=0, microsecond=0),
                   t.confirm_time.time().replace(second=0, microsecond=0),
                   t.entry_time.time().replace(second=0, microsecond=0), float(t.entry_price),
                   float(t.stop_loss), float(t.target),
                   t.exit_time.time().replace(second=0, microsecond=0), float(t.exit_price),
                   t.exit_reason, float(t.pnl), t.pnl_pct, round(t.minutes_held, 1)])
    for row in wd.iter_rows(min_row=2):
        row[1].number_format = "yyyy-mm-dd"
        for i in (8, 9, 10, 14):
            row[i].number_format = "hh:mm"
        for i in (7, 11, 12, 13, 15):
            row[i].number_format = "#,##0.00"
        row[17].number_format = MONEY
        row[18].number_format = PCT
    widths(wd, [8, 12, 14, 40, 7, 6, 7, 11, 11, 13, 11, 11, 11, 11, 11, 11, 22, 12, 9, 12])
    wd.freeze_panes = "D2"
    wd.auto_filter.ref = wd.dimensions

    # ------------------------------------------------------------ Summary
    sm = wb.create_sheet("Summary")
    sm.column_dimensions["A"].width = 34
    for c in "BCDEFGHIJKLMN":
        sm.column_dimensions[c].width = 15
    r = 1

    def title(text):
        nonlocal r
        sm.cell(r, 1, text).font = Font(bold=True, size=13, color=NAVY)
        r += 1

    def kv(label, value, fmt=None):
        nonlocal r
        sm.cell(r, 1, label)
        cell = sm.cell(r, 2, value)
        if fmt and value is not None:
            cell.number_format = fmt
        cell.alignment = Alignment(horizontal="right")
        r += 1

    def table(headers, rows, formats=None):
        nonlocal r
        for j, h in enumerate(headers, 1):
            sm.cell(r, j, h)
        style_header(sm, r)
        r += 1
        for row in rows:
            for j, v in enumerate(row, 1):
                cell = sm.cell(r, j, v)
                if formats and formats.get(j - 1) and v is not None:
                    cell.number_format = formats[j - 1]
            r += 1
        r += 1

    title("PDH / PDL pattern - research summary")
    sm.cell(r, 1, "Historical simulation only - no brokerage, taxes or slippage; not a prediction of future results.").font = Font(italic=True, color="64748B")
    r += 2

    title("Rules used")
    kv("Long", "Candle crosses above previous-day high, next candle red -> BUY")
    kv("Short", "Candle crosses below previous-day low, next candle green -> SELL")
    kv("Quantity", P.qty)
    kv("Stop loss %", float(P.sl_pct))
    kv("Target %", float(P.target_pct))
    kv("A 'cross' is detected on", "candle close" if P.cross_on == "close" else "candle high / low (wick)")
    kv("Entry fill", "close of the confirming candle" if P.entry == "close" else "open of the following candle")
    kv("End-of-day exit", f"{P.squareoff:%H:%M}" if P.squareoff else "close of the day's last candle")
    kv("If SL and target are in one candle", "stop loss assumed first")
    kv("Date filter", f"{args.date_from or 'start'} to {args.date_to or 'end'}")
    r += 1

    title("Data coverage")
    kv("Symbols in scope", stats.symbols)
    kv("Symbols with candles", stats.symbols_with_data)
    kv("5-minute candles read", stats.candles, "#,##0")
    kv("Trading days scanned", stats.days_scanned, "#,##0")
    kv("First / last day scanned", f"{stats.first_day or '-'} / {stats.last_day or '-'}")
    kv("Days skipped: no previous day in data", stats.days_no_previous)
    kv("Days skipped: previous day too far back", stats.days_gap)
    kv("Days skipped: previous day too few candles", stats.days_prev_short)
    r += 1

    title("Signal funnel (crosses seen while no position was open)")
    kv("PDH crosses", stats.pdh_crosses)
    kv("  ...followed by a red candle", stats.pdh_confirmed)
    kv("PDL crosses", stats.pdl_crosses)
    kv("  ...followed by a green candle", stats.pdl_confirmed)
    kv("Trades taken", len(trades))
    kv("Skipped: no candle left to exit on", stats.no_exit_candle)
    kv("Skipped: after the square-off time", stats.after_squareoff)
    kv("Skipped: stop/target rounded onto entry", stats.bad_levels)
    kv("Skipped: one candle crossed both levels", stats.ambiguous_cross)
    kv("Trades where SL and target were in the same candle", stats.both_levels_same_candle)
    r += 1

    overall = summarize(trades)
    title("Overall result")
    fmts = {"Win rate %": PCT, "Total P&L": MONEY, "Average P&L per trade": MONEY, "Average winner": MONEY,
            "Average loser": MONEY, "Profit factor": "0.00", "Best trade": MONEY, "Worst trade": MONEY,
            "Max drawdown": MONEY, "Average minutes held": "0.0"}
    for k, v in overall.items():
        kv(k, v if v is not None else "n/a", fmts.get(k))
    r += 1

    metric_cols = ["Trades", "Winners", "Win rate %", "Total P&L", "Average P&L per trade", "Profit factor"]
    metric_fmt = {2: PCT, 3: MONEY, 4: MONEY, 5: "0.00"}

    def breakdown(label, header, keyfunc, order=None):
        groups = group_rows(trades, keyfunc)
        keys = order or sorted(groups)
        rows = []
        for key in keys:
            if key in groups:
                s = summarize(groups[key])
                rows.append([key] + [s[m] for m in metric_cols])
        title(label)
        table([header] + metric_cols, rows, metric_fmt)

    breakdown("Result by setup", "Setup", lambda t: t.setup)
    breakdown("Result by exit reason", "Exit reason", lambda t: t.exit_reason)
    breakdown("Result by month", "Month", lambda t: t.day.strftime("%Y-%m"))
    breakdown("Result by weekday", "Weekday", lambda t: t.day.strftime("%A"),
              ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    breakdown("Result by entry hour", "Entry hour", lambda t: f"{t.entry_time.hour:02d}:00")

    # ------------------------------------------------------------ By symbol
    wsy = wb.create_sheet("By symbol")
    wsy.append(["Symbol", "Trades", "Winners", "Losers", "Win rate %", "Total P&L", "Average P&L",
                "Long (PDH) trades", "Short (PDL) trades", "Best trade", "Worst trade"])
    style_header(wsy)
    rows = []
    for sym, ts in group_rows(trades, lambda t: t.symbol).items():
        s = summarize(ts)
        rows.append([sym, s["Trades"], s["Winners"], s["Losers"], s["Win rate %"], s["Total P&L"],
                     s["Average P&L per trade"], sum(1 for t in ts if t.side == BUY),
                     sum(1 for t in ts if t.side == SELL), s["Best trade"], s["Worst trade"]])
    for row in sorted(rows, key=lambda x: -x[5]):
        wsy.append(row)
    for row in wsy.iter_rows(min_row=2):
        row[4].number_format = PCT
        for i in (5, 6, 9, 10):
            row[i].number_format = MONEY
    widths(wsy, [14, 8, 9, 8, 11, 13, 13, 17, 17, 12, 12])
    wsy.freeze_panes = "B2"
    wsy.auto_filter.ref = wsy.dimensions

    # ------------------------------------------------------------ Daily P&L
    wdp = wb.create_sheet("Daily P&L")
    wdp.append(["Date", "Trades", "Winners", "P&L", "Cumulative P&L"])
    style_header(wdp)
    cum = Decimal(0)
    daily = group_rows(trades, lambda t: t.day)
    for day in sorted(daily):
        ts = daily[day]
        day_pnl = sum((t.pnl for t in ts), Decimal(0))
        cum += day_pnl
        wdp.append([day, len(ts), sum(1 for t in ts if t.pnl > 0), float(day_pnl), float(cum)])
    for row in wdp.iter_rows(min_row=2):
        row[0].number_format = "yyyy-mm-dd"
        row[3].number_format = MONEY
        row[4].number_format = MONEY
    widths(wdp, [12, 8, 9, 13, 16])
    wdp.freeze_panes = "A2"
    if wdp.max_row > 2:
        chart = LineChart()
        chart.title = "Cumulative P&L"
        chart.height, chart.width = 8.5, 20
        chart.add_data(Reference(wdp, min_col=5, min_row=1, max_row=wdp.max_row), titles_from_data=True)
        chart.set_categories(Reference(wdp, min_col=1, min_row=2, max_row=wdp.max_row))
        chart.legend = None
        wdp.add_chart(chart, "G2")

    wb.active = 0                                   # the file opens on the Trades sheet
    return wb


# ==========================================================================
# command line
# ==========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Find the PDH / PDL pattern in all historical 5-minute data and write an Excel file.")
    p.add_argument("--output", "-o", default="pdh_pdl_backtest.xlsx", help="Excel file to write (default: script/output/pdh_pdl_backtest.xlsx)")
    p.add_argument("--symbols", help="comma-separated Company codes (default: all)")
    p.add_argument("--from", dest="date_from", type=lambda s: dt.date.fromisoformat(s), help="first day to trade, YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", type=lambda s: dt.date.fromisoformat(s), help="last day to trade, YYYY-MM-DD")
    p.add_argument("--qty", type=int, default=10, help="quantity per trade (default: 10)")
    p.add_argument("--sl", type=Decimal, default=Decimal("1"), help="stop loss, percent of entry price (default: 1)")
    p.add_argument("--target", type=Decimal, default=Decimal("2"), help="target, percent of entry price (default: 2)")
    p.add_argument("--cross-on", choices=["close", "high"], default="close",
                   help="'close' (default): candle closes beyond the level; 'high': any wick beyond it (high for PDH, low for PDL)")
    p.add_argument("--entry", choices=["close", "next-open"], default="close",
                   help="'close' (default): fill at the close of the confirming candle; 'next-open': at the open of the one after it")
    p.add_argument("--squareoff", type=lambda s: dt.datetime.strptime(s, "%H:%M").time(),
                   help="close open trades from this IST time, e.g. 15:15 (default: end of the day's data)")
    p.add_argument("--max-gap-days", type=int, default=7, help="skip a day if the previous day in your data is further back than this (default: 7)")
    p.add_argument("--min-prev-candles", type=int, default=0,
                   help="skip a day if the previous day has fewer 5-minute candles; a full NSE day has 75 (default: 0 = no check)")
    p.add_argument("--settings", help="Django settings module (default: found from manage.py)")
    p.add_argument("--app", default="market", help="Django app that holds Company and FiveMinuteData (default: stocks)")
    args = p.parse_args(argv)
    if args.qty < 1:
        p.error("--qty must be at least 1")
    if not (0 < args.sl < 100):
        p.error("--sl must be between 0 and 100")
    if args.target <= 0:
        p.error("--target must be above 0")
    if args.date_from and args.date_to and args.date_from > args.date_to:
        p.error("--from is after --to")
    return args


def main(argv=None):
    args = parse_args(argv)
    P = Params(qty=args.qty, sl_pct=args.sl, target_pct=args.target, cross_on=args.cross_on,
               entry=args.entry, squareoff=args.squareoff, max_gap_days=args.max_gap_days,
               min_prev_candles=args.min_prev_candles)
    print("PDH / PDL pattern scan")
    trades, stats = load_and_run(args, P)
    wb = build_workbook(trades, stats, P, args)
    args.output = output_path(args.output)
    wb.save(args.output)

    s = summarize(trades)
    print(f"\n  trades: {s['Trades']}   winners: {s['Winners']}   losers: {s['Losers']}")
    if s["Trades"]:
        print(f"  win rate: {s['Win rate %']:.1f}%   total P&L: {s['Total P&L']:,.2f}   "
              f"avg/trade: {s['Average P&L per trade']:,.2f}   max drawdown: {s['Max drawdown']:,.2f}")
    else:
        print("  no trades found for these settings (see the Summary sheet for the signal funnel)")
    print(f"  written: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

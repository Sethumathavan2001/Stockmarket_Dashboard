#!/usr/bin/env python3
"""
intraday_strategies_research.py - back-test the six intraday strategies of the
Paper Trading page on ALL your historical 5-minute data and write an Excel
report: win rate, average result, profit factor, drawdown, stability over time,
best time of day, long vs short, effect of charges, best stocks ... and a
verdict per strategy on whether it is usable for intraday trading.

    1. Opening Range Breakout   close beyond the 09:15-09:30 range
    2. VWAP trend pullback      trend day, dip to VWAP, bullish/bearish close
    3. RSI reversal at S/R      RSI leaves oversold/overbought at a known level
    4. Inside-bar breakout      break of an inside bar, with the trend
    5. Breakout and retest      break of PDH/PDL/opening range, retest, hold
    6. Supertrend + ADX         Supertrend flip while ADX >= 25

The detection, entry, stop, exit and charges come straight from
paper_trading/signals.py, so this measures exactly what the page shows:
signal on a candle's close, entry at the next candle's open, stop / target /
15:15 square-off, stop assumed first if both are hit in one candle, estimated
intraday charges on Rs 1,00,000 per trade.  Results are in R (multiples of
the risk taken), after charges.

Each strategy is measured at targets of 1, 1.5 and 2 R, with and without the
money-flow filter (last 3 candles >= 1.2x usual volume and MFI agreeing).
"Walk-forward" replays the page's own 'Qualified' rule: on every day, a
strategy counts as qualified for a stock only from that stock's previous 60
trading days (>= 10 trades, >= 50 % win, positive average) - the honest test
of whether the label predicts anything.

Only liquid stock-days are traded (price >= --min-price, 20-day average daily
turnover >= --min-turnover crore); earlier days still feed the indicators.
Stocks are processed in parallel worker processes (database only - Yahoo is
not used).

USAGE (from the project folder; output goes to script/output/)
    python script/intraday_strategies_research.py
    python script/intraday_strategies_research.py --symbols TCS.NS,INFY.NS --from 2026-03-01
    python script/intraday_strategies_research.py --workers 2 --csv
Historical simulation only - not a promise of future results, not advice.
"""
import argparse
import datetime as dt
import os
import sys
import time as _time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed

# This script lives in <project>/script/; Excel output goes to <project>/script/output/.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def output_path(name):
    """A bare file name goes into script/output/; a path with a folder is used as given."""
    path = name if os.path.dirname(name) else os.path.join(OUTPUT_DIR, name)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


CRORE = 1e7
TARGETS = (1.0, 1.5, 2.0)
TCOL = {1.0: "10", 1.5: "15", 2.0: "20"}       # column suffix per target
OPEN_MIN = 9 * 60 + 15
REASONS = ["Target", "Stop loss", "Square-off", "Day end"]


# ==========================================================================
# worker (one stock) - runs in a separate process
# ==========================================================================
def _init_django(settings_module):
    sys.path.insert(0, PROJECT_ROOT)
    os.environ["DJANGO_SETTINGS_MODULE"] = settings_module
    import django
    django.setup()


def _load_days(company_id):
    from market.models import FiveMinuteData
    from paper_trading import services as svc
    from paper_trading import signals as sg
    rows = (FiveMinuteData.objects.filter(company_id=company_id).order_by("timestamp")
            .values_list("timestamp", "open", "high", "low", "close", "volume"))
    days, cur, bucket = [], None, []
    for ts, o, h, l, c, v in rows.iterator(chunk_size=20000):
        if None in (o, h, l, c):
            continue
        ts = svc.to_ist(ts)
        if ts.date() != cur:
            if bucket:
                days.append(bucket)
            cur, bucket = ts.date(), []
        bucket.append(sg.Bar(ts, float(o), float(h), float(l), float(c), float(v or 0)))
    if bucket:
        days.append(bucket)
    return days


def _short_reason(r):
    return "Target" if r.startswith("Target") else "Stop loss" if r.startswith("Stop") else \
        "Square-off" if r.startswith("Square") else "Day end"


def _walk_forward(items, hist_days, sg):
    """
    The page's 'Qualified' decision for each trade of ONE strategy, made from the stock's previous
    `hist_days` trading days only.  items: [(day_no, flow_ok, r_by_target)] in time order.
    Returns [(qualified, target, flow_only)] in the same order.
    """
    out, cache, lo = [], {}, 0
    for idx, (d, _f, _r) in enumerate(items):
        if d not in cache:
            while items[lo][0] < d - hist_days:
                lo += 1
            window = [x for x in items[lo:idx] if x[0] < d]
            best = None
            for flow_only in (False, True):
                pool = [x for x in window if x[1] or not flow_only]
                n = len(pool)
                if n < sg.MIN_TRADES:
                    continue
                for ti, t in enumerate(TARGETS):
                    rs = [x[2][ti] for x in pool]
                    win = sum(1 for r in rs if r > 0) / n * 100
                    avg = sum(rs) / n
                    if win >= sg.MIN_WIN_RATE and avg > 0 and (best is None or (avg, win) > best[0]):
                        best = ((avg, win), t, flow_only)
            cache[d] = (True, best[1], best[2]) if best else (False, float("nan"), False)
        out.append(cache[d])
    return out


def run_stock(company_id, code, opts, days=None):
    """
    All signals of the six strategies for one stock -> (code, DataFrame of liquid-day trades, candles).
    `days` (lists of signals.Bar per day) can be passed in when the caller has already loaded them.
    """
    import pandas as pd
    from paper_trading import signals as sg
    if days is None:
        days = _load_days(company_id)
    n_bars = sum(len(d) for d in days)
    if len(days) < 2:
        return code, None, n_bars
    date_from, date_to = opts["date_from"], opts["date_to"]

    # liquid days: previous close >= min price and 20-day avg turnover >= min
    eligible = [False] * len(days)
    turnover = [0.0] * len(days)
    hist = deque(maxlen=20)
    for d, day in enumerate(days):
        if d and len(hist) >= 10:
            avg_cr = sum(hist) / len(hist) / CRORE
            turnover[d] = avg_cr
            day_date = day[0].ts.date()
            in_range = (date_from is None or day_date >= date_from) and (date_to is None or day_date <= date_to)
            eligible[d] = in_range and days[d - 1][-1].c >= opts["min_price"] and avg_cr >= opts["min_turnover"]
        hist.append(sum(b.c * b.v for b in day))

    F = sg.build_features(days)
    sigs = sg.find_signals(F, list(range(1, len(days))))      # day order, so per strategy in time order

    cols = defaultdict(list)
    wf_items = defaultdict(list)
    wf_pos = []                                               # (strategy, index in its list) per signal
    for s in sigs:
        d = F.day_of[s.k]
        entry_ts = F.bars[s.k + 1].ts
        rs = []
        for t in TARGETS:
            res = sg.simulate(F, s, t)
            rs.append(res["r_net"])
            c = TCOL[t]
            cols["r" + c].append(res["r_net"])
            cols["x" + c].append(_short_reason(res["reason"]))
            cols["h" + c].append((F.bars[res["exit_k"]].ts - entry_ts).total_seconds() / 60)
            if t == 1.5:
                cols["gross15"].append(res["r_gross"])
        end = F.day_start[d + 1] if d + 1 < len(F.day_start) else len(F.bars)
        mfe = 0.0                      # best move in the trade's favour before the square-off, in R
        for j in range(s.k + 1, end):
            b = F.bars[j]
            if b.ts.time() >= sg.SQUAREOFF:
                break
            mfe = max(mfe, (b.h - s.entry) if s.side > 0 else (s.entry - b.l))
        sig_start = F.bars[s.k].ts
        cols["date"].append(F.bars[s.k].ts.date())
        cols["strat"].append(s.strategy)
        cols["side"].append(s.side)
        cols["sig_m"].append(sig_start.hour * 60 + sig_start.minute)
        cols["entry_m"].append(entry_ts.hour * 60 + entry_ts.minute)
        cols["entry"].append(s.entry)
        cols["stop"].append(s.stop)
        cols["risk"].append(s.risk / s.entry * 100)
        cols["flow"].append(s.flow)
        cols["mfi"].append(s.mfi)
        cols["flow_ok"].append(s.flow_ok)
        cols["turn"].append(turnover[d])
        cols["mfe"].append(mfe / s.risk)
        cols["elig"].append(eligible[d])
        wf_pos.append((s.strategy, len(wf_items[s.strategy])))
        wf_items[s.strategy].append((d, s.flow_ok, rs))

    if not sigs:
        return code, None, n_bars
    wf = {k: _walk_forward(v, sg.HISTORY_DAYS, sg) for k, v in wf_items.items()}
    q = [wf[k][i] for k, i in wf_pos]
    cols["wfq"] = [x[0] for x in q]
    cols["wft"] = [x[1] for x in q]
    cols["wff"] = [x[2] for x in q]

    df = pd.DataFrame(cols)
    df = df[df["elig"]].drop(columns="elig")
    if df.empty:
        return code, None, n_bars
    df.insert(0, "sym", code)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("side",):
        df[c] = df[c].astype("int8")
    for c in ("sig_m", "entry_m", "h10", "h15", "h20"):
        df[c] = df[c].astype("int16")
    for c in ("entry", "stop", "risk", "flow", "mfi", "turn", "mfe", "r10", "r15", "r20", "gross15", "wft"):
        df[c] = df[c].astype("float32")
    for c in ("strat", "x10", "x15", "x20"):
        df[c] = df[c].astype("category")
    return code, df, n_bars


# ==========================================================================
# statistics
# ==========================================================================
METRICS = ["Trades", "Win rate %", "Avg net R", "Total net R", "Profit factor", "Avg win R", "Avg loss R",
           "Max drawdown R"]


def stats(rs):
    """rs: R results in time order (numpy array or sequence)."""
    import numpy as np
    rs = np.asarray(rs, dtype="float64")
    n = len(rs)
    if not n:
        return {"Trades": 0}
    wins, losses = rs[rs > 0], rs[rs < 0]
    gw, gl = wins.sum(), -losses.sum()
    cum = np.cumsum(rs)
    dd = float(np.max(np.maximum.accumulate(np.maximum(cum, 0)) - cum))
    return {
        "Trades": n,
        "Win rate %": len(wins) / n * 100,
        "Avg net R": float(rs.mean()),
        "Total net R": float(rs.sum()),
        "Profit factor": float(gw / gl) if gl else None,
        "Avg win R": float(wins.mean()) if len(wins) else None,
        "Avg loss R": float(losses.mean()) if len(losses) else None,
        "Max drawdown R": dd,
    }


def verdict(full, first, second, min_trades):
    if full["Trades"] < min_trades:
        return "Too few trades"
    a, h1, h2 = full["Avg net R"], first.get("Avg net R"), second.get("Avg net R")
    if a > 0.05 and h1 is not None and h2 is not None and h1 > 0 and h2 > 0:
        return "SUITABLE - positive in both halves of the data"
    if a > 0:
        return "MARGINAL - positive overall but not stable"
    if a > -0.1:
        return "NOT SUITABLE - about break-even after charges"
    return "NOT SUITABLE - loses money after charges"


def slot_label(minutes):
    s0 = OPEN_MIN + ((minutes - OPEN_MIN) // 30) * 30
    return f"{s0 // 60:02d}:{s0 % 60:02d}-{(s0 + 30) // 60:02d}:{(s0 + 30) % 60:02d}"


# ==========================================================================
# Excel
# ==========================================================================
def build_workbook(df, info, args):
    import numpy as np
    from openpyxl import Workbook
    from openpyxl.chart import LineChart, Reference
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from paper_trading import signals as sg

    NAVY, MUTED = "123B7A", "64748B"
    head_font, head_fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor=NAVY)
    PCT, R2, R1 = '0.0"%"', '0.00" R";[Red]-0.00" R"', '0.0" R";[Red]-0.0" R"'
    FMT = {"Win rate %": PCT, "Avg net R": R2, "Total net R": R1, "Profit factor": "0.00",
           "Avg win R": R2, "Avg loss R": R2, "Max drawdown R": '0.0" R"', "Avg gross R": R2, "Charges R": R2,
           "Avg risk %": '0.00"%"', "Avg MFE R": "0.00", "Reach 1R %": PCT, "Reach 1.5R %": PCT,
           "Reach 2R %": PCT, "Avg minutes held": "0", "In-sample avg R": R2, "Out-of-sample avg R": R2,
           "Profitable months %": PCT}
    scale = dict(start_type="num", start_value=-0.3, start_color="F8696B", mid_type="num", mid_value=0,
                 mid_color="FFFFFF", end_type="num", end_value=0.3, end_color="63BE7B")
    NAMES = {k: n for k, n, _ in sg.STRATEGIES}
    KEYS = [k for k, _, _ in sg.STRATEGIES]
    REF = "r" + TCOL[args.ref_r]
    df = df.sort_values(["date", "entry_m", "sym"], kind="stable").reset_index(drop=True)
    dates = df["date"].drop_duplicates().sort_values().tolist()
    mid = dates[len(dates) // 2] if dates else None
    by_s = {k: g for k, g in df.groupby("strat", observed=True)}
    empty = df.iloc[0:0]
    wb = Workbook()

    def style_header(ws, row):
        for c in ws[row]:
            if c.value is not None:
                c.font, c.fill = head_font, head_fill
                c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    class Sheet:
        def __init__(self, title, first=30, rest=13):
            self.ws = wb.create_sheet(title)
            self.r = 1
            self.ws.column_dimensions["A"].width = first
            for k in range(2, 30):
                self.ws.column_dimensions[get_column_letter(k)].width = rest

        def title(self, text, size=12):
            self.ws.cell(self.r, 1, text).font = Font(bold=True, size=size, color=NAVY)
            self.r += 1

        def note(self, text):
            self.ws.cell(self.r, 1, text).font = Font(italic=True, color=MUTED)
            self.r += 1

        def table(self, header, rows, color_cols=("Avg net R",), fmts=None):
            fm = dict(FMT, **(fmts or {}))
            for j, h in enumerate(header, 1):
                self.ws.cell(self.r, j, h)
            style_header(self.ws, self.r)
            top = self.r = self.r + 1
            for row in rows:
                for j, v in enumerate(row, 1):
                    if isinstance(v, (np.floating, np.integer)):
                        v = v.item()
                    if isinstance(v, float) and v != v:            # NaN -> blank
                        v = None
                    c = self.ws.cell(self.r, j, v)
                    f = fm.get(header[j - 1])
                    if f and isinstance(v, (int, float)) and not isinstance(v, bool):
                        c.number_format = f
                self.r += 1
            if rows:
                for col in color_cols:
                    if col in header:
                        L = get_column_letter(header.index(col) + 1)
                        self.ws.conditional_formatting.add(f"{L}{top}:{L}{self.r - 1}", ColorScaleRule(**scale))
            self.r += 1

        def matrix(self, label, groups, cols, metric, fmt):
            """groups: {(strategy, col): R array}"""
            self.title(label, 11)
            body = []
            for k in KEYS:
                row = [NAMES[k]]
                for c in cols:
                    rs = groups.get((k, c))
                    if rs is None or not len(rs):
                        row.append(None)
                    else:
                        row.append(len(rs) if metric == "Trades" else stats(rs).get(metric))
                body.append(row)
            self.table([""] + list(cols), body, color_cols=cols if fmt == R2 else (),
                       fmts={c: fmt for c in cols})

    def grouped(col_values):
        """{(strategy, value): REF results} for a per-trade key column."""
        out = {}
        for (k, v), g in df.groupby([df["strat"], col_values], observed=True, sort=False):
            out[(k, v)] = g[REF].to_numpy()
        return out

    # ------------------------------------------------------------ Read me
    rm = wb.active
    rm.title = "Read me"
    rm.column_dimensions["A"].width = 125
    text = [
        ("Six intraday strategies - historical back-test", "t"),
        ("Historical simulation on your 5-minute data. Not a prediction of future results, not advice.", "n"),
        ("", None),
        ("WHAT WAS TESTED", "h"),
        ("The six strategies of the Paper Trading page, with the page's own code (paper_trading/signals.py):", None),
        *[(f"   {i}. {n} ({kind})", None) for i, (_, n, kind) in enumerate(sg.STRATEGIES, 1)],
        (f"Signal on a candle's close between {sg.FIRST_SIGNAL:%H:%M} and {sg.LAST_SIGNAL:%H:%M}; entry at the NEXT candle's "
         f"open; exit at the stop, the target or the {sg.SQUAREOFF:%H:%M} square-off. Stop first if both are hit in one candle.", None),
        (f"Stops at least {sg.MIN_RISK_PCT:g} % away (widened if tighter); signals needing more than {sg.MAX_RISK_PCT:g} % "
         f"are skipped. At most {sg.MAX_PER_DAY} signals per strategy per stock per day, {sg.COOLDOWN} candles apart.", None),
        (f"Charges: estimated intraday costs (brokerage, STT, exchange, SEBI, stamp, GST) on Rs {sg.CAPITAL:,} per trade.", None),
        (f"Only liquid stock-days: previous close >= Rs {args.min_price:g} and 20-day average daily turnover >= "
         f"Rs {args.min_turnover:g} crore.", None),
        ("", None),
        ("HOW TO READ IT", "h"),
        ("R = profit or loss in multiples of the risk (entry to stop), AFTER charges. -1 R = a full stop; +1.5 R = a 1.5 R target.", None),
        ("Avg net R is the number that decides: above 0 the strategy made money after charges. Colour: red < 0 < green.", None),
        ("Win rate alone misleads: with 2 R targets 40 % wins can be profitable; with 1 R targets you need well over 50 %.", None),
        ("Profit factor = money won / money lost (above 1 = profitable). Max drawdown = worst peak-to-trough fall of the R curve.", None),
        (f"Breakdown sheets use the {args.ref_r:g} R target and all signals unless they say otherwise.", None),
        ("", None),
        ("SHEETS", "h"),
        ("Verdict            - one line per strategy: is it suitable for intraday? Best configuration, stability check.", None),
        ("All configurations - every strategy x target (1 / 1.5 / 2 R) x money-flow filter on / off.", None),
        ("Walk-forward       - does the page's 'Qualified' label (chosen from the previous 60 days) actually help?", None),
        ("Long vs short, Time of day, Month, Weekday, Exit reasons - where and when each strategy works.", None),
        ("Charges & MFE      - how much charges cost per trade, and how far trades went in your favour (target sizing).", None),
        ("Equity curves      - cumulative R per strategy over time, with a chart.", None),
        ("By symbol          - each strategy's result on each stock (20+ trades).", None),
        (f"Trades             - the latest {args.max_rows:,} trades with every detail (filterable). --csv writes all of them.", None),
    ]
    for k, (line, kind) in enumerate(text, 1):
        c = rm.cell(k, 1, line)
        if kind == "t":
            c.font = Font(bold=True, size=15, color=NAVY)
        elif kind == "h":
            c.font = Font(bold=True, size=12, color=NAVY)
        elif kind == "n":
            c.font = Font(italic=True, color=MUTED)

    # ------------------------------------------------------------ Verdict
    v = Sheet("Verdict", 26, 14)
    v.title("Are these strategies suitable for intraday trading?", 14)
    v.note(f"{len(df):,} trades on {info['symbols_traded']:,} stocks, {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}. "
           f"'Both halves' = first half of the dates vs second half (before / from {mid:%Y-%m-%d})." if dates else "No trades.")
    v.r += 1
    rows = []
    for k in KEYS:
        g = by_s.get(k, empty)
        base = stats(g[REF].to_numpy())
        best = None
        for flow_only in (False, True):
            pool = g[g["flow_ok"]] if flow_only else g
            for t in TARGETS:
                s = stats(pool["r" + TCOL[t]].to_numpy())
                if s["Trades"] >= args.min_trades and (best is None or s["Avg net R"] > best[0]["Avg net R"]):
                    best = (s, t, flow_only, pool)
        if best:
            s, t, fo, pool = best
            col = "r" + TCOL[t]
            h1 = stats(pool.loc[pool["date"] < mid, col].to_numpy())
            h2 = stats(pool.loc[pool["date"] >= mid, col].to_numpy())
            monthly = pool.groupby(pool["date"].dt.strftime("%Y-%m"))[col].sum()
            pm = float((monthly > 0).mean() * 100) if len(monthly) else None
            rows.append([NAMES[k], base["Trades"], base.get("Win rate %"), base.get("Avg net R"),
                         f"{t:g} R, money flow {'ON' if fo else 'off'}", s["Trades"], s["Win rate %"],
                         s["Avg net R"], s["Profit factor"], h1.get("Avg net R"), h2.get("Avg net R"), pm,
                         verdict(s, h1, h2, args.min_trades)])
        else:
            rows.append([NAMES[k], base["Trades"], base.get("Win rate %"), base.get("Avg net R"), "-", 0,
                         None, None, None, None, None, None, "Too few trades"])
    v.table(["Strategy", "Trades", "Win rate %", "Avg net R", "Best configuration", "Trades ", "Win rate % ",
             "Avg net R ", "Profit factor", "In-sample avg R", "Out-of-sample avg R", "Profitable months %",
             "Verdict"], rows, color_cols=("Avg net R", "Avg net R ", "In-sample avg R", "Out-of-sample avg R"),
            fmts={"Win rate % ": PCT, "Avg net R ": R2})
    v.ws.column_dimensions["E"].width = 24
    v.ws.column_dimensions["M"].width = 48
    v.note(f"Columns B-D: every signal at the {args.ref_r:g} R target. Columns E-L: the best of the 6 configurations "
           "(picked with hindsight on ALL data - check the In-sample / Out-of-sample columns before trusting it).")
    v.note("SUITABLE needs a positive average in BOTH halves of the data. 'Walk-forward' shows how the page's own "
           "per-stock choice did on unseen days.")

    # ------------------------------------------------------------ All configurations
    ac = Sheet("All configurations", 26)
    ac.title("Every strategy x target x money-flow filter", 13)
    ac.r += 1
    rows = []
    for k in KEYS:
        g = by_s.get(k, empty)
        for flow_only in (False, True):
            pool = g[g["flow_ok"]] if flow_only else g
            for t in TARGETS:
                s = stats(pool["r" + TCOL[t]].to_numpy())
                rows.append([NAMES[k], f"{t:g} R", "ON" if flow_only else "off"] + [s.get(m) for m in METRICS])
    ac.table(["Strategy", "Target", "Money flow"] + METRICS, rows)

    # ------------------------------------------------------------ Walk-forward
    wfs = Sheet("Walk-forward", 30)
    wfs.title("Does the page's 'Qualified' label help on unseen days?", 13)
    wfs.note("For every trade, the strategy was judged on that stock's previous 60 trading days only (>= 10 trades, "
             ">= 50 % win, positive average) - exactly what the page does. Qualified trades use the target and filter "
             "chosen from that history.")
    wfs.r += 1

    def qualified_r(g):
        q = g[g["wfq"] & (g["flow_ok"] | ~g["wff"])]
        if q.empty:
            return np.array([])
        pick = np.select([q["wft"] == 1.0, q["wft"] == 1.5], [q["r10"], q["r15"]], q["r20"])
        return pick

    rows, allq, allu = [], [], []
    for k in KEYS:
        g = by_s.get(k, empty)
        q, u = qualified_r(g), g.loc[~g["wfq"], REF].to_numpy()
        allq.append(q)
        allu.append(u)
        for label, rs in (("Qualified", q), ("Not qualified", u)):
            s = stats(rs)
            rows.append([NAMES[k], label] + [s.get(m) for m in METRICS[:5]])
    for label, parts in (("Qualified", allq), ("Not qualified", allu)):
        s = stats(np.concatenate(parts) if parts else [])
        rows.append(["ALL STRATEGIES", label] + [s.get(m) for m in METRICS[:5]])
    wfs.table(["Strategy", "Page label"] + METRICS[:5], rows)
    wfs.note("If 'Qualified' is not clearly better than 'Not qualified', a good past record on a stock does not "
             "predict the next days - treat the label as information, not as a filter you can rely on.")

    # ------------------------------------------------------------ Long vs short
    ls = Sheet("Long vs short", 26)
    ls.title(f"Long vs short ({args.ref_r:g} R target, all signals)", 13)
    ls.r += 1
    rows = []
    for k in KEYS:
        g = by_s.get(k, empty)
        for side, label in ((1, "Long (BUY)"), (-1, "Short (SELL)")):
            s = stats(g.loc[g["side"] == side, REF].to_numpy())
            rows.append([NAMES[k], label] + [s.get(m) for m in METRICS])
    ls.table(["Strategy", "Side"] + METRICS, rows)

    # ------------------------------------------------------------ Time of day
    slot = df["entry_m"].map(slot_label)
    slots = sorted(slot.unique())
    tg = grouped(slot)
    tod = Sheet("Time of day", 26, 12)
    tod.title(f"By entry time - 30-minute slots ({args.ref_r:g} R target)", 13)
    tod.r += 1
    tod.matrix("Avg net R", tg, slots, "Avg net R", R2)
    tod.matrix("Win rate %", tg, slots, "Win rate %", PCT)
    tod.matrix("Trades", tg, slots, "Trades", "#,##0")

    # ------------------------------------------------------------ Month / weekday
    month = df["date"].dt.strftime("%Y-%m")
    months = sorted(month.unique())
    mg = grouped(month)
    mo = Sheet("Month", 26, 10)
    mo.title(f"Stability over time - by month ({args.ref_r:g} R target)", 13)
    mo.note("A strategy you can rely on is green in most months, not just in one lucky stretch.")
    mo.r += 1
    mo.matrix("Avg net R", mg, months, "Avg net R", R2)
    mo.matrix("Total net R", mg, months, "Total net R", R1)
    mo.matrix("Trades", mg, months, "Trades", "#,##0")
    wg = grouped(df["date"].dt.day_name())
    wd = Sheet("Weekday", 26, 12)
    wd.title(f"By weekday ({args.ref_r:g} R target)", 13)
    wd.r += 1
    days_ = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    wd.matrix("Avg net R", wg, days_, "Avg net R", R2)
    wd.matrix("Win rate %", wg, days_, "Win rate %", PCT)

    # ------------------------------------------------------------ Exit reasons
    ex = Sheet("Exit reasons", 26, 14)
    ex.title("How trades ended, per target", 13)
    ex.r += 1
    rows = []
    for k in KEYS:
        g = by_s.get(k, empty)
        n = len(g) or 1
        for t in TARGETS:
            c = TCOL[t]
            cnt = g["x" + c].value_counts()
            sq = g.loc[g["x" + c] == "Square-off", "r" + c]
            rows.append([NAMES[k], f"{t:g} R"] + [cnt.get(r, 0) / n * 100 for r in REASONS] +
                        [float(sq.mean()) if len(sq) else None, float(g["h" + c].mean()) if len(g) else None])
    ex.table(["Strategy", "Target", "Target %", "Stop loss %", "Square-off %", "Day end %", "Square-off avg R",
              "Avg minutes held"], rows, color_cols=("Square-off avg R",),
             fmts={"Target %": PCT, "Stop loss %": PCT, "Square-off %": PCT, "Day end %": PCT, "Square-off avg R": R2})

    # ------------------------------------------------------------ Charges & MFE
    cm = Sheet("Charges & MFE", 26, 14)
    cm.title("What charges cost, and how far trades ran in your favour", 13)
    cm.note(f"Gross vs net at the 1.5 R target. MFE = the best move in the trade's favour before {sg.SQUAREOFF:%H:%M}, "
            "in R (if most trades never reach 1.5 R, a 1.5 R target is too far).")
    cm.r += 1
    rows = []
    for k in KEYS:
        g = by_s.get(k, empty)
        if g.empty:
            rows.append([NAMES[k], 0] + [None] * 8)
            continue
        gr, nr = float(g["gross15"].mean()), float(g["r15"].mean())
        rows.append([NAMES[k], len(g), gr, nr, gr - nr, float(g["risk"].mean()), float(g["mfe"].mean()),
                     float((g["mfe"] >= 1).mean() * 100), float((g["mfe"] >= 1.5).mean() * 100),
                     float((g["mfe"] >= 2).mean() * 100)])
    cm.table(["Strategy", "Trades", "Avg gross R", "Avg net R", "Charges R", "Avg risk %", "Avg MFE R",
              "Reach 1R %", "Reach 1.5R %", "Reach 2R %"], rows, color_cols=("Avg gross R", "Avg net R"))

    # ------------------------------------------------------------ Equity curves
    eq = wb.create_sheet("Equity curves")
    eq.append(["Date"] + [NAMES[k] for k in KEYS])
    style_header(eq, 1)
    piv = df.pivot_table(index="date", columns="strat", values=REF, aggfunc="sum", observed=True).reindex(
        columns=KEYS).fillna(0).cumsum()
    for d, row in piv.iterrows():
        eq.append([d.date()] + [round(float(x), 2) for x in row.tolist()])
    for row in eq.iter_rows(min_row=2):
        row[0].number_format = "yyyy-mm-dd"
    eq.column_dimensions["A"].width = 12
    for k in range(2, 8):
        eq.column_dimensions[get_column_letter(k)].width = 16
    if eq.max_row > 2:
        ch = LineChart()
        ch.title = f"Cumulative net R per strategy ({args.ref_r:g} R target)"
        ch.height, ch.width = 11, 26
        ch.add_data(Reference(eq, min_col=2, max_col=7, min_row=1, max_row=eq.max_row), titles_from_data=True)
        ch.set_categories(Reference(eq, min_col=1, min_row=2, max_row=eq.max_row))
        eq.add_chart(ch, "I2")

    # ------------------------------------------------------------ By symbol
    bs = wb.create_sheet("By symbol")
    bs.append(["Symbol", "Strategy", "Trades", "Win rate %", "Avg net R", "Total net R", "Profit factor"])
    style_header(bs, 1)
    out = []
    for (sym, k), g in df.groupby(["sym", "strat"], observed=True, sort=False):
        if len(g) >= 20:
            s = stats(g[REF].to_numpy())
            out.append([sym, NAMES[k], s["Trades"], s["Win rate %"], s["Avg net R"], s["Total net R"], s["Profit factor"]])
    out.sort(key=lambda x: -x[4])
    for row in out:
        bs.append(row)
    for row in bs.iter_rows(min_row=2):
        row[3].number_format, row[4].number_format, row[5].number_format = PCT, R2, '0.0" R"'
        row[6].number_format = "0.00"
    if bs.max_row > 1:
        bs.conditional_formatting.add(f"E2:E{bs.max_row}", ColorScaleRule(**scale))
    for col, w in zip("ABCDEFG", (16, 26, 8, 11, 11, 12, 12)):
        bs.column_dimensions[col].width = w
    bs.freeze_panes = "A2"
    bs.auto_filter.ref = bs.dimensions

    # ------------------------------------------------------------ Trades
    tr = wb.create_sheet("Trades")
    hdr = TRADE_HEADER
    tr.append(hdr)
    style_header(tr, 1)
    for row in trade_rows(df.iloc[-args.max_rows:], NAMES):
        tr.append(row)
    for row in tr.iter_rows(min_row=2):
        row[0].number_format = "yyyy-mm-dd"
    for k, w in enumerate([11, 15, 24, 6, 7, 9, 10, 10, 8, 9, 6, 9, 10] + [9, 11, 9] * 3 + [11, 8, 10, 9], 1):
        tr.column_dimensions[get_column_letter(k)].width = w
    tr.freeze_panes = "D2"
    tr.auto_filter.ref = tr.dimensions

    wb.active = 1
    return wb


TRADE_HEADER = (["Date", "Symbol", "Strategy", "Side", "Signal candle", "Entry time", "Entry", "Stop", "Risk %", "Volume x",
                 "MFI", "Money flow OK", "Turnover Rs cr"] +
                [f"{c} @{t:g}R" for t in TARGETS for c in ("R", "Exit", "Minutes")] +
                ["Gross R @1.5R", "MFE R", "Page: qualified", "Page: target"])


def trade_rows(df, names):
    hm = lambda m: f"{m // 60:02d}:{m % 60:02d}"
    rnd = lambda x, k: None if x != x else round(float(x), k)
    for t in df.itertuples(index=False):
        row = [t.date.date(), t.sym, names[t.strat], "BUY" if t.side > 0 else "SELL", hm(t.sig_m), hm(t.entry_m),
               rnd(t.entry, 2), rnd(t.stop, 2), rnd(t.risk, 3), rnd(t.flow, 2), rnd(t.mfi, 1),
               "yes" if t.flow_ok else "no", rnd(t.turn, 1)]
        for c in ("10", "15", "20"):
            row += [rnd(getattr(t, "r" + c), 3), getattr(t, "x" + c), int(getattr(t, "h" + c))]
        row += [rnd(t.gross15, 3), rnd(t.mfe, 2), "yes" if t.wfq else "no", rnd(t.wft, 1)]
        yield row


# ==========================================================================
# main
# ==========================================================================
def find_settings():
    import re
    if os.environ.get("DJANGO_SETTINGS_MODULE"):
        return os.environ["DJANGO_SETTINGS_MODULE"]
    m = re.search(r"DJANGO_SETTINGS_MODULE['\"]\s*,\s*['\"]([\w.]+)['\"]", open(os.path.join(PROJECT_ROOT, "manage.py"), encoding="utf-8").read())
    return m.group(1) if m else None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Back-test the six Paper Trading strategies and write an Excel report.")
    p.add_argument("--output", "-o", default="intraday_strategies_research.xlsx")
    p.add_argument("--symbols", help="comma-separated company codes (default: all)")
    p.add_argument("--from", dest="date_from", type=dt.date.fromisoformat, help="first day to trade, YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", type=dt.date.fromisoformat, help="last day to trade, YYYY-MM-DD")
    p.add_argument("--min-price", type=float, default=50, help="skip stock-days below this price (default 50)")
    p.add_argument("--min-turnover", type=float, default=5,
                   help="skip stock-days with 20-day avg turnover below this, Rs crore (default 5)")
    p.add_argument("--ref-r", type=float, default=1.5, choices=TARGETS, help="target used in the breakdown sheets (default 1.5)")
    p.add_argument("--min-trades", type=int, default=100, help="trades needed for a verdict (default 100)")
    p.add_argument("--max-rows", type=int, default=150_000, help="latest trades written to the Trades sheet (default 150000)")
    p.add_argument("--csv", action="store_true", help="also write every trade to a .csv next to the Excel file")
    p.add_argument("--workers", type=int, default=4, help="parallel processes (default 4; lower it if memory is short)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    settings_module = find_settings()
    if not settings_module:
        sys.exit("Run this next to manage.py")
    _init_django(settings_module)
    import pandas as pd
    from market.models import Company
    from paper_trading import signals as sg

    qs = Company.objects.order_by("code").exclude(code__in=["NIFTYBEES.NS", "QNIFTY.NS"])
    if args.symbols:
        qs = qs.filter(code__in=[s.strip() for s in args.symbols.split(",") if s.strip()])
    companies = list(qs.values_list("id", "code"))
    opts = {"date_from": args.date_from, "date_to": args.date_to, "min_price": args.min_price,
            "min_turnover": args.min_turnover}
    print(f"Six-strategy back-test: {len(companies)} stocks, {args.workers} worker processes")

    frames, candles, traded, n_trades = [], 0, 0, 0
    started = _time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_django, initargs=(settings_module,)) as ex:
        futs = [ex.submit(run_stock, cid, code, opts) for cid, code in companies]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                code, part, n = f.result()
            except Exception as exc:
                print(f"  a stock failed: {exc}")
                continue
            candles += n
            if part is not None:
                frames.append(part)
                traded += 1
                n_trades += len(part)
            if k % 50 == 0 or k == len(futs):
                print(f"  [{k}/{len(futs)}] {candles:,} candles, {n_trades:,} trades, {_time.time() - started:.0f}s",
                      flush=True)
    if not frames:
        sys.exit("No trades found.")
    df = pd.concat(frames, ignore_index=True)
    frames.clear()
    df["strat"] = df["strat"].astype("category")
    for c in ("x10", "x15", "x20"):
        df[c] = df[c].astype(str).astype("category")

    print("  writing Excel ...")
    wb = build_workbook(df, {"symbols_traded": traded}, args)
    args.output = output_path(args.output)
    wb.save(args.output)
    if args.csv:
        import csv
        csv_path = os.path.splitext(args.output)[0] + "_trades.csv"
        names = {k: n for k, n, _ in sg.STRATEGIES}
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(TRADE_HEADER)
            w.writerows(trade_rows(df.sort_values(["date", "entry_m", "sym"]), names))
        print(f"  written: {os.path.abspath(csv_path)}")

    ref = "r" + TCOL[args.ref_r]
    print(f"\n  {len(df):,} trades  (at {args.ref_r:g} R target, after charges)")
    for key, g in df.groupby("strat", observed=True):
        s = stats(g[ref].to_numpy())
        print(f"    {key:<14} {s['Trades']:>9,}  win {s['Win rate %']:5.1f}%  avg {s['Avg net R']:+.3f} R")
    print(f"  written: {os.path.abspath(args.output)}  ({_time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()

"""
Strategy signals for the paper-trading chart: where to enter and where to exit.

Six intraday setups are detected on 5-minute candles.  Each one is first
back-tested on the stock's OWN recent history (the trading days before the
session day - nothing from the session day itself), and only setups that
historically won at least MIN_WIN_RATE % of their trades with a positive
average result after charges are "qualified".  The session day's signals are
then generated candle by candle with indicators that only look backwards, so a
signal at 10:05 uses nothing that happened after 10:05.

    1. Opening Range Breakout   close beyond the 09:15-09:30 range
    2. VWAP trend pullback      trend day, dip to VWAP, bullish/bearish close
    3. RSI reversal at S/R      RSI leaves oversold/overbought at a known level
    4. Inside-bar breakout      break of an inside bar, with the trend
    5. Breakout and retest      break of PDH/PDL/opening range, retest, hold
    6. Supertrend + ADX         Supertrend flip while ADX >= 25

"Money flow" filter (tested both ways per strategy): the last 3 candles traded
at least FLOW_MIN x their usual volume for those time slots, and the Money
Flow Index agrees with the direction.  Each strategy keeps the variant and the
target (in R) that did best historically.

Entry is the OPEN of the candle after the signal candle.  Exit: stop / target
touched (stop assumed first if both in one candle), a gap through a level, or
the square-off at 15:15.  Historical win rates are not a promise about the
next trade.
"""
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import List, Optional

from django.core.cache import cache

from market.models import FiveMinuteData

from . import services as svc

IST = svc.IST
CANDLE = timedelta(minutes=5)

HISTORY_DAYS = 60            # trading days back-tested before the session day
MIN_TRADES = 10              # fewer historical trades than this -> can't qualify
MIN_WIN_RATE = 50.0          # %
TARGETS_R = (1.0, 1.5, 2.0)  # targets tried, in multiples of the risk
FIRST_SIGNAL = time(9, 30)   # signal candle must END at or after this
LAST_SIGNAL = time(14, 30)   # ...and at or before this
SQUAREOFF = time(15, 15)
MIN_RISK_PCT = 0.25          # stop at least this far from entry (else widened) - keeps charges a small part of the risk
MAX_RISK_PCT = 2.5           # skip signals that need a wider stop
COOLDOWN = 6                 # candles before the same strategy may fire again
MAX_PER_DAY = 3              # signals per strategy per day
FLOW_MIN = 1.2               # money-flow filter: volume vs usual
CAPITAL = 100_000            # rupees per trade, only used to estimate charges

STRATEGIES = [
    ("orb", "Opening Range Breakout", "Trend-following"),
    ("vwap_pullback", "VWAP Trend Pullback", "Trend-following"),
    ("rsi_sr", "RSI Reversal at S/R", "Mean-reversion"),
    ("inside_bar", "Inside-Bar Breakout", "Volatility expansion"),
    ("retest", "Breakout and Retest", "Confirmation"),
    ("st_adx", "Supertrend + ADX", "Indicator-based"),
]
NAMES = {k: n for k, n, _ in STRATEGIES}
SHORT = {"orb": "ORB", "vwap_pullback": "VWAP", "rsi_sr": "RSI", "inside_bar": "IB", "retest": "RETEST",
         "st_adx": "ST"}


# ==========================================================================
# data
# ==========================================================================
@dataclass
class Bar:
    ts: datetime      # candle start, IST
    o: float
    h: float
    l: float
    c: float
    v: float          # 0 = not reported (Yahoo leaves the 09:15 candle at 0 most days)


def _history_bars(company, before_day, days=HISTORY_DAYS):
    """The `days` trading days of DB candles before `before_day`, oldest first, grouped by day."""
    start = datetime.combine(before_day - timedelta(days=days * 2 + 20), time.min, tzinfo=IST)
    end = datetime.combine(before_day, time.min, tzinfo=IST)
    rows = (FiveMinuteData.objects
            .filter(company=company, timestamp__gte=svc.to_db_dt(start), timestamp__lt=svc.to_db_dt(end))
            .order_by("timestamp").values_list("timestamp", "open", "high", "low", "close", "volume"))
    by_day = defaultdict(list)
    for ts, o, h, l, c, v in rows:
        if None in (o, h, l, c):
            continue
        ts = svc.to_ist(ts)
        by_day[ts.date()].append(Bar(ts, float(o), float(h), float(l), float(c), float(v or 0)))
    return [by_day[d] for d in sorted(by_day)[-days:]]


def _replay_day_bars(company, day):
    start, end = svc._day_bounds_db(day)
    rows = (FiveMinuteData.objects.filter(company=company, timestamp__gte=start, timestamp__lt=end)
            .order_by("timestamp").values_list("timestamp", "open", "high", "low", "close", "volume"))
    return [Bar(svc.to_ist(ts), float(o), float(h), float(l), float(c), float(v or 0))
            for ts, o, h, l, c, v in rows if None not in (o, h, l, c)]


def _live_day_bars(company, now=None):
    """Today's COMPLETED 5-minute candles from Yahoo (the forming candle is left out)."""
    df = svc.fetch_live_today(company)
    now = now or datetime.now(IST)
    out = []
    for r in df.itertuples(index=False):
        ts = svc.to_ist(r.timestamp.to_pydatetime())
        if ts + CANDLE > now:
            continue
        out.append(Bar(ts, float(r.open), float(r.high), float(r.low), float(r.close), float(getattr(r, "volume", 0) or 0)))
    return out


# ==========================================================================
# indicators (all causal: value at k uses bars 0..k only)
# ==========================================================================
def _rma(values, n):
    out, avg = [], None
    seed = []
    for x in values:
        if avg is None:
            seed.append(x)
            if len(seed) == n:
                avg = sum(seed) / n
            out.append(avg)
        else:
            avg = (avg * (n - 1) + x) / n
            out.append(avg)
    return out


def _ema(values, n):
    out, e, k = [], None, 2 / (n + 1)
    for x in values:
        e = x if e is None else e + k * (x - e)
        out.append(e)
    return out


@dataclass
class Features:
    bars: List[Bar]
    day_of: List[int]              # day number of each bar
    day_start: List[int]           # index of the first bar of each day (by day number)
    pos_in_day: List[int]
    vwap: List[float] = field(default_factory=list)
    ema20: List[float] = field(default_factory=list)
    rsi: List[Optional[float]] = field(default_factory=list)
    adx: List[Optional[float]] = field(default_factory=list)
    pdi: List[Optional[float]] = field(default_factory=list)
    mdi: List[Optional[float]] = field(default_factory=list)
    st_line: List[Optional[float]] = field(default_factory=list)
    st_up: List[Optional[bool]] = field(default_factory=list)
    mfi: List[Optional[float]] = field(default_factory=list)
    flow: List[Optional[float]] = field(default_factory=list)   # last-3-candle volume vs usual
    prev_day: dict = field(default_factory=dict)                # day number -> (high, low, close)


def build_features(days):
    bars, day_of, day_start, pos = [], [], [], []
    for d, day in enumerate(days):
        day_start.append(len(bars))
        for j, b in enumerate(day):
            bars.append(b)
            day_of.append(d)
            pos.append(j)
    F = Features(bars, day_of, day_start, pos)
    n = len(bars)
    if not n:
        return F
    H = [b.h for b in bars]
    L = [b.l for b in bars]
    C = [b.c for b in bars]

    # VWAP (resets daily; plain average price until volume is reported)
    pv = vol = tps = 0.0
    for k, b in enumerate(bars):
        if pos[k] == 0:
            pv = vol = tps = 0.0
        tp = (b.h + b.l + b.c) / 3
        pv += tp * b.v
        vol += b.v
        tps += tp
        F.vwap.append(pv / vol if vol else tps / (pos[k] + 1))

    F.ema20 = _ema(C, 20)

    # RSI 14 (Wilder)
    gains = [0.0] + [max(C[k] - C[k - 1], 0) for k in range(1, n)]
    losses = [0.0] + [max(C[k - 1] - C[k], 0) for k in range(1, n)]
    ag, al = _rma(gains, 14), _rma(losses, 14)
    F.rsi = [None if g is None or l_ is None else (100.0 if l_ == 0 else 100 - 100 / (1 + g / l_))
             for g, l_ in zip(ag, al)]

    # ADX 14 / DI
    tr = [H[0] - L[0]] + [max(H[k] - L[k], abs(H[k] - C[k - 1]), abs(L[k] - C[k - 1])) for k in range(1, n)]
    pdm = [0.0] + [(H[k] - H[k - 1]) if (H[k] - H[k - 1]) > (L[k - 1] - L[k]) and (H[k] - H[k - 1]) > 0 else 0.0
                   for k in range(1, n)]
    mdm = [0.0] + [(L[k - 1] - L[k]) if (L[k - 1] - L[k]) > (H[k] - H[k - 1]) and (L[k - 1] - L[k]) > 0 else 0.0
                   for k in range(1, n)]
    atr14, spdm, smdm = _rma(tr, 14), _rma(pdm, 14), _rma(mdm, 14)
    dx = []
    for a, p, m in zip(atr14, spdm, smdm):
        if a is None or not a:
            F.pdi.append(None)
            F.mdi.append(None)
            dx.append(None)
            continue
        pdi, mdi = 100 * p / a, 100 * m / a
        F.pdi.append(pdi)
        F.mdi.append(mdi)
        dx.append(100 * abs(pdi - mdi) / (pdi + mdi) if pdi + mdi else 0.0)
    first = next((k for k, x in enumerate(dx) if x is not None), None)
    F.adx = [None] * n
    if first is not None:
        sm = _rma(dx[first:], 14)
        for k, x in enumerate(sm):
            F.adx[first + k] = x

    # Supertrend (10, 3)
    atr10 = _rma(tr, 10)
    fu = fl = None
    up = True
    for k in range(n):
        a = atr10[k]
        if a is None:
            F.st_line.append(None)
            F.st_up.append(None)
            continue
        mid = (H[k] + L[k]) / 2
        bu, bl = mid + 3 * a, mid - 3 * a
        if fu is None:
            fu, fl = bu, bl
        else:
            fu = bu if bu < fu or C[k - 1] > fu else fu
            fl = bl if bl > fl or C[k - 1] < fl else fl
        if up and C[k] < fl:
            up = False
        elif not up and C[k] > fu:
            up = True
        F.st_up.append(up)
        F.st_line.append(fl if up else fu)

    # MFI 14
    tp = [(H[k] + L[k] + C[k]) / 3 for k in range(n)]
    posf = deque()
    negf = deque()
    for k in range(n):
        raw = tp[k] * bars[k].v
        posf.append(raw if k and tp[k] > tp[k - 1] else 0.0)
        negf.append(raw if k and tp[k] < tp[k - 1] else 0.0)
        if len(posf) > 14:
            posf.popleft()
            negf.popleft()
        p, m = sum(posf), sum(negf)
        F.mfi.append(None if k < 14 or (p + m) == 0 else (100.0 if m == 0 else 100 - 100 / (1 + p / m)))

    # money flow: last 3 candles' volume vs the same slots' average over the previous 20 days
    hist = defaultdict(lambda: deque(maxlen=20))
    cur_day = -1
    avg = {}
    for k, b in enumerate(bars):
        if day_of[k] != cur_day:
            if cur_day >= 0:
                for kk in range(day_start[cur_day], k):
                    if bars[kk].v:
                        hist[bars[kk].ts.time()].append(bars[kk].v)
            cur_day = day_of[k]
            avg = {t: sum(q) / len(q) for t, q in hist.items() if len(q) >= 5}
        lo = max(day_start[cur_day], k - 2)
        got = usual = 0.0
        for kk in range(lo, k + 1):
            a = avg.get(bars[kk].ts.time())
            if a and bars[kk].v:
                got += bars[kk].v
                usual += a
        F.flow.append(got / usual if usual else None)

    for d in range(1, len(days)):
        prev = days[d - 1]
        F.prev_day[d] = (max(b.h for b in prev), min(b.l for b in prev), prev[-1].c)
    return F


# ==========================================================================
# the six setups: return (side, stop, note) for a signal on bar k's close
# ==========================================================================
def _orb(F, k, st):
    j = F.pos_in_day[k]
    if j < 3:
        return None
    s = F.day_start[F.day_of[k]]
    if F.bars[s].ts.time() != time(9, 15):
        return None
    hi = max(b.h for b in F.bars[s:s + 3])
    lo = min(b.l for b in F.bars[s:s + 3])
    b, pc = F.bars[k], F.bars[k - 1].c
    rising = F.ema20[k] > F.ema20[k - 3]
    done = st.setdefault("done", set())
    if b.c > hi >= pc and b.c > F.vwap[k] and rising and "L" not in done:
        done.add("L")
        return 1, b.l, f"Close above opening range high {hi:.2f}"
    if b.c < lo <= pc and b.c < F.vwap[k] and not rising and "S" not in done:
        done.add("S")
        return -1, b.h, f"Close below opening range low {lo:.2f}"
    return None


def _vwap_pullback(F, k, st):
    j = F.pos_in_day[k]
    if j < 12:
        return None
    b, vw = F.bars[k], F.vwap[k]
    above = sum(1 for kk in range(k - 12, k) if F.bars[kk].c > F.vwap[kk])
    lows = [x.l for x in F.bars[k - 3:k + 1]]
    highs = [x.h for x in F.bars[k - 3:k + 1]]
    if (above >= 9 and F.ema20[k] > vw and F.ema20[k] > F.ema20[k - 6]
            and b.l <= vw * 1.002 and b.c > vw and b.c > b.o):
        return 1, min(lows), f"Uptrend pullback to VWAP {vw:.2f}"
    if (above <= 3 and F.ema20[k] < vw and F.ema20[k] < F.ema20[k - 6]
            and b.h >= vw * 0.998 and b.c < vw and b.c < b.o):
        return -1, max(highs), f"Downtrend pullback to VWAP {vw:.2f}"
    return None


def _rsi_sr(F, k, st):
    j = F.pos_in_day[k]
    d = F.day_of[k]
    if j < 3 or d not in F.prev_day or F.rsi[k] is None or k < 4:
        return None
    pdh, pdl, pdc = F.prev_day[d]
    s = F.day_start[d]
    b = F.bars[k]
    body = abs(b.c - b.o)
    recent = [x for x in F.rsi[k - 3:k] if x is not None]
    if not recent:
        return None
    supports = [pdl, pdc] + ([min(x.l for x in F.bars[s:k - 5])] if k - 5 > s else [])
    resists = [pdh, pdc] + ([max(x.h for x in F.bars[s:k - 5])] if k - 5 > s else [])
    if min(recent) < 30 <= F.rsi[k] and b.c > b.o and (min(b.o, b.c) - b.l) >= body * 0.5:
        lvl = next((x for x in supports if abs(b.l - x) / x <= 0.003 or b.l <= x <= b.c), None)
        if lvl is not None:
            return 1, min(b.l, lvl), f"RSI back above 30 at support {lvl:.2f}"
    if max(recent) > 70 >= F.rsi[k] and b.c < b.o and (b.h - max(b.o, b.c)) >= body * 0.5:
        lvl = next((x for x in resists if abs(b.h - x) / x <= 0.003 or b.c <= x <= b.h), None)
        if lvl is not None:
            return -1, max(b.h, lvl), f"RSI back below 70 at resistance {lvl:.2f}"
    return None


def _inside_bar(F, k, st):
    if F.pos_in_day[k] < 3:
        return None
    m, ib, b = F.bars[k - 2], F.bars[k - 1], F.bars[k]
    if not (ib.h <= m.h and ib.l >= m.l and (ib.h - ib.l) < (m.h - m.l)):
        return None
    if b.c > ib.h and b.c > F.ema20[k] and b.c > F.vwap[k]:
        return 1, ib.l, f"Break above inside bar {ib.h:.2f}"
    if b.c < ib.l and b.c < F.ema20[k] and b.c < F.vwap[k]:
        return -1, ib.h, f"Break below inside bar {ib.l:.2f}"
    return None


def _retest(F, k, st):
    j = F.pos_in_day[k]
    d = F.day_of[k]
    s = F.day_start[d]
    if j < 4:
        return None
    levels = {}
    if d in F.prev_day:
        levels["PDH"], levels["PDL"] = F.prev_day[d][0], F.prev_day[d][1]
    if F.bars[s].ts.time() == time(9, 15):
        levels["OR high"] = max(x.h for x in F.bars[s:s + 3])
        levels["OR low"] = min(x.l for x in F.bars[s:s + 3])
    b = F.bars[k]
    # remember breaks seen on earlier bars (state lives for the day)
    prev = F.bars[k - 1]
    pprev_c = F.bars[k - 2].c
    for name, lv in levels.items():
        if prev.c > lv >= pprev_c:
            st.setdefault("up", {})[name] = k - 1
        if prev.c < lv <= pprev_c:
            st.setdefault("down", {})[name] = k - 1
    for name, at in list(st.get("up", {}).items()):
        lv = levels[name]
        if k - at > 12:
            continue
        if b.l <= lv * 1.0015 and b.c > lv and b.c > b.o and ("L", name) not in st.setdefault("done", set()):
            st["done"].add(("L", name))
            return 1, min(x.l for x in F.bars[at + 1:k + 1]), f"Retest of {name} {lv:.2f} held"
    for name, at in list(st.get("down", {}).items()):
        lv = levels[name]
        if k - at > 12:
            continue
        if b.h >= lv * 0.9985 and b.c < lv and b.c < b.o and ("S", name) not in st.setdefault("done", set()):
            st["done"].add(("S", name))
            return -1, max(x.h for x in F.bars[at + 1:k + 1]), f"Retest of {name} {lv:.2f} held"
    return None


def _st_adx(F, k, st):
    if F.pos_in_day[k] < 1 or F.st_up[k] is None or F.st_up[k - 1] is None or F.adx[k] is None:
        return None
    if F.st_up[k] == F.st_up[k - 1] or F.adx[k] < 25:
        return None
    if F.st_up[k] and F.pdi[k] > F.mdi[k]:
        return 1, F.st_line[k], f"Supertrend turned up, ADX {F.adx[k]:.0f}"
    if not F.st_up[k] and F.mdi[k] > F.pdi[k]:
        return -1, F.st_line[k], f"Supertrend turned down, ADX {F.adx[k]:.0f}"
    return None


DETECTORS = {"orb": _orb, "vwap_pullback": _vwap_pullback, "rsi_sr": _rsi_sr, "inside_bar": _inside_bar,
             "retest": _retest, "st_adx": _st_adx}


# ==========================================================================
# signals and outcomes
# ==========================================================================
@dataclass
class Signal:
    strategy: str
    k: int                 # signal bar index (entry is k + 1)
    side: int              # +1 long, -1 short
    entry: float
    stop: float
    risk: float
    note: str
    flow_ok: bool
    flow: Optional[float]
    mfi: Optional[float]


def charges(buy, sell, qty):
    """Estimated NSE intraday equity charges for a round trip (Zerodha-style rates)."""
    tb, ts = buy * qty, sell * qty
    brokerage = min(20.0, 0.0003 * tb) + min(20.0, 0.0003 * ts)
    other = 0.0000297 * (tb + ts) + 0.000001 * (tb + ts)
    return brokerage + 0.00025 * ts + other + 0.00003 * tb + 0.18 * (brokerage + other)


def find_signals(F, day_numbers):
    """Signals for every strategy on the given days (bar indices into F)."""
    out = []
    for d in day_numbers:
        s = F.day_start[d]
        e = F.day_start[d + 1] if d + 1 < len(F.day_start) else len(F.bars)
        for key, det in DETECTORS.items():
            state, fired, last = {}, 0, -10 ** 9
            for k in range(s, e):
                if k + 1 >= e:
                    break
                end_t = (F.bars[k].ts + CANDLE).time()
                hit = det(F, k, state)          # always called, so stateful detectors see every bar
                if not hit or end_t < FIRST_SIGNAL or end_t > LAST_SIGNAL:
                    continue
                if fired >= MAX_PER_DAY or k - last < COOLDOWN:
                    continue
                nxt = F.bars[k + 1]
                if nxt.ts.time() >= SQUAREOFF:
                    continue
                side, stop, note = hit
                entry = nxt.o
                risk = (entry - stop) if side > 0 else (stop - entry)
                if risk <= 0 or risk / entry * 100 > MAX_RISK_PCT:
                    continue
                if risk / entry * 100 < MIN_RISK_PCT:
                    risk = entry * MIN_RISK_PCT / 100
                    stop = entry - risk if side > 0 else entry + risk
                flow, mfi = F.flow[k], F.mfi[k]
                flow_ok = flow is not None and flow >= FLOW_MIN and mfi is not None and (
                    mfi >= 50 if side > 0 else mfi <= 50)
                out.append(Signal(key, k, side, entry, stop, risk, note, flow_ok, flow, mfi))
                fired += 1
                last = k
    return out


def simulate(F, sig, target_r, open_ended=False):
    """
    Walk forward from the entry candle.  Returns a dict with the exit (exit_k, exit_price, reason,
    r_gross, r_net, target), or {"open": True, ...} when `open_ended` (a live day still trading)
    and nothing has been hit yet.
    """
    side, entry, stop = sig.side, sig.entry, sig.stop
    target = entry + side * target_r * sig.risk
    d = F.day_of[sig.k]
    e = F.day_start[d + 1] if d + 1 < len(F.day_start) else len(F.bars)
    res = None
    for j in range(sig.k + 1, e):
        b = F.bars[j]
        if b.ts.time() >= SQUAREOFF:
            res = (j, b.o, "Square-off 15:15")
            break
        if j > sig.k + 1:
            if (b.o <= stop) if side > 0 else (b.o >= stop):
                res = (j, b.o, "Stop loss (gap)")
                break
            if (b.o >= target) if side > 0 else (b.o <= target):
                res = (j, b.o, "Target (gap)")
                break
        hit_sl = b.l <= stop if side > 0 else b.h >= stop
        hit_tp = b.h >= target if side > 0 else b.l <= target
        if hit_sl:
            res = (j, stop, "Stop loss")
            break
        if hit_tp:
            res = (j, target, "Target")
            break
    if res is None:
        if open_ended:
            return {"open": True, "target": target}
        j = e - 1
        res = (j, F.bars[j].c, "Day end")
    j, px, reason = res
    gross = (px - entry) * side
    qty = max(1, int(CAPITAL // entry))
    cost = charges(entry if side > 0 else px, px if side > 0 else entry, qty)
    return {"open": False, "exit_k": j, "exit_price": px, "reason": reason, "target": target,
            "r_gross": gross / sig.risk, "r_net": (gross * qty - cost) / (sig.risk * qty)}


def _stats(results):
    n = len(results)
    if not n:
        return {"trades": 0, "win_rate": None, "avg_r": None}
    wins = sum(1 for r in results if r["r_net"] > 0)
    return {"trades": n, "win_rate": wins / n * 100, "avg_r": sum(r["r_net"] for r in results) / n}


def backtest(F, day_numbers):
    """Per strategy: every (variant, target) tried, and the best qualified choice."""
    sigs = find_signals(F, day_numbers)
    by = defaultdict(list)
    for s in sigs:
        by[s.strategy].append(s)
    report = {}
    for key, name, kind in STRATEGIES:
        options = []
        for flow_only in (False, True):
            pool = [s for s in by[key] if s.flow_ok or not flow_only]
            for t in TARGETS_R:
                st = _stats([simulate(F, s, t) for s in pool])
                st.update({"money_flow": flow_only, "target_r": t})
                options.append(st)
        ok = [o for o in options if o["trades"] >= MIN_TRADES and o["win_rate"] >= MIN_WIN_RATE and o["avg_r"] > 0]
        tried = [o for o in options if o["trades"]]
        best = max(ok, key=lambda o: (o["avg_r"], o["win_rate"])) if ok else (
            max(tried, key=lambda o: (o["avg_r"], o["win_rate"])) if tried else None)
        if ok:
            why = ""
        elif not tried or max(o["trades"] for o in tried) < MIN_TRADES:
            why = f"Fewer than {MIN_TRADES} past signals"
        else:
            why = f"Best past result {best['win_rate']:.0f}% win, {best['avg_r']:+.2f} R"
        report[key] = {"key": key, "name": name, "type": kind, "qualified": bool(ok), "reason": why,
                       "choice": best, "options": options}
    return report


# ==========================================================================
# public entry point
# ==========================================================================
def _history_for(company, day, cache_key):
    hist = _history_bars(company, day)
    report = cache.get(cache_key)
    if report is None:
        F = build_features(hist)
        report = {"report": backtest(F, list(range(1, len(hist)))),
                  "days": len(hist),
                  "from": hist[0][0].ts.date().isoformat() if hist else None,
                  "to": hist[-1][0].ts.date().isoformat() if hist else None}
        cache.set(cache_key, report, 6 * 3600)
    return report, hist


def session_signals(company, mode, day, now=None):
    """Strategy report + the session day's signals (entries, exits so far)."""
    live = mode == svc.LIVE
    day = svc.today_ist() if live else day
    key = f"paper_trading:strategy:{company.pk}:{day.isoformat()}"
    hist_info, hist = _history_for(company, day, key)
    today = _live_day_bars(company, now) if live else _replay_day_bars(company, day)

    days = hist + ([today] if today else [])
    F = build_features(days)
    report = hist_info["report"]
    signals = []
    if today:
        d = len(days) - 1
        for s in find_signals(F, [d]):
            rep = report[s.strategy]
            choice = rep["choice"]
            target_r = choice["target_r"] if choice else 1.5
            res = simulate(F, s, target_r, open_ended=live)
            b, nb = F.bars[s.k], F.bars[s.k + 1]
            item = {
                "strategy": s.strategy, "name": NAMES[s.strategy], "label": SHORT[s.strategy],
                "qualified": rep["qualified"],
                "passes_filter": s.flow_ok or not (choice and choice["money_flow"]),
                "side": "BUY" if s.side > 0 else "SELL",
                "signal_x": b.ts.isoformat(), "entry_x": nb.ts.isoformat(),
                "entry": round(s.entry, 2), "stop": round(s.stop, 2), "target": round(res["target"], 2),
                "target_r": target_r, "risk_pct": round(s.risk / s.entry * 100, 2),
                "note": s.note,
                "flow": round(s.flow, 2) if s.flow is not None else None,
                "mfi": round(s.mfi, 1) if s.mfi is not None else None,
                "status": "open" if res["open"] else "closed",
            }
            if not res["open"]:
                xb = F.bars[res["exit_k"]]
                item.update({"exit_x": xb.ts.isoformat(), "exit_price": round(res["exit_price"], 2),
                             "exit_reason": res["reason"], "result_r": round(res["r_net"], 2),
                             "exit_at_open": res["reason"].startswith("Square") or "(gap)" in res["reason"]})
            signals.append(item)

    strategies = []
    for k, _, _ in STRATEGIES:
        rep = report[k]
        c = rep["choice"]
        strategies.append({
            "key": k, "name": rep["name"], "label": SHORT[k], "type": rep["type"], "qualified": rep["qualified"],
            "reason": rep["reason"],
            "trades": c["trades"] if c else 0,
            "win_rate": round(c["win_rate"], 1) if c and c["win_rate"] is not None else None,
            "avg_r": round(c["avg_r"], 2) if c and c["avg_r"] is not None else None,
            "target_r": c["target_r"] if c else None,
            "money_flow": bool(c and c["money_flow"]),
        })
    return {
        "company": company.code, "mode": mode, "date": day.isoformat(),
        "history": {"days": hist_info["days"], "from": hist_info["from"], "to": hist_info["to"]},
        "rules": {"min_win_rate": MIN_WIN_RATE, "min_trades": MIN_TRADES, "flow_min": FLOW_MIN,
                  "last_signal": LAST_SIGNAL.strftime("%H:%M"), "squareoff": SQUAREOFF.strftime("%H:%M")},
        "strategies": strategies,
        "signals": signals,
    }

"""
The fast part of Refresh orders, shared by Claude Orders (claude_board.py) and Claude Orders 2.0 (claude_board2.py).

Analysing a stock is mostly Python work, and Python threads can't share that, so the stocks go to worker processes:

  1. Workers start in the background while the list of stocks is being chosen.
  2. Live data arrives in bulk, side by side: Yahoo's candles for every stock in a few batched downloads, and Angel
     One's last trades, 50 stocks a call.  During market hours each stock's last candle is brought up to its latest
     Angel One trade, because Yahoo's lag a little.  A stock Yahoo missed is simply fetched by itself as before.
  3. Each worker analyses its share with board.gather(), the same code and rules as before, reading each stock's
     history from the database once.

There are as many workers as free memory allows (MAX_WORKERS at most), running at low priority so the server and
the stop loss / target watcher stay quick.  They only read: no worker starts the watcher (paper_trading/apps.py).
If the workers can't run, the remaining stocks are analysed in this process.  CLAUDE_BOARD_WORKERS in .env sets the
number of workers; 0 turns them off.
"""
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

MAX_WORKERS = 6
MIN_WORKERS = 2
WORKER_MB = 300               # a worker's memory, with some room (measured about 210 MB)
SPARE_MB = 700                # free memory left for everything else, the refresh itself included
YAHOO_BATCH = 50
SEED_SECONDS = 1800           # a worker lives for one refresh; this only has to outlast it
QUOTE_FRESH_SECONDS = 600     # an older Angel One trade doesn't move the last candle
CANDLES = {"period": "5d", "interval": "5m", "prepost": False}
DAILY = {"period": "3mo", "interval": "1d"}

_market_pk = [None]


def worker_count():
    """How many workers to start: 0 means analyse in this process."""
    raw = (os.environ.get("CLAUDE_BOARD_WORKERS") or "").strip()
    if raw.isdigit():
        return min(int(raw), MAX_WORKERS)
    if "test" in sys.argv:
        return 0
    import psutil
    by_memory = ((psutil.virtual_memory().available >> 20) - SPARE_MB) // WORKER_MB
    return max(MIN_WORKERS, min(MAX_WORKERS, (os.cpu_count() or 2) - 1, by_memory))


def together(**jobs):
    """Run the named functions side by side: {name: result}.  A job's exception comes out here."""
    from django.db import connection

    def run(fn):
        try:
            return fn()
        finally:
            connection.close()                                    # this thread's own database connection

    with ThreadPoolExecutor(len(jobs), thread_name_prefix="claude-board-read") as pool:
        started = {name: pool.submit(run, fn) for name, fn in jobs.items()}
        return {name: job.result() for name, job in started.items()}


# ---------------------------------------------------------------- live data in bulk
def yahoo_batch(symbols, **kwargs):
    """{symbol: Yahoo frame} from batched downloads.  yfinance runs each batch's requests itself; DATA_LOCK keeps
    anything else from calling it meanwhile (see ai_prediction/tools.py)."""
    import yfinance as yf

    from ai_prediction.tools import DATA_LOCK

    out = {}
    with DATA_LOCK:
        for i in range(0, len(symbols), YAHOO_BATCH):
            part = symbols[i:i + YAHOO_BATCH]
            try:
                raw = yf.download(part, group_by="ticker", auto_adjust=True, threads=True, progress=False, **kwargs)
            except Exception as exc:
                log.warning("Claude Orders refresh: Yahoo batch failed: %s", exc)
                continue
            for symbol in part:
                try:
                    frame = raw[symbol].dropna(subset=["Close"])
                except KeyError:
                    continue
                if not frame.empty:
                    out[symbol] = frame
    return out


def angel_trades(codes):
    """{code: (last price, trade time)} from Angel One.  Empty outside market hours or when Angel One can't answer."""
    from prediction.market_time import IST, get_market_status

    from . import angel

    if get_market_status() != "OPEN":
        return {}
    tokens = {}
    for code in codes:
        try:
            tokens[angel.instrument(code)["token"]] = code
        except angel.AngelError:                                  # not an NSE equity on Angel One
            pass
    try:
        quotes = angel.quotes(list(tokens))
    except angel.AngelError as exc:
        log.warning("Claude Orders refresh: Angel One quotes unavailable: %s", exc)
        return {}
    oldest = datetime.now(IST) - timedelta(seconds=QUOTE_FRESH_SECONDS)
    out = {}
    for token, q in quotes.items():
        traded = angel.trade_time(q)
        if token in tokens and q.get("ltp") and traded and traded >= oldest:
            out[tokens[token]] = (float(q["ltp"]), traded)
    return out


def with_last_trade(frame, price, traded):
    """Yahoo's 5-minute frame brought up to a later trade: the candle it falls in moves, or a new candle starts."""
    import pandas as pd

    if frame.empty or frame.index.tz is None:
        return frame
    stamp = pd.Timestamp(traded).floor("5min").tz_convert(frame.index.tz)
    last = frame.index[-1]
    if stamp < last:
        return frame
    out = frame.copy()
    if stamp == last:
        out.loc[last, "High"] = max(out.loc[last, "High"], price)
        out.loc[last, "Low"] = min(out.loc[last, "Low"], price)
        out.loc[last, "Close"] = price
    else:
        out.loc[stamp, ["Open", "High", "Low", "Close", "Volume"]] = [price, price, price, price, 0]
    return out


def prefetch(codes):
    """Yields ({code: {cache key: frame}}, Angel One prices used) a batch at a time, so workers start on the first
    batch while the next downloads.  The frames are ready for _yahoo(), with the market's added to each stock."""
    from prediction.datasets import MARKET_SYMBOL, yahoo_key
    from prediction.tickers import get_yahoo_ticker

    market = None
    for i in range(0, len(codes), YAHOO_BATCH):
        part = codes[i:i + YAHOO_BATCH]
        symbols = {code: get_yahoo_ticker(code) for code in part}
        wanted = sorted(set(symbols.values()) | ({MARKET_SYMBOL} if market is None else set()))
        got = together(candles=lambda: yahoo_batch(wanted, **CANDLES), daily=lambda: yahoo_batch(wanted, **DAILY),
                       angel=lambda: angel_trades(part))
        trades = got["angel"]

        def frames(symbol, trade=None):
            out = {}
            if symbol in got["candles"]:
                frame = got["candles"][symbol]
                out[yahoo_key(symbol, **CANDLES)] = with_last_trade(frame, *trade) if trade else frame
            if symbol in got["daily"]:
                out[yahoo_key(symbol, **DAILY)] = got["daily"][symbol]
            return out

        if market is None:
            market = frames(MARKET_SYMBOL)
        log.info("Claude Orders refresh: batch of %d: Yahoo %d, Angel One %d", len(part), len(got["candles"]), len(trades))
        yield {code: {**market, **frames(symbol, trades.get(code))} for code, symbol in symbols.items()}, len(trades)


def _seed(frames):
    from django.core.cache import cache
    for key, frame in frames.items():
        cache.set(key, frame, SEED_SECONDS)


# ---------------------------------------------------------------- the worker's side
def _start_worker():
    import django
    import psutil

    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except (psutil.Error, AttributeError):
        pass
    django.setup()
    from ai_prediction import tools  # noqa: F401  loaded now, while the data is still downloading
    from market.models import Company
    from news import signal  # noqa: F401
    from patterns import engine  # noqa: F401
    from prediction import data
    from prediction.datasets import MARKET_SYMBOL

    from . import claude_board  # noqa: F401
    _market_pk[0] = Company.objects.filter(code=MARKET_SYMBOL).values_list("pk", flat=True).first()
    data.remember_only(_market_pk[0])


def _ready():
    return os.getpid()


def _analyse(code, why, frames):
    from prediction import data

    from . import claude_board as board

    _seed(frames)
    data.remember_only(_market_pk[0])                             # the market stays; the last stock's history goes
    try:
        return board.gather(code, why)
    except Exception:
        log.warning("Claude Orders refresh: %s failed", code, exc_info=True)
        return None


# ---------------------------------------------------------------- the refresh's side
class Workers:
    """with Workers() as w: ... w.gather(...).  Enter it early: the workers start while the rest of the refresh runs."""

    def __enter__(self):
        self.pool = None
        self.live_prices = 0
        self.size = worker_count()
        if self.size:
            try:
                self.pool = ProcessPoolExecutor(self.size, mp_context=multiprocessing.get_context("spawn"),
                                                initializer=_start_worker)
                for _ in range(self.size):
                    self.pool.submit(_ready)
            except Exception:
                log.exception("Claude Orders refresh: workers did not start")
                self.pool = None
        return self

    def __exit__(self, *exc):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def gather(self, cands, why, note=None, each=None):
        """Analyses every stock.  note(step, done) reports progress; each(code, stock or None) gets each result as it
        arrives, so scoring happens while the workers carry on."""
        from . import claude_board as board

        note = note or (lambda step, done: None)
        each = each or (lambda code, stock: None)
        seen, frames, jobs = set(), {}, {}

        def finished(code, stock):
            seen.add(code)
            each(code, stock)
            note(f"Analysed {code.removesuffix('.NS')}", len(seen))

        if self.pool:
            try:
                note("Downloading live data from Yahoo and Angel One", 0)
                for batch, prices in prefetch(cands):
                    self.live_prices += prices
                    frames.update(batch)
                    jobs.update({self.pool.submit(_analyse, code, why[code], data): code for code, data in batch.items()})
                    note(f"Downloading live data from Yahoo and Angel One ({len(frames)} of {len(cands)})", 0)
                for job in as_completed(jobs):
                    finished(jobs[job], job.result())
            except BrokenProcessPool:
                log.exception("Claude Orders refresh: a worker stopped; finishing in this process")
        for code in cands:
            if code not in seen:
                note(f"Analysing {code.removesuffix('.NS')}", len(seen))
                _seed(frames.get(code, {}))
                try:
                    stock = board.gather(code, why[code])
                except Exception:
                    log.warning("Claude Orders refresh: %s failed", code, exc_info=True)
                    stock = None
                finished(code, stock)

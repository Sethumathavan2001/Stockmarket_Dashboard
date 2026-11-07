"""
Indices on the Stock Prediction and Candle Predictor pages.

An index goes through the same training and forecasting code as a stock, as an IndexAsset: code "NIFTY.IDX", with the
few fields those need.  Indices are not Company rows, so they stay out of every other page and the Claude Orders lists.

History comes from market/index_data.py's tables, and the days since from Angel One (Yahoo doesn't carry them all).
Indices have no volume, so it is 0 and the volume features say nothing.  Training and the Candle Predictor use the
latest HISTORY_ROWS candles, about the most any stock has.

last_prices() gives the LTP for the Trained stocks list.
"""
import logging
from datetime import datetime, timedelta

import pandas as pd
from django.core.cache import cache
from django.db.models import Max
from django.http import Http404

from market.models import Company, DailyData, MarketIndex

from .market_time import IST

logger = logging.getLogger(__name__)

SUFFIX = ".IDX"
NO_NEWS = "News is matched to single stocks, so it isn't used for an index."
HISTORY_ROWS = 50_000
RECENT_SECONDS = 30
LTP_SECONDS = 60
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
WINDOW_DAYS = {"FIVE_MINUTE": (5, 60), "ONE_DAY": (10, 120)}          # least / most days asked for since the stored data
MINUTE_DAYS = 28                                                      # Angel One gives 30 days of 1-minute candles a call


class IndexAsset:
    is_index = True
    sector = "Index"
    industry = ""

    def __init__(self, row):
        self.row = row
        self.key = row.key
        self.code = row.key + SUFFIX
        self.name = row.name
        self.pk = f"index:{row.key}"


def is_index(asset):
    return getattr(asset, "is_index", False)


def all_assets():
    return [IndexAsset(row) for row in MarketIndex.objects.all()]


def find(code):
    """The index for a code like "NIFTY.IDX", or None."""
    if not code.endswith(SUFFIX):
        return None
    row = MarketIndex.objects.filter(key=code.removesuffix(SUFFIX)).first()
    return IndexAsset(row) if row else None


def asset_or_404(code):
    asset = find(code) or Company.objects.filter(code=code).first()
    if asset is None:
        raise Http404(f"No stock or index '{code}'.")
    return asset


# ---------------------------------------------------------------- candles
def history(asset):
    """Stored 5-minute candles, oldest first: the columns load_historical_candles() gives."""
    rows = list(asset.row.five_min.order_by("-timestamp").values_list("timestamp", "open", "high", "low", "close")[:HISTORY_ROWS])
    if not rows:
        return pd.DataFrame(columns=COLUMNS + ["source"])
    df = pd.DataFrame(rows[::-1], columns=COLUMNS[:5])
    ts = pd.to_datetime(df["timestamp"])
    df["timestamp"] = ts.dt.tz_localize(IST) if ts.dt.tz is None else ts.dt.tz_convert(IST)
    df["volume"] = 0.0
    df["source"] = "database"
    return df


def daily(asset):
    """Stored daily candles: the columns load_daily() works with."""
    rows = asset.row.daily.order_by("date").values_list("date", "open", "high", "low", "close")
    df = pd.DataFrame(list(rows), columns=["date", "open", "high", "low", "close"])
    df["volume"] = 0.0
    return df


def _window(asset, interval):
    """Days of candles to ask Angel One for: back to the stored data, within limits."""
    if interval == "ONE_MINUTE":
        return MINUTE_DAYS
    table = asset.row.five_min if interval == "FIVE_MINUTE" else asset.row.daily
    last = table.aggregate(d=Max("date"))["d"]
    least, most = WINDOW_DAYS[interval]
    gap = (datetime.now(IST).date() - last).days + 1 if last else most
    return max(least, min(most, gap))


def recent(asset, interval="FIVE_MINUTE"):
    """Angel One's latest candles (a timestamp column, like _yahoo_frame()).  Empty when Angel One can't answer: the
    forecast then starts from the stored candles."""
    from paper_trading import angel

    key = f"prediction:index:{asset.key}:{interval}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    now = datetime.now(IST)
    params = {"exchange": asset.row.exchange, "symboltoken": asset.row.angel_token, "interval": interval,
              "fromdate": (now - timedelta(days=_window(asset, interval))).strftime("%Y-%m-%d %H:%M"),
              "todate": now.strftime("%Y-%m-%d %H:%M")}
    try:
        rows = (angel.read_any(lambda sc: sc.getCandleData(dict(params)))[1] or {}).get("data") or []
    except angel.AngelError as exc:
        logger.warning("Angel One candles for %s: %s", asset.code, exc)
        rows = []
    df = pd.DataFrame([r[:6] for r in rows], columns=COLUMNS)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(IST)
        df[COLUMNS[1:]] = df[COLUMNS[1:]].astype(float)
    cache.set(key, df, RECENT_SECONDS)
    return df


# ---------------------------------------------------------------- LTP
def _angel_prices(codes):
    from paper_trading import angel

    out, tokens = {}, {"NSE": {}, "BSE": {}}
    for code in codes:
        asset = find(code)
        try:
            if asset:
                tokens[asset.row.exchange][asset.row.angel_token] = code
            else:
                tokens["NSE"][angel.instrument(code)["token"]] = code
        except angel.AngelError:                                  # not an NSE equity on Angel One
            pass
    for exchange, wanted in tokens.items():
        if wanted:
            for token, q in angel.quotes(list(wanted), exchange).items():
                if token in wanted and q.get("ltp"):
                    out[wanted[token]] = float(q["ltp"])
    return out


def _stored_price(code):
    asset = find(code)
    if asset:
        return asset.row.daily.order_by("-date").values_list("close", flat=True).first()
    close = DailyData.objects.filter(company__code=code).order_by("-date").values_list("close", flat=True).first()
    return float(close) if close is not None else None


def last_prices(codes):
    """{code: last price}: Angel One's, or the latest stored close when it can't answer."""
    from paper_trading import angel

    prices = cache.get("prediction:ltps") or {}
    missing = [c for c in codes if c not in prices]
    if missing:
        try:
            found = _angel_prices(missing)
        except angel.AngelError as exc:
            logger.warning("Angel One prices for the Trained stocks list: %s", exc)
            found = {}
        prices = {**prices, **found, **{c: _stored_price(c) for c in missing if c not in found}}
        cache.set("prediction:ltps", prices, LTP_SECONDS)
    return {c: prices.get(c) for c in codes}

import json
from datetime import date, datetime, time, timedelta
from unittest import mock

import numpy as np
import pandas as pd
from django.core.cache import cache
from django.test import TestCase

from market.models import Company

from . import engine
from prediction.market_time import IST


def frame(days=30, seed=3, interval_min=5):
    """Synthetic session candles: `days` weekdays of 09:15-15:30."""
    rng = np.random.default_rng(seed)
    rows, price, d = [], 100.0, date(2026, 3, 2)
    made = 0
    per_day = 375 // interval_min
    while made < days:
        if d.weekday() < 5:
            start = datetime.combine(d, time(9, 15), tzinfo=IST)
            for i in range(per_day):
                o = price
                price = max(1.0, price * (1 + rng.normal(0, 0.002)))
                rows.append((start + timedelta(minutes=interval_min * i), o, max(o, price) * 1.001,
                             min(o, price) * 0.999, price, float(rng.integers(1000, 5000))))
            made += 1
        d += timedelta(days=1)
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def candles(spec, day=date(2026, 3, 2)):
    """Hand-made 5-minute candles from (open, high, low, close) tuples."""
    start = datetime.combine(day, time(9, 15), tzinfo=IST)
    return pd.DataFrame([(start + timedelta(minutes=5 * i), o, h, l, c, 1000.0) for i, (o, h, l, c) in enumerate(spec)],
                        columns=["timestamp", "open", "high", "low", "close", "volume"])


def found(df):
    b = engine.Bars(df, "5m")
    return {(k, key) for k, key in engine.detect(b)}


class PatternDetectionTests(TestCase):
    def test_bullish_engulfing(self):
        spec = [(100, 101, 99, 100.5)] * 3 + [(101, 101.2, 99.8, 100.0), (99.8, 101.8, 99.6, 101.5)]
        self.assertIn((4, "bull_engulfing"), found(candles(spec)))

    def test_hammer_after_a_fall(self):
        spec = [(105, 105.5, 104, 104.2), (104.2, 104.4, 103, 103.2), (103.2, 103.4, 102, 102.2), (102.2, 102.4, 101.5, 101.6),
                (101.6, 101.75, 99.0, 101.7)]                      # long lower wick, tiny body near the high
        self.assertIn((4, "hammer"), found(candles(spec)))

    def test_inside_bar_breakout(self):
        spec = [(100, 100.5, 99.5, 100.2)] * 2 + [(100, 103, 98, 101), (100.5, 101.5, 99.5, 101), (101, 103.5, 100.8, 103.2)]
        self.assertIn((4, "ib_break_up"), found(candles(spec)))

    def test_detection_uses_only_past_candles(self):
        df = frame(days=6)
        full = {x for x in engine.detect(engine.Bars(df, "5m"))}
        cut = 300
        part = {x for x in engine.detect(engine.Bars(df.iloc[:cut].reset_index(drop=True), "5m"))}
        self.assertEqual({x for x in full if x[0] < cut}, part)


class PredictorTests(TestCase):
    def setUp(self):
        cache.clear()
        self.company = Company.objects.create(name="Test Co", code="TST.NS", exchange="NSE")

    def test_forecast_uses_only_earlier_days_and_reports_tests(self):
        df = frame(days=40)
        with mock.patch.object(engine, "load_candles", return_value=df):
            last_day = df["timestamp"].dt.date.iloc[-1]
            out = engine.predict(self.company, "5m", day=last_day, asof=time(11, 0))
        self.assertEqual(out["history"]["days"], 39)                    # the selected day is never evidence
        self.assertEqual(out["asof"][11:16], "11:00")
        f = out["forecast"]
        self.assertEqual(f["ahead"], 24)                               # 2 hours of 5-minute candles
        self.assertEqual(len(f["candles"]), 24)
        for c in f["candles"]:
            self.assertLessEqual(c["l"], min(c["o"], c["c"]))
            self.assertGreaterEqual(c["h"], max(c["o"], c["c"]))
        self.assertTrue(any(c["future"] for c in out["candles"]))      # later real candles kept for comparison
        self.assertIsNotNone(out["backtest"])

    def test_session_end_stops_the_forecast(self):
        df = frame(days=20)
        with mock.patch.object(engine, "load_candles", return_value=df):
            out = engine.predict(self.company, "5m", asof=time(15, 25))
        self.assertIsNone(out["forecast"])

    def test_api(self):
        df = frame(days=20)
        with mock.patch.object(engine, "load_candles", return_value=df):
            r = self.client.get("/api/candle-predictor/TST.NS/?interval=5m")
            self.assertEqual(r.status_code, 200)
            json.loads(r.content)                                      # valid JSON, no NaN
            self.assertEqual(self.client.get("/api/candle-predictor/TST.NS/?interval=3m").status_code, 400)
            self.assertEqual(self.client.get("/api/candle-predictor/TST.NS/?day=2019-01-01").status_code, 400)
        self.assertEqual(self.client.get("/candle-predictor/").status_code, 200)

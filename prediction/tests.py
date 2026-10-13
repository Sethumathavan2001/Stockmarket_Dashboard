import json
import shutil
import tempfile
from datetime import date, datetime, time, timedelta
from unittest import mock

import numpy as np
import pandas as pd
from django.test import TestCase, override_settings

from market.models import Company, DailyData, FiveMinuteData

from . import datasets as ds
from . import engine
from .market_time import IST


def make_5min(company, n_days=40, seed=1):
    rng = np.random.default_rng(seed)
    rows, price = [], 100.0
    day = date(2026, 1, 1)
    made = 0
    while made < n_days:
        day += timedelta(days=1)
        if day.weekday() >= 5:
            continue
        made += 1
        start = datetime.combine(day, time(9, 15), tzinfo=IST)
        for i in range(75):
            o = price
            price = max(1.0, price * (1 + rng.normal(0, 0.002)))
            rows.append(FiveMinuteData(company=company, timestamp=start + timedelta(minutes=5 * i), open=o,
                                       high=max(o, price) * 1.001, low=min(o, price) * 0.999, close=price,
                                       volume=int(rng.integers(1000, 5000))))
    FiveMinuteData.objects.bulk_create(rows)


def make_daily(company, n=700, seed=2):
    rng = np.random.default_rng(seed)
    rows, price, d = [], 100.0, date(2023, 1, 2)
    while len(rows) < n:
        d += timedelta(days=1)
        if d.weekday() >= 5:
            continue
        o = price
        price = max(1.0, price * (1 + rng.normal(0.0003, 0.015)))
        rows.append(DailyData(company=company, date=d, open=o, high=max(o, price) * 1.005,
                              low=min(o, price) * 0.995, close=price, volume=int(rng.integers(10_000, 50_000))))
    DailyData.objects.bulk_create(rows)


class PredictionEngineTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.company = Company.objects.create(name="Test Co", code="TST.NS", exchange="NSE", sector="Technology")
        make_5min(cls.company)
        make_daily(cls.company)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.settings_override = override_settings(ML_MODELS_DIR=self.tmp)
        self.settings_override.enable()
        no_yahoo = mock.patch.object(ds, "_yahoo", return_value=pd.DataFrame())
        no_yahoo.start()
        self.addCleanup(no_yahoo.stop)

    def tearDown(self):
        self.settings_override.disable()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_features_do_not_look_ahead(self):
        df, X, _ = ds.intraday_dataset(self.company)
        cut = 1500
        Xc = ds.intraday_features(df.iloc[:cut].reset_index(drop=True), None)
        full = ds.intraday_features(df, None)
        self.assertEqual(float((full.iloc[:cut] - Xc).abs().max().max()), 0.0)

    def test_intraday_targets_stay_inside_the_session(self):
        df, _, Y = ds.intraday_dataset(self.company)
        last_of_day = df["timestamp"].dt.date != df["timestamp"].dt.date.shift(-1)
        self.assertTrue(Y.loc[last_of_day, "fwd_3"].isna().all())

    def test_train_save_and_predict(self):
        manifest = engine.train_stock(self.company, only=["ridge", "et"])
        files = sorted(p.name for p in (engine.stock_dir("TST.NS")).iterdir())
        self.assertEqual(files, ["daily_h1.joblib", "daily_h5.joblib", "intraday_h12.joblib", "intraday_h3.joblib",
                                 "intraday_h6.joblib", "manifest.json"])
        board = manifest["timeframes"]["intraday"]["horizons"]["3"]["leaderboard"]
        self.assertEqual({m["key"] for m in board}, {"ridge", "et", "ens_all", "ens_top3"})

        out = engine.predict_stock(self.company, "intraday")
        self.assertEqual([f["label"] for f in sorted(out["forecasts"], key=lambda f: f["horizon"])],
                         ["15 min", "30 min", "60 min"])
        f = out["forecasts"][0]
        self.assertLess(f["range_80"][0], f["range_80"][1])
        self.assertTrue(out["candles"])
        daily = engine.predict_stock(self.company, "daily")
        self.assertEqual(len(daily["forecasts"]), 2)

    def test_untrained_stock(self):
        with self.assertRaises(engine.NotTrained):
            engine.predict_stock(self.company, "intraday")

    def test_api(self):
        engine.train_stock(self.company, timeframes=("daily",), only=["ridge"])
        r = self.client.get("/api/predict/TST.NS/?timeframe=daily")
        self.assertEqual(r.status_code, 200)
        json.loads(r.content)                                   # valid JSON (no NaN)
        self.assertEqual(self.client.get("/api/predict/TST.NS/?timeframe=intraday").status_code, 404)
        self.assertEqual(self.client.get("/api/predict/TST.NS/?timeframe=weekly").status_code, 400)
        stocks = self.client.get("/api/predict/").json()["stocks"]
        self.assertEqual([s["company"] for s in stocks], ["TST.NS"])
        self.assertEqual(self.client.get("/stock_prediction/").status_code, 200)

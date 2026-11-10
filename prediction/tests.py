import json
import shutil
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from unittest import mock

import numpy as np
import pandas as pd
from django.test import TestCase, override_settings
from django.urls import reverse

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

    def test_include_news_tilts_the_prediction(self):
        engine.train_stock(self.company, timeframes=("daily",), only=["ridge"])
        signal = {"bias": 0.8, "intensity": 0.6, "label": "Positive", "stock_bias": 0.9, "market_bias": 0.1,
                  "fresh_count": 3, "positive": 3, "negative": 0, "items": [], "since": "2026-10-22T15:30:00+05:30"}
        news = mock.Mock(return_value=signal)
        plain = engine.predict_stock(self.company, "daily")
        old = engine.predict_stock(self.company, "daily", news=news)          # data ends in 2025: not current
        self.assertIsNone(old["news"])
        self.assertIn("latest session", old["news_status"])
        news.assert_not_called()
        with mock.patch.object(engine, "news_applies", return_value=True):
            out = engine.predict_stock(self.company, "daily", news=news)
        self.assertEqual(out["news"]["widen"], 1.3)
        for f0, f in zip(plain["forecasts"], out["forecasts"]):
            self.assertAlmostEqual(f["base"]["predicted_return_pct"], f0["predicted_return_pct"])
            self.assertGreater(f["news_tilt_pct"], 0)                         # positive news -> higher
            self.assertAlmostEqual(f["predicted_return_pct"], f0["predicted_return_pct"] + f["news_tilt_pct"])
            w0, w = f0["range_80"][1] - f0["range_80"][0], f["range_80"][1] - f["range_80"][0]
            self.assertAlmostEqual(w / w0, 1.3, places=2)                     # widened by the intensity
            self.assertNotIn("_r", f)
        with mock.patch.object(engine, "news_applies", return_value=True),              mock.patch("news.signal.news_signal", return_value=signal):
            d = self.client.get("/api/predict/TST.NS/?timeframe=daily&news=1").json()
        self.assertEqual(d["news"]["label"], "Positive")

    def test_news_applies_only_to_current_data(self):
        now = datetime(2026, 10, 23, 11, 0, tzinfo=IST)                       # a Friday
        self.assertTrue(engine.news_applies("intraday", pd.Timestamp("2026-10-23 10:55", tz=IST), now))
        self.assertFalse(engine.news_applies("intraday", pd.Timestamp("2026-10-22 15:25", tz=IST), now))
        self.assertTrue(engine.news_applies("daily", date(2026, 10, 22), now))   # previous session's close
        self.assertFalse(engine.news_applies("daily", date(2026, 10, 20), now))


class IndexPredictionTests(TestCase):
    """Indices on the Stock Prediction and Candle Predictor pages, and the Trained stocks list."""

    def setUp(self):
        from django.core.cache import cache
        from market.models import Company, DailyData, IndexDaily, IndexFiveMinute, MarketIndex
        cache.clear()
        self.nifty = MarketIndex.objects.create(key="NIFTY", name="NIFTY 50", exchange="NSE", angel_token="99926000")
        self.vix = MarketIndex.objects.create(key="INDIAVIX", name="INDIA VIX", exchange="NSE", angel_token="99926017", sort=1)
        start = datetime(2026, 11, 2, 3, 45, tzinfo=timezone.utc)      # 09:15 IST
        IndexFiveMinute.objects.bulk_create([
            IndexFiveMinute(index=self.nifty, timestamp=start + timedelta(minutes=5 * i), date=date(2026, 11, 2),
                            open=100 + i, high=101 + i, low=99 + i, close=100.5 + i) for i in range(3)])
        IndexDaily.objects.create(index=self.nifty, date=date(2026, 11, 2), open=1, high=2, low=1, close=25000)
        IndexDaily.objects.create(index=self.vix, date=date(2026, 11, 2), open=1, high=2, low=1, close=17.5)
        self.low = Company.objects.create(code="LOW.NS", name="Low Price")
        self.high = Company.objects.create(code="HIGH.NS", name="High Price")
        self.edge = Company.objects.create(code="EDGE.NS", name="Exactly Twenty")
        for c, close in ((self.low, 15), (self.high, 500), (self.edge, 20)):
            DailyData.objects.create(company=c, date=date(2026, 11, 2), open=1, high=1, low=1, close=close)

    def no_angel(self):
        from paper_trading import angel
        return mock.patch.object(angel, "configured", return_value=False)

    def test_a_code_finds_an_index_or_a_stock(self):
        from django.http import Http404
        from prediction import indices
        self.assertEqual(indices.asset_or_404("NIFTY.IDX").name, "NIFTY 50")
        self.assertEqual(indices.asset_or_404("LOW.NS").name, "Low Price")
        with self.assertRaises(Http404):
            indices.asset_or_404("NOPE.IDX")

    def test_index_history_reads_like_a_stock_in_ist(self):
        from prediction import data, indices
        df = data.load_historical_candles(indices.find("NIFTY.IDX"))
        self.assertEqual(len(df), 3)
        self.assertEqual(df["timestamp"].iloc[0].strftime("%H:%M"), "09:15")
        self.assertEqual((df["close"].iloc[-1], df["volume"].iloc[-1]), (102.5, 0.0))

    def test_history_keeps_the_latest_rows(self):
        from prediction import indices
        with mock.patch.object(indices, "HISTORY_ROWS", 2):
            df = indices.history(indices.find("NIFTY.IDX"))
        self.assertEqual(list(df["close"]), [101.5, 102.5])

    def test_live_candles_come_from_angel_one_and_never_yahoo(self):
        from paper_trading import angel
        from prediction import datasets, indices
        rows = {"data": [["2026-11-02T09:25:00+05:30", 1, 2, 1, 150.0, 0], ["2026-11-02T09:30:00+05:30", 1, 2, 1, 151.0, 0]]}
        with mock.patch.object(angel, "read_any", return_value=("vivek", rows)), mock.patch("yfinance.Ticker") as ticker:
            df = datasets.load_intraday(indices.find("NIFTY.IDX"), live=True)
        ticker.assert_not_called()
        self.assertEqual(list(df["close"]), [100.5, 101.5, 150.0, 151.0])     # Angel One's newer version of 09:25 wins

    def test_a_failed_angel_one_call_falls_back_to_stored_candles(self):
        from paper_trading import angel
        from prediction import datasets, indices
        with mock.patch.object(angel, "read_any", side_effect=angel.AngelError("down")):
            df = datasets.load_intraday(indices.find("NIFTY.IDX"), live=True)
        self.assertEqual(len(df), 3)

    def test_a_real_vix_jump_keeps_the_daily_history(self):
        from market.models import IndexDaily
        from prediction import datasets, indices
        closes = [12, 12.5, 25, 13, 13.5]                          # doubles for a day, then falls back
        for i, c in enumerate(closes):
            IndexDaily.objects.create(index=self.vix, date=date(2026, 3, 2) + timedelta(days=i), open=c, high=c, low=c, close=c)
        self.assertEqual(len(datasets.load_daily(indices.find("INDIAVIX.IDX"))), 6)

    def test_pages_show_the_index_dropdown(self):
        for url in (reverse("stock_prediction_page"), reverse("candle_predictor_page")):
            r = self.client.get(url)
            self.assertContains(r, 'id="indexSelect"')
            self.assertContains(r, 'value="NIFTY.IDX"')

    def test_index_without_models_asks_for_training(self):
        with mock.patch.object(engine, "load_manifest", return_value=None):
            r = self.client.get(reverse("predict_company", args=["NIFTY.IDX"]))
        self.assertEqual(r.status_code, 404)
        self.assertTrue(r.json()["not_trained"])

    def test_news_is_not_used_for_an_index(self):
        from prediction import engine
        with mock.patch.object(engine, "predict_stock", return_value={"company": "NIFTY.IDX"}) as p:
            d = self.client.get(reverse("predict_company", args=["NIFTY.IDX"]), {"news": "1"}).json()
        self.assertIsNone(p.call_args.kwargs["news"])
        self.assertIn("isn't used", d["news_status"])

    def overview(self, codes):
        from prediction import engine
        rows = [{"company": c, "name": c, "horizons": [], "edges": 0} for c in codes]
        with mock.patch.object(engine, "overview", return_value=rows), self.no_angel():
            return self.client.get(reverse("predict_all")).json()

    def test_list_has_ltp_and_drops_stocks_at_20_or_below(self):
        d = self.overview(["LOW.NS", "HIGH.NS", "EDGE.NS"])
        self.assertEqual([(s["company"], s["ltp"]) for s in d["stocks"]], [("HIGH.NS", 500.0)])
        self.assertEqual(d["hidden"], 2)

    def test_an_index_stays_at_any_level_and_comes_first(self):
        d = self.overview(["HIGH.NS", "INDIAVIX.IDX", "NIFTY.IDX"])
        self.assertEqual([s["company"] for s in d["stocks"]], ["INDIAVIX.IDX", "NIFTY.IDX", "HIGH.NS"])
        self.assertEqual(d["stocks"][0]["ltp"], 17.5)

    def test_a_stock_with_no_price_is_kept(self):
        d = self.overview(["NEW.NS"])
        self.assertEqual((d["stocks"][0]["company"], d["stocks"][0]["ltp"], d["hidden"]), ("NEW.NS", None, 0))

    def test_ltp_comes_from_angel_one_by_exchange(self):
        from market.models import MarketIndex
        from paper_trading import angel
        from prediction import indices
        MarketIndex.objects.create(key="SENSEX", name="SENSEX", exchange="BSE", angel_token="99919000", sort=2)
        answers = {"NSE": {"1": {"ltp": 30.5}, "99926000": {"ltp": 25100.0}}, "BSE": {"99919000": {"ltp": 81000.0}}}
        with mock.patch.object(angel, "instrument", return_value={"token": "1"}), \
             mock.patch.object(angel, "quotes", side_effect=lambda tokens, exchange: answers[exchange]):
            out = indices.last_prices(["HIGH.NS", "NIFTY.IDX", "SENSEX.IDX"])
        self.assertEqual(out, {"HIGH.NS": 30.5, "NIFTY.IDX": 25100.0, "SENSEX.IDX": 81000.0})



class FiveXOnlyTests(TestCase):
    """Stock Prediction lists, and bulk training covers, only the 5x stocks."""

    def setUp(self):
        for code, lev, sector in (("FIVE.NS", 5.0, "Tech"), ("TWO.NS", 2.0, "Tech"), ("NONE.NS", 0.0, "Tech"),
                                  ("NEW.NS", None, "Tech"), ("FUND.NS", 5.0, "ETF")):
            Company.objects.create(name=code, code=code, sector=sector, leverage=lev)

    def test_dropdown_lists_only_5x(self):
        with mock.patch.object(engine, "list_manifests", return_value=[{"company": "FIVE.NS"}]):
            r = self.client.get("/stock_prediction/")
        self.assertEqual(r.context["companies"], [{"code": "FIVE.NS", "name": "FIVE.NS", "trained": True}])
        self.assertContains(r, 'value="FIVE.NS"')
        self.assertNotContains(r, 'value="TWO.NS"')

    def test_indices_stay_in_the_dropdown(self):
        from . import indices
        with mock.patch.object(engine, "list_manifests", return_value=[]):
            r = self.client.get("/stock_prediction/")
        self.assertEqual(len(r.context["indices"]), len(indices.all_assets()))

    def test_train_all_covers_only_5x(self):
        from io import StringIO
        from django.core.management import call_command
        with mock.patch.object(engine, "list_manifests", return_value=[]), \
             mock.patch.object(engine, "train_stock", return_value={"timeframes": {}}) as train:
            call_command("train_models", "--all", stdout=StringIO())
        self.assertEqual([c.args[0].code for c in train.call_args_list], ["FIVE.NS"])

    def test_top_covers_only_5x(self):
        from prediction.management.commands.train_models import most_traded
        for company in Company.objects.all():
            DailyData.objects.create(company=company, date=date(2026, 11, 6), open=10, high=11, low=9, close=10, volume=1000)
        self.assertEqual(most_traded(10), ["FIVE.NS"])

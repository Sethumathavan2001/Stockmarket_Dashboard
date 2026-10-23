from unittest import mock

import pandas as pd
from django.core.cache import cache
from django.test import TestCase

from . import indices

NSE = {"timestamp": "19-Oct-2026 13:46", "data": [
    {"index": "NIFTY 50", "last": 22861.7, "variation": -284.45, "percentChange": -1.23, "open": 23150.85, "high": 23160,
     "low": 22840, "previousClose": 23146.15, "advances": 5, "declines": 45},
    {"index": "INDIA VIX", "last": 14.87, "variation": 0.81, "percentChange": 5.7, "previousClose": 14.06},
    {"index": "NIFTY SOMETHING ELSE", "last": 1},
]}
SENSEX = pd.DataFrame({"Open": [74000.0, 74700.0], "High": [74900.0, 74800.0], "Low": [73900.0, 73700.0],
                       "Close": [74729.27, 73800.0]},
                      index=pd.DatetimeIndex([pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timedelta(days=1),
                                              pd.Timestamp.now(tz="Asia/Kolkata")]))


class IndicesTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_strip_data(self):
        with mock.patch("news.fetch._nse_json", return_value=NSE), \
             mock.patch("prediction.datasets._yahoo", return_value=SENSEX):
            d = self.client.get("/api/market/indices/").json()
        names = [i["short"] for i in d["indices"]]
        self.assertEqual(names, ["NIFTY 50", "SENSEX", "INDIA VIX"])           # SENSEX second; unknown indices skipped
        nifty, sensex = d["indices"][0], d["indices"][1]
        self.assertEqual((nifty["last"], nifty["pct"], nifty["delayed"]), (22861.7, -1.23, False))
        self.assertTrue(sensex["delayed"])
        self.assertAlmostEqual(sensex["change"], 73800.0 - 74729.27, places=2)
        self.assertEqual(d["errors"], [])

    def test_cached_and_falls_back_to_last_good_values(self):
        with mock.patch("news.fetch._nse_json", return_value=NSE) as nse, \
             mock.patch("prediction.datasets._yahoo", return_value=SENSEX):
            indices.live_indices()
            indices.live_indices()
        self.assertEqual(nse.call_count, 1)                                     # second call served from the cache
        cache.delete(indices.CACHE_KEY)
        with mock.patch("news.fetch._nse_json", side_effect=RuntimeError("NSE refused")), \
             mock.patch("prediction.datasets._yahoo", return_value=SENSEX):
            d = indices.live_indices()
        self.assertIn("NSE", d["errors"])
        nifty = next(i for i in d["indices"] if i["short"] == "NIFTY 50")
        self.assertTrue(nifty["stale"])

    def test_navbar_menus_and_strip_on_every_page(self):
        r = self.client.get("/news/")
        self.assertContains(r, 'id="siteTicker"')
        self.assertContains(r, ">Data")
        self.assertContains(r, "<b>Paper Trading 2</b>")
        self.assertContains(r, "/api/market/indices/")


class IndexChartTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_chart_has_today_and_previous_day_ohlc(self):
        now = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
        idx = pd.DatetimeIndex([now - pd.Timedelta(days=1) + pd.Timedelta(hours=9, minutes=15 + 5 * i) for i in range(3)]
                               + [now + pd.Timedelta(hours=9, minutes=15 + 5 * i) for i in range(2)], name="Datetime")
        raw = pd.DataFrame({"Open": [100, 101, 102, 110, 111], "High": [101, 105, 103, 112, 113],
                            "Low": [99, 100, 98, 109, 108], "Close": [101, 102, 103, 111, 112], "Volume": [0] * 5}, index=idx)
        with mock.patch("prediction.datasets._yahoo", return_value=raw):
            d = self.client.get("/api/market/indices/nifty50/chart/").json()
        self.assertEqual(d["previous"], {"date": d["previous"]["date"], "open": 100.0, "high": 105.0, "low": 98.0, "close": 103.0})
        self.assertEqual(d["today"]["open"], 110.0)
        self.assertEqual(len(d["candles"]), 2)
        self.assertEqual(self.client.get("/api/market/indices/nope/chart/").status_code, 404)


class StockIndicesTests(TestCase):
    def setUp(self):
        from . import index_membership as im
        self.im = im
        im._mem["data"] = None
        import tempfile
        from pathlib import Path
        self.p = mock.patch.object(im, "STORE", Path(tempfile.mkdtemp()) / "c.json")
        self.p.start()
        self.addCleanup(self.p.stop)

    def test_membership_from_constituent_lists(self):
        fake = {"as_of": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), "failed": [],
                "members": {"NIFTY 50": ["HDFCBANK", "TCS"], "NIFTY BANK": ["HDFCBANK"], "NIFTY 500": ["HDFCBANK", "HEXT"]}}
        with mock.patch.object(self.im, "_download", return_value=fake) as dl:
            d = self.client.get("/api/market/stock-indices/HDFCBANK.NS/").json()
            self.im.indices_for("HEXT.NS")                                        # same day: no second download
        self.assertEqual(dl.call_count, 1)
        self.assertEqual([i["name"] for i in d["indices"]], ["NIFTY 50", "NIFTY 500", "NIFTY BANK"])
        self.assertEqual({i["name"]: i["key"] for i in d["indices"]}["NIFTY BANK"], "banknifty")
        self.assertTrue(self.im.STORE.exists())                                  # kept on disk for restarts

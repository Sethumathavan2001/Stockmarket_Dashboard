import json
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


class AnalysisScreenerTests(TestCase):
    """Analysis page indicators (market/screener.py) and its data endpoint."""

    def frame(self, closes, vol=1000):
        d = pd.bdate_range("2025-01-01", periods=len(closes)).date
        c = pd.Series(closes, dtype=float)
        return pd.DataFrame({"date": d, "open": c.shift(1).fillna(c.iloc[0]), "high": c * 1.01, "low": c * 0.99,
                             "close": c, "volume": [vol] * (len(c) - 1) + [vol * 4]})

    def test_uptrend_scores_high_and_downtrend_low(self):
        from . import screener
        up = screener.stock_metrics(self.frame([100 * 1.004 ** i for i in range(300)]), nifty_r3m=2.0)
        self.assertEqual(up["r1d"], 0.4)
        self.assertGreater(up["vs50"], 0)
        self.assertGreater(up["vs200"], 0)
        self.assertGreater(up["rsi"], 70)                     # rises every day
        self.assertEqual(up["streak"], 299)
        self.assertEqual(up["vol_x"], 4.0)                    # last day 4x the 20-day average
        self.assertTrue(up["new_hi"])
        self.assertAlmostEqual(up["rs3m"], up["r3m"] - 2.0, places=1)
        self.assertGreaterEqual(up["score"], 70)
        down = screener.stock_metrics(self.frame([300 * 0.996 ** i for i in range(300)]), nifty_r3m=None)
        self.assertLess(down["rsi"], 30)
        self.assertTrue(down["new_lo"])
        self.assertIsNone(down["rs3m"])
        self.assertLessEqual(down["score"], 15)

    def test_golden_cross_is_found_with_its_age(self):
        from . import screener
        closes = [200 * 0.995 ** i for i in range(220)] + [66 * 1.03 ** i for i in range(60)]
        m = screener.stock_metrics(self.frame(closes), nifty_r3m=0)
        self.assertIsNotNone(m["cross"])
        self.assertGreater(m["cross"], 0)                     # 50-day crossed above the 200-day
        a = pd.Series([1, 1, 1, 3.0]); b = pd.Series([2, 2, 2, 2.0])
        self.assertEqual(screener._cross_age(a, b), 1)        # crossed above on the last bar = today
        self.assertEqual(screener._cross_age(b, a), -1)

    def test_data_endpoint(self):
        import tempfile
        from pathlib import Path
        from . import screener
        tmp = Path(tempfile.mkdtemp()) / "snap.json"
        with mock.patch.object(screener, "SNAPSHOT", tmp):
            self.assertEqual(self.client.get("/analysis/data/").status_code, 404)
            tmp.write_text(json.dumps({"as_of": "2026-10-30", "breadth": {}, "sectors": [], "stocks": []}))
            self.assertEqual(self.client.get("/analysis/data/").json()["as_of"], "2026-10-30")
            self.assertEqual(self.client.get("/analysis/").status_code, 200)
        self.assertEqual(self.client.get("/analysis/levels/").status_code, 200)


class IndexHistoryTests(TestCase):
    """Index candles, expiry scan and the Index Analysis endpoints - no network."""

    def setUp(self):
        from datetime import datetime, timedelta
        from prediction.market_time import IST
        from . import index_data
        from .models import IndexDaily, IndexFiveMinute
        self.d, self.IST = index_data, IST
        self.ix = index_data.sync_catalog()
        cache.clear()
        # two NIFTY days: 3 Mar 2025 (Mon) and 4 Mar 2025 (Tue, an expiry in this test)
        for day, base, step in ((datetime(2025, 3, 3, 9, 15, tzinfo=IST), 100.0, 0.1), (datetime(2025, 3, 4, 9, 15, tzinfo=IST), 110.0, -0.2)):
            rows = []
            for k in range(75):
                t = day + timedelta(minutes=5 * k)
                o = base + step * k
                rows.append(IndexFiveMinute(index=self.ix["NIFTY"], timestamp=t, date=t.date(), open=o, high=o + 0.5, low=o - 0.5, close=o + step))
            IndexFiveMinute.objects.bulk_create(rows)
            IndexDaily.objects.create(index=self.ix["NIFTY"], date=day.date(), open=base, high=rows[-1].high, low=base - 0.5,
                                      close=rows[-1].close)

    def test_day_stats(self):
        from .index_analysis import day_stats
        bars = [(555, 100, 101, 99.5, 100.5), (560, 100.5, 100.8, 100, 100.6), (565, 100.6, 102, 100.5, 101.8), (870, 101.8, 102.5, 101.5, 102.4)]
        s = day_stats(bars, prev_close=99)
        self.assertEqual((s["high_time"], s["low_time"], s["type"]), ("14:30", "09:15", "Trend up"))
        self.assertAlmostEqual(s["gap"], 1.01, places=2)
        self.assertAlmostEqual(s["first15"], 1.8, places=2)                 # 09:25 candle close vs open
        self.assertAlmostEqual(s["last_hour"], (102.4 / 101.8 - 1) * 100, places=2)
        self.assertEqual(day_stats([(555, 100, 103, 99, 100.2)])["type"], "Range")

    def test_expiry_kinds(self):
        from datetime import date
        from .index_analysis import expiry_kinds
        k = expiry_kinds([date(2025, 3, 4), date(2025, 3, 11), date(2025, 3, 27), date(2025, 4, 1)])
        self.assertEqual([k[x] for x in sorted(k)], ["weekly", "weekly", "monthly", "monthly"])

    def test_expiry_counts_only_when_that_days_file_has_it(self):
        from datetime import date
        from .models import IndexExpiry
        files = {   # day -> what the NSE file lists; 6 Mar's listing of 13 Mar is moved later to 12 Mar
            date(2025, 3, 4): {"NIFTY": {date(2025, 3, 4), date(2025, 3, 13)}, "BANKNIFTY": {date(2025, 3, 26)}},
            date(2025, 3, 11): {"NIFTY": {date(2025, 3, 12), date(2025, 3, 27)}},
            date(2025, 3, 12): {"NIFTY": {date(2025, 3, 12), date(2025, 3, 27)}},
        }
        with mock.patch.object(self.d, "nse_expiries", side_effect=lambda day, s: files.get(day)), \
             mock.patch.object(self.d, "bse_expiries", return_value=None), \
             mock.patch.object(self.d, "_scan_state", return_value={"nse_next": "2025-03-04", "bse_next": "2025-03-14"}), \
             mock.patch.object(self.d, "_save_scan"):
            self.d.update_expiries(self.ix, today=date(2025, 3, 14))
        got = sorted((e.index.key, e.date) for e in IndexExpiry.objects.exclude(source="schedule"))
        self.assertEqual(got, [("NIFTY", date(2025, 3, 4)), ("NIFTY", date(2025, 3, 12))])   # 13 Mar never confirmed

    def test_update_five_min_skips_a_forming_candle(self):
        from datetime import datetime, timedelta
        from .models import IndexFiveMinute
        now = datetime(2025, 3, 5, 9, 27, tzinfo=self.IST)
        rows = [(datetime(2025, 3, 5, 9, 15, tzinfo=self.IST) + timedelta(minutes=5 * k), 1, 2, 0.5, 1.5) for k in range(3)]
        with mock.patch.object(self.d, "_candles", return_value=rows):
            self.d.update_five_min(self.ix["NIFTY"], now=now)
        stored = IndexFiveMinute.objects.filter(index=self.ix["NIFTY"], date="2025-03-05").count()
        self.assertEqual(stored, 2)                                          # 09:25 candle still forming at 09:27

    def test_day_and_expiry_endpoints(self):
        from datetime import date
        from django.urls import reverse
        from .models import IndexExpiry
        IndexExpiry.objects.create(index=self.ix["NIFTY"], date=date(2025, 3, 4), source="nse_bhav")
        d = self.client.get(reverse("market:index_day"), {"index": "NIFTY", "date": "2025-03-04"}).json()
        self.assertEqual((d["date"], d["prev_day"], d["expiry"], len(d["candles"])), ("2025-03-04", "2025-03-03", "monthly", 75))
        self.assertAlmostEqual(d["stats"]["change"], (d["stats"]["close"] / d["stats"]["prev_close"] - 1) * 100, places=2)
        self.assertEqual(self.client.get(reverse("market:index_day"), {"index": "NIFTY", "date": "2025-03-08"}).json()["date"],
                         "2025-03-04")                                       # weekend -> the trading day before
        e = self.client.get(reverse("market:index_expiries"), {"index": "NIFTY"}).json()
        self.assertEqual([r["date"] for r in e["rows"]], ["2025-03-04"])
        self.assertEqual((e["summary"]["expiry"]["days"], e["summary"]["other"]), (1, None))
        self.assertFalse(self.client.get(reverse("market:index_expiries"), {"index": "INDIAVIX"}).json()["options"])
        self.assertEqual(self.client.get(reverse("market:index_analysis")).status_code, 200)


class GlobalStripAndMostActiveTests(TestCase):
    def setUp(self):
        cache.clear()

    def yahoo(self, zone="America/New_York"):
        idx = pd.DatetimeIndex(["2026-11-03", "2026-11-04"]).tz_localize(zone)
        return pd.DataFrame({"Open": [99.0, 100.0], "High": [101.0, 106.0], "Low": [98.0, 99.0], "Close": [100.0, 105.0]}, index=idx)

    def test_world_strip_keeps_each_index_when_another_fails(self):
        gift = {"giftnifty": {"LASTPRICE": 24435, "DAYCHANGE": 61, "PERCHANGE": 0.25, "EXPIRYDATE": "23-Nov-2026", "TIMESTMP": "05-Nov-2026 10:24"}}
        with mock.patch("news.fetch._nse_json", return_value=gift), mock.patch("prediction.datasets._yahoo", return_value=self.yahoo()):
            d = self.client.get("/api/market/global/").json()
        self.assertEqual([i["short"] for i in d["indices"]],
                         ["GIFT Nifty", "US NASDAQ", "US S&P 500", "Japan Nikkei 225", "Hong Kong Hang Seng"])
        self.assertEqual((d["indices"][0]["last"], d["indices"][0]["prev_close"], d["indices"][0]["key"]), (24435.0, 24374.0, None))
        us = d["indices"][1]
        self.assertAlmostEqual(us["pct"], 5.0)                               # a closed market keeps its last session's move
        self.assertEqual((us["tag"], us["delayed"], us["key"]), ("4 Nov close", False, "nasdaq"))
        cache.delete(indices.GLOBAL_KEY)
        with mock.patch("news.fetch._nse_json", side_effect=RuntimeError("refused")), \
             mock.patch("prediction.datasets._yahoo", return_value=self.yahoo()):
            d = indices.global_indices()
        self.assertEqual(d["errors"], ["GIFT Nifty"])
        self.assertTrue(d["indices"][0]["stale"])
        self.assertEqual(len(d["indices"]), 5)

    def test_us_session_is_one_day_on_the_popup_chart(self):
        ts = pd.date_range("2026-11-04 09:30", periods=78, freq="5min", tz="America/New_York")     # 20:00 to 02:25 IST
        raw = pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 0}, index=ts.rename("Datetime"))
        with mock.patch("prediction.datasets._yahoo", return_value=raw):
            d = indices.index_chart("nasdaq")
        self.assertEqual((d["session"], len(d["candles"]), d["delayed"]), ("2026-11-04", 78, True))

    def test_most_active_survives_nse_refusing(self):
        from . import views
        rows = {"data": [{"symbol": "HDFCBANK", "totalTradedValuecsv": 18579059059.2, "totalTradedVolumecsv": 24580352}]}
        with mock.patch("news.fetch._nse_json", return_value=rows):
            r = views.most_active("value")[0]
        self.assertAlmostEqual(r["value_cr"], 1857.9, places=1)
        self.assertAlmostEqual(r["volume_lakh"], 245.8, places=1)
        with mock.patch("news.fetch._nse_json", side_effect=RuntimeError("refused")):
            self.assertEqual(views.most_active("volume"), [])

    def test_navbar_has_the_world_strip(self):
        self.assertContains(self.client.get("/news/"), 'id="siteWorld"')


def behaviour_matrix(n=60):
    """A market with known answers. WITH follows the index, AGAINST does the opposite, MONDAY rises only on Mondays,
    THIN has too little turnover and SHORT traded on three days. Odd days are index up, even days down."""
    import numpy as np
    days = np.datetime64("2026-01-05") + np.arange(n)                 # 5 Jan 2026 is a Monday
    days = days[(days.astype(np.int64) + 3) % 7 < 5]
    n = len(days)
    index = np.where(np.arange(n) % 2 == 1, 1.0, -1.0).astype(np.float32)
    weekday = (days.astype(np.int64) + 3) % 7
    o2c = np.zeros((n, 5), dtype=np.float32)
    o2c[:, 0], o2c[:, 1], o2c[:, 3] = index, -index, index
    o2c[:, 2] = np.where(weekday == 0, 2.0, -0.5)
    o2c[:, 4] = np.nan
    o2c[:3, 4] = 1.0
    expiry = np.zeros((2, n), dtype=np.int8)
    expiry[0, ::5] = 1
    expiry[0, 4::20] = 2
    rows5 = np.arange(n)
    sret = np.stack([o2c, -o2c, o2c * 0.5])
    srng = np.abs(sret) + np.array([3.0, 1.0, 1.0], dtype=np.float32)[:, None, None]
    srng[1, :, 1] = 9.0                                               # AGAINST swings most in the afternoon
    ix_sret = np.stack([np.stack([index, index]), np.stack([-index, -index]), np.stack([index, index])])
    return {"days": days, "rows5": rows5, "codes": np.array(["WITH.NS", "AGAINST.NS", "MONDAY.NS", "THIN.NS", "SHORT.NS"]),
            "names": np.array(["With", "Against", "Monday", "Thin", "Short"]),
            "sectors": np.array(["Technology", "Energy", "Technology", "Energy", "Technology"]),
            "turnover": np.array([100, 100, 100, 1, 100], dtype=np.float32), "o2c": o2c, "c2c": o2c.copy(), "rng": np.abs(o2c) + 1,
            "sret": sret, "srng": srng, "ix_keys": np.array(["NIFTY", "NIFTYIT"]), "ix_names": np.array(["Nifty 50", "Nifty IT"]),
            "ix_o2c": np.stack([index, -index]), "ix_c2c": np.stack([index, -index]), "ix_rng": np.ones((2, n), dtype=np.float32),
            "ix_sret": ix_sret, "ix_srng": np.ones((3, 2, n), dtype=np.float32), "ix_expiry": expiry,
            "built": np.array("2026-11-05T12:00:00+05:30")}


class StockBehaviourEngineTests(TestCase):
    def engine(self, **kw):
        from .stock_behaviour import Engine, Filters
        return Engine(behaviour_matrix(), Filters(**{"period": "all", **kw}))

    @staticmethod
    def symbols(rows):
        return [r["symbol"] for r in rows]

    def test_the_four_index_lists(self):
        q = self.engine().quadrants()
        self.assertEqual(q["up"]["stocks_up"][0]["symbol"], "WITH")
        self.assertEqual(q["up"]["stocks_up"][0]["hit"], 100.0)
        self.assertEqual(self.symbols(q["up"]["stocks_down"])[0], "AGAINST")
        self.assertEqual(self.symbols(q["down"]["stocks_up"])[0], "AGAINST")
        self.assertEqual(self.symbols(q["down"]["stocks_down"])[0], "WITH")

    def test_low_turnover_and_short_records_stay_out(self):
        q = self.engine().quadrants()
        listed = self.symbols(q["up"]["stocks_up"] + q["down"]["stocks_up"])
        self.assertNotIn("THIN", listed)
        self.assertNotIn("SHORT", listed)
        self.assertIn("THIN", self.symbols(self.engine(turnover=0).quadrants()["up"]["stocks_up"]))

    def test_an_up_list_needs_a_positive_average(self):
        import numpy as np
        from .stock_behaviour import Engine, Filters
        data = behaviour_matrix()
        data["o2c"][:, 0] = np.where(np.arange(len(data["days"])) % 4 == 0, -9.0, 1.0)     # mostly up, losing overall
        rows = Engine(data, Filters(period="all", trend="any"))
        self.assertNotIn("WITH", self.symbols(rows.top(rows.mask(), 1)))

    def test_against_the_index_lists_are_not_empty_when_no_stock_averages_that_way(self):
        import numpy as np
        from .stock_behaviour import Engine, Filters
        data = behaviour_matrix()
        index = data["ix_o2c"][0]
        odd = (np.arange(len(index)) // 2) % 3 == 0                   # up on a third of the index's down days
        data["o2c"][:, 1] = np.where(index < 0, np.where(odd, 0.5, -3.0), 1.0)
        rows = Engine(data, Filters(period="all")).quadrants()["down"]["stocks_up"]
        self.assertEqual(rows[0]["symbol"], "AGAINST")
        self.assertLess(rows[0]["hit"], 50)
        self.assertLess(rows[0]["avg"], 0)

    def test_recheck_tells_a_lasting_habit_from_a_lucky_start(self):
        import numpy as np
        from .stock_behaviour import Engine, Filters
        rng = np.random.default_rng(1)
        data = behaviour_matrix(200)
        n, m = len(data["days"]), 60
        data.update(codes=np.array([f"S{i}.NS" for i in range(m)]), names=np.array([f"S{i}" for i in range(m)]),
                    sectors=np.array(["Technology"] * m), turnover=np.full(m, 100, dtype=np.float32))
        noise = rng.normal(size=(n, m)).astype(np.float32)
        data["o2c"] = noise
        e = Engine(data, Filters(period="all"))
        c = e.recheck(e.mask(), 1)
        self.assertGreater(c["older"], 55)                            # the best of a noisy first half look good...
        self.assertLess(abs(c["newer"] - c["typical"]), 6)            # ...and then are ordinary
        data["o2c"] = noise + np.where(np.arange(m) < 20, 1.0, 0.0).astype(np.float32)
        e = Engine(data, Filters(period="all"))
        c = e.recheck(e.mask(), 1)
        self.assertGreater(c["newer"] - c["typical"], 10)
        few = Engine(data, Filters(period="all", weekday=0, expiry="monthly"))
        self.assertIsNone(few.recheck(few.mask(), 1))

    def test_weekday_panel_finds_the_monday_stock_and_keeps_all_days(self):
        w = self.engine(weekday=2).weekdays()                         # the panel ignores its own filter
        self.assertEqual([x["label"] for x in w], ["Mon", "Tue", "Wed", "Thu", "Fri"])
        self.assertIn("MONDAY", self.symbols(w[0]["up"]))
        self.assertNotIn("MONDAY", self.symbols(w[1]["up"]))

    def test_filters_combine(self):
        e = self.engine(trend="up", weekday=0)
        days = e.mask()
        self.assertTrue(days.any())
        self.assertTrue((e.weekday[days] == 0).all())
        self.assertTrue((e.index_move(None)[days] > 0).all())
        expiry = self.engine(expiry="monthly", trend="up").mask()
        self.assertTrue((behaviour_matrix()["ix_expiry"][0][expiry] == 2).all())
        self.assertEqual(self.symbols(self.engine(sector="Energy").quadrants()["up"]["stocks_down"]), ["AGAINST"])

    def test_trend_threshold(self):
        self.assertTrue(self.engine(trend="up").mask().any())
        self.assertFalse(self.engine(trend="up", move=1.5).mask().any())

    def test_session_measures_inside_that_session(self):
        # the index's afternoon runs opposite to its day, and so do the stocks'
        e = self.engine(session="afternoon", trend="up")
        self.assertEqual(self.symbols(e.top(e.mask(), 1))[0], "WITH")
        self.assertTrue((e.index_move(None)[e.mask()] < 0).all())

    def test_volatility_lists_each_session(self):
        v = self.engine().volatility()
        self.assertEqual([s["name"] for s in v["sessions"]], ["morning", "afternoon", "evening"])
        self.assertEqual(v["sessions"][1]["biggest"][0]["symbol"], "AGAINST")
        self.assertEqual(v["sessions"][1]["peak"][0]["symbol"], "AGAINST")

    def test_sector_table_ignores_its_own_filter_and_has_every_split(self):
        s = self.engine(sector="Energy").sectors()
        self.assertEqual([r["sector"] for r in s["rows"]], ["Energy", "Technology"])
        keys = [c["key"] for c in s["columns"]]
        self.assertEqual(keys, ["all", "up", "down", "w0", "w1", "w2", "w3", "w4", "expiry"])
        energy = s["rows"][0]
        self.assertEqual(energy["cells"]["up"]["avg"], -1.0)          # only AGAINST, THIN is under the turnover
        self.assertEqual(energy["cells"]["down"]["avg"], 1.0)

    def test_expiry_compares_only_indices_with_expiries(self):
        e = self.engine().expiry()
        self.assertEqual([r["index"] for r in e["indices"]], ["NIFTY"])
        self.assertTrue(e["has_expiry"])
        row = e["indices"][0]
        self.assertEqual(row["expiry"]["days"] + row["other"]["days"], len(behaviour_matrix()["days"]))
        none = self.engine(index="NIFTYIT").expiry()
        self.assertFalse(none["has_expiry"])
        self.assertNotIn("up", none)

    def test_too_few_days_gives_empty_lists_not_an_error(self):
        e = self.engine(period="6m", weekday=0, expiry="monthly", trend="up")
        panels = e.all_panels()
        self.assertEqual(panels["quadrants"]["up"]["stocks_up"], [])
        json.dumps(panels)

    def test_wilson_prefers_the_longer_record(self):
        from .stock_behaviour import wilson_lower
        short, long = wilson_lower([6, 60], [8, 90])
        self.assertGreater(long, short)

    def test_filters_parse_bad_values_to_defaults(self):
        from .stock_behaviour import Filters
        f = Filters.parse({"weekday": "9", "trend": "sideways", "turnover": "nan", "period": "x", "move": "-1"})
        self.assertEqual((f.weekday, f.trend, f.turnover, f.period, f.move), (-1, "any", 25.0, "3y", 0.0))

    def test_profile_covers_every_split(self):
        p = self.engine().profile("WITH.NS")
        self.assertEqual(len(p["weekdays"]), 5)
        self.assertEqual(len(p["sessions"]), 3)
        nifty, it = p["indices"]
        self.assertEqual((nifty["up"]["up"], nifty["down"]["up"]), (100.0, 0.0))
        self.assertEqual(it["up"]["up"], 0.0)                         # NIFTYIT runs opposite in this market
        self.assertIsNone(it["expiry"])
        self.assertIsNone(self.engine().profile("NOPE.NS"))


class StockBehaviourBuildTests(TestCase):
    """The build over real tables, and the endpoints."""

    def setUp(self):
        import tempfile
        from datetime import datetime, timedelta
        from pathlib import Path
        from prediction.market_time import IST
        from . import behaviour_data, index_data
        from .models import Company, DailyData, FiveMinuteData, IndexDaily, IndexExpiry, IndexFiveMinute
        self.b = behaviour_data
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patcher = mock.patch.object(behaviour_data, "path", return_value=Path(folder.name) / "m.npz")
        patcher.start()
        self.addCleanup(patcher.stop)
        ix = index_data.sync_catalog()
        stock = Company.objects.create(name="Acme", code="ACME.NS", exchange="NSE", sector="Technology")
        for day, bars in ((datetime(2026, 3, 2, 9, 15, tzinfo=IST), 75), (datetime(2026, 3, 3, 9, 15, tzinfo=IST), 75),
                          (datetime(2026, 3, 4, 18, 0, tzinfo=IST), 12)):             # the last is a Muhurat-style session
            index_rows, stock_rows = [], []
            for k in range(bars):
                t = day + timedelta(minutes=5 * k)
                o = 100.0 + k
                index_rows.append(IndexFiveMinute(index=ix["NIFTY"], timestamp=t, date=t.date(), open=o, high=o + 1, low=o, close=o + 1))
                stock_rows.append(FiveMinuteData(company=stock, timestamp=t, open=o, high=o + 1, low=o, close=o + 1))
            IndexFiveMinute.objects.bulk_create(index_rows)
            FiveMinuteData.objects.bulk_create(stock_rows)
            IndexDaily.objects.create(index=ix["NIFTY"], date=day.date(), open=100, high=175, low=100, close=175)
            DailyData.objects.create(company=stock, date=day.date(), open=100, high=120, low=95, close=110, volume=2000000)
        IndexExpiry.objects.create(index=ix["NIFTY"], date=datetime(2026, 3, 3).date(), source="nse_bhav")

    def test_build_splits_sessions_in_ist_and_drops_short_days(self):
        result = self.b.build()
        self.assertEqual(result, {"days": 2, "stocks": 1, "session_days": 2})
        d = self.b.load()
        self.assertEqual(str(d["days"][0]), "2026-03-02")
        # 09:15-11:30 is 27 bars: open 100, last close 127
        self.assertAlmostEqual(float(d["sret"][0, 0, 0]), 27.0, places=3)
        # 11:30-13:30 is 24 bars: open 127, last close 151
        self.assertAlmostEqual(float(d["sret"][1, 0, 0]), (151 / 127 - 1) * 100, places=3)
        self.assertAlmostEqual(float(d["o2c"][0, 0]), 10.0, places=3)
        self.assertAlmostEqual(float(d["c2c"][1, 0]), 0.0, places=3)
        self.assertAlmostEqual(float(d["turnover"][0]), 22.0, places=2)
        keys = [str(k) for k in d["ix_keys"]]
        self.assertEqual(d["ix_expiry"][keys.index("NIFTY")].tolist(), [0, 2])

    def test_endpoints(self):
        self.assertFalse(self.client.get("/api/stock-behaviour/meta/").json()["ready"])
        self.assertEqual(self.client.get("/api/stock-behaviour/data/").status_code, 409)
        self.b.build()
        meta = self.client.get("/api/stock-behaviour/meta/").json()
        self.assertTrue(meta["ready"])
        self.assertEqual(meta["symbols"][0]["code"], "ACME.NS")
        self.assertIn("quadrants", self.client.get("/api/stock-behaviour/data/?turnover=0&period=all").json())
        self.assertEqual(self.client.get("/api/stock-behaviour/data/?index=NOPE").status_code, 400)
        self.assertEqual(self.client.get("/api/stock-behaviour/stock/?code=ACME.NS").json()["stock"]["symbol"], "ACME")
        self.assertEqual(self.client.get("/api/stock-behaviour/stock/?code=NOPE").status_code, 404)
        self.assertEqual(self.client.get("/api/stock-behaviour/build/").status_code, 405)

    def test_page_and_navbar(self):
        html = self.client.get("/stock-behaviour/").content.decode()
        self.assertIn("Stock Behaviour", html)
        menu = html[html.index('class="site-dd-panel"'):html.index("</div></div>", html.index('class="site-dd-panel"'))]
        self.assertIn("/index-analysis/", menu)

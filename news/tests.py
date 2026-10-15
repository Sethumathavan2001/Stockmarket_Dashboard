from types import SimpleNamespace as NS
from unittest import mock

from django.core.cache import cache
from django.test import TestCase

from market.models import Company

from . import fetch

RSS = b"""<?xml version="1.0" encoding="UTF-8"?><rss><channel>
<item><title>Coal India Q2 profit jumps 20%</title><link>https://example.com/a</link>
  <description>&lt;p&gt;Strong &lt;b&gt;offtake&lt;/b&gt;&lt;/p&gt;</description><pubDate>Thu, 15 Oct 2026 09:30:00 +0530</pubDate></item>
<item><title>RBI keeps rates unchanged</title><link>https://example.com/b</link><pubDate>Thu, 15 Oct 2026 10:00:00 +0530</pubDate></item>
<item><title>Bad link</title><link>javascript:alert(1)</link></item>
</channel></rss>"""

GOOGLE = b"""<?xml version="1.0" encoding="UTF-8"?><rss><channel>
<item><title>Coal India\xe2\x80\x99s offtake rises - Moneycontrol</title><link>https://news.google.com/x</link>
  <source url="https://moneycontrol.com">Moneycontrol</source><pubDate>Wed, 14 Oct 2026 08:00:00 GMT</pubDate></item>
</channel></rss>"""

NSE = [
    {"symbol": "COALINDIA", "sm_name": "Coal India Limited", "desc": "Outcome of Board Meeting",
     "attchmntText": "Results for the quarter", "an_dt": "15-Oct-2026 16:05:00",
     "attchmntFile": "https://nsearchives.nseindia.com/x.pdf", "smIndustry": "Mining"},
    {"symbol": "OTHERCO", "sm_name": "Other Co", "desc": "Copy of Newspaper Publication",
     "attchmntText": "Order of the court", "an_dt": "15-Oct-2026 11:00:00", "attchmntFile": ""},
]


class FetchTests(TestCase):
    def setUp(self):
        cache.clear()
        Company.objects.create(name="Coal India Limited", code="COALINDIA.NS", exchange="NSE", sector="Energy")
        Company.objects.create(name="Larsen & Toubro Limited", code="LT.NS", exchange="NSE")

    def test_parse_rss_cleans_and_drops_unsafe_links(self):
        items = fetch.parse_rss(RSS, "ET")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["summary"], "Strong offtake")
        self.assertTrue(items[0]["time"].startswith("2026-10-15T09:30"))

    def test_google_news_publisher_and_encoding(self):
        it = fetch.parse_rss(GOOGLE, "Google News")[0]
        self.assertEqual(it["source"], "Moneycontrol")
        self.assertEqual(it["title"], "Coal India’s offtake rises")
        self.assertTrue(it["time"].startswith("2026-10-14T13:30"))          # GMT -> IST

    def test_short_names_keep_india(self):
        self.assertEqual(fetch._short_name("Coal India Limited"), "Coal India")
        self.assertEqual(fetch._short_name("Bharat Electronics Ltd."), "Bharat Electronics")

    def test_mentions(self):
        self.assertEqual(fetch.mentions("Coal India and Larsen & Toubro gain"), ["COALINDIA.NS", "LT.NS"])
        self.assertEqual(fetch.mentions("RBI policy: what it means"), [])

    def test_market_news_survives_a_dead_feed(self):
        def fake_get(url, **kw):
            if "moneycontrol" in url:
                raise fetch.SourceError("HTTP 503")
            return NS(content=RSS)
        with mock.patch.object(fetch, "_get", side_effect=fake_get):
            out = fetch.market_news()
        self.assertEqual(len(out["items"]), 2)                               # duplicates across feeds removed
        self.assertEqual(out["items"][0]["title"], "RBI keeps rates unchanged")  # newest first
        self.assertEqual(out["items"][1]["mentions"], ["COALINDIA.NS"])
        down = [s for s in out["sources"] if not s["ok"]]
        self.assertEqual([s["name"] for s in down], ["Moneycontrol"])

    def test_announcements_flag_importance_by_category(self):
        with mock.patch.object(fetch, "_nse_json", return_value=NSE):
            out = fetch.announcements(1)
        coal, other = out["items"]
        self.assertTrue(coal["important"])
        self.assertEqual(coal["code"], "COALINDIA.NS")
        self.assertFalse(other["important"])                                 # "order" in the text is not enough
        self.assertIsNone(other["code"])

    def test_stock_news_reports_partial_failure(self):
        c = Company.objects.get(code="COALINDIA.NS")
        with mock.patch.object(fetch, "_nse_json", side_effect=fetch.SourceError("NSE refused")), \
                mock.patch.object(fetch, "_get", return_value=NS(content=GOOGLE)):
            out = fetch.stock_news(c)
        self.assertEqual(len(out["news"]), 1)
        self.assertEqual(out["announcements"], [])
        self.assertIn("NSE announcements", out["errors"][0])


class ViewTests(TestCase):
    def setUp(self):
        cache.clear()
        Company.objects.create(name="Coal India Limited", code="COALINDIA.NS", exchange="NSE")

    def test_page_and_api(self):
        r = self.client.get("/news/")
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "NSE announcements")
        with mock.patch.object(fetch, "_get", return_value=NS(content=RSS)):
            self.assertEqual(len(self.client.get("/api/news/market/").json()["items"]), 2)
        with mock.patch.object(fetch, "_nse_json", side_effect=fetch.SourceError("down")):
            self.assertEqual(self.client.get("/api/news/announcements/?days=x").status_code, 502)
        self.assertEqual(self.client.get("/api/news/stock/NOPE/").status_code, 404)

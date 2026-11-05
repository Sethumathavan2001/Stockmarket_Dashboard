import json
import random
import threading
from datetime import datetime, time, timedelta
from decimal import Decimal
from unittest import mock

import pandas as pd
from django.core.cache import cache
from django.db import connection
from django.test import TestCase, TransactionTestCase
from django.urls import reverse

from market.models import Company, DailyData, FiveMinuteData

from . import services as svc
from .models import PaperAccount, PaperOrder, PaperPosition

D = Decimal


def make_candles(company, day, closes):
    """One 5-min candle per close, starting 09:15 IST, each with a 1-point range around it."""
    start = datetime.combine(day, time(9, 15), tzinfo=svc.IST)
    rows = []
    prev = closes[0]
    for i, close in enumerate(closes):
        c = D(str(close))
        o = D(str(prev))
        rows.append(
            FiveMinuteData(
                timestamp=start + timedelta(minutes=5 * i),
                company=company,
                open=o,
                high=max(o, c) + D("1"),
                low=min(o, c) - D("1"),
                close=c,
            )
        )
        prev = close
    FiveMinuteData.objects.bulk_create(rows)
    return rows


class BaseCase(TestCase):
    def setUp(self):
        cache.clear()      # DB primary keys repeat between tests, so cached live data must not leak
        self.company = Company.objects.create(name="Test Co", code="TST")
        self.other = Company.objects.create(name="Other Co", code="OTH")
        self.day = svc.today_ist() - timedelta(days=3)
        make_candles(self.company, self.day, [100, 102, 101, 105, 103, 99])
        self.account = PaperAccount.objects.create(owner_key="test", balance=D("10000"))

    def trade(self, side, qty, price, company=None):
        return svc.execute_order(
            account=self.account,
            company=company or self.company,
            side=side,
            quantity=qty,
            price=D(str(price)),
            mode=svc.LIVE,
            replay_date=None,
            market_time=None,
        )

    def balance(self):
        self.account.refresh_from_db()
        return self.account.balance

    def post(self, url, payload):
        return self.client.post(url, json.dumps(payload), content_type="application/json")


class WalletTests(BaseCase):
    def test_add_amount(self):
        acct = svc.add_to_wallet(self.account, "2500.50")
        self.assertEqual(acct.balance, D("12500.50"))

    def test_rejects_bad_amounts(self):
        for bad in ["0", "-5", "abc", "", None, "NaN", "Infinity", "999999999999"]:
            with self.assertRaises(svc.TradeError, msg=repr(bad)):
                svc.add_to_wallet(self.account, bad)
        self.assertEqual(self.balance(), D("10000"))

    def test_api_add_then_balance_visible(self):
        r = self.post(reverse("paper_trading:wallet_add"), {"amount": 1000})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["balance"], 1000.0)
        r = self.client.get(reverse("paper_trading:account"), {"mode": "live"})
        self.assertEqual(r.json()["balance"], 1000.0)

    def test_api_rejects_negative(self):
        r = self.post(reverse("paper_trading:wallet_add"), {"amount": -10})
        self.assertEqual(r.status_code, 400)


class EngineTests(BaseCase):
    def test_buy_deducts_wallet_and_opens_long(self):
        _, pos, _ = self.trade("BUY", 10, 100)
        self.assertEqual(self.balance(), D("9000"))
        self.assertEqual((pos.quantity, pos.avg_price), (10, D("100")))

    def test_insufficient_balance_rejected_and_nothing_changes(self):
        with self.assertRaises(svc.TradeError):
            self.trade("BUY", 101, 100)  # needs 10,100 > 10,000
        self.assertEqual(self.balance(), D("10000"))
        self.assertFalse(PaperOrder.objects.exists())
        self.assertFalse(PaperPosition.objects.exclude(quantity=0).exists())

    def test_exact_balance_is_allowed(self):
        self.trade("BUY", 100, 100)
        self.assertEqual(self.balance(), D("0"))

    def test_sell_long_at_profit(self):
        self.trade("BUY", 10, 100)
        order, pos, _ = self.trade("SELL", 10, 110)
        self.assertEqual(order.realized_pnl, D("100"))
        self.assertEqual(self.balance(), D("10100"))
        self.assertEqual((pos.quantity, pos.cost_basis), (0, D("0")))

    def test_sell_long_at_loss(self):
        self.trade("BUY", 10, 100)
        order, _, _ = self.trade("SELL", 10, 95)
        self.assertEqual(order.realized_pnl, D("-50"))
        self.assertEqual(self.balance(), D("9950"))

    def test_average_price_when_adding(self):
        self.trade("BUY", 10, 100)
        _, pos, _ = self.trade("BUY", 10, 110)
        self.assertEqual(pos.quantity, 20)
        self.assertEqual(pos.avg_price, D("105"))

    def test_partial_close_keeps_average(self):
        self.trade("BUY", 10, 100)
        self.trade("BUY", 10, 110)  # avg 105
        order, pos, _ = self.trade("SELL", 5, 120)
        self.assertEqual(order.realized_pnl, D("75"))  # (120-105)*5
        self.assertEqual((pos.quantity, pos.avg_price), (15, D("105")))

    def test_short_sell_blocks_notional_then_cover_at_profit(self):
        _, pos, _ = self.trade("SELL", 10, 100)
        self.assertEqual(pos.quantity, -10)
        self.assertEqual(self.balance(), D("9000"))  # notional blocked
        order, pos, _ = self.trade("BUY", 10, 90)
        self.assertEqual(order.realized_pnl, D("100"))
        self.assertEqual(self.balance(), D("10100"))
        self.assertEqual(pos.quantity, 0)

    def test_short_cover_at_loss(self):
        self.trade("SELL", 10, 100)
        order, _, _ = self.trade("BUY", 10, 108)
        self.assertEqual(order.realized_pnl, D("-80"))
        self.assertEqual(self.balance(), D("9920"))

    def test_short_needs_wallet_too(self):
        with self.assertRaises(svc.TradeError):
            self.trade("SELL", 101, 100)

    def test_flip_long_to_short_in_one_order(self):
        self.trade("BUY", 10, 100)  # wallet 9000
        order, pos, _ = self.trade("SELL", 15, 110)
        # closes 10 @ +10 = +100 -> wallet 9000 + 1100 = 10100; opens short 5 @110 blocks 550
        self.assertEqual(order.realized_pnl, D("100"))
        self.assertEqual((pos.quantity, pos.avg_price), (-5, D("110")))
        self.assertEqual(self.balance(), D("9550"))

    def test_flip_rejected_if_remainder_unaffordable(self):
        self.trade("BUY", 100, 100)  # wallet 0
        # closing 100 @100 returns 10,000; remainder 101 @100 needs 10,100 -> reject
        with self.assertRaises(svc.TradeError):
            self.trade("SELL", 201, 100)
        self.assertEqual(self.balance(), D("0"))
        self.assertEqual(PaperPosition.objects.get().quantity, 100)

    def test_positions_are_per_company(self):
        self.trade("BUY", 5, 100, self.company)
        self.trade("SELL", 5, 100, self.other)
        self.assertEqual(PaperPosition.objects.exclude(quantity=0).count(), 2)

    def test_conservation_invariant_over_random_orders(self):
        """wallet + blocked basis must always equal deposits + realized P&L, to the paisa."""
        rng = random.Random(7)
        self.account.balance = D("50000")
        self.account.save()
        deposits, realized_total = D("50000"), D("0")
        for _ in range(300):
            side = rng.choice(["BUY", "SELL"])
            qty = rng.randint(1, 40)
            price = D(str(round(rng.uniform(90, 120), 2)))
            company = rng.choice([self.company, self.other])
            try:
                order, _, _ = self.trade(side, qty, price, company)
                realized_total += order.realized_pnl
            except svc.TradeError:
                pass
            blocked = sum(p.cost_basis for p in PaperPosition.objects.all())
            self.assertEqual(self.balance() + blocked, deposits + realized_total)
            self.assertGreaterEqual(self.balance(), 0)


class ReplayApiTests(BaseCase):
    def setUp(self):
        super().setUp()
        self.client.post(reverse("paper_trading:wallet_add"), json.dumps({"amount": 10000}), content_type="application/json")

    def candle(self, i):
        return svc.load_replay_candles(self.company, self.day)[i]

    def order(self, side, qty, i, price=None, day=None, company="TST"):
        c = self.candle(i)
        return self.post(
            reverse("paper_trading:order"),
            {
                "company": company,
                "side": side,
                "quantity": qty,
                "mode": "replay",
                "date": (day or self.day).isoformat(),
                "market_time": c["x"],
                "price": price if price is not None else c["c"],
            },
        )

    def test_candles_endpoint_returns_the_day(self):
        r = self.client.get(reverse("paper_trading:candles", args=["TST"]), {"mode": "replay", "date": self.day.isoformat()})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(len(data["candles"]), 6)
        self.assertTrue(data["candles"][0]["x"].endswith("+05:30"))
        self.assertIn("T09:15:00", data["candles"][0]["x"])

    def test_today_and_future_dates_rejected(self):
        for d in (svc.today_ist(), svc.today_ist() + timedelta(days=2)):
            r = self.client.get(reverse("paper_trading:candles", args=["TST"]), {"mode": "replay", "date": d.isoformat()})
            self.assertEqual(r.status_code, 400)

    def test_day_without_data_gives_clear_error(self):
        d = self.day - timedelta(days=10)
        r = self.client.get(reverse("paper_trading:candles", args=["TST"]), {"mode": "replay", "date": d.isoformat()})
        self.assertEqual(r.status_code, 400)
        self.assertIn("No 5-minute candles", r.json()["error"])

    def test_date_range(self):
        r = self.client.get(reverse("paper_trading:date_range", args=["TST"]))
        self.assertEqual(r.json(), {"min": self.day.isoformat(), "max": self.day.isoformat()})
        r = self.client.get(reverse("paper_trading:date_range", args=["OTH"]))
        self.assertEqual(r.json(), {"min": None, "max": None})

    def test_buy_then_sell_round_trip_updates_wallet(self):
        r = self.order("BUY", 10, 0, price="100.00")
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()["balance"], 9000.0)
        r = self.order("SELL", 10, 3, price="105.00")
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()["order"]["realized_pnl"], 50.0)
        self.assertEqual(r.json()["balance"], 10050.0)
        self.assertIsNone(r.json()["position"])

    def test_price_outside_candle_range_rejected(self):
        c = self.candle(0)  # 100 candle: range 99..101
        r = self.order("BUY", 1, 0, price=str(c["h"] + 5))
        self.assertEqual(r.status_code, 400)
        self.assertIn("outside", r.json()["error"])

    def test_wrong_date_or_missing_candle_rejected(self):
        r = self.order("BUY", 1, 0, day=self.day - timedelta(days=1))
        self.assertEqual(r.status_code, 400)

    def test_insufficient_wallet_message(self):
        r = self.order("BUY", 500, 0, price="100.00")
        self.assertEqual(r.status_code, 400)
        self.assertIn("Insufficient wallet balance", r.json()["error"])

    def test_bad_inputs(self):
        for qty in (0, -3, 1.5, "abc", None):
            self.assertEqual(self.order("BUY", qty, 0).status_code, 400, qty)
        for bad_price in ("NaN", "Infinity", "-5", "0", "abc"):
            self.assertEqual(self.order("BUY", 1, 0, price=bad_price).status_code, 400, bad_price)
        self.assertEqual(self.order("HOLD", 1, 0).status_code, 400)
        self.assertEqual(self.order("BUY", 1, 0, company="NOPE").status_code, 400)

    def test_positions_orders_and_resume_time(self):
        self.order("BUY", 4, 2, price="101.00")
        r = self.client.get(
            reverse("paper_trading:account"),
            {"mode": "replay", "date": self.day.isoformat(), "company": "TST"},
        )
        data = r.json()
        self.assertEqual(len(data["positions"]), 1)
        self.assertEqual(data["positions"][0]["quantity"], 4)
        self.assertEqual(data["positions"][0]["side"], "LONG")
        self.assertEqual(data["blocked"], 404.0)
        self.assertEqual(len(data["orders"]), 1)
        self.assertEqual(data["resume_market_time"], self.candle(2)["x"])

        # Live context is separate: no positions, no resume time.
        live = self.client.get(reverse("paper_trading:account"), {"mode": "live"}).json()
        self.assertEqual(live["positions"], [])
        self.assertIsNone(live["resume_market_time"])

    def test_no_resume_once_flat(self):
        self.order("BUY", 4, 2, price="101.00")
        self.order("SELL", 4, 3, price="104.00")
        data = self.client.get(
            reverse("paper_trading:account"),
            {"mode": "replay", "date": self.day.isoformat(), "company": "TST"},
        ).json()
        self.assertEqual(data["positions"], [])
        self.assertIsNone(data["resume_market_time"])
        self.assertEqual(data["realized_pnl"], 12.0)

    def test_account_is_the_same_in_every_browser_but_books_are_separate(self):
        other = self.client_class()                        # a new browser / session sees the same book 1
        self.assertEqual(other.get(reverse("paper_trading:account"), {"mode": "live"}).json()["balance"], 10000.0)
        book2 = other.get(reverse("paper_trading:account"), {"mode": "live", "book": 2}).json()
        self.assertEqual(book2["balance"], 0.0)            # Paper Trading 2 has its own account
        self.assertEqual(other.get(reverse("paper_trading:account"), {"mode": "live", "book": 3}).status_code, 400)


def live_frame(minutes_ago, close=250.0):
    ts = datetime.now(svc.IST) - timedelta(minutes=minutes_ago)
    return pd.DataFrame(
        [{"timestamp": pd.Timestamp(ts), "open": close - 1, "high": close + 1, "low": close - 2, "close": close, "volume": 10.0, "source": "yfinance"}]
    )


class LiveApiTests(BaseCase):
    def setUp(self):
        super().setUp()
        self.client.post(reverse("paper_trading:wallet_add"), json.dumps({"amount": 10000}), content_type="application/json")

    def live_order(self, side="BUY", qty=2):
        return self.post(reverse("paper_trading:order"), {"company": "TST", "side": side, "quantity": qty, "mode": "live"})

    def test_rejected_when_market_closed(self):
        with mock.patch.object(svc, "get_market_status", return_value="CLOSED"), \
             mock.patch.object(svc, "fetch_live_today", return_value=live_frame(2)):
            r = self.live_order()
        self.assertEqual(r.status_code, 400)
        self.assertIn("market is closed", r.json()["error"].lower())
        self.assertFalse(PaperOrder.objects.exists())

    def test_rejected_when_data_stale(self):
        with mock.patch.object(svc, "get_market_status", return_value="OPEN"), \
             mock.patch.object(svc, "fetch_live_today", return_value=live_frame(180)):
            r = self.live_order()
        self.assertEqual(r.status_code, 400)
        self.assertIn("stale", r.json()["error"].lower())

    def test_rejected_when_no_live_data(self):
        with mock.patch.object(svc, "get_market_status", return_value="OPEN"), \
             mock.patch.object(svc, "fetch_live_today", return_value=pd.DataFrame()):
            self.assertEqual(self.live_order().status_code, 400)

    def test_server_uses_its_own_live_price_and_ignores_client_price(self):
        with mock.patch.object(svc, "get_market_status", return_value="OPEN"), \
             mock.patch.object(svc, "fetch_live_today", return_value=live_frame(2, close=250.0)):
            r = self.post(reverse("paper_trading:order"),
                          {"company": "TST", "side": "BUY", "quantity": 2, "mode": "live", "price": 1})
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()["order"]["price"], 250.0)
        self.assertEqual(r.json()["balance"], 9500.0)
        self.assertEqual(r.json()["position"]["context"], "live")

    def test_live_snapshot_flags(self):
        with mock.patch.object(svc, "get_market_status", return_value="OPEN"), \
             mock.patch.object(svc, "fetch_live_today", return_value=live_frame(2, close=250.0)):
            data = self.client.get(reverse("paper_trading:candles", args=["TST"]), {"mode": "live"}).json()
        self.assertTrue(data["tradable"])
        self.assertEqual(data["ltp"], 250.0)
        self.assertEqual(len(data["candles"]), 1)

        with mock.patch.object(svc, "get_market_status", return_value="POST_MARKET"), \
             mock.patch.object(svc, "fetch_live_today", return_value=live_frame(2, close=250.0)):
            data = self.client.get(reverse("paper_trading:candles", args=["TST"]), {"mode": "live"}).json()
        self.assertFalse(data["tradable"])

        with mock.patch.object(svc, "get_market_status", return_value="CLOSED"), \
             mock.patch.object(svc, "fetch_live_today", return_value=pd.DataFrame()):
            data = self.client.get(reverse("paper_trading:candles", args=["TST"]), {"mode": "live"}).json()
        self.assertEqual(data["candles"], [])
        self.assertFalse(data["tradable"])

    def test_live_fetch_never_touches_the_database(self):
        before = FiveMinuteData.objects.count()
        with mock.patch.object(svc, "get_market_status", return_value="OPEN"), \
             mock.patch.object(svc, "fetch_live_today", return_value=live_frame(2)):
            self.live_order()
            self.client.get(reverse("paper_trading:candles", args=["TST"]), {"mode": "live"})
        self.assertEqual(FiveMinuteData.objects.count(), before)


class PageTests(BaseCase):
    def test_page_renders_with_companies_and_csrf_cookie(self):
        r = self.client.get(reverse("paper_trading:page"))
        self.assertEqual(r.status_code, 200)
        html = r.content.decode()
        self.assertIn("TST — Test Co", html)
        self.assertIn("csrftoken", r.cookies)
        self.assertIn("/api/paper/order/", html)


class BracketServiceTests(BaseCase):
    def buy(self, qty, price, **levels):
        return svc.execute_order(
            account=self.account, company=self.company, side="BUY", quantity=qty, price=D(str(price)),
            mode=svc.LIVE, replay_date=None, market_time=None, levels=levels or None,
        )

    def sell(self, qty, price, **levels):
        return svc.execute_order(
            account=self.account, company=self.company, side="SELL", quantity=qty, price=D(str(price)),
            mode=svc.LIVE, replay_date=None, market_time=None, levels=levels or None,
        )

    def test_long_percent_levels_resolve_from_entry_price(self):
        _, pos, _ = self.buy(10, 100, sl_percent=D("2"), target_percent=D("5"))
        self.assertEqual(pos.stop_loss, D("98.00"))
        self.assertEqual(pos.target_price, D("105.00"))

    def test_short_percent_levels_are_mirrored(self):
        _, pos, _ = self.sell(10, 100, sl_percent=D("2"), target_percent=D("5"))
        self.assertEqual(pos.stop_loss, D("102.00"))
        self.assertEqual(pos.target_price, D("95.00"))

    def test_absolute_prices_are_stored_as_given(self):
        _, pos, _ = self.buy(10, 100, sl_price=D("97.5"), target_price=D("110"))
        self.assertEqual((pos.stop_loss, pos.target_price), (D("97.50"), D("110.00")))

    def test_percent_wins_over_price_when_both_given(self):
        _, pos, _ = self.buy(10, 100, sl_percent=D("1"), sl_price=D("50"))
        self.assertEqual(pos.stop_loss, D("99.00"))

    def test_wrong_side_levels_rejected_and_order_rolled_back(self):
        bad = [
            dict(sl_price=D("101")),       # long: stop above entry
            dict(target_price=D("99")),    # long: target below entry
            dict(sl_price=D("100")),       # equal to entry
            dict(sl_percent=D("100")),     # parse limit is enforced in parse_bracket; here -> price 0
        ]
        for levels in bad:
            with self.assertRaises(svc.TradeError, msg=str(levels)):
                self.buy(10, 100, **levels)
        self.assertEqual(self.balance(), D("10000"))
        self.assertFalse(PaperOrder.objects.exists())
        self.assertFalse(PaperPosition.objects.exclude(quantity=0).exists())

    def test_short_wrong_side_rejected(self):
        with self.assertRaises(svc.TradeError):
            self.sell(10, 100, sl_price=D("99"))
        with self.assertRaises(svc.TradeError):
            self.sell(10, 100, target_price=D("101"))

    def test_adding_without_levels_keeps_existing_levels(self):
        self.buy(10, 100, sl_price=D("95"), target_price=D("110"))
        _, pos, _ = self.buy(10, 104)
        self.assertEqual((pos.stop_loss, pos.target_price), (D("95.00"), D("110.00")))

    def test_adding_with_percent_anchors_to_blended_average(self):
        self.buy(10, 100)
        _, pos, _ = self.buy(10, 110, sl_percent=D("2"))   # avg 105 -> 102.9
        self.assertEqual(pos.stop_loss, D("102.90"))

    def test_partial_exit_leaves_levels_alone(self):
        self.buy(10, 100, sl_price=D("95"), target_price=D("110"))
        _, pos, _ = self.sell(4, 103)
        self.assertEqual(pos.quantity, 6)
        self.assertEqual((pos.stop_loss, pos.target_price), (D("95.00"), D("110.00")))

    def test_partial_exit_ignores_levels_passed_with_it(self):
        self.buy(10, 100, sl_price=D("95"))
        _, pos, _ = self.sell(4, 103, sl_price=D("1"))     # would be invalid for a short, but this is only a reduce
        self.assertEqual(pos.stop_loss, D("95.00"))

    def test_full_exit_clears_levels(self):
        self.buy(10, 100, sl_price=D("95"), target_price=D("110"))
        _, pos, _ = self.sell(10, 101)
        self.assertEqual(pos.quantity, 0)
        self.assertIsNone(pos.stop_loss)
        self.assertIsNone(pos.target_price)

    def test_flip_clears_old_levels_or_applies_new_ones(self):
        self.buy(10, 100, sl_price=D("95"), target_price=D("110"))
        _, pos, _ = self.sell(15, 101)                    # flips to short 5, no new levels
        self.assertEqual(pos.quantity, -5)
        self.assertIsNone(pos.stop_loss)
        self.assertIsNone(pos.target_price)

        _, pos, _ = self.buy(5, 100)                      # flat again
        _, pos, _ = self.buy(10, 100, sl_price=D("95"))
        _, pos, _ = self.sell(15, 101, sl_percent=D("2"), target_percent=D("3"))   # flips, new short levels
        self.assertEqual(pos.quantity, -5)
        self.assertEqual((pos.stop_loss, pos.target_price), (D("103.02"), D("97.97")))

    def test_check_levels_hit(self):
        hit = svc.check_levels_hit
        self.assertEqual(hit(5, D("95"), D("110"), D("95")), "stop_loss")
        self.assertEqual(hit(5, D("95"), D("110"), D("94")), "stop_loss")
        self.assertEqual(hit(5, D("95"), D("110"), D("110")), "target")
        self.assertIsNone(hit(5, D("95"), D("110"), D("100")))
        self.assertEqual(hit(-5, D("105"), D("90"), D("105")), "stop_loss")
        self.assertEqual(hit(-5, D("105"), D("90"), D("89")), "target")
        self.assertIsNone(hit(-5, D("105"), D("90"), D("100")))
        self.assertIsNone(hit(0, D("95"), D("110"), D("1")))

    def test_conservation_invariant_still_holds_with_levels(self):
        rng = random.Random(11)
        self.account.balance = D("50000")
        self.account.save()
        deposits, realized_total = D("50000"), D("0")
        for _ in range(200):
            side = rng.choice(["BUY", "SELL"])
            price = D(str(round(rng.uniform(90, 120), 2)))
            levels = {"sl_percent": D("1.5"), "target_percent": D("3")} if rng.random() < 0.5 else None
            try:
                order, _, _ = svc.execute_order(
                    account=self.account, company=self.company, side=side, quantity=rng.randint(1, 30),
                    price=price, mode=svc.LIVE, replay_date=None, market_time=None, levels=levels,
                )
                realized_total += order.realized_pnl
            except svc.TradeError:
                pass
            blocked = sum(p.cost_basis for p in PaperPosition.objects.all())
            self.assertEqual(self.balance() + blocked, deposits + realized_total)


class BracketParsingTests(TestCase):
    def test_blank_inputs_are_none(self):
        parsed = svc.parse_bracket({"stop_loss_percent": "", "stop_loss_price": None, "target_percent": "  ", "target_price": ""})
        self.assertEqual(set(parsed.values()), {None})

    def test_valid_inputs(self):
        parsed = svc.parse_bracket({"stop_loss_percent": "2.5", "target_price": "123.456"})
        self.assertEqual(parsed["sl_percent"], D("2.5"))
        self.assertEqual(parsed["target_price"], D("123.46"))

    def test_bad_inputs_rejected(self):
        for key, val in [
            ("stop_loss_percent", "0"), ("stop_loss_percent", "-1"), ("stop_loss_percent", "100"),
            ("stop_loss_percent", "abc"), ("stop_loss_percent", "NaN"),
            ("target_percent", "1000"), ("target_percent", "Infinity"),
            ("stop_loss_price", "0"), ("stop_loss_price", "-3"), ("target_price", "x"), ("target_price", "NaN"),
        ]:
            with self.assertRaises(svc.TradeError, msg=f"{key}={val}"):
                svc.parse_bracket({key: val})


class BracketApiTests(BaseCase):
    def setUp(self):
        super().setUp()
        self.client.post(reverse("paper_trading:wallet_add"), json.dumps({"amount": 10000}), content_type="application/json")

    def candle(self, i):
        return svc.load_replay_candles(self.company, self.day)[i]

    def order(self, side, qty, i, price, **extra):
        c = self.candle(i)
        payload = {"company": "TST", "side": side, "quantity": qty, "mode": "replay", "date": self.day.isoformat(),
                   "market_time": c["x"], "price": price}
        payload.update(extra)
        return self.post(reverse("paper_trading:order"), payload)

    def state(self):
        return self.client.get(reverse("paper_trading:account"),
                               {"mode": "replay", "date": self.day.isoformat(), "company": "TST"}).json()

    def bracket(self, **payload):
        base = {"company": "TST", "mode": "replay", "date": self.day.isoformat()}
        base.update(payload)
        return self.post(reverse("paper_trading:bracket"), base)

    def test_order_with_percent_levels_exposes_prices_in_account_state(self):
        r = self.order("BUY", 10, 0, "100.00", stop_loss_percent="2", target_percent="4")
        self.assertEqual(r.status_code, 200, r.content)
        pos = r.json()["position"]
        self.assertEqual((pos["stop_loss"], pos["target_price"]), (98.0, 104.0))
        pos = self.state()["positions"][0]
        self.assertEqual((pos["stop_loss"], pos["target_price"]), (98.0, 104.0))

    def test_order_with_price_levels(self):
        r = self.order("BUY", 10, 0, "100.00", stop_loss_price="97", target_price="106")
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual((r.json()["position"]["stop_loss"], r.json()["position"]["target_price"]), (97.0, 106.0))

    def test_invalid_levels_reject_the_whole_order(self):
        r = self.order("BUY", 10, 0, "100.00", stop_loss_price="103")
        self.assertEqual(r.status_code, 400)
        self.assertIn("below your entry price", r.json()["error"])
        self.assertEqual(self.state()["positions"], [])
        self.assertEqual(self.state()["balance"], 10000.0)

    def test_malformed_level_input_is_a_clean_400(self):
        r = self.order("BUY", 1, 0, "100.00", stop_loss_percent="abc")
        self.assertEqual(r.status_code, 400)

    def test_auto_exit_reason_is_recorded_and_clears_levels(self):
        self.order("BUY", 10, 0, "100.00", stop_loss_price="98", target_price="106")
        r = self.order("SELL", 10, 5, "98.00", reason="stop_loss")   # candle 5 spans 98..104
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()["order"]["reason"], "stop_loss")
        self.assertEqual(r.json()["order"]["realized_pnl"], -20.0)
        self.assertEqual(self.state()["positions"], [])
        self.assertEqual(r.json()["balance"], 9980.0)

    def test_unknown_reason_is_ignored_not_trusted(self):
        r = self.order("BUY", 1, 0, "100.00", reason="<script>")
        self.assertEqual(r.json()["order"]["reason"], "")

    def test_update_levels_on_open_position(self):
        self.order("BUY", 10, 0, "100.00")
        r = self.bracket(stop_loss_percent="1", target_price="108")
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual((r.json()["position"]["stop_loss"], r.json()["position"]["target_price"]), (99.0, 108.0))

    def test_update_does_not_touch_the_wallet_or_orders(self):
        self.order("BUY", 10, 0, "100.00")
        before = self.state()
        self.bracket(stop_loss_price="95")
        after = self.state()
        self.assertEqual(before["balance"], after["balance"])
        self.assertEqual(len(before["orders"]), len(after["orders"]))

    def test_update_only_changes_the_fields_given(self):
        self.order("BUY", 10, 0, "100.00", stop_loss_price="95", target_price="110")
        self.bracket(stop_loss_price="96")
        pos = self.state()["positions"][0]
        self.assertEqual((pos["stop_loss"], pos["target_price"]), (96.0, 110.0))

    def test_clear_flags(self):
        self.order("BUY", 10, 0, "100.00", stop_loss_price="95", target_price="110")
        self.assertEqual(self.bracket(clear_stop_loss=True).status_code, 200)
        pos = self.state()["positions"][0]
        self.assertEqual((pos["stop_loss"], pos["target_price"]), (None, 110.0))
        self.assertEqual(self.bracket(clear_target=True).status_code, 200)
        pos = self.state()["positions"][0]
        self.assertEqual((pos["stop_loss"], pos["target_price"]), (None, None))

    def test_update_needs_an_open_position(self):
        r = self.bracket(stop_loss_percent="1")
        self.assertEqual(r.status_code, 400)
        self.assertIn("no open intraday position", r.json()["error"].lower())

    def test_update_with_nothing_to_change_is_rejected(self):
        self.order("BUY", 10, 0, "100.00")
        self.assertEqual(self.bracket().status_code, 400)

    def test_update_validates_against_average_entry_price(self):
        self.order("BUY", 10, 0, "100.00")
        self.assertEqual(self.bracket(stop_loss_price="101").status_code, 400)
        self.assertEqual(self.bracket(target_price="99").status_code, 400)

    def test_short_position_levels(self):
        self.order("SELL", 10, 0, "100.00", stop_loss_percent="3", target_percent="6")
        pos = self.state()["positions"][0]
        self.assertEqual((pos["stop_loss"], pos["target_price"]), (103.0, 94.0))
        self.assertEqual(self.bracket(stop_loss_price="99").status_code, 400)

    def test_levels_are_per_context(self):
        self.order("BUY", 10, 0, "100.00", stop_loss_price="95")
        live = self.client.get(reverse("paper_trading:account"), {"mode": "live"}).json()
        self.assertEqual(live["positions"], [])


class PreviousDayTests(BaseCase):
    def test_falls_back_to_aggregating_five_minute_candles(self):
        # BaseCase seeded 6 candles on self.day: closes [100,102,101,105,103,99], +-1 ranges
        prev = svc.previous_day_ohlc(self.company, self.day + timedelta(days=1))
        self.assertEqual(prev["date"], self.day.isoformat())
        self.assertEqual(prev["open"], 100.0)
        self.assertEqual(prev["close"], 99.0)
        self.assertEqual(prev["high"], 106.0)    # 105 candle high = max(open 101, close 105) + 1
        self.assertEqual(prev["low"], 98.0)      # 99 candle low  = min(open 103, close 99) - 1

    def test_prefers_daily_data_when_available(self):
        DailyData.objects.create(date=self.day, company=self.company,
                                 open=D("1"), high=D("9"), low=D("0.5"), close=D("5"))
        prev = svc.previous_day_ohlc(self.company, self.day + timedelta(days=1))
        self.assertEqual((prev["open"], prev["high"], prev["low"], prev["close"]), (1.0, 9.0, 0.5, 5.0))

    def test_incomplete_daily_row_falls_back_to_candles(self):
        DailyData.objects.create(date=self.day, company=self.company, open=D("1"), high=None, low=D("0.5"), close=D("5"))
        prev = svc.previous_day_ohlc(self.company, self.day + timedelta(days=1))
        self.assertEqual(prev["open"], 100.0)

    def test_none_when_no_earlier_data(self):
        self.assertIsNone(svc.previous_day_ohlc(self.company, self.day))
        self.assertIsNone(svc.previous_day_ohlc(self.other, self.day + timedelta(days=5)))

    def test_skips_non_trading_days(self):
        prev = svc.previous_day_ohlc(self.company, self.day + timedelta(days=4))   # gap of empty days
        self.assertEqual(prev["date"], self.day.isoformat())

    def test_replay_snapshot_previous_day_comes_from_the_database(self):
        later = self.day + timedelta(days=1)
        make_candles(self.company, later, [110, 111])
        data = svc.replay_snapshot(self.company, later)
        self.assertEqual(data["previous_day"]["date"], self.day.isoformat())
        self.assertEqual(data["previous_day"]["close"], 99.0)

    def test_api_exposes_previous_day(self):
        later = self.day + timedelta(days=1)
        make_candles(self.company, later, [110, 111])
        r = self.client.get(reverse("paper_trading:candles", args=["TST"]), {"mode": "replay", "date": later.isoformat()})
        self.assertEqual(r.json()["previous_day"]["close"], 99.0)
        r = self.client.get(reverse("paper_trading:candles", args=["TST"]), {"mode": "replay", "date": self.day.isoformat()})
        self.assertIsNone(r.json()["previous_day"])


# ===========================================================================
# helpers for the product / live-Yahoo / quotes / square-off tests
# ===========================================================================
def daily_frame(rows):
    """Yahoo-style daily DataFrame. rows: [(date, open, high, low, close), ...]"""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]).tz_localize("Asia/Kolkata") for r in rows])
    return pd.DataFrame(
        {"Open": [r[1] for r in rows], "High": [r[2] for r in rows], "Low": [r[3] for r in rows],
         "Close": [r[4] for r in rows], "Volume": [1] * len(rows)},
        index=idx,
    )


def at(day, hh, mm):
    return datetime.combine(day, time(hh, mm), tzinfo=svc.IST)


class ProductTests(BaseCase):
    def go(self, side, qty, price, product, company=None, mode=svc.LIVE, replay_date=None, market_time=None, **kw):
        return svc.execute_order(
            account=self.account, company=company or self.company, side=side, quantity=qty, price=D(str(price)),
            mode=mode, replay_date=replay_date, market_time=market_time, product=product, **kw,
        )

    def test_default_product_is_intraday(self):
        order, pos, _ = svc.execute_order(account=self.account, company=self.company, side="BUY", quantity=1,
                                          price=D("100"), mode=svc.LIVE, replay_date=None, market_time=None)
        self.assertEqual((order.product, pos.product), ("INTRADAY", "INTRADAY"))

    def test_same_stock_can_be_held_in_both_products_separately(self):
        self.go("BUY", 5, 100, "INTRADAY")
        self.go("BUY", 7, 100, "DELIVERY")
        rows = {p.product: p.quantity for p in PaperPosition.objects.exclude(quantity=0)}
        self.assertEqual(rows, {"INTRADAY": 5, "DELIVERY": 7})

    def test_a_delivery_buy_does_not_cover_an_intraday_short(self):
        self.go("SELL", 5, 100, "INTRADAY")                   # short 5 intraday
        self.go("BUY", 5, 100, "DELIVERY")                    # a *separate* long 5
        rows = {p.product: p.quantity for p in PaperPosition.objects.exclude(quantity=0)}
        self.assertEqual(rows, {"INTRADAY": -5, "DELIVERY": 5})

    def test_delivery_sell_without_holding_is_rejected_and_leaves_nothing_behind(self):
        with self.assertRaises(svc.TradeError) as cm:
            self.go("SELL", 1, 100, "DELIVERY")
        self.assertIn("nothing to sell", str(cm.exception))
        self.assertEqual(PaperPosition.objects.count(), 0)        # the get_or_create row was rolled back
        self.assertFalse(PaperOrder.objects.exists())
        self.assertEqual(self.balance(), D("10000"))

    def test_delivery_cannot_sell_more_than_held(self):
        self.go("BUY", 5, 100, "DELIVERY")
        with self.assertRaises(svc.TradeError) as cm:
            self.go("SELL", 6, 100, "DELIVERY")
        self.assertIn("hold only 5", str(cm.exception))
        self.assertEqual(PaperPosition.objects.get().quantity, 5)

    def test_delivery_sells_down_to_flat_with_correct_pnl(self):
        self.go("BUY", 10, 100, "DELIVERY")
        o1, p, _ = self.go("SELL", 4, 110, "DELIVERY")
        self.assertEqual((o1.realized_pnl, p.quantity), (D("40"), 6))
        o2, p, _ = self.go("SELL", 6, 95, "DELIVERY")
        self.assertEqual((o2.realized_pnl, p.quantity), (D("-30"), 0))
        self.assertEqual(self.balance(), D("10010"))

    def test_intraday_shorting_is_unchanged(self):
        o, p, _ = self.go("SELL", 5, 100, "INTRADAY")
        self.assertEqual(p.quantity, -5)

    def test_levels_apply_per_product(self):
        self.go("BUY", 5, 100, "INTRADAY", levels={"sl_price": D("95")})
        _, p, _ = self.go("BUY", 5, 100, "DELIVERY", levels={"sl_price": D("90"), "target_price": D("130")})
        by = {x.product: (x.stop_loss, x.target_price) for x in PaperPosition.objects.exclude(quantity=0)}
        self.assertEqual(by["INTRADAY"], (D("95.00"), None))
        self.assertEqual(by["DELIVERY"], (D("90.00"), D("130.00")))

    def test_last_entry_at_tracks_entries_not_exits(self):
        t1, t2 = at(self.day, 10, 0), at(self.day, 11, 0)
        _, p, _ = self.go("BUY", 10, 100, "INTRADAY", market_time=t1)
        self.assertEqual(svc.to_ist(p.last_entry_at), t1)
        _, p, _ = self.go("BUY", 5, 100, "INTRADAY", market_time=t2)
        self.assertEqual(svc.to_ist(p.last_entry_at), t2)
        _, p, _ = self.go("SELL", 3, 101, "INTRADAY", market_time=at(self.day, 12, 0))
        self.assertEqual(svc.to_ist(p.last_entry_at), t2)           # a partial exit is not an entry
        _, p, _ = self.go("SELL", 12, 101, "INTRADAY", market_time=at(self.day, 12, 5))
        self.assertIsNone(p.last_entry_at)                           # flat

    def test_conservation_invariant_with_mixed_products(self):
        rng = random.Random(21)
        self.account.balance = D("80000")
        self.account.save()
        realized_total = D("0")
        for _ in range(250):
            product = rng.choice(["INTRADAY", "DELIVERY"])
            side = rng.choice(["BUY", "SELL"])
            price = D(str(round(rng.uniform(90, 120), 2)))
            company = rng.choice([self.company, self.other])
            try:
                order, _, _ = self.go(side, rng.randint(1, 25), price, product, company=company)
                realized_total += order.realized_pnl
            except svc.TradeError:
                pass
            blocked = sum(p.cost_basis for p in PaperPosition.objects.all())
            self.assertEqual(self.balance() + blocked, D("80000") + realized_total)
            # delivery is never short
            self.assertFalse(PaperPosition.objects.filter(product="DELIVERY", quantity__lt=0).exists())


class ReplayDeliveryTests(BaseCase):
    """Old-mode Delivery holdings carry across replay dates; intraday stay on their day."""

    def setUp(self):
        super().setUp()
        self.d1 = self.day
        self.d2 = self.day + timedelta(days=1)
        make_candles(self.company, self.d2, [100, 101, 102])

    def trade(self, side, qty, price, product, day, hh=10, mm=0, company=None):
        return svc.execute_order(
            account=self.account, company=company or self.company, side=side, quantity=qty, price=D(str(price)),
            mode=svc.REPLAY, replay_date=day, market_time=at(day, hh, mm), product=product,
        )

    def snap(self, day):
        return svc.account_snapshot(self.account, svc.REPLAY, day, self.company)

    def test_storage_contexts(self):
        self.trade("BUY", 5, 100, "INTRADAY", self.d1)
        self.trade("BUY", 5, 100, "DELIVERY", self.d1)
        ctx = {p.product: p.context for p in PaperPosition.objects.all()}
        self.assertEqual(ctx, {"INTRADAY": self.d1.isoformat(), "DELIVERY": "replay"})
        self.assertEqual(set(PaperOrder.objects.values_list("context", flat=True)), {self.d1.isoformat()})

    def test_delivery_holding_is_visible_on_later_dates_but_intraday_is_not(self):
        self.trade("BUY", 5, 100, "INTRADAY", self.d1)
        self.trade("BUY", 7, 100, "DELIVERY", self.d1)
        d1_view = {p["product"]: p["quantity"] for p in self.snap(self.d1)["positions"]}
        d2_view = {p["product"]: p["quantity"] for p in self.snap(self.d2)["positions"]}
        before = {p["product"]: p["quantity"] for p in self.snap(self.d1 - timedelta(days=2))["positions"]}
        self.assertEqual(d1_view, {"INTRADAY": 5, "DELIVERY": 7})
        self.assertEqual(d2_view, {"DELIVERY": 7})
        self.assertEqual(before, {})

    def test_sell_the_holding_on_a_later_date(self):
        self.trade("BUY", 10, 100, "DELIVERY", self.d1, 10, 0)
        order, pos, _ = self.trade("SELL", 10, 102, "DELIVERY", self.d2, 9, 15)
        self.assertEqual(order.realized_pnl, D("20"))
        self.assertEqual(pos.quantity, 0)
        self.assertEqual(order.context, self.d2.isoformat())      # shown in the day it was sold
        self.assertEqual(self.snap(self.d2)["positions"], [])

    def test_cannot_trade_a_holding_before_its_own_last_trade(self):
        self.trade("BUY", 10, 100, "DELIVERY", self.d1, 11, 0)
        with self.assertRaises(svc.TradeError) as cm:
            self.trade("SELL", 5, 100, "DELIVERY", self.d1, 10, 0)           # same day, earlier time
        self.assertIn("before your last Delivery trade", str(cm.exception))
        with self.assertRaises(svc.TradeError):
            self.trade("SELL", 5, 100, "DELIVERY", self.d1 - timedelta(days=1), 12, 0)   # earlier date
        with self.assertRaises(svc.TradeError):
            self.trade("BUY", 5, 100, "DELIVERY", self.d1, 10, 30)          # adding in the past too
        self.assertEqual(PaperPosition.objects.get().quantity, 10)
        self.trade("SELL", 5, 100, "DELIVERY", self.d1, 11, 0)               # same moment is fine

    def test_resume_time_uses_latest_order_across_all_stocks(self):
        make_candles(self.other, self.d1, [50, 51, 52, 53])
        self.trade("BUY", 5, 100, "INTRADAY", self.d1, 10, 0, company=self.company)
        self.trade("BUY", 5, 50, "INTRADAY", self.d1, 10, 40, company=self.other)
        snap = svc.account_snapshot(self.account, svc.REPLAY, self.d1, self.other)
        self.assertEqual(snap["resume_market_time"], at(self.d1, 10, 40).isoformat())
        # ... and it's the same clock whichever stock you open
        snap = svc.account_snapshot(self.account, svc.REPLAY, self.d1, self.company)
        self.assertEqual(snap["resume_market_time"], at(self.d1, 10, 40).isoformat())

    def test_all_stocks_appear_in_the_snapshot(self):
        make_candles(self.other, self.d1, [50, 51, 52])
        self.trade("BUY", 5, 100, "INTRADAY", self.d1, 10, 0, company=self.company)
        self.trade("SELL", 3, 50, "INTRADAY", self.d1, 10, 5, company=self.other)
        snap = self.snap(self.d1)
        self.assertEqual({(p["company"], p["quantity"]) for p in snap["positions"]}, {("TST", 5), ("OTH", -3)})

    def test_a_carried_holding_alone_does_not_force_a_resume(self):
        self.trade("BUY", 5, 100, "DELIVERY", self.d1, 10, 0)
        self.assertIsNone(self.snap(self.d2)["resume_market_time"])         # fresh day from 9:15
        self.assertEqual(self.snap(self.d1)["resume_market_time"], at(self.d1, 10, 0).isoformat())

    def test_bracket_update_finds_delivery_holding_on_a_later_date(self):
        self.trade("BUY", 5, 100, "DELIVERY", self.d1)
        pos = svc.update_bracket(account=self.account, company=self.company, mode=svc.REPLAY, replay_date=self.d2,
                                 levels={"sl_price": D("95")}, product="DELIVERY")
        self.assertEqual(pos.stop_loss, D("95.00"))
        with self.assertRaises(svc.TradeError):
            svc.update_bracket(account=self.account, company=self.company, mode=svc.REPLAY, replay_date=self.d2,
                               levels={"sl_price": D("95")}, product="INTRADAY")


class ProductApiTests(BaseCase):
    def setUp(self):
        super().setUp()
        self.client.post(reverse("paper_trading:wallet_add"), json.dumps({"amount": 10000}), content_type="application/json")

    def order(self, side, qty, i, price, **extra):
        c = svc.load_replay_candles(self.company, self.day)[i]
        payload = {"company": "TST", "side": side, "quantity": qty, "mode": "replay", "date": self.day.isoformat(),
                   "market_time": c["x"], "price": price}
        payload.update(extra)
        return self.post(reverse("paper_trading:order"), payload)

    def state(self):
        return self.client.get(reverse("paper_trading:account"),
                               {"mode": "replay", "date": self.day.isoformat(), "company": "TST"}).json()

    def test_product_is_recorded_and_returned(self):
        r = self.order("BUY", 5, 0, "100.00", product="DELIVERY")
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()["order"]["product"], "DELIVERY")
        self.assertEqual(r.json()["position"]["product"], "DELIVERY")
        self.assertEqual(self.state()["positions"][0]["product"], "DELIVERY")

    def test_missing_product_defaults_to_intraday(self):
        r = self.order("BUY", 5, 0, "100.00")
        self.assertEqual(r.json()["order"]["product"], "INTRADAY")

    def test_bad_product_is_a_clean_400(self):
        for bad in ("MARGIN", "x", 5):
            self.assertEqual(self.order("BUY", 1, 0, "100.00", product=bad).status_code, 400, bad)

    def test_delivery_short_sell_is_rejected_with_guidance(self):
        r = self.order("SELL", 5, 0, "100.00", product="DELIVERY")
        self.assertEqual(r.status_code, 400)
        self.assertIn("Switch to Intraday", r.json()["error"])
        self.assertEqual(self.state()["positions"], [])

    def test_lowercase_product_accepted(self):
        self.assertEqual(self.order("BUY", 1, 0, "100.00", product="delivery").json()["order"]["product"], "DELIVERY")

    def test_bracket_endpoint_is_product_aware(self):
        self.order("BUY", 5, 0, "100.00", product="DELIVERY")
        base = {"company": "TST", "mode": "replay", "date": self.day.isoformat(), "stop_loss_price": "95"}
        self.assertEqual(self.post(reverse("paper_trading:bracket"), {**base, "product": "INTRADAY"}).status_code, 400)
        r = self.post(reverse("paper_trading:bracket"), {**base, "product": "DELIVERY"})
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()["position"]["stop_loss"], 95.0)


class LiveYahooPreviousDayTests(BaseCase):
    """Live mode takes the previous day from Yahoo - never from the database."""

    def test_previous_day_comes_from_yahoo_not_the_database(self):
        yday = svc.today_ist() - timedelta(days=1)
        DailyData.objects.create(date=yday, company=self.company, open=D("1"), high=D("2"), low=D("0.5"), close=D("1.5"))
        make_candles(self.company, yday, [10, 11])                                  # DB candles for yesterday too
        frame = daily_frame([(yday - timedelta(days=1), 200, 210, 195, 205), (yday, 300, 320, 290, 310)])
        with mock.patch.object(svc, "_yf_history", return_value=frame):
            prev = svc.live_previous_day(self.company)
        self.assertEqual(prev, {"date": yday.isoformat(), "open": 300.0, "high": 320.0, "low": 290.0, "close": 310.0})

    def test_todays_partial_row_is_never_used(self):
        today, yday = svc.today_ist(), svc.today_ist() - timedelta(days=1)
        frame = daily_frame([(yday, 300, 320, 290, 310), (today, 311, 312, 309, 311.5)])
        with mock.patch.object(svc, "_yf_history", return_value=frame):
            self.assertEqual(svc.live_previous_day(self.company)["date"], yday.isoformat())

    def test_yahoo_failure_gives_none_and_does_not_fall_back_to_the_database(self):
        yday = svc.today_ist() - timedelta(days=1)
        DailyData.objects.create(date=yday, company=self.company, open=D("1"), high=D("2"), low=D("0.5"), close=D("1.5"))
        make_candles(self.company, yday, [10, 11])
        with mock.patch.object(svc, "_yf_history", side_effect=RuntimeError("rate limited")):
            self.assertIsNone(svc.live_previous_day(self.company))
        cache.clear()
        with mock.patch.object(svc, "_yf_history", return_value=pd.DataFrame()):
            self.assertIsNone(svc.live_previous_day(self.company))

    def test_nan_rows_are_skipped(self):
        yday = svc.today_ist() - timedelta(days=1)
        frame = daily_frame([(yday - timedelta(days=1), 1, 2, 0.5, float("nan")), (yday, 3, 4, 2, 3.5)])
        with mock.patch.object(svc, "_yf_history", return_value=frame):
            self.assertEqual(svc.live_previous_day(self.company)["close"], 3.5)

    def test_results_are_cached(self):
        yday = svc.today_ist() - timedelta(days=1)
        with mock.patch.object(svc, "_yf_history", return_value=daily_frame([(yday, 3, 4, 2, 3.5)])) as m:
            svc.live_previous_day(self.company)
            svc.live_previous_day(self.company)
            svc.daily_close_on(self.company, yday)
        # one daily and one 5-minute request, each cached - nothing repeated
        intervals = sorted(c.kwargs.get("interval") for c in m.call_args_list)
        self.assertEqual(intervals, ["1d", "5m"])

    def test_ticker_mapping_is_used(self):
        yday = svc.today_ist() - timedelta(days=1)
        with mock.patch.object(svc, "_yf_history", return_value=daily_frame([(yday, 3, 4, 2, 3.5)])) as m:
            svc.live_previous_day(self.company)
        self.assertEqual(m.call_args.args[0], "TST.NS")

    def test_live_snapshot_uses_yahoo_while_replay_snapshot_uses_the_database(self):
        yday = svc.today_ist() - timedelta(days=1)
        make_candles(self.company, yday, [10, 11])           # DB says yesterday closed at 11
        frame = daily_frame([(yday, 300, 320, 290, 310)])
        with mock.patch.object(svc, "get_market_status", return_value="OPEN"), \
             mock.patch.object(svc, "fetch_live_today", return_value=live_frame(2)), \
             mock.patch.object(svc, "_yf_history", return_value=frame):
            live = svc.live_snapshot(self.company)
        self.assertEqual(live["previous_day"]["close"], 310.0)
        replay = svc.replay_snapshot(self.company, yday)
        self.assertEqual(replay["previous_day"]["date"], self.day.isoformat())   # DB, an earlier stored day

    def test_live_candles_endpoint_exposes_yahoo_previous_day(self):
        yday = svc.today_ist() - timedelta(days=1)
        with mock.patch.object(svc, "get_market_status", return_value="OPEN"), \
             mock.patch.object(svc, "fetch_live_today", return_value=live_frame(2)), \
             mock.patch.object(svc, "_yf_history", return_value=daily_frame([(yday, 300, 320, 290, 310)])):
            r = self.client.get(reverse("paper_trading:candles", args=["TST"]), {"mode": "live"})
        self.assertEqual(r.json()["previous_day"]["high"], 320.0)

    def test_daily_close_on(self):
        yday = svc.today_ist() - timedelta(days=1)
        with mock.patch.object(svc, "_yf_history", return_value=daily_frame([(yday, 3, 4, 2, 3.5)])):
            self.assertEqual(svc.daily_close_on(self.company, yday), 3.5)
            self.assertIsNone(svc.daily_close_on(self.company, yday - timedelta(days=9)))


class QuotesApiTests(BaseCase):
    def frames(self):
        return {"TST": live_frame(2, close=250.0), "OTH": live_frame(2, close=77.5)}

    def get(self, companies):
        frames = self.frames()
        with mock.patch.object(svc, "get_market_status", return_value="OPEN"), \
             mock.patch.object(svc, "fetch_live_today", side_effect=lambda c, use_cache=True: frames[c.code]):
            return self.client.get(reverse("paper_trading:quotes"), {"companies": companies})

    def test_returns_a_quote_per_company(self):
        r = self.get("TST,OTH").json()
        self.assertEqual(r["quotes"]["TST"]["ltp"], 250.0)
        self.assertEqual(r["quotes"]["OTH"]["ltp"], 77.5)
        self.assertTrue(r["quotes"]["OTH"]["tradable"])
        self.assertEqual(r["quotes"]["OTH"]["market_status"], "OPEN")

    def test_unknown_codes_are_reported_not_fatal(self):
        r = self.get("TST,NOPE").json()
        self.assertEqual(list(r["quotes"]), ["TST"])
        self.assertEqual(r["unknown"], ["NOPE"])

    def test_duplicates_blank_and_empty(self):
        self.assertEqual(list(self.get("TST,,TST , ").json()["quotes"]), ["TST"])
        self.assertEqual(self.get("").json(), {"quotes": {}, "unknown": []})

    def test_closed_market_marks_quotes_untradable(self):
        frames = self.frames()
        with mock.patch.object(svc, "get_market_status", return_value="CLOSED"), \
             mock.patch.object(svc, "fetch_live_today", side_effect=lambda c, use_cache=True: frames[c.code]):
            q = self.client.get(reverse("paper_trading:quotes"), {"companies": "TST"}).json()["quotes"]["TST"]
        self.assertFalse(q["tradable"])
        self.assertEqual(q["ltp"], 250.0)

    def test_company_without_data_has_null_price(self):
        with mock.patch.object(svc, "get_market_status", return_value="OPEN"), \
             mock.patch.object(svc, "fetch_live_today", return_value=pd.DataFrame()):
            q = self.client.get(reverse("paper_trading:quotes"), {"companies": "TST"}).json()["quotes"]["TST"]
        self.assertIsNone(q["ltp"])
        self.assertFalse(q["tradable"])


class SquareoffTests(BaseCase):
    def open_live(self, qty, price, when, product="INTRADAY", side="BUY", company=None):
        return svc.execute_order(account=self.account, company=company or self.company, side=side, quantity=qty,
                                 price=D(str(price)), mode=svc.LIVE, replay_date=None, market_time=when, product=product)

    def test_position_left_from_an_earlier_day_is_settled_at_that_days_close(self):
        yday = svc.today_ist() - timedelta(days=1)
        self.open_live(10, 100, at(yday, 10, 0))
        with mock.patch.object(svc, "_yf_history", return_value=daily_frame([(yday, 99, 105, 98, 103)])):
            res = svc.squareoff_live_intraday(self.account, now=at(svc.today_ist(), 9, 30))
        self.assertEqual(len(res["closed"]), 1)
        order = res["closed"][0]
        self.assertEqual((order.price, order.reason, order.side), (D("103.0000"), "session_end", "SELL"))
        self.assertEqual(order.realized_pnl, D("30"))
        self.assertEqual(self.balance(), D("10030"))
        self.assertFalse(PaperPosition.objects.exclude(quantity=0).exists())

    def test_short_left_overnight_is_covered_at_that_days_close(self):
        yday = svc.today_ist() - timedelta(days=1)
        self.open_live(10, 100, at(yday, 10, 0), side="SELL")
        with mock.patch.object(svc, "_yf_history", return_value=daily_frame([(yday, 99, 105, 98, 96)])):
            res = svc.squareoff_live_intraday(self.account, now=at(svc.today_ist(), 9, 30))
        self.assertEqual((res["closed"][0].side, res["closed"][0].realized_pnl), ("BUY", D("40")))

    def test_todays_position_is_closed_after_the_cutoff_even_if_data_is_no_longer_fresh(self):
        today = svc.today_ist()
        self.open_live(10, 100, at(today, 10, 0))
        stale = live_frame(60, close=108.0)                       # an hour old: "stale" for trading, fine for square-off
        with mock.patch.object(svc, "fetch_live_today", return_value=stale):
            res = svc.squareoff_live_intraday(self.account, now=at(today, 15, 21))
        self.assertEqual(res["closed"][0].price, D("108.0000"))
        self.assertEqual(res["closed"][0].realized_pnl, D("80"))

    def test_nothing_happens_before_the_cutoff(self):
        today = svc.today_ist()
        self.open_live(10, 100, at(today, 10, 0))
        with mock.patch.object(svc, "fetch_live_today", return_value=live_frame(2, close=108.0)):
            res = svc.squareoff_live_intraday(self.account, now=at(today, 14, 0))
        self.assertEqual((res["closed"], res["skipped"]), ([], []))
        self.assertEqual(PaperPosition.objects.get().quantity, 10)

    def test_delivery_is_never_squared_off(self):
        yday, today = svc.today_ist() - timedelta(days=1), svc.today_ist()
        self.open_live(10, 100, at(yday, 10, 0), product="DELIVERY")
        with mock.patch.object(svc, "_yf_history", return_value=daily_frame([(yday, 99, 105, 98, 103)])), \
             mock.patch.object(svc, "fetch_live_today", return_value=live_frame(2, close=108.0)):
            res = svc.squareoff_live_intraday(self.account, now=at(today, 15, 25))
        self.assertEqual(res["closed"], [])
        self.assertEqual(PaperPosition.objects.get().quantity, 10)

    def test_missing_price_is_skipped_and_the_position_stays_open(self):
        yday, today = svc.today_ist() - timedelta(days=1), svc.today_ist()
        self.open_live(10, 100, at(yday, 10, 0))
        with mock.patch.object(svc, "_yf_history", return_value=pd.DataFrame()):
            res = svc.squareoff_live_intraday(self.account, now=at(today, 9, 30))
        self.assertEqual(res["closed"], [])
        self.assertIn("TST", res["skipped"][0])
        self.assertEqual(PaperPosition.objects.get().quantity, 10)

        self.open_live(5, 50, at(today, 10, 0), company=self.other)
        with mock.patch.object(svc, "fetch_live_today", return_value=pd.DataFrame()):
            res = svc.squareoff_live_intraday(self.account, now=at(today, 15, 30))
        self.assertTrue(any("OTH" in m for m in res["skipped"]))

    def test_replay_positions_are_not_touched(self):
        svc.execute_order(account=self.account, company=self.company, side="BUY", quantity=5, price=D("100"),
                          mode=svc.REPLAY, replay_date=self.day, market_time=at(self.day, 10, 0))
        res = svc.squareoff_live_intraday(self.account, now=at(svc.today_ist(), 15, 30))
        self.assertEqual(res["closed"], [])
        self.assertEqual(PaperPosition.objects.get().quantity, 5)

    def test_endpoint(self):
        self.client.post(reverse("paper_trading:wallet_add"), json.dumps({"amount": 10000}), content_type="application/json")
        r = self.post(reverse("paper_trading:squareoff"), {})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["closed"], [])
        self.assertEqual(self.client.get(reverse("paper_trading:squareoff")).status_code, 405)


class DirectUrlTests(BaseCase):
    def initial(self, url):
        r = self.client.get(url)
        return r, (r.context["initial"] if r.context else None)

    def test_live_link(self):
        r, ini = self.initial("/paper-trading/TST/live/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(ini, {"company": "TST", "mode": "live", "date": None, "error": "", "autostart": True})
        self.assertIn("csrftoken", r.cookies)

    def test_old_link_with_date_in_the_path(self):
        r, ini = self.initial(f"/paper-trading/TST/old/{self.day.isoformat()}/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual((ini["mode"], ini["date"], ini["autostart"]), ("replay", self.day.isoformat(), True))

    def test_old_link_with_date_as_query_string(self):
        r, ini = self.initial(f"/paper-trading/TST/old/?date={self.day.isoformat()}")
        self.assertEqual((r.status_code, ini["date"], ini["autostart"]), (200, self.day.isoformat(), True))

    def test_old_without_a_date_shows_the_reason_and_does_not_autostart(self):
        r, ini = self.initial("/paper-trading/TST/old/")
        self.assertEqual(r.status_code, 400)
        self.assertFalse(ini["autostart"])
        self.assertIn("valid date", ini["error"])
        self.assertIn("TST — Test Co", r.content.decode())           # the page itself still renders

    def test_bad_dates_are_rejected_politely(self):
        for bad in ("not-a-date", "2026-13-45", svc.today_ist().isoformat(),
                    (svc.today_ist() + timedelta(days=3)).isoformat()):
            r, ini = self.initial(f"/paper-trading/TST/old/{bad}/")
            self.assertEqual(r.status_code, 400, bad)
            self.assertFalse(ini["autostart"], bad)
            self.assertEqual(ini["company"], "TST")

    def test_unknown_company_and_unknown_mode_are_404(self):
        self.assertEqual(self.client.get("/paper-trading/NOPE/live/").status_code, 404)
        self.assertEqual(self.client.get(f"/paper-trading/NOPE/old/{self.day.isoformat()}/").status_code, 404)
        self.assertEqual(self.client.get("/paper-trading/TST/sideways/").status_code, 404)
        self.assertEqual(self.client.get("/paper-trading/TST/").status_code, 404)

    def test_plain_page_has_nothing_preselected(self):
        r, ini = self.initial("/paper-trading/")
        self.assertEqual(ini, {"company": "", "mode": "live", "date": None, "error": "", "autostart": False})

    def test_links_are_get_only(self):
        for url in ("/paper-trading/TST/live/", f"/paper-trading/TST/old/{self.day.isoformat()}/"):
            self.assertEqual(self.client.post(url).status_code, 405)
            self.assertEqual(self.client.head(url).status_code, 200)

    def test_url_names_reverse_to_the_documented_paths(self):
        self.assertEqual(reverse("paper_trading:page_live", args=["TST"]), "/paper-trading/TST/live/")
        self.assertEqual(reverse("paper_trading:page_old", args=["TST", "2026-10-05"]), "/paper-trading/TST/old/2026-10-05/")

    def test_initial_state_is_embedded_safely_in_the_html(self):
        r = self.client.get(f"/paper-trading/TST/old/{self.day.isoformat()}/")
        self.assertIn('id="initial-state"', r.content.decode())


# --------------------------------------------------------------------------
# strategy signals
# --------------------------------------------------------------------------
from . import signals as sg  # noqa: E402


def bar(day, hhmm, o, h, l, c, v=1000):
    hh, mm = map(int, hhmm.split(":"))
    return sg.Bar(datetime.combine(day, time(hh, mm), tzinfo=svc.IST), o, h, l, c, v)


def flat_day(day, price=100.0, n=75):
    start = datetime.combine(day, time(9, 15), tzinfo=svc.IST)
    return [sg.Bar(start + timedelta(minutes=5 * i), price, price + 0.2, price - 0.2, price, 1000) for i in range(n)]


class SignalEngineTests(TestCase):
    def setUp(self):
        self.d0 = datetime(2026, 1, 5).date()
        self.d1 = datetime(2026, 1, 6).date()

    def orb_day(self):
        """Opening range 99.5-100.5, then a rising close above it at 09:30 and a run up to 104."""
        day = [bar(self.d1, "09:15", 100, 100.5, 99.5, 100.2),
               bar(self.d1, "09:20", 100.2, 100.4, 99.8, 100.0),
               bar(self.d1, "09:25", 100.0, 100.3, 99.9, 100.1),
               bar(self.d1, "09:30", 100.1, 101.2, 100.0, 101.0),     # closes above 100.5
               bar(self.d1, "09:35", 101.0, 101.4, 100.9, 101.3)]     # entry at this open
        price = 101.3
        start = datetime.combine(self.d1, time(9, 40), tzinfo=svc.IST)
        for i in range(60):
            price += 0.05
            day.append(sg.Bar(start + timedelta(minutes=5 * i), price - 0.05, price + 0.05, price - 0.08, price, 1000))
        return day

    def test_orb_long_signal_enters_next_open_with_stop_at_breakout_low(self):
        F = sg.build_features([flat_day(self.d0), self.orb_day()])
        sigs = [s for s in sg.find_signals(F, [1]) if s.strategy == "orb"]
        self.assertEqual(len(sigs), 1)
        s = sigs[0]
        self.assertEqual(s.side, 1)
        self.assertEqual(F.bars[s.k].ts.time(), time(9, 30))
        self.assertEqual(s.entry, 101.0)                 # open of the 09:35 candle
        self.assertEqual(s.stop, 100.0)                  # low of the breakout candle

    def test_signal_uses_only_past_bars(self):
        """Changing what happens after the signal must not change the signal itself."""
        a = self.orb_day()
        b = self.orb_day()[:6] + [bar(self.d1, "09:40", 101.3, 101.3, 90, 90)] * 1
        sa = [s for s in sg.find_signals(sg.build_features([flat_day(self.d0), a]), [1]) if s.strategy == "orb"][0]
        sb = [s for s in sg.find_signals(sg.build_features([flat_day(self.d0), b]), [1]) if s.strategy == "orb"][0]
        self.assertEqual((sa.k, sa.entry, sa.stop), (sb.k, sb.entry, sb.stop))

    def test_stop_assumed_first_when_both_levels_in_one_candle(self):
        day = flat_day(self.d1, 100, 20)
        day[5] = bar(self.d1, "09:40", 100, 103, 97, 100)          # touches both
        F = sg.build_features([flat_day(self.d0), day])
        s = sg.Signal("orb", F.day_start[1] + 3, 1, 100.0, 99.0, 1.0, "", True, None, None)
        res = sg.simulate(F, s, 2.0)
        self.assertEqual(res["reason"], "Stop loss")
        self.assertLess(res["r_net"], -1.0)                         # -1 R plus charges

    def test_live_trade_without_exit_stays_open(self):
        day = flat_day(self.d1, 100, 20)
        F = sg.build_features([flat_day(self.d0), day])
        s = sg.Signal("orb", F.day_start[1] + 3, 1, 100.0, 99.0, 1.0, "", True, None, None)
        self.assertTrue(sg.simulate(F, s, 2.0, open_ended=True)["open"])
        self.assertEqual(sg.simulate(F, s, 2.0)["reason"], "Day end")

    def test_strategy_needs_enough_trades_to_qualify(self):
        F = sg.build_features([flat_day(self.d0), self.orb_day()])
        rep = sg.backtest(F, [1])
        self.assertFalse(rep["orb"]["qualified"])
        self.assertIn("Fewer than", rep["orb"]["reason"])


class SignalApiTests(BaseCase):
    def test_replay_signals_endpoint(self):
        url = reverse("paper_trading:signals", args=[self.company.code])
        res = self.client.get(url, {"mode": "replay", "date": self.day.isoformat()})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual([s["key"] for s in data["strategies"]], [k for k, _, _ in sg.STRATEGIES])
        self.assertIn("signals", data)
        self.assertEqual(data["date"], self.day.isoformat())

    def test_unknown_company(self):
        res = self.client.get(reverse("paper_trading:signals", args=["NOPE"]), {"mode": "replay", "date": self.day.isoformat()})
        self.assertEqual(res.status_code, 400)


class LiveSignalGapTests(BaseCase):
    """Live mode fills the days the database is missing from Yahoo; Old mode never calls Yahoo."""

    def yahoo_frame(self, day):
        idx = pd.date_range(datetime.combine(day, time(9, 15)), periods=75, freq="5min", tz=svc.IST)
        return pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000}, index=idx)

    def test_live_history_includes_days_missing_from_database(self):
        today = svc.today_ist()
        missing = today - timedelta(days=1)            # the database stops at self.day (3 days ago)
        with mock.patch.object(svc, "_yf_history", return_value=self.yahoo_frame(missing)) as yf_hist, \
                mock.patch.object(svc, "fetch_live_today", return_value=pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"])):
            data = sg.session_signals(self.company, svc.LIVE, None)
        self.assertTrue(yf_hist.called)
        self.assertEqual(data["history"]["to"], missing.isoformat())
        self.assertEqual(data["history"]["from_yahoo"], 1)
        self.assertEqual(data["history"]["days"], 2)                 # database day + Yahoo day

    def test_yahoo_gap_ignores_today_and_known_days(self):
        today = svc.today_ist()
        frame = pd.concat([self.yahoo_frame(self.day), self.yahoo_frame(today - timedelta(days=1)),
                           self.yahoo_frame(today)])
        with mock.patch.object(svc, "_yf_history", return_value=frame):
            days = sg._yahoo_gap_bars(self.company, self.day, today)
        self.assertEqual([d[0].ts.date() for d in days], [today - timedelta(days=1)])

    def test_old_mode_does_not_call_yahoo(self):
        with mock.patch.object(svc, "_yf_history") as yf_hist:
            sg.session_signals(self.company, svc.REPLAY, self.day)
        yf_hist.assert_not_called()


class LivePreviousDayFrom5mTests(BaseCase):
    """Yahoo's daily feed can lag a day; Live mode then builds the previous day from 5-minute candles."""

    def test_uses_newer_day_from_5_minute_candles(self):
        today = svc.today_ist()
        older, newer = today - timedelta(days=3), today - timedelta(days=1)
        daily = pd.DataFrame({"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5]},
                             index=pd.DatetimeIndex([datetime.combine(older, time())], tz=svc.IST))
        idx = pd.date_range(datetime.combine(newer, time(9, 15)), periods=3, freq="5min", tz=svc.IST)
        five = pd.DataFrame({"Open": [20.0, 21.0, 22.0], "High": [21.0, 25.0, 23.0],
                             "Low": [19.0, 20.0, 18.0], "Close": [21.0, 22.0, 22.5]}, index=idx)

        def fake(symbol, **kw):
            return five if kw.get("interval") == "5m" else daily

        with mock.patch.object(svc, "_yf_history", side_effect=fake):
            prev = svc.live_previous_day(self.company)
        self.assertEqual(prev, {"date": newer.isoformat(), "open": 20.0, "high": 25.0, "low": 18.0, "close": 22.5})

    def test_keeps_daily_candle_when_it_is_current(self):
        today = svc.today_ist()
        newer = today - timedelta(days=1)
        daily = pd.DataFrame({"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5]},
                             index=pd.DatetimeIndex([datetime.combine(newer, time())], tz=svc.IST))
        with mock.patch.object(svc, "_yf_history", return_value=daily):
            prev = svc.live_previous_day(self.company)
        self.assertEqual((prev["date"], prev["close"]), (newer.isoformat(), 10.5))


# --------------------------------------------------------------------------
# wallet withdraw, limit orders, trade log, 1-minute candles
# --------------------------------------------------------------------------
from .models import PaperLimitOrder  # noqa: E402


class WithdrawTests(BaseCase):
    def test_withdraw_reduces_balance(self):
        acct = svc.withdraw_from_wallet(self.account, "2500")
        self.assertEqual(acct.balance, D("7500"))

    def test_cannot_withdraw_more_than_available(self):
        for bad in ["10000.01", "0", "-1", "abc"]:
            with self.assertRaises(svc.TradeError, msg=bad):
                svc.withdraw_from_wallet(self.account, bad)

    def test_withdraw_api(self):
        self.post(reverse("paper_trading:wallet_add"), {"amount": 500})
        r = self.post(reverse("paper_trading:wallet_withdraw"), {"amount": 200})
        self.assertEqual(r.json()["balance"], 300.0)
        self.assertEqual(self.post(reverse("paper_trading:wallet_withdraw"), {"amount": 301}).status_code, 400)


class ReplayLimitTests(BaseCase):
    """Old mode: candles 100, 102, 101, 105, 103, 99 (each with a 1-point range around open/close)."""

    def setUp(self):
        super().setUp()
        self.post(reverse("paper_trading:wallet_add"), {"amount": 10000})
        self.c = svc.load_replay_candles(self.company, self.day)

    def place(self, side, qty, limit, i, price=None, product="INTRADAY"):
        return self.post(reverse("paper_trading:order"), {
            "company": "TST", "side": side, "quantity": qty, "mode": "replay", "date": self.day.isoformat(),
            "market_time": self.c[i]["x"], "price": price if price is not None else self.c[i]["c"],
            "order_type": "LIMIT", "limit_price": limit, "product": product,
        }).json()

    def snap(self):
        return self.client.get(reverse("paper_trading:account"), {"mode": "replay", "date": self.day.isoformat()}).json()

    def test_marketable_limit_fills_at_once_at_the_current_price(self):
        res = self.place("BUY", 2, "103", 1, price="102")          # price 102 <= limit 103
        self.assertEqual(res["limit_order"]["status"], "FILLED")
        self.assertEqual((res["order"]["price"], res["order"]["order_type"]), (102.0, "LIMIT"))

    def test_resting_limit_waits_then_fills_at_the_limit(self):
        res = self.place("BUY", 2, "100.50", 1, price="102")
        self.assertIsNone(res["order"])
        lo = res["limit_order"]
        self.assertEqual(lo["status"], "OPEN")
        self.assertEqual(self.snap()["limit_orders"][0]["status"], "OPEN")
        # candle 2 (open 102, close 101, low 100) reaches 100.50
        r = self.post(reverse("paper_trading:limit_fill", args=[lo["id"]]), {"market_time": self.c[2]["x"]}).json()
        self.assertEqual((r["limit_order"]["status"], r["order"]["price"]), ("FILLED", 100.5))
        self.assertEqual(self.snap()["positions"][0]["quantity"], 2)

    def test_fill_is_checked_against_the_candle(self):
        lo = self.place("BUY", 1, "95", 1, price="102")["limit_order"]
        r = self.post(reverse("paper_trading:limit_fill", args=[lo["id"]]), {"market_time": self.c[3]["x"]})
        self.assertEqual(r.status_code, 400)                        # candle 3 never traded at 95
        self.assertIn("did not reach", r.json()["error"])

    def test_sell_limit_fills_at_the_limit_or_a_better_gap_open(self):
        # SELL limit 102.5 placed in candle 2 (price 101). Candle 3 opens 101 and trades up to 106: fills at 102.5.
        lo = self.place("SELL", 1, "102.5", 2, price="101")["limit_order"]
        r = self.post(reverse("paper_trading:limit_fill", args=[lo["id"]]), {"market_time": self.c[3]["x"]}).json()
        self.assertEqual(r["order"]["price"], 102.5)
        # another one, reported in candle 4 which OPENS at 105 - above the limit: fills at the better open
        lo2 = self.place("SELL", 1, "102.5", 2, price="101")["limit_order"]
        r2 = self.post(reverse("paper_trading:limit_fill", args=[lo2["id"]]), {"market_time": self.c[4]["x"]}).json()
        self.assertEqual(r2["order"]["price"], 105.0)

    def test_cancel_and_expire(self):
        a = self.place("BUY", 1, "90", 1, price="102")["limit_order"]
        b = self.place("BUY", 1, "91", 1, price="102")["limit_order"]
        self.assertEqual(self.post(reverse("paper_trading:limit_cancel", args=[a["id"]]), {}).json()["limit_order"]["status"], "CANCELLED")
        self.assertEqual(self.post(reverse("paper_trading:limit_cancel", args=[a["id"]]), {}).status_code, 400)
        self.post(reverse("paper_trading:limit_expire"), {"date": self.day.isoformat()})
        self.assertEqual(PaperLimitOrder.objects.get(pk=b["id"]).status, "EXPIRED")

    def test_unaffordable_fill_is_rejected_not_executed(self):
        lo = self.place("BUY", 500, "100.50", 1, price="102")["limit_order"]    # needs 50,250 > 10,000
        r = self.post(reverse("paper_trading:limit_fill", args=[lo["id"]]), {"market_time": self.c[2]["x"]}).json()
        self.assertEqual(r["limit_order"]["status"], "REJECTED")
        self.assertIsNone(r["order"])
        self.assertEqual(self.snap()["positions"], [])

    def test_books_do_not_see_each_others_orders(self):
        lo = self.place("BUY", 1, "90", 1, price="102")["limit_order"]
        r = self.post(reverse("paper_trading:limit_cancel", args=[lo["id"]]), {"book": 2})
        self.assertEqual(r.status_code, 400)                       # book 2 can't touch book 1's order


class LiveLimitTests(BaseCase):
    def setUp(self):
        super().setUp()
        self.account = svc.get_account(1)
        self.account.balance = D("10000")
        self.account.save()

    def frame(self, rows):
        return pd.DataFrame([{"timestamp": pd.Timestamp(ts), "open": o, "high": h, "low": l, "close": c, "volume": 10.0,
                              "source": "yfinance"} for ts, o, h, l, c in rows])

    def test_open_live_limit_fills_from_later_candles(self):
        now = datetime.now(svc.IST).replace(second=0, microsecond=0)
        placed = now - timedelta(minutes=20)
        lo = PaperLimitOrder.objects.create(account=self.account, company=self.company, side="BUY", quantity=3,
                                            limit_price=D("99"), mode="live", context="live", placed_at=placed)
        rows = [(placed - timedelta(minutes=5), 100, 101, 95, 100),     # before the order: ignored
                (placed + timedelta(minutes=5), 100, 101, 99.5, 100),
                (placed + timedelta(minutes=10), 100, 100, 98.5, 99)]   # reaches 99
        with mock.patch.object(svc, "fetch_live_today", return_value=self.frame(rows)):
            filled = svc.process_live_limits(self.account, now=now)
        lo.refresh_from_db()
        self.assertEqual((lo.status, len(filled), filled[0].price), ("FILLED", 1, D("99.0000")))

    def test_old_live_limit_expires(self):
        lo = PaperLimitOrder.objects.create(account=self.account, company=self.company, side="BUY", quantity=1,
                                            limit_price=D("50"), mode="live", context="live",
                                            placed_at=datetime.now(svc.IST) - timedelta(days=1))
        with mock.patch.object(svc, "_yahoo_day_5m", return_value=pd.DataFrame()):
            svc.process_live_limits(self.account)
        lo.refresh_from_db()
        self.assertEqual(lo.status, "EXPIRED")

    def test_reopened_next_day_still_fills_from_its_own_day(self):
        placed = datetime.combine(svc.today_ist() - timedelta(days=1), time(10, 0), tzinfo=svc.IST)
        lo = PaperLimitOrder.objects.create(account=self.account, company=self.company, side="SELL", quantity=2,
                                            limit_price=D("110"), mode="live", context="live", placed_at=placed)
        day = self.frame([(placed + timedelta(minutes=30), 108, 111, 107, 110.5)])
        with mock.patch.object(svc, "_yahoo_day_5m", return_value=day):
            svc.process_live_limits(self.account)
        lo.refresh_from_db()
        self.assertEqual((lo.status, lo.filled_order.price), ("FILLED", D("110.0000")))


class TradeLogTests(BaseCase):
    def trade(self, side, qty, price, minute, reason=""):
        t = datetime.combine(self.day, time(9, 15), tzinfo=svc.IST) + timedelta(minutes=minute)
        return svc.execute_order(account=self.account, company=self.company, side=side, quantity=qty,
                                 price=D(str(price)), mode=svc.REPLAY, replay_date=self.day, market_time=t,
                                 reason=reason)

    def test_buy_then_sell_is_one_trade(self):
        self.trade("BUY", 10, 100, 0)
        self.trade("SELL", 10, 104, 25, reason="target")
        rows = svc.trade_log(self.account, svc.REPLAY, self.day)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["side"], r["quantity"], r["entry_price"], r["exit_price"], r["pnl"], r["minutes"], r["exit_reason"]),
                         ("LONG", 10, 100.0, 104.0, 40.0, 25, "target"))

    def test_partial_exits_and_open_remainder(self):
        self.trade("BUY", 10, 100, 0)
        self.trade("BUY", 10, 102, 5)                  # average 101
        self.trade("SELL", 5, 105, 10)
        rows = svc.trade_log(self.account, svc.REPLAY, self.day)
        open_row, closed = rows[0], rows[1]
        self.assertEqual((open_row["status"], open_row["quantity"], open_row["entry_price"]), ("OPEN", 15, 101.0))
        self.assertEqual((closed["quantity"], closed["entry_price"], closed["pnl"]), (5, 101.0, 20.0))

    def test_short_trade(self):
        self.trade("SELL", 4, 100, 0)
        self.trade("BUY", 4, 97, 10)
        r = svc.trade_log(self.account, svc.REPLAY, self.day)[0]
        self.assertEqual((r["side"], r["pnl"]), ("SHORT", 12.0))


class OneMinuteCandleTests(BaseCase):
    def test_old_dates_beyond_yahoo_history_are_refused(self):
        with self.assertRaises(svc.TradeError):
            svc.replay_candles_1m(self.company, svc.today_ist() - timedelta(days=60))

    def test_minute_candles_from_yahoo(self):
        idx = pd.date_range(datetime.combine(self.day, time(9, 15)), periods=3, freq="1min", tz=svc.IST)
        raw = pd.DataFrame({"Open": [1.0, 2, 3], "High": [2.0, 3, 4], "Low": [0.5, 1, 2], "Close": [1.5, 2.5, 3.5]}, index=idx)
        with mock.patch.object(svc, "_yf_history", return_value=raw):
            r = self.client.get(reverse("paper_trading:candles", args=["TST"]),
                                {"mode": "replay", "date": self.day.isoformat(), "interval": "1m"}).json()
        self.assertEqual([c["c"] for c in r["candles"]], [1.5, 2.5, 3.5])


class PaperTrading2PageTests(BaseCase):
    def test_page2_uses_book_2(self):
        r = self.client.get(reverse("paper_trading:page2"))
        self.assertContains(r, "const BOOK = 2;")
        self.assertContains(r, "Paper Trading 2")
        self.assertContains(self.client.get(reverse("paper_trading:page")), "const BOOK = 1;")


class DuplicateExitTests(BaseCase):
    """Several open tabs sending the same stop loss / target / exit must never open a new position."""

    def setUp(self):
        super().setUp()
        self.post(reverse("paper_trading:wallet_add"), {"amount": 100000})

    def live(self, **body):
        payload = {"company": "TST", "mode": "live", "product": "INTRADAY", **body}
        with mock.patch.object(svc, "get_market_status", return_value="OPEN"), \
             mock.patch.object(svc, "fetch_live_today", return_value=live_frame(1, close=95.0)):
            return self.post(reverse("paper_trading:order"), payload)

    def position(self):
        pos = PaperPosition.objects.filter(account__owner_key="book:1", company=self.company).first()
        return pos.quantity if pos else 0

    def short_with_target(self):
        r = self.live(side="SELL", quantity=100, target_price="90")
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(self.position(), -100)

    def test_target_exit_sent_twice_closes_once(self):
        self.short_with_target()
        first = self.live(side="BUY", quantity=100, reason="target")
        second = self.live(side="BUY", quantity=100, reason="target")          # the other tab
        self.assertEqual(first.status_code, 200, first.content)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(second.json()["code"], "already_closed")
        self.assertEqual(self.position(), 0)                                    # no new LONG
        self.assertEqual(PaperOrder.objects.filter(reason="target").count(), 1)

    def test_exit_button_never_flips_and_is_capped(self):
        self.short_with_target()
        r = self.live(side="BUY", quantity=500, close_only=True)                # stale, bigger quantity
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()["order"]["quantity"], 100)
        self.assertEqual(self.position(), 0)
        again = self.live(side="BUY", quantity=100, close_only=True)
        self.assertEqual(again.status_code, 409)

    def test_stop_loss_exit_after_the_level_was_removed_is_refused(self):
        self.live(side="BUY", quantity=10, stop_loss_price="90")
        self.post(reverse("paper_trading:bracket"), {"company": "TST", "mode": "live", "product": "INTRADAY",
                                                     "clear_stop_loss": True})
        r = self.live(side="SELL", quantity=10, reason="stop_loss")
        self.assertEqual(r.status_code, 409)
        self.assertEqual(self.position(), 10)

    def test_same_order_id_runs_once(self):
        a = self.live(side="BUY", quantity=5, client_order_id="abc-123")
        b = self.live(side="BUY", quantity=5, client_order_id="abc-123")         # double click / retry
        self.assertEqual(a.status_code, 200)
        self.assertEqual(b.status_code, 409)
        self.assertEqual(self.position(), 5)

    def test_normal_orders_can_still_reverse(self):
        self.live(side="BUY", quantity=10)
        r = self.live(side="SELL", quantity=30)                                 # a deliberate manual reversal
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.position(), -20)

    def test_squareoff_from_two_tabs_closes_once(self):
        today = svc.today_ist()
        svc.execute_order(account=svc.get_account(1), company=self.company, side="BUY", quantity=10,
                          price=D("100"), mode=svc.LIVE, replay_date=None, market_time=at(today, 10, 0))
        with mock.patch.object(svc, "fetch_live_today", return_value=live_frame(1, close=101.0)):
            a = svc.squareoff_live_intraday(svc.get_account(1), now=at(today, 15, 21))
            b = svc.squareoff_live_intraday(svc.get_account(1), now=at(today, 15, 21))
        self.assertEqual((len(a["closed"]), len(b["closed"])), (1, 0))
        self.assertEqual(self.position(), 0)


class ConcurrentExitTests(TransactionTestCase):
    """Two exits arriving at the same moment (two tabs) - the order lock makes them run one at a time."""

    def test_simultaneous_exits(self):
        company = Company.objects.create(name="Test Co", code="TST")
        account = PaperAccount.objects.create(owner_key="book:1", balance=D("100000"))
        svc.execute_order(account=account, company=company, side="SELL", quantity=100, price=D("100"),
                          mode=svc.LIVE, replay_date=None, market_time=None, levels={"target_price": D("95")})
        barrier, results = threading.Barrier(2), []

        def exit_once():
            barrier.wait()
            try:
                svc.execute_order(account=account, company=company, side="BUY", quantity=100, price=D("95"),
                                  mode=svc.LIVE, replay_date=None, market_time=None, reason="target", close_only=True)
                results.append("filled")
            except svc.AlreadyClosed:
                results.append("already_closed")
            finally:
                connection.close()

        threads = [threading.Thread(target=exit_once) for _ in range(2)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(sorted(results), ["already_closed", "filled"])
        self.assertEqual(PaperPosition.objects.get(company=company).quantity, 0)


class ChargesTests(BaseCase):
    """Zerodha / Angel One charges (paper_trading/charges.py) and their use in the trade log."""

    def test_zerodha_intraday_worked_example(self):
        from . import charges
        # buy 1000 @ 100, sell 1000 @ 101 (NSE, intraday)
        c = charges.trade_charges("zerodha", "INTRADAY", D("100000"), D("101000"),
                                  [(D("100000"), D("1")), (D("101000"), D("1"))])
        self.assertEqual(c["brokerage"], D("40.00"))                 # 0.03% = Rs 30 -> capped at Rs 20, x2 orders
        self.assertEqual(c["stt"], D("25.25"))                       # 0.025% of the sell side
        self.assertEqual(c["exchange"], D("6.17"))                   # 0.00307% of 2,01,000
        self.assertEqual(c["stamp"], D("3.00"))                      # 0.003% of the buy side
        self.assertEqual(c["gst"], D("8.35"))                        # 18% of brokerage + exchange + SEBI
        self.assertEqual(c["dp"], D("0.00"))
        self.assertEqual(c["total"], sum(v for k, v in c.items() if k != "total"))

    def test_angel_one_brokerage_rate_minimum_and_cap(self):
        from . import charges
        self.assertEqual(charges.order_brokerage("angelone", "INTRADAY", D("10000")), D("10"))     # 0.1%
        self.assertEqual(charges.order_brokerage("angelone", "INTRADAY", D("1000")), D("5"))       # minimum
        self.assertEqual(charges.order_brokerage("angelone", "DELIVERY", D("500000")), D("20"))    # cap
        self.assertEqual(charges.order_brokerage("zerodha", "INTRADAY", D("10000")), D("3"))       # 0.03%
        self.assertEqual(charges.order_brokerage("zerodha", "DELIVERY", D("500000")), D("0"))      # free

    def test_delivery_pays_stt_both_sides_and_dp(self):
        from . import charges
        z = charges.trade_charges("zerodha", "DELIVERY", D("50000"), D("52000"), [(D("50000"), 1), (D("52000"), 1)])
        a = charges.trade_charges("angelone", "DELIVERY", D("50000"), D("52000"), [(D("50000"), 1), (D("52000"), 1)])
        self.assertEqual(z["stt"], D("102.00"))                      # 0.1% of 50,000 + 52,000
        self.assertEqual(z["stamp"], D("7.50"))                      # 0.015% of the buy
        self.assertEqual((z["brokerage"], z["dp"]), (D("0.00"), D("15.34")))
        self.assertEqual((a["brokerage"], a["dp"]), (D("40.00"), D("23.60")))

    def test_trade_log_charges_each_order_once(self):
        # two buys of 5 then one sell of 10: three orders -> three brokerages in one trade
        self.trade("BUY", 5, 100)
        self.trade("BUY", 5, 102)
        self.trade("SELL", 10, 103)
        row = [r for r in svc.trade_log(self.account, "live") if r["status"] == "CLOSED"][0]
        z = row["charges"]["zerodha"]
        expected = 0.0003 * (500 + 510 + 1030)                      # 0.03% each, all under the Rs 20 cap
        self.assertAlmostEqual(z["brokerage"], round(expected, 2), places=2)
        self.assertAlmostEqual(row["net_pnl"]["zerodha"], row["pnl"] - z["total"], places=2)

    def test_partial_exit_splits_the_entry_brokerage(self):
        self.trade("BUY", 10, 1000)                                  # Rs 10,000 entry order -> Rs 3 brokerage (Zerodha)
        self.trade("SELL", 4, 1010)
        rows = svc.trade_log(self.account, "live")
        closed = [r for r in rows if r["status"] == "CLOSED"][0]
        still = [r for r in rows if r["status"] == "OPEN"][0]
        # 40% of the entry order's Rs 3 + the whole exit order's brokerage (0.03% of 4,040 = 1.212)
        self.assertAlmostEqual(closed["charges"]["zerodha"]["brokerage"], round(3 * 0.4 + 1.212, 2), places=2)
        self.assertEqual(still["entry_legs"], [[10000.0, 0.6]])

    def test_page_ships_the_rates(self):
        r = self.client.get(reverse("paper_trading:page"))
        self.assertContains(r, 'id="brokers-data"')
        self.assertContains(r, "Angel One")
        self.assertContains(r, "zerodha.com/charges")


class ClaudeOrdersTests(BaseCase):
    """Claude Orders page: recommendations from claude_orders.json placed into Paper Trading 2."""

    def setUp(self):
        super().setUp()
        import tempfile
        from pathlib import Path
        from . import claude_orders
        self.mod = claude_orders
        self.tmp = Path(tempfile.mkdtemp()) / "claude_orders.json"
        self.patcher = mock.patch.object(claude_orders, "ORDERS_FILE", self.tmp)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        no_lists = mock.patch("paper_trading.surveillance.nse_lists", return_value={})     # no NSE download in tests
        no_lists.start()
        self.addCleanup(no_lists.stop)
        svc.add_to_wallet(svc.get_account(2), "100000")

    def write(self, session=None, **over):
        order = {"id": "20261019-1024-TST-BUY", "code": "TST", "name": "Test Co", "sector": "", "side": "BUY",
                 "product": "INTRADAY", "conviction": "Medium", "score": 1.0, "price_at": 100.0,
                 "price_time": "2026-10-19T10:20:00+05:30", "stop": 99.0, "target": 101.5, "stop_pct": 1.0,
                 "target_pct": 1.5, "reasons": [], "news": [], "square_off_by": "15:20"}
        order.update(over)
        self.tmp.write_text(json.dumps({"generated": "2026-10-19T10:24:00+05:30",
                                        "session": session or svc.today_ist().isoformat(), "orders": [order]}),
                            encoding="utf-8")

    def place(self, close=100.2, qty=10):
        with mock.patch.object(svc, "get_market_status", return_value="OPEN"), \
             mock.patch.object(svc, "fetch_live_today", return_value=live_frame(1, close=close)):
            return self.post(reverse("paper_trading:claude_orders_place"), {"id": "20261019-1024-TST-BUY", "quantity": qty})

    def test_places_into_book_2_with_percent_levels_from_the_fill(self):
        self.write()
        r = self.place(close=100.2)
        self.assertEqual(r.status_code, 200, r.content)
        o = PaperOrder.objects.get()
        self.assertEqual((o.account.owner_key, o.side, o.quantity, o.reason), ("book:2", "BUY", 10, "claude"))
        pos = PaperPosition.objects.get(account=o.account)
        self.assertAlmostEqual(float(pos.stop_loss), 100.2 * 0.99, places=2)        # 1% below the actual fill
        self.assertAlmostEqual(float(pos.target_price), 100.2 * 1.015, places=2)
        self.assertFalse(PaperOrder.objects.filter(account__owner_key="book:1").exists())

    def test_cannot_place_twice(self):
        self.write()
        self.assertEqual(self.place().status_code, 200)
        r = self.place()
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["code"], "already_placed")
        self.assertEqual(PaperOrder.objects.count(), 1)

    def test_old_recommendations_are_refused(self):
        self.write(session="2026-01-01")
        r = self.place()
        self.assertEqual(r.status_code, 400)
        self.assertIn("2026-01-01", r.json()["error"])

    def test_refused_when_price_already_passed_stop_or_target(self):
        self.write()
        self.assertEqual(self.place(close=98.5).json()["code"], "missed")          # below the 99 stop
        self.assertEqual(self.place(close=102.0).json()["code"], "missed")         # above the 101.5 target
        self.assertFalse(PaperOrder.objects.exists())

    def test_page_and_state(self):
        self.write()
        r = self.client.get(reverse("paper_trading:claude_orders"))
        self.assertContains(r, "Claude Orders")
        self.assertContains(r, "20261019-1024-TST-BUY")
        st = self.client.get(reverse("paper_trading:claude_orders_state")).json()
        self.assertTrue(st["is_today"])
        self.assertIsNone(st["orders"][0]["placed"])


class MonitorTests(BaseCase):
    """Server-side stop loss / target watcher (paper_trading/monitor.py) - works with no page open."""

    def setUp(self):
        super().setUp()
        from . import monitor
        self.mon = monitor
        self.book = svc.get_account(2)
        svc.add_to_wallet(self.book, "100000")
        self.today = svc.today_ist()

    def frame(self, rows):
        """5-minute candles today: [(HH:MM, open, high, low, close)]."""
        return pd.DataFrame([{"timestamp": pd.Timestamp(datetime.combine(self.today, time(*map(int, hm.split(":"))),
                                                                         tzinfo=svc.IST)),
                              "open": o, "high": h, "low": l, "close": c, "volume": 1000}
                             for hm, o, h, l, c in rows])

    def open_long(self, at="10:02", sl="99", tp="102"):
        svc.execute_order(account=self.book, company=self.company, side="BUY", quantity=10, price=D("100"),
                          mode=svc.LIVE, replay_date=None, market_time=at_(self.today, at),
                          levels={"sl_price": D(sl), "target_price": D(tp)})
        pos = PaperPosition.objects.get(account=self.book)
        # pretend the position (and its levels) were last changed at entry time
        PaperPosition.objects.filter(pk=pos.pk).update(updated_at=at_(self.today, at))
        return PaperPosition.objects.get(pk=pos.pk)

    def test_level_hit_rules(self):
        pos = self.open_long()
        since = at_(self.today, "10:02")
        before = self.frame([("10:00", 100, 100.5, 98.0, 99.5)])                    # dips below SL but before entry
        self.assertIsNone(self.mon.level_hit(pos, before, since))
        tp = self.frame([("10:05", 100, 102.4, 99.8, 102.1)])
        self.assertEqual(self.mon.level_hit(pos, tp, since)[:2], ("target", 102.0))
        gap = self.frame([("10:05", 98.5, 99.0, 98.2, 98.6)])                        # opened below the stop
        self.assertEqual(self.mon.level_hit(pos, gap, since)[:2], ("stop_loss", 98.5))
        both = self.frame([("10:05", 100, 102.5, 98.5, 101)])
        self.assertEqual(self.mon.level_hit(pos, both, since)[0], "stop_loss")

    def test_run_once_closes_on_target_and_never_twice(self):
        self.open_long()
        df = self.frame([("10:05", 100, 100.6, 99.6, 100.4), ("10:10", 100.4, 102.3, 100.2, 102.1)])
        with mock.patch.object(svc, "fetch_live_today", return_value=df):
            exits, _ = self.mon.run_once(now=at_(self.today, "10:16"))
            again, _ = self.mon.run_once(now=at_(self.today, "10:17"))
        self.assertEqual(len(exits), 1)
        self.assertEqual((exits[0].reason, exits[0].side, exits[0].price), ("target", "SELL", D("102.0000")))
        self.assertEqual(again, [])
        self.assertEqual(PaperPosition.objects.get(account=self.book).quantity, 0)

    def test_squares_off_intraday_at_cutoff(self):
        svc.execute_order(account=self.book, company=self.company, side="BUY", quantity=5, price=D("100"),
                          mode=svc.LIVE, replay_date=None, market_time=at_(self.today, "10:00"))
        df = self.frame([("15:15", 101, 101.2, 100.8, 101.0)])
        with mock.patch.object(svc, "fetch_live_today", return_value=df):
            _, closed = self.mon.run_once(now=at_(self.today, "15:21"))
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].reason, "session_end")


def at_(day, hm):
    h, m = map(int, hm.split(":"))
    return datetime.combine(day, time(h, m), tzinfo=svc.IST)


class ClaudeBoardTests(BaseCase):
    """Refresh orders: the signal board's decision rules and the background refresh endpoint."""

    def stock(self, forecast_up=True, cred=True, price=101.0, **over):
        pu = 0.7 if forecast_up else 0.3
        s = {"code": "TST.NS", "name": "Test Co", "sector": "Tech", "price": price, "asof": "2026-10-21T10:00:00+05:30",
             "has_candles": True, "vwap_last": 100.0, "levels": {"pdc": 100.0}, "active": [], "pattern_vote": 0, "ml": None,
             "forecast": {"band": [[i, price - 1, price + 1] for i in range(12)],
                          "main": {"label": "15 min", "p_up": pu, "ret_pct": 0.1, "accuracy": 0.56 if cred else 0.48,
                                   "baseline": 0.52},
                          "news": None, "news_items": []}}
        s.update(over)
        return s

    def test_buy_needs_a_model_signal_with_price_action(self):
        from . import claude_board as b
        s = b.score(self.stock(), {"dir": 1, "day_pct": 0.5, "last30_pct": 0.1}, set(), set())
        self.assertEqual(s["decision"], "BUY")
        self.assertAlmostEqual(s["plan"]["risk_pct"], 0.990, places=2)            # (101 - 100) / 101, inside 0.4-1.2 %
        self.assertAlmostEqual(s["plan"]["target"], 101 * (1 + 1.5 * 0.0099), places=1)
        order = b.to_order(s, datetime(2026, 10, 21, 10, 5, tzinfo=svc.IST))
        self.assertEqual(order["id"], "20261021-1005-TST-BUY")
        self.assertEqual([r["signal"] for r in order["reasons"]], ["Candle forecast", "Price action", "Market (NIFTY)"])

    def test_untested_forecast_or_market_against_gives_watch(self):
        from . import claude_board as b
        up = {"dir": 1, "day_pct": 0.5, "last30_pct": 0.1}
        self.assertEqual(b.score(self.stock(cred=False), up, set(), set())["decision"], "WATCH")
        self.assertEqual(b.score(self.stock(), {"dir": -1, "day_pct": -0.5, "last30_pct": -0.1}, set(), set())["decision"], "WATCH")
        self.assertEqual(b.score(self.stock(has_candles=False), up, set(), set())["decision"], "SKIP")

    def test_refresh_runs_in_background_and_writes_the_file(self):
        import tempfile
        from pathlib import Path
        from . import claude_board, claude_orders
        tmp = Path(tempfile.mkdtemp()) / "claude_orders.json"
        payload = {"generated": "2026-10-21T10:05:00+05:30", "session": svc.today_ist().isoformat(), "orders": []}
        ran = threading.Event()

        def fake_build():
            ran.set()
            return payload
        with mock.patch.object(claude_orders, "ORDERS_FILE", tmp), mock.patch.object(claude_board, "build", fake_build), \
             mock.patch.object(claude_board, "archive"), mock.patch("django.db.connection.close"):
            with mock.patch("threading.Thread.start"):                               # first: check the busy guard
                claude_board._set(running=True)
                self.assertEqual(self.client.post(reverse("paper_trading:claude_orders_refresh")).status_code, 409)
                claude_board._set(running=False)
            r = self.client.post(reverse("paper_trading:claude_orders_refresh"))
            self.assertEqual(r.status_code, 202)
            for _ in range(100):
                if not claude_board.progress()["running"]:
                    break
                threading.Event().wait(0.05)
            self.assertTrue(ran.is_set())
            self.assertEqual(json.loads(tmp.read_text(encoding="utf-8"))["generated"], payload["generated"])
            st = self.client.get(reverse("paper_trading:claude_orders_state")).json()
        self.assertFalse(st["refresh"]["running"])
        self.assertEqual(st["refresh"]["orders"], 0)
        self.assertIsNone(st["refresh"]["error"])


class FakeAngel:
    """Stands in for paper_trading.angel: an order book in memory, prices set by the test."""

    def __init__(self):
        self.price, self.orders, self.n = 100.0, {}, 0
        self.fill_market, self.cancel_works = True, True

    def _new(self, **o):
        self.n += 1
        oid = f"A{self.n}"
        self.orders[oid] = o
        return oid

    def place_market(self, account, inst, side, qty, consent=False):
        return self._new(kind="market", side=side, qty=qty, status="complete" if self.fill_market else "open",
                         avg=self.price if self.fill_market else 0, consent=consent)

    def place_stop(self, account, inst, side, qty, trigger, consent=False):
        return self._new(kind="stop", side=side, qty=qty, status="trigger pending", avg=0, trigger=trigger,
                         consent=consent), D(str(trigger))

    def cancel(self, account, oid, variety):
        if self.cancel_works and self.orders[oid]["status"] == "trigger pending":
            self.orders[oid]["status"] = "cancelled"

    def order_status(self, account, oid):
        o = self.orders[oid]
        return o["status"], o["avg"], o["qty"] if o["status"] == "complete" else 0, ""

    def wait_for(self, account, oid, seconds=0, step=0):
        return self.order_status(account, oid)

    def patches(self, live=True, configured=True):
        from . import angel
        cfg = ({k: "x" for k in angel.KEYS}, [] if configured else ["ANGEL_VIVEK_API_KEY"])
        return [mock.patch.object(angel, "config", return_value=cfg),
                mock.patch.object(angel, "live_enabled", return_value=live),
                mock.patch.object(angel, "instrument", return_value={"symbol": "TST-EQ", "token": "1", "tick": D("0.05")}),
                mock.patch.object(angel, "ltp", side_effect=lambda a, i: self.price)] + \
               [mock.patch.object(angel, n, getattr(self, n)) for n in ("place_market", "place_stop", "cancel", "order_status", "wait_for")]


class BrokerTests(TestCase):
    """Real Angel One trades from Claude Orders - against FakeAngel, never the real broker."""

    MON_10 = datetime(2026, 10, 26, 10, 0, tzinfo=svc.IST)

    def setUp(self):
        from . import broker
        self.b = broker
        self.fake = FakeAngel()
        self.start(self.fake.patches())
        self.start([mock.patch.object(broker, "get_market_status", return_value="OPEN"),
                    mock.patch("paper_trading.surveillance.nse_lists", return_value={})])

    def start(self, ps):
        for p in ps:
            p.start()
            self.addCleanup(p.stop)

    def place(self, side="BUY", sl=2, tp=3, now=None, **kw):
        return self.b.place("vivek", rec_id="r1", code="TST.NS", side=side, quantity=10, stop_pct=sl, target_pct=tp,
                            now=now or self.MON_10, **kw)

    def test_refusals(self):
        from . import angel
        with mock.patch.object(angel, "live_enabled", return_value=False):
            with self.assertRaisesMessage(self.b.BrokerError, "ANGEL_LIVE_TRADING=1"):
                self.place()
        with mock.patch.object(angel, "config", return_value=({}, ["ANGEL_VIVEK_PIN"])):
            with self.assertRaisesMessage(self.b.BrokerError, "ANGEL_VIVEK_PIN"):
                self.place()
        with self.assertRaisesMessage(self.b.BrokerError, "09:15 and 15:00"):
            self.place(now=datetime(2026, 10, 26, 15, 5, tzinfo=svc.IST))
        with self.assertRaisesMessage(self.b.BrokerError, "already passed the recommendation's stop"):
            self.place(guard=(101, 110))
        with self.assertRaisesMessage(self.b.BrokerError, "between 0.05% and 20%"):
            self.place(sl=25)
        with mock.patch.object(angel, "max_order_value", return_value=500):
            with self.assertRaisesMessage(self.b.BrokerError, "above the limit"):
                self.place()
        self.assertEqual(self.fake.orders, {})                              # nothing reached the broker

    def test_entry_then_stop_order_at_broker_and_no_duplicate(self):
        bo = self.place()
        self.assertEqual((bo.status, bo.entry_price, bo.sl_trigger), ("open", D("100.00"), D("98")))
        stop = self.fake.orders[bo.sl_order_id]
        self.assertEqual((stop["kind"], stop["side"], stop["qty"]), ("stop", "SELL", 10))
        with self.assertRaisesMessage(self.b.BrokerError, "Already placed today"):
            self.place()

    def test_target_cancels_the_stop_first_then_exits(self):
        bo = self.place()
        self.fake.price = 102.0
        self.b.watch(self.MON_10)
        bo.refresh_from_db()
        self.assertEqual(bo.status, "open")
        self.fake.price = 103.5
        self.b.watch(self.MON_10)
        bo.refresh_from_db()
        self.assertEqual(self.fake.orders[bo.sl_order_id]["status"], "cancelled")
        self.assertEqual((bo.status, bo.exit_reason, bo.exit_price, bo.pnl), ("closed", "target", D("103.50"), D("35.0000")))

    def test_no_exit_while_the_stop_cannot_be_cancelled(self):
        bo = self.place()
        self.fake.cancel_works = False
        self.fake.price = 104.0
        self.b.watch(self.MON_10)
        bo.refresh_from_db()
        self.assertEqual((bo.status, bo.exit_order_id), ("open", ""))           # never two exits
        self.assertIn("still 'trigger pending'", bo.message)

    def test_stop_filled_at_broker_and_1515_exit(self):
        bo = self.place()
        self.fake.orders[bo.sl_order_id].update(status="complete", avg=97.9)
        self.b.watch(self.MON_10)
        bo.refresh_from_db()
        self.assertEqual((bo.status, bo.exit_reason, bo.exit_price), ("closed", "stop_loss", D("97.90")))
        bo2 = self.b.place("sethu", rec_id="r1", code="TST.NS", side="SELL", quantity=5, stop_pct=2, target_pct=3, now=self.MON_10)
        self.assertEqual((bo2.stop, bo2.target), (D("102.00"), D("97.00")))
        self.b.watch(datetime(2026, 10, 26, 15, 15, tzinfo=svc.IST))
        bo2.refresh_from_db()
        self.assertEqual((bo2.status, bo2.exit_reason), ("closed", "squareoff"))
        self.assertEqual(self.fake.orders[bo2.exit_order_id]["side"], "BUY")

    def test_pending_entry_finished_by_the_watcher(self):
        self.fake.fill_market = False
        bo = self.place()
        self.assertEqual(bo.status, "pending")
        self.fake.orders[bo.entry_order_id].update(status="complete", avg=100.2)
        self.b.watch(self.MON_10)
        bo.refresh_from_db()
        self.assertEqual((bo.status, bo.entry_price), ("open", D("100.20")))
        self.assertEqual((bo.stop, bo.target), (D("98.20"), D("103.21")))         # % from the actual fill
        self.assertEqual(self.fake.orders[bo.sl_order_id]["trigger"], D("98.20"))


class ClaudeOrdersAccountsTests(ClaudeOrdersTests):
    """Claude Orders page: per-account placing (paper with edited levels, Angel One via broker.place)."""

    def test_paper_with_edited_stop_and_target(self):
        self.write()
        with mock.patch.object(svc, "get_market_status", return_value="OPEN"), \
             mock.patch.object(svc, "fetch_live_today", return_value=live_frame(1, close=100.2)):
            r = self.post(reverse("paper_trading:claude_orders_place"),
                          {"id": "20261019-1024-TST-BUY", "account": "paper", "quantity": 10, "stop_pct": 0.7, "target_pct": 1.8})
        self.assertEqual(r.status_code, 200, r.content)
        pos = PaperPosition.objects.get(account=svc.get_account(2), quantity=10)
        self.assertEqual((pos.stop_loss, pos.target_price), (D("99.50"), D("102.00")))

    def test_angel_account_goes_through_the_broker(self):
        from . import broker
        fake = FakeAngel()
        fake.price = 100.2
        now = datetime.combine(svc.today_ist(), time(10, 0), tzinfo=svc.IST)
        for p in fake.patches() + [mock.patch.object(broker, "get_market_status", return_value="OPEN"),
                                   mock.patch.object(broker, "now_ist", return_value=now)]:
            p.start()
            self.addCleanup(p.stop)
        self.write()
        r = self.post(reverse("paper_trading:claude_orders_place"),
                      {"id": "20261019-1024-TST-BUY", "account": "vivek", "quantity": 7, "stop_pct": 1, "target_pct": 1.3})
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()["broker"]["status"], "open")
        self.assertEqual((r.json()["broker"]["stop"], r.json()["broker"]["target"]), (99.2, 101.5))    # from the 100.2 fill
        st = self.client.get(reverse("paper_trading:claude_orders_state")).json()
        self.assertEqual(st["orders"][0]["broker"]["vivek"]["quantity"], 7)
        self.assertIsNone(st["orders"][0]["broker"]["sethu"])
        self.assertIsNone(st["orders"][0]["placed"])                                  # paper untouched
        again = self.post(reverse("paper_trading:claude_orders_place"),
                          {"id": "20261019-1024-TST-BUY", "account": "vivek", "quantity": 7})
        self.assertEqual(again.status_code, 409)

    def test_wallets(self):
        from . import angel
        with mock.patch.object(angel, "funds", side_effect=lambda acc, use_cache=True: {"configured": acc == "vivek", "available": 12345.0}):
            w = self.client.get(reverse("paper_trading:claude_orders_wallets")).json()
        self.assertEqual(w["vivek"]["available"], 12345.0)
        self.assertFalse(w["sethu"]["configured"])
        self.assertEqual(w["paper"]["available"], 100000.0)


class SurveillanceTests(TestCase):
    """Stocks Angel One calls "cautionary listings": NSE ASM / GSM lists and its own recent refusals."""

    def test_caution_from_nse_lists_and_past_refusals(self):
        from . import surveillance as sv
        from .models import BrokerOrder
        nse = {"GANDHAR": "Long Term Additional Surveillance Measure (LTASM) - Stage I"}
        self.assertIn("LTASM", sv.caution("GANDHAR.NS", nse, {}))
        self.assertIsNone(sv.caution("INFY.NS", nse, {}))
        BrokerOrder.objects.create(account="vivek", session=svc.today_ist(), code="HEXT.NS", side="SELL", quantity=1,
                                   stop=1, target=1, status="rejected",
                                   message="Angel One: The order cannot be processed as the token is categorised under cautionary listings by the exchange.")
        self.assertIn("flagged it", sv.caution("HEXT.NS", nse, None))

    def test_broker_refuses_before_sending(self):
        from . import broker, surveillance as sv
        fake = FakeAngel()
        for p in fake.patches() + [mock.patch.object(broker, "get_market_status", return_value="OPEN"),
                                   mock.patch.object(sv, "nse_lists", return_value={"TST": "ASM Stage I"})]:
            p.start()
            self.addCleanup(p.stop)
        now = datetime(2026, 10, 26, 10, 0, tzinfo=svc.IST)
        with self.assertRaises(broker.BrokerError) as cm:                       # flagged: asks before sending
            broker.place("vivek", rec_id="r", code="TST.NS", side="BUY", quantity=1, stop_pct=1, target_pct=2, now=now)
        self.assertEqual(cm.exception.code, "consent_required")
        self.assertEqual(fake.orders, {})
        bo = broker.place("vivek", rec_id="r", code="TST.NS", side="BUY", quantity=1, stop_pct=1, target_pct=2,
                          consent=True, now=now)
        self.assertTrue(bo.scrip_consent)
        self.assertEqual([o["consent"] for o in fake.orders.values()], [True, True])     # entry and stop-loss order

    def test_scripconsent_in_the_request_and_first_ab4036_asks_for_consent(self):
        from . import angel, broker
        self.assertEqual(angel._base({"symbol": "X-EQ", "token": "1"}, "SELL", 5, consent=True)["scripconsent"], "yes")
        self.assertNotIn("scripconsent", angel._base({"symbol": "X-EQ", "token": "1"}, "SELL", 5))
        fake = FakeAngel()

        def refuse(account, inst, side, qty, consent=False):
            raise angel.AngelError("Angel One: The order cannot be processed as the token is categorised under "
                                   "cautionary listings by the exchange. (AB4036)")
        for p in [x for x in fake.patches() if getattr(x, "attribute", "") != "place_market"] + [
                mock.patch.object(angel, "place_market", refuse),
                mock.patch.object(broker, "get_market_status", return_value="OPEN"),
                mock.patch("paper_trading.surveillance.nse_lists", return_value={})]:
            p.start()
            self.addCleanup(p.stop)
        with self.assertRaises(broker.BrokerError) as cm:
            broker.place("vivek", rec_id="r", code="ENRIN.NS", side="SELL", quantity=1, stop_pct=1, target_pct=2,
                         now=datetime(2026, 10, 26, 10, 0, tzinfo=svc.IST))
        self.assertEqual(cm.exception.code, "consent_required")


class AngelRetryTests(TestCase):
    """Network retries: logins / reads retry any network error, orders only when certainly never sent."""

    def test_retry_rules(self):
        import requests
        from . import angel
        calls = []

        def flaky(exc, n):
            def fn():
                calls.append(1)
                if len(calls) <= n:
                    raise exc
                return "ok"
            return fn
        with mock.patch.object(angel._time, "sleep"):
            self.assertEqual(angel._retry(flaky(requests.exceptions.ConnectTimeout("x"), 2), safe_any=False), "ok")
            calls.clear()
            self.assertEqual(angel._retry(flaky(requests.exceptions.ReadTimeout("x"), 1), safe_any=True), "ok")
            calls.clear()
            with self.assertRaises(requests.exceptions.ReadTimeout):              # an order may have arrived
                angel._retry(flaky(requests.exceptions.ReadTimeout("x"), 1), safe_any=False)
            self.assertEqual(len(calls), 1)

    def test_uncertain_order_is_reported_and_blocks_a_resend(self):
        import requests
        from . import angel, broker
        fake = FakeAngel()
        sc = mock.Mock()
        sc.placeOrderFullResponse.side_effect = requests.exceptions.ReadTimeout("slow")
        ps = [p for p in fake.patches() if getattr(p, "attribute", "") != "place_market"] + [
            mock.patch.object(angel, "client", return_value=sc),
            mock.patch.object(broker, "get_market_status", return_value="OPEN"),
            mock.patch("paper_trading.surveillance.nse_lists", return_value={})]
        for p in ps:
            p.start()
            self.addCleanup(p.stop)
        now = datetime(2026, 10, 26, 10, 0, tzinfo=svc.IST)
        with self.assertRaisesMessage(broker.BrokerError, "MAY have gone through"):
            broker.place("vivek", rec_id="r", code="TST.NS", side="BUY", quantity=1, stop_pct=1, target_pct=2, now=now)
        self.assertEqual(sc.placeOrderFullResponse.call_count, 1)               # not sent twice
        with self.assertRaisesMessage(broker.BrokerError, "Already placed today"):
            broker.place("vivek", rec_id="r", code="TST.NS", side="BUY", quantity=1, stop_pct=1, target_pct=2, now=now)


class AngelSessionTests(TestCase):
    def test_cautionary_rejection_is_not_a_session_error_and_is_sent_once(self):
        from . import angel
        self.assertFalse(angel._session_expired("The order cannot be processed as the token is categorised under "
                                                "cautionary listings by the exchange.", "AB4036"))
        self.assertTrue(angel._session_expired("Invalid Token", "AG8001"))
        sc = mock.Mock()
        sc.placeOrderFullResponse.return_value = {"status": False, "errorcode": "AB4036", "data": None,
                                                  "message": "The order cannot be processed as the token is categorised "
                                                             "under cautionary listings by the exchange."}
        with mock.patch.object(angel, "client", return_value=sc):
            with self.assertRaisesMessage(angel.AngelError, "AB4036"):
                angel.place_market("vivek", {"symbol": "X-EQ", "token": "1", "tick": D("0.05")}, "SELL", 1)
        self.assertEqual(sc.placeOrderFullResponse.call_count, 1)


class ExitAndRouteTests(TestCase):
    """Exiting a real trade, and per-account network routing for Angel One."""

    def setUp(self):
        from . import broker
        self.b = broker
        self.fake = FakeAngel()
        for p in self.fake.patches() + [mock.patch.object(broker, "get_market_status", return_value="OPEN"),
                                        mock.patch("paper_trading.surveillance.nse_lists", return_value={})]:
            p.start()
            self.addCleanup(p.stop)
        self.now = datetime.combine(svc.today_ist(), time(10, 0), tzinfo=svc.IST)

    def place(self, acc):
        return self.b.place(acc, rec_id="r", code="TST.NS", side="SELL", quantity=5, stop_pct=1, target_pct=2, now=self.now)

    def test_exit_trade_cancels_the_stop_first_then_exits(self):
        bo = self.place("vivek")
        out = self.b.exit_trade(bo.pk)
        self.assertEqual((out.status, out.exit_reason), ("closed", "manual"))
        self.assertEqual(self.fake.orders[bo.sl_order_id]["status"], "cancelled")
        self.assertEqual(self.fake.orders[out.exit_order_id]["side"], "BUY")
        with self.assertRaisesMessage(self.b.BrokerError, "is not open"):
            self.b.exit_trade(bo.pk)

    def test_exit_api_takes_one_trade(self):
        bo = self.place("vivek")
        url = reverse("paper_trading:claude_orders_exit")
        self.assertEqual(self.client.post(url, {"account": "vivek"}, content_type="application/json").status_code, 400)
        bo.refresh_from_db()
        self.assertEqual(bo.status, "open")
        r = self.client.post(url, {"id": bo.pk}, content_type="application/json").json()
        self.assertTrue(r["ok"])

    def test_route_binds_each_account_to_its_adapter(self):
        import os
        from . import netroute
        ads = {"Wi-Fi": {"up": True, "ipv4": ["192.168.1.10"]}, "Ethernet 3": {"up": True, "ipv4": ["192.168.42.129"]}}
        env = {"ANGEL_VIVEK_INTERFACE": "Wi-Fi", "ANGEL_SETHU_INTERFACE": "Ethernet 3"}
        with mock.patch.object(netroute, "adapters", return_value=ads), mock.patch.dict(os.environ, env):
            self.assertEqual(netroute.source_ip("vivek"), "192.168.1.10")
            self.assertEqual(netroute.source_ip("sethu"), "192.168.42.129")
            with netroute.via("sethu") as ip:
                self.assertEqual(ip, "192.168.42.129")
                ad = netroute._local.session.get_adapter("https://apiconnect.angelone.in")
                self.assertEqual(ad.ip, "192.168.42.129")
            self.assertIsNone(getattr(netroute._local, "session", None))      # restored after the block
        with mock.patch.object(netroute, "adapters", return_value={"Wi-Fi": ads["Wi-Fi"]}), mock.patch.dict(os.environ, env):
            with self.assertRaisesMessage(netroute.RouteError, "not connected"):  # phone unplugged: refuse
                netroute.source_ip("sethu")
        with mock.patch.dict(os.environ, {"ANGEL_SETHU_INTERFACE": "", "ANGEL_SETHU_SOURCE_IP": ""}):
            self.assertIsNone(netroute.source_ip("sethu"))                       # not set: normal route


class TurnoverParseTests(TestCase):
    def test_both_nse_formats(self):
        from .claude_board import turnover_lakh
        self.assertEqual(turnover_lakh(2000), 2000.0)                 # old: a number in Rs lakh
        self.assertAlmostEqual(turnover_lakh("3,812.46 Cr"), 381246.0, places=6)   # new: text in Rs crore
        self.assertAlmostEqual(turnover_lakh("80.76 Cr"), 8076.0, places=6)
        self.assertEqual(turnover_lakh("950.5 L"), 950.5)
        self.assertEqual(turnover_lakh(None), 0.0)
        self.assertEqual(turnover_lakh("-"), 0.0)


class OrderValueLimitTests(TestCase):
    def test_limit_values(self):
        import os
        from . import angel
        for raw, want in (("unlimited", None), ("0", None), ("none", None), ("500000", 500000.0), ("", 200000.0), ("abc", 200000.0)):
            with mock.patch.dict(os.environ, {"ANGEL_MAX_ORDER_VALUE": raw}):
                self.assertEqual(angel.max_order_value(), want, raw)

    def test_unlimited_lets_a_big_order_through(self):
        import os
        from . import broker
        fake = FakeAngel()
        fake.price = 2703.20
        for p in fake.patches() + [mock.patch.object(broker, "get_market_status", return_value="OPEN"),
                                   mock.patch("paper_trading.surveillance.nse_lists", return_value={}),
                                   mock.patch.dict(os.environ, {"ANGEL_MAX_ORDER_VALUE": "unlimited"})]:
            p.start()
            self.addCleanup(p.stop)
        bo = broker.place("vivek", rec_id="r", code="TST.NS", side="BUY", quantity=100, stop_pct=1, target_pct=2,
                          now=datetime(2026, 10, 26, 10, 0, tzinfo=svc.IST))           # Rs 2,70,320
        self.assertEqual(bo.status, "open")


class ClaudeOrderHistoryTests(TestCase):
    def test_refresh_lists_are_archived_once_and_history_renders(self):
        from . import claude_board
        from .models import BrokerOrder, ClaudeOrderList
        payload = {"generated": "2026-11-02T10:05:00+05:30", "session": svc.today_ist().isoformat(), "counts": {"SELL": 1},
                   "orders": [{"id": "x-TST-SELL", "code": "TST.NS", "name": "Test", "side": "SELL", "price_at": 100,
                               "stop": 101, "target": 98, "stop_pct": 1, "target_pct": 2, "conviction": "High"}]}
        claude_board.archive(payload)
        claude_board.archive(payload)
        self.assertEqual(ClaudeOrderList.objects.count(), 1)
        BrokerOrder.objects.create(account="vivek", session=svc.today_ist(), rec_id="x-TST-SELL", code="TST.NS", side="SELL",
                                   quantity=5, stop=101, target=98, status="closed", entry_price=100, exit_price=98)
        r = self.client.get(reverse("paper_trading:claude_orders_history"))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "Vivek · 5 · closed")
        self.assertContains(r, "₹10.00")
        self.assertEqual(self.client.get(reverse("paper_trading:claude_orders_history") + "?days=7&account=sethu").status_code, 200)


class MostActiveCandidateTests(TestCase):
    """Most active stocks are looked at on Refresh, and shown on an order - never a vote for its direction."""

    def setUp(self):
        for code in ("HDFCBANK.NS", "PENNY.NS"):
            Company.objects.create(name=code, code=code)
        Company.objects.create(name="Gold ETF", code="GOLDBEES.NS", sector="ETF")

    def candidates(self, live):
        from . import claude_board
        with mock.patch("news.fetch.market_news", return_value={"items": []}), \
             mock.patch("pathlib.Path.glob", return_value=[]):
            return claude_board.candidates(live)

    def test_listed_with_the_reason_and_cheap_or_etf_rows_left_out(self):
        order, why = self.candidates({
            "most_active_value": [{"symbol": "HDFCBANK", "value_cr": 1857.9}, {"symbol": "GOLDBEES", "value_cr": 900.0}],
            "most_active_volume": [{"symbol": "HDFCBANK", "value_cr": 1857.9}, {"symbol": "PENNY", "value_cr": 3.5}]})
        self.assertEqual(order, ["HDFCBANK.NS"])
        self.assertEqual(why["HDFCBANK.NS"], ["Live: most active by value (₹1,858 Cr traded)",
                                              "Live: most active by volume (₹1,858 Cr traded)"])

    def test_shown_on_the_order_without_changing_the_signals(self):
        from datetime import datetime
        from . import claude_board
        s = {"code": "HDFCBANK.NS", "name": "HDFC", "side": "SELL", "decision": "SELL", "conviction": "Medium", "score": -0.5,
             "price": 100.0, "asof": "x", "plan": {"stop": 101.0, "target": 98.5, "risk_pct": 1.0},
             "signals": [{"name": "ML model", "dir": -1, "cred": True, "detail": "down"}],
             "why": ["Live: most active by value (₹1,858 Cr traded)", "Live: top loser -2.00%"]}
        o = claude_board.to_order(s, datetime(2026, 11, 5, 10, 0))
        self.assertEqual(o["reasons"], [{"signal": "ML model", "detail": "down"},
                                        {"signal": "Most active", "detail": "value (₹1,858 Cr traded)"}])


def signal(name, direction, weight=1.0):
    return {"name": name, "dir": direction, "w": weight, "cred": bool(direction and weight), "detail": ""}


class ClaudeBoard2DecisionTests(TestCase):
    """Claude Orders 2.0: which signals may make a BUY / SELL."""

    def setUp(self):
        from . import claude_board2
        self.b = claude_board2

    def decide(self, *signals):
        return self.b.decide([signal(*s) for s in signals])[0]

    def test_needs_stock_signals_and_a_model(self):
        b = self.b
        self.assertEqual(self.decide(("Candle forecast", 1), ("Price action", 1), (b.MARKET, 1)), "BUY")
        self.assertEqual(self.decide(("Candle forecast", -1), (b.BEHAVIOUR, -1), (b.SECTOR, -1)), "SELL")
        self.assertEqual(self.decide(("Candle forecast", 1), ("Price action", 1)), "WATCH")          # only two agree
        # the market alone can't carry a stock, however many indices are up
        self.assertEqual(self.decide(("Candle forecast", 1), (b.MARKET, 1), (b.ALL_INDICES, 1), (b.GLOBAL, 1), (b.SECTOR, 1)), "WATCH")
        self.assertEqual(self.decide(("Price action", 1), (b.BEHAVIOUR, 1), (b.MARKET, 1)), "WATCH")  # no tested model
        self.assertEqual(self.decide((b.ML_DAILY, 1, 0.5), ("Price action", 1), (b.BEHAVIOUR, 1), (b.MARKET, 1)), "WATCH")  # next-days only

    def test_behaviour_or_sector_against_it_stops_the_order(self):
        b = self.b
        base = [("Candle forecast", 1), ("ML model", 1), ("Reliable patterns", 1), ("Price action", 1), (b.MARKET, 1)]
        self.assertEqual(self.decide(*base), "BUY")
        self.assertEqual(self.decide(*base, (b.BEHAVIOUR, -1, 0.5)), "WATCH")
        self.assertEqual(self.decide(*base, (b.SECTOR, -1)), "WATCH")

    def test_against_nifty_takes_four_stock_signals(self):
        b = self.b
        three = [("Candle forecast", 1), ("ML model", 1), ("Price action", 1), (b.SECTOR, 1)]
        self.assertEqual(self.decide(*three, (b.MARKET, -1)), "WATCH")
        self.assertEqual(self.decide(*three, ("Reliable patterns", 1), (b.MARKET, -1)), "BUY")

    def test_conviction_wants_two_models_and_behaviour(self):
        b = self.b
        medium = b.decide([signal("Candle forecast", 1), signal("ML model", 1), signal(b.MARKET, 1)])
        higher = b.decide([signal("Candle forecast", 1), signal("ML model", 1), signal(b.BEHAVIOUR, 1)])
        self.assertEqual((medium[2], higher[2]), ("Medium", "Higher"))

    def test_stop_is_widened_by_the_usual_session_range_within_the_limits(self):
        s = {"price": 100.0, "forecast": {"band": [[0, 99.7, 100.3]]}}
        self.assertAlmostEqual(self.b.plan_for(s, 1, None)["risk_pct"], 0.4)                 # floor
        self.assertAlmostEqual(self.b.plan_for(s, 1, 3.0)["risk_pct"], 0.75)                 # a quarter of 3%
        self.assertAlmostEqual(self.b.plan_for(s, -1, 20.0)["risk_pct"], 1.2)                # ceiling
        self.assertLess(self.b.plan_for(s, -1, 3.0)["target"], 100)
        self.assertIsNone(self.b.plan_for({"price": 100.0}, 1, 3.0))


class ClaudeBoard2TodayTests(TestCase):
    """Reading the day: indices, world, session, expiry."""

    def setUp(self):
        from . import claude_board2
        self.b = claude_board2

    def test_index_direction_needs_the_close_and_the_open_to_agree(self):
        row = self.b._index_row
        self.assertEqual(row("A", 101, 100, 1.5)["dir"], 1)
        self.assertEqual(row("A", 99, 100, 1.5)["dir"], 0)              # gapped up, then sold
        self.assertEqual(row("A", 99, 100, -1.5)["dir"], -1)
        self.assertEqual(row("A", 100.1, 100, 0.1)["dir"], 0)           # flat
        self.assertEqual(row("A", None, 100, None)["dir"], 0)

    def test_breadth_counts_broad_and_sector_indices_but_not_vix_or_themes(self):
        def rows(*pcts):
            return {f"I{i}": {"name": f"I{i}", "pct": p, "group": "SECTORAL INDICES"} for i, p in enumerate(pcts)}
        indices = {**rows(1, 1, 1, -1), "INDIA VIX": {"name": "INDIA VIX", "pct": 9.0, "group": "SECTORAL INDICES"},
                   "X": {"name": "X", "pct": -5, "group": "THEMATIC INDICES"}}
        b = self.b.breadth(indices)
        self.assertEqual((b["dir"], b["up"], b["down"], b["total"], b["vix_pct"]), (1, 3, 1, 4, 9.0))
        self.assertEqual(self.b.breadth(rows(1, -1))["dir"], 0)
        self.assertEqual(self.b.breadth({})["dir"], 0)

    def test_world_direction_is_a_clear_majority(self):
        def world(*pcts):
            return self.b.world([{"short": str(i), "pct": p} for i, p in enumerate(pcts)])
        self.assertEqual(world(1, 1, 1, -1, 0)["dir"], 1)
        self.assertEqual(world(-1, -1, -1, 1)["dir"], -1)
        self.assertEqual(world(1, 1, -1, -1, 0)["dir"], 0)
        self.assertEqual(self.b.world([{"short": "A", "pct": 1, "stale": True}])["total"], 0)

    def test_session_follows_the_clock(self):
        at = lambda h, m: datetime(2026, 11, 5, h, m)
        self.assertEqual([self.b.current_session(at(*t)) for t in ((8, 0), (11, 29), (11, 30), (13, 30), (16, 0))],
                         ["morning", "morning", "afternoon", "evening", "evening"])

    def test_session_move_starts_at_the_session_open(self):
        bars = [[f"2026-11-05T{h}:00+05:30", 200, 0, 0, c] for h, c in (("11:25", 300), ("11:30", 200), ("11:35", 201), ("11:40", 202))]
        self.assertAlmostEqual(self.b.session_move({"candles": bars}, "afternoon"), 1.0)
        self.assertIsNone(self.b.session_move({"candles": bars[:3]}, "afternoon"))           # too early to say
        self.assertIsNone(self.b.session_move({}, "morning"))

    def test_today_expiries_from_the_last_files(self):
        from datetime import date
        today = date(2026, 11, 5)                                     # a Thursday
        self.b._expiry_day.clear()
        self.addCleanup(self.b._expiry_day.clear)
        nse = {"NIFTY": {date(2026, 11, 10)}, "BANKNIFTY": {today, date(2026, 12, 29)}}
        bse = {"SENSEX": {today, date(2026, 11, 12)}}
        with mock.patch("market.index_data.nse_expiries", return_value=nse) as n, \
             mock.patch("market.index_data.bse_expiries", return_value=bse):
            self.assertEqual(self.b.today_expiries(today), {"BANKNIFTY": "monthly", "SENSEX": "weekly"})
            self.b.today_expiries(today)
        self.assertEqual(n.call_count, 1)                             # once a day
        self.assertEqual(n.call_args[0][0], date(2026, 11, 4))

    def test_a_source_that_fails_leaves_the_rest(self):
        boom = RuntimeError("down")
        with mock.patch.object(self.b, "read_indices", side_effect=boom), mock.patch("market.indices.index_chart", side_effect=boom), \
             mock.patch("market.indices.global_indices", side_effect=boom), mock.patch.object(self.b, "today_expiries", side_effect=boom):
            ctx = self.b.read_context(None, datetime(2026, 11, 5, 10, 0, tzinfo=self.b.IST))
        self.assertEqual((ctx["indices"], ctx["expiry"], ctx["breadth"]["total"], ctx["world"]["total"], ctx["session"]),
                         ({}, {}, 0, 0, "morning"))
        self.assertEqual(self.b.backdrop_signals(ctx), [])
        self.assertIn("Thursday, Morning", self.b.summary(ctx, None))


class ClaudeBoard2BehaviourTests(TestCase):
    """Stock Behaviour with today's filters: only what repeats gets a vote."""

    def market(self, habit):
        """80 stocks that follow nothing, the first 25 (HOT among them) leaning up by `habit`."""
        import numpy as np
        from market.tests import behaviour_matrix
        rng = np.random.default_rng(3)
        data = behaviour_matrix(560)
        n, m = len(data["days"]), 80
        noise = rng.normal(size=(n, m)).astype(np.float32)
        noise[:, :25] += habit
        names = [f"S{i}" for i in range(m)]
        names[0] = "HOT"
        data.update(codes=np.array([f"{x}.NS" for x in names]), names=np.array(names), sectors=np.array(["Technology"] * m),
                    turnover=np.full(m, 100, dtype=np.float32), o2c=noise, c2c=noise, rng=np.abs(noise) + 1)
        return data

    def condition(self, data, **filters):
        from market.stock_behaviour import Engine, Filters
        from .claude_board2 import Condition
        return Condition("x", "These days", Engine(data, Filters(period="all", **filters)))

    def test_a_habit_that_repeats_votes(self):
        c = self.condition(self.market(1.0))
        self.assertTrue(c.repeats())
        sign, strength, words = c.vote("HOT.NS")
        self.assertEqual((sign, strength), (1, 1.0))
        self.assertIn("typical stock", words)
        self.assertIsNone(c.vote("S70.NS"))                          # a stock with no habit
        self.assertIsNone(c.vote("NOPE.NS"))
        self.assertIn(("HOT.NS", 1), c.candidates())

    def test_noise_gets_no_vote_even_if_a_stock_looks_good(self):
        c = self.condition(self.market(0.0))
        self.assertFalse(c.repeats())
        self.assertTrue(all(c.vote(code) is None for code in c.position))
        self.assertEqual(c.candidates(), [])

    def test_too_few_days_is_no_condition(self):
        c = self.condition(self.market(1.0), weekday=0, expiry="monthly", trend="up", move=50.0)
        self.assertEqual((c.sides, c.vote("HOT.NS")), ({}, None))

    def test_signal_says_what_did_not_repeat(self):
        from . import claude_board2 as b
        good, noise = self.condition(self.market(1.0)), self.condition(self.market(0.0))
        s = b.behaviour_signal("HOT.NS", [good, noise])
        self.assertEqual((s["dir"], s["w"], s["cred"]), (1, 1.0, True))
        self.assertIn("didn't repeat: These days", s["detail"])
        s = b.behaviour_signal("S70.NS", [good])
        self.assertEqual((s["dir"], s["w"], s["cred"]), (0, 0.0, False))

    def test_todays_filters_become_conditions(self):
        from market.tests import behaviour_matrix
        from . import claude_board2 as b
        ctx = {"session": "afternoon", "weekday": 0, "session_move": 0.5, "expiry": {"NIFTY": "weekly"},
               "indices": {"NIFTY 50": {"o2c": -1.0}, "NIFTY IT": {"o2c": 0.05}}}
        common, home = b.conditions(behaviour_matrix(200), ctx)
        self.assertEqual([c.label for c in common], ["NIFTY down days", "Afternoon with NIFTY up", "Mondays with NIFTY down",
                                                     "Monday afternoons", "NIFTY weekly expiries"])
        self.assertEqual([c.engine.f.trend for c in common[:3]], ["down", "up", "down"])
        self.assertEqual(common[1].engine.f.session, "afternoon")
        self.assertEqual(common[2].engine.f.weekday, 0)
        self.assertEqual(list(home), ["NIFTYIT"])
        self.assertEqual(home["NIFTYIT"].engine.f.trend, "any")     # NIFTY IT is flat from its open
        self.assertIn("MONDAY.NS", b.session_ranges(behaviour_matrix(200), "morning"))


class ClaudeBoard2SignalTests(TestCase):
    def setUp(self):
        from . import claude_board2
        self.b = claude_board2
        self.ctx = {"indices": {"NIFTY BANK": {"name": "NIFTY BANK", "pct": 1.0, "dir": 1},
                                "NIFTY FINANCIAL SERVICES": {"name": "NIFTY FINANCIAL SERVICES", "pct": 0.8, "dir": 1},
                                "NIFTY METAL": {"name": "NIFTY METAL", "pct": -1.0, "dir": -1}},
                    "breadth": {"dir": 1, "up": 8, "down": 1, "total": 10, "vix_pct": -2.0},
                    "world": {"dir": -1, "up": 0, "down": 3, "total": 5, "detail": "x"}}

    def test_sector_signal_uses_the_stocks_own_indices_else_its_sector(self):
        member = lambda code: {"indices": [{"name": "NIFTY FIN SERVICE", "group": "sector"}, {"name": "NIFTY BANK", "group": "sector"},
                                           {"name": "NIFTY 50", "group": "broad"}]}
        s = self.b.sector_signal({"code": "A.NS"}, self.ctx, member)
        self.assertEqual((s["dir"], s["w"]), (1, 1.0))
        self.assertIn("NIFTY BANK", s["detail"])
        none = lambda code: {"indices": []}
        self.assertEqual(self.b.sector_signal({"code": "A.NS", "sector": "Basic Materials"}, self.ctx, none)["dir"], -1)
        self.assertIsNone(self.b.sector_signal({"code": "A.NS", "sector": "Unknown"}, self.ctx, none))

    def test_home_index_is_the_narrowest(self):
        member = lambda *names: (lambda code: {"indices": [{"name": n} for n in names]})
        self.assertEqual(self.b.home_index("A.NS", member("NIFTY 50", "NIFTY FIN SERVICE", "NIFTY BANK")), "BANKNIFTY")
        self.assertEqual(self.b.home_index("A.NS", member("NIFTY SMALLCAP 100")), "NIFTYSML100")
        self.assertIsNone(self.b.home_index("A.NS", member("NIFTY 50")))

    def test_ml_daily_counts_only_with_an_edge_and_at_half_weight(self):
        f = lambda edge: [{"in": "1 day", "direction": "up", "ret_pct": 0.5, "edge": edge}]
        self.assertIsNone(self.b.ml_daily_signal({"ml_daily": f(False)}))
        self.assertIsNone(self.b.ml_daily_signal({}))
        self.assertEqual(self.b.ml_daily_signal({"ml_daily": f(True)})["w"], 0.5)

    def test_open_interest_reads_with_the_price(self):
        oi = {"A": {"avgInOI": 40.0}, "B": {"avgInOI": 3.0}}
        stock = lambda code, price: {"code": code, "price": price, "levels": {"pdc": 100.0}}
        self.assertEqual(self.b.oi_signal(stock("A.NS", 101), oi)["dir"], 1)
        self.assertIn("short build-up", self.b.oi_signal(stock("A.NS", 99), oi)["detail"])
        self.assertIsNone(self.b.oi_signal(stock("B.NS", 101), oi))
        self.assertIsNone(self.b.oi_signal(stock("C.NS", 101), oi))

    def test_rescore_leaves_a_skipped_stock_alone(self):
        s = {"code": "A.NS", "decision": "SKIP"}
        self.assertEqual(self.b.rescore(s, self.ctx, []), {"code": "A.NS", "decision": "SKIP"})

    def test_rescore_decides_again_with_the_new_signals(self):
        s = {"code": "A.NS", "price": 100.0, "sector": "Basic Materials", "decision": "BUY", "forecast": {"band": [[0, 99.0, 101.0]]},
             "signals": [signal("Candle forecast", 1), signal("Price action", 1), signal("ML model", 1), signal(self.b.MARKET, 1)]}
        with mock.patch("market.index_membership.indices_for", return_value={"indices": []}):
            self.b.rescore(s, self.ctx, [])
        self.assertEqual(s["decision"], "WATCH")                     # metals are falling: the first version said BUY
        self.assertIsNone(s["plan"])
        self.assertEqual([x["name"] for x in s["signals"]][-3:], [self.b.SECTOR, self.b.ALL_INDICES, self.b.GLOBAL])


class ClaudeOrders2PageTests(ClaudeOrdersTests):
    """The 2.0 page: the first version's page and order handling, on its own list."""

    def setUp(self):
        super().setUp()
        from . import claude_orders2
        self.tmp2 = self.tmp.with_name("claude_orders2.json")
        patcher = mock.patch.object(claude_orders2, "ORDERS_FILE", self.tmp2)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write2(self):
        self.tmp2.write_text(json.dumps({
            "generated": "2026-10-19T10:30:00+05:30", "session": svc.today_ist().isoformat(), "source": "Signal board 2.0",
            "orders": [{"id": "v2-20261019-1030-TST-BUY", "code": "TST", "name": "Test Co", "sector": "", "side": "BUY",
                        "product": "INTRADAY", "conviction": "Higher", "score": 0.8, "price_at": 100.0,
                        "price_time": "2026-10-19T10:20:00+05:30", "stop": 99.0, "target": 101.5, "stop_pct": 1.0,
                        "target_pct": 1.5, "reasons": [], "news": [], "square_off_by": "15:20"}]}), encoding="utf-8")

    def place2(self):
        with mock.patch.object(svc, "get_market_status", return_value="OPEN"), \
             mock.patch.object(svc, "fetch_live_today", return_value=live_frame(1, close=100.2)):
            return self.post(reverse("paper_trading:claude_orders2_place"), {"id": "v2-20261019-1030-TST-BUY", "quantity": 10})

    def test_secure_exit_is_gone_from_both_pages(self):
        for name in ("claude_orders", "claude_orders2"):
            page = self.client.get(reverse(f"paper_trading:{name}")).content.decode().lower()
            for word in ("secure exit", "secureexit", "secure_exit", "exit all", "exitall"):
                self.assertNotIn(word, page)
            self.assertIn('id="bulkmsg"', page)

    def test_each_page_shows_its_own_list_and_uses_its_own_urls(self):
        self.write()
        self.write2()
        one, two = self.client.get(reverse("paper_trading:claude_orders")), self.client.get(reverse("paper_trading:claude_orders2"))
        self.assertContains(two, "<title>Claude Orders 2.0</title>")
        self.assertContains(two, reverse("paper_trading:claude_orders2_refresh"))
        self.assertContains(two, reverse("paper_trading:claude_orders2_place"))
        self.assertContains(two, "v2-20261019-1030-TST-BUY")
        self.assertContains(one, "<title>Claude Orders</title>")
        self.assertNotContains(one, reverse("paper_trading:claude_orders2_refresh"))
        self.assertNotContains(one, "v2-20261019-1030-TST-BUY")
        self.assertEqual(self.client.get(reverse("paper_trading:claude_orders2_state")).json()["source"], "Signal board 2.0")

    def test_both_sit_in_the_orders_dropdown(self):
        page = self.client.get(reverse("paper_trading:claude_orders2")).content.decode()
        menu = page[page.index(">Orders\n"):]
        menu = menu[:menu.index("</nav>")]
        self.assertIn(reverse("paper_trading:claude_orders"), menu)
        self.assertIn(reverse("paper_trading:claude_orders2"), menu)
        self.assertIn('class="active"><b>Claude Orders 2.0', menu)
        self.assertNotIn('class="active"><b>Claude Orders<', menu)
        first = self.client.get(reverse("paper_trading:claude_orders")).content.decode()
        self.assertIn('class="active"><b>Claude Orders<', first)
        self.assertNotIn('class="active"><b>Claude Orders 2.0', first)

    def test_placing_from_either_page_blocks_the_other(self):
        self.write()
        self.write2()
        r = self.place2()
        self.assertEqual(r.status_code, 200, r.content)
        o = PaperOrder.objects.get()
        self.assertEqual((o.account.owner_key, o.side, o.quantity, o.reason), ("book:2", "BUY", 10, "claude"))
        self.assertEqual(self.place().status_code, 409)
        self.assertEqual(PaperOrder.objects.count(), 1)
        st = self.client.get(reverse("paper_trading:claude_orders2_state")).json()
        self.assertEqual(st["orders"][0]["placed"]["quantity"], 10)

    def test_a_first_version_id_is_unknown_here(self):
        self.write()
        self.write2()
        r = self.post(reverse("paper_trading:claude_orders2_place"), {"id": "20261019-1024-TST-BUY", "quantity": 1})
        self.assertEqual(r.status_code, 404)

    def test_refresh_runs_apart_from_the_first_version(self):
        from . import claude_board, claude_board2
        fake = {"generated": "2026-11-05T10:00:00+05:30", "session": svc.today_ist().isoformat(), "orders": [],
                "source": "Signal board 2.0"}
        with mock.patch.object(claude_board2, "build", return_value=fake), mock.patch.object(claude_board, "archive"), \
             mock.patch("django.db.connection.close"):
            claude_board._set(running=True)                          # the first version busy doesn't block 2.0
            self.addCleanup(claude_board._set, running=False)
            self.assertEqual(self.client.post(reverse("paper_trading:claude_orders2_refresh")).status_code, 202)
            for _ in range(200):
                if not claude_board2.progress()["running"]:
                    break
                threading.Event().wait(0.01)
        self.assertEqual(json.loads(self.tmp2.read_text(encoding="utf-8"))["source"], "Signal board 2.0")
        self.assertFalse(self.tmp.exists())                          # the first version's list wasn't touched

    def test_build_keeps_the_order_format_and_marks_its_ids(self):
        from . import claude_board, claude_board2 as b
        Company.objects.update_or_create(code="TST.NS", defaults={"name": "Test Co", "sector": "Technology"})

        def gathered(code, why):
            return {"code": code, "name": "Test Co", "sector": "Technology", "why": why, "price": 100.0, "asof": "2026-11-05T10:00:00+05:30",
                    "forecast": {"band": [[0, 99.0, 101.0]], "news_items": []}, "decision": "BUY", "conviction": "Medium", "score": 1,
                    "signals": [signal("Candle forecast", 1), signal("ML model", 1), signal("Price action", 1)]}

        ctx = {"day": svc.today_ist(), "weekday": 3, "session": "morning", "indices": {}, "session_move": None, "expiry": {},
               "breadth": {"dir": 1, "up": 8, "down": 1, "total": 10, "vix_pct": 1.0},
               "world": {"dir": 0, "up": 1, "down": 1, "total": 5, "detail": "x"}}
        live = {"gainers": [{"x": 1}], "open_intrest": [{"symbol": "TST", "avgInOI": 30.0}, {"symbol": "FINNIFTY", "avgInOI": 30.0}]}
        with mock.patch("market.views.live_snapshot", return_value=live), mock.patch.object(claude_board, "market_direction", return_value=None), \
             mock.patch.object(b, "read_context", return_value=ctx), mock.patch("market.behaviour_data.load", return_value=None), \
             mock.patch.object(claude_board, "candidates", return_value=([], {})), mock.patch.object(claude_board, "gather", side_effect=gathered), \
             mock.patch.object(claude_board, "score", side_effect=lambda s, *a: s), mock.patch.object(b, "home_index", return_value=None), \
             mock.patch("market.index_membership.indices_for", return_value={"indices": []}), mock.patch("django.db.connection.close"):
            out = b.build()
        self.assertEqual((out["checked"], out["counts"]["BUY"], len(out["orders"])), (1, 1, 1))     # FINNIFTY is not a company
        order = out["orders"][0]
        self.assertTrue(order["id"].startswith("v2-") and order["id"].endswith("-TST-BUY"))
        for key in ("code", "side", "conviction", "price_at", "stop", "target", "stop_pct", "target_pct", "reasons", "news", "square_off_by"):
            self.assertIn(key, order)
        self.assertIn("Stock Behaviour data is not built", out["method"])

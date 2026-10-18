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

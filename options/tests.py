from datetime import date, datetime, timedelta
from datetime import time as time_
from decimal import Decimal
from unittest import mock

from django.core.cache import cache
from django.test import TestCase
from django.urls import reverse

from prediction.market_time import IST

from . import chain as ch
from . import services as svc
from .models import OptionOrder, OptionPosition

D = Decimal
EXP = date(2026, 10, 27)
MON_10 = datetime(2026, 10, 26, 10, 0, tzinfo=IST)


def q(ltp, bid=None, ask=None, iv=15.0):
    return {"ltp": ltp, "bid": ltp - 0.5 if bid is None else bid, "ask": ltp + 0.5 if ask is None else ask,
            "bid_qty": 100, "ask_qty": 100, "oi": 1000, "chg_oi": 0, "vol": 10, "iv": iv, "chg": 0, "pchg": 0}


class OptionsBase(TestCase):
    """NIFTY, lot 65, spot 22600; quotes and market status are set per test."""

    def setUp(self):
        cache.clear()
        self.quotes = {}
        self.spot = 22600.0
        self.status = "OPEN"
        patches = [
            mock.patch.object(ch, "quote", side_effect=lambda s, e, k, t, max_age=3: (self.quotes.get((float(k), t)), self.spot, {})),
            mock.patch.object(ch, "lot_size", return_value=65),
            mock.patch.object(svc, "get_market_status", side_effect=lambda now=None: self.status),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.acct = svc.get_account()

    def order(self, side, lots=1, strike=22600, t="CE", now=MON_10, **kw):
        return svc.place_order(self.acct, symbol="NIFTY", expiry=EXP, strike=strike, option_type=t, side=side,
                               lots=lots, now=now, **kw)

    def pos(self, strike=22600, t="CE", product="NRML"):
        return OptionPosition.objects.get(account=self.acct, strike=strike, option_type=t, product=product)

    def bal(self):
        self.acct.refresh_from_db()
        return self.acct.balance


class ChargesAndMarginTests(TestCase):
    def test_zerodha_option_charges(self):
        buy, sell = svc.charges("BUY", 6500), svc.charges("SELL", 6500)
        self.assertEqual(buy["brokerage"], D("20.00"))
        self.assertEqual(buy["stt"], D("0.00"))
        self.assertEqual(sell["stt"], D("9.75"))                       # 0.15 % of the sell premium
        self.assertEqual(buy["stamp"], D("0.20"))                      # 0.003 % on the buy side only
        self.assertEqual(sell["stamp"], D("0.00"))
        self.assertEqual(buy["exchange"], D("2.31"))                   # 0.03553 %
        self.assertEqual(buy["gst"], D("4.02"))                        # 18 % of brokerage + exchange + SEBI
        self.assertEqual(svc.charges("SELL", 0, exercise_value=10000)["stt"], D("15.00"))

    def test_short_margin_is_lower_out_of_the_money(self):
        atm = svc.short_margin("NIFTY", 22600, 22600, "CE", 65)
        otm = svc.short_margin("NIFTY", 22600, 23600, "CE", 65)
        self.assertAlmostEqual(float(atm), 22600 * 65 * 0.085, delta=1)
        self.assertLess(otm, atm)
        self.assertGreater(svc.short_margin("RELIANCE", 1190, 1190, "PE", 500) / (1190 * 500), D("0.18"))


class OrderTests(OptionsBase):
    def test_market_buy_fills_at_the_ask_and_pays_premium_plus_charges(self):
        self.quotes[(22600.0, "CE")] = q(130, bid=129.5, ask=131)
        start = self.bal()
        o = self.order("BUY", lots=2)
        self.assertEqual((o.status, o.price, o.quantity), ("COMPLETE", D("131"), 130))
        p = self.pos()
        self.assertEqual((p.quantity, p.avg_price, p.blocked_margin), (130, D("131"), 0))
        self.assertEqual(self.bal(), start - D("131") * 130 - o.charges)

    def test_partial_then_full_exit_books_pnl(self):
        self.quotes[(22600.0, "CE")] = q(100, bid=99, ask=100)
        start = self.bal()
        b = self.order("BUY", lots=2)
        self.quotes[(22600.0, "CE")] = q(120, bid=120, ask=121)
        s1 = self.order("SELL", lots=1)
        self.assertEqual(s1.realized_pnl, D("1300.00"))                # (120 - 100) x 65
        self.assertEqual(self.pos().quantity, 65)
        s2 = svc.exit_position(self.acct, self.pos().pk, now=MON_10)
        self.assertEqual(self.pos().quantity, 0)
        self.assertEqual(self.pos().realized_pnl, D("2600"))
        self.assertEqual(self.bal(), start + D("2600") - b.charges - s1.charges - s2.charges)
        with self.assertRaises(svc.AlreadyClosed):
            svc.exit_position(self.acct, self.pos().pk, now=MON_10)

    def test_writing_blocks_margin_and_buying_back_releases_it(self):
        self.quotes[(22600.0, "PE")] = q(140, bid=140, ask=141)
        start = self.bal()
        s = self.order("SELL", t="PE")
        p = self.pos(t="PE")
        self.assertEqual(p.quantity, -65)
        margin = svc.short_margin("NIFTY", 22600, 22600, "PE", 65)
        self.assertEqual(p.blocked_margin, margin)
        self.assertEqual(self.bal(), start + D("140") * 65 - s.charges)            # premium credited
        self.assertEqual(svc.available(self.acct), self.bal() - margin)
        self.quotes[(22600.0, "PE")] = q(100, bid=99, ask=100)
        b = svc.exit_position(self.acct, p.pk, now=MON_10)
        self.assertEqual(b.side, "BUY")
        self.assertEqual(b.realized_pnl, D("2600.00"))                             # (140 - 100) x 65
        self.assertEqual(self.pos(t="PE").blocked_margin, 0)
        self.assertEqual(self.bal(), start + D("2600") - s.charges - b.charges)

    def test_insufficient_funds_and_bad_input_are_refused(self):
        self.quotes[(22600.0, "CE")] = q(130)
        with self.assertRaisesMessage(svc.TradeError, "Insufficient funds"):
            self.order("BUY", lots=100)                                            # 100 x 65 x 130 = 8.45 L
        with self.assertRaisesMessage(svc.TradeError, "Lots must be between"):
            self.order("BUY", lots=0)
        with self.assertRaisesMessage(svc.TradeError, "Stop loss must be below"):
            self.order("BUY", stop_loss=140)
        with self.assertRaisesMessage(svc.TradeError, "Target must be below"):
            self.order("SELL", target=140)
        with self.assertRaisesMessage(svc.TradeError, "expired"):
            self.order("BUY", now=datetime(2026, 10, 27, 15, 31, tzinfo=IST))
        self.assertFalse(OptionOrder.objects.exists())

    def test_market_closed_market_refused_limit_waits_and_fills_at_open(self):
        self.status = "CLOSED"
        self.quotes[(22600.0, "CE")] = q(130, bid=129, ask=131)
        with self.assertRaisesMessage(svc.TradeError, "market is closed"):
            self.order("BUY")
        o = self.order("BUY", order_type="LIMIT", limit_price=125, stop_loss=100)
        self.assertEqual(o.status, "PENDING")
        self.assertEqual(svc.process_pending(self.acct, MON_10), [])                 # still closed
        self.status = "OPEN"
        self.assertEqual(svc.process_pending(self.acct, MON_10), [])                 # ask 131 > 125
        self.quotes[(22600.0, "CE")] = q(124, bid=123.5, ask=124.5)
        filled = svc.process_pending(self.acct, MON_10)
        self.assertEqual([x.price for x in filled], [D("124.5")])                    # at the ask, inside the limit
        self.assertEqual(self.pos().stop_loss, D("100"))                             # level from the order

    def test_cancel_pending(self):
        self.quotes[(22600.0, "CE")] = q(130)
        o = self.order("BUY", order_type="LIMIT", limit_price=100)
        svc.cancel_order(self.acct, o.pk)
        with self.assertRaisesMessage(svc.TradeError, "already cancelled"):
            svc.cancel_order(self.acct, o.pk)


class WatcherTests(OptionsBase):
    def test_stop_loss_on_premium_exits_at_the_bid(self):
        self.quotes[(22600.0, "CE")] = q(130, ask=130)
        self.order("BUY", stop_loss=110, target=170)
        self.quotes[(22600.0, "CE")] = q(115)
        self.assertEqual(svc.check_levels(self.acct, MON_10), [])
        self.quotes[(22600.0, "CE")] = q(109, bid=108.5)
        [x] = svc.check_levels(self.acct, MON_10)
        self.assertEqual((x.reason, x.price, x.realized_pnl), ("stop_loss", D("108.5"), D("-1397.50")))

    def test_short_target_and_mis_squareoff(self):
        self.quotes[(23000.0, "CE")] = q(50, bid=50)
        self.order("SELL", strike=23000, target=30)
        self.quotes[(23000.0, "CE")] = q(29, ask=29.5)
        [x] = svc.check_levels(self.acct, MON_10)
        self.assertEqual((x.reason, x.side, x.price), ("target", "BUY", D("29.5")))
        self.quotes[(22600.0, "PE")] = q(140)
        self.order("BUY", t="PE", product="MIS")
        self.assertEqual(svc.squareoff_mis(self.acct, MON_10), [])
        with self.assertRaisesMessage(svc.TradeError, "MIS orders are closed"):
            self.order("BUY", t="PE", product="MIS", now=datetime(2026, 10, 26, 15, 21, tzinfo=IST))
        [sq] = svc.squareoff_mis(self.acct, datetime(2026, 10, 26, 15, 20, tzinfo=IST))
        self.assertEqual((sq.reason, self.pos(t="PE", product="MIS").quantity), ("squareoff", 0))

    def test_expiry_settles_at_intrinsic_value(self):
        self.quotes[(22500.0, "CE")] = q(150, ask=150)
        self.quotes[(22800.0, "CE")] = q(40, bid=40)
        long_ = self.order("BUY", strike=22500)
        short = self.order("SELL", strike=22800)
        before = self.bal()
        price_of = mock.Mock(return_value=22700.0)
        self.assertEqual(svc.settle_expired(self.acct, datetime(2026, 10, 27, 15, 29, tzinfo=IST), price_of), [])
        out = svc.settle_expired(self.acct, datetime(2026, 10, 27, 15, 31, tzinfo=IST), price_of)
        by = {o.strike: o for o in out}
        ex = by[D("22500")]
        self.assertEqual((ex.price, ex.realized_pnl, ex.reason), (D("200.00"), D("3250.00"), "expiry"))   # (200-150) x 65
        self.assertEqual(ex.charges_detail["stt"], 19.5)                                   # 0.15 % of 13,000 exercised
        wl = by[D("22800")]
        self.assertEqual((wl.price, wl.realized_pnl, wl.charges), (D("0.00"), D("2600.00"), D("0")))        # expired worthless
        self.assertEqual(self.bal(), before + D("200") * 65 - ex.charges)
        self.assertEqual(svc.blocked_margin(self.acct), 0)
        self.assertEqual(OptionPosition.objects.exclude(quantity=0).count(), 0)
        self.assertEqual(svc.settle_expired(self.acct, datetime(2026, 10, 28, 10, 0, tzinfo=IST), price_of), [])


class ApiTests(OptionsBase):
    def test_page_place_duplicate_and_state(self):
        self.assertEqual(self.client.get(reverse("options:page")).status_code, 200)
        self.quotes[(22600.0, "CE")] = q(130)
        body = {"symbol": "NIFTY", "expiry": EXP.isoformat(), "strike": 22600, "option_type": "CE", "side": "BUY",
                "lots": 1, "product": "NRML", "order_type": "MARKET", "client_order_id": "abc"}
        with mock.patch.object(svc, "now_ist", return_value=MON_10):
            r = self.client.post(reverse("options:place"), body, content_type="application/json")
            self.assertEqual(r.status_code, 200, r.content)
            self.assertEqual(r.json()["order"]["status"], "COMPLETE")
            dup = self.client.post(reverse("options:place"), body, content_type="application/json")
            self.assertEqual(dup.status_code, 409)
            pv = self.client.post(reverse("options:preview"), {**body, "side": "SELL"}, content_type="application/json").json()
            self.assertEqual(pv["closing"], 65)
            self.assertEqual(pv["margin"], 0)
            with mock.patch.object(ch, "chain", return_value={"spot": 22600.0, "rows": [{"strike": 22600.0, "CE": q(140), "PE": None}]}):
                st = self.client.get(reverse("options:state")).json()
        [p] = st["positions"]
        self.assertEqual((p["quantity"], p["ltp"], p["mtm"]), (65, 140, (140 - 130.5) * 65))
        self.assertIsNotNone(p["greeks"]["delta"])
        self.assertEqual(len(st["history"]), 1)


class ReplayTests(TestCase):
    """Replay on a made-up Wed 5 Mar 2025 (NIFTY 22300 CE / PE, expiry Thu 6 Mar) and the expiry day after it."""

    D1, D2, X = date(2025, 3, 5), date(2025, 3, 6), "2025-03-06"

    def bhav(self, day):
        from . import history as h
        if day == self.D1:
            rows = [[22300.0, "CE", 100.0, 180.0, 60.0, 150.0, 90.0, 150.0, 5000, 100, 1000, True],
                    [22300.0, "PE", 110.0, 150.0, 40.0, 50.0, 120.0, 50.0, 5000, 100, 1000, True],
                    [22900.0, "CE", 0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 10, 0, 0, True]]          # no trades that day
            under = 22300.0
        elif day == self.D2:
            rows = [[22300.0, "CE", 140.0, 200.0, 90.0, 100.0, 150.0, 100.0, 4000, -1000, 900, True],
                    [22300.0, "PE", 40.0, 60.0, 0.05, 0.05, 50.0, 0.05, 4000, -1000, 900, True]]
            under = 22400.0
        else:
            raise h.HistoryError("no file")
        return {"day": day.isoformat(), "contracts": {"NIFTY": {self.X: rows}}, "underlying": {"NIFTY": under},
                "lots": {"NIFTY": {self.X: 75}}}

    def bars(self, symbol, day):
        start, end = (22200.0, 22300.0) if day == self.D1 else (22300.0, 22400.0)
        out, t = [], datetime(2025, 1, 1, 9, 15)
        for i in range(75):
            o = start + (end - start) * i / 75
            c = start + (end - start) * (i + 1) / 75
            out.append(((t + timedelta(minutes=5 * i)).time(), o, max(o, c) + 2, min(o, c) - 2, c))
        return 5, out

    def setUp(self):
        from . import history as h
        self.h = h
        h._paths.clear(); h._ivs.clear(); h._mem.clear(); h._bar_cache.clear()
        for p in (mock.patch.object(h, "bhavcopy", side_effect=self.bhav), mock.patch.object(h, "_bars", side_effect=self.bars)):
            p.start()
            self.addCleanup(p.stop)
        self.acct = svc.get_account(replay=True)
        self.live = svc.get_account()

    def ce(self, hm):
        now = datetime.combine(self.D1, datetime.strptime(hm, "%H:%M").time(), tzinfo=IST)
        q, spot = self.h.quote("NIFTY", date(2025, 3, 6), 22300, "CE", now)
        return q, spot

    def test_prices_are_real_at_open_and_close_estimated_inside_the_range_between(self):
        self.assertEqual(self.ce("09:15")[0]["ltp"], 100.0)
        self.assertEqual(self.ce("15:30")[0]["ltp"], 150.0)
        mids = [self.ce(t)[0]["ltp"] for t in ("10:00", "11:30", "13:00", "14:45")]
        self.assertTrue(all(60.0 <= p <= 180.0 for p in mids), mids)
        self.assertLess(mids[0], mids[-1])                                  # underlying rises through the day
        self.assertAlmostEqual(self.ce("12:00")[1], 22200 + 100 * 165 / 375, places=1)
        q, _ = self.h.quote("NIFTY", date(2025, 3, 6), 22900, "CE", datetime.combine(self.D1, time_(12), tzinfo=IST))
        self.assertIsNone(q)                                                # did not trade -> not tradable

    def test_replay_session_clock_orders_levels_and_expiry(self):
        with self.assertRaisesMessage(svc.TradeError, "Pick a date"):
            svc.place_order(self.acct, symbol="NIFTY", expiry=date(2025, 3, 6), strike=22300, option_type="CE", side="BUY", lots=1)
        svc.replay_start(self.acct, self.D1)
        o = svc.place_order(self.acct, symbol="NIFTY", expiry=date(2025, 3, 6), strike=22300, option_type="CE",
                            side="BUY", lots=2, target=170)
        self.assertEqual((o.status, o.price, o.quantity), ("COMPLETE", D("100"), 150))      # the real open
        self.assertEqual(svc.to_ist(o.market_time).date(), self.D1)
        with self.assertRaisesMessage(svc.TradeError, "did not trade"):
            svc.place_order(self.acct, symbol="NIFTY", expiry=date(2025, 3, 6), strike=22900, option_type="CE", side="BUY", lots=1)
        with self.assertRaisesMessage(svc.TradeError, "open replay positions"):
            svc.replay_start(self.acct, self.D2)
        ev = svc.replay_advance(self.acct, "60")
        self.assertEqual(self.acct.replay_time.strftime("%H:%M"), "10:15")
        self.assertEqual(ev["exits"], [])
        ev = svc.replay_advance(self.acct, "close")
        self.assertEqual(self.acct.replay_time.strftime("%H:%M"), "15:30")
        p = OptionPosition.objects.get(account=self.acct)
        self.assertEqual(p.quantity, 150)                                   # 170 target never reached (close 150)
        with self.assertRaisesMessage(svc.TradeError, "day is over"):
            svc.replay_advance(self.acct, "5")
        svc.replay_advance(self.acct, "next_day")
        self.assertEqual((self.acct.replay_date, self.acct.replay_time.strftime("%H:%M")), (self.D2, "09:15"))
        ev = svc.replay_advance(self.acct, "close")                         # expiry day: settles at 15:30
        [s] = ev["settled"]
        self.assertEqual((s.price, s.realized_pnl), (D("100.00"), D("0.00")))     # 22400 - 22300 = 100, bought at 100
        self.assertEqual(svc.to_ist(s.market_time).date(), self.D2)
        self.assertFalse(OptionPosition.objects.filter(account=self.acct).exclude(quantity=0).exists())
        self.assertFalse(self.live.orders.exists())                         # the live wallet is untouched
        with self.assertRaisesMessage(svc.TradeError, "No later trading day"):
            svc.replay_advance(self.acct, "next_day")
        svc.replay_reset(self.acct)
        self.acct.refresh_from_db()
        self.assertEqual((self.acct.balance, self.acct.replay_date, self.acct.orders.count()), (svc.START_BALANCE, None, 0))

    def test_replay_api(self):
        r = self.client.post(reverse("options:replay_start"), {"date": self.D1.isoformat()}, content_type="application/json")
        self.assertEqual(r.json()["time"], "09:15")
        c = self.client.get(reverse("options:chain", args=["NIFTY"]), {"expiry": self.X, "mode": "replay"}).json()
        self.assertTrue(c["replay"])
        self.assertEqual(c["lot_size"], 75)
        self.assertEqual(c["resolution"], "5-minute")
        body = {"mode": "replay", "symbol": "NIFTY", "expiry": self.X, "strike": 22300, "option_type": "PE", "side": "SELL", "lots": 1}
        r = self.client.post(reverse("options:place"), body, content_type="application/json")
        self.assertEqual(r.json()["order"]["price"], 110.0)
        st = self.client.get(reverse("options:state"), {"mode": "replay"}).json()
        self.assertEqual((st["mode"], st["replay"]["date"], len(st["positions"])), ("replay", self.D1.isoformat(), 1))
        self.assertEqual(self.client.get(reverse("options:state")).json()["positions"], [])     # live: separate
        r = self.client.post(reverse("options:replay_advance"), {"to": "15"}, content_type="application/json")
        self.assertEqual(r.json()["time"], "09:30")
        sr = self.client.get(reverse("options:replay_series"),
                             {"symbol": "NIFTY", "expiry": self.X, "strike": 22300, "option_type": "PE"}).json()
        self.assertEqual(len(sr["underlying"]), 3)                          # 09:15, 09:20, 09:25 candles - not the future
        self.assertEqual([x["x"][11:16] for x in sr["premium"]], ["09:15", "09:20", "09:25", "09:30"])
        self.assertEqual(sr["premium"][0]["v"], 110.0)                      # the real open
        self.assertEqual([(t["side"], t["price"], t["x"][11:16]) for t in sr["trades"]], [("SELL", 110.0, "09:15")])

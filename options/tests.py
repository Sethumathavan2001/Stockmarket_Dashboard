from datetime import date, datetime
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

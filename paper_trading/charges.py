"""
Brokerage, taxes and fees for a paper trade, as Zerodha or Angel One would charge them (NSE, equity).

Rates come from each broker's official price list (checked 18 Oct 2026):
  Zerodha    https://zerodha.com/charges
  Angel One  https://www.angelone.in/exchange-transaction-charges

Per executed ORDER:  brokerage                      (so a trade built from several orders pays it several times)
On traded VALUE:     STT, exchange transaction charges, SEBI fee, IPFT, stamp duty
On top:              GST 18 % on (brokerage + exchange charges + SEBI fee + IPFT)
Delivery sells:      DP charge per stock sold.

Contract notes round STT and stamp duty to the rupee; this keeps paise, so a real note can differ
by about a rupee.  Not included: account opening, demat AMC, call & trade, auto square-off by the
broker's dealer, pledging - none of them depend on a single trade.
"""
from decimal import Decimal, ROUND_HALF_UP

D = Decimal
PAISE = D("0.01")
CRORE = D("10000000")

BROKERS = {
    "zerodha": {
        "name": "Zerodha",
        "source": "https://zerodha.com/charges",
        "checked": "2026-10-18",
        "brokerage": {"INTRADAY": {"pct": D("0.03"), "cap": D("20"), "min": D("0")},
                      "DELIVERY": {"pct": D("0"), "cap": D("0"), "min": D("0")}},
        "exchange_pct": D("0.00307"),                 # NSE transaction charges
        "dp_per_sell": D("15.34"),                    # incl. GST: Rs 3.5 CDSL + Rs 9.5 Zerodha + GST
        "notes": "Delivery brokerage is zero. Intraday: 0.03% or Rs 20 per executed order, whichever is lower.",
    },
    "angelone": {
        "name": "Angel One",
        "source": "https://www.angelone.in/exchange-transaction-charges",
        "checked": "2026-10-18",
        "brokerage": {"INTRADAY": {"pct": D("0.1"), "cap": D("20"), "min": D("5")},
                      "DELIVERY": {"pct": D("0.1"), "cap": D("20"), "min": D("5")}},
        "exchange_pct": D("0.0030699"),
        "dp_per_sell": D("23.60"),                    # Rs 20 + 18 % GST per ISIN debit
        "notes": "Intraday and delivery: 0.1% or Rs 20 per executed order, whichever is lower, minimum Rs 5. "
                 "The Rs 0 offer for the first 30 days after opening an account is ignored.",
    },
}

# statutory / exchange charges - the same for every broker
STT = {"INTRADAY": {"buy": D("0"), "sell": D("0.025")}, "DELIVERY": {"buy": D("0.1"), "sell": D("0.1")}}   # %
STAMP_BUY = {"INTRADAY": D("0.003"), "DELIVERY": D("0.015")}                                                 # %
SEBI_PER_CRORE = D("10")
IPFT_PER_CRORE = D("0.01")
GST = D("0.18")


def _pct(value, pct):
    return value * pct / 100


def order_brokerage(broker, product, order_value):
    """Brokerage of one executed order."""
    b = BROKERS[broker]["brokerage"][product]
    if b["pct"] == 0:
        return D("0")
    fee = min(_pct(D(order_value), b["pct"]), b["cap"])
    return max(fee, b["min"])


def trade_charges(broker, product, buy_value, sell_value, legs):
    """
    Charges of one round trip.
      buy_value / sell_value : rupees bought / sold in this trade (sell_value 0 = still open)
      legs : [(order_value, share)] for every order that took part - `share` is the part of that
             order belonging to this trade (an order split across trades pays its brokerage once).
    """
    cfg = BROKERS[broker]
    buy_value, sell_value = D(buy_value), D(sell_value)
    turnover = buy_value + sell_value
    brokerage = sum((order_brokerage(broker, product, v) * D(share) for v, share in legs), D("0"))
    stt = _pct(buy_value, STT[product]["buy"]) + _pct(sell_value, STT[product]["sell"])
    exchange = _pct(turnover, cfg["exchange_pct"])
    sebi = turnover * SEBI_PER_CRORE / CRORE
    ipft = turnover * IPFT_PER_CRORE / CRORE
    stamp = _pct(buy_value, STAMP_BUY[product])
    gst = (brokerage + exchange + sebi + ipft) * GST
    dp = cfg["dp_per_sell"] if product == "DELIVERY" and sell_value > 0 else D("0")
    parts = {"brokerage": brokerage, "stt": stt, "exchange": exchange, "sebi": sebi + ipft, "stamp": stamp,
             "gst": gst, "dp": dp}
    parts = {k: v.quantize(PAISE, ROUND_HALF_UP) for k, v in parts.items()}
    parts["total"] = sum(parts.values(), D("0"))
    return parts


def broker_info():
    """Rates for the page (shown under each broker's table)."""
    return {key: {"name": b["name"], "source": b["source"], "checked": b["checked"], "notes": b["notes"],
                  "brokerage": {p: {k: float(v) for k, v in r.items()} for p, r in b["brokerage"].items()},
                  "exchange_pct": float(b["exchange_pct"]), "dp_per_sell": float(b["dp_per_sell"])}
            for key, b in BROKERS.items()} | {"_common": {
                "stt": {p: {k: float(v) for k, v in r.items()} for p, r in STT.items()},
                "stamp_buy": {p: float(v) for p, v in STAMP_BUY.items()},
                "sebi_per_crore": float(SEBI_PER_CRORE), "ipft_per_crore": float(IPFT_PER_CRORE),
                "gst": float(GST)}}

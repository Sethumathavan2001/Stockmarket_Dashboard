"""
How did a saved Claude call actually play out?

For a trade plan (analysis) or a pick: walk the 5-minute candles that START after the call was
made, on the session it was made for (the same day if made before 15:30, otherwise the next
session):  trigger the entry (MARKET = first candle's open; BREAKOUT / PULLBACK / pick entries =
when price reaches the level, at the open if it gaps through), then the first of stop / target
wins - both inside one candle counts as the stop.  Still open at 15:20 = closed there.
Candles come from the database, else Yahoo (recent days), else today's live feed.
"""
from datetime import datetime, time, timedelta

import pandas as pd
from django.core.cache import cache

from market.models import FiveMinuteData
from prediction.market_time import IST, get_market_status

from . import tools

EXIT_TIME = time(15, 20)
SESSION_END = time(15, 30)


def _day_candles(company, day):
    from paper_trading.services import _day_bounds_db, _yahoo_day_5m, fetch_live_today

    today = datetime.now(IST).date()
    if day == today:
        df = fetch_live_today(company)
        return df[["timestamp", "open", "high", "low", "close"]] if not df.empty else df
    start, end = _day_bounds_db(day)
    rows = list(FiveMinuteData.objects.filter(company=company, timestamp__gte=start, timestamp__lt=end)
                .order_by("timestamp").values("timestamp", "open", "high", "low", "close"))
    if rows:
        df = pd.DataFrame(rows)
        for c in ("open", "high", "low", "close"):
            df[c] = df[c].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(IST)
        return df.dropna()
    if (today - day).days <= 6:
        return _yahoo_day_5m(company, day)
    return pd.DataFrame()


def _session_for(company, created):
    """(day, candles) of the session a call made at `created` (IST) is about."""
    today = datetime.now(IST).date()
    day = created.date()
    if created.time() >= SESSION_END or day.weekday() >= 5:
        day += timedelta(days=1)
    while day <= today:
        if day.weekday() < 5:
            with tools.DATA_LOCK:
                df = _day_candles(company, day)
            if not df.empty:
                return day, df
            if day == today:
                return day, df
        day += timedelta(days=1)
    return None, pd.DataFrame()


def simulate(df, created, side, entry, stop, target, entry_type="TOUCH", valid_until=None):
    """side +1 long / -1 short. Returns dict(status, fill, exit, pnl_pct, r)."""
    df = df[df["timestamp"] >= created].reset_index(drop=True)
    df = df[df["timestamp"].dt.time < SESSION_END]
    if df.empty:
        return {"status": "WAITING"}
    fill = None
    for i, r in enumerate(df.itertuples(index=False)):
        t = r.timestamp.time()
        if fill is None:
            if valid_until and t > valid_until or t >= EXIT_TIME:
                return {"status": "NOT TRIGGERED"}
            if entry_type == "MARKET":
                fill = r.open
            elif entry_type == "BREAKOUT":
                if (r.high >= entry) if side > 0 else (r.low <= entry):
                    fill = max(entry, r.open) if side > 0 else min(entry, r.open)
            else:                                       # PULLBACK / limit-like touch
                if (r.low <= entry) if side > 0 else (r.high >= entry):
                    fill = min(entry, r.open) if side > 0 else max(entry, r.open)
                elif r.low <= entry <= r.high:
                    fill = entry
            if fill is None:
                continue
        hit_stop = (r.low <= stop) if side > 0 else (r.high >= stop)
        hit_target = (r.high >= target) if side > 0 else (r.low <= target)
        if hit_stop or hit_target or t >= EXIT_TIME:
            if hit_stop:
                status, exit_px = "STOP HIT", stop
            elif hit_target:
                status, exit_px = "TARGET HIT", target
            else:
                status, exit_px = "CLOSED 15:20", r.close
            pnl = (exit_px / fill - 1) * side * 100
            risk = abs(fill - stop) / fill * 100
            return {"status": status, "fill": round(fill, 2), "exit": round(exit_px, 2), "pnl_pct": round(pnl, 3),
                    "r": round(pnl / risk, 2) if risk else None, "exit_time": t.strftime("%H:%M")}
    last = df.iloc[-1]
    if fill is None:
        return {"status": "WAITING"}
    pnl = (last["close"] / fill - 1) * side * 100
    return {"status": "RUNNING", "fill": round(fill, 2), "exit": round(float(last["close"]), 2),
            "pnl_pct": round(pnl, 3)}


def _finalish(out, day):
    """Final results can be cached for long; anything about today only briefly."""
    return 86400 if day and day < datetime.now(IST).date() and out.get("status") not in ("WAITING", "RUNNING") else 120


def _parse_hhmm(text):
    try:
        return datetime.strptime((text or "").strip()[:5], "%H:%M").time()
    except ValueError:
        return None


def evaluate_report(report):
    key = f"ai_prediction:eval:{report.pk}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    created = report.created_at.astimezone(IST)
    if report.kind == "analysis":
        out = _evaluate_analysis(report, created)
    else:
        out = _evaluate_picks(report, created)
    cache.set(key, out, out.pop("_ttl", 120))
    return out


def _evaluate_analysis(report, created):
    day, df = _session_for(report.company, created)
    if day is None or df.empty:
        return {"status": "WAITING", "day": day.isoformat() if day else None, "_ttl": 120}
    plan = report.result.get("trade_plan", {})
    bias = report.result.get("bias")
    after = df[df["timestamp"] >= created]
    price = report.price_at or (float(after["open"].iloc[0]) if not after.empty else None)
    last = float(after["close"].iloc[-1]) if not after.empty else None
    move = (last / price - 1) * 100 if price and last else None
    out = {"day": day.isoformat(), "move_pct": round(move, 3) if move is not None else None,
           "bias_right": None if move is None or bias == "NEUTRAL" or abs(move) < 0.1 else
           (move > 0) == (bias == "BULLISH"),
           "complete": day < datetime.now(IST).date() or get_market_status() != "OPEN"}
    if plan.get("action") in ("BUY", "SELL") and None not in (plan.get("entry"), plan.get("stop_loss"), plan.get("target_1")):
        out.update(simulate(df, created, 1 if plan["action"] == "BUY" else -1, plan["entry"], plan["stop_loss"],
                            plan["target_1"], plan.get("entry_type") or "TOUCH", _parse_hhmm(plan.get("valid_until"))))
    else:
        out["status"] = "NO TRADE"
    out["_ttl"] = _finalish(out, day)
    return out


def _evaluate_picks(report, created):
    rows, day_seen = [], None
    for p in report.result.get("picks", []):
        try:
            company = tools.find_company(p["company_code"])
        except tools.ToolError:
            continue
        day, df = _session_for(company, created)
        day_seen = day_seen or day
        if day is None or df.empty:
            rows.append({"company_code": p["company_code"], "status": "WAITING"})
            continue
        r = simulate(df, created, 1 if p["direction"] == "LONG" else -1, p["entry"], p["stop_loss"], p["target"])
        rows.append({"company_code": p["company_code"], "day": day.isoformat(), **r})
    done = [r for r in rows if "pnl_pct" in r and r["status"] != "RUNNING"]
    final = bool(rows) and all(r["status"] not in ("WAITING", "RUNNING") for r in rows)
    return {"picks": rows, "day": day_seen.isoformat() if day_seen else None,
            "closed": len(done), "wins": sum(r["pnl_pct"] > 0 for r in done),
            "avg_pnl_pct": round(sum(r["pnl_pct"] for r in done) / len(done), 3) if done else None,
            "_ttl": 86400 if final and day_seen and day_seen < datetime.now(IST).date() else 120}

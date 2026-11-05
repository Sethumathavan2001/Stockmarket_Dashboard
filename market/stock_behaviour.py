"""
Stock Behaviour page: how stocks have behaved on chosen kinds of day, over the matrix in market/behaviour_data.py.

  GET  /stock-behaviour/                  the page
  GET  /api/stock-behaviour/meta/         indices, sectors, coverage, rebuild progress
  GET  /api/stock-behaviour/data/         every panel for the filters below
  GET  /api/stock-behaviour/stock/?code=  one stock under the same filters: weekdays, sessions, every index
  POST /api/stock-behaviour/build/        rebuild the matrix in the background

Filters: index, trend (any/up/down), weekday (any/0-4), expiry (any/all/weekly/monthly/none), session (day/morning/
afternoon/evening), sector, period (6m/1y/3y/5y/all), measure (o2c/c2c), turnover (crore), move (0 = automatic, or the %
the index must move to count as a trend day).

Every filter applies to every panel, except that a panel doesn't filter on what it splits by: the Index panel shows both
trends, the Weekday panel all five days, the Volatility panel all three sessions, the Sector table every sector.

With a session chosen, both the index's trend and the stock's move are measured inside that session, and only the
months of stored 5-minute candles count.

Lists are ranked by the lower bound of the hit rate (Wilson, 90%), so 6 out of 8 days does not outrank 60 out of 90.
"Steady" means the hit rate was above half in both the older and the newer half of the matching days.
The repeat check under a list (Engine.recheck) picks the best stocks on the older half alone and reports how they did on
the newer half against the typical stock: the honest test of whether that list means anything.
"""
import math
from dataclasses import dataclass, replace

import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods

from . import behaviour_data
from .behaviour_data import MONTHLY, SESSIONS, WEEKLY

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
PERIODS = {"6m": 183, "1y": 365, "3y": 1096, "5y": 1826, "all": None}
SESSION_NAMES = [s[0] for s in SESSIONS]
DAY_MOVE, SESSION_MOVE = 0.30, 0.15             # default % the index must move to count as a trend
MIN_DAYS = 8                                    # fewer matching days than this says nothing about a stock
COVERAGE = 0.6                                  # a stock must have traded on this share of the matching days
Z = 1.645
TOP, TOP_SMALL = 15, 8
RECHECK = 20                                    # how many stocks the repeat check follows
HIDDEN_SECTORS = ("", "ETF")                    # left out unless asked for by name


def _clock(minute):
    return f"{minute // 60:02d}:{minute % 60:02d}"


@dataclass(frozen=True)
class Filters:
    index: str = "NIFTY"
    trend: str = "any"
    weekday: int = -1
    expiry: str = "any"
    session: str = "day"
    sector: str = ""
    period: str = "3y"
    measure: str = "o2c"
    turnover: float = 25.0
    move: float = 0.0

    @classmethod
    def parse(cls, get):
        def pick(name, allowed, default):
            value = get.get(name, default)
            return value if value in allowed else default

        def number(name, default, cast=float):
            try:
                value = cast(get.get(name, default))
            except (TypeError, ValueError):
                return default
            return value if math.isfinite(value) else default

        weekday = number("weekday", -1, int)
        return cls(index=get.get("index", "NIFTY"), trend=pick("trend", ("any", "up", "down"), "any"),
                   weekday=weekday if weekday in range(-1, 5) else -1,
                   expiry=pick("expiry", ("any", "all", "weekly", "monthly", "none"), "any"),
                   session=pick("session", ["day"] + SESSION_NAMES, "day"), sector=get.get("sector", ""),
                   period=pick("period", PERIODS, "3y"), measure=pick("measure", ("o2c", "c2c"), "o2c"),
                   turnover=max(number("turnover", 25.0), 0.0), move=max(number("move", 0.0), 0.0))


def _round(x, digits=2):
    return None if x is None or not np.isfinite(x) else round(float(x), digits)


def _finite(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _mean(values):
    values = _finite(values)
    return float(values.mean()) if values.size else None


def _up_pct(values):
    values = _finite(values)
    return float((values > 0).mean() * 100) if values.size else None


def wilson_lower(hits, n):
    """Lower bound of the true hit rate (0-1), element-wise; 0 where n is 0."""
    hits, n = np.asarray(hits, dtype=np.float64), np.asarray(n, dtype=np.float64)
    safe = np.maximum(n, 1)
    p = hits / safe
    z2 = Z * Z
    centre = p + z2 / (2 * safe)
    margin = Z * np.sqrt(p * (1 - p) / safe + z2 / (4 * safe * safe))
    return np.where(n > 0, (centre - margin) / (1 + z2 / safe), 0.0)


def _column_stats(moves, sign):
    """Per stock, on these day rows: days traded, hits towards `sign`, average move, older / newer half hit rates."""
    n = np.isfinite(moves).sum(axis=0)
    hits = (moves * sign > 0).sum(axis=0)
    avg = np.nansum(moves, axis=0) / np.maximum(n, 1)
    half = moves.shape[0] // 2
    older, newer = moves[:half], moves[half:]
    older_rate = (older * sign > 0).sum(axis=0) / np.maximum(np.isfinite(older).sum(axis=0), 1) * 100
    newer_rate = (newer * sign > 0).sum(axis=0) / np.maximum(np.isfinite(newer).sum(axis=0), 1) * 100
    return n, hits, avg, older_rate, newer_rate


class Engine:
    def __init__(self, data, filters):
        self.d, self.f = data, filters
        keys = [str(k) for k in data["ix_keys"]]
        if filters.index not in keys:
            raise ValueError(f"Unknown index {filters.index}")
        self.i = keys.index(filters.index)
        self.days = data["days"]
        self.weekday = (self.days.astype("datetime64[D]").astype(np.int64) + 3) % 7       # 1 Jan 1970 was a Thursday
        self.pos5 = np.full(len(self.days), -1, dtype=np.int64)
        self.pos5[data["rows5"]] = np.arange(len(data["rows5"]))
        self.session = SESSION_NAMES.index(filters.session) if filters.session in SESSION_NAMES else None
        sectors = data["sectors"]
        base = data["turnover"] >= filters.turnover
        self.all_sectors = np.where(base & ~np.isin(sectors, HIDDEN_SECTORS))[0]
        self.cols = np.where(base & (sectors == filters.sector))[0] if filters.sector else self.all_sectors

    # ------------------------------------------------------------------ days
    def index_move(self, session, index=None):
        i = self.i if index is None else index
        if session is not None:
            return self.d["ix_sret"][session, i]
        return self.d["ix_c2c" if self.f.measure == "c2c" else "ix_o2c"][i]

    def threshold(self, session):
        return self.f.move or (DAY_MOVE if session is None else SESSION_MOVE)

    def in_period(self):
        days = PERIODS[self.f.period]
        if days is None:
            return np.ones(len(self.days), dtype=bool)
        return self.days >= self.days[-1] - np.timedelta64(days, "D")

    def expiry_mask(self, kind, index=None):
        e = self.d["ix_expiry"][self.i if index is None else index]
        return {"any": np.ones(len(e), dtype=bool), "all": e > 0, "weekly": e == WEEKLY, "monthly": e == MONTHLY,
                "none": e == 0}[kind]

    def mask(self, *, trend=None, weekday=None, expiry=None, session="filter"):
        """The matching days; any argument overrides that filter (a panel's "all but my own split")."""
        trend = self.f.trend if trend is None else trend
        weekday = self.f.weekday if weekday is None else weekday
        expiry = self.f.expiry if expiry is None else expiry
        session = self.session if session == "filter" else session
        move = self.index_move(session)
        m = self.in_period() & self.expiry_mask(expiry) & np.isfinite(move)
        if weekday >= 0:
            m &= self.weekday == weekday
        limit = self.threshold(session)
        if trend == "up":
            m &= move >= limit
        elif trend == "down":
            m &= move <= -limit
        if session is not None:
            m &= self.pos5 >= 0
        return m

    # ------------------------------------------------------------------ stocks
    def moves(self, mask, session, cols):
        rows = np.where(mask)[0]
        if session is None:
            return self.d[self.f.measure][np.ix_(rows, cols)]
        return self.d["sret"][session][np.ix_(self.pos5[rows], cols)]

    def ranges(self, mask, session, cols):
        rows = np.where(mask)[0]
        if session is None:
            return self.d["rng"][np.ix_(rows, cols)]
        return self.d["srng"][session][np.ix_(self.pos5[rows], cols)]

    def need(self, rows):
        return max(MIN_DAYS, math.ceil(rows * COVERAGE))

    def stock(self, col, **extra):
        code = str(self.d["codes"][col])
        return {"code": code, "symbol": code.removesuffix(".NS"), "name": str(self.d["names"][col]),
                "sector": str(self.d["sectors"][col]), "turnover": _round(self.d["turnover"][col], 0), **extra}

    def top(self, mask, sign, limit=TOP, session="filter", against=False):
        """Stocks that moved towards `sign` on these days, surest first. A stock needs an average move that way too,
        except `against` the index, where almost none has one: there it's whichever went that way most often."""
        session = self.session if session == "filter" else session
        rows = int(mask.sum())
        if rows < MIN_DAYS or not len(self.cols):
            return []
        n, hits, avg, older, newer = _column_stats(self.moves(mask, session, self.cols), sign)
        ok = (n >= self.need(rows)) & (hits > 0 if against else avg * sign > 0)
        score = np.where(ok, wilson_lower(hits, n), -1.0)
        order = np.argsort(-score, kind="stable")[:limit]
        return [self.stock(int(self.cols[j]), hit=_round(hits[j] / n[j] * 100, 1), avg=_round(avg[j]), days=int(n[j]),
                           older=_round(older[j], 0), newer=_round(newer[j], 0),
                           steady=bool(older[j] > 50 and newer[j] > 50))
                for j in order if score[j] >= 0]

    def recheck(self, mask, sign, session="filter"):
        """Does this list repeat? Picks the RECHECK best stocks on the older half of the days only, then compares their
        hit rate on the newer half with the typical stock's. None when there are too few days to split."""
        session = self.session if session == "filter" else session
        rows = int(mask.sum())
        if rows < 2 * MIN_DAYS or not len(self.cols):
            return None
        moves = self.moves(mask, session, self.cols)
        half = rows // 2
        older, newer = moves[:half], moves[half:]
        n_old, n_new = np.isfinite(older).sum(axis=0), np.isfinite(newer).sum(axis=0)
        ok = np.where((n_old >= self.need(half)) & (n_new >= self.need(rows - half)))[0]
        if len(ok) < 2 * RECHECK:
            return None
        old_rate = (older[:, ok] * sign > 0).sum(axis=0) / n_old[ok] * 100
        new_rate = (newer[:, ok] * sign > 0).sum(axis=0) / n_new[ok] * 100
        best = np.argsort(-old_rate, kind="stable")[:RECHECK]
        return {"older": _round(old_rate[best].mean(), 1), "newer": _round(new_rate[best].mean(), 1),
                "typical": _round(new_rate.mean(), 1), "stocks": RECHECK}

    # ------------------------------------------------------------------ panels
    def summary(self):
        m = self.mask()
        move = self.index_move(self.session)[m]
        breadth = None
        if m.any() and len(self.cols):
            s = self.moves(m, self.session, self.cols)
            valid = np.isfinite(s).sum()
            breadth = (s > 0).sum() / valid * 100 if valid else None
        days = self.days[m]
        return {"days": int(m.sum()), "first": str(days[0]) if len(days) else None,
                "last": str(days[-1]) if len(days) else None, "stocks": int(len(self.cols)),
                "index_avg": _round(_mean(move)), "index_up_pct": _round(_up_pct(move), 1),
                "breadth": _round(breadth, 1), "threshold": self.threshold(self.session)}

    def quadrants(self):
        out = {}
        for trend in ("up", "down"):
            m = self.mask(trend=trend)
            out[trend] = {"days": int(m.sum()), "stocks_up": self.top(m, 1, against=trend == "down"),
                          "stocks_down": self.top(m, -1, against=trend == "up"),
                          "check_up": self.recheck(m, 1), "check_down": self.recheck(m, -1)}
        return out

    def volatility(self):
        m = self.mask(session=None) & (self.pos5 >= 0)
        rows = int(m.sum())
        out = {"days": rows, "sessions": []}
        if rows < MIN_DAYS or not len(self.cols):
            return out
        count = len(SESSIONS)
        ranges = [self.ranges(m, s, self.cols) for s in range(count)]
        moves = [np.abs(self.moves(m, s, self.cols)) for s in range(count)]
        n = [np.isfinite(r).sum(axis=0) for r in ranges]
        avg = [np.nansum(r, axis=0) / np.maximum(c, 1) for r, c in zip(ranges, n)]
        total = np.maximum(sum(avg), 1e-9)
        need = self.need(rows)
        for s, (name, label, start, end) in enumerate(SESSIONS):
            ok = n[s] >= need
            share = avg[s] / total * 100
            absolute = np.nansum(moves[s], axis=0) / np.maximum(n[s], 1)

            def listed(key, s=s, ok=ok, share=share, absolute=absolute):
                order = np.argsort(-np.where(ok, key, -1.0), kind="stable")[:TOP]
                return [self.stock(int(self.cols[j]), range=_round(avg[s][j]), move=_round(absolute[j]),
                                   share=_round(share[j], 0), days=int(n[s][j])) for j in order if ok[j]]

            out["sessions"].append({"name": name, "label": label, "time": f"{_clock(start)}-{_clock(end)}",
                                    "typical": _round(np.median(avg[s][ok])) if ok.any() else None,
                                    "biggest": listed(avg[s]), "peak": listed(share)})
        return out

    def weekdays(self):
        out = []
        for w, label in enumerate(WEEKDAYS):
            m = self.mask(weekday=w)
            move = self.index_move(self.session)[m]
            out.append({"weekday": w, "label": label, "days": int(m.sum()), "index_avg": _round(_mean(move)),
                        "index_up_pct": _round(_up_pct(move), 1),
                        "up": self.top(m, 1, TOP_SMALL), "down": self.top(m, -1, TOP_SMALL),
                        "check_up": self.recheck(m, 1), "check_down": self.recheck(m, -1)})
        return out

    @staticmethod
    def _by_sector(values, onehot):
        """Equal-weighted average per sector for each day row -> (days, sectors)."""
        valid = np.isfinite(values)
        return (np.where(valid, values, 0.0).astype(np.float64) @ onehot) / np.maximum(valid.astype(np.float64) @ onehot, 1)

    def sectors(self):
        names = sorted({str(self.d["sectors"][c]) for c in self.all_sectors})
        if not names:
            return {"columns": [], "rows": []}
        lookup = {n: i for i, n in enumerate(names)}
        onehot = np.zeros((len(self.all_sectors), len(names)))
        for row, col in enumerate(self.all_sectors):
            onehot[row, lookup[str(self.d["sectors"][col])]] = 1
        columns = [("all", "These days", {}, self.mask()),
                   ("up", "Index up", {"trend": "up"}, self.mask(trend="up")),
                   ("down", "Index down", {"trend": "down"}, self.mask(trend="down"))]
        columns += [(f"w{w}", label, {"weekday": w}, self.mask(weekday=w)) for w, label in enumerate(WEEKDAYS)]
        if (self.d["ix_expiry"][self.i] > 0).any():
            kind = self.f.expiry if self.f.expiry in ("weekly", "monthly") else "all"
            columns.append(("expiry", "Expiry", {"expiry": kind}, self.mask(expiry=kind)))
        rows = [{"sector": n, "stocks": int(onehot[:, i].sum()), "cells": {}, "range": {}} for i, n in enumerate(names)]
        for key, _, _, m in columns:
            if not m.any():
                continue
            series = self._by_sector(self.moves(m, self.session, self.all_sectors), onehot)
            for i, row in enumerate(rows):
                row["cells"][key] = {"avg": _round(series[:, i].mean()), "up": _round((series[:, i] > 0).mean() * 100, 0),
                                     "days": int(m.sum())}
        vol = self.mask(session=None) & (self.pos5 >= 0)
        if vol.any():
            for s, name in enumerate(SESSION_NAMES):
                series = self._by_sector(self.ranges(vol, s, self.all_sectors), onehot)
                for i, row in enumerate(rows):
                    row["range"][name] = _round(series[:, i].mean())
        return {"columns": [{"key": k, "label": label, "set": s} for k, label, s, _ in columns], "rows": rows}

    def expiry_kind(self):
        return self.f.expiry if self.f.expiry in ("weekly", "monthly") else "all"

    def expiry_row(self, index):
        """One index's expiry days against its other days, over the chosen period."""
        name, kind = str(self.d["ix_keys"][index]), self.expiry_kind()
        period = self.in_period()
        day_move, day_range = self.index_move(None, index), self.d["ix_rng"][index]
        exp = period & self.expiry_mask(kind, index) & np.isfinite(day_move)
        other = period & self.expiry_mask("none", index) & np.isfinite(day_move)

        def side(m):
            sessions = {n: _round(_mean(self.d["ix_srng"][s, index][m])) for s, n in enumerate(SESSION_NAMES)}
            return {"days": int(m.sum()), "avg": _round(_mean(day_move[m])), "abs": _round(_mean(np.abs(day_move[m]))),
                    "up": _round(_up_pct(day_move[m]), 0), "range": _round(_mean(day_range[m])), "sessions": sessions}

        engine = self if index == self.i else Engine(self.d, replace(self.f, index=name))
        m = engine.mask(trend="any", weekday=-1, expiry=kind, session=None)
        best, worst = engine.top(m, 1, 1, session=None), engine.top(m, -1, 1, session=None)
        return {"index": name, "name": str(self.d["ix_names"][index]), "expiry": side(exp), "other": side(other),
                "best": best[0] if best else None, "worst": worst[0] if worst else None}

    def expiry(self):
        has = [j for j in range(len(self.d["ix_keys"])) if (self.d["ix_expiry"][j] > 0).any()]
        out = {"indices": [self.expiry_row(j) for j in has], "selected": self.f.index, "has_expiry": self.i in has}
        if not out["has_expiry"]:
            return out
        kind = self.expiry_kind()
        exp, other = self.mask(expiry=kind), self.mask(expiry="none")
        out.update(kind=kind, days=int(exp.sum()), up=self.top(exp, 1), down=self.top(exp, -1), wider=self._wider(exp, other),
                   check_up=self.recheck(exp, 1), check_down=self.recheck(exp, -1))
        return out

    def _wider(self, exp, other):
        """Stocks whose range grows most on expiry days compared with their other days."""
        rows = int(exp.sum())
        if rows < MIN_DAYS or not other.any() or not len(self.cols):
            return []
        a, b = self.ranges(exp, self.session, self.cols), self.ranges(other, self.session, self.cols)
        n, n_other = np.isfinite(a).sum(axis=0), np.isfinite(b).sum(axis=0)
        avg, base = np.nansum(a, axis=0) / np.maximum(n, 1), np.nansum(b, axis=0) / np.maximum(n_other, 1)
        ok = (n >= self.need(rows)) & (n_other >= MIN_DAYS) & (base > 0)
        ratio = np.where(ok, avg / np.maximum(base, 1e-9), -1.0)
        order = np.argsort(-ratio, kind="stable")[:TOP]
        return [self.stock(int(self.cols[j]), range=_round(avg[j]), normal=_round(base[j]), ratio=_round(ratio[j]),
                           days=int(n[j])) for j in order if ratio[j] >= 0]

    def all_panels(self):
        return {"summary": self.summary(), "quadrants": self.quadrants(), "volatility": self.volatility(),
                "weekdays": self.weekdays(), "sectors": self.sectors(), "expiry": self.expiry()}

    # ------------------------------------------------------------------ one stock
    def _line(self, col, mask, session):
        """A stock's record on these days."""
        one = np.array([col])
        moves = self.moves(mask, session, one)[:, 0]
        return {"days": int(np.isfinite(moves).sum()), "up": _round(_up_pct(moves), 0), "avg": _round(_mean(moves)),
                "range": _round(_mean(self.ranges(mask, session, one)[:, 0]))}

    def profile(self, code):
        found = np.where(self.d["codes"] == code)[0]
        if not len(found):
            return None
        col, s, kind = int(found[0]), self.session, self.expiry_kind()
        out = {"stock": self.stock(col), "these_days": self._line(col, self.mask(), s),
               "weekdays": [{"label": label, "weekday": w, **self._line(col, self.mask(weekday=w), s)}
                            for w, label in enumerate(WEEKDAYS)],
               "sessions": [{"label": label, "time": f"{_clock(a)}-{_clock(b)}", "name": name,
                             **self._line(col, self.mask(session=i), i)} for i, (name, label, a, b) in enumerate(SESSIONS)],
               "indices": []}
        for j, key in enumerate(self.d["ix_keys"]):
            engine = self if j == self.i else Engine(self.d, replace(self.f, index=str(key)))
            has_expiry = bool((self.d["ix_expiry"][j] > 0).any())
            out["indices"].append({"index": str(key), "name": str(self.d["ix_names"][j]),
                                   "up": engine._line(col, engine.mask(trend="up"), s),
                                   "down": engine._line(col, engine.mask(trend="down"), s),
                                   "expiry": engine._line(col, engine.mask(expiry=kind), s) if has_expiry else None})
        return out


def _engine(request):
    data = behaviour_data.load()
    if data is None:
        return None, JsonResponse({"error": "Not built yet", "code": "not_built"}, status=409)
    try:
        return Engine(data, Filters.parse(request.GET)), None
    except ValueError as exc:
        return None, JsonResponse({"error": str(exc)}, status=400)


@ensure_csrf_cookie
def page(request):
    return render(request, "market/stock_behaviour.html")


@require_GET
def meta(request):
    data = behaviour_data.load()
    out = {"build": behaviour_data.progress(), "ready": data is not None}
    if data is None:
        return JsonResponse(out)
    expiry, rows5 = data["ix_expiry"], data["rows5"]
    out.update(indices=[{"key": str(k), "name": str(n), "expiry": bool((expiry[i] > 0).any())}
                        for i, (k, n) in enumerate(zip(data["ix_keys"], data["ix_names"]))],
               sectors=sorted({str(s) for s in data["sectors"] if s}), first=str(data["days"][0]), last=str(data["days"][-1]),
               days=int(len(data["days"])), stocks=int(len(data["codes"])), built=str(data["built"]),
               session_days=int(len(rows5)), session_first=str(data["days"][rows5[0]]) if len(rows5) else None,
               session_last=str(data["days"][rows5[-1]]) if len(rows5) else None,
               symbols=[{"code": str(c), "name": str(n)} for c, n in zip(data["codes"], data["names"])])
    return JsonResponse(out)


@require_GET
def panels(request):
    engine, error = _engine(request)
    return error or JsonResponse(engine.all_panels())


@require_GET
def stock(request):
    engine, error = _engine(request)
    if error:
        return error
    result = engine.profile(request.GET.get("code", ""))
    return JsonResponse(result) if result else JsonResponse({"error": "Unknown stock"}, status=404)


@require_http_methods(["POST"])
def build(request):
    started = behaviour_data.start_build()
    return JsonResponse({"started": started, "build": behaviour_data.progress()}, status=202 if started else 409)

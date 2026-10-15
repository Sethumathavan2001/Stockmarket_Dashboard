import json
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace as NS
from unittest import mock

import pandas as pd
from django.core.cache import cache
from django.test import TestCase

from market.models import Company
from prediction.market_time import IST

from . import agent, evaluate, tools
from .models import AIReport


def good_analysis(code="TST.NS", **plan):
    p = {"action": "BUY", "entry_type": "BREAKOUT", "entry": 101.0, "stop_loss": 99.5, "target_1": 103.5,
         "target_2": 105.0, "valid_until": "13:00", "invalidation": "Back below VWAP", "position_note": "1% risk"}
    p.update(plan)
    return {"company_code": code, "headline": "Breakout above PDH", "bias": "BULLISH", "confidence": 62,
            "price_at_analysis": 100.6, "trade_plan": p,
            "price_outlook": [{"horizon": "1 hour", "expected_price": 101.5, "low": 100, "high": 102.4,
                               "direction": "UP", "probability": 58}],
            "key_levels": [{"label": "PDH", "price": 101.0, "kind": "resistance"}],
            "scenarios": [{"name": "Breakout", "probability": 55, "trigger": "15m close > 101", "path": "to 103.5"}],
            "technical_view": "Above VWAP", "news_view": "Order win", "market_view": "NIFTY firm",
            "catalysts": ["Order win"], "risks": ["Weak market"],
            "sources": [{"title": "News", "url": "https://example.com/a"}]}


class ValidationTests(TestCase):
    def setUp(self):
        self.c = Company.objects.create(name="Test Co", code="TST.NS", exchange="NSE")

    def test_good_analysis_passes(self):
        self.assertEqual(tools.validate_analysis(good_analysis(), self.c), [])

    def test_inconsistent_levels_rejected(self):
        errs = tools.validate_analysis(good_analysis(stop_loss=102.0), self.c)
        self.assertTrue(any("stop_loss < entry" in e for e in errs))

    def test_missing_and_unknown_fields_rejected(self):
        d = good_analysis()
        del d["risks"]
        d["extra"] = 1
        errs = tools.validate_analysis(d, self.c)
        self.assertIn("risks: missing", errs)
        self.assertIn("extra: unknown field", errs)

    def test_no_trade_needs_no_levels(self):
        d = good_analysis(action="NO_TRADE", entry_type="NONE", entry=None, stop_loss=None, target_1=None, target_2=None)
        self.assertEqual(tools.validate_analysis(d, self.c), [])

    def test_picks_must_be_known_companies(self):
        pick = {"company_code": "NOPE.NS", "direction": "LONG", "setup": "ORB", "entry": 10, "stop_loss": 9,
                "target": 12, "confidence": 55, "reason": "x", "risk": "y"}
        d = {"market_bias": "BULLISH", "market_summary": "ok", "picks": [pick], "avoid": []}
        self.assertTrue(any("not in this app" in e for e in tools.validate_picks(d)))
        pick["company_code"] = "tst"                       # resolved to the real code
        self.assertEqual(tools.validate_picks(d), [])
        self.assertEqual(pick["company_code"], "TST.NS")


def day_frame(spec, day=date(2026, 3, 2)):
    start = datetime.combine(day, time(9, 15), tzinfo=IST)
    return pd.DataFrame([(start + timedelta(minutes=5 * i), o, h, l, c) for i, (o, h, l, c) in enumerate(spec)],
                        columns=["timestamp", "open", "high", "low", "close"])


class SimulateTests(TestCase):
    created = datetime(2026, 3, 2, 9, 15, tzinfo=IST)

    def test_breakout_then_target(self):
        df = day_frame([(100, 100.5, 99.8, 100.2), (100.2, 101.2, 100.1, 101.0), (101, 103.6, 100.9, 103.4)])
        r = evaluate.simulate(df, self.created, 1, 101.0, 99.5, 103.5, "BREAKOUT")
        self.assertEqual(r["status"], "TARGET HIT")
        self.assertEqual(r["fill"], 101.0)
        self.assertGreater(r["pnl_pct"], 0)

    def test_stop_and_target_in_one_candle_counts_as_stop(self):
        df = day_frame([(101, 104, 99, 102)])
        r = evaluate.simulate(df, self.created, 1, 101.0, 99.5, 103.5, "MARKET")
        self.assertEqual(r["status"], "STOP HIT")

    def test_short_not_triggered_before_valid_until(self):
        df = day_frame([(100, 100.5, 99.8, 100.2)] * 30)
        r = evaluate.simulate(df, self.created, -1, 98.0, 99.0, 96.0, "BREAKOUT", time(10, 0))
        self.assertEqual(r["status"], "NOT TRIGGERED")

    def test_candles_before_the_call_are_ignored(self):
        df = day_frame([(100, 104, 100, 103)] + [(100, 100.2, 99.9, 100)] * 3)
        r = evaluate.simulate(df, self.created + timedelta(minutes=5), 1, 100.1, 99.0, 103.5, "BREAKOUT")
        self.assertEqual(r["status"], "RUNNING")


# ---------------------------------------------------------------- fake Anthropic client
class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for b in self.message.content:
            yield NS(type="content_block_stop", content_block=b)

    def get_final_message(self):
        return self.message


def msg(blocks, stop="tool_use", searches=0):
    usage = NS(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0, cache_read_input_tokens=500,
               server_tool_use=NS(web_search_requests=searches))
    return NS(content=blocks, stop_reason=stop, usage=usage, model="claude-opus-5-5")


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.beta = NS(messages=NS(stream=self.stream))

    def stream(self, **kwargs):
        self.calls.append(dict(kwargs, messages=list(kwargs["messages"])))     # snapshot: the loop keeps appending
        return FakeStream(self.script.pop(0))


class AgentLoopTests(TestCase):
    def setUp(self):
        cache.clear()
        self.c = Company.objects.create(name="Test Co", code="TST.NS", exchange="NSE", sector="Industrials")

    def run_job(self, script):
        client = FakeClient(script)
        job = {"id": "j1", "kind": "analysis", "status": "running", "company": self.c.code, "question": "",
               "steps": [], "usage": {}, "cost_usd": 0, "report_id": None, "error": None}
        with mock.patch.object(agent, "_client", return_value=client), \
                mock.patch.object(agent, "has_credentials", return_value=True):
            agent._work(job, self.c, "")
        return job, client

    def test_full_run_with_tools_pause_and_a_corrected_answer(self):
        search = NS(type="server_tool_use", id="s1", name="web_search", input={"query": "Test Co news"})
        results = NS(type="web_search_tool_result", tool_use_id="s1",
                     content=[NS(url="https://example.com/a", title="News", page_age="1 day")])
        bad = good_analysis(stop_loss=102.0)
        script = [
            msg([NS(type="thinking", thinking="Need data first."), search, results], stop="pause_turn", searches=1),
            msg([NS(type="tool_use", id="t1", name="get_stock_profile", input={"company_code": "TST.NS"})]),
            msg([NS(type="tool_use", id="t2", name="submit_analysis", input=bad)]),
            msg([NS(type="tool_use", id="t3", name="submit_analysis", input=good_analysis())]),
        ]
        job, client = self.run_job(script)
        self.assertEqual(job["status"], "done", job.get("error"))
        report = AIReport.objects.get(pk=job["report_id"])
        self.assertEqual(report.result["bias"], "BULLISH")
        self.assertEqual(report.price_at, 100.6)
        self.assertEqual(report.sources[0]["url"], "https://example.com/a")
        self.assertEqual(report.usage["turns"], 4)
        self.assertEqual(report.usage["web_search_requests"], 1)
        self.assertAlmostEqual(report.cost_usd, (4 * (1000 * 4 + 200 * 20 + 500 * 0.2)) / 1e6 + 0.01, places=6)
        kinds = [s["kind"] for s in report.steps]
        for k in ("think", "search", "result", "tool", "info", "done"):
            self.assertIn(k, kinds)
        # the data tool really ran and its result went back; the bad answer came back as an error
        sent = client.calls[2]["messages"]
        tool_result = sent[-1]["content"][0]
        self.assertEqual(tool_result["tool_use_id"], "t1")
        self.assertIn("Test Co", tool_result["content"])
        fix = client.calls[3]["messages"][-1]["content"][0]
        self.assertTrue(fix["is_error"])
        # web search + data tools + submit tool, adaptive thinking, fallbacks
        call = client.calls[0]
        self.assertEqual(call["model"], agent.MODEL)
        self.assertEqual(call["tools"][0]["type"], "web_search_20260209")
        self.assertIn("submit_analysis", [t.get("name") for t in call["tools"]])
        self.assertEqual(call["thinking"]["type"], "adaptive")
        self.assertEqual(call["fallbacks"], "default")
        self.assertIsNone(cache.get("ai_prediction:running:analysis"))

    def test_refusal_is_reported(self):
        job, _ = self.run_job([msg([], stop="refusal")])
        self.assertEqual(job["status"], "error")
        self.assertIn("declined", job["error"])
        self.assertFalse(AIReport.objects.exists())

    def test_answer_without_submit_is_nudged_once(self):
        text = NS(type="text", text="Here is my view...")
        job, client = self.run_job([msg([text], stop="end_turn"), msg([text], stop="end_turn")])
        self.assertEqual(job["status"], "error")
        self.assertIn("submit_analysis", client.calls[1]["messages"][-1]["content"])

    def test_missing_key_fails_cleanly(self):
        job = {"id": "j2", "kind": "picks", "status": "running", "company": None, "question": "", "steps": [],
               "usage": {}, "cost_usd": 0, "report_id": None, "error": None}
        with mock.patch.object(agent, "has_credentials", return_value=False):
            agent._work(job, None, "")
        self.assertEqual(job["status"], "error")
        self.assertIn("ANTHROPIC_API_KEY", job["error"])


class ViewTests(TestCase):
    def setUp(self):
        cache.clear()
        self.c = Company.objects.create(name="Test Co", code="TST.NS", exchange="NSE")

    def test_page_and_setup_note(self):
        with mock.patch.object(agent, "has_credentials", return_value=False):
            r = self.client.get("/ai-prediction/")
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "ANTHROPIC_API_KEY")
        self.assertContains(r, "AI Prediction")

    def test_start_needs_key_and_known_company(self):
        with mock.patch.object(agent, "has_credentials", return_value=False):
            r = self.client.post("/api/ai-prediction/analyze/", json.dumps({"company": "TST.NS"}),
                                 content_type="application/json")
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/ai-prediction/analyze/", json.dumps({"company": "NOPE"}), content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_start_runs_in_background(self):
        with mock.patch.object(agent, "has_credentials", return_value=True), \
                mock.patch.object(agent.threading, "Thread") as thread:
            r = self.client.post("/api/ai-prediction/analyze/", json.dumps({"company": "tst"}), content_type="application/json")
            self.assertEqual(r.status_code, 202)
            job_id = r.json()["job"]["id"]
            thread.return_value.start.assert_called_once()
            again = self.client.post("/api/ai-prediction/analyze/", json.dumps({"company": "TST.NS"}),
                                     content_type="application/json")
            self.assertEqual(again.status_code, 409)              # one analysis at a time
        self.assertEqual(self.client.get(f"/api/ai-prediction/job/{job_id}/").json()["status"], "running")
        self.assertEqual(self.client.get("/api/ai-prediction/job/nope/").status_code, 404)

    def test_reports_api(self):
        AIReport.objects.create(kind="analysis", company=self.c, model="claude-opus-5-5", price_at=100.6,
                                result=good_analysis(), cost_usd=0.3)
        with mock.patch.object(evaluate, "_session_for", return_value=(None, pd.DataFrame())):
            d = self.client.get("/api/ai-prediction/reports/?kind=analysis&company=TST.NS&evaluate=1").json()
        self.assertEqual(len(d["reports"]), 1)
        self.assertEqual(d["reports"][0]["bias"], "BULLISH")
        self.assertEqual(d["reports"][0]["outcome"]["status"], "WAITING")

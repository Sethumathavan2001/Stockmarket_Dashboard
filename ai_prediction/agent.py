"""
Claude agent behind the AI Prediction page.

A job runs in a background thread:  Claude (Messages API, streaming, adaptive thinking) gets
the live web search tool plus this app's data tools (tools.py), researches, and finishes by
calling submit_analysis / submit_picks.  That input is validated; a bad one goes back to
Claude as an error so it can fix it.  Progress (searches, data reads, thinking summaries) is
written to the cache so the page can poll it; the finished answer is stored as an AIReport.

Credentials: ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) from the environment or the
project's .env file - never stored in code or the database.
"""
import json
import logging
import os
import threading
import time as _time
import traceback
import uuid
from datetime import datetime

from django.conf import settings
from django.core.cache import cache
from django.db import close_old_connections

from prediction.market_time import IST, get_market_status

from . import tools
from .models import AIReport

logger = logging.getLogger(__name__)

MODEL = getattr(settings, "AI_PREDICTION_MODEL", "claude-opus-5-5")
EFFORT = getattr(settings, "AI_PREDICTION_EFFORT", "high")
MAX_TURNS = 16                  # model round trips per job
MAX_DATA_CALLS = 30             # data-tool calls per job
MAX_SEARCHES = {"analysis": 8, "picks": 10}
JOB_TTL = 6 * 3600
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens: input, output, cache write (5 min), cache read.  Web search: $10 / 1000.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 5.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 6.25, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 6.25, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 2.5, 0.20),
    "claude-fable-5-1": (10.0, 50.0, 12.5, 0.25),
}
WEB_SEARCH_USD = 0.01

_JOBS_LOCK = threading.Lock()


class AgentError(Exception):
    pass


def has_credentials():
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _client():
    import anthropic

    return anthropic.Anthropic(max_retries=3, timeout=900)


# ==========================================================================
# prompts (kept byte-stable so they are served from the prompt cache)
# ==========================================================================
COMMON = """You are the intraday research analyst inside a personal NSE (India) stock dashboard.
The user trades intraday (positions squared off by 15:15 IST) with a paper-trading account and wants
an honest, data-backed view - not hype.

You have two kinds of evidence:
- This app's own data, through the get_* tools: the user's database of daily and 5-minute candles
  (topped up from Yahoo where the database lags), indicators, the app's tested candle-pattern engine
  and its machine-learning forecasts. The database can lag by a few sessions; get_live_quote and
  get_intraday_candles include today's candles.
- The web, through web_search: news, results, corporate actions, analyst and brokerage calls,
  sector and global cues (US markets, GIFT Nifty, crude, dollar-rupee, FII/DII flows).

How to work:
- Gather the data before forming a view. Call independent data tools together in one turn.
- Search for what can move the stock today: company news from the last few days, results dates,
  block/bulk deals, regulatory news, sector news and the overall market mood. Prefer recent,
  reputable sources and note the date of what you cite.
- Short-horizon price prediction is hard; most signals are close to coin flips after costs
  (about 0.08 % round trip for intraday). Say so when the evidence is mixed. The app's pattern and
  ML tools report tested accuracy against a baseline - weigh them by that, and ignore those with no edge.
- Levels must be realistic for intraday: stops and targets sized from ATR and today's range, at
  real support/resistance (previous day high/low/close, opening range, VWAP, pivots, CPR, round
  numbers). Use a reward-to-risk of at least 1.5 when you recommend a trade.
- All prices in INR, all times IST.
- Cite only pages your searches actually returned, with their exact URLs.
- When the research is complete, call the submit tool exactly once with your answer. Keep text
  fields concise and specific (numbers, levels, dates); no generic disclaimers."""

ANALYSIS_SYSTEM = COMMON + """

Task: a complete intraday analysis of ONE stock, chosen by the user. Cover trend and momentum on
daily and 5-minute charts, where price sits versus VWAP / previous-day levels / opening range,
volume, the app's pattern and ML evidence, company and sector news, and the market backdrop.
Then give a trade plan (or NO_TRADE when there is no clear edge), price outlooks for the next
30 minutes, 1 hour, 2 hours and the close (only horizons still inside today's session; if the
market is closed, plan for the next session), the key levels and two or three scenarios."""

PICKS_SYSTEM = COMMON + """

Task: recommend the best intraday trade ideas for the current session from the stocks in this
app's database only (use the codes exactly as screen_stocks returns them). Read the market and
sector mood first, screen the database from a few angles (for example volume_surge, momentum,
near_20d_high / near_20d_low, narrow_range_compression), look for stocks with fresh news or
catalysts, then check today's price action of your shortlist with get_live_quote or
get_intraday_candles before committing. Return 4-6 picks, longs and shorts as the market
suggests, each liquid (turnover >= 10 crore a day) with entry, stop and target, plus any
stocks worth avoiding today (for example results today or extreme gaps)."""


def _now_line():
    now = datetime.now(IST)
    return f"Now: {now:%A %d %B %Y, %H:%M} IST. NSE market status: {get_market_status()}."


def _analysis_prompt(company, question):
    text = (f"{_now_line()}\n\nAnalyse {company.code} ({company.name}; sector: {company.sector or 'n/a'}; "
            f"industry: {company.industry or 'n/a'}) for intraday trading.")
    if question:
        text += f"\n\nThe user adds: {question}"
    return text


def _picks_prompt(question):
    text = f"{_now_line()}\n\nFind the best intraday trades for the current (or, if the market is closed, the next) session."
    if question:
        text += f"\n\nThe user adds: {question}"
    return text


# ==========================================================================
# job state (cache)
# ==========================================================================
def _key(job_id):
    return f"ai_prediction:job:{job_id}"


def get_job(job_id):
    return cache.get(_key(job_id))


def _save(job):
    cache.set(_key(job["id"]), job, JOB_TTL)


def running_job(kind):
    job_id = cache.get(f"ai_prediction:running:{kind}")
    job = get_job(job_id) if job_id else None
    return job if job and job["status"] == "running" else None


class Run:
    """Mutable state of one job; every change is pushed to the cache for the page to poll."""

    def __init__(self, job):
        self.job = job
        self.sources = {}                 # url -> {title, url, page_age}
        self.data_calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 0, "web_search_requests": 0, "turns": 0}
        self.cost = 0.0
        self.served_by = MODEL

    def step(self, kind, text, **extra):
        self.job["steps"].append({"t": datetime.now(IST).strftime("%H:%M:%S"), "kind": kind, "text": text[:600], **extra})
        self.job["usage"], self.job["cost_usd"] = self.usage, round(self.cost, 4)
        _save(self.job)

    def add_usage(self, message):
        u = message.usage
        model = getattr(message, "model", None) or MODEL
        self.served_by = model
        p_in, p_out, p_cw, p_cr = PRICES.get(model, PRICES["claude-opus-5-5"])
        vals = {k: int(getattr(u, k, 0) or 0) for k in
                ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
        searches = int(getattr(getattr(u, "server_tool_use", None), "web_search_requests", 0) or 0)
        for k, v in vals.items():
            self.usage[k] += v
        self.usage["web_search_requests"] += searches
        self.usage["turns"] += 1
        self.cost += (vals["input_tokens"] * p_in + vals["output_tokens"] * p_out
                      + vals["cache_creation_input_tokens"] * p_cw + vals["cache_read_input_tokens"] * p_cr) / 1e6
        self.cost += searches * WEB_SEARCH_USD


# ==========================================================================
# the agent loop
# ==========================================================================
def _describe_block(run, block):
    """Turn a finished content block into a progress step (and collect web sources)."""
    t = getattr(block, "type", "")
    if t == "server_tool_use":
        inp = getattr(block, "input", None) or {}
        if getattr(block, "name", "") == "web_search":
            run.step("search", f"Web search: {inp.get('query', '')}")
    elif t == "web_search_tool_result":
        content = getattr(block, "content", None)
        if isinstance(content, list):
            for r in content:
                url = getattr(r, "url", None)
                if url and url not in run.sources:
                    run.sources[url] = {"title": getattr(r, "title", "") or url, "url": url,
                                        "page_age": getattr(r, "page_age", None)}
            run.step("result", f"{len(content)} pages found", pages=[
                {"title": getattr(r, "title", ""), "url": getattr(r, "url", "")} for r in content[:5]])
        else:
            run.step("error", f"Web search failed: {getattr(content, 'error_code', 'unknown error')}")
    elif t == "tool_use":
        args = getattr(block, "input", None) or {}
        if block.name in tools.TOOL_LABELS:
            detail = ", ".join(f"{k}={v}" for k, v in args.items()) if isinstance(args, dict) else ""
            run.step("tool", f"{tools.TOOL_LABELS[block.name]}" + (f" ({detail})" if detail else ""))
    elif t == "thinking":
        text = (getattr(block, "thinking", "") or "").strip()
        if text:
            run.step("think", text)
    elif t == "text":
        text = (getattr(block, "text", "") or "").strip()
        if text:
            run.step("text", text)
    elif t == "fallback":
        frm, to = getattr(block, "from_", None), getattr(block, "to", None)
        run.step("info", f"{getattr(frm, 'model', '?')} declined; {getattr(to, 'model', '?')} continued.")


def run_agent(run, kind, system, user_text, data_tool_names, submit_name, validate):
    import anthropic

    client = _client()
    tool_defs = ([{"type": "web_search_20260209", "name": "web_search", "max_uses": MAX_SEARCHES[kind],
                   "user_location": {"type": "approximate", "country": "IN", "timezone": "Asia/Kolkata"}}]
                 + [tools.DATA_TOOLS[n][1] for n in data_tool_names] + [tools.SUBMIT_TOOLS[submit_name]])
    messages = [{"role": "user", "content": user_text}]
    nudged = json_retries = 0
    turn = 0
    while turn < MAX_TURNS:
        try:
            with client.beta.messages.stream(
                model=MODEL, max_tokens=64000, system=system, tools=tool_defs, messages=messages,
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": EFFORT}, cache_control={"type": "ephemeral"},
                betas=[FALLBACK_BETA], fallbacks="default",
            ) as stream:
                for event in stream:
                    if event.type == "content_block_stop":
                        _describe_block(run, event.content_block)
                response = stream.get_final_message()
            json_retries = 0
        except ValueError:
            # a client-tool input the SDK could not parse at all: no tool_use id to answer, re-issue the turn
            json_retries += 1
            if json_retries > 2:
                raise AgentError("Claude produced an unreadable tool call three times in a row.")
            continue
        except anthropic.AuthenticationError:
            raise AgentError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY in the .env file.")
        except anthropic.PermissionDeniedError as exc:
            raise AgentError(f"The API key is not allowed to do this: {exc.message}")
        except anthropic.RateLimitError:
            raise AgentError("Anthropic rate limit reached - wait a minute and try again.")
        except anthropic.BadRequestError as exc:
            raise AgentError(f"Request rejected by the API: {exc.message}")
        except anthropic.APIStatusError as exc:
            raise AgentError(f"Anthropic API error {exc.status_code}: {exc.message}")
        except anthropic.APIConnectionError:
            raise AgentError("Could not reach the Anthropic API - check the internet connection.")

        turn += 1
        run.add_usage(response)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            raise AgentError("Claude declined this request.")
        if response.stop_reason == "pause_turn":              # a long server-tool turn: resume it
            continue
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason == "max_tokens":
            raise AgentError("The answer was cut off (output limit reached).")
        if not tool_uses:
            if nudged:
                raise AgentError("Claude finished without submitting an answer.")
            nudged = 1
            messages.append({"role": "user", "content": f"Please call {submit_name} now with your final answer."})
            continue

        results, final = [], None
        for b in tool_uses:
            if b.name == submit_name:
                errors = validate(b.input) if isinstance(b.input, dict) else ["input must be a JSON object"]
                if errors:
                    run.step("info", "Answer failed validation, asking Claude to fix: " + "; ".join(errors[:4]))
                    results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                    "content": "Fix these problems and call the tool again:\n- " + "\n- ".join(errors[:20])})
                else:
                    final = b.input
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": "Saved."})
            elif b.name in data_tool_names:
                run.data_calls += 1
                if run.data_calls > MAX_DATA_CALLS:
                    payload, err = {"error": "Data-call budget for this job is used up - submit your answer now."}, True
                else:
                    payload, err = tools.run_data_tool(b.name, b.input)
                if err:
                    run.step("error", f"{tools.TOOL_LABELS.get(b.name, b.name)}: {payload['error']}")
                results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": err,
                                "content": json.dumps(payload, default=str, separators=(",", ":"))})
            else:
                results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                "content": f"Unknown tool {b.name}."})
        if final is not None:
            return final
        messages.append({"role": "user", "content": results})
    raise AgentError(f"No answer after {MAX_TURNS} rounds.")


# ==========================================================================
# public: start jobs
# ==========================================================================
def _start(kind, company, question):
    with _JOBS_LOCK:
        busy = running_job(kind)
        if busy:
            return busy, False
        job = {"id": uuid.uuid4().hex, "kind": kind, "status": "running", "company": company.code if company else None,
               "question": question, "started": datetime.now(IST).isoformat(), "steps": [], "usage": {},
               "cost_usd": 0, "report_id": None, "error": None, "model": MODEL}
        _save(job)
        cache.set(f"ai_prediction:running:{kind}", job["id"], JOB_TTL)
    threading.Thread(target=_work, args=(job, company, question), daemon=True, name=f"ai-{kind}").start()
    return job, True


def start_analysis(company, question=""):
    return _start(AIReport.KIND_ANALYSIS, company, question)


def start_picks(question=""):
    return _start(AIReport.KIND_PICKS, None, question)


def _work(job, company, question):
    run = Run(job)
    started = _time.monotonic()
    try:
        if not has_credentials():
            raise AgentError("No Anthropic API key. Add ANTHROPIC_API_KEY=... to the .env file next to manage.py "
                             "and restart the server.")
        if job["kind"] == AIReport.KIND_ANALYSIS:
            run.step("info", f"Claude ({MODEL}) is analysing {company.code}...")
            result = run_agent(run, "analysis", ANALYSIS_SYSTEM, _analysis_prompt(company, question),
                               tools.ANALYSIS_DATA_TOOLS, "submit_analysis",
                               lambda d: tools.validate_analysis(d, company))
            price = result.get("price_at_analysis")
        else:
            run.step("info", f"Claude ({MODEL}) is scanning the market for intraday trades...")
            result = run_agent(run, "picks", PICKS_SYSTEM, _picks_prompt(question), tools.PICKS_DATA_TOOLS,
                               "submit_picks", tools.validate_picks)
            price = None
        # keep the cited sources that the searches really returned; list every page searched as well
        cited = [s for s in result.get("sources", []) if s.get("url") in run.sources]
        report = AIReport.objects.create(
            kind=job["kind"], company=company, question=question, model=run.served_by, price_at=price,
            result=result, sources=cited or list(run.sources.values())[:12], steps=job["steps"],
            usage=run.usage, cost_usd=round(run.cost, 4))
        report.steps = job["steps"] + [{"t": datetime.now(IST).strftime("%H:%M:%S"), "kind": "done",
                                        "text": f"Done in {_time.monotonic() - started:.0f}s."}]
        report.save(update_fields=["steps"])
        job.update(status="done", report_id=report.pk, steps=report.steps, usage=run.usage, cost_usd=round(run.cost, 4))
        _save(job)
    except AgentError as exc:
        run.step("error", str(exc))
        job.update(status="error", error=str(exc))
        _save(job)
    except Exception as exc:  # never leave the page polling a dead job
        logger.error("AI prediction job failed:\n%s", traceback.format_exc())
        run.step("error", f"{type(exc).__name__}: {exc}")
        job.update(status="error", error=f"{type(exc).__name__}: {exc}")
        _save(job)
    finally:
        cache.delete(f"ai_prediction:running:{job['kind']}")
        close_old_connections()

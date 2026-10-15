"""
AI Prediction page (Claude) and its API.

  GET  /ai-prediction/                              the page
  POST /api/ai-prediction/analyze/                  {"company": code, "question": ""} -> start a stock analysis
  POST /api/ai-prediction/picks/                    {"question": ""}                  -> start intraday picks
  GET  /api/ai-prediction/job/<id>/                 progress / result of a job
  GET  /api/ai-prediction/running/                  jobs still running (to re-attach after a reload)
  GET  /api/ai-prediction/reports/?kind=&company=   saved calls, newest first (with outcomes if evaluate=1)
  GET  /api/ai-prediction/report/<id>/              one saved call + how it played out
  GET  /api/ai-prediction/candles/<code>/           the last two sessions of 5-minute candles for the chart
"""
import json
import logging

from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from market.models import Company

from . import agent, evaluate, tools
from .models import AIReport

logger = logging.getLogger(__name__)
MAX_QUESTION = 1000


def _body(request):
    try:
        data = json.loads(request.body or b"{}")
    except ValueError:
        data = {}
    return data if isinstance(data, dict) else {}


def report_json(r, with_eval=False):
    out = {"id": r.pk, "kind": r.kind, "company": r.company.code if r.company_id else None,
           "name": r.company.name if r.company_id else None, "created_at": r.created_at.isoformat(),
           "question": r.question, "model": r.model, "price_at": r.price_at, "result": r.result,
           "sources": r.sources, "steps": r.steps, "usage": r.usage, "cost_usd": r.cost_usd}
    if with_eval:
        try:
            out["outcome"] = evaluate.evaluate_report(r)
        except Exception as exc:  # outcome is a nice-to-have; never fail the request for it
            logger.warning("Could not evaluate AI report %s: %s", r.pk, exc)
            out["outcome"] = {"status": "UNKNOWN", "error": str(exc)}
    return out


@require_GET
@ensure_csrf_cookie
def page(request):
    companies = Company.objects.exclude(sector="ETF").order_by("code").values("code", "name", "sector")
    latest_picks = AIReport.objects.filter(kind=AIReport.KIND_PICKS).first()
    return render(request, "ai_prediction/ai_prediction.html", {
        "companies": list(companies),
        "initial_company": request.GET.get("company", ""),
        "has_key": agent.has_credentials(),
        "model": agent.MODEL,
        "latest_picks": report_json(latest_picks) if latest_picks else None,
    })


@require_POST
def start_analysis(request):
    data = _body(request)
    try:
        company = tools.find_company(data.get("company"))
    except tools.ToolError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    if not agent.has_credentials():
        return JsonResponse({"error": "No Anthropic API key configured (see the setup note on the page)."}, status=400)
    job, started = agent.start_analysis(company, (data.get("question") or "").strip()[:MAX_QUESTION])
    return JsonResponse({"job": job, "started": started}, status=202 if started else 409)


@require_POST
def start_picks(request):
    if not agent.has_credentials():
        return JsonResponse({"error": "No Anthropic API key configured (see the setup note on the page)."}, status=400)
    job, started = agent.start_picks((_body(request).get("question") or "").strip()[:MAX_QUESTION])
    return JsonResponse({"job": job, "started": started}, status=202 if started else 409)


@require_GET
def job_status(request, job_id):
    job = agent.get_job(job_id)
    if job is None:
        return JsonResponse({"error": "Unknown or expired job."}, status=404)
    out = dict(job)
    if job["status"] == "done" and job.get("report_id"):
        report = AIReport.objects.filter(pk=job["report_id"]).first()
        if report:
            out["report"] = report_json(report)
    return JsonResponse(out)


@require_GET
def running(request):
    return JsonResponse({k: agent.running_job(k) for k in (AIReport.KIND_ANALYSIS, AIReport.KIND_PICKS)})


@require_GET
def reports(request):
    qs = AIReport.objects.select_related("company")
    kind = request.GET.get("kind")
    if kind:
        qs = qs.filter(kind=kind)
    if request.GET.get("company"):
        try:
            qs = qs.filter(company=tools.find_company(request.GET["company"]))
        except tools.ToolError as exc:
            return JsonResponse({"error": str(exc)}, status=400)
    with_eval = request.GET.get("evaluate") == "1"
    rows = []
    for r in qs[:20]:
        full = report_json(r, with_eval)
        res = r.result or {}
        rows.append({"id": r.pk, "kind": r.kind, "company": full["company"], "created_at": full["created_at"],
                     "model": r.model, "cost_usd": r.cost_usd, "price_at": r.price_at,
                     "bias": res.get("bias") or res.get("market_bias"), "confidence": res.get("confidence"),
                     "headline": res.get("headline") or res.get("market_summary", "")[:160],
                     "plan": res.get("trade_plan"), "outcome": full.get("outcome")})
    return JsonResponse({"reports": rows})


@require_GET
def report_detail(request, report_id):
    r = get_object_or_404(AIReport.objects.select_related("company"), pk=report_id)
    return JsonResponse(report_json(r, with_eval=request.GET.get("evaluate", "1") == "1"))


@require_GET
def candles(request, company_code):
    from patterns.engine import load_candles

    try:
        company = tools.find_company(company_code)
    except tools.ToolError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    with tools.DATA_LOCK:
        df = load_candles(company, "5m")
    if df.empty:
        return JsonResponse({"company": company.code, "candles": []})
    days = sorted(df["timestamp"].dt.date.unique())[-2:]
    df = df[df["timestamp"].dt.date.isin(days)]
    return JsonResponse({"company": company.code, "name": company.name, "candles": [
        {"x": t.isoformat(), "o": o, "h": h, "l": l, "c": c, "v": v}
        for t, o, h, l, c, v in zip(df["timestamp"], df["open"], df["high"], df["low"], df["close"], df["volume"])]})

"""Internal telemetry dashboard — staff-only views."""

import json
from collections import Counter
from datetime import timedelta

from django.conf import settings
from django.contrib.auth.decorators import user_passes_test
from django.db.models import Avg, Count, Q, Sum
from django.db.models.functions import TruncDate
from django.shortcuts import render
from django.utils import timezone

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAdminUser
from rest_framework.response import Response

from accounts.models import Subscription, User
from chat.models import TraceLog, UsageLog

TIER_PRICES = {"founder": 19, "pro": 29, "team": 79, "enterprise": 199}
PAID_TIERS = list(TIER_PRICES.keys())


def _get_days(request):
    try:
        return min(int(request.GET.get("days", 30)), 365)
    except (TypeError, ValueError):
        return 30


# ---------------------------------------------------------------------------
# Page view
# ---------------------------------------------------------------------------

@user_passes_test(lambda u: u.is_staff, login_url="/login/")
def dashboard_view(request):
    return render(request, "internal_dashboard.html")


# ---------------------------------------------------------------------------
# API: Overview KPIs
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsAdminUser])
def api_overview(request):
    days = _get_days(request)
    now = timezone.now()
    today = now.date()

    total_users = User.objects.count()
    active_today = User.objects.filter(last_active_at__date=today).count()

    day_usage = UsageLog.objects.filter(date=today).aggregate(
        requests=Sum("request_count"),
        errors=Sum("error_count"),
    )

    avg_latency = TraceLog.objects.filter(
        created_at__gte=now - timedelta(hours=24),
        total_time_ms__isnull=False,
    ).aggregate(avg=Avg("total_time_ms"))["avg"]

    mrr = sum(
        User.objects.filter(tier=t).count() * p for t, p in TIER_PRICES.items()
    )

    paid = User.objects.filter(tier__in=PAID_TIERS).count()
    conversion = round(paid / total_users * 100, 1) if total_users else 0

    return Response({
        "total_users": total_users,
        "active_today": active_today,
        "requests_today": day_usage["requests"] or 0,
        "errors_today": day_usage["errors"] or 0,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency else None,
        "mrr": mrr,
        "conversion_rate": conversion,
    })


# ---------------------------------------------------------------------------
# API: Users
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsAdminUser])
def api_users(request):
    days = _get_days(request)
    since = timezone.now() - timedelta(days=days)

    signups = (
        User.objects.filter(date_joined__gte=since)
        .annotate(date=TruncDate("date_joined"))
        .values("date")
        .annotate(count=Count("id"))
        .order_by("date")
    )

    tiers = (
        User.objects.values("tier")
        .annotate(count=Count("id"))
        .order_by("tier")
    )

    industries = (
        User.objects.exclude(industry="")
        .values("industry")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    roles = (
        User.objects.exclude(role="")
        .values("role")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    experience = (
        User.objects.exclude(experience_level="")
        .values("experience_level")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    active_trend = (
        User.objects.filter(last_active_at__gte=since, last_active_at__isnull=False)
        .annotate(date=TruncDate("last_active_at"))
        .values("date")
        .annotate(count=Count("id"))
        .order_by("date")
    )

    total = User.objects.count()
    verified = User.objects.filter(email_verified=True).count()

    return Response({
        "signups": [{"date": str(s["date"]), "count": s["count"]} for s in signups],
        "tiers": list(tiers),
        "industries": list(industries),
        "roles": list(roles),
        "experience": list(experience),
        "active_trend": [{"date": str(a["date"]), "count": a["count"]} for a in active_trend],
        "verification_rate": round(verified / total * 100, 1) if total else 0,
        "verified_count": verified,
        "total_count": total,
    })


# ---------------------------------------------------------------------------
# API: Usage
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsAdminUser])
def api_usage(request):
    days = _get_days(request)
    since = timezone.now().date() - timedelta(days=days)

    daily_requests = (
        UsageLog.objects.filter(date__gte=since)
        .values("date")
        .annotate(total=Sum("request_count"))
        .order_by("date")
    )

    # Aggregate domain_counts JSON in Python (fine at alpha scale)
    domain_totals = Counter()
    for log in UsageLog.objects.filter(date__gte=since).exclude(domain_counts__isnull=True):
        if log.domain_counts:
            for domain, count in log.domain_counts.items():
                domain_totals[domain] += count

    daily_tokens = (
        UsageLog.objects.filter(date__gte=since)
        .values("date")
        .annotate(input=Sum("tokens_input"), output=Sum("tokens_output"))
        .order_by("date")
    )

    daily_errors = (
        UsageLog.objects.filter(date__gte=since)
        .values("date")
        .annotate(errors=Sum("error_count"), requests=Sum("request_count"))
        .order_by("date")
    )

    return Response({
        "daily_requests": [
            {"date": str(d["date"]), "count": d["total"]} for d in daily_requests
        ],
        "domain_popularity": domain_totals.most_common(20),
        "daily_tokens": [
            {
                "date": str(d["date"]),
                "input": d["input"] or 0,
                "output": d["output"] or 0,
            }
            for d in daily_tokens
        ],
        "daily_errors": [
            {
                "date": str(d["date"]),
                "errors": d["errors"] or 0,
                "requests": d["requests"] or 0,
                "rate": round(
                    (d["errors"] or 0) / d["requests"] * 100, 2
                ) if d["requests"] else 0,
            }
            for d in daily_errors
        ],
    })


# ---------------------------------------------------------------------------
# API: Performance
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsAdminUser])
def api_performance(request):
    days = _get_days(request)
    since = timezone.now() - timedelta(days=days)
    traces = TraceLog.objects.filter(created_at__gte=since)

    latency_trend = (
        traces.filter(total_time_ms__isnull=False)
        .annotate(date=TruncDate("created_at"))
        .values("date")
        .annotate(avg_ms=Avg("total_time_ms"), count=Count("id"))
        .order_by("date")
    )

    stage_avgs = traces.aggregate(
        safety=Avg("safety_time_ms"),
        intuition=Avg("intuition_time_ms"),
        reasoner=Avg("reasoner_time_ms"),
        verifier=Avg("verifier_time_ms"),
        lm=Avg("lm_time_ms"),
    )

    gate_trend = (
        traces.annotate(date=TruncDate("created_at"))
        .values("date")
        .annotate(
            total=Count("id"),
            passed=Count("id", filter=Q(gate_passed=True)),
        )
        .order_by("date")
    )

    error_stages = (
        traces.exclude(error_stage="")
        .values("error_stage")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    return Response({
        "latency_trend": [
            {"date": str(r["date"]), "avg_ms": round(r["avg_ms"], 1), "count": r["count"]}
            for r in latency_trend
        ],
        "stage_breakdown": {
            k: round(v, 1) if v else 0 for k, v in stage_avgs.items()
        },
        "gate_trend": [
            {
                "date": str(g["date"]),
                "total": g["total"],
                "passed": g["passed"],
                "rate": round(g["passed"] / g["total"] * 100, 1) if g["total"] else 0,
            }
            for g in gate_trend
        ],
        "error_stages": list(error_stages),
    })


# ---------------------------------------------------------------------------
# API: Business
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsAdminUser])
def api_business(request):
    days = _get_days(request)
    since = timezone.now().date() - timedelta(days=days)

    # Revenue by tier
    revenue = {}
    for tier, price in TIER_PRICES.items():
        count = User.objects.filter(tier=tier).count()
        revenue[tier] = {"count": count, "mrr": count * price}

    # Conversion funnel
    total = User.objects.count()
    verified = User.objects.filter(email_verified=True).count()
    queried = User.objects.filter(total_queries__gt=0).count()
    paid = User.objects.filter(tier__in=PAID_TIERS).count()

    # Churn
    churning = Subscription.objects.filter(cancel_at_period_end=True).count()
    active_subs = Subscription.objects.filter(status="active").count()

    # Founder slots
    founder_count = User.objects.filter(tier="founder").count()

    # Feature adoption (paid users only)
    tool_usage = Counter()
    paid_users = User.objects.filter(tier__in=PAID_TIERS)
    for log in UsageLog.objects.filter(
        date__gte=since, user__in=paid_users
    ).exclude(domain_counts__isnull=True):
        if log.domain_counts:
            for domain, count in log.domain_counts.items():
                tool_usage[domain] += count

    return Response({
        "revenue": revenue,
        "funnel": {
            "total": total,
            "verified": verified,
            "queried": queried,
            "paid": paid,
        },
        "churn": {
            "cancelling": churning,
            "active_subscriptions": active_subs,
        },
        "founder_slots": {"used": founder_count, "total": 100},
        "feature_adoption": tool_usage.most_common(15),
    })


# ---------------------------------------------------------------------------
# API: AI Insights (POST — calls Anthropic)
# ---------------------------------------------------------------------------

@api_view(["POST"])
@permission_classes([IsAdminUser])
def api_insights(request):
    prompt = request.data.get(
        "prompt",
        "Analyze this data and provide actionable insights for product growth, "
        "user retention, and feature development.",
    )
    snapshot = _build_data_snapshot()

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            system=(
                "You are an analytics advisor for Svend, a decision science SaaS. "
                "Svend provides statistical analysis, DOE, SPC, Bayesian reasoning, "
                "forecasting, A3 reports, value stream mapping, and other quality/"
                "operations tools. Tiers: Free ($0), Founder ($19/mo), Pro ($29/mo), "
                "Team ($79/mo), Enterprise ($199/mo). Target audience: engineers, "
                "analysts, managers, and consultants in manufacturing, healthcare, "
                "tech, and consulting. Analyze the data provided and give specific, "
                "actionable insights. Cite numbers. Prioritize by impact. Use "
                "markdown formatting with headers and bullet points."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Dashboard data snapshot:\n```json\n"
                    f"{json.dumps(snapshot, indent=2, default=str)}\n```\n\n{prompt}"
                ),
            }],
        )
        return Response({"insights": response.content[0].text})
    except Exception as e:
        return Response({"error": str(e)}, status=500)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_data_snapshot(days=30):
    """Anonymized aggregate snapshot for AI analysis."""
    now = timezone.now()
    since = now - timedelta(days=days)
    since_date = since.date()

    total_users = User.objects.count()
    verified = User.objects.filter(email_verified=True).count()
    queried = User.objects.filter(total_queries__gt=0).count()
    paid = User.objects.filter(tier__in=PAID_TIERS).count()

    tier_dist = dict(
        User.objects.values_list("tier")
        .annotate(c=Count("id"))
        .values_list("tier", "c")
    )

    mrr = sum(tier_dist.get(t, 0) * p for t, p in TIER_PRICES.items())

    usage = UsageLog.objects.filter(date__gte=since_date).aggregate(
        total_requests=Sum("request_count"),
        total_errors=Sum("error_count"),
        total_tokens_in=Sum("tokens_input"),
        total_tokens_out=Sum("tokens_output"),
    )

    perf = TraceLog.objects.filter(created_at__gte=since).aggregate(
        avg_latency=Avg("total_time_ms"),
        total_traces=Count("id"),
        gates_passed=Count("id", filter=Q(gate_passed=True)),
        fallbacks=Count("id", filter=Q(fallback_used=True)),
    )

    domain_totals = Counter()
    for log in UsageLog.objects.filter(
        date__gte=since_date
    ).exclude(domain_counts__isnull=True):
        if log.domain_counts:
            for d, c in log.domain_counts.items():
                domain_totals[d] += c

    industries = dict(
        User.objects.exclude(industry="")
        .values_list("industry")
        .annotate(c=Count("id"))
        .values_list("industry", "c")
    )
    roles = dict(
        User.objects.exclude(role="")
        .values_list("role")
        .annotate(c=Count("id"))
        .values_list("role", "c")
    )

    churning = Subscription.objects.filter(cancel_at_period_end=True).count()

    signups = list(
        User.objects.filter(date_joined__gte=since)
        .annotate(date=TruncDate("date_joined"))
        .values("date")
        .annotate(count=Count("id"))
        .order_by("date")
        .values_list("date", "count")
    )

    return {
        "period_days": days,
        "users": {
            "total": total_users,
            "verified": verified,
            "with_queries": queried,
            "paid": paid,
            "tier_distribution": tier_dist,
            "industries": industries,
            "roles": roles,
        },
        "revenue": {
            "mrr": mrr,
            "churning_subscriptions": churning,
            "founder_slots_used": tier_dist.get("founder", 0),
            "founder_slots_total": 100,
        },
        "usage": {k: v or 0 for k, v in usage.items()},
        "performance": {
            "avg_latency_ms": (
                round(perf["avg_latency"], 1) if perf["avg_latency"] else None
            ),
            "total_traces": perf["total_traces"],
            "gate_pass_rate": (
                round(perf["gates_passed"] / perf["total_traces"] * 100, 1)
                if perf["total_traces"] else None
            ),
            "fallback_rate": (
                round(perf["fallbacks"] / perf["total_traces"] * 100, 1)
                if perf["total_traces"] else None
            ),
        },
        "top_domains": domain_totals.most_common(10),
        "daily_signups": [{"date": str(d), "count": c} for d, c in signups],
    }

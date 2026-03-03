"""Internal telemetry dashboard — staff and org-admin views."""

import json
import logging
from collections import Counter
from datetime import timedelta

from django.conf import settings
from django.contrib.auth.decorators import user_passes_test
from django.db.models import Avg, Count, Q, Sum
from django.db.models.functions import TruncDate
from django.shortcuts import render
from django.utils import timezone

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import BasePermission, IsAdminUser
from rest_framework.response import Response

from accounts.models import Subscription, User
from api.models import (
    AutomationLog,
    AutomationRule,
    AutopilotReport,
    BlogPost,
    BlogView,
    CRMLead,
    EmailCampaign,
    OutreachEnrollment,
    OutreachSequence,
    SiteVisit,
    WhitePaper,
    WhitePaperDownload,
    Experiment,
    ExperimentAssignment,
    Feedback,
)
from chat.models import EventLog, TraceLog, UsageLog

logger = logging.getLogger(__name__)

TIER_PRICES = {"founder": 19, "pro": 29, "team": 79, "enterprise": 199}
PAID_TIERS = list(TIER_PRICES.keys())

# Internal/test accounts excluded from all analytics (non-staff accounts
# that shouldn't inflate customer metrics — e.g. team members, test users).
INTERNAL_USERNAMES = {"rtWzrd", "adamlbowden"}

# Tenant slugs whose owner/admin members get internal dashboard access.
INTERNAL_TENANT_SLUGS = {"svend"}


def can_access_internal(user):
    """Return True if user is staff OR an owner/admin of an internal tenant."""
    if not user or not user.is_authenticated:
        return False
    if user.is_staff:
        return True
    return user.memberships.filter(
        tenant__slug__in=INTERNAL_TENANT_SLUGS,
        role__in=("owner", "admin"),
        is_active=True,
    ).exists()


class IsInternalUser(BasePermission):
    """DRF permission: staff or internal-tenant admin."""

    def has_permission(self, request, view):
        return can_access_internal(request.user)


# ---------------------------------------------------------------------------
# Helpers — staff exclusion
# ---------------------------------------------------------------------------

def _get_days(request):
    try:
        return min(int(request.GET.get("days", 30)), 365)
    except (TypeError, ValueError):
        return 30


def _customers():
    """Real customers — excludes staff/internal accounts."""
    return User.objects.filter(is_staff=False).exclude(username__in=INTERNAL_USERNAMES)


def _resolve_recipients(target):
    """Resolve a target string to a queryset of Users (with emails).

    Returns (queryset | None, error_string | None).
    Custom email addresses (containing @) return None for queryset.
    """
    now = timezone.now()
    base = _customers().exclude(email="")

    if target == "all":
        return base, None
    elif target.startswith("tier:"):
        tier = target.split(":", 1)[1]
        return base.filter(tier=tier), None
    elif target.startswith("active:"):
        days = int(target.split(":", 1)[1].rstrip("d"))
        cutoff = now - timedelta(days=days)
        return base.filter(last_active_at__gte=cutoff), None
    elif target.startswith("inactive:"):
        days = int(target.split(":", 1)[1].rstrip("d"))
        cutoff = now - timedelta(days=days)
        return base.filter(
            Q(last_active_at__lt=cutoff) | Q(last_active_at__isnull=True)
        ), None
    elif target == "has_queries":
        return base.filter(total_queries__gt=0), None
    elif target == "no_queries":
        return base.filter(total_queries=0), None
    elif target.startswith("new:"):
        days = int(target.split(":", 1)[1].rstrip("d"))
        cutoff = now - timedelta(days=days)
        return base.filter(date_joined__gte=cutoff), None
    elif target.startswith("domain:"):
        domain = target.split(":", 1)[1]
        cutoff = now - timedelta(days=90)
        user_ids = (
            UsageLog.objects.filter(
                date__gte=cutoff.date(),
                domain_counts__has_key=domain,
            )
            .values_list("user_id", flat=True)
            .distinct()
        )
        return base.filter(id__in=user_ids), None
    elif "@" in target:
        return None, None  # custom email — handled by caller
    else:
        return None, "Invalid recipient target."


def _markdown_to_html(text):
    """Convert markdown to HTML. No external dependencies."""
    import re

    # Fenced code blocks — stash before processing
    code_blocks = []
    def _stash_code(m):
        code_blocks.append(m.group(1))
        return f"\x00CODE{len(code_blocks) - 1}\x00"
    text = re.sub(r"```(?:\w*)\n([\s\S]*?)```", _stash_code, text)

    # Inline code — stash to protect from other processing
    inline_codes = []
    def _stash_inline(m):
        inline_codes.append(m.group(1))
        return f"\x00INLINE{len(inline_codes) - 1}\x00"
    text = re.sub(r"`([^`]+)`", _stash_inline, text)

    # Process line by line
    lines = text.split("\n")
    html_parts = []
    ul_pat = re.compile(r"^[\-\*]\s+")
    ol_pat = re.compile(r"^\d+[\.\)]\s+")
    para_buf = []

    def flush_para():
        if not para_buf:
            return
        joined = " ".join(para_buf)
        html_parts.append(f"<p style='margin:8px 0;'>{joined}</p>")
        para_buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Blank line — flush paragraph
        if not line:
            flush_para()
            i += 1
            continue

        # Code block placeholder
        if line.startswith("\x00CODE"):
            flush_para()
            html_parts.append(line)
            i += 1
            continue

        # Headers
        if line.startswith("### "):
            flush_para()
            html_parts.append(f"<h3 style='margin:16px 0 8px;font-size:16px;'>{line[4:]}</h3>")
            i += 1
            continue
        if line.startswith("## "):
            flush_para()
            html_parts.append(f"<h2 style='margin:16px 0 8px;font-size:18px;'>{line[3:]}</h2>")
            i += 1
            continue
        if line.startswith("# "):
            flush_para()
            html_parts.append(f"<h1 style='margin:16px 0 8px;font-size:22px;'>{line[2:]}</h1>")
            i += 1
            continue

        # Unordered list — collect consecutive list items
        if re.match(r"^[\-\*]\s", line):
            flush_para()
            items = []
            while i < len(lines) and re.match(r"^[\-\*]\s", lines[i].strip()):
                items.append("<li>" + ul_pat.sub("", lines[i].strip()) + "</li>")
                i += 1
            html_parts.append(f"<ul style='margin:8px 0;padding-left:24px;'>{''.join(items)}</ul>")
            continue

        # Ordered list
        if re.match(r"^\d+[\.\)]\s", line):
            flush_para()
            items = []
            while i < len(lines) and re.match(r"^\d+[\.\)]\s", lines[i].strip()):
                items.append("<li>" + ol_pat.sub("", lines[i].strip()) + "</li>")
                i += 1
            html_parts.append(f"<ol style='margin:8px 0;padding-left:24px;'>{''.join(items)}</ol>")
            continue

        # Regular text — accumulate into paragraph
        para_buf.append(line)
        i += 1

    flush_para()

    html = "\n".join(html_parts)

    # Inline formatting
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
    html = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" style="color:#4a9f6e;">\1</a>', html)

    # Auto-linkify bare URLs not already inside an <a> tag (safety net for click tracking)
    html = re.sub(
        r'(?<!href=")(?<!">)(https?://[^\s<"]+)',
        r'<a href="\1" style="color:#4a9f6e;">\1</a>',
        html,
    )

    # Restore inline code
    for j, code in enumerate(inline_codes):
        html = html.replace(
            f"\x00INLINE{j}\x00",
            f'<code style="background:#f0f0f0;padding:2px 5px;border-radius:3px;font-size:13px;">{code}</code>',
        )

    # Restore code blocks
    for j, code in enumerate(code_blocks):
        html = html.replace(
            f"\x00CODE{j}\x00",
            f'<pre style="background:#f5f5f5;padding:12px;border-radius:6px;overflow-x:auto;font-size:13px;"><code>{code}</code></pre>',
        )

    return html


def _staff_ids():
    """Staff + internal user UUIDs for TraceLog filtering (uses UUIDField, not FK)."""
    return list(
        User.objects.filter(
            Q(is_staff=True) | Q(username__in=INTERNAL_USERNAMES)
        ).values_list("id", flat=True)
    )


# ---------------------------------------------------------------------------
# Page view
# ---------------------------------------------------------------------------

@user_passes_test(can_access_internal, login_url="/login/")
def dashboard_view(request):
    return render(request, "internal_dashboard.html")


# ---------------------------------------------------------------------------
# API: Overview KPIs
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_overview(request):
    days = _get_days(request)
    now = timezone.now()
    today = now.date()
    customers = _customers()
    staff_ids = _staff_ids()

    total_users = customers.count()
    active_today = customers.filter(last_active_at__date=today).count()

    day_usage = (
        UsageLog.objects.filter(date=today)
        .exclude(user__is_staff=True)
        .exclude(user__username__in=INTERNAL_USERNAMES)
        .aggregate(
            requests=Sum("request_count"),
            errors=Sum("error_count"),
        )
    )

    avg_latency = (
        TraceLog.objects.filter(
            created_at__gte=now - timedelta(hours=24),
            total_time_ms__isnull=False,
        )
        .exclude(user_id__in=staff_ids)
        .aggregate(avg=Avg("total_time_ms"))["avg"]
    )

    mrr = sum(
        customers.filter(tier=t).count() * p for t, p in TIER_PRICES.items()
    )

    paid = customers.filter(tier__in=PAID_TIERS).count()
    conversion = round(paid / total_users * 100, 1) if total_users else 0

    # Week-over-week changes: compare last 7d vs preceding 7d
    week_ago = today - timedelta(days=7)
    two_weeks_ago = today - timedelta(days=14)
    usage_base = (
        UsageLog.objects.exclude(user__is_staff=True)
        .exclude(user__username__in=INTERNAL_USERNAMES)
    )
    this_week = usage_base.filter(date__gt=week_ago, date__lte=today).aggregate(
        requests=Sum("request_count"), errors=Sum("error_count"),
    )
    last_week = usage_base.filter(date__gt=two_weeks_ago, date__lte=week_ago).aggregate(
        requests=Sum("request_count"), errors=Sum("error_count"),
    )
    active_this_week = customers.filter(last_active_at__date__gt=week_ago).count()
    active_last_week = customers.filter(
        last_active_at__date__gt=two_weeks_ago, last_active_at__date__lte=week_ago
    ).count()
    signups_this_week = customers.filter(date_joined__date__gt=week_ago).count()
    signups_last_week = customers.filter(
        date_joined__date__gt=two_weeks_ago, date_joined__date__lte=week_ago
    ).count()

    def _wow(current, previous):
        if not previous:
            return None
        return round((current - previous) / previous * 100, 1)

    changes = {
        "users": _wow(signups_this_week, signups_last_week),
        "active": _wow(active_this_week, active_last_week),
        "requests": _wow(this_week["requests"] or 0, last_week["requests"] or 0),
        "errors": _wow(this_week["errors"] or 0, last_week["errors"] or 0),
    }

    return Response({
        "total_users": total_users,
        "active_today": active_today,
        "requests_today": day_usage["requests"] or 0,
        "errors_today": day_usage["errors"] or 0,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency else None,
        "mrr": mrr,
        "conversion_rate": conversion,
        "changes": changes,
    })


# ---------------------------------------------------------------------------
# API: Users
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_users(request):
    days = _get_days(request)
    since = timezone.now() - timedelta(days=days)
    customers = _customers()

    signups = (
        customers.filter(date_joined__gte=since)
        .annotate(date=TruncDate("date_joined"))
        .values("date")
        .annotate(count=Count("id"))
        .order_by("date")
    )

    tiers = (
        customers.values("tier")
        .annotate(count=Count("id"))
        .order_by("tier")
    )

    industries = (
        customers.exclude(industry="")
        .values("industry")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    roles = (
        customers.exclude(role="")
        .values("role")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    experience = (
        customers.exclude(experience_level="")
        .values("experience_level")
        .annotate(count=Count("id"))
        .order_by("-count")
    )

    active_trend = (
        customers.filter(last_active_at__gte=since, last_active_at__isnull=False)
        .annotate(date=TruncDate("last_active_at"))
        .values("date")
        .annotate(count=Count("id"))
        .order_by("date")
    )

    total = customers.count()
    verified = customers.filter(email_verified=True).count()

    # Churn risk: paid users who haven't been active recently
    now = timezone.now()
    at_risk = []
    paid_users = customers.filter(tier__in=PAID_TIERS)
    for u in paid_users.order_by("last_active_at")[:20]:
        days_inactive = (now - u.last_active_at).days if u.last_active_at else 999
        if days_inactive < 14:
            continue
        at_risk.append({
            "username": u.username,
            "tier": u.tier,
            "days_inactive": days_inactive,
            "total_queries": u.total_queries,
            "email": u.email,
        })
        if len(at_risk) >= 10:
            break

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
        "churn_risk": at_risk,
    })


# ---------------------------------------------------------------------------
# API: Usage
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_usage(request):
    days = _get_days(request)
    since = timezone.now().date() - timedelta(days=days)
    logs = UsageLog.objects.filter(date__gte=since).exclude(user__is_staff=True).exclude(user__username__in=INTERNAL_USERNAMES)

    daily_requests = (
        logs.values("date")
        .annotate(total=Sum("request_count"))
        .order_by("date")
    )

    # Aggregate domain_counts JSON in Python (fine at alpha scale)
    domain_totals = Counter()
    for log in logs.exclude(domain_counts__isnull=True):
        if log.domain_counts:
            for domain, count in log.domain_counts.items():
                domain_totals[domain] += count

    daily_tokens = (
        logs.values("date")
        .annotate(input=Sum("tokens_input"), output=Sum("tokens_output"))
        .order_by("date")
    )

    daily_errors = (
        logs.values("date")
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
@permission_classes([IsInternalUser])
def api_performance(request):
    days = _get_days(request)
    since = timezone.now() - timedelta(days=days)
    traces = (
        TraceLog.objects.filter(created_at__gte=since)
        .exclude(user_id__in=_staff_ids())
    )

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
@permission_classes([IsInternalUser])
def api_business(request):
    days = _get_days(request)
    since = timezone.now().date() - timedelta(days=days)
    customers = _customers()

    # Revenue by tier
    revenue = {}
    for tier, price in TIER_PRICES.items():
        count = customers.filter(tier=tier).count()
        revenue[tier] = {"count": count, "mrr": count * price}

    # Conversion funnel
    total = customers.count()
    verified = customers.filter(email_verified=True).count()
    queried = customers.filter(total_queries__gt=0).count()
    paid = customers.filter(tier__in=PAID_TIERS).count()

    # Churn
    churning = Subscription.objects.filter(
        cancel_at_period_end=True,
    ).exclude(user__is_staff=True).exclude(user__username__in=INTERNAL_USERNAMES).count()
    active_subs = Subscription.objects.filter(
        status="active",
    ).exclude(user__is_staff=True).exclude(user__username__in=INTERNAL_USERNAMES).count()

    # Founder slots
    founder_count = customers.filter(tier="founder").count()

    # Feature adoption (paid customers only)
    tool_usage = Counter()
    paid_users = customers.filter(tier__in=PAID_TIERS)
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
        "founder_slots": {"used": founder_count, "total": 50},
        "feature_adoption": tool_usage.most_common(15),
    })


@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_cohort_retention(request):
    """Monthly cohort retention: what % of each signup-month cohort were active in subsequent months."""
    now = timezone.now()
    months_back = min(int(request.GET.get("months", 6)), 12)
    customers = _customers()

    cohorts = []
    for m in range(months_back - 1, -1, -1):
        # First day of each cohort month
        cohort_start = (now.replace(day=1) - timedelta(days=m * 30)).replace(day=1)
        if cohort_start.month == 12:
            cohort_end = cohort_start.replace(year=cohort_start.year + 1, month=1)
        else:
            cohort_end = cohort_start.replace(month=cohort_start.month + 1)

        cohort_users = customers.filter(
            date_joined__gte=cohort_start, date_joined__lt=cohort_end
        )
        cohort_size = cohort_users.count()
        if cohort_size == 0:
            continue

        cohort_ids = list(cohort_users.values_list("id", flat=True))
        label = cohort_start.strftime("%Y-%m")

        retention = []
        # Month 0 is always 100%
        retention.append({"month": 0, "retained": 100})

        # For each subsequent month
        for offset in range(1, months_back - m):
            check_start = cohort_start.replace(day=1)
            # Advance by offset months
            check_month = check_start.month + offset
            check_year = check_start.year + (check_month - 1) // 12
            check_month = ((check_month - 1) % 12) + 1
            check_start = check_start.replace(year=check_year, month=check_month)

            if check_start > now:
                break

            active_count = (
                UsageLog.objects.filter(
                    user_id__in=cohort_ids,
                    date__gte=check_start.date(),
                    date__lt=(check_start.replace(
                        month=check_start.month + 1 if check_start.month < 12 else 1,
                        year=check_start.year if check_start.month < 12 else check_start.year + 1
                    )).date(),
                )
                .values("user_id")
                .distinct()
                .count()
            )
            pct = round(active_count / cohort_size * 100, 1)
            retention.append({"month": offset, "retained": pct})

        cohorts.append({
            "label": label,
            "size": cohort_size,
            "retention": retention,
        })

    return Response({"cohorts": cohorts})


# ---------------------------------------------------------------------------
# API: AI Insights (POST — calls Anthropic)
# ---------------------------------------------------------------------------

@api_view(["POST"])
@permission_classes([IsInternalUser])
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
                "operations tools. Tiers: Free ($0), Professional ($49/mo), "
                "Team ($99/mo), Enterprise ($299/mo). Target audience: engineers, "
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
# API: Send Email
# ---------------------------------------------------------------------------

EMAIL_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f4f7f4;font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f7f4;padding:40px 20px;">
<tr><td align="center">
<table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;">
<tr><td style="background:#1a2a1a;padding:24px 32px;">
  <span style="color:#4a9f6e;font-size:20px;font-weight:600;letter-spacing:1px;">SVEND</span>
</td></tr>
<tr><td style="padding:32px;color:#1a1a1a;font-size:15px;line-height:1.7;">
  {body}
</td></tr>
<tr><td style="padding:16px 32px 24px;border-top:1px solid #e8efe8;color:#7a8f7a;font-size:12px;">
  SVEND &middot; Decision Science Workbench &middot; <a href="https://svend.ai" style="color:#4a9f6e;">svend.ai</a>
  <br><a href="{unsub_url}" style="color:#7a8f7a;">Unsubscribe</a>
</td></tr>
</table>
</td></tr>
</table>
</body>
</html>"""


@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_email_preview(request):
    """Return count of users matching a segment target."""
    target = request.GET.get("target", "")
    if not target or "@" in target:
        return Response({"count": None})
    qs, err = _resolve_recipients(target)
    if err:
        return Response({"count": 0, "error": err})
    return Response({"count": qs.count() if qs is not None else 0})


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_send_email(request):
    """Send HTML email to customers with tracking. Supports individual, tier-based, or all."""
    import re
    from django.core.mail import send_mail as django_send_mail
    from api.models import EmailCampaign, EmailRecipient

    target = request.data.get("to", "")
    subject = request.data.get("subject", "").strip()
    body_md = request.data.get("body", "").strip()
    test_mode = request.data.get("test", False)

    if not subject or not body_md:
        return Response({"error": "Subject and body are required."}, status=400)

    # Convert markdown to HTML
    body_html = _markdown_to_html(body_md)

    # Resolve recipients
    recipients = []  # list of (user_or_none, email)
    if test_mode:
        recipients = [(request.user, request.user.email)]
    elif "@" in target:
        recipients = [(None, target)]
    else:
        qs, err = _resolve_recipients(target)
        if err:
            return Response({"error": err}, status=400)
        if qs is not None:
            recipients = [(u, u.email) for u in qs]

    if not recipients:
        return Response({"error": "No recipients found."}, status=400)

    # Create campaign record
    campaign = EmailCampaign.objects.create(
        subject=subject,
        body_md=body_md,
        target=target,
        sent_by=request.user,
        recipient_count=len(recipients),
        is_test=test_mode,
    )

    # Check for active email_subject experiment
    from api.experiments import assign_variant
    from api.views import make_unsubscribe_url
    active_subject_exp = Experiment.objects.filter(
        experiment_type="email_subject", status="running"
    ).first()

    sent = 0
    failed = 0
    skipped = 0
    for user, email in recipients:
        # Skip opted-out users
        if user and getattr(user, "email_opted_out", False):
            skipped += 1
            continue

        # Create recipient record
        rcpt = EmailRecipient.objects.create(
            campaign=campaign,
            user=user,
            email=email,
        )

        # A/B test: if email_subject experiment is running, assign variant subject
        actual_subject = subject
        if active_subject_exp and user:
            variant_name, variant_config = assign_variant(user, active_subject_exp.name)
            if variant_config and "subject" in variant_config:
                actual_subject = variant_config["subject"]

        # Personalize
        personalized = body_html
        if user:
            personalized = personalized.replace(
                "{{name}}", user.display_name or user.username
            ).replace(
                "{{firstname}}", user.first_name or user.display_name or user.username
            ).replace(
                "{{email}}", user.email
            ).replace(
                "{{tier}}", user.tier
            )

        # Rewrite links for click tracking
        def _track_link(match):
            url = match.group(1)
            return f'href="https://svend.ai/api/email/click/{rcpt.id}/?url={url}"'
        personalized = re.sub(r'href="(https?://[^"]+)"', _track_link, personalized)

        # Add tracking pixel
        pixel = f'<img src="https://svend.ai/api/email/open/{rcpt.id}/" width="1" height="1" style="display:none;" alt="">'
        unsub_url = make_unsubscribe_url(user) if user else "https://svend.ai"
        full_html = EMAIL_TEMPLATE.format(body=personalized + pixel, unsub_url=unsub_url)

        try:
            django_send_mail(
                subject=actual_subject,
                message="",
                from_email=None,
                recipient_list=[email],
                html_message=full_html,
            )
            sent += 1
        except Exception:
            rcpt.failed = True
            rcpt.save(update_fields=["failed"])
            failed += 1

    return Response({
        "sent": sent,
        "failed": failed,
        "skipped_opted_out": skipped,
        "campaign_id": str(campaign.id),
        "experiment": active_subject_exp.name if active_subject_exp else None,
    })


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_save_email_draft(request):
    """Save or update an email draft. Supports multiple drafts."""
    import uuid as _uuid

    user = request.user
    prefs = user.preferences or {}
    drafts = prefs.get("email_drafts", [])

    draft_id = request.data.get("id")
    draft_data = {
        "to": request.data.get("to", ""),
        "custom_to": request.data.get("custom_to", ""),
        "subject": request.data.get("subject", ""),
        "body": request.data.get("body", ""),
        "updated_at": timezone.now().isoformat(),
    }

    if draft_id:
        # Update existing
        for d in drafts:
            if d.get("id") == draft_id:
                d.update(draft_data)
                break
    else:
        # New draft
        draft_id = str(_uuid.uuid4())[:8]
        draft_data["id"] = draft_id
        draft_data["created_at"] = draft_data["updated_at"]
        drafts.insert(0, draft_data)

    prefs["email_drafts"] = drafts
    user.preferences = prefs
    user.save(update_fields=["preferences"])
    return Response({"saved": True, "id": draft_id})


@api_view(["GET", "DELETE"])
@permission_classes([IsInternalUser])
def api_get_email_draft(request):
    """GET: list all drafts. DELETE: remove a draft by ?id=."""
    user = request.user
    prefs = user.preferences or {}
    drafts = prefs.get("email_drafts", [])

    # Migrate single-draft format if present
    if "email_draft" in prefs and prefs["email_draft"]:
        old = prefs.pop("email_draft")
        old["id"] = "migrated"
        old["created_at"] = old.get("updated_at", timezone.now().isoformat())
        old["updated_at"] = old["created_at"]
        drafts.insert(0, old)
        prefs["email_drafts"] = drafts
        user.preferences = prefs
        user.save(update_fields=["preferences"])

    if request.method == "DELETE":
        draft_id = request.GET.get("id")
        if draft_id:
            prefs["email_drafts"] = [d for d in drafts if d.get("id") != draft_id]
            user.preferences = prefs
            user.save(update_fields=["preferences"])
            return Response({"deleted": True})
        return Response({"error": "id required"}, status=400)

    return Response({"drafts": drafts})


@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_email_campaigns(request):
    """List email campaigns with tracking stats."""
    from api.models import EmailCampaign, EmailRecipient
    from django.db.models import Count, Q

    days = _get_days(request)
    since = timezone.now() - timedelta(days=days)

    campaigns = (
        EmailCampaign.objects.filter(created_at__gte=since)
        .annotate(
            total_sent=Count("recipients", filter=Q(recipients__failed=False)),
            total_failed=Count("recipients", filter=Q(recipients__failed=True)),
            total_opened=Count("recipients", filter=Q(recipients__opened_at__isnull=False)),
            total_clicked=Count("recipients", filter=Q(recipients__clicked_at__isnull=False)),
        )
        .order_by("-created_at")
    )

    # Compute conversions: recipients who upgraded within 7 days of campaign
    campaign_list = list(campaigns)
    conversion_map = {}
    for c in campaign_list:
        window_end = c.created_at + timedelta(days=7)
        recipient_user_ids = (
            EmailRecipient.objects.filter(campaign=c, user__isnull=False)
            .values_list("user_id", flat=True)
        )
        conversions = Subscription.objects.filter(
            user_id__in=recipient_user_ids,
            status="active",
            current_period_start__gte=c.created_at,
            current_period_start__lte=window_end,
        ).count()
        conversion_map[c.id] = conversions

    return Response({
        "campaigns": [
            {
                "id": str(c.id),
                "subject": c.subject,
                "target": c.target,
                "is_test": c.is_test,
                "recipient_count": c.recipient_count,
                "sent": c.total_sent,
                "failed": c.total_failed,
                "opened": c.total_opened,
                "clicked": c.total_clicked,
                "open_rate": round(c.total_opened / c.total_sent * 100, 1) if c.total_sent else 0,
                "click_rate": round(c.total_clicked / c.total_sent * 100, 1) if c.total_sent else 0,
                "conversions": conversion_map.get(c.id, 0),
                "created_at": c.created_at.isoformat(),
            }
            for c in campaign_list
        ],
    })


# ---------------------------------------------------------------------------
# API: Activity (Event Tracking)
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_activity(request):
    days = _get_days(request)
    since = timezone.now() - timedelta(days=days)
    events = EventLog.objects.filter(created_at__gte=since).exclude(user__is_staff=True).exclude(user__username__in=INTERNAL_USERNAMES)

    # Page popularity
    page_views = (
        events.filter(event_type="page_view")
        .values("page")
        .annotate(count=Count("id"))
        .order_by("-count")[:20]
    )

    # Feature heatmap
    feature_use = (
        events.filter(event_type="feature_use")
        .values("category", "action")
        .annotate(count=Count("id"))
        .order_by("-count")[:20]
    )

    # Daily active sessions
    daily_sessions = (
        events.filter(event_type="session_start")
        .annotate(date=TruncDate("created_at"))
        .values("date")
        .annotate(count=Count("session_id", distinct=True))
        .order_by("date")
    )

    # Recent user journeys (most recent first, customers only)
    recent_events = (
        events.select_related("user")
        .order_by("-created_at")[:200]
    )
    journeys = {}
    for evt in recent_events:
        uid = str(evt.user_id) if evt.user_id else "anon"
        if uid not in journeys:
            journeys[uid] = {
                "username": evt.user.username if evt.user else "anon",
                "events": [],
            }
        if len(journeys[uid]["events"]) < 20:
            journeys[uid]["events"].append({
                "type": evt.event_type,
                "category": evt.category,
                "action": evt.action,
                "page": evt.page,
                "time": evt.created_at.isoformat(),
            })

    # Feature use over time (daily)
    daily_features = (
        events.filter(event_type="feature_use")
        .annotate(date=TruncDate("created_at"))
        .values("date")
        .annotate(count=Count("id"))
        .order_by("date")
    )

    # Totals
    total_events = events.count()
    total_page_views = events.filter(event_type="page_view").count()
    total_feature_uses = events.filter(event_type="feature_use").count()
    unique_sessions = events.values("session_id").distinct().count()

    return Response({
        "page_views": list(page_views),
        "feature_use": list(feature_use),
        "daily_sessions": [
            {"date": str(d["date"]), "count": d["count"]} for d in daily_sessions
        ],
        "journeys": list(journeys.values())[:10],
        "daily_features": [
            {"date": str(d["date"]), "count": d["count"]} for d in daily_features
        ],
        "totals": {
            "events": total_events,
            "page_views": total_page_views,
            "feature_uses": total_feature_uses,
            "unique_sessions": unique_sessions,
        },
    })


# ---------------------------------------------------------------------------
# API: Onboarding Analytics
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_onboarding(request):
    """Onboarding funnel, survey distributions, and email stats."""
    from api.models import OnboardingEmail, OnboardingSurvey

    customers = _customers()
    total = customers.count()
    completed = customers.filter(onboarding_completed_at__isnull=False).count()
    surveys = OnboardingSurvey.objects.filter(user__is_staff=False).exclude(user__username__in=INTERNAL_USERNAMES)

    # Funnel
    verified = customers.filter(email_verified=True).count()
    queried = customers.filter(total_queries__gt=0).count()
    paid = customers.filter(tier__in=PAID_TIERS).count()

    # Survey distributions
    industry_dist = dict(
        surveys.exclude(industry="")
        .values_list("industry")
        .annotate(c=Count("id"))
        .values_list("industry", "c")
    )
    role_dist = dict(
        surveys.exclude(role="")
        .values_list("role")
        .annotate(c=Count("id"))
        .values_list("role", "c")
    )
    experience_dist = dict(
        surveys.exclude(experience_level="")
        .values_list("experience_level")
        .annotate(c=Count("id"))
        .values_list("experience_level", "c")
    )
    goal_dist = dict(
        surveys.exclude(primary_goal="")
        .values_list("primary_goal")
        .annotate(c=Count("id"))
        .values_list("primary_goal", "c")
    )
    path_dist = dict(
        surveys.exclude(learning_path="")
        .values_list("learning_path")
        .annotate(c=Count("id"))
        .values_list("learning_path", "c")
    )

    # Confidence and urgency averages
    conf_avg = surveys.aggregate(avg=Avg("confidence_stats"))["avg"]
    urg_avg = surveys.aggregate(avg=Avg("urgency"))["avg"]

    # Tools used (JSON array field — need to iterate)
    tool_counts = {}
    for survey in surveys.exclude(tools_used=[]):
        for t in (survey.tools_used or []):
            tool_counts[t] = tool_counts.get(t, 0) + 1

    # Top challenges (free text — just return recent ones for the dashboard)
    challenges = list(
        surveys.exclude(biggest_challenge="")
        .order_by("-created_at")
        .values_list("biggest_challenge", flat=True)[:20]
    )

    # Email stats
    emails = OnboardingEmail.objects.filter(user__is_staff=False).exclude(user__username__in=INTERNAL_USERNAMES)
    email_stats = {}
    for key in ["welcome", "getting_started", "tips", "learning_path", "checkin"]:
        key_emails = emails.filter(email_key=key)
        email_stats[key] = {
            "total": key_emails.count(),
            "sent": key_emails.filter(status="sent").count(),
            "pending": key_emails.filter(status="pending").count(),
            "failed": key_emails.filter(status="failed").count(),
        }

    # Completion over time
    completions = list(
        customers.filter(onboarding_completed_at__isnull=False)
        .annotate(date=TruncDate("onboarding_completed_at"))
        .values("date")
        .annotate(count=Count("id"))
        .order_by("date")
    )

    return Response({
        "funnel": {
            "registered": total,
            "onboarded": completed,
            "verified": verified,
            "queried": queried,
            "paid": paid,
        },
        "completion_rate": round(completed / total * 100, 1) if total else 0,
        "distributions": {
            "industry": industry_dist,
            "role": role_dist,
            "experience": experience_dist,
            "goal": goal_dist,
            "learning_path": path_dist,
            "tools": tool_counts,
        },
        "averages": {
            "confidence": round(conf_avg, 1) if conf_avg else None,
            "urgency": round(urg_avg, 1) if urg_avg else None,
        },
        "challenges": challenges,
        "email_stats": email_stats,
        "completions_over_time": [
            {"date": str(c["date"]), "count": c["count"]} for c in completions
        ],
    })


# ---------------------------------------------------------------------------
# API: Blog Management (Content tab)
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_blog_list(request):
    """List all blog posts (drafts, scheduled, and published)."""
    posts = BlogPost.objects.all().order_by("-created_at")
    return Response({
        "posts": [
            {
                "id": str(p.id),
                "title": p.title,
                "slug": p.slug,
                "status": p.status,
                "meta_description": p.meta_description,
                "created_at": p.created_at.isoformat(),
                "updated_at": p.updated_at.isoformat(),
                "published_at": p.published_at.isoformat() if p.published_at else None,
                "scheduled_at": p.scheduled_at.isoformat() if p.scheduled_at else None,
            }
            for p in posts
        ],
        "counts": {
            "total": posts.count(),
            "published": posts.filter(status="published").count(),
            "scheduled": posts.filter(status="scheduled").count(),
            "draft": posts.filter(status="draft").count(),
        },
    })


@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_blog_get(request, post_id):
    """Get full blog post content for editing."""
    try:
        post = BlogPost.objects.get(id=post_id)
    except BlogPost.DoesNotExist:
        return Response({"error": "Post not found."}, status=404)
    return Response({
        "id": str(post.id),
        "title": post.title,
        "slug": post.slug,
        "body": post.body,
        "meta_description": post.meta_description,
        "status": post.status,
        "created_at": post.created_at.isoformat(),
        "published_at": post.published_at.isoformat() if post.published_at else None,
        "scheduled_at": post.scheduled_at.isoformat() if post.scheduled_at else None,
    })


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_blog_save(request):
    """Create or update a blog post."""
    data = request.data
    post_id = data.get("id")

    if post_id:
        try:
            post = BlogPost.objects.get(id=post_id)
        except BlogPost.DoesNotExist:
            return Response({"error": "Post not found."}, status=404)
    else:
        post = BlogPost(author=request.user)

    post.title = data.get("title", post.title or "Untitled")
    post.body = data.get("body", post.body or "")
    post.meta_description = data.get("meta_description", post.meta_description or "")

    # Allow manual slug override
    if data.get("slug"):
        post.slug = data["slug"]

    post.save()
    return Response({
        "id": str(post.id),
        "slug": post.slug,
        "status": post.status,
    })


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_blog_publish(request, post_id):
    """Publish, schedule, or unpublish a blog post."""
    try:
        post = BlogPost.objects.get(id=post_id)
    except BlogPost.DoesNotExist:
        return Response({"error": "Post not found."}, status=404)

    action = request.data.get("action", "publish")
    scheduled_at = request.data.get("scheduled_at")

    if action == "schedule" and scheduled_at:
        from django.utils.dateparse import parse_datetime
        dt = parse_datetime(scheduled_at)
        if not dt:
            return Response({"error": "Invalid datetime format."}, status=400)
        if timezone.is_naive(dt):
            dt = timezone.make_aware(dt)
        post.status = BlogPost.Status.SCHEDULED
        post.scheduled_at = dt
    elif action == "publish":
        post.status = BlogPost.Status.PUBLISHED
        post.scheduled_at = None
        if not post.published_at:
            post.published_at = timezone.now()
    else:
        post.status = BlogPost.Status.DRAFT
        post.scheduled_at = None
    post.save()
    return Response({
        "status": post.status,
        "scheduled_at": post.scheduled_at.isoformat() if post.scheduled_at else None,
    })


@api_view(["DELETE"])
@permission_classes([IsInternalUser])
def api_blog_delete(request, post_id):
    """Delete a blog post."""
    try:
        post = BlogPost.objects.get(id=post_id)
    except BlogPost.DoesNotExist:
        return Response({"error": "Post not found."}, status=404)
    post.delete()
    return Response({"deleted": True})


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_blog_generate(request):
    """Generate a blog post draft using AI."""
    topic = request.data.get("topic", "").strip()
    keywords = request.data.get("keywords", "").strip()
    tone = request.data.get("tone", "professional")

    if not topic:
        return Response({"error": "Topic is required."}, status=400)

    prompt = (
        f"Write a blog post about: {topic}\n\n"
        f"Target SEO keywords: {keywords}\n"
        f"Tone: {tone}\n\n"
        "Requirements:\n"
        "- Write in markdown format\n"
        "- 800-1200 words\n"
        "- Include a compelling introduction that hooks the reader\n"
        "- Use H2 and H3 headers to structure the content\n"
        "- Include practical examples and actionable advice\n"
        "- End with a conclusion that ties back to decision science / quality engineering\n"
        "- Naturally incorporate the SEO keywords without stuffing\n"
        "- Write for quality engineers, analysts, and operations professionals\n"
    )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=3000,
            system=(
                "You are a content writer for Svend, a decision science SaaS platform. "
                "Svend provides statistical analysis (like Minitab), SPC control charts, "
                "DOE, capability studies, A3 reports, value stream mapping, forecasting, "
                "and AI-powered decision support. Target audience: quality engineers, "
                "Six Sigma practitioners, analysts, and operations managers in manufacturing, "
                "healthcare, tech, and consulting. Write authoritative, practical content "
                "that demonstrates deep domain expertise. Don't be promotional — be genuinely "
                "useful. Include real statistical concepts and practical examples."
            ),
            messages=[{"role": "user", "content": prompt}],
        )

        body = response.content[0].text

        # Also generate meta description
        meta_response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Write a 150-character SEO meta description for this blog post. "
                    f"Include the primary keyword. Return ONLY the description, nothing else.\n\n"
                    f"Title topic: {topic}\n"
                    f"Keywords: {keywords}"
                ),
            }],
        )
        meta = meta_response.content[0].text.strip()[:160]

        return Response({
            "title": topic,
            "body": body,
            "meta_description": meta,
        })
    except Exception as e:
        return Response({"error": str(e)}, status=500)


# ---------------------------------------------------------------------------
# API: Blog Analytics
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_blog_analytics(request):
    """Blog performance metrics: views, referrers, top posts, trends."""
    days = _get_days(request)
    since = timezone.now() - timedelta(days=days)
    views = BlogView.objects.filter(viewed_at__gte=since, is_bot=False)

    # Views over time (daily)
    daily_views = list(
        views.annotate(date=TruncDate("viewed_at"))
        .values("date")
        .annotate(total=Count("id"), unique=Count("ip_hash", distinct=True))
        .order_by("date")
    )

    # Top posts by views
    top_posts = list(
        views.values("post__title", "post__slug", "post_id")
        .annotate(total=Count("id"), unique=Count("ip_hash", distinct=True))
        .order_by("-total")[:20]
    )

    # Referrer domains (where traffic comes from), including direct
    referrers_raw = list(
        views.exclude(referrer_domain="")
        .values("referrer_domain")
        .annotate(count=Count("id"))
        .order_by("-count")[:15]
    )
    direct = views.filter(referrer_domain="").count()
    referrers = [{"domain": "Direct", "count": direct}] if direct else []
    referrers += [{"domain": r["referrer_domain"], "count": r["count"]} for r in referrers_raw]

    # Totals
    total_views = views.count()
    unique_visitors = views.values("ip_hash").distinct().count()
    bot_hits = BlogView.objects.filter(viewed_at__gte=since, is_bot=True).count()

    # Per-post breakdown (for the post list)
    post_stats = {}
    for p in top_posts:
        post_stats[str(p["post_id"])] = {
            "views": p["total"],
            "unique": p["unique"],
        }

    return Response({
        "daily_views": [
            {"date": str(d["date"]), "total": d["total"], "unique": d["unique"]}
            for d in daily_views
        ],
        "top_posts": [
            {
                "title": p["post__title"],
                "slug": p["post__slug"],
                "views": p["total"],
                "unique": p["unique"],
            }
            for p in top_posts
        ],
        "referrers": referrers,
        "totals": {
            "views": total_views,
            "unique_visitors": unique_visitors,
            "bot_hits": bot_hits,
        },
        "post_stats": post_stats,
    })


# ---------------------------------------------------------------------------
# Whitepapers
# ---------------------------------------------------------------------------


@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_whitepaper_list(request):
    """List all white papers with counts."""
    papers = WhitePaper.objects.annotate(
        download_count=Count("downloads")
    ).order_by("-created_at")
    return Response({
        "papers": [
            {
                "id": str(p.id),
                "title": p.title,
                "slug": p.slug,
                "topic": p.topic,
                "status": p.status,
                "gated": p.gated,
                "meta_description": p.meta_description,
                "download_count": p.download_count,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "updated_at": p.updated_at.isoformat() if p.updated_at else None,
                "published_at": p.published_at.isoformat() if p.published_at else None,
            }
            for p in papers
        ],
        "counts": {
            "total": papers.count(),
            "published": papers.filter(status="published").count(),
            "draft": papers.filter(status="draft").count(),
        },
    })


@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_whitepaper_get(request, paper_id):
    """Get a single white paper for editing."""
    try:
        p = WhitePaper.objects.get(id=paper_id)
    except WhitePaper.DoesNotExist:
        return Response({"error": "not found"}, status=404)
    return Response({
        "id": str(p.id),
        "title": p.title,
        "slug": p.slug,
        "topic": p.topic,
        "description": p.description,
        "body": p.body,
        "meta_description": p.meta_description,
        "status": p.status,
        "gated": p.gated,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "published_at": p.published_at.isoformat() if p.published_at else None,
    })


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_whitepaper_save(request):
    """Create or update a white paper."""
    data = request.data
    paper_id = data.get("id")
    if paper_id:
        try:
            paper = WhitePaper.objects.get(id=paper_id)
        except WhitePaper.DoesNotExist:
            return Response({"error": "not found"}, status=404)
    else:
        paper = WhitePaper(author=request.user)

    paper.title = data.get("title", paper.title or "Untitled")
    paper.body = data.get("body", paper.body or "")
    paper.description = data.get("description", paper.description or "")
    paper.meta_description = data.get("meta_description", paper.meta_description or "")
    paper.topic = data.get("topic", paper.topic or "")
    paper.gated = data.get("gated", paper.gated)
    if data.get("slug"):
        paper.slug = data["slug"]
    paper.save()
    return Response({"id": str(paper.id), "slug": paper.slug, "status": paper.status})


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_whitepaper_publish(request, paper_id):
    """Publish or unpublish a white paper."""
    try:
        paper = WhitePaper.objects.get(id=paper_id)
    except WhitePaper.DoesNotExist:
        return Response({"error": "not found"}, status=404)

    action = request.data.get("action", "publish")
    if action == "publish":
        paper.status = WhitePaper.Status.PUBLISHED
        paper.published_at = timezone.now()
    else:
        paper.status = WhitePaper.Status.DRAFT
        paper.published_at = None
    paper.save()
    return Response({"status": paper.status})


@api_view(["DELETE"])
@permission_classes([IsInternalUser])
def api_whitepaper_delete(request, paper_id):
    """Delete a white paper."""
    try:
        paper = WhitePaper.objects.get(id=paper_id)
    except WhitePaper.DoesNotExist:
        return Response({"error": "not found"}, status=404)
    paper.delete()
    return Response({"ok": True})


@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_whitepaper_analytics(request):
    """White paper performance metrics: downloads, referrers, top papers."""
    days = _get_days(request)
    since = timezone.now() - timedelta(days=days)
    downloads = WhitePaperDownload.objects.filter(downloaded_at__gte=since, is_bot=False)

    # Downloads over time (daily)
    daily = list(
        downloads.annotate(date=TruncDate("downloaded_at"))
        .values("date")
        .annotate(total=Count("id"), unique=Count("ip_hash", distinct=True))
        .order_by("date")
    )

    # Top papers by downloads
    top_papers = list(
        downloads.values("paper__title", "paper__slug", "paper_id")
        .annotate(total=Count("id"), unique=Count("ip_hash", distinct=True))
        .order_by("-total")[:20]
    )

    # Referrer domains
    referrers_raw = list(
        downloads.exclude(referrer_domain="")
        .values("referrer_domain")
        .annotate(count=Count("id"))
        .order_by("-count")[:15]
    )
    direct = downloads.filter(referrer_domain="").count()
    referrers = [{"domain": "Direct", "count": direct}] if direct else []
    referrers += [{"domain": r["referrer_domain"], "count": r["count"]} for r in referrers_raw]

    # Totals
    total_downloads = downloads.count()
    unique_downloaders = downloads.values("ip_hash").distinct().count()
    emails_captured = downloads.exclude(email="").values("email").distinct().count()
    bot_hits = WhitePaperDownload.objects.filter(downloaded_at__gte=since, is_bot=True).count()

    return Response({
        "daily_downloads": [
            {"date": str(d["date"]), "total": d["total"], "unique": d["unique"]}
            for d in daily
        ],
        "top_papers": [
            {
                "title": p["paper__title"],
                "slug": p["paper__slug"],
                "downloads": p["total"],
                "unique": p["unique"],
            }
            for p in top_papers
        ],
        "referrers": referrers,
        "totals": {
            "downloads": total_downloads,
            "unique_downloaders": unique_downloaders,
            "emails_captured": emails_captured,
            "bot_hits": bot_hits,
        },
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_data_snapshot(days=30):
    """Anonymized aggregate snapshot for AI analysis (customers only)."""
    now = timezone.now()
    since = now - timedelta(days=days)
    since_date = since.date()
    customers = _customers()
    staff_ids = _staff_ids()

    total_users = customers.count()
    verified = customers.filter(email_verified=True).count()
    queried = customers.filter(total_queries__gt=0).count()
    paid = customers.filter(tier__in=PAID_TIERS).count()

    tier_dist = dict(
        customers.values_list("tier")
        .annotate(c=Count("id"))
        .values_list("tier", "c")
    )

    mrr = sum(tier_dist.get(t, 0) * p for t, p in TIER_PRICES.items())

    usage = (
        UsageLog.objects.filter(date__gte=since_date)
        .exclude(user__is_staff=True)
        .exclude(user__username__in=INTERNAL_USERNAMES)
        .aggregate(
            total_requests=Sum("request_count"),
            total_errors=Sum("error_count"),
            total_tokens_in=Sum("tokens_input"),
            total_tokens_out=Sum("tokens_output"),
        )
    )

    perf = (
        TraceLog.objects.filter(created_at__gte=since)
        .exclude(user_id__in=staff_ids)
        .aggregate(
            avg_latency=Avg("total_time_ms"),
            total_traces=Count("id"),
            gates_passed=Count("id", filter=Q(gate_passed=True)),
            fallbacks=Count("id", filter=Q(fallback_used=True)),
        )
    )

    domain_totals = Counter()
    for log in (
        UsageLog.objects.filter(date__gte=since_date)
        .exclude(user__is_staff=True)
        .exclude(user__username__in=INTERNAL_USERNAMES)
        .exclude(domain_counts__isnull=True)
    ):
        if log.domain_counts:
            for d, c in log.domain_counts.items():
                domain_totals[d] += c

    industries = dict(
        customers.exclude(industry="")
        .values_list("industry")
        .annotate(c=Count("id"))
        .values_list("industry", "c")
    )
    roles = dict(
        customers.exclude(role="")
        .values_list("role")
        .annotate(c=Count("id"))
        .values_list("role", "c")
    )

    churning = Subscription.objects.filter(
        cancel_at_period_end=True,
    ).exclude(user__is_staff=True).exclude(user__username__in=INTERNAL_USERNAMES).count()

    signups = list(
        customers.filter(date_joined__gte=since)
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
            "founder_slots_total": 50,
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


# ---------------------------------------------------------------------------
# Automation: Experiments
# ---------------------------------------------------------------------------

@api_view(["GET", "POST"])
@permission_classes([IsInternalUser])
def api_experiments(request):
    """List experiments or create a new one."""
    if request.method == "GET":
        exps = Experiment.objects.all()[:50]
        data = []
        for exp in exps:
            assigned = ExperimentAssignment.objects.filter(experiment=exp).count()
            data.append({
                "id": str(exp.id),
                "name": exp.name,
                "hypothesis": exp.hypothesis,
                "experiment_type": exp.experiment_type,
                "metric": exp.metric,
                "variants": exp.variants,
                "status": exp.status,
                "winner": exp.winner,
                "target": exp.target,
                "min_sample_size": exp.min_sample_size,
                "results": exp.results,
                "assigned": assigned,
                "started_at": exp.started_at,
                "ended_at": exp.ended_at,
                "created_at": exp.created_at,
            })
        return Response({"experiments": data})

    # POST — create or update
    d = request.data
    exp_id = d.get("id")
    if exp_id:
        try:
            exp = Experiment.objects.get(id=exp_id)
        except Experiment.DoesNotExist:
            return Response({"error": "not_found"}, status=404)
        for field in ["name", "hypothesis", "experiment_type", "metric", "variants",
                       "status", "target", "min_sample_size"]:
            if field in d:
                setattr(exp, field, d[field])
        if d.get("status") == "running" and not exp.started_at:
            exp.started_at = timezone.now()
        if d.get("status") == "concluded" and not exp.ended_at:
            exp.ended_at = timezone.now()
        exp.save()
    else:
        exp = Experiment.objects.create(
            name=d.get("name", ""),
            hypothesis=d.get("hypothesis", ""),
            experiment_type=d.get("experiment_type", "feature_flag"),
            metric=d.get("metric", "conversion"),
            variants=d.get("variants", []),
            status=d.get("status", "draft"),
            target=d.get("target", "all"),
            min_sample_size=d.get("min_sample_size", 100),
            started_at=timezone.now() if d.get("status") == "running" else None,
        )
    return Response({"id": str(exp.id), "status": exp.status})


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_experiment_evaluate(request, experiment_id):
    """Manually trigger experiment evaluation."""
    from api.experiments import evaluate_experiment

    try:
        exp = Experiment.objects.get(id=experiment_id)
    except Experiment.DoesNotExist:
        return Response({"error": "not_found"}, status=404)

    results = evaluate_experiment(exp)
    return Response({
        "results": results,
        "status": exp.status,
        "winner": exp.winner,
    })


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_experiment_conclude(request, experiment_id):
    """Manually conclude an experiment with a chosen winner."""
    try:
        exp = Experiment.objects.get(id=experiment_id)
    except Experiment.DoesNotExist:
        return Response({"error": "not_found"}, status=404)

    winner = request.data.get("winner", "")
    exp.winner = winner
    exp.status = "concluded"
    exp.ended_at = timezone.now()
    exp.save(update_fields=["winner", "status", "ended_at"])
    return Response({"status": "concluded", "winner": winner})


# ---------------------------------------------------------------------------
# Automation: Rules
# ---------------------------------------------------------------------------

@api_view(["GET", "POST"])
@permission_classes([IsInternalUser])
def api_automation_rules(request):
    """List or create automation rules."""
    if request.method == "GET":
        rules = AutomationRule.objects.all()
        data = []
        for rule in rules:
            data.append({
                "id": str(rule.id),
                "name": rule.name,
                "description": rule.description,
                "trigger": rule.trigger,
                "trigger_config": rule.trigger_config,
                "trigger_2": rule.trigger_2,
                "trigger_2_config": rule.trigger_2_config,
                "trigger_logic": rule.trigger_logic,
                "action": rule.action,
                "action_config": rule.action_config,
                "is_active": rule.is_active,
                "cooldown_hours": rule.cooldown_hours,
                "times_fired": rule.times_fired,
                "last_fired_at": rule.last_fired_at,
                "created_at": rule.created_at,
            })
        return Response({"rules": data})

    # POST — create or update
    d = request.data
    rule_id = d.get("id")
    compound_fields = ["trigger_2", "trigger_2_config", "trigger_logic"]
    if rule_id:
        try:
            rule = AutomationRule.objects.get(id=rule_id)
        except AutomationRule.DoesNotExist:
            return Response({"error": "not_found"}, status=404)
        for field in ["name", "description", "trigger", "trigger_config",
                       "action", "action_config", "is_active", "cooldown_hours"] + compound_fields:
            if field in d:
                setattr(rule, field, d[field])
        rule.save()
    else:
        rule = AutomationRule.objects.create(
            name=d.get("name", ""),
            description=d.get("description", ""),
            trigger=d.get("trigger", "inactive_days"),
            trigger_config=d.get("trigger_config", {}),
            trigger_2=d.get("trigger_2", ""),
            trigger_2_config=d.get("trigger_2_config"),
            trigger_logic=d.get("trigger_logic", "and"),
            action=d.get("action", "send_email"),
            action_config=d.get("action_config", {}),
            cooldown_hours=d.get("cooldown_hours", 72),
        )
    return Response({"id": str(rule.id)})


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_automation_rule_toggle(request, rule_id):
    """Toggle an automation rule on/off."""
    try:
        rule = AutomationRule.objects.get(id=rule_id)
    except AutomationRule.DoesNotExist:
        return Response({"error": "not_found"}, status=404)

    rule.is_active = not rule.is_active
    rule.save(update_fields=["is_active"])
    return Response({"id": str(rule.id), "is_active": rule.is_active})


# ---------------------------------------------------------------------------
# Automation: Log
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_automation_log(request):
    """Recent automation log entries."""
    limit = int(request.GET.get("limit", 50))
    rule_id = request.GET.get("rule_id")
    qs = AutomationLog.objects.select_related("rule", "user").all()
    if rule_id:
        qs = qs.filter(rule_id=rule_id)
    entries = []
    for log in qs[:limit]:
        entries.append({
            "id": str(log.id),
            "rule": log.rule.name,
            "rule_id": str(log.rule_id),
            "user": log.user.username,
            "user_email": log.user.email,
            "action_taken": log.action_taken,
            "result": log.result,
            "fired_at": log.fired_at,
        })
    return Response({"log": entries})


# ---------------------------------------------------------------------------
# Automation: Autopilot
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_autopilot(request):
    """Get autopilot reports."""
    reports = AutopilotReport.objects.all()[:10]
    data = []
    for r in reports:
        data.append({
            "id": str(r.id),
            "created_at": r.created_at,
            "insights": r.insights,
            "recommendations": r.recommendations,
            "alerts": r.alerts,
            "status": r.status,
        })
    return Response({"reports": data})


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_autopilot_approve(request, report_id):
    """Approve a recommendation from an autopilot report."""
    try:
        report = AutopilotReport.objects.get(id=report_id)
    except AutopilotReport.DoesNotExist:
        return Response({"error": "not_found"}, status=404)

    rec_index = request.data.get("index")
    if rec_index is None or rec_index >= len(report.recommendations):
        return Response({"error": "invalid_index"}, status=400)

    rec = report.recommendations[rec_index]
    rec_type = rec.get("type")
    config = rec.get("config", {})
    result = {"action": rec_type, "status": "skipped"}

    if rec_type == "experiment":
        exp = Experiment.objects.create(
            name=config.get("name", rec.get("title", "Autopilot experiment")),
            hypothesis=config.get("hypothesis", ""),
            experiment_type=config.get("type", "feature_flag"),
            variants=config.get("variants", []),
            status="running",
            started_at=timezone.now(),
        )
        result = {"action": "experiment_created", "id": str(exp.id)}

    elif rec_type == "email":
        # Create and send a campaign
        from api.internal_views import EMAIL_TEMPLATE
        subject = config.get("subject", rec.get("title", ""))
        body = config.get("body_preview", "")
        target = config.get("target", "all")

        if subject and body:
            # Delegate to existing send logic
            result = {"action": "email_queued", "subject": subject, "target": target}

    elif rec_type == "blog":
        title = config.get("title", "")
        if title:
            from api.models import BlogPost
            post = BlogPost.objects.create(
                title=title,
                body=f"Draft generated from autopilot recommendation.\n\nTarget keyword: {config.get('target_keyword', '')}",
                status="draft",
            )
            result = {"action": "blog_draft_created", "id": str(post.id)}

    elif rec_type == "rule_tweak":
        rule_name = config.get("rule_name", "")
        if rule_name:
            result = {"action": "rule_tweak_noted", "rule": rule_name, "change": config.get("change", "")}

    # Mark recommendation as approved and store result for tracking
    report.recommendations[rec_index]["approved"] = True
    report.recommendations[rec_index]["result"] = result
    report.save(update_fields=["recommendations"])

    return Response(result)


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_autopilot_run(request):
    """Manually trigger a Claude growth review."""
    from syn.sched.scheduler import schedule_task

    schedule_task(
        name="manual_growth_review",
        func="api.claude_growth_review",
        args={},
        delay_seconds=0,
        priority=2,
        queue="core",
    )
    return Response({"status": "scheduled"})


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@api_view(["GET", "POST"])
@permission_classes([IsInternalUser])
def api_feedback(request):
    """List feedback or update status."""
    if request.method == "POST":
        feedback_id = request.data.get("id")
        new_status = request.data.get("status")
        notes = request.data.get("notes")
        try:
            fb = Feedback.objects.get(id=feedback_id)
            update_fields = []
            if new_status:
                fb.status = new_status
                update_fields.append("status")
            if notes is not None:
                fb.internal_notes = notes
                update_fields.append("internal_notes")
            if update_fields:
                fb.save(update_fields=update_fields)
            return Response({"status": fb.status, "notes": fb.internal_notes})
        except Feedback.DoesNotExist:
            return Response({"error": "not_found"}, status=404)

    # GET — list recent feedback
    status_filter = request.GET.get("status")
    qs = Feedback.objects.select_related("user").all()
    if status_filter:
        qs = qs.filter(status=status_filter)
    entries = []
    for fb in qs[:100]:
        entries.append({
            "id": str(fb.id),
            "user": fb.user.username if fb.user else "anonymous",
            "user_email": fb.user.email if fb.user else "",
            "user_tier": fb.user.tier if fb.user else "",
            "category": fb.category,
            "message": fb.message,
            "page": fb.page,
            "status": fb.status,
            "internal_notes": fb.internal_notes,
            "created_at": fb.created_at,
        })

    # Summary counts (always unfiltered)
    all_fb = Feedback.objects.all()
    by_status = dict(all_fb.values_list("status").annotate(c=Count("id")).values_list("status", "c"))
    by_category = dict(all_fb.values_list("category").annotate(c=Count("id")).values_list("category", "c"))
    summary = {
        "total": all_fb.count(),
        "by_status": {
            "new": by_status.get("new", 0),
            "reviewed": by_status.get("reviewed", 0),
            "resolved": by_status.get("resolved", 0),
        },
        "by_category": {
            "bug": by_category.get("bug", 0),
            "feature": by_category.get("feature", 0),
            "question": by_category.get("question", 0),
            "other": by_category.get("other", 0),
        },
    }
    return Response({"feedback": entries, "summary": summary})

# ---------------------------------------------------------------------------
# CRM — Outbound Outreach Management
# ---------------------------------------------------------------------------


@api_view(["GET", "POST"])
@permission_classes([IsInternalUser])
def api_crm_leads(request):
    """List (filterable) or create/update a CRM lead."""
    if request.method == "GET":
        qs = CRMLead.objects.all()
        stage = request.GET.get("stage")
        source = request.GET.get("source")
        search = request.GET.get("search", "").strip()
        if stage:
            qs = qs.filter(stage=stage)
        if source:
            qs = qs.filter(source=source)
        if search:
            qs = qs.filter(
                Q(name__icontains=search)
                | Q(email__icontains=search)
                | Q(company__icontains=search)
            )
        leads = []
        for lead in qs[:200]:
            leads.append({
                "id": str(lead.id),
                "name": lead.name,
                "email": lead.email,
                "company": lead.company,
                "role": lead.role,
                "industry": lead.industry,
                "source": lead.source,
                "stage": lead.stage,
                "notes": lead.notes,
                "tags": lead.tags,
                "email_opted_out": lead.email_opted_out,
                "last_contacted_at": lead.last_contacted_at.isoformat() if lead.last_contacted_at else None,
                "next_followup_at": lead.next_followup_at.isoformat() if lead.next_followup_at else None,
                "created_at": lead.created_at.isoformat(),
                "enrollments": [
                    {
                        "id": str(e.id),
                        "sequence": e.sequence.name,
                        "status": e.status,
                        "current_step": e.current_step,
                    }
                    for e in lead.enrollments.select_related("sequence").all()[:5]
                ],
            })
        return Response({"leads": leads})

    # POST — create or update
    d = request.data
    lead_id = d.get("id")
    if lead_id:
        try:
            lead = CRMLead.objects.get(id=lead_id)
        except CRMLead.DoesNotExist:
            return Response({"error": "not_found"}, status=404)
    else:
        lead = CRMLead()

    for field in ["name", "email", "company", "role", "industry", "source",
                   "stage", "notes", "tags", "email_opted_out"]:
        if field in d:
            setattr(lead, field, d[field])

    if d.get("next_followup_at"):
        from django.utils.dateparse import parse_datetime
        dt = parse_datetime(d["next_followup_at"])
        if dt:
            if timezone.is_naive(dt):
                dt = timezone.make_aware(dt)
            lead.next_followup_at = dt

    lead.save()
    return Response({"id": str(lead.id), "stage": lead.stage})


@api_view(["DELETE"])
@permission_classes([IsInternalUser])
def api_crm_lead_delete(request, lead_id):
    """Delete a CRM lead."""
    try:
        lead = CRMLead.objects.get(id=lead_id)
    except CRMLead.DoesNotExist:
        return Response({"error": "not_found"}, status=404)
    lead.delete()
    return Response({"deleted": True})


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_crm_lead_stage(request, lead_id):
    """Update a lead's pipeline stage."""
    try:
        lead = CRMLead.objects.get(id=lead_id)
    except CRMLead.DoesNotExist:
        return Response({"error": "not_found"}, status=404)
    new_stage = request.data.get("stage")
    if new_stage not in dict(CRMLead.Stage.choices):
        return Response({"error": "invalid_stage"}, status=400)
    lead.stage = new_stage
    lead.save(update_fields=["stage", "updated_at"])
    return Response({"id": str(lead.id), "stage": lead.stage})


@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_crm_pipeline(request):
    """Pipeline overview: stage counts, due follow-ups, active sequence count."""
    now = timezone.now()
    leads = CRMLead.objects.all()

    stage_counts = dict(
        leads.values_list("stage")
        .annotate(c=Count("id"))
        .values_list("stage", "c")
    )

    due_followups = leads.filter(
        next_followup_at__lte=now,
        stage__in=["prospect", "contacted", "engaged", "demo", "trial"],
    ).count()

    active_enrollments = OutreachEnrollment.objects.filter(status="active").count()

    source_counts = dict(
        leads.values_list("source")
        .annotate(c=Count("id"))
        .values_list("source", "c")
    )

    return Response({
        "stages": stage_counts,
        "due_followups": due_followups,
        "active_enrollments": active_enrollments,
        "sources": source_counts,
        "total_leads": leads.count(),
    })


@api_view(["GET", "POST"])
@permission_classes([IsInternalUser])
def api_crm_sequences(request):
    """List or create/update outreach sequences."""
    if request.method == "GET":
        seqs = OutreachSequence.objects.all()
        data = []
        for seq in seqs:
            enrolled = OutreachEnrollment.objects.filter(sequence=seq).count()
            active = OutreachEnrollment.objects.filter(sequence=seq, status="active").count()
            data.append({
                "id": str(seq.id),
                "name": seq.name,
                "description": seq.description,
                "is_active": seq.is_active,
                "steps": seq.steps,
                "step_count": len(seq.steps),
                "enrolled": enrolled,
                "active": active,
                "created_at": seq.created_at.isoformat(),
            })
        return Response({"sequences": data})

    # POST — create or update
    d = request.data
    seq_id = d.get("id")
    if seq_id:
        try:
            seq = OutreachSequence.objects.get(id=seq_id)
        except OutreachSequence.DoesNotExist:
            return Response({"error": "not_found"}, status=404)
    else:
        seq = OutreachSequence()

    for field in ["name", "description", "is_active", "steps"]:
        if field in d:
            setattr(seq, field, d[field])
    seq.save()
    return Response({"id": str(seq.id), "name": seq.name})


@api_view(["DELETE"])
@permission_classes([IsInternalUser])
def api_crm_sequence_delete(request, sequence_id):
    """Delete an outreach sequence."""
    try:
        seq = OutreachSequence.objects.get(id=sequence_id)
    except OutreachSequence.DoesNotExist:
        return Response({"error": "not_found"}, status=404)
    seq.delete()
    return Response({"deleted": True})


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_crm_enroll(request, sequence_id):
    """Enroll a lead in an outreach sequence. Assigns A/B variant via SHA256."""
    import hashlib

    try:
        seq = OutreachSequence.objects.get(id=sequence_id)
    except OutreachSequence.DoesNotExist:
        return Response({"error": "sequence_not_found"}, status=404)

    lead_id = request.data.get("lead_id")
    if not lead_id:
        return Response({"error": "lead_id required"}, status=400)

    try:
        lead = CRMLead.objects.get(id=lead_id)
    except CRMLead.DoesNotExist:
        return Response({"error": "lead_not_found"}, status=404)

    if lead.email_opted_out:
        return Response({"error": "lead_opted_out"}, status=400)

    if OutreachEnrollment.objects.filter(lead=lead, sequence=seq).exists():
        return Response({"error": "already_enrolled"}, status=400)

    # Deterministic A/B assignment (same pattern as experiments.py)
    hash_input = f"{lead.id}-{seq.id}"
    hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16) % 2
    variant = "a" if hash_val == 0 else "b"

    # Calculate first send time
    now = timezone.now()
    first_step = seq.steps[0] if seq.steps else None
    delay_days = first_step.get("delay_days", 0) if first_step else 0
    next_send = now + timedelta(days=delay_days)

    enrollment = OutreachEnrollment.objects.create(
        lead=lead,
        sequence=seq,
        variant=variant,
        current_step=0,
        next_send_at=next_send,
    )

    return Response({
        "id": str(enrollment.id),
        "variant": variant,
        "next_send_at": next_send.isoformat(),
    })


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_crm_generate_email(request):
    """Use Claude to generate personalized A/B email variants for outreach."""
    d = request.data
    lead_name = d.get("name", "")
    lead_company = d.get("company", "")
    lead_role = d.get("role", "")
    lead_industry = d.get("industry", "")
    step_purpose = d.get("purpose", "Introduction")
    step_number = d.get("step_number", 1)
    total_steps = d.get("total_steps", 1)
    custom_notes = d.get("notes", "")

    prompt = (
        f"Generate two A/B email variants for an outreach sequence.\n\n"
        f"Lead context:\n"
        f"- Name: {lead_name}\n"
        f"- Company: {lead_company}\n"
        f"- Role: {lead_role}\n"
        f"- Industry: {lead_industry}\n\n"
        f"Email context:\n"
        f"- Step {step_number} of {total_steps}\n"
        f"- Purpose: {step_purpose}\n"
        f"{'- Notes: ' + custom_notes if custom_notes else ''}\n\n"
        "Requirements:\n"
        "- Variant A: More direct/professional tone\n"
        "- Variant B: More conversational/value-led tone\n"
        "- Keep subject lines under 60 characters\n"
        "- Body should be 3-5 short paragraphs\n"
        "- Use {{name}} placeholder for the recipient's name\n"
        "- End with a clear, low-friction CTA\n"
        "- Never mention competitors by name\n\n"
        "Return ONLY valid JSON with this exact structure:\n"
        '{"subject_a": "...", "body_a": "...", "subject_b": "...", "body_b": "..."}\n'
        "Use \\n for newlines in the body text."
    )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            system=(
                "You are a sales copywriter for Svend, a decision science SaaS platform. "
                "Svend provides statistical analysis (like Minitab at $2,594/yr but modern, "
                "AI-powered, and starting at $49/mo), SPC, DOE, capability studies, forecasting, "
                "quality tools (A3, FMEA, RCA, VSM), and a collaborative knowledge graph. "
                "Target buyers: quality engineers, CI managers, analysts, and ops leaders in "
                "manufacturing, healthcare, tech, and consulting. Svend's edge: AI-guided "
                "analysis, real-time collaboration, integrated hypothesis tracking, and "
                "10x lower cost than legacy tools. Write concise, credible outreach — "
                "no hype, no buzzwords, no generic sales language. Show domain knowledge."
            ),
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
        return Response(result)
    except json.JSONDecodeError:
        return Response({"error": "Failed to parse AI response as JSON", "raw": text}, status=500)
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_crm_outreach_metrics(request):
    """Per-sequence outreach performance metrics."""
    from api.models import EmailRecipient

    sequences = OutreachSequence.objects.all()
    metrics = []
    for seq in sequences:
        enrollments = OutreachEnrollment.objects.filter(sequence=seq)
        enrolled = enrollments.count()
        active = enrollments.filter(status="active").count()
        completed = enrollments.filter(status="completed").count()
        replied = enrollments.filter(status="replied").count()

        # Gather all recipient IDs from send logs
        recipient_ids = []
        for e in enrollments:
            for entry in (e.send_log or []):
                if entry.get("recipient_id"):
                    recipient_ids.append(entry["recipient_id"])

        total_sent = len(recipient_ids)
        opens = 0
        clicks = 0
        if recipient_ids:
            opens = EmailRecipient.objects.filter(
                id__in=recipient_ids, opened_at__isnull=False
            ).count()
            clicks = EmailRecipient.objects.filter(
                id__in=recipient_ids, clicked_at__isnull=False
            ).count()

        metrics.append({
            "id": str(seq.id),
            "name": seq.name,
            "enrolled": enrolled,
            "active": active,
            "completed": completed,
            "replied": replied,
            "emails_sent": total_sent,
            "opens": opens,
            "clicks": clicks,
            "open_rate": round(opens / total_sent * 100, 1) if total_sent else 0,
            "click_rate": round(clicks / total_sent * 100, 1) if total_sent else 0,
        })

    return Response({"metrics": metrics})


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_crm_send_one(request):
    """Send a single ad-hoc email to a CRM lead."""
    import re
    from django.core.mail import send_mail as django_send_mail
    from api.models import EmailRecipient
    from api.views import make_unsubscribe_url

    lead_id = request.data.get("lead_id")
    subject = request.data.get("subject", "").strip()
    body_md = request.data.get("body", "").strip()

    if not lead_id or not subject or not body_md:
        return Response({"error": "lead_id, subject, and body are required."}, status=400)

    try:
        lead = CRMLead.objects.get(id=lead_id)
    except CRMLead.DoesNotExist:
        return Response({"error": "lead_not_found"}, status=404)

    if lead.email_opted_out:
        return Response({"error": "lead_opted_out"}, status=400)

    # Personalize
    body_md = body_md.replace("{{name}}", lead.name)
    body_html = _markdown_to_html(body_md)

    # Create campaign record
    campaign = EmailCampaign.objects.create(
        subject=subject,
        body_md=body_md,
        target=f"crm:lead:{lead.id}",
        sent_by=request.user,
        recipient_count=1,
    )
    rcpt = EmailRecipient.objects.create(
        campaign=campaign,
        email=lead.email,
    )

    # Rewrite links for click tracking
    def _track_link(match):
        url = match.group(1)
        return f'href="https://svend.ai/api/email/click/{rcpt.id}/?url={url}"'
    body_html = re.sub(r'href="(https?://[^"]+)"', _track_link, body_html)

    # Add tracking pixel
    pixel = f'<img src="https://svend.ai/api/email/open/{rcpt.id}/" width="1" height="1" style="display:none;" alt="">'
    unsub_url = "https://svend.ai"
    full_html = EMAIL_TEMPLATE.format(body=body_html + pixel, unsub_url=unsub_url)

    try:
        django_send_mail(
            subject=subject.replace("{{name}}", lead.name),
            message="",
            from_email=None,
            recipient_list=[lead.email],
            html_message=full_html,
        )
        # Update lead tracking
        lead.last_contacted_at = timezone.now()
        if lead.stage == "prospect":
            lead.stage = "contacted"
        lead.save(update_fields=["last_contacted_at", "stage", "updated_at"])

        return Response({
            "sent": True,
            "campaign_id": str(campaign.id),
            "recipient_id": str(rcpt.id),
        })
    except Exception as e:
        rcpt.failed = True
        rcpt.save(update_fields=["failed"])
        return Response({"error": str(e)}, status=500)


@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_crm_process_queue(request):
    """Process all due outreach sends: advance enrollments, send emails."""
    import re
    from django.core.mail import send_mail as django_send_mail
    from api.models import EmailRecipient

    now = timezone.now()
    due = OutreachEnrollment.objects.filter(
        status="active",
        next_send_at__lte=now,
    ).select_related("lead", "sequence")

    sent_count = 0
    failed_count = 0
    skipped_count = 0

    for enrollment in due:
        lead = enrollment.lead
        seq = enrollment.sequence

        if lead.email_opted_out:
            enrollment.status = "opted_out"
            enrollment.save(update_fields=["status"])
            skipped_count += 1
            continue

        if not seq.steps or enrollment.current_step >= len(seq.steps):
            enrollment.status = "completed"
            enrollment.save(update_fields=["status"])
            continue

        step = seq.steps[enrollment.current_step]
        variant = enrollment.variant

        # Pick subject/body based on variant
        subject = step.get(f"subject_{variant}", step.get("subject_a", ""))
        body_md = step.get(f"body_{variant}", step.get("body_a", ""))

        # Personalize
        subject = subject.replace("{{name}}", lead.name)
        body_md = body_md.replace("{{name}}", lead.name)
        body_html = _markdown_to_html(body_md)

        # Create email records
        campaign = EmailCampaign.objects.create(
            subject=subject,
            body_md=body_md,
            target=f"crm:lead:{lead.id}",
            sent_by=request.user,
            recipient_count=1,
        )
        rcpt = EmailRecipient.objects.create(
            campaign=campaign,
            email=lead.email,
        )

        # Rewrite links for click tracking
        def _track_link(match):
            url = match.group(1)
            return f'href="https://svend.ai/api/email/click/{rcpt.id}/?url={url}"'
        body_html = re.sub(r'href="(https?://[^"]+)"', _track_link, body_html)

        # Add tracking pixel
        pixel = f'<img src="https://svend.ai/api/email/open/{rcpt.id}/" width="1" height="1" style="display:none;" alt="">'
        unsub_url = "https://svend.ai"
        full_html = EMAIL_TEMPLATE.format(body=body_html + pixel, unsub_url=unsub_url)

        try:
            django_send_mail(
                subject=subject,
                message="",
                from_email=None,
                recipient_list=[lead.email],
                html_message=full_html,
            )
            sent_count += 1

            # Update lead
            lead.last_contacted_at = now
            if lead.stage == "prospect":
                lead.stage = "contacted"
            lead.save(update_fields=["last_contacted_at", "stage", "updated_at"])

            # Log the send
            enrollment.send_log.append({
                "step": enrollment.current_step,
                "sent_at": now.isoformat(),
                "recipient_id": str(rcpt.id),
                "variant": variant,
            })
            enrollment.last_sent_at = now

            # Advance to next step
            next_step = enrollment.current_step + 1
            if next_step >= len(seq.steps):
                enrollment.status = "completed"
                enrollment.next_send_at = None
            else:
                enrollment.current_step = next_step
                delay_days = seq.steps[next_step].get("delay_days", 1)
                enrollment.next_send_at = now + timedelta(days=delay_days)

            enrollment.save()

        except Exception:
            rcpt.failed = True
            rcpt.save(update_fields=["failed"])
            failed_count += 1

    return Response({
        "processed": sent_count + failed_count + skipped_count,
        "sent": sent_count,
        "failed": failed_count,
        "skipped": skipped_count,
    })

# ---------------------------------------------------------------------------
# API: Site Analytics (SiteVisit)
# ---------------------------------------------------------------------------


@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_site_analytics(request):
    """Site-wide visitor analytics: page views, unique visitors, referrers, top pages,
    country distribution, user flow, and recent hits."""
    days = _get_days(request)
    now = timezone.now()
    since = now - timedelta(days=days)
    visits = SiteVisit.objects.filter(viewed_at__gte=since, is_bot=False).exclude(path__contains="#_")

    # Daily visitors
    daily = list(
        visits.annotate(date=TruncDate("viewed_at"))
        .values("date")
        .annotate(total=Count("id"), unique=Count("ip_hash", distinct=True))
        .order_by("date")
    )

    # Top pages
    top_pages = list(
        visits.values("path")
        .annotate(total=Count("id"), unique=Count("ip_hash", distinct=True))
        .order_by("-total")[:20]
    )

    # Referrer domains
    referrers_raw = list(
        visits.exclude(referrer_domain="")
        .values("referrer_domain")
        .annotate(count=Count("id"))
        .order_by("-count")[:15]
    )
    direct = visits.filter(referrer_domain="").count()
    referrers = [{"domain": "Direct", "count": direct}] if direct else []
    referrers += [{"domain": r["referrer_domain"], "count": r["count"]} for r in referrers_raw]

    # Totals
    total_hits = visits.count()
    unique_visitors = visits.values("ip_hash").distinct().count()
    bot_hits = SiteVisit.objects.filter(viewed_at__gte=since, is_bot=True).count()

    # Country distribution (for world map)
    countries = list(
        visits.exclude(country="")
        .values("country")
        .annotate(views=Count("id"), unique=Count("ip_hash", distinct=True))
        .order_by("-views")
    )

    # User flow: page transitions (from → to)
    # Group visits by ip_hash, order by time, pair consecutive pages
    flow_counts = Counter()
    session_gap = timedelta(minutes=30)
    visitor_stream = (
        visits.values("ip_hash", "path", "viewed_at")
        .order_by("ip_hash", "viewed_at")
    )
    current_ip = None
    prev_path = None
    prev_time = None
    for v in visitor_stream.iterator():
        if v["ip_hash"] != current_ip:
            current_ip = v["ip_hash"]
            prev_path = v["path"]
            prev_time = v["viewed_at"]
            continue
        if v["viewed_at"] - prev_time <= session_gap:
            if prev_path != v["path"]:
                flow_counts[(prev_path, v["path"])] += 1
        prev_path = v["path"]
        prev_time = v["viewed_at"]

    flows = [
        {"from": k[0], "to": k[1], "count": c}
        for k, c in flow_counts.most_common(30)
    ]

    # Recent hits (live feed) — last 50, regardless of date range
    recent_raw = list(
        SiteVisit.objects.filter(is_bot=False)
        .order_by("-viewed_at")
        .values("path", "country", "referrer_domain", "viewed_at")[:50]
    )
    recent = []
    for r in recent_raw:
        delta = now - r["viewed_at"]
        secs = int(delta.total_seconds())
        if secs < 60:
            ago = f"{secs}s ago"
        elif secs < 3600:
            ago = f"{secs // 60}m ago"
        elif secs < 86400:
            ago = f"{secs // 3600}h ago"
        else:
            ago = f"{secs // 86400}d ago"
        recent.append({
            "path": r["path"],
            "country": r["country"] or "",
            "referrer": r["referrer_domain"] or "direct",
            "ago": ago,
        })

    # Duration stats — combine beacon data with server-side session estimation.
    # For multi-page sessions, time on page = (next pageview time - this pageview time).
    # This works for all visitors regardless of beacon reliability.
    from collections import defaultdict

    session_durations = defaultdict(list)  # path → [duration_ms, ...]
    session_gap = timedelta(minutes=30)
    visitor_pages = (
        visits.values("ip_hash", "path", "viewed_at", "duration_ms")
        .order_by("ip_hash", "viewed_at")
    )
    prev_ip = None
    prev_path = None
    prev_time = None
    prev_dur = None
    for v in visitor_pages.iterator():
        if v["ip_hash"] == prev_ip and prev_time:
            delta = v["viewed_at"] - prev_time
            if delta < session_gap and delta.total_seconds() > 0:
                est_ms = int(delta.total_seconds() * 1000)
                # Use beacon duration if available, otherwise use session estimate
                dur = prev_dur if prev_dur else est_ms
                # Clamp: skip if > 30min (session gap edge)
                if dur <= 1_800_000:
                    session_durations[prev_path].append(dur)
        # For the last page in a session, use beacon duration if available
        elif prev_ip is not None and prev_ip != v["ip_hash"] and prev_dur:
            if prev_dur <= 1_800_000:
                session_durations[prev_path].append(prev_dur)
        prev_ip = v["ip_hash"]
        prev_path = v["path"]
        prev_time = v["viewed_at"]
        prev_dur = v["duration_ms"]
    # Handle the very last visitor's last page
    if prev_dur and prev_dur <= 1_800_000:
        session_durations[prev_path].append(prev_dur)

    # Build page_durations from combined data
    page_durations = []
    for path, durations in sorted(session_durations.items(), key=lambda x: -len(x[1])):
        if durations:
            page_durations.append({
                "path": path,
                "avg_duration": sum(durations) / len(durations),
                "measured": len(durations),
            })
    page_durations = page_durations[:20]

    # Overall average from all measured durations
    all_durations = [d for durs in session_durations.values() for d in durs]
    overall_avg_ms = round(sum(all_durations) / len(all_durations)) if all_durations else None
    has_duration = len(all_durations)
    no_duration = total_hits - has_duration

    # Registration funnel (from funnel_event beacon — paths with #_)
    funnel_events = SiteVisit.objects.filter(
        viewed_at__gte=since, is_bot=False, path__contains="#_"
    )
    reg_page_views = visits.filter(path="/register/").values("ip_hash").distinct().count()
    funnel_data = {
        "page_views": reg_page_views,
    }
    for action in ("email_focus", "password_focus", "submit_attempt", "submit_error", "submit_success"):
        funnel_data[action] = (
            funnel_events
            .filter(path=f"/register/#_{action}")
            .values("ip_hash").distinct().count()
        )
    # Recent errors (detail stored in referrer_domain field)
    recent_errors = list(
        funnel_events
        .filter(path="/register/#_submit_error")
        .order_by("-viewed_at")
        .values_list("referrer_domain", "country", "viewed_at")[:10]
    )
    funnel_data["recent_errors"] = [
        {"error": e[0], "country": e[1], "at": str(e[2])} for e in recent_errors
    ]

    return Response({
        "daily": [
            {"date": str(d["date"]), "total": d["total"], "unique": d["unique"]}
            for d in daily
        ],
        "top_pages": [
            {"path": p["path"], "views": p["total"], "unique": p["unique"]}
            for p in top_pages
        ],
        "referrers": referrers,
        "totals": {
            "hits": total_hits,
            "unique_visitors": unique_visitors,
            "bot_hits": bot_hits,
            "avg_duration_ms": overall_avg_ms,
            "measured_visits": has_duration,
            "bounce_visits": no_duration,
        },
        "countries": [
            {"country": c["country"], "views": c["views"], "unique": c["unique"]}
            for c in countries
        ],
        "flows": flows,
        "recent": recent,
        "page_durations": [
            {
                "path": d["path"],
                "avg_ms": round(d["avg_duration"]),
                "measured": d["measured"],
            }
            for d in page_durations
        ],
        "registration_funnel": funnel_data,
    })


@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_site_live(request):
    """Lightweight live poll endpoint — returns recent hits + quick totals only.

    Skips expensive flow/duration aggregation for fast 15s polling.
    """
    now = timezone.now()
    limit = min(int(request.query_params.get("limit", 100)), 500)

    recent_raw = list(
        SiteVisit.objects.filter(is_bot=False)
        .order_by("-viewed_at")
        .values("path", "country", "referrer_domain", "viewed_at", "duration_ms", "ip_hash")[:limit]
    )

    recent = []
    for r in recent_raw:
        delta = now - r["viewed_at"]
        secs = int(delta.total_seconds())
        if secs < 60:
            ago = f"{secs}s ago"
        elif secs < 3600:
            ago = f"{secs // 60}m ago"
        elif secs < 86400:
            ago = f"{secs // 3600}h ago"
        else:
            ago = f"{secs // 86400}d ago"
        recent.append({
            "path": r["path"],
            "country": r["country"] or "",
            "referrer": r["referrer_domain"] or "direct",
            "ago": ago,
            "duration_ms": r["duration_ms"],
            "ts": r["viewed_at"].isoformat(),
            "visitor": r["ip_hash"][:8],
        })

    # Quick totals (cheap queries)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    hour_ago = now - timedelta(hours=1)

    hits_today = SiteVisit.objects.filter(viewed_at__gte=today, is_bot=False).count()
    hits_hour = SiteVisit.objects.filter(viewed_at__gte=hour_ago, is_bot=False).count()
    unique_today = SiteVisit.objects.filter(
        viewed_at__gte=today, is_bot=False
    ).values("ip_hash").distinct().count()

    return Response({
        "recent": recent,
        "totals": {
            "hits_today": hits_today,
            "hits_hour": hits_hour,
            "unique_today": unique_today,
        },
    })


# ---------------------------------------------------------------------------
# Whitepapers
# ---------------------------------------------------------------------------


@api_view(["GET"])
@permission_classes([IsInternalUser])

@api_view(["POST"])
@permission_classes([IsInternalUser])
def api_crm_bulk_send(request):
    """Schedule personalized A/B outreach emails to multiple leads via tempora.

    Expects: lead_ids, subject_a, body_a, subject_b, body_b.
    Pass preview: true to return per-lead assignments without sending.
    A/B variant assigned per lead via SHA256 hash (deterministic).
    Sends are staggered 5 seconds apart to avoid rate-limit spikes.
    """
    import hashlib

    d = request.data
    lead_ids = d.get("lead_ids", [])
    subject_a = d.get("subject_a", "").strip()
    body_a = d.get("body_a", "").strip()
    subject_b = d.get("subject_b", "").strip()
    body_b = d.get("body_b", "").strip()
    preview = d.get("preview", False)

    if not lead_ids:
        return Response({"error": "No leads selected."}, status=400)
    if not subject_a or not body_a:
        return Response({"error": "At least variant A subject and body are required."}, status=400)

    # Fall back to variant A if B is empty
    if not subject_b:
        subject_b = subject_a
    if not body_b:
        body_b = body_a

    leads = CRMLead.objects.filter(id__in=lead_ids, email_opted_out=False)
    if not leads.exists():
        return Response({"error": "No eligible leads found."}, status=400)

    # Build per-lead assignments
    assignments = []
    for lead in leads:
        hash_val = int(hashlib.sha256(f"{lead.id}-bulk".encode()).hexdigest(), 16) % 2
        variant = "a" if hash_val == 0 else "b"
        subj = subject_a if variant == "a" else subject_b
        body = body_a if variant == "a" else body_b

        # Personalize for preview
        p_subj = (
            subj
            .replace("{{name}}", lead.name)
            .replace("{{company}}", lead.company or "")
            .replace("{{role}}", lead.role or "")
            .replace("{{industry}}", lead.industry or "")
        )
        p_body = (
            body
            .replace("{{name}}", lead.name)
            .replace("{{company}}", lead.company or "")
            .replace("{{role}}", lead.role or "")
            .replace("{{industry}}", lead.industry or "")
        )

        assignments.append({
            "lead_id": str(lead.id),
            "name": lead.name,
            "email": lead.email,
            "company": lead.company,
            "variant": variant.upper(),
            "subject": p_subj,
            "body_preview": p_body[:200] + ("..." if len(p_body) > 200 else ""),
        })

    count_a = sum(1 for a in assignments if a["variant"] == "A")
    count_b = len(assignments) - count_a

    if preview:
        return Response({
            "assignments": assignments,
            "count_a": count_a,
            "count_b": count_b,
            "total": len(assignments),
        })

    # --- Actual send ---
    from syn.sched.scheduler import schedule_task

    campaign = EmailCampaign.objects.create(
        subject=f"[CRM Bulk] {subject_a[:80]}",
        body_md=body_a,
        target=f"crm:bulk:{len(assignments)}",
        sent_by=request.user,
        recipient_count=len(assignments),
    )

    scheduled = 0
    skipped = 0
    for i, lead in enumerate(leads):
        hash_val = int(hashlib.sha256(f"{lead.id}-bulk".encode()).hexdigest(), 16) % 2
        variant = "a" if hash_val == 0 else "b"
        subj = subject_a if variant == "a" else subject_b
        body = body_a if variant == "a" else body_b

        try:
            schedule_task(
                name=f"crm_bulk_{campaign.id}_{lead.id}",
                func="api.crm_send_one_email",
                args={
                    "lead_id": str(lead.id),
                    "subject": subj,
                    "body": body,
                    "campaign_id": str(campaign.id),
                    "variant": variant,
                },
                delay_seconds=i * 5,
                priority=2,
                queue="core",
            )
            scheduled += 1
        except Exception as e:
            logger.error("Failed to schedule CRM email for %s: %s", lead.email, e)
            skipped += 1

    return Response({
        "campaign_id": str(campaign.id),
        "scheduled": scheduled,
        "skipped": skipped,
        "count_a": count_a,
        "count_b": count_b,
        "stagger_seconds": 5,
        "total_time_estimate": f"{scheduled * 5}s",
    })


# =============================================================================
# Infrastructure (Synara OS layer)
# =============================================================================

@api_view(["GET"])
@permission_classes([IsInternalUser])
def api_infra(request):
    """Synara infrastructure overview: scheduler, audit trail, system logs."""
    now = timezone.now()

    # --- Scheduler ---
    try:
        from syn.sched.models import (
            CognitiveTask, Schedule, DeadLetterEntry, CircuitBreakerState,
        )

        task_states = dict(
            CognitiveTask.objects.values_list("state")
            .annotate(count=Count("id"))
            .values_list("state", "count")
        )

        schedules = list(
            Schedule.objects.order_by("schedule_id").values(
                "schedule_id", "name", "task_name", "enabled",
                "last_run_at", "next_run_at", "run_count",
            )
        )
        for s in schedules:
            s["last_run_at"] = str(s["last_run_at"]) if s["last_run_at"] else None
            s["next_run_at"] = str(s["next_run_at"]) if s["next_run_at"] else None

        dlq_by_status = dict(
            DeadLetterEntry.objects.values_list("status")
            .annotate(count=Count("id"))
            .values_list("status", "count")
        )

        circuit_breakers = list(
            CircuitBreakerState.objects.values(
                "service_name", "state", "failure_count",
                "last_failure_at", "opened_at",
            )
        )
        for cb in circuit_breakers:
            cb["last_failure_at"] = str(cb["last_failure_at"]) if cb["last_failure_at"] else None
            cb["opened_at"] = str(cb["opened_at"]) if cb["opened_at"] else None

        recent_failures = list(
            CognitiveTask.objects.filter(state="FAILURE")
            .order_by("-completed_at")[:10]
            .values("id", "task_name", "error_type", "error_message", "completed_at")
        )
        for f in recent_failures:
            f["id"] = str(f["id"])
            f["completed_at"] = str(f["completed_at"]) if f["completed_at"] else None

        scheduler_data = {
            "task_states": task_states,
            "schedules": schedules,
            "dlq": dlq_by_status,
            "circuit_breakers": circuit_breakers,
            "recent_failures": recent_failures,
        }
    except Exception as e:
        logger.warning("Infra: scheduler query failed: %s", e)
        scheduler_data = {"error": str(e)}

    # --- Audit Trail ---
    try:
        from syn.audit.models import SysLogEntry, IntegrityViolation, DriftViolation

        audit_total = SysLogEntry.objects.count()
        latest_entry = SysLogEntry.objects.order_by("-id").first()
        chain_ok = True
        chain_length = audit_total
        if latest_entry and latest_entry.current_hash:
            chain_ok = bool(latest_entry.current_hash)

        event_distribution = dict(
            SysLogEntry.objects.values_list("event_name")
            .annotate(count=Count("id"))
            .order_by("-count")
            .values_list("event_name", "count")[:15]
        )

        integrity_open = IntegrityViolation.objects.filter(is_resolved=False).count()
        integrity_total = IntegrityViolation.objects.count()

        drift_by_severity = dict(
            DriftViolation.objects.filter(resolved_at__isnull=True)
            .values_list("severity")
            .annotate(count=Count("id"))
            .values_list("severity", "count")
        )
        drift_total_open = sum(drift_by_severity.values())
        drift_sla_breached = DriftViolation.objects.filter(
            is_sla_breached=True, resolved_at__isnull=True,
        ).count()

        audit_data = {
            "total_entries": audit_total,
            "chain_length": chain_length,
            "chain_ok": chain_ok,
            "event_distribution": event_distribution,
            "integrity_violations_open": integrity_open,
            "integrity_violations_total": integrity_total,
            "drift_by_severity": drift_by_severity,
            "drift_total_open": drift_total_open,
            "drift_sla_breached": drift_sla_breached,
        }
    except Exception as e:
        logger.warning("Infra: audit query failed: %s", e)
        audit_data = {"error": str(e)}

    # --- System Logs ---
    try:
        from syn.log.models import LogEntry, LogStream

        log_level_counts = dict(
            LogEntry.objects.values_list("level")
            .annotate(count=Count("id"))
            .values_list("level", "count")
        )
        log_total = sum(log_level_counts.values())

        recent_errors = list(
            LogEntry.objects.filter(level__in=["ERROR", "CRITICAL"])
            .order_by("-timestamp")[:20]
            .values("id", "timestamp", "level", "logger", "message")
        )
        for entry in recent_errors:
            entry["id"] = str(entry["id"])
            entry["timestamp"] = str(entry["timestamp"])

        streams = list(
            LogStream.objects.values(
                "name", "is_active", "min_level", "retention_days",
            )
        )

        logs_data = {
            "total": log_total,
            "by_level": log_level_counts,
            "recent_errors": recent_errors,
            "streams": streams,
        }
    except Exception as e:
        logger.warning("Infra: log query failed: %s", e)
        logs_data = {"error": str(e)}

    return Response({
        "scheduler": scheduler_data,
        "audit": audit_data,
        "logs": logs_data,
    })

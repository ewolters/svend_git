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
from api.models import BlogPost, BlogView
from chat.models import EventLog, TraceLog, UsageLog

TIER_PRICES = {"founder": 19, "pro": 29, "team": 79, "enterprise": 199}
PAID_TIERS = list(TIER_PRICES.keys())


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
    return User.objects.filter(is_staff=False)


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
    """Staff user UUIDs for TraceLog filtering (uses UUIDField, not FK)."""
    return list(User.objects.filter(is_staff=True).values_list("id", flat=True))


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
    customers = _customers()
    staff_ids = _staff_ids()

    total_users = customers.count()
    active_today = customers.filter(last_active_at__date=today).count()

    day_usage = (
        UsageLog.objects.filter(date=today)
        .exclude(user__is_staff=True)
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
    logs = UsageLog.objects.filter(date__gte=since).exclude(user__is_staff=True)

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
@permission_classes([IsAdminUser])
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
@permission_classes([IsAdminUser])
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
    ).exclude(user__is_staff=True).count()
    active_subs = Subscription.objects.filter(
        status="active",
    ).exclude(user__is_staff=True).count()

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
</td></tr>
</table>
</td></tr>
</table>
</body>
</html>"""


@api_view(["POST"])
@permission_classes([IsAdminUser])
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
    elif target == "all":
        recipients = [(u, u.email) for u in _customers().exclude(email="")]
    elif target.startswith("tier:"):
        tier = target.split(":", 1)[1]
        recipients = [(u, u.email) for u in _customers().filter(tier=tier).exclude(email="")]
    elif "@" in target:
        recipients = [(None, target)]
    else:
        return Response({"error": "Invalid recipient target."}, status=400)

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

    sent = 0
    failed = 0
    for user, email in recipients:
        # Create recipient record
        rcpt = EmailRecipient.objects.create(
            campaign=campaign,
            user=user,
            email=email,
        )

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
        full_html = EMAIL_TEMPLATE.format(body=personalized + pixel)

        try:
            django_send_mail(
                subject=subject,
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
        "campaign_id": str(campaign.id),
    })


@api_view(["POST"])
@permission_classes([IsAdminUser])
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
@permission_classes([IsAdminUser])
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
@permission_classes([IsAdminUser])
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
                "created_at": c.created_at.isoformat(),
            }
            for c in campaigns
        ],
    })


# ---------------------------------------------------------------------------
# API: Activity (Event Tracking)
# ---------------------------------------------------------------------------

@api_view(["GET"])
@permission_classes([IsAdminUser])
def api_activity(request):
    days = _get_days(request)
    since = timezone.now() - timedelta(days=days)
    events = EventLog.objects.filter(created_at__gte=since).exclude(user__is_staff=True)

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
@permission_classes([IsAdminUser])
def api_onboarding(request):
    """Onboarding funnel, survey distributions, and email stats."""
    from api.models import OnboardingEmail, OnboardingSurvey

    customers = _customers()
    total = customers.count()
    completed = customers.filter(onboarding_completed_at__isnull=False).count()
    surveys = OnboardingSurvey.objects.filter(user__is_staff=False)

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
    emails = OnboardingEmail.objects.filter(user__is_staff=False)
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
@permission_classes([IsAdminUser])
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
@permission_classes([IsAdminUser])
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
@permission_classes([IsAdminUser])
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
@permission_classes([IsAdminUser])
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
@permission_classes([IsAdminUser])
def api_blog_delete(request, post_id):
    """Delete a blog post."""
    try:
        post = BlogPost.objects.get(id=post_id)
    except BlogPost.DoesNotExist:
        return Response({"error": "Post not found."}, status=404)
    post.delete()
    return Response({"deleted": True})


@api_view(["POST"])
@permission_classes([IsAdminUser])
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
@permission_classes([IsAdminUser])
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
    ).exclude(user__is_staff=True).count()

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

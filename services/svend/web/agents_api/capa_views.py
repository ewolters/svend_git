"""CAPA (Corrective and Preventive Action) views — Team/Enterprise tier.

ISO 9001:2015 §10.2 / FDA 21 CFR 820.90. Standalone CAPA lifecycle with
source linking, evidence hooks, and RCA bridge.
"""

import json
import logging
from collections import defaultdict
from datetime import date

from django.db.models import Avg, Count, F, Q, Sum
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from accounts.models import User
from accounts.permissions import require_team
from core.models import Project
from notifications.helpers import notify

from .evidence_bridge import create_tool_evidence
from .models import (
    CAPAReport,
    CAPAStatusChange,
    NonconformanceRecord,
    QMSFieldChange,
    RCASession,
)

logger = logging.getLogger(__name__)


def _log_field_changes(record_type, record_id, record, data, fields, user):
    """Log field-level changes for QMS records."""
    for field in fields:
        if field in data:
            old_val = getattr(record, field, "")
            new_val = data[field]
            if str(old_val or "") != str(new_val or ""):
                QMSFieldChange.objects.create(
                    record_type=record_type,
                    record_id=record_id,
                    field_name=field,
                    old_value=str(old_val or ""),
                    new_value=str(new_val or ""),
                    changed_by=user,
                )


def _ensure_capa_project(capa, user):
    """Create a Study (core.Project) for a CAPA if it doesn't have one."""
    if capa.project_id:
        return capa.project

    project = Project.objects.create(
        user=user,
        title=capa.title or "CAPA Investigation",
        methodology="none",
        tags=["auto-created", "capa"],
    )
    capa.project = project
    capa.save(update_fields=["project"])
    project.log_event("study_created", f"Auto-created from CAPA: {capa.title}", user=user)
    logger.info("Auto-created project %s for CAPA %s", project.id, capa.id)
    return project


def _capa_connect_investigation(request, investigation_id, capa, data):
    """CANON-002 §12 — connect CAPA findings to investigation graph."""
    from core.models import MeasurementSystem

    from .investigation_bridge import HypothesisSpec, connect_tool

    try:
        tool_output, _ = MeasurementSystem.objects.get_or_create(
            name="CAPA",
            owner=request.user,
            defaults={"system_type": "variable"},
        )
        specs = []
        if data.get("root_cause"):
            specs.append(HypothesisSpec(description=f"CAPA root cause: {data['root_cause'][:300]}", prior=0.6))
        if data.get("corrective_action"):
            specs.append(HypothesisSpec(description=f"CAPA corrective: {data['corrective_action'][:300]}", prior=0.5))
        if data.get("preventive_action"):
            specs.append(HypothesisSpec(description=f"CAPA preventive: {data['preventive_action'][:300]}", prior=0.5))
        if specs:
            connect_tool(
                investigation_id=investigation_id,
                tool_output=tool_output,
                tool_type="capa",
                user=request.user,
                spec=specs,
            )
    except Exception:
        logger.exception("CAPA investigation bridge error for %s", capa.id)


@require_team
@require_http_methods(["GET", "POST"])
def capa_list_create(request):
    """List CAPAs or create a new one."""
    user = request.user

    if request.method == "GET":
        capas = CAPAReport.objects.filter(owner=user)
        status = request.GET.get("status")
        priority = request.GET.get("priority")
        source_type = request.GET.get("source_type")
        assigned_to = request.GET.get("assigned_to")
        sort = request.GET.get("sort", "-created_at")
        if status:
            capas = capas.filter(status=status)
        if priority:
            capas = capas.filter(priority=priority)
        if source_type:
            capas = capas.filter(source_type=source_type)
        if assigned_to:
            capas = capas.filter(assigned_to_id=assigned_to)
        allowed_sorts = {
            "created_at",
            "-created_at",
            "priority",
            "-priority",
            "status",
            "-status",
            "due_date",
            "-due_date",
            "title",
            "-title",
        }
        if sort in allowed_sorts:
            capas = capas.order_by(sort)
        return JsonResponse([c.to_dict() for c in capas[:100]], safe=False)

    # POST — create
    data = json.loads(request.body)
    if not data.get("title"):
        return JsonResponse({"error": "title is required"}, status=400)

    assigned_to_user = None
    if data.get("assigned_to"):
        try:
            assigned_to_user = User.objects.get(id=data["assigned_to"])
        except User.DoesNotExist:
            pass

    capa = CAPAReport.objects.create(
        owner=user,
        assigned_to=assigned_to_user,
        title=data["title"],
        description=data.get("description", ""),
        priority=data.get("priority", "medium"),
        source_type=data.get("source_type", ""),
        source_id=data.get("source_id") or None,
        due_date=data.get("due_date") or None,
        containment_action=data.get("containment_action", ""),
    )

    # Link RCA session if provided
    rca_session_id = data.get("rca_session_id")
    if rca_session_id:
        try:
            rca = RCASession.objects.get(id=rca_session_id, owner=user)
            capa.rca_session = rca
            capa.save(update_fields=["rca_session"])
        except RCASession.DoesNotExist:
            pass

    # Auto-create Study
    _ensure_capa_project(capa, user)
    if capa.project:
        capa.project.log_event(
            "capa_created",
            f"CAPA raised: {capa.title} [{capa.priority}]",
            user=user,
        )

    return JsonResponse(capa.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def capa_detail(request, capa_id):
    """Get, update, or delete a CAPA."""
    try:
        capa = CAPAReport.objects.get(id=capa_id, owner=request.user)
    except CAPAReport.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(capa.to_dict())

    if request.method == "DELETE":
        capa.delete()
        return JsonResponse({"ok": True})

    # PUT — update
    data = json.loads(request.body)

    # Handle status transitions
    new_status = data.get("status")
    if new_status and new_status != capa.status:
        # Apply field updates before checking transition requirements
        capa_fields = [
            "title",
            "description",
            "containment_action",
            "root_cause",
            "corrective_action",
            "preventive_action",
            "verification_method",
            "verification_result",
        ]
        for field in capa_fields:
            if field in data:
                setattr(capa, field, data[field])

        ok, msg = capa.can_transition(new_status)
        if not ok:
            return JsonResponse({"error": msg}, status=400)

        old_status = capa.status
        capa.status = new_status

        CAPAStatusChange.objects.create(
            capa=capa,
            from_status=old_status,
            to_status=new_status,
            changed_by=request.user,
            note=data.get("status_note", ""),
        )

        # NTF-001 §10.1 — notify CAPA owner of status transition
        notify(
            recipient=capa.owner,
            notification_type="capa_status",
            title=f"CAPA {capa.title[:80]} moved to {new_status}",
            message=f"Status changed from {old_status} to {new_status}.",
            entity_type="capa",
            entity_id=capa.id,
        )

        if new_status == "closed" and not capa.closed_at:
            capa.closed_at = timezone.now()

    # Update fields (with change logging)
    capa_fields = [
        "title",
        "description",
        "containment_action",
        "root_cause",
        "corrective_action",
        "preventive_action",
        "verification_method",
        "verification_result",
        "priority",
    ]
    _log_field_changes("capa", capa.id, capa, data, capa_fields, request.user)
    for field in capa_fields:
        if field in data:
            setattr(capa, field, data[field])

    if "due_date" in data:
        capa.due_date = data["due_date"] or None
    if "cost_of_poor_quality" in data:
        capa.cost_of_poor_quality = data["cost_of_poor_quality"] or None
    if "copq_category" in data:
        capa.copq_category = data["copq_category"] or ""
    if "copq_paf_class" in data:
        capa.copq_paf_class = data["copq_paf_class"] or ""
    if "recurrence_check" in data:
        capa.is_recurrence_checked = data["recurrence_check"]
    if "effectiveness_check_date" in data:
        capa.effectiveness_check_date = data["effectiveness_check_date"] or None

    # Assigned_to
    if "assigned_to" in data:
        if data["assigned_to"]:
            try:
                capa.assigned_to = User.objects.get(id=data["assigned_to"])
            except User.DoesNotExist:
                pass
        else:
            capa.assigned_to = None

    # RCA session link
    if "rca_session_id" in data:
        if data["rca_session_id"]:
            try:
                rca = RCASession.objects.get(id=data["rca_session_id"], owner=request.user)
                capa.rca_session = rca
            except RCASession.DoesNotExist:
                pass
        else:
            capa.rca_session = None

    capa.save()

    _ensure_capa_project(capa, request.user)

    # CANON-002 §12 — investigation bridge
    investigation_id = data.get("investigation_id")
    if investigation_id and (data.get("root_cause") or data.get("corrective_action") or data.get("preventive_action")):
        _capa_connect_investigation(request, investigation_id, capa, data)

    # Recurrence detection on close (FEAT-012)
    if new_status == "closed":
        _check_recurrence(capa, request.user)

    # Evidence on close
    if new_status == "closed" and capa.project:
        create_tool_evidence(
            project=capa.project,
            user=request.user,
            summary=f"CAPA closed: {capa.title}",
            source_tool="capa",
            source_id=str(capa.id),
            source_field="status_closed",
            details=(
                f"Corrective: {capa.corrective_action or 'N/A'}\n"
                f"Preventive: {capa.preventive_action or 'N/A'}\n"
                f"Verification: {capa.verification_result or 'N/A'}"
            ),
            source_type="observation",
        )

    return JsonResponse(capa.to_dict())


@require_team
@require_http_methods(["GET"])
def capa_stats(request):
    """CAPA statistics for dashboard."""
    user = request.user
    capas = CAPAReport.objects.filter(owner=user)
    open_capas = capas.exclude(status="closed")
    overdue = open_capas.filter(due_date__lt=date.today()).count()

    avg_close = capas.filter(closed_at__isnull=False).aggregate(avg_days=Avg(F("closed_at") - F("created_at")))[
        "avg_days"
    ]
    avg_close_days = avg_close.days if avg_close else None

    by_status = {}
    for row in capas.values("status").annotate(c=Count("id")):
        by_status[row["status"]] = row["c"]

    by_priority = {}
    for row in capas.values("priority").annotate(c=Count("id")):
        by_priority[row["priority"]] = row["c"]

    # Aging: average time spent in each state (from CAPAStatusChange records)
    aging = {}
    changes = CAPAStatusChange.objects.filter(capa__owner=user).order_by("capa_id", "created_at")
    # Group by capa, compute time between consecutive transitions
    state_durations = defaultdict(list)
    prev_by_capa = {}
    for sc in changes:
        capa_pk = sc.capa_id
        if capa_pk in prev_by_capa:
            prev = prev_by_capa[capa_pk]
            delta = (sc.created_at - prev.created_at).total_seconds() / 86400.0
            state_durations[prev.to_status].append(delta)
        prev_by_capa[capa_pk] = sc
    for status_key, durations in state_durations.items():
        aging[status_key] = round(sum(durations) / len(durations), 1)

    return JsonResponse(
        {
            "total": capas.count(),
            "open": open_capas.count(),
            "overdue": overdue,
            "avg_close_days": avg_close_days,
            "by_status": by_status,
            "by_priority": by_priority,
            "aging": aging,
        }
    )


@require_team
@require_http_methods(["GET"])
def copq_summary(request):
    """Cost of Poor Quality summary with PAF breakdown.

    GET /api/capa/copq/?months=12
    Returns total CoPQ, breakdown by category and PAF class, monthly trending.
    """
    from django.db.models.functions import TruncMonth

    user = request.user
    months = int(request.GET.get("months", 12))
    cutoff = date.today() - __import__("datetime").timedelta(days=months * 30)
    capas = CAPAReport.objects.filter(
        owner=user,
        cost_of_poor_quality__isnull=False,
        created_at__gte=cutoff,
    )

    total_copq = capas.aggregate(total=Sum("cost_of_poor_quality"))["total"] or 0

    # By cost category
    by_category = [
        {"category": row["copq_category"] or "unclassified", "total": str(row["total"])}
        for row in capas.values("copq_category").annotate(total=Sum("cost_of_poor_quality")).order_by("-total")
    ]

    # By PAF class
    by_paf = [
        {"paf_class": row["copq_paf_class"] or "unclassified", "total": str(row["total"])}
        for row in capas.values("copq_paf_class").annotate(total=Sum("cost_of_poor_quality")).order_by("-total")
    ]

    # Monthly trending
    monthly = (
        capas.annotate(month=TruncMonth("created_at"))
        .values("month")
        .annotate(total=Sum("cost_of_poor_quality"), count=Count("id"))
        .order_by("month")
    )
    trending = [
        {"month": row["month"].strftime("%Y-%m"), "total": str(row["total"]), "count": row["count"]} for row in monthly
    ]

    return JsonResponse(
        {
            "total_copq": str(total_copq),
            "by_category": by_category,
            "by_paf_class": by_paf,
            "trending": trending,
            "capa_count": capas.count(),
            "period_months": months,
        }
    )


@require_team
@require_http_methods(["POST"])
def capa_launch_rca(request, capa_id):
    """Create an RCA session linked to this CAPA."""
    try:
        capa = CAPAReport.objects.get(id=capa_id, owner=request.user)
    except CAPAReport.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    _ensure_capa_project(capa, request.user)

    # Pre-populate RCA from source NCR if available
    event_text = capa.description or capa.title
    rca_chain = []
    if capa.source_type == "ncr" and capa.source_id:
        try:
            ncr = NonconformanceRecord.objects.get(
                id=capa.source_id,
                owner=request.user,
            )
            event_text = f"{ncr.title}\n\n{ncr.description}" if ncr.description else ncr.title
            # Seed chain with NCR containment if present
            if ncr.containment_action:
                rca_chain.append(
                    {
                        "claim": f"Containment (from NCR): {ncr.containment_action}",
                        "source": "ncr",
                        "accepted": True,
                    }
                )
        except NonconformanceRecord.DoesNotExist:
            pass

    session = RCASession.objects.create(
        owner=request.user,
        title=f"RCA for CAPA: {capa.title}",
        event=event_text,
        chain=rca_chain,
        project=capa.project,
        status="draft",
    )
    capa.rca_session = session
    capa.save(update_fields=["rca_session"])

    if capa.project:
        capa.project.log_event(
            "rca_launched",
            f"RCA session created from CAPA: {session.title}",
            user=request.user,
        )

    return JsonResponse(
        {
            "ok": True,
            "rca_session_id": str(session.id),
            "pre_populated_from_ncr": bool(rca_chain),
        },
        status=201,
    )


# =========================================================================
# Recurrence Detection (FEAT-012, ISO 9001 §10.2)
# =========================================================================


def _check_recurrence(capa, user):
    """Check if this CAPA's root cause matches historical patterns.

    Matches by: source_type, root_cause keyword overlap.
    If 3+ matches: auto-escalate priority to critical, mark recurrence flag.
    """
    if not capa.root_cause:
        return

    # Find historical closed CAPAs from same user with root causes
    historical = CAPAReport.objects.filter(owner=user, status="closed", root_cause__gt="").exclude(id=capa.id)

    # Match by source_type if present
    if capa.source_type:
        source_matches = historical.filter(source_type=capa.source_type)
    else:
        source_matches = historical

    # Keyword overlap: extract significant words (>3 chars) from root cause
    words = {w.lower() for w in capa.root_cause.split() if len(w) > 3}
    if not words:
        return

    # Build Q filter for keyword matching
    keyword_q = Q()
    for word in list(words)[:10]:  # Limit to 10 keywords
        keyword_q |= Q(root_cause__icontains=word)

    matches = source_matches.filter(keyword_q).distinct()
    match_count = matches.count()

    if match_count > 0:
        capa.is_recurrence_checked = True

    if match_count >= 3 and capa.priority != "critical":
        capa.priority = "critical"
        capa.save(update_fields=["priority", "is_recurrence_checked"])
        # Notify quality manager
        notify(
            recipient=user,
            notification_type="capa_status",
            title="Recurring Root Cause Detected",
            message=f"CAPA '{capa.title}' matches {match_count} previous CAPAs. Priority escalated to critical.",
            entity_type="capa",
            entity_id=capa.id,
        )
    elif match_count > 0:
        capa.save(update_fields=["is_recurrence_checked"])


@require_team
@require_http_methods(["GET"])
def recurrence_report(request):
    """Recurrence report — which root causes keep coming back.

    GET /api/capa/recurrence/
    Groups closed CAPAs by root_cause keyword similarity, flags repeat offenders.
    """
    user = request.user
    capas = CAPAReport.objects.filter(
        owner=user,
        status="closed",
        root_cause__gt="",
    ).order_by("-closed_at")

    # Group by source_type and find repeat root causes
    from collections import defaultdict

    by_source = defaultdict(list)
    for c in capas[:200]:  # Limit for performance
        by_source[c.source_type or "unknown"].append(
            {
                "id": str(c.id),
                "title": c.title,
                "root_cause": c.root_cause[:200],
                "priority": c.priority,
                "closed_at": c.closed_at.isoformat() if c.closed_at else None,
                "is_recurrence": c.is_recurrence_checked,
            }
        )

    recurrences = []
    for source_type, items in by_source.items():
        if len(items) >= 2:
            recurrences.append(
                {
                    "source_type": source_type,
                    "count": len(items),
                    "capas": items,
                    "escalated": any(i["priority"] == "critical" and i["is_recurrence"] for i in items),
                }
            )

    recurrences.sort(key=lambda r: r["count"], reverse=True)

    return JsonResponse(
        {
            "recurrences": recurrences,
            "total_closed_capas": capas.count(),
            "total_flagged": capas.filter(is_recurrence_checked=True).count(),
        }
    )

"""CAPA (Corrective and Preventive Action) views — Team/Enterprise tier.

DEPRECATED: CAPAReport model is superseded by Investigation (LOOP-001).
CAPA is now a generated report from investigation data via ForgeDoc, not
a standalone lifecycle. These views are retained for backward compatibility
with existing templates. Remove when iso_9001_qms.html is fully replaced
by the QMS workbench.

Original: ISO 9001:2015 §10.2 / FDA 21 CFR 820.90.
"""

import json
import logging
from collections import defaultdict
from datetime import date

from django.db.models import Avg, Count, F, Sum
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from accounts.models import User
from accounts.permissions import require_team
from agents_api.evidence_bridge import create_tool_evidence
from agents_api.models import (
    CAPAReport,
    CAPAStatusChange,
    NonconformanceRecord,
    QMSFieldChange,
    RCASession,
)
from agents_api.permissions import qms_can_edit, qms_queryset, qms_set_ownership
from core.models import Project
from notifications.helpers import notify

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
            specs.append(
                HypothesisSpec(
                    description=f"CAPA root cause: {data['root_cause'][:300]}",
                    prior=0.6,
                )
            )
        if data.get("corrective_action"):
            specs.append(
                HypothesisSpec(
                    description=f"CAPA corrective: {data['corrective_action'][:300]}",
                    prior=0.5,
                )
            )
        if data.get("preventive_action"):
            specs.append(
                HypothesisSpec(
                    description=f"CAPA preventive: {data['preventive_action'][:300]}",
                    prior=0.5,
                )
            )
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
        capas = qms_queryset(CAPAReport, user)[0]
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

    from .permissions import resolve_site

    site, err = resolve_site(request.user, data.get("site_id"))
    if err:
        return err

    capa = CAPAReport(
        assigned_to=assigned_to_user,
        title=data["title"],
        description=data.get("description", ""),
        priority=data.get("priority", "medium"),
        source_type=data.get("source_type", ""),
        source_id=data.get("source_id") or None,
        due_date=data.get("due_date") or None,
        containment_action=data.get("containment_action", ""),
    )
    qms_set_ownership(capa, user, site)
    capa.save()

    # Link RCA session if provided
    rca_session_id = data.get("rca_session_id")
    if rca_session_id:
        try:
            rca = qms_queryset(RCASession, user)[0].get(id=rca_session_id)
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
    qs, tenant, _is_admin = qms_queryset(CAPAReport, request.user)
    try:
        capa = qs.get(id=capa_id)
    except CAPAReport.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(capa.to_dict())

    if not qms_can_edit(request.user, capa, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

    if request.method == "DELETE":
        # Soft-delete: archive instead of destroying quality records (MED-02)
        old_status = capa.status
        capa.status = "cancelled"
        capa.save(update_fields=["status"])
        CAPAStatusChange.objects.create(
            capa=capa,
            from_status=old_status,
            to_status="cancelled",
            changed_by=request.user,
            note="Record archived via DELETE (soft-delete).",
        )
        return JsonResponse({"ok": True, "archived": True})

    # PUT — update
    data = json.loads(request.body)

    # Handle status transitions
    new_status = data.get("status")
    if new_status and new_status != capa.status:
        # Log field changes BEFORE setting values — captures true old_val (MED-01)
        transition_fields = [
            "title",
            "description",
            "containment_action",
            "root_cause",
            "corrective_action",
            "preventive_action",
            "verification_method",
            "verification_result",
        ]
        _log_field_changes("capa", capa.id, capa, data, transition_fields, request.user)
        # Apply field updates before checking transition requirements
        for field in transition_fields:
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
                rca = qms_queryset(RCASession, request.user)[0].get(id=data["rca_session_id"])
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

    # D3: Auto-create closure signature on CAPA close
    if new_status == "closed":
        from .iso_views import _create_workflow_signature

        _create_workflow_signature(request, "capa", capa.id, "approved", record=capa)

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
    capas = qms_queryset(CAPAReport, user)[0]
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
    capa_qs = qms_queryset(CAPAReport, user)[0]
    changes = CAPAStatusChange.objects.filter(capa__in=capa_qs).order_by("capa_id", "created_at")
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
    capas = qms_queryset(CAPAReport, user)[0].filter(
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
        {
            "paf_class": row["copq_paf_class"] or "unclassified",
            "total": str(row["total"]),
        }
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
        {
            "month": row["month"].strftime("%Y-%m"),
            "total": str(row["total"]),
            "count": row["count"],
        }
        for row in monthly
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
    qs, tenant, _is_admin = qms_queryset(CAPAReport, request.user)
    try:
        capa = qs.get(id=capa_id)
    except CAPAReport.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)
    if not qms_can_edit(request.user, capa, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

    _ensure_capa_project(capa, request.user)

    # Pre-populate RCA from source NCR if available
    event_text = capa.description or capa.title
    rca_chain = []
    if capa.source_type == "ncr" and capa.source_id:
        try:
            ncr = qms_queryset(NonconformanceRecord, request.user)[0].get(
                id=capa.source_id,
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

    session = RCASession(
        title=f"RCA for CAPA: {capa.title}",
        event=event_text,
        chain=rca_chain,
        project=capa.project,
        status="draft",
    )
    qms_set_ownership(session, request.user, capa.site)
    session.save()
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


_STOP_WORDS = frozenset(
    "the and for are but not you all any can had her was one our out has been have does that with this will your from they"
    " been into more when than them some what also each made were said does most only over such some very after before".split()
)


def _extract_keywords(text):
    """Extract significant keywords from text, filtering stop words and short tokens."""
    import re

    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if len(w) > 3 and w not in _STOP_WORDS}


def _keyword_overlap_score(words_a, words_b):
    """Jaccard similarity between two keyword sets (0.0 to 1.0)."""
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _check_recurrence(capa, user):
    """E3: Check if this CAPA's root cause matches historical patterns.

    Uses keyword overlap scoring (Jaccard similarity) instead of naive
    icontains matching. Scores ≥0.3 count as a match. Threshold of 2+
    matches escalates to critical with notification.
    """
    if not capa.root_cause:
        return

    capa_keywords = _extract_keywords(capa.root_cause)
    if not capa_keywords:
        return

    historical = qms_queryset(CAPAReport, user)[0].filter(status="closed", root_cause__gt="").exclude(id=capa.id)

    # Score each historical CAPA by root cause similarity
    matches = []
    for hist in historical[:200]:
        hist_keywords = _extract_keywords(hist.root_cause)
        score = _keyword_overlap_score(capa_keywords, hist_keywords)
        if score >= 0.3:
            matches.append({"id": hist.id, "title": hist.title, "score": score})

    match_count = len(matches)
    if match_count > 0:
        capa.is_recurrence_checked = True

    if match_count >= 2 and capa.priority != "critical":
        capa.priority = "critical"
        capa.save(update_fields=["priority", "is_recurrence_checked"])
        top_match = max(matches, key=lambda m: m["score"])
        notify(
            recipient=user,
            notification_type="capa_status",
            title="Recurring Root Cause Detected",
            message=(
                f"CAPA '{capa.title}' matches {match_count} previous CAPAs "
                f"(strongest: '{top_match['title']}', similarity {top_match['score']:.0%}). "
                f"Priority escalated to critical."
            ),
            entity_type="capa",
            entity_id=capa.id,
        )
    elif match_count > 0:
        capa.save(update_fields=["is_recurrence_checked"])


@require_team
@require_http_methods(["GET"])
def recurrence_report(request):
    """E3: Recurrence report — root cause clusters across CAPAs.

    GET /api/capa/recurrence/
    Clusters closed CAPAs by root cause keyword similarity (greedy clustering
    with Jaccard ≥0.25 threshold). Returns clusters sorted by size, with
    shared keywords that characterize each cluster.
    """
    user = request.user
    capas = list(
        qms_queryset(CAPAReport, user)[0].filter(status="closed", root_cause__gt="").order_by("-closed_at")[:200]
    )

    if not capas:
        return JsonResponse({"clusters": [], "total_closed_capas": 0, "total_flagged": 0})

    # Extract keywords for each CAPA
    capa_data = []
    for c in capas:
        kw = _extract_keywords(c.root_cause)
        capa_data.append(
            {
                "id": str(c.id),
                "title": c.title,
                "root_cause": c.root_cause[:300],
                "priority": c.priority,
                "source_type": c.source_type or "unknown",
                "closed_at": c.closed_at.isoformat() if c.closed_at else None,
                "is_recurrence": c.is_recurrence_checked,
                "keywords": kw,
            }
        )

    # Greedy single-linkage clustering by keyword similarity
    clusters = []
    assigned = set()

    for i, item in enumerate(capa_data):
        if i in assigned:
            continue
        cluster = [item]
        assigned.add(i)
        for j in range(i + 1, len(capa_data)):
            if j in assigned:
                continue
            score = _keyword_overlap_score(item["keywords"], capa_data[j]["keywords"])
            if score >= 0.25:
                cluster.append(capa_data[j])
                assigned.add(j)
        if len(cluster) >= 2:
            # Find shared keywords across the cluster
            shared = set.intersection(*(c["keywords"] for c in cluster)) if cluster else set()
            clusters.append(
                {
                    "size": len(cluster),
                    "shared_keywords": sorted(shared)[:10],
                    "escalated": any(c["priority"] == "critical" and c["is_recurrence"] for c in cluster),
                    "capas": [{k: v for k, v in c.items() if k != "keywords"} for c in cluster],
                }
            )

    clusters.sort(key=lambda c: c["size"], reverse=True)

    total_flagged = sum(1 for c in capas if c.is_recurrence_checked)

    return JsonResponse(
        {
            "clusters": clusters,
            "total_closed_capas": len(capas),
            "total_flagged": total_flagged,
            "singleton_count": len(capa_data) - sum(c["size"] for c in clusters),
        }
    )

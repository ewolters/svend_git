"""ISO 9001 QMS views — Team/Enterprise tier.

Covers:
- NCR Tracker (clause 10.2) — complete
- Internal Audit Scheduler (clause 9.2) — complete
- Training Matrix (clause 7.2) — complete
- Management Review (clause 9.3) — complete
- Document Control (clause 7.5) — skeleton
- Supplier Management (clause 8.4) — skeleton
"""

import json
import logging
from datetime import date, timedelta

from django.db import IntegrityError
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from accounts.models import User
from accounts.permissions import require_team
from core.models import Evidence, Project, StudyAction
from notifications.helpers import notify

from .evidence_bridge import create_tool_evidence
from .models import (
    FMEA,
    AuditChecklist,
    AuditFinding,
    CAPAReport,
    ControlledDocument,
    DocumentRevision,
    DocumentStatusChange,
    ElectronicSignature,
    InternalAudit,
    ManagementReview,
    ManagementReviewTemplate,
    NCRStatusChange,
    NonconformanceRecord,
    QMSAttachment,
    QMSFieldChange,
    RCASession,
    SupplierRecord,
    SupplierStatusChange,
    TrainingRecord,
    TrainingRecordChange,
    TrainingRequirement,
)

logger = logging.getLogger(__name__)


def _log_field_changes(record_type, record_id, record, data, fields, user):
    """Log field-level changes for any QMS record.

    Args:
        record_type: "ncr", "audit", "document", "supplier"
        record_id: UUID of the record
        record: the Django model instance (to read old values)
        data: the incoming request data dict
        fields: list of field names to check
        user: the request.user
    """
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


def _ensure_ncr_project(ncr, user):
    """Silently create a Study (core.Project) for an NCR if it doesn't have one.

    Invisible to the user — no toast, no notification. The Study is
    plumbing, not UX. Tagged ["auto-created", "ncr"] so it can be
    filtered or annotated later.
    """
    if ncr.project_id:
        return ncr.project

    project = Project.objects.create(
        user=user,
        title=ncr.title or "NCR Investigation",
        methodology="none",
        tags=["auto-created", "ncr"],
    )
    ncr.project = project
    ncr.save(update_fields=["project"])
    project.log_event("study_created", f"Auto-created from NCR: {ncr.title}", user=user)
    logger.info("Auto-created project %s for NCR %s", project.id, ncr.id)
    return project


def _ncr_connect_investigation(request, investigation_id, ncr, data):
    """CANON-002 §12 — connect NCR findings to investigation graph."""
    from core.models import MeasurementSystem

    from .investigation_bridge import HypothesisSpec, connect_tool

    try:
        tool_output, _ = MeasurementSystem.objects.get_or_create(
            name="NCR",
            owner=request.user,
            defaults={"system_type": "variable"},
        )
        specs = []
        if data.get("root_cause"):
            specs.append(HypothesisSpec(description=f"NCR root cause: {data['root_cause'][:300]}", prior=0.6))
        if data.get("corrective_action"):
            specs.append(HypothesisSpec(description=f"NCR corrective: {data['corrective_action'][:300]}", prior=0.5))
        if specs:
            connect_tool(
                investigation_id=investigation_id,
                tool_output=tool_output,
                tool_type="ncr",
                user=request.user,
                spec=specs,
            )
    except Exception:
        logger.exception("NCR investigation bridge error for %s", ncr.id)


# =========================================================================
# Dashboard overview
# =========================================================================


@require_team
@require_http_methods(["GET"])
def iso_dashboard(request):
    """QMS dashboard overview — clause coverage, KPIs, trends."""
    user = request.user
    now = timezone.now()
    today = now.date()
    fourteen_days = today + timedelta(days=14)
    thirty_days = now + timedelta(days=30)

    # ---- NCR KPIs ----
    ncrs = NonconformanceRecord.objects.filter(owner=user)
    open_ncrs = ncrs.exclude(status="closed")
    overdue = open_ncrs.filter(capa_due_date__lt=today).count()
    capa_due_soon = list(
        open_ncrs.filter(
            capa_due_date__isnull=False,
            capa_due_date__lte=fourteen_days,
            capa_due_date__gte=today,
        ).values_list("id", "title", "capa_due_date")[:10]
    )

    # NCR by severity
    by_severity = {}
    for s in ["minor", "major", "critical"]:
        by_severity[s] = open_ncrs.filter(severity=s).count()

    # NCR trend — last 4 weeks
    trend = []
    for i in range(4):
        week_end = today - timedelta(days=i * 7)
        week_start = week_end - timedelta(days=7)
        cnt = ncrs.filter(created_at__date__gte=week_start, created_at__date__lt=week_end).count()
        trend.append({"week": str(week_start), "count": cnt})
    trend.reverse()

    # ---- Audit KPIs ----
    audits = InternalAudit.objects.filter(owner=user)
    upcoming_audits = list(
        audits.filter(
            scheduled_date__gte=today,
            status__in=["planned", "in_progress"],
        )
        .order_by("scheduled_date")
        .values("id", "title", "scheduled_date", "status")[:3]
    )
    for a in upcoming_audits:
        a["id"] = str(a["id"])
        a["scheduled_date"] = str(a["scheduled_date"])

    # ---- Training KPIs ----
    reqs = TrainingRequirement.objects.filter(owner=user)
    all_records = TrainingRecord.objects.filter(requirement__owner=user)
    total_records = all_records.count()
    complete_records = all_records.filter(status="complete").count()
    compliance_rate = round(complete_records / total_records * 100) if total_records else 100
    gaps_count = all_records.exclude(status="complete").count()
    expiring_count = all_records.filter(
        expires_at__isnull=False,
        expires_at__lte=thirty_days,
    ).count()

    # ---- Last review ----
    last_review = (
        ManagementReview.objects.filter(
            owner=user,
            status="complete",
        )
        .order_by("-meeting_date")
        .values("meeting_date")
        .first()
    )
    last_review_data = None
    if last_review:
        days_ago = (today - last_review["meeting_date"]).days
        last_review_data = {"date": str(last_review["meeting_date"]), "days_ago": days_ago}

    # ---- Document KPIs ----
    docs_qs = ControlledDocument.objects.filter(owner=user)
    review_due_count = docs_qs.filter(
        review_due_date__isnull=False,
        review_due_date__lte=fourteen_days,
        status="approved",
    ).count()

    # ---- Supplier KPIs ----
    suppliers_qs = SupplierRecord.objects.filter(owner=user)
    eval_overdue_count = suppliers_qs.filter(
        next_evaluation_date__isnull=False,
        next_evaluation_date__lt=today,
        status__in=["approved", "preferred", "conditional"],
    ).count()

    # ---- Clause coverage ----
    # Determine which clauses have active records (NCRs, audits, training, docs, reviews)
    active_clauses = set()
    for c in ncrs.values_list("iso_clause", flat=True):
        if c:
            active_clauses.add(c.split(".")[0] if "." in c else c)
    for a in audits.values_list("iso_clauses", flat=True):
        if a:
            for c in a:
                active_clauses.add(c.split(".")[0] if "." in c else c)
    for t in reqs.values_list("iso_clause", flat=True):
        if t:
            active_clauses.add(t.split(".")[0] if "." in t else t)
    for d in docs_qs.values_list("iso_clause", flat=True):
        if d:
            active_clauses.add(d.split(".")[0] if "." in d else d)

    ISO_CLAUSES = [
        ("4", "Context of the Organization"),
        ("5", "Leadership"),
        ("6", "Planning"),
        ("7", "Support"),
        ("8", "Operation"),
        ("9", "Performance Evaluation"),
        ("10", "Improvement"),
    ]
    clause_coverage = []
    for clause_num, clause_name in ISO_CLAUSES:
        if clause_num in active_clauses:
            status = "active"
        elif reqs.count() > 0 or audits.count() > 0:
            status = "framed"
        else:
            status = "planned"
        clause_coverage.append({"clause": clause_num, "name": clause_name, "status": status})

    return JsonResponse(
        {
            "clause_coverage": clause_coverage,
            "ncrs": {
                "open": open_ncrs.count(),
                "by_severity": by_severity,
                "overdue_capas": overdue,
                "trend": trend,
            },
            "upcoming_audits": upcoming_audits,
            "training": {
                "compliance_rate": compliance_rate,
                "gaps_count": gaps_count,
                "expiring_count": expiring_count,
            },
            "last_review": last_review_data,
            "capa_due_soon": [{"id": str(c[0]), "title": c[1], "due_date": str(c[2])} for c in capa_due_soon],
            "documents": {
                "total": docs_qs.count(),
                "approved": docs_qs.filter(status="approved").count(),
                "review_due_soon": review_due_count,
            },
            "suppliers": {
                "total": suppliers_qs.count(),
                "approved": suppliers_qs.filter(status="approved").count(),
                "eval_overdue": eval_overdue_count,
            },
        }
    )


# =========================================================================
# Team Members (for QMS assignment dropdowns)
# =========================================================================


@require_team
@require_http_methods(["GET"])
def team_members(request):
    """Return users available for QMS assignment."""
    from core.models.tenant import Membership

    user = request.user
    members = [{"id": str(user.id), "name": user.get_full_name() or user.email}]
    seen = {user.id}
    # Add team members if user belongs to any tenant
    try:
        tenant_memberships = (
            Membership.objects.filter(
                tenant__membership__user=user,
                is_active=True,
            )
            .select_related("user")
            .exclude(user=user)
        )
        for m in tenant_memberships:
            if m.user_id not in seen:
                seen.add(m.user_id)
                members.append({"id": str(m.user.id), "name": m.user.get_full_name() or m.user.email})
    except Exception:
        pass  # Individual users without tenants
    return JsonResponse({"members": members})


# =========================================================================
# NCR Tracker
# =========================================================================


@require_team
@require_http_methods(["GET", "POST"])
def ncr_list_create(request):
    """List NCRs or create a new one."""
    user = request.user

    if request.method == "GET":
        ncrs = NonconformanceRecord.objects.filter(owner=user)
        status = request.GET.get("status")
        severity = request.GET.get("severity")
        assigned_to = request.GET.get("assigned_to")
        iso_clause = request.GET.get("iso_clause")
        sort = request.GET.get("sort", "-created_at")
        if status:
            ncrs = ncrs.filter(status=status)
        if severity:
            ncrs = ncrs.filter(severity=severity)
        if assigned_to:
            ncrs = ncrs.filter(assigned_to_id=assigned_to)
        if iso_clause:
            ncrs = ncrs.filter(iso_clause__icontains=iso_clause)
        # Validate sort field
        allowed_sorts = {
            "created_at",
            "-created_at",
            "severity",
            "-severity",
            "status",
            "-status",
            "capa_due_date",
            "-capa_due_date",
            "title",
            "-title",
        }
        if sort in allowed_sorts:
            ncrs = ncrs.order_by(sort)
        return JsonResponse([n.to_dict() for n in ncrs[:100]], safe=False)

    data = json.loads(request.body)
    assigned_to_user = None
    if data.get("assigned_to"):
        try:
            assigned_to_user = User.objects.get(id=data["assigned_to"])
        except User.DoesNotExist:
            pass

    ncr = NonconformanceRecord.objects.create(
        owner=user,
        raised_by=user,
        assigned_to=assigned_to_user,
        title=data.get("title", ""),
        description=data.get("description", ""),
        severity=data.get("severity", "minor"),
        source=data.get("source", "other"),
        iso_clause=data.get("iso_clause", ""),
        containment_action=data.get("containment_action", ""),
        capa_due_date=data.get("capa_due_date") or None,
    )
    # Link existing RCA session if provided
    rca_session_id = data.get("rca_session_id")
    if rca_session_id:
        try:
            rca = RCASession.objects.get(id=rca_session_id, owner=user)
            ncr.rca_session = rca
            ncr.save(update_fields=["rca_session"])
        except RCASession.DoesNotExist:
            pass

    # Attach files if provided
    file_ids = data.get("file_ids", [])
    if file_ids:
        from files.models import UserFile

        files = UserFile.objects.filter(id__in=file_ids, user=user)
        ncr.files.set(files)

    # Auto-create Study (invisible)
    _ensure_ncr_project(ncr, user)
    if ncr.project:
        ncr.project.log_event("ncr_created", f"NCR raised: {ncr.title} [{ncr.severity}]", user=user)

    return JsonResponse(ncr.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def ncr_detail(request, ncr_id):
    """Get, update, or delete an NCR."""
    try:
        ncr = NonconformanceRecord.objects.get(id=ncr_id, owner=request.user)
    except NonconformanceRecord.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(ncr.to_dict())

    if request.method == "DELETE":
        ncr.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)

    # Handle status transitions with workflow enforcement
    new_status = data.get("status")
    if new_status and new_status != ncr.status:
        # Resolve FK references first, fail loudly on bad IDs (BUG-03)
        if "assigned_to" in data:
            if data["assigned_to"]:
                try:
                    ncr.assigned_to = User.objects.get(id=data["assigned_to"])
                except User.DoesNotExist:
                    return JsonResponse({"error": "Assigned user not found"}, status=400)
            else:
                ncr.assigned_to = None
        if "approved_by" in data:
            if data["approved_by"]:
                try:
                    ncr.approved_by = User.objects.get(id=data["approved_by"])
                except User.DoesNotExist:
                    return JsonResponse({"error": "Approver not found"}, status=400)
            else:
                ncr.approved_by = None

        # Validate transition AFTER FK resolution (BUG-04)
        ok, msg = ncr.can_transition(new_status)
        if not ok:
            ncr.refresh_from_db()
            return JsonResponse({"error": msg}, status=400)

        old_status = ncr.status
        ncr.status = new_status

        # Record status change
        NCRStatusChange.objects.create(
            ncr=ncr,
            from_status=old_status,
            to_status=new_status,
            changed_by=request.user,
            note=data.get("status_note", ""),
        )

        # NTF-001 §10.1 — notify assignee of NCR assignment during transition
        if ncr.assigned_to and ncr.assigned_to != request.user:
            notify(
                recipient=ncr.assigned_to,
                notification_type="ncr_assigned",
                title=f"NCR {ncr.title[:80]} assigned to you",
                message=f"Status: {new_status}.",
                entity_type="ncr",
                entity_id=ncr.id,
            )

        if new_status == "closed" and not ncr.closed_at:
            ncr.closed_at = timezone.now()

    # Update other fields (with change logging)
    ncr_fields = [
        "title",
        "description",
        "severity",
        "source",
        "iso_clause",
        "containment_action",
        "root_cause",
        "corrective_action",
        "verification_result",
    ]
    _log_field_changes("ncr", ncr.id, ncr, data, ncr_fields, request.user)
    for field in ncr_fields:
        if field in data:
            setattr(ncr, field, data[field])
    if "capa_due_date" in data:
        old_due = str(ncr.capa_due_date) if ncr.capa_due_date else ""
        new_due = str(data["capa_due_date"]) if data["capa_due_date"] else ""
        if old_due != new_due:
            QMSFieldChange.objects.create(
                record_type="ncr",
                record_id=ncr.id,
                field_name="capa_due_date",
                old_value=old_due,
                new_value=new_due,
                changed_by=request.user,
            )
        ncr.capa_due_date = data["capa_due_date"] or None
    # Handle assigned_to/approved_by if not already handled by transition
    if "assigned_to" in data and not new_status:
        if data["assigned_to"]:
            try:
                assignee = User.objects.get(id=data["assigned_to"])
                ncr.assigned_to = assignee
                # NTF-001 §10.1 — notify new assignee (standalone assignment)
                if assignee != request.user:
                    notify(
                        recipient=assignee,
                        notification_type="ncr_assigned",
                        title=f"NCR {ncr.title[:80]} assigned to you",
                        entity_type="ncr",
                        entity_id=ncr.id,
                    )
            except User.DoesNotExist:
                pass
        else:
            ncr.assigned_to = None
    if "approved_by" in data and not new_status:
        if data["approved_by"]:
            try:
                ncr.approved_by = User.objects.get(id=data["approved_by"])
            except User.DoesNotExist:
                pass
        else:
            ncr.approved_by = None
    # Link existing RCA session
    if "rca_session_id" in data:
        if data["rca_session_id"]:
            try:
                rca = RCASession.objects.get(id=data["rca_session_id"], owner=request.user)
                ncr.rca_session = rca
            except RCASession.DoesNotExist:
                pass
        else:
            ncr.rca_session = None
    # File attachments
    if "file_ids" in data:
        from files.models import UserFile

        files = UserFile.objects.filter(id__in=data["file_ids"], user=request.user)
        ncr.files.set(files)
    ncr.save()

    _ensure_ncr_project(ncr, request.user)

    # CANON-002 §12 — investigation bridge
    investigation_id = data.get("investigation_id")
    if investigation_id and (data.get("root_cause") or data.get("corrective_action")):
        _ncr_connect_investigation(request, investigation_id, ncr, data)

    # Evidence on status transition to closed
    if new_status == "closed" and ncr.project:
        create_tool_evidence(
            project=ncr.project,
            user=request.user,
            summary=f"NCR closed: {ncr.title}",
            source_tool="ncr",
            source_id=str(ncr.id),
            source_field="status_closed",
            details=(f"Resolution: {ncr.corrective_action or 'N/A'}\nVerification: {ncr.verification_result or 'N/A'}"),
            source_type="observation",
        )

    return JsonResponse(ncr.to_dict())


@require_team
@require_http_methods(["GET"])
def ncr_stats(request):
    """NCR statistics for dashboard."""
    user = request.user
    ncrs = NonconformanceRecord.objects.filter(owner=user)
    open_ncrs = ncrs.exclude(status="closed")
    overdue = open_ncrs.filter(capa_due_date__lt=date.today()).count()

    # Average close time
    from django.db.models import Avg, F

    avg_close = ncrs.filter(closed_at__isnull=False).aggregate(avg_days=Avg(F("closed_at") - F("created_at")))[
        "avg_days"
    ]
    avg_close_days = avg_close.days if avg_close else None

    return JsonResponse(
        {
            "total": ncrs.count(),
            "open": open_ncrs.count(),
            "overdue_capas": overdue,
            "avg_close_days": avg_close_days,
        }
    )


@require_team
@require_http_methods(["GET"])
def ncr_analytics(request):
    """NCR Pareto analysis and trending.

    GET /api/iso/ncrs/analytics/?severity=major&source=supplier&months=12
    Returns Pareto by source category and monthly trending.
    """

    from django.db.models import Count
    from django.db.models.functions import TruncMonth

    user = request.user
    ncrs = NonconformanceRecord.objects.filter(owner=user)

    # Optional filters
    severity = request.GET.get("severity")
    if severity:
        ncrs = ncrs.filter(severity=severity)
    source = request.GET.get("source")
    if source:
        ncrs = ncrs.filter(source=source)
    months = int(request.GET.get("months", 12))
    cutoff = date.today() - timedelta(days=months * 30)
    ncrs = ncrs.filter(created_at__gte=cutoff)

    # --- Pareto by source ---
    source_counts = ncrs.values("source").annotate(count=Count("id")).order_by("-count")
    total = sum(row["count"] for row in source_counts)
    cumulative = 0
    pareto = []
    for row in source_counts:
        cumulative += row["count"]
        pareto.append(
            {
                "source": row["source"],
                "count": row["count"],
                "percent": round(row["count"] / total * 100, 1) if total else 0,
                "cumulative_percent": round(cumulative / total * 100, 1) if total else 0,
            }
        )

    # --- Pareto by severity ---
    severity_counts = ncrs.values("severity").annotate(count=Count("id")).order_by("-count")
    severity_pareto = [{"severity": row["severity"], "count": row["count"]} for row in severity_counts]

    # --- Monthly trending ---
    monthly = (
        ncrs.annotate(month=TruncMonth("created_at")).values("month").annotate(count=Count("id")).order_by("month")
    )
    trending = [{"month": row["month"].strftime("%Y-%m"), "count": row["count"]} for row in monthly]

    return JsonResponse(
        {
            "pareto_by_source": pareto,
            "pareto_by_severity": severity_pareto,
            "trending": trending,
            "total": total,
            "filters": {
                "severity": severity,
                "source": source,
                "months": months,
            },
        }
    )


@require_team
@require_http_methods(["POST"])
def ncr_launch_rca(request, ncr_id):
    """Create an RCA session linked to this NCR."""
    try:
        ncr = NonconformanceRecord.objects.get(id=ncr_id, owner=request.user)
    except NonconformanceRecord.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    # Ensure NCR has a project before linking
    _ensure_ncr_project(ncr, request.user)

    session = RCASession.objects.create(
        owner=request.user,
        title=f"RCA for {ncr.title}",
        event=ncr.description or ncr.title,
        project=ncr.project,  # Land in same Study as NCR
        status="draft",
    )
    ncr.rca_session = session
    ncr.save(update_fields=["rca_session"])
    if ncr.project:
        ncr.project.log_event("rca_launched", f"RCA started for NCR: {ncr.title}", user=request.user)
    return JsonResponse({"rca_session_id": str(session.id)}, status=201)


@require_team
@require_http_methods(["POST", "DELETE"])
def ncr_files(request, ncr_id):
    """Attach or detach files from an NCR."""
    try:
        ncr = NonconformanceRecord.objects.get(id=ncr_id, owner=request.user)
    except NonconformanceRecord.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    data = json.loads(request.body)
    file_id = data.get("file_id")
    if not file_id:
        return JsonResponse({"error": "file_id required"}, status=400)

    from files.models import UserFile

    try:
        uf = UserFile.objects.get(id=file_id, user=request.user)
    except UserFile.DoesNotExist:
        return JsonResponse({"error": "File not found"}, status=404)

    if request.method == "POST":
        ncr.files.add(uf)
    else:
        ncr.files.remove(uf)

    return JsonResponse({"ok": True, "file_ids": [str(f.id) for f in ncr.files.all()]})


# =========================================================================
# Internal Audit Scheduler
# =========================================================================


@require_team
@require_http_methods(["GET", "POST"])
def audit_list_create(request):
    """List audits or create a new one."""
    user = request.user

    if request.method == "GET":
        audits = InternalAudit.objects.filter(owner=user)
        status = request.GET.get("status")
        if status:
            audits = audits.filter(status=status)
        return JsonResponse([a.to_dict() for a in audits[:50]], safe=False)

    data = json.loads(request.body)
    audit = InternalAudit.objects.create(
        owner=user,
        title=data.get("title", ""),
        scheduled_date=data.get("scheduled_date", date.today()),
        lead_auditor=data.get("lead_auditor", ""),
        iso_clauses=data.get("iso_clauses", []),
        departments=data.get("departments", []),
        scope=data.get("scope", ""),
    )
    return JsonResponse(audit.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def audit_detail(request, audit_id):
    """Get, update, or delete an audit."""
    try:
        audit = InternalAudit.objects.get(id=audit_id, owner=request.user)
    except InternalAudit.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(audit.to_dict())

    if request.method == "DELETE":
        audit.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)
    new_status = data.get("status")

    # Enforce: "complete" requires >= 1 finding
    if new_status == "complete" and audit.findings.count() == 0:
        return JsonResponse({"error": "Cannot complete audit with no findings"}, status=400)
    # Enforce: "report_issued" requires all findings not "open"
    if new_status == "report_issued":
        open_findings = audit.findings.filter(status="open").count()
        if open_findings > 0:
            return JsonResponse(
                {"error": f"Cannot issue report: {open_findings} finding(s) still open"},
                status=400,
            )

    audit_fields = ["title", "status", "lead_auditor", "iso_clauses", "departments", "scope", "summary"]
    _log_field_changes("audit", audit.id, audit, data, audit_fields, request.user)
    for field in audit_fields:
        if field in data:
            setattr(audit, field, data[field])
    if "scheduled_date" in data:
        old_sd = str(audit.scheduled_date) if audit.scheduled_date else ""
        if old_sd != str(data["scheduled_date"]):
            QMSFieldChange.objects.create(
                record_type="audit",
                record_id=audit.id,
                field_name="scheduled_date",
                old_value=old_sd,
                new_value=str(data["scheduled_date"]),
                changed_by=request.user,
            )
        audit.scheduled_date = data["scheduled_date"]
    if "completed_date" in data:
        audit.completed_date = data["completed_date"] or None
    if new_status == "complete" and not audit.completed_date:
        audit.completed_date = date.today()
    audit.save()
    return JsonResponse(audit.to_dict())


@require_team
@require_http_methods(["POST"])
def audit_finding_create(request, audit_id):
    """Add a finding to an audit. Auto-creates NCR for NC findings."""
    try:
        audit = InternalAudit.objects.get(id=audit_id, owner=request.user)
    except InternalAudit.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    data = json.loads(request.body)
    finding_type = data.get("finding_type", "observation")
    clause = data.get("iso_clause", "")

    finding = AuditFinding.objects.create(
        audit=audit,
        finding_type=finding_type,
        description=data.get("description", ""),
        iso_clause=clause,
        evidence=data.get("evidence", ""),
        corrective_action=data.get("corrective_action", ""),
        due_date=data.get("due_date") or None,
        status=data.get("status", "open"),
    )

    # Auto-create NCR for nonconformity findings
    ncr_id = None
    if finding_type in ("nc_major", "nc_minor"):
        severity_map = {"nc_major": "critical", "nc_minor": "major"}
        clause_label = f" — {clause}" if clause else ""
        ncr = NonconformanceRecord.objects.create(
            owner=request.user,
            raised_by=request.user,
            title=f"Audit Finding — {audit.title}{clause_label}",
            description=finding.description,
            severity=severity_map[finding_type],
            source="internal_audit",
            iso_clause=clause,
        )
        _ensure_ncr_project(ncr, request.user)
        finding.ncr = ncr
        finding.save(update_fields=["ncr"])
        ncr_id = str(ncr.id)

    result = finding.to_dict()
    if ncr_id:
        result["ncr_id"] = ncr_id
    return JsonResponse(result, status=201)


# =========================================================================
# Audit Checklists
# =========================================================================


@require_team
@require_http_methods(["GET", "POST"])
def audit_checklist_list_create(request):
    """List or create audit checklists."""
    user = request.user

    if request.method == "GET":
        checklists = AuditChecklist.objects.filter(owner=user)
        return JsonResponse([c.to_dict() for c in checklists], safe=False)

    data = json.loads(request.body)
    checklist = AuditChecklist.objects.create(
        owner=user,
        name=data.get("name", ""),
        iso_clause=data.get("iso_clause", ""),
        check_items=data.get("check_items", []),
    )
    return JsonResponse(checklist.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def audit_checklist_detail(request, checklist_id):
    """Get, update, or delete an audit checklist."""
    try:
        checklist = AuditChecklist.objects.get(id=checklist_id, owner=request.user)
    except AuditChecklist.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(checklist.to_dict())

    if request.method == "DELETE":
        checklist.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)
    for field in ["name", "iso_clause", "check_items"]:
        if field in data:
            setattr(checklist, field, data[field])
    checklist.save()
    return JsonResponse(checklist.to_dict())


# =========================================================================
# Training Matrix
# =========================================================================


@require_team
@require_http_methods(["GET", "POST"])
def training_list_create(request):
    """List training requirements or create a new one."""
    user = request.user

    if request.method == "GET":
        reqs = TrainingRequirement.objects.filter(owner=user).prefetch_related(
            "records", "records__changes", "records__changes__changed_by"
        )
        return JsonResponse([r.to_dict() for r in reqs], safe=False)

    data = json.loads(request.body)
    req = TrainingRequirement.objects.create(
        owner=user,
        name=data.get("name", ""),
        description=data.get("description", ""),
        iso_clause=data.get("iso_clause", ""),
        frequency_months=data.get("frequency_months", 0),
        is_mandatory=data.get("is_mandatory", False),
    )
    # Link to controlled document if provided
    doc_id = data.get("document_id")
    if doc_id:
        try:
            doc = ControlledDocument.objects.get(id=doc_id, owner=user)
            req.document = doc
            req.document_version = doc.current_version
            req.save()
        except ControlledDocument.DoesNotExist:
            pass
    return JsonResponse(req.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def training_detail(request, req_id):
    """Get, update, or delete a training requirement."""
    try:
        req = TrainingRequirement.objects.get(id=req_id, owner=request.user)
    except TrainingRequirement.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(req.to_dict())

    if request.method == "DELETE":
        req.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)
    for field in ["name", "description", "iso_clause", "frequency_months", "is_mandatory"]:
        if field in data:
            setattr(req, field, data[field])
    # Handle document linkage
    if "document_id" in data:
        doc_id = data["document_id"]
        if doc_id:
            try:
                doc = ControlledDocument.objects.get(id=doc_id, owner=request.user)
                req.document = doc
                req.document_version = doc.current_version
            except ControlledDocument.DoesNotExist:
                pass
        else:
            req.document = None
            req.document_version = ""
    req.save()
    return JsonResponse(req.to_dict())


@require_team
@require_http_methods(["POST", "DELETE"])
def training_record_files(request, record_id):
    """Attach or detach artifact files from a training record."""
    try:
        record = TrainingRecord.objects.get(
            id=record_id,
            requirement__owner=request.user,
        )
    except TrainingRecord.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    data = json.loads(request.body)
    file_id = data.get("file_id")
    if not file_id:
        return JsonResponse({"error": "file_id required"}, status=400)

    from files.models import UserFile

    try:
        uf = UserFile.objects.get(id=file_id, user=request.user)
    except UserFile.DoesNotExist:
        return JsonResponse({"error": "File not found"}, status=404)

    if request.method == "POST":
        record.artifacts.add(uf)
    else:
        record.artifacts.remove(uf)

    return JsonResponse({"ok": True, "artifact_ids": [str(f.id) for f in record.artifacts.all()]})


@require_team
@require_http_methods(["POST"])
def training_record_create(request, req_id):
    """Add a training record to a requirement."""
    try:
        req = TrainingRequirement.objects.get(id=req_id, owner=request.user)
    except TrainingRequirement.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    data = json.loads(request.body)
    record = TrainingRecord.objects.create(
        requirement=req,
        employee_name=data.get("employee_name", ""),
        employee_email=data.get("employee_email", ""),
        status=data.get("status", "not_started"),
        competency_level=max(0, min(4, int(data.get("competency_level", 0)))),
    )

    # If marking complete, set completed_at and compute expires_at
    if record.status == "complete":
        record.completed_at = timezone.now()
        if req.frequency_months > 0:
            record.expires_at = timezone.now() + timedelta(days=req.frequency_months * 30)
        record.save()

    return JsonResponse(record.to_dict(), status=201)


@require_team
@require_http_methods(["PUT", "DELETE"])
def training_record_update(request, record_id):
    """Update or delete a training record."""
    try:
        record = TrainingRecord.objects.get(
            id=record_id,
            requirement__owner=request.user,
        )
    except TrainingRecord.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "DELETE":
        record.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)

    def log_change(field, old_val, new_val):
        if str(old_val or "") != str(new_val or ""):
            TrainingRecordChange.objects.create(
                record=record,
                field_name=field,
                old_value=str(old_val or ""),
                new_value=str(new_val or ""),
                changed_by=request.user,
            )

    # Recertification: reset completion and recalculate expiry
    if data.get("action") == "recertify":
        req = record.requirement
        old_completed = record.completed_at.isoformat() if record.completed_at else ""
        old_expires = record.expires_at.isoformat() if record.expires_at else ""
        old_status = record.status

        record.status = "complete"
        record.completed_at = timezone.now()
        record.expires_at = (
            timezone.now() + timedelta(days=req.frequency_months * 30) if req.frequency_months > 0 else None
        )

        log_change("status", old_status, "complete")
        log_change("completed_at", old_completed, record.completed_at.isoformat())
        log_change("expires_at", old_expires, record.expires_at.isoformat() if record.expires_at else "")
        record.save()
        return JsonResponse(record.to_dict())

    # Standard field updates with change logging
    for field in ["employee_name", "employee_email", "notes"]:
        if field in data:
            old_val = getattr(record, field)
            log_change(field, old_val, data[field])
            setattr(record, field, data[field])

    if "competency_level" in data:
        new_level = max(0, min(4, int(data["competency_level"])))
        log_change("competency_level", record.competency_level, new_level)
        record.competency_level = new_level

    if "status" in data:
        old_status = record.status
        new_status = data["status"]
        log_change("status", old_status, new_status)
        record.status = new_status
        if new_status == "complete" and not record.completed_at:
            record.completed_at = timezone.now()
            req = record.requirement
            if req.frequency_months > 0:
                record.expires_at = timezone.now() + timedelta(days=req.frequency_months * 30)
            log_change("completed_at", "", record.completed_at.isoformat())
            if record.expires_at:
                log_change("expires_at", "", record.expires_at.isoformat())

    record.save()
    return JsonResponse(record.to_dict())


# =========================================================================
# Management Review
# =========================================================================


@require_team
@require_http_methods(["GET", "POST"])
def review_list_create(request):
    """List reviews or create a new one with auto-populated snapshot."""
    user = request.user

    if request.method == "GET":
        reviews = ManagementReview.objects.filter(owner=user)
        return JsonResponse([r.to_dict() for r in reviews[:50]], safe=False)

    # Auto-populate rich snapshot per ISO 9001:2015 clause 9.3.2
    ncrs = NonconformanceRecord.objects.filter(owner=user)
    audits = InternalAudit.objects.filter(owner=user)
    records = TrainingRecord.objects.filter(requirement__owner=user)
    total_records = records.count()
    complete_records = records.filter(status="complete").count()

    # Prior actions from most recent completed review
    prior_actions = {}
    last_review = (
        ManagementReview.objects.filter(
            owner=user,
            status="complete",
        )
        .order_by("-meeting_date")
        .first()
    )
    if last_review:
        prior_actions = last_review.outputs or {}

    # NCR summary
    open_ncrs = ncrs.exclude(status="closed")
    ncr_summary = {
        "total": ncrs.count(),
        "open": open_ncrs.count(),
        "closed": ncrs.filter(status="closed").count(),
        "by_severity": {
            "minor": open_ncrs.filter(severity="minor").count(),
            "major": open_ncrs.filter(severity="major").count(),
            "critical": open_ncrs.filter(severity="critical").count(),
        },
    }

    # Audit summary
    completed_audits = audits.filter(status__in=["complete", "report_issued"])
    open_findings = AuditFinding.objects.filter(
        audit__owner=user,
        status="open",
    ).count()
    audit_summary = {
        "completed": completed_audits.count(),
        "open_findings": open_findings,
    }

    # Training summary
    gap_count = records.exclude(status="complete").count()
    training_summary = {
        "compliance_rate": round(complete_records / total_records * 100) if total_records else 100,
        "gap_count": gap_count,
    }

    snapshot = {
        "prior_actions": prior_actions,
        "ncr_summary": ncr_summary,
        "audit_summary": audit_summary,
        "training_summary": training_summary,
        "captured_at": timezone.now().isoformat(),
    }

    data = json.loads(request.body) if request.body else {}

    # Resolve template
    template = None
    template_id = data.get("template_id")
    if template_id:
        try:
            template = ManagementReviewTemplate.objects.get(id=template_id)
        except ManagementReviewTemplate.DoesNotExist:
            return JsonResponse({"error": "Template not found"}, status=404)

    # Build section-based inputs from template
    inputs = {}
    if template:
        auto_data_map = {
            "prior_actions": prior_actions,
            "ncr_summary": ncr_summary,
            "audit_summary": audit_summary,
            "training_summary": training_summary,
        }
        for section in template.sections:
            key = section["key"]
            auto_data = auto_data_map.get(section.get("auto_query")) if section.get("data_source") == "auto" else None
            inputs[key] = {
                "title": section["title"],
                "content": "",
                "auto_data": auto_data,
            }

    review = ManagementReview.objects.create(
        owner=user,
        template=template,
        title=data.get("title", f"Management Review — {date.today().strftime('%B %Y')}"),
        meeting_date=data.get("meeting_date", date.today()),
        inputs=inputs or data.get("inputs", {}),
        data_snapshot=snapshot,
    )
    return JsonResponse(review.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def review_detail(request, review_id):
    """Get, update, or delete a management review."""
    try:
        review = ManagementReview.objects.get(id=review_id, owner=request.user)
    except ManagementReview.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(review.to_dict())

    if request.method == "DELETE":
        review.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)
    for field in ["title", "status", "attendees", "inputs", "outputs", "minutes"]:
        if field in data:
            setattr(review, field, data[field])
    if "meeting_date" in data:
        review.meeting_date = data["meeting_date"]
    review.save()
    return JsonResponse(review.to_dict())


# =========================================================================
# Management Review Templates
# =========================================================================


@require_team
@require_http_methods(["GET", "POST"])
def review_template_list_create(request):
    """List review templates or create a new one."""
    user = request.user

    if request.method == "GET":
        templates = ManagementReviewTemplate.objects.filter(owner=user)
        return JsonResponse([t.to_dict() for t in templates[:50]], safe=False)

    data = json.loads(request.body)
    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "Title is required"}, status=400)

    sections = data.get("sections", ManagementReviewTemplate.DEFAULT_SECTIONS)
    template = ManagementReviewTemplate.objects.create(
        owner=user,
        title=title,
        description=data.get("description", ""),
        sections=sections,
        is_default=data.get("is_default", False),
    )
    return JsonResponse(template.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def review_template_detail(request, template_id):
    """Get, update, or delete a review template."""
    try:
        template = ManagementReviewTemplate.objects.get(id=template_id, owner=request.user)
    except ManagementReviewTemplate.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(template.to_dict())

    if request.method == "DELETE":
        template.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)
    for field in ["title", "description", "sections", "is_default"]:
        if field in data:
            setattr(template, field, data[field])
    template.save()
    return JsonResponse(template.to_dict())


@require_team
@require_http_methods(["POST"])
def review_template_default(request):
    """Create (or reset) the ISO 9001 default template for this user."""
    user = request.user
    # Delete existing defaults
    ManagementReviewTemplate.objects.filter(owner=user, is_default=True).delete()
    template = ManagementReviewTemplate.objects.create(
        owner=user,
        title="ISO 9001 §9.3.2 Default",
        description="Standard management review template covering all ISO 9001:2015 §9.3.2 required inputs.",
        sections=ManagementReviewTemplate.DEFAULT_SECTIONS,
        is_default=True,
    )
    return JsonResponse(template.to_dict(), status=201)


# =========================================================================
# Document Control (clause 7.5)
# =========================================================================


@require_team
@require_http_methods(["GET", "POST"])
def document_list_create(request):
    """List controlled documents or create a new one."""
    user = request.user

    if request.method == "GET":
        docs = ControlledDocument.objects.filter(owner=user)
        status = request.GET.get("status")
        category = request.GET.get("category")
        search = request.GET.get("search")
        sort = request.GET.get("sort", "-updated_at")
        if status:
            docs = docs.filter(status=status)
        if category:
            docs = docs.filter(category=category)
        if search:
            from django.db.models import Q

            docs = docs.filter(Q(title__icontains=search) | Q(document_number__icontains=search))
        allowed_sorts = {
            "title",
            "-title",
            "document_number",
            "-document_number",
            "status",
            "-status",
            "current_version",
            "-current_version",
            "review_due_date",
            "-review_due_date",
            "created_at",
            "-created_at",
            "updated_at",
            "-updated_at",
        }
        if sort in allowed_sorts:
            docs = docs.order_by(sort)
        return JsonResponse([d.to_dict() for d in docs[:100]], safe=False)

    data = json.loads(request.body)
    doc = ControlledDocument.objects.create(
        owner=user,
        title=data.get("title", ""),
        document_number=data.get("document_number", ""),
        category=data.get("category", ""),
        iso_clause=data.get("iso_clause", ""),
        content=data.get("content", ""),
        review_due_date=data.get("review_due_date") or None,
        retention_years=data.get("retention_years", 7),
    )
    return JsonResponse(doc.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def document_detail(request, doc_id):
    """Get, update, or delete a controlled document."""
    try:
        doc = ControlledDocument.objects.get(id=doc_id, owner=request.user)
    except ControlledDocument.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(doc.to_dict())

    if request.method == "DELETE":
        doc.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)

    # Handle status transitions with workflow enforcement
    new_status = data.get("status")
    if new_status and new_status != doc.status:
        # Pre-set approved_by_user before transition check
        if "approved_by_user" in data and data["approved_by_user"]:
            try:
                doc.approved_by_user = User.objects.get(id=data["approved_by_user"])
            except User.DoesNotExist:
                pass

        ok, msg = doc.can_transition(new_status)
        if not ok:
            return JsonResponse({"error": msg}, status=400)

        old_status = doc.status
        old_version = doc.current_version

        # Revision cycle: approved → review creates a version snapshot and bumps version
        if old_status == "approved" and new_status == "review":
            DocumentRevision.objects.create(
                document=doc,
                version=old_version,
                content_snapshot=doc.content,
                change_summary=data.get("revision_note", "Revision cycle initiated"),
                changed_by=request.user,
            )
            # Bump version: "1.0" → "1.1", "2.3" → "2.4"
            try:
                parts = old_version.split(".")
                parts[-1] = str(int(parts[-1]) + 1)
                doc.current_version = ".".join(parts)
            except (ValueError, IndexError):
                doc.current_version = old_version + ".1"

            # Flag completed training records linked to this document for retraining
            # Loop individually to fire audit trail (bulk .update() bypasses save/signals)
            flagged_records = TrainingRecord.objects.filter(
                requirement__document=doc,
                status="complete",
            )
            for rec in flagged_records:
                TrainingRecordChange.objects.create(
                    record=rec,
                    field_name="status",
                    old_value="complete",
                    new_value="expired",
                    changed_by=request.user,
                )
                rec.status = "expired"
                rec.save(update_fields=["status"])

        doc.status = new_status

        # Set approved_at on approval
        if new_status == "approved":
            doc.approved_at = timezone.now()
            if doc.approved_by_user:
                doc.approved_by = doc.approved_by_user.display_name or doc.approved_by_user.email

        # Clear approved_at when leaving approved state
        if old_status == "approved" and new_status != "approved":
            doc.approved_at = None

        DocumentStatusChange.objects.create(
            document=doc,
            from_status=old_status,
            to_status=new_status,
            changed_by=request.user,
            note=data.get("status_note", ""),
        )

    # Update fields (skip status — handled above), with change logging
    doc_fields = ["title", "document_number", "category", "iso_clause", "current_version", "content"]
    _log_field_changes("document", doc.id, doc, data, doc_fields, request.user)
    for field in doc_fields:
        if field in data:
            setattr(doc, field, data[field])
    if "review_due_date" in data:
        doc.review_due_date = data["review_due_date"] or None
    if "retention_years" in data:
        old_ret = str(doc.retention_years)
        new_ret = str(data["retention_years"])
        if old_ret != new_ret:
            QMSFieldChange.objects.create(
                record_type="document",
                record_id=doc.id,
                field_name="retention_years",
                old_value=old_ret,
                new_value=new_ret,
                changed_by=request.user,
            )
        doc.retention_years = data["retention_years"]
    if "approved_by_user" in data and not new_status:
        if data["approved_by_user"]:
            try:
                doc.approved_by_user = User.objects.get(id=data["approved_by_user"])
            except User.DoesNotExist:
                pass
        else:
            doc.approved_by_user = None

    doc.save()
    return JsonResponse(doc.to_dict())


@require_team
@require_http_methods(["POST", "DELETE"])
def document_files(request, doc_id):
    """Attach or detach files from a document."""
    try:
        doc = ControlledDocument.objects.get(id=doc_id, owner=request.user)
    except ControlledDocument.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    data = json.loads(request.body)
    file_id = data.get("file_id")
    if not file_id:
        return JsonResponse({"error": "file_id required"}, status=400)

    from files.models import UserFile

    try:
        uf = UserFile.objects.get(id=file_id, user=request.user)
    except UserFile.DoesNotExist:
        return JsonResponse({"error": "File not found"}, status=404)

    if request.method == "POST":
        doc.files.add(uf)
    else:
        doc.files.remove(uf)

    return JsonResponse({"ok": True, "file_ids": [str(f.id) for f in doc.files.all()]})


# =========================================================================
# Supplier Management (clause 8.4)
# =========================================================================


@require_team
@require_http_methods(["GET", "POST"])
def supplier_list_create(request):
    """List suppliers or create a new one."""
    user = request.user

    if request.method == "GET":
        suppliers = SupplierRecord.objects.filter(owner=user)
        status = request.GET.get("status")
        supplier_type = request.GET.get("supplier_type")
        search = request.GET.get("search")
        sort = request.GET.get("sort", "name")
        if status:
            suppliers = suppliers.filter(status=status)
        if supplier_type:
            suppliers = suppliers.filter(supplier_type=supplier_type)
        if search:
            from django.db.models import Q

            suppliers = suppliers.filter(Q(name__icontains=search) | Q(contact_name__icontains=search))
        allowed_sorts = {
            "name",
            "-name",
            "status",
            "-status",
            "quality_rating",
            "-quality_rating",
            "next_evaluation_date",
            "-next_evaluation_date",
            "last_evaluation_date",
            "-last_evaluation_date",
            "created_at",
            "-created_at",
        }
        if sort in allowed_sorts:
            suppliers = suppliers.order_by(sort)
        return JsonResponse([s.to_dict() for s in suppliers[:100]], safe=False)

    data = json.loads(request.body)
    supplier = SupplierRecord.objects.create(
        owner=user,
        name=data.get("name", ""),
        supplier_type=data.get("supplier_type", "other"),
        contact_name=data.get("contact_name", ""),
        contact_email=data.get("contact_email", ""),
        contact_phone=data.get("contact_phone", ""),
        products_services=data.get("products_services", ""),
        notes=data.get("notes", ""),
    )
    return JsonResponse(supplier.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def supplier_detail(request, supplier_id):
    """Get, update, or delete a supplier."""
    try:
        supplier = SupplierRecord.objects.get(id=supplier_id, owner=request.user)
    except SupplierRecord.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(supplier.to_dict())

    if request.method == "DELETE":
        supplier.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)

    # Handle status transitions with workflow enforcement
    new_status = data.get("status")
    if new_status and new_status != supplier.status:
        # Pre-set fields that transitions depend on
        if "quality_rating" in data:
            supplier.quality_rating = data["quality_rating"]
        if "notes" in data:
            supplier.notes = data["notes"]
        if "disqualification_reason" in data:
            supplier.disqualification_reason = data["disqualification_reason"]

        ok, msg = supplier.can_transition(new_status)
        if not ok:
            return JsonResponse({"error": msg}, status=400)

        old_status = supplier.status
        supplier.status = new_status

        SupplierStatusChange.objects.create(
            supplier=supplier,
            from_status=old_status,
            to_status=new_status,
            changed_by=request.user,
            note=data.get("status_note", ""),
        )

    # Update other fields (skip fields already set during transition), with change logging
    transition_fields = {"notes", "quality_rating", "disqualification_reason"} if new_status else set()
    supplier_fields = [
        "name",
        "supplier_type",
        "contact_name",
        "contact_email",
        "contact_phone",
        "products_services",
        "quality_rating",
        "notes",
        "disqualification_reason",
    ]
    non_transition = {f: data[f] for f in supplier_fields if f in data and f not in transition_fields}
    _log_field_changes("supplier", supplier.id, supplier, non_transition, list(non_transition.keys()), request.user)
    for field, val in non_transition.items():
        setattr(supplier, field, val)
    if "next_evaluation_date" in data:
        supplier.next_evaluation_date = data["next_evaluation_date"] or None
    if "last_evaluation_date" in data:
        supplier.last_evaluation_date = data["last_evaluation_date"] or None
    # Evaluation scores — auto-compute quality_rating from average
    if "evaluation_scores" in data:
        supplier.evaluation_scores = data["evaluation_scores"]
        scores = [v for v in data["evaluation_scores"].values() if isinstance(v, (int, float))]
        if scores:
            avg = sum(scores) / len(scores)
            supplier.quality_rating = round(avg)
            # Auto-suspend on critically low score (avg < 2) if currently active
            if avg < 2 and supplier.status in ("approved", "preferred"):
                old_status = supplier.status
                supplier.status = "suspended"
                supplier.notes = supplier.notes or ""
                if supplier.notes:
                    supplier.notes += "\n"
                supplier.notes += f"Auto-suspended: evaluation average {avg:.1f} below threshold"
                SupplierStatusChange.objects.create(
                    supplier=supplier,
                    from_status=old_status,
                    to_status="suspended",
                    changed_by=request.user,
                    note=f"Auto-suspended: evaluation average {avg:.1f} < 2.0",
                )
    if "metadata" in data:
        supplier.metadata = data["metadata"]

    supplier.save()
    return JsonResponse(supplier.to_dict())


# =============================================================================
# Study Output Actions (Phase 7) — QMS routing from Studies
# =============================================================================


def _get_study_for_action(request):
    """Validate and return the study (project) from request body."""
    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        return None, {}, JsonResponse({"error": "Invalid JSON"}, status=400)

    project_id = data.get("project_id")
    if not project_id:
        return None, data, JsonResponse({"error": "project_id required"}, status=400)

    try:
        project = Project.objects.get(id=project_id, user=request.user)
    except Project.DoesNotExist:
        return None, data, JsonResponse({"error": "Study not found"}, status=404)

    return project, data, None


def _get_study_context(project):
    """Extract problem statement and root cause from Study evidence for pre-fill."""
    context = {"problem": project.title, "root_cause": ""}

    # Use 5W2H problem description if available
    whats = project.problem_whats or []
    if whats:
        context["problem"] = "; ".join(whats)

    # Pull directly from linked RCA sessions first (richer than evidence summary)
    rca = RCASession.objects.filter(project=project, root_cause__gt="").order_by("-updated_at").first()
    if rca:
        parts = []
        if rca.chain:
            chain_str = "\n".join(f"{i + 1}. {s.get('claim', '')}" for i, s in enumerate(rca.chain))
            parts.append(f"**Causal Chain:**\n{chain_str}")
        parts.append(f"**Root Cause:** {rca.root_cause}")
        if rca.countermeasure:
            parts.append(f"**Countermeasure:** {rca.countermeasure}")
        context["root_cause"] = "\n\n".join(parts)
    else:
        # Fall back to evidence-based root cause
        rca_evidence = (
            Evidence.objects.filter(project=project, source_description__contains=":root_cause")
            .order_by("-created_at")
            .first()
        )
        if rca_evidence:
            context["root_cause"] = rca_evidence.summary

    return context


@require_team
@require_http_methods(["POST"])
def study_raise_capa(request):
    """Create a CAPA from a Study (standalone CAPAReport).

    Pre-fills description and root cause from Study evidence.
    Creates StudyAction for traceability.
    """
    project, data, err = _get_study_for_action(request)
    if err:
        return err

    context = _get_study_context(project)
    title = data.get("title", f"CAPA — {project.title}")

    capa = CAPAReport.objects.create(
        owner=request.user,
        project=project,
        title=title,
        description=context["problem"],
        root_cause=context["root_cause"] or "",
        priority=data.get("priority", "medium"),
        source_type=data.get("source_type", ""),
        source_id=data.get("source_id") or None,
    )

    StudyAction.objects.create(
        project=project,
        action_type="raise_capa",
        target_type="capa",
        target_id=capa.id,
        notes=f"CAPA raised from Study: {project.title}",
        created_by=request.user,
    )

    project.log_event("capa_raised", f"CAPA created: {capa.title}", user=request.user)
    logger.info("Study %s: raised CAPA %s", project.id, capa.id)
    return JsonResponse(
        {
            "ok": True,
            "action": "raise_capa",
            "capa": capa.to_dict(),
        },
        status=201,
    )


@require_team
@require_http_methods(["POST"])
def study_schedule_audit(request):
    """Schedule a verification audit from a Study.

    Pre-fills scope from Study title and corrective action summary.
    Creates StudyAction for traceability.
    """
    project, data, err = _get_study_for_action(request)
    if err:
        return err

    context = _get_study_context(project)
    scheduled_date = data.get("scheduled_date")
    if not scheduled_date:
        scheduled_date = date.today() + timedelta(days=30)

    scope = data.get("scope", "")
    if not scope:
        scope = f"Verification audit for Study: {project.title}"
        if context["root_cause"]:
            scope += f"\nRoot cause: {context['root_cause'][:200]}"

    audit = InternalAudit.objects.create(
        owner=request.user,
        title=data.get("title", f"Verification Audit — {project.title}"),
        scheduled_date=scheduled_date,
        scope=scope,
        lead_auditor=data.get("lead_auditor", ""),
    )

    StudyAction.objects.create(
        project=project,
        action_type="schedule_audit",
        target_type="audit",
        target_id=audit.id,
        notes=f"Verification audit scheduled from Study: {project.title}",
        created_by=request.user,
    )

    project.log_event("audit_scheduled", f"Verification audit scheduled: {audit.title}", user=request.user)
    logger.info("Study %s: scheduled verification audit %s", project.id, audit.id)
    return JsonResponse(
        {
            "ok": True,
            "action": "schedule_audit",
            "audit": audit.to_dict(),
        },
        status=201,
    )


@require_team
@require_http_methods(["POST"])
def study_request_doc_update(request):
    """Request a document update from a Study.

    Creates a controlled document change request with justification
    linked to Study findings. Creates StudyAction for traceability.
    """
    project, data, err = _get_study_for_action(request)
    if err:
        return err

    context = _get_study_context(project)
    title = data.get("title", f"Document Update — {project.title}")

    # Build justification from study context
    justification = data.get("content", "")
    if not justification:
        justification = f"Change requested based on Study: {project.title}\n"
        if context["root_cause"]:
            justification += f"\nRoot cause: {context['root_cause'][:300]}"

    doc = ControlledDocument.objects.create(
        owner=request.user,
        title=title,
        category=data.get("category", "Change Request"),
        content=justification,
        document_number=data.get("document_number", ""),
        iso_clause=data.get("iso_clause", ""),
        source_study=project,
    )

    StudyAction.objects.create(
        project=project,
        action_type="request_doc_update",
        target_type="document",
        target_id=doc.id,
        notes=f"Document update requested from Study: {project.title}",
        created_by=request.user,
    )

    project.log_event("doc_update_requested", f"Document update requested: {title}", user=request.user)
    logger.info("Study %s: requested document update %s", project.id, doc.id)
    return JsonResponse(
        {
            "ok": True,
            "action": "request_doc_update",
            "document": doc.to_dict(),
        },
        status=201,
    )


@require_team
@require_http_methods(["POST"])
def study_flag_training_gap(request):
    """Flag a training gap from a Study.

    Creates a training requirement with justification linked to the Study.
    Creates StudyAction for traceability.
    """
    project, data, err = _get_study_for_action(request)
    if err:
        return err

    name = data.get("name", "")
    if not name:
        return JsonResponse({"error": "Training requirement name is required"}, status=400)

    description = data.get("description", "")
    if not description:
        description = f"Training gap identified from Study: {project.title}"

    req = TrainingRequirement.objects.create(
        owner=request.user,
        name=name,
        description=description,
        iso_clause=data.get("iso_clause", ""),
        frequency_months=data.get("frequency_months", 0),
        is_mandatory=data.get("is_mandatory", True),
    )

    StudyAction.objects.create(
        project=project,
        action_type="flag_training_gap",
        target_type="training",
        target_id=req.id,
        notes=f"Training gap flagged from Study: {project.title}",
        created_by=request.user,
    )

    project.log_event("training_gap_flagged", f"Training gap: {name}", user=request.user)
    logger.info("Study %s: flagged training gap %s", project.id, req.id)
    return JsonResponse(
        {
            "ok": True,
            "action": "flag_training_gap",
            "training": req.to_dict(),
        },
        status=201,
    )


@require_team
@require_http_methods(["POST"])
def study_flag_fmea_update(request):
    """Flag an FMEA for review from a Study.

    Sets FMEA status to "review" and creates a StudyAction link.
    Lightweight — no new record type needed.
    """
    project, data, err = _get_study_for_action(request)
    if err:
        return err

    fmea_id = data.get("fmea_id")
    if not fmea_id:
        return JsonResponse({"error": "fmea_id required"}, status=400)

    try:
        fmea = FMEA.objects.get(id=fmea_id, owner=request.user)
    except FMEA.DoesNotExist:
        return JsonResponse({"error": "FMEA not found"}, status=404)

    # Set status to review if it's not already
    if fmea.status != "review":
        fmea.status = "review"
        fmea.save(update_fields=["status", "updated_at"])

    StudyAction.objects.create(
        project=project,
        action_type="flag_fmea_update",
        target_type="fmea",
        target_id=fmea.id,
        notes=data.get("notes", f"FMEA review needed — see Study: {project.title}"),
        created_by=request.user,
    )

    project.log_event("fmea_review_flagged", f"FMEA flagged for review: {fmea.title}", user=request.user)
    logger.info("Study %s: flagged FMEA %s for review", project.id, fmea.id)
    return JsonResponse(
        {
            "ok": True,
            "action": "flag_fmea_update",
            "fmea_id": str(fmea.id),
            "fmea_title": fmea.title,
        },
        status=201,
    )


# =============================================================================
# Electronic Signatures — 21 CFR Part 11 (FEAT-003)
# =============================================================================

# Model → (class, owner_field) for document resolution
_SIGNABLE_MODELS = {
    "ncr": (NonconformanceRecord, "owner"),
    "capa": (CAPAReport, "owner"),
    "document": (ControlledDocument, "owner"),
    "review": (ManagementReview, "owner"),
    "audit": (InternalAudit, "owner"),
    "training": (TrainingRecord, "requirement__owner"),
    "fmea": (FMEA, "owner"),
}


def _get_client_ip(request):
    """Extract client IP from request headers."""
    xff = request.META.get("HTTP_X_FORWARDED_FOR")
    if xff:
        return xff.split(",")[0].strip()
    xri = request.META.get("HTTP_X_REAL_IP")
    if xri:
        return xri
    return request.META.get("REMOTE_ADDR", "unknown")


def _resolve_tenant_id(user):
    """Get tenant_id for a user, or None for individual users."""
    from core.models import Membership

    m = Membership.objects.filter(user=user).first()
    return m.tenant_id if m else None


@require_team
@require_http_methods(["GET", "POST"])
def signature_list_or_sign(request):
    """List signatures (GET) or create a new electronic signature (POST).

    POST requires re-authentication per 21 CFR Part 11 §11.100.
    """
    user = request.user

    if request.method == "GET":
        tenant_id = _resolve_tenant_id(user)
        if tenant_id:
            sigs = ElectronicSignature.objects.filter(tenant_id=tenant_id)
        else:
            sigs = ElectronicSignature.objects.filter(signer=user)

        doc_type = request.GET.get("document_type")
        doc_id = request.GET.get("document_id")
        if doc_type:
            sigs = sigs.filter(document_type=doc_type)
        if doc_id:
            sigs = sigs.filter(document_id=doc_id)

        return JsonResponse([s.to_dict() for s in sigs[:200]], safe=False)

    # POST — sign
    data = json.loads(request.body)
    document_type = data.get("document_type", "")
    document_id = data.get("document_id", "")
    meaning = data.get("meaning", "")
    password = data.get("password", "")
    reason = data.get("reason", "")

    # Validate required fields
    if not document_type or not document_id or not meaning:
        return JsonResponse(
            {"error": "document_type, document_id, and meaning are required"},
            status=400,
        )
    if not password:
        return JsonResponse({"error": "password is required for signing"}, status=400)

    # Validate document_type
    if document_type not in _SIGNABLE_MODELS:
        return JsonResponse(
            {"error": f"Invalid document_type: {document_type}. Valid: {', '.join(_SIGNABLE_MODELS.keys())}"},
            status=400,
        )

    # Validate meaning
    valid_meanings = [c[0] for c in ElectronicSignature.Meaning.choices]
    if meaning not in valid_meanings:
        return JsonResponse(
            {"error": f"Invalid meaning: {meaning}. Valid: {', '.join(valid_meanings)}"},
            status=400,
        )

    # Re-authenticate (21 CFR Part 11 §11.100)
    if not user.check_password(password):
        return JsonResponse({"error": "Password verification failed"}, status=403)

    # Rejection requires reason (21 CFR Part 11 §11.50)
    if meaning == "rejected" and not reason:
        return JsonResponse({"error": "Reason is required for rejection"}, status=400)

    # Resolve document
    model_cls, owner_field = _SIGNABLE_MODELS[document_type]
    try:
        doc = model_cls.objects.get(id=document_id, **{owner_field: user})
    except model_cls.DoesNotExist:
        return JsonResponse({"error": "Document not found"}, status=404)

    # Capture document snapshot (CFR §11.10(c))
    doc_snapshot = doc.to_dict() if hasattr(doc, "to_dict") else {"id": str(doc.id)}

    # Resolve tenant
    tenant_id = _resolve_tenant_id(user)
    client_ip = _get_client_ip(request)
    user_agent_str = request.META.get("HTTP_USER_AGENT", "")

    # Create signature
    try:
        sig = ElectronicSignature(
            signer=user,
            document_type=document_type,
            document_id=document_id,
            meaning=meaning,
            reason=reason,
            user_agent=user_agent_str,
            # SynaraImmutableLog fields
            event_name=f"esig.{document_type}.{meaning}",
            actor=user.email,
            actor_ip=client_ip,
            tenant_id=tenant_id,
            after_snapshot=doc_snapshot,
        )
        sig.save()
    except IntegrityError:
        return JsonResponse(
            {"error": "You have already signed this document with this meaning"},
            status=409,
        )

    logger.info(
        "E-Sig: %s signed %s/%s as %s",
        user.email,
        document_type,
        document_id,
        meaning,
    )

    # NTF-001 §10.1 — notify document owner of new signature
    doc_owner = getattr(doc, "owner", None)
    if doc_owner:
        notify(
            recipient=doc_owner,
            notification_type="esig_request",
            title=f"E-signature ({meaning}) on {document_type} recorded",
            message=f"Signed by {user.email}.",
            entity_type=document_type,
            entity_id=document_id,
        )

    return JsonResponse(sig.to_dict(), status=201)


@require_team
@require_http_methods(["GET"])
def signature_verify(request, sig_id):
    """Verify integrity of a single electronic signature."""
    try:
        sig = ElectronicSignature.objects.get(id=sig_id)
    except ElectronicSignature.DoesNotExist:
        return JsonResponse({"error": "Signature not found"}, status=404)

    # Access check
    tenant_id = _resolve_tenant_id(request.user)
    if tenant_id:
        if sig.tenant_id != tenant_id:
            return JsonResponse({"error": "Not found"}, status=404)
    else:
        if sig.signer_id != request.user.id:
            return JsonResponse({"error": "Not found"}, status=404)

    is_valid = sig.verify_integrity()
    return JsonResponse(
        {
            "id": str(sig.id),
            "is_valid": is_valid,
            "entry_hash": sig.entry_hash,
        }
    )


@require_team
@require_http_methods(["GET"])
def signature_verify_chain(request):
    """Verify hash chain integrity for all signatures in the tenant."""
    tenant_id = _resolve_tenant_id(request.user)
    result = ElectronicSignature.verify_chain(tenant_id=tenant_id)
    return JsonResponse(result)


# =========================================================================
# QMS Attachments (ISO 9001 §7.5 — Documented Information)
# =========================================================================


@require_team
@require_http_methods(["GET", "POST"])
def qms_attachment_list_create(request):
    """List or create QMS attachments.

    GET: /api/qms/attachments/?entity_type=capa&entity_id=<uuid>
    POST: /api/qms/attachments/ {entity_type, entity_id, file_id, description?, attachment_type?}
    """
    user = request.user

    if request.method == "GET":
        entity_type = request.GET.get("entity_type")
        entity_id = request.GET.get("entity_id")
        if not entity_type or not entity_id:
            return JsonResponse({"error": "entity_type and entity_id required"}, status=400)
        if entity_type not in QMSAttachment.ENTITY_MODEL_MAP:
            return JsonResponse(
                {"error": f"Invalid entity_type. Valid: {list(QMSAttachment.ENTITY_MODEL_MAP.keys())}"},
                status=400,
            )
        attachments = QMSAttachment.objects.filter(
            entity_type=entity_type,
            entity_id=entity_id,
            uploaded_by=user,
        ).select_related("file")
        return JsonResponse([a.to_dict() for a in attachments], safe=False)

    # POST — create attachment
    data = json.loads(request.body)
    entity_type = data.get("entity_type")
    entity_id = data.get("entity_id")
    file_id = data.get("file_id")

    if not entity_type or not entity_id or not file_id:
        return JsonResponse({"error": "entity_type, entity_id, and file_id required"}, status=400)

    if entity_type not in QMSAttachment.ENTITY_MODEL_MAP:
        return JsonResponse(
            {"error": f"Invalid entity_type. Valid: {list(QMSAttachment.ENTITY_MODEL_MAP.keys())}"},
            status=400,
        )

    # Validate the target entity exists and belongs to user
    model_name = QMSAttachment.ENTITY_MODEL_MAP[entity_type]
    from . import models as _models

    ModelClass = getattr(_models, model_name)
    owner_field = "owner" if hasattr(ModelClass, "owner") else "user"
    try:
        ModelClass.objects.get(id=entity_id, **{owner_field: user})
    except ModelClass.DoesNotExist:
        return JsonResponse({"error": f"{entity_type} not found"}, status=404)

    # Validate file exists and belongs to user
    from files.models import UserFile

    try:
        uf = UserFile.objects.get(id=file_id, user=user)
    except UserFile.DoesNotExist:
        return JsonResponse({"error": "File not found"}, status=404)

    attachment_type = data.get("attachment_type", "evidence")
    valid_types = [c[0] for c in QMSAttachment.AttachmentType.choices]
    if attachment_type not in valid_types:
        attachment_type = "evidence"

    attachment = QMSAttachment.objects.create(
        entity_type=entity_type,
        entity_id=entity_id,
        file=uf,
        uploaded_by=user,
        description=data.get("description", ""),
        attachment_type=attachment_type,
    )
    return JsonResponse(attachment.to_dict(), status=201)


@require_team
@require_http_methods(["DELETE"])
def qms_attachment_delete(request, attachment_id):
    """Delete a QMS attachment (does not delete the underlying file)."""
    try:
        attachment = QMSAttachment.objects.get(id=attachment_id, uploaded_by=request.user)
    except QMSAttachment.DoesNotExist:
        return JsonResponse({"error": "Attachment not found"}, status=404)

    attachment.delete()
    return JsonResponse({"ok": True})

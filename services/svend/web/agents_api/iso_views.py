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
    Checklist,
    ChecklistExecution,
    ControlledDocument,
    CustomerComplaint,
    DocumentRevision,
    DocumentStatusChange,
    ElectronicSignature,
    InternalAudit,
    ManagementReview,
    ManagementReviewTemplate,
    MeasurementEquipment,
    NCRStatusChange,
    NonconformanceRecord,
    QMSAttachment,
    QMSFieldChange,
    RCASession,
    Risk,
    Site,
    SupplierRecord,
    SupplierStatusChange,
    TrainingRecord,
    TrainingRecordChange,
    TrainingRequirement,
)
from .permissions import qms_can_edit, qms_queryset, qms_set_ownership

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
# Sites (for site selector in QMS forms)
# =========================================================================


@require_team
@require_http_methods(["GET"])
def qms_sites(request):
    """List sites accessible to the current user for QMS forms."""
    from .permissions import get_accessible_sites, get_tenant

    tenant = get_tenant(request.user)
    if not tenant:
        return JsonResponse({"sites": [], "has_org": False})

    sites, is_admin = get_accessible_sites(request.user, tenant)
    return JsonResponse(
        {
            "sites": [{"id": str(s.id), "name": s.name, "code": s.code} for s in sites.order_by("name")],
            "has_org": True,
            "is_org_admin": is_admin,
        }
    )


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
    ncrs = qms_queryset(NonconformanceRecord, user)[0]
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
    audits = qms_queryset(InternalAudit, user)[0]
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
    reqs = qms_queryset(TrainingRequirement, user)[0]
    all_records = TrainingRecord.objects.filter(requirement__in=reqs)
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
    docs_qs = qms_queryset(ControlledDocument, user)[0]
    review_due_count = docs_qs.filter(
        review_due_date__isnull=False,
        review_due_date__lte=fourteen_days,
        status="approved",
    ).count()

    # ---- Supplier KPIs ----
    suppliers_qs = qms_queryset(SupplierRecord, user)[0]
    eval_overdue_count = suppliers_qs.filter(
        next_evaluation_date__isnull=False,
        next_evaluation_date__lt=today,
        status__in=["approved", "preferred", "conditional"],
    ).count()

    # ---- Customer Complaint KPIs ----
    complaints_qs = qms_queryset(CustomerComplaint, user)[0]
    open_complaints = complaints_qs.exclude(status="closed").count()

    # ---- Risk Register KPIs ----
    risks_qs = qms_queryset(Risk, user)[0]
    active_risks = risks_qs.exclude(status="closed")
    high_risks = active_risks.filter(risk_score__gte=12).count()
    risks_review_overdue = active_risks.filter(review_date__isnull=False, review_date__lt=today).count()

    # ---- Calibration Equipment KPIs ----
    equipment_qs = qms_queryset(MeasurementEquipment, user)[0]
    cal_overdue = equipment_qs.filter(
        next_calibration_due__isnull=False,
        next_calibration_due__lt=today,
        status__in=["in_service", "due"],
    ).count()
    cal_due_soon = equipment_qs.filter(
        next_calibration_due__isnull=False,
        next_calibration_due__gte=today,
        next_calibration_due__lte=fourteen_days,
        status="in_service",
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

    # A5: Major audit findings without linked NCRs — ISO 9001 §9.2.2 requires corrective action
    major_findings_no_ncr = list(
        AuditFinding.objects.filter(
            audit__in=audits,
            finding_type="nc_major",
            ncr__isnull=True,
            is_resolved=False,
        ).values("id", "description", "iso_clause", "audit__title")[:10]
    )
    for f in major_findings_no_ncr:
        f["id"] = str(f["id"])

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
            "complaints": {
                "open": open_complaints,
                "total": complaints_qs.count(),
            },
            "risks": {
                "active": active_risks.count(),
                "high_risks": high_risks,
                "review_overdue": risks_review_overdue,
            },
            "calibration": {
                "total": equipment_qs.count(),
                "overdue": cal_overdue,
                "due_soon": cal_due_soon,
            },
            "warnings": {
                "major_findings_no_ncr": major_findings_no_ncr,
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
        ncrs, _tenant, _is_admin = qms_queryset(NonconformanceRecord, user)
        status = request.GET.get("status")
        severity = request.GET.get("severity")
        assigned_to = request.GET.get("assigned_to")
        iso_clause = request.GET.get("iso_clause")
        source = request.GET.get("source")
        site_id = request.GET.get("site_id")
        sort = request.GET.get("sort", "-created_at")
        if status:
            ncrs = ncrs.filter(status=status)
        if severity:
            ncrs = ncrs.filter(severity=severity)
        if assigned_to:
            ncrs = ncrs.filter(assigned_to_id=assigned_to)
        if iso_clause:
            ncrs = ncrs.filter(iso_clause__icontains=iso_clause)
        if source:
            ncrs = ncrs.filter(source=source)
        if site_id:
            ncrs = ncrs.filter(site_id=site_id)
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
        # Pagination
        limit = min(int(request.GET.get("limit", 100)), 500)
        offset = int(request.GET.get("offset", 0))
        total = ncrs.count()
        results = [n.to_dict() for n in ncrs[offset : offset + limit]]
        return JsonResponse({"results": results, "total": total, "limit": limit, "offset": offset})

    data = json.loads(request.body)
    assigned_to_user = None
    if data.get("assigned_to"):
        try:
            assigned_to_user = User.objects.get(id=data["assigned_to"])
        except User.DoesNotExist:
            pass

    # Resolve site if provided (ORG-001 §5.2)
    site = None
    site_id = data.get("site_id")
    if site_id:
        try:
            site = Site.objects.get(id=site_id)
        except Site.DoesNotExist:
            return JsonResponse({"error": "Site not found"}, status=404)

    # Resolve supplier if provided
    supplier = None
    supplier_id = data.get("supplier_id")
    if supplier_id:
        try:
            supplier = SupplierRecord.objects.get(id=supplier_id)
        except SupplierRecord.DoesNotExist:
            pass

    ncr = NonconformanceRecord(
        raised_by=user,
        assigned_to=assigned_to_user,
        title=data.get("title", ""),
        description=data.get("description", ""),
        severity=data.get("severity", "minor"),
        source=data.get("source", "other"),
        iso_clause=data.get("iso_clause", ""),
        containment_action=data.get("containment_action", ""),
        capa_due_date=data.get("capa_due_date") or None,
        supplier=supplier,
    )
    qms_set_ownership(ncr, user, site)
    ncr.save()
    # Link existing RCA session if provided
    rca_session_id = data.get("rca_session_id")
    if rca_session_id:
        try:
            rca = qms_queryset(RCASession, user)[0].get(id=rca_session_id)
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
    qs, tenant, _is_admin = qms_queryset(NonconformanceRecord, request.user)
    try:
        ncr = qs.get(id=ncr_id)
    except NonconformanceRecord.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(ncr.to_dict())

    if not qms_can_edit(request.user, ncr, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

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
        # Apply text fields that may be required for transition (e.g. root_cause for capa)
        for tf in ("root_cause", "corrective_action", "containment_action", "verification_result"):
            if tf in data:
                setattr(ncr, tf, data[tf])

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

        if new_status == "closed":
            now = timezone.now()
            if not ncr.approved_at:
                ncr.approved_at = now
            if not ncr.closed_at:
                ncr.closed_at = now

            # NTF-001 — notify approver when NCR is closed with their approval
            if ncr.approved_by and ncr.approved_by != request.user:
                notify(
                    recipient=ncr.approved_by,
                    notification_type="ncr_assigned",
                    title=f"NCR {ncr.title[:80]} closed with your approval",
                    message="The NCR has been verified and closed.",
                    entity_type="ncr",
                    entity_id=ncr.id,
                )

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
        old_name = (ncr.assigned_to.display_name or ncr.assigned_to.email) if ncr.assigned_to else ""
        if data["assigned_to"]:
            try:
                assignee = User.objects.get(id=data["assigned_to"])
                ncr.assigned_to = assignee
                new_name = assignee.display_name or assignee.email
                if old_name != new_name:
                    QMSFieldChange.objects.create(
                        record_type="ncr",
                        record_id=ncr.id,
                        field_name="assigned_to",
                        old_value=old_name,
                        new_value=new_name,
                        changed_by=request.user,
                    )
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
            if old_name:
                QMSFieldChange.objects.create(
                    record_type="ncr",
                    record_id=ncr.id,
                    field_name="assigned_to",
                    old_value=old_name,
                    new_value="",
                    changed_by=request.user,
                )
            ncr.assigned_to = None
    if "approved_by" in data and not new_status:
        old_name = (ncr.approved_by.display_name or ncr.approved_by.email) if ncr.approved_by else ""
        if data["approved_by"]:
            try:
                approver = User.objects.get(id=data["approved_by"])
                new_name = approver.display_name or approver.email
                if old_name != new_name:
                    QMSFieldChange.objects.create(
                        record_type="ncr",
                        record_id=ncr.id,
                        field_name="approved_by",
                        old_value=old_name,
                        new_value=new_name,
                        changed_by=request.user,
                    )
                ncr.approved_by = approver
            except User.DoesNotExist:
                pass
        else:
            if old_name:
                QMSFieldChange.objects.create(
                    record_type="ncr",
                    record_id=ncr.id,
                    field_name="approved_by",
                    old_value=old_name,
                    new_value="",
                    changed_by=request.user,
                )
            ncr.approved_by = None
    # Link existing RCA session
    if "rca_session_id" in data:
        old_rca = str(ncr.rca_session_id) if ncr.rca_session_id else ""
        if data["rca_session_id"]:
            try:
                rca = qms_queryset(RCASession, request.user)[0].get(id=data["rca_session_id"])
                ncr.rca_session = rca
                new_rca = str(rca.id)
                if old_rca != new_rca:
                    QMSFieldChange.objects.create(
                        record_type="ncr",
                        record_id=ncr.id,
                        field_name="rca_session",
                        old_value=old_rca,
                        new_value=new_rca,
                        changed_by=request.user,
                    )
            except RCASession.DoesNotExist:
                pass
        else:
            ncr.rca_session = None
    # Supplier link
    if "supplier_id" in data:
        old_supplier = ncr.supplier.name if ncr.supplier_id and ncr.supplier else ""
        if data["supplier_id"]:
            try:
                sup = SupplierRecord.objects.get(id=data["supplier_id"])
                if old_supplier != sup.name:
                    QMSFieldChange.objects.create(
                        record_type="ncr",
                        record_id=ncr.id,
                        field_name="supplier",
                        old_value=old_supplier,
                        new_value=sup.name,
                        changed_by=request.user,
                    )
                ncr.supplier = sup
            except SupplierRecord.DoesNotExist:
                pass
        else:
            if old_supplier:
                QMSFieldChange.objects.create(
                    record_type="ncr",
                    record_id=ncr.id,
                    field_name="supplier",
                    old_value=old_supplier,
                    new_value="",
                    changed_by=request.user,
                )
            ncr.supplier = None
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

    # D2: Auto-create closure signature on NCR close
    if new_status == "closed":
        _create_workflow_signature(request, "ncr", ncr.id, "approved", record=ncr)

    return JsonResponse(ncr.to_dict())


@require_team
@require_http_methods(["GET"])
def ncr_stats(request):
    """NCR statistics for dashboard."""
    user = request.user
    ncrs = qms_queryset(NonconformanceRecord, user)[0]
    open_ncrs = ncrs.exclude(status="closed")
    overdue = open_ncrs.filter(capa_due_date__lt=date.today()).count()

    # Average close time
    from django.db.models import Avg, F

    avg_close = ncrs.filter(closed_at__isnull=False).aggregate(avg_days=Avg(F("closed_at") - F("created_at")))[
        "avg_days"
    ]
    avg_close_days = avg_close.days if avg_close else None

    from django.db.models import Count

    by_severity = dict(ncrs.values_list("severity").annotate(c=Count("id")).values_list("severity", "c"))
    by_source = dict(ncrs.values_list("source").annotate(c=Count("id")).values_list("source", "c"))
    by_status = dict(ncrs.values_list("status").annotate(c=Count("id")).values_list("status", "c"))

    return JsonResponse(
        {
            "total": ncrs.count(),
            "open": open_ncrs.count(),
            "overdue_capas": overdue,
            "avg_close_days": avg_close_days,
            "by_severity": by_severity,
            "by_source": by_source,
            "by_status": by_status,
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
    ncrs = qms_queryset(NonconformanceRecord, user)[0]

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
    qs, tenant, _is_admin = qms_queryset(NonconformanceRecord, request.user)
    try:
        ncr = qs.get(id=ncr_id)
    except NonconformanceRecord.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)
    if not qms_can_edit(request.user, ncr, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

    # Ensure NCR has a project before linking
    _ensure_ncr_project(ncr, request.user)

    session = RCASession(
        title=f"RCA for {ncr.title}",
        event=ncr.description or ncr.title,
        project=ncr.project,  # Land in same Study as NCR
        status="draft",
    )
    qms_set_ownership(session, request.user, ncr.site)
    session.save()
    ncr.rca_session = session
    ncr.save(update_fields=["rca_session"])
    if ncr.project:
        ncr.project.log_event("rca_launched", f"RCA started for NCR: {ncr.title}", user=request.user)
    return JsonResponse({"rca_session_id": str(session.id)}, status=201)


@require_team
@require_http_methods(["POST", "DELETE"])
def ncr_files(request, ncr_id):
    """Attach or detach files from an NCR."""
    qs, tenant, _is_admin = qms_queryset(NonconformanceRecord, request.user)
    try:
        ncr = qs.get(id=ncr_id)
    except NonconformanceRecord.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)
    if not qms_can_edit(request.user, ncr, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

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
        audits = qms_queryset(InternalAudit, user)[0]
        status = request.GET.get("status")
        if status:
            audits = audits.filter(status=status)
        return JsonResponse([a.to_dict() for a in audits[:50]], safe=False)

    data = json.loads(request.body)
    site = None
    if data.get("site_id"):
        try:
            site = Site.objects.get(id=data["site_id"])
        except Site.DoesNotExist:
            return JsonResponse({"error": "Site not found"}, status=404)
    audit = InternalAudit(
        title=data.get("title", ""),
        scheduled_date=data.get("scheduled_date", date.today()),
        lead_auditor=data.get("lead_auditor", ""),
        iso_clauses=data.get("iso_clauses", []),
        departments=data.get("departments", []),
        scope=data.get("scope", ""),
    )
    qms_set_ownership(audit, user, site)
    audit.save()
    return JsonResponse(audit.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def audit_detail(request, audit_id):
    """Get, update, or delete an audit."""
    qs, tenant, _is_admin = qms_queryset(InternalAudit, request.user)
    try:
        audit = qs.get(id=audit_id)
    except InternalAudit.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(audit.to_dict())

    if not qms_can_edit(request.user, audit, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

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
    qs, tenant, _is_admin = qms_queryset(InternalAudit, request.user)
    try:
        audit = qs.get(id=audit_id)
    except InternalAudit.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)
    if not qms_can_edit(request.user, audit, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

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
        ncr = NonconformanceRecord(
            raised_by=request.user,
            title=f"Audit Finding — {audit.title}{clause_label} — {finding.description[:100]}",
            description=finding.description,
            severity=severity_map[finding_type],
            source="internal_audit",
            iso_clause=clause,
        )
        qms_set_ownership(ncr, request.user, audit.site)
        ncr.save()
        _ensure_ncr_project(ncr, request.user)
        finding.ncr = ncr
        finding.save(update_fields=["ncr"])
        ncr_id = str(ncr.id)

    result = finding.to_dict()
    if ncr_id:
        result["ncr_id"] = ncr_id
    return JsonResponse(result, status=201)


@require_team
@require_http_methods(["PUT", "DELETE"])
def audit_finding_detail(request, audit_id, finding_id):
    """Update or delete an audit finding."""
    qs, tenant, _is_admin = qms_queryset(InternalAudit, request.user)
    try:
        audit = qs.get(id=audit_id)
    except InternalAudit.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)
    if not qms_can_edit(request.user, audit, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

    try:
        finding = audit.findings.get(id=finding_id)
    except AuditFinding.DoesNotExist:
        return JsonResponse({"error": "Finding not found"}, status=404)

    if request.method == "DELETE":
        finding.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)
    for field in ["finding_type", "description", "iso_clause", "evidence", "corrective_action", "status"]:
        if field in data:
            setattr(finding, field, data[field])
    if "due_date" in data:
        finding.due_date = data["due_date"] or None
    if data.get("status") == "closed":
        finding.is_resolved = True
    finding.save()
    return JsonResponse(finding.to_dict())


@require_team
@require_http_methods(["POST"])
def audit_apply_checklist(request, audit_id):
    """Apply a checklist template to an audit — creates findings from check items.

    POST body: {"checklist_id": "uuid"}
    Returns the checklist items with pass/fail tracking stored in the audit's
    checklist_results field (JSONField on InternalAudit).

    Or: POST body: {"checklist_id": "uuid", "results": [{"question": "...", "result": "pass|fail|na", "notes": "..."}]}
    to save execution results.
    """
    qs, tenant, _is_admin = qms_queryset(InternalAudit, request.user)
    try:
        audit = qs.get(id=audit_id)
    except InternalAudit.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)
    if not qms_can_edit(request.user, audit, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

    data = json.loads(request.body)
    checklist_id = data.get("checklist_id")
    if not checklist_id:
        return JsonResponse({"error": "checklist_id required"}, status=400)

    try:
        checklist = AuditChecklist.objects.get(id=checklist_id, owner=request.user)
    except AuditChecklist.DoesNotExist:
        return JsonResponse({"error": "Checklist not found"}, status=404)

    results = data.get("results")
    if results is not None:
        # Save execution results
        existing = audit.checklist_results or {}
        existing[str(checklist_id)] = {
            "checklist_name": checklist.name,
            "iso_clause": checklist.iso_clause,
            "items": results,
            "completed_at": timezone.now().isoformat(),
        }
        audit.checklist_results = existing
        audit.save(update_fields=["checklist_results"])
        return JsonResponse({"ok": True, "checklist_name": checklist.name})

    # Return checklist items for execution (with any existing results)
    existing = (audit.checklist_results or {}).get(str(checklist_id), {})
    items = checklist.check_items or []
    existing_items = existing.get("items", [])

    # Merge: overlay existing results onto template
    merged = []
    for i, item in enumerate(items):
        entry = {"question": item.get("question", ""), "guidance": item.get("guidance", "")}
        if i < len(existing_items):
            entry["result"] = existing_items[i].get("result", "")
            entry["notes"] = existing_items[i].get("notes", "")
        else:
            entry["result"] = ""
            entry["notes"] = ""
        merged.append(entry)

    return JsonResponse(
        {
            "checklist_id": str(checklist.id),
            "checklist_name": checklist.name,
            "iso_clause": checklist.iso_clause,
            "items": merged,
            "completed_at": existing.get("completed_at"),
        }
    )


@require_team
@require_http_methods(["GET"])
def audit_clause_coverage(request):
    """Clause coverage tracking — which ISO clauses have been audited and when.

    Returns a map of clause → {count, last_audit_date, finding_count, open_findings}.
    """
    user = request.user
    qs = qms_queryset(InternalAudit, user)[0]
    audits = qs.exclude(status="cancelled")

    # All ISO 9001 clauses
    all_clauses = ["4", "5", "6", "7", "8", "9", "10"]
    coverage = {}

    for audit in audits:
        clauses = audit.iso_clauses or []
        for clause in clauses:
            key = clause.split(".")[0] if "." in clause else clause
            if key not in coverage:
                coverage[key] = {
                    "clause": key,
                    "audit_count": 0,
                    "last_audit_date": None,
                    "finding_count": 0,
                    "open_findings": 0,
                    "audits": [],
                }
            coverage[key]["audit_count"] += 1
            audit_date = str(audit.scheduled_date) if audit.scheduled_date else str(audit.created_at.date())
            if not coverage[key]["last_audit_date"] or audit_date > coverage[key]["last_audit_date"]:
                coverage[key]["last_audit_date"] = audit_date

            # Also track at the specific sub-clause level
            if clause not in coverage:
                coverage[clause] = {
                    "clause": clause,
                    "audit_count": 0,
                    "last_audit_date": None,
                    "finding_count": 0,
                    "open_findings": 0,
                    "audits": [],
                }
            coverage[clause]["audit_count"] += 1
            if not coverage[clause]["last_audit_date"] or audit_date > coverage[clause]["last_audit_date"]:
                coverage[clause]["last_audit_date"] = audit_date

    # Count findings per clause
    for audit in audits:
        for finding in audit.findings.all():
            clause = finding.iso_clause
            if clause and clause in coverage:
                coverage[clause]["finding_count"] += 1
                if finding.status != "closed":
                    coverage[clause]["open_findings"] += 1

    # Fill in missing top-level clauses
    for clause in all_clauses:
        if clause not in coverage:
            coverage[clause] = {
                "clause": clause,
                "audit_count": 0,
                "last_audit_date": None,
                "finding_count": 0,
                "open_findings": 0,
            }

    return JsonResponse({"coverage": sorted(coverage.values(), key=lambda c: c["clause"])})


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
        reqs = qms_queryset(TrainingRequirement, user)[0].prefetch_related(
            "records", "records__changes", "records__changes__changed_by"
        )
        return JsonResponse([r.to_dict() for r in reqs], safe=False)

    data = json.loads(request.body)
    site = None
    if data.get("site_id"):
        try:
            site = Site.objects.get(id=data["site_id"])
        except Site.DoesNotExist:
            return JsonResponse({"error": "Site not found"}, status=404)
    req = TrainingRequirement(
        name=data.get("name", ""),
        description=data.get("description", ""),
        iso_clause=data.get("iso_clause", ""),
        frequency_months=data.get("frequency_months", 0),
        is_mandatory=data.get("is_mandatory", False),
    )
    qms_set_ownership(req, user, site)
    req.save()
    # Link to controlled document if provided
    doc_id = data.get("document_id")
    if doc_id:
        try:
            doc = qms_queryset(ControlledDocument, user)[0].get(id=doc_id)
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
    qs, tenant, _is_admin = qms_queryset(TrainingRequirement, request.user)
    try:
        req = qs.get(id=req_id)
    except TrainingRequirement.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(req.to_dict())

    if not qms_can_edit(request.user, req, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

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
                doc = qms_queryset(ControlledDocument, request.user)[0].get(id=doc_id)
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
        req_qs = qms_queryset(TrainingRequirement, request.user)[0]
        record = TrainingRecord.objects.get(
            id=record_id,
            requirement__in=req_qs,
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
        req = qms_queryset(TrainingRequirement, request.user)[0].get(id=req_id)
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
        req_qs = qms_queryset(TrainingRequirement, request.user)[0]
        record = TrainingRecord.objects.get(
            id=record_id,
            requirement__in=req_qs,
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
    ncrs = qms_queryset(NonconformanceRecord, user)[0]
    audits = qms_queryset(InternalAudit, user)[0]
    req_qs = qms_queryset(TrainingRequirement, user)[0]
    records = TrainingRecord.objects.filter(requirement__in=req_qs)
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
        audit__in=audits,
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

    # A3: ISO 9001 §9.3 — completion requires attendees and outputs
    if review.status == "complete":
        if not review.attendees:
            return JsonResponse(
                {"error": "Attendees are required to complete a management review (ISO 9001 §9.3)"}, status=400
            )
        if not review.outputs:
            return JsonResponse(
                {"error": "Outputs are required to complete a management review (ISO 9001 §9.3.3)"}, status=400
            )
        # D4: Auto-create completion signature on management review
        _create_workflow_signature(request, "review", review.id, "approved", record=review)

    review.save()
    return JsonResponse(review.to_dict())


# =========================================================================
# E8: Management Review Auto-Narrative
# =========================================================================


@require_team
@require_http_methods(["POST"])
def review_narrative(request, review_id):
    """E8: Generate an executive summary narrative from management review data.

    Uses the auto-captured snapshot + any manual inputs to generate a
    structured executive summary per ISO 9001 §9.3.3 outputs format.
    Calls Claude for interpretation — rate-limited like the AI guide.
    """
    try:
        review = ManagementReview.objects.get(id=review_id, owner=request.user)
    except ManagementReview.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    snapshot = review.data_snapshot or {}
    inputs = review.inputs or {}
    attendees = review.attendees or []

    # Build context for Claude
    context_parts = [
        f"Management Review: {review.title}",
        f"Date: {review.meeting_date}",
        f"Attendees: {', '.join(attendees) if attendees else 'Not recorded'}",
    ]

    # Snapshot data
    ncr = snapshot.get("ncr_summary", {})
    if ncr:
        context_parts.append(
            f"\nNCR Status: {ncr.get('total', 0)} total, {ncr.get('open', 0)} open, "
            f"{ncr.get('closed', 0)} closed. "
            f"By severity: {ncr.get('by_severity', {}).get('critical', 0)} critical, "
            f"{ncr.get('by_severity', {}).get('major', 0)} major, "
            f"{ncr.get('by_severity', {}).get('minor', 0)} minor."
        )

    audit = snapshot.get("audit_summary", {})
    if audit:
        context_parts.append(
            f"Audits: {audit.get('completed', 0)} completed, {audit.get('open_findings', 0)} open findings."
        )

    training = snapshot.get("training_summary", {})
    if training:
        context_parts.append(
            f"Training: {training.get('compliance_rate', 'N/A')}% compliance, {training.get('gap_count', 0)} gaps."
        )

    prior = snapshot.get("prior_actions", {})
    if prior:
        context_parts.append(f"Prior review actions: {json.dumps(prior, indent=2)[:500]}")

    # Manual inputs
    for key, val in inputs.items():
        if isinstance(val, dict) and val.get("content"):
            context_parts.append(f"{val.get('title', key)}: {val['content'][:300]}")
        elif isinstance(val, str) and val.strip():
            context_parts.append(f"{key}: {val[:300]}")

    # Also include audit readiness if available
    try:
        from django.test import RequestFactory

        factory = RequestFactory()
        readiness_req = factory.get("/api/iso/audit-readiness/")
        readiness_req.user = request.user
        readiness_resp = audit_readiness(readiness_req)
        readiness_data = json.loads(readiness_resp.content)
        context_parts.append(
            f"\nAudit Readiness Score: {readiness_data['overall_score']}/100 ({readiness_data['overall_rag']})"
        )
        for f in readiness_data.get("top_findings", [])[:3]:
            context_parts.append(f"  Gap: §{f['clause']} {f['name']}: {f['detail']}")
    except Exception:
        pass

    context = "\n".join(context_parts)

    # Call Claude
    from svend_config.config import get_anthropic_client

    try:
        client = get_anthropic_client()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a quality management consultant generating an executive summary for an "
                        "ISO 9001:2015 management review (clause 9.3.3). Based on the following QMS data, "
                        "write a concise executive summary with these sections:\n\n"
                        "1. **Overall QMS Health** — one paragraph assessment\n"
                        "2. **Key Findings** — 3-5 bullet points on what the data shows\n"
                        "3. **Actions Required** — specific, actionable items for management\n"
                        "4. **Resource Needs** — any resource implications\n"
                        "5. **Opportunities for Improvement** — forward-looking recommendations\n\n"
                        "Be specific and reference the numbers. Do not use generic quality platitudes. "
                        "Write for a plant manager or quality director who has 5 minutes.\n\n"
                        f"QMS DATA:\n{context}"
                    ),
                }
            ],
        )
        narrative = message.content[0].text
    except Exception as e:
        logger.exception("Review narrative generation failed")
        return JsonResponse({"error": f"Narrative generation failed: {e}"}, status=500)

    return JsonResponse(
        {
            "narrative": narrative,
            "review_id": str(review.id),
            "data_sources": list(snapshot.keys()),
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        }
    )


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
        docs = qms_queryset(ControlledDocument, user)[0]
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
    site = None
    if data.get("site_id"):
        try:
            site = Site.objects.get(id=data["site_id"])
        except Site.DoesNotExist:
            return JsonResponse({"error": "Site not found"}, status=404)
    doc = ControlledDocument(
        title=data.get("title", ""),
        document_number=data.get("document_number", ""),
        category=data.get("category", ""),
        iso_clause=data.get("iso_clause", ""),
        content=data.get("content", ""),
        review_due_date=data.get("review_due_date") or None,
        retention_years=data.get("retention_years", 7),
    )
    qms_set_ownership(doc, user, site)
    doc.save()
    return JsonResponse(doc.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def document_detail(request, doc_id):
    """Get, update, or delete a controlled document."""
    qs, tenant, _is_admin = qms_queryset(ControlledDocument, request.user)
    try:
        doc = qs.get(id=doc_id)
    except ControlledDocument.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(doc.to_dict())

    if not qms_can_edit(request.user, doc, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

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
            # D1: Auto-create approval signature
            _create_workflow_signature(request, "document", doc.id, "approved", record=doc)

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
    qs, tenant, _is_admin = qms_queryset(ControlledDocument, request.user)
    try:
        doc = qs.get(id=doc_id)
    except ControlledDocument.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)
    if not qms_can_edit(request.user, doc, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

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
        suppliers = qms_queryset(SupplierRecord, user)[0]
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
    supplier = SupplierRecord(
        name=data.get("name", ""),
        supplier_type=data.get("supplier_type", "other"),
        contact_name=data.get("contact_name", ""),
        contact_email=data.get("contact_email", ""),
        contact_phone=data.get("contact_phone", ""),
        products_services=data.get("products_services", ""),
        notes=data.get("notes", ""),
    )
    qms_set_ownership(supplier, user)
    supplier.save()
    return JsonResponse(supplier.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def supplier_detail(request, supplier_id):
    """Get, update, or delete a supplier."""
    qs, tenant, _is_admin = qms_queryset(SupplierRecord, request.user)
    try:
        supplier = qs.get(id=supplier_id)
    except SupplierRecord.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(supplier.to_dict())

    if not qms_can_edit(request.user, supplier, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

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

    capa = CAPAReport(
        project=project,
        title=title,
        description=context["problem"],
        root_cause=context["root_cause"] or "",
        priority=data.get("priority", "medium"),
        source_type=data.get("source_type", ""),
        source_id=data.get("source_id") or None,
    )
    qms_set_ownership(capa, request.user)
    capa.save()

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
def study_raise_ncr(request):
    """B1: Create an NCR from SPC out-of-control signal or Study findings.

    Extracts key variables from SPC output (chart_type, OOC points, rule
    violations, control limits) following the notebook variable extraction
    pattern. Pre-populates NCR description with structured SPC context.

    Request body:
    {
        "project_id": "uuid",
        "title": "optional override",
        "spc_data": {                        # Optional — from SPC chart result
            "chart_type": "I-MR",
            "out_of_control": [{"index": 5, "value": 16.2, "reason": "Above UCL"}],
            "run_violations": [{"rule": "Rule 2", "indices": [...], "description": "..."}],
            "limits": {"ucl": 15.42, "cl": 10.0, "lcl": 4.58},
            "in_control": false,
            "statistics": {"cpk": 0.85, "ppk": 0.78}
        },
        "severity": "minor|major|critical",
        "site_id": "uuid"
    }
    """
    project, data, err = _get_study_for_action(request)
    if err:
        return err

    spc = data.get("spc_data", {})

    # Extract key variables from SPC output (notebook-style hierarchical extraction)
    chart_type = spc.get("chart_type", "")
    ooc = spc.get("out_of_control", [])
    run_viols = spc.get("run_violations", [])
    limits = spc.get("limits", {})
    stats = spc.get("statistics", {})

    # Build structured description from SPC context
    desc_parts = [f"SPC signal detected in Study: {project.title}"]
    if chart_type:
        desc_parts.append(f"\n**Chart type:** {chart_type}")
    if limits:
        desc_parts.append(
            f"**Control limits:** UCL={limits.get('ucl', '—')}, "
            f"CL={limits.get('cl', '—')}, LCL={limits.get('lcl', '—')}"
        )
    if ooc:
        desc_parts.append(f"\n**Out-of-control points ({len(ooc)}):**")
        for pt in ooc[:10]:
            desc_parts.append(f"  • Index {pt.get('index')}: value={pt.get('value')} ({pt.get('reason')})")
    if run_viols:
        desc_parts.append(f"\n**Run rule violations ({len(run_viols)}):**")
        for rv in run_viols[:5]:
            desc_parts.append(f"  • {rv.get('rule', '')}: {rv.get('description', '')}")
    # Key variables (following notebook _extract_stats pattern)
    key_vars = []
    for k in ["cpk", "ppk", "cp", "pp", "sigma_level", "yield_pct"]:
        v = stats.get(k) or spc.get(k)
        if v is not None:
            key_vars.append(f"{k}={v}")
    if key_vars:
        desc_parts.append(f"\n**Key metrics:** {', '.join(key_vars)}")

    description = "\n".join(desc_parts)

    # Determine severity from data if not explicit
    severity = data.get("severity", "minor")
    if not data.get("severity"):
        cpk = stats.get("cpk") or spc.get("cpk")
        if cpk is not None and cpk < 1.0:
            severity = "major"
        if len(ooc) >= 3 or (cpk is not None and cpk < 0.67):
            severity = "critical"

    title = data.get("title", f"SPC Signal — {chart_type or 'Process'} — {project.title}"[:300])

    ncr = NonconformanceRecord(
        title=title,
        description=description,
        severity=severity,
        source="process",
        project=project,
    )
    qms_set_ownership(ncr, request.user)
    if data.get("site_id"):
        try:
            ncr.site = Site.objects.get(id=data["site_id"])
        except Site.DoesNotExist:
            pass
    ncr.save()

    StudyAction.objects.create(
        project=project,
        action_type="raise_ncr",
        target_type="ncr",
        target_id=ncr.id,
        notes=f"NCR raised from SPC signal: {chart_type or 'process'}, {len(ooc)} OOC points",
        created_by=request.user,
    )

    project.log_event("ncr_raised", f"NCR created from SPC: {ncr.title}", user=request.user)
    logger.info("Study %s: raised NCR %s from SPC", project.id, ncr.id)
    return JsonResponse(
        {
            "ok": True,
            "action": "raise_ncr",
            "ncr": ncr.to_dict(),
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

    audit = InternalAudit(
        title=data.get("title", f"Verification Audit — {project.title}"),
        scheduled_date=scheduled_date,
        scope=scope,
        lead_auditor=data.get("lead_auditor", ""),
    )
    qms_set_ownership(audit, request.user)
    audit.save()

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

    doc = ControlledDocument(
        title=title,
        category=data.get("category", "Change Request"),
        content=justification,
        document_number=data.get("document_number", ""),
        iso_clause=data.get("iso_clause", ""),
        source_study=project,
    )
    qms_set_ownership(doc, request.user)
    doc.save()

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

    req = TrainingRequirement(
        name=name,
        description=description,
        iso_clause=data.get("iso_clause", ""),
        frequency_months=data.get("frequency_months", 0),
        is_mandatory=data.get("is_mandatory", True),
    )
    qms_set_ownership(req, request.user)
    req.save()

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
        fmea = qms_queryset(FMEA, request.user)[0].get(id=fmea_id)
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


def _create_workflow_signature(request, document_type, document_id, meaning, record=None):
    """D-helper: Create an electronic signature from a QMS workflow transition.

    Called automatically when key transitions occur (document approval, NCR/CAPA
    closure, management review completion). Captures document snapshot, user agent,
    and IP for audit trail.

    If `password` is in the request body and valid, creates a fully re-authenticated
    signature per 21 CFR Part 11 §11.100. Otherwise creates a workflow-triggered
    signature (sufficient for ISO 9001, not for FDA-regulated).

    Returns the signature dict or None if creation fails.
    """
    user = request.user
    tenant_id = _resolve_tenant_id(user)

    # Capture snapshot
    snapshot = {}
    if record and hasattr(record, "to_dict"):
        snapshot = record.to_dict()

    # Check optional re-auth
    password = None
    try:
        data = json.loads(request.body) if request.body else {}
        password = data.get("password")
    except (json.JSONDecodeError, AttributeError):
        pass

    re_authed = False
    if password and user.check_password(password):
        re_authed = True

    try:
        sig = ElectronicSignature(
            signer=user,
            document_type=document_type,
            document_id=document_id,
            meaning=meaning,
            user_agent=request.META.get("HTTP_USER_AGENT", ""),
            # SynaraImmutableLog fields
            event_name=f"workflow_{meaning}",
            actor=user.email or str(user.id),
            tenant_id=tenant_id,
            before_snapshot=snapshot,
            after_snapshot={},
            changes={"workflow_triggered": True, "re_authenticated": re_authed},
            reason=f"Workflow transition: {document_type} {meaning}",
        )
        sig.save()
        logger.info(
            "Workflow signature: %s %s %s/%s (re_auth=%s)",
            user.email,
            meaning,
            document_type,
            document_id,
            re_authed,
        )
        return sig.to_dict()
    except Exception:
        logger.exception("Failed to create workflow signature for %s/%s", document_type, document_id)
        return None


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


# =========================================================================
# C1: Customer Complaints — ISO 9001 §9.1.2
# =========================================================================


@require_team
@require_http_methods(["GET", "POST"])
def complaint_list_create(request):
    """List or create customer complaints."""
    user = request.user
    qs, tenant, is_admin = qms_queryset(CustomerComplaint, user)

    if request.method == "GET":
        complaints = qs.select_related("assigned_to")
        status = request.GET.get("status")
        if status:
            complaints = complaints.filter(status=status)
        severity = request.GET.get("severity")
        if severity:
            complaints = complaints.filter(severity=severity)
        return JsonResponse([c.to_dict() for c in complaints[:50]], safe=False)

    data = json.loads(request.body)
    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "Title is required"}, status=400)

    complaint = CustomerComplaint(
        title=title,
        description=data.get("description", ""),
        source=data.get("source", "other"),
        severity=data.get("severity", "medium"),
        product_service=data.get("product_service", ""),
        customer_name=data.get("customer_name", ""),
        customer_contact=data.get("customer_contact", ""),
        date_received=data.get("date_received"),
    )
    qms_set_ownership(complaint, user)
    if data.get("site_id"):
        try:
            complaint.site = Site.objects.get(id=data["site_id"])
        except Site.DoesNotExist:
            pass
    complaint.save()
    return JsonResponse(complaint.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def complaint_detail(request, complaint_id):
    """Get, update, or delete a customer complaint."""
    qs, tenant, _is_admin = qms_queryset(CustomerComplaint, request.user)
    try:
        complaint = qs.get(id=complaint_id)
    except CustomerComplaint.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(complaint.to_dict())

    if not qms_can_edit(request.user, complaint, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

    if request.method == "DELETE":
        complaint.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)

    # Status transitions with workflow enforcement
    new_status = data.get("status")
    if new_status and new_status != complaint.status:
        # Pre-set fields that transitions require
        for field in ["assigned_to", "resolution", "satisfaction_followup"]:
            if field in data:
                if field == "assigned_to" and data[field]:
                    try:
                        setattr(complaint, field, User.objects.get(id=data[field]))
                    except User.DoesNotExist:
                        pass
                else:
                    setattr(complaint, field, data[field])

        ok, msg = complaint.can_transition(new_status)
        if not ok:
            return JsonResponse({"error": msg}, status=400)

        if new_status == "acknowledged" and not complaint.date_acknowledged:
            complaint.date_acknowledged = timezone.now().date()
        complaint.status = new_status

    # Update fields
    for field in [
        "title",
        "description",
        "source",
        "severity",
        "product_service",
        "customer_name",
        "customer_contact",
        "root_cause",
        "resolution",
        "preventive_action",
        "satisfaction_followup",
    ]:
        if field in data and field != "status":
            setattr(complaint, field, data[field])
    if "customer_satisfied" in data:
        complaint.customer_satisfied = data["customer_satisfied"]
    if "date_received" in data:
        complaint.date_received = data["date_received"]

    complaint.save()
    return JsonResponse(complaint.to_dict())


# =========================================================================
# C2: Risk Register — ISO 9001 §6.1
# =========================================================================


@require_team
@require_http_methods(["GET", "POST"])
def risk_list_create(request):
    """List or create risks/opportunities."""
    user = request.user
    qs, tenant, is_admin = qms_queryset(Risk, user)

    if request.method == "GET":
        risks = qs
        status = request.GET.get("status")
        if status:
            risks = risks.filter(status=status)
        category = request.GET.get("category")
        if category:
            risks = risks.filter(category=category)
        risk_type = request.GET.get("risk_type")
        if risk_type:
            risks = risks.filter(risk_type=risk_type)
        return JsonResponse([r.to_dict() for r in risks[:50]], safe=False)

    data = json.loads(request.body)
    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "Title is required"}, status=400)

    risk = Risk(
        title=title,
        description=data.get("description", ""),
        risk_type=data.get("risk_type", "risk"),
        category=data.get("category", "operational"),
        likelihood=min(5, max(1, int(data.get("likelihood", 1)))),
        impact=min(5, max(1, int(data.get("impact", 1)))),
        review_frequency_months=data.get("review_frequency_months", 3),
    )
    qms_set_ownership(risk, user)
    if data.get("site_id"):
        try:
            risk.site = Site.objects.get(id=data["site_id"])
        except Site.DoesNotExist:
            pass
    if data.get("risk_owner"):
        try:
            risk.risk_owner = User.objects.get(id=data["risk_owner"])
        except User.DoesNotExist:
            pass
    risk.save()  # auto-computes risk_score
    return JsonResponse(risk.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def risk_detail(request, risk_id):
    """Get, update, or delete a risk."""
    qs, tenant, _is_admin = qms_queryset(Risk, request.user)
    try:
        risk = qs.get(id=risk_id)
    except Risk.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(risk.to_dict())

    if not qms_can_edit(request.user, risk, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

    if request.method == "DELETE":
        risk.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)
    for field in [
        "title",
        "description",
        "risk_type",
        "category",
        "status",
        "mitigation_actions",
        "review_frequency_months",
    ]:
        if field in data:
            setattr(risk, field, data[field])
    for int_field in ["likelihood", "impact", "residual_likelihood", "residual_impact"]:
        if int_field in data and data[int_field] is not None:
            setattr(risk, int_field, min(5, max(1, int(data[int_field]))))
    if "review_date" in data:
        risk.review_date = data["review_date"]
    if "risk_owner" in data:
        try:
            risk.risk_owner = User.objects.get(id=data["risk_owner"]) if data["risk_owner"] else None
        except User.DoesNotExist:
            pass

    risk.save()  # auto-computes risk_score and residual_risk_score
    return JsonResponse(risk.to_dict())


# =========================================================================
# C3: Calibration Equipment Register — ISO 9001 §7.1.5
# =========================================================================


@require_team
@require_http_methods(["GET", "POST"])
def equipment_list_create(request):
    """List or create measurement equipment."""
    user = request.user
    qs, tenant, is_admin = qms_queryset(MeasurementEquipment, user)

    if request.method == "GET":
        equipment = qs
        status = request.GET.get("status")
        if status:
            equipment = equipment.filter(status=status)
        eq_type = request.GET.get("type")
        if eq_type:
            equipment = equipment.filter(equipment_type=eq_type)
        return JsonResponse([e.to_dict() for e in equipment[:50]], safe=False)

    data = json.loads(request.body)
    name = data.get("name", "").strip()
    if not name:
        return JsonResponse({"error": "Name is required"}, status=400)

    eq = MeasurementEquipment(
        name=name,
        asset_id=data.get("asset_id", ""),
        serial_number=data.get("serial_number", ""),
        manufacturer=data.get("manufacturer", ""),
        model_number=data.get("model_number", ""),
        equipment_type=data.get("equipment_type", "other"),
        location=data.get("location", ""),
        calibration_interval_months=data.get("calibration_interval_months", 12),
        last_calibration_date=data.get("last_calibration_date"),
        next_calibration_due=data.get("next_calibration_due"),
        calibration_provider=data.get("calibration_provider", ""),
        measurement_range=data.get("measurement_range", ""),
        resolution=data.get("resolution", ""),
        accuracy=data.get("accuracy", ""),
        notes=data.get("notes", ""),
    )
    qms_set_ownership(eq, user)
    if data.get("site_id"):
        try:
            eq.site = Site.objects.get(id=data["site_id"])
        except Site.DoesNotExist:
            pass
    eq.save()
    return JsonResponse(eq.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def equipment_detail(request, equipment_id):
    """Get, update, or delete measurement equipment."""
    qs, tenant, _is_admin = qms_queryset(MeasurementEquipment, request.user)
    try:
        eq = qs.get(id=equipment_id)
    except MeasurementEquipment.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(eq.to_dict())

    if not qms_can_edit(request.user, eq, tenant):
        return JsonResponse({"error": "Permission denied"}, status=403)

    if request.method == "DELETE":
        eq.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)
    for field in [
        "name",
        "asset_id",
        "serial_number",
        "manufacturer",
        "model_number",
        "equipment_type",
        "location",
        "status",
        "calibration_provider",
        "calibration_certificate",
        "measurement_range",
        "resolution",
        "accuracy",
        "notes",
    ]:
        if field in data:
            setattr(eq, field, data[field])
    if "calibration_interval_months" in data:
        eq.calibration_interval_months = data["calibration_interval_months"]
    if "last_calibration_date" in data:
        eq.last_calibration_date = data["last_calibration_date"]
    if "next_calibration_due" in data:
        eq.next_calibration_due = data["next_calibration_due"]
    if "gage_studies" in data:
        eq.gage_studies = data["gage_studies"]

    eq.save()
    return JsonResponse(eq.to_dict())


# =========================================================================
# =========================================================================
# Universal Checklists — prompt-response model
# =========================================================================


@require_team
@require_http_methods(["GET", "POST"])
def checklist_v2_list_create(request):
    """List or create universal checklists."""
    user = request.user

    if request.method == "GET":
        checklists = Checklist.objects.filter(owner=user)
        cat = request.GET.get("category")
        if cat:
            checklists = checklists.filter(category=cat)
        return JsonResponse([c.to_dict() for c in checklists[:100]], safe=False)

    data = json.loads(request.body)
    name = data.get("name", "").strip()
    if not name:
        return JsonResponse({"error": "Name is required"}, status=400)

    cl = Checklist(
        owner=user,
        name=name,
        description=data.get("description", ""),
        checklist_type=data.get("checklist_type", "read_do"),
        category=data.get("category", "general"),
        version=data.get("version", "1.0"),
        items=data.get("items", []),
    )
    if data.get("site_id"):
        try:
            cl.site = Site.objects.get(id=data["site_id"])
        except Site.DoesNotExist:
            pass
    cl.save()
    return JsonResponse(cl.to_dict(), status=201)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def checklist_v2_detail(request, cl_id):
    """Get, update, or delete a checklist."""
    try:
        cl = Checklist.objects.get(id=cl_id, owner=request.user)
    except Checklist.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(cl.to_dict())

    if request.method == "DELETE":
        cl.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)
    for field in ["name", "description", "checklist_type", "category", "version", "items"]:
        if field in data:
            setattr(cl, field, data[field])
    cl.save()
    return JsonResponse(cl.to_dict())


@require_team
@require_http_methods(["POST"])
def checklist_execute(request):
    """Start or continue a checklist execution against an entity.

    POST body:
    {
        "checklist_id": "uuid",
        "entity_type": "audit|project|kaizen|ncr|capa|equipment|...",
        "entity_id": "uuid",
        "responses": [{"value": "pass", "notes": "...", "file_ids": [...]}]  // optional — save responses
    }
    """
    data = json.loads(request.body)
    checklist_id = data.get("checklist_id")
    entity_type = data.get("entity_type", "")
    entity_id = data.get("entity_id", "")

    if not checklist_id or not entity_type or not entity_id:
        return JsonResponse({"error": "checklist_id, entity_type, and entity_id are required"}, status=400)

    try:
        cl = Checklist.objects.get(id=checklist_id, owner=request.user)
    except Checklist.DoesNotExist:
        return JsonResponse({"error": "Checklist not found"}, status=404)

    # Get or create execution
    execution, created = ChecklistExecution.objects.get_or_create(
        checklist=cl,
        entity_type=entity_type,
        entity_id=entity_id,
        defaults={"executor": request.user, "status": "not_started"},
    )

    responses = data.get("responses")
    if responses is not None:
        # Save responses — auto-compute out_of_spec for numeric items
        items = cl.items or []
        for i, resp in enumerate(responses):
            if i < len(items) and items[i].get("response_type") == "numeric":
                val = resp.get("value")
                if val is not None:
                    try:
                        num_val = float(val)
                        accept_min = items[i].get("accept_min")
                        accept_max = items[i].get("accept_max")
                        oos = False
                        if accept_min is not None and num_val < accept_min:
                            oos = True
                        if accept_max is not None and num_val > accept_max:
                            oos = True
                        resp["out_of_spec"] = oos
                    except (ValueError, TypeError):
                        pass
            # Stamp responder
            if resp.get("value") is not None:
                resp["responded_by"] = request.user.email
                resp["responded_at"] = timezone.now().isoformat()

        execution.responses = responses

        # Update status based on completion
        answered = sum(1 for r in responses if r.get("value") is not None)
        total = len(items)
        if not execution.started_at:
            execution.started_at = timezone.now()
        if answered == 0:
            execution.status = "not_started"
        elif answered >= total:
            execution.status = "complete"
            execution.completed_at = timezone.now()
        else:
            execution.status = "in_progress"

        execution.save()
        return JsonResponse(execution.to_dict())

    # No responses — return current state with checklist items merged
    result = execution.to_dict()
    result["items"] = cl.items
    return JsonResponse(result)


@require_team
@require_http_methods(["GET", "PUT", "DELETE"])
def checklist_execution_detail(request, exec_id):
    """Get, update, or delete a checklist execution."""
    try:
        execution = ChecklistExecution.objects.select_related("checklist").get(id=exec_id, executor=request.user)
    except ChecklistExecution.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "GET":
        result = execution.to_dict()
        result["items"] = execution.checklist.items
        return JsonResponse(result)

    if request.method == "DELETE":
        execution.delete()
        return JsonResponse({"ok": True})

    # PUT — update responses
    data = json.loads(request.body)
    if "responses" in data:
        execution.responses = data["responses"]
    if "status" in data:
        execution.status = data["status"]
        if data["status"] == "complete" and not execution.completed_at:
            execution.completed_at = timezone.now()
    execution.save()
    return JsonResponse(execution.to_dict())


@require_team
@require_http_methods(["GET"])
def checklist_execution_list(request):
    """List executions, optionally filtered by entity_type + entity_id."""
    qs = ChecklistExecution.objects.filter(executor=request.user).select_related("checklist")
    entity_type = request.GET.get("entity_type")
    entity_id = request.GET.get("entity_id")
    if entity_type:
        qs = qs.filter(entity_type=entity_type)
    if entity_id:
        qs = qs.filter(entity_id=entity_id)
    return JsonResponse([e.to_dict() for e in qs[:50]], safe=False)


# E4: Audit Readiness Scoring — ISO 9001 Full Clause Coverage
# =========================================================================

# Scoring thresholds per check — returns (score 0-100, rag, detail)
# green = 80-100, amber = 50-79, red = 0-49


def _score_ncrs(user):
    """§8.7 + §10.2 — NCR management health."""
    qs = qms_queryset(NonconformanceRecord, user)[0]
    total = qs.count()
    if total == 0:
        return 60, "amber", "No NCRs recorded — cannot demonstrate nonconformity management"

    open_ncrs = qs.exclude(status="closed")
    overdue = open_ncrs.filter(capa_due_date__lt=timezone.now().date()).count()
    open_count = open_ncrs.count()

    # Check for NCRs without corrective action
    no_ca = open_ncrs.filter(corrective_action="").exclude(status="open").count()

    findings = []
    score = 100
    if overdue > 0:
        score -= min(40, overdue * 15)
        findings.append(f"{overdue} NCR(s) past CAPA due date")
    if open_count > 5:
        score -= 10
        findings.append(f"{open_count} open NCRs (consider prioritization)")
    if no_ca > 0:
        score -= min(20, no_ca * 10)
        findings.append(f"{no_ca} NCR(s) in progress without corrective action documented")

    rag = "green" if score >= 80 else "amber" if score >= 50 else "red"
    return max(0, score), rag, "; ".join(findings) if findings else "NCR management is current"


def _score_capas(user):
    """§10.2 — CAPA effectiveness."""
    qs = qms_queryset(CAPAReport, user)[0]
    total = qs.count()
    if total == 0:
        # Also check legacy Report-based CAPAs
        from .models import Report

        legacy = Report.objects.filter(owner=user, report_type="capa").count()
        if legacy > 0:
            return 70, "amber", f"{legacy} CAPA(s) in legacy system — consider migrating"
        return 50, "amber", "No CAPAs recorded — required by §10.2"

    open_capas = qs.exclude(status="closed")
    overdue = open_capas.filter(due_date__lt=timezone.now().date()).count()

    findings = []
    score = 100
    if overdue > 0:
        score -= min(40, overdue * 15)
        findings.append(f"{overdue} CAPA(s) past due date")
    if open_capas.count() > 3:
        score -= 10
        findings.append(f"{open_capas.count()} open CAPAs")

    rag = "green" if score >= 80 else "amber" if score >= 50 else "red"
    return max(0, score), rag, "; ".join(findings) if findings else "CAPA management is current"


def _score_training(user):
    """§7.2 — Competence and training."""
    reqs = qms_queryset(TrainingRequirement, user)[0]
    if reqs.count() == 0:
        return 40, "red", "No training requirements defined — §7.2 requires documented competence"

    records = TrainingRecord.objects.filter(requirement__in=reqs)
    total = records.count()
    if total == 0:
        return 30, "red", "Training requirements exist but no records — no evidence of competence"

    complete = records.filter(status="complete").count()
    expired = records.filter(status="expired").count()
    rate = round(complete / total * 100) if total else 0

    findings = []
    score = rate  # Compliance rate IS the score
    if expired > 0:
        score -= min(20, expired * 5)
        findings.append(f"{expired} expired training record(s)")
    if rate < 80:
        findings.append(f"Training compliance at {rate}% — target ≥90%")

    rag = "green" if score >= 80 else "amber" if score >= 50 else "red"
    return max(0, score), rag, "; ".join(findings) if findings else f"Training compliance: {rate}%"


def _score_documents(user):
    """§7.5 — Documented information control."""
    qs = qms_queryset(ControlledDocument, user)[0]
    total = qs.count()
    if total == 0:
        return 30, "red", "No controlled documents — §7.5 requires documented QMS"

    approved = qs.filter(status="approved").count()
    today = timezone.now().date()
    review_overdue = qs.filter(review_due_date__isnull=False, review_due_date__lt=today, status="approved").count()

    findings = []
    score = 100
    if review_overdue > 0:
        score -= min(30, review_overdue * 10)
        findings.append(f"{review_overdue} document(s) past review due date")
    draft_ratio = (total - approved) / total if total else 0
    if draft_ratio > 0.3:
        score -= 15
        findings.append(f"{total - approved} of {total} documents not in approved status")

    rag = "green" if score >= 80 else "amber" if score >= 50 else "red"
    return max(0, score), rag, "; ".join(findings) if findings else f"{approved}/{total} documents approved"


def _score_audits(user):
    """§9.2 — Internal audit program."""
    qs = qms_queryset(InternalAudit, user)[0]
    total = qs.count()
    if total == 0:
        return 20, "red", "No internal audits scheduled — §9.2 requires an audit program"

    complete = qs.filter(status__in=["complete", "report_issued"]).count()
    findings_qs = AuditFinding.objects.filter(audit__in=qs)
    open_major = findings_qs.filter(finding_type="nc_major", is_resolved=False).count()
    open_findings_no_ncr = findings_qs.filter(finding_type="nc_major", ncr__isnull=True, is_resolved=False).count()

    findings = []
    score = 100
    if complete == 0:
        score -= 30
        findings.append("No completed audits")
    if open_major > 0:
        score -= min(30, open_major * 15)
        findings.append(f"{open_major} unresolved major finding(s)")
    if open_findings_no_ncr > 0:
        score -= 10
        findings.append(f"{open_findings_no_ncr} major finding(s) without linked NCR")

    rag = "green" if score >= 80 else "amber" if score >= 50 else "red"
    return max(0, score), rag, "; ".join(findings) if findings else f"{complete}/{total} audits complete"


def _score_management_review(user):
    """§9.3 — Management review."""
    qs = ManagementReview.objects.filter(owner=user)
    complete = qs.filter(status="complete")
    if complete.count() == 0:
        if qs.count() > 0:
            return 50, "amber", "Management review scheduled but not yet completed"
        return 20, "red", "No management reviews — §9.3 requires periodic review"

    latest = complete.order_by("-meeting_date").first()
    days_ago = (timezone.now().date() - latest.meeting_date).days

    findings = []
    score = 100
    if days_ago > 365:
        score -= 40
        findings.append(f"Last review was {days_ago} days ago — annual minimum required")
    elif days_ago > 180:
        score -= 15
        findings.append(f"Last review was {days_ago} days ago")

    # Check outputs
    if not latest.outputs:
        score -= 20
        findings.append("Most recent review has no documented outputs")

    rag = "green" if score >= 80 else "amber" if score >= 50 else "red"
    return max(0, score), rag, "; ".join(findings) if findings else f"Last review: {days_ago} days ago"


def _score_suppliers(user):
    """§8.4 — External provider control."""
    qs = qms_queryset(SupplierRecord, user)[0]
    total = qs.count()
    if total == 0:
        return 60, "amber", "No suppliers registered — acceptable if no external providers"

    today = timezone.now().date()
    eval_overdue = qs.filter(
        next_evaluation_date__isnull=False,
        next_evaluation_date__lt=today,
        status__in=["approved", "preferred", "conditional"],
    ).count()
    suspended = qs.filter(status="suspended").count()

    findings = []
    score = 100
    if eval_overdue > 0:
        score -= min(30, eval_overdue * 10)
        findings.append(f"{eval_overdue} supplier(s) past evaluation due date")
    if suspended > 0:
        score -= 10
        findings.append(f"{suspended} supplier(s) in suspended status")

    rag = "green" if score >= 80 else "amber" if score >= 50 else "red"
    return max(0, score), rag, "; ".join(findings) if findings else f"{total} suppliers managed"


def _score_complaints(user):
    """§9.1.2 — Customer satisfaction monitoring."""
    qs = qms_queryset(CustomerComplaint, user)[0]
    total = qs.count()
    if total == 0:
        return 60, "amber", "No complaints recorded — acceptable if process captures feedback elsewhere"

    open_complaints = qs.exclude(status="closed").count()
    unresolved_old = qs.filter(
        status__in=["open", "acknowledged"],
        created_at__lt=timezone.now() - timedelta(days=30),
    ).count()

    findings = []
    score = 100
    if unresolved_old > 0:
        score -= min(30, unresolved_old * 15)
        findings.append(f"{unresolved_old} complaint(s) open >30 days")
    if open_complaints > 5:
        score -= 10
        findings.append(f"{open_complaints} open complaints")

    rag = "green" if score >= 80 else "amber" if score >= 50 else "red"
    return max(0, score), rag, "; ".join(findings) if findings else f"{total} complaints tracked"


def _score_risks(user):
    """§6.1 — Actions to address risks and opportunities."""
    qs = qms_queryset(Risk, user)[0]
    total = qs.count()
    if total == 0:
        return 30, "red", "No risks identified — §6.1 requires risk-based thinking"

    active = qs.exclude(status="closed")
    high = active.filter(risk_score__gte=12).count()
    today = timezone.now().date()
    review_overdue = active.filter(review_date__isnull=False, review_date__lt=today).count()
    no_mitigation = active.filter(mitigation_actions=[]).exclude(status="accepted").count()

    findings = []
    score = 100
    if high > 0 and no_mitigation > 0:
        score -= min(30, no_mitigation * 10)
        findings.append(f"{no_mitigation} risk(s) without mitigation actions")
    if review_overdue > 0:
        score -= min(20, review_overdue * 10)
        findings.append(f"{review_overdue} risk(s) past review date")

    rag = "green" if score >= 80 else "amber" if score >= 50 else "red"
    return max(0, score), rag, "; ".join(findings) if findings else f"{total} risks registered"


def _score_calibration(user):
    """§7.1.5 — Monitoring and measurement resources."""
    qs = qms_queryset(MeasurementEquipment, user)[0]
    total = qs.count()
    if total == 0:
        return 60, "amber", "No equipment registered — acceptable for service organizations"

    today = timezone.now().date()
    overdue = qs.filter(
        next_calibration_due__isnull=False,
        next_calibration_due__lt=today,
        status__in=["in_service", "due"],
    ).count()
    out_of_cal = qs.filter(status="out_of_calibration").count()

    findings = []
    score = 100
    if overdue > 0:
        score -= min(40, overdue * 15)
        findings.append(f"{overdue} instrument(s) past calibration due date")
    if out_of_cal > 0:
        score -= min(30, out_of_cal * 10)
        findings.append(f"{out_of_cal} instrument(s) out of calibration — impact assessment required")

    rag = "green" if score >= 80 else "amber" if score >= 50 else "red"
    return max(0, score), rag, "; ".join(findings) if findings else f"{total} instruments managed"


# Clause → scoring functions + weights
_CLAUSE_CHECKS = [
    ("4", "Context of the Organization", None, 0),  # No automated check yet
    ("5", "Leadership", None, 0),  # No automated check yet
    ("6", "Planning", _score_risks, 10),
    ("7.1.5", "Monitoring & Measurement", _score_calibration, 8),
    ("7.2", "Competence", _score_training, 12),
    ("7.5", "Documented Information", _score_documents, 12),
    ("8.4", "External Providers", _score_suppliers, 8),
    ("8.7", "Nonconforming Outputs", _score_ncrs, 12),
    ("9.1.2", "Customer Satisfaction", _score_complaints, 10),
    ("9.2", "Internal Audit", _score_audits, 12),
    ("9.3", "Management Review", _score_management_review, 8),
    ("10.2", "Corrective Action", _score_capas, 8),
]


@require_team
@require_http_methods(["GET"])
def audit_readiness(request):
    """E4: ISO 9001 audit readiness score.

    Returns per-clause RAG status and overall readiness score (0-100).
    Deterministic — no LLM. All scoring from live QMS data.

    Optional query param: ?narrative=1 triggers Claude interpretation
    (future — not implemented in v1).
    """
    user = request.user
    clauses = []
    weighted_sum = 0
    total_weight = 0

    for clause_num, clause_name, score_fn, weight in _CLAUSE_CHECKS:
        if score_fn is None:
            clauses.append(
                {
                    "clause": clause_num,
                    "name": clause_name,
                    "score": None,
                    "rag": "amber",
                    "detail": "Not yet automated — manual assessment required",
                    "weight": weight,
                }
            )
            continue

        score, rag, detail = score_fn(user)
        clauses.append(
            {
                "clause": clause_num,
                "name": clause_name,
                "score": score,
                "rag": rag,
                "detail": detail,
                "weight": weight,
            }
        )
        weighted_sum += score * weight
        total_weight += weight

    overall_score = round(weighted_sum / total_weight) if total_weight else 0
    overall_rag = "green" if overall_score >= 80 else "amber" if overall_score >= 50 else "red"

    red_count = sum(1 for c in clauses if c["rag"] == "red")
    amber_count = sum(1 for c in clauses if c["rag"] == "amber")
    green_count = sum(1 for c in clauses if c["rag"] == "green")

    # Top findings (red clauses first, then amber, sorted by score ascending)
    scored = [c for c in clauses if c["score"] is not None]
    top_findings = sorted(scored, key=lambda c: c["score"] or 0)[:5]

    return JsonResponse(
        {
            "overall_score": overall_score,
            "overall_rag": overall_rag,
            "red": red_count,
            "amber": amber_count,
            "green": green_count,
            "clauses": clauses,
            "top_findings": [
                {"clause": f["clause"], "name": f["name"], "score": f["score"], "detail": f["detail"]}
                for f in top_findings
            ],
            "interpretation": _build_interpretation(
                overall_score, overall_rag, red_count, amber_count, clauses, top_findings
            ),
        }
    )


def _build_interpretation(score, rag, red_count, amber_count, clauses, top_findings):
    """Build human-readable interpretation of the audit readiness score."""

    # Overall assessment
    if score >= 90:
        headline = "Audit-ready"
        summary = (
            "Your QMS is well-maintained with strong evidence across ISO 9001 clauses. "
            "A registrar would find a functioning system with current records. "
            "Focus on maintaining this level \u2014 review the amber items below to close remaining gaps."
        )
    elif score >= 80:
        headline = "Near audit-ready"
        summary = (
            "Your QMS covers most requirements with some gaps to close. "
            "A registrar would likely find minor nonconformities but no systemic issues. "
            "Address the findings below to reach full readiness."
        )
    elif score >= 65:
        headline = "Progressing"
        summary = (
            "Your QMS has good foundation but significant gaps remain. "
            "A registrar would likely raise several findings. "
            "Prioritize the red clauses below \u2014 these represent the highest audit risk."
        )
    elif score >= 50:
        headline = "Early stage"
        summary = (
            "Your QMS modules are in place but most need to be populated with records. "
            "This is normal for a new system. "
            "The score reflects what a registrar would see today \u2014 "
            "start with the top findings below and the score will climb as you add records."
        )
    else:
        headline = "Getting started"
        summary = (
            "Your QMS is set up but hasn\u2019t been populated yet. "
            "This score is expected for a new deployment \u2014 it\u2019s not a failure, it\u2019s a starting point. "
            "Each red clause below tells you exactly what to do next. "
            "Most organizations reach 70+ within the first month of active use."
        )

    # Next actions from top findings
    next_actions = []
    for f in top_findings[:3]:
        clause = f["clause"]
        detail = f["detail"]
        if "No " in detail and "recorded" in detail:
            next_actions.append(f"Create your first record in {f['name']} (clause {clause})")
        elif "No " in detail and "scheduled" in detail:
            next_actions.append(f"Schedule your first activity in {f['name']} (clause {clause})")
        elif "No " in detail and "defined" in detail:
            next_actions.append(f"Define requirements for {f['name']} (clause {clause})")
        elif "No " in detail and "identified" in detail:
            next_actions.append(f"Identify and register items for {f['name']} (clause {clause})")
        elif "overdue" in detail.lower() or "past due" in detail.lower():
            next_actions.append(f"Address overdue items in {f['name']} (clause {clause})")
        else:
            next_actions.append(f"Review {f['name']} (clause {clause}): {detail}")

    return {
        "headline": headline,
        "summary": summary,
        "next_actions": next_actions,
        "methodology": (
            "This score is computed from live QMS data across 10 ISO 9001 clauses. "
            "Each clause is scored 0\u2013100 based on record completeness, overdue items, "
            "and compliance gaps. Clauses are weighted by audit importance. "
            "Green (80\u2013100) means a registrar would find conforming evidence. "
            "Amber (50\u201379) means partial evidence or minor gaps. "
            "Red (0\u201349) means a registrar would likely raise a finding. "
            "The score updates in real time as you create and manage QMS records."
        ),
    }

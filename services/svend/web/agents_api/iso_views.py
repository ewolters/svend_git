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
from datetime import date, timedelta

from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from accounts.permissions import require_team

from accounts.models import User

from .models import (
    NonconformanceRecord,
    NCRStatusChange,
    InternalAudit,
    AuditFinding,
    AuditChecklist,
    TrainingRequirement,
    TrainingRecord,
    ManagementReview,
    ControlledDocument,
    SupplierRecord,
    RCASession,
)


# =========================================================================
# Dashboard overview
# =========================================================================

@require_team
@require_http_methods(["GET"])
def iso_dashboard(request):
    """QMS dashboard overview — KPIs across all modules."""
    user = request.user
    now = timezone.now()
    thirty_days = now + timedelta(days=30)
    ninety_ago = now - timedelta(days=90)

    # NCR stats
    ncrs = NonconformanceRecord.objects.filter(owner=user)
    open_ncrs = ncrs.exclude(status="closed")
    overdue = open_ncrs.filter(capa_due_date__lt=now.date()).count()

    # Audit stats
    audits = InternalAudit.objects.filter(owner=user)
    upcoming = audits.filter(
        scheduled_date__gte=now.date(),
        scheduled_date__lte=thirty_days.date(),
        status__in=["planned", "in_progress"],
    ).count()

    # Training stats
    reqs = TrainingRequirement.objects.filter(owner=user)
    all_records = TrainingRecord.objects.filter(requirement__owner=user)
    total_records = all_records.count()
    complete_records = all_records.filter(status="complete").count()
    expiring = all_records.filter(
        expires_at__isnull=False,
        expires_at__lte=thirty_days,
    ).count()

    # Review stats
    reviews = ManagementReview.objects.filter(owner=user)
    next_review = reviews.filter(
        meeting_date__gte=now.date(), status__in=["scheduled", "in_progress"],
    ).order_by("meeting_date").values_list("meeting_date", flat=True).first()

    # Document stats
    docs = ControlledDocument.objects.filter(owner=user)
    docs_approved = docs.filter(status="approved").count()
    docs_overdue = docs.filter(
        review_due_date__lt=now.date(), status="approved",
    ).count()

    # Supplier stats
    suppliers = SupplierRecord.objects.filter(owner=user)
    suppliers_approved = suppliers.filter(status="approved").count()
    eval_due = suppliers.filter(
        next_evaluation_date__lt=now.date(),
    ).count()

    return JsonResponse({
        "ncrs": {
            "open": open_ncrs.count(),
            "overdue": overdue,
            "total_90d": ncrs.filter(created_at__gte=ninety_ago).count(),
            "closed_90d": ncrs.filter(closed_at__gte=ninety_ago).count(),
        },
        "audits": {"upcoming_30d": upcoming, "total": audits.count()},
        "training": {
            "compliance_rate": round(complete_records / total_records * 100) if total_records else 100,
            "expiring_30d": expiring,
            "total_requirements": reqs.count(),
        },
        "reviews": {
            "next_review": str(next_review) if next_review else None,
            "total": reviews.count(),
        },
        "documents": {
            "total": docs.count(),
            "approved": docs_approved,
            "review_overdue": docs_overdue,
        },
        "suppliers": {
            "total": suppliers.count(),
            "approved": suppliers_approved,
            "evaluation_due": eval_due,
        },
    })


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
        allowed_sorts = {"created_at", "-created_at", "severity", "-severity",
                         "status", "-status", "capa_due_date", "-capa_due_date",
                         "title", "-title"}
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
    # Attach files if provided
    file_ids = data.get("file_ids", [])
    if file_ids:
        from files.models import UserFile
        files = UserFile.objects.filter(id__in=file_ids, user=user)
        ncr.files.set(files)
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
        # Set assigned_to / approved_by before transition check
        if "assigned_to" in data:
            if data["assigned_to"]:
                try:
                    ncr.assigned_to = User.objects.get(id=data["assigned_to"])
                except User.DoesNotExist:
                    pass
            else:
                ncr.assigned_to = None
        if "approved_by" in data:
            if data["approved_by"]:
                try:
                    ncr.approved_by = User.objects.get(id=data["approved_by"])
                except User.DoesNotExist:
                    pass
            else:
                ncr.approved_by = None

        ok, msg = ncr.can_transition(new_status)
        if not ok:
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

        if new_status == "closed" and not ncr.closed_at:
            ncr.closed_at = timezone.now()

    # Update other fields
    for field in ["title", "description", "severity", "source",
                   "iso_clause", "containment_action", "root_cause",
                   "corrective_action", "verification_result"]:
        if field in data:
            setattr(ncr, field, data[field])
    if "capa_due_date" in data:
        ncr.capa_due_date = data["capa_due_date"] or None
    # Handle assigned_to/approved_by if not already handled by transition
    if "assigned_to" in data and not new_status:
        if data["assigned_to"]:
            try:
                ncr.assigned_to = User.objects.get(id=data["assigned_to"])
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
    # File attachments
    if "file_ids" in data:
        from files.models import UserFile
        files = UserFile.objects.filter(id__in=data["file_ids"], user=request.user)
        ncr.files.set(files)
    ncr.save()
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
    avg_close = ncrs.filter(closed_at__isnull=False).aggregate(
        avg_days=Avg(F("closed_at") - F("created_at"))
    )["avg_days"]
    avg_close_days = avg_close.days if avg_close else None

    return JsonResponse({
        "total": ncrs.count(),
        "open": open_ncrs.count(),
        "overdue_capas": overdue,
        "avg_close_days": avg_close_days,
    })


@require_team
@require_http_methods(["POST"])
def ncr_launch_rca(request, ncr_id):
    """Create an RCA session linked to this NCR."""
    try:
        ncr = NonconformanceRecord.objects.get(id=ncr_id, owner=request.user)
    except NonconformanceRecord.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    session = RCASession.objects.create(
        owner=request.user,
        title=f"RCA for {ncr.title}",
        event=ncr.description or ncr.title,
        status="draft",
    )
    ncr.rca_session = session
    ncr.save(update_fields=["rca_session"])
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

    for field in ["title", "status", "lead_auditor", "iso_clauses",
                   "departments", "scope", "summary"]:
        if field in data:
            setattr(audit, field, data[field])
    if "scheduled_date" in data:
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
        finding.ncr = ncr
        finding.save(update_fields=["ncr"])
        ncr_id = str(ncr.id)

    result = finding.to_dict()
    if ncr_id:
        result["ncr_id"] = ncr_id
    return JsonResponse(result, status=201)


# =========================================================================
# Training Matrix
# =========================================================================

@require_team
@require_http_methods(["GET", "POST"])
def training_list_create(request):
    """List training requirements or create a new one."""
    user = request.user

    if request.method == "GET":
        reqs = TrainingRequirement.objects.filter(owner=user).prefetch_related("records")
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
    req.save()
    return JsonResponse(req.to_dict())


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
            id=record_id, requirement__owner=request.user,
        )
    except TrainingRecord.DoesNotExist:
        return JsonResponse({"error": "Not found"}, status=404)

    if request.method == "DELETE":
        record.delete()
        return JsonResponse({"ok": True})

    data = json.loads(request.body)
    if "status" in data:
        record.status = data["status"]
        if data["status"] == "complete" and not record.completed_at:
            record.completed_at = timezone.now()
            req = record.requirement
            if req.frequency_months > 0:
                record.expires_at = timezone.now() + timedelta(days=req.frequency_months * 30)
    if "employee_name" in data:
        record.employee_name = data["employee_name"]
    if "notes" in data:
        record.notes = data["notes"]
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

    # Auto-populate snapshot
    ncrs = NonconformanceRecord.objects.filter(owner=user)
    audits = InternalAudit.objects.filter(owner=user)
    records = TrainingRecord.objects.filter(requirement__owner=user)
    total_records = records.count()
    complete_records = records.filter(status="complete").count()

    snapshot = {
        "ncrs": {
            "total": ncrs.count(),
            "open": ncrs.exclude(status="closed").count(),
        },
        "audits": {
            "total": audits.count(),
            "complete": audits.filter(status="complete").count(),
        },
        "training": {
            "total_records": total_records,
            "complete": complete_records,
            "rate": round(complete_records / total_records * 100) if total_records else 100,
        },
        "captured_at": timezone.now().isoformat(),
    }

    data = json.loads(request.body) if request.body else {}
    review = ManagementReview.objects.create(
        owner=user,
        title=data.get("title", f"Management Review — {date.today().strftime('%B %Y')}"),
        meeting_date=data.get("meeting_date", date.today()),
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
# Document Control — skeleton
# =========================================================================

@require_team
@require_http_methods(["GET", "POST"])
def document_list_create(request):
    """List controlled documents or create a new one."""
    user = request.user

    if request.method == "GET":
        docs = ControlledDocument.objects.filter(owner=user)
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
    for field in ["title", "document_number", "status", "category",
                   "iso_clause", "current_version", "approved_by", "content"]:
        if field in data:
            setattr(doc, field, data[field])
    if "review_due_date" in data:
        doc.review_due_date = data["review_due_date"] or None
    if data.get("status") == "approved" and not doc.approved_at:
        doc.approved_at = timezone.now()
    doc.save()
    return JsonResponse(doc.to_dict())


# =========================================================================
# Supplier Management — skeleton
# =========================================================================

@require_team
@require_http_methods(["GET", "POST"])
def supplier_list_create(request):
    """List suppliers or create a new one."""
    user = request.user

    if request.method == "GET":
        suppliers = SupplierRecord.objects.filter(owner=user)
        return JsonResponse([s.to_dict() for s in suppliers[:100]], safe=False)

    data = json.loads(request.body)
    supplier = SupplierRecord.objects.create(
        owner=user,
        name=data.get("name", ""),
        contact_name=data.get("contact_name", ""),
        contact_email=data.get("contact_email", ""),
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
    for field in ["name", "status", "contact_name", "contact_email",
                   "products_services", "quality_rating", "notes"]:
        if field in data:
            setattr(supplier, field, data[field])
    if "next_evaluation_date" in data:
        supplier.next_evaluation_date = data["next_evaluation_date"] or None
    if "last_evaluation_date" in data:
        supplier.last_evaluation_date = data["last_evaluation_date"] or None
    supplier.save()
    return JsonResponse(supplier.to_dict())

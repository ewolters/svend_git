"""Supplier Claims and CoA views — extracted from loop/views.py.

Supplier quality claims (per supplier_claim_api_spec.md) and
Certificate of Analysis (Object 271 — Phases 5-7).
"""

import csv
import io
import json
import logging

from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid
from qms_core.permissions import get_tenant as _get_tenant

from .models import (
    ClaimVerification,
    SupplierClaim,
    SupplierCoA,
    SupplierResponse,
)

logger = logging.getLogger("svend.loop")


# =============================================================================
# SUPPLIER CLAIMS — per supplier_claim_api_spec.md
# =============================================================================


def _serialize_claim(claim, include_responses=False):
    data = {
        "id": str(claim.id),
        "title": claim.title,
        "description": claim.description,
        "claim_type": claim.claim_type,
        "supplier_id": str(claim.supplier_id),
        "supplier_name": claim.supplier.name if hasattr(claim, "supplier") else "",
        "ncr_id": str(claim.ncr_id) if claim.ncr_id else None,
        "part_number": claim.part_number,
        "lot_number": claim.lot_number,
        "quantity_affected": claim.quantity_affected,
        "quantity_rejected": claim.quantity_rejected,
        "defect_description": claim.defect_description,
        "inspection_method": claim.inspection_method,
        "evidence_photos": claim.evidence_photos,
        "cost_of_quality": float(claim.cost_of_quality) if claim.cost_of_quality else None,
        "credit_requested": float(claim.credit_requested) if claim.credit_requested else None,
        "credit_received": float(claim.credit_received) if claim.credit_received else None,
        "disposition": claim.disposition,
        "status": claim.status,
        "filed_at": claim.filed_at.isoformat() if claim.filed_at else None,
        "response_due_date": claim.response_due_date.isoformat() if claim.response_due_date else None,
        "portal_is_valid": claim.portal_is_valid,
        "response_is_overdue": claim.response_is_overdue,
        "linked_process_node_ids": claim.linked_process_node_ids,
        "response_count": claim.responses.count(),
        "verification_count": claim.verifications.count(),
        "created_by_id": str(claim.created_by_id),
        "created_at": claim.created_at.isoformat() if claim.created_at else None,
        "updated_at": claim.updated_at.isoformat() if claim.updated_at else None,
    }
    if include_responses:
        data["responses"] = [_serialize_response(r) for r in claim.responses.all()]
        data["verifications"] = [_serialize_verification(v) for v in claim.verifications.all()]
    return data


def _serialize_response(r):
    return {
        "id": str(r.id),
        "revision": r.revision,
        "root_cause_category": r.root_cause_category,
        "root_cause_description": r.root_cause_description,
        "corrective_action": r.corrective_action,
        "preventive_action": r.preventive_action,
        "implementation_date": r.implementation_date.isoformat() if r.implementation_date else None,
        "evidence_files": r.evidence_files,
        "is_repeat_root_cause": r.is_repeat_root_cause,
        "response_quality_score": r.response_quality_score,
        "quality_flags": r.quality_flags,
        "accepted": r.accepted,
        "reviewer_notes": r.reviewer_notes,
        "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def _serialize_verification(v):
    return {
        "id": str(v.id),
        "verification_type": v.verification_type,
        "result": v.result,
        "evidence_notes": v.evidence_notes,
        "evidence_files": v.evidence_files,
        "verified_by_id": str(v.verified_by_id) if v.verified_by_id else None,
        "verified_at": v.verified_at.isoformat() if v.verified_at else None,
    }


@gated_paid
@require_http_methods(["GET", "POST"])
def claim_list_create(request):
    """List or create supplier claims."""
    tenant = _get_tenant(request.user)
    if not tenant:
        return JsonResponse({"error": "No tenant"}, status=400)

    if request.method == "GET":
        qs = SupplierClaim.objects.filter(tenant=tenant).select_related("supplier")
        supplier_id = request.GET.get("supplier_id")
        if supplier_id:
            qs = qs.filter(supplier_id=supplier_id)
        status_filter = request.GET.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)
        return JsonResponse([_serialize_claim(c) for c in qs[:100]], safe=False)

    data = json.loads(request.body)
    supplier_id = data.get("supplier_id")
    if not supplier_id:
        return JsonResponse({"error": "supplier_id required"}, status=400)

    claim = SupplierClaim.objects.create(
        tenant=tenant,
        supplier_id=supplier_id,
        ncr_id=data.get("ncr_id"),
        claim_type=data.get("claim_type", "quality_defect"),
        title=data.get("title", ""),
        description=data.get("description", ""),
        part_number=data.get("part_number", ""),
        lot_number=data.get("lot_number", ""),
        quantity_affected=data.get("quantity_affected", 0),
        quantity_rejected=data.get("quantity_rejected", 0),
        defect_description=data.get("defect_description", ""),
        inspection_method=data.get("inspection_method", ""),
        disposition=data.get("disposition", ""),
        cost_of_quality=data.get("cost_of_quality"),
        credit_requested=data.get("credit_requested"),
        response_due_date=data.get("response_due_date"),
        linked_process_node_ids=data.get("linked_process_node_ids", []),
        created_by=request.user,
    )
    return JsonResponse(_serialize_claim(claim), status=201)


@gated_paid
@require_http_methods(["GET", "POST", "DELETE"])
def claim_detail(request, claim_id):
    """Get claim detail, perform lifecycle action, or delete draft."""
    tenant = _get_tenant(request.user)
    if not tenant:
        return JsonResponse({"error": "No tenant"}, status=400)

    try:
        claim = SupplierClaim.objects.select_related("supplier").get(id=claim_id, tenant=tenant)
    except SupplierClaim.DoesNotExist:
        return JsonResponse({"error": "Claim not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(_serialize_claim(claim, include_responses=True))

    if request.method == "DELETE":
        if claim.status != "draft":
            return JsonResponse({"error": "Can only delete draft claims"}, status=400)
        claim.delete()
        return JsonResponse({"deleted": True})

    # POST — lifecycle action
    data = json.loads(request.body)
    action = data.get("action")

    if action == "file":
        if claim.status != "draft":
            return JsonResponse({"error": "Can only file from draft"}, status=400)
        claim.status = "filed"
        claim.filed_at = timezone.now()
        claim.generate_portal_token()
        claim.save(update_fields=["status", "filed_at", "updated_at"])

        # Send email to supplier
        try:
            from notifications.email_service import email_service

            email_service.send(
                to=claim.supplier.contact_email,
                subject=f"Quality Claim: {claim.title}",
                body_text=(
                    f"A quality claim has been filed regarding: {claim.title}\n\n"
                    f"Claim type: {claim.get_claim_type_display()}\n"
                    f"Part: {claim.part_number}\n"
                    f"Description: {claim.defect_description[:200]}\n\n"
                    f"Please respond at: https://svend.ai/supplier-claim/{claim.portal_token}/\n\n"
                    f"Response due: {claim.response_due_date or 'Not specified'}"
                ),
                wrap_template=True,
            )
        except Exception as e:
            logger.warning("Failed to send claim email: %s", e)

        return JsonResponse(_serialize_claim(claim))

    elif action == "add_verification":
        verification = ClaimVerification.objects.create(
            claim=claim,
            verification_type=data.get("verification_type", "next_shipment"),
            result=data.get("result", "conforming"),
            evidence_notes=data.get("evidence_notes", ""),
            evidence_files=data.get("evidence_files", []),
            verified_by=request.user,
        )
        return JsonResponse(_serialize_verification(verification), status=201)

    elif action in ("acknowledge", "submit_for_review", "accept", "reject", "escalate", "verify"):
        target_status = {
            "acknowledge": "acknowledged",
            "submit_for_review": "under_review",
            "accept": "verified",
            "reject": "rejected",
            "escalate": "escalated",
            "verify": "closed",
        }.get(action)

        valid = SupplierClaim.VALID_TRANSITIONS.get(claim.status, set())
        if target_status not in valid:
            return JsonResponse(
                {"error": f"Cannot {action} from {claim.status}. Valid: {valid}"},
                status=400,
            )

        claim.status = target_status
        claim.save(update_fields=["status", "updated_at"])

        # If accepting/rejecting, update latest response
        if action in ("accept", "reject"):
            latest = claim.responses.order_by("-revision").first()
            if latest:
                latest.accepted = action == "accept"
                latest.reviewer = request.user
                latest.reviewer_notes = data.get("reviewer_notes", "")
                latest.reviewed_at = timezone.now()
                latest.save(update_fields=["accepted", "reviewer", "reviewer_notes", "reviewed_at"])

        return JsonResponse(_serialize_claim(claim))

    return JsonResponse({"error": f"Unknown action: {action}"}, status=400)


@gated_paid
@require_http_methods(["POST"])
def claim_respond(request, claim_id):
    """Submit a CAPA response to a claim (internal or via portal)."""
    tenant = _get_tenant(request.user)
    if not tenant:
        return JsonResponse({"error": "No tenant"}, status=400)

    try:
        claim = SupplierClaim.objects.select_related("supplier").get(id=claim_id, tenant=tenant)
    except SupplierClaim.DoesNotExist:
        return JsonResponse({"error": "Claim not found"}, status=404)

    if claim.status not in ("filed", "acknowledged", "rejected"):
        return JsonResponse(
            {"error": f"Cannot respond in status {claim.status}"},
            status=400,
        )

    data = json.loads(request.body)
    revision = claim.responses.count() + 1

    response = SupplierResponse.objects.create(
        claim=claim,
        root_cause_category=data.get("root_cause_category", "method"),
        root_cause_description=data.get("root_cause_description", ""),
        corrective_action=data.get("corrective_action", ""),
        preventive_action=data.get("preventive_action", ""),
        implementation_date=data.get("implementation_date"),
        evidence_files=data.get("evidence_files", []),
        revision=revision,
    )

    # Run quality scoring
    from .response_quality import evaluate_response_quality

    score, flags = evaluate_response_quality(claim, response, claim.supplier)
    response.response_quality_score = score
    response.quality_flags = flags
    response.save(update_fields=["response_quality_score", "quality_flags", "is_repeat_root_cause"])

    # Transition claim
    claim.status = "responded"
    claim.save(update_fields=["status", "updated_at"])

    return JsonResponse(_serialize_response(response), status=201)


# ── Portal endpoints (token-authenticated, no login) ──


@require_http_methods(["GET"])
def claim_portal_data(request, token):
    """Get claim data for supplier portal (no auth — token in URL)."""
    try:
        claim = SupplierClaim.objects.select_related("supplier").get(portal_token=token)
    except SupplierClaim.DoesNotExist:
        return JsonResponse({"error": "Invalid or expired link"}, status=404)

    if not claim.portal_is_valid:
        return JsonResponse({"error": "Link expired"}, status=403)

    # Auto-acknowledge on first access
    if claim.status == "filed":
        claim.status = "acknowledged"
        claim.save(update_fields=["status", "updated_at"])

    return JsonResponse(
        {
            "title": claim.title,
            "claim_type": claim.get_claim_type_display(),
            "description": claim.description,
            "part_number": claim.part_number,
            "lot_number": claim.lot_number,
            "quantity_affected": claim.quantity_affected,
            "defect_description": claim.defect_description,
            "inspection_method": claim.inspection_method,
            "disposition": claim.get_disposition_display() if claim.disposition else "",
            "response_due_date": claim.response_due_date.isoformat() if claim.response_due_date else None,
            "status": claim.status,
            "response_is_overdue": claim.response_is_overdue,
            "responses": [_serialize_response(r) for r in claim.responses.all()],
        }
    )


@require_http_methods(["POST"])
def claim_portal_respond(request, token):
    """Supplier submits CAPA response via portal (no auth — token in URL)."""
    try:
        claim = SupplierClaim.objects.select_related("supplier").get(portal_token=token)
    except SupplierClaim.DoesNotExist:
        return JsonResponse({"error": "Invalid or expired link"}, status=404)

    if not claim.portal_is_valid:
        return JsonResponse({"error": "Link expired"}, status=403)

    if claim.status not in ("filed", "acknowledged", "rejected"):
        return JsonResponse(
            {"error": f"Cannot respond in status {claim.status}"},
            status=400,
        )

    data = json.loads(request.body)
    revision = claim.responses.count() + 1

    response = SupplierResponse.objects.create(
        claim=claim,
        root_cause_category=data.get("root_cause_category", "method"),
        root_cause_description=data.get("root_cause_description", ""),
        corrective_action=data.get("corrective_action", ""),
        preventive_action=data.get("preventive_action", ""),
        implementation_date=data.get("implementation_date"),
        evidence_files=data.get("evidence_files", []),
        revision=revision,
    )

    # Run quality scoring
    from .response_quality import evaluate_response_quality

    score, flags = evaluate_response_quality(claim, response, claim.supplier)
    response.response_quality_score = score
    response.quality_flags = flags
    response.save(update_fields=["response_quality_score", "quality_flags", "is_repeat_root_cause"])

    claim.status = "responded"
    claim.save(update_fields=["status", "updated_at"])

    return JsonResponse({"success": True, "revision": revision, "quality_score": score})


# =============================================================================
# SUPPLIER CoA (Object 271 — Phases 5-7)
# =============================================================================


def _serialize_coa(coa):
    return {
        "id": str(coa.id),
        "supplier_id": str(coa.supplier_id),
        "supplier_name": coa.supplier.name if hasattr(coa, "supplier") else "",
        "coa_number": coa.coa_number,
        "lot_number": coa.lot_number,
        "part_number": coa.part_number,
        "date_issued": coa.date_issued.isoformat() if coa.date_issued else None,
        "measurements": coa.measurements,
        "extraction_method": coa.extraction_method,
        "all_conforming": coa.all_conforming,
        "nonconforming_parameters": coa.nonconforming_parameters,
        "spc_data_ingested": coa.spc_data_ingested,
        "linked_process_node_ids": coa.linked_process_node_ids,
        "status": coa.status,
        "rejection_reason": coa.rejection_reason,
        "created_at": coa.created_at.isoformat() if coa.created_at else None,
        "updated_at": coa.updated_at.isoformat() if coa.updated_at else None,
    }


@gated_paid
@require_http_methods(["GET", "POST"])
def coa_list_create(request):
    """List or create supplier CoAs."""
    tenant = _get_tenant(request.user)
    if not tenant:
        return JsonResponse({"error": "No tenant"}, status=400)

    if request.method == "GET":
        qs = SupplierCoA.objects.filter(tenant=tenant).select_related("supplier")
        supplier_id = request.GET.get("supplier_id")
        if supplier_id:
            qs = qs.filter(supplier_id=supplier_id)
        status_filter = request.GET.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)
        return JsonResponse([_serialize_coa(c) for c in qs[:100]], safe=False)

    data = json.loads(request.body)
    supplier_id = data.get("supplier_id")
    if not supplier_id:
        return JsonResponse({"error": "supplier_id required"}, status=400)

    measurements = data.get("measurements", [])

    coa = SupplierCoA.objects.create(
        tenant=tenant,
        supplier_id=supplier_id,
        coa_number=data.get("coa_number", ""),
        lot_number=data.get("lot_number", ""),
        part_number=data.get("part_number", ""),
        date_issued=data.get("date_issued"),
        measurements=measurements,
        extraction_method=data.get("extraction_method", "manual"),
        created_by=request.user,
    )

    # Auto-check compliance
    if measurements:
        coa.check_compliance()

    return JsonResponse(_serialize_coa(coa), status=201)


@gated_paid
@require_http_methods(["GET", "POST"])
def coa_detail(request, coa_id):
    """Get or action a CoA."""
    tenant = _get_tenant(request.user)
    if not tenant:
        return JsonResponse({"error": "No tenant"}, status=400)

    try:
        coa = SupplierCoA.objects.select_related("supplier").get(id=coa_id, tenant=tenant)
    except SupplierCoA.DoesNotExist:
        return JsonResponse({"error": "CoA not found"}, status=404)

    if request.method == "GET":
        return JsonResponse(_serialize_coa(coa))

    data = json.loads(request.body)
    action = data.get("action")

    if action == "review":
        coa.status = SupplierCoA.Status.REVIEWED
        coa.reviewed_by = request.user
        coa.reviewed_at = timezone.now()
        coa.save(update_fields=["status", "reviewed_by", "reviewed_at", "updated_at"])

    elif action == "accept":
        coa.status = SupplierCoA.Status.ACCEPTED
        if not coa.reviewed_by:
            coa.reviewed_by = request.user
            coa.reviewed_at = timezone.now()
        coa.save(update_fields=["status", "reviewed_by", "reviewed_at", "updated_at"])

    elif action == "reject":
        coa.status = SupplierCoA.Status.REJECTED
        coa.rejection_reason = data.get("reason", "")
        coa.reviewed_by = request.user
        coa.reviewed_at = timezone.now()
        coa.save(update_fields=["status", "rejection_reason", "reviewed_by", "reviewed_at", "updated_at"])

    elif action == "ingest":
        # Push measurements to graph nodes + SPC charts
        from graph.integrations import coa_to_graph_and_spc

        ingest_result = coa_to_graph_and_spc(
            tenant_id=tenant.id,
            coa_id=coa.id,
            user=request.user,
        )

        coa.status = SupplierCoA.Status.INGESTED
        coa.spc_data_ingested = True
        coa.spc_ingestion_date = timezone.now()
        coa.save(update_fields=["status", "spc_data_ingested", "spc_ingestion_date", "updated_at"])

        result = _serialize_coa(coa)
        result["ingest_result"] = ingest_result
        return JsonResponse(result)

    elif action == "update_measurements":
        coa.measurements = data.get("measurements", [])
        coa.check_compliance()

    elif action == "recheck":
        coa.check_compliance()

    else:
        return JsonResponse({"error": f"Unknown action: {action}"}, status=400)

    return JsonResponse(_serialize_coa(coa))


@gated_paid
@require_http_methods(["POST"])
def coa_csv_upload(request):
    """Parse CSV CoA data and create a CoA record.

    CSV format: parameter,value,unit,spec_min,spec_max,method
    First row is header. One measurement per row.
    """
    tenant = _get_tenant(request.user)
    if not tenant:
        return JsonResponse({"error": "No tenant"}, status=400)

    data = json.loads(request.body)
    supplier_id = data.get("supplier_id")
    csv_text = data.get("csv_data", "")

    if not supplier_id or not csv_text:
        return JsonResponse({"error": "supplier_id and csv_data required"}, status=400)

    measurements = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        try:
            m = {
                "parameter": row.get("parameter", "").strip(),
                "unit": row.get("unit", "").strip(),
                "method": row.get("method", "").strip(),
            }
            val = row.get("value", "").strip()
            m["value"] = float(val) if val else None
            spec_min = row.get("spec_min", "").strip()
            m["spec_min"] = float(spec_min) if spec_min else None
            spec_max = row.get("spec_max", "").strip()
            m["spec_max"] = float(spec_max) if spec_max else None

            if m["parameter"]:
                measurements.append(m)
        except (ValueError, TypeError):
            continue

    if not measurements:
        return JsonResponse({"error": "No valid measurements found in CSV"}, status=400)

    coa = SupplierCoA.objects.create(
        tenant=tenant,
        supplier_id=supplier_id,
        coa_number=data.get("coa_number", ""),
        lot_number=data.get("lot_number", ""),
        part_number=data.get("part_number", ""),
        date_issued=data.get("date_issued"),
        measurements=measurements,
        extraction_method="csv",
        created_by=request.user,
    )
    coa.check_compliance()

    return JsonResponse(_serialize_coa(coa), status=201)

"""Loop API views — LOOP-001 §3.

Endpoints for Signals, Commitments, and ModeTransitions.
"""

import json
import logging
from datetime import date

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid

from .models import (
    Commitment,
    ForcedFailureTest,
    ModeTransition,
    PCObservationItem,
    ProcessConfirmation,
    Signal,
    TrainingReflection,
)
from .services import fulfill_commitment

logger = logging.getLogger("svend.loop")


# =============================================================================
# SERIALIZERS
# =============================================================================


def _serialize_signal(s):
    return {
        "id": str(s.id),
        "title": s.title,
        "description": s.description,
        "source_type": s.source_type,
        "severity": s.severity,
        "triage_state": s.triage_state,
        "source_object_id": str(s.source_object_id) if s.source_object_id else None,
        "resolved_by_investigation_id": (
            str(s.resolved_by_investigation_id) if s.resolved_by_investigation_id else None
        ),
        "dismissed_reason": s.dismissed_reason,
        "created_by_id": str(s.created_by_id),
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "resolved_at": s.resolved_at.isoformat() if s.resolved_at else None,
    }


def _serialize_commitment(c):
    return {
        "id": str(c.id),
        "title": c.title,
        "description": c.description,
        "owner_id": str(c.owner_id),
        "due_date": c.due_date.isoformat() if c.due_date else None,
        "preconditions": c.preconditions,
        "status": c.status,
        "transition_type": c.transition_type,
        "source_type": c.source_type,
        "source_investigation_id": (str(c.source_investigation_id) if c.source_investigation_id else None),
        "target_object_id": str(c.target_object_id) if c.target_object_id else None,
        "is_overdue": c.is_overdue,
        "is_blocked": c.is_blocked,
        "created_by_id": str(c.created_by_id),
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "fulfilled_at": c.fulfilled_at.isoformat() if c.fulfilled_at else None,
    }


def _serialize_transition(t):
    return {
        "id": str(t.id),
        "transition_type": t.transition_type,
        "from_mode": t.from_mode,
        "to_mode": t.to_mode,
        "source_object_id": str(t.source_object_id),
        "target_object_id": str(t.target_object_id),
        "triggered_by_id": str(t.triggered_by_id) if t.triggered_by_id else None,
        "created_at": t.created_at.isoformat(),
    }


# =============================================================================
# SIGNALS (LOOP-001 §3.1)
# =============================================================================


@gated_paid
@require_http_methods(["GET", "POST"])
def signal_list_create(request):
    """
    GET  — List signals (filterable by triage_state, severity).
    POST — Create a signal.
    """
    if request.method == "GET":
        qs = Signal.objects.filter(created_by=request.user)
        state = request.GET.get("triage_state")
        if state:
            qs = qs.filter(triage_state=state)
        severity = request.GET.get("severity")
        if severity:
            qs = qs.filter(severity=severity)
        return JsonResponse({"signals": [_serialize_signal(s) for s in qs[:100]]})

    # POST
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "title is required"}, status=400)

    source_type = data.get("source_type", "other")
    if source_type not in Signal.SourceType.values:
        return JsonResponse(
            {"error": f"Invalid source_type. Valid: {list(Signal.SourceType.values)}"},
            status=400,
        )

    signal = Signal.objects.create(
        title=title,
        description=data.get("description", ""),
        source_type=source_type,
        severity=data.get("severity", "warning"),
        created_by=request.user,
    )

    logger.info("signal.created", extra={"signal_id": str(signal.id)})
    return JsonResponse({"signal": _serialize_signal(signal)}, status=201)


@gated_paid
@require_http_methods(["GET", "POST"])
def signal_detail(request, signal_id):
    """
    GET  — Signal detail.
    POST — Triage action: acknowledge, link_investigation, resolve, dismiss.
    """
    try:
        signal = Signal.objects.get(id=signal_id, created_by=request.user)
    except Signal.DoesNotExist:
        return JsonResponse({"error": "Signal not found"}, status=404)

    if request.method == "GET":
        return JsonResponse({"signal": _serialize_signal(signal)})

    # POST — triage action
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    action = data.get("action")

    if action == "acknowledge":
        signal.acknowledge()

    elif action == "link_investigation":
        from core.models import Investigation

        inv_id = data.get("investigation_id")
        if not inv_id:
            return JsonResponse({"error": "investigation_id required"}, status=400)
        try:
            inv = Investigation.objects.get(id=inv_id)
        except Investigation.DoesNotExist:
            return JsonResponse({"error": "Investigation not found"}, status=404)
        signal.link_investigation(inv)

    elif action == "resolve":
        signal.resolve()

    elif action == "dismiss":
        reason = data.get("reason", "").strip()
        try:
            signal.dismiss(reason)
        except ValueError as e:
            return JsonResponse({"error": str(e)}, status=400)

    else:
        return JsonResponse(
            {"error": f"Unknown action '{action}'. Valid: acknowledge, link_investigation, resolve, dismiss"},
            status=400,
        )

    logger.info("signal.triaged", extra={"signal_id": str(signal.id), "action": action})
    return JsonResponse({"signal": _serialize_signal(signal)})


# =============================================================================
# COMMITMENTS (LOOP-001 §3.3)
# =============================================================================


@gated_paid
@require_http_methods(["GET", "POST"])
def commitment_list_create(request):
    """
    GET  — List commitments (filterable by status, owner).
    POST — Create a commitment.
    """
    if request.method == "GET":
        qs = Commitment.objects.filter(owner=request.user)
        status = request.GET.get("status")
        if status:
            qs = qs.filter(status=status)
        return JsonResponse({"commitments": [_serialize_commitment(c) for c in qs[:100]]})

    # POST
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "title is required"}, status=400)

    due_date_str = data.get("due_date")
    if not due_date_str:
        return JsonResponse(
            {"error": "due_date is required — a commitment without a date is a wish"},
            status=400,
        )
    try:
        due = date.fromisoformat(due_date_str)
    except ValueError:
        return JsonResponse({"error": "due_date must be YYYY-MM-DD format"}, status=400)

    # Resolve owner — default to request.user
    owner = request.user
    owner_id = data.get("owner_id")
    if owner_id:
        from django.contrib.auth import get_user_model

        User = get_user_model()
        try:
            owner = User.objects.get(id=owner_id)
        except User.DoesNotExist:
            return JsonResponse({"error": "Owner user not found"}, status=404)

    commitment = Commitment.objects.create(
        title=title,
        description=data.get("description", ""),
        owner=owner,
        due_date=due,
        preconditions=data.get("preconditions", []),
        transition_type=data.get("transition_type", ""),
        source_type=data.get("source_type", "manual"),
        source_investigation_id=data.get("source_investigation_id"),
        created_by=request.user,
    )

    logger.info("commitment.created", extra={"commitment_id": str(commitment.id)})
    return JsonResponse({"commitment": _serialize_commitment(commitment)}, status=201)


@gated_paid
@require_http_methods(["GET", "POST"])
def commitment_detail(request, commitment_id):
    """
    GET  — Commitment detail.
    POST — Status action: start, fulfill, break, cancel.
    """
    try:
        commitment = Commitment.objects.get(id=commitment_id)
    except Commitment.DoesNotExist:
        return JsonResponse({"error": "Commitment not found"}, status=404)

    if request.method == "GET":
        return JsonResponse({"commitment": _serialize_commitment(commitment)})

    # POST — status action
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    action = data.get("action")

    if action == "start":
        commitment.status = Commitment.Status.IN_PROGRESS
        commitment.save(update_fields=["status", "updated_at"])

    elif action == "fulfill":
        result = fulfill_commitment(commitment, request.user)
        logger.info("commitment.fulfilled", extra={"commitment_id": str(commitment.id)})
        return JsonResponse({"commitment": _serialize_commitment(commitment), "result": result})

    elif action == "break":
        reason = data.get("reason", "")
        commitment.mark_broken(reason)

    elif action == "cancel":
        commitment.status = Commitment.Status.CANCELLED
        commitment.save(update_fields=["status", "updated_at"])

    else:
        return JsonResponse(
            {"error": f"Unknown action '{action}'. Valid: start, fulfill, break, cancel"},
            status=400,
        )

    logger.info(
        "commitment.action",
        extra={"commitment_id": str(commitment.id), "action": action},
    )
    return JsonResponse({"commitment": _serialize_commitment(commitment)})


# =============================================================================
# MODE TRANSITIONS (LOOP-001 §3.2) — Read-only
# =============================================================================


@gated_paid
@require_http_methods(["GET"])
def transition_list(request):
    """List mode transitions — immutable audit trail."""
    qs = ModeTransition.objects.all()

    investigation_id = request.GET.get("investigation_id")
    if investigation_id:
        from django.contrib.contenttypes.models import ContentType

        from core.models import Investigation

        ct = ContentType.objects.get_for_model(Investigation)
        qs = qs.filter(source_content_type=ct, source_object_id=investigation_id)

    return JsonResponse({"transitions": [_serialize_transition(t) for t in qs[:100]]})


# =============================================================================
# INVESTIGATION → COMMITMENT BRIDGE
# =============================================================================


@gated_paid
@require_http_methods(["GET", "POST"])
def investigation_commitments(request, investigation_id):
    """
    GET  — List commitments for an investigation.
    POST — Create a commitment linked to this investigation.

    This is the bridge: when an investigation concludes, the
    investigator creates commitments that encode what happens next.
    """
    from core.models import Investigation

    try:
        inv = Investigation.objects.get(id=investigation_id)
    except Investigation.DoesNotExist:
        return JsonResponse({"error": "Investigation not found"}, status=404)

    if request.method == "GET":
        commitments = Commitment.objects.filter(source_investigation=inv)
        return JsonResponse({"commitments": [_serialize_commitment(c) for c in commitments]})

    # POST — create commitment for this investigation
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "title is required"}, status=400)

    due_date_str = data.get("due_date")
    if not due_date_str:
        return JsonResponse({"error": "due_date is required"}, status=400)
    try:
        due = date.fromisoformat(due_date_str)
    except ValueError:
        return JsonResponse({"error": "due_date must be YYYY-MM-DD format"}, status=400)

    owner = request.user
    owner_id = data.get("owner_id")
    if owner_id:
        from django.contrib.auth import get_user_model

        User = get_user_model()
        try:
            owner = User.objects.get(id=owner_id)
        except User.DoesNotExist:
            return JsonResponse({"error": "Owner user not found"}, status=404)

    commitment = Commitment.objects.create(
        title=title,
        description=data.get("description", ""),
        owner=owner,
        due_date=due,
        preconditions=data.get("preconditions", []),
        transition_type=data.get("transition_type", ""),
        source_type=Commitment.SourceType.INVESTIGATION,
        source_investigation=inv,
        tenant=inv.tenant,
        created_by=request.user,
    )

    logger.info(
        "commitment.created_from_investigation",
        extra={
            "commitment_id": str(commitment.id),
            "investigation_id": str(inv.id),
            "transition_type": commitment.transition_type,
        },
    )
    return JsonResponse({"commitment": _serialize_commitment(commitment)}, status=201)


# =============================================================================
# SERIALIZERS — Verify mode
# =============================================================================


def _serialize_pc(pc):
    return {
        "id": str(pc.id),
        "controlled_document_id": str(pc.controlled_document_id) if pc.controlled_document_id else None,
        "document_version": pc.document_version,
        "operator_id": str(pc.operator_id) if pc.operator_id else None,
        "observer_id": str(pc.observer_id),
        "process_area": pc.process_area,
        "diagnosis": pc.diagnosis,
        "pass_rate": pc.pass_rate,
        "comfort_level": pc.comfort_level,
        "close_loop_method": pc.close_loop_method,
        "created_at": pc.created_at.isoformat() if pc.created_at else None,
        "items": [
            {
                "id": str(item.id),
                "step_text": item.step_text,
                "key_point": item.key_point,
                "followed": item.followed,
                "followed_na": item.followed_na,
                "outcome_pass": item.outcome_pass,
                "outcome_na": item.outcome_na,
                "deviation_severity": item.deviation_severity,
                "notes": item.notes,
            }
            for item in pc.observation_items.all()
        ],
    }


def _serialize_fft(fft):
    return {
        "id": str(fft.id),
        "test_mode": fft.test_mode,
        "fmea_row_id": str(fft.fmea_row_id) if fft.fmea_row_id else None,
        "test_plan": fft.test_plan,
        "control_being_tested": fft.control_being_tested,
        "safety_reviewed": fft.safety_reviewed,
        "result": fft.result,
        "detection_count": fft.detection_count,
        "injection_count": fft.injection_count,
        "detection_rate": fft.detection_rate,
        "conducted_by_id": str(fft.conducted_by_id),
        "conducted_at": fft.conducted_at.isoformat() if fft.conducted_at else None,
        "created_at": fft.created_at.isoformat() if fft.created_at else None,
    }


def _serialize_reflection(r):
    return {
        "id": str(r.id),
        "training_record_id": str(r.training_record_id),
        "controlled_document_id": str(r.controlled_document_id),
        "document_version": r.document_version,
        "reflection_text": r.reflection_text,
        "self_assessed_level": r.self_assessed_level,
        "flagged_section_ids": [str(s.id) for s in r.flagged_sections.all()],
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


# =============================================================================
# PROCESS CONFIRMATIONS (LOOP-001 §7.1)
# =============================================================================


@gated_paid
@require_http_methods(["GET", "POST"])
def pc_list_create(request):
    """
    GET  — List PCs (filterable by document, operator).
    POST — Create a PC with observation items.
    """
    if request.method == "GET":
        qs = ProcessConfirmation.objects.filter(observer=request.user).select_related("controlled_document", "operator")
        doc_id = request.GET.get("document_id")
        if doc_id:
            qs = qs.filter(controlled_document_id=doc_id)
        return JsonResponse({"process_confirmations": [_serialize_pc(pc) for pc in qs[:50]]})

    # POST
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    pc = ProcessConfirmation.objects.create(
        tenant=getattr(request.user, "active_tenant", None),
        controlled_document_id=data.get("controlled_document_id"),
        document_version=data.get("document_version", ""),
        operator_id=data.get("operator_id"),
        observer=request.user,
        process_area=data.get("process_area", ""),
        shift=data.get("shift", ""),
    )

    # Create observation items
    for item_data in data.get("items", []):
        PCObservationItem.objects.create(
            process_confirmation=pc,
            sort_order=item_data.get("sort_order", 0),
            step_text=item_data.get("step_text", ""),
            key_point=item_data.get("key_point", ""),
            reason_why=item_data.get("reason_why", ""),
            linked_section_id=item_data.get("linked_section_id"),
            followed=item_data.get("followed", True),
            followed_na=item_data.get("followed_na", False),
            outcome_pass=item_data.get("outcome_pass", True),
            outcome_na=item_data.get("outcome_na", False),
            deviation_severity=item_data.get("deviation_severity", ""),
            notes=item_data.get("notes", ""),
        )

    # Compute diagnosis from items
    pc.compute_diagnosis()
    pc.save(update_fields=["diagnosis"])

    logger.info("pc.created", extra={"pc_id": str(pc.id), "diagnosis": pc.diagnosis})
    return JsonResponse({"process_confirmation": _serialize_pc(pc)}, status=201)


@gated_paid
@require_http_methods(["GET", "POST"])
def pc_detail(request, pc_id):
    """
    GET  — PC detail with items.
    POST — Update PC (add interaction, close loop, recompute diagnosis).
    """
    try:
        pc = ProcessConfirmation.objects.prefetch_related("observation_items").get(id=pc_id)
    except ProcessConfirmation.DoesNotExist:
        return JsonResponse({"error": "Process confirmation not found"}, status=404)

    if request.method == "GET":
        return JsonResponse({"process_confirmation": _serialize_pc(pc)})

    # POST — update
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    for field in (
        "operator_id",
        "process_area",
        "shift",
        "observer_notes",
        "improvements_observed",
        "operator_interaction",
        "comfort_level",
        "close_loop_method",
        "close_loop_notes",
    ):
        if field in data:
            setattr(pc, field, data[field])

    pc.compute_diagnosis()
    pc.save()

    logger.info("pc.updated", extra={"pc_id": str(pc.id), "diagnosis": pc.diagnosis})
    return JsonResponse({"process_confirmation": _serialize_pc(pc)})


# =============================================================================
# FORCED FAILURE TESTS (LOOP-001 §7.2)
# =============================================================================


@gated_paid
@require_http_methods(["GET", "POST"])
def fft_list_create(request):
    """
    GET  — List FFTs.
    POST — Create an FFT plan (safety review required before recording results).
    """
    if request.method == "GET":
        qs = ForcedFailureTest.objects.filter(conducted_by=request.user)
        return JsonResponse({"forced_failure_tests": [_serialize_fft(f) for f in qs[:50]]})

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    test_plan = data.get("test_plan", "").strip()
    if not test_plan:
        return JsonResponse({"error": "test_plan is required"}, status=400)

    fft = ForcedFailureTest.objects.create(
        tenant=getattr(request.user, "active_tenant", None),
        test_mode=data.get("test_mode", "hypothesis_driven"),
        fmea_row_id=data.get("fmea_row_id"),
        test_plan=test_plan,
        control_being_tested=data.get("control_being_tested", ""),
        conducted_by=request.user,
    )

    logger.info("fft.created", extra={"fft_id": str(fft.id)})
    return JsonResponse({"forced_failure_test": _serialize_fft(fft)}, status=201)


@gated_paid
@require_http_methods(["GET", "POST"])
def fft_detail(request, fft_id):
    """
    GET  — FFT detail.
    POST — Actions: safety_review, record_results.
    """
    try:
        fft = ForcedFailureTest.objects.get(id=fft_id)
    except ForcedFailureTest.DoesNotExist:
        return JsonResponse({"error": "Forced failure test not found"}, status=404)

    if request.method == "GET":
        return JsonResponse({"forced_failure_test": _serialize_fft(fft)})

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    action = data.get("action")

    if action == "safety_review":
        fft.safety_reviewed = True
        fft.safety_reviewer = request.user
        fft.save(update_fields=["safety_reviewed", "safety_reviewer", "updated_at"])

    elif action == "record_results":
        if not fft.safety_reviewed:
            return JsonResponse(
                {"error": "Safety review required before recording results"},
                status=400,
            )
        fft.detection_count = data.get("detection_count", 0)
        fft.injection_count = data.get("injection_count", 0)
        fft.evidence_notes = data.get("evidence_notes", "")
        from django.utils import timezone

        fft.conducted_at = timezone.now()
        try:
            fft.save()
        except ValueError as e:
            return JsonResponse({"error": str(e)}, status=400)

    else:
        return JsonResponse(
            {"error": f"Unknown action '{action}'. Valid: safety_review, record_results"},
            status=400,
        )

    logger.info("fft.action", extra={"fft_id": str(fft.id), "action": action})
    return JsonResponse({"forced_failure_test": _serialize_fft(fft)})


# =============================================================================
# TRAINING REFLECTIONS (LOOP-001 §6.2)
# =============================================================================


@gated_paid
@require_http_methods(["GET", "POST"])
def reflection_list_create(request):
    """
    GET  — List reflections (filterable by document).
    POST — Create a reflection for a training record.
    """
    if request.method == "GET":
        qs = TrainingReflection.objects.all()
        doc_id = request.GET.get("document_id")
        if doc_id:
            qs = qs.filter(controlled_document_id=doc_id)
        return JsonResponse({"reflections": [_serialize_reflection(r) for r in qs[:50]]})

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    training_record_id = data.get("training_record_id")
    if not training_record_id:
        return JsonResponse({"error": "training_record_id is required"}, status=400)

    reflection_text = data.get("reflection_text", "").strip()
    if not reflection_text:
        return JsonResponse({"error": "reflection_text is required"}, status=400)

    controlled_document_id = data.get("controlled_document_id")
    if not controlled_document_id:
        return JsonResponse({"error": "controlled_document_id is required"}, status=400)

    reflection = TrainingReflection.objects.create(
        training_record_id=training_record_id,
        controlled_document_id=controlled_document_id,
        document_version=data.get("document_version", ""),
        reflection_text=reflection_text,
        self_assessed_level=data.get("self_assessed_level", 0),
    )

    flagged_ids = data.get("flagged_section_ids", [])
    if flagged_ids:
        from agents_api.models import ISOSection

        sections = ISOSection.objects.filter(id__in=flagged_ids)
        reflection.flagged_sections.set(sections)

    logger.info("reflection.created", extra={"reflection_id": str(reflection.id)})
    return JsonResponse({"reflection": _serialize_reflection(reflection)}, status=201)

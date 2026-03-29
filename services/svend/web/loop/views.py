"""Loop API views — LOOP-001 §3, §4, §11.

Endpoints for Signals, Commitments, ModeTransitions, QMS Policy, and Auditor Portal.
"""

import json
import logging
from datetime import date, timedelta

from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid

from .models import (
    FMIS,
    AuditorPortalToken,
    Commitment,
    FMISRow,
    ForcedFailureTest,
    InvestigationEntry,
    ModeTransition,
    PCObservationItem,
    PolicyCondition,
    ProcessConfirmation,
    QMSPolicy,
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

    elif action == "open_investigation":
        from core.models import Investigation, InvestigationMembership

        inv_title = data.get("investigation_title", signal.title)
        inv = Investigation.objects.create(
            title=inv_title,
            description=f"Opened from signal: {signal.title}\n\n{signal.description}",
            owner=request.user,
        )
        InvestigationMembership.objects.create(
            investigation=inv,
            user=request.user,
            role=Investigation.MemberRole.OWNER,
        )
        signal.link_investigation(inv)
        logger.info(
            "signal.opened_investigation",
            extra={"signal_id": str(signal.id), "investigation_id": str(inv.id)},
        )
        return JsonResponse(
            {
                "signal": _serialize_signal(signal),
                "investigation": {
                    "id": str(inv.id),
                    "title": inv.title,
                    "status": inv.status,
                },
            }
        )

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
            {
                "error": f"Unknown action '{action}'. "
                "Valid: acknowledge, open_investigation, link_investigation, resolve, dismiss"
            },
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

    # Notify owner if assigned to someone else
    if owner != request.user:
        from notifications.helpers import notify

        notify(
            recipient=owner,
            notification_type="assignment",
            title=f"New commitment: {commitment.title}",
            message=f"You have been assigned a commitment due {commitment.due_date}.",
            entity_type="commitment",
            entity_id=commitment.id,
        )

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

    elif action == "reassign":
        new_owner_id = data.get("owner_id")
        if not new_owner_id:
            return JsonResponse({"error": "owner_id is required for reassign"}, status=400)
        from django.contrib.auth import get_user_model

        User = get_user_model()
        try:
            new_owner = User.objects.get(id=new_owner_id)
        except User.DoesNotExist:
            return JsonResponse({"error": "Owner not found"}, status=404)
        old_owner = commitment.owner
        commitment.owner = new_owner
        commitment.save(update_fields=["owner", "updated_at"])
        # Notify the new owner
        from notifications.helpers import notify

        notify(
            recipient=new_owner,
            notification_type="assignment",
            title=f"Commitment assigned: {commitment.title}",
            message=f"You have been assigned a commitment due {commitment.due_date}.",
            entity_type="commitment",
            entity_id=commitment.id,
        )
        logger.info(
            "commitment.reassigned",
            extra={"commitment_id": str(commitment.id), "from": str(old_owner.id), "to": str(new_owner.id)},
        )

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

    # Emit for PolicyEvaluator (LOOP-001 §4.6.2)
    from agents_api.tool_events import tool_events

    tool_events.emit("pc.completed", pc, user=request.user)

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

        # Emit for PolicyEvaluator (LOOP-001 §4.6.2)
        from agents_api.tool_events import tool_events

        tool_events.emit("forced_failure.completed", fft, user=request.user)

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


# =============================================================================
# ACCOUNTABILITY DASHBOARD (LOOP-001 §16.2)
# =============================================================================


@gated_paid
@require_http_methods(["GET"])
def dashboard_data(request):
    """Dashboard data endpoint — aggregates loop state for the daily surface.

    Returns: commitments (my, overdue, blocked), conditions (active),
    signals (open), investigations (active), recent transitions.
    """
    from core.models import Investigation

    user = request.user

    # My commitments
    my_commitments = (
        Commitment.objects.filter(owner=user)
        .exclude(status__in=[Commitment.Status.FULFILLED, Commitment.Status.CANCELLED])
        .order_by("due_date")[:20]
    )

    overdue_commitments = [c for c in my_commitments if c.is_overdue]
    blocked_commitments = [c for c in my_commitments if c.is_blocked]
    due_today = [c for c in my_commitments if c.due_date == date.today()]

    # Active conditions
    active_conditions_qs = PolicyCondition.objects.filter(
        status=PolicyCondition.Status.ACTIVE,
    ).order_by("-created_at")
    conditions_count = active_conditions_qs.count()
    active_conditions = list(active_conditions_qs[:20])

    # Open signals
    open_signals_qs = Signal.objects.filter(
        created_by=user,
        triage_state__in=[Signal.TriageState.UNTRIAGED, Signal.TriageState.ACKNOWLEDGED],
    ).order_by("-created_at")
    untriaged_count = open_signals_qs.filter(triage_state=Signal.TriageState.UNTRIAGED).count()
    open_signals = list(open_signals_qs[:10])

    # Active investigations
    active_investigations = Investigation.objects.filter(
        status__in=[Investigation.Status.OPEN, Investigation.Status.ACTIVE],
        owner=user,
    ).order_by("-updated_at")[:10]

    # Recent transitions (audit trail)
    recent_transitions = ModeTransition.objects.all().order_by("-created_at")[:10]

    # Summary counts
    total_commitments = Commitment.objects.filter(owner=user).exclude(status__in=[Commitment.Status.CANCELLED]).count()
    fulfilled_commitments = Commitment.objects.filter(owner=user, status=Commitment.Status.FULFILLED).count()

    return JsonResponse(
        {
            "commitments": {
                "items": [_serialize_commitment(c) for c in my_commitments],
                "overdue_count": len(overdue_commitments),
                "blocked_count": len(blocked_commitments),
                "due_today_count": len(due_today),
                "total": total_commitments,
                "fulfilled": fulfilled_commitments,
            },
            "conditions": {
                "items": [
                    {
                        "id": str(c.id),
                        "condition_type": c.condition_type,
                        "severity": c.severity,
                        "title": c.title,
                        "created_at": c.created_at.isoformat() if c.created_at else None,
                    }
                    for c in active_conditions
                ],
                "count": conditions_count,
            },
            "signals": {
                "items": [_serialize_signal(s) for s in open_signals],
                "untriaged_count": untriaged_count,
            },
            "investigations": {
                "items": [
                    {
                        "id": str(inv.id),
                        "title": inv.title,
                        "status": inv.status,
                        "updated_at": inv.updated_at.isoformat() if inv.updated_at else None,
                        "commitment_count": Commitment.objects.filter(source_investigation=inv)
                        .exclude(status=Commitment.Status.CANCELLED)
                        .count(),
                    }
                    for inv in active_investigations
                ],
            },
            "transitions": {
                "items": [_serialize_transition(t) for t in recent_transitions],
            },
        }
    )


# =============================================================================
# REPORT GENERATION (LOOP-001 §5.2)
# =============================================================================


@gated_paid
@require_http_methods(["GET", "POST"])
def generate_report(request, investigation_id):
    """
    GET  — Preview report with completeness scoring.
    POST — Generate and return full report.

    Query params / body:
        template: iso_9001_capa (default), iatf_8d
    """
    from core.models import Investigation

    from .report_engine import TEMPLATES, assemble_report

    try:
        inv = Investigation.objects.get(id=investigation_id)
    except Investigation.DoesNotExist:
        return JsonResponse({"error": "Investigation not found"}, status=404)

    if request.method == "GET":
        # Return available templates + current completeness for each
        results = {}
        for tid in TEMPLATES:
            try:
                report = assemble_report(inv, tid)
                results[tid] = {
                    "name": report["template"]["name"],
                    "standard": report["template"]["standard"],
                    "completeness": report["completeness"],
                    "sections": [
                        {
                            "key": s["key"],
                            "title": s["title"],
                            "populated": s["populated"],
                            "required": s["required"],
                        }
                        for s in report["sections"]
                    ],
                }
            except Exception as e:
                results[tid] = {"error": str(e)}

        return JsonResponse({"templates": results})

    # POST — generate full report
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        data = {}

    template_id = data.get("template", request.GET.get("template", "iso_9001_capa"))

    try:
        report = assemble_report(inv, template_id)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    logger.info(
        "report.generated",
        extra={
            "investigation_id": str(inv.id),
            "template": template_id,
            "completeness": report["completeness"],
        },
    )

    return JsonResponse({"report": report})


# =============================================================================
# INVESTIGATION ENTRIES (LOOP-001 §16.3)
# =============================================================================


def _serialize_entry(e):
    return {
        "id": str(e.id),
        "entry_type": e.entry_type,
        "title": e.title,
        "content": e.content,
        "structured_data": e.structured_data,
        "tool_link_id": str(e.tool_link_id) if e.tool_link_id else None,
        "author_id": str(e.author_id),
        "created_at": e.created_at.isoformat() if e.created_at else None,
    }


@gated_paid
@require_http_methods(["GET", "POST"])
def investigation_entries(request, investigation_id):
    """
    GET  — List entries for an investigation (chronological).
    POST — Create an entry (narrative, tool output, evidence, data).
    """
    from core.models import Investigation

    try:
        inv = Investigation.objects.get(id=investigation_id)
    except Investigation.DoesNotExist:
        return JsonResponse({"error": "Investigation not found"}, status=404)

    if request.method == "GET":
        entries = InvestigationEntry.objects.filter(investigation=inv)
        return JsonResponse({"entries": [_serialize_entry(e) for e in entries]})

    # POST — create entry
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    entry_type = data.get("entry_type", "narrative")
    content = data.get("content", "").strip()

    if entry_type == "narrative" and not content:
        return JsonResponse({"error": "Content is required for narrative entries"}, status=400)

    entry = InvestigationEntry.objects.create(
        investigation=inv,
        entry_type=entry_type,
        title=data.get("title", ""),
        content=content,
        structured_data=data.get("structured_data", {}),
        author=request.user,
    )

    logger.info(
        "entry.created",
        extra={
            "entry_id": str(entry.id),
            "investigation_id": str(inv.id),
            "type": entry_type,
        },
    )
    return JsonResponse({"entry": _serialize_entry(entry)}, status=201)


# =============================================================================
# FMIS (LOOP-001 §8, §16.5)
# =============================================================================


def _serialize_fmis(f):
    return {
        "id": str(f.id),
        "title": f.title,
        "description": f.description,
        "methodology": f.methodology,
        "investigation_id": str(f.investigation_id) if f.investigation_id else None,
        "row_count": f.rows.count(),
        "created_at": f.created_at.isoformat() if f.created_at else None,
    }


def _serialize_fmis_row(r):
    from .bayesian import (
        SEVERITY_CATEGORIES,
        beta_credible_interval,
        beta_mean,
        dirichlet_mean,
    )

    sev_dist = None
    if r.severity_alpha:
        probs = dirichlet_mean(r.severity_alpha)
        sev_dist = dict(zip(SEVERITY_CATEGORIES, [round(p, 3) for p in probs]))

    det_ci = beta_credible_interval(r.detection_alpha, r.detection_beta)
    occ_ci = beta_credible_interval(r.occurrence_alpha, r.occurrence_beta)
    det_rate = beta_mean(r.detection_alpha, r.detection_beta)
    occ_rate = beta_mean(r.occurrence_alpha, r.occurrence_beta)

    return {
        "id": str(r.id),
        "failure_mode_text": r.failure_mode_text,
        "effect_text": r.effect_text,
        "cause_text": r.cause_text,
        "prevention_control": r.prevention_control,
        "detection_control": r.detection_control,
        # Scores (active method)
        "severity_score": r.severity_score,
        "occurrence_score": r.occurrence_score,
        "detection_score": r.detection_score,
        "rpn": r.rpn,
        # Severity posterior
        "severity_method": r.severity_method,
        "severity_manual": r.severity_manual,
        "severity_distribution": sev_dist,
        "severity_alpha": r.severity_alpha,
        # Occurrence posterior
        "occurrence_method": r.occurrence_method,
        "occurrence_manual": r.occurrence_manual,
        "occurrence_rate": round(occ_rate, 4),
        "occurrence_ci": list(occ_ci),
        "occurrence_failures": r.occurrence_failures,
        "occurrence_units": r.occurrence_units,
        # Detection posterior
        "detection_method": r.detection_method,
        "detection_manual": r.detection_manual,
        "detection_rate": round(det_rate, 4),
        "detection_ci": list(det_ci),
        "detection_detected": r.detection_detected,
        "detection_injected": r.detection_injected,
        # Metadata
        "has_operational_definitions": r.has_operational_definitions,
        "undefined_terms": r.undefined_terms,
        "last_evidence_date": r.last_evidence_date.isoformat() if r.last_evidence_date else None,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        # Parent context
        "fmis_id": str(r.fmis_id),
        "fmis_title": r.fmis.title if hasattr(r, "fmis") and r.fmis else None,
        "investigation_id": str(r.investigation_id) if r.investigation_id else None,
        "investigation_title": r.investigation.title if hasattr(r, "investigation") and r.investigation else None,
    }


@gated_paid
@require_http_methods(["GET"])
def fmis_global(request):
    """Global FMIS view — all failure modes across the org.

    Returns all FMIS rows with their parent FMIS document info,
    sorted by RPN descending. This is the org's risk landscape.
    """
    rows = FMISRow.objects.select_related("fmis", "investigation").order_by("-rpn")

    # Get or create a default FMIS for the org
    fmis_docs = FMIS.objects.filter(created_by=request.user)

    return JsonResponse(
        {
            "rows": [_serialize_fmis_row(r) for r in rows[:200]],
            "fmis_documents": [_serialize_fmis(f) for f in fmis_docs],
            "total_rows": rows.count(),
            "gaps": rows.filter(
                failure_mode_text="",
            ).count(),
        }
    )


@gated_paid
@require_http_methods(["GET", "POST"])
def fmis_list_create(request):
    """
    GET  — List FMIS documents.
    POST — Create FMIS.
    """
    if request.method == "GET":
        qs = FMIS.objects.filter(created_by=request.user)
        return JsonResponse({"fmis_documents": [_serialize_fmis(f) for f in qs[:50]]})

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "title is required"}, status=400)

    fmis = FMIS.objects.create(
        title=title,
        description=data.get("description", ""),
        methodology=data.get("methodology", "svend_bayesian"),
        investigation_id=data.get("investigation_id"),
        created_by=request.user,
    )
    return JsonResponse({"fmis": _serialize_fmis(fmis)}, status=201)


@gated_paid
@require_http_methods(["GET"])
def fmis_detail(request, fmis_id):
    """FMIS detail with all rows."""
    try:
        fmis = FMIS.objects.get(id=fmis_id)
    except FMIS.DoesNotExist:
        return JsonResponse({"error": "FMIS not found"}, status=404)

    rows = FMISRow.objects.filter(fmis=fmis)
    return JsonResponse(
        {
            "fmis": _serialize_fmis(fmis),
            "rows": [_serialize_fmis_row(r) for r in rows],
        }
    )


@gated_paid
@require_http_methods(["POST"])
def fmis_add_row(request, fmis_id):
    """Add a row to FMIS."""
    try:
        fmis = FMIS.objects.get(id=fmis_id)
    except FMIS.DoesNotExist:
        return JsonResponse({"error": "FMIS not found"}, status=404)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    fm = data.get("failure_mode_text", "").strip()
    if not fm:
        return JsonResponse({"error": "failure_mode_text is required"}, status=400)

    row = FMISRow.objects.create(
        fmis=fmis,
        investigation_id=fmis.investigation_id,
        failure_mode_text=fm,
        effect_text=data.get("effect_text", ""),
        cause_text=data.get("cause_text", ""),
        prevention_control=data.get("prevention_control", ""),
        detection_control=data.get("detection_control", ""),
        severity_manual=data.get("severity_manual"),
        severity_method=data.get("severity_method", "manual"),
        occurrence_manual=data.get("occurrence_manual"),
        occurrence_method=data.get("occurrence_method", "manual"),
        detection_manual=data.get("detection_manual"),
        detection_method=data.get("detection_method", "manual"),
    )
    return JsonResponse({"row": _serialize_fmis_row(row)}, status=201)


@gated_paid
@require_http_methods(["POST"])
def fmis_row_update_posterior(request, fmis_id, row_id):
    """Update a row's posterior from evidence.

    POST body:
      {"type": "detection", "detected": 6, "injected": 8}
      {"type": "occurrence", "failures": 3, "units": 500}
      {"type": "severity", "category": "moderate"}
    """
    try:
        row = FMISRow.objects.get(id=row_id, fmis_id=fmis_id)
    except FMISRow.DoesNotExist:
        return JsonResponse({"error": "Row not found"}, status=404)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    update_type = data.get("type")

    if update_type == "detection":
        detected = data.get("detected", 0)
        injected = data.get("injected", 0)
        if injected <= 0:
            return JsonResponse({"error": "injected must be > 0"}, status=400)
        from .bayesian import beta_update

        row.detection_detected += detected
        row.detection_injected += injected
        row.detection_alpha, row.detection_beta = beta_update(
            row.detection_alpha, row.detection_beta, detected, injected
        )
        row.detection_method = "bayesian"

    elif update_type == "occurrence":
        failures = data.get("failures", 0)
        units = data.get("units", 0)
        if units <= 0:
            return JsonResponse({"error": "units must be > 0"}, status=400)
        from .bayesian import beta_update

        row.occurrence_failures += failures
        row.occurrence_units += units
        row.occurrence_alpha, row.occurrence_beta = beta_update(
            row.occurrence_alpha, row.occurrence_beta, failures, units
        )
        row.occurrence_method = "bayesian"

    elif update_type == "severity":
        category = data.get("category", "")
        from .bayesian import SEVERITY_VALUES, dirichlet_update_by_name

        if category not in SEVERITY_VALUES:
            return JsonResponse(
                {"error": f"Invalid category. Valid: {list(SEVERITY_VALUES.keys())}"},
                status=400,
            )
        if not row.severity_alpha:
            row.severity_alpha = [1, 1, 1, 1, 1]
        row.severity_alpha = dirichlet_update_by_name(row.severity_alpha, category)
        row.severity_method = "bayesian"

    else:
        return JsonResponse(
            {"error": "type must be: detection, occurrence, or severity"},
            status=400,
        )

    from django.utils import timezone

    row.last_evidence_date = timezone.now()
    row.save()

    return JsonResponse({"row": _serialize_fmis_row(row)})


# =============================================================================
# CI READINESS SCORE (LOOP-001 §10)
# =============================================================================


@gated_paid
@require_http_methods(["GET"])
def readiness_score(request):
    """Compute and return the CI Readiness Score.

    Returns: score (0-100), 10 indicators with individual scores,
    weights used, computation timestamp.
    """
    from .readiness import compute_readiness_score

    result = compute_readiness_score(user=request.user)
    return JsonResponse({"readiness": result})


# =============================================================================
# QMS POLICY MANAGEMENT (LOOP-001 §4)
# =============================================================================

# Rule key → parameter schema for each scope.
# The UI uses this to render typed parameter forms.
POLICY_RULE_REGISTRY = {
    "process_confirmation": {
        "pc.retraining_threshold": {
            "label": "Retraining Threshold",
            "description": "PC pass rate below which a retraining condition is surfaced",
            "params": {
                "threshold": {
                    "type": "number",
                    "min": 0,
                    "max": 1,
                    "step": 0.05,
                    "default": 0.80,
                    "label": "Pass rate threshold",
                },
                "trailing_count": {
                    "type": "integer",
                    "min": 1,
                    "max": 50,
                    "default": 5,
                    "label": "Trailing observations",
                },
                "cooldown_days": {"type": "integer", "min": 0, "max": 365, "default": 30, "label": "Cooldown (days)"},
            },
            "linked_standard": "ISO 9001:2015 §7.2",
        },
        "pc.escalation_to_revision": {
            "label": "Escalate to Standard Revision",
            "description": "When N operators fail the same standard, surface revision signal instead of N retraining signals",
            "params": {
                "escalate_after_n_operators": {
                    "type": "integer",
                    "min": 2,
                    "max": 20,
                    "default": 3,
                    "label": "Operator threshold",
                },
                "escalation_window_days": {
                    "type": "integer",
                    "min": 1,
                    "max": 90,
                    "default": 30,
                    "label": "Window (days)",
                },
            },
            "linked_standard": "ISO 9001:2015 §7.5.2",
        },
    },
    "spc_monitoring": {
        "spc.out_of_control_signal": {
            "label": "Out-of-Control Signal",
            "description": "When to surface an OOC condition for investigation",
            "params": {
                "rule_violations": {
                    "type": "integer",
                    "min": 1,
                    "max": 8,
                    "default": 1,
                    "label": "Western Electric rules violated",
                },
                "severity": {
                    "type": "select",
                    "options": ["info", "warning", "critical"],
                    "default": "warning",
                    "label": "Condition severity",
                },
            },
            "linked_standard": "ISO 9001:2015 §8.5.1",
        },
    },
    "forced_failure": {
        "fft.detection_gap_threshold": {
            "label": "Detection Gap Threshold",
            "description": "Detection rate below which a gap condition is surfaced",
            "params": {
                "threshold": {
                    "type": "number",
                    "min": 0,
                    "max": 1,
                    "step": 0.05,
                    "default": 0.70,
                    "label": "Min detection rate",
                },
                "min_tests": {
                    "type": "integer",
                    "min": 1,
                    "max": 50,
                    "default": 3,
                    "label": "Minimum tests before evaluation",
                },
            },
            "linked_standard": "IATF 16949 §8.5.6.1",
        },
    },
    "review_frequency": {
        "review.document_review_months": {
            "label": "Document Review Frequency",
            "description": "Months between mandatory document reviews",
            "params": {
                "months": {"type": "integer", "min": 1, "max": 36, "default": 12, "label": "Review interval (months)"},
                "severity_on_overdue": {
                    "type": "select",
                    "options": ["info", "warning", "critical"],
                    "default": "warning",
                    "label": "Overdue severity",
                },
            },
            "linked_standard": "ISO 9001:2015 §7.5.2",
        },
        "review.fmea_review_months": {
            "label": "FMEA Review Frequency",
            "description": "Months between mandatory FMEA reviews",
            "params": {
                "months": {"type": "integer", "min": 1, "max": 24, "default": 6, "label": "Review interval (months)"},
            },
            "linked_standard": "AIAG FMEA 4th Ed §2.2",
        },
    },
    "training_coverage": {
        "training.coverage_threshold": {
            "label": "Training Coverage Threshold",
            "description": "Minimum percentage of operators trained on each controlled document",
            "params": {
                "threshold": {
                    "type": "number",
                    "min": 0,
                    "max": 1,
                    "step": 0.05,
                    "default": 0.90,
                    "label": "Coverage threshold",
                },
            },
            "linked_standard": "ISO 9001:2015 §7.2",
        },
    },
    "recurrence": {
        "recurrence.detection_window": {
            "label": "Recurrence Detection Window",
            "description": "Time window and count for detecting recurring failure modes",
            "params": {
                "window_days": {"type": "integer", "min": 7, "max": 365, "default": 90, "label": "Window (days)"},
                "min_occurrences": {
                    "type": "integer",
                    "min": 2,
                    "max": 20,
                    "default": 3,
                    "label": "Minimum occurrences",
                },
            },
            "linked_standard": "ISO 9001:2015 §10.2.1",
        },
    },
    "commitment_fulfillment": {
        "commitment.overdue_escalation": {
            "label": "Commitment Overdue Escalation",
            "description": "When to escalate overdue commitments",
            "params": {
                "warning_days": {
                    "type": "integer",
                    "min": 1,
                    "max": 30,
                    "default": 3,
                    "label": "Warning threshold (days overdue)",
                },
                "critical_days": {
                    "type": "integer",
                    "min": 1,
                    "max": 60,
                    "default": 7,
                    "label": "Critical threshold (days overdue)",
                },
            },
            "linked_standard": "",
        },
    },
    "calibration": {
        "calibration.overdue_check": {
            "label": "Calibration Overdue Check",
            "description": "Surface conditions for equipment past calibration due date",
            "params": {
                "warning_days_before": {
                    "type": "integer",
                    "min": 0,
                    "max": 90,
                    "default": 14,
                    "label": "Warn days before due",
                },
                "severity": {
                    "type": "select",
                    "options": ["warning", "critical"],
                    "default": "critical",
                    "label": "Overdue severity",
                },
            },
            "linked_standard": "ISO 9001:2015 §7.1.5.2",
        },
    },
    "verification_schedule": {
        "verification.pc_schedule": {
            "label": "PC Schedule Requirements",
            "description": "Minimum process confirmation frequency per area",
            "params": {
                "min_per_week": {"type": "integer", "min": 0, "max": 50, "default": 3, "label": "Min PCs per week"},
                "min_per_month": {"type": "integer", "min": 0, "max": 200, "default": 12, "label": "Min PCs per month"},
            },
            "linked_standard": "David Mann — Creating a Lean Culture §4",
        },
    },
    "fmis": {
        "fmis.methodology": {
            "label": "FMEA Methodology",
            "description": "Which scoring methodology this org uses",
            "params": {
                "method": {
                    "type": "select",
                    "options": ["aiag_4th", "svend_bayesian", "svend_full"],
                    "default": "svend_bayesian",
                    "label": "Scoring method",
                },
            },
            "linked_standard": "AIAG FMEA 4th Ed",
        },
    },
    "investigation": {
        "investigation.required_tools": {
            "label": "Required Investigation Tools",
            "description": "Tools that must be run before an investigation can be concluded",
            "params": {
                "require_5_why": {"type": "boolean", "default": True, "label": "Require 5-Why analysis"},
                "require_fmea_row": {"type": "boolean", "default": False, "label": "Require FMIS row linkage"},
                "min_entries": {
                    "type": "integer",
                    "min": 1,
                    "max": 20,
                    "default": 3,
                    "label": "Minimum investigation entries",
                },
            },
            "linked_standard": "ISO 9001:2015 §10.2.1",
        },
    },
    "complaint": {
        "complaint.auto_signal": {
            "label": "Customer Complaint Signal Policy",
            "description": "Severity thresholds for complaint-triggered conditions",
            "params": {
                "auto_critical_on_safety": {
                    "type": "boolean",
                    "default": True,
                    "label": "Safety complaints → critical",
                },
                "auto_signal_on_repeat": {"type": "boolean", "default": True, "label": "Repeat complaints → signal"},
                "repeat_window_days": {
                    "type": "integer",
                    "min": 7,
                    "max": 365,
                    "default": 90,
                    "label": "Repeat window (days)",
                },
            },
            "linked_standard": "ISO 9001:2015 §8.2.1",
        },
    },
    "training": {
        "training.hansei_threshold": {
            "label": "Hansei Reflection Threshold",
            "description": "When to trigger document revision from reflection data",
            "params": {
                "threshold": {
                    "type": "number",
                    "min": 0,
                    "max": 1,
                    "step": 0.05,
                    "default": 0.60,
                    "label": "Comprehension threshold",
                },
                "min_reflections": {
                    "type": "integer",
                    "min": 2,
                    "max": 50,
                    "default": 5,
                    "label": "Min reflections before evaluation",
                },
            },
            "linked_standard": "TRN-001 §9.3",
        },
    },
}


def _serialize_policy(p):
    return {
        "id": str(p.id),
        "scope": p.scope,
        "scope_display": p.get_scope_display(),
        "rule_key": p.rule_key,
        "parameters": p.parameters,
        "linked_standard": p.linked_standard,
        "effective_date": p.effective_date.isoformat(),
        "approved_by_id": str(p.approved_by_id),
        "approved_by_name": p.approved_by.get_full_name() or p.approved_by.email if p.approved_by else "",
        "version": p.version,
        "is_active": p.is_active,
        "conditions_count": p.conditions.filter(status=PolicyCondition.Status.ACTIVE).count(),
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }


@gated_paid
@require_http_methods(["GET"])
def policy_registry(request):
    """Return the rule registry — scope→rule_key→parameter schema.

    The UI uses this to render typed parameter forms dynamically.
    """
    return JsonResponse({"registry": POLICY_RULE_REGISTRY})


@gated_paid
@require_http_methods(["GET", "POST"])
def policy_list_create(request):
    """
    GET  — List policies (filterable by scope, is_active).
    POST — Create a new policy rule.
    """
    if request.method == "GET":
        qs = QMSPolicy.objects.select_related("approved_by").all()

        scope = request.GET.get("scope")
        if scope:
            qs = qs.filter(scope=scope)

        active_only = request.GET.get("active")
        if active_only == "true":
            qs = qs.filter(is_active=True)

        return JsonResponse({"policies": [_serialize_policy(p) for p in qs[:200]]})

    # POST — create
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    scope = data.get("scope", "").strip()
    rule_key = data.get("rule_key", "").strip()

    if not scope or scope not in QMSPolicy.Scope.values:
        return JsonResponse({"error": f"Invalid scope. Valid: {list(QMSPolicy.Scope.values)}"}, status=400)
    if not rule_key:
        return JsonResponse({"error": "rule_key is required"}, status=400)

    # Validate rule_key belongs to scope
    scope_rules = POLICY_RULE_REGISTRY.get(scope, {})
    if rule_key not in scope_rules:
        return JsonResponse(
            {"error": f"Unknown rule_key for scope '{scope}'. Valid: {list(scope_rules.keys())}"}, status=400
        )

    # Merge defaults with provided parameters
    rule_def = scope_rules[rule_key]
    defaults = {k: v["default"] for k, v in rule_def["params"].items()}
    params = data.get("parameters", {})
    merged = {**defaults, **params}

    effective_date = data.get("effective_date", date.today().isoformat())

    # Determine version (next version for this rule_key)
    latest_version = (
        QMSPolicy.objects.filter(rule_key=rule_key).order_by("-version").values_list("version", flat=True).first()
    )
    version = (latest_version or 0) + 1

    # Deactivate previous versions
    QMSPolicy.objects.filter(rule_key=rule_key, is_active=True).update(is_active=False)

    policy = QMSPolicy.objects.create(
        scope=scope,
        rule_key=rule_key,
        parameters=merged,
        linked_standard=rule_def.get("linked_standard", data.get("linked_standard", "")),
        effective_date=effective_date,
        approved_by=request.user,
        version=version,
        is_active=True,
    )

    logger.info(
        "policy.created",
        extra={"policy_id": str(policy.id), "rule_key": rule_key, "version": version},
    )
    return JsonResponse({"policy": _serialize_policy(policy)}, status=201)


@gated_paid
@require_http_methods(["GET", "PUT", "DELETE"])
def policy_detail(request, policy_id):
    """
    GET    — Policy detail with version history and active conditions.
    PUT    — Update parameters (creates new version).
    DELETE — Deactivate policy (soft delete).
    """
    try:
        policy = QMSPolicy.objects.select_related("approved_by").get(id=policy_id)
    except QMSPolicy.DoesNotExist:
        return JsonResponse({"error": "Policy not found"}, status=404)

    if request.method == "GET":
        # Include version history
        versions = QMSPolicy.objects.filter(rule_key=policy.rule_key).order_by("-version")
        conditions = PolicyCondition.objects.filter(policy_rule=policy).order_by("-created_at")[:20]

        return JsonResponse(
            {
                "policy": _serialize_policy(policy),
                "versions": [
                    {
                        "id": str(v.id),
                        "version": v.version,
                        "is_active": v.is_active,
                        "effective_date": v.effective_date.isoformat(),
                        "parameters": v.parameters,
                        "created_at": v.created_at.isoformat() if v.created_at else None,
                    }
                    for v in versions
                ],
                "conditions": [
                    {
                        "id": str(c.id),
                        "condition_type": c.condition_type,
                        "severity": c.severity,
                        "title": c.title,
                        "status": c.status,
                        "created_at": c.created_at.isoformat() if c.created_at else None,
                    }
                    for c in conditions
                ],
            }
        )

    if request.method == "PUT":
        try:
            data = json.loads(request.body)
        except (json.JSONDecodeError, ValueError):
            return JsonResponse({"error": "Invalid JSON body"}, status=400)

        new_params = data.get("parameters", {})
        merged = {**policy.parameters, **new_params}
        effective_date = data.get("effective_date", date.today().isoformat())

        # Deactivate current
        policy.is_active = False
        policy.save(update_fields=["is_active", "updated_at"])

        # Create new version
        new_policy = QMSPolicy.objects.create(
            scope=policy.scope,
            rule_key=policy.rule_key,
            parameters=merged,
            linked_standard=data.get("linked_standard", policy.linked_standard),
            effective_date=effective_date,
            approved_by=request.user,
            version=policy.version + 1,
            is_active=True,
        )

        logger.info(
            "policy.updated",
            extra={
                "policy_id": str(new_policy.id),
                "rule_key": policy.rule_key,
                "version": new_policy.version,
                "previous_version": policy.version,
            },
        )
        return JsonResponse({"policy": _serialize_policy(new_policy)})

    # DELETE — soft deactivate
    policy.is_active = False
    policy.save(update_fields=["is_active", "updated_at"])
    logger.info("policy.deactivated", extra={"policy_id": str(policy.id), "rule_key": policy.rule_key})
    return JsonResponse({"status": "deactivated"})


# =============================================================================
# AUDITOR PORTAL (LOOP-001 §11, §16.9)
# =============================================================================

# ISO 9001:2015 clause structure for organizing evidence
ISO_9001_CLAUSES = {
    "4.4": {
        "title": "Quality Management System and its Processes",
        "queries": ["policies", "process_model"],
    },
    "5.2": {
        "title": "Quality Policy",
        "queries": ["policies"],
    },
    "6.1": {
        "title": "Actions to Address Risks and Opportunities",
        "queries": ["fmis_rows"],
    },
    "7.1.5": {
        "title": "Monitoring and Measuring Resources",
        "queries": ["calibration"],
    },
    "7.2": {
        "title": "Competence",
        "queries": ["training"],
    },
    "7.5": {
        "title": "Documented Information",
        "queries": ["documents"],
    },
    "8.2.1": {
        "title": "Customer Communication",
        "queries": ["complaints"],
    },
    "8.5.1": {
        "title": "Control of Production and Service Provision",
        "queries": ["process_confirmations", "spc"],
    },
    "9.1.2": {
        "title": "Customer Satisfaction",
        "queries": ["complaints"],
    },
    "9.1.3": {
        "title": "Analysis and Evaluation",
        "queries": ["readiness"],
    },
    "9.2": {
        "title": "Internal Audit",
        "queries": ["compliance"],
    },
    "10.2": {
        "title": "Nonconformity and Corrective Action",
        "queries": ["investigations", "signals"],
    },
    "10.3": {
        "title": "Continual Improvement",
        "queries": ["readiness", "commitments"],
    },
}


def _resolve_auditor_token(request, token_str):
    """Validate an auditor portal token. Returns (token, error_response)."""
    try:
        tok = AuditorPortalToken.objects.select_related("created_by").get(token=token_str)
    except AuditorPortalToken.DoesNotExist:
        return None, JsonResponse({"error": "Invalid token"}, status=404)

    if not tok.is_valid:
        reason = "revoked" if tok.revoked_at else "expired"
        return None, JsonResponse({"error": f"Token {reason}"}, status=403)

    tok.record_access()
    return tok, None


# ── Token Management (authenticated) ──────────────────────────────────


@gated_paid
@require_http_methods(["GET", "POST"])
def auditor_token_list_create(request):
    """
    GET  — List auditor portal tokens.
    POST — Create a new token.
    """
    if request.method == "GET":
        tokens = AuditorPortalToken.objects.filter(created_by=request.user)
        return JsonResponse(
            {
                "tokens": [
                    {
                        "id": str(t.id),
                        "label": t.label,
                        "token": t.token,
                        "expires_at": t.expires_at.isoformat(),
                        "is_valid": t.is_valid,
                        "access_count": t.access_count,
                        "last_accessed_at": t.last_accessed_at.isoformat() if t.last_accessed_at else None,
                        "created_at": t.created_at.isoformat(),
                        "revoked_at": t.revoked_at.isoformat() if t.revoked_at else None,
                        "portal_url": f"/audit/{t.token}/",
                    }
                    for t in tokens
                ]
            }
        )

    # POST — create token
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    label = data.get("label", "").strip()
    if not label:
        return JsonResponse({"error": "label is required"}, status=400)

    expires_days = data.get("expires_days", 30)
    expires_days = min(max(int(expires_days), 1), 365)

    tok = AuditorPortalToken.objects.create(
        label=label,
        created_by=request.user,
        expires_at=timezone.now() + timedelta(days=expires_days),
    )

    logger.info(
        "auditor_token.created",
        extra={"token_id": str(tok.id), "label": label, "expires_days": expires_days},
    )
    return JsonResponse(
        {
            "token": {
                "id": str(tok.id),
                "label": tok.label,
                "token": tok.token,
                "portal_url": f"/audit/{tok.token}/",
                "expires_at": tok.expires_at.isoformat(),
            }
        },
        status=201,
    )


@gated_paid
@require_http_methods(["DELETE"])
def auditor_token_revoke(request, token_id):
    """Revoke an auditor portal token."""
    try:
        tok = AuditorPortalToken.objects.get(id=token_id, created_by=request.user)
    except AuditorPortalToken.DoesNotExist:
        return JsonResponse({"error": "Token not found"}, status=404)

    tok.revoke()
    logger.info("auditor_token.revoked", extra={"token_id": str(tok.id)})
    return JsonResponse({"status": "revoked"})


# ── Portal Data (token-authenticated, read-only) ──────────────────────


@csrf_exempt
@require_http_methods(["GET"])
def auditor_portal_data(request, token):
    """Auditor portal data API — clause-organized evidence.

    Query params:
        standard: iso_9001 (default)
        clause: specific clause number (optional, returns all if omitted)
    """
    tok, err = _resolve_auditor_token(request, token)
    if err:
        return err

    user = tok.created_by  # Scope data to the token creator's org
    clause_filter = request.GET.get("clause")

    clauses = ISO_9001_CLAUSES
    if clause_filter:
        clauses = {k: v for k, v in clauses.items() if k == clause_filter}

    result = {}
    for clause_num, clause_def in clauses.items():
        clause_data = {
            "title": clause_def["title"],
            "clause": clause_num,
        }

        for query_type in clause_def["queries"]:
            clause_data.update(_query_clause_data(query_type, user))

        result[clause_num] = clause_data

    # Also return summary stats
    from .readiness import compute_readiness_score

    readiness = compute_readiness_score(user=user)

    return JsonResponse(
        {
            "organization": user.get_full_name() or user.email,
            "token_label": tok.label,
            "expires_at": tok.expires_at.isoformat(),
            "readiness": readiness,
            "clauses": result,
        }
    )


def _query_clause_data(query_type, user):
    """Fetch data for a specific query type, scoped to user's org."""
    from core.models import Investigation

    if query_type == "investigations":
        invs = Investigation.objects.filter(owner=user).order_by("-updated_at")[:50]
        concluded = invs.filter(status="concluded")
        active = invs.filter(status__in=["open", "active"])

        inv_list = []
        for inv in invs[:20]:
            commitments = Commitment.objects.filter(source_investigation=inv).exclude(
                status=Commitment.Status.CANCELLED
            )
            transitions = ModeTransition.objects.filter(source_object_id=inv.id).order_by("created_at")

            inv_list.append(
                {
                    "id": str(inv.id),
                    "title": inv.title,
                    "status": inv.status,
                    "created_at": inv.created_at.isoformat() if inv.created_at else None,
                    "concluded_at": inv.concluded_at.isoformat() if inv.concluded_at else None,
                    "commitment_count": commitments.count(),
                    "fulfilled_count": commitments.filter(status=Commitment.Status.FULFILLED).count(),
                    "transitions": [
                        {
                            "type": t.transition_type,
                            "from": t.from_mode,
                            "to": t.to_mode,
                            "date": t.created_at.isoformat(),
                        }
                        for t in transitions[:10]
                    ],
                }
            )

        return {
            "investigations": {
                "total": invs.count(),
                "concluded": concluded.count(),
                "active": active.count(),
                "avg_days_to_conclusion": _avg_conclusion_days(concluded),
                "items": inv_list,
            }
        }

    elif query_type == "signals":
        signals = Signal.objects.filter(created_by=user).order_by("-created_at")[:50]
        return {
            "signals": {
                "total": signals.count(),
                "resolved": signals.filter(triage_state=Signal.TriageState.RESOLVED).count(),
                "dismissed": signals.filter(triage_state=Signal.TriageState.DISMISSED).count(),
                "open": signals.filter(
                    triage_state__in=[Signal.TriageState.UNTRIAGED, Signal.TriageState.ACKNOWLEDGED]
                ).count(),
            }
        }

    elif query_type == "commitments":
        comms = Commitment.objects.filter(owner=user).exclude(status=Commitment.Status.CANCELLED)
        total = comms.count()
        fulfilled = comms.filter(status=Commitment.Status.FULFILLED).count()
        return {
            "commitments": {
                "total": total,
                "fulfilled": fulfilled,
                "fulfillment_rate": round(fulfilled / total, 2) if total > 0 else None,
                "overdue": sum(1 for c in comms if c.is_overdue),
            }
        }

    elif query_type == "documents":
        from agents_api.models import ControlledDocument

        docs = ControlledDocument.objects.filter(created_by=user).order_by("-updated_at")
        return {
            "documents": {
                "total": docs.count(),
                "approved": docs.filter(status="approved").count(),
                "draft": docs.filter(status="draft").count(),
                "review_overdue": docs.filter(review_due_date__lt=date.today(), status="approved").count(),
                "items": [
                    {
                        "id": str(d.id),
                        "title": d.title,
                        "document_number": d.document_number,
                        "status": d.status,
                        "version": d.current_version,
                        "review_due_date": d.review_due_date.isoformat() if d.review_due_date else None,
                        "approved_at": d.approved_at.isoformat() if d.approved_at else None,
                    }
                    for d in docs[:30]
                ],
            }
        }

    elif query_type == "training":
        from agents_api.models import TrainingRecord, TrainingRequirement

        reqs = TrainingRequirement.objects.filter(owner=user)
        records = TrainingRecord.objects.filter(requirement__owner=user)
        total_records = records.count()
        complete = records.filter(status="complete").count()
        return {
            "training": {
                "requirements": reqs.count(),
                "total_records": total_records,
                "complete": complete,
                "coverage": round(complete / total_records, 2) if total_records > 0 else None,
                "avg_competency": _avg_competency(records),
                "expired": records.filter(status="expired").count(),
            }
        }

    elif query_type == "complaints":
        from agents_api.models import CustomerComplaint

        complaints = CustomerComplaint.objects.filter(created_by=user)
        return {
            "complaints": {
                "total": complaints.count(),
                "open": complaints.filter(status__in=["open", "acknowledged", "investigating"]).count(),
                "resolved": complaints.filter(status__in=["resolved", "closed"]).count(),
                "critical": complaints.filter(severity="critical").count(),
            }
        }

    elif query_type == "process_confirmations":
        pcs = ProcessConfirmation.objects.filter(created_by=user)
        total = pcs.count()
        passing = pcs.filter(overall_result="pass").count()
        return {
            "process_confirmations": {
                "total": total,
                "pass_rate": round(passing / total, 2) if total > 0 else None,
                "this_month": pcs.filter(
                    created_at__month=date.today().month,
                    created_at__year=date.today().year,
                ).count(),
            }
        }

    elif query_type == "fmis_rows":
        rows = FMISRow.objects.select_related("fmis").order_by("-rpn")
        return {
            "fmis": {
                "total_failure_modes": rows.count(),
                "high_rpn": rows.filter(rpn__gte=200).count(),
                "medium_rpn": rows.filter(rpn__gte=100, rpn__lt=200).count(),
                "items": [
                    {
                        "failure_mode": r.failure_mode_text,
                        "rpn": r.rpn,
                        "severity": r.severity_score,
                        "occurrence": r.occurrence_score,
                        "detection": r.detection_score,
                        "fmis_title": r.fmis.title if r.fmis else None,
                    }
                    for r in rows[:20]
                ],
            }
        }

    elif query_type == "policies":
        policies = QMSPolicy.objects.filter(is_active=True).order_by("scope")
        return {
            "policies": {
                "total": policies.count(),
                "items": [
                    {
                        "scope": p.get_scope_display(),
                        "rule_key": p.rule_key,
                        "parameters": p.parameters,
                        "linked_standard": p.linked_standard,
                        "effective_date": p.effective_date.isoformat(),
                        "version": p.version,
                    }
                    for p in policies
                ],
            }
        }

    elif query_type == "compliance":
        from syn.audit.models import ComplianceReport

        reports = ComplianceReport.objects.order_by("-created_at")[:5]
        return {
            "compliance": {
                "latest_reports": [
                    {
                        "id": str(r.id),
                        "passed": r.passed,
                        "failed": r.failed,
                        "total": r.total,
                        "created_at": r.created_at.isoformat(),
                    }
                    for r in reports
                ]
            }
        }

    elif query_type == "readiness":
        from .readiness import compute_readiness_score

        return {"readiness_detail": compute_readiness_score(user=user)}

    return {}


def _avg_conclusion_days(concluded_qs):
    """Average days from creation to conclusion."""
    days = []
    for inv in concluded_qs:
        if inv.concluded_at and inv.created_at:
            delta = (inv.concluded_at - inv.created_at).days
            days.append(delta)
    return round(sum(days) / len(days), 1) if days else None


def _avg_competency(records_qs):
    """Average competency level across training records."""
    vals = list(records_qs.filter(competency_level__gt=0).values_list("competency_level", flat=True))
    return round(sum(vals) / len(vals), 1) if vals else None


# ── Portal Template View (no auth required — token in URL) ────────────


@csrf_exempt
@require_http_methods(["GET"])
def auditor_portal_view(request, token):
    """Render the auditor portal template.

    Token validation happens client-side via the data API.
    We do a quick token check here to avoid rendering for bad tokens.
    """
    try:
        tok = AuditorPortalToken.objects.get(token=token)
    except AuditorPortalToken.DoesNotExist:
        return render(request, "loop_auditor.html", {"token_error": "Invalid token"})

    if not tok.is_valid:
        reason = "revoked" if tok.revoked_at else "expired"
        return render(request, "loop_auditor.html", {"token_error": f"Token {reason}"})

    return render(
        request,
        "loop_auditor.html",
        {
            "token": token,
            "token_label": tok.label,
            "expires_at": tok.expires_at.isoformat(),
            "org_name": tok.created_by.get_full_name() or tok.created_by.email,
        },
    )

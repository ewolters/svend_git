"""Loop API views — LOOP-001 §3.

Endpoints for Signals, Commitments, and ModeTransitions.
"""

import json
import logging
from datetime import date

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid

from .models import Commitment, ModeTransition, Signal
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

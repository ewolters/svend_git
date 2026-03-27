"""Loop services — transition execution and commitment wiring.

LOOP-001 §3.2: When a Commitment with a transition_type is fulfilled,
the system creates the corresponding artifact and records a ModeTransition.

LOOP-001 §3.3: Commitments are created at investigation conclusion
with specific transition types that encode what happens next.
"""

import logging

from django.contrib.contenttypes.models import ContentType

from .models import Commitment, ModeTransition, Signal

logger = logging.getLogger("svend.loop")


def fulfill_commitment(commitment, user):
    """Fulfill a commitment and execute its mode transition if applicable.

    This is the core loop mechanic: fulfillment → transition → artifact.

    Args:
        commitment: Commitment instance to fulfill
        user: User performing the fulfillment

    Returns:
        dict with 'commitment', 'transition' (if created), 'target' (if created)
    """
    commitment.fulfill(user)

    result = {
        "commitment_id": str(commitment.id),
        "status": commitment.status,
        "transition": None,
        "target": None,
    }

    if not commitment.transition_type:
        return result

    # Execute the mode transition
    transition, target = _execute_transition(commitment, user)
    if transition:
        result["transition"] = {
            "id": str(transition.id),
            "type": transition.transition_type,
            "from_mode": transition.from_mode,
            "to_mode": transition.to_mode,
        }
    if target:
        result["target"] = {
            "id": str(target.id) if hasattr(target, "id") else str(target.pk),
            "type": target.__class__.__name__,
        }

    return result


def _execute_transition(commitment, user):
    """Create the target artifact and record the ModeTransition.

    Returns (ModeTransition, target_artifact) or (None, None).
    """
    handler = TRANSITION_HANDLERS.get(commitment.transition_type)
    if not handler:
        logger.warning(
            "No handler for transition type %s on commitment %s",
            commitment.transition_type,
            commitment.id,
        )
        return None, None

    # Create the target artifact
    target = handler(commitment, user)
    if not target:
        return None, None

    # Determine source artifact (the investigation or whatever created the commitment)
    source = commitment.source_investigation
    if not source:
        logger.warning(
            "Commitment %s has no source_investigation — ModeTransition will lack source",
            commitment.id,
        )
        return None, target

    # Record the immutable transition
    source_ct = ContentType.objects.get_for_model(source)
    target_ct = ContentType.objects.get_for_model(target)

    transition = ModeTransition(
        transition_type=commitment.transition_type,
        source_content_type=source_ct,
        source_object_id=source.id,
        target_content_type=target_ct,
        target_object_id=target.id if hasattr(target, "id") else target.pk,
        triggered_by=commitment,
    )
    transition.save()

    # Update the commitment's target artifact link
    commitment.target_content_type = target_ct
    commitment.target_object_id = target.id if hasattr(target, "id") else target.pk
    commitment.save(update_fields=["target_content_type", "target_object_id"])

    logger.info(
        "loop.transition_executed",
        extra={
            "transition_id": str(transition.id),
            "type": commitment.transition_type,
            "commitment_id": str(commitment.id),
            "target_type": target.__class__.__name__,
            "target_id": str(target.id) if hasattr(target, "id") else str(target.pk),
        },
    )

    return transition, target


# =============================================================================
# TRANSITION HANDLERS
# =============================================================================
# Each handler creates the target artifact for its transition type.
# Returns the created artifact or None.


def _handle_revise_document(commitment, user):
    """Investigate → Standardize: open document editor with existing doc."""
    # For now, create a placeholder revision record.
    # The actual document revision happens in the editor UI.
    # This handler creates the linkage record so the transition is tracked.

    # Look for a document reference in commitment description or context
    # In v1, the commitment description should mention which document.
    # Future: structured field linking to specific ControlledDocument.
    logger.info(
        "revise_document transition — commitment %s. Document editor should open.",
        commitment.id,
    )
    # Return None for now — the target artifact (revised document) is created
    # when the user actually publishes the revision in the editor.
    # The ModeTransition will be recorded at publish time instead.
    return None


def _handle_create_document(commitment, user):
    """Investigate → Standardize: open document editor for new standard."""
    from agents_api.models import ISODocument

    doc = ISODocument.objects.create(
        title=f"New standard from investigation: {commitment.title}",
        status="draft",
        document_type="work_instruction",
        created_by=user,
    )
    logger.info("create_document: ISODocument %s created from commitment %s", doc.id, commitment.id)
    return doc


def _handle_add_control(commitment, user):
    """Investigate → Standardize: create FMIS row (or FMEA row for now)."""
    # In v1, create a standard FMEA row linked to the investigation.
    # FMIS model comes in Layer 4.
    from agents_api.models import FMEA, FMEARow

    investigation = commitment.source_investigation
    if not investigation:
        return None

    # Find or create an FMEA for this investigation's scope
    fmea, created = FMEA.objects.get_or_create(
        user=user,
        title=f"Controls from {investigation.title}",
        defaults={"description": f"Auto-created from investigation {investigation.id}"},
    )

    row = FMEARow.objects.create(
        fmea=fmea,
        failure_mode=commitment.title,
        effect="To be determined from investigation evidence",
        cause="See investigation for causal analysis",
        severity=5,
        occurrence=5,
        detection=5,
    )
    logger.info("add_control: FMEARow %s created from commitment %s", row.id, commitment.id)
    return row


def _handle_process_confirmation(commitment, user):
    """Standardize → Verify: schedule PC. Model comes in Layer 2."""
    logger.info("process_confirmation transition — deferred to Layer 2. Commitment %s", commitment.id)
    return None


def _handle_forced_failure(commitment, user):
    """Standardize → Verify: schedule forced failure test. Model comes in Layer 2."""
    logger.info("forced_failure transition — deferred to Layer 2. Commitment %s", commitment.id)
    return None


def _handle_monitor(commitment, user):
    """Standardize → Verify: set up SPC monitoring."""
    logger.info("monitor transition — deferred to Layer 2. Commitment %s", commitment.id)
    return None


def _handle_audit_zone(commitment, user):
    """Standardize → Verify: schedule frontier card audit."""
    logger.info("audit_zone transition — deferred to Layer 2. Commitment %s", commitment.id)
    return None


def _handle_first_article(commitment, user):
    """Standardize → Verify: first article inspection. Deferred."""
    logger.info("first_article transition — deferred. Commitment %s", commitment.id)
    return None


TRANSITION_HANDLERS = {
    Commitment.TransitionType.REVISE_DOCUMENT: _handle_revise_document,
    Commitment.TransitionType.CREATE_DOCUMENT: _handle_create_document,
    Commitment.TransitionType.ADD_CONTROL: _handle_add_control,
    Commitment.TransitionType.PROCESS_CONFIRMATION: _handle_process_confirmation,
    Commitment.TransitionType.FORCED_FAILURE: _handle_forced_failure,
    Commitment.TransitionType.MONITOR: _handle_monitor,
    Commitment.TransitionType.AUDIT_ZONE: _handle_audit_zone,
    Commitment.TransitionType.FIRST_ARTICLE: _handle_first_article,
}


# =============================================================================
# SIGNAL HELPERS
# =============================================================================


def create_signal_from_investigation(investigation, user, title, source_type, severity="warning"):
    """Create a Signal linked to an investigation."""
    ct = ContentType.objects.get_for_model(investigation)
    return Signal.objects.create(
        tenant=investigation.tenant,
        title=title,
        source_type=source_type,
        severity=severity,
        source_content_type=ct,
        source_object_id=investigation.id,
        created_by=user,
    )

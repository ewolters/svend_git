"""
Synara Audit Signals (AUD-001 §9)
==================================

Django signals for automatic event emission on audit model changes.

Standard:     AUD-001 §9 (Event Emission)
Compliance:   SOC 2 CC7.2, ISO 27001 A.12.7
Architecture: SBL-001 §3 (Signal → Cortex flow)
"""

import logging

from django.db.models.signals import post_save
from django.dispatch import receiver

logger = logging.getLogger(__name__)


def _emit_event(event_name: str, payload: dict):
    """
    Emit an event through Cortex with fallback handling.

    SBL-001 §3: Signal handlers emit to Cortex event bus.
    """
    try:
        from syn.synara.cortex import Cortex

        Cortex.publish(event_name, payload)
        logger.debug(f"[AUDIT] Emitted {event_name}")
    except ImportError:
        logger.warning("[AUDIT] Cortex not available, skipping event emission")
    except Exception as e:
        logger.error(f"[AUDIT] Error emitting {event_name}: {e}", exc_info=True)


# =============================================================================
# SYSLOG ENTRY SIGNALS
# =============================================================================

# TD-AUD-EVENTS-001: FIXED 2025-12-26
# Added "audit." prefix to Cortex.INTERNAL_EVENT_PREFIXES to bypass governance layer.
# This prevents GovernanceJudgement tenant_id constraint failure since audit events
# are short-circuited before reaching the governance layer.


@receiver(post_save, sender="audit.SysLogEntry")
def on_syslog_entry_created(sender, instance, created, **kwargs):
    """
    Emit event when a SysLogEntry is created.

    Note: SysLogEntry cannot be updated (immutable), so we only handle created.
    Audit events are short-circuited in Cortex to avoid governance evaluation.

    Standard: AUD-001 §9, SBL-001 INV-011
    """
    if not created:
        return

    from syn.audit.events import (
        build_entry_created_payload,
        build_genesis_created_payload,
    )

    # Build payload for entry created event
    payload = build_entry_created_payload(instance)
    _emit_event("audit.entry.created", payload)

    # Also emit genesis event if this is the first entry in chain
    if instance.is_genesis:
        genesis_payload = build_genesis_created_payload(instance)
        _emit_event("audit.entry.genesis_created", genesis_payload)


# =============================================================================
# INTEGRITY VIOLATION SIGNALS
# =============================================================================


@receiver(post_save, sender="audit.IntegrityViolation")
def on_integrity_violation_saved(sender, instance, created, **kwargs):
    """
    Emit event when an IntegrityViolation is created or resolved.

    This is a compliance-critical event that should be forwarded to SIEM.
    """
    from syn.audit.events import (
        build_governance_integrity_violation_payload,
        build_integrity_violation_payload,
        build_violation_resolved_payload,
    )

    if created:
        # New violation detected
        payload = build_integrity_violation_payload(instance)
        _emit_event("audit.integrity.violation_detected", payload)

        # Also emit governance alert for critical violations
        gov_payload = build_governance_integrity_violation_payload(
            instance,
            requires_investigation=True,
        )
        _emit_event("governance.audit_integrity_violation", gov_payload)

    elif instance.is_resolved and instance.resolved_at:
        # Violation has been resolved
        payload = build_violation_resolved_payload(
            instance,
            resolved_by=getattr(instance, "resolved_by", "system") or "system",
        )
        _emit_event("audit.integrity.violation_resolved", payload)


# =============================================================================
# DRIFT VIOLATION SIGNALS
# =============================================================================


@receiver(post_save, sender="audit.DriftViolation")
def on_drift_violation_saved(sender, instance, created, **kwargs):
    """
    Emit event when a DriftViolation is created, escalated, or resolved.

    Drift violations require careful tracking for architecture compliance.
    """
    from django.utils import timezone

    from syn.audit.events import (
        build_drift_governance_escalated_payload,
        build_drift_remediation_available_payload,
        build_drift_resolved_payload,
        build_drift_sla_breached_payload,
        build_drift_violation_payload,
        build_governance_drift_alert_payload,
    )

    if created:
        # New drift violation detected
        payload = build_drift_violation_payload(instance)
        _emit_event("audit.drift.violation_detected", payload)

        # Emit remediation available if applicable
        if instance.is_remediation_available:
            remediation_payload = build_drift_remediation_available_payload(instance)
            _emit_event("audit.drift.remediation_available", remediation_payload)

        # Emit governance alert for HIGH/CRITICAL severity
        if instance.severity in ("CRITICAL", "HIGH"):
            gov_payload = build_governance_drift_alert_payload(
                instance,
                alert_reason=f"New {instance.severity} drift violation detected",
            )
            _emit_event("governance.drift_alert", gov_payload)

    else:
        # Check for state transitions
        # Resolved
        if instance.resolved_at and instance.resolved_by:
            # Calculate resolution duration
            duration_hours = 0
            if instance.detected_at:
                delta = instance.resolved_at - instance.detected_at
                duration_hours = delta.total_seconds() / 3600

            payload = build_drift_resolved_payload(
                instance,
                resolved_by=instance.resolved_by,
                resolution_duration_hours=duration_hours,
            )
            _emit_event("audit.drift.resolved", payload)

        # Governance escalated
        if instance.is_governance_escalated:
            payload = build_drift_governance_escalated_payload(instance)
            _emit_event("audit.drift.governance_escalated", payload)

        # SLA breached
        if instance.is_sla_breached and instance.remediation_due_at:
            now = timezone.now()
            hours_overdue = (now - instance.remediation_due_at).total_seconds() / 3600

            payload = build_drift_sla_breached_payload(
                instance,
                hours_overdue=hours_overdue,
            )
            _emit_event("audit.drift.sla_breached", payload)


# =============================================================================
# REGISTRATION
# =============================================================================


def register_audit_signals():
    """
    Register all audit signals.

    Called from apps.py ready() hook to ensure signals are connected.
    """
    logger.info("[AUDIT] Audit signals registered for SBL-001 event emission")

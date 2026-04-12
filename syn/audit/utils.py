"""
Audit logging utilities for structured log entry creation.

Compliance: SOC 2 CC7.2 / ISO 27001 A.12.7
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from django.core.exceptions import ValidationError

if TYPE_CHECKING:
    from syn.audit.models import IntegrityViolation, SysLogEntry

logger = logging.getLogger(__name__)


def generate_entry(
    tenant_id: str,
    actor: str,
    event_name: str,
    payload: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> "SysLogEntry":
    """
    Generate a tamper-proof audit log entry.

    Creates a new immutable log entry in the hash chain for the tenant.
    The entry is automatically linked to the previous entry via hash chaining.

    Args:
        tenant_id: Tenant identifier for isolation
        actor: User or system component performing the action
        event_name: Name of the event being logged
        payload: Event data and context (default: empty dict)
        correlation_id: Optional correlation ID for distributed tracing

    Returns:
        Created SysLogEntry instance

    Raises:
        ValidationError: If entry creation fails

    Example:
        entry = generate_entry(
            tenant_id="tenant-123",
            actor="user@example.com",
            event_name="user.login",
            payload={"ip": "192.168.1.1", "method": "oauth"},
            correlation_id="corr-abc123"
        )

    Compliance:
    - SOC 2 CC7.2: Structured audit logging
    - ISO 27001 A.12.7: Audit log completeness
    """
    from syn.audit.models import SysLogEntry

    try:
        # Create entry (hash chain is automatic)
        entry = SysLogEntry(
            tenant_id=tenant_id,
            actor=actor,
            event_name=event_name,
            payload=payload or {},
            correlation_id=correlation_id,
        )

        # Save triggers hash computation and chain linkage
        entry.save()

        logger.debug(f"Audit entry created: {event_name} by {actor} (chain: {entry.current_hash[:8]}...)")

        # TD-AUD-EVENTS-001: FIXED 2025-12-26
        # Event emission now works because audit.* events are short-circuited
        # in Cortex.INTERNAL_EVENT_PREFIXES (bypasses governance layer).
        # The signal handler on SysLogEntry.post_save will emit the event.

        return entry

    except Exception as e:
        logger.error(f"Failed to create audit entry: {event_name} by {actor}: {e}", exc_info=True)
        raise ValidationError(f"Failed to create audit log entry: {e}")


def verify_chain_integrity(tenant_id: str) -> dict[str, Any]:
    """
    Verify the integrity of the entire audit log chain for a tenant.

    Checks:
    1. Each entry's hash is correct
    2. Each entry correctly links to the previous entry
    3. No gaps in the chain
    4. Genesis entry is valid

    Args:
        tenant_id: Tenant identifier

    Returns:
        Dictionary with verification results:
        - is_valid: Boolean indicating if chain is intact
        - total_entries: Number of entries checked
        - violations: List of detected violations
        - genesis_valid: Whether genesis entry is valid

    Compliance:
    - SOC 2 CC7.2: Audit log integrity verification
    - ISO 27001 A.12.7: Log tamper detection
    """
    from syn.audit.models import SysLogEntry

    violations = []
    total_entries = 0

    try:
        # Get all entries for tenant in order
        entries = SysLogEntry.objects.filter(tenant_id=tenant_id).order_by("id")

        total_entries = entries.count()

        if total_entries == 0:
            return {
                "is_valid": True,
                "total_entries": 0,
                "violations": [],
                "genesis_valid": None,
                "message": "No entries to verify",
            }

        # Verify genesis entry
        genesis = entries.first()
        genesis_valid = genesis.is_genesis and genesis.previous_hash == "0" * 64

        if not genesis_valid:
            violations.append(
                {
                    "type": "invalid_genesis",
                    "entry_id": genesis.id,
                    "message": "Genesis entry is invalid",
                }
            )

        # Verify each entry
        previous_entry = None

        for entry in entries:
            # Verify hash
            if not entry.verify_hash():
                violations.append(
                    {
                        "type": "hash_mismatch",
                        "entry_id": entry.id,
                        "message": f"Hash mismatch for entry {entry.id}",
                    }
                )

            # Verify chain link
            if not entry.verify_chain_link():
                violations.append(
                    {
                        "type": "chain_break",
                        "entry_id": entry.id,
                        "message": f"Chain break at entry {entry.id}",
                    }
                )

            # Check for ID gaps — but distinguish benign sequence gaps
            # (PostgreSQL auto-increment consumed by rolled-back transactions)
            # from actual missing entries (hash chain broken across gap).
            if previous_entry is not None:
                expected_id = previous_entry.id + 1
                if entry.id != expected_id:
                    # Only flag as violation if hash chain is also broken across the gap.
                    # If previous_hash matches the prior entry's current_hash, the chain
                    # is continuous despite the ID gap — this is a benign sequence gap.
                    if entry.previous_hash != previous_entry.current_hash:
                        violations.append(
                            {
                                "type": "missing_entry",
                                "entry_id": entry.id,
                                "message": f"Gap in chain with broken hash link: expected ID {expected_id}, got {entry.id}",
                            }
                        )

            previous_entry = entry

        is_valid = len(violations) == 0

        return {
            "is_valid": is_valid,
            "total_entries": total_entries,
            "violations": violations,
            "genesis_valid": genesis_valid,
            "message": ("Chain is intact" if is_valid else f"Found {len(violations)} violations"),
        }

    except Exception as e:
        logger.error(f"Chain verification failed: {e}", exc_info=True)
        return {
            "is_valid": False,
            "total_entries": total_entries,
            "violations": [{"type": "verification_error", "message": str(e)}],
            "genesis_valid": None,
            "message": f"Verification error: {e}",
        }


def record_integrity_violation(
    tenant_id: str,
    violation_type: str,
    entry_id: int | None = None,
    details: dict[str, Any] | None = None,
) -> "IntegrityViolation":
    """
    Record a detected integrity violation.

    Creates a permanent record of the violation and emits
    a governance event for alerting.

    Args:
        tenant_id: Tenant identifier
        violation_type: Type of violation (hash_mismatch, chain_break, etc.)
        entry_id: Optional ID of the affected entry
        details: Additional information about the violation

    Returns:
        Created IntegrityViolation instance

    Compliance: SOC 2 CC7.2 - Security incident tracking
    """
    from syn.audit.models import IntegrityViolation

    try:
        # Create violation record
        violation = IntegrityViolation.objects.create(
            tenant_id=tenant_id,
            violation_type=violation_type,
            entry_id=entry_id,
            details=details or {},
        )

        # Emit audit events per AUD-001 §8
        try:
            from syn.audit.events import (
                build_governance_integrity_violation_payload,
                build_integrity_violation_payload,
                emit_audit_event,
            )

            # Emit integrity violation event
            violation_payload = build_integrity_violation_payload(violation)
            emit_audit_event(
                "audit.integrity.violation_detected",
                violation_payload,
                tenant_id=tenant_id,
            )

            # Emit governance event for escalation
            governance_payload = build_governance_integrity_violation_payload(
                violation,
                requires_investigation=True,
            )
            emit_audit_event(
                "governance.audit_integrity_violation",
                governance_payload,
                tenant_id=tenant_id,
            )

            logger.critical(f"Audit integrity violation detected: {violation_type} for tenant {tenant_id}")

        except Exception as e:
            logger.error(f"Failed to emit violation event: {e}")

        return violation

    except Exception as e:
        logger.error(f"Failed to record violation: {e}", exc_info=True)
        raise


def get_audit_trail(
    tenant_id: str,
    event_name: str | None = None,
    actor: str | None = None,
    correlation_id: str | None = None,
    limit: int = 100,
) -> list:
    """
    Get audit trail entries with optional filtering.

    Args:
        tenant_id: Tenant identifier
        event_name: Filter by event name
        actor: Filter by actor
        correlation_id: Filter by correlation ID
        limit: Maximum number of entries to return

    Returns:
        List of SysLogEntry instances

    Compliance: SOC 2 CC7.2 - Audit trail retrieval
    """
    from syn.audit.models import SysLogEntry

    queryset = SysLogEntry.objects.filter(tenant_id=tenant_id)

    if event_name:
        queryset = queryset.filter(event_name=event_name)

    if actor:
        queryset = queryset.filter(actor=actor)

    if correlation_id:
        queryset = queryset.filter(correlation_id=correlation_id)

    return list(queryset.order_by("-timestamp")[:limit])

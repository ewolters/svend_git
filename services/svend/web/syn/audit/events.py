"""
Synara Audit Events Catalog (AUD-001 §8, AUD-002, POL-002 §7)
=============================================================

Event definitions for audit log lifecycle, integrity verification,
drift detection, and compliance tracking.

Standard:     AUD-001 §8 (Event Taxonomy), AUD-002 §5 (SIEM Integration)
              POL-002 §7 (Event Redaction)
Compliance:   SOC 2 CC7.2, ISO 27001 A.12.7, NIST SP 800-53 AU-2
Location:     syn/audit/events.py
Version:      1.1.0
"""

import logging
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# SENSITIVE FIELD PATTERNS (POL-002 §7)
# =============================================================================

# Fields that should always be redacted in event payloads
SENSITIVE_FIELD_PATTERNS: Set[str] = {
    "password", "secret", "token", "api_key", "apikey",
    "ssn", "social_security", "credit_card", "card_number",
    "cvv", "pin", "private_key", "secret_key",
}

# Fields that contain PII and need masking
PII_FIELD_PATTERNS: Set[str] = {
    "email", "phone", "address", "name", "first_name", "last_name",
    "dob", "date_of_birth", "ssn", "license", "passport",
}


def _is_sensitive_field(field_name: str) -> bool:
    """Check if a field name matches sensitive patterns."""
    field_lower = field_name.lower()
    return any(pattern in field_lower for pattern in SENSITIVE_FIELD_PATTERNS)


def _is_pii_field(field_name: str) -> bool:
    """Check if a field name matches PII patterns."""
    field_lower = field_name.lower()
    return any(pattern in field_lower for pattern in PII_FIELD_PATTERNS)


# =============================================================================
# AUDIT EVENTS CATALOG (AUD-001 §8)
# =============================================================================

AUDIT_EVENTS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # SYSLOG ENTRY EVENTS
    # =========================================================================

    "audit.entry.created": {
        "description": "New audit log entry created in hash chain",
        "category": "audit",
        "audit": False,  # Avoid recursive auditing
        "priority": 3,
        "siem_forward": False,
        "payload_schema": {
            "type": "object",
            "properties": {
                "entry_id": {"type": "integer"},
                "correlation_id": {"type": ["string", "null"], "format": "uuid"},
                "tenant_id": {"type": "string"},
                "event_name": {"type": "string"},
                "actor": {"type": "string"},
                "current_hash": {"type": "string"},
                "previous_hash": {"type": "string"},
                "is_genesis": {"type": "boolean"},
            },
            "required": ["entry_id", "tenant_id", "event_name", "actor"],
        },
    },

    "audit.entry.genesis_created": {
        "description": "Genesis (first) entry created for tenant chain",
        "category": "audit",
        "audit": False,
        "priority": 2,
        "siem_forward": True,
        "payload_schema": {
            "type": "object",
            "properties": {
                "entry_id": {"type": "integer"},
                "tenant_id": {"type": "string"},
                "current_hash": {"type": "string"},
            },
            "required": ["entry_id", "tenant_id", "current_hash"],
        },
    },

    # =========================================================================
    # INTEGRITY EVENTS
    # =========================================================================

    "audit.integrity.violation_detected": {
        "description": "Audit log integrity violation detected",
        "category": "audit",
        "audit": False,
        "priority": 1,
        "siem_forward": True,
        "siem_severity": "critical",
        "payload_schema": {
            "type": "object",
            "properties": {
                "violation_id": {"type": "string", "format": "uuid"},
                "correlation_id": {"type": "string", "format": "uuid"},
                "tenant_id": {"type": "string"},
                "violation_type": {
                    "type": "string",
                    "enum": ["hash_mismatch", "chain_break", "missing_entry", "duplicate_hash"],
                },
                "entry_id": {"type": ["integer", "null"]},
                "details": {"type": "object"},
            },
            "required": ["violation_id", "tenant_id", "violation_type"],
        },
    },

    "audit.integrity.chain_verified": {
        "description": "Audit log chain integrity verified",
        "category": "audit",
        "audit": False,
        "priority": 3,
        "siem_forward": False,
        "payload_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "correlation_id": {"type": "string", "format": "uuid"},
                "total_entries": {"type": "integer"},
                "is_valid": {"type": "boolean"},
                "genesis_valid": {"type": ["boolean", "null"]},
                "verification_duration_ms": {"type": "number"},
            },
            "required": ["tenant_id", "total_entries", "is_valid"],
        },
    },

    "audit.integrity.chain_verification_failed": {
        "description": "Audit log chain verification failed",
        "category": "audit",
        "audit": False,
        "priority": 1,
        "siem_forward": True,
        "siem_severity": "high",
        "payload_schema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "correlation_id": {"type": "string", "format": "uuid"},
                "total_entries": {"type": "integer"},
                "violations_count": {"type": "integer"},
                "violation_types": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tenant_id", "violations_count"],
        },
    },

    "audit.integrity.violation_resolved": {
        "description": "Audit integrity violation resolved",
        "category": "audit",
        "audit": True,
        "priority": 2,
        "siem_forward": True,
        "siem_severity": "medium",
        "payload_schema": {
            "type": "object",
            "properties": {
                "violation_id": {"type": "string", "format": "uuid"},
                "correlation_id": {"type": "string", "format": "uuid"},
                "tenant_id": {"type": "string"},
                "resolved_by": {"type": "string"},
                "resolution_notes": {"type": "string"},
            },
            "required": ["violation_id", "tenant_id", "resolved_by"],
        },
    },

    # =========================================================================
    # DRIFT VIOLATION EVENTS (DRF-001 §7)
    # =========================================================================

    "audit.drift.violation_detected": {
        "description": "Architectural drift violation detected",
        "category": "audit",
        "audit": True,
        "priority": 1,
        "siem_forward": True,
        "siem_severity": "high",
        "payload_schema": {
            "type": "object",
            "properties": {
                "drift_id": {"type": "string", "format": "uuid"},
                "drift_signature": {"type": "string"},
                "correlation_id": {"type": "string", "format": "uuid"},
                "tenant_id": {"type": ["string", "null"], "format": "uuid"},
                "enforcement_check": {"type": "string"},
                "severity": {"type": "string", "enum": ["CRITICAL", "HIGH", "MEDIUM", "LOW"]},
                "file_path": {"type": "string"},
                "line_number": {"type": ["integer", "null"]},
                "function_name": {"type": ["string", "null"]},
                "violation_message": {"type": "string"},
                "git_commit_sha": {"type": ["string", "null"]},
                "detected_by": {"type": "string"},
            },
            "required": ["drift_id", "enforcement_check", "severity", "file_path", "violation_message"],
        },
    },

    "audit.drift.remediation_available": {
        "description": "Automated remediation available for drift violation",
        "category": "audit",
        "audit": True,
        "priority": 2,
        "siem_forward": False,
        "payload_schema": {
            "type": "object",
            "properties": {
                "drift_id": {"type": "string", "format": "uuid"},
                "correlation_id": {"type": "string", "format": "uuid"},
                "enforcement_check": {"type": "string"},
                "auto_fix_safe": {"type": "boolean"},
                "remediation_script": {"type": ["string", "null"]},
            },
            "required": ["drift_id", "auto_fix_safe"],
        },
    },

    "audit.drift.sla_breached": {
        "description": "Drift violation remediation SLA breached",
        "category": "audit",
        "audit": True,
        "priority": 1,
        "siem_forward": True,
        "siem_severity": "high",
        "payload_schema": {
            "type": "object",
            "properties": {
                "drift_id": {"type": "string", "format": "uuid"},
                "correlation_id": {"type": "string", "format": "uuid"},
                "tenant_id": {"type": ["string", "null"], "format": "uuid"},
                "enforcement_check": {"type": "string"},
                "severity": {"type": "string"},
                "remediation_due_at": {"type": "string", "format": "date-time"},
                "hours_overdue": {"type": "number"},
            },
            "required": ["drift_id", "enforcement_check", "remediation_due_at"],
        },
    },

    "audit.drift.governance_escalated": {
        "description": "Drift violation escalated to governance layer",
        "category": "audit",
        "audit": True,
        "priority": 1,
        "siem_forward": True,
        "siem_severity": "medium",
        "payload_schema": {
            "type": "object",
            "properties": {
                "drift_id": {"type": "string", "format": "uuid"},
                "correlation_id": {"type": "string", "format": "uuid"},
                "tenant_id": {"type": ["string", "null"], "format": "uuid"},
                "enforcement_check": {"type": "string"},
                "severity": {"type": "string"},
                "governance_rule_id": {"type": ["string", "null"], "format": "uuid"},
            },
            "required": ["drift_id", "enforcement_check", "severity"],
        },
    },

    "audit.drift.resolved": {
        "description": "Drift violation resolved",
        "category": "audit",
        "audit": True,
        "priority": 2,
        "siem_forward": True,
        "payload_schema": {
            "type": "object",
            "properties": {
                "drift_id": {"type": "string", "format": "uuid"},
                "correlation_id": {"type": "string", "format": "uuid"},
                "tenant_id": {"type": ["string", "null"], "format": "uuid"},
                "enforcement_check": {"type": "string"},
                "severity": {"type": "string"},
                "resolved_by": {"type": "string"},
                "resolution_notes": {"type": "string"},
                "resolution_duration_hours": {"type": "number"},
            },
            "required": ["drift_id", "resolved_by"],
        },
    },

    # =========================================================================
    # GOVERNANCE INTEGRATION EVENTS
    # =========================================================================

    "governance.audit_integrity_violation": {
        "description": "Audit integrity violation reported to governance",
        "category": "governance",
        "audit": True,
        "priority": 1,
        "siem_forward": True,
        "siem_severity": "critical",
        "payload_schema": {
            "type": "object",
            "properties": {
                "violation_id": {"type": "string", "format": "uuid"},
                "correlation_id": {"type": "string", "format": "uuid"},
                "tenant_id": {"type": "string"},
                "violation_type": {"type": "string"},
                "entry_id": {"type": ["integer", "null"]},
                "requires_investigation": {"type": "boolean"},
            },
            "required": ["violation_id", "tenant_id", "violation_type"],
        },
    },

    "governance.drift_alert": {
        "description": "Drift violation alert sent to governance",
        "category": "governance",
        "audit": True,
        "priority": 1,
        "siem_forward": True,
        "siem_severity": "high",
        "payload_schema": {
            "type": "object",
            "properties": {
                "drift_id": {"type": "string", "format": "uuid"},
                "correlation_id": {"type": "string", "format": "uuid"},
                "tenant_id": {"type": ["string", "null"], "format": "uuid"},
                "enforcement_check": {"type": "string"},
                "severity": {"type": "string"},
                "alert_reason": {"type": "string"},
            },
            "required": ["drift_id", "enforcement_check", "severity", "alert_reason"],
        },
    },

    # =========================================================================
    # AUDIT TRAIL QUERY EVENTS
    # =========================================================================

    "audit.trail.queried": {
        "description": "Audit trail query executed",
        "category": "audit",
        "audit": True,
        "priority": 3,
        "siem_forward": False,
        "payload_schema": {
            "type": "object",
            "properties": {
                "correlation_id": {"type": "string", "format": "uuid"},
                "tenant_id": {"type": "string"},
                "actor": {"type": "string"},
                "query_params": {"type": "object"},
                "results_count": {"type": "integer"},
            },
            "required": ["tenant_id", "actor", "results_count"],
        },
    },

    "audit.trail.export_requested": {
        "description": "Audit trail export requested",
        "category": "audit",
        "audit": True,
        "priority": 2,
        "siem_forward": True,
        "payload_schema": {
            "type": "object",
            "properties": {
                "correlation_id": {"type": "string", "format": "uuid"},
                "tenant_id": {"type": "string"},
                "actor": {"type": "string"},
                "date_range_start": {"type": "string", "format": "date-time"},
                "date_range_end": {"type": "string", "format": "date-time"},
                "export_format": {"type": "string"},
                "entries_count": {"type": "integer"},
            },
            "required": ["tenant_id", "actor", "export_format"],
        },
    },

    # =========================================================================
    # RETENTION EVENTS
    # =========================================================================

    "audit.retention.policy_applied": {
        "description": "Audit log retention policy applied",
        "category": "audit",
        "audit": True,
        "priority": 2,
        "siem_forward": True,
        "payload_schema": {
            "type": "object",
            "properties": {
                "correlation_id": {"type": "string", "format": "uuid"},
                "tenant_id": {"type": "string"},
                "policy_name": {"type": "string"},
                "entries_archived": {"type": "integer"},
                "entries_deleted": {"type": "integer"},
                "retention_days": {"type": "integer"},
            },
            "required": ["tenant_id", "policy_name"],
        },
    },
}


# =============================================================================
# EVENT PAYLOAD REDACTION (POL-002 §7)
# =============================================================================


def redact_event_payload(
    payload: Dict[str, Any],
    depth: int = 0,
    max_depth: int = 10,
) -> tuple[Dict[str, Any], int, List[str]]:
    """
    Redact sensitive fields from an event payload.

    POL-002 §7 Requirements:
    - Event payloads MUST redact sensitive fields
    - Event metadata MUST include redaction_count
    - Redacted events MUST include sensitivity_tags

    Args:
        payload: Event payload to redact
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Tuple of (redacted_payload, redaction_count, sensitivity_tags)
    """
    if depth > max_depth:
        return payload, 0, []

    redacted = {}
    redaction_count = 0
    sensitivity_tags: List[str] = []

    for key, value in payload.items():
        # Skip internal metadata fields
        if key.startswith("_"):
            redacted[key] = value
            continue

        # Check if field is sensitive
        if _is_sensitive_field(key):
            redacted[key] = "***REDACTED***"
            redaction_count += 1
            if "SENSITIVE" not in sensitivity_tags:
                sensitivity_tags.append("SENSITIVE")
            continue

        # Check if field contains PII
        if _is_pii_field(key):
            # Apply masking for PII (show partial info)
            if isinstance(value, str):
                if "@" in value:  # Email
                    parts = value.rsplit("@", 1)
                    if len(parts) == 2 and len(parts[0]) >= 2:
                        redacted[key] = f"{parts[0][:2]}***@{parts[1]}"
                    else:
                        redacted[key] = "***@***"
                elif len(value) > 4:
                    redacted[key] = f"{value[:2]}***{value[-2:]}"
                else:
                    redacted[key] = "***"
            else:
                redacted[key] = "***REDACTED***"
            redaction_count += 1
            if "PII" not in sensitivity_tags:
                sensitivity_tags.append("PII")
            continue

        # Recursively process nested dicts
        if isinstance(value, dict):
            nested_redacted, nested_count, nested_tags = redact_event_payload(
                value, depth + 1, max_depth
            )
            redacted[key] = nested_redacted
            redaction_count += nested_count
            for tag in nested_tags:
                if tag not in sensitivity_tags:
                    sensitivity_tags.append(tag)
            continue

        # Recursively process lists
        if isinstance(value, list):
            redacted_list = []
            for item in value:
                if isinstance(item, dict):
                    nested_redacted, nested_count, nested_tags = redact_event_payload(
                        item, depth + 1, max_depth
                    )
                    redacted_list.append(nested_redacted)
                    redaction_count += nested_count
                    for tag in nested_tags:
                        if tag not in sensitivity_tags:
                            sensitivity_tags.append(tag)
                else:
                    redacted_list.append(item)
            redacted[key] = redacted_list
            continue

        # Non-sensitive field - keep as-is
        redacted[key] = value

    return redacted, redaction_count, sensitivity_tags


# =============================================================================
# EVENT EMISSION FUNCTIONS
# =============================================================================


def emit_audit_event(
    event_name: str,
    payload: Dict[str, Any],
    correlation_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    skip_redaction: bool = False,
) -> bool:
    """
    Emit an audit event through the kernel event system.

    Args:
        event_name: Event name (must be in AUDIT_EVENTS)
        payload: Event payload data
        correlation_id: CTG correlation ID (auto-generated if not provided)
        tenant_id: Tenant ID for scoping (required for SYS-200 INV-001 compliance)
        actor_id: Actor who triggered the event
        skip_redaction: Skip redaction (only for internal events that are already clean)

    Returns:
        bool: True if event was emitted successfully

    Compliance:
    - AUD-001 §8: Event taxonomy
    - CTG-001 §5: Correlation tracking
    - SYS-200 INV-001: Tenant isolation (tenant_id should be provided)
    - EVT-001 §4.3: Event emission
    - POL-002 §7: Event redaction
    """
    if event_name not in AUDIT_EVENTS:
        logger.warning(f"[AUDIT EVENTS] Unknown event: {event_name}")
        return False

    try:
        from syn.synara.cortex import Cortex

        # Ensure correlation_id (CTG-001 §5)
        final_correlation_id = correlation_id or str(uuid4())

        # SYS-200 INV-001: Warn if tenant_id is not provided
        if not tenant_id:
            logger.warning(
                f"[AUDIT EVENTS] emit_audit_event called without tenant_id for {event_name}. "
                f"SYS-200 INV-001 requires tenant_id for multi-tenant isolation. "
                f"correlation_id={final_correlation_id}"
            )

        # POL-002 §7: Apply redaction to event payload
        redaction_count = 0
        sensitivity_tags: List[str] = []

        if not skip_redaction:
            payload, redaction_count, sensitivity_tags = redact_event_payload(payload)
            if redaction_count > 0:
                logger.debug(
                    f"[AUDIT EVENTS] Redacted {redaction_count} sensitive fields "
                    f"from {event_name} (tags: {sensitivity_tags})"
                )

        # Enrich payload with metadata
        enriched_payload = {
            **payload,
            "correlation_id": final_correlation_id,
            "_event_source": "audit.events",
            # POL-002 §7: Event metadata MUST include redaction_count
            "_redaction_count": redaction_count,
            "_redaction_applied": redaction_count > 0,
        }

        # POL-002 §7: Redacted events MUST include sensitivity_tags
        if sensitivity_tags:
            enriched_payload["_sensitivity_tags"] = sensitivity_tags

        if tenant_id:
            enriched_payload["tenant_id"] = tenant_id
        if actor_id:
            enriched_payload["actor_id"] = actor_id

        # Publish event
        Cortex.publish(event_name, enriched_payload)

        logger.info(
            f"[AUDIT EVENTS] Emitted {event_name}: "
            f"correlation_id={final_correlation_id}"
        )
        return True

    except ImportError:
        logger.warning("[AUDIT EVENTS] Cortex not available, skipping event emission")
        return False
    except Exception as e:
        logger.error(f"[AUDIT EVENTS] Error emitting {event_name}: {e}", exc_info=True)
        return False


# =============================================================================
# PAYLOAD BUILDER FUNCTIONS
# =============================================================================


def build_entry_created_payload(entry) -> Dict[str, Any]:
    """Build payload for audit.entry.created event."""
    return {
        "entry_id": entry.id,
        "correlation_id": str(entry.correlation_id) if entry.correlation_id else None,
        "tenant_id": entry.tenant_id,
        "event_name": entry.event_name,
        "actor": entry.actor,
        "current_hash": entry.current_hash,
        "previous_hash": entry.previous_hash,
        "is_genesis": entry.is_genesis,
    }


def build_genesis_created_payload(entry) -> Dict[str, Any]:
    """Build payload for audit.entry.genesis_created event."""
    return {
        "entry_id": entry.id,
        "tenant_id": entry.tenant_id,
        "current_hash": entry.current_hash,
    }


def build_integrity_violation_payload(
    violation,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build payload for audit.integrity.violation_detected event."""
    return {
        "violation_id": str(violation.id),
        "correlation_id": correlation_id or str(uuid4()),
        "tenant_id": violation.tenant_id,
        "violation_type": violation.violation_type,
        "entry_id": violation.entry_id,
        "details": violation.details,
    }


def build_chain_verified_payload(
    tenant_id: str,
    total_entries: int,
    is_valid: bool,
    genesis_valid: Optional[bool] = None,
    verification_duration_ms: Optional[float] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build payload for audit.integrity.chain_verified event."""
    return {
        "tenant_id": tenant_id,
        "correlation_id": correlation_id or str(uuid4()),
        "total_entries": total_entries,
        "is_valid": is_valid,
        "genesis_valid": genesis_valid,
        "verification_duration_ms": verification_duration_ms,
    }


def build_chain_verification_failed_payload(
    tenant_id: str,
    total_entries: int,
    violations_count: int,
    violation_types: list,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build payload for audit.integrity.chain_verification_failed event."""
    return {
        "tenant_id": tenant_id,
        "correlation_id": correlation_id or str(uuid4()),
        "total_entries": total_entries,
        "violations_count": violations_count,
        "violation_types": violation_types,
    }


def build_violation_resolved_payload(
    violation,
    resolved_by: str,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build payload for audit.integrity.violation_resolved event."""
    return {
        "violation_id": str(violation.id),
        "correlation_id": correlation_id or str(uuid4()),
        "tenant_id": violation.tenant_id,
        "resolved_by": resolved_by,
        "resolution_notes": violation.resolution_notes,
    }


def build_drift_violation_payload(drift) -> Dict[str, Any]:
    """Build payload for audit.drift.violation_detected event."""
    return {
        "drift_id": str(drift.id),
        "drift_signature": drift.drift_signature,
        "correlation_id": str(drift.correlation_id),
        "tenant_id": str(drift.tenant_id) if drift.tenant_id else None,
        "enforcement_check": drift.enforcement_check,
        "severity": drift.severity,
        "file_path": drift.file_path,
        "line_number": drift.line_number,
        "function_name": drift.function_name,
        "violation_message": drift.violation_message,
        "git_commit_sha": drift.git_commit_sha,
        "detected_by": drift.detected_by,
    }


def build_drift_remediation_available_payload(
    drift,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build payload for audit.drift.remediation_available event."""
    return {
        "drift_id": str(drift.id),
        "correlation_id": correlation_id or str(drift.correlation_id),
        "enforcement_check": drift.enforcement_check,
        "auto_fix_safe": drift.auto_fix_safe,
        "remediation_script": drift.remediation_script,
    }


def build_drift_sla_breached_payload(
    drift,
    hours_overdue: float,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build payload for audit.drift.sla_breached event."""
    return {
        "drift_id": str(drift.id),
        "correlation_id": correlation_id or str(drift.correlation_id),
        "tenant_id": str(drift.tenant_id) if drift.tenant_id else None,
        "enforcement_check": drift.enforcement_check,
        "severity": drift.severity,
        "remediation_due_at": drift.remediation_due_at.isoformat() if drift.remediation_due_at else None,
        "hours_overdue": hours_overdue,
    }


def build_drift_governance_escalated_payload(
    drift,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build payload for audit.drift.governance_escalated event."""
    return {
        "drift_id": str(drift.id),
        "correlation_id": correlation_id or str(drift.correlation_id),
        "tenant_id": str(drift.tenant_id) if drift.tenant_id else None,
        "enforcement_check": drift.enforcement_check,
        "severity": drift.severity,
        "governance_rule_id": str(drift.governance_rule_id) if drift.governance_rule_id else None,
    }


def build_drift_resolved_payload(
    drift,
    resolved_by: str,
    resolution_duration_hours: float,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build payload for audit.drift.resolved event."""
    return {
        "drift_id": str(drift.id),
        "correlation_id": correlation_id or str(drift.correlation_id),
        "tenant_id": str(drift.tenant_id) if drift.tenant_id else None,
        "enforcement_check": drift.enforcement_check,
        "severity": drift.severity,
        "resolved_by": resolved_by,
        "resolution_notes": drift.resolution_notes,
        "resolution_duration_hours": resolution_duration_hours,
    }


def build_governance_integrity_violation_payload(
    violation,
    requires_investigation: bool = True,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build payload for governance.audit_integrity_violation event."""
    return {
        "violation_id": str(violation.id),
        "correlation_id": correlation_id or str(uuid4()),
        "tenant_id": violation.tenant_id,
        "violation_type": violation.violation_type,
        "entry_id": violation.entry_id,
        "requires_investigation": requires_investigation,
    }


def build_governance_drift_alert_payload(
    drift,
    alert_reason: str,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build payload for governance.drift_alert event."""
    return {
        "drift_id": str(drift.id),
        "correlation_id": correlation_id or str(drift.correlation_id),
        "tenant_id": str(drift.tenant_id) if drift.tenant_id else None,
        "enforcement_check": drift.enforcement_check,
        "severity": drift.severity,
        "alert_reason": alert_reason,
    }


def build_trail_queried_payload(
    tenant_id: str,
    actor: str,
    query_params: dict,
    results_count: int,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build payload for audit.trail.queried event."""
    return {
        "correlation_id": correlation_id or str(uuid4()),
        "tenant_id": tenant_id,
        "actor": actor,
        "query_params": query_params,
        "results_count": results_count,
    }


def build_trail_export_payload(
    tenant_id: str,
    actor: str,
    export_format: str,
    date_range_start=None,
    date_range_end=None,
    entries_count: int = 0,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build payload for audit.trail.export_requested event."""
    return {
        "correlation_id": correlation_id or str(uuid4()),
        "tenant_id": tenant_id,
        "actor": actor,
        "date_range_start": date_range_start.isoformat() if date_range_start else None,
        "date_range_end": date_range_end.isoformat() if date_range_end else None,
        "export_format": export_format,
        "entries_count": entries_count,
    }


def build_retention_policy_payload(
    tenant_id: str,
    policy_name: str,
    entries_archived: int = 0,
    entries_deleted: int = 0,
    retention_days: int = 0,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build payload for audit.retention.policy_applied event."""
    return {
        "correlation_id": correlation_id or str(uuid4()),
        "tenant_id": tenant_id,
        "policy_name": policy_name,
        "entries_archived": entries_archived,
        "entries_deleted": entries_deleted,
        "retention_days": retention_days,
    }

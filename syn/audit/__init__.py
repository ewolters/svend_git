"""
Synara Audit Module (AUD-001/002)
=================================

Tamper-proof audit logging with blockchain-style hash chaining,
integrity verification, drift detection, and SIEM integration.

Standard:     AUD-001 (Conceptual), AUD-002 (Implementation)
Compliance:   SOC 2 CC7.2, ISO 27001 A.12.7, NIST SP 800-53 AU-2
Location:     syn/audit/
Version:      1.0.0

Usage:
------
    from syn.audit.models import SysLogEntry, IntegrityViolation, DriftViolation
    from syn.audit.utils import generate_entry, verify_chain_integrity
    from syn.audit.events import emit_audit_event, AUDIT_EVENTS
"""

__version__ = "1.0.0"
__standard__ = "AUD-001"

_model_names = {"SysLogEntry", "IntegrityViolation", "DriftViolation"}
_util_names = {
    "generate_entry",
    "verify_chain_integrity",
    "record_integrity_violation",
    "get_audit_trail",
}
_event_names = {
    "AUDIT_EVENTS",
    "emit_audit_event",
    "build_entry_created_payload",
    "build_genesis_created_payload",
    "build_integrity_violation_payload",
    "build_chain_verified_payload",
    "build_chain_verification_failed_payload",
    "build_violation_resolved_payload",
    "build_drift_violation_payload",
    "build_drift_remediation_available_payload",
    "build_drift_sla_breached_payload",
    "build_drift_governance_escalated_payload",
    "build_drift_resolved_payload",
    "build_governance_integrity_violation_payload",
    "build_governance_drift_alert_payload",
    "build_trail_queried_payload",
    "build_trail_export_payload",
    "build_retention_policy_payload",
}


def __getattr__(name):
    if name == "AuditConfig":
        from syn.audit.apps import AuditConfig

        return AuditConfig
    elif name in _model_names:
        from syn.audit import models

        return getattr(models, name)
    elif name in _util_names:
        from syn.audit import utils

        return getattr(utils, name)
    elif name in _event_names:
        from syn.audit import events

        return getattr(events, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_model_names | _util_names | _event_names | {"__version__", "__standard__", "AuditConfig"})

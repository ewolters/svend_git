"""
Synara Governance Module (GOV-001/002, POL-001/002)
===================================================

Governance layer implementing preflight validation, approval workflows,
and escalation handling per GOV-001/002 and POL-001/002 standards.

Standard:     GOV-001 (Patterns), GOV-002 (Implementation)
              POL-001 (Policy Framework), POL-002 (Compliance/Redaction)
Compliance:   ISO 27001:2022 A.5.1, SOC 2 CC8.1, 21 CFR Part 11
Location:     syn/governance/
Version:      1.0.0

Features:
---------
- GovernanceRule: Database-backed governance rules with conditions
- GovernanceJudgement: Immutable evaluation records (AUD-001)
- ApprovalRequest: Multi-step approval workflows
- ApprovalSignature: 21 CFR Part 11 compliant signatures
- Judgment types: ALLOW, BLOCK, ESCALATE (GOV-001 §5)
- Timeout classes: low, medium, high, critical (GOV-001 §12)
- Domain inheritance: inherit, extend, override (GOV-001 §13)
- Fail-safe behavior: BLOCK on timeout/error (GOV-001 §5.2)

Usage:
------
    from syn.governance import (
        GovernanceRule,
        GovernanceJudgement,
        ApprovalRequest,
        JudgmentType,
        RuleStatus,
        emit_governance_event,
    )

    # Create a governance rule
    rule = GovernanceRule.objects.create(
        tenant_id=tenant_uuid,
        rule_id="gov-change-001",
        name="Require approval for production changes",
        trigger_event="deployment.production.requested",
        conditions={"environment": "production"},
        judgment_type=JudgmentType.ESCALATE.value,
        status=RuleStatus.ACTIVE.value,
        created_by="admin@example.com",
    )

    # Evaluate rules for an event
    rules = GovernanceRule.get_active_rules_for_event(
        tenant_id=tenant_uuid,
        event_name="deployment.production.requested",
    )

    for rule in rules:
        judgment, confidence, reason = rule.evaluate(event_payload)
        GovernanceJudgement.record(
            tenant_id=tenant_uuid,
            correlation_id=correlation_uuid,
            event_name="deployment.production.requested",
            event_payload=payload,
            rule=rule,
            judgment=judgment,
            judgment_reason=reason,
            actor="user@example.com",
            confidence_score=confidence,
        )
"""

__version__ = "1.0.0"
__standard__ = "GOV-001"

# Import models and utilities directly from submodules:
#   from syn.governance.models import GovernanceRule, GovernanceJudgement
#   from syn.governance.types import JudgmentType, RuleStatus
#   from syn.governance.events import emit_governance_event, GOVERNANCE_EVENTS

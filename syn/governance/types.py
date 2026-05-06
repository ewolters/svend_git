"""
Synara Governance Primitives (GOV-001/002, POL-001/002)
=======================================================

Type definitions and enumerations for the governance layer.

Standard:     GOV-001 §3-5, GOV-002, POL-001/002
Compliance:   ISO 27001:2022 A.5.1, SOC 2 CC8.1, 21 CFR Part 11
Version:      1.0.0
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List

# =============================================================================
# JUDGMENT TYPES (GOV-001 §5.1)
# =============================================================================


class JudgmentType(str, Enum):
    """
    Governance judgment types per GOV-001 §5.1 and CGS-1001.

    Defines the four possible outcomes of governance rule evaluation:
    - ALLOW: Event proceeds to execution
    - BLOCK: Event halted immediately
    - ESCALATE: Event queued for human review
    - MONITOR: Event proceeds but is flagged for audit (CGS-1001)
    """

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"
    MONITOR = "MONITOR"

    @property
    def requires_approval(self) -> bool:
        """Whether this judgment requires human approval."""
        return self == JudgmentType.ESCALATE

    @property
    def is_terminal(self) -> bool:
        """Whether this judgment is terminal (no further processing)."""
        return self in (JudgmentType.ALLOW, JudgmentType.BLOCK, JudgmentType.MONITOR)

    @property
    def is_audit_only(self) -> bool:
        """Whether this judgment is for audit-only monitoring (CGS-1001)."""
        return self == JudgmentType.MONITOR


# =============================================================================
# RULE STATUS (GOV-002 §models.GovernanceRule)
# =============================================================================


class RuleStatus(str, Enum):
    """
    Governance rule lifecycle status per GOV-002.

    Rules progress through: DRAFT -> ACTIVE -> DEPRECATED
    """

    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"

    @classmethod
    def valid_transitions(cls) -> Dict["RuleStatus", List["RuleStatus"]]:
        """Valid state transitions for rule lifecycle."""
        return {
            cls.DRAFT: [cls.ACTIVE, cls.DEPRECATED],
            cls.ACTIVE: [cls.DEPRECATED],
            cls.DEPRECATED: [],  # Terminal state
        }

    def can_transition_to(self, new_status: "RuleStatus") -> bool:
        """Check if transition to new_status is allowed."""
        return new_status in self.valid_transitions().get(self, [])


# =============================================================================
# TIMEOUT CLASSES (GOV-001 §12)
# =============================================================================


class TimeoutClass(str, Enum):
    """
    Timeout classifications per GOV-001 §12.

    Defines approval timeout windows based on change criticality:
    - LOW: 4 hours (routine approvals)
    - MEDIUM: 8 hours (standard reviews)
    - HIGH: 24 hours (high-impact changes)
    - CRITICAL: 72 hours (infrastructure invariants)
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def default_hours(self) -> int:
        """Default timeout in hours."""
        defaults = {
            TimeoutClass.LOW: 4,
            TimeoutClass.MEDIUM: 8,
            TimeoutClass.HIGH: 24,
            TimeoutClass.CRITICAL: 72,
        }
        return defaults[self]

    @property
    def min_hours(self) -> int:
        """Minimum timeout in hours."""
        mins = {
            TimeoutClass.LOW: 1,
            TimeoutClass.MEDIUM: 4,
            TimeoutClass.HIGH: 8,
            TimeoutClass.CRITICAL: 24,
        }
        return mins[self]

    @property
    def max_hours(self) -> int:
        """Maximum timeout in hours."""
        maxs = {
            TimeoutClass.LOW: 8,
            TimeoutClass.MEDIUM: 24,
            TimeoutClass.HIGH: 72,
            TimeoutClass.CRITICAL: 168,
        }
        return maxs[self]

    @property
    def on_timeout(self) -> JudgmentType:
        """Judgment when timeout occurs."""
        # CRITICAL times out to BLOCK; others ESCALATE
        if self == TimeoutClass.CRITICAL:
            return JudgmentType.BLOCK
        return JudgmentType.ESCALATE


# =============================================================================
# INHERITANCE MODES (GOV-001 §13)
# =============================================================================


class InheritanceMode(str, Enum):
    """
    Domain inheritance modes per GOV-001 §13.

    Controls how child domains inherit governance rules:
    - INHERIT: Parent rules apply as-is (default)
    - EXTEND: Child adds to parent rules (most restrictive wins)
    - OVERRIDE: Child replaces parent rules
    """

    INHERIT = "inherit"
    EXTEND = "extend"
    OVERRIDE = "override"


# =============================================================================
# APPROVAL STATUS (GOV-001 §6)
# =============================================================================


class ApprovalStatus(str, Enum):
    """
    Approval request status per GOV-001 §6.

    Tracks the lifecycle of approval requests:
    - PENDING: Awaiting approver action
    - APPROVED: All required approvals received
    - REJECTED: Approver rejected the request
    - EXPIRED: Timeout reached without decision
    - CANCELLED: Requester cancelled
    """

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        """Whether this status is terminal (no further changes)."""
        return self in (
            ApprovalStatus.APPROVED,
            ApprovalStatus.REJECTED,
            ApprovalStatus.EXPIRED,
            ApprovalStatus.CANCELLED,
        )

    @property
    def is_successful(self) -> bool:
        """Whether this status represents successful approval."""
        return self == ApprovalStatus.APPROVED


# =============================================================================
# RULE CLASSIFICATIONS (GOV-001 §4.2)
# =============================================================================


class RuleClassification(str, Enum):
    """
    Governance rule classification per GOV-001 §4.2.

    Categories for organizing governance rules:
    - CHANGE: Change management rules
    - APPROVAL: Approval workflow rules
    - COMPLIANCE: Regulatory compliance rules
    - QUALITY: Quality assurance rules
    - RISK: Risk management rules
    - OPERATIONAL: Operational controls
    - MIGRATION: Database migration governance
    - INFRASTRUCTURE: Infrastructure change governance
    - SCHEMA: Schema alteration governance
    """

    CHANGE = "governance.change"
    APPROVAL = "governance.approval"
    COMPLIANCE = "governance.compliance"
    QUALITY = "governance.quality"
    RISK = "governance.risk"
    OPERATIONAL = "governance.operational"
    MIGRATION = "governance.migration"
    INFRASTRUCTURE = "governance.infrastructure"
    SCHEMA = "governance.schema"

    @property
    def category(self) -> str:
        """Short category name."""
        return self.value.split(".")[-1]


# =============================================================================
# DATA CLASSIFICATION (POL-002 §4)
# =============================================================================


class DataClassification(str, Enum):
    """
    Data classification levels per POL-002 §4 and DAT-001.

    Security labels for field-level classification:
    - PUBLIC: No redaction or masking required
    - INTERNAL: Optional masking
    - CONFIDENTIAL: Required masking on logs
    - RESTRICTED: Required redaction on logs, masking on events
    - PII: Required redaction on logs/events, masking on responses
    - PHI: Strict redaction and masking on all channels
    - FINANCIAL: Required redaction on logs, Luhn-safe masking
    - SAFETY: Context-sensitive redaction
    - PROPRIETARY: Required redaction on logs, masking on events
    """

    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"
    PII = "PII"
    PHI = "PHI"
    FINANCIAL = "FINANCIAL"
    SAFETY = "SAFETY"
    PROPRIETARY = "PROPRIETARY"

    @property
    def requires_redaction_on_logs(self) -> bool:
        """Whether this classification requires redaction in logs."""
        return self in (
            DataClassification.RESTRICTED,
            DataClassification.PII,
            DataClassification.PHI,
            DataClassification.FINANCIAL,
            DataClassification.PROPRIETARY,
        )

    @property
    def requires_masking_on_events(self) -> bool:
        """Whether this classification requires masking in events."""
        return self in (
            DataClassification.RESTRICTED,
            DataClassification.PII,
            DataClassification.PHI,
            DataClassification.PROPRIETARY,
        )


# =============================================================================
# ESCALATION TRIGGERS (GOV-001 §7.1)
# =============================================================================


class EscalationTrigger(str, Enum):
    """
    Escalation trigger types per GOV-001 §7.1.

    Defines what causes an escalation:
    - LOW_CONFIDENCE: Rule confidence below threshold
    - POLICY_VIOLATION: Rule returns ESCALATE judgment
    - ANOMALY: Statistical deviation detected
    - MANUAL: User explicitly requests review
    - THRESHOLD_BREACH: Business limit exceeded
    - TIMEOUT: Approval window expired
    """

    LOW_CONFIDENCE = "low_confidence"
    POLICY_VIOLATION = "policy_violation"
    ANOMALY = "anomaly_detection"
    MANUAL = "manual_request"
    THRESHOLD_BREACH = "threshold_breach"
    TIMEOUT = "timeout"


# =============================================================================
# APPROVAL CHAIN PATTERNS (GOV-001 §6.2)
# =============================================================================


APPROVAL_CHAIN_PATTERNS = {
    "single": {
        "description": "Single Approver",
        "pattern": ["manager"],
        "use_case": "Low-risk changes",
    },
    "sequential": {
        "description": "Sequential Approval",
        "pattern": ["tech_lead", "manager", "director"],
        "use_case": "High-risk changes",
    },
    "parallel": {
        "description": "Parallel Approval",
        "pattern": [["security", "compliance"]],
        "use_case": "Multi-discipline review",
    },
    "hybrid": {
        "description": "Hybrid Approval",
        "pattern": ["tech_lead", ["security", "compliance"], "director"],
        "use_case": "Complex changes",
    },
}


# =============================================================================
# FAIL-SAFE CONFIGURATION (GOV-001 §5.2)
# =============================================================================


FAIL_SAFE_CONFIG = {
    "evaluation_timeout_ms": 500,
    "timeout_judgment": JudgmentType.BLOCK,
    "error_judgment": JudgmentType.BLOCK,
    "no_rules_judgment": JudgmentType.ALLOW,
    "database_unavailable_judgment": JudgmentType.BLOCK,
}


def get_fail_safe_judgment(condition: str) -> JudgmentType:
    """
    Get fail-safe judgment for a given condition.

    Per GOV-001 §5.2, fail-safe behavior ensures security
    when governance evaluation cannot complete normally.

    Args:
        condition: One of 'timeout', 'error', 'no_rules', 'database_unavailable'

    Returns:
        JudgmentType for the fail-safe condition
    """
    mapping = {
        "timeout": FAIL_SAFE_CONFIG["timeout_judgment"],
        "error": FAIL_SAFE_CONFIG["error_judgment"],
        "no_rules": FAIL_SAFE_CONFIG["no_rules_judgment"],
        "database_unavailable": FAIL_SAFE_CONFIG["database_unavailable_judgment"],
    }
    return mapping.get(condition, JudgmentType.BLOCK)


# =============================================================================
# REDACTION PATTERNS (POL-002 §6)
# =============================================================================


REDACTION_PATTERNS = {
    "email": "***@***",
    "phone": "***-***-****",
    "ssn": "***-**-****",
    "credit_card": "**** **** **** ****",
    "api_key": "***API_KEY***",
    "token": "***TOKEN***",
    "password": "***PASSWORD***",
    "default": "***REDACTED***",
}


def get_redaction_pattern(field_type: str) -> str:
    """
    Get redaction pattern for a field type per POL-002 §6.

    Args:
        field_type: Type of field (email, phone, ssn, etc.)

    Returns:
        Redaction pattern string
    """
    return REDACTION_PATTERNS.get(field_type, REDACTION_PATTERNS["default"])

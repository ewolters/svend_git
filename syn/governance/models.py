"""
Synara Governance Models (GOV-001/002, POL-001/002)
===================================================

Database models for governance layer implementing:
- GovernanceRule: Database-backed governance rules
- GovernanceJudgement: Immutable evaluation records
- ApprovalRequest: Approval workflow tracking
- ApprovalSignature: Electronic approval signatures

Standard:     GOV-001 §4-9, GOV-002 §models
Compliance:   ISO 27001:2022 A.5.1, SOC 2 CC8.1, 21 CFR Part 11
Version:      1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

from django.db import models, transaction
from django.db.models import Q
from django.utils import timezone

from syn.core.base_models import SynaraEntity
from syn.governance.types import (
    ApprovalStatus,
    InheritanceMode,
    JudgmentType,
    RuleClassification,
    RuleStatus,
    TimeoutClass,
    get_fail_safe_judgment,
)

logger = logging.getLogger(__name__)


# =============================================================================
# GOVERNANCE RULE (GOV-002 §models.GovernanceRule)
# =============================================================================


class GovernanceRule(SynaraEntity):
    """
    Database-backed governance rule per GOV-002.

    Governance rules define conditions and judgments for event evaluation.
    Rules are tenant-scoped and can be domain-scoped via ORG-001 integration.

    Standard: GOV-002 §models.GovernanceRule
    Compliance: ISO 27001:2022 A.5.1, SOC 2 CC8.1

    Example:
        >>> rule = GovernanceRule.objects.create(
        ...     tenant_id=tenant_uuid,
        ...     rule_id="gov-change-001",
        ...     name="Require approval for production changes",
        ...     trigger_event="deployment.production.requested",
        ...     conditions={"environment": "production"},
        ...     judgment_type=JudgmentType.ESCALATE.value,
        ...     status=RuleStatus.ACTIVE.value,
        ... )
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Unique rule instance identifier",
    )
    tenant_id = models.UUIDField(
        db_index=True,
        help_text="Tenant UUID for multi-tenant isolation (SEC-001 §5.2)",
    )
    org_domain_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="ORG-001 domain for domain-scoped rules",
    )

    # Rule identification
    # POL-001 §3.2 REQ-3.2-1: rule_id is tenant-scoped, not globally unique
    rule_id = models.CharField(
        max_length=50,
        db_index=True,
        help_text="Rule identifier scoped to tenant (e.g., gov-change-001)",
    )
    name = models.CharField(
        max_length=200,
        help_text="Human-readable rule name",
    )
    description = models.TextField(
        blank=True,
        default="",
        help_text="Detailed rule description",
    )

    # Rule configuration
    trigger_event = models.CharField(
        max_length=100,
        db_index=True,
        help_text="Event name that triggers this rule (EVT-001)",
    )
    conditions = models.JSONField(
        default=dict,
        help_text="Condition expression (JSON) for rule evaluation",
    )
    judgment_type = models.CharField(
        max_length=20,
        choices=[(j.value, j.value) for j in JudgmentType],
        help_text="Judgment when rule matches: ALLOW, BLOCK, ESCALATE",
    )
    judgment_message = models.TextField(
        blank=True,
        default="",
        help_text="Message explaining the judgment",
    )

    # Classification and priority
    classification = models.CharField(
        max_length=50,
        choices=[(c.value, c.value) for c in RuleClassification],
        default=RuleClassification.CHANGE.value,
        db_index=True,
        help_text="Rule classification (GOV-001 §4.2)",
    )
    priority = models.IntegerField(
        default=100,
        db_index=True,
        help_text="Rule priority (lower = higher priority)",
    )

    # Escalation configuration
    escalation_path = models.JSONField(
        default=list,
        help_text="Approval chain for ESCALATE judgments (GOV-001 §6.2)",
    )
    timeout_class = models.CharField(
        max_length=20,
        choices=[(t.value, t.value) for t in TimeoutClass],
        default=TimeoutClass.MEDIUM.value,
        help_text="Timeout class for approvals (GOV-001 §12)",
    )
    confidence_threshold = models.FloatField(
        default=0.8,
        help_text="Confidence threshold for ESCALATE (0.0-1.0)",
    )

    # Lifecycle
    status = models.CharField(
        max_length=20,
        choices=[(s.value, s.value) for s in RuleStatus],
        default=RuleStatus.DRAFT.value,
        db_index=True,
        help_text="Rule lifecycle status",
    )
    effective_from = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When rule becomes effective",
    )
    effective_until = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When rule expires",
    )

    # Domain inheritance (GOV-001 §13)
    inheritance_mode = models.CharField(
        max_length=20,
        choices=[(m.value, m.value) for m in InheritanceMode],
        default=InheritanceMode.INHERIT.value,
        help_text="How child domains inherit this rule",
    )

    # Audit fields
    created_by = models.CharField(
        max_length=255,
        help_text="User who created the rule",
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Creation timestamp",
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Last update timestamp",
    )
    version = models.IntegerField(
        default=1,
        help_text="Rule version for change tracking",
    )

    class Meta(SynaraEntity.Meta):
        db_table = "governance_rule"
        ordering = ["priority", "-created_at"]
        # POL-001 §3.2 REQ-3.2-1: rule_id is tenant-scoped with optional domain scope
        # Same rule_id can exist in different tenants or different domains
        unique_together = [["tenant_id", "rule_id", "org_domain_id"]]
        indexes = [
            models.Index(
                fields=["tenant_id", "trigger_event", "status"],
                name="gov_rule_tenant_event_status",
            ),
            models.Index(
                fields=["tenant_id", "classification", "status"],
                name="gov_rule_tenant_class_status",
            ),
        ]

    class SynaraMeta:
        event_domain = "syn.governance.governance_rule"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        return f"{self.rule_id}: {self.name}"

    @property
    def is_active(self) -> bool:
        """Check if rule is currently active and effective."""
        if self.status != RuleStatus.ACTIVE.value:
            return False

        now = timezone.now()
        if self.effective_from and now < self.effective_from:
            return False
        if self.effective_until and now > self.effective_until:
            return False

        return True

    def evaluate(
        self,
        event_payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[JudgmentType, float, str]:
        """
        Evaluate this rule against an event payload.

        Args:
            event_payload: Event data to evaluate
            context: Additional context (user, tenant, etc.)

        Returns:
            Tuple of (JudgmentType, confidence_score, reason)

        Standard: GOV-001 §5.1
        """
        if not self.is_active:
            return JudgmentType.ALLOW, 1.0, "Rule not active"

        try:
            # Simple condition matching (can be extended with expression language)
            matches = self._match_conditions(event_payload, context or {})

            if matches:
                judgment = JudgmentType(self.judgment_type)
                confidence = 1.0 if matches else 0.0
                reason = self.judgment_message or f"Rule {self.rule_id} matched"
                return judgment, confidence, reason
            else:
                return JudgmentType.ALLOW, 1.0, "Conditions not met"

        except Exception as e:
            # SEC-001 §9.19: Log full error internally, return generic message
            logger.error(
                f"Rule evaluation error for {self.rule_id}: {e}",
                exc_info=True,
            )
            return get_fail_safe_judgment("error"), 0.0, "Rule evaluation failed"

    def _match_conditions(
        self,
        payload: Dict[str, Any],
        context: Dict[str, Any],
    ) -> bool:
        """
        Match conditions against payload and context.

        Simple field-value matching. Can be extended with expression language.
        """
        if not self.conditions:
            return True  # No conditions = always match

        combined = {**payload, **context}

        for key, expected in self.conditions.items():
            actual = combined.get(key)

            # Handle various comparison operators
            if isinstance(expected, dict):
                op = expected.get("op", "eq")
                value = expected.get("value")

                if op == "eq" and actual != value:
                    return False
                elif op == "ne" and actual == value:
                    return False
                elif op == "in" and actual not in value:
                    return False
                elif op == "not_in" and actual in value:
                    return False
                elif op == "gt" and not (actual is not None and actual > value):
                    return False
                elif op == "gte" and not (actual is not None and actual >= value):
                    return False
                elif op == "lt" and not (actual is not None and actual < value):
                    return False
                elif op == "lte" and not (actual is not None and actual <= value):
                    return False
            else:
                # Simple equality
                if actual != expected:
                    return False

        return True

    @classmethod
    def get_active_rules_for_event(
        cls,
        tenant_id: uuid.UUID,
        event_name: str,
        org_domain_id: Optional[uuid.UUID] = None,
    ) -> models.QuerySet:
        """
        Get all active rules that match an event.

        Args:
            tenant_id: Tenant UUID
            event_name: Event name to match
            org_domain_id: Optional domain for domain-scoped rules

        Returns:
            QuerySet of matching GovernanceRule objects
        """
        now = timezone.now()
        queryset = cls.objects.filter(
            tenant_id=tenant_id,
            trigger_event=event_name,
            status=RuleStatus.ACTIVE.value,
        ).filter(
            Q(effective_from__isnull=True) | Q(effective_from__lte=now),
            Q(effective_until__isnull=True) | Q(effective_until__gte=now),
        )

        if org_domain_id:
            queryset = queryset.filter(Q(org_domain_id__isnull=True) | Q(org_domain_id=org_domain_id))
        else:
            queryset = queryset.filter(org_domain_id__isnull=True)

        return queryset.order_by("priority")


# =============================================================================
# GOVERNANCE JUDGEMENT (GOV-002 §models.GovernanceJudgement)
# =============================================================================


class GovernanceJudgement(SynaraEntity):
    """
    Immutable governance evaluation record per GOV-002.

    Records the result of governance rule evaluation for audit purposes.
    This model is immutable after creation per AUD-001 §5.3.

    Standard: GOV-002 §models.GovernanceJudgement
    Compliance: ISO 27001:2022, 21 CFR Part 11 §11.10

    Example:
        >>> judgement = GovernanceJudgement.record(
        ...     tenant_id=tenant_uuid,
        ...     correlation_id=corr_uuid,
        ...     event_name="deployment.production.requested",
        ...     event_payload={"environment": "production"},
        ...     rule=rule,
        ...     judgment=JudgmentType.ESCALATE,
        ...     confidence_score=0.95,
        ...     actor="user@example.com",
        ... )
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Unique judgement record identifier",
    )
    tenant_id = models.UUIDField(
        db_index=True,
        help_text="Tenant UUID for multi-tenant isolation",
    )

    # Event context
    correlation_id = models.UUIDField(
        db_index=True,
        help_text="Correlation ID for event tracing (SBL-001)",
    )
    parent_correlation_id = models.UUIDField(
        null=True,
        blank=True,
        help_text="Parent correlation ID for event lineage",
    )
    event_name = models.CharField(
        max_length=100,
        db_index=True,
        help_text="Event that was evaluated",
    )
    event_payload = models.JSONField(
        default=dict,
        help_text="Snapshot of event payload at evaluation time",
    )

    # Evaluation result
    rule = models.ForeignKey(
        GovernanceRule,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
        related_name="judgements",
        help_text="Rule that produced this judgment (nullable for default-permit per SBL-001 §3.5.6)",
    )
    judgment = models.CharField(
        max_length=20,
        choices=[(j.value, j.value) for j in JudgmentType],
        db_index=True,
        help_text="Judgment result: ALLOW, BLOCK, ESCALATE, MONITOR",
    )
    judgment_reason = models.TextField(
        help_text="Reason for the judgment",
    )
    confidence_score = models.FloatField(
        null=True,
        help_text="Confidence score (0.0-1.0) for the judgment",
    )
    obligations = models.JSONField(
        default=list,
        blank=True,
        help_text="Required actions from policy decision (POL-001 §3.1)",
    )

    # Audit fields
    actor = models.CharField(
        max_length=255,
        db_index=True,
        help_text="User or system that triggered the event",
    )
    judged_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        help_text="When the judgment was made",
    )

    # Integrity hash (21 CFR Part 11)
    integrity_hash = models.CharField(
        max_length=64,
        help_text="SHA-256 hash for integrity verification",
    )

    class Meta(SynaraEntity.Meta):
        db_table = "governance_judgement"
        ordering = ["-judged_at"]
        indexes = [
            models.Index(
                fields=["tenant_id", "event_name", "judged_at"],
                name="gov_judgement_tenant_event",
            ),
            models.Index(
                fields=["correlation_id", "judged_at"],
                name="gov_judgement_correlation",
            ),
        ]

    class SynaraMeta:
        event_domain = "syn.governance.governance_judgement"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        return f"{self.event_name} -> {self.judgment} ({self.correlation_id})"

    def save(self, *args, **kwargs):
        """
        Override save to enforce immutability.

        Per AUD-001 §5.3, governance judgements are immutable after creation.
        """
        if self.pk and GovernanceJudgement.objects.filter(pk=self.pk).exists():
            raise ValueError("GovernanceJudgement records are immutable after creation")

        # Compute integrity hash before save
        self.integrity_hash = self._compute_integrity_hash()

        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """
        Override delete to enforce immutability (SYS-200 INV-008).

        GovernanceJudgement records are immutable audit artifacts and cannot
        be deleted per AUD-001 §5.3 and 21 CFR Part 11 requirements.

        Raises:
            PermissionError: Always raised to prevent deletion
        """
        raise PermissionError(
            "GovernanceJudgement records are immutable audit artifacts and cannot be deleted. "
            "See AUD-001 §5.3 and 21 CFR Part 11 §11.10(e)."
        )

    def _compute_integrity_hash(self) -> str:
        """Compute SHA-256 integrity hash for this record."""
        data = {
            "tenant_id": str(self.tenant_id),
            "correlation_id": str(self.correlation_id),
            "event_name": self.event_name,
            "event_payload": self.event_payload,
            "rule_id": str(self.rule_id) if self.rule_id else None,
            "judgment": self.judgment,
            "judgment_reason": self.judgment_reason,
            "confidence_score": self.confidence_score,
            "actor": self.actor,
        }
        data_json = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_json.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify the integrity hash matches computed value."""
        expected = self._compute_integrity_hash()
        return self.integrity_hash == expected

    @classmethod
    def record(
        cls,
        tenant_id: uuid.UUID,
        correlation_id: uuid.UUID,
        event_name: str,
        event_payload: Dict[str, Any],
        rule: GovernanceRule,
        judgment: JudgmentType,
        judgment_reason: str,
        actor: str,
        confidence_score: Optional[float] = None,
        parent_correlation_id: Optional[uuid.UUID] = None,
    ) -> "GovernanceJudgement":
        """
        Record a governance judgment.

        Factory method for creating immutable judgement records.

        Args:
            tenant_id: Tenant UUID
            correlation_id: Event correlation ID
            event_name: Event that was evaluated
            event_payload: Event payload snapshot
            rule: Rule that produced the judgment
            judgment: Judgment result
            judgment_reason: Reason for judgment
            actor: User/system that triggered the event
            confidence_score: Optional confidence score
            parent_correlation_id: Optional parent correlation

        Returns:
            Created GovernanceJudgement instance
        """
        return cls.objects.create(
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            parent_correlation_id=parent_correlation_id,
            event_name=event_name,
            event_payload=event_payload,
            rule=rule,
            judgment=judgment.value if isinstance(judgment, JudgmentType) else judgment,
            judgment_reason=judgment_reason,
            confidence_score=confidence_score,
            actor=actor,
        )


# =============================================================================
# APPROVAL REQUEST (GOV-001 §6)
# =============================================================================


class ApprovalRequest(SynaraEntity):
    """
    Approval workflow request per GOV-001 §6.

    Tracks approval requests created when governance rules return ESCALATE.
    Integrates with binder signatures per 21 CFR Part 11.

    Standard: GOV-001 §6, BIND-001 §signatures
    Compliance: 21 CFR Part 11 §11.50, ISO 27001:2022
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Unique approval request identifier",
    )
    tenant_id = models.UUIDField(
        db_index=True,
        help_text="Tenant UUID for multi-tenant isolation",
    )
    org_domain_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="ORG-001 domain for domain-scoped approvals",
    )

    # Link to governance judgement
    judgement = models.ForeignKey(
        GovernanceJudgement,
        on_delete=models.PROTECT,
        related_name="approval_requests",
        help_text="Judgement that triggered this approval",
    )

    # Request context
    correlation_id = models.UUIDField(
        db_index=True,
        help_text="Correlation ID for tracing",
    )
    request_type = models.CharField(
        max_length=100,
        help_text="Type of approval request",
    )
    request_title = models.CharField(
        max_length=200,
        help_text="Human-readable request title",
    )
    request_description = models.TextField(
        blank=True,
        default="",
        help_text="Detailed request description",
    )
    request_payload = models.JSONField(
        default=dict,
        help_text="Original request data",
    )

    # Approval chain (GOV-001 §6.2)
    approval_chain = models.JSONField(
        default=list,
        help_text="Approval chain pattern (sequential/parallel/hybrid)",
    )
    current_step = models.IntegerField(
        default=0,
        help_text="Current position in approval chain",
    )

    # Status
    status = models.CharField(
        max_length=20,
        choices=[(s.value, s.value) for s in ApprovalStatus],
        default=ApprovalStatus.PENDING.value,
        db_index=True,
        help_text="Current approval status",
    )

    # Timeout (GOV-001 §12)
    timeout_class = models.CharField(
        max_length=20,
        choices=[(t.value, t.value) for t in TimeoutClass],
        default=TimeoutClass.MEDIUM.value,
        help_text="Timeout class for this request",
    )
    expires_at = models.DateTimeField(
        db_index=True,
        help_text="When this request expires",
    )

    # Requester
    requester_id = models.CharField(
        max_length=255,
        help_text="User who requested approval",
    )
    requester_name = models.CharField(
        max_length=255,
        help_text="Display name of requester",
    )

    # Resolution
    resolved_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When request was resolved",
    )
    resolved_by = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="User who resolved the request",
    )
    resolution_notes = models.TextField(
        blank=True,
        default="",
        help_text="Notes on resolution",
    )

    # Audit
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Creation timestamp",
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Last update timestamp",
    )

    class Meta(SynaraEntity.Meta):
        db_table = "governance_approval_request"
        ordering = ["-created_at"]
        indexes = [
            models.Index(
                fields=["tenant_id", "status", "expires_at"],
                name="gov_approval_tenant_status",
            ),
            models.Index(
                fields=["correlation_id"],
                name="gov_approval_correlation",
            ),
        ]

    class SynaraMeta:
        event_domain = "syn.governance.approval_request"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        return f"{self.request_title} ({self.status})"

    def save(self, *args, **kwargs):
        """Set expiration based on timeout class if not set."""
        if not self.expires_at:
            timeout = TimeoutClass(self.timeout_class)
            self.expires_at = timezone.now() + timedelta(hours=timeout.default_hours)
        super().save(*args, **kwargs)

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        return timezone.now() > self.expires_at

    @property
    def is_pending(self) -> bool:
        """Check if request is still pending."""
        return self.status == ApprovalStatus.PENDING.value

    def get_current_approvers(self) -> List[str]:
        """
        Get approvers for the current step.

        Handles sequential, parallel, and hybrid chains per GOV-001 §6.2.
        """
        if self.current_step >= len(self.approval_chain):
            return []

        current = self.approval_chain[self.current_step]

        if isinstance(current, list):
            # Parallel approval step
            return current
        else:
            # Single approver
            return [current]

    @transaction.atomic
    def approve(
        self,
        approver_id: str,
        approver_name: str,
        notes: str = "",
    ) -> bool:
        """
        Record an approval signature.

        Args:
            approver_id: ID of approving user
            approver_name: Display name of approver
            notes: Optional approval notes

        Returns:
            True if all approvals complete, False if more needed
        """
        if not self.is_pending:
            raise ValueError(f"Cannot approve request in {self.status} status")

        if self.is_expired:
            self.status = ApprovalStatus.EXPIRED.value
            self.save()
            raise ValueError("Approval request has expired")

        # Create signature
        ApprovalSignature.objects.create(
            tenant_id=self.tenant_id,
            approval_request=self,
            signer_id=approver_id,
            signer_name=approver_name,
            action="APPROVE",
            notes=notes,
        )

        # Check if current step is complete
        current_approvers = self.get_current_approvers()
        self.signatures.filter(
            action="APPROVE",
            signer_id__in=[approver_id],  # Check this approver
        )

        # For parallel steps, check if all approvers have signed
        if isinstance(self.approval_chain[self.current_step], list):
            signed_ids = set(self.signatures.filter(action="APPROVE").values_list("signer_id", flat=True))
            if not all(a in signed_ids for a in current_approvers):
                return False  # Still waiting for parallel approvers

        # Move to next step
        self.current_step += 1

        if self.current_step >= len(self.approval_chain):
            # All steps complete
            self.status = ApprovalStatus.APPROVED.value
            self.resolved_at = timezone.now()
            self.resolved_by = approver_id
            self.resolution_notes = notes
            self.save()
            return True
        else:
            self.save()
            return False

    @transaction.atomic
    def reject(
        self,
        rejector_id: str,
        rejector_name: str,
        reason: str,
    ) -> None:
        """
        Reject the approval request.

        Args:
            rejector_id: ID of rejecting user
            rejector_name: Display name of rejector
            reason: Reason for rejection
        """
        if not self.is_pending:
            raise ValueError(f"Cannot reject request in {self.status} status")

        ApprovalSignature.objects.create(
            tenant_id=self.tenant_id,
            approval_request=self,
            signer_id=rejector_id,
            signer_name=rejector_name,
            action="REJECT",
            notes=reason,
        )

        self.status = ApprovalStatus.REJECTED.value
        self.resolved_at = timezone.now()
        self.resolved_by = rejector_id
        self.resolution_notes = reason
        self.save()

    @transaction.atomic
    def cancel(self, canceller_id: str, reason: str = "") -> None:
        """
        Cancel the approval request.

        Args:
            canceller_id: ID of user cancelling
            reason: Optional cancellation reason
        """
        if not self.is_pending:
            raise ValueError(f"Cannot cancel request in {self.status} status")

        self.status = ApprovalStatus.CANCELLED.value
        self.resolved_at = timezone.now()
        self.resolved_by = canceller_id
        self.resolution_notes = reason
        self.save()

    @classmethod
    def check_expired_requests(cls) -> int:
        """
        Check and expire overdue approval requests.

        Called by scheduled task per GOV-001 §12.3.

        Returns:
            Number of requests expired
        """
        expired = cls.objects.filter(
            status=ApprovalStatus.PENDING.value,
            expires_at__lt=timezone.now(),
        )
        count = expired.update(
            status=ApprovalStatus.EXPIRED.value,
            resolved_at=timezone.now(),
            resolution_notes="Expired due to timeout",
        )
        return count


# =============================================================================
# APPROVAL SIGNATURE (GOV-001 §9, 21 CFR Part 11)
# =============================================================================


class ApprovalSignature(SynaraEntity):
    """
    Electronic approval signature per 21 CFR Part 11.

    Records individual approval/rejection signatures on approval requests.

    Standard: GOV-001 §9, BIND-001 §signatures
    Compliance: 21 CFR Part 11 §11.50, §11.70, §11.100
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Unique signature identifier",
    )
    approval_request = models.ForeignKey(
        ApprovalRequest,
        on_delete=models.PROTECT,
        related_name="signatures",
        help_text="Approval request being signed",
    )

    # Signer identification (21 CFR Part 11 §11.50)
    signer_id = models.CharField(
        max_length=255,
        help_text="Unique identifier of signer",
    )
    signer_name = models.CharField(
        max_length=255,
        help_text="Printed name of signer",
    )
    signer_role = models.CharField(
        max_length=100,
        blank=True,
        default="",
        help_text="Role of signer (manager, tech_lead, etc.)",
    )

    # Signature action
    action = models.CharField(
        max_length=20,
        choices=[("APPROVE", "APPROVE"), ("REJECT", "REJECT")],
        help_text="Signature action",
    )
    notes = models.TextField(
        blank=True,
        default="",
        help_text="Signature notes or rejection reason",
    )

    # Cryptographic signature (21 CFR Part 11 §11.70)
    signature_hash = models.CharField(
        max_length=64,
        help_text="SHA-256 hash of signature data",
    )

    # Timestamps
    signed_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        help_text="When signature was applied",
    )

    class Meta(SynaraEntity.Meta):
        db_table = "governance_approval_signature"
        ordering = ["signed_at"]
        indexes = [
            models.Index(
                fields=["approval_request", "signer_id"],
                name="gov_sig_request_signer",
            ),
        ]

    class SynaraMeta:
        event_domain = "syn.governance.approval_signature"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        return f"{self.signer_name} {self.action} ({self.signed_at})"

    def save(self, *args, **kwargs):
        """Compute signature hash before save."""
        if not self.signature_hash:
            self.signature_hash = self._compute_signature_hash()
        super().save(*args, **kwargs)

    def _compute_signature_hash(self) -> str:
        """Compute SHA-256 hash for signature per 21 CFR Part 11."""
        data = {
            "approval_request_id": str(self.approval_request_id),
            "signer_id": self.signer_id,
            "signer_name": self.signer_name,
            "action": self.action,
            "notes": self.notes,
        }
        data_json = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_json.encode("utf-8")).hexdigest()

    def verify(self) -> bool:
        """Verify signature hash integrity."""
        expected = self._compute_signature_hash()
        return self.signature_hash == expected


# =============================================================================
# TIMEOUT CONFIGURATION (GOV-001 §12)
# =============================================================================


class TimeoutConfiguration(SynaraEntity):
    """
    Timeout configuration per GOV-001 §12.

    Allows tenant-specific timeout class overrides.
    """

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    tenant_id = models.UUIDField(
        db_index=True,
        help_text="Tenant UUID",
    )
    timeout_class = models.CharField(
        max_length=20,
        choices=[(t.value, t.value) for t in TimeoutClass],
        help_text="Timeout class being configured",
    )
    default_hours = models.IntegerField(
        help_text="Default timeout in hours",
    )
    min_hours = models.IntegerField(
        help_text="Minimum timeout in hours",
    )
    max_hours = models.IntegerField(
        help_text="Maximum timeout in hours",
    )
    on_timeout = models.CharField(
        max_length=20,
        choices=[(j.value, j.value) for j in JudgmentType],
        default=JudgmentType.ESCALATE.value,
        help_text="Judgment when timeout occurs",
    )
    escalation_path = models.JSONField(
        default=list,
        help_text="Escalation path on timeout",
    )
    notification_channels = models.JSONField(
        default=list,
        help_text="Channels for timeout notifications",
    )

    class Meta(SynaraEntity.Meta):
        db_table = "governance_timeout_config"
        unique_together = [["tenant_id", "timeout_class"]]

    class SynaraMeta:
        event_domain = "syn.governance.timeout_configuration"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        return f"{self.timeout_class} ({self.default_hours}h)"

"""Loop mechanism models — LOOP-001 §3, §4.

Three formally defined mechanisms:
- Signal (§3.1): event that demands attention — entry point to the loop
- Commitment (§3.3): bilateral accountability contract (David Mann)
- ModeTransition (§3.2): structural linkage between artifacts across modes

QMS Policy service (§4):
- QMSPolicy: org-defined rules as structured data
- PolicyCondition: surfaced condition from policy evaluation
"""

import uuid

from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models
from django.utils import timezone

from syn.core.base_models import SynaraEntity

# =============================================================================
# SIGNAL (LOOP-001 §3.1)
# =============================================================================


class Signal(SynaraEntity):
    """An event that demands attention — entry point to the loop.

    Signals are NEVER auto-created in v1. A human or a policy-defined
    threshold creates them. The system surfaces CONDITIONS (via
    PolicyCondition); a human promotes them to Signals.

    LOOP-001 §3.1
    """

    class SourceType(models.TextChoices):
        SPC_VIOLATION = "spc_violation", "SPC Out-of-Control"
        PC_THRESHOLD = "pc_threshold", "PC Threshold Breach"
        CUSTOMER_COMPLAINT = "customer_complaint", "Customer Complaint"
        AUDIT_FINDING = "audit_finding", "Audit Finding"
        FORCED_FAILURE_MISS = "forced_failure_miss", "Forced Failure Detection Miss"
        FRONTIER_CARD = "frontier_card", "Frontier Card Finding"
        OPERATOR_REPORT = "operator_report", "Operator Report"
        MANAGEMENT_REVIEW = "management_review", "Management Review Action"
        TRAINING_REFLECTION = "training_reflection", "Training Reflection Pattern"
        SUPPLIER_ISSUE = "supplier_issue", "Supplier Quality Issue"
        FIELD_RETURN = "field_return", "Field Return / Warranty"
        OTHER = "other", "Other"

    class Severity(models.TextChoices):
        INFO = "info", "Information"
        WARNING = "warning", "Warning"
        CRITICAL = "critical", "Critical"

    class TriageState(models.TextChoices):
        UNTRIAGED = "untriaged", "Untriaged"
        ACKNOWLEDGED = "acknowledged", "Acknowledged"
        INVESTIGATING = "investigating", "Investigating"
        RESOLVED = "resolved", "Resolved"
        DISMISSED = "dismissed", "Dismissed"

    tenant = models.ForeignKey(
        "core.Tenant",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="signals",
    )
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True, default="")
    source_type = models.CharField(
        max_length=30,
        choices=SourceType.choices,
    )
    severity = models.CharField(
        max_length=10,
        choices=Severity.choices,
        default=Severity.WARNING,
    )
    triage_state = models.CharField(
        max_length=20,
        choices=TriageState.choices,
        default=TriageState.UNTRIAGED,
        db_index=True,
    )

    # Generic FK to the source record (SPC result, complaint, audit finding, etc.)
    source_content_type = models.ForeignKey(
        ContentType,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="+",
    )
    source_object_id = models.UUIDField(null=True, blank=True)
    source_record = GenericForeignKey("source_content_type", "source_object_id")

    # Resolution
    resolved_by_investigation = models.ForeignKey(
        "core.Investigation",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="source_signals",
    )
    dismissed_reason = models.TextField(blank=True, default="")

    # Provenance
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="created_signals",
    )
    resolved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "loop_signal"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["triage_state", "-created_at"]),
            models.Index(fields=["tenant", "triage_state"]),
        ]

    def __str__(self):
        return f"[{self.severity}] {self.title}"

    def acknowledge(self):
        """Mark signal as seen — prevents duplicates."""
        if self.triage_state == self.TriageState.UNTRIAGED:
            self.triage_state = self.TriageState.ACKNOWLEDGED
            self.save(update_fields=["triage_state", "updated_at"])

    def link_investigation(self, investigation):
        """Link to an investigation — signal is being worked."""
        self.triage_state = self.TriageState.INVESTIGATING
        self.resolved_by_investigation = investigation
        self.save(
            update_fields=[
                "triage_state",
                "resolved_by_investigation",
                "updated_at",
            ]
        )

    def resolve(self, investigation=None):
        """Mark signal as resolved."""
        self.triage_state = self.TriageState.RESOLVED
        self.resolved_at = timezone.now()
        if investigation:
            self.resolved_by_investigation = investigation
        self.save(
            update_fields=[
                "triage_state",
                "resolved_at",
                "resolved_by_investigation",
                "updated_at",
            ]
        )

    def dismiss(self, reason):
        """Dismiss signal with required reason (audit trail)."""
        if not reason or not reason.strip():
            raise ValueError("Dismissing a signal requires a reason")
        self.triage_state = self.TriageState.DISMISSED
        self.dismissed_reason = reason
        self.resolved_at = timezone.now()
        self.save(
            update_fields=[
                "triage_state",
                "dismissed_reason",
                "resolved_at",
                "updated_at",
            ]
        )


# =============================================================================
# COMMITMENT (LOOP-001 §3.3)
# =============================================================================


class Commitment(SynaraEntity):
    """A bilateral accountability contract — person commits to deliverable
    given preconditions from the organization.

    Per David Mann's daily management system: the commitment is a contract.
    The owner commits to the work. The organization commits to the
    preconditions. If preconditions are not met, the commitment is blocked,
    not late.

    LOOP-001 §3.3
    """

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        IN_PROGRESS = "in_progress", "In Progress"
        FULFILLED = "fulfilled", "Fulfilled"
        BROKEN = "broken", "Broken"
        CANCELLED = "cancelled", "Cancelled"

    class TransitionType(models.TextChoices):
        """Mode transition triggered when this commitment is fulfilled."""

        REVISE_DOCUMENT = "revise_document", "Revise Document"
        CREATE_DOCUMENT = "create_document", "Create Document"
        ADD_CONTROL = "add_control", "Add Control (FMIS)"
        TRAIN = "train", "Training"
        PROCESS_CONFIRMATION = "process_confirmation", "Schedule PC"
        FORCED_FAILURE = "forced_failure", "Schedule Forced Failure"
        MONITOR = "monitor", "Set Up Monitoring"
        AUDIT_ZONE = "audit_zone", "Schedule Zone Audit"
        FIRST_ARTICLE = "first_article", "First Article Inspection"

    class SourceType(models.TextChoices):
        INVESTIGATION = "investigation", "Investigation"
        MANAGEMENT_REVIEW = "management_review", "Management Review"
        KAIZEN_CHARTER = "kaizen_charter", "Kaizen Charter"
        MANUAL = "manual", "Manual"

    tenant = models.ForeignKey(
        "core.Tenant",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="commitments",
    )

    # The contract
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="commitments",
        help_text="Person who commits to the deliverable",
    )
    title = models.CharField(
        max_length=300,
        help_text="Specific deliverable — not a vague task",
    )
    description = models.TextField(blank=True, default="")
    due_date = models.DateField(
        help_text="When the deliverable is due. Required — a commitment without a date is a wish.",
    )
    preconditions = models.JSONField(
        default=list,
        blank=True,
        help_text='What the org must provide: [{"description": "...", "status": "pending|met|unmet", "blocked_by_commitment_id": "uuid|null"}]',
    )

    # Status
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.OPEN,
        db_index=True,
    )

    # Mode transition (nullable — not all commitments trigger transitions)
    transition_type = models.CharField(
        max_length=30,
        choices=TransitionType.choices,
        blank=True,
        default="",
        help_text="If set, fulfilling this commitment triggers a mode transition (LOOP-001 §3.2)",
    )

    # Source
    source_type = models.CharField(
        max_length=30,
        choices=SourceType.choices,
        default=SourceType.MANUAL,
    )
    source_investigation = models.ForeignKey(
        "core.Investigation",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="commitments",
    )

    # Target artifact created on fulfillment (populated by mode transition)
    target_content_type = models.ForeignKey(
        ContentType,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="+",
    )
    target_object_id = models.UUIDField(null=True, blank=True)
    target_artifact = GenericForeignKey("target_content_type", "target_object_id")

    # Provenance
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="created_commitments",
    )
    fulfilled_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "loop_commitment"
        ordering = ["due_date", "-created_at"]
        indexes = [
            models.Index(fields=["status", "due_date"]),
            models.Index(fields=["owner", "status"]),
            models.Index(fields=["tenant", "status"]),
        ]

    def __str__(self):
        return f"{self.title} ({self.owner} → {self.due_date})"

    def save(self, *args, **kwargs):
        if not self.due_date:
            raise ValueError(
                "A commitment without a due date is not a commitment — it is a wish. Set due_date before saving."
            )
        super().save(*args, **kwargs)

    @property
    def is_overdue(self):
        from datetime import date

        return self.status in (self.Status.OPEN, self.Status.IN_PROGRESS) and self.due_date < date.today()

    @property
    def is_blocked(self):
        """True if any precondition is unmet or blocked."""
        if not self.preconditions:
            return False
        return any(p.get("status") in ("pending", "unmet") for p in self.preconditions)

    def fulfill(self, user):
        """Mark commitment as fulfilled. If transition_type is set,
        the caller is responsible for creating the ModeTransition
        and target artifact.
        """
        self.status = self.Status.FULFILLED
        self.fulfilled_at = timezone.now()
        self.save(update_fields=["status", "fulfilled_at", "updated_at"])

    def mark_broken(self, reason=""):
        """Mark commitment as broken — a system signal, not punishment."""
        self.status = self.Status.BROKEN
        self.description = f"{self.description}\n\n---\nBroken: {reason}" if reason else self.description
        self.save(update_fields=["status", "description", "updated_at"])


# =============================================================================
# MODE TRANSITION (LOOP-001 §3.2)
# =============================================================================


class ModeTransition(models.Model):
    """Structural linkage between artifacts across modes — immutable.

    When a Commitment with a transition_type is fulfilled, the system
    creates the corresponding artifact and records this transition.
    ModeTransitions are append-only audit records.

    LOOP-001 §3.2
    """

    class TransitionType(models.TextChoices):
        REVISE_DOCUMENT = "revise_document", "Investigate → Standardize (revise)"
        CREATE_DOCUMENT = "create_document", "Investigate → Standardize (create)"
        ADD_CONTROL = "add_control", "Investigate → Standardize (FMIS)"
        TRAIN = "train", "Standardize → Verify (training)"
        PROCESS_CONFIRMATION = "process_confirmation", "Standardize → Verify (PC)"
        FORCED_FAILURE = "forced_failure", "Standardize → Verify (FFT)"
        MONITOR = "monitor", "Standardize → Verify (SPC)"
        AUDIT_ZONE = "audit_zone", "Standardize → Verify (frontier)"
        FIRST_ARTICLE = "first_article", "Standardize → Verify (FAI)"

    class Mode(models.TextChoices):
        INVESTIGATE = "investigate", "Investigate"
        STANDARDIZE = "standardize", "Standardize"
        VERIFY = "verify", "Verify"

    # Transition types → mode mapping
    MODE_MAP = {
        "revise_document": ("investigate", "standardize"),
        "create_document": ("investigate", "standardize"),
        "add_control": ("investigate", "standardize"),
        "train": ("standardize", "verify"),
        "process_confirmation": ("standardize", "verify"),
        "forced_failure": ("standardize", "verify"),
        "monitor": ("standardize", "verify"),
        "audit_zone": ("standardize", "verify"),
        "first_article": ("standardize", "verify"),
    }

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    transition_type = models.CharField(
        max_length=30,
        choices=TransitionType.choices,
    )
    from_mode = models.CharField(max_length=20, choices=Mode.choices)
    to_mode = models.CharField(max_length=20, choices=Mode.choices)

    # Source artifact (what produced this transition)
    source_content_type = models.ForeignKey(
        ContentType,
        on_delete=models.CASCADE,
        related_name="+",
    )
    source_object_id = models.UUIDField()
    source_artifact = GenericForeignKey("source_content_type", "source_object_id")

    # Target artifact (what was created by this transition)
    target_content_type = models.ForeignKey(
        ContentType,
        on_delete=models.CASCADE,
        related_name="+",
    )
    target_object_id = models.UUIDField()
    target_artifact = GenericForeignKey("target_content_type", "target_object_id")

    # The commitment that triggered this (nullable for auto-triggered like `train`)
    triggered_by = models.ForeignKey(
        Commitment,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="transitions",
    )

    # Immutable timestamp
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "loop_mode_transition"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.from_mode} → {self.to_mode} ({self.transition_type})"

    def save(self, *args, **kwargs):
        # Auto-set from_mode and to_mode from transition_type
        if self.transition_type and not self.from_mode:
            modes = self.MODE_MAP.get(self.transition_type)
            if modes:
                self.from_mode, self.to_mode = modes

        # Immutability: reject updates on existing records
        if self.pk and ModeTransition.objects.filter(pk=self.pk).exists():
            raise ValueError("ModeTransition records are immutable. Create a new record instead.")
        super().save(*args, **kwargs)


# =============================================================================
# QMS POLICY (LOOP-001 §4)
# =============================================================================


class QMSPolicy(SynaraEntity):
    """Org-defined rule as structured data — informs system behavior.

    Policies are machine-readable. Human-readable documentation is
    auto-generated as a ControlledDocument (the artifact for auditors).
    No drift between what's enforced and what's documented.

    LOOP-001 §4
    """

    class Scope(models.TextChoices):
        PROCESS_CONFIRMATION = "process_confirmation", "Process Confirmation"
        SPC_MONITORING = "spc_monitoring", "SPC Monitoring"
        FORCED_FAILURE = "forced_failure", "Forced Failure"
        COMPLAINT = "complaint", "Customer Complaint"
        TRAINING = "training", "Training"
        REVIEW_FREQUENCY = "review_frequency", "Review Frequency"
        TRAINING_COVERAGE = "training_coverage", "Training Coverage"
        RECURRENCE = "recurrence", "Recurrence Detection"
        COMMITMENT_FULFILLMENT = "commitment_fulfillment", "Commitment Fulfillment"
        CALIBRATION = "calibration", "Calibration"
        VERIFICATION_SCHEDULE = "verification_schedule", "Verification Schedule"
        FMIS = "fmis", "FMIS Methodology"
        INVESTIGATION = "investigation", "Investigation"
        CAPA_REPORT = "capa_report", "CAPA Report"
        SEVERITY_DISPLAY = "severity_display", "Severity Display"

    tenant = models.ForeignKey(
        "core.Tenant",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="qms_policies",
    )
    scope = models.CharField(
        max_length=30,
        choices=Scope.choices,
        db_index=True,
    )
    rule_key = models.CharField(
        max_length=100,
        help_text="Machine-readable key: e.g., pc.retraining_threshold",
    )
    parameters = models.JSONField(
        default=dict,
        help_text="Structured rule configuration — typed parameters",
    )
    linked_standard = models.CharField(
        max_length=100,
        blank=True,
        default="",
        help_text="Which ISO/IATF/AS clause this policy satisfies",
    )
    effective_date = models.DateField()
    approved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="approved_policies",
    )
    version = models.PositiveIntegerField(default=1)
    is_active = models.BooleanField(default=True, db_index=True)

    # Auto-generated policy document for auditors
    controlled_document = models.ForeignKey(
        "agents_api.ControlledDocument",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="qms_policies",
    )

    class Meta:
        db_table = "loop_qms_policy"
        ordering = ["-effective_date"]
        indexes = [
            models.Index(fields=["tenant", "scope", "is_active"]),
            models.Index(fields=["rule_key", "is_active"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["tenant", "rule_key", "version"],
                name="unique_policy_version",
            ),
        ]

    def __str__(self):
        return f"{self.rule_key} v{self.version} ({'active' if self.is_active else 'inactive'})"


# =============================================================================
# POLICY CONDITION (LOOP-001 §4.6.4)
# =============================================================================


class PolicyCondition(SynaraEntity):
    """A surfaced condition from policy evaluation — NOT a Signal.

    The system noticed something. A human reviews conditions and decides
    which warrant investigation (promoting them to Signals).

    LOOP-001 §4.6.4
    """

    class ConditionType(models.TextChoices):
        RETRAINING_NEEDED = "retraining_needed", "Retraining Needed"
        STANDARD_REVISION_NEEDED = "standard_revision_needed", "Standard Revision Needed"
        REVIEW_OVERDUE = "review_overdue", "Review Overdue"
        RECURRENCE_DETECTED = "recurrence_detected", "Recurrence Detected"
        DETECTION_GAP = "detection_gap", "Detection Gap"
        THRESHOLD_BREACH = "threshold_breach", "Threshold Breach"
        SCHEDULE_MISS = "schedule_miss", "Schedule Miss"
        CALIBRATION_OVERDUE = "calibration_overdue", "Calibration Overdue"
        COMMITMENT_OVERDUE = "commitment_overdue", "Commitment Overdue"
        QUALIFICATION_GAP = "qualification_gap", "Investigator Qualification Gap"

    class Severity(models.TextChoices):
        INFO = "info", "Information"
        WARNING = "warning", "Warning"
        CRITICAL = "critical", "Critical"

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        ACKNOWLEDGED = "acknowledged", "Acknowledged"
        RESOLVED = "resolved", "Resolved"
        DISMISSED = "dismissed", "Dismissed"

    tenant = models.ForeignKey(
        "core.Tenant",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="policy_conditions",
    )
    condition_type = models.CharField(
        max_length=30,
        choices=ConditionType.choices,
    )
    severity = models.CharField(
        max_length=10,
        choices=Severity.choices,
        default=Severity.WARNING,
    )
    title = models.CharField(max_length=300)
    context = models.JSONField(
        default=dict,
        help_text="Structured data: affected entities, measurements, thresholds, evidence",
    )

    # Which policy rule generated this condition
    policy_rule = models.ForeignKey(
        QMSPolicy,
        on_delete=models.CASCADE,
        related_name="conditions",
    )

    # Event that triggered evaluation (null for aggregate sweep)
    source_event = models.CharField(
        max_length=100,
        blank=True,
        default="",
        help_text="ToolEventBus event name, or empty for PolicySweepEvaluator",
    )

    # Resolution
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE,
        db_index=True,
    )
    resolved_by_signal = models.ForeignKey(
        Signal,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="source_conditions",
    )
    dismissed_reason = models.TextField(blank=True, default="")
    resolved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "loop_policy_condition"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "-created_at"]),
            models.Index(fields=["tenant", "status"]),
        ]

    def __str__(self):
        return f"[{self.severity}] {self.title}"

    def promote_to_signal(self, user, investigation=None):
        """Create a Signal from this condition. Returns the new Signal."""
        signal = Signal.objects.create(
            tenant=self.tenant,
            title=self.title,
            description=f"Promoted from policy condition: {self.context}",
            source_type=Signal.SourceType.OTHER,
            severity=self.severity,
            created_by=user,
        )
        self.status = self.Status.RESOLVED
        self.resolved_by_signal = signal
        self.resolved_at = timezone.now()
        self.save(
            update_fields=[
                "status",
                "resolved_by_signal",
                "resolved_at",
                "updated_at",
            ]
        )
        return signal

    def dismiss(self, reason):
        """Dismiss with required reason (audit trail)."""
        if not reason or not reason.strip():
            raise ValueError("Dismissing a condition requires a reason")
        self.status = self.Status.DISMISSED
        self.dismissed_reason = reason
        self.resolved_at = timezone.now()
        self.save(
            update_fields=[
                "status",
                "dismissed_reason",
                "resolved_at",
                "updated_at",
            ]
        )

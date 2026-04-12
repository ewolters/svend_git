"""Loop mechanism models — LOOP-001 §3, §4, §6, §7, §8, §11, §16.

Three formally defined mechanisms:
- Signal (§3.1): event that demands attention — entry point to the loop
- Commitment (§3.3): bilateral accountability contract (David Mann)
- ModeTransition (§3.2): structural linkage between artifacts across modes

QMS Policy service (§4):
- QMSPolicy: org-defined rules as structured data
- PolicyCondition: surfaced condition from policy evaluation

Verify mode (§6, §7):
- TrainingReflection (§6.2): hansei response from operator
- ProcessConfirmation (§7.1): gemba observation with diagnostic matrix
- PCObservationItem: individual step observation within a PC
- ForcedFailureTest (§7.2): detection verification

Investigation workspace (§16.3):
- InvestigationEntry: notebook-style entries (narrative, tool output, evidence)

FMIS (§8):
- FMIS: investigation-native FMEA container
- FMISRow: failure mode with Bayesian S/O/D posteriors

Auditor Portal (§11):
- AuditorPortalToken: time-limited, multi-use token for external auditors
"""

import secrets
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
        GRAPH_STALENESS = "graph_staleness", "Graph Edge Staleness"
        GRAPH_CONTRADICTION = "graph_contradiction", "Graph Edge Contradiction"
        GRAPH_EXPANSION = "graph_expansion", "Graph Expansion Signal"
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
    triaged_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When this signal was first triaged — feeds resolution velocity metric (OLR-001 §13.4)",
    )

    # Graph linkage (OLR-001) — which process nodes this signal relates to
    linked_process_node_ids = models.JSONField(
        default=list,
        blank=True,
        help_text="UUIDs of ProcessNode records this signal relates to",
    )

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

    # Linked artifacts — UUIDs of outputs produced for this commitment
    linked_artifacts = models.JSONField(
        default=list,
        blank=True,
        help_text="List of {type, id, title, created_at} for artifacts linked to this commitment",
    )

    # Resource needs — what the owner needs to fulfill this
    resource_needs = models.JSONField(
        default=list,
        blank=True,
        help_text="List of {description, status: needed|available|blocked} resource requirements",
    )

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


class CommitmentResource(models.Model):
    """Employee assignment to a Loop commitment with role and availability.

    Parallels Hoshin's ResourceCommitment but for Loop commitments instead
    of Hoshin projects. Same Employee model, same availability checking.
    QMS-002 §2.2.
    """

    ROLE_CHOICES = [
        ("facilitator", "Facilitator"),
        ("team_member", "Team Member"),
        ("sponsor", "Sponsor"),
        ("process_owner", "Process Owner"),
        ("subject_expert", "Subject Expert"),
    ]

    STATUS_CHOICES = [
        ("requested", "Requested"),
        ("confirmed", "Confirmed"),
        ("declined", "Declined"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    commitment = models.ForeignKey(
        Commitment,
        on_delete=models.CASCADE,
        related_name="resources",
    )
    employee = models.ForeignKey(
        "agents_api.Employee",
        on_delete=models.CASCADE,
        related_name="loop_commitments",
    )
    role = models.CharField(max_length=30, choices=ROLE_CHOICES)
    hours_needed = models.DecimalField(max_digits=5, decimal_places=1, null=True, blank=True)
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="requested")
    requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="requested_loop_resources",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "loop_commitment_resource"

    def __str__(self):
        return f"{self.employee.name} → {self.commitment} ({self.role})"

    @classmethod
    def check_availability(cls, employee, start_date, end_date, exclude_id=None):
        """Return overlapping Loop + Hoshin commitments for this employee."""
        from agents_api.models import ResourceCommitment

        # Loop overlaps
        loop_qs = cls.objects.filter(
            employee=employee,
            start_date__lt=end_date,
            end_date__gt=start_date,
        ).exclude(status="declined")
        if exclude_id:
            loop_qs = loop_qs.exclude(pk=exclude_id)

        # Hoshin overlaps
        hoshin_qs = ResourceCommitment.check_availability(employee, start_date, end_date)

        return {
            "loop": list(loop_qs),
            "hoshin": list(hoshin_qs),
            "has_conflicts": loop_qs.exists() or hoshin_qs.exists(),
        }

    def to_dict(self):
        return {
            "id": str(self.id),
            "commitment_id": str(self.commitment_id),
            "employee_id": str(self.employee_id),
            "employee_name": self.employee.name,
            "role": self.role,
            "hours_needed": float(self.hours_needed) if self.hours_needed else None,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }


class CommitmentNote(models.Model):
    """Threaded note on a commitment — communication between owner and org.

    Notes are append-only (immutable once created). They capture status updates,
    blockers, questions, and decisions that don't belong in the formal status field.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    commitment = models.ForeignKey(
        Commitment,
        on_delete=models.CASCADE,
        related_name="notes",
    )
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="commitment_notes",
    )
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "loop_commitment_note"
        ordering = ["created_at"]

    def __str__(self):
        return f"Note on {self.commitment_id} by {self.author_id}"


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

    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="mode_transitions",
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


# =============================================================================
# TRAINING REFLECTION (LOOP-001 §6.2)
# =============================================================================


class TrainingReflection(SynaraEntity):
    """Hansei response from operator after training completion.

    Reflections are REQUIRED for training completion. A TrainingRecord
    without a linked TrainingReflection is incomplete.

    Reflections aggregate per-document (not per-operator) to surface
    standard revision signals when confusion threshold is exceeded.

    LOOP-001 §6.2, TRN-001 §9.3
    """

    training_record = models.OneToOneField(
        "agents_api.TrainingRecord",
        on_delete=models.CASCADE,
        related_name="reflection",
    )
    controlled_document = models.ForeignKey(
        "agents_api.ControlledDocument",
        on_delete=models.CASCADE,
        related_name="training_reflections",
    )
    document_version = models.CharField(
        max_length=20,
        help_text="Snapshot of document version at training time",
    )

    # Operator can flag specific sections as confusing
    flagged_sections = models.ManyToManyField(
        "agents_api.ISOSection",
        blank=True,
        related_name="flagged_in_reflections",
        help_text="Sections the operator found unclear",
    )

    # The reflection itself
    reflection_text = models.TextField(
        help_text="Free-text reflection prompted by: What was clear? What was confusing? "
        "What would you change? What's different from how you actually do it?",
    )

    # Self-assessment
    self_assessed_level = models.IntegerField(
        default=0,
        help_text="Operator's self-assessed TWI competency level (0-4) post-training",
    )

    class Meta:
        db_table = "loop_training_reflection"
        ordering = ["-created_at"]

    def __str__(self):
        return f"Reflection on {self.controlled_document} by {self.training_record}"


# =============================================================================
# PROCESS CONFIRMATION (LOOP-001 §7.1)
# =============================================================================


class ProcessConfirmation(SynaraEntity):
    """Structured gemba observation — answers two questions:
    1. Was the standard followed?
    2. Did following the standard produce the expected outcome?

    The diagnostic matrix (followed × outcome) auto-computes a diagnosis.
    This is an interactive instrument, not a form.

    LOOP-001 §7.1
    """

    class Diagnosis(models.TextChoices):
        SYSTEM_WORKS = "system_works", "System Works"
        STANDARD_UNCLEAR = "standard_unclear", "Standard Unclear / Training Gap"
        PROCESS_GAP = "process_gap", "Process Design Broken"
        INCOMPLETE = "incomplete", "Observation Incomplete"

    class CloseLoopMethod(models.TextChoices):
        IMMEDIATE = "immediate", "Immediate (on the spot)"
        WITHIN_24H = "within_24h", "Within 24 Hours"
        PENDING = "pending", "Pending"
        NOT_DONE = "not_done", "Not Done"

    class ComfortLevel(models.TextChoices):
        COMFORTABLE = "comfortable", "Comfortable"
        NEUTRAL = "neutral", "Neutral"
        UNCOMFORTABLE = "uncomfortable", "Uncomfortable"

    tenant = models.ForeignKey(
        "core.Tenant",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="process_confirmations",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="process_confirmations",
    )

    # What's being confirmed
    controlled_document = models.ForeignKey(
        "agents_api.ControlledDocument",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="process_confirmations",
        help_text="The standard being confirmed — required before observation can begin",
    )
    document_version = models.CharField(max_length=20)

    # Who
    operator = models.ForeignKey(
        "agents_api.Employee",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="process_confirmations_as_operator",
        help_text="The operator being observed — assigned at gemba, not at scheduling time",
    )
    observer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="process_confirmations_as_observer",
        help_text="The person conducting the PC",
    )

    # Where
    process_area = models.CharField(max_length=255, blank=True, default="")
    shift = models.CharField(max_length=20, blank=True, default="")

    # Auto-computed diagnosis from observation items
    diagnosis = models.CharField(
        max_length=30,
        choices=Diagnosis.choices,
        default=Diagnosis.INCOMPLETE,
    )

    # Observation notes
    observer_notes = models.TextField(blank=True, default="")
    improvements_observed = models.TextField(
        blank=True,
        default="",
        help_text="What's going RIGHT — acknowledge good practice first",
    )

    # Operator interaction (STOP methodology)
    operator_interaction = models.JSONField(
        default=dict,
        blank=True,
        help_text='{"what_going_well": "...", "challenges": "...", "what_would_change": "..."}',
    )
    comfort_level = models.CharField(
        max_length=20,
        choices=ComfortLevel.choices,
        blank=True,
        default="",
    )

    # Close-the-loop
    close_loop_method = models.CharField(
        max_length=20,
        choices=CloseLoopMethod.choices,
        blank=True,
        default="",
    )
    close_loop_notes = models.TextField(blank=True, default="")

    class Meta:
        db_table = "loop_process_confirmation"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["operator", "controlled_document"]),
            models.Index(fields=["tenant", "-created_at"]),
        ]

    def __str__(self):
        return f"PC: {self.controlled_document} — {self.operator} ({self.diagnosis})"

    def compute_diagnosis(self):
        """Compute diagnosis from observation items using the diagnostic matrix.

        | Followed? | Outcome? | Diagnosis |
        |-----------|----------|-----------|
        | Yes       | Yes      | system_works |
        | No        | —        | standard_unclear (training gap or impractical standard) |
        | Yes       | No       | process_gap (standard encodes broken process — HIGH VALUE) |
        """
        items = self.observation_items.all()
        if not items.exists():
            self.diagnosis = self.Diagnosis.INCOMPLETE
            return self.diagnosis

        has_not_followed = items.filter(followed=False).exclude(followed_na=True).exists()
        has_bad_outcome = items.filter(outcome_pass=False).exclude(outcome_na=True).exists()
        all_followed = not has_not_followed

        if has_not_followed:
            self.diagnosis = self.Diagnosis.STANDARD_UNCLEAR
        elif all_followed and has_bad_outcome:
            # Maximum information event: standard followed but outcome wrong
            self.diagnosis = self.Diagnosis.PROCESS_GAP
        else:
            self.diagnosis = self.Diagnosis.SYSTEM_WORKS

        return self.diagnosis

    @property
    def pass_rate(self):
        """Fraction of items where standard was followed AND outcome was correct."""
        items = self.observation_items.exclude(followed_na=True).exclude(outcome_na=True)
        total = items.count()
        if total == 0:
            return None
        passed = items.filter(followed=True, outcome_pass=True).count()
        return passed / total


class PCObservationItem(models.Model):
    """Individual step observation within a Process Confirmation.

    Each item corresponds to a key step in the standard.
    Two questions per step: followed? outcome correct?

    LOOP-001 §7.1
    """

    class DeviationSeverity(models.TextChoices):
        CRITICAL = "critical", "Critical"
        HIGH = "high", "High"
        MEDIUM = "medium", "Medium"
        LOW = "low", "Low"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    process_confirmation = models.ForeignKey(
        ProcessConfirmation,
        on_delete=models.CASCADE,
        related_name="observation_items",
    )
    sort_order = models.IntegerField(default=0)

    # Which step
    step_text = models.CharField(max_length=500, help_text="The step being observed")
    key_point = models.CharField(max_length=500, blank=True, default="")
    reason_why = models.CharField(max_length=500, blank=True, default="")

    # Linked section from the standard (optional)
    linked_section = models.ForeignKey(
        "agents_api.ISOSection",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )

    # Observation — single-tap on mobile
    followed = models.BooleanField(default=True, help_text="Was the standard followed?")
    followed_na = models.BooleanField(default=False, help_text="Not applicable")
    outcome_pass = models.BooleanField(default=True, help_text="Was the outcome correct?")
    outcome_na = models.BooleanField(default=False, help_text="Not applicable")

    # Deviation classification (when followed=False or outcome_pass=False)
    deviation_severity = models.CharField(
        max_length=10,
        choices=DeviationSeverity.choices,
        blank=True,
        default="",
    )

    # Notes and evidence
    notes = models.TextField(blank=True, default="")
    photo = models.ForeignKey(
        "files.UserFile",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
        help_text="Photo evidence linked to this step",
    )

    class Meta:
        db_table = "loop_pc_observation_item"
        ordering = ["sort_order"]

    def __str__(self):
        status = "✓" if (self.followed and self.outcome_pass) else "✗"
        return f"{status} {self.step_text[:50]}"


# =============================================================================
# FORCED FAILURE TEST (LOOP-001 §7.2)
# =============================================================================


class ForcedFailureTest(SynaraEntity):
    """Detection verification — deliberately create failure conditions
    and observe whether detection controls catch them.

    Results feed directly into FMIS detection posterior (§8.3):
    Beta(α + detected, β + (injected - detected))

    LOOP-001 §7.2
    """

    class TestMode(models.TextChoices):
        HYPOTHESIS_DRIVEN = "hypothesis_driven", "Hypothesis-Driven (linked to FMIS row)"
        EXPLORATORY = "exploratory", "Exploratory (boundary probing)"

    class Result(models.TextChoices):
        DETECTED = "detected", "Detected"
        NOT_DETECTED = "not_detected", "Not Detected"
        PARTIALLY_DETECTED = "partially_detected", "Partially Detected"

    tenant = models.ForeignKey(
        "core.Tenant",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="forced_failure_tests",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="forced_failure_tests",
    )

    # What's being tested
    test_mode = models.CharField(
        max_length=20,
        choices=TestMode.choices,
        default=TestMode.HYPOTHESIS_DRIVEN,
    )
    fmea_row = models.ForeignKey(
        "agents_api.FMEARow",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="forced_failure_tests",
        help_text="Linked FMEA row for hypothesis-driven tests (nullable for exploratory)",
    )

    # Test plan
    test_plan = models.TextField(
        help_text="What conditions will be created, what's the expected detection response",
    )
    control_being_tested = models.TextField(
        blank=True,
        default="",
        help_text="Description of the detection control under test",
    )

    # Safety review — HARD GATE
    safety_reviewed = models.BooleanField(
        default=False,
        help_text="Confirmation that test can be conducted safely — REQUIRED before execution",
    )
    safety_reviewer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="safety_reviewed_tests",
    )

    # Results — integer counts for Sunrise tracking
    result = models.CharField(
        max_length=20,
        choices=Result.choices,
        blank=True,
        default="",
    )
    detection_count = models.IntegerField(
        default=0,
        help_text="How many injected failures were detected",
    )
    injection_count = models.IntegerField(
        default=0,
        help_text="How many failures were injected",
    )

    # Evidence
    evidence_notes = models.TextField(blank=True, default="")
    evidence_photos = models.ManyToManyField(
        "files.UserFile",
        blank=True,
        related_name="forced_failure_evidence",
    )

    # Who
    conducted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="conducted_forced_failure_tests",
    )
    conducted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "loop_forced_failure_test"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["fmea_row", "-created_at"]),
            models.Index(fields=["tenant", "-created_at"]),
        ]

    def __str__(self):
        return f"FFT: {self.detection_count}/{self.injection_count} detected ({self.test_mode})"

    def save(self, *args, **kwargs):
        # Safety gate: cannot record results without safety review
        if self.result and not self.safety_reviewed:
            raise ValueError(
                "Cannot record forced failure test results without safety review. "
                "Set safety_reviewed=True and safety_reviewer before recording results."
            )
        # Auto-classify result from counts
        if self.injection_count > 0 and not self.result:
            if self.detection_count == self.injection_count:
                self.result = self.Result.DETECTED
            elif self.detection_count == 0:
                self.result = self.Result.NOT_DETECTED
            else:
                self.result = self.Result.PARTIALLY_DETECTED
        super().save(*args, **kwargs)

    @property
    def detection_rate(self):
        """Empirical detection rate from this test."""
        if self.injection_count == 0:
            return None
        return self.detection_count / self.injection_count


# =============================================================================
# FMIS — Failure Modes Investigation System (LOOP-001 §8)
# =============================================================================


class FMIS(SynaraEntity):
    """Investigation-native FMEA container.

    Coexists with the standalone AIAG FMEA tool (agents_api.FMEA).
    FMIS participates in the Investigate → Standardize → Verify loop
    with Bayesian S/O/D scoring.

    LOOP-001 §8.6, §8.7
    """

    class Methodology(models.TextChoices):
        AIAG_4TH = "aiag_4th", "AIAG 4th Edition (manual 1-10)"
        SVEND_BAYESIAN = "svend_bayesian", "Svend Bayesian (posteriors + integer mapping)"
        SVEND_FULL = "svend_full", "Svend Full (posteriors + operational definitions required)"

    tenant = models.ForeignKey(
        "core.Tenant",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="fmis_documents",
    )
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True, default="")
    methodology = models.CharField(
        max_length=20,
        choices=Methodology.choices,
        default=Methodology.SVEND_BAYESIAN,
    )

    # Source investigation
    investigation = models.ForeignKey(
        "core.Investigation",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="fmis_documents",
    )

    # Owner
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="fmis_documents",
    )

    class Meta:
        db_table = "loop_fmis"
        ordering = ["-updated_at"]
        verbose_name = "FMIS"
        verbose_name_plural = "FMIS Documents"

    def __str__(self):
        return f"FMIS: {self.title} ({self.methodology})"


class FMISRow(SynaraEntity):
    """A single failure mode entry with Bayesian posteriors.

    Detection: Beta-Binomial, updated by forced failure tests.
    Occurrence: Beta-Binomial, updated by production data / SPC.
    Severity: Categorical-Dirichlet (5 categories), updated by consequence observations.

    LOOP-001 §8.3, §8.4, §8.5, §8.7
    """

    fmis = models.ForeignKey(
        FMIS,
        on_delete=models.CASCADE,
        related_name="rows",
    )
    investigation = models.ForeignKey(
        "core.Investigation",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="fmis_rows",
    )

    # ── Failure mode description ──
    # Text fields for human-readable description
    failure_mode_text = models.CharField(max_length=500)
    effect_text = models.CharField(max_length=500, blank=True, default="")
    cause_text = models.CharField(max_length=500, blank=True, default="")

    # Operational definition links — will be FK → graph.ProcessNode in GRAPH-001 Phase 1.
    # Legacy core.Entity FKs removed (Object 271 Phase 0 — were always null).

    # ── Controls ──
    prevention_control = models.TextField(blank=True, default="")
    detection_control = models.TextField(blank=True, default="")

    # ── Severity: Categorical-Dirichlet (§8.5) ──
    # 5 categories: [negligible, minor, moderate, severe, catastrophic]
    severity_alpha = models.JSONField(
        default=list,
        help_text="Dirichlet alpha vector [α₁, α₂, α₃, α₄, α₅]. Default: [1,1,1,1,1]",
    )
    severity_manual = models.IntegerField(
        null=True,
        blank=True,
        help_text="Manual override / initial committee estimate (1-10)",
    )
    severity_method = models.CharField(
        max_length=10,
        default="manual",
        help_text="manual or bayesian — determines which value is displayed",
    )

    # ── Occurrence: Beta-Binomial (§8.4) ──
    occurrence_alpha = models.FloatField(default=1.0, help_text="Beta prior α")
    occurrence_beta = models.FloatField(default=1.0, help_text="Beta prior β")
    occurrence_failures = models.IntegerField(default=0, help_text="Cumulative failure count")
    occurrence_units = models.IntegerField(default=0, help_text="Cumulative units observed")
    occurrence_manual = models.IntegerField(
        null=True,
        blank=True,
        help_text="Manual override (1-10)",
    )
    occurrence_method = models.CharField(max_length=10, default="manual")

    # ── Detection: Beta-Binomial (§8.3) ──
    detection_alpha = models.FloatField(default=1.0, help_text="Beta prior α")
    detection_beta = models.FloatField(default=1.0, help_text="Beta prior β")
    detection_detected = models.IntegerField(default=0, help_text="Cumulative detections from FFTs")
    detection_injected = models.IntegerField(default=0, help_text="Cumulative injections from FFTs")
    detection_manual = models.IntegerField(
        null=True,
        blank=True,
        help_text="Manual override (1-10)",
    )
    detection_method = models.CharField(max_length=10, default="manual")

    # ── Computed ──
    rpn = models.IntegerField(default=0, help_text="S × O × D (computed on save)")
    last_evidence_date = models.DateTimeField(null=True, blank=True)

    # ── Migration bridge ──
    legacy_fmea_row = models.ForeignKey(
        "agents_api.FMEARow",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="fmis_row",
        help_text="Link to legacy AIAG FMEARow for migration",
    )

    class Meta:
        db_table = "loop_fmis_row"
        ordering = ["-rpn", "created_at"]

    def __str__(self):
        return f"{self.failure_mode_text[:50]} (RPN={self.rpn})"

    def save(self, *args, **kwargs):
        # Initialize severity_alpha if empty
        if not self.severity_alpha:
            self.severity_alpha = [1, 1, 1, 1, 1]

        # Compute RPN from active method (manual or bayesian)
        self.rpn = self._compute_rpn()
        super().save(*args, **kwargs)

    def _compute_rpn(self):
        """Compute RPN from the active scoring method."""
        s = self.severity_score
        o = self.occurrence_score
        d = self.detection_score
        if s is None or o is None or d is None:
            return 0
        return s * o * d

    # ── Computed properties ──

    @property
    def severity_score(self):
        """Active severity score (1-10) based on method."""
        if self.severity_method == "bayesian" and self.severity_alpha:
            from .bayesian import dirichlet_to_aiag_10

            return dirichlet_to_aiag_10(self.severity_alpha)
        return self.severity_manual

    @property
    def occurrence_score(self):
        """Active occurrence score (1-10) based on method."""
        if self.occurrence_method == "bayesian":
            from .bayesian import beta_mean, beta_mean_to_aiag_10

            rate = beta_mean(self.occurrence_alpha, self.occurrence_beta)
            # For occurrence: high rate = high score
            return beta_mean_to_aiag_10(1.0 - rate)  # Invert: high rate → high score
        return self.occurrence_manual

    @property
    def detection_score(self):
        """Active detection score (1-10) based on method."""
        if self.detection_method == "bayesian":
            from .bayesian import beta_mean, beta_mean_to_aiag_10

            rate = beta_mean(self.detection_alpha, self.detection_beta)
            # For detection: high rate = low score (good detection)
            return beta_mean_to_aiag_10(rate)
        return self.detection_manual

    @property
    def severity_distribution(self):
        """Severity posterior as category probabilities."""
        if not self.severity_alpha:
            return None
        from .bayesian import SEVERITY_CATEGORIES, dirichlet_mean

        probs = dirichlet_mean(self.severity_alpha)
        return dict(zip(SEVERITY_CATEGORIES, probs))

    @property
    def detection_credible_interval(self):
        """90% credible interval for detection rate."""
        from .bayesian import beta_credible_interval

        return beta_credible_interval(self.detection_alpha, self.detection_beta)

    @property
    def occurrence_credible_interval(self):
        """90% credible interval for occurrence rate."""
        from .bayesian import beta_credible_interval

        return beta_credible_interval(self.occurrence_alpha, self.occurrence_beta)

    @property
    def has_operational_definitions(self):
        """True if all three text fields are populated.

        Will check FK → graph.ProcessNode linkage in GRAPH-001 Phase 1.
        """
        return all(
            [
                self.failure_mode_text,
                self.effect_text,
                self.cause_text,
            ]
        )

    @property
    def undefined_terms(self):
        """List of terms without operational definitions (knowledge gaps)."""
        gaps = []
        if not self.failure_mode_text:
            gaps.append("failure_mode")
        if not self.effect_text:
            gaps.append("effect")
        if not self.cause_text:
            gaps.append("cause")
        return gaps

    # ── Evidence update methods ──

    def update_detection_from_fft(self, fft):
        """Update detection posterior from a ForcedFailureTest.

        Beta(α + d, β + (n - d)) where d = detected, n = injected.

        LOOP-001 §8.3 (Sunrise Problem)
        """
        from .bayesian import beta_update

        self.detection_detected += fft.detection_count
        self.detection_injected += fft.injection_count
        self.detection_alpha, self.detection_beta = beta_update(
            self.detection_alpha,
            self.detection_beta,
            fft.detection_count,
            fft.injection_count,
        )
        self.detection_method = "bayesian"
        self.last_evidence_date = timezone.now()
        self.save(
            update_fields=[
                "detection_detected",
                "detection_injected",
                "detection_alpha",
                "detection_beta",
                "detection_method",
                "last_evidence_date",
                "rpn",
                "updated_at",
            ]
        )

    def update_occurrence_from_data(self, failures, units):
        """Update occurrence posterior from production data.

        Beta(α + f, β + (n - f)) where f = failures, n = units.

        LOOP-001 §8.4 (Sunrise Problem)
        """
        from .bayesian import beta_update

        self.occurrence_failures += failures
        self.occurrence_units += units
        self.occurrence_alpha, self.occurrence_beta = beta_update(
            self.occurrence_alpha,
            self.occurrence_beta,
            failures,
            units,
        )
        self.occurrence_method = "bayesian"
        self.last_evidence_date = timezone.now()
        self.save(
            update_fields=[
                "occurrence_failures",
                "occurrence_units",
                "occurrence_alpha",
                "occurrence_beta",
                "occurrence_method",
                "last_evidence_date",
                "rpn",
                "updated_at",
            ]
        )

    def update_severity_from_observation(self, category_name):
        """Update severity posterior from a consequence observation.

        Dirichlet(α₁, ..., αᵢ + 1, ..., αₖ) where i = observed category.

        LOOP-001 §8.5 (Categorical-Dirichlet)
        """
        from .bayesian import dirichlet_update_by_name

        if not self.severity_alpha:
            self.severity_alpha = [1, 1, 1, 1, 1]
        self.severity_alpha = dirichlet_update_by_name(self.severity_alpha, category_name)
        self.severity_method = "bayesian"
        self.last_evidence_date = timezone.now()
        self.save(
            update_fields=[
                "severity_alpha",
                "severity_method",
                "last_evidence_date",
                "rpn",
                "updated_at",
            ]
        )


# =============================================================================
# INVESTIGATION ENTRY (LOOP-001 §16.3)
# =============================================================================


class InvestigationEntry(SynaraEntity):
    """A notebook-style entry within an investigation workspace.

    Entries are the chronological record of an investigation. They can be:
    - Narrative: written by the investigator
    - Tool output: auto-created when a tool runs in investigation context
    - Evidence: auto-extracted from tool outputs per CANON-002

    Entries are immutable once created — they can be superseded but not edited.
    This preserves the investigation record for compliance.

    LOOP-001 §16.3
    """

    class EntryType(models.TextChoices):
        NARRATIVE = "narrative", "Narrative"
        TOOL_OUTPUT = "tool_output", "Tool Output"
        EVIDENCE = "evidence", "Evidence"
        PHOTO = "photo", "Photo"
        DATA = "data", "Data"

    investigation = models.ForeignKey(
        "core.Investigation",
        on_delete=models.CASCADE,
        related_name="entries",
    )
    entry_type = models.CharField(
        max_length=20,
        choices=EntryType.choices,
        default=EntryType.NARRATIVE,
    )
    title = models.CharField(max_length=300, blank=True, default="")
    content = models.TextField(
        blank=True,
        default="",
        help_text="Markdown content for narrative entries. JSON for tool/evidence/data.",
    )

    # Tool output linkage (for entry_type=tool_output)
    tool_link = models.OneToOneField(
        "core.InvestigationToolLink",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="entry",
        help_text="Link to InvestigationToolLink for tool output entries",
    )

    # Structured data (charts, tables, analysis results)
    structured_data = models.JSONField(
        default=dict,
        blank=True,
        help_text="Chart spec JSON, table data, analysis parameters/results",
    )

    # Author
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="investigation_entries",
    )

    # Sort order (for manual reordering — defaults to creation time)
    sort_order = models.IntegerField(default=0)

    class Meta:
        db_table = "loop_investigation_entry"
        ordering = ["sort_order", "created_at"]

    def __str__(self):
        return f"[{self.entry_type}] {self.title or self.content[:50]}"


# =============================================================================
# AUDITOR PORTAL TOKEN (LOOP-001 §11)
# =============================================================================


class AuditorPortalToken(models.Model):
    """Time-limited access token for external auditors.

    Quality manager generates a link, shares with auditor. No account
    required. Token expires after org-defined period. Multi-use (auditors
    come back multiple times during an audit). Access logged.

    LOOP-001 §11.2, §16.9
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="auditor_tokens",
    )
    token = models.CharField(max_length=64, unique=True, db_index=True)
    label = models.CharField(
        max_length=200,
        help_text="Human-readable label, e.g. 'ISO 9001 Surveillance — BSI March 2026'",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="created_auditor_tokens",
    )
    expires_at = models.DateTimeField()
    revoked_at = models.DateTimeField(null=True, blank=True)
    last_accessed_at = models.DateTimeField(null=True, blank=True)
    access_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "loop_auditor_portal_token"
        ordering = ["-created_at"]

    def save(self, *args, **kwargs):
        if not self.token:
            self.token = secrets.token_urlsafe(32)
        super().save(*args, **kwargs)

    @property
    def is_valid(self):
        return self.revoked_at is None and self.expires_at > timezone.now()

    def record_access(self):
        from django.db.models import F

        AuditorPortalToken.objects.filter(pk=self.pk).update(
            last_accessed_at=timezone.now(),
            access_count=F("access_count") + 1,
        )

    def revoke(self):
        self.revoked_at = timezone.now()
        self.save(update_fields=["revoked_at"])

    def __str__(self):
        status = "valid" if self.is_valid else "expired/revoked"
        return f"AuditorToken({self.label}) — {status}"


# =============================================================================
# SUPPLIER CLAIM (Supplier Accountability System — Object 271)
# =============================================================================


class SupplierClaim(SynaraEntity):
    """Formal quality claim against a supplier.

    Lifecycle: draft → filed → acknowledged → responded → under_review
               → verified → closed | rejected | escalated

    The claim is the external-facing document. The NCR is the internal record.
    A claim can exist without an NCR (contractual non-performance) and vice versa.
    """

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        FILED = "filed", "Filed"
        ACKNOWLEDGED = "acknowledged", "Acknowledged"
        RESPONDED = "responded", "Responded"
        UNDER_REVIEW = "under_review", "Under Review"
        VERIFIED = "verified", "Verified"
        CLOSED = "closed", "Closed"
        REJECTED = "rejected", "Rejected"
        ESCALATED = "escalated", "Escalated"

    class ClaimType(models.TextChoices):
        QUALITY_DEFECT = "quality_defect", "Quality Defect"
        DELIVERY = "delivery", "Delivery Issue"
        DOCUMENTATION = "documentation", "Documentation"
        SPECIFICATION = "specification", "Specification Deviation"
        PACKAGING = "packaging", "Packaging / Handling"
        CONTAMINATION = "contamination", "Contamination"

    class Disposition(models.TextChoices):
        RETURNED = "returned", "Return to Supplier"
        SCRAPPED = "scrapped", "Scrapped"
        REWORKED = "reworked", "Reworked"
        USE_AS_IS = "use_as_is", "Use As-Is (concession)"
        SORTED = "sorted", "100% Sorted"

    tenant = models.ForeignKey(
        "core.Tenant",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="supplier_claims",
    )
    supplier = models.ForeignKey(
        "agents_api.SupplierRecord",
        on_delete=models.CASCADE,
        related_name="claims",
    )
    ncr = models.ForeignKey(
        "agents_api.NonconformanceRecord",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="supplier_claims",
    )

    # What happened
    claim_type = models.CharField(max_length=20, choices=ClaimType.choices)
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True, default="")

    # Affected product
    part_number = models.CharField(max_length=100, blank=True, default="")
    lot_number = models.CharField(max_length=100, blank=True, default="")
    quantity_affected = models.IntegerField(default=0)
    quantity_rejected = models.IntegerField(default=0)

    # Defect evidence
    defect_description = models.TextField(blank=True, default="")
    inspection_method = models.TextField(blank=True, default="")
    evidence_photos = models.JSONField(default=list, blank=True)

    # Financial impact
    cost_of_quality = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Internal cost: scrap, rework, sort, downtime",
    )
    credit_requested = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    credit_received = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    disposition = models.CharField(max_length=20, choices=Disposition.choices, blank=True, default="")

    # Lifecycle
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT, db_index=True)
    filed_at = models.DateTimeField(null=True, blank=True)
    response_due_date = models.DateField(null=True, blank=True)

    # Portal access for supplier
    portal_token = models.CharField(max_length=64, unique=True, blank=True, default="")
    portal_expires_at = models.DateTimeField(null=True, blank=True)

    # Graph integration
    linked_process_node_ids = models.JSONField(default=list, blank=True)

    # Provenance
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="created_supplier_claims",
    )

    class Meta:
        db_table = "loop_supplier_claim"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["supplier", "status"]),
            models.Index(fields=["tenant", "status"]),
            models.Index(fields=["portal_token"]),
        ]

    def __str__(self):
        return f"Claim: {self.title} ({self.supplier.name})"

    def generate_portal_token(self):
        """Generate a unique portal token for supplier access."""
        import secrets

        self.portal_token = secrets.token_urlsafe(48)
        self.portal_expires_at = timezone.now() + timezone.timedelta(days=30)
        self.save(update_fields=["portal_token", "portal_expires_at"])
        return self.portal_token

    @property
    def portal_is_valid(self):
        return self.portal_token and self.portal_expires_at and self.portal_expires_at > timezone.now()

    @property
    def response_is_overdue(self):
        from datetime import date

        return (
            self.response_due_date
            and self.status in (self.Status.FILED, self.Status.ACKNOWLEDGED)
            and self.response_due_date < date.today()
        )

    VALID_TRANSITIONS = {
        "draft": {"filed"},
        "filed": {"acknowledged", "responded"},
        "acknowledged": {"responded"},
        "responded": {"under_review"},
        "under_review": {"verified", "rejected", "escalated"},
        "rejected": {"responded"},  # supplier revises
        "verified": {"closed"},
        "escalated": {"closed", "under_review"},
    }


class SupplierResponse(models.Model):
    """Supplier's CAPA response to a claim.

    Multiple responses possible per claim (revision chain when rejected).
    Response quality is auto-scored by rule engine.
    """

    class RootCauseCategory(models.TextChoices):
        METHOD = "method", "Method / Process"
        MATERIAL = "material", "Material"
        MACHINE = "machine", "Machine / Equipment"
        MAN = "man", "Personnel / Training"
        MEASUREMENT = "measurement", "Measurement System"
        ENVIRONMENT = "environment", "Environment"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    claim = models.ForeignKey(
        SupplierClaim,
        on_delete=models.CASCADE,
        related_name="responses",
    )

    # Root cause
    root_cause_category = models.CharField(max_length=20, choices=RootCauseCategory.choices)
    root_cause_description = models.TextField()

    # Corrective & preventive actions
    corrective_action = models.TextField()
    preventive_action = models.TextField()
    implementation_date = models.DateField()

    # Evidence
    evidence_files = models.JSONField(default=list, blank=True, help_text="UserFile UUIDs")

    # Quality analysis (system-computed)
    is_repeat_root_cause = models.BooleanField(default=False)
    response_quality_score = models.FloatField(
        null=True,
        blank=True,
        help_text="0-1 score computed by response quality rules",
    )
    quality_flags = models.JSONField(
        default=list,
        blank=True,
        help_text="List of string flags from quality analysis",
    )

    # Review
    accepted = models.BooleanField(null=True, blank=True, help_text="null=pending, true=accepted, false=rejected")
    reviewer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="reviewed_supplier_responses",
    )
    reviewer_notes = models.TextField(blank=True, default="")
    reviewed_at = models.DateTimeField(null=True, blank=True)

    # Metadata
    revision = models.IntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "loop_supplier_response"
        ordering = ["revision"]

    def __str__(self):
        return f"Response #{self.revision} on {self.claim.title}"


class ClaimVerification(models.Model):
    """Verification that supplier's corrective action actually worked.

    Evidence from next shipment inspection, supplier audit, or testing.
    """

    class VerificationType(models.TextChoices):
        NEXT_SHIPMENT = "next_shipment", "Next Shipment Inspection"
        AUDIT = "audit", "Supplier Audit"
        TEST = "test", "Test / Re-Qualification"
        DOCUMENT_REVIEW = "document_review", "Document Review"

    class Result(models.TextChoices):
        CONFORMING = "conforming", "Conforming"
        NONCONFORMING = "nonconforming", "Nonconforming"
        PARTIAL = "partial", "Partially Conforming"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    claim = models.ForeignKey(
        SupplierClaim,
        on_delete=models.CASCADE,
        related_name="verifications",
    )

    verification_type = models.CharField(max_length=20, choices=VerificationType.choices)
    result = models.CharField(max_length=20, choices=Result.choices)
    evidence_notes = models.TextField(blank=True, default="")
    evidence_files = models.JSONField(default=list, blank=True)

    verified_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="claim_verifications",
    )
    verified_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "loop_claim_verification"
        ordering = ["-verified_at"]

    def __str__(self):
        return f"Verification ({self.verification_type}): {self.result}"


# =============================================================================
# SUPPLIER CoA (Certificate of Analysis — Object 271)
# =============================================================================


class SupplierCoA(SynaraEntity):
    """Inbound Certificate of Analysis from a supplier.

    Contains extracted measurement data that can be validated against specs,
    ingested into SPC charts, and linked to graph material property nodes.

    Lifecycle: uploaded → reviewed → accepted → ingested | rejected
    """

    class Status(models.TextChoices):
        UPLOADED = "uploaded", "Uploaded"
        REVIEWED = "reviewed", "Reviewed"
        ACCEPTED = "accepted", "Accepted"
        INGESTED = "ingested", "Ingested into SPC"
        REJECTED = "rejected", "Rejected"

    class ExtractionMethod(models.TextChoices):
        MANUAL = "manual", "Manual Entry"
        CSV = "csv", "CSV Upload"

    tenant = models.ForeignKey(
        "core.Tenant",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="supplier_coas",
    )
    supplier = models.ForeignKey(
        "agents_api.SupplierRecord",
        on_delete=models.CASCADE,
        related_name="coas",
    )

    # Document reference
    document = models.ForeignKey(
        "files.UserFile",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="coas",
        help_text="Uploaded PDF/file if applicable",
    )
    coa_number = models.CharField(max_length=100, blank=True, default="")
    lot_number = models.CharField(max_length=100, blank=True, default="")
    part_number = models.CharField(max_length=100, blank=True, default="")
    date_issued = models.DateField(null=True, blank=True)

    # Extracted measurements
    measurements = models.JSONField(
        default=list,
        blank=True,
        help_text="[{parameter, value, unit, spec_min, spec_max, method, conforming}]",
    )
    extraction_method = models.CharField(
        max_length=10,
        choices=ExtractionMethod.choices,
        default=ExtractionMethod.MANUAL,
    )

    # Compliance
    all_conforming = models.BooleanField(default=True)
    nonconforming_parameters = models.JSONField(
        default=list,
        blank=True,
        help_text="Parameter names that failed spec check",
    )

    # SPC integration
    spc_data_ingested = models.BooleanField(default=False)
    spc_ingestion_date = models.DateTimeField(null=True, blank=True)
    linked_process_node_ids = models.JSONField(
        default=list,
        blank=True,
        help_text="ProcessNode UUIDs for material property nodes",
    )

    # Lifecycle
    status = models.CharField(
        max_length=10,
        choices=Status.choices,
        default=Status.UPLOADED,
        db_index=True,
    )
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="reviewed_coas",
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)
    rejection_reason = models.TextField(blank=True, default="")

    # Provenance
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="created_coas",
    )

    class Meta:
        db_table = "loop_supplier_coa"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["supplier", "status"]),
            models.Index(fields=["tenant", "status"]),
            models.Index(fields=["part_number", "lot_number"]),
        ]

    def __str__(self):
        return f"CoA {self.coa_number or self.id} ({self.supplier.name})"

    def check_compliance(self):
        """Evaluate all measurements against spec limits."""
        nonconforming = []
        for m in self.measurements or []:
            val = m.get("value")
            spec_min = m.get("spec_min")
            spec_max = m.get("spec_max")
            if val is None:
                continue
            conforming = True
            if spec_min is not None and val < spec_min:
                conforming = False
            if spec_max is not None and val > spec_max:
                conforming = False
            m["conforming"] = conforming
            if not conforming:
                nonconforming.append(m.get("parameter", "Unknown"))

        self.nonconforming_parameters = nonconforming
        self.all_conforming = len(nonconforming) == 0
        self.save(
            update_fields=[
                "measurements",
                "nonconforming_parameters",
                "all_conforming",
                "updated_at",
            ]
        )

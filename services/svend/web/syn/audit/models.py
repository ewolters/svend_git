"""
Tamper-proof audit log models with blockchain-style hash chaining.

Provides immutable audit trail where each entry contains a hash that
includes the previous entry's hash, making tampering detectable.

Compliance: SOC 2 CC7.2 / ISO 27001 A.12.7
"""

import hashlib
import json
import uuid

from django.core.exceptions import ValidationError
from django.db import models, transaction

from syn.core.base_models import SynaraImmutableLog
from django.utils import timezone


class SysLogEntry(SynaraImmutableLog):
    """
    Immutable system log entry with hash chain integrity.

    Each entry contains a hash that includes the previous entry's hash,
    creating a blockchain-style chain that makes tampering detectable.

    Features:
    - Immutability: Edits prevented via save() override
    - Hash chain: Each entry links to previous via hash
    - Tenant isolation: Separate chains per tenant
    - Audit trail: Complete forensic record

    Compliance:
    - SOC 2 CC7.2: System activity monitoring and logging
    - ISO 27001 A.12.7: Audit log protection and tamper-proofing
    """

    id = models.BigAutoField(primary_key=True, help_text="Sequential ID for ordering")

    timestamp = models.DateTimeField(default=timezone.now, db_index=True, help_text="When this log entry was created")

    actor = models.CharField(
        max_length=255, db_index=True, default="system",
        help_text="User or system component that performed the action"
    )

    event_name = models.CharField(max_length=255, db_index=True, help_text="Name of the event being logged")

    payload = models.JSONField(default=dict, help_text="Event data and context")

    payload_hash = models.CharField(max_length=64, help_text="SHA-256 hash of the payload")

    correlation_id = models.UUIDField(
        db_index=True, null=True, blank=True, help_text="Correlation ID for distributed tracing"
    )

    tenant_id = models.UUIDField(
        db_index=True, null=True, blank=True, help_text="Tenant identifier for multi-tenant isolation (SEC-001 §5.2)"
    )

    previous_hash = models.CharField(
        max_length=64, default="0" * 64, help_text="Hash of the previous log entry in chain"
    )

    current_hash = models.CharField(max_length=64, unique=True, help_text="Hash of this entry (includes previous_hash)")

    is_genesis = models.BooleanField(default=False, help_text="True if this is the first entry in the chain")

    class Meta(SynaraImmutableLog.Meta):
        db_table = "audit_syslog_entry"
        ordering = ["id"]
        indexes = [
            models.Index(fields=["tenant_id", "timestamp"], name="audit_tenant_time"),
            models.Index(fields=["tenant_id", "id"], name="audit_tenant_id"),
            models.Index(fields=["event_name"], name="audit_event_name"),
            models.Index(fields=["correlation_id"], name="audit_correlation"),
            models.Index(fields=["current_hash"], name="audit_current_hash"),
        ]
        verbose_name = "System Log Entry"
        verbose_name_plural = "System Log Entries"
        # SYS-200 INV-008: Audit logs are immutable
        default_permissions = ("add", "view")  # No change/delete permissions

    class SynaraMeta:
        event_domain = "syn.audit.sys_log_entry"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        return f"[{self.timestamp}] {self.actor}: {self.event_name}"

    def save(self, *args, **kwargs):
        """
        Override save to enforce immutability and hash chain integrity.

        Prevents updates to existing entries (immutability).
        Computes hashes and chain linkage for new entries.
        Validates required fields for compliance.

        Raises:
            ValidationError: If attempting to update existing entry

        Compliance:
        - SYS-200 INV-008: Immutable audit logs
        - CTG-001 §5: Correlation tracking required
        """
        # Enforce immutability: prevent updates
        if self.pk is not None:
            raise ValidationError(
                "Audit log entries are immutable and cannot be modified. " "Create a new entry instead."
            )

        # CTG-001 §5: Validate correlation_id is provided
        # Log warning if missing (field allows null for legacy compatibility)
        if not self.correlation_id:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"[AUDIT] SysLogEntry created without correlation_id: "
                f"event_name={self.event_name}, actor={self.actor}, tenant_id={self.tenant_id}. "
                f"CTG-001 §5 requires correlation_id for causal tracing."
            )

        # Compute payload hash
        self.payload_hash = self._compute_payload_hash()

        # Get previous entry for chain linkage
        with transaction.atomic():
            # Lock the table to ensure chain integrity
            previous_entry = (
                SysLogEntry.objects.select_for_update().filter(tenant_id=self.tenant_id).order_by("-id").first()
            )

            if previous_entry is None:
                # First entry in chain (genesis)
                self.is_genesis = True
                self.previous_hash = "0" * 64
            else:
                # Link to previous entry
                self.is_genesis = False
                self.previous_hash = previous_entry.current_hash

            # Compute current hash (includes previous hash)
            self.current_hash = self._compute_current_hash()

            # Set entry_hash (SynaraImmutableLog field) to match current_hash
            # to satisfy the base model's non-nullable field
            self.entry_hash = self.current_hash

            # Call models.Model.save() directly — skip SynaraImmutableLog.save()
            # which would recompute the hash chain and corrupt previous_hash
            models.Model.save(self, *args, **kwargs)

    def _compute_payload_hash(self) -> str:
        """
        Compute SHA-256 hash of the payload.

        Returns:
            Hexadecimal hash string
        """
        # Serialize payload to JSON with sorted keys for consistency
        payload_json = json.dumps(self.payload, sort_keys=True)
        return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    def _compute_current_hash(self) -> str:
        """
        Compute SHA-256 hash of this entry.

        Includes:
        - timestamp
        - actor
        - event_name
        - payload_hash
        - correlation_id
        - tenant_id
        - previous_hash (creates chain linkage)

        Returns:
            Hexadecimal hash string
        """
        hash_data = {
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "event_name": self.event_name,
            "payload_hash": self.payload_hash,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "tenant_id": str(self.tenant_id) if self.tenant_id else None,
            "previous_hash": self.previous_hash,
        }

        # Serialize to JSON with sorted keys
        hash_json = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_json.encode("utf-8")).hexdigest()

    def verify_hash(self) -> bool:
        """
        Verify that this entry's hash is correct.

        Recomputes the hash and compares with stored value.

        Returns:
            True if hash is valid, False otherwise
        """
        expected_hash = self._compute_current_hash()
        return self.current_hash == expected_hash

    def verify_chain_link(self) -> bool:
        """
        Verify that this entry correctly links to previous entry.

        Checks that previous_hash matches the previous entry's current_hash.

        Returns:
            True if chain link is valid, False otherwise
        """
        if self.is_genesis:
            # Genesis entry should have zero hash
            return self.previous_hash == "0" * 64

        # Get previous entry
        previous_entry = SysLogEntry.objects.filter(tenant_id=self.tenant_id, id__lt=self.id).order_by("-id").first()

        if previous_entry is None:
            # No previous entry but not genesis - chain broken
            return False

        # Check that our previous_hash matches their current_hash
        return self.previous_hash == previous_entry.current_hash

    def delete(self, *args, **kwargs):
        """
        SYS-200 INV-008: SysLogEntry entries cannot be deleted.
        ISO 27001 A.12.7: Audit log protection requirement.

        Deletion would break the hash chain integrity.
        """
        raise ValueError(
            "SysLogEntry entries cannot be deleted. "
            "SYS-200 INV-008 violation. Deletion would break hash chain integrity."
        )

    @classmethod
    def get_chain_head(cls, tenant_id: str):
        """
        Get the most recent entry in the chain for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Latest SysLogEntry or None
        """
        return cls.objects.filter(tenant_id=tenant_id).order_by("-id").first()

    @classmethod
    def get_genesis(cls, tenant_id: str):
        """
        Get the genesis (first) entry in the chain for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            First SysLogEntry or None
        """
        return cls.objects.filter(tenant_id=tenant_id, is_genesis=True).order_by("id").first()


class IntegrityViolation(SynaraImmutableLog):
    """
    Record of detected integrity violations in the audit log.

    Created when hash chain verification fails, indicating
    potential tampering or corruption.

    Compliance: SOC 2 CC7.2 - Security incident tracking
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    detected_at = models.DateTimeField(auto_now_add=True, db_index=True, help_text="When the violation was detected")

    tenant_id = models.UUIDField(db_index=True, null=True, blank=True, help_text="Tenant affected by the violation (SEC-001 §5.2)")

    violation_type = models.CharField(
        max_length=50,
        choices=[
            ("hash_mismatch", "Hash Mismatch"),
            ("chain_break", "Chain Break"),
            ("missing_entry", "Missing Entry"),
            ("duplicate_hash", "Duplicate Hash"),
        ],
        help_text="Type of integrity violation",
    )

    entry_id = models.BigIntegerField(null=True, blank=True, help_text="ID of the log entry with violation")

    details = models.JSONField(default=dict, help_text="Detailed information about the violation")

    is_resolved = models.BooleanField(default=False, help_text="Whether this violation has been investigated and resolved")

    resolved_at = models.DateTimeField(null=True, blank=True, help_text="When the violation was resolved")

    resolution_notes = models.TextField(blank=True, help_text="Notes about how the violation was resolved")

    class Meta(SynaraImmutableLog.Meta):
        db_table = "audit_integrity_violation"
        ordering = ["-detected_at"]
        indexes = [
            models.Index(fields=["tenant_id", "detected_at"], name="violation_tenant_time"),
            models.Index(fields=["is_resolved"], name="violation_resolved"),
        ]
        verbose_name = "Integrity Violation"
        verbose_name_plural = "Integrity Violations"

    class SynaraMeta:
        event_domain = "syn.audit.integrity_violation"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        return f"[{self.detected_at}] {self.violation_type} - Tenant {self.tenant_id}"


class DriftViolation(SynaraImmutableLog):
    """
    Records architectural drift violations detected by enforcement checks.

    Tracks violations of canonical architecture patterns with full forensic
    traceability, governance integration, and remediation tracking.

    Standard: DRF-001 §7.2.1 - Drift Violation Schema
    Compliance: SOC 2 CC7.2, ISO 27001 A.12.7
    """

    # Severity levels for drift violations
    SEVERITY_CHOICES = [
        ("CRITICAL", "Critical"),
        ("HIGH", "High"),
        ("MEDIUM", "Medium"),
        ("LOW", "Low"),
    ]

    # Enforcement check identifiers
    ENFORCEMENT_CHECK_CHOICES = [
        ("ENC-001", "ENC-001 - Primitive Organization"),
        ("ENC-002", "ENC-002 - Layer Imports"),
        ("ENC-003", "ENC-003 - Field Naming"),
        ("ENC-004", "ENC-004 - Event Naming"),
        ("ENC-005", "ENC-005 - Security Patterns"),
        ("ENC-006", "ENC-006 - Serializer Patterns"),
        ("ENC-007", "ENC-007 - Pure Functions"),
        ("ENC-008", "ENC-008 - Pydantic V2"),
        ("ENC-009", "ENC-009 - Primitive Pattern"),
        ("ENC-010", "ENC-010 - Test Coverage"),
        ("ENC-011", "ENC-011 - Tenant Isolation"),
    ]

    # ========== Identity ==========

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="Unique identifier for this drift violation"
    )

    drift_signature = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
        help_text="Unique hash signature of this drift violation (prevents duplicates)"
    )

    # ========== Classification ==========

    severity = models.CharField(
        max_length=10,
        choices=SEVERITY_CHOICES,
        db_index=True,
        help_text="Severity level of the drift violation"
    )

    enforcement_check = models.CharField(
        max_length=10,
        choices=ENFORCEMENT_CHECK_CHOICES,
        db_index=True,
        help_text="Enforcement check that detected this violation"
    )

    # ========== Location ==========

    file_path = models.CharField(
        max_length=512,
        db_index=True,
        help_text="File path where the violation was detected"
    )

    line_number = models.IntegerField(
        null=True,
        blank=True,
        help_text="Line number where the violation was detected"
    )

    function_name = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        db_index=True,
        help_text="Function or class name where the violation was detected"
    )

    # ========== Detection Metadata ==========

    detected_at = models.DateTimeField(
        default=timezone.now,
        db_index=True,
        help_text="When this violation was detected"
    )

    detected_by = models.CharField(
        max_length=255,
        help_text="System or user that detected the violation"
    )

    # ========== Git Context ==========

    git_commit_sha = models.CharField(
        max_length=40,
        null=True,
        blank=True,
        db_index=True,
        help_text="Git commit SHA where the violation was detected"
    )

    git_author = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="Git commit author (for accountability)"
    )

    # ========== Violation Details ==========

    violation_message = models.TextField(
        help_text="Human-readable description of the violation"
    )

    code_snippet = models.TextField(
        null=True,
        blank=True,
        help_text="Code snippet showing the violation"
    )

    canonical_pattern = models.TextField(
        null=True,
        blank=True,
        help_text="Description or example of the canonical pattern"
    )

    # ========== Remediation ==========

    is_remediation_available = models.BooleanField(
        default=False,
        help_text="Whether automated remediation is available"
    )

    is_auto_fix_safe = models.BooleanField(
        default=False,
        help_text="Whether automated fix can be safely applied without review"
    )

    remediation_script = models.CharField(
        max_length=512,
        null=True,
        blank=True,
        help_text="Path to remediation script or tool"
    )

    remediation_sla_hours = models.IntegerField(
        null=True,
        blank=True,
        help_text="SLA hours for remediation based on severity"
    )

    remediation_due_at = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
        help_text="When remediation is due (detected_at + SLA)"
    )

    is_sla_breached = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Whether the remediation SLA has been breached"
    )

    # ========== Governance Integration ==========

    is_governance_escalated = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Whether this violation has been escalated to governance"
    )

    governance_rule_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Governance rule UUID (FK removed for Svend integration)"
    )

    governance_judgment_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Governance judgment UUID (FK removed for Svend integration)"
    )

    # ========== Causal Trace Graph Integration ==========

    correlation_id = models.UUIDField(
        db_index=True,
        default=uuid.uuid4,
        help_text="Correlation ID for causal trace graph"
    )

    ctg_node_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="CTG node UUID (FK removed for Svend integration)"
    )

    # ========== Multi-Tenancy ==========

    tenant_id = models.UUIDField(
        db_index=True,
        null=True,
        blank=True,
        help_text="Tenant identifier for multi-tenant deployments"
    )

    # ========== Resolution ==========

    resolved_at = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
        help_text="When this violation was resolved"
    )

    resolved_by = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        help_text="User or system that resolved the violation"
    )

    resolution_notes = models.TextField(
        blank=True,
        help_text="Notes about how the violation was resolved"
    )

    class Meta(SynaraImmutableLog.Meta):
        db_table = "audit_drift_violation"
        ordering = ["-detected_at"]
        indexes = [
            # Detection and tracking
            models.Index(fields=["tenant_id", "detected_at"], name="drift_tenant_detected"),
            models.Index(fields=["enforcement_check", "severity"], name="drift_check_severity"),
            models.Index(fields=["file_path"], name="drift_file_path"),
            models.Index(fields=["git_commit_sha"], name="drift_git_commit"),
            # Remediation tracking
            models.Index(fields=["is_sla_breached"], name="drift_sla_breached"),
            models.Index(fields=["remediation_due_at"], name="drift_remediation_due"),
            models.Index(fields=["resolved_at"], name="drift_resolved"),
            # Governance integration
            models.Index(fields=["is_governance_escalated"], name="drift_gov_escalated"),
            models.Index(fields=["correlation_id"], name="drift_correlation"),
        ]
        verbose_name = "Drift Violation"
        verbose_name_plural = "Drift Violations"

    class SynaraMeta:
        event_domain = "syn.audit.drift_violation"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        return f"[{self.enforcement_check}] {self.severity}: {self.file_path}:{self.line_number or '?'}"

    def is_overdue(self) -> bool:
        """Check if remediation is overdue."""
        if not self.remediation_due_at or self.resolved_at:
            return False
        return timezone.now() > self.remediation_due_at

    def save(self, *args, **kwargs):
        """Override save to compute derived fields."""
        # Compute remediation due date if SLA is set
        if self.remediation_sla_hours and not self.remediation_due_at:
            from datetime import timedelta
            self.remediation_due_at = self.detected_at + timedelta(hours=self.remediation_sla_hours)

        # Update SLA breach status
        if self.remediation_due_at and not self.resolved_at:
            self.is_sla_breached = timezone.now() > self.remediation_due_at

        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """
        SYS-200 INV-008: Drift violations are immutable audit records.

        Deletion is blocked to maintain compliance audit trail.
        Use resolved_at/resolution_notes to close violations instead.
        """
        raise ValueError(
            "DriftViolation entries cannot be deleted. "
            "SYS-200 INV-008 violation: audit records must be immutable. "
            "Use resolved_at and resolution_notes to close violations."
        )

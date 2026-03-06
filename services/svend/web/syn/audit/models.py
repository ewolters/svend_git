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
from django.utils import timezone

from syn.core.base_models import SynaraImmutableLog


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
        max_length=255, db_index=True, default="system", help_text="User or system component that performed the action"
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
            raise ValidationError("Audit log entries are immutable and cannot be modified. Create a new entry instead.")

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
            # Advisory lock serializes all chain appends for this tenant.
            # select_for_update() alone has a TOCTOU race: TX2 waits for TX1's
            # row lock, but its query already returned the stale "latest" entry.
            # pg_advisory_xact_lock blocks until no other transaction holds the
            # same lock, then we query the true latest entry.
            from django.db import connection

            lock_id = hash(str(self.tenant_id)) & 0x7FFFFFFF
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_advisory_xact_lock(%s)", [lock_id])

            previous_entry = SysLogEntry.objects.filter(tenant_id=self.tenant_id).order_by("-id").first()

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

    tenant_id = models.UUIDField(
        db_index=True, null=True, blank=True, help_text="Tenant affected by the violation (SEC-001 §5.2)"
    )

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

    is_resolved = models.BooleanField(
        default=False, help_text="Whether this violation has been investigated and resolved"
    )

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

    Standard: CMP-001 §5.6 - Drift Detection Lifecycle
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
        ("STD", "STD - Standards Compliance"),
        ("CAL", "CAL - Statistical Calibration"),
    ]

    # ========== Identity ==========

    id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False, help_text="Unique identifier for this drift violation"
    )

    drift_signature = models.CharField(
        max_length=64,
        unique=True,
        db_index=True,
        help_text="Unique hash signature of this drift violation (prevents duplicates)",
    )

    # ========== Classification ==========

    severity = models.CharField(
        max_length=10, choices=SEVERITY_CHOICES, db_index=True, help_text="Severity level of the drift violation"
    )

    enforcement_check = models.CharField(
        max_length=20,
        choices=ENFORCEMENT_CHECK_CHOICES,
        db_index=True,
        help_text="Enforcement check that detected this violation",
    )

    # ========== Location ==========

    file_path = models.CharField(max_length=512, db_index=True, help_text="File path where the violation was detected")

    line_number = models.IntegerField(null=True, blank=True, help_text="Line number where the violation was detected")

    function_name = models.CharField(
        max_length=255,
        null=True,
        blank=True,
        db_index=True,
        help_text="Function or class name where the violation was detected",
    )

    # ========== Detection Metadata ==========

    detected_at = models.DateTimeField(
        default=timezone.now, db_index=True, help_text="When this violation was detected"
    )

    detected_by = models.CharField(max_length=255, help_text="System or user that detected the violation")

    # ========== Git Context ==========

    git_commit_sha = models.CharField(
        max_length=40, null=True, blank=True, db_index=True, help_text="Git commit SHA where the violation was detected"
    )

    git_author = models.CharField(
        max_length=255, null=True, blank=True, help_text="Git commit author (for accountability)"
    )

    # ========== Violation Details ==========

    violation_message = models.TextField(help_text="Human-readable description of the violation")

    code_snippet = models.TextField(null=True, blank=True, help_text="Code snippet showing the violation")

    canonical_pattern = models.TextField(
        null=True, blank=True, help_text="Description or example of the canonical pattern"
    )

    # ========== Remediation ==========

    is_remediation_available = models.BooleanField(
        default=False, help_text="Whether automated remediation is available"
    )

    is_auto_fix_safe = models.BooleanField(
        default=False, help_text="Whether automated fix can be safely applied without review"
    )

    remediation_script = models.CharField(
        max_length=512, null=True, blank=True, help_text="Path to remediation script or tool"
    )

    remediation_sla_hours = models.IntegerField(
        null=True, blank=True, help_text="SLA hours for remediation based on severity"
    )

    remediation_due_at = models.DateTimeField(
        null=True, blank=True, db_index=True, help_text="When remediation is due (detected_at + SLA)"
    )

    is_sla_breached = models.BooleanField(
        default=False, db_index=True, help_text="Whether the remediation SLA has been breached"
    )

    # ========== Governance Integration ==========

    is_governance_escalated = models.BooleanField(
        default=False, db_index=True, help_text="Whether this violation has been escalated to governance"
    )

    governance_rule_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="Governance rule UUID (FK removed for Svend integration)"
    )

    governance_judgment_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="Governance judgment UUID (FK removed for Svend integration)"
    )

    # ========== Causal Trace Graph Integration ==========

    correlation_id = models.UUIDField(
        db_index=True, default=uuid.uuid4, help_text="Correlation ID for causal trace graph"
    )

    ctg_node_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="CTG node UUID (FK removed for Svend integration)"
    )

    # ========== Multi-Tenancy ==========

    tenant_id = models.UUIDField(
        db_index=True, null=True, blank=True, help_text="Tenant identifier for multi-tenant deployments"
    )

    # ========== Resolution ==========

    resolved_at = models.DateTimeField(
        null=True, blank=True, db_index=True, help_text="When this violation was resolved"
    )

    resolved_by = models.CharField(
        max_length=255, null=True, blank=True, help_text="User or system that resolved the violation"
    )

    resolution_notes = models.TextField(blank=True, help_text="Notes about how the violation was resolved")

    # UUID chain backlink — which ChangeRequest remediates this violation
    remediation_change_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="UUID of the ChangeRequest that remediates this violation (CHG-001 §8.4)",
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

    # Fields that may be updated post-creation (resolution tracking).
    # All other fields are immutable per SynaraImmutableLog contract.
    MUTABLE_FIELDS = {
        "resolved_at",
        "resolved_by",
        "resolution_notes",
        "remediation_change_id",
        "is_sla_breached",
        "is_governance_escalated",
        "governance_judgment_id",
    }

    def save(self, *args, **kwargs):
        """Save with immutability enforcement + resolution field exemption.

        New records: compute derived fields (SLA due date, breach status), then
        delegate to SynaraImmutableLog.save() for hash chain computation.

        Existing records: only MUTABLE_FIELDS may be updated. All others are
        blocked per AUD-001 / 21 CFR Part 11 §11.10(e). Resolution updates
        bypass the immutable base class via models.Model.save() directly.
        """
        if self.pk and self.__class__.objects.filter(pk=self.pk).exists():
            # Existing record — only allow mutable field updates
            update_fields = kwargs.get("update_fields")
            if not update_fields or not set(update_fields).issubset(self.MUTABLE_FIELDS):
                raise ValueError(
                    f"DriftViolation records are immutable except for resolution fields: "
                    f"{sorted(self.MUTABLE_FIELDS)}. Got: {update_fields}"
                )
            # Bypass SynaraImmutableLog.save() — go straight to models.Model.save()
            models.Model.save(self, *args, **kwargs)
            return

        # New record — compute derived fields before creation
        if self.remediation_sla_hours and not self.remediation_due_at:
            from datetime import timedelta

            self.remediation_due_at = self.detected_at + timedelta(hours=self.remediation_sla_hours)

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


class ComplianceCheck(models.Model):
    """
    Result of an automated compliance check.

    Each check verifies a specific security/compliance control and stores
    the result with details. Checks run on a rotating daily schedule.

    Compliance: SOC 2 CC4.1 (COSO Principle 16: Monitoring Activities)
    """

    STATUS_CHOICES = [
        ("pass", "Pass"),
        ("fail", "Fail"),
        ("warning", "Warning"),
        ("error", "Error"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    check_name = models.CharField(max_length=100, db_index=True)
    category = models.CharField(max_length=50)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    details = models.JSONField(default=dict)
    soc2_controls = models.JSONField(default=list)
    duration_ms = models.FloatField(default=0)
    run_at = models.DateTimeField(auto_now_add=True, db_index=True)
    tenant_id = models.UUIDField(null=True, blank=True, db_index=True)

    # UUID chain backlink — which ChangeRequest remediates this finding
    remediation_change_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="UUID of the ChangeRequest that remediates this finding (CHG-001 §8.4)",
    )

    class Meta:
        db_table = "syn_audit_compliance_check"
        ordering = ["-run_at"]
        indexes = [
            models.Index(fields=["check_name", "-run_at"], name="compliance_check_name_time"),
            models.Index(fields=["status", "-run_at"], name="compliance_status_time"),
        ]

    def __str__(self):
        return f"[{self.run_at}] {self.check_name}: {self.status}"


class HealthPing(models.Model):
    """
    Health endpoint ping result for real-time availability measurement.

    Replaces derived proxy metric (SLA-001 §5.1).
    Stored per-minute, retained 90 days, cleaned by audit.cleanup_violations task.

    Compliance: SOC 2 CC9.2 (Availability), SLA-001 §5.1
    """

    id = models.BigAutoField(primary_key=True)
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    is_healthy = models.BooleanField(help_text="True if /api/health/ returned 200 with expected JSON")
    status_code = models.IntegerField(null=True, blank=True, help_text="HTTP status code (null if connection failed)")
    response_time_ms = models.FloatField(null=True, blank=True, help_text="Response time in milliseconds")
    error = models.CharField(max_length=500, blank=True, default="", help_text="Error message if ping failed")

    class Meta:
        db_table = "audit_health_ping"
        ordering = ["-timestamp"]
        indexes = [
            models.Index(
                fields=["timestamp", "is_healthy"],
                name="health_ping_time_status",
            ),
        ]
        verbose_name = "Health Ping"
        verbose_name_plural = "Health Pings"

    def __str__(self):
        status = "UP" if self.is_healthy else "DOWN"
        return f"[{self.timestamp}] {status} ({self.response_time_ms}ms)"


# ---------------------------------------------------------------------------
# Incident Response Models (INC-001)
# ---------------------------------------------------------------------------


class Incident(models.Model):
    """
    Production incident record with lifecycle tracking and SLA measurement.

    Tracks incidents from detection through resolution and post-mortem with
    immutable IncidentLog entries at each state transition.

    Standard: INC-001 §3-§5
    Compliance: SOC 2 CC7.1 (Incident Detection), CC7.4 (Incident Response)
    """

    SEVERITY_CHOICES = [
        ("critical", "Critical"),
        ("high", "High"),
        ("medium", "Medium"),
        ("low", "Low"),
    ]

    STATUS_CHOICES = [
        ("detected", "Detected"),
        ("acknowledged", "Acknowledged"),
        ("investigating", "Investigating"),
        ("mitigating", "Mitigating"),
        ("resolved", "Resolved"),
        ("post_mortem", "Post-Mortem"),
        ("closed", "Closed"),
    ]

    CATEGORY_CHOICES = [
        ("outage", "Outage"),
        ("degradation", "Degradation"),
        ("security", "Security"),
        ("data", "Data Integrity"),
        ("dependency", "Dependency"),
        ("other", "Other"),
    ]

    # ========== Identity ==========

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    title = models.CharField(max_length=255, help_text="Short description of the incident")

    description = models.TextField(help_text="Detailed description of the incident and its impact")

    # ========== Classification ==========

    severity = models.CharField(
        max_length=10, choices=SEVERITY_CHOICES, db_index=True, help_text="Incident severity (INC-001 §3.1)"
    )

    status = models.CharField(
        max_length=15,
        choices=STATUS_CHOICES,
        default="detected",
        db_index=True,
        help_text="Current lifecycle state (INC-001 §5.1)",
    )

    category = models.CharField(
        max_length=15, choices=CATEGORY_CHOICES, default="other", help_text="Incident category (INC-001 §3.2)"
    )

    # ========== Lifecycle Timestamps (INC-001 §5.3) ==========

    detected_at = models.DateTimeField(default=timezone.now, help_text="When the incident was detected")

    acknowledged_at = models.DateTimeField(null=True, blank=True, help_text="When staff acknowledged the incident")

    investigating_at = models.DateTimeField(null=True, blank=True, help_text="When investigation began")

    mitigating_at = models.DateTimeField(null=True, blank=True, help_text="When mitigation/fix began")

    resolved_at = models.DateTimeField(null=True, blank=True, help_text="When the incident was resolved")

    closed_at = models.DateTimeField(
        null=True, blank=True, help_text="When post-mortem was completed and incident closed"
    )

    # ========== Actors ==========

    reported_by = models.CharField(max_length=255, default="system", help_text="Who or what reported the incident")

    assigned_to = models.CharField(
        max_length=255, blank=True, default="", help_text="Staff member assigned to the incident"
    )

    # ========== Linking ==========

    change_request = models.ForeignKey(
        "ChangeRequest",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="incidents",
        help_text="Remediation ChangeRequest (INC-001 §8.3)",
    )

    correlation_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="Correlation ID for audit trail linkage"
    )

    # ========== Resolution ==========

    root_cause = models.TextField(blank=True, default="", help_text="Root cause analysis (INC-001 §8.2)")

    resolution_summary = models.TextField(blank=True, default="", help_text="How the incident was resolved")

    post_mortem_notes = models.TextField(
        blank=True, default="", help_text="Post-mortem analysis, lessons learned, prevention measures"
    )

    class Meta:
        db_table = "syn_audit_incident"
        ordering = ["-detected_at"]
        indexes = [
            models.Index(fields=["severity", "status"], name="inc_sev_status"),
            models.Index(fields=["-detected_at"], name="inc_detected"),
        ]
        verbose_name = "Incident"
        verbose_name_plural = "Incidents"

    def __str__(self):
        return f"[{self.severity.upper()}] {self.title} ({self.status})"

    # ========== SLA Properties (INC-001 §6) ==========

    @property
    def ack_elapsed_hours(self):
        """Hours between detection and acknowledgement (or now if not yet acked)."""
        if not self.acknowledged_at:
            return (timezone.now() - self.detected_at).total_seconds() / 3600
        return (self.acknowledged_at - self.detected_at).total_seconds() / 3600

    @property
    def resolution_elapsed_hours(self):
        """Hours between detection and resolution (or now if not yet resolved)."""
        if not self.resolved_at:
            return (timezone.now() - self.detected_at).total_seconds() / 3600
        return (self.resolved_at - self.detected_at).total_seconds() / 3600

    @property
    def is_ack_sla_breached(self):
        """True if ack SLA target exceeded for this severity (SLA-001 §8)."""
        targets = {"critical": 1, "high": 1, "medium": 4, "low": 8}
        return self.ack_elapsed_hours > targets.get(self.severity, 8)

    @property
    def is_resolution_sla_breached(self):
        """True if resolution SLA target exceeded for this severity (SLA-001 §8)."""
        targets = {"critical": 8, "high": 24, "medium": 72, "low": 168}
        return self.resolution_elapsed_hours > targets.get(self.severity, 168)


class IncidentLog(models.Model):
    """
    Immutable log entry recording a state transition or event in an incident's lifecycle.

    Every Incident state transition, escalation, and comment creates an IncidentLog
    entry forming an auditable chain.

    Standard: INC-001 §5.2
    Compliance: SOC 2 CC7.4 (Incident Response)
    """

    ACTION_CHOICES = [
        ("detected", "Detected"),
        ("acknowledged", "Acknowledged"),
        ("investigating", "Investigating"),
        ("mitigating", "Mitigating"),
        ("resolved", "Resolved"),
        ("post_mortem", "Post-Mortem"),
        ("closed", "Closed"),
        ("escalated", "Escalated"),
        ("comment", "Comment"),
        ("reassigned", "Reassigned"),
        ("severity_changed", "Severity Changed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    incident = models.ForeignKey(
        Incident, on_delete=models.CASCADE, related_name="logs", help_text="The incident this log entry belongs to"
    )

    timestamp = models.DateTimeField(auto_now_add=True, db_index=True, help_text="When this log entry was created")

    actor = models.CharField(max_length=255, help_text="Who or what created this log entry")

    action = models.CharField(
        max_length=20, choices=ACTION_CHOICES, db_index=True, help_text="Type of action being logged"
    )

    from_state = models.CharField(max_length=20, blank=True, help_text="Previous state of the incident")

    to_state = models.CharField(max_length=20, blank=True, help_text="New state of the incident")

    details = models.JSONField(default=dict, help_text="Structured details: severity change, CR link, etc.")

    message = models.TextField(blank=True, help_text="Human-readable description of what happened")

    class Meta:
        db_table = "syn_audit_incident_log"
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["incident", "timestamp"], name="inclog_incident_time"),
            models.Index(fields=["action", "-timestamp"], name="inclog_action_time"),
        ]
        verbose_name = "Incident Log"
        verbose_name_plural = "Incident Logs"
        default_permissions = ("add", "view")  # Immutable — no change/delete

    def __str__(self):
        return (
            f"[{self.timestamp}] {self.action}: {self.message[:80]}"
            if self.message
            else f"[{self.timestamp}] {self.action}"
        )

    def save(self, *args, **kwargs):
        """Enforce immutability — prevent updates to existing log entries."""
        if self.pk and IncidentLog.objects.filter(pk=self.pk).exists():
            raise ValidationError(
                "Incident log entries are immutable and cannot be modified. "
                "Create a new entry instead. INC-001 §5.2 violation."
            )
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """Block deletion — incident logs are audit records."""
        raise ValueError(
            "Incident log entries cannot be deleted. "
            "INC-001 §5.2 / SOC 2 CC7.4 violation: incident audit trail must be preserved."
        )


class ComplianceReport(models.Model):
    """
    Monthly aggregate compliance report.

    Generated on the 1st of each month from the previous month's checks.
    Contains both a full internal report and a redacted public version.

    Compliance: SOC 2 CC4.1, CC2.1 (Communication and Information)
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    period_start = models.DateField()
    period_end = models.DateField()
    generated_at = models.DateTimeField(auto_now_add=True)
    total_checks = models.IntegerField(default=0)
    passed = models.IntegerField(default=0)
    failed = models.IntegerField(default=0)
    warnings = models.IntegerField(default=0)
    pass_rate = models.FloatField(default=0)
    summary = models.JSONField(default=dict)
    full_report = models.JSONField(default=dict)
    public_report = models.JSONField(default=dict)
    is_published = models.BooleanField(default=False)

    class Meta:
        db_table = "syn_audit_compliance_report"
        ordering = ["-period_start"]
        indexes = [
            models.Index(fields=["-period_start"], name="compliance_report_period"),
        ]

    def __str__(self):
        return f"Compliance Report {self.period_start} - {self.period_end} ({self.pass_rate:.0f}%)"


# ---------------------------------------------------------------------------
# Change Management Models (CHG-001)
# ---------------------------------------------------------------------------


class ChangeRequest(models.Model):
    """
    Formal record of a proposed change to the codebase, infrastructure, or configuration.

    Tracks the full lifecycle from planning through verification with linked logs,
    risk assessments, and audit trail integration.

    Standard: CHG-001 §4-§5
    Compliance: SOC 2 CC8.1 (Change Management), NIST SP 800-53 CM-3
    """

    CHANGE_TYPE_CHOICES = [
        ("feature", "Feature"),
        ("enhancement", "Enhancement"),
        ("bugfix", "Bug Fix"),
        ("hotfix", "Hotfix"),
        ("security", "Security"),
        ("infrastructure", "Infrastructure"),
        ("migration", "Migration"),
        ("documentation", "Documentation"),
        ("plan", "Plan"),
        ("debt", "Debt Closure"),
    ]

    RISK_LEVEL_CHOICES = [
        ("critical", "Critical"),
        ("high", "High"),
        ("medium", "Medium"),
        ("low", "Low"),
    ]

    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("submitted", "Submitted"),
        ("risk_assessed", "Risk Assessed"),
        ("approved", "Approved"),
        ("rejected", "Rejected"),
        ("in_progress", "In Progress"),
        ("testing", "Testing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("rolled_back", "Rolled Back"),
        ("cancelled", "Cancelled"),
    ]

    PRIORITY_CHOICES = [
        ("critical", "Critical"),
        ("high", "High"),
        ("medium", "Medium"),
        ("low", "Low"),
    ]

    # ========== Identity ==========

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    title = models.CharField(max_length=255, help_text="Short description of the change")

    description = models.TextField(help_text="Detailed description of what is being changed and why")

    # ========== Classification ==========

    change_type = models.CharField(
        max_length=20, choices=CHANGE_TYPE_CHOICES, db_index=True, help_text="Category of change (CHG-001 §4.1)"
    )

    risk_level = models.CharField(
        max_length=10,
        choices=RISK_LEVEL_CHOICES,
        default="medium",
        db_index=True,
        help_text="Risk classification (CHG-001 §4.2)",
    )

    priority = models.CharField(
        max_length=10, choices=PRIORITY_CHOICES, default="medium", help_text="Priority level per DEBT-001.md"
    )

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="draft",
        db_index=True,
        help_text="Current lifecycle state (CHG-001 §5.1)",
    )

    is_emergency = models.BooleanField(
        default=False, db_index=True, help_text="Emergency change flag — bypasses normal approval (CHG-001 §9)"
    )

    # ========== Planning ==========

    justification = models.TextField(blank=True, help_text="Business justification for the change")

    affected_files = models.JSONField(default=list, blank=True, help_text="List of files to be modified")

    implementation_plan = models.JSONField(default=dict, blank=True, help_text="Steps for implementing the change")

    rollback_plan = models.JSONField(default=dict, blank=True, help_text="Steps for reverting the change if it fails")

    testing_plan = models.JSONField(
        default=dict, blank=True, help_text="Steps for verifying the change after implementation"
    )

    # ========== Planning Linkage (CHG-001 §5.6.1) ==========

    feature_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="Feature UUID from planning system (FEAT-xxx)"
    )

    task_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="PlanTask UUID from planning system (TASK-xxx)"
    )

    # ========== Linking ==========

    issue_url = models.URLField(blank=True, help_text="Link to source issue (GitHub, internal tracker)")

    parent_change_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="Parent change request (for sub-tasks)"
    )

    related_change_ids = models.JSONField(default=list, blank=True, help_text="List of related change request UUIDs")

    debt_item = models.CharField(max_length=255, blank=True, help_text="Reference to DEBT.md item being closed")

    commit_shas = models.JSONField(default=list, blank=True, help_text="Git commit SHAs associated with this change")

    log_md_ref = models.CharField(max_length=255, blank=True, help_text="Reference to log.md section")

    # ========== UUID Linking (Synara Convention) ==========

    compliance_check_ids = models.JSONField(
        default=list, blank=True, help_text="UUIDs of ComplianceCheck records that triggered or relate to this change"
    )

    drift_violation_ids = models.JSONField(
        default=list, blank=True, help_text="UUIDs of DriftViolation records this change remediates"
    )

    audit_entry_ids = models.JSONField(
        default=list, blank=True, help_text="IDs of SysLogEntry records related to this change"
    )

    # ========== Lifecycle Timestamps ==========

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    submitted_at = models.DateTimeField(null=True, blank=True)
    approved_at = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    # ========== Actors ==========

    author = models.CharField(max_length=255, help_text="Who created the change request")

    approver = models.CharField(max_length=255, blank=True, help_text="Who approved the change")

    # ========== Correlation ==========

    correlation_id = models.UUIDField(
        default=uuid.uuid4, db_index=True, help_text="Correlation ID for audit trail linkage"
    )

    tenant_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="Tenant identifier for multi-tenant isolation"
    )

    class Meta:
        db_table = "syn_audit_change_request"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["change_type", "-created_at"], name="chg_type_created"),
            models.Index(fields=["status", "-created_at"], name="chg_status_created"),
            models.Index(fields=["risk_level", "-created_at"], name="chg_risk_created"),
            models.Index(fields=["is_emergency"], name="chg_emergency"),
            models.Index(fields=["correlation_id"], name="chg_correlation"),
            models.Index(fields=["author", "-created_at"], name="chg_author_created"),
        ]
        verbose_name = "Change Request"
        verbose_name_plural = "Change Requests"

    def __str__(self):
        return f"[{self.change_type}] {self.title} ({self.status})"

    # ========== Validation (CHG-001 §7.1.1) ==========

    # Types exempt from most field requirements
    EXEMPT_TYPES = {"documentation", "plan"}

    # Types requiring rollback_plan
    ROLLBACK_REQUIRED_TYPES = {"feature", "migration", "infrastructure", "security"}

    # Types requiring multi-agent risk assessment (4 votes)
    MULTI_AGENT_TYPES = {"feature", "migration"}

    # ALL types requiring any risk assessment
    RISK_ASSESSMENT_TYPES = {
        "feature",
        "enhancement",
        "bugfix",
        "security",
        "infrastructure",
        "migration",
        "debt",
    }

    # States that are "past approved" for enforcement purposes
    PAST_APPROVED = {"approved", "in_progress", "testing", "completed"}

    def clean(self):
        """Validate fields required at draft creation (CHG-001 §7.1.1).

        Called by full_clean(). Catches empty CRs at creation time.
        Transition-level validation is in validate_for_transition().
        """
        from django.core.exceptions import ValidationError

        errors = {}
        if not self.title or len(self.title.strip()) < 10:
            errors["title"] = "Title must be at least 10 characters."
        if not self.description or len(self.description.strip()) < 20:
            errors["description"] = "Description must be at least 20 characters."
        if not self.author:
            errors["author"] = "Author is required."
        if errors:
            raise ValidationError(errors)

    def validate_for_transition(self, target_state):
        """Validate fields required before transitioning to target_state.

        Called by api_change_transition. Returns list of error strings.
        Cumulative — checks all requirements up to the target state.
        CHG-001 §7.1.1 enforcement.
        """
        errors = []
        is_exempt = self.change_type in self.EXEMPT_TYPES

        SUBMITTED_PLUS = {
            "submitted",
            "risk_assessed",
            "approved",
            "in_progress",
            "testing",
            "completed",
        }
        APPROVED_PLUS = {"approved", "in_progress", "testing", "completed"}

        # submitted+ requirements
        if target_state in SUBMITTED_PLUS and not is_exempt:
            if not self.justification or not self.justification.strip():
                errors.append("justification is required before submission")
            if not self.affected_files:
                errors.append("affected_files is required before submission")

        # approved+ requirements
        if target_state in APPROVED_PLUS:
            if not is_exempt:
                if not self.implementation_plan or self.implementation_plan == {}:
                    errors.append("implementation_plan is required before approval")
                if not self.testing_plan or self.testing_plan == {}:
                    errors.append("testing_plan is required before approval")
            if self.change_type in self.ROLLBACK_REQUIRED_TYPES:
                if not self.rollback_plan or self.rollback_plan == {}:
                    errors.append("rollback_plan is required for this change type")
            if self.change_type in self.RISK_ASSESSMENT_TYPES:
                if not self.risk_assessments.exists():
                    errors.append(f"RiskAssessment required for {self.change_type} before approval")
            # IVR-001: Mechanical veto — security_analyst rejection blocks transition
            if self.change_type in self.MULTI_AGENT_TYPES:
                for ra in self.risk_assessments.all():
                    if ra.votes.filter(
                        agent_role="security_analyst",
                        recommendation="reject",
                    ).exists():
                        errors.append("Security analyst veto — change blocked (IVR-001)")
                        break

        # completed requirements
        if target_state == "completed" and not is_exempt:
            if not self.commit_shas:
                errors.append("commit_shas required before completion")
            if not self.log_md_ref:
                errors.append("log_md_ref required before completion")

        return errors

    # ========== Planning Linkage (CHG-001 §5.6.1) ==========

    def link_planning(self, feature_id=None, task_id=None, actor="system"):
        """Bidirectional link: set CR fields AND write back to planning models."""
        changed = []
        if feature_id and self.feature_id != feature_id:
            self.feature_id = feature_id
            changed.append("feature_id")
        if task_id and self.task_id != task_id:
            self.task_id = task_id
            changed.append("task_id")

        if not changed:
            return

        self.save(update_fields=changed + ["updated_at"])

        # Write back to planning models (best-effort)
        try:
            from api.models import Feature, PlanTask

            if task_id:
                PlanTask.objects.filter(id=task_id, change_request_id__isnull=True).update(change_request_id=self.id)
            if feature_id:
                feat = Feature.objects.filter(id=feature_id).first()
                if feat and str(self.id) not in [str(x) for x in feat.change_request_ids]:
                    feat.change_request_ids = feat.change_request_ids + [str(self.id)]
                    feat.save(update_fields=["change_request_ids", "updated_at"])
        except Exception:
            pass  # Planning models may not exist in all contexts

        ChangeLog.objects.create(
            change_request=self,
            actor=actor,
            action="linked",
            message=f"Linked to planning: {', '.join(changed)}",
            details={
                "feature_id": str(feature_id) if feature_id else None,
                "task_id": str(task_id) if task_id else None,
            },
        )

    # ========== Linking Methods (CHG-001 §8.4/§8.5) ==========

    def link_related(self, other_id, *, actor="system", message="", log=True):
        """Bidirectionally link this CR to another CR.

        Sets related_change_ids on both sides and creates ChangeLog entries
        on both CRs. This is the ONLY correct way to link CRs — never
        manipulate related_change_ids directly.

        Usage (CHG-001 §8.4)::

            cr.link_related(other_cr.id, actor="claude@svend.ai", message="Related: compliance expansion")

        Args:
            other_id: UUID string of the other ChangeRequest.
            actor: Who is creating the link.
            message: Optional context for the log entry.
            log: Whether to create ChangeLog entries (default True).

        Returns:
            True if link was created, False if already linked.
        """
        other_str = str(other_id)
        self_str = str(self.id)

        # Already linked?
        if other_str in (self.related_change_ids or []):
            return False

        # Forward
        if not self.related_change_ids:
            self.related_change_ids = []
        self.related_change_ids.append(other_str)
        self.save(update_fields=["related_change_ids", "updated_at"])

        # Reverse
        try:
            other = ChangeRequest.objects.get(id=other_str)
            if not other.related_change_ids:
                other.related_change_ids = []
            if self_str not in other.related_change_ids:
                other.related_change_ids.append(self_str)
                other.save(update_fields=["related_change_ids", "updated_at"])
        except ChangeRequest.DoesNotExist:
            pass  # External or future CR — forward link still valid

        if log:
            log_msg = message or f"Linked to CR {other_str[:8]}"
            ChangeLog.objects.create(
                change_request=self,
                actor=actor,
                action="linked",
                message=log_msg,
                details={"linked_cr": other_str, "direction": "bidirectional"},
                from_state=self.status,
                to_state=self.status,
            )

        return True

    def link_compliance_checks(self, check_ids, *, actor="system", message=""):
        """Bidirectionally link this CR to ComplianceCheck records (CHG-001 §8.5).

        Sets compliance_check_ids on this CR and remediation_change_id on each
        ComplianceCheck. Creates a single ChangeLog entry.

        Usage (CHG-001 §8.4/§8.5)::

            cr.link_compliance_checks(
                [str(c.id) for c in source_checks],
                actor="claude@svend.ai",
                message="Remediated session_security, error_handling warnings",
            )

        Args:
            check_ids: List of ComplianceCheck UUID strings.
            actor: Who is creating the link.
            message: Optional context for the log entry.
        """
        from syn.audit.models import ComplianceCheck

        self_str = str(self.id)
        check_strs = [str(cid) for cid in check_ids]
        check_names = []

        # Merge into compliance_check_ids (no duplicates)
        existing = set(self.compliance_check_ids or [])
        new_ids = [cid for cid in check_strs if cid not in existing]
        if not new_ids and all(cid in existing for cid in check_strs):
            return  # Already fully linked

        self.compliance_check_ids = list(existing | set(check_strs))
        self.save(update_fields=["compliance_check_ids", "updated_at"])

        # Reverse: set remediation_change_id on each check
        for cid in check_strs:
            try:
                check = ComplianceCheck.objects.get(id=cid)
                check.remediation_change_id = self_str
                check.save(update_fields=["remediation_change_id"])
                check_names.append(check.check_name)
            except ComplianceCheck.DoesNotExist:
                pass

        ChangeLog.objects.create(
            change_request=self,
            actor=actor,
            action="linked",
            message=message or f"Linked {len(check_strs)} compliance check(s)",
            details={
                "compliance_check_ids": check_strs,
                "check_names": check_names,
            },
            from_state=self.status,
            to_state=self.status,
        )

    def link_drift_violations(self, violation_ids, *, actor="system", message=""):
        """Bidirectionally link this CR to DriftViolation records.

        Sets drift_violation_ids on this CR and remediation_change_id on each
        DriftViolation. Creates a single ChangeLog entry.

        Usage (CHG-001 §8.4)::

            cr.link_drift_violations([violation.id], actor="claude@svend.ai")

        Args:
            violation_ids: List of DriftViolation UUID strings.
            actor: Who is creating the link.
            message: Optional context for the log entry.
        """
        from syn.audit.models import DriftViolation

        self_str = str(self.id)
        viol_strs = [str(vid) for vid in violation_ids]

        existing = set(self.drift_violation_ids or [])
        self.drift_violation_ids = list(existing | set(viol_strs))
        self.save(update_fields=["drift_violation_ids", "updated_at"])

        for vid in viol_strs:
            try:
                dv = DriftViolation.objects.get(id=vid)
                dv.remediation_change_id = self_str
                dv.save(update_fields=["remediation_change_id"])
            except DriftViolation.DoesNotExist:
                pass

        ChangeLog.objects.create(
            change_request=self,
            actor=actor,
            action="linked",
            message=message or f"Linked {len(viol_strs)} drift violation(s)",
            details={"drift_violation_ids": viol_strs},
            from_state=self.status,
            to_state=self.status,
        )


class ChangeLog(models.Model):
    """
    Immutable log entry recording a state transition or checkpoint in a change's lifecycle.

    Every ChangeRequest state transition, risk assessment, approval, and verification
    creates a ChangeLog entry forming an auditable chain.

    Standard: CHG-001 §5.2, §8.1
    Compliance: SOC 2 CC8.1, NIST SP 800-53 CM-3
    """

    ACTION_CHOICES = [
        ("plan_created", "Plan Created"),
        ("submitted", "Submitted"),
        ("risk_assessed", "Risk Assessed"),
        ("approved", "Approved"),
        ("rejected", "Rejected"),
        ("implementation_started", "Implementation Started"),
        ("testing_completed", "Testing Completed"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("rolled_back", "Rolled Back"),
        ("cancelled", "Cancelled"),
        ("comment", "Comment"),
        ("linked", "Linked"),
        ("retroactive_review", "Retroactive Review"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    change_request = models.ForeignKey(
        ChangeRequest,
        on_delete=models.CASCADE,
        related_name="logs",
        help_text="The change request this log entry belongs to",
    )

    timestamp = models.DateTimeField(auto_now_add=True, db_index=True, help_text="When this log entry was created")

    actor = models.CharField(max_length=255, help_text="Who or what created this log entry")

    action = models.CharField(
        max_length=30, choices=ACTION_CHOICES, db_index=True, help_text="Type of action being logged"
    )

    from_state = models.CharField(max_length=30, blank=True, help_text="Previous state of the change request")

    to_state = models.CharField(max_length=30, blank=True, help_text="New state of the change request")

    details = models.JSONField(
        default=dict, help_text="Structured details: commit_sha, log_md_ref, issue_url, test_results, etc."
    )

    message = models.TextField(blank=True, help_text="Human-readable description of what happened")

    class Meta:
        db_table = "syn_audit_change_log"
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["change_request", "timestamp"], name="chglog_request_time"),
            models.Index(fields=["action", "-timestamp"], name="chglog_action_time"),
        ]
        verbose_name = "Change Log"
        verbose_name_plural = "Change Logs"
        default_permissions = ("add", "view")  # Immutable — no change/delete

    def __str__(self):
        return (
            f"[{self.timestamp}] {self.action}: {self.message[:80]}"
            if self.message
            else f"[{self.timestamp}] {self.action}"
        )

    def save(self, *args, **kwargs):
        """Enforce immutability — prevent updates to existing log entries."""
        if self.pk and ChangeLog.objects.filter(pk=self.pk).exists():
            raise ValidationError(
                "Change log entries are immutable and cannot be modified. "
                "Create a new entry instead. CHG-001 §5.2 violation."
            )
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """Block deletion — change logs are audit records."""
        raise ValueError(
            "Change log entries cannot be deleted. "
            "CHG-001 §5.2 / SOC 2 CC8.1 violation: change audit trail must be preserved."
        )


class RiskAssessment(models.Model):
    """
    Risk analysis for a change request with multi-agent voting.

    Evaluates impact across SOC 2 trust service categories:
    Security, Availability, Processing Integrity, Confidentiality, Privacy.

    Standard: CHG-001 §6
    Compliance: SOC 2 CC3.4, NIST SP 800-53 CM-4
    """

    ASSESSMENT_TYPE_CHOICES = [
        ("multi_agent", "Multi-Agent"),
        ("single_agent", "Single Agent"),
        ("expedited", "Expedited"),
        ("automated", "Automated"),
    ]

    RECOMMENDATION_CHOICES = [
        ("approve", "Approve"),
        ("approve_with_conditions", "Approve with Conditions"),
        ("reject", "Reject"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    change_request = models.ForeignKey(
        ChangeRequest,
        on_delete=models.CASCADE,
        related_name="risk_assessments",
        help_text="The change request being assessed",
    )

    assessment_type = models.CharField(
        max_length=20, choices=ASSESSMENT_TYPE_CHOICES, help_text="Type of risk assessment performed"
    )

    # Aggregate risk scores (1-5 per dimension)
    security_score = models.FloatField(default=0, help_text="Security risk score (1-5)")
    availability_score = models.FloatField(default=0, help_text="Availability risk score (1-5)")
    integrity_score = models.FloatField(default=0, help_text="Processing integrity risk score (1-5)")
    confidentiality_score = models.FloatField(default=0, help_text="Confidentiality risk score (1-5)")
    privacy_score = models.FloatField(default=0, help_text="Privacy risk score (1-5)")
    overall_score = models.FloatField(default=0, help_text="Overall risk score (max of dimensions)")

    overall_recommendation = models.CharField(
        max_length=30,
        choices=RECOMMENDATION_CHOICES,
        default="approve",
        help_text="Aggregate recommendation from all votes",
    )

    conditions = models.JSONField(default=list, help_text="Required conditions/mitigations if approve_with_conditions")

    summary = models.TextField(blank=True, help_text="Human-readable summary of the risk assessment")

    is_retroactive = models.BooleanField(
        default=False, help_text="True if this is a retroactive assessment for an emergency change"
    )

    assessed_at = models.DateTimeField(auto_now_add=True, db_index=True)

    assessed_by = models.CharField(max_length=255, default="system", help_text="Who or what performed the assessment")

    class Meta:
        db_table = "syn_audit_risk_assessment"
        ordering = ["-assessed_at"]
        indexes = [
            models.Index(fields=["change_request", "-assessed_at"], name="risk_change_time"),
            models.Index(fields=["overall_recommendation"], name="risk_recommendation"),
        ]
        verbose_name = "Risk Assessment"
        verbose_name_plural = "Risk Assessments"

    def __str__(self):
        return f"Risk Assessment for {self.change_request_id}: {self.overall_recommendation} ({self.overall_score:.1f})"

    def compute_aggregate(self):
        """Compute aggregate scores from agent votes."""
        votes = self.votes.all()
        if not votes:
            return

        scores = {"security": [], "availability": [], "integrity": [], "confidentiality": [], "privacy": []}
        recommendations = []

        for vote in votes:
            risk_scores = vote.risk_scores or {}
            for dim in scores:
                if dim in risk_scores:
                    scores[dim].append(risk_scores[dim])
            recommendations.append(vote.recommendation)

        # Per-dimension: average
        self.security_score = sum(scores["security"]) / len(scores["security"]) if scores["security"] else 0
        self.availability_score = (
            sum(scores["availability"]) / len(scores["availability"]) if scores["availability"] else 0
        )
        self.integrity_score = sum(scores["integrity"]) / len(scores["integrity"]) if scores["integrity"] else 0
        self.confidentiality_score = (
            sum(scores["confidentiality"]) / len(scores["confidentiality"]) if scores["confidentiality"] else 0
        )
        self.privacy_score = sum(scores["privacy"]) / len(scores["privacy"]) if scores["privacy"] else 0

        # Overall: max of dimensions
        self.overall_score = max(
            self.security_score,
            self.availability_score,
            self.integrity_score,
            self.confidentiality_score,
            self.privacy_score,
        )

        # Recommendation: majority vote, security_analyst has veto on reject
        if any(v.recommendation == "reject" and v.agent_role == "security_analyst" for v in votes):
            self.overall_recommendation = "reject"
        elif recommendations.count("reject") > len(recommendations) / 2:
            self.overall_recommendation = "reject"
        elif recommendations.count("approve_with_conditions") > 0:
            self.overall_recommendation = "approve_with_conditions"
            # Merge conditions from all approve_with_conditions votes
            all_conditions = []
            for v in votes:
                if v.recommendation == "approve_with_conditions" and v.conditions:
                    all_conditions.extend(v.conditions)
            self.conditions = all_conditions
        else:
            self.overall_recommendation = "approve"

        self.save()


class AgentVote(models.Model):
    """
    Individual agent assessment during multi-agent risk evaluation.

    Each agent evaluates from a different role perspective to ensure
    diverse risk coverage.

    Standard: CHG-001 §6.2
    Compliance: SOC 2 CC3.4 (Risk Assessment)
    """

    AGENT_ROLE_CHOICES = [
        ("security_analyst", "Security Analyst"),
        ("architect", "Architect"),
        ("operations", "Operations"),
        ("quality", "Quality"),
    ]

    RECOMMENDATION_CHOICES = [
        ("approve", "Approve"),
        ("approve_with_conditions", "Approve with Conditions"),
        ("reject", "Reject"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    risk_assessment = models.ForeignKey(
        RiskAssessment,
        on_delete=models.CASCADE,
        related_name="votes",
        help_text="The risk assessment this vote belongs to",
    )

    agent_role = models.CharField(
        max_length=30, choices=AGENT_ROLE_CHOICES, help_text="Role perspective of the voting agent"
    )

    recommendation = models.CharField(max_length=30, choices=RECOMMENDATION_CHOICES, help_text="Agent's recommendation")

    risk_scores = models.JSONField(
        default=dict, help_text="Risk scores per dimension: {security: 1-5, availability: 1-5, ...}"
    )

    rationale = models.TextField(help_text="Structured justification for the recommendation")

    conditions = models.JSONField(default=list, help_text="Required mitigations if approve_with_conditions")

    voted_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "syn_audit_agent_vote"
        ordering = ["voted_at"]
        indexes = [
            models.Index(fields=["risk_assessment", "agent_role"], name="vote_assessment_role"),
        ]
        verbose_name = "Agent Vote"
        verbose_name_plural = "Agent Votes"

    def __str__(self):
        return f"[{self.agent_role}] {self.recommendation}"


# ---------------------------------------------------------------------------
# Risk Registry (RISK-001, FEAT-090)
# ---------------------------------------------------------------------------


class RiskEntry(models.Model):
    """
    Persistent risk register entry with FMEA-style RPN scoring.

    Tracks identified risks beyond individual ChangeRequests — enables
    trending, mitigation tracking, and compliance verification.

    Standard: RISK-001 §3
    Compliance: SOC 2 CC3.2 (Risk Assessment), CC9.1 (Risk Mitigation)
    References: AS9100 §6.1, ISO 9001 §6.1, FDA 21 CFR 820.30(g)
    """

    CATEGORY_CHOICES = [
        ("security", "Security"),
        ("availability", "Availability"),
        ("integrity", "Integrity"),
        ("confidentiality", "Confidentiality"),
        ("privacy", "Privacy"),
        ("operational", "Operational"),
        ("compliance", "Compliance"),
    ]

    STATUS_CHOICES = [
        ("identified", "Identified"),
        ("mitigating", "Mitigating"),
        ("accepted", "Accepted"),
        ("mitigated", "Mitigated"),
        ("closed", "Closed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=200)
    description = models.TextField()
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    likelihood = models.IntegerField(help_text="1-5: probability of occurrence")
    severity = models.IntegerField(help_text="1-5: impact if it occurs")
    detectability = models.IntegerField(help_text="1-5: difficulty of detection (5=hardest)")
    rpn = models.IntegerField(default=0, help_text="Risk Priority Number (L×S×D, max 125)")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="identified")
    mitigation_plan = models.TextField(blank=True, default="")
    owner = models.CharField(max_length=100, help_text="Person or role responsible")
    source_cr = models.ForeignKey(
        "ChangeRequest",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="risk_entries",
        help_text="Originating ChangeRequest",
    )
    related_crs = models.JSONField(default=list, blank=True, help_text="UUIDs of related ChangeRequests")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "syn_audit_risk_entry"
        ordering = ["-rpn", "-created_at"]
        indexes = [
            models.Index(fields=["status"], name="risk_entry_status"),
            models.Index(fields=["-rpn"], name="risk_entry_rpn"),
        ]
        verbose_name = "Risk Entry"
        verbose_name_plural = "Risk Entries"

    def save(self, *args, **kwargs):
        self.rpn = self.likelihood * self.severity * self.detectability
        super().save(*args, **kwargs)

    @property
    def risk_level(self):
        """Return risk level based on RPN thresholds."""
        if self.rpn > 60:
            return "high"
        if self.rpn > 20:
            return "medium"
        return "low"

    def __str__(self):
        return f"[{self.risk_level.upper()}] {self.title} (RPN={self.rpn})"


# ---------------------------------------------------------------------------
# Calibration Reports (CAL-001)
# ---------------------------------------------------------------------------


class CalibrationReport(models.Model):
    """
    Point-in-time calibration snapshot.

    Records code coverage, calibration pass rates, endpoint coverage,
    and golden file counts. The ratchet_baseline field ensures coverage
    never decreases between cycles.

    Standard: CAL-001 §11
    Compliance: SOC 2 CC4.1 (Monitoring), ISO/IEC 17025:2017
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    date = models.DateField(db_index=True)
    overall_coverage = models.FloatField(null=True, blank=True, help_text="Overall line coverage %")
    tier1_coverage = models.FloatField(null=True, blank=True, help_text="Tier 1 (Customer Trust) coverage %")
    tier2_coverage = models.FloatField(null=True, blank=True, help_text="Tier 2 (Revenue Path) coverage %")
    tier3_coverage = models.FloatField(null=True, blank=True, help_text="Tier 3 (Feature Surface) coverage %")
    tier4_coverage = models.FloatField(null=True, blank=True, help_text="Tier 4 (Infrastructure) coverage %")
    calibration_pass_rate = models.FloatField(null=True, blank=True, help_text="Calibration case pass rate %")
    calibration_cases_run = models.IntegerField(default=0, help_text="Number of calibration cases executed")
    calibration_cases_passed = models.IntegerField(default=0, help_text="Number of calibration cases that passed")
    endpoint_coverage = models.FloatField(null=True, blank=True, help_text="Endpoint smoke test coverage %")
    golden_file_count = models.IntegerField(default=0, help_text="Number of golden files in agents_api/tests/golden/")
    complexity_violations = models.IntegerField(
        default=0, help_text="Files exceeding 3000-line limit without exemption"
    )
    ratchet_baseline = models.FloatField(default=0.0, help_text="Coverage floor — next report must meet or exceed this")
    is_certificate = models.BooleanField(default=False, help_text="True for monthly calibration certificates")
    details = models.JSONField(default=dict, help_text="Per-module breakdown and additional metrics")

    class Meta:
        db_table = "syn_audit_calibration_report"
        ordering = ["-date"]
        indexes = [
            models.Index(fields=["-date"], name="calibration_report_date"),
        ]
        default_permissions = ("add", "view")  # Immutable — no change/delete
        verbose_name = "Calibration Report"
        verbose_name_plural = "Calibration Reports"

    def __str__(self):
        cert = " [CERTIFICATE]" if self.is_certificate else ""
        cov = f"{self.overall_coverage:.1f}%" if self.overall_coverage is not None else "unmeasured"
        return f"Calibration {self.date} — {cov}{cert}"

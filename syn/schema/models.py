"""
Schema models for SCHEMA-001 compliant universal cognitive schema framework.

Provides versioned schemas for event validation, entity definitions,
field nodes, and behavior bindings with checksums and tenant isolation.

Standard: SCHEMA-001 §5
Compliance: SOC 2 CC6.1, CC7.2 / ISO 27001 A.8.1, A.8.3 / ISO 9001:2015 §7.5.3
"""

import hashlib
import json
import uuid
from typing import Optional

from django.db import models
from django.utils import timezone

from syn.core.base_models import SynaraEntity, SynaraImmutableLog

# =============================================================================
# Constants (SCHEMA-001 §5)
# =============================================================================

SCHEMA_STATUS_CHOICES = [
    ("DRAFT", "Draft"),
    ("REVIEW", "Review"),
    ("APPROVED", "Approved"),
    ("ACTIVE", "Active"),
    ("DEPRECATED", "Deprecated"),
    ("RETIRED", "Retired"),
]

NODE_TYPE_CHOICES = [
    ("string", "String"),
    ("integer", "Integer"),
    ("float", "Float"),
    ("boolean", "Boolean"),
    ("datetime", "DateTime"),
    ("object", "Object"),
    ("array", "Array"),
    ("reference", "Reference"),
]

BEHAVIOR_TRIGGER_CHOICES = [
    ("on_create", "On Create"),
    ("on_update", "On Update"),
    ("on_delete", "On Delete"),
    ("on_validate", "On Validate"),
]

BEHAVIOR_ACTION_CHOICES = [
    ("emit_event", "Emit Event"),
    ("generate_task", "Generate Task"),
    ("invoke_policy", "Invoke Policy"),
    ("invoke_reasoning", "Invoke Reasoning"),
]

GOVERNANCE_TAG_CHOICES = [
    ("PII", "Personally Identifiable Information"),
    ("PHI", "Protected Health Information"),
    ("FINANCIAL", "Financial Data"),
    ("SAFETY", "Safety Critical"),
    ("PROPRIETARY", "Proprietary Information"),
]


class EventSchema(SynaraEntity):
    """
    Versioned JSON schema for event payload validation.

    Stores JSON schemas for events with version control, checksums,
    and activation status. Supports tenant-specific schemas.

    Features:
    - Version control: Multiple versions per event
    - Checksums: SHA-256 hash for integrity verification
    - Activation: Only one active version per event/tenant
    - Tenant isolation: Separate schemas per tenant

    Compliance:
    - SOC 2 CC6.6: Logical and physical access controls
    - ISO 27001 A.14.2.7: Secure system engineering principles
    """

    id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False, help_text="Unique identifier for this schema"
    )

    name = models.CharField(max_length=255, db_index=True, help_text="Event name (e.g., 'user.login', 'task.created')")

    version = models.CharField(max_length=50, help_text="Schema version (e.g., '1.0.0', '2.1.0')")

    checksum = models.CharField(max_length=64, help_text="SHA-256 checksum of schema_json")

    schema_json = models.JSONField(help_text="JSON Schema definition (draft-07 compatible)")

    is_active = models.BooleanField(
        default=False, db_index=True, help_text="Whether this is the active version for validation"
    )

    tenant_id = models.UUIDField(
        db_index=True, null=True, blank=True, help_text="Tenant identifier (null for global schemas) (SEC-001 §5.2)"
    )

    created_at = models.DateTimeField(auto_now_add=True, help_text="When this schema version was created")

    updated_at = models.DateTimeField(auto_now=True, help_text="Last update timestamp")

    description = models.TextField(blank=True, help_text="Human-readable description of this schema version")

    class Meta(SynaraEntity.Meta):
        db_table = "schema_event_schema"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["name", "is_active"], name="schema_name_active"),
            models.Index(fields=["tenant_id", "name"], name="schema_tenant_name"),
            models.Index(fields=["checksum"], name="schema_checksum"),
        ]
        unique_together = [
            ["name", "version", "tenant_id"],  # Unique version per event/tenant
        ]
        verbose_name = "Event Schema"
        verbose_name_plural = "Event Schemas"

    class SynaraMeta:
        event_domain = "syn.schema.event_schema"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        tenant_str = f"[{self.tenant_id}] " if self.tenant_id else "[Global] "
        active_str = " (active)" if self.is_active else ""
        return f"{tenant_str}{self.name} v{self.version}{active_str}"

    def save(self, *args, **kwargs):
        """
        Override save to compute checksum and enforce activation rules.

        - Automatically computes SHA-256 checksum of schema_json
        - Deactivates other versions when activating this one
        """
        # Compute checksum
        self.checksum = self._compute_checksum()

        # If activating this schema, deactivate others
        if self.is_active:
            EventSchema.objects.filter(name=self.name, tenant_id=self.tenant_id, is_active=True).exclude(
                id=self.id
            ).update(is_active=False)

        super().save(*args, **kwargs)

    def _compute_checksum(self) -> str:
        """
        Compute SHA-256 checksum of schema_json.

        Returns:
            Hexadecimal checksum string
        """
        # Serialize schema to JSON with sorted keys for consistency
        schema_json = json.dumps(self.schema_json, sort_keys=True)
        return hashlib.sha256(schema_json.encode("utf-8")).hexdigest()

    def verify_checksum(self) -> bool:
        """
        Verify that stored checksum matches computed checksum.

        Returns:
            True if checksum is valid, False otherwise
        """
        return self.checksum == self._compute_checksum()

    @classmethod
    def get_active_schema(cls, event_name: str, tenant_id: str):
        """
        Get the active schema for an event and tenant.

        TNT-INV-001 Exception: This method has a DOCUMENTED fallback to global
        schemas when tenant_id is provided but no tenant-specific schema exists.
        This is intentional - schemas can be global (platform-wide) or tenant-specific.

        Args:
            event_name: Name of the event
            tenant_id: Required tenant identifier. Pass explicit value; the method
                       will automatically fall back to global schemas if no
                       tenant-specific schema exists.

        Returns:
            Active EventSchema or None

        Priority:
        1. Tenant-specific active schema (if tenant_id provided)
        2. Global active schema (tenant_id__isnull=True)

        Raises:
            ValueError: If event_name is None or empty
        """
        if not event_name:
            raise ValueError("event_name is required")

        # Try tenant-specific first
        if tenant_id:
            schema = cls.objects.filter(name=event_name, tenant_id=tenant_id, is_active=True).first()

            if schema:
                return schema

        # Fall back to global schema (this is intentional - schemas can be platform-wide)
        return cls.objects.filter(name=event_name, tenant_id__isnull=True, is_active=True).first()

    @classmethod
    def get_all_versions(cls, event_name: str, tenant_id: str, include_global: bool = True):
        """
        Get all schema versions for an event.

        TNT-INV-001 Compliant: tenant_id is now required. Use include_global=True
        to also retrieve global (platform-wide) schemas.

        Args:
            event_name: Name of the event
            tenant_id: Required tenant identifier (SEC-001 §5.2.1)
            include_global: If True, also include global schemas (tenant_id=None)

        Returns:
            QuerySet of EventSchema instances

        Raises:
            ValueError: If event_name is None or empty
        """
        if not event_name:
            raise ValueError("event_name is required")

        from django.db.models import Q

        if include_global:
            # Include both tenant-specific and global schemas
            queryset = cls.objects.filter(Q(tenant_id=tenant_id) | Q(tenant_id__isnull=True), name=event_name)
        else:
            # Tenant-specific only
            if not tenant_id:
                raise ValueError("tenant_id is required when include_global=False (TNT-INV-001)")
            queryset = cls.objects.filter(name=event_name, tenant_id=tenant_id)

        return queryset.order_by("-created_at")


class SchemaValidationLog(SynaraImmutableLog):
    """
    Log of schema validation attempts.

    Records validation successes and failures for auditing
    and monitoring data quality.

    Compliance: SOC 2 CC6.6 - Access control monitoring
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    timestamp = models.DateTimeField(default=timezone.now, db_index=True, help_text="When validation occurred")

    event_name = models.CharField(max_length=255, db_index=True, help_text="Event name being validated")

    tenant_id = models.UUIDField(db_index=True, null=True, blank=True, help_text="Tenant identifier (SEC-001 §5.2)")

    schema = models.ForeignKey(
        EventSchema, on_delete=models.SET_NULL, null=True, blank=True, help_text="Schema used for validation"
    )

    valid = models.BooleanField(help_text="Whether validation passed")

    errors = models.JSONField(default=list, help_text="Validation errors if any")

    payload_sample = models.JSONField(null=True, blank=True, help_text="Sample of the payload (for debugging)")

    correlation_id = models.UUIDField(null=True, blank=True, db_index=True, help_text="Correlation ID from event")

    class Meta(SynaraImmutableLog.Meta):
        db_table = "schema_validation_log"
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["event_name", "timestamp"], name="schemalog_event_time"),
            models.Index(fields=["tenant_id", "valid"], name="schemalog_tenant_valid"),
            models.Index(fields=["valid"], name="schemalog_valid"),
        ]
        verbose_name = "Schema Validation Log"
        verbose_name_plural = "Schema Validation Logs"

    class SynaraMeta:
        event_domain = "syn.schema.schema_validation_log"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        status = "✓" if self.valid else "✗"
        return f"{status} {self.event_name} @ {self.timestamp}"


class Schema(SynaraEntity):
    """
    Universal schema registry for entity definitions.

    Standard: SCHEMA-001 §5 (core_concepts.schema)
    Table: syn_schema

    Features:
    - Domain-based namespace organization
    - Semantic versioning with governance
    - Multi-tenant schema isolation
    - Governance workflow integration

    Compliance:
    - ISO 9001:2015 §7.5.3: Document Control
    - ISO 27001 A.8.1: Asset Classification
    - CGS-1001 §6: Governance Approval Workflow
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, help_text="Unique schema identifier")
    correlation_id = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        editable=False,
        db_index=True,
        help_text="Correlation ID for causal trace graph (CTG-001 §5)",
    )
    tenant_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="Tenant identifier (null for global schemas, SEC-001 §5.2)"
    )
    domain = models.CharField(
        max_length=100, db_index=True, help_text="Domain namespace (e.g., 'qms', 'healthcare', 'manufacturing')"
    )
    entity = models.CharField(
        max_length=100, db_index=True, help_text="Entity type (e.g., 'ncr', 'patient', 'workorder')"
    )
    version = models.CharField(max_length=50, help_text="Semantic version (MAJOR.MINOR.PATCH)")
    definition = models.JSONField(help_text="Full schema definition in JSON/YAML format")
    status = models.CharField(
        max_length=15,
        choices=SCHEMA_STATUS_CHOICES,
        default="DRAFT",
        db_index=True,
        help_text="Schema lifecycle status",
    )
    governance_tags = models.JSONField(
        default=list, blank=True, help_text="Governance tags (PII, PHI, SAFETY, FINANCIAL, PROPRIETARY)"
    )
    governance_metadata = models.JSONField(default=dict, blank=True, help_text="CGS-1001 approval workflow metadata")
    checksum = models.CharField(max_length=64, help_text="SHA-256 checksum of definition")
    created_at = models.DateTimeField(auto_now_add=True, help_text="When this schema was created")
    updated_at = models.DateTimeField(auto_now=True, help_text="When this schema was last updated")
    created_by = models.CharField(max_length=255, null=True, blank=True, help_text="User ID of schema author")

    class Meta(SynaraEntity.Meta):
        db_table = "syn_schema"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["domain", "entity"], name="idx_schema_domain_entity"),
            models.Index(fields=["tenant_id", "status"], name="idx_schema_tenant_status"),
            models.Index(fields=["status", "domain"], name="idx_schema_status_domain"),
        ]
        unique_together = [
            ["domain", "entity", "version", "tenant_id"],
        ]
        verbose_name = "Schema"
        verbose_name_plural = "Schemas"

    class SynaraMeta:
        event_domain = "syn.schema.schema"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        tenant_str = f"[T:{self.tenant_id}] " if self.tenant_id else "[Global] "
        return f"{tenant_str}{self.domain}.{self.entity} v{self.version} ({self.status})"

    def save(self, *args, **kwargs):
        """Compute checksum before saving."""
        self.checksum = self._compute_checksum()
        super().save(*args, **kwargs)

    def _compute_checksum(self) -> str:
        """Compute SHA-256 checksum of definition."""
        definition_json = json.dumps(self.definition, sort_keys=True)
        return hashlib.sha256(definition_json.encode("utf-8")).hexdigest()

    @property
    def schema_id(self) -> str:
        """Get the canonical schema ID."""
        return f"syn.{self.domain}.{self.entity}.v{self.version.split('.')[0]}"


class SchemaNode(SynaraEntity):
    """
    Field or entity node within a schema graph.

    Standard: SCHEMA-001 §5 (core_concepts.schema_node)
    Table: syn_schema_node

    Features:
    - Typed field definitions
    - Constraint specifications
    - Cognitive metadata for AI
    - Governance tag inheritance

    Compliance:
    - ISO 9001:2015 §8.5.1: Process Control
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, help_text="Unique node identifier")
    correlation_id = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        editable=False,
        db_index=True,
        help_text="Correlation ID for causal trace graph (CTG-001 §5)",
    )
    schema = models.ForeignKey(Schema, on_delete=models.CASCADE, related_name="nodes", help_text="Parent schema")
    name = models.CharField(max_length=100, help_text="Field/node name")
    node_type = models.CharField(max_length=20, choices=NODE_TYPE_CHOICES, help_text="Data type for this node")
    required = models.BooleanField(default=False, help_text="Whether this field is required")
    default_value = models.JSONField(null=True, blank=True, help_text="Default value for this field")
    constraints = models.JSONField(
        default=dict, blank=True, help_text="Validation constraints (min, max, regex, enum, format)"
    )
    metadata = models.JSONField(
        default=dict, blank=True, help_text="Display metadata (display_name, description, units)"
    )
    cognitive_metadata = models.JSONField(
        default=dict, blank=True, help_text="Cognitive metadata (confidence_threshold, reasoning_workflow)"
    )
    governance_tags = models.JSONField(
        default=list, blank=True, help_text="Governance tags for this field (PII, PHI, etc.)"
    )
    order = models.PositiveIntegerField(default=0, help_text="Display order within schema")
    created_at = models.DateTimeField(auto_now_add=True, help_text="When this node was created")
    updated_at = models.DateTimeField(auto_now=True, help_text="When this node was last updated")

    class Meta(SynaraEntity.Meta):
        db_table = "syn_schema_node"
        ordering = ["schema", "order", "name"]
        indexes = [
            models.Index(fields=["schema", "name"], name="idx_schemanode_schema_name"),
            models.Index(fields=["node_type"], name="idx_schemanode_type"),
        ]
        unique_together = [
            ["schema", "name"],
        ]
        verbose_name = "Schema Node"
        verbose_name_plural = "Schema Nodes"

    class SynaraMeta:
        event_domain = "syn.schema.schema_node"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        req = "*" if self.required else ""
        return f"{self.schema.entity}.{self.name}{req} ({self.node_type})"


class SchemaBehavior(SynaraEntity):
    """
    Declarative behavior binding for schema events.

    Standard: SCHEMA-001 §5 (core_concepts.schema_behavior)
    Table: syn_schema_behavior

    Features:
    - Trigger-based event emission
    - Task generation integration
    - Policy invocation binding
    - Reasoning workflow hooks

    Compliance:
    - EVT-001 §4: Event Taxonomy
    - SCH-001 §18: DAG Integration
    - POL-001 §8: Policy Framework
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, help_text="Unique behavior identifier")
    correlation_id = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        editable=False,
        db_index=True,
        help_text="Correlation ID for causal trace graph (CTG-001 §5)",
    )
    schema = models.ForeignKey(Schema, on_delete=models.CASCADE, related_name="behaviors", help_text="Parent schema")
    name = models.CharField(max_length=100, help_text="Behavior name")
    trigger_event = models.CharField(
        max_length=20, choices=BEHAVIOR_TRIGGER_CHOICES, help_text="When this behavior triggers"
    )
    action_type = models.CharField(
        max_length=20, choices=BEHAVIOR_ACTION_CHOICES, help_text="Type of action to perform"
    )
    action_config = models.JSONField(
        default=dict, help_text="Configuration for action (event_name, task_reflex, policy_namespace)"
    )
    condition = models.JSONField(
        null=True, blank=True, help_text="Optional JSONLogic condition for conditional execution"
    )
    priority = models.PositiveIntegerField(default=100, help_text="Execution priority (lower = higher priority)")
    is_active = models.BooleanField(default=True, db_index=True, help_text="Whether this behavior is active")
    created_at = models.DateTimeField(auto_now_add=True, help_text="When this behavior was created")
    updated_at = models.DateTimeField(auto_now=True, help_text="When this behavior was last updated")

    class Meta(SynaraEntity.Meta):
        db_table = "syn_schema_behavior"
        ordering = ["schema", "priority", "name"]
        indexes = [
            models.Index(fields=["schema", "trigger_event"], name="idx_schemabehavior_trigger"),
            models.Index(fields=["action_type", "is_active"], name="idx_schemabehavior_action"),
        ]
        verbose_name = "Schema Behavior"
        verbose_name_plural = "Schema Behaviors"

    class SynaraMeta:
        event_domain = "syn.schema.schema_behavior"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        return f"{self.schema.entity}.{self.name}: {self.trigger_event} -> {self.action_type}"


class SchemaVersion(SynaraEntity):
    """
    Tracks schema version per tenant/domain/entity for migration management.

    Standard: SCHEMA-001 §5, MKT-001 §6
    Table: syn_schema_version

    Features:
    - Per-tenant schema version tracking
    - Migration audit trail with hash verification
    - Upgrade path management
    - Rollback support

    Compliance:
    - ISO 9001:2015 §7.5.3: Document Control
    - 21 CFR Part 11 §11.10: Audit Trail
    - SOC 2 CC6.1: Logical Access Controls
    """

    id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False, help_text="Unique schema version identifier"
    )
    correlation_id = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        editable=False,
        db_index=True,
        help_text="Correlation ID for causal trace graph (CTG-001 §5)",
    )
    tenant_id = models.UUIDField(db_index=True, help_text="Tenant identifier (SEC-001 §5.2)")
    domain = models.CharField(
        max_length=100, db_index=True, help_text="Domain namespace (e.g., 'marketplace', 'qms', 'risk')"
    )
    entity = models.CharField(
        max_length=100, db_index=True, help_text="Entity type (e.g., 'MarketplaceListing', 'NCR')"
    )
    version = models.CharField(max_length=20, help_text="Semantic version (MAJOR.MINOR.PATCH)")
    applied_at = models.DateTimeField(auto_now_add=True, db_index=True, help_text="When this version was applied")
    migration_hash = models.CharField(
        max_length=64, help_text="SHA-256 hash of migration definition for integrity verification"
    )
    migration_definition = models.JSONField(default=dict, blank=True, help_text="Migration operations and metadata")
    applied_by = models.CharField(max_length=255, help_text="User or system that applied this migration")
    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("applying", "Applying"),
            ("applied", "Applied"),
            ("failed", "Failed"),
            ("rolled_back", "Rolled Back"),
        ],
        default="applied",
        db_index=True,
        help_text="Migration status",
    )
    rollback_available = models.BooleanField(default=True, help_text="Whether this migration can be rolled back")
    rollback_definition = models.JSONField(default=dict, blank=True, help_text="Rollback operations if available")
    previous_version = models.CharField(
        max_length=20, blank=True, default="", help_text="Previous version before this migration"
    )
    migration_duration_ms = models.IntegerField(
        null=True, blank=True, help_text="Migration execution time in milliseconds"
    )
    error_message = models.TextField(blank=True, default="", help_text="Error message if migration failed")

    class Meta(SynaraEntity.Meta):
        db_table = "syn_schema_version"
        ordering = ["-applied_at"]
        indexes = [
            models.Index(fields=["tenant_id", "domain", "entity"], name="idx_schemaversion_tde"),
            models.Index(fields=["domain", "entity", "version"], name="idx_schemaversion_dev"),
            models.Index(fields=["status", "applied_at"], name="idx_schemaversion_status"),
        ]
        unique_together = [
            ["tenant_id", "domain", "entity", "version"],
        ]
        verbose_name = "Schema Version"
        verbose_name_plural = "Schema Versions"

    class SynaraMeta:
        event_domain = "syn.schema.schema_version"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        return f"[T:{self.tenant_id}] {self.domain}.{self.entity} v{self.version} ({self.status})"

    def save(self, *args, **kwargs):
        """Compute migration hash before saving."""
        if not self.migration_hash:
            self.migration_hash = self._compute_migration_hash()
        super().save(*args, **kwargs)

    def _compute_migration_hash(self) -> str:
        """Compute SHA-256 hash of migration definition."""
        migration_json = json.dumps(self.migration_definition, sort_keys=True)
        return hashlib.sha256(migration_json.encode("utf-8")).hexdigest()

    def verify_hash(self) -> bool:
        """Verify migration hash integrity."""
        return self.migration_hash == self._compute_migration_hash()

    @classmethod
    def get_current_version(cls, tenant_id: str, domain: str, entity: str) -> Optional["SchemaVersion"]:
        """
        Get the current active schema version for a tenant/domain/entity.

        Args:
            tenant_id: Tenant identifier
            domain: Domain namespace
            entity: Entity type

        Returns:
            Latest successfully applied SchemaVersion or None
        """
        return (
            cls.objects.filter(
                tenant_id=tenant_id,
                domain=domain,
                entity=entity,
                status="applied",
            )
            .order_by("-applied_at")
            .first()
        )

    @classmethod
    def get_migration_history(cls, tenant_id: str, domain: str, entity: str) -> models.QuerySet:
        """
        Get complete migration history for a schema.

        Args:
            tenant_id: Tenant identifier
            domain: Domain namespace
            entity: Entity type

        Returns:
            QuerySet of SchemaVersion ordered by applied_at
        """
        return cls.objects.filter(
            tenant_id=tenant_id,
            domain=domain,
            entity=entity,
        ).order_by("applied_at")


# EventPrimitiveCompatibilitySnapshot was here but has been removed for kjerne.
# It depended on cognition.schema_matcher which is not being ported.

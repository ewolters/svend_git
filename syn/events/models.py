# events/models.py
"""
⟡  S Y N A R A
   Event Schema Registry & Custom Event Definitions

Standard:     EVT-001/002, MKT-001
Compliance:   ISO 27001 A.12.1.1, SOC 2 CC7.2
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

import jsonschema
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models
from django.utils import timezone

from syn.core.base_models import SynaraEntity

# -----------------------------
# Event Schema Registry
# -----------------------------


class EventSchemaRegistry(SynaraEntity):
    """
    Event schema registry for validating event payloads (EVT-001 §6.2).

    Maintains versioned JSON schemas for all Synara events, ensuring:
    - Event naming follows EVT-001 §4.1 (domain.entity.action)
    - Payload validation against JSON Schema specifications
    - Semantic versioning for schema evolution
    - Multi-tenant isolation per SEC-001 §5.2

    Architecture:
    - event_name: Canonical event name (EVT-001 §4.1)
    - version: Semantic version (MAJOR.MINOR.PATCH)
    - schema: JSON Schema for payload validation
    - tenant_id: Tenant isolation boundary (SEC-001 §5.2)
    - active: Enable/disable schema enforcement

    Status: ACTIVE - Production canonical pattern
    References: EVT-001 §4.1, §6.2 | SEC-001 §5.2

    Example:
        EventSchemaRegistry(
            event_name="qms.capa.created",
            version="1.0.0",
            schema={
                "type": "object",
                "properties": {
                    "capa_id": {"type": "string"},
                    "severity": {"type": "string", "enum": ["MINOR", "MAJOR", "CRITICAL"]},
                },
                "required": ["capa_id", "severity"],
            },
            tenant_id=uuid.uuid4(),
            active=True,
        )
    """

    # Event naming pattern per EVT-001 §4.1
    EVENT_NAME_PATTERN = re.compile(r"^[a-z_]+\.[a-z_]+\.[a-z_]+$")

    # Core fields
    event_name = models.CharField(
        max_length=255,
        unique=True,
        db_index=True,
        help_text="Canonical event name following EVT-001 §4.1 (domain.entity.action)",
    )

    version = models.CharField(
        max_length=10,
        help_text="Semantic version (MAJOR.MINOR.PATCH)",
    )

    schema = models.JSONField(
        help_text="JSON Schema definition for event payload validation",
    )

    # Multi-tenancy (SEC-001 §5.2)
    tenant_id = models.UUIDField(
        db_index=True,
        help_text="Tenant ID for multi-tenant isolation (SEC-001 §5.2)",
    )

    # State
    is_active = models.BooleanField(
        default=True,
        db_index=True,
        help_text="Enable/disable schema validation enforcement",
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta(SynaraEntity.Meta):
        db_table = "events_schema_registry"
        verbose_name = "Event Schema Registry"
        verbose_name_plural = "Event Schema Registries"
        indexes = [
            models.Index(fields=["event_name", "is_active"], name="events_schema_name_active"),
            models.Index(fields=["tenant_id", "is_active"], name="events_schema_tenant_active"),
            models.Index(fields=["version"], name="events_schema_version"),
        ]
        ordering = ["event_name", "-version"]

    class SynaraMeta:
        event_domain = "syn.events.event_schema_registry"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        status = "✓" if self.is_active else "✗"
        return f"{status} {self.event_name} v{self.version}"

    def clean(self) -> None:
        """
        Validate event_name follows EVT-001 §4.1 naming pattern.

        Raises:
            ValidationError: If event_name doesn't match pattern
        """
        super().clean()

        if not self.EVENT_NAME_PATTERN.match(self.event_name):
            raise ValidationError(
                {
                    "event_name": (
                        f"Event name '{self.event_name}' must match pattern "
                        rf"'^[a-z_]+\.[a-z_]+\.[a-z_]+$' (domain.entity.action) "
                        f"per EVT-001 §4.1"
                    )
                }
            )

    def save(self, *args, **kwargs) -> None:
        """
        Override save to run validation before persisting.
        """
        self.full_clean()
        super().save(*args, **kwargs)

    def validate_payload(self, payload: dict) -> tuple[bool, Optional[str]]:
        """
        Validate event payload against JSON Schema.

        Uses jsonschema.validate() to enforce schema compliance.
        Returns validation result and error message if invalid.

        Args:
            payload: Event payload dict to validate

        Returns:
            Tuple of (is_valid, error_message):
            - (True, None) if payload is valid
            - (False, error_msg) if payload is invalid

        References:
            - EVT-001 §6.2: Schema validation requirements

        Example:
            >>> schema = EventSchemaRegistry.objects.get(event_name="qms.capa.created")
            >>> valid, error = schema.validate_payload({"capa_id": "123", "severity": "MAJOR"})
            >>> assert valid is True
            >>> assert error is None
            >>>
            >>> valid, error = schema.validate_payload({"capa_id": "123"})  # Missing required field
            >>> assert valid is False
            >>> assert "required" in error.lower()
        """
        try:
            jsonschema.validate(instance=payload, schema=self.schema)
            return (True, None)
        except jsonschema.ValidationError as exc:
            error_msg = f"Schema validation failed: {exc.message}"
            return (False, error_msg)
        except jsonschema.SchemaError as exc:
            # Schema itself is invalid (should not happen if properly validated on creation)
            error_msg = f"Invalid schema definition: {exc.message}"
            return (False, error_msg)

    @classmethod
    def get_latest_version(cls, event_name: str) -> Optional[EventSchemaRegistry]:
        """
        Get the latest version of an event schema by name.

        Queries by event_name, orders by version descending (lexicographic),
        and returns the first active schema.

        Args:
            event_name: Canonical event name (EVT-001 §4.1)

        Returns:
            Latest EventSchemaRegistry instance, or None if not found

        References:
            - EVT-001 §6.2: Event registry lookup

        Example:
            >>> latest = EventSchemaRegistry.get_latest_version("qms.capa.created")
            >>> print(latest.version)
            "1.2.5"
        """
        return cls.objects.filter(event_name=event_name, is_active=True).order_by("-version").first()


# =============================================================================
# CUSTOM EVENT DEFINITIONS (EVT-001 §12.4, MKT-001)
# =============================================================================


class CustomEventDefinition(SynaraEntity):
    """
    User-created custom event type definitions.

    Allows developers to create and publish custom event types that can be
    used by tenants in governance rules, reflexes, and integrations.

    Standard:     EVT-001 §12.4 (Custom Event Registration)
    Marketplace:  MKT-001 §2 (item type "event")
    Compliance:   SEC-001 §5.2 (tenant isolation)

    Workflow:
    1. Developer creates event in 'draft' status
    2. Developer submits for validation ('pending_validation')
    3. System validates against EVT-001/002 standards
    4. If valid → 'validated', if invalid → 'validation_failed'
    5. Developer can publish to marketplace ('published')
    6. Other tenants can install published events
    """

    # Event naming pattern per EVT-001 §4.1
    EVENT_NAME_PATTERN = re.compile(r"^[a-z_]+\.[a-z_]+\.[a-z_]+$")

    # Reserved system domains that cannot be used by custom events
    RESERVED_DOMAINS = [
        "forms",
        "task",
        "security",
        "governance",
        "cognition",
        "api",
        "telemetry",
        "risk",
        "org",
        "federation",
        "schema",
        "log",
        "validation",
        "engine",
        "cli",
        "ctg",
        "erm",
        "error",
        "err",
        "system",
        "synara",
        "kernel",
        "cortex",
        "limbic",
        "reflex",
    ]

    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("pending_validation", "Pending Validation"),
        ("validation_failed", "Validation Failed"),
        ("validated", "Validated"),
        ("published", "Published"),
        ("deprecated", "Deprecated"),
    ]

    # Version lifecycle status per EVT-001 §6
    LIFECYCLE_CHOICES = [
        ("draft", "Draft"),
        ("experimental", "Experimental"),
        ("stable", "Stable"),
        ("deprecated", "Deprecated"),
        ("removed", "Removed"),
    ]

    CATEGORY_CHOICES = [
        ("general", "General"),
        ("workflow", "Workflow"),
        ("integration", "Integration"),
        ("notification", "Notification"),
        ("analytics", "Analytics"),
        ("compliance", "Compliance"),
        ("custom", "Custom"),
    ]

    # Primary key
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Event identity (EVT-001 §4.1)
    event_name = models.CharField(
        max_length=255, db_index=True, help_text="Event name following domain.entity.action pattern (EVT-001 §4.1)"
    )
    version = models.CharField(max_length=20, default="1.0.0", help_text="Semantic version (MAJOR.MINOR.PATCH)")

    # Human-readable metadata
    display_name = models.CharField(
        max_length=100, blank=True, default="", help_text="Human-readable display name (e.g., 'Document Created')"
    )
    description = models.TextField(help_text="Human-readable description of the event")
    help_text = models.TextField(
        blank=True, default="", help_text="Additional help text explaining when/how the event is triggered"
    )
    documentation_url = models.URLField(
        blank=True, default="", help_text="URL to detailed documentation for this event"
    )

    # Event metadata
    category = models.CharField(
        max_length=50, choices=CATEGORY_CHOICES, default="custom", help_text="Event category for organization"
    )
    audit = models.BooleanField(default=True, help_text="Whether this event should be included in audit logs")
    priority = models.IntegerField(default=3, help_text="Event priority (1=critical, 5=low)")
    payload_schema = models.JSONField(default=dict, help_text="JSON Schema for event payload validation")

    # Workflow integration (WFL-001)
    workflow_enabled = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Whether this event can be used as a workflow trigger (editable via marketplace)",
    )

    # Ownership (was FK to marketplace.DeveloperAccount — stripped for kjerne)
    developer_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Developer who created this event (UUID reference)",
    )

    # Tenant scope (SEC-001 §5.2)
    tenant_id_owner = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Tenant that owns this event (null for global events)",
    )
    is_global = models.BooleanField(default=False, help_text="If True, event is available to all tenants (marketplace)")

    # Review/validation status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="draft", db_index=True)
    validation_errors = models.JSONField(
        default=list, blank=True, help_text="List of validation errors if validation failed"
    )
    validation_warnings = models.JSONField(default=list, blank=True, help_text="List of validation warnings")
    validated_at = models.DateTimeField(null=True, blank=True, help_text="When the event was validated")
    validated_by = models.CharField(
        max_length=100, blank=True, default="", help_text="Who/what validated the event (system or reviewer)"
    )

    # Marketplace listing (was FK — stripped for kjerne)
    marketplace_listing_id = models.UUIDField(
        null=True,
        blank=True,
        help_text="Associated marketplace listing UUID if published",
    )

    # Compliance tags
    compliance_refs = models.JSONField(
        default=list, blank=True, help_text="Compliance references (e.g., ['ISO 27001 A.12.1', 'SOC 2 CC7.2'])"
    )

    # Usage metrics
    install_count = models.IntegerField(default=0, help_text="Number of tenant installations")
    emit_count = models.BigIntegerField(default=0, help_text="Total number of times this event has been emitted")

    # ==========================================================================
    # VERSION LIFECYCLE (EVT-001 §6)
    # ==========================================================================
    lifecycle_status = models.CharField(
        max_length=20,
        choices=LIFECYCLE_CHOICES,
        default="draft",
        db_index=True,
        help_text="Version lifecycle status per EVT-001 §6",
    )

    # Version checksum for integrity
    version_checksum = models.CharField(
        max_length=64, blank=True, default="", help_text="SHA-256 checksum of payload_schema for integrity"
    )

    # Deprecation metadata per SCHEMA-001 §6.3
    sunset_date = models.DateTimeField(
        null=True, blank=True, help_text="When this version reaches end of life (min 180 days from deprecation)"
    )

    successor_version = models.CharField(
        max_length=20, blank=True, default="", help_text="Recommended successor version to migrate to"
    )

    migration_path = models.TextField(
        blank=True, default="", help_text="Migration instructions from this version to successor"
    )

    # Audit fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_at = models.DateTimeField(null=True, blank=True, help_text="When the event was published to marketplace")
    deprecated_at = models.DateTimeField(null=True, blank=True, help_text="When the event was deprecated")

    class Meta(SynaraEntity.Meta):
        db_table = "events_custom_definition"
        verbose_name = "Custom Event Definition"
        verbose_name_plural = "Custom Event Definitions"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["event_name", "version"], name="events_custom_name_version"),
            models.Index(fields=["developer_id", "status"], name="events_custom_dev_status"),
            models.Index(fields=["tenant_id_owner", "status"], name="events_custom_tenant_status"),
            models.Index(fields=["status", "is_global"], name="events_custom_status_global"),
            models.Index(fields=["category"], name="events_custom_category"),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["event_name", "version", "tenant_id_owner"], name="events_custom_unique_name_version_tenant"
            ),
        ]

    class SynaraMeta:
        event_domain = "syn.events.custom_event_definition"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        status_icon = {
            "draft": "📝",
            "pending_validation": "⏳",
            "validation_failed": "❌",
            "validated": "✅",
            "published": "🌐",
            "deprecated": "📦",
        }.get(self.status, "❓")
        return f"{status_icon} {self.event_name} v{self.version}"

    def clean(self) -> None:
        """Validate event definition before save."""
        super().clean()

        # Validate event name pattern (EVT-001 §4.1)
        if not self.EVENT_NAME_PATTERN.match(self.event_name):
            raise ValidationError(
                {
                    "event_name": (
                        f"Event name '{self.event_name}' must match pattern "
                        f"'domain.entity.action' (e.g., 'custom.order.placed') "
                        f"per EVT-001 §4.1"
                    )
                }
            )

        # Check reserved domains
        domain = self.event_name.split(".")[0]
        if domain in self.RESERVED_DOMAINS:
            raise ValidationError(
                {"event_name": (f"Domain '{domain}' is reserved for system events. Please use a custom domain name.")}
            )

        # Validate priority range
        if self.priority < 1 or self.priority > 5:
            raise ValidationError({"priority": "Priority must be between 1 (critical) and 5 (low)"})

    def save(self, *args, **kwargs) -> None:
        """Override save to run validation and register event."""
        is_new = self._state.adding
        old_status = None
        if not is_new:
            try:
                old_instance = CustomEventDefinition.objects.get(pk=self.pk)
                old_status = old_instance.status
            except CustomEventDefinition.DoesNotExist:
                pass

        self.full_clean()
        super().save(*args, **kwargs)

        # Auto-register to schema registry when validated or published
        if self.status in ["validated", "published"] and old_status != self.status:
            self._register_to_schema_registry()

    def _register_to_schema_registry(self) -> None:
        """
        Register this custom event to the EventSchemaRegistry.

        Called automatically when event transitions to validated/published status.
        This ensures the event is discoverable and its schema is enforced.

        Standard: EVT-001 §12.4 (Event Registration)
        """
        try:
            # Check if already registered
            existing = EventSchemaRegistry.objects.filter(
                event_name=self.event_name,
                version=self.version,
            ).first()

            if existing:
                # Update existing registration
                existing.schema = self.payload_schema
                existing.is_active = True
                existing.save(update_fields=["schema", "is_active", "updated_at"])
            else:
                # Create new registration
                # Note: EventSchemaRegistry only has: event_name, version, schema, tenant_id, is_active
                EventSchemaRegistry.objects.create(
                    event_name=self.event_name,
                    version=self.version,
                    schema=self.payload_schema,
                    is_active=True,
                    tenant_id=self.tenant_id,
                )
        except Exception as e:
            # Log but don't fail the save
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to register event {self.event_name} to schema registry: {e}")

    def submit_for_validation(self) -> None:
        """Submit event definition for validation."""
        if self.status != "draft":
            raise ValidationError("Only draft events can be submitted for validation")
        self.status = "pending_validation"
        self.save(update_fields=["status", "updated_at"])

    def mark_validated(self, validated_by: str = "system") -> None:
        """Mark event as validated."""
        if self.status != "pending_validation":
            raise ValidationError("Only pending events can be marked as validated")
        self.status = "validated"
        self.validated_at = timezone.now()
        self.validated_by = validated_by
        self.validation_errors = []
        self.save(update_fields=["status", "validated_at", "validated_by", "validation_errors", "updated_at"])

    def mark_validation_failed(self, errors: list, warnings: list = None) -> None:
        """Mark event as validation failed."""
        if self.status != "pending_validation":
            raise ValidationError("Only pending events can have validation results")
        self.status = "validation_failed"
        self.validation_errors = errors
        self.validation_warnings = warnings or []
        self.save(update_fields=["status", "validation_errors", "validation_warnings", "updated_at"])

    def publish(self) -> None:
        """Publish event to marketplace."""
        if self.status != "validated":
            raise ValidationError("Only validated events can be published")
        self.status = "published"
        self.is_global = True
        self.published_at = timezone.now()
        self.save(update_fields=["status", "is_global", "published_at", "updated_at"])

    def deprecate(self, reason: str = "", successor: Optional[str] = None, sunset_days: int = 180) -> None:
        """
        Deprecate the event with full lifecycle management.

        Per SCHEMA-001 §6.3: Minimum 180 days until sunset.

        Args:
            reason: Deprecation reason
            successor: Successor version to migrate to
            sunset_days: Days until end of life (minimum 180)
        """
        from datetime import timedelta

        if self.status not in ["validated", "published"]:
            raise ValidationError("Only validated or published events can be deprecated")

        # Ensure minimum sunset period
        if sunset_days < 180:
            sunset_days = 180

        self.status = "deprecated"
        self.lifecycle_status = "deprecated"
        self.deprecated_at = timezone.now()
        self.sunset_date = timezone.now() + timedelta(days=sunset_days)

        if successor:
            self.successor_version = successor
        if reason:
            self.migration_path = reason

        self.save(
            update_fields=[
                "status",
                "lifecycle_status",
                "deprecated_at",
                "sunset_date",
                "successor_version",
                "migration_path",
                "updated_at",
            ]
        )

    # ==========================================================================
    # VERSION LIFECYCLE METHODS (EVT-001 §6)
    # ==========================================================================

    def get_semantic_version(self):
        """Parse version string into SemanticVersion object."""
        from syn.core.versioning import SemanticVersion

        return SemanticVersion.parse(self.version)

    def bump_version(self, change_type: str = "patch") -> str:
        """
        Compute next version based on change type.

        Args:
            change_type: "major", "minor", or "patch"

        Returns:
            New version string
        """
        from syn.core.versioning import VersionChangeType

        current = self.get_semantic_version()
        type_map = {
            "major": VersionChangeType.MAJOR,
            "minor": VersionChangeType.MINOR,
            "patch": VersionChangeType.PATCH,
        }
        change = type_map.get(change_type, VersionChangeType.PATCH)

        if change == VersionChangeType.MAJOR:
            return str(current.bump_major())
        elif change == VersionChangeType.MINOR:
            return str(current.bump_minor())
        else:
            return str(current.bump_patch())

    def create_new_version(
        self, schema_updates: dict, change_type: str = "minor", user=None
    ) -> "CustomEventDefinition":
        """
        Create new version of this event definition.

        Args:
            schema_updates: Updates to payload_schema
            change_type: Version bump type ("major", "minor", "patch")
            user: User creating the new version

        Returns:
            New CustomEventDefinition instance
        """
        new_version = self.bump_version(change_type)
        new_schema = {**self.payload_schema, **schema_updates}

        return CustomEventDefinition.objects.create(
            event_name=self.event_name,
            version=new_version,
            display_name=self.display_name,
            description=self.description,
            help_text=self.help_text,
            documentation_url=self.documentation_url,
            category=self.category,
            audit=self.audit,
            priority=self.priority,
            payload_schema=new_schema,
            workflow_enabled=self.workflow_enabled,
            developer_id=self.developer_id,
            tenant_id_owner=self.tenant_id_owner,
            is_global=False,  # New versions start as non-global
            status="draft",
            lifecycle_status="draft",
            compliance_refs=self.compliance_refs,
        )

    def compute_checksum(self) -> str:
        """Compute and store checksum for payload_schema."""
        from syn.core.versioning import compute_version_checksum

        self.version_checksum = compute_version_checksum(self.payload_schema)
        return self.version_checksum

    def verify_checksum(self) -> bool:
        """Verify stored checksum matches current payload_schema."""
        from syn.core.versioning import verify_version_checksum

        if not self.version_checksum:
            return True
        return verify_version_checksum(self.payload_schema, self.version_checksum)

    def is_deprecated(self) -> bool:
        """Check if event is deprecated or beyond."""
        return self.lifecycle_status in ("deprecated", "removed")

    def is_usable(self) -> bool:
        """Check if event can be used for new bindings."""
        return self.lifecycle_status in ("draft", "experimental", "stable")

    def get_version_history(self) -> list:
        """Get all versions of this event."""
        return list(
            CustomEventDefinition.objects.filter(
                event_name=self.event_name,
                tenant_id_owner=self.tenant_id_owner,
            )
            .order_by("-version")
            .values("id", "version", "lifecycle_status", "created_at", "status")
        )

    def to_catalog_format(self) -> dict:
        """
        Convert to the event catalog format used by documentation.

        Returns dict matching the structure in syn/*/events.py catalogs.
        """
        return {
            "display_name": self.display_name or self.event_name,
            "description": self.description,
            "help_text": self.help_text,
            "documentation_url": self.documentation_url,
            "category": self.category,
            "audit": self.audit,
            "priority": self.priority,
            "payload_schema": self.payload_schema,
            "compliance_refs": self.compliance_refs,
            "workflow_enabled": self.workflow_enabled,
            "custom": True,
            "developer": str(self.developer_id),
            "version": self.version,
            # Version lifecycle info (EVT-001 §6)
            "lifecycle_status": self.lifecycle_status,
            "is_deprecated": self.is_deprecated(),
            "sunset_date": self.sunset_date.isoformat() if self.sunset_date else None,
            "successor_version": self.successor_version or None,
        }


class CustomEventInstallation(SynaraEntity):
    """
    Tracks which tenants have installed which custom events.

    When a tenant installs a marketplace event, this creates a record
    allowing them to use that event in their governance rules and reflexes.

    Standard:     EVT-001 §12.5 (Event Installation)
    Compliance:   SEC-001 §5.2 (tenant isolation)
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Event reference
    event_definition = models.ForeignKey(CustomEventDefinition, on_delete=models.CASCADE, related_name="installations")

    # Tenant who installed it
    tenant_id_owner = models.UUIDField(
        db_index=True,
        help_text="Tenant that installed this event",
    )

    # Version tracking
    installed_version = models.CharField(max_length=20, help_text="Version that was installed")
    auto_update = models.BooleanField(default=True, help_text="Automatically update to latest version")

    # Status
    is_active = models.BooleanField(default=True, db_index=True)

    # Installation metadata
    installed_at = models.DateTimeField(auto_now_add=True)
    installed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True, related_name="installed_events"
    )

    # Usage tracking
    emit_count = models.BigIntegerField(default=0, help_text="Number of times this event was emitted by this tenant")
    last_emitted_at = models.DateTimeField(null=True, blank=True, help_text="Last time this event was emitted")

    class Meta(SynaraEntity.Meta):
        db_table = "events_custom_installation"
        verbose_name = "Custom Event Installation"
        verbose_name_plural = "Custom Event Installations"
        ordering = ["-installed_at"]
        indexes = [
            models.Index(fields=["tenant_id_owner", "is_active"], name="events_install_tenant_active"),
            models.Index(fields=["event_definition", "is_active"], name="events_install_event_active"),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["event_definition", "tenant_id_owner"], name="events_install_unique_event_tenant"
            ),
        ]

    class SynaraMeta:
        event_domain = "syn.events.custom_event_installation"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        status = "✓" if self.is_active else "✗"
        return f"{status} {self.event_definition.event_name} @ {self.tenant_id_owner}"

    def uninstall(self) -> None:
        """Uninstall (deactivate) the event from this tenant."""
        self.is_active = False
        self.save(update_fields=["is_active"])

    def reactivate(self) -> None:
        """Reactivate a previously uninstalled event."""
        self.is_active = True
        self.save(update_fields=["is_active"])

    def update_version(self, new_version: str) -> None:
        """Update to a new version."""
        self.installed_version = new_version
        self.save(update_fields=["installed_version"])


# =============================================================================
# SYSTEM EVENT TYPE REGISTRY (EVT-001 §4)
# =============================================================================


class SystemEventType(SynaraEntity):
    """
    Registry of built-in system event types (EVT-001 §4).

    These are core Synara events (document.created, form.submitted, etc.)
    that are always available to all tenants. Unlike CustomEventDefinition,
    these cannot be created by users - only seeded by the system.

    IMPORTANT: Overrides tenant_id from SynaraEntity to be nullable, allowing
    global system events with tenant_id=NULL. System events are not tenant-scoped -
    they are globally defined event types available to all tenants.

    Standard:     EVT-001 §4 (System Event Types)
    Compliance:   WFL-001 §7 (Workflow Triggers)

    The workflow_enabled flag determines which events can be used as
    workflow triggers. This can only be edited via the marketplace
    event editor or system administration.
    """

    CATEGORY_CHOICES = [
        ("document", "Document"),
        ("record", "Record"),
        ("form", "Form"),
        ("task", "Task"),
        ("workflow", "Workflow"),
        ("approval", "Approval"),
        ("schedule", "Schedule"),
        ("integration", "Integration"),
        ("system", "System"),
    ]

    # Primary key
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Event identity (EVT-001 §4.1)
    event_name = models.CharField(
        max_length=255, unique=True, db_index=True, help_text="Canonical event name (domain.entity.action pattern)"
    )

    # Human-readable metadata
    display_name = models.CharField(max_length=100, help_text="Human-readable display name (e.g., 'Document Created')")
    description = models.TextField(blank=True, default="", help_text="Description of what triggers this event")
    help_text = models.TextField(
        blank=True, default="", help_text="Additional help text for users configuring this event"
    )
    documentation_url = models.URLField(blank=True, default="", help_text="URL to detailed documentation")

    # Categorization
    category = models.CharField(
        max_length=50,
        choices=CATEGORY_CHOICES,
        default="system",
        db_index=True,
        help_text="Event category for organization",
    )

    # Workflow integration (WFL-001 §7)
    workflow_enabled = models.BooleanField(
        default=False, db_index=True, help_text="Whether this event can be used as a workflow trigger"
    )

    # State
    is_active = models.BooleanField(default=True, db_index=True, help_text="Enable/disable this event type")

    # Module that owns this event
    source_module = models.CharField(
        max_length=50,
        blank=True,
        default="",
        help_text="Module that emits this event (e.g., 'document', 'form', 'workflow')",
    )

    # Payload schema for documentation
    payload_schema = models.JSONField(default=dict, blank=True, help_text="JSON Schema describing the event payload")

    # Override tenant_id to be nullable for global system events (EVT-001 §4)
    tenant_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="Tenant ID (null for global system events per EVT-001 §4)"
    )

    class Meta(SynaraEntity.Meta):
        db_table = "events_system_type"
        verbose_name = "System Event Type"
        verbose_name_plural = "System Event Types"
        ordering = ["category", "event_name"]
        indexes = [
            models.Index(fields=["workflow_enabled", "is_active"], name="events_sys_wfl_active"),
            models.Index(fields=["category", "is_active"], name="events_sys_category_active"),
            models.Index(fields=["source_module"], name="events_sys_module"),
        ]

    class SynaraMeta:
        event_domain = "syn.events.system_event_type"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        wfl = "⚡" if self.workflow_enabled else ""
        status = "✓" if self.is_active else "✗"
        return f"{status}{wfl} {self.display_name} ({self.event_name})"

    def to_trigger_format(self) -> dict:
        """
        Convert to format used by workflow trigger dropdowns.

        Returns:
            Dict with 'value' and 'label' keys for select options.
        """
        return {
            "value": self.event_name,
            "label": self.display_name,
            "description": self.description,
            "help_text": self.help_text,
            "category": self.category,
        }

    @classmethod
    def get_workflow_triggers(cls) -> list:
        """
        Get all events that can be used as workflow triggers.

        Returns:
            List of dicts with 'value' and 'label' keys.
        """
        return list(
            cls.objects.filter(
                workflow_enabled=True,
                is_active=True,
            )
            .values("event_name", "display_name", "description", "help_text", "category")
            .order_by("category", "display_name")
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_tenant_custom_events(tenant_id) -> dict:
    """
    Get all custom events available to a tenant.

    Returns events that are:
    1. Owned by the tenant (draft or published)
    2. Published globally (marketplace)
    3. Installed by the tenant

    Args:
        tenant_id: Tenant ID (UUID or int)

    Returns:
        Dict of event_name -> event catalog format
    """
    events = {}

    # 1. Tenant's own events
    own_events = CustomEventDefinition.objects.filter(
        tenant_id=tenant_id,
        status__in=["validated", "published"],
    )
    for event in own_events:
        events[event.event_name] = event.to_catalog_format()

    # 2. Installed marketplace events
    installations = CustomEventInstallation.objects.filter(
        tenant_id=tenant_id,
        is_active=True,
    ).select_related("event_definition")

    for install in installations:
        event = install.event_definition
        if event.status == "published" and event.event_name not in events:
            events[event.event_name] = event.to_catalog_format()

    return events


# =============================================================================
# EVENT FAMILIES (EVT-001 §13 - Governed Event Bundling)
# =============================================================================


class EventFamilyCategory(models.TextChoices):
    """Event family categories for semantic grouping per EVT-001 §13.1."""

    LIFECYCLE = "lifecycle", "Lifecycle Events"
    STATE_CHANGE = "state_change", "State Change Events"
    THRESHOLD = "threshold", "Threshold Events"
    CREATION = "creation", "Creation Events"
    WORKFLOW = "workflow", "Workflow Events"
    AUDIT = "audit", "Audit Events"
    CUSTOM = "custom", "Custom"


class EventFamily(SynaraEntity):
    """
    Governed bundle of semantically related events that emit together.

    Event families allow objects to emit multiple events from a single state
    change while enforcing governance rules to prevent cascade storms,
    infinite loops, and fan-out explosions.

    Per EVT-001 §13 (Governed Event Families):
    - Events in a family share a common correlation_id
    - Families enforce max_events_per_emission limits
    - Families can require schema compatibility across members
    - Families participate in CTG lineage as atomic units

    Philosophy (from system-reminder):
    > "An event is justified only when a materially relevant fact has changed."
    > Objects should emit ONE event per meaningful semantic change.
    > Families group semantically related changes (e.g., status + substatus).

    Example Families:
    - "task.lifecycle" → [task.created, task.status.changed, task.completed]
    - "risk.assessment" → [risk.rpn.changed, risk.level.changed]
    - "form.submission" → [form.submitted, form.validated, form.signature.required]

    Standard:     EVT-001 §13 (Governed Event Families)
    Architecture: SBL-001 §3.3 (Event Cortex Integration)
    Compliance:   ISO 27001 A.8.15, SOC 2 CC6.1
    """

    # Identity
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, help_text="Human-readable family name (e.g., 'Task Lifecycle')")
    slug = models.SlugField(max_length=100, db_index=True, help_text="URL-safe identifier (e.g., 'task-lifecycle')")
    description = models.TextField(
        blank=True, default="", help_text="Description of what semantic change this family represents"
    )

    # Family identity (EVT-001 §13.1)
    family_key = models.CharField(
        max_length=255,
        unique=True,
        db_index=True,
        help_text="Canonical family key (e.g., 'task.lifecycle', 'risk.assessment')",
    )
    version = models.CharField(max_length=20, default="1.0.0", help_text="Semantic version (MAJOR.MINOR.PATCH)")

    # Categorization
    category = models.CharField(
        max_length=30,
        choices=EventFamilyCategory.choices,
        default=EventFamilyCategory.CUSTOM,
        db_index=True,
        help_text="Family category for organization",
    )
    domain = models.CharField(
        max_length=50, db_index=True, help_text="Domain this family belongs to (e.g., 'task', 'risk', 'form')"
    )

    # Governance rules (EVT-001 §13.2)
    max_events_per_emission = models.PositiveSmallIntegerField(
        default=5, help_text="Maximum events that can be emitted per family invocation (cascade limit)"
    )
    require_schema_compatibility = models.BooleanField(
        default=True, help_text="All member events must have compatible payload schemas"
    )
    allow_partial_emission = models.BooleanField(
        default=False, help_text="If True, emit available events even if some fail validation"
    )
    emission_order = models.CharField(
        max_length=20,
        choices=[
            ("sequential", "Sequential (ordered)"),
            ("parallel", "Parallel (unordered)"),
        ],
        default="sequential",
        help_text="How member events are emitted",
    )

    # Rate limiting (SBL-001 §6.4)
    rate_limit_per_minute = models.PositiveIntegerField(
        default=100, help_text="Max family emissions per tenant per minute"
    )
    rate_limit_per_hour = models.PositiveIntegerField(
        default=1000, help_text="Max family emissions per tenant per hour"
    )

    # Multi-tenancy (SEC-001 §5.2)
    tenant_id_owner = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Tenant that owns this family (null for system families)",
    )
    is_system = models.BooleanField(default=False, db_index=True, help_text="True for built-in system families")

    # Ownership for custom families (was FK to marketplace.DeveloperAccount)
    developer_id_owner = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Developer who created this family (UUID reference)",
    )

    # Lifecycle
    is_active = models.BooleanField(default=True, db_index=True, help_text="Enable/disable family")

    # Workflow integration (WFL-001)
    workflow_enabled = models.BooleanField(
        default=False, db_index=True, help_text="Whether this family can be used as a workflow trigger"
    )

    # Usage metrics
    emit_count = models.BigIntegerField(default=0, help_text="Total number of times this family was emitted")
    last_emitted_at = models.DateTimeField(null=True, blank=True, help_text="Last time this family was emitted")

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta(SynaraEntity.Meta):
        db_table = "events_family"
        verbose_name = "Event Family"
        verbose_name_plural = "Event Families"
        ordering = ["domain", "name"]
        indexes = [
            models.Index(fields=["family_key", "is_active"], name="events_family_key_active"),
            models.Index(fields=["tenant_id_owner", "is_active"], name="events_family_tenant_active"),
            models.Index(fields=["domain", "category"], name="events_family_domain_cat"),
            models.Index(fields=["workflow_enabled", "is_active"], name="events_family_wfl_active"),
        ]
        constraints = [
            models.UniqueConstraint(fields=["slug", "tenant_id_owner"], name="events_family_unique_slug_tenant"),
        ]

    class SynaraMeta:
        event_domain = "syn.events.event_family"
        emit_events = ["created", "updated", "deleted", "emitted"]

    def __str__(self) -> str:
        status = "✓" if self.is_active else "✗"
        return f"{status} {self.name} ({self.family_key})"

    def clean(self) -> None:
        """Validate family before save."""
        super().clean()
        from django.core.exceptions import ValidationError

        # Validate family_key format: domain.category or domain.entity.category
        if not re.match(r"^[a-z_]+(\.[a-z_]+){1,2}$", self.family_key):
            raise ValidationError(
                {
                    "family_key": (
                        f"Family key '{self.family_key}' must match pattern "
                        f"'domain.category' or 'domain.entity.category'"
                    )
                }
            )

        # Extract domain from family_key
        self.domain = self.family_key.split(".")[0]

    def save(self, *args, **kwargs) -> None:
        """Override save to auto-generate slug."""
        if not self.slug:
            from django.utils.text import slugify

            self.slug = slugify(self.name)
        self.full_clean()
        super().save(*args, **kwargs)

    def get_members(self) -> list:
        """Get all active member events in emission order."""
        return list(
            self.members.filter(is_active=True)
            .select_related("event_definition", "system_event")
            .order_by("emission_order", "created_at")
        )

    def get_event_names(self) -> list:
        """Get list of event names in this family."""
        members = self.get_members()
        return [m.get_event_name() for m in members]

    def can_emit(self) -> tuple:
        """
        Check if family can emit (governance validation).

        Returns:
            Tuple of (can_emit: bool, reason: str)
        """
        if not self.is_active:
            return (False, "Family is inactive")

        members = self.get_members()
        if not members:
            return (False, "Family has no active members")

        if len(members) > self.max_events_per_emission:
            return (False, f"Family has {len(members)} members, max is {self.max_events_per_emission}")

        return (True, "OK")

    def to_trigger_format(self) -> dict:
        """Convert to format used by workflow/reflex trigger dropdowns."""
        return {
            "value": self.family_key,
            "label": self.name,
            "description": self.description,
            "category": self.get_category_display(),
            "type": "family",
            "member_count": self.members.filter(is_active=True).count(),
        }


class EventFamilyMember(SynaraEntity):
    """
    Junction model linking events to families with ordering.

    Defines which events belong to a family and in what order they emit.
    Supports both system events (SystemEventType) and custom events
    (CustomEventDefinition).

    Standard:     EVT-001 §13.3 (Family Member Registration)
    Compliance:   SEC-001 §5.2 (tenant isolation via family)

    Note: tenant_id is nullable because tenant isolation is enforced via
    the family FK (family.tenant_id). This allows the migration to work
    without requiring a default value for existing rows.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Override tenant_id to be nullable - isolation is via family FK
    tenant_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Tenant ID (nullable - isolation via family FK per EVT-001 §13.3)",
    )

    # Family relationship
    family = models.ForeignKey(
        EventFamily, on_delete=models.CASCADE, related_name="members", help_text="Parent family this event belongs to"
    )

    # Event reference (one of these must be set)
    system_event = models.ForeignKey(
        SystemEventType,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="family_memberships",
        help_text="System event reference (if system event)",
    )
    event_definition = models.ForeignKey(
        CustomEventDefinition,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="family_memberships",
        help_text="Custom event reference (if custom event)",
    )
    # For events not in DB (e.g., hardcoded catalogs)
    event_name_override = models.CharField(
        max_length=255, blank=True, default="", help_text="Direct event name (if event is from catalog, not DB)"
    )

    # Emission configuration
    emission_order = models.PositiveSmallIntegerField(
        default=0, db_index=True, help_text="Order in which this event emits (lower = first)"
    )
    is_required = models.BooleanField(default=True, help_text="If True, family emission fails if this event fails")
    is_active = models.BooleanField(default=True, db_index=True, help_text="Enable/disable this member")

    # Payload transformation (optional)
    payload_template = models.JSONField(
        default=dict, blank=True, help_text="Template for transforming family payload to this event's schema"
    )

    # Condition for conditional emission
    condition = models.JSONField(
        default=dict, blank=True, help_text="JSONLogic condition - only emit if condition is true"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta(SynaraEntity.Meta):
        db_table = "events_family_member"
        verbose_name = "Event Family Member"
        verbose_name_plural = "Event Family Members"
        ordering = ["family", "emission_order"]
        indexes = [
            models.Index(fields=["family", "is_active"], name="events_member_family_active"),
            models.Index(fields=["system_event"], name="events_member_sys_event"),
            models.Index(fields=["event_definition"], name="events_member_custom_event"),
        ]
        constraints = [
            # Each event can only be in a family once
            models.UniqueConstraint(
                fields=["family", "system_event"],
                condition=models.Q(system_event__isnull=False),
                name="events_member_unique_sys",
            ),
            models.UniqueConstraint(
                fields=["family", "event_definition"],
                condition=models.Q(event_definition__isnull=False),
                name="events_member_unique_custom",
            ),
            models.UniqueConstraint(
                fields=["family", "event_name_override"],
                condition=models.Q(event_name_override__gt=""),
                name="events_member_unique_override",
            ),
        ]

    class SynaraMeta:
        event_domain = "syn.events.event_family_member"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self) -> str:
        event_name = self.get_event_name()
        status = "✓" if self.is_active else "✗"
        return f"{status} {self.family.name} → {event_name} (#{self.emission_order})"

    def clean(self) -> None:
        """Validate that exactly one event reference is set."""
        super().clean()
        from django.core.exceptions import ValidationError

        refs = [
            self.system_event is not None,
            self.event_definition is not None,
            bool(self.event_name_override),
        ]
        if sum(refs) != 1:
            raise ValidationError("Exactly one of system_event, event_definition, or event_name_override must be set")

    def save(self, *args, **kwargs) -> None:
        self.full_clean()
        super().save(*args, **kwargs)

    def get_event_name(self) -> str:
        """Get the canonical event name for this member."""
        if self.system_event:
            return self.system_event.event_name
        elif self.event_definition:
            return self.event_definition.event_name
        else:
            return self.event_name_override

    def get_payload_schema(self) -> dict:
        """Get the payload schema for this member's event."""
        if self.system_event:
            return self.system_event.payload_schema or {}
        elif self.event_definition:
            return self.event_definition.payload_schema or {}
        return {}

    def should_emit(self, context: dict) -> bool:
        """
        Check if this member should emit based on its condition.

        Args:
            context: Event context/payload for condition evaluation

        Returns:
            True if should emit, False otherwise
        """
        if not self.is_active:
            return False

        if not self.condition:
            return True

        try:
            from json_logic import jsonLogic

            return bool(jsonLogic(self.condition, context))
        except Exception as e:
            # LOG-001 §9.4 INV-LOG-001: Log before returning fallback
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Condition evaluation failed for EventFamilyMember {self.id}, "
                f"returning is_required={self.is_required}: {e}",
                extra={"member_id": str(self.id), "family_id": str(self.family_id)},
            )
            return self.is_required


def get_workflow_trigger_events() -> list:
    """
    Get all events that can be used as workflow triggers.

    Combines:
    1. System events with workflow_enabled=True
    2. Custom events with workflow_enabled=True and status='published'

    Returns:
        List of dicts with 'value' and 'label' keys for select dropdowns.
    """
    triggers = []

    # 1. System events (uses all_objects to include system tenant events)
    system_events = SystemEventType.all_objects.filter(
        workflow_enabled=True,
        is_active=True,
        is_deleted=False,
    ).order_by("category", "display_name")

    for event in system_events:
        triggers.append(
            {
                "value": event.event_name,
                "label": event.display_name,
                "category": event.get_category_display(),
                "description": event.description,
            }
        )

    # 2. Published custom events with workflow enabled
    # Note: CustomEventDefinition table may not have SynaraEntity fields yet
    # This will work once migrations are run
    try:
        custom_events = CustomEventDefinition.objects.filter(
            workflow_enabled=True,
            status="published",
            is_global=True,
        ).order_by("category", "display_name")

        for event in custom_events:
            triggers.append(
                {
                    "value": event.event_name,
                    "label": event.display_name or event.event_name,
                    "category": f"Custom: {event.get_category_display()}",
                    "description": event.description,
                }
            )
    except Exception as e:
        # LOG-001 §9.4 INV-LOG-001: Log before returning fallback
        # Custom events table may not have workflow_enabled column yet
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(f"Could not load custom workflow triggers (may be expected during migration): {e}")

    return triggers

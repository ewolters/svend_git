"""
Synara SDK Mixins (SDK-001 §6)
==============================

Reusable mixins for extending Django models with Synara OS capabilities.

Use these mixins when you need specific functionality without
inheriting from the full SynaraEntity base class.

Standard:     SDK-001 §6
Compliance:   ISO 27001 A.8.1, SOC 2 CC6.1
Version:      1.0.0

Usage:
    from syn.core.mixins import (
        CorrelationMixin,
        TenantMixin,
        AuditMixin,
        SoftDeleteMixin,
        EventEmitterMixin,
        LifecycleMixin,
    )

    class LegacyModel(CorrelationMixin, TenantMixin, models.Model):
        '''Add correlation and tenant support to existing model'''
        name = models.CharField(max_length=100)
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from django.db import models
from django.utils import timezone

if TYPE_CHECKING:
    from uuid import UUID


logger = logging.getLogger(__name__)


# =============================================================================
# CORRELATION MIXIN (CTG-001 §5)
# =============================================================================


class CorrelationMixin(models.Model):
    """
    Adds correlation ID tracking for CTG integration.

    Provides:
    - Unique correlation_id for entity tracing
    - Parent correlation for causal chains
    - CTG edge creation methods

    Standard: SDK-001 §6.1, CTG-001 §5
    """

    correlation_id = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        db_index=True,
        editable=False,
        help_text="Unique correlation ID for CTG tracing (CTG-001 §5)",
    )

    parent_correlation_id = models.UUIDField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Parent entity correlation ID for causal chain",
    )

    class Meta:
        abstract = True

    def link_to_parent(self, parent_correlation_id: UUID):
        """Link this entity to a parent in the causal chain."""
        self.parent_correlation_id = parent_correlation_id
        self.save(update_fields=["parent_correlation_id"])

    def create_ctg_edge(
        self,
        child_correlation_id: UUID,
        causal_type: str = "causal",
        event_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Create a CTG edge to a child entity."""
        try:
            from syn.ctg.models import CausalEdge

            tenant_id = getattr(self, "tenant_id", None)

            CausalEdge.objects.create(
                tenant_id=tenant_id,
                parent_correlation_id=self.correlation_id,
                child_correlation_id=child_correlation_id,
                causal_type=causal_type,
                event_name=event_name or "",
                metadata=metadata or {},
            )
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to create CTG edge: {e}")


# =============================================================================
# TENANT MIXIN (SEC-001 §5.2)
# =============================================================================


class TenantMixin(models.Model):
    """
    Adds multi-tenant isolation support.

    Provides:
    - tenant_id field for data isolation
    - Scoped query helpers

    Standard: SDK-001 §6.2, SEC-001 §5.2
    """

    tenant_id = models.UUIDField(db_index=True, help_text="Tenant ID for multi-tenant isolation (SEC-001 §5.2)")

    class Meta:
        abstract = True

    @classmethod
    def for_tenant(cls, tenant_id: UUID):
        """Get queryset filtered by tenant."""
        return cls.objects.filter(tenant_id=tenant_id)


# =============================================================================
# AUDIT MIXIN (AUD-001)
# =============================================================================


class AuditMixin(models.Model):
    """
    Adds audit timestamp fields.

    Provides:
    - created_at, updated_at timestamps
    - created_by, updated_by tracking

    Standard: SDK-001 §6.3, AUD-001
    """

    created_at = models.DateTimeField(auto_now_add=True, db_index=True, help_text="Record creation timestamp")

    updated_at = models.DateTimeField(auto_now=True, help_text="Last modification timestamp")

    created_by = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="User ID who created this record",
    )

    updated_by = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="User ID who last modified this record",
    )

    class Meta:
        abstract = True

    def set_created_by(self, user_id: str):
        """Set the creator (typically called once on creation)."""
        self.created_by = user_id
        self.updated_by = user_id

    def set_updated_by(self, user_id: str):
        """Set the modifier (called on each update)."""
        self.updated_by = user_id


# =============================================================================
# SOFT DELETE MIXIN (DAT-001 §9)
# =============================================================================


class SoftDeleteManager(models.Manager):
    """Manager that excludes soft-deleted records by default."""

    def get_queryset(self):
        return super().get_queryset().filter(is_deleted=False)

    def with_deleted(self):
        """Include soft-deleted records."""
        return super().get_queryset()

    def deleted_only(self):
        """Return only soft-deleted records."""
        return super().get_queryset().filter(is_deleted=True)


class SoftDeleteMixin(models.Model):
    """
    Adds soft delete capability.

    Provides:
    - is_deleted flag and deleted_at timestamp
    - Filtered default queryset
    - restore() method

    Standard: SDK-001 §6.4, DAT-001 §9
    """

    is_deleted = models.BooleanField(default=False, db_index=True, help_text="Soft delete flag")

    deleted_at = models.DateTimeField(null=True, blank=True, help_text="Deletion timestamp")

    deleted_by = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="User ID who deleted this record",
    )

    objects = SoftDeleteManager()
    all_objects = models.Manager()

    class Meta:
        abstract = True

    def soft_delete(self, deleted_by: str = ""):
        """Perform soft delete."""
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.deleted_by = deleted_by
        self.save(update_fields=["is_deleted", "deleted_at", "deleted_by"])

    def restore(self, restored_by: str = ""):
        """Restore soft-deleted record."""
        if not self.is_deleted:
            return

        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = ""

        # Update updated_by if the mixin is present
        if hasattr(self, "updated_by"):
            self.updated_by = restored_by
            self.save(update_fields=["is_deleted", "deleted_at", "deleted_by", "updated_by"])
        else:
            self.save(update_fields=["is_deleted", "deleted_at", "deleted_by"])


# =============================================================================
# EVENT EMITTER MIXIN (EVT-001)
# =============================================================================


class EventEmitterMixin(models.Model):
    """
    Adds event emission capabilities.

    Provides:
    - emit_event() method for custom events
    - Automatic event emission on save (if configured)

    Standard: SDK-001 §6.5, EVT-001

    Configure via class attribute:
        class MyModel(EventEmitterMixin, models.Model):
            event_domain = 'myapp.mymodel'
            emit_on_save = True
    """

    # Class-level configuration (override in subclass)
    event_domain: str = ""
    emit_on_save: bool = False

    class Meta:
        abstract = True

    def emit_event(
        self,
        action: str,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ):
        """
        Emit a domain event.

        Args:
            action: Event action (e.g., 'created', 'approved')
            payload: Additional event payload
            correlation_id: Override correlation ID
        """
        if not self.event_domain:
            logger.warning(f"{self.__class__.__name__} has no event_domain configured")
            return

        event_name = f"{self.event_domain}.{action}"

        # Build payload
        event_payload = {
            "entity_id": str(self.pk),
            "entity_type": self.__class__.__name__,
            "action": action,
            "timestamp": timezone.now().isoformat(),
        }

        # Add correlation_id if available
        if hasattr(self, "correlation_id"):
            event_payload["correlation_id"] = str(self.correlation_id)

        # Add tenant_id if available
        if hasattr(self, "tenant_id"):
            event_payload["tenant_id"] = str(self.tenant_id)

        if payload:
            event_payload.update(payload)

        # Emit through kernel
        try:
            from syn.kernel.event_primitives import emit_event

            emit_event(
                event_name=event_name,
                payload=event_payload,
                correlation_id=correlation_id or event_payload.get("correlation_id"),
            )
        except ImportError:
            logger.info(f"Event (no kernel): {event_name}", extra=event_payload)
        except Exception as e:
            logger.warning(f"Failed to emit event {event_name}: {e}")

    def save(self, *args, **kwargs):
        """Save with optional event emission."""
        is_new = self._state.adding

        super().save(*args, **kwargs)

        if self.emit_on_save and self.event_domain:
            action = "created" if is_new else "updated"
            self.emit_event(action)


# =============================================================================
# LIFECYCLE MIXIN (GOV-001)
# =============================================================================


class LifecycleMixin(models.Model):
    """
    Adds lifecycle state machine support.

    Provides:
    - Status field with configurable states
    - Transition validation
    - Terminal state detection

    Standard: SDK-001 §6.6, GOV-001

    Configure via class attributes:
        class MyModel(LifecycleMixin, models.Model):
            lifecycle_states = ['draft', 'submitted', 'approved', 'rejected']
            terminal_states = ['approved', 'rejected']
            initial_state = 'draft'
            transitions = {
                'draft': ['submitted'],
                'submitted': ['approved', 'rejected'],
            }
    """

    # Class-level configuration (override in subclass)
    lifecycle_states: list[str] = []
    terminal_states: list[str] = []
    initial_state: str = ""
    transitions: dict[str, list[str]] = {}

    # Instance field - override with ForeignKey to registry in subclass
    # This provides a fallback CharField if no registry is used
    # status = models.CharField(max_length=50, db_index=True)

    class Meta:
        abstract = True

    def get_status_value(self) -> str:
        """
        Get current status value.

        Override if using a registry FK instead of CharField.
        """
        if hasattr(self, "status"):
            status = self.status
            # Handle FK to registry
            if hasattr(status, "code"):
                return status.code
            return str(status) if status else ""
        return ""

    def is_terminal(self) -> bool:
        """Check if current status is terminal."""
        return self.get_status_value() in self.terminal_states

    def can_transition_to(self, new_status: str) -> bool:
        """
        Check if transition to new_status is valid.

        Args:
            new_status: Target status code

        Returns:
            True if transition is allowed
        """
        current = self.get_status_value()

        # Already in terminal state
        if self.is_terminal():
            return False

        # Check transitions map
        if self.transitions:
            allowed = self.transitions.get(current, [])
            return new_status in allowed

        # No transitions defined = allow any
        return new_status in self.lifecycle_states

    def transition_to(
        self,
        new_status: str,
        actor: str = "",
        reason: str = "",
        force: bool = False,
    ) -> bool:
        """
        Attempt to transition to a new status.

        Args:
            new_status: Target status code
            actor: User performing transition
            reason: Reason for transition
            force: Skip validation (admin override)

        Returns:
            True if transition succeeded

        Raises:
            ValueError: If transition is invalid and force=False
        """
        if not force and not self.can_transition_to(new_status):
            current = self.get_status_value()
            raise ValueError(
                f"Cannot transition from '{current}' to '{new_status}'. Allowed: {self.transitions.get(current, [])}"
            )

        old_status = self.get_status_value()

        # Set new status (implementation depends on field type)
        self._set_status(new_status)

        # Emit transition event if mixin is present
        if hasattr(self, "emit_event"):
            self.emit_event(
                "status_changed",
                {
                    "from_status": old_status,
                    "to_status": new_status,
                    "actor": actor,
                    "reason": reason,
                },
            )

        return True

    def _set_status(self, new_status: str):
        """
        Set the status field.

        Override if using a registry FK.
        """
        if hasattr(self, "status"):
            # Handle FK to registry - subclass should override
            self.status = new_status
            self.save(update_fields=["status"])


# =============================================================================
# METADATA MIXIN (MOD-001 §10)
# =============================================================================


class MetadataMixin(models.Model):
    """
    Adds extensible metadata JSONField.

    Provides:
    - metadata JSONField for arbitrary data
    - get/set helpers with dot notation

    Standard: SDK-001 §6.7, MOD-001 §10
    """

    metadata = models.JSONField(default=dict, blank=True, help_text="Extensible metadata")

    class Meta:
        abstract = True

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value by key.

        Supports dot notation: get_metadata('nested.key')
        """
        keys = key.split(".")
        value = self.metadata

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def set_metadata(self, key: str, value: Any):
        """
        Set metadata value by key.

        Supports dot notation: set_metadata('nested.key', 'value')
        """
        keys = key.split(".")
        data = self.metadata

        # Navigate to parent
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]

        # Set value
        data[keys[-1]] = value
        self.save(update_fields=["metadata"])

    def remove_metadata(self, key: str):
        """Remove metadata key."""
        keys = key.split(".")
        data = self.metadata

        # Navigate to parent
        for k in keys[:-1]:
            if k not in data:
                return
            data = data[k]

        # Remove key
        if keys[-1] in data:
            del data[keys[-1]]
            self.save(update_fields=["metadata"])


# =============================================================================
# VERSIONING MIXIN
# =============================================================================


class VersioningMixin(models.Model):
    """
    Adds semantic versioning support.

    Provides:
    - version field with semantic versioning
    - Increment helpers

    Standard: SDK-001 §6.8
    """

    version = models.CharField(max_length=20, default="1.0.0", help_text="Semantic version (MAJOR.MINOR.PATCH)")

    version_number = models.PositiveIntegerField(default=1, help_text="Sequential version number")

    class Meta:
        abstract = True

    def parse_version(self) -> tuple:
        """Parse version string into (major, minor, patch) tuple."""
        try:
            parts = self.version.split(".")
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        except (IndexError, ValueError):
            return (1, 0, 0)

    def increment_patch(self):
        """Increment patch version (1.0.0 -> 1.0.1)."""
        major, minor, patch = self.parse_version()
        self.version = f"{major}.{minor}.{patch + 1}"
        self.version_number += 1

    def increment_minor(self):
        """Increment minor version (1.0.0 -> 1.1.0)."""
        major, minor, patch = self.parse_version()
        self.version = f"{major}.{minor + 1}.0"
        self.version_number += 1

    def increment_major(self):
        """Increment major version (1.0.0 -> 2.0.0)."""
        major, minor, patch = self.parse_version()
        self.version = f"{major + 1}.0.0"
        self.version_number += 1


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CorrelationMixin",
    "TenantMixin",
    "AuditMixin",
    "SoftDeleteMixin",
    "SoftDeleteManager",
    "EventEmitterMixin",
    "LifecycleMixin",
    "MetadataMixin",
    "VersioningMixin",
]

"""
Synara SDK Base Models (SDK-001)
================================

Abstract base classes for all Synara domain entities.

These base classes provide automatic integration with Synara OS primitives:
- Correlation tracking (CTG-001)
- Tenant isolation (SEC-001 §5.2)
- Audit timestamps (AUD-001)
- Soft delete (DAT-001 §9)
- Event emission hooks (EVT-001)
- Governance integration (GOV-001)

Standard:     SDK-001 (Synara Developer Kit)
Compliance:   ISO 27001 A.8.1, SOC 2 CC6.1, 21 CFR Part 11
Version:      1.0.0

Usage:
    from syn.core.base_models import SynaraEntity, SynaraRegistry, SynaraImmutableLog

    class Claim(SynaraEntity):
        '''Healthcare billing claim - inherits all OS behaviors'''
        claim_number = models.CharField(max_length=50)
        status = models.ForeignKey('ClaimStatusRegistry', ...)

        class SynaraMeta:
            event_domain = 'healthcare.claim'
            emit_events = ['created', 'updated', 'deleted']
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from django.db import models
from django.utils import timezone

if TYPE_CHECKING:
    from uuid import UUID


logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("syn.audit")


# =============================================================================
# SYNARA ENTITY BASE CLASS (SDK-001 §3)
# =============================================================================


class SynaraEntityManager(models.Manager):
    """
    Default manager for SynaraEntity that excludes soft-deleted records.

    Per DAT-001 §9: Active records filter by default.
    """

    def get_queryset(self):
        return super().get_queryset().filter(is_deleted=False)

    def with_deleted(self):
        """Include soft-deleted records in queryset."""
        return super().get_queryset()

    def deleted_only(self):
        """Return only soft-deleted records."""
        return super().get_queryset().filter(is_deleted=True)

    def for_tenant(self, tenant_id: UUID):
        """Filter queryset for specific tenant (SEC-001 §5.2)."""
        return self.get_queryset().filter(tenant_id=tenant_id)


class SynaraEntityAllManager(models.Manager):
    """
    Manager that includes all records (including soft-deleted).

    Use when you need to query deleted records explicitly.
    """

    pass


class SynaraEntity(models.Model):
    """
    Abstract base class for ALL domain entities in Synara ecosystem.

    Provides:
    - Correlation tracking (CTG-001 §5)
    - Tenant isolation (SEC-001 §5.2)
    - Audit timestamps (AUD-001)
    - Soft delete (DAT-001 §9)
    - Event emission hooks (EVT-001)
    - Governance integration points (GOV-001)

    Standard: SDK-001 §3, MOD-001 §5
    Compliance: ISO 27001 A.8.1, SOC 2 CC6.1, 21 CFR Part 11 §11.10

    Example:
        class Claim(SynaraEntity):
            claim_number = models.CharField(max_length=50)
            amount = models.DecimalField(max_digits=12, decimal_places=2)

            class SynaraMeta:
                event_domain = 'healthcare.claim'
                emit_events = ['created', 'updated', 'deleted']
    """

    # =========================================================================
    # PRIMARY KEY
    # =========================================================================

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, help_text="Primary key (UUID v4)")

    # =========================================================================
    # CORRELATION TRACKING (CTG-001 §5)
    # =========================================================================

    correlation_id = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        db_index=True,
        editable=False,
        help_text="Unique correlation ID for CTG tracing (CTG-001 §5)",
    )

    parent_correlation_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="Parent entity correlation ID for causal chain (CTG-001 §5)"
    )

    # =========================================================================
    # TENANT ISOLATION (SEC-001 §5.2)
    # =========================================================================

    tenant_id = models.UUIDField(
        db_index=True,
        null=True,
        blank=True,
        help_text="Tenant ID for multi-tenant isolation (SEC-001 §5.2). Null for system-wide entities.",
    )

    # =========================================================================
    # AUDIT TIMESTAMPS (AUD-001)
    # =========================================================================

    created_at = models.DateTimeField(auto_now_add=True, db_index=True, help_text="Record creation timestamp (AUD-001)")

    updated_at = models.DateTimeField(auto_now=True, help_text="Last modification timestamp (AUD-001)")

    created_by = models.CharField(max_length=255, blank=True, default="", help_text="User ID who created this record")

    updated_by = models.CharField(
        max_length=255, blank=True, default="", help_text="User ID who last modified this record"
    )

    # =========================================================================
    # SOFT DELETE (DAT-001 §9)
    # =========================================================================

    is_deleted = models.BooleanField(default=False, db_index=True, help_text="Soft delete flag (DAT-001 §9)")

    deleted_at = models.DateTimeField(null=True, blank=True, help_text="Deletion timestamp (DAT-001 §9)")

    deleted_by = models.CharField(max_length=255, blank=True, default="", help_text="User ID who deleted this record")

    # =========================================================================
    # EXTENSIBILITY (MOD-001 §10)
    # =========================================================================

    metadata = models.JSONField(default=dict, blank=True, help_text="Extensible metadata (MOD-001 §10)")

    # =========================================================================
    # MANAGERS
    # =========================================================================

    objects = SynaraEntityManager()
    all_objects = SynaraEntityAllManager()

    class Meta:
        abstract = True
        ordering = ["-created_at"]

    class SynaraMeta:
        """
        Synara-specific metadata for entity behavior configuration.

        Override in subclasses to customize behavior:

            class SynaraMeta:
                event_domain = 'healthcare.claim'
                emit_events = ['created', 'updated', 'deleted']
                lifecycle_states = ['draft', 'submitted', 'approved', 'paid']
                terminal_states = ['paid', 'denied', 'voided']
        """

        event_domain: str = ""
        emit_events: list[str] = []
        lifecycle_states: list[str] = []
        terminal_states: list[str] = []
        require_governance: bool = False

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    def save(self, *args, **kwargs):
        """
        Save with automatic event emission.

        Per SBL-001: No business logic in save().
        Event emission triggers governance and downstream processing.
        """
        is_new = self._state.adding

        super().save(*args, **kwargs)

        # Emit events if configured
        synara_meta = self._get_synara_meta()
        if synara_meta.event_domain:
            action = "created" if is_new else "updated"
            if action in synara_meta.emit_events:
                self._emit_event(action)

    def delete(self, using=None, keep_parents=False, hard_delete=False, deleted_by: str = ""):
        """
        Soft delete by default. Use hard_delete=True for permanent deletion.

        Per DAT-001 §9: Soft delete preserves audit trail.
        """
        if hard_delete:
            super().delete(using=using, keep_parents=keep_parents)
        else:
            self.is_deleted = True
            self.deleted_at = timezone.now()
            self.deleted_by = deleted_by
            self.save(update_fields=["is_deleted", "deleted_at", "deleted_by", "updated_at"])

            # Emit deleted event if configured
            synara_meta = self._get_synara_meta()
            if synara_meta.event_domain and "deleted" in synara_meta.emit_events:
                self._emit_event("deleted")

    def restore(self, restored_by: str = ""):
        """
        Restore a soft-deleted record.

        Per DAT-001 §9: Restoration clears delete markers.
        """
        if not self.is_deleted:
            return

        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by = ""
        self.updated_by = restored_by
        self.save(update_fields=["is_deleted", "deleted_at", "deleted_by", "updated_by", "updated_at"])

        # Emit restored event
        synara_meta = self._get_synara_meta()
        if synara_meta.event_domain:
            self._emit_event("restored")

    # =========================================================================
    # EVENT EMISSION (EVT-001)
    # =========================================================================

    def _get_synara_meta(self) -> type[SynaraEntity.SynaraMeta]:
        """Get SynaraMeta from class or parent."""
        return getattr(self.__class__, "SynaraMeta", SynaraEntity.SynaraMeta)

    def _emit_event(self, action: str, payload: dict[str, Any] | None = None):
        """
        Emit domain event through Synara event system.

        Event name format: {domain}.{action} (e.g., 'healthcare.claim.created')

        Args:
            action: Event action (created, updated, deleted, etc.)
            payload: Optional additional payload data
        """
        synara_meta = self._get_synara_meta()
        if not synara_meta.event_domain:
            return

        event_name = f"{synara_meta.event_domain}.{action}"
        event_payload = self._build_event_payload(action)

        if payload:
            event_payload.update(payload)

        try:
            from syn.kernel.event_primitives import emit_event

            emit_event(
                event_name=event_name,
                payload=event_payload,
                correlation_id=str(self.correlation_id),
                parent_correlation_id=str(self.parent_correlation_id) if self.parent_correlation_id else None,
            )

            logger.debug(
                f"Emitted event: {event_name}",
                extra={
                    "event_name": event_name,
                    "correlation_id": str(self.correlation_id),
                    "tenant_id": str(self.tenant_id),
                },
            )
        except ImportError:
            # Kernel not available, log locally
            logger.info(
                f"Event (no kernel): {event_name}",
                extra={
                    "event_name": event_name,
                    "payload": event_payload,
                    "correlation_id": str(self.correlation_id),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to emit event {event_name}: {e}")

    def _build_event_payload(self, action: str) -> dict[str, Any]:
        """
        Build standard event payload.

        Override in subclasses for custom payload structure.
        """
        return {
            "entity_id": str(self.id),
            "entity_type": self.__class__.__name__,
            "correlation_id": str(self.correlation_id),
            "parent_correlation_id": str(self.parent_correlation_id) if self.parent_correlation_id else None,
            # Handle global entities (tenant_id=None) properly - don't convert to "None" string
            "tenant_id": str(self.tenant_id) if self.tenant_id else None,
            "action": action,
            "timestamp": timezone.now().isoformat(),
        }

    def emit_event(self, action: str, payload: dict[str, Any] | None = None):
        """
        Public method to emit custom events.

        Use for domain-specific events beyond CRUD:
            claim.emit_event('submitted', {'submitted_to': 'medicare'})
        """
        self._emit_event(action, payload)

    # =========================================================================
    # CTG INTEGRATION (CTG-001)
    # =========================================================================

    def create_ctg_edge(
        self,
        child_correlation_id: UUID,
        causal_type: str = "causal",
        event_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Create a CTG edge to a child entity.

        Args:
            child_correlation_id: Correlation ID of child entity
            causal_type: Edge type (temporal, conditional, causal, evidential)
            event_name: Optional event name for the edge
            metadata: Optional edge metadata

        Standard: CTG-001 §4-7
        """
        try:
            from syn.ctg.models import CausalEdge

            CausalEdge.objects.create(
                tenant_id=self.tenant_id,
                parent_correlation_id=self.correlation_id,
                child_correlation_id=child_correlation_id,
                causal_type=causal_type,
                event_name=event_name or f"{self._get_synara_meta().event_domain}.linked",
                metadata=metadata or {},
            )
        except ImportError:
            logger.warning("CTG module not available")
        except Exception as e:
            logger.warning(f"Failed to create CTG edge: {e}")

    def get_causal_parents(self) -> list[UUID]:
        """
        Get all parent correlation IDs from CTG.

        Returns list of correlation IDs that causally precede this entity.
        """
        try:
            from syn.ctg.models import CausalEdge

            edges = CausalEdge.objects.filter(
                child_correlation_id=self.correlation_id,
                tenant_id=self.tenant_id,
            ).values_list("parent_correlation_id", flat=True)
            return list(edges)
        except ImportError:
            return []

    def get_causal_children(self) -> list[UUID]:
        """
        Get all child correlation IDs from CTG.

        Returns list of correlation IDs that were caused by this entity.
        """
        try:
            from syn.ctg.models import CausalEdge

            edges = CausalEdge.objects.filter(
                parent_correlation_id=self.correlation_id,
                tenant_id=self.tenant_id,
            ).values_list("child_correlation_id", flat=True)
            return list(edges)
        except ImportError:
            return []

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def to_dict(self, include_metadata: bool = True) -> dict[str, Any]:
        """
        Serialize entity to dictionary.

        Override in subclasses for custom serialization.
        """
        data = {
            "id": str(self.id),
            "correlation_id": str(self.correlation_id),
            "parent_correlation_id": str(self.parent_correlation_id) if self.parent_correlation_id else None,
            "tenant_id": str(self.tenant_id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "is_deleted": self.is_deleted,
        }

        if include_metadata:
            data["metadata"] = self.metadata

        return data


# =============================================================================
# SYNARA REGISTRY BASE CLASS (SDK-001 §4, MOD-001 §4)
# =============================================================================


class SynaraRegistryManager(models.Manager):
    """
    Manager for registry models that filters by active status.
    """

    def get_queryset(self):
        return super().get_queryset().filter(is_active=True)

    def all_including_inactive(self):
        """Include inactive registry entries."""
        return super().get_queryset()

    def for_tenant(self, tenant_id: UUID):
        """Filter for specific tenant."""
        return self.get_queryset().filter(models.Q(tenant_id=tenant_id) | models.Q(tenant_id__isnull=True))


class SynaraRegistry(models.Model):
    """
    Abstract base class for database-backed enumeration registries.

    Provides:
    - Dynamic, auditable enumeration values
    - Tenant-scoped or global entries
    - Display ordering for UI
    - Soft delete with cascade deactivation

    Standard: SDK-001 §4, MOD-001 §4
    Compliance: ISO 9001:2015 §7.5 (Controlled Vocabulary)

    Example:
        class ClaimStatusRegistry(SynaraRegistry):
            '''Status values for healthcare claims'''

            class Meta(SynaraRegistry.Meta):
                db_table = 'healthcare_claim_status_registry'
    """

    # =========================================================================
    # PRIMARY KEY
    # =========================================================================

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )

    # =========================================================================
    # REGISTRY IDENTIFICATION
    # =========================================================================

    code = models.CharField(max_length=100, db_index=True, help_text="Unique code (e.g., 'submitted', 'approved')")

    label = models.CharField(max_length=255, help_text="Display label for UI")

    description = models.TextField(blank=True, default="", help_text="Detailed description")

    # =========================================================================
    # DISPLAY & ORDERING
    # =========================================================================

    display_order = models.PositiveIntegerField(default=0, db_index=True, help_text="UI display order (lower = first)")

    icon = models.CharField(
        max_length=50, blank=True, default="", help_text="Icon name for UI (e.g., 'check', 'clock')"
    )

    color = models.CharField(max_length=7, blank=True, default="", help_text="Hex color for UI (e.g., '#4CAF50')")

    # =========================================================================
    # TENANT ISOLATION
    # =========================================================================

    tenant_id = models.UUIDField(null=True, blank=True, db_index=True, help_text="Tenant ID (null for global entries)")

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    is_active = models.BooleanField(default=True, db_index=True, help_text="Active entries appear in dropdowns")

    is_system = models.BooleanField(default=False, help_text="System entries cannot be deleted by users")

    # =========================================================================
    # AUDIT
    # =========================================================================

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # =========================================================================
    # EXTENSIBILITY
    # =========================================================================

    metadata = models.JSONField(default=dict, blank=True, help_text="Additional configuration")

    # =========================================================================
    # MANAGERS
    # =========================================================================

    objects = SynaraRegistryManager()
    all_objects = models.Manager()

    class Meta:
        abstract = True
        ordering = ["display_order", "label"]

    def __str__(self):
        return self.label

    def save(self, *args, **kwargs):
        """Ensure code is lowercase and valid."""
        if self.code:
            self.code = self.code.lower().strip()
        super().save(*args, **kwargs)

    def deactivate(self):
        """Soft-deactivate registry entry."""
        self.is_active = False
        self.save(update_fields=["is_active", "updated_at"])

    def activate(self):
        """Re-activate registry entry."""
        self.is_active = True
        self.save(update_fields=["is_active", "updated_at"])

    @classmethod
    def get_by_code(cls, code: str, tenant_id: UUID | None = None):
        """
        Get registry entry by code.

        Looks up tenant-specific entry first, then global.
        """
        code = code.lower().strip()

        # Try tenant-specific first
        if tenant_id:
            try:
                return cls.objects.get(code=code, tenant_id=tenant_id)
            except cls.DoesNotExist:
                pass

        # Fall back to global
        try:
            return cls.objects.get(code=code, tenant_id__isnull=True)
        except cls.DoesNotExist:
            raise cls.DoesNotExist(f"Registry entry '{code}' not found")

    @classmethod
    def get_choices(cls, tenant_id: UUID | None = None) -> list[tuple]:
        """
        Get choices for Django form/serializer fields.

        Returns: List of (code, label) tuples
        """
        qs = cls.objects.all()
        if tenant_id:
            qs = qs.filter(models.Q(tenant_id=tenant_id) | models.Q(tenant_id__isnull=True))
        return list(qs.values_list("code", "label"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "id": str(self.id),
            "code": self.code,
            "label": self.label,
            "description": self.description,
            "display_order": self.display_order,
            "icon": self.icon,
            "color": self.color,
            "is_active": self.is_active,
            "is_system": self.is_system,
            "metadata": self.metadata,
        }


# =============================================================================
# SYNARA IMMUTABLE LOG BASE CLASS (SDK-001 §5, AUD-001)
# =============================================================================


class SynaraImmutableLog(models.Model):
    """
    Abstract base class for immutable audit log tables.

    Provides:
    - Write-once semantics (no updates or deletes)
    - Hash chain integrity for tamper detection
    - Correlation tracking for distributed tracing
    - Tenant isolation

    Standard: SDK-001 §5, AUD-001
    Compliance: 21 CFR Part 11 §11.10(e), SOC 2 CC7.2, ISO 27001 A.12.4.1

    Example:
        class ClaimAuditLog(SynaraImmutableLog):
            '''Immutable audit log for claim changes'''
            claim_id = models.UUIDField()
            action = models.CharField(max_length=50)

            class Meta(SynaraImmutableLog.Meta):
                db_table = 'healthcare_claim_audit_log'
    """

    # =========================================================================
    # PRIMARY KEY
    # =========================================================================

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )

    # =========================================================================
    # CORRELATION TRACKING (CTG-001)
    # =========================================================================

    correlation_id = models.UUIDField(
        default=uuid.uuid4, db_index=True, editable=False, help_text="Correlation ID for tracing"
    )

    parent_correlation_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="Parent entity correlation ID"
    )

    # =========================================================================
    # TENANT ISOLATION
    # =========================================================================

    tenant_id = models.UUIDField(
        db_index=True, null=True, blank=True, help_text="Tenant ID (SEC-001 §5.2). Null for system-wide logs."
    )

    # =========================================================================
    # AUDIT CONTEXT
    # =========================================================================

    event_name = models.CharField(max_length=255, db_index=True, help_text="Event name (EVT-001 format)")

    actor = models.CharField(max_length=255, default="system", help_text="User or system that performed the action")

    actor_ip = models.GenericIPAddressField(null=True, blank=True, help_text="IP address of actor")

    # =========================================================================
    # PAYLOAD
    # =========================================================================

    before_snapshot = models.JSONField(default=dict, blank=True, help_text="State before change")

    after_snapshot = models.JSONField(default=dict, blank=True, help_text="State after change")

    changes = models.JSONField(default=dict, blank=True, help_text="Delta of changes")

    reason = models.TextField(blank=True, default="", help_text="Reason for change")

    metadata = models.JSONField(default=dict, blank=True, help_text="Additional context")

    # =========================================================================
    # INTEGRITY (21 CFR Part 11)
    # =========================================================================

    entry_hash = models.CharField(
        max_length=64, db_index=True, editable=False, help_text="SHA-256 hash of entry content"
    )

    previous_hash = models.CharField(
        max_length=64, blank=True, default="", help_text="Hash of previous entry (chain link)"
    )

    # =========================================================================
    # TIMESTAMP
    # =========================================================================

    created_at = models.DateTimeField(auto_now_add=True, db_index=True, help_text="Immutable creation timestamp")

    class Meta:
        abstract = True
        ordering = ["-created_at"]

    def save(self, *args, **kwargs):
        """
        Save with immutability enforcement and hash computation.

        Per 21 CFR Part 11 §11.10(e): Records cannot be altered after creation.
        """
        # Prevent updates to existing records
        if self.pk and self.__class__.objects.filter(pk=self.pk).exists():
            raise ValueError(
                f"{self.__class__.__name__} records are immutable after creation. "
                "See AUD-001 and 21 CFR Part 11 §11.10(e)."
            )

        # Compute hash chain
        self._compute_hash_chain()

        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """
        Prevent deletion of immutable log entries.

        Per 21 CFR Part 11: Audit records cannot be deleted.
        """
        raise PermissionError(
            f"{self.__class__.__name__} records are immutable audit artifacts and cannot be deleted. See AUD-001."
        )

    def _compute_hash_chain(self):
        """
        Compute hash chain for tamper detection.

        Links this entry to the previous entry in the chain.
        """
        # Get previous entry for this tenant
        previous = self.__class__.objects.filter(tenant_id=self.tenant_id).order_by("-created_at").first()

        if previous:
            self.previous_hash = previous.entry_hash

        # Compute this entry's hash
        self.entry_hash = self._compute_entry_hash()

    def _compute_entry_hash(self) -> str:
        """
        Compute SHA-256 hash of entry content.

        Includes all significant fields to detect tampering.
        """
        data = {
            "correlation_id": str(self.correlation_id),
            "parent_correlation_id": str(self.parent_correlation_id) if self.parent_correlation_id else "",
            "tenant_id": str(self.tenant_id),
            "event_name": self.event_name,
            "actor": self.actor,
            "before_snapshot": json.dumps(self.before_snapshot, sort_keys=True),
            "after_snapshot": json.dumps(self.after_snapshot, sort_keys=True),
            "changes": json.dumps(self.changes, sort_keys=True),
            "reason": self.reason,
            "previous_hash": self.previous_hash,
        }
        data_json = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_json.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        """
        Verify entry hash matches computed hash.

        Returns True if entry is untampered.
        """
        expected_hash = self._compute_entry_hash()
        return self.entry_hash == expected_hash

    @classmethod
    def verify_chain(cls, tenant_id: UUID, limit: int = 1000) -> dict[str, Any]:
        """
        Verify hash chain integrity for a tenant.

        Args:
            tenant_id: Tenant to verify
            limit: Maximum entries to check

        Returns:
            Dict with is_valid, entries_checked, failures list
        """
        entries = cls.objects.filter(tenant_id=tenant_id).order_by("created_at")[:limit]

        entries_checked = 0
        failures = []
        previous_hash = ""

        for entry in entries:
            entries_checked += 1

            # Verify entry hash
            if not entry.verify_integrity():
                failures.append(
                    {
                        "entry_id": str(entry.id),
                        "error": "entry_hash_mismatch",
                    }
                )

            # Verify chain link
            if entry.previous_hash != previous_hash:
                failures.append(
                    {
                        "entry_id": str(entry.id),
                        "error": "chain_link_broken",
                    }
                )

            previous_hash = entry.entry_hash

        return {
            "is_valid": len(failures) == 0,
            "entries_checked": entries_checked,
            "failures": failures,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SynaraEntity",
    "SynaraEntityManager",
    "SynaraEntityAllManager",
    "SynaraRegistry",
    "SynaraRegistryManager",
    "SynaraImmutableLog",
]

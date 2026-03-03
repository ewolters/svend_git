"""
Core models for syn.core app.

This module re-exports models from other modules for Django's migration system.

SDK Base Classes:
- SynaraEntity: Abstract base for all domain entities
- SynaraRegistry: Abstract base for database-backed enums
- SynaraImmutableLog: Abstract base for immutable audit logs

SDK Mixins:
- CorrelationMixin: Adds CTG correlation tracking
- TenantMixin: Adds multi-tenant isolation
- AuditMixin: Adds audit timestamps
- SoftDeleteMixin: Adds soft delete capability
- EventEmitterMixin: Adds event emission
- LifecycleMixin: Adds state machine support
- MetadataMixin: Adds extensible metadata
- VersioningMixin: Adds semantic versioning
"""

from syn.core.secrets import SecretStore

# SDK Base Classes
from syn.core.base_models import (
    SynaraEntity,
    SynaraEntityManager,
    SynaraEntityAllManager,
    SynaraRegistry,
    SynaraRegistryManager,
    SynaraImmutableLog,
)

# SDK Mixins
from syn.core.mixins import (
    CorrelationMixin,
    TenantMixin,
    AuditMixin,
    SoftDeleteMixin,
    SoftDeleteManager,
    EventEmitterMixin,
    LifecycleMixin,
    MetadataMixin,
    VersioningMixin,
)

__all__ = [
    # Secrets
    "SecretStore",
    # SDK Base Classes
    "SynaraEntity",
    "SynaraEntityManager",
    "SynaraEntityAllManager",
    "SynaraRegistry",
    "SynaraRegistryManager",
    "SynaraImmutableLog",
    # SDK Mixins
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

"""
Synara Schemas Module (SCONF-001)
=================================

Schema configuration framework for forms, events, and validation.

Standard:     SCONF-001 (Schema Configuration Standard)
Compliance:   SCHEMA-001 §4, VAL-002 §4
Location:     syn/schemas/
Version:      1.0.0

Features:
- SchemaType: Category classification for schemas
- FieldType: Input field type definitions
- SchemaLifecycle: Version management and deprecation
- SchemaRegistry: Schema loading and caching

Usage:
------
    from syn.schemas import (
        SchemaType,
        FieldType,
        SchemaStatus,
        SchemaRegistry,
        SchemaLoader,
        get_registry,
        bootstrap_schemas,
    )

    # Bootstrap QMS schemas
    bootstrap_schemas()

    # Get a schema
    registry = get_registry()
    schema = registry.get("sconf://qms/capa/1.0.0")
"""

from syn.schemas.lifecycle import (
    VALID_TRANSITIONS,
    LifecycleTransition,
    SchemaLifecycle,
    SemanticVersion,
    determine_change_type,
)

# =============================================================================
# Registry (SCONF-001 §10)
# =============================================================================
from syn.schemas.registry import (
    DEFAULT_CONFIG,
    SchemaDefinition,
    SchemaField,
    SchemaLoader,
    SchemaRegistry,
    SchemaRegistryConfig,
    bootstrap_schemas,
    get_registry,
)
from syn.schemas.types import (
    BOOTSTRAP_SCHEMAS,
    FIELD_TYPE_METADATA,
    LIFECYCLE_CONFIG,
    QMS_REQUIRED_FIELDS,
    QMS_SCHEMA_PATTERNS,
    QMS_SCHEMA_URIS,
    ChangeType,
    CommonSchemaId,
    # Field types
    FieldType,
    # Governance
    GovernanceTag,
    # QMS identifiers
    QMSSchemaId,
    SchemaStatus,
    # Schema types
    SchemaType,
    ValidationRuleType,
    get_field_constraints,
    get_json_type,
)

__version__ = "1.0.0"
__standard__ = "SCONF-001"

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    "__standard__",
    # Types - Schema
    "SchemaType",
    "SchemaStatus",
    "ChangeType",
    # Types - Fields
    "FieldType",
    "FIELD_TYPE_METADATA",
    "get_json_type",
    "get_field_constraints",
    # Types - Governance
    "GovernanceTag",
    "ValidationRuleType",
    # Types - Identifiers
    "QMSSchemaId",
    "CommonSchemaId",
    "QMS_SCHEMA_URIS",
    "QMS_SCHEMA_PATTERNS",
    "QMS_REQUIRED_FIELDS",
    "BOOTSTRAP_SCHEMAS",
    "LIFECYCLE_CONFIG",
    # Lifecycle
    "SchemaLifecycle",
    "LifecycleTransition",
    "SemanticVersion",
    "determine_change_type",
    "VALID_TRANSITIONS",
    # Registry
    "SchemaRegistry",
    "SchemaRegistryConfig",
    "SchemaDefinition",
    "SchemaField",
    "SchemaLoader",
    "get_registry",
    "bootstrap_schemas",
    "DEFAULT_CONFIG",
]

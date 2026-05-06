"""
Synara Schema Module (SCHEMA-001)
=================================

SCHEMA-001 compliant universal cognitive schema framework with
versioned schemas, node definitions, behavior bindings, and
three-layer validation.

Standard:     SCHEMA-001
Compliance:   ISO 9001:2015 §7.5.3, ISO 27001 A.8.1, SOC 2 CC6.1/CC7.2
Location:     syn/schema/
Version:      1.0.0

Features:
---------
- EventSchema: Event payload schemas (legacy, VAL-002 §4.4)
- Schema: Universal entity schemas (SCHEMA-001 §5)
- SchemaNode: Field/node definitions (SCHEMA-001 §5)
- SchemaBehavior: Event/task/policy bindings (SCHEMA-001 §5)
- SchemaValidationLog: Validation audit log
- Events: Schema lifecycle and validation events

Usage:
------
    from syn.schema.models import (
        Schema,
        SchemaNode,
        SchemaBehavior,
        EventSchema,
    )
    from syn.schema.events import emit_schema_event, SCHEMA_EVENTS
"""

__version__ = "1.0.0"
__standard__ = "SCHEMA-001"

# =============================================================================
# App Config
# =============================================================================

default_app_config = "syn.schema.apps.SchemaConfig"

# =============================================================================
# Lazy imports to avoid AppRegistryNotReady errors
# =============================================================================
# Import models and other components directly from their modules:
#   from syn.schema.models import Schema, SchemaNode, SchemaBehavior
#   from syn.schema.events import emit_schema_event
#   from syn.schema.utils import validate_payload
# =============================================================================

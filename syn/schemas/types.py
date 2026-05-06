"""
Schema Types (SCONF-001 §4)
===========================

Type definitions for schema configuration framework.

Standard:     SCONF-001 §4.2-4.3 (Schema Types, Field Types)
Compliance:   SCHEMA-001 §4, SBL-001 §5
Location:     syn/schemas/types.py
Version:      1.0.0

Features:
- SchemaType: Category classification for schemas
- SchemaStatus: Lifecycle states for schemas
- FieldType: Input field type definitions
- GovernanceTag: Data classification tags
"""

from enum import Enum
from typing import Dict, List, Optional

# =============================================================================
# SCHEMA TYPES (SCONF-001 §4.2)
# =============================================================================


class SchemaType(str, Enum):
    """
    Schema type categories per SCONF-001 §4.2.

    Taxonomy: System-level constant (NOT user-customizable)
    """

    # Form schemas
    FORM_QMS = "form.qms"
    FORM_COMMON = "form.common"
    FORM_CUSTOM = "form.custom"

    # Event schemas
    EVENT_SECURITY = "event.security"
    EVENT_IDENTITY = "event.identity"
    EVENT_TENANT = "event.tenant"
    EVENT_FORMS = "event.forms"
    EVENT_BINDER = "event.binder"
    EVENT_GOVERNANCE = "event.governance"
    EVENT_ERM = "event.erm"
    EVENT_COGNITION = "event.cognition"
    EVENT_SCHEDULER = "event.scheduler"
    EVENT_QMS = "event.qms"

    # Validation schemas
    VALIDATION_FIELD = "validation.field"
    VALIDATION_CROSS_FIELD = "validation.cross_field"
    VALIDATION_AST = "validation.ast"


class SchemaStatus(str, Enum):
    """
    Schema lifecycle status per SCONF-001 §4.2.

    Taxonomy: System-level constant (NOT user-customizable)
    """

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    END_OF_LIFE = "end_of_life"
    RETIRED = "retired"


# =============================================================================
# FIELD TYPES (SCONF-001 §4.3)
# =============================================================================


class FieldType(str, Enum):
    """
    Input field types per SCONF-001 §4.3.

    Taxonomy: System-level constant (NOT user-customizable)
    """

    # Text inputs
    TEXT = "text"
    TEXTAREA = "textarea"
    RICH_TEXT = "rich_text"

    # Numeric inputs
    NUMBER = "number"
    RATING = "rating"
    SCORE = "score"

    # Contact inputs
    EMAIL = "email"
    PHONE = "phone"

    # Date inputs
    DATE = "date"
    DATETIME = "datetime"

    # Selection inputs
    SELECT = "select"
    MULTISELECT = "multiselect"
    CHECKBOX = "checkbox"
    RADIO = "radio"

    # File inputs
    FILE = "file"
    SIGNATURE = "signature"

    # Relational inputs
    RELATION = "relation"


# Field type metadata
FIELD_TYPE_METADATA: Dict[FieldType, Dict] = {
    FieldType.TEXT: {"json_type": "string"},
    FieldType.TEXTAREA: {"json_type": "string"},
    FieldType.RICH_TEXT: {"json_type": "string"},
    FieldType.NUMBER: {"json_type": "integer"},
    FieldType.RATING: {"json_type": "integer", "constraints": {"minimum": 1, "maximum": 5}},
    FieldType.SCORE: {"json_type": "integer", "constraints": {"minimum": 0, "maximum": 100}},
    FieldType.EMAIL: {"json_type": "string", "format": "email"},
    FieldType.PHONE: {"json_type": "string"},
    FieldType.DATE: {"json_type": "string", "format": "date"},
    FieldType.DATETIME: {"json_type": "string", "format": "date-time"},
    FieldType.SELECT: {"json_type": "string"},
    FieldType.MULTISELECT: {"json_type": "array"},
    FieldType.CHECKBOX: {"json_type": "boolean"},
    FieldType.RADIO: {"json_type": "string"},
    FieldType.FILE: {"json_type": "string", "format": "uri"},
    FieldType.SIGNATURE: {"json_type": "string", "format": "uri"},
    FieldType.RELATION: {"json_type": "string"},
}


def get_json_type(field_type: FieldType) -> str:
    """Get JSON Schema type for a field type."""
    return FIELD_TYPE_METADATA.get(field_type, {}).get("json_type", "string")


def get_field_constraints(field_type: FieldType) -> Optional[Dict]:
    """Get JSON Schema constraints for a field type."""
    return FIELD_TYPE_METADATA.get(field_type, {}).get("constraints")


# =============================================================================
# GOVERNANCE TAGS (SCONF-001 §4.4)
# =============================================================================


class GovernanceTag(str, Enum):
    """
    Data governance classification tags.

    Standard: SCHEMA-001 §5 (Data Classification)
    """

    PII = "PII"  # Personally Identifiable Information
    PHI = "PHI"  # Protected Health Information
    FINANCIAL = "FINANCIAL"  # Financial data
    SAFETY = "SAFETY"  # Safety-critical data
    PROPRIETARY = "PROPRIETARY"  # Proprietary/trade secret


# =============================================================================
# VALIDATION RULE TYPES (SCONF-001 §9)
# =============================================================================


class ValidationRuleType(str, Enum):
    """
    Validation rule types per SCONF-001 §9.1.
    """

    REQUIRED = "required"
    FORMAT = "format"
    RANGE = "range"
    PATTERN = "pattern"
    ENUM = "enum"
    CUSTOM = "custom"


# =============================================================================
# SCHEMA VERSIONING (SCONF-001 §5)
# =============================================================================


class ChangeType(str, Enum):
    """
    Schema change types for versioning per SCONF-001 §5.
    """

    MAJOR = "major"  # Breaking: field removal, type change, constraint tightening
    MINOR = "minor"  # Compatible: optional field added, constraint relaxed
    PATCH = "patch"  # Non-functional: description update, typo fix


# Lifecycle configuration
LIFECYCLE_CONFIG = {
    "deprecation_notice_days": 30,
    "eol_grace_days": 90,
    "max_versions_retained": 10,
}


# =============================================================================
# QMS SCHEMA IDENTIFIERS (SCONF-001 §6, §11)
# =============================================================================


class QMSSchemaId(str, Enum):
    """
    QMS starter schema identifiers per SCONF-001 §6, §11.
    """

    CAPA = "qms.capa"
    NCR = "qms.ncr"
    SCR = "qms.scr"
    DCR = "qms.dcr"
    MRB = "qms.mrb"


class CommonSchemaId(str, Enum):
    """
    Common starter schema identifiers per SCONF-001 §7.
    """

    FEEDBACK = "common.feedback"
    AUDIT = "common.audit"


# QMS Schema URIs
QMS_SCHEMA_URIS: Dict[QMSSchemaId, str] = {
    QMSSchemaId.CAPA: "sconf://qms/capa/1.0.0",
    QMSSchemaId.NCR: "sconf://qms/ncr/1.0.0",
    QMSSchemaId.SCR: "sconf://qms/scr/1.0.0",
    QMSSchemaId.DCR: "sconf://qms/dcr/1.0.0",
    QMSSchemaId.MRB: "sconf://qms/mrb/1.0.0",
}

# QMS Schema patterns
QMS_SCHEMA_PATTERNS: Dict[QMSSchemaId, str] = {
    QMSSchemaId.CAPA: "CAPA-YYYY-NNNN",
    QMSSchemaId.NCR: "NCR-YYYY-NNNN",
    QMSSchemaId.SCR: "SCR-YYYY-NNNN",
    QMSSchemaId.DCR: "DCR-YYYY-NNNN",
    QMSSchemaId.MRB: "MRB-YYYY-NNNN",
}

# Required fields per QMS schema
QMS_REQUIRED_FIELDS: Dict[QMSSchemaId, List[str]] = {
    QMSSchemaId.CAPA: [
        "issue_description",
        "issue_date",
        "severity",
        "source",
        "corrective_action",
        "target_completion_date",
        "responsible_party_id",
    ],
    QMSSchemaId.NCR: [
        "product_name",
        "nonconformance_type",
        "specification_reference",
        "actual_value",
        "expected_value",
        "disposition",
    ],
    QMSSchemaId.SCR: [
        "supplier_name",
        "issue_category",
        "issue_description",
        "response_required_date",
    ],
    QMSSchemaId.DCR: [
        "document_number",
        "document_title",
        "change_type",
        "change_description",
        "reason_for_change",
    ],
    QMSSchemaId.MRB: [
        "ncr_reference",
        "material_description",
        "quantity",
        "disposition_decision",
        "disposition_justification",
        "mrb_members",
    ],
}


# =============================================================================
# BOOTSTRAP SCHEMAS (SCONF-001 §10)
# =============================================================================


BOOTSTRAP_SCHEMAS: List[str] = [
    "sconf://qms/capa/1.0.0",
    "sconf://qms/ncr/1.0.0",
    "sconf://qms/scr/1.0.0",
    "sconf://qms/dcr/1.0.0",
    "sconf://qms/mrb/1.0.0",
    "sconf://common/feedback/1.0.0",
    "sconf://common/audit/1.0.0",
]

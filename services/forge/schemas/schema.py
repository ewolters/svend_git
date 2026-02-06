"""
Forge Schema Definitions

Defines the structure for synthetic data generation.
Users upload JSON following this schema format.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import json
import re


class FieldType(Enum):
    """Supported field types."""
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    CATEGORY = "category"
    DATE = "date"


@dataclass
class FieldSpec:
    """Specification for a single field."""
    name: str
    field_type: FieldType
    nullable: bool = False
    null_rate: float = 0.0  # 0.0 to 1.0

    # Numeric constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # Distribution for numeric types
    distribution: str = "uniform"  # uniform, normal, beta, exponential
    dist_alpha: Optional[float] = None  # For beta distribution
    dist_beta: Optional[float] = None   # For beta distribution

    # String constraints
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None  # Regex pattern

    # Category constraints
    values: Optional[list[str]] = None
    weights: Optional[list[float]] = None  # Probability weights for categories

    # Date constraints
    min_date: Optional[str] = None  # ISO format
    max_date: Optional[str] = None

    def validate(self) -> list[str]:
        """Validate the field spec. Returns list of errors."""
        errors = []

        if not self.name:
            errors.append("Field name is required")

        if self.nullable and self.null_rate <= 0:
            self.null_rate = 0.1  # Default 10% nulls if nullable

        if self.null_rate < 0 or self.null_rate > 1:
            errors.append(f"null_rate must be between 0 and 1, got {self.null_rate}")

        valid_distributions = ["uniform", "normal", "beta", "exponential"]
        if self.distribution not in valid_distributions:
            errors.append(f"distribution must be one of {valid_distributions}")

        if self.field_type == FieldType.CATEGORY:
            if not self.values or len(self.values) == 0:
                errors.append(f"Field '{self.name}': category type requires 'values' list")
            elif len(self.values) > 1000:
                errors.append(f"Field '{self.name}': max 1000 category values allowed")
            elif any(len(v) > 1000 for v in self.values):
                errors.append(f"Field '{self.name}': category values must be <= 1000 chars")
            if self.weights and len(self.weights) != len(self.values or []):
                errors.append(f"Field '{self.name}': weights length must match values length")

        if self.field_type in (FieldType.INT, FieldType.FLOAT):
            if self.min_value is not None and self.max_value is not None:
                if self.min_value > self.max_value:
                    errors.append(f"Field '{self.name}': min_value > max_value")

        if self.field_type == FieldType.STRING:
            if self.max_length is not None and self.max_length > 10000:
                errors.append(f"Field '{self.name}': max_length cannot exceed 10000")
            if self.min_length is not None and self.max_length is not None:
                if self.min_length > self.max_length:
                    errors.append(f"Field '{self.name}': min_length > max_length")
            if self.pattern:
                # Security: Limit pattern length to prevent ReDoS
                if len(self.pattern) > 100:
                    errors.append(f"Field '{self.name}': regex pattern too long (max 100 chars)")
                # Security: Block dangerous patterns (nested quantifiers)
                elif re.search(r'\([^)]*[+*][^)]*\)[+*]', self.pattern):
                    errors.append(f"Field '{self.name}': regex pattern contains nested quantifiers (ReDoS risk)")
                else:
                    try:
                        re.compile(self.pattern)
                    except re.error as e:
                        errors.append(f"Field '{self.name}': invalid regex pattern: {e}")

        return errors


@dataclass
class TabularSchema:
    """Schema for tabular data generation."""
    fields: list[FieldSpec] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate the entire schema. Returns list of errors."""
        errors = []

        if not self.fields:
            errors.append("Schema must have at least one field")
            return errors

        # Check for duplicate names
        names = [f.name for f in self.fields]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            errors.append(f"Duplicate field names: {set(duplicates)}")

        # Validate each field
        for field_spec in self.fields:
            errors.extend(field_spec.validate())

        return errors

    @classmethod
    def from_dict(cls, data: dict) -> "TabularSchema":
        """
        Create schema from dictionary (parsed JSON).

        Expected format:
        {
            "fields": {
                "field_name": {"type": "string", "nullable": false, ...},
                ...
            }
        }
        """
        fields_data = data.get("fields", {})
        fields = []

        for name, spec in fields_data.items():
            field_type_str = spec.get("type", "string")
            try:
                field_type = FieldType(field_type_str)
            except ValueError:
                raise SchemaValidationError(
                    f"Unknown field type '{field_type_str}' for field '{name}'. "
                    f"Valid types: {[t.value for t in FieldType]}"
                )

            field_spec = FieldSpec(
                name=name,
                field_type=field_type,
                nullable=spec.get("nullable", False),
                null_rate=spec.get("null_rate", 0.0),
                min_value=spec.get("min"),
                max_value=spec.get("max"),
                distribution=spec.get("distribution", "uniform"),
                dist_alpha=spec.get("dist_alpha"),
                dist_beta=spec.get("dist_beta"),
                min_length=spec.get("min_length"),
                max_length=spec.get("max_length"),
                pattern=spec.get("pattern"),
                values=spec.get("values"),
                weights=spec.get("weights"),
                min_date=spec.get("min_date"),
                max_date=spec.get("max_date"),
            )
            fields.append(field_spec)

        return cls(fields=fields)

    @classmethod
    def from_json(cls, json_str: str) -> "TabularSchema":
        """Create schema from JSON string."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise SchemaValidationError(f"Invalid JSON: {e}")
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert schema to dictionary."""
        return {
            "fields": {
                f.name: {
                    "type": f.field_type.value,
                    "nullable": f.nullable,
                    "null_rate": f.null_rate,
                    **({"min": f.min_value} if f.min_value is not None else {}),
                    **({"max": f.max_value} if f.max_value is not None else {}),
                    **({"distribution": f.distribution} if f.distribution != "uniform" else {}),
                    **({"dist_alpha": f.dist_alpha} if f.dist_alpha is not None else {}),
                    **({"dist_beta": f.dist_beta} if f.dist_beta is not None else {}),
                    **({"min_length": f.min_length} if f.min_length is not None else {}),
                    **({"max_length": f.max_length} if f.max_length is not None else {}),
                    **({"pattern": f.pattern} if f.pattern else {}),
                    **({"values": f.values} if f.values else {}),
                    **({"weights": f.weights} if f.weights else {}),
                    **({"min_date": f.min_date} if f.min_date else {}),
                    **({"max_date": f.max_date} if f.max_date else {}),
                }
                for f in self.fields
            }
        }


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""
    pass


def validate_schema_json(json_str: str) -> tuple[TabularSchema, list[str]]:
    """
    Validate a JSON schema string.

    Returns:
        (schema, errors) - schema object and list of validation errors
    """
    try:
        schema = TabularSchema.from_json(json_str)
    except SchemaValidationError as e:
        return None, [str(e)]

    errors = schema.validate()
    return schema, errors

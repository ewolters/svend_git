"""
Schema Validator

Validates data against schemas:
- JSON Schema validation
- YAML structure validation
- Config file validation
- Custom schema definitions

Pure Python - no external dependencies required.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path


@dataclass
class ValidationError:
    """A single validation error."""
    path: str           # JSON path to error (e.g., "data.items[0].name")
    message: str
    expected: str = ""
    actual: str = ""


@dataclass
class ValidationResult:
    """Results from schema validation."""
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "error_count": self.error_count,
            "errors": [
                {
                    "path": e.path,
                    "message": e.message,
                    "expected": e.expected,
                    "actual": e.actual,
                }
                for e in self.errors
            ],
            "warnings": [
                {"path": w.path, "message": w.message}
                for w in self.warnings
            ],
        }


class SchemaValidator:
    """
    Validate data against schemas.

    Supports a subset of JSON Schema (draft-07):
    - type (string, number, integer, boolean, array, object, null)
    - required
    - properties
    - items
    - enum
    - minimum, maximum
    - minLength, maxLength
    - pattern
    - minItems, maxItems

    Usage:
        validator = SchemaValidator()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = validator.validate(data, schema)
        print(result.valid)
    """

    TYPE_MAP = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    def validate(self, data: Any, schema: dict, path: str = "$") -> ValidationResult:
        """Validate data against a JSON Schema."""
        errors = []
        warnings = []

        self._validate_node(data, schema, path, errors, warnings)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _validate_node(self, data: Any, schema: dict, path: str,
                       errors: list, warnings: list) -> None:
        """Recursively validate a node."""
        # Type check
        if "type" in schema:
            expected_type = schema["type"]
            if isinstance(expected_type, list):
                # Union type
                valid = any(self._check_type(data, t) for t in expected_type)
                if not valid:
                    errors.append(ValidationError(
                        path=path,
                        message=f"Type mismatch",
                        expected=f"one of {expected_type}",
                        actual=type(data).__name__,
                    ))
                    return  # Skip further validation if type is wrong
            else:
                if not self._check_type(data, expected_type):
                    errors.append(ValidationError(
                        path=path,
                        message=f"Type mismatch",
                        expected=expected_type,
                        actual=type(data).__name__,
                    ))
                    return

        # Enum check
        if "enum" in schema:
            if data not in schema["enum"]:
                errors.append(ValidationError(
                    path=path,
                    message="Value not in allowed enum",
                    expected=str(schema["enum"]),
                    actual=str(data),
                ))

        # String validations
        if isinstance(data, str):
            if "minLength" in schema and len(data) < schema["minLength"]:
                errors.append(ValidationError(
                    path=path,
                    message=f"String too short",
                    expected=f"minLength {schema['minLength']}",
                    actual=f"length {len(data)}",
                ))
            if "maxLength" in schema and len(data) > schema["maxLength"]:
                errors.append(ValidationError(
                    path=path,
                    message=f"String too long",
                    expected=f"maxLength {schema['maxLength']}",
                    actual=f"length {len(data)}",
                ))
            if "pattern" in schema:
                if not re.match(schema["pattern"], data):
                    errors.append(ValidationError(
                        path=path,
                        message="String does not match pattern",
                        expected=schema["pattern"],
                        actual=data[:50],
                    ))

        # Number validations
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            if "minimum" in schema and data < schema["minimum"]:
                errors.append(ValidationError(
                    path=path,
                    message="Value below minimum",
                    expected=f"minimum {schema['minimum']}",
                    actual=str(data),
                ))
            if "maximum" in schema and data > schema["maximum"]:
                errors.append(ValidationError(
                    path=path,
                    message="Value above maximum",
                    expected=f"maximum {schema['maximum']}",
                    actual=str(data),
                ))
            if "exclusiveMinimum" in schema and data <= schema["exclusiveMinimum"]:
                errors.append(ValidationError(
                    path=path,
                    message="Value at or below exclusive minimum",
                    expected=f"exclusiveMinimum {schema['exclusiveMinimum']}",
                    actual=str(data),
                ))
            if "exclusiveMaximum" in schema and data >= schema["exclusiveMaximum"]:
                errors.append(ValidationError(
                    path=path,
                    message="Value at or above exclusive maximum",
                    expected=f"exclusiveMaximum {schema['exclusiveMaximum']}",
                    actual=str(data),
                ))

        # Array validations
        if isinstance(data, list):
            if "minItems" in schema and len(data) < schema["minItems"]:
                errors.append(ValidationError(
                    path=path,
                    message="Array too short",
                    expected=f"minItems {schema['minItems']}",
                    actual=f"length {len(data)}",
                ))
            if "maxItems" in schema and len(data) > schema["maxItems"]:
                errors.append(ValidationError(
                    path=path,
                    message="Array too long",
                    expected=f"maxItems {schema['maxItems']}",
                    actual=f"length {len(data)}",
                ))
            if "items" in schema:
                for i, item in enumerate(data):
                    self._validate_node(item, schema["items"], f"{path}[{i}]", errors, warnings)

        # Object validations
        if isinstance(data, dict):
            # Required fields
            if "required" in schema:
                for req_field in schema["required"]:
                    if req_field not in data:
                        errors.append(ValidationError(
                            path=path,
                            message=f"Missing required field: {req_field}",
                            expected=req_field,
                            actual="(missing)",
                        ))

            # Properties
            if "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    if prop_name in data:
                        self._validate_node(
                            data[prop_name],
                            prop_schema,
                            f"{path}.{prop_name}",
                            errors,
                            warnings,
                        )

            # Additional properties
            if "additionalProperties" in schema:
                allowed_props = set(schema.get("properties", {}).keys())
                for prop_name in data.keys():
                    if prop_name not in allowed_props:
                        if schema["additionalProperties"] is False:
                            errors.append(ValidationError(
                                path=f"{path}.{prop_name}",
                                message="Additional property not allowed",
                                expected="(none)",
                                actual=prop_name,
                            ))
                        elif isinstance(schema["additionalProperties"], dict):
                            self._validate_node(
                                data[prop_name],
                                schema["additionalProperties"],
                                f"{path}.{prop_name}",
                                errors,
                                warnings,
                            )

    def _check_type(self, data: Any, expected_type: str) -> bool:
        """Check if data matches expected type."""
        if expected_type == "null":
            return data is None
        if expected_type == "integer":
            return isinstance(data, int) and not isinstance(data, bool)
        if expected_type == "number":
            return isinstance(data, (int, float)) and not isinstance(data, bool)

        python_type = self.TYPE_MAP.get(expected_type)
        if python_type:
            return isinstance(data, python_type)
        return True


def validate_json(data: Any, schema: dict) -> dict:
    """Convenience function for quick validation."""
    validator = SchemaValidator()
    result = validator.validate(data, schema)
    return result.to_dict()


def validate_json_file(data_path: Path, schema_path: Path) -> ValidationResult:
    """Validate a JSON file against a schema file."""
    data = json.loads(data_path.read_text())
    schema = json.loads(schema_path.read_text())

    validator = SchemaValidator()
    return validator.validate(data, schema)


# CLI
def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate JSON/YAML against schema")
    parser.add_argument("data", type=Path, help="Data file (JSON)")
    parser.add_argument("schema", type=Path, help="Schema file (JSON Schema)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    result = validate_json_file(args.data, args.schema)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        if result.valid:
            print(f"✓ Valid: {args.data.name}")
        else:
            print(f"✗ Invalid: {args.data.name}")
            print(f"  {result.error_count} error(s) found:")
            for error in result.errors:
                print(f"  - {error.path}: {error.message}")
                if error.expected:
                    print(f"      Expected: {error.expected}")
                    print(f"      Actual: {error.actual}")

    sys.exit(0 if result.valid else 1)


if __name__ == "__main__":
    main()

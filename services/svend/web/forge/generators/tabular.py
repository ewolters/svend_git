"""Tabular data generator."""

import csv
import io
import json
import random
from typing import Any

from .fields import (
    StringGenerator,
    IntGenerator,
    FloatGenerator,
    BoolGenerator,
    DateGenerator,
    DateTimeGenerator,
    EmailGenerator,
    UUIDGenerator,
    CategoryGenerator,
    PhoneGenerator,
    URLGenerator,
)


def create_field_generator(field_type: str, constraints: dict, field_name: str = ""):
    """Create appropriate generator for field type."""
    constraints = constraints or {}

    if field_type == "string":
        return StringGenerator(
            min_length=constraints.get("min_length", 5),
            max_length=constraints.get("max_length", 50),
            pattern=constraints.get("pattern"),
            field_name=field_name,
        )
    elif field_type == "int":
        return IntGenerator(
            min_val=int(constraints.get("min", 0)),
            max_val=int(constraints.get("max", 1000)),
        )
    elif field_type == "float":
        return FloatGenerator(
            min_val=float(constraints.get("min", 0)),
            max_val=float(constraints.get("max", 1000)),
            precision=constraints.get("precision", 2),
        )
    elif field_type == "bool":
        return BoolGenerator(
            true_probability=constraints.get("true_probability", 0.5),
        )
    elif field_type == "category":
        values = constraints.get("values", ["A", "B", "C"])
        probs = constraints.get("probabilities")
        return CategoryGenerator(values, probs)
    elif field_type == "date":
        return DateGenerator()
    elif field_type == "datetime":
        return DateTimeGenerator()
    elif field_type == "email":
        return EmailGenerator()
    elif field_type == "uuid":
        return UUIDGenerator()
    elif field_type == "phone":
        return PhoneGenerator()
    elif field_type == "url":
        return URLGenerator()
    else:
        # Default to string
        return StringGenerator(field_name=field_name)


class TabularGenerator:
    """Generate tabular (structured) data based on a schema."""

    def __init__(self, schema: dict, domain: str = "general"):
        self.schema = schema
        self.domain = domain
        self.generators = {}
        self.nullable_fields = set()
        self._build_generators()

    def _build_generators(self):
        """Build field generators from schema."""
        for field_name, field_def in self.schema.items():
            if isinstance(field_def, dict):
                field_type = field_def.get("type", "string")
                constraints = field_def.get("constraints", {})
                nullable = field_def.get("nullable", False)
            else:
                field_type = "string"
                constraints = {}
                nullable = False

            if nullable:
                self.nullable_fields.add(field_name)

            self.generators[field_name] = create_field_generator(
                field_type, constraints, field_name
            )

    def generate_record(self) -> dict:
        """Generate a single record."""
        record = {}
        for field_name, generator in self.generators.items():
            if field_name in self.nullable_fields and random.random() < 0.1:
                record[field_name] = None
            else:
                record[field_name] = generator.generate()
        return record

    def generate(self, count: int) -> list[dict]:
        """Generate multiple records."""
        return [self.generate_record() for _ in range(count)]

    def to_json(self, records: list) -> str:
        return json.dumps(records, indent=2, default=str)

    def to_jsonl(self, records: list) -> str:
        return "\n".join(json.dumps(r, default=str) for r in records)

    def to_csv(self, records: list) -> str:
        if not records:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
        return output.getvalue()

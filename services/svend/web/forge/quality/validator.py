"""Quality validators for generated data."""

import statistics
from typing import Any


def validate_records(records: list[dict], schema: dict, quality_level: str = "standard") -> dict:
    """
    Validate generated records against schema and quality standards.

    Returns quality report with score and check results.
    """
    checks = []

    # 1. Schema compliance
    schema_check = check_schema_compliance(records, schema)
    checks.append(schema_check)

    # 2. Uniqueness check
    uniqueness_check = check_uniqueness(records)
    checks.append(uniqueness_check)

    # 3. Null rate check
    null_check = check_null_rate(records)
    checks.append(null_check)

    # 4. Distribution check (for premium)
    if quality_level == "premium":
        dist_check = check_distributions(records, schema)
        checks.append(dist_check)

    # Calculate overall score
    passed_checks = sum(1 for c in checks if c["passed"])
    overall_score = passed_checks / len(checks) if checks else 1.0

    return {
        "overall_score": overall_score,
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "record_count": len(records),
    }


def check_schema_compliance(records: list[dict], schema: dict) -> dict:
    """Check all records have required fields with correct types."""
    if not records or not schema:
        return {"name": "schema", "passed": True, "message": "No schema to validate", "score": 1.0}

    errors = 0
    for i, record in enumerate(records[:100]):  # Sample first 100
        for field_name, field_def in schema.items():
            if field_name not in record:
                if not (isinstance(field_def, dict) and field_def.get("nullable")):
                    errors += 1
                    continue

            value = record.get(field_name)
            if value is None:
                continue

            expected_type = field_def.get("type") if isinstance(field_def, dict) else "string"
            if not check_type(value, expected_type):
                errors += 1

    passed = errors == 0
    score = 1.0 - (errors / (len(records[:100]) * len(schema))) if schema else 1.0

    return {
        "name": "schema",
        "passed": passed,
        "message": f"{errors} schema violations found" if errors else "Schema valid",
        "score": max(0, score),
    }


def check_type(value: Any, expected_type: str) -> bool:
    """Check if value matches expected type."""
    type_checks = {
        "string": lambda v: isinstance(v, str),
        "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "bool": lambda v: isinstance(v, bool),
        "category": lambda v: isinstance(v, str),
        "date": lambda v: isinstance(v, str),  # ISO format string
        "datetime": lambda v: isinstance(v, str),
        "email": lambda v: isinstance(v, str) and "@" in v,
        "uuid": lambda v: isinstance(v, str),
        "phone": lambda v: isinstance(v, str),
        "url": lambda v: isinstance(v, str),
    }
    checker = type_checks.get(expected_type, lambda v: True)
    return checker(value)


def check_uniqueness(records: list[dict]) -> dict:
    """Check for duplicate records."""
    if not records:
        return {"name": "uniqueness", "passed": True, "message": "No records", "score": 1.0}

    # Hash records for comparison
    seen = set()
    duplicates = 0

    for record in records:
        record_hash = hash(frozenset((k, str(v)) for k, v in sorted(record.items())))
        if record_hash in seen:
            duplicates += 1
        seen.add(record_hash)

    duplicate_rate = duplicates / len(records)
    passed = duplicate_rate < 0.01  # Less than 1% duplicates

    return {
        "name": "uniqueness",
        "passed": passed,
        "message": f"{duplicates} duplicates ({duplicate_rate:.1%})" if duplicates else "No duplicates",
        "score": 1.0 - duplicate_rate,
    }


def check_null_rate(records: list[dict]) -> dict:
    """Check null rate is acceptable."""
    if not records:
        return {"name": "null_rate", "passed": True, "message": "No records", "score": 1.0}

    total_values = 0
    null_values = 0

    for record in records:
        for value in record.values():
            total_values += 1
            if value is None:
                null_values += 1

    null_rate = null_values / total_values if total_values > 0 else 0
    passed = null_rate < 0.2  # Less than 20% nulls

    return {
        "name": "null_rate",
        "passed": passed,
        "message": f"Null rate: {null_rate:.1%}",
        "score": 1.0 - null_rate,
    }


def check_distributions(records: list[dict], schema: dict) -> dict:
    """Check value distributions match constraints (premium check)."""
    if not records or not schema:
        return {"name": "distribution", "passed": True, "message": "No data", "score": 1.0}

    issues = []

    for field_name, field_def in schema.items():
        if not isinstance(field_def, dict):
            continue

        constraints = field_def.get("constraints", {})
        values = [r.get(field_name) for r in records if r.get(field_name) is not None]

        if not values:
            continue

        field_type = field_def.get("type", "string")

        # Check numeric constraints
        if field_type in ["int", "float"] and values:
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values:
                min_val = constraints.get("min")
                max_val = constraints.get("max")

                actual_min = min(numeric_values)
                actual_max = max(numeric_values)

                if min_val is not None and actual_min < min_val:
                    issues.append(f"{field_name}: min {actual_min} < {min_val}")
                if max_val is not None and actual_max > max_val:
                    issues.append(f"{field_name}: max {actual_max} > {max_val}")

        # Check category values
        if field_type == "category":
            allowed = constraints.get("values", [])
            if allowed:
                invalid = [v for v in values if v not in allowed]
                if invalid:
                    issues.append(f"{field_name}: {len(invalid)} invalid categories")

    passed = len(issues) == 0
    score = 1.0 - (len(issues) / len(schema)) if schema else 1.0

    return {
        "name": "distribution",
        "passed": passed,
        "message": "; ".join(issues[:3]) if issues else "Distributions valid",
        "score": max(0, score),
    }

"""Template schema validator and computed field evaluator.

Validates ToolTemplate.schema against expected structure.
Evaluates computed grid columns (e.g. RPN = severity * occurrence * detection).

v1 computed fields: only support `a * b * c` style multiplication formulas.
"""

import operator
import re

VALID_PRIMITIVE_TYPES = {"text", "grid", "tree", "checklist", "action_list"}
VALID_CELL_TYPES = {"text", "score", "computed", "dropdown"}


class SchemaValidationError(Exception):
    """Raised when a template schema is invalid."""

    def __init__(self, errors):
        self.errors = errors if isinstance(errors, list) else [errors]
        super().__init__("; ".join(self.errors))


def validate_template_schema(schema):
    """Validate a ToolTemplate schema dict.

    Returns list of error strings (empty = valid).
    """
    errors = []

    if not isinstance(schema, dict):
        return ["Schema must be a dict"]

    sections = schema.get("sections")
    if not isinstance(sections, list):
        return ["Schema must have a 'sections' list"]

    if not sections:
        return ["Schema must have at least one section"]

    seen_keys = set()
    for i, section in enumerate(sections):
        prefix = f"sections[{i}]"

        if not isinstance(section, dict):
            errors.append(f"{prefix}: must be a dict")
            continue

        # Required fields
        key = section.get("key")
        if not key or not isinstance(key, str):
            errors.append(f"{prefix}: 'key' is required (non-empty string)")
        elif key in seen_keys:
            errors.append(f"{prefix}: duplicate key '{key}'")
        else:
            seen_keys.add(key)

        ptype = section.get("type")
        if ptype not in VALID_PRIMITIVE_TYPES:
            errors.append(f"{prefix}: 'type' must be one of {VALID_PRIMITIVE_TYPES}, got '{ptype}'")

        if "label" not in section:
            errors.append(f"{prefix}: 'label' is required")

        # Type-specific validation
        config = section.get("config", {})
        if ptype == "grid":
            errors.extend(_validate_grid_config(config, prefix))
        elif ptype == "score":
            # Score is a cell type, not a section type
            errors.append(f"{prefix}: 'score' is a cell type, not a section type")

    return errors


def _validate_grid_config(config, prefix):
    """Validate grid section config (columns definition)."""
    errors = []
    columns = config.get("columns")
    if not isinstance(columns, list):
        errors.append(f"{prefix}.config: 'columns' list is required for grid type")
        return errors

    if not columns:
        errors.append(f"{prefix}.config: 'columns' must not be empty")
        return errors

    col_keys = set()
    for j, col in enumerate(columns):
        cpfx = f"{prefix}.config.columns[{j}]"
        if not isinstance(col, dict):
            errors.append(f"{cpfx}: must be a dict")
            continue

        col_key = col.get("key")
        if not col_key:
            errors.append(f"{cpfx}: 'key' is required")
        elif col_key in col_keys:
            errors.append(f"{cpfx}: duplicate column key '{col_key}'")
        else:
            col_keys.add(col_key)

        col_type = col.get("type")
        if col_type not in VALID_CELL_TYPES:
            errors.append(f"{cpfx}: 'type' must be one of {VALID_CELL_TYPES}, got '{col_type}'")

        if col_type == "score":
            if "min" not in col or "max" not in col:
                errors.append(f"{cpfx}: score type requires 'min' and 'max'")

        if col_type == "computed":
            formula = col.get("formula", "")
            if not formula:
                errors.append(f"{cpfx}: computed type requires 'formula'")
            else:
                refs = _parse_formula_refs(formula)
                missing = refs - col_keys
                # Allow forward references — they'll be validated at the end
                # But warn about unknown refs
                if missing:
                    # Check if they're defined later in columns
                    all_col_keys = {c.get("key") for c in columns if isinstance(c, dict)}
                    truly_missing = missing - all_col_keys
                    if truly_missing:
                        errors.append(f"{cpfx}: formula references unknown columns: {truly_missing}")

    return errors


def _parse_formula_refs(formula):
    """Extract column references from a formula string.

    v1: only supports `a * b * c` multiplication chains.
    Returns set of referenced column keys.
    """
    # Strip whitespace, split on *
    refs = set()
    for token in re.split(r"\s*\*\s*", formula.strip()):
        token = token.strip()
        if token and not token.isdigit():
            refs.add(token)
    return refs


def evaluate_computed_fields(data, config):
    """Evaluate computed columns in grid data.

    Takes a grid section's data dict and its config, returns
    a new data dict with computed columns filled in.
    """
    columns = config.get("columns", [])
    computed_cols = [col for col in columns if col.get("type") == "computed"]
    if not computed_cols:
        return data

    rows = data.get("rows", [])
    if not rows:
        return data

    new_rows = []
    for row in rows:
        new_row = dict(row)
        for col in computed_cols:
            key = col["key"]
            formula = col.get("formula", "")
            value = _eval_formula(formula, new_row)
            if value is not None:
                new_row[key] = value
        new_rows.append(new_row)

    return {**data, "rows": new_rows}


def _eval_formula(formula, row):
    """Evaluate a multiplication formula against a row dict.

    v1: only supports `a * b * c` chains.
    Returns numeric result or None if any ref is missing/non-numeric.
    """
    refs = re.split(r"\s*\*\s*", formula.strip())
    result = 1
    for ref in refs:
        ref = ref.strip()
        if not ref:
            continue
        if ref.isdigit():
            result *= int(ref)
            continue
        val = row.get(ref)
        if val is None:
            return None
        try:
            result = operator.mul(result, float(val))
        except (TypeError, ValueError):
            return None
    # Return int if result is whole number
    if isinstance(result, float) and result == int(result):
        return int(result)
    return result

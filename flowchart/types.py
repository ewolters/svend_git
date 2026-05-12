"""Semantic type system for flowchart port connections.

Ported from sandbox/semantic_types.py (54/54 tests).
Production version — used by execution engine and connection validation.

8 categories: metric, spec, config, data, chart, text, list, document.
Domain-specific categories (vsm:*, hoshin:*) extend naturally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

_TYPE_PATTERN = re.compile(r"^([a-z_]+):([a-z_*]+(?:\[\])?)$")


@dataclass(frozen=True)
class SemanticType:
    """A parsed semantic type like 'metric:cpk' or 'data:column[]'."""

    category: str
    subtype: str
    is_array: bool = False

    @classmethod
    def parse(cls, type_str: str) -> SemanticType:
        m = _TYPE_PATTERN.match(type_str)
        if not m:
            raise ValueError(f"Invalid semantic type '{type_str}'. Expected 'category:subtype' (e.g. 'metric:cpk')")
        category, subtype = m.group(1), m.group(2)
        is_array = subtype.endswith("[]")
        if is_array:
            subtype = subtype[:-2]
        return cls(category=category, subtype=subtype, is_array=is_array)

    @property
    def is_wildcard(self) -> bool:
        return self.subtype == "*"

    def accepts(self, other: SemanticType) -> bool:
        if self.category != other.category:
            return False
        if self.is_array != other.is_array:
            return False
        if self.is_wildcard:
            return True
        return self.subtype == other.subtype

    def __str__(self) -> str:
        arr = "[]" if self.is_array else ""
        return f"{self.category}:{self.subtype}{arr}"


def validate_connection(source_type: str, target_type: str) -> Dict:
    """Validate that a source output type is compatible with a target input type."""
    try:
        source = SemanticType.parse(source_type)
        target = SemanticType.parse(target_type)
    except ValueError as e:
        return {"valid": False, "error": str(e)}

    if not target.accepts(source):
        return {"valid": False, "error": f"Type mismatch: {source} -> {target}"}

    return {"valid": True, "error": ""}

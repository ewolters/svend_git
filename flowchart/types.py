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


def _has_cycle(devices: list, connections: list) -> bool:
    """Check if connections form a cycle using Kahn's algorithm."""
    adj = {d["id"]: [] for d in devices}
    in_deg = {d["id"]: 0 for d in devices}

    for conn in connections:
        src_dev = conn["source"].split(".", 1)[0]
        tgt_dev = conn["target"].split(".", 1)[0]
        if tgt_dev not in adj.get(src_dev, []):
            adj.setdefault(src_dev, []).append(tgt_dev)
            in_deg[tgt_dev] = in_deg.get(tgt_dev, 0) + 1

    queue = [d for d in in_deg if in_deg[d] == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbor in adj.get(node, []):
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)

    return visited != len(in_deg)


def _find_port(devices: list, port_ref: str) -> dict:
    """Find a port definition by 'device_id.port_name' reference.

    Returns dict with type/multi/direction keys, or None if not found.
    """
    device_id, port_name = port_ref.split(".", 1)
    for dev in devices:
        if dev["id"] != device_id:
            continue
        for p in dev.get("ports", {}).get("inputs", []):
            if p["name"] == port_name:
                return {**p, "direction": "input"}
        for p in dev.get("ports", {}).get("outputs", []):
            if p["name"] == port_name:
                return {**p, "direction": "output"}
    return None


def validate_connection_in_context(definition: dict, source: str, target: str) -> Dict:
    """Validate a proposed connection within a full flowchart definition.

    Checks:
    1. Source port exists and is an output
    2. Target port exists and is an input
    3. Semantic types are compatible
    4. Adding this connection doesn't create a cycle
    5. Connection doesn't already exist
    """
    devices = definition.get("devices", [])
    connections = definition.get("connections", [])

    src_port = _find_port(devices, source)
    if not src_port:
        return {"valid": False, "error": f"Source port not found: {source}"}
    if src_port["direction"] != "output":
        return {"valid": False, "error": f"Source must be an output port: {source}"}

    tgt_port = _find_port(devices, target)
    if not tgt_port:
        return {"valid": False, "error": f"Target port not found: {target}"}
    if tgt_port["direction"] != "input":
        return {"valid": False, "error": f"Target must be an input port: {target}"}

    type_result = validate_connection(src_port["type"], tgt_port["type"])
    if not type_result["valid"]:
        return type_result

    for conn in connections:
        if conn["source"] == source and conn["target"] == target:
            return {"valid": False, "error": f"Connection already exists: {source} -> {target}"}

    test_connections = connections + [{"source": source, "target": target}]
    if _has_cycle(devices, test_connections):
        return {"valid": False, "error": f"Connection would create a cycle: {source} -> {target}"}

    return {"valid": True, "error": "", "type": src_port["type"]}

#!/usr/bin/env python3
"""
Sandbox: Semantic Type System for SVEND Flowchart Engine
========================================================

Tests the type system that enforces methodology-correct connections
between devices. No Django, no production code — pure Python.

Run:  python3 ~/kjerne/sandbox/semantic_types.py

What we're testing:
  1. Semantic type parsing (category:subtype)
  2. Port declarations with multiplicity (single vs multi)
  3. Connection validation (can output X connect to input Y?)
  4. Wildcard matching (metric:* accepts any metric subtype)
  5. Data source device (entry point for flowcharts)
  6. Full template wiring (PPAP, Green Belt DMAIC, Quick Cpk)
  7. Flowchart-level validation (all required ports connected?)
  8. Edge cases that will break production code

Reference: svend-product-spec.md Section 4 "Semantic Type System"

DECISIONS MADE (from sandbox round 1, 2026-05-12):
  - spec/config are SEPARATE categories from metric. A spec limit is a
    requirement, not a measurement. config is operational. Merging them
    under metric:* would let you cable a measurement into a spec slot.
  - list is SEPARATE from text. Lists are structured (violations, actions),
    text is narrative. Report builder has explicit ports for both.
  - Ports have multiplicity: 'single' (default) or 'multi'. Multi-ports
    collect all connected values into a list. Required for report builder
    (N charts from N devices) and any aggregation device.
  - Fan-out is allowed: one output port -> multiple input ports. This is
    standard DAG behavior (e.g., same data feeding cap + control chart).
  - PCL mapping is deferred — needs semantic_type field on Measure model.
    That's a migration, not a sandbox concern. Noted as debt.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 1. Semantic Type System
# ---------------------------------------------------------------------------
#
# LEARNED: The original spec (Section 4) listed 4 color categories but the
# device port tables used 8 actual categories: metric, data, chart, text,
# spec, config, list, document. All 8 are real — collapsing them loses
# the methodology enforcement that's the whole point.
#
# Category boundaries are LOAD-BEARING. Raj (May 11): "you wouldn't plug
# meters into a slot expecting PSI." Same principle — spec:usl is not
# metric:usl. They flow into different port types on a device for a reason.

_TYPE_PATTERN = re.compile(r"^([a-z_]+):([a-z_*]+(?:\[\])?)$")


@dataclass(frozen=True)
class SemanticType:
    """A parsed semantic type like 'metric:cpk' or 'data:column'.

    Format: category:subtype
    Array:  category:subtype[]
    Wild:   category:*

    Categories (from spec + sandbox testing):
      metric   — numbers with measurement meaning (cpk, ppk, mean, p_value)
      spec     — specification limits (usl, lsl, target) — NOT metrics
      config   — operational parameters (subgroup_size, alpha) — NOT metrics
      data     — vectors/columns (column, categorical, datetime, matrix)
      chart    — visualizations (histogram, control_chart, scatter, pareto)
      text     — narrative/prose (summary, interpretation, narrative)
      list     — structured sequences (violations, actions, goals)
      document — output artifacts (pdf, html) — always terminal
    """

    category: str
    subtype: str
    is_array: bool = False

    @classmethod
    def parse(cls, type_str: str) -> "SemanticType":
        """Parse 'category:subtype' or 'category:subtype[]'.

        Raises ValueError on malformed input. Enforces lowercase.
        """
        m = _TYPE_PATTERN.match(type_str)
        if not m:
            raise ValueError(
                f"Invalid semantic type '{type_str}'. Expected 'category:subtype' (e.g. 'metric:cpk', 'data:column[]')"
            )
        category, subtype = m.group(1), m.group(2)
        is_array = subtype.endswith("[]")
        if is_array:
            subtype = subtype[:-2]
        return cls(category=category, subtype=subtype, is_array=is_array)

    @property
    def is_wildcard(self) -> bool:
        return self.subtype == "*"

    def accepts(self, other: "SemanticType") -> bool:
        """Can this type (as an INPUT port) accept `other` (an OUTPUT port)?

        Rules:
          - Category must match exactly (no cross-category)
          - Wildcard (category:*) accepts any subtype in same category
          - Exact subtype match required for non-wildcard
          - Array dimensions must match (no scalar->array or vice versa)
        """
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


# ---------------------------------------------------------------------------
# 2. Port Declarations
# ---------------------------------------------------------------------------
#
# LEARNED (round 1): Single-port model breaks on real templates. The PPAP
# template needs cap.histogram AND cc.chart both going to report_builder.charts.
# Without multiplicity, you can't wire it.
#
# Solution: Port.multi = True means the port collects all connections into a
# list. Default is False (single connection only). Multi-ports are the norm
# for aggregation devices (report builder, composite metrics).
#
# LEARNED: Required + default interact. If required=True and default is set,
# the port works unconnected (uses default) but validates connections when
# present. This is how config ports work — subgroup_size defaults to 1 but
# accepts a connection if you wire one.
#
# DECIDED (round 3): Config ports ARE regular ports. No special flag, no
# separate list, no `connectable` property. A "process config" device can
# output config:subgroup_size and capability study receives it through a
# cable — same as any other connection. When NOT connected, the device panel
# shows config ports as editable form fields with their default value.
# When connected, the form field is disabled/locked to the incoming value.
# This also solves shared config: one process config device fans out to
# N analysis devices. "Same version of every number" = just another cable.


@dataclass
class Port:
    """A typed input or output port on a device.

    Attributes:
        name: Port identifier, unique within device inputs/outputs.
        semantic_type: What kind of data flows through this port.
        multi: If True, accepts multiple connections (values collected as list).
               Default False = single connection only, second wire is an error.
        required: For input ports — must be connected (or have default)?
        default: Value to use when input port is not connected.
        description: Human-readable purpose.
    """

    name: str
    semantic_type: SemanticType
    multi: bool = False
    required: bool = True
    default: Any = None
    description: str = ""

    @classmethod
    def from_str(cls, name: str, type_str: str, **kwargs) -> "Port":
        return cls(name=name, semantic_type=SemanticType.parse(type_str), **kwargs)


@dataclass
class DeviceSchema:
    """Schema for a device (plugin) — declares all typed ports.

    This is the DECLARATION, not the runtime instance. In production,
    this maps to what Plugin.get_metadata() returns, extended with
    port declarations.
    """

    name: str
    description: str
    inputs: List[Port] = field(default_factory=list)
    outputs: List[Port] = field(default_factory=list)

    def get_input(self, name: str) -> Optional[Port]:
        return next((p for p in self.inputs if p.name == name), None)

    def get_output(self, name: str) -> Optional[Port]:
        return next((p for p in self.outputs if p.name == name), None)

    def input_names(self) -> List[str]:
        return [p.name for p in self.inputs]

    def output_names(self) -> List[str]:
        return [p.name for p in self.outputs]

    def required_inputs(self) -> List[Port]:
        """Inputs that MUST be connected (no default, required=True)."""
        return [p for p in self.inputs if p.required and p.default is None]


# ---------------------------------------------------------------------------
# 3. Connection + Flowchart Validation
# ---------------------------------------------------------------------------


@dataclass
class Connection:
    """An edge in the flowchart: source output -> target input."""

    source_device: str
    source_port: str
    target_device: str
    target_port: str


@dataclass
class ValidationResult:
    """Result of validating a single connection."""

    valid: bool
    connection: Connection
    source_type: Optional[SemanticType] = None
    target_type: Optional[SemanticType] = None
    error: str = ""


@dataclass
class FlowchartValidation:
    """Result of validating an entire flowchart (all connections + completeness)."""

    valid: bool = True
    connection_errors: List[str] = field(default_factory=list)
    missing_inputs: List[str] = field(default_factory=list)
    multiplicity_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_connection(
    connection: Connection,
    devices: Dict[str, DeviceSchema],
) -> ValidationResult:
    """Validate that a connection between two device ports is type-compatible."""

    result = ValidationResult(valid=False, connection=connection)

    source = devices.get(connection.source_device)
    if not source:
        result.error = f"Source device '{connection.source_device}' not found"
        return result

    target = devices.get(connection.target_device)
    if not target:
        result.error = f"Target device '{connection.target_device}' not found"
        return result

    source_port = source.get_output(connection.source_port)
    if not source_port:
        result.error = (
            f"Output port '{connection.source_port}' not found on "
            f"'{connection.source_device}'. "
            f"Available: {source.output_names()}"
        )
        return result

    target_port = target.get_input(connection.target_port)
    if not target_port:
        result.error = (
            f"Input port '{connection.target_port}' not found on "
            f"'{connection.target_device}'. "
            f"Available: {target.input_names()}"
        )
        return result

    result.source_type = source_port.semantic_type
    result.target_type = target_port.semantic_type

    if not target_port.semantic_type.accepts(source_port.semantic_type):
        result.error = (
            f"Type mismatch: {source_port.semantic_type} -> "
            f"{target_port.semantic_type}. "
            f"'{connection.target_device}.{connection.target_port}' "
            f"expects {target_port.semantic_type.category}:"
            f"{target_port.semantic_type.subtype}, "
            f"got {source_port.semantic_type.category}:"
            f"{source_port.semantic_type.subtype}"
        )
        return result

    if connection.source_device == connection.target_device:
        result.error = "Cannot connect a device to itself"
        return result

    result.valid = True
    return result


def validate_flowchart(
    devices: Dict[str, DeviceSchema],
    connections: List[Connection],
) -> FlowchartValidation:
    """Validate an entire flowchart: all connections + required inputs + multiplicity.

    This is what runs when a user saves a template or switches to persistent mode.
    Scratch mode skips this (you can have dangling wires while exploring).
    """
    result = FlowchartValidation()

    # Track connections per target port for multiplicity check
    # key = (device_name, port_name), value = list of source connections
    port_connections: Dict[tuple, List[Connection]] = {}

    # 1. Validate each connection individually
    for conn in connections:
        r = validate_connection(conn, devices)
        if not r.valid:
            result.valid = False
            result.connection_errors.append(r.error)
        else:
            key = (conn.target_device, conn.target_port)
            port_connections.setdefault(key, []).append(conn)

    # 2. Check multiplicity — single ports can't have >1 connection
    for (dev_name, port_name), conns in port_connections.items():
        device = devices[dev_name]
        port = device.get_input(port_name)
        if port and not port.multi and len(conns) > 1:
            result.valid = False
            sources = [f"{c.source_device}.{c.source_port}" for c in conns]
            result.multiplicity_errors.append(
                f"'{dev_name}.{port_name}' is single-connection but has {len(conns)} connections from: {sources}"
            )

    # 3. Check required inputs are connected (or have defaults)
    connected_inputs = set(port_connections.keys())
    for dev_name, device in devices.items():
        for port in device.required_inputs():
            if (dev_name, port.name) not in connected_inputs:
                result.valid = False
                result.missing_inputs.append(
                    f"'{dev_name}.{port.name}' ({port.semantic_type}) is required but not connected"
                )

    # 4. Cycle detection (Kahn's algorithm — topological sort)
    # GOTCHA #3 resolved: individual connection validation won't catch A->B->C->A.
    adj: Dict[str, set] = {name: set() for name in devices}
    in_deg: Dict[str, int] = {name: 0 for name in devices}
    for conn in connections:
        if conn.target_device not in adj.get(conn.source_device, set()):
            adj.setdefault(conn.source_device, set()).add(conn.target_device)
            in_deg[conn.target_device] = in_deg.get(conn.target_device, 0) + 1

    queue = [d for d in devices if in_deg.get(d, 0) == 0]
    sorted_count = 0
    while queue:
        node = queue.pop(0)
        sorted_count += 1
        for neighbor in adj.get(node, set()):
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)

    if sorted_count != len(devices):
        result.valid = False
        cycled = [d for d in devices if in_deg.get(d, 0) > 0]
        result.warnings.append(f"CYCLE detected involving: {cycled}")

    # 5. Warnings (non-blocking)
    # Detect terminal devices with no outgoing connections
    sources_used = {conn.source_device for conn in connections}
    for dev_name in devices:
        if dev_name not in sources_used:
            device = devices[dev_name]
            if device.outputs:
                result.warnings.append(
                    f"'{dev_name}' has outputs but nothing is connected to them (terminal device or incomplete wiring)"
                )

    return result


# ---------------------------------------------------------------------------
# 4. Devices
# ---------------------------------------------------------------------------
#
# LEARNED: The data source device is REQUIRED. Without it, there's no way to
# get data:column into the flowchart. It's the entry point — paste CSV,
# pick columns, each column becomes a typed output.
#
# LEARNED: Report builder needs list:* input in addition to text:*.
# Control chart outputs list:violations, which is structured (not narrative).
# Report builder renders lists as bullet points, text as paragraphs.


def make_data_source() -> DeviceSchema:
    """Data source device — the entry point for every flowchart.

    This is Tomasz's "paste 50 measurements" device. In the UI it's:
    - Paste/upload CSV
    - Column picker assigns semantic types to each column
    - Outputs typed columns for downstream devices

    GOTCHA: Column count is dynamic. A 3-column CSV produces 3 output ports.
    Static schema can't declare that. Options:
      A) Fixed max ports (data_1, data_2, ...) — ugly, arbitrary limit
      B) Single output port with multi=True — loses column identity
      C) Dynamic schema — device declares ports at config time, not class time
    --> Going with (C) for production. For sandbox, use fixed ports.
    """
    return DeviceSchema(
        name="data_source",
        description="Data ingestion — paste/upload measurements",
        inputs=[
            # Data source has NO typed inputs — it's a root device.
            # Config (column names, types) comes from the UI, not from ports.
        ],
        outputs=[
            Port.from_str("measurements", "data:column", description="Primary measurement column"),
            Port.from_str("usl", "spec:usl", description="Upper spec limit (from data or entered)"),
            Port.from_str("lsl", "spec:lsl", description="Lower spec limit (from data or entered)"),
        ],
    )


def make_capability_study() -> DeviceSchema:
    return DeviceSchema(
        name="capability_study",
        description="Process capability analysis — Cp, Cpk, Pp, Ppk, sigma level",
        inputs=[
            Port.from_str("data", "data:column", description="Measurement data"),
            Port.from_str("usl", "spec:usl", description="Upper spec limit"),
            Port.from_str("lsl", "spec:lsl", description="Lower spec limit"),
            Port.from_str("subgroup_size", "config:subgroup_size", required=False, default=1),
        ],
        outputs=[
            Port.from_str("cpk", "metric:cpk"),
            Port.from_str("ppk", "metric:ppk"),
            Port.from_str("sigma_level", "metric:sigma_level"),
            Port.from_str("histogram", "chart:histogram"),
            Port.from_str("qq_plot", "chart:qq_plot"),
            Port.from_str("summary", "text:summary"),
        ],
    )


def make_control_chart() -> DeviceSchema:
    return DeviceSchema(
        name="control_chart",
        description="SPC control chart — I-MR, Xbar-R, Xbar-S, p, c, u, np",
        inputs=[
            Port.from_str("data", "data:column", description="Measurement data"),
            Port.from_str("chart_type", "config:chart_type", required=False, default="i_mr"),
            Port.from_str("subgroup_size", "config:subgroup_size", required=False, default=1),
        ],
        outputs=[
            Port.from_str("chart", "chart:control_chart"),
            Port.from_str("mean", "metric:mean"),
            Port.from_str("ucl", "metric:ucl"),
            Port.from_str("lcl", "metric:lcl"),
            Port.from_str("violations", "list:violations"),
        ],
    )


def make_gage_rr() -> DeviceSchema:
    """Gage R&R — measurement system analysis.

    Tomasz's use case: "my Excel spreadsheet may have errors,
    customer flagged my %GRR as unusual." This device validates
    the measurement system BEFORE you trust the data.

    GOTCHA: Gage R&R input is a structured dataset (part, operator,
    measurement), not a single column. Need data:matrix or multiple
    data:column inputs. Going with multiple columns — matches how
    the data actually arrives.
    """
    return DeviceSchema(
        name="gage_rr",
        description="Gage R&R measurement system analysis",
        inputs=[
            Port.from_str("measurements", "data:column", description="Measurement values"),
            Port.from_str("parts", "data:categorical", description="Part identifiers"),
            Port.from_str("operators", "data:categorical", description="Operator identifiers"),
        ],
        outputs=[
            Port.from_str("grr_percent", "metric:grr_percent"),
            Port.from_str("ndc", "metric:ndc"),
            Port.from_str("repeatability", "metric:repeatability"),
            Port.from_str("reproducibility", "metric:reproducibility"),
            Port.from_str("chart", "chart:grr_chart"),
            Port.from_str("summary", "text:summary"),
        ],
    )


def make_monte_carlo() -> DeviceSchema:
    return DeviceSchema(
        name="monte_carlo_sim",
        description="Monte Carlo simulation — parameter distributions + sampling",
        inputs=[
            # Accepts ANY metric — this is the wildcard case.
            # multi=True because you feed it N parameters from N devices.
            Port.from_str(
                "parameters", "metric:*", multi=True, description="Any metric values as simulation parameters"
            ),
        ],
        outputs=[
            Port.from_str("percentiles", "metric:percentiles"),
            Port.from_str("distribution", "chart:distribution"),
            Port.from_str("probability", "metric:probability"),
        ],
    )


def make_report_builder() -> DeviceSchema:
    """Report builder — the 'last mile' device.

    LEARNED: Needs both list:* and text:* inputs. Control chart outputs
    list:violations (structured), capability outputs text:summary (narrative).
    Report builder renders them differently — bullets vs paragraphs.

    All inputs are multi=True — collects from every upstream device.
    """
    return DeviceSchema(
        name="report_builder",
        description="Document assembly — accepts charts, metrics, text, lists",
        inputs=[
            Port.from_str("charts", "chart:*", multi=True, description="Any charts (rendered as figures)"),
            Port.from_str("metrics", "metric:*", multi=True, description="Any metrics (rendered as summary table)"),
            Port.from_str("narrative", "text:*", multi=True, description="Narrative text (rendered as paragraphs)"),
            Port.from_str("lists", "list:*", multi=True, description="Structured lists (rendered as bullet points)"),
        ],
        outputs=[
            Port.from_str("pdf", "document:pdf"),
            Port.from_str("html", "document:html"),
        ],
    )


def make_vsm() -> DeviceSchema:
    return DeviceSchema(
        name="vsm",
        description="Value Stream Map — process flow with cycle/changeover/WIP",
        inputs=[
            Port.from_str("cycle_times", "metric:cycle_time[]"),
            Port.from_str("changeover_times", "metric:changeover_time[]"),
            Port.from_str("wip", "metric:wip[]"),
        ],
        outputs=[
            Port.from_str("lead_time", "metric:lead_time"),
            Port.from_str("takt", "metric:takt"),
            Port.from_str("process_map", "chart:process_map"),
            Port.from_str("pce", "metric:pce"),
        ],
    )


def make_process_config() -> DeviceSchema:
    """Process config device — shared settings for analysis devices.

    DECIDED (round 3): Config is just another port. This device outputs
    config values that multiple analysis devices can connect to. Solves
    "same version of every number" — one source of truth for subgroup_size,
    alpha, chart_type, etc.

    When NOT connected to downstream devices, those devices show config
    ports as editable form fields with defaults. When connected, the
    form field locks to the incoming value.
    """
    return DeviceSchema(
        name="process_config",
        description="Shared process configuration — subgroup size, alpha, chart type",
        inputs=[],  # Root device — user enters values directly
        outputs=[
            Port.from_str("subgroup_size", "config:subgroup_size", description="Subgroup size for SPC calculations"),
            Port.from_str("chart_type", "config:chart_type", description="Control chart type (i_mr, xbar_r, etc.)"),
            Port.from_str("alpha", "config:alpha", description="Significance level for hypothesis tests"),
        ],
    )


def make_fishbone() -> DeviceSchema:
    """Fishbone / Ishikawa diagram.

    Part of the Green Belt DMAIC template. Takes a problem statement
    and outputs categorized causes + a diagram.

    GOTCHA: Fishbone is unusual — its primary input is text, not data.
    The problem statement comes from the user or from a text:* output
    of another device. Its output (list of causes) feeds into action
    planning or FMEA.
    """
    return DeviceSchema(
        name="fishbone",
        description="Ishikawa / cause-and-effect diagram",
        inputs=[
            Port.from_str(
                "problem", "text:summary", required=False, description="Problem statement (or enter directly)"
            ),
        ],
        outputs=[
            Port.from_str("causes", "list:causes"),
            Port.from_str("diagram", "chart:fishbone"),
        ],
    )


# ---------------------------------------------------------------------------
# 5. Test Runner
# ---------------------------------------------------------------------------


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []

    def test(self, name: str, condition: bool, detail: str = ""):
        if condition:
            self.passed += 1
            print(f"  PASS  {name}")
        else:
            self.failed += 1
            msg = f"  FAIL  {name}"
            if detail:
                msg += f" -- {detail}"
            print(msg)
            self.errors.append(name)

    def section(self, title: str):
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"  RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("  FAILURES:")
            for e in self.errors:
                print(f"    - {e}")
        print(f"{'=' * 60}")
        return self.failed == 0


# ---------------------------------------------------------------------------
# 6. Tests
# ---------------------------------------------------------------------------


def run_tests():
    t = TestRunner()

    # ---------------------------------------------------------------
    t.section("Type Parsing")
    # ---------------------------------------------------------------

    st = SemanticType.parse("metric:cpk")
    t.test("parse metric:cpk", st.category == "metric" and st.subtype == "cpk")

    st = SemanticType.parse("data:column")
    t.test("parse data:column", st.category == "data" and st.subtype == "column")

    st = SemanticType.parse("metric:cycle_time[]")
    t.test("parse array type", st.is_array and st.subtype == "cycle_time")

    st = SemanticType.parse("chart:*")
    t.test("parse wildcard", st.is_wildcard and st.category == "chart")

    for bad, label in [
        ("cpk", "bare name"),
        ("metric:cpk:extra", "triple-colon"),
        ("", "empty string"),
        ("Metric:Cpk", "uppercase"),
    ]:
        try:
            SemanticType.parse(bad)
            t.test(f"reject {label}", False, f"'{bad}' should raise ValueError")
        except ValueError:
            t.test(f"reject {label}", True)

    # ---------------------------------------------------------------
    t.section("Type Compatibility — The Raj Tests")
    # ---------------------------------------------------------------
    # These encode the methodology enforcement that Raj called out:
    # "you wouldn't plug meters into a slot expecting PSI."

    cpk = SemanticType.parse("metric:cpk")
    ppk = SemanticType.parse("metric:ppk")
    p_value = SemanticType.parse("metric:p_value")
    any_metric = SemanticType.parse("metric:*")
    data_col = SemanticType.parse("data:column")
    any_chart = SemanticType.parse("chart:*")

    t.test("exact: metric:cpk -> metric:cpk", cpk.accepts(cpk))
    t.test("wildcard: metric:* accepts metric:cpk", any_metric.accepts(cpk))
    t.test("wildcard: metric:* accepts metric:ppk", any_metric.accepts(ppk))

    # THE RAJ TEST: p-value into Cpk slot must fail
    t.test("RAJ: metric:cpk REJECTS metric:p_value", not cpk.accepts(p_value))

    # Cross-category must fail
    t.test("cross-category: data:column rejects metric:cpk", not data_col.accepts(cpk))
    t.test("cross-category: chart:* rejects metric:cpk", not any_chart.accepts(cpk))

    # spec is NOT metric
    spec_usl = SemanticType.parse("spec:usl")
    t.test("DECISION: metric:* rejects spec:usl (spec != metric)", not any_metric.accepts(spec_usl))

    # config is NOT metric
    config_sg = SemanticType.parse("config:subgroup_size")
    t.test("DECISION: metric:* rejects config:subgroup_size", not any_metric.accepts(config_sg))

    # list is NOT text
    list_v = SemanticType.parse("list:violations")
    text_wild = SemanticType.parse("text:*")
    t.test("DECISION: text:* rejects list:violations (list != text)", not text_wild.accepts(list_v))

    # Array compatibility
    arr_in = SemanticType.parse("metric:cycle_time[]")
    arr_out = SemanticType.parse("metric:cycle_time[]")
    scalar = SemanticType.parse("metric:cycle_time")
    t.test("array: cycle_time[] -> cycle_time[]", arr_in.accepts(arr_out))
    t.test("array rejects scalar", not arr_in.accepts(scalar))
    t.test("scalar rejects array", not scalar.accepts(arr_out))

    # ---------------------------------------------------------------
    t.section("Port Multiplicity")
    # ---------------------------------------------------------------
    # LEARNED: Report builder needs multi=True on all inputs. PPAP template
    # wires cap.histogram AND cc.chart both to rb.charts.

    rb = make_report_builder()
    t.test("report_builder.charts is multi", rb.get_input("charts").multi)
    t.test("report_builder.metrics is multi", rb.get_input("metrics").multi)
    t.test("report_builder.narrative is multi", rb.get_input("narrative").multi)
    t.test("report_builder.lists is multi", rb.get_input("lists").multi)

    cap = make_capability_study()
    t.test("capability.data is NOT multi (single data source)", not cap.get_input("data").multi)

    mc = make_monte_carlo()
    t.test("monte_carlo.parameters is multi (N params from N devices)", mc.get_input("parameters").multi)

    # ---------------------------------------------------------------
    t.section("Data Source Device")
    # ---------------------------------------------------------------

    ds = make_data_source()
    t.test("data_source has no required inputs (root device)", len(ds.required_inputs()) == 0)
    t.test("data_source outputs data:column", str(ds.get_output("measurements").semantic_type) == "data:column")
    t.test("data_source outputs spec:usl", str(ds.get_output("usl").semantic_type) == "spec:usl")
    t.test("data_source outputs spec:lsl", str(ds.get_output("lsl").semantic_type) == "spec:lsl")

    # ---------------------------------------------------------------
    t.section("Template: Quick Cpk (Tomasz's 30-second test)")
    # ---------------------------------------------------------------
    # data_source -> capability_study -> (done, or -> report_builder)
    # This is the Q shortcut. Minimal wiring.

    devices = {
        "ds": make_data_source(),
        "cap": make_capability_study(),
    }
    connections = [
        Connection("ds", "measurements", "cap", "data"),
        Connection("ds", "usl", "cap", "usl"),
        Connection("ds", "lsl", "cap", "lsl"),
    ]

    fv = validate_flowchart(devices, connections)
    t.test("Quick Cpk: all connections valid", len(fv.connection_errors) == 0)
    t.test("Quick Cpk: no multiplicity errors", len(fv.multiplicity_errors) == 0)
    t.test("Quick Cpk: no missing required inputs", len(fv.missing_inputs) == 0)
    t.test("Quick Cpk: flowchart valid", fv.valid)
    if fv.warnings:
        print(f"    Warnings: {fv.warnings}")
        # Expected: capability outputs not connected to anything — that's fine
        # for Quick Cpk (results shown in device panel, no report needed)

    # ---------------------------------------------------------------
    t.section("Template: PPAP Package (Dana's audit trail)")
    # ---------------------------------------------------------------
    # data_source -> capability + control_chart -> report_builder
    # Both devices share the same data source (fan-out).
    # Report builder collects from both (multi-port).

    devices = {
        "ds": make_data_source(),
        "cap": make_capability_study(),
        "cc": make_control_chart(),
        "rb": make_report_builder(),
    }
    connections = [
        # Data source feeds both analysis devices (fan-out)
        Connection("ds", "measurements", "cap", "data"),
        Connection("ds", "usl", "cap", "usl"),
        Connection("ds", "lsl", "cap", "lsl"),
        Connection("ds", "measurements", "cc", "data"),
        # Capability -> report
        Connection("cap", "histogram", "rb", "charts"),
        Connection("cap", "qq_plot", "rb", "charts"),
        Connection("cap", "cpk", "rb", "metrics"),
        Connection("cap", "ppk", "rb", "metrics"),
        Connection("cap", "sigma_level", "rb", "metrics"),
        Connection("cap", "summary", "rb", "narrative"),
        # Control chart -> report
        Connection("cc", "chart", "rb", "charts"),
        Connection("cc", "mean", "rb", "metrics"),
        Connection("cc", "ucl", "rb", "metrics"),
        Connection("cc", "lcl", "rb", "metrics"),
        Connection("cc", "violations", "rb", "lists"),
    ]

    fv = validate_flowchart(devices, connections)
    t.test("PPAP: all connections valid", len(fv.connection_errors) == 0)
    t.test("PPAP: no multiplicity errors", len(fv.multiplicity_errors) == 0)
    t.test("PPAP: no missing required inputs", len(fv.missing_inputs) == 0)
    t.test("PPAP: flowchart valid", fv.valid)

    if not fv.valid:
        for e in fv.connection_errors:
            print(f"    CONNECTION: {e}")
        for e in fv.multiplicity_errors:
            print(f"    MULTIPLICITY: {e}")
        for e in fv.missing_inputs:
            print(f"    MISSING: {e}")

    if fv.warnings:
        for w in fv.warnings:
            print(f"    WARNING: {w}")

    # ---------------------------------------------------------------
    t.section("Template: Green Belt DMAIC (Carmen's classroom)")
    # ---------------------------------------------------------------
    # data_source -> capability -> control_chart -> fishbone -> report

    devices = {
        "ds": make_data_source(),
        "cap": make_capability_study(),
        "cc": make_control_chart(),
        "fish": make_fishbone(),
        "rb": make_report_builder(),
    }
    connections = [
        # Data in
        Connection("ds", "measurements", "cap", "data"),
        Connection("ds", "usl", "cap", "usl"),
        Connection("ds", "lsl", "cap", "lsl"),
        Connection("ds", "measurements", "cc", "data"),
        # Capability summary feeds fishbone problem statement
        Connection("cap", "summary", "fish", "problem"),
        # Everything -> report
        Connection("cap", "histogram", "rb", "charts"),
        Connection("cap", "cpk", "rb", "metrics"),
        Connection("cap", "summary", "rb", "narrative"),
        Connection("cc", "chart", "rb", "charts"),
        Connection("cc", "violations", "rb", "lists"),
        Connection("fish", "diagram", "rb", "charts"),
        Connection("fish", "causes", "rb", "lists"),
    ]

    fv = validate_flowchart(devices, connections)
    t.test("DMAIC: all connections valid", len(fv.connection_errors) == 0)
    t.test("DMAIC: no multiplicity errors", len(fv.multiplicity_errors) == 0)
    t.test("DMAIC: no missing required inputs", len(fv.missing_inputs) == 0)
    t.test("DMAIC: flowchart valid", fv.valid)

    if not fv.valid:
        for e in fv.connection_errors + fv.multiplicity_errors + fv.missing_inputs:
            print(f"    ERROR: {e}")

    # ---------------------------------------------------------------
    t.section("Multiplicity Enforcement")
    # ---------------------------------------------------------------
    # Two connections to a single-input port must fail.

    devices = {
        "ds1": make_data_source(),
        "ds2": make_data_source(),
        "cap": make_capability_study(),
    }
    connections = [
        Connection("ds1", "measurements", "cap", "data"),
        Connection("ds2", "measurements", "cap", "data"),  # SECOND wire to single port
        Connection("ds1", "usl", "cap", "usl"),
        Connection("ds1", "lsl", "cap", "lsl"),
    ]

    fv = validate_flowchart(devices, connections)
    t.test("REJECT two connections to single port cap.data", len(fv.multiplicity_errors) > 0)
    if fv.multiplicity_errors:
        print(f"    Caught: {fv.multiplicity_errors[0]}")

    # ---------------------------------------------------------------
    t.section("Missing Required Inputs")
    # ---------------------------------------------------------------
    # Capability without spec limits must fail validation.

    devices = {
        "ds": make_data_source(),
        "cap": make_capability_study(),
    }
    connections = [
        Connection("ds", "measurements", "cap", "data"),
        # NOT connecting usl or lsl — required inputs missing
    ]

    fv = validate_flowchart(devices, connections)
    t.test("REJECT capability without spec limits", len(fv.missing_inputs) > 0)
    t.test("identifies usl as missing", any("usl" in m for m in fv.missing_inputs))
    t.test("identifies lsl as missing", any("lsl" in m for m in fv.missing_inputs))
    for m in fv.missing_inputs:
        print(f"    Missing: {m}")

    # ---------------------------------------------------------------
    t.section("Invalid Connections (Type Enforcement)")
    # ---------------------------------------------------------------

    devices = {
        "cap": make_capability_study(),
        "cc": make_control_chart(),
        "rb": make_report_builder(),
    }

    # Try to wire metric:cpk into data:column — must fail
    conns = [Connection("cap", "cpk", "cc", "data")]
    fv = validate_flowchart(devices, conns)
    t.test("REJECT metric:cpk -> data:column", len(fv.connection_errors) > 0)
    if fv.connection_errors:
        print(f"    Caught: {fv.connection_errors[0]}")

    # Try to wire list:violations into chart:* — must fail
    conns = [Connection("cc", "violations", "rb", "charts")]
    fv = validate_flowchart(devices, conns)
    t.test("REJECT list:violations -> chart:*", len(fv.connection_errors) > 0)

    # ---------------------------------------------------------------
    t.section("Edge Cases — Things That Will Break Production")
    # ---------------------------------------------------------------

    # 1. Dynamic ports (data source with N columns)
    print("\n  GOTCHA #1: Dynamic ports")
    print("    Data source has N columns at runtime, but DeviceSchema is")
    print("    declared at class time. Production needs dynamic schema —")
    print("    device creates ports after user uploads CSV.")
    print("    Impact: DeviceSchema can't be a class attribute on Plugin.")
    print("    Fix: Plugin.get_schema(config) -> DeviceSchema (method, not attr)")
    print()

    # 2. PCL binding (device <-> PCL, not device <-> device)
    print("  GOTCHA #2: PCL binding is a different connection type")
    print("    Device port -> PCL Measure, or PCL Measure -> device port.")
    print("    PCL Measure uses (measure_type, value_type), not semantic_type.")
    print("    Need: either add semantic_type to Measure model, or a mapping.")
    print("    PCL has no data yet — adding a field is cheap (migration).")
    print()

    # 3. Circular connections (A -> B -> A)
    # RESOLVED: JS sandbox implements cycle detection via Kahn's algorithm
    # (topological sort). Ported back to Python here.
    print("  GOTCHA #3: Circular connections — RESOLVED")
    print("    Cycle detection via Kahn's algorithm in validate_flowchart().")
    print("    JS sandbox: implemented + tested (A->B->C->A detected).")
    print("    Python sandbox: implemented below.")
    print()

    # 4. Optional multi-port with zero connections
    print("  GOTCHA #4: Multi-port with zero connections")
    print("    Report builder with no charts connected — valid in scratch,")
    print("    but what does the report contain? Empty section? Omitted?")
    print("    Impact: report builder template rendering must handle empty lists.")
    print()

    # 5. Config ports — RESOLVED (round 3)
    print("  GOTCHA #5: Config ports — RESOLVED")
    print("    Config ports ARE regular ports. No special flag, no separate")
    print("    concept. A process config device outputs config:subgroup_size,")
    print("    capability study receives it through a cable. When NOT connected,")
    print("    the device panel shows config ports as editable form fields.")
    print("    When connected, the form field is disabled/locked.")
    print()

    # 6. Shared config — RESOLVED (round 3)
    print("  GOTCHA #6: Shared config — RESOLVED")
    print("    One process config device fans out to N analysis devices.")
    print("    Same pattern as data source. 'Same version of every number'")
    print("    = just another cable in the flowchart.")
    print()

    # ---------------------------------------------------------------
    t.section("Shared Config via Process Config Device")
    # ---------------------------------------------------------------
    # DECIDED: config ports are regular ports. Process config device
    # fans out to N analysis devices. "Same version of every number."

    devices = {
        "ds": make_data_source(),
        "cfg": make_process_config(),
        "cap": make_capability_study(),
        "cc": make_control_chart(),
        "rb": make_report_builder(),
    }
    connections = [
        # Data in
        Connection("ds", "measurements", "cap", "data"),
        Connection("ds", "usl", "cap", "usl"),
        Connection("ds", "lsl", "cap", "lsl"),
        Connection("ds", "measurements", "cc", "data"),
        # Shared config — one source, two consumers
        Connection("cfg", "subgroup_size", "cap", "subgroup_size"),
        Connection("cfg", "subgroup_size", "cc", "subgroup_size"),
        Connection("cfg", "chart_type", "cc", "chart_type"),
        # Results → report
        Connection("cap", "histogram", "rb", "charts"),
        Connection("cap", "cpk", "rb", "metrics"),
        Connection("cap", "summary", "rb", "narrative"),
        Connection("cc", "chart", "rb", "charts"),
        Connection("cc", "violations", "rb", "lists"),
    ]

    fv = validate_flowchart(devices, connections)
    t.test("Shared config: flowchart valid", fv.valid)
    t.test("Shared config: no connection errors", len(fv.connection_errors) == 0)
    t.test("Shared config: no multiplicity errors", len(fv.multiplicity_errors) == 0)
    if not fv.valid:
        for e in fv.connection_errors + fv.multiplicity_errors + fv.missing_inputs:
            print(f"    ERROR: {e}")

    # ---------------------------------------------------------------
    t.section("Cycle Detection (Kahn's Algorithm)")
    # ---------------------------------------------------------------
    # GOTCHA #3 resolved: ported from JS sandbox.

    loop_a = DeviceSchema(
        "loop_a",
        "test",
        [
            Port.from_str("in", "metric:*"),
        ],
        [
            Port.from_str("out", "metric:cpk"),
        ],
    )
    loop_b = DeviceSchema(
        "loop_b",
        "test",
        [
            Port.from_str("in", "metric:*"),
        ],
        [
            Port.from_str("out", "metric:ppk"),
        ],
    )
    loop_c = DeviceSchema(
        "loop_c",
        "test",
        [
            Port.from_str("in", "metric:*"),
        ],
        [
            Port.from_str("out", "metric:mean"),
        ],
    )

    cycle_devices = {"a": loop_a, "b": loop_b, "c": loop_c}
    cycle_conns = [
        Connection("a", "out", "b", "in"),
        Connection("b", "out", "c", "in"),
        Connection("c", "out", "a", "in"),
    ]
    fv = validate_flowchart(cycle_devices, cycle_conns)
    t.test("REJECT circular A->B->C->A", not fv.valid)
    cycle_warnings = [w for w in fv.warnings if "CYCLE" in w]
    t.test("Cycle detected in warnings", len(cycle_warnings) > 0)
    if cycle_warnings:
        print(f"    Caught: {cycle_warnings[0]}")

    # Acyclic should pass
    acyclic_conns = [
        Connection("a", "out", "b", "in"),
        Connection("b", "out", "c", "in"),
    ]
    fv2 = validate_flowchart(cycle_devices, acyclic_conns)
    acyclic_cycles = [w for w in fv2.warnings if "CYCLE" in w]
    t.test("Acyclic graph has no cycle warning", len(acyclic_cycles) == 0)

    # -- Summary --
    ok = t.summary()

    print("\n" + "=" * 60)
    print("  DECISIONS CONFIRMED (round 2)")
    print("=" * 60)
    print("""
  1. 8 categories, not 4. All load-bearing. No merging.
  2. Port.multi solves report builder aggregation.
  3. Data source device is the flowchart entry point.
  4. Fishbone/qualitative devices work in the type system.
  5. Fan-out (one output -> N inputs) works naturally.
  6. Flowchart validation catches missing inputs + multiplicity.

  NEW GOTCHAS for production:
  7.  Dynamic ports — get_schema(config) not class-level schema.
  8.  PCL needs semantic_type field (migration, cheap).
  9.  Cycle detection — RESOLVED (Kahn's algorithm, tested).
  10. Config ports — RESOLVED (regular ports, no special flag).
  11. Shared config — RESOLVED (process config device fans out).
  12. Multi-port with zero connections — render behavior TBD.
""")

    return ok


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)

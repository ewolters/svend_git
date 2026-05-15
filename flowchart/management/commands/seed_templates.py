"""Seed starter FlowchartTemplate library.

Creates Quick Cpk, PPAP Package, and Green Belt DMAIC templates with
full port metadata so the JS renderer can draw ports, color-code cables,
and validate connections client-side without a server round-trip.

Idempotent — safe to run multiple times (update_or_create on name).
"""
# @ftb:decision [locked] — Management command, not data migration. Templates are
#   seed data (like fixtures), not schema — they can be re-run, updated, extended.
#   Migration would freeze them at a point in time.
# @ftb:wire [output] — Writes to flowchart_template table. Consumed by:
#   GET /api/flowchart/templates/ (views.py) and future JS renderer.
# @ftb:todo [P0] — data_source, report_builder, fishbone plugins DO NOT EXIST YET.
#   Templates reference them by name. Execution will fail until those device
#   wrappers are written and registered in plugins/apps.py ready().
# @ftb:todo [P1] — Tenant scoping. Seed templates have no tenant_id (system-wide).
#   When multi-tenant ships, decide: are seed templates global or per-tenant?
#   SynaraEntity.tenant_id is nullable, so current behavior is fine for now.

from django.core.management.base import BaseCommand

from flowchart.models import FlowchartTemplate

# @ftb:decision [locked] — PORT_COLORS lives in template definition JSON, not in a
#   separate CSS/JS config. Renderer is self-contained from the definition blob.
#   Trade-off: changing a color means re-running seed command (or migrating JSON).
# @ftb:spec — 8 categories map 1:1 to the semantic type system (types.py).
#   Adding a 9th category here without updating SemanticType.parse() is a bug.
PORT_COLORS = {
    "metric": "#22c55e",
    "spec": "#eab308",
    "config": "#f97316",
    "data": "#3b82f6",
    "chart": "#a855f7",
    "text": "#6b7280",
    "list": "#14b8a6",
    "document": "#ef4444",
}


# ── Device port schemas (reusable across templates) ──────────────────
# @ftb:decision [locked] — Port schemas are self-contained in definition JSON,
#   not derived from Plugin classes. The definition is the snapshot contract —
#   when a user creates a FlowchartInstance, they copy the definition, and that
#   copy IS their configuration frozen at that moment. Deriving from Plugin
#   classes would break instance stability on plugin updates (renamed port =
#   broken saved connections). Plugin.input_schema (Pydantic) is the engine
#   authority; definition JSON is the renderer/audit authority. Sync test (P1)
#   validates they agree at seed time.
# @ftb:todo [P1] — Write a test that loads each template, finds its plugins in
#   the registry, and asserts port names match between template JSON and
#   Plugin.input_schema fields. Catches drift.

# @ftb:todo [P0] — data_source plugin does not exist. When built, it must:
#   1. Parse CSV/pasted data into columns (Plugin.get_schema(config) for dynamic ports)
#   2. Output spec limits from header row or user input
#   3. Expose each column as a separate data:column output port
# @ftb:decision [locked] — Single data_source plugin with universal ports.
#   CSV/paste/PCL are input MODES (configured via config), not separate plugins.
#   Instance copies the definition — unused ports in the snapshot cost nothing,
#   and the user can wire them later in build mode.
# @ftb:decision [locked] — data_source always has ALL ports (including problem_statement).
#   Unused ports just have no cable — they cost nothing in the snapshot. DMAIC wires
#   problem_statement to fishbone; Quick Cpk ignores it. No device variants needed.
DATA_SOURCE_PORTS = {
    "inputs": [],
    "outputs": [
        {"name": "measurements", "type": "data:column"},
        {"name": "usl", "type": "spec:usl"},
        {"name": "lsl", "type": "spec:lsl"},
        {"name": "problem_statement", "type": "text:problem"},
    ],
}

# @ftb:wire [input] — capability_study plugin EXISTS (plugins/capability.py).
#   CapabilityInput has: data (List[float]), usl, lsl, target, measurement.
#   Port names here (data, usl, lsl) must match what engine.py routes as
#   input_data keys, which CapabilityInput validates via Pydantic.
# @ftb:spec — target port intentionally omitted from templates. Target is optional
#   in CapabilityInput and most users don't set it. Can be added via build mode.
CAPABILITY_STUDY_PORTS = {
    "inputs": [
        {"name": "data", "type": "data:column"},
        {"name": "usl", "type": "spec:usl"},
        {"name": "lsl", "type": "spec:lsl"},
    ],
    "outputs": [
        {"name": "cpk", "type": "metric:cpk"},
        {"name": "ppk", "type": "metric:ppk"},
        {"name": "histogram", "type": "chart:histogram"},
        {"name": "summary", "type": "text:summary"},
    ],
}

# @ftb:wire [input] — control_chart plugin EXISTS (plugins/control_chart.py).
# @ftb:spec — subgroup_size is a config port. When NOT connected, renderer shows
#   editable form field with default from config dict (5). When connected (e.g.
#   from a process_config device), field locks to incoming value.
CONTROL_CHART_PORTS = {
    "inputs": [
        {"name": "data", "type": "data:column"},
        {"name": "subgroup_size", "type": "config:subgroup_size"},
    ],
    "outputs": [
        {"name": "chart", "type": "chart:control"},
        {"name": "signals", "type": "list:signals"},
        {"name": "summary", "type": "text:summary"},
    ],
}

# @ftb:todo [P0] — fishbone plugin does not exist. When built:
#   1. Input: problem statement (text). Output: cause list + Ishikawa diagram.
#   2. Existing RCA engine in agents_api/rca_views.py — wrap, don't rebuild.
#   3. Diagram output should be chart:fishbone (ForgeViz ChartSpec, not SVG string).
FISHBONE_PORTS = {
    "inputs": [
        {"name": "problem_statement", "type": "text:problem"},
    ],
    "outputs": [
        {"name": "causes", "type": "list:causes"},
        {"name": "diagram", "type": "chart:fishbone"},
    ],
}

# @ftb:done — report_builder plugin EXISTS (plugins/report_builder.py).
#   Multi-port aggregation DONE (engine.py _build_multi_port_lookup).
# @ftb:todo [P2] — Output format selection (PDF, HTML, PPTX — Dana needs customer-ready)
#   3. Template/layout system for different report styles (PPAP vs DMAIC)
# @ftb:spec — All report_builder input ports use wildcard types (chart:*, metric:*).
#   This means ANY chart subtype connects to the charts port. Validated by
#   SemanticType.accepts() — wildcard matches same category, any subtype.
# @ftb:decision [locked] — lists port is SEPARATE from text port. Fishbone causes
#   are structured (list:causes), not narrative (text:summary). Report builder
#   renders them differently — bullet list vs paragraph.
REPORT_BUILDER_PORTS = {
    "inputs": [
        {"name": "charts", "type": "chart:*", "multi": True},
        {"name": "metrics", "type": "metric:*", "multi": True},
        {"name": "text", "type": "text:*", "multi": True},
        {"name": "lists", "type": "list:*", "multi": True},
    ],
    "outputs": [
        {"name": "report", "type": "document:pdf"},
    ],
}


# ── Template declarations ────────────────────────────────────────────
# @ftb:decision [locked] — TEMPLATES list is the sole data source. Adding a 4th
#   template = append one dict. handle() is generic loop, no template-specific code.
# @ftb:api [contract] — Template definition JSON schema (consumed by renderer):
#   { port_colors: {category: hex}, devices: [{plugin, id, label, ports}],
#     connections: [{source, target, type}], config: {id: {params}},
#     positions: {id: {x, y}} }
# @ftb:todo [P1] — Template versioning. When we update a template definition,
#   existing FlowchartInstances still reference the old definition (they copy it).
#   But the template itself updates. Need to decide: version templates, or accept
#   that instances are snapshots and templates are mutable?
# @ftb:todo [P2] — Template categories/tags for the picker UI. Quick Cpk = "getting
#   started", PPAP = "automotive", DMAIC = "training". Not needed until UI exists.

# ── New device port schemas ─────────────────────────────────────────

VSM_ANALYSIS_PORTS = {
    "inputs": [
        {"name": "steps", "type": "data:vsm_steps"},
    ],
    "outputs": [
        {"name": "lead_time", "type": "metric:lead_time"},
        {"name": "process_time", "type": "metric:process_time"},
        {"name": "pce", "type": "metric:pce"},
        {"name": "total_wip", "type": "metric:wip"},
        {"name": "takt_time", "type": "metric:takt_time"},
        {"name": "summary", "type": "text:vsm_summary"},
    ],
}

SAVINGS_ANALYSIS_PORTS = {
    "inputs": [
        {"name": "baseline", "type": "metric:lead_time"},
        {"name": "actual", "type": "metric:lead_time"},
    ],
    "outputs": [
        {"name": "savings", "type": "metric:dollar_value"},
        {"name": "improvement_pct", "type": "metric:percentage"},
        {"name": "result", "type": "text:savings_summary"},
    ],
}

CONTRACT_ENVELOPE_PORTS = {
    "inputs": [
        {"name": "problem", "type": "text:problem"},
        {"name": "metric_baseline", "type": "metric:lead_time"},
        {"name": "metric_target", "type": "metric:lead_time"},
        {"name": "savings_estimate", "type": "metric:dollar_value"},
    ],
    "outputs": [
        {"name": "contract", "type": "document:improvement_contract"},
        {"name": "action_items", "type": "list:action_items"},
        {"name": "savings_estimate", "type": "metric:dollar_value"},
        {"name": "gap_pct", "type": "metric:percentage"},
    ],
}

CONTRACT_ROUTER_PORTS = {
    "inputs": [
        {"name": "contract", "type": "document:improvement_contract"},
    ],
    "outputs": [
        {"name": "strategic", "type": "list:hoshin_projects"},
        {"name": "tactical", "type": "list:tactical_projects"},
        {"name": "quick_wins", "type": "list:quick_wins"},
        {"name": "total_savings", "type": "metric:dollar_value"},
    ],
}

STRATEGIC_CASCADE_PORTS = {
    "inputs": [
        {"name": "objective", "type": "text:strategic_objective"},
        {"name": "contracts", "type": "list:hoshin_projects", "multi": True},
    ],
    "outputs": [
        {"name": "cascade", "type": "document:hoshin_cascade"},
        {"name": "projects", "type": "list:breakthrough_projects"},
        {"name": "total_savings", "type": "metric:dollar_value"},
    ],
}

# FMEA requires structured rows (process_step, failure_mode, S, O, D).
# This is an entry-point device — user fills in the FMEA table, plugin scores it.
# No input ports: data comes from user at execution time, like data_source.
FMEA_ANALYSIS_PORTS = {
    "inputs": [],
    "outputs": [
        {"name": "rpn_table", "type": "text:rpn_table"},
        {"name": "max_rpn", "type": "metric:rpn"},
        {"name": "high_risk_count", "type": "metric:count"},
        {"name": "summary", "type": "text:fmea_summary"},
    ],
}


# NOTE: text_input output key is DYNAMIC — it equals the configured `subtype`.
# Each template instance using text_input must set the output port name to
# match its subtype config. Use _text_input_ports(subtype) helper below.
def _text_input_ports(subtype: str = "note") -> dict:
    return {
        "inputs": [],
        "outputs": [
            {"name": subtype, "type": f"text:{subtype}"},
        ],
    }


# pcl_source outputs depend on config:
# - history_count=0: outputs {slug} as metric (latest value)
# - history_count>0: outputs {slug}_series as data (list of floats) + {slug} as metric
# Port schemas must match the configured slug names.
def _pcl_source_ports(slugs: list, history: bool = False) -> dict:
    outputs = []
    for slug in slugs:
        if history:
            outputs.append({"name": f"{slug}_series", "type": "data:column"})
        outputs.append({"name": slug, "type": "metric:process_metric"})
    return {"inputs": [], "outputs": outputs}


CONDITIONAL_PORTS = {
    "inputs": [
        {"name": "value", "type": "metric:process_metric"},
    ],
    "outputs": [
        {"name": "result", "type": "metric:bool"},
        {"name": "pass_value", "type": "metric:process_metric"},
        {"name": "fail_value", "type": "metric:process_metric"},
        {"name": "summary", "type": "text:summary"},
    ],
}

DESCRIPTIVE_STATS_PORTS = {
    "inputs": [
        {"name": "data", "type": "data:column"},
    ],
    "outputs": [
        {"name": "mean", "type": "metric:mean"},
        {"name": "std", "type": "metric:std"},
        {"name": "median", "type": "metric:median"},
        {"name": "n", "type": "metric:count"},
        {"name": "result", "type": "text:summary"},
    ],
}

CORRELATION_PORTS = {
    "inputs": [
        {"name": "data", "type": "data:multivariate"},
    ],
    "outputs": [
        {"name": "r", "type": "metric:correlation"},
        {"name": "p_value", "type": "metric:p_value"},
        {"name": "result", "type": "text:summary"},
    ],
}

HYPOTHESIS_TEST_PORTS = {
    "inputs": [
        {"name": "data", "type": "data:column"},
    ],
    "outputs": [
        {"name": "p_value", "type": "metric:p_value"},
        {"name": "statistic", "type": "metric:test_statistic"},
        {"name": "result", "type": "text:summary"},
    ],
}


# ── Template declarations ────────────────────────────────────────────

TEMPLATES = [
    {
        "name": "Quick Cpk",
        "description": (
            "Simplest flowchart: paste measurement data and specification "
            "limits, get Cpk/Ppk with histogram. Two devices, three cables."
        ),
        "devices_used": ["data_source", "capability_study"],
        "definition": {
            "port_colors": PORT_COLORS,
            "devices": [
                {"plugin": "data_source", "id": "ds1", "label": "Data Source", "ports": DATA_SOURCE_PORTS},
                {
                    "plugin": "capability_study",
                    "id": "cap1",
                    "label": "Capability Study",
                    "ports": CAPABILITY_STUDY_PORTS,
                },
            ],
            "connections": [
                {"source": "ds1.measurements", "target": "cap1.data", "type": "data:column"},
                {"source": "ds1.usl", "target": "cap1.usl", "type": "spec:usl"},
                {"source": "ds1.lsl", "target": "cap1.lsl", "type": "spec:lsl"},
            ],
            "config": {
                "ds1": {},
                "cap1": {},
            },
            "positions": {
                "ds1": {"x": 100, "y": 200},
                "cap1": {"x": 400, "y": 200},
            },
        },
    },
    # @ftb:spec — PPAP is Dana's use case. Must produce customer-ready output.
    #   Ford wants one Cpk format, GM wants another. Report builder needs
    #   output format templates (future: config:output_format port on rpt1).
    # @ftb:perf — 4 devices, 10 connections. Execution is sequential (topo sort).
    #   ds1 → [cap1, cc1] → rpt1. Two middle devices are independent — could
    #   parallelize in engine.py (same topo level). Not needed yet.
    {
        "name": "PPAP Package",
        "description": (
            "Production Part Approval Process package. Capability study and "
            "control chart run in parallel, then converge into a report. "
            "Dana: 'If the Cpk I ran on Tuesday lives somewhere and my "
            "control plan knows about it.'"
        ),
        "devices_used": ["data_source", "capability_study", "control_chart", "report_builder"],
        "definition": {
            "port_colors": PORT_COLORS,
            "devices": [
                {"plugin": "data_source", "id": "ds1", "label": "Data Source", "ports": DATA_SOURCE_PORTS},
                {
                    "plugin": "capability_study",
                    "id": "cap1",
                    "label": "Capability Study",
                    "ports": CAPABILITY_STUDY_PORTS,
                },
                {"plugin": "control_chart", "id": "cc1", "label": "Control Chart", "ports": CONTROL_CHART_PORTS},
                {"plugin": "report_builder", "id": "rpt1", "label": "PPAP Report", "ports": REPORT_BUILDER_PORTS},
            ],
            "connections": [
                # data_source → parallel analyses
                {"source": "ds1.measurements", "target": "cap1.data", "type": "data:column"},
                {"source": "ds1.usl", "target": "cap1.usl", "type": "spec:usl"},
                {"source": "ds1.lsl", "target": "cap1.lsl", "type": "spec:lsl"},
                {"source": "ds1.measurements", "target": "cc1.data", "type": "data:column"},
                # analyses → report
                {"source": "cap1.histogram", "target": "rpt1.charts", "type": "chart:histogram"},
                {"source": "cap1.cpk", "target": "rpt1.metrics", "type": "metric:cpk"},
                {"source": "cap1.ppk", "target": "rpt1.metrics", "type": "metric:ppk"},
                {"source": "cap1.summary", "target": "rpt1.text", "type": "text:summary"},
                {"source": "cc1.chart", "target": "rpt1.charts", "type": "chart:control"},
                {"source": "cc1.summary", "target": "rpt1.text", "type": "text:summary"},
            ],
            "config": {
                "ds1": {},
                "cap1": {},
                "cc1": {"subgroup_size": 5},
                "rpt1": {"title": "PPAP Capability Package"},
            },
            "positions": {
                "ds1": {"x": 100, "y": 250},
                "cap1": {"x": 400, "y": 120},
                "cc1": {"x": 400, "y": 380},
                "rpt1": {"x": 700, "y": 250},
            },
        },
    },
    # @ftb:spec — DMAIC is Carmen's use case. 50-70 students/yr load this template.
    #   Must be shareable (is_shared=True) and clonable (FlowchartTemplate.clone()).
    #   Students modify their copy, Carmen's original stays intact.
    # @ftb:wire [output] — When templates are shared, they appear in all users'
    #   template picker. Future: sharing scopes (org, public, link-based).
    # @ftb:todo [P2] — "Green Belt" is ILSSI branding. May need to parameterize
    #   template names per training organization (Carmen vs other trainers).
    {
        "name": "Green Belt DMAIC",
        "description": (
            "Full DMAIC pipeline for Green Belt training. Capability study, "
            "control chart, and fishbone analysis run in parallel, all "
            "converging into a report. Carmen teaches 50-70 students/yr — "
            "needs shareable, repeatable project templates."
        ),
        "devices_used": ["data_source", "capability_study", "control_chart", "fishbone", "report_builder"],
        "definition": {
            "port_colors": PORT_COLORS,
            "devices": [
                {"plugin": "data_source", "id": "ds1", "label": "Data Source", "ports": DATA_SOURCE_PORTS},
                {
                    "plugin": "capability_study",
                    "id": "cap1",
                    "label": "Capability Study",
                    "ports": CAPABILITY_STUDY_PORTS,
                },
                {"plugin": "control_chart", "id": "cc1", "label": "Control Chart", "ports": CONTROL_CHART_PORTS},
                {"plugin": "fishbone", "id": "fb1", "label": "Fishbone Diagram", "ports": FISHBONE_PORTS},
                {"plugin": "report_builder", "id": "rpt1", "label": "DMAIC Report", "ports": REPORT_BUILDER_PORTS},
            ],
            "connections": [
                # data_source → parallel analyses
                {"source": "ds1.measurements", "target": "cap1.data", "type": "data:column"},
                {"source": "ds1.usl", "target": "cap1.usl", "type": "spec:usl"},
                {"source": "ds1.lsl", "target": "cap1.lsl", "type": "spec:lsl"},
                {"source": "ds1.measurements", "target": "cc1.data", "type": "data:column"},
                {"source": "ds1.problem_statement", "target": "fb1.problem_statement", "type": "text:problem"},
                # analyses → report
                {"source": "cap1.histogram", "target": "rpt1.charts", "type": "chart:histogram"},
                {"source": "cap1.cpk", "target": "rpt1.metrics", "type": "metric:cpk"},
                {"source": "cap1.ppk", "target": "rpt1.metrics", "type": "metric:ppk"},
                {"source": "cap1.summary", "target": "rpt1.text", "type": "text:summary"},
                {"source": "cc1.chart", "target": "rpt1.charts", "type": "chart:control"},
                {"source": "cc1.summary", "target": "rpt1.text", "type": "text:summary"},
                {"source": "fb1.diagram", "target": "rpt1.charts", "type": "chart:fishbone"},
                {"source": "fb1.causes", "target": "rpt1.lists", "type": "list:causes"},
            ],
            "config": {
                "ds1": {},
                "cap1": {},
                "cc1": {"subgroup_size": 5},
                "fb1": {},
                "rpt1": {"title": "Green Belt DMAIC Report"},
            },
            "positions": {
                "ds1": {"x": 100, "y": 250},
                "cap1": {"x": 400, "y": 80},
                "cc1": {"x": 400, "y": 250},
                "fb1": {"x": 400, "y": 420},
                "rpt1": {"x": 700, "y": 250},
            },
        },
    },
    # ── VSM Improvement Cycle ───────────────────────────────────────
    # VSM current/future → savings → contract → route → Hoshin cascade.
    # The full methodology loop from value stream analysis to strategic deployment.
    {
        "name": "VSM Improvement Cycle",
        "description": (
            "Value stream current/future state analysis through to Hoshin "
            "breakthrough projects. Maps the gap, calculates savings, packages "
            "as improvement contracts, routes strategic items to Hoshin cascade."
        ),
        "devices_used": [
            "text_input",
            "vsm_analysis",
            "savings_analysis",
            "contract_envelope",
            "contract_router",
            "strategic_cascade",
        ],
        "definition": {
            "port_colors": PORT_COLORS,
            "devices": [
                {
                    "plugin": "text_input",
                    "id": "obj",
                    "label": "Strategic Objective",
                    "ports": _text_input_ports("strategic_objective"),
                },
                {"plugin": "vsm_analysis", "id": "vsm_cur", "label": "VSM Current State", "ports": VSM_ANALYSIS_PORTS},
                {"plugin": "vsm_analysis", "id": "vsm_fut", "label": "VSM Future State", "ports": VSM_ANALYSIS_PORTS},
                {
                    "plugin": "savings_analysis",
                    "id": "fpa",
                    "label": "Savings Analysis",
                    "ports": SAVINGS_ANALYSIS_PORTS,
                },
                {
                    "plugin": "contract_envelope",
                    "id": "contract",
                    "label": "Improvement Contract",
                    "ports": CONTRACT_ENVELOPE_PORTS,
                },
                {
                    "plugin": "contract_router",
                    "id": "router",
                    "label": "Route to Tracker",
                    "ports": CONTRACT_ROUTER_PORTS,
                },
                {
                    "plugin": "strategic_cascade",
                    "id": "cascade",
                    "label": "Hoshin Cascade",
                    "ports": STRATEGIC_CASCADE_PORTS,
                },
            ],
            "connections": [
                {"source": "vsm_cur.lead_time", "target": "fpa.baseline", "type": "metric:lead_time"},
                {"source": "vsm_fut.lead_time", "target": "fpa.actual", "type": "metric:lead_time"},
                {"source": "vsm_cur.lead_time", "target": "contract.metric_baseline", "type": "metric:lead_time"},
                {"source": "vsm_fut.lead_time", "target": "contract.metric_target", "type": "metric:lead_time"},
                {"source": "fpa.savings", "target": "contract.savings_estimate", "type": "metric:dollar_value"},
                {"source": "contract.contract", "target": "router.contract", "type": "document:improvement_contract"},
                {
                    "source": "obj.strategic_objective",
                    "target": "cascade.objective",
                    "type": "text:strategic_objective",
                },
                {"source": "router.strategic", "target": "cascade.contracts", "type": "list:hoshin_projects"},
            ],
            "config": {
                "obj": {"content": "Reduce manufacturing lead time 40% by Q4", "subtype": "strategic_objective"},
                "vsm_cur": {},
                "vsm_fut": {},
                "fpa": {"analysis_type": "savings", "method": "time_reduction", "volume": 1, "cost_per_unit": 1},
                "contract": {
                    "metric_name": "lead_time_days",
                    "priority": "strategic",
                    "source_type": "vsm",
                    "timeline_months": 12,
                },
                "router": {},
                "cascade": {
                    "objective": "",
                    "target_metric": "lead_time_days",
                    "baseline_value": 85,
                    "target_value": 51,
                    "timeline_months": 12,
                },
            },
            "positions": {
                "obj": {"x": 50, "y": 50},
                "vsm_cur": {"x": 50, "y": 200},
                "vsm_fut": {"x": 50, "y": 350},
                "fpa": {"x": 300, "y": 275},
                "contract": {"x": 500, "y": 275},
                "router": {"x": 700, "y": 275},
                "cascade": {"x": 700, "y": 50},
            },
        },
    },
    # ── FMEA Risk Assessment ────────────────────────────────────────
    # FMEA is an entry point (user fills S/O/D table). Summary + max RPN
    # flow into contract envelope, which routes by priority.
    {
        "name": "FMEA Risk Assessment",
        "description": (
            "Failure Mode and Effects Analysis with automatic risk-based "
            "routing. Fill in the FMEA table — high-RPN items become contracts "
            "routed to Hoshin (strategic), project tracker (tactical), or "
            "quick-win Kanban."
        ),
        "devices_used": ["fmea_analysis", "contract_envelope", "contract_router"],
        "definition": {
            "port_colors": PORT_COLORS,
            "devices": [
                {"plugin": "fmea_analysis", "id": "fmea", "label": "FMEA Analysis", "ports": FMEA_ANALYSIS_PORTS},
                {
                    "plugin": "contract_envelope",
                    "id": "contract",
                    "label": "Risk Contract",
                    "ports": CONTRACT_ENVELOPE_PORTS,
                },
                {"plugin": "contract_router", "id": "router", "label": "Route by Risk", "ports": CONTRACT_ROUTER_PORTS},
            ],
            "connections": [
                {"source": "fmea.summary", "target": "contract.problem", "type": "text:fmea_summary"},
                {"source": "fmea.max_rpn", "target": "contract.metric_baseline", "type": "metric:rpn"},
                {"source": "contract.contract", "target": "router.contract", "type": "document:improvement_contract"},
            ],
            "config": {
                "fmea": {},
                "contract": {"metric_name": "rpn", "source_type": "fmea", "priority": "tactical"},
                "router": {},
            },
            "positions": {
                "fmea": {"x": 50, "y": 150},
                "contract": {"x": 350, "y": 150},
                "router": {"x": 650, "y": 150},
            },
        },
    },
    # ── Process Monitoring ──────────────────────────────────────────
    # PCL (history mode) → control chart + conditional gate → contract if out of spec.
    # pcl_source with history_count=50 outputs {slug}_series (List[float]) for cc
    # and {slug} (latest metric) for the conditional gate.
    {
        "name": "Process Monitoring",
        "description": (
            "Continuous process monitoring. Pulls historical measures from PCL, "
            "runs control chart on the series, gates latest value on specification. "
            "Out-of-spec conditions automatically generate improvement contracts."
        ),
        "devices_used": ["pcl_source", "control_chart", "conditional", "contract_envelope"],
        "definition": {
            "port_colors": PORT_COLORS,
            "devices": [
                {
                    "plugin": "pcl_source",
                    "id": "pcl",
                    "label": "Process Measures",
                    "ports": _pcl_source_ports(["critical_dimension_1"], history=True),
                },
                {"plugin": "control_chart", "id": "cc", "label": "Control Chart", "ports": CONTROL_CHART_PORTS},
                {"plugin": "conditional", "id": "gate", "label": "In Spec?", "ports": CONDITIONAL_PORTS},
                {
                    "plugin": "contract_envelope",
                    "id": "contract",
                    "label": "Out-of-Spec Contract",
                    "ports": CONTRACT_ENVELOPE_PORTS,
                },
            ],
            "connections": [
                # Series (list) → control chart; latest value → conditional gate
                {"source": "pcl.critical_dimension_1_series", "target": "cc.data", "type": "data:column"},
                {"source": "pcl.critical_dimension_1", "target": "gate.value", "type": "metric:process_metric"},
                {"source": "gate.fail_value", "target": "contract.metric_baseline", "type": "metric:process_metric"},
            ],
            "config": {
                "pcl": {"measure_slugs": ["critical_dimension_1"], "history_count": 50},
                "cc": {"subgroup_size": 1},
                "gate": {"operator": ">=", "threshold": 1.33, "label": "Cpk threshold"},
                "contract": {
                    "metric_name": "cpk",
                    "source_type": "capability",
                    "priority": "tactical",
                    "timeline_months": 3,
                },
            },
            "positions": {
                "pcl": {"x": 50, "y": 150},
                "cc": {"x": 300, "y": 80},
                "gate": {"x": 300, "y": 250},
                "contract": {"x": 550, "y": 250},
            },
        },
    },
    # ── Data Exploration ────────────────────────────────────────────
    # The "what does my data look like?" template. EDA first, then decide.
    {
        "name": "Data Exploration",
        "description": (
            "Exploratory data analysis: descriptive statistics, correlation, "
            "and hypothesis testing in parallel. The 'explore before you decide' "
            "template for any new dataset."
        ),
        "devices_used": ["data_source", "descriptive_stats", "correlation", "hypothesis_test", "report_builder"],
        "definition": {
            "port_colors": PORT_COLORS,
            "devices": [
                {"plugin": "data_source", "id": "ds", "label": "Data Source", "ports": DATA_SOURCE_PORTS},
                {
                    "plugin": "descriptive_stats",
                    "id": "desc",
                    "label": "Descriptive Stats",
                    "ports": DESCRIPTIVE_STATS_PORTS,
                },
                {"plugin": "hypothesis_test", "id": "hyp", "label": "Hypothesis Test", "ports": HYPOTHESIS_TEST_PORTS},
                {"plugin": "report_builder", "id": "rpt", "label": "EDA Report", "ports": REPORT_BUILDER_PORTS},
            ],
            "connections": [
                {"source": "ds.measurements", "target": "desc.data", "type": "data:column"},
                {"source": "ds.measurements", "target": "hyp.data", "type": "data:column"},
                {"source": "desc.mean", "target": "rpt.metrics", "type": "metric:mean"},
                {"source": "desc.std", "target": "rpt.metrics", "type": "metric:std"},
                {"source": "desc.result", "target": "rpt.text", "type": "text:summary"},
                {"source": "hyp.p_value", "target": "rpt.metrics", "type": "metric:p_value"},
                {"source": "hyp.result", "target": "rpt.text", "type": "text:summary"},
            ],
            "config": {
                "ds": {},
                "desc": {},
                "hyp": {"test_type": "one_sample_t", "mu": 0.0},
                "rpt": {"title": "Exploratory Data Analysis"},
            },
            "positions": {
                "ds": {"x": 50, "y": 200},
                "desc": {"x": 350, "y": 100},
                "hyp": {"x": 350, "y": 300},
                "rpt": {"x": 650, "y": 200},
            },
        },
    },
]


class Command(BaseCommand):
    help = "Seed starter FlowchartTemplate library (idempotent)."

    # @ftb:spec — Idempotent via update_or_create keyed on name. Running twice
    #   updates definitions (picks up port schema changes) without creating dupes.
    # @ftb:todo [P1] — Add --dry-run flag to show what WOULD change without writing DB.
    # @ftb:todo [P2] — Add --validate flag to check all referenced plugins exist
    #   in the registry before writing. Currently writes templates even if plugins
    #   are missing — templates are data, not executable until flowchart runs.

    def handle(self, *args, **options):
        created_count = 0
        updated_count = 0

        for spec in TEMPLATES:
            _, created = FlowchartTemplate.objects.update_or_create(
                name=spec["name"],
                defaults={
                    "description": spec["description"],
                    "definition": spec["definition"],
                    "devices_used": spec["devices_used"],
                    "is_shared": True,
                },
            )
            verb = "Created" if created else "Updated"
            created_count += created
            updated_count += not created
            self.stdout.write(self.style.SUCCESS(f"  {verb}: {spec['name']}"))

        self.stdout.write(self.style.SUCCESS(f"\nDone. {created_count} created, {updated_count} updated."))

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

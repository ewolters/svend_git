"""
Graph models — GRAPH-001 §3, §4, §13.

ProcessGraph: org-wide process knowledge container (one per tenant, federated via parent_graph).
ProcessNode: process entities — things you can measure, control, observe, or specify.
ProcessEdge: claimed relationships with Bayesian posteriors from evidence stacking.
EdgeEvidence: timestamped, immutable evidence records on edges.
"""

import uuid

from django.conf import settings
from django.db import models

# =============================================================================
# ProcessGraph — GRAPH-001 §13.1
# =============================================================================


class ProcessGraph(models.Model):
    """Org-wide process knowledge container.

    One primary graph per tenant. Federated: subgraphs reference a parent
    via parent_graph (D4 decision). process_area tags enable filtering
    (e.g., "safety", "assembly", "molding").
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        related_name="process_graphs",
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")

    # Federation (D4): subgraphs point to parent
    parent_graph = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="subgraphs",
    )
    # Filtering tag (D4): "safety", "assembly", "molding", etc.
    process_area = models.CharField(max_length=100, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_graphs",
    )

    class Meta:
        db_table = "graph_process_graph"
        indexes = [
            models.Index(fields=["tenant"]),
            models.Index(fields=["tenant", "process_area"]),
        ]

    def __str__(self):
        area = f" [{self.process_area}]" if self.process_area else ""
        return f"{self.name}{area} ({self.tenant})"

    @property
    def node_count(self) -> int:
        return self.nodes.count()

    @property
    def edge_count(self) -> int:
        return self.edges.count()


# =============================================================================
# ProcessNode — GRAPH-001 §3
# =============================================================================


class ProcessNode(models.Model):
    """A discrete, identifiable entity in the process.

    Something you can measure, control, observe, or specify.
    The type taxonomy is extensible — GRAPH-001 §3.2.
    """

    class NodeType(models.TextChoices):
        PROCESS_PARAMETER = "process_parameter", "Process Parameter"
        QUALITY_CHARACTERISTIC = "quality_characteristic", "Quality Characteristic"
        FAILURE_MODE = "failure_mode", "Failure Mode"
        ENVIRONMENTAL_FACTOR = "environmental_factor", "Environmental Factor"
        MATERIAL_PROPERTY = "material_property", "Material Property"
        MEASUREMENT = "measurement", "Measurement"
        SPECIFICATION = "specification", "Specification"
        EQUIPMENT = "equipment", "Equipment"
        HUMAN_FACTOR = "human_factor", "Human Factor"
        CUSTOM = "custom", "Custom"

    class Controllability(models.TextChoices):
        DIRECT = "direct", "Directly Controllable"
        INDIRECT = "indirect", "Indirectly Controllable"
        NOISE = "noise", "Noise (uncontrollable)"
        FIXED = "fixed", "Fixed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    graph = models.ForeignKey(
        ProcessGraph,
        on_delete=models.CASCADE,
        related_name="nodes",
    )

    # Identity
    name = models.CharField(max_length=255)
    node_type = models.CharField(
        max_length=30,
        choices=NodeType.choices,
        default=NodeType.PROCESS_PARAMETER,
    )
    description = models.TextField(blank=True, default="")
    custom_type = models.CharField(max_length=100, blank=True, default="")

    # Operating state (nullable — populated as data arrives)
    unit = models.CharField(max_length=50, blank=True, default="")
    distribution = models.JSONField(
        null=True,
        blank=True,
        help_text="Current distribution: {mean, std, shape, n, source, as_of}",
    )
    spec_limits = models.JSONField(
        null=True,
        blank=True,
        help_text="Specification limits: {usl, lsl, target}",
    )
    control_limits = models.JSONField(
        null=True,
        blank=True,
        help_text="Control limits: {ucl, lcl, cl}",
    )

    # Controllability
    controllability = models.CharField(
        max_length=10,
        choices=Controllability.choices,
        blank=True,
        default="",
    )
    control_method = models.CharField(max_length=255, blank=True, default="")

    # Federation (D4): shared nodes can appear in multiple graphs
    shared = models.BooleanField(
        default=False,
        help_text="If True, this node can be referenced by edges in sibling graphs",
    )

    # Linkage (UUIDs stored as JSON — loose coupling to other apps)
    linked_fmis_rows = models.JSONField(
        default=list,
        blank=True,
        help_text="UUIDs of FMISRow records where this node appears",
    )
    linked_equipment = models.JSONField(
        default=list,
        blank=True,
        help_text="UUIDs of MeasurementEquipment records",
    )
    linked_spc_chart = models.UUIDField(
        null=True,
        blank=True,
        help_text="Active SPC chart monitoring this node",
    )

    # Provenance
    provenance = models.CharField(
        max_length=30,
        blank=True,
        default="manual",
        help_text="How this node was created: fmea_seed, investigation, manual, spc",
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_process_nodes",
    )

    class Meta:
        db_table = "graph_process_node"
        indexes = [
            models.Index(fields=["graph", "node_type"]),
            models.Index(fields=["graph", "name"]),
        ]

    def __str__(self):
        return f"{self.name} ({self.get_node_type_display()})"


# =============================================================================
# ProcessEdge — GRAPH-001 §4
# =============================================================================


class ProcessEdge(models.Model):
    """A claimed relationship between two nodes.

    Every edge is a Bayesian belief: "we believe X influences Y with
    this strength." Evidence from DOE, investigation, SPC, etc. updates
    the posterior via Synara.
    """

    class RelationType(models.TextChoices):
        CAUSAL = "causal", "Causal"
        CORRELATIONAL = "correlational", "Correlational"
        CONFOUNDED = "confounded", "Confounded"
        SPECIFICATION = "specification", "Specification"
        MEASUREMENT = "measurement", "Measurement"

    class Direction(models.TextChoices):
        POSITIVE = "positive", "Positive"
        NEGATIVE = "negative", "Negative"
        NONLINEAR = "nonlinear", "Nonlinear"
        UNKNOWN = "unknown", "Unknown"

    class Linearity(models.TextChoices):
        LINEAR = "linear", "Linear"
        NONLINEAR = "nonlinear", "Nonlinear"
        THRESHOLD = "threshold", "Threshold"
        UNKNOWN = "unknown", "Unknown"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    graph = models.ForeignKey(
        ProcessGraph,
        on_delete=models.CASCADE,
        related_name="edges",
    )

    # The edge: source → target
    source = models.ForeignKey(
        ProcessNode,
        on_delete=models.CASCADE,
        related_name="outgoing_edges",
    )
    target = models.ForeignKey(
        ProcessNode,
        on_delete=models.CASCADE,
        related_name="incoming_edges",
    )
    relation_type = models.CharField(
        max_length=20,
        choices=RelationType.choices,
        default=RelationType.CAUSAL,
    )

    # ── MODEL LAYER (Bayesian belief state) ──

    effect_size = models.FloatField(null=True, blank=True)
    effect_ci_lower = models.FloatField(null=True, blank=True)
    effect_ci_upper = models.FloatField(null=True, blank=True)
    posterior_strength = models.FloatField(
        default=0.5,
        help_text="Synara posterior — belief this relationship exists (0.0-1.0)",
    )

    direction = models.CharField(
        max_length=10,
        choices=Direction.choices,
        default=Direction.UNKNOWN,
    )
    linearity = models.CharField(
        max_length=10,
        choices=Linearity.choices,
        default=Linearity.UNKNOWN,
    )

    # Operating region (edge metadata — §5)
    operating_region = models.JSONField(
        null=True,
        blank=True,
        help_text='Conditions under which this edge manifests: {"humidity": {">": 60}}',
    )

    # ── EVIDENCE LAYER ──

    evidence_count = models.IntegerField(default=0)

    # ── PROVENANCE ──

    provenance = models.CharField(
        max_length=30,
        blank=True,
        default="manual",
        help_text="How created: fmea_assertion, doe, investigation, operator, spc, literature",
    )
    source_investigation = models.UUIDField(
        null=True,
        blank=True,
        help_text="Investigation that created/last calibrated this edge",
    )
    calibration_date = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When this edge was last calibrated with empirical data",
    )

    # ── INTERACTION TERMS (§5) ──

    interaction_terms = models.JSONField(
        default=list,
        blank=True,
        help_text="Interaction terms: [{modulating_node, modulation_type, fit_result, calibrated}]",
    )

    # ── HEALTH ──

    is_stale = models.BooleanField(default=False)
    staleness_reason = models.CharField(max_length=100, blank=True, default="")
    is_contradicted = models.BooleanField(default=False)
    contradiction_signal_id = models.UUIDField(
        null=True,
        blank=True,
        help_text="LOOP-001 Signal raised for this contradiction",
    )

    # ── METADATA ──

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_process_edges",
    )

    class Meta:
        db_table = "graph_process_edge"
        indexes = [
            models.Index(fields=["graph", "relation_type"]),
            models.Index(fields=["source", "relation_type"]),
            models.Index(fields=["target", "relation_type"]),
            models.Index(fields=["graph", "is_stale"]),
            models.Index(fields=["graph", "is_contradicted"]),
        ]

    def __str__(self):
        return f"{self.source.name} → {self.target.name} ({self.relation_type})"

    @property
    def is_calibrated(self) -> bool:
        return self.evidence_count > 0 and self.provenance != "fmea_assertion"


# =============================================================================
# EdgeEvidence — GRAPH-001 §4.4
# =============================================================================


class EdgeEvidence(models.Model):
    """A timestamped, immutable evidence record on an edge.

    Evidence is never deleted (GRAPH-001 §16.3). Erroneous entries are
    retracted (excluded from posterior computation but visible in audit trail).
    """

    class SourceType(models.TextChoices):
        DOE = "doe", "Design of Experiments"
        INVESTIGATION = "investigation", "Investigation"
        SPC = "spc", "Statistical Process Control"
        PROCESS_CONFIRMATION = "process_confirmation", "Process Confirmation"
        FORCED_FAILURE_TEST = "forced_failure_test", "Forced Failure Test"
        GAGE_RR = "gage_rr", "Gage R&R"
        OPERATOR = "operator", "Operator Observation"
        LITERATURE = "literature", "Literature / Reference"
        SAFETY = "safety", "Safety Observation"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    edge = models.ForeignKey(
        ProcessEdge,
        on_delete=models.CASCADE,
        related_name="evidence_stack",
    )

    # What was observed
    effect_size = models.FloatField(null=True, blank=True)
    confidence_interval = models.JSONField(
        null=True,
        blank=True,
        help_text="CI: {lower, upper}",
    )
    sample_size = models.IntegerField(null=True, blank=True)
    p_value = models.FloatField(null=True, blank=True)

    # Source
    source_type = models.CharField(
        max_length=30,
        choices=SourceType.choices,
    )
    source_id = models.UUIDField(
        null=True,
        blank=True,
        help_text="FK to source object (Investigation, DOE, etc.)",
    )
    source_description = models.TextField(
        blank=True,
        default="",
        help_text="Human-readable description of what was observed",
    )

    # Reliability
    strength = models.FloatField(
        default=1.0,
        help_text="Measurement reliability / study quality (0.0-1.0)",
    )

    # Temporal
    observed_at = models.DateTimeField(
        help_text="When the evidence was generated",
    )

    # Retraction (O1 decision): immutable except for this flag
    retracted = models.BooleanField(default=False)
    retracted_reason = models.TextField(blank=True, default="")

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_edge_evidence",
    )

    class Meta:
        db_table = "graph_edge_evidence"
        ordering = ["-observed_at"]
        indexes = [
            models.Index(fields=["edge", "-observed_at"]),
            models.Index(fields=["edge", "source_type"]),
        ]

    def __str__(self):
        status = " [RETRACTED]" if self.retracted else ""
        return f"Evidence on {self.edge_id} ({self.source_type}){status}"

"""
PROVA — Predictive Reasoning Over Validated Alternatives

The problem-solving engine for SVEND. Maintains operating/working knowledge
graphs, manages trials as the only mechanism for graph promotion, detects
conflicts, and propagates signals through causal relationships.

Built on forgesia (graph math, Bayesian updates) and forgestat (likelihoods).
PROVA is the service layer; forgesia is the computation layer.
"""

from django.conf import settings
from django.db import models

from syn.core.base_models import SynaraEntity

# =============================================================================
# ENUMS
# =============================================================================


class EdgeType(models.TextChoices):
    CAUSES = "causes", "Causes"
    INHIBITS = "inhibits", "Inhibits"
    AMPLIFIES = "amplifies", "Amplifies"
    REQUIRES = "requires", "Requires"
    SUPPORTS = "supports", "Supports"
    CONTRADICTS = "contradicts", "Contradicts"
    PRECEDES = "precedes", "Precedes"


class EdgeStatus(models.TextChoices):
    ACTIVE = "active", "Active"
    BROKEN = "broken", "Broken (conflict)"
    DARK = "dark", "Dark (premise inactive)"


class NodeType(models.TextChoices):
    FACTOR = "factor", "Factor"
    OUTCOME = "outcome", "Outcome"
    PROCESS = "process", "Process"
    CONDITION = "condition", "Condition"
    EQUIPMENT = "equipment", "Equipment"
    MATERIAL = "material", "Material"
    METHOD = "method", "Method"
    ENVIRONMENT = "environment", "Environment"
    MEASUREMENT = "measurement", "Measurement"
    PERSONNEL = "personnel", "Personnel"


class EditOperation(models.TextChoices):
    CHALLENGE_EDGE = "challenge_edge", "Challenge edge"
    ADD_CONDITION = "add_condition", "Add condition"
    ADD_EDGE = "add_edge", "Add edge"
    REMOVE_EDGE = "remove_edge", "Remove edge"
    ADD_NODE = "add_node", "Add node"
    REPLACE_NODE = "replace_node", "Replace node"
    MODIFY_STRENGTH = "modify_strength", "Modify strength"


class HypothesisStatus(models.TextChoices):
    PROPOSED = "proposed", "Proposed"
    TESTING = "testing", "Testing"
    CONFIRMED = "confirmed", "Confirmed"
    REJECTED = "rejected", "Rejected"
    MERGED = "merged", "Merged into operating"


class TrialStatus(models.TextChoices):
    PLANNED = "planned", "Planned"
    DRAFT = "draft", "Draft (data entry)"
    RUNNING = "running", "Running"
    COMPLETED = "completed", "Completed"
    INVALID = "invalid", "Invalid"


class ComplexityTier(models.TextChoices):
    GREEN = "green", "Green — no stats needed"
    BLUE = "blue", "Blue — basic comparison"
    PURPLE = "purple", "Purple — advanced design"


class ConflictStatus(models.TextChoices):
    OPEN = "open", "Open"
    INVESTIGATING = "investigating", "Investigating"
    RESOLVED = "resolved", "Resolved"
    ACCEPTED = "accepted", "Accepted (known uncertainty)"


class SignalType(models.TextChoices):
    PREMISE_DARK = "premise_dark", "Premise goes dark"
    DENYING_CONSEQUENT = "denying_consequent", "Denying the consequent"


class SignalStatus(models.TextChoices):
    ACTIVE = "active", "Active"
    ACKNOWLEDGED = "acknowledged", "Acknowledged"
    RESOLVED = "resolved", "Resolved"


# =============================================================================
# OPERATING GRAPH — versioned truth, one live instance per tenant
# =============================================================================


class OperatingGraph(SynaraEntity):
    """
    The site-level knowledge graph. ONE live set of nodes/edges per tenant.
    Versions are frozen JSON snapshots for rollback, not duplicate rows.
    """

    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        related_name="operating_graphs",
    )
    current_version = models.PositiveIntegerField(
        default=0,
        help_text="Increments on each trial promotion.",
    )
    predictive_score = models.FloatField(
        default=0.0,
        help_text="Overall model quality score (0-1).",
    )
    last_evaluated = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Last time the full graph was evaluated against reality.",
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["tenant"],
                condition=models.Q(is_deleted=False),
                name="unique_active_operating_graph_per_tenant",
            ),
        ]

    def __str__(self):
        return f"OperatingGraph v{self.current_version} ({self.tenant})"


class GraphVersion(SynaraEntity):
    """
    Immutable snapshot of the operating graph at a point in time.
    Created before each promotion so the prior state is preserved.
    """

    operating_graph = models.ForeignKey(
        OperatingGraph,
        on_delete=models.CASCADE,
        related_name="versions",
    )
    version_number = models.PositiveIntegerField()
    snapshot = models.JSONField(
        help_text="Frozen forgesia state (nodes, edges, params) at this version.",
    )
    promoted_by_trial = models.ForeignKey(
        "prova.Trial",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="promoted_versions",
        help_text="The trial that caused this version to be created.",
    )
    parent_version = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="children",
    )
    notes = models.TextField(blank=True, default="")

    class Meta:
        ordering = ["-version_number"]
        constraints = [
            models.UniqueConstraint(
                fields=["operating_graph", "version_number"],
                name="unique_version_per_graph",
            ),
        ]

    def __str__(self):
        return f"v{self.version_number} of {self.operating_graph}"


# =============================================================================
# GRAPH STRUCTURE — live nodes and edges
# =============================================================================


class GraphNode(SynaraEntity):
    """
    A factor, outcome, process, or condition in the operating graph.
    Beta-distributed confidence: alpha/(alpha+beta) = expected confidence.
    """

    operating_graph = models.ForeignKey(
        OperatingGraph,
        on_delete=models.CASCADE,
        related_name="nodes",
    )
    label = models.CharField(max_length=255)
    node_type = models.CharField(
        max_length=20,
        choices=NodeType.choices,
        default=NodeType.FACTOR,
    )
    entity = models.ForeignKey(
        "core.Entity",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="prova_nodes",
        help_text="Optional bridge to core.Entity for compatibility.",
    )
    alpha = models.FloatField(
        default=1.0,
        help_text="Beta prior success count.",
    )
    beta = models.FloatField(
        default=1.0,
        help_text="Beta prior failure count.",
    )

    class Meta:
        indexes = [
            models.Index(fields=["operating_graph", "node_type"]),
        ]

    @property
    def confidence(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def uncertainty(self):
        total = self.alpha + self.beta
        return (self.alpha * self.beta) ** 0.5 / total

    def __str__(self):
        return f"{self.label} ({self.node_type})"


class GraphEdge(SynaraEntity):
    """
    A relationship in the operating graph. IF-THEN with conditions.
    Conditions are DSL expressions evaluable against process data.
    Evidence links to core.Evidence via M2M.
    """

    operating_graph = models.ForeignKey(
        OperatingGraph,
        on_delete=models.CASCADE,
        related_name="edges",
    )
    source = models.ForeignKey(
        GraphNode,
        on_delete=models.CASCADE,
        related_name="outgoing_edges",
    )
    target = models.ForeignKey(
        GraphNode,
        on_delete=models.CASCADE,
        related_name="incoming_edges",
    )
    edge_type = models.CharField(
        max_length=20,
        choices=EdgeType.choices,
        default=EdgeType.CAUSES,
    )
    weight = models.FloatField(
        default=0.5,
        help_text="Strength of relationship (0-1).",
    )
    confidence = models.FloatField(
        default=0.5,
        help_text="Certainty that this edge exists (0-1).",
    )
    conditions = models.JSONField(
        default=list,
        blank=True,
        help_text="DSL expressions defining WHEN this relationship holds.",
    )
    truth_frequency = models.FloatField(
        default=1.0,
        help_text="How often the premise is satisfied (0-1). Updated by propagation.",
    )
    status = models.CharField(
        max_length=10,
        choices=EdgeStatus.choices,
        default=EdgeStatus.ACTIVE,
    )
    evidence = models.ManyToManyField(
        "core.Evidence",
        blank=True,
        related_name="prova_edges",
        help_text="Evidence backing this relationship.",
    )
    cycle_length = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Physical cycle boundary for feedback loops (job length, batch size).",
    )

    class Meta:
        indexes = [
            models.Index(fields=["operating_graph", "status"]),
            models.Index(fields=["source", "target"]),
        ]

    def __str__(self):
        return f"{self.source.label} —[{self.edge_type}]→ {self.target.label}"


# =============================================================================
# WORKING GRAPH — mutable scratchpad
# =============================================================================


class WorkingGraph(SynaraEntity):
    """
    Mutable scratchpad for hypotheses, curiosities, floor observations.
    Each user can have working graphs per project or standalone.
    Based on a specific operating graph version.
    """

    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        related_name="working_graphs",
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="working_graphs",
    )
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="working_graphs",
    )
    operating_graph = models.ForeignKey(
        OperatingGraph,
        on_delete=models.CASCADE,
        related_name="working_graphs",
        help_text="The operating graph this working graph branches from.",
    )
    state = models.JSONField(
        default=dict,
        blank=True,
        help_text="Serialized forgesia working state.",
    )

    def __str__(self):
        label = self.project or "standalone"
        return f"WorkingGraph ({self.owner}) — {label}"


# =============================================================================
# HYPOTHESES — proposed graph edits
# =============================================================================


class ProvaHypothesis(SynaraEntity):
    """
    A proposed edit to the operating graph. Built through the structured
    hypothesis builder — never free-text. Contains one or more GraphEdits.

    Curation: visible if outcome is concrete AND trial committed by date.
    """

    working_graph = models.ForeignKey(
        WorkingGraph,
        on_delete=models.CASCADE,
        related_name="hypotheses",
    )
    parent = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="refinements",
        help_text="Parent hypothesis if this is a refinement.",
    )
    description = models.TextField(
        help_text="Human-readable description of the proposed change.",
    )
    status = models.CharField(
        max_length=15,
        choices=HypothesisStatus.choices,
        default=HypothesisStatus.PROPOSED,
    )
    prior = models.FloatField(
        default=0.5,
        help_text="Prior belief (0-1) before any trials.",
    )
    posterior = models.FloatField(
        default=0.5,
        help_text="Current belief (0-1) after evidence.",
    )
    outcome_label = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="Concrete outcome from the outcome library.",
    )
    trial_commitment_date = models.DateField(
        null=True,
        blank=True,
        help_text="When the creator commits to trial this. Null = no intent.",
    )

    class Meta:
        verbose_name_plural = "PROVA hypotheses"

    @property
    def has_concrete_outcome(self):
        return bool(self.outcome_label)

    @property
    def has_trial_intent(self):
        return self.trial_commitment_date is not None

    @property
    def curation_tier(self):
        """Two-question curation filter."""
        if self.has_concrete_outcome and self.has_trial_intent:
            return "active"
        if self.has_concrete_outcome:
            return "curated"
        return "noise"

    def __str__(self):
        return f"Hypothesis: {self.description[:80]}"


class GraphEdit(SynaraEntity):
    """
    Single atomic change proposed by a hypothesis.
    The hypothesis is a SET of these edits — a diff on the operating graph.
    """

    hypothesis = models.ForeignKey(
        ProvaHypothesis,
        on_delete=models.CASCADE,
        related_name="edits",
    )
    operation = models.CharField(
        max_length=20,
        choices=EditOperation.choices,
    )
    target_edge = models.ForeignKey(
        GraphEdge,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="proposed_edits",
    )
    target_node = models.ForeignKey(
        GraphNode,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="proposed_edits",
    )
    params = models.JSONField(
        default=dict,
        help_text="Operation-specific parameters (new weight, condition DSL, etc.).",
    )

    def __str__(self):
        target = self.target_edge or self.target_node or "new"
        return f"{self.operation} → {target}"


# =============================================================================
# TRIALS — the only thing that changes the operating graph
# =============================================================================


class Trial(SynaraEntity):
    """
    A formal test of a relationship. Generates outcome ~ factor data.
    Only trials can promote changes from working graph to operating graph.

    DOE Lite: minimum shape enforced (known factors, outcome columns).
    Complexity tiers: green (no stats), blue (basic), purple (advanced).
    """

    hypothesis = models.ForeignKey(
        ProvaHypothesis,
        on_delete=models.CASCADE,
        related_name="trials",
    )
    edge = models.ForeignKey(
        GraphEdge,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="trials",
        help_text="The operating graph edge being tested (null for new edges).",
    )
    doe_spec = models.JSONField(
        default=dict,
        help_text="Trial design: factors, outcomes, design type, runs.",
    )
    raw_data = models.JSONField(
        default=dict,
        blank=True,
        help_text="Observed data (rows of factor/outcome values).",
    )
    evaluation = models.JSONField(
        default=dict,
        blank=True,
        help_text="Results: Bayes factor, discriminating power, graph delta, convergence.",
    )
    status = models.CharField(
        max_length=15,
        choices=TrialStatus.choices,
        default=TrialStatus.PLANNED,
    )
    meets_minimum_validity = models.BooleanField(
        default=False,
        help_text="Whether the trial data meets DOE Lite minimum shape.",
    )
    complexity_tier = models.CharField(
        max_length=10,
        choices=ComplexityTier.choices,
        default=ComplexityTier.GREEN,
    )

    def __str__(self):
        return f"Trial ({self.complexity_tier}) — {self.hypothesis}"


# =============================================================================
# CONFLICTS — broken edges
# =============================================================================


class Conflict(SynaraEntity):
    """
    A broken edge where evidence contradicts the operating graph.
    The edge weight drops out of evaluation entirely until resolved.
    System proposes economic resolution trials combinatorially.
    """

    edge = models.ForeignKey(
        GraphEdge,
        on_delete=models.CASCADE,
        related_name="conflicts",
    )
    operating_graph = models.ForeignKey(
        OperatingGraph,
        on_delete=models.CASCADE,
        related_name="conflicts",
    )
    competing_evidence = models.ManyToManyField(
        "core.Evidence",
        blank=True,
        related_name="prova_conflicts",
        help_text="The evidence that contradicts this edge.",
    )
    magnitude = models.FloatField(
        default=0.0,
        help_text="Size of the contradiction (0-1).",
    )
    evaluation_cost = models.FloatField(
        default=0.0,
        help_text="Predictive power lost by dropping this edge.",
    )
    proposed_resolutions = models.JSONField(
        default=list,
        blank=True,
        help_text="Combinatorial trial suggestions ranked by feasibility.",
    )
    status = models.CharField(
        max_length=15,
        choices=ConflictStatus.choices,
        default=ConflictStatus.OPEN,
    )
    resolved_by_trial = models.ForeignKey(
        Trial,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="resolved_conflicts",
    )

    def __str__(self):
        return f"Conflict on {self.edge} ({self.status})"


# =============================================================================
# PROPAGATION SIGNALS — emergent from upstream changes
# =============================================================================


class PropagationSignal(SynaraEntity):
    """
    Emergent signal from propagation after upstream changes.

    premise_dark: downstream premise almost never true (orphaned branch).
    denying_consequent: premise almost always true BUT outcome almost never
        true — hidden relationship the graph doesn't capture.
    """

    edge = models.ForeignKey(
        GraphEdge,
        on_delete=models.CASCADE,
        related_name="signals",
        help_text="The downstream edge affected by the upstream change.",
    )
    signal_type = models.CharField(
        max_length=25,
        choices=SignalType.choices,
    )
    triggered_by_trial = models.ForeignKey(
        Trial,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="triggered_signals",
        help_text="The trial whose promotion caused this signal.",
    )
    magnitude = models.FloatField(
        default=0.0,
        help_text="Severity of the signal (0-1).",
    )
    status = models.CharField(
        max_length=15,
        choices=SignalStatus.choices,
        default=SignalStatus.ACTIVE,
    )

    def __str__(self):
        return f"{self.signal_type} on {self.edge} ({self.status})"

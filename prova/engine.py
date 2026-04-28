"""
PROVA Engine — the service layer wrapping forgesia.

All access to forgesia goes through here. No other SVEND app imports
forgesia directly. PROVA mediates graph loading, evaluation, trial
promotion, conflict detection, and propagation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from django.db import transaction
from django.db.models import Q
from django.utils import timezone
from forgesia.graph.model import (
    CausalGraph,
)
from forgesia.graph.model import (
    Edge as SiaEdge,
)
from forgesia.graph.model import (
    EdgeType as SiaEdgeType,
)
from forgesia.graph.model import (
    Node as SiaNode,
)
from forgesia.graph.model import (
    NodeType as SiaNodeType,
)
from forgesia.inference.belief import (
    information_gain,
    suggest_likelihood_ratio,
)

from .models import (
    Conflict,
    ConflictStatus,
    EdgeStatus,
    GraphEdge,
    GraphNode,
    GraphVersion,
    OperatingGraph,
    PropagationSignal,
    ProvaHypothesis,
    SignalStatus,
    SignalType,
    Trial,
    TrialStatus,
)

logger = logging.getLogger("prova.engine")

# Minimum graph delta to consider evidence meaningful
MINIMUM_GRAPH_DELTA = 0.05

# Propagation signal thresholds
PREMISE_DARK_THRESHOLD = 0.1  # truth_frequency below this = dark
DENYING_CONSEQUENT_THRESHOLD = 0.9  # premise above this AND outcome below inverse


# =============================================================================
# DATA CLASSES — engine results
# =============================================================================


@dataclass
class EvidenceWeight:
    """Computed evidence weight from trial statistics."""

    theoretical_weight: float  # f(significance, effect_size)
    graph_delta: float  # actual change in predictive power
    is_meaningful: bool  # delta > MINIMUM_GRAPH_DELTA
    reliability: float  # CANON-002 reliability (source rank × sample × measurement)
    discriminating_power: float  # Bayes factor between graph versions


@dataclass
class PromotionResult:
    """Result of promoting a trial to the operating graph."""

    success: bool
    version_before: int
    version_after: int
    graph_delta: float
    signals: list[PropagationSignal]
    conflicts: list[Conflict]
    message: str


@dataclass
class PropagationResult:
    """Result of propagating changes through the graph."""

    edges_updated: int
    signals_created: list[PropagationSignal]
    total_truth_frequency_change: float


@dataclass
class TrialProposal:
    """A proposed trial design from the combinatorial engine."""

    description: str
    doe_spec: dict
    complexity_tier: str  # green, blue, purple
    discriminating_power: float
    estimated_cost: str  # qualitative: "low", "medium", "high"
    factors: list[str]
    outcome: str
    suggested_n: int


# =============================================================================
# GRAPH LOADING — Django models ↔ forgesia CausalGraph
# =============================================================================


def load_graph(operating_graph: OperatingGraph) -> CausalGraph:
    """Load Django models into a forgesia CausalGraph for computation.

    This is the bridge between PROVA's persistence layer and forgesia's
    computation layer. Returns a new CausalGraph every time — forgesia
    never holds state between calls.
    """
    graph = CausalGraph()

    nodes = GraphNode.objects.filter(operating_graph=operating_graph)
    for node in nodes:
        sia_node = SiaNode(
            id=str(node.id),
            node_type=SiaNodeType(node.node_type)
            if node.node_type in [e.value for e in SiaNodeType]
            else SiaNodeType.FACTOR,
            label=node.label,
            alpha=node.alpha,
            beta=node.beta,
            metadata=node.metadata or {},
        )
        graph.add_node(sia_node)

    edges = GraphEdge.objects.filter(
        operating_graph=operating_graph,
        status=EdgeStatus.ACTIVE,
    ).select_related("source", "target")
    for edge in edges:
        sia_edge = SiaEdge(
            id=str(edge.id),
            source=str(edge.source_id),
            target=str(edge.target_id),
            edge_type=SiaEdgeType(edge.edge_type)
            if edge.edge_type in [e.value for e in SiaEdgeType]
            else SiaEdgeType.CAUSES,
            weight=edge.weight,
            confidence=edge.confidence,
            metadata={
                "conditions": edge.conditions,
                "truth_frequency": edge.truth_frequency,
                "cycle_length": edge.cycle_length,
            },
        )
        graph.add_edge(sia_edge)

    return graph


def snapshot_graph(operating_graph: OperatingGraph) -> dict:
    """Create a JSON snapshot of the current operating graph state.

    Used for GraphVersion snapshots before promotion.
    """
    graph = load_graph(operating_graph)
    snapshot = graph.to_dict()

    # Add PROVA-specific metadata not in forgesia
    edges = GraphEdge.objects.filter(operating_graph=operating_graph)
    edge_meta = {}
    for edge in edges:
        edge_meta[str(edge.id)] = {
            "status": edge.status,
            "truth_frequency": edge.truth_frequency,
            "conditions": edge.conditions,
            "cycle_length": edge.cycle_length,
            "evidence_ids": list(edge.evidence.values_list("id", flat=True).distinct()[:100]),
        }

    snapshot["prova_edge_meta"] = edge_meta
    snapshot["version"] = operating_graph.current_version
    snapshot["predictive_score"] = operating_graph.predictive_score
    snapshot["timestamp"] = timezone.now().isoformat()

    return snapshot


# =============================================================================
# EVIDENCE WEIGHT — practical significance as arbiter
# =============================================================================


def compute_evidence_weight(
    p_value: Optional[float] = None,
    effect_size: Optional[float] = None,
    sample_size: Optional[int] = None,
    source_rank: float = 0.5,
    measurement_validity: float = 1.0,
) -> float:
    """Compute theoretical evidence weight.

    Combines statistical significance and effect size with CANON-002
    reliability factors. The REAL test is whether this weight produces
    a meaningful graph delta — that happens at evaluation time.

    Returns:
        Theoretical weight (0-1). This is reliability, not relevance.
        Relevance is determined by discriminating power at eval time.
    """
    # Base: forgesia's heuristic LR
    suggest_likelihood_ratio(
        p_value=p_value,
        effect_size=effect_size,
        sample_size=sample_size,
    )

    # Practical significance: combine significance and effect
    if p_value is not None and effect_size is not None:
        # Inverse significance × effect size, smoothed with log
        sig_component = -math.log10(max(p_value, 1e-10)) / 10  # 0-1 range
        eff_component = min(abs(effect_size), 2.0) / 2.0  # 0-1 range
        practical = (sig_component * eff_component) ** 0.5  # geometric mean
    elif effect_size is not None:
        practical = min(abs(effect_size), 2.0) / 2.0
    elif p_value is not None:
        practical = -math.log10(max(p_value, 1e-10)) / 10
    else:
        practical = 0.5

    # CANON-002 reliability
    reliability = source_rank * measurement_validity

    # Final weight: practical significance × reliability
    weight = practical * reliability

    return max(0.05, min(0.99, weight))


def compute_discriminating_power(
    operating_graph: OperatingGraph,
    hypothesis: ProvaHypothesis,
    trial_data: dict,
) -> EvidenceWeight:
    """Compute how well trial data discriminates between operating and proposed graphs.

    This is the core PROVA computation: Bayes factor between competing
    graph versions. The graph is its own arbiter.
    """
    graph = load_graph(operating_graph)

    # Compute predictive score of current graph
    current_score = _compute_predictive_score(graph)

    # Apply hypothesis edits to get proposed graph
    proposed_graph = _apply_edits(graph, hypothesis)

    # Compute predictive score of proposed graph
    proposed_score = _compute_predictive_score(proposed_graph)

    # Graph delta
    delta = proposed_score - current_score

    # Theoretical weight from trial statistics
    stats = trial_data.get("statistics", {})
    theoretical = compute_evidence_weight(
        p_value=stats.get("p_value"),
        effect_size=stats.get("effect_size"),
        sample_size=stats.get("sample_size"),
        source_rank=stats.get("source_rank", 0.5),
        measurement_validity=stats.get("measurement_validity", 1.0),
    )

    # Bayes factor: ratio of likelihoods under each model
    # > 1 favors proposed, < 1 favors current
    if current_score > 0:
        bayes_factor = proposed_score / current_score
    else:
        bayes_factor = proposed_score / 0.01

    return EvidenceWeight(
        theoretical_weight=theoretical,
        graph_delta=delta,
        is_meaningful=abs(delta) > MINIMUM_GRAPH_DELTA,
        reliability=theoretical,
        discriminating_power=bayes_factor,
    )


# =============================================================================
# TRIAL PROMOTION — the only thing that changes the operating graph
# =============================================================================


@transaction.atomic
def promote_trial(trial: Trial) -> PromotionResult:
    """Promote a completed trial's findings to the operating graph.

    1. Snapshot current state (create GraphVersion)
    2. Apply hypothesis edits to live models
    3. Run propagation (update truth frequencies)
    4. Detect conflicts and signals
    5. Update operating graph version + predictive score
    """
    hypothesis = trial.hypothesis
    working_graph = hypothesis.working_graph
    operating_graph = working_graph.operating_graph

    if trial.status != TrialStatus.COMPLETED:
        return PromotionResult(
            success=False,
            version_before=operating_graph.current_version,
            version_after=operating_graph.current_version,
            graph_delta=0.0,
            signals=[],
            conflicts=[],
            message="Trial must be completed before promotion.",
        )

    if not trial.meets_minimum_validity:
        return PromotionResult(
            success=False,
            version_before=operating_graph.current_version,
            version_after=operating_graph.current_version,
            graph_delta=0.0,
            signals=[],
            conflicts=[],
            message="Trial does not meet minimum DOE validity.",
        )

    # 1. Snapshot current state
    version_before = operating_graph.current_version
    snap = snapshot_graph(operating_graph)

    parent_version = (
        GraphVersion.objects.filter(
            operating_graph=operating_graph,
        )
        .order_by("-version_number")
        .first()
    )

    GraphVersion.objects.create(
        operating_graph=operating_graph,
        version_number=version_before,
        snapshot=snap,
        promoted_by_trial=trial,
        parent_version=parent_version,
        notes=f"Pre-promotion snapshot before trial: {trial.id}",
    )

    # 2. Apply edits to live models
    changed_edges = _apply_edits_to_models(hypothesis, operating_graph)

    # 3. Propagation
    prop_result = propagate(operating_graph, changed_edges)

    # 4. Detect conflicts
    new_conflicts = detect_conflicts(operating_graph, trial)

    # 5. Update operating graph
    operating_graph.current_version = version_before + 1
    operating_graph.predictive_score = _evaluate_graph_score(operating_graph)
    operating_graph.last_evaluated = timezone.now()
    operating_graph.save(
        update_fields=[
            "current_version",
            "predictive_score",
            "last_evaluated",
            "updated_at",
        ]
    )

    # Mark hypothesis as merged
    hypothesis.status = "merged"
    hypothesis.save(update_fields=["status", "updated_at"])

    graph_delta = operating_graph.predictive_score - snap.get("predictive_score", 0.0)

    logger.info(
        "Trial %s promoted → v%d (delta=%.4f, signals=%d, conflicts=%d)",
        trial.id,
        operating_graph.current_version,
        graph_delta,
        len(prop_result.signals_created),
        len(new_conflicts),
    )

    return PromotionResult(
        success=True,
        version_before=version_before,
        version_after=operating_graph.current_version,
        graph_delta=graph_delta,
        signals=prop_result.signals_created,
        conflicts=new_conflicts,
        message=f"Promoted to v{operating_graph.current_version}.",
    )


# =============================================================================
# PROPAGATION — truth frequency updates + signal detection
# =============================================================================


def propagate(
    operating_graph: OperatingGraph,
    changed_edges: list[GraphEdge],
) -> PropagationResult:
    """Propagate upstream changes to downstream premise truth frequencies.

    For each changed edge, find all downstream edges whose premise
    depends on the changed edge's target. Update their truth_frequency.
    Detect premise_dark and denying_consequent signals.
    """
    signals_created = []
    edges_updated = 0
    total_change = 0.0

    for changed_edge in changed_edges:
        # Find downstream edges: edges whose source is the target of the changed edge
        downstream_edges = GraphEdge.objects.filter(
            operating_graph=operating_graph,
            source=changed_edge.target,
            status=EdgeStatus.ACTIVE,
        )

        for downstream in downstream_edges:
            old_freq = downstream.truth_frequency

            # New truth frequency: based on the upstream edge's confidence × weight
            # This is simplified — the real computation would evaluate DSL conditions
            # against process data. For now: upstream confidence × weight = how often
            # the premise is satisfied.
            new_freq = changed_edge.confidence * changed_edge.weight
            if downstream.cycle_length:
                # Feedback loop: damped sampling
                new_freq = _damped_sample(
                    new_freq,
                    downstream.cycle_length,
                    old_freq,
                )

            downstream.truth_frequency = max(0.0, min(1.0, new_freq))
            downstream.save(update_fields=["truth_frequency", "updated_at"])

            delta = abs(new_freq - old_freq)
            total_change += delta
            edges_updated += 1

            # Signal detection
            signal = _detect_signal(downstream, changed_edge)
            if signal:
                signals_created.append(signal)

    return PropagationResult(
        edges_updated=edges_updated,
        signals_created=signals_created,
        total_truth_frequency_change=total_change,
    )


def _detect_signal(
    edge: GraphEdge,
    triggered_by_edge: GraphEdge,
) -> Optional[PropagationSignal]:
    """Detect premise_dark or denying_consequent on a downstream edge."""

    # Premise goes dark: truth_frequency drops below threshold
    if edge.truth_frequency < PREMISE_DARK_THRESHOLD:
        if edge.status != EdgeStatus.DARK:
            edge.status = EdgeStatus.DARK
            edge.save(update_fields=["status", "updated_at"])

        signal = PropagationSignal.objects.create(
            edge=edge,
            signal_type=SignalType.PREMISE_DARK,
            magnitude=1.0 - edge.truth_frequency,
            status=SignalStatus.ACTIVE,
        )
        logger.warning(
            "PREMISE DARK: %s (truth_freq=%.3f)",
            edge,
            edge.truth_frequency,
        )
        return signal

    # Denying the consequent: premise almost always true,
    # but the target node's confidence is very low
    if edge.truth_frequency > DENYING_CONSEQUENT_THRESHOLD:
        target_node = edge.target
        if target_node.confidence < (1 - DENYING_CONSEQUENT_THRESHOLD):
            signal = PropagationSignal.objects.create(
                edge=edge,
                signal_type=SignalType.DENYING_CONSEQUENT,
                magnitude=edge.truth_frequency * (1 - target_node.confidence),
                status=SignalStatus.ACTIVE,
            )
            logger.warning(
                "DENYING CONSEQUENT: %s (premise=%.3f, outcome=%.3f)",
                edge,
                edge.truth_frequency,
                target_node.confidence,
            )
            return signal

    return None


def _damped_sample(
    new_freq: float,
    cycle_length: int,
    current_freq: float,
    damping: float = 0.1,
) -> float:
    """Damped sampling for feedback loops.

    Simulates N iterations of the loop with physical cycle boundaries.
    Converges toward new_freq with damping. If it doesn't converge
    within cycle_length, the terminal state is the evidence.
    """
    value = current_freq
    for i in range(cycle_length):
        delta = new_freq - value
        value += delta * damping
        if abs(delta) < 0.001:
            break  # Converged before cycle end
    return value


# =============================================================================
# CONFLICT DETECTION
# =============================================================================


def detect_conflicts(
    operating_graph: OperatingGraph,
    trial: Trial,
) -> list[Conflict]:
    """Detect contradictions between trial evidence and operating graph edges.

    A conflict occurs when new evidence strongly contradicts an existing
    edge. The edge breaks — weight drops out of evaluation entirely.
    """
    conflicts = []
    evaluation = trial.evaluation or {}
    contradictions = evaluation.get("contradictions", [])

    for contradiction in contradictions:
        edge_id = contradiction.get("edge_id")
        if not edge_id:
            continue

        try:
            edge = GraphEdge.objects.get(id=edge_id, operating_graph=operating_graph)
        except GraphEdge.DoesNotExist:
            continue

        magnitude = contradiction.get("magnitude", 0.5)

        # Break the edge
        edge.status = EdgeStatus.BROKEN
        edge.save(update_fields=["status", "updated_at"])

        # Compute evaluation cost (how much predictive power we lose)
        eval_cost = _compute_edge_evaluation_cost(operating_graph, edge)

        conflict = Conflict.objects.create(
            edge=edge,
            operating_graph=operating_graph,
            magnitude=magnitude,
            evaluation_cost=eval_cost,
            proposed_resolutions=_propose_resolutions(edge, contradiction),
            status=ConflictStatus.OPEN,
        )

        logger.warning(
            "CONFLICT: %s (magnitude=%.3f, eval_cost=%.3f)",
            edge,
            magnitude,
            eval_cost,
        )

        conflicts.append(conflict)

    return conflicts


def _compute_edge_evaluation_cost(
    operating_graph: OperatingGraph,
    broken_edge: GraphEdge,
) -> float:
    """Estimate predictive power loss from dropping this edge.

    Loads the graph, removes the edge, compares predictive scores.
    """
    graph_with = load_graph(operating_graph)
    score_with = _compute_predictive_score(graph_with)

    graph_without = load_graph(operating_graph)
    edge_id = str(broken_edge.id)
    if edge_id in graph_without.edges:
        graph_without.remove_edge(edge_id)
    score_without = _compute_predictive_score(graph_without)

    return max(0.0, score_with - score_without)


def _propose_resolutions(edge: GraphEdge, contradiction: dict) -> list[dict]:
    """Generate economic trial proposals to resolve a conflict.

    Returns ranked list of resolution options from green (simplest) to
    purple (most rigorous).
    """
    source_label = edge.source.label if edge.source else "?"
    target_label = edge.target.label if edge.target else "?"

    return [
        {
            "tier": "green",
            "description": f"Run 10 observations each of {source_label} present vs absent, record {target_label}.",
            "estimated_runs": 20,
            "factors": [source_label],
            "outcome": target_label,
        },
        {
            "tier": "blue",
            "description": "Paired comparison: 5 days with current process, 5 days with change. Control for shift and material.",
            "estimated_runs": 10,
            "factors": [source_label, "shift", "material"],
            "outcome": target_label,
        },
        {
            "tier": "purple",
            "description": f"2^k fractional factorial on {source_label} and related factors.",
            "estimated_runs": 16,
            "factors": [source_label] + contradiction.get("related_factors", []),
            "outcome": target_label,
        },
    ]


# =============================================================================
# TRIAL PROPOSAL — combinatorial ranking by discriminating power
# =============================================================================


def propose_trials(
    hypothesis: ProvaHypothesis,
    operating_graph: Optional[OperatingGraph] = None,
) -> list[TrialProposal]:
    """Propose trial designs ranked by discriminating power and feasibility.

    The combinatorial engine evaluates the hypothesis edits against the
    operating graph and generates options at each complexity tier.
    """
    if operating_graph is None:
        operating_graph = hypothesis.working_graph.operating_graph

    graph = load_graph(operating_graph)
    edits = hypothesis.edits.all().select_related("target_edge", "target_node")

    proposals = []
    for edit in edits:
        target_label = ""
        outcome_label = hypothesis.outcome_label or "outcome"

        if edit.target_edge:
            source_node = edit.target_edge.source
            target_node = edit.target_edge.target
            target_label = source_node.label if source_node else "factor"
            outcome_label = target_node.label if target_node else outcome_label
        elif edit.target_node:
            target_label = edit.target_node.label

        if not target_label:
            target_label = "factor"

        # Estimate discriminating power using information gain
        node_id = str(edit.target_node_id or edit.target_edge_id or "")
        sia_node = graph.nodes.get(node_id)
        if sia_node:
            disc_power = information_gain(sia_node, 2.0)
        else:
            disc_power = 0.5

        # Green tier: simplest possible
        proposals.append(
            TrialProposal(
                description=f"Observe {target_label} vs {outcome_label}: 10 with, 10 without.",
                doe_spec={
                    "type": "simple_comparison",
                    "factors": [target_label],
                    "outcome": outcome_label,
                    "runs": 20,
                },
                complexity_tier="green",
                discriminating_power=disc_power * 0.6,
                estimated_cost="low",
                factors=[target_label],
                outcome=outcome_label,
                suggested_n=20,
            )
        )

        # Blue tier: controlled comparison
        proposals.append(
            TrialProposal(
                description=f"Paired comparison of {target_label} effect on {outcome_label}, controlling for shift and material.",
                doe_spec={
                    "type": "paired_comparison",
                    "factors": [target_label, "shift", "material"],
                    "outcome": outcome_label,
                    "runs": 20,
                    "blocking": ["shift"],
                },
                complexity_tier="blue",
                discriminating_power=disc_power * 0.8,
                estimated_cost="medium",
                factors=[target_label, "shift", "material"],
                outcome=outcome_label,
                suggested_n=20,
            )
        )

        # Purple tier: fractional factorial
        proposals.append(
            TrialProposal(
                description=f"2^(k-1) fractional factorial on {target_label} and related factors.",
                doe_spec={
                    "type": "fractional_factorial",
                    "factors": [target_label, "shift", "material", "operator"],
                    "outcome": outcome_label,
                    "runs": 8,
                    "resolution": "III",
                },
                complexity_tier="purple",
                discriminating_power=disc_power,
                estimated_cost="high",
                factors=[target_label, "shift", "material", "operator"],
                outcome=outcome_label,
                suggested_n=8,
            )
        )

    # Sort by discriminating power descending
    proposals.sort(key=lambda p: p.discriminating_power, reverse=True)

    return proposals


# =============================================================================
# GRAPH EVALUATION — predictive power scoring
# =============================================================================


def _compute_predictive_score(graph: CausalGraph) -> float:
    """Compute overall predictive power of a graph.

    Based on: edge confidence coverage, node belief convergence,
    structural completeness. Higher = better model of reality.
    """
    if graph.n_nodes == 0:
        return 0.0

    # Edge confidence: average confidence of all edges
    if graph.n_edges > 0:
        edge_conf = sum(e.confidence for e in graph.edges.values()) / graph.n_edges
    else:
        edge_conf = 0.0

    # Node convergence: how far nodes are from maximum uncertainty (0.5)
    node_convergence = 0.0
    for node in graph.nodes.values():
        node_convergence += abs(node.confidence - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1
    node_convergence /= graph.n_nodes

    # Structural score: ratio of edges to possible connections (sparse is fine)
    max_edges = graph.n_nodes * (graph.n_nodes - 1)
    if max_edges > 0:
        structural = min(1.0, graph.n_edges / (max_edges * 0.1))  # 10% density = 1.0
    else:
        structural = 0.0

    # Weighted combination
    score = 0.5 * edge_conf + 0.3 * node_convergence + 0.2 * structural

    return max(0.0, min(1.0, score))


def _evaluate_graph_score(operating_graph: OperatingGraph) -> float:
    """Evaluate the live operating graph's predictive score."""
    graph = load_graph(operating_graph)
    return _compute_predictive_score(graph)


# =============================================================================
# HYPOTHESIS EDIT APPLICATION — working graph → operating graph
# =============================================================================


def _apply_edits(
    graph: CausalGraph,
    hypothesis: ProvaHypothesis,
) -> CausalGraph:
    """Apply hypothesis edits to a forgesia CausalGraph (in-memory).

    Returns a modified copy for comparison. Does NOT touch Django models.
    """
    import copy

    proposed = copy.deepcopy(graph)

    for edit in hypothesis.edits.all():
        op = edit.operation
        params = edit.params or {}

        if op == "modify_strength" and edit.target_edge:
            edge_id = str(edit.target_edge_id)
            if edge_id in proposed.edges:
                proposed.edges[edge_id].weight = params.get("weight", proposed.edges[edge_id].weight)
                proposed.edges[edge_id].confidence = params.get("confidence", proposed.edges[edge_id].confidence)

        elif op == "remove_edge" and edit.target_edge:
            edge_id = str(edit.target_edge_id)
            if edge_id in proposed.edges:
                proposed.remove_edge(edge_id)

        elif op == "add_node":
            node_id = params.get("id", str(edit.id))
            proposed.add_node(
                SiaNode(
                    id=node_id,
                    node_type=SiaNodeType(params.get("node_type", "factor")),
                    label=params.get("label", "new node"),
                    alpha=params.get("alpha", 1.0),
                    beta=params.get("beta", 1.0),
                )
            )

        elif op == "add_edge":
            edge_id = params.get("id", str(edit.id))
            proposed.add_edge(
                SiaEdge(
                    id=edge_id,
                    source=params.get("source", ""),
                    target=params.get("target", ""),
                    edge_type=SiaEdgeType(params.get("edge_type", "causes")),
                    weight=params.get("weight", 0.5),
                    confidence=params.get("confidence", 0.5),
                )
            )

        elif op == "challenge_edge" and edit.target_edge:
            edge_id = str(edit.target_edge_id)
            if edge_id in proposed.edges:
                # Reduce confidence significantly
                proposed.edges[edge_id].confidence *= params.get("factor", 0.3)

        elif op == "replace_node" and edit.target_node:
            old_id = str(edit.target_node_id)
            new_label = params.get("new_label", "replacement")
            if old_id in proposed.nodes:
                proposed.nodes[old_id].label = new_label
                proposed.nodes[old_id].alpha = params.get("alpha", 1.0)
                proposed.nodes[old_id].beta = params.get("beta", 1.0)

        elif op == "add_condition" and edit.target_edge:
            edge_id = str(edit.target_edge_id)
            if edge_id in proposed.edges:
                conditions = proposed.edges[edge_id].metadata.get("conditions", [])
                conditions.append(params.get("condition", {}))
                proposed.edges[edge_id].metadata["conditions"] = conditions

    return proposed


def _apply_edits_to_models(
    hypothesis: ProvaHypothesis,
    operating_graph: OperatingGraph,
) -> list[GraphEdge]:
    """Apply hypothesis edits to live Django models.

    Returns list of changed edges for propagation.
    """
    changed = []

    for edit in hypothesis.edits.all().select_related("target_edge", "target_node"):
        op = edit.operation
        params = edit.params or {}

        if op == "modify_strength" and edit.target_edge:
            edge = edit.target_edge
            if "weight" in params:
                edge.weight = params["weight"]
            if "confidence" in params:
                edge.confidence = params["confidence"]
            edge.save(update_fields=["weight", "confidence", "updated_at"])
            changed.append(edge)

        elif op == "remove_edge" and edit.target_edge:
            edge = edit.target_edge
            edge.is_deleted = True
            edge.deleted_at = timezone.now()
            edge.save(update_fields=["is_deleted", "deleted_at", "updated_at"])

        elif op == "add_node":
            GraphNode.objects.create(
                operating_graph=operating_graph,
                label=params.get("label", "new node"),
                node_type=params.get("node_type", "factor"),
                alpha=params.get("alpha", 1.0),
                beta=params.get("beta", 1.0),
                metadata=params.get("metadata", {}),
            )

        elif op == "add_edge":
            source_id = params.get("source_id")
            target_id = params.get("target_id")
            if source_id and target_id:
                new_edge = GraphEdge.objects.create(
                    operating_graph=operating_graph,
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=params.get("edge_type", "causes"),
                    weight=params.get("weight", 0.5),
                    confidence=params.get("confidence", 0.5),
                    conditions=params.get("conditions", []),
                )
                changed.append(new_edge)

        elif op == "challenge_edge" and edit.target_edge:
            edge = edit.target_edge
            edge.confidence *= params.get("factor", 0.3)
            edge.save(update_fields=["confidence", "updated_at"])
            changed.append(edge)

        elif op == "replace_node" and edit.target_node:
            node = edit.target_node
            node.label = params.get("new_label", node.label)
            node.alpha = params.get("alpha", 1.0)
            node.beta = params.get("beta", 1.0)
            node.save(update_fields=["label", "alpha", "beta", "updated_at"])
            # All edges connected to this node are affected
            for edge in GraphEdge.objects.filter(
                operating_graph=operating_graph,
                status=EdgeStatus.ACTIVE,
            ).filter(
                Q(source=node) | Q(target=node),
            ):
                changed.append(edge)

        elif op == "add_condition" and edit.target_edge:
            edge = edit.target_edge
            conditions = edge.conditions or []
            conditions.append(params.get("condition", {}))
            edge.conditions = conditions
            edge.save(update_fields=["conditions", "updated_at"])
            changed.append(edge)

    return changed


# =============================================================================
# CONVENIENCE — get or create operating graph for tenant
# =============================================================================


def get_or_create_operating_graph(tenant) -> OperatingGraph:
    """Get the active operating graph for a tenant, or create one."""
    graph, created = OperatingGraph.objects.get_or_create(
        tenant=tenant,
        is_deleted=False,
        defaults={
            "current_version": 0,
            "predictive_score": 0.0,
        },
    )
    if created:
        logger.info("Created operating graph for tenant %s", tenant)
    return graph

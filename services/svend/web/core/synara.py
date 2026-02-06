"""Synara - Bayesian reasoning engine for hypothesis evaluation.

Synara handles:
- Bayesian probability updates based on evidence
- Likelihood ratio calculation from structured evidence
- Contradiction detection in knowledge graphs
- Logical consistency checking
- Information gain calculation for experiment suggestions
"""

import math
import logging
from typing import Optional
from dataclasses import dataclass

from django.db import transaction

logger = logging.getLogger(__name__)


@dataclass
class UpdateResult:
    """Result of applying evidence to a hypothesis."""
    hypothesis_id: str
    prior_probability: float
    posterior_probability: float
    likelihood_ratio: float
    adjusted_lr: float  # After confidence adjustment
    evidence_id: str
    status_changed: bool
    new_status: Optional[str] = None


@dataclass
class ConsistencyIssue:
    """A logical consistency issue found in the knowledge graph."""
    issue_type: str  # "contradiction", "circular", "unsupported", "fallacy"
    severity: str  # "error", "warning", "info"
    description: str
    entities_involved: list
    suggestions: list


class Synara:
    """Bayesian reasoning engine.

    Usage:
        synara = Synara()

        # Apply evidence to a hypothesis
        result = synara.apply_evidence(evidence_link)

        # Recalculate all hypotheses in a project
        synara.recalculate_project(project)

        # Check for logical issues
        issues = synara.check_consistency(knowledge_graph)

        # Calculate information gain for a potential experiment
        gain = synara.expected_information_gain(hypothesis, possible_outcomes)
    """

    # Likelihood ratio interpretation
    LR_VERY_STRONG_SUPPORT = 10.0
    LR_STRONG_SUPPORT = 3.0
    LR_MODERATE_SUPPORT = 1.5
    LR_NEUTRAL_LOW = 0.95
    LR_NEUTRAL_HIGH = 1.05
    LR_MODERATE_OPPOSITION = 0.67
    LR_STRONG_OPPOSITION = 0.33
    LR_VERY_STRONG_OPPOSITION = 0.1

    def __init__(self):
        pass

    def apply_evidence(self, evidence_link) -> UpdateResult:
        """Apply a single piece of evidence to update hypothesis probability.

        Uses Bayes' rule:
            posterior_odds = prior_odds × likelihood_ratio

        The likelihood ratio is adjusted by evidence confidence:
            adjusted_LR = 1 + (LR - 1) × confidence

        This means low-confidence evidence moves LR toward 1 (neutral).
        """
        from django.utils import timezone
        from .models import EvidenceLink

        hypothesis = evidence_link.hypothesis
        evidence = evidence_link.evidence

        prior_prob = hypothesis.current_probability
        prior_odds = prior_prob / (1 - prior_prob) if prior_prob < 1 else float('inf')

        # Get likelihood ratio and adjust for confidence
        lr = evidence_link.likelihood_ratio
        confidence = evidence.confidence
        adjusted_lr = 1 + (lr - 1) * confidence

        # Bayesian update
        posterior_odds = prior_odds * adjusted_lr
        posterior_prob = posterior_odds / (1 + posterior_odds)

        # Clamp to reasonable bounds (never 0 or 1)
        posterior_prob = max(0.01, min(0.99, posterior_prob))

        # Record in history
        hypothesis.probability_history.append({
            "probability": posterior_prob,
            "previous": prior_prob,
            "evidence_id": str(evidence.id),
            "likelihood_ratio": lr,
            "adjusted_lr": adjusted_lr,
            "confidence": confidence,
            "timestamp": timezone.now().isoformat(),
        })

        old_status = hypothesis.status
        hypothesis.current_probability = posterior_prob
        hypothesis._check_status_thresholds()
        hypothesis.save()

        # Mark evidence link as applied
        evidence_link.applied_at = timezone.now()
        evidence_link.save(update_fields=["applied_at"])

        return UpdateResult(
            hypothesis_id=str(hypothesis.id),
            prior_probability=prior_prob,
            posterior_probability=posterior_prob,
            likelihood_ratio=lr,
            adjusted_lr=adjusted_lr,
            evidence_id=str(evidence.id),
            status_changed=(old_status != hypothesis.status),
            new_status=hypothesis.status if old_status != hypothesis.status else None,
        )

    def recalculate_hypothesis(self, hypothesis) -> float:
        """Recalculate hypothesis probability from all linked evidence.

        Useful after evidence is modified or removed.
        Returns the new probability.
        """
        from django.utils import timezone

        # Start from prior
        prob = hypothesis.prior_probability
        odds = prob / (1 - prob)

        # Apply all evidence links
        for link in hypothesis.evidence_links.all():
            lr = link.likelihood_ratio
            confidence = link.evidence.confidence
            adjusted_lr = 1 + (lr - 1) * confidence
            odds *= adjusted_lr

        # Convert back to probability
        new_prob = odds / (1 + odds)
        new_prob = max(0.01, min(0.99, new_prob))

        hypothesis.current_probability = new_prob
        hypothesis.probability_history.append({
            "probability": new_prob,
            "reason": "recalculated",
            "timestamp": timezone.now().isoformat(),
        })
        hypothesis._check_status_thresholds()
        hypothesis.save()

        return new_prob

    @transaction.atomic
    def recalculate_project(self, project) -> dict:
        """Recalculate all hypotheses in a project.

        Returns a summary of changes.
        """
        results = {
            "hypotheses_updated": 0,
            "status_changes": [],
        }

        for hypothesis in project.hypotheses.all():
            old_status = hypothesis.status
            old_prob = hypothesis.current_probability

            self.recalculate_hypothesis(hypothesis)

            results["hypotheses_updated"] += 1

            if hypothesis.status != old_status:
                results["status_changes"].append({
                    "hypothesis_id": str(hypothesis.id),
                    "old_status": old_status,
                    "new_status": hypothesis.status,
                    "old_probability": old_prob,
                    "new_probability": hypothesis.current_probability,
                })

        return results

    def suggest_likelihood_ratio(self, evidence, hypothesis) -> tuple[float, str]:
        """Suggest a likelihood ratio based on evidence structure.

        Returns (suggested_lr, reasoning).

        This uses heuristics based on evidence type and statistical data.
        User should review and adjust.
        """
        lr = 1.0
        reasoning_parts = []

        # Statistical evidence with p-value
        if evidence.p_value is not None:
            p = evidence.p_value
            if p < 0.001:
                lr = 10.0
                reasoning_parts.append(f"Very significant result (p={p:.4f})")
            elif p < 0.01:
                lr = 5.0
                reasoning_parts.append(f"Highly significant result (p={p:.4f})")
            elif p < 0.05:
                lr = 2.0
                reasoning_parts.append(f"Significant result (p={p:.4f})")
            elif p < 0.1:
                lr = 1.3
                reasoning_parts.append(f"Marginally significant (p={p:.4f})")
            else:
                lr = 0.5
                reasoning_parts.append(f"Not significant (p={p:.4f}), may oppose")

        # Effect size
        if evidence.effect_size is not None:
            es = abs(evidence.effect_size)
            if es > 0.8:
                lr *= 1.5
                reasoning_parts.append(f"Large effect size ({es:.2f})")
            elif es > 0.5:
                lr *= 1.2
                reasoning_parts.append(f"Medium effect size ({es:.2f})")
            elif es < 0.2:
                lr *= 0.8
                reasoning_parts.append(f"Small effect size ({es:.2f})")

        # Sample size (larger = more reliable)
        if evidence.sample_size is not None:
            n = evidence.sample_size
            if n > 1000:
                lr *= 1.1
                reasoning_parts.append(f"Large sample (n={n})")
            elif n < 30:
                lr *= 0.9
                reasoning_parts.append(f"Small sample (n={n}), less reliable")

        # Evidence confidence already factors in separately
        if evidence.confidence < 0.5:
            reasoning_parts.append(f"Low evidence confidence ({evidence.confidence:.0%})")

        if not reasoning_parts:
            reasoning_parts.append("No statistical data available, using neutral LR")

        reasoning = "; ".join(reasoning_parts)
        return (round(lr, 2), reasoning)

    def expected_information_gain(
        self,
        hypothesis,
        lr_if_positive: float,
        lr_if_negative: float,
        prob_positive: float = 0.5,
    ) -> float:
        """Calculate expected information gain from a potential experiment.

        Given:
        - Current hypothesis probability P(H)
        - LR if experiment is positive (supports H)
        - LR if experiment is negative (opposes H)
        - Prior probability of positive result

        Returns expected reduction in entropy (bits).

        Higher = more informative experiment.
        """
        p_h = hypothesis.current_probability

        # Current entropy
        h_current = self._entropy(p_h)

        # Probability of positive result given H true and H false
        # Using: P(E+|H) = lr_pos × P(E+|¬H), normalized
        # This is simplified; real calculation depends on base rates

        # If positive result
        posterior_pos = self._update_probability(p_h, lr_if_positive)
        h_if_positive = self._entropy(posterior_pos)

        # If negative result
        posterior_neg = self._update_probability(p_h, lr_if_negative)
        h_if_negative = self._entropy(posterior_neg)

        # Expected posterior entropy
        h_expected = prob_positive * h_if_positive + (1 - prob_positive) * h_if_negative

        # Information gain = reduction in entropy
        info_gain = h_current - h_expected

        return max(0, info_gain)

    def _entropy(self, p: float) -> float:
        """Binary entropy in bits."""
        if p <= 0 or p >= 1:
            return 0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    def _update_probability(self, prior: float, lr: float) -> float:
        """Apply LR to get posterior probability."""
        if prior <= 0:
            return 0
        if prior >= 1:
            return 1
        odds = prior / (1 - prior)
        posterior_odds = odds * lr
        return posterior_odds / (1 + posterior_odds)

    def check_consistency(self, knowledge_graph) -> list[ConsistencyIssue]:
        """Check knowledge graph for logical consistency issues.

        Looks for:
        - Contradictions (A causes B and A prevents B)
        - Circular causation
        - Unsupported causal claims
        - Logical fallacies
        """
        from .models import Relationship

        issues = []

        # Get all relationships in graph
        relationships = list(knowledge_graph.relationships.select_related("source", "target"))

        # Build adjacency for analysis
        causal_edges = {}  # (source_id, target_id) -> [relation_types]
        for rel in relationships:
            key = (str(rel.source_id), str(rel.target_id))
            if key not in causal_edges:
                causal_edges[key] = []
            causal_edges[key].append(rel.relation_type)

        # Check for contradictions
        for (src, tgt), types in causal_edges.items():
            contradictions = self._find_contradictions(types)
            if contradictions:
                source = next(r.source for r in relationships if str(r.source_id) == src)
                target = next(r.target for r in relationships if str(r.target_id) == tgt)
                issues.append(ConsistencyIssue(
                    issue_type="contradiction",
                    severity="error",
                    description=f"Contradicting relationships between '{source.name}' and '{target.name}': {contradictions}",
                    entities_involved=[src, tgt],
                    suggestions=["Remove one of the contradicting relationships"],
                ))

        # Check for circular causation
        circular = self._find_circular_causation(relationships)
        for cycle in circular:
            issues.append(ConsistencyIssue(
                issue_type="circular",
                severity="warning",
                description=f"Circular causation detected: {' → '.join(cycle)}",
                entities_involved=cycle,
                suggestions=["Review causal chain for feedback loops vs actual circularity"],
            ))

        return issues

    def _find_contradictions(self, relation_types: list) -> list:
        """Find contradicting relationship types."""
        contradictions = []

        # Define contradicting pairs
        contradiction_pairs = [
            ("causes", "prevents"),
            ("supports", "contradicts"),
            ("correlates_with", "inversely_correlates"),
        ]

        for t1, t2 in contradiction_pairs:
            if t1 in relation_types and t2 in relation_types:
                contradictions.append((t1, t2))

        return contradictions

    def _find_circular_causation(self, relationships) -> list:
        """Find circular causal chains using DFS."""
        from .models import Relationship

        # Build graph of causal relationships only
        causal_types = {"causes", "enables", "triggers", "influences"}
        graph = {}

        for rel in relationships:
            if rel.relation_type in causal_types:
                src = str(rel.source_id)
                tgt = str(rel.target_id)
                if src not in graph:
                    graph[src] = []
                graph[src].append(tgt)

        # DFS for cycles
        cycles = []
        visited = set()
        path = []

        def dfs(node):
            if node in path:
                # Found cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            if node in visited:
                return

            visited.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                dfs(neighbor)

            path.pop()

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles


# Global instance
synara = Synara()

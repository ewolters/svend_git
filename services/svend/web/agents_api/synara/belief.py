"""
Synara Belief Engine: Bayesian Update Logic

Evidence reshapes belief mass - it never proves or disproves.

P(h | e) ∝ P(e | h) × P(h)

Where:
- P(h) is the prior (current belief)
- P(e | h) is the likelihood (how expected is e given h?)
- P(h | e) is the posterior (updated belief)

The key insight: Bayes is the algebra of belief under partial observability.
"""

import math
from typing import Optional
from .kernel import HypothesisRegion, Evidence, CausalGraph, ExpansionSignal


# Threshold below which likelihoods are considered "contradicting"
EXPANSION_THRESHOLD = 0.1

# Minimum probability mass (prevents complete certainty)
MIN_PROBABILITY = 0.001
MAX_PROBABILITY = 0.999


class BeliefEngine:
    """
    Bayesian belief update engine.

    Computes likelihoods, updates posteriors, detects expansion signals.
    """

    def __init__(
        self,
        expansion_threshold: float = EXPANSION_THRESHOLD,
        use_soft_evidence: bool = True,
    ):
        self.expansion_threshold = expansion_threshold
        self.use_soft_evidence = use_soft_evidence

    def compute_likelihood(
        self,
        evidence: Evidence,
        hypothesis: HypothesisRegion,
    ) -> float:
        """
        Compute P(e | h): How likely is this evidence given this hypothesis?

        Factors:
        1. Explicit support/weaken declarations
        2. Context match with hypothesis domain
        3. Behavior class alignment
        4. Evidence strength (measurement reliability)
        """
        # 1. Explicit declarations take precedence
        if hypothesis.id in evidence.supports:
            # Evidence explicitly supports this hypothesis
            base_likelihood = 0.8
        elif hypothesis.id in evidence.weakens:
            # Evidence explicitly weakens this hypothesis
            base_likelihood = 0.2
        else:
            # 2. Compute from domain match
            context_match = hypothesis.matches_context(evidence.context)

            # 3. Behavior alignment (if we can assess it)
            behavior_match = self._assess_behavior_alignment(evidence, hypothesis)

            # Combine: weighted average
            base_likelihood = 0.4 * context_match + 0.4 * behavior_match + 0.2 * 0.5

        # 4. Scale by evidence strength (measurement reliability)
        # Low strength evidence has likelihood closer to 0.5 (neutral)
        likelihood = self._apply_evidence_strength(base_likelihood, evidence.strength)

        return max(MIN_PROBABILITY, min(MAX_PROBABILITY, likelihood))

    def _assess_behavior_alignment(
        self,
        evidence: Evidence,
        hypothesis: HypothesisRegion,
    ) -> float:
        """
        Does the evidence align with the expected behavior class?

        This is a heuristic - in production, could be more sophisticated.
        """
        if not hypothesis.behavior_class:
            return 0.5  # No expected behavior = neutral

        event_lower = evidence.event.lower()
        behavior_lower = hypothesis.behavior_class.lower()

        # Simple keyword matching
        # Could be enhanced with embeddings or structured behavior ontology
        positive_indicators = [
            "increase", "rise", "high", "above", "exceed", "spike",
            "positive", "confirmed", "detected", "present"
        ]
        negative_indicators = [
            "decrease", "drop", "low", "below", "under", "fall",
            "negative", "absent", "not_detected", "missing"
        ]

        event_positive = any(ind in event_lower for ind in positive_indicators)
        event_negative = any(ind in event_lower for ind in negative_indicators)
        behavior_positive = any(ind in behavior_lower for ind in positive_indicators)
        behavior_negative = any(ind in behavior_lower for ind in negative_indicators)

        # Aligned directions = higher likelihood
        if (event_positive and behavior_positive) or (event_negative and behavior_negative):
            return 0.7
        elif (event_positive and behavior_negative) or (event_negative and behavior_positive):
            return 0.3
        else:
            return 0.5

    def _apply_evidence_strength(
        self,
        base_likelihood: float,
        strength: float,
    ) -> float:
        """
        Adjust likelihood based on evidence strength (reliability).

        Strong evidence (strength=1.0): likelihood unchanged
        Weak evidence (strength=0.0): likelihood → 0.5 (neutral)

        This models: "I'm not sure I observed this correctly"
        """
        neutral = 0.5
        return neutral + (base_likelihood - neutral) * strength

    def update_posteriors(
        self,
        graph: CausalGraph,
        evidence: Evidence,
        likelihoods: dict[str, float],
    ) -> dict[str, float]:
        """
        Bayesian update: P(h | e) ∝ P(e | h) × P(h)

        Updates all hypothesis posteriors in the graph.
        Returns the new posteriors.
        """
        # Compute unnormalized posteriors
        unnormalized = {}
        for h_id, hypothesis in graph.hypotheses.items():
            likelihood = likelihoods.get(h_id, 0.5)
            prior = hypothesis.posterior  # Current belief is the prior
            unnormalized[h_id] = likelihood * prior

        # Normalize so posteriors sum to 1
        total = sum(unnormalized.values())

        if total > 0:
            posteriors = {
                h_id: max(MIN_PROBABILITY, min(MAX_PROBABILITY, val / total))
                for h_id, val in unnormalized.items()
            }
        else:
            # Edge case: all likelihoods were 0
            # Keep priors but flag for expansion
            posteriors = {h_id: h.posterior for h_id, h in graph.hypotheses.items()}

        # Apply updates to graph
        for h_id, posterior in posteriors.items():
            graph.hypotheses[h_id].posterior = posterior

            # Track evidence
            if evidence.id not in graph.hypotheses[h_id].evidence_for and \
               evidence.id not in graph.hypotheses[h_id].evidence_against:
                if likelihoods.get(h_id, 0.5) > 0.5:
                    graph.hypotheses[h_id].evidence_for.append(evidence.id)
                elif likelihoods.get(h_id, 0.5) < 0.5:
                    graph.hypotheses[h_id].evidence_against.append(evidence.id)

        return posteriors

    def propagate_belief(
        self,
        graph: CausalGraph,
        updated_h_id: str,
    ) -> dict[str, float]:
        """
        Propagate belief changes through the causal graph.

        When a hypothesis's probability changes, downstream hypotheses
        should also be updated based on the causal links.

        P(h₂) = Σᵢ P(h₂ | hᵢ) × P(hᵢ)  for all upstream hᵢ
        """
        changes = {}
        hypothesis = graph.hypotheses.get(updated_h_id)
        if not hypothesis:
            return changes

        # Propagate to downstream hypotheses
        for downstream_id in hypothesis.downstream:
            downstream_h = graph.hypotheses.get(downstream_id)
            if not downstream_h:
                continue

            # Find the link
            link = next(
                (l for l in graph.links if l.from_id == updated_h_id and l.to_id == downstream_id),
                None
            )
            if not link:
                continue

            # Compute influence: P(downstream | upstream) × P(upstream)
            # This is a simplified model - full Bayesian network would be more complex
            influence = link.strength * hypothesis.posterior

            # Combine with other upstream influences
            other_upstream = [
                uid for uid in graph.get_upstream(downstream_id) if uid != updated_h_id
            ]

            if other_upstream:
                # Weighted combination of all upstream influences
                total_influence = influence
                for other_id in other_upstream:
                    other_h = graph.hypotheses.get(other_id)
                    other_link = next(
                        (l for l in graph.links if l.from_id == other_id and l.to_id == downstream_id),
                        None
                    )
                    if other_h and other_link:
                        total_influence += other_link.strength * other_h.posterior

                # Normalize
                new_posterior = min(MAX_PROBABILITY, total_influence / (len(other_upstream) + 1))
            else:
                new_posterior = influence

            # Update if changed significantly
            if abs(new_posterior - downstream_h.posterior) > 0.01:
                downstream_h.posterior = new_posterior
                changes[downstream_id] = new_posterior

                # Recurse
                downstream_changes = self.propagate_belief(graph, downstream_id)
                changes.update(downstream_changes)

        return changes

    def check_expansion(
        self,
        evidence: Evidence,
        likelihoods: dict[str, float],
    ) -> Optional[ExpansionSignal]:
        """
        Check if evidence suggests an incomplete causal surface.

        When: ∀h ∈ H: P(e | h) < threshold

        This means the evidence doesn't fit any hypothesis well.
        Either:
        1. Missing disjunct: need new hypothesis h_new
        2. Missing conjunct: need to expand existing h with new premises
        """
        if not likelihoods:
            return None

        # Check if all likelihoods are below threshold
        all_below_threshold = all(
            likelihood < self.expansion_threshold
            for likelihood in likelihoods.values()
        )

        if not all_below_threshold:
            return None

        # Generate expansion signal
        signal = ExpansionSignal(
            triggering_evidence=evidence.id,
            event=evidence.event,
            context=evidence.context,
            likelihoods=likelihoods,
            threshold=self.expansion_threshold,
            message=(
                f"Evidence '{evidence.event}' contradicts all hypotheses. "
                f"Causal surface may be incomplete."
            ),
            possible_causes=self._suggest_possible_causes(evidence, likelihoods),
        )

        return signal

    def _suggest_possible_causes(
        self,
        evidence: Evidence,
        likelihoods: dict[str, float],
    ) -> list[str]:
        """
        Suggest possible causes for an expansion signal.

        This is a heuristic starting point - could be enhanced with:
        - LLM-based cause generation
        - Domain knowledge bases
        - Historical expansion patterns
        """
        suggestions = []

        # Context-based suggestions
        for key, value in evidence.context.items():
            suggestions.append(f"Factor related to {key}={value}")

        # Event-based suggestions
        suggestions.append(f"Unknown cause producing '{evidence.event}'")

        # Suggest expanding existing hypotheses
        for h_id, likelihood in likelihoods.items():
            if likelihood > 0.05:  # Closest hypothesis
                suggestions.append(f"Additional premise for hypothesis '{h_id}'")

        return suggestions

    def compute_all_likelihoods(
        self,
        graph: CausalGraph,
        evidence: Evidence,
    ) -> dict[str, float]:
        """Compute likelihoods for all hypotheses in the graph."""
        return {
            h_id: self.compute_likelihood(evidence, hypothesis)
            for h_id, hypothesis in graph.hypotheses.items()
        }

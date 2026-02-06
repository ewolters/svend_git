"""
Synara: The Bayesian Belief Engine

Main interface for the epistemic motion system.

Synara is not a model of truth.
Synara is a model of epistemic motion -
how belief flows across a fractured causal landscape.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from uuid import uuid4

from .kernel import (
    HypothesisRegion,
    Evidence,
    CausalLink,
    CausalGraph,
    ExpansionSignal,
)
from .belief import BeliefEngine


@dataclass
class UpdateResult:
    """Result of adding evidence to Synara."""
    evidence_id: str

    # Likelihood of the evidence under each hypothesis
    likelihoods: dict[str, float]

    # Updated posteriors
    posteriors: dict[str, float]

    # Changes from belief propagation
    propagated_changes: dict[str, float]

    # Expansion signal (if causal surface appears incomplete)
    expansion_signal: Optional[ExpansionSignal] = None

    # Summary
    most_supported: Optional[str] = None  # Hypothesis with highest posterior
    most_weakened: Optional[str] = None   # Hypothesis with largest decrease

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "likelihoods": self.likelihoods,
            "posteriors": self.posteriors,
            "propagated_changes": self.propagated_changes,
            "expansion_signal": self.expansion_signal.to_dict() if self.expansion_signal else None,
            "most_supported": self.most_supported,
            "most_weakened": self.most_weakened,
        }


class Synara:
    """
    The Bayesian Belief Engine.

    Usage:
        synara = Synara()

        # Add hypotheses (behavioral regions)
        synara.add_hypothesis(HypothesisRegion(
            id="h_temp",
            description="Temperature drift causes defects",
            domain_conditions={"shift": "night"},
            behavior_class="defect_increase",
            prior=0.4,
        ))

        # Link hypotheses (build causal graph)
        synara.add_link(CausalLink(
            from_id="h_temp",
            to_id="h_viscosity",
            mechanism="Temperature affects coolant viscosity",
            strength=0.8,
        ))

        # Add evidence (reshapes belief)
        result = synara.add_evidence(Evidence(
            id="e_1",
            event="out_of_control_point",
            context={"shift": "night", "time": "03:00"},
        ))

        # Check for expansion signals
        if result.expansion_signal:
            print("Causal surface incomplete:", result.expansion_signal.message)
    """

    def __init__(
        self,
        expansion_threshold: float = 0.1,
    ):
        self.graph = CausalGraph()
        self.belief_engine = BeliefEngine(expansion_threshold=expansion_threshold)
        self.expansion_signals: list[ExpansionSignal] = []
        self.update_history: list[UpdateResult] = []

    # =========================================================================
    # Hypothesis Management
    # =========================================================================

    def add_hypothesis(
        self,
        hypothesis: HypothesisRegion,
    ) -> HypothesisRegion:
        """Add a hypothesis region to the causal graph."""
        # Initialize posterior to prior if not set
        if hypothesis.posterior == 0.5 and hypothesis.prior != 0.5:
            hypothesis.posterior = hypothesis.prior

        self.graph.add_hypothesis(hypothesis)
        return hypothesis

    def create_hypothesis(
        self,
        description: str,
        domain_conditions: dict = None,
        behavior_class: str = "",
        latent_causes: list[str] = None,
        prior: float = 0.5,
        source: str = "user",
    ) -> HypothesisRegion:
        """Convenience method to create and add a hypothesis."""
        h = HypothesisRegion(
            id=f"h_{uuid4().hex[:8]}",
            description=description,
            domain_conditions=domain_conditions or {},
            behavior_class=behavior_class,
            latent_causes=latent_causes or [],
            prior=prior,
            posterior=prior,
            source=source,
        )
        return self.add_hypothesis(h)

    def get_hypothesis(self, h_id: str) -> Optional[HypothesisRegion]:
        """Get a hypothesis by ID."""
        return self.graph.hypotheses.get(h_id)

    def get_all_hypotheses(self) -> list[HypothesisRegion]:
        """Get all hypotheses, sorted by posterior (descending)."""
        return sorted(
            self.graph.hypotheses.values(),
            key=lambda h: h.posterior,
            reverse=True,
        )

    # =========================================================================
    # Causal Links
    # =========================================================================

    def add_link(self, link: CausalLink) -> CausalLink:
        """Add a causal link between hypotheses."""
        self.graph.add_link(link)
        return link

    def create_link(
        self,
        from_id: str,
        to_id: str,
        mechanism: str = "",
        strength: float = 0.7,
    ) -> CausalLink:
        """Convenience method to create and add a causal link."""
        link = CausalLink(
            from_id=from_id,
            to_id=to_id,
            mechanism=mechanism,
            strength=strength,
        )
        return self.add_link(link)

    # =========================================================================
    # Evidence & Belief Update
    # =========================================================================

    def add_evidence(
        self,
        evidence: Evidence,
        propagate: bool = True,
    ) -> UpdateResult:
        """
        Add evidence and update beliefs.

        This is the core operation:
        1. Compute likelihoods P(e | h) for all hypotheses
        2. Check for expansion signal (all likelihoods low)
        3. Update posteriors P(h | e) ∝ P(e | h) × P(h)
        4. Propagate changes through causal graph

        Returns UpdateResult with all changes and any expansion signals.
        """
        # Store evidence in graph
        self.graph.evidence.append(evidence)

        # Store prior posteriors for comparison
        prior_posteriors = {
            h_id: h.posterior for h_id, h in self.graph.hypotheses.items()
        }

        # 1. Compute likelihoods
        likelihoods = self.belief_engine.compute_all_likelihoods(
            self.graph, evidence
        )

        # 2. Check for expansion signal BEFORE updating
        expansion_signal = self.belief_engine.check_expansion(evidence, likelihoods)
        if expansion_signal:
            self.expansion_signals.append(expansion_signal)

        # 3. Update posteriors
        posteriors = self.belief_engine.update_posteriors(
            self.graph, evidence, likelihoods
        )

        # 4. Propagate through causal graph
        propagated_changes = {}
        if propagate:
            for h_id in self.graph.hypotheses:
                changes = self.belief_engine.propagate_belief(self.graph, h_id)
                propagated_changes.update(changes)

        # Compute summary
        most_supported = None
        most_weakened = None
        max_increase = 0
        max_decrease = 0

        for h_id, new_posterior in posteriors.items():
            old_posterior = prior_posteriors.get(h_id, 0.5)
            change = new_posterior - old_posterior

            if change > max_increase:
                max_increase = change
                most_supported = h_id
            if change < max_decrease:
                max_decrease = change
                most_weakened = h_id

        result = UpdateResult(
            evidence_id=evidence.id,
            likelihoods=likelihoods,
            posteriors=posteriors,
            propagated_changes=propagated_changes,
            expansion_signal=expansion_signal,
            most_supported=most_supported,
            most_weakened=most_weakened,
        )

        self.update_history.append(result)
        return result

    def create_evidence(
        self,
        event: str,
        context: dict = None,
        supports: list[str] = None,
        weakens: list[str] = None,
        strength: float = 1.0,
        source: str = "user",
        data: dict = None,
    ) -> UpdateResult:
        """Convenience method to create evidence and add it."""
        e = Evidence(
            id=f"e_{uuid4().hex[:8]}",
            event=event,
            context=context or {},
            supports=supports or [],
            weakens=weakens or [],
            strength=strength,
            source=source,
            data=data,
        )
        return self.add_evidence(e)

    # =========================================================================
    # Expansion Handling
    # =========================================================================

    def get_pending_expansions(self) -> list[ExpansionSignal]:
        """Get all unresolved expansion signals."""
        return [s for s in self.expansion_signals if not s.resolved]

    def resolve_expansion(
        self,
        signal_id: str,
        resolution: str,
        new_hypothesis: Optional[HypothesisRegion] = None,
    ) -> bool:
        """
        Resolve an expansion signal.

        resolution options:
        - "new_hypothesis": a new disjunct was added to C
        - "expanded_hypothesis": an existing hypothesis was expanded
        - "dismissed": signal was a false positive
        """
        signal = next(
            (s for s in self.expansion_signals if s.id == signal_id),
            None
        )
        if not signal:
            return False

        signal.resolved = True
        signal.resolution = resolution

        if resolution == "new_hypothesis" and new_hypothesis:
            self.add_hypothesis(new_hypothesis)

        return True

    def suggest_hypothesis_from_expansion(
        self,
        signal: ExpansionSignal,
    ) -> HypothesisRegion:
        """
        Generate a suggested hypothesis from an expansion signal.

        This creates a skeleton hypothesis that the user can refine.
        """
        return HypothesisRegion(
            id=f"h_expanded_{uuid4().hex[:8]}",
            description=f"Unknown cause for '{signal.event}'",
            domain_conditions=signal.context.copy(),
            behavior_class="unknown",
            latent_causes=["unknown_factor"],
            prior=0.3,  # Start with moderate prior
            posterior=0.3,
            source="expansion",
        )

    # =========================================================================
    # Queries
    # =========================================================================

    def get_most_likely_cause(self) -> Optional[HypothesisRegion]:
        """Get the hypothesis with highest posterior."""
        if not self.graph.hypotheses:
            return None
        return max(self.graph.hypotheses.values(), key=lambda h: h.posterior)

    def get_competing_hypotheses(
        self,
        threshold: float = 0.1,
    ) -> list[HypothesisRegion]:
        """Get hypotheses within threshold of the top hypothesis."""
        if not self.graph.hypotheses:
            return []

        top_posterior = max(h.posterior for h in self.graph.hypotheses.values())
        return [
            h for h in self.graph.hypotheses.values()
            if h.posterior >= top_posterior - threshold
        ]

    def get_causal_chains_to(self, h_id: str) -> list[list[str]]:
        """Get all causal chains leading to a hypothesis."""
        return self.graph.get_paths_to(h_id)

    def explain_belief(self, h_id: str) -> dict:
        """
        Explain why a hypothesis has its current probability.

        Returns evidence for/against and causal influences.
        """
        hypothesis = self.graph.hypotheses.get(h_id)
        if not hypothesis:
            return {"error": "Hypothesis not found"}

        # Get supporting and weakening evidence
        supporting_evidence = [
            e for e in self.graph.evidence if e.id in hypothesis.evidence_for
        ]
        weakening_evidence = [
            e for e in self.graph.evidence if e.id in hypothesis.evidence_against
        ]

        # Get causal influences
        upstream = self.graph.get_upstream(h_id)
        downstream = self.graph.get_downstream(h_id)

        return {
            "hypothesis_id": h_id,
            "description": hypothesis.description,
            "prior": hypothesis.prior,
            "posterior": hypothesis.posterior,
            "evidence_for": [
                {"id": e.id, "event": e.event} for e in supporting_evidence
            ],
            "evidence_against": [
                {"id": e.id, "event": e.event} for e in weakening_evidence
            ],
            "upstream_causes": upstream,
            "downstream_effects": downstream,
            "domain_conditions": hypothesis.domain_conditions,
            "behavior_class": hypothesis.behavior_class,
        }

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict:
        """Serialize Synara state to dictionary."""
        return {
            "graph": self.graph.to_dict(),
            "expansion_signals": [s.to_dict() for s in self.expansion_signals],
            "update_history": [r.to_dict() for r in self.update_history[-50:]],  # Last 50
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Synara":
        """Deserialize Synara state from dictionary."""
        synara = cls()

        # Restore hypotheses
        for h_id, h_data in data.get("graph", {}).get("hypotheses", {}).items():
            synara.add_hypothesis(HypothesisRegion.from_dict(h_data))

        # Restore links
        for link_data in data.get("graph", {}).get("links", []):
            synara.add_link(CausalLink(**link_data))

        # Restore evidence (without triggering updates)
        for e_data in data.get("graph", {}).get("evidence", []):
            synara.graph.evidence.append(Evidence.from_dict(e_data))

        # Restore expansion signals
        for s_data in data.get("expansion_signals", []):
            synara.expansion_signals.append(ExpansionSignal(**s_data))

        return synara

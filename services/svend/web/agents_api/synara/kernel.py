"""
Synara Kernel: Core Data Structures

HypothesisRegion: Not a point claim, but a behavioral domain
Evidence: Observation with context that reshapes belief
CausalLink: Directed edge in the causal graph (h₁ → h₂)
CausalGraph: DAG of hypothesis regions
ExpansionSignal: Indicates incomplete causal surface
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Any
from uuid import uuid4


@dataclass
class HypothesisRegion:
    """
    A hypothesis is not a point claim - it's a behavioral region.

    x ∈ S ⇒ f(x) ~ B

    Where:
    - S is the domain_subset (conditions where this hypothesis applies)
    - B is the behavior_class (expected behavior pattern, not point value)

    A hypothesis represents one disjunct in the causal surface:
        effect ⇐ h₁ ∨ h₂ ∨ ... ∨ hₙ

    Each hypothesis may itself be a conjunction of latent causes:
        hᵢ = c₁ ∧ c₂ ∧ c₃
    """
    id: str
    description: str

    # Domain subset S - conditions where this hypothesis applies
    # e.g., {"shift": "night", "machine": "CNC-3", "material": "aluminum"}
    domain_conditions: dict = field(default_factory=dict)

    # Behavior class B - expected behavior pattern (not point prediction)
    # e.g., "defect_rate_increase", "weight_loss", "latency_spike"
    behavior_class: str = ""

    # Optional: statistical description of the behavior
    # e.g., {"distribution": "normal", "mean_shift": 0.5, "variance": 0.1}
    behavior_params: dict = field(default_factory=dict)

    # Latent causes this hypothesis represents (the conjunction)
    # e.g., ["coolant_viscosity_drop", "ambient_temp_rise"]
    # This is the ∧ in: hᵢ = c₁ ∧ c₂ ∧ ...
    latent_causes: list[str] = field(default_factory=list)

    # Probability mass
    prior: float = 0.5
    posterior: float = 0.5

    # Evidence links
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "user"  # user, agent, expansion

    # For chaining - what this hypothesis depends on / produces
    # (edges stored in CausalGraph, but useful for quick reference)
    upstream: list[str] = field(default_factory=list)  # hypotheses that cause this
    downstream: list[str] = field(default_factory=list)  # hypotheses this causes

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "HypothesisRegion":
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)

    def matches_context(self, context: dict) -> float:
        """
        How well does a given context match this hypothesis's domain?
        Returns 0.0 to 1.0.

        If context is a superset of domain_conditions, full match.
        Partial matches get partial scores.
        """
        if not self.domain_conditions:
            return 0.5  # No conditions = neutral match

        matches = 0
        total = len(self.domain_conditions)

        for key, expected in self.domain_conditions.items():
            if key in context:
                actual = context[key]
                if actual == expected:
                    matches += 1
                elif isinstance(expected, (list, tuple)) and actual in expected:
                    matches += 1
                # Could add fuzzy matching for numeric ranges here

        return matches / total if total > 0 else 0.5


@dataclass
class Evidence:
    """
    An observation with context that reshapes belief.

    Evidence does not prove or disprove hypotheses.
    It shifts probability mass between hypothesis regions.

    P(h | e) ∝ P(e | h) × P(h)
    """
    id: str

    # What was observed
    event: str  # e.g., "out_of_control_point", "weight_gain", "test_passed"

    # Conditions under which it was observed
    # e.g., {"shift": "night", "operator": "John", "time": "03:00"}
    context: dict = field(default_factory=dict)

    # Confidence in the observation itself (measurement reliability)
    strength: float = 1.0  # 0.0 to 1.0

    # Source of the evidence
    source: str = "user"  # user, spc, experiment, research, simulation

    # Which hypotheses this evidence speaks to (can be set explicitly or computed)
    supports: list[str] = field(default_factory=list)  # hypothesis IDs
    weakens: list[str] = field(default_factory=list)  # hypothesis IDs

    # Raw data associated with this evidence (optional)
    data: Optional[dict] = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Evidence":
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class CausalLink:
    """
    A directed edge in the causal graph: h₁ → h₂

    Represents: "h₁ being true increases probability of h₂"
    With mechanism: how does h₁ produce h₂?

    This enables hypothesis chaining:
        h₁ → h₂ → h₃ → effect

    Where each link has a conditional probability P(to | from).
    """
    from_id: str  # Source hypothesis
    to_id: str    # Target hypothesis

    # How does the cause produce the effect?
    mechanism: str = ""  # e.g., "temperature rise → viscosity drop"

    # Conditional probability P(to | from)
    # "If from is true, how likely is to?"
    strength: float = 0.7

    # Is this a necessary or sufficient condition?
    relation: str = "contributes"  # "contributes", "necessary", "sufficient"

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "user"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d


@dataclass
class CausalGraph:
    """
    Directed Acyclic Graph of hypothesis regions.

    The full causal surface is:
        effect ⇐ ⋁ᵢ (⋀ⱼ aᵢⱼ)

    Where:
    - Each path through the DAG represents a conjunction (chain of causes)
    - The set of all paths represents the disjunction (alternative explanations)

    Belief propagates through this graph when evidence is added.
    """
    hypotheses: dict[str, HypothesisRegion] = field(default_factory=dict)
    links: list[CausalLink] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)

    # Root hypotheses (no upstream causes in the graph)
    # These are the "base causes" we reason about
    @property
    def roots(self) -> list[str]:
        has_upstream = {link.to_id for link in self.links}
        return [h_id for h_id in self.hypotheses if h_id not in has_upstream]

    # Terminal hypotheses (no downstream effects in the graph)
    # These typically represent the observed effect
    @property
    def terminals(self) -> list[str]:
        has_downstream = {link.from_id for link in self.links}
        return [h_id for h_id in self.hypotheses if h_id not in has_downstream]

    def add_hypothesis(self, h: HypothesisRegion) -> None:
        self.hypotheses[h.id] = h

    def add_link(self, link: CausalLink) -> None:
        """Add a causal link, updating hypothesis references."""
        self.links.append(link)

        # Update hypothesis upstream/downstream references
        if link.from_id in self.hypotheses:
            if link.to_id not in self.hypotheses[link.from_id].downstream:
                self.hypotheses[link.from_id].downstream.append(link.to_id)

        if link.to_id in self.hypotheses:
            if link.from_id not in self.hypotheses[link.to_id].upstream:
                self.hypotheses[link.to_id].upstream.append(link.from_id)

    def get_upstream(self, h_id: str) -> list[str]:
        """Get all hypotheses that directly cause h_id."""
        return [link.from_id for link in self.links if link.to_id == h_id]

    def get_downstream(self, h_id: str) -> list[str]:
        """Get all hypotheses that h_id directly causes."""
        return [link.to_id for link in self.links if link.from_id == h_id]

    def get_all_ancestors(self, h_id: str, visited: set = None) -> set[str]:
        """Get all hypotheses upstream of h_id (transitive closure)."""
        if visited is None:
            visited = set()

        for upstream_id in self.get_upstream(h_id):
            if upstream_id not in visited:
                visited.add(upstream_id)
                self.get_all_ancestors(upstream_id, visited)

        return visited

    def get_all_descendants(self, h_id: str, visited: set = None) -> set[str]:
        """Get all hypotheses downstream of h_id (transitive closure)."""
        if visited is None:
            visited = set()

        for downstream_id in self.get_downstream(h_id):
            if downstream_id not in visited:
                visited.add(downstream_id)
                self.get_all_descendants(downstream_id, visited)

        return visited

    def get_paths_to(self, target_id: str) -> list[list[str]]:
        """
        Get all paths from roots to target.
        Each path is a chain: [root, ..., target]
        Each path represents one conjunction in the DNF.
        """
        paths = []

        def dfs(current: str, path: list[str]):
            if current == target_id:
                paths.append(path + [current])
                return

            for downstream_id in self.get_downstream(current):
                if downstream_id not in path:  # Avoid cycles
                    dfs(downstream_id, path + [current])

        for root in self.roots:
            dfs(root, [])

        return paths

    def to_dict(self) -> dict:
        return {
            "hypotheses": {h_id: h.to_dict() for h_id, h in self.hypotheses.items()},
            "links": [link.to_dict() for link in self.links],
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass
class ExpansionSignal:
    """
    Signal that the causal surface is incomplete.

    When: ∀h ∈ H: P(e | h) ≈ 0

    This does NOT mean "all hypotheses are wrong."
    It means: "there exists a cause c_new ∉ C such that P(c_new | e) >> 0"

    Two possibilities:
    1. Missing disjunct: need to add h_new to H
    2. Missing conjunct: need to expand existing h with new premises
    """
    # Required fields first (no defaults)
    triggering_evidence: str  # Evidence ID
    event: str

    # Fields with defaults
    id: str = field(default_factory=lambda: str(uuid4()))
    context: dict = field(default_factory=dict)

    # Likelihood scores that triggered the signal
    likelihoods: dict[str, float] = field(default_factory=dict)

    # Threshold used for detection
    threshold: float = 0.1

    # Analysis
    message: str = "Evidence outside all hypothesis regions"

    # Possible interpretations
    possible_causes: list[str] = field(default_factory=list)

    # Suggestions
    suggested_hypotheses: list[dict] = field(default_factory=list)

    # Status
    resolved: bool = False
    resolution: Optional[str] = None  # "new_hypothesis", "expanded_hypothesis", "dismissed"

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

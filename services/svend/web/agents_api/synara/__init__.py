"""
Synara: Bayesian Belief Engine for Causal Inference

A computable epistemology that models epistemic motion over causal manifolds.

Core insight: Human reasoning is discrete inference over continuous reality.
We don't see f(x) at points - we reason over behavioral regions.

    x ∈ S ⇒ f(x) ~ B  (not f(x) = y)

The causal surface is disjunctive normal form:

    effect ⇐ ⋁ᵢ (⋀ⱼ aᵢⱼ)

Evidence reshapes belief mass - it never proves or disproves.
When evidence contradicts all hypotheses, the model is INCOMPLETE, not wrong.

DSL for formal hypotheses uses operators:
    ALWAYS, NEVER, SOMETIMES, ALL, NONE, SOME
    AND, OR, XOR, NOT
    >, <, >=, <=, =, !=
    IF...THEN, WHEN

Example: if [num_holidays] > 3 then [monthly_sales] < 100000
"""

from .kernel import (
    HypothesisRegion,
    Evidence,
    CausalLink,
    CausalGraph,
    ExpansionSignal,
)
from .belief import BeliefEngine
from .synara import Synara, UpdateResult
from .llm_interface import SynaraLLMInterface, LogicalIssue, GraphAnalysis
from .dsl import (
    DSLParser,
    Hypothesis,
    Quantifier,
    LogicalOp,
    ComparisonOp,
    Variable,
    Literal,
    Comparison,
    LogicalExpr,
    Implication,
    Quantified,
    DomainCondition,
    format_hypothesis,
)
from .logic_engine import (
    LogicEngine,
    EvaluationResult,
    HypothesisEvaluation,
    Fallacy,
    FallacyType,
    parse_and_evaluate,
    validate_hypothesis,
)

__all__ = [
    # Kernel
    "HypothesisRegion",
    "Evidence",
    "CausalLink",
    "CausalGraph",
    "ExpansionSignal",
    # Belief Engine
    "BeliefEngine",
    "Synara",
    "UpdateResult",
    # LLM Interface
    "SynaraLLMInterface",
    "LogicalIssue",
    "GraphAnalysis",
    # DSL
    "DSLParser",
    "Hypothesis",
    "Quantifier",
    "LogicalOp",
    "ComparisonOp",
    "Variable",
    "Literal",
    "Comparison",
    "LogicalExpr",
    "Implication",
    "Quantified",
    "DomainCondition",
    "format_hypothesis",
    # Logic Engine
    "LogicEngine",
    "EvaluationResult",
    "HypothesisEvaluation",
    "Fallacy",
    "FallacyType",
    "parse_and_evaluate",
    "validate_hypothesis",
]

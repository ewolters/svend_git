"""
Neuro-Symbolic Reasoning for Code Generation

Applies Bayesian reasoning to code quality assessment:
- Treat solution approaches as hypotheses
- Verification results as evidence
- Track confidence, not just pass/fail
- Use structured reasoning for test design

Integrates concepts from /home/eric/Desktop/experiments/neuro_symbolic/
"""

from dataclasses import dataclass, field
from typing import Any
import math


@dataclass
class CodeHypothesis:
    """A hypothesis about a solution approach."""
    id: str
    description: str
    code: str
    prior: float = 0.5  # Initial confidence
    posterior: float = 0.5  # Current confidence
    evidence: list[dict] = field(default_factory=list)

    def update(self, evidence_name: str, likelihood_ratio: float):
        """Bayesian update based on evidence."""
        # Odds form: posterior_odds = LR * prior_odds
        prior_odds = self.posterior / (1 - self.posterior + 1e-10)
        posterior_odds = likelihood_ratio * prior_odds
        self.posterior = posterior_odds / (1 + posterior_odds)

        self.evidence.append({
            "name": evidence_name,
            "lr": likelihood_ratio,
            "posterior": self.posterior,
        })


@dataclass
class QualityAssessment:
    """Structured quality assessment with confidence."""
    overall_confidence: float  # 0-1
    dimensions: dict[str, float] = field(default_factory=dict)  # Per-dimension scores
    evidence_trail: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class CodeReasoner:
    """
    Structured reasoning about code quality.

    Instead of binary pass/fail, tracks:
    - Confidence in correctness
    - Evidence for/against
    - Specific quality dimensions
    """

    # Evidence likelihood ratios
    EVIDENCE_LR = {
        # Syntax
        "syntax_pass": 3.0,
        "syntax_fail": 0.01,

        # Execution
        "runs_without_error": 5.0,
        "runtime_error": 0.1,
        "timeout": 0.2,

        # Tests
        "all_tests_pass": 10.0,
        "some_tests_pass": 2.0,
        "all_tests_fail": 0.05,

        # Lint
        "no_lint_errors": 2.0,
        "minor_lint_issues": 0.8,
        "major_lint_issues": 0.3,

        # Intent alignment (from LLM check)
        "perfectly_aligned": 5.0,
        "mostly_aligned": 2.0,
        "partially_aligned": 0.5,
        "misaligned": 0.1,

        # Code style
        "has_docstrings": 1.5,
        "has_type_hints": 1.5,
        "good_naming": 1.3,
    }

    def __init__(self, prior: float = 0.5):
        self.base_prior = prior
        self.hypotheses: list[CodeHypothesis] = []

    def add_hypothesis(self, id: str, description: str, code: str, prior: float = None):
        """Add a solution hypothesis."""
        h = CodeHypothesis(
            id=id,
            description=description,
            code=code,
            prior=prior or self.base_prior,
            posterior=prior or self.base_prior,
        )
        self.hypotheses.append(h)
        return h

    def update_with_evidence(self, hypothesis_id: str, evidence_type: str, passed: bool = True):
        """Update hypothesis confidence with evidence."""
        h = self._get_hypothesis(hypothesis_id)
        if not h:
            return

        # Get likelihood ratio
        if evidence_type in self.EVIDENCE_LR:
            lr = self.EVIDENCE_LR[evidence_type]
        else:
            lr = 2.0 if passed else 0.5  # Default

        h.update(evidence_type, lr)

    def assess_code(self, code: str, verification_results: dict, alignment_score: float) -> QualityAssessment:
        """
        Assess code quality using structured reasoning.

        Returns confidence scores across dimensions.
        """
        dimensions = {}
        evidence = []
        recommendations = []

        # Start with base prior
        confidence = self.base_prior

        # Syntax dimension
        if verification_results.get("syntax", False):
            dimensions["syntax"] = 1.0
            confidence = self._bayesian_update(confidence, self.EVIDENCE_LR["syntax_pass"])
            evidence.append("Syntax valid")
        else:
            dimensions["syntax"] = 0.0
            confidence = self._bayesian_update(confidence, self.EVIDENCE_LR["syntax_fail"])
            evidence.append("Syntax error")
            recommendations.append("Fix syntax errors before proceeding")

        # Execution dimension
        if verification_results.get("execution", False):
            dimensions["execution"] = 1.0
            confidence = self._bayesian_update(confidence, self.EVIDENCE_LR["runs_without_error"])
            evidence.append("Executes without error")
        else:
            dimensions["execution"] = 0.0
            error = verification_results.get("execution_error", "unknown")
            if "timeout" in str(error).lower():
                confidence = self._bayesian_update(confidence, self.EVIDENCE_LR["timeout"])
                evidence.append("Execution timed out")
                recommendations.append("Check for infinite loops or long-running operations")
            else:
                confidence = self._bayesian_update(confidence, self.EVIDENCE_LR["runtime_error"])
                evidence.append(f"Runtime error: {error}")

        # Lint dimension
        if verification_results.get("lint", False):
            dimensions["lint"] = 1.0
            confidence = self._bayesian_update(confidence, self.EVIDENCE_LR["no_lint_errors"])
            evidence.append("No lint issues")
        else:
            dimensions["lint"] = 0.5
            confidence = self._bayesian_update(confidence, self.EVIDENCE_LR["minor_lint_issues"])
            evidence.append("Minor lint issues")

        # Intent alignment dimension
        dimensions["alignment"] = alignment_score
        if alignment_score > 0.8:
            confidence = self._bayesian_update(confidence, self.EVIDENCE_LR["perfectly_aligned"])
            evidence.append("Perfectly aligned with intent")
        elif alignment_score > 0.6:
            confidence = self._bayesian_update(confidence, self.EVIDENCE_LR["mostly_aligned"])
            evidence.append("Mostly aligned with intent")
        elif alignment_score > 0.4:
            confidence = self._bayesian_update(confidence, self.EVIDENCE_LR["partially_aligned"])
            evidence.append("Partially aligned - may have drifted")
            recommendations.append("Review code against original intent")
        else:
            confidence = self._bayesian_update(confidence, self.EVIDENCE_LR["misaligned"])
            evidence.append("Misaligned with intent")
            recommendations.append("Regenerate with clearer focus on original request")

        # Code quality indicators
        if '"""' in code or "'''" in code:
            dimensions["documentation"] = 1.0
            confidence = self._bayesian_update(confidence, self.EVIDENCE_LR["has_docstrings"])
            evidence.append("Has docstrings")
        else:
            dimensions["documentation"] = 0.3

        if "->" in code or ": " in code:
            dimensions["type_hints"] = 1.0
            confidence = self._bayesian_update(confidence, self.EVIDENCE_LR["has_type_hints"])
            evidence.append("Has type hints")
        else:
            dimensions["type_hints"] = 0.3

        return QualityAssessment(
            overall_confidence=confidence,
            dimensions=dimensions,
            evidence_trail=evidence,
            recommendations=recommendations,
        )

    def _bayesian_update(self, prior: float, likelihood_ratio: float) -> float:
        """Single Bayesian update."""
        prior_odds = prior / (1 - prior + 1e-10)
        posterior_odds = likelihood_ratio * prior_odds
        return posterior_odds / (1 + posterior_odds)

    def _get_hypothesis(self, id: str) -> CodeHypothesis | None:
        for h in self.hypotheses:
            if h.id == id:
                return h
        return None

    def suggest_tests(self, code: str, intent: str) -> list[str]:
        """Suggest tests that would provide most information about code quality."""
        # These are the experiments that would most update our beliefs
        suggestions = [
            "Test with typical input values",
            "Test with edge cases (empty, zero, negative)",
            "Test with invalid input types",
        ]

        # Add intent-specific suggestions
        intent_lower = intent.lower()
        if "sort" in intent_lower:
            suggestions.append("Test with already sorted input")
            suggestions.append("Test with reverse sorted input")
        if "search" in intent_lower or "find" in intent_lower:
            suggestions.append("Test when item not found")
            suggestions.append("Test with duplicate items")
        if "file" in intent_lower or "read" in intent_lower or "write" in intent_lower:
            suggestions.append("Test with non-existent file")
            suggestions.append("Test with permission errors")

        return suggestions

    def format_assessment(self, assessment: QualityAssessment) -> str:
        """Format assessment for display."""
        lines = [
            f"Overall Confidence: {assessment.overall_confidence:.0%}",
            "",
            "Dimensions:",
        ]

        for dim, score in sorted(assessment.dimensions.items()):
            bar = "=" * int(score * 10) + "-" * (10 - int(score * 10))
            lines.append(f"  {dim:15} [{bar}] {score:.0%}")

        lines.append("")
        lines.append("Evidence:")
        for e in assessment.evidence_trail:
            lines.append(f"  + {e}")

        if assessment.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for r in assessment.recommendations:
                lines.append(f"  ! {r}")

        return "\n".join(lines)

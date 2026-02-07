"""
Synara Logic Engine: Deterministic Evaluation of Formal Hypotheses

Evaluates parsed DSL hypotheses against actual data.
This is separate from LLM - purely deterministic logic.

The engine:
1. Validates logical structure (detects fallacies)
2. Evaluates hypotheses against datasets
3. Reports support/refutation with evidence
4. Escalates to LLM when ambiguity detected
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Any, Callable
from enum import Enum

from .dsl import (
    Hypothesis, Expression, Variable, Literal, Comparison, LogicalExpr,
    Implication, Quantified, DomainCondition, Quantifier, ComparisonOp, LogicalOp,
    DSLParser,
)


class EvaluationResult(Enum):
    """Result of evaluating a hypothesis."""
    SUPPORTED = "supported"           # Evidence supports hypothesis
    REFUTED = "refuted"               # Evidence contradicts hypothesis
    UNDETERMINED = "undetermined"     # Insufficient evidence
    PARTIAL = "partial"               # Some conditions met, others not
    ERROR = "error"                   # Evaluation failed


class FallacyType(Enum):
    """Types of logical fallacies detected."""
    AFFIRMING_CONSEQUENT = "affirming_consequent"  # P→Q, Q ∴ P
    DENYING_ANTECEDENT = "denying_antecedent"      # P→Q, ¬P ∴ ¬Q
    CIRCULAR_REASONING = "circular_reasoning"
    FALSE_DICHOTOMY = "false_dichotomy"
    HASTY_GENERALIZATION = "hasty_generalization"
    OVERGENERALIZATION = "overgeneralization"
    UNFALSIFIABLE = "unfalsifiable"


@dataclass
class Fallacy:
    """A detected logical fallacy."""
    type: FallacyType
    description: str
    location: str  # Which part of hypothesis
    severity: str  # "error", "warning"
    suggestion: Optional[str] = None


@dataclass
class EvaluationEvidence:
    """Evidence from evaluation."""
    row_indices: list[int]
    matching_count: int
    total_count: int
    sample_values: list[dict]  # Sample rows


@dataclass
class HypothesisEvaluation:
    """Complete evaluation result for a hypothesis."""
    hypothesis: Hypothesis
    result: EvaluationResult
    confidence: float  # 0.0 - 1.0
    supporting_evidence: EvaluationEvidence
    refuting_evidence: EvaluationEvidence
    fallacies: list[Fallacy]
    explanation: str
    needs_llm_review: bool = False
    llm_review_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "result": self.result.value,
            "confidence": self.confidence,
            "supporting": {
                "count": self.supporting_evidence.matching_count,
                "total": self.supporting_evidence.total_count,
                "samples": self.supporting_evidence.sample_values[:5],
            },
            "refuting": {
                "count": self.refuting_evidence.matching_count,
                "total": self.refuting_evidence.total_count,
                "samples": self.refuting_evidence.sample_values[:5],
            },
            "fallacies": [
                {"type": f.type.value, "description": f.description, "severity": f.severity}
                for f in self.fallacies
            ],
            "explanation": self.explanation,
            "needs_llm_review": self.needs_llm_review,
            "llm_review_reason": self.llm_review_reason,
        }


class LogicEngine:
    """
    Deterministic logic engine for hypothesis evaluation.

    Separates concerns:
    - DSL Parser: text → AST
    - Logic Engine: AST + data → evaluation
    - LLM Interface: ambiguity → guidance
    """

    def __init__(self):
        self.parser = DSLParser()

    def parse(self, text: str) -> Hypothesis:
        """Parse hypothesis text into AST."""
        return self.parser.parse(text)

    def validate(self, hypothesis: Hypothesis) -> list[Fallacy]:
        """
        Validate hypothesis for logical fallacies.

        This is deterministic - no LLM needed.
        """
        fallacies = []

        # Check parse errors
        for error in hypothesis.parse_errors:
            fallacies.append(Fallacy(
                type=FallacyType.UNFALSIFIABLE,
                description=error,
                location="parse",
                severity="error",
            ))

        # Check structure
        fallacies.extend(self._check_structure(hypothesis.ast))

        # Check for common fallacy patterns
        fallacies.extend(self._check_fallacy_patterns(hypothesis.ast))

        return fallacies

    def evaluate(
        self,
        hypothesis: Hypothesis,
        data: list[dict],
        variable_context: Optional[dict[str, str]] = None,
    ) -> HypothesisEvaluation:
        """
        Evaluate hypothesis against dataset.

        Args:
            hypothesis: Parsed hypothesis
            data: List of row dictionaries
            variable_context: Optional mapping of variable names to column names
        """
        # First validate
        fallacies = self.validate(hypothesis)

        if not data:
            return HypothesisEvaluation(
                hypothesis=hypothesis,
                result=EvaluationResult.UNDETERMINED,
                confidence=0.0,
                supporting_evidence=EvaluationEvidence([], 0, 0, []),
                refuting_evidence=EvaluationEvidence([], 0, 0, []),
                fallacies=fallacies,
                explanation="No data provided for evaluation",
            )

        # Build variable resolver
        resolver = self._build_resolver(variable_context or {})

        # Evaluate based on AST type
        try:
            result, support_idx, refute_idx = self._evaluate_ast(
                hypothesis.ast, data, resolver
            )
        except Exception as e:
            return HypothesisEvaluation(
                hypothesis=hypothesis,
                result=EvaluationResult.ERROR,
                confidence=0.0,
                supporting_evidence=EvaluationEvidence([], 0, len(data), []),
                refuting_evidence=EvaluationEvidence([], 0, len(data), []),
                fallacies=fallacies,
                explanation=f"Evaluation error: {str(e)}",
                needs_llm_review=True,
                llm_review_reason=f"Evaluation error: {str(e)}",
            )

        # Build evidence objects
        supporting = EvaluationEvidence(
            row_indices=support_idx,
            matching_count=len(support_idx),
            total_count=len(data),
            sample_values=[data[i] for i in support_idx[:10]],
        )
        refuting = EvaluationEvidence(
            row_indices=refute_idx,
            matching_count=len(refute_idx),
            total_count=len(data),
            sample_values=[data[i] for i in refute_idx[:10]],
        )

        # Calculate confidence
        confidence = self._calculate_confidence(result, supporting, refuting, hypothesis)

        # Generate explanation
        explanation = self._generate_explanation(result, hypothesis, supporting, refuting)

        # Check if LLM review needed
        needs_review, review_reason = self._check_needs_review(
            result, confidence, fallacies, hypothesis
        )

        return HypothesisEvaluation(
            hypothesis=hypothesis,
            result=result,
            confidence=confidence,
            supporting_evidence=supporting,
            refuting_evidence=refuting,
            fallacies=fallacies,
            explanation=explanation,
            needs_llm_review=needs_review,
            llm_review_reason=review_reason,
        )

    def _build_resolver(self, context: dict[str, str]) -> Callable[[str, dict], Any]:
        """Build a variable resolver function."""
        def resolve(var_name: str, row: dict) -> Any:
            # First check context mapping
            col_name = context.get(var_name, var_name)
            if col_name in row:
                return row[col_name]
            # Try case-insensitive match
            for key in row:
                if key.lower() == col_name.lower():
                    return row[key]
            return None
        return resolve

    def _evaluate_ast(
        self,
        ast: Expression,
        data: list[dict],
        resolver: Callable,
    ) -> tuple[EvaluationResult, list[int], list[int]]:
        """
        Evaluate AST against data.

        Returns (result, supporting_indices, refuting_indices)
        """
        if isinstance(ast, Quantified):
            return self._evaluate_quantified(ast, data, resolver)
        elif isinstance(ast, Implication):
            return self._evaluate_implication(ast, data, resolver)
        elif isinstance(ast, LogicalExpr):
            return self._evaluate_logical(ast, data, resolver)
        elif isinstance(ast, Comparison):
            return self._evaluate_comparison_over_data(ast, data, resolver)
        else:
            return EvaluationResult.ERROR, [], []

    def _evaluate_quantified(
        self,
        node: Quantified,
        data: list[dict],
        resolver: Callable,
    ) -> tuple[EvaluationResult, list[int], list[int]]:
        """Evaluate quantified expression."""
        # Filter by domain if present
        if node.domain:
            filtered_data = []
            filtered_indices = []
            for i, row in enumerate(data):
                if self._evaluate_condition(node.domain.condition, row, resolver):
                    filtered_data.append(row)
                    filtered_indices.append(i)
        else:
            filtered_data = data
            filtered_indices = list(range(len(data)))

        if not filtered_data:
            return EvaluationResult.UNDETERMINED, [], []

        # Evaluate body for each row
        supporting = []
        refuting = []

        for i, (row, orig_idx) in enumerate(zip(filtered_data, filtered_indices)):
            result = self._evaluate_body(node.body, row, resolver)
            if result:
                supporting.append(orig_idx)
            else:
                refuting.append(orig_idx)

        # Interpret based on quantifier
        if node.quantifier in (Quantifier.ALWAYS, Quantifier.ALL):
            # All must be true
            if not refuting:
                return EvaluationResult.SUPPORTED, supporting, refuting
            else:
                return EvaluationResult.REFUTED, supporting, refuting

        elif node.quantifier in (Quantifier.NEVER, Quantifier.NONE):
            # None must be true (all must be false)
            if not supporting:
                return EvaluationResult.SUPPORTED, [], refuting
            else:
                return EvaluationResult.REFUTED, supporting, refuting

        elif node.quantifier in (Quantifier.SOMETIMES, Quantifier.SOME):
            # At least one must be true
            if supporting:
                return EvaluationResult.SUPPORTED, supporting, refuting
            else:
                return EvaluationResult.REFUTED, supporting, refuting

        return EvaluationResult.UNDETERMINED, supporting, refuting

    def _evaluate_implication(
        self,
        node: Implication,
        data: list[dict],
        resolver: Callable,
    ) -> tuple[EvaluationResult, list[int], list[int]]:
        """
        Evaluate implication: if P then Q.

        P→Q is violated only when P is true and Q is false.
        """
        supporting = []
        refuting = []

        for i, row in enumerate(data):
            p = self._evaluate_body(node.antecedent, row, resolver)
            q = self._evaluate_body(node.consequent, row, resolver)

            if p and not q:
                # Counterexample found
                refuting.append(i)
            elif p and q:
                # Supporting case
                supporting.append(i)
            # If P is false, the implication is vacuously true (not counted)

        if refuting:
            return EvaluationResult.REFUTED, supporting, refuting
        elif supporting:
            return EvaluationResult.SUPPORTED, supporting, refuting
        else:
            return EvaluationResult.UNDETERMINED, supporting, refuting

    def _evaluate_logical(
        self,
        node: LogicalExpr,
        data: list[dict],
        resolver: Callable,
    ) -> tuple[EvaluationResult, list[int], list[int]]:
        """Evaluate logical expression over all data."""
        supporting = []
        refuting = []

        for i, row in enumerate(data):
            result = self._evaluate_body(node, row, resolver)
            if result:
                supporting.append(i)
            else:
                refuting.append(i)

        # Default: treated as ALWAYS (all must be true)
        if not refuting:
            return EvaluationResult.SUPPORTED, supporting, refuting
        else:
            return EvaluationResult.REFUTED, supporting, refuting

    def _evaluate_comparison_over_data(
        self,
        node: Comparison,
        data: list[dict],
        resolver: Callable,
    ) -> tuple[EvaluationResult, list[int], list[int]]:
        """Evaluate simple comparison over all data (implicit ALWAYS)."""
        supporting = []
        refuting = []

        for i, row in enumerate(data):
            result = self._evaluate_comparison(node, row, resolver)
            if result:
                supporting.append(i)
            else:
                refuting.append(i)

        if not refuting:
            return EvaluationResult.SUPPORTED, supporting, refuting
        else:
            return EvaluationResult.REFUTED, supporting, refuting

    def _evaluate_body(
        self,
        node: Union[Comparison, LogicalExpr, Implication],
        row: dict,
        resolver: Callable,
    ) -> bool:
        """Evaluate expression body for a single row."""
        if isinstance(node, Comparison):
            return self._evaluate_comparison(node, row, resolver)
        elif isinstance(node, LogicalExpr):
            return self._evaluate_logical_expr(node, row, resolver)
        elif isinstance(node, Implication):
            p = self._evaluate_body(node.antecedent, row, resolver)
            q = self._evaluate_body(node.consequent, row, resolver)
            return (not p) or q  # P→Q ≡ ¬P ∨ Q
        return False

    def _evaluate_condition(
        self,
        node: Union[Comparison, LogicalExpr],
        row: dict,
        resolver: Callable,
    ) -> bool:
        """Evaluate a condition (for domain filtering)."""
        return self._evaluate_body(node, row, resolver)

    def _evaluate_comparison(
        self,
        node: Comparison,
        row: dict,
        resolver: Callable,
    ) -> bool:
        """Evaluate a comparison for a single row."""
        left = self._resolve_value(node.left, row, resolver)
        right = self._resolve_value(node.right, row, resolver)

        if left is None or right is None:
            return False  # Missing data

        try:
            if node.op == ComparisonOp.GT:
                return left > right
            elif node.op == ComparisonOp.LT:
                return left < right
            elif node.op == ComparisonOp.GTE:
                return left >= right
            elif node.op == ComparisonOp.LTE:
                return left <= right
            elif node.op == ComparisonOp.EQ:
                return left == right
            elif node.op == ComparisonOp.NEQ:
                return left != right
        except TypeError:
            return False

        return False

    def _evaluate_logical_expr(
        self,
        node: LogicalExpr,
        row: dict,
        resolver: Callable,
    ) -> bool:
        """Evaluate logical expression for a single row."""
        if node.op == LogicalOp.NOT:
            return not self._evaluate_body(node.operands[0], row, resolver)
        elif node.op == LogicalOp.AND:
            return all(self._evaluate_body(op, row, resolver) for op in node.operands)
        elif node.op == LogicalOp.OR:
            return any(self._evaluate_body(op, row, resolver) for op in node.operands)
        elif node.op == LogicalOp.XOR:
            results = [self._evaluate_body(op, row, resolver) for op in node.operands]
            return sum(results) == 1  # Exactly one true
        return False

    def _resolve_value(
        self,
        node: Union[Variable, Literal],
        row: dict,
        resolver: Callable,
    ) -> Any:
        """Resolve a variable or literal to a value."""
        if isinstance(node, Literal):
            return node.value
        elif isinstance(node, Variable):
            return resolver(node.name, row)
        return None

    def _calculate_confidence(
        self,
        result: EvaluationResult,
        supporting: EvaluationEvidence,
        refuting: EvaluationEvidence,
        hypothesis: Hypothesis,
    ) -> float:
        """Calculate confidence in the evaluation."""
        total = supporting.total_count
        if total == 0:
            return 0.0

        if result == EvaluationResult.SUPPORTED:
            # All supporting → high confidence
            return min(1.0, 0.5 + (supporting.matching_count / total) * 0.5)

        elif result == EvaluationResult.REFUTED:
            # Confidence in refutation based on strength of counterexamples
            refute_ratio = refuting.matching_count / total
            return min(1.0, 0.5 + refute_ratio * 0.5)

        return 0.5  # Undetermined

    def _generate_explanation(
        self,
        result: EvaluationResult,
        hypothesis: Hypothesis,
        supporting: EvaluationEvidence,
        refuting: EvaluationEvidence,
    ) -> str:
        """Generate human-readable explanation."""
        total = supporting.total_count

        if result == EvaluationResult.SUPPORTED:
            return (
                f"Hypothesis supported: {supporting.matching_count}/{total} "
                f"observations consistent with claim."
            )

        elif result == EvaluationResult.REFUTED:
            return (
                f"Hypothesis refuted: {refuting.matching_count}/{total} "
                f"counterexamples found. "
                f"Only {supporting.matching_count} observations support the claim."
            )

        elif result == EvaluationResult.UNDETERMINED:
            return "Insufficient data to evaluate hypothesis."

        return f"Evaluation result: {result.value}"

    def _check_needs_review(
        self,
        result: EvaluationResult,
        confidence: float,
        fallacies: list[Fallacy],
        hypothesis: Hypothesis,
    ) -> tuple[bool, Optional[str]]:
        """Check if LLM review is needed."""
        # Low confidence
        if confidence < 0.6:
            return True, "Low confidence evaluation - may need human interpretation"

        # Has fallacies
        errors = [f for f in fallacies if f.severity == "error"]
        if errors:
            return True, f"Logical issues detected: {errors[0].description}"

        # Partial result
        if result == EvaluationResult.PARTIAL:
            return True, "Mixed evidence - some conditions met, others not"

        return False, None

    def _check_structure(self, ast: Expression) -> list[Fallacy]:
        """Check AST structure for issues."""
        fallacies = []

        # Check for nested implications (complex logic that may be unclear)
        if self._has_nested_implications(ast):
            fallacies.append(Fallacy(
                type=FallacyType.CIRCULAR_REASONING,
                description="Nested implications may create circular or unclear logic",
                location="structure",
                severity="warning",
                suggestion="Consider breaking into separate, simpler hypotheses",
            ))

        return fallacies

    def _has_nested_implications(self, ast: Expression) -> bool:
        """Check for nested implications."""
        if isinstance(ast, Implication):
            return (
                isinstance(ast.antecedent, Implication) or
                isinstance(ast.consequent, Implication)
            )
        elif isinstance(ast, LogicalExpr):
            return any(self._has_nested_implications(op) for op in ast.operands)
        elif isinstance(ast, Quantified):
            return self._has_nested_implications(ast.body)
        return False

    def _check_fallacy_patterns(self, ast: Expression) -> list[Fallacy]:
        """Check for common fallacy patterns in the AST.

        Detects:
        1. Affirming the consequent: P→Q, Q ∴ P  (converse error)
        2. Denying the antecedent: P→Q, ¬P ∴ ¬Q (inverse error)
        3. False dichotomy: binary quantifiers on continuous variables
        4. Hasty generalization: universal quantifier (ALWAYS/NEVER/ALL/NONE)
           on a single comparison without domain restriction
        5. Overgeneralization: nested universal quantifiers
        """
        fallacies = []

        # Collect all implications and quantified statements
        implications = self._collect_nodes(ast, Implication)
        quantifieds = self._collect_nodes(ast, Quantified)

        # --- 1. Affirming the consequent ---
        # Detectable when the same variable appears as consequent in one
        # implication and as the sole antecedent in another.
        # e.g. "if [rain] > 0 then [wet] = 1" + "[wet] = 1" (bare assertion)
        # This is a structural warning about the implication direction.
        if len(implications) >= 2:
            consequent_vars = set()
            antecedent_vars = set()
            for imp in implications:
                consequent_vars.update(self._get_variables(imp.consequent))
                antecedent_vars.update(self._get_variables(imp.antecedent))
            # If a consequent variable is used as a standalone antecedent
            # in another implication, warn about directionality
            shared = consequent_vars & antecedent_vars
            if shared:
                fallacies.append(Fallacy(
                    type=FallacyType.AFFIRMING_CONSEQUENT,
                    description=(
                        f"Variables {shared} appear as both consequent and antecedent "
                        f"across implications — risk of affirming the consequent "
                        f"(P→Q, Q ∴ P is invalid)"
                    ),
                    location="implication chain",
                    severity="warning",
                    suggestion="Verify causal direction: does the consequent truly cause the antecedent?",
                ))

        # --- 2. Denying the antecedent ---
        # Detectable when an implication's antecedent is negated elsewhere.
        # e.g. "if [rain] > 0 then [wet] = 1" combined with "NOT [rain] > 0"
        for imp in implications:
            # Check if the antecedent is explicitly negated in the AST
            if self._contains_negation_of(ast, imp.antecedent):
                fallacies.append(Fallacy(
                    type=FallacyType.DENYING_ANTECEDENT,
                    description=(
                        f"The antecedent of an implication is negated — "
                        f"'if P then Q; not P' does not entail 'not Q'"
                    ),
                    location="implication + negation",
                    severity="warning",
                    suggestion="The consequent may still be true for other reasons.",
                ))

        # --- 3. False dichotomy ---
        # Detected when a hypothesis uses only = comparisons on what looks
        # like a continuous variable, or uses XOR with exactly 2 options.
        logical_exprs = self._collect_nodes(ast, LogicalExpr)
        for expr in logical_exprs:
            if expr.op == LogicalOp.XOR and len(expr.operands) == 2:
                # Binary XOR — might be a false dichotomy
                fallacies.append(Fallacy(
                    type=FallacyType.FALSE_DICHOTOMY,
                    description=(
                        "XOR with exactly 2 options — there may be additional "
                        "possibilities not considered"
                    ),
                    location="logical expression",
                    severity="warning",
                    suggestion="Consider whether a third option exists.",
                ))

        # Check for "NEVER X AND NEVER NOT-X" pattern (exhaustive negation)
        never_quants = [q for q in quantifieds
                        if q.quantifier in (Quantifier.NEVER, Quantifier.NONE)]
        if len(never_quants) >= 2:
            # Check if two NEVER quantifiers cover complementary conditions
            # on the same variable
            for i, q1 in enumerate(never_quants):
                q1_vars = self._get_variables(q1.body)
                for q2 in never_quants[i + 1:]:
                    q2_vars = self._get_variables(q2.body)
                    if q1_vars & q2_vars:
                        fallacies.append(Fallacy(
                            type=FallacyType.FALSE_DICHOTOMY,
                            description=(
                                f"Multiple NEVER constraints on overlapping "
                                f"variables {q1_vars & q2_vars} — may create "
                                f"an impossible condition (false dichotomy)"
                            ),
                            location="quantifier pair",
                            severity="warning",
                            suggestion="Check if these constraints can be simultaneously satisfied.",
                        ))

        # --- 4. Hasty generalization ---
        # Universal quantifier without domain restriction on a bare comparison
        for q in quantifieds:
            if q.quantifier in (Quantifier.ALWAYS, Quantifier.NEVER,
                                Quantifier.ALL, Quantifier.NONE):
                if q.domain is None and isinstance(q.body, Comparison):
                    fallacies.append(Fallacy(
                        type=FallacyType.HASTY_GENERALIZATION,
                        description=(
                            f"Universal claim ({q.quantifier.value.upper()}) "
                            f"without domain restriction (WHEN clause) — "
                            f"may not hold across all conditions"
                        ),
                        location="quantifier",
                        severity="warning",
                        suggestion="Add a WHEN clause to restrict the domain.",
                    ))

        # --- 5. Overgeneralization: nested quantifiers ---
        for q in quantifieds:
            if isinstance(q.body, Quantified):
                fallacies.append(Fallacy(
                    type=FallacyType.OVERGENERALIZATION,
                    description="Nested quantifiers detected — hypothesis may be overly general",
                    location="nested quantifier",
                    severity="warning",
                    suggestion="Consider splitting into separate, simpler hypotheses.",
                ))

        return fallacies

    def _collect_nodes(self, ast: Expression, node_type: type) -> list:
        """Recursively collect all nodes of a given type from the AST."""
        found = []
        if isinstance(ast, node_type):
            found.append(ast)
        if isinstance(ast, LogicalExpr):
            for op in ast.operands:
                found.extend(self._collect_nodes(op, node_type))
        elif isinstance(ast, Implication):
            found.extend(self._collect_nodes(ast.antecedent, node_type))
            found.extend(self._collect_nodes(ast.consequent, node_type))
        elif isinstance(ast, Quantified):
            found.extend(self._collect_nodes(ast.body, node_type))
            if ast.domain:
                found.extend(self._collect_nodes(ast.domain.condition, node_type))
        return found

    def _get_variables(self, ast: Expression) -> set:
        """Extract all variable names from an AST node."""
        if isinstance(ast, Variable):
            return {ast.name}
        elif isinstance(ast, Comparison):
            return self._get_variables(ast.left) | self._get_variables(ast.right)
        elif isinstance(ast, LogicalExpr):
            result = set()
            for op in ast.operands:
                result |= self._get_variables(op)
            return result
        elif isinstance(ast, Implication):
            return self._get_variables(ast.antecedent) | self._get_variables(ast.consequent)
        elif isinstance(ast, Quantified):
            result = self._get_variables(ast.body)
            if ast.over:
                result.add(ast.over.name)
            return result
        return set()

    def _contains_negation_of(self, ast: Expression, target: Expression) -> bool:
        """Check if the AST contains a NOT wrapping something similar to target."""
        if isinstance(ast, LogicalExpr) and ast.op == LogicalOp.NOT:
            # Check if the negated operand shares variables with target
            neg_vars = self._get_variables(ast.operands[0])
            target_vars = self._get_variables(target)
            if neg_vars and neg_vars == target_vars:
                return True
        # Recurse
        if isinstance(ast, LogicalExpr):
            return any(self._contains_negation_of(op, target) for op in ast.operands)
        elif isinstance(ast, Implication):
            return (self._contains_negation_of(ast.antecedent, target) or
                    self._contains_negation_of(ast.consequent, target))
        elif isinstance(ast, Quantified):
            return self._contains_negation_of(ast.body, target)
        return False


# =============================================================================
# Convenience Functions
# =============================================================================

def parse_and_evaluate(
    hypothesis_text: str,
    data: list[dict],
    variable_context: Optional[dict[str, str]] = None,
) -> HypothesisEvaluation:
    """
    Convenience function to parse and evaluate in one step.
    """
    engine = LogicEngine()
    hypothesis = engine.parse(hypothesis_text)
    return engine.evaluate(hypothesis, data, variable_context)


def validate_hypothesis(hypothesis_text: str) -> dict:
    """
    Validate a hypothesis without data.

    Returns dict with:
    - valid: bool
    - hypothesis: parsed structure
    - fallacies: list of issues
    - variables: list of referenced variables
    """
    engine = LogicEngine()
    hypothesis = engine.parse(hypothesis_text)
    fallacies = engine.validate(hypothesis)

    return {
        "valid": len([f for f in fallacies if f.severity == "error"]) == 0,
        "hypothesis": hypothesis.to_dict(),
        "fallacies": [
            {
                "type": f.type.value,
                "description": f.description,
                "severity": f.severity,
                "suggestion": f.suggestion,
            }
            for f in fallacies
        ],
        "variables": hypothesis.variables,
        "is_falsifiable": hypothesis.is_falsifiable,
    }

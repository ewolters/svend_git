"""
Code Complexity Analyzer

Calculates code complexity metrics:
- Cyclomatic Complexity (McCabe)
- Cognitive Complexity
- Lines of Code (LOC, SLOC)
- Maintainability Index

Supports Python code analysis using AST.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FunctionMetrics:
    """Metrics for a single function."""
    name: str
    line: int
    cyclomatic: int
    cognitive: int
    loc: int
    parameters: int
    returns: int  # Number of return statements

    @property
    def risk(self) -> str:
        """Risk assessment based on cyclomatic complexity."""
        if self.cyclomatic <= 5:
            return "low"
        elif self.cyclomatic <= 10:
            return "moderate"
        elif self.cyclomatic <= 20:
            return "high"
        else:
            return "very high"


@dataclass
class ComplexityResult:
    """Results from complexity analysis."""
    total_loc: int                  # Total lines
    sloc: int                       # Source lines (non-blank, non-comment)
    comment_lines: int
    blank_lines: int

    function_count: int
    class_count: int

    total_cyclomatic: int
    avg_cyclomatic: float
    max_cyclomatic: int

    total_cognitive: int
    avg_cognitive: float

    maintainability_index: float    # 0-100

    functions: list[FunctionMetrics] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        """Overall risk assessment."""
        if self.avg_cyclomatic <= 5 and self.max_cyclomatic <= 10:
            return "Low - Well maintained"
        elif self.avg_cyclomatic <= 10 and self.max_cyclomatic <= 20:
            return "Moderate - Consider refactoring complex functions"
        elif self.avg_cyclomatic <= 15:
            return "High - Refactoring recommended"
        else:
            return "Very High - Significant refactoring needed"

    def to_dict(self) -> dict:
        return {
            "loc": self.total_loc,
            "sloc": self.sloc,
            "comment_lines": self.comment_lines,
            "blank_lines": self.blank_lines,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "cyclomatic": {
                "total": self.total_cyclomatic,
                "average": round(self.avg_cyclomatic, 2),
                "max": self.max_cyclomatic,
            },
            "cognitive": {
                "total": self.total_cognitive,
                "average": round(self.avg_cognitive, 2),
            },
            "maintainability_index": round(self.maintainability_index, 1),
            "risk_level": self.risk_level,
            "functions": [
                {
                    "name": f.name,
                    "line": f.line,
                    "cyclomatic": f.cyclomatic,
                    "cognitive": f.cognitive,
                    "loc": f.loc,
                    "parameters": f.parameters,
                    "risk": f.risk,
                }
                for f in self.functions
            ],
        }


class CodeComplexity:
    """
    Analyze Python code complexity.

    Usage:
        analyzer = CodeComplexity()
        result = analyzer.analyze(code_string)
        print(result.avg_cyclomatic)
        print(result.risk_level)
    """

    def analyze(self, code: str) -> ComplexityResult:
        """Analyze code and return complexity metrics."""
        # Basic line counts
        lines = code.split('\n')
        total_loc = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        sloc = total_loc - blank_lines - comment_lines

        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            # Return basic metrics if parse fails
            return ComplexityResult(
                total_loc=total_loc,
                sloc=sloc,
                comment_lines=comment_lines,
                blank_lines=blank_lines,
                function_count=0,
                class_count=0,
                total_cyclomatic=0,
                avg_cyclomatic=0,
                max_cyclomatic=0,
                total_cognitive=0,
                avg_cognitive=0,
                maintainability_index=0,
                functions=[],
            )

        # Count classes and analyze functions
        class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        functions = self._analyze_functions(tree, code)

        function_count = len(functions)

        # Aggregate metrics
        if functions:
            total_cyclomatic = sum(f.cyclomatic for f in functions)
            avg_cyclomatic = total_cyclomatic / function_count
            max_cyclomatic = max(f.cyclomatic for f in functions)
            total_cognitive = sum(f.cognitive for f in functions)
            avg_cognitive = total_cognitive / function_count
        else:
            total_cyclomatic = 1  # Module-level
            avg_cyclomatic = 1
            max_cyclomatic = 1
            total_cognitive = 0
            avg_cognitive = 0

        # Maintainability Index
        # MI = 171 - 5.2*ln(V) - 0.23*G - 16.2*ln(L)
        # Simplified: MI = 171 - 5.2*ln(sloc) - 0.23*cyclomatic - 16.2*ln(sloc)
        import math
        if sloc > 0:
            halstead_volume = sloc * math.log2(max(sloc, 1))  # Simplified
            mi = 171 - 5.2 * math.log(halstead_volume + 1) - 0.23 * total_cyclomatic - 16.2 * math.log(sloc + 1)
            mi = max(0, min(100, mi * 100 / 171))  # Normalize to 0-100
        else:
            mi = 100

        return ComplexityResult(
            total_loc=total_loc,
            sloc=sloc,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            function_count=function_count,
            class_count=class_count,
            total_cyclomatic=total_cyclomatic,
            avg_cyclomatic=avg_cyclomatic,
            max_cyclomatic=max_cyclomatic,
            total_cognitive=total_cognitive,
            avg_cognitive=avg_cognitive,
            maintainability_index=mi,
            functions=functions,
        )

    def _analyze_functions(self, tree: ast.AST, code: str) -> list[FunctionMetrics]:
        """Analyze all functions in the AST."""
        functions = []
        lines = code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Get function extent
                start_line = node.lineno
                end_line = self._get_end_line(node)
                loc = end_line - start_line + 1

                # Count parameters
                params = len(node.args.args) + len(node.args.posonlyargs) + len(node.args.kwonlyargs)
                if node.args.vararg:
                    params += 1
                if node.args.kwarg:
                    params += 1

                # Count returns
                returns = sum(1 for n in ast.walk(node) if isinstance(n, ast.Return))

                # Calculate complexities
                cyclomatic = self._cyclomatic_complexity(node)
                cognitive = self._cognitive_complexity(node)

                functions.append(FunctionMetrics(
                    name=node.name,
                    line=start_line,
                    cyclomatic=cyclomatic,
                    cognitive=cognitive,
                    loc=loc,
                    parameters=params,
                    returns=returns,
                ))

        return functions

    def _get_end_line(self, node: ast.AST) -> int:
        """Get the last line of a node."""
        end_line = node.lineno
        for child in ast.walk(node):
            if hasattr(child, 'lineno'):
                end_line = max(end_line, child.lineno)
            if hasattr(child, 'end_lineno') and child.end_lineno:
                end_line = max(end_line, child.end_lineno)
        return end_line

    def _cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function."""
        # CC = E - N + 2P (edges - nodes + 2*connected components)
        # Simplified: CC = 1 + decision points

        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Decision points
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1  # List/dict/set comprehension
            # Boolean operators add paths
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _cognitive_complexity(self, node: ast.AST, nesting: int = 0) -> int:
        """
        Calculate cognitive complexity.

        Cognitive complexity penalizes:
        - Nesting (more deeply nested = harder to understand)
        - Breaks in linear flow (else, elif, etc.)
        """
        complexity = 0

        for child in ast.iter_child_nodes(node):
            # Structural increment + nesting penalty
            if isinstance(child, (ast.If, ast.For, ast.While, ast.AsyncFor)):
                complexity += 1 + nesting
                complexity += self._cognitive_complexity(child, nesting + 1)
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1 + nesting
                complexity += self._cognitive_complexity(child, nesting + 1)
            # Else/elif adds to cognitive load
            elif isinstance(child, ast.orelse) if hasattr(ast, 'orelse') else False:
                complexity += 1
            # Boolean operators
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            # Recursion
            elif isinstance(child, (ast.Call,)):
                # Check if recursive call
                if isinstance(child.func, ast.Name):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if child.func.id == node.name:
                            complexity += 1
            else:
                complexity += self._cognitive_complexity(child, nesting)

        return complexity


def analyze_complexity(code: str) -> dict:
    """Convenience function for quick analysis."""
    analyzer = CodeComplexity()
    result = analyzer.analyze(code)
    return result.to_dict()


# CLI
def main():
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Analyze Python code complexity")
    parser.add_argument("input", type=Path, help="Python file to analyze")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--functions", action="store_true", help="Show per-function breakdown")

    args = parser.parse_args()

    code = args.input.read_text()
    analyzer = CodeComplexity()
    result = analyzer.analyze(code)

    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Complexity Analysis: {args.input.name}")
        print("=" * 60)
        print(f"Lines of Code: {result.total_loc} (SLOC: {result.sloc})")
        print(f"Functions: {result.function_count}, Classes: {result.class_count}")
        print("-" * 60)
        print(f"Cyclomatic Complexity:")
        print(f"  Total: {result.total_cyclomatic}")
        print(f"  Average: {result.avg_cyclomatic:.2f}")
        print(f"  Max: {result.max_cyclomatic}")
        print(f"Cognitive Complexity:")
        print(f"  Total: {result.total_cognitive}")
        print(f"  Average: {result.avg_cognitive:.2f}")
        print("-" * 60)
        print(f"Maintainability Index: {result.maintainability_index:.1f}/100")
        print(f"Risk Level: {result.risk_level}")

        if args.functions and result.functions:
            print("\n" + "=" * 60)
            print("Per-Function Breakdown:")
            print("-" * 60)

            # Sort by cyclomatic (highest first)
            sorted_funcs = sorted(result.functions, key=lambda f: -f.cyclomatic)
            for f in sorted_funcs:
                risk_icon = {"low": "✓", "moderate": "~", "high": "!", "very high": "!!"}
                print(f"{risk_icon.get(f.risk, '?')} {f.name} (line {f.line})")
                print(f"    Cyclomatic: {f.cyclomatic}, Cognitive: {f.cognitive}, LOC: {f.loc}")


if __name__ == "__main__":
    main()

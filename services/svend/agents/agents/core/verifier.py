"""
Code Verifier

Ensemble verification system:
1. Syntax check (can it parse?)
2. Lint check (style/common errors)
3. Test generation and execution
4. Intent alignment check
"""

import ast
from dataclasses import dataclass, field
from enum import Enum

from .executor import CodeExecutor, ExecutionResult


class VerificationStatus(Enum):
    """Overall verification status."""
    PASSED = "passed"
    WARNINGS = "warnings"
    FAILED = "failed"


@dataclass
class VerificationResult:
    """Result of verification checks."""

    status: VerificationStatus
    checks: dict[str, bool] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    generated_tests: str | None = None
    test_results: ExecutionResult | None = None

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED

    def summary(self) -> str:
        lines = [f"Status: {self.status.value}"]
        for check, passed in self.checks.items():
            icon = "OK" if passed else "FAIL"
            lines.append(f"  [{icon}] {check}")
        if self.messages:
            lines.append("Messages:")
            for msg in self.messages:
                lines.append(f"  - {msg}")
        return "\n".join(lines)


class CodeVerifier:
    """
    Multi-stage code verification.

    Checks:
    1. Syntax - Does it parse?
    2. Lint - Common issues?
    3. Tests - Does it work?
    4. Intent - Does it match what was asked?
    """

    def __init__(self, executor: CodeExecutor | None = None, llm=None):
        self.executor = executor or CodeExecutor()
        self.llm = llm

    def verify(self, code: str, intent: str = None) -> VerificationResult:
        """Run all verification checks."""
        checks = {}
        messages = []

        # 1. Syntax check
        syntax_ok, syntax_msg = self._check_syntax(code)
        checks["syntax"] = syntax_ok
        if not syntax_ok:
            messages.append(f"Syntax error: {syntax_msg}")

        # 2. Lint check (only if syntax passes)
        if syntax_ok:
            lint_ok, lint_msg = self._check_lint(code)
            checks["lint"] = lint_ok
            if not lint_ok:
                messages.append(f"Lint issues: {lint_msg}")
        else:
            checks["lint"] = False

        # 3. Basic execution check (does it run without error?)
        if syntax_ok:
            exec_ok, exec_msg = self._check_execution(code)
            checks["execution"] = exec_ok
            if not exec_ok:
                messages.append(f"Execution error: {exec_msg}")
        else:
            checks["execution"] = False

        # 4. Intent alignment (if intent provided and LLM available)
        if intent and self.llm:
            align_ok, align_msg = self._check_intent_alignment(code, intent)
            checks["intent_alignment"] = align_ok
            if not align_ok:
                messages.append(f"Intent mismatch: {align_msg}")

        # Determine overall status
        critical_checks = ["syntax", "execution"]
        critical_passed = all(checks.get(c, False) for c in critical_checks)

        if critical_passed and all(checks.values()):
            status = VerificationStatus.PASSED
        elif critical_passed:
            status = VerificationStatus.WARNINGS
        else:
            status = VerificationStatus.FAILED

        return VerificationResult(
            status=status,
            checks=checks,
            messages=messages,
        )

    def _check_syntax(self, code: str) -> tuple[bool, str]:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    def _check_lint(self, code: str) -> tuple[bool, str]:
        """Run linter on code."""
        result = self.executor.lint_code(code)
        if result.success:
            return True, ""
        return False, result.stderr or result.error or "Lint failed"

    def _check_execution(self, code: str) -> tuple[bool, str]:
        """Check if code runs without error."""
        # Wrap in try/except to catch runtime errors
        wrapped = f"""
try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
    print("__EXECUTION_OK__")
except Exception as e:
    print(f"__EXECUTION_ERROR__: {{e}}")
"""
        result = self.executor.execute_python(wrapped)

        if "__EXECUTION_OK__" in result.stdout:
            return True, ""
        elif "__EXECUTION_ERROR__" in result.stdout:
            error = result.stdout.split("__EXECUTION_ERROR__:")[-1].strip()
            return False, error
        else:
            return False, result.error or result.stderr or "Unknown error"

    def _check_intent_alignment(self, code: str, intent: str) -> tuple[bool, str]:
        """Check if code aligns with stated intent."""
        if not self.llm:
            return True, "No LLM available for intent check"

        # This would use the LLM to verify alignment
        # For now, do a simple keyword check
        intent_words = set(intent.lower().split())
        code_lower = code.lower()

        # Check if key intent words appear in code (comments, names, etc.)
        matches = sum(1 for w in intent_words if w in code_lower and len(w) > 3)
        score = matches / len(intent_words) if intent_words else 0

        if score > 0.3:
            return True, ""
        return False, "Code may not match the stated intent"

    def generate_tests(self, code: str, intent: str = None) -> str:
        """Generate basic tests for the code."""
        # Parse code to find functions
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return "# Could not parse code to generate tests"

        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        if not functions:
            return "# No functions found to test"

        test_lines = ["import unittest", ""]

        for func in functions:
            func_name = func.name
            # Skip private functions
            if func_name.startswith("_"):
                continue

            test_lines.append(f"class Test{func_name.title()}(unittest.TestCase):")
            test_lines.append(f"    def test_{func_name}_exists(self):")
            test_lines.append(f"        self.assertTrue(callable({func_name}))")
            test_lines.append("")

        test_lines.append("if __name__ == '__main__':")
        test_lines.append("    unittest.main()")

        return "\n".join(test_lines)

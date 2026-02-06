"""
Coding Agent

Code generation with:
- Intent tracking (anti-drift)
- Ensemble verification
- Iterative refinement
"""

import json
import re
from dataclasses import dataclass, field
from typing import Literal

import sys
sys.path.insert(0, '/home/eric/Desktop/agents')

from core.intent import IntentTracker, Action, AlignmentStatus
from core.executor import CodeExecutor
from core.verifier import CodeVerifier, VerificationStatus
from core.reasoning import CodeReasoner, QualityAssessment
from core.context import CodebaseReader, ProjectContext, format_context_for_prompt
from coder.security import SecurityAnalyzer, SecurityReport, check_security


@dataclass
class CodingTask:
    """A coding task with intent and constraints."""
    description: str
    language: str = "python"
    constraints: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


@dataclass
class CodingResult:
    """Result of a coding task."""
    code: str
    verified: bool
    verification_summary: str
    intent_alignment: float
    iterations: int
    reasoning: str
    quality_assessment: QualityAssessment = None  # Bayesian quality breakdown
    execution_output: str = ""  # Demo output from running the code
    execution_error: str = ""   # Any errors from execution
    security_report: SecurityReport = None  # Security analysis results

    def save(self, path: str) -> str:
        """Save code to file."""
        from pathlib import Path
        p = Path(path)
        p.write_text(self.code)
        return str(p.absolute())

    def human_review_checklist(self) -> list[str]:
        """Generate checklist of items a human should review."""
        items = []

        # Based on verification results
        if not self.verified:
            items.append("Code did not pass all verification checks - review errors")

        # Based on confidence
        if self.quality_assessment:
            conf = self.quality_assessment.overall_confidence
            if conf < 0.5:
                items.append(f"Low confidence ({conf:.0%}) - thorough review recommended")
            elif conf < 0.8:
                items.append(f"Medium confidence ({conf:.0%}) - spot check recommended")

            # Check individual dimensions
            if self.quality_assessment.dimensions.get("execution", 1.0) < 0.5:
                items.append("Execution tests may be incomplete - test edge cases")
            if self.quality_assessment.dimensions.get("lint", 1.0) < 0.7:
                items.append("Some lint warnings present - review code style")

        # Based on iterations
        if self.iterations > 2:
            items.append(f"Required {self.iterations} iterations - may indicate complexity")

        # Based on alignment
        if self.intent_alignment < 0.8:
            items.append("Intent alignment < 80% - verify it does what you asked")

        # Security issues
        if self.security_report and self.security_report.issues:
            critical = self.security_report.critical_count
            high = self.security_report.high_count
            if critical > 0:
                items.insert(0, f"CRITICAL: {critical} critical security issue(s) - DO NOT deploy")
            if high > 0:
                items.append(f"Security: {high} high-severity issue(s) require review")

        # Always recommend
        items.append("Review edge cases and error handling")
        if not self.security_report or self.security_report.passed:
            items.append("Verify security implications if handling user input")

        return items

    def qa_report(self) -> str:
        """Generate full QA report."""
        lines = [
            "=" * 60,
            "CODE QUALITY ASSURANCE REPORT",
            "=" * 60,
            "",
            "## Summary",
            f"  Verified: {'YES' if self.verified else 'NO'}",
            f"  Iterations: {self.iterations}",
            f"  Intent Alignment: {self.intent_alignment:.0%}",
        ]

        if self.quality_assessment:
            lines.append(f"  Overall Confidence: {self.quality_assessment.overall_confidence:.0%}")

        lines.extend([
            "",
            "## Verification Results",
            self.verification_summary,
            "",
        ])

        # Execution demo
        if self.execution_output:
            lines.extend([
                "## Execution Demo",
                "```",
                self.execution_output[:1000],
                "```",
                "",
            ])

        if self.execution_error:
            lines.extend([
                "## Execution Errors",
                "```",
                self.execution_error,
                "```",
                "",
            ])

        # Security Analysis
        if self.security_report:
            status = "PASSED" if self.security_report.passed else "FAILED"
            lines.extend([
                f"## Security Analysis: {status}",
                f"  Score: {self.security_report.score:.0f}/100",
                f"  Issues: {len(self.security_report.issues)} "
                f"(Critical: {self.security_report.critical_count}, "
                f"High: {self.security_report.high_count})",
                "",
            ])
            if self.security_report.issues:
                for issue in self.security_report.issues[:5]:
                    lines.append(f"  {issue}")
                    if issue.recommendation:
                        lines.append(f"      Fix: {issue.recommendation}")
                if len(self.security_report.issues) > 5:
                    lines.append(f"  ... and {len(self.security_report.issues) - 5} more")
                lines.append("")

        # Human review section
        lines.extend([
            "## Human Review Checklist",
            "",
        ])
        for item in self.human_review_checklist():
            lines.append(f"  [ ] {item}")

        # Quality breakdown
        if self.quality_assessment and self.quality_assessment.dimensions:
            lines.extend([
                "",
                "## Quality Dimensions",
            ])
            for dim, score in self.quality_assessment.dimensions.items():
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                lines.append(f"  {dim:12} [{bar}] {score:.0%}")

            if self.quality_assessment.recommendations:
                lines.extend([
                    "",
                    "## Recommendations",
                ])
                for rec in self.quality_assessment.recommendations:
                    lines.append(f"  - {rec}")

        lines.extend([
            "",
            "## Code",
            "```python",
            self.code,
            "```",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Export as markdown."""
        return self.qa_report()


class CodingAgent:
    """
    Code generation agent with quality controls.

    Flow:
    1. Parse intent from user request
    2. Generate code with LLM (with codebase context)
    3. Verify code (syntax, lint, execution)
    4. Assess quality with Bayesian reasoning
    5. Check intent alignment
    6. Iterate if needed
    """

    SYSTEM_PROMPT = """You are a precise coding assistant. Your job is to write code that exactly matches the user's request.

Rules:
1. Write ONLY the code requested - no extras
2. Include docstrings and type hints
3. Keep it simple and readable
4. Do not add features not explicitly requested
5. If unclear, implement the simplest interpretation

Output format:
```python
# Your code here
```

After the code, briefly explain what it does in 1-2 sentences."""

    def __init__(self, llm=None, max_iterations: int = 3, project_path: str = None):
        self.llm = llm
        self.max_iterations = max_iterations
        self.intent_tracker = IntentTracker(llm=llm)  # Enable LLM-based alignment
        self.executor = CodeExecutor()
        self.verifier = CodeVerifier(self.executor)
        self.reasoner = CodeReasoner(prior=0.5)

        # Load project context if provided
        self.project_context = None
        if project_path:
            reader = CodebaseReader()
            try:
                self.project_context = reader.read_project(project_path)
            except Exception as e:
                pass  # Continue without context

    def _llm_generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text from LLM, handling different API styles."""
        if hasattr(self.llm, 'generate'):
            return self.llm.generate(prompt, max_tokens=max_tokens)
        elif hasattr(self.llm, 'complete'):
            return self.llm.complete(prompt, max_tokens=max_tokens)
        else:
            raise ValueError("LLM must have generate() or complete() method")

    def run(self, task: CodingTask) -> CodingResult:
        """Execute a coding task."""
        # Set intent
        constraints = task.constraints + [
            "Do not add unrequested features",
            "Do not refactor beyond what's asked",
        ]

        self.intent_tracker.set_intent(
            raw_input=task.description,
            parsed_goal=task.description,
            constraints=constraints,
        )

        # Generate and verify
        code = None
        verification = None
        reasoning = ""
        quality_assessment = None

        for iteration in range(self.max_iterations):
            # Generate code
            if code is None:
                code, reasoning = self._generate_code(task)
            else:
                # Refine based on verification feedback + quality assessment
                feedback = verification.messages if verification else []
                if quality_assessment and quality_assessment.recommendations:
                    feedback.extend(quality_assessment.recommendations)
                code, reasoning = self._refine_code(task, code, feedback)

            # Check alignment before verification
            status, score, alignment_msg = self.intent_tracker.check_alignment(
                f"Generated code: {reasoning}"
            )

            if status == AlignmentStatus.OFF_TRACK:
                reasoning = f"Drift detected: {alignment_msg}. Regenerating..."
                code = None
                continue

            # Verify
            verification = self.verifier.verify(code, task.description)

            # Build verification results dict for Bayesian assessment
            verification_results = {
                "syntax": verification.checks.get("syntax", False),
                "lint": verification.checks.get("lint", False),
                "execution": verification.checks.get("execution", False),
                "execution_error": verification.messages[0] if verification.messages else "",
            }

            # Bayesian quality assessment
            quality_assessment = self.reasoner.assess_code(
                code, verification_results, score
            )

            # Record action with confidence
            action = Action(
                id=f"gen_{iteration}",
                description=f"Generated code (iteration {iteration + 1})",
                action_type="code_generation",
                content=code,
                alignment_score=score,
                reasoning=f"{reasoning}\nConfidence: {quality_assessment.overall_confidence:.0%}",
            )
            self.intent_tracker.record_action(action)

            # Check if we're done - use Bayesian confidence threshold
            # High confidence (>0.8) = likely correct even with minor issues
            if verification.passed or quality_assessment.overall_confidence > 0.85:
                break

            # Check if we're drifting
            if self.intent_tracker.is_drifting():
                reasoning += f"\n\nWarning: {self.intent_tracker.get_course_correction()}"

            # Low confidence = regenerate
            if quality_assessment.overall_confidence < 0.3:
                code = None

        # Build final summary with Bayesian assessment
        summary_parts = []
        if verification:
            summary_parts.append(verification.summary())
        if quality_assessment:
            summary_parts.append(f"Quality confidence: {quality_assessment.overall_confidence:.0%}")

        # Run demo execution to show output
        exec_output, exec_error = self._run_demo(code or "", task)

        # Run security analysis
        security_report = None
        if code:
            security_analyzer = SecurityAnalyzer()
            security_report = security_analyzer.analyze(code)
            if not security_report.passed:
                summary_parts.append(f"Security: {security_report.critical_count}C/{security_report.high_count}H issues")

        return CodingResult(
            code=code or "",
            verified=verification.passed if verification else False,
            verification_summary=" | ".join(summary_parts) if summary_parts else "No verification",
            intent_alignment=self.intent_tracker.current_intent.alignment_score,
            iterations=iteration + 1,
            reasoning=reasoning,
            quality_assessment=quality_assessment,
            execution_output=exec_output,
            execution_error=exec_error,
            security_report=security_report,
        )

    def _run_demo(self, code: str, task: CodingTask) -> tuple[str, str]:
        """Run code with demo inputs to show it works."""
        if not code.strip():
            return "", "No code to execute"

        # Build demo script that imports and runs the code
        demo_script = f'''
{code}

# === Demo Execution ===
import inspect

# Find callable functions/classes defined in the code
defined_names = [name for name, obj in list(locals().items())
                 if callable(obj) and not name.startswith("_")]

if defined_names:
    print(f"Defined: {{', '.join(defined_names)}}")

    # Try to run the first function with sample inputs
    for name in defined_names[:1]:
        obj = locals()[name]
        if callable(obj):
            sig = inspect.signature(obj)
            params = list(sig.parameters.keys())

            # Generate sample inputs based on parameter names
            sample_args = []
            for p in params:
                p_lower = p.lower()
                if "str" in p_lower or "text" in p_lower or "s" == p_lower:
                    sample_args.append('"hello"')
                elif "list" in p_lower or "arr" in p_lower:
                    sample_args.append("[1, 2, 3]")
                elif "dict" in p_lower:
                    sample_args.append("{{}}")
                elif "n" == p_lower or "num" in p_lower or "int" in p_lower:
                    sample_args.append("5")
                elif "x" == p_lower or "y" == p_lower:
                    sample_args.append("10")
                else:
                    sample_args.append("None")

            if params:
                args_str = ", ".join(sample_args)
                print(f"\\nDemo: {{name}}({{args_str}})")
                try:
                    result = eval(f"{{name}}({{args_str}})")
                    print(f"Result: {{result}}")
                except Exception as e:
                    print(f"Demo error: {{e}}")
            else:
                print(f"\\nDemo: {{name}}()")
                try:
                    result = obj()
                    print(f"Result: {{result}}")
                except Exception as e:
                    print(f"Demo error: {{e}}")
else:
    print("No functions defined - code may be a script or class")
'''

        try:
            result = self.executor.execute_python(demo_script)
            return result.stdout, result.stderr or result.error or ""
        except Exception as e:
            return "", str(e)

    def _generate_code(self, task: CodingTask) -> tuple[str, str]:
        """Generate initial code."""
        if self.llm is None:
            return self._mock_generate(task)

        # Build context section if we have project context
        context_section = ""
        if self.project_context:
            context_section = f"""
## Project Context
{format_context_for_prompt(self.project_context, task.description, max_tokens=1500)}

Use the patterns and conventions from this codebase.
"""

        prompt = f"""{self.SYSTEM_PROMPT}
{context_section}
Task: {task.description}

Constraints:
{chr(10).join(f'- {c}' for c in task.constraints) or '- None'}

Examples:
{chr(10).join(task.examples) or 'None provided'}

Write the code:"""

        response = self._llm_generate(prompt, max_tokens=1000)
        code = self._extract_code(response)
        reasoning = self._extract_reasoning(response)
        return code, reasoning

    def _refine_code(self, task: CodingTask, current_code: str,
                     issues: list[str]) -> tuple[str, str]:
        """Refine code based on verification issues."""
        if self.llm is None:
            return current_code, "Mock refinement"

        prompt = f"""{self.SYSTEM_PROMPT}

Original task: {task.description}

Current code:
```python
{current_code}
```

Issues found:
{chr(10).join(f'- {issue}' for issue in issues)}

Fix the issues while staying true to the original task. Do not add new features.

Write the fixed code:"""

        response = self._llm_generate(prompt, max_tokens=1000)
        code = self._extract_code(response)
        reasoning = self._extract_reasoning(response)

        return code, reasoning

    def _extract_code(self, response: str) -> str:
        """Extract code block from LLM response."""
        # Look for ```python ... ``` blocks
        pattern = r"```python\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Fall back to ``` ... ``` blocks
        pattern = r"```\s*(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # No code blocks, return as-is (might be just code)
        return response.strip()

    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning/explanation from LLM response."""
        # Remove code blocks
        text = re.sub(r"```.*?```", "", response, flags=re.DOTALL)
        return text.strip()

    def _mock_generate(self, task: CodingTask) -> tuple[str, str]:
        """Generate mock code for testing without LLM."""
        desc = task.description.lower()

        if "fibonacci" in desc:
            code = '''def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
'''
        elif "factorial" in desc:
            code = '''def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''
        elif "sort" in desc:
            code = '''def sort_list(items: list) -> list:
    """Sort a list in ascending order."""
    return sorted(items)
'''
        elif "hello" in desc or "greet" in desc:
            code = '''def greet(name: str) -> str:
    """Return a greeting for the given name."""
    return f"Hello, {name}!"
'''
        elif "encrypt" in desc or "cipher" in desc:
            # Extract charset from description if provided
            import re
            charset_match = re.search(r'\[([^\]]+)\]', task.description)
            if charset_match:
                chars = [c.strip().strip("'\"") for c in charset_match.group(1).split(',')]
                charset = ''.join(chars)
            else:
                charset = 'BbLlSsWw12045'  # default

            code = f'''import random

CHARSET = "{charset}"

def encrypt(plaintext: str, seed: int = None) -> str:
    """
    Encrypt plaintext using a substitution cipher with charset: {charset}

    Each character is mapped to a character from the charset based on
    a seeded random shuffle for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    # Create mapping from all printable ASCII to charset
    shuffled = list(CHARSET * ((128 // len(CHARSET)) + 1))
    random.shuffle(shuffled)

    encrypted = []
    for char in plaintext:
        idx = ord(char) % len(CHARSET)
        encrypted.append(shuffled[idx])

    return ''.join(encrypted)


def decrypt(ciphertext: str, seed: int = None) -> str:
    """
    Decrypt ciphertext encrypted with the same seed.
    """
    if seed is not None:
        random.seed(seed)

    # Recreate the same mapping
    shuffled = list(CHARSET * ((128 // len(CHARSET)) + 1))
    random.shuffle(shuffled)

    # Build reverse lookup (first occurrence)
    reverse_map = {{}}
    for i, char in enumerate(shuffled[:128]):
        if char not in reverse_map:
            reverse_map[char] = i

    decrypted = []
    for char in ciphertext:
        if char in reverse_map:
            decrypted.append(chr(reverse_map[char]))
        else:
            decrypted.append(char)

    return ''.join(decrypted)


# Example usage
if __name__ == "__main__":
    message = "Hello World"
    key = 42

    encrypted = encrypt(message, seed=key)
    print(f"Original: {{message}}")
    print(f"Encrypted: {{encrypted}}")

    decrypted = decrypt(encrypted, seed=key)
    print(f"Decrypted: {{decrypted}}")
'''
        elif "prime" in desc:
            code = '''def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def get_primes(limit: int) -> list[int]:
    """Get all prime numbers up to limit."""
    return [n for n in range(2, limit + 1) if is_prime(n)]
'''
        elif "csv" in desc or "parse" in desc:
            code = '''import csv
from typing import List, Dict

def parse_csv(filepath: str) -> List[Dict[str, str]]:
    """Parse a CSV file and return list of row dictionaries."""
    rows = []
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def write_csv(filepath: str, data: List[Dict], fieldnames: List[str] = None):
    """Write list of dictionaries to CSV file."""
    if not data:
        return

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
'''
        elif "json" in desc:
            code = '''import json
from typing import Any

def load_json(filepath: str) -> Any:
    """Load JSON from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(filepath: str, data: Any, indent: int = 2):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def parse_json(text: str) -> Any:
    """Parse JSON string."""
    return json.loads(text)
'''
        elif "api" in desc or "request" in desc or "fetch" in desc:
            code = '''import urllib.request
import json
from typing import Any, Dict, Optional

def fetch_json(url: str, headers: Optional[Dict[str, str]] = None) -> Any:
    """Fetch JSON from URL."""
    req = urllib.request.Request(url)
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)

    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode('utf-8'))


def post_json(url: str, data: Dict, headers: Optional[Dict[str, str]] = None) -> Any:
    """POST JSON to URL and return response."""
    body = json.dumps(data).encode('utf-8')

    req = urllib.request.Request(url, data=body, method='POST')
    req.add_header('Content-Type', 'application/json')
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)

    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode('utf-8'))
'''
        else:
            code = '''def placeholder():
    """Placeholder function."""
    pass
'''

        reasoning = f"Generated code for: {task.description}"
        return code, reasoning

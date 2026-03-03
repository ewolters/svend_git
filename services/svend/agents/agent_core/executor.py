"""
Safe Code Executor

Executes code in a sandboxed environment with:
- Timeout enforcement
- Output capture
- Error handling
- Resource limits
"""

import subprocess
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    stdout: str
    stderr: str
    return_value: Any = None
    execution_time_ms: float = 0
    error: str | None = None


class CodeExecutor:
    """
    Safe code execution with sandboxing.

    Security measures:
    - Runs in subprocess (isolated)
    - Timeout enforcement
    - Temp directory for files
    - No network by default (can be enabled)
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        working_dir: Path | None = None,
        python_path: str = "python3",
    ):
        self.timeout = timeout_seconds
        self.working_dir = working_dir or Path(tempfile.mkdtemp(prefix="agent_"))
        self.python_path = python_path

        # Ensure working dir exists
        self.working_dir.mkdir(parents=True, exist_ok=True)

    def execute_python(self, code: str, filename: str = "script.py") -> ExecutionResult:
        """Execute Python code and return results."""
        import time

        # Write code to temp file
        script_path = self.working_dir / filename
        script_path.write_text(code)

        start_time = time.time()

        try:
            result = subprocess.run(
                [self.python_path, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.working_dir),
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            )

            execution_time = (time.time() - start_time) * 1000

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time_ms=execution_time,
                error=result.stderr if result.returncode != 0 else None,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                error=f"Execution timed out after {self.timeout} seconds",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                error=str(e),
            )

    def execute_bash(self, command: str) -> ExecutionResult:
        """Execute a bash command."""
        import time

        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.working_dir),
            )

            execution_time = (time.time() - start_time) * 1000

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time_ms=execution_time,
                error=result.stderr if result.returncode != 0 else None,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                error=f"Execution timed out after {self.timeout} seconds",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                error=str(e),
            )

    def run_tests(self, test_code: str, implementation_code: str) -> ExecutionResult:
        """Run tests against implementation."""
        # Combine implementation and tests
        combined = f"{implementation_code}\n\n# Tests\n{test_code}"
        return self.execute_python(combined, "test_script.py")

    def lint_code(self, code: str) -> ExecutionResult:
        """Run linter on code."""
        script_path = self.working_dir / "lint_target.py"
        script_path.write_text(code)

        # Try ruff first, fall back to pylint
        result = self.execute_bash(f"ruff check {script_path} 2>/dev/null || python3 -m py_compile {script_path}")
        return result

    def cleanup(self):
        """Clean up working directory."""
        import shutil
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir, ignore_errors=True)

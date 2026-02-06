"""
Workflow Engine

Dead simple sequential execution with shared context.
No AI orchestration. Just a for loop.

Usage:
    workflow = Workflow("My Analysis")
    workflow.add(ResearchStep("ATP mechanism"))
    workflow.add(CoderStep("simulate pump", uses=["research"]))
    workflow.add(WriterStep("lab_report", uses=["research", "code"]))
    result = workflow.run()
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from enum import Enum


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Output from a workflow step."""
    name: str
    status: StepStatus
    output: Any = None
    error: str = None
    duration_seconds: float = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "output": self._serialize_output(),
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 2),
            "metadata": self.metadata,
        }

    def _serialize_output(self) -> Any:
        """Serialize output for JSON."""
        if self.output is None:
            return None
        if hasattr(self.output, 'to_dict'):
            return self.output.to_dict()
        if hasattr(self.output, '__dict__'):
            return {k: str(v) for k, v in self.output.__dict__.items()}
        return str(self.output)


@dataclass
class Step:
    """
    A single workflow step.

    Subclass this or use the factory functions (ResearchStep, CoderStep, etc.)
    """
    name: str
    description: str = ""
    uses: list[str] = field(default_factory=list)  # Names of previous steps to use
    condition: Callable[[dict], bool] = None  # Optional: only run if condition(context) is True

    def execute(self, context: dict) -> StepResult:
        """
        Execute this step with the given context.

        Override this in subclasses.
        """
        raise NotImplementedError("Subclasses must implement execute()")

    def get_inputs(self, context: dict) -> dict:
        """Get inputs from previous steps."""
        inputs = {}
        for step_name in self.uses:
            if step_name in context:
                result = context[step_name]
                if isinstance(result, StepResult):
                    inputs[step_name] = result.output
                else:
                    inputs[step_name] = result
        return inputs

    def should_run(self, context: dict) -> bool:
        """Check if this step should run."""
        if self.condition is None:
            return True
        return self.condition(context)


@dataclass
class WorkflowResult:
    """Complete workflow execution result."""
    name: str
    steps: list[StepResult]
    success: bool
    total_duration: float
    started_at: str
    completed_at: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "success": self.success,
            "total_duration_seconds": round(self.total_duration, 2),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_markdown(self) -> str:
        """Generate markdown summary."""
        lines = [
            f"# Workflow: {self.name}",
            "",
            f"**Status:** {'Success' if self.success else 'Failed'}",
            f"**Duration:** {self.total_duration:.1f}s",
            f"**Started:** {self.started_at}",
            "",
            "## Steps",
            "",
        ]

        for i, step in enumerate(self.steps, 1):
            icon = "✓" if step.status == StepStatus.COMPLETED else "✗" if step.status == StepStatus.FAILED else "○"
            lines.append(f"{i}. {icon} **{step.name}** ({step.duration_seconds:.1f}s)")
            if step.error:
                lines.append(f"   - Error: {step.error}")

        return "\n".join(lines)

    def get_output(self, step_name: str) -> Any:
        """Get output from a specific step."""
        for step in self.steps:
            if step.name == step_name:
                return step.output
        return None

    def save(self, path: Path):
        """Save workflow result to JSON."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


class Workflow:
    """
    Simple sequential workflow executor.

    The orchestrator is a for loop. That's it.
    Intelligence is in the agents, not here.
    """

    def __init__(self, name: str = "Workflow"):
        self.name = name
        self.steps: list[Step] = []
        self.context: dict[str, StepResult] = {}
        self.on_step_complete: Callable[[Step, StepResult], None] = None

    def add(self, step: Step) -> "Workflow":
        """Add a step to the workflow. Returns self for chaining."""
        self.steps.append(step)
        return self

    def run(self, initial_context: dict = None) -> WorkflowResult:
        """
        Execute all steps sequentially.

        Args:
            initial_context: Optional initial data available to all steps

        Returns:
            WorkflowResult with all step outputs
        """
        self.context = dict(initial_context or {})
        results = []
        success = True
        start_time = time.time()
        started_at = datetime.now().isoformat()

        for step in self.steps:
            # Check condition
            if not step.should_run(self.context):
                result = StepResult(
                    name=step.name,
                    status=StepStatus.SKIPPED,
                    metadata={"reason": "Condition not met"}
                )
                results.append(result)
                continue

            # Execute step
            step_start = time.time()
            try:
                result = step.execute(self.context)
                result.duration_seconds = time.time() - step_start

                if result.status == StepStatus.FAILED:
                    success = False

            except Exception as e:
                result = StepResult(
                    name=step.name,
                    status=StepStatus.FAILED,
                    error=str(e),
                    duration_seconds=time.time() - step_start,
                )
                success = False

            # Store result in context for next steps
            self.context[step.name] = result
            results.append(result)

            # Callback
            if self.on_step_complete:
                self.on_step_complete(step, result)

            # Stop on failure (could make this configurable)
            if result.status == StepStatus.FAILED:
                break

        return WorkflowResult(
            name=self.name,
            steps=results,
            success=success,
            total_duration=time.time() - start_time,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
        )

    def to_python(self) -> str:
        """
        Export workflow as reproducible Python script.

        This is the key feature - users can save and re-run workflows.
        """
        lines = [
            '"""',
            f'Workflow: {self.name}',
            f'Generated: {datetime.now().isoformat()}',
            '"""',
            '',
            'from workflow import Workflow',
            'from workflow.steps import *',
            '',
            f'workflow = Workflow("{self.name}")',
            '',
        ]

        for step in self.steps:
            # Generate step instantiation
            step_class = step.__class__.__name__
            args = []

            # Common attributes
            if hasattr(step, 'query'):
                args.append(f'"{step.query}"')
            if hasattr(step, 'prompt'):
                args.append(f'"{step.prompt}"')
            if hasattr(step, 'template'):
                args.append(f'template="{step.template}"')
            if step.uses:
                args.append(f'uses={step.uses}')

            args_str = ", ".join(args)
            lines.append(f'workflow.add({step_class}({args_str}))')

        lines.extend([
            '',
            '# Run workflow',
            'result = workflow.run()',
            'print(result.to_markdown())',
            '',
            '# Save results',
            '# result.save("output.json")',
        ])

        return "\n".join(lines)

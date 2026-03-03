"""
Workflow Engine

Simple sequential workflow execution.
The orchestrator is a for loop - intelligence is in the agents.

Usage:
    from workflow import Workflow
    from workflow.steps import ResearchStep, CoderStep, WriterStep

    workflow = Workflow("My Analysis")
    workflow.add(ResearchStep("ATP mechanism"))
    workflow.add(CoderStep("simulate pump", uses=["research"]))
    workflow.add(WriterStep("lab_report", uses=["research", "code"]))

    result = workflow.run()
    print(result.to_markdown())

    # Export as reproducible Python script
    print(workflow.to_python())
"""

from .engine import (
    Workflow,
    Step,
    StepResult,
    StepStatus,
    WorkflowResult,
)

from .steps import (
    ResearchStep,
    CoderStep,
    WriterStep,
    ReviewStep,
    ExperimentStep,
    ChemistryStep,
    ChartStep,
    ExportStep,
    CustomStep,
)

__all__ = [
    # Engine
    "Workflow",
    "Step",
    "StepResult",
    "StepStatus",
    "WorkflowResult",
    # Steps
    "ResearchStep",
    "CoderStep",
    "WriterStep",
    "ReviewStep",
    "ExperimentStep",
    "ChemistryStep",
    "ChartStep",
    "ExportStep",
    "CustomStep",
]

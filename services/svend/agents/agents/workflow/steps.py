"""
Workflow Steps

Pre-built steps that wrap the agents.
Each step takes structured input and produces structured output.
"""

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from .engine import Step, StepResult, StepStatus


@dataclass
class ResearchStep(Step):
    """Research step using the researcher agent."""
    query: str = ""
    focus: str = "general"  # scientific, market, general
    depth: str = "standard"  # quick, standard, thorough

    def __post_init__(self):
        if not self.name:
            self.name = "research"
        if not self.description:
            self.description = f"Research: {self.query[:50]}"

    def execute(self, context: dict) -> StepResult:
        try:
            from researcher.agent import ResearchAgent, ResearchQuery

            # Build query, incorporating any previous context
            query_text = self.query
            inputs = self.get_inputs(context)
            if inputs:
                # Append relevant context
                for key, val in inputs.items():
                    if hasattr(val, 'summary'):
                        query_text += f"\n\nContext from {key}: {val.summary}"

            agent = ResearchAgent(llm=None)  # Mock mode by default
            query = ResearchQuery(
                question=query_text,
                focus=self.focus,
                depth=self.depth,
            )
            result = agent.run(query)

            return StepResult(
                name=self.name,
                status=StepStatus.COMPLETED,
                output=result,
                metadata={"sources": len(result.sources), "focus": self.focus}
            )
        except Exception as e:
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                error=str(e)
            )


@dataclass
class CoderStep(Step):
    """Code generation step using the coder agent."""
    prompt: str = ""
    language: str = "python"
    constraints: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            self.name = "code"
        if not self.description:
            self.description = f"Generate code: {self.prompt[:50]}"

    def execute(self, context: dict) -> StepResult:
        try:
            from coder.agent import CodingAgent, CodingTask

            # Build prompt with context
            full_prompt = self.prompt
            inputs = self.get_inputs(context)
            if inputs:
                for key, val in inputs.items():
                    if hasattr(val, 'summary'):
                        full_prompt += f"\n\nFrom {key}: {val.summary}"

            agent = CodingAgent(llm=None)  # Mock mode
            task = CodingTask(
                description=full_prompt,
                constraints=self.constraints,
            )
            result = agent.run(task)

            return StepResult(
                name=self.name,
                status=StepStatus.COMPLETED,
                output=result,
                metadata={"language": self.language}
            )
        except Exception as e:
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                error=str(e)
            )


@dataclass
class WriterStep(Step):
    """Document generation step using the writer agent."""
    topic: str = ""
    template: str = "technical_report"
    tone: str = "formal"
    voice_profile: str = None  # Name of saved voice profile

    def __post_init__(self):
        if not self.name:
            self.name = "document"
        if not self.description:
            self.description = f"Write {self.template}: {self.topic[:30]}"

    def execute(self, context: dict) -> StepResult:
        try:
            from writer.agent import WriterAgent, DocumentRequest, DocumentType

            # Map template string to enum
            doc_type = DocumentType.TECHNICAL_REPORT
            for dt in DocumentType:
                if dt.value == self.template:
                    doc_type = dt
                    break

            # Gather notes from previous steps
            notes = []
            inputs = self.get_inputs(context)
            for key, val in inputs.items():
                if hasattr(val, 'summary'):
                    notes.append(f"[{key}] {val.summary}")
                elif hasattr(val, 'to_markdown'):
                    notes.append(f"[{key}]\n{val.to_markdown()[:500]}")

            agent = WriterAgent(llm=None)
            request = DocumentRequest(
                topic=self.topic or "Analysis Report",
                doc_type=doc_type,
                tone=self.tone,
                notes=notes,
            )
            result = agent.write(request)

            return StepResult(
                name=self.name,
                status=StepStatus.COMPLETED,
                output=result,
                metadata={"template": self.template, "word_count": result.word_count}
            )
        except Exception as e:
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                error=str(e)
            )


@dataclass
class ReviewStep(Step):
    """Document review step."""
    doc_type: str = "general"

    def __post_init__(self):
        if not self.name:
            self.name = "review"
        if not self.description:
            self.description = "Review document quality"

    def execute(self, context: dict) -> StepResult:
        try:
            from reviewer.agent import DocumentReviewer

            # Get document from previous step
            inputs = self.get_inputs(context)
            document = ""
            title = "Document"

            for key, val in inputs.items():
                if hasattr(val, 'to_markdown'):
                    document = val.to_markdown()
                    title = getattr(val, 'title', key)
                    break
                elif hasattr(val, 'content'):
                    document = val.content
                    title = getattr(val, 'title', key)
                    break

            if not document:
                return StepResult(
                    name=self.name,
                    status=StepStatus.FAILED,
                    error="No document found in previous steps"
                )

            reviewer = DocumentReviewer()
            result = reviewer.review(document, title, self.doc_type)

            return StepResult(
                name=self.name,
                status=StepStatus.COMPLETED,
                output=result,
                metadata={
                    "findings": len(result.findings),
                    "score": f"{result.overall_score:.0%}"
                }
            )
        except Exception as e:
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                error=str(e)
            )


@dataclass
class ExperimentStep(Step):
    """Experimental design step."""
    goal: str = ""
    test_type: str = "ttest_ind"
    effect_size: float = 0.5
    factors: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            self.name = "experiment"
        if not self.description:
            self.description = f"Design experiment: {self.goal[:30]}"

    def execute(self, context: dict) -> StepResult:
        try:
            from experimenter.agent import ExperimenterAgent, ExperimentRequest

            agent = ExperimenterAgent()
            request = ExperimentRequest(
                goal=self.goal,
                request_type="full" if self.factors else "power",
                test_type=self.test_type,
                effect_size=self.effect_size,
                factors=self.factors,
                include_plots=True,
            )
            result = agent.design_experiment(request)

            return StepResult(
                name=self.name,
                status=StepStatus.COMPLETED,
                output=result,
                metadata={
                    "sample_size": result.power_result.sample_size if result.power_result else None,
                    "design_runs": result.design.num_runs if result.design else None,
                }
            )
        except Exception as e:
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                error=str(e)
            )


@dataclass
class ChemistryStep(Step):
    """Chemistry documentation step."""
    reactions: list[str] = field(default_factory=list)
    include_thermodynamics: bool = False
    include_mechanism: bool = False

    def __post_init__(self):
        if not self.name:
            self.name = "chemistry"
        if not self.description:
            self.description = "Format chemistry documentation"

    def execute(self, context: dict) -> StepResult:
        try:
            from docs.chemistry import ChemistryFormatter

            formatter = ChemistryFormatter()

            # Get reactions from input or self
            reactions = self.reactions.copy()
            inputs = self.get_inputs(context)

            # Look for reactions in previous outputs
            for key, val in inputs.items():
                if hasattr(val, 'reactions'):
                    reactions.extend(val.reactions)

            output = {
                "reactions_latex": [],
                "thermodynamics": [],
            }

            for rxn in reactions:
                latex = formatter.reaction_to_latex(rxn)
                output["reactions_latex"].append(latex)

                if self.include_thermodynamics:
                    thermo = formatter.estimate_thermodynamics(rxn)
                    output["thermodynamics"].append(thermo)

            return StepResult(
                name=self.name,
                status=StepStatus.COMPLETED,
                output=output,
                metadata={"reactions": len(reactions)}
            )
        except Exception as e:
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                error=str(e)
            )


@dataclass
class ChartStep(Step):
    """Generate charts from data."""
    chart_type: str = "line"  # line, bar, scatter, heatmap
    title: str = ""
    x_label: str = ""
    y_label: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = "chart"
        if not self.description:
            self.description = f"Generate {self.chart_type} chart"

    def execute(self, context: dict) -> StepResult:
        try:
            from docs.charts import ChartGenerator

            generator = ChartGenerator()
            inputs = self.get_inputs(context)

            # Look for plottable data in inputs
            data = None
            for key, val in inputs.items():
                if hasattr(val, 'to_matrix'):
                    data = val.to_matrix()
                    break
                elif isinstance(val, dict) and 'data' in val:
                    data = val['data']
                    break

            if data is None:
                # Generate placeholder
                data = {"x": [1, 2, 3], "y": [1, 4, 9]}

            plot_path = generator.create_chart(
                data=data,
                chart_type=self.chart_type,
                title=self.title,
                x_label=self.x_label,
                y_label=self.y_label,
            )

            return StepResult(
                name=self.name,
                status=StepStatus.COMPLETED,
                output={"chart_path": plot_path},
                metadata={"chart_type": self.chart_type}
            )
        except Exception as e:
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                error=str(e)
            )


@dataclass
class ExportStep(Step):
    """Export workflow results to file."""
    output_path: str = ""
    format: str = "markdown"  # markdown, pdf, html, latex

    def __post_init__(self):
        if not self.name:
            self.name = "export"
        if not self.description:
            self.description = f"Export to {self.format}"

    def execute(self, context: dict) -> StepResult:
        try:
            from docs.export import DocumentExporter

            exporter = DocumentExporter()
            inputs = self.get_inputs(context)

            # Collect all content
            sections = []
            for key, val in inputs.items():
                if hasattr(val, 'to_markdown'):
                    sections.append({"title": key, "content": val.to_markdown()})
                elif isinstance(val, dict):
                    sections.append({"title": key, "content": str(val)})

            output_path = Path(self.output_path) if self.output_path else Path(f"output.{self.format}")

            result_path = exporter.export(
                sections=sections,
                output_path=output_path,
                format=self.format,
            )

            return StepResult(
                name=self.name,
                status=StepStatus.COMPLETED,
                output={"path": str(result_path)},
                metadata={"format": self.format}
            )
        except Exception as e:
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                error=str(e)
            )


@dataclass
class CustomStep(Step):
    """Custom step with user-provided function."""
    func: callable = None

    def execute(self, context: dict) -> StepResult:
        if self.func is None:
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                error="No function provided"
            )

        try:
            inputs = self.get_inputs(context)
            output = self.func(inputs)

            return StepResult(
                name=self.name,
                status=StepStatus.COMPLETED,
                output=output,
            )
        except Exception as e:
            return StepResult(
                name=self.name,
                status=StepStatus.FAILED,
                error=str(e)
            )

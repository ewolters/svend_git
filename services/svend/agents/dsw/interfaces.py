"""
Service Interfaces

Clean integration surfaces for all services.
Each service follows the same pattern:

    input → Service.run() → output

Services can be composed:

    Research → Schema → Forge → Scrub → Analyst

Each interface is:
1. A dataclass for input
2. A dataclass for output
3. A protocol the service must implement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, Any, Optional
from datetime import datetime


# =============================================================================
# Base Protocol - All services implement this
# =============================================================================

class ServiceProtocol(Protocol):
    """All services implement this interface."""

    def run(self, request: Any) -> Any:
        """Execute the service."""
        ...

    def validate(self, request: Any) -> list[str]:
        """Validate request, return list of errors (empty = valid)."""
        ...


# =============================================================================
# Research Service Interface
# =============================================================================

@dataclass
class ResearchRequest:
    """Input for Research service."""
    query: str
    focus: str = "general"  # scientific, market, general
    depth: str = "standard"  # quick, standard, thorough
    output_format: str = "markdown"  # markdown, json, schema


@dataclass
class ResearchOutput:
    """Output from Research service."""
    content: str  # Main research content
    sources: list[dict] = field(default_factory=list)
    schema_hint: dict = None  # If output_format=schema, suggested problem schema
    metadata: dict = field(default_factory=dict)

    def to_schema_json(self) -> str:
        """Convert research to schema JSON for Forge."""
        import json
        if self.schema_hint:
            return json.dumps(self.schema_hint, indent=2)
        return "{}"


# =============================================================================
# Forge Service Interface (Synthetic Data Generation)
# =============================================================================

@dataclass
class ForgeRequest:
    """Input for Forge service."""
    # Option 1: Provide schema directly
    schema: dict = None

    # Option 2: Provide JSON that implies schema
    sample_json: str = None

    # Option 3: Reference a ProblemSchema
    problem_schema: Any = None  # ProblemSchema object

    # Generation parameters
    record_count: int = 1000
    quality_level: str = "standard"  # standard, premium
    output_format: str = "dataframe"  # dataframe, jsonl, csv

    # Edge case generation
    include_edge_cases: bool = True
    edge_case_ratio: float = 0.1  # 10% edge cases

    def to_api_request(self) -> dict:
        """Convert to Forge API request format."""
        if self.problem_schema:
            return self.problem_schema.to_forge_request()
        elif self.schema:
            return {
                "data_type": "tabular",
                "schema": self.schema,
                "record_count": self.record_count,
                "quality_level": self.quality_level,
            }
        elif self.sample_json:
            # Infer schema from sample
            import json
            sample = json.loads(self.sample_json)
            schema = self._infer_schema(sample)
            return {
                "data_type": "tabular",
                "schema": schema,
                "record_count": self.record_count,
            }
        return {}

    def _infer_schema(self, sample: dict) -> dict:
        """Infer schema from a sample JSON object."""
        schema = {}
        for key, value in sample.items():
            if isinstance(value, bool):
                schema[key] = {"type": "bool"}
            elif isinstance(value, int):
                schema[key] = {"type": "int"}
            elif isinstance(value, float):
                schema[key] = {"type": "float"}
            elif isinstance(value, str):
                schema[key] = {"type": "string"}
            elif isinstance(value, list):
                if value and isinstance(value[0], str):
                    schema[key] = {"type": "category", "constraints": {"values": value}}
        return schema


@dataclass
class ForgeOutput:
    """Output from Forge service."""
    data: Any  # DataFrame or path to file
    record_count: int
    schema_used: dict
    edge_cases_generated: int = 0
    cost_cents: int = 0
    job_id: str = ""


# =============================================================================
# Scrub Service Interface (Data Cleaning)
# =============================================================================

@dataclass
class ScrubRequest:
    """Input for Scrub service."""
    data: Any  # DataFrame

    # What to do
    detect_outliers: bool = True
    handle_missing: bool = True
    normalize_factors: bool = True
    correct_types: bool = True

    # Domain rules
    domain_rules: dict = field(default_factory=dict)  # column -> (min, max)

    # From schema (if available)
    problem_schema: Any = None  # ProblemSchema - provides domain rules


@dataclass
class ScrubOutput:
    """Output from Scrub service."""
    data: Any  # Cleaned DataFrame
    original_shape: tuple
    cleaned_shape: tuple

    # What happened
    outliers_flagged: int = 0
    missing_filled: int = 0
    values_normalized: int = 0
    types_corrected: int = 0

    # Details
    outlier_flags: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Full result object
    result: Any = None  # CleaningResult


# =============================================================================
# Analyst Service Interface (ML Training)
# =============================================================================

@dataclass
class AnalystRequest:
    """Input for Analyst service."""
    data: Any  # DataFrame
    target: str  # Target column name

    # Optional guidance
    intent: str = ""
    problem_schema: Any = None  # ProblemSchema

    # Model selection
    model_type: str = None  # None = auto-select
    priority: str = "balanced"  # accuracy, interpretability, speed

    # Training params
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class AnalystOutput:
    """Output from Analyst service."""
    model: Any  # Trained model
    model_type: str
    task_type: str  # classification, regression

    # Performance
    metrics: dict = field(default_factory=dict)
    feature_importance: list = field(default_factory=list)

    # Code and reports
    code: str = ""
    report_markdown: str = ""

    # Full result object
    result: Any = None  # TrainingResult


# =============================================================================
# Coder Service Interface
# =============================================================================

@dataclass
class CoderRequest:
    """Input for Coder service."""
    description: str
    language: str = "python"
    constraints: list[str] = field(default_factory=list)
    context: str = ""  # Additional context


@dataclass
class CoderOutput:
    """Output from Coder service."""
    code: str
    verified: bool
    verification_summary: str
    execution_output: str = ""
    qa_report: str = ""


# =============================================================================
# Service Adapters - Wrap existing services with clean interfaces
# =============================================================================

class ScrubAdapter:
    """Adapter to make Scrub follow the interface."""

    def __init__(self):
        from scrub import DataCleaner, CleaningConfig
        self._cleaner = DataCleaner()
        self._config_class = CleaningConfig

    def run(self, request: ScrubRequest) -> ScrubOutput:
        """Run Scrub service."""
        # Build domain rules from schema if provided
        domain_rules = request.domain_rules.copy()
        if request.problem_schema:
            for feature in request.problem_schema.features:
                if hasattr(feature.constraints, 'min_value'):
                    domain_rules[feature.name] = (
                        feature.constraints.min_value,
                        feature.constraints.max_value,
                    )

        config = self._config_class(
            detect_outliers=request.detect_outliers,
            handle_missing=request.handle_missing,
            normalize_factors=request.normalize_factors,
            correct_types=request.correct_types,
            domain_rules=domain_rules,
        )

        cleaned_data, result = self._cleaner.clean(request.data, config)

        return ScrubOutput(
            data=cleaned_data,
            original_shape=result.original_shape,
            cleaned_shape=result.cleaned_shape,
            outliers_flagged=result.outliers.count if result.outliers else 0,
            missing_filled=result.missing.total_filled if result.missing else 0,
            values_normalized=result.normalization.total_changes if result.normalization else 0,
            outlier_flags=result.outliers.flags if result.outliers else [],
            warnings=result.warnings,
            result=result,
        )

    def validate(self, request: ScrubRequest) -> list[str]:
        """Validate request."""
        errors = []
        if request.data is None:
            errors.append("Data is required")
        return errors


class AnalystAdapter:
    """Adapter to make Analyst follow the interface."""

    def __init__(self):
        from analyst import Analyst
        self._analyst = Analyst()

    def run(self, request: AnalystRequest) -> AnalystOutput:
        """Run Analyst service."""
        from analyst import ModelType

        model_type = None
        if request.model_type:
            model_type = ModelType(request.model_type)

        result = self._analyst.train(
            data=request.data,
            target=request.target,
            intent=request.intent,
            model_type=model_type,
            priority=request.priority,
            test_size=request.test_size,
            random_state=request.random_state,
        )

        return AnalystOutput(
            model=result.model,
            model_type=result.model_type.value,
            task_type=result.task_type.value,
            metrics={m.name: m.value for m in result.report.metrics},
            feature_importance=[
                {"feature": f.feature, "importance": f.importance}
                for f in result.report.feature_importance
            ],
            code=result.code,
            report_markdown=result.report.to_markdown(),
            result=result,
        )

    def validate(self, request: AnalystRequest) -> list[str]:
        """Validate request."""
        errors = []
        if request.data is None:
            errors.append("Data is required")
        if not request.target:
            errors.append("Target column is required")
        return errors


# =============================================================================
# Pipeline Composition Helper
# =============================================================================

@dataclass
class PipelineStep:
    """A step in a pipeline."""
    name: str
    service: Any  # Service adapter
    request_builder: callable  # Function to build request from previous output
    enabled: bool = True


class Pipeline:
    """
    Compose services into a pipeline.

    Usage:
        pipeline = Pipeline()
        pipeline.add("research", research_adapter, lambda _: ResearchRequest(...))
        pipeline.add("forge", forge_adapter, lambda prev: ForgeRequest(schema=prev.schema_hint))
        pipeline.add("scrub", scrub_adapter, lambda prev: ScrubRequest(data=prev.data))
        pipeline.add("analyst", analyst_adapter, lambda prev: AnalystRequest(data=prev.data, target="y"))

        result = pipeline.run()
    """

    def __init__(self):
        self.steps: list[PipelineStep] = []
        self.results: dict[str, Any] = {}

    def add(
        self,
        name: str,
        service: Any,
        request_builder: callable,
        enabled: bool = True,
    ):
        """Add a step to the pipeline."""
        self.steps.append(PipelineStep(
            name=name,
            service=service,
            request_builder=request_builder,
            enabled=enabled,
        ))
        return self

    def run(self, initial_input: Any = None) -> dict[str, Any]:
        """Run the pipeline."""
        previous_output = initial_input

        for step in self.steps:
            if not step.enabled:
                continue

            # Build request from previous output
            request = step.request_builder(previous_output)

            # Validate
            errors = step.service.validate(request)
            if errors:
                raise ValueError(f"Validation failed for {step.name}: {errors}")

            # Run
            output = step.service.run(request)
            self.results[step.name] = output
            previous_output = output

        return self.results

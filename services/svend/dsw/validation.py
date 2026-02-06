"""
DSW Pipeline Validation

Provides:
1. Stage tracking - monitor pipeline progress
2. Interface validation - ensure data flows correctly
3. Quality propagation - pass quality metrics through pipeline
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import time


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageValidationResult:
    """Validation result for a single stage."""
    stage_name: str
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    input_validated: bool = False
    output_validated: bool = False

    def add_error(self, error: str):
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str):
        self.warnings.append(warning)


@dataclass
class StageMetrics:
    """Metrics for a completed stage."""
    stage_name: str
    status: StageStatus = StageStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # Quality metrics propagated through pipeline
    quality_metrics: dict = field(default_factory=dict)

    # Stage-specific output summary
    output_summary: dict = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.status in [StageStatus.COMPLETED, StageStatus.FAILED, StageStatus.SKIPPED]


@dataclass
class PipelineValidationReport:
    """Complete validation report for a DSW pipeline run."""
    pipeline_name: str
    stages: list[StageMetrics] = field(default_factory=list)
    validations: list[StageValidationResult] = field(default_factory=list)

    overall_valid: bool = True
    total_duration: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Quality propagation summary
    quality_chain: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Generate text summary."""
        status = "PASSED" if self.overall_valid else "FAILED"
        lines = [
            "=" * 60,
            f"DSW PIPELINE VALIDATION: {status}",
            "=" * 60,
            "",
            f"Pipeline: {self.pipeline_name}",
            f"Duration: {self.total_duration:.2f}s",
            "",
            "## Stage Summary",
            "",
        ]

        for stage in self.stages:
            icon = {
                StageStatus.COMPLETED: "[OK]",
                StageStatus.FAILED: "[FAIL]",
                StageStatus.SKIPPED: "[SKIP]",
                StageStatus.RUNNING: "[...]",
                StageStatus.PENDING: "[ ]",
            }.get(stage.status, "[ ]")

            lines.append(f"  {icon} {stage.stage_name} ({stage.duration_seconds:.2f}s)")

        if self.errors:
            lines.extend(["", "## Errors"])
            for err in self.errors:
                lines.append(f"  ! {err}")

        if self.warnings:
            lines.extend(["", "## Warnings"])
            for warn in self.warnings:
                lines.append(f"  ~ {warn}")

        if self.quality_chain:
            lines.extend(["", "## Quality Propagation"])
            for stage, metrics in self.quality_chain.items():
                lines.append(f"  {stage}:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        lines.append(f"    {key}: {value:.2%}")
                    else:
                        lines.append(f"    {key}: {value}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "pipeline_name": self.pipeline_name,
            "overall_valid": self.overall_valid,
            "total_duration": self.total_duration,
            "stages": [
                {
                    "name": s.stage_name,
                    "status": s.status.value,
                    "duration": s.duration_seconds,
                    "output_summary": s.output_summary,
                }
                for s in self.stages
            ],
            "validations": [
                {
                    "stage": v.stage_name,
                    "valid": v.valid,
                    "errors": v.errors,
                    "warnings": v.warnings,
                }
                for v in self.validations
            ],
            "quality_chain": self.quality_chain,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class PipelineStageTracker:
    """
    Tracks pipeline stage execution.

    Usage:
        tracker = PipelineStageTracker("my_pipeline")

        tracker.start_stage("Schema")
        # ... do schema work ...
        tracker.complete_stage("Schema", output_summary={"features": 5})

        tracker.start_stage("Forge")
        # ... do forge work ...
        tracker.complete_stage("Forge", output_summary={"records": 1000})

        print(tracker.get_report().summary())
    """

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.stages: dict[str, StageMetrics] = {}
        self.stage_order: list[str] = []
        self.current_stage: Optional[str] = None
        self._stage_start_time: Optional[float] = None
        self.pipeline_start_time: Optional[float] = None

    def start_stage(self, stage_name: str):
        """Mark a stage as starting."""
        if self.pipeline_start_time is None:
            self.pipeline_start_time = time.time()

        self.current_stage = stage_name
        self._stage_start_time = time.time()

        self.stages[stage_name] = StageMetrics(
            stage_name=stage_name,
            status=StageStatus.RUNNING,
            start_time=datetime.now(),
        )

        if stage_name not in self.stage_order:
            self.stage_order.append(stage_name)

    def complete_stage(
        self,
        stage_name: str,
        output_summary: dict = None,
        quality_metrics: dict = None,
    ):
        """Mark a stage as completed."""
        if stage_name not in self.stages:
            self.stages[stage_name] = StageMetrics(stage_name=stage_name)

        stage = self.stages[stage_name]
        stage.status = StageStatus.COMPLETED
        stage.end_time = datetime.now()

        if self._stage_start_time:
            stage.duration_seconds = time.time() - self._stage_start_time

        stage.output_summary = output_summary or {}
        stage.quality_metrics = quality_metrics or {}

        self.current_stage = None
        self._stage_start_time = None

    def fail_stage(self, stage_name: str, error: str):
        """Mark a stage as failed."""
        if stage_name not in self.stages:
            self.stages[stage_name] = StageMetrics(stage_name=stage_name)

        stage = self.stages[stage_name]
        stage.status = StageStatus.FAILED
        stage.end_time = datetime.now()

        if self._stage_start_time:
            stage.duration_seconds = time.time() - self._stage_start_time

        stage.output_summary = {"error": error}

        self.current_stage = None
        self._stage_start_time = None

    def skip_stage(self, stage_name: str, reason: str = ""):
        """Mark a stage as skipped."""
        self.stages[stage_name] = StageMetrics(
            stage_name=stage_name,
            status=StageStatus.SKIPPED,
            output_summary={"reason": reason} if reason else {},
        )
        if stage_name not in self.stage_order:
            self.stage_order.append(stage_name)

    def get_report(self) -> PipelineValidationReport:
        """Get the complete pipeline report."""
        report = PipelineValidationReport(pipeline_name=self.pipeline_name)

        # Add stages in order
        for stage_name in self.stage_order:
            if stage_name in self.stages:
                report.stages.append(self.stages[stage_name])

        # Calculate total duration
        if self.pipeline_start_time:
            report.total_duration = time.time() - self.pipeline_start_time

        # Check for any failures
        report.overall_valid = all(
            s.status != StageStatus.FAILED for s in report.stages
        )

        # Build quality chain
        for stage_name in self.stage_order:
            if stage_name in self.stages:
                qm = self.stages[stage_name].quality_metrics
                if qm:
                    report.quality_chain[stage_name] = qm

        return report


class InterfaceValidator:
    """
    Validates data flowing between pipeline stages.

    Checks:
    - Data has expected columns/shape
    - No unexpected nulls introduced
    - Data types are correct
    - Quality metrics are propagated
    """

    # Expected columns by stage
    STAGE_REQUIREMENTS = {
        "Schema": {
            "output_fields": ["name", "intent", "target_name", "features"],
        },
        "Forge": {
            "input_requires": ["schema"],
            "output_type": "DataFrame",
            "output_min_rows": 1,
        },
        "Scrub": {
            "input_requires": ["data"],
            "output_type": "DataFrame",
            "output_preserves_target": True,
        },
        "Analyst": {
            "input_requires": ["data", "target"],
            "output_fields": ["model", "metrics"],
        },
    }

    def validate_stage_input(
        self,
        stage_name: str,
        input_data: Any,
        previous_output: Any = None,
    ) -> StageValidationResult:
        """Validate input to a stage."""
        result = StageValidationResult(stage_name=stage_name)

        reqs = self.STAGE_REQUIREMENTS.get(stage_name, {})

        # Check required inputs
        for req in reqs.get("input_requires", []):
            if req == "data":
                if input_data is None:
                    result.add_error(f"Stage '{stage_name}' requires data input")
                elif hasattr(input_data, '__len__') and len(input_data) == 0:
                    result.add_error(f"Stage '{stage_name}' received empty data")

            if req == "schema":
                if previous_output is None:
                    result.add_error(f"Stage '{stage_name}' requires schema from previous stage")

        result.input_validated = True
        return result

    def validate_stage_output(
        self,
        stage_name: str,
        output_data: Any,
        input_data: Any = None,
    ) -> StageValidationResult:
        """Validate output from a stage."""
        result = StageValidationResult(stage_name=stage_name)

        reqs = self.STAGE_REQUIREMENTS.get(stage_name, {})

        # Check output type
        expected_type = reqs.get("output_type")
        if expected_type == "DataFrame":
            import pandas as pd
            if not isinstance(output_data, pd.DataFrame):
                result.add_error(f"Stage '{stage_name}' should output DataFrame, got {type(output_data)}")
            else:
                # Check minimum rows
                min_rows = reqs.get("output_min_rows", 0)
                if len(output_data) < min_rows:
                    result.add_error(f"Stage '{stage_name}' output has {len(output_data)} rows (min: {min_rows})")

                # Check for unexpected complete null columns
                null_cols = output_data.columns[output_data.isna().all()].tolist()
                if null_cols:
                    result.add_warning(f"Stage '{stage_name}' output has completely null columns: {null_cols}")

        # Check required output fields
        for field in reqs.get("output_fields", []):
            if not hasattr(output_data, field):
                result.add_error(f"Stage '{stage_name}' output missing required field: {field}")

        # Check target preservation (for Scrub)
        if reqs.get("output_preserves_target") and input_data is not None:
            import pandas as pd
            if isinstance(input_data, pd.DataFrame) and isinstance(output_data, pd.DataFrame):
                input_cols = set(input_data.columns)
                output_cols = set(output_data.columns)
                missing = input_cols - output_cols
                if missing:
                    result.add_warning(f"Stage '{stage_name}' dropped columns: {missing}")

        result.output_validated = True
        return result

    def validate_data_quality_propagation(
        self,
        from_stage: str,
        to_stage: str,
        from_quality: dict,
        to_quality: dict,
    ) -> StageValidationResult:
        """
        Validate that quality metrics are properly propagated.

        Checks that critical quality issues from earlier stages
        are not ignored in later stages.
        """
        result = StageValidationResult(stage_name=f"{from_stage}→{to_stage}")

        # Check if source had quality issues
        if from_quality.get("overall_quality") == "poor":
            if to_quality.get("quality_acknowledged") != True:
                result.add_warning(
                    f"Stage '{to_stage}' received poor quality data from '{from_stage}' - "
                    "ensure this is acknowledged in results"
                )

        # Check if critical warnings propagated
        from_warnings = from_quality.get("critical_warnings", [])
        to_acknowledged = to_quality.get("acknowledged_warnings", [])

        for warning in from_warnings:
            if warning not in to_acknowledged:
                result.add_warning(
                    f"Critical warning from '{from_stage}' not acknowledged in '{to_stage}': {warning[:50]}..."
                )

        return result


class DSWPipelineValidator:
    """
    Complete validation for DSW pipelines.

    Combines:
    - Stage tracking
    - Interface validation
    - Quality propagation

    Usage:
        validator = DSWPipelineValidator()

        # Track each stage
        validator.start_stage("Schema")
        schema = generate_schema(intent)
        validator.complete_stage("Schema", output=schema)

        validator.start_stage("Forge")
        data = forge.generate(schema)
        validator.complete_stage("Forge", output=data,
                                 quality_metrics={"records": len(data)})

        # Get report
        report = validator.get_validation_report()
        print(report.summary())
    """

    def __init__(self, pipeline_name: str = "DSW Pipeline"):
        self.tracker = PipelineStageTracker(pipeline_name)
        self.interface_validator = InterfaceValidator()
        self.validations: list[StageValidationResult] = []
        self._previous_output: Any = None
        self._quality_chain: dict = {}

    def start_stage(self, stage_name: str):
        """Start tracking a stage."""
        self.tracker.start_stage(stage_name)

    def complete_stage(
        self,
        stage_name: str,
        output: Any = None,
        quality_metrics: dict = None,
        validate_output: bool = True,
    ):
        """Complete a stage with validation."""
        # Validate output
        if validate_output and output is not None:
            validation = self.interface_validator.validate_stage_output(
                stage_name, output, self._previous_output
            )
            self.validations.append(validation)

        # Store quality metrics
        if quality_metrics:
            self._quality_chain[stage_name] = quality_metrics

        # Complete tracking
        output_summary = {}
        if output is not None:
            if hasattr(output, '__len__'):
                output_summary["record_count"] = len(output)
            if hasattr(output, 'shape'):
                output_summary["shape"] = output.shape

        self.tracker.complete_stage(
            stage_name,
            output_summary=output_summary,
            quality_metrics=quality_metrics,
        )

        self._previous_output = output

    def fail_stage(self, stage_name: str, error: str):
        """Mark a stage as failed."""
        self.tracker.fail_stage(stage_name, error)
        self.validations.append(StageValidationResult(
            stage_name=stage_name,
            valid=False,
            errors=[error],
        ))

    def skip_stage(self, stage_name: str, reason: str = ""):
        """Skip a stage."""
        self.tracker.skip_stage(stage_name, reason)

    def validate_input(self, stage_name: str, input_data: Any) -> bool:
        """Validate input before processing."""
        validation = self.interface_validator.validate_stage_input(
            stage_name, input_data, self._previous_output
        )
        self.validations.append(validation)
        return validation.valid

    def get_validation_report(self) -> PipelineValidationReport:
        """Get complete validation report."""
        report = self.tracker.get_report()
        report.validations = self.validations
        report.quality_chain = self._quality_chain

        # Collect all errors and warnings
        for v in self.validations:
            report.errors.extend(v.errors)
            report.warnings.extend(v.warnings)

        # Update overall validity
        report.overall_valid = report.overall_valid and all(v.valid for v in self.validations)

        return report


def validate_dsw_result(result: "DSWResult") -> PipelineValidationReport:
    """
    Validate a completed DSW result.

    Quick validation of a DSWResult after execution.
    """
    from .workbench import DSWResult

    validator = DSWPipelineValidator("DSW Result Validation")

    # Check schema
    if result.schema:
        validator.start_stage("Schema")
        validator.complete_stage("Schema", result.schema)
    else:
        validator.skip_stage("Schema", "No schema in result")

    # Check data
    if result.synthetic_data is not None:
        validator.start_stage("Forge")
        validator.complete_stage("Forge", result.synthetic_data,
                                 quality_metrics={"records": len(result.synthetic_data)})

    if result.cleaned_data is not None:
        validator.start_stage("Scrub")
        validator.complete_stage("Scrub", result.cleaned_data,
                                 quality_metrics={"records": len(result.cleaned_data)})

    # Check model
    if result.model:
        validator.start_stage("Analyst")
        quality = {
            "model_type": result.model_type,
            "metrics": result.metrics,
        }
        validator.complete_stage("Analyst", result.model, quality_metrics=quality)
    else:
        validator.fail_stage("Analyst", "No model in result")

    return validator.get_validation_report()

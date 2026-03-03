"""
Decision Science Workbench

The main orchestrator that connects all services:

    Intent → Research → Schema → Forge → Scrub → Analyst → Deployment

Supports two entry points:
1. from_intent() - Start with just an idea, no data
2. from_data() - Start with existing data

Produces:
- Trained model
- Deployment code
- Improvement plan
- Full documentation
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from .schema import ProblemSchema, SchemaGenerator
from .forge_client import ForgeClient, ForgeConfig
from .interfaces import (
    ScrubAdapter, AnalystAdapter,
    ScrubRequest, AnalystRequest,
    ForgeRequest,
)
from .validation import (
    DSWPipelineValidator,
    PipelineValidationReport,
    validate_dsw_result,
)


@dataclass
class ImprovementPlan:
    """Plan for improving the model after initial deployment."""
    data_to_collect: list[str]
    validation_strategy: str
    monitoring_metrics: list[str]
    iteration_schedule: str
    risks: list[str]

    def to_markdown(self) -> str:
        lines = [
            "# Improvement Plan",
            "",
            "## Data to Collect",
            "",
        ]
        for item in self.data_to_collect:
            lines.append(f"- {item}")

        lines.extend([
            "",
            "## Validation Strategy",
            "",
            self.validation_strategy,
            "",
            "## Monitoring Metrics",
            "",
        ])
        for metric in self.monitoring_metrics:
            lines.append(f"- {metric}")

        lines.extend([
            "",
            "## Iteration Schedule",
            "",
            self.iteration_schedule,
            "",
            "## Risks & Mitigations",
            "",
        ])
        for risk in self.risks:
            lines.append(f"- {risk}")

        return "\n".join(lines)


@dataclass
class DSWResult:
    """Complete result from Decision Science Workbench."""
    # Schema
    schema: ProblemSchema = None

    # Data
    synthetic_data: Any = None  # DataFrame from Forge
    cleaned_data: Any = None    # DataFrame from Scrub
    original_data: Any = None   # If started with data

    # Model
    model: Any = None
    model_type: str = ""
    metrics: dict = field(default_factory=dict)
    feature_importance: list = field(default_factory=list)

    # Outputs
    deployment_code: str = ""
    improvement_plan: ImprovementPlan = None

    # Reports
    cleaning_report: str = ""
    training_report: str = ""

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Validation report (if validation enabled)
    validation_report: PipelineValidationReport = None

    def summary(self) -> str:
        """Quick summary."""
        lines = [
            "=" * 60,
            "DECISION SCIENCE WORKBENCH RESULT",
            "=" * 60,
            "",
            f"Timestamp: {self.timestamp}",
            f"Pipeline: {' → '.join(self.pipeline_steps)}",
            "",
        ]

        if self.schema:
            lines.extend([
                f"Problem: {self.schema.name}",
                f"Domain: {self.schema.domain}",
                f"Intent: {self.schema.intent}",
                "",
            ])

        if self.model:
            lines.extend([
                f"Model: {self.model_type}",
                "Metrics:",
            ])
            for name, value in self.metrics.items():
                lines.append(f"  {name}: {value:.4f}")
            lines.append("")

        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def full_report(self) -> str:
        """Complete markdown report."""
        lines = [
            "# Decision Science Workbench Report",
            "",
            f"**Generated:** {self.timestamp}",
            "",
            "---",
            "",
        ]

        # Schema section
        if self.schema:
            lines.extend([
                "## Problem Definition",
                "",
                self.schema.summary(),
                "",
                "---",
                "",
            ])

        # Data section
        if self.synthetic_data is not None:
            lines.extend([
                "## Data Generation",
                "",
                f"Generated {len(self.synthetic_data)} synthetic records",
                "",
            ])

        if self.cleaned_data is not None:
            lines.extend([
                "## Data Cleaning",
                "",
                self.cleaning_report or f"Cleaned to {len(self.cleaned_data)} records",
                "",
                "---",
                "",
            ])

        # Model section
        if self.model:
            lines.extend([
                "## Model Training",
                "",
                self.training_report,
                "",
                "---",
                "",
            ])

        # Code section
        if self.deployment_code:
            lines.extend([
                "## Deployment Code",
                "",
                "```python",
                self.deployment_code[:2000],  # Truncate for readability
                "```",
                "",
                "---",
                "",
            ])

        # Improvement plan
        if self.improvement_plan:
            lines.append(self.improvement_plan.to_markdown())

        return "\n".join(lines)

    def save(self, directory: str) -> dict:
        """Save all artifacts."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Schema
        if self.schema:
            schema_path = directory / "schema.json"
            schema_path.write_text(self.schema.to_json())
            saved["schema"] = str(schema_path)

        # Data
        if self.synthetic_data is not None:
            data_path = directory / "data" / "synthetic.csv"
            data_path.parent.mkdir(exist_ok=True)
            self.synthetic_data.to_csv(data_path, index=False)
            saved["synthetic_data"] = str(data_path)

        if self.cleaned_data is not None:
            clean_path = directory / "data" / "cleaned.csv"
            clean_path.parent.mkdir(exist_ok=True)
            self.cleaned_data.to_csv(clean_path, index=False)
            saved["cleaned_data"] = str(clean_path)

        # Model
        if self.model:
            import pickle
            model_path = directory / "model" / "model.pkl"
            model_path.parent.mkdir(exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
            saved["model"] = str(model_path)

        # Code
        if self.deployment_code:
            code_path = directory / "code" / "deploy.py"
            code_path.parent.mkdir(exist_ok=True)
            code_path.write_text(self.deployment_code)
            saved["deployment_code"] = str(code_path)

        # Reports
        report_path = directory / "reports" / "full_report.md"
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(self.full_report())
        saved["full_report"] = str(report_path)

        if self.improvement_plan:
            plan_path = directory / "reports" / "improvement_plan.md"
            plan_path.write_text(self.improvement_plan.to_markdown())
            saved["improvement_plan"] = str(plan_path)

        # Metadata
        meta = {
            "timestamp": self.timestamp,
            "pipeline_steps": self.pipeline_steps,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "warnings": self.warnings,
        }
        meta_path = directory / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        saved["metadata"] = str(meta_path)

        return saved


class DecisionScienceWorkbench:
    """
    Complete ML pipeline orchestrator.

    Usage:
        dsw = DecisionScienceWorkbench()

        # From intent (zero data)
        result = dsw.from_intent(
            intent="Predict customer churn for a SaaS product",
            domain="churn",
            n_records=1000,
        )

        # From existing data
        result = dsw.from_data(
            data=df,
            target="churned",
            intent="Predict which customers will churn",
        )

        # Get results
        print(result.summary())
        result.save("output/")
    """

    def __init__(
        self,
        forge_config: ForgeConfig = None,
    ):
        self.schema_generator = SchemaGenerator()
        self.forge_client = ForgeClient(forge_config or ForgeConfig())
        self.scrub = ScrubAdapter()
        self.analyst = AnalystAdapter()

    def from_intent(
        self,
        intent: str,
        domain: str = None,
        n_records: int = 1000,
        priority: str = "balanced",
        clean_data: bool = True,
        validate: bool = True,
    ) -> DSWResult:
        """
        Start from just an intent, no data.

        Pipeline: Intent → Schema → Forge → Scrub → Analyst

        Args:
            intent: What you want to predict
            domain: Optional domain hint (churn, fraud, lead_scoring)
            n_records: How many records to generate
            priority: Model selection priority (accuracy, interpretability, speed, balanced)
            clean_data: Whether to run Scrub
            validate: Whether to run pipeline validation (default True)

        Returns:
            DSWResult with model, code, and improvement plan
        """
        result = DSWResult()
        result.pipeline_steps = []

        # Initialize validator
        validator = DSWPipelineValidator(f"from_intent: {intent[:50]}...") if validate else None

        # Step 1: Generate schema from intent
        if validator:
            validator.start_stage("Schema")
        result.pipeline_steps.append("Schema")
        try:
            result.schema = self.schema_generator.from_intent(intent, domain)
            result.schema.sample_size_recommendation = n_records
            if validator:
                validator.complete_stage("Schema", result.schema,
                                        quality_metrics={"features": len(result.schema.features)})
        except Exception as e:
            if validator:
                validator.fail_stage("Schema", str(e))
            raise

        # Step 2: Generate data with Forge
        if validator:
            validator.start_stage("Forge")
        result.pipeline_steps.append("Forge")
        try:
            result.synthetic_data = self.forge_client.generate_from_schema(
                result.schema, n=n_records
            )
            if validator:
                validator.complete_stage("Forge", result.synthetic_data,
                                        quality_metrics={"records": len(result.synthetic_data)})
        except Exception as e:
            if validator:
                validator.fail_stage("Forge", str(e))
            raise

        # Step 3: Clean data with Scrub (optional)
        working_data = result.synthetic_data
        if clean_data:
            if validator:
                validator.start_stage("Scrub")
            result.pipeline_steps.append("Scrub")
            try:
                scrub_output = self.scrub.run(ScrubRequest(
                    data=working_data,
                    problem_schema=result.schema,
                ))
                working_data = scrub_output.data
                result.cleaned_data = working_data
                result.warnings.extend(scrub_output.warnings)

                # Generate cleaning report
                if scrub_output.result:
                    result.cleaning_report = scrub_output.result.summary()

                if validator:
                    quality = {
                        "outliers_flagged": scrub_output.outliers_flagged,
                        "missing_filled": scrub_output.missing_filled,
                    }
                    validator.complete_stage("Scrub", working_data, quality_metrics=quality)
            except Exception as e:
                if validator:
                    validator.fail_stage("Scrub", str(e))
                raise
        else:
            if validator:
                validator.skip_stage("Scrub", "clean_data=False")

        # Step 4: Train model with Analyst
        if validator:
            validator.start_stage("Analyst")
        result.pipeline_steps.append("Analyst")
        try:
            analyst_output = self.analyst.run(AnalystRequest(
                data=working_data,
                target=result.schema.target_name,
                intent=intent,
                problem_schema=result.schema,
                priority=priority,
            ))

            result.model = analyst_output.model
            result.model_type = analyst_output.model_type
            result.metrics = analyst_output.metrics
            result.feature_importance = analyst_output.feature_importance
            result.deployment_code = analyst_output.code
            result.training_report = analyst_output.report_markdown

            if validator:
                quality = {
                    "model_type": result.model_type,
                    **result.metrics,
                }
                validator.complete_stage("Analyst", result.model, quality_metrics=quality)
        except Exception as e:
            if validator:
                validator.fail_stage("Analyst", str(e))
            raise

        # Step 5: Generate improvement plan
        result.improvement_plan = self._generate_improvement_plan(result)

        # Attach validation report
        if validator:
            result.validation_report = validator.get_validation_report()
            # Add validation warnings to result warnings
            result.warnings.extend(result.validation_report.warnings)

        return result

    def from_data(
        self,
        data,
        target: str,
        intent: str = "",
        clean_data: bool = True,
        priority: str = "balanced",
        domain_rules: dict = None,
        validate: bool = True,
    ) -> DSWResult:
        """
        Start from existing data.

        Pipeline: Data → Scrub → Analyst

        Args:
            data: pandas DataFrame
            target: Target column name
            intent: Description of what you're predicting
            clean_data: Whether to run Scrub
            priority: Model selection priority
            domain_rules: Column constraints for Scrub
            validate: Whether to run pipeline validation (default True)

        Returns:
            DSWResult with model, code, and improvement plan
        """
        result = DSWResult()
        result.pipeline_steps = []
        result.original_data = data.copy()

        # Initialize validator
        validator = DSWPipelineValidator(f"from_data: {target}") if validate else None

        # Create schema from data
        result.schema = self._infer_schema_from_data(data, target, intent)

        # Step 1: Clean data with Scrub (optional)
        working_data = data
        if clean_data:
            if validator:
                validator.start_stage("Scrub")
            result.pipeline_steps.append("Scrub")
            try:
                scrub_output = self.scrub.run(ScrubRequest(
                    data=working_data,
                    domain_rules=domain_rules or {},
                ))
                working_data = scrub_output.data
                result.cleaned_data = working_data
                result.warnings.extend(scrub_output.warnings)

                if scrub_output.result:
                    result.cleaning_report = scrub_output.result.summary()

                if validator:
                    quality = {
                        "outliers_flagged": scrub_output.outliers_flagged,
                        "missing_filled": scrub_output.missing_filled,
                    }
                    validator.complete_stage("Scrub", working_data, quality_metrics=quality)
            except Exception as e:
                if validator:
                    validator.fail_stage("Scrub", str(e))
                raise
        else:
            result.cleaned_data = working_data
            if validator:
                validator.skip_stage("Scrub", "clean_data=False")

        # Step 2: Train model with Analyst
        if validator:
            validator.start_stage("Analyst")
        result.pipeline_steps.append("Analyst")
        try:
            analyst_output = self.analyst.run(AnalystRequest(
                data=working_data,
                target=target,
                intent=intent,
                priority=priority,
            ))

            result.model = analyst_output.model
            result.model_type = analyst_output.model_type
            result.metrics = analyst_output.metrics
            result.feature_importance = analyst_output.feature_importance
            result.deployment_code = analyst_output.code
            result.training_report = analyst_output.report_markdown

            if validator:
                quality = {
                    "model_type": result.model_type,
                    **result.metrics,
                }
                validator.complete_stage("Analyst", result.model, quality_metrics=quality)
        except Exception as e:
            if validator:
                validator.fail_stage("Analyst", str(e))
            raise

        # Step 3: Generate improvement plan
        result.improvement_plan = self._generate_improvement_plan(result)

        # Attach validation report
        if validator:
            result.validation_report = validator.get_validation_report()
            result.warnings.extend(result.validation_report.warnings)

        return result

    def _infer_schema_from_data(
        self,
        data,
        target: str,
        intent: str,
    ) -> ProblemSchema:
        """Infer schema from existing data."""
        import pandas as pd
        import numpy as np

        # Determine task type
        y = data[target]
        if pd.api.types.is_numeric_dtype(y):
            unique_ratio = y.nunique() / len(y)
            task_type = "classification" if unique_ratio < 0.05 or y.nunique() <= 10 else "regression"
        else:
            task_type = "classification"

        # Get target classes
        target_classes = []
        if task_type == "classification":
            target_classes = [str(c) for c in y.unique()]

        # Infer features
        from .schema import FeatureSpec, FeatureType, NumericConstraints, CategoricalConstraints

        features = []
        for col in data.columns:
            if col == target:
                continue

            if pd.api.types.is_numeric_dtype(data[col]):
                features.append(FeatureSpec(
                    name=col,
                    feature_type=FeatureType.NUMERIC,
                    description=f"Numeric feature: {col}",
                    constraints=NumericConstraints(
                        min_value=float(data[col].min()) if pd.notna(data[col].min()) else None,
                        max_value=float(data[col].max()) if pd.notna(data[col].max()) else None,
                        mean=float(data[col].mean()) if pd.notna(data[col].mean()) else None,
                        std=float(data[col].std()) if pd.notna(data[col].std()) else None,
                    ),
                ))
            elif pd.api.types.is_bool_dtype(data[col]):
                features.append(FeatureSpec(
                    name=col,
                    feature_type=FeatureType.BOOLEAN,
                    description=f"Boolean feature: {col}",
                ))
            else:
                features.append(FeatureSpec(
                    name=col,
                    feature_type=FeatureType.CATEGORICAL,
                    description=f"Categorical feature: {col}",
                    constraints=CategoricalConstraints(
                        categories=[str(c) for c in data[col].dropna().unique()[:50]],
                    ),
                ))

        return ProblemSchema(
            name="Data-Driven Model",
            intent=intent,
            domain="inferred",
            task_type=task_type,
            target_name=target,
            target_description=f"Target column: {target}",
            target_classes=target_classes,
            features=features,
            sample_size_recommendation=len(data),
        )

    def _generate_improvement_plan(self, result: DSWResult) -> ImprovementPlan:
        """Generate improvement plan based on results."""
        data_to_collect = []
        if result.schema:
            data_to_collect = result.schema.data_to_collect.copy()
        data_to_collect.extend([
            "Actual outcomes from production predictions",
            "Feature values from real-world data",
            "Edge cases encountered in production",
        ])

        # Add based on model performance
        if result.metrics:
            acc = result.metrics.get("Accuracy", result.metrics.get("R² Score", 0))
            if acc < 0.7:
                data_to_collect.append("Additional features that may improve prediction")

        validation_strategy = """
1. **Holdout Validation**: Keep 20% of new real data as validation set
2. **A/B Testing**: Test model predictions vs current baseline
3. **Shadow Mode**: Run model in parallel with current process before full deployment
4. **Monitoring**: Track prediction distribution and actual outcomes
"""

        monitoring_metrics = [
            "Prediction accuracy (daily/weekly)",
            "Feature distribution drift",
            "Prediction confidence distribution",
            "False positive/negative rates",
            "Model latency",
        ]

        iteration_schedule = """
- **Week 1-2**: Shadow mode deployment, collect real predictions
- **Week 3-4**: Compare predictions to actual outcomes
- **Month 2**: First model update with real data
- **Quarterly**: Regular model refresh with accumulated data
"""

        risks = [
            "Synthetic data may not capture all real-world patterns",
            "Feature distributions may differ in production",
            "Edge cases may be under-represented",
            "Model may need recalibration with real data",
        ]

        return ImprovementPlan(
            data_to_collect=data_to_collect,
            validation_strategy=validation_strategy,
            monitoring_metrics=monitoring_metrics,
            iteration_schedule=iteration_schedule,
            risks=risks,
        )


def quick_dsw(
    intent: str = None,
    data=None,
    target: str = None,
    domain: str = None,
    **kwargs,
) -> DSWResult:
    """
    Quick helper for Decision Science Workbench.

    Args:
        intent: What you want to predict (for from_intent flow)
        data: Existing data (for from_data flow)
        target: Target column name
        domain: Domain hint
        **kwargs: Passed to from_intent or from_data

    Returns:
        DSWResult
    """
    dsw = DecisionScienceWorkbench()

    if data is not None and target:
        return dsw.from_data(data, target=target, intent=intent or "", **kwargs)
    elif intent:
        return dsw.from_intent(intent, domain=domain, **kwargs)
    else:
        raise ValueError("Provide either 'intent' or 'data' + 'target'")

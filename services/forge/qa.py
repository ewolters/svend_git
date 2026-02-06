"""
Forge QA - Quality Assurance for Generated Data

Provides EDA stats and quality metrics for synthetic data.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from core.quality import (
    QualityReport,
    QualityGrade,
    QualityMetric,
    QualityIssue,
    QualitySeverity,
)
from .schemas.schema import TabularSchema, FieldType


@dataclass
class FieldStats:
    """Statistics for a single field."""
    name: str
    dtype: str
    count: int
    null_count: int
    null_percent: float
    unique_count: int

    # Numeric stats (None for non-numeric)
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None

    # Categorical stats (None for numeric)
    top_values: Optional[dict[str, int]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "name": self.name,
            "dtype": self.dtype,
            "count": self.count,
            "null_count": self.null_count,
            "null_percent": round(self.null_percent, 2),
            "unique_count": self.unique_count,
        }

        if self.mean is not None:
            result.update({
                "mean": round(self.mean, 4),
                "std": round(self.std, 4) if self.std else None,
                "min": self.min_val,
                "max": self.max_val,
                "median": round(self.median, 4) if self.median else None,
                "q25": round(self.q25, 4) if self.q25 else None,
                "q75": round(self.q75, 4) if self.q75 else None,
            })

        if self.top_values is not None:
            result["top_values"] = self.top_values

        return result


@dataclass
class ForgeQualityReport(QualityReport):
    """Quality report for Forge synthetic data generation."""
    agent_name: str = "Forge"

    # Forge-specific fields
    row_count: int = 0
    column_count: int = 0
    field_stats: list[FieldStats] = field(default_factory=list)
    schema_compliance: float = 100.0
    constraint_violations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update({
            "row_count": self.row_count,
            "column_count": self.column_count,
            "field_stats": [f.to_dict() for f in self.field_stats],
            "schema_compliance": self.schema_compliance,
            "constraint_violations": self.constraint_violations,
        })
        return base


class ForgeQA:
    """
    Quality assurance for Forge-generated data.

    Analyzes generated data and produces EDA stats + quality report.

    Usage:
        qa = ForgeQA()
        report = qa.analyze(df, schema)
        print(report.summary())
    """

    def analyze(
        self,
        df: "pd.DataFrame",
        schema: TabularSchema = None,
    ) -> ForgeQualityReport:
        """
        Analyze generated data.

        Args:
            df: Generated pandas DataFrame
            schema: Optional schema to check compliance against

        Returns:
            ForgeQualityReport with EDA stats and quality metrics
        """
        import pandas as pd
        import numpy as np

        report = ForgeQualityReport(
            row_count=len(df),
            column_count=len(df.columns),
        )

        # Calculate per-field stats
        for col in df.columns:
            stats = self._compute_field_stats(df[col])
            report.field_stats.append(stats)

        # Add overall metrics
        total_nulls = df.isna().sum().sum()
        total_cells = df.size
        null_rate = (total_nulls / total_cells * 100) if total_cells > 0 else 0

        report.add_metric("rows", report.row_count)
        report.add_metric("columns", report.column_count)
        report.add_metric("null_rate", null_rate, max_threshold=50, unit="%")

        # Check schema compliance if provided
        if schema:
            compliance, violations = self._check_schema_compliance(df, schema)
            report.schema_compliance = compliance
            report.constraint_violations = violations

            report.add_metric(
                "schema_compliance",
                compliance,
                min_threshold=95,
                unit="%",
            )

            for v in violations:
                severity = QualitySeverity.WARNING
                if v.get("violation_rate", 0) > 10:
                    severity = QualitySeverity.ERROR
                report.add_issue(
                    severity=severity,
                    category="constraint",
                    message=f"{v['field']}: {v['issue']}",
                    details=v,
                )

        # Calculate score
        report.score = self._calculate_score(report, schema)
        report.finalize()

        return report

    def _compute_field_stats(self, series: "pd.Series") -> FieldStats:
        """Compute statistics for a single field."""
        import pandas as pd
        import numpy as np

        name = series.name
        dtype = str(series.dtype)
        count = len(series)
        null_count = int(series.isna().sum())
        null_percent = (null_count / count * 100) if count > 0 else 0
        unique_count = int(series.nunique())

        stats = FieldStats(
            name=name,
            dtype=dtype,
            count=count,
            null_count=null_count,
            null_percent=null_percent,
            unique_count=unique_count,
        )

        # Numeric stats
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                stats.mean = float(non_null.mean())
                stats.std = float(non_null.std()) if len(non_null) > 1 else 0.0
                stats.min_val = float(non_null.min())
                stats.max_val = float(non_null.max())
                stats.median = float(non_null.median())
                stats.q25 = float(non_null.quantile(0.25))
                stats.q75 = float(non_null.quantile(0.75))
        else:
            # Categorical stats - top 10 values
            value_counts = series.value_counts().head(10)
            stats.top_values = {str(k): int(v) for k, v in value_counts.items()}

        return stats

    def _check_schema_compliance(
        self,
        df: "pd.DataFrame",
        schema: TabularSchema,
    ) -> tuple[float, list[dict]]:
        """
        Check if data complies with schema constraints.

        Returns:
            (compliance_percent, list of violations)
        """
        import pandas as pd
        import numpy as np

        violations = []
        total_checks = 0
        passed_checks = 0

        schema_fields = {f.name: f for f in schema.fields}

        for col in df.columns:
            if col not in schema_fields:
                continue

            spec = schema_fields[col]
            series = df[col]
            non_null = series.dropna()

            # Check nullable constraint
            total_checks += 1
            if not spec.nullable and series.isna().any():
                null_count = int(series.isna().sum())
                violations.append({
                    "field": col,
                    "issue": f"Found {null_count} nulls but field is not nullable",
                    "violation_count": null_count,
                    "violation_rate": null_count / len(series) * 100,
                })
            else:
                passed_checks += 1

            # Check numeric constraints
            if spec.field_type in (FieldType.INT, FieldType.FLOAT) and len(non_null) > 0:
                if spec.min_value is not None:
                    total_checks += 1
                    below_min = (non_null < spec.min_value).sum()
                    if below_min > 0:
                        violations.append({
                            "field": col,
                            "issue": f"{below_min} values below min ({spec.min_value})",
                            "violation_count": int(below_min),
                            "violation_rate": below_min / len(non_null) * 100,
                        })
                    else:
                        passed_checks += 1

                if spec.max_value is not None:
                    total_checks += 1
                    above_max = (non_null > spec.max_value).sum()
                    if above_max > 0:
                        violations.append({
                            "field": col,
                            "issue": f"{above_max} values above max ({spec.max_value})",
                            "violation_count": int(above_max),
                            "violation_rate": above_max / len(non_null) * 100,
                        })
                    else:
                        passed_checks += 1

            # Check category values
            if spec.field_type == FieldType.CATEGORY and spec.values:
                total_checks += 1
                invalid = ~non_null.isin(spec.values)
                invalid_count = invalid.sum()
                if invalid_count > 0:
                    violations.append({
                        "field": col,
                        "issue": f"{invalid_count} values not in allowed categories",
                        "violation_count": int(invalid_count),
                        "violation_rate": invalid_count / len(non_null) * 100,
                    })
                else:
                    passed_checks += 1

        compliance = (passed_checks / total_checks * 100) if total_checks > 0 else 100.0
        return compliance, violations

    def _calculate_score(
        self,
        report: ForgeQualityReport,
        schema: TabularSchema = None,
    ) -> float:
        """Calculate overall quality score."""
        score = 100.0

        # Penalize high null rate
        null_metric = next((m for m in report.metrics if m.name == "null_rate"), None)
        if null_metric and null_metric.value > 20:
            score -= min(20, null_metric.value - 20)

        # Penalize schema violations
        if schema:
            compliance_metric = next(
                (m for m in report.metrics if m.name == "schema_compliance"),
                None,
            )
            if compliance_metric:
                score -= (100 - compliance_metric.value) * 0.5

        # Penalize for issues
        for issue in report.issues:
            if issue.severity == QualitySeverity.ERROR:
                score -= 10
            elif issue.severity == QualitySeverity.WARNING:
                score -= 5

        return max(0, min(100, score))

    def summary_stats(self, df: "pd.DataFrame") -> dict:
        """
        Quick summary stats without full QA report.

        Returns dict suitable for JSON response.
        """
        report = self.analyze(df)
        return {
            "row_count": report.row_count,
            "column_count": report.column_count,
            "fields": [f.to_dict() for f in report.field_stats],
            "quality": {
                "score": report.score,
                "grade": report.grade.value,
                "passed": report.passed,
            },
        }

"""
Analyst Reporter

Generates educational QA reports that explain:
- What the model learned
- How to interpret results
- What to watch out for
- How to improve
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import json

from .models import ModelType, TaskType, get_model_info

if TYPE_CHECKING:
    from .trainer import DataQualityAssessment


@dataclass
class MetricResult:
    """A single evaluation metric."""
    name: str
    value: float
    interpretation: str
    is_good: bool = True


@dataclass
class FeatureImportance:
    """Feature importance with interpretation."""
    feature: str
    importance: float
    rank: int
    interpretation: str = ""


@dataclass
class AnalystReport:
    """Complete training report with educational content."""
    # Basic info
    model_type: ModelType
    task_type: TaskType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Data info
    n_samples: int = 0
    n_features: int = 0
    n_classes: int = 0
    target_column: str = ""
    feature_columns: list[str] = field(default_factory=list)

    # Training info
    train_size: int = 0
    test_size: int = 0
    training_time_seconds: float = 0.0

    # Metrics
    metrics: list[MetricResult] = field(default_factory=list)

    # Feature importance
    feature_importance: list[FeatureImportance] = field(default_factory=list)

    # Warnings and recommendations
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Cross-validation results
    cv_scores: list[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0

    # Data quality from EDA
    data_quality_summary: str = ""
    data_quality_warnings: list[str] = field(default_factory=list)
    data_quality_grade: str = ""  # good, acceptable, poor

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 60,
            "ANALYST TRAINING REPORT",
            "=" * 60,
            "",
            f"Model: {self.model_type.value}",
            f"Task: {self.task_type.value}",
            f"Timestamp: {self.timestamp}",
            "",
        ]

        # Data quality section (if available)
        if self.data_quality_grade:
            grade_icon = {"good": "✓", "acceptable": "~", "poor": "✗"}[self.data_quality_grade]
            lines.extend([
                "## Data Quality",
                f"  Grade: {self.data_quality_grade.upper()} [{grade_icon}]",
            ])
            for w in self.data_quality_warnings[:3]:
                lines.append(f"    ⚠ {w}")
            lines.append("")

        lines.extend([
            "## Data Summary",
            f"  Samples: {self.n_samples}",
            f"  Features: {self.n_features}",
            f"  Train/Test: {self.train_size}/{self.test_size}",
        ])

        if self.n_classes > 0:
            lines.append(f"  Classes: {self.n_classes}")

        lines.extend(["", "## Performance Metrics"])
        for m in self.metrics:
            status = "OK" if m.is_good else "WARN"
            lines.append(f"  {m.name}: {m.value:.4f} [{status}]")
            lines.append(f"    → {m.interpretation}")

        if self.cv_scores:
            lines.extend([
                "",
                "## Cross-Validation",
                f"  Scores: {[f'{s:.3f}' for s in self.cv_scores]}",
                f"  Mean: {self.cv_mean:.4f} (±{self.cv_std:.4f})",
            ])

        if self.feature_importance:
            lines.extend(["", "## Top Features"])
            for fi in self.feature_importance[:10]:
                lines.append(f"  {fi.rank}. {fi.feature}: {fi.importance:.4f}")

        if self.warnings:
            lines.extend(["", "## Warnings"])
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        if self.recommendations:
            lines.extend(["", "## Recommendations"])
            for r in self.recommendations:
                lines.append(f"  → {r}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        info = get_model_info(self.model_type)

        lines = [
            "# ML Training Report",
            "",
            f"**Generated:** {self.timestamp}",
            "",
            "---",
            "",
            "## Model Overview",
            "",
            f"**Model:** {info.name}",
            "",
            f"**Task:** {self.task_type.value.title()}",
            "",
            f"> {info.description}",
            "",
            "---",
            "",
            "## Data Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total samples | {self.n_samples} |",
            f"| Features | {self.n_features} |",
            f"| Training set | {self.train_size} |",
            f"| Test set | {self.test_size} |",
        ]

        if self.n_classes > 0:
            lines.append(f"| Classes | {self.n_classes} |")

        # Data Quality section (if available)
        if self.data_quality_grade:
            grade_emoji = {"good": "✅", "acceptable": "⚠️", "poor": "🚨"}[self.data_quality_grade]
            lines.extend([
                "",
                "---",
                "",
                "## Data Quality Assessment",
                "",
                f"**Overall Quality:** {self.data_quality_grade.upper()} {grade_emoji}",
                "",
            ])
            if self.data_quality_warnings:
                lines.append("**Issues Found:**")
                lines.append("")
                for w in self.data_quality_warnings:
                    lines.append(f"- {w}")
                lines.append("")
            else:
                lines.append("No data quality issues detected.")
                lines.append("")

        lines.extend([
            "",
            "---",
            "",
            "## Performance Metrics",
            "",
            "| Metric | Value | Status | Interpretation |",
            "|--------|-------|--------|----------------|",
        ])

        for m in self.metrics:
            status = "✓" if m.is_good else "⚠"
            lines.append(f"| {m.name} | {m.value:.4f} | {status} | {m.interpretation} |")

        if self.cv_scores:
            lines.extend([
                "",
                "### Cross-Validation",
                "",
                f"**5-Fold CV:** {self.cv_mean:.4f} (±{self.cv_std:.4f})",
                "",
                "Cross-validation tests how well your model generalizes to unseen data. "
                "A small standard deviation means consistent performance across folds.",
                "",
            ])

        if self.feature_importance:
            lines.extend([
                "---",
                "",
                "## Feature Importance",
                "",
                "These are the features that most influence the model's predictions:",
                "",
                "| Rank | Feature | Importance |",
                "|------|---------|------------|",
            ])
            for fi in self.feature_importance[:10]:
                lines.append(f"| {fi.rank} | {fi.feature} | {fi.importance:.4f} |")

            if len(self.feature_importance) > 10:
                lines.append(f"| ... | *{len(self.feature_importance) - 10} more features* | |")

        if self.warnings:
            lines.extend([
                "",
                "---",
                "",
                "## Warnings",
                "",
            ])
            for w in self.warnings:
                lines.append(f"- ⚠️ {w}")

        if self.recommendations:
            lines.extend([
                "",
                "---",
                "",
                "## Recommendations for Improvement",
                "",
            ])
            for i, r in enumerate(self.recommendations, 1):
                lines.append(f"{i}. {r}")

        # Educational section
        lines.extend([
            "",
            "---",
            "",
            "## Understanding Your Model",
            "",
            f"### How {info.name} Works",
            "",
            info.how_it_works.strip(),
            "",
            "### Strengths",
            "",
        ])
        for s in info.strengths:
            lines.append(f"- {s}")

        lines.extend([
            "",
            "### Limitations",
            "",
        ])
        for w in info.weaknesses:
            lines.append(f"- {w}")

        lines.extend([
            "",
            "### Key Hyperparameters",
            "",
            "| Parameter | Description |",
            "|-----------|-------------|",
        ])
        for param, desc in info.hyperparameters.items():
            lines.append(f"| {param} | {desc} |")

        lines.extend([
            "",
            "---",
            "",
            "*Generated by Svend Analyst*",
        ])

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "model_type": self.model_type.value,
            "task_type": self.task_type.value,
            "timestamp": self.timestamp,
            "data": {
                "n_samples": self.n_samples,
                "n_features": self.n_features,
                "n_classes": self.n_classes,
                "target_column": self.target_column,
                "feature_columns": self.feature_columns,
            },
            "training": {
                "train_size": self.train_size,
                "test_size": self.test_size,
                "time_seconds": self.training_time_seconds,
            },
            "metrics": [
                {"name": m.name, "value": m.value, "is_good": m.is_good}
                for m in self.metrics
            ],
            "feature_importance": [
                {"feature": f.feature, "importance": f.importance, "rank": f.rank}
                for f in self.feature_importance
            ],
            "cross_validation": {
                "scores": self.cv_scores,
                "mean": self.cv_mean,
                "std": self.cv_std,
            },
            "data_quality": {
                "grade": self.data_quality_grade,
                "warnings": self.data_quality_warnings,
                "summary": self.data_quality_summary,
            },
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


class AnalystReporter:
    """
    Generate educational reports for trained models.

    Usage:
        reporter = AnalystReporter()
        report = reporter.create_report(
            model=trained_model,
            model_type=ModelType.RANDOM_FOREST,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
        )
        print(report.to_markdown())
    """

    def create_report(
        self,
        model,
        model_type: ModelType,
        task_type: TaskType,
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names: list[str] = None,
        target_name: str = "target",
        training_time: float = 0.0,
        data_quality: "DataQualityAssessment" = None,
    ) -> AnalystReport:
        """Create a comprehensive training report."""
        import numpy as np

        report = AnalystReport(
            model_type=model_type,
            task_type=task_type,
            n_samples=len(X_train) + len(X_test),
            n_features=X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]),
            train_size=len(X_train),
            test_size=len(X_test),
            training_time_seconds=training_time,
            target_column=target_name,
            feature_columns=feature_names or [f"feature_{i}" for i in range(X_train.shape[1])],
        )

        if task_type == TaskType.CLASSIFICATION:
            report.n_classes = len(np.unique(y_train))

        # Include data quality from EDA
        if data_quality:
            report.data_quality_grade = data_quality.overall_quality
            report.data_quality_summary = data_quality.summary()
            report.data_quality_warnings = [str(w) for w in data_quality.warnings]

        # Calculate metrics
        report.metrics = self._calculate_metrics(model, X_test, y_test, task_type)

        # Cross-validation
        report.cv_scores, report.cv_mean, report.cv_std = self._cross_validate(
            model, X_train, y_train, task_type
        )

        # Feature importance
        report.feature_importance = self._get_feature_importance(
            model, model_type, feature_names or report.feature_columns
        )

        # Generate warnings and recommendations
        report.warnings = self._generate_warnings(report, model, X_train, y_train)
        report.recommendations = self._generate_recommendations(report, model_type, data_quality)

        return report

    def _calculate_metrics(self, model, X_test, y_test, task_type: TaskType) -> list[MetricResult]:
        """Calculate evaluation metrics with interpretations."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_squared_error, mean_absolute_error, r2_score
        )
        import numpy as np

        metrics = []
        y_pred = model.predict(X_test)

        if task_type == TaskType.CLASSIFICATION:
            # Accuracy
            acc = accuracy_score(y_test, y_pred)
            metrics.append(MetricResult(
                name="Accuracy",
                value=acc,
                interpretation=self._interpret_accuracy(acc),
                is_good=acc > 0.7,
            ))

            # Precision, Recall, F1 (handle multiclass)
            n_classes = len(np.unique(y_test))
            average = 'binary' if n_classes == 2 else 'weighted'

            prec = precision_score(y_test, y_pred, average=average, zero_division=0)
            metrics.append(MetricResult(
                name="Precision",
                value=prec,
                interpretation=self._interpret_precision(prec),
                is_good=prec > 0.7,
            ))

            rec = recall_score(y_test, y_pred, average=average, zero_division=0)
            metrics.append(MetricResult(
                name="Recall",
                value=rec,
                interpretation=self._interpret_recall(rec),
                is_good=rec > 0.7,
            ))

            f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
            metrics.append(MetricResult(
                name="F1 Score",
                value=f1,
                interpretation=self._interpret_f1(f1),
                is_good=f1 > 0.7,
            ))

        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            metrics.append(MetricResult(
                name="RMSE",
                value=rmse,
                interpretation=f"Average prediction error of {rmse:.2f} units",
                is_good=True,  # Context-dependent
            ))

            mae = mean_absolute_error(y_test, y_pred)
            metrics.append(MetricResult(
                name="MAE",
                value=mae,
                interpretation=f"Typical prediction off by {mae:.2f} units",
                is_good=True,
            ))

            r2 = r2_score(y_test, y_pred)
            metrics.append(MetricResult(
                name="R² Score",
                value=r2,
                interpretation=self._interpret_r2(r2),
                is_good=r2 > 0.5,
            ))

        return metrics

    def _interpret_accuracy(self, acc: float) -> str:
        if acc > 0.95:
            return "Excellent - but verify it's not overfitting"
        elif acc > 0.85:
            return "Good predictive performance"
        elif acc > 0.7:
            return "Reasonable, may need improvement"
        else:
            return "Low accuracy - consider feature engineering or different model"

    def _interpret_precision(self, prec: float) -> str:
        if prec > 0.9:
            return "High precision - few false positives"
        elif prec > 0.7:
            return "Good precision"
        else:
            return "Many false positives - model is too permissive"

    def _interpret_recall(self, rec: float) -> str:
        if rec > 0.9:
            return "High recall - catches most positive cases"
        elif rec > 0.7:
            return "Good recall"
        else:
            return "Missing many positive cases - model is too conservative"

    def _interpret_f1(self, f1: float) -> str:
        if f1 > 0.9:
            return "Excellent balance of precision and recall"
        elif f1 > 0.7:
            return "Good overall performance"
        else:
            return "Trade-off between precision and recall needs attention"

    def _interpret_r2(self, r2: float) -> str:
        if r2 > 0.9:
            return f"Explains {r2*100:.0f}% of variance - excellent fit"
        elif r2 > 0.7:
            return f"Explains {r2*100:.0f}% of variance - good fit"
        elif r2 > 0.5:
            return f"Explains {r2*100:.0f}% of variance - moderate fit"
        else:
            return f"Only explains {r2*100:.0f}% of variance - weak model"

    def _cross_validate(self, model, X, y, task_type: TaskType) -> tuple:
        """Perform cross-validation."""
        from sklearn.model_selection import cross_val_score
        import numpy as np

        scoring = 'accuracy' if task_type == TaskType.CLASSIFICATION else 'r2'

        try:
            # Clone the model for CV
            from sklearn.base import clone
            model_clone = clone(model)
            scores = cross_val_score(model_clone, X, y, cv=5, scoring=scoring)
            return list(scores), float(np.mean(scores)), float(np.std(scores))
        except Exception:
            return [], 0.0, 0.0

    def _get_feature_importance(
        self,
        model,
        model_type: ModelType,
        feature_names: list[str],
    ) -> list[FeatureImportance]:
        """Extract feature importance from model."""
        import numpy as np

        importances = []

        # Try different ways to get importance
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            imp = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            coef = model.coef_
            if coef.ndim > 1:
                imp = np.abs(coef).mean(axis=0)
            else:
                imp = np.abs(coef)
        else:
            return []

        # Create sorted list
        indices = np.argsort(imp)[::-1]
        for rank, idx in enumerate(indices, 1):
            importances.append(FeatureImportance(
                feature=feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
                importance=float(imp[idx]),
                rank=rank,
            ))

        return importances

    def _generate_warnings(self, report: AnalystReport, model, X_train, y_train) -> list[str]:
        """Generate warnings based on analysis."""
        import numpy as np

        warnings = []

        # Check for overfitting signs
        if report.cv_std > 0.1:
            warnings.append(
                f"High variance in cross-validation (±{report.cv_std:.3f}). "
                "Model performance may be unstable."
            )

        # Check class imbalance
        if report.n_classes > 0:
            from collections import Counter
            counts = Counter(y_train)
            ratio = min(counts.values()) / max(counts.values())
            if ratio < 0.3:
                warnings.append(
                    f"Class imbalance detected (ratio {ratio:.2f}). "
                    "Consider using class_weight='balanced' or SMOTE."
                )

        # Check for low-performing metrics
        for m in report.metrics:
            if not m.is_good:
                warnings.append(f"Low {m.name} ({m.value:.3f}). {m.interpretation}")

        # Check feature importance concentration
        if report.feature_importance:
            top_importance = sum(f.importance for f in report.feature_importance[:3])
            total_importance = sum(f.importance for f in report.feature_importance)
            if total_importance > 0 and top_importance / total_importance > 0.8:
                warnings.append(
                    "Model relies heavily on top 3 features. "
                    "Consider if other features are needed."
                )

        return warnings

    def _generate_recommendations(
        self,
        report: AnalystReport,
        model_type: ModelType,
        data_quality: "DataQualityAssessment" = None,
    ) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Data quality recommendations (highest priority)
        if data_quality and data_quality.warnings:
            for w in data_quality.warnings:
                if w.category == "missing":
                    recommendations.append(
                        "Address missing data before training: use imputation (mean/median/KNN) "
                        "or remove rows with missing values if they're a small fraction."
                    )
                    break  # Only one recommendation per category

            for w in data_quality.warnings:
                if w.category == "outliers":
                    recommendations.append(
                        "Review flagged outliers: consider removing clear errors, "
                        "or use robust scaling (RobustScaler) to reduce their impact."
                    )
                    break

            for w in data_quality.warnings:
                if w.category == "correlation":
                    recommendations.append(
                        "Multicollinearity detected: consider removing one feature from "
                        "highly correlated pairs, or use PCA for dimensionality reduction."
                    )
                    break

            for w in data_quality.warnings:
                if w.category == "imbalance":
                    recommendations.append(
                        "Class imbalance detected: use class_weight='balanced', "
                        "SMOTE oversampling, or stratified sampling for better results."
                    )
                    break

        # Based on metrics
        acc_metric = next((m for m in report.metrics if m.name == "Accuracy"), None)
        if acc_metric and acc_metric.value < 0.8:
            recommendations.append(
                "Try feature engineering: create interaction terms, "
                "polynomial features, or domain-specific transformations."
            )

        r2_metric = next((m for m in report.metrics if m.name == "R² Score"), None)
        if r2_metric and r2_metric.value < 0.7:
            recommendations.append(
                "Consider adding more relevant features or trying "
                "a non-linear model if relationships are complex."
            )

        # Model-specific recommendations
        if model_type in [ModelType.LINEAR_REGRESSION, ModelType.LOGISTIC_REGRESSION]:
            recommendations.append(
                "Linear models benefit from feature scaling. "
                "Consider StandardScaler or MinMaxScaler."
            )

        if model_type == ModelType.KNN:
            recommendations.append(
                "KNN is sensitive to feature scales. Ensure features are normalized. "
                "Try different values of k (neighbors)."
            )

        if model_type == ModelType.RANDOM_FOREST:
            recommendations.append(
                "Tune n_estimators (more trees = better but slower) "
                "and max_depth to balance bias-variance."
            )

        if model_type == ModelType.GRADIENT_BOOSTING:
            recommendations.append(
                "Start with a low learning_rate (0.01-0.1) and increase n_estimators. "
                "Use early stopping to prevent overfitting."
            )

        # General recommendations
        if report.cv_std > 0.05:
            recommendations.append(
                "Cross-validation variance is high. Try collecting more data "
                "or using regularization."
            )

        if not report.feature_importance:
            recommendations.append(
                "Consider using a model that provides feature importance "
                "to understand what drives predictions."
            )

        return recommendations

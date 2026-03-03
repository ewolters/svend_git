"""
Unified Quality Framework

Base classes for quality reporting across all agents.
Provides consistent quality metrics, grading, and reporting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class QualityGrade(Enum):
    """Standardized quality grades."""
    A = "A"  # Excellent: 90-100
    B = "B"  # Good: 80-89
    C = "C"  # Acceptable: 70-79
    D = "D"  # Poor: 60-69
    F = "F"  # Failing: <60

    @classmethod
    def from_score(cls, score: float) -> "QualityGrade":
        """Convert numeric score (0-100) to grade."""
        if score >= 90:
            return cls.A
        elif score >= 80:
            return cls.B
        elif score >= 70:
            return cls.C
        elif score >= 60:
            return cls.D
        return cls.F


class QualitySeverity(Enum):
    """Severity levels for quality issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityIssue:
    """A single quality issue."""
    severity: QualitySeverity
    category: str
    message: str
    details: dict = field(default_factory=dict)
    location: str = ""  # Where in the output

    def __str__(self):
        icons = {
            QualitySeverity.INFO: "ℹ️",
            QualitySeverity.WARNING: "⚠️",
            QualitySeverity.ERROR: "❌",
            QualitySeverity.CRITICAL: "🚨",
        }
        icon = icons.get(self.severity, "•")
        return f"{icon} [{self.category}] {self.message}"


@dataclass
class QualityMetric:
    """A single quality metric."""
    name: str
    value: float
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    unit: str = ""
    description: str = ""

    @property
    def passed(self) -> bool:
        """Check if metric passes thresholds."""
        if self.min_threshold is not None and self.value < self.min_threshold:
            return False
        if self.max_threshold is not None and self.value > self.max_threshold:
            return False
        return True

    def __str__(self):
        status = "✓" if self.passed else "✗"
        value_str = f"{self.value:.2f}" if isinstance(self.value, float) else str(self.value)
        if self.unit:
            value_str += f" {self.unit}"
        return f"[{status}] {self.name}: {value_str}"


@dataclass
class QualityReport(ABC):
    """
    Base class for all quality reports.

    Subclass this for each agent's quality assessment.
    """
    # Required fields
    agent_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Quality assessment
    score: float = 100.0  # 0-100
    grade: QualityGrade = QualityGrade.A
    passed: bool = True

    # Details
    metrics: list[QualityMetric] = field(default_factory=list)
    issues: list[QualityIssue] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # For chaining
    upstream_quality: Optional["QualityReport"] = None

    def add_metric(self, name: str, value: float, **kwargs) -> "QualityReport":
        """Add a quality metric."""
        self.metrics.append(QualityMetric(name=name, value=value, **kwargs))
        return self

    def add_issue(
        self,
        severity: QualitySeverity,
        category: str,
        message: str,
        **kwargs,
    ) -> "QualityReport":
        """Add a quality issue."""
        self.issues.append(QualityIssue(
            severity=severity,
            category=category,
            message=message,
            **kwargs,
        ))
        return self

    def add_recommendation(self, recommendation: str) -> "QualityReport":
        """Add a recommendation."""
        self.recommendations.append(recommendation)
        return self

    def calculate_grade(self) -> QualityGrade:
        """Calculate grade from score and issues."""
        # Start with score-based grade
        base_grade = QualityGrade.from_score(self.score)

        # Downgrade for critical issues
        critical_count = sum(1 for i in self.issues if i.severity == QualitySeverity.CRITICAL)
        error_count = sum(1 for i in self.issues if i.severity == QualitySeverity.ERROR)

        grade_value = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}[base_grade.value]

        if critical_count > 0:
            grade_value = max(0, grade_value - 2)
        if error_count > 2:
            grade_value = max(0, grade_value - 1)

        grade_map = {4: QualityGrade.A, 3: QualityGrade.B, 2: QualityGrade.C,
                     1: QualityGrade.D, 0: QualityGrade.F}
        return grade_map[grade_value]

    def finalize(self) -> "QualityReport":
        """Finalize the report - calculate grade and passed status."""
        self.grade = self.calculate_grade()
        self.passed = self.grade.value in ["A", "B", "C"]

        # Check if all metrics pass
        failed_metrics = [m for m in self.metrics if not m.passed]
        if failed_metrics:
            self.passed = False

        return self

    @property
    def critical_issues(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == QualitySeverity.CRITICAL]

    @property
    def error_issues(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == QualitySeverity.ERROR]

    @property
    def warning_issues(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == QualitySeverity.WARNING]

    def summary(self) -> str:
        """Generate text summary."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            "=" * 50,
            f"{self.agent_name} QUALITY REPORT: {status}",
            "=" * 50,
            "",
            f"Score: {self.score:.0f}/100",
            f"Grade: {self.grade.value}",
            "",
        ]

        if self.metrics:
            lines.append("## Metrics")
            for m in self.metrics:
                lines.append(f"  {m}")
            lines.append("")

        if self.issues:
            lines.append("## Issues")
            for i in self.issues:
                lines.append(f"  {i}")
            lines.append("")

        if self.recommendations:
            lines.append("## Recommendations")
            for r in self.recommendations[:5]:
                lines.append(f"  → {r}")
            lines.append("")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
            "score": self.score,
            "grade": self.grade.value,
            "passed": self.passed,
            "metrics": [
                {"name": m.name, "value": m.value, "passed": m.passed}
                for m in self.metrics
            ],
            "issues": [
                {"severity": i.severity.value, "category": i.category, "message": i.message}
                for i in self.issues
            ],
            "recommendations": self.recommendations,
        }


# =============================================================================
# Agent-Specific Quality Reports
# =============================================================================

@dataclass
class ResearcherQualityReport(QualityReport):
    """Quality report for Researcher agent."""
    agent_name: str = "Researcher"

    # Research-specific metrics
    source_count: int = 0
    source_diversity: float = 0.0
    claim_support_rate: float = 0.0
    consistency_score: float = 0.0


@dataclass
class WriterQualityReport(QualityReport):
    """Quality report for Writer agent."""
    agent_name: str = "Writer"

    # Writing-specific metrics
    readability_grade: float = 0.0
    grammar_score: float = 0.0
    template_compliant: bool = True
    citation_confidence: float = 0.0


@dataclass
class CoderQualityReport(QualityReport):
    """Quality report for Coder agent."""
    agent_name: str = "Coder"

    # Code-specific metrics
    syntax_valid: bool = True
    execution_successful: bool = False
    complexity_score: float = 0.0
    security_issues: int = 0


@dataclass
class AnalystQualityReport(QualityReport):
    """Quality report for Analyst agent."""
    agent_name: str = "Analyst"

    # ML-specific metrics
    data_quality_grade: str = ""
    model_accuracy: float = 0.0
    feature_count: int = 0
    cv_variance: float = 0.0


@dataclass
class ExperimenterQualityReport(QualityReport):
    """Quality report for Experimenter agent."""
    agent_name: str = "Experimenter"

    # Experiment-specific metrics
    power: float = 0.0
    sample_size: int = 0
    design_valid: bool = True


# =============================================================================
# Quality Chain for Pipeline Propagation
# =============================================================================

@dataclass
class QualityChain:
    """
    Tracks quality through a pipeline.

    Allows downstream agents to see upstream quality issues.
    """
    reports: list[QualityReport] = field(default_factory=list)

    def add(self, report: QualityReport) -> "QualityChain":
        """Add a report to the chain."""
        if self.reports:
            report.upstream_quality = self.reports[-1]
        self.reports.append(report)
        return self

    @property
    def overall_passed(self) -> bool:
        """Check if all reports in chain passed."""
        return all(r.passed for r in self.reports)

    @property
    def lowest_grade(self) -> QualityGrade:
        """Get the lowest grade in the chain."""
        if not self.reports:
            return QualityGrade.A
        grades = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        min_grade = min(grades[r.grade.value] for r in self.reports)
        return {4: QualityGrade.A, 3: QualityGrade.B, 2: QualityGrade.C,
                1: QualityGrade.D, 0: QualityGrade.F}[min_grade]

    @property
    def all_issues(self) -> list[QualityIssue]:
        """Get all issues from all reports."""
        issues = []
        for report in self.reports:
            for issue in report.issues:
                issues.append(QualityIssue(
                    severity=issue.severity,
                    category=f"{report.agent_name}:{issue.category}",
                    message=issue.message,
                    details=issue.details,
                ))
        return issues

    def summary(self) -> str:
        """Generate chain summary."""
        lines = [
            "=" * 50,
            "QUALITY CHAIN SUMMARY",
            "=" * 50,
            "",
            f"Agents: {' → '.join(r.agent_name for r in self.reports)}",
            f"Overall: {'PASSED' if self.overall_passed else 'FAILED'}",
            f"Lowest Grade: {self.lowest_grade.value}",
            "",
        ]

        for report in self.reports:
            status = "✓" if report.passed else "✗"
            lines.append(f"  [{status}] {report.agent_name}: Grade {report.grade.value} ({report.score:.0f}/100)")

        critical = [i for i in self.all_issues if i.severity == QualitySeverity.CRITICAL]
        if critical:
            lines.extend(["", "Critical Issues:"])
            for issue in critical[:5]:
                lines.append(f"  🚨 {issue.category}: {issue.message}")

        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)


# =============================================================================
# Helper Functions
# =============================================================================

def create_quality_report(
    agent_name: str,
    score: float,
    metrics: dict = None,
    issues: list = None,
) -> QualityReport:
    """
    Quick helper to create a quality report.

    Args:
        agent_name: Name of the agent
        score: Quality score 0-100
        metrics: Dict of metric_name -> value
        issues: List of (severity, category, message) tuples
    """
    # Use the generic QualityReport by creating a concrete instance
    @dataclass
    class GenericQualityReport(QualityReport):
        pass

    report = GenericQualityReport(agent_name=agent_name, score=score)

    if metrics:
        for name, value in metrics.items():
            report.add_metric(name, value)

    if issues:
        for severity, category, message in issues:
            if isinstance(severity, str):
                severity = QualitySeverity(severity)
            report.add_issue(severity, category, message)

    return report.finalize()

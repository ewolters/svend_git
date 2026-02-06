"""
Data Cleaner - Main Orchestrator

Combines all cleaning operations:
1. Type inference and correction
2. Missing data handling
3. Outlier detection (flag, don't remove)
4. Factor normalization

Produces clean data + comprehensive report.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from .outliers import OutlierDetector, OutlierResult, OutlierMethod
from .missing import MissingHandler, MissingResult, ImputationStrategy
from .normalize import FactorNormalizer, NormalizationResult
from .types import TypeInferrer, TypeResult


@dataclass
class CleaningConfig:
    """Configuration for data cleaning."""
    # Outlier detection
    detect_outliers: bool = True
    outlier_methods: list[OutlierMethod] = field(default_factory=lambda: [OutlierMethod.IQR])
    domain_rules: dict = field(default_factory=dict)  # column -> (min, max)

    # Missing data
    handle_missing: bool = True
    imputation_strategies: dict = field(default_factory=dict)  # column -> strategy
    drop_threshold: float = 0.5  # Drop columns with > 50% missing

    # Normalization
    normalize_factors: bool = True
    case_style: str = "title"
    custom_mappings: dict = field(default_factory=dict)

    # Type correction
    correct_types: bool = True
    target_types: dict = field(default_factory=dict)


@dataclass
class CleaningResult:
    """Complete result of data cleaning."""
    original_shape: tuple
    cleaned_shape: tuple
    outliers: OutlierResult = None
    missing: MissingResult = None
    normalization: NormalizationResult = None
    types: TypeResult = None
    warnings: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "SCRUB DATA CLEANING REPORT",
            "=" * 50,
            "",
            f"Original shape: {self.original_shape[0]} rows × {self.original_shape[1]} columns",
            f"Cleaned shape: {self.cleaned_shape[0]} rows × {self.cleaned_shape[1]} columns",
            f"Timestamp: {self.timestamp}",
            "",
        ]

        if self.types:
            lines.append("## Type Corrections")
            lines.append(self.types.summary())
            lines.append("")

        if self.missing:
            lines.append("## Missing Data")
            lines.append(self.missing.summary())
            lines.append("")

        if self.outliers and self.outliers.count > 0:
            lines.append("## Outliers Flagged (Review Required)")
            lines.append(self.outliers.summary())
            lines.append("")

        if self.normalization and self.normalization.total_changes > 0:
            lines.append("## Factor Normalization")
            lines.append(self.normalization.summary())
            lines.append("")

        if self.warnings:
            lines.append("## Warnings")
            for w in self.warnings:
                lines.append(f"  - {w}")
            lines.append("")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "original_shape": self.original_shape,
            "cleaned_shape": self.cleaned_shape,
            "outliers_count": self.outliers.count if self.outliers else 0,
            "missing_filled": self.missing.total_filled if self.missing else 0,
            "normalizations": self.normalization.total_changes if self.normalization else 0,
            "warnings": self.warnings,
            "timestamp": self.timestamp,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Data Cleaning Report",
            "",
            f"**Generated:** {self.timestamp}",
            "",
            "## Overview",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Original rows | {self.original_shape[0]} |",
            f"| Original columns | {self.original_shape[1]} |",
            f"| Cleaned rows | {self.cleaned_shape[0]} |",
            f"| Cleaned columns | {self.cleaned_shape[1]} |",
            "",
        ]

        if self.types and self.types.corrections:
            lines.extend([
                "## Type Corrections",
                "",
                "| Column | From | To | Status |",
                "|--------|------|----|---------",
            ])
            for corr in self.types.corrections:
                status = "OK" if corr.failed == 0 else f"{corr.failed} failed"
                lines.append(f"| {corr.column} | {corr.original_dtype} | {corr.target_dtype} | {status} |")
            lines.append("")

        if self.missing and self.missing.columns:
            lines.extend([
                "## Missing Data Handling",
                "",
                "| Column | Missing | Strategy | Fill Value |",
                "|--------|---------|----------|------------|",
            ])
            for col in self.missing.columns:
                lines.append(
                    f"| {col.column} | {col.missing_count} ({col.missing_percent:.1f}%) | "
                    f"{col.strategy_used.value} | {col.fill_value or 'N/A'} |"
                )
            lines.append("")

        if self.outliers and self.outliers.count > 0:
            lines.extend([
                "## Outliers Flagged",
                "",
                "**These require human review.** Outliers are flagged but not removed.",
                "",
                "| Row | Column | Value | Reason | Severity |",
                "|-----|--------|-------|--------|----------|",
            ])
            for flag in self.outliers.flags[:20]:  # Top 20
                lines.append(
                    f"| {flag.row_index} | {flag.column} | {flag.value} | "
                    f"{flag.reason} | {flag.severity} |"
                )
            if self.outliers.count > 20:
                lines.append(f"\n*...and {self.outliers.count - 20} more outliers*")
            lines.append("")

        if self.normalization and self.normalization.total_changes > 0:
            lines.extend([
                "## Factor Normalization",
                "",
            ])
            for col, mapping in self.normalization.mappings.items():
                lines.append(f"### {col}")
                lines.append("")
                lines.append("| Original | Normalized |")
                lines.append("|----------|------------|")
                for orig, norm in list(mapping.items())[:10]:
                    lines.append(f"| {orig} | {norm} |")
                if len(mapping) > 10:
                    lines.append(f"\n*...and {len(mapping) - 10} more*")
                lines.append("")

        if self.warnings:
            lines.extend([
                "## Warnings",
                "",
            ])
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        return "\n".join(lines)


class DataCleaner:
    """
    Main data cleaning orchestrator.

    Usage:
        cleaner = DataCleaner()
        df_clean, result = cleaner.clean(df)
        print(result.summary())

        # Or with config
        config = CleaningConfig(
            detect_outliers=True,
            domain_rules={'age': (0, 120), 'salary': (0, None)},
        )
        df_clean, result = cleaner.clean(df, config)
    """

    def __init__(self):
        self.outlier_detector = OutlierDetector()
        self.missing_handler = MissingHandler()
        self.normalizer = FactorNormalizer()
        self.type_inferrer = TypeInferrer()

    # Excel error values to treat as NaN
    EXCEL_ERRORS = ['#NUM!', '#DIV/0!', '#VALUE!', '#REF!', '#NAME?', '#N/A', '#NULL!', '#ERROR!']

    def clean(
        self,
        df,
        config: CleaningConfig = None,
    ) -> tuple:
        """
        Clean a dataframe.

        Args:
            df: pandas DataFrame
            config: CleaningConfig (optional)

        Returns:
            (cleaned_df, CleaningResult)
        """
        import pandas as pd
        import numpy as np

        config = config or CleaningConfig()
        df = df.copy()

        result = CleaningResult(
            original_shape=(len(df), len(df.columns)),
            cleaned_shape=(0, 0),  # Will update
        )

        # 0. Replace Excel error values with NaN (before any other processing)
        excel_error_count = 0
        for col in df.columns:
            mask = df[col].isin(self.EXCEL_ERRORS)
            count = mask.sum()
            if count > 0:
                df.loc[mask, col] = np.nan
                excel_error_count += count
        if excel_error_count > 0:
            result.warnings.append(f"Replaced {excel_error_count} Excel error values (#NUM!, #DIV/0!, etc.) with NaN")

        # 1. Type correction (first, so other steps work correctly)
        if config.correct_types:
            df, result.types = self.type_inferrer.correct_types(
                df,
                target_types=config.target_types,
            )
            if result.types.corrections:
                failed_cols = [c.column for c in result.types.corrections if c.failed > 0]
                if failed_cols:
                    result.warnings.append(
                        f"Type conversion had failures in: {', '.join(failed_cols)}"
                    )

        # 2. Missing data handling
        if config.handle_missing:
            df, result.missing = self.missing_handler.handle(
                df,
                strategies=config.imputation_strategies,
            )
            if result.missing.rows_dropped > 0:
                result.warnings.append(
                    f"{result.missing.rows_dropped} rows dropped (>80% missing)"
                )

        # 3. Factor normalization
        if config.normalize_factors:
            # Get categorical columns
            cat_cols = df.select_dtypes(
                include=['object', 'string', 'category']
            ).columns.tolist()

            if cat_cols:
                df, result.normalization = self.normalizer.normalize(
                    df,
                    columns=cat_cols,
                    custom_mappings=config.custom_mappings,
                    case_style=config.case_style,
                )

        # 4. Outlier detection (last, after cleaning)
        if config.detect_outliers:
            result.outliers = self.outlier_detector.detect(
                df,
                methods=config.outlier_methods,
                domain_rules=config.domain_rules,
            )
            if result.outliers.high_severity_count > 0:
                result.warnings.append(
                    f"{result.outliers.high_severity_count} high-severity outliers require review"
                )

        result.cleaned_shape = (len(df), len(df.columns))

        return df, result

    def analyze(self, df) -> dict:
        """
        Analyze data without modifying it.

        Returns dict with analysis of what would be cleaned.
        """
        import pandas as pd

        analysis = {
            "shape": (len(df), len(df.columns)),
            "columns": {},
        }

        for col in df.columns:
            col_analysis = {
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "missing_pct": float(df[col].isna().sum() / len(df) * 100),
                "unique": int(df[col].nunique()),
            }

            # Inferred type
            inferred = self.type_inferrer.infer_types(df[[col]])
            col_analysis["inferred_type"] = inferred.get(col, str(df[col].dtype))

            # For numeric columns, add stats
            if pd.api.types.is_numeric_dtype(df[col]):
                col_analysis["min"] = float(df[col].min()) if pd.notna(df[col].min()) else None
                col_analysis["max"] = float(df[col].max()) if pd.notna(df[col].max()) else None
                col_analysis["mean"] = float(df[col].mean()) if pd.notna(df[col].mean()) else None

            # For categorical, show value counts
            else:
                vc = df[col].value_counts().head(5).to_dict()
                col_analysis["top_values"] = {str(k): int(v) for k, v in vc.items()}

            analysis["columns"][col] = col_analysis

        return analysis

    def save_report(self, result: CleaningResult, path: Path, format: str = "markdown"):
        """Save cleaning report to file."""
        path = Path(path)

        if format == "markdown":
            path.write_text(result.to_markdown())
        elif format == "json":
            path.write_text(json.dumps(result.to_dict(), indent=2))
        elif format == "text":
            path.write_text(result.summary())
        else:
            raise ValueError(f"Unknown format: {format}")

        return path


def quick_clean(df, **kwargs):
    """Quick helper to clean data with defaults."""
    config = CleaningConfig(**kwargs)
    cleaner = DataCleaner()
    return cleaner.clean(df, config)

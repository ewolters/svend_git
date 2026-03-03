"""
Outlier Detection

Multiple methods for detecting outliers:
- IQR (Interquartile Range) - simple, robust
- Z-score - for normally distributed data
- Isolation Forest - for complex patterns
- Domain rules - user-defined constraints

Outliers are FLAGGED, not removed. Human decides.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal
from enum import Enum


class OutlierMethod(Enum):
    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"


@dataclass
class OutlierFlag:
    """A flagged outlier for review."""
    row_index: int
    column: str
    value: any
    reason: str
    method: str
    severity: Literal["low", "medium", "high"]
    suggestion: str = ""

    def to_dict(self) -> dict:
        return {
            "row": self.row_index,
            "column": self.column,
            "value": self.value,
            "reason": self.reason,
            "method": self.method,
            "severity": self.severity,
            "suggestion": self.suggestion,
        }


@dataclass
class OutlierResult:
    """Result of outlier detection."""
    flags: list[OutlierFlag] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.flags)

    @property
    def high_severity_count(self) -> int:
        return len([f for f in self.flags if f.severity == "high"])

    def by_column(self) -> dict[str, list[OutlierFlag]]:
        result = {}
        for flag in self.flags:
            if flag.column not in result:
                result[flag.column] = []
            result[flag.column].append(flag)
        return result

    def summary(self) -> str:
        lines = [f"Outliers detected: {self.count}"]
        if self.count > 0:
            lines.append(f"  High severity: {self.high_severity_count}")
            by_col = self.by_column()
            for col, flags in by_col.items():
                lines.append(f"  {col}: {len(flags)} outliers")
        return "\n".join(lines)


class OutlierDetector:
    """
    Detect outliers using multiple methods.

    Usage:
        detector = OutlierDetector()
        result = detector.detect(df)

        # Review flags
        for flag in result.flags:
            print(f"Row {flag.row_index}: {flag.reason}")
    """

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        zscore_threshold: float = 3.0,
        contamination: float = 0.05,
    ):
        self.iqr_multiplier = iqr_multiplier
        self.zscore_threshold = zscore_threshold
        self.contamination = contamination

    def detect(
        self,
        df,
        columns: list[str] = None,
        methods: list[OutlierMethod] = None,
        domain_rules: dict = None,
    ) -> OutlierResult:
        """
        Detect outliers in dataframe.

        Args:
            df: pandas DataFrame
            columns: Columns to check (default: all numeric)
            methods: Detection methods to use (default: IQR)
            domain_rules: Dict of column -> (min, max) constraints

        Returns:
            OutlierResult with flagged outliers
        """
        import pandas as pd

        flags = []
        stats = {}

        # Default to numeric columns
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Default to IQR method
        if methods is None:
            methods = [OutlierMethod.IQR]

        for col in columns:
            if col not in df.columns:
                continue

            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            col_stats = {
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "q1": float(col_data.quantile(0.25)),
                "q3": float(col_data.quantile(0.75)),
            }
            stats[col] = col_stats

            # IQR method
            if OutlierMethod.IQR in methods:
                flags.extend(self._detect_iqr(df, col, col_stats))

            # Z-score method
            if OutlierMethod.ZSCORE in methods:
                flags.extend(self._detect_zscore(df, col, col_stats))

            # Isolation Forest
            if OutlierMethod.ISOLATION_FOREST in methods:
                flags.extend(self._detect_isolation_forest(df, col))

        # Domain rules
        if domain_rules:
            flags.extend(self._apply_domain_rules(df, domain_rules))

        # Auto-detect obvious issues
        flags.extend(self._detect_obvious(df, columns))

        # Deduplicate flags (same row/column)
        seen = set()
        unique_flags = []
        for flag in flags:
            key = (flag.row_index, flag.column)
            if key not in seen:
                seen.add(key)
                unique_flags.append(flag)

        return OutlierResult(flags=unique_flags, stats=stats)

    def _detect_iqr(self, df, column: str, stats: dict) -> list[OutlierFlag]:
        """Detect outliers using IQR method."""
        import pandas as pd
        flags = []

        q1 = stats["q1"]
        q3 = stats["q3"]
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr

        for idx, value in df[column].items():
            if pd.isna(value):
                continue
            if value < lower:
                severity = "high" if value < q1 - 3 * iqr else "medium"
                flags.append(OutlierFlag(
                    row_index=idx,
                    column=column,
                    value=value,
                    reason=f"Below IQR lower bound ({lower:.2f})",
                    method="IQR",
                    severity=severity,
                    suggestion=f"Expected range: {lower:.2f} - {upper:.2f}",
                ))
            elif value > upper:
                severity = "high" if value > q3 + 3 * iqr else "medium"
                flags.append(OutlierFlag(
                    row_index=idx,
                    column=column,
                    value=value,
                    reason=f"Above IQR upper bound ({upper:.2f})",
                    method="IQR",
                    severity=severity,
                    suggestion=f"Expected range: {lower:.2f} - {upper:.2f}",
                ))

        return flags

    def _detect_zscore(self, df, column: str, stats: dict) -> list[OutlierFlag]:
        """Detect outliers using Z-score method."""
        import pandas as pd
        flags = []

        mean = stats["mean"]
        std = stats["std"]

        if std == 0:
            return flags

        for idx, value in df[column].items():
            if pd.isna(value):
                continue
            zscore = abs((value - mean) / std)
            if zscore > self.zscore_threshold:
                severity = "high" if zscore > 4.0 else "medium"
                flags.append(OutlierFlag(
                    row_index=idx,
                    column=column,
                    value=value,
                    reason=f"Z-score = {zscore:.2f} (threshold: {self.zscore_threshold})",
                    method="Z-score",
                    severity=severity,
                    suggestion=f"Value is {zscore:.1f} standard deviations from mean",
                ))

        return flags

    def _detect_isolation_forest(self, df, column: str) -> list[OutlierFlag]:
        """Detect outliers using Isolation Forest."""
        import pandas as pd

        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            return []

        flags = []
        col_data = df[column].dropna()

        if len(col_data) < 10:
            return flags

        X = col_data.values.reshape(-1, 1)
        iso = IsolationForest(contamination=self.contamination, random_state=42)
        predictions = iso.fit_predict(X)

        outlier_indices = col_data.index[predictions == -1]

        for idx in outlier_indices:
            value = df.loc[idx, column]
            flags.append(OutlierFlag(
                row_index=idx,
                column=column,
                value=value,
                reason="Detected by Isolation Forest",
                method="Isolation Forest",
                severity="medium",
                suggestion="Unusual pattern detected - review manually",
            ))

        return flags

    def _apply_domain_rules(self, df, rules: dict) -> list[OutlierFlag]:
        """Apply user-defined domain rules."""
        import pandas as pd
        flags = []

        for column, (min_val, max_val) in rules.items():
            if column not in df.columns:
                continue

            for idx, value in df[column].items():
                if pd.isna(value):
                    continue
                if min_val is not None and value < min_val:
                    flags.append(OutlierFlag(
                        row_index=idx,
                        column=column,
                        value=value,
                        reason=f"Below domain minimum ({min_val})",
                        method="Domain rule",
                        severity="high",
                        suggestion=f"Valid range: {min_val} - {max_val}",
                    ))
                if max_val is not None and value > max_val:
                    flags.append(OutlierFlag(
                        row_index=idx,
                        column=column,
                        value=value,
                        reason=f"Above domain maximum ({max_val})",
                        method="Domain rule",
                        severity="high",
                        suggestion=f"Valid range: {min_val} - {max_val}",
                    ))

        return flags

    def _detect_obvious(self, df, columns: list[str]) -> list[OutlierFlag]:
        """Detect obvious issues (negative values where inappropriate, etc.)."""
        import pandas as pd
        flags = []

        # Columns that shouldn't be negative
        non_negative_patterns = ["age", "salary", "price", "count", "quantity", "amount", "revenue", "cost"]

        for col in columns:
            col_lower = col.lower()

            # Check for negative values in non-negative columns
            if any(p in col_lower for p in non_negative_patterns):
                for idx, value in df[col].items():
                    if pd.isna(value):
                        continue
                    if value < 0:
                        flags.append(OutlierFlag(
                            row_index=idx,
                            column=col,
                            value=value,
                            reason=f"Negative value in '{col}' (likely should be positive)",
                            method="Domain inference",
                            severity="high",
                            suggestion="Verify this isn't a data entry error",
                        ))

            # Check for unrealistic ages
            if "age" in col_lower:
                for idx, value in df[col].items():
                    if pd.isna(value):
                        continue
                    if value > 120 or value < 0:
                        flags.append(OutlierFlag(
                            row_index=idx,
                            column=col,
                            value=value,
                            reason=f"Unrealistic age value: {value}",
                            method="Domain inference",
                            severity="high",
                            suggestion="Age should typically be 0-120",
                        ))

        return flags

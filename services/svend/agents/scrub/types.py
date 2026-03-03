"""
Type Inference and Correction

Detect and fix type issues:
- Infer correct types from data
- Convert strings to numbers/dates
- Detect mixed types
- Report conversion failures

Ensures data is ready for ML models.
"""

import re
from dataclasses import dataclass, field
from typing import Literal
from datetime import datetime


@dataclass
class TypeCorrection:
    """A type correction applied to a column."""
    column: str
    original_dtype: str
    target_dtype: str
    successful: int
    failed: int
    failed_rows: list[int] = field(default_factory=list)
    sample_failures: list[tuple] = field(default_factory=list)  # (row, value)


@dataclass
class TypeResult:
    """Result of type inference and correction."""
    corrections: list[TypeCorrection] = field(default_factory=list)
    inferred_types: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["Type analysis:"]

        for corr in self.corrections:
            status = "OK" if corr.failed == 0 else f"{corr.failed} failures"
            lines.append(f"  {corr.column}: {corr.original_dtype} → {corr.target_dtype} ({status})")

            if corr.sample_failures:
                lines.append("    Failed conversions:")
                for row, val in corr.sample_failures[:3]:
                    lines.append(f"      Row {row}: '{val}'")

        return "\n".join(lines)


class TypeInferrer:
    """
    Infer and correct data types.

    Usage:
        inferrer = TypeInferrer()
        df_typed, result = inferrer.correct_types(df)
        print(result.summary())
    """

    # Common date formats to try
    DATE_FORMATS = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]

    def infer_types(self, df) -> dict[str, str]:
        """Infer appropriate types for each column."""
        import pandas as pd

        inferred = {}

        for col in df.columns:
            current_dtype = str(df[col].dtype)
            sample = df[col].dropna()

            if len(sample) == 0:
                inferred[col] = current_dtype
                continue

            # Already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if it should be integer
                if pd.api.types.is_float_dtype(df[col]):
                    if (sample == sample.astype(int)).all():
                        inferred[col] = "int64"
                    else:
                        inferred[col] = "float64"
                else:
                    inferred[col] = current_dtype
                continue

            # Already datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                inferred[col] = "datetime64"
                continue

            # Try to infer from string content
            sample_str = sample.astype(str)

            # Try numeric
            numeric_count = sum(self._is_numeric(v) for v in sample_str.head(100))
            if numeric_count / min(100, len(sample_str)) > 0.9:
                # Check if float or int
                if any('.' in str(v) for v in sample_str.head(100)):
                    inferred[col] = "float64"
                else:
                    inferred[col] = "int64"
                continue

            # Try datetime
            date_count = sum(self._is_date(v) for v in sample_str.head(100))
            if date_count / min(100, len(sample_str)) > 0.9:
                inferred[col] = "datetime64"
                continue

            # Try boolean
            bool_values = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}
            if set(str(v).lower().strip() for v in sample_str.head(100)) <= bool_values:
                inferred[col] = "bool"
                continue

            # Keep as object/category
            unique_ratio = len(sample.unique()) / len(sample)
            if unique_ratio < 0.05:  # Low cardinality
                inferred[col] = "category"
            else:
                inferred[col] = "object"

        return inferred

    def correct_types(
        self,
        df,
        target_types: dict[str, str] = None,
        infer: bool = True,
    ) -> tuple:
        """
        Correct column types.

        Args:
            df: pandas DataFrame
            target_types: Dict of column -> target type (overrides inference)
            infer: Whether to infer types automatically

        Returns:
            (typed_df, TypeResult)
        """
        import pandas as pd

        df = df.copy()
        target_types = target_types or {}

        result = TypeResult()

        # Infer types if requested
        if infer:
            result.inferred_types = self.infer_types(df)

        # Merge with explicit targets
        types_to_apply = {**result.inferred_types, **target_types}

        for col, target in types_to_apply.items():
            if col not in df.columns:
                continue

            current = str(df[col].dtype)
            if current == target:
                continue

            # Apply conversion
            correction = self._convert_column(df, col, target)
            if correction:
                result.corrections.append(correction)

        return df, result

    def _convert_column(self, df, column: str, target: str) -> TypeCorrection:
        """Convert a column to target type."""
        import pandas as pd

        original_dtype = str(df[column].dtype)
        successful = 0
        failed = 0
        failed_rows = []
        sample_failures = []

        if target in ["int64", "int32", "int"]:
            for idx, val in df[column].items():
                if pd.isna(val):
                    continue
                try:
                    df.at[idx, column] = int(float(str(val).replace(',', '')))
                    successful += 1
                except (ValueError, TypeError):
                    failed += 1
                    failed_rows.append(idx)
                    if len(sample_failures) < 5:
                        sample_failures.append((idx, val))

            if failed == 0:
                df[column] = df[column].astype('Int64')  # Nullable int

        elif target in ["float64", "float32", "float"]:
            for idx, val in df[column].items():
                if pd.isna(val):
                    continue
                try:
                    df.at[idx, column] = float(str(val).replace(',', ''))
                    successful += 1
                except (ValueError, TypeError):
                    failed += 1
                    failed_rows.append(idx)
                    if len(sample_failures) < 5:
                        sample_failures.append((idx, val))

            if failed == 0:
                df[column] = df[column].astype('float64')

        elif target in ["datetime64", "datetime"]:
            for idx, val in df[column].items():
                if pd.isna(val):
                    continue
                parsed = self._parse_date(val)
                if parsed:
                    df.at[idx, column] = parsed
                    successful += 1
                else:
                    failed += 1
                    failed_rows.append(idx)
                    if len(sample_failures) < 5:
                        sample_failures.append((idx, val))

            if failed == 0:
                df[column] = pd.to_datetime(df[column])

        elif target == "bool":
            bool_map = {
                'true': True, 'false': False,
                'yes': True, 'no': False,
                '1': True, '0': False,
                't': True, 'f': False,
                'y': True, 'n': False,
            }
            for idx, val in df[column].items():
                if pd.isna(val):
                    continue
                key = str(val).lower().strip()
                if key in bool_map:
                    df.at[idx, column] = bool_map[key]
                    successful += 1
                else:
                    failed += 1
                    failed_rows.append(idx)
                    if len(sample_failures) < 5:
                        sample_failures.append((idx, val))

            if failed == 0:
                df[column] = df[column].astype('boolean')

        elif target == "category":
            df[column] = df[column].astype('category')
            successful = len(df[column])

        else:
            # Keep as is
            return None

        return TypeCorrection(
            column=column,
            original_dtype=original_dtype,
            target_dtype=target,
            successful=successful,
            failed=failed,
            failed_rows=failed_rows[:100],
            sample_failures=sample_failures,
        )

    def _is_numeric(self, value: str) -> bool:
        """Check if string is numeric."""
        try:
            float(str(value).replace(',', ''))
            return True
        except (ValueError, TypeError):
            return False

    def _is_date(self, value: str) -> bool:
        """Check if string is a date."""
        return self._parse_date(value) is not None

    def _parse_date(self, value) -> datetime:
        """Try to parse a date string."""
        if value is None:
            return None

        s = str(value).strip()

        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue

        # Try pandas parser as fallback
        try:
            import pandas as pd
            return pd.to_datetime(s)
        except:
            return None

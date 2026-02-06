"""
Missing Data Handler

Strategies for handling missing values:
- Drop rows/columns (only if minimal impact)
- Imputation (mean, median, mode, forward fill, etc.)
- Flag for manual review
- Model-based imputation (KNN, iterative)

Always reports what was done and why.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class ImputationStrategy(Enum):
    DROP = "drop"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    KNN = "knn"
    FLAG_ONLY = "flag"


@dataclass
class MissingInfo:
    """Information about missing values in a column."""
    column: str
    missing_count: int
    missing_percent: float
    strategy_used: ImputationStrategy
    fill_value: any = None
    rows_affected: list[int] = field(default_factory=list)


@dataclass
class MissingResult:
    """Result of missing data handling."""
    columns: list[MissingInfo] = field(default_factory=list)
    total_missing: int = 0
    total_filled: int = 0
    rows_dropped: int = 0

    def summary(self) -> str:
        lines = [
            f"Missing data summary:",
            f"  Total missing values: {self.total_missing}",
            f"  Values filled: {self.total_filled}",
            f"  Rows dropped: {self.rows_dropped}",
        ]
        if self.columns:
            lines.append("\nBy column:")
            for col in self.columns:
                lines.append(f"  {col.column}: {col.missing_count} ({col.missing_percent:.1f}%) → {col.strategy_used.value}")
                if col.fill_value is not None:
                    lines.append(f"    Fill value: {col.fill_value}")
        return "\n".join(lines)


class MissingHandler:
    """
    Handle missing data with configurable strategies.

    Usage:
        handler = MissingHandler()
        df_clean, result = handler.handle(df)
        print(result.summary())
    """

    # Default strategy by column type
    DEFAULT_STRATEGIES = {
        "numeric": ImputationStrategy.MEDIAN,
        "categorical": ImputationStrategy.MODE,
        "datetime": ImputationStrategy.FORWARD_FILL,
    }

    def __init__(
        self,
        drop_threshold: float = 0.5,  # Drop columns with > 50% missing
        row_drop_threshold: float = 0.8,  # Drop rows with > 80% missing
    ):
        self.drop_threshold = drop_threshold
        self.row_drop_threshold = row_drop_threshold

    def analyze(self, df) -> dict[str, MissingInfo]:
        """Analyze missing data without modifying."""
        import pandas as pd

        analysis = {}
        total_rows = len(df)

        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                missing_pct = missing / total_rows * 100
                missing_rows = df[df[col].isna()].index.tolist()

                # Determine column type
                dtype = self._get_column_type(df[col])

                # Suggest strategy
                if missing_pct > self.drop_threshold * 100:
                    strategy = ImputationStrategy.DROP
                else:
                    strategy = self.DEFAULT_STRATEGIES.get(dtype, ImputationStrategy.MODE)

                analysis[col] = MissingInfo(
                    column=col,
                    missing_count=int(missing),
                    missing_percent=missing_pct,
                    strategy_used=strategy,
                    rows_affected=missing_rows[:100],  # Cap at 100 for memory
                )

        return analysis

    def handle(
        self,
        df,
        strategies: dict[str, ImputationStrategy] = None,
        constants: dict[str, any] = None,
    ) -> tuple:
        """
        Handle missing data.

        Args:
            df: pandas DataFrame
            strategies: Dict of column -> strategy overrides
            constants: Dict of column -> constant value for CONSTANT strategy

        Returns:
            (cleaned_df, MissingResult)
        """
        import pandas as pd

        df = df.copy()
        strategies = strategies or {}
        constants = constants or {}

        result = MissingResult()
        result.total_missing = df.isna().sum().sum()

        # First, drop rows that are mostly empty
        row_missing = df.isna().sum(axis=1) / len(df.columns)
        rows_to_drop = row_missing[row_missing > self.row_drop_threshold].index
        if len(rows_to_drop) > 0:
            df = df.drop(rows_to_drop)
            result.rows_dropped = len(rows_to_drop)

        # Analyze each column
        analysis = self.analyze(df)

        for col, info in analysis.items():
            # Use override strategy if provided
            strategy = strategies.get(col, info.strategy_used)

            # Apply strategy
            fill_value = self._apply_strategy(df, col, strategy, constants.get(col))

            info.strategy_used = strategy
            info.fill_value = fill_value
            result.columns.append(info)

        result.total_filled = result.total_missing - df.isna().sum().sum() - result.rows_dropped

        return df, result

    def _get_column_type(self, series) -> str:
        """Determine column type for strategy selection."""
        import pandas as pd

        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        else:
            return "categorical"

    def _apply_strategy(
        self,
        df,
        column: str,
        strategy: ImputationStrategy,
        constant: any = None,
    ) -> any:
        """Apply imputation strategy to column."""
        import pandas as pd

        if strategy == ImputationStrategy.DROP:
            df.drop(columns=[column], inplace=True)
            return None

        elif strategy == ImputationStrategy.MEAN:
            fill_value = df[column].mean()
            df[column] = df[column].fillna(fill_value)
            return float(fill_value) if pd.notna(fill_value) else None

        elif strategy == ImputationStrategy.MEDIAN:
            fill_value = df[column].median()
            df[column] = df[column].fillna(fill_value)
            return float(fill_value) if pd.notna(fill_value) else None

        elif strategy == ImputationStrategy.MODE:
            mode_values = df[column].mode()
            fill_value = mode_values.iloc[0] if len(mode_values) > 0 else None
            if fill_value is not None:
                df[column] = df[column].fillna(fill_value)
            return fill_value

        elif strategy == ImputationStrategy.CONSTANT:
            df[column] = df[column].fillna(constant)
            return constant

        elif strategy == ImputationStrategy.FORWARD_FILL:
            df[column] = df[column].ffill()
            return "forward fill"

        elif strategy == ImputationStrategy.BACKWARD_FILL:
            df[column] = df[column].bfill()
            return "backward fill"

        elif strategy == ImputationStrategy.KNN:
            return self._apply_knn_imputation(df, column)

        elif strategy == ImputationStrategy.FLAG_ONLY:
            return None  # Don't modify, just flag

        return None

    def _apply_knn_imputation(self, df, column: str) -> str:
        """Apply KNN imputation."""
        try:
            from sklearn.impute import KNNImputer
            import pandas as pd

            # Get numeric columns for KNN
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if column not in numeric_cols:
                # Fallback to mode for non-numeric
                mode_val = df[column].mode()
                if len(mode_val) > 0:
                    df[column] = df[column].fillna(mode_val.iloc[0])
                return "mode (KNN fallback)"

            if len(numeric_cols) < 2:
                # Not enough columns for KNN
                df[column] = df[column].fillna(df[column].median())
                return "median (KNN fallback)"

            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            return "KNN (k=5)"

        except ImportError:
            # Fallback to median
            df[column] = df[column].fillna(df[column].median())
            return "median (sklearn not available)"

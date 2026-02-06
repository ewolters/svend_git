"""
Automated Exploratory Data Analysis (EDA)

Run standard EDA on any dataset with one function call:
- Data summary and types
- Missing value analysis
- Distribution plots for all variables
- Correlation matrix
- Outlier detection summary

Usage:
    from analyst import quick_eda

    report = quick_eda(df)
    print(report.summary())
    report.save('eda_report/')
"""

import io
import base64
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    """Profile of a single column."""
    name: str
    dtype: str
    count: int
    missing: int
    missing_pct: float
    unique: int
    unique_pct: float

    # Numeric stats (None if not numeric)
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q25: Optional[float] = None
    median: Optional[float] = None
    q75: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None

    # Categorical stats (None if not categorical)
    top_values: Optional[list] = None  # [(value, count), ...]

    # Flags
    is_numeric: bool = False
    is_categorical: bool = False
    is_datetime: bool = False
    is_id_like: bool = False  # High cardinality, likely ID
    has_outliers: bool = False
    outlier_count: int = 0


@dataclass
class EDAReport:
    """Complete EDA report."""
    dataset_name: str
    n_rows: int
    n_cols: int
    memory_mb: float

    columns: list[ColumnProfile]

    # Overall stats
    total_missing: int
    total_missing_pct: float
    duplicate_rows: int
    duplicate_pct: float

    # Correlations (numeric columns only)
    correlation_matrix: Optional[pd.DataFrame] = None
    high_correlations: list = field(default_factory=list)  # [(col1, col2, corr), ...]

    # Charts (base64 encoded PNGs)
    charts: dict = field(default_factory=dict)  # name -> base64 string

    # Timestamp
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        """Get text summary of the report."""
        lines = [
            f"# EDA Report: {self.dataset_name}",
            f"Generated: {self.generated_at}",
            "",
            "## Dataset Overview",
            f"- **Rows:** {self.n_rows:,}",
            f"- **Columns:** {self.n_cols}",
            f"- **Memory:** {self.memory_mb:.2f} MB",
            f"- **Missing Values:** {self.total_missing:,} ({self.total_missing_pct:.1%})",
            f"- **Duplicate Rows:** {self.duplicate_rows:,} ({self.duplicate_pct:.1%})",
            "",
            "## Column Types",
        ]

        numeric_cols = [c for c in self.columns if c.is_numeric]
        cat_cols = [c for c in self.columns if c.is_categorical]
        dt_cols = [c for c in self.columns if c.is_datetime]
        id_cols = [c for c in self.columns if c.is_id_like]

        lines.append(f"- Numeric: {len(numeric_cols)}")
        lines.append(f"- Categorical: {len(cat_cols)}")
        lines.append(f"- Datetime: {len(dt_cols)}")
        lines.append(f"- ID-like (high cardinality): {len(id_cols)}")

        # Missing values
        cols_with_missing = [c for c in self.columns if c.missing > 0]
        if cols_with_missing:
            lines.extend(["", "## Missing Values"])
            for col in sorted(cols_with_missing, key=lambda x: x.missing_pct, reverse=True)[:10]:
                lines.append(f"- **{col.name}:** {col.missing:,} ({col.missing_pct:.1%})")

        # Outliers
        cols_with_outliers = [c for c in self.columns if c.has_outliers]
        if cols_with_outliers:
            lines.extend(["", "## Outliers Detected"])
            for col in cols_with_outliers:
                lines.append(f"- **{col.name}:** {col.outlier_count} outliers")

        # High correlations
        if self.high_correlations:
            lines.extend(["", "## High Correlations (|r| > 0.7)"])
            for col1, col2, corr in self.high_correlations[:10]:
                lines.append(f"- {col1} ↔ {col2}: {corr:.3f}")

        # Column details
        lines.extend(["", "## Column Details", ""])

        for col in self.columns:
            lines.append(f"### {col.name}")
            lines.append(f"- Type: {col.dtype}")
            lines.append(f"- Missing: {col.missing_pct:.1%}")
            lines.append(f"- Unique: {col.unique} ({col.unique_pct:.1%})")

            if col.is_numeric and col.mean is not None:
                lines.append(f"- Mean: {col.mean:.4g}, Std: {col.std:.4g}")
                lines.append(f"- Range: [{col.min:.4g}, {col.max:.4g}]")
                lines.append(f"- Quartiles: {col.q25:.4g} | {col.median:.4g} | {col.q75:.4g}")
                if col.skewness is not None:
                    skew_desc = "symmetric" if abs(col.skewness) < 0.5 else ("right-skewed" if col.skewness > 0 else "left-skewed")
                    lines.append(f"- Skewness: {col.skewness:.3f} ({skew_desc})")

            if col.is_categorical and col.top_values:
                lines.append("- Top values:")
                for val, count in col.top_values[:5]:
                    lines.append(f"  - {val}: {count:,}")

            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "memory_mb": self.memory_mb,
            "total_missing": self.total_missing,
            "total_missing_pct": self.total_missing_pct,
            "duplicate_rows": self.duplicate_rows,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "missing_pct": c.missing_pct,
                    "unique": c.unique,
                    "is_numeric": c.is_numeric,
                    "is_categorical": c.is_categorical,
                }
                for c in self.columns
            ],
            "high_correlations": self.high_correlations,
            "generated_at": self.generated_at,
        }

    def save(self, directory: str) -> dict:
        """Save report and charts to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        saved = {}

        # Save summary markdown
        summary_path = directory / "eda_report.md"
        summary_path.write_text(self.summary())
        saved["summary"] = str(summary_path)

        # Save JSON data
        import json
        json_path = directory / "eda_data.json"
        json_path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        saved["json"] = str(json_path)

        # Save charts
        for name, b64_data in self.charts.items():
            chart_path = directory / f"{name}.png"
            chart_path.write_bytes(base64.b64decode(b64_data))
            saved[name] = str(chart_path)

        # Save correlation matrix if exists
        if self.correlation_matrix is not None:
            corr_path = directory / "correlations.csv"
            self.correlation_matrix.to_csv(corr_path)
            saved["correlations"] = str(corr_path)

        return saved


class EDAAnalyzer:
    """
    Automated Exploratory Data Analysis.

    Usage:
        analyzer = EDAAnalyzer()
        report = analyzer.analyze(df, name="my_dataset")

        # Get summary
        print(report.summary())

        # Save everything
        report.save('eda_output/')
    """

    def __init__(self, generate_charts: bool = True, chart_style: str = "dark"):
        """
        Args:
            generate_charts: Whether to generate matplotlib charts
            chart_style: 'dark' for dark theme, 'light' for light theme
        """
        self.generate_charts = generate_charts
        self.chart_style = chart_style

    def analyze(self, df: pd.DataFrame, name: str = "dataset") -> EDAReport:
        """
        Run full EDA on a DataFrame.

        Args:
            df: pandas DataFrame to analyze
            name: Name for the dataset

        Returns:
            EDAReport with all analysis results
        """
        # Basic stats
        n_rows, n_cols = df.shape
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

        # Missing values
        total_missing = df.isna().sum().sum()
        total_cells = n_rows * n_cols
        total_missing_pct = total_missing / total_cells if total_cells > 0 else 0

        # Duplicates
        duplicate_rows = df.duplicated().sum()
        duplicate_pct = duplicate_rows / n_rows if n_rows > 0 else 0

        # Profile each column
        columns = []
        for col in df.columns:
            profile = self._profile_column(df[col], col, n_rows)
            columns.append(profile)

        # Correlation matrix (numeric columns only)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = None
        high_correlations = []

        if len(numeric_cols) >= 2:
            correlation_matrix = df[numeric_cols].corr()

            # Find high correlations
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr = correlation_matrix.loc[col1, col2]
                    if abs(corr) > 0.7 and not np.isnan(corr):
                        high_correlations.append((col1, col2, round(corr, 3)))

            high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        # Generate charts
        charts = {}
        if self.generate_charts:
            charts = self._generate_charts(df, columns, correlation_matrix)

        return EDAReport(
            dataset_name=name,
            n_rows=n_rows,
            n_cols=n_cols,
            memory_mb=memory_mb,
            columns=columns,
            total_missing=total_missing,
            total_missing_pct=total_missing_pct,
            duplicate_rows=duplicate_rows,
            duplicate_pct=duplicate_pct,
            correlation_matrix=correlation_matrix,
            high_correlations=high_correlations,
            charts=charts,
        )

    def _profile_column(self, series: pd.Series, name: str, n_rows: int) -> ColumnProfile:
        """Profile a single column."""
        dtype = str(series.dtype)
        count = series.count()
        missing = series.isna().sum()
        missing_pct = missing / n_rows if n_rows > 0 else 0
        unique = series.nunique()
        unique_pct = unique / n_rows if n_rows > 0 else 0

        profile = ColumnProfile(
            name=name,
            dtype=dtype,
            count=count,
            missing=missing,
            missing_pct=missing_pct,
            unique=unique,
            unique_pct=unique_pct,
        )

        # Determine column type
        if pd.api.types.is_bool_dtype(series):
            # Boolean columns - treat as categorical
            profile.is_categorical = True
            value_counts = series.value_counts().head(10)
            profile.top_values = [(str(v), int(c)) for v, c in value_counts.items()]

        elif pd.api.types.is_numeric_dtype(series):
            profile.is_numeric = True

            # Numeric stats
            clean = series.dropna()
            if len(clean) > 0:
                profile.mean = float(clean.mean())
                profile.std = float(clean.std())
                profile.min = float(clean.min())
                profile.q25 = float(clean.quantile(0.25))
                profile.median = float(clean.median())
                profile.q75 = float(clean.quantile(0.75))
                profile.max = float(clean.max())

                # Skewness
                if profile.std > 0:
                    profile.skewness = float(clean.skew())

                # Outlier detection (IQR method)
                iqr = profile.q75 - profile.q25
                lower = profile.q25 - 1.5 * iqr
                upper = profile.q75 + 1.5 * iqr
                outliers = ((clean < lower) | (clean > upper)).sum()
                if outliers > 0:
                    profile.has_outliers = True
                    profile.outlier_count = int(outliers)

        elif pd.api.types.is_datetime64_any_dtype(series):
            profile.is_datetime = True

        else:
            # Categorical
            profile.is_categorical = True

            # Top values
            value_counts = series.value_counts().head(10)
            profile.top_values = [(str(v), int(c)) for v, c in value_counts.items()]

        # Check if ID-like (high cardinality)
        if unique_pct > 0.9 and unique > 100:
            profile.is_id_like = True

        return profile

    def _generate_charts(self, df: pd.DataFrame, columns: list[ColumnProfile],
                         correlation_matrix: Optional[pd.DataFrame]) -> dict:
        """Generate visualization charts."""
        charts = {}

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            # Apply style
            if self.chart_style == "dark":
                plt.style.use('dark_background')
                bg_color = '#1a1a2e'
                text_color = '#e0e0e0'
                accent_color = '#4a9f6e'
            else:
                plt.style.use('default')
                bg_color = '#ffffff'
                text_color = '#333333'
                accent_color = '#2e7d32'

            # 1. Missing values chart
            missing_data = [(c.name, c.missing_pct) for c in columns if c.missing_pct > 0]
            if missing_data:
                charts["missing_values"] = self._chart_missing_values(
                    missing_data, bg_color, text_color, accent_color
                )

            # 2. Distribution charts for numeric columns
            numeric_cols = [c for c in columns if c.is_numeric and not c.is_id_like]
            for col in numeric_cols[:8]:  # Limit to 8
                series = df[col.name].dropna()
                if len(series) > 0:
                    chart_name = f"dist_{col.name.replace(' ', '_')[:20]}"
                    charts[chart_name] = self._chart_distribution(
                        series, col.name, bg_color, text_color, accent_color
                    )

            # 3. Categorical value counts
            cat_cols = [c for c in columns if c.is_categorical and not c.is_id_like]
            for col in cat_cols[:5]:  # Limit to 5
                if col.top_values:
                    chart_name = f"cat_{col.name.replace(' ', '_')[:20]}"
                    charts[chart_name] = self._chart_categorical(
                        col.top_values[:8], col.name, bg_color, text_color, accent_color
                    )

            # 4. Correlation heatmap
            if correlation_matrix is not None and len(correlation_matrix) >= 2:
                charts["correlation_matrix"] = self._chart_correlation(
                    correlation_matrix, bg_color, text_color
                )

            plt.close('all')

        except ImportError:
            pass  # No matplotlib, skip charts

        return charts

    def _chart_missing_values(self, data: list, bg_color: str, text_color: str,
                               accent_color: str) -> str:
        """Generate missing values bar chart."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.4)))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        names, pcts = zip(*sorted(data, key=lambda x: x[1], reverse=True))
        bars = ax.barh(names, [p * 100 for p in pcts], color=accent_color)

        ax.set_xlabel('Missing %', color=text_color)
        ax.set_title('Missing Values by Column', color=text_color, fontsize=12)
        ax.tick_params(colors=text_color)

        for spine in ax.spines.values():
            spine.set_color(text_color)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _chart_distribution(self, series: pd.Series, name: str, bg_color: str,
                            text_color: str, accent_color: str) -> str:
        """Generate distribution histogram."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        ax.hist(series, bins=30, color=accent_color, alpha=0.7, edgecolor=text_color, linewidth=0.5)
        ax.axvline(series.mean(), color='#ff6b6b', linestyle='--', label=f'Mean: {series.mean():.2f}')
        ax.axvline(series.median(), color='#4ecdc4', linestyle='--', label=f'Median: {series.median():.2f}')

        ax.set_xlabel(name, color=text_color)
        ax.set_ylabel('Frequency', color=text_color)
        ax.set_title(f'Distribution: {name}', color=text_color, fontsize=12)
        ax.tick_params(colors=text_color)
        ax.legend(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color)

        for spine in ax.spines.values():
            spine.set_color(text_color)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _chart_categorical(self, top_values: list, name: str, bg_color: str,
                           text_color: str, accent_color: str) -> str:
        """Generate categorical value counts bar chart."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, max(3, len(top_values) * 0.4)))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        labels, counts = zip(*top_values)
        # Truncate long labels
        labels = [str(l)[:20] + '...' if len(str(l)) > 20 else str(l) for l in labels]

        bars = ax.barh(labels, counts, color=accent_color)

        ax.set_xlabel('Count', color=text_color)
        ax.set_title(f'Top Values: {name}', color=text_color, fontsize=12)
        ax.tick_params(colors=text_color)

        for spine in ax.spines.values():
            spine.set_color(text_color)

        ax.invert_yaxis()  # Top value at top
        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _chart_correlation(self, corr_matrix: pd.DataFrame, bg_color: str,
                           text_color: str) -> str:
        """Generate correlation heatmap."""
        import matplotlib.pyplot as plt

        size = min(12, max(6, len(corr_matrix) * 0.6))
        fig, ax = plt.subplots(figsize=(size, size))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        im = ax.imshow(corr_matrix.values, cmap='RdYlGn', vmin=-1, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors=text_color)

        # Set ticks
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))

        # Truncate long column names
        labels = [str(c)[:12] for c in corr_matrix.columns]
        ax.set_xticklabels(labels, rotation=45, ha='right', color=text_color)
        ax.set_yticklabels(labels, color=text_color)

        ax.set_title('Correlation Matrix', color=text_color, fontsize=12)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        import matplotlib.pyplot as plt

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                   facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return b64


def quick_eda(df: pd.DataFrame, name: str = "dataset",
              generate_charts: bool = True) -> EDAReport:
    """
    Quick one-liner EDA.

    Args:
        df: pandas DataFrame
        name: Dataset name for the report
        generate_charts: Whether to generate visualization charts

    Returns:
        EDAReport with full analysis

    Example:
        from analyst import quick_eda

        report = quick_eda(df, "customer_data")
        print(report.summary())
        report.save('eda_output/')
    """
    analyzer = EDAAnalyzer(generate_charts=generate_charts)
    return analyzer.analyze(df, name)

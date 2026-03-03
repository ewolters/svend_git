"""
Chart Generation

Consistent, styled charts for documentation.
Built on matplotlib with a custom theme (Svend theme to be added later).

Chart types:
- Line plots
- Bar charts
- Scatter plots
- Heatmaps
- Box plots
- Histograms
"""

import io
import base64
from pathlib import Path
from typing import Any, Optional, Union
from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# Svend color palette (to be customized)
SVEND_COLORS = {
    "primary": "#2563eb",      # Blue
    "secondary": "#7c3aed",    # Purple
    "success": "#059669",      # Green
    "warning": "#d97706",      # Orange
    "error": "#dc2626",        # Red
    "neutral": "#6b7280",      # Gray
}

SVEND_PALETTE = [
    "#2563eb",  # Blue
    "#7c3aed",  # Purple
    "#059669",  # Green
    "#d97706",  # Orange
    "#dc2626",  # Red
    "#0891b2",  # Cyan
    "#4f46e5",  # Indigo
    "#84cc16",  # Lime
]


def apply_svend_style():
    """Apply Svend theme to matplotlib."""
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams.update({
        # Fonts
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,

        # Colors
        'axes.prop_cycle': plt.cycler(color=SVEND_PALETTE),
        'axes.facecolor': '#fafafa',
        'figure.facecolor': 'white',

        # Grid
        'grid.color': '#e5e7eb',
        'grid.linewidth': 0.5,

        # Axes
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#d1d5db',

        # Legend
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#e5e7eb',

        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    width: float = 10
    height: float = 6
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    legend: bool = True
    grid: bool = True
    style: str = "svend"  # svend, minimal, classic


class ChartGenerator:
    """
    Generate styled charts for documentation.

    All charts return either a file path or base64 string.
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("/tmp/svend_charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        apply_svend_style()

    def _save_or_encode(self, fig: Figure, output_path: Optional[Path] = None) -> str:
        """Save figure or return as base64."""
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return str(output_path)
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return f"data:image/png;base64,{b64}"

    def create_chart(self, data: Any, chart_type: str = "line",
                     output_path: Path = None, **kwargs) -> str:
        """
        Create a chart from data.

        Args:
            data: Dict with x/y keys, list of dicts, or numpy array
            chart_type: line, bar, scatter, heatmap, box, histogram
            output_path: Optional path to save
            **kwargs: Additional options (title, x_label, y_label, etc.)
        """
        method = getattr(self, f"_{chart_type}_chart", None)
        if method is None:
            raise ValueError(f"Unknown chart type: {chart_type}")

        return method(data, output_path, **kwargs)

    def _line_chart(self, data: dict, output_path: Path = None, **kwargs) -> str:
        """Generate line chart."""
        fig, ax = plt.subplots(figsize=(kwargs.get('width', 10), kwargs.get('height', 6)))

        if isinstance(data, dict):
            if 'x' in data and 'y' in data:
                # Single series
                ax.plot(data['x'], data['y'], marker='o', linewidth=2, markersize=4)
            elif 'series' in data:
                # Multiple series
                for series in data['series']:
                    ax.plot(series['x'], series['y'], marker='o', linewidth=2,
                           markersize=4, label=series.get('name', ''))
                if kwargs.get('legend', True):
                    ax.legend()

        ax.set_xlabel(kwargs.get('x_label', ''))
        ax.set_ylabel(kwargs.get('y_label', ''))
        ax.set_title(kwargs.get('title', ''))

        if kwargs.get('grid', True):
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return self._save_or_encode(fig, output_path)

    def _bar_chart(self, data: dict, output_path: Path = None, **kwargs) -> str:
        """Generate bar chart."""
        fig, ax = plt.subplots(figsize=(kwargs.get('width', 10), kwargs.get('height', 6)))

        if isinstance(data, dict):
            if 'categories' in data and 'values' in data:
                x = range(len(data['categories']))
                bars = ax.bar(x, data['values'], color=SVEND_COLORS['primary'])
                ax.set_xticks(x)
                ax.set_xticklabels(data['categories'], rotation=45, ha='right')

                # Add value labels on bars
                for bar, val in zip(bars, data['values']):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=9)

            elif 'groups' in data:
                # Grouped bar chart
                n_groups = len(data['groups'])
                n_bars = len(data['series'])
                width = 0.8 / n_bars

                for i, series in enumerate(data['series']):
                    x = np.arange(n_groups) + i * width
                    ax.bar(x, series['values'], width, label=series['name'])

                ax.set_xticks(np.arange(n_groups) + width * (n_bars - 1) / 2)
                ax.set_xticklabels(data['groups'])
                ax.legend()

        ax.set_xlabel(kwargs.get('x_label', ''))
        ax.set_ylabel(kwargs.get('y_label', ''))
        ax.set_title(kwargs.get('title', ''))

        fig.tight_layout()
        return self._save_or_encode(fig, output_path)

    def _scatter_chart(self, data: dict, output_path: Path = None, **kwargs) -> str:
        """Generate scatter plot."""
        fig, ax = plt.subplots(figsize=(kwargs.get('width', 10), kwargs.get('height', 6)))

        if isinstance(data, dict) and 'x' in data and 'y' in data:
            colors = data.get('colors', SVEND_COLORS['primary'])
            sizes = data.get('sizes', 50)
            ax.scatter(data['x'], data['y'], c=colors, s=sizes, alpha=0.7)

            # Add trend line if requested
            if kwargs.get('trendline', False):
                z = np.polyfit(data['x'], data['y'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(data['x']), max(data['x']), 100)
                ax.plot(x_line, p(x_line), '--', color=SVEND_COLORS['neutral'], alpha=0.8)

        ax.set_xlabel(kwargs.get('x_label', ''))
        ax.set_ylabel(kwargs.get('y_label', ''))
        ax.set_title(kwargs.get('title', ''))
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return self._save_or_encode(fig, output_path)

    def _heatmap_chart(self, data: Union[dict, np.ndarray], output_path: Path = None, **kwargs) -> str:
        """Generate heatmap."""
        fig, ax = plt.subplots(figsize=(kwargs.get('width', 10), kwargs.get('height', 8)))

        if isinstance(data, dict):
            matrix = np.array(data.get('matrix', data.get('values', [[]])))
            x_labels = data.get('x_labels', [])
            y_labels = data.get('y_labels', [])
        else:
            matrix = np.array(data)
            x_labels = []
            y_labels = []

        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto')

        if x_labels:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(kwargs.get('colorbar_label', ''))

        # Add text annotations if matrix is small
        if matrix.shape[0] <= 10 and matrix.shape[1] <= 10:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    color = 'white' if abs(val) > np.max(np.abs(matrix)) * 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           color=color, fontsize=9)

        ax.set_xlabel(kwargs.get('x_label', ''))
        ax.set_ylabel(kwargs.get('y_label', ''))
        ax.set_title(kwargs.get('title', ''))

        fig.tight_layout()
        return self._save_or_encode(fig, output_path)

    def _box_chart(self, data: dict, output_path: Path = None, **kwargs) -> str:
        """Generate box plot."""
        fig, ax = plt.subplots(figsize=(kwargs.get('width', 10), kwargs.get('height', 6)))

        if isinstance(data, dict) and 'groups' in data:
            box_data = [data['groups'][k] for k in data['groups']]
            labels = list(data['groups'].keys())

            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)

            # Style boxes
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(SVEND_PALETTE[i % len(SVEND_PALETTE)])
                patch.set_alpha(0.7)

        ax.set_xlabel(kwargs.get('x_label', ''))
        ax.set_ylabel(kwargs.get('y_label', ''))
        ax.set_title(kwargs.get('title', ''))
        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        return self._save_or_encode(fig, output_path)

    def _histogram_chart(self, data: Union[dict, list], output_path: Path = None, **kwargs) -> str:
        """Generate histogram."""
        fig, ax = plt.subplots(figsize=(kwargs.get('width', 10), kwargs.get('height', 6)))

        if isinstance(data, dict):
            values = data.get('values', data.get('data', []))
        else:
            values = data

        bins = kwargs.get('bins', 'auto')
        ax.hist(values, bins=bins, color=SVEND_COLORS['primary'],
               edgecolor='white', alpha=0.8)

        ax.set_xlabel(kwargs.get('x_label', ''))
        ax.set_ylabel(kwargs.get('y_label', 'Frequency'))
        ax.set_title(kwargs.get('title', ''))
        ax.grid(True, alpha=0.3, axis='y')

        fig.tight_layout()
        return self._save_or_encode(fig, output_path)

    def multi_panel(self, charts: list[dict], cols: int = 2,
                    output_path: Path = None, **kwargs) -> str:
        """
        Create multi-panel figure.

        Args:
            charts: List of chart configs, each with 'type' and 'data'
            cols: Number of columns
        """
        n = len(charts)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(kwargs.get('width', 5*cols),
                                         kwargs.get('height', 4*rows)))

        if n == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for i, chart in enumerate(charts):
            row, col = i // cols, i % cols
            ax = axes[row, col]

            # Create subplot based on chart type
            chart_type = chart.get('type', 'line')
            data = chart.get('data', {})

            if chart_type == 'line' and 'x' in data and 'y' in data:
                ax.plot(data['x'], data['y'], marker='o', linewidth=2, markersize=3)
            elif chart_type == 'bar' and 'categories' in data:
                ax.bar(range(len(data['categories'])), data['values'])
                ax.set_xticks(range(len(data['categories'])))
                ax.set_xticklabels(data['categories'], rotation=45, ha='right', fontsize=8)
            elif chart_type == 'scatter' and 'x' in data:
                ax.scatter(data['x'], data['y'], alpha=0.7)

            ax.set_title(chart.get('title', ''), fontsize=10)
            ax.set_xlabel(chart.get('x_label', ''), fontsize=9)
            ax.set_ylabel(chart.get('y_label', ''), fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(n, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)

        fig.suptitle(kwargs.get('title', ''), fontsize=14, y=1.02)
        fig.tight_layout()
        return self._save_or_encode(fig, output_path)


def quick_chart(data: dict, chart_type: str = "line", **kwargs) -> str:
    """Quick helper to create a chart."""
    generator = ChartGenerator()
    return generator.create_chart(data, chart_type, **kwargs)

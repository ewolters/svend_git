"""
Visualization for Experimental Design

Generate publication-ready plots:
- Power curves
- Design matrices (heatmaps)
- Interaction plots
- Sample size curves
"""

import io
import base64
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import special
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .stats import PowerAnalyzer, PowerResult
from .doe import ExperimentDesign, Factor


def _fig_to_base64(fig: Figure) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _save_or_return(fig: Figure, output_path: Optional[Path] = None) -> str:
    """Save figure to file or return as base64."""
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return str(output_path)
    else:
        b64 = _fig_to_base64(fig)
        plt.close(fig)
        return f"data:image/png;base64,{b64}"


def plot_power_curve(test_type: str = "ttest_ind",
                     effect_sizes: list[float] = None,
                     sample_sizes: list[int] = None,
                     alpha: float = 0.05,
                     output_path: Path = None,
                     **kwargs) -> str:
    """
    Plot power as a function of effect size or sample size.

    Args:
        test_type: Type of test (ttest_ind, anova, correlation)
        effect_sizes: List of effect sizes to plot
        sample_sizes: List of sample sizes (if varying n instead of effect)
        alpha: Significance level
        output_path: Optional path to save figure
        **kwargs: Additional arguments for specific tests (e.g., groups for ANOVA)

    Returns:
        Path to saved file or base64-encoded image
    """
    analyzer = PowerAnalyzer()

    fig, ax = plt.subplots(figsize=(10, 6))

    if sample_sizes:
        # Plot power vs sample size for fixed effect size
        effect_size = kwargs.get('effect_size', 0.5)
        powers = []

        for n in sample_sizes:
            if test_type == "ttest_ind":
                n_per_group = n // 2
                z_alpha = 1.96  # Two-tailed
                se = np.sqrt(2 / n_per_group)
                z_power = effect_size / se - z_alpha
                power = float(np.clip(0.5 * (1 + special.erf(z_power / np.sqrt(2))), 0, 1))
            elif test_type == "anova":
                groups = kwargs.get('groups', 3)
                n_per_group = n // groups
                # Simplified power calculation
                lambda_nc = (effect_size ** 2) * n
                power = min(0.99, lambda_nc / (lambda_nc + 10))
            else:
                power = 0.8
            powers.append(power)

        ax.plot(sample_sizes, powers, 'b-', linewidth=2, marker='o', markersize=4)
        ax.axhline(y=0.8, color='r', linestyle='--', label='Power = 0.80')
        ax.axhline(y=0.9, color='orange', linestyle='--', label='Power = 0.90')

        ax.set_xlabel('Total Sample Size', fontsize=12)
        ax.set_ylabel('Statistical Power', fontsize=12)
        ax.set_title(f'Power Curve: {test_type}\n(Effect size = {effect_size})', fontsize=14)

        # Find required n for 80% power
        for i, power in enumerate(powers):
            if power >= 0.8:
                ax.axvline(x=sample_sizes[i], color='gray', linestyle=':',
                          label=f'n={sample_sizes[i]} for 80% power')
                break

    else:
        # Plot power vs effect size for fixed sample size
        if effect_sizes is None:
            effect_sizes = np.linspace(0.1, 1.5, 50).tolist()

        sample_size = kwargs.get('sample_size', 100)
        powers = []

        for es in effect_sizes:
            if test_type == "ttest_ind":
                n_per_group = sample_size // 2
                z_alpha = 1.96
                se = np.sqrt(2 / n_per_group)
                z_power = es / se - z_alpha
                power = float(np.clip(0.5 * (1 + special.erf(z_power / np.sqrt(2))), 0, 1))
            elif test_type == "anova":
                groups = kwargs.get('groups', 3)
                lambda_nc = (es ** 2) * sample_size
                power = min(0.99, lambda_nc / (lambda_nc + 10))
            else:
                power = 0.8
            powers.append(power)

        ax.plot(effect_sizes, powers, 'b-', linewidth=2)
        ax.axhline(y=0.8, color='r', linestyle='--', label='Power = 0.80')

        # Mark Cohen's conventions
        conventions = {'small': 0.2, 'medium': 0.5, 'large': 0.8}
        for name, es in conventions.items():
            if min(effect_sizes) <= es <= max(effect_sizes):
                ax.axvline(x=es, color='gray', linestyle=':', alpha=0.5)
                ax.text(es, 0.05, f'{name}\n({es})', ha='center', fontsize=9)

        ax.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
        ax.set_ylabel('Statistical Power', fontsize=12)
        ax.set_title(f'Power Curve: {test_type}\n(n = {sample_size})', fontsize=14)

    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_sample_size_curve(test_type: str = "ttest_ind",
                           effect_sizes: list[float] = None,
                           power_levels: list[float] = None,
                           alpha: float = 0.05,
                           output_path: Path = None) -> str:
    """
    Plot required sample size as a function of effect size.

    Args:
        test_type: Type of test
        effect_sizes: Effect sizes to evaluate
        power_levels: Power levels to show (default: [0.80, 0.90])
        alpha: Significance level
        output_path: Optional path to save figure
    """
    if effect_sizes is None:
        effect_sizes = np.linspace(0.2, 1.2, 50).tolist()
    if power_levels is None:
        power_levels = [0.80, 0.90]

    analyzer = PowerAnalyzer()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'orange', 'green']

    for i, power in enumerate(power_levels):
        sample_sizes = []
        for es in effect_sizes:
            if test_type == "ttest_ind":
                result = analyzer.power_ttest_ind(es, alpha, power)
            elif test_type == "paired":
                result = analyzer.power_ttest_paired(es, alpha, power)
            elif test_type == "correlation":
                result = analyzer.power_correlation(es, alpha, power)
            else:
                result = analyzer.power_ttest_ind(es, alpha, power)
            sample_sizes.append(result.sample_size)

        ax.plot(effect_sizes, sample_sizes, color=colors[i % len(colors)],
                linewidth=2, label=f'Power = {power:.0%}')

    ax.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
    ax.set_ylabel('Required Sample Size', fontsize=12)
    ax.set_title(f'Sample Size Requirements: {test_type}\n(α = {alpha})', fontsize=14)

    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_design_matrix(design: ExperimentDesign,
                       coded: bool = True,
                       output_path: Path = None) -> str:
    """
    Plot design matrix as a heatmap.

    Args:
        design: Experimental design to visualize
        coded: Use coded levels (-1, 0, +1) vs actual values
        output_path: Optional path to save figure
    """
    matrix = np.array(design.to_matrix(coded=coded))
    n_runs, n_factors = matrix.shape

    fig, ax = plt.subplots(figsize=(max(8, n_factors * 1.5), max(6, n_runs * 0.3)))

    # Create heatmap
    if coded:
        cmap = 'RdBu'
        vmin, vmax = -1.5, 1.5
    else:
        cmap = 'viridis'
        vmin, vmax = None, None

    im = ax.imshow(matrix.astype(float), cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    # Labels
    ax.set_xticks(range(n_factors))
    ax.set_xticklabels([f.name for f in design.factors], rotation=45, ha='right')
    ax.set_yticks(range(n_runs))

    # Use run order for labels
    run_labels = []
    for run in sorted(design.runs, key=lambda r: r.run_order):
        label = str(run.run_order)
        if run.is_center_point:
            label += "*"
        run_labels.append(label)
    ax.set_yticklabels(run_labels)

    ax.set_xlabel('Factors', fontsize=12)
    ax.set_ylabel('Run Order', fontsize=12)
    ax.set_title(f'{design.name}\n{design.design_type}', fontsize=14)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    if coded:
        cbar.set_label('Coded Level')
        cbar.set_ticks([-1, 0, 1])
    else:
        cbar.set_label('Factor Level')

    # Add text annotations for coded values
    if coded and n_runs <= 32:
        for i in range(n_runs):
            for j in range(n_factors):
                val = matrix[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:+.0f}' if val != 0 else '0',
                       ha='center', va='center', color=color, fontsize=9)

    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_factor_effects(design: ExperimentDesign,
                        response_name: str = "Response",
                        responses: list[float] = None,
                        output_path: Path = None) -> str:
    """
    Plot main effects (requires response data).

    If no responses provided, shows the design structure only.
    """
    n_factors = len(design.factors)

    fig, axes = plt.subplots(1, n_factors, figsize=(4 * n_factors, 4))
    if n_factors == 1:
        axes = [axes]

    for i, (ax, factor) in enumerate(zip(axes, design.factors)):
        levels = factor.levels
        level_names = factor.level_names

        if responses:
            # Calculate mean response at each level
            level_means = []
            for level in levels:
                mask = [r.factor_levels[factor.name] == level for r in design.runs]
                level_responses = [responses[j] for j, m in enumerate(mask) if m]
                level_means.append(np.mean(level_responses) if level_responses else 0)

            ax.plot(range(len(levels)), level_means, 'bo-', linewidth=2, markersize=8)
            ax.set_ylabel(response_name, fontsize=11)
        else:
            # Just show levels
            ax.bar(range(len(levels)), [1] * len(levels), alpha=0.3)
            ax.set_ylabel('Design Points', fontsize=11)

        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(level_names)
        ax.set_xlabel(factor.name, fontsize=11)
        ax.set_title(f'Effect of {factor.name}', fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Main Effects Plot: {design.name}', fontsize=14, y=1.02)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_interaction(design: ExperimentDesign,
                     factor1: str,
                     factor2: str,
                     responses: list[float],
                     response_name: str = "Response",
                     output_path: Path = None) -> str:
    """
    Plot two-factor interaction.

    Args:
        design: Experimental design
        factor1: Name of first factor (x-axis)
        factor2: Name of second factor (lines)
        responses: Response values for each run
        response_name: Label for y-axis
        output_path: Optional path to save
    """
    f1 = next(f for f in design.factors if f.name == factor1)
    f2 = next(f for f in design.factors if f.name == factor2)

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(f2.levels)))

    for j, level2 in enumerate(f2.levels):
        means = []
        for level1 in f1.levels:
            mask = [r.factor_levels[factor1] == level1 and
                   r.factor_levels[factor2] == level2
                   for r in design.runs]
            level_responses = [responses[i] for i, m in enumerate(mask) if m]
            means.append(np.mean(level_responses) if level_responses else np.nan)

        ax.plot(range(len(f1.levels)), means, 'o-',
                color=colors[j], linewidth=2, markersize=8,
                label=f'{factor2}={f2.level_names[j]}')

    ax.set_xticks(range(len(f1.levels)))
    ax.set_xticklabels(f1.level_names)
    ax.set_xlabel(factor1, fontsize=12)
    ax.set_ylabel(response_name, fontsize=12)
    ax.set_title(f'Interaction: {factor1} × {factor2}', fontsize=14)
    ax.legend(title=factor2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save_or_return(fig, output_path)


def plot_residuals(predicted: list[float],
                   actual: list[float],
                   output_path: Path = None) -> str:
    """
    Plot residual diagnostics (4-panel).

    Args:
        predicted: Predicted/fitted values
        actual: Actual response values
        output_path: Optional path to save
    """
    predicted = np.array(predicted)
    actual = np.array(actual)
    residuals = actual - predicted

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(predicted, residuals, alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted')

    # 2. Normal Q-Q
    ax = axes[0, 1]
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, alpha=0.7)
    ax.plot(osm, slope * np.array(osm) + intercept, 'r--')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title('Normal Q-Q Plot')

    # 3. Scale-Location
    ax = axes[1, 0]
    sqrt_abs_resid = np.sqrt(np.abs(residuals))
    ax.scatter(predicted, sqrt_abs_resid, alpha=0.7)
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('√|Residuals|')
    ax.set_title('Scale-Location')

    # 4. Histogram of residuals
    ax = axes[1, 1]
    ax.hist(residuals, bins='auto', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    ax.set_title('Residual Distribution')

    fig.suptitle('Residual Diagnostics', fontsize=14, y=1.02)
    fig.tight_layout()
    return _save_or_return(fig, output_path)

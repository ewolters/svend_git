"""
Experimenter Agent

Experimental design service with deterministic calculations:
- Power analysis and sample size
- Design of Experiments (DOE)
- Visualization

Usage:
    from experimenter import ExperimenterAgent, quick_power, quick_factorial

    # Power analysis
    result = quick_power(effect_size=0.5, test="ttest")
    print(result.summary())

    # Factorial design
    design = quick_factorial({
        "Temperature": [100, 150],
        "Pressure": [1, 2],
    })
    print(design.to_markdown())
"""

from .stats import (
    PowerAnalyzer,
    PowerResult,
    SampleSizeResult,
    sample_size_for_mean,
    sample_size_for_proportion,
    effect_size_from_means,
    interpret_effect_size,
)

from .doe import (
    DOEGenerator,
    ExperimentDesign,
    Factor,
    ExperimentRun,
)

from .plots import (
    plot_power_curve,
    plot_sample_size_curve,
    plot_design_matrix,
    plot_factor_effects,
    plot_interaction,
    plot_residuals,
)

from .agent import (
    ExperimenterAgent,
    ExperimentRequest,
    ExperimentResult,
    quick_power,
    quick_factorial,
)

__all__ = [
    # Stats
    "PowerAnalyzer",
    "PowerResult",
    "SampleSizeResult",
    "sample_size_for_mean",
    "sample_size_for_proportion",
    "effect_size_from_means",
    "interpret_effect_size",
    # DOE
    "DOEGenerator",
    "ExperimentDesign",
    "Factor",
    "ExperimentRun",
    # Plots
    "plot_power_curve",
    "plot_sample_size_curve",
    "plot_design_matrix",
    "plot_factor_effects",
    "plot_interaction",
    "plot_residuals",
    # Agent
    "ExperimenterAgent",
    "ExperimentRequest",
    "ExperimentResult",
    "quick_power",
    "quick_factorial",
]

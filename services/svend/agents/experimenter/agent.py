"""
Experimenter Agent

Experimental design service with:
- Power analysis and sample size calculations
- DOE generation (factorial, fractional, CCD, etc.)
- Visualization (power curves, design matrices)
- Structured protocol output

The LLM translates user intent; the tools do the math.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .stats import (
    PowerAnalyzer, PowerResult, SampleSizeResult,
    sample_size_for_mean, sample_size_for_proportion,
    effect_size_from_means, interpret_effect_size
)
from .doe import (
    DOEGenerator, ExperimentDesign, Factor, ExperimentRun
)
from .plots import (
    plot_power_curve, plot_sample_size_curve,
    plot_design_matrix, plot_factor_effects
)


@dataclass
class ExperimentRequest:
    """User's experiment design request."""
    goal: str  # What they want to learn
    request_type: Literal["power", "sample_size", "design", "full"]

    # For power/sample size
    test_type: str = "ttest_ind"
    effect_size: float = None
    alpha: float = 0.05
    power: float = 0.80
    groups: int = 2

    # For DOE
    factors: list[dict] = field(default_factory=list)  # {name, levels, units}
    design_type: str = "full_factorial"
    response_name: str = "Response"

    # Options
    include_plots: bool = True
    seed: int = None  # For reproducibility


@dataclass
class ExperimentResult:
    """Complete experiment design output."""
    request: ExperimentRequest

    # Power analysis
    power_result: PowerResult = None
    sample_size_result: SampleSizeResult = None

    # DOE
    design: ExperimentDesign = None

    # Plots (as base64 or file paths)
    plots: dict[str, str] = field(default_factory=dict)

    # Interpretation (from LLM or template)
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        result = {
            "goal": self.request.goal,
            "request_type": self.request.request_type,
        }

        if self.power_result:
            result["power_analysis"] = self.power_result.to_dict()

        if self.sample_size_result:
            result["sample_size"] = {
                "test_type": self.sample_size_result.test_type,
                "required_n": self.sample_size_result.sample_size,
                "parameters": self.sample_size_result.parameters,
            }

        if self.design:
            result["design"] = self.design.to_dict()

        result["summary"] = self.summary
        result["recommendations"] = self.recommendations
        result["warnings"] = self.warnings
        result["plots"] = list(self.plots.keys())

        return result

    def to_markdown(self) -> str:
        """Generate full markdown report."""
        lines = [
            "# Experiment Design Report",
            "",
            f"**Goal:** {self.request.goal}",
            "",
        ]

        # Summary
        if self.summary:
            lines.extend([
                "## Summary",
                "",
                self.summary,
                "",
            ])

        # Power Analysis
        if self.power_result:
            lines.extend([
                "## Power Analysis",
                "",
                f"- **Test Type:** {self.power_result.test_type}",
                f"- **Effect Size:** {self.power_result.effect_size} ({interpret_effect_size(self.power_result.effect_size)})",
                f"- **Significance Level (α):** {self.power_result.alpha}",
                f"- **Statistical Power:** {self.power_result.power:.1%}",
                f"- **Required Sample Size:** {self.power_result.sample_size}",
            ])
            if self.power_result.sample_size_per_group:
                lines.append(f"- **Per Group:** {self.power_result.sample_size_per_group}")
            lines.append("")

        # Sample Size
        if self.sample_size_result:
            lines.extend([
                "## Sample Size Calculation",
                "",
                self.sample_size_result.summary(),
                "",
            ])

        # Design
        if self.design:
            lines.extend([
                "## Experimental Design",
                "",
                self.design.to_markdown(),
                "",
            ])

        # Recommendations
        if self.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Warnings
        if self.warnings:
            lines.extend([
                "## Warnings",
                "",
            ])
            for warn in self.warnings:
                lines.append(f"- ⚠️ {warn}")
            lines.append("")

        # Plots
        if self.plots:
            lines.extend([
                "## Visualizations",
                "",
            ])
            for name, plot_data in self.plots.items():
                if plot_data.startswith("data:image"):
                    lines.append(f"### {name}")
                    lines.append(f"![{name}]({plot_data})")
                else:
                    lines.append(f"- {name}: {plot_data}")
                lines.append("")

        return "\n".join(lines)


class ExperimenterAgent:
    """
    Experimental design agent.

    Deterministic tools do all calculations.
    LLM (optional) helps interpret requests and explain results.
    """

    def __init__(self, llm=None, seed: int = None):
        self.llm = llm
        self.power_analyzer = PowerAnalyzer()
        self.doe_generator = DOEGenerator(seed=seed)

    def design_experiment(self, request: ExperimentRequest) -> ExperimentResult:
        """
        Process an experiment design request.

        Args:
            request: Experiment request with goals and parameters

        Returns:
            Complete experiment result with design, analysis, and plots
        """
        result = ExperimentResult(request=request)

        # Power Analysis
        if request.request_type in ["power", "full"]:
            result.power_result = self._run_power_analysis(request)

            if request.include_plots:
                # Power curve
                plot = plot_power_curve(
                    test_type=request.test_type,
                    sample_sizes=list(range(10, 500, 10)),
                    effect_size=request.effect_size or 0.5,
                    alpha=request.alpha,
                )
                result.plots["Power Curve"] = plot

        # Sample Size
        if request.request_type in ["sample_size", "full"]:
            if request.effect_size:
                result.power_result = self._run_power_analysis(request)

        # Design of Experiments
        if request.request_type in ["design", "full"] and request.factors:
            result.design = self._generate_design(request)

            if request.include_plots:
                plot = plot_design_matrix(result.design, coded=True)
                result.plots["Design Matrix"] = plot

        # Generate summary and recommendations
        self._generate_interpretation(result)

        return result

    def _run_power_analysis(self, request: ExperimentRequest) -> PowerResult:
        """Run appropriate power analysis."""
        effect_size = request.effect_size or 0.5

        if request.test_type == "ttest_ind":
            return self.power_analyzer.power_ttest_ind(
                effect_size=effect_size,
                alpha=request.alpha,
                power=request.power,
            )
        elif request.test_type == "ttest_paired":
            return self.power_analyzer.power_ttest_paired(
                effect_size=effect_size,
                alpha=request.alpha,
                power=request.power,
            )
        elif request.test_type == "anova":
            return self.power_analyzer.power_anova(
                effect_size=effect_size,
                groups=request.groups,
                alpha=request.alpha,
                power=request.power,
            )
        elif request.test_type == "correlation":
            return self.power_analyzer.power_correlation(
                r=effect_size,
                alpha=request.alpha,
                power=request.power,
            )
        elif request.test_type == "chi_square":
            return self.power_analyzer.power_chi_square(
                effect_size=effect_size,
                df=(request.groups - 1),
                alpha=request.alpha,
                power=request.power,
            )
        else:
            # Default to t-test
            return self.power_analyzer.power_ttest_ind(
                effect_size=effect_size,
                alpha=request.alpha,
                power=request.power,
            )

    def _generate_design(self, request: ExperimentRequest) -> ExperimentDesign:
        """Generate appropriate experimental design."""
        # Convert factor dicts to Factor objects
        factors = []
        for f_dict in request.factors:
            factors.append(Factor(
                name=f_dict["name"],
                levels=f_dict["levels"],
                units=f_dict.get("units", ""),
                is_categorical=f_dict.get("categorical", False),
            ))

        design_type = request.design_type.lower()

        if design_type == "full_factorial":
            return self.doe_generator.full_factorial(factors)

        elif design_type == "fractional_factorial":
            resolution = 4  # Default to resolution IV
            return self.doe_generator.fractional_factorial(factors, resolution=resolution)

        elif design_type == "ccd" or design_type == "response_surface":
            return self.doe_generator.central_composite(factors)

        elif design_type == "latin_square":
            if len(factors) == 1:
                return self.doe_generator.latin_square(factors[0].levels)
            else:
                return self.doe_generator.full_factorial(factors)

        elif design_type == "rcbd" or design_type == "randomized_block":
            treatments = factors[0].levels if factors else ["A", "B", "C"]
            blocks = len(factors[1].levels) if len(factors) > 1 else 4
            return self.doe_generator.randomized_block(treatments, blocks)

        else:
            # Default to full factorial
            return self.doe_generator.full_factorial(factors)

    def _generate_interpretation(self, result: ExperimentResult):
        """Generate summary and recommendations (template-based or LLM)."""
        request = result.request
        recommendations = []
        warnings = []

        # Power analysis interpretation
        if result.power_result:
            pr = result.power_result
            effect_interp = interpret_effect_size(pr.effect_size)

            result.summary = (
                f"To detect a {effect_interp} effect (d={pr.effect_size}) with "
                f"{pr.power:.0%} power at α={pr.alpha}, you need **{pr.sample_size} total participants**"
            )
            if pr.sample_size_per_group:
                result.summary += f" ({pr.sample_size_per_group} per group)"
            result.summary += "."

            # Recommendations
            if pr.sample_size > 500:
                recommendations.append(
                    "Large sample required. Consider whether a smaller effect size is "
                    "scientifically meaningful, or if a sequential design could reduce sample needs."
                )

            if pr.effect_size < 0.3:
                warnings.append(
                    f"Small effect size ({pr.effect_size}). Ensure this is realistic "
                    "based on prior research or pilot data."
                )

            if pr.power < 0.8:
                warnings.append(
                    f"Power ({pr.power:.0%}) is below the conventional 80% threshold. "
                    "Consider increasing sample size."
                )

        # DOE interpretation
        if result.design:
            design = result.design

            # Calculate total possible combinations
            total_combinations = 1
            for f in design.factors:
                total_combinations *= len(f.levels)

            if not result.summary:
                result.summary = (
                    f"Generated a {design.design_type} with {design.num_runs} runs "
                    f"across {len(design.factors)} factors."
                )

            # Explain efficiency if runs < possible combinations
            if design.num_runs < total_combinations:
                result.summary += (
                    f"\n\n**Why {design.num_runs} runs instead of {total_combinations}?** "
                    f"Factorial designs use statistical principles to efficiently estimate "
                    f"main effects and interactions without testing every possible combination. "
                    f"This design can detect which factors matter most with {design.num_runs} "
                    f"strategically chosen runs."
                )
            elif design.num_runs == total_combinations:
                result.summary += (
                    f" This tests all {total_combinations} factor combinations."
                )

            # Design-specific recommendations
            if design.resolution and design.resolution < 4:
                warnings.append(
                    "Resolution III design: main effects are confounded with 2-factor "
                    "interactions. Results may be ambiguous if interactions exist."
                )

            if design.num_runs > 64:
                recommendations.append(
                    "Consider a fractional factorial or screening design to reduce "
                    f"runs from {design.num_runs}."
                )

            if design.num_center_points > 0:
                recommendations.append(
                    "Center points included. These allow detection of curvature "
                    "(non-linear effects) in the response."
                )

            # Always recommend randomization
            recommendations.append(
                "Run experiments in the randomized order shown to minimize bias "
                "from time-related factors."
            )

            # If 2-level design, explain options for more levels
            all_two_level = all(len(f.levels) == 2 for f in design.factors)
            if all_two_level and len(design.factors) >= 2:
                recommendations.append(
                    "This 2-level design tests HIGH/LOW extremes. If your process has "
                    "many intermediate settings (e.g., 48 machine configurations), first use "
                    "this screening design to identify the critical factors, then follow up "
                    "with a response surface design on those factors to optimize settings."
                )

        result.recommendations = recommendations
        result.warnings = warnings


# Convenience functions for common tasks

def quick_power(effect_size: float, test: str = "ttest",
                alpha: float = 0.05, power: float = 0.80) -> PowerResult:
    """Quick power analysis."""
    agent = ExperimenterAgent()
    request = ExperimentRequest(
        goal="Power analysis",
        request_type="power",
        test_type=test,
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        include_plots=False,
    )
    result = agent.design_experiment(request)
    return result.power_result


def quick_factorial(factors: dict[str, list], seed: int = None) -> ExperimentDesign:
    """
    Quick factorial design.

    Args:
        factors: Dict of factor_name -> [level1, level2, ...]
        seed: Random seed for reproducibility

    Example:
        design = quick_factorial({
            "Temperature": [100, 150],
            "Pressure": [1, 2, 3],
            "Catalyst": ["A", "B"],
        })
    """
    agent = ExperimenterAgent(seed=seed)
    factor_list = [
        {"name": name, "levels": levels}
        for name, levels in factors.items()
    ]
    request = ExperimentRequest(
        goal="Factorial experiment",
        request_type="design",
        factors=factor_list,
        design_type="full_factorial",
        include_plots=False,
    )
    result = agent.design_experiment(request)
    return result.design

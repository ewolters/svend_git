"""
Statistical Calculations for Experimental Design

Deterministic tools - no LLM needed:
- Power analysis
- Sample size calculations
- Effect size conversions
- Confidence intervals
"""

import math
from dataclasses import dataclass
from typing import Literal
from scipy import stats
import numpy as np


@dataclass
class PowerResult:
    """Result of power analysis."""
    test_type: str
    effect_size: float
    alpha: float
    power: float
    sample_size: int
    sample_size_per_group: int = None  # For multi-group tests

    def to_dict(self) -> dict:
        result = {
            "test_type": self.test_type,
            "effect_size": self.effect_size,
            "alpha": self.alpha,
            "power": self.power,
            "total_sample_size": self.sample_size,
        }
        if self.sample_size_per_group:
            result["sample_size_per_group"] = self.sample_size_per_group
        return result

    def summary(self) -> str:
        lines = [
            f"Power Analysis: {self.test_type}",
            f"  Effect size (Cohen's d): {self.effect_size}",
            f"  Significance level (α): {self.alpha}",
            f"  Statistical power: {self.power:.1%}",
            f"  Required sample size: {self.sample_size}",
        ]
        if self.sample_size_per_group:
            lines.append(f"  Per group: {self.sample_size_per_group}")
        return "\n".join(lines)


@dataclass
class SampleSizeResult:
    """Result of sample size calculation."""
    test_type: str
    parameters: dict
    sample_size: int
    sample_size_per_group: int = None
    margin_of_error: float = None
    confidence_level: float = None

    def summary(self) -> str:
        lines = [f"Sample Size Calculation: {self.test_type}"]
        for k, v in self.parameters.items():
            lines.append(f"  {k}: {v}")
        lines.append(f"  Required sample size: {self.sample_size}")
        if self.sample_size_per_group:
            lines.append(f"  Per group: {self.sample_size_per_group}")
        return "\n".join(lines)


class PowerAnalyzer:
    """
    Power analysis and sample size calculations.

    All calculations are deterministic - no LLM involved.
    """

    # Cohen's conventions for effect sizes
    EFFECT_SIZE_CONVENTIONS = {
        "small": 0.2,
        "medium": 0.5,
        "large": 0.8,
    }

    def power_ttest_ind(self, effect_size: float, alpha: float = 0.05,
                        power: float = 0.8, ratio: float = 1.0) -> PowerResult:
        """
        Power analysis for independent samples t-test.

        Args:
            effect_size: Cohen's d (0.2=small, 0.5=medium, 0.8=large)
            alpha: Significance level (default 0.05)
            power: Desired power (default 0.80)
            ratio: Ratio of group sizes n2/n1 (default 1.0 = equal)

        Returns:
            PowerResult with required sample sizes
        """
        # Using the formula: n = 2 * ((z_alpha + z_beta) / d)^2
        # For unequal groups, adjust by ratio

        z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
        z_beta = stats.norm.ppf(power)

        # Sample size per group for group 1
        n1 = ((z_alpha + z_beta) ** 2 * (1 + 1/ratio)) / (effect_size ** 2)
        n1 = math.ceil(n1)
        n2 = math.ceil(n1 * ratio)

        return PowerResult(
            test_type="Independent t-test",
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            sample_size=n1 + n2,
            sample_size_per_group=n1 if ratio == 1.0 else None,
        )

    def power_ttest_paired(self, effect_size: float, alpha: float = 0.05,
                           power: float = 0.8) -> PowerResult:
        """Power analysis for paired t-test."""
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)

        n = ((z_alpha + z_beta) / effect_size) ** 2
        n = math.ceil(n)

        return PowerResult(
            test_type="Paired t-test",
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            sample_size=n,
        )

    def power_anova(self, effect_size: float, groups: int, alpha: float = 0.05,
                    power: float = 0.8) -> PowerResult:
        """
        Power analysis for one-way ANOVA.

        Args:
            effect_size: Cohen's f (0.1=small, 0.25=medium, 0.4=large)
            groups: Number of groups
            alpha: Significance level
            power: Desired power
        """
        # Convert Cohen's f to lambda (noncentrality parameter)
        # Use iterative approach to find n

        for n_per_group in range(2, 10000):
            total_n = n_per_group * groups
            df1 = groups - 1
            df2 = total_n - groups

            # Noncentrality parameter
            lambda_nc = (effect_size ** 2) * total_n

            # Critical F value
            f_crit = stats.f.ppf(1 - alpha, df1, df2)

            # Power = P(F > f_crit | H1)
            achieved_power = 1 - stats.ncf.cdf(f_crit, df1, df2, lambda_nc)

            if achieved_power >= power:
                return PowerResult(
                    test_type=f"One-way ANOVA ({groups} groups)",
                    effect_size=effect_size,
                    alpha=alpha,
                    power=achieved_power,
                    sample_size=total_n,
                    sample_size_per_group=n_per_group,
                )

        # Fallback for very small effect sizes
        return PowerResult(
            test_type=f"One-way ANOVA ({groups} groups)",
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            sample_size=10000 * groups,
            sample_size_per_group=10000,
        )

    def power_chi_square(self, effect_size: float, df: int, alpha: float = 0.05,
                         power: float = 0.8) -> PowerResult:
        """
        Power analysis for chi-square test.

        Args:
            effect_size: Cohen's w (0.1=small, 0.3=medium, 0.5=large)
            df: Degrees of freedom
            alpha: Significance level
            power: Desired power
        """
        z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)

        # Approximate formula
        n = ((z_alpha + z_beta) / effect_size) ** 2
        n = math.ceil(n)

        return PowerResult(
            test_type=f"Chi-square test (df={df})",
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            sample_size=n,
        )

    def power_correlation(self, r: float, alpha: float = 0.05,
                          power: float = 0.8) -> PowerResult:
        """
        Power analysis for correlation test.

        Args:
            r: Expected correlation coefficient
            alpha: Significance level
            power: Desired power
        """
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)

        # Fisher's z transformation
        z_r = 0.5 * math.log((1 + abs(r)) / (1 - abs(r)))

        n = ((z_alpha + z_beta) / z_r) ** 2 + 3
        n = math.ceil(n)

        return PowerResult(
            test_type="Correlation test",
            effect_size=r,
            alpha=alpha,
            power=power,
            sample_size=n,
        )

    def power_proportion(self, p1: float, p2: float, alpha: float = 0.05,
                         power: float = 0.8, ratio: float = 1.0) -> PowerResult:
        """
        Power analysis for comparing two proportions.

        Args:
            p1: Expected proportion in group 1
            p2: Expected proportion in group 2
            alpha: Significance level
            power: Desired power
            ratio: Ratio of group sizes n2/n1
        """
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)

        p_bar = (p1 + ratio * p2) / (1 + ratio)

        numerator = z_alpha * math.sqrt((1 + 1/ratio) * p_bar * (1 - p_bar))
        numerator += z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)

        n1 = (numerator / (p1 - p2)) ** 2
        n1 = math.ceil(n1)
        n2 = math.ceil(n1 * ratio)

        # Effect size (Cohen's h)
        h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

        return PowerResult(
            test_type="Two proportions",
            effect_size=abs(h),
            alpha=alpha,
            power=power,
            sample_size=n1 + n2,
            sample_size_per_group=n1 if ratio == 1.0 else None,
        )

    def calculate_power_curve(self, test_type: str, effect_sizes: list[float],
                              alpha: float = 0.05, sample_size: int = None,
                              **kwargs) -> list[tuple[float, float]]:
        """
        Calculate power for a range of effect sizes.

        Returns list of (effect_size, power) tuples for plotting.
        """
        results = []

        for es in effect_sizes:
            if test_type == "ttest_ind":
                # Calculate achieved power for given n
                if sample_size:
                    n_per_group = sample_size // 2
                    z_alpha = stats.norm.ppf(1 - alpha/2)
                    se = math.sqrt(2 / n_per_group)
                    z_beta = es / se - z_alpha
                    achieved_power = stats.norm.cdf(z_beta)
                    results.append((es, achieved_power))
                else:
                    result = self.power_ttest_ind(es, alpha, 0.8)
                    results.append((es, 0.8))
            elif test_type == "anova":
                groups = kwargs.get("groups", 3)
                if sample_size:
                    n_per_group = sample_size // groups
                    df1 = groups - 1
                    df2 = sample_size - groups
                    lambda_nc = (es ** 2) * sample_size
                    f_crit = stats.f.ppf(1 - alpha, df1, df2)
                    achieved_power = 1 - stats.ncf.cdf(f_crit, df1, df2, lambda_nc)
                    results.append((es, achieved_power))

        return results


def sample_size_for_mean(margin_of_error: float, std_dev: float,
                         confidence: float = 0.95) -> SampleSizeResult:
    """
    Calculate sample size for estimating a mean.

    Args:
        margin_of_error: Desired margin of error
        std_dev: Expected standard deviation
        confidence: Confidence level (default 0.95)
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    n = math.ceil((z * std_dev / margin_of_error) ** 2)

    return SampleSizeResult(
        test_type="Estimating mean",
        parameters={
            "margin_of_error": margin_of_error,
            "std_dev": std_dev,
            "confidence": f"{confidence:.0%}",
        },
        sample_size=n,
        margin_of_error=margin_of_error,
        confidence_level=confidence,
    )


def sample_size_for_proportion(margin_of_error: float, p: float = 0.5,
                               confidence: float = 0.95) -> SampleSizeResult:
    """
    Calculate sample size for estimating a proportion.

    Args:
        margin_of_error: Desired margin of error
        p: Expected proportion (0.5 is most conservative)
        confidence: Confidence level
    """
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    n = math.ceil((z ** 2 * p * (1 - p)) / (margin_of_error ** 2))

    return SampleSizeResult(
        test_type="Estimating proportion",
        parameters={
            "margin_of_error": margin_of_error,
            "expected_proportion": p,
            "confidence": f"{confidence:.0%}",
        },
        sample_size=n,
        margin_of_error=margin_of_error,
        confidence_level=confidence,
    )


def effect_size_from_means(mean1: float, mean2: float,
                           pooled_std: float) -> float:
    """Calculate Cohen's d from two means and pooled standard deviation."""
    return abs(mean1 - mean2) / pooled_std


def interpret_effect_size(d: float, test_type: str = "d") -> str:
    """Interpret effect size magnitude."""
    if test_type == "d":  # Cohen's d
        if abs(d) < 0.2:
            return "negligible"
        elif abs(d) < 0.5:
            return "small"
        elif abs(d) < 0.8:
            return "medium"
        else:
            return "large"
    elif test_type == "r":  # Correlation
        if abs(d) < 0.1:
            return "negligible"
        elif abs(d) < 0.3:
            return "small"
        elif abs(d) < 0.5:
            return "medium"
        else:
            return "large"
    elif test_type == "f":  # Cohen's f (ANOVA)
        if abs(d) < 0.1:
            return "negligible"
        elif abs(d) < 0.25:
            return "small"
        elif abs(d) < 0.4:
            return "medium"
        else:
            return "large"
    return "unknown"

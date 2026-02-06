"""
Statistical Process Control (SPC) Module

Provides Minitab-like functionality for:
- Control Charts (X-bar/R, X-bar/S, I-MR, p, np, c, u)
- Process Capability (Cp, Cpk, Pp, Ppk, sigma level)
- Process Performance metrics
- Statistical summaries

Designed to integrate with the Problem/DMAIC workflow.
"""

import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
import json


# =============================================================================
# Control Chart Constants
# =============================================================================

# Constants for control chart calculations (from statistical tables)
# A2, A3, B3, B4, D3, D4 for subgroup sizes 2-25
CONTROL_CHART_CONSTANTS = {
    2:  {"A2": 1.880, "A3": 2.659, "B3": 0.000, "B4": 3.267, "D3": 0.000, "D4": 3.267, "d2": 1.128, "c4": 0.7979},
    3:  {"A2": 1.023, "A3": 1.954, "B3": 0.000, "B4": 2.568, "D3": 0.000, "D4": 2.574, "d2": 1.693, "c4": 0.8862},
    4:  {"A2": 0.729, "A3": 1.628, "B3": 0.000, "B4": 2.266, "D3": 0.000, "D4": 2.282, "d2": 2.059, "c4": 0.9213},
    5:  {"A2": 0.577, "A3": 1.427, "B3": 0.000, "B4": 2.089, "D3": 0.000, "D4": 2.114, "d2": 2.326, "c4": 0.9400},
    6:  {"A2": 0.483, "A3": 1.287, "B3": 0.030, "B4": 1.970, "D3": 0.000, "D4": 2.004, "d2": 2.534, "c4": 0.9515},
    7:  {"A2": 0.419, "A3": 1.182, "B3": 0.118, "B4": 1.882, "D3": 0.076, "D4": 1.924, "d2": 2.704, "c4": 0.9594},
    8:  {"A2": 0.373, "A3": 1.099, "B3": 0.185, "B4": 1.815, "D3": 0.136, "D4": 1.864, "d2": 2.847, "c4": 0.9650},
    9:  {"A2": 0.337, "A3": 1.032, "B3": 0.239, "B4": 1.761, "D3": 0.184, "D4": 1.816, "d2": 2.970, "c4": 0.9693},
    10: {"A2": 0.308, "A3": 0.975, "B3": 0.284, "B4": 1.716, "D3": 0.223, "D4": 1.777, "d2": 3.078, "c4": 0.9727},
}

# For I-MR charts (subgroup size = 1)
IMR_CONSTANTS = {
    "E2": 2.660,  # For Individual chart
    "D3": 0.000,  # For MR chart lower limit
    "D4": 3.267,  # For MR chart upper limit
    "d2": 1.128,  # For estimating sigma from MR
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ControlLimits:
    """Control limits for a control chart."""
    ucl: float  # Upper Control Limit
    cl: float   # Center Line
    lcl: float  # Lower Control Limit

    # Optional specification limits
    usl: Optional[float] = None  # Upper Spec Limit
    lsl: Optional[float] = None  # Lower Spec Limit


@dataclass
class ControlChartResult:
    """Result from control chart analysis."""
    chart_type: str
    data_points: list[float]
    limits: ControlLimits

    # Out of control points
    out_of_control: list[dict]  # [{index, value, reason}]

    # Run rules violations (Western Electric rules)
    run_violations: list[dict]  # [{rule, indices, description}]

    # Summary
    in_control: bool
    summary: str

    # For X-bar charts, also include R or S chart
    secondary_chart: Optional["ControlChartResult"] = None

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.secondary_chart:
            result["secondary_chart"] = self.secondary_chart.to_dict()
        return result


@dataclass
class ProcessCapability:
    """Process capability analysis results."""
    # Short-term capability (within subgroup variation)
    cp: float       # Capability index
    cpk: float      # Capability index (centered)
    cpu: float      # Upper capability
    cpl: float      # Lower capability

    # Long-term performance (total variation)
    pp: float       # Performance index
    ppk: float      # Performance index (centered)
    ppu: float      # Upper performance
    ppl: float      # Lower performance

    # Sigma metrics
    sigma_within: float   # Within-subgroup std dev
    sigma_overall: float  # Overall std dev
    sigma_level: float    # Process sigma level (Z score)

    # Defect metrics
    dpmo: float           # Defects per million opportunities
    yield_percent: float  # Process yield %

    # Specs
    usl: float
    lsl: float
    target: Optional[float]

    # Data summary
    mean: float
    n_samples: int

    # Interpretation
    interpretation: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StatisticalSummary:
    """Statistical summary of a dataset."""
    n: int
    mean: float
    median: float
    std_dev: float
    variance: float
    min_val: float
    max_val: float
    range_val: float
    q1: float
    q3: float
    iqr: float
    skewness: float
    kurtosis: float

    # Normality indicators
    anderson_darling: Optional[float] = None
    is_normal: Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Statistical Functions
# =============================================================================

def calculate_summary(data: list[float]) -> StatisticalSummary:
    """Calculate comprehensive statistical summary."""
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 data points")

    sorted_data = sorted(data)
    mean = statistics.mean(data)

    # Quartiles
    q1_idx = n // 4
    q3_idx = (3 * n) // 4
    q1 = sorted_data[q1_idx]
    q3 = sorted_data[q3_idx]

    # Variance and std dev
    variance = statistics.variance(data)
    std_dev = math.sqrt(variance)

    # Skewness (Fisher's)
    if std_dev > 0:
        skewness = sum((x - mean) ** 3 for x in data) / (n * std_dev ** 3)
    else:
        skewness = 0.0

    # Kurtosis (excess)
    if std_dev > 0:
        kurtosis = sum((x - mean) ** 4 for x in data) / (n * std_dev ** 4) - 3
    else:
        kurtosis = 0.0

    return StatisticalSummary(
        n=n,
        mean=mean,
        median=statistics.median(data),
        std_dev=std_dev,
        variance=variance,
        min_val=min(data),
        max_val=max(data),
        range_val=max(data) - min(data),
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def z_to_dpmo(z: float) -> float:
    """Convert Z score to DPMO (using 1.5 sigma shift)."""
    # Approximate using normal CDF
    # P(defect) = 1 - Phi(z) for upper tail
    # With 1.5 sigma shift: use z - 1.5
    z_shifted = z - 1.5

    # Approximate normal CDF using error function approximation
    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    # Two-tailed defect rate
    defect_rate = 1 - norm_cdf(z_shifted) + norm_cdf(-z_shifted - 3)  # Simplified
    defect_rate = max(0, min(1, 1 - norm_cdf(z_shifted)))  # Upper tail only for simplicity

    return defect_rate * 1_000_000


def dpmo_to_sigma(dpmo: float) -> float:
    """Convert DPMO to sigma level (with 1.5 shift)."""
    if dpmo <= 0:
        return 6.0  # Perfect
    if dpmo >= 1_000_000:
        return 0.0

    # Inverse normal approximation
    # Using Beasley-Springer-Moro algorithm approximation
    p = 1 - dpmo / 1_000_000

    if p <= 0:
        return 0.0
    if p >= 1:
        return 6.0

    # Approximate inverse normal
    a = [0, -3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [0, -5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [0, -7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [0, 7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) / \
            ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        z = (((((a[1]*r + a[2])*r + a[3])*r + a[4])*r + a[5])*r + a[6])*q / \
            (((((b[1]*r + b[2])*r + b[3])*r + b[4])*r + b[5])*r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        z = -(((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) / \
             ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)

    # Add 1.5 sigma shift
    return z + 1.5


# =============================================================================
# Control Charts
# =============================================================================

def individuals_moving_range_chart(
    data: list[float],
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
) -> ControlChartResult:
    """
    Create I-MR (Individuals and Moving Range) control chart.

    Used for continuous data with subgroup size = 1.
    """
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 data points")

    # Calculate moving ranges
    moving_ranges = [abs(data[i] - data[i-1]) for i in range(1, n)]

    # Center lines
    x_bar = statistics.mean(data)
    mr_bar = statistics.mean(moving_ranges)

    # Estimate sigma from moving range
    sigma = mr_bar / IMR_CONSTANTS["d2"]

    # Control limits for Individuals chart
    i_ucl = x_bar + 3 * sigma
    i_lcl = x_bar - 3 * sigma

    # Control limits for MR chart
    mr_ucl = IMR_CONSTANTS["D4"] * mr_bar
    mr_lcl = IMR_CONSTANTS["D3"] * mr_bar

    # Find out of control points (Individuals)
    out_of_control = []
    for i, val in enumerate(data):
        if val > i_ucl:
            out_of_control.append({"index": i, "value": val, "reason": "Above UCL"})
        elif val < i_lcl:
            out_of_control.append({"index": i, "value": val, "reason": "Below LCL"})

    # Check Western Electric rules
    run_violations = check_western_electric_rules(data, x_bar, sigma)

    in_control = len(out_of_control) == 0 and len(run_violations) == 0

    # Create MR chart result
    mr_out_of_control = []
    for i, val in enumerate(moving_ranges):
        if val > mr_ucl:
            mr_out_of_control.append({"index": i + 1, "value": val, "reason": "Above UCL"})

    mr_chart = ControlChartResult(
        chart_type="MR",
        data_points=moving_ranges,
        limits=ControlLimits(ucl=mr_ucl, cl=mr_bar, lcl=mr_lcl),
        out_of_control=mr_out_of_control,
        run_violations=[],
        in_control=len(mr_out_of_control) == 0,
        summary=f"MR Chart: Mean={mr_bar:.4f}, UCL={mr_ucl:.4f}",
    )

    summary_parts = [
        f"I-MR Chart Analysis (n={n})",
        f"Process Mean: {x_bar:.4f}",
        f"Estimated Sigma: {sigma:.4f}",
        f"UCL: {i_ucl:.4f}, LCL: {i_lcl:.4f}",
    ]
    if out_of_control:
        summary_parts.append(f"Out of control: {len(out_of_control)} points")
    if run_violations:
        summary_parts.append(f"Run rule violations: {len(run_violations)}")
    if in_control:
        summary_parts.append("Process is IN CONTROL")
    else:
        summary_parts.append("Process is OUT OF CONTROL")

    return ControlChartResult(
        chart_type="I-MR",
        data_points=data,
        limits=ControlLimits(ucl=i_ucl, cl=x_bar, lcl=i_lcl, usl=usl, lsl=lsl),
        out_of_control=out_of_control,
        run_violations=run_violations,
        in_control=in_control,
        summary="\n".join(summary_parts),
        secondary_chart=mr_chart,
    )


def xbar_r_chart(
    subgroups: list[list[float]],
    usl: Optional[float] = None,
    lsl: Optional[float] = None,
) -> ControlChartResult:
    """
    Create X-bar and R control chart.

    Used for continuous data with subgroups of size 2-10.
    """
    # Validate subgroups
    n_subgroups = len(subgroups)
    if n_subgroups < 2:
        raise ValueError("Need at least 2 subgroups")

    subgroup_size = len(subgroups[0])
    if not all(len(sg) == subgroup_size for sg in subgroups):
        raise ValueError("All subgroups must have the same size")

    if subgroup_size < 2 or subgroup_size > 10:
        raise ValueError("Subgroup size must be between 2 and 10 for X-bar R chart")

    constants = CONTROL_CHART_CONSTANTS[subgroup_size]

    # Calculate subgroup means and ranges
    subgroup_means = [statistics.mean(sg) for sg in subgroups]
    subgroup_ranges = [max(sg) - min(sg) for sg in subgroups]

    # Grand mean and average range
    x_bar_bar = statistics.mean(subgroup_means)
    r_bar = statistics.mean(subgroup_ranges)

    # Control limits for X-bar chart
    xbar_ucl = x_bar_bar + constants["A2"] * r_bar
    xbar_lcl = x_bar_bar - constants["A2"] * r_bar

    # Control limits for R chart
    r_ucl = constants["D4"] * r_bar
    r_lcl = constants["D3"] * r_bar

    # Estimate sigma
    sigma = r_bar / constants["d2"]

    # Find out of control points (X-bar)
    out_of_control = []
    for i, val in enumerate(subgroup_means):
        if val > xbar_ucl:
            out_of_control.append({"index": i, "value": val, "reason": "Above UCL"})
        elif val < xbar_lcl:
            out_of_control.append({"index": i, "value": val, "reason": "Below LCL"})

    # Check run rules
    run_violations = check_western_electric_rules(subgroup_means, x_bar_bar, sigma / math.sqrt(subgroup_size))

    # R chart out of control
    r_out_of_control = []
    for i, val in enumerate(subgroup_ranges):
        if val > r_ucl:
            r_out_of_control.append({"index": i, "value": val, "reason": "Above UCL"})
        elif val < r_lcl:
            r_out_of_control.append({"index": i, "value": val, "reason": "Below LCL"})

    r_chart = ControlChartResult(
        chart_type="R",
        data_points=subgroup_ranges,
        limits=ControlLimits(ucl=r_ucl, cl=r_bar, lcl=r_lcl),
        out_of_control=r_out_of_control,
        run_violations=[],
        in_control=len(r_out_of_control) == 0,
        summary=f"R Chart: R-bar={r_bar:.4f}, UCL={r_ucl:.4f}",
    )

    in_control = len(out_of_control) == 0 and len(run_violations) == 0 and len(r_out_of_control) == 0

    summary_parts = [
        f"X-bar R Chart Analysis (k={n_subgroups}, n={subgroup_size})",
        f"Grand Mean (X-bar-bar): {x_bar_bar:.4f}",
        f"Average Range (R-bar): {r_bar:.4f}",
        f"Estimated Sigma: {sigma:.4f}",
        f"X-bar UCL: {xbar_ucl:.4f}, LCL: {xbar_lcl:.4f}",
    ]
    if in_control:
        summary_parts.append("Process is IN CONTROL")
    else:
        summary_parts.append("Process is OUT OF CONTROL")

    return ControlChartResult(
        chart_type="X-bar R",
        data_points=subgroup_means,
        limits=ControlLimits(ucl=xbar_ucl, cl=x_bar_bar, lcl=xbar_lcl, usl=usl, lsl=lsl),
        out_of_control=out_of_control,
        run_violations=run_violations,
        in_control=in_control,
        summary="\n".join(summary_parts),
        secondary_chart=r_chart,
    )


def p_chart(
    defectives: list[int],
    sample_sizes: list[int],
) -> ControlChartResult:
    """
    Create p-chart for proportion defective.

    Used for attribute data (pass/fail, defective/non-defective).
    """
    if len(defectives) != len(sample_sizes):
        raise ValueError("defectives and sample_sizes must have same length")

    n_samples = len(defectives)

    # Calculate proportions
    proportions = [d / n for d, n in zip(defectives, sample_sizes)]

    # Average proportion and sample size
    total_defectives = sum(defectives)
    total_inspected = sum(sample_sizes)
    p_bar = total_defectives / total_inspected
    n_bar = total_inspected / n_samples

    # Control limits (can vary by sample size, use average for simplicity)
    sigma_p = math.sqrt(p_bar * (1 - p_bar) / n_bar)
    ucl = p_bar + 3 * sigma_p
    lcl = max(0, p_bar - 3 * sigma_p)

    # Find out of control points
    out_of_control = []
    for i, (p, n) in enumerate(zip(proportions, sample_sizes)):
        # Use exact limits for each sample
        sigma_i = math.sqrt(p_bar * (1 - p_bar) / n)
        ucl_i = p_bar + 3 * sigma_i
        lcl_i = max(0, p_bar - 3 * sigma_i)

        if p > ucl_i:
            out_of_control.append({"index": i, "value": p, "reason": "Above UCL"})
        elif p < lcl_i:
            out_of_control.append({"index": i, "value": p, "reason": "Below LCL"})

    in_control = len(out_of_control) == 0

    summary_parts = [
        f"p-Chart Analysis (k={n_samples})",
        f"Average Proportion Defective (p-bar): {p_bar:.4f} ({p_bar*100:.2f}%)",
        f"Total Defectives: {total_defectives} / {total_inspected}",
        f"UCL: {ucl:.4f}, LCL: {lcl:.4f}",
    ]
    if in_control:
        summary_parts.append("Process is IN CONTROL")
    else:
        summary_parts.append(f"Process is OUT OF CONTROL ({len(out_of_control)} points)")

    return ControlChartResult(
        chart_type="p",
        data_points=proportions,
        limits=ControlLimits(ucl=ucl, cl=p_bar, lcl=lcl),
        out_of_control=out_of_control,
        run_violations=[],
        in_control=in_control,
        summary="\n".join(summary_parts),
    )


def c_chart(
    defect_counts: list[int],
) -> ControlChartResult:
    """
    Create c-chart for count of defects per unit.

    Used when counting defects in same-sized units.
    """
    n_samples = len(defect_counts)

    # Average defect count
    c_bar = statistics.mean(defect_counts)

    # Control limits (Poisson-based)
    sigma_c = math.sqrt(c_bar)
    ucl = c_bar + 3 * sigma_c
    lcl = max(0, c_bar - 3 * sigma_c)

    # Find out of control points
    out_of_control = []
    for i, c in enumerate(defect_counts):
        if c > ucl:
            out_of_control.append({"index": i, "value": c, "reason": "Above UCL"})
        elif c < lcl:
            out_of_control.append({"index": i, "value": c, "reason": "Below LCL"})

    in_control = len(out_of_control) == 0

    summary_parts = [
        f"c-Chart Analysis (k={n_samples})",
        f"Average Defects (c-bar): {c_bar:.2f}",
        f"UCL: {ucl:.2f}, LCL: {lcl:.2f}",
    ]
    if in_control:
        summary_parts.append("Process is IN CONTROL")
    else:
        summary_parts.append(f"Process is OUT OF CONTROL ({len(out_of_control)} points)")

    return ControlChartResult(
        chart_type="c",
        data_points=[float(c) for c in defect_counts],
        limits=ControlLimits(ucl=ucl, cl=c_bar, lcl=lcl),
        out_of_control=out_of_control,
        run_violations=[],
        in_control=in_control,
        summary="\n".join(summary_parts),
    )


def check_nelson_rules(
    data: list[float],
    center: float,
    sigma: float,
) -> list[dict]:
    """
    Check Nelson Rules (complete set of 8 rules) for control charts.

    Nelson Rules:
    1. One point beyond 3 sigma - detected as out-of-control point
    2. Nine points in a row on same side of center
    3. Six points in a row steadily increasing or decreasing
    4. Fourteen points in a row alternating up and down
    5. Two of three consecutive points beyond 2 sigma (same side)
    6. Four of five consecutive points beyond 1 sigma (same side)
    7. Fifteen consecutive points within 1 sigma of center (stratification)
    8. Eight points in a row beyond 1 sigma (either side, mixture)
    """
    violations = []
    n = len(data)

    if n < 2 or sigma <= 0:
        return violations

    # Calculate z-scores for zone classification
    zones = [(x - center) / sigma for x in data]

    # Rule 2: Nine points in a row on same side of center
    if n >= 9:
        for i in range(8, n):
            window = zones[i-8:i+1]
            if all(z > 0 for z in window):
                violations.append({
                    "rule": 2,
                    "indices": list(range(i-8, i+1)),
                    "description": "9 consecutive points above center line",
                })
            elif all(z < 0 for z in window):
                violations.append({
                    "rule": 2,
                    "indices": list(range(i-8, i+1)),
                    "description": "9 consecutive points below center line",
                })

    # Rule 3: Six points in a row steadily increasing or decreasing
    if n >= 6:
        for i in range(5, n):
            window = data[i-5:i+1]
            increasing = all(window[j] < window[j+1] for j in range(5))
            decreasing = all(window[j] > window[j+1] for j in range(5))
            if increasing:
                violations.append({
                    "rule": 3,
                    "indices": list(range(i-5, i+1)),
                    "description": "6 consecutive points steadily increasing",
                })
            elif decreasing:
                violations.append({
                    "rule": 3,
                    "indices": list(range(i-5, i+1)),
                    "description": "6 consecutive points steadily decreasing",
                })

    # Rule 4: Fourteen points in a row alternating up and down
    if n >= 14:
        for i in range(13, n):
            window = data[i-13:i+1]
            alternating = True
            for j in range(13):
                if j % 2 == 0:
                    if window[j] >= window[j+1]:
                        alternating = False
                        break
                else:
                    if window[j] <= window[j+1]:
                        alternating = False
                        break
            if not alternating:
                # Check opposite pattern
                alternating = True
                for j in range(13):
                    if j % 2 == 0:
                        if window[j] <= window[j+1]:
                            alternating = False
                            break
                    else:
                        if window[j] >= window[j+1]:
                            alternating = False
                            break
            if alternating:
                violations.append({
                    "rule": 4,
                    "indices": list(range(i-13, i+1)),
                    "description": "14 consecutive points alternating up and down",
                })

    # Rule 5: Two of three consecutive points beyond 2 sigma (same side)
    if n >= 3:
        for i in range(2, n):
            window = zones[i-2:i+1]
            above_2 = sum(1 for z in window if z > 2)
            below_2 = sum(1 for z in window if z < -2)
            if above_2 >= 2:
                violations.append({
                    "rule": 5,
                    "indices": list(range(i-2, i+1)),
                    "description": "2 of 3 points beyond +2 sigma",
                })
            if below_2 >= 2:
                violations.append({
                    "rule": 5,
                    "indices": list(range(i-2, i+1)),
                    "description": "2 of 3 points beyond -2 sigma",
                })

    # Rule 6: Four of five consecutive points beyond 1 sigma (same side)
    if n >= 5:
        for i in range(4, n):
            window = zones[i-4:i+1]
            above_1 = sum(1 for z in window if z > 1)
            below_1 = sum(1 for z in window if z < -1)
            if above_1 >= 4:
                violations.append({
                    "rule": 6,
                    "indices": list(range(i-4, i+1)),
                    "description": "4 of 5 points beyond +1 sigma",
                })
            if below_1 >= 4:
                violations.append({
                    "rule": 6,
                    "indices": list(range(i-4, i+1)),
                    "description": "4 of 5 points beyond -1 sigma",
                })

    # Rule 7: Fifteen consecutive points within 1 sigma of center (stratification)
    if n >= 15:
        for i in range(14, n):
            window = zones[i-14:i+1]
            if all(-1 < z < 1 for z in window):
                violations.append({
                    "rule": 7,
                    "indices": list(range(i-14, i+1)),
                    "description": "15 consecutive points within +/- 1 sigma (stratification)",
                })

    # Rule 8: Eight points in a row beyond 1 sigma (either side, mixture)
    if n >= 8:
        for i in range(7, n):
            window = zones[i-7:i+1]
            if all(abs(z) > 1 for z in window):
                # Check that it's a mixture (not all on one side)
                above = sum(1 for z in window if z > 1)
                below = sum(1 for z in window if z < -1)
                if above > 0 and below > 0:
                    violations.append({
                        "rule": 8,
                        "indices": list(range(i-7, i+1)),
                        "description": "8 consecutive points beyond +/- 1 sigma (mixture)",
                    })

    # Deduplicate violations (same rule, overlapping indices)
    seen = set()
    unique_violations = []
    for v in violations:
        key = (v["rule"], tuple(v["indices"]))
        if key not in seen:
            seen.add(key)
            unique_violations.append(v)

    return unique_violations


# Alias for backwards compatibility
def check_western_electric_rules(
    data: list[float],
    center: float,
    sigma: float,
) -> list[dict]:
    """Alias for check_nelson_rules for backwards compatibility."""
    return check_nelson_rules(data, center, sigma)


# =============================================================================
# Process Capability
# =============================================================================

def calculate_capability(
    data: list[float],
    usl: float,
    lsl: float,
    target: Optional[float] = None,
    subgroup_size: int = 1,
) -> ProcessCapability:
    """
    Calculate process capability indices.

    Args:
        data: Measurement data (flat list)
        usl: Upper specification limit
        lsl: Lower specification limit
        target: Target value (defaults to midpoint of specs)
        subgroup_size: Size of rational subgroups for within-group sigma estimate
    """
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 data points")

    if usl <= lsl:
        raise ValueError("USL must be greater than LSL")

    if target is None:
        target = (usl + lsl) / 2

    mean = statistics.mean(data)

    # Overall standard deviation
    sigma_overall = statistics.stdev(data)

    # Within-subgroup standard deviation estimate
    if subgroup_size == 1:
        # Use moving range method
        moving_ranges = [abs(data[i] - data[i-1]) for i in range(1, n)]
        mr_bar = statistics.mean(moving_ranges)
        sigma_within = mr_bar / IMR_CONSTANTS["d2"]
    else:
        # Use pooled within-subgroup variance
        # Reshape data into subgroups
        n_subgroups = n // subgroup_size
        subgroups = [data[i*subgroup_size:(i+1)*subgroup_size] for i in range(n_subgroups)]

        if subgroup_size in CONTROL_CHART_CONSTANTS:
            # Use R-bar method
            ranges = [max(sg) - min(sg) for sg in subgroups]
            r_bar = statistics.mean(ranges)
            sigma_within = r_bar / CONTROL_CHART_CONSTANTS[subgroup_size]["d2"]
        else:
            # Use pooled std dev
            within_vars = [statistics.variance(sg) for sg in subgroups if len(sg) > 1]
            sigma_within = math.sqrt(statistics.mean(within_vars)) if within_vars else sigma_overall

    # Specification width
    spec_width = usl - lsl

    # Short-term capability (Cp, Cpk)
    cp = spec_width / (6 * sigma_within) if sigma_within > 0 else 0
    cpu = (usl - mean) / (3 * sigma_within) if sigma_within > 0 else 0
    cpl = (mean - lsl) / (3 * sigma_within) if sigma_within > 0 else 0
    cpk = min(cpu, cpl)

    # Long-term performance (Pp, Ppk)
    pp = spec_width / (6 * sigma_overall) if sigma_overall > 0 else 0
    ppu = (usl - mean) / (3 * sigma_overall) if sigma_overall > 0 else 0
    ppl = (mean - lsl) / (3 * sigma_overall) if sigma_overall > 0 else 0
    ppk = min(ppu, ppl)

    # Sigma level (based on Cpk)
    sigma_level = 3 * cpk if cpk > 0 else 0

    # DPMO calculation
    # Using normal distribution approximation
    def norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    z_upper = (usl - mean) / sigma_overall if sigma_overall > 0 else float('inf')
    z_lower = (mean - lsl) / sigma_overall if sigma_overall > 0 else float('inf')

    p_above_usl = 1 - norm_cdf(z_upper)
    p_below_lsl = norm_cdf(-z_lower)
    total_defect_rate = p_above_usl + p_below_lsl

    dpmo = total_defect_rate * 1_000_000
    yield_percent = (1 - total_defect_rate) * 100

    # Interpretation
    if cpk >= 2.0:
        interpretation = "Excellent: World-class capability (Six Sigma level)"
    elif cpk >= 1.67:
        interpretation = "Very Good: Capable process with good margin"
    elif cpk >= 1.33:
        interpretation = "Good: Process is capable"
    elif cpk >= 1.0:
        interpretation = "Marginal: Process barely meets specs, improvement needed"
    elif cpk >= 0.67:
        interpretation = "Poor: Process produces significant defects"
    else:
        interpretation = "Very Poor: Process is not capable, major improvement needed"

    return ProcessCapability(
        cp=cp,
        cpk=cpk,
        cpu=cpu,
        cpl=cpl,
        pp=pp,
        ppk=ppk,
        ppu=ppu,
        ppl=ppl,
        sigma_within=sigma_within,
        sigma_overall=sigma_overall,
        sigma_level=sigma_level,
        dpmo=dpmo,
        yield_percent=yield_percent,
        usl=usl,
        lsl=lsl,
        target=target,
        mean=mean,
        n_samples=n,
        interpretation=interpretation,
    )


# =============================================================================
# Helper Functions for API
# =============================================================================

def recommend_chart_type(
    data_type: Literal["continuous", "attribute"],
    subgroup_size: int = 1,
    attribute_type: Optional[Literal["defectives", "defects"]] = None,
) -> str:
    """Recommend appropriate control chart type."""
    if data_type == "continuous":
        if subgroup_size == 1:
            return "I-MR"
        elif subgroup_size <= 10:
            return "X-bar R"
        else:
            return "X-bar S"
    else:  # attribute
        if attribute_type == "defectives":
            return "p" if subgroup_size > 1 else "np"
        else:  # defects
            return "c" if subgroup_size == 1 else "u"


def parse_csv_data(csv_text: str, has_header: bool = True) -> list[float]:
    """Parse CSV data into flat list of floats."""
    lines = csv_text.strip().split('\n')
    if has_header:
        lines = lines[1:]

    data = []
    for line in lines:
        for val in line.split(','):
            try:
                data.append(float(val.strip()))
            except ValueError:
                continue

    return data


def parse_subgroup_data(csv_text: str, has_header: bool = True) -> list[list[float]]:
    """Parse CSV data into subgroups (each row is a subgroup)."""
    lines = csv_text.strip().split('\n')
    if has_header:
        lines = lines[1:]

    subgroups = []
    for line in lines:
        subgroup = []
        for val in line.split(','):
            try:
                subgroup.append(float(val.strip()))
            except ValueError:
                continue
        if subgroup:
            subgroups.append(subgroup)

    return subgroups


# =============================================================================
# File Parsing for XLSX/CSV Upload
# =============================================================================

@dataclass
class ColumnInfo:
    """Information about a data column."""
    name: str
    dtype: str  # 'numeric', 'datetime', 'text', 'mixed'
    sample_values: list
    null_count: int
    unique_count: int
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParsedDataset:
    """Result of parsing an uploaded file."""
    filename: str
    row_count: int
    columns: list[ColumnInfo]
    data: dict[str, list]  # column_name -> values
    errors: list[str]

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "row_count": self.row_count,
            "columns": [c.to_dict() for c in self.columns],
            "preview": {col: vals[:10] for col, vals in self.data.items()},  # First 10 rows
            "errors": self.errors,
        }


def parse_uploaded_file(file_path: str, filename: str) -> ParsedDataset:
    """
    Parse an uploaded XLSX or CSV file.

    Returns column information and data for field mapping.
    """
    import pandas as pd
    from pathlib import Path

    errors = []
    ext = Path(filename).suffix.lower()

    try:
        if ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, engine='openpyxl')
        elif ext == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        return ParsedDataset(
            filename=filename,
            row_count=0,
            columns=[],
            data={},
            errors=[f"Failed to parse file: {str(e)}"],
        )

    # Analyze columns
    columns = []
    data = {}

    for col in df.columns:
        series = df[col]
        null_count = series.isna().sum()

        # Determine data type
        if pd.api.types.is_numeric_dtype(series):
            dtype = "numeric"
            valid_values = series.dropna()
            col_info = ColumnInfo(
                name=str(col),
                dtype=dtype,
                sample_values=valid_values.head(5).tolist(),
                null_count=int(null_count),
                unique_count=int(series.nunique()),
                min_val=float(valid_values.min()) if len(valid_values) > 0 else None,
                max_val=float(valid_values.max()) if len(valid_values) > 0 else None,
                mean_val=float(valid_values.mean()) if len(valid_values) > 0 else None,
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            dtype = "datetime"
            col_info = ColumnInfo(
                name=str(col),
                dtype=dtype,
                sample_values=[str(v) for v in series.dropna().head(5).tolist()],
                null_count=int(null_count),
                unique_count=int(series.nunique()),
            )
        else:
            # Try to detect datetime strings
            try:
                pd.to_datetime(series.dropna().head(10))
                dtype = "datetime"
            except (ValueError, TypeError):
                dtype = "text"

            col_info = ColumnInfo(
                name=str(col),
                dtype=dtype,
                sample_values=[str(v) for v in series.dropna().head(5).tolist()],
                null_count=int(null_count),
                unique_count=int(series.nunique()),
            )

        columns.append(col_info)

        # Store data (convert to native Python types)
        if dtype == "numeric":
            data[str(col)] = [None if pd.isna(v) else float(v) for v in series]
        else:
            data[str(col)] = [None if pd.isna(v) else str(v) for v in series]

    return ParsedDataset(
        filename=filename,
        row_count=len(df),
        columns=columns,
        data=data,
        errors=errors,
    )


def extract_spc_data(
    parsed: ParsedDataset,
    value_column: str,
    subgroup_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
) -> dict:
    """
    Extract SPC-ready data from a parsed dataset.

    Args:
        parsed: ParsedDataset from parse_uploaded_file
        value_column: Name of column containing measurement values
        subgroup_column: Optional column for subgroup identifier
        timestamp_column: Optional column for timestamps (for ordering)

    Returns:
        Dict with 'data' (flat list or subgroups) and 'metadata'
    """
    if value_column not in parsed.data:
        raise ValueError(f"Column '{value_column}' not found in dataset")

    values = parsed.data[value_column]

    # Filter out None values
    if subgroup_column and subgroup_column in parsed.data:
        # Group by subgroup column
        subgroups_dict: dict[str, list[float]] = {}
        subgroup_ids = parsed.data[subgroup_column]

        for i, (val, sg_id) in enumerate(zip(values, subgroup_ids)):
            if val is not None and sg_id is not None:
                sg_key = str(sg_id)
                if sg_key not in subgroups_dict:
                    subgroups_dict[sg_key] = []
                subgroups_dict[sg_key].append(float(val))

        # Convert to list of lists (ordered by subgroup key)
        subgroup_keys = sorted(subgroups_dict.keys())
        subgroups = [subgroups_dict[k] for k in subgroup_keys]

        return {
            "type": "subgroups",
            "data": subgroups,
            "subgroup_ids": subgroup_keys,
            "n_subgroups": len(subgroups),
            "subgroup_size": len(subgroups[0]) if subgroups else 0,
        }
    else:
        # Flat data
        flat_data = [float(v) for v in values if v is not None]

        return {
            "type": "individual",
            "data": flat_data,
            "n_points": len(flat_data),
        }

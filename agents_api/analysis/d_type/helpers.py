"""D-Type shared helpers — KDE, JSD, noise floor, capability, and narrative.

CR: 3c0d0e53
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared mathematical core
# ---------------------------------------------------------------------------


def _kde_density(x, grid, bandwidth=None):
    """Gaussian KDE with ISJ bandwidth via KDEpy FFTKDE (fast).

    Falls back to scipy gaussian_kde with Silverman if KDEpy unavailable.
    Returns density array evaluated at `grid` points.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.ones_like(grid) / (grid[-1] - grid[0])

    try:
        from KDEpy import FFTKDE

        bw = bandwidth or "ISJ"
        # FFTKDE evaluates on its own grid; we interpolate to ours
        _grid, density = FFTKDE(bw=bw, kernel="gaussian").fit(x).evaluate(len(grid))
        # Interpolate to our grid
        density = np.interp(grid, _grid, density)
    except Exception:
        from scipy.stats import gaussian_kde

        bw = bandwidth or "silverman"
        try:
            kde = gaussian_kde(x, bw_method=float(bw) if isinstance(bw, (int, float)) else bw)
            density = kde(grid)
        except Exception:
            kde = gaussian_kde(x, bw_method="silverman")
            density = kde(grid)

    # Clamp and normalize
    density = np.maximum(density, 0)
    total = np.trapz(density, grid)
    if total > 0:
        density = density / total
    return density


def _jsd(p, q, grid):
    """Jensen-Shannon Divergence in bits (base 2), bounded [0, 1].

    Uses scipy's jensenshannon (which returns the *distance*, i.e. sqrt(JSD)).
    We square it to get the actual divergence.
    """
    from scipy.spatial.distance import jensenshannon

    # Normalize to proper PMFs over grid with epsilon floor
    # (scipy's jensenshannon uses rel_entr internally which gives inf when q=0, p>0)
    p = np.maximum(p, 0)
    q = np.maximum(q, 0)
    dx = np.diff(grid)
    dx = np.append(dx, dx[-1])
    p_pmf = p * dx + 1e-300
    q_pmf = q * dx + 1e-300
    p_pmf = p_pmf / p_pmf.sum()
    q_pmf = q_pmf / q_pmf.sum()

    js_dist = jensenshannon(p_pmf, q_pmf, base=2)
    if not np.isfinite(js_dist):
        return 0.0
    return float(js_dist**2)  # divergence = distance²


def _jsd_tail(p, q, grid, lsl=None, usl=None):
    """Tail contribution to total JSD — the portion of divergence in out-of-spec regions.

    Uses the same PMF normalization as the full JSD, then sums only the
    element-wise divergence terms that fall outside spec limits. This gives a
    proper decomposition: tail_contribution + body_contribution = total JSD.
    """
    tail_mask = np.zeros_like(grid, dtype=bool)
    if lsl is not None:
        tail_mask |= grid < lsl
    if usl is not None:
        tail_mask |= grid > usl

    if not tail_mask.any():
        return 0.0

    # Build PMFs with same normalization as _jsd
    p = np.maximum(p, 0)
    q = np.maximum(q, 0)
    dx = np.diff(grid)
    dx = np.append(dx, dx[-1])
    p_pmf = p * dx + 1e-300
    q_pmf = q * dx + 1e-300
    p_pmf = p_pmf / p_pmf.sum()
    q_pmf = q_pmf / q_pmf.sum()

    # M = midpoint distribution
    m_pmf = 0.5 * (p_pmf + q_pmf)

    # Element-wise JSD contribution: 0.5 * [p*log(p/m) + q*log(q/m)]
    # Only sum terms in the tail region
    eps = 1e-300
    tail_jsd = 0.0
    for i in np.where(tail_mask)[0]:
        pi, qi, mi = p_pmf[i], q_pmf[i], m_pmf[i]
        if mi > eps:
            if pi > eps:
                tail_jsd += 0.5 * pi * np.log2(pi / mi)
            if qi > eps:
                tail_jsd += 0.5 * qi * np.log2(qi / mi)

    return max(0.0, float(tail_jsd))


def _decompose_divergence(fdata, ref_data):
    """Decompose distributional divergence into location and scale components.

    Returns (location_pct, scale_pct) — percentage of divergence attributable
    to mean shift vs variance change. Uses squared z-score decomposition.
    """
    mu_f, mu_r = fdata.mean(), ref_data.mean()
    sd_f, sd_r = fdata.std(ddof=1), ref_data.std(ddof=1)

    if sd_r == 0 or sd_f == 0:
        return 0.5, 0.5  # can't decompose

    # Squared standardized effects
    location_effect = ((mu_f - mu_r) / sd_r) ** 2
    scale_effect = (sd_f / sd_r - 1) ** 2

    total = location_effect + scale_effect
    if total == 0:
        return 0.5, 0.5
    return location_effect / total, scale_effect / total


def _noise_floor(pooled, n_per_group, grid, B=200, quantile=0.95, rng=None):
    """Bootstrap noise floor: expected JSD from random splits of pooled data.

    Splits pooled array into two halves B times, computes JSD each time,
    returns the `quantile`-th percentile as the noise floor.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pooled = np.asarray(pooled, dtype=float)
    pooled = pooled[np.isfinite(pooled)]
    n = len(pooled)
    if n < 10:
        return 0.0

    half = max(n // 2, 5)
    jsds = []
    for _ in range(B):
        idx = rng.permutation(n)
        a = pooled[idx[:half]]
        b = pooled[idx[half : half * 2]]
        pa = _kde_density(a, grid)
        pb = _kde_density(b, grid)
        jsds.append(_jsd(pa, pb, grid))

    return float(np.percentile(jsds, quantile * 100))


def _build_grid(data, n_points=512):
    """Build evaluation grid spanning the data range with 10% padding."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    lo, hi = data.min(), data.max()
    margin = (hi - lo) * 0.1 if hi > lo else 1.0
    return np.linspace(lo - margin, hi + margin, n_points)


# ---------------------------------------------------------------------------
# Capability helpers
# ---------------------------------------------------------------------------


def _p_within_spec(density, grid, lsl, usl):
    """Probability of being within spec limits given a density over grid."""
    mask = np.ones_like(grid, dtype=bool)
    if lsl is not None:
        mask &= grid >= lsl
    if usl is not None:
        mask &= grid <= usl
    return float(np.trapz(density[mask], grid[mask]))


def _compute_cpk(data, lsl, usl):
    """Classical Cpk from data."""
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return 0.0
    mu = data.mean()
    sigma = data.std(ddof=1)
    if sigma == 0:
        return 0.0
    cpks = []
    if usl is not None:
        cpks.append((usl - mu) / (3 * sigma))
    if lsl is not None:
        cpks.append((mu - lsl) / (3 * sigma))
    return min(cpks) if cpks else 0.0


def _bernoulli_jsd(p1, p2):
    """JSD between two Bernoulli distributions Bernoulli(p1) and Bernoulli(p2)."""
    from scipy.spatial.distance import jensenshannon

    # Clamp to avoid log(0)
    eps = 1e-12
    p1 = np.clip(p1, eps, 1 - eps)
    p2 = np.clip(p2, eps, 1 - eps)
    dist1 = np.array([p1, 1 - p1])
    dist2 = np.array([p2, 1 - p2])
    js_dist = jensenshannon(dist1, dist2, base=2)
    return float(js_dist**2)


def _cpk_noise_floor(values, n_factors, grid, lsl, usl, B=200):
    """Bootstrap noise floor for capability JSD.

    Randomly assigns factor labels and computes JSD of resulting Bernoulli capabilities.
    """
    rng = np.random.default_rng(42)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 10:
        return 0.0

    pooled_density = _kde_density(values, grid)
    pooled_pws = _p_within_spec(pooled_density, grid, lsl, usl)

    jsds = []
    group_size = max(n // n_factors, 10)
    for _ in range(B):
        idx = rng.permutation(n)
        subset = values[idx[:group_size]]
        sub_density = _kde_density(subset, grid)
        sub_pws = _p_within_spec(sub_density, grid, lsl, usl)
        jsds.append(_bernoulli_jsd(sub_pws, pooled_pws))

    return float(np.percentile(jsds, 95))


# ---------------------------------------------------------------------------
# Narrative helpers
# ---------------------------------------------------------------------------


def _d_narrative(title, body, next_steps, chart_guidance):
    """Build HTML narrative string matching DSW standard format."""
    parts = [f'<div class="dsw-verdict">{title}</div>', f"<p>{body}</p>"]
    if chart_guidance:
        parts.append(f"<p><strong>In the chart:</strong> {chart_guidance}</p>")
    if next_steps:
        parts.append(f'<div class="dsw-next"><strong>Next &rarr;</strong> {next_steps}</div>')
    return "\n".join(parts)


def _d_chart_body(sorted_factors, noise, any_above_noise, variable, factor):
    top = sorted_factors[0] if sorted_factors else ("N/A", 0)
    if any_above_noise and top[1] > noise * 2:
        return (
            f"The divergence analysis reveals that <strong>{top[0]}</strong> is the most divergent "
            f"level of {factor}, with a cumulative information score of {top[1]:.4f} — "
            f"well above the noise floor of {noise:.4f}. This means {top[0]} produces a "
            f"systematically different distribution of {variable} compared to the overall process."
        )
    elif any_above_noise:
        return (
            f"Some windows show factor divergence above the noise floor ({noise:.4f}), "
            f"but the cumulative pattern is moderate. The top factor is {top[0]} "
            f"with an information score of {top[1]:.4f}."
        )
    return (
        f"All levels of {factor} produce distributions of {variable} that are "
        f"statistically indistinguishable from the pooled distribution (all below "
        f"noise floor of {noise:.4f}). The process is factor-invariant."
    )


def _d_chart_nextsteps(sorted_factors, noise, any_above_noise, factor):
    top = sorted_factors[0] if sorted_factors else ("N/A", 0)
    if any_above_noise and top[1] > noise * 2:
        return (
            f"Investigate what is different about {top[0]}. "
            f"Run a targeted comparison (B-tTest or B-ANOVA) between {top[0]} and other levels. "
            f"Consider a D-Cpk analysis to quantify capability impact."
        )
    return "No action required — factor divergence is within expected random variation."


def _d_cpk_body(factor_results, noise, pooled_cpk, variable, factor, spec_str):
    significant = [fr for fr in factor_results if fr["jsd"] > noise]
    if not significant:
        return (
            f"No factor level shows significant capability divergence ({spec_str}). "
            f"The pooled Cpk of {pooled_cpk:.3f} is attributable to common-cause variation "
            f"across all levels of {factor}."
        )
    worst = significant[0]
    return (
        f"<strong>{worst['factor']}</strong> is the primary driver of capability divergence. "
        f"Its Cpk ({worst['cpk']:.3f}) differs significantly from the pooled value ({pooled_cpk:.3f}). "
        f"Counterfactual analysis shows removing {worst['factor']} would change Cpk to "
        f"{worst['cpk_without']:.3f} (Δ = {worst.get('delta_cpk', 0):+.3f})."
    )


def _d_cpk_nextsteps(factor_results, noise, factor):
    significant = [fr for fr in factor_results if fr["jsd"] > noise]
    if not significant:
        return "No action required — capability is consistent across all factor levels."
    worst = significant[0]
    if worst["direction"] > 0:
        return (
            f"Focus improvement on {worst['factor']}. "
            f"Investigate root cause of degraded capability — consider an RCA session. "
            f"Run a D-Chart to track whether divergence is chronic or intermittent."
        )
    return (
        f"Factor {worst['factor']} actually performs BETTER than pooled. "
        f"Study what makes this level effective and transfer the practice to other levels."
    )

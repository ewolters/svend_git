"""DSW Exploratory — multivariate analyses (Hotelling T², MANOVA, Nested ANOVA)."""

import logging

import numpy as np
from scipy import stats

from ..common import _narrative

logger = logging.getLogger(__name__)


def run_hotelling_t2(df, config):
    """
    Hotelling's T² Test — multivariate extension of the two-sample t-test.
    Tests whether two groups have different mean vectors across multiple response variables.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    responses = config.get("responses", [])
    group_var = config.get("group_var") or config.get("factor")
    alpha = 1 - config.get("conf", 95) / 100

    data = df[responses + [group_var]].dropna()
    groups = sorted(data[group_var].unique().tolist(), key=str)

    if len(groups) != 2:
        result["summary"] = f"Hotelling's T² requires exactly 2 groups. Found {len(groups)}: {groups}"
        return result

    g1_data = data[data[group_var] == groups[0]][responses].values
    g2_data = data[data[group_var] == groups[1]][responses].values
    n1, n2 = len(g1_data), len(g2_data)
    p = len(responses)

    mean1 = np.mean(g1_data, axis=0)
    mean2 = np.mean(g2_data, axis=0)
    diff = mean1 - mean2

    # Pooled covariance matrix
    S1 = np.cov(g1_data, rowvar=False, ddof=1)
    S2 = np.cov(g2_data, rowvar=False, ddof=1)
    S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

    # Hotelling's T²
    S_inv = np.linalg.inv(S_pooled * (1.0 / n1 + 1.0 / n2))
    T2 = float(diff @ S_inv @ diff)

    # Convert to F-statistic
    df1 = p
    df2 = n1 + n2 - p - 1
    F_stat = T2 * df2 / (p * (n1 + n2 - 2))
    p_value = float(1 - stats.f.cdf(F_stat, df1, df2))

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>HOTELLING'S T² TEST<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Responses:<</COLOR>> {', '.join(responses)}\n"
    summary += f"<<COLOR:highlight>>Group variable:<</COLOR>> {group_var}\n"
    summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} (n={n1}) vs {groups[1]} (n={n2})\n\n"

    summary += "<<COLOR:accent>>── Group Means ──<</COLOR>>\n"
    summary += f"{'Variable':<20} {str(groups[0]):>12} {str(groups[1]):>12} {'Difference':>12}\n"
    summary += f"{'─' * 58}\n"
    for i, var in enumerate(responses):
        summary += f"{var:<20} {mean1[i]:>12.4f} {mean2[i]:>12.4f} {diff[i]:>12.4f}\n"

    summary += "\n<<COLOR:accent>>── Test Statistics ──<</COLOR>>\n"
    summary += f"  Hotelling's T²: {T2:.4f}\n"
    summary += f"  F-statistic: {F_stat:.4f} (df1={df1}, df2={df2})\n"
    summary += f"  p-value: {p_value:.4f}\n\n"

    if p_value < alpha:
        summary += f"<<COLOR:good>>Mean vectors differ significantly (p < {alpha})<</COLOR>>\n"
        summary += (
            "<<COLOR:text>>The groups have different multivariate profiles across the response variables.<</COLOR>>"
        )
    else:
        summary += f"<<COLOR:text>>No significant difference in mean vectors (p >= {alpha})<</COLOR>>"

    result["summary"] = summary

    # Radar/profile plot of group means
    traces = []
    for idx, grp in enumerate(groups):
        grp_means = [float(mean1[i]) if idx == 0 else float(mean2[i]) for i in range(p)]
        colors = ["#4a9f6e", "#47a5e8"]
        traces.append(
            {
                "type": "scatterpolar",
                "r": grp_means + [grp_means[0]],
                "theta": responses + [responses[0]],
                "name": str(grp),
                "fill": "toself",
                "fillcolor": f"rgba({','.join(str(int(c, 16)) for c in [colors[idx][1:3], colors[idx][3:5], colors[idx][5:7]])}, 0.15)",
                "line": {"color": colors[idx]},
            }
        )
    result["plots"].append(
        {
            "title": "Multivariate Profile — Group Means",
            "data": traces,
            "layout": {"height": 350, "polar": {"radialaxis": {"visible": True}}},
        }
    )

    # Per-variable box plots by group
    box_traces = []
    grp_colors = ["#4a9f6e", "#47a5e8"]
    for gi, grp in enumerate(groups):
        grp_data = data[data[group_var] == grp]
        for vi, var in enumerate(responses):
            box_traces.append(
                {
                    "type": "box",
                    "y": grp_data[var].tolist(),
                    "x": [var] * len(grp_data),
                    "name": str(grp),
                    "marker": {"color": grp_colors[gi]},
                    "legendgroup": str(grp),
                    "showlegend": vi == 0,
                }
            )
    result["plots"].append(
        {
            "title": "Response Distributions by Group",
            "data": box_traces,
            "layout": {
                "height": 300,
                "boxmode": "group",
                "xaxis": {"title": "Response Variable"},
                "yaxis": {"title": "Value"},
            },
        }
    )

    result["guide_observation"] = f"Hotelling's T² = {T2:.2f}, F = {F_stat:.2f}, p = {p_value:.4f}. " + (
        "Groups differ." if p_value < alpha else "No difference."
    )
    result["statistics"] = {
        "T2": T2,
        "F_statistic": F_stat,
        "p_value": p_value,
        "df1": df1,
        "df2": df2,
        "mean_diff": diff.tolist(),
    }
    if p_value < alpha:
        result["narrative"] = _narrative(
            f"Multivariate means differ (T\u00b2 = {T2:.2f}, p = {p_value:.4f})",
            "The groups differ when considering all response variables simultaneously. Hotelling's T\u00b2 is the multivariate extension of the t-test.",
            next_steps="Examine individual variables to identify which drive the difference. Consider MANOVA for more than 2 groups.",
        )
    else:
        result["narrative"] = _narrative(
            f"No multivariate difference (p = {p_value:.4f})",
            "The groups do not differ significantly across the combined set of variables.",
            next_steps="Even if individual variables differ, the multivariate test considers their correlation structure.",
        )

    return result


def run_manova(df, config):
    """
    One-Way MANOVA — Multivariate Analysis of Variance.
    Tests whether group means differ across multiple response variables simultaneously.
    Reports Wilks' Lambda, Pillai's Trace, Hotelling-Lawley Trace, Roy's Largest Root.

    This function handles both MANOVA code paths from the original file.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    responses = config.get("responses", [])
    factor = config.get("factor") or config.get("group_var") or config.get("group")
    alpha = 1 - config.get("conf", 95) / 100

    if not responses and config.get("response"):
        responses = [config["response"]]

    # Determine which code path based on config structure
    # The first MANOVA block uses "factor" or "group_var" with no fallback to "group"
    # and does not check for config.get("response") as single item
    # We unify both paths here — the second path is more general
    # Try the first (more detailed) path first if we have proper responses
    if responses and factor:
        return _manova_detailed(df, config, responses, factor, alpha)
    else:
        # Fallback — shouldn't normally happen but maintain compatibility
        result["summary"] = "MANOVA requires response variables and a factor."
        return result


def _manova_detailed(df, config, responses, factor, alpha):
    """First MANOVA implementation — detailed with all four test statistics."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    data = df[responses + [factor]].dropna()
    groups = sorted(data[factor].unique().tolist(), key=str)
    k = len(groups)
    p = len(responses)
    N = len(data)

    if k < 2:
        result["summary"] = f"MANOVA requires at least 2 groups. Found {k}."
        return result

    # Overall mean
    grand_mean = data[responses].values.mean(axis=0)

    # Between-groups (hypothesis) and within-groups (error) SSCP matrices
    H = np.zeros((p, p))  # Hypothesis SSCP
    E = np.zeros((p, p))  # Error SSCP

    group_means = {}
    for grp in groups:
        grp_data = data[data[factor] == grp][responses].values
        n_g = len(grp_data)
        grp_mean = grp_data.mean(axis=0)
        group_means[grp] = {"mean": grp_mean, "n": n_g}

        diff = (grp_mean - grand_mean).reshape(-1, 1)
        H += n_g * (diff @ diff.T)

        centered = grp_data - grp_mean
        E += centered.T @ centered

    # Four test statistics
    try:
        E_inv = np.linalg.inv(E)
        HE_inv = H @ E_inv
        eigenvalues = np.real(np.linalg.eigvals(HE_inv))
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[eigenvalues > 0]
    except np.linalg.LinAlgError:
        result["summary"] = "MANOVA: Error matrix is singular. Check for collinear responses or insufficient data."
        return result

    s = min(p, k - 1)

    # 1. Wilks' Lambda
    wilks = float(np.linalg.det(E) / np.linalg.det(E + H))

    # Wilks' Lambda → F approximation (Rao's F)
    df_h = p * (k - 1)
    df_e = N - k
    if p == 1 or k == 2:
        F_wilks = ((1 - wilks) / wilks) * (df_e / df_h)
        df1_w, df2_w = df_h, df_e
    elif p == 2 or k == 3:
        wilks_sqrt = np.sqrt(wilks) if wilks > 0 else 0
        r = df_e - (p - k + 2) / 2
        F_wilks = ((1 - wilks_sqrt) / wilks_sqrt) * (r / df_h) if wilks_sqrt > 0 else 0
        df1_w, df2_w = df_h, 2 * (r - 1) if r > 1 else 1
    else:
        # General case: Chi-square approximation
        t = np.sqrt((p**2 * (k - 1) ** 2 - 4) / (p**2 + (k - 1) ** 2 - 5)) if (p**2 + (k - 1) ** 2 - 5) > 0 else 1
        df1_w = p * (k - 1)
        ms = N - 1 - (p + k) / 2
        df2_w = ms * t - df1_w / 2 + 1
        wilks_t = wilks ** (1 / t) if wilks > 0 and t > 0 else 0
        F_wilks = ((1 - wilks_t) / wilks_t) * (df2_w / df1_w) if wilks_t > 0 else 0

    p_wilks = float(1 - stats.f.cdf(max(F_wilks, 0), max(df1_w, 1), max(df2_w, 1)))

    # 2. Pillai's Trace
    pillai = float(np.sum(eigenvalues / (1 + eigenvalues)))
    df1_p = s * max(p, k - 1)
    df2_p = s * (N - k - p + s)
    F_pillai = (pillai / s) * (df2_p / (max(p, k - 1))) / (1 - pillai / s) if (1 - pillai / s) > 0 else 0
    p_pillai = float(1 - stats.f.cdf(max(F_pillai, 0), max(df1_p, 1), max(df2_p, 1)))

    # 3. Hotelling-Lawley Trace
    hl_trace = float(np.sum(eigenvalues))
    df1_hl = s * max(p, k - 1)
    df2_hl = s * (N - k - p - 1) + 2
    F_hl = (hl_trace / s) * (df2_hl / max(p, k - 1)) if max(p, k - 1) > 0 else 0
    p_hl = float(1 - stats.f.cdf(max(F_hl, 0), max(df1_hl, 1), max(df2_hl, 1)))

    # 4. Roy's Largest Root
    roy = float(eigenvalues[0]) if len(eigenvalues) > 0 else 0
    df1_r = max(p, k - 1)
    df2_r = N - k - max(p, k - 1) + 1
    F_roy = roy * df2_r / df1_r if df1_r > 0 else 0
    p_roy = float(1 - stats.f.cdf(max(F_roy, 0), max(df1_r, 1), max(df2_r, 1)))

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>ONE-WAY MANOVA<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Responses:<</COLOR>> {', '.join(responses)}\n"
    summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor} ({k} groups)\n"
    summary += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

    summary += "<<COLOR:accent>>── Group Means ──<</COLOR>>\n"
    header = f"{'Variable':<20}" + "".join(f"{str(g):>12}" for g in groups)
    summary += header + "\n" + "─" * len(header) + "\n"
    for i, var in enumerate(responses):
        row = f"{var:<20}" + "".join(f"{group_means[g]['mean'][i]:>12.4f}" for g in groups)
        summary += row + "\n"
    summary += "\n"

    summary += "<<COLOR:accent>>── MANOVA Test Statistics ──<</COLOR>>\n"
    summary += f"{'Test':<25} {'Value':>10} {'F':>10} {'p-value':>10} {'Sig':>5}\n"
    summary += f"{'─' * 62}\n"

    tests = [
        ("Wilks' Lambda", wilks, F_wilks, p_wilks),
        ("Pillai's Trace", pillai, F_pillai, p_pillai),
        ("Hotelling-Lawley Trace", hl_trace, F_hl, p_hl),
        ("Roy's Largest Root", roy, F_roy, p_roy),
    ]
    for name, val, f_val, p_val in tests:
        sig = "<<COLOR:good>>*<</COLOR>>" if p_val < alpha else ""
        summary += f"{name:<25} {val:>10.4f} {f_val:>10.4f} {p_val:>10.4f} {sig:>5}\n"

    summary += (
        f"\n<<COLOR:accent>>── Eigenvalues of H·E⁻¹ ──<</COLOR>> {', '.join(f'{e:.4f}' for e in eigenvalues)}\n\n"
    )

    # Overall interpretation (use Pillai's — most robust)
    if p_pillai < alpha:
        summary += f"<<COLOR:good>>Significant multivariate effect (Pillai's Trace, p < {alpha})<</COLOR>>\n"
        summary += "<<COLOR:text>>Group means differ across the response variables considered jointly.<</COLOR>>"
    else:
        summary += f"<<COLOR:text>>No significant multivariate effect (p >= {alpha})<</COLOR>>"

    result["summary"] = summary

    # Group centroid plot (first 2 responses, or first 2 discriminant functions)
    if p >= 2:
        traces = []
        colors = [
            "#4a9f6e",
            "#47a5e8",
            "#e89547",
            "#9f4a4a",
            "#6c5ce7",
            "#e84393",
            "#00b894",
            "#fdcb6e",
        ]
        for i, grp in enumerate(groups):
            grp_data = data[data[factor] == grp]
            traces.append(
                {
                    "type": "scatter",
                    "x": grp_data[responses[0]].tolist(),
                    "y": grp_data[responses[1]].tolist(),
                    "mode": "markers",
                    "name": str(grp),
                    "marker": {
                        "color": colors[i % len(colors)],
                        "size": 6,
                        "opacity": 0.6,
                    },
                }
            )
            # Centroid
            traces.append(
                {
                    "type": "scatter",
                    "x": [float(group_means[grp]["mean"][0])],
                    "y": [float(group_means[grp]["mean"][1])],
                    "mode": "markers",
                    "marker": {
                        "color": colors[i % len(colors)],
                        "size": 14,
                        "symbol": "diamond",
                        "line": {"color": "white", "width": 2},
                    },
                    "showlegend": False,
                }
            )
        result["plots"].append(
            {
                "title": f"Group Centroids — {responses[0]} vs {responses[1]}",
                "data": traces,
                "layout": {
                    "height": 350,
                    "xaxis": {"title": responses[0]},
                    "yaxis": {"title": responses[1]},
                },
            }
        )

    # Per-response box plots by group
    box_traces_m = []
    m_colors = [
        "#4a9f6e",
        "#47a5e8",
        "#e89547",
        "#9f4a4a",
        "#6c5ce7",
        "#e84393",
        "#00b894",
        "#fdcb6e",
    ]
    for gi, grp in enumerate(groups):
        grp_d = data[data[factor] == grp]
        for vi, var in enumerate(responses):
            box_traces_m.append(
                {
                    "type": "box",
                    "y": grp_d[var].tolist(),
                    "x": [var] * len(grp_d),
                    "name": str(grp),
                    "marker": {"color": m_colors[gi % len(m_colors)]},
                    "legendgroup": str(grp),
                    "showlegend": vi == 0,
                }
            )
    result["plots"].append(
        {
            "title": "Response Distributions by Group",
            "data": box_traces_m,
            "layout": {
                "height": 300,
                "boxmode": "group",
                "xaxis": {"title": "Response"},
                "yaxis": {"title": "Value"},
            },
        }
    )

    # Correlation heatmap of response variables
    corr_mat = data[responses].corr().values
    result["plots"].append(
        {
            "data": [
                {
                    "z": corr_mat.tolist(),
                    "x": responses,
                    "y": responses,
                    "type": "heatmap",
                    "colorscale": [[0, "#d94a4a"], [0.5, "#f0f4f0"], [1, "#2c5f2d"]],
                    "zmin": -1,
                    "zmax": 1,
                    "text": [[f"{corr_mat[i][j]:.3f}" for j in range(len(responses))] for i in range(len(responses))],
                    "texttemplate": "%{text}",
                    "showscale": True,
                }
            ],
            "layout": {"title": "Response Correlation Matrix", "height": 300},
        }
    )

    result["guide_observation"] = (
        f"MANOVA: Wilks' Λ = {wilks:.4f}, p = {p_wilks:.4f}; Pillai's V = {pillai:.4f}, p = {p_pillai:.4f}. "
        + ("Multivariate effect detected." if p_pillai < alpha else "No multivariate effect.")
    )
    result["statistics"] = {
        "wilks_lambda": wilks,
        "wilks_F": F_wilks,
        "wilks_p": p_wilks,
        "pillai_trace": pillai,
        "pillai_F": F_pillai,
        "pillai_p": p_pillai,
        "hotelling_lawley": hl_trace,
        "hl_F": F_hl,
        "hl_p": p_hl,
        "roys_root": roy,
        "roy_F": F_roy,
        "roy_p": p_roy,
        "eigenvalues": eigenvalues.tolist(),
        "n_groups": k,
        "n_responses": p,
        "N": N,
    }

    # Narrative
    _mv_sig = p_pillai < alpha
    _mv_eta2 = float(pillai)  # Pillai's trace approximates multivariate η²
    _mv_mag = "large" if _mv_eta2 > 0.25 else ("medium" if _mv_eta2 > 0.10 else "small")
    result["narrative"] = _narrative(
        f"MANOVA — {'Significant' if _mv_sig else 'No significant'} multivariate effect (Pillai's V = {pillai:.4f})",
        f"Testing {p} response variables across {k} groups (N = {N}). "
        + (
            f"The factor has a {_mv_mag} multivariate effect (Wilks' Λ = {wilks:.4f}, p = {p_wilks:.4f}). Examine per-response ANOVAs to identify which variables drive the difference."
            if _mv_sig
            else f"No evidence of group differences across the response variables jointly (Wilks' Λ = {wilks:.4f}, p = {p_wilks:.4f})."
        ),
        next_steps=(
            "Run univariate ANOVAs per response with Bonferroni correction to identify which variables differ."
            if _mv_sig
            else "Check individual ANOVAs — marginal effects may exist that the joint test misses."
        ),
        chart_guidance="The scatter plot shows group separation in the first two response dimensions. Non-overlapping clusters confirm a multivariate effect.",
    )

    return result


def run_nested_anova(df, config):
    """
    Nested (Hierarchical) ANOVA — random effects model.
    Tests fixed factor effect while accounting for a random nesting factor.
    E.g., operators nested within machines, batches within suppliers.
    Uses linear mixed-effects model (statsmodels mixedlm).
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    response = config.get("response") or config.get("var")
    fixed_factor = config.get("fixed_factor") or config.get("factor")
    random_factor = config.get("random_factor") or config.get("group_var")
    alpha = 1 - config.get("conf", 95) / 100

    try:
        from statsmodels.formula.api import mixedlm

        data = df[[response, fixed_factor, random_factor]].dropna()
        N = len(data)

        # Fit mixed model: response ~ fixed_factor with random intercept for random_factor
        formula = f"{response} ~ C({fixed_factor})"
        model = mixedlm(formula, data, groups=data[random_factor])
        fit = model.fit(reml=True)

        # Extract results
        fixed_effects = {}
        for name, val in fit.fe_params.items():
            pval = float(fit.pvalues[name]) if name in fit.pvalues else None
            se = float(fit.bse[name]) if name in fit.bse else None
            fixed_effects[name] = {"coef": float(val), "se": se, "p_value": pval}

        # Variance components
        var_random = float(fit.cov_re.iloc[0, 0]) if hasattr(fit.cov_re, "iloc") else float(fit.cov_re)
        var_residual = float(fit.scale)
        var_total = var_random + var_residual
        icc = var_random / var_total if var_total > 0 else 0

        # Group stats
        fixed_levels = sorted(data[fixed_factor].unique().tolist(), key=str)
        random_levels = sorted(data[random_factor].unique().tolist(), key=str)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>NESTED ANOVA (Mixed-Effects Model)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Fixed factor:<</COLOR>> {fixed_factor} ({len(fixed_levels)} levels)\n"
        summary += (
            f"<<COLOR:highlight>>Random factor (nesting):<</COLOR>> {random_factor} ({len(random_levels)} levels)\n"
        )
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

        summary += "<<COLOR:accent>>── Fixed Effects ──<</COLOR>>\n"
        summary += f"{'Term':<30} {'Coef':>8} {'SE':>8} {'p-value':>8} {'Sig':>5}\n"
        summary += f"{'─' * 62}\n"
        for name, fe in fixed_effects.items():
            sig = "<<COLOR:good>>*<</COLOR>>" if fe["p_value"] is not None and fe["p_value"] < alpha else ""
            p_str = f"{fe['p_value']:.4f}" if fe["p_value"] is not None else "N/A"
            se_str = f"{fe['se']:.4f}" if fe["se"] is not None else "N/A"
            summary += f"{name:<30} {fe['coef']:>8.4f} {se_str:>8} {p_str:>8} {sig:>5}\n"

        summary += "\n<<COLOR:accent>>── Variance Components ──<</COLOR>>\n"
        summary += f"  {random_factor} (random): {var_random:.4f} ({icc * 100:.1f}% of total)\n"
        summary += f"  Residual: {var_residual:.4f} ({(1 - icc) * 100:.1f}% of total)\n"
        summary += f"  Total: {var_total:.4f}\n"
        summary += f"  ICC (Intraclass Correlation): {icc:.4f}\n\n"

        if icc > 0.1:
            summary += (
                f"<<COLOR:good>>ICC = {icc:.3f} — substantial variation attributed to {random_factor}.<</COLOR>>\n"
            )
            summary += f"<<COLOR:text>>The nesting structure accounts for {icc * 100:.1f}% of the variance. Ignoring it would inflate Type I error.<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>ICC = {icc:.3f} — low variation from {random_factor}. A standard ANOVA may suffice.<</COLOR>>\n"

        # Check if fixed factor is significant
        sig_fixed = any(
            fe["p_value"] is not None and fe["p_value"] < alpha
            for name, fe in fixed_effects.items()
            if name != "Intercept"
        )
        if sig_fixed:
            summary += f"<<COLOR:good>>Fixed factor {fixed_factor} has significant effect.<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>Fixed factor {fixed_factor} not significant after accounting for {random_factor}.<</COLOR>>"

        result["summary"] = summary

        # Box plot with nesting structure
        traces = []
        for i, fl in enumerate(fixed_levels):
            subset = data[data[fixed_factor] == fl]
            traces.append(
                {
                    "type": "box",
                    "y": subset[response].tolist(),
                    "x": [str(fl)] * len(subset),
                    "name": str(fl),
                    "boxpoints": "all",
                    "jitter": 0.3,
                    "pointpos": 0,
                    "marker": {"size": 4, "opacity": 0.5},
                }
            )

        result["plots"].append(
            {
                "title": f"Nested ANOVA: {response} by {fixed_factor} (nested in {random_factor})",
                "data": traces,
                "layout": {
                    "height": 300,
                    "yaxis": {"title": response},
                    "xaxis": {"title": fixed_factor},
                },
            }
        )

        # Residuals vs fitted values
        fitted_vals = fit.fittedvalues
        resid_vals = fit.resid
        result["plots"].append(
            {
                "title": "Residuals vs Fitted Values",
                "data": [
                    {
                        "x": fitted_vals.tolist(),
                        "y": resid_vals.tolist(),
                        "mode": "markers",
                        "type": "scatter",
                        "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.6},
                    }
                ],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": "Fitted Values"},
                    "yaxis": {"title": "Residuals"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": float(fitted_vals.min()),
                            "x1": float(fitted_vals.max()),
                            "y0": 0,
                            "y1": 0,
                            "line": {"color": "#e89547", "dash": "dash"},
                        }
                    ],
                },
            }
        )

        # Normal Q-Q plot of residuals
        from scipy import stats as qstats

        sorted_resid = np.sort(resid_vals.values)
        n_qq = len(sorted_resid)
        theoretical_q = [float(qstats.norm.ppf((i + 0.5) / n_qq)) for i in range(n_qq)]
        result["plots"].append(
            {
                "title": "Normal Q-Q Plot of Residuals",
                "data": [
                    {
                        "x": theoretical_q,
                        "y": sorted_resid.tolist(),
                        "mode": "markers",
                        "type": "scatter",
                        "marker": {"color": "#4a9f6e", "size": 4},
                        "name": "Residuals",
                    },
                    {
                        "x": [theoretical_q[0], theoretical_q[-1]],
                        "y": [
                            theoretical_q[0] * np.std(sorted_resid) + np.mean(sorted_resid),
                            theoretical_q[-1] * np.std(sorted_resid) + np.mean(sorted_resid),
                        ],
                        "mode": "lines",
                        "line": {"color": "#e89547", "dash": "dash"},
                        "name": "Reference",
                    },
                ],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": "Theoretical Quantiles"},
                    "yaxis": {"title": "Sample Quantiles"},
                },
            }
        )

        result["guide_observation"] = (
            f"Nested ANOVA: ICC = {icc:.3f} ({icc * 100:.1f}% variance from {random_factor}). "
            + ("Fixed effect significant." if sig_fixed else "Fixed effect not significant.")
        )
        if sig_fixed:
            result["narrative"] = _narrative(
                f"Nested ANOVA: fixed effect significant, ICC = {icc:.3f}",
                f"{icc * 100:.1f}% of variation comes from <strong>{random_factor}</strong>. The fixed effect significantly affects the response.",
                next_steps="The ICC tells you how much of the variation is between groups vs within. High ICC = groups are very different.",
            )
        else:
            result["narrative"] = _narrative(
                f"Nested ANOVA: no significant fixed effect (ICC = {icc:.3f})",
                f"{icc * 100:.1f}% of variation comes from {random_factor}. The fixed effect is not significant.",
                next_steps="Low ICC and non-significant fixed effect suggests most variation is within groups.",
            )
        result["statistics"] = {
            "fixed_effects": fixed_effects,
            "var_random": var_random,
            "var_residual": var_residual,
            "icc": icc,
            "aic": float(fit.aic) if hasattr(fit, "aic") else None,
            "bic": float(fit.bic) if hasattr(fit, "bic") else None,
            "converged": fit.converged if hasattr(fit, "converged") else True,
        }

    except ImportError:
        result["summary"] = "Nested ANOVA requires statsmodels. Install with: pip install statsmodels"
    except Exception as e:
        result["summary"] = f"Nested ANOVA error: {str(e)}"

    return result


def run_manova_v2(df, config):
    """
    Multivariate ANOVA (second implementation) — tests group differences across multiple response variables.
    Uses Pillai's trace, Wilks' lambda, Hotelling-Lawley, Roy's greatest root.
    Includes per-response univariate ANOVAs.
    """
    result = {"plots": [], "summary": "", "guide_observation": ""}

    responses = config.get("responses", [])
    factor = config.get("factor") or config.get("group_var") or config.get("group")
    alpha = 1 - config.get("conf", 95) / 100

    if not responses and config.get("response"):
        responses = [config["response"]]

    try:
        all_cols = responses + [factor]
        data = df[all_cols].dropna()
        N = len(data)
        groups = sorted(data[factor].unique().tolist(), key=str)
        k = len(groups)
        p = len(responses)

        # Compute group means and overall mean
        overall_mean = data[responses].mean().values
        group_data = {g: data[data[factor] == g][responses].values for g in groups}
        group_means = {g: v.mean(axis=0) for g, v in group_data.items()}
        group_ns = {g: len(v) for g, v in group_data.items()}

        # Between-group SSCP matrix (H)
        H = np.zeros((p, p))
        for g in groups:
            diff = (group_means[g] - overall_mean).reshape(-1, 1)
            H += group_ns[g] * diff @ diff.T

        # Within-group SSCP matrix (E)
        E = np.zeros((p, p))
        for g in groups:
            centered = group_data[g] - group_means[g]
            E += centered.T @ centered

        # Test statistics
        df_h = k - 1
        df_e = N - k

        # Eigenvalues of E^-1 H
        try:
            E_inv = np.linalg.inv(E)
            eigvals = np.real(np.linalg.eigvals(E_inv @ H))
            eigvals = np.sort(eigvals)[::-1]
        except np.linalg.LinAlgError:
            eigvals = np.array([0.0] * p)

        # Pillai's trace
        pillai = np.sum(eigvals / (1 + eigvals))

        # Wilks' lambda
        wilks = np.prod(1 / (1 + eigvals))

        # Hotelling-Lawley trace
        hotelling = np.sum(eigvals)

        # Roy's greatest root
        roy = eigvals[0] if len(eigvals) > 0 else 0

        # Approximate F-test for Wilks' lambda
        s = min(p, df_h)
        _m = (abs(p - df_h) - 1) / 2  # noqa: F841
        (df_e - p - 1) / 2
        if s > 0 and wilks > 0:
            r = df_e - (p - df_h + 1) / 2
            u = (p * df_h - 2) / 4
            if p**2 + df_h**2 - 5 > 0:
                t = np.sqrt((p**2 * df_h**2 - 4) / (p**2 + df_h**2 - 5))
            else:
                t = 1
            df1 = p * df_h
            df2 = r * t - 2 * u
            if df2 > 0:
                f_wilks = ((1 - wilks ** (1 / t)) / (wilks ** (1 / t))) * (df2 / df1)
                from scipy import stats as fstats

                p_wilks = 1 - fstats.f.cdf(f_wilks, df1, df2)
            else:
                f_wilks = None
                p_wilks = None
        else:
            f_wilks = None
            p_wilks = None

        summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary_text += "<<COLOR:title>>MULTIVARIATE ANALYSIS OF VARIANCE (MANOVA)<</COLOR>>\n"
        summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary_text += f"<<COLOR:highlight>>Responses:<</COLOR>> {', '.join(responses)}\n"
        summary_text += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor} ({k} groups)\n"
        summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

        summary_text += "<<COLOR:accent>>── Multivariate Test Statistics ──<</COLOR>>\n"
        summary_text += f"{'Test':<25} {'Value':>10} {'Approx F':>10} {'p-value':>10}\n"
        summary_text += f"{'─' * 57}\n"
        pillai_label = "Pillai's Trace"
        summary_text += f"{pillai_label:<25} {pillai:>10.4f} {'':>10} {'':>10}\n"
        wilks_f_str = f"{f_wilks:.4f}" if f_wilks is not None else "N/A"
        wilks_p_str = f"{p_wilks:.4f}" if p_wilks is not None else "N/A"
        wilks_label = "Wilks' Lambda"
        summary_text += f"{wilks_label:<25} {wilks:>10.4f} {wilks_f_str:>10} {wilks_p_str:>10}\n"
        summary_text += f"{'Hotelling-Lawley':<25} {hotelling:>10.4f} {'':>10} {'':>10}\n"
        roy_label = "Roy's Greatest Root"
        summary_text += f"{roy_label:<25} {roy:>10.4f} {'':>10} {'':>10}\n\n"

        # Univariate ANOVAs
        summary_text += "<<COLOR:accent>>── Univariate ANOVA per Response ──<</COLOR>>\n"
        summary_text += f"{'Response':<20} {'F':>10} {'p-value':>10} {'Sig':>5}\n"
        summary_text += f"{'─' * 47}\n"
        from scipy import stats as fstats

        for resp in responses:
            group_vals = [data[data[factor] == g][resp].values for g in groups]
            f_stat, p_val = fstats.f_oneway(*group_vals)
            sig = "<<COLOR:good>>*<</COLOR>>" if p_val < alpha else ""
            summary_text += f"{resp:<20} {f_stat:>10.4f} {p_val:>10.4f} {sig:>5}\n"

        if p_wilks is not None and p_wilks < alpha:
            summary_text += f"\n<<COLOR:good>>Significant multivariate effect (Wilks' Λ = {wilks:.4f}, p = {p_wilks:.4f}).<</COLOR>>"
        elif p_wilks is not None:
            summary_text += f"\n<<COLOR:text>>No significant multivariate effect (Wilks' Λ = {wilks:.4f}, p = {p_wilks:.4f}).<</COLOR>>"

        result["summary"] = summary_text

        # Mean profiles plot
        for resp in responses:
            means = [float(data[data[factor] == g][resp].mean()) for g in groups]
            sds = [float(data[data[factor] == g][resp].std()) for g in groups]
            result["plots"].append(
                {
                    "title": f"Group Means: {resp} by {factor}",
                    "data": [
                        {
                            "x": [str(g) for g in groups],
                            "y": means,
                            "error_y": {"type": "data", "array": sds, "visible": True},
                            "type": "bar",
                            "marker": {"color": "#4a90d9"},
                        }
                    ],
                    "layout": {
                        "height": 250,
                        "yaxis": {"title": resp},
                        "xaxis": {"title": factor},
                    },
                }
            )

        result["statistics"] = {
            "pillai": float(pillai),
            "wilks_lambda": float(wilks),
            "hotelling_lawley": float(hotelling),
            "roys_greatest_root": float(roy),
            "f_wilks": float(f_wilks) if f_wilks else None,
            "p_wilks": float(p_wilks) if p_wilks else None,
            "n_groups": k,
            "n_responses": p,
            "n": N,
        }
        result["guide_observation"] = (
            f"MANOVA: {', '.join(responses)} by {factor}. Wilks' Λ={wilks:.4f}"
            + (f", p={p_wilks:.4f}" if p_wilks else "")
            + "."
        )

        # Narrative
        try:
            _mv2_sig = p_wilks is not None and p_wilks < alpha
            result["narrative"] = _narrative(
                f"MANOVA — {'Significant' if _mv2_sig else 'No significant'} multivariate effect",
                f"Testing {', '.join(responses)} jointly by {factor} ({k} groups, N = {N}). "
                + (f"Wilks' Λ = {wilks:.4f}" + (f" (p = {p_wilks:.4f})" if p_wilks else "") + ". ")
                + (
                    "The factor significantly affects the responses jointly."
                    if _mv2_sig
                    else "No evidence of a joint multivariate effect."
                ),
                next_steps=(
                    "Examine the per-response bar charts to see which variables drive the group separation."
                    if _mv2_sig
                    else None
                ),
                chart_guidance="Bar charts show group means ± SD for each response. Large non-overlapping error bars suggest meaningful differences.",
            )
        except Exception:
            pass

    except Exception as e:
        result["summary"] = f"MANOVA error: {str(e)}"

    return result

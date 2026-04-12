"""DSW Exploratory — meta-analysis and effect size analyses."""

import logging

import numpy as np
from scipy import stats as sp_stats

from ..common import (
    _narrative,
)

logger = logging.getLogger(__name__)


def run_meta_analysis(df, config):
    """Meta-analysis — fixed and random effects models."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    try:
        mode = config.get("mode", "precomputed")
        config.get("subgroup_col", "")

        if mode == "precomputed":
            effect_col = config.get("effect_col", "")
            se_col = config.get("se_col", "")
            study_col = config.get("study_col", "")
            if not all([effect_col, se_col, study_col]):
                result["summary"] = "Please specify effect size, SE, and study label columns."
                return result
            effects = df[effect_col].dropna().values.astype(float)
            ses = df[se_col].dropna().values.astype(float)
            studies = df[study_col].values[: len(effects)]
        else:
            # Raw mode: compute Cohen's d
            m1c, s1c, n1c = (
                config.get("mean1_col", ""),
                config.get("sd1_col", ""),
                config.get("n1_col", ""),
            )
            m2c, s2c, n2c = (
                config.get("mean2_col", ""),
                config.get("sd2_col", ""),
                config.get("n2_col", ""),
            )
            study_col = config.get("study_col", "")
            if not all([m1c, s1c, n1c, m2c, s2c, n2c]):
                result["summary"] = "Please specify all 6 raw data columns (mean, SD, n for each group)."
                return result
            m1 = df[m1c].values.astype(float)
            s1 = df[s1c].values.astype(float)
            n1 = df[n1c].values.astype(float)
            m2 = df[m2c].values.astype(float)
            s2 = df[s2c].values.astype(float)
            n2 = df[n2c].values.astype(float)
            sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            effects = (m1 - m2) / sp
            ses = np.sqrt(1 / n1 + 1 / n2 + effects**2 / (2 * (n1 + n2)))
            studies = df[study_col].values[: len(effects)] if study_col else np.arange(1, len(effects) + 1)

        k = len(effects)
        if k < 2:
            result["summary"] = "Need at least 2 studies for meta-analysis."
            return result

        # Weights
        w = 1.0 / (ses**2)

        # Fixed effects
        fe_est = float(np.sum(w * effects) / np.sum(w))
        fe_se = float(np.sqrt(1.0 / np.sum(w)))
        fe_ci_lo = fe_est - 1.96 * fe_se
        fe_ci_hi = fe_est + 1.96 * fe_se
        fe_z = fe_est / fe_se
        from scipy.stats import norm as norm_dist

        fe_p = float(2 * (1 - norm_dist.cdf(abs(fe_z))))

        # Heterogeneity
        Q = float(np.sum(w * (effects - fe_est) ** 2))
        df_q = k - 1
        from scipy.stats import chi2 as chi2_dist_fn

        Q_p = float(1 - chi2_dist_fn.cdf(Q, df_q))
        I2 = max(0, (Q - df_q) / Q * 100) if Q > 0 else 0
        H2 = Q / df_q if df_q > 0 else 1

        # DerSimonian-Laird tau²
        C = float(np.sum(w) - np.sum(w**2) / np.sum(w))
        tau2 = max(0, (Q - df_q) / C) if C > 0 else 0

        # Random effects
        w_re = 1.0 / (ses**2 + tau2)
        re_est = float(np.sum(w_re * effects) / np.sum(w_re))
        re_se = float(np.sqrt(1.0 / np.sum(w_re)))
        re_ci_lo = re_est - 1.96 * re_se
        re_ci_hi = re_est + 1.96 * re_se
        re_z = re_est / re_se
        re_p = float(2 * (1 - norm_dist.cdf(abs(re_z))))

        interpretation = (
            "Low heterogeneity" if I2 < 25 else ("Moderate heterogeneity" if I2 < 75 else "High heterogeneity")
        )
        if I2 < 40:
            interpretation += " — fixed and random effects models give similar results."
        else:
            interpretation += " — random effects model is more appropriate."

        summary = f"""META-ANALYSIS
{"=" * 50}
Studies: {k}

Fixed Effects Model:
  Pooled estimate: {fe_est:.4f} (95% CI: {fe_ci_lo:.4f}, {fe_ci_hi:.4f})
  z = {fe_z:.3f}, p = {fe_p:.4f}

Random Effects Model (DerSimonian-Laird):
  Pooled estimate: {re_est:.4f} (95% CI: {re_ci_lo:.4f}, {re_ci_hi:.4f})
  z = {re_z:.3f}, p = {re_p:.4f}

Heterogeneity:
  Q = {Q:.2f} (df={df_q}, p={Q_p:.4f})
  I² = {I2:.1f}% (variation due to heterogeneity)
  tau² = {tau2:.4f} (between-study variance)
  H² = {H2:.2f}

Interpretation:
  {interpretation}"""

        result["summary"] = summary

        # Table
        table_html = "<table class='result-table'><tr><th>Study</th><th>Effect</th><th>SE</th><th>95% CI</th><th>Weight (FE)</th><th>Weight (RE)</th></tr>"
        for i in range(k):
            ci_lo = effects[i] - 1.96 * ses[i]
            ci_hi = effects[i] + 1.96 * ses[i]
            w_fe_pct = w[i] / np.sum(w) * 100
            w_re_pct = w_re[i] / np.sum(w_re) * 100
            table_html += f"<tr><td>{studies[i]}</td><td>{effects[i]:.4f}</td><td>{ses[i]:.4f}</td>"
            table_html += f"<td>[{ci_lo:.4f}, {ci_hi:.4f}]</td><td>{w_fe_pct:.1f}%</td><td>{w_re_pct:.1f}%</td></tr>"
        table_html += f"<tr style='font-weight:bold;border-top:2px solid #666;'><td>Fixed Effects</td><td>{fe_est:.4f}</td><td>{fe_se:.4f}</td><td>[{fe_ci_lo:.4f}, {fe_ci_hi:.4f}]</td><td>100%</td><td>-</td></tr>"
        table_html += f"<tr style='font-weight:bold;'><td>Random Effects</td><td>{re_est:.4f}</td><td>{re_se:.4f}</td><td>[{re_ci_lo:.4f}, {re_ci_hi:.4f}]</td><td>-</td><td>100%</td></tr>"
        table_html += "</table>"
        result["tables"] = [{"title": "Study Results", "html": table_html}]

        # Forest plot
        forest_y = list(range(k, 0, -1))
        ci_lows = [effects[i] - 1.96 * ses[i] for i in range(k)]
        ci_highs = [effects[i] + 1.96 * ses[i] for i in range(k)]
        study_labels = [str(s) for s in studies]

        forest_data = [
            # Study CIs (horizontal lines)
            {
                "type": "scatter",
                "x": list(effects),
                "y": forest_y,
                "mode": "markers",
                "name": "Studies",
                "marker": {
                    "size": [max(4, min(16, float(w_re[i] / np.max(w_re) * 16))) for i in range(k)],
                    "color": "rgba(74,144,217,0.8)",
                },
                "error_x": {
                    "type": "data",
                    "symmetric": False,
                    "array": [ci_highs[i] - effects[i] for i in range(k)],
                    "arrayminus": [effects[i] - ci_lows[i] for i in range(k)],
                    "color": "rgba(74,144,217,0.5)",
                    "thickness": 2,
                },
                "text": study_labels,
                "hovertemplate": "%{text}: %{x:.4f}<extra></extra>",
            },
            # Pooled diamond (RE)
            {
                "type": "scatter",
                "x": [re_ci_lo, re_est, re_ci_hi, re_est, re_ci_lo],
                "y": [0, -0.3, 0, 0.3, 0],
                "mode": "lines",
                "fill": "toself",
                "fillcolor": "rgba(232,71,71,0.3)",
                "line": {"color": "rgba(232,71,71,0.8)"},
                "name": "Pooled (RE)",
                "hoverinfo": "skip",
            },
            # Zero line
            {
                "type": "scatter",
                "x": [0, 0],
                "y": [-1, k + 1],
                "mode": "lines",
                "line": {"color": "gray", "dash": "dash", "width": 1},
                "showlegend": False,
                "hoverinfo": "skip",
            },
        ]
        result["plots"].append(
            {
                "data": forest_data,
                "layout": {
                    "title": "Forest Plot",
                    "height": max(350, k * 30 + 120),
                    "yaxis": {
                        "tickvals": forest_y + [0],
                        "ticktext": study_labels + ["Pooled (RE)"],
                        "range": [-1, k + 1],
                    },
                    "xaxis": {"title": "Effect Size", "zeroline": True},
                    "showlegend": False,
                    "margin": {"l": 120},
                },
            }
        )

        # Funnel plot
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "scatter",
                        "x": list(effects),
                        "y": list(ses),
                        "mode": "markers",
                        "marker": {"size": 8, "color": "rgba(74,144,217,0.7)"},
                        "name": "Studies",
                        "text": study_labels,
                        "hovertemplate": "%{text}: effect=%{x:.4f}, SE=%{y:.4f}<extra></extra>",
                    },
                    # Funnel lines
                    {
                        "type": "scatter",
                        "x": [
                            re_est - 1.96 * max(ses),
                            re_est,
                            re_est + 1.96 * max(ses),
                        ],
                        "y": [max(ses), 0, max(ses)],
                        "mode": "lines",
                        "line": {"color": "gray", "dash": "dash"},
                        "name": "95% CI",
                        "hoverinfo": "skip",
                    },
                    {
                        "type": "scatter",
                        "x": [re_est, re_est],
                        "y": [0, max(ses) * 1.1],
                        "mode": "lines",
                        "line": {"color": "rgba(232,71,71,0.5)", "dash": "dot"},
                        "name": "Pooled",
                        "hoverinfo": "skip",
                    },
                ],
                "layout": {
                    "title": "Funnel Plot",
                    "height": 350,
                    "xaxis": {"title": "Effect Size"},
                    "yaxis": {"title": "Standard Error", "autorange": "reversed"},
                },
            }
        )

        result["guide_observation"] = (
            f"Meta-analysis of {k} studies. RE pooled: {re_est:.4f}. I²={I2:.1f}%. {interpretation}"
        )

        # Narrative
        _ma_sig = "statistically significant" if re_p < 0.05 else "not statistically significant"
        _ma_het = "low" if I2 < 25 else ("moderate" if I2 < 75 else "high")
        _ma_model = (
            "Fixed effects may suffice." if I2 < 25 else "Random effects model is preferred due to heterogeneity."
        )
        result["narrative"] = _narrative(
            f"Meta-Analysis — {k} studies, pooled effect = {re_est:.4f} (RE)",
            f"The random-effects pooled estimate is {re_est:.4f} (95% CI: {re_ci_lo:.4f} to {re_ci_hi:.4f}), {_ma_sig} (p = {re_p:.4f}). "
            f"Heterogeneity is {_ma_het} (I² = {I2:.1f}%, τ² = {tau2:.4f}). {_ma_model}",
            next_steps="Inspect the funnel plot for asymmetry (publication bias). Consider subgroup analysis if I² is high.",
            chart_guidance="In the forest plot, study size is proportional to marker area. The red diamond is the pooled estimate. The funnel plot should be symmetric if no publication bias exists.",
        )

    except Exception as e:
        result["summary"] = f"Meta-analysis error: {str(e)}"

    return result


def run_effect_size_calculator(df, config):
    """Effect size calculator — Cohen's d, Hedges' g, Glass's delta, odds/risk ratio."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    try:
        effect_type = config.get("effect_type", "cohens_d")
        results_list = []  # noqa: F841

        if effect_type in ("cohens_d", "hedges_g", "glass_delta"):
            m1 = float(config.get("mean1", 0))
            s1 = float(config.get("sd1", 1))
            n1 = int(config.get("n1", 10))
            m2 = float(config.get("mean2", 0))
            s2 = float(config.get("sd2", 1))
            n2 = int(config.get("n2", 10))

            sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

            if effect_type == "cohens_d":
                d = (m1 - m2) / sp if sp > 0 else 0
                se = np.sqrt(1 / n1 + 1 / n2 + d**2 / (2 * (n1 + n2)))
                name = "Cohen's d"
            elif effect_type == "hedges_g":
                d_raw = (m1 - m2) / sp if sp > 0 else 0
                J = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
                d = d_raw * J
                se = np.sqrt(1 / n1 + 1 / n2 + d**2 / (2 * (n1 + n2))) * J
                name = "Hedges' g"
            else:  # glass_delta
                d = (m1 - m2) / s2 if s2 > 0 else 0
                se = np.sqrt(1 / n1 + d**2 / (2 * n2))
                name = "Glass's delta"

            ci_lo = d - 1.96 * se
            ci_hi = d + 1.96 * se

            if abs(d) < 0.2:
                magnitude = "Negligible"
            elif abs(d) < 0.5:
                magnitude = "Small"
            elif abs(d) < 0.8:
                magnitude = "Medium"
            else:
                magnitude = "Large"

            summary = f"""EFFECT SIZE CALCULATOR
{"=" * 50}
Type: {name}

Input:
  Group 1: M={m1}, SD={s1}, n={n1}
  Group 2: M={m2}, SD={s2}, n={n2}

Result:
  {name} = {d:.4f}
  Standard Error = {se:.4f}
  95% CI: ({ci_lo:.4f}, {ci_hi:.4f})

Interpretation:
  Magnitude: {magnitude} (Cohen's benchmarks: 0.2=small, 0.5=medium, 0.8=large)
  Direction: {"Group 1 > Group 2" if d > 0 else "Group 2 > Group 1" if d < 0 else "No difference"}"""

            # Plot
            result["plots"].append(
                {
                    "data": [
                        {
                            "type": "scatter",
                            "x": [d],
                            "y": [name],
                            "mode": "markers",
                            "marker": {"size": 14, "color": "rgba(74,144,217,0.8)"},
                            "error_x": {
                                "type": "data",
                                "symmetric": False,
                                "array": [ci_hi - d],
                                "arrayminus": [d - ci_lo],
                                "color": "rgba(74,144,217,0.5)",
                                "thickness": 3,
                            },
                        },
                        {
                            "type": "scatter",
                            "x": [0, 0],
                            "y": [-0.5, 1.5],
                            "mode": "lines",
                            "line": {"color": "gray", "dash": "dash"},
                            "showlegend": False,
                        },
                    ],
                    "layout": {
                        "title": f"{name} with 95% CI",
                        "height": 200,
                        "xaxis": {"title": "Effect Size"},
                        "showlegend": False,
                    },
                }
            )

        elif effect_type in ("odds_ratio", "risk_ratio"):
            a = int(config.get("a", 0))
            b = int(config.get("b", 0))
            c = int(config.get("c", 0))
            dd = int(config.get("d", 0))

            if effect_type == "odds_ratio":
                if b * c > 0:
                    es = (a * dd) / (b * c)
                    se_ln = np.sqrt(1 / max(a, 0.5) + 1 / max(b, 0.5) + 1 / max(c, 0.5) + 1 / max(dd, 0.5))
                else:
                    es, se_ln = 0, 0
                name = "Odds Ratio"
            else:
                r1 = a / (a + b) if (a + b) > 0 else 0
                r2 = c / (c + dd) if (c + dd) > 0 else 0
                es = r1 / r2 if r2 > 0 else 0
                se_ln = np.sqrt(1 / max(a, 0.5) - 1 / max(a + b, 1) + 1 / max(c, 0.5) - 1 / max(c + dd, 1))
                name = "Risk Ratio"

            ci_lo = es * np.exp(-1.96 * se_ln) if es > 0 else 0
            ci_hi = es * np.exp(1.96 * se_ln) if es > 0 else 0

            summary = f"""EFFECT SIZE CALCULATOR
{"=" * 50}
Type: {name}

Input (2x2 table):
               Outcome+  Outcome-
  Exposed      {a:<9} {b}
  Not Exposed  {c:<9} {dd}

Result:
  {name} = {es:.4f}
  95% CI: ({ci_lo:.4f}, {ci_hi:.4f})
  ln({name}) SE = {se_ln:.4f}

Interpretation:
  {"No association (= 1.0)" if abs(es - 1.0) < 0.01 else ("Positive association" if es > 1 else "Negative association")}"""

            result["plots"].append(
                {
                    "data": [
                        {
                            "type": "scatter",
                            "x": [es],
                            "y": [name],
                            "mode": "markers",
                            "marker": {"size": 14, "color": "rgba(232,149,71,0.8)"},
                            "error_x": {
                                "type": "data",
                                "symmetric": False,
                                "array": [ci_hi - es],
                                "arrayminus": [es - ci_lo],
                                "color": "rgba(232,149,71,0.5)",
                                "thickness": 3,
                            },
                        },
                        {
                            "type": "scatter",
                            "x": [1, 1],
                            "y": [-0.5, 1.5],
                            "mode": "lines",
                            "line": {"color": "gray", "dash": "dash"},
                            "showlegend": False,
                        },
                    ],
                    "layout": {
                        "title": f"{name} with 95% CI",
                        "height": 200,
                        "xaxis": {"title": name},
                        "showlegend": False,
                    },
                }
            )
        else:
            summary = f"Unknown effect size type: {effect_type}"

        result["summary"] = summary
        result["guide_observation"] = f"Effect size calculated: {effect_type}"

        # Narrative
        try:
            if effect_type in ("cohens_d", "hedges_g", "glass_delta"):
                result["narrative"] = _narrative(
                    f"{name} = {d:.4f} — {magnitude} effect",
                    f"The standardized difference between groups is {d:.4f} (95% CI: {ci_lo:.4f} to {ci_hi:.4f}). "
                    f"By Cohen's benchmarks this is a <strong>{magnitude.lower()}</strong> effect. "
                    + (
                        "The CI excludes zero, confirming a meaningful difference."
                        if (ci_lo > 0 or ci_hi < 0)
                        else "The CI includes zero — the difference may not be meaningful."
                    ),
                    next_steps="Report this alongside the p-value for a complete picture of both statistical and practical significance.",
                )
            elif effect_type in ("odds_ratio", "risk_ratio"):
                result["narrative"] = _narrative(
                    f"{name} = {es:.4f}",
                    f"The {name.lower()} is {es:.4f} (95% CI: {ci_lo:.4f} to {ci_hi:.4f}). "
                    + (
                        "The CI excludes 1.0, indicating a significant association."
                        if (ci_lo > 1 or ci_hi < 1)
                        else "The CI includes 1.0 — the association is not statistically significant."
                    ),
                    next_steps="Consider confounding variables. An adjusted analysis (logistic regression) may be more appropriate.",
                )
        except Exception:
            pass

    except Exception as e:
        result["summary"] = f"Effect size error: {str(e)}"

    return result


def run_copula(df, config):
    """Copula-based dependency modeling."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    var1 = config.get("var1") or config.get("variable1")
    var2 = config.get("var2") or config.get("variable2")

    if not var1 or not var2 or var1 not in df.columns or var2 not in df.columns:
        result["summary"] = "Error: Specify two numeric columns."
        return result

    x = df[var1].dropna()
    y = df[var2].loc[x.index].dropna()
    x = x.loc[y.index].values.astype(float)
    y = y.values.astype(float)
    n = len(x)

    if n < 20:
        result["summary"] = "Error: Need at least 20 paired observations."
        return result

    # Transform to pseudo-observations (empirical CDF / ranks)
    from scipy.stats import rankdata

    u = rankdata(x) / (n + 1)
    v = rankdata(y) / (n + 1)

    # Pearson and Spearman for reference
    pearson_r, pearson_p = sp_stats.pearsonr(x, y)
    spearman_r, spearman_p = sp_stats.spearmanr(x, y)
    kendall_tau, kendall_p = sp_stats.kendalltau(x, y)

    # Fit copulas via maximum pseudo-likelihood
    # 1. Gaussian copula: parameter = correlation on normal quantiles
    z_u = sp_stats.norm.ppf(np.clip(u, 0.001, 0.999))
    z_v = sp_stats.norm.ppf(np.clip(v, 0.001, 0.999))
    rho_gauss = float(np.corrcoef(z_u, z_v)[0, 1])
    # Log-likelihood for Gaussian copula
    ll_gauss = 0.0
    for i in range(n):
        rho2 = rho_gauss**2
        if rho2 < 1:
            ll_gauss += (
                -0.5 * np.log(1 - rho2)
                + rho2 * (z_u[i] ** 2 + z_v[i] ** 2) / (2 * (1 - rho2))
                - rho_gauss * z_u[i] * z_v[i] / (1 - rho2)
                + rho_gauss * z_u[i] * z_v[i] / (1 - rho2)
            )
    # Simplified: use the bivariate normal copula density
    ll_gauss = float(
        np.sum(
            sp_stats.multivariate_normal.logpdf(
                np.column_stack([z_u, z_v]),
                mean=[0, 0],
                cov=[[1, rho_gauss], [rho_gauss, 1]],
            )
            - sp_stats.norm.logpdf(z_u)
            - sp_stats.norm.logpdf(z_v)
        )
    )

    # 2. Clayton copula: theta > 0 for positive dependence
    # Kendall's tau = theta / (theta + 2)
    if kendall_tau > 0.01:
        theta_clayton = max(2 * kendall_tau / (1 - kendall_tau), 0.01)
    else:
        theta_clayton = 0.01
    # Clayton copula density: c(u,v) = (1+theta) * (u*v)^(-1-theta) * (u^-theta + v^-theta - 1)^(-1/theta - 2)
    try:
        t = theta_clayton
        term = np.clip(u ** (-t) + v ** (-t) - 1, 1e-15, None)
        ll_clayton = float(np.sum(np.log(1 + t) + (-1 - t) * (np.log(u) + np.log(v)) + (-1 / t - 2) * np.log(term)))
    except Exception:
        ll_clayton = -np.inf

    # 3. Frank copula: use Kendall's tau to estimate theta
    # tau = 1 - 4/theta * (1 - D1(theta)/theta) where D1 is Debye function
    # Approximate: theta ≈ solve tau equation via bisection
    def _frank_tau(theta):
        if abs(theta) < 1e-6:
            return 0.0
        # Debye function D1(x) = (1/x) * integral_0^x t/(exp(t)-1) dt
        from scipy.integrate import quad as _quad

        d1, _ = _quad(lambda t: t / (np.exp(t) - 1 + 1e-15), 0, abs(theta))
        d1 /= abs(theta)
        return 1 - 4 / theta * (1 - d1)

    # Bisection search for Frank theta
    theta_frank = 0.0
    if abs(kendall_tau) > 0.01:
        lo_f, hi_f = -30.0, 30.0
        for _ in range(50):
            mid = (lo_f + hi_f) / 2
            if abs(mid) < 1e-6:
                mid = 0.01
            tau_mid = _frank_tau(mid)
            if tau_mid < kendall_tau:
                lo_f = mid
            else:
                hi_f = mid
        theta_frank = (lo_f + hi_f) / 2

    # Frank copula density
    try:
        t = theta_frank
        if abs(t) > 0.01:
            num = -t * (np.exp(-t) - 1) * np.exp(-t * (u + v))
            den = ((np.exp(-t * u) - 1) * (np.exp(-t * v) - 1) + (np.exp(-t) - 1)) ** 2
            ll_frank = float(np.sum(np.log(np.clip(num / den, 1e-300, None))))
        else:
            ll_frank = 0.0
    except Exception:
        ll_frank = -np.inf

    # Compare by AIC (each copula has 1 parameter)
    copulas = [
        {
            "name": "Gaussian",
            "param": rho_gauss,
            "param_name": "\u03c1",
            "ll": ll_gauss,
            "aic": -2 * ll_gauss + 2,
        },
        {
            "name": "Clayton",
            "param": theta_clayton,
            "param_name": "\u03b8",
            "ll": ll_clayton,
            "aic": -2 * ll_clayton + 2,
        },
        {
            "name": "Frank",
            "param": theta_frank,
            "param_name": "\u03b8",
            "ll": ll_frank,
            "aic": -2 * ll_frank + 2,
        },
    ]
    copulas.sort(key=lambda c: c["aic"])
    best_cop = copulas[0]

    # Tail dependence
    # Clayton: lower tail = 2^(-1/theta), upper = 0
    # Gaussian: both = 0 (for |rho| < 1)
    # Frank: both = 0
    lower_tail = 2 ** (-1 / theta_clayton) if theta_clayton > 0.01 else 0.0
    tail_note = ""
    if best_cop["name"] == "Clayton":
        tail_note = f"Clayton copula has lower-tail dependence = {lower_tail:.3f} \u2014 variables are more correlated in the low/defect region."
    elif best_cop["name"] == "Gaussian":
        tail_note = "Gaussian copula has no tail dependence \u2014 correlation is symmetric across the distribution."
    elif best_cop["name"] == "Frank":
        tail_note = "Frank copula has no tail dependence \u2014 dependency is symmetric and concentrated in the middle."

    summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>COPULA DEPENDENCY MODELING<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:text>>{var1}<</COLOR>> \u00d7 <<COLOR:text>>{var2}<</COLOR>>    N: {n}\n\n"
    summary += "<<COLOR:accent>>\u2500\u2500 Marginal Correlations \u2500\u2500<</COLOR>>\n"
    summary += f"  Pearson r:    {pearson_r:.4f} (p={pearson_p:.4f})\n"
    summary += f"  Spearman \u03c1:  {spearman_r:.4f} (p={spearman_p:.4f})\n"
    summary += f"  Kendall \u03c4:   {kendall_tau:.4f} (p={kendall_p:.4f})\n\n"
    summary += "<<COLOR:accent>>\u2500\u2500 Copula Fit (ranked by AIC) \u2500\u2500<</COLOR>>\n"
    for j, c in enumerate(copulas):
        marker = " \u2190 best" if j == 0 else ""
        summary += f"  {c['name']:<12} {c['param_name']}={c['param']:.4f}  AIC={c['aic']:.1f}{marker}\n"
    summary += f"\n{tail_note}\n"

    result["summary"] = summary
    result["statistics"] = {
        "best_copula": best_cop["name"],
        "best_param": best_cop["param"],
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "kendall_tau": float(kendall_tau),
        "lower_tail_dep": float(lower_tail),
        "copulas": [{k: v for k, v in c.items() if k != "ll"} for c in copulas],
    }
    result["guide_observation"] = (
        f"Copula: best fit = {best_cop['name']} ({best_cop['param_name']}={best_cop['param']:.3f}). Kendall \u03c4 = {kendall_tau:.3f}."
    )

    result["narrative"] = _narrative(
        f"Copula \u2014 {best_cop['name']} fits best ({best_cop['param_name']} = {best_cop['param']:.3f})",
        f"The {best_cop['name']} copula best describes the dependency between <strong>{var1}</strong> and <strong>{var2}</strong> "
        f"(AIC = {best_cop['aic']:.1f}). Kendall \u03c4 = {kendall_tau:.4f}. {tail_note}",
        next_steps="Copulas separate dependency structure from marginal distributions. "
        "Use this for joint probability calculations (e.g., P(both variables exceed spec)) "
        "that would be wrong under a bivariate normal assumption.",
        chart_guidance="The pseudo-observation scatter shows the dependency structure in [0,1]\u00b2 space. "
        "Clustering in corners = tail dependence. Uniform spread = independence.",
    )

    # Plot 1: pseudo-observation scatter
    result["plots"].append(
        {
            "title": "Pseudo-Observations (Copula Space)",
            "data": [
                {
                    "type": "scatter",
                    "x": u.tolist(),
                    "y": v.tolist(),
                    "mode": "markers",
                    "marker": {"size": 4, "color": "#4a90d9", "opacity": 0.5},
                    "name": "Pseudo-obs",
                }
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": f"F({var1})", "range": [0, 1]},
                "yaxis": {"title": f"F({var2})", "range": [0, 1]},
            },
        }
    )

    # Plot 2: original scatter with marginal densities
    result["plots"].append(
        {
            "title": f"{var1} vs {var2}",
            "data": [
                {
                    "type": "scatter",
                    "x": x.tolist(),
                    "y": y.tolist(),
                    "mode": "markers",
                    "marker": {"size": 4, "color": "#4a9f6e", "opacity": 0.5},
                }
            ],
            "layout": {
                "height": 280,
                "xaxis": {"title": var1},
                "yaxis": {"title": var2},
            },
        }
    )

    return result

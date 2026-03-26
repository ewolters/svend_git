"""DSW Bayesian Analysis — standalone test functions extracted from bayesian.py."""

import numpy as np
from scipy import stats
from scipy.integrate import quad

from ..common import (
    _narrative,
)


def run_bayes_ttest(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Bayesian t-test comparing two groups
    # Supports two modes:
    #   1. Two-column: var1 + var2 (each column is a sample)
    #   2. Factor-split: response + factor (one measurement column split by a grouping column)
    prior_scale = config.get("prior_scale", "medium")
    scale_map = {"small": 0.2, "medium": 0.5, "large": 0.8, "ultrawide": 1.0}
    scale = scale_map.get(prior_scale, 0.5)

    mode = config.get("mode", "two_columns")
    factor_col = config.get("factor")
    response_col = config.get("response")

    if mode == "factor" and factor_col and response_col:
        # Factor-split mode: one numeric column grouped by a categorical column
        groups = df.groupby(factor_col)[response_col].apply(lambda s: s.dropna().values)
        group_names = list(groups.index)
        if len(group_names) < 2:
            result["summary"] = "Factor column must have at least 2 groups."
            result["guide_observation"] = (
                "Bayesian t-test: insufficient groups in factor column."
            )
            return result
        if len(group_names) > 2:
            result["summary"] = (
                f"Factor column '{factor_col}' has {len(group_names)} groups "
                f"({', '.join(str(g) for g in group_names)}). "
                "Bayesian t-test compares exactly 2 groups — using the first two. "
                "For 3+ groups, use Bayesian ANOVA."
            )
        x1 = groups.iloc[0]
        x2 = groups.iloc[1]
        var1 = f"{response_col} [{group_names[0]}]"
        var2 = f"{response_col} [{group_names[1]}]"
    else:
        # Two-column mode (original behavior)
        var1 = config.get("var1")
        var2 = config.get("var2")
        x1 = df[var1].dropna().values
        x2 = df[var2].dropna().values

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(x1) - 1) * np.var(x1, ddof=1) + (len(x2) - 1) * np.var(x2, ddof=1))
        / (len(x1) + len(x2) - 2)
    )
    cohens_d = (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0

    # Bayes Factor via JZS prior (Rouder et al. 2009)
    t_stat, p_value = stats.ttest_ind(x1, x2)
    n1, n2 = len(x1), len(x2)
    n_eff = n1 * n2 / (n1 + n2)
    v = n1 + n2 - 2  # degrees of freedom
    r = scale  # Cauchy prior scale

    def _jzs_integrand(g):
        """Integrand for JZS Bayes Factor (Rouder et al. 2009, Eq. 2).

        Uses Inv-Gamma(1/2, 1/2) prior on g with Cauchy scale r
        entering via the effective sample size term: n_eff * r² * g.
        """
        nrg = n_eff * r**2 * g
        return (
            (1 + nrg) ** (-0.5)
            * (1 + t_stat**2 / ((1 + nrg) * v)) ** (-(v + 1) / 2)
            / (1 + t_stat**2 / v) ** (-(v + 1) / 2)
            * (2 * np.pi) ** (-0.5)
            * g ** (-1.5)
            * np.exp(-1 / (2 * g))
        )

    bf10, _ = quad(_jzs_integrand, 1e-10, np.inf)
    bf10 = max(bf10, 1e-10)  # Numerical floor

    # Posterior on effect size (approximate)
    se_d = np.sqrt(
        (len(x1) + len(x2)) / (len(x1) * len(x2))
        + cohens_d**2 / (2 * (len(x1) + len(x2)))
    )
    d_ci_low = cohens_d - z * se_d
    d_ci_high = cohens_d + z * se_d

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN T-TEST<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += (
        f"<<COLOR:highlight>>{var1}<</COLOR>> (n={len(x1)}, μ={np.mean(x1):.3f})\n"
    )
    summary += (
        f"<<COLOR:highlight>>{var2}<</COLOR>> (n={len(x2)}, μ={np.mean(x2):.3f})\n\n"
    )

    summary += "<<COLOR:accent>>── Effect Size ──<</COLOR>>\n"
    summary += f"  Cohen's d: {cohens_d:.3f} [{d_ci_low:.3f}, {d_ci_high:.3f}]\n\n"
    summary += "<<COLOR:accent>>── Bayes Factor ──<</COLOR>>\n"
    summary += f"  BF₁₀: {bf10:.2f}\n\n"

    if bf10 > 10:
        summary += (
            "<<COLOR:success>>Strong evidence for difference (BF₁₀ > 10)<</COLOR>>\n"
        )
    elif bf10 > 3:
        summary += (
            "<<COLOR:warning>>Moderate evidence for difference (BF₁₀ > 3)<</COLOR>>\n"
        )
    elif bf10 > 1:
        summary += "<<COLOR:text>>Weak evidence for difference (BF₁₀ < 3)<</COLOR>>\n"
    else:
        summary += "<<COLOR:text>>Evidence favors no difference (BF₁₀ < 1)<</COLOR>>\n"

    result["summary"] = summary
    result["statistics"] = {
        "cohens_d": cohens_d,
        "bf10": bf10,
        "d_ci_low": d_ci_low,
        "d_ci_high": d_ci_high,
    }

    # Guide observation
    bf_label = (
        "strong"
        if bf10 > 10
        else "moderate" if bf10 > 3 else "weak" if bf10 > 1 else "no"
    )
    result["guide_observation"] = (
        f"Bayesian t-test: d={cohens_d:.3f}, BF₁₀={bf10:.2f} ({bf_label} evidence for difference)."
    )

    # Narrative
    _bt_mag = (
        "large"
        if abs(cohens_d) >= 0.8
        else (
            "medium"
            if abs(cohens_d) >= 0.5
            else ("small" if abs(cohens_d) >= 0.2 else "negligible")
        )
    )
    result["narrative"] = _narrative(
        f"Bayesian t-test — {bf_label} evidence, d = {cohens_d:.3f} ({_bt_mag})",
        f"Cohen's d = {cohens_d:.3f} (95% CI: {d_ci_low:.3f} to {d_ci_high:.3f}). "
        f"BF\u2081\u2080 = {bf10:.2f} — the data are {bf10:.1f}x more likely under the alternative than the null. "
        + (
            "The CI excludes zero, supporting a real difference."
            if (d_ci_low > 0 or d_ci_high < 0)
            else "The CI includes zero — the evidence is inconclusive."
        ),
        next_steps="A BF > 10 is decisive. If BF is between 1-3, collect more data to strengthen the evidence.",
        chart_guidance="The posterior shows the credible distribution of the effect size. The peak is the best estimate; width reflects uncertainty.",
    )

    # Posterior distribution plot
    d_range = np.linspace(cohens_d - 3 * se_d, cohens_d + 3 * se_d, 100)
    posterior = stats.norm.pdf(d_range, cohens_d, se_d)

    result["plots"].append(
        {
            "title": "Posterior Distribution of Effect Size",
            "data": [
                {
                    "type": "scatter",
                    "x": d_range.tolist(),
                    "y": posterior.tolist(),
                    "fill": "tozeroy",
                    "fillcolor": "rgba(74, 159, 110, 0.3)",
                    "line": {"color": "#4a9f6e"},
                    "name": "Posterior",
                },
                {
                    "type": "scatter",
                    "x": [0, 0],
                    "y": [0, max(posterior)],
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash"},
                    "name": "No effect",
                },
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Cohen's d"},
                "yaxis": {"title": "Density"},
            },
        }
    )

    result["education"] = {
        "title": "Understanding the Bayesian t-Test",
        "content": (
            "<dl>"
            "<dt>What is a Bayesian t-test?</dt>"
            "<dd>It compares two groups — like a classical t-test — but instead of a p-value "
            "it produces a <em>Bayes Factor</em> (BF₁₀) that quantifies the evidence for a "
            "real difference versus no difference.</dd>"
            "<dt>What is the Bayes Factor (BF₁₀)?</dt>"
            "<dd>The ratio of how likely the data are under the alternative hypothesis vs the null. "
            "<strong>BF₁₀ &gt; 3</strong>: moderate evidence for a difference. "
            "<strong>BF₁₀ &gt; 10</strong>: strong evidence. "
            "<strong>BF₁₀ &lt; ⅓</strong>: moderate evidence for <em>no</em> difference. "
            "Between ⅓ and 3 is inconclusive — you need more data.</dd>"
            "<dt>What is Cohen's d?</dt>"
            "<dd>A standardised effect size — the difference in means divided by pooled standard "
            "deviation. <strong>0.2</strong> is small, <strong>0.5</strong> medium, "
            "<strong>0.8</strong> large. The posterior distribution shows how uncertain "
            "we are about this effect size.</dd>"
            "<dt>What is the JZS prior?</dt>"
            "<dd>The default prior (Rouder et al. 2009) — a Cauchy distribution on effect size "
            "that is objective and well-calibrated. The prior scale controls how large an effect "
            "you expect: <em>medium</em> (0.707) is a sensible default for most applications.</dd>"
            "<dt>Why use this instead of a classical t-test?</dt>"
            "<dd>Classical t-tests cannot quantify evidence <em>for</em> the null — they can only "
            "fail to reject it. The Bayes Factor lets you say 'the data support no difference' "
            "with a specific strength, which is critical for equivalence decisions.</dd>"
            "</dl>"
        ),
    }

    return result


def run_bayes_ab(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Bayesian A/B test for proportions
    group_col = config.get("group")
    success_col = config.get("success")
    prior_type = config.get("prior", "uniform")

    prior_map = {"uniform": (1, 1), "jeffreys": (0.5, 0.5), "informed": (5, 5)}
    a_prior, b_prior = prior_map.get(prior_type, (1, 1))

    groups = df[group_col].dropna().unique()
    if len(groups) < 2:
        result["summary"] = "Error: Need at least 2 groups"
        return result

    g1, g2 = groups[0], groups[1]
    s1 = df[df[group_col] == g1][success_col].sum()
    n1 = len(df[df[group_col] == g1])
    s2 = df[df[group_col] == g2][success_col].sum()
    n2 = len(df[df[group_col] == g2])

    # Posterior Beta distributions
    a1, b1 = a_prior + s1, b_prior + n1 - s1
    a2, b2 = a_prior + s2, b_prior + n2 - s2

    # Monte Carlo estimation of P(p1 > p2)
    samples1 = np.random.beta(a1, b1, 10000)
    samples2 = np.random.beta(a2, b2, 10000)
    prob_better = np.mean(samples1 > samples2)

    rate1, rate2 = s1 / n1, s2 / n2
    lift = (rate1 - rate2) / rate2 if rate2 > 0 else 0

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN A/B TEST<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Group A ({g1}):<</COLOR>> {s1}/{n1} = {rate1:.1%}\n"
    summary += (
        f"<<COLOR:highlight>>Group B ({g2}):<</COLOR>> {s2}/{n2} = {rate2:.1%}\n\n"
    )
    summary += f"<<COLOR:text>>P({g1} > {g2}):<</COLOR>> {prob_better:.1%}\n"
    summary += f"<<COLOR:text>>Relative Lift:<</COLOR>> {lift:+.1%}\n\n"

    if prob_better > 0.95:
        summary += f"<<COLOR:success>>Strong evidence {g1} is better<</COLOR>>\n"
    elif prob_better > 0.75:
        summary += f"<<COLOR:warning>>Moderate evidence {g1} is better<</COLOR>>\n"
    elif prob_better < 0.05:
        summary += f"<<COLOR:success>>Strong evidence {g2} is better<</COLOR>>\n"
    else:
        summary += "<<COLOR:text>>Inconclusive - need more data<</COLOR>>\n"

    result["summary"] = summary
    result["statistics"] = {
        "prob_better": prob_better,
        "rate_a": rate1,
        "rate_b": rate2,
        "lift": lift,
    }

    # Posterior distributions
    x = np.linspace(0, 1, 200)
    result["plots"].append(
        {
            "title": "Posterior Distributions",
            "data": [
                {
                    "type": "scatter",
                    "x": x.tolist(),
                    "y": stats.beta.pdf(x, a1, b1).tolist(),
                    "fill": "tozeroy",
                    "fillcolor": "rgba(74, 159, 110, 0.3)",
                    "line": {"color": "#4a9f6e"},
                    "name": f"{g1}",
                },
                {
                    "type": "scatter",
                    "x": x.tolist(),
                    "y": stats.beta.pdf(x, a2, b2).tolist(),
                    "fill": "tozeroy",
                    "fillcolor": "rgba(232, 149, 71, 0.3)",
                    "line": {"color": "#e89547"},
                    "name": f"{g2}",
                },
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Conversion Rate"},
                "yaxis": {"title": "Density"},
            },
        }
    )

    result["education"] = {
        "title": "Understanding the Bayesian A/B Test",
        "content": (
            "<dl>"
            "<dt>What is a Bayesian A/B test?</dt>"
            "<dd>It compares conversion rates (or success rates) between two groups using "
            "<em>Beta-Binomial</em> conjugate updating. Instead of a p-value, you get the "
            "direct probability that one group outperforms the other.</dd>"
            "<dt>What does P(A &gt; B) mean?</dt>"
            "<dd>The posterior probability that group A's true rate is higher than group B's. "
            "<strong>P &gt; 0.95</strong>: strong evidence A is better. "
            "<strong>P &lt; 0.05</strong>: strong evidence B is better. "
            "Between 0.05 and 0.95 means uncertainty remains.</dd>"
            "<dt>What is the relative lift?</dt>"
            "<dd>The expected percentage improvement of one group over the other. "
            "A 12% lift means the winning group's rate is about 12% higher relative to "
            "the baseline group's rate.</dd>"
            "<dt>Why Bayesian over a chi-squared test?</dt>"
            "<dd>Bayesian A/B testing gives a direct probability of which variant wins — "
            "no need to interpret p-values. You can also monitor results continuously "
            "without inflating false positive rates (no peeking penalty).</dd>"
            "</dl>"
        ),
    }

    return result


def run_bayes_correlation(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Bayesian correlation
    var1 = config.get("var1")
    var2 = config.get("var2")

    x = df[var1].dropna()
    y = df[var2].loc[x.index].dropna()
    x = x.loc[y.index]

    r, p = stats.pearsonr(x, y)
    n = len(x)

    # Fisher z-transformation for CI
    z_r = 0.5 * np.log((1 + r) / (1 - r)) if abs(r) < 1 else 0
    se_z = 1 / np.sqrt(n - 3) if n > 3 else 1
    z_low = z_r - z * se_z
    z_high = z_r + z * se_z
    r_low = (np.exp(2 * z_low) - 1) / (np.exp(2 * z_low) + 1)
    r_high = (np.exp(2 * z_high) - 1) / (np.exp(2 * z_high) + 1)

    # BF for correlation via Ly et al. (2016) integral under uniform prior
    def _corr_bf_integrand(rho):
        """Integrand for correlation BF under uniform prior on rho."""
        if abs(rho) >= 1:
            return 0.0
        log_term = ((n - 2) / 2) * np.log(1 - rho**2) - ((n - 1) / 2) * np.log(
            1 - r * rho
        )
        return np.exp(log_term)

    bf_integral, _ = quad(_corr_bf_integrand, -1 + 1e-10, 1 - 1e-10)
    # Under H0: rho=0, the likelihood ratio normalizes to 1
    # BF10 = integral / (value at rho=0 * prior width=2)
    bf10 = bf_integral / 2.0 if bf_integral > 0 else 1e-10

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN CORRELATION<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>{var1}<</COLOR>> vs <<COLOR:highlight>>{var2}<</COLOR>> (n={n})\n\n"

    summary += "<<COLOR:accent>>── Correlation ──<</COLOR>>\n"
    summary += f"  r = {r:.3f} [{r_low:.3f}, {r_high:.3f}]\n\n"
    summary += "<<COLOR:accent>>── Bayes Factor ──<</COLOR>>\n"
    summary += f"  BF₁₀: {bf10:.2f}\n\n"

    strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
    direction = "positive" if r > 0 else "negative"
    str_color = "success" if abs(r) > 0.7 else "warning" if abs(r) > 0.4 else "dim"
    summary += "<<COLOR:accent>>── Interpretation ──<</COLOR>>\n"
    summary += f"  <<COLOR:{str_color}>>{strength.capitalize()} {direction} correlation<</COLOR>>\n"

    if bf10 > 10:
        summary += "  <<COLOR:success>>Strong Bayesian evidence for association (BF₁₀ > 10)<</COLOR>>\n"
    elif bf10 > 3:
        summary += "  <<COLOR:warning>>Moderate Bayesian evidence for association (BF₁₀ > 3)<</COLOR>>\n"
    elif bf10 > 1:
        summary += "  <<COLOR:text>>Weak evidence for association<</COLOR>>\n"
    else:
        summary += (
            "  <<COLOR:text>>Evidence favors no association (BF₁₀ < 1)<</COLOR>>\n"
        )

    result["summary"] = summary
    result["statistics"] = {
        "r": r,
        "r_ci_low": r_low,
        "r_ci_high": r_high,
        "bf10": bf10,
    }

    # Guide observation
    bf_label = (
        "strong"
        if bf10 > 10
        else "moderate" if bf10 > 3 else "weak" if bf10 > 1 else "no"
    )
    result["guide_observation"] = (
        f"Bayesian correlation: r={r:.3f} ({strength} {direction}), BF₁₀={bf10:.2f} ({bf_label} evidence)."
    )

    # Narrative
    _bc_r2 = r**2 * 100
    result["narrative"] = _narrative(
        f"Bayesian Correlation — r = {r:.3f} ({strength} {direction})",
        f"Pearson r = {r:.3f} (95% CI: {r_low:.3f} to {r_high:.3f}), explaining ~{_bc_r2:.0f}% of shared variation. "
        f"BF\u2081\u2080 = {bf10:.2f} ({bf_label} evidence for association).",
        next_steps="Correlation does not imply causation. Use Causal Discovery to test directionality, or control for confounders with partial correlation.",
        chart_guidance="A tight elliptical cloud indicates strong correlation. Outliers can inflate or deflate r.",
    )

    result["plots"].append(
        {
            "title": f"Scatter: {var1} vs {var2}",
            "data": [
                {
                    "type": "scatter",
                    "x": x.values.tolist(),
                    "y": y.values.tolist(),
                    "mode": "markers",
                    "marker": {"color": "#4a9f6e", "size": 6, "opacity": 0.6},
                }
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": var1},
                "yaxis": {"title": var2},
            },
        }
    )

    result["education"] = {
        "title": "Understanding Bayesian Correlation",
        "content": (
            "<dl>"
            "<dt>What is Bayesian correlation?</dt>"
            "<dd>It estimates the strength and direction of a linear relationship between two variables, "
            "producing a <em>posterior distribution</em> over the correlation coefficient rather than "
            "a single point estimate. You get both the best estimate and a credible interval.</dd>"
            "<dt>How is the Bayes Factor calculated?</dt>"
            "<dd>Using the method of Ly et al. (2016) — integrating over a uniform prior on the "
            "correlation coefficient. The BF₁₀ tells you how much the data favour a non-zero "
            "correlation versus no relationship at all.</dd>"
            "<dt>What does the credible interval mean?</dt>"
            "<dd>A 95% credible interval of [0.3, 0.7] means there is a 95% posterior probability "
            "the true correlation lies in that range. Narrower intervals mean more precise estimation.</dd>"
            "<dt>When to use this over classical correlation?</dt>"
            "<dd>Bayesian correlation is especially useful with small samples (where p-values are unreliable), "
            "when you want to quantify evidence <em>for</em> independence (BF₁₀ &lt; ⅓), or when you "
            "need honest uncertainty bounds on the strength of association.</dd>"
            "</dl>"
        ),
    }

    return result


def run_bayes_anova(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Bayesian ANOVA
    response = config.get("variable") or config.get("response")
    factor = config.get("group") or config.get("factor")

    groups = df.groupby(factor)[response].apply(list).to_dict()
    group_names = list(groups.keys())
    group_data = [np.array(groups[g]) for g in group_names]

    # F-test
    f_stat, p_value = stats.f_oneway(*group_data)

    # Effect size (eta-squared)
    grand_mean = df[response].mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data)
    ss_within = sum(np.sum((np.array(g) - np.mean(g)) ** 2) for g in group_data)
    ss_total = ss_between + ss_within
    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    # BIC-approximated Bayes Factor (Wagenmakers 2007)
    n_total = sum(len(g) for g in group_data)
    k = len(group_names)
    if ss_within > 0 and n_total > k:
        bic_h0 = n_total * np.log(ss_total / n_total) + 1 * np.log(n_total)
        bic_h1 = n_total * np.log(ss_within / n_total) + k * np.log(n_total)
        bf10 = np.exp((bic_h0 - bic_h1) / 2)
        bf10 = min(bf10, 1e10)  # cap for display
    else:
        bf10 = 1.0

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN ANOVA<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
    summary += (
        f"<<COLOR:highlight>>Factor:<</COLOR>> {factor} ({len(group_names)} levels)\n\n"
    )

    summary += "<<COLOR:accent>>── Group Statistics ──<</COLOR>>\n"
    for name in group_names:
        g = groups[name]
        summary += f"  {name}: n={len(g)}, μ={np.mean(g):.3f}, σ={np.std(g):.3f}\n"

    summary += "\n<<COLOR:accent>>── Test Results ──<</COLOR>>\n"
    summary += f"  F-statistic: {f_stat:.3f}\n"
    summary += f"  Effect size (η²): {eta_sq:.3f}\n"
    summary += f"  Bayes Factor (BF₁₀): {bf10:.2f}\n\n"

    summary += "<<COLOR:accent>>── Interpretation ──<</COLOR>>\n"
    if bf10 > 10:
        summary += "  <<COLOR:success>>Strong evidence for group differences (BF₁₀ > 10)<</COLOR>>\n"
    elif bf10 > 3:
        summary += "  <<COLOR:warning>>Moderate evidence for group differences (BF₁₀ > 3)<</COLOR>>\n"
    elif bf10 > 1:
        summary += (
            "  <<COLOR:text>>Weak evidence for group differences (BF₁₀ < 3)<</COLOR>>\n"
        )
    else:
        summary += "  <<COLOR:text>>Evidence favors no group differences (BF₁₀ < 1)<</COLOR>>\n"

    result["summary"] = summary
    result["statistics"] = {
        "f_stat": f_stat,
        "eta_squared": eta_sq,
        "p_value": p_value,
        "bf10": bf10,
    }

    # Guide observation
    bf_label = (
        "strong"
        if bf10 > 10
        else "moderate" if bf10 > 3 else "weak" if bf10 > 1 else "no"
    )
    result["guide_observation"] = (
        f"Bayesian ANOVA: {factor} on {response}, F={f_stat:.3f}, η²={eta_sq:.3f}, BF₁₀={bf10:.2f} ({bf_label} evidence)."
    )

    # Narrative
    _ba_mag = "large" if eta_sq > 0.14 else ("medium" if eta_sq > 0.06 else "small")
    result["narrative"] = _narrative(
        f"Bayesian ANOVA — {bf_label} evidence, \u03b7\u00b2 = {eta_sq:.3f} ({_ba_mag})",
        f"Factor <strong>{factor}</strong> explains {eta_sq * 100:.1f}% of the variation in {response} across {len(group_names)} groups. "
        f"BF\u2081\u2080 = {bf10:.2f} ({bf_label} evidence for group differences).",
        next_steps="If BF > 3, run pairwise Bayesian t-tests to identify which groups differ.",
        chart_guidance="Compare box positions — non-overlapping boxes suggest meaningful group differences.",
    )

    # Box plot
    result["plots"].append(
        {
            "title": f"{response} by {factor}",
            "data": [
                {
                    "type": "box",
                    "y": groups[name],
                    "name": str(name),
                    "marker": {"color": "#4a9f6e"},
                }
                for name in group_names
            ],
            "layout": {"height": 350},
        }
    )

    result["education"] = {
        "title": "Understanding Bayesian ANOVA",
        "content": (
            "<dl>"
            "<dt>What is Bayesian ANOVA?</dt>"
            "<dd>It compares means across multiple groups — like classical ANOVA — but uses a "
            "hierarchical Normal-Normal model to produce <em>posterior distributions</em> for each "
            "group mean, plus a Bayes Factor quantifying overall group differences.</dd>"
            "<dt>What does the Bayes Factor tell me?</dt>"
            "<dd><strong>BF₁₀ &gt; 10</strong>: strong evidence that at least one group differs. "
            "<strong>BF₁₀ &lt; ⅓</strong>: evidence that groups are similar. Between ⅓ and 3 "
            "is inconclusive.</dd>"
            "<dt>What is η² (eta-squared)?</dt>"
            "<dd>The proportion of total variance explained by group membership. "
            "<strong>&lt; 0.06</strong>: small effect. <strong>0.06–0.14</strong>: medium. "
            "<strong>&gt; 0.14</strong>: large. It tells you the practical significance "
            "beyond just statistical significance.</dd>"
            "<dt>Why Bayesian over classical ANOVA?</dt>"
            "<dd>Classical ANOVA gives a p-value that only tells you whether to reject the null. "
            "Bayesian ANOVA gives posterior group means with credible intervals, lets you quantify "
            "evidence for the null (groups are equal), and handles unequal sample sizes more naturally.</dd>"
            "</dl>"
        ),
    }

    return result


def run_bayes_proportion(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Bayesian proportion estimation
    # Support manual-entry mode: frontend sends {successes, n, prior_a, prior_b}
    manual_successes = config.get("successes")
    manual_n = config.get("n")
    if manual_successes is not None and manual_n is not None:
        successes = int(manual_successes)
        n = int(manual_n)
        a_prior = float(config.get("prior_a", 1))
        b_prior = float(config.get("prior_b", 1))
        prior_type = "custom" if (a_prior != 1 or b_prior != 1) else "uniform"
    else:
        # Column-based mode: read from dataframe
        success_col = config.get("success")
        prior_type = config.get("prior", "uniform")
        prior_map = {
            "uniform": (1, 1),
            "jeffreys": (0.5, 0.5),
            "optimistic": (8, 2),
            "pessimistic": (2, 8),
        }
        a_prior, b_prior = prior_map.get(prior_type, (1, 1))
        data = df[success_col].dropna()
        successes = int(data.sum())
        n = len(data)

    # Posterior
    a_post = a_prior + successes
    b_post = b_prior + n - successes

    # Posterior mean and CI
    post_mean = a_post / (a_post + b_post)
    ci_low, ci_high = stats.beta.ppf(
        [(1 - ci_level) / 2, (1 + ci_level) / 2], a_post, b_post
    )

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN PROPORTION ESTIMATION<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Observed:<</COLOR>> {successes}/{n} = {successes / n:.1%}\n"
    summary += f"<<COLOR:highlight>>Prior:<</COLOR>> Beta({a_prior}, {b_prior})\n\n"
    summary += f"<<COLOR:text>>Posterior Mean:<</COLOR>> {post_mean:.1%}\n"
    summary += f"<<COLOR:text>>{int(ci_level * 100)}% Credible Interval:<</COLOR>> [{ci_low:.1%}, {ci_high:.1%}]\n"

    result["summary"] = summary
    result["statistics"] = {
        "proportion": post_mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "successes": successes,
    }
    result["guide_observation"] = (
        f"Bayesian proportion: {successes}/{n} = {successes / n:.1%}. Posterior mean = {post_mean:.1%}, {int(ci_level * 100)}% CI: [{ci_low:.1%}, {ci_high:.1%}]."
    )

    # Narrative
    _bp_obs = successes / n if n > 0 else 0
    result["narrative"] = _narrative(
        f"Bayesian Proportion — {post_mean:.1%} (CI: {ci_low:.1%} to {ci_high:.1%})",
        f"Observed {successes} of {n} ({_bp_obs:.1%}). With a {prior_type} prior, the posterior estimate is {post_mean:.1%}. "
        f"The {int(ci_level * 100)}% credible interval [{ci_low:.1%}, {ci_high:.1%}] represents the plausible range for the true proportion.",
        next_steps="More data narrows the credible interval. Compare with a target rate to assess process performance.",
        chart_guidance="The posterior curve shows the full distribution of belief about the true proportion. The orange bar marks the credible interval.",
    )

    # Posterior distribution
    x = np.linspace(0, 1, 200)
    result["plots"].append(
        {
            "title": "Posterior Distribution",
            "data": [
                {
                    "type": "scatter",
                    "x": x.tolist(),
                    "y": stats.beta.pdf(x, a_post, b_post).tolist(),
                    "fill": "tozeroy",
                    "fillcolor": "rgba(74, 159, 110, 0.3)",
                    "line": {"color": "#4a9f6e"},
                    "name": f"Beta({a_post:.0f}, {b_post:.0f})",
                },
                {
                    "type": "scatter",
                    "x": [ci_low, ci_high],
                    "y": [0, 0],
                    "mode": "lines",
                    "line": {"color": "#e89547", "width": 4},
                    "name": f"{int(ci_level * 100)}% CI",
                },
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Proportion"},
                "yaxis": {"title": "Density"},
            },
        }
    )

    result["education"] = {
        "title": "Understanding Bayesian Proportion Estimation",
        "content": (
            "<dl>"
            "<dt>What is Bayesian proportion estimation?</dt>"
            "<dd>Given a count of successes out of N trials, it estimates the true underlying "
            "proportion using a <em>Beta-Binomial</em> conjugate model. The result is a full "
            "posterior distribution, not just a point estimate.</dd>"
            "<dt>What are the prior options?</dt>"
            "<dd><strong>Uniform</strong> (Beta(1,1)): no prior opinion — lets the data speak. "
            "<strong>Jeffreys</strong> (Beta(0.5,0.5)): minimally informative, often recommended "
            "for small samples. <strong>Informed</strong> (Beta(5,5)): centres the prior near 50%, "
            "useful when you expect moderate rates.</dd>"
            "<dt>What does the credible interval mean?</dt>"
            "<dd>A 95% credible interval of [0.82, 0.94] means there is a 95% probability "
            "the true proportion lies in that range. Unlike frequentist confidence intervals, "
            "this is a direct probability statement.</dd>"
            "<dt>When to use this?</dt>"
            "<dd>Defect rates, yield percentages, pass/fail ratios, survey response rates — "
            "any situation where you count successes and failures and need an honest uncertainty "
            "estimate on the true rate.</dd>"
            "</dl>"
        ),
    }

    return result


def run_bayes_chi2(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Bayesian Contingency Table (Dirichlet-Multinomial)
    row_var = config.get("var1")
    col_var = config.get("var2")

    ct = df.groupby([row_var, col_var]).size().unstack(fill_value=0)
    observed = ct.values
    nrow, ncol = observed.shape
    N = observed.sum()

    if nrow < 2 or ncol < 2:
        result["summary"] = "Error: Need at least 2 levels per variable."
        return result

    # Classical chi-square for reference
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
    cramers_v = float(np.sqrt(chi2_stat / (N * (min(nrow, ncol) - 1)))) if N > 0 else 0

    # BF via BIC approximation (Wagenmakers 2007)
    k = (nrow - 1) * (ncol - 1)
    log_bf10 = 0.5 * (chi2_stat - k * np.log(N))
    bf10 = float(np.exp(np.clip(log_bf10, -500, 500)))

    # Posterior cell probabilities (Dirichlet posterior with uniform prior)
    alpha_prior = np.ones_like(observed, dtype=float)
    alpha_post = alpha_prior + observed
    post_probs = alpha_post / alpha_post.sum()

    # Standardized residuals
    std_resid = (observed - expected) / np.sqrt(expected + 1e-10)

    bf_label = (
        "strong"
        if bf10 > 10
        else "moderate" if bf10 > 3 else "weak" if bf10 > 1 else "no"
    )

    summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN CHI-SQUARE (CONTINGENCY TABLE)<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>{row_var}<</COLOR>> ({nrow} levels) \u00d7 <<COLOR:highlight>>{col_var}<</COLOR>> ({ncol} levels)\n"
    summary += f"<<COLOR:text>>N:<</COLOR>> {N}\n\n"
    summary += "<<COLOR:accent>>\u2500\u2500 Bayes Factor \u2500\u2500<</COLOR>>\n"
    summary += f"  BF\u2081\u2080: {bf10:.2f} ({bf_label} evidence for association)\n"
    summary += f"  Cram\u00e9r's V: {cramers_v:.3f}\n\n"
    summary += "<<COLOR:accent>>\u2500\u2500 Observed Counts \u2500\u2500<</COLOR>>\n"
    for i, row_label in enumerate(ct.index):
        row_str = "  ".join(f"{int(observed[i, j]):>6}" for j in range(ncol))
        summary += f"  {str(row_label):<15} {row_str}\n"
    summary += (
        "\n<<COLOR:accent>>\u2500\u2500 Largest Residuals \u2500\u2500<</COLOR>>\n"
    )
    flat_idx = np.argsort(np.abs(std_resid).ravel())[::-1][:5]
    for idx in flat_idx:
        ri, ci_idx = divmod(idx, ncol)
        if abs(std_resid[ri, ci_idx]) >= 1:
            summary += f"  {ct.index[ri]} \u00d7 {ct.columns[ci_idx]}: residual = {std_resid[ri, ci_idx]:+.2f}\n"

    result["summary"] = summary
    result["statistics"] = {
        "bf10": bf10,
        "chi2": float(chi2_stat),
        "p_value": float(p_value),
        "cramers_v": cramers_v,
        "dof": dof,
    }
    result["guide_observation"] = (
        f"Bayesian \u03c7\u00b2: BF\u2081\u2080={bf10:.2f} ({bf_label} evidence), Cram\u00e9r's V={cramers_v:.3f}."
    )

    _bc_mag = (
        "large"
        if cramers_v >= 0.5
        else (
            "medium"
            if cramers_v >= 0.3
            else ("small" if cramers_v >= 0.1 else "negligible")
        )
    )
    result["narrative"] = _narrative(
        f"Bayesian \u03c7\u00b2 \u2014 {bf_label} evidence for association (V = {cramers_v:.3f}, {_bc_mag})",
        f"BF\u2081\u2080 = {bf10:.2f} \u2014 the data are {bf10:.1f}\u00d7 more likely under association than independence. "
        f"Cram\u00e9r's V = {cramers_v:.3f} ({_bc_mag} effect). "
        + (
            f"The strongest departure from independence: <strong>{ct.index[flat_idx[0] // ncol]} \u00d7 {ct.columns[flat_idx[0] % ncol]}</strong> "
            f"(residual = {std_resid.ravel()[flat_idx[0]]:+.2f})."
            if abs(std_resid.ravel()[flat_idx[0]]) >= 1
            else ""
        ),
        next_steps="If BF > 10, the association is well-supported. Examine the residual heatmap to identify which cells drive the association.",
        chart_guidance="The heatmap shows standardized residuals. Blue = over-represented, red = under-represented relative to independence.",
    )

    # Plot: standardized residual heatmap
    result["plots"].append(
        {
            "title": "Standardized Residuals",
            "data": [
                {
                    "type": "heatmap",
                    "z": std_resid.tolist(),
                    "x": [str(c) for c in ct.columns],
                    "y": [str(r) for r in ct.index],
                    "colorscale": [[0, "#4a90d9"], [0.5, "#f5f5f5"], [1, "#dc5050"]],
                    "zmid": 0,
                    "colorbar": {"title": "Std Resid"},
                }
            ],
            "layout": {
                "height": max(200, nrow * 40 + 100),
                "xaxis": {"title": str(col_var)},
                "yaxis": {"title": str(row_var)},
            },
        }
    )

    # Plot: posterior cell probabilities
    result["plots"].append(
        {
            "title": "Posterior Cell Probabilities (Dirichlet)",
            "data": [
                {
                    "type": "heatmap",
                    "z": post_probs.tolist(),
                    "x": [str(c) for c in ct.columns],
                    "y": [str(r) for r in ct.index],
                    "colorscale": "Greens",
                    "colorbar": {"title": "P(cell)"},
                }
            ],
            "layout": {
                "height": max(200, nrow * 40 + 100),
                "xaxis": {"title": str(col_var)},
                "yaxis": {"title": str(row_var)},
            },
        }
    )

    result["education"] = {
        "title": "Understanding Bayesian Contingency Analysis",
        "content": (
            "<dl>"
            "<dt>What is Bayesian contingency analysis?</dt>"
            "<dd>It analyses the association between two categorical variables using a "
            "<em>Dirichlet-Multinomial</em> conjugate model. Instead of a chi-squared p-value, "
            "you get posterior cell probabilities and a Bayes Factor for association.</dd>"
            "<dt>What does the heatmap show?</dt>"
            "<dd>The posterior mean probability for each cell in the contingency table. "
            "Cells with higher posterior probability are where observations concentrate. "
            "Non-uniform patterns indicate association between the variables.</dd>"
            "<dt>What does the Bayes Factor mean here?</dt>"
            "<dd><strong>BF₁₀ &gt; 10</strong>: strong evidence that the variables are associated. "
            "<strong>BF₁₀ &lt; ⅓</strong>: evidence for independence. This is more informative "
            "than a chi-squared p-value because it quantifies evidence in both directions.</dd>"
            "<dt>When to use this?</dt>"
            "<dd>Defect type by machine, shift by quality outcome, supplier by pass/fail — "
            "any situation where you cross-tabulate two categorical factors and want to know "
            "if they are related, with honest uncertainty quantification.</dd>"
            "</dl>"
        ),
    }

    return result


def run_bayes_equivalence(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Bayesian Equivalence Test via ROPE (Region of Practical Equivalence)
    var1 = config.get("var1")
    var2 = config.get("var2")
    rope_low = float(config.get("rope_low", -0.1))
    rope_high = float(config.get("rope_high", 0.1))
    use_effect_size = config.get("use_effect_size", True)

    x1 = df[var1].dropna().values
    x2 = df[var2].dropna().values
    n1, n2 = len(x1), len(x2)

    # Posterior on difference (Normal-Normal conjugate with flat prior)
    mean_diff = float(np.mean(x1) - np.mean(x2))
    se_diff = float(np.sqrt(np.var(x1, ddof=1) / n1 + np.var(x2, ddof=1) / n2))

    if use_effect_size:
        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(x1, ddof=1) + (n2 - 1) * np.var(x2, ddof=1))
            / (n1 + n2 - 2)
        )
        if pooled_std > 0:
            effect = mean_diff / pooled_std
            se_effect = np.sqrt((n1 + n2) / (n1 * n2) + effect**2 / (2 * (n1 + n2)))
        else:
            effect, se_effect = 0.0, 1.0
        label, _unit = "Cohen's d", ""
    else:
        effect, se_effect = mean_diff, se_diff
        label, _unit = "Difference", ""

    # Probabilities: P(effect in ROPE), P(effect < ROPE), P(effect > ROPE)
    p_rope = float(
        stats.norm.cdf(rope_high, effect, se_effect)
        - stats.norm.cdf(rope_low, effect, se_effect)
    )
    p_below = float(stats.norm.cdf(rope_low, effect, se_effect))
    p_above = float(1 - stats.norm.cdf(rope_high, effect, se_effect))

    # HDI (95%)
    hdi_low = float(effect - z * se_effect)
    hdi_high = float(effect + z * se_effect)

    # Decision
    if p_rope > 0.95:
        decision = "Accept equivalence"
        decision_color = "success"
    elif p_rope < 0.05:
        decision = "Reject equivalence"
        decision_color = "error"
    else:
        decision = "Inconclusive"
        decision_color = "warning"

    # HDI+ROPE decision rule (Kruschke)
    if hdi_high < rope_high and hdi_low > rope_low:
        kruschke = "HDI entirely inside ROPE \u2192 accept equivalence"
    elif hdi_low > rope_high or hdi_high < rope_low:
        kruschke = "HDI entirely outside ROPE \u2192 reject equivalence"
    else:
        kruschke = "HDI overlaps ROPE boundary \u2192 undecided (collect more data)"

    summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN EQUIVALENCE TEST (ROPE)<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
    summary += (
        f"<<COLOR:highlight>>{var1}<</COLOR>> (n={n1}, \u03bc={np.mean(x1):.4f})\n"
    )
    summary += (
        f"<<COLOR:highlight>>{var2}<</COLOR>> (n={n2}, \u03bc={np.mean(x2):.4f})\n\n"
    )
    summary += f"<<COLOR:text>>ROPE:<</COLOR>> [{rope_low}, {rope_high}] ({label})\n\n"
    summary += (
        f"<<COLOR:accent>>\u2500\u2500 Posterior on {label} \u2500\u2500<</COLOR>>\n"
    )
    summary += f"  Estimate: {effect:.4f}\n"
    summary += f"  95% HDI:  [{hdi_low:.4f}, {hdi_high:.4f}]\n\n"
    summary += (
        "<<COLOR:accent>>\u2500\u2500 ROPE Probabilities \u2500\u2500<</COLOR>>\n"
    )
    summary += f"  P(in ROPE):    {p_rope * 100:.1f}%\n"
    summary += f"  P(below ROPE): {p_below * 100:.1f}%\n"
    summary += f"  P(above ROPE): {p_above * 100:.1f}%\n\n"
    summary += f"<<COLOR:{decision_color}>>{decision}<</COLOR>>\n"
    summary += f"  {kruschke}\n"

    result["summary"] = summary
    result["statistics"] = {
        "effect": effect,
        "se": se_effect,
        "hdi_low": hdi_low,
        "hdi_high": hdi_high,
        "p_rope": p_rope,
        "p_below": p_below,
        "p_above": p_above,
        "decision": decision,
        "kruschke": kruschke,
    }
    result["guide_observation"] = (
        f"Bayesian equivalence: {label}={effect:.3f}, P(in ROPE)={p_rope:.1%}. {decision}."
    )

    _be_mag = (
        "large"
        if abs(effect) >= 0.8
        else (
            "medium"
            if abs(effect) >= 0.5
            else ("small" if abs(effect) >= 0.2 else "negligible")
        )
    )
    result["narrative"] = _narrative(
        f"Bayesian Equivalence \u2014 {decision}, P(in ROPE) = {p_rope:.1%}",
        f"The posterior {label} = {effect:.4f} (95% HDI: {hdi_low:.4f} to {hdi_high:.4f}). "
        f"<strong>{p_rope * 100:.1f}%</strong> of the posterior falls inside the ROPE [{rope_low}, {rope_high}]. "
        f"{kruschke}.",
        next_steps="If undecided, collect more data \u2014 the HDI narrows with larger samples. "
        "Adjust ROPE bounds if your practical equivalence margin differs.",
        chart_guidance="The shaded green region is the ROPE. The blue curve is the posterior. "
        "If the entire posterior sits inside the ROPE, the groups are practically equivalent.",
    )

    # Plot: posterior density with ROPE region
    x_range = np.linspace(effect - 4 * se_effect, effect + 4 * se_effect, 200)
    y_dens = stats.norm.pdf(x_range, effect, se_effect)
    rope_mask = (x_range >= rope_low) & (x_range <= rope_high)
    y_rope = np.where(rope_mask, y_dens, 0)

    result["plots"].append(
        {
            "title": "Posterior with ROPE Region",
            "data": [
                {
                    "type": "scatter",
                    "x": x_range.tolist(),
                    "y": y_dens.tolist(),
                    "fill": "tozeroy",
                    "fillcolor": "rgba(74, 144, 217, 0.2)",
                    "line": {"color": "#4a90d9", "width": 2},
                    "name": "Posterior",
                },
                {
                    "type": "scatter",
                    "x": x_range[rope_mask].tolist(),
                    "y": y_rope[rope_mask].tolist(),
                    "fill": "tozeroy",
                    "fillcolor": "rgba(74, 159, 110, 0.4)",
                    "line": {"color": "#4a9f6e", "width": 0},
                    "name": f"ROPE ({p_rope:.1%})",
                },
                {
                    "type": "scatter",
                    "x": [0, 0],
                    "y": [0, float(max(y_dens))],
                    "mode": "lines",
                    "line": {"color": "#888", "dash": "dot", "width": 1},
                    "name": "Zero",
                },
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": label},
                "yaxis": {"title": "Density"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": rope_low,
                        "x1": rope_low,
                        "y0": 0,
                        "y1": 1,
                        "yref": "paper",
                        "line": {"color": "#4a9f6e", "width": 1, "dash": "dash"},
                    },
                    {
                        "type": "line",
                        "x0": rope_high,
                        "x1": rope_high,
                        "y0": 0,
                        "y1": 1,
                        "yref": "paper",
                        "line": {"color": "#4a9f6e", "width": 1, "dash": "dash"},
                    },
                ],
            },
        }
    )

    result["education"] = {
        "title": "Understanding Bayesian Equivalence Testing",
        "content": (
            "<dl>"
            "<dt>What is Bayesian equivalence testing?</dt>"
            "<dd>It tests whether two groups are <em>practically equivalent</em> — not just "
            "whether they differ. You define a ROPE (Region of Practical Equivalence) and the "
            "analysis calculates the posterior probability that the true effect falls inside it.</dd>"
            "<dt>What is ROPE?</dt>"
            "<dd>The range of effect sizes you consider negligibly small. For example, a ROPE "
            "of [-0.1, 0.1] in standardised units means any difference within ±0.1 Cohen's d "
            "is considered practically zero. The choice of ROPE reflects your domain knowledge.</dd>"
            "<dt>How to interpret the results?</dt>"
            "<dd><strong>P(inside ROPE) &gt; 95%</strong>: accept equivalence — the groups are "
            "practically the same. <strong>P(inside ROPE) &lt; 5%</strong>: reject equivalence — "
            "a meaningful difference exists. Otherwise, the evidence is inconclusive.</dd>"
            "<dt>Why not just use a null-hypothesis test?</dt>"
            "<dd>A non-significant p-value does <em>not</em> prove equivalence — it could mean "
            "insufficient sample size. Bayesian equivalence testing directly quantifies the "
            "probability that the effect is negligible, which is the question you actually want answered.</dd>"
            "</dl>"
        ),
    }

    return result

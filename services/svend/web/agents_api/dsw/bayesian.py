"""DSW Bayesian Analysis — Bayesian inference methods for hypothesis testing."""

import numpy as np
from scipy import stats

from .common import (
    COLOR_BAD,
    COLOR_GOOD,
    COLOR_REFERENCE,
    COLOR_WARNING,
    SVEND_COLORS,
    _narrative,
    _rgba,
)


def run_bayesian_analysis(df, analysis_id, config):
    """Run Bayesian inference analyses - feeds Synara hypothesis testing."""

    result = {"plots": [], "summary": "", "guide_observation": ""}

    ci_level = float(config.get("ci", 0.95))
    z = stats.norm.ppf((1 + ci_level) / 2)

    if analysis_id == "bayes_regression":
        # Bayesian Linear Regression with credible intervals
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler

        target = config.get("target")
        features = config.get("features", [])

        if not target or not features:
            result["summary"] = "Error: Select target and at least one feature"
            return result

        y = df[target].dropna()
        X = df[features].loc[y.index].dropna()
        y = y.loc[X.index]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = BayesianRidge(compute_score=True)
        model.fit(X_scaled, y)

        y_pred, y_std = model.predict(X_scaled, return_std=True)
        coef_mean = model.coef_
        coef_std = np.sqrt(1.0 / model.lambda_) * np.ones_like(coef_mean)
        r2 = model.score(X_scaled, y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>R²:<</COLOR>> {r2:.4f}\n\n"
        summary += f"<<COLOR:text>>Coefficient Posteriors ({int(ci_level * 100)}% Credible Intervals):<</COLOR>>\n\n"

        for i, feat in enumerate(features):
            mean = coef_mean[i]
            std = coef_std[i]
            ci_low = mean - z * std
            ci_high = mean + z * std
            sig = "***" if ci_low > 0 or ci_high < 0 else ""
            summary += f"  {feat:<20} β = {mean:>8.4f}  [{ci_low:>8.4f}, {ci_high:>8.4f}] {sig}\n"

        result["summary"] = summary
        result["plots"].append(
            {
                "title": "Coefficient Posteriors",
                "data": [
                    {
                        "type": "scatter",
                        "x": coef_mean.tolist(),
                        "y": features,
                        "mode": "markers",
                        "marker": {"color": "#4a9f6e", "size": 10},
                        "error_x": {
                            "type": "data",
                            "array": (z * coef_std).tolist(),
                            "color": "#4a9f6e",
                        },
                        "name": f"β ± {int(ci_level * 100)}% CI",
                    }
                ],
                "layout": {
                    "height": max(300, len(features) * 30),
                    "xaxis": {"zeroline": True},
                    "margin": {"l": 150},
                },
            }
        )

        result["synara_weights"] = {
            "analysis_type": "bayesian_regression",
            "target": target,
            "coefficients": [
                {
                    "feature": feat,
                    "mean": float(coef_mean[i]),
                    "ci_low": float(coef_mean[i] - z * coef_std[i]),
                    "ci_high": float(coef_mean[i] + z * coef_std[i]),
                }
                for i, feat in enumerate(features)
            ],
        }

        result["education"] = {
            "title": "Understanding Bayesian Regression",
            "content": (
                "<dl>"
                "<dt>What is Bayesian regression?</dt>"
                "<dd>Like ordinary regression, it estimates how predictors relate to an outcome — "
                "but instead of single point estimates it returns <em>posterior distributions</em> "
                "for each coefficient. You get a full picture of uncertainty, not just a best guess.</dd>"
                "<dt>What are credible intervals?</dt>"
                "<dd>The Bayesian equivalent of confidence intervals. A 95% credible interval means "
                "there is a 95% probability the true coefficient lies within that range — a direct "
                "probability statement that frequentist confidence intervals cannot make.</dd>"
                "<dt>How do I know if a predictor matters?</dt>"
                "<dd>If the credible interval <strong>excludes zero</strong> (marked with ***), "
                "the predictor has a credible effect. If it spans zero, the data does not provide "
                "strong evidence for that predictor.</dd>"
                "<dt>Why Bayesian over ordinary regression?</dt>"
                "<dd>Bayesian regression naturally handles uncertainty in small samples, avoids "
                "overfitting through regularising priors, and gives probabilistic statements about "
                "parameters. Especially valuable when sample sizes are modest or you need to "
                "quantify prediction uncertainty.</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_ttest":
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
            groups = df.groupby(factor_col)[response_col].apply(
                lambda s: s.dropna().values
            )
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

        from scipy.integrate import quad

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
        summary += f"<<COLOR:highlight>>{var2}<</COLOR>> (n={len(x2)}, μ={np.mean(x2):.3f})\n\n"

        summary += "<<COLOR:accent>>── Effect Size ──<</COLOR>>\n"
        summary += f"  Cohen's d: {cohens_d:.3f} [{d_ci_low:.3f}, {d_ci_high:.3f}]\n\n"
        summary += "<<COLOR:accent>>── Bayes Factor ──<</COLOR>>\n"
        summary += f"  BF₁₀: {bf10:.2f}\n\n"

        if bf10 > 10:
            summary += "<<COLOR:success>>Strong evidence for difference (BF₁₀ > 10)<</COLOR>>\n"
        elif bf10 > 3:
            summary += "<<COLOR:warning>>Moderate evidence for difference (BF₁₀ > 3)<</COLOR>>\n"
        elif bf10 > 1:
            summary += (
                "<<COLOR:text>>Weak evidence for difference (BF₁₀ < 3)<</COLOR>>\n"
            )
        else:
            summary += (
                "<<COLOR:text>>Evidence favors no difference (BF₁₀ < 1)<</COLOR>>\n"
            )

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

    elif analysis_id == "bayes_ab":
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
        summary += (
            f"<<COLOR:highlight>>Group A ({g1}):<</COLOR>> {s1}/{n1} = {rate1:.1%}\n"
        )
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

    elif analysis_id == "bayes_correlation":
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
        from scipy.integrate import quad

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

    elif analysis_id == "bayes_anova":
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
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor} ({len(group_names)} levels)\n\n"

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
            summary += "  <<COLOR:text>>Weak evidence for group differences (BF₁₀ < 3)<</COLOR>>\n"
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

    elif analysis_id == "bayes_changepoint":
        # Bayesian change point detection via BIC-approximated Bayes Factors
        var = config.get("var")
        time_col = config.get("time")
        max_cp = int(config.get("max_cp", 2))

        data = df[var].dropna().values
        n = len(data)

        if time_col:
            time_idx = df[time_col].loc[df[var].dropna().index].values
        else:
            time_idx = np.arange(n)

        def _seg_bic(segment):
            """BIC for a segment under N(μ̂, σ̂²): n*log(σ̂²) + 2*log(n)."""
            m = len(segment)
            if m < 2:
                return 0.0
            ss = np.sum((segment - np.mean(segment)) ** 2)
            return m * np.log(max(ss / m, 1e-15)) + 2 * np.log(m)

        # Iteratively find change points by scanning within segments
        min_seg = max(3, n // 20)
        segments = [(0, n)]  # list of (start, end) boundaries
        changepoints = []  # list of (index, bayes_factor)

        for _ in range(max_cp):
            best_bf = 0.0
            best_cp = None
            best_seg_idx = None

            for seg_idx, (seg_start, seg_end) in enumerate(segments):
                seg_data = data[seg_start:seg_end]
                seg_n = len(seg_data)

                if seg_n < 2 * min_seg:
                    continue

                bic_null = _seg_bic(seg_data)

                for tau in range(min_seg, seg_n - min_seg + 1):
                    bic_alt = _seg_bic(seg_data[:tau]) + _seg_bic(seg_data[tau:])
                    bf = np.exp((bic_null - bic_alt) / 2)

                    if bf > best_bf:
                        best_bf = bf
                        best_cp = seg_start + tau
                        best_seg_idx = seg_idx

            if best_cp is not None and best_bf > 3:  # moderate evidence
                changepoints.append((best_cp, best_bf))
                old_start, old_end = segments[best_seg_idx]
                segments[best_seg_idx] = (old_start, best_cp)
                segments.insert(best_seg_idx + 1, (best_cp, old_end))
            else:
                break

        changepoints.sort(key=lambda x: x[0])

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN CHANGE POINT DETECTION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n"
        summary += (
            "<<COLOR:dim>>Method: BIC-approximated Bayes Factor scan<</COLOR>>\n\n"
        )

        if len(changepoints) > 0:
            summary += "<<COLOR:accent>>── Change Points ──<</COLOR>>\n"
            summary += f"  <<COLOR:success>>Detected {len(changepoints)} change point(s)<</COLOR>>\n\n"
            for i, (cp, bf) in enumerate(changepoints):
                before = data[:cp]
                after = data[cp:]
                shift = np.mean(after) - np.mean(before)
                pooled_std = (
                    np.sqrt((np.var(before) + np.var(after)) / 2)
                    if len(before) > 1 and len(after) > 1
                    else 1.0
                )
                effect_d = abs(shift) / pooled_std if pooled_std > 0 else 0.0
                summary += f"  Point {i + 1}: index {cp}\n"
                summary += f"    BF₁₀ = {bf:.1f}"
                summary += f"  |  before μ = {np.mean(before):.4f}, after μ = {np.mean(after):.4f}\n"
                summary += (
                    f"    Shift = {shift:+.4f}  |  Effect size (d) = {effect_d:.2f}\n"
                )
        else:
            summary += "<<COLOR:accent>>── Result ──<</COLOR>>\n"
            summary += "  <<COLOR:text>>No significant change points detected (BF₁₀ < 3)<</COLOR>>\n"

        summary += "\n<<COLOR:accent>>── Interpretation ──<</COLOR>>\n"
        if len(changepoints) > 0:
            best_bf = max(bf for _, bf in changepoints)
            if best_bf > 10:
                summary += "  <<COLOR:success>>Strong evidence for at least one process shift<</COLOR>>\n"
            else:
                summary += "  <<COLOR:warning>>Moderate evidence for process shift(s)<</COLOR>>\n"
        else:
            summary += "  <<COLOR:text>>Process appears stable — no evidence of mean shifts<</COLOR>>\n"

        result["summary"] = summary
        cp_indices = [cp for cp, _ in changepoints]
        cp_bfs = [bf for _, bf in changepoints]
        result["statistics"] = {
            "n_changepoints": len(changepoints),
            "changepoint_indices": cp_indices,
            "bayes_factors": cp_bfs,
        }

        # Guide observation
        if changepoints:
            result["guide_observation"] = (
                f"Bayesian changepoint: {len(changepoints)} shift(s) detected in {var}. Best BF₁₀={max(cp_bfs):.1f}."
            )
        else:
            result["guide_observation"] = (
                f"Bayesian changepoint: no significant shifts detected in {var} (n={n})."
            )

        # Narrative
        if changepoints:
            _bcp_best = max(changepoints, key=lambda x: x[1])
            _bcp_idx, _bcp_bf = _bcp_best
            _bcp_before = data[:_bcp_idx]
            _bcp_after = data[_bcp_idx:]
            _bcp_shift = float(np.mean(_bcp_after) - np.mean(_bcp_before))
            result["narrative"] = _narrative(
                f"Bayesian Changepoint — {len(changepoints)} shift{'s' if len(changepoints) > 1 else ''} detected",
                f"Strongest change at observation {_bcp_idx} (BF\u2081\u2080 = {_bcp_bf:.1f}): mean shifted by {_bcp_shift:+.4f}. "
                + (
                    f"Total of {len(changepoints)} change points in {n} observations."
                    if len(changepoints) > 1
                    else ""
                ),
                next_steps="Investigate what happened at the change point(s). Align with process logs or external events.",
                chart_guidance="Red dashed vertical lines mark detected shifts. Compare the mean level before and after each line.",
            )
        else:
            result["narrative"] = _narrative(
                "Bayesian Changepoint — no shifts detected",
                f"No significant mean shifts found in {n} observations of {var}. The process appears stable.",
                next_steps="If you suspect a shift, try a smaller minimum segment size or add more data.",
            )

        # Time series plot with change points
        plot_data = [
            {
                "type": "scatter",
                "x": (
                    time_idx.tolist() if hasattr(time_idx, "tolist") else list(time_idx)
                ),
                "y": data.tolist(),
                "mode": "lines+markers",
                "marker": {"size": 4, "color": "#4a9f6e"},
                "line": {"color": "#4a9f6e"},
                "name": var,
            }
        ]

        for cp_idx, _cp_bf in changepoints:
            plot_data.append(
                {
                    "type": "scatter",
                    "x": [time_idx[cp_idx], time_idx[cp_idx]],
                    "y": [min(data), max(data)],
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 2},
                    "name": f"Change @ {cp_idx}",
                }
            )

        result["plots"].append(
            {
                "title": "Time Series with Change Points",
                "data": plot_data,
                "layout": {
                    "height": 350,
                    "xaxis": {"title": time_col or "Index"},
                    "yaxis": {"title": var},
                },
            }
        )

        result["education"] = {
            "title": "Understanding Bayesian Changepoint Detection",
            "content": (
                "<dl>"
                "<dt>What is changepoint detection?</dt>"
                "<dd>It identifies the most likely point(s) in a time series where the underlying "
                "process shifted — a change in mean, variance, or both. Unlike control chart alarms "
                "that flag individual points, this pinpoints <em>when</em> the regime changed.</dd>"
                "<dt>How does the Bayesian approach work?</dt>"
                "<dd>For each candidate split point, the model compares two hypotheses: 'one segment' "
                "vs 'two segments with different parameters'. BIC-approximated Bayes Factors rank "
                "all candidate points by evidence strength.</dd>"
                "<dt>What does the Bayes Factor mean here?</dt>"
                "<dd>A higher BF₁₀ at a candidate point means stronger evidence that a real shift "
                "occurred there. <strong>BF₁₀ &gt; 10</strong>: strong evidence of a changepoint. "
                "Multiple changepoints are detected iteratively by segmenting recursively.</dd>"
                "<dt>When to use this?</dt>"
                "<dd>After process interventions (new material, equipment change, shift handover), "
                "to verify whether a suspected change actually occurred and locate it precisely. "
                "Also useful for segmenting historical data into stable regimes before running SPC.</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_proportion":
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

    elif analysis_id == "bayes_capability_prediction":
        """
        Bayesian Capability Prediction — posterior predictive distribution on Cp/Cpk.
        "Not just what IS my Cpk, but what WILL it be after N more samples."
        Uses Normal-Inverse-Chi-Squared conjugate prior for (mu, sigma²).
        """
        var = config.get("var") or config.get("var1")
        lsl = config.get("lsl")
        usl = config.get("usl")

        if lsl is None and usl is None:
            result["summary"] = (
                "Error: Specify at least one spec limit (LSL and/or USL)."
            )
            return result

        lsl = float(lsl) if lsl is not None and lsl != "" else None
        usl = float(usl) if usl is not None and usl != "" else None
        target = float(
            config.get(
                "target",
                (
                    (lsl + usl) / 2
                    if lsl is not None and usl is not None
                    else (lsl if lsl is not None else usl)
                ),
            )
        )

        x = df[var].dropna().values.astype(float)
        n = len(x)
        if n < 3:
            result["summary"] = "Need at least 3 data points."
            return result

        x_bar = float(np.mean(x))
        s2 = float(np.var(x, ddof=1))

        # Vague Normal-Inverse-Chi-Squared prior
        mu_0 = x_bar  # center on data (vague)
        kappa_0 = 0.01  # very low prior weight
        nu_0 = 0.01  # very low prior df
        s2_0 = s2  # prior variance centered on sample

        # Posterior parameters (conjugate update)
        kappa_n = kappa_0 + n
        mu_n = (kappa_0 * mu_0 + n * x_bar) / kappa_n
        nu_n = nu_0 + n
        s2_n = (
            nu_0 * s2_0 + (n - 1) * s2 + (kappa_0 * n * (x_bar - mu_0) ** 2) / kappa_n
        ) / nu_n

        # Draw from posterior predictive for Cpk
        n_draws = 10000
        rng = np.random.default_rng(42)

        # Sample sigma² from Inverse-Chi-Squared(nu_n, s2_n)
        sigma2_draws = nu_n * s2_n / rng.chisquare(nu_n, size=n_draws)
        sigma_draws = np.sqrt(sigma2_draws)

        # Sample mu from Normal(mu_n, sigma²/kappa_n)
        mu_draws = rng.normal(mu_n, np.sqrt(sigma2_draws / kappa_n))

        # Compute Cpk for each draw
        cpk_draws = np.zeros(n_draws)
        cp_draws = np.zeros(n_draws)
        for i in range(n_draws):
            if lsl is not None and usl is not None:
                cp_draws[i] = (usl - lsl) / (6 * sigma_draws[i])
                cpu = (usl - mu_draws[i]) / (3 * sigma_draws[i])
                cpl = (mu_draws[i] - lsl) / (3 * sigma_draws[i])
                cpk_draws[i] = min(cpu, cpl)
            elif usl is not None:
                cp_draws[i] = cpk_draws[i] = (usl - mu_draws[i]) / (3 * sigma_draws[i])
            else:
                cp_draws[i] = cpk_draws[i] = (mu_draws[i] - lsl) / (3 * sigma_draws[i])

        # Credible intervals
        cpk_mean = float(np.mean(cpk_draws))
        cpk_median = float(np.median(cpk_draws))
        cpk_ci = (
            float(np.percentile(cpk_draws, 2.5)),
            float(np.percentile(cpk_draws, 97.5)),
        )
        cp_mean = float(np.mean(cp_draws))

        # P(Cpk > threshold) for common targets
        cpk_targets = [1.0, 1.33, 1.5, 1.67, 2.0]
        prob_above = {t: float(np.mean(cpk_draws > t)) for t in cpk_targets}

        # Predictive: how many more samples to reach 95% confidence Cpk > 1.33?
        _future_ns = [10, 20, 50, 100, 200, 500]
        _future_probs = []
        for fn in _future_ns:
            kappa_f = kappa_n + fn
            nu_f = nu_n + fn
            sigma_f_draws = nu_f * s2_n / rng.chisquare(nu_f, size=3000)
            sigma_f = np.sqrt(sigma_f_draws)
            mu_f = rng.normal(mu_n, np.sqrt(sigma_f_draws / kappa_f))
            if lsl is not None and usl is not None:
                cpu_f = (usl - mu_f) / (3 * sigma_f)
                cpl_f = (mu_f - lsl) / (3 * sigma_f)
                cpk_f = np.minimum(cpu_f, cpl_f)
            elif usl is not None:
                cpk_f = (usl - mu_f) / (3 * sigma_f)
            else:
                cpk_f = (mu_f - lsl) / (3 * sigma_f)
            _future_probs.append(float(np.mean(cpk_f > 1.33)))

        # Summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN CAPABILITY PREDICTION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var} (n = {n})\n"
        if lsl is not None:
            summary += f"<<COLOR:highlight>>LSL:<</COLOR>> {lsl}\n"
        if usl is not None:
            summary += f"<<COLOR:highlight>>USL:<</COLOR>> {usl}\n"
        summary += "\n<<COLOR:accent>>── Posterior Capability ──<</COLOR>>\n"
        summary += f"  Cpk (posterior mean): {cpk_mean:.3f}\n"
        summary += f"  Cpk (posterior median): {cpk_median:.3f}\n"
        summary += f"  95% Credible Interval: [{cpk_ci[0]:.3f}, {cpk_ci[1]:.3f}]\n"
        if lsl is not None and usl is not None:
            summary += f"  Cp (posterior mean): {cp_mean:.3f}\n"

        summary += "\n<<COLOR:accent>>── P(Cpk > target) ──<</COLOR>>\n"
        for t in cpk_targets:
            p = prob_above[t]
            color = "good" if p > 0.9 else ("highlight" if p > 0.5 else "bad")
            summary += f"  <<COLOR:{color}>>P(Cpk > {t:.2f}) = {p:.1%}<</COLOR>>\n"

        summary += "\n<<COLOR:accent>>── Sample Size Forecast ──<</COLOR>>\n"
        summary += "  Additional samples → P(Cpk > 1.33):\n"
        for fn, fp in zip(_future_ns, _future_probs):
            summary += f"    n + {fn:<5} → {fp:.1%}\n"

        result["summary"] = summary

        # Posterior Cpk distribution plot
        result["plots"].append(
            {
                "title": "Posterior Cpk Distribution",
                "data": [
                    {
                        "type": "histogram",
                        "x": cpk_draws.tolist(),
                        "marker": {
                            "color": "rgba(74, 159, 110, 0.4)",
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                        "name": "Posterior Cpk",
                        "nbinsx": 60,
                    }
                ],
                "layout": {
                    "height": 320,
                    "xaxis": {"title": "Cpk"},
                    "yaxis": {"title": "Frequency"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 1.33,
                            "x1": 1.33,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": "#e89547", "dash": "dash", "width": 2},
                        },
                        {
                            "type": "line",
                            "x0": cpk_ci[0],
                            "x1": cpk_ci[0],
                            "y0": 0,
                            "y1": 0.05,
                            "yref": "paper",
                            "line": {"color": "#d94a4a", "width": 2},
                        },
                        {
                            "type": "line",
                            "x0": cpk_ci[1],
                            "x1": cpk_ci[1],
                            "y0": 0,
                            "y1": 0.05,
                            "yref": "paper",
                            "line": {"color": "#d94a4a", "width": 2},
                        },
                    ],
                    "annotations": [
                        {
                            "x": 1.33,
                            "y": 1,
                            "yref": "paper",
                            "text": "Target 1.33",
                            "showarrow": False,
                            "font": {"color": "#e89547", "size": 10},
                        }
                    ],
                },
            }
        )

        # Sample size forecast plot
        result["plots"].append(
            {
                "title": "Sample Size Forecast: P(Cpk > 1.33)",
                "data": [
                    {
                        "type": "scatter",
                        "x": [n + fn for fn in _future_ns],
                        "y": [p * 100 for p in _future_probs],
                        "mode": "lines+markers",
                        "line": {"color": "#4a9f6e", "width": 2},
                        "marker": {"size": 8},
                        "name": "P(Cpk > 1.33)",
                    },
                    {
                        "type": "scatter",
                        "x": [n + _future_ns[0], n + _future_ns[-1]],
                        "y": [95, 95],
                        "mode": "lines",
                        "line": {"color": "#e89547", "dash": "dash"},
                        "name": "95% confidence",
                    },
                ],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": "Total Sample Size"},
                    "yaxis": {"title": "P(Cpk > 1.33) %", "range": [0, 105]},
                },
            }
        )

        # Narrative
        _cap_label = (
            "capable"
            if cpk_mean >= 1.33
            else ("marginally capable" if cpk_mean >= 1.0 else "not capable")
        )
        _confidence_133 = prob_above[1.33]
        result["narrative"] = _narrative(
            f"Process is {_cap_label} (Cpk = {cpk_mean:.3f}, 95% CI [{cpk_ci[0]:.3f}, {cpk_ci[1]:.3f}])",
            f"There is a <strong>{_confidence_133:.0%}</strong> probability that true Cpk exceeds 1.33. "
            + (
                f"The 95% credible interval [{cpk_ci[0]:.3f}, {cpk_ci[1]:.3f}] {'entirely exceeds' if cpk_ci[0] > 1.33 else 'straddles'} the 1.33 target."
                if lsl is not None and usl is not None
                else f"One-sided capability index = {cpk_mean:.3f}."
            ),
            next_steps="The sample size forecast shows how confidence improves with more data. "
            + (
                f"With {_future_ns[2]} more samples, P(Cpk > 1.33) reaches {_future_probs[2]:.0%}."
                if _future_probs[2] < 0.95
                else "Current sample size provides strong confidence."
            ),
            chart_guidance="The histogram shows the posterior belief about Cpk. The dashed line at 1.33 is the typical capability target. Points to the right of this line represent 'capable' outcomes.",
        )

        result["guide_observation"] = (
            f"Bayesian Cpk = {cpk_mean:.3f} (95% CI [{cpk_ci[0]:.3f}, {cpk_ci[1]:.3f}]). P(Cpk > 1.33) = {_confidence_133:.1%}."
        )
        result["statistics"] = {
            "cpk_mean": cpk_mean,
            "cpk_median": cpk_median,
            "cpk_ci_low": cpk_ci[0],
            "cpk_ci_high": cpk_ci[1],
            "cp_mean": cp_mean,
            "prob_above_133": _confidence_133,
            "n": n,
        }

        result["education"] = {
            "title": "Understanding Bayesian Capability Prediction",
            "content": (
                "<dl>"
                "<dt>What is Bayesian Cpk prediction?</dt>"
                "<dd>Traditional Cpk is a point estimate from your current sample. Bayesian Cpk "
                "prediction uses a <em>Normal-Inverse-Chi-Squared</em> conjugate prior to produce "
                "a full posterior distribution on Cpk — telling you the probability that your "
                "process is truly capable, not just that a single number exceeded a threshold.</dd>"
                "<dt>What does P(Cpk &ge; 1.33) mean?</dt>"
                "<dd>The posterior probability that the true process capability exceeds the common "
                "threshold of 1.33. <strong>&gt; 90%</strong>: strong confidence in capability. "
                "<strong>50–90%</strong>: moderate confidence, consider collecting more data. "
                "<strong>&lt; 50%</strong>: the process is more likely incapable than capable.</dd>"
                "<dt>Why not just use the point Cpk?</dt>"
                "<dd>A Cpk of 1.4 from 20 samples is far less certain than 1.4 from 200 samples. "
                "The Bayesian approach captures this uncertainty — the posterior width shrinks "
                "as you add data, giving you an honest picture of how confident the estimate is.</dd>"
                "<dt>What is the predictive distribution?</dt>"
                "<dd>It forecasts what Cpk you would expect after collecting additional samples. "
                "This lets you plan sample sizes: how many more measurements do you need before "
                "the credible interval is narrow enough to make a confident capability decision?</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_equivalence":
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
        summary += f"<<COLOR:highlight>>{var2}<</COLOR>> (n={n2}, \u03bc={np.mean(x2):.4f})\n\n"
        summary += (
            f"<<COLOR:text>>ROPE:<</COLOR>> [{rope_low}, {rope_high}] ({label})\n\n"
        )
        summary += f"<<COLOR:accent>>\u2500\u2500 Posterior on {label} \u2500\u2500<</COLOR>>\n"
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

    elif analysis_id == "bayes_chi2":
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
        cramers_v = (
            float(np.sqrt(chi2_stat / (N * (min(nrow, ncol) - 1)))) if N > 0 else 0
        )

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
        summary += (
            f"  BF\u2081\u2080: {bf10:.2f} ({bf_label} evidence for association)\n"
        )
        summary += f"  Cram\u00e9r's V: {cramers_v:.3f}\n\n"
        summary += (
            "<<COLOR:accent>>\u2500\u2500 Observed Counts \u2500\u2500<</COLOR>>\n"
        )
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
                        "colorscale": [
                            [0, "#4a90d9"],
                            [0.5, "#f5f5f5"],
                            [1, "#dc5050"],
                        ],
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

    elif analysis_id == "bayes_poisson":
        # Bayesian Poisson Rate (Gamma-Poisson conjugate)
        var1 = config.get("var1")
        var2 = config.get("var2")
        exposure_col = config.get("exposure")

        x1 = df[var1].dropna().values.astype(float)
        n1 = len(x1)
        total1 = float(x1.sum())
        exposure1 = (
            float(df[exposure_col].dropna().sum())
            if exposure_col and exposure_col in df.columns
            else float(n1)
        )

        # Gamma posterior: Gamma(alpha + sum(x), beta + exposure)
        # Jeffreys prior: Gamma(0.5, 0)
        alpha_prior, beta_prior = 0.5, 0.0
        a1 = alpha_prior + total1
        b1 = beta_prior + exposure1

        rate1_mean = float(a1 / b1)
        rate1_ci = (
            float(stats.gamma.ppf((1 - ci_level) / 2, a1, scale=1 / b1)),
            float(stats.gamma.ppf((1 + ci_level) / 2, a1, scale=1 / b1)),
        )

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN POISSON RATE<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>{var1}<</COLOR>>: {int(total1)} events in {exposure1:.0f} units\n"
        summary += f"  Posterior rate: {rate1_mean:.4f} [{rate1_ci[0]:.4f}, {rate1_ci[1]:.4f}]\n"

        # Rate range for plotting
        lo_plot = max(0, rate1_mean - 4 * np.sqrt(a1) / b1)
        hi_plot = rate1_mean + 4 * np.sqrt(a1) / b1
        x_range = np.linspace(lo_plot, hi_plot, 200)
        y1_dens = stats.gamma.pdf(x_range, a1, scale=1 / b1)

        plot_data = [
            {
                "type": "scatter",
                "x": x_range.tolist(),
                "y": y1_dens.tolist(),
                "fill": "tozeroy",
                "fillcolor": "rgba(74, 144, 217, 0.3)",
                "line": {"color": "#4a90d9", "width": 2},
                "name": var1,
            },
        ]

        two_sample = var2 and var2 in df.columns
        if two_sample:
            x2 = df[var2].dropna().values.astype(float)
            n2 = len(x2)
            total2 = float(x2.sum())
            exposure2 = (
                float(df[exposure_col].dropna().sum())
                if exposure_col and exposure_col in df.columns
                else float(n2)
            )
            a2 = alpha_prior + total2
            b2 = beta_prior + exposure2
            rate2_mean = float(a2 / b2)
            rate2_ci = (
                float(stats.gamma.ppf((1 - ci_level) / 2, a2, scale=1 / b2)),
                float(stats.gamma.ppf((1 + ci_level) / 2, a2, scale=1 / b2)),
            )

            summary += f"\n<<COLOR:highlight>>{var2}<</COLOR>>: {int(total2)} events in {exposure2:.0f} units\n"
            summary += f"  Posterior rate: {rate2_mean:.4f} [{rate2_ci[0]:.4f}, {rate2_ci[1]:.4f}]\n"

            # P(rate1 > rate2) via Monte Carlo from posteriors
            mc_samples = 50000
            rng = np.random.default_rng(42)
            s1 = stats.gamma.rvs(a1, scale=1 / b1, size=mc_samples, random_state=rng)
            s2 = stats.gamma.rvs(a2, scale=1 / b2, size=mc_samples, random_state=rng)
            p_greater = float(np.mean(s1 > s2))
            rate_ratio_samples = s1 / (s2 + 1e-15)
            rr_mean = float(np.mean(rate_ratio_samples))
            rr_ci = (
                float(np.percentile(rate_ratio_samples, (1 - ci_level) / 2 * 100)),
                float(np.percentile(rate_ratio_samples, (1 + ci_level) / 2 * 100)),
            )

            summary += (
                "\n<<COLOR:accent>>\u2500\u2500 Comparison \u2500\u2500<</COLOR>>\n"
            )
            summary += f"  P({var1} rate > {var2} rate): {p_greater:.1%}\n"
            summary += f"  Rate ratio: {rr_mean:.3f} [{rr_ci[0]:.3f}, {rr_ci[1]:.3f}]\n"

            y2_dens = stats.gamma.pdf(x_range, a2, scale=1 / b2)
            plot_data.append(
                {
                    "type": "scatter",
                    "x": x_range.tolist(),
                    "y": y2_dens.tolist(),
                    "fill": "tozeroy",
                    "fillcolor": "rgba(220, 80, 80, 0.2)",
                    "line": {"color": "#dc5050", "width": 2},
                    "name": var2,
                }
            )

        result["summary"] = summary
        stat_dict = {
            "rate1_mean": rate1_mean,
            "rate1_ci_low": rate1_ci[0],
            "rate1_ci_high": rate1_ci[1],
        }
        if two_sample:
            stat_dict.update(
                {
                    "rate2_mean": rate2_mean,
                    "p_greater": p_greater,
                    "rate_ratio": rr_mean,
                }
            )
        result["statistics"] = stat_dict

        if two_sample:
            result["guide_observation"] = (
                f"Bayesian Poisson: rate\u2081={rate1_mean:.4f}, rate\u2082={rate2_mean:.4f}. P(rate\u2081 > rate\u2082) = {p_greater:.1%}."
            )
            higher = var1 if p_greater > 0.5 else var2
            prob_higher = max(p_greater, 1 - p_greater)
            result["narrative"] = _narrative(
                f"Bayesian Poisson \u2014 P({higher} rate is higher) = {prob_higher:.1%}",
                f"Posterior rates: <strong>{var1}</strong> = {rate1_mean:.4f} (95% CI: {rate1_ci[0]:.4f}\u2013{rate1_ci[1]:.4f}), "
                f"<strong>{var2}</strong> = {rate2_mean:.4f} (95% CI: {rate2_ci[0]:.4f}\u2013{rate2_ci[1]:.4f}). "
                f"Rate ratio = {rr_mean:.3f} (95% CI: {rr_ci[0]:.3f}\u2013{rr_ci[1]:.3f}).",
                next_steps="A rate ratio CI excluding 1.0 confirms the rates differ. "
                "Consider whether the exposure measure adequately accounts for opportunity.",
                chart_guidance="Overlapping posteriors = uncertain which rate is higher. Separated posteriors = clear difference.",
            )
        else:
            result["guide_observation"] = (
                f"Bayesian Poisson: rate = {rate1_mean:.4f} (95% CI: {rate1_ci[0]:.4f}\u2013{rate1_ci[1]:.4f})."
            )
            result["narrative"] = _narrative(
                f"Bayesian Poisson Rate = {rate1_mean:.4f}",
                f"Posterior rate: {rate1_mean:.4f} (95% credible interval: {rate1_ci[0]:.4f} to {rate1_ci[1]:.4f}). "
                f"Based on {int(total1)} events in {exposure1:.0f} exposure units.",
                next_steps="Use this posterior to set Bayesian control limits on count charts or to plan inspection intervals.",
                chart_guidance="The curve shows the posterior belief about the true rate. Width reflects uncertainty \u2014 more data narrows it.",
            )

        result["plots"].append(
            {
                "title": "Posterior Rate Distribution",
                "data": plot_data,
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Rate (\u03bb)"},
                    "yaxis": {"title": "Density"},
                },
            }
        )

        result["education"] = {
            "title": "Understanding Bayesian Poisson Rate Estimation",
            "content": (
                "<dl>"
                "<dt>What is Bayesian Poisson rate estimation?</dt>"
                "<dd>It estimates the true rate of count events (defects per unit, failures per hour, "
                "incidents per shift) using a <em>Gamma-Poisson</em> conjugate model. The result is "
                "a posterior distribution over the rate parameter λ.</dd>"
                "<dt>What is the Gamma-Poisson conjugate?</dt>"
                "<dd>When count data follows a Poisson process and you use a Gamma prior, the "
                "posterior is also Gamma — mathematically exact, no approximation needed. "
                "The prior shape and rate parameters control your initial belief about the rate.</dd>"
                "<dt>What does the credible interval mean?</dt>"
                "<dd>A 95% credible interval on λ directly bounds the true event rate with 95% "
                "probability. If the interval is [2.1, 4.3] defects/unit, you are 95% confident "
                "the true rate is within that range.</dd>"
                "<dt>When to use this?</dt>"
                "<dd>NCR counts per period, defects per lot, customer complaints per month, "
                "equipment failures per 1000 hours — any rare-event count data where you "
                "need to estimate and track the underlying rate with uncertainty.</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_logistic":
        # Bayesian Logistic Regression (Laplace approximation)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        target = config.get("target")
        features = config.get("features", [])

        if not target or not features:
            result["summary"] = (
                "Error: Select a binary target and at least one feature."
            )
            return result

        y = df[target].dropna()
        classes = sorted(y.unique())
        if len(classes) != 2:
            result["summary"] = (
                f"Error: Target must have exactly 2 classes (found {len(classes)})."
            )
            return result

        X = df[features].loc[y.index].dropna()
        y = y.loc[X.index]
        n = len(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # L2 regularization acts as Gaussian prior on coefficients
        prior_width = float(config.get("prior_width", 1.0))
        model = LogisticRegression(C=prior_width, max_iter=1000, solver="lbfgs")
        model.fit(X_scaled, y)

        coef = model.coef_[0]
        model.intercept_[0]
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        accuracy = float(np.mean(model.predict(X_scaled) == y))

        # Laplace approximation: Hessian of log-posterior at MAP
        p_hat = y_pred_proba
        W = p_hat * (1 - p_hat)
        H = X_scaled.T @ np.diag(W) @ X_scaled + (1 / prior_width) * np.eye(
            len(features)
        )
        try:
            cov = np.linalg.inv(H)
            coef_se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            coef_se = np.full(len(features), np.nan)

        # Odds ratios (per std dev of feature)
        odds_ratios = np.exp(coef)
        or_ci_low = np.exp(coef - z * coef_se)
        or_ci_high = np.exp(coef + z * coef_se)

        # P(coefficient > 0)
        p_positive = np.array(
            [
                (
                    float(1 - stats.norm.cdf(0, coef[i], coef_se[i]))
                    if not np.isnan(coef_se[i])
                    else 0.5
                )
                for i in range(len(features))
            ]
        )

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN LOGISTIC REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += (
            f"<<COLOR:text>>Target:<</COLOR>> {target} ({classes[0]} vs {classes[1]})\n"
        )
        summary += f"<<COLOR:text>>N:<</COLOR>> {n}    Accuracy: {accuracy:.1%}\n\n"
        summary += "<<COLOR:accent>>\u2500\u2500 Coefficient Posteriors (per std dev) \u2500\u2500<</COLOR>>\n"
        _bl_header = "P(\u03b2>0)"
        _bl_rule = "\u2500" * 60
        summary += f"  {'Feature':<20} {'OR':>8} {'95% CI':>20} {_bl_header:>8}\n"
        summary += f"  {_bl_rule}\n"

        for i, feat in enumerate(features):
            ci_str = f"[{or_ci_low[i]:.3f}, {or_ci_high[i]:.3f}]"
            sig = "***" if or_ci_low[i] > 1 or or_ci_high[i] < 1 else ""
            summary += f"  {feat:<20} {odds_ratios[i]:>8.3f} {ci_str:>20} {p_positive[i]:>7.1%} {sig}\n"

        result["summary"] = summary
        result["statistics"] = {
            "accuracy": accuracy,
            "n": n,
            "coefficients": {
                feat: {
                    "coef": float(coef[i]),
                    "se": float(coef_se[i]),
                    "or": float(odds_ratios[i]),
                    "p_positive": float(p_positive[i]),
                }
                for i, feat in enumerate(features)
            },
        }

        strongest_idx = int(np.argmax(np.abs(coef)))
        strongest = features[strongest_idx]
        strongest_or = odds_ratios[strongest_idx]
        result["guide_observation"] = (
            f"Bayesian logistic: accuracy={accuracy:.1%}. Strongest predictor: {strongest} (OR={strongest_or:.2f})."
        )

        result["narrative"] = _narrative(
            f"Bayesian Logistic \u2014 accuracy {accuracy:.1%}, strongest predictor: {strongest}",
            f"The model classifies {classes[0]} vs {classes[1]} with {accuracy:.1%} accuracy. "
            f"<strong>{strongest}</strong> has the largest effect (OR = {strongest_or:.2f}, "
            f"P(\u03b2 > 0) = {p_positive[strongest_idx]:.1%}). "
            + (
                f"An OR of {strongest_or:.2f} means a 1-SD increase in {strongest} "
                f"{'increases' if strongest_or > 1 else 'decreases'} the odds by {abs(strongest_or - 1) * 100:.0f}%."
                if not np.isnan(strongest_or)
                else ""
            ),
            next_steps="Odds ratios > 1 increase the probability of the target class. "
            "Features with P(\u03b2>0) near 50% have uncertain direction \u2014 more data needed.",
            chart_guidance="The forest plot shows odds ratios with credible intervals. CIs crossing 1.0 = uncertain effect.",
        )

        # Forest plot of odds ratios
        sorted_idx = np.argsort(np.abs(coef))[::-1]
        result["plots"].append(
            {
                "title": "Odds Ratios (95% Credible Interval)",
                "data": [
                    {
                        "type": "scatter",
                        "y": [features[i] for i in sorted_idx],
                        "x": [float(odds_ratios[i]) for i in sorted_idx],
                        "error_x": {
                            "type": "data",
                            "symmetric": False,
                            "array": [
                                float(or_ci_high[i] - odds_ratios[i])
                                for i in sorted_idx
                            ],
                            "arrayminus": [
                                float(odds_ratios[i] - or_ci_low[i]) for i in sorted_idx
                            ],
                        },
                        "mode": "markers",
                        "marker": {"size": 10, "color": "#4a90d9"},
                    }
                ],
                "layout": {
                    "height": max(200, len(features) * 30 + 80),
                    "xaxis": {"title": "Odds Ratio", "type": "log"},
                    "yaxis": {"autorange": "reversed"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 1,
                            "x1": 1,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": "#888", "dash": "dash"},
                        }
                    ],
                },
            }
        )

        # P(beta > 0) bar chart
        result["plots"].append(
            {
                "title": "P(\u03b2 > 0) per Feature",
                "data": [
                    {
                        "type": "bar",
                        "x": [features[i] for i in sorted_idx],
                        "y": [float(p_positive[i]) for i in sorted_idx],
                        "marker": {
                            "color": [
                                (
                                    "#4a9f6e"
                                    if p_positive[i] > 0.975 or p_positive[i] < 0.025
                                    else (
                                        "#d4a24a"
                                        if p_positive[i] > 0.95 or p_positive[i] < 0.05
                                        else "#999"
                                    )
                                )
                                for i in sorted_idx
                            ]
                        },
                    }
                ],
                "layout": {
                    "height": 250,
                    "yaxis": {"title": "P(\u03b2 > 0)", "range": [0, 1.05]},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": -0.5,
                            "x1": len(features) - 0.5,
                            "y0": 0.5,
                            "y1": 0.5,
                            "line": {"color": "#888", "dash": "dot"},
                        }
                    ],
                },
            }
        )

        result["education"] = {
            "title": "Understanding Bayesian Logistic Regression",
            "content": (
                "<dl>"
                "<dt>What is Bayesian logistic regression?</dt>"
                "<dd>It models binary outcomes (pass/fail, defective/good, yes/no) as a function "
                "of predictors. The Bayesian version produces <em>posterior distributions</em> on "
                "each coefficient, giving you odds ratios with credible intervals.</dd>"
                "<dt>What are odds ratios?</dt>"
                "<dd>An odds ratio of 2.0 means each unit increase in that predictor doubles the "
                "odds of the outcome. <strong>OR &gt; 1</strong>: increases probability. "
                "<strong>OR &lt; 1</strong>: decreases probability. "
                "<strong>OR = 1</strong>: no effect.</dd>"
                "<dt>What does P(β &gt; 0) mean?</dt>"
                "<dd>The posterior probability that a predictor has a positive effect on the outcome. "
                "<strong>&gt; 97.5%</strong> (or <strong>&lt; 2.5%</strong>): credibly significant. "
                "Near 50% means the data cannot determine the direction of effect.</dd>"
                "<dt>Why Bayesian over classical logistic regression?</dt>"
                "<dd>Small samples and rare events make classical maximum-likelihood estimates unstable. "
                "The Bayesian approach regularises through priors, avoids separation problems, and gives "
                "full posterior uncertainty on predictions.</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_survival":
        # Bayesian Weibull Survival Analysis (grid posterior)
        time_col = config.get("var1")
        event_col = config.get("var2")

        times = df[time_col].dropna().values.astype(float)
        if event_col and event_col in df.columns:
            events = (
                df[event_col].loc[df[time_col].notna()].fillna(1).values.astype(float)
            )
        else:
            events = np.ones_like(times)

        mask = times > 0
        times, events = times[mask], events[mask]
        n = len(times)

        if n < 5:
            result["summary"] = "Error: Need at least 5 observations."
            return result

        # Grid posterior over Weibull (shape=beta, scale=eta)
        n_grid = 100
        try:
            shape_mle, _, scale_mle = stats.weibull_min.fit(times, floc=0)
        except Exception:
            shape_mle, scale_mle = 1.0, float(np.median(times))

        beta_range = np.linspace(max(0.1, shape_mle * 0.3), shape_mle * 3.0, n_grid)
        eta_range = np.linspace(max(0.1, scale_mle * 0.3), scale_mle * 3.0, n_grid)
        beta_grid, eta_grid = np.meshgrid(beta_range, eta_range)

        # Log-likelihood on grid
        log_lik = np.zeros_like(beta_grid)
        for i in range(n):
            t_i = times[i]
            e_i = events[i]
            log_lik += e_i * (
                np.log(beta_grid + 1e-15)
                + (beta_grid - 1) * np.log(t_i + 1e-15)
                - beta_grid * np.log(eta_grid + 1e-15)
            )
            log_lik -= (t_i / (eta_grid + 1e-15)) ** beta_grid

        # Flat prior: posterior proportional to likelihood
        log_post = log_lik - log_lik.max()
        posterior = np.exp(log_post)
        posterior /= posterior.sum()

        # Marginal posteriors
        beta_marginal = posterior.sum(axis=0)
        beta_marginal /= beta_marginal.sum()
        eta_marginal = posterior.sum(axis=1)
        eta_marginal /= eta_marginal.sum()

        beta_mean = float(np.sum(beta_range * beta_marginal))
        eta_mean = float(np.sum(eta_range * eta_marginal))
        beta_ci = (
            float(
                beta_range[
                    np.searchsorted(np.cumsum(beta_marginal), (1 - ci_level) / 2)
                ]
            ),
            float(
                beta_range[
                    min(
                        n_grid - 1,
                        np.searchsorted(np.cumsum(beta_marginal), (1 + ci_level) / 2),
                    )
                ]
            ),
        )
        eta_ci = (
            float(
                eta_range[np.searchsorted(np.cumsum(eta_marginal), (1 - ci_level) / 2)]
            ),
            float(
                eta_range[
                    min(
                        n_grid - 1,
                        np.searchsorted(np.cumsum(eta_marginal), (1 + ci_level) / 2),
                    )
                ]
            ),
        )

        # Posterior predictive reliability metrics via MC
        n_mc = 10000
        rng = np.random.default_rng(42)
        beta_idx = rng.choice(n_grid, size=n_mc, p=beta_marginal)
        eta_idx = rng.choice(n_grid, size=n_mc, p=eta_marginal)
        beta_samples = beta_range[beta_idx]
        eta_samples = eta_range[eta_idx]

        # B10 life: t where R(t)=0.90
        b10_samples = eta_samples * ((-np.log(0.9)) ** (1.0 / beta_samples))
        b10_mean = float(np.mean(b10_samples))
        b10_ci = (
            float(np.percentile(b10_samples, (1 - ci_level) / 2 * 100)),
            float(np.percentile(b10_samples, (1 + ci_level) / 2 * 100)),
        )

        # MTTF
        from scipy.special import gamma as gamma_fn

        mttf_samples = eta_samples * gamma_fn(1 + 1.0 / beta_samples)
        mttf_mean = float(np.mean(mttf_samples))
        mttf_ci = (
            float(np.percentile(mttf_samples, (1 - ci_level) / 2 * 100)),
            float(np.percentile(mttf_samples, (1 + ci_level) / 2 * 100)),
        )

        if beta_mean < 0.95:
            phase = "infant mortality (\u03b2 < 1)"
        elif beta_mean <= 1.05:
            phase = "constant failure rate (\u03b2 \u2248 1)"
        else:
            phase = "wear-out (\u03b2 > 1)"

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN WEIBULL SURVIVAL<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Observations:<</COLOR>> {n} ({int(events.sum())} events, {int(n - events.sum())} censored)\n\n"
        summary += "<<COLOR:accent>>\u2500\u2500 Shape (\u03b2) Posterior \u2500\u2500<</COLOR>>\n"
        summary += f"  Mean: {beta_mean:.3f}  [{beta_ci[0]:.3f}, {beta_ci[1]:.3f}]\n"
        summary += f"  Phase: {phase}\n\n"
        summary += "<<COLOR:accent>>\u2500\u2500 Scale (\u03b7) Posterior \u2500\u2500<</COLOR>>\n"
        summary += f"  Mean: {eta_mean:.2f}  [{eta_ci[0]:.2f}, {eta_ci[1]:.2f}]\n\n"
        summary += (
            "<<COLOR:accent>>\u2500\u2500 Reliability Metrics \u2500\u2500<</COLOR>>\n"
        )
        summary += f"  B10 Life: {b10_mean:.2f}  [{b10_ci[0]:.2f}, {b10_ci[1]:.2f}]\n"
        summary += f"  MTTF:    {mttf_mean:.2f}  [{mttf_ci[0]:.2f}, {mttf_ci[1]:.2f}]\n"

        result["summary"] = summary
        result["statistics"] = {
            "beta_mean": beta_mean,
            "beta_ci": list(beta_ci),
            "eta_mean": eta_mean,
            "eta_ci": list(eta_ci),
            "b10_mean": b10_mean,
            "b10_ci": list(b10_ci),
            "mttf_mean": mttf_mean,
            "mttf_ci": list(mttf_ci),
            "phase": phase,
            "n": n,
            "n_events": int(events.sum()),
        }
        result["guide_observation"] = (
            f"Bayesian Weibull: \u03b2={beta_mean:.2f} ({phase}), \u03b7={eta_mean:.1f}, B10={b10_mean:.1f}, MTTF={mttf_mean:.1f}."
        )

        result["narrative"] = _narrative(
            f"Bayesian Weibull \u2014 {phase}, B10 = {b10_mean:.1f}",
            f"Shape \u03b2 = {beta_mean:.3f} (95% CI: {beta_ci[0]:.3f}\u2013{beta_ci[1]:.3f}) indicates <strong>{phase}</strong>. "
            f"Scale \u03b7 = {eta_mean:.1f} (characteristic life). "
            f"B10 life = {b10_mean:.1f} (95% CI: {b10_ci[0]:.1f}\u2013{b10_ci[1]:.1f}), "
            f"MTTF = {mttf_mean:.1f} (95% CI: {mttf_ci[0]:.1f}\u2013{mttf_ci[1]:.1f}).",
            next_steps="The credible intervals on B10 and MTTF give honest uncertainty bounds \u2014 "
            "use these for warranty planning instead of point estimates.",
            chart_guidance="The posterior survival curve shows the credible band around reliability. "
            "Width = epistemic uncertainty that shrinks with more data.",
        )

        # Posterior survival curve with credible band
        t_plot = np.linspace(0, float(np.percentile(times, 99)) * 1.5, 100)
        surv_curves = []
        for j in range(min(n_mc, 2000)):
            surv_curves.append(np.exp(-((t_plot / eta_samples[j]) ** beta_samples[j])))
        surv_matrix = np.array(surv_curves)
        surv_mean = np.mean(surv_matrix, axis=0)
        surv_lo = np.percentile(surv_matrix, 2.5, axis=0)
        surv_hi = np.percentile(surv_matrix, 97.5, axis=0)

        result["plots"].append(
            {
                "title": "Posterior Predictive Survival Curve",
                "data": [
                    {
                        "type": "scatter",
                        "x": t_plot.tolist(),
                        "y": surv_hi.tolist(),
                        "line": {"width": 0},
                        "showlegend": False,
                        "name": "Upper 95%",
                    },
                    {
                        "type": "scatter",
                        "x": t_plot.tolist(),
                        "y": surv_lo.tolist(),
                        "fill": "tonexty",
                        "fillcolor": "rgba(74, 159, 110, 0.2)",
                        "line": {"width": 0},
                        "name": "95% Credible Band",
                    },
                    {
                        "type": "scatter",
                        "x": t_plot.tolist(),
                        "y": surv_mean.tolist(),
                        "line": {"color": "#4a9f6e", "width": 2},
                        "name": "Posterior Mean",
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Reliability R(t)", "range": [0, 1.05]},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 0,
                            "x1": float(t_plot[-1]),
                            "y0": 0.9,
                            "y1": 0.9,
                            "line": {"color": "#d4a24a", "dash": "dot"},
                        }
                    ],
                },
            }
        )

        # Shape parameter posterior
        result["plots"].append(
            {
                "title": "Shape Parameter (\u03b2) Posterior",
                "data": [
                    {
                        "type": "scatter",
                        "x": beta_range.tolist(),
                        "y": beta_marginal.tolist(),
                        "fill": "tozeroy",
                        "fillcolor": "rgba(74, 144, 217, 0.3)",
                        "line": {"color": "#4a90d9", "width": 2},
                        "name": "\u03b2 Posterior",
                    },
                    {
                        "type": "scatter",
                        "x": [1, 1],
                        "y": [0, float(max(beta_marginal))],
                        "mode": "lines",
                        "line": {"color": "#888", "dash": "dash"},
                        "name": "\u03b2=1 (exponential)",
                    },
                ],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": "Shape \u03b2"},
                    "yaxis": {"title": "Density"},
                },
            }
        )

        result["education"] = {
            "title": "Understanding Bayesian Weibull Survival Analysis",
            "content": (
                "<dl>"
                "<dt>What is Bayesian survival analysis?</dt>"
                "<dd>It models time-to-event data (time to failure, time to repair) using a "
                "<em>Weibull distribution</em> with Bayesian parameter estimation. You get "
                "posterior distributions on the shape and scale parameters, plus credible "
                "intervals on survival probabilities and percentile life estimates.</dd>"
                "<dt>What does the shape parameter (β) tell you?</dt>"
                "<dd><strong>β &lt; 1</strong>: decreasing failure rate (infant mortality). "
                "<strong>β = 1</strong>: constant failure rate (exponential, random failures). "
                "<strong>β &gt; 1</strong>: increasing failure rate (wear-out). The dashed line "
                "at β=1 on the posterior plot marks the exponential boundary.</dd>"
                "<dt>What are censored observations?</dt>"
                "<dd>Units that were removed from test or had not yet failed when observation ended. "
                "The Bayesian model properly accounts for censoring — ignoring it would bias "
                "the survival estimates downward.</dd>"
                "<dt>Why Bayesian over Kaplan-Meier?</dt>"
                "<dd>Kaplan-Meier is nonparametric and makes no distributional assumptions, but "
                "cannot extrapolate beyond the data. The Bayesian Weibull model gives probabilistic "
                "predictions of future failures and honest uncertainty on life percentiles (B10, B50).</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_meta":
        # Bayesian Random-Effects Meta-Analysis (Normal-Normal hierarchical)
        effects_col = config.get("var1")
        se_col = config.get("var2")

        y = df[effects_col].dropna().values.astype(float)
        se = df[se_col].dropna().values.astype(float)
        k = len(y)

        if k < 2:
            result["summary"] = "Error: Need at least 2 studies."
            return result

        n_use = min(len(y), len(se))
        y, se = y[:n_use], se[:n_use]
        k = n_use

        # Grid posterior over tau (between-study SD)
        tau_max = max(3 * np.std(y), 1.0)
        n_tau = 200
        tau_range = np.linspace(0, tau_max, n_tau)
        log_marginal = np.zeros(n_tau)

        for i, tau in enumerate(tau_range):
            w = 1.0 / (se**2 + tau**2 + 1e-15)
            mu_hat = np.sum(w * y) / np.sum(w)
            var_total = se**2 + tau**2
            log_marginal[i] = -0.5 * np.sum(
                np.log(2 * np.pi * var_total) + (y - mu_hat) ** 2 / var_total
            )

        # Posterior on tau (flat prior)
        log_post_tau = log_marginal - log_marginal.max()
        post_tau = np.exp(log_post_tau)
        post_tau /= post_tau.sum() * (tau_range[1] - tau_range[0])

        tau_pmf = post_tau * (tau_range[1] - tau_range[0])
        tau_pmf /= tau_pmf.sum()
        tau_mean = float(np.sum(tau_range * tau_pmf))
        tau_cdf = np.cumsum(tau_pmf)
        tau_ci = (
            float(tau_range[np.searchsorted(tau_cdf, (1 - ci_level) / 2)]),
            float(
                tau_range[min(n_tau - 1, np.searchsorted(tau_cdf, (1 + ci_level) / 2))]
            ),
        )

        # Posterior on mu (integrate over tau)
        n_mc = 10000
        rng = np.random.default_rng(42)
        tau_samples = tau_range[rng.choice(n_tau, size=n_mc, p=tau_pmf)]
        mu_samples = np.zeros(n_mc)
        for j in range(n_mc):
            w = 1.0 / (se**2 + tau_samples[j] ** 2 + 1e-15)
            mu_hat = np.sum(w * y) / np.sum(w)
            mu_se = 1.0 / np.sqrt(np.sum(w))
            mu_samples[j] = rng.normal(mu_hat, mu_se)

        mu_mean = float(np.mean(mu_samples))
        mu_ci = (
            float(np.percentile(mu_samples, (1 - ci_level) / 2 * 100)),
            float(np.percentile(mu_samples, (1 + ci_level) / 2 * 100)),
        )

        # Study-specific shrunken estimates
        shrunk = []
        for j_study in range(k):
            w_j = 1.0 / (se[j_study] ** 2 + tau_mean**2 + 1e-15)
            w_pool = np.sum(1.0 / (se**2 + tau_mean**2 + 1e-15))
            shrink_factor = w_j / (w_j + 1.0 / (1.0 / w_pool + 1e-15))
            est = shrink_factor * y[j_study] + (1 - shrink_factor) * mu_mean
            shrunk.append(float(est))

        i2 = (
            float(tau_mean**2 / (tau_mean**2 + np.mean(se**2)) * 100)
            if tau_mean > 0
            else 0.0
        )
        het_label = (
            "high"
            if i2 > 75
            else ("moderate" if i2 > 50 else ("low" if i2 > 25 else "negligible"))
        )

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN META-ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Studies:<</COLOR>> {k}\n\n"
        summary += "<<COLOR:accent>>\u2500\u2500 Pooled Effect (\u03bc) \u2500\u2500<</COLOR>>\n"
        summary += f"  Posterior mean: {mu_mean:.4f}\n"
        summary += f"  95% Credible Interval: [{mu_ci[0]:.4f}, {mu_ci[1]:.4f}]\n\n"
        summary += "<<COLOR:accent>>\u2500\u2500 Heterogeneity (\u03c4) \u2500\u2500<</COLOR>>\n"
        summary += f"  Posterior mean: {tau_mean:.4f}\n"
        summary += f"  95% CI: [{tau_ci[0]:.4f}, {tau_ci[1]:.4f}]\n"
        summary += f"  I\u00b2 analog: {i2:.1f}% ({het_label})\n\n"
        summary += (
            "<<COLOR:accent>>\u2500\u2500 Study Estimates \u2500\u2500<</COLOR>>\n"
        )
        for j_study in range(k):
            summary += f"  Study {j_study + 1}: {y[j_study]:.4f} \u00b1 {se[j_study]:.4f} \u2192 shrunk {shrunk[j_study]:.4f}\n"

        result["summary"] = summary
        result["statistics"] = {
            "mu_mean": mu_mean,
            "mu_ci": list(mu_ci),
            "tau_mean": tau_mean,
            "tau_ci": list(tau_ci),
            "i2": i2,
            "k": k,
            "shrunk_estimates": shrunk,
        }
        result["guide_observation"] = (
            f"Bayesian meta-analysis ({k} studies): pooled = {mu_mean:.4f} (95% CI: {mu_ci[0]:.4f}\u2013{mu_ci[1]:.4f}), \u03c4 = {tau_mean:.4f}, I\u00b2 = {i2:.0f}%."
        )

        result["narrative"] = _narrative(
            f"Bayesian Meta-Analysis \u2014 pooled effect = {mu_mean:.4f}, I\u00b2 = {i2:.0f}% ({het_label})",
            f"Across {k} studies, the pooled effect is {mu_mean:.4f} (95% credible interval: {mu_ci[0]:.4f} to {mu_ci[1]:.4f}). "
            f"Between-study heterogeneity \u03c4 = {tau_mean:.4f} (I\u00b2 \u2248 {i2:.0f}%, {het_label}). "
            + (
                "The CI excludes zero, supporting a real effect."
                if (mu_ci[0] > 0 or mu_ci[1] < 0)
                else "The CI includes zero \u2014 the overall effect is uncertain."
            ),
            next_steps="High I\u00b2 means the studies disagree. Investigate moderators (subgroup analysis) to explain the heterogeneity. "
            "The shrunken estimates show how each study's estimate is pulled toward the grand mean.",
            chart_guidance="The forest plot shows each study's estimate (with CI) and the shrunken Bayesian estimate. "
            "The diamond at the bottom is the pooled posterior.",
        )

        # Forest plot
        study_labels = [f"Study {i + 1}" for i in range(k)]
        result["plots"].append(
            {
                "title": "Bayesian Forest Plot",
                "data": [
                    {
                        "type": "scatter",
                        "y": study_labels,
                        "x": y.tolist(),
                        "error_x": {"type": "data", "array": (z * se).tolist()},
                        "mode": "markers",
                        "marker": {"size": 8, "color": "#4a90d9"},
                        "name": "Observed",
                    },
                    {
                        "type": "scatter",
                        "y": study_labels,
                        "x": shrunk,
                        "mode": "markers",
                        "marker": {"size": 8, "symbol": "diamond", "color": "#d4a24a"},
                        "name": "Shrunken",
                    },
                    {
                        "type": "scatter",
                        "y": ["Pooled"],
                        "x": [mu_mean],
                        "error_x": {
                            "type": "data",
                            "symmetric": False,
                            "array": [mu_ci[1] - mu_mean],
                            "arrayminus": [mu_mean - mu_ci[0]],
                        },
                        "mode": "markers",
                        "marker": {"size": 14, "symbol": "diamond", "color": "#4a9f6e"},
                        "name": "Pooled",
                    },
                ],
                "layout": {
                    "height": max(200, k * 35 + 100),
                    "xaxis": {"title": "Effect Size"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 0,
                            "x1": 0,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": "#888", "dash": "dash"},
                        }
                    ],
                },
            }
        )

        # Posterior on tau
        result["plots"].append(
            {
                "title": "Posterior on Heterogeneity (\u03c4)",
                "data": [
                    {
                        "type": "scatter",
                        "x": tau_range.tolist(),
                        "y": (post_tau * (tau_range[1] - tau_range[0])).tolist(),
                        "fill": "tozeroy",
                        "fillcolor": "rgba(212, 162, 74, 0.3)",
                        "line": {"color": "#d4a24a", "width": 2},
                        "name": "\u03c4 Posterior",
                    }
                ],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": "Between-study SD (\u03c4)"},
                    "yaxis": {"title": "Density"},
                },
            }
        )

        result["education"] = {
            "title": "Understanding Bayesian Meta-Analysis",
            "content": (
                "<dl>"
                "<dt>What is Bayesian meta-analysis?</dt>"
                "<dd>It combines effect sizes from multiple studies into a single pooled estimate "
                "using a <em>Normal-Normal hierarchical</em> model. Each study contributes in "
                "proportion to its precision, and between-study heterogeneity (τ) is estimated.</dd>"
                "<dt>What is τ (tau)?</dt>"
                "<dd>The between-study standard deviation — how much true effect sizes vary across "
                "studies. <strong>τ ≈ 0</strong>: studies agree, fixed-effect model suffices. "
                "<strong>τ large</strong>: substantial heterogeneity, the pooled estimate is less "
                "certain and individual study contexts matter more.</dd>"
                "<dt>What is the pooled effect?</dt>"
                "<dd>The posterior mean of the overall effect size, accounting for both within-study "
                "uncertainty and between-study variability. Its credible interval is wider than "
                "any individual study because it honestly propagates heterogeneity.</dd>"
                "<dt>When to use this?</dt>"
                "<dd>Combining results across multiple experiments, plants, or time periods. "
                "For example: pooling Cpk estimates from 5 production lines, combining treatment "
                "effects from repeated trials, or synthesising defect rate studies.</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_demo":
        # Bayesian Reliability Demonstration (Beta-Binomial conjugate)
        n_tested = int(config.get("n_tested", 50))
        n_failures = int(config.get("n_failures", 0))
        target_r = float(config.get("target_reliability", 0.99))
        prior_a = float(config.get("prior_a", 1.0))
        prior_b = float(config.get("prior_b", 1.0))

        n_success = n_tested - n_failures
        post_a = prior_a + n_success
        post_b = prior_b + n_failures

        r_vals = np.linspace(0, 1, 500)
        prior_pdf = stats.beta.pdf(r_vals, prior_a, prior_b)
        post_pdf = stats.beta.pdf(r_vals, post_a, post_b)

        post_mean = float(post_a / (post_a + post_b))
        post_ci = (
            float(stats.beta.ppf((1 - ci_level) / 2, post_a, post_b)),
            float(stats.beta.ppf((1 + ci_level) / 2, post_a, post_b)),
        )
        prob_exceed = float(1 - stats.beta.cdf(target_r, post_a, post_b))

        verdict = "PASS" if prob_exceed >= 0.5 else "FAIL"

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN RELIABILITY DEMONSTRATION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Units tested:<</COLOR>> {n_tested}\n"
        summary += f"<<COLOR:highlight>>Failures:<</COLOR>> {n_failures}\n"
        summary += f"<<COLOR:highlight>>Target reliability:<</COLOR>> {target_r}\n\n"
        summary += f"<<COLOR:highlight>>Posterior mean R:<</COLOR>> {post_mean:.4f}\n"
        summary += f"<<COLOR:highlight>>{int(ci_level * 100)}% Credible interval:<</COLOR>> [{post_ci[0]:.4f}, {post_ci[1]:.4f}]\n"
        summary += f"<<COLOR:highlight>>P(R \u2265 {target_r}):<</COLOR>> {prob_exceed:.4f}\n\n"

        if prob_exceed >= 0.95:
            summary += "<<COLOR:good>>Strong evidence that reliability meets target.<</COLOR>>\n"
        elif prob_exceed >= 0.5:
            summary += "<<COLOR:warning>>Moderate evidence for reliability target \u2014 consider more testing.<</COLOR>>\n"
        else:
            summary += "<<COLOR:bad>>Insufficient evidence that reliability meets target.<</COLOR>>\n"

        result["summary"] = summary
        result["statistics"] = {
            "posterior_mean": post_mean,
            "credible_interval": list(post_ci),
            "prob_exceed_target": prob_exceed,
            "prior_alpha": prior_a,
            "prior_beta": prior_b,
            "posterior_alpha": float(post_a),
            "posterior_beta": float(post_b),
        }
        result["guide_observation"] = (
            f"Bayesian demo: P(R\u2265{target_r})={prob_exceed:.3f}, posterior mean={post_mean:.4f}"
        )
        result["narrative"] = _narrative(
            verdict,
            f"With {n_tested} units tested and {n_failures} failures, the posterior probability "
            f"that reliability meets {target_r} is {prob_exceed:.1%}. "
            f"Posterior mean reliability = {post_mean:.4f} ({int(ci_level * 100)}% CI: {post_ci[0]:.4f}\u2013{post_ci[1]:.4f}).",
            [
                "Increase sample size to narrow credible interval",
                "Use sequential updating as more units complete testing",
                "Consider informative prior from similar products",
            ],
        )

        # Plot 1: Prior vs Posterior
        result["plots"].append(
            {
                "title": "Prior vs Posterior on Reliability",
                "data": [
                    {
                        "type": "scatter",
                        "x": r_vals.tolist(),
                        "y": prior_pdf.tolist(),
                        "line": {"color": COLOR_REFERENCE, "dash": "dash", "width": 2},
                        "name": "Prior",
                    },
                    {
                        "type": "scatter",
                        "x": r_vals.tolist(),
                        "y": post_pdf.tolist(),
                        "fill": "tozeroy",
                        "fillcolor": _rgba(SVEND_COLORS[0], 0.3),
                        "line": {"color": SVEND_COLORS[0], "width": 2},
                        "name": "Posterior",
                    },
                ],
                "layout": {
                    "height": 350,
                    "xaxis": {"title": "Reliability (R)"},
                    "yaxis": {"title": "Density"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": target_r,
                            "x1": target_r,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": COLOR_BAD, "dash": "dot", "width": 2},
                        }
                    ],
                    "annotations": [
                        {
                            "x": target_r,
                            "y": 1,
                            "yref": "paper",
                            "text": f"Target={target_r}",
                            "showarrow": False,
                            "yanchor": "bottom",
                            "font": {"color": COLOR_BAD},
                        }
                    ],
                },
            }
        )

        # Plot 2: Exceedance curve
        exceedance = np.array([1 - stats.beta.cdf(r, post_a, post_b) for r in r_vals])
        result["plots"].append(
            {
                "title": "Exceedance Probability P(R > r)",
                "data": [
                    {
                        "type": "scatter",
                        "x": r_vals.tolist(),
                        "y": exceedance.tolist(),
                        "fill": "tozeroy",
                        "fillcolor": _rgba(SVEND_COLORS[1], 0.2),
                        "line": {"color": SVEND_COLORS[1], "width": 2},
                        "name": "P(R > r)",
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Reliability Threshold (r)"},
                    "yaxis": {"title": "Probability"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": target_r,
                            "x1": target_r,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": COLOR_BAD, "dash": "dot"},
                        },
                        {
                            "type": "line",
                            "x0": 0,
                            "x1": 1,
                            "y0": prob_exceed,
                            "y1": prob_exceed,
                            "line": {"color": COLOR_WARNING, "dash": "dash"},
                        },
                    ],
                },
            }
        )

        # Plot 3: Sequential posterior update
        seq_means, seq_lo, seq_hi = [], [], []
        for i in range(1, n_tested + 1):
            fail_so_far = int(round(n_failures * i / n_tested))
            succ_so_far = i - fail_so_far
            a_i = prior_a + succ_so_far
            b_i = prior_b + fail_so_far
            seq_means.append(float(a_i / (a_i + b_i)))
            seq_lo.append(float(stats.beta.ppf(0.025, a_i, b_i)))
            seq_hi.append(float(stats.beta.ppf(0.975, a_i, b_i)))
        units_x = list(range(1, n_tested + 1))
        result["plots"].append(
            {
                "title": "Sequential Posterior Update",
                "data": [
                    {
                        "type": "scatter",
                        "x": units_x,
                        "y": seq_hi,
                        "line": {"width": 0},
                        "showlegend": False,
                        "name": "Upper",
                    },
                    {
                        "type": "scatter",
                        "x": units_x,
                        "y": seq_lo,
                        "fill": "tonexty",
                        "fillcolor": _rgba(SVEND_COLORS[0], 0.2),
                        "line": {"width": 0},
                        "name": "95% CI",
                    },
                    {
                        "type": "scatter",
                        "x": units_x,
                        "y": seq_means,
                        "line": {"color": SVEND_COLORS[0], "width": 2},
                        "name": "Posterior Mean",
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Units Tested"},
                    "yaxis": {"title": "Reliability Estimate"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 1,
                            "x1": n_tested,
                            "y0": target_r,
                            "y1": target_r,
                            "line": {"color": COLOR_BAD, "dash": "dot"},
                        }
                    ],
                },
            }
        )

        result["education"] = {
            "title": "Understanding Bayesian Reliability Demonstration",
            "content": (
                "<dl>"
                "<dt>What is reliability demonstration?</dt>"
                "<dd>It answers: 'given N units tested with F failures, what is the probability "
                "that the true reliability meets or exceeds a target?' Uses a <em>Beta-Binomial</em> "
                "conjugate model to produce a posterior distribution on the reliability parameter.</dd>"
                "<dt>What does P(R ≥ target) mean?</dt>"
                "<dd>The posterior probability that the true reliability meets your target. "
                "<strong>&gt; 90%</strong>: strong confidence the target is met. "
                "<strong>50–90%</strong>: encouraging but more testing may be warranted. "
                "<strong>&lt; 50%</strong>: the product likely does not meet the target.</dd>"
                "<dt>How does sample size affect the result?</dt>"
                "<dd>The sequential plot shows how the posterior reliability estimate evolves as "
                "more units are tested. With zero failures, each additional test unit tightens the "
                "credible interval and increases confidence. Even one failure can significantly "
                "shift the posterior.</dd>"
                "<dt>When to use this?</dt>"
                "<dd>Qualification testing, lot acceptance, warranty requirement verification — "
                "whenever you need to demonstrate that a product meets a reliability target "
                "with quantified confidence.</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_spares":
        # Bayesian Spare Parts Planning (Gamma-Poisson conjugate)
        demand_col = config.get("var1")
        planning_horizon = float(config.get("planning_horizon", 12))
        service_level = float(config.get("service_level", 0.95))
        holding_cost = float(config.get("holding_cost", 10))
        stockout_cost = float(config.get("stockout_cost", 100))
        prior_a = float(config.get("prior_a", 1.0))
        prior_b = float(config.get("prior_b", 1.0))

        if demand_col and demand_col in df.columns:
            demands = df[demand_col].dropna().values.astype(float)
            total_demand = float(demands.sum())
            n_periods = len(demands)
        else:
            total_demand = float(config.get("total_demand", 10))
            n_periods = float(config.get("n_periods", 12))

        post_a = prior_a + total_demand
        post_b = prior_b + n_periods
        rate_mean = float(post_a / post_b)
        rate_ci = (
            float(stats.gamma.ppf((1 - ci_level) / 2, post_a, scale=1 / post_b)),
            float(stats.gamma.ppf((1 + ci_level) / 2, post_a, scale=1 / post_b)),
        )

        expected_demand = rate_mean * planning_horizon
        nb_r = post_a
        nb_p = float(post_b / (post_b + planning_horizon))
        max_stock = int(expected_demand * 3) + 50
        stock_range = np.arange(0, max_stock + 1)
        cdf_vals = stats.nbinom.cdf(stock_range, nb_r, nb_p)

        optimal_idx = np.searchsorted(cdf_vals, service_level)
        optimal_stock = int(stock_range[min(optimal_idx, len(stock_range) - 1)])
        p_stockout = float(1 - stats.nbinom.cdf(optimal_stock, nb_r, nb_p))

        stock_eval = np.arange(0, optimal_stock * 2 + 10)
        costs = []
        d_vals = np.arange(0, int(expected_demand * 4) + 1)
        pmf_vals = stats.nbinom.pmf(d_vals, nb_r, nb_p)
        for s in stock_eval:
            hold = holding_cost * float(np.sum(np.maximum(0, s - d_vals) * pmf_vals))
            out = stockout_cost * float(np.sum(np.maximum(0, d_vals - s) * pmf_vals))
            costs.append(hold + out)

        min_cost = float(min(costs))
        min_cost_stock = int(stock_eval[np.argmin(costs)])

        verdict = "PASS" if p_stockout <= (1 - service_level) else "WARNING"
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN SPARE PARTS PLANNING<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Historical demand:<</COLOR>> {total_demand:.0f} over {n_periods:.0f} periods\n"
        summary += f"<<COLOR:highlight>>Planning horizon:<</COLOR>> {planning_horizon:.0f} periods\n"
        summary += f"<<COLOR:highlight>>Service level target:<</COLOR>> {service_level:.1%}\n\n"
        summary += f"<<COLOR:highlight>>Posterior demand rate:<</COLOR>> {rate_mean:.2f}/period ({int(ci_level * 100)}% CI: {rate_ci[0]:.2f}\u2013{rate_ci[1]:.2f})\n"
        summary += f"<<COLOR:highlight>>Expected demand (horizon):<</COLOR>> {expected_demand:.1f}\n"
        summary += f"<<COLOR:highlight>>Optimal stock level:<</COLOR>> <<COLOR:good>>{optimal_stock}<</COLOR>>\n"
        summary += f"<<COLOR:highlight>>P(stockout):<</COLOR>> {p_stockout:.4f}\n"
        summary += f"<<COLOR:highlight>>Cost-optimal stock:<</COLOR>> {min_cost_stock} (total cost: {min_cost:.0f})\n"

        result["summary"] = summary
        result["statistics"] = {
            "rate_mean": rate_mean,
            "rate_ci": list(rate_ci),
            "expected_demand": expected_demand,
            "optimal_stock": optimal_stock,
            "p_stockout": p_stockout,
            "min_cost": min_cost,
            "min_cost_stock": min_cost_stock,
        }
        result["guide_observation"] = (
            f"Bayes spares: optimal stock={optimal_stock} for {service_level:.0%} SL, rate={rate_mean:.2f}/period"
        )
        result["narrative"] = _narrative(
            verdict,
            f"Demand rate posterior: {rate_mean:.2f}/period (CI: {rate_ci[0]:.2f}\u2013{rate_ci[1]:.2f}). "
            f"For a {planning_horizon:.0f}-period horizon at {service_level:.0%} service level, "
            f"stock {optimal_stock} units. P(stockout) = {p_stockout:.4f}.",
            [
                "Review holding vs stockout cost trade-off on the cost curve",
                "Update posterior as new demand data arrives",
                "Consider safety stock adjustment for demand seasonality",
            ],
        )

        rate_vals = np.linspace(0, rate_ci[1] * 2, 300)
        prior_pdf_r = stats.gamma.pdf(rate_vals, prior_a, scale=1 / prior_b)
        post_pdf_r = stats.gamma.pdf(rate_vals, post_a, scale=1 / post_b)
        result["plots"].append(
            {
                "title": "Prior vs Posterior on Demand Rate",
                "data": [
                    {
                        "type": "scatter",
                        "x": rate_vals.tolist(),
                        "y": prior_pdf_r.tolist(),
                        "line": {"color": COLOR_REFERENCE, "dash": "dash", "width": 2},
                        "name": "Prior",
                    },
                    {
                        "type": "scatter",
                        "x": rate_vals.tolist(),
                        "y": post_pdf_r.tolist(),
                        "fill": "tozeroy",
                        "fillcolor": _rgba(SVEND_COLORS[0], 0.3),
                        "line": {"color": SVEND_COLORS[0], "width": 2},
                        "name": "Posterior",
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Demand Rate (\u03bb)"},
                    "yaxis": {"title": "Density"},
                },
            }
        )

        sl_plot_max = min(optimal_stock * 2 + 10, len(cdf_vals))
        result["plots"].append(
            {
                "title": "Service Level vs Stock Level",
                "data": [
                    {
                        "type": "scatter",
                        "x": stock_range[:sl_plot_max].tolist(),
                        "y": cdf_vals[:sl_plot_max].tolist(),
                        "line": {"color": SVEND_COLORS[1], "width": 2},
                        "name": "Service Level",
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Stock Level"},
                    "yaxis": {"title": "Service Level P(D \u2264 S)"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 0,
                            "x1": sl_plot_max,
                            "y0": service_level,
                            "y1": service_level,
                            "line": {"color": COLOR_BAD, "dash": "dot"},
                        },
                        {
                            "type": "line",
                            "x0": optimal_stock,
                            "x1": optimal_stock,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": COLOR_GOOD, "dash": "dash"},
                        },
                    ],
                    "annotations": [
                        {
                            "x": optimal_stock,
                            "y": service_level,
                            "text": f"S*={optimal_stock}",
                            "showarrow": True,
                            "arrowhead": 2,
                            "font": {"color": COLOR_GOOD},
                        }
                    ],
                },
            }
        )

        result["plots"].append(
            {
                "title": "Total Expected Cost vs Stock Level",
                "data": [
                    {
                        "type": "scatter",
                        "x": stock_eval.tolist(),
                        "y": costs,
                        "line": {"color": SVEND_COLORS[2], "width": 2},
                        "name": "Total Cost",
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Stock Level"},
                    "yaxis": {"title": "Expected Cost"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": min_cost_stock,
                            "x1": min_cost_stock,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": COLOR_GOOD, "dash": "dash"},
                        }
                    ],
                    "annotations": [
                        {
                            "x": min_cost_stock,
                            "y": min_cost,
                            "text": f"Optimal={min_cost_stock}",
                            "showarrow": True,
                            "arrowhead": 2,
                            "font": {"color": COLOR_GOOD},
                        }
                    ],
                },
            }
        )

        result["what_if_data"] = {
            "type": "bayes_spares",
            "rate_mean": rate_mean,
            "planning_horizon": planning_horizon,
            "service_level": service_level,
            "holding_cost": holding_cost,
            "stockout_cost": stockout_cost,
            "optimal_stock": optimal_stock,
            "min_cost": min_cost,
        }

        result["education"] = {
            "title": "Understanding Bayesian Spare Parts Planning",
            "content": (
                "<dl>"
                "<dt>What is Bayesian spare parts planning?</dt>"
                "<dd>It estimates optimal inventory levels using a <em>Gamma-Poisson</em> conjugate "
                "model. Historical demand data updates a prior on the demand rate, and the posterior "
                "predictive distribution tells you how many spares you need to meet a target "
                "service level over a planning horizon.</dd>"
                "<dt>What is the service level?</dt>"
                "<dd>The probability that you will not stock out during the planning period. "
                "A 95% service level means a 95% chance of having enough spares on hand. "
                "Higher targets require more safety stock.</dd>"
                "<dt>How does the cost optimisation work?</dt>"
                "<dd>The model balances holding costs (carrying extra inventory) against stockout "
                "costs (production downtime, expediting). The optimal stock level minimises "
                "total expected cost across the posterior predictive distribution.</dd>"
                "<dt>Why Bayesian over simple averages?</dt>"
                "<dd>Spare parts demand is often low-volume, making point estimates unreliable. "
                "The Bayesian approach captures demand rate uncertainty — especially critical for "
                "expensive, slow-moving parts where overstocking and understocking are both costly.</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_system":
        # Bayesian System Reliability (MC propagation through topology)
        import json as _json

        components_raw = config.get("components", "[]")
        if isinstance(components_raw, str):
            try:
                components = _json.loads(components_raw)
            except Exception:
                result["summary"] = (
                    'Error: Invalid components JSON. Use format: [{"name":"Motor","n":100,"failures":2}, ...]'
                )
                return result
        else:
            components = components_raw

        topology = config.get("topology", "series")
        k_val = int(config.get("k", 2))
        if not components or len(components) < 2:
            result["summary"] = "Error: Provide at least 2 components."
            return result

        n_mc = 10000
        rng = np.random.default_rng(42)
        comp_names, comp_means, comp_cis, comp_draws, comp_post_params = (
            [],
            [],
            [],
            [],
            [],
        )

        for c in components:
            name = c.get("name", f"C{len(comp_names) + 1}")
            n_i = int(c.get("n", 100))
            k_i = int(c.get("failures", 0))
            pa = float(c.get("prior_a", 1))
            pb = float(c.get("prior_b", 1))
            a_post = pa + (n_i - k_i)
            b_post = pb + k_i
            draws = rng.beta(a_post, b_post, size=n_mc)
            comp_names.append(name)
            comp_means.append(float(np.mean(draws)))
            comp_cis.append(
                (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
            )
            comp_draws.append(draws)
            comp_post_params.append((float(a_post), float(b_post)))

        draws_matrix = np.column_stack(comp_draws)
        n_comp = len(components)

        if topology == "series":
            sys_draws = np.prod(draws_matrix, axis=1)
        elif topology == "parallel":
            sys_draws = 1 - np.prod(1 - draws_matrix, axis=1)
        elif topology == "k_of_n":
            sys_draws = np.zeros(n_mc)
            for i in range(n_mc):
                dp = np.zeros(n_comp + 1)
                dp[0] = 1.0
                for j_c in range(n_comp):
                    r_j = draws_matrix[i, j_c]
                    for s in range(j_c + 1, 0, -1):
                        dp[s] = dp[s] * (1 - r_j) + dp[s - 1] * r_j
                    dp[0] *= 1 - r_j
                sys_draws[i] = float(np.sum(dp[k_val:]))
        else:
            sys_draws = np.prod(draws_matrix, axis=1)

        sys_mean = float(np.mean(sys_draws))
        sys_ci = (
            float(np.percentile(sys_draws, (1 - ci_level) / 2 * 100)),
            float(np.percentile(sys_draws, (1 + ci_level) / 2 * 100)),
        )

        importance = [
            float(np.corrcoef(draws_matrix[:, j], sys_draws)[0, 1])
            for j in range(n_comp)
        ]
        weakest = comp_names[int(np.argmin(comp_means))]

        topology_label = topology.replace("_", "-")
        if topology == "k_of_n":
            topology_label = f"{k_val}-of-{n_comp}"

        verdict = (
            "PASS" if sys_mean > 0.9 else ("WARNING" if sys_mean > 0.8 else "FAIL")
        )

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN SYSTEM RELIABILITY ({topology_label.upper()})<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Topology:<</COLOR>> {topology_label}\n"
        summary += f"<<COLOR:highlight>>Components:<</COLOR>> {n_comp}\n\n"
        for j, name in enumerate(comp_names):
            summary += f"  {name}: R = {comp_means[j]:.4f} (CI: {comp_cis[j][0]:.4f}\u2013{comp_cis[j][1]:.4f})\n"
        summary += f"\n<<COLOR:highlight>>System Reliability:<</COLOR>> <<COLOR:good>>{sys_mean:.4f}<</COLOR>> "
        summary += f"(CI: {sys_ci[0]:.4f}\u2013{sys_ci[1]:.4f})\n"
        summary += f"<<COLOR:highlight>>Weakest component:<</COLOR>> {weakest}\n"

        result["summary"] = summary
        result["statistics"] = {
            "system_reliability": sys_mean,
            "system_ci": list(sys_ci),
            "topology": topology_label,
            "weakest": weakest,
            "components": {
                n: {"mean": m, "ci": list(c)}
                for n, m, c in zip(comp_names, comp_means, comp_cis)
            },
        }
        result["guide_observation"] = (
            f"Bayes system ({topology_label}): R_sys={sys_mean:.4f}, weakest={weakest}"
        )
        result["narrative"] = _narrative(
            verdict,
            f"{topology_label} system with {n_comp} components: R_sys = {sys_mean:.4f} "
            f"({int(ci_level * 100)}% CI: {sys_ci[0]:.4f}\u2013{sys_ci[1]:.4f}). "
            f"Weakest link: {weakest} (R = {min(comp_means):.4f}).",
            [
                "Focus improvement on highest-importance / lowest-reliability component",
                "Consider redundancy (parallel) for critical components",
                "Update component priors with field data",
            ],
        )

        # Plot 1: Component posterior overlay
        comp_traces = []
        for j, name in enumerate(comp_names):
            hist_j, edges_j = np.histogram(comp_draws[j], bins=80, density=True)
            centers_j = (edges_j[:-1] + edges_j[1:]) / 2
            comp_traces.append(
                {
                    "type": "scatter",
                    "x": centers_j.tolist(),
                    "y": hist_j.tolist(),
                    "line": {"color": SVEND_COLORS[j % len(SVEND_COLORS)], "width": 2},
                    "name": name,
                }
            )
        result["plots"].append(
            {
                "title": "Component Reliability Posteriors",
                "data": comp_traces,
                "layout": {
                    "height": 350,
                    "xaxis": {"title": "Reliability"},
                    "yaxis": {"title": "Density"},
                },
            }
        )

        # Plot 2: System reliability posterior
        hist_sys, edges_sys = np.histogram(sys_draws, bins=60, density=True)
        centers_sys = (edges_sys[:-1] + edges_sys[1:]) / 2
        result["plots"].append(
            {
                "title": f"System Reliability Posterior ({topology_label})",
                "data": [
                    {
                        "type": "bar",
                        "x": centers_sys.tolist(),
                        "y": hist_sys.tolist(),
                        "marker": {"color": _rgba(SVEND_COLORS[0], 0.7)},
                        "name": "System R",
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "System Reliability"},
                    "yaxis": {"title": "Density"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": sys_mean,
                            "x1": sys_mean,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": COLOR_GOOD, "dash": "dash", "width": 2},
                        }
                    ],
                },
            }
        )

        # Plot 3: Component importance bar
        result["plots"].append(
            {
                "title": "Component Importance (Correlation with System R)",
                "data": [
                    {
                        "type": "bar",
                        "x": comp_names,
                        "y": importance,
                        "marker": {
                            "color": [
                                SVEND_COLORS[j % len(SVEND_COLORS)]
                                for j in range(n_comp)
                            ]
                        },
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Component"},
                    "yaxis": {"title": "Birnbaum Importance"},
                },
            }
        )

        result["what_if_data"] = {
            "type": "bayes_system",
            "topology": topology,
            "k": k_val if topology == "k_of_n" else None,
            "components": [
                {
                    "name": comp_names[j],
                    "mean": comp_means[j],
                    "post_a": comp_post_params[j][0],
                    "post_b": comp_post_params[j][1],
                }
                for j in range(n_comp)
            ],
            "system_mean": sys_mean,
            "system_ci": list(sys_ci),
        }

        result["education"] = {
            "title": "Understanding Bayesian System Reliability",
            "content": (
                "<dl>"
                "<dt>What is Bayesian system reliability?</dt>"
                "<dd>It estimates the reliability of a multi-component system by propagating "
                "component-level posterior distributions through a <em>system topology</em> "
                "(series, parallel, or mixed) via Monte Carlo simulation.</dd>"
                "<dt>How does topology affect reliability?</dt>"
                "<dd><strong>Series</strong>: all components must work — system reliability is the "
                "product of component reliabilities (weakest link). <strong>Parallel</strong>: "
                "any one working suffices — redundancy increases system reliability above "
                "any single component.</dd>"
                "<dt>What do the credible intervals show?</dt>"
                "<dd>The system-level credible interval propagates uncertainty from all components "
                "simultaneously. Even if each component is well-characterised, the system interval "
                "may be wide if there are many components in series.</dd>"
                "<dt>When to use this?</dt>"
                "<dd>Evaluating system design choices (add redundancy?), predicting fleet reliability, "
                "identifying the weakest components, or demonstrating system-level reliability to "
                "customers with honest uncertainty bounds.</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_warranty":
        # Bayesian Warranty Forecast (Gamma-Poisson conjugate)
        time_col = config.get("var1")
        warranty_period = float(config.get("warranty_period", 12))
        fleet_size = int(config.get("fleet_size", 1000))
        forecast_periods = int(config.get("forecast_period", 12))
        prior_a = float(config.get("prior_a", 1.0))
        prior_b = float(config.get("prior_b", 1.0))

        if not time_col or time_col not in df.columns:
            result["summary"] = "Error: Select a time-to-failure column."
            return result

        failure_times = df[time_col].dropna().values.astype(float)
        n_warranty = int(np.sum(failure_times <= warranty_period))
        total_time = float(np.sum(np.minimum(failure_times, warranty_period)))
        n_total = len(failure_times)

        post_a = prior_a + n_warranty
        post_b = prior_b + total_time
        rate_mean = float(post_a / post_b)
        rate_ci = (
            float(stats.gamma.ppf((1 - ci_level) / 2, post_a, scale=1 / post_b)),
            float(stats.gamma.ppf((1 + ci_level) / 2, post_a, scale=1 / post_b)),
        )
        claims_per_period = rate_mean * fleet_size

        n_mc = 5000
        rng = np.random.default_rng(42)
        rate_samples = rng.gamma(post_a, 1 / post_b, size=n_mc)
        monthly_claims = np.zeros((n_mc, forecast_periods))
        for m in range(forecast_periods):
            monthly_claims[:, m] = rng.poisson(rate_samples * fleet_size)
        cum_claims = np.cumsum(monthly_claims, axis=1)

        cum_mean = np.mean(cum_claims, axis=0)
        cum_lo = np.percentile(cum_claims, (1 - ci_level) / 2 * 100, axis=0)
        cum_hi = np.percentile(cum_claims, (1 + ci_level) / 2 * 100, axis=0)
        monthly_mean = np.mean(monthly_claims, axis=0)
        monthly_lo = np.percentile(monthly_claims, (1 - ci_level) / 2 * 100, axis=0)
        monthly_hi = np.percentile(monthly_claims, (1 + ci_level) / 2 * 100, axis=0)

        total_forecast = float(cum_mean[-1]) if len(cum_mean) > 0 else 0
        total_ci = (float(cum_lo[-1]), float(cum_hi[-1]))
        verdict = "WARNING" if rate_mean > 0.01 else "PASS"

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN WARRANTY FORECAST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Warranty period:<</COLOR>> {warranty_period}\n"
        summary += f"<<COLOR:highlight>>Fleet size:<</COLOR>> {fleet_size:,}\n"
        summary += f"<<COLOR:highlight>>Observed failures (in warranty):<</COLOR>> {n_warranty} / {n_total}\n\n"
        summary += f"<<COLOR:highlight>>Posterior failure rate:<</COLOR>> {rate_mean:.4f}/unit-time (CI: {rate_ci[0]:.4f}\u2013{rate_ci[1]:.4f})\n"
        summary += f"<<COLOR:highlight>>Expected claims/period:<</COLOR>> {claims_per_period:.1f}\n"
        summary += f"<<COLOR:highlight>>Forecast ({forecast_periods} periods):<</COLOR>> <<COLOR:warning>>{total_forecast:.0f}<</COLOR>> (CI: {total_ci[0]:.0f}\u2013{total_ci[1]:.0f})\n"

        result["summary"] = summary
        result["statistics"] = {
            "rate_mean": rate_mean,
            "rate_ci": list(rate_ci),
            "claims_per_period": claims_per_period,
            "total_forecast": total_forecast,
            "total_ci": list(total_ci),
        }
        result["guide_observation"] = (
            f"Bayes warranty: rate={rate_mean:.4f}, {forecast_periods}-period forecast={total_forecast:.0f}"
        )
        result["narrative"] = _narrative(
            verdict,
            f"Failure rate posterior: {rate_mean:.4f}/unit-time (CI: {rate_ci[0]:.4f}\u2013{rate_ci[1]:.4f}). "
            f"For fleet of {fleet_size:,}, expect ~{claims_per_period:.0f} claims/period. "
            f"{forecast_periods}-period total: {total_forecast:.0f} (CI: {total_ci[0]:.0f}\u2013{total_ci[1]:.0f}).",
            [
                "Set warranty reserves based on upper credible bound",
                "Update forecast monthly as new claims data arrives",
                "Investigate root causes if rate exceeds prior expectations",
            ],
        )

        # Plot 1: Prior vs Posterior on failure rate
        rv = np.linspace(0, rate_ci[1] * 2.5, 300)
        result["plots"].append(
            {
                "title": "Prior vs Posterior on Failure Rate",
                "data": [
                    {
                        "type": "scatter",
                        "x": rv.tolist(),
                        "y": stats.gamma.pdf(rv, prior_a, scale=1 / prior_b).tolist(),
                        "line": {"color": COLOR_REFERENCE, "dash": "dash", "width": 2},
                        "name": "Prior",
                    },
                    {
                        "type": "scatter",
                        "x": rv.tolist(),
                        "y": stats.gamma.pdf(rv, post_a, scale=1 / post_b).tolist(),
                        "fill": "tozeroy",
                        "fillcolor": _rgba(SVEND_COLORS[0], 0.3),
                        "line": {"color": SVEND_COLORS[0], "width": 2},
                        "name": "Posterior",
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Failure Rate (\u03bb)"},
                    "yaxis": {"title": "Density"},
                },
            }
        )

        # Plot 2: Cumulative returns forecast
        periods = list(range(1, forecast_periods + 1))
        result["plots"].append(
            {
                "title": "Cumulative Warranty Claims Forecast",
                "data": [
                    {
                        "type": "scatter",
                        "x": periods,
                        "y": cum_hi.tolist(),
                        "line": {"width": 0},
                        "showlegend": False,
                        "name": "Upper",
                    },
                    {
                        "type": "scatter",
                        "x": periods,
                        "y": cum_lo.tolist(),
                        "fill": "tonexty",
                        "fillcolor": _rgba(SVEND_COLORS[1], 0.2),
                        "line": {"width": 0},
                        "name": f"{int(ci_level * 100)}% CI",
                    },
                    {
                        "type": "scatter",
                        "x": periods,
                        "y": cum_mean.tolist(),
                        "line": {"color": SVEND_COLORS[1], "width": 2},
                        "name": "Expected",
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Period"},
                    "yaxis": {"title": "Cumulative Claims"},
                },
            }
        )

        # Plot 3: Monthly forecast bars
        result["plots"].append(
            {
                "title": "Monthly Claims Forecast",
                "data": [
                    {
                        "type": "bar",
                        "x": periods,
                        "y": monthly_mean.tolist(),
                        "error_y": {
                            "type": "data",
                            "symmetric": False,
                            "array": (monthly_hi - monthly_mean).tolist(),
                            "arrayminus": (monthly_mean - monthly_lo).tolist(),
                        },
                        "marker": {"color": _rgba(SVEND_COLORS[2], 0.7)},
                        "name": "Claims",
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Period"},
                    "yaxis": {"title": "Claims"},
                },
            }
        )

        result["what_if_data"] = {
            "type": "bayes_warranty",
            "rate_mean": rate_mean,
            "fleet_size": fleet_size,
            "forecast_periods": forecast_periods,
            "claims_per_period": claims_per_period,
            "total_forecast": total_forecast,
            "total_ci": list(total_ci),
        }

        result["education"] = {
            "title": "Understanding Bayesian Warranty Forecasting",
            "content": (
                "<dl>"
                "<dt>What is Bayesian warranty forecasting?</dt>"
                "<dd>It predicts future warranty claims using a <em>Gamma-Poisson</em> conjugate "
                "model. Historical claim data updates a prior on the claim rate, and the posterior "
                "predictive distribution forecasts cumulative claims over a planning horizon.</dd>"
                "<dt>What does the forecast show?</dt>"
                "<dd>Expected claims per period with credible intervals. The width of the band "
                "reflects both the inherent randomness of claims and the uncertainty in the "
                "estimated claim rate — wider bands mean less historical data to learn from.</dd>"
                "<dt>How is fleet size used?</dt>"
                "<dd>The claim rate is per-unit. Multiplying by fleet size gives the total expected "
                "claims. Larger fleets produce narrower <em>relative</em> uncertainty (law of "
                "large numbers) but higher absolute claim volumes.</dd>"
                "<dt>When to use this?</dt>"
                "<dd>Financial provisioning for warranty reserves, planning field service capacity, "
                "evaluating warranty extension pricing, or detecting whether a design change has "
                "altered the claim rate.</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_repairable":
        # Bayesian NHPP Repairable Systems (Power Law Process / Crow-AMSAA)
        time_col = config.get("var1")
        if not time_col or time_col not in df.columns:
            result["summary"] = "Error: Select a failure time column."
            return result

        times_raw = df[time_col].dropna().values.astype(float)
        times_raw = np.sort(times_raw[times_raw > 0])
        n_events = len(times_raw)
        if n_events < 3:
            result["summary"] = "Error: Need at least 3 failure events."
            return result

        T = float(times_raw[-1])
        log_ratios = np.log(T / times_raw)
        beta_mle = float(n_events / np.sum(log_ratios))
        theta_mle = float(T / (n_events ** (1 / beta_mle)))

        n_grid = 100
        beta_range = np.linspace(max(0.1, beta_mle * 0.3), beta_mle * 3, n_grid)
        theta_range = np.linspace(max(0.1, theta_mle * 0.3), theta_mle * 3, n_grid)
        beta_g, theta_g = np.meshgrid(beta_range, theta_range)

        log_lik = np.zeros_like(beta_g)
        log_lik += n_events * np.log(beta_g + 1e-15)
        log_lik -= n_events * beta_g * np.log(theta_g + 1e-15)
        for t_i in times_raw:
            log_lik += (beta_g - 1) * np.log(t_i + 1e-15)
        log_lik -= (T / (theta_g + 1e-15)) ** beta_g

        log_post = log_lik - log_lik.max()
        posterior = np.exp(log_post)
        posterior /= posterior.sum()

        beta_marg = posterior.sum(axis=0)
        beta_marg /= beta_marg.sum()
        theta_marg = posterior.sum(axis=1)
        theta_marg /= theta_marg.sum()

        beta_post_mean = float(np.sum(beta_range * beta_marg))
        theta_post_mean = float(np.sum(theta_range * theta_marg))
        beta_post_ci = (
            float(
                beta_range[np.searchsorted(np.cumsum(beta_marg), (1 - ci_level) / 2)]
            ),
            float(
                beta_range[
                    min(
                        n_grid - 1,
                        np.searchsorted(np.cumsum(beta_marg), (1 + ci_level) / 2),
                    )
                ]
            ),
        )
        p_deteriorating = float(np.sum(beta_marg[beta_range > 1]))

        n_mc = 5000
        rng = np.random.default_rng(42)
        beta_idx = rng.choice(n_grid, size=n_mc, p=beta_marg)
        theta_idx = rng.choice(n_grid, size=n_mc, p=theta_marg)
        beta_s = beta_range[beta_idx]
        theta_s = theta_range[theta_idx]

        t_eval = np.linspace(0, T * 1.3, 200)
        mcf_samples = np.zeros((n_mc, len(t_eval)))
        for i in range(n_mc):
            mcf_samples[i] = (t_eval / theta_s[i]) ** beta_s[i]
        mcf_mean = np.mean(mcf_samples, axis=0)
        mcf_lo = np.percentile(mcf_samples, (1 - ci_level) / 2 * 100, axis=0)
        mcf_hi = np.percentile(mcf_samples, (1 + ci_level) / 2 * 100, axis=0)

        dt = T * 0.1
        next_samples = ((T + dt) / theta_s) ** beta_s - (T / theta_s) ** beta_s
        next_mean = float(np.mean(next_samples))
        next_ci = (
            float(np.percentile(next_samples, 2.5)),
            float(np.percentile(next_samples, 97.5)),
        )

        obs_mcf = np.arange(1, n_events + 1)
        trend_label = (
            "deteriorating"
            if p_deteriorating > 0.8
            else ("improving" if p_deteriorating < 0.2 else "stable")
        )
        verdict = (
            "FAIL"
            if p_deteriorating > 0.8
            else ("PASS" if p_deteriorating < 0.2 else "WARNING")
        )

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN REPAIRABLE SYSTEM (NHPP)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Events:<</COLOR>> {n_events}\n"
        summary += f"<<COLOR:highlight>>Observation window:<</COLOR>> [0, {T:.1f}]\n\n"
        summary += f"<<COLOR:highlight>>\u03b2 posterior mean:<</COLOR>> {beta_post_mean:.3f} (CI: {beta_post_ci[0]:.3f}\u2013{beta_post_ci[1]:.3f})\n"
        summary += f"<<COLOR:highlight>>\u03b8 posterior mean:<</COLOR>> {theta_post_mean:.1f}\n"
        color_tag = "bad" if p_deteriorating > 0.8 else "good"
        summary += f"<<COLOR:highlight>>P(deteriorating, \u03b2>1):<</COLOR>> <<COLOR:{color_tag}>>{p_deteriorating:.3f}<</COLOR>>\n"
        summary += f"<<COLOR:highlight>>Trend:<</COLOR>> {trend_label}\n"
        summary += f"<<COLOR:highlight>>Expected failures next {dt:.1f} time units:<</COLOR>> {next_mean:.1f} (CI: {next_ci[0]:.1f}\u2013{next_ci[1]:.1f})\n"

        result["summary"] = summary
        result["statistics"] = {
            "beta_mean": beta_post_mean,
            "beta_ci": list(beta_post_ci),
            "theta_mean": theta_post_mean,
            "p_deteriorating": p_deteriorating,
            "trend": trend_label,
            "next_period_mean": next_mean,
            "next_period_ci": list(next_ci),
        }
        result["guide_observation"] = (
            f"Bayes repairable: \u03b2={beta_post_mean:.3f}, P(deteriorating)={p_deteriorating:.3f}, trend={trend_label}"
        )
        result["narrative"] = _narrative(
            verdict,
            f"Power Law Process: \u03b2 = {beta_post_mean:.3f} (CI: {beta_post_ci[0]:.3f}\u2013{beta_post_ci[1]:.3f}). "
            f"P(\u03b2 > 1) = {p_deteriorating:.3f} \u2192 system is {trend_label}. "
            f"Expected {next_mean:.1f} failures in next {dt:.1f} time units.",
            [
                "If deteriorating, schedule preventive overhaul",
                "If improving, maintenance actions may be working \u2014 continue monitoring",
                "Segment by system if data spans multiple units",
            ],
        )

        # Plot 1: MCF with credible band
        result["plots"].append(
            {
                "title": "Mean Cumulative Function (MCF) with Credible Band",
                "data": [
                    {
                        "type": "scatter",
                        "x": t_eval.tolist(),
                        "y": mcf_hi.tolist(),
                        "line": {"width": 0},
                        "showlegend": False,
                        "name": "Upper",
                    },
                    {
                        "type": "scatter",
                        "x": t_eval.tolist(),
                        "y": mcf_lo.tolist(),
                        "fill": "tonexty",
                        "fillcolor": _rgba(SVEND_COLORS[0], 0.2),
                        "line": {"width": 0},
                        "name": f"{int(ci_level * 100)}% CI",
                    },
                    {
                        "type": "scatter",
                        "x": t_eval.tolist(),
                        "y": mcf_mean.tolist(),
                        "line": {"color": SVEND_COLORS[0], "width": 2},
                        "name": "Posterior Mean",
                    },
                    {
                        "type": "scatter",
                        "x": times_raw.tolist(),
                        "y": obs_mcf.tolist(),
                        "mode": "markers",
                        "marker": {"color": COLOR_BAD, "size": 5},
                        "name": "Observed",
                    },
                ],
                "layout": {
                    "height": 350,
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Cumulative Failures"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": T,
                            "x1": T,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": "#888", "dash": "dot"},
                        }
                    ],
                    "annotations": [
                        {
                            "x": T,
                            "y": 1,
                            "yref": "paper",
                            "text": "End of observation",
                            "showarrow": False,
                            "yanchor": "bottom",
                        }
                    ],
                },
            }
        )

        # Plot 2: beta posterior with reference at 1
        result["plots"].append(
            {
                "title": "Posterior on Shape Parameter \u03b2",
                "data": [
                    {
                        "type": "scatter",
                        "x": beta_range.tolist(),
                        "y": beta_marg.tolist(),
                        "fill": "tozeroy",
                        "fillcolor": _rgba(SVEND_COLORS[1], 0.3),
                        "line": {"color": SVEND_COLORS[1], "width": 2},
                        "name": "\u03b2 Posterior",
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "\u03b2 (Shape)"},
                    "yaxis": {"title": "Density"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 1,
                            "x1": 1,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": COLOR_BAD, "dash": "dot", "width": 2},
                        }
                    ],
                    "annotations": [
                        {
                            "x": 1,
                            "y": 1,
                            "yref": "paper",
                            "text": "\u03b2=1 (HPP)",
                            "showarrow": False,
                            "yanchor": "bottom",
                            "font": {"color": COLOR_BAD},
                        }
                    ],
                },
            }
        )

        # Plot 3: Expected failures forecast
        n_fp = 5
        forecast_bars, forecast_lo_bars, forecast_hi_bars, period_labels = (
            [],
            [],
            [],
            [],
        )
        for p in range(1, n_fp + 1):
            t_start = T + (p - 1) * dt
            t_end = T + p * dt
            f_samp = ((t_end / theta_s) ** beta_s) - ((t_start / theta_s) ** beta_s)
            forecast_bars.append(float(np.mean(f_samp)))
            forecast_lo_bars.append(float(np.percentile(f_samp, 2.5)))
            forecast_hi_bars.append(float(np.percentile(f_samp, 97.5)))
            period_labels.append(f"P{p}")
        result["plots"].append(
            {
                "title": "Expected Failures in Future Periods",
                "data": [
                    {
                        "type": "bar",
                        "x": period_labels,
                        "y": forecast_bars,
                        "error_y": {
                            "type": "data",
                            "symmetric": False,
                            "array": [
                                h - m for h, m in zip(forecast_hi_bars, forecast_bars)
                            ],
                            "arrayminus": [
                                m - lo for m, lo in zip(forecast_bars, forecast_lo_bars)
                            ],
                        },
                        "marker": {"color": _rgba(SVEND_COLORS[2], 0.7)},
                        "name": "Expected Failures",
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Forecast Period"},
                    "yaxis": {"title": "Expected Failures"},
                },
            }
        )

        result["education"] = {
            "title": "Understanding Bayesian Repairable Systems (NHPP)",
            "content": (
                "<dl>"
                "<dt>What is a repairable systems model?</dt>"
                "<dd>Unlike one-shot reliability (unit fails once), repairable systems are repaired "
                "and returned to service repeatedly. The <em>Non-Homogeneous Poisson Process</em> "
                "(NHPP) with Power Law (Crow-AMSAA) models how the failure intensity changes over time.</dd>"
                "<dt>What does the β parameter mean?</dt>"
                "<dd><strong>β &lt; 1</strong>: reliability growth — failures are becoming less frequent "
                "(improvement). <strong>β = 1</strong>: constant failure rate (no trend). "
                "<strong>β &gt; 1</strong>: reliability deterioration — failures are accelerating (wear-out).</dd>"
                "<dt>What is the failure intensity function?</dt>"
                "<dd>The instantaneous rate of failure at any given time. The NHPP allows this rate "
                "to change — unlike a homogeneous Poisson process which assumes constant rate. "
                "The Bayesian posterior gives a credible band on this intensity.</dd>"
                "<dt>When to use this?</dt>"
                "<dd>Fleet vehicles, industrial equipment, IT systems — any asset that is repeatedly "
                "repaired. Track whether reliability is improving (after maintenance programme changes) "
                "or degrading (approaching end of life), and forecast future failure counts.</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_rul":
        # Bayesian Remaining Useful Life (Degradation modeling)
        time_col = config.get("var1")
        meas_col = config.get("var2")
        unit_col = config.get("var3")
        threshold = float(config.get("threshold", 100))
        direction = config.get("direction", "increasing")

        if not time_col or not meas_col:
            result["summary"] = "Error: Select time and measurement columns."
            return result

        cols = [c for c in [time_col, meas_col, unit_col] if c and c in df.columns]
        df_clean = df[cols].dropna()
        if len(df_clean) < 5:
            result["summary"] = "Error: Need at least 5 observations."
            return result

        if unit_col and unit_col in df_clean.columns:
            units = df_clean[unit_col].unique()
        else:
            units = ["all"]
            df_clean = df_clean.copy()
            df_clean["_unit"] = "all"
            unit_col = "_unit"

        slopes, intercepts = [], []
        for u in units:
            mask = df_clean[unit_col] == u
            t_u = df_clean.loc[mask, time_col].values.astype(float)
            y_u = df_clean.loc[mask, meas_col].values.astype(float)
            if len(t_u) < 2:
                continue
            try:
                b, a, _, _, _ = stats.linregress(t_u, y_u)
                if np.isfinite(b):
                    slopes.append(b)
                    intercepts.append(a)
            except Exception:
                continue

        # Fallback: if per-unit regression failed (e.g. 1 obs per unit), do global regression
        if len(slopes) == 0 and len(df_clean) >= 2:
            t_all = df_clean[time_col].values.astype(float)
            y_all = df_clean[meas_col].values.astype(float)
            try:
                b, a, _, _, _ = stats.linregress(t_all, y_all)
                if np.isfinite(b):
                    slopes.append(b)
                    intercepts.append(a)
                    units = ["all"]
                    unit_col = "_unit"
                    df_clean = df_clean.copy()
                    df_clean["_unit"] = "all"
            except Exception:
                pass

        slopes = np.array(slopes)
        intercepts = np.array(intercepts)
        n_units = len(slopes)
        if n_units < 1:
            result["summary"] = (
                "Error: Could not estimate degradation rates. Ensure each unit has at least 2 time-measurement pairs."
            )
            return result

        slope_mean = float(np.mean(slopes))
        slope_var = (
            float(np.var(slopes, ddof=1)) if n_units > 1 else float(slope_mean**2 * 0.1)
        )
        slope_std = float(np.sqrt(slope_var))

        all_times = df_clean[time_col].values.astype(float)
        all_meas = df_clean[meas_col].values.astype(float)
        t_current = float(np.max(all_times))
        last_mask = all_times == all_times.max()
        y_current = (
            float(np.mean(all_meas[last_mask]))
            if np.sum(last_mask) > 0
            else float(all_meas[-1])
        )

        n_mc = 10000
        rng = np.random.default_rng(42)
        if n_units > 2:
            df_t = n_units - 1
            slope_samples = stats.t.rvs(
                df_t,
                loc=slope_mean,
                scale=slope_std / np.sqrt(n_units),
                size=n_mc,
                random_state=42,
            )
        else:
            slope_samples = rng.normal(
                slope_mean,
                slope_std if slope_std > 0 else abs(slope_mean) * 0.1,
                size=n_mc,
            )

        if direction == "decreasing":
            rul_samples = np.where(
                slope_samples < 0, (threshold - y_current) / slope_samples, np.inf
            )
        else:
            rul_samples = np.where(
                slope_samples > 0, (threshold - y_current) / slope_samples, np.inf
            )

        valid_rul = rul_samples[(rul_samples > 0) & np.isfinite(rul_samples)]
        if len(valid_rul) < 100:
            valid_rul = np.abs(rul_samples[np.isfinite(rul_samples)])
            valid_rul = valid_rul[valid_rul > 0]

        if len(valid_rul) > 0:
            rul_mean = float(np.mean(valid_rul))
            rul_ci = (
                float(np.percentile(valid_rul, (1 - ci_level) / 2 * 100)),
                float(np.percentile(valid_rul, (1 + ci_level) / 2 * 100)),
            )
        else:
            rul_mean, rul_ci = float("inf"), (0.0, float("inf"))

        horizon = float(
            config.get(
                "horizon", rul_mean * 1.5 if np.isfinite(rul_mean) else t_current
            )
        )
        p_fail_horizon = (
            float(np.mean(valid_rul <= horizon)) if len(valid_rul) > 0 else 0.0
        )
        verdict = "WARNING" if p_fail_horizon > 0.5 else "PASS"

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN REMAINING USEFUL LIFE<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Units analyzed:<</COLOR>> {n_units}\n"
        summary += f"<<COLOR:highlight>>Threshold:<</COLOR>> {threshold}\n"
        summary += f"<<COLOR:highlight>>Direction:<</COLOR>> {direction}\n\n"
        summary += f"<<COLOR:highlight>>Degradation rate:<</COLOR>> {slope_mean:.4f} \u00b1 {slope_std:.4f}/time\n"
        summary += f"<<COLOR:highlight>>Current measurement:<</COLOR>> {y_current:.2f} at t={t_current:.1f}\n"
        summary += f"<<COLOR:highlight>>RUL estimate:<</COLOR>> <<COLOR:warning>>{rul_mean:.1f}<</COLOR>> (CI: {rul_ci[0]:.1f}\u2013{rul_ci[1]:.1f})\n"
        summary += f"<<COLOR:highlight>>P(fail before {horizon:.0f}):<</COLOR>> {p_fail_horizon:.3f}\n"

        result["summary"] = summary
        result["statistics"] = {
            "slope_mean": slope_mean,
            "slope_std": slope_std,
            "rul_mean": rul_mean,
            "rul_ci": list(rul_ci),
            "p_fail_horizon": p_fail_horizon,
            "n_units": n_units,
        }
        result["guide_observation"] = (
            f"Bayes RUL: mean={rul_mean:.1f}, P(fail before {horizon:.0f})={p_fail_horizon:.3f}"
        )
        result["narrative"] = _narrative(
            verdict,
            f"Degradation rate: {slope_mean:.4f} \u00b1 {slope_std:.4f}/time ({n_units} units). "
            f"RUL estimate: {rul_mean:.1f} ({int(ci_level * 100)}% CI: {rul_ci[0]:.1f}\u2013{rul_ci[1]:.1f}). "
            f"P(fail before {horizon:.0f}) = {p_fail_horizon:.1%}.",
            [
                "Schedule maintenance before lower credible bound",
                "Increase monitoring frequency as degradation approaches threshold",
                "Use unit-specific tracking for fleet heterogeneity",
            ],
        )

        # Plot 1: Degradation paths with threshold
        deg_traces = []
        for u_idx, u in enumerate(units[:20]):
            mask = df_clean[unit_col] == u
            t_u = df_clean.loc[mask, time_col].values.astype(float)
            y_u = df_clean.loc[mask, meas_col].values.astype(float)
            deg_traces.append(
                {
                    "type": "scatter",
                    "x": t_u.tolist(),
                    "y": y_u.tolist(),
                    "mode": "lines+markers",
                    "line": {
                        "color": SVEND_COLORS[u_idx % len(SVEND_COLORS)],
                        "width": 1.5,
                    },
                    "marker": {"size": 3},
                    "name": str(u),
                    "showlegend": n_units <= 10,
                }
            )
        result["plots"].append(
            {
                "title": "Degradation Paths",
                "data": deg_traces,
                "layout": {
                    "height": 350,
                    "xaxis": {"title": time_col},
                    "yaxis": {"title": meas_col},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 0,
                            "x1": t_current * 1.5,
                            "y0": threshold,
                            "y1": threshold,
                            "line": {"color": COLOR_BAD, "dash": "dot", "width": 2},
                        }
                    ],
                    "annotations": [
                        {
                            "x": t_current * 1.5,
                            "y": threshold,
                            "text": f"Threshold={threshold}",
                            "showarrow": False,
                            "xanchor": "right",
                            "font": {"color": COLOR_BAD},
                        }
                    ],
                },
            }
        )

        # Plot 2: Posterior on degradation rate
        rate_x = (
            np.linspace(slope_mean - 4 * slope_std, slope_mean + 4 * slope_std, 200)
            if slope_std > 0
            else np.linspace(slope_mean * 0.5, slope_mean * 1.5, 200)
        )
        if n_units > 2:
            rate_pdf = stats.t.pdf(
                rate_x, n_units - 1, loc=slope_mean, scale=slope_std / np.sqrt(n_units)
            )
        else:
            rate_pdf = stats.norm.pdf(
                rate_x,
                slope_mean,
                slope_std if slope_std > 0 else abs(slope_mean) * 0.1,
            )
        result["plots"].append(
            {
                "title": "Posterior on Degradation Rate",
                "data": [
                    {
                        "type": "scatter",
                        "x": rate_x.tolist(),
                        "y": rate_pdf.tolist(),
                        "fill": "tozeroy",
                        "fillcolor": _rgba(SVEND_COLORS[1], 0.3),
                        "line": {"color": SVEND_COLORS[1], "width": 2},
                        "name": "Rate Posterior",
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Degradation Rate"},
                    "yaxis": {"title": "Density"},
                },
            }
        )

        # Plot 3: RUL posterior predictive histogram
        if len(valid_rul) > 10:
            hist_rul, edges_rul = np.histogram(valid_rul, bins=60, density=True)
            centers_rul = (edges_rul[:-1] + edges_rul[1:]) / 2
            result["plots"].append(
                {
                    "title": "RUL Posterior Predictive",
                    "data": [
                        {
                            "type": "bar",
                            "x": centers_rul.tolist(),
                            "y": hist_rul.tolist(),
                            "marker": {"color": _rgba(SVEND_COLORS[2], 0.7)},
                            "name": "RUL",
                        }
                    ],
                    "layout": {
                        "height": 300,
                        "xaxis": {"title": "Remaining Useful Life"},
                        "yaxis": {"title": "Density"},
                        "shapes": [
                            {
                                "type": "line",
                                "x0": rul_mean,
                                "x1": rul_mean,
                                "y0": 0,
                                "y1": 1,
                                "yref": "paper",
                                "line": {
                                    "color": COLOR_WARNING,
                                    "dash": "dash",
                                    "width": 2,
                                },
                            }
                        ],
                    },
                }
            )

        result["education"] = {
            "title": "Understanding Bayesian Remaining Useful Life (RUL)",
            "content": (
                "<dl>"
                "<dt>What is RUL prediction?</dt>"
                "<dd>It estimates how much operational life remains before a component reaches "
                "a failure threshold. The Bayesian approach models degradation trajectories "
                "and produces a <em>posterior predictive distribution</em> on the time to failure.</dd>"
                "<dt>How does degradation modelling work?</dt>"
                "<dd>A linear or nonlinear model fits measurement data (vibration, wear, resistance) "
                "over time. The Bayesian posterior on degradation rate captures unit-to-unit "
                "variability, then extrapolates to predict when the threshold will be crossed.</dd>"
                "<dt>What does the RUL distribution show?</dt>"
                "<dd>The probability distribution over remaining life. A narrow distribution means "
                "the prediction is confident; a wide one means significant uncertainty remains — "
                "more monitoring data would help narrow it.</dd>"
                "<dt>When to use this?</dt>"
                "<dd>Condition-based maintenance scheduling, prognostics and health management (PHM), "
                "deciding when to replace bearings, batteries, filters, or any degrading component "
                "before it fails in service.</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_alt":
        # Bayesian Accelerated Life Testing
        time_col = config.get("var1")
        stress_col = config.get("var2")
        model_type = config.get("model", "arrhenius")
        use_stress = float(config.get("use_stress", 1.0))

        if not time_col or not stress_col:
            result["summary"] = "Error: Select time and stress columns."
            return result

        df_alt = df[[time_col, stress_col]].dropna()
        times_alt = df_alt[time_col].values.astype(float)
        stresses = df_alt[stress_col].values.astype(float)
        if len(times_alt) < 5:
            result["summary"] = "Error: Need at least 5 observations."
            return result

        unique_stresses = np.sort(np.unique(stresses))
        if len(unique_stresses) < 2:
            result["summary"] = "Error: Need at least 2 stress levels."
            return result

        if model_type == "arrhenius":
            1.0 / (stresses + 1e-15)
            x_use = 1.0 / use_stress if use_stress > 0 else 1.0
        else:
            np.log(stresses + 1e-15)
            x_use = np.log(use_stress) if use_stress > 0 else 0.0

        log_lives, stress_x, weights = [], [], []
        for s in unique_stresses:
            mask = stresses == s
            t_s = times_alt[mask]
            if len(t_s) < 2:
                continue
            try:
                _, _, scale = stats.weibull_min.fit(t_s, floc=0)
                log_lives.append(np.log(scale))
                stress_x.append(
                    1.0 / s
                    if (model_type == "arrhenius" and s > 0)
                    else np.log(s + 1e-15)
                )
                weights.append(len(t_s))
            except Exception:
                continue

        if len(log_lives) < 2:
            result["summary"] = "Error: Could not fit Weibull at 2+ stress levels."
            return result

        log_lives = np.array(log_lives)
        stress_x = np.array(stress_x)
        b, a, r_value, _, std_err = stats.linregress(stress_x, log_lives)
        r2 = r_value**2
        n_pts = len(log_lives)
        residuals = log_lives - (a + b * stress_x)
        s2 = float(np.sum(residuals**2) / max(1, n_pts - 2))

        log_life_use = a + b * x_use
        life_use = float(np.exp(log_life_use))

        n_mc = 10000
        rng = np.random.default_rng(42)
        x_mean_s = np.mean(stress_x)
        Sxx = np.sum((stress_x - x_mean_s) ** 2)

        if n_pts > 2:
            sigma2_samples = (
                (n_pts - 2) * s2 / rng.chisquare(max(1, n_pts - 2), size=n_mc)
            )
        else:
            sigma2_samples = np.full(n_mc, s2)

        a_samples = rng.normal(
            a, np.sqrt(sigma2_samples * (1 / n_pts + x_mean_s**2 / max(Sxx, 1e-15)))
        )
        b_samples = rng.normal(b, np.sqrt(sigma2_samples / max(Sxx, 1e-15)))

        life_use_samples = np.exp(a_samples + b_samples * x_use)
        life_use_samples = life_use_samples[
            np.isfinite(life_use_samples) & (life_use_samples > 0)
        ]

        if len(life_use_samples) > 0:
            life_mean = float(np.mean(life_use_samples))
            life_ci = (
                float(np.percentile(life_use_samples, (1 - ci_level) / 2 * 100)),
                float(np.percentile(life_use_samples, (1 + ci_level) / 2 * 100)),
            )
            b10_use = float(np.percentile(life_use_samples, 10))
        else:
            life_mean = life_use
            life_ci = (life_use * 0.5, life_use * 2.0)
            b10_use = life_use * 0.5

        af = (
            float(life_use / float(np.exp(a + b * stress_x[-1])))
            if len(stress_x) > 0
            else 1.0
        )
        verdict = "PASS"

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN ACCELERATED LIFE TESTING<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> {model_type}\n"
        summary += (
            f"<<COLOR:highlight>>Stress levels:<</COLOR>> {len(unique_stresses)}\n"
        )
        summary += f"<<COLOR:highlight>>Use stress:<</COLOR>> {use_stress}\n"
        summary += f"<<COLOR:highlight>>R\u00b2:<</COLOR>> {r2:.4f}\n\n"
        summary += f"<<COLOR:highlight>>Life at use condition:<</COLOR>> <<COLOR:good>>{life_mean:.1f}<</COLOR>> (CI: {life_ci[0]:.1f}\u2013{life_ci[1]:.1f})\n"
        summary += f"<<COLOR:highlight>>B10 at use:<</COLOR>> {b10_use:.1f}\n"
        summary += f"<<COLOR:highlight>>Acceleration factor:<</COLOR>> {af:.1f}\u00d7\n"

        result["summary"] = summary
        result["statistics"] = {
            "model": model_type,
            "r2": r2,
            "intercept": float(a),
            "slope": float(b),
            "life_at_use": life_mean,
            "life_ci": list(life_ci),
            "b10_use": b10_use,
            "acceleration_factor": af,
        }
        result["guide_observation"] = (
            f"Bayes ALT ({model_type}): life at use={life_mean:.1f}, B10={b10_use:.1f}, AF={af:.1f}\u00d7"
        )
        result["narrative"] = _narrative(
            verdict,
            f"{model_type.title()} model (R\u00b2 = {r2:.3f}). At use stress = {use_stress}, "
            f"characteristic life = {life_mean:.1f} ({int(ci_level * 100)}% CI: {life_ci[0]:.1f}\u2013{life_ci[1]:.1f}). "
            f"B10 = {b10_use:.1f}. Acceleration factor = {af:.1f}\u00d7.",
            [
                "Verify model adequacy with residual plots",
                "Consider adding intermediate stress levels for better fit",
                "Validate extrapolation with limited use-condition testing",
            ],
        )

        # Plot 1: Life vs Stress with credible band
        x_plot = np.linspace(
            min(min(stress_x), x_use) * 0.9, max(max(stress_x), x_use) * 1.1, 100
        )
        log_life_fit = a + b * x_plot
        log_life_hi = np.percentile(
            a_samples[:, None] + b_samples[:, None] * x_plot[None, :],
            (1 + ci_level) / 2 * 100,
            axis=0,
        )
        log_life_lo = np.percentile(
            a_samples[:, None] + b_samples[:, None] * x_plot[None, :],
            (1 - ci_level) / 2 * 100,
            axis=0,
        )

        if model_type == "arrhenius":
            x_labels = (1.0 / x_plot).tolist()
            obs_x = (1.0 / np.array(stress_x)).tolist()
        else:
            x_labels = np.exp(x_plot).tolist()
            obs_x = np.exp(np.array(stress_x)).tolist()

        result["plots"].append(
            {
                "title": "Life vs Stress (Log Scale)",
                "data": [
                    {
                        "type": "scatter",
                        "x": x_labels,
                        "y": np.exp(log_life_hi).tolist(),
                        "line": {"width": 0},
                        "showlegend": False,
                        "name": "Upper",
                    },
                    {
                        "type": "scatter",
                        "x": x_labels,
                        "y": np.exp(log_life_lo).tolist(),
                        "fill": "tonexty",
                        "fillcolor": _rgba(SVEND_COLORS[0], 0.2),
                        "line": {"width": 0},
                        "name": f"{int(ci_level * 100)}% CI",
                    },
                    {
                        "type": "scatter",
                        "x": x_labels,
                        "y": np.exp(log_life_fit).tolist(),
                        "line": {"color": SVEND_COLORS[0], "width": 2},
                        "name": "Regression",
                    },
                    {
                        "type": "scatter",
                        "x": obs_x,
                        "y": np.exp(log_lives).tolist(),
                        "mode": "markers",
                        "marker": {"color": COLOR_BAD, "size": 8},
                        "name": "Observed",
                    },
                ],
                "layout": {
                    "height": 350,
                    "xaxis": {"title": "Stress", "type": "log"},
                    "yaxis": {"title": "Characteristic Life", "type": "log"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": use_stress,
                            "x1": use_stress,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": COLOR_GOOD, "dash": "dash"},
                        }
                    ],
                    "annotations": [
                        {
                            "x": np.log10(max(use_stress, 0.01)),
                            "y": 1,
                            "yref": "paper",
                            "text": f"Use={use_stress}",
                            "showarrow": False,
                            "yanchor": "bottom",
                            "font": {"color": COLOR_GOOD},
                        }
                    ],
                },
            }
        )

        # Plot 2: Posterior survival at use condition
        try:
            shape_overall, _, _ = stats.weibull_min.fit(times_alt, floc=0)
        except Exception:
            shape_overall = 1.5
        t_surv = np.linspace(0, life_ci[1] * 1.5, 200)
        surv_mean = np.exp(-((t_surv / life_mean) ** shape_overall))
        surv_lo = (
            np.exp(-((t_surv / life_ci[0]) ** shape_overall))
            if life_ci[0] > 0
            else np.ones_like(t_surv)
        )
        surv_hi = np.exp(-((t_surv / life_ci[1]) ** shape_overall))
        result["plots"].append(
            {
                "title": "Posterior Survival at Use Condition",
                "data": [
                    {
                        "type": "scatter",
                        "x": t_surv.tolist(),
                        "y": surv_hi.tolist(),
                        "line": {"width": 0},
                        "showlegend": False,
                        "name": "Upper",
                    },
                    {
                        "type": "scatter",
                        "x": t_surv.tolist(),
                        "y": surv_lo.tolist(),
                        "fill": "tonexty",
                        "fillcolor": _rgba(SVEND_COLORS[1], 0.2),
                        "line": {"width": 0},
                        "name": f"{int(ci_level * 100)}% CI",
                    },
                    {
                        "type": "scatter",
                        "x": t_surv.tolist(),
                        "y": surv_mean.tolist(),
                        "line": {"color": SVEND_COLORS[1], "width": 2},
                        "name": "S(t) at Use",
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Survival Probability"},
                },
            }
        )

        # Plot 3: B10 posterior histogram
        b10_samples = life_use_samples * ((-np.log(0.9)) ** (1.0 / shape_overall))
        hist_b10, edges_b10 = np.histogram(b10_samples[:5000], bins=60, density=True)
        centers_b10 = (edges_b10[:-1] + edges_b10[1:]) / 2
        result["plots"].append(
            {
                "title": "B10 Life Posterior at Use Condition",
                "data": [
                    {
                        "type": "bar",
                        "x": centers_b10.tolist(),
                        "y": hist_b10.tolist(),
                        "marker": {"color": _rgba(SVEND_COLORS[2], 0.7)},
                        "name": "B10",
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "B10 Life"},
                    "yaxis": {"title": "Density"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": b10_use,
                            "x1": b10_use,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": COLOR_GOOD, "dash": "dash", "width": 2},
                        }
                    ],
                },
            }
        )

        result["education"] = {
            "title": "Understanding Bayesian Accelerated Life Testing (ALT)",
            "content": (
                "<dl>"
                "<dt>What is accelerated life testing?</dt>"
                "<dd>ALT stresses units beyond normal operating conditions (higher temperature, "
                "voltage, humidity) to induce failures faster. A physics-of-failure model "
                "(Arrhenius, Eyring) then extrapolates results back to use conditions.</dd>"
                "<dt>What is the acceleration model?</dt>"
                "<dd><strong>Arrhenius</strong>: relates failure rate to temperature via activation "
                "energy — standard for thermal degradation. <strong>Eyring</strong>: extends to "
                "multiple stresses (temperature + humidity). The Bayesian posterior captures "
                "uncertainty in the acceleration parameters.</dd>"
                "<dt>What is B10 life at use conditions?</dt>"
                "<dd>The time by which 10% of units are expected to fail under normal operating "
                "conditions. The Bayesian posterior on B10 gives a credible interval — critical "
                "for warranty and design life decisions.</dd>"
                "<dt>Why Bayesian ALT?</dt>"
                "<dd>ALT typically has few failures at each stress level. Bayesian estimation "
                "handles small samples naturally, propagates parameter uncertainty into life "
                "predictions, and avoids the overconfidence of point extrapolations.</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_comprisk":
        # Bayesian Competing Risks (Dirichlet-Multinomial + per-mode Weibull)
        time_col = config.get("var1")
        event_col = config.get("var2")
        if not time_col or not event_col:
            result["summary"] = "Error: Select time and event/mode columns."
            return result

        df_cr = df[[time_col, event_col]].dropna()
        times_cr = df_cr[time_col].values.astype(float)
        events_cr = df_cr[event_col].values.astype(int)
        mask = times_cr > 0
        times_cr, events_cr = times_cr[mask], events_cr[mask]
        n_cr = len(times_cr)

        if n_cr < 5:
            result["summary"] = "Error: Need at least 5 observations."
            return result

        modes = sorted([m for m in np.unique(events_cr) if m > 0])
        n_modes = len(modes)
        if n_modes < 2:
            result["summary"] = (
                "Error: Need at least 2 failure modes (event: 0=censored, 1,2,...=modes)."
            )
            return result

        mode_counts = np.array([np.sum(events_cr == m) for m in modes], dtype=float)
        n_failed = int(np.sum(events_cr > 0))
        alpha_post = mode_counts + 1
        mode_probs = alpha_post / alpha_post.sum()

        n_mc = 5000
        rng = np.random.default_rng(42)
        dir_samples = rng.dirichlet(alpha_post, size=n_mc)

        mode_names = [f"Mode {m}" for m in modes]
        mode_ci = []
        for j in range(n_modes):
            lo = float(np.percentile(dir_samples[:, j], (1 - ci_level) / 2 * 100))
            hi = float(np.percentile(dir_samples[:, j], (1 + ci_level) / 2 * 100))
            mode_ci.append((lo, hi))

        # Per-mode Weibull grid posterior
        n_grid = 80
        mode_weibull = {}
        for j, m in enumerate(modes):
            t_m = times_cr[events_cr == m]
            if len(t_m) < 2:
                mode_weibull[m] = {
                    "shape": 1.0,
                    "scale": float(np.median(times_cr)),
                    "shape_ci": (0.5, 2.0),
                    "scale_ci": (1.0, float(np.max(times_cr))),
                }
                continue
            try:
                shape_mle, _, scale_mle = stats.weibull_min.fit(t_m, floc=0)
            except Exception:
                shape_mle, scale_mle = 1.0, float(np.median(t_m))

            b_range = np.linspace(max(0.1, shape_mle * 0.3), shape_mle * 3, n_grid)
            e_range = np.linspace(max(0.1, scale_mle * 0.3), scale_mle * 3, n_grid)
            bg, eg = np.meshgrid(b_range, e_range)
            ll = np.zeros_like(bg)
            for t_i in t_m:
                ll += (
                    np.log(bg + 1e-15)
                    + (bg - 1) * np.log(t_i + 1e-15)
                    - bg * np.log(eg + 1e-15)
                )
                ll -= (t_i / (eg + 1e-15)) ** bg
            lp = ll - ll.max()
            post = np.exp(lp)
            post /= post.sum()

            b_marg = post.sum(axis=0)
            b_marg /= b_marg.sum()
            e_marg = post.sum(axis=1)
            e_marg /= e_marg.sum()

            b_mean = float(np.sum(b_range * b_marg))
            e_mean = float(np.sum(e_range * e_marg))
            b_ci_m = (
                float(b_range[np.searchsorted(np.cumsum(b_marg), (1 - ci_level) / 2)]),
                float(
                    b_range[
                        min(
                            n_grid - 1,
                            np.searchsorted(np.cumsum(b_marg), (1 + ci_level) / 2),
                        )
                    ]
                ),
            )
            e_ci_m = (
                float(e_range[np.searchsorted(np.cumsum(e_marg), (1 - ci_level) / 2)]),
                float(
                    e_range[
                        min(
                            n_grid - 1,
                            np.searchsorted(np.cumsum(e_marg), (1 + ci_level) / 2),
                        )
                    ]
                ),
            )
            mode_weibull[m] = {
                "shape": b_mean,
                "scale": e_mean,
                "shape_ci": b_ci_m,
                "scale_ci": e_ci_m,
            }

        # CIF via MC
        t_eval = np.linspace(0, float(np.max(times_cr)) * 1.2, 150)
        dt_eval = np.diff(t_eval, prepend=0)
        cif_samples = {m: np.zeros((n_mc, len(t_eval))) for m in modes}
        for i in range(n_mc):
            p_i = dir_samples[i]
            h_total = np.zeros(len(t_eval))
            h_per_mode = {}
            for j, m in enumerate(modes):
                w = mode_weibull[m]
                h_j = (w["shape"] / w["scale"]) * ((t_eval + 1e-15) / w["scale"]) ** (
                    w["shape"] - 1
                )
                h_per_mode[m] = h_j
                h_total += p_i[j] * h_j

            cum_hazard = np.cumsum(h_total * dt_eval)
            S = np.exp(-cum_hazard)
            for j, m in enumerate(modes):
                integrand = S * p_i[j] * h_per_mode[m] * dt_eval
                cif_samples[m][i] = np.cumsum(integrand)

        dominant_mode = modes[int(np.argmax(mode_probs))]
        verdict = "PASS"

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN COMPETING RISKS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n_cr} ({n_failed} failed, {n_cr - n_failed} censored)\n"
        summary += f"<<COLOR:highlight>>Failure modes:<</COLOR>> {n_modes}\n\n"
        for j, m in enumerate(modes):
            w = mode_weibull[m]
            summary += f"  Mode {m}: P = {mode_probs[j]:.3f} (CI: {mode_ci[j][0]:.3f}\u2013{mode_ci[j][1]:.3f}), "
            summary += f"\u03b2 = {w['shape']:.2f}, \u03b7 = {w['scale']:.1f}\n"
        summary += f"\n<<COLOR:highlight>>Dominant mode:<</COLOR>> <<COLOR:warning>>Mode {dominant_mode}<</COLOR>> "
        summary += f"(P = {mode_probs[modes.index(dominant_mode)]:.3f})\n"

        result["summary"] = summary
        result["statistics"] = {
            "n_modes": n_modes,
            "dominant_mode": int(dominant_mode),
            "mode_probabilities": {
                str(m): float(mode_probs[j]) for j, m in enumerate(modes)
            },
            "mode_weibull": {str(m): mode_weibull[m] for m in modes},
        }
        result["guide_observation"] = (
            f"Bayes competing risks: {n_modes} modes, dominant=Mode {dominant_mode}"
        )
        result["narrative"] = _narrative(
            verdict,
            f"{n_modes} competing failure modes identified. "
            f"Dominant: Mode {dominant_mode} (P = {mode_probs[modes.index(dominant_mode)]:.3f}). "
            + "; ".join(
                [
                    f"Mode {m}: \u03b2={mode_weibull[m]['shape']:.2f}, \u03b7={mode_weibull[m]['scale']:.1f}"
                    for m in modes
                ]
            ),
            [
                "Focus corrective action on dominant failure mode",
                "Monitor mode proportions over time for shifts",
                "Use CIF (not 1-KM) for mode-specific risk estimates",
            ],
        )

        # Plot 1: CIF curves with credible bands
        cif_traces = []
        for j, m in enumerate(modes):
            cif_mean = np.mean(cif_samples[m], axis=0)
            cif_lo = np.percentile(cif_samples[m], (1 - ci_level) / 2 * 100, axis=0)
            cif_hi = np.percentile(cif_samples[m], (1 + ci_level) / 2 * 100, axis=0)
            color = SVEND_COLORS[j % len(SVEND_COLORS)]
            cif_traces.extend(
                [
                    {
                        "type": "scatter",
                        "x": t_eval.tolist(),
                        "y": cif_hi.tolist(),
                        "line": {"width": 0},
                        "showlegend": False,
                        "name": f"M{m} Upper",
                    },
                    {
                        "type": "scatter",
                        "x": t_eval.tolist(),
                        "y": cif_lo.tolist(),
                        "fill": "tonexty",
                        "fillcolor": _rgba(color, 0.15),
                        "line": {"width": 0},
                        "showlegend": False,
                        "name": f"M{m} CI",
                    },
                    {
                        "type": "scatter",
                        "x": t_eval.tolist(),
                        "y": cif_mean.tolist(),
                        "line": {"color": color, "width": 2},
                        "name": f"Mode {m}",
                    },
                ]
            )
        result["plots"].append(
            {
                "title": "Cumulative Incidence Functions (CIF) with Credible Bands",
                "data": cif_traces,
                "layout": {
                    "height": 400,
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Cumulative Incidence"},
                },
            }
        )

        # Plot 2: Mode probability bar with CI
        result["plots"].append(
            {
                "title": "Failure Mode Probabilities",
                "data": [
                    {
                        "type": "bar",
                        "x": mode_names,
                        "y": mode_probs.tolist(),
                        "error_y": {
                            "type": "data",
                            "symmetric": False,
                            "array": [ci[1] - p for p, ci in zip(mode_probs, mode_ci)],
                            "arrayminus": [
                                p - ci[0] for p, ci in zip(mode_probs, mode_ci)
                            ],
                        },
                        "marker": {
                            "color": [
                                SVEND_COLORS[j % len(SVEND_COLORS)]
                                for j in range(n_modes)
                            ]
                        },
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Failure Mode"},
                    "yaxis": {"title": "Probability"},
                },
            }
        )

        # Plot 3: Mode-specific Weibull shape posteriors
        shape_traces = []
        for j, m in enumerate(modes):
            w = mode_weibull[m]
            b_lo, b_hi = w["shape_ci"]
            b_std_m = (b_hi - b_lo) / (2 * 1.96)
            b_x = np.linspace(b_lo * 0.5, b_hi * 1.5, 100)
            b_pdf = stats.norm.pdf(b_x, w["shape"], max(b_std_m, 0.01))
            shape_traces.append(
                {
                    "type": "scatter",
                    "x": b_x.tolist(),
                    "y": b_pdf.tolist(),
                    "fill": "tozeroy",
                    "fillcolor": _rgba(SVEND_COLORS[j % len(SVEND_COLORS)], 0.15),
                    "line": {"color": SVEND_COLORS[j % len(SVEND_COLORS)], "width": 2},
                    "name": f"Mode {m}",
                }
            )
        result["plots"].append(
            {
                "title": "Mode-Specific Weibull Shape (\u03b2) Posteriors",
                "data": shape_traces,
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Shape (\u03b2)"},
                    "yaxis": {"title": "Density"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 1,
                            "x1": 1,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": "#888", "dash": "dot"},
                        }
                    ],
                },
            }
        )

        result["education"] = {
            "title": "Understanding Bayesian Competing Risks",
            "content": (
                "<dl>"
                "<dt>What is competing risks analysis?</dt>"
                "<dd>When a unit can fail from multiple causes (e.g. bearing wear, electrical fault, "
                "corrosion), competing risks analysis decomposes overall failure into cause-specific "
                "contributions. Each failure mode 'competes' to be the first to cause failure.</dd>"
                "<dt>How does the Bayesian model work?</dt>"
                "<dd>A <em>Dirichlet-Multinomial</em> model estimates the probability of each failure "
                "mode, while per-mode <em>Weibull</em> distributions estimate the time-to-failure "
                "characteristics for each cause. The posterior captures uncertainty in both.</dd>"
                "<dt>What is the Cumulative Incidence Function (CIF)?</dt>"
                "<dd>The probability of failing from a specific cause by time t, accounting for "
                "the presence of other competing causes. Unlike cause-specific hazards, CIFs "
                "sum to the overall failure probability — giving a complete decomposition.</dd>"
                "<dt>When to use this?</dt>"
                "<dd>Warranty root cause decomposition, maintenance strategy design (which failure "
                "mode to address first), product redesign prioritisation, or any reliability "
                "analysis where multiple distinct failure mechanisms are at play.</dd>"
                "</dl>"
            ),
        }

    elif analysis_id == "bayes_ewma":
        # Bayesian EWMA — EWMA smoothing with posterior inference for shift detection
        measurement = config.get("measurement")
        target = config.get("target")
        lambda_param = float(config.get("lambda_param", 0.2))
        L = float(config.get("L", 3))
        prior_scale_name = config.get("prior_scale", "medium")

        data = df[measurement].dropna().values
        n = len(data)

        if target is None or target == 0:
            target = float(np.mean(data))
        else:
            target = float(target)

        sigma = float(np.std(data, ddof=1))
        if sigma < 1e-15:
            sigma = 1.0

        # Prior precision scaling
        scale_map = {"tight": 5.0, "medium": 1.0, "wide": 0.2}
        kappa_0 = scale_map.get(prior_scale_name, 1.0)

        # ── EWMA smoothing (same as spc.py) ──
        ewma = np.zeros(n)
        ewma[0] = lambda_param * data[0] + (1 - lambda_param) * target
        for i in range(1, n):
            ewma[i] = lambda_param * data[i] + (1 - lambda_param) * ewma[i - 1]

        # Classical variable control limits
        factor = lambda_param / (2 - lambda_param)
        indices = np.arange(1, n + 1)
        cl_sigma = sigma * np.sqrt(factor * (1 - (1 - lambda_param) ** (2 * indices)))
        ucl = target + L * cl_sigma
        lcl = target - L * cl_sigma
        ucl_ss = target + L * sigma * np.sqrt(factor)
        lcl_ss = target - L * sigma * np.sqrt(factor)

        # OOC detection
        ooc_indices = [i for i in range(n) if ewma[i] > ucl[i] or ewma[i] < lcl[i]]

        # ── Bayesian posterior inference ──
        # Conjugate Normal posterior: prior N(target, σ²/κ₀)
        # At each step, update with EWMA observation as a pseudo-observation
        # with effective sample size proportional to smoothing weight
        ewma_var = sigma**2 * factor  # steady-state EWMA variance
        posterior_means = np.zeros(n)
        posterior_vars = np.zeros(n)
        shift_probs = np.zeros(n)

        prior_var = sigma**2 / kappa_0
        prior_mean = target

        for i in range(n):
            # Effective observation precision from EWMA
            obs_var = sigma**2 * factor * (1 - (1 - lambda_param) ** (2 * (i + 1)))
            if obs_var < 1e-15:
                obs_var = ewma_var

            obs_precision = 1.0 / obs_var
            prior_precision = 1.0 / prior_var

            # Posterior update
            post_precision = prior_precision + obs_precision
            post_var = 1.0 / post_precision
            post_mean = post_var * (
                prior_precision * prior_mean + obs_precision * ewma[i]
            )

            posterior_means[i] = post_mean
            posterior_vars[i] = post_var

            # P(|μ - target| > 1σ) — probability of meaningful shift
            delta = sigma  # 1-sigma shift threshold
            post_std = np.sqrt(post_var)
            if post_std > 0:
                p_above = 1 - stats.norm.cdf(target + delta, post_mean, post_std)
                p_below = stats.norm.cdf(target - delta, post_mean, post_std)
                shift_probs[i] = p_above + p_below
            else:
                shift_probs[i] = 1.0 if abs(post_mean - target) > delta else 0.0

            # Sequential update: current posterior becomes next prior
            prior_mean = post_mean
            prior_var = post_var + ewma_var * 0.1  # add process noise

        # ── Overall Bayes Factor ──
        # Compare H₁: mean ≠ target vs H₀: mean = target
        # Using final posterior: BF₁₀ ≈ P(data|H₁)/P(data|H₀)
        # Savage-Dickey density ratio at μ = target
        final_mean = posterior_means[-1]
        final_std = np.sqrt(posterior_vars[-1])
        prior_std_0 = sigma / np.sqrt(kappa_0)

        # Density of posterior at target / density of prior at target
        post_density_at_target = stats.norm.pdf(target, final_mean, final_std)
        prior_density_at_target = stats.norm.pdf(target, target, prior_std_0)

        if post_density_at_target > 0:
            bf10 = prior_density_at_target / post_density_at_target
        else:
            bf10 = 100.0  # very strong evidence for shift

        bf10 = max(bf10, 0.01)  # floor

        # BF label
        if bf10 > 100:
            bf_label = "extreme"
        elif bf10 > 30:
            bf_label = "very strong"
        elif bf10 > 10:
            bf_label = "strong"
        elif bf10 > 3:
            bf_label = "moderate"
        elif bf10 > 1:
            bf_label = "anecdotal"
        else:
            bf_label = "supports null"

        # Credible intervals
        ci_mult = z  # z from ci_level at top of function
        ci_upper = posterior_means + ci_mult * np.sqrt(posterior_vars)
        ci_lower = posterior_means - ci_mult * np.sqrt(posterior_vars)

        max_shift_prob = float(np.max(shift_probs))

        # ── Statistics ──
        result["statistics"] = {
            "n": n,
            "lambda_param": lambda_param,
            "L": L,
            "target": float(target),
            "sigma": sigma,
            "n_ooc": len(ooc_indices),
            "ucl_steady": float(ucl_ss),
            "lcl_steady": float(lcl_ss),
            "max_shift_prob": max_shift_prob,
            "bf10": float(bf10),
            "bf_label": bf_label,
        }

        # ── Summary ──
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN EWMA ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target:.4f}\n"
        summary += f"<<COLOR:dim>>λ = {lambda_param}, L = {L}, prior = {prior_scale_name}<</COLOR>>\n\n"
        summary += "<<COLOR:accent>>── Control Limits (steady-state) ──<</COLOR>>\n"
        summary += f"  UCL: {ucl_ss:.4f}\n"
        summary += f"  LCL: {lcl_ss:.4f}\n\n"
        summary += "<<COLOR:accent>>── Bayesian Inference ──<</COLOR>>\n"
        summary += f"  BF₁₀ = {bf10:.2f} ({bf_label} evidence)\n"
        summary += f"  Max P(shift) = {max_shift_prob:.4f}\n"
        summary += f"  Out-of-control points: {len(ooc_indices)}\n\n"

        if len(ooc_indices) == 0 and bf10 < 3:
            summary += "<<COLOR:success>>Process appears stable — no Bayesian evidence of shift<</COLOR>>\n"
        elif bf10 >= 10:
            summary += (
                "<<COLOR:error>>Strong Bayesian evidence of process shift<</COLOR>>\n"
            )
        elif bf10 >= 3:
            summary += "<<COLOR:warning>>Moderate Bayesian evidence of process shift<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>OOC points detected but Bayesian evidence is {bf_label}<</COLOR>>\n"

        result["summary"] = summary

        # ── Guide observation ──
        if len(ooc_indices) == 0 and bf10 < 3:
            result["guide_observation"] = (
                f"Bayesian EWMA: process stable. BF₁₀={bf10:.2f} ({bf_label}), 0 OOC points."
            )
        else:
            result["guide_observation"] = (
                f"Bayesian EWMA: {len(ooc_indices)} OOC point{'s' if len(ooc_indices) != 1 else ''}. "
                f"BF₁₀={bf10:.2f} ({bf_label}), max P(shift)={max_shift_prob:.4f}."
            )

        # ── Narrative ──
        if len(ooc_indices) == 0 and bf10 < 3:
            result["narrative"] = _narrative(
                "Process is in statistical control",
                f"No EWMA points exceed control limits (λ={lambda_param}, L={L}). "
                f"Bayesian analysis confirms stability: BF₁₀ = {bf10:.2f} ({bf_label}), "
                f"maximum posterior shift probability = {max_shift_prob:.4f}.",
                next_steps="Process is stable — EWMA is sensitive to small sustained shifts; "
                "the Bayesian posterior provides additional confidence in this conclusion.",
                chart_guidance="The EWMA line smooths out noise. The shaded band shows the 95% credible interval "
                "for the true process mean. The shift probability plot shows where the posterior "
                "concentrates away from target.",
            )
        else:
            result["narrative"] = _narrative(
                f"Bayesian EWMA — {'strong' if bf10 >= 10 else 'moderate' if bf10 >= 3 else 'weak'} evidence of shift",
                f"The smoothed mean has {'exceeded control limits' if ooc_indices else 'shifted'} "
                f"with BF₁₀ = {bf10:.2f} ({bf_label}). "
                f"Maximum posterior probability of a 1σ shift = {max_shift_prob:.4f}.",
                next_steps="Identify when the drift began (peak in shift probability plot) and "
                "correlate with process changes. The Bayesian credible intervals show "
                "where uncertainty about the true mean is highest.",
                chart_guidance="Red diamonds mark classical OOC points. The purple shift probability "
                "curve shows Bayesian confidence in a process shift at each observation. "
                "Values near 1.0 indicate near-certainty of drift.",
            )

        # ── Plots ──
        x_list = list(range(n))

        # Plot 1: EWMA Control Chart
        ewma_chart_data = [
            {
                "type": "scatter",
                "x": x_list,
                "y": ewma.tolist(),
                "mode": "lines+markers",
                "name": "EWMA",
                "marker": {
                    "color": _rgba(COLOR_GOOD, 0.4),
                    "size": 5,
                    "line": {"color": COLOR_GOOD, "width": 1.5},
                },
                "line": {"color": COLOR_GOOD},
            },
            {
                "type": "scatter",
                "x": x_list,
                "y": [target] * n,
                "mode": "lines",
                "name": "Target",
                "line": {"color": COLOR_REFERENCE, "width": 1, "dash": "dot"},
            },
            {
                "type": "scatter",
                "x": x_list,
                "y": ucl.tolist(),
                "mode": "lines",
                "name": "UCL",
                "line": {"color": COLOR_BAD, "width": 1.5, "dash": "dash"},
            },
            {
                "type": "scatter",
                "x": x_list,
                "y": lcl.tolist(),
                "mode": "lines",
                "name": "LCL",
                "line": {"color": COLOR_BAD, "width": 1.5, "dash": "dash"},
            },
        ]

        # Add OOC markers
        if ooc_indices:
            ewma_chart_data.append(
                {
                    "type": "scatter",
                    "x": ooc_indices,
                    "y": [ewma[i] for i in ooc_indices],
                    "mode": "markers",
                    "name": "OOC",
                    "marker": {"color": COLOR_BAD, "size": 10, "symbol": "diamond"},
                }
            )

        result["plots"].append(
            {
                "title": "Bayesian EWMA Control Chart",
                "data": ewma_chart_data,
                "layout": {
                    "xaxis": {"title": "Observation"},
                    "yaxis": {"title": measurement},
                },
            }
        )

        # Plot 2: Posterior Shift Probability
        result["plots"].append(
            {
                "title": "Posterior Probability of Shift",
                "data": [
                    {
                        "type": "scatter",
                        "x": x_list,
                        "y": shift_probs.tolist(),
                        "mode": "lines",
                        "name": "P(shift > 1σ)",
                        "line": {"color": SVEND_COLORS[4], "width": 2},
                    },
                    {
                        "type": "scatter",
                        "x": x_list,
                        "y": [0.95] * n,
                        "mode": "lines",
                        "name": "95% threshold",
                        "line": {"color": COLOR_WARNING, "width": 1, "dash": "dash"},
                    },
                ],
                "layout": {
                    "xaxis": {"title": "Observation"},
                    "yaxis": {"title": "P(shift)", "range": [0, 1.05]},
                },
            }
        )

        # Plot 3: Credible Interval Band
        result["plots"].append(
            {
                "title": f"{int(ci_level * 100)}% Credible Interval for Process Mean",
                "data": [
                    {
                        "type": "scatter",
                        "x": x_list,
                        "y": ci_upper.tolist(),
                        "mode": "lines",
                        "name": f"Upper {int(ci_level * 100)}% CI",
                        "line": {"color": _rgba(SVEND_COLORS[1], 0.3), "width": 0},
                    },
                    {
                        "type": "scatter",
                        "x": x_list,
                        "y": ci_lower.tolist(),
                        "mode": "lines",
                        "name": f"Lower {int(ci_level * 100)}% CI",
                        "line": {"color": _rgba(SVEND_COLORS[1], 0.3), "width": 0},
                        "fill": "tonexty",
                        "fillcolor": _rgba(SVEND_COLORS[1], 0.15),
                    },
                    {
                        "type": "scatter",
                        "x": x_list,
                        "y": posterior_means.tolist(),
                        "mode": "lines",
                        "name": "Posterior Mean",
                        "line": {"color": SVEND_COLORS[1], "width": 2},
                    },
                    {
                        "type": "scatter",
                        "x": x_list,
                        "y": [target] * n,
                        "mode": "lines",
                        "name": "Target",
                        "line": {"color": COLOR_REFERENCE, "width": 1, "dash": "dot"},
                    },
                ],
                "layout": {
                    "xaxis": {"title": "Observation"},
                    "yaxis": {"title": "Process Mean"},
                },
            }
        )

        # ── Education ──
        result["education"] = {
            "title": "Understanding Bayesian EWMA",
            "content": (
                "<dl>"
                "<dt>What is Bayesian EWMA?</dt>"
                "<dd>It combines the classical EWMA (Exponentially Weighted Moving Average) chart "
                "with Bayesian posterior inference. Instead of just flagging points outside ±Lσ limits, "
                "it estimates the <em>posterior probability</em> that the process mean has shifted at "
                "each observation — giving you a continuous measure of confidence in process stability.</dd>"
                "<dt>How is it different from classical EWMA?</dt>"
                "<dd>Classical EWMA gives binary signals (in-control/out-of-control). Bayesian EWMA "
                "gives <em>probabilities</em>: 'there is a 92% posterior probability the process mean "
                "has shifted by more than 1σ from target.' It also provides credible intervals around "
                "the estimated process mean — the Bayesian analogue of confidence intervals.</dd>"
                "<dt>What is the Bayes Factor here?</dt>"
                "<dd>BF₁₀ compares two hypotheses: H₁ (the process has shifted from target) vs "
                "H₀ (the process is still at target). BF₁₀ &gt; 10 is strong evidence of shift; "
                "BF₁₀ &lt; 1/3 is evidence of stability. It uses the Savage-Dickey density ratio "
                "from the posterior distribution.</dd>"
                "<dt>When to use Bayesian EWMA over classical?</dt>"
                "<dd>When you need to <em>quantify</em> shift evidence rather than just detect it. "
                "Particularly useful for small sample sizes where classical control limits may be "
                "unreliable, or when you need to communicate shift risk probabilistically to "
                "decision-makers (e.g., 'P(shifted) = 0.87' vs 'one OOC point').</dd>"
                "</dl>"
            ),
        }

    return result

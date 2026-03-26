"""Bayesian regression analyses — linear, logistic, and Poisson rate models."""

import numpy as np
from scipy import stats
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..common import (
    _narrative,
)


def run_bayes_regression(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Bayesian Linear Regression with credible intervals
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
        summary += (
            f"  {feat:<20} β = {mean:>8.4f}  [{ci_low:>8.4f}, {ci_high:>8.4f}] {sig}\n"
        )

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

    return result


def run_bayes_logistic(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Bayesian Logistic Regression (Laplace approximation)
    target = config.get("target")
    features = config.get("features", [])

    if not target or not features:
        result["summary"] = "Error: Select a binary target and at least one feature."
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
    H = X_scaled.T @ np.diag(W) @ X_scaled + (1 / prior_width) * np.eye(len(features))
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
    summary += "<<COLOR:accent>>── Coefficient Posteriors (per std dev) ──<</COLOR>>\n"
    _bl_header = "P(β>0)"
    _bl_rule = "─" * 60
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
        f"Bayesian Logistic — accuracy {accuracy:.1%}, strongest predictor: {strongest}",
        f"The model classifies {classes[0]} vs {classes[1]} with {accuracy:.1%} accuracy. "
        f"<strong>{strongest}</strong> has the largest effect (OR = {strongest_or:.2f}, "
        f"P(β > 0) = {p_positive[strongest_idx]:.1%}). "
        + (
            f"An OR of {strongest_or:.2f} means a 1-SD increase in {strongest} "
            f"{'increases' if strongest_or > 1 else 'decreases'} the odds by {abs(strongest_or - 1) * 100:.0f}%."
            if not np.isnan(strongest_or)
            else ""
        ),
        next_steps="Odds ratios > 1 increase the probability of the target class. "
        "Features with P(β>0) near 50% have uncertain direction — more data needed.",
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
                            float(or_ci_high[i] - odds_ratios[i]) for i in sorted_idx
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
            "title": "P(β > 0) per Feature",
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
                "yaxis": {"title": "P(β > 0)", "range": [0, 1.05]},
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

    return result


def run_bayes_poisson(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

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
    summary += (
        f"  Posterior rate: {rate1_mean:.4f} [{rate1_ci[0]:.4f}, {rate1_ci[1]:.4f}]\n"
    )

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

        summary += "\n<<COLOR:accent>>── Comparison ──<</COLOR>>\n"
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
            {"rate2_mean": rate2_mean, "p_greater": p_greater, "rate_ratio": rr_mean}
        )
    result["statistics"] = stat_dict

    if two_sample:
        result["guide_observation"] = (
            f"Bayesian Poisson: rate₁={rate1_mean:.4f}, rate₂={rate2_mean:.4f}. P(rate₁ > rate₂) = {p_greater:.1%}."
        )
        higher = var1 if p_greater > 0.5 else var2
        prob_higher = max(p_greater, 1 - p_greater)
        result["narrative"] = _narrative(
            f"Bayesian Poisson — P({higher} rate is higher) = {prob_higher:.1%}",
            f"Posterior rates: <strong>{var1}</strong> = {rate1_mean:.4f} (95% CI: {rate1_ci[0]:.4f}–{rate1_ci[1]:.4f}), "
            f"<strong>{var2}</strong> = {rate2_mean:.4f} (95% CI: {rate2_ci[0]:.4f}–{rate2_ci[1]:.4f}). "
            f"Rate ratio = {rr_mean:.3f} (95% CI: {rr_ci[0]:.3f}–{rr_ci[1]:.3f}).",
            next_steps="A rate ratio CI excluding 1.0 confirms the rates differ. "
            "Consider whether the exposure measure adequately accounts for opportunity.",
            chart_guidance="Overlapping posteriors = uncertain which rate is higher. Separated posteriors = clear difference.",
        )
    else:
        result["guide_observation"] = (
            f"Bayesian Poisson: rate = {rate1_mean:.4f} (95% CI: {rate1_ci[0]:.4f}–{rate1_ci[1]:.4f})."
        )
        result["narrative"] = _narrative(
            f"Bayesian Poisson Rate = {rate1_mean:.4f}",
            f"Posterior rate: {rate1_mean:.4f} (95% credible interval: {rate1_ci[0]:.4f} to {rate1_ci[1]:.4f}). "
            f"Based on {int(total1)} events in {exposure1:.0f} exposure units.",
            next_steps="Use this posterior to set Bayesian control limits on count charts or to plan inspection intervals.",
            chart_guidance="The curve shows the posterior belief about the true rate. Width reflects uncertainty — more data narrows it.",
        )

    result["plots"].append(
        {
            "title": "Posterior Rate Distribution",
            "data": plot_data,
            "layout": {
                "height": 300,
                "xaxis": {"title": "Rate (λ)"},
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

    return result

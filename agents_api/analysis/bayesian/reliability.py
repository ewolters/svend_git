"""Bayesian reliability analysis functions extracted from bayesian.py.

Each function corresponds to a DSW analysis branch and returns a result dict
with keys: plots, summary, guide_observation, and optionally statistics,
narrative, education.
"""

import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_fn

from ..common import (
    COLOR_BAD,
    COLOR_GOOD,
    COLOR_REFERENCE,
    COLOR_WARNING,
    SVEND_COLORS,
    _narrative,
    _rgba,
)


def run_bayes_survival(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Bayesian Weibull Survival Analysis (grid posterior)
    time_col = config.get("var1")
    event_col = config.get("var2")

    times = df[time_col].dropna().values.astype(float)
    if event_col and event_col in df.columns:
        events = df[event_col].loc[df[time_col].notna()].fillna(1).values.astype(float)
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
            np.log(beta_grid + 1e-15) + (beta_grid - 1) * np.log(t_i + 1e-15) - beta_grid * np.log(eta_grid + 1e-15)
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
        float(beta_range[np.searchsorted(np.cumsum(beta_marginal), (1 - ci_level) / 2)]),
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
        float(eta_range[np.searchsorted(np.cumsum(eta_marginal), (1 - ci_level) / 2)]),
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
    summary += (
        f"<<COLOR:text>>Observations:<</COLOR>> {n} ({int(events.sum())} events, {int(n - events.sum())} censored)\n\n"
    )
    summary += "<<COLOR:accent>>\u2500\u2500 Shape (\u03b2) Posterior \u2500\u2500<</COLOR>>\n"
    summary += f"  Mean: {beta_mean:.3f}  [{beta_ci[0]:.3f}, {beta_ci[1]:.3f}]\n"
    summary += f"  Phase: {phase}\n\n"
    summary += "<<COLOR:accent>>\u2500\u2500 Scale (\u03b7) Posterior \u2500\u2500<</COLOR>>\n"
    summary += f"  Mean: {eta_mean:.2f}  [{eta_ci[0]:.2f}, {eta_ci[1]:.2f}]\n\n"
    summary += "<<COLOR:accent>>\u2500\u2500 Reliability Metrics \u2500\u2500<</COLOR>>\n"
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
            "<dt>What does the shape parameter (\u03b2) tell you?</dt>"
            "<dd><strong>\u03b2 &lt; 1</strong>: decreasing failure rate (infant mortality). "
            "<strong>\u03b2 = 1</strong>: constant failure rate (exponential, random failures). "
            "<strong>\u03b2 &gt; 1</strong>: increasing failure rate (wear-out). The dashed line "
            "at \u03b2=1 on the posterior plot marks the exponential boundary.</dd>"
            "<dt>What are censored observations?</dt>"
            "<dd>Units that were removed from test or had not yet failed when observation ended. "
            "The Bayesian model properly accounts for censoring \u2014 ignoring it would bias "
            "the survival estimates downward.</dd>"
            "<dt>Why Bayesian over Kaplan-Meier?</dt>"
            "<dd>Kaplan-Meier is nonparametric and makes no distributional assumptions, but "
            "cannot extrapolate beyond the data. The Bayesian Weibull model gives probabilistic "
            "predictions of future failures and honest uncertainty on life percentiles (B10, B50).</dd>"
            "</dl>"
        ),
    }

    return result


def run_bayes_demo(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

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
    summary += (
        f"<<COLOR:highlight>>{int(ci_level * 100)}% Credible interval:<</COLOR>> [{post_ci[0]:.4f}, {post_ci[1]:.4f}]\n"
    )
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
            "<dt>What does P(R \u2265 target) mean?</dt>"
            "<dd>The posterior probability that the true reliability meets your target. "
            "<strong>&gt; 90%</strong>: strong confidence the target is met. "
            "<strong>50\u201390%</strong>: encouraging but more testing may be warranted. "
            "<strong>&lt; 50%</strong>: the product likely does not meet the target.</dd>"
            "<dt>How does sample size affect the result?</dt>"
            "<dd>The sequential plot shows how the posterior reliability estimate evolves as "
            "more units are tested. With zero failures, each additional test unit tightens the "
            "credible interval and increases confidence. Even one failure can significantly "
            "shift the posterior.</dd>"
            "<dt>When to use this?</dt>"
            "<dd>Qualification testing, lot acceptance, warranty requirement verification \u2014 "
            "whenever you need to demonstrate that a product meets a reliability target "
            "with quantified confidence.</dd>"
            "</dl>"
        ),
    }

    return result


def run_bayes_repairable(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

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
        float(beta_range[np.searchsorted(np.cumsum(beta_marg), (1 - ci_level) / 2)]),
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
    trend_label = "deteriorating" if p_deteriorating > 0.8 else ("improving" if p_deteriorating < 0.2 else "stable")
    verdict = "FAIL" if p_deteriorating > 0.8 else ("PASS" if p_deteriorating < 0.2 else "WARNING")

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
    forecast_bars, forecast_lo_bars, forecast_hi_bars, period_labels = [], [], [], []
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
                        "array": [h - m for h, m in zip(forecast_hi_bars, forecast_bars)],
                        "arrayminus": [m - lo for m, lo in zip(forecast_bars, forecast_lo_bars)],
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
            "<dt>What does the \u03b2 parameter mean?</dt>"
            "<dd><strong>\u03b2 &lt; 1</strong>: reliability growth \u2014 failures are becoming less frequent "
            "(improvement). <strong>\u03b2 = 1</strong>: constant failure rate (no trend). "
            "<strong>\u03b2 &gt; 1</strong>: reliability deterioration \u2014 failures are accelerating (wear-out).</dd>"
            "<dt>What is the failure intensity function?</dt>"
            "<dd>The instantaneous rate of failure at any given time. The NHPP allows this rate "
            "to change \u2014 unlike a homogeneous Poisson process which assumes constant rate. "
            "The Bayesian posterior gives a credible band on this intensity.</dd>"
            "<dt>When to use this?</dt>"
            "<dd>Fleet vehicles, industrial equipment, IT systems \u2014 any asset that is repeatedly "
            "repaired. Track whether reliability is improving (after maintenance programme changes) "
            "or degrading (approaching end of life), and forecast future failure counts.</dd>"
            "</dl>"
        ),
    }

    return result


def run_bayes_rul(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

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
    slope_var = float(np.var(slopes, ddof=1)) if n_units > 1 else float(slope_mean**2 * 0.1)
    slope_std = float(np.sqrt(slope_var))

    all_times = df_clean[time_col].values.astype(float)
    all_meas = df_clean[meas_col].values.astype(float)
    t_current = float(np.max(all_times))
    last_mask = all_times == all_times.max()
    y_current = float(np.mean(all_meas[last_mask])) if np.sum(last_mask) > 0 else float(all_meas[-1])

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
        slope_samples = rng.normal(slope_mean, slope_std if slope_std > 0 else abs(slope_mean) * 0.1, size=n_mc)

    if direction == "decreasing":
        rul_samples = np.where(slope_samples < 0, (threshold - y_current) / slope_samples, np.inf)
    else:
        rul_samples = np.where(slope_samples > 0, (threshold - y_current) / slope_samples, np.inf)

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

    horizon = float(config.get("horizon", rul_mean * 1.5 if np.isfinite(rul_mean) else t_current))
    p_fail_horizon = float(np.mean(valid_rul <= horizon)) if len(valid_rul) > 0 else 0.0
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
    result["guide_observation"] = f"Bayes RUL: mean={rul_mean:.1f}, P(fail before {horizon:.0f})={p_fail_horizon:.3f}"
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
        rate_pdf = stats.t.pdf(rate_x, n_units - 1, loc=slope_mean, scale=slope_std / np.sqrt(n_units))
    else:
        rate_pdf = stats.norm.pdf(rate_x, slope_mean, slope_std if slope_std > 0 else abs(slope_mean) * 0.1)
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
            "the prediction is confident; a wide one means significant uncertainty remains \u2014 "
            "more monitoring data would help narrow it.</dd>"
            "<dt>When to use this?</dt>"
            "<dd>Condition-based maintenance scheduling, prognostics and health management (PHM), "
            "deciding when to replace bearings, batteries, filters, or any degrading component "
            "before it fails in service.</dd>"
            "</dl>"
        ),
    }

    return result


def run_bayes_alt(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

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
            stress_x.append(1.0 / s if (model_type == "arrhenius" and s > 0) else np.log(s + 1e-15))
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
        sigma2_samples = (n_pts - 2) * s2 / rng.chisquare(max(1, n_pts - 2), size=n_mc)
    else:
        sigma2_samples = np.full(n_mc, s2)

    a_samples = rng.normal(a, np.sqrt(sigma2_samples * (1 / n_pts + x_mean_s**2 / max(Sxx, 1e-15))))
    b_samples = rng.normal(b, np.sqrt(sigma2_samples / max(Sxx, 1e-15)))

    life_use_samples = np.exp(a_samples + b_samples * x_use)
    life_use_samples = life_use_samples[np.isfinite(life_use_samples) & (life_use_samples > 0)]

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

    af = float(life_use / float(np.exp(a + b * stress_x[-1]))) if len(stress_x) > 0 else 1.0
    verdict = "PASS"

    summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
    summary += "<<COLOR:title>>BAYESIAN ACCELERATED LIFE TESTING<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:highlight>>Model:<</COLOR>> {model_type}\n"
    summary += f"<<COLOR:highlight>>Stress levels:<</COLOR>> {len(unique_stresses)}\n"
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
    x_plot = np.linspace(min(min(stress_x), x_use) * 0.9, max(max(stress_x), x_use) * 1.1, 100)
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
    surv_lo = np.exp(-((t_surv / life_ci[0]) ** shape_overall)) if life_ci[0] > 0 else np.ones_like(t_surv)
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
            "energy \u2014 standard for thermal degradation. <strong>Eyring</strong>: extends to "
            "multiple stresses (temperature + humidity). The Bayesian posterior captures "
            "uncertainty in the acceleration parameters.</dd>"
            "<dt>What is B10 life at use conditions?</dt>"
            "<dd>The time by which 10% of units are expected to fail under normal operating "
            "conditions. The Bayesian posterior on B10 gives a credible interval \u2014 critical "
            "for warranty and design life decisions.</dd>"
            "<dt>Why Bayesian ALT?</dt>"
            "<dd>ALT typically has few failures at each stress level. Bayesian estimation "
            "handles small samples naturally, propagates parameter uncertainty into life "
            "predictions, and avoids the overconfidence of point extrapolations.</dd>"
            "</dl>"
        ),
    }

    return result


def run_bayes_comprisk(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

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
        result["summary"] = "Error: Need at least 2 failure modes (event: 0=censored, 1,2,...=modes)."
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
            ll += np.log(bg + 1e-15) + (bg - 1) * np.log(t_i + 1e-15) - bg * np.log(eg + 1e-15)
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
            h_j = (w["shape"] / w["scale"]) * ((t_eval + 1e-15) / w["scale"]) ** (w["shape"] - 1)
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
        "mode_probabilities": {str(m): float(mode_probs[j]) for j, m in enumerate(modes)},
        "mode_weibull": {str(m): mode_weibull[m] for m in modes},
    }
    result["guide_observation"] = f"Bayes competing risks: {n_modes} modes, dominant=Mode {dominant_mode}"
    result["narrative"] = _narrative(
        verdict,
        f"{n_modes} competing failure modes identified. "
        f"Dominant: Mode {dominant_mode} (P = {mode_probs[modes.index(dominant_mode)]:.3f}). "
        + "; ".join(
            [f"Mode {m}: \u03b2={mode_weibull[m]['shape']:.2f}, \u03b7={mode_weibull[m]['scale']:.1f}" for m in modes]
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
                        "arrayminus": [p - ci[0] for p, ci in zip(mode_probs, mode_ci)],
                    },
                    "marker": {"color": [SVEND_COLORS[j % len(SVEND_COLORS)] for j in range(n_modes)]},
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
            "sum to the overall failure probability \u2014 giving a complete decomposition.</dd>"
            "<dt>When to use this?</dt>"
            "<dd>Warranty root cause decomposition, maintenance strategy design (which failure "
            "mode to address first), product redesign prioritisation, or any reliability "
            "analysis where multiple distinct failure mechanisms are at play.</dd>"
            "</dl>"
        ),
    }

    return result

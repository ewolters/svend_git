"""Bayesian analysis operations extracted from the monolithic bayesian.py.

Each function takes (df, config, ci_level, z) and returns a result dict
with keys: plots, summary, guide_observation, and optional statistics,
narrative, what_if_data, education.
"""

import json as _json

import numpy as np
from scipy import stats

from ..common import (
    COLOR_BAD,
    COLOR_GOOD,
    COLOR_REFERENCE,
    SVEND_COLORS,
    _narrative,
    _rgba,
)


def run_bayes_spares(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

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
            "layout": {"height": 300, "xaxis": {"title": "Demand Rate (\u03bb)"}, "yaxis": {"title": "Density"}},
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

    return result


def run_bayes_system(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    # Bayesian System Reliability (MC propagation through topology)
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
    comp_names, comp_means, comp_cis, comp_draws, comp_post_params = [], [], [], [], []

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
        comp_cis.append((float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))))
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

    importance = [float(np.corrcoef(draws_matrix[:, j], sys_draws)[0, 1]) for j in range(n_comp)]
    weakest = comp_names[int(np.argmin(comp_means))]

    topology_label = topology.replace("_", "-")
    if topology == "k_of_n":
        topology_label = f"{k_val}-of-{n_comp}"

    verdict = "PASS" if sys_mean > 0.9 else ("WARNING" if sys_mean > 0.8 else "FAIL")

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
        "components": {n: {"mean": m, "ci": list(c)} for n, m, c in zip(comp_names, comp_means, comp_cis)},
    }
    result["guide_observation"] = f"Bayes system ({topology_label}): R_sys={sys_mean:.4f}, weakest={weakest}"
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
            "layout": {"height": 350, "xaxis": {"title": "Reliability"}, "yaxis": {"title": "Density"}},
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
                    "marker": {"color": [SVEND_COLORS[j % len(SVEND_COLORS)] for j in range(n_comp)]},
                }
            ],
            "layout": {"height": 300, "xaxis": {"title": "Component"}, "yaxis": {"title": "Birnbaum Importance"}},
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

    return result


def run_bayes_warranty(df, config, ci_level, z):
    result = {"plots": [], "summary": "", "guide_observation": ""}

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
            "layout": {"height": 300, "xaxis": {"title": "Failure Rate (\u03bb)"}, "yaxis": {"title": "Density"}},
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
            "layout": {"height": 300, "xaxis": {"title": "Period"}, "yaxis": {"title": "Cumulative Claims"}},
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
            "layout": {"height": 300, "xaxis": {"title": "Period"}, "yaxis": {"title": "Claims"}},
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

    return result

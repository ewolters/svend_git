"""
Bayesian DOE analysis suite — 5 tools for designed experiments.

Tools:
    bayes_doe_effects   — Effect screening (P(practical significance))
    bayes_doe_model     — Model selection (marginal likelihood)
    bayes_doe_samplesize — Pre-posterior sample size analysis
    bayes_doe_optimize  — Response optimization with uncertainty
    bayes_doe_next      — Sequential experiment suggestion

All tools share the conjugate Normal-Inverse-Gamma linear model
from bayes_core.py.  No MCMC, O(p³) — industrial-grade fast.
"""

from itertools import combinations

import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import t as t_dist

from .bayes_core import (
    bayesian_linear_posterior,
    contrast_posterior,
    marginal_log_likelihood,
    predictive_posterior,
)

# ---------------------------------------------------------------------------
# Design Matrix Builder
# ---------------------------------------------------------------------------


def build_doe_design_matrix(
    df, factor_cols, response_col, include_2fi=True, include_quad=False
):
    """
    Build a coded design matrix from a DataFrame of DOE data.

    Coding rules (deterministic, reversible, stored):
        - 2 unique values  → binary: sorted low=-1, high=+1
        - Numeric >2 levels → center=(max+min)/2, scale=(max-min)/2, code to [-1,+1]
        - Categorical >2    → deviation coding (sum-to-zero, not dummy)

    Parameters
    ----------
    df : DataFrame
        Experimental data.
    factor_cols : list of str
        Column names for factors.
    response_col : str
        Column name for response.
    include_2fi : bool
        Include two-factor interaction columns.
    include_quad : bool
        Include quadratic (squared) columns for numeric factors.

    Returns
    -------
    X : ndarray (n, p)
        Design matrix with intercept as column 0.
    y : ndarray (n,)
        Response vector.
    term_names : list of str
        Human-readable name for each column of X.
    coding_records : list of dict
        One per factor — metadata for decoding back to natural units.
    """
    n = len(df)
    y = df[response_col].values.astype(np.float64)

    coded_columns = []  # list of ndarray (n,), one per main effect
    term_names = ["Intercept"]
    coding_records = []

    for col in factor_cols:
        vals = df[col]
        unique = sorted(vals.dropna().unique())
        n_unique = len(unique)

        if n_unique < 2:
            # Constant factor — skip with warning record
            coding_records.append(
                {
                    "column": col,
                    "coding_type": "constant",
                    "center": float(unique[0]) if n_unique == 1 else 0.0,
                    "scale": 0.0,
                    "level_map": None,
                }
            )
            coded_columns.append(np.zeros(n))
            term_names.append(col)
            continue

        # Try numeric first
        try:
            numeric_vals = vals.astype(np.float64)
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False

        if is_numeric and n_unique == 2:
            # Binary numeric: -1/+1
            low, high = float(unique[0]), float(unique[1])
            center = (low + high) / 2.0
            scale = (high - low) / 2.0
            coded = (numeric_vals.values - center) / scale
            coding_records.append(
                {
                    "column": col,
                    "coding_type": "binary",
                    "center": center,
                    "scale": scale,
                    "level_map": {str(low): -1, str(high): 1},
                }
            )
            coded_columns.append(coded)
            term_names.append(col)

        elif is_numeric and n_unique > 2:
            # Numeric scaled: center/scale to [-1, +1]
            low, high = float(unique[0]), float(unique[-1])
            center = (low + high) / 2.0
            scale = (high - low) / 2.0
            if scale < 1e-15:
                scale = 1.0
            coded = (numeric_vals.values - center) / scale
            coding_records.append(
                {
                    "column": col,
                    "coding_type": "numeric_scaled",
                    "center": center,
                    "scale": scale,
                    "level_map": None,
                }
            )
            coded_columns.append(coded)
            term_names.append(col)

        elif not is_numeric and n_unique == 2:
            # Binary categorical: sorted alphabetically, low=-1, high=+1
            level_map = {str(unique[0]): -1, str(unique[1]): 1}
            coded = np.array([level_map[str(v)] for v in vals], dtype=np.float64)
            coding_records.append(
                {
                    "column": col,
                    "coding_type": "binary",
                    "center": 0.0,
                    "scale": 1.0,
                    "level_map": level_map,
                }
            )
            coded_columns.append(coded)
            term_names.append(col)

        else:
            # Categorical >2 levels: deviation coding (sum-to-zero)
            # k levels → k-1 columns
            levels = [str(u) for u in unique]
            k = len(levels)
            level_map = {}
            cat_coded = np.zeros((n, k - 1))
            for i, lev in enumerate(levels[:-1]):
                level_map[lev] = i  # column index
            level_map[levels[-1]] = -1  # reference level (coded as all -1)

            for row_idx in range(n):
                v = str(vals.iloc[row_idx])
                if v == levels[-1]:
                    cat_coded[row_idx, :] = -1.0 / k  # deviation coding
                elif v in level_map and level_map[v] >= 0:
                    col_idx = level_map[v]
                    cat_coded[row_idx, :] = -1.0 / k
                    cat_coded[row_idx, col_idx] = 1.0 - 1.0 / k

            for i in range(k - 1):
                coded_columns.append(cat_coded[:, i])
                term_names.append(f"{col}[{levels[i]}]")

            coding_records.append(
                {
                    "column": col,
                    "coding_type": "categorical_effect",
                    "center": 0.0,
                    "scale": 1.0,
                    "level_map": {lev: idx for lev, idx in level_map.items()},
                }
            )

    # Build X: intercept + main effects
    n_main = len(coded_columns)
    X_parts = [np.ones((n, 1))]  # intercept
    X_parts.extend([c.reshape(-1, 1) for c in coded_columns])

    # Two-factor interactions
    if include_2fi and n_main >= 2:
        main_indices = list(range(n_main))
        for i, j in combinations(main_indices, 2):
            interaction = coded_columns[i] * coded_columns[j]
            X_parts.append(interaction.reshape(-1, 1))
            # Names: skip bracket terms for readability
            name_i = term_names[1 + i]  # +1 for intercept
            name_j = term_names[1 + j]
            term_names.append(f"{name_i}:{name_j}")

    # Quadratic terms (only for numeric factors)
    if include_quad:
        for i in range(len(factor_cols)):
            rec = coding_records[i]
            if rec["coding_type"] in ("binary", "numeric_scaled"):
                quad = coded_columns[i] ** 2
                X_parts.append(quad.reshape(-1, 1))
                term_names.append(f"{factor_cols[i]}²")

    X = np.hstack(X_parts)
    return X, y, term_names, coding_records


def _decode_coded_to_natural(coded_vals, coding_records, factor_cols):
    """
    Convert coded factor values back to natural units.

    Parameters
    ----------
    coded_vals : dict or ndarray
        {factor_name: coded_value} or array matching factor_cols order.
    coding_records : list of dict
    factor_cols : list of str

    Returns
    -------
    dict : {factor_name: natural_value}
    """
    if isinstance(coded_vals, np.ndarray):
        coded_vals = {col: coded_vals[i] for i, col in enumerate(factor_cols)}

    result = {}
    for rec in coding_records:
        col = rec["column"]
        if col not in coded_vals:
            continue
        coded = coded_vals[col]
        if rec["coding_type"] in ("binary", "numeric_scaled"):
            result[col] = coded * rec["scale"] + rec["center"]
        elif rec["coding_type"] == "categorical_effect":
            # Find closest level
            if rec["level_map"]:
                inv = {v: k for k, v in rec["level_map"].items()}
                result[col] = inv.get(round(coded), str(coded))
        else:
            result[col] = coded
    return result


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------


def run_bayesian_doe(df, analysis_id, config):
    """
    Dispatch to the appropriate Bayesian DOE tool.

    Called from dsw_views.run_spc_analysis() via:
        elif analysis_id.startswith("bayes_doe_"):
            from .bayes_doe import run_bayesian_doe
            return run_bayesian_doe(df, analysis_id, config)

    Parameters
    ----------
    df : DataFrame
        Experimental data (all factor + response columns).
    analysis_id : str
        One of: bayes_doe_effects, bayes_doe_model, bayes_doe_samplesize,
                bayes_doe_optimize, bayes_doe_next
    config : dict
        Analysis configuration from request body.

    Returns
    -------
    dict with keys: summary, plots, guide_observation, statistics
    """
    handlers = {
        "bayes_doe_effects": _run_doe_effects,
        "bayes_doe_model": _run_doe_model,
        "bayes_doe_samplesize": _run_doe_samplesize,
        "bayes_doe_optimize": _run_doe_optimize,
        "bayes_doe_next": _run_doe_next,
    }

    handler = handlers.get(analysis_id)
    if handler is None:
        return {
            "summary": f"Unknown Bayesian DOE analysis: {analysis_id}",
            "plots": [],
            "guide_observation": None,
            "statistics": {},
        }

    try:
        return handler(df, config)
    except Exception as e:
        import traceback

        return {
            "summary": f"<<COLOR:error>>Error in {analysis_id}: {str(e)}<</COLOR>>\n\n"
            f"```\n{traceback.format_exc()}\n```",
            "plots": [],
            "guide_observation": str(e),
            "statistics": {"error": str(e)},
        }


# ---------------------------------------------------------------------------
# Tool 1: Effect Screening
# ---------------------------------------------------------------------------


def _run_doe_effects(df, config):
    """
    Bayesian effect screening — P(practical significance) for each term.

    For each effect j: P(|βj| > threshold) via Student-t posterior.
    Verdict: P > 0.90 → ACTIVE, 0.50-0.90 → POSSIBLY ACTIVE, < 0.50 → INERT.
    """
    factor_cols = config.get("factor_columns", [])
    response_col = config.get("response_column", "")
    threshold = config.get("threshold")
    include_2fi = config.get("include_2fi", True)

    if not factor_cols or not response_col:
        return {
            "summary": "<<COLOR:error>>Please specify factor columns and a response column.<</COLOR>>",
            "plots": [],
            "guide_observation": None,
            "statistics": {},
        }

    # Build design matrix
    X, y, term_names, coding_records = build_doe_design_matrix(
        df, factor_cols, response_col, include_2fi=include_2fi
    )
    n, p = X.shape

    if n < p:
        return {
            "summary": f"<<COLOR:error>>Not enough data: {n} runs for {p} parameters. "
            f"Need at least {p} runs.<</COLOR>>",
            "plots": [],
            "guide_observation": None,
            "statistics": {},
        }

    # Auto threshold: 10% of response range
    if threshold is None or threshold <= 0:
        y_range = float(np.ptp(y))
        threshold = 0.1 * y_range if y_range > 0 else 0.1

    # Fit posterior
    mu_n, Lambda_n, L_n, alpha_n, beta_n = bayesian_linear_posterior(X, y)

    # Compute effect posteriors (skip intercept at index 0)
    effects = []
    for j in range(1, p):
        e_j = np.zeros(p)
        e_j[j] = 1.0
        loc, scale, df_t = contrast_posterior(e_j, mu_n, Lambda_n, L_n, alpha_n, beta_n)

        # P(|βj| > threshold)
        p_above = (
            1.0
            - t_dist.cdf(threshold, df_t, loc=loc, scale=scale)
            + t_dist.cdf(-threshold, df_t, loc=loc, scale=scale)
        )

        # 95% credible interval
        ci_half = t_dist.ppf(0.975, df_t) * scale
        ci_low = loc - ci_half
        ci_high = loc + ci_half

        # Natural-unit effect size (for ±1 coded factors: effect = 2 * coefficient)
        # Find which coding record this term belongs to
        natural_effect = loc  # default
        natural_ci_low = ci_low
        natural_ci_high = ci_high
        term_name = term_names[j]

        # For main effects of coded factors, decode
        for rec in coding_records:
            if rec["column"] == term_name and rec["coding_type"] in (
                "binary",
                "numeric_scaled",
            ):
                # Full swing from -1 to +1 is 2*coefficient in coded units
                # In natural units: 2*loc*scale_factor (going from low to high)
                natural_effect = 2.0 * loc * rec["scale"]
                natural_ci_low = 2.0 * ci_low * rec["scale"]
                natural_ci_high = 2.0 * ci_high * rec["scale"]
                break

        # Verdict
        if p_above >= 0.90:
            verdict = "ACTIVE"
            color = "green"
        elif p_above >= 0.50:
            verdict = "POSSIBLY ACTIVE"
            color = "amber"
        else:
            verdict = "INERT"
            color = "red"

        effects.append(
            {
                "term": term_name,
                "coded_coeff": round(loc, 4),
                "ci_low": round(ci_low, 4),
                "ci_high": round(ci_high, 4),
                "natural_effect": round(natural_effect, 4),
                "natural_ci_low": round(natural_ci_low, 4),
                "natural_ci_high": round(natural_ci_high, 4),
                "p_practical": round(p_above, 4),
                "verdict": verdict,
                "color": color,
                "scale_t": round(scale, 4),
                "df_t": round(df_t, 2),
            }
        )

    # Sort by P(practical significance) descending for Pareto
    effects_sorted = sorted(effects, key=lambda e: e["p_practical"], reverse=True)

    # --- Build summary ---
    summary_lines = [
        "## Bayesian Effect Screening\n",
        f"**Design:** {n} runs, {len(factor_cols)} factors, "
        f"{p - 1} model terms (incl. {'interactions' if include_2fi else 'main effects only'})",
        f"**Practical threshold:** |effect| > {threshold:.4g} (coded units)\n",
        "| Term | Coded β | 95% CI | P(Active) | Verdict |",
        "|------|---------|--------|-----------|---------|",
    ]
    for e in effects_sorted:
        v_tag = {
            "ACTIVE": "<<COLOR:green>>ACTIVE<</COLOR>>",
            "POSSIBLY ACTIVE": "<<COLOR:amber>>POSSIBLY ACTIVE<</COLOR>>",
            "INERT": "<<COLOR:red>>INERT<</COLOR>>",
        }[e["verdict"]]
        summary_lines.append(
            f"| {e['term']} | {e['coded_coeff']:.4f} | "
            f"[{e['ci_low']:.4f}, {e['ci_high']:.4f}] | "
            f"{e['p_practical']:.1%} | {v_tag} |"
        )

    # Active effects in natural units
    active = [e for e in effects_sorted if e["verdict"] == "ACTIVE"]
    if active:
        summary_lines.append("\n### Active Effects (Natural Units)")
        for e in active:
            for rec in coding_records:
                if rec["column"] == e["term"]:
                    summary_lines.append(
                        f"- **{e['term']}**: changing from low to high shifts response by "
                        f"{e['natural_effect']:+.4g} "
                        f"[{e['natural_ci_low']:+.4g}, {e['natural_ci_high']:+.4g}]"
                    )
                    break
            else:
                # Interaction or other
                summary_lines.append(
                    f"- **{e['term']}**: coded coefficient = {e['coded_coeff']:.4f}"
                )

    summary_lines.append(
        "\n*Bayesian screening eliminates p-value thresholds. "
        "P(Active) directly answers: what's the probability this effect is practically meaningful?*"
    )

    # --- Plots ---
    plots = []

    # Plot 1: Horizontal bars — P(practical significance) per term
    bar_terms = [e["term"] for e in effects_sorted]
    bar_probs = [e["p_practical"] for e in effects_sorted]
    bar_colors = []
    for e in effects_sorted:
        if e["p_practical"] >= 0.90:
            bar_colors.append("rgba(46, 204, 113, 0.8)")
        elif e["p_practical"] >= 0.50:
            bar_colors.append("rgba(241, 196, 15, 0.8)")
        else:
            bar_colors.append("rgba(231, 76, 60, 0.8)")

    plots.append(
        {
            "data": [
                {
                    "type": "bar",
                    "y": bar_terms,
                    "x": bar_probs,
                    "orientation": "h",
                    "marker": {"color": bar_colors},
                    "text": [f"{p:.1%}" for p in bar_probs],
                    "textposition": "auto",
                }
            ],
            "layout": {
                "title": "P(Practical Significance) by Term",
                "xaxis": {"title": "Probability", "range": [0, 1]},
                "yaxis": {"autorange": "reversed"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": 0.90,
                        "x1": 0.90,
                        "y0": -0.5,
                        "y1": len(bar_terms) - 0.5,
                        "line": {"color": "green", "width": 2, "dash": "dash"},
                    },
                    {
                        "type": "line",
                        "x0": 0.50,
                        "x1": 0.50,
                        "y0": -0.5,
                        "y1": len(bar_terms) - 0.5,
                        "line": {"color": "orange", "width": 1, "dash": "dot"},
                    },
                ],
                "margin": {"l": 120},
            },
        }
    )

    # Plot 2: Posterior ridge — top 6 effects with threshold lines
    top_effects = effects_sorted[:6]
    ridge_traces = []
    for i, e in enumerate(top_effects):
        df_t = e["df_t"]
        loc_val = e["coded_coeff"]
        scale_val = e["scale_t"]
        x_range = np.linspace(loc_val - 4 * scale_val, loc_val + 4 * scale_val, 200)
        y_pdf = t_dist.pdf(x_range, df_t, loc=loc_val, scale=scale_val)
        ridge_traces.append(
            {
                "type": "scatter",
                "x": x_range.tolist(),
                "y": (y_pdf + i * 0.1).tolist(),  # offset for ridge
                "mode": "lines",
                "name": e["term"],
                "fill": "tonexty" if i > 0 else None,
            }
        )

    plots.append(
        {
            "data": ridge_traces,
            "layout": {
                "title": "Posterior Distributions (Top Effects)",
                "xaxis": {"title": "Coded Coefficient"},
                "yaxis": {"title": "Density (offset)", "showticklabels": False},
                "shapes": [
                    {
                        "type": "line",
                        "x0": threshold,
                        "x1": threshold,
                        "y0": 0,
                        "y1": 1,
                        "yref": "paper",
                        "line": {"color": "red", "width": 1, "dash": "dash"},
                    },
                    {
                        "type": "line",
                        "x0": -threshold,
                        "x1": -threshold,
                        "y0": 0,
                        "y1": 1,
                        "yref": "paper",
                        "line": {"color": "red", "width": 1, "dash": "dash"},
                    },
                ],
                "showlegend": True,
            },
        }
    )

    # Plot 3: Effect size with CI error bars, Pareto-ordered
    pareto_terms = [e["term"] for e in effects_sorted]
    pareto_coeff = [e["coded_coeff"] for e in effects_sorted]
    pareto_ci_low = [e["ci_low"] for e in effects_sorted]
    pareto_ci_high = [e["ci_high"] for e in effects_sorted]

    plots.append(
        {
            "data": [
                {
                    "type": "scatter",
                    "x": pareto_terms,
                    "y": pareto_coeff,
                    "mode": "markers",
                    "marker": {"size": 10, "color": bar_colors},
                    "error_y": {
                        "type": "data",
                        "symmetric": False,
                        "array": [h - c for h, c in zip(pareto_ci_high, pareto_coeff)],
                        "arrayminus": [
                            c - lo for c, lo in zip(pareto_coeff, pareto_ci_low)
                        ],
                    },
                    "name": "Effect ± 95% CI",
                }
            ],
            "layout": {
                "title": "Effect Sizes (Pareto Order)",
                "xaxis": {"title": "Term"},
                "yaxis": {"title": "Coded Coefficient"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": -0.5,
                        "x1": len(pareto_terms) - 0.5,
                        "y0": 0,
                        "y1": 0,
                        "line": {"color": "gray", "width": 1},
                    }
                ],
            },
        }
    )

    # Guide observation
    n_active = len([e for e in effects if e["verdict"] == "ACTIVE"])
    n_possible = len([e for e in effects if e["verdict"] == "POSSIBLY ACTIVE"])
    guide_obs = (
        f"Bayesian DOE effect screening: {n_active} active, {n_possible} possibly active "
        f"out of {len(effects)} terms. "
        + (
            f"Top effect: {effects_sorted[0]['term']} (P={effects_sorted[0]['p_practical']:.1%})."
            if effects_sorted
            else ""
        )
    )

    return {
        "summary": "\n".join(summary_lines),
        "plots": plots,
        "guide_observation": guide_obs,
        "statistics": {
            "n_runs": n,
            "n_terms": p - 1,
            "threshold": threshold,
            "effects": effects_sorted,
            "posterior_alpha": round(alpha_n, 4),
            "posterior_beta": round(beta_n, 4),
        },
    }


# ---------------------------------------------------------------------------
# Tool 2: Model Selection
# ---------------------------------------------------------------------------


def _run_doe_model(df, config):
    """
    Bayesian model selection — compare model families via marginal likelihood.

    M1: intercept + main effects
    M2: intercept + main + 2FI
    M3: intercept + main + 2FI + quadratic (if n > p)
    """
    factor_cols = config.get("factor_columns", [])
    response_col = config.get("response_column", "")

    if not factor_cols or not response_col:
        return {
            "summary": "<<COLOR:error>>Please specify factor columns and a response column.<</COLOR>>",
            "plots": [],
            "guide_observation": None,
            "statistics": {},
        }

    n_obs = len(df)

    # Build all three model families
    models = []

    # M1: Main effects only
    X1, y, terms1, coding = build_doe_design_matrix(
        df, factor_cols, response_col, include_2fi=False, include_quad=False
    )
    if n_obs >= X1.shape[1]:
        models.append(("Main Effects", X1, terms1))

    # M2: Main + 2FI
    X2, _, terms2, _ = build_doe_design_matrix(
        df, factor_cols, response_col, include_2fi=True, include_quad=False
    )
    if n_obs >= X2.shape[1]:
        models.append(("Main + Interactions", X2, terms2))

    # M3: Main + 2FI + Quadratic
    X3, _, terms3, _ = build_doe_design_matrix(
        df, factor_cols, response_col, include_2fi=True, include_quad=True
    )
    if n_obs >= X3.shape[1] and X3.shape[1] > X2.shape[1]:
        models.append(("Full Quadratic", X3, terms3))

    if not models:
        return {
            "summary": f"<<COLOR:error>>Not enough data ({n_obs} runs) for any model.<</COLOR>>",
            "plots": [],
            "guide_observation": None,
            "statistics": {},
        }

    # Compute posterior and marginal log-likelihood for each model
    model_results = []
    for name, X_k, terms_k in models:
        n_k, p_k = X_k.shape
        mu_n, Lambda_n, L_n, alpha_n, beta_n = bayesian_linear_posterior(X_k, y)

        # Need L_0 for marginal likelihood — reconstruct prior precision
        XtX = X_k.T @ X_k
        trace_val = np.trace(XtX)
        lam = 1e-8 * trace_val / p_k if p_k > 0 and trace_val > 0 else 1e-8
        XtX_reg = XtX + lam * np.eye(p_k)
        g = max(n_k, p_k)
        Lambda0 = XtX_reg / g
        L_0 = np.linalg.cholesky(Lambda0)

        alpha0 = 2.0
        s2 = np.var(y, ddof=1) if n_k > 1 else 1.0
        beta0 = max(alpha0 * s2, 1e-10)

        log_ml = marginal_log_likelihood(
            Lambda0, Lambda_n, L_0, L_n, alpha0, alpha_n, beta0, beta_n, n_k
        )

        # Fitted values
        y_hat = X_k @ mu_n

        model_results.append(
            {
                "name": name,
                "n_params": p_k,
                "terms": terms_k,
                "log_ml": log_ml,
                "mu_n": mu_n,
                "y_hat": y_hat,
                "alpha_n": alpha_n,
                "beta_n": beta_n,
                "residual_sigma": float(np.sqrt(beta_n / alpha_n)),
            }
        )

    # Posterior model probabilities via log-softmax
    log_mls = np.array([m["log_ml"] for m in model_results])
    log_mls_shifted = log_mls - np.max(log_mls)  # numerical stability
    probs = np.exp(log_mls_shifted)
    probs = probs / np.sum(probs)

    for i, m in enumerate(model_results):
        m["probability"] = float(probs[i])

    # Per-factor importance: sum P(Mk) over models containing that factor
    factor_importance = {}
    for col in factor_cols:
        imp = 0.0
        for m in model_results:
            # Check if any term contains this factor name
            for term in m["terms"]:
                if col in term and term != "Intercept":
                    imp += m["probability"]
                    break
            # Main effects are always included in all models, so importance = 1.0
            # for main effects. Interactions/quadratic contribute differently.
        factor_importance[col] = round(imp, 4)

    # Model-averaged predictions
    y_avg = np.zeros(n_obs)
    for m in model_results:
        y_avg += m["probability"] * m["y_hat"]

    # Best model
    best = max(model_results, key=lambda m: m["probability"])

    # --- Summary ---
    summary_lines = [
        "## Bayesian Model Selection\n",
        f"**Data:** {n_obs} runs, {len(factor_cols)} factors\n",
        "| Model | Parameters | P(Model|Data) | σ estimate |",
        "|-------|-----------|---------------|------------|",
    ]
    for m in sorted(model_results, key=lambda x: x["probability"], reverse=True):
        prob_str = f"{m['probability']:.1%}"
        if m["name"] == best["name"]:
            prob_str = f"<<COLOR:green>>{prob_str}<</COLOR>>"
        summary_lines.append(
            f"| {m['name']} | {m['n_params']} | {prob_str} | {m['residual_sigma']:.4f} |"
        )

    summary_lines.append(
        f"\n**Best model:** {best['name']} (P = {best['probability']:.1%})"
    )

    summary_lines.append("\n### Factor Importance")
    summary_lines.append(
        "*Probability that at least one model containing this factor is correct:*\n"
    )
    for col in factor_cols:
        imp = factor_importance[col]
        if imp >= 0.90:
            tag = "<<COLOR:green>>HIGH<</COLOR>>"
        elif imp >= 0.50:
            tag = "<<COLOR:amber>>MODERATE<</COLOR>>"
        else:
            tag = "<<COLOR:red>>LOW<</COLOR>>"
        summary_lines.append(f"- **{col}**: {imp:.1%} {tag}")

    summary_lines.append(
        "\n*Model selection uses the exact marginal likelihood — "
        "no information criteria approximations (AIC/BIC). "
        "Automatically penalizes complexity via the prior.*"
    )

    # --- Plots ---
    plots = []

    # Plot 1: Model probability bars
    plots.append(
        {
            "data": [
                {
                    "type": "bar",
                    "x": [m["name"] for m in model_results],
                    "y": [m["probability"] for m in model_results],
                    "marker": {
                        "color": [
                            (
                                "rgba(46, 204, 113, 0.8)"
                                if m["name"] == best["name"]
                                else "rgba(149, 165, 166, 0.6)"
                            )
                            for m in model_results
                        ]
                    },
                    "text": [f"{m['probability']:.1%}" for m in model_results],
                    "textposition": "auto",
                }
            ],
            "layout": {
                "title": "Posterior Model Probabilities",
                "yaxis": {"title": "P(Model | Data)", "range": [0, 1]},
            },
        }
    )

    # Plot 2: Factor importance bars
    fi_cols = list(factor_importance.keys())
    fi_vals = [factor_importance[c] for c in fi_cols]
    fi_colors = [
        (
            "rgba(46, 204, 113, 0.8)"
            if v >= 0.90
            else "rgba(241, 196, 15, 0.8)" if v >= 0.50 else "rgba(231, 76, 60, 0.8)"
        )
        for v in fi_vals
    ]
    plots.append(
        {
            "data": [
                {
                    "type": "bar",
                    "x": fi_cols,
                    "y": fi_vals,
                    "marker": {"color": fi_colors},
                    "text": [f"{v:.1%}" for v in fi_vals],
                    "textposition": "auto",
                }
            ],
            "layout": {
                "title": "Factor Importance (Marginal Inclusion Probability)",
                "yaxis": {"title": "Confidence Factor Matters", "range": [0, 1]},
            },
        }
    )

    # Plot 3: Observed vs model-averaged predicted
    plots.append(
        {
            "data": [
                {
                    "type": "scatter",
                    "x": y.tolist(),
                    "y": y_avg.tolist(),
                    "mode": "markers",
                    "marker": {"size": 8, "color": "rgba(52, 152, 219, 0.7)"},
                    "name": "Model-Averaged",
                },
                {
                    "type": "scatter",
                    "x": [float(np.min(y)), float(np.max(y))],
                    "y": [float(np.min(y)), float(np.max(y))],
                    "mode": "lines",
                    "line": {"color": "gray", "dash": "dash"},
                    "name": "Perfect fit",
                },
            ],
            "layout": {
                "title": "Observed vs Model-Averaged Predicted",
                "xaxis": {"title": "Observed"},
                "yaxis": {"title": "Predicted"},
            },
        }
    )

    guide_obs = (
        f"Bayesian model selection: {best['name']} preferred "
        f"(P={best['probability']:.1%}). "
        f"{len(model_results)} models compared."
    )

    return {
        "summary": "\n".join(summary_lines),
        "plots": plots,
        "guide_observation": guide_obs,
        "statistics": {
            "n_runs": n_obs,
            "n_factors": len(factor_cols),
            "models": [
                {
                    "name": m["name"],
                    "n_params": m["n_params"],
                    "probability": round(m["probability"], 4),
                    "log_ml": round(m["log_ml"], 4),
                    "residual_sigma": round(m["residual_sigma"], 4),
                }
                for m in model_results
            ],
            "best_model": best["name"],
            "factor_importance": factor_importance,
        },
    }


# ---------------------------------------------------------------------------
# Tool 3: Sample Size (Pre-Posterior)
# ---------------------------------------------------------------------------


def _run_doe_samplesize(df, config):
    """
    Pre-posterior sample size analysis — no data input required.

    Simulates datasets with known effects to find minimum n for
    P(detect) >= 0.90.
    """
    n_factors = int(config.get("n_factors", 3))
    expected_effect = float(config.get("expected_effect", 1.0))
    expected_sigma = float(config.get("expected_sigma", 1.0))
    n_sim = int(config.get("n_simulations", 500))

    if n_factors < 1 or n_factors > 10:
        return {
            "summary": "<<COLOR:error>>Number of factors must be between 1 and 10.<</COLOR>>",
            "plots": [],
            "guide_observation": None,
            "statistics": {},
        }

    rng = np.random.default_rng(42)

    # Candidate sample sizes
    base = 2**n_factors
    candidates = sorted(
        set(
            [
                base,
                max(base, 2 * (n_factors + 1)),  # minimum for main effects
                base * 2,
                base * 3,
                min(base * 4, 64),
                min(base * 6, 96),
                min(base * 8, 128),
            ]
        )
    )
    candidates = [c for c in candidates if c <= 128]
    if not candidates:
        candidates = [2 * (n_factors + 1), 4 * (n_factors + 1)]

    p = 1 + n_factors  # intercept + main effects
    threshold = 0.1 * expected_effect  # practical threshold at 10% of expected

    results = []
    for n_runs in candidates:
        # Construct design matrix once (balanced, ±1 coding)
        # Use fractional factorial if n_runs < 2^k
        if n_runs >= base:
            # Full factorial with replicates
            n_reps = n_runs // base
            remainder = n_runs - n_reps * base
            levels = np.array([-1.0, 1.0])
            grid = np.array(np.meshgrid(*[levels] * n_factors)).T.reshape(-1, n_factors)
            X_factors = np.tile(grid, (n_reps, 1))
            if remainder > 0:
                idx = rng.choice(base, remainder, replace=False)
                X_factors = np.vstack([X_factors, grid[idx]])
        else:
            # Random ±1 design
            X_factors = rng.choice([-1.0, 1.0], size=(n_runs, n_factors))

        X = np.column_stack([np.ones(len(X_factors)), X_factors])
        actual_n = len(X_factors)

        # Vectorized simulation
        # β_true: intercept=0, first factor has expected_effect, rest drawn from half-normal
        beta_true = np.zeros((n_sim, p))
        beta_true[:, 1] = expected_effect  # first factor always has the effect
        for j in range(2, p):
            # Other factors: 50% chance of having an effect
            has_effect = rng.random(n_sim) < 0.5
            beta_true[:, j] = has_effect * rng.normal(0, expected_effect * 0.5, n_sim)

        # σ_true drawn from half-normal centered on expected_sigma
        sigma_true = np.abs(rng.normal(expected_sigma, expected_sigma * 0.2, n_sim))
        sigma_true = np.maximum(sigma_true, 1e-6)

        # Generate Y = X @ β_true.T + ε  (n × n_sim)
        Y = X @ beta_true.T  # (n, n_sim)
        epsilon = rng.normal(0, 1, (actual_n, n_sim)) * sigma_true[np.newaxis, :]
        Y += epsilon

        # For each simulation: compute posterior → P(detect first factor)
        detect_probs = np.zeros(n_sim)
        ci_widths = np.zeros(n_sim)

        for s in range(n_sim):
            y_s = Y[:, s]
            mu_n, Lambda_n, L_n, alpha_n, beta_n = bayesian_linear_posterior(X, y_s)

            # Contrast for first factor
            e1 = np.zeros(p)
            e1[1] = 1.0
            loc, scale, df_t = contrast_posterior(
                e1, mu_n, Lambda_n, L_n, alpha_n, beta_n
            )

            # P(|β1| > threshold)
            p_detect = (
                1.0
                - t_dist.cdf(threshold, df_t, loc=loc, scale=scale)
                + t_dist.cdf(-threshold, df_t, loc=loc, scale=scale)
            )
            detect_probs[s] = p_detect

            # CI width
            ci_widths[s] = 2.0 * t_dist.ppf(0.975, df_t) * scale

        results.append(
            {
                "n": actual_n,
                "mean_detect": float(np.mean(detect_probs)),
                "p10_detect": float(np.percentile(detect_probs, 10)),
                "p90_detect": float(np.percentile(detect_probs, 90)),
                "mean_ci_width": float(np.mean(ci_widths)),
                "p10_ci": float(np.percentile(ci_widths, 10)),
                "p90_ci": float(np.percentile(ci_widths, 90)),
            }
        )

    # Find minimum n for mean P(detect) >= 0.90
    recommended_n = None
    for r in results:
        if r["mean_detect"] >= 0.90:
            recommended_n = r["n"]
            break

    # --- Summary ---
    summary_lines = [
        "## Bayesian DOE Sample Size Analysis\n",
        f"**Scenario:** {n_factors} factors, expected effect = {expected_effect}, "
        f"σ ≈ {expected_sigma}, signal-to-noise ≈ {expected_effect / expected_sigma:.2f}",
        f"**Simulations:** {n_sim} per candidate sample size\n",
        "| Runs | Mean P(Detect) | 10th-90th %ile | Mean CI Width |",
        "|------|---------------|----------------|---------------|",
    ]
    for r in results:
        marker = " ← recommended" if r["n"] == recommended_n else ""
        color = (
            "green"
            if r["mean_detect"] >= 0.90
            else "amber" if r["mean_detect"] >= 0.70 else "red"
        )
        detect_str = f"<<COLOR:{color}>>{r['mean_detect']:.1%}<</COLOR>>"
        summary_lines.append(
            f"| {r['n']} | {detect_str} | "
            f"[{r['p10_detect']:.1%}, {r['p90_detect']:.1%}] | "
            f"{r['mean_ci_width']:.3f}{marker} |"
        )

    if recommended_n:
        summary_lines.append(
            f"\n**Recommendation:** <<COLOR:green>>{recommended_n} runs<</COLOR>> "
            f"for ≥90% probability of detecting an effect of size {expected_effect}."
        )
    else:
        summary_lines.append(
            "\n<<COLOR:amber>>Warning: None of the tested sample sizes achieved 90% detection. "
            "Consider a larger experiment or increasing the expected effect size.<</COLOR>>"
        )

    summary_lines.append(
        "\n*Pre-posterior analysis: simulates the experiment before you run it. "
        "This tells you how many runs you need to be confident in the result.*"
    )

    # --- Plots ---
    plots = []
    ns = [r["n"] for r in results]

    # Plot 1: n vs P(detect) with band
    plots.append(
        {
            "data": [
                {
                    "type": "scatter",
                    "x": ns,
                    "y": [r["p90_detect"] for r in results],
                    "mode": "lines",
                    "line": {"width": 0},
                    "showlegend": False,
                },
                {
                    "type": "scatter",
                    "x": ns,
                    "y": [r["p10_detect"] for r in results],
                    "mode": "lines",
                    "fill": "tonexty",
                    "fillcolor": "rgba(52, 152, 219, 0.2)",
                    "line": {"width": 0},
                    "name": "10th-90th percentile",
                },
                {
                    "type": "scatter",
                    "x": ns,
                    "y": [r["mean_detect"] for r in results],
                    "mode": "lines+markers",
                    "line": {"color": "rgba(52, 152, 219, 1)", "width": 2},
                    "marker": {"size": 8},
                    "name": "Mean P(Detect)",
                },
            ],
            "layout": {
                "title": "Detection Probability vs Sample Size",
                "xaxis": {"title": "Number of Runs"},
                "yaxis": {"title": "P(Detect Effect)", "range": [0, 1]},
                "shapes": [
                    {
                        "type": "line",
                        "x0": min(ns),
                        "x1": max(ns),
                        "y0": 0.90,
                        "y1": 0.90,
                        "line": {"color": "green", "width": 2, "dash": "dash"},
                    }
                ],
            },
        }
    )

    # Plot 2: n vs CI width
    plots.append(
        {
            "data": [
                {
                    "type": "scatter",
                    "x": ns,
                    "y": [r["p90_ci"] for r in results],
                    "mode": "lines",
                    "line": {"width": 0},
                    "showlegend": False,
                },
                {
                    "type": "scatter",
                    "x": ns,
                    "y": [r["p10_ci"] for r in results],
                    "mode": "lines",
                    "fill": "tonexty",
                    "fillcolor": "rgba(155, 89, 182, 0.2)",
                    "line": {"width": 0},
                    "name": "10th-90th percentile",
                },
                {
                    "type": "scatter",
                    "x": ns,
                    "y": [r["mean_ci_width"] for r in results],
                    "mode": "lines+markers",
                    "line": {"color": "rgba(155, 89, 182, 1)", "width": 2},
                    "marker": {"size": 8},
                    "name": "Mean CI Width",
                },
            ],
            "layout": {
                "title": "Expected 95% CI Width vs Sample Size",
                "xaxis": {"title": "Number of Runs"},
                "yaxis": {"title": "CI Width (coded units)"},
            },
        }
    )

    guide_obs = (
        f"DOE sample size: {n_factors} factors, effect={expected_effect}, σ={expected_sigma}. "
        + (
            f"Recommended {recommended_n} runs."
            if recommended_n
            else "No size achieved 90% detection."
        )
    )

    return {
        "summary": "\n".join(summary_lines),
        "plots": plots,
        "guide_observation": guide_obs,
        "statistics": {
            "n_factors": n_factors,
            "expected_effect": expected_effect,
            "expected_sigma": expected_sigma,
            "n_simulations": n_sim,
            "recommended_n": recommended_n,
            "results": results,
        },
    }


# ---------------------------------------------------------------------------
# Tool 4: Response Optimization
# ---------------------------------------------------------------------------


def _run_doe_optimize(df, config):
    """
    Bayesian response optimization — propagates parameter uncertainty.

    Grid search over coded factor space with predictive posterior.
    Reports expected desirability, not just point estimates.
    """
    factor_cols = config.get("factor_columns", [])
    response_col = config.get("response_column", "")
    goal = config.get("goal", "maximize")  # maximize, minimize, target
    target_value = config.get("target_value")
    include_2fi = config.get("include_2fi", True)

    if not factor_cols or not response_col:
        return {
            "summary": "<<COLOR:error>>Please specify factor columns and a response column.<</COLOR>>",
            "plots": [],
            "guide_observation": None,
            "statistics": {},
        }

    X, y, term_names, coding_records = build_doe_design_matrix(
        df, factor_cols, response_col, include_2fi=include_2fi
    )
    n, p = X.shape

    if n < p:
        return {
            "summary": f"<<COLOR:error>>Not enough data: {n} runs for {p} parameters.<</COLOR>>",
            "plots": [],
            "guide_observation": None,
            "statistics": {},
        }

    mu_n, Lambda_n, L_n, alpha_n, beta_n = bayesian_linear_posterior(X, y)

    # Grid search: cap at 3 factors for full grid, hold others at midpoint
    k = len(factor_cols)
    n_grid_pts = 21

    if k <= 3:
        grid_factors = list(range(k))
    else:
        # Pick top 3 by effect magnitude
        effect_mags = []
        for j in range(k):
            e_j = np.zeros(p)
            e_j[1 + j] = 1.0  # +1 for intercept
            loc, _, _ = contrast_posterior(e_j, mu_n, Lambda_n, L_n, alpha_n, beta_n)
            effect_mags.append((abs(loc), j))
        effect_mags.sort(reverse=True)
        grid_factors = [idx for _, idx in effect_mags[:3]]

    # Build grid
    grid_1d = np.linspace(-1, 1, n_grid_pts)
    n_active = len(grid_factors)

    if n_active == 1:
        grid_points = grid_1d.reshape(-1, 1)
    elif n_active == 2:
        g1, g2 = np.meshgrid(grid_1d, grid_1d)
        grid_points = np.column_stack([g1.ravel(), g2.ravel()])
    else:
        g = np.meshgrid(*[grid_1d] * n_active)
        grid_points = np.column_stack([gi.ravel() for gi in g])

    n_grid = len(grid_points)

    # Build full coded vectors (non-grid factors at midpoint = 0)
    coded_full = np.zeros((n_grid, k))
    for i, gf_idx in enumerate(grid_factors):
        coded_full[:, gf_idx] = grid_points[:, i]

    # Build X_pred matching the model terms
    X_pred = np.ones((n_grid, 1))  # intercept
    for j in range(k):
        X_pred = np.column_stack([X_pred, coded_full[:, j]])
    # Add interactions if needed
    if include_2fi and k >= 2:
        for i, j in combinations(range(k), 2):
            X_pred = np.column_stack([X_pred, coded_full[:, i] * coded_full[:, j]])

    # Compute predictive posterior for each grid point
    # Use MC for desirability (100 draws per point for speed)
    rng = np.random.default_rng(42)
    n_mc = 200
    best_score = -np.inf
    best_idx = 0
    scores = np.zeros(n_grid)
    pred_means = np.zeros(n_grid)
    pred_lows = np.zeros(n_grid)
    pred_highs = np.zeros(n_grid)

    for i in range(n_grid):
        x_i = X_pred[i]
        loc, scale, df_t = predictive_posterior(
            x_i, mu_n, Lambda_n, L_n, alpha_n, beta_n
        )
        pred_means[i] = loc

        # MC draws from Student-t
        draws = t_dist.rvs(df_t, loc=loc, scale=scale, size=n_mc, random_state=rng)
        pred_lows[i] = float(np.percentile(draws, 2.5))
        pred_highs[i] = float(np.percentile(draws, 97.5))

        # Desirability
        if goal == "maximize":
            score = float(np.mean(draws))
        elif goal == "minimize":
            score = -float(np.mean(draws))
        elif goal == "target" and target_value is not None:
            score = -float(np.mean((draws - target_value) ** 2))
        else:
            score = float(np.mean(draws))

        scores[i] = score
        if score > best_score:
            best_score = score
            best_idx = i

    # Decode optimal point
    optimal_coded = coded_full[best_idx]
    optimal_natural = _decode_coded_to_natural(
        {factor_cols[j]: optimal_coded[j] for j in range(k)},
        coding_records,
        factor_cols,
    )
    optimal_pred = pred_means[best_idx]
    optimal_ci = (pred_lows[best_idx], pred_highs[best_idx])

    # --- Summary ---
    summary_lines = [
        "## Bayesian Response Optimization\n",
        f"**Goal:** {goal.capitalize()} {response_col}",
    ]
    if goal == "target" and target_value is not None:
        summary_lines[-1] += f" (target = {target_value})"

    summary_lines.extend(
        [
            f"**Design:** {n} runs, {k} factors\n",
            "### Optimal Settings",
            "| Factor | Coded | Natural |",
            "|--------|-------|---------|",
        ]
    )
    for col in factor_cols:
        coded_val = optimal_coded[factor_cols.index(col)]
        natural_val = optimal_natural.get(col, coded_val)
        if isinstance(natural_val, float):
            summary_lines.append(f"| {col} | {coded_val:+.3f} | {natural_val:.4g} |")
        else:
            summary_lines.append(f"| {col} | {coded_val:+.3f} | {natural_val} |")

    summary_lines.extend(
        [
            f"\n**Predicted {response_col}:** {optimal_pred:.4g} [95% CI: {optimal_ci[0]:.4g}, {optimal_ci[1]:.4g}]",
            "\n*Unlike frequentist optimization, this propagates parameter uncertainty "
            "through the prediction. The CI reflects both model and noise uncertainty.*",
        ]
    )

    if k > 3:
        held = [c for i, c in enumerate(factor_cols) if i not in grid_factors]
        summary_lines.append(
            f"\n*Note: {', '.join(held)} held at midpoint (coded=0) — "
            f"optimization over top 3 factors by effect magnitude.*"
        )

    # --- Plots ---
    plots = []

    if n_active >= 2:
        # Plot 1: Contour of predicted response (top 2 factors)
        f1_idx, f2_idx = grid_factors[0], grid_factors[1]
        f1_name = factor_cols[f1_idx]
        f2_name = factor_cols[f2_idx]

        # Extract 2D slice for contour
        z_grid = pred_means.reshape([n_grid_pts] * n_active)
        if n_active == 2:
            z_2d = z_grid
        else:
            # Slice at optimal value of 3rd factor
            opt_3rd_coded = optimal_coded[grid_factors[2]]
            slice_idx = int(round((opt_3rd_coded + 1) / 2 * (n_grid_pts - 1)))
            slice_idx = max(0, min(n_grid_pts - 1, slice_idx))
            z_2d = z_grid[:, :, slice_idx]

        plots.append(
            {
                "data": [
                    {
                        "type": "contour",
                        "x": grid_1d.tolist(),
                        "y": grid_1d.tolist(),
                        "z": z_2d.tolist(),
                        "colorscale": "Viridis",
                        "colorbar": {"title": response_col},
                    },
                    {
                        "type": "scatter",
                        "x": [float(optimal_coded[f1_idx])],
                        "y": [float(optimal_coded[f2_idx])],
                        "mode": "markers",
                        "marker": {"size": 14, "color": "red", "symbol": "star"},
                        "name": "Optimal",
                    },
                ],
                "layout": {
                    "title": f"Predicted {response_col} Surface",
                    "xaxis": {"title": f"{f1_name} (coded)"},
                    "yaxis": {"title": f"{f2_name} (coded)"},
                },
            }
        )

    # Plot 2: Marginal response curves per factor
    marginal_traces = []
    for j in range(min(k, 4)):
        x_sweep = np.zeros((n_grid_pts, k))
        x_sweep[:, j] = grid_1d
        # Build design vectors
        X_sweep = np.ones((n_grid_pts, 1))
        for jj in range(k):
            X_sweep = np.column_stack([X_sweep, x_sweep[:, jj]])
        if include_2fi and k >= 2:
            for ii, jj in combinations(range(k), 2):
                X_sweep = np.column_stack([X_sweep, x_sweep[:, ii] * x_sweep[:, jj]])

        means_j = np.zeros(n_grid_pts)
        lows_j = np.zeros(n_grid_pts)
        highs_j = np.zeros(n_grid_pts)
        for i in range(n_grid_pts):
            loc, scale, df_t = predictive_posterior(
                X_sweep[i], mu_n, Lambda_n, L_n, alpha_n, beta_n
            )
            means_j[i] = loc
            ci_half = t_dist.ppf(0.975, df_t) * scale
            lows_j[i] = loc - ci_half
            highs_j[i] = loc + ci_half

        marginal_traces.append(
            {
                "type": "scatter",
                "x": grid_1d.tolist(),
                "y": means_j.tolist(),
                "mode": "lines",
                "name": factor_cols[j],
                "line": {"width": 2},
            }
        )
        marginal_traces.append(
            {
                "type": "scatter",
                "x": grid_1d.tolist() + grid_1d[::-1].tolist(),
                "y": highs_j.tolist() + lows_j[::-1].tolist(),
                "fill": "toself",
                "fillcolor": "rgba(0,0,0,0.05)",
                "line": {"width": 0},
                "showlegend": False,
            }
        )

    plots.append(
        {
            "data": marginal_traces,
            "layout": {
                "title": "Marginal Response Curves (others at midpoint)",
                "xaxis": {"title": "Coded Factor Level"},
                "yaxis": {"title": response_col},
            },
        }
    )

    # Plot 3: Optimal point annotation
    plots.append(
        {
            "data": [
                {
                    "type": "scatter",
                    "x": y.tolist(),
                    "y": (X @ mu_n).tolist(),
                    "mode": "markers",
                    "marker": {"size": 8, "color": "rgba(52, 152, 219, 0.7)"},
                    "name": "Data (obs vs fit)",
                },
                {
                    "type": "scatter",
                    "x": [float(optimal_pred)],
                    "y": [float(optimal_pred)],
                    "mode": "markers",
                    "marker": {"size": 14, "color": "red", "symbol": "star"},
                    "name": f"Optimal: {optimal_pred:.3g}",
                },
                {
                    "type": "scatter",
                    "x": [float(np.min(y)), float(np.max(y))],
                    "y": [float(np.min(y)), float(np.max(y))],
                    "mode": "lines",
                    "line": {"color": "gray", "dash": "dash"},
                    "showlegend": False,
                },
            ],
            "layout": {
                "title": "Model Fit + Optimal Prediction",
                "xaxis": {"title": "Observed"},
                "yaxis": {"title": "Predicted"},
                "annotations": [
                    {
                        "x": float(optimal_pred),
                        "y": float(optimal_pred),
                        "text": f"Optimal: {optimal_pred:.3g}<br>[{optimal_ci[0]:.3g}, {optimal_ci[1]:.3g}]",
                        "showarrow": True,
                        "arrowhead": 2,
                    }
                ],
            },
        }
    )

    guide_obs = (
        f"DOE optimization ({goal}): optimal {response_col} = {optimal_pred:.4g} "
        f"[{optimal_ci[0]:.4g}, {optimal_ci[1]:.4g}]."
    )

    return {
        "summary": "\n".join(summary_lines),
        "plots": plots,
        "guide_observation": guide_obs,
        "statistics": {
            "goal": goal,
            "target_value": target_value,
            "optimal_coded": {
                factor_cols[j]: round(float(optimal_coded[j]), 4) for j in range(k)
            },
            "optimal_natural": {
                k_: round(v, 4) if isinstance(v, float) else v
                for k_, v in optimal_natural.items()
            },
            "predicted_response": round(optimal_pred, 4),
            "prediction_ci": [round(optimal_ci[0], 4), round(optimal_ci[1], 4)],
        },
    }


# ---------------------------------------------------------------------------
# Tool 5: Next Experiment (Sequential DOE)
# ---------------------------------------------------------------------------


def _run_doe_next(df, config):
    """
    Suggest next experiments to maximize information gain.

    Criterion: expected reduction in predictive variance across the factor space.
    Uses rank-1 precision updates (no re-inversion).
    """
    factor_cols = config.get("factor_columns", [])
    response_col = config.get("response_column", "")
    n_suggest = int(config.get("n_suggest", 4))
    include_2fi = config.get("include_2fi", True)

    if not factor_cols or not response_col:
        return {
            "summary": "<<COLOR:error>>Please specify factor columns and a response column.<</COLOR>>",
            "plots": [],
            "guide_observation": None,
            "statistics": {},
        }

    X, y, term_names, coding_records = build_doe_design_matrix(
        df, factor_cols, response_col, include_2fi=include_2fi
    )
    n, p_dim = X.shape

    if n < p_dim:
        return {
            "summary": f"<<COLOR:error>>Not enough data: {n} runs for {p_dim} parameters.<</COLOR>>",
            "plots": [],
            "guide_observation": None,
            "statistics": {},
        }

    mu_n, Lambda_n, L_n, alpha_n, beta_n = bayesian_linear_posterior(X, y)

    k = len(factor_cols)

    # Representative prediction points: Latin hypercube over [-1, 1]^k
    rng = np.random.default_rng(42)
    n_pred = 50  # evaluation points for average predictive variance
    pred_points_coded = np.zeros((n_pred, k))
    for j in range(k):
        perm = rng.permutation(n_pred)
        pred_points_coded[:, j] = (perm + rng.random(n_pred)) / n_pred * 2 - 1

    # Build prediction design vectors
    X_pred_pts = np.ones((n_pred, 1))
    for j in range(k):
        X_pred_pts = np.column_stack([X_pred_pts, pred_points_coded[:, j]])
    if include_2fi and k >= 2:
        for i, j in combinations(range(k), 2):
            X_pred_pts = np.column_stack(
                [X_pred_pts, pred_points_coded[:, i] * pred_points_coded[:, j]]
            )

    # Current average predictive variance
    scale_factor = beta_n / alpha_n

    def avg_pred_var(Lambda_mat, L_mat):
        """Average predictive variance across representative points."""
        total = 0.0
        for i in range(n_pred):
            x_i = X_pred_pts[i]
            v = cho_solve((L_mat, True), x_i)
            total += 1.0 + float(x_i @ v)
        return scale_factor * total / n_pred

    current_apv = avg_pred_var(Lambda_n, L_n)

    # Candidate points: Latin hypercube, 200 candidates
    n_candidates = 200
    candidates_coded = np.zeros((n_candidates, k))
    for j in range(k):
        perm = rng.permutation(n_candidates)
        candidates_coded[:, j] = (
            perm + rng.random(n_candidates)
        ) / n_candidates * 2 - 1

    # Build candidate design vectors
    X_cand = np.ones((n_candidates, 1))
    for j in range(k):
        X_cand = np.column_stack([X_cand, candidates_coded[:, j]])
    if include_2fi and k >= 2:
        for i, j in combinations(range(k), 2):
            X_cand = np.column_stack(
                [X_cand, candidates_coded[:, i] * candidates_coded[:, j]]
            )

    # Evaluate information gain for each candidate
    gains = np.zeros(n_candidates)
    for c_idx in range(n_candidates):
        x_new = X_cand[c_idx]

        # Rank-1 update: Λ_{n+1} = Λn + x_new x_new'
        Lambda_new = Lambda_n + np.outer(x_new, x_new)
        try:
            L_new = np.linalg.cholesky(Lambda_new)
            new_apv = avg_pred_var(Lambda_new, L_new)
            gains[c_idx] = current_apv - new_apv
        except np.linalg.LinAlgError:
            gains[c_idx] = 0.0

    # Rank and select top suggestions
    top_indices = np.argsort(gains)[::-1][:n_suggest]

    suggestions = []
    for rank, idx in enumerate(top_indices):
        coded_vals = candidates_coded[idx]
        natural_vals = _decode_coded_to_natural(
            {factor_cols[j]: coded_vals[j] for j in range(k)},
            coding_records,
            factor_cols,
        )
        reduction_pct = (
            float(gains[idx] / current_apv * 100) if current_apv > 0 else 0.0
        )
        suggestions.append(
            {
                "rank": rank + 1,
                "coded": {
                    factor_cols[j]: round(float(coded_vals[j]), 3) for j in range(k)
                },
                "natural": {
                    kk: round(v, 4) if isinstance(v, float) else v
                    for kk, v in natural_vals.items()
                },
                "info_gain": round(float(gains[idx]), 6),
                "reduction_pct": round(reduction_pct, 1),
            }
        )

    # --- Summary ---
    summary_lines = [
        "## Next Experiment Suggestions\n",
        f"**Current data:** {n} runs, {k} factors",
        f"**Current avg predictive variance:** {current_apv:.4g}\n",
        "### Suggested Runs (ranked by information gain)",
    ]

    header = "| Rank |"
    sep = "|------|"
    for col in factor_cols:
        header += f" {col} |"
        sep += "------|"
    header += " Uncertainty Reduction |"
    sep += "----------------------|"
    summary_lines.extend([header, sep])

    for s in suggestions:
        row = f"| {s['rank']} |"
        for col in factor_cols:
            val = s["natural"].get(col, s["coded"][col])
            if isinstance(val, float):
                row += f" {val:.3g} |"
            else:
                row += f" {val} |"
        row += f" {s['reduction_pct']:.1f}% |"
        summary_lines.append(row)

    total_reduction = sum(s["reduction_pct"] for s in suggestions)
    summary_lines.extend(
        [
            f"\n**Expected total uncertainty reduction:** {total_reduction:.1f}% (if all {n_suggest} runs are added)",
            "\n*Sequential DOE: each suggested point maximally reduces prediction uncertainty "
            "across the factor space. Run these experiments, add the data, and re-analyze.*",
        ]
    )

    # --- Plots ---
    plots = []

    # Plot 1: Heatmap of information gain (if k=2) or bar chart
    if k == 2:
        # 2D heatmap
        grid_1d = np.linspace(-1, 1, 31)
        g1, g2 = np.meshgrid(grid_1d, grid_1d)
        gain_grid = np.zeros((31, 31))

        for i in range(31):
            for j in range(31):
                x_test = np.ones(p_dim)
                x_test[1] = g1[i, j]
                x_test[2] = g2[i, j]
                if include_2fi:
                    x_test[3] = g1[i, j] * g2[i, j]
                Lambda_test = Lambda_n + np.outer(x_test, x_test)
                try:
                    L_test = np.linalg.cholesky(Lambda_test)
                    test_apv = avg_pred_var(Lambda_test, L_test)
                    gain_grid[i, j] = current_apv - test_apv
                except np.linalg.LinAlgError:
                    gain_grid[i, j] = 0.0

        plots.append(
            {
                "data": [
                    {
                        "type": "heatmap",
                        "x": grid_1d.tolist(),
                        "y": grid_1d.tolist(),
                        "z": gain_grid.tolist(),
                        "colorscale": "YlOrRd",
                        "colorbar": {"title": "Info Gain"},
                    },
                    {
                        "type": "scatter",
                        "x": [s["coded"][factor_cols[0]] for s in suggestions],
                        "y": [s["coded"][factor_cols[1]] for s in suggestions],
                        "mode": "markers+text",
                        "marker": {"size": 12, "color": "blue", "symbol": "star"},
                        "text": [str(s["rank"]) for s in suggestions],
                        "textposition": "top center",
                        "name": "Suggestions",
                    },
                    {
                        "type": "scatter",
                        "x": X[:, 1].tolist(),
                        "y": X[:, 2].tolist(),
                        "mode": "markers",
                        "marker": {
                            "size": 6,
                            "color": "white",
                            "line": {"width": 1, "color": "black"},
                        },
                        "name": "Existing runs",
                    },
                ],
                "layout": {
                    "title": "Information Gain Across Factor Space",
                    "xaxis": {"title": f"{factor_cols[0]} (coded)"},
                    "yaxis": {"title": f"{factor_cols[1]} (coded)"},
                },
            }
        )
    else:
        # Bar chart of suggestions
        plots.append(
            {
                "data": [
                    {
                        "type": "bar",
                        "x": [f"Run {s['rank']}" for s in suggestions],
                        "y": [s["reduction_pct"] for s in suggestions],
                        "marker": {"color": "rgba(52, 152, 219, 0.8)"},
                        "text": [f"{s['reduction_pct']:.1f}%" for s in suggestions],
                        "textposition": "auto",
                    }
                ],
                "layout": {
                    "title": "Expected Uncertainty Reduction per Suggested Run",
                    "yaxis": {"title": "Variance Reduction (%)"},
                },
            }
        )

    # Plot 2: Table visualization as a simple scatter showing coded settings
    traces = []
    for j in range(k):
        traces.append(
            {
                "type": "scatter",
                "x": [s["rank"] for s in suggestions],
                "y": [s["coded"][factor_cols[j]] for s in suggestions],
                "mode": "markers+lines",
                "name": factor_cols[j],
                "marker": {"size": 10},
            }
        )
    plots.append(
        {
            "data": traces,
            "layout": {
                "title": "Suggested Factor Settings",
                "xaxis": {"title": "Suggestion Rank", "dtick": 1},
                "yaxis": {"title": "Coded Level", "range": [-1.2, 1.2]},
            },
        }
    )

    guide_obs = f"Sequential DOE: {n_suggest} experiments suggested. Total expected variance reduction: {total_reduction:.1f}%."

    return {
        "summary": "\n".join(summary_lines),
        "plots": plots,
        "guide_observation": guide_obs,
        "statistics": {
            "n_existing": n,
            "current_avg_pred_var": round(current_apv, 6),
            "suggestions": suggestions,
        },
    }

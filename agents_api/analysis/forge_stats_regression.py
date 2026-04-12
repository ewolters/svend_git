"""Forge-backed regression analysis handlers.

Split from forge_stats.py for compliance (3000-line limit).
Object 271 — Analysis Workbench migration.
"""

import logging

import numpy as np
import pandas as pd

from .forge_stats import _alpha, _pval_str, _rich_summary, _to_chart

logger = logging.getLogger(__name__)


# =============================================================================
# Regression
# =============================================================================


def forge_regression(df, config):
    """Linear regression via forgestat."""
    from forgestat.regression.linear import ols
    from forgeviz.charts.diagnostic import four_in_one
    from forgeviz.charts.scatter import scatter

    response = config.get("response") or config.get("column")
    predictors = config.get("predictors", [])
    if not response:
        raise ValueError("Regression requires 'response'")
    if not predictors:
        nums = df.select_dtypes(include="number").columns.tolist()
        predictors = [c for c in nums if c != response]
    if not predictors:
        raise ValueError("No predictors available")

    sub = df[[response] + predictors].copy()
    for c in [response] + predictors:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 3:
        raise ValueError("Not enough valid observations for regression")
    y = sub[response].values
    X = sub[predictors].values

    result = ols(X, y, feature_names=predictors, alpha=_alpha(config))

    charts = []
    # Scatter + regression line for simple regression
    if len(predictors) == 1:
        spec = scatter(
            x=X[:, 0].tolist(),
            y=y.tolist(),
            x_label=predictors[0],
            y_label=response,
            title=f"{response} vs {predictors[0]}",
            show_regression=True,
        )
        charts.append(_to_chart(spec))

    # Diagnostic 4-in-1
    if result.fitted is not None and result.residuals is not None:
        diag_specs = four_in_one(
            fitted=result.fitted.tolist() if hasattr(result.fitted, "tolist") else list(result.fitted),
            residuals=result.residuals.tolist() if hasattr(result.residuals, "tolist") else list(result.residuals),
        )
        for s in diag_specs:
            charts.append(_to_chart(s))

    stats = {
        "r_squared": round(result.r_squared, 4),
        "adj_r_squared": round(result.adj_r_squared, 4),
        "f_statistic": round(result.f_statistic, 4) if result.f_statistic else None,
        "f_p_value": result.f_p_value,
        "durbin_watson": round(result.durbin_watson, 4) if result.durbin_watson else None,
        "rmse": round(result.rmse, 4),
        "n": result.n,
    }
    coeffs = result.coefficients if isinstance(result.coefficients, dict) else {}
    pvals = result.p_values if isinstance(result.p_values, dict) else {}
    for key, val in coeffs.items():
        stats[f"coeff({key})"] = round(val, 4)
        if key in pvals:
            stats[f"p({key})"] = pvals[key]

    return {
        "plots": charts,
        "statistics": stats,
        "assumptions": {},
        "summary": f"Regression: R²={result.r_squared:.4f}, adj R²={result.adj_r_squared:.4f}, F={result.f_statistic:.3f}, p={_pval_str(result.f_p_value)}.",
        "narrative": {
            "verdict": f"Model explains {result.r_squared * 100:.1f}% of variation",
            "body": f"Linear regression with {len(predictors)} predictor(s). R²={result.r_squared:.4f}, RMSE={result.rmse:.4f}.",
            "next_steps": "Check residual diagnostics for model adequacy.",
            "chart_guidance": "Diagnostic plots show residual patterns. Look for non-random patterns.",
        },
        "guide_observation": f"Regression: R²={result.r_squared:.4f}, p={_pval_str(result.f_p_value)}.",
        "diagnostics": [],
    }


def forge_logistic(df, config):
    """Logistic regression via forgestat."""
    from forgestat.regression.logistic import logistic_regression

    response = config.get("response") or config.get("column")
    predictors = config.get("predictors", [])
    if not response:
        raise ValueError("Logistic regression requires 'response'")
    if not predictors:
        nums = df.select_dtypes(include="number").columns.tolist()
        predictors = [c for c in nums if c != response]

    sub = df[[response] + predictors].copy()
    for c in [response] + predictors:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 3:
        raise ValueError("Not enough valid observations for logistic regression")
    y = sub[response].values.astype(int)
    X = sub[predictors].values

    result = logistic_regression(X, y, feature_names=predictors)

    stats = {
        "pseudo_r_squared": round(result.pseudo_r_squared, 4),
        "aic": round(result.aic, 2),
        "n": result.n,
        "converged": result.converged,
    }
    coeffs = result.coefficients if isinstance(result.coefficients, dict) else {}
    odds = result.odds_ratios if isinstance(result.odds_ratios, dict) else {}
    pvals = result.p_values if isinstance(result.p_values, dict) else {}
    for key, val in coeffs.items():
        stats[f"coeff({key})"] = round(val, 4)
        if key in odds:
            stats[f"OR({key})"] = round(odds[key], 4)
        if key in pvals:
            stats[f"p({key})"] = pvals[key]

    return {
        "plots": [],
        "statistics": stats,
        "assumptions": {},
        "summary": f"Logistic regression: pseudo R²={result.pseudo_r_squared:.4f}, AIC={result.aic:.1f}, n={result.n}.",
        "narrative": {
            "verdict": f"Model {'converged' if result.converged else 'did not converge'}",
            "body": f"Logistic regression with {len(predictors)} predictors. Pseudo R²={result.pseudo_r_squared:.4f}.",
            "next_steps": "Examine odds ratios for effect interpretation.",
            "chart_guidance": "",
        },
        "guide_observation": f"Logistic: pseudo R²={result.pseudo_r_squared:.4f}.",
        "diagnostics": [],
    }


def forge_stepwise(df, config):
    """Stepwise regression via forgestat."""
    from forgestat.regression.stepwise import stepwise

    response = config.get("response") or config.get("column")
    predictors = config.get("predictors", [])
    if not response:
        raise ValueError("Stepwise requires 'response'")
    if not predictors:
        nums = df.select_dtypes(include="number").columns.tolist()
        predictors = [c for c in nums if c != response]

    sub = df[[response] + predictors].copy()
    for c in [response] + predictors:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna()
    if len(sub) < 3:
        raise ValueError("Not enough valid observations for stepwise")
    y = sub[response].values
    X = sub[predictors].values

    result = stepwise(X, y, feature_names=predictors, method=config.get("method", "both"))

    stats = {"selected_features": result.selected_features, "n_steps": len(result.steps)}
    if result.final_model:
        stats["r_squared"] = round(result.final_model.r_squared, 4)
        stats["adj_r_squared"] = round(result.final_model.adj_r_squared, 4)

    return {
        "plots": [],
        "statistics": stats,
        "assumptions": {},
        "summary": f"Stepwise ({result.method}): selected {len(result.selected_features)} of {len(predictors)} predictors"
        + (f", R²={result.final_model.r_squared:.4f}" if result.final_model else "")
        + ".",
        "narrative": {
            "verdict": f"Selected: {', '.join(result.selected_features) if result.selected_features else 'none'}",
            "body": f"Stepwise selection completed in {len(result.steps)} steps.",
            "next_steps": "Validate with holdout data.",
            "chart_guidance": "",
        },
        "guide_observation": f"Stepwise: {len(result.selected_features)} predictors selected.",
        "diagnostics": [],
    }


# =============================================================================
# Robust Regression
# =============================================================================


def forge_robust_regression(df, config):
    """Robust regression via forgestat."""
    from forgestat.regression.robust import robust_regression
    from forgeviz.charts.scatter import scatter

    response = config.get("response") or config.get("var")
    predictors = list(config.get("predictors", []))
    method = config.get("method", "huber")

    if response in predictors:
        predictors.remove(response)
    if not predictors:
        raise ValueError("Need at least one predictor")

    data_clean = df[[response] + predictors].dropna()
    y = pd.to_numeric(data_clean[response], errors="coerce").values
    X = data_clean[predictors].apply(pd.to_numeric, errors="coerce").values

    result = robust_regression(X, y, feature_names=predictors, method=method)

    # Chart: fitted vs actual
    fitted = y - np.array(result.residuals)
    chart = scatter(
        x=fitted.tolist(),
        y=y.tolist(),
        title=f"Robust Regression: Fitted vs Actual ({method})",
        x_label="Fitted",
        y_label="Actual",
    )
    plots = [_to_chart(chart)]

    coef_items = [
        (
            name,
            f"{val:.4f} (OLS: {result.ols_coefficients.get(name, 0):.4f}, Δ={result.coefficient_changes.get(name, 0):.1f}%)",
        )
        for name, val in result.coefficients.items()
    ]

    return {
        "plots": plots,
        "statistics": {
            "method": result.method,
            "coefficients": result.coefficients,
            "ols_coefficients": result.ols_coefficients,
            "coefficient_changes_pct": result.coefficient_changes,
            "r_squared": round(result.r_squared, 4),
            "n_downweighted": result.n_downweighted,
            "n": len(y),
        },
        "assumptions": {},
        "summary": _rich_summary(
            f"ROBUST REGRESSION ({method.upper()})",
            [
                (
                    "Design",
                    [
                        ("Response", response),
                        ("Predictors", ", ".join(predictors)),
                        ("Method", method),
                        ("N", str(len(y))),
                    ],
                ),
                ("Coefficients (Robust vs OLS)", coef_items),
                (
                    "Fit",
                    [
                        ("R²", f"{result.r_squared:.4f}"),
                        ("Downweighted obs", str(result.n_downweighted)),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"Robust regression ({method}): R² = {result.r_squared:.3f}, {result.n_downweighted} observations downweighted",
            "body": (
                f"Comparing robust vs OLS coefficients shows where outliers pull the fit. "
                f"{result.n_downweighted} observations received reduced weight. "
                + (
                    "Large coefficient changes suggest OLS is sensitive to outliers here."
                    if any(abs(v) > 20 for v in result.coefficient_changes.values())
                    else "Coefficients are stable — outlier influence is minimal."
                )
            ),
            "next_steps": "Compare robust and OLS residuals to identify influential observations.",
            "chart_guidance": "Scatter plot shows fitted vs actual values. Points near the diagonal indicate good fit.",
        },
        "guide_observation": f"Robust regression ({method}): R²={result.r_squared:.3f}, {result.n_downweighted} downweighted.",
        "diagnostics": [],
    }


# =============================================================================
# Poisson Regression
# =============================================================================


def forge_poisson_regression(df, config):
    """Poisson regression via forgestat."""
    from forgestat.regression.logistic import poisson_regression
    from forgeviz.charts.generic import bar

    response = config.get("response") or config.get("var")
    predictors = list(config.get("predictors", []))

    if response in predictors:
        predictors.remove(response)
    if not predictors:
        raise ValueError("Need at least one predictor")

    data_clean = df[[response] + predictors].dropna()
    y = pd.to_numeric(data_clean[response], errors="coerce").values.astype(int)
    X = data_clean[predictors].apply(pd.to_numeric, errors="coerce").values

    result = poisson_regression(X, y, feature_names=predictors)

    # Chart: IRR bar chart
    irr_names = [k for k in result.irr.keys() if k != "intercept"]
    irr_vals = [result.irr[k] for k in irr_names]
    chart = bar(
        categories=irr_names,
        values=irr_vals,
        title="Incidence Rate Ratios",
        y_label="IRR",
    )
    plots = [_to_chart(chart)]

    coef_items = [
        (
            name,
            f"β={result.coefficients.get(name, 0):.4f}, IRR={result.irr.get(name, 0):.4f}, p={_pval_str(result.p_values.get(name, 1))}",
        )
        for name in irr_names
    ]

    sig_predictors = [k for k in irr_names if result.p_values.get(k, 1) < 0.05]

    return {
        "plots": plots,
        "statistics": {
            "coefficients": result.coefficients,
            "irr": result.irr,
            "p_values": result.p_values,
            "deviance": round(result.deviance, 4),
            "pearson_chi2": round(result.pearson_chi2, 4),
            "aic": round(result.aic, 4),
            "n": result.n,
        },
        "assumptions": {},
        "summary": _rich_summary(
            "POISSON REGRESSION",
            [
                (
                    "Design",
                    [
                        ("Response", f"{response} (count data)"),
                        ("Predictors", ", ".join(predictors)),
                        ("N", str(result.n)),
                    ],
                ),
                ("Coefficients", coef_items),
                (
                    "Model Fit",
                    [
                        ("Deviance", f"{result.deviance:.4f}"),
                        ("Pearson χ²", f"{result.pearson_chi2:.4f}"),
                        ("AIC", f"{result.aic:.4f}"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"Poisson regression: {len(sig_predictors)} significant predictors, AIC = {result.aic:.1f}",
            "body": (
                "IRR > 1 means increased rate; IRR < 1 means decreased rate. "
                + (
                    f"Significant: <strong>{', '.join(sig_predictors)}</strong>."
                    if sig_predictors
                    else "No predictors reached significance."
                )
                + (
                    f" Check for overdispersion: Pearson χ²/df = {result.pearson_chi2 / (result.n - len(predictors) - 1):.2f} "
                    f"({'overdispersed — consider negative binomial' if result.pearson_chi2 / (result.n - len(predictors) - 1) > 1.5 else 'acceptable'})."
                    if result.n > len(predictors) + 1
                    else ""
                )
            ),
            "next_steps": "Check for overdispersion. If Pearson χ²/df >> 1, switch to negative binomial regression.",
            "chart_guidance": "Bar chart shows incidence rate ratios. IRR = 1 means no effect; > 1 means increased count rate.",
        },
        "guide_observation": f"Poisson regression: {len(sig_predictors)} significant, AIC={result.aic:.1f}.",
        "diagnostics": [],
    }


# =============================================================================
# Nonlinear Regression (Curve Fitting)
# =============================================================================


def forge_nonlinear_regression(df, config):
    """Nonlinear regression (curve fitting) via forgestat."""
    from forgestat.regression.nonlinear import curve_fit
    from forgeviz.charts.scatter import scatter

    x_var = config.get("x_var") or config.get("var1") or config.get("predictor")
    y_var = config.get("y_var") or config.get("var2") or config.get("response")
    model_type = config.get("model", "exponential")

    if not x_var or not y_var:
        raise ValueError("Nonlinear regression requires x_var and y_var")

    x_data = pd.to_numeric(df[x_var], errors="coerce").dropna()
    y_data = pd.to_numeric(df[y_var], errors="coerce").dropna()
    # Align indices
    common = x_data.index.intersection(y_data.index)
    x_vals = x_data.loc[common].values
    y_vals = y_data.loc[common].values

    result = curve_fit(x_vals, y_vals, model=model_type)

    if not result.converged:
        return {
            "plots": [],
            "statistics": {"converged": False, "model": model_type},
            "summary": f"Model '{model_type}' did not converge. Try a different model or initial parameters.",
            "narrative": {
                "verdict": "Model did not converge",
                "body": "",
                "next_steps": "Try a different model type.",
                "chart_guidance": "",
            },
            "guide_observation": f"Nonlinear fit ({model_type}) did not converge.",
            "diagnostics": [],
        }

    # Chart: data + fitted curve
    chart = scatter(
        x=x_vals.tolist(),
        y=y_vals.tolist(),
        title=f"Nonlinear Fit: {model_type}",
        x_label=x_var,
        y_label=y_var,
    )
    plots = [_to_chart(chart)]

    param_items = [
        (name, f"{val:.6f}" + (f" ± {result.std_errors.get(name, 0):.6f}" if result.std_errors.get(name) else ""))
        for name, val in result.parameters.items()
    ]

    return {
        "plots": plots,
        "statistics": {
            "model": result.model,
            "parameters": result.parameters,
            "std_errors": result.std_errors,
            "r_squared": round(result.r_squared, 4),
            "rmse": round(result.rmse, 4),
            "aic": round(result.aic, 4),
            "bic": round(result.bic, 4),
            "n": result.n,
            "converged": result.converged,
        },
        "assumptions": {},
        "summary": _rich_summary(
            f"NONLINEAR REGRESSION ({model_type.upper()})",
            [
                (
                    "Design",
                    [
                        ("X", x_var),
                        ("Y", y_var),
                        ("Model", model_type),
                        ("N", str(result.n)),
                    ],
                ),
                ("Parameters", param_items),
                (
                    "Fit",
                    [
                        ("R²", f"{result.r_squared:.4f}"),
                        ("RMSE", f"{result.rmse:.4f}"),
                        ("AIC", f"{result.aic:.4f}"),
                        ("BIC", f"{result.bic:.4f}"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"Nonlinear fit ({model_type}): R² = {result.r_squared:.3f}",
            "body": f"The {model_type} model explains {result.r_squared * 100:.1f}% of variance. RMSE = {result.rmse:.4f}.",
            "next_steps": "Compare against other model types (exponential, logistic, power) to find the best fit.",
            "chart_guidance": "Scatter shows observed data. The fitted curve overlays the model prediction.",
        },
        "guide_observation": f"Nonlinear ({model_type}): R²={result.r_squared:.3f}, RMSE={result.rmse:.4f}.",
        "diagnostics": [],
    }


# =============================================================================
# Best Subsets Regression
# =============================================================================


def forge_best_subsets(df, config):
    """Best subsets regression via forgestat."""
    from forgestat.regression.best_subsets import best_subsets

    response = config.get("response") or config.get("var")
    predictors = list(config.get("predictors", []))

    if response in predictors:
        predictors.remove(response)
    if not predictors:
        raise ValueError("Need at least one predictor")

    data_clean = df[[response] + predictors].dropna()
    y = pd.to_numeric(data_clean[response], errors="coerce").values
    X = data_clean[predictors].apply(pd.to_numeric, errors="coerce").values

    result = best_subsets(X, y, feature_names=predictors)

    # Build summary from best models
    sections = [
        (
            "Design",
            [
                ("Response", response),
                ("Candidate predictors", ", ".join(predictors)),
                ("N", str(len(y))),
            ],
        ),
    ]

    if result.best_aic:
        sections.append(
            (
                "Best AIC Model",
                [
                    ("Predictors", ", ".join(result.best_aic.features)),
                    ("AIC", f"{result.best_aic.aic:.4f}"),
                    ("R²", f"{result.best_aic.r_squared:.4f}"),
                    ("Adj R²", f"{result.best_aic.adj_r_squared:.4f}"),
                ],
            )
        )
    if result.best_bic:
        sections.append(
            (
                "Best BIC Model",
                [
                    ("Predictors", ", ".join(result.best_bic.features)),
                    ("BIC", f"{result.best_bic.bic:.4f}"),
                    ("R²", f"{result.best_bic.r_squared:.4f}"),
                    ("Adj R²", f"{result.best_bic.adj_r_squared:.4f}"),
                ],
            )
        )
    if result.best_adj_r2:
        sections.append(
            (
                "Best Adj R² Model",
                [
                    ("Predictors", ", ".join(result.best_adj_r2.features)),
                    ("Adj R²", f"{result.best_adj_r2.adj_r_squared:.4f}"),
                ],
            )
        )

    best = result.best_bic or result.best_aic or result.best_adj_r2
    best_feats = ", ".join(best.features) if best else "none"

    return {
        "plots": [],
        "statistics": {
            "best_aic": {
                "features": result.best_aic.features,
                "aic": round(result.best_aic.aic, 4),
                "r_squared": round(result.best_aic.r_squared, 4),
            }
            if result.best_aic
            else None,
            "best_bic": {
                "features": result.best_bic.features,
                "bic": round(result.best_bic.bic, 4),
                "r_squared": round(result.best_bic.r_squared, 4),
            }
            if result.best_bic
            else None,
            "best_adj_r2": {
                "features": result.best_adj_r2.features,
                "adj_r_squared": round(result.best_adj_r2.adj_r_squared, 4),
            }
            if result.best_adj_r2
            else None,
            "n_subsets_evaluated": len(result.all_subsets),
            "n": len(y),
        },
        "assumptions": {},
        "summary": _rich_summary("BEST SUBSETS REGRESSION", sections),
        "narrative": {
            "verdict": f"Best model uses {len(best.features) if best else 0} of {len(predictors)} predictors",
            "body": f"Best BIC model: <strong>{best_feats}</strong>." if best else "No valid model found.",
            "next_steps": "Use the recommended subset as starting predictors. Validate with out-of-sample data.",
            "chart_guidance": "",
        },
        "guide_observation": f"Best subsets: {best_feats} (of {len(predictors)} candidates).",
        "diagnostics": [],
    }


# =============================================================================
# GLM (Generalized Linear Model)
# =============================================================================


def forge_glm(df, config):
    """Generalized linear model via forgestat."""
    from forgestat.regression.glm import glm

    response = config.get("response") or config.get("var")
    predictors = list(config.get("predictors", []))
    family = config.get("family", "gaussian")

    if response in predictors:
        predictors.remove(response)
    if not predictors:
        raise ValueError("Need at least one predictor")

    data_clean = df[[response] + predictors].dropna()
    y = pd.to_numeric(data_clean[response], errors="coerce").values
    X = data_clean[predictors].apply(pd.to_numeric, errors="coerce").values

    result = glm(X, y, feature_names=predictors, family=family)

    coef_items = [
        (name, f"{val:.4f} (p = {_pval_str(result.p_values.get(name, 1))})")
        for name, val in result.coefficients.items()
    ]
    sig_preds = [k for k in predictors if result.p_values.get(k, 1) < 0.05]

    return {
        "plots": [],
        "statistics": {
            "family": result.family,
            "coefficients": result.coefficients,
            "std_errors": result.std_errors,
            "p_values": result.p_values,
            "deviance": round(result.deviance, 4),
            "aic": round(result.aic, 4),
            "n": result.n,
        },
        "assumptions": {},
        "summary": _rich_summary(
            f"GLM ({family.upper()})",
            [
                (
                    "Design",
                    [
                        ("Response", response),
                        ("Predictors", ", ".join(predictors)),
                        ("Family", family),
                        ("N", str(result.n)),
                    ],
                ),
                ("Coefficients", coef_items),
                (
                    "Model Fit",
                    [
                        ("Deviance", f"{result.deviance:.4f}"),
                        ("AIC", f"{result.aic:.4f}"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"GLM ({family}): {len(sig_preds)} significant predictors, AIC = {result.aic:.1f}",
            "body": (
                f"Significant: <strong>{', '.join(sig_preds)}</strong>."
                if sig_preds
                else "No predictors reached significance."
            ),
            "next_steps": "Consider alternative link functions or families if deviance is large relative to df.",
            "chart_guidance": "",
        },
        "guide_observation": f"GLM ({family}): {len(sig_preds)} significant, AIC={result.aic:.1f}.",
        "diagnostics": [],
    }


# =============================================================================
# Ordinal Logistic Regression
# =============================================================================


def forge_ordinal_logistic(df, config):
    """Ordinal logistic regression via forgestat."""
    from forgestat.regression.glm import ordinal_logistic

    response = config.get("response") or config.get("var")
    predictors = list(config.get("predictors", []))

    if response in predictors:
        predictors.remove(response)
    if not predictors:
        raise ValueError("Need at least one predictor")

    data_clean = df[[response] + predictors].dropna()
    y = data_clean[response].values
    X = data_clean[predictors].apply(pd.to_numeric, errors="coerce").values

    result = ordinal_logistic(X, y, feature_names=predictors)

    coef_items = [(name, f"{val:.4f}") for name, val in result.coefficients.items()]

    return {
        "plots": [],
        "statistics": {
            "coefficients": result.coefficients,
            "thresholds": result.thresholds,
            "categories": result.categories,
            "n": result.n,
            "log_likelihood": round(result.log_likelihood, 4),
        },
        "assumptions": {},
        "summary": _rich_summary(
            "ORDINAL LOGISTIC REGRESSION",
            [
                (
                    "Design",
                    [
                        ("Response", f"{response} ({len(result.categories)} ordered levels)"),
                        ("Predictors", ", ".join(predictors)),
                        ("N", str(result.n)),
                    ],
                ),
                ("Coefficients", coef_items),
                ("Thresholds", [(f"τ{i + 1}", f"{t:.4f}") for i, t in enumerate(result.thresholds)]),
                ("Fit", [("Log-likelihood", f"{result.log_likelihood:.4f}")]),
            ],
        ),
        "narrative": {
            "verdict": f"Ordinal logistic: {len(result.categories)} response levels, {len(predictors)} predictors",
            "body": f"Positive coefficients increase probability of higher categories. Response levels: {', '.join(str(c) for c in result.categories)}.",
            "next_steps": "Check proportional odds assumption. Consider nominal logistic if assumption is violated.",
            "chart_guidance": "",
        },
        "guide_observation": f"Ordinal logistic: {len(result.categories)} levels, LL={result.log_likelihood:.1f}.",
        "diagnostics": [],
    }


# =============================================================================
# Orthogonal Regression (Deming)
# =============================================================================


def forge_orthogonal_regression(df, config):
    """Orthogonal (Deming) regression via forgestat."""
    from forgestat.regression.glm import orthogonal_regression
    from forgeviz.charts.scatter import scatter

    x_var = config.get("x_var") or config.get("var1")
    y_var = config.get("y_var") or config.get("var2")
    error_ratio = float(config.get("error_ratio", 1.0))

    if not x_var or not y_var:
        raise ValueError("Orthogonal regression requires x_var and y_var")

    x_data = pd.to_numeric(df[x_var], errors="coerce").dropna()
    y_data = pd.to_numeric(df[y_var], errors="coerce").dropna()
    common = x_data.index.intersection(y_data.index)
    x_vals = x_data.loc[common].values
    y_vals = y_data.loc[common].values

    result = orthogonal_regression(x_vals, y_vals, error_ratio=error_ratio)

    chart = scatter(
        x=x_vals.tolist(),
        y=y_vals.tolist(),
        title=f"Orthogonal Regression: {y_var} vs {x_var}",
        x_label=x_var,
        y_label=y_var,
    )
    plots = [_to_chart(chart)]

    return {
        "plots": plots,
        "statistics": {
            "slope_orthogonal": round(result.slope, 6),
            "intercept_orthogonal": round(result.intercept, 6),
            "slope_ols": round(result.slope_ols, 6),
            "intercept_ols": round(result.intercept_ols, 6),
            "error_ratio": result.error_ratio,
            "n": len(x_vals),
        },
        "assumptions": {},
        "summary": _rich_summary(
            "ORTHOGONAL (DEMING) REGRESSION",
            [
                (
                    "Design",
                    [
                        ("X", x_var),
                        ("Y", y_var),
                        ("Error ratio (σ²_x/σ²_y)", f"{error_ratio}"),
                        ("N", str(len(x_vals))),
                    ],
                ),
                (
                    "Orthogonal Fit",
                    [
                        ("Slope", f"{result.slope:.6f}"),
                        ("Intercept", f"{result.intercept:.6f}"),
                    ],
                ),
                (
                    "OLS Comparison",
                    [
                        ("Slope (OLS)", f"{result.slope_ols:.6f}"),
                        ("Intercept (OLS)", f"{result.intercept_ols:.6f}"),
                    ],
                ),
            ],
        ),
        "narrative": {
            "verdict": f"Orthogonal slope = {result.slope:.4f} (OLS: {result.slope_ols:.4f})",
            "body": (
                f"Orthogonal regression accounts for measurement error in both X and Y. "
                f"Slope difference from OLS: {abs(result.slope - result.slope_ols):.4f}. "
                + (
                    "Substantial difference — measurement error in X biases OLS."
                    if abs(result.slope - result.slope_ols) > 0.05 * abs(result.slope_ols)
                    else "OLS and orthogonal fits are similar — X measurement error is minor."
                )
            ),
            "next_steps": "Use orthogonal regression when both variables have measurement error (e.g., method comparison studies).",
            "chart_guidance": "Scatter plot shows the data. Orthogonal regression minimizes perpendicular distances, not vertical.",
        },
        "guide_observation": f"Orthogonal regression: slope={result.slope:.4f} vs OLS={result.slope_ols:.4f}.",
        "diagnostics": [],
    }


# =============================================================================
# Nominal Logistic Regression
# =============================================================================


def forge_nominal_logistic(df, config):
    """Nominal (multinomial) logistic regression via sklearn."""
    from forgeviz.charts.generic import bar
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import LabelEncoder

    response = config.get("response") or config.get("var")
    predictors = list(config.get("predictors", []))

    if response in predictors:
        predictors.remove(response)
    if not predictors:
        raise ValueError("Need at least one predictor")

    data_clean = df[[response] + predictors].dropna()
    y_raw = data_clean[response]
    classes = sorted(y_raw.unique().tolist(), key=str)

    if len(classes) < 2:
        raise ValueError(f"Response '{response}' has only {len(classes)} unique value(s). Need at least 2.")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()

    X = data_clean[predictors].copy()
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype) == "category":
            X = pd.get_dummies(X, columns=[col], drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").values

    model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = float((y_pred == y).mean())
    cm = confusion_matrix(y, y_pred)

    # Chart: accuracy by class
    per_class_acc = []
    for i, cn in enumerate(class_names):
        if cm[i].sum() > 0:
            per_class_acc.append(round(float(cm[i, i] / cm[i].sum()), 4))
        else:
            per_class_acc.append(0.0)

    chart = bar(
        categories=[str(c) for c in class_names],
        values=per_class_acc,
        title="Per-Class Accuracy",
        y_label="Accuracy",
    )
    plots = [_to_chart(chart)]

    return {
        "plots": plots,
        "statistics": {
            "accuracy": round(accuracy, 4),
            "classes": [str(c) for c in class_names],
            "n_classes": len(class_names),
            "per_class_accuracy": {str(c): a for c, a in zip(class_names, per_class_acc)},
            "confusion_matrix": cm.tolist(),
            "n": len(y),
        },
        "assumptions": {},
        "summary": _rich_summary(
            "NOMINAL LOGISTIC REGRESSION",
            [
                (
                    "Design",
                    [
                        ("Response", f"{response} ({len(class_names)} categories)"),
                        ("Predictors", ", ".join(predictors)),
                        ("N", str(len(y))),
                    ],
                ),
                (
                    "Overall",
                    [
                        ("Accuracy", f"{accuracy:.1%}"),
                    ],
                ),
                ("Per-class Accuracy", [(str(c), f"{a:.1%}") for c, a in zip(class_names, per_class_acc)]),
            ],
        ),
        "narrative": {
            "verdict": f"Nominal logistic: {accuracy:.1%} overall accuracy across {len(class_names)} categories",
            "body": (
                f"Multinomial model classifying {response} into {len(class_names)} categories. "
                f"Weakest class: {class_names[per_class_acc.index(min(per_class_acc))]} ({min(per_class_acc):.1%})."
            ),
            "next_steps": "Check confusion matrix for systematic misclassifications. Consider ordinal logistic if categories are ordered.",
            "chart_guidance": "Bar chart shows per-class classification accuracy.",
        },
        "guide_observation": f"Nominal logistic: {accuracy:.1%} accuracy, {len(class_names)} classes.",
        "diagnostics": [],
    }


FORGE_REGRESSION_HANDLERS = {
    "regression": forge_regression,
    "logistic": forge_logistic,
    "stepwise": forge_stepwise,
    "robust_regression": forge_robust_regression,
    "poisson_regression": forge_poisson_regression,
    "nonlinear_regression": forge_nonlinear_regression,
    "best_subsets": forge_best_subsets,
    "glm": forge_glm,
    "ordinal_logistic": forge_ordinal_logistic,
    "orthogonal_regression": forge_orthogonal_regression,
    "nominal_logistic": forge_nominal_logistic,
}

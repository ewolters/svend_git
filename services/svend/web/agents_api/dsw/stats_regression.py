"""DSW Statistical Analysis — regression models (OLS, logistic, GLM, etc.)."""

import logging

import numpy as np
from scipy import stats as sp_stats

from .common import (
    _check_normality,
    _effect_magnitude,
    _narrative,
    _practical_block,
)

logger = logging.getLogger(__name__)


def _run_regression(analysis_id, df, config):
    """Run regression analysis."""
    import pandas as pd
    from scipy import stats

    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "regression":
        response = config.get("response")
        predictors = config.get("predictors", [])
        degree = int(config.get("degree", 1))
        interactions = config.get("interactions", "none")

        # Exclude specified observations (for click-to-exclude delta comparison)
        exclude_indices = config.get("exclude_indices", [])
        if exclude_indices:
            df = df.drop(
                index=[i for i in exclude_indices if i in df.index]
            ).reset_index(drop=True)

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.preprocessing import PolynomialFeatures

        X_raw = df[predictors].dropna()
        y = df[response].loc[X_raw.index]
        n = len(y)

        # Build feature names and transform data
        feature_names = list(predictors)

        if degree > 1 or interactions == "all":
            # Use PolynomialFeatures for polynomial and/or interaction terms
            include_interaction = interactions == "all"
            poly = PolynomialFeatures(
                degree=degree,
                include_bias=False,
                interaction_only=(degree == 1 and include_interaction),
            )
            X = poly.fit_transform(X_raw)

            # Generate readable feature names
            feature_names = []
            for powers in poly.powers_:
                parts = []
                for i, power in enumerate(powers):
                    if power == 0:
                        continue
                    elif power == 1:
                        parts.append(predictors[i])
                    else:
                        parts.append(f"{predictors[i]}^{power}")
                if parts:
                    feature_names.append(
                        "·".join(parts) if len(parts) > 1 else parts[0]
                    )
        else:
            X = X_raw.values

        p = X.shape[1]  # Number of features after transformation

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Calculate comprehensive statistics
        residuals = y.values - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y.values - np.mean(y.values)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mse = ss_res / (n - p - 1) if n > p + 1 else 1e-10
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Calculate standard errors and t-statistics
        X_with_const = np.column_stack([np.ones(n), X])
        coefs = np.concatenate([[model.intercept_], model.coef_])
        _collinear_warning = False
        try:
            var_coef = mse * np.linalg.inv(X_with_const.T @ X_with_const).diagonal()
            var_coef = np.maximum(var_coef, 0)
            se = np.sqrt(var_coef)
            t_stats = coefs / np.where(se > 0, se, 1e-10)
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
        except np.linalg.LinAlgError:
            # Collinear predictors — pseudoinverse fallback
            var_coef = mse * np.linalg.pinv(X_with_const.T @ X_with_const).diagonal()
            var_coef = np.maximum(var_coef, 0)
            se = np.sqrt(var_coef)
            t_stats = coefs / np.where(se > 0, se, 1e-10)
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
            _collinear_warning = True

        # F-statistic
        f_stat = (ss_tot - ss_res) / p / mse if p > 0 else 0
        f_pvalue = 1 - stats.f.cdf(f_stat, p, n - p - 1) if p > 0 else 1

        # Durbin-Watson statistic
        dw = np.sum(np.diff(residuals) ** 2) / ss_res

        # Build colored summary output
        model_type = (
            "Linear" if degree == 1 else "Quadratic" if degree == 2 else "Cubic"
        )
        if interactions == "all":
            model_type += " + Interactions"

        summary = "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n"
        summary += f"<<COLOR:accent>>                          {model_type.upper()} REGRESSION RESULTS<</COLOR>>\n"
        summary += "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n\n"

        summary += f"<<COLOR:dim>>Dep. Variable:<</COLOR>>    <<COLOR:text>>{response}<</COLOR>>\n"
        summary += (
            f"<<COLOR:dim>>No. Observations:<</COLOR>> <<COLOR:text>>{n}<</COLOR>>\n"
        )
        summary += f"<<COLOR:dim>>No. Features:<</COLOR>>     <<COLOR:text>>{p}<</COLOR>> (from {len(predictors)} predictors)\n"
        summary += f"<<COLOR:dim>>Model:<</COLOR>>            <<COLOR:text>>OLS - {model_type}<</COLOR>>\n\n"

        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += (
            "<<COLOR:accent>>                               MODEL FIT<</COLOR>>\n"
        )
        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"

        r2_color = "success" if r2 > 0.7 else "warning" if r2 > 0.4 else "danger"
        summary += f"<<COLOR:dim>>Residual std. error:<</COLOR>> <<COLOR:text>>{np.sqrt(mse):.4f}<</COLOR>> on <<COLOR:text>>{n - p - 1}<</COLOR>> degrees of freedom\n"
        summary += f"<<COLOR:dim>>Multiple R-squared:<</COLOR>>  <<COLOR:{r2_color}>>{r2:.4f}<</COLOR>>\n"
        summary += f"<<COLOR:dim>>Adjusted R-squared:<</COLOR>> <<COLOR:{r2_color}>>{adj_r2:.4f}<</COLOR>>\n"

        f_color = (
            "success" if f_pvalue < 0.05 else "warning" if f_pvalue < 0.1 else "danger"
        )
        summary += f"<<COLOR:dim>>F-statistic:<</COLOR>>        <<COLOR:text>>{f_stat:.2f}<</COLOR>> on <<COLOR:text>>{p}<</COLOR>> and <<COLOR:text>>{n - p - 1}<</COLOR>> DF\n"
        summary += f"<<COLOR:dim>>p-value:<</COLOR>>            <<COLOR:{f_color}>>{f_pvalue:.4e}<</COLOR>>\n"
        summary += f"<<COLOR:dim>>Durbin-Watson:<</COLOR>>      <<COLOR:text>>{dw:.3f}<</COLOR>>\n\n"

        summary += "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n"
        summary += (
            "<<COLOR:accent>>                              COEFFICIENTS<</COLOR>>\n"
        )
        summary += "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n"
        _t_crit_reg = stats.t.ppf(0.975, n - p - 1)
        summary += "<<COLOR:dim>>                            Estimate    Std.Err    t value    Pr(>|t|)          [95% CI]<</COLOR>>\n"

        names = ["(Intercept)"] + feature_names
        non_sig_predictors = []
        for i, name in enumerate(names):
            pv = p_values[i]
            sig = (
                "***"
                if pv < 0.001
                else (
                    "** "
                    if pv < 0.01
                    else "*  " if pv < 0.05 else ".  " if pv < 0.1 else "   "
                )
            )
            p_color = "success" if pv < 0.05 else "warning" if pv < 0.1 else "dim"
            _ci_lo = coefs[i] - _t_crit_reg * se[i]
            _ci_hi = coefs[i] + _t_crit_reg * se[i]
            summary += f"<<COLOR:text>>{name:<24}<</COLOR>> {coefs[i]:>10.4f}   {se[i]:>9.4f}   {t_stats[i]:>8.3f}    <<COLOR:{p_color}>>{pv:>9.4f}  {sig}<</COLOR>>  [{_ci_lo:.4f}, {_ci_hi:.4f}]\n"
            if (
                i > 0 and pv >= 0.1
            ):  # Track non-significant predictors (excluding intercept)
                non_sig_predictors.append(name)

        summary += "<<COLOR:dim>>---<</COLOR>>\n"
        summary += "<<COLOR:dim>>Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1<</COLOR>>\n"
        if _collinear_warning:
            summary += "\n<<COLOR:danger>>⚠ WARNING: Near-collinear predictors detected. Standard errors computed via pseudoinverse — interpret p-values with caution. Consider removing redundant predictors.<</COLOR>>\n"
        summary += "\n"

        # Diagnostics interpretation
        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += "<<COLOR:accent>>                            DIAGNOSTICS SUMMARY<</COLOR>>\n"
        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"

        # Interpret R²
        if r2 > 0.7:
            summary += f"<<COLOR:success>>✓ Good fit:<</COLOR>> Model explains {r2 * 100:.1f}% of variance\n"
        elif r2 > 0.4:
            summary += f"<<COLOR:warning>>◐ Moderate fit:<</COLOR>> Model explains {r2 * 100:.1f}% of variance\n"
        else:
            summary += f"<<COLOR:danger>>✗ Poor fit:<</COLOR>> Model explains only {r2 * 100:.1f}% of variance\n"

        # Interpret Durbin-Watson
        if 1.5 < dw < 2.5:
            summary += (
                "<<COLOR:success>>✓ No autocorrelation:<</COLOR>> Durbin-Watson ≈ 2\n"
            )
        else:
            summary += f"<<COLOR:warning>>◐ Possible autocorrelation:<</COLOR>> Durbin-Watson = {dw:.2f}\n"

        # Interpret F-test
        if f_pvalue < 0.05:
            summary += (
                "<<COLOR:success>>✓ Model significant:<</COLOR>> F-test p < 0.05\n"
            )
        else:
            summary += f"<<COLOR:danger>>✗ Model not significant:<</COLOR>> F-test p = {f_pvalue:.3f}\n"

        # Calculate diagnostic values first (needed for suggestions and plots)
        std_residuals = residuals / np.std(residuals)

        # Leverage (hat values)
        try:
            H = (
                X_with_const
                @ np.linalg.inv(X_with_const.T @ X_with_const)
                @ X_with_const.T
            )
            leverage = np.diag(H)
        except np.linalg.LinAlgError:
            H = (
                X_with_const
                @ np.linalg.pinv(X_with_const.T @ X_with_const)
                @ X_with_const.T
            )
            leverage = np.diag(H)

        # Cook's distance
        cooks_d = (std_residuals**2 / (p + 1)) * (
            leverage / (1 - leverage + 1e-10) ** 2
        )

        # Square root of standardized residuals for scale-location
        sqrt_std_resid = np.sqrt(np.abs(std_residuals))

        # Model improvement suggestions
        suggestions = []

        # Low R² suggestions
        if r2 < 0.4:
            suggestions.append("Add more predictors or interaction terms (X1*X2)")
            suggestions.append(
                "Try polynomial terms (X², X³) for non-linear relationships"
            )
            suggestions.append("Check for outliers that may be distorting the fit")
        elif r2 < 0.7:
            suggestions.append(
                "Consider adding interaction terms or polynomial features"
            )

        # Non-significant predictors
        if non_sig_predictors:
            if len(non_sig_predictors) <= 3:
                suggestions.append(
                    f"Consider removing non-significant: {', '.join(non_sig_predictors)}"
                )
            else:
                suggestions.append(
                    f"Consider removing {len(non_sig_predictors)} non-significant predictors"
                )

        # Autocorrelation
        if dw < 1.5:
            suggestions.append(
                "Positive autocorrelation detected - consider time series methods or lag terms"
            )
        elif dw > 2.5:
            suggestions.append(
                "Negative autocorrelation detected - check data ordering"
            )

        # Model not significant
        if f_pvalue >= 0.05:
            suggestions.append(
                "Model not significant - try different predictors or check data quality"
            )

        # High leverage points
        high_leverage = int(np.sum(leverage > 2 * (p + 1) / n)) if n > 0 else 0
        if high_leverage > 0:
            suggestions.append(
                f"{high_leverage} high-leverage points detected - check for influential outliers"
            )

        # Large Cook's distance
        high_cooks = int(np.sum(cooks_d > 4 / n)) if n > 0 else 0
        if high_cooks > 0:
            suggestions.append(
                f"{high_cooks} influential observations (Cook's D) - consider robust regression"
            )

        if suggestions:
            summary += "\n<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            summary += "<<COLOR:accent>>                          IMPROVEMENT SUGGESTIONS<</COLOR>>\n"
            summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            for i, sug in enumerate(suggestions[:5], 1):  # Limit to 5 suggestions
                summary += (
                    f"<<COLOR:warning>>{i}.<</COLOR>> <<COLOR:text>>{sug}<</COLOR>>\n"
                )

        # Practical significance: R² as effect size
        r2_label, r2_meaningful = _effect_magnitude(r2, "r_squared")
        summary += _practical_block(
            "R²",
            r2,
            "r_squared",
            f_pvalue,
            context=f"The model explains {r2 * 100:.1f}% of the variation in '{response}'. "
            f"RMSE = {rmse:.4f} — on average, predictions are off by this amount in the original units.",
        )

        result["summary"] = summary
        # Build guide_observation for regression
        sig_predictors = [names[i] for i in range(1, len(names)) if p_values[i] < 0.05]
        obs = f"Regression: R²={r2:.3f} ({r2_label}), F p={f_pvalue:.4f}."
        if sig_predictors:
            obs += f" Significant predictors: {', '.join(sig_predictors[:5])}."
        if r2_meaningful:
            obs += f" Model explains {r2 * 100:.0f}% of variation — practically useful."
        else:
            obs += f" Model explains only {r2 * 100:.0f}% of variation — limited practical use."
        result["guide_observation"] = obs

        # Narrative (enhanced — coefficient interpretation, R² warning, VIF)
        _narr_coef_parts = []
        for _si in sig_predictors[:3]:
            _idx = names.index(_si) if _si in names else -1
            if _idx > 0:
                _c = coefs[_idx]
                _dir = "increases" if _c > 0 else "decreases"
                _narr_coef_parts.append(
                    f"<strong>{_si}</strong> ({_dir} {response} by {abs(_c):.4f} per unit)"
                )
        _r2_warn = ""
        if r2 > 0.95 and p > 2:
            _r2_warn = " R&sup2; > 0.95 with multiple predictors may indicate overfitting &mdash; validate on held-out data."
        _vif_warn = ""
        if p > 2 and n > p:
            try:
                _X_for_vif = X.values if hasattr(X, "values") else X
                _corr = np.corrcoef(_X_for_vif, rowvar=False)
                _high_vifs = []
                for _vi in range(min(_corr.shape[0], len(names) - 1)):
                    _minor = np.delete(np.delete(_corr, _vi, 0), _vi, 1)
                    _det = np.linalg.det(_minor) if _minor.size > 0 else 1
                    _det_full = np.linalg.det(_corr) if _corr.size > 0 else 1
                    if _det_full > 1e-10:
                        _vif_val = _det / _det_full if _det > 0 else 0
                    else:
                        _vif_val = 99
                    if _vif_val > 5:
                        _high_vifs.append(
                            names[_vi + 1] if _vi + 1 < len(names) else f"X{_vi}"
                        )
                if _high_vifs:
                    _vif_warn = f" Possible multicollinearity in: {', '.join(_high_vifs[:3])} (VIF > 5)."
            except Exception:
                pass

        if r2_meaningful and sig_predictors:
            verdict = f"Model explains {r2 * 100:.0f}% of the variation in {response}"
            body = f"Significant predictors: {', '.join(_narr_coef_parts) if _narr_coef_parts else ', '.join(sig_predictors[:3])}. R&sup2; = {r2:.3f}, RMSE = {rmse:.4f}.{_r2_warn}{_vif_warn}"
            nexts = "Use the What-If Explorer below to see how changing predictor values affects the predicted response."
        elif sig_predictors:
            verdict = "Some predictors are significant but the model is weak"
            body = f"The model explains only {r2 * 100:.0f}% of the variation (R&sup2; = {r2:.3f}). Significant predictors: {', '.join(_narr_coef_parts) if _narr_coef_parts else ', '.join(sig_predictors[:3])}. Consider adding more predictors or using a nonlinear model.{_vif_warn}"
            nexts = "Try polynomial regression or add interaction terms. Check residual plots for patterns."
        else:
            verdict = f"Model does not explain {response} well"
            body = f"No predictors are statistically significant. R&sup2; = {r2:.3f} &mdash; the model captures very little of the variation."
            nexts = "Review variable selection. The current predictors may not be the right drivers."
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps=nexts,
            chart_guidance="The 4 diagnostic plots: (1) Residuals vs Fitted &mdash; random scatter means adequate fit, patterns mean missing terms. (2) Q-Q plot &mdash; points on the line mean normal residuals. (3) Scale-Location &mdash; flat trend means constant variance. (4) Residuals vs Leverage &mdash; points in the upper-right corner are influential outliers.",
        )

        # What-If data for client-side interactive predictor explorer (degree 1 only)
        if degree == 1 and interactions == "none":
            result["what_if_data"] = {
                "type": "regression",
                "intercept": float(model.intercept_),
                "coefficients": {
                    feat: float(c) for feat, c in zip(predictors, model.coef_)
                },
                "residual_std": float(np.sqrt(mse)) if mse > 0 else 0.0,
                "n": int(n),
                "feature_ranges": {
                    feat: {
                        "min": float(X_raw[feat].min()),
                        "max": float(X_raw[feat].max()),
                        "mean": float(X_raw[feat].mean()),
                        "std": float(X_raw[feat].std()),
                    }
                    for feat in predictors
                },
                "response_name": response,
            }

        # ── Diagnostics ──
        diagnostics = []
        # Residual normality
        _norm_resid = _check_normality(residuals, label="Residuals", alpha=0.05)
        if _norm_resid:
            _norm_resid["detail"] = (
                "Non-normal residuals may indicate model misspecification or outlier influence. CIs and p-values may be unreliable."
            )
            diagnostics.append(_norm_resid)
        # Multicollinearity
        if _collinear_warning:
            diagnostics.append(
                {
                    "level": "error",
                    "title": "Near-collinear predictors detected",
                    "detail": "Standard errors are unreliable. Consider removing redundant predictors or using Ridge/Lasso regression.",
                }
            )
        elif _vif_warn:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": "Possible multicollinearity",
                    "detail": _vif_warn.strip(),
                }
            )
        # Autocorrelation
        if dw < 1.5 or dw > 2.5:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Autocorrelation (Durbin-Watson = {dw:.2f})",
                    "detail": "Residuals are not independent. Consider time series methods or adding lag terms.",
                }
            )
        # Influential points
        if high_cooks > 0:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"{high_cooks} influential observations (Cook's D > {4 / n:.3f})",
                    "detail": "These points disproportionately affect the model. Investigate whether they are errors or genuine extreme cases.",
                }
            )
        # Effect size emphasis
        if r2 >= 0.7 and f_pvalue < 0.05:
            diagnostics.append(
                {
                    "level": "info",
                    "title": f"Strong model (R\u00b2 = {r2:.3f})",
                    "detail": f"The model explains {r2 * 100:.0f}% of the variation \u2014 practically useful for prediction.",
                }
            )
        elif r2 < 0.1 and f_pvalue < 0.05:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Significant but weak model (R\u00b2 = {r2:.3f})",
                    "detail": f"The model is statistically significant but explains only {r2 * 100:.0f}% of the variation. Not useful for prediction.",
                }
            )
        result["diagnostics"] = diagnostics

        # Regression metrics for exclude-and-compare delta display
        result["regression_metrics"] = {
            "r_squared": round(float(r2), 6),
            "adj_r_squared": round(float(adj_r2), 6),
            "f_stat": round(float(f_stat), 4),
            "rmse": round(float(rmse), 4),
        }

        # Create 4-panel diagnostic plots
        _diag_row_indices = list(range(len(y_pred)))
        _diag_cd = [[i, float(cooks_d[i])] for i in _diag_row_indices]

        # 1. Residuals vs Fitted
        result["plots"].append(
            {
                "title": "1. Residuals vs Fitted",
                "data": [
                    {
                        "type": "scatter",
                        "x": y_pred.tolist(),
                        "y": residuals.tolist(),
                        "mode": "markers",
                        "marker": {
                            "color": "rgba(74, 159, 110, 0.5)",
                            "size": 6,
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                        "name": "Residuals",
                        "customdata": _diag_cd,
                        "hovertemplate": "Fitted: %{x:.4f}<br>Residual: %{y:.4f}<br>Obs #%{customdata[0]}<br>Cook's D: %{customdata[1]:.4f}<extra></extra>",
                    },
                    {
                        "type": "scatter",
                        "x": [min(y_pred), max(y_pred)],
                        "y": [0, 0],
                        "mode": "lines",
                        "line": {"color": "#9f4a4a", "dash": "dash"},
                        "name": "Zero line",
                    },
                ],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": "Fitted values"},
                    "yaxis": {"title": "Residuals"},
                },
                "interactive": {"type": "regression_diagnostic"},
            }
        )

        # 2. Normal Q-Q Plot
        sorted_std_resid = np.sort(std_residuals)
        _qq_order = np.argsort(std_residuals)
        _qq_cd = [
            [int(_qq_order[i]), float(cooks_d[_qq_order[i]])]
            for i in range(len(_qq_order))
        ]
        theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_std_resid)))
        result["plots"].append(
            {
                "title": "2. Normal Q-Q",
                "data": [
                    {
                        "type": "scatter",
                        "x": theoretical_q.tolist(),
                        "y": sorted_std_resid.tolist(),
                        "mode": "markers",
                        "marker": {
                            "color": "rgba(74, 159, 110, 0.5)",
                            "size": 6,
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                        "name": "Residuals",
                        "customdata": _qq_cd,
                        "hovertemplate": "Theoretical: %{x:.3f}<br>Std. Resid: %{y:.3f}<br>Obs #%{customdata[0]}<br>Cook's D: %{customdata[1]:.4f}<extra></extra>",
                    },
                    {
                        "type": "scatter",
                        "x": [-3, 3],
                        "y": [-3, 3],
                        "mode": "lines",
                        "line": {"color": "#9f4a4a", "dash": "dash"},
                        "name": "Normal line",
                    },
                ],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": "Theoretical Quantiles"},
                    "yaxis": {"title": "Std. Residuals"},
                },
                "interactive": {"type": "regression_diagnostic"},
            }
        )

        # 3. Scale-Location
        result["plots"].append(
            {
                "title": "3. Scale-Location",
                "data": [
                    {
                        "type": "scatter",
                        "x": y_pred.tolist(),
                        "y": sqrt_std_resid.tolist(),
                        "mode": "markers",
                        "marker": {
                            "color": "rgba(74, 159, 110, 0.5)",
                            "size": 6,
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                        "name": "\u221a|Std. Residuals|",
                        "customdata": _diag_cd,
                        "hovertemplate": "Fitted: %{x:.4f}<br>\u221a|Std. Resid|: %{y:.4f}<br>Obs #%{customdata[0]}<br>Cook's D: %{customdata[1]:.4f}<extra></extra>",
                    }
                ],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": "Fitted values"},
                    "yaxis": {"title": "\u221a|Standardized residuals|"},
                },
                "interactive": {"type": "regression_diagnostic"},
            }
        )

        # 4. Residuals vs Leverage
        result["plots"].append(
            {
                "title": "4. Residuals vs Leverage",
                "data": [
                    {
                        "type": "scatter",
                        "x": leverage.tolist(),
                        "y": std_residuals.tolist(),
                        "mode": "markers",
                        "marker": {
                            "color": cooks_d.tolist(),
                            "colorscale": [
                                [0, "#4a9f6e"],
                                [0.5, "#e89547"],
                                [1, "#9f4a4a"],
                            ],
                            "size": 6,
                            "colorbar": {"title": "Cook's D", "len": 0.5},
                        },
                        "name": "Observations",
                        "customdata": _diag_cd,
                        "hovertemplate": "Leverage: %{x:.4f}<br>Std. Resid: %{y:.4f}<br>Obs #%{customdata[0]}<br>Cook's D: %{customdata[1]:.4f}<extra></extra>",
                    },
                    {
                        "type": "scatter",
                        "x": [0, max(leverage)],
                        "y": [0, 0],
                        "mode": "lines",
                        "line": {"color": "#9f4a4a", "dash": "dash"},
                        "showlegend": False,
                    },
                ],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": "Leverage"},
                    "yaxis": {"title": "Std. Residuals"},
                },
                "interactive": {"type": "regression_diagnostic"},
            }
        )

        # Explicit statistics for Synara integration
        result["statistics"] = {
            "R²": float(r2),
            "Adj_R²": float(adj_r2),
            "RMSE": float(rmse),
            "F_statistic": float(f_stat),
            "F_p_value": float(f_pvalue),
            "n": int(n),
            "predictors": int(p),
        }
        # Add coefficients as statistics
        for i, feat in enumerate(feature_names):
            result["statistics"][f"coef({feat})"] = float(model.coef_[i])
            if i < len(p_values) - 1:
                result["statistics"][f"p_value({feat})"] = float(p_values[i + 1])

    elif analysis_id == "nominal_logistic":
        """
        Nominal Logistic Regression — for multi-class categorical outcomes.
        Uses sklearn's multinomial logistic regression.
        Auto-excludes response from predictors if accidentally included.
        If response has only 2 levels, suggests binary logistic instead.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix
        from sklearn.preprocessing import LabelEncoder

        response = config.get("response")
        predictors = list(config.get("predictors", []))

        # Convenience: auto-exclude response from predictors
        if response in predictors:
            predictors.remove(response)

        if not predictors:
            result["summary"] = "Please select at least one predictor variable."
            return result

        data_clean = df[[response] + predictors].dropna()
        y_raw = data_clean[response]
        classes = sorted(y_raw.unique().tolist(), key=str)

        if len(classes) < 2:
            result["summary"] = (
                f"Response '{response}' has only {len(classes)} unique value(s). Need at least 2."
            )
            return result
        if len(classes) == 2:
            result["summary"] = (
                f"<<COLOR:warning>>Response '{response}' has only 2 levels. Consider using binary logistic regression ('logistic') instead for simpler interpretation.<</COLOR>>\n\nProceeding with nominal logistic..."
            )

        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        class_names = le.classes_.tolist()
        ref_class = class_names[0]

        X = data_clean[predictors]
        # Encode categorical predictors
        for col in X.columns:
            if X[col].dtype == "object" or str(X[col].dtype) == "category":
                X = pd.get_dummies(X, columns=[col], drop_first=True)

        model = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=1000
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        accuracy = float((y_pred == y).mean())

        cm = confusion_matrix(y, y_pred)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>NOMINAL LOGISTIC REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response} ({len(classes)} categories)\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        summary += f"<<COLOR:highlight>>Reference category:<</COLOR>> {ref_class}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {len(data_clean)}\n"
        summary += f"<<COLOR:highlight>>Accuracy:<</COLOR>> {accuracy:.1%}\n\n"

        # Coefficients per class (vs reference)
        pred_names = list(X.columns)
        summary += f"<<COLOR:accent>>── Coefficients (vs reference '{ref_class}') ──<</COLOR>>\n"
        summary += f"  {'Predictor':<25}"
        for cls in class_names[1:]:
            summary += f" {str(cls):>12}"
        summary += "\n"
        summary += f"  {'─' * (25 + 13 * (len(class_names) - 1))}\n"

        for j, pred in enumerate(pred_names):
            summary += f"  {pred:<25}"
            for i in range(1, len(class_names)):
                coef = model.coef_[i][j] if i < len(model.coef_) else 0
                summary += f" {coef:>12.4f}"
            summary += "\n"

        # Odds ratios with CIs (approximate SEs via Fisher information)
        _nom_se = {}
        _nom_se_warning = False
        try:
            _probs = model.predict_proba(X)
            _Xv = X.values
            for _ki in range(1, len(class_names)):
                _wk = _probs[:, _ki] * (1 - _probs[:, _ki])
                _info = _Xv.T @ (_wk[:, None] * _Xv)
                try:
                    _nom_se[_ki] = np.sqrt(np.maximum(np.diag(np.linalg.inv(_info)), 0))
                except np.linalg.LinAlgError:
                    _info_reg = _info + 1e-6 * np.eye(_info.shape[0])
                    _nom_se[_ki] = np.sqrt(
                        np.maximum(np.diag(np.linalg.inv(_info_reg)), 0)
                    )
                    _nom_se_warning = True
        except Exception:
            _nom_se_warning = True

        summary += "\n<<COLOR:accent>>── Odds Ratios (exp(coef)) ──<</COLOR>>\n"
        if _nom_se_warning:
            summary += "<<COLOR:danger>>⚠ Near-singular Fisher information for some classes — SEs are approximate<</COLOR>>\n"
        _has_ci = len(_nom_se) > 0
        summary += f"  {'Predictor':<25}"
        for cls in class_names[1:]:
            summary += f" {str(cls):>12}"
            if _has_ci:
                summary += f" {'95% CI':>22}"
        summary += "\n"
        _col_w = (13 + (23 if _has_ci else 0)) * (len(class_names) - 1)
        summary += f"  {'─' * (25 + _col_w)}\n"
        for j, pred in enumerate(pred_names):
            summary += f"  {pred:<25}"
            for i in range(1, len(class_names)):
                coef = model.coef_[i][j] if i < len(model.coef_) else 0
                summary += f" {np.exp(coef):>12.4f}"
                if _has_ci and i in _nom_se and j < len(_nom_se[i]):
                    _se_j = _nom_se[i][j]
                    summary += f" [{np.exp(coef - 1.96 * _se_j):>8.4f}, {np.exp(coef + 1.96 * _se_j):>8.4f}]"
                elif _has_ci:
                    summary += f" {'':>22}"
            summary += "\n"

        # Confusion matrix
        summary += "\n<<COLOR:accent>>── Confusion Matrix ──<</COLOR>>\n"
        _cm_header = "Actual \\ Pred"
        summary += f"  {_cm_header:<15}"
        for cls in class_names:
            summary += f" {str(cls):>8}"
        summary += "\n"
        for i, cls in enumerate(class_names):
            summary += f"  {str(cls):<15}"
            for j in range(len(class_names)):
                summary += f" {cm[i, j]:>8}"
            summary += "\n"

        result["summary"] = summary

        # Predicted probability heatmap
        probs = model.predict_proba(X)
        avg_probs = []
        for i, cls in enumerate(class_names):
            avg_probs.append(float(probs[:, i].mean()))
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "bar",
                        "x": [str(c) for c in class_names],
                        "y": avg_probs,
                        "marker": {
                            "color": [
                                "#4a9f6e",
                                "#4a90d9",
                                "#e8c547",
                                "#c75a3a",
                                "#7a5fb8",
                                "#5a9fd4",
                                "#d4a05a",
                                "#5ad4a0",
                            ][: len(class_names)]
                        },
                        "text": [f"{p:.3f}" for p in avg_probs],
                        "textposition": "outside",
                    }
                ],
                "layout": {
                    "title": "Average Predicted Probability by Class",
                    "yaxis": {
                        "title": "Avg Probability",
                        "range": [0, max(avg_probs) * 1.2 + 0.05],
                    },
                    "height": 280,
                },
            }
        )

        # Coefficient plot (grouped bar)
        if len(class_names) > 2:
            bar_traces = []
            colors = ["#4a90d9", "#e8c547", "#c75a3a", "#7a5fb8", "#5a9fd4"]
            for i in range(1, len(class_names)):
                coefs_i = [
                    (
                        float(model.coef_[i][j])
                        if i < len(model.coef_) and j < len(model.coef_[i])
                        else 0
                    )
                    for j in range(len(pred_names))
                ]
                bar_traces.append(
                    {
                        "type": "bar",
                        "x": pred_names,
                        "y": coefs_i,
                        "name": f"vs {ref_class} → {class_names[i]}",
                        "marker": {"color": colors[(i - 1) % len(colors)]},
                    }
                )
            result["plots"].append(
                {
                    "data": bar_traces,
                    "layout": {
                        "title": "Coefficients by Category",
                        "barmode": "group",
                        "yaxis": {"title": "Coefficient"},
                        "height": 300,
                    },
                }
            )

        result["guide_observation"] = (
            f"Nominal logistic: {len(classes)} categories, accuracy={accuracy:.1%}."
        )
        result["narrative"] = _narrative(
            f"Nominal Logistic: {accuracy:.1%} accuracy across {len(classes)} categories",
            f"Multinomial logistic regression for <strong>{response}</strong> with {len(predictors)} predictor{'s' if len(predictors) > 1 else ''}.",
            next_steps="Check per-class accuracy in the classification report. Imbalanced classes may inflate overall accuracy.",
        )
        result["statistics"] = {
            "n": len(data_clean),
            "n_classes": len(classes),
            "classes": [str(c) for c in class_names],
            "accuracy": accuracy,
            "reference_class": str(ref_class),
        }

    elif analysis_id == "orthogonal_regression":
        """
        Orthogonal / Deming Regression — minimizes perpendicular distance to line.
        Used when both X and Y have measurement error (method comparison studies).
        Error ratio delta = var(eps_x) / var(eps_y), default=1 (equal errors).
        Includes Bland-Altman plot for method agreement assessment.
        """
        var_x = config.get("var1") or config.get("var_x")
        var_y = config.get("var2") or config.get("var_y")
        delta = float(config.get("error_ratio", 1.0))
        alpha = 1 - float(config.get("conf", 95)) / 100

        data_clean = df[[var_x, var_y]].dropna()
        x = data_clean[var_x].values.astype(float)
        y = data_clean[var_y].values.astype(float)
        n = len(x)

        if n < 3:
            result["summary"] = "Need at least 3 observations for regression."
            return result

        x_bar = float(np.mean(x))
        y_bar = float(np.mean(y))
        sxx = float(np.sum((x - x_bar) ** 2) / (n - 1))
        syy = float(np.sum((y - y_bar) ** 2) / (n - 1))
        sxy = float(np.sum((x - x_bar) * (y - y_bar)) / (n - 1))

        # Deming slope
        discriminant = (syy - delta * sxx) ** 2 + 4 * delta * sxy**2
        b1_deming = (
            float((syy - delta * sxx + np.sqrt(discriminant)) / (2 * sxy))
            if sxy != 0
            else 1.0
        )
        b0_deming = float(y_bar - b1_deming * x_bar)

        # OLS for comparison
        b1_ols = float(sxy / sxx) if sxx > 0 else 0.0
        b0_ols = float(y_bar - b1_ols * x_bar)

        # Residuals and R-squared
        y_pred_deming = b0_deming + b1_deming * x
        ss_total = float(np.sum((y - y_bar) ** 2))
        ss_resid = float(np.sum((y - y_pred_deming) ** 2))
        r_squared = 1 - ss_resid / ss_total if ss_total > 0 else 0

        # Bootstrap CI for Deming parameters
        rng = np.random.RandomState(42)
        boot_slopes, boot_intercepts = [], []
        for _ in range(1000):
            idx = rng.choice(n, n, replace=True)
            xb, yb = x[idx], y[idx]
            xb_bar, yb_bar = float(np.mean(xb)), float(np.mean(yb))
            sxxb = float(np.sum((xb - xb_bar) ** 2) / (n - 1))
            syyb = float(np.sum((yb - yb_bar) ** 2) / (n - 1))
            sxyb = float(np.sum((xb - xb_bar) * (yb - yb_bar)) / (n - 1))
            if sxyb != 0:
                disc = (syyb - delta * sxxb) ** 2 + 4 * delta * sxyb**2
                b1b = (syyb - delta * sxxb + np.sqrt(disc)) / (2 * sxyb)
                boot_slopes.append(b1b)
                boot_intercepts.append(yb_bar - b1b * xb_bar)

        ci_slope = (
            float(np.percentile(boot_slopes, 100 * alpha / 2)),
            float(np.percentile(boot_slopes, 100 * (1 - alpha / 2))),
        )
        ci_intercept = (
            float(np.percentile(boot_intercepts, 100 * alpha / 2)),
            float(np.percentile(boot_intercepts, 100 * (1 - alpha / 2))),
        )

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>ORTHOGONAL (DEMING) REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>X:<</COLOR>> {var_x}  |  <<COLOR:highlight>>Y:<</COLOR>> {var_y}\n"
        summary += f"<<COLOR:highlight>>Error ratio (delta):<</COLOR>> {delta}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {n}\n\n"
        summary += "<<COLOR:accent>>── Deming Regression Results ──<</COLOR>>\n"
        summary += (
            f"  Slope:     {b1_deming:>10.4f}  ({ci_slope[0]:.4f}, {ci_slope[1]:.4f})\n"
        )
        summary += f"  Intercept: {b0_deming:>10.4f}  ({ci_intercept[0]:.4f}, {ci_intercept[1]:.4f})\n"
        summary += f"  R-squared: {r_squared:>10.4f}\n\n"
        summary += "<<COLOR:accent>>── OLS Comparison ──<</COLOR>>\n"
        summary += f"  Slope:     {b1_ols:>10.4f}\n"
        summary += f"  Intercept: {b0_ols:>10.4f}\n\n"
        summary += "<<COLOR:accent>>── Interpretation ──<</COLOR>>\n"
        if abs(b1_deming - 1.0) < 0.1 and abs(b0_deming) < (np.std(y) * 0.1):
            summary += "  <<COLOR:good>>Methods show good agreement (slope ~ 1, intercept ~ 0).<</COLOR>>\n"
        elif abs(b1_deming - 1.0) < 0.1:
            summary += f"  <<COLOR:warning>>Proportional agreement but constant bias (intercept = {b0_deming:.4f}).<</COLOR>>\n"
        else:
            summary += "  <<COLOR:warning>>Methods disagree -- both slope and intercept differ from ideal (1, 0).<</COLOR>>\n"

        result["summary"] = summary

        x_line = np.linspace(float(x.min()), float(x.max()), 100)
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "scatter",
                        "x": x.tolist(),
                        "y": y.tolist(),
                        "mode": "markers",
                        "marker": {"color": "#4a9f6e", "size": 5},
                        "name": "Data",
                    },
                    {
                        "type": "scatter",
                        "x": x_line.tolist(),
                        "y": (b0_deming + b1_deming * x_line).tolist(),
                        "mode": "lines",
                        "line": {"color": "#e89547", "width": 2},
                        "name": "Deming",
                    },
                    {
                        "type": "scatter",
                        "x": x_line.tolist(),
                        "y": (b0_ols + b1_ols * x_line).tolist(),
                        "mode": "lines",
                        "line": {"color": "#47a5e8", "dash": "dash", "width": 2},
                        "name": "OLS",
                    },
                    {
                        "type": "scatter",
                        "x": x_line.tolist(),
                        "y": x_line.tolist(),
                        "mode": "lines",
                        "line": {"color": "#888", "dash": "dot", "width": 1},
                        "name": "Identity (y=x)",
                    },
                ],
                "layout": {
                    "title": "Deming vs OLS Regression",
                    "xaxis": {"title": var_x},
                    "yaxis": {"title": var_y},
                    "height": 350,
                },
            }
        )

        # Bland-Altman plot
        mean_xy = (x + y) / 2
        diff_xy = y - x
        diff_mean = float(np.mean(diff_xy))
        diff_std = float(np.std(diff_xy, ddof=1))
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "scatter",
                        "x": mean_xy.tolist(),
                        "y": diff_xy.tolist(),
                        "mode": "markers",
                        "marker": {"color": "#4a9f6e", "size": 5},
                        "name": "Differences",
                    },
                ],
                "layout": {
                    "title": "Bland-Altman Plot",
                    "xaxis": {"title": f"Mean of {var_x} and {var_y}"},
                    "yaxis": {"title": f"{var_y} - {var_x}"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": float(mean_xy.min()),
                            "x1": float(mean_xy.max()),
                            "y0": diff_mean,
                            "y1": diff_mean,
                            "line": {"color": "#e89547", "width": 2},
                        },
                        {
                            "type": "line",
                            "x0": float(mean_xy.min()),
                            "x1": float(mean_xy.max()),
                            "y0": diff_mean + 1.96 * diff_std,
                            "y1": diff_mean + 1.96 * diff_std,
                            "line": {"color": "#d94a4a", "dash": "dash", "width": 1},
                        },
                        {
                            "type": "line",
                            "x0": float(mean_xy.min()),
                            "x1": float(mean_xy.max()),
                            "y0": diff_mean - 1.96 * diff_std,
                            "y1": diff_mean - 1.96 * diff_std,
                            "line": {"color": "#d94a4a", "dash": "dash", "width": 1},
                        },
                    ],
                    "height": 300,
                },
            }
        )

        result["guide_observation"] = (
            f"Deming regression: slope={b1_deming:.4f}, intercept={b0_deming:.4f}, R2={r_squared:.4f}."
        )
        result["narrative"] = _narrative(
            f"Deming (Orthogonal) Regression: R\u00b2 = {r_squared:.4f}",
            f"Slope = {b1_deming:.4f}, intercept = {b0_deming:.4f}. Unlike OLS, Deming regression accounts for measurement error in both X and Y.",
            next_steps="Use Deming regression for method comparison studies where both measurements have error. A slope near 1 and intercept near 0 indicates good agreement.",
        )
        result["statistics"] = {
            "deming_slope": b1_deming,
            "deming_intercept": b0_deming,
            "ols_slope": b1_ols,
            "ols_intercept": b0_ols,
            "r_squared": r_squared,
            "error_ratio": delta,
            "n": n,
            "slope_ci": list(ci_slope),
            "intercept_ci": list(ci_intercept),
            "bland_altman_bias": diff_mean,
            "bland_altman_loa": [
                diff_mean - 1.96 * diff_std,
                diff_mean + 1.96 * diff_std,
            ],
        }

    elif analysis_id == "nonlinear_regression":
        """
        Nonlinear Regression — fit preset or user-specified curve models.
        Uses scipy.optimize.curve_fit (Levenberg-Marquardt).
        Presets: exponential, power, logistic, logarithmic, polynomial2, polynomial3,
                 michaelis_menten, gompertz, hill.
        """
        var_x = config.get("var1") or config.get("var_x")
        var_y = config.get("var2") or config.get("var_y")
        model_type = config.get("model", "exponential")
        initial_params = config.get("initial_params")
        alpha = 1 - float(config.get("conf", 95)) / 100

        data_clean = df[[var_x, var_y]].dropna()
        x = data_clean[var_x].values.astype(float)
        y = data_clean[var_y].values.astype(float)
        n = len(x)

        if n < 3:
            result["summary"] = "Need at least 3 observations for curve fitting."
            return result

        from scipy.optimize import curve_fit

        models = {
            "exponential": (lambda x, a, b: a * np.exp(b * x), ["a", "b"], [1.0, 0.01]),
            "power": (
                lambda x, a, b: a * np.power(np.maximum(x, 1e-10), b),
                ["a", "b"],
                [1.0, 1.0],
            ),
            "logistic": (
                lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0))),
                ["L", "k", "x0"],
                [float(max(y)), 1.0, float(np.median(x))],
            ),
            "logarithmic": (
                lambda x, a, b: a * np.log(np.maximum(x, 1e-10)) + b,
                ["a", "b"],
                [1.0, 0.0],
            ),
            "polynomial2": (
                lambda x, a, b, c: a * x**2 + b * x + c,
                ["a", "b", "c"],
                [0.0, 1.0, 0.0],
            ),
            "polynomial3": (
                lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d,
                ["a", "b", "c", "d"],
                [0.0, 0.0, 1.0, 0.0],
            ),
            "michaelis_menten": (
                lambda x, Vmax, Km: Vmax * x / (Km + x),
                ["Vmax", "Km"],
                [float(max(y)), float(np.median(x))],
            ),
            "gompertz": (
                lambda x, a, b, c: a * np.exp(-b * np.exp(-c * x)),
                ["a", "b", "c"],
                [float(max(y)), 1.0, 0.1],
            ),
            "hill": (
                lambda x, Vmax, Kd, n_h: Vmax * x**n_h / (Kd**n_h + x**n_h),
                ["Vmax", "Kd", "n"],
                [float(max(y)), float(np.median(x)), 1.0],
            ),
        }

        if model_type not in models:
            result["summary"] = (
                f"Unknown model '{model_type}'. Available: {', '.join(models.keys())}"
            )
            return result

        func, param_names, p0_default = models[model_type]
        p0 = (
            initial_params
            if initial_params and len(initial_params) == len(param_names)
            else p0_default
        )

        try:
            popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=10000)
        except Exception as e:
            result["summary"] = (
                f"Curve fitting failed: {str(e)}. Try different initial parameters or a different model."
            )
            return result

        y_pred = func(x, *popt)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = float(np.sqrt(ss_res / n))

        perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros(len(popt))

        k_params = len(popt)
        aic = float(n * np.log(ss_res / n) + 2 * k_params) if ss_res > 0 else 0
        bic = float(n * np.log(ss_res / n) + k_params * np.log(n)) if ss_res > 0 else 0

        summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>NONLINEAR REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> {model_type}\n"
        summary += f"<<COLOR:highlight>>X:<</COLOR>> {var_x}  |  <<COLOR:highlight>>Y:<</COLOR>> {var_y}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {n}\n\n"
        summary += "<<COLOR:accent>>── Fitted Parameters ──<</COLOR>>\n"
        summary += f"  {'Parameter':<12} {'Estimate':>12} {'Std Error':>12}\n"
        summary += f"  {'-' * 38}\n"
        for name, val, se in zip(param_names, popt, perr):
            summary += f"  {name:<12} {float(val):>12.6f} {float(se):>12.6f}\n"
        summary += "\n<<COLOR:accent>>── Goodness of Fit ──<</COLOR>>\n"
        summary += f"  R-squared: {r_squared:.4f}\n"
        summary += f"  RMSE:      {rmse:.4f}\n"
        summary += f"  AIC:       {aic:.2f}\n"
        summary += f"  BIC:       {bic:.2f}\n"

        result["summary"] = summary

        x_smooth = np.linspace(float(x.min()), float(x.max()), 200)
        y_smooth = func(x_smooth, *popt)
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "scatter",
                        "x": x.tolist(),
                        "y": y.tolist(),
                        "mode": "markers",
                        "marker": {"color": "#4a9f6e", "size": 5},
                        "name": "Data",
                    },
                    {
                        "type": "scatter",
                        "x": x_smooth.tolist(),
                        "y": y_smooth.tolist(),
                        "mode": "lines",
                        "line": {"color": "#e89547", "width": 2},
                        "name": f"Fitted ({model_type})",
                    },
                ],
                "layout": {
                    "title": f"Nonlinear Fit: {model_type}",
                    "xaxis": {"title": var_x},
                    "yaxis": {"title": var_y},
                    "height": 350,
                },
            }
        )

        residuals_nlr = y - y_pred
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "scatter",
                        "x": y_pred.tolist(),
                        "y": residuals_nlr.tolist(),
                        "mode": "markers",
                        "marker": {"color": "#47a5e8", "size": 5},
                        "name": "Residuals",
                    },
                ],
                "layout": {
                    "title": "Residuals vs Fitted",
                    "xaxis": {"title": "Fitted values"},
                    "yaxis": {"title": "Residual"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": float(y_pred.min()),
                            "x1": float(y_pred.max()),
                            "y0": 0,
                            "y1": 0,
                            "line": {"color": "#888", "dash": "dash"},
                        }
                    ],
                    "height": 250,
                },
            }
        )

        result["guide_observation"] = (
            f"Nonlinear regression ({model_type}): R2={r_squared:.4f}, RMSE={rmse:.4f}."
        )
        _nl_label = (
            "strong" if r_squared > 0.7 else "moderate" if r_squared > 0.3 else "weak"
        )
        result["narrative"] = _narrative(
            f"Nonlinear Regression ({model_type}): R\u00b2 = {r_squared:.4f} ({_nl_label})",
            f"RMSE = {rmse:.4f}. The {model_type} model captures non-linear patterns in the data.",
            next_steps="Compare with linear regression. If R\u00b2 improves substantially, the non-linear model is justified.",
        )
        result["statistics"] = {
            "model": model_type,
            "n": n,
            "parameters": {name: float(val) for name, val in zip(param_names, popt)},
            "parameter_se": {name: float(se) for name, se in zip(param_names, perr)},
            "r_squared": r_squared,
            "rmse": rmse,
            "aic": aic,
            "bic": bic,
        }

    elif analysis_id == "poisson_regression":
        """
        Poisson Regression — models count data as a function of predictors.
        Uses log link: log(E[Y]) = Xβ. Fits via GLM with Poisson family.
        Reports coefficients, IRR (incidence rate ratios), deviance goodness-of-fit.
        """
        import statsmodels.api as sm

        response_pr = config.get("response") or config.get("var")
        predictors_pr = config.get("predictors") or config.get("features", [])
        if isinstance(predictors_pr, str):
            predictors_pr = [predictors_pr]
        offset_col_pr = config.get("offset")  # exposure/offset variable (optional)

        data_pr = df[
            [response_pr] + predictors_pr + ([offset_col_pr] if offset_col_pr else [])
        ].dropna()
        y_pr = data_pr[response_pr].values.astype(float)

        # Check for non-negative integers
        if np.any(y_pr < 0):
            result["summary"] = "Poisson regression requires non-negative count data."
            return result

        # Build design matrix with dummies for categorical
        X_parts_pr = []
        feature_names_pr = []
        for pred in predictors_pr:
            if data_pr[pred].dtype == object or data_pr[pred].nunique() < 6:
                dummies = pd.get_dummies(
                    data_pr[pred], prefix=pred, drop_first=True, dtype=float
                )
                X_parts_pr.append(dummies.values)
                feature_names_pr.extend(dummies.columns.tolist())
            else:
                X_parts_pr.append(data_pr[[pred]].values.astype(float))
                feature_names_pr.append(pred)

        X_pr = np.column_stack(X_parts_pr) if X_parts_pr else np.ones((len(data_pr), 0))
        X_pr = sm.add_constant(X_pr)
        feature_names_pr = ["Intercept"] + feature_names_pr

        offset_vals_pr = (
            np.log(data_pr[offset_col_pr].values.astype(float))
            if offset_col_pr
            else None
        )

        try:
            model_pr = sm.GLM(
                y_pr, X_pr, family=sm.families.Poisson(), offset=offset_vals_pr
            ).fit()

            n_pr = int(model_pr.nobs)
            dev_pr = float(model_pr.deviance)
            pearson_chi2_pr = float(model_pr.pearson_chi2)
            df_resid_pr = int(model_pr.df_resid)
            aic_pr = float(model_pr.aic)
            bic_pr = float(model_pr.bic)
            llf_pr = float(model_pr.llf)

            # Dispersion test: deviance/df should be ~1 for Poisson
            dispersion_pr = dev_pr / df_resid_pr if df_resid_pr > 0 else float("nan")

            # Coefficients table
            coefs_pr = []
            for i, name in enumerate(feature_names_pr):
                coef_val = float(model_pr.params[i])
                se_val = float(model_pr.bse[i])
                z_val = float(model_pr.tvalues[i])
                p_val = float(model_pr.pvalues[i])
                irr_val = float(np.exp(coef_val))
                irr_lo = float(np.exp(model_pr.conf_int()[i, 0]))
                irr_hi = float(np.exp(model_pr.conf_int()[i, 1]))
                coefs_pr.append(
                    {
                        "name": name,
                        "coef": coef_val,
                        "se": se_val,
                        "z": z_val,
                        "p": p_val,
                        "irr": irr_val,
                        "irr_lo": irr_lo,
                        "irr_hi": irr_hi,
                    }
                )

            summary_pr = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_pr += "<<COLOR:title>>POISSON REGRESSION<</COLOR>>\n"
            summary_pr += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_pr += (
                f"<<COLOR:highlight>>Response:<</COLOR>> {response_pr} (count)\n"
            )
            summary_pr += (
                f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors_pr)}\n"
            )
            if offset_col_pr:
                summary_pr += f"<<COLOR:highlight>>Offset (exposure):<</COLOR>> log({offset_col_pr})\n"
            summary_pr += f"<<COLOR:highlight>>N:<</COLOR>> {n_pr}\n\n"

            summary_pr += "<<COLOR:accent>>── Coefficients ──<</COLOR>>\n"
            summary_pr += f"{'Term':<25} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>8} {'IRR':>8} {'95% CI':>16}\n"
            summary_pr += f"{'─' * 97}\n"
            for c in coefs_pr:
                sig = "<<COLOR:good>>*<</COLOR>>" if c["p"] < 0.05 else " "
                summary_pr += f"{c['name']:<25} {c['coef']:>8.4f} {c['se']:>8.4f} {c['z']:>8.3f} {c['p']:>8.4f} {c['irr']:>8.3f} [{c['irr_lo']:.3f}, {c['irr_hi']:.3f}] {sig}\n"

            summary_pr += "\n<<COLOR:accent>>── Model Fit ──<</COLOR>>\n"
            summary_pr += f"  Deviance: {dev_pr:.2f}  (df={df_resid_pr})\n"
            summary_pr += f"  Pearson χ²: {pearson_chi2_pr:.2f}\n"
            summary_pr += f"  Dispersion (Dev/df): {dispersion_pr:.3f}"
            if dispersion_pr > 1.5:
                summary_pr += "  <<COLOR:warning>>⚠ Overdispersion detected — consider negative binomial<</COLOR>>"
            summary_pr += (
                f"\n  AIC: {aic_pr:.1f}   BIC: {bic_pr:.1f}   Log-lik: {llf_pr:.1f}\n"
            )

            result["summary"] = summary_pr

            # IRR forest plot
            non_intercept = [c for c in coefs_pr if c["name"] != "Intercept"]
            if non_intercept:
                result["plots"].append(
                    {
                        "title": "Incidence Rate Ratios (95% CI)",
                        "data": [
                            {
                                "type": "scatter",
                                "mode": "markers",
                                "x": [c["irr"] for c in non_intercept],
                                "y": [c["name"] for c in non_intercept],
                                "marker": {
                                    "color": [
                                        "#4a9f6e" if c["p"] < 0.05 else "#5a6a5a"
                                        for c in non_intercept
                                    ],
                                    "size": 10,
                                },
                                "error_x": {
                                    "type": "data",
                                    "symmetric": False,
                                    "array": [
                                        c["irr_hi"] - c["irr"] for c in non_intercept
                                    ],
                                    "arrayminus": [
                                        c["irr"] - c["irr_lo"] for c in non_intercept
                                    ],
                                    "color": "#5a6a5a",
                                },
                                "showlegend": False,
                            }
                        ],
                        "layout": {
                            "height": max(250, 40 * len(non_intercept)),
                            "xaxis": {"title": "Incidence Rate Ratio", "type": "log"},
                            "yaxis": {"automargin": True},
                            "shapes": [
                                {
                                    "type": "line",
                                    "x0": 1,
                                    "x1": 1,
                                    "y0": -0.5,
                                    "y1": len(non_intercept) - 0.5,
                                    "line": {"color": "#e89547", "dash": "dash"},
                                }
                            ],
                        },
                    }
                )

            # Deviance residuals vs fitted
            fitted_pr = model_pr.mu
            resid_dev_pr = model_pr.resid_deviance
            result["plots"].append(
                {
                    "title": "Deviance Residuals vs Fitted",
                    "data": [
                        {
                            "type": "scatter",
                            "mode": "markers",
                            "x": fitted_pr.tolist(),
                            "y": resid_dev_pr.tolist(),
                            "marker": {"color": "#4a9f6e", "size": 4, "opacity": 0.6},
                            "showlegend": False,
                        }
                    ],
                    "layout": {
                        "height": 300,
                        "xaxis": {"title": "Fitted values"},
                        "yaxis": {"title": "Deviance residuals"},
                        "shapes": [
                            {
                                "type": "line",
                                "x0": min(fitted_pr),
                                "x1": max(fitted_pr),
                                "y0": 0,
                                "y1": 0,
                                "line": {"color": "#e89547", "dash": "dash"},
                            }
                        ],
                    },
                }
            )

            n_sig_pr = sum(
                1 for c in coefs_pr if c["p"] < 0.05 and c["name"] != "Intercept"
            )
            result["guide_observation"] = (
                f"Poisson regression: {n_sig_pr}/{len(non_intercept)} predictors significant. Dispersion={dispersion_pr:.2f}."
            )
            _disp_note = (
                " Overdispersion detected — consider negative binomial."
                if dispersion_pr > 2
                else ""
            )
            result["narrative"] = _narrative(
                f"Poisson Regression: {n_sig_pr} significant predictors",
                f"{n_sig_pr} of {len(non_intercept)} predictors significant. Dispersion = {dispersion_pr:.2f}.{_disp_note}",
                next_steps="Poisson regression models count data. Coefficients are log rate ratios — exp(coef) gives the rate multiplier.",
            )
            result["statistics"] = {
                "n": n_pr,
                "deviance": dev_pr,
                "df_resid": df_resid_pr,
                "pearson_chi2": pearson_chi2_pr,
                "dispersion": dispersion_pr,
                "aic": aic_pr,
                "bic": bic_pr,
                "log_likelihood": llf_pr,
                "coefficients": coefs_pr,
            }

        except Exception as e:
            result["summary"] = f"Poisson regression error: {str(e)}"

    # =====================================================================
    # Multiple Sampling Plan Comparison
    # =====================================================================
    elif analysis_id == "logistic":
        """
        Logistic Regression - for binary outcomes.
        Returns odds ratios, confidence intervals, and ROC curve.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import auc, confusion_matrix, roc_curve
        from sklearn.model_selection import train_test_split

        response = config.get("response")
        predictors = config.get("predictors", [])

        if not predictors:
            result["summary"] = "Please select at least one predictor."
            return result

        # Prepare data
        X = df[predictors].dropna()
        y = df[response].loc[X.index]

        # Encode target if needed
        unique_vals = y.unique()
        if len(unique_vals) != 2:
            result["summary"] = (
                f"Logistic regression requires binary outcome. Found {len(unique_vals)} unique values."
            )
            return result

        # Map to 0/1
        if y.dtype == "object" or str(y.dtype) == "category":
            y = (y == unique_vals[1]).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Coefficients and odds ratios
        coefs = model.coef_[0]
        odds_ratios = np.exp(coefs)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>LOGISTIC REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        summary += f"<<COLOR:highlight>>AUC (ROC):<</COLOR>> {roc_auc:.4f}\n\n"

        # Approximate SEs via Fisher information matrix
        _p_hat = model.predict_proba(X_train)[:, 1]
        _W = _p_hat * (1 - _p_hat)
        _X_mat = X_train.values
        _se_coefs = None
        _logistic_se_warning = None
        try:
            _XWX = _X_mat.T @ (_W[:, None] * _X_mat)
            _se_coefs = np.sqrt(np.maximum(np.diag(np.linalg.inv(_XWX)), 0))
        except np.linalg.LinAlgError:
            try:
                _XWX_reg = _XWX + 1e-6 * np.eye(_XWX.shape[0])
                _se_coefs = np.sqrt(np.maximum(np.diag(np.linalg.inv(_XWX_reg)), 0))
                _logistic_se_warning = "near-singular Fisher information (possible perfect separation or collinearity) — SEs are approximate"
            except Exception:
                _logistic_se_warning = "singular Fisher information matrix — cannot compute SEs. Check for perfect separation or collinear predictors"

        summary += "<<COLOR:accent>>── Coefficients & Odds Ratios ──<</COLOR>>\n"
        if _logistic_se_warning:
            summary += f"<<COLOR:danger>>⚠ {_logistic_se_warning}<</COLOR>>\n"
        if _se_coefs is not None:
            summary += f"  {'Predictor':<20} {'Coef':>10} {'SE':>10} {'OR':>10} {'95% CI for OR':>22}\n"
            summary += f"  {'-' * 74}\n"
            for i, (pred, coef, odds) in enumerate(zip(predictors, coefs, odds_ratios)):
                _or_lo = np.exp(coef - 1.96 * _se_coefs[i])
                _or_hi = np.exp(coef + 1.96 * _se_coefs[i])
                summary += f"  {pred:<20} {coef:>10.4f} {_se_coefs[i]:>10.4f} {odds:>10.4f} [{_or_lo:>8.4f}, {_or_hi:>8.4f}]\n"
        else:
            summary += f"  {'Predictor':<20} {'Coef':>10} {'Odds Ratio':>12}\n"
            summary += f"  {'-' * 44}\n"
            for pred, coef, odds in zip(predictors, coefs, odds_ratios):
                summary += f"  {pred:<20} {coef:>10.4f} {odds:>12.4f}\n"

        summary += "\n<<COLOR:accent>>── Confusion Matrix ──<</COLOR>>\n"
        summary += "  Predicted:    0      1\n"
        summary += f"  Actual 0:  {cm[0, 0]:>4}   {cm[0, 1]:>4}\n"
        summary += f"  Actual 1:  {cm[1, 0]:>4}   {cm[1, 1]:>4}\n"

        # ROC curve plot
        result["plots"].append(
            {
                "title": "ROC Curve",
                "data": [
                    {
                        "type": "scatter",
                        "x": fpr.tolist(),
                        "y": tpr.tolist(),
                        "mode": "lines",
                        "name": f"AUC = {roc_auc:.3f}",
                        "line": {"color": "#4a9f6e", "width": 2},
                    },
                    {
                        "type": "scatter",
                        "x": [0, 1],
                        "y": [0, 1],
                        "mode": "lines",
                        "name": "Random",
                        "line": {"color": "#9aaa9a", "dash": "dash"},
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "False Positive Rate"},
                    "yaxis": {"title": "True Positive Rate"},
                },
            }
        )

        # Odds ratio forest plot
        result["plots"].append(
            {
                "title": "Odds Ratios (Log Scale)",
                "data": [
                    {
                        "type": "bar",
                        "x": odds_ratios.tolist(),
                        "y": predictors,
                        "orientation": "h",
                        "marker": {
                            "color": [
                                "#4a9f6e" if o > 1 else "#e85747" for o in odds_ratios
                            ]
                        },
                    }
                ],
                "layout": {
                    "height": 250,
                    "xaxis": {"type": "log", "title": "Odds Ratio"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 1,
                            "x1": 1,
                            "y0": -0.5,
                            "y1": len(predictors) - 0.5,
                            "line": {"color": "#9aaa9a", "dash": "dash"},
                        }
                    ],
                },
            }
        )

        result["summary"] = summary
        result["guide_observation"] = (
            f"Logistic regression with AUC = {roc_auc:.3f}. Odds ratios > 1 increase probability of outcome."
        )
        result["statistics"] = {
            "AUC": float(roc_auc),
            "accuracy": float((y_pred == y_test).mean()),
        }
        _auc_label = (
            "excellent"
            if roc_auc > 0.9
            else "good" if roc_auc > 0.8 else "fair" if roc_auc > 0.7 else "poor"
        )
        _acc = float((y_pred == y_test).mean())
        verdict = f"Logistic model AUC = {roc_auc:.3f} ({_auc_label} discrimination)"
        body = f"The model classifies {response} using {len(predictors)} predictor{'s' if len(predictors) > 1 else ''} with {_acc * 100:.1f}% accuracy on held-out data. AUC = {roc_auc:.3f}."
        result["narrative"] = _narrative(
            verdict,
            body,
            next_steps="AUC > 0.8 = good model. Check the confusion matrix for class-specific performance. Odds ratios > 1 increase the probability of the outcome.",
            chart_guidance="ROC curve: the further from the diagonal, the better the model discriminates. The confusion matrix shows where the model makes errors.",
        )

        # ── Diagnostics ──
        diagnostics = []
        # Check for separation (perfect prediction) — extremely large coefficients
        for i, (pred, coef) in enumerate(zip(predictors, coefs)):
            if abs(coef) > 10:
                diagnostics.append(
                    {
                        "level": "warning",
                        "title": f"Possible separation: |coef| = {abs(coef):.1f} for {pred}",
                        "detail": f"Coefficient for '{pred}' is extremely large ({coef:.2f}), suggesting perfect or quasi-perfect separation. The model may be unreliable — consider penalized regression (Ridge/LASSO) or Firth's method.",
                    }
                )
        # Sample size per predictor (rule of thumb: 10-20 events per predictor)
        _n_events = int(y_train.sum())
        _n_non_events = int(len(y_train) - _n_events)
        _min_class = min(_n_events, _n_non_events)
        _epp = _min_class / len(predictors) if len(predictors) > 0 else _min_class
        if _epp < 10:
            diagnostics.append(
                {
                    "level": "warning",
                    "title": f"Low events per predictor ({_epp:.1f} EPP)",
                    "detail": f"Only {_min_class} events for {len(predictors)} predictor{'s' if len(predictors) > 1 else ''} (EPP = {_epp:.1f}). Rule of thumb is 10–20 EPP. Results may be overfitted and unstable.",
                }
            )
        # Multicollinearity check (VIF for multiple predictors)
        if len(predictors) >= 2:
            try:
                from statsmodels.stats.outliers_influence import (
                    variance_inflation_factor,
                )

                _X_vif = X_train.values
                _vif_vals = []
                for _vi in range(len(predictors)):
                    _vif_vals.append(variance_inflation_factor(_X_vif, _vi))
                _high_vif = [
                    (predictors[_vi], _vif_vals[_vi])
                    for _vi in range(len(predictors))
                    if _vif_vals[_vi] > 5
                ]
                if _high_vif:
                    _vif_str = ", ".join(f"{n} (VIF={v:.1f})" for n, v in _high_vif)
                    diagnostics.append(
                        {
                            "level": "warning",
                            "title": "Multicollinearity detected",
                            "detail": f"High VIF: {_vif_str}. Correlated predictors inflate standard errors and make individual coefficients unreliable.",
                        }
                    )
            except Exception:
                pass
        # Effect size emphasis: odds ratios
        for i, (pred, odds) in enumerate(zip(predictors, odds_ratios)):
            if odds > 5 or odds < 0.2:
                diagnostics.append(
                    {
                        "level": "info",
                        "title": f"Large odds ratio for {pred} (OR = {odds:.2f})",
                        "detail": f"A one-unit increase in '{pred}' {'multiplies' if odds > 1 else 'divides'} the odds of {response} by {odds:.2f} — a practically significant effect.",
                    }
                )
            elif 0.8 <= odds <= 1.2 and _se_coefs is not None:
                # Check if statistically significant despite trivial OR
                _z = coefs[i] / _se_coefs[i] if _se_coefs[i] > 0 else 0
                _p_coef = 2 * (1 - sp_stats.norm.cdf(abs(_z)))
                if _p_coef < 0.05:
                    diagnostics.append(
                        {
                            "level": "warning",
                            "title": f"Significant but trivial effect for {pred} (OR = {odds:.2f})",
                            "detail": f"'{pred}' is statistically significant (p = {_p_coef:.4f}) but the odds ratio is near 1 — the practical impact is negligible.",
                        }
                    )
        result["diagnostics"] = diagnostics

    elif analysis_id == "stepwise":
        """
        Stepwise Regression - automatic variable selection.
        Uses forward selection, backward elimination, or both.
        """
        import statsmodels.api as sm
        from sklearn.linear_model import LinearRegression

        response = config.get("response")
        predictors = config.get("predictors", [])
        method = config.get("method", "both")  # forward, backward, both
        alpha_enter = float(config.get("alpha_enter", 0.05))
        alpha_remove = float(config.get("alpha_remove", 0.10))

        X_full = df[predictors].dropna()
        y = df[response].loc[X_full.index]
        n = len(y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>STEPWISE REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += (
            f"<<COLOR:highlight>>Candidate Predictors:<</COLOR>> {len(predictors)}\n"
        )
        summary += f"<<COLOR:highlight>>Method:<</COLOR>> {method}\n"
        summary += f"<<COLOR:highlight>>α to enter:<</COLOR>> {alpha_enter}\n"
        summary += f"<<COLOR:highlight>>α to remove:<</COLOR>> {alpha_remove}\n\n"

        selected = []
        remaining = list(predictors)
        step_history = []

        def get_pvalue(X_subset, y_data, var_name):
            """Get p-value for adding a variable."""
            X_with_const = sm.add_constant(X_subset)
            try:
                model = sm.OLS(y_data, X_with_const).fit()
                idx = list(X_subset.columns).index(var_name) + 1  # +1 for constant
                return model.pvalues.iloc[idx]
            except Exception:
                return 1.0

        # Stepwise selection
        step = 0
        while True:
            step += 1
            changed = False

            # Forward step: try adding variables
            if method in ["forward", "both"] and remaining:
                best_pval = 1.0
                best_var = None

                for var in remaining:
                    if selected:
                        X_test = X_full[selected + [var]]
                    else:
                        X_test = X_full[[var]]

                    pval = get_pvalue(X_test, y, var)
                    if pval < best_pval:
                        best_pval = pval
                        best_var = var

                if best_var and best_pval < alpha_enter:
                    selected.append(best_var)
                    remaining.remove(best_var)
                    step_history.append(
                        f"Step {step}: ADD {best_var} (p={best_pval:.4f})"
                    )
                    changed = True

            # Backward step: try removing variables
            if method in ["backward", "both"] and selected:
                worst_pval = 0.0
                worst_var = None

                X_current = X_full[selected]
                X_with_const = sm.add_constant(X_current)
                try:
                    model = sm.OLS(y, X_with_const).fit()
                    for i, var in enumerate(selected):
                        pval = model.pvalues.iloc[i + 1]  # +1 for constant
                        if pval > worst_pval:
                            worst_pval = pval
                            worst_var = var
                except Exception:
                    pass

                if worst_var and worst_pval > alpha_remove:
                    selected.remove(worst_var)
                    remaining.append(worst_var)
                    step_history.append(
                        f"Step {step}: REMOVE {worst_var} (p={worst_pval:.4f})"
                    )
                    changed = True

            if not changed:
                break

            if step > 50:  # Safety limit
                break

        summary += "<<COLOR:accent>>── Selection History ──<</COLOR>>\n"
        for hist in step_history:
            summary += f"  {hist}\n"
        summary += "\n"

        # Final model
        if selected:
            X_final = X_full[selected]
            X_with_const = sm.add_constant(X_final)
            final_model = sm.OLS(y, X_with_const).fit()

            summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            summary += (
                "<<COLOR:accent>>                              FINAL MODEL<</COLOR>>\n"
            )
            summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            summary += f"<<COLOR:highlight>>Selected Variables:<</COLOR>> {', '.join(selected)}\n"
            summary += f"<<COLOR:highlight>>R²:<</COLOR>> {final_model.rsquared:.4f}\n"
            summary += f"<<COLOR:highlight>>Adjusted R²:<</COLOR>> {final_model.rsquared_adj:.4f}\n"
            summary += f"<<COLOR:highlight>>F-statistic:<</COLOR>> {final_model.fvalue:.4f} (p={final_model.f_pvalue:.4e})\n\n"

            summary += "<<COLOR:accent>>── Coefficients ──<</COLOR>>\n"
            summary += f"  {'Variable':<20} {'Coef':>12} {'Std Err':>12} {'t':>10} {'P>|t|':>10}\n"
            summary += f"  {'-' * 66}\n"
            for i, name in enumerate(["const"] + selected):
                summary += f"  {name:<20} {final_model.params.iloc[i]:>12.4f} {final_model.bse.iloc[i]:>12.4f} {final_model.tvalues.iloc[i]:>10.3f} {final_model.pvalues.iloc[i]:>10.4f}\n"

            result["statistics"] = {
                "R²": float(final_model.rsquared),
                "Adj_R²": float(final_model.rsquared_adj),
                "n_selected": len(selected),
                "selected_vars": selected,
            }

            # Coefficient plot
            result["plots"].append(
                {
                    "title": "Stepwise Selected Coefficients",
                    "data": [
                        {
                            "type": "bar",
                            "x": selected,
                            "y": [
                                float(final_model.params.iloc[i + 1])
                                for i in range(len(selected))
                            ],
                            "marker": {"color": "#4a9f6e"},
                        }
                    ],
                    "layout": {"height": 250, "yaxis": {"title": "Coefficient"}},
                }
            )
        else:
            summary += (
                "<<COLOR:warning>>No variables met the selection criteria.<</COLOR>>\n"
            )
            result["statistics"] = {"n_selected": 0, "selected_vars": []}

        result["summary"] = summary
        result["guide_observation"] = (
            f"Stepwise regression selected {len(selected)} of {len(predictors)} predictors."
        )
        result["narrative"] = _narrative(
            f"Stepwise selected {len(selected)} of {len(predictors)} predictors",
            f"Automatic variable selection retained: <strong>{', '.join(selected[:5]) if selected else 'none'}</strong>."
            + (f" ({len(selected) - 5} more)" if len(selected) > 5 else ""),
            next_steps="Stepwise results should be validated. The selected model may overfit — use cross-validation or best subsets for comparison.",
        )

    elif analysis_id == "best_subsets":
        """
        Best Subsets Regression - evaluate all possible models.
        Compares models by R², Adjusted R², Cp, BIC.
        """
        from itertools import combinations

        from sklearn.linear_model import LinearRegression

        response = config.get("response")
        predictors = config.get("predictors", [])
        max_predictors = min(int(config.get("max_predictors", 8)), len(predictors), 8)

        X_full = df[predictors].dropna()
        y = df[response].loc[X_full.index]
        n = len(y)
        p_full = len(predictors)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BEST SUBSETS REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Candidate Predictors:<</COLOR>> {p_full}\n"
        summary += f"<<COLOR:highlight>>Max subset size:<</COLOR>> {max_predictors}\n\n"

        # Calculate full model MSE for Cp
        model_full = LinearRegression().fit(X_full, y)
        mse_full = np.mean((y - model_full.predict(X_full)) ** 2)

        results_list = []

        # Evaluate all subsets up to max_predictors
        for k in range(1, max_predictors + 1):
            for combo in combinations(predictors, k):
                X_sub = X_full[list(combo)]
                model = LinearRegression().fit(X_sub, y)
                y_pred = model.predict(X_sub)

                sse = np.sum((y - y_pred) ** 2)
                r2 = 1 - sse / np.sum((y - y.mean()) ** 2)
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

                # Mallows' Cp
                cp = sse / mse_full - n + 2 * (k + 1)

                # BIC
                bic = n * np.log(sse / n) + k * np.log(n)

                results_list.append(
                    {
                        "vars": combo,
                        "k": k,
                        "r2": r2,
                        "adj_r2": adj_r2,
                        "cp": cp,
                        "bic": bic,
                    }
                )

        # Sort by different criteria
        sorted(results_list, key=lambda x: -x["r2"])
        by_adj_r2 = sorted(results_list, key=lambda x: -x["adj_r2"])
        by_cp = sorted(results_list, key=lambda x: x["cp"])
        sorted(results_list, key=lambda x: x["bic"])

        summary += "<<COLOR:accent>>── Best Models by Criterion ──<</COLOR>>\n\n"

        summary += "<<COLOR:accent>>By Adjusted R² (top 5):<</COLOR>>\n"
        summary += f"  {'Vars':<8} {'R²':>8} {'Adj R²':>8} {'Cp':>10} {'BIC':>10} {'Predictors'}\n"
        summary += f"  {'-' * 70}\n"
        for res in by_adj_r2[:5]:
            vars_str = ", ".join(res["vars"][:3]) + (
                "..." if len(res["vars"]) > 3 else ""
            )
            summary += f"  {res['k']:<8} {res['r2']:>8.4f} {res['adj_r2']:>8.4f} {res['cp']:>10.2f} {res['bic']:>10.2f} {vars_str}\n"

        summary += "\n<<COLOR:accent>>By Mallows' Cp (top 5):<</COLOR>>\n"
        summary += f"  {'Vars':<8} {'R²':>8} {'Adj R²':>8} {'Cp':>10} {'BIC':>10}\n"
        summary += f"  {'-' * 50}\n"
        for res in by_cp[:5]:
            summary += f"  {res['k']:<8} {res['r2']:>8.4f} {res['adj_r2']:>8.4f} {res['cp']:>10.2f} {res['bic']:>10.2f}\n"

        # Best model recommendation
        best = by_adj_r2[0]
        summary += "\n<<COLOR:success>>RECOMMENDED MODEL:<</COLOR>>\n"
        summary += f"  Variables: {', '.join(best['vars'])}\n"
        summary += f"  R² = {best['r2']:.4f}, Adj R² = {best['adj_r2']:.4f}\n"

        result["summary"] = summary
        result["guide_observation"] = (
            f"Best subsets: recommended {best['k']}-variable model with Adj R² = {best['adj_r2']:.4f}"
        )
        result["narrative"] = _narrative(
            f"Best subsets: {best['k']}-variable model recommended (Adj R\u00b2 = {best['adj_r2']:.4f})",
            f"Best predictors: <strong>{', '.join(best.get('vars', []))}</strong>. Evaluated all variable combinations to find the best trade-off between fit and complexity.",
            next_steps="More variables = higher R\u00b2 but risk overfitting. Choose the model where Adj R\u00b2 plateaus.",
            chart_guidance="Each row is a model size (k variables). Stars mark the best model at each size. Choose where improvement flattens.",
        )
        result["statistics"] = {
            "best_r2": float(best["r2"]),
            "best_adj_r2": float(best["adj_r2"]),
            "best_vars": list(best["vars"]),
            "models_evaluated": len(results_list),
        }

        # Plot: R² and Adj R² by number of variables
        k_values = sorted(set(r["k"] for r in results_list))
        best_r2_by_k = [
            max(r["r2"] for r in results_list if r["k"] == k) for k in k_values
        ]
        best_adj_r2_by_k = [
            max(r["adj_r2"] for r in results_list if r["k"] == k) for k in k_values
        ]

        result["plots"].append(
            {
                "title": "Best Subsets: R² by Model Size",
                "data": [
                    {
                        "type": "scatter",
                        "x": k_values,
                        "y": best_r2_by_k,
                        "mode": "lines+markers",
                        "name": "R²",
                        "line": {"color": "#4a9f6e"},
                    },
                    {
                        "type": "scatter",
                        "x": k_values,
                        "y": best_adj_r2_by_k,
                        "mode": "lines+markers",
                        "name": "Adj R²",
                        "line": {"color": "#47a5e8"},
                    },
                ],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": "Number of Predictors"},
                    "yaxis": {"title": "R²"},
                },
            }
        )

    elif analysis_id == "robust_regression":
        """
        Robust Regression - outlier-resistant regression.
        Uses Huber or MM estimators.
        """
        import statsmodels.api as sm
        from statsmodels.robust.robust_linear_model import RLM

        response = config.get("response")
        predictors = config.get("predictors", [])
        method = config.get("method", "huber")  # huber, bisquare, andrews

        X = df[predictors].dropna()
        y = df[response].loc[X.index]
        n = len(y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>ROBUST REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        summary += (
            f"<<COLOR:highlight>>Method:<</COLOR>> {method.title()} M-estimator\n"
        )
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n\n"

        # Select M-estimator
        if method == "huber":
            M_estimator = sm.robust.norms.HuberT()
            summary += (
                "<<COLOR:dim>>Huber's T: downweights residuals > 1.345σ<</COLOR>>\n"
            )
        elif method == "bisquare":
            M_estimator = sm.robust.norms.TukeyBiweight()
            summary += "<<COLOR:dim>>Tukey's Bisquare: zero weight for residuals > 4.685σ<</COLOR>>\n"
        elif method == "andrews":
            M_estimator = sm.robust.norms.AndrewWave()
            summary += (
                "<<COLOR:dim>>Andrew's Wave: sinusoidal downweighting<</COLOR>>\n"
            )
        else:
            M_estimator = sm.robust.norms.HuberT()

        X_const = sm.add_constant(X)

        # Fit OLS for comparison
        ols_model = sm.OLS(y, X_const).fit()

        # Fit robust model
        robust_model = RLM(y, X_const, M=M_estimator).fit()

        summary += "\n<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += "<<COLOR:accent>>                        COMPARISON: OLS vs ROBUST<</COLOR>>\n"
        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += f"  {'Variable':<20} {'OLS Coef':>12} {'Robust Coef':>12} {'Difference':>12}\n"
        summary += f"  {'-' * 58}\n"

        var_names = ["const"] + predictors
        for i, name in enumerate(var_names):
            ols_coef = ols_model.params.iloc[i]
            rob_coef = robust_model.params.iloc[i]
            diff = rob_coef - ols_coef
            diff_pct = 100 * diff / abs(ols_coef) if abs(ols_coef) > 1e-10 else 0
            flag = "<<COLOR:warning>>*<</COLOR>>" if abs(diff_pct) > 10 else ""
            summary += f"  {name:<20} {ols_coef:>12.4f} {rob_coef:>12.4f} {diff:>+12.4f} {flag}\n"

        summary += "\n<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += "<<COLOR:accent>>                           ROBUST MODEL DETAILS<</COLOR>>\n"
        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += (
            f"  {'Variable':<20} {'Coef':>12} {'Std Err':>12} {'z':>10} {'P>|z|':>10}\n"
        )
        summary += f"  {'-' * 66}\n"

        for i, name in enumerate(var_names):
            summary += f"  {name:<20} {robust_model.params.iloc[i]:>12.4f} {robust_model.bse.iloc[i]:>12.4f} {robust_model.tvalues.iloc[i]:>10.3f} {robust_model.pvalues.iloc[i]:>10.4f}\n"

        # Identify outliers (low weights)
        weights = robust_model.weights
        outlier_threshold = 0.5
        outliers = np.where(weights < outlier_threshold)[0]

        summary += f"\n<<COLOR:text>>Observations with low weight (<{outlier_threshold}):<</COLOR>> {len(outliers)}\n"
        if len(outliers) > 0 and len(outliers) <= 10:
            summary += f"  Indices: {list(outliers)}\n"

        summary += "\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        coef_diffs = np.abs(robust_model.params.values - ols_model.params.values)
        if np.max(coef_diffs[1:]) > 0.1 * np.max(np.abs(ols_model.params.values[1:])):
            summary += "  <<COLOR:warning>>Coefficients differ substantially - outliers are influential.<</COLOR>>\n"
            summary += "  Robust estimates may be more reliable.\n"
        else:
            summary += "  <<COLOR:good>>Coefficients are similar - outliers have minimal influence.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = (
            f"Robust regression ({method}) identified {len(outliers)} low-weight observations."
        )
        result["narrative"] = _narrative(
            f"Robust Regression ({method}): {len(outliers)} influential points down-weighted",
            f"Robust regression reduces the influence of outliers. {len(outliers)} observation{'s' if len(outliers) != 1 else ''} received low weight.",
            next_steps="Compare coefficients with OLS regression. Large differences indicate outliers were distorting the OLS fit.",
        )
        result["statistics"] = {
            "n_outliers": len(outliers),
            "method": method,
            **{
                f"coef_{name}": float(robust_model.params.iloc[i])
                for i, name in enumerate(var_names)
            },
        }

        # Plot: OLS residuals vs Robust residuals
        ols_resid = ols_model.resid
        rob_resid = robust_model.resid

        result["plots"].append(
            {
                "title": "Residual Comparison",
                "data": [
                    {
                        "type": "scatter",
                        "x": ols_resid.tolist(),
                        "y": rob_resid.tolist(),
                        "mode": "markers",
                        "marker": {
                            "color": weights.tolist(),
                            "colorscale": [[0, "#e85747"], [1, "#4a9f6e"]],
                            "size": 6,
                            "colorbar": {"title": "Weight"},
                        },
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "OLS Residuals"},
                    "yaxis": {"title": "Robust Residuals"},
                },
            }
        )

        # Weights histogram
        result["plots"].append(
            {
                "title": "Observation Weights",
                "data": [
                    {
                        "type": "histogram",
                        "x": weights.tolist(),
                        "marker": {
                            "color": "rgba(74, 159, 110, 0.4)",
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                    }
                ],
                "layout": {
                    "height": 200,
                    "xaxis": {"title": "Weight", "range": [0, 1.1]},
                },
            }
        )

    elif analysis_id == "glm":
        """
        General Linear Model — unified engine for ANOVA, ANCOVA, multivariate regression,
        and mixed-effects models. Handles:
        - Pure ANOVA (factors only)
        - Pure regression (covariates only)
        - ANCOVA (factors + covariates with interactions and LS-Means)
        - Mixed models (fixed + random factors)
        - Two-way+ interactions between factors and factor*covariate
        Produces Type III ANOVA table, LS-Means, effect sizes, interaction plots,
        and full residual diagnostics (4-panel).
        """
        response = config.get("response") or config.get("var")
        fixed_factors = config.get("fixed_factors", [])
        random_factors = config.get("random_factors", [])
        covariates = config.get("covariates", [])
        include_interactions = config.get("interactions", True)
        include_factor_cov_interactions = config.get(
            "factor_covariate_interactions", False
        )
        alpha = 1 - config.get("conf", 95) / 100

        # Support single-value fallbacks from generic dialogs
        if not fixed_factors and config.get("factor"):
            fixed_factors = [config["factor"]]
        if not random_factors and config.get("random_factor"):
            random_factors = [config["random_factor"]]
        if not covariates and config.get("covariate"):
            covariates = [config["covariate"]]

        all_cols = [response] + fixed_factors + random_factors + covariates
        try:
            data = df[all_cols].dropna()
            N = len(data)
            from scipy import stats as qstats

            # ── Build formula ──
            terms = []
            for f in fixed_factors:
                terms.append(f"C({f})")
            for c in covariates:
                terms.append(c)

            # Factor*Factor interactions
            if include_interactions and len(fixed_factors) >= 2:
                for i in range(len(fixed_factors)):
                    for j in range(i + 1, len(fixed_factors)):
                        terms.append(f"C({fixed_factors[i]}):C({fixed_factors[j]})")

            # Factor*Covariate interactions (ANCOVA: test homogeneity of slopes)
            if covariates and fixed_factors:
                if include_factor_cov_interactions or len(fixed_factors) == 1:
                    # Auto-include for single-factor ANCOVA to test assumption
                    for f in fixed_factors:
                        for c in covariates:
                            terms.append(f"C({f}):{c}")

            formula = (
                f"{response} ~ " + " + ".join(terms) if terms else f"{response} ~ 1"
            )

            # Determine model type label
            has_factors = len(fixed_factors) > 0
            has_covariates = len(covariates) > 0
            has_random = len(random_factors) > 0
            if has_factors and has_covariates:
                model_label = "ANCOVA" if not has_random else "Mixed ANCOVA"
            elif has_factors and not has_covariates:
                model_label = "ANOVA (GLM)" if not has_random else "Mixed-Effects ANOVA"
            elif has_covariates and not has_factors:
                model_label = (
                    "Multiple Regression"
                    if len(covariates) > 1
                    else "Simple Regression"
                )
            else:
                model_label = "Intercept-Only Model"

            # ═══════════════════════════════════════════
            # MIXED MODEL (random factors present)
            # ═══════════════════════════════════════════
            if has_random:
                from statsmodels.formula.api import mixedlm

                group_var = random_factors[0]
                model = mixedlm(formula, data, groups=data[group_var])
                fit = model.fit(reml=True)

                var_random = (
                    float(fit.cov_re.iloc[0, 0])
                    if hasattr(fit.cov_re, "iloc")
                    else float(fit.cov_re)
                )
                var_residual = float(fit.scale)
                var_total = var_random + var_residual
                icc = var_random / var_total if var_total > 0 else 0

                summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
                summary_text += (
                    f"<<COLOR:title>>GENERAL LINEAR MODEL — {model_label}<</COLOR>>\n"
                )
                summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
                summary_text += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
                summary_text += f"<<COLOR:highlight>>Fixed:<</COLOR>> {', '.join(fixed_factors + covariates)}\n"
                summary_text += f"<<COLOR:highlight>>Random:<</COLOR>> {', '.join(random_factors)}\n"
                summary_text += f"<<COLOR:highlight>>Formula:<</COLOR>> {formula}\n"
                summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

                summary_text += "<<COLOR:accent>>── Fixed Effects ──<</COLOR>>\n"
                summary_text += f"{'Term':<35} {'Coef':>10} {'SE':>10} {'z':>8} {'p-value':>10} {'Sig':>5} {f'{int((1 - alpha) * 100)}% CI':>22}\n"
                summary_text += f"{'─' * 105}\n"
                for name in fit.fe_params.index:
                    coef = float(fit.fe_params[name])
                    se = float(fit.bse[name]) if name in fit.bse.index else None
                    pv = float(fit.pvalues[name]) if name in fit.pvalues.index else None
                    z = coef / se if se and se > 0 else 0
                    sig = (
                        "<<COLOR:good>>*<</COLOR>>"
                        if pv is not None and pv < alpha
                        else ""
                    )
                    p_str = f"{pv:.4f}" if pv is not None else "N/A"
                    se_str = f"{se:.4f}" if se is not None else "N/A"
                    _ci_str = (
                        f"[{coef - 1.96 * se:.4f}, {coef + 1.96 * se:.4f}]"
                        if se
                        else ""
                    )
                    summary_text += f"{str(name):<35} {coef:>10.4f} {se_str:>10} {z:>8.2f} {p_str:>10} {sig:>5} {_ci_str:>22}\n"

                summary_text += (
                    "\n<<COLOR:accent>>── Variance Components ──<</COLOR>>\n"
                )
                summary_text += f"  {group_var} (random): {var_random:.4f} ({icc * 100:.1f}% of total)\n"
                summary_text += f"  Residual: {var_residual:.4f} ({(1 - icc) * 100:.1f}% of total)\n"
                summary_text += f"  ICC (Intraclass Correlation): {icc:.4f}\n"

                if icc > 0.1:
                    summary_text += f"\n<<COLOR:good>>ICC = {icc:.3f} — substantial clustering. Mixed model is appropriate.<</COLOR>>"
                else:
                    summary_text += f"\n<<COLOR:text>>ICC = {icc:.3f} — low clustering. A fixed-effects model may suffice.<</COLOR>>"

                fitted_vals = fit.fittedvalues
                resid_vals = fit.resid

                result["statistics"] = {
                    "model_type": "mixed",
                    "model_label": model_label,
                    "n": N,
                    "formula": formula,
                    "var_random": var_random,
                    "var_residual": var_residual,
                    "icc": icc,
                    "aic": float(fit.aic) if hasattr(fit, "aic") else None,
                }

            # ═══════════════════════════════════════════
            # FIXED MODEL (OLS — ANOVA/ANCOVA/Regression)
            # ═══════════════════════════════════════════
            else:
                import statsmodels.api as sm
                from statsmodels.formula.api import ols

                model = ols(formula, data=data).fit()

                # Type III ANOVA table
                try:
                    anova_table = sm.stats.anova_lm(model, typ=3)
                except Exception:
                    anova_table = sm.stats.anova_lm(model, typ=2)

                # Compute effect sizes (partial eta-squared)
                ss_residual = (
                    float(anova_table.loc["Residual", "sum_sq"])
                    if "Residual" in anova_table.index
                    else 0
                )
                ss_total = float(anova_table["sum_sq"].sum())

                summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
                summary_text += (
                    f"<<COLOR:title>>GENERAL LINEAR MODEL — {model_label}<</COLOR>>\n"
                )
                summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
                summary_text += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
                if fixed_factors:
                    summary_text += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(fixed_factors)}\n"
                if covariates:
                    summary_text += f"<<COLOR:highlight>>Covariates:<</COLOR>> {', '.join(covariates)}\n"
                summary_text += f"<<COLOR:highlight>>Formula:<</COLOR>> {formula}\n"
                summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}, R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}\n\n"

                # ANOVA table with partial eta-squared
                summary_text += "<<COLOR:accent>>── Analysis of Variance (Type III SS) ──<</COLOR>>\n"
                eta_header = " η²p" if has_factors else ""
                summary_text += f"{'Source':<30} {'DF':>5} {'Adj SS':>12} {'Adj MS':>12} {'F':>10} {'p-value':>10}{eta_header:>8}\n"
                summary_text += f"{'─' * (87 + (8 if has_factors else 0))}\n"

                for idx in anova_table.index:
                    row = anova_table.loc[idx]
                    df_val = int(row["df"]) if "df" in row else ""
                    ss = float(row["sum_sq"]) if "sum_sq" in row else 0
                    ms = float(row["mean_sq"]) if "mean_sq" in row else 0
                    f_val = (
                        float(row["F"])
                        if "F" in row and not np.isnan(row["F"])
                        else None
                    )
                    pv = (
                        float(row["PR(>F)"])
                        if "PR(>F)" in row and not np.isnan(row["PR(>F)"])
                        else None
                    )
                    sig = (
                        "<<COLOR:good>>*<</COLOR>>"
                        if pv is not None and pv < alpha
                        else ""
                    )
                    f_str = f"{f_val:.4f}" if f_val is not None else ""
                    p_str = f"{pv:.4f}" if pv is not None else ""
                    # Partial eta-squared = SS_effect / (SS_effect + SS_error)
                    if (
                        has_factors
                        and idx != "Residual"
                        and idx != "Intercept"
                        and ss_residual > 0
                    ):
                        eta_p = ss / (ss + ss_residual)
                        eta_str = f"{eta_p:>7.3f}"
                    else:
                        eta_str = "" if has_factors else ""
                    summary_text += f"{str(idx):<30} {df_val:>5} {ss:>12.4f} {ms:>12.4f} {f_str:>10} {p_str:>10} {sig} {eta_str}\n"

                summary_text += "\n<<COLOR:accent>>── Model Summary ──<</COLOR>>\n"
                summary_text += f"  S (root MSE): {np.sqrt(model.mse_resid):.4f}\n"
                summary_text += (
                    f"  R²: {model.rsquared:.4f}  Adj R²: {model.rsquared_adj:.4f}\n"
                )
                summary_text += f"  AIC: {model.aic:.1f}  BIC: {model.bic:.1f}\n"

                # Coefficients table with CIs
                _glm_ci = model.conf_int(alpha=alpha)
                summary_text += "\n<<COLOR:accent>>── Coefficients ──<</COLOR>>\n"
                summary_text += f"{'Term':<35} {'Coef':>10} {'SE':>10} {'t':>8} {'p-value':>10} {f'{int((1 - alpha) * 100)}% CI':>22}\n"
                summary_text += f"{'─' * 97}\n"
                for name in model.params.index:
                    coef = float(model.params[name])
                    se = float(model.bse[name])
                    t = float(model.tvalues[name])
                    pv = float(model.pvalues[name])
                    _ci_lo = float(_glm_ci.loc[name, 0])
                    _ci_hi = float(_glm_ci.loc[name, 1])
                    summary_text += f"{str(name):<35} {coef:>10.4f} {se:>10.4f} {t:>8.2f} {pv:>10.4f} [{_ci_lo:.4f}, {_ci_hi:.4f}]\n"

                # ── LS-Means (Adjusted Means) for ANCOVA ──
                if has_factors and has_covariates:
                    summary_text += f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>\n"
                    summary_text += "<<COLOR:accent>>── Least-Squares Means (Adjusted Means) ──<</COLOR>>\n"
                    summary_text += (
                        "<<COLOR:text>>Covariates held at their means: "
                        + ", ".join([f"{c}={data[c].mean():.4f}" for c in covariates])
                        + "<</COLOR>>\n\n"
                    )

                    for factor in fixed_factors:
                        levels = sorted(data[factor].unique().tolist(), key=str)
                        raw_means = data.groupby(factor)[response].mean()
                        raw_sds = data.groupby(factor)[response].std()
                        raw_ns = data.groupby(factor)[response].count()

                        # Compute LS-means by predicting at covariate means
                        cov_means = {c: data[c].mean() for c in covariates}
                        ls_means = {}
                        for lev in levels:
                            pred_data = data[data[factor] == lev].copy()
                            for c in covariates:
                                pred_data[c] = cov_means[c]
                            ls_means[lev] = float(model.predict(pred_data).mean())

                        summary_text += f"  {factor}:\n"
                        summary_text += f"  {'Level':<20} {'N':>6} {'Raw Mean':>10} {'Adj Mean':>10} {'Std Dev':>10}\n"
                        summary_text += f"  {'─' * 58}\n"
                        for lev in levels:
                            rm = float(raw_means[lev])
                            adj = ls_means[lev]
                            sd = float(raw_sds[lev])
                            n_lev = int(raw_ns[lev])
                            summary_text += f"  {str(lev):<20} {n_lev:>6} {rm:>10.4f} {adj:>10.4f} {sd:>10.4f}\n"

                        diff = max(ls_means.values()) - min(ls_means.values())
                        raw_diff = float(raw_means.max()) - float(raw_means.min())
                        if abs(raw_diff - diff) > 0.01 * abs(raw_diff + 0.001):
                            summary_text += "\n  <<COLOR:highlight>>Note: Adjusted means differ from raw means — covariate adjustment matters.<</COLOR>>\n"
                            summary_text += f"  Raw range: {raw_diff:.4f} → Adjusted range: {diff:.4f}\n"

                    # Homogeneity of slopes test (factor*covariate interaction significance)
                    fxcov_terms = [
                        f"C({f}):{c}" for f in fixed_factors for c in covariates
                    ]
                    sig_interactions = []
                    for term in fxcov_terms:
                        for idx in anova_table.index:
                            if (
                                term.replace("C(", "").replace(")", "") in str(idx)
                                or str(idx) == term
                            ):
                                pv_term = (
                                    float(anova_table.loc[idx, "PR(>F)"])
                                    if "PR(>F)" in anova_table.columns
                                    and not np.isnan(anova_table.loc[idx, "PR(>F)"])
                                    else None
                                )
                                if pv_term is not None and pv_term < alpha:
                                    sig_interactions.append((str(idx), pv_term))

                    if sig_interactions:
                        summary_text += "\n<<COLOR:bad>>⚠ Homogeneity of Slopes Violated:<</COLOR>>\n"
                        for term, pv in sig_interactions:
                            summary_text += f"  {term}: p = {pv:.4f} — slopes differ across groups. ANCOVA assumption violated.\n"
                        summary_text += "  <<COLOR:text>>Consider: separate regressions per group, or remove the covariate.<</COLOR>>\n"
                    elif has_factors and has_covariates:
                        summary_text += "\n<<COLOR:good>>Homogeneity of slopes OK — factor*covariate interactions are not significant.<</COLOR>>\n"

                fitted_vals = model.fittedvalues
                resid_vals = model.resid

                result["statistics"] = {
                    "model_type": "fixed",
                    "model_label": model_label,
                    "n": N,
                    "formula": formula,
                    "r_squared": float(model.rsquared),
                    "adj_r_squared": float(model.rsquared_adj),
                    "f_statistic": float(model.fvalue),
                    "f_pvalue": float(model.f_pvalue),
                    "aic": float(model.aic),
                    "bic": float(model.bic),
                    "root_mse": float(np.sqrt(model.mse_resid)),
                }

            result["summary"] = summary_text

            # ═══════════════════════════════════════════
            # PLOTS — Full 4-panel residual diagnostics
            # ═══════════════════════════════════════════

            # 1. Residuals vs Fitted
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
                        "xaxis": {"title": "Fitted Value"},
                        "yaxis": {"title": "Residual"},
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

            # 2. Normal Q-Q Plot
            sorted_resid = np.sort(resid_vals.values)
            n_qq = len(sorted_resid)
            theoretical_q = [
                float(qstats.norm.ppf((i + 0.5) / n_qq)) for i in range(n_qq)
            ]
            result["plots"].append(
                {
                    "title": "Normal Probability Plot of Residuals",
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
                                theoretical_q[0] * np.std(sorted_resid)
                                + np.mean(sorted_resid),
                                theoretical_q[-1] * np.std(sorted_resid)
                                + np.mean(sorted_resid),
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

            # 3. Residuals Histogram
            hist_vals, bin_edges = np.histogram(resid_vals.values, bins="auto")
            bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist()
            result["plots"].append(
                {
                    "title": "Histogram of Residuals",
                    "data": [
                        {
                            "type": "bar",
                            "x": bin_centers,
                            "y": hist_vals.tolist(),
                            "marker": {"color": "#4a90d9", "opacity": 0.7},
                        }
                    ],
                    "layout": {
                        "height": 250,
                        "xaxis": {"title": "Residual"},
                        "yaxis": {"title": "Frequency"},
                    },
                }
            )

            # 4. Residuals vs Observation Order
            result["plots"].append(
                {
                    "title": "Residuals vs Observation Order",
                    "data": [
                        {
                            "x": list(range(1, len(resid_vals) + 1)),
                            "y": resid_vals.tolist(),
                            "mode": "lines+markers",
                            "type": "scatter",
                            "marker": {"color": "#4a9f6e", "size": 3},
                            "line": {"color": "#4a9f6e", "width": 1},
                        }
                    ],
                    "layout": {
                        "height": 250,
                        "xaxis": {"title": "Observation Order"},
                        "yaxis": {"title": "Residual"},
                        "shapes": [
                            {
                                "type": "line",
                                "x0": 1,
                                "x1": len(resid_vals),
                                "y0": 0,
                                "y1": 0,
                                "line": {"color": "#e89547", "dash": "dash"},
                            }
                        ],
                    },
                }
            )

            # ── Main Effects Plots ──
            if fixed_factors:
                for factor in fixed_factors:
                    levels = sorted(data[factor].unique().tolist(), key=str)
                    means = [
                        float(data[data[factor] == lev][response].mean())
                        for lev in levels
                    ]
                    cis = [
                        1.96
                        * float(data[data[factor] == lev][response].std())
                        / np.sqrt(len(data[data[factor] == lev]))
                        for lev in levels
                    ]
                    grand_mean = float(data[response].mean())
                    result["plots"].append(
                        {
                            "title": f"Main Effects Plot: {factor}",
                            "data": [
                                {
                                    "x": [str(lvl) for lvl in levels],
                                    "y": means,
                                    "error_y": {
                                        "type": "data",
                                        "array": cis,
                                        "visible": True,
                                        "color": "#4a90d9",
                                    },
                                    "mode": "lines+markers",
                                    "type": "scatter",
                                    "marker": {"color": "#4a90d9", "size": 8},
                                    "line": {"color": "#4a90d9", "width": 2},
                                    "name": "Mean",
                                }
                            ],
                            "layout": {
                                "height": 260,
                                "xaxis": {"title": factor},
                                "yaxis": {"title": f"Mean of {response}"},
                                "shapes": [
                                    {
                                        "type": "line",
                                        "x0": -0.5,
                                        "x1": len(levels) - 0.5,
                                        "y0": grand_mean,
                                        "y1": grand_mean,
                                        "line": {
                                            "color": "#999",
                                            "dash": "dash",
                                            "width": 1,
                                        },
                                    }
                                ],
                            },
                        }
                    )

            # ── Interaction Plots (Factor × Factor) ──
            if len(fixed_factors) >= 2:
                for i in range(len(fixed_factors)):
                    for j in range(i + 1, len(fixed_factors)):
                        f1, f2 = fixed_factors[i], fixed_factors[j]
                        traces = []
                        colors_int = [
                            "#4a90d9",
                            "#d94a4a",
                            "#4a9f6e",
                            "#d9a04a",
                            "#9b59b6",
                            "#e67e22",
                        ]
                        f2_levels = sorted(data[f2].unique().tolist(), key=str)
                        f1_levels = sorted(data[f1].unique().tolist(), key=str)
                        for ci, lev2 in enumerate(f2_levels):
                            sub = data[data[f2] == lev2]
                            means_int = [
                                (
                                    float(sub[sub[f1] == lev1][response].mean())
                                    if len(sub[sub[f1] == lev1]) > 0
                                    else None
                                )
                                for lev1 in f1_levels
                            ]
                            traces.append(
                                {
                                    "x": [str(lvl) for lvl in f1_levels],
                                    "y": means_int,
                                    "mode": "lines+markers",
                                    "name": f"{f2}={lev2}",
                                    "marker": {
                                        "color": colors_int[ci % len(colors_int)],
                                        "size": 7,
                                    },
                                    "line": {
                                        "color": colors_int[ci % len(colors_int)],
                                        "width": 2,
                                    },
                                }
                            )
                        result["plots"].append(
                            {
                                "title": f"Interaction Plot: {f1} × {f2}",
                                "data": traces,
                                "layout": {
                                    "height": 280,
                                    "xaxis": {"title": f1},
                                    "yaxis": {"title": f"Mean of {response}"},
                                },
                            }
                        )

            # ── ANCOVA: Covariate scatter by factor ──
            if has_factors and has_covariates:
                for c in covariates:
                    for f in fixed_factors:
                        traces_cov = []
                        colors_cov = [
                            "#4a90d9",
                            "#d94a4a",
                            "#4a9f6e",
                            "#d9a04a",
                            "#9b59b6",
                        ]
                        f_levels = sorted(data[f].unique().tolist(), key=str)
                        for fi, lev in enumerate(f_levels):
                            sub = data[data[f] == lev]
                            traces_cov.append(
                                {
                                    "x": sub[c].tolist(),
                                    "y": sub[response].tolist(),
                                    "mode": "markers",
                                    "name": f"{f}={lev}",
                                    "marker": {
                                        "color": colors_cov[fi % len(colors_cov)],
                                        "size": 5,
                                        "opacity": 0.7,
                                    },
                                }
                            )
                            # Add regression line per group
                            if len(sub) > 2:
                                slope, intercept, _, _, _ = qstats.linregress(
                                    sub[c].values, sub[response].values
                                )
                                x_line = [float(sub[c].min()), float(sub[c].max())]
                                y_line = [intercept + slope * x for x in x_line]
                                traces_cov.append(
                                    {
                                        "x": x_line,
                                        "y": y_line,
                                        "mode": "lines",
                                        "name": f"{lev} fit",
                                        "line": {
                                            "color": colors_cov[fi % len(colors_cov)],
                                            "width": 1.5,
                                            "dash": "dash",
                                        },
                                        "showlegend": False,
                                    }
                                )
                        result["plots"].append(
                            {
                                "title": f"Covariate Plot: {response} vs {c} by {f}",
                                "data": traces_cov,
                                "layout": {
                                    "height": 300,
                                    "xaxis": {"title": c},
                                    "yaxis": {"title": response},
                                },
                            }
                        )

            result["guide_observation"] = (
                f"{model_label}: {response} ~ {' + '.join(fixed_factors + covariates + random_factors)}. N={N}."
            )

            # Narrative
            try:
                _r2 = float(model.rsquared)
                _f_p = float(model.f_pvalue)
                _sig_terms = [
                    t
                    for t in model.pvalues.index
                    if model.pvalues[t] < 0.05 and t != "Intercept"
                ]
                _r2_pct = _r2 * 100
                _r2_label = (
                    "strong" if _r2 > 0.7 else "moderate" if _r2 > 0.3 else "weak"
                )
                verdict = f"{model_label}: R² = {_r2:.3f} ({_r2_label} fit, {_r2_pct:.1f}% variance explained)"
                body = (
                    f"The model {response} ~ {' + '.join(fixed_factors + covariates + random_factors)} "
                    f"is {'significant' if _f_p < 0.05 else 'not significant'} overall (F p = {_f_p:.4f}). "
                )
                if _sig_terms:
                    body += f"Significant terms: <strong>{', '.join(str(t) for t in _sig_terms[:5])}</strong>."
                else:
                    body += "No individual terms are significant at α = 0.05."
                nxt = "Check the residual plots for model adequacy: random scatter = good fit; patterns = missing terms or non-linearity."
                result["narrative"] = _narrative(
                    verdict,
                    body,
                    next_steps=nxt,
                    chart_guidance="Residuals vs Fitted: random scatter = adequate model. Normal Q-Q: points on diagonal = normally distributed residuals.",
                )
            except Exception:
                pass

            # ── Diagnostics ──
            diagnostics = []
            # Normality of residuals
            try:
                _norm_r = _check_normality(
                    resid_vals.values, label="Model residuals", alpha=0.05
                )
                if _norm_r:
                    _norm_r["detail"] = (
                        "GLM assumes normality of residuals. Non-normality may affect confidence intervals and p-values."
                    )
                    diagnostics.append(_norm_r)
            except Exception:
                pass
            # Check deviance residuals for patterns (mean should be ~0, look for systematic bias)
            try:
                _resid_mean = float(np.mean(resid_vals.values))
                _resid_std = float(np.std(resid_vals.values))
                # Check for skewness in residuals (pattern indicator)
                from scipy.stats import skew as _skew_fn

                _resid_skew = float(_skew_fn(resid_vals.values))
                if abs(_resid_skew) > 1.0:
                    diagnostics.append(
                        {
                            "level": "warning",
                            "title": f"Residuals are skewed (skewness = {_resid_skew:.2f})",
                            "detail": "Skewed residuals suggest the model may be missing non-linear terms or the response needs transformation.",
                        }
                    )
            except Exception:
                pass
            # Report pseudo-R² / R² and model fit assessment
            try:
                if has_random:
                    # Mixed model: use marginal R² approximation
                    _ss_resid_m = float(np.sum(resid_vals.values**2))
                    _ss_total_m = float(
                        np.sum((data[response].values - data[response].mean()) ** 2)
                    )
                    _pseudo_r2 = 1 - _ss_resid_m / _ss_total_m if _ss_total_m > 0 else 0
                    if _pseudo_r2 > 0.7:
                        diagnostics.append(
                            {
                                "level": "info",
                                "title": f"Good model fit (marginal R² ≈ {_pseudo_r2:.3f})",
                                "detail": f"The model explains approximately {_pseudo_r2 * 100:.1f}% of the total variation.",
                            }
                        )
                    elif _pseudo_r2 < 0.1:
                        diagnostics.append(
                            {
                                "level": "warning",
                                "title": f"Weak model (marginal R² ≈ {_pseudo_r2:.3f})",
                                "detail": f"The model explains only {_pseudo_r2 * 100:.1f}% of variation. Consider adding predictors or checking model specification.",
                            }
                        )
                else:
                    # Fixed model: use R² from OLS
                    _r2_val = float(model.rsquared)
                    _f_pval = float(model.f_pvalue)
                    if _r2_val > 0.7 and _f_pval < 0.05:
                        diagnostics.append(
                            {
                                "level": "info",
                                "title": f"Strong model (R² = {_r2_val:.3f})",
                                "detail": f"The model explains {_r2_val * 100:.0f}% of the variation — practically useful for prediction.",
                            }
                        )
                    elif _r2_val < 0.1 and _f_pval < 0.05:
                        diagnostics.append(
                            {
                                "level": "warning",
                                "title": f"Significant but weak model (R² = {_r2_val:.3f})",
                                "detail": f"The model is statistically significant but explains only {_r2_val * 100:.0f}% of the variation. Not useful for prediction.",
                            }
                        )
                    elif _r2_val < 0.1 and _f_pval >= 0.05:
                        diagnostics.append(
                            {
                                "level": "warning",
                                "title": f"Weak and non-significant model (R² = {_r2_val:.3f}, F p = {_f_pval:.4f})",
                                "detail": "The model barely reduces deviance from the null model. Consider different predictors or a simpler model.",
                            }
                        )
            except Exception:
                pass
            # Overdispersion check (residual deviance / df >> 1)
            try:
                if not has_random:
                    _resid_dev = float(np.sum(resid_vals.values**2))
                    _df_resid = N - len(model.params)
                    _disp_ratio = _resid_dev / _df_resid if _df_resid > 0 else 1.0
                    if _disp_ratio > 2.0:
                        diagnostics.append(
                            {
                                "level": "warning",
                                "title": f"Possible overdispersion (residual deviance/df = {_disp_ratio:.2f})",
                                "detail": "The residual variance is much larger than expected. If using count or proportion data, consider a quasi-likelihood approach or negative binomial model.",
                            }
                        )
            except Exception:
                pass
            # Effect size emphasis from ANOVA table (partial eta-squared for significant effects)
            try:
                if not has_random and has_factors:
                    for idx in anova_table.index:
                        if idx in ("Residual", "Intercept"):
                            continue
                        if (
                            "PR(>F)" in anova_table.columns
                            and "sum_sq" in anova_table.columns
                        ):
                            _pv = (
                                float(anova_table.loc[idx, "PR(>F)"])
                                if not np.isnan(anova_table.loc[idx, "PR(>F)"])
                                else 1.0
                            )
                            _ss_eff = float(anova_table.loc[idx, "sum_sq"])
                            _eta_p = (
                                _ss_eff / (_ss_eff + ss_residual)
                                if (_ss_eff + ss_residual) > 0
                                else 0
                            )
                            if _eta_p > 0.14 and _pv < 0.05:
                                diagnostics.append(
                                    {
                                        "level": "info",
                                        "title": f"Large effect: {idx} (η²p = {_eta_p:.3f})",
                                        "detail": f"{idx} explains {_eta_p * 100:.1f}% of remaining variation — a practically important effect.",
                                    }
                                )
            except Exception:
                pass
            result["diagnostics"] = diagnostics

        except Exception as e:
            result["summary"] = f"GLM error: {str(e)}"

    elif analysis_id == "ordinal_logistic":
        """
        Ordinal Logistic Regression — proportional odds model for ordered categorical outcomes.
        Uses statsmodels OrderedModel.
        """
        response = config.get("response") or config.get("var")
        predictors = config.get("predictors", [])
        if not predictors and config.get("predictor"):
            predictors = [config["predictor"]]

        try:
            from statsmodels.miscmodels.ordinal_model import OrderedModel

            all_cols = [response] + predictors
            data = df[all_cols].dropna()
            N = len(data)

            # Encode response as ordered categorical
            categories = sorted(data[response].unique().tolist(), key=str)
            cat_map = {c: i for i, c in enumerate(categories)}
            data["_y_ordinal"] = data[response].map(cat_map)

            # Fit proportional odds model
            X = data[predictors].values.astype(float)
            y = data["_y_ordinal"].values

            model = OrderedModel(y, X, distr="logit")
            fit = model.fit(method="bfgs", disp=False)

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += "<<COLOR:title>>ORDINAL LOGISTIC REGRESSION (Proportional Odds)<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += f"<<COLOR:highlight>>Response:<</COLOR>> {response} ({len(categories)} ordered levels)\n"
            summary_text += f"<<COLOR:highlight>>Levels:<</COLOR>> {' < '.join(str(c) for c in categories)}\n"
            summary_text += (
                f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
            )
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n\n"

            summary_text += "<<COLOR:accent>>── Coefficients ──<</COLOR>>\n"
            summary_text += f"{'Parameter':<25} {'Coef':>10} {'SE':>10} {'z':>8} {'p-value':>10} {'OR':>10} {'95% CI (OR)':>20}\n"
            summary_text += f"{'─' * 97}\n"

            param_names = list(predictors) + [
                f"threshold_{i}" for i in range(len(categories) - 1)
            ]
            for i, name in enumerate(param_names):
                if i < len(fit.params):
                    coef = float(fit.params[i])
                    se = float(fit.bse[i]) if i < len(fit.bse) else None
                    pv = float(fit.pvalues[i]) if i < len(fit.pvalues) else None
                    z = coef / se if se and se > 0 else 0
                    odds_ratio = np.exp(coef) if i < len(predictors) else None
                    se_str = f"{se:.4f}" if se else "N/A"
                    p_str = f"{pv:.4f}" if pv else "N/A"
                    or_str = f"{odds_ratio:.4f}" if odds_ratio else ""
                    ci_str = (
                        f"[{np.exp(coef - 1.96 * se):.4f}, {np.exp(coef + 1.96 * se):.4f}]"
                        if odds_ratio and se
                        else ""
                    )
                    summary_text += f"{name:<25} {coef:>10.4f} {se_str:>10} {z:>8.2f} {p_str:>10} {or_str:>10} {ci_str:>20}\n"

            if hasattr(fit, "llf"):
                summary_text += (
                    f"\n<<COLOR:accent>>── Log-Likelihood ──<</COLOR>> {fit.llf:.2f}\n"
                )
            if hasattr(fit, "aic"):
                summary_text += f"<<COLOR:accent>>── AIC ──<</COLOR>> {fit.aic:.2f}\n"

            result["summary"] = summary_text

            # Predicted probabilities for first predictor
            if len(predictors) >= 1:
                x_range = np.linspace(
                    float(data[predictors[0]].min()),
                    float(data[predictors[0]].max()),
                    100,
                )
                X_pred = np.zeros((100, len(predictors)))
                X_pred[:, 0] = x_range
                for j in range(1, len(predictors)):
                    X_pred[:, j] = data[predictors[j]].mean()

                pred_probs = fit.predict(X_pred)
                traces = []
                colors_cat = ["#4a90d9", "#d9a04a", "#4a9f6e", "#d94a4a", "#9b59b6"]
                for ci, cat in enumerate(categories):
                    col_idx = (
                        ci if ci < pred_probs.shape[1] else pred_probs.shape[1] - 1
                    )
                    traces.append(
                        {
                            "x": x_range.tolist(),
                            "y": pred_probs[:, col_idx].tolist(),
                            "mode": "lines",
                            "name": str(cat),
                            "line": {
                                "color": colors_cat[ci % len(colors_cat)],
                                "width": 2,
                            },
                        }
                    )
                result["plots"].append(
                    {
                        "title": f"Predicted Probabilities by {predictors[0]}",
                        "data": traces,
                        "layout": {
                            "height": 300,
                            "xaxis": {"title": predictors[0]},
                            "yaxis": {"title": "Probability"},
                        },
                    }
                )

            result["statistics"] = {
                "n": N,
                "n_categories": len(categories),
                "log_likelihood": float(fit.llf) if hasattr(fit, "llf") else None,
                "aic": float(fit.aic) if hasattr(fit, "aic") else None,
            }
            result["guide_observation"] = (
                f"Ordinal logistic: {response} ({len(categories)} levels) ~ {', '.join(predictors)}. N={N}."
            )
            result["narrative"] = _narrative(
                f"Ordinal Logistic: {response} ({len(categories)} ordered levels)",
                f"Predicting ordered outcome using {len(predictors)} predictor{'s' if len(predictors) > 1 else ''}. N = {N}.",
                next_steps="Coefficients represent log odds of being in a higher category. Check the proportional odds assumption.",
            )

        except ImportError:
            result["summary"] = (
                "Ordinal logistic requires statsmodels >= 0.13. Install with: pip install --upgrade statsmodels"
            )
        except Exception as e:
            result["summary"] = f"Ordinal logistic error: {str(e)}"

    # ── Data Profile ──────────────────────────────────────────────────────

    return result

"""Forge-backed advanced analysis handlers: survival, time series, MSA.

Split from forge_stats.py for compliance (3000-line limit).
Object 271 — Analysis Workbench migration.
"""

import logging

import numpy as np
import pandas as pd

from .forge_stats import _col, _col2, _pval_str

logger = logging.getLogger(__name__)


# =============================================================================
# Survival Analysis
# =============================================================================


def forge_kaplan_meier(df, config):
    """Kaplan-Meier survival estimator via forgestat."""
    from forgestat.reliability.survival import kaplan_meier, log_rank_test

    time_col = config.get("time_col") or config.get("column") or config.get("var1")
    event_col = config.get("event_col") or config.get("var2") or config.get("event")
    group_col = config.get("group") or config.get("factor")

    if not time_col or time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found")

    times = pd.to_numeric(df[time_col], errors="coerce").dropna()
    events = None
    if event_col and event_col in df.columns:
        events = df.loc[times.index, event_col].astype(bool).values
    times = times.values

    result = kaplan_meier(times, events)

    stats = {
        "median_survival": result.median_survival,
        "mean_survival": round(result.mean_survival, 4) if result.mean_survival else None,
        "n_total": result.n_total,
        "n_events": result.n_events,
        "n_censored": result.n_censored,
    }

    # Log-rank test if groups provided
    log_rank = None
    if group_col and group_col in df.columns:
        groups = df[group_col].dropna().unique()
        if len(groups) == 2:
            mask1 = df[group_col] == groups[0]
            mask2 = df[group_col] == groups[1]
            t1 = pd.to_numeric(df.loc[mask1, time_col], errors="coerce").dropna().values
            t2 = pd.to_numeric(df.loc[mask2, time_col], errors="coerce").dropna().values
            e1 = (
                df.loc[mask1 & df.index.isin(pd.to_numeric(df[time_col], errors="coerce").dropna().index), event_col]
                .astype(bool)
                .values
                if event_col and event_col in df.columns
                else np.ones(len(t1), dtype=bool)
            )
            e2 = (
                df.loc[mask2 & df.index.isin(pd.to_numeric(df[time_col], errors="coerce").dropna().index), event_col]
                .astype(bool)
                .values
                if event_col and event_col in df.columns
                else np.ones(len(t2), dtype=bool)
            )
            # Align lengths
            min_len1 = min(len(t1), len(e1))
            min_len2 = min(len(t2), len(e2))
            try:
                lr = log_rank_test(
                    t1[:min_len1],
                    e1[:min_len1],
                    t2[:min_len2],
                    e2[:min_len2],
                    group_names=(str(groups[0]), str(groups[1])),
                )
                log_rank = lr
                stats["log_rank_chi2"] = round(lr.chi_square, 4)
                stats["log_rank_p"] = round(lr.p_value, 6)
            except Exception:
                pass

    median_str = f"{result.median_survival:.2f}" if result.median_survival else "not reached"
    sig_str = ""
    if log_rank:
        sig_str = f"\n<<COLOR:text>>Log-rank test: \u03c7\u00b2 = {log_rank.chi_square:.3f}, p = {_pval_str(log_rank.p_value)}<</COLOR>>"

    return {
        "plots": [],
        "statistics": stats,
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Kaplan-Meier Survival Analysis<</COLOR>>\n\n"
            f"<<COLOR:text>>N = {result.n_total} ({result.n_events} events, {result.n_censored} censored)<</COLOR>>\n"
            f"<<COLOR:text>>Median survival: {median_str}<</COLOR>>\n"
            f"<<COLOR:text>>Mean survival: {result.mean_survival:.2f}<</COLOR>>" + sig_str
        ),
        "narrative": {
            "verdict": f"Median survival = {median_str}",
            "body": (
                f"Kaplan-Meier analysis of {result.n_total} subjects: {result.n_events} events observed, "
                f"{result.n_censored} censored. Median survival time {'= ' + f'{result.median_survival:.2f}' if result.median_survival else 'not reached'}."
            ),
            "next_steps": "Consider Cox PH regression if covariates are available.",
            "chart_guidance": "Step function shows estimated survival probability over time. Wider CI bands indicate fewer subjects at risk.",
        },
        "guide_observation": f"KM: median={median_str}, n={result.n_total}, events={result.n_events}.",
        "diagnostics": [],
    }


def forge_cox_ph(df, config):
    """Cox proportional hazards regression via forgestat."""
    from forgestat.reliability.cox import cox_ph

    time_col = config.get("time_col") or config.get("column") or config.get("var1")
    event_col = config.get("event_col") or config.get("var2") or config.get("event")
    covariate_cols = config.get("covariates", [])

    if not time_col or time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found")

    clean = df[
        [time_col] + ([event_col] if event_col and event_col in df.columns else []) + list(covariate_cols)
    ].dropna()
    times = pd.to_numeric(clean[time_col], errors="coerce").dropna()
    clean = clean.loc[times.index]
    times = times.values

    events = (
        clean[event_col].astype(bool).values
        if event_col and event_col in clean.columns
        else np.ones(len(times), dtype=bool)
    )

    covariates = {}
    for col in covariate_cols:
        if col in clean.columns:
            covariates[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0).values.tolist()

    if not covariates:
        raise ValueError("No valid covariates for Cox PH model")

    result = cox_ph(times, events, covariates)

    stats = {
        "concordance": round(result.concordance, 4),
        "log_likelihood": round(result.log_likelihood, 4),
        "n": result.n,
        "n_events": result.n_events,
    }
    for name in covariates.keys():
        stats[f"hr_{name}"] = round(result.hazard_ratios.get(name, 0), 4)
        stats[f"p_{name}"] = round(result.p_values.get(name, 1), 6)
        stats[f"coef_{name}"] = round(result.coefficients.get(name, 0), 4)

    covar_lines = []
    for name in covariates.keys():
        hr = result.hazard_ratios.get(name, 0)
        p = result.p_values.get(name, 1)
        covar_lines.append(f"  <<COLOR:highlight>>{name}:<</COLOR>> HR = {hr:.3f}, p = {_pval_str(p)}")

    return {
        "plots": [],
        "statistics": stats,
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Cox Proportional Hazards Regression<</COLOR>>\n\n"
            f"<<COLOR:text>>N = {result.n} ({result.n_events} events), Concordance = {result.concordance:.3f}<</COLOR>>\n\n"
            + "\n".join(covar_lines)
        ),
        "narrative": {
            "verdict": f"Cox PH: concordance = {result.concordance:.3f}",
            "body": (
                f"Cox regression on {len(covariates)} covariates. "
                f"Concordance index = {result.concordance:.3f} (1.0 = perfect discrimination)."
            ),
            "next_steps": "Check proportional hazards assumption with Schoenfeld residuals.",
            "chart_guidance": "Hazard ratios > 1 indicate increased risk; < 1 indicate protective effect.",
        },
        "guide_observation": f"Cox PH: C={result.concordance:.3f}, {len(covariates)} covariates, n={result.n}.",
        "diagnostics": [],
    }


def forge_weibull(df, config):
    """Weibull reliability fit via forgestat."""
    from forgestat.reliability.distributions import weibull_fit

    data, col_name = _col(df, config, "column", "var1")
    censored = None
    censor_col = config.get("censor_col") or config.get("event_col")
    if censor_col and censor_col in df.columns:
        censored = (~df[censor_col].astype(bool)).values[: len(data)]

    result = weibull_fit(data.tolist(), censored=censored.tolist() if censored is not None else None)

    failure_mode = result.failure_mode or "unknown"
    return {
        "plots": [],
        "statistics": {
            "shape": round(result.shape, 4),
            "scale": round(result.scale, 4),
            "location": round(result.location, 4) if result.location else 0,
            "b10_life": round(result.b10_life, 4) if result.b10_life else None,
            "mean_life": round(result.mean_life, 4) if result.mean_life else None,
            "median_life": round(result.median_life, 4) if result.median_life else None,
            "ks_statistic": round(result.ks_statistic, 4) if result.ks_statistic else None,
            "ks_p_value": round(result.ks_p_value, 6) if result.ks_p_value else None,
            "failure_mode": failure_mode,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Weibull Reliability Analysis<</COLOR>>\n\n"
            f"<<COLOR:text>>Shape (\u03b2) = {result.shape:.3f}, Scale (\u03b7) = {result.scale:.3f}<</COLOR>>\n"
            f"<<COLOR:text>>B10 life = {result.b10_life:.2f}, Mean life = {result.mean_life:.2f}<</COLOR>>\n"
            f"<<COLOR:text>>Failure mode: {failure_mode}<</COLOR>>\n"
            + (
                f"<<COLOR:text>>KS test: D = {result.ks_statistic:.4f}, p = {_pval_str(result.ks_p_value)}<</COLOR>>"
                if result.ks_statistic
                else ""
            )
        ),
        "narrative": {
            "verdict": f"Weibull: \u03b2 = {result.shape:.3f}, \u03b7 = {result.scale:.3f} ({failure_mode})",
            "body": (
                f"Weibull fit: shape \u03b2 = {result.shape:.3f} "
                f"({'infant mortality' if result.shape < 1 else 'random failures' if abs(result.shape - 1) < 0.1 else 'wear-out'}), "
                f"scale \u03b7 = {result.scale:.3f}. B10 life = {result.b10_life:.2f}."
            ),
            "next_steps": "Compare with lognormal or exponential fit. Use B10 for warranty/maintenance planning.",
            "chart_guidance": "Probability plot should show points along a straight line for good fit.",
        },
        "guide_observation": f"Weibull: \u03b2={result.shape:.3f}, \u03b7={result.scale:.3f}, B10={result.b10_life:.2f}.",
        "diagnostics": [],
    }


# =============================================================================
# Time Series
# =============================================================================


def forge_acf_pacf(df, config):
    """ACF/PACF analysis via forgestat."""
    from forgestat.timeseries.correlation import acf_pacf

    data, col_name = _col(df, config, "column", "var1")
    n_lags = int(config.get("max_lags", config.get("n_lags", min(40, len(data) // 2 - 1))))

    result = acf_pacf(data.tolist(), n_lags=n_lags)

    n_sig_acf = len(result.significant_acf_lags) if result.significant_acf_lags else 0
    n_sig_pacf = len(result.significant_pacf_lags) if result.significant_pacf_lags else 0
    stats = {
        "n": len(data),
        "n_lags": n_lags,
        "n_significant_acf": n_sig_acf,
        "n_significant_pacf": n_sig_pacf,
        "confidence_bound": round(result.confidence_bound, 4),
        "suggested_order": result.suggested_order,
        "ljung_box_p": round(result.ljung_box_p, 6) if result.ljung_box_p else None,
    }

    return {
        "plots": [],
        "statistics": stats,
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>ACF / PACF Analysis<</COLOR>>\n\n"
            f"<<COLOR:text>>N = {len(data)}, lags = {n_lags}<</COLOR>>\n"
            f"<<COLOR:text>>Significant ACF lags: {n_sig_acf}, PACF lags: {n_sig_pacf}<</COLOR>>\n"
            f"<<COLOR:text>>Suggested ARIMA order: {result.suggested_order}<</COLOR>>\n"
            f"<<COLOR:text>>95% confidence bound: \u00b1{result.confidence_bound:.4f}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"ACF: {n_sig_acf} significant lags, PACF: {n_sig_pacf} significant lags",
            "body": (
                f"ACF shows {n_sig_acf} lags outside the 95% band — "
                f"{'slow decay suggests non-stationarity or MA component' if n_sig_acf > 5 else 'few significant lags suggest weak autocorrelation'}. "
                f"PACF has {n_sig_pacf} significant lags — "
                f"{'suggests AR({}) component'.format(n_sig_pacf) if n_sig_pacf <= 3 else 'complex structure'}. "
                f"Suggested ARIMA order: {result.suggested_order}."
            ),
            "next_steps": "Use ACF/PACF patterns to select ARIMA(p,d,q) orders.",
            "chart_guidance": "Bars outside the shaded band are statistically significant at 95%.",
        },
        "guide_observation": f"ACF/PACF: {n_sig_acf} sig ACF, {n_sig_pacf} sig PACF. Suggested order: {result.suggested_order}.",
        "diagnostics": [],
    }


def forge_ccf(df, config):
    """Cross-correlation function via forgestat."""
    from forgestat.timeseries.correlation import cross_correlation

    c1, n1, c2, n2 = _col2(df, config)
    max_lags = int(config.get("max_lags", min(40, len(c1) // 2 - 1)))

    result = cross_correlation(c1.tolist(), c2.tolist(), max_lag=max_lags)

    peak_lag = result.peak_lag
    peak_val = result.peak_value

    return {
        "plots": [],
        "statistics": {
            "n": len(c1),
            "max_lags": max_lags,
            "peak_lag": peak_lag,
            "peak_ccf": round(float(peak_val), 4),
            "confidence_bound": round(result.confidence_bound, 4),
            "lead_lag": result.lead_lag_interpretation,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Cross-Correlation Function<</COLOR>>\n\n"
            f"<<COLOR:text>>{n1} vs {n2}, N = {len(c1)}<</COLOR>>\n"
            f"<<COLOR:text>>Peak CCF = {peak_val:.4f} at lag {peak_lag}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Peak cross-correlation at lag {peak_lag} (CCF = {peak_val:.4f})",
            "body": f"Strongest relationship between {n1} and {n2} occurs at lag {peak_lag}.",
            "next_steps": "Positive lag means the first series leads. Consider Granger causality for formal testing.",
            "chart_guidance": "Bars outside the shaded band are significant at 95%.",
        },
        "guide_observation": f"CCF: peak={peak_val:.4f} at lag {peak_lag}.",
        "diagnostics": [],
    }


def forge_decomposition(df, config):
    """Classical time series decomposition via forgestat."""
    from forgestat.timeseries.decomposition import classical_decompose

    data, col_name = _col(df, config, "column", "var1")
    period = int(config.get("period", 12))
    model = config.get("model", "additive")

    result = classical_decompose(data.tolist(), period=period, model=model)

    return {
        "plots": [],
        "statistics": {
            "n": len(data),
            "period": period,
            "model": model,
            "trend_strength": round(result.trend_strength, 4)
            if hasattr(result, "trend_strength") and result.trend_strength
            else None,
            "seasonal_strength": round(result.seasonal_strength, 4)
            if hasattr(result, "seasonal_strength") and result.seasonal_strength
            else None,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Time Series Decomposition ({model.title()})<</COLOR>>\n\n"
            f"<<COLOR:text>>N = {len(data)}, period = {period}<</COLOR>>\n"
            f"<<COLOR:text>>Components: trend, seasonal, residual<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"{model.title()} decomposition with period {period}",
            "body": f"Decomposed {len(data)} observations into trend, seasonal (period={period}), and residual components using {model} model.",
            "next_steps": "Examine residuals for patterns. If residuals show structure, consider a different model or period.",
            "chart_guidance": "Four panels: observed, trend, seasonal, residual. Residuals should look random.",
        },
        "guide_observation": f"Decomposition: {model}, period={period}, n={len(data)}.",
        "diagnostics": [],
    }


def forge_granger(df, config):
    """Granger causality test via forgestat."""
    from forgestat.timeseries.causality import granger_causality

    c1, n1, c2, n2 = _col2(df, config)
    max_lag = int(config.get("max_lags", config.get("max_lag", 5)))

    result = granger_causality(c1.tolist(), c2.tolist(), max_lag=max_lag)

    stats = {
        "max_lag": max_lag,
        "n": len(c1),
        "x_causes_y": result.x_causes_y,
        "best_lag": result.best_lag,
        "best_p_value": round(result.best_p_value, 6) if result.best_p_value else None,
    }
    sig_lags = []
    for lr in result.results_by_lag:
        lag = lr["lag"]
        stats[f"f_stat_lag{lag}"] = round(lr["f_stat"], 4)
        stats[f"p_value_lag{lag}"] = round(lr["p_value"], 6)
        if lr["p_value"] < 0.05:
            sig_lags.append(lag)

    return {
        "plots": [],
        "statistics": stats,
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Granger Causality Test<</COLOR>>\n\n"
            f"<<COLOR:text>>{n2} \u2192 {n1} (does {n2} Granger-cause {n1}?)<</COLOR>>\n"
            f"<<COLOR:text>>Tested lags: 1\u2013{max_lag}<</COLOR>>\n"
            f"<<COLOR:text>>Significant at lags: {sig_lags if sig_lags else 'none'}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Granger causality {'detected' if sig_lags else 'not detected'} (lags {sig_lags})"
            if sig_lags
            else "No Granger causality detected",
            "body": (
                f"Testing whether past values of {n2} improve prediction of {n1}. "
                + (f"Significant at lag(s) {sig_lags} (p < 0.05)." if sig_lags else "No significant lags found.")
            ),
            "next_steps": "Granger causality is predictive, not true causation. Consider VAR models for multivariate analysis.",
            "chart_guidance": "F-statistics per lag. Higher values indicate stronger predictive relationship.",
        },
        "guide_observation": f"Granger: {n2}\u2192{n1}, sig lags={sig_lags or 'none'}.",
        "diagnostics": [],
    }


def forge_changepoint(df, config):
    """Changepoint detection via forgestat."""
    from forgestat.timeseries.changepoint import pelt

    data, col_name = _col(df, config, "column", "var1")
    penalty = config.get("penalty", "bic")

    result = pelt(data.tolist(), penalty=penalty)

    cps = [cp.index for cp in result.changepoints] if result.changepoints else []
    n_changes = len(cps)

    return {
        "plots": [],
        "statistics": {
            "n": len(data),
            "n_changepoints": n_changes,
            "changepoint_indices": cps,
            "penalty": penalty,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Changepoint Detection (PELT)<</COLOR>>\n\n"
            f"<<COLOR:text>>N = {len(data)}, penalty = {penalty}<</COLOR>>\n"
            f"<<COLOR:text>>Changepoints detected: {n_changes}<</COLOR>>\n"
            + (f"<<COLOR:text>>At indices: {cps}<</COLOR>>" if cps else "")
        ),
        "narrative": {
            "verdict": f"{n_changes} changepoint{'s' if n_changes != 1 else ''} detected",
            "body": (
                f"PELT algorithm found {n_changes} changepoint(s) in {len(data)} observations"
                + (f" at positions {cps}." if cps else ".")
            ),
            "next_steps": "Investigate what changed at each changepoint. Consider BOCPD for online detection.",
            "chart_guidance": "Vertical lines mark detected changepoints. Segments between them are stationary regimes.",
        },
        "guide_observation": f"Changepoints: {n_changes} detected at {cps}.",
        "diagnostics": [],
    }


def forge_arima(df, config):
    """ARIMA forecasting via forgestat."""
    from forgestat.timeseries.forecasting import arima

    data, col_name = _col(df, config, "column", "var1")
    order = config.get("order", [1, 1, 1])
    if isinstance(order, str):
        order = [int(x) for x in order.split(",")]
    forecast_steps = int(config.get("forecast_steps", config.get("h", 10)))

    result = arima(data.tolist(), order=tuple(order), forecast_steps=forecast_steps)

    stats = {
        "order": list(order),
        "n": len(data),
        "forecast_steps": forecast_steps,
        "aic": round(result.aic, 2) if result.aic else None,
        "bic": round(result.bic, 2) if result.bic else None,
    }
    if result.ljung_box_p is not None:
        stats["ljung_box_p"] = round(result.ljung_box_p, 6)
    if result.adf_p_value is not None:
        stats["adf_p_value"] = round(result.adf_p_value, 6)
    stats["is_stationary"] = result.is_stationary

    return {
        "plots": [],
        "statistics": stats,
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>ARIMA({order[0]},{order[1]},{order[2]}) Forecast<</COLOR>>\n\n"
            f"<<COLOR:text>>N = {len(data)}, forecast horizon = {forecast_steps}<</COLOR>>\n"
            + (f"<<COLOR:text>>AIC = {result.aic:.2f}, BIC = {result.bic:.2f}<</COLOR>>\n" if result.aic else "")
        ),
        "narrative": {
            "verdict": f"ARIMA({order[0]},{order[1]},{order[2]}) fit, {forecast_steps}-step forecast",
            "body": f"Fitted ARIMA({order[0]},{order[1]},{order[2]}) to {len(data)} observations. Forecast {forecast_steps} steps ahead.",
            "next_steps": "Check residual ACF for remaining autocorrelation. Try different orders or SARIMA for seasonal data.",
            "chart_guidance": "Solid line = fitted/forecast, shaded area = prediction interval.",
        },
        "guide_observation": f"ARIMA({order[0]},{order[1]},{order[2]}): AIC={result.aic:.1f}, h={forecast_steps}."
        if result.aic
        else f"ARIMA fit, h={forecast_steps}.",
        "diagnostics": [],
    }


def forge_sarima(df, config):
    """Seasonal ARIMA forecasting via forgestat."""
    from forgestat.timeseries.forecasting import sarima

    data, col_name = _col(df, config, "column", "var1")
    order = config.get("order", [1, 1, 1])
    seasonal_order = config.get("seasonal_order", [1, 1, 1, 12])
    if isinstance(order, str):
        order = [int(x) for x in order.split(",")]
    if isinstance(seasonal_order, str):
        seasonal_order = [int(x) for x in seasonal_order.split(",")]
    forecast_steps = int(config.get("forecast_steps", config.get("h", 10)))

    result = sarima(
        data.tolist(), order=tuple(order), seasonal_order=tuple(seasonal_order), forecast_steps=forecast_steps
    )

    stats = {
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "n": len(data),
        "forecast_steps": forecast_steps,
        "aic": round(result.aic, 2) if result.aic else None,
        "bic": round(result.bic, 2) if result.bic else None,
    }

    so = seasonal_order
    return {
        "plots": [],
        "statistics": stats,
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>SARIMA({order[0]},{order[1]},{order[2]})({so[0]},{so[1]},{so[2]})[{so[3]}] Forecast<</COLOR>>\n\n"
            f"<<COLOR:text>>N = {len(data)}, forecast horizon = {forecast_steps}<</COLOR>>\n"
            + (f"<<COLOR:text>>AIC = {result.aic:.2f}, BIC = {result.bic:.2f}<</COLOR>>\n" if result.aic else "")
        ),
        "narrative": {
            "verdict": f"SARIMA fit with seasonal period {so[3]}, {forecast_steps}-step forecast",
            "body": (
                f"Fitted SARIMA({order[0]},{order[1]},{order[2]})({so[0]},{so[1]},{so[2]})[{so[3]}] "
                f"to {len(data)} observations."
            ),
            "next_steps": "Verify seasonal period matches domain knowledge. Check residual diagnostics.",
            "chart_guidance": "Solid line = fitted/forecast, shaded area = prediction interval.",
        },
        "guide_observation": f"SARIMA: AIC={result.aic:.1f}, period={so[3]}, h={forecast_steps}."
        if result.aic
        else f"SARIMA fit, period={so[3]}, h={forecast_steps}.",
        "diagnostics": [],
    }


# =============================================================================
# Dispatch
# =============================================================================

FORGE_ADVANCED_HANDLERS = {
    # Survival
    "kaplan_meier": forge_kaplan_meier,
    "cox_ph": forge_cox_ph,
    "weibull": forge_weibull,
    # Time series
    "acf_pacf": forge_acf_pacf,
    "ccf": forge_ccf,
    "decomposition": forge_decomposition,
    "granger": forge_granger,
    "changepoint": forge_changepoint,
    "arima": forge_arima,
    "sarima": forge_sarima,
}

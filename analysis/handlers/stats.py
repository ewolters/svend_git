"""Stats handler — routes 100+ statistical analyses through forgestat.

Generic pattern: parse config → call forgestat function → convert TestResult to output.
TestResult is standardized across all forgestat functions, so one converter handles everything.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np
import pandas as pd
from forgeviz.charts.distribution import box_plot, histogram
from forgeviz.charts.scatter import scatter

logger = logging.getLogger(__name__)


def _get_dispatch():
    """Build dispatch map: analysis_id → (module, function, parser).

    Parsers reuse a small set of patterns — most analyses need
    the same data extraction (one column, two columns, response+factor, etc.).
    """
    return {
        # ── Hypothesis Testing ───────────────────────────────────────
        "ttest": ("forgestat.parametric.ttest", "one_sample", _parse_one_sample),
        "ttest2": ("forgestat.parametric.ttest", "two_sample", _parse_two_sample),
        "paired_t": ("forgestat.parametric.ttest", "paired", _parse_paired),
        "anova": ("forgestat.parametric.anova", "one_way_from_dict", _parse_anova),
        "anova2": ("forgestat.parametric.anova", "two_way", _parse_two_way_anova),
        "repeated_measures_anova": (
            "forgestat.parametric.repeated_measures",
            "repeated_measures_anova",
            _parse_rm_anova,
        ),
        "nested_anova": ("forgestat.parametric.mixed", "nested_anova", _parse_nested),
        "split_plot_anova": ("forgestat.parametric.split_plot", "split_plot_anova", _parse_two_way_anova),
        "chi2": ("forgestat.parametric.chi_square", "chi_square_independence", _parse_chi2),
        "fisher_exact": ("forgestat.parametric.chi_square", "fisher_exact", _parse_chi2),
        "correlation": ("forgestat.parametric.correlation", "correlation", _parse_correlation),
        "equivalence": ("forgestat.parametric.equivalence", "tost", _parse_equivalence),
        "prop_1sample": ("forgestat.parametric.proportion", "one_proportion", _parse_one_prop),
        "prop_2sample": ("forgestat.parametric.proportion", "two_proportions", _parse_two_prop),
        "variance_test": ("forgestat.parametric.variance", "variance_test", _parse_two_sample),
        "f_test": ("forgestat.parametric.variance", "f_test", _parse_two_sample),
        "poisson_1sample": ("forgestat.parametric.proportion", "one_proportion", _parse_one_sample),
        "poisson_2sample": ("forgestat.parametric.proportion", "two_proportions", _parse_two_sample),
        "manova": ("forgestat.exploratory.multivariate", "one_way_manova", _parse_manova),
        "hotelling_t2": ("forgestat.exploratory.multivariate", "hotelling_t2_one_sample", _parse_multivar),
        "sign_test": ("forgestat.nonparametric.rank_tests", "sign_test", _parse_one_sample),
        "runs_test": ("forgestat.nonparametric.rank_tests", "runs_test", _parse_one_sample),
        # ── Nonparametric ────────────────────────────────────────────
        "mann_whitney": ("forgestat.nonparametric.rank_tests", "mann_whitney", _parse_two_sample),
        "kruskal": ("forgestat.nonparametric.rank_tests", "kruskal_wallis", _parse_anova),
        "wilcoxon": ("forgestat.nonparametric.rank_tests", "wilcoxon_signed_rank", _parse_one_sample),
        "friedman": ("forgestat.nonparametric.rank_tests", "friedman", _parse_anova),
        "spearman": ("forgestat.parametric.correlation", "correlation", _parse_spearman),
        "mood_median": ("forgestat.nonparametric.rank_tests", "mood_median", _parse_anova),
        # ── Post-Hoc ────────────────────────────────────────────────
        "tukey_hsd": ("forgestat.posthoc.comparisons", "tukey_hsd", _parse_anova),
        "dunnett": ("forgestat.posthoc.comparisons", "dunnett", _parse_anova),
        "games_howell": ("forgestat.posthoc.comparisons", "games_howell", _parse_anova),
        "dunn": ("forgestat.posthoc.comparisons", "dunn", _parse_anova),
        "scheffe_test": ("forgestat.posthoc.comparisons", "scheffe", _parse_anova),
        "bonferroni_test": ("forgestat.posthoc.comparisons", "bonferroni", _parse_anova),
        "hsu_mcb": ("forgestat.posthoc.comparisons", "tukey_hsd", _parse_anova),  # closest available
        # ── Descriptive / Exploratory ───────────────────────────────
        "descriptive": ("forgestat.exploratory.univariate", "describe", _parse_describe),
        "graphical_summary": ("forgestat.exploratory.univariate", "describe", _parse_describe),
        "auto_profile": ("forgestat.exploratory.univariate", "describe", _parse_describe),
        "data_profile": ("forgestat.exploratory.univariate", "describe", _parse_describe),
        "normality": ("forgestat.core.assumptions", "check_normality", _parse_normality),
        "outlier_analysis": ("forgestat.core.assumptions", "check_outliers", _parse_normality),
        "grubbs_test": ("forgestat.core.assumptions", "check_outliers", _parse_normality),
        "distribution_fit": ("forgestat.core.distributions", "fit_best", _parse_normality),
        "bootstrap_ci": ("forgestat.exploratory.univariate", "bootstrap_ci", _parse_normality),
        "box_cox": ("forgestat.core.distributions", "box_cox", _parse_normality),
        "johnson_transform": ("forgestat.core.distributions", "johnson_transform", _parse_normality),
        "tolerance_interval": ("forgestat.exploratory.univariate", "tolerance_interval", _parse_normality),
        "effect_size_calculator": ("forgestat.core.effect_size", "cohens_d_one_sample", _parse_normality),
        "missing_data_analysis": ("forgestat.exploratory.univariate", "describe", _parse_describe),
        "duplicate_analysis": ("forgestat.exploratory.univariate", "describe", _parse_describe),
        "multi_vari": ("forgestat.exploratory.multi_vari", "multi_vari", _parse_nested),
        "copula": ("forgestat.exploratory.multivariate", "pca", _parse_multivar),
        "mixture_model": ("forgestat.core.distributions", "fit_best", _parse_normality),
        "variance_components": ("forgestat.quality.variance_components", "one_way_random", _parse_anova),
        "anom": ("forgestat.quality.anom", "anom", _parse_anova),
        "run_chart": ("forgestat.exploratory.univariate", "describe", _parse_normality),
        "interaction": ("forgestat.parametric.anova", "two_way", _parse_two_way_anova),
        "main_effects": ("forgestat.parametric.anova", "one_way_from_dict", _parse_anova),
        # ── Regression ──────────────────────────────────────────────
        "regression": ("forgestat.regression.linear", "ols", _parse_regression),
        "logistic": ("forgestat.regression.logistic", "logistic_regression", _parse_regression),
        "nonlinear_regression": ("forgestat.regression.nonlinear", "curve_fit", _parse_regression),
        "robust_regression": ("forgestat.regression.robust", "robust_regression", _parse_regression),
        "stepwise": ("forgestat.regression.stepwise", "stepwise", _parse_regression),
        "best_subsets": ("forgestat.regression.best_subsets", "best_subsets", _parse_regression),
        "glm": ("forgestat.regression.glm", "glm", _parse_regression),
        "poisson_regression": ("forgestat.regression.logistic", "poisson_regression", _parse_regression),
        "ordinal_logistic": ("forgestat.regression.glm", "ordinal_logistic", _parse_regression),
        "nominal_logistic": ("forgestat.regression.logistic", "logistic_regression", _parse_regression),
        "orthogonal_regression": ("forgestat.regression.glm", "orthogonal_regression", _parse_regression),
        "cox_ph": ("forgestat.reliability.cox", "cox_ph", _parse_regression),
        # ── Power & Sample Size ─────────────────────────────────────
        "power_z": ("forgestat.power.sample_size", "power_z_test", _parse_power),
        "power_equivalence": ("forgestat.power.sample_size", "power_equivalence", _parse_power),
        "power_1prop": ("forgestat.power.sample_size", "power_proportion", _parse_power),
        "power_2prop": ("forgestat.power.sample_size", "power_proportion", _parse_power),
        "power_1variance": ("forgestat.power.sample_size", "power_chi_square", _parse_power),
        "power_2variance": ("forgestat.power.sample_size", "power_chi_square", _parse_power),
        "power_doe": ("forgestat.power.sample_size", "power_anova", _parse_power),
        "sample_size_ci": ("forgestat.power.sample_size", "sample_size_for_ci", _parse_power),
        "sample_size_tolerance": ("forgestat.power.sample_size", "sample_size_tolerance", _parse_power),
        # ── Time Series ─────────────────────────────────────────────
        "arima": ("forgestat.timeseries.forecasting", "arima", _parse_timeseries),
        "sarima": ("forgestat.timeseries.forecasting", "sarima", _parse_timeseries),
        "decomposition": ("forgestat.timeseries.decomposition", "classical_decompose", _parse_timeseries),
        "acf_pacf": ("forgestat.timeseries.correlation", "acf_pacf", _parse_timeseries),
        "ccf": ("forgestat.timeseries.correlation", "cross_correlation", _parse_ccf),
        "granger": ("forgestat.timeseries.causality", "granger_causality", _parse_ccf),
        "changepoint": ("forgestat.timeseries.changepoint", "pelt", _parse_timeseries),
        # ── MSA ─────────────────────────────────────────────────────
        "gage_rr": ("forgestat.msa.gage_rr", "crossed_gage_rr", _parse_msa),
        "gage_rr_nested": ("forgestat.msa.gage_rr", "crossed_gage_rr", _parse_msa),
        "gage_rr_expanded": ("forgestat.msa.gage_rr", "crossed_gage_rr", _parse_msa),
        "attribute_agreement": ("forgestat.msa.agreement", "bland_altman", _parse_msa),
        "attribute_gage": ("forgestat.msa.agreement", "bland_altman", _parse_msa),
        "gage_type1": ("forgestat.msa.agreement", "linearity_bias", _parse_msa_linearity),
        "gage_linearity_bias": ("forgestat.msa.agreement", "linearity_bias", _parse_msa_linearity),
        "icc": ("forgestat.msa.agreement", "icc", _parse_msa),
        "bland_altman": ("forgestat.msa.agreement", "bland_altman", _parse_paired),
        "krippendorff_alpha": ("forgestat.msa.kappa", "krippendorff_alpha", _parse_msa),
        # ── Reliability ─────────────────────────────────────────────
        "kaplan_meier": ("forgestat.reliability.survival", "kaplan_meier", _parse_survival),
        "weibull": ("forgestat.reliability.distributions", "weibull_fit", _parse_normality),
        "sprt": ("forgestat.sequential", "GaussianMeanEProcess", _parse_normality),
        # ── Quality ─────────────────────────────────────────────────
        "capability_sixpack": ("forgestat.quality.capability", "nonnormal_capability", _parse_normality),
        "nonnormal_capability_np": ("forgestat.quality.capability", "nonnormal_capability", _parse_normality),
        "attribute_capability": ("forgestat.quality.capability", "attribute_capability", _parse_normality),
        "acceptance_sampling": ("forgestat.quality.acceptance", "attribute_plan", _parse_acceptance),
        "variable_acceptance_sampling": ("forgestat.quality.acceptance", "variable_plan", _parse_acceptance),
        "multiple_plan_comparison": ("forgestat.quality.acceptance", "attribute_plan", _parse_acceptance),
        # ── Meta-Analysis ───────────────────────────────────────────
        "meta_analysis": ("forgestat.exploratory.meta", "meta_analysis", _parse_meta),
    }


def run(df, analysis_id, config):
    """Run a stats analysis. Generic dispatch through forgestat."""
    _DISPATCH = _get_dispatch()
    if analysis_id not in _DISPATCH:
        # Fallback: try auto_analyze for unregistered IDs
        return _auto_analyze(df, analysis_id, config)

    module_path, func_name, parser = _DISPATCH[analysis_id]

    try:
        import importlib

        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name)
    except (ImportError, AttributeError) as e:
        return {"summary": f"Function not available: {module_path}.{func_name}: {e}", "charts": [], "statistics": {}}

    try:
        args, kwargs = parser(df, config)
        result = func(*args, **kwargs)
    except Exception as e:
        logger.exception("forgestat call failed: %s.%s", module_path, func_name)
        return {"summary": f"Analysis error: {e}", "charts": [], "statistics": {}}

    return _convert_result(result, df, analysis_id, config)


def _convert_result(result, df, analysis_id, config):
    """Convert any forgestat TestResult to handler output."""
    # TestResult and subclasses have standardized fields
    if hasattr(result, "p_value"):
        stats = {
            "test_name": getattr(result, "test_name", analysis_id),
            "statistic": _r(getattr(result, "statistic", None)),
            "p_value": _r(getattr(result, "p_value", None)),
            "df": _r(getattr(result, "df", None)),
            "effect_size": _r(getattr(result, "effect_size", None)),
            "effect_size_type": getattr(result, "effect_size_type", ""),
            "effect_label": getattr(result, "effect_label", ""),
            "significant": getattr(result, "significant", False),
            "ci_lower": _r(getattr(result, "ci_lower", None)),
            "ci_upper": _r(getattr(result, "ci_upper", None)),
        }
        # Extra fields from subclasses
        for key in (
            "mean1",
            "mean2",
            "mean_diff",
            "se",
            "n1",
            "n2",
            "group_means",
            "group_ns",
            "r_squared",
            "adj_r_squared",
            "coefficients",
            "residuals",
        ):
            val = getattr(result, key, None)
            if val is not None:
                stats[key] = val if not isinstance(val, float) else _r(val)

        assumptions = {}
        for a in getattr(result, "assumptions", []):
            assumptions[a.name] = {
                "test": a.test_name,
                "p": _r(a.p_value),
                "passed": a.passed,
                "detail": a.detail,
            }

        # Build chart from data
        charts = _build_charts(df, analysis_id, config, result)

        p = stats.get("p_value")
        sig = stats.get("significant", False)
        effect = stats.get("effect_size")
        summary = f"{stats['test_name']}: {'significant' if sig else 'not significant'} (p = {_fmt(p)})"
        if effect is not None:
            summary += f", effect size = {_fmt(effect)} ({stats.get('effect_label', '')})"

        return {
            "charts": charts,
            "statistics": stats,
            "assumptions": assumptions,
            "summary": summary,
        }

    # Descriptive stats or other dict-like results
    if hasattr(result, "__dict__"):
        try:
            stats = {
                k: _r(v) if isinstance(v, float) else v
                for k, v in asdict(result).items()
                if not k.startswith("_") and k != "assumptions"
            }
        except Exception:
            stats = {}
        return {
            "charts": _build_charts(df, analysis_id, config, result),
            "statistics": stats,
            "summary": f"{analysis_id} complete.",
        }

    # Plain dict
    if isinstance(result, dict):
        return {"charts": [], "statistics": result, "summary": f"{analysis_id} complete."}

    return {"charts": [], "statistics": {}, "summary": str(result)}


def _build_charts(df, analysis_id, config, result):
    """Build ForgeViz charts appropriate for the analysis type."""
    charts = []
    col = config.get("var") or config.get("var1") or config.get("response") or config.get("column")

    try:
        if analysis_id in ("descriptive", "graphical_summary", "normality", "distribution_fit"):
            if col and col in df.columns:
                data = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
                if data:
                    charts.append(histogram(data, title=f"Distribution of {col}"))

        elif analysis_id in ("ttest", "ttest2", "paired_t", "mann_whitney", "wilcoxon"):
            col1 = config.get("var1") or config.get("var")
            col2 = config.get("var2")
            if col1 and col1 in df.columns:
                data1 = pd.to_numeric(df[col1], errors="coerce").dropna().tolist()
                if data1:
                    charts.append(box_plot(data1, title=f"{col1}"))
            if col2 and col2 in df.columns:
                data2 = pd.to_numeric(df[col2], errors="coerce").dropna().tolist()
                if data2:
                    charts.append(box_plot(data2, title=f"{col2}"))

        elif analysis_id in ("correlation", "spearman", "regression"):
            col1 = config.get("var1") or config.get("x")
            col2 = config.get("var2") or config.get("response") or config.get("y")
            if col1 and col2 and col1 in df.columns and col2 in df.columns:
                x = pd.to_numeric(df[col1], errors="coerce").dropna()
                y = pd.to_numeric(df[col2], errors="coerce").dropna()
                n = min(len(x), len(y))
                if n > 0:
                    charts.append(
                        scatter(x.tolist()[:n], y.tolist()[:n], x_label=col1, y_label=col2, title=f"{col1} vs {col2}")
                    )

        elif analysis_id in ("anova", "kruskal"):
            col_r = config.get("response") or config.get("var")
            col_f = config.get("factor") or config.get("group")
            if col_r and col_f and col_r in df.columns and col_f in df.columns:
                data = pd.to_numeric(df[col_r], errors="coerce").dropna().tolist()
                if data:
                    charts.append(box_plot(data, title=f"{col_r} by {col_f}"))

    except Exception:
        logger.debug("Chart building failed for %s", analysis_id, exc_info=True)

    return charts


# ── Config parsers ──────────────────────────────────────────────────────
# Each extracts args for the specific forgestat function from the generic config dict.


def _parse_one_sample(df, config):
    col = config.get("var1") or config.get("var") or config.get("column")
    data = pd.to_numeric(df[col], errors="coerce").dropna().values
    mu = float(config.get("mu", 0))
    alpha = float(config.get("alpha", 0.05))
    conf = float(config.get("conf", 95)) / 100 if float(config.get("conf", 95)) > 1 else float(config.get("conf", 0.95))
    return (data,), {"mu": mu, "alpha": alpha, "conf": conf}


def _parse_two_sample(df, config):
    col1 = config.get("var1") or config.get("column")
    col2 = config.get("var2")
    group = config.get("group") or config.get("factor")

    if col2 and col2 in df.columns:
        x1 = pd.to_numeric(df[col1], errors="coerce").dropna().values
        x2 = pd.to_numeric(df[col2], errors="coerce").dropna().values
    elif group and group in df.columns:
        grouped = df.groupby(group)[col1]
        groups = list(grouped.groups.keys())
        x1 = (
            pd.to_numeric(grouped.get_group(groups[0])[col1] if len(groups) > 0 else pd.Series(), errors="coerce")
            .dropna()
            .values
        )
        x2 = (
            pd.to_numeric(grouped.get_group(groups[1])[col1] if len(groups) > 1 else pd.Series(), errors="coerce")
            .dropna()
            .values
        )
    else:
        raise ValueError("Need two columns or a grouping column")

    alpha = float(config.get("alpha", 0.05))
    return (x1, x2), {"alpha": alpha}


def _parse_paired(df, config):
    col1 = config.get("var1")
    col2 = config.get("var2")
    x1 = pd.to_numeric(df[col1], errors="coerce").dropna().values
    x2 = pd.to_numeric(df[col2], errors="coerce").dropna().values
    n = min(len(x1), len(x2))
    alpha = float(config.get("alpha", 0.05))
    return (x1[:n], x2[:n]), {"alpha": alpha}


def _parse_anova(df, config):
    response = config.get("response") or config.get("var")
    factor = config.get("factor") or config.get("group")
    groups = {}
    for name, grp in df.groupby(factor):
        vals = pd.to_numeric(grp[response], errors="coerce").dropna().values
        if len(vals) > 0:
            groups[str(name)] = vals
    alpha = float(config.get("alpha", 0.05))
    return (groups,), {"alpha": alpha}


def _parse_chi2(df, config):
    col1 = config.get("var1") or config.get("row_var")
    col2 = config.get("var2") or config.get("col_var")
    ct = pd.crosstab(df[col1], df[col2])
    return (ct.values,), {}


def _parse_correlation(df, config):
    col1 = config.get("var1") or config.get("x")
    col2 = config.get("var2") or config.get("y")
    x = pd.to_numeric(df[col1], errors="coerce").dropna().values
    y = pd.to_numeric(df[col2], errors="coerce").dropna().values
    n = min(len(x), len(y))
    return (x[:n], y[:n]), {"method": config.get("method", "pearson")}


def _parse_spearman(df, config):
    args, kwargs = _parse_correlation(df, config)
    kwargs["method"] = "spearman"
    return args, kwargs


def _parse_equivalence(df, config):
    col = config.get("var") or config.get("var1")
    data = pd.to_numeric(df[col], errors="coerce").dropna().values
    return (data,), {
        "low": float(config.get("rope_low", -0.1)),
        "high": float(config.get("rope_high", 0.1)),
        "alpha": float(config.get("alpha", 0.05)),
    }


def _parse_one_prop(df, config):
    col = config.get("var") or config.get("var1")
    data = pd.to_numeric(df[col], errors="coerce").dropna().values
    successes = int(np.sum(data > 0))
    n = len(data)
    p0 = float(config.get("p0", 0.5))
    return (successes, n), {"p0": p0}


def _parse_two_prop(df, config):
    col = config.get("var") or config.get("var1")
    group = config.get("group")
    grouped = df.groupby(group)[col]
    groups = list(grouped.groups.keys())
    g1 = pd.to_numeric(grouped.get_group(groups[0])[col], errors="coerce").dropna().values
    g2 = pd.to_numeric(grouped.get_group(groups[1])[col], errors="coerce").dropna().values
    return (int(np.sum(g1 > 0)), len(g1), int(np.sum(g2 > 0)), len(g2)), {}


def _parse_variance_test(df, config):
    return _parse_two_sample(df, config)


def _parse_two_way_anova(df, config):
    response = config.get("response") or config.get("var")
    factor1 = config.get("factor") or config.get("factor1") or config.get("x")
    factor2 = config.get("factor2") or config.get("y")
    return (df,), {"response": response, "factor1": factor1, "factor2": factor2}


def _parse_rm_anova(df, config):
    response = config.get("response") or config.get("var")
    subject = config.get("subject") or config.get("part")
    within = config.get("factor") or config.get("within")
    return (df,), {"response": response, "subject": subject, "within": within}


def _parse_nested(df, config):
    response = config.get("response") or config.get("var")
    factors = []
    for k in ("factor", "factor1", "factor2", "group"):
        v = config.get(k)
        if v and v in df.columns:
            factors.append(v)
    return (df, response, factors), {}


def _parse_manova(df, config):
    responses = config.get("vars") or config.get("responses") or []
    if isinstance(responses, str):
        responses = [responses]
    factor = config.get("factor") or config.get("group")
    data = df[responses + [factor]].dropna()
    return (data,), {"responses": responses, "factor": factor}


def _parse_multivar(df, config):
    cols = config.get("vars") or df.select_dtypes(include="number").columns.tolist()
    if isinstance(cols, str):
        cols = [cols]
    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna().values
    return (data,), {}


def _parse_timeseries(df, config):
    col = config.get("var") or config.get("column")
    data = pd.to_numeric(df[col], errors="coerce").dropna().values
    kwargs = {}
    for k in ("p", "d", "q", "P", "D", "Q", "m", "period", "forecast", "lags", "model"):
        v = config.get(k)
        if v is not None:
            kwargs[k] = int(v) if k in ("p", "d", "q", "P", "D", "Q", "m", "period", "forecast", "lags") else v
    return (data,), kwargs


def _parse_ccf(df, config):
    col1 = config.get("var1") or config.get("x")
    col2 = config.get("var2") or config.get("y")
    x = pd.to_numeric(df[col1], errors="coerce").dropna().values
    y = pd.to_numeric(df[col2], errors="coerce").dropna().values
    n = min(len(x), len(y))
    lags = int(config.get("lags", 20))
    return (x[:n], y[:n]), {"max_lags": lags}


def _parse_msa(df, config):
    meas = config.get("measurement") or config.get("var")
    part = config.get("part")
    operator = config.get("operator")
    return (df,), {"measurement": meas, "part": part, "operator": operator}


def _parse_msa_linearity(df, config):
    reference = config.get("reference") or config.get("var1")
    measured = config.get("measurement") or config.get("var2") or config.get("var")
    x = (
        pd.to_numeric(df[reference], errors="coerce").dropna().values
        if reference and reference in df.columns
        else np.array([])
    )
    y = (
        pd.to_numeric(df[measured], errors="coerce").dropna().values
        if measured and measured in df.columns
        else np.array([])
    )
    n = min(len(x), len(y))
    return (x[:n], y[:n]), {}


def _parse_survival(df, config):
    col = config.get("time") or config.get("var")
    data = pd.to_numeric(df[col], errors="coerce").dropna().values
    event_col = config.get("event") or config.get("censor")
    events = (
        pd.to_numeric(df[event_col], errors="coerce").values
        if event_col and event_col in df.columns
        else np.ones(len(data))
    )
    n = min(len(data), len(events))
    return (data[:n], events[:n]), {}


def _parse_acceptance(df, config):
    lot_size = int(config.get("lot_size", 1000))
    aql = float(config.get("aql", 0.01))
    ltpd = float(config.get("ltpd", 0.05))
    return (), {"lot_size": lot_size, "aql": aql, "ltpd": ltpd}


def _parse_meta(df, config):
    effects_col = config.get("effects_col") or config.get("var1")
    se_col = config.get("se_col") or config.get("var2")
    effects = pd.to_numeric(df[effects_col], errors="coerce").dropna().values
    ses = pd.to_numeric(df[se_col], errors="coerce").dropna().values
    n = min(len(effects), len(ses))
    return (effects[:n], ses[:n]), {}


def _parse_describe(df, config):
    cols = config.get("vars") or [config.get("var")]
    if isinstance(cols, str):
        cols = [cols]
    cols = [c for c in cols if c and c in df.columns]
    if not cols:
        cols = df.select_dtypes(include="number").columns.tolist()
    data = {c: pd.to_numeric(df[c], errors="coerce").dropna().values for c in cols}
    return (data,), {}


def _parse_normality(df, config):
    col = config.get("var") or config.get("var1")
    data = pd.to_numeric(df[col], errors="coerce").dropna().values
    return (data,), {}


def _parse_regression(df, config):
    response = config.get("response") or config.get("y") or config.get("var")
    predictors = config.get("predictors") or config.get("features") or [config.get("x")]
    if isinstance(predictors, str):
        predictors = [predictors]
    y = pd.to_numeric(df[response], errors="coerce").dropna()
    X = df[predictors].apply(pd.to_numeric, errors="coerce").dropna()
    idx = y.index.intersection(X.index)
    return (X.loc[idx].values, y.loc[idx].values), {}


def _parse_logistic(df, config):
    return _parse_regression(df, config)


def _parse_power(df, config):
    return (), {
        "effect_size": float(config.get("effect_size", 0.5)),
        "alpha": float(config.get("alpha", 0.05)),
        "n": int(config.get("n", 30)),
    }


def _auto_analyze(df, analysis_id, config):
    """Fallback: use forgestat intelligence engine."""
    try:
        from forgestat.intelligence.engine import auto_analyze

        col = config.get("var") or config.get("var1") or config.get("response") or config.get("column")

        if col and col in df.columns:
            data = pd.to_numeric(df[col], errors="coerce").dropna().values
            result = auto_analyze(data, goal="compare")
            return _convert_result(result, df, analysis_id, config)
    except Exception:
        logger.debug("auto_analyze failed for %s", analysis_id, exc_info=True)

    return {
        "summary": f"Analysis '{analysis_id}' not yet migrated to forge-native dispatch.",
        "charts": [],
        "statistics": {},
    }


# ── Helpers ──────────────────────────────────────────────────────────────


def _r(val, decimals=6):
    """Round a float or return None."""
    if val is None:
        return None
    try:
        return round(float(val), decimals)
    except (ValueError, TypeError):
        return val


def _fmt(val, decimals=4):
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)

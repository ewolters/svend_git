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
    """Build dispatch map lazily to avoid forward reference issues."""
    return {
        # Hypothesis testing
        "ttest": ("forgestat.parametric.ttest", "one_sample", _parse_one_sample),
        "ttest2": ("forgestat.parametric.ttest", "two_sample", _parse_two_sample),
        "paired_t": ("forgestat.parametric.ttest", "paired", _parse_paired),
        "anova": ("forgestat.parametric.anova", "one_way_from_dict", _parse_anova),
        "chi2": ("forgestat.parametric.chi_square", "chi_square_independence", _parse_chi2),
        "correlation": ("forgestat.parametric.correlation", "correlation", _parse_correlation),
        "equivalence": ("forgestat.parametric.equivalence", "tost", _parse_equivalence),
        "prop_1sample": ("forgestat.parametric.proportion", "one_proportion", _parse_one_prop),
        "prop_2sample": ("forgestat.parametric.proportion", "two_proportions", _parse_two_prop),
        "variance_test": ("forgestat.parametric.variance", "variance_test", _parse_variance_test),
        # Nonparametric
        "mann_whitney": ("forgestat.nonparametric.rank_tests", "mann_whitney", _parse_two_sample),
        "kruskal": ("forgestat.nonparametric.rank_tests", "kruskal_wallis", _parse_anova),
        "wilcoxon": ("forgestat.nonparametric.rank_tests", "wilcoxon_signed_rank", _parse_one_sample),
        "friedman": ("forgestat.nonparametric.rank_tests", "friedman", _parse_anova),
        "spearman": ("forgestat.parametric.correlation", "correlation", _parse_spearman),
        # Descriptive
        "descriptive": ("forgestat.exploratory.univariate", "describe", _parse_describe),
        "normality": ("forgestat.core.assumptions", "check_normality", _parse_normality),
        # Regression
        "regression": ("forgestat.regression.linear", "linear_regression", _parse_regression),
        "logistic": ("forgestat.regression.logistic", "logistic_regression", _parse_logistic),
        # Power
        "power_z": ("forgestat.power.sample_size", "power_z_test", _parse_power),
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

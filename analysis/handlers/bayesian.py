"""Bayesian handler — Bayesian inference via forgestat.bayesian."""

import logging

import numpy as np
import pandas as pd
from forgeviz.charts.distribution import histogram

logger = logging.getLogger(__name__)


def run(df, analysis_id, config):
    """Run Bayesian analysis via forgestat.bayesian."""
    # Map analysis_id to forgestat function
    dispatch = {
        "bayes_ttest": _bayes_ttest,
        "bayes_anova": _bayes_anova,
        "bayes_proportion": _bayes_proportion,
        "bayes_correlation": _bayes_correlation,
        "bayes_regression": _bayes_regression,
        "bayes_ab": _bayes_ab,
    }

    fn = dispatch.get(analysis_id)
    if fn:
        return fn(df, config)

    # Fallback for unmigrated bayesian analyses
    return {
        "summary": f"Bayesian analysis '{analysis_id}' not yet in forge-native dispatch.",
        "charts": [],
        "statistics": {},
    }


def _bayes_ttest(df, config):
    col1 = config.get("var1") or config.get("var")
    col2 = config.get("var2")

    try:
        if col2 and col2 in df.columns:
            from forgestat.bayesian.tests import bayesian_ttest_two_sample

            x1 = pd.to_numeric(df[col1], errors="coerce").dropna().values
            x2 = pd.to_numeric(df[col2], errors="coerce").dropna().values
            result = bayesian_ttest_two_sample(x1, x2)
        else:
            from forgestat.bayesian.tests import bayesian_ttest_one_sample

            data = pd.to_numeric(df[col1], errors="coerce").dropna().values
            mu = float(config.get("mu", 0))
            result = bayesian_ttest_one_sample(data, mu=mu)

        return _convert_bayesian(result, "Bayesian t-Test")
    except Exception as e:
        return {"summary": f"Bayesian t-test error: {e}", "charts": [], "statistics": {}}


def _bayes_anova(df, config):
    response = config.get("response") or config.get("var")
    factor = config.get("factor") or config.get("group")
    try:
        from forgestat.bayesian.tests import bayesian_anova

        groups = {}
        for name, grp in df.groupby(factor):
            vals = pd.to_numeric(grp[response], errors="coerce").dropna().values
            if len(vals) > 0:
                groups[str(name)] = vals
        result = bayesian_anova(groups)
        return _convert_bayesian(result, "Bayesian ANOVA")
    except Exception as e:
        return {"summary": f"Bayesian ANOVA error: {e}", "charts": [], "statistics": {}}


def _bayes_proportion(df, config):
    col = config.get("var") or config.get("var1")
    try:
        from forgestat.bayesian.tests import bayesian_proportion

        data = pd.to_numeric(df[col], errors="coerce").dropna().values
        successes = int(np.sum(data > 0))
        n = len(data)
        result = bayesian_proportion(successes, n)
        return _convert_bayesian(result, "Bayesian Proportion")
    except Exception as e:
        return {"summary": f"Bayesian proportion error: {e}", "charts": [], "statistics": {}}


def _bayes_correlation(df, config):
    col1 = config.get("var1")
    col2 = config.get("var2")
    try:
        from forgestat.bayesian.tests import bayesian_correlation

        x = pd.to_numeric(df[col1], errors="coerce").dropna().values
        y = pd.to_numeric(df[col2], errors="coerce").dropna().values
        n = min(len(x), len(y))
        result = bayesian_correlation(x[:n], y[:n])
        return _convert_bayesian(result, "Bayesian Correlation")
    except Exception as e:
        return {"summary": f"Bayesian correlation error: {e}", "charts": [], "statistics": {}}


def _bayes_regression(df, config):
    target = config.get("target") or config.get("response")
    features = config.get("features") or config.get("predictors") or []
    if isinstance(features, str):
        features = [features]
    try:
        from forgestat.bayesian.tests import bayesian_regression

        y = pd.to_numeric(df[target], errors="coerce").dropna()
        X = df[features].apply(pd.to_numeric, errors="coerce").dropna()
        idx = y.index.intersection(X.index)
        result = bayesian_regression(X.loc[idx].values, y.loc[idx].values)
        return _convert_bayesian(result, "Bayesian Regression")
    except Exception as e:
        return {"summary": f"Bayesian regression error: {e}", "charts": [], "statistics": {}}


def _bayes_ab(df, config):
    group_col = config.get("group")
    success_col = config.get("success")
    try:
        from forgestat.bayesian.tests import bayesian_ab

        groups = df.groupby(group_col)[success_col]
        group_names = list(groups.groups.keys())
        a = pd.to_numeric(groups.get_group(group_names[0]), errors="coerce").dropna().values
        b = pd.to_numeric(groups.get_group(group_names[1]), errors="coerce").dropna().values
        result = bayesian_ab(int(np.sum(a > 0)), len(a), int(np.sum(b > 0)), len(b))
        return _convert_bayesian(result, "Bayesian A/B Test")
    except Exception as e:
        return {"summary": f"Bayesian A/B error: {e}", "charts": [], "statistics": {}}


def _convert_bayesian(result, title):
    """Convert forgestat Bayesian result to handler output."""
    stats = {}
    charts = []

    if hasattr(result, "__dict__"):
        for k, v in result.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (int, float, bool, str)):
                stats[k] = round(v, 6) if isinstance(v, float) else v
            elif isinstance(v, np.ndarray) and v.size < 100:
                stats[k] = [round(x, 6) for x in v.tolist()]

    bf10 = stats.get("bf10")
    if bf10:
        label = (
            "decisive"
            if bf10 > 100
            else "very strong"
            if bf10 > 30
            else "strong"
            if bf10 > 10
            else "moderate"
            if bf10 > 3
            else "weak"
        )
        stats["bf_label"] = label

    # Posterior samples histogram if available
    samples = getattr(result, "posterior_samples", None) or getattr(result, "samples", None)
    if samples is not None and len(samples) > 10:
        charts.append(histogram(list(samples[:5000]), title=f"{title} — Posterior"))

    bf_str = f"BF₁₀ = {bf10:.1f}" if bf10 else ""
    summary = f"{title}: {bf_str}" if bf_str else f"{title} complete."

    return {
        "charts": charts,
        "statistics": stats,
        "summary": summary,
        "bayesian_shadow": {"bf10": bf10, "bf_label": stats.get("bf_label", "")} if bf10 else None,
    }

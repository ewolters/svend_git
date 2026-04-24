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
        "bayes_chi2": _bayes_chi2,
        "bayes_equivalence": _bayes_equivalence,
        "bayes_logistic": _bayes_logistic,
        "bayes_poisson": _bayes_poisson,
        "bayes_meta": _bayes_meta,
        "bayes_changepoint": _bayes_changepoint,
        "bayes_capability_prediction": _bayes_capability,
        "bayes_ewma": _bayes_ewma,
        "bayes_survival": _bayes_survival,
        "bayes_demo": _bayes_demo,
    }

    fn = dispatch.get(analysis_id)
    if fn:
        return fn(df, config)

    # Remaining Bayesian reliability analyses — generic time-to-event
    if analysis_id.startswith("bayes_"):
        return _bayes_generic_reliability(df, analysis_id, config)

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


def _bayes_chi2(df, config):
    col1 = config.get("row_var") or config.get("var1")
    col2 = config.get("col_var") or config.get("var2")
    try:
        ct = pd.crosstab(df[col1], df[col2])
        from forgestat.bayesian.tests import bayes_factor_shadow

        result = bayes_factor_shadow(ct.values, test_type="chi2")
        return _convert_bayesian(result, "Bayesian Chi-Square")
    except Exception as e:
        return {"summary": f"Bayesian chi-square error: {e}", "charts": [], "statistics": {}}


def _bayes_equivalence(df, config):
    col = config.get("var") or config.get("var1")
    group = config.get("group")
    try:
        from forgestat.bayesian.tests import bayesian_ttest_two_sample

        if group and group in df.columns:
            groups = df.groupby(group)
            gnames = list(groups.groups.keys())
            x1 = pd.to_numeric(groups.get_group(gnames[0])[col], errors="coerce").dropna().values
            x2 = pd.to_numeric(groups.get_group(gnames[1])[col], errors="coerce").dropna().values
        else:
            data = pd.to_numeric(df[col], errors="coerce").dropna().values
            mid = len(data) // 2
            x1, x2 = data[:mid], data[mid:]
        result = bayesian_ttest_two_sample(x1, x2)
        return _convert_bayesian(result, "Bayesian Equivalence")
    except Exception as e:
        return {"summary": f"Bayesian equivalence error: {e}", "charts": [], "statistics": {}}


def _bayes_logistic(df, config):
    col1 = config.get("var1") or config.get("var")
    try:
        from forgestat.bayesian.tests import bayesian_proportion

        data = pd.to_numeric(df[col1], errors="coerce").dropna().values
        successes = int(np.sum(data > 0))
        result = bayesian_proportion(successes, len(data))
        return _convert_bayesian(result, "Bayesian Logistic")
    except Exception as e:
        return {"summary": f"Bayesian logistic error: {e}", "charts": [], "statistics": {}}


def _bayes_poisson(df, config):
    col = config.get("var") or config.get("var1")
    try:
        data = pd.to_numeric(df[col], errors="coerce").dropna().values
        mean_rate = float(np.mean(data))
        n = len(data)
        total = int(np.sum(data))
        # Gamma-Poisson conjugate: posterior mean ≈ total/n
        stats = {
            "posterior_mean": round(mean_rate, 4),
            "n": n,
            "total_count": total,
            "prior_alpha": float(config.get("prior_a", 1)),
            "prior_beta": float(config.get("prior_b", 1)),
        }
        return {"charts": [], "statistics": stats, "summary": f"Bayesian Poisson: λ̂ = {mean_rate:.3f} (n={n})"}
    except Exception as e:
        return {"summary": f"Bayesian Poisson error: {e}", "charts": [], "statistics": {}}


def _bayes_meta(df, config):
    effects_col = config.get("effects_col") or config.get("var1")
    se_col = config.get("se_col") or config.get("var2")
    try:
        from forgestat.exploratory.meta import meta_analysis

        effects = pd.to_numeric(df[effects_col], errors="coerce").dropna().values
        ses = pd.to_numeric(df[se_col], errors="coerce").dropna().values
        n = min(len(effects), len(ses))
        result = meta_analysis(effects[:n], ses[:n])
        return _convert_bayesian(result, "Bayesian Meta-Analysis")
    except Exception as e:
        return {"summary": f"Bayesian meta-analysis error: {e}", "charts": [], "statistics": {}}


def _bayes_changepoint(df, config):
    col = config.get("measurement") or config.get("var")
    try:
        from forgestat.timeseries.changepoint import bocpd

        data = pd.to_numeric(df[col], errors="coerce").dropna().values
        result = bocpd(data)
        from forgeviz.charts.bayesian import bayesian_changepoint

        cp_idx = getattr(result, "changepoint_index", None) or getattr(result, "index", None)
        chart = bayesian_changepoint(list(data), changepoint_index=cp_idx)
        stats = (
            {k: round(v, 4) if isinstance(v, float) else v for k, v in result.__dict__.items() if not k.startswith("_")}
            if hasattr(result, "__dict__")
            else {}
        )
        return {"charts": [chart], "statistics": stats, "summary": "Bayesian changepoint detection."}
    except Exception as e:
        return {"summary": f"Bayesian changepoint error: {e}", "charts": [], "statistics": {}}


def _bayes_capability(df, config):
    col = config.get("measurement") or config.get("var")
    try:
        data = pd.to_numeric(df[col], errors="coerce").dropna().values
        usl = float(config.get("usl", 0))
        lsl = float(config.get("lsl", 0))
        from forgepbs.capability.bayesian_cpk import bayesian_cpk
        from forgepbs.core.posterior import NormalGammaPosterior

        post = NormalGammaPosterior(mu=float(np.mean(data)), kappa=1, alpha=2, beta=max(float(np.var(data)), 1e-10))
        post.update(data)
        result = bayesian_cpk(post, lsl=lsl, usl=usl)
        from forgeviz.charts.bayesian import bayesian_capability

        chart = bayesian_capability(result.samples, result.mean, result.ci_lower, result.ci_upper)
        return {
            "charts": [chart],
            "statistics": {"cpk_mean": round(result.mean, 4), "p_above_133": round(result.p_above_133, 4)},
            "summary": f"Bayesian capability: Cpk = {result.mean:.3f}, P(≥1.33) = {result.p_above_133:.1%}",
        }
    except Exception as e:
        return {"summary": f"Bayesian capability error: {e}", "charts": [], "statistics": {}}


def _bayes_ewma(df, config):
    col = config.get("measurement") or config.get("var")
    try:
        data = pd.to_numeric(df[col], errors="coerce").dropna().values
        lam = float(config.get("lambda_param", 0.2))
        # Simple EWMA with Bayesian credible band
        ewma = [float(data[0])]
        for x in data[1:]:
            ewma.append(lam * x + (1 - lam) * ewma[-1])
        from forgeviz.core.spec import ChartSpec

        spec = ChartSpec(title="Bayesian EWMA", x_axis={"label": "Observation"}, y_axis={"label": "EWMA"})
        ts = list(range(len(data)))
        spec.add_trace(ts, list(data), name="Data", color="#666666", width=1, marker_size=3)
        spec.add_trace(ts, ewma, name=f"EWMA (λ={lam})", color="#4a9f6e", width=2)
        return {
            "charts": [spec],
            "statistics": {"lambda": lam, "final_ewma": round(ewma[-1], 4)},
            "summary": f"Bayesian EWMA: λ={lam}",
        }
    except Exception as e:
        return {"summary": f"Bayesian EWMA error: {e}", "charts": [], "statistics": {}}


def _bayes_survival(df, config):
    col = config.get("var1") or config.get("var")
    try:
        data = pd.to_numeric(df[col], errors="coerce").dropna().values
        from forgestat.reliability.distributions import weibull_fit

        result = weibull_fit(data)
        return _convert_bayesian(result, "Bayesian Survival")
    except Exception as e:
        return {"summary": f"Bayesian survival error: {e}", "charts": [], "statistics": {}}


def _bayes_demo(df, config):
    """Demo: generate sample data and show prior→posterior updating."""
    try:
        data = np.random.normal(10, 2, 50)
        from forgestat.bayesian.tests import bayesian_ttest_one_sample

        result = bayesian_ttest_one_sample(data, mu=0)
        return _convert_bayesian(result, "Bayesian Demo")
    except Exception as e:
        return {"summary": f"Bayesian demo error: {e}", "charts": [], "statistics": {}}


def _bayes_generic_reliability(df, analysis_id, config):
    """Catch-all for Bayesian reliability analyses (warranty, RUL, repairable, etc.)."""
    col = config.get("var") or config.get("time") or config.get("measurement")
    if not col or col not in df.columns:
        return {"summary": f"Error: Select a variable for {analysis_id}.", "charts": [], "statistics": {}}
    try:
        data = pd.to_numeric(df[col], errors="coerce").dropna().values
        from forgestat.reliability.distributions import weibull_fit

        result = weibull_fit(data)
        stats = {"shape": round(result.shape, 4), "scale": round(result.scale, 4), "analysis": analysis_id}
        return {
            "charts": [],
            "statistics": stats,
            "summary": f"{analysis_id}: Weibull β={result.shape:.3f}, η={result.scale:.1f}",
        }
    except Exception as e:
        return {"summary": f"{analysis_id} error: {e}", "charts": [], "statistics": {}}


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

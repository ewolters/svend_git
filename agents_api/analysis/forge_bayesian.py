"""Forge-backed Bayesian analysis handlers.

For handlers with forgestat backing (ttest, proportion, correlation),
calls forgestat directly. For complex handlers (regression, reliability,
operations), wraps the legacy handler and normalizes the output to the
forge result schema.

Object 271 — Analysis Workbench migration.
"""

import logging

import pandas as pd

from .forge_stats import _col, _col2

logger = logging.getLogger(__name__)


_LEGACY_NARRATIVE_HINTS = {
    "bayes_ab": {
        "next_steps": "If BF₁₀ > 3, the treatment variant is likely better. Consider sample size if BF is near 1.",
        "chart_guidance": "Posterior distributions for each variant. Non-overlapping densities indicate a clear winner.",
    },
    "bayes_anova": {
        "next_steps": "If BF₁₀ > 3, at least one group differs. Follow up with Bayesian pairwise comparisons.",
        "chart_guidance": "Posterior group means with credible intervals. Non-overlapping intervals suggest real differences.",
    },
    "bayes_chi2": {
        "next_steps": "BF₁₀ > 3 suggests association. Examine which cells contribute most to the Bayes Factor.",
        "chart_guidance": "Posterior probability of association. Values near 1 indicate strong dependence.",
    },
    "bayes_equivalence": {
        "next_steps": "BF₁₀ > 3 for equivalence means the groups are practically equivalent within the ROPE.",
        "chart_guidance": "Posterior difference with ROPE boundaries. Density inside ROPE supports equivalence.",
    },
    "bayes_poisson": {
        "next_steps": "BF₁₀ > 3 suggests the rate differs from the null. Examine the posterior rate for planning.",
        "chart_guidance": "Posterior distribution of the Poisson rate parameter.",
    },
    "bayes_regression": {
        "next_steps": "Examine posterior coefficient distributions. Coefficients whose CrI excludes zero are credibly non-zero.",
        "chart_guidance": "Posterior coefficient densities. Narrower posteriors indicate more precise estimates.",
    },
    "bayes_logistic": {
        "next_steps": "Examine posterior odds ratios. CrI excluding 1 indicates a credible effect on the outcome.",
        "chart_guidance": "Posterior log-odds coefficients. Transform to odds ratios for interpretation.",
    },
    "bayes_survival": {
        "next_steps": "Posterior hazard ratio indicates relative risk. CrI excluding 1 suggests a real effect on survival.",
        "chart_guidance": "Posterior survival curves with credible bands. Separation indicates group differences.",
    },
    "bayes_meta": {
        "next_steps": "Posterior pooled effect summarizes evidence across studies. Check heterogeneity (τ²) for consistency.",
        "chart_guidance": "Forest plot with posterior effect sizes per study and pooled estimate.",
    },
    "bayes_demo": {
        "next_steps": "This is a demonstration of Bayesian inference with synthetic data.",
        "chart_guidance": "Prior vs posterior shows how observed data updates the initial belief.",
    },
    "bayes_spares": {
        "next_steps": "Posterior spare parts demand distribution informs stocking decisions. Use the credible interval for safety stock.",
        "chart_guidance": "Posterior predictive distribution for future demand periods.",
    },
    "bayes_system": {
        "next_steps": "System reliability posterior gives P(system survives to time t). Use for maintenance scheduling.",
        "chart_guidance": "Posterior system reliability function with credible bands.",
    },
    "bayes_warranty": {
        "next_steps": "Posterior warranty return rate informs reserve planning. CrI width reflects data sufficiency.",
        "chart_guidance": "Posterior return rate over warranty period with credible bands.",
    },
    "bayes_repairable": {
        "next_steps": "Posterior repair rate trend (NHPP intensity) indicates whether the system is improving or degrading.",
        "chart_guidance": "Posterior intensity function — increasing means degradation, decreasing means improvement.",
    },
    "bayes_rul": {
        "next_steps": "Posterior remaining useful life distribution informs replacement timing. Median RUL is the planning target.",
        "chart_guidance": "Posterior RUL density — the mode is the most likely time to failure.",
    },
    "bayes_alt": {
        "next_steps": "Extrapolate posterior life distribution to use conditions. Credible interval reflects accelerated test uncertainty.",
        "chart_guidance": "Posterior life distribution at use stress with credible bands.",
    },
    "bayes_comprisk": {
        "next_steps": "Posterior cause-specific hazards identify dominant failure modes. Prioritize the mode with highest hazard.",
        "chart_guidance": "Posterior cumulative incidence functions by failure cause.",
    },
}


def _wrap_legacy(analysis_id, df, config):
    """Call legacy bayesian handler, normalize output to forge schema."""
    from .bayesian import run_bayesian_analysis

    result = run_bayesian_analysis(df, analysis_id, config)
    if result is None:
        return None

    # Normalize: ensure all required keys exist
    result.setdefault("plots", [])
    result.setdefault("statistics", {})
    result.setdefault("summary", "")

    hints = _LEGACY_NARRATIVE_HINTS.get(analysis_id, {})
    if "narrative" not in result or not isinstance(result.get("narrative"), dict):
        result["narrative"] = {
            "verdict": result.get("guide_observation", "") or result.get("summary", "").split("\n")[0][:120],
            "body": result.get("summary", ""),
            "next_steps": hints.get(
                "next_steps", "Examine the posterior distribution and Bayes Factor for decision-making."
            ),
            "chart_guidance": hints.get(
                "chart_guidance", "Posterior density shows updated beliefs after observing data."
            ),
        }
    else:
        # Narrative exists but may have empty fields — fill from hints
        narr = result["narrative"]
        if not narr.get("next_steps"):
            narr["next_steps"] = hints.get(
                "next_steps", "Examine the posterior distribution and Bayes Factor for decision-making."
            )
        if not narr.get("chart_guidance"):
            narr["chart_guidance"] = hints.get(
                "chart_guidance", "Posterior density shows updated beliefs after observing data."
            )
        if not narr.get("verdict"):
            narr["verdict"] = result.get("guide_observation", "") or result.get("summary", "").split("\n")[0][:120]
        if not narr.get("body"):
            narr["body"] = result.get("summary", "")

    result.setdefault("assumptions", {})
    result.setdefault("diagnostics", [])
    result.setdefault("guide_observation", "")
    return result


_FORGE_NATIVE_NEXT_STEPS = {
    "One-Sample t-Test": "BF₁₀ > 10 = strong evidence the mean differs from the test value. Assess the credible interval for practical significance.",
    "Two-Sample t-Test": "BF₁₀ > 10 = strong evidence the groups differ. The posterior difference gives the likely effect magnitude.",
    "Correlation": "BF₁₀ > 10 = strong evidence of correlation. The posterior gives the likely strength, not just direction.",
    "Proportion": "BF₁₀ > 10 = strong evidence the proportion differs from null. Use the posterior for planning.",
}


def _bayes_result(r, test_name):
    """Format a forgestat BayesianTestResult into the dispatch schema."""
    # Prior vs posterior density chart
    plots = []
    try:
        import numpy as np
        from forgeviz.charts.generic import multi_line
        from scipy.stats import norm

        mu_post = r.posterior_mean
        sd_post = r.posterior_std
        # Prior: vague normal centered at 0
        mu_prior = 0
        sd_prior = max(sd_post * 5, 1)

        x_min = min(mu_prior - 3 * sd_prior, mu_post - 4 * sd_post)
        x_max = max(mu_prior + 3 * sd_prior, mu_post + 4 * sd_post)
        x = np.linspace(x_min, x_max, 200).tolist()
        prior_y = norm.pdf(x, mu_prior, sd_prior).tolist()
        post_y = norm.pdf(x, mu_post, sd_post).tolist()

        spec = multi_line(
            x=x,
            series={"Prior": prior_y, "Posterior": post_y},
            title=f"Bayesian {test_name}: Prior vs Posterior",
        )
        plots = [spec.to_dict()]
    except Exception:
        pass  # Chart is optional — don't block result

    # Analysis-specific next steps
    next_steps = _FORGE_NATIVE_NEXT_STEPS.get(
        test_name,
        "BF₁₀ > 3 = moderate, > 10 = strong, > 30 = very strong evidence.",
    )

    return {
        "plots": plots,
        "statistics": {
            "bf10": round(r.bf10, 4),
            "bf01": round(r.bf01, 4),
            "bf_label": r.bf_label,
            "posterior_mean": round(r.posterior_mean, 4),
            "posterior_std": round(r.posterior_std, 4),
            "credible_interval": [round(r.credible_interval[0], 4), round(r.credible_interval[1], 4)],
            "ci_level": r.ci_level,
            "p_rope": round(r.p_rope, 4) if r.p_rope is not None else None,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Bayesian {test_name}<</COLOR>>\n\n"
            f"<<COLOR:text>>BF\u2081\u2080 = {r.bf10:.2f} ({r.bf_label})<</COLOR>>\n"
            f"<<COLOR:text>>Posterior: {r.posterior_mean:.4f} \u00b1 {r.posterior_std:.4f}<</COLOR>>\n"
            f"<<COLOR:text>>{r.ci_level * 100:.0f}% CrI: [{r.credible_interval[0]:.4f}, {r.credible_interval[1]:.4f}]<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"BF\u2081\u2080 = {r.bf10:.2f} ({r.bf_label} evidence)",
            "body": (
                f"Bayes Factor = {r.bf10:.2f}: data are {r.bf10:.1f}\u00d7 more likely under H\u2081 than H\u2080 "
                f"({r.bf_label}). Posterior mean = {r.posterior_mean:.4f}."
            ),
            "next_steps": next_steps,
            "chart_guidance": "Prior vs posterior density shows how data updated beliefs. The posterior narrowing relative to the prior reflects information gained.",
        },
        "guide_observation": f"Bayes {test_name}: BF\u2081\u2080={r.bf10:.2f} ({r.bf_label}), post={r.posterior_mean:.4f}.",
        "diagnostics": [],
    }


# =============================================================================
# Forge-backed Bayesian Tests (forgestat)
# =============================================================================


def forge_bayes_ttest(df, config):
    """Bayesian t-test via forgestat."""
    from forgestat.bayesian.tests import bayesian_ttest_one_sample, bayesian_ttest_two_sample

    var1 = config.get("var1") or config.get("column")
    var2 = config.get("var2")
    factor = config.get("factor")
    response = config.get("response")

    if factor and response and factor in df.columns and response in df.columns:
        groups = df.groupby(factor)[response].apply(lambda s: s.dropna().values)
        if len(groups) >= 2:
            r = bayesian_ttest_two_sample(groups.iloc[0].tolist(), groups.iloc[1].tolist())
            return _bayes_result(r, "Two-Sample t-Test")

    if var2 and var2 in df.columns:
        x1 = pd.to_numeric(df[var1], errors="coerce").dropna().values
        x2 = pd.to_numeric(df[var2], errors="coerce").dropna().values
        r = bayesian_ttest_two_sample(x1.tolist(), x2.tolist())
        return _bayes_result(r, "Two-Sample t-Test")

    data, col_name = _col(df, config, "column", "var1")
    mu = float(config.get("test_value", config.get("mu", 0)))
    r = bayesian_ttest_one_sample(data.tolist(), mu=mu)
    return _bayes_result(r, "One-Sample t-Test")


def forge_bayes_correlation(df, config):
    """Bayesian correlation via forgestat."""
    from forgestat.bayesian.tests import bayesian_correlation

    c1, n1, c2, n2 = _col2(df, config)
    r = bayesian_correlation(c1.tolist(), c2.tolist())
    return _bayes_result(r, "Correlation")


def forge_bayes_proportion(df, config):
    """Bayesian proportion test via forgestat."""
    from forgestat.bayesian.tests import bayesian_proportion

    successes = int(config.get("successes", 0))
    n = int(config.get("n", 1))
    prior_a = float(config.get("prior_a", 1))
    prior_b = float(config.get("prior_b", 1))

    if successes == 0 and "column" in config:
        data, _ = _col(df, config, "column", "var1")
        successes = int((data > 0).sum())
        n = len(data)

    r = bayesian_proportion(successes, n, prior_a=prior_a, prior_b=prior_b)
    return _bayes_result(r, "Proportion")


# =============================================================================
# Forge-backed Bayesian SPC (forgespc)
# =============================================================================


def forge_bayes_changepoint(df, config):
    """Bayesian changepoint detection via forgespc."""
    from forgespc.bayesian import bayesian_changepoint

    data, col_name = _col(df, config, "column", "var1")
    lsl = config.get("lsl")
    usl = config.get("usl")

    result = bayesian_changepoint(
        data.tolist(),
        min_segment=int(config.get("min_segment", 10)),
        usl=float(usl) if usl is not None else None,
        lsl=float(lsl) if lsl is not None else None,
    )

    n_cp = len(result.changepoints) if result.changepoints else 0
    return {
        "plots": [],
        "statistics": {
            "n_changepoints": n_cp,
            "changepoints": result.changepoints,
            "n_segments": len(result.segments) if result.segments else 0,
            "log_evidence": round(result.log_evidence, 4) if result.log_evidence else None,
            "n": result.n,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Bayesian Changepoint Detection<</COLOR>>\n\n"
            f"<<COLOR:text>>{n_cp} changepoint(s) detected in {result.n} observations<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"{n_cp} Bayesian changepoint{'s' if n_cp != 1 else ''}",
            "body": f"Bayesian changepoint analysis found {n_cp} regime changes.",
            "next_steps": "Examine segment means and variances for process understanding.",
            "chart_guidance": "Vertical lines at changepoints, segment means shown.",
        },
        "guide_observation": f"Bayes changepoint: {n_cp} detected.",
        "diagnostics": [],
    }


def forge_bayes_capability_prediction(df, config):
    """Bayesian capability prediction via forgespc."""
    from forgespc.bayesian import bayesian_capability

    data, col_name = _col(df, config, "column", "var1")
    usl = config.get("usl")
    lsl = config.get("lsl")
    target = config.get("target")

    result = bayesian_capability(
        data.tolist(),
        usl=float(usl) if usl is not None else None,
        lsl=float(lsl) if lsl is not None else None,
        target=float(target) if target is not None else None,
    )

    return {
        "plots": [],
        "statistics": {
            "cpk_median": round(result.cpk_median, 4),
            "cpk_ci": [round(result.cpk_ci[0], 4), round(result.cpk_ci[1], 4)],
            "cp_median": round(result.cp_median, 4) if result.cp_median else None,
            "p_gt_133": round(result.p_gt_133, 4),
            "sigma_level": round(result.sigma_level, 2) if result.sigma_level else None,
            "yield_pct": round(result.yield_pct, 4) if result.yield_pct else None,
            "verdict": result.verdict,
            "n": result.n,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Bayesian Capability Prediction<</COLOR>>\n\n"
            f"<<COLOR:text>>Cpk = {result.cpk_median:.4f} [{result.cpk_ci[0]:.4f}, {result.cpk_ci[1]:.4f}]<</COLOR>>\n"
            f"<<COLOR:text>>P(Cpk > 1.33) = {result.p_gt_133:.1%}<</COLOR>>\n"
            f"<<COLOR:text>>{result.verdict}<</COLOR>>"
        ),
        "narrative": {
            "verdict": result.verdict,
            "body": f"Bayesian Cpk = {result.cpk_median:.4f} with {result.p_gt_133:.1%} probability of exceeding 1.33.",
            "next_steps": "Bayesian Cpk accounts for parameter uncertainty — more honest than frequentist.",
            "chart_guidance": "Posterior distribution of Cpk with reference lines at 1.0 and 1.33.",
        },
        "guide_observation": f"Bayes Cpk: {result.cpk_median:.4f}, P(>1.33)={result.p_gt_133:.1%}.",
        "diagnostics": [],
    }


def forge_bayes_ewma(df, config):
    """Bayesian EWMA control chart via forgespc."""
    from forgespc.bayesian import bayesian_control_chart

    data, col_name = _col(df, config, "column", "var1")
    credible = float(config.get("credible_level", 0.99))

    result = bayesian_control_chart(data.tolist(), credible_level=credible)

    n_ooc = len(result.out_of_control) if result.out_of_control else 0
    status = "IN CONTROL" if result.in_control else "OUT OF CONTROL"

    return {
        "plots": [],
        "statistics": {
            "n": result.n,
            "n_ooc": n_ooc,
            "status": status,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Bayesian Control Chart<</COLOR>>\n\n"
            f"<<COLOR:text>>N = {result.n}, {n_ooc} OOC points<</COLOR>>\n"
            f"<<COLOR:text>>Status: {status}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Bayesian control: {status}",
            "body": f"Bayesian control chart with {credible * 100:.0f}% credible limits. {n_ooc} out-of-control points.",
            "next_steps": "Bayesian limits adapt as data accumulates.",
            "chart_guidance": "Posterior mean with credible interval bands.",
        },
        "guide_observation": f"Bayes control: {n_ooc} OOC, {status}.",
        "diagnostics": [],
    }


# =============================================================================
# Legacy-wrapped Bayesian handlers
# =============================================================================


def forge_bayes_ab(df, config):
    return _wrap_legacy("bayes_ab", df, config)


def forge_bayes_anova(df, config):
    return _wrap_legacy("bayes_anova", df, config)


def forge_bayes_chi2(df, config):
    return _wrap_legacy("bayes_chi2", df, config)


def forge_bayes_equivalence(df, config):
    return _wrap_legacy("bayes_equivalence", df, config)


def forge_bayes_poisson(df, config):
    return _wrap_legacy("bayes_poisson", df, config)


def forge_bayes_regression(df, config):
    return _wrap_legacy("bayes_regression", df, config)


def forge_bayes_logistic(df, config):
    return _wrap_legacy("bayes_logistic", df, config)


def forge_bayes_survival(df, config):
    return _wrap_legacy("bayes_survival", df, config)


def forge_bayes_meta(df, config):
    return _wrap_legacy("bayes_meta", df, config)


def forge_bayes_demo(df, config):
    return _wrap_legacy("bayes_demo", df, config)


def forge_bayes_spares(df, config):
    return _wrap_legacy("bayes_spares", df, config)


def forge_bayes_system(df, config):
    return _wrap_legacy("bayes_system", df, config)


def forge_bayes_warranty(df, config):
    return _wrap_legacy("bayes_warranty", df, config)


def forge_bayes_repairable(df, config):
    return _wrap_legacy("bayes_repairable", df, config)


def forge_bayes_rul(df, config):
    return _wrap_legacy("bayes_rul", df, config)


def forge_bayes_alt(df, config):
    return _wrap_legacy("bayes_alt", df, config)


def forge_bayes_comprisk(df, config):
    return _wrap_legacy("bayes_comprisk", df, config)


# =============================================================================
# Dispatch
# =============================================================================

FORGE_BAYESIAN_HANDLERS = {
    # Forgestat-backed
    "bayes_ttest": forge_bayes_ttest,
    "bayes_correlation": forge_bayes_correlation,
    "bayes_proportion": forge_bayes_proportion,
    # Forgespc-backed
    "bayes_changepoint": forge_bayes_changepoint,
    "bayes_capability_prediction": forge_bayes_capability_prediction,
    "bayes_ewma": forge_bayes_ewma,
    # Legacy-wrapped
    "bayes_ab": forge_bayes_ab,
    "bayes_anova": forge_bayes_anova,
    "bayes_chi2": forge_bayes_chi2,
    "bayes_equivalence": forge_bayes_equivalence,
    "bayes_poisson": forge_bayes_poisson,
    "bayes_regression": forge_bayes_regression,
    "bayes_logistic": forge_bayes_logistic,
    "bayes_survival": forge_bayes_survival,
    "bayes_meta": forge_bayes_meta,
    "bayes_demo": forge_bayes_demo,
    "bayes_spares": forge_bayes_spares,
    "bayes_system": forge_bayes_system,
    "bayes_warranty": forge_bayes_warranty,
    "bayes_repairable": forge_bayes_repairable,
    "bayes_rul": forge_bayes_rul,
    "bayes_alt": forge_bayes_alt,
    "bayes_comprisk": forge_bayes_comprisk,
}


def run_forge_bayesian(analysis_id, df, config):
    """Run a forge-backed Bayesian analysis.

    Returns the result dict, or None if not ported.
    """
    handler = FORGE_BAYESIAN_HANDLERS.get(analysis_id)
    if handler is None:
        return None
    try:
        return handler(df, config)
    except Exception:
        logger.exception(f"Forge Bayesian handler failed for {analysis_id}, falling back to legacy")
        return None

"""Forge-backed MSA, agreement, nonparametric, and exploratory handlers.

Split from forge_stats.py for compliance (3000-line limit).
Object 271 — Analysis Workbench migration.
"""

import logging

import pandas as pd

from .forge_stats import _alpha, _col, _col2, _pval_str

logger = logging.getLogger(__name__)


# =============================================================================
# MSA / Agreement
# =============================================================================


def forge_icc(df, config):
    """Intraclass correlation via forgestat."""
    from forgestat.msa.agreement import icc

    rater_cols = config.get("raters") or config.get("columns") or []
    if not rater_cols:
        rater_cols = list(df.select_dtypes(include="number").columns)
    if len(rater_cols) < 2:
        raise ValueError("ICC requires at least 2 rater columns")

    clean = df[rater_cols].dropna()
    ratings = [clean[c].tolist() for c in rater_cols]
    icc_type = config.get("icc_type", "ICC(3,1)")

    result = icc(ratings, icc_type=icc_type)

    return {
        "plots": [],
        "statistics": {
            "icc": round(result.icc, 4),
            "icc_type": result.icc_type,
            "ci_lower": round(result.ci_lower, 4),
            "ci_upper": round(result.ci_upper, 4),
            "f_statistic": round(result.f_statistic, 4),
            "p_value": round(result.p_value, 6),
            "n_subjects": result.n_subjects,
            "n_raters": result.n_raters,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Intraclass Correlation ({result.icc_type})<</COLOR>>\n\n"
            f"<<COLOR:text>>ICC = {result.icc:.4f} [{result.ci_lower:.4f}, {result.ci_upper:.4f}]<</COLOR>>\n"
            f"<<COLOR:text>>F = {result.f_statistic:.3f}, p = {_pval_str(result.p_value)}<</COLOR>>\n"
            f"<<COLOR:text>>{result.n_subjects} subjects, {result.n_raters} raters<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"ICC = {result.icc:.4f} ({'excellent' if result.icc > 0.9 else 'good' if result.icc > 0.75 else 'moderate' if result.icc > 0.5 else 'poor'})",
            "body": f"Intraclass correlation ({result.icc_type}) = {result.icc:.4f}, 95% CI [{result.ci_lower:.4f}, {result.ci_upper:.4f}].",
            "next_steps": "ICC > 0.75 is generally considered good reliability. Consider sources of disagreement if low.",
            "chart_guidance": "",
        },
        "guide_observation": f"ICC({result.icc_type}) = {result.icc:.4f}, p={_pval_str(result.p_value)}.",
        "diagnostics": [],
    }


def forge_bland_altman(df, config):
    """Bland-Altman agreement analysis via forgestat."""
    from forgestat.msa.agreement import bland_altman

    c1, n1, c2, n2 = _col2(df, config)
    result = bland_altman(c1.tolist(), c2.tolist())

    return {
        "plots": [],
        "statistics": {
            "mean_diff": round(result.mean_diff, 4),
            "std_diff": round(result.std_diff, 4),
            "loa_lower": round(result.loa_lower, 4),
            "loa_upper": round(result.loa_upper, 4),
            "n": result.n,
            "proportional_bias": result.proportional_bias,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Bland-Altman Agreement<</COLOR>>\n\n"
            f"<<COLOR:text>>{n1} vs {n2}, N = {result.n}<</COLOR>>\n"
            f"<<COLOR:text>>Mean difference (bias): {result.mean_diff:.4f}<</COLOR>>\n"
            f"<<COLOR:text>>Limits of agreement: [{result.loa_lower:.4f}, {result.loa_upper:.4f}]<</COLOR>>\n"
            f"<<COLOR:text>>Proportional bias: {'yes' if result.proportional_bias else 'no'}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Bias = {result.mean_diff:.4f}, LoA = [{result.loa_lower:.4f}, {result.loa_upper:.4f}]",
            "body": f"Mean difference = {result.mean_diff:.4f} \u00b1 {result.std_diff:.4f}. 95% limits of agreement: [{result.loa_lower:.4f}, {result.loa_upper:.4f}].",
            "next_steps": "Check if LoA are clinically acceptable. Proportional bias suggests method agreement varies with magnitude.",
            "chart_guidance": "Points should scatter randomly around zero with no trend.",
        },
        "guide_observation": f"Bland-Altman: bias={result.mean_diff:.4f}, LoA=[{result.loa_lower:.4f}, {result.loa_upper:.4f}].",
        "diagnostics": [],
    }


def forge_krippendorff_alpha(df, config):
    """Krippendorff's alpha via forgestat."""
    from forgestat.msa.kappa import krippendorff_alpha

    rater_cols = config.get("raters") or config.get("columns") or list(df.columns)
    level = config.get("level", "nominal")
    clean = df[rater_cols].values.tolist()

    result = krippendorff_alpha(clean, level=level)

    return {
        "plots": [],
        "statistics": {
            "alpha": round(result.value, 4),
            "method": result.method,
            "ci_lower": round(result.ci_lower, 4) if result.ci_lower else None,
            "ci_upper": round(result.ci_upper, 4) if result.ci_upper else None,
            "n_subjects": result.n_subjects,
            "n_raters": result.n_raters,
            "interpretation": result.interpretation,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Krippendorff's Alpha ({level})<</COLOR>>\n\n"
            f"<<COLOR:text>>\u03b1 = {result.value:.4f} ({result.interpretation})<</COLOR>>\n"
            f"<<COLOR:text>>{result.n_subjects} subjects, {result.n_raters} raters<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Krippendorff's \u03b1 = {result.value:.4f} ({result.interpretation})",
            "body": f"Inter-rater reliability: \u03b1 = {result.value:.4f} at {level} level. {result.interpretation}.",
            "next_steps": "\u03b1 > 0.80 is reliable, 0.67-0.80 tentatively acceptable. Below 0.67 needs improvement.",
            "chart_guidance": "",
        },
        "guide_observation": f"Krippendorff's \u03b1={result.value:.4f} ({result.interpretation}).",
        "diagnostics": [],
    }


def forge_gage_linearity_bias(df, config):
    """Gage linearity and bias via forgestat."""
    from forgestat.msa.agreement import linearity_bias

    ref_col = config.get("reference") or config.get("var1")
    meas_col = config.get("measured") or config.get("var2") or config.get("column")
    if not ref_col or ref_col not in df.columns:
        raise ValueError(f"Reference column '{ref_col}' not found")
    if not meas_col or meas_col not in df.columns:
        raise ValueError(f"Measured column '{meas_col}' not found")

    clean = df[[ref_col, meas_col]].dropna()
    reference = pd.to_numeric(clean[ref_col], errors="coerce").dropna()
    measured = pd.to_numeric(clean[meas_col], errors="coerce").dropna()
    n = min(len(reference), len(measured))

    result = linearity_bias(reference.values[:n].tolist(), measured.values[:n].tolist())

    return {
        "plots": [],
        "statistics": {
            "overall_bias": round(result.overall_bias, 4),
            "linearity_slope": round(result.linearity_slope, 4),
            "linearity_r_squared": round(result.linearity_r_squared, 4),
            "linearity_p_value": round(result.linearity_p_value, 6),
            "bias_significant": result.bias_significant,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Gage Linearity & Bias<</COLOR>>\n\n"
            f"<<COLOR:text>>Overall bias: {result.overall_bias:.4f}<</COLOR>>\n"
            f"<<COLOR:text>>Linearity: slope = {result.linearity_slope:.4f}, R\u00b2 = {result.linearity_r_squared:.4f}, p = {_pval_str(result.linearity_p_value)}<</COLOR>>\n"
            f"<<COLOR:text>>Bias significant: {'yes' if result.bias_significant else 'no'}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Bias = {result.overall_bias:.4f}, linearity slope = {result.linearity_slope:.4f}",
            "body": f"Overall bias = {result.overall_bias:.4f}. Linearity slope = {result.linearity_slope:.4f} (ideally 0). R\u00b2 = {result.linearity_r_squared:.4f}.",
            "next_steps": "Significant linearity means bias changes across the measurement range — calibration needed.",
            "chart_guidance": "Scatter plot of bias vs reference value. Flat line = no linearity issue.",
        },
        "guide_observation": f"Linearity: slope={result.linearity_slope:.4f}, bias={result.overall_bias:.4f}.",
        "diagnostics": [],
    }


def forge_attribute_agreement(df, config):
    """Attribute agreement analysis via forgestat (Fleiss' kappa)."""
    from forgestat.msa.kappa import fleiss_kappa

    rater_cols = config.get("raters") or config.get("columns") or list(df.select_dtypes(include="number").columns)
    n_raters = len(rater_cols)
    if n_raters < 2:
        raise ValueError("Need at least 2 raters")

    ratings_matrix = df[rater_cols].dropna().values.astype(int).tolist()
    result = fleiss_kappa(ratings_matrix, n_raters=n_raters)

    return {
        "plots": [],
        "statistics": {
            "kappa": round(result.value, 4),
            "method": result.method,
            "ci_lower": round(result.ci_lower, 4) if result.ci_lower else None,
            "ci_upper": round(result.ci_upper, 4) if result.ci_upper else None,
            "n_subjects": result.n_subjects,
            "n_raters": result.n_raters,
            "interpretation": result.interpretation,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Attribute Agreement Analysis<</COLOR>>\n\n"
            f"<<COLOR:text>>\u03ba = {result.value:.4f} ({result.interpretation})<</COLOR>>\n"
            f"<<COLOR:text>>{result.n_subjects} subjects, {result.n_raters} raters<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Fleiss' \u03ba = {result.value:.4f} ({result.interpretation})",
            "body": f"Inter-rater agreement: \u03ba = {result.value:.4f}. {result.interpretation}.",
            "next_steps": "\u03ba > 0.75 is excellent. 0.40-0.75 is fair to good. Below 0.40 is poor.",
            "chart_guidance": "",
        },
        "guide_observation": f"Attribute agreement: \u03ba={result.value:.4f} ({result.interpretation}).",
        "diagnostics": [],
    }


# =============================================================================
# Nonparametric — remaining
# =============================================================================


def forge_multi_vari(df, config):
    """Multi-vari analysis via forgestat."""
    from forgestat.exploratory.multi_vari import multi_vari

    response = config.get("response") or config.get("var1") or config.get("column")
    factors = config.get("factors") or config.get("group_vars") or []
    if not response or response not in df.columns:
        raise ValueError(f"Response column '{response}' not found")
    if not factors:
        factors = [c for c in df.columns if c != response and df[c].dtype == object][:2]

    data_dict = {response: df[response].tolist()}
    for f in factors:
        data_dict[f] = df[f].tolist()

    result = multi_vari(data_dict, response=response, factors=factors)

    source_lines = []
    for src in result.sources:
        source_lines.append(
            f"  <<COLOR:highlight>>{src.factor}:<</COLOR>> variance = {src.variance:.4f} ({src.pct_contribution:.1f}%)"
        )

    return {
        "plots": [],
        "statistics": {
            "total_variance": round(result.total_variance, 4),
            "within_variance": round(result.within_variance, 4),
            "within_pct": round(result.within_pct, 1),
            "dominant_source": result.dominant_source,
            "grand_mean": round(result.grand_mean, 4),
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Multi-Vari Analysis<</COLOR>>\n\n"
            f"<<COLOR:text>>Response: {response}, Factors: {factors}<</COLOR>>\n"
            f"<<COLOR:text>>Dominant source: {result.dominant_source}<</COLOR>>\n\n" + "\n".join(source_lines)
        ),
        "narrative": {
            "verdict": f"Dominant variation source: {result.dominant_source}",
            "body": f"Multi-vari chart shows variation decomposition across {len(factors)} factors. Within-unit variation = {result.within_pct:.1f}%.",
            "next_steps": "Focus improvement on the dominant source of variation.",
            "chart_guidance": "Nested dots show variation sources. Wider spread = more variation from that factor.",
        },
        "guide_observation": f"Multi-vari: dominant source={result.dominant_source}, within={result.within_pct:.1f}%.",
        "diagnostics": [],
    }


def forge_runs_test(df, config):
    """Runs test for randomness via forgestat."""
    from forgestat.nonparametric.rank_tests import runs_test

    data, col_name = _col(df, config, "column", "var1")
    cutoff = config.get("cutoff")
    if cutoff is not None:
        cutoff = float(cutoff)

    result = runs_test(data.tolist(), cutoff=cutoff)

    return {
        "plots": [],
        "statistics": {
            "test_statistic": round(result.statistic, 4),
            "p_value": round(result.p_value, 6),
            "n": len(data),
            "significant": result.p_value < _alpha(config),
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Runs Test for Randomness<</COLOR>>\n\n"
            f"<<COLOR:text>>Z = {result.statistic:.4f}, p = {_pval_str(result.p_value)}<</COLOR>>\n"
            f"<<COLOR:text>>{'Non-random pattern detected' if result.p_value < _alpha(config) else 'No evidence of non-randomness'}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"{'Non-random' if result.p_value < _alpha(config) else 'Random'} (p = {_pval_str(result.p_value)})",
            "body": f"Runs test: Z = {result.statistic:.4f}, p = {_pval_str(result.p_value)}.",
            "next_steps": "If non-random, look for trends, cycles, or shifts in the data.",
            "chart_guidance": "Sequence colored by runs above/below the median.",
        },
        "guide_observation": f"Runs test: Z={result.statistic:.4f}, p={_pval_str(result.p_value)}.",
        "diagnostics": [],
    }


# =============================================================================
# Exploratory
# =============================================================================


def forge_hotelling_t2(df, config):
    """Hotelling's T-squared test via forgestat."""
    from forgestat.exploratory.multivariate import hotelling_t2_one_sample

    cols = config.get("columns") or config.get("variables") or list(df.select_dtypes(include="number").columns)
    mu = config.get("mu") or config.get("test_values")
    data_dict = {}
    for c in cols:
        if c in df.columns:
            data_dict[c] = pd.to_numeric(df[c], errors="coerce").dropna().tolist()

    result = hotelling_t2_one_sample(data_dict, mu=mu)

    return {
        "plots": [],
        "statistics": {
            "t2_statistic": round(result.t2_statistic, 4),
            "f_statistic": round(result.f_statistic, 4),
            "p_value": round(result.p_value, 6),
            "df1": result.df1,
            "df2": result.df2,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Hotelling's T\u00b2 Test<</COLOR>>\n\n"
            f"<<COLOR:text>>T\u00b2 = {result.t2_statistic:.4f}, F = {result.f_statistic:.4f}, p = {_pval_str(result.p_value)}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"T\u00b2 = {result.t2_statistic:.4f}, p = {_pval_str(result.p_value)}",
            "body": f"Hotelling's T\u00b2 = {result.t2_statistic:.4f} (F({result.df1},{result.df2}) = {result.f_statistic:.4f}, p = {_pval_str(result.p_value)}).",
            "next_steps": "If significant, examine individual variables to identify which contribute.",
            "chart_guidance": "",
        },
        "guide_observation": f"Hotelling T\u00b2={result.t2_statistic:.4f}, p={_pval_str(result.p_value)}.",
        "diagnostics": [],
    }


def forge_manova(df, config):
    """MANOVA via forgestat."""
    from forgestat.exploratory.multivariate import one_way_manova

    response_cols = config.get("responses") or config.get("columns") or list(df.select_dtypes(include="number").columns)
    group_col = config.get("factor") or config.get("group_var")
    if not group_col or group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found")

    data_dict = {}
    clean = df[response_cols + [group_col]].dropna()
    for c in response_cols:
        data_dict[c] = pd.to_numeric(clean[c], errors="coerce").tolist()
    groups = clean[group_col].tolist()

    result = one_way_manova(data_dict, groups=groups)

    return {
        "plots": [],
        "statistics": {
            "wilks_lambda": round(result.wilks_lambda, 4),
            "f_statistic": round(result.f_statistic, 4),
            "p_value": round(result.p_value, 6),
            "df1": result.df1,
            "df2": result.df2,
            "pillai_trace": round(result.pillai_trace, 4) if result.pillai_trace else None,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>MANOVA<</COLOR>>\n\n"
            f"<<COLOR:text>>Wilks' \u039b = {result.wilks_lambda:.4f}, F({result.df1},{result.df2}) = {result.f_statistic:.4f}, p = {_pval_str(result.p_value)}<</COLOR>>"
            + (f"\n<<COLOR:text>>Pillai's trace = {result.pillai_trace:.4f}<</COLOR>>" if result.pillai_trace else "")
        ),
        "narrative": {
            "verdict": f"MANOVA: {'significant' if result.p_value < 0.05 else 'not significant'} (p = {_pval_str(result.p_value)})",
            "body": f"Wilks' \u039b = {result.wilks_lambda:.4f}. Groups {'differ' if result.p_value < 0.05 else 'do not differ'} significantly on the multivariate response.",
            "next_steps": "Follow up with univariate ANOVAs and discriminant analysis.",
            "chart_guidance": "",
        },
        "guide_observation": f"MANOVA: \u039b={result.wilks_lambda:.4f}, p={_pval_str(result.p_value)}.",
        "diagnostics": [],
    }


def forge_meta_analysis(df, config):
    """Meta-analysis via forgestat."""
    from forgestat.exploratory.meta import meta_analysis

    effect_col = config.get("effect_col") or config.get("effect") or "effect"
    se_col = config.get("se_col") or config.get("se") or "se"
    name_col = config.get("study_col") or config.get("study") or "study"
    model = config.get("model", "random")

    if effect_col not in df.columns or se_col not in df.columns:
        raise ValueError(f"Need '{effect_col}' and '{se_col}' columns")

    clean = df.dropna(subset=[effect_col, se_col])
    effects = pd.to_numeric(clean[effect_col], errors="coerce").dropna()
    clean = clean.loc[effects.index]
    ses = pd.to_numeric(clean[se_col], errors="coerce").values
    effects = effects.values
    names = clean[name_col].tolist() if name_col in clean.columns else None

    result = meta_analysis(effects.tolist(), ses.tolist(), study_names=names, model=model)

    return {
        "plots": [],
        "statistics": {
            "pooled_effect": round(result.pooled_effect, 4),
            "pooled_se": round(result.pooled_se, 4),
            "pooled_ci_lower": round(result.pooled_ci_lower, 4),
            "pooled_ci_upper": round(result.pooled_ci_upper, 4),
            "pooled_p": round(result.pooled_p, 6),
            "i_squared": round(result.i_squared, 1),
            "tau_squared": round(result.tau_squared, 4) if result.tau_squared else None,
            "q_statistic": round(result.q_statistic, 4),
            "q_p_value": round(result.q_p_value, 6),
            "k": result.k,
            "model": model,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Meta-Analysis ({model.title()} Effects)<</COLOR>>\n\n"
            f"<<COLOR:text>>Pooled effect = {result.pooled_effect:.4f} [{result.pooled_ci_lower:.4f}, {result.pooled_ci_upper:.4f}]<</COLOR>>\n"
            f"<<COLOR:text>>p = {_pval_str(result.pooled_p)}, k = {result.k} studies<</COLOR>>\n"
            f"<<COLOR:text>>Heterogeneity: I\u00b2 = {result.i_squared:.1f}%, Q = {result.q_statistic:.2f} (p = {_pval_str(result.q_p_value)})<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Pooled effect = {result.pooled_effect:.4f} (p = {_pval_str(result.pooled_p)}), I\u00b2 = {result.i_squared:.1f}%",
            "body": (
                f"{model.title()} effects meta-analysis of {result.k} studies. "
                f"I\u00b2 = {result.i_squared:.1f}% ({'low' if result.i_squared < 25 else 'moderate' if result.i_squared < 75 else 'high'} heterogeneity)."
            ),
            "next_steps": "High I\u00b2 suggests moderator analysis. Consider funnel plot for publication bias.",
            "chart_guidance": "Forest plot shows individual and pooled effects with confidence intervals.",
        },
        "guide_observation": f"Meta: pooled={result.pooled_effect:.4f}, I\u00b2={result.i_squared:.1f}%, k={result.k}.",
        "diagnostics": [],
    }


def forge_bootstrap_ci(df, config):
    """Bootstrap confidence interval via forgestat."""
    from forgestat.exploratory.univariate import bootstrap_ci

    data, col_name = _col(df, config, "column", "var1")
    stat = config.get("statistic", "mean")
    n_boot = int(config.get("n_bootstrap", config.get("n_boot", 10000)))
    conf = float(config.get("confidence", config.get("conf", 0.95)))

    result = bootstrap_ci(data.tolist(), statistic=stat, n_bootstrap=n_boot, ci_level=conf)

    ci_width = result.ci_upper - result.ci_lower
    return {
        "plots": [],
        "statistics": {
            "estimate": round(result.estimate, 4),
            "ci_lower": round(result.ci_lower, 4),
            "ci_upper": round(result.ci_upper, 4),
            "ci_width": round(ci_width, 4),
            "n_bootstrap": n_boot,
            "confidence": conf,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Bootstrap Confidence Interval<</COLOR>>\n\n"
            f"<<COLOR:text>>Statistic: {stat}, estimate = {result.estimate:.4f}<</COLOR>>\n"
            f"<<COLOR:text>>{conf * 100:.0f}% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]<</COLOR>>\n"
            f"<<COLOR:text>>CI width = {ci_width:.4f}, n_bootstrap = {n_boot}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"{stat} = {result.estimate:.4f} [{result.ci_lower:.4f}, {result.ci_upper:.4f}]",
            "body": f"Bootstrap ({n_boot} resamples): {stat} = {result.estimate:.4f}, {conf * 100:.0f}% CI [{result.ci_lower:.4f}, {result.ci_upper:.4f}].",
            "next_steps": "Bootstrap CIs are non-parametric — valid regardless of distribution shape.",
            "chart_guidance": "Histogram of bootstrap distribution with CI bounds.",
        },
        "guide_observation": f"Bootstrap {stat}: {result.estimate:.4f} [{result.ci_lower:.4f}, {result.ci_upper:.4f}].",
        "diagnostics": [],
    }


# =============================================================================
# MSA — Remaining Gage Studies
# =============================================================================


def forge_gage_type1(df, config):
    """Type 1 gage study (Cg, Cgk)."""
    data, col_name = _col(df, config, "column", "var1")
    reference = float(config.get("reference", config.get("ref", data.mean())))
    tolerance = float(config.get("tolerance", config.get("tol", 6 * data.std(ddof=1))))

    mean = float(data.mean())
    std = float(data.std(ddof=1))
    bias = mean - reference
    n = len(data)

    # Cg = (k * tolerance) / (6 * sigma) where k typically 0.2
    k = float(config.get("k", 0.2))
    cg = (k * tolerance) / (6 * std) if std > 0 else 0
    cgk = (k * tolerance / 2 - abs(bias)) / (3 * std) if std > 0 else 0

    return {
        "plots": [],
        "statistics": {
            "cg": round(cg, 4),
            "cgk": round(cgk, 4),
            "bias": round(bias, 4),
            "repeatability_std": round(std, 4),
            "reference": reference,
            "tolerance": tolerance,
            "n": n,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Type 1 Gage Study<</COLOR>>\n\n"
            f"<<COLOR:text>>Cg = {cg:.4f}, Cgk = {cgk:.4f}<</COLOR>>\n"
            f"<<COLOR:text>>Bias = {bias:.4f}, Repeatability \u03c3 = {std:.4f}<</COLOR>>\n"
            f"<<COLOR:text>>Reference = {reference}, Tolerance = {tolerance}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Cg = {cg:.4f}, Cgk = {cgk:.4f} ({'capable' if cg >= 1.33 and cgk >= 1.33 else 'not capable'})",
            "body": f"Gage capability: Cg = {cg:.4f} (resolution), Cgk = {cgk:.4f} (resolution + bias). Both should exceed 1.33.",
            "next_steps": "Low Cg = poor resolution. Low Cgk = significant bias — recalibrate.",
            "chart_guidance": "",
        },
        "guide_observation": f"Type 1: Cg={cg:.4f}, Cgk={cgk:.4f}, bias={bias:.4f}.",
        "diagnostics": [],
    }


def forge_gage_rr_nested(df, config):
    """Nested Gage R&R (parts nested within operators)."""
    meas_col = config.get("measurement") or config.get("column") or config.get("var")
    part_col = config.get("part") or config.get("part_col")
    op_col = config.get("operator") or config.get("operator_col")

    if not all(c and c in df.columns for c in [meas_col, part_col, op_col]):
        raise ValueError("Nested Gage R&R requires measurement, part, and operator columns")

    clean = df[[meas_col, part_col, op_col]].dropna()
    meas = pd.to_numeric(clean[meas_col], errors="coerce").dropna()
    clean = clean.loc[meas.index]

    # Nested ANOVA: parts nested within operators
    grand_mean = float(meas.mean())
    groups = clean.groupby([op_col, part_col])[meas_col]

    # Operator effect
    op_means = clean.groupby(op_col)[meas_col].mean()
    ss_operator = sum(clean.groupby(op_col).size() * (op_means - grand_mean) ** 2)
    df_operator = len(op_means) - 1

    # Parts within operator
    cell_means = groups.mean()
    cell_sizes = groups.size()
    ss_parts = 0
    for (op, part), cell_mean in cell_means.items():
        ss_parts += cell_sizes[(op, part)] * (cell_mean - op_means[op]) ** 2
    df_parts = len(cell_means) - len(op_means)

    # Repeatability (within cell)
    ss_repeat = sum(groups.apply(lambda g: ((g - g.mean()) ** 2).sum()))
    df_repeat = len(meas) - len(cell_means)

    ms_op = ss_operator / df_operator if df_operator > 0 else 0
    ms_parts = ss_parts / df_parts if df_parts > 0 else 0
    ms_repeat = ss_repeat / df_repeat if df_repeat > 0 else 0

    var_repeat = ms_repeat
    n_per = len(meas) / len(cell_means) if len(cell_means) > 0 else 1
    var_parts = max(0, (ms_parts - ms_repeat) / n_per)
    var_operator = max(0, (ms_op - ms_parts) / (len(meas) / len(op_means))) if len(op_means) > 0 else 0

    total_var = var_repeat + var_parts + var_operator
    grr_pct = (var_repeat + var_operator) / total_var * 100 if total_var > 0 else 0
    part_pct = var_parts / total_var * 100 if total_var > 0 else 0

    return {
        "plots": [],
        "statistics": {
            "var_repeatability": round(var_repeat, 6),
            "var_parts": round(var_parts, 6),
            "var_operator": round(var_operator, 6),
            "total_variance": round(total_var, 6),
            "grr_pct": round(grr_pct, 1),
            "part_pct": round(part_pct, 1),
            "n": len(meas),
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Nested Gage R&R<</COLOR>>\n\n"
            f"<<COLOR:text>>GRR = {grr_pct:.1f}%, Part-to-Part = {part_pct:.1f}%<</COLOR>>\n"
            f"<<COLOR:text>>Repeatability = {var_repeat:.6f}, Operator = {var_operator:.6f}<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"GRR = {grr_pct:.1f}% ({'acceptable' if grr_pct < 10 else 'marginal' if grr_pct < 30 else 'unacceptable'})",
            "body": f"Nested design: parts are unique to each operator. GRR = {grr_pct:.1f}%.",
            "next_steps": "GRR < 10% = acceptable. 10-30% = marginal. > 30% = unacceptable.",
            "chart_guidance": "",
        },
        "guide_observation": f"Nested GRR: {grr_pct:.1f}%, part-to-part: {part_pct:.1f}%.",
        "diagnostics": [],
    }


def forge_gage_rr_expanded(df, config):
    """Expanded Gage R&R (>2 factors)."""
    # This is structurally similar to nested but with additional factors
    # For now, delegate to nested with the same interface
    return forge_gage_rr_nested(df, config)


def forge_attribute_gage(df, config):
    """Attribute gage study (pass/fail agreement)."""
    rater_cols = config.get("raters") or config.get("columns") or list(df.select_dtypes(include="number").columns)
    reference_col = config.get("reference")

    if len(rater_cols) < 1:
        raise ValueError("Need at least 1 rater column")

    clean = df[rater_cols + ([reference_col] if reference_col and reference_col in df.columns else [])].dropna()
    n = len(clean)

    # Within-appraiser agreement (repeatability not applicable for single measurement)
    # Between-appraiser agreement
    if len(rater_cols) >= 2:
        agree_count = sum(clean[rater_cols].nunique(axis=1) == 1)
        between_pct = agree_count / n * 100
    else:
        between_pct = 100.0
        agree_count = n

    # vs reference
    ref_pct = None
    if reference_col and reference_col in clean.columns:
        ref_agree = sum(clean[rater_cols[0]] == clean[reference_col])
        ref_pct = ref_agree / n * 100

    return {
        "plots": [],
        "statistics": {
            "n": n,
            "n_raters": len(rater_cols),
            "between_appraiser_pct": round(between_pct, 1),
            "vs_reference_pct": round(ref_pct, 1) if ref_pct is not None else None,
            "agree_count": agree_count,
        },
        "assumptions": {},
        "summary": (
            f"<<COLOR:header>>Attribute Gage Study<</COLOR>>\n\n"
            f"<<COLOR:text>>Between-appraiser agreement: {between_pct:.1f}%<</COLOR>>\n"
            + (f"<<COLOR:text>>vs Reference: {ref_pct:.1f}%<</COLOR>>\n" if ref_pct is not None else "")
            + f"<<COLOR:text>>{n} subjects, {len(rater_cols)} raters<</COLOR>>"
        ),
        "narrative": {
            "verdict": f"Agreement = {between_pct:.1f}%",
            "body": f"Attribute agreement across {len(rater_cols)} raters: {between_pct:.1f}%.",
            "next_steps": "Target > 90% agreement. Retrain raters if low.",
            "chart_guidance": "",
        },
        "guide_observation": f"Attribute gage: {between_pct:.1f}% agreement, {len(rater_cols)} raters.",
        "diagnostics": [],
    }


# =============================================================================
# Dispatch
# =============================================================================

FORGE_MSA_HANDLERS = {
    # MSA / Agreement
    "icc": forge_icc,
    "bland_altman": forge_bland_altman,
    "krippendorff_alpha": forge_krippendorff_alpha,
    "gage_linearity_bias": forge_gage_linearity_bias,
    "attribute_agreement": forge_attribute_agreement,
    "gage_type1": forge_gage_type1,
    "gage_rr_nested": forge_gage_rr_nested,
    "gage_rr_expanded": forge_gage_rr_expanded,
    "attribute_gage": forge_attribute_gage,
    # Nonparametric remaining
    "multi_vari": forge_multi_vari,
    "runs_test": forge_runs_test,
    # Exploratory
    "hotelling_t2": forge_hotelling_t2,
    "manova": forge_manova,
    "meta_analysis": forge_meta_analysis,
    "bootstrap_ci": forge_bootstrap_ci,
}

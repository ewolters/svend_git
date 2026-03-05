"""DSW Output Post-Processor — enforces canonical output schema on all results.

Called from dispatch.py before safe_json_response(). Ensures every analysis
result has the required fields, applies chart defaults, and validates
enrichment presence.

CR: 5528303a — INIT-009 / E9-002
"""

import logging
import re

from .registry import get_entry, ANALYSIS_REGISTRY
from .chart_defaults import apply_chart_defaults

logger = logging.getLogger(__name__)


# ── Required output fields ─────────────────────────────────────────────────
# Every result dict MUST contain these keys after standardization.
# Missing keys are filled with defaults; presence of enrichment fields
# (education, evidence_grade, bayesian_shadow) is validated and logged.

REQUIRED_FIELDS = {
    "summary": "",
    "plots": [],
    "narrative": None,
    "education": None,
    "diagnostics": [],
    "guide_observation": "",
    "evidence_grade": None,
    "bayesian_shadow": None,
    "what_if": None,
}


def standardize_output(result, analysis_type, analysis_id):
    """Post-process an analysis result dict to enforce output schema.

    Args:
        result: Dict returned by the analysis function.
        analysis_type: e.g. "stats", "spc", "ml"
        analysis_id: e.g. "ttest", "capability", "clustering"

    Returns:
        The same dict, mutated in place with defaults filled and charts styled.
    """
    if not isinstance(result, dict):
        return result

    entry = get_entry(analysis_type, analysis_id)
    key = f"{analysis_type}/{analysis_id}"

    # ── 1. Fill missing required fields ────────────────────────────────
    for field, default in REQUIRED_FIELDS.items():
        if field not in result:
            if isinstance(default, list):
                result[field] = list(default)  # fresh list copy
            else:
                result[field] = default

    # ── 2. Normalize narrative to canonical dict ────────────────────────
    if isinstance(result.get("narrative"), str):
        # Analysis returned a plain string — wrap in canonical dict
        result["narrative"] = _narrative_from_summary(result["narrative"])
    elif not result["narrative"] and result.get("summary"):
        result["narrative"] = _narrative_from_summary(result["summary"])

    # ── 3. Generate guide_observation if missing ───────────────────────
    if not result["guide_observation"] and result.get("summary"):
        clean = re.sub(r"<<COLOR:\w+>>|<</COLOR>>", "", result["summary"])
        result["guide_observation"] = clean[:300] if clean else ""

    # ── 4. Apply chart defaults to all plots ───────────────────────────
    plots = result.get("plots", [])
    for i, plot in enumerate(plots):
        if isinstance(plot, dict):
            apply_chart_defaults(plot)

    # ── 5. Inject education from centralized store ─────────────────────
    if not result["education"]:
        try:
            from .education import get_education
            edu = get_education(analysis_type, analysis_id)
            if edu:
                result["education"] = edu
        except ImportError:
            pass  # education.py not yet created — will be E9-005

    if not result["education"] and entry:
        logger.debug("DSW output: missing education for %s", key)

    # ── 6. Auto-generate bayesian_shadow for shadow-eligible analyses ──
    if entry and entry["shadow_type"] and not result.get("bayesian_shadow"):
        try:
            shadow = _auto_shadow(result, entry["shadow_type"])
            if shadow:
                result["bayesian_shadow"] = shadow
            else:
                logger.debug("DSW output: could not auto-generate shadow for %s", key)
        except Exception:
            logger.debug("DSW output: shadow generation failed for %s", key, exc_info=True)

    # ── 7. Auto-generate evidence_grade for p-value analyses ─────────
    p_val = _extract_p_value(result)
    if p_val is not None and not result.get("evidence_grade"):
        try:
            from .common import _evidence_grade
            bf10 = None
            if result.get("bayesian_shadow"):
                bf10 = result["bayesian_shadow"].get("bf10")
            effect_mag = _classify_effect(result)
            grade = _evidence_grade(p_val, bf10=bf10, effect_magnitude=effect_mag)
            if grade:
                result["evidence_grade"] = grade.get("grade")
                result.setdefault("evidence_rationale", grade.get("rationale", ""))
        except Exception:
            logger.debug("DSW output: evidence grade generation failed for %s", key, exc_info=True)

    # ── 8. Normalize what-if patterns to unified schema ────────────────
    if not result.get("what_if"):
        what_if = _normalize_what_if(result, entry)
        if what_if:
            result["what_if"] = what_if

    # ── 9. Validate statistical output bounds (QUAL-001 §6.2) ─────────
    _validate_statistics_bounds(result)

    # ── 10. Tag with registry metadata ─────────────────────────────────
    result["_analysis_type"] = analysis_type
    result["_analysis_id"] = analysis_id

    return result


# ── Bounds validation (QUAL-001 §6.2) ─────────────────────────────────

# Metrics with bounded valid ranges: (key_names, lo, hi)
_BOUNDED_METRICS = [
    (("p_value",), 0.0, 1.0),
    (("correlation", "pearson_r", "spearman_rho", "r"), -1.0, 1.0),
    (("r_squared", "R2", "r2", "adj_r_squared"), 0.0, 1.0),
    (("eta_squared", "partial_eta_squared"), 0.0, 1.0),
    (("cramers_v",), 0.0, 1.0),
]
# Metrics that must be positive: (key_names,)
_POSITIVE_METRICS = ("bf10",)
# Metrics that must be finite: (key_names,)
_FINITE_METRICS = ("cp", "cpk", "pp", "ppk", "cohens_d", "cohens_f",
                    "odds_ratio", "relative_risk")


def _validate_statistics_bounds(result):
    """Validate statistical outputs against mathematically possible bounds.

    Clamps out-of-bound values and replaces NaN/Inf with None.
    Operates on both top-level keys and nested 'statistics' dict.
    """
    import math

    targets = [result]
    stats = result.get("statistics")
    if isinstance(stats, dict):
        targets.append(stats)

    for d in targets:
        # Bounded metrics
        for keys, lo, hi in _BOUNDED_METRICS:
            for key in keys:
                val = d.get(key)
                if val is None:
                    continue
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(fval):
                    logger.warning("QUAL-001: %s is %s, setting to None", key, val)
                    d[key] = None
                elif fval < lo or fval > hi:
                    clamped = max(lo, min(hi, fval))
                    logger.warning("QUAL-001: %s=%s out of [%s,%s], clamping to %s",
                                   key, fval, lo, hi, clamped)
                    d[key] = clamped

        # Positive metrics
        for key in _POSITIVE_METRICS:
            val = d.get(key)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fval) or fval <= 0:
                logger.warning("QUAL-001: %s=%s invalid (must be positive finite), setting to None",
                               key, val)
                d[key] = None

        # Finite metrics
        for key in _FINITE_METRICS:
            val = d.get(key)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fval):
                logger.warning("QUAL-001: %s=%s is not finite, setting to None", key, val)
                d[key] = None


def _narrative_from_summary(summary):
    """Build a minimal narrative dict from a summary string.

    Real narratives are richer (verdict, body, next_steps, chart_guidance).
    This fallback ensures every result has *something* renderable.
    """
    # Strip color tags for plain text
    clean = re.sub(r"<<COLOR:\w+>>|<</COLOR>>", "", summary)
    if not clean.strip():
        return None

    # Split on double newline or bullet patterns to extract sections
    lines = [l.strip() for l in clean.split("\n") if l.strip()]
    if not lines:
        return None

    verdict = lines[0]
    body = "\n".join(lines[1:]) if len(lines) > 1 else ""

    return {
        "verdict": verdict,
        "body": body,
        "next_steps": "",
        "chart_guidance": "",
    }


def _extract_p_value(result):
    """Extract p-value from result dict (various locations)."""
    # Direct p_value key
    if result.get("p_value") is not None:
        try:
            return float(result["p_value"])
        except (TypeError, ValueError):
            pass
    # Nested in statistics dict
    stats = result.get("statistics", {})
    if isinstance(stats, dict) and stats.get("p_value") is not None:
        try:
            return float(stats["p_value"])
        except (TypeError, ValueError):
            pass
    return None


def _classify_effect(result):
    """Classify effect magnitude from result statistics."""
    stats = result.get("statistics", {})
    if not isinstance(stats, dict):
        return None

    # Cohen's d
    d = stats.get("cohens_d") or stats.get("effect_size_d")
    if d is not None:
        d = abs(float(d))
        if d >= 0.8:
            return "large"
        elif d >= 0.5:
            return "medium"
        elif d >= 0.2:
            return "small"
        return "negligible"

    # Eta squared / partial eta squared
    eta = stats.get("eta_squared") or stats.get("partial_eta_squared") or stats.get("epsilon_squared")
    if eta is not None:
        eta = abs(float(eta))
        if eta >= 0.14:
            return "large"
        elif eta >= 0.06:
            return "medium"
        elif eta >= 0.01:
            return "small"
        return "negligible"

    # Effect r
    r = stats.get("effect_size_r") or stats.get("effect_r") or stats.get("spearman_rho")
    if r is not None:
        r = abs(float(r))
        if r >= 0.5:
            return "large"
        elif r >= 0.3:
            return "medium"
        elif r >= 0.1:
            return "small"
        return "negligible"

    # R-squared (regression)
    r2 = stats.get("r_squared") or stats.get("R2")
    if r2 is not None:
        r2 = float(r2)
        if r2 >= 0.26:
            return "large"
        elif r2 >= 0.13:
            return "medium"
        elif r2 >= 0.02:
            return "small"
        return "negligible"

    return None


def _auto_shadow(result, shadow_type):
    """Auto-generate a Bayesian shadow from result statistics.

    Uses the shadow_type from the registry to determine which
    _bayesian_shadow variant to call, extracting arguments from
    the result dict.
    """
    from .common import _bayesian_shadow
    stats = result.get("statistics", {})
    if not isinstance(stats, dict):
        stats = {}

    try:
        if shadow_type in ("ttest_1samp", "ttest_2samp", "ttest_paired"):
            # These need raw data — can only be computed in the analysis itself.
            # If they weren't computed there, we can't do it here.
            return None

        elif shadow_type == "anova":
            # Needs group data — same limitation
            return None

        elif shadow_type == "correlation":
            # Needs raw x, y arrays
            return None

        elif shadow_type == "chi2":
            chi2 = stats.get("chi2_statistic") or stats.get("chi_squared")
            dof = stats.get("df")
            n_obs = stats.get("n") or stats.get("n_obs") or stats.get("n_total")
            if chi2 is not None and dof is not None and n_obs is not None:
                return _bayesian_shadow("chi2", chi2_stat=chi2, dof=dof, n_obs=n_obs)

        elif shadow_type == "proportion":
            x = stats.get("successes") or stats.get("above")
            n = stats.get("n") or stats.get("n_total")
            p0 = stats.get("p0", 0.5)
            if x is not None and n is not None:
                return _bayesian_shadow("proportion", x=x, n=n, p0=p0)

        elif shadow_type == "regression":
            r2 = stats.get("r_squared") or stats.get("R2")
            n = stats.get("n") or stats.get("n_obs") or stats.get("n_total")
            k = stats.get("k_predictors") or stats.get("n_features") or stats.get("df_model")
            if r2 is not None and n is not None and k is not None:
                return _bayesian_shadow("regression", r_squared=r2, n_obs=n, k_predictors=k)

        elif shadow_type == "variance":
            f_stat = stats.get("F_statistic") or stats.get("f_value")
            df1 = stats.get("df1") or stats.get("df_between")
            df2 = stats.get("df2") or stats.get("df_within")
            if f_stat is not None and df1 is not None and df2 is not None:
                n_obs = stats.get("n") or (df1 + df2 + 2)
                return _bayesian_shadow("variance", f_stat=f_stat, df1=df1, df2=df2, n_obs=n_obs)

        elif shadow_type == "nonparametric":
            effect_r = stats.get("effect_size_r") or stats.get("effect_r") or stats.get("spearman_rho")
            n = stats.get("n") or stats.get("n_obs") or stats.get("n_total") or stats.get("n_pairs")
            if effect_r is not None and n is not None:
                return _bayesian_shadow("nonparametric", effect_r=effect_r, n_obs=n)

    except Exception:
        return None

    return None


def _normalize_what_if(result, entry):
    """Normalize existing what-if patterns to unified schema.

    Converts power_explorer and what_if_data legacy patterns to the
    canonical what_if schema: {type, parameters, endpoint, recompute_fields}.
    """
    # Legacy power_explorer pattern
    pe = result.get("power_explorer")
    if pe and isinstance(pe, dict):
        params = []
        test_type = pe.get("test_type", "")
        obs_n = pe.get("observed_n", 30)
        obs_d = pe.get("cohens_d", 0.5)
        alpha = pe.get("alpha", 0.05)

        params.append({"name": "effect_size", "label": "Effect Size (Cohen's d)",
                        "min": 0.1, "max": 2.0, "step": 0.05, "value": round(obs_d, 2)})
        params.append({"name": "sample_size", "label": "Sample Size",
                        "min": 5, "max": 500, "step": 5, "value": obs_n})
        params.append({"name": "alpha", "label": "Significance Level",
                        "min": 0.01, "max": 0.10, "step": 0.01, "value": alpha})

        return {
            "type": "slider",
            "parameters": params,
            "endpoint": "/api/dsw/run/",
            "recompute_fields": ["power", "sample_size"],
            "legacy_source": "power_explorer",
        }

    # Legacy what_if_data pattern (regression)
    wid = result.get("what_if_data")
    if wid and isinstance(wid, dict) and wid.get("type") == "regression":
        params = []
        for feat, ranges in (wid.get("feature_ranges") or {}).items():
            params.append({
                "name": feat,
                "label": feat,
                "min": round(ranges.get("min", 0), 4),
                "max": round(ranges.get("max", 1), 4),
                "step": round((ranges.get("max", 1) - ranges.get("min", 0)) / 20, 4),
                "value": round(ranges.get("mean", 0), 4),
            })

        return {
            "type": "slider",
            "parameters": params,
            "endpoint": "/api/dsw/run/",
            "recompute_fields": ["predicted_y"],
            "client_model": {
                "intercept": wid.get("intercept"),
                "coefficients": wid.get("coefficients"),
            },
            "legacy_source": "what_if_data",
        }

    # If entry says what-if tier but no legacy pattern, return a stub
    if entry and entry.get("what_if_tier", 0) > 0:
        return {
            "type": "slider" if entry["what_if_tier"] == 1 else "sensitivity",
            "parameters": [],
            "endpoint": "/api/dsw/run/",
            "recompute_fields": [],
        }

    return None

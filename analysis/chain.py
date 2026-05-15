"""Analysis chain — assembles the 10-key contract from chain steps.

Each step is a pure function: takes partial result, returns enriched result.
The chain runs them in order. Any step can be swapped or extended.

Contract output:
    summary, plots, statistics, narrative, education, diagnostics,
    assumptions, evidence_grade, bayesian_shadow, guide_observation
"""

import logging
import re

logger = logging.getLogger(__name__)


def assemble(raw, analysis_type, analysis_id):
    """Run the chain: raw handler output → 10-key contract.

    Args:
        raw: Dict from a handler — must have at least 'statistics' and 'charts'.
        analysis_type: e.g. "pbs", "stats", "spc"
        analysis_id: e.g. "pbs_belief", "ttest", "capability"

    Returns:
        Dict conforming to the 10-key contract.
    """
    result = {
        "summary": raw.get("summary", ""),
        "plots": _serialize_charts(raw.get("charts", [])),
        "statistics": raw.get("statistics", {}),
        "narrative": _build_narrative(raw, analysis_type, analysis_id),
        "education": _build_education(raw, analysis_type, analysis_id),
        "diagnostics": raw.get("diagnostics", []),
        "assumptions": raw.get("assumptions", {}),
        "evidence_grade": _build_evidence_grade(raw),
        "bayesian_shadow": raw.get("bayesian_shadow", None),
        "guide_observation": _build_guide_observation(raw),
        "what_if": raw.get("what_if", None),
        "_analysis_type": analysis_type,
        "_analysis_id": analysis_id,
    }

    # Layout hint for compose/trellis rendering
    if raw.get("_layout"):
        result["_layout"] = raw["_layout"]

    # Write numeric statistics to PCL
    _write_statistics_to_pcl(result, raw.get("_config", {}))

    return result


def _serialize_charts(charts):
    """Convert ChartSpec objects (or dicts) to JSON-serializable dicts."""
    out = []
    for chart in charts:
        if hasattr(chart, "to_dict"):
            out.append(chart.to_dict())
        elif isinstance(chart, dict):
            out.append(chart)
        else:
            logger.warning("Skipping non-serializable chart: %s", type(chart))
    return out


def _build_narrative(raw, analysis_type=None, analysis_id=None):
    """Build narrative dict — prefer forgenarr, fall back to handler or generic."""
    # If handler already provided a narrative dict, use it
    narr = raw.get("narrative")
    if isinstance(narr, dict) and narr.get("verdict"):
        return narr

    # Try forgenarr
    try:
        from forgenarr import narrate

        result = narrate(
            analysis_type=analysis_type or "",
            analysis_id=analysis_id or "",
            statistics=raw.get("statistics", {}),
            config=raw.get("_config", {}),
            summary=raw.get("summary", ""),
        )
        if result.get("verdict"):
            return result
    except ImportError:
        pass  # forgenarr not installed
    except Exception:
        logger.debug("forgenarr failed, falling back", exc_info=True)

    # Fall back to handler-provided string or summary
    if isinstance(narr, str):
        return {
            "verdict": narr.split(".")[0] + "." if "." in narr else narr,
            "body": narr,
            "next_steps": "",
            "chart_guidance": "",
        }
    summary = raw.get("summary", "")
    if summary:
        clean = re.sub(r"<<COLOR:\w+>>|<<COLOR>>", "", summary)
        return {
            "verdict": clean.split(".")[0] + "." if "." in clean else clean,
            "body": clean,
            "next_steps": "",
            "chart_guidance": "",
        }
    return {"verdict": "", "body": "", "next_steps": "", "chart_guidance": ""}


def _build_education(raw, analysis_type, analysis_id):
    """Look up education content for this analysis."""
    if raw.get("education"):
        return raw["education"]
    try:
        from agents_api.analysis.education import get_education

        return get_education(analysis_type, analysis_id)
    except (ImportError, Exception):
        return None


def _build_evidence_grade(raw):
    """Derive evidence grade from statistics."""
    if raw.get("evidence_grade"):
        return raw["evidence_grade"]

    stats = raw.get("statistics", {})
    p = stats.get("p_value")
    bf10 = None
    shadow = raw.get("bayesian_shadow")
    if shadow:
        bf10 = shadow.get("bf10")

    if p is None and bf10 is None:
        return None

    # Simple grading — can be replaced by forgenarr later
    if bf10 is not None and bf10 > 10:
        return "strong"
    if p is not None and p < 0.01:
        return "strong"
    if p is not None and p < 0.05:
        return "moderate"
    if bf10 is not None and bf10 > 3:
        return "moderate"
    return "weak"


def _build_guide_observation(raw):
    """Build short guide observation from summary."""
    if raw.get("guide_observation"):
        return raw["guide_observation"]
    summary = raw.get("summary", "")
    clean = re.sub(r"<<COLOR:\w+>>|<<COLOR>>", "", summary)
    return clean[:300] if clean else ""


# ---------------------------------------------------------------------------
# PCL write-back — workbench results become addressable measures
# ---------------------------------------------------------------------------

# Statistics keys worth writing to PCL (numeric, meaningful to downstream tools)
_PCL_WORTHY_KEYS = {
    "cpk",
    "ppk",
    "cp",
    "pp",
    "sigma_level",
    "dpmo",
    "yield_pct",
    "p_value",
    "test_statistic",
    "effect_size",
    "r_squared",
    "mean",
    "std",
    "median",
    "n",
    "grr_pct",
    "ndc",
    "repeatability_pct",
    "reproducibility_pct",
    "center_line",
    "ucl",
    "lcl",
    "bf10",
    "posterior_mean",
    "credible_lower",
    "credible_upper",
    "mttf",
    "failure_rate",
    "availability",
    "oee",
    "throughput",
}


def _write_statistics_to_pcl(result, config):
    """Write numeric statistics from a workbench analysis to PCL.

    Slug convention: wb/{analysis_type}/{analysis_id}/{stat_key}
    If config has a 'measurement' or 'column' key, it's included in the slug
    for disambiguation: wb/{analysis_type}/{column}/{stat_key}

    Non-fatal — failures are logged, never raised.
    """
    stats = result.get("statistics")
    if not stats or not isinstance(stats, dict):
        return

    analysis_type = result.get("_analysis_type", "")
    analysis_id = result.get("_analysis_id", "")
    if not analysis_type:
        return

    # Build slug prefix from analysis context
    column = config.get("measurement") or config.get("column") or config.get("var") or ""
    if column:
        prefix = f"wb/{analysis_type}/{column}"
    else:
        prefix = f"wb/{analysis_type}/{analysis_id}"

    try:
        from pcl.service import ensure_and_write
    except ImportError:
        return

    # Determine unit hints for known keys
    _UNITS = {
        "cpk": "",
        "ppk": "",
        "cp": "",
        "pp": "",
        "sigma_level": "σ",
        "dpmo": "ppm",
        "yield_pct": "%",
        "p_value": "",
        "r_squared": "",
        "effect_size": "",
        "mean": "",
        "std": "",
        "median": "",
        "n": "count",
        "grr_pct": "%",
        "ndc": "",
        "oee": "%",
    }

    written = 0
    for key, value in stats.items():
        if key not in _PCL_WORTHY_KEYS:
            continue
        if not isinstance(value, (int, float)):
            continue
        if value != value:  # NaN check
            continue

        slug = f"{prefix}/{key}"
        try:
            ensure_and_write(
                slug=slug,
                value=float(value),
                source_type="workbench",
                actor="system",
                unit=_UNITS.get(key, ""),
                provenance="calculated",
                notes=f"{analysis_type}/{analysis_id}",
            )
            written += 1
        except Exception:
            logger.debug("PCL write failed for %s", slug, exc_info=True)

    if written:
        logger.info("PCL: wrote %d statistics from %s/%s", written, analysis_type, analysis_id)

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

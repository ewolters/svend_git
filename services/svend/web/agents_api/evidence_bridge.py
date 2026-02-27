"""Evidence bridge — shared helper for problem-solving tool integration.

All problem-solving tools (NCR, A3, RCA, CAPA/8D) use this to push
findings into the core evidence system. Follows the proven pattern
from fmea_views.py:361-440 (link_to_hypothesis).

Key design decisions:
- Idempotent: same (tool, id, field) never duplicates evidence.
- Neutral confidence: all tool-generated evidence starts at 0.5.
  Synara's challenge process is what elevates confidence, not the tool.
- Feature-flagged: controlled by settings.EVIDENCE_INTEGRATION_ENABLED.
"""

import logging

from django.conf import settings

from core.models import Evidence, EvidenceLink, Hypothesis

logger = logging.getLogger(__name__)


def create_tool_evidence(
    project,
    user,
    summary,
    source_tool,
    source_id,
    source_field,
    details="",
    source_type="observation",
    confidence=0.5,
    hypothesis_id=None,
    likelihood_ratio=None,
):
    """Create Evidence from a problem-solving tool finding.

    Args:
        project: core.Project instance (the Study hub).
        user: request.user who triggered the update.
        summary: Human-readable summary, e.g. "NCR root cause: ..."
        source_tool: Tool identifier — "ncr" | "a3" | "rca" | "report".
        source_id: UUID of the tool record (NCR id, A3 id, etc.).
        source_field: Field that changed — "root_cause", "chain_step_3", etc.
        details: Extended description (optional).
        source_type: Evidence.SourceType value — "observation", "analysis", "experiment".
        confidence: Always 0.5 for tool-generated evidence.
        hypothesis_id: Optional UUID — auto-link to hypothesis if provided.
        likelihood_ratio: Required if hypothesis_id is provided.

    Returns:
        (Evidence, EvidenceLink | None) tuple, or (None, None) if disabled/skipped.
    """
    if not getattr(settings, "EVIDENCE_INTEGRATION_ENABLED", False):
        return None, None

    if not project:
        logger.warning(
            "create_tool_evidence called without project: %s:%s:%s",
            source_tool, source_id, source_field,
        )
        return None, None

    # Idempotency key
    source_description = f"{source_tool}:{source_id}:{source_field}"

    # Check for existing evidence with same key
    existing = Evidence.objects.filter(
        project=project,
        source_description=source_description,
    ).first()

    if existing:
        # Update the existing record rather than duplicating
        updated = False
        if existing.summary != summary:
            existing.summary = summary
            updated = True
        if details and existing.details != details:
            existing.details = details
            updated = True
        if updated:
            existing.save(update_fields=["summary", "details", "updated_at"])
            logger.info("Updated existing evidence %s for %s", existing.id, source_description)
        return existing, None

    evidence = Evidence.objects.create(
        project=project,
        summary=summary,
        details=details,
        source_type=source_type,
        source_description=source_description,
        result_type="qualitative",
        confidence=confidence,
        created_by=user,
    )
    logger.info("Created evidence %s from %s", evidence.id, source_description)

    # Optional hypothesis linking
    link = None
    if hypothesis_id and likelihood_ratio is not None:
        try:
            hypothesis = Hypothesis.objects.get(
                id=hypothesis_id,
                project=project,
            )
            link = EvidenceLink.objects.create(
                hypothesis=hypothesis,
                evidence=evidence,
                likelihood_ratio=likelihood_ratio,
                reasoning=f"Auto-linked from {source_tool}: {summary[:200]}",
                is_manual=False,
            )
            hypothesis.apply_evidence(link)
            logger.info(
                "Linked evidence %s to hypothesis %s (LR=%.2f)",
                evidence.id, hypothesis_id, likelihood_ratio,
            )
        except Hypothesis.DoesNotExist:
            logger.warning(
                "Hypothesis %s not found for project %s — evidence created without link",
                hypothesis_id, project.id,
            )

    return evidence, link

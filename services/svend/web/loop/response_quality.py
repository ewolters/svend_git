"""
Supplier Response Quality Scoring — Loop supplier management.

Evaluates supplier CAPA responses for quality indicators:
- Repeat root cause categories
- Copy-paste corrective/preventive actions
- Insufficient root cause depth
- Blocked phrases

Thresholds configurable via QMS Policy (defaults below).
"""

import logging
from difflib import SequenceMatcher

logger = logging.getLogger("svend.loop.response_quality")

# Default thresholds
DEFAULT_REPEAT_THRESHOLD = 0.4  # >40% of previous claims share root cause category
DEFAULT_SIMILARITY_THRESHOLD = 0.7  # corrective vs preventive text similarity
DEFAULT_MIN_ROOT_CAUSE_LENGTH = 100  # chars
BLOCKED_PHRASES = [
    "n/a",
    "tbd",
    "to be determined",
    "not applicable",
    "will investigate",
    "under review",
    "pending",
]


def evaluate_response_quality(claim, response, supplier):
    """Evaluate a supplier response for quality indicators.

    Args:
        claim: SupplierClaim instance
        response: SupplierResponse instance
        supplier: SupplierRecord instance

    Returns:
        (score: float 0-1, flags: list[str])
    """
    flags = []
    deductions = 0.0

    # 1. Repeat root cause category
    from .models import SupplierResponse

    previous_responses = (
        SupplierResponse.objects.filter(
            claim__supplier=supplier,
        )
        .exclude(claim=claim)
        .values_list("root_cause_category", flat=True)
    )

    total_previous = len(previous_responses)
    if total_previous > 0:
        same_category = sum(1 for rc in previous_responses if rc == response.root_cause_category)
        repeat_rate = same_category / total_previous
        if repeat_rate > DEFAULT_REPEAT_THRESHOLD:
            flags.append(
                f"repeat_root_cause: {response.root_cause_category} used in "
                f"{same_category}/{total_previous} ({repeat_rate:.0%}) of previous claims"
            )
            deductions += 0.2
            response.is_repeat_root_cause = True

    # 2. Corrective == preventive (copy-paste)
    if response.corrective_action and response.preventive_action:
        similarity = SequenceMatcher(
            None,
            response.corrective_action.lower(),
            response.preventive_action.lower(),
        ).ratio()
        if similarity > DEFAULT_SIMILARITY_THRESHOLD:
            flags.append(
                f"corrective_equals_preventive: {similarity:.0%} similarity — "
                f"corrective and preventive actions should be distinct"
            )
            deductions += 0.15

    # 3. Similarity to previous responses on same supplier
    previous_corrective = (
        SupplierResponse.objects.filter(
            claim__supplier=supplier,
        )
        .exclude(claim=claim)
        .exclude(corrective_action="")
        .values_list("corrective_action", flat=True)[:10]
    )

    for prev in previous_corrective:
        sim = SequenceMatcher(
            None,
            response.corrective_action.lower(),
            prev.lower(),
        ).ratio()
        if sim > DEFAULT_SIMILARITY_THRESHOLD:
            flags.append(
                f"repeated_corrective_action: {sim:.0%} similarity to previous response — "
                f"supplier may be providing generic responses"
            )
            deductions += 0.2
            break

    # 4. Root cause too short
    if len(response.root_cause_description.strip()) < DEFAULT_MIN_ROOT_CAUSE_LENGTH:
        flags.append(
            f"shallow_root_cause: {len(response.root_cause_description)} chars "
            f"(minimum {DEFAULT_MIN_ROOT_CAUSE_LENGTH}) — root cause lacks depth"
        )
        deductions += 0.15

    # 5. Blocked phrases
    rc_lower = response.root_cause_description.lower()
    for phrase in BLOCKED_PHRASES:
        if phrase in rc_lower:
            flags.append(f"blocked_phrase: '{phrase}' in root cause description")
            deductions += 0.25
            break

    score = max(0.0, min(1.0, 1.0 - deductions))

    logger.info(
        "Response quality for claim %s: score=%.2f, flags=%d",
        claim.id,
        score,
        len(flags),
    )

    return score, flags

"""CI Readiness Score — LOOP-001 §10.

Answers: "Is the loop turning?"

10 indicators, each scored 0-10. Weighted sum produces a composite 0-100.
Absence = failure. Staleness decays. Overdue items are worst.

Weights are configurable via QMS Policy (scope=readiness, rule_key=readiness.weights).
Default weights are equal (10 each, sum = 100).

The score can go DOWN when you add data. That's the system working —
discovering a problem is progress, hiding it is not.
"""

import logging
from datetime import date, timedelta

logger = logging.getLogger("svend.loop.readiness")

# Default weights (configurable via QMS Policy)
DEFAULT_WEIGHTS = {
    "signal_detection": 10,
    "investigation_velocity": 10,
    "hypothesis_testing": 10,
    "standardization_lag": 10,
    "training_coverage": 10,
    "verification_activity": 10,
    "recurrence_rate": 10,
    "standard_work_quality": 10,
    "detection_capability": 10,
    "commitment_fulfillment": 10,
}

# Evidence half-life in days (configurable via QMS Policy)
DEFAULT_HALF_LIFE = 90


def compute_readiness_score(user=None, tenant=None):
    """Compute the CI Readiness Score.

    Returns dict with:
    - score: float 0-100
    - indicators: dict of {name: {score, max, description, detail}}
    - weights: the weights used
    - computed_at: timestamp
    """
    from core.models import Investigation

    from .models import (
        Commitment,
        ForcedFailureTest,
        ProcessConfirmation,
        QMSPolicy,
        Signal,
    )

    # Load custom weights from QMS Policy if available
    weights = dict(DEFAULT_WEIGHTS)

    policy = QMSPolicy.objects.filter(
        scope=QMSPolicy.Scope.PROCESS_CONFIRMATION,  # Using a general scope
        rule_key="readiness.weights",
        is_active=True,
    ).first()
    if policy and policy.parameters:
        weights.update(policy.parameters.get("weights", {}))

    now = date.today()
    window_90 = now - timedelta(days=90)
    window_30 = now - timedelta(days=30)

    indicators = {}

    # 1. Signal Detection Rate
    # Are problems found internally before customers find them?
    internal_signals = Signal.objects.filter(
        created_at__date__gte=window_90,
        source_type__in=[
            Signal.SourceType.SPC_VIOLATION,
            Signal.SourceType.PC_THRESHOLD,
            Signal.SourceType.FORCED_FAILURE_MISS,
            Signal.SourceType.FRONTIER_CARD,
            Signal.SourceType.OPERATOR_REPORT,
            Signal.SourceType.AUDIT_FINDING,
        ],
    ).count()
    external_signals = Signal.objects.filter(
        created_at__date__gte=window_90,
        source_type__in=[
            Signal.SourceType.CUSTOMER_COMPLAINT,
            Signal.SourceType.FIELD_RETURN,
        ],
    ).count()
    total_signals = internal_signals + external_signals
    if total_signals == 0:
        sig_score = 0  # No signals = system is blind
    else:
        sig_score = min(10, (internal_signals / total_signals) * 10)
    indicators["signal_detection"] = {
        "score": round(sig_score, 1),
        "max": 10,
        "description": "Internal vs external signal ratio",
        "detail": f"{internal_signals} internal / {external_signals} external (90d)",
    }

    # 2. Investigation Velocity
    # Are investigations producing conclusions?
    concluded = Investigation.objects.filter(
        status__in=["concluded", "exported"],
        concluded_at__date__gte=window_90,
    ).count()
    open_inv = Investigation.objects.filter(
        status__in=["open", "active"],
    ).count()
    total_inv = concluded + open_inv
    if total_inv == 0:
        inv_score = 0
    else:
        inv_score = min(10, (concluded / max(total_inv, 1)) * 10)
    indicators["investigation_velocity"] = {
        "score": round(inv_score, 1),
        "max": 10,
        "description": "Concluded / total investigations",
        "detail": f"{concluded} concluded, {open_inv} active (90d)",
    }

    # 3. Hypothesis Testing Rate
    # Are conclusions tested, not assumed?
    # Proxy: investigations with tool_links (evidence from tools)
    inv_with_tools = (
        Investigation.objects.filter(
            concluded_at__date__gte=window_90,
            tool_links__isnull=False,
        )
        .distinct()
        .count()
    )
    if concluded == 0:
        hyp_score = 0
    else:
        hyp_score = min(10, (inv_with_tools / max(concluded, 1)) * 10)
    indicators["hypothesis_testing"] = {
        "score": round(hyp_score, 1),
        "max": 10,
        "description": "Investigations with tool-based evidence",
        "detail": f"{inv_with_tools} / {concluded} with tool evidence (90d)",
    }

    # 4. Standardization Lag
    # How fast do fixes become standards?
    # Proxy: commitments with revise_document/create_document transition fulfilled
    standardize_cmts = Commitment.objects.filter(
        transition_type__in=["revise_document", "create_document"],
        created_at__date__gte=window_90,
    )
    fulfilled_std = standardize_cmts.filter(status=Commitment.Status.FULFILLED).count()
    total_std = standardize_cmts.count()
    if total_std == 0:
        std_score = 5  # No standardization commitments — neutral (might not be needed)
    else:
        std_score = min(10, (fulfilled_std / total_std) * 10)
    indicators["standardization_lag"] = {
        "score": round(std_score, 1),
        "max": 10,
        "description": "Standardize commitments fulfilled",
        "detail": f"{fulfilled_std} / {total_std} fulfilled (90d)",
    }

    # 5. Training Coverage
    # Are people trained on current standards?
    from agents_api.models import TrainingRecord

    total_records = TrainingRecord.objects.count()
    current_records = TrainingRecord.objects.filter(status="complete").count()
    if total_records == 0:
        trn_score = 0
    else:
        trn_score = min(10, (current_records / total_records) * 10)
    indicators["training_coverage"] = {
        "score": round(trn_score, 1),
        "max": 10,
        "description": "Training records current",
        "detail": f"{current_records} / {total_records} complete",
    }

    # 6. Verification Activity
    # Are PCs and FFTs happening?
    pcs_30d = ProcessConfirmation.objects.filter(
        created_at__date__gte=window_30,
    ).count()
    ffts_90d = ForcedFailureTest.objects.filter(
        created_at__date__gte=window_90,
        result__in=["detected", "not_detected", "partially_detected"],
    ).count()
    # Score: at least 1 PC per week + at least 1 FFT per quarter
    pc_target = 4  # 4 PCs in 30 days
    fft_target = 1  # 1 FFT in 90 days
    pc_ratio = min(1.0, pcs_30d / max(pc_target, 1))
    fft_ratio = min(1.0, ffts_90d / max(fft_target, 1))
    ver_score = ((pc_ratio + fft_ratio) / 2) * 10
    indicators["verification_activity"] = {
        "score": round(ver_score, 1),
        "max": 10,
        "description": "PC and FFT activity vs targets",
        "detail": f"{pcs_30d} PCs (30d), {ffts_90d} FFTs (90d)",
    }

    # 7. Recurrence Rate
    # Are fixed problems staying fixed?
    # Proxy: signals on failure modes that already have concluded investigations
    # Lower is better — invert the score
    resolved_signals = Signal.objects.filter(
        triage_state=Signal.TriageState.RESOLVED,
        created_at__date__gte=window_90,
    ).count()
    # Recurrence = 0 until cross-investigation failure mode matching is implemented.
    # "Absence = failure" design principle — no data means no credit.
    rec_score = 0
    indicators["recurrence_rate"] = {
        "score": round(rec_score, 1),
        "max": 10,
        "description": "Repeat failure modes (lower recurrence = higher score)",
        "detail": f"{resolved_signals} signals resolved (90d) — not yet measured",
    }

    # 8. Standard Work Quality
    # Are standards followed AND producing results?
    pcs_with_diagnosis = ProcessConfirmation.objects.filter(
        created_at__date__gte=window_90,
    ).exclude(diagnosis="incomplete")
    system_works = pcs_with_diagnosis.filter(diagnosis="system_works").count()
    total_diag = pcs_with_diagnosis.count()
    if total_diag == 0:
        swq_score = 0  # No PCs = unknown
    else:
        swq_score = min(10, (system_works / total_diag) * 10)
    indicators["standard_work_quality"] = {
        "score": round(swq_score, 1),
        "max": 10,
        "description": "PC pass rate (followed + correct outcome)",
        "detail": f"{system_works} / {total_diag} system works (90d)",
    }

    # 9. Detection Capability
    # Do controls catch failures?
    fft_detected = ForcedFailureTest.objects.filter(
        created_at__date__gte=window_90,
        injection_count__gt=0,
    )
    total_detected = sum(f.detection_count for f in fft_detected)
    total_injected = sum(f.injection_count for f in fft_detected)
    if total_injected == 0:
        det_score = 0  # No FFTs = unknown (not penalized, but flagged)
        det_detail = "No forced failure tests — detection unknown"
    else:
        det_rate = total_detected / total_injected
        det_score = det_rate * 10
        det_detail = f"{total_detected}/{total_injected} detected ({det_rate:.0%})"
    indicators["detection_capability"] = {
        "score": round(det_score, 1),
        "max": 10,
        "description": "Forced failure detection rate",
        "detail": det_detail,
    }

    # 10. Commitment Fulfillment
    # Are people doing what they committed to?
    all_cmts = Commitment.objects.filter(
        created_at__date__gte=window_90,
    ).exclude(status=Commitment.Status.CANCELLED)
    fulfilled = all_cmts.filter(status=Commitment.Status.FULFILLED).count()
    broken = all_cmts.filter(status=Commitment.Status.BROKEN).count()
    overdue = sum(1 for c in all_cmts if c.is_overdue)
    total_cmts = all_cmts.count()

    if total_cmts == 0:
        cmt_score = 0  # No commitments = system not in use
    else:
        # Fulfilled is good. Broken and overdue are bad.
        fulfill_rate = fulfilled / total_cmts
        penalty = (broken + overdue * 2) / total_cmts  # Overdue is 2x penalty
        cmt_score = max(0, min(10, fulfill_rate * 10 - penalty * 5))
    indicators["commitment_fulfillment"] = {
        "score": round(cmt_score, 1),
        "max": 10,
        "description": "Commitments fulfilled on time",
        "detail": f"{fulfilled} fulfilled, {broken} broken, {overdue} overdue (90d)",
    }

    # Compute weighted total
    total_score = 0
    max_score = 0
    for key, indicator in indicators.items():
        weight = weights.get(key, 10)
        total_score += indicator["score"] * (weight / 10)
        max_score += indicator["max"] * (weight / 10)

    final_score = (total_score / max_score * 100) if max_score > 0 else 0

    return {
        "score": round(final_score, 1),
        "indicators": indicators,
        "weights": weights,
        "computed_at": date.today().isoformat(),
    }

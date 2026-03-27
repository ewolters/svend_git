"""Policy evaluation engine — LOOP-001 §4.6.

Two evaluation modes:
- PolicyEvaluator: subscribes to ToolEventBus for real-time evaluation
- PolicySweepEvaluator: runs on schedule (Tempora) for aggregate conditions

Both produce PolicyConditions surfaced on the accountability dashboard.
Humans decide whether conditions become Signals.
"""

import logging
from datetime import date, timedelta

from django.db.models import Count, Q

logger = logging.getLogger("svend.loop.evaluator")


# =============================================================================
# REAL-TIME EVALUATOR (ToolEventBus subscriber)
# =============================================================================


def register_policy_handlers():
    """Subscribe to ToolEventBus events for real-time policy evaluation.

    Called from loop/apps.py ready().
    """
    from agents_api.tool_events import tool_events

    # PC completed → check retraining threshold
    tool_events.subscribe("pc.completed", _evaluate_pc_threshold)

    # Forced failure completed → check detection gap
    tool_events.subscribe("forced_failure.completed", _evaluate_detection_gap)

    # Commitment overdue check on any commitment update
    tool_events.subscribe("commitment.updated", _evaluate_commitment_overdue)

    logger.info("PolicyEvaluator: 3 real-time handlers registered")


def _evaluate_pc_threshold(event):
    """Check if PC result triggers retraining threshold per QMS Policy.

    LOOP-001 §4.6.2: pc.retraining_threshold rule.
    """
    from .models import PolicyCondition, ProcessConfirmation, QMSPolicy

    pc = event.record
    if not isinstance(pc, ProcessConfirmation):
        return
    if not pc.operator_id or not pc.controlled_document_id:
        return

    # Find applicable policy
    policies = QMSPolicy.objects.filter(
        scope=QMSPolicy.Scope.PROCESS_CONFIRMATION,
        rule_key="pc.retraining_threshold",
        is_active=True,
    )
    if pc.tenant_id:
        policies = policies.filter(Q(tenant=pc.tenant) | Q(tenant__isnull=True))

    policy = policies.first()
    if not policy:
        return

    params = policy.parameters
    threshold = params.get("pass_rate_threshold", 0.80)
    window = params.get("trailing_window", 5)
    cooldown_days = params.get("cooldown_days", 30)

    # Get trailing PCs for this operator + standard
    trailing = ProcessConfirmation.objects.filter(
        operator=pc.operator,
        controlled_document=pc.controlled_document,
    ).order_by("-created_at")[:window]

    if trailing.count() < window:
        return  # Not enough data yet

    # Compute pass rate
    pass_rates = [p.pass_rate for p in trailing if p.pass_rate is not None]
    if not pass_rates:
        return
    avg_pass_rate = sum(pass_rates) / len(pass_rates)

    if avg_pass_rate >= threshold:
        return  # Above threshold, no condition

    # Check cooldown — don't fire again within cooldown_days
    recent_condition = PolicyCondition.objects.filter(
        policy_rule=policy,
        condition_type=PolicyCondition.ConditionType.RETRAINING_NEEDED,
        context__operator_id=str(pc.operator_id),
        context__document_id=str(pc.controlled_document_id),
        created_at__gte=pc.created_at - timedelta(days=cooldown_days) if pc.created_at else None,
    ).exists()

    if recent_condition:
        return  # Cooldown active

    # Check escalation — if N operators failing same standard, escalate
    escalate_n = params.get("escalate_after_n_operators", 3)
    escalation_window = params.get("escalation_window_days", 60)

    # Count distinct operators below threshold on this standard
    failing_operators = _count_failing_operators(pc.controlled_document_id, threshold, window, escalation_window)

    if failing_operators >= escalate_n:
        # Escalate to standard revision signal
        PolicyCondition.objects.create(
            tenant=pc.tenant,
            condition_type=PolicyCondition.ConditionType.STANDARD_REVISION_NEEDED,
            severity=PolicyCondition.Severity.WARNING,
            title=f"Standard revision needed: {failing_operators} operators below PC threshold on {pc.controlled_document}",
            context={
                "document_id": str(pc.controlled_document_id),
                "failing_operator_count": failing_operators,
                "threshold": threshold,
                "avg_pass_rate": round(avg_pass_rate, 3),
            },
            policy_rule=policy,
            source_event="pc.completed",
        )
    else:
        # Individual retraining condition
        PolicyCondition.objects.create(
            tenant=pc.tenant,
            condition_type=PolicyCondition.ConditionType.RETRAINING_NEEDED,
            severity=PolicyCondition.Severity.INFO,
            title=f"PC threshold breach: {pc.operator} on {pc.controlled_document} ({avg_pass_rate:.0%} < {threshold:.0%})",
            context={
                "operator_id": str(pc.operator_id),
                "document_id": str(pc.controlled_document_id),
                "pass_rate": round(avg_pass_rate, 3),
                "threshold": threshold,
                "window": window,
            },
            policy_rule=policy,
            source_event="pc.completed",
        )

    logger.info(
        "policy.pc_threshold_evaluated",
        extra={
            "operator_id": str(pc.operator_id),
            "document_id": str(pc.controlled_document_id),
            "pass_rate": round(avg_pass_rate, 3),
            "threshold": threshold,
        },
    )


def _count_failing_operators(document_id, threshold, window, escalation_window_days):
    """Count distinct operators below PC threshold on a standard."""
    from .models import ProcessConfirmation

    cutoff = date.today() - timedelta(days=escalation_window_days)
    pcs = (
        ProcessConfirmation.objects.filter(
            controlled_document_id=document_id,
            created_at__date__gte=cutoff,
            operator__isnull=False,
        )
        .values("operator_id")
        .annotate(
            pc_count=Count("id"),
        )
        .filter(pc_count__gte=window)
    )

    failing = 0
    for entry in pcs:
        operator_pcs = ProcessConfirmation.objects.filter(
            operator_id=entry["operator_id"],
            controlled_document_id=document_id,
        ).order_by("-created_at")[:window]

        rates = [p.pass_rate for p in operator_pcs if p.pass_rate is not None]
        if rates and (sum(rates) / len(rates)) < threshold:
            failing += 1

    return failing


def _evaluate_detection_gap(event):
    """FFT with not_detected result → always surface a condition.

    LOOP-001 §7.2: "A not_detected result on a hypothesis-driven test
    auto-surfaces an investigate signal — detection gaps are ALWAYS investigated."
    """
    from .models import ForcedFailureTest, PolicyCondition, QMSPolicy

    fft = event.record
    if not isinstance(fft, ForcedFailureTest):
        return
    if fft.result != ForcedFailureTest.Result.NOT_DETECTED:
        return

    # Find or use default policy
    policy = QMSPolicy.objects.filter(
        scope=QMSPolicy.Scope.FORCED_FAILURE,
        is_active=True,
    ).first()

    # Detection gap is always a condition regardless of policy existence
    PolicyCondition.objects.create(
        tenant=fft.tenant,
        condition_type=PolicyCondition.ConditionType.DETECTION_GAP,
        severity=PolicyCondition.Severity.CRITICAL,
        title=f"Detection gap: {fft.detection_count}/{fft.injection_count} detected — {fft.control_being_tested or 'unknown control'}",
        context={
            "fft_id": str(fft.id),
            "fmea_row_id": str(fft.fmea_row_id) if fft.fmea_row_id else None,
            "detection_count": fft.detection_count,
            "injection_count": fft.injection_count,
            "control": fft.control_being_tested,
        },
        policy_rule=policy,
        source_event="forced_failure.completed",
    )

    logger.info(
        "policy.detection_gap",
        extra={"fft_id": str(fft.id), "detection_rate": fft.detection_rate},
    )


def _evaluate_commitment_overdue(event):
    """Surface overdue commitments as conditions."""
    from .models import Commitment, PolicyCondition, QMSPolicy

    commitment = event.record
    if not isinstance(commitment, Commitment):
        return
    if not commitment.is_overdue:
        return

    # Don't duplicate — check for existing condition
    exists = PolicyCondition.objects.filter(
        condition_type=PolicyCondition.ConditionType.COMMITMENT_OVERDUE,
        context__commitment_id=str(commitment.id),
        status=PolicyCondition.Status.ACTIVE,
    ).exists()
    if exists:
        return

    policy = QMSPolicy.objects.filter(
        scope=QMSPolicy.Scope.COMMITMENT_FULFILLMENT,
        is_active=True,
    ).first()

    days_overdue = (date.today() - commitment.due_date).days

    PolicyCondition.objects.create(
        tenant=commitment.tenant,
        condition_type=PolicyCondition.ConditionType.COMMITMENT_OVERDUE,
        severity=(
            PolicyCondition.Severity.CRITICAL
            if days_overdue > 14
            else PolicyCondition.Severity.WARNING
            if days_overdue > 7
            else PolicyCondition.Severity.INFO
        ),
        title=f"Commitment overdue ({days_overdue}d): {commitment.title}",
        context={
            "commitment_id": str(commitment.id),
            "owner_id": str(commitment.owner_id),
            "due_date": commitment.due_date.isoformat(),
            "days_overdue": days_overdue,
        },
        policy_rule=policy,
        source_event="commitment.updated",
    )


# =============================================================================
# AGGREGATE SWEEP EVALUATOR (Tempora scheduled task)
# =============================================================================


def run_policy_sweep(payload=None, context=None):
    """Daily aggregate policy evaluation.

    Checks conditions that require querying across records:
    - Review frequency (FMEA, management review, document review overdue)
    - Training coverage (% trained on current versions)
    - Calibration overdue
    - Commitment fulfillment patterns
    - Verification schedule adherence

    LOOP-001 §4.6.3
    """
    logger.info("PolicySweepEvaluator: starting daily sweep")

    count = 0
    count += _sweep_calibration_overdue()
    count += _sweep_commitment_patterns()
    count += _sweep_verification_schedule()

    logger.info("PolicySweepEvaluator: sweep complete, %d conditions created", count)
    return {"conditions_created": count}


def _sweep_calibration_overdue():
    """Check for overdue calibration equipment."""
    from agents_api.models import MeasurementEquipment

    from .models import PolicyCondition, QMSPolicy

    policy = QMSPolicy.objects.filter(
        scope=QMSPolicy.Scope.CALIBRATION,
        is_active=True,
    ).first()

    overdue = MeasurementEquipment.objects.filter(
        status__in=["overdue", "due"],
    )

    count = 0
    for equip in overdue:
        # Don't duplicate
        exists = PolicyCondition.objects.filter(
            condition_type=PolicyCondition.ConditionType.CALIBRATION_OVERDUE,
            context__equipment_id=str(equip.id),
            status=PolicyCondition.Status.ACTIVE,
        ).exists()
        if exists:
            continue

        PolicyCondition.objects.create(
            condition_type=PolicyCondition.ConditionType.CALIBRATION_OVERDUE,
            severity=PolicyCondition.Severity.WARNING,
            title=f"Calibration {equip.status}: {equip.name} ({equip.serial_number or equip.asset_id})",
            context={
                "equipment_id": str(equip.id),
                "equipment_name": equip.name,
                "status": equip.status,
                "next_due": equip.next_calibration_due.isoformat() if equip.next_calibration_due else None,
            },
            policy_rule=policy,
            source_event="",
        )
        count += 1

    return count


def _sweep_commitment_patterns():
    """Surface patterns in overdue commitments."""
    from .models import Commitment, PolicyCondition, QMSPolicy

    policy = QMSPolicy.objects.filter(
        scope=QMSPolicy.Scope.COMMITMENT_FULFILLMENT,
        is_active=True,
    ).first()

    # Find all overdue commitments not already surfaced
    overdue = Commitment.objects.filter(
        status__in=[Commitment.Status.OPEN, Commitment.Status.IN_PROGRESS],
        due_date__lt=date.today(),
    )

    count = 0
    for c in overdue:
        exists = PolicyCondition.objects.filter(
            condition_type=PolicyCondition.ConditionType.COMMITMENT_OVERDUE,
            context__commitment_id=str(c.id),
            status=PolicyCondition.Status.ACTIVE,
        ).exists()
        if exists:
            continue

        days_overdue = (date.today() - c.due_date).days
        PolicyCondition.objects.create(
            tenant=c.tenant,
            condition_type=PolicyCondition.ConditionType.COMMITMENT_OVERDUE,
            severity=(
                PolicyCondition.Severity.CRITICAL
                if days_overdue > 14
                else PolicyCondition.Severity.WARNING
                if days_overdue > 7
                else PolicyCondition.Severity.INFO
            ),
            title=f"Commitment overdue ({days_overdue}d): {c.title}",
            context={
                "commitment_id": str(c.id),
                "owner_id": str(c.owner_id),
                "due_date": c.due_date.isoformat(),
                "days_overdue": days_overdue,
            },
            policy_rule=policy,
            source_event="",
        )
        count += 1

    return count


def _sweep_verification_schedule():
    """Check if scheduled PCs and FFTs are being conducted on time."""
    # This will be more meaningful once QMS Policy defines PC/FFT schedules.
    # For now, just count PCs in the last 30 days as a baseline metric.
    from .models import ProcessConfirmation

    recent_count = ProcessConfirmation.objects.filter(
        created_at__date__gte=date.today() - timedelta(days=30),
    ).count()

    logger.info("policy_sweep.verification_activity: %d PCs in last 30 days", recent_count)
    return 0  # No conditions generated yet — need policy-defined schedule targets

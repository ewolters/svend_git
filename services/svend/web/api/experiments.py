"""A/B experiment engine — deterministic variant assignment, conversion tracking, evaluation."""

import hashlib
import logging

from django.utils import timezone

logger = logging.getLogger(__name__)


def assign_variant(user, experiment_name):
    """Assign a user to a variant deterministically. Returns (variant_name, config) or (None, None)."""
    from api.models import Experiment, ExperimentAssignment

    try:
        exp = Experiment.objects.get(name=experiment_name, status="running")
    except Experiment.DoesNotExist:
        return None, None

    # Check existing assignment
    existing = ExperimentAssignment.objects.filter(experiment=exp, user=user).first()
    if existing:
        # Find the variant config
        for v in exp.variants:
            if v["name"] == existing.variant:
                return existing.variant, v.get("config", {})
        return existing.variant, {}

    # Deterministic hash-based assignment
    variants = exp.variants
    if not variants:
        return None, None

    total_weight = sum(v.get("weight", 1) for v in variants)
    hash_input = f"{user.id}-{exp.id}"
    hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16) % total_weight

    cumulative = 0
    chosen = variants[0]
    for v in variants:
        cumulative += v.get("weight", 1)
        if hash_val < cumulative:
            chosen = v
            break

    # Record assignment
    ExperimentAssignment.objects.create(
        experiment=exp,
        user=user,
        variant=chosen["name"],
    )

    return chosen["name"], chosen.get("config", {})


def get_variant(user, experiment_name):
    """Get variant config for a user. Returns config dict or None if no active experiment."""
    _, config = assign_variant(user, experiment_name)
    return config


def record_conversion(user, experiment_name, value=None):
    """Mark a user's experiment assignment as converted."""
    from api.models import ExperimentAssignment

    updated = ExperimentAssignment.objects.filter(
        experiment__name=experiment_name,
        user=user,
        converted=False,
    ).update(
        converted=True,
        converted_at=timezone.now(),
        conversion_value=value,
    )
    return updated > 0


def evaluate_experiment(experiment):
    """Compute per-variant stats and auto-conclude if significant.

    Uses chi-squared test for conversion metrics.
    Returns the results dict.
    """
    from scipy.stats import chi2_contingency

    from api.models import ExperimentAssignment

    assignments = ExperimentAssignment.objects.filter(experiment=experiment)
    variant_names = [v["name"] for v in experiment.variants]

    results = {}
    for name in variant_names:
        qs = assignments.filter(variant=name)
        total = qs.count()
        converted = qs.filter(converted=True).count()
        results[name] = {
            "total": total,
            "converted": converted,
            "rate": round(converted / total * 100, 1) if total else 0,
        }

    # Chi-squared significance test (only meaningful with 2+ variants)
    significant = False
    p_value = None
    if len(variant_names) >= 2:
        observed = []
        for name in variant_names:
            r = results[name]
            observed.append([r["converted"], r["total"] - r["converted"]])

        # Only run test if we have enough data
        if all(r["total"] >= 5 for r in results.values()):
            try:
                chi2, p, dof, expected = chi2_contingency(observed)
                p_value = round(p, 4)
                significant = p < 0.05
                results["_chi2"] = round(chi2, 2)
                results["_p_value"] = p_value
                results["_significant"] = significant
            except Exception:
                pass

    # Auto-conclude if significant and min sample reached
    min_met = all(
        results.get(v, {}).get("total", 0) >= experiment.min_sample_size
        for v in variant_names
    )
    if significant and min_met and experiment.status == "running":
        # Pick winner (highest conversion rate)
        best = max(variant_names, key=lambda v: results.get(v, {}).get("rate", 0))
        experiment.winner = best
        experiment.status = "concluded"
        experiment.ended_at = timezone.now()
        experiment.results = results
        experiment.save(update_fields=["winner", "status", "ended_at", "results"])
        logger.info("Experiment %s auto-concluded. Winner: %s", experiment.name, best)
    else:
        experiment.results = results
        experiment.save(update_fields=["results"])

    return results

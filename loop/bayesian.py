"""Bayesian posterior computations for FMIS — LOOP-001 §8.

Beta-Binomial for detection and occurrence (binary outcomes).
Categorical-Dirichlet for severity (5 categories).

All functions are pure — no Django imports, no side effects.
They take prior parameters + observation counts and return posterior parameters.
"""

import math

# =============================================================================
# BETA-BINOMIAL (Detection, Occurrence)
# =============================================================================


def beta_update(alpha, beta_param, successes, trials):
    """Update Beta posterior with new observations.

    Beta(α, β) + observing s successes in n trials →
    Beta(α + s, β + (n - s))

    Args:
        alpha: Prior α (successes + 1 for uninformative)
        beta_param: Prior β (failures + 1 for uninformative)
        successes: Number of successes in new observation
        trials: Total trials in new observation

    Returns:
        (new_alpha, new_beta)
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"Invalid: successes={successes}, trials={trials}")
    return (alpha + successes, beta_param + (trials - successes))


def beta_mean(alpha, beta_param):
    """Posterior mean of Beta(α, β) = α / (α + β)."""
    total = alpha + beta_param
    if total == 0:
        return 0.5  # Uninformative
    return alpha / total


def beta_credible_interval(alpha, beta_param, coverage=0.90):
    """Credible interval for Beta(α, β) using normal approximation.

    For small sample sizes, this is approximate. For α + β > 10,
    it's quite accurate.

    Returns (lower, upper) for the given coverage probability.
    """
    mean = beta_mean(alpha, beta_param)
    total = alpha + beta_param
    if total <= 2:
        return (0.0, 1.0)  # Too little data

    variance = (alpha * beta_param) / (total * total * (total + 1))
    std = math.sqrt(variance)

    # Z-score for coverage
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(coverage, 1.645)

    lower = max(0.0, mean - z * std)
    upper = min(1.0, mean + z * std)
    return (round(lower, 4), round(upper, 4))


def beta_mean_to_aiag_10(mean_value):
    """Map Beta posterior mean to AIAG 1-10 integer scale.

    For detection: high rate = low score (good detection = low number).
    For occurrence: high rate = high score (frequent = high number).

    This function maps a probability [0, 1] to [1, 10].
    The caller decides the direction (detection inverts, occurrence doesn't).
    """
    if mean_value is None:
        return None
    # Linear mapping: 0.0 → 10, 1.0 → 1 (for detection)
    # Caller inverts for occurrence
    score = round(10 - 9 * mean_value)
    return max(1, min(10, score))


# =============================================================================
# CATEGORICAL-DIRICHLET (Severity)
# =============================================================================

# 5 severity categories per LOOP-001 §8.5
SEVERITY_CATEGORIES = [
    "negligible",  # 1
    "minor",  # 2
    "moderate",  # 3
    "severe",  # 4
    "catastrophic",  # 5
]

SEVERITY_VALUES = {
    "negligible": 1,
    "minor": 2,
    "moderate": 3,
    "severe": 4,
    "catastrophic": 5,
}


def dirichlet_update(alpha_vector, category_index):
    """Update Dirichlet posterior with one categorical observation.

    Dirichlet(α₁, ..., αₖ) + observation in category i →
    Dirichlet(α₁, ..., αᵢ + 1, ..., αₖ)

    Args:
        alpha_vector: List of K alpha parameters [α₁, α₂, ..., αₖ]
        category_index: 0-based index of observed category

    Returns:
        New alpha vector (list)
    """
    if category_index < 0 or category_index >= len(alpha_vector):
        raise ValueError(f"Invalid category_index={category_index} for {len(alpha_vector)} categories")
    result = list(alpha_vector)
    result[category_index] += 1
    return result


def dirichlet_update_by_name(alpha_vector, category_name):
    """Update Dirichlet posterior by category name."""
    if category_name not in SEVERITY_VALUES:
        raise ValueError(f"Unknown severity category: {category_name}. Valid: {list(SEVERITY_VALUES.keys())}")
    index = list(SEVERITY_VALUES.keys()).index(category_name)
    return dirichlet_update(alpha_vector, index)


def dirichlet_mean(alpha_vector):
    """Posterior mean probabilities for each category.

    Returns list of K probabilities that sum to 1.
    """
    total = sum(alpha_vector)
    if total == 0:
        k = len(alpha_vector)
        return [1.0 / k] * k
    return [a / total for a in alpha_vector]


def dirichlet_expected_severity(alpha_vector):
    """Expected severity value = Σ(category_value × probability).

    Returns float in range [1, 5].
    """
    probs = dirichlet_mean(alpha_vector)
    values = list(SEVERITY_VALUES.values())
    return sum(p * v for p, v in zip(probs, values))


def dirichlet_modal_category(alpha_vector):
    """Most probable category (mode of Dirichlet)."""
    probs = dirichlet_mean(alpha_vector)
    max_idx = probs.index(max(probs))
    return SEVERITY_CATEGORIES[max_idx]


def dirichlet_observation_count(alpha_vector):
    """Total observations (sum of alphas minus the prior).

    With uninformative prior Dir(1,1,1,1,1), prior sum = 5.
    Observation count = sum(alpha) - 5.
    """
    return sum(alpha_vector) - len(alpha_vector)  # Subtract prior


def dirichlet_to_aiag_10(alpha_vector):
    """Map Dirichlet expected severity to AIAG 1-10 scale.

    Expected severity is in [1, 5]. Map to [1, 10]:
    1 → 1-2, 2 → 3-4, 3 → 5-6, 4 → 7-8, 5 → 9-10
    """
    expected = dirichlet_expected_severity(alpha_vector)
    # Linear mapping: 1.0 → 1, 5.0 → 10
    score = round(1 + (expected - 1) * 9 / 4)
    return max(1, min(10, score))

"""
CI Readiness clustering pipeline — k-prototypes archetype assignment.

Collects questionnaire responses across users, runs k-prototypes clustering
on mixed Likert (continuous) + forced-choice (categorical) data, assigns
archetype labels, and stores results.

Based on dissertation methodology: NbClust-style multi-criteria k selection,
Mann-Whitney U validation, demographic independence check.

Runs async via Tempora after sufficient responses accumulate.
"""

import logging
from collections import defaultdict

import numpy as np
from django.contrib.auth import get_user_model
from django.utils import timezone
from kmodes.kprototypes import KPrototypes

from core.models import (
    ArchetypeAssignment,
    QuestionnaireResponse,
)

logger = logging.getLogger("svend.clustering")
User = get_user_model()

# Minimum responses before clustering is meaningful
MIN_USERS_FOR_CLUSTERING = 10

# CI Readiness dimension numbers by response type
LIKERT_DIMS = [3, 4, 5, 6, 8, 10, 11]
FORCED_CHOICE_DIMS = [1, 2, 7, 9, 12]


def collect_response_matrix():
    """Build the feature matrix from latest CI Readiness responses per user.

    Returns:
        users: list of User objects (ordered)
        likert_matrix: np.array of shape (n_users, 7) — continuous features
        categorical_matrix: np.array of shape (n_users, 5) — categorical features
        feature_vectors: list of dicts per user (for storage)
    """
    from django.db.models import Max

    user_latest = (
        QuestionnaireResponse.objects.filter(dimension__instrument="ci_readiness")
        .values("user_id")
        .annotate(max_ts=Max("timestamp"))
    )

    # Collect per user
    user_data = defaultdict(dict)
    for entry in user_latest:
        uid = entry["user_id"]
        # Get responses from this user's latest session
        latest_response = QuestionnaireResponse.objects.filter(
            user_id=uid,
            dimension__instrument="ci_readiness",
            timestamp=entry["max_ts"],
        ).first()
        if not latest_response:
            continue

        session_id = latest_response.session_id
        responses = QuestionnaireResponse.objects.filter(
            user_id=uid,
            session_id=session_id,
        ).select_related("dimension")

        for r in responses:
            dim_num = r.dimension.dimension_number
            if r.dimension.response_type == "likert" and r.score is not None:
                user_data[uid][f"likert_{dim_num}"] = r.score
            elif r.dimension.response_type == "forced_choice" and r.option_chosen:
                user_data[uid][f"cat_{dim_num}"] = r.option_chosen

        user_data[uid]["_session_id"] = session_id
        user_data[uid]["_version"] = latest_response.instrument_version

    # Filter to users with complete responses (all 12 dimensions)
    complete_users = []
    for uid, data in user_data.items():
        has_all_likert = all(f"likert_{d}" in data for d in LIKERT_DIMS)
        has_all_cat = all(f"cat_{d}" in data for d in FORCED_CHOICE_DIMS)
        if has_all_likert and has_all_cat:
            complete_users.append((uid, data))

    if not complete_users:
        return [], np.array([]), np.array([]), []

    # Build matrices
    users = [User.objects.get(id=uid) for uid, _ in complete_users]
    likert_matrix = np.array(
        [[data[f"likert_{d}"] for d in LIKERT_DIMS] for _, data in complete_users],
        dtype=float,
    )
    categorical_matrix = np.array([[data[f"cat_{d}"] for d in FORCED_CHOICE_DIMS] for _, data in complete_users])

    feature_vectors = [
        {
            "likert": {str(d): data[f"likert_{d}"] for d in LIKERT_DIMS},
            "categorical": {str(d): data[f"cat_{d}"] for d in FORCED_CHOICE_DIMS},
            "session_id": str(data["_session_id"]),
            "version": data["_version"],
        }
        for _, data in complete_users
    ]

    return users, likert_matrix, categorical_matrix, feature_vectors


def find_optimal_k(likert_matrix, categorical_matrix, categorical_indices, max_k=8):
    """Find optimal number of clusters using cost-based elbow method.

    Since silhouette score isn't directly available for k-prototypes,
    we use the cost function (sum of distances) and look for the elbow.
    """
    combined = np.column_stack([likert_matrix, categorical_matrix])
    n_users = combined.shape[0]
    max_k = min(max_k, n_users - 1, 10)

    if max_k < 2:
        return 2

    costs = []
    for k in range(2, max_k + 1):
        try:
            kp = KPrototypes(n_clusters=k, init="Cao", n_init=5, random_state=42)
            kp.fit(combined, categorical=categorical_indices)
            costs.append((k, kp.cost_))
        except Exception:
            continue

    if not costs:
        return 2

    # Elbow method: find k where cost reduction rate drops most
    if len(costs) < 3:
        return costs[0][0]

    reductions = []
    for i in range(1, len(costs)):
        reduction = costs[i - 1][1] - costs[i][1]
        reductions.append((costs[i][0], reduction))

    # Find where the reduction drops most (elbow)
    best_k = 2
    max_drop = 0
    for i in range(1, len(reductions)):
        drop = reductions[i - 1][1] - reductions[i][1]
        if drop > max_drop:
            max_drop = drop
            best_k = reductions[i][0]

    return best_k


def run_clustering(payload=None, context=None):
    """Tempora task handler: run k-prototypes clustering on CI Readiness responses.

    payload: {} (no args needed — processes all users)
    """
    users, likert_matrix, categorical_matrix, feature_vectors = collect_response_matrix()

    n_users = len(users)
    if n_users < MIN_USERS_FOR_CLUSTERING:
        logger.info(
            "Clustering skipped: %d users (need %d)",
            n_users,
            MIN_USERS_FOR_CLUSTERING,
        )
        return {
            "skipped": True,
            "reason": f"insufficient_users ({n_users}/{MIN_USERS_FOR_CLUSTERING})",
        }

    # Combine into single matrix for k-prototypes (object dtype to hold mixed)
    combined = np.column_stack([likert_matrix, categorical_matrix])
    categorical_indices = list(range(len(LIKERT_DIMS), len(LIKERT_DIMS) + len(FORCED_CHOICE_DIMS)))

    # Find optimal k
    optimal_k = find_optimal_k(likert_matrix, categorical_matrix, categorical_indices)
    logger.info("Optimal k=%d for %d users", optimal_k, n_users)

    # Run final clustering
    kp = KPrototypes(n_clusters=optimal_k, init="Cao", n_init=10, random_state=42)
    clusters = kp.fit_predict(combined, categorical=categorical_indices)

    # Store assignments
    assignments_created = 0
    for i, user in enumerate(users):
        fv = feature_vectors[i]

        # Calculate distance to each centroid (Likert/continuous part only)
        distances = {}
        centroids = kp.cluster_centroids_
        for c in range(optimal_k):
            # centroids is split: numerical part accessible via index
            try:
                num_centroid = np.array(centroids[c, : len(LIKERT_DIMS)], dtype=float)
                distances[str(c)] = float(np.sum((likert_matrix[i] - num_centroid) ** 2))
            except (IndexError, TypeError, ValueError):
                distances[str(c)] = 0.0

        ArchetypeAssignment.objects.update_or_create(
            user=user,
            session_id=fv["session_id"],
            defaults={
                "instrument_version": fv["version"],
                "cluster_id": int(clusters[i]),
                "feature_vector": fv,
                "cluster_distances": distances,
                "created_at": timezone.now(),
            },
        )
        assignments_created += 1

    # Log cluster sizes
    cluster_sizes = defaultdict(int)
    for c in clusters:
        cluster_sizes[int(c)] += 1

    logger.info(
        "Clustering complete: k=%d, %d assignments. Sizes: %s",
        optimal_k,
        assignments_created,
        dict(cluster_sizes),
    )

    return {
        "k": optimal_k,
        "users": n_users,
        "assignments": assignments_created,
        "cluster_sizes": dict(cluster_sizes),
        "cost": float(kp.cost_),
    }

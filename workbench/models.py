"""
Analysis Workbench models — session-based persistence.

Replaces the old reasoning-engine era Workbench/Artifact/KnowledgeGraph/EpistemicLog
with a clean session → dataset → analysis hierarchy.

Session is the saveable unit. Datasets are tabs. Analyses accumulate.
The source contract for pull integration addresses SessionAnalysis by ID
and cherry-picks sub-artifact fields (statistics, narrative, charts, etc.).

Architecture:
    AnalysisSession
    ├── SessionDataset[]         # Multiple dataset tabs
    │   └── parent_datasets[]    # Transform/merge lineage
    └── SessionAnalysis[]        # Accumulated analysis runs
        ├── statistics{}
        ├── narrative{}
        ├── charts[]             # ForgeViz ChartSpec dicts
        ├── diagnostics[]
        ├── assumptions{}
        ├── education{}
        ├── bayesian_shadow{}
        └── evidence_grade
"""

import uuid

from django.conf import settings
from django.db import models


class AnalysisSession(models.Model):
    """Persistent container for a workbench session.

    The top-level saveable unit. Contains datasets and analyses.
    Implements the pull contract as a source container — other tools
    (A3, RCA, investigations) pull from specific analyses within a session.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="analysis_sessions",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="analysis_sessions",
    )
    title = models.CharField(max_length=200, blank=True, default="")
    description = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "analysis_sessions"
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["user", "-updated_at"]),
        ]

    def __str__(self):
        return self.title or f"Session {str(self.id)[:8]}"

    def get_manifest(self):
        """Return pullable manifest for the source contract.

        Lists all analyses with their available sub-artifact keys.
        Consumers browse this to decide what to reference.
        """
        analyses = []
        for a in self.analyses.order_by("created_at"):
            keys = []
            if a.statistics:
                keys.append("statistics")
            if a.narrative:
                keys.append("narrative")
            if a.charts:
                keys.extend(f"charts/{i}" for i in range(len(a.charts)))
            if a.summary:
                keys.append("summary")
            if a.diagnostics:
                keys.append("diagnostics")
            if a.assumptions:
                keys.append("assumptions")
            if a.education:
                keys.append("education")
            if a.bayesian_shadow:
                keys.append("bayesian_shadow")
            if a.evidence_grade:
                keys.append("evidence_grade")
            if a.guide_observation:
                keys.append("guide_observation")

            analyses.append(
                {
                    "id": str(a.id),
                    "analysis_type": a.analysis_type,
                    "analysis_id": a.analysis_id,
                    "dataset_name": a.dataset.name if a.dataset else None,
                    "columns_used": a.columns_used,
                    "created_at": a.created_at.isoformat(),
                    "available_keys": keys,
                }
            )

        return {
            "session_id": str(self.id),
            "title": self.title,
            "datasets": [
                {"id": str(d.id), "name": d.name, "row_count": d.row_count}
                for d in self.datasets.order_by("created_at")
            ],
            "analyses": analyses,
        }


class SessionDataset(models.Model):
    """One dataset tab in a session.

    Created by upload (with optional triage clean), or by transform/merge
    operations on existing datasets. Lineage tracked via parent_datasets.
    Data stored as JSON — fully self-contained, no file system dependency.
    """

    class Source(models.TextChoices):
        UPLOAD = "upload", "Upload"
        TRIAGE = "triage", "Triage (cleaned)"
        TRANSFORM = "transform", "Transform"
        MERGE = "merge", "Merge"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        AnalysisSession,
        on_delete=models.CASCADE,
        related_name="datasets",
    )
    name = models.CharField(max_length=200)
    source = models.CharField(
        max_length=20,
        choices=Source.choices,
        default=Source.UPLOAD,
    )
    parent_datasets = models.ManyToManyField(
        "self",
        symmetrical=False,
        blank=True,
        help_text="Datasets that were merged/transformed to create this one",
    )

    # The actual data — JSON columnar format: {"col1": [v1, v2, ...], "col2": [...]}
    data = models.JSONField(default=dict)

    # Column metadata: [{"name": "col1", "dtype": "numeric", "missing_pct": 0.02, "unique": 45}]
    columns_meta = models.JSONField(default=list)

    row_count = models.IntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "session_datasets"
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.name} ({self.row_count} rows)"


class SessionAnalysis(models.Model):
    """One analysis run within a session.

    The addressable artifact for the pull contract. Contains the full
    10-key result contract: statistics, narrative, charts, diagnostics,
    assumptions, education, bayesian_shadow, evidence_grade, summary,
    guide_observation.

    Consumers address sub-artifacts via key path:
        /sessions/<id>/analyses/<id>/statistics/p_value
        /sessions/<id>/analyses/<id>/charts/0
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        AnalysisSession,
        on_delete=models.CASCADE,
        related_name="analyses",
    )
    dataset = models.ForeignKey(
        SessionDataset,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="analyses",
    )

    # What was run
    analysis_type = models.CharField(max_length=50)
    analysis_id = models.CharField(max_length=100)
    columns_used = models.JSONField(default=list)
    config = models.JSONField(default=dict)

    # Result container — the 10-key contract
    statistics = models.JSONField(default=dict)
    narrative = models.JSONField(default=dict)
    summary = models.TextField(blank=True, default="")
    charts = models.JSONField(default=list)
    diagnostics = models.JSONField(default=list)
    assumptions = models.JSONField(default=dict)
    education = models.JSONField(null=True, blank=True)
    bayesian_shadow = models.JSONField(null=True, blank=True)
    evidence_grade = models.CharField(max_length=20, blank=True, default="")
    guide_observation = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "session_analyses"
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["session", "created_at"]),
        ]

    def __str__(self):
        return f"{self.analysis_type}/{self.analysis_id} on {self.dataset}"

    def get_sub_artifact(self, key_path):
        """Retrieve a sub-artifact by key path for the pull contract.

        Examples:
            get_sub_artifact("statistics/p_value") → 0.003
            get_sub_artifact("charts/0") → {ChartSpec dict}
            get_sub_artifact("narrative/verdict") → "Significant difference..."
        """
        parts = key_path.strip("/").split("/")
        obj = self

        for part in parts:
            if isinstance(obj, models.Model):
                obj = getattr(obj, part, None)
            elif isinstance(obj, dict):
                obj = obj.get(part)
            elif isinstance(obj, list):
                try:
                    obj = obj[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None

        return obj

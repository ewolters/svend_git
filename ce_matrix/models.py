# ce_matrix models — extracted from agents_api.models
# db_table overrides point at existing tables, no migration needed

import uuid

from django.conf import settings
from django.db import models


class CEMatrix(models.Model):
    """Cause & Effect Matrix — scoring grid for prioritizing causes.

    Separate tool from Ishikawa. Inputs (X's) scored against outputs (Y's)
    with standard 0/1/3/9 scoring and output importance weights (1-10).
    """

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        SCORING = "scoring", "Scoring"
        COMPLETE = "complete", "Complete"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="ce_matrices",
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="ce_mat_matrices",
    )
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="ce_mat_matrices",
    )
    title = models.CharField(max_length=255, blank=True)
    outputs = models.JSONField(
        default=list,
        help_text='Process outputs (Y\'s): [{"name": "...", "weight": 1-10}]',
    )
    inputs = models.JSONField(
        default=list,
        help_text='Process inputs (X\'s): [{"name": "..."}]',
    )
    scores = models.JSONField(
        default=dict,
        help_text="Scoring grid: str(input_idx) → str(output_idx) → 0|1|3|9",
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.DRAFT,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "ce_matrices"

        ordering = ["-updated_at"]
        verbose_name = "C&E Matrix"
        verbose_name_plural = "C&E Matrices"

    def __str__(self):
        return f"C&E Matrix: {self.title or 'Untitled'} ({self.status})"

    def compute_totals(self):
        """Compute weighted total for each input.

        Returns list of {"input_name": str, "total": float} sorted descending.
        """
        totals = []
        for i, inp in enumerate(self.inputs):
            total = 0.0
            input_scores = self.scores.get(str(i), {})
            for j, out in enumerate(self.outputs):
                score = input_scores.get(str(j), 0)
                weight = out.get("weight", 1)
                total += score * weight
            totals.append({"input_name": inp.get("name", ""), "input_index": i, "total": total})
        totals.sort(key=lambda x: x["total"], reverse=True)
        return totals

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id) if self.project_id else None,
            "title": self.title,
            "outputs": self.outputs,
            "inputs": self.inputs,
            "scores": self.scores,
            "totals": self.compute_totals(),
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

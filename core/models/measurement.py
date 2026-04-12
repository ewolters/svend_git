"""
Measurement system models — CANON-002 §4, §12.2.

MeasurementSystem represents a physical instrument. GageStudy
represents a Gage R&R or attribute agreement study linked to that
instrument. Together they implement the measurement validity gate.

Reference: docs/standards/CANON-002.md §4.1-4.5, §12.2
"""

from django.conf import settings
from django.db import models

from syn.core.base_models import SynaraEntity


class MeasurementSystem(SynaraEntity):
    """
    A physical measurement instrument or system (CANON-002 §12.2).

    The measurement system is not a weighting factor — it is a validity gate.
    If the instrument cannot distinguish between parts, evidence from it is
    unreliable regardless of study design or sample size.
    """

    class SystemType(models.TextChoices):
        VARIABLE = "variable", "Variable"
        ATTRIBUTE = "attribute", "Attribute"

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        INACTIVE = "inactive", "Inactive"
        QUARANTINED = "quarantined", "Quarantined"

    name = models.CharField(max_length=200, help_text='Instrument identifier, e.g. "Keyence IM-8000 #3"')
    system_type = models.CharField(
        max_length=20,
        choices=SystemType.choices,
        help_text="Variable (continuous) or attribute (pass/fail)",
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="measurement_systems",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="measurement_systems",
    )
    calibration_due = models.DateField(null=True, blank=True, help_text="Next calibration due date")
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE,
        help_text="Quarantined = failed GRR, pending resolution",
    )

    class Meta:
        db_table = "core_measurement_system"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.get_system_type_display()}, {self.status})"

    @property
    def current_validity(self) -> float:
        """
        Returns measurement_validity from the most recent completed GageStudy,
        or 0.55 (unvalidated default) if no study exists.
        CANON-002 §4.3.
        """
        study = self.gage_studies.filter(completed_at__isnull=False).order_by("-completed_at").first()
        if not study:
            return 0.55
        return study.measurement_validity


class GageStudy(SynaraEntity):
    """
    A Gage R&R or attribute agreement study linked to an instrument (CANON-002 §12.2).

    On save, auto-quarantines the parent measurement system when %GRR > 30%
    (variable) or Kappa < 0.50 (attribute).
    """

    class StudyType(models.TextChoices):
        GRR_CROSSED = "grr_crossed", "GRR Crossed"
        GRR_NESTED = "grr_nested", "GRR Nested"
        ATTRIBUTE_AGREEMENT = "attribute_agreement", "Attribute Agreement"

    measurement_system = models.ForeignKey(
        MeasurementSystem,
        on_delete=models.CASCADE,
        related_name="gage_studies",
    )
    study_type = models.CharField(max_length=30, choices=StudyType.choices)
    completed_at = models.DateTimeField(null=True, blank=True, help_text="When study was completed")

    # Variable study results
    grr_percent = models.FloatField(null=True, blank=True, help_text="%GRR for variable studies")
    ndc = models.IntegerField(null=True, blank=True, help_text="Number of distinct categories")

    # Attribute study results
    kappa = models.FloatField(null=True, blank=True, help_text="Kappa for attribute studies")
    percent_agreement = models.FloatField(null=True, blank=True, help_text="%Agreement for attribute studies")

    class Meta:
        db_table = "core_gage_study"
        ordering = ["-completed_at"]

    def __str__(self):
        return f"{self.get_study_type_display()} for {self.measurement_system.name}"

    @property
    def measurement_validity(self) -> float:
        """
        Compute measurement validity per CANON-002 §4.1 (variable) / §4.4 (attribute).

        Variable (%GRR):
            ≤ 10%   → 1.0  (valid)
            10-20%  → 0.80 (marginal)
            20-30%  → 0.50 (poor)
            > 30%   → 0.10 (invalid)

        Attribute (Kappa):
            ≥ 0.90  → 1.0  (valid)
            0.75-0.90 → 0.80 (marginal)
            0.50-0.75 → 0.50 (poor)
            < 0.50  → 0.10 (invalid)
        """
        if self.study_type == self.StudyType.ATTRIBUTE_AGREEMENT:
            k = self.kappa
            if k is None:
                return 0.55
            if k >= 0.90:
                return 1.0
            if k >= 0.75:
                return 0.80
            if k >= 0.50:
                return 0.50
            return 0.10
        else:
            grr = self.grr_percent
            if grr is None:
                return 0.55
            if grr <= 10:
                return 1.0
            if grr <= 20:
                return 0.80
            if grr <= 30:
                return 0.50
            return 0.10

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        # Auto-quarantine per CANON-002 §12.2
        if self.completed_at and self._should_quarantine():
            ms = self.measurement_system
            if ms.status != MeasurementSystem.Status.QUARANTINED:
                ms.status = MeasurementSystem.Status.QUARANTINED
                ms.save(update_fields=["status", "updated_at"])

    def _should_quarantine(self) -> bool:
        """Check if this study result warrants quarantine."""
        if self.study_type == self.StudyType.ATTRIBUTE_AGREEMENT:
            return self.kappa is not None and self.kappa < 0.50
        else:
            return self.grr_percent is not None and self.grr_percent > 30

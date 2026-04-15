# fmea models — extracted from agents_api.models
# db_table overrides point at existing tables, no migration needed

import uuid

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class FMEA(models.Model):
    """Failure Mode and Effects Analysis study."""

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        ACTIVE = "active", "Active"
        REVIEW = "review", "Under Review"
        COMPLETE = "complete", "Complete"

    class FMEAType(models.TextChoices):
        PROCESS = "process", "Process FMEA"
        DESIGN = "design", "Design FMEA"
        SYSTEM = "system", "System FMEA"

    class ScoringMethod(models.TextChoices):
        RPN = "rpn", "Risk Priority Number"
        AP = "ap", "Action Priority (AIAG/VDA)"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="fmea_fmeas",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="fmea_records_ext",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="fmea_fmeas",
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="fmea_fmeas_created",
    )

    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="fmea_fmeas",
    )

    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)
    fmea_type = models.CharField(max_length=20, choices=FMEAType.choices, default=FMEAType.PROCESS)
    scoring_method = models.CharField(max_length=10, choices=ScoringMethod.choices, default=ScoringMethod.RPN)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "fmeas"

        ordering = ["-updated_at"]
        verbose_name = "FMEA"
        verbose_name_plural = "FMEAs"

    def __str__(self):
        return f"FMEA: {self.title} ({self.status})"

    def to_dict(self):
        rows = list(self.rows.order_by("sort_order"))
        return {
            "id": str(self.id),
            "project_id": str(self.project_id) if self.project_id else None,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "site_id": str(self.site_id) if self.site_id else None,
            "created_by_id": str(self.created_by_id) if self.created_by_id else None,
            "fmea_type": self.fmea_type,
            "scoring_method": self.scoring_method,
            "rows": [r.to_dict() for r in rows],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_manifest(self):
        """Pull contract manifest — lists all rows as sub-artifacts."""
        rows = list(self.rows.order_by("sort_order"))
        return {
            "container_id": str(self.id),
            "container_type": "FMEA",
            "title": self.title,
            "status": self.status,
            "fmea_type": self.fmea_type,
            "scoring_method": self.scoring_method,
            "artifacts": [
                {
                    "id": str(r.id),
                    "type": "FMEARow",
                    "label": r.failure_mode,
                    "available_keys": [
                        "severity",
                        "occurrence",
                        "detection",
                        "rpn",
                        "cause",
                        "effect",
                        "process_step",
                        "recommended_action",
                        "action_status",
                    ],
                }
                for r in rows
            ],
            "updated_at": self.updated_at.isoformat(),
        }


class FMEARow(models.Model):
    """A single failure mode row in an FMEA study."""

    class ActionStatus(models.TextChoices):
        NOT_STARTED = "not_started", "Not Started"
        IN_PROGRESS = "in_progress", "In Progress"
        COMPLETE = "complete", "Complete"

    class FailureModeClass(models.TextChoices):
        FORM = "form", "Form"
        FIT = "fit", "Fit"
        FUNCTION = "function", "Function"
        SAFETY = "safety", "Safety"
        REGULATORY = "regulatory", "Regulatory"

    class ControlType(models.TextChoices):
        PREVENT = "prevent", "Prevention"
        DETECT = "detect", "Detection"
        BOTH = "both", "Prevention & Detection"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    fmea = models.ForeignKey(FMEA, on_delete=models.CASCADE, related_name="rows")
    sort_order = models.IntegerField(default=0)

    process_step = models.CharField(max_length=255, blank=True)
    failure_mode = models.CharField(max_length=255)
    effect = models.TextField(blank=True)

    severity = models.IntegerField(default=1, validators=[MinValueValidator(1), MaxValueValidator(10)])
    cause = models.TextField(blank=True)
    occurrence = models.IntegerField(default=1, validators=[MinValueValidator(1), MaxValueValidator(10)])
    current_controls = models.TextField(blank=True)
    prevention_controls = models.TextField(blank=True)
    detection_controls = models.TextField(blank=True)
    failure_mode_class = models.CharField(max_length=20, choices=FailureModeClass.choices, blank=True)
    control_type = models.CharField(max_length=20, choices=ControlType.choices, blank=True)
    detection = models.IntegerField(default=1, validators=[MinValueValidator(1), MaxValueValidator(10)])

    rpn = models.IntegerField(default=1)

    recommended_action = models.TextField(blank=True)
    action_owner = models.CharField(max_length=255, blank=True)
    action_status = models.CharField(
        max_length=20,
        choices=ActionStatus.choices,
        default=ActionStatus.NOT_STARTED,
    )

    revised_severity = models.IntegerField(null=True, blank=True)
    revised_occurrence = models.IntegerField(null=True, blank=True)
    revised_detection = models.IntegerField(null=True, blank=True)
    revised_rpn = models.IntegerField(null=True, blank=True)

    spc_measurement = models.CharField(max_length=255, blank=True, default="")

    hypothesis_link = models.ForeignKey(
        "core.Hypothesis",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="fmea_rows_ext",
        help_text="Hypothesis this failure mode relates to",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "fmea_rows"

        ordering = ["sort_order"]

    def __str__(self):
        return f"{self.failure_mode} (RPN={self.rpn})"

    def save(self, *args, **kwargs):
        from django.core.exceptions import ValidationError

        for field_name in ("severity", "occurrence", "detection"):
            val = getattr(self, field_name)
            if val is None:
                setattr(self, field_name, 1)
            elif not (1 <= val <= 10):
                raise ValidationError({field_name: f"Must be 1-10, got {val}"})
        self.rpn = self.severity * self.occurrence * self.detection
        revised = [self.revised_severity, self.revised_occurrence, self.revised_detection]
        has_revised = [v is not None for v in revised]
        if any(has_revised) and not all(has_revised):
            raise ValidationError("Revised scores must set all three (S/O/D) or none")
        if all(has_revised):
            for field_name in ("revised_severity", "revised_occurrence", "revised_detection"):
                val = getattr(self, field_name)
                if not (1 <= val <= 10):
                    raise ValidationError({field_name: f"Must be 1-10, got {val}"})
            self.revised_rpn = self.revised_severity * self.revised_occurrence * self.revised_detection
        else:
            self.revised_rpn = None
        super().save(*args, **kwargs)

    @staticmethod
    def compute_action_priority(severity, occurrence, detection):
        """Compute AIAG/VDA Action Priority (H/M/L) from S, O, D scores."""
        s, o, d = severity, occurrence, detection
        if s >= 9:
            if o >= 4:
                return "H"
            if o >= 2 and d >= 2:
                return "H"
            if o >= 2:
                return "H"
        if s >= 7:
            if o >= 5:
                return "H"
            if o >= 4 and d >= 4:
                return "H"
        if s >= 5:
            if o >= 8:
                return "H"
        if s >= 9:
            return "M"
        if s >= 7:
            if o >= 3:
                return "M"
            if o >= 2 and d >= 4:
                return "M"
        if s >= 5:
            if o >= 5:
                return "M"
            if o >= 4 and d >= 4:
                return "M"
        if s >= 4:
            if o >= 7:
                return "M"
        if s >= 2:
            if o >= 8 and d >= 7:
                return "M"
        return "L"

    def to_dict(self):
        return {
            "id": str(self.id),
            "fmea_id": str(self.fmea_id),
            "sort_order": self.sort_order,
            "process_step": self.process_step,
            "failure_mode": self.failure_mode,
            "effect": self.effect,
            "severity": self.severity,
            "cause": self.cause,
            "occurrence": self.occurrence,
            "current_controls": self.current_controls,
            "prevention_controls": self.prevention_controls,
            "detection_controls": self.detection_controls,
            "failure_mode_class": self.failure_mode_class,
            "control_type": self.control_type,
            "detection": self.detection,
            "rpn": self.rpn,
            "action_priority": self.compute_action_priority(self.severity, self.occurrence, self.detection),
            "recommended_action": self.recommended_action,
            "action_owner": self.action_owner,
            "action_status": self.action_status,
            "revised_severity": self.revised_severity,
            "revised_occurrence": self.revised_occurrence,
            "revised_detection": self.revised_detection,
            "revised_rpn": self.revised_rpn,
            "hypothesis_id": (str(self.hypothesis_link_id) if self.hypothesis_link_id else None),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

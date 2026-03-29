"""
Training center models — ILSSI partner onboarding infrastructure.

Manages training organizations, programs (cohorts), and student enrollments.
Staff-only administration via internal dashboard.

Flow:
  TrainingCenter (ILSSI partner org)
    → TrainingProgram (a cohort/class with dates)
      → StudentEnrollment (user → program, tracks enrollment → graduation → conversion)

Pricing:
  - Students: free Pro during enrollment
  - Graduates: 50% lifetime discount (student_pro Stripe price)
  - Instructors: free Enterprise (manual upgrade by staff)
  - NGOs: separate pricing tier (ngo_pro, ngo_team Stripe prices)
"""

import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone


class TrainingCenter(models.Model):
    """An ILSSI-accredited training organization.

    Examples: Go Lean and Grow (Switzerland/SA), ILSSI direct, etc.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    country = models.CharField(max_length=2, help_text="ISO 3166-1 alpha-2 code")
    contact_name = models.CharField(max_length=255, blank=True, default="")
    contact_email = models.EmailField(blank=True, default="")
    website = models.URLField(blank=True, default="")

    # The instructor/admin user at the center (gets free Enterprise)
    instructor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="training_centers",
        help_text="Primary instructor — receives free Enterprise tier",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="training_centers",
    )

    is_ilssi_partner = models.BooleanField(default=True)
    is_ngo = models.BooleanField(default=False, help_text="NGO/non-profit — eligible for NGO pricing")
    notes = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "core_training_center"
        ordering = ["name"]

    def __str__(self):
        return self.name

    @property
    def active_programs(self):
        return self.programs.filter(status=TrainingProgram.Status.ACTIVE)

    @property
    def total_students(self):
        return StudentEnrollment.objects.filter(program__center=self).count()


class TrainingProgram(models.Model):
    """A cohort or class run by a training center.

    Has a start/end date. Students enrolled in the program get free Pro access.
    On graduation, they have 30 days to convert at the alumni discount rate.
    """

    class Status(models.TextChoices):
        PLANNED = "planned", "Planned"
        ACTIVE = "active", "Active"
        COMPLETED = "completed", "Completed"
        CANCELLED = "cancelled", "Cancelled"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    center = models.ForeignKey(
        TrainingCenter,
        on_delete=models.CASCADE,
        related_name="programs",
    )
    title = models.CharField(max_length=300, help_text="e.g., 'Lean Six Sigma Green Belt — SA Cohort 1'")
    description = models.TextField(blank=True, default="")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PLANNED)

    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)

    # Region for pricing (maps to REGIONAL_PRICES key in billing.py)
    region = models.CharField(
        max_length=10,
        default="us",
        help_text="Pricing region code (za, in, us, etc.)",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "core_training_program"
        ordering = ["-start_date"]

    def __str__(self):
        return f"{self.title} ({self.center.name})"

    @property
    def enrolled_count(self):
        return self.enrollments.filter(status=StudentEnrollment.Status.ENROLLED).count()

    @property
    def graduated_count(self):
        return self.enrollments.filter(status=StudentEnrollment.Status.GRADUATED).count()


class StudentEnrollment(models.Model):
    """Links a user to a training program.

    Lifecycle: enrolled → graduated → converted (or expired)
    - enrolled: free Pro access
    - graduated: 30-day window to subscribe at alumni discount
    - converted: subscribed at alumni rate (50% lifetime)
    - expired: did not convert within 30-day window
    """

    class Status(models.TextChoices):
        ENROLLED = "enrolled", "Enrolled"
        GRADUATED = "graduated", "Graduated"
        CONVERTED = "converted", "Converted"
        EXPIRED = "expired", "Expired"
        WITHDRAWN = "withdrawn", "Withdrawn"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="enrollments",
    )
    program = models.ForeignKey(
        TrainingProgram,
        on_delete=models.CASCADE,
        related_name="enrollments",
    )
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.ENROLLED)

    enrolled_at = models.DateTimeField(auto_now_add=True)
    graduated_at = models.DateTimeField(null=True, blank=True)
    conversion_deadline = models.DateTimeField(
        null=True,
        blank=True,
        help_text="30 days after graduation — must subscribe by this date for alumni rate",
    )
    converted_at = models.DateTimeField(null=True, blank=True)

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        on_delete=models.SET_NULL,
        related_name="+",
        help_text="Staff member who created this enrollment",
    )

    class Meta:
        db_table = "core_student_enrollment"
        unique_together = [("user", "program")]
        ordering = ["-enrolled_at"]

    def __str__(self):
        return f"{self.user.email} → {self.program.title} ({self.status})"

    def graduate(self):
        """Mark student as graduated and set 30-day conversion window."""
        self.status = self.Status.GRADUATED
        self.graduated_at = timezone.now()
        self.conversion_deadline = timezone.now() + timezone.timedelta(days=30)
        self.save(update_fields=["status", "graduated_at", "conversion_deadline"])

    def convert(self):
        """Mark student as converted (subscribed at alumni rate)."""
        self.status = self.Status.CONVERTED
        self.converted_at = timezone.now()
        self.save(update_fields=["status", "converted_at"])

    def expire(self):
        """Mark as expired (did not convert within window)."""
        self.status = self.Status.EXPIRED
        self.save(update_fields=["status"])

    @property
    def is_within_conversion_window(self):
        if self.status != self.Status.GRADUATED:
            return False
        if not self.conversion_deadline:
            return False
        return timezone.now() < self.conversion_deadline

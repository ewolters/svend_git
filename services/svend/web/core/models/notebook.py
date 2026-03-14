"""
Notebook models — NB-001.

The Notebook is the tactical execution layer for continuous improvement.
It is a scrapbook with compute: trials, calculator outputs, simulator
results, and any other artifact generated while improving a process.

Hierarchy:
  Project (Charter) → Notebook[] → Trial[], NotebookPage[]
  Notebook → HanseiKai (on conclusion)
  HanseiKai → Yokoten (if carry_forward=True)
  Yokoten → YokotenAdoption[]

Reference: docs/standards/NB-001.md
"""

import uuid

from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models

# =============================================================================
# NOTEBOOK
# =============================================================================


class Notebook(models.Model):
    """
    A tactical execution log attached to a Project (Charter).

    NB-001 §2.1. State machine: open → active → concluded → archived.
    """

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        ACTIVE = "active", "Active"
        CONCLUDED = "concluded", "Concluded"
        ARCHIVED = "archived", "Archived"

    VALID_TRANSITIONS = {
        "open": ["active"],
        "active": ["concluded"],
        "concluded": ["archived"],
        "archived": [],
    }

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.CASCADE,
        related_name="notebooks",
        help_text="The charter this notebook executes against",
    )
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True, default="")
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.OPEN,
        db_index=True,
    )

    # Baseline — the starting point
    baseline_summary = models.TextField(blank=True, default="")
    baseline_metric = models.CharField(max_length=100, blank=True, default="")
    baseline_value = models.FloatField(null=True, blank=True)
    baseline_unit = models.CharField(max_length=50, blank=True, default="")
    baseline_analysis = models.ForeignKey(
        "agents_api.DSWResult",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    baseline_date = models.DateField(null=True, blank=True)

    # Current state — updated after each adopted trial
    current_value = models.FloatField(null=True, blank=True)
    current_analysis = models.ForeignKey(
        "agents_api.DSWResult",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    current_date = models.DateField(null=True, blank=True)

    # Active trial quick-access
    active_trial = models.ForeignKey(
        "core.Trial",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )

    # Synara causal graph (per notebook)
    synara_state = models.JSONField(default=dict, blank=True)

    # Ownership
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="notebooks",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="notebooks",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    concluded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "core_notebook"
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["owner", "status"]),
            models.Index(fields=["project", "status"]),
        ]

    def __str__(self):
        return f"{self.title} ({self.status})"

    def transition_to(self, target_status):
        """State machine per NB-001 §2.1."""
        valid = self.VALID_TRANSITIONS.get(self.status, [])
        if target_status not in valid:
            raise ValueError(f"Cannot transition from {self.status} to {target_status}")
        self.status = target_status
        if target_status == self.Status.CONCLUDED:
            from django.utils import timezone

            self.concluded_at = timezone.now()
        self.save()

    def auto_activate(self):
        """Transition to active if still open (called when first trial/page created)."""
        if self.status == self.Status.OPEN:
            self.status = self.Status.ACTIVE
            self.save(update_fields=["status", "updated_at"])

    def update_current_from_trial(self, trial):
        """Update current state from an adopted trial's after values."""
        if trial.adopted and trial.after_value is not None:
            self.current_value = trial.after_value
            self.current_analysis = trial.after_analysis
            self.current_date = trial.after_date
            self.save(
                update_fields=[
                    "current_value",
                    "current_analysis",
                    "current_date",
                    "updated_at",
                ]
            )

    @property
    def progress_pct(self):
        """Progress toward charter goal as percentage."""
        if self.baseline_value is None or self.current_value is None:
            return None
        if not self.project:
            return None
        target = None
        baseline = None
        try:
            target = float(self.project.goal_target) if self.project.goal_target else None
            baseline = float(self.project.goal_baseline) if self.project.goal_baseline else None
        except (ValueError, TypeError):
            return None
        if target is None or baseline is None or target == baseline:
            return None
        return ((self.current_value - self.baseline_value) / (target - baseline)) * 100


# =============================================================================
# TRIAL
# =============================================================================


class Trial(models.Model):
    """
    A structured before/after comparison within a Notebook.

    NB-001 §2.2. The core primitive of improvement tracking.
    """

    class Verdict(models.TextChoices):
        IMPROVED = "improved", "Improved"
        NO_EFFECT = "no_effect", "No Effect"
        DEGRADED = "degraded", "Degraded"
        INCONCLUSIVE = "inconclusive", "Inconclusive"
        PENDING = "pending", "Pending"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    notebook = models.ForeignKey(
        Notebook,
        on_delete=models.CASCADE,
        related_name="trials",
    )
    sequence = models.PositiveIntegerField(
        help_text="Auto-increment within notebook (Trial 1, 2, 3...)",
    )
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True, default="")

    # Before state
    before_value = models.FloatField(null=True, blank=True)
    before_analysis = models.ForeignKey(
        "agents_api.DSWResult",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    before_date = models.DateField(null=True, blank=True)

    # After state
    after_value = models.FloatField(null=True, blank=True)
    after_analysis = models.ForeignKey(
        "agents_api.DSWResult",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="+",
    )
    after_date = models.DateField(null=True, blank=True)

    # Verdict
    verdict = models.CharField(
        max_length=20,
        choices=Verdict.choices,
        default=Verdict.PENDING,
    )
    verdict_narrative = models.TextField(
        blank=True,
        default="",
        help_text="System-generated narrative with statistical significance",
    )
    delta = models.FloatField(null=True, blank=True)
    delta_pct = models.FloatField(null=True, blank=True)
    adopted = models.BooleanField(default=False)

    # Evidence linkage
    evidence_links = models.ManyToManyField(
        "core.Evidence",
        blank=True,
        related_name="trials",
    )

    # Metadata
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="created_trials",
    )

    class Meta:
        db_table = "core_trial"
        ordering = ["sequence"]
        unique_together = [("notebook", "sequence")]

    def __str__(self):
        return f"Trial {self.sequence}: {self.title} ({self.verdict})"

    def save(self, **kwargs):
        # Auto-assign sequence
        if self.sequence is None or self.sequence == 0:
            max_seq = (
                Trial.objects.filter(notebook=self.notebook).aggregate(models.Max("sequence")).get("sequence__max") or 0
            )
            self.sequence = max_seq + 1
        # Compute delta
        if self.before_value is not None and self.after_value is not None:
            self.delta = self.after_value - self.before_value
            if self.before_value != 0:
                self.delta_pct = (self.delta / self.before_value) * 100
        super().save(**kwargs)
        # Auto-activate notebook
        self.notebook.auto_activate()

    def complete(self, verdict, adopted=False):
        """Mark trial as completed with a verdict."""
        from django.utils import timezone

        self.verdict = verdict
        self.adopted = adopted
        self.completed_at = timezone.now()
        self.save()
        if adopted:
            self.notebook.update_current_from_trial(self)


# =============================================================================
# NOTEBOOK PAGE
# =============================================================================


class NotebookPage(models.Model):
    """
    A frozen snapshot of calculator, simulator, analysis, or note output.

    NB-001 §2.3. Pages are immutable once created — inputs and outputs
    are point-in-time records, not live links.
    """

    class PageType(models.TextChoices):
        CALCULATOR = "calculator", "Calculator"
        SIMULATOR = "simulator", "Simulator"
        ANALYSIS = "analysis", "Analysis"
        NOTE = "note", "Note"
        ATTACHMENT = "attachment", "Attachment"

    class TrialRole(models.TextChoices):
        BEFORE = "before", "Before"
        AFTER = "after", "After"
        SUPPORTING = "supporting", "Supporting"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    notebook = models.ForeignKey(
        Notebook,
        on_delete=models.CASCADE,
        related_name="pages",
    )
    page_type = models.CharField(max_length=20, choices=PageType.choices)
    title = models.CharField(max_length=300)
    source_tool = models.CharField(
        max_length=50,
        blank=True,
        default="",
        help_text="Tool identifier: oee, kanban, takt, cpk, plantsim, dsw, etc.",
    )
    inputs = models.JSONField(default=dict, blank=True, help_text="Frozen inputs")
    outputs = models.JSONField(default=dict, blank=True, help_text="Frozen outputs")
    rendered_html = models.TextField(blank=True, default="", help_text="Frozen visual")
    narrative = models.TextField(blank=True, default="")
    sequence = models.PositiveIntegerField(default=0)

    # Optional trial linkage
    trial = models.ForeignKey(
        Trial,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="pages",
    )
    trial_role = models.CharField(
        max_length=20,
        choices=TrialRole.choices,
        blank=True,
        default="",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="notebook_pages",
    )

    class Meta:
        db_table = "core_notebook_page"
        ordering = ["sequence", "created_at"]

    def __str__(self):
        return f"{self.title} ({self.page_type})"

    def save(self, **kwargs):
        if self.sequence == 0:
            max_seq = (
                NotebookPage.objects.filter(notebook=self.notebook)
                .aggregate(models.Max("sequence"))
                .get("sequence__max")
                or 0
            )
            self.sequence = max_seq + 1
        super().save(**kwargs)
        self.notebook.auto_activate()


# =============================================================================
# TRIAL TOOL LINK
# =============================================================================


class TrialToolLink(models.Model):
    """
    Links any tool output to a Trial via generic FK.

    NB-001 §2.4. Absorbs InvestigationToolLink pattern.
    """

    class Role(models.TextChoices):
        BEFORE = "before", "Before"
        AFTER = "after", "After"
        SUPPORTING = "supporting", "Supporting"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    trial = models.ForeignKey(
        Trial,
        on_delete=models.CASCADE,
        related_name="tool_links",
    )
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.UUIDField()
    tool_output = GenericForeignKey("content_type", "object_id")
    tool_type = models.CharField(max_length=30)
    role = models.CharField(max_length=20, choices=Role.choices)
    linked_at = models.DateTimeField(auto_now_add=True)
    linked_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )

    class Meta:
        db_table = "core_trial_tool_link"
        unique_together = [("trial", "content_type", "object_id")]


# =============================================================================
# TRIAL ACTION
# =============================================================================


class TrialAction(models.Model):
    """
    Records actions triggered from a Trial (CAPA, audit, etc.).

    NB-001 §2.5. Absorbs StudyAction.
    """

    class ActionType(models.TextChoices):
        RAISE_CAPA = "raise_capa", "Raise CAPA"
        SCHEDULE_AUDIT = "schedule_audit", "Schedule Verification Audit"
        REQUEST_DOC_UPDATE = "request_doc_update", "Request Document Update"
        FLAG_TRAINING_GAP = "flag_training_gap", "Flag Training Gap"
        FLAG_FMEA_UPDATE = "flag_fmea_update", "Flag FMEA Update"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    trial = models.ForeignKey(
        Trial,
        on_delete=models.CASCADE,
        related_name="actions",
    )
    action_type = models.CharField(max_length=30, choices=ActionType.choices)
    target_type = models.CharField(max_length=30)
    target_id = models.UUIDField()
    notes = models.TextField(blank=True, default="")
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "core_trial_action"
        ordering = ["-created_at"]


# =============================================================================
# HANSEI KAI (REFLECTION)
# =============================================================================


class HanseiKai(models.Model):
    """
    Reflection recorded when a Notebook is concluded.

    NB-001 §2.6. Three questions + key learning distillation.
    If carry_forward=True, a Yokoten is auto-created.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    notebook = models.OneToOneField(
        Notebook,
        on_delete=models.CASCADE,
        related_name="hansei_kai",
    )
    what_went_well = models.TextField()
    what_didnt = models.TextField()
    what_next = models.TextField()
    key_learning = models.TextField(help_text="One sentence distillation — the yokoten seed")
    carry_forward = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )

    class Meta:
        db_table = "core_hansei_kai"

    def __str__(self):
        return f"Hansei Kai: {self.notebook.title}"

    def save(self, **kwargs):
        super().save(**kwargs)
        if self.carry_forward:
            Yokoten.objects.get_or_create(
                source_notebook=self.notebook,
                defaults={
                    "learning": self.key_learning,
                    "context": "",
                    "applicable_to": [],
                    "created_by": self.created_by,
                },
            )


# =============================================================================
# YOKOTEN (LATERAL TRANSFER)
# =============================================================================


class Yokoten(models.Model):
    """
    A distilled learning carried forward from one notebook to others.

    NB-001 §2.7. First-class object with adoption tracking.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    source_notebook = models.ForeignKey(
        Notebook,
        on_delete=models.CASCADE,
        related_name="yokoten",
    )
    source_trial = models.ForeignKey(
        Trial,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="yokoten",
    )
    learning = models.TextField()
    context = models.TextField(blank=True, default="", help_text="Conditions that made this work")
    applicable_to = models.JSONField(
        default=list,
        blank=True,
        help_text="Tags for matching: process types, defect categories, industries",
    )

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "core_yokoten"
        ordering = ["-created_at"]

    def __str__(self):
        return f"Yokoten: {self.learning[:80]}"


class YokotenAdoption(models.Model):
    """
    Tracks when a notebook adopts a yokoten learning.

    NB-001 §2.7. Records outcome to measure learning effectiveness.
    """

    class Outcome(models.TextChoices):
        HELPED = "helped", "Helped"
        NOT_APPLICABLE = "not_applicable", "Not Applicable"
        ADAPTED = "adapted", "Adapted"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    yokoten = models.ForeignKey(
        Yokoten,
        on_delete=models.CASCADE,
        related_name="adoptions",
    )
    target_notebook = models.ForeignKey(
        Notebook,
        on_delete=models.CASCADE,
        related_name="adopted_yokoten",
    )
    adopted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
    )
    adopted_at = models.DateTimeField(auto_now_add=True)
    outcome = models.CharField(
        max_length=20,
        choices=Outcome.choices,
        blank=True,
        default="",
    )
    notes = models.TextField(blank=True, default="")

    class Meta:
        db_table = "core_yokoten_adoption"
        unique_together = [("yokoten", "target_notebook")]

    def __str__(self):
        return f"{self.target_notebook.title} adopted: {self.yokoten.learning[:50]}"

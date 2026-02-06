"""Project model - the container for hypothesis-driven investigation.

Consolidates workbench.Project and agents_api.Problem into a single,
robust model that serves as the home for all decision science work.
"""

import uuid
from django.conf import settings
from django.db import models


class Project(models.Model):
    """A project is a container for hypothesis-driven investigation.

    Projects contain:
    - A problem/question being investigated
    - Hypotheses about potential causes/explanations
    - Evidence gathered from various sources
    - Links to knowledge graph entities

    Projects can be personal (user-owned) or shared (tenant-owned).
    """

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        ON_HOLD = "on_hold", "On Hold"
        RESOLVED = "resolved", "Resolved"
        ABANDONED = "abandoned", "Abandoned"

    class Methodology(models.TextChoices):
        """Problem-solving methodology being used."""
        NONE = "none", "General Investigation"
        DMAIC = "dmaic", "Six Sigma DMAIC"
        DOE = "doe", "Design of Experiments"
        PDCA = "pdca", "Plan-Do-Check-Act"
        A3 = "a3", "A3 Problem Solving"
        SCIENTIFIC = "scientific", "Scientific Method"

    class Phase(models.TextChoices):
        """Current phase of investigation."""
        # General phases
        DEFINE = "define", "Define Problem"
        HYPOTHESIZE = "hypothesize", "Generate Hypotheses"
        GATHER = "gather", "Gather Evidence"
        ANALYZE = "analyze", "Analyze"
        CONCLUDE = "conclude", "Conclude"

        # DMAIC-specific
        MEASURE = "measure", "Measure"
        IMPROVE = "improve", "Improve"
        CONTROL = "control", "Control"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Ownership: user OR tenant
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="core_projects",  # Avoid clash with workbench.Project during migration
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="core_projects",
    )

    # Basic info
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE,
    )

    # The problem/question being investigated
    problem_statement = models.TextField(
        blank=True,
        help_text="What are you trying to understand or solve?",
    )
    effect_description = models.TextField(
        blank=True,
        help_text="What effect or outcome are you observing?",
    )
    effect_magnitude = models.CharField(
        max_length=255,
        blank=True,
        help_text="How big is the effect? (e.g., '40% increase', '$50k loss')",
    )

    # Context
    domain = models.CharField(
        max_length=100,
        blank=True,
        help_text="Domain area (e.g., 'manufacturing', 'SaaS', 'healthcare')",
    )
    stakeholders = models.JSONField(
        default=list,
        blank=True,
        help_text="Who is affected or involved?",
    )
    constraints = models.JSONField(
        default=list,
        blank=True,
        help_text="Constraints on the investigation or solution",
    )
    available_data = models.TextField(
        blank=True,
        help_text="What data do you have access to?",
    )
    can_experiment = models.BooleanField(
        default=True,
        help_text="Can you run controlled experiments?",
    )

    # Methodology and phase tracking
    methodology = models.CharField(
        max_length=20,
        choices=Methodology.choices,
        default=Methodology.SCIENTIFIC,
    )
    current_phase = models.CharField(
        max_length=20,
        choices=Phase.choices,
        default=Phase.DEFINE,
    )
    phase_history = models.JSONField(
        default=list,
        blank=True,
        help_text="History of phase transitions [{phase, entered_at, notes}]",
    )

    # Interview/onboarding state (for guided setup)
    interview_state = models.JSONField(
        null=True,
        blank=True,
        help_text="Saved interview progress",
    )
    interview_completed = models.BooleanField(default=False)

    # Resolution
    resolution_summary = models.TextField(
        blank=True,
        help_text="What did we learn? What was the answer?",
    )
    resolution_confidence = models.FloatField(
        null=True,
        blank=True,
        help_text="Confidence in resolution (0.0 to 1.0)",
    )

    # Links to knowledge graph
    graph = models.ForeignKey(
        "core.KnowledgeGraph",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="projects",
        help_text="Knowledge graph this project references",
    )

    # Tags for organization
    tags = models.JSONField(default=list, blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    resolved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "core_project"
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["user", "status"]),
            models.Index(fields=["tenant", "status"]),
            models.Index(fields=["-updated_at"]),
        ]
        constraints = [
            models.CheckConstraint(
                check=(
                    models.Q(user__isnull=False, tenant__isnull=True) |
                    models.Q(user__isnull=True, tenant__isnull=False)
                ),
                name="project_has_single_owner",
            )
        ]

    def __str__(self):
        return self.title

    @property
    def owner(self):
        """Return the owner (user or tenant)."""
        return self.user or self.tenant

    @property
    def hypothesis_count(self) -> int:
        return self.hypotheses.count()

    @property
    def active_hypotheses(self):
        return self.hypotheses.filter(status="active")

    @property
    def evidence_count(self) -> int:
        return Evidence.objects.filter(hypothesis_links__hypothesis__project=self).distinct().count()

    def advance_phase(self, new_phase: str, notes: str = ""):
        """Advance to a new phase, recording history."""
        from django.utils import timezone

        self.phase_history.append({
            "phase": new_phase,
            "entered_at": timezone.now().isoformat(),
            "notes": notes,
        })
        self.current_phase = new_phase
        self.save(update_fields=["current_phase", "phase_history", "updated_at"])

    def resolve(self, summary: str, confidence: float = None):
        """Mark project as resolved."""
        from django.utils import timezone

        self.status = self.Status.RESOLVED
        self.resolution_summary = summary
        self.resolution_confidence = confidence
        self.resolved_at = timezone.now()
        self.save(update_fields=[
            "status", "resolution_summary", "resolution_confidence",
            "resolved_at", "updated_at"
        ])


# Import here to avoid circular import, but make Evidence available
from .hypothesis import Evidence


class Dataset(models.Model):
    """A dataset attached to a project.

    Datasets can be:
    - Uploaded CSV/Excel files
    - Results from experiments
    - External data sources
    """

    class DataType(models.TextChoices):
        CSV = "csv", "CSV"
        EXCEL = "excel", "Excel"
        JSON = "json", "JSON"
        EXPERIMENT = "experiment", "Experiment Results"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="datasets",
    )

    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    data_type = models.CharField(
        max_length=20,
        choices=DataType.choices,
        default=DataType.CSV,
    )

    # File storage
    file = models.FileField(
        upload_to="datasets/%Y/%m/",
        null=True,
        blank=True,
    )

    # Or inline data (for smaller datasets / experiment results)
    data = models.JSONField(
        null=True,
        blank=True,
        help_text="Inline data storage for smaller datasets",
    )

    # Schema information
    columns = models.JSONField(
        default=list,
        blank=True,
        help_text="Column definitions [{name, type, description}]",
    )
    row_count = models.IntegerField(default=0)

    # Link to experiment design (if this is experiment results)
    experiment_design = models.ForeignKey(
        "core.ExperimentDesign",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="result_datasets",
    )

    # Metadata
    source = models.CharField(max_length=255, blank=True)
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "core_dataset"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.project.title})"


class ExperimentDesign(models.Model):
    """A DOE design attached to a project.

    Stores the planned experiment design and can be compared
    against actual results for execution quality review.
    """

    class DesignType(models.TextChoices):
        FULL_FACTORIAL = "full_factorial", "Full Factorial"
        FRACTIONAL_FACTORIAL = "fractional_factorial", "Fractional Factorial"
        CCD = "ccd", "Central Composite"
        BOX_BEHNKEN = "box_behnken", "Box-Behnken"
        PLACKETT_BURMAN = "plackett_burman", "Plackett-Burman"
        DEFINITIVE_SCREENING = "definitive_screening", "Definitive Screening"
        TAGUCHI = "taguchi", "Taguchi"
        RCBD = "rcbd", "Randomized Block"
        LATIN_SQUARE = "latin_square", "Latin Square"
        CUSTOM = "custom", "Custom"

    class Status(models.TextChoices):
        PLANNED = "planned", "Planned"
        IN_PROGRESS = "in_progress", "In Progress"
        COMPLETED = "completed", "Completed"
        REVIEWED = "reviewed", "Reviewed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="experiment_designs",
    )

    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    design_type = models.CharField(
        max_length=30,
        choices=DesignType.choices,
        default=DesignType.FULL_FACTORIAL,
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PLANNED,
    )

    # The full design specification
    design_spec = models.JSONField(
        help_text="Full design specification from DOE generator",
    )

    # Factor definitions
    factors = models.JSONField(
        default=list,
        help_text="Factor definitions [{name, levels, units, categorical}]",
    )

    # Response variable(s)
    responses = models.JSONField(
        default=list,
        help_text="Response variables [{name, units, goal}]",
    )

    # Design properties
    num_runs = models.IntegerField(default=0)
    num_replicates = models.IntegerField(default=1)
    num_center_points = models.IntegerField(default=0)
    resolution = models.IntegerField(null=True, blank=True)

    # Execution review results
    execution_review = models.JSONField(
        null=True,
        blank=True,
        help_text="Results of design execution review",
    )
    execution_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Overall execution quality score (0-100)",
    )

    # Link to hypothesis being tested
    hypothesis = models.ForeignKey(
        "core.Hypothesis",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="experiment_designs",
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "core_experiment_design"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.design_type})"

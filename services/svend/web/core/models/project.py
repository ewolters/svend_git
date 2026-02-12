"""Project Charter model - structured for report assembly and logic.

The project charter captures all Define-phase information in a structured way:
- Problem Definition (5W2H): What, Where, When, Magnitude, Trend
- Business Impact: Financial, Customer, Safety, Quality, Regulatory
- Goal Statement: SMART format with metric, baseline, target
- Scope: In/Out, Constraints, Assumptions
- Team: Champion, Leader, Members
- Timeline: Phase, Target, Milestones

All fields are optional to allow incremental completion.
Lists use JSONField for multiple entries (e.g., multiple "whats").
"""

import uuid
from django.conf import settings
from django.db import models


class Project(models.Model):
    """A project charter for hypothesis-driven investigation.

    Structured to support:
    - CAPA report generation
    - 8D report generation
    - A3 report generation
    - AI-assisted problem analysis
    - Field-level validation and logic
    """

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        ON_HOLD = "on_hold", "On Hold"
        RESOLVED = "resolved", "Resolved"
        ABANDONED = "abandoned", "Abandoned"

    class Methodology(models.TextChoices):
        NONE = "none", "General Investigation"
        DMAIC = "dmaic", "Six Sigma DMAIC"
        DOE = "doe", "Design of Experiments"
        PDCA = "pdca", "Plan-Do-Check-Act"
        A3 = "a3", "A3 Problem Solving"
        SCIENTIFIC = "scientific", "Scientific Method"

    class Phase(models.TextChoices):
        DEFINE = "define", "Define"
        MEASURE = "measure", "Measure"
        ANALYZE = "analyze", "Analyze"
        IMPROVE = "improve", "Improve"
        CONTROL = "control", "Control"

    class Trend(models.TextChoices):
        INCREASING = "increasing", "Getting Worse"
        STABLE = "stable", "Stable"
        DECREASING = "decreasing", "Improving"
        UNKNOWN = "unknown", "Unknown"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Ownership
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="core_projects",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="core_projects",
    )

    # =========================================================================
    # HEADER
    # =========================================================================
    title = models.CharField(max_length=255)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE,
    )

    # =========================================================================
    # PROBLEM DEFINITION - 5W2H Style
    # Each can have multiple entries (user adds as many as relevant)
    # =========================================================================
    problem_whats = models.JSONField(
        default=list,
        blank=True,
        help_text="What is happening? What is the defect/issue? [list of statements]",
    )
    problem_wheres = models.JSONField(
        default=list,
        blank=True,
        help_text="Where is it occurring? Location, process step, product line [list]",
    )
    problem_whens = models.JSONField(
        default=list,
        blank=True,
        help_text="When does it happen? Time patterns, shifts, seasons [list]",
    )
    problem_magnitude = models.TextField(
        blank=True,
        help_text="How much? How often? Quantify the problem.",
    )
    problem_trend = models.CharField(
        max_length=20,
        choices=Trend.choices,
        default=Trend.UNKNOWN,
        help_text="Is the problem getting worse, stable, or improving?",
    )
    problem_since = models.CharField(
        max_length=255,
        blank=True,
        help_text="When did this problem start? First observed date/event.",
    )

    # Free-form problem statement (can be auto-generated from above)
    problem_statement = models.TextField(
        blank=True,
        help_text="Consolidated problem statement (can be generated from 5W2H fields)",
    )

    # =========================================================================
    # BUSINESS IMPACT - Separate fields for CAPA/8D mapping
    # =========================================================================
    impact_financial = models.TextField(
        blank=True,
        help_text="Financial impact: cost, revenue loss, scrap, rework",
    )
    impact_customer = models.TextField(
        blank=True,
        help_text="Customer impact: complaints, returns, satisfaction, churn",
    )
    impact_safety = models.TextField(
        blank=True,
        help_text="Safety impact: injuries, near-misses, risk level",
    )
    impact_quality = models.TextField(
        blank=True,
        help_text="Quality impact: defect rates, yield, specs out of tolerance",
    )
    impact_regulatory = models.TextField(
        blank=True,
        help_text="Regulatory/compliance impact: violations, audit findings",
    )
    impact_delivery = models.TextField(
        blank=True,
        help_text="Delivery impact: on-time delivery, lead time, backlog",
    )
    impact_other = models.JSONField(
        default=list,
        blank=True,
        help_text="Other impacts [list of {category, description}]",
    )

    # =========================================================================
    # GOAL STATEMENT - SMART Format
    # =========================================================================
    goal_statement = models.TextField(
        blank=True,
        help_text="Full goal statement (can be generated from fields below)",
    )
    goal_metric = models.CharField(
        max_length=255,
        blank=True,
        help_text="What metric are we improving? (e.g., 'First Pass Yield')",
    )
    goal_baseline = models.CharField(
        max_length=100,
        blank=True,
        help_text="Current baseline value (e.g., '87%')",
    )
    goal_target = models.CharField(
        max_length=100,
        blank=True,
        help_text="Target value (e.g., '95%')",
    )
    goal_unit = models.CharField(
        max_length=50,
        blank=True,
        help_text="Unit of measure (e.g., '%', 'ppm', 'hours')",
    )
    goal_deadline = models.DateField(
        null=True,
        blank=True,
        help_text="Target completion date",
    )

    # =========================================================================
    # SCOPE
    # =========================================================================
    scope_in = models.JSONField(
        default=list,
        blank=True,
        help_text="What is IN scope [list of items]",
    )
    scope_out = models.JSONField(
        default=list,
        blank=True,
        help_text="What is OUT of scope [list of items]",
    )
    constraints = models.JSONField(
        default=list,
        blank=True,
        help_text="Constraints and limitations [list]",
    )
    assumptions = models.JSONField(
        default=list,
        blank=True,
        help_text="Assumptions being made [list]",
    )

    # =========================================================================
    # TEAM
    # =========================================================================
    champion_name = models.CharField(
        max_length=255,
        blank=True,
        help_text="Executive sponsor name",
    )
    champion_title = models.CharField(
        max_length=255,
        blank=True,
        help_text="Executive sponsor title",
    )
    leader_name = models.CharField(
        max_length=255,
        blank=True,
        help_text="Project leader name",
    )
    leader_title = models.CharField(
        max_length=255,
        blank=True,
        help_text="Project leader title",
    )
    team_members = models.JSONField(
        default=list,
        blank=True,
        help_text="Team members [list of {name, role, department}]",
    )

    # =========================================================================
    # METHODOLOGY & PHASE TRACKING
    # =========================================================================
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
        help_text="Phase transitions [{phase, entered_at, notes}]",
    )

    # =========================================================================
    # TIMELINE & MILESTONES
    # =========================================================================
    target_completion = models.DateField(
        null=True,
        blank=True,
        help_text="Target project completion date",
    )
    milestones = models.JSONField(
        default=list,
        blank=True,
        help_text="Key milestones [{name, target_date, actual_date, status}]",
    )

    # =========================================================================
    # RESOLUTION
    # =========================================================================
    resolution_summary = models.TextField(
        blank=True,
        help_text="What was the root cause? What did we learn?",
    )
    resolution_actions = models.JSONField(
        default=list,
        blank=True,
        help_text="Actions taken [{action, owner, due_date, status}]",
    )
    resolution_verification = models.TextField(
        blank=True,
        help_text="How was the fix verified? Evidence of effectiveness.",
    )
    resolution_confidence = models.FloatField(
        null=True,
        blank=True,
        help_text="Confidence in resolution (0.0 to 1.0)",
    )
    resolved_at = models.DateTimeField(null=True, blank=True)

    # =========================================================================
    # CONTEXT & METADATA
    # =========================================================================
    domain = models.CharField(
        max_length=100,
        blank=True,
        help_text="Domain area (manufacturing, SaaS, healthcare, etc.)",
    )
    can_experiment = models.BooleanField(
        default=True,
        help_text="Can controlled experiments be run?",
    )
    tags = models.JSONField(default=list, blank=True)

    # Links to knowledge graph
    graph = models.ForeignKey(
        "core.KnowledgeGraph",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="projects",
    )

    # Synara belief engine state
    synara_state = models.JSONField(
        null=True,
        blank=True,
        help_text="Persisted Synara belief engine state",
    )

    # Interview/onboarding state
    interview_state = models.JSONField(null=True, blank=True)
    interview_completed = models.BooleanField(default=False)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

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
        from .hypothesis import Evidence
        return Evidence.objects.filter(hypothesis_links__hypothesis__project=self).distinct().count()

    def generate_problem_statement(self) -> str:
        """Generate a problem statement from 5W2H fields."""
        parts = []

        if self.problem_whats:
            parts.append("What: " + "; ".join(self.problem_whats))
        if self.problem_wheres:
            parts.append("Where: " + "; ".join(self.problem_wheres))
        if self.problem_whens:
            parts.append("When: " + "; ".join(self.problem_whens))
        if self.problem_magnitude:
            parts.append(f"Magnitude: {self.problem_magnitude}")
        if self.problem_trend != self.Trend.UNKNOWN:
            parts.append(f"Trend: {self.get_problem_trend_display()}")

        return "\n".join(parts)

    def generate_goal_statement(self) -> str:
        """Generate a SMART goal statement from goal fields."""
        if not self.goal_metric:
            return ""

        parts = [f"Improve {self.goal_metric}"]
        if self.goal_baseline and self.goal_target:
            parts.append(f"from {self.goal_baseline} to {self.goal_target}")
        elif self.goal_target:
            parts.append(f"to {self.goal_target}")
        if self.goal_unit:
            parts.append(self.goal_unit)
        if self.goal_deadline:
            parts.append(f"by {self.goal_deadline.strftime('%Y-%m-%d')}")

        return " ".join(parts)

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


class Dataset(models.Model):
    """A dataset attached to a project."""

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

    file = models.FileField(upload_to="datasets/%Y/%m/", null=True, blank=True)
    data = models.JSONField(null=True, blank=True)
    columns = models.JSONField(default=list, blank=True)
    row_count = models.IntegerField(default=0)

    experiment_design = models.ForeignKey(
        "core.ExperimentDesign",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="result_datasets",
    )

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
    """A DOE design attached to a project."""

    class DesignType(models.TextChoices):
        FULL_FACTORIAL = "full_factorial", "Full Factorial"
        FRACTIONAL_FACTORIAL = "fractional_factorial", "Fractional Factorial"
        CCD = "ccd", "Central Composite"
        BOX_BEHNKEN = "box_behnken", "Box-Behnken"
        PLACKETT_BURMAN = "plackett_burman", "Plackett-Burman"
        DEFINITIVE_SCREENING = "definitive_screening", "Definitive Screening"
        TAGUCHI = "taguchi", "Taguchi"
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

    design_spec = models.JSONField(default=dict)
    factors = models.JSONField(default=list)
    responses = models.JSONField(default=list)

    num_runs = models.IntegerField(default=0)
    num_replicates = models.IntegerField(default=1)
    num_center_points = models.IntegerField(default=0)
    resolution = models.IntegerField(null=True, blank=True)

    execution_review = models.JSONField(null=True, blank=True)
    execution_score = models.FloatField(null=True, blank=True)

    hypothesis = models.ForeignKey(
        "core.Hypothesis",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="experiment_designs",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "core_experiment_design"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.design_type})"

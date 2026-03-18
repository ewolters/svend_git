"""
Harada Method models — personal development system for CI practitioners.

Integrates into the Notebook UI as the personal operating system underneath
improvement campaigns. Six components:

1. Questionnaire (Harada 36 + CI Readiness 12) → archetype via clustering
2. Long-term Goals (cascade: 3-5yr → FY → month → immediate)
3. 64-Window (8 goals × 8 actions — tasks + routines)
4. Routine Tracker (daily binary check, notification hooks)
5. Daily Diary (time blocks, 8-dimension scoring, reflection)
6. Completion → Hansei Kai → new cycle

Schema locked per spec: mixed Likert + forced-choice, versioned responses,
scenario rotation, k-prototypes clustering support.
"""

import uuid

from django.conf import settings
from django.db import models

# =============================================================================
# QUESTIONNAIRE INFRASTRUCTURE
# =============================================================================


class QuestionDimension(models.Model):
    """A dimension measured by a questionnaire (Harada 36 or CI Readiness 12).

    Defines the structure: which instrument, what format, how many options.
    Questions themselves are stored as text on this model — the dimension IS
    the question (or scenario bank for forced-choice).
    """

    class Instrument(models.TextChoices):
        HARADA_36 = "harada_36", "Harada 36"
        CI_READINESS = "ci_readiness", "CI Readiness"

    class ResponseType(models.TextChoices):
        LIKERT = "likert", "Likert 1-5"
        FORCED_CHOICE = "forced_choice", "Forced Choice Scenario"

    class Category(models.TextChoices):
        # Harada categories
        THINKING = "thinking", "Thinking Faculty"
        ACTION = "action", "Action Taking"
        HUMAN = "human", "Human Talent"
        LEADERSHIP = "leadership", "Leadership"
        # CI Readiness (no sub-categories — all 12 are top-level)
        CI = "ci", "CI Readiness"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    instrument = models.CharField(max_length=20, choices=Instrument.choices)
    dimension_number = models.PositiveSmallIntegerField(
        help_text="1-36 for Harada, 1-12 for CI Readiness",
    )
    name = models.CharField(max_length=100, help_text="e.g., 'Preparation', 'Process vs. Person Attribution'")
    description = models.TextField(blank=True, default="")
    category = models.CharField(max_length=20, choices=Category.choices)
    response_type = models.CharField(max_length=20, choices=ResponseType.choices)

    # For Likert: the question text itself
    question_text = models.TextField(
        blank=True,
        default="",
        help_text="Likert: the question. Forced-choice: general description (scenarios in ScenarioBank).",
    )

    class Meta:
        db_table = "core_question_dimension"
        unique_together = [("instrument", "dimension_number")]
        ordering = ["instrument", "dimension_number"]

    def __str__(self):
        return f"[{self.instrument}] {self.dimension_number}. {self.name}"


class Scenario(models.Model):
    """A forced-choice scenario for a dimension. Multiple per dimension for rotation.

    Each scenario has 4 archetype options (not a gradient — no right answer).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dimension = models.ForeignKey(
        QuestionDimension,
        on_delete=models.CASCADE,
        related_name="scenarios",
    )
    scenario_key = models.CharField(
        max_length=20,
        help_text="Short ID for tracking (e.g., 'd1_s1', 'd1_s2')",
    )
    situation = models.TextField(help_text="The scenario description presented to the user")

    # 4 archetype options
    option_a = models.TextField(help_text="Archetype A response")
    option_a_label = models.CharField(max_length=50, help_text="Short label for clustering (e.g., 'system_thinker')")
    option_b = models.TextField()
    option_b_label = models.CharField(max_length=50)
    option_c = models.TextField()
    option_c_label = models.CharField(max_length=50)
    option_d = models.TextField()
    option_d_label = models.CharField(max_length=50)

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "core_scenario"
        unique_together = [("dimension", "scenario_key")]
        ordering = ["dimension", "scenario_key"]

    def __str__(self):
        return f"{self.dimension.name} — {self.scenario_key}"


class QuestionnaireResponse(models.Model):
    """A single response to a questionnaire dimension. Versioned for longitudinal tracking.

    Stores BOTH scenario_id + option_chosen (forced-choice) and score (Likert).
    One of these pairs is populated per response depending on response_type.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="questionnaire_responses",
    )
    dimension = models.ForeignKey(
        QuestionDimension,
        on_delete=models.CASCADE,
        related_name="responses",
    )
    instrument_version = models.PositiveSmallIntegerField(
        default=1,
        help_text="Version of the instrument (for retake comparison)",
    )
    session_id = models.UUIDField(
        default=uuid.uuid4,
        help_text="Groups responses from the same sitting",
    )

    # Likert response
    score = models.PositiveSmallIntegerField(
        null=True,
        blank=True,
        help_text="1-5 for Likert dimensions",
    )

    # Forced-choice response
    scenario = models.ForeignKey(
        Scenario,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="responses",
        help_text="Which scenario was presented (for retake analysis)",
    )
    option_chosen = models.CharField(
        max_length=50,
        blank=True,
        default="",
        help_text="Archetype label chosen (e.g., 'system_thinker')",
    )

    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "core_questionnaire_response"
        indexes = [
            models.Index(fields=["user", "dimension", "-timestamp"]),
            models.Index(fields=["user", "session_id"]),
        ]

    def __str__(self):
        if self.score is not None:
            return f"{self.user.email} → {self.dimension.name}: {self.score}"
        return f"{self.user.email} → {self.dimension.name}: {self.option_chosen}"


# =============================================================================
# LONG-TERM GOALS
# =============================================================================


class HaradaGoal(models.Model):
    """Cascading goal from the Harada Long-Term Goal Form.

    Hierarchy: 3-5yr → FY → Month → Immediate.
    Linked to a user, not a notebook — this is the personal layer.
    """

    class Horizon(models.TextChoices):
        LONG_TERM = "long_term", "3-5 Year"
        FISCAL_YEAR = "fiscal_year", "Current FY"
        MONTHLY = "monthly", "Current Month"
        IMMEDIATE = "immediate", "Immediate"

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        ACHIEVED = "achieved", "Achieved"
        REVISED = "revised", "Revised"
        ABANDONED = "abandoned", "Abandoned"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="harada_goals",
    )
    parent = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="children",
        help_text="Links FY → long-term, month → FY, immediate → month",
    )
    horizon = models.CharField(max_length=20, choices=Horizon.choices)
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True, default="")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.ACTIVE)

    # Service to others (Harada principle)
    service_at_home = models.TextField(blank=True, default="", help_text="How this goal serves others at home")
    service_at_work = models.TextField(blank=True, default="", help_text="How this goal serves others at work")

    # Four perspectives (tangible/intangible × self/others)
    perspectives = models.JSONField(
        default=dict,
        blank=True,
        help_text='{"tangible_self": [], "tangible_others": [], "intangible_self": [], "intangible_others": []}',
    )

    target_date = models.DateField(null=True, blank=True)
    achieved_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "core_harada_goal"
        ordering = ["horizon", "-created_at"]

    def __str__(self):
        return f"[{self.get_horizon_display()}] {self.title}"


# =============================================================================
# 64-WINDOW
# =============================================================================


class Window64(models.Model):
    """One cell in the 64-window mandala.

    8 core goals (center cells) each surrounded by 8 action cells.
    Actions are either one-time tasks or recurring routines.
    Total: 8 goals × (1 center + 8 surrounding) = 72 cells, but canonically
    the 8 center cells ARE the goals, so 8 × 8 = 64 action cells.
    """

    class CellType(models.TextChoices):
        GOAL = "goal", "Goal (center)"
        TASK = "task", "Task (one-time)"
        ROUTINE = "routine", "Routine (recurring)"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="window64_cells",
    )
    goal_number = models.PositiveSmallIntegerField(
        help_text="Which of the 8 goals this belongs to (1-8)",
    )
    position = models.PositiveSmallIntegerField(
        help_text="0 = center (goal), 1-8 = surrounding cells",
    )
    cell_type = models.CharField(max_length=10, choices=CellType.choices)
    text = models.CharField(max_length=300)
    is_completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)

    # Link to parent goal for context
    harada_goal = models.ForeignKey(
        HaradaGoal,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="window_cells",
        help_text="Optional link to the long-term goal this supports",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "core_window64"
        unique_together = [("user", "goal_number", "position")]
        ordering = ["goal_number", "position"]

    def __str__(self):
        return f"G{self.goal_number}.{self.position}: {self.text}"


# =============================================================================
# ROUTINE TRACKER
# =============================================================================


class RoutineCheck(models.Model):
    """Daily binary check for a routine from the 64-window.

    One record per routine per day. Binary: did you do it or not.
    Integrates with notification system for reminders.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="routine_checks",
    )
    window_cell = models.ForeignKey(
        Window64,
        on_delete=models.CASCADE,
        related_name="checks",
        help_text="The routine being tracked",
    )
    date = models.DateField()
    is_completed = models.BooleanField(default=False)
    notes = models.CharField(max_length=200, blank=True, default="")

    class Meta:
        db_table = "core_routine_check"
        unique_together = [("user", "window_cell", "date")]
        indexes = [
            models.Index(fields=["user", "date"]),
        ]

    def __str__(self):
        check = "+" if self.is_completed else "-"
        return f"[{check}] {self.window_cell.text} ({self.date})"


# =============================================================================
# DAILY DIARY
# =============================================================================


class DailyDiary(models.Model):
    """Daily diary entry — plan, reflect, score across 8 dimensions.

    One entry per day. Combines the Harada daily diary with Leadership
    Standardized Work (daily task management + routine tracking).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="daily_diaries",
    )
    date = models.DateField()
    daily_phrase = models.CharField(max_length=200, blank=True, default="", help_text="Motivational phrase for the day")

    # Time-blocked plan (JSON: [{"time": "06:00", "plan": "...", "reflection": "..."}, ...])
    time_blocks = models.JSONField(default=list, blank=True)

    # Top 5 tasks for the day
    top_tasks = models.JSONField(
        default=list,
        blank=True,
        help_text='[{"task": "...", "completed": bool}, ...]',
    )

    # 8-dimension self-scoring (0-5 each)
    score_overall = models.PositiveSmallIntegerField(null=True, blank=True)
    score_mental = models.PositiveSmallIntegerField(null=True, blank=True)
    score_body = models.PositiveSmallIntegerField(null=True, blank=True)
    score_work = models.PositiveSmallIntegerField(null=True, blank=True)
    score_relations = models.PositiveSmallIntegerField(null=True, blank=True)
    score_life = models.PositiveSmallIntegerField(null=True, blank=True)
    score_learning = models.PositiveSmallIntegerField(null=True, blank=True)
    score_routines = models.PositiveSmallIntegerField(null=True, blank=True)

    # Reflections
    challenges = models.TextField(blank=True, default="", help_text="Challenges and positive takeaways")
    what_differently = models.TextField(blank=True, default="", help_text="What would you have done differently?")
    notes = models.TextField(blank=True, default="", help_text="Additional notes / future actions")

    # Scores
    score_comments = models.JSONField(
        default=dict,
        blank=True,
        help_text='{"mental": "Anxiety", "body": "Tired", ...}',
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "core_daily_diary"
        unique_together = [("user", "date")]
        ordering = ["-date"]

    def __str__(self):
        return f"{self.user.email} — {self.date}"

    @property
    def total_score(self):
        """Sum of all 8 dimension scores (out of 40)."""
        scores = [
            self.score_overall,
            self.score_mental,
            self.score_body,
            self.score_work,
            self.score_relations,
            self.score_life,
            self.score_learning,
            self.score_routines,
        ]
        filled = [s for s in scores if s is not None]
        return sum(filled) if filled else None

    @property
    def tasks_completed(self):
        """Count of completed top tasks."""
        return sum(1 for t in (self.top_tasks or []) if t.get("completed"))


# =============================================================================
# ARCHETYPE ASSIGNMENT (k-prototypes clustering result)
# =============================================================================


class ArchetypeAssignment(models.Model):
    """A user's cluster assignment from k-prototypes on CI Readiness responses.

    Stored per session so longitudinal cluster migration is trackable.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="archetype_assignments",
    )
    session_id = models.UUIDField(help_text="Which questionnaire session produced this assignment")
    instrument_version = models.PositiveSmallIntegerField()

    cluster_id = models.PositiveSmallIntegerField(help_text="Cluster number (0-indexed)")
    cluster_label = models.CharField(
        max_length=100,
        blank=True,
        default="",
        help_text="Human-readable archetype label (assigned after cluster analysis)",
    )

    # The feature vector used for clustering (for reproducibility)
    feature_vector = models.JSONField(
        default=dict,
        help_text="{'likert': {dim: score}, 'categorical': {dim: label}}",
    )

    # Cluster metadata
    cluster_distances = models.JSONField(
        default=dict,
        blank=True,
        help_text="Distance to each cluster centroid",
    )
    silhouette_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Individual silhouette score for this assignment",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "core_archetype_assignment"
        indexes = [
            models.Index(fields=["user", "-created_at"]),
        ]

    def __str__(self):
        label = self.cluster_label or f"Cluster {self.cluster_id}"
        return f"{self.user.email} → {label} (v{self.instrument_version})"

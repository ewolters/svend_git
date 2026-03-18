"""Django admin configuration for core models."""

from django.contrib import admin

from .models import (
    DailyDiary,
    Dataset,
    Evidence,
    EvidenceLink,
    ExperimentDesign,
    HaradaGoal,
    Hypothesis,
    Membership,
    Notebook,
    NotebookPage,
    OrgInvitation,
    Project,
    QuestionDimension,
    QuestionnaireResponse,
    RoutineCheck,
    Scenario,
    StudentEnrollment,
    Tenant,
    TrainingCenter,
    TrainingProgram,
    Trial,
    Window64,
    Yokoten,
)


class MembershipInline(admin.TabularInline):
    model = Membership
    extra = 0
    fields = ["user", "role", "is_active", "joined_at"]
    readonly_fields = ["joined_at"]


class OrgInvitationInline(admin.TabularInline):
    model = OrgInvitation
    extra = 0
    fields = ["email", "role", "status", "invited_by", "created_at", "expires_at"]
    readonly_fields = ["token", "created_at"]


@admin.register(Tenant)
class TenantAdmin(admin.ModelAdmin):
    list_display = ["name", "slug", "plan", "member_count", "max_members", "is_active", "created_at"]
    list_filter = ["plan", "is_active"]
    search_fields = ["name", "slug"]
    readonly_fields = ["id", "created_at", "updated_at"]
    inlines = [MembershipInline, OrgInvitationInline]


@admin.register(Membership)
class MembershipAdmin(admin.ModelAdmin):
    list_display = ["user", "tenant", "role", "is_active", "joined_at"]
    list_filter = ["role", "is_active"]
    search_fields = ["user__email", "tenant__name"]
    readonly_fields = ["id", "invited_at"]


@admin.register(OrgInvitation)
class OrgInvitationAdmin(admin.ModelAdmin):
    list_display = ["email", "tenant", "role", "status", "invited_by", "created_at", "expires_at"]
    list_filter = ["status", "role"]
    search_fields = ["email", "tenant__name"]
    readonly_fields = ["id", "token", "created_at"]


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ["title", "user", "current_phase", "created_at"]
    list_filter = ["current_phase", "created_at"]
    search_fields = ["title", "description"]
    readonly_fields = ["id", "created_at", "updated_at"]


@admin.register(Hypothesis)
class HypothesisAdmin(admin.ModelAdmin):
    list_display = ["statement_preview", "project", "status", "current_probability", "created_at"]
    list_filter = ["status", "created_at"]
    search_fields = ["statement"]
    readonly_fields = ["id", "created_at", "updated_at"]

    def statement_preview(self, obj):
        return obj.statement[:50] + "..." if len(obj.statement) > 50 else obj.statement

    statement_preview.short_description = "Statement"


@admin.register(Evidence)
class EvidenceAdmin(admin.ModelAdmin):
    list_display = ["summary_preview", "project", "source_type", "confidence", "created_at"]
    list_filter = ["source_type", "result_type", "created_at"]
    search_fields = ["summary"]
    readonly_fields = ["id", "created_at"]

    def summary_preview(self, obj):
        return obj.summary[:50] + "..." if len(obj.summary) > 50 else obj.summary

    summary_preview.short_description = "Summary"


@admin.register(EvidenceLink)
class EvidenceLinkAdmin(admin.ModelAdmin):
    list_display = ["evidence", "hypothesis", "direction", "likelihood_ratio"]
    list_filter = ["direction"]


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ["name", "project", "row_count", "created_at"]
    search_fields = ["name"]
    readonly_fields = ["id", "created_at"]


@admin.register(ExperimentDesign)
class ExperimentDesignAdmin(admin.ModelAdmin):
    list_display = ["name", "project", "design_type", "status", "created_at"]
    list_filter = ["design_type", "status"]
    search_fields = ["name"]
    readonly_fields = ["id", "created_at", "updated_at"]


# ---------------------------------------------------------------------------
# Notebook models (NB-001)
# ---------------------------------------------------------------------------


class TrialInline(admin.TabularInline):
    model = Trial
    extra = 0
    fields = ["sequence", "title", "verdict", "before_value", "after_value", "is_adopted"]
    readonly_fields = ["sequence", "started_at"]


class NotebookPageInline(admin.TabularInline):
    model = NotebookPage
    extra = 0
    fields = ["sequence", "title", "page_type", "source_tool", "trial_role"]
    readonly_fields = ["created_at"]


@admin.register(Notebook)
class NotebookAdmin(admin.ModelAdmin):
    list_display = ["title", "project", "owner", "status", "baseline_value", "current_value", "created_at"]
    list_filter = ["status", "created_at"]
    search_fields = ["title", "owner__email"]
    readonly_fields = ["id", "created_at", "updated_at", "concluded_at"]
    inlines = [TrialInline, NotebookPageInline]


@admin.register(Trial)
class TrialAdmin(admin.ModelAdmin):
    list_display = ["title", "notebook", "sequence", "verdict", "before_value", "after_value", "is_adopted"]
    list_filter = ["verdict", "is_adopted"]
    search_fields = ["title", "notebook__title"]
    readonly_fields = ["id", "started_at", "completed_at"]


@admin.register(NotebookPage)
class NotebookPageAdmin(admin.ModelAdmin):
    list_display = ["title", "notebook", "page_type", "source_tool", "trial_role", "sequence", "created_at"]
    list_filter = ["page_type", "source_tool", "trial_role"]
    search_fields = ["title", "notebook__title"]
    readonly_fields = ["id", "created_at"]


@admin.register(Yokoten)
class YokotenAdmin(admin.ModelAdmin):
    list_display = ["learning_preview", "source_notebook", "created_by", "created_at"]
    search_fields = ["learning", "source_notebook__title"]
    readonly_fields = ["id", "created_at"]

    def learning_preview(self, obj):
        return obj.learning[:60] + "..." if len(obj.learning) > 60 else obj.learning

    learning_preview.short_description = "Learning"


# ---------------------------------------------------------------------------
# Training center models
# ---------------------------------------------------------------------------


class TrainingProgramInline(admin.TabularInline):
    model = TrainingProgram
    extra = 0
    fields = ["title", "status", "region", "start_date", "end_date"]


class StudentEnrollmentInline(admin.TabularInline):
    model = StudentEnrollment
    extra = 0
    fields = ["user", "status", "enrolled_at", "graduated_at", "conversion_deadline"]
    readonly_fields = ["enrolled_at"]


@admin.register(TrainingCenter)
class TrainingCenterAdmin(admin.ModelAdmin):
    list_display = ["name", "country", "contact_name", "is_ilssi_partner", "is_ngo", "created_at"]
    list_filter = ["is_ilssi_partner", "is_ngo", "country"]
    search_fields = ["name", "contact_name", "contact_email"]
    readonly_fields = ["id", "created_at", "updated_at"]
    inlines = [TrainingProgramInline]


@admin.register(TrainingProgram)
class TrainingProgramAdmin(admin.ModelAdmin):
    list_display = ["title", "center", "status", "region", "start_date", "end_date"]
    list_filter = ["status", "region"]
    search_fields = ["title", "center__name"]
    readonly_fields = ["id", "created_at", "updated_at"]
    inlines = [StudentEnrollmentInline]


@admin.register(StudentEnrollment)
class StudentEnrollmentAdmin(admin.ModelAdmin):
    list_display = ["user", "program", "status", "enrolled_at", "graduated_at", "conversion_deadline"]
    list_filter = ["status"]
    search_fields = ["user__email", "program__title"]
    readonly_fields = ["id", "enrolled_at"]


# ---------------------------------------------------------------------------
# Harada Method models
# ---------------------------------------------------------------------------


class ScenarioInline(admin.TabularInline):
    model = Scenario
    extra = 0
    fields = [
        "scenario_key",
        "situation",
        "option_a_label",
        "option_b_label",
        "option_c_label",
        "option_d_label",
        "is_active",
    ]


@admin.register(QuestionDimension)
class QuestionDimensionAdmin(admin.ModelAdmin):
    list_display = ["dimension_number", "instrument", "name", "category", "response_type"]
    list_filter = ["instrument", "category", "response_type"]
    ordering = ["instrument", "dimension_number"]
    inlines = [ScenarioInline]


@admin.register(QuestionnaireResponse)
class QuestionnaireResponseAdmin(admin.ModelAdmin):
    list_display = ["user", "dimension", "score", "option_chosen", "instrument_version", "timestamp"]
    list_filter = ["dimension__instrument", "instrument_version"]
    search_fields = ["user__email"]
    readonly_fields = ["id", "timestamp"]


@admin.register(HaradaGoal)
class HaradaGoalAdmin(admin.ModelAdmin):
    list_display = ["title", "user", "horizon", "status", "target_date"]
    list_filter = ["horizon", "status"]
    search_fields = ["title", "user__email"]
    readonly_fields = ["id", "created_at", "updated_at"]


@admin.register(Window64)
class Window64Admin(admin.ModelAdmin):
    list_display = ["text", "user", "goal_number", "position", "cell_type", "is_completed"]
    list_filter = ["cell_type", "is_completed", "goal_number"]
    search_fields = ["text", "user__email"]


@admin.register(RoutineCheck)
class RoutineCheckAdmin(admin.ModelAdmin):
    list_display = ["window_cell", "user", "date", "is_completed"]
    list_filter = ["is_completed", "date"]
    search_fields = ["user__email"]


@admin.register(DailyDiary)
class DailyDiaryAdmin(admin.ModelAdmin):
    list_display = ["user", "date", "total_score", "daily_phrase"]
    list_filter = ["date"]
    search_fields = ["user__email"]
    readonly_fields = ["id", "created_at", "updated_at"]

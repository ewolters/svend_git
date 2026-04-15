"""Admin configuration for Analysis Workbench models."""

from django.contrib import admin

from .models import AnalysisSession, SessionAnalysis, SessionDataset


class SessionDatasetInline(admin.TabularInline):
    model = SessionDataset
    extra = 0
    fields = ["name", "source", "row_count", "created_at"]
    readonly_fields = ["created_at"]


class SessionAnalysisInline(admin.TabularInline):
    model = SessionAnalysis
    extra = 0
    fields = ["analysis_type", "analysis_id", "dataset", "evidence_grade", "created_at"]
    readonly_fields = ["created_at"]


@admin.register(AnalysisSession)
class AnalysisSessionAdmin(admin.ModelAdmin):
    list_display = ["title", "user", "dataset_count", "analysis_count", "updated_at"]
    list_filter = ["created_at"]
    search_fields = ["title", "user__email"]
    readonly_fields = ["id", "created_at", "updated_at"]
    inlines = [SessionDatasetInline, SessionAnalysisInline]

    def dataset_count(self, obj):
        return obj.datasets.count()

    dataset_count.short_description = "Datasets"

    def analysis_count(self, obj):
        return obj.analyses.count()

    analysis_count.short_description = "Analyses"

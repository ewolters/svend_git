"""Admin configuration for Workbench models."""

from django.contrib import admin
from .models import Workbench, Artifact


class ArtifactInline(admin.TabularInline):
    """Inline display of artifacts in workbench admin."""
    model = Artifact
    extra = 0
    fields = ["artifact_type", "title", "source", "created_at"]
    readonly_fields = ["created_at"]
    show_change_link = True


@admin.register(Workbench)
class WorkbenchAdmin(admin.ModelAdmin):
    """Admin for Workbench model."""
    list_display = ["title", "user", "template", "status", "artifact_count", "updated_at"]
    list_filter = ["template", "status", "created_at"]
    search_fields = ["title", "description", "user__username", "user__email"]
    readonly_fields = ["id", "created_at", "updated_at"]
    inlines = [ArtifactInline]

    fieldsets = [
        (None, {
            "fields": ["id", "user", "title", "description", "template", "status"]
        }),
        ("Template State", {
            "fields": ["template_state"],
            "classes": ["collapse"],
        }),
        ("Canvas", {
            "fields": ["connections", "layout", "datasets"],
            "classes": ["collapse"],
        }),
        ("AI Guide", {
            "fields": ["guide_observations"],
            "classes": ["collapse"],
        }),
        ("Conclusion", {
            "fields": ["conclusion", "conclusion_confidence"],
            "classes": ["collapse"],
        }),
        ("Timestamps", {
            "fields": ["created_at", "updated_at"],
        }),
    ]

    def artifact_count(self, obj):
        return obj.artifacts.count()
    artifact_count.short_description = "Artifacts"


@admin.register(Artifact)
class ArtifactAdmin(admin.ModelAdmin):
    """Admin for Artifact model."""
    list_display = ["title_or_id", "artifact_type", "workbench", "source", "created_at"]
    list_filter = ["artifact_type", "source", "created_at"]
    search_fields = ["title", "workbench__title", "content"]
    readonly_fields = ["id", "created_at", "updated_at"]

    fieldsets = [
        (None, {
            "fields": ["id", "workbench", "artifact_type", "title", "source"]
        }),
        ("Content", {
            "fields": ["content"],
        }),
        ("Relationships", {
            "fields": ["source_artifact_id", "probability", "supports_hypotheses", "weakens_hypotheses"],
            "classes": ["collapse"],
        }),
        ("Model", {
            "fields": ["model_path"],
            "classes": ["collapse"],
        }),
        ("Metadata", {
            "fields": ["tags", "created_at", "updated_at"],
        }),
    ]

    def title_or_id(self, obj):
        return obj.title or str(obj.id)[:8]
    title_or_id.short_description = "Title"

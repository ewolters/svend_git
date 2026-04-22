from django.contrib import admin

from .models import Artifact, ArtifactSection, ToolTemplate
from .workflow_models import (
    SignalTypeRegistry,
    WorkflowPhase,
    WorkflowTemplate,
    WorkflowTransition,
)


@admin.register(ToolTemplate)
class ToolTemplateAdmin(admin.ModelAdmin):
    list_display = ("name", "slug", "version", "is_system", "tenant", "updated_at")
    list_filter = ("is_system",)
    search_fields = ("name", "slug")
    readonly_fields = ("id", "created_at", "updated_at")


class ArtifactSectionInline(admin.TabularInline):
    model = ArtifactSection
    extra = 0
    readonly_fields = ("id",)


@admin.register(Artifact)
class ArtifactAdmin(admin.ModelAdmin):
    list_display = ("title", "template", "status", "owner", "updated_at")
    list_filter = ("status", "template")
    search_fields = ("title",)
    readonly_fields = ("id", "created_at", "updated_at")
    inlines = [ArtifactSectionInline]


@admin.register(WorkflowTemplate)
class WorkflowTemplateAdmin(admin.ModelAdmin):
    list_display = ("name", "is_system", "is_active", "tenant")
    list_filter = ("is_system", "is_active")


@admin.register(WorkflowPhase)
class WorkflowPhaseAdmin(admin.ModelAdmin):
    list_display = ("label", "key", "workflow", "sort_order")
    list_filter = ("workflow",)


@admin.register(WorkflowTransition)
class WorkflowTransitionAdmin(admin.ModelAdmin):
    list_display = ("label", "workflow", "from_phase", "to_phase")
    list_filter = ("workflow",)


@admin.register(SignalTypeRegistry)
class SignalTypeRegistryAdmin(admin.ModelAdmin):
    list_display = ("key", "label", "default_severity", "is_system")
    list_filter = ("is_system",)

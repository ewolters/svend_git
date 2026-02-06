"""Admin configuration for chat models."""

from django.contrib import admin
from django.utils import timezone

from .models import (
    Conversation,
    Message,
    ModelVersion,
    SharedConversation,
    TraceLog,
    TrainingCandidate,
    UsageLog,
)


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    """Admin for hot-swapping model checkpoints."""

    list_display = ["model_type", "name", "is_active", "checkpoint_path", "activated_at"]
    list_filter = ["model_type", "is_active"]
    search_fields = ["name", "checkpoint_path"]
    readonly_fields = ["created_at", "activated_at"]
    actions = ["activate_selected"]

    @admin.action(description="Activate selected model version")
    def activate_selected(self, request, queryset):
        if queryset.count() != 1:
            self.message_user(request, "Select exactly one model to activate", level="error")
            return
        model = queryset.first()
        model.activate()
        self.message_user(request, f"Activated: {model.name}")


@admin.register(UsageLog)
class UsageLogAdmin(admin.ModelAdmin):
    list_display = ["user", "date", "request_count", "tokens_input", "tokens_output"]
    list_filter = ["date"]
    search_fields = ["user__username", "user__email"]
    date_hierarchy = "date"
    readonly_fields = ["user", "date"]


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ["title", "user", "message_count", "created_at", "updated_at"]
    list_filter = ["created_at"]
    search_fields = ["title", "user__username"]
    readonly_fields = ["created_at", "updated_at"]

    def message_count(self, obj):
        return obj.messages.count()


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ["role", "content_preview", "domain", "verified", "blocked", "created_at"]
    list_filter = ["role", "blocked", "verified", "domain"]
    search_fields = ["content"]
    readonly_fields = ["created_at"]

    def content_preview(self, obj):
        return obj.content[:80] + "..." if len(obj.content) > 80 else obj.content


@admin.register(SharedConversation)
class SharedConversationAdmin(admin.ModelAdmin):
    list_display = ["conversation", "view_count", "created_at", "expires_at"]
    readonly_fields = ["created_at"]


@admin.register(TraceLog)
class TraceLogAdmin(admin.ModelAdmin):
    list_display = ["domain", "safety_passed", "verified", "total_time_ms", "created_at"]
    list_filter = ["safety_passed", "verified", "gate_passed", "fallback_used", "domain"]
    search_fields = ["input_text"]
    date_hierarchy = "created_at"
    readonly_fields = ["created_at"]


@admin.register(TrainingCandidate)
class TrainingCandidateAdmin(admin.ModelAdmin):
    list_display = ["candidate_type", "status", "domain", "input_preview", "created_at"]
    list_filter = ["candidate_type", "status", "domain"]
    search_fields = ["input_text"]
    actions = ["approve_selected", "reject_selected", "export_approved"]

    def input_preview(self, obj):
        return obj.input_text[:60] + "..." if len(obj.input_text) > 60 else obj.input_text

    @admin.action(description="Approve selected for training")
    def approve_selected(self, request, queryset):
        count = queryset.update(status=TrainingCandidate.Status.APPROVED)
        self.message_user(request, f"Approved {count} candidates")

    @admin.action(description="Reject selected")
    def reject_selected(self, request, queryset):
        count = queryset.update(status=TrainingCandidate.Status.REJECTED)
        self.message_user(request, f"Rejected {count} candidates")

    @admin.action(description="Export approved to JSON")
    def export_approved(self, request, queryset):
        approved = queryset.filter(status=TrainingCandidate.Status.APPROVED)
        data = [c.to_training_format() for c in approved]
        approved.update(status=TrainingCandidate.Status.EXPORTED)
        self.message_user(request, f"Exported {len(data)} candidates (check TrainingCandidate.to_training_format())")

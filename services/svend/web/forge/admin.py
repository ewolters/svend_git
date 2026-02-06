"""Forge admin."""

from django.contrib import admin
from .models import APIKey, Job, UsageLog, SchemaTemplate


@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = ["name", "user", "key_prefix", "tier", "is_active", "created_at", "last_used_at"]
    list_filter = ["tier", "is_active"]
    search_fields = ["name", "user__email", "key_prefix"]
    readonly_fields = ["key_hash", "key_prefix", "created_at", "last_used_at"]


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ["job_id", "api_key", "data_type", "record_count", "status", "created_at"]
    list_filter = ["status", "data_type", "quality_level"]
    search_fields = ["job_id", "api_key__name"]
    readonly_fields = ["job_id", "created_at", "started_at", "completed_at"]
    date_hierarchy = "created_at"


@admin.register(UsageLog)
class UsageLogAdmin(admin.ModelAdmin):
    list_display = ["api_key", "job", "data_type", "record_count", "cost_cents", "created_at"]
    list_filter = ["data_type", "quality_level"]
    date_hierarchy = "created_at"


@admin.register(SchemaTemplate)
class SchemaTemplateAdmin(admin.ModelAdmin):
    list_display = ["name", "domain", "data_type", "is_builtin", "created_at"]
    list_filter = ["domain", "data_type", "is_builtin"]
    search_fields = ["name", "description"]

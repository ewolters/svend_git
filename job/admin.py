from django.contrib import admin

from job.models import Job, JobOutput


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ("id", "status", "actor", "canvas_id", "is_scratch", "created_at")
    list_filter = ("status", "is_scratch")
    readonly_fields = ("id", "correlation_id", "created_at", "updated_at")


@admin.register(JobOutput)
class JobOutputAdmin(admin.ModelAdmin):
    list_display = ("id", "job", "output_key", "output_type", "provenance", "created_at")
    list_filter = ("output_type", "provenance")
    readonly_fields = ("id", "correlation_id", "entry_hash", "previous_hash", "created_at")

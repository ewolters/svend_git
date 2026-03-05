"""
Django admin interface for audit system.

Provides read-only views of audit logs and integrity violations.

Compliance: SOC 2 CC7.2 / ISO 27001 A.12.7
"""

from django.contrib import admin
from django.utils.html import format_html

from syn.audit.models import DriftViolation, IntegrityViolation, SysLogEntry


@admin.register(SysLogEntry)
class SysLogEntryAdmin(admin.ModelAdmin):
    """
    Admin interface for audit log entries.

    Read-only to enforce immutability.
    """

    list_display = [
        "id",
        "timestamp",
        "tenant_id",
        "actor",
        "event_name",
        "hash_display",
        "is_genesis",
    ]

    list_filter = [
        "tenant_id",
        "event_name",
        "is_genesis",
        "timestamp",
    ]

    search_fields = [
        "tenant_id",
        "actor",
        "event_name",
        "correlation_id",
        "current_hash",
    ]

    readonly_fields = [
        "id",
        "timestamp",
        "actor",
        "event_name",
        "payload",
        "payload_hash",
        "correlation_id",
        "tenant_id",
        "previous_hash",
        "current_hash",
        "is_genesis",
        "hash_display",
        "chain_link_display",
    ]

    ordering = ["-id"]

    date_hierarchy = "timestamp"

    def has_add_permission(self, request):
        """Prevent adding entries via admin (use generate_entry())."""
        return False

    def has_change_permission(self, request, obj=None):
        """Prevent editing entries (immutability)."""
        return False

    def has_delete_permission(self, request, obj=None):
        """Prevent deleting entries (tamper-proof)."""
        return False

    def hash_display(self, obj):
        """Display abbreviated hash."""
        return format_html('<code style="font-size: 0.9em;">{}</code>', obj.current_hash[:16] + "...")

    hash_display.short_description = "Hash"

    def chain_link_display(self, obj):
        """Display chain linkage status."""
        is_valid = obj.verify_chain_link()
        if is_valid:
            return format_html('<span style="color: green;">✓ Valid</span>')
        else:
            return format_html('<span style="color: red;">✗ Broken</span>')

    chain_link_display.short_description = "Chain Link"


@admin.register(IntegrityViolation)
class IntegrityViolationAdmin(admin.ModelAdmin):
    """Admin interface for integrity violations."""

    list_display = [
        "id",
        "detected_at",
        "tenant_id",
        "violation_type",
        "entry_id",
        "is_resolved",
        "severity_display",
    ]

    list_filter = [
        "violation_type",
        "is_resolved",
        "detected_at",
        "tenant_id",
    ]

    search_fields = [
        "tenant_id",
        "entry_id",
        "details",
    ]

    readonly_fields = [
        "id",
        "detected_at",
        "tenant_id",
        "violation_type",
        "entry_id",
        "details",
    ]

    fieldsets = [
        (
            "Violation Details",
            {
                "fields": [
                    "id",
                    "detected_at",
                    "tenant_id",
                    "violation_type",
                    "entry_id",
                    "details",
                ]
            },
        ),
        (
            "Resolution",
            {
                "fields": [
                    "is_resolved",
                    "resolved_at",
                    "resolution_notes",
                ]
            },
        ),
    ]

    ordering = ["-detected_at"]

    date_hierarchy = "detected_at"

    def has_add_permission(self, request):
        """Violations are recorded automatically."""
        return False

    def has_delete_permission(self, request, obj=None):
        """Violations should not be deleted."""
        return request.user.is_superuser

    def severity_display(self, obj):
        """Display severity indicator."""
        if obj.is_resolved:
            return format_html('<span style="color: green;">Resolved</span>')
        else:
            return format_html('<span style="color: red; font-weight: bold;">⚠ Active</span>')

    severity_display.short_description = "Status"


@admin.register(DriftViolation)
class DriftViolationAdmin(admin.ModelAdmin):
    """Admin interface for drift violations."""

    list_display = [
        "id",
        "enforcement_check",
        "severity",
        "file_path",
        "line_number",
        "detected_at",
        "sla_status_display",
        "is_governance_escalated",
        "resolved_at",
    ]

    list_filter = [
        "enforcement_check",
        "severity",
        "is_sla_breached",
        "is_governance_escalated",
        "resolved_at",
        "detected_at",
        "is_remediation_available",
        "is_auto_fix_safe",
    ]

    search_fields = [
        "file_path",
        "function_name",
        "violation_message",
        "git_commit_sha",
        "git_author",
        "drift_signature",
    ]

    readonly_fields = [
        "id",
        "drift_signature",
        "detected_at",
        "detected_by",
        "correlation_id",
        "is_overdue_display",
    ]

    fieldsets = [
        (
            "Violation Details",
            {
                "fields": [
                    "id",
                    "drift_signature",
                    "enforcement_check",
                    "severity",
                    "violation_message",
                    "code_snippet",
                    "canonical_pattern",
                ]
            },
        ),
        (
            "Location",
            {
                "fields": [
                    "file_path",
                    "line_number",
                    "function_name",
                ]
            },
        ),
        (
            "Detection Metadata",
            {
                "fields": [
                    "detected_at",
                    "detected_by",
                    "git_commit_sha",
                    "git_author",
                ]
            },
        ),
        (
            "Remediation",
            {
                "fields": [
                    "remediation_available",
                    "auto_fix_safe",
                    "remediation_script",
                    "remediation_sla_hours",
                    "remediation_due_at",
                    "sla_breached",
                    "is_overdue_display",
                ]
            },
        ),
        (
            "Governance",
            {
                "fields": [
                    "governance_escalated",
                    "governance_rule_id",
                    "governance_judgment",
                ]
            },
        ),
        (
            "Correlation & Tracing",
            {
                "fields": [
                    "correlation_id",
                    "ctg_node_id",
                    "tenant_id",
                ]
            },
        ),
        (
            "Resolution",
            {
                "fields": [
                    "resolved_at",
                    "resolved_by",
                    "resolution_notes",
                ]
            },
        ),
    ]

    ordering = ["-detected_at"]

    date_hierarchy = "detected_at"

    def has_add_permission(self, request):
        """Violations are recorded automatically by drift detection."""
        return False

    def has_delete_permission(self, request, obj=None):
        """Violations should not be deleted for audit trail."""
        return request.user.is_superuser

    def sla_status_display(self, obj):
        """Display SLA status with color coding."""
        if obj.resolved_at:
            return format_html('<span style="color: green;">✓ Resolved</span>')
        elif obj.sla_breached or obj.is_overdue():
            return format_html('<span style="color: red; font-weight: bold;">⚠ SLA Breached</span>')
        elif obj.remediation_due_at:
            return format_html('<span style="color: orange;">⏱ Pending</span>')
        else:
            return format_html('<span style="color: gray;">No SLA</span>')

    sla_status_display.short_description = "SLA Status"

    def is_overdue_display(self, obj):
        """Display whether the violation is overdue."""
        if obj.is_overdue():
            return format_html('<span style="color: red; font-weight: bold;">Yes - Overdue!</span>')
        else:
            return format_html('<span style="color: green;">No</span>')

    is_overdue_display.short_description = "Overdue"

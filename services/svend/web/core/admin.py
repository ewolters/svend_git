from django.contrib import admin
from .models import (
    Tenant, Membership, OrgInvitation,
    Project, Hypothesis, Evidence, EvidenceLink, Dataset, ExperimentDesign,
)


class MembershipInline(admin.TabularInline):
    model = Membership
    extra = 0
    fields = ['user', 'role', 'is_active', 'joined_at']
    readonly_fields = ['joined_at']


class OrgInvitationInline(admin.TabularInline):
    model = OrgInvitation
    extra = 0
    fields = ['email', 'role', 'status', 'invited_by', 'created_at', 'expires_at']
    readonly_fields = ['token', 'created_at']


@admin.register(Tenant)
class TenantAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug', 'plan', 'member_count', 'max_members', 'is_active', 'created_at']
    list_filter = ['plan', 'is_active']
    search_fields = ['name', 'slug']
    readonly_fields = ['id', 'created_at', 'updated_at']
    inlines = [MembershipInline, OrgInvitationInline]


@admin.register(Membership)
class MembershipAdmin(admin.ModelAdmin):
    list_display = ['user', 'tenant', 'role', 'is_active', 'joined_at']
    list_filter = ['role', 'is_active']
    search_fields = ['user__email', 'tenant__name']
    readonly_fields = ['id', 'invited_at']


@admin.register(OrgInvitation)
class OrgInvitationAdmin(admin.ModelAdmin):
    list_display = ['email', 'tenant', 'role', 'status', 'invited_by', 'created_at', 'expires_at']
    list_filter = ['status', 'role']
    search_fields = ['email', 'tenant__name']
    readonly_fields = ['id', 'token', 'created_at']


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'current_phase', 'created_at']
    list_filter = ['current_phase', 'created_at']
    search_fields = ['title', 'description']
    readonly_fields = ['id', 'created_at', 'updated_at']


@admin.register(Hypothesis)
class HypothesisAdmin(admin.ModelAdmin):
    list_display = ['statement_preview', 'project', 'status', 'current_probability', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['statement']
    readonly_fields = ['id', 'created_at', 'updated_at']

    def statement_preview(self, obj):
        return obj.statement[:50] + '...' if len(obj.statement) > 50 else obj.statement
    statement_preview.short_description = 'Statement'


@admin.register(Evidence)
class EvidenceAdmin(admin.ModelAdmin):
    list_display = ['summary_preview', 'project', 'source_type', 'confidence', 'created_at']
    list_filter = ['source_type', 'result_type', 'created_at']
    search_fields = ['summary']
    readonly_fields = ['id', 'created_at']

    def summary_preview(self, obj):
        return obj.summary[:50] + '...' if len(obj.summary) > 50 else obj.summary
    summary_preview.short_description = 'Summary'


@admin.register(EvidenceLink)
class EvidenceLinkAdmin(admin.ModelAdmin):
    list_display = ['evidence', 'hypothesis', 'direction', 'likelihood_ratio']
    list_filter = ['direction']


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'project', 'row_count', 'created_at']
    search_fields = ['name']
    readonly_fields = ['id', 'created_at']


@admin.register(ExperimentDesign)
class ExperimentDesignAdmin(admin.ModelAdmin):
    list_display = ['name', 'project', 'design_type', 'status', 'created_at']
    list_filter = ['design_type', 'status']
    search_fields = ['name']
    readonly_fields = ['id', 'created_at', 'updated_at']

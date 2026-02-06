"""Admin configuration for accounts models."""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

from .models import InviteCode, Subscription, User


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ["username", "email", "tier", "queries_today", "is_active"]
    list_filter = ["tier", "is_active", "is_staff"]
    search_fields = ["username", "email"]
    ordering = ["-date_joined"]

    fieldsets = BaseUserAdmin.fieldsets + (
        ("Subscription", {"fields": ("tier", "stripe_customer_id")}),
        ("Rate Limiting", {"fields": ("queries_today", "queries_reset_at")}),
        ("Profile", {"fields": ("display_name", "avatar_url", "bio", "current_theme")}),
        ("Referrals", {"fields": ("referral_code", "referred_by")}),
        ("Analytics", {"fields": ("last_active_at", "total_queries", "total_tokens_used")}),
        ("Preferences", {"fields": ("preferences",)}),
    )


@admin.register(Subscription)
class SubscriptionAdmin(admin.ModelAdmin):
    list_display = ["user", "status", "current_period_end", "cancel_at_period_end"]
    list_filter = ["status", "cancel_at_period_end"]
    search_fields = ["user__username", "user__email", "stripe_subscription_id"]
    readonly_fields = ["stripe_subscription_id", "created_at", "updated_at"]


@admin.register(InviteCode)
class InviteCodeAdmin(admin.ModelAdmin):
    list_display = ["code", "times_used", "max_uses", "is_active", "note"]
    list_filter = ["is_active"]
    search_fields = ["code", "note"]
    actions = ["generate_codes"]

    @admin.action(description="Generate 5 new invite codes")
    def generate_codes(self, request, queryset):
        codes = InviteCode.generate(count=5, note="Admin generated")
        self.message_user(request, f"Generated: {', '.join(c.code for c in codes)}")

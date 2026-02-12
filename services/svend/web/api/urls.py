"""API URL configuration."""

from django.urls import path

from . import internal_views, views

app_name = "api"

urlpatterns = [
    # Health check
    path("health/", views.health, name="health"),

    # User
    path("user/", views.user_info, name="user_info"),

    # Conversations
    path("conversations/", views.conversations, name="conversations"),
    path("conversations/<uuid:conversation_id>/", views.conversation_detail, name="conversation_detail"),

    # Chat
    path("chat/", views.chat, name="chat"),

    # Sharing
    path("share/", views.share_conversation, name="share"),

    # Feedback / Training data collection
    path("messages/<uuid:message_id>/flag/", views.flag_message, name="flag_message"),

    # Monitoring
    path("stats/traces/", views.trace_stats, name="trace_stats"),
    path("stats/flywheel/", views.flywheel_stats, name="flywheel_stats"),

    # Auth
    path("auth/register/", views.register, name="register"),
    path("auth/login/", views.login, name="login"),
    path("auth/logout/", views.logout, name="logout"),
    path("auth/me/", views.me, name="me"),
    path("auth/profile/", views.update_profile, name="update_profile"),
    path("auth/password/", views.change_password, name="change_password"),
    path("auth/send-verification/", views.send_verification_email, name="send_verification"),
    path("auth/verify-email/", views.verify_email, name="verify_email"),

    # Export
    path("export/pdf/", views.export_pdf, name="export_pdf"),

    # Internal telemetry (staff-only)
    path("internal/overview/", internal_views.api_overview, name="internal_overview"),
    path("internal/users/", internal_views.api_users, name="internal_users"),
    path("internal/usage/", internal_views.api_usage, name="internal_usage"),
    path("internal/performance/", internal_views.api_performance, name="internal_performance"),
    path("internal/business/", internal_views.api_business, name="internal_business"),
    path("internal/insights/", internal_views.api_insights, name="internal_insights"),
]

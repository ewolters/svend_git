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
    path("auth/onboarding/", views.onboarding_status, name="onboarding_status"),
    path("auth/onboarding/complete/", views.onboarding_complete, name="onboarding_complete"),

    # Event tracking
    path("events/", views.track_event, name="track_event"),

    # Export
    path("export/pdf/", views.export_pdf, name="export_pdf"),

    # Internal telemetry (staff-only)
    path("internal/overview/", internal_views.api_overview, name="internal_overview"),
    path("internal/users/", internal_views.api_users, name="internal_users"),
    path("internal/usage/", internal_views.api_usage, name="internal_usage"),
    path("internal/performance/", internal_views.api_performance, name="internal_performance"),
    path("internal/business/", internal_views.api_business, name="internal_business"),
    path("internal/insights/", internal_views.api_insights, name="internal_insights"),
    path("internal/activity/", internal_views.api_activity, name="internal_activity"),
    path("internal/send-email/", internal_views.api_send_email, name="internal_send_email"),
    path("internal/email-draft/save/", internal_views.api_save_email_draft, name="internal_email_draft_save"),
    path("internal/email-draft/", internal_views.api_get_email_draft, name="internal_email_draft"),
    path("internal/email-campaigns/", internal_views.api_email_campaigns, name="internal_email_campaigns"),
    path("internal/onboarding/", internal_views.api_onboarding, name="internal_onboarding"),

    # Email tracking (public — hit by email clients)
    path("email/open/<uuid:recipient_id>/", views.email_track_open, name="email_track_open"),
    path("email/click/<uuid:recipient_id>/", views.email_track_click, name="email_track_click"),
    path("email/unsubscribe/", views.email_unsubscribe, name="email_unsubscribe"),

    # Feedback
    path("feedback/", views.submit_feedback, name="submit_feedback"),

    # Blog management (staff-only)
    path("internal/blog/", internal_views.api_blog_list, name="internal_blog_list"),
    path("internal/blog/save/", internal_views.api_blog_save, name="internal_blog_save"),
    path("internal/blog/generate/", internal_views.api_blog_generate, name="internal_blog_generate"),
    path("internal/blog/<uuid:post_id>/", internal_views.api_blog_get, name="internal_blog_get"),
    path("internal/blog/<uuid:post_id>/publish/", internal_views.api_blog_publish, name="internal_blog_publish"),
    path("internal/blog/<uuid:post_id>/delete/", internal_views.api_blog_delete, name="internal_blog_delete"),
    path("internal/blog/analytics/", internal_views.api_blog_analytics, name="internal_blog_analytics"),

    # Automation: Experiments
    path("internal/experiments/", internal_views.api_experiments, name="internal_experiments"),
    path("internal/experiments/<uuid:experiment_id>/evaluate/", internal_views.api_experiment_evaluate, name="internal_experiment_evaluate"),
    path("internal/experiments/<uuid:experiment_id>/conclude/", internal_views.api_experiment_conclude, name="internal_experiment_conclude"),

    # Automation: Rules
    path("internal/automation/rules/", internal_views.api_automation_rules, name="internal_automation_rules"),
    path("internal/automation/rules/<uuid:rule_id>/toggle/", internal_views.api_automation_rule_toggle, name="internal_automation_rule_toggle"),

    # Automation: Log
    path("internal/automation/log/", internal_views.api_automation_log, name="internal_automation_log"),

    # Automation: Autopilot
    path("internal/autopilot/", internal_views.api_autopilot, name="internal_autopilot"),
    path("internal/autopilot/<uuid:report_id>/approve/", internal_views.api_autopilot_approve, name="internal_autopilot_approve"),
    path("internal/autopilot/run/", internal_views.api_autopilot_run, name="internal_autopilot_run"),

    # Feedback (staff)
    path("internal/feedback/", internal_views.api_feedback, name="internal_feedback"),
]

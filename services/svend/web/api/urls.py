"""API URL configuration."""

from django.urls import path

from accounts import api_key_views

from . import internal_views, training_views, views

app_name = "api"

urlpatterns = [
    # Health check
    path("health/", views.health, name="health"),
    # User
    path("user/", views.user_info, name="user_info"),
    # Conversations
    path("conversations/", views.conversations, name="conversations"),
    path(
        "conversations/<uuid:conversation_id>/",
        views.conversation_detail,
        name="conversation_detail",
    ),
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
    path(
        "auth/send-verification/",
        views.send_verification_email,
        name="send_verification",
    ),
    path("auth/verify-email/", views.verify_email, name="verify_email"),
    path("auth/onboarding/", views.onboarding_status, name="onboarding_status"),
    path(
        "auth/onboarding/complete/",
        views.onboarding_complete,
        name="onboarding_complete",
    ),
    # API key management (SEC-001 §4.5)
    path("auth/keys/", api_key_views.key_list_create, name="api_key_list_create"),
    path("auth/keys/<uuid:key_id>/", api_key_views.key_revoke, name="api_key_revoke"),
    # Event tracking
    path("events/", views.track_event, name="track_event"),
    # Export
    path("export/pdf/", views.export_pdf, name="export_pdf"),
    # Internal telemetry (staff-only)
    path("internal/overview/", internal_views.api_overview, name="internal_overview"),
    path("internal/users/", internal_views.api_users, name="internal_users"),
    path(
        "internal/dsw-analytics/",
        internal_views.api_dsw_analytics,
        name="internal_dsw_analytics",
    ),
    path(
        "internal/hypothesis-health/",
        internal_views.api_hypothesis_health,
        name="internal_hypothesis_health",
    ),
    path("internal/anthropic/", internal_views.api_anthropic, name="internal_anthropic"),
    path(
        "internal/rate-limit-override/",
        internal_views.api_rate_limit_override,
        name="internal_rate_limit_override",
    ),
    path(
        "internal/performance/",
        internal_views.api_performance,
        name="internal_performance",
    ),
    path("internal/business/", internal_views.api_business, name="internal_business"),
    path("internal/insights/", internal_views.api_insights, name="internal_insights"),
    path("internal/activity/", internal_views.api_activity, name="internal_activity"),
    path(
        "internal/send-email/",
        internal_views.api_send_email,
        name="internal_send_email",
    ),
    path(
        "internal/email-preview/",
        internal_views.api_email_preview,
        name="internal_email_preview",
    ),
    path(
        "internal/cohort-retention/",
        internal_views.api_cohort_retention,
        name="internal_cohort_retention",
    ),
    path(
        "internal/email-draft/save/",
        internal_views.api_save_email_draft,
        name="internal_email_draft_save",
    ),
    path(
        "internal/email-draft/",
        internal_views.api_get_email_draft,
        name="internal_email_draft",
    ),
    path(
        "internal/email-campaigns/",
        internal_views.api_email_campaigns,
        name="internal_email_campaigns",
    ),
    path(
        "internal/onboarding/",
        internal_views.api_onboarding,
        name="internal_onboarding",
    ),
    # Email tracking (public — hit by email clients)
    path(
        "email/open/<uuid:recipient_id>/",
        views.email_track_open,
        name="email_track_open",
    ),
    path(
        "email/click/<uuid:recipient_id>/",
        views.email_track_click,
        name="email_track_click",
    ),
    path("email/unsubscribe/", views.email_unsubscribe, name="email_unsubscribe"),
    # Feedback
    path("feedback/", views.submit_feedback, name="submit_feedback"),
    # Blog management (staff-only)
    path("internal/blog/", internal_views.api_blog_list, name="internal_blog_list"),
    path("internal/blog/save/", internal_views.api_blog_save, name="internal_blog_save"),
    path(
        "internal/blog/generate/",
        internal_views.api_blog_generate,
        name="internal_blog_generate",
    ),
    path(
        "internal/blog/<uuid:post_id>/",
        internal_views.api_blog_get,
        name="internal_blog_get",
    ),
    path(
        "internal/blog/<uuid:post_id>/publish/",
        internal_views.api_blog_publish,
        name="internal_blog_publish",
    ),
    path(
        "internal/blog/<uuid:post_id>/delete/",
        internal_views.api_blog_delete,
        name="internal_blog_delete",
    ),
    path(
        "internal/blog/analytics/",
        internal_views.api_blog_analytics,
        name="internal_blog_analytics",
    ),
    # Whitepaper management (staff-only)
    path(
        "internal/whitepapers/",
        internal_views.api_whitepaper_list,
        name="internal_whitepaper_list",
    ),
    path(
        "internal/whitepapers/save/",
        internal_views.api_whitepaper_save,
        name="internal_whitepaper_save",
    ),
    path(
        "internal/whitepapers/<uuid:paper_id>/",
        internal_views.api_whitepaper_get,
        name="internal_whitepaper_get",
    ),
    path(
        "internal/whitepapers/<uuid:paper_id>/publish/",
        internal_views.api_whitepaper_publish,
        name="internal_whitepaper_publish",
    ),
    path(
        "internal/whitepapers/<uuid:paper_id>/delete/",
        internal_views.api_whitepaper_delete,
        name="internal_whitepaper_delete",
    ),
    path(
        "internal/whitepapers/analytics/",
        internal_views.api_whitepaper_analytics,
        name="internal_whitepaper_analytics",
    ),
    # Automation: Experiments
    path(
        "internal/experiments/",
        internal_views.api_experiments,
        name="internal_experiments",
    ),
    path(
        "internal/experiments/<uuid:experiment_id>/evaluate/",
        internal_views.api_experiment_evaluate,
        name="internal_experiment_evaluate",
    ),
    path(
        "internal/experiments/<uuid:experiment_id>/conclude/",
        internal_views.api_experiment_conclude,
        name="internal_experiment_conclude",
    ),
    # Automation: Rules
    path(
        "internal/automation/rules/",
        internal_views.api_automation_rules,
        name="internal_automation_rules",
    ),
    path(
        "internal/automation/rules/<uuid:rule_id>/toggle/",
        internal_views.api_automation_rule_toggle,
        name="internal_automation_rule_toggle",
    ),
    # Automation: Log
    path(
        "internal/automation/log/",
        internal_views.api_automation_log,
        name="internal_automation_log",
    ),
    # Automation: Autopilot
    path("internal/autopilot/", internal_views.api_autopilot, name="internal_autopilot"),
    path(
        "internal/autopilot/<uuid:report_id>/approve/",
        internal_views.api_autopilot_approve,
        name="internal_autopilot_approve",
    ),
    path(
        "internal/autopilot/run/",
        internal_views.api_autopilot_run,
        name="internal_autopilot_run",
    ),
    # Feedback (staff)
    path("internal/feedback/", internal_views.api_feedback, name="internal_feedback"),
    # CRM — Outbound Outreach
    path("internal/crm/leads/", internal_views.api_crm_leads, name="internal_crm_leads"),
    path(
        "internal/crm/leads/<uuid:lead_id>/",
        internal_views.api_crm_lead_delete,
        name="internal_crm_lead_delete",
    ),
    path(
        "internal/crm/leads/<uuid:lead_id>/stage/",
        internal_views.api_crm_lead_stage,
        name="internal_crm_lead_stage",
    ),
    path(
        "internal/crm/pipeline/",
        internal_views.api_crm_pipeline,
        name="internal_crm_pipeline",
    ),
    path(
        "internal/crm/sequences/",
        internal_views.api_crm_sequences,
        name="internal_crm_sequences",
    ),
    path(
        "internal/crm/sequences/<uuid:sequence_id>/",
        internal_views.api_crm_sequence_delete,
        name="internal_crm_sequence_delete",
    ),
    path(
        "internal/crm/sequences/<uuid:sequence_id>/enroll/",
        internal_views.api_crm_enroll,
        name="internal_crm_enroll",
    ),
    path(
        "internal/crm/generate-email/",
        internal_views.api_crm_generate_email,
        name="internal_crm_generate_email",
    ),
    path(
        "internal/crm/outreach-metrics/",
        internal_views.api_crm_outreach_metrics,
        name="internal_crm_outreach_metrics",
    ),
    path(
        "internal/crm/send-one/",
        internal_views.api_crm_send_one,
        name="internal_crm_send_one",
    ),
    path(
        "internal/crm/process-queue/",
        internal_views.api_crm_process_queue,
        name="internal_crm_process_queue",
    ),
    path(
        "internal/crm/bulk-send/",
        internal_views.api_crm_bulk_send,
        name="internal_crm_bulk_send",
    ),
    # Site analytics (staff-only)
    path(
        "internal/site-analytics/",
        internal_views.api_site_analytics,
        name="internal_site_analytics",
    ),
    path("internal/site-live/", internal_views.api_site_live, name="internal_site_live"),
    # Infrastructure (Synara OS layer, staff-only)
    path("internal/infra/", internal_views.api_infra, name="internal_infra"),
    path(
        "internal/audit-entries/",
        internal_views.api_audit_entries,
        name="internal_audit_entries",
    ),
    # Compliance (staff-only)
    path(
        "internal/compliance/",
        internal_views.api_compliance,
        name="internal_compliance",
    ),
    path(
        "internal/compliance/run/",
        internal_views.api_compliance_run,
        name="internal_compliance_run",
    ),
    path(
        "internal/compliance/publish/<uuid:report_id>/",
        internal_views.api_compliance_publish,
        name="internal_compliance_publish",
    ),
    # Risk Registry (staff-only, RISK-001)
    path("internal/risks/", internal_views.api_risk_registry, name="internal_risks"),
    # Change Management (staff-only, CHG-001)
    path(
        "internal/changes/",
        internal_views.api_change_management,
        name="internal_changes",
    ),
    path(
        "internal/changes/create/",
        internal_views.api_change_create,
        name="internal_change_create",
    ),
    path(
        "internal/changes/<uuid:change_id>/",
        internal_views.api_change_detail,
        name="internal_change_detail",
    ),
    path(
        "internal/changes/<uuid:change_id>/transition/",
        internal_views.api_change_transition,
        name="internal_change_transition",
    ),
    # Training Center Management (staff-only)
    path(
        "internal/training/centers/",
        training_views.list_create_centers,
        name="internal_training_centers",
    ),
    path(
        "internal/training/programs/",
        training_views.list_create_programs,
        name="internal_training_programs",
    ),
    path(
        "internal/training/enrollments/",
        training_views.list_enrollments,
        name="internal_training_enrollments",
    ),
    path(
        "internal/training/enroll/",
        training_views.batch_enroll,
        name="internal_training_enroll",
    ),
    path(
        "internal/training/graduate/",
        training_views.batch_graduate,
        name="internal_training_graduate",
    ),
    # Calibration (staff-only, CAL-001)
    path(
        "internal/calibration/",
        internal_views.api_calibration,
        name="internal_calibration",
    ),
    path(
        "internal/calibration/run/",
        internal_views.api_calibration_run,
        name="internal_calibration_run",
    ),
    # Incident Management (staff-only, INC-001)
    path(
        "internal/incidents/",
        internal_views.api_incident_list,
        name="internal_incidents",
    ),
    path(
        "internal/incidents/create/",
        internal_views.api_incident_create,
        name="internal_incident_create",
    ),
    path(
        "internal/incidents/<uuid:incident_id>/",
        internal_views.api_incident_detail,
        name="internal_incident_detail",
    ),
    path(
        "internal/incidents/<uuid:incident_id>/transition/",
        internal_views.api_incident_transition,
        name="internal_incident_transition",
    ),
    # Standards library
    path("internal/standards/", internal_views.api_standards, name="internal_standards"),
    # Roadmap management (staff-only)
    path(
        "internal/roadmap/",
        internal_views.api_roadmap_list,
        name="internal_roadmap_list",
    ),
    path(
        "internal/roadmap/save/",
        internal_views.api_roadmap_save,
        name="internal_roadmap_save",
    ),
    path(
        "internal/roadmap/<uuid:item_id>/",
        internal_views.api_roadmap_get,
        name="internal_roadmap_get",
    ),
    path(
        "internal/roadmap/<uuid:item_id>/delete/",
        internal_views.api_roadmap_delete,
        name="internal_roadmap_delete",
    ),
    # Plan documents (staff-only)
    path("internal/plans/", internal_views.api_plans_list, name="internal_plans_list"),
    path(
        "internal/plans/save/",
        internal_views.api_plans_save,
        name="internal_plans_save",
    ),
    path(
        "internal/plans/<uuid:plan_id>/",
        internal_views.api_plans_get,
        name="internal_plans_get",
    ),
    path(
        "internal/plans/<uuid:plan_id>/delete/",
        internal_views.api_plans_delete,
        name="internal_plans_delete",
    ),
    # Feature planning (staff-only)
    path(
        "internal/features/",
        internal_views.api_features_list,
        name="internal_features_list",
    ),
    path(
        "internal/features/<uuid:feature_id>/",
        internal_views.api_features_get,
        name="internal_features_get",
    ),
    path(
        "internal/features/<uuid:feature_id>/status/",
        internal_views.api_features_update_status,
        name="internal_features_status",
    ),
    path(
        "internal/features/<uuid:feature_id>/save/",
        internal_views.api_features_save,
        name="internal_features_save",
    ),
    path(
        "internal/features/<uuid:feature_id>/note/",
        internal_views.api_features_add_note,
        name="internal_features_note",
    ),
    path(
        "internal/tasks/<uuid:task_id>/status/",
        internal_views.api_tasks_update_status,
        name="internal_tasks_status",
    ),
    # Site duration beacon (public — fired by sendBeacon, no auth)
    path("site-duration/", views.site_duration, name="site_duration"),
    # Funnel events (public — pre-auth form interaction tracking)
    path("funnel-event/", views.funnel_event, name="funnel_event"),
]

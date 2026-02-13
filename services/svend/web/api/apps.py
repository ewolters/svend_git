"""API app configuration."""

from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api"

    def ready(self):
        # Register Tempora task handlers
        import api.tasks  # noqa: F401

        # Schedule recurring tasks (idempotent — skips if already exists)
        try:
            from tempora.models import Schedule
            from tempora.scheduler import schedule_task

            if not Schedule.objects.filter(schedule_id="publish_scheduled_posts").exists():
                schedule_task(
                    name="publish_scheduled_posts",
                    func="api.publish_scheduled_posts",
                    cron="*/15 * * * *",
                    priority=1,
                    queue="core",
                )

            if not Schedule.objects.filter(schedule_id="process_onboarding_drip").exists():
                schedule_task(
                    name="process_onboarding_drip",
                    func="api.process_onboarding_drip",
                    cron="*/10 * * * *",
                    priority=2,
                    queue="core",
                )

            # Automation engine: every 30 min
            if not Schedule.objects.filter(schedule_id="process_automations").exists():
                schedule_task(
                    name="process_automations",
                    func="api.process_automations",
                    cron="*/30 * * * *",
                    priority=2,
                    queue="core",
                )

            # Experiment evaluation: daily at 06:00 UTC
            if not Schedule.objects.filter(schedule_id="evaluate_experiments").exists():
                schedule_task(
                    name="evaluate_experiments",
                    func="api.evaluate_experiments",
                    cron="0 6 * * *",
                    priority=1,
                    queue="core",
                )

            # Claude growth review: weekly Sunday 20:00 UTC
            if not Schedule.objects.filter(schedule_id="claude_growth_review").exists():
                schedule_task(
                    name="claude_growth_review",
                    func="api.claude_growth_review",
                    cron="0 20 * * 0",
                    priority=1,
                    queue="core",
                )
        except Exception:
            pass  # Tempora tables may not exist yet during migrations

        # Seed default automation rules (idempotent)
        self._seed_automation_rules()

    def _seed_automation_rules(self):
        """Create built-in automation rules if they don't exist."""
        try:
            from api.models import AutomationRule

            defaults = [
                {
                    "name": "Activate new signup",
                    "description": "Send quick start guide to users who signed up but haven't run any queries",
                    "trigger": "signup_no_query",
                    "trigger_config": {"days": 3},
                    "action": "send_email",
                    "action_config": {"template": "activation"},
                    "cooldown_hours": 168,
                },
                {
                    "name": "Nudge inactive free",
                    "description": "Re-engage free users who haven't been active for 7 days",
                    "trigger": "inactive_days",
                    "trigger_config": {"days": 7},
                    "action": "send_email",
                    "action_config": {"template": "inactive_nudge"},
                    "cooldown_hours": 168,
                },
                {
                    "name": "Upgrade nudge",
                    "description": "Encourage upgrade when free users hit 80% of daily limit",
                    "trigger": "query_limit_near",
                    "trigger_config": {"threshold": 80},
                    "action": "send_email",
                    "action_config": {"template": "upgrade_nudge"},
                    "cooldown_hours": 72,
                },
                {
                    "name": "Churn prevention",
                    "description": "Reach out to users who have scheduled subscription cancellation",
                    "trigger": "churn_signal",
                    "trigger_config": {},
                    "action": "send_email",
                    "action_config": {"template": "churn_prevention"},
                    "cooldown_hours": 720,
                },
                {
                    "name": "Feature discovery: DOE",
                    "description": "Introduce DOE to paid users who haven't tried it",
                    "trigger": "feature_unused",
                    "trigger_config": {"feature": "doe", "days": 14},
                    "action": "send_email",
                    "action_config": {"template": "feature_discovery"},
                    "cooldown_hours": 336,
                },
                {
                    "name": "Milestone: 100 queries",
                    "description": "Celebrate when a user hits 100 total analyses",
                    "trigger": "milestone",
                    "trigger_config": {"count": 100},
                    "action": "send_email",
                    "action_config": {"template": "milestone"},
                    "cooldown_hours": 0,  # Never re-fire (handled by milestone logic)
                },
                {
                    "name": "Win-back",
                    "description": "Re-engage formerly paid users inactive for 45+ days",
                    "trigger": "inactive_days",
                    "trigger_config": {"days": 45, "was_paid": True},
                    "action": "send_email",
                    "action_config": {"template": "winback"},
                    "cooldown_hours": 720,
                },
            ]

            for rule_data in defaults:
                AutomationRule.objects.get_or_create(
                    name=rule_data["name"],
                    defaults=rule_data,
                )
        except Exception:
            pass  # Table may not exist yet

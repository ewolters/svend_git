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
        except Exception:
            pass  # Tempora tables may not exist yet during migrations

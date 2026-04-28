"""Initial migration for llm app.

State-only: registers model ownership. Tables already exist from agents_api.
"""

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.CreateModel(
                    name="LLMUsage",
                    fields=[
                        ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                        ("date", models.DateField(db_index=True)),
                        ("model", models.CharField(max_length=50)),
                        ("request_count", models.IntegerField(default=0)),
                        ("input_tokens", models.IntegerField(default=0)),
                        ("output_tokens", models.IntegerField(default=0)),
                        ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="llm_usage_records", to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        "db_table": "agents_api_llmusage",
                        "indexes": [models.Index(fields=["user", "date"], name="agents_api__user_id_e0d904_idx")],
                        "unique_together": {("user", "date", "model")},
                    },
                ),
                migrations.CreateModel(
                    name="RateLimitOverride",
                    fields=[
                        ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                        ("tier", models.CharField(max_length=20, unique=True)),
                        ("daily_llm_limit", models.PositiveIntegerField(help_text="Max LLM requests/day")),
                        ("daily_query_limit", models.PositiveIntegerField(help_text="Max query requests/day")),
                        ("updated_at", models.DateTimeField(auto_now=True)),
                        ("updated_by", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        "db_table": "agents_api_ratelimitoverride",
                    },
                ),
            ],
            database_operations=[],  # Tables already exist from agents_api
        ),
    ]

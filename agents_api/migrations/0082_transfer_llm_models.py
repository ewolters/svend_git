"""Transfer LLM models from agents_api to llm app.

State-only: removes model ownership from agents_api's migration state.
No database changes — the tables stay as-is, now owned by llm/.
"""

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        (
            "agents_api",
            "0081_alter_afe_fmea_remove_fmearow_fmea_alter_risk_fmea_and_more",
        ),
        ("llm", "0001_initial"),  # llm must exist first
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.RemoveField(
                    model_name="ratelimitoverride",
                    name="updated_by",
                ),
                migrations.DeleteModel(
                    name="LLMUsage",
                ),
                migrations.DeleteModel(
                    name="RateLimitOverride",
                ),
            ],
            database_operations=[],  # No DB changes — tables already exist, llm/ points at them
        ),
    ]

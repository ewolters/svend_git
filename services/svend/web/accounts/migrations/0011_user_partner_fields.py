"""Add training partner fields to User model."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0010_encrypt_existing_data"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="partner_code",
            field=models.CharField(
                blank=True,
                help_text="Training partner code (e.g., 'contiprove')",
                max_length=50,
            ),
        ),
        migrations.AddField(
            model_name="user",
            name="partner_discount_ends_at",
            field=models.DateTimeField(
                blank=True, help_text="When partner free access expires", null=True
            ),
        ),
    ]

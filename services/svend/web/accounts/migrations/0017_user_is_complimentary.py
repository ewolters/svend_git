from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0016_rename_was_successful_to_is_successful"),
    ]

    operations = [
        migrations.AddField(
            model_name="user",
            name="is_complimentary",
            field=models.BooleanField(
                default=False,
                help_text="Partner/sponsor account — full tier access, excluded from MRR",
            ),
        ),
    ]

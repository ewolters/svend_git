from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0017_training_center_models"),
    ]

    operations = [
        migrations.RenameField(
            model_name="trial",
            old_name="adopted",
            new_name="is_adopted",
        ),
        migrations.RenameField(
            model_name="hanseikai",
            old_name="carry_forward",
            new_name="is_carry_forward",
        ),
    ]

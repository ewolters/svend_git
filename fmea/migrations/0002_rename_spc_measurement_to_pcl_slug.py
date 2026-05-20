"""Rename spc_measurement → pcl_slug on FMEARow.

Zero-data-loss: Postgres RENAME COLUMN is a catalog-only operation.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("fmea", "0001_initial"),
    ]

    operations = [
        migrations.RenameField(
            model_name="fmearow",
            old_name="spc_measurement",
            new_name="pcl_slug",
        ),
        migrations.AlterField(
            model_name="fmearow",
            name="pcl_slug",
            field=models.CharField(
                max_length=255,
                blank=True,
                default="",
                help_text="Optional PCL measure slug — binds this row to a live measurement (e.g. wb/capability/diameter/cpk)",
            ),
        ),
    ]

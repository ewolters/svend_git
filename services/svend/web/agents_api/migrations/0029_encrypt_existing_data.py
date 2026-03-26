"""Encrypt existing plaintext data in agents_api models."""

from django.db import migrations


def encrypt_dsw_results(apps, schema_editor):
    DSWResult = apps.get_model("agents_api", "DSWResult")
    for result in DSWResult.objects.all():
        result.save()


def encrypt_triage_results(apps, schema_editor):
    TriageResult = apps.get_model("agents_api", "TriageResult")
    for result in TriageResult.objects.all():
        result.save()


class Migration(migrations.Migration):

    dependencies = [
        (
            "agents_api",
            "0028_alter_dswresult_data_alter_triageresult_cleaned_csv_and_more",
        ),
    ]

    operations = [
        migrations.RunPython(encrypt_dsw_results, migrations.RunPython.noop),
        migrations.RunPython(encrypt_triage_results, migrations.RunPython.noop),
    ]

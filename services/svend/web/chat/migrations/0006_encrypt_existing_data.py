"""Encrypt existing plaintext data in chat models.

Reads each row via ORM (from_db_value gracefully handles plaintext),
then saves (get_prep_value encrypts). Small DB so this is fast.
"""

from django.db import migrations


def encrypt_messages(apps, schema_editor):
    Message = apps.get_model("chat", "Message")
    for msg in Message.objects.all():
        msg.save()


def encrypt_tracelogs(apps, schema_editor):
    TraceLog = apps.get_model("chat", "TraceLog")
    for trace in TraceLog.objects.all():
        trace.save()


def encrypt_training_candidates(apps, schema_editor):
    TrainingCandidate = apps.get_model("chat", "TrainingCandidate")
    for tc in TrainingCandidate.objects.all():
        tc.save()


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0005_alter_message_content_alter_message_reasoning_trace_and_more"),
    ]

    operations = [
        migrations.RunPython(encrypt_messages, migrations.RunPython.noop),
        migrations.RunPython(encrypt_tracelogs, migrations.RunPython.noop),
        migrations.RunPython(encrypt_training_candidates, migrations.RunPython.noop),
    ]

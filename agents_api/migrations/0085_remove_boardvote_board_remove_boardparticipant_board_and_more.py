# State-only migration: release Board models to whiteboard app.
# Tables are NOT dropped — whiteboard takes ownership via its own migration.

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("agents_api", "0084_remove_dead_qms_models"),
        ("whiteboard", "0001_initial"),  # whiteboard must claim first
    ]

    # State-only: tell Django these models no longer belong to agents_api.
    # No database operations — tables stay as-is.
    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.RemoveField(model_name="boardvote", name="board"),
                migrations.RemoveField(model_name="boardparticipant", name="board"),
                migrations.RemoveField(model_name="boardguestinvite", name="board"),
                migrations.RemoveField(model_name="boardvote", name="guest_invite"),
                migrations.AlterUniqueTogether(name="boardparticipant", unique_together=None),
                migrations.RemoveField(model_name="boardparticipant", name="user"),
                migrations.RemoveField(model_name="boardvote", name="user"),
                migrations.DeleteModel(name="Board"),
                migrations.DeleteModel(name="BoardGuestInvite"),
                migrations.DeleteModel(name="BoardParticipant"),
                migrations.DeleteModel(name="BoardVote"),
            ],
            database_operations=[],
        ),
    ]

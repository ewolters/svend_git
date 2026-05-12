# State-only migration: claim Board models from agents_api.
# Tables already exist at agents_api_board etc — no DB changes needed.

import django.db.models.deletion
import uuid
import whiteboard.models
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("core", "0025_alter_notebook_baseline_analysis_and_more"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    # State-only: tell Django these models now belong to whiteboard.
    # Tables already exist from agents_api — no CREATE TABLE needed.
    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.CreateModel(
                    name="Board",
                    fields=[
                        ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                        ("room_code", models.CharField(db_index=True, default=whiteboard.models.generate_room_code, max_length=10, unique=True)),
                        ("name", models.CharField(default="Untitled Board", max_length=255)),
                        ("elements", models.JSONField(default=list)),
                        ("connections", models.JSONField(default=list)),
                        ("zoom", models.FloatField(default=1.0)),
                        ("pan_x", models.FloatField(default=0.0)),
                        ("pan_y", models.FloatField(default=0.0)),
                        ("is_voting_active", models.BooleanField(db_column="voting_active", default=False)),
                        ("votes_per_user", models.IntegerField(default=3)),
                        ("created_at", models.DateTimeField(auto_now_add=True)),
                        ("updated_at", models.DateTimeField(auto_now=True)),
                        ("version", models.IntegerField(default=0)),
                        ("owner", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="wb_owned_boards", to=settings.AUTH_USER_MODEL)),
                        ("project", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="wb_boards", to="core.project")),
                        ("tenant", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name="whiteboard_boards", to="core.tenant")),
                    ],
                    options={"db_table": "agents_api_board", "ordering": ["-updated_at"]},
                ),
                migrations.CreateModel(
                    name="BoardGuestInvite",
                    fields=[
                        ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                        ("token", models.CharField(db_index=True, max_length=64, unique=True)),
                        ("display_name", models.CharField(blank=True, max_length=100)),
                        ("permission", models.CharField(choices=[("view", "View Only"), ("edit", "Edit"), ("edit_vote", "Edit + Vote")], default="view", max_length=10)),
                        ("created_at", models.DateTimeField(auto_now_add=True)),
                        ("expires_at", models.DateTimeField()),
                        ("is_active", models.BooleanField(default=True)),
                        ("color", models.CharField(default="#ff7eb9", max_length=7)),
                        ("last_seen", models.DateTimeField(blank=True, null=True)),
                        ("cursor_x", models.FloatField(blank=True, null=True)),
                        ("cursor_y", models.FloatField(blank=True, null=True)),
                        ("board", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="wb_guest_invites", to="whiteboard.board")),
                    ],
                    options={"db_table": "agents_api_boardguestinvite"},
                ),
                migrations.CreateModel(
                    name="BoardParticipant",
                    fields=[
                        ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                        ("color", models.CharField(default="#4a9f6e", max_length=7)),
                        ("last_seen", models.DateTimeField(auto_now=True)),
                        ("cursor_x", models.FloatField(blank=True, null=True)),
                        ("cursor_y", models.FloatField(blank=True, null=True)),
                        ("board", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="wb_participants", to="whiteboard.board")),
                        ("user", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="wb_board_participations", to=settings.AUTH_USER_MODEL)),
                    ],
                    options={"db_table": "agents_api_boardparticipant"},
                ),
                migrations.CreateModel(
                    name="BoardVote",
                    fields=[
                        ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                        ("element_id", models.CharField(max_length=50)),
                        ("created_at", models.DateTimeField(auto_now_add=True)),
                        ("board", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="wb_votes", to="whiteboard.board")),
                        ("guest_invite", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name="wb_votes", to="whiteboard.boardguestinvite")),
                        ("user", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name="wb_board_votes", to=settings.AUTH_USER_MODEL)),
                    ],
                    options={"db_table": "agents_api_boardvote"},
                ),
                migrations.AddIndex(
                    model_name="boardguestinvite",
                    index=models.Index(fields=["board", "is_active"], name="wb_guest_board_active_idx"),
                ),
                migrations.AlterUniqueTogether(
                    name="boardparticipant",
                    unique_together={("board", "user")},
                ),
                migrations.AddConstraint(
                    model_name="boardvote",
                    constraint=models.UniqueConstraint(
                        condition=models.Q(("user__isnull", False)),
                        fields=("board", "user", "element_id"),
                        name="wb_unique_user_vote",
                    ),
                ),
                migrations.AddConstraint(
                    model_name="boardvote",
                    constraint=models.UniqueConstraint(
                        condition=models.Q(("guest_invite__isnull", False)),
                        fields=("board", "guest_invite", "element_id"),
                        name="wb_unique_guest_vote",
                    ),
                ),
            ],
            database_operations=[],
        ),
    ]

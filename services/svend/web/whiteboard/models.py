# whiteboard models — extracted from agents_api.models
# db_table overrides point at existing tables, no migration needed

import secrets
import uuid

from django.conf import settings
from django.db import models


def generate_room_code():
    """Generate a unique 6-character room code."""
    import string

    chars = string.ascii_uppercase + string.digits
    while True:
        code = "".join(secrets.choice(chars) for _ in range(6))
        if not Board.objects.filter(room_code=code).exists():
            return code


class Board(models.Model):
    """Collaborative whiteboard for kaizen sessions."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="whiteboard_boards",
    )
    room_code = models.CharField(max_length=10, unique=True, default=generate_room_code, db_index=True)
    name = models.CharField(max_length=255, default="Untitled Board")

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="wb_owned_boards",
    )

    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="wb_boards",
    )

    elements = models.JSONField(default=list)
    connections = models.JSONField(default=list)

    zoom = models.FloatField(default=1.0)
    pan_x = models.FloatField(default=0.0)
    pan_y = models.FloatField(default=0.0)

    is_voting_active = models.BooleanField(default=False, db_column="voting_active")
    votes_per_user = models.IntegerField(default=3)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    version = models.IntegerField(default=0)

    class Meta:
        db_table = "agents_api_board"
        managed = False
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.name} ({self.room_code})"

    def save(self, *args, **kwargs):
        from django.db.models import F

        is_new = self._state.adding
        super().save(*args, **kwargs)
        if not is_new:
            type(self).objects.filter(pk=self.pk).update(version=F("version") + 1)
            self.refresh_from_db(fields=["version"])


class BoardParticipant(models.Model):
    """Track who's in a board session."""

    board = models.ForeignKey(Board, on_delete=models.CASCADE, related_name="wb_participants")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="wb_board_participations",
    )
    color = models.CharField(max_length=7, default="#4a9f6e")
    last_seen = models.DateTimeField(auto_now=True)
    cursor_x = models.FloatField(null=True, blank=True)
    cursor_y = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "agents_api_boardparticipant"
        managed = False
        unique_together = ("board", "user")

    def __str__(self):
        return f"{self.user.username} in {self.board.room_code}"


class BoardVote(models.Model):
    """Dot vote on a board element."""

    board = models.ForeignKey(Board, on_delete=models.CASCADE, related_name="wb_votes")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="wb_board_votes",
    )
    guest_invite = models.ForeignKey(
        "BoardGuestInvite",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="wb_votes",
    )
    element_id = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "agents_api_boardvote"
        managed = False
        constraints = [
            models.UniqueConstraint(
                fields=["board", "user", "element_id"],
                condition=models.Q(user__isnull=False),
                name="wb_unique_user_vote",
            ),
            models.UniqueConstraint(
                fields=["board", "guest_invite", "element_id"],
                condition=models.Q(guest_invite__isnull=False),
                name="wb_unique_guest_vote",
            ),
        ]

    def __str__(self):
        if self.user:
            return f"{self.user.username} voted on {self.element_id}"
        return f"Guest voted on {self.element_id}"


class BoardGuestInvite(models.Model):
    """Guest invite for board access without a Svend account."""

    class Permission(models.TextChoices):
        VIEW = "view", "View Only"
        EDIT = "edit", "Edit"
        EDIT_VOTE = "edit_vote", "Edit + Vote"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    board = models.ForeignKey(Board, on_delete=models.CASCADE, related_name="wb_guest_invites")
    token = models.CharField(max_length=64, unique=True, db_index=True)
    display_name = models.CharField(max_length=100, blank=True)
    permission = models.CharField(
        max_length=10,
        choices=Permission.choices,
        default=Permission.VIEW,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    is_active = models.BooleanField(default=True)
    color = models.CharField(max_length=7, default="#ff7eb9")
    last_seen = models.DateTimeField(null=True, blank=True)
    cursor_x = models.FloatField(null=True, blank=True)
    cursor_y = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = "agents_api_boardguestinvite"
        managed = False
        indexes = [
            models.Index(fields=["board", "is_active"], name="wb_guest_board_active_idx"),
        ]

    def __str__(self):
        name = self.display_name or "(unnamed)"
        return f"Guest '{name}' on {self.board.room_code} ({self.permission})"

    @property
    def is_expired(self):
        from django.utils import timezone

        return timezone.now() > self.expires_at

    @property
    def is_valid(self):
        return self.is_active and not self.is_expired

    @staticmethod
    def generate_token():
        return secrets.token_urlsafe(32)

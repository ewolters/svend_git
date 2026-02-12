"""Multi-tenancy models for team and enterprise plans.

Tenants (Organizations) allow:
- Shared knowledge graphs across team members
- Collaborative projects
- Centralized billing and user management
"""

import uuid
from datetime import timedelta

from django.conf import settings
from django.db import models
from django.utils import timezone


class Tenant(models.Model):
    """An organization/team that shares resources.

    For TEAM and ENTERPRISE tiers, users belong to a tenant
    and share a knowledge graph.
    """

    class Plan(models.TextChoices):
        TEAM = "team", "Team"
        ENTERPRISE = "enterprise", "Enterprise"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=100, unique=True, db_index=True)
    plan = models.CharField(max_length=20, choices=Plan.choices, default=Plan.TEAM)

    # Billing
    stripe_customer_id = models.CharField(max_length=255, blank=True, db_index=True)

    # Settings
    settings = models.JSONField(default=dict, blank=True)

    # Limits
    max_members = models.IntegerField(default=10)
    max_projects = models.IntegerField(default=100)

    # Status
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "core_tenant"

    def __str__(self):
        return self.name

    @property
    def member_count(self) -> int:
        return self.memberships.filter(is_active=True).count()


class Membership(models.Model):
    """User membership in a tenant."""

    class Role(models.TextChoices):
        OWNER = "owner", "Owner"
        ADMIN = "admin", "Admin"
        MEMBER = "member", "Member"
        VIEWER = "viewer", "Viewer"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        Tenant,
        on_delete=models.CASCADE,
        related_name="memberships",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="memberships",
    )
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.MEMBER)

    # Status
    is_active = models.BooleanField(default=True)
    invited_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="invitations_sent",
    )
    invited_at = models.DateTimeField(auto_now_add=True)
    joined_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "core_membership"
        unique_together = [["tenant", "user"]]

    def __str__(self):
        return f"{self.user} @ {self.tenant} ({self.role})"

    @property
    def can_edit(self) -> bool:
        """Can this member edit projects and hypotheses?"""
        return self.role in (self.Role.OWNER, self.Role.ADMIN, self.Role.MEMBER)

    @property
    def can_admin(self) -> bool:
        """Can this member manage the tenant?"""
        return self.role in (self.Role.OWNER, self.Role.ADMIN)


def _invite_expiry():
    return timezone.now() + timedelta(days=7)


class OrgInvitation(models.Model):
    """Email invitation to join a tenant/organization.

    Created by owner/admin, sent to an email address.
    The recipient can accept via a unique token link.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        ACCEPTED = "accepted", "Accepted"
        EXPIRED = "expired", "Expired"
        CANCELLED = "cancelled", "Cancelled"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        Tenant,
        on_delete=models.CASCADE,
        related_name="invitations",
    )
    email = models.EmailField(help_text="Email address of the invitee")
    role = models.CharField(
        max_length=20,
        choices=Membership.Role.choices,
        default=Membership.Role.MEMBER,
    )
    token = models.UUIDField(default=uuid.uuid4, unique=True, db_index=True)
    invited_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="org_invitations_sent",
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(default=_invite_expiry)

    class Meta:
        db_table = "core_org_invitation"
        ordering = ["-created_at"]

    def __str__(self):
        return f"Invite {self.email} → {self.tenant.name} ({self.status})"

    @property
    def is_expired(self) -> bool:
        return timezone.now() > self.expires_at

    @property
    def is_valid(self) -> bool:
        return self.status == self.Status.PENDING and not self.is_expired

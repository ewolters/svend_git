"""Multi-tenancy models for team and enterprise plans.

Tenants (Organizations) allow:
- Shared knowledge graphs across team members
- Collaborative projects
- Centralized billing and user management
"""

import uuid
from django.conf import settings
from django.db import models


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

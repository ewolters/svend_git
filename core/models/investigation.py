"""
Investigation models — CANON-002 §7, §11.

Investigation is the structured problem-solving container that replaces
the implicit project-based pattern. It owns a Synara causal graph
(serialized as JSON), manages membership, and tracks tool linkage.

Reference: docs/standards/CANON-002.md §7.1.1, §7.2-7.4, §11.1
"""

import copy

from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models

from syn.core.base_models import SynaraEntity


class Investigation(SynaraEntity):
    """
    A structured problem-solving session (CANON-002 §7).

    The investigation graph lives in synara_state (JSON).
    Tools connect via InvestigationToolLink. Evidence flows through Synara.
    State machine: open → active → concluded → exported.
    """

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        ACTIVE = "active", "Active"
        CONCLUDED = "concluded", "Concluded"
        EXPORTED = "exported", "Exported"

    class MemberRole(models.TextChoices):
        OWNER = "owner", "Owner"
        CONTRIBUTOR = "contributor", "Contributor"
        VIEWER = "viewer", "Viewer"

    # Valid state transitions per CANON-002 §7.2
    VALID_TRANSITIONS = {
        "open": ["active"],
        "active": ["concluded"],
        "concluded": ["exported"],
        "exported": [],  # Terminal
    }

    title = models.CharField(max_length=300)
    description = models.TextField(blank=True, default="")
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.OPEN,
        db_index=True,
    )

    # Ownership
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="owned_investigations",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="investigations",
    )

    # Synara state — the causal graph serialized as JSON
    synara_state = models.JSONField(default=dict, blank=True)

    # Versioning (§7.3)
    version = models.PositiveIntegerField(default=1)
    parent_version = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="child_versions",
        help_text="Previous version this was reopened from",
    )

    # Layer 3 linkage (for export)
    exported_to_project = models.ForeignKey(
        "core.Project",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="source_investigations",
    )
    export_package = models.JSONField(
        null=True,
        blank=True,
        help_text="Conclusion package JSON (§9) frozen at export time",
    )

    # Timestamps (beyond SynaraEntity's created_at/updated_at)
    concluded_at = models.DateTimeField(null=True, blank=True)
    exported_at = models.DateTimeField(null=True, blank=True)

    # Members
    members = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        through="InvestigationMembership",
        related_name="investigations",
    )

    class Meta:
        db_table = "core_investigation"
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.title} (v{self.version}, {self.status})"

    def transition_to(self, target_status, user):
        """
        State machine per CANON-002 §7.2.
        Returns True on success, raises ValueError on invalid transition.
        """
        valid = self.VALID_TRANSITIONS.get(self.status, [])
        if target_status not in valid:
            raise ValueError(f"Cannot transition from {self.status} to {target_status}")

        self.status = target_status

        if target_status == self.Status.CONCLUDED:
            from django.utils import timezone

            self.concluded_at = timezone.now()

        if target_status == self.Status.EXPORTED:
            from django.utils import timezone

            self.exported_at = timezone.now()

        self.save()
        return True

    def reopen(self, user):
        """
        Create a new version from a concluded investigation (§7.3).
        Returns the new Investigation instance.
        """
        if self.status not in (self.Status.CONCLUDED, self.Status.EXPORTED):
            raise ValueError("Can only reopen concluded or exported investigations")

        new_inv = Investigation(
            title=self.title,
            description=self.description,
            status=self.Status.ACTIVE,
            owner=user,
            tenant=self.tenant,
            synara_state=copy.deepcopy(self.synara_state),
            version=self.version + 1,
            parent_version=self,
        )
        new_inv.save()

        # Copy membership
        for membership in self.investigationmembership_set.all():
            InvestigationMembership.objects.create(
                investigation=new_inv,
                user=membership.user,
                role=membership.role,
            )

        return new_inv


class InvestigationMembership(models.Model):
    """
    M2M through table for investigation membership (CANON-002 §7.4).

    Roles: owner, contributor, viewer.
    """

    import uuid as _uuid

    id = models.UUIDField(primary_key=True, default=_uuid.uuid4, editable=False)
    investigation = models.ForeignKey(Investigation, on_delete=models.CASCADE)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    role = models.CharField(
        max_length=20,
        choices=Investigation.MemberRole.choices,
        default=Investigation.MemberRole.CONTRIBUTOR,
    )
    joined_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "core_investigation_membership"
        unique_together = [("investigation", "user")]


class InvestigationToolLink(models.Model):
    """
    Links any tool output to an investigation via generic FK (CANON-002 §11.1).

    Avoids adding M2M to every tool model. Records tool_type and
    tool_function for querying and display.
    """

    import uuid as _uuid

    id = models.UUIDField(primary_key=True, default=_uuid.uuid4, editable=False)
    investigation = models.ForeignKey(
        Investigation,
        on_delete=models.CASCADE,
        related_name="tool_links",
    )

    # Generic FK to any tool model
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.UUIDField()
    tool_output = GenericForeignKey("content_type", "object_id")

    # Metadata
    tool_type = models.CharField(max_length=30)  # "spc", "rca", "ishikawa", etc.
    tool_function = models.CharField(
        max_length=20,
        choices=[
            ("information", "Information"),
            ("inference", "Inference"),
            ("intent", "Intent"),
            ("report", "Report"),
        ],
    )
    linked_at = models.DateTimeField(auto_now_add=True)
    linked_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    class Meta:
        db_table = "core_investigation_tool_link"
        unique_together = [("investigation", "content_type", "object_id")]

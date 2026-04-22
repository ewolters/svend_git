"""Composable QMS — primitive-based tool templates and artifact instances.

ToolTemplate defines a tool's schema (sections of primitives).
Artifact is a concrete instance of a template (the work product).
ArtifactSection holds primitive data for one section of an artifact.

Five primitives: text, grid, tree, checklist, action_list.

Architecture: plan spec — Composable QMS Primitives + Templates
"""

import uuid

from django.conf import settings
from django.db import models

from .schema import evaluate_computed_fields


class ToolTemplate(models.Model):
    """Defines a composable tool from primitives.

    System templates ship with Svend. Orgs can clone + customize
    or build from scratch.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="tool_templates",
        help_text="null = system template",
    )
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=100)
    description = models.TextField(blank=True)
    icon = models.CharField(max_length=50, blank=True)
    is_system = models.BooleanField(default=False)
    version = models.IntegerField(default=1)
    schema = models.JSONField(help_text="Section definitions: [{key, type, label, required, config}, ...]")
    status_flow = models.JSONField(
        default=list,
        blank=True,
        help_text='Ordered status list, e.g. ["draft", "active", "review", "complete"]',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "qms_tool_templates"
        constraints = [
            models.UniqueConstraint(
                fields=["tenant", "slug", "version"],
                name="unique_tool_template_per_tenant",
            ),
        ]
        ordering = ["name"]

    def __str__(self):
        suffix = f" v{self.version}" if self.version > 1 else ""
        return f"{self.name}{suffix}"

    def get_section_defs(self):
        """Return the sections list from the schema."""
        return self.schema.get("sections", [])


class Artifact(models.Model):
    """An instance of a template — the actual work product."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    template = models.ForeignKey(
        ToolTemplate,
        on_delete=models.PROTECT,
        related_name="artifacts",
    )
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="qms_artifacts",
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="qms_artifacts",
    )
    site = models.ForeignKey(
        "agents_api.Site",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="qms_artifacts",
    )
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="qms_artifacts",
    )
    title = models.CharField(max_length=300)
    status = models.CharField(max_length=50, default="draft")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "qms_artifacts"
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.template.name}: {self.title} ({self.status})"

    def to_dict(self):
        """Assemble all sections into a unified dict."""
        result = {
            "id": str(self.id),
            "template_id": str(self.template_id),
            "template_name": self.template.name,
            "template_slug": self.template.slug,
            "title": self.title,
            "status": self.status,
            "owner_id": self.owner_id,
            "site_id": str(self.site_id) if self.site_id else None,
            "project_id": str(self.project_id) if self.project_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "sections": {},
        }
        for section in self.sections.all():
            data = section.data
            # Evaluate computed fields for grid sections
            if section.primitive_type == "grid":
                section_def = self._get_section_def(section.section_key)
                if section_def:
                    data = evaluate_computed_fields(data, section_def.get("config", {}))
            result["sections"][section.section_key] = {
                "type": section.primitive_type,
                "data": data,
            }
        return result

    def to_manifest(self):
        """Pull contract manifest — lists available sections as sub-artifacts."""
        section_defs = {s["key"]: s for s in self.template.get_section_defs()}
        sections = self.sections.all()

        artifacts = []
        for section in sections:
            sdef = section_defs.get(section.section_key, {})
            artifacts.append(
                {
                    "id": str(self.id),
                    "type": section.primitive_type,
                    "key": section.section_key,
                    "label": sdef.get("label", section.section_key),
                }
            )

        return {
            "container_id": str(self.id),
            "container_type": "Artifact",
            "template": self.template.slug,
            "title": self.title,
            "status": self.status,
            "available_keys": [s.section_key for s in sections],
            "artifacts": artifacts,
            "updated_at": self.updated_at.isoformat(),
        }

    def get_sub_artifact(self, key_path):
        """Navigate to a section/field by key path.

        Examples:
            get_sub_artifact("background") -> section data
            get_sub_artifact("failure_modes/rows/0/severity") -> field value
        """
        parts = key_path.strip("/").split("/")
        section_key = parts[0]

        try:
            section = self.sections.get(section_key=section_key)
        except ArtifactSection.DoesNotExist:
            return None

        if len(parts) == 1:
            return section.data

        # Navigate into the data
        obj = section.data
        for part in parts[1:]:
            if isinstance(obj, dict):
                obj = obj.get(part)
            elif isinstance(obj, list):
                try:
                    obj = obj[int(part)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
            if obj is None:
                return None
        return obj

    def _get_section_def(self, section_key):
        """Look up a section definition from the template schema."""
        for sdef in self.template.get_section_defs():
            if sdef.get("key") == section_key:
                return sdef
        return None


class ArtifactSection(models.Model):
    """One section of an artifact — maps to a primitive type.

    Primitive types: text, grid, tree, checklist, action_list.
    """

    PRIMITIVE_CHOICES = [
        ("text", "Text"),
        ("grid", "Grid"),
        ("tree", "Tree"),
        ("checklist", "Checklist"),
        ("action_list", "Action List"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    artifact = models.ForeignKey(
        Artifact,
        on_delete=models.CASCADE,
        related_name="sections",
    )
    section_key = models.CharField(max_length=100)
    primitive_type = models.CharField(max_length=20, choices=PRIMITIVE_CHOICES)
    data = models.JSONField(default=dict)
    sort_order = models.IntegerField(default=0)

    class Meta:
        db_table = "qms_artifact_sections"
        unique_together = [("artifact", "section_key")]
        ordering = ["sort_order"]

    def __str__(self):
        return f"{self.artifact.title} / {self.section_key} ({self.primitive_type})"

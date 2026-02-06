"""Knowledge Graph models.

The knowledge graph stores entities and their relationships.
- Personal graphs for individual users
- Shared graphs for tenants (TEAM/ENTERPRISE)

Entities represent things in the domain (concepts, variables, actors, events, etc.)
Relationships connect entities with typed edges (causal, correlational, etc.)
"""

import uuid
from django.conf import settings
from django.db import models


class KnowledgeGraph(models.Model):
    """A knowledge graph belonging to a user or tenant.

    Each user has a personal graph. Tenants have a shared graph.
    Projects reference entities from the graph.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, default="Knowledge Graph")

    # Ownership: either user OR tenant, not both
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="knowledge_graph",
    )
    tenant = models.OneToOneField(
        "core.Tenant",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="knowledge_graph",
    )

    # Metadata
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "core_knowledge_graph"
        constraints = [
            models.CheckConstraint(
                check=(
                    models.Q(user__isnull=False, tenant__isnull=True) |
                    models.Q(user__isnull=True, tenant__isnull=False)
                ),
                name="graph_has_single_owner",
            )
        ]

    def __str__(self):
        owner = self.user or self.tenant
        return f"{self.name} ({owner})"

    @property
    def entity_count(self) -> int:
        return self.entities.count()

    @property
    def relationship_count(self) -> int:
        return self.relationships.count()


class Entity(models.Model):
    """A node in the knowledge graph.

    Entities represent things in the user's domain:
    - Concepts: abstract ideas, categories
    - Variables: measurable quantities (revenue, churn_rate)
    - Actors: people, teams, systems
    - Events: things that happened
    - DataSources: datasets, reports, APIs
    - Findings: conclusions from analysis
    """

    class EntityType(models.TextChoices):
        CONCEPT = "concept", "Concept"
        VARIABLE = "variable", "Variable"
        ACTOR = "actor", "Actor"
        EVENT = "event", "Event"
        DATA_SOURCE = "data_source", "Data Source"
        FINDING = "finding", "Finding"
        CUSTOM = "custom", "Custom"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    graph = models.ForeignKey(
        KnowledgeGraph,
        on_delete=models.CASCADE,
        related_name="entities",
    )

    # Identity
    name = models.CharField(max_length=255)
    entity_type = models.CharField(
        max_length=20,
        choices=EntityType.choices,
        default=EntityType.CONCEPT,
    )
    description = models.TextField(blank=True)

    # For custom entity types
    custom_type = models.CharField(max_length=100, blank=True)

    # Properties (flexible key-value storage)
    properties = models.JSONField(default=dict, blank=True)

    # For variables: track units and typical range
    unit = models.CharField(max_length=50, blank=True)  # e.g., "USD", "percent", "count"
    typical_min = models.FloatField(null=True, blank=True)
    typical_max = models.FloatField(null=True, blank=True)

    # For events: track when it happened
    occurred_at = models.DateTimeField(null=True, blank=True)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_entities",
    )

    class Meta:
        db_table = "core_entity"
        verbose_name_plural = "entities"
        indexes = [
            models.Index(fields=["graph", "entity_type"]),
            models.Index(fields=["graph", "name"]),
        ]

    def __str__(self):
        return f"{self.name} ({self.entity_type})"


class Relationship(models.Model):
    """An edge in the knowledge graph connecting two entities.

    Relationships have types that define their semantics:
    - Causal: X causes Y (with optional strength/confidence)
    - Correlational: X correlates with Y
    - Hierarchical: X is part of Y, X is a type of Y
    - Temporal: X happens before/after/during Y
    - Associative: X is related to Y (general)
    """

    class RelationType(models.TextChoices):
        # Causal
        CAUSES = "causes", "causes"
        PREVENTS = "prevents", "prevents"
        ENABLES = "enables", "enables"
        INFLUENCES = "influences", "influences"

        # Correlational
        CORRELATES_WITH = "correlates_with", "correlates with"
        INVERSELY_CORRELATES = "inversely_correlates", "inversely correlates with"

        # Hierarchical
        IS_PART_OF = "is_part_of", "is part of"
        CONTAINS = "contains", "contains"
        IS_TYPE_OF = "is_type_of", "is a type of"
        HAS_TYPE = "has_type", "has type"

        # Temporal
        HAPPENS_BEFORE = "happens_before", "happens before"
        HAPPENS_AFTER = "happens_after", "happens after"
        HAPPENS_DURING = "happens_during", "happens during"
        TRIGGERS = "triggers", "triggers"

        # Associative
        RELATED_TO = "related_to", "related to"
        DEPENDS_ON = "depends_on", "depends on"
        CONTRADICTS = "contradicts", "contradicts"
        SUPPORTS = "supports", "supports"

        # Custom
        CUSTOM = "custom", "custom"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    graph = models.ForeignKey(
        KnowledgeGraph,
        on_delete=models.CASCADE,
        related_name="relationships",
    )

    # The edge: source -> relation_type -> target
    source = models.ForeignKey(
        Entity,
        on_delete=models.CASCADE,
        related_name="outgoing_relationships",
    )
    target = models.ForeignKey(
        Entity,
        on_delete=models.CASCADE,
        related_name="incoming_relationships",
    )
    relation_type = models.CharField(
        max_length=30,
        choices=RelationType.choices,
        default=RelationType.RELATED_TO,
    )

    # For custom relationship types
    custom_type = models.CharField(max_length=100, blank=True)

    # Strength/confidence of the relationship
    strength = models.FloatField(
        default=1.0,
        help_text="Strength of relationship (0.0 to 1.0)",
    )
    confidence = models.FloatField(
        default=1.0,
        help_text="Confidence in this relationship existing (0.0 to 1.0)",
    )

    # For causal relationships: estimated effect size
    effect_size = models.FloatField(
        null=True,
        blank=True,
        help_text="Estimated causal effect size",
    )

    # Evidence supporting this relationship
    evidence_summary = models.TextField(blank=True)
    evidence_count = models.IntegerField(default=0)

    # Properties (flexible)
    properties = models.JSONField(default=dict, blank=True)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_relationships",
    )

    class Meta:
        db_table = "core_relationship"
        indexes = [
            models.Index(fields=["graph", "relation_type"]),
            models.Index(fields=["source", "relation_type"]),
            models.Index(fields=["target", "relation_type"]),
        ]

    def __str__(self):
        return f"{self.source.name} {self.relation_type} {self.target.name}"

    @property
    def is_causal(self) -> bool:
        return self.relation_type in (
            self.RelationType.CAUSES,
            self.RelationType.PREVENTS,
            self.RelationType.ENABLES,
            self.RelationType.INFLUENCES,
        )

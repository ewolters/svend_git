"""Serializers for core models."""

from rest_framework import serializers
from .models import (
    Tenant, Membership, KnowledgeGraph, Entity, Relationship,
    Project, Dataset, ExperimentDesign, Hypothesis, Evidence, EvidenceLink,
)


# =============================================================================
# Tenant & Membership
# =============================================================================

class MembershipSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    email = serializers.CharField(source="user.email", read_only=True)

    class Meta:
        model = Membership
        fields = ["id", "user", "username", "email", "role", "is_active", "joined_at"]
        read_only_fields = ["id", "joined_at"]


class TenantSerializer(serializers.ModelSerializer):
    member_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Tenant
        fields = ["id", "name", "slug", "plan", "member_count", "is_active", "created_at"]
        read_only_fields = ["id", "created_at"]


# =============================================================================
# Knowledge Graph
# =============================================================================

class EntitySerializer(serializers.ModelSerializer):
    class Meta:
        model = Entity
        fields = [
            "id", "name", "entity_type", "custom_type", "description",
            "properties", "unit", "typical_min", "typical_max",
            "occurred_at", "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class RelationshipSerializer(serializers.ModelSerializer):
    source_name = serializers.CharField(source="source.name", read_only=True)
    target_name = serializers.CharField(source="target.name", read_only=True)

    class Meta:
        model = Relationship
        fields = [
            "id", "source", "source_name", "target", "target_name",
            "relation_type", "custom_type", "strength", "confidence",
            "effect_size", "evidence_summary", "properties", "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class KnowledgeGraphSerializer(serializers.ModelSerializer):
    entity_count = serializers.IntegerField(read_only=True)
    relationship_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = KnowledgeGraph
        fields = ["id", "name", "description", "entity_count", "relationship_count", "created_at"]
        read_only_fields = ["id", "created_at"]


class KnowledgeGraphDetailSerializer(KnowledgeGraphSerializer):
    entities = EntitySerializer(many=True, read_only=True)
    relationships = RelationshipSerializer(many=True, read_only=True)

    class Meta(KnowledgeGraphSerializer.Meta):
        fields = KnowledgeGraphSerializer.Meta.fields + ["entities", "relationships"]


# =============================================================================
# Evidence
# =============================================================================

class EvidenceSerializer(serializers.ModelSerializer):
    hypothesis_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Evidence
        fields = [
            "id", "summary", "details", "source_type", "source_description",
            "result_type", "confidence",
            # Statistical fields
            "p_value", "confidence_interval_low", "confidence_interval_high",
            "effect_size", "sample_size", "statistical_test",
            # Quantitative fields
            "measured_value", "expected_value", "unit",
            # Raw data
            "raw_output",
            # Reproducibility
            "is_reproducible", "code_reference", "data_reference",
            # Metadata
            "hypothesis_count", "created_at",
        ]
        read_only_fields = ["id", "created_at", "hypothesis_count"]


class EvidenceLinkSerializer(serializers.ModelSerializer):
    evidence_summary = serializers.CharField(source="evidence.summary", read_only=True)
    evidence_source = serializers.CharField(source="evidence.source_type", read_only=True)
    strength_description = serializers.CharField(source="strength", read_only=True)

    class Meta:
        model = EvidenceLink
        fields = [
            "id", "hypothesis", "evidence", "evidence_summary", "evidence_source",
            "likelihood_ratio", "direction", "reasoning",
            "strength_description", "is_manual", "created_at", "applied_at",
        ]
        read_only_fields = ["id", "direction", "created_at", "applied_at"]


# =============================================================================
# Hypothesis
# =============================================================================

class HypothesisSerializer(serializers.ModelSerializer):
    evidence_count = serializers.IntegerField(read_only=True)
    odds = serializers.FloatField(read_only=True)

    class Meta:
        model = Hypothesis
        fields = [
            "id", "statement",
            # Structured If/Then/Because
            "if_clause", "then_clause", "because_clause",
            # Variables
            "independent_variable", "independent_var_values",
            "dependent_variable", "dependent_var_unit",
            "predicted_direction", "predicted_magnitude",
            # Rationale & Testing
            "rationale", "test_method", "success_criteria", "data_requirements",
            # Probability
            "prior_probability", "current_probability", "probability_history",
            "status", "confirmation_threshold", "rejection_threshold",
            "is_testable", "test_suggestions",
            "evidence_count", "odds", "created_at", "updated_at",
        ]
        read_only_fields = ["id", "current_probability", "probability_history", "odds", "created_at", "updated_at"]


class HypothesisDetailSerializer(HypothesisSerializer):
    evidence_links = EvidenceLinkSerializer(many=True, read_only=True)
    supporting_evidence = EvidenceLinkSerializer(many=True, read_only=True)
    opposing_evidence = EvidenceLinkSerializer(many=True, read_only=True)

    class Meta(HypothesisSerializer.Meta):
        fields = HypothesisSerializer.Meta.fields + [
            "evidence_links", "supporting_evidence", "opposing_evidence",
        ]


# =============================================================================
# Dataset & Experiment Design
# =============================================================================

class DatasetSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()
    size_display = serializers.SerializerMethodField()

    class Meta:
        model = Dataset
        fields = [
            "id", "name", "description", "data_type",
            "file", "file_url", "data", "columns", "row_count",
            "experiment_design", "source",
            "created_at", "updated_at", "size_display",
        ]
        read_only_fields = ["id", "file_url", "created_at", "updated_at", "size_display"]

    def get_file_url(self, obj):
        if obj.file:
            request = self.context.get("request")
            if request:
                return request.build_absolute_uri(obj.file.url)
            return obj.file.url
        return None

    def get_size_display(self, obj):
        if obj.file:
            size = obj.file.size
            if size < 1024:
                return f"{size} B"
            elif size < 1024 * 1024:
                return f"{size / 1024:.1f} KB"
            else:
                return f"{size / (1024 * 1024):.1f} MB"
        elif obj.data:
            return f"{obj.row_count} rows"
        return "Empty"


class ExperimentDesignSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExperimentDesign
        fields = [
            "id", "name", "description", "design_type", "status",
            "design_spec", "factors", "responses",
            "num_runs", "num_replicates", "num_center_points", "resolution",
            "execution_review", "execution_score",
            "hypothesis", "created_at", "updated_at", "completed_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class ExperimentDesignDetailSerializer(ExperimentDesignSerializer):
    result_datasets = DatasetSerializer(many=True, read_only=True)
    hypothesis_statement = serializers.CharField(source="hypothesis.statement", read_only=True)

    class Meta(ExperimentDesignSerializer.Meta):
        fields = ExperimentDesignSerializer.Meta.fields + [
            "result_datasets", "hypothesis_statement",
        ]


# =============================================================================
# Project
# =============================================================================

class ProjectListSerializer(serializers.ModelSerializer):
    hypothesis_count = serializers.IntegerField(read_only=True)
    evidence_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Project
        fields = [
            "id", "title", "problem_statement", "status", "domain",
            "methodology", "current_phase",
            "hypothesis_count", "evidence_count",
            "created_at", "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class ProjectDetailSerializer(serializers.ModelSerializer):
    hypotheses = HypothesisSerializer(many=True, read_only=True)
    datasets = DatasetSerializer(many=True, read_only=True)
    experiment_designs = ExperimentDesignSerializer(many=True, read_only=True)
    hypothesis_count = serializers.IntegerField(read_only=True)
    dataset_count = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = [
            "id", "title", "status", "domain",
            # Problem Definition (5W2H)
            "problem_statement",
            "problem_whats", "problem_wheres", "problem_whens",
            "problem_magnitude", "problem_trend", "problem_since",
            # Business Impact
            "impact_financial", "impact_customer", "impact_safety",
            "impact_quality", "impact_regulatory", "impact_delivery", "impact_other",
            # Goal Statement (SMART)
            "goal_statement", "goal_metric", "goal_baseline", "goal_target", "goal_unit", "goal_deadline",
            # Scope
            "scope_in", "scope_out", "constraints", "assumptions",
            # Team
            "champion_name", "champion_title", "leader_name", "leader_title", "team_members",
            # Timeline
            "milestones", "target_completion",
            # Methodology
            "methodology", "current_phase", "phase_history", "can_experiment",
            # Interview
            "interview_state", "interview_completed",
            # Resolution
            "resolution_summary", "resolution_actions", "resolution_verification", "resolution_confidence",
            # Metadata
            "tags", "synara_state",
            # Related
            "hypothesis_count", "hypotheses", "datasets", "dataset_count", "experiment_designs",
            "created_at", "updated_at", "resolved_at",
        ]
        read_only_fields = ["id", "phase_history", "created_at", "updated_at", "resolved_at"]

    def get_dataset_count(self, obj):
        return obj.datasets.count()


# =============================================================================
# Evidence Creation (for Coder/DSW integration)
# =============================================================================

class CreateEvidenceFromCodeSerializer(serializers.Serializer):
    """Create evidence from code execution results."""
    project_id = serializers.UUIDField()
    hypothesis_ids = serializers.ListField(
        child=serializers.UUIDField(),
        required=False,
        allow_empty=True,
    )

    # Evidence details
    summary = serializers.CharField(max_length=500)
    details = serializers.CharField(required=False, allow_blank=True)
    source_type = serializers.ChoiceField(
        choices=Evidence.SourceType.choices,
        default=Evidence.SourceType.SIMULATION,
    )

    # Code and output
    code = serializers.CharField(required=False, allow_blank=True)
    output = serializers.JSONField(required=False)

    # Optional statistical data
    p_value = serializers.FloatField(required=False, allow_null=True)
    effect_size = serializers.FloatField(required=False, allow_null=True)
    sample_size = serializers.IntegerField(required=False, allow_null=True)
    confidence = serializers.FloatField(default=0.8)

    # Likelihood ratios for each hypothesis
    likelihood_ratios = serializers.DictField(
        child=serializers.FloatField(),
        required=False,
        help_text="Map of hypothesis_id -> likelihood_ratio",
    )


class CreateEvidenceFromAnalysisSerializer(serializers.Serializer):
    """Create evidence from DSW analysis results."""
    project_id = serializers.UUIDField()
    hypothesis_ids = serializers.ListField(
        child=serializers.UUIDField(),
        required=False,
        allow_empty=True,
    )

    # Evidence details
    summary = serializers.CharField(max_length=500)
    analysis_type = serializers.CharField(max_length=100)  # "regression", "classification", "correlation", etc.

    # Analysis results
    results = serializers.JSONField()

    # Statistical data (extracted from results)
    metrics = serializers.JSONField(required=False)  # accuracy, f1, r2, etc.

    confidence = serializers.FloatField(default=0.8)
    likelihood_ratios = serializers.DictField(
        child=serializers.FloatField(),
        required=False,
    )

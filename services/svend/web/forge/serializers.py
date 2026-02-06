"""Forge API serializers."""

from rest_framework import serializers
from .models import Job, APIKey, SchemaTemplate, DataType, QualityLevel


class FieldConstraintSerializer(serializers.Serializer):
    """Constraints for a schema field."""
    min = serializers.FloatField(required=False)
    max = serializers.FloatField(required=False)
    values = serializers.ListField(child=serializers.CharField(), required=False)
    pattern = serializers.CharField(required=False)


class FieldDefinitionSerializer(serializers.Serializer):
    """Definition for a single schema field."""
    type = serializers.ChoiceField(choices=[
        "string", "int", "float", "bool", "category",
        "email", "uuid", "datetime", "date", "url", "phone"
    ])
    constraints = FieldConstraintSerializer(required=False)
    nullable = serializers.BooleanField(default=False)
    generator = serializers.CharField(required=False)  # Custom generator name


class SchemaDefinitionSerializer(serializers.Serializer):
    """Schema definition as key-value pairs."""
    # Dynamic fields handled in view


class GenerateRequestSerializer(serializers.Serializer):
    """Request to generate synthetic data."""
    data_type = serializers.ChoiceField(choices=DataType.choices, default=DataType.TABULAR)
    domain = serializers.CharField(max_length=100, required=False, default="")
    record_count = serializers.IntegerField(min_value=1, max_value=1000000)
    schema = serializers.DictField(required=False)  # Custom schema
    template = serializers.CharField(max_length=100, required=False)  # Or use template
    quality_level = serializers.ChoiceField(choices=QualityLevel.choices, default=QualityLevel.STANDARD)
    output_format = serializers.ChoiceField(
        choices=["json", "jsonl", "csv"],
        default="jsonl"
    )

    def validate(self, data):
        if not data.get("schema") and not data.get("template"):
            raise serializers.ValidationError(
                "Either 'schema' or 'template' must be provided"
            )
        return data


class JobSerializer(serializers.ModelSerializer):
    """Job status and results."""

    class Meta:
        model = Job
        fields = [
            "job_id", "data_type", "domain", "record_count",
            "quality_level", "output_format", "status", "progress",
            "records_generated", "quality_score", "cost_cents",
            "error_message", "created_at", "started_at", "completed_at"
        ]
        read_only_fields = fields


class JobResultSerializer(serializers.Serializer):
    """Job result with download URL."""
    job_id = serializers.UUIDField()
    download_url = serializers.URLField()
    expires_at = serializers.DateTimeField()
    size_bytes = serializers.IntegerField()
    record_count = serializers.IntegerField()
    output_format = serializers.CharField()


class GenerateResponseSerializer(serializers.Serializer):
    """Response after creating a generation job."""
    job_id = serializers.UUIDField()
    status = serializers.CharField()
    record_count = serializers.IntegerField()
    estimated_cost_cents = serializers.IntegerField()
    message = serializers.CharField()
    # For sync mode (small jobs)
    data = serializers.ListField(child=serializers.DictField(), required=False)


class SchemaTemplateSerializer(serializers.ModelSerializer):
    """Schema template."""

    class Meta:
        model = SchemaTemplate
        fields = ["id", "name", "description", "domain", "data_type", "schema_def", "is_builtin"]
        read_only_fields = ["id", "is_builtin"]


class UsageSummarySerializer(serializers.Serializer):
    """Usage summary for billing period."""
    period_start = serializers.DateTimeField()
    period_end = serializers.DateTimeField()
    total_records = serializers.IntegerField()
    total_cost_cents = serializers.IntegerField()
    jobs_completed = serializers.IntegerField()
    jobs_failed = serializers.IntegerField()
    records_by_type = serializers.DictField(child=serializers.IntegerField())


class UsageResponseSerializer(serializers.Serializer):
    """Full usage response."""
    current_period = UsageSummarySerializer()
    tier = serializers.CharField()
    tier_limit = serializers.IntegerField()
    records_remaining = serializers.IntegerField()

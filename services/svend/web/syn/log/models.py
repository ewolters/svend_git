"""
Logging models for LOG-001/002 compliant structured logging.

Standard: LOG-002 §4
Compliance: NIST SP 800-53 AU-2, AU-3, AU-9 / ISO 27001 A.12.4.1-4
"""

import uuid

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone

from syn.core.base_models import SynaraEntity, SynaraImmutableLog

# =============================================================================
# Log Level Constants (LOG-001 §5.2, RFC 5424 aligned)
# =============================================================================

LOG_LEVEL_CHOICES = [
    ("DEBUG", "Debug"),
    ("INFO", "Info"),
    ("WARNING", "Warning"),
    ("ERROR", "Error"),
    ("CRITICAL", "Critical"),
]

LOG_LEVEL_NUMERIC = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}

AGGREGATION_SINK_CHOICES = [
    ("elk", "Elasticsearch/Logstash/Kibana"),
    ("datadog", "Datadog"),
    ("splunk", "Splunk"),
    ("cloudwatch", "AWS CloudWatch"),
]

LOG_SOURCE_CHOICES = [
    ("application", "Application"),
    ("reflex", "Reflex"),
    ("primitive", "Primitive"),
    ("middleware", "Middleware"),
    ("handler", "Handler"),
]


class LogStream(SynaraEntity):
    """
    Logical grouping of log entries with retention policies.

    Standard: LOG-002 §4.3
    Table: syn_log_stream

    Features:
    - Retention and archive policies per stream
    - Minimum log level filtering
    - Aggregation sink configuration
    - SIEM integration toggle

    Compliance:
    - ISO 27001 A.12.4.2: Protection of log information
    """

    id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False, help_text="Unique log stream identifier"
    )
    correlation_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        db_index=True,
        help_text="Correlation ID for causal trace graph (CTG-001 §5)",
    )
    name = models.CharField(
        max_length=100,
        unique=True,
        db_index=True,
        help_text="Unique stream name (e.g., 'application', 'security', 'audit')",
    )
    description = models.TextField(blank=True, help_text="Human-readable description of this log stream")
    tenant_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="Tenant identifier for multi-tenant isolation (SEC-001 §5.2)"
    )
    retention_days = models.PositiveIntegerField(
        default=90,
        validators=[MinValueValidator(1), MaxValueValidator(3650)],
        help_text="Number of days to retain logs in hot storage",
    )
    archive_days = models.PositiveIntegerField(
        default=365,
        validators=[MinValueValidator(1), MaxValueValidator(3650)],
        help_text="Number of days to retain logs in archive storage",
    )
    min_level = models.CharField(
        max_length=10,
        choices=LOG_LEVEL_CHOICES,
        default="INFO",
        help_text="Minimum log level to capture for this stream",
    )
    is_aggregation_enabled = models.BooleanField(
        default=True, help_text="Whether to compute aggregated metrics for this stream"
    )
    aggregation_sink = models.CharField(
        max_length=20,
        choices=AGGREGATION_SINK_CHOICES,
        null=True,
        blank=True,
        help_text="External aggregation sink for this stream",
    )
    is_siem_enabled = models.BooleanField(default=False, help_text="Whether to forward logs to SIEM (SEC-002 §8)")
    is_active = models.BooleanField(
        default=True, db_index=True, help_text="Whether this stream is actively accepting logs"
    )
    created_at = models.DateTimeField(auto_now_add=True, help_text="When this stream was created")
    updated_at = models.DateTimeField(auto_now=True, help_text="When this stream was last updated")

    class Meta(SynaraEntity.Meta):
        db_table = "syn_log_stream"
        ordering = ["name"]
        indexes = [
            models.Index(fields=["tenant_id", "is_active"], name="logstream_tenant_active"),
        ]
        verbose_name = "Log Stream"
        verbose_name_plural = "Log Streams"

    class SynaraMeta:
        event_domain = "syn.log.log_stream"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        return f"{self.name} (min_level={self.min_level}, retention={self.retention_days}d)"

    @property
    def min_level_numeric(self) -> int:
        """Get numeric value for minimum log level."""
        return LOG_LEVEL_NUMERIC.get(self.min_level, 20)


class LogEntry(SynaraImmutableLog):
    """
    Core log entry model storing LOG-001 compliant structured log records.

    Standard: LOG-002 §4.2
    Table: syn_log_entry

    Features:
    - All LOG-001 required fields (timestamp, level, logger, message, correlation_id)
    - Optional structured context and metadata
    - Exception details for error logs
    - SBL-001 layer tracking
    - Archive flag for cold storage

    Compliance:
    - LOG-001 §5: Structured log format
    - NIST SP 800-53 AU-3: Content of audit records
    - ISO 27001 A.12.4.1: Event logging
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, help_text="Unique log entry identifier")
    timestamp = models.DateTimeField(default=timezone.now, db_index=True, help_text="Log event timestamp (UTC)")
    level = models.CharField(max_length=10, choices=LOG_LEVEL_CHOICES, db_index=True, help_text="Log severity level")
    level_numeric = models.SmallIntegerField(
        db_index=True, help_text="Numeric log level for range queries (10=DEBUG, 50=CRITICAL)"
    )
    logger = models.CharField(
        max_length=255, db_index=True, help_text="Logger name (module path, e.g., 'syn.cortex.publisher')"
    )
    message = models.TextField(help_text="Human-readable log message")
    correlation_id = models.UUIDField(db_index=True, help_text="Request/operation correlation ID (CTG-001 §5)")
    tenant_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="Tenant identifier for multi-tenant isolation (SEC-001 §5.2)"
    )
    context = models.JSONField(
        default=dict, blank=True, help_text="Structured context (user_id, operation, request_id, etc.)"
    )
    metadata = models.JSONField(
        default=dict, blank=True, help_text="System metadata (hostname, process_id, thread_id, etc.)"
    )
    exception = models.JSONField(
        null=True, blank=True, help_text="Exception details for error logs (type, message, traceback)"
    )
    layer = models.SmallIntegerField(
        null=True,
        blank=True,
        db_index=True,
        validators=[MinValueValidator(1), MaxValueValidator(10)],
        help_text="SBL-001 biological layer (1-10)",
    )
    source = models.CharField(
        max_length=20,
        choices=LOG_SOURCE_CHOICES,
        default="application",
        db_index=True,
        help_text="Log source (application, reflex, primitive)",
    )
    stream = models.ForeignKey(
        LogStream, on_delete=models.PROTECT, related_name="entries", help_text="Parent log stream"
    )
    is_archived = models.BooleanField(default=False, db_index=True, help_text="Archived to cold storage")

    class Meta(SynaraImmutableLog.Meta):
        db_table = "syn_log_entry"
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["tenant_id", "timestamp"], name="idx_log_tenant_time"),
            models.Index(fields=["correlation_id", "timestamp"], name="idx_log_correlation_time"),
            models.Index(fields=["level", "timestamp"], name="idx_log_level_time"),
            models.Index(fields=["logger", "level", "timestamp"], name="idx_log_logger_level_time"),
            models.Index(fields=["stream", "timestamp"], name="idx_log_stream_time"),
        ]
        verbose_name = "Log Entry"
        verbose_name_plural = "Log Entries"

    class SynaraMeta:
        event_domain = "syn.log.log_entry"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        return f"[{self.timestamp}] {self.level} {self.logger}: {self.message[:50]}"

    def save(self, *args, **kwargs):
        """Override save to compute level_numeric."""
        if self.level:
            self.level_numeric = LOG_LEVEL_NUMERIC.get(self.level, 20)
        super().save(*args, **kwargs)

    @property
    def is_error(self) -> bool:
        """Check if this is an error-level log."""
        return self.level in ("ERROR", "CRITICAL")


class LogAlert(SynaraEntity):
    """
    Alert configuration for log-based monitoring.

    Standard: LOG-002 §4.4
    Table: syn_log_alert

    Features:
    - Threshold-based alerting per stream
    - Configurable time windows
    - Cooldown to prevent alert storms
    - Reflex event integration

    Compliance:
    - ISO 27001 A.12.4.3: Administrator and operator logs
    """

    ALERT_LEVEL_CHOICES = [
        ("WARNING", "Warning"),
        ("ERROR", "Error"),
        ("CRITICAL", "Critical"),
    ]

    id = models.UUIDField(
        primary_key=True, default=uuid.uuid4, editable=False, help_text="Unique alert configuration identifier"
    )
    correlation_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        db_index=True,
        help_text="Correlation ID for causal trace graph (CTG-001 §5)",
    )
    name = models.CharField(max_length=100, db_index=True, help_text="Alert configuration name")
    stream = models.ForeignKey(
        LogStream, on_delete=models.CASCADE, related_name="alerts", help_text="Log stream to monitor"
    )
    level = models.CharField(
        max_length=10, choices=ALERT_LEVEL_CHOICES, default="ERROR", help_text="Log level to trigger alert"
    )
    threshold_count = models.PositiveIntegerField(
        default=10, validators=[MinValueValidator(1)], help_text="Number of matching logs to trigger alert"
    )
    threshold_window_minutes = models.PositiveIntegerField(
        default=5,
        validators=[MinValueValidator(1), MaxValueValidator(1440)],
        help_text="Time window for threshold evaluation (minutes)",
    )
    reflex_event = models.CharField(
        max_length=100, default="log.alert.triggered", help_text="Reflex event to emit when triggered"
    )
    cooldown_minutes = models.PositiveIntegerField(
        default=15,
        validators=[MinValueValidator(1), MaxValueValidator(1440)],
        help_text="Cooldown period between alerts (minutes)",
    )
    is_enabled = models.BooleanField(default=True, db_index=True, help_text="Whether this alert is active")
    tenant_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="Tenant identifier for multi-tenant isolation (SEC-001 §5.2)"
    )
    last_triggered_at = models.DateTimeField(null=True, blank=True, help_text="When this alert was last triggered")
    created_at = models.DateTimeField(auto_now_add=True, help_text="When this alert was created")
    updated_at = models.DateTimeField(auto_now=True, help_text="When this alert was last updated")

    class Meta(SynaraEntity.Meta):
        db_table = "syn_log_alert"
        ordering = ["name"]
        indexes = [
            models.Index(fields=["stream", "is_enabled"], name="logalert_stream_enabled"),
            models.Index(fields=["tenant_id", "is_enabled"], name="logalert_tenant_enabled"),
        ]
        verbose_name = "Log Alert"
        verbose_name_plural = "Log Alerts"

    class SynaraMeta:
        event_domain = "syn.log.log_alert"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        return (
            f"{self.name} ({self.stream.name}: {self.level} >= {self.threshold_count}/{self.threshold_window_minutes}m)"
        )

    def is_in_cooldown(self) -> bool:
        """Check if alert is currently in cooldown period."""
        if not self.last_triggered_at:
            return False
        from datetime import timedelta

        cooldown_end = self.last_triggered_at + timedelta(minutes=self.cooldown_minutes)
        return timezone.now() < cooldown_end


class LogMetric(SynaraEntity):
    """
    Aggregated log metrics for dashboards and analysis.

    Standard: LOG-002 §4.5
    Table: syn_log_metric

    Features:
    - Time-bucketed aggregations
    - Per-stream and per-logger metrics
    - Error count tracking
    - Configurable bucket duration

    Compliance:
    - TEL-001 §5: Metrics integration
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, help_text="Unique metric identifier")
    correlation_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        db_index=True,
        help_text="Correlation ID for causal trace graph (CTG-001 §5)",
    )
    bucket_start = models.DateTimeField(db_index=True, help_text="Start of the aggregation bucket")
    bucket_duration_minutes = models.PositiveSmallIntegerField(
        default=5,
        validators=[MinValueValidator(1), MaxValueValidator(60)],
        help_text="Duration of the bucket in minutes",
    )
    stream = models.ForeignKey(
        LogStream, on_delete=models.CASCADE, related_name="metrics", help_text="Log stream for this metric"
    )
    tenant_id = models.UUIDField(
        null=True, blank=True, db_index=True, help_text="Tenant identifier for multi-tenant isolation (SEC-001 §5.2)"
    )
    logger = models.CharField(
        max_length=255, null=True, blank=True, db_index=True, help_text="Logger name (if aggregating per logger)"
    )
    level = models.CharField(
        max_length=10, choices=LOG_LEVEL_CHOICES, db_index=True, help_text="Log level for this metric bucket"
    )
    count = models.PositiveIntegerField(default=0, help_text="Total log entries in this bucket")
    error_count = models.PositiveIntegerField(default=0, help_text="Error-level log entries in this bucket")
    created_at = models.DateTimeField(auto_now_add=True, help_text="When this metric was created")

    class Meta(SynaraEntity.Meta):
        db_table = "syn_log_metric"
        ordering = ["-bucket_start"]
        indexes = [
            models.Index(fields=["stream", "bucket_start"], name="logmetric_stream_bucket"),
            models.Index(fields=["tenant_id", "bucket_start"], name="logmetric_tenant_bucket"),
            models.Index(fields=["logger", "level", "bucket_start"], name="logmetric_logger_level"),
        ]
        unique_together = [["stream", "tenant_id", "logger", "level", "bucket_start"]]
        verbose_name = "Log Metric"
        verbose_name_plural = "Log Metrics"

    class SynaraMeta:
        event_domain = "syn.log.log_metric"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        return f"{self.stream.name} @ {self.bucket_start}: {self.count} entries ({self.error_count} errors)"

    @property
    def bucket_end(self):
        """Calculate bucket end time."""
        from datetime import timedelta

        return self.bucket_start + timedelta(minutes=self.bucket_duration_minutes)

    @property
    def error_rate(self) -> float:
        """Calculate error rate for this bucket."""
        if self.count == 0:
            return 0.0
        return self.error_count / self.count


# =============================================================================
# Request Metric (HTTP telemetry — SLA-001 §6)
# =============================================================================


class RequestMetric(models.Model):
    """
    Pre-aggregated HTTP request metrics in 5-minute buckets.

    Standard: SLA-001 §6, LOG-001 §5
    Purpose: Feeds the Performance dashboard and SLA response_time measurement.

    Design: Bucketed aggregation with reservoir sampling for percentile estimation.
    A site at 100 req/s generates ~288 rows/day instead of 8.6M per-request rows.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    bucket_start = models.DateTimeField(db_index=True)
    bucket_minutes = models.PositiveSmallIntegerField(default=5)
    method = models.CharField(max_length=10)
    path_pattern = models.CharField(max_length=255, db_index=True)
    status_class = models.CharField(max_length=3)  # "2xx", "3xx", "4xx", "5xx"

    # Aggregates
    request_count = models.PositiveIntegerField(default=0)
    error_count = models.PositiveIntegerField(default=0)  # 5xx only
    total_duration_ms = models.FloatField(default=0.0)
    min_duration_ms = models.FloatField(default=0.0)
    max_duration_ms = models.FloatField(default=0.0)

    # Reservoir sample for percentile estimation (JSON list of up to 100 durations)
    duration_samples = models.JSONField(default=list)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "syn_request_metric"
        unique_together = [("bucket_start", "method", "path_pattern", "status_class")]
        indexes = [
            models.Index(
                fields=["bucket_start", "path_pattern"],
                name="reqmetric_bucket_path",
            ),
        ]
        ordering = ["-bucket_start"]
        verbose_name = "Request Metric"
        verbose_name_plural = "Request Metrics"

    def __str__(self):
        return f"{self.method} {self.path_pattern} {self.status_class} @ {self.bucket_start}: {self.request_count} reqs"

    @property
    def avg_duration_ms(self):
        if self.request_count == 0:
            return 0.0
        return self.total_duration_ms / self.request_count

    def percentile(self, p):
        """Compute percentile from reservoir samples using linear interpolation."""
        if not self.duration_samples:
            return None
        sorted_s = sorted(self.duration_samples)
        n = len(sorted_s)
        if n == 1:
            return sorted_s[0]
        k = (n - 1) * (p / 100)
        f = int(k)
        c = min(f + 1, n - 1)
        return sorted_s[f] + (k - f) * (sorted_s[c] - sorted_s[f])

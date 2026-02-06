"""Chat models for conversations and messages."""

import uuid
from django.conf import settings
from django.db import models


class ModelVersion(models.Model):
    """Track deployed model checkpoints for hot-swapping.

    Allows updating models without restart via admin endpoint.
    """

    class ModelType(models.TextChoices):
        SAFETY_ROUTER = "safety_router", "Safety Router"
        INTUITION = "intuition", "Intuition"
        DIFFUSION = "diffusion", "Diffusion"
        REASONER = "reasoner", "MoE Reasoner"
        VERIFIER = "verifier", "Verifier"
        LANGUAGE_MODEL = "language_model", "Language Model"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model_type = models.CharField(
        max_length=30,
        choices=ModelType.choices,
        db_index=True,
    )

    # Checkpoint info
    name = models.CharField(max_length=100)  # e.g., "reasoner-epoch2-v1"
    checkpoint_path = models.CharField(max_length=500)
    is_active = models.BooleanField(default=False, db_index=True)

    # Metadata
    description = models.TextField(blank=True)
    metrics = models.JSONField(null=True, blank=True)  # accuracy, loss, etc.

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    activated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "model_versions"
        ordering = ["-created_at"]
        constraints = [
            # Only one active version per model type
            models.UniqueConstraint(
                fields=["model_type"],
                condition=models.Q(is_active=True),
                name="unique_active_model_per_type",
            )
        ]

    def __str__(self):
        status = "ACTIVE" if self.is_active else "inactive"
        return f"{self.model_type}: {self.name} ({status})"

    def activate(self):
        """Activate this version, deactivating others of same type."""
        from django.utils import timezone

        # Deactivate other versions of this type
        ModelVersion.objects.filter(
            model_type=self.model_type,
            is_active=True,
        ).exclude(pk=self.pk).update(is_active=False)

        # Activate this one
        self.is_active = True
        self.activated_at = timezone.now()
        self.save(update_fields=["is_active", "activated_at"])


class UsageLog(models.Model):
    """Daily usage tracking for rate limiting and analytics.

    One row per user per day. Enables usage-based pricing later.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="usage_logs",
    )
    date = models.DateField(db_index=True)

    # Counts
    request_count = models.IntegerField(default=0)
    tokens_input = models.BigIntegerField(default=0)
    tokens_output = models.BigIntegerField(default=0)
    blocked_count = models.IntegerField(default=0)
    error_count = models.IntegerField(default=0)

    # Timing
    total_inference_ms = models.BigIntegerField(default=0)

    # Domain breakdown (JSON: {"math_algebra": 5, "science_physics": 3})
    domain_counts = models.JSONField(null=True, blank=True)

    class Meta:
        db_table = "usage_logs"
        unique_together = ["user", "date"]
        ordering = ["-date"]

    def __str__(self):
        return f"{self.user.username} - {self.date} ({self.request_count} requests)"

    @classmethod
    def log_request(
        cls,
        user,
        tokens_in: int = 0,
        tokens_out: int = 0,
        inference_ms: int = 0,
        domain: str = "",
        blocked: bool = False,
        error: bool = False,
    ):
        """Log a request, creating today's row if needed."""
        from django.utils import timezone

        today = timezone.now().date()
        log, _ = cls.objects.get_or_create(user=user, date=today)

        log.request_count += 1
        log.tokens_input += tokens_in
        log.tokens_output += tokens_out
        log.total_inference_ms += inference_ms

        if blocked:
            log.blocked_count += 1
        if error:
            log.error_count += 1

        # Update domain counts
        if domain:
            if log.domain_counts is None:
                log.domain_counts = {}
            log.domain_counts[domain] = log.domain_counts.get(domain, 0) + 1

        log.save()
        return log


class Conversation(models.Model):
    """A conversation (chat session) with multiple messages."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="conversations",
    )
    title = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "conversations"
        ordering = ["-updated_at"]

    def __str__(self):
        return self.title or f"Conversation {self.id}"

    def generate_title(self):
        """Generate title from first message."""
        first_msg = self.messages.filter(role="user").first()
        if first_msg:
            text = first_msg.content[:50]
            self.title = text + "..." if len(first_msg.content) > 50 else text
            self.save(update_fields=["title"])


class Message(models.Model):
    """A single message in a conversation."""

    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"
        SYSTEM = "system", "System"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name="messages",
    )
    role = models.CharField(max_length=10, choices=Role.choices)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    # Pipeline metadata (for assistant messages)
    domain = models.CharField(max_length=50, blank=True)
    difficulty = models.FloatField(null=True, blank=True)
    verified = models.BooleanField(null=True)
    verification_confidence = models.FloatField(null=True, blank=True)
    blocked = models.BooleanField(default=False)
    block_reason = models.CharField(max_length=255, blank=True)

    # Reasoning trace (JSON)
    reasoning_trace = models.JSONField(null=True, blank=True)
    tool_calls = models.JSONField(null=True, blank=True)

    # Timing
    inference_time_ms = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = "messages"
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."


class SharedConversation(models.Model):
    """A publicly shared conversation."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.OneToOneField(
        Conversation,
        on_delete=models.CASCADE,
        related_name="share",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    view_count = models.IntegerField(default=0)

    class Meta:
        db_table = "shared_conversations"

    def __str__(self):
        return f"Shared: {self.conversation.title}"


class TraceLog(models.Model):
    """Full diagnostic trace log for every inference request.

    Stores complete pipeline state for debugging and analysis.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.OneToOneField(
        Message,
        on_delete=models.CASCADE,
        related_name="trace_log",
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    # Input
    input_text = models.TextField()
    user_id = models.UUIDField(null=True, blank=True)

    # Safety Router
    safety_passed = models.BooleanField(default=True)
    safety_confidence = models.FloatField(null=True, blank=True)

    # Intuition
    domain = models.CharField(max_length=50, blank=True)
    difficulty = models.FloatField(null=True, blank=True)
    conditioning_norm = models.FloatField(null=True, blank=True)  # Sanity check

    # Reasoner
    reasoning_trace = models.JSONField(null=True, blank=True)
    tool_calls = models.JSONField(null=True, blank=True)
    reasoner_raw_output = models.TextField(blank=True)  # Raw model output

    # Verifier
    verified = models.BooleanField(null=True)
    verification_confidence = models.FloatField(null=True, blank=True)

    # Language Model
    lm_prompt = models.TextField(blank=True)  # What we sent to LM
    lm_raw_output = models.TextField(blank=True)  # Raw LM output before cleanup
    response = models.TextField(blank=True)  # Final cleaned response

    # Gate decisions
    gate_passed = models.BooleanField(default=True)
    gate_reason = models.CharField(max_length=255, blank=True)
    fallback_used = models.BooleanField(default=False)

    # Timing breakdown (ms)
    safety_time_ms = models.IntegerField(null=True, blank=True)
    intuition_time_ms = models.IntegerField(null=True, blank=True)
    reasoner_time_ms = models.IntegerField(null=True, blank=True)
    verifier_time_ms = models.IntegerField(null=True, blank=True)
    lm_time_ms = models.IntegerField(null=True, blank=True)
    total_time_ms = models.IntegerField(null=True, blank=True)

    # Error tracking
    error_stage = models.CharField(max_length=50, blank=True)  # Which stage failed
    error_message = models.TextField(blank=True)

    class Meta:
        db_table = "trace_logs"
        ordering = ["-created_at"]

    def __str__(self):
        return f"Trace {self.id} - {self.domain} ({self.total_time_ms}ms)"


class TrainingCandidate(models.Model):
    """Candidate training pairs collected from production failures.

    Low-confidence responses, errors, and user-flagged issues
    get stored here for potential inclusion in training data.
    """

    class CandidateType(models.TextChoices):
        LOW_CONFIDENCE = "low_confidence", "Low Confidence"
        VERIFICATION_FAILED = "verification_failed", "Verification Failed"
        ERROR = "error", "Pipeline Error"
        USER_FLAGGED = "user_flagged", "User Flagged"
        RANDOM_SAMPLE = "random_sample", "Random Sample"

    class Status(models.TextChoices):
        PENDING = "pending", "Pending Review"
        APPROVED = "approved", "Approved for Training"
        REJECTED = "rejected", "Rejected"
        EXPORTED = "exported", "Exported"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    trace_log = models.ForeignKey(
        TraceLog,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="training_candidates",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    # Classification
    candidate_type = models.CharField(
        max_length=30,
        choices=CandidateType.choices,
        default=CandidateType.LOW_CONFIDENCE,
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )

    # The actual data (duplicated for persistence if trace_log deleted)
    input_text = models.TextField()
    domain = models.CharField(max_length=50, blank=True)
    difficulty = models.FloatField(null=True, blank=True)
    reasoning_trace = models.JSONField(null=True, blank=True)
    model_response = models.TextField(blank=True)

    # Correction (filled in during review)
    corrected_response = models.TextField(blank=True)
    reviewer_notes = models.TextField(blank=True)

    # Metadata for filtering
    verification_confidence = models.FloatField(null=True, blank=True)
    error_type = models.CharField(max_length=100, blank=True)

    class Meta:
        db_table = "training_candidates"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.candidate_type} - {self.input_text[:50]}..."

    def to_training_format(self) -> dict:
        """Export as training data format."""
        return {
            "question": self.input_text,
            "reasoning": self.reasoning_trace or [],
            "answer": self.corrected_response or self.model_response,
            "_metadata": {
                "domain": self.domain,
                "difficulty": self.difficulty,
                "source": "production_candidate",
                "candidate_id": str(self.id),
            }
        }

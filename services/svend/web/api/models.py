"""API app models."""

import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone
from django.utils.text import slugify


class BlogPost(models.Model):
    """Blog post for SEO content marketing."""

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        SCHEDULED = "scheduled", "Scheduled"
        PUBLISHED = "published", "Published"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True, blank=True)
    body = models.TextField(help_text="Markdown content")
    meta_description = models.CharField(max_length=160, blank=True)
    status = models.CharField(
        max_length=10, choices=Status.choices, default=Status.DRAFT, db_index=True
    )
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="blog_posts",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_at = models.DateTimeField(null=True, blank=True, db_index=True)
    scheduled_at = models.DateTimeField(null=True, blank=True, db_index=True)

    class Meta:
        db_table = "blog_posts"
        ordering = ["-published_at", "-created_at"]

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)[:200]
            # Ensure uniqueness
            base_slug = self.slug
            counter = 1
            while BlogPost.objects.filter(slug=self.slug).exclude(pk=self.pk).exists():
                self.slug = f"{base_slug[:190]}-{counter}"
                counter += 1
        super().save(*args, **kwargs)


class BlogView(models.Model):
    """Tracks individual blog post views for analytics."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    post = models.ForeignKey(
        BlogPost,
        on_delete=models.CASCADE,
        related_name="views",
    )
    viewed_at = models.DateTimeField(auto_now_add=True, db_index=True)
    referrer = models.URLField(max_length=500, blank=True)
    referrer_domain = models.CharField(max_length=200, blank=True, db_index=True)
    path = models.CharField(max_length=300, blank=True)
    ip_hash = models.CharField(max_length=64, blank=True)  # SHA-256 hashed IP for unique visitor counting
    user_agent = models.CharField(max_length=500, blank=True)
    country = models.CharField(max_length=2, blank=True)  # ISO code, future use
    is_bot = models.BooleanField(default=False)

    class Meta:
        db_table = "blog_views"
        ordering = ["-viewed_at"]
        indexes = [
            models.Index(fields=["post", "viewed_at"]),
            models.Index(fields=["referrer_domain", "viewed_at"]),
        ]

    def __str__(self):
        return f"View: {self.post.title} @ {self.viewed_at}"


class OnboardingSurvey(models.Model):
    """Stores onboarding survey responses per user."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="onboarding_survey",
    )

    # Demographics (mirrors User fields but captured at onboarding time)
    industry = models.CharField(max_length=20, blank=True)
    role = models.CharField(max_length=20, blank=True)
    experience_level = models.CharField(max_length=20, blank=True)
    organization_size = models.CharField(max_length=20, blank=True)

    # Goals — what brought them here
    primary_goal = models.CharField(max_length=40, blank=True)
    tools_used = models.JSONField(default=list, blank=True)  # ["minitab", "jmp", "excel", ...]

    # Self-assessment — "how do you feel" style (your dissertation insight)
    confidence_stats = models.IntegerField(default=3)  # 1-5 scale
    urgency = models.IntegerField(default=3)  # 1-5: how soon do you need results?

    # Free text
    biggest_challenge = models.TextField(blank=True)

    # Computed learning path
    learning_path = models.CharField(max_length=40, blank=True)  # quality_engineer, analyst, beginner, researcher, manager

    # Feedback tracking
    helpful_emails = models.JSONField(default=list, blank=True)  # email IDs user found useful
    skipped_emails = models.JSONField(default=list, blank=True)  # email IDs user skipped/unsubscribed

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "onboarding_surveys"

    def __str__(self):
        return f"Onboarding: {self.user}"

    def compute_learning_path(self):
        """Determine personalized learning path from survey responses."""
        # Quality engineer path: manufacturing + engineer + intermediate+
        if self.industry == "manufacturing" and self.role == "engineer":
            return "quality_engineer"
        # Manager path: managers/executives want dashboards + summaries
        if self.role in ("manager", "executive"):
            return "manager"
        # Researcher path: advanced stats users
        if self.experience_level == "advanced" or self.role == "researcher":
            return "researcher"
        # Analyst path: intermediate users with data focus
        if self.experience_level == "intermediate" and self.role == "analyst":
            return "analyst"
        # Student path
        if self.role == "student":
            return "student"
        # Default: beginner-friendly guided path
        return "beginner"


class OnboardingEmail(models.Model):
    """Tracks scheduled and sent onboarding drip emails."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        SENT = "sent", "Sent"
        FAILED = "failed", "Failed"
        SKIPPED = "skipped", "Skipped"  # user unsubscribed or already converted

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="onboarding_emails",
    )
    email_key = models.CharField(max_length=40)  # e.g. "welcome", "getting_started", "tips_1", "learning_path", "checkin"
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING)
    scheduled_for = models.DateTimeField(db_index=True)
    sent_at = models.DateTimeField(null=True, blank=True)
    opened_at = models.DateTimeField(null=True, blank=True)  # future: track opens
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "onboarding_emails"
        ordering = ["scheduled_for"]
        unique_together = [("user", "email_key")]

    def __str__(self):
        return f"{self.user} - {self.email_key} ({self.status})"


class EmailCampaign(models.Model):
    """A sent email campaign for traceability."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    subject = models.CharField(max_length=200)
    body_md = models.TextField()
    target = models.CharField(max_length=50)  # "all", "tier:free", "eric@example.com", etc.
    sent_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
    )
    recipient_count = models.IntegerField(default=0)
    is_test = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "email_campaigns"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.subject} ({self.target})"


class EmailRecipient(models.Model):
    """Per-recipient tracking for a campaign."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    campaign = models.ForeignKey(EmailCampaign, on_delete=models.CASCADE, related_name="recipients")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True,
    )
    email = models.EmailField()
    sent_at = models.DateTimeField(auto_now_add=True)
    opened_at = models.DateTimeField(null=True, blank=True)
    clicked_at = models.DateTimeField(null=True, blank=True)
    failed = models.BooleanField(default=False)

    class Meta:
        db_table = "email_recipients"
        ordering = ["-sent_at"]

    def __str__(self):
        return f"{self.email} - {'opened' if self.opened_at else 'sent'}"

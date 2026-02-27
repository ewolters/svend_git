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


class SiteVisit(models.Model):
    """Tracks anonymous page visits across the entire site for marketing analytics."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    path = models.CharField(max_length=300, db_index=True)
    viewed_at = models.DateTimeField(auto_now_add=True, db_index=True)
    referrer = models.URLField(max_length=500, blank=True)
    referrer_domain = models.CharField(max_length=200, blank=True, db_index=True)
    ip_hash = models.CharField(max_length=64, blank=True)
    user_agent = models.CharField(max_length=500, blank=True)
    country = models.CharField(max_length=2, blank=True)
    is_bot = models.BooleanField(default=False)
    method = models.CharField(max_length=10, blank=True)
    duration_ms = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = "site_visits"
        ordering = ["-viewed_at"]
        indexes = [
            models.Index(fields=["path", "viewed_at"]),
            models.Index(fields=["referrer_domain", "viewed_at"]),
            models.Index(fields=["is_bot", "viewed_at"]),
            models.Index(fields=["country", "viewed_at"]),
        ]

    def __str__(self):
        return f"{self.path} @ {self.viewed_at}"


class WhitePaper(models.Model):
    """White paper / long-form gated content for SEO and lead generation."""

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        PUBLISHED = "published", "Published"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True, blank=True)
    description = models.TextField(blank=True, help_text="Short marketing copy / abstract")
    body = models.TextField(blank=True, help_text="Full markdown content")
    meta_description = models.CharField(max_length=160, blank=True)
    topic = models.CharField(max_length=100, blank=True, db_index=True)
    status = models.CharField(
        max_length=10, choices=Status.choices, default=Status.DRAFT, db_index=True
    )
    gated = models.BooleanField(default=True, help_text="Require email to download")
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="white_papers",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_at = models.DateTimeField(null=True, blank=True, db_index=True)

    class Meta:
        db_table = "white_papers"
        ordering = ["-published_at", "-created_at"]

    def __str__(self):
        return self.title

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)[:200]
            base_slug = self.slug
            counter = 1
            while WhitePaper.objects.filter(slug=self.slug).exclude(pk=self.pk).exists():
                self.slug = f"{base_slug[:190]}-{counter}"
                counter += 1
        super().save(*args, **kwargs)


class WhitePaperDownload(models.Model):
    """Tracks individual white paper downloads/views for analytics."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    paper = models.ForeignKey(
        WhitePaper,
        on_delete=models.CASCADE,
        related_name="downloads",
    )
    downloaded_at = models.DateTimeField(auto_now_add=True, db_index=True)
    referrer_domain = models.CharField(max_length=200, blank=True, db_index=True)
    ip_hash = models.CharField(max_length=64, blank=True)
    user_agent = models.CharField(max_length=500, blank=True)
    email = models.EmailField(blank=True)  # Captured if gated
    is_bot = models.BooleanField(default=False)

    class Meta:
        db_table = "whitepaper_downloads"
        ordering = ["-downloaded_at"]
        indexes = [
            models.Index(fields=["paper", "downloaded_at"]),
            models.Index(fields=["referrer_domain", "downloaded_at"]),
        ]

    def __str__(self):
        return f"Download: {self.paper.title} @ {self.downloaded_at}"


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


# ---------------------------------------------------------------------------
# Automation Framework
# ---------------------------------------------------------------------------


class Experiment(models.Model):
    """A/B test experiment."""

    class ExperimentType(models.TextChoices):
        EMAIL_SUBJECT = "email_subject", "Email Subject"
        EMAIL_BODY = "email_body", "Email Body"
        ONBOARDING_FLOW = "onboarding_flow", "Onboarding Flow"
        FEATURE_FLAG = "feature_flag", "Feature Flag"

    class Metric(models.TextChoices):
        CONVERSION = "conversion", "Conversion"
        RETENTION = "retention", "Retention"
        REVENUE = "revenue", "Revenue"
        ENGAGEMENT = "engagement", "Engagement"

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        RUNNING = "running", "Running"
        CONCLUDED = "concluded", "Concluded"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    hypothesis = models.TextField(blank=True)
    experiment_type = models.CharField(max_length=20, choices=ExperimentType.choices)
    metric = models.CharField(max_length=20, choices=Metric.choices, default=Metric.CONVERSION)
    variants = models.JSONField(default=list)  # [{"name": "A", "weight": 50, "config": {...}}, ...]
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.DRAFT, db_index=True)
    winner = models.CharField(max_length=50, blank=True)  # winning variant name
    target = models.CharField(max_length=100, default="all")  # all / tier:X / new_users
    min_sample_size = models.IntegerField(default=100)
    results = models.JSONField(default=dict, blank=True)  # computed stats per variant
    started_at = models.DateTimeField(null=True, blank=True)
    ended_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "experiments"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.status})"


class ExperimentAssignment(models.Model):
    """Tracks which user got which variant."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name="assignments")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="experiment_assignments")
    variant = models.CharField(max_length=50)
    assigned_at = models.DateTimeField(auto_now_add=True)
    converted = models.BooleanField(default=False)
    converted_at = models.DateTimeField(null=True, blank=True)
    conversion_value = models.FloatField(null=True, blank=True)  # for revenue metrics

    class Meta:
        db_table = "experiment_assignments"
        unique_together = [("experiment", "user")]
        ordering = ["-assigned_at"]

    def __str__(self):
        return f"{self.user} → {self.experiment.name}:{self.variant}"


class AutomationRule(models.Model):
    """Behavioral trigger rule: if X → do Y."""

    class Trigger(models.TextChoices):
        QUERY_LIMIT_NEAR = "query_limit_near", "Query limit near"
        INACTIVE_DAYS = "inactive_days", "Inactive for N days"
        SIGNUP_NO_QUERY = "signup_no_query", "Signup with no queries"
        CHURN_SIGNAL = "churn_signal", "Churn signal"
        MILESTONE = "milestone", "Query milestone"
        FEATURE_UNUSED = "feature_unused", "Feature unused"

    class Action(models.TextChoices):
        SEND_EMAIL = "send_email", "Send email"
        INTERNAL_ALERT = "internal_alert", "Internal alert"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    trigger = models.CharField(max_length=20, choices=Trigger.choices)
    trigger_config = models.JSONField(default=dict)  # {"days": 7}, {"threshold": 80}, etc.
    trigger_2 = models.CharField(max_length=20, choices=Trigger.choices, blank=True, default="")
    trigger_2_config = models.JSONField(null=True, blank=True)
    trigger_logic = models.CharField(max_length=3, default="and", choices=[("and", "AND"), ("or", "OR")])
    action = models.CharField(max_length=20, choices=Action.choices)
    action_config = models.JSONField(default=dict)  # {"template": "inactive_nudge"}
    is_active = models.BooleanField(default=True)
    cooldown_hours = models.IntegerField(default=72)
    times_fired = models.IntegerField(default=0)
    last_fired_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "automation_rules"
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({'active' if self.is_active else 'off'})"


class AutomationLog(models.Model):
    """Audit trail for automation rule fires."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    rule = models.ForeignKey(AutomationRule, on_delete=models.CASCADE, related_name="logs")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="automation_logs")
    fired_at = models.DateTimeField(auto_now_add=True)
    action_taken = models.CharField(max_length=200)
    result = models.CharField(max_length=20)  # success / failed / skipped

    class Meta:
        db_table = "automation_logs"
        ordering = ["-fired_at"]
        indexes = [
            models.Index(fields=["rule", "user", "fired_at"]),
        ]

    def __str__(self):
        return f"{self.rule.name} → {self.user} ({self.result})"


class AutopilotReport(models.Model):
    """Weekly Claude growth review report."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    data_snapshot = models.JSONField(default=dict)
    insights = models.JSONField(default=list)
    recommendations = models.JSONField(default=list)
    alerts = models.JSONField(default=list)
    status = models.CharField(max_length=20, default="pending_review")  # pending_review / reviewed

    class Meta:
        db_table = "autopilot_reports"
        ordering = ["-created_at"]

    def __str__(self):
        return f"Autopilot {self.created_at.date()} ({self.status})"


class Feedback(models.Model):
    """In-app user feedback."""

    class Category(models.TextChoices):
        BUG = "bug", "Bug Report"
        FEATURE = "feature", "Feature Request"
        QUESTION = "question", "Question"
        OTHER = "other", "Other"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="feedback",
        null=True,
        blank=True,
    )
    category = models.CharField(max_length=10, choices=Category.choices, default=Category.OTHER)
    message = models.TextField()
    page = models.CharField(max_length=200, blank=True)  # URL path where feedback was submitted
    status = models.CharField(max_length=10, default="new")  # new / reviewed / resolved
    internal_notes = models.TextField(blank=True, default="")  # Staff-only annotations
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "feedback"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.get_category_display()}: {self.message[:50]}"


# ---------------------------------------------------------------------------
# CRM — Outbound Outreach Management
# ---------------------------------------------------------------------------


class CRMLead(models.Model):
    """External lead for outbound outreach."""

    class Source(models.TextChoices):
        LINKEDIN = "linkedin", "LinkedIn"
        REFERRAL = "referral", "Referral"
        INBOUND = "inbound", "Inbound"
        CONFERENCE = "conference", "Conference"
        COLD = "cold", "Cold"
        WHITEPAPER = "whitepaper", "Whitepaper"
        OTHER = "other", "Other"

    class Stage(models.TextChoices):
        PROSPECT = "prospect", "Prospect"
        CONTACTED = "contacted", "Contacted"
        ENGAGED = "engaged", "Engaged"
        DEMO = "demo", "Demo"
        TRIAL = "trial", "Trial"
        CUSTOMER = "customer", "Customer"
        CHURNED = "churned", "Churned"
        LOST = "lost", "Lost"
        BOUNCED = "bounced", "Bounced"
        INVALID = "invalid", "Invalid"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    email = models.EmailField(unique=True)
    company = models.CharField(max_length=200, blank=True)
    role = models.CharField(max_length=100, blank=True)
    industry = models.CharField(max_length=100, blank=True)
    source = models.CharField(max_length=20, choices=Source.choices, default=Source.OTHER)
    stage = models.CharField(max_length=20, choices=Stage.choices, default=Stage.PROSPECT, db_index=True)
    notes = models.TextField(blank=True)
    tags = models.JSONField(default=list, blank=True)
    email_opted_out = models.BooleanField(default=False)
    last_contacted_at = models.DateTimeField(null=True, blank=True)
    next_followup_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "crm_leads"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["stage", "next_followup_at"]),
            models.Index(fields=["source", "created_at"]),
        ]

    def __str__(self):
        return f"{self.name} ({self.stage})"


class OutreachSequence(models.Model):
    """Multi-step outreach sequence with A/B variants per step."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    steps = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "outreach_sequences"
        ordering = ["-created_at"]

    def __str__(self):
        return self.name


class OutreachEnrollment(models.Model):
    """A lead enrolled in an outreach sequence."""

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        COMPLETED = "completed", "Completed"
        PAUSED = "paused", "Paused"
        REPLIED = "replied", "Replied"
        OPTED_OUT = "opted_out", "Opted Out"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    lead = models.ForeignKey(CRMLead, on_delete=models.CASCADE, related_name="enrollments")
    sequence = models.ForeignKey(OutreachSequence, on_delete=models.CASCADE, related_name="enrollments")
    current_step = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.ACTIVE)
    variant = models.CharField(max_length=1)
    last_sent_at = models.DateTimeField(null=True, blank=True)
    next_send_at = models.DateTimeField(null=True, blank=True)
    send_log = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "outreach_enrollments"
        ordering = ["-created_at"]
        unique_together = [("lead", "sequence")]
        indexes = [
            models.Index(fields=["status", "next_send_at"]),
        ]

    def __str__(self):
        return f"{self.lead.name} → {self.sequence.name} (step {self.current_step})"

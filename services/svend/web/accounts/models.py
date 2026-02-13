"""User models and billing."""

import secrets
import uuid
from datetime import timedelta

from django.contrib.auth.models import AbstractUser
from django.db import models

from .constants import (
    Tier, Industry, Role, ExperienceLevel, OrganizationSize,
    get_daily_limit, has_feature, is_paid_tier,
)


class Subscription(models.Model):
    """Stripe subscription tracking.

    Handles the billing relationship. User.tier controls access,
    this tracks the payment side.
    """

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        PAST_DUE = "past_due", "Past Due"
        CANCELED = "canceled", "Canceled"
        INCOMPLETE = "incomplete", "Incomplete"
        INCOMPLETE_EXPIRED = "incomplete_expired", "Incomplete Expired"
        TRIALING = "trialing", "Trialing"
        UNPAID = "unpaid", "Unpaid"
        PAUSED = "paused", "Paused"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        "User",
        on_delete=models.CASCADE,
        related_name="subscription",
    )

    # Stripe IDs
    stripe_subscription_id = models.CharField(max_length=255, unique=True, db_index=True)
    stripe_price_id = models.CharField(max_length=255, blank=True)

    # Status
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.INCOMPLETE,
    )

    # Billing period
    current_period_start = models.DateTimeField(null=True, blank=True)
    current_period_end = models.DateTimeField(null=True, blank=True)
    cancel_at_period_end = models.BooleanField(default=False)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "subscriptions"

    def __str__(self):
        return f"{self.user.username} - {self.status}"

    @property
    def is_active(self) -> bool:
        """Check if subscription grants access."""
        return self.status in (self.Status.ACTIVE, self.Status.TRIALING)


class InviteCode(models.Model):
    """Invite codes for alpha access.

    Set SVEND_REQUIRE_INVITE=false in .env to disable invite requirement.
    """

    code = models.CharField(max_length=20, unique=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Usage tracking
    max_uses = models.IntegerField(default=1)  # How many times this code can be used
    times_used = models.IntegerField(default=0)

    # Who used it
    used_by = models.ManyToManyField(
        "User",
        blank=True,
        related_name="invite_codes_used",
    )

    # Optional: who created it / notes
    note = models.CharField(max_length=255, blank=True)  # e.g., "For Mom"
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = "invite_codes"

    def __str__(self):
        return f"{self.code} ({self.times_used}/{self.max_uses})"

    @property
    def is_valid(self) -> bool:
        """Check if code can still be used."""
        return self.is_active and self.times_used < self.max_uses

    def use(self, user: "User") -> bool:
        """Mark code as used by a user. Returns True if successful."""
        if not self.is_valid:
            return False
        self.times_used += 1
        self.used_by.add(user)
        self.save(update_fields=["times_used"])
        return True

    @classmethod
    def generate(cls, count: int = 1, max_uses: int = 1, note: str = "") -> list["InviteCode"]:
        """Generate new invite codes."""
        codes = []
        for _ in range(count):
            # Generate readable code: XXXX-XXXX format
            code = f"{secrets.token_hex(2).upper()}-{secrets.token_hex(2).upper()}"
            invite = cls.objects.create(code=code, max_uses=max_uses, note=note)
            codes.append(invite)
        return codes


class User(AbstractUser):
    """Custom user model for Svend."""

    # Use unified Tier from constants
    Tier = Tier  # Re-export for backwards compatibility

    tier = models.CharField(
        max_length=12,
        choices=Tier.choices,
        default=Tier.FREE,
    )
    queries_today = models.IntegerField(default=0)
    queries_reset_at = models.DateTimeField(null=True, blank=True)

    # Founder's rate lock - users who signed up at founder pricing keep it
    is_founder_locked = models.BooleanField(default=False)

    # Stripe
    stripe_customer_id = models.CharField(max_length=255, blank=True, db_index=True)

    # Legacy fields (kept for backwards compat, use Subscription model instead)
    subscription_active = models.BooleanField(default=False)
    subscription_ends_at = models.DateTimeField(null=True, blank=True)

    # === Future features (nullable, no migration needed to enable) ===

    # Referrals
    referral_code = models.CharField(max_length=20, unique=True, null=True, blank=True)
    referred_by = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="referrals",
    )

    # Profile
    display_name = models.CharField(max_length=100, blank=True)
    avatar_url = models.URLField(blank=True)
    bio = models.TextField(blank=True)

    # Profile — demographics (for personalized onboarding + learning paths)
    industry = models.CharField(max_length=20, choices=Industry.choices, blank=True)
    role = models.CharField(max_length=20, choices=Role.choices, blank=True)
    experience_level = models.CharField(max_length=20, choices=ExperienceLevel.choices, blank=True)
    organization_size = models.CharField(max_length=20, choices=OrganizationSize.choices, blank=True)

    # Preferences (JSON blob for flexibility)
    preferences = models.JSONField(null=True, blank=True)  # theme, shortcuts, etc.

    # Analytics
    last_active_at = models.DateTimeField(null=True, blank=True)
    total_queries = models.IntegerField(default=0)
    total_tokens_used = models.BigIntegerField(default=0)

    # Halloween/seasonal (for your mockup!)
    current_theme = models.CharField(max_length=50, blank=True)  # "halloween", "winter", etc.

    # Onboarding
    onboarding_completed_at = models.DateTimeField(null=True, blank=True)

    # Email verification
    email_verified = models.BooleanField(default=False)
    email_verification_token = models.CharField(max_length=64, blank=True, db_index=True)
    email_opted_out = models.BooleanField(default=False)  # Unsubscribed from marketing/automation emails

    class Meta:
        db_table = "users"

    def __str__(self):
        return self.email or self.username

    @property
    def daily_limit(self) -> int:
        """Queries allowed per day based on tier."""
        return get_daily_limit(self.tier)

    @property
    def has_full_access(self) -> bool:
        """Check if user has paid tier with full feature access."""
        return is_paid_tier(self.tier)

    @property
    def has_ai_assistant(self) -> bool:
        """Check if user has access to Anthropic AI assistant (Enterprise only)."""
        return has_feature(self.tier, "ai_assistant")

    @property
    def can_collaborate(self) -> bool:
        """Check if user has team collaboration features."""
        return has_feature(self.tier, "collaboration")

    def can_query(self) -> bool:
        """Check if user can make another query."""
        from django.utils import timezone

        # Reset daily counter if needed
        if self.queries_reset_at is None or self.queries_reset_at < timezone.now():
            self.queries_today = 0
            self.queries_reset_at = timezone.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            self.save(update_fields=["queries_today", "queries_reset_at"])

        return self.queries_today < self.daily_limit

    def increment_queries(self):
        """Increment query count."""
        self.queries_today += 1
        self.total_queries += 1
        self.save(update_fields=["queries_today", "total_queries"])

    def generate_verification_token(self) -> str:
        """Generate and save a new email verification token."""
        self.email_verification_token = secrets.token_urlsafe(32)
        self.save(update_fields=["email_verification_token"])
        return self.email_verification_token

    def send_verification_email(self):
        """Send verification email to user."""
        from django.core.mail import send_mail
        from django.conf import settings as django_settings

        if not self.email:
            return False

        token = self.generate_verification_token()
        verify_url = f"https://svend.ai/verify?token={token}"

        send_mail(
            subject="Verify your SVEND account",
            message=f"""Welcome to SVEND!

Please verify your email by clicking this link:
{verify_url}

If you didn't create this account, you can ignore this email.

- The SVEND Team
""",
            from_email=django_settings.DEFAULT_FROM_EMAIL,
            recipient_list=[self.email],
            fail_silently=False,
        )
        return True

    def verify_email(self, token: str) -> bool:
        """Verify email with token. Returns True if successful."""
        if self.email_verification_token and self.email_verification_token == token:
            self.email_verified = True
            self.email_verification_token = ""
            self.save(update_fields=["email_verified", "email_verification_token"])
            return True
        return False

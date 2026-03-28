"""User models and billing."""

import secrets
import uuid
from datetime import timedelta

from django.contrib.auth.models import AbstractUser
from django.db import models

from core.encryption import EncryptedCharField, hash_token

from .constants import (
    ExperienceLevel,
    Industry,
    OrganizationSize,
    Role,
    Tier,
    get_daily_limit,
    has_feature,
    is_paid_tier,
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
    is_cancel_at_period_end = models.BooleanField(default=False, db_column="cancel_at_period_end")

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
        """Mark code as used by a user. Returns True if successful.
        Uses atomic F() update to prevent TOCTOU race condition (BUG-07).
        """
        from django.db.models import F

        # Atomic: only increment if still under max_uses
        updated = InviteCode.objects.filter(
            pk=self.pk,
            is_active=True,
            times_used__lt=self.max_uses,
        ).update(times_used=F("times_used") + 1)
        if not updated:
            return False
        self.refresh_from_db()
        self.used_by.add(user)
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


class LoginAttempt(models.Model):
    """Track failed login attempts for account lockout (SOC 2 CC6.1).

    Lockout policy: 5 failed attempts in any rolling window triggers a
    15-minute lockout for that username/email identifier.
    """

    MAX_ATTEMPTS = 5
    LOCKOUT_MINUTES = 15

    username = models.CharField(max_length=255, db_index=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    attempted_at = models.DateTimeField(auto_now_add=True, db_index=True)
    is_successful = models.BooleanField(default=False)

    class Meta:
        db_table = "login_attempts"

    @classmethod
    def is_locked_out(cls, username: str) -> bool:
        """Check if the username is currently locked out."""
        from django.utils import timezone

        window_start = timezone.now() - timedelta(minutes=cls.LOCKOUT_MINUTES)
        recent_failures = cls.objects.filter(
            username__iexact=username,
            is_successful=False,
            attempted_at__gte=window_start,
        ).count()
        return recent_failures >= cls.MAX_ATTEMPTS

    @classmethod
    def record(cls, username: str, ip_address=None, is_successful: bool = False):
        """Record a login attempt."""
        cls.objects.create(
            username=username[:255],
            ip_address=ip_address,
            is_successful=is_successful,
        )

    @classmethod
    def clear_on_success(cls, username: str):
        """Clear recent failed attempts after successful login."""
        from django.utils import timezone

        window_start = timezone.now() - timedelta(minutes=cls.LOCKOUT_MINUTES)
        cls.objects.filter(
            username__iexact=username,
            is_successful=False,
            attempted_at__gte=window_start,
        ).delete()


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

    # Complimentary access — partners, sponsors, ILSSI instructors, etc.
    # Gets full tier features but excluded from MRR calculations.
    is_complimentary = models.BooleanField(
        default=False,
        help_text="Partner/sponsor account — full tier access, excluded from MRR",
    )

    # Stripe (encrypted at rest, hash column for lookups)
    stripe_customer_id = EncryptedCharField(blank=True)
    stripe_customer_id_hash = models.CharField(max_length=64, blank=True, db_index=True)

    # Legacy fields (kept for backwards compat, use Subscription model instead)
    is_subscription_active = models.BooleanField(default=False, db_column="subscription_active")
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

    # Email verification (token stored as SHA-256 hash)
    is_email_verified = models.BooleanField(default=False, db_column="email_verified")
    email_verification_token = models.CharField(max_length=64, blank=True, db_index=True)
    email_verification_token_sent_at = models.DateTimeField(null=True, blank=True)
    is_email_opted_out = models.BooleanField(default=False, db_column="email_opted_out")

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
            self.queries_reset_at = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
                days=1
            )
            self.save(update_fields=["queries_today", "queries_reset_at"])

        return self.queries_today < self.daily_limit

    def increment_queries(self):
        """Increment query count atomically."""
        from django.db.models import F

        type(self).objects.filter(pk=self.pk).update(
            queries_today=F("queries_today") + 1,
            total_queries=F("total_queries") + 1,
        )
        self.refresh_from_db(fields=["queries_today", "total_queries"])

    def generate_verification_token(self) -> str:
        """Generate a new email verification token.

        Returns plaintext token (for the email link) but stores only
        a SHA-256 hash in the database. Token expires in 24 hours.
        """
        from django.utils import timezone

        plaintext = secrets.token_urlsafe(32)
        self.email_verification_token = hash_token(plaintext)
        self.email_verification_token_sent_at = timezone.now()
        self.save(
            update_fields=[
                "email_verification_token",
                "email_verification_token_sent_at",
            ]
        )
        return plaintext

    def send_verification_email(self):
        """Send verification email to user."""
        from django.conf import settings as django_settings
        from django.core.mail import send_mail

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
        """Verify email with token. Compares hash of input to stored hash.

        Token expires 24 hours after generation (SOC 2 CC6.2).
        """
        from django.utils import timezone

        if not self.email_verification_token:
            return False

        # Expiry check — 24 hours
        if self.email_verification_token_sent_at:
            age = timezone.now() - self.email_verification_token_sent_at
            if age.total_seconds() > 86400:
                return False

        if self.email_verification_token == hash_token(token):
            self.is_email_verified = True
            self.email_verification_token = ""
            self.email_verification_token_sent_at = None
            self.save(
                update_fields=[
                    "is_email_verified",
                    "email_verification_token",
                    "email_verification_token_sent_at",
                ]
            )
            return True
        return False


class DataExportRequest(models.Model):
    """Self-service data export request (PRIV-001 §5.1, SOC 2 P1.8).

    Tracks the lifecycle of a user's data export request from creation
    through async generation to download and expiry.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        EXPIRED = "expired", "Expired"
        CANCELLED = "cancelled", "Cancelled"

    class ExportFormat(models.TextChoices):
        JSON = "json", "JSON"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        "User",
        on_delete=models.CASCADE,
        related_name="data_export_requests",
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING,
    )
    export_format = models.CharField(
        max_length=10,
        choices=ExportFormat.choices,
        default=ExportFormat.JSON,
    )
    file_path = models.CharField(max_length=500, blank=True)
    file_size_bytes = models.BigIntegerField(null=True, blank=True)
    error_message = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    processing_started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    downloaded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "data_export_requests"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"], name="export_user_created"),
            models.Index(fields=["status", "expires_at"], name="export_status_expires"),
        ]

    def __str__(self):
        return f"Export {self.id} ({self.status})"


# ---------------------------------------------------------------------------
# API Key Authentication — SEC-001 §4.5
# ---------------------------------------------------------------------------

# Tier limits for active API keys
API_KEY_LIMITS = {
    Tier.FREE: 0,
    Tier.FOUNDER: 2,
    Tier.PRO: 5,
    Tier.TEAM: 10,
    Tier.ENTERPRISE: 50,
}


class APIKey(models.Model):
    """Platform-wide API key for programmatic access.

    ⚠ SECURITY-CRITICAL: Keys are bearer credentials. The raw key is returned
    exactly once at creation and never stored. Only the SHA-256 hash is persisted.
    All existing auth decorators work transparently — the credential resolver
    middleware sets request.user from the key.

    Standard: SEC-001 §4.5
    Compliance: SOC 2 CC6.1 (Logical Access Security)
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        "accounts.User",
        on_delete=models.CASCADE,
        related_name="api_keys",
    )

    # Key stored as hash — plaintext never persisted
    key_hash = models.CharField(max_length=64, unique=True, db_index=True)
    key_prefix = models.CharField(max_length=11)  # sv_ + 8 chars for display

    # Metadata
    name = models.CharField(max_length=100, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)

    # Lifecycle
    is_active = models.BooleanField(default=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    revoked_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "api_keys"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "is_active"], name="apikey_user_active"),
        ]

    def __str__(self):
        return f"{self.key_prefix}... ({self.name or 'unnamed'})"

    @property
    def is_expired(self):
        if self.expires_at is None:
            return False
        from django.utils import timezone

        return self.expires_at < timezone.now()

    @property
    def is_usable(self):
        return self.is_active and not self.is_expired and self.revoked_at is None

    def revoke(self):
        """Revoke this key. Immediate effect."""
        from django.utils import timezone

        self.is_active = False
        self.revoked_at = timezone.now()
        self.save(update_fields=["is_active", "revoked_at"])

    def bump_last_used(self):
        """Update last_used_at, throttled to 1h intervals to reduce DB writes."""
        from django.utils import timezone

        now = timezone.now()
        if self.last_used_at is None or (now - self.last_used_at) > timedelta(hours=1):
            type(self).objects.filter(pk=self.pk).update(last_used_at=now)

    @staticmethod
    def generate_raw_key():
        """Generate a raw API key with sv_ prefix."""
        return f"sv_{secrets.token_urlsafe(32)}"

    @classmethod
    def create_for_user(cls, user, name="", expires_at=None):
        """Create a new API key for a user.

        Returns (plaintext_key, api_key_instance). The plaintext is shown
        exactly once — if lost, revoke and create a new one.

        Raises ValueError if user has reached their tier's key limit.
        """
        # Check tier limit
        limit = API_KEY_LIMITS.get(user.tier, 0)
        active_count = cls.objects.filter(user=user, is_active=True, revoked_at__isnull=True).count()
        if active_count >= limit:
            raise ValueError(
                f"API key limit reached ({limit} for {user.tier} tier). Revoke an existing key or upgrade your plan."
            )

        plaintext = cls.generate_raw_key()
        key_obj = cls.objects.create(
            user=user,
            key_hash=hash_token(plaintext),
            key_prefix=plaintext[:11],  # sv_ + 8 chars
            name=name,
            expires_at=expires_at,
        )
        return plaintext, key_obj

    @classmethod
    def resolve(cls, raw_key):
        """Resolve a raw API key to its APIKey instance, or None.

        Returns None if the key is invalid, revoked, or expired.
        """
        from django.utils import timezone

        key_hash = hash_token(raw_key)
        try:
            api_key = cls.objects.select_related("user").get(
                key_hash=key_hash,
                is_active=True,
                revoked_at__isnull=True,
            )
        except cls.DoesNotExist:
            return None

        if api_key.expires_at and api_key.expires_at < timezone.now():
            return None

        api_key.bump_last_used()
        return api_key

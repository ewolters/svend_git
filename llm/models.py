"""LLM usage tracking and rate limiting.

Standard:     LLM-001 §3 (LLM Integration)
Compliance:   BILL-001 §4 (Tier-Based Model Selection)
"""

from django.conf import settings
from django.db import models


class LLMUsage(models.Model):
    """Track LLM API usage per user for rate limiting and cost control."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="llm_usage_records",
    )
    date = models.DateField(db_index=True)
    model = models.CharField(max_length=50)  # haiku, sonnet, opus
    request_count = models.IntegerField(default=0)
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)

    class Meta:
        db_table = "agents_api_llmusage"
        unique_together = ("user", "date", "model")
        indexes = [
            models.Index(fields=["user", "date"], name="agents_api__user_id_e0d904_idx"),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.date} - {self.model}: {self.request_count} requests"

    @classmethod
    def get_daily_usage(cls, user, date=None):
        """Get total requests for a user on a given date."""
        from django.utils import timezone

        if date is None:
            date = timezone.now().date()
        return cls.objects.filter(user=user, date=date).aggregate(
            total_requests=models.Sum("request_count"),
            total_input_tokens=models.Sum("input_tokens"),
            total_output_tokens=models.Sum("output_tokens"),
        )

    @classmethod
    def record_usage(cls, user, model, input_tokens=0, output_tokens=0):
        """Record an LLM request. Returns updated usage."""
        from django.utils import timezone

        today = timezone.now().date()

        usage, created = cls.objects.get_or_create(
            user=user,
            date=today,
            model=model,
            defaults={"request_count": 0, "input_tokens": 0, "output_tokens": 0},
        )

        from django.db.models import F

        cls.objects.filter(pk=usage.pk).update(
            request_count=F("request_count") + 1,
            input_tokens=F("input_tokens") + input_tokens,
            output_tokens=F("output_tokens") + output_tokens,
        )
        usage.refresh_from_db()

        return usage


# Rate limits by tier (requests per day)
LLM_RATE_LIMITS = {
    "FREE": 10,
    "FOUNDER": 50,
    "PRO": 200,
    "TEAM": 500,
    "ENTERPRISE": 10000,  # Effectively unlimited
}


def check_rate_limit(user):
    """Check if user is within their rate limit.

    Returns:
        (allowed: bool, remaining: int, limit: int)
    """
    from django.utils import timezone

    tier = getattr(user, "tier", "free") or "free"
    overrides = RateLimitOverride.get_overrides()
    if tier.upper() in overrides:
        limit = overrides[tier.upper()]["llm"]
    else:
        limit = LLM_RATE_LIMITS.get(tier.upper(), LLM_RATE_LIMITS["FREE"])

    usage = LLMUsage.get_daily_usage(user, timezone.now().date())
    current = usage["total_requests"] or 0

    return (current < limit, limit - current, limit)


class RateLimitOverride(models.Model):
    """Runtime-configurable rate limit overrides (staff-editable from dashboard)."""

    tier = models.CharField(max_length=20, unique=True)
    daily_llm_limit = models.PositiveIntegerField(help_text="Max LLM requests/day")
    daily_query_limit = models.PositiveIntegerField(help_text="Max query requests/day")
    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)

    class Meta:
        db_table = "agents_api_ratelimitoverride"

    def __str__(self):
        return f"{self.tier}: LLM={self.daily_llm_limit}, Query={self.daily_query_limit}"

    def save(self, *args, **kwargs):
        from django.core.cache import cache

        super().save(*args, **kwargs)
        cache.delete("rate_limit_overrides")

    @classmethod
    def get_overrides(cls):
        """Cache-backed lookup. Returns {tier: {llm: N, query: N}}."""
        from django.core.cache import cache

        key = "rate_limit_overrides"
        result = cache.get(key)
        if result is None:
            result = {o.tier: {"llm": o.daily_llm_limit, "query": o.daily_query_limit} for o in cls.objects.all()}
            cache.set(key, result, 300)
        return result

"""Unified tier definitions and limits for Svend.

All tier-related constants live here. Import from this module, not from models.

Pricing:
- FREE: $0/month - Trial tier, limited queries
- FOUNDER: $19/month - Legacy (existing users only, no longer sold)
- PRO: $49/month - Professional tier (individual seat)
- TEAM: $99/month per seat - Collaboration features
- ENTERPRISE: $299/month per seat - Full access + AI assistant + SSO
"""

from django.db import models


class Tier(models.TextChoices):
    """User subscription tiers.

    Used by User model and throughout the app for access control.
    """

    FREE = "free", "Free"
    FOUNDER = "founder", "Founder ($19/month)"  # Legacy — no longer sold
    PRO = "pro", "Professional ($49/month)"
    TEAM = "team", "Team ($99/month)"
    ENTERPRISE = "enterprise", "Enterprise ($299/month)"


# Maximum founder slots available
FOUNDER_SLOTS = 50


# Daily query limits by tier
TIER_LIMITS = {
    Tier.FREE: 5,  # Trial - basic DSW only
    Tier.FOUNDER: 50,  # Legacy, locked at $19 forever
    Tier.PRO: 50,  # Professional ($49/mo)
    Tier.TEAM: 200,  # Team ($99/mo) + collaboration
    Tier.ENTERPRISE: 1000,  # Enterprise ($299/mo) + AI + SSO
}


# Feature flags by tier
TIER_FEATURES = {
    Tier.FREE: {
        "basic_dsw": True,  # Basic decision science workbench
        "basic_ml": False,  # No ML model training
        "full_tools": False,  # Limited tool access
        "ai_assistant": False,  # No Anthropic
        "collaboration": False,  # No team features
        "forge_api": False,  # No synthetic data API
        "hoshin_kanri": False,  # No CI project management
        "priority_support": False,
    },
    Tier.FOUNDER: {
        "basic_dsw": True,
        "basic_ml": True,  # Basic ML (same as PRO)
        "full_tools": True,  # Most tools
        "ai_assistant": False,  # No Anthropic
        "collaboration": False,
        "forge_api": True,
        "hoshin_kanri": False,
        "priority_support": True,  # Legacy founders get priority
    },
    Tier.PRO: {
        "basic_dsw": True,
        "basic_ml": True,  # Basic ML model training
        "full_tools": True,  # Most tools
        "ai_assistant": False,  # No Anthropic
        "collaboration": False,
        "forge_api": True,
        "hoshin_kanri": False,
        "priority_support": False,
    },
    Tier.TEAM: {
        "basic_dsw": True,
        "basic_ml": True,
        "full_tools": True,
        "ai_assistant": False,  # Still no Anthropic
        "collaboration": True,  # Team collaboration
        "forge_api": True,
        "hoshin_kanri": False,
        "priority_support": True,
    },
    Tier.ENTERPRISE: {
        "basic_dsw": True,
        "basic_ml": True,
        "full_tools": True,
        "ai_assistant": True,  # Anthropic access
        "collaboration": True,
        "forge_api": True,
        "hoshin_kanri": True,  # Hoshin Kanri CI project management
        "priority_support": True,
    },
}


# Guest invite limits per board (by owner's tier)
GUEST_INVITE_LIMITS = {
    Tier.FREE: 0,
    Tier.FOUNDER: 2,
    Tier.PRO: 5,
    Tier.TEAM: 15,
    Tier.ENTERPRISE: 9999,
}

# Guest invite expiry in days (0 = permanent, stored as 100 years)
GUEST_INVITE_EXPIRY_DAYS = {
    Tier.FREE: 0,
    Tier.FOUNDER: 7,
    Tier.PRO: 7,
    Tier.TEAM: 0,
    Tier.ENTERPRISE: 0,
}


# --- User profile choices ---


class Industry(models.TextChoices):
    """User profile industry choices."""

    MANUFACTURING = "manufacturing", "Manufacturing"
    HEALTHCARE = "healthcare", "Healthcare"
    TECHNOLOGY = "technology", "Technology"
    FINANCE = "finance", "Finance"
    EDUCATION = "education", "Education"
    GOVERNMENT = "government", "Government"
    CONSULTING = "consulting", "Consulting"
    OTHER = "other", "Other"


class Role(models.TextChoices):
    """User profile role choices."""

    ENGINEER = "engineer", "Engineer"
    MANAGER = "manager", "Manager"
    ANALYST = "analyst", "Analyst"
    EXECUTIVE = "executive", "Executive"
    STUDENT = "student", "Student"
    CONSULTANT = "consultant", "Consultant"
    RESEARCHER = "researcher", "Researcher"
    OTHER = "other", "Other"


class ExperienceLevel(models.TextChoices):
    """User profile experience level choices."""

    BEGINNER = "beginner", "Beginner"
    INTERMEDIATE = "intermediate", "Intermediate"
    ADVANCED = "advanced", "Advanced"


class OrganizationSize(models.TextChoices):
    """User profile organization size choices."""

    SMALL = "small", "Small (1-50)"
    MEDIUM = "medium", "Medium (51-500)"
    LARGE = "large", "Large (501-5000)"
    ENTERPRISE = "enterprise", "Enterprise (5000+)"


def get_daily_limit(tier: str) -> int:
    """Get daily query limit for a tier."""
    return TIER_LIMITS.get(tier, TIER_LIMITS[Tier.FREE])


def has_feature(tier: str, feature: str) -> bool:
    """Check if a tier has a specific feature."""
    tier_features = TIER_FEATURES.get(tier, TIER_FEATURES[Tier.FREE])
    return tier_features.get(feature, False)


def is_paid_tier(tier: str) -> bool:
    """Check if tier is a paid tier (not FREE)."""
    return tier in (Tier.FOUNDER, Tier.PRO, Tier.TEAM, Tier.ENTERPRISE)


def can_use_anthropic(tier: str) -> bool:
    """Check if tier can use Anthropic models.

    All paid tiers get access — LLMManager selects model quality by tier:
    FOUNDER→haiku, PRO/TEAM→sonnet, ENTERPRISE→opus.
    """
    return is_paid_tier(tier)


def can_use_ml(tier: str) -> bool:
    """Check if tier can use ML model training."""
    return has_feature(tier, "basic_ml")


def can_use_tools(tier: str) -> bool:
    """Check if tier has full tool access."""
    return has_feature(tier, "full_tools")


def get_founder_availability() -> dict:
    """Check how many founder slots are available.

    Returns:
        {
            "total": 100,
            "used": 42,
            "remaining": 58,
            "available": True
        }
    """
    # Import here to avoid circular imports
    from django.contrib.auth import get_user_model

    User = get_user_model()

    used = User.objects.filter(tier=Tier.FOUNDER).count()
    remaining = max(0, FOUNDER_SLOTS - used)

    return {
        "total": FOUNDER_SLOTS,
        "used": used,
        "remaining": remaining,
        "available": remaining > 0,
    }

"""Unified tier definitions and limits for Svend.

All tier-related constants live here. Import from this module, not from models.

Pricing:
- FREE: $0/month - Trial tier, limited queries
- FOUNDER: $19/month - First 100 users, locked rate forever
- PRO: $29/month - Standard paid tier
- TEAM: $79/month - Collaboration features
- ENTERPRISE: $199/month - Full access + AI assistant
"""

from django.db import models


class Tier(models.TextChoices):
    """User subscription tiers.

    Used by User model and throughout the app for access control.
    """
    FREE = "free", "Free"
    FOUNDER = "founder", "Founder ($19/month)"
    PRO = "pro", "Pro ($29/month)"
    TEAM = "team", "Team ($79/month)"
    ENTERPRISE = "enterprise", "Enterprise ($199/month)"


# Maximum founder slots available
FOUNDER_SLOTS = 100


# Daily query limits by tier
TIER_LIMITS = {
    Tier.FREE: 5,           # Trial - basic DSW only
    Tier.FOUNDER: 50,       # Same as PRO, locked at $19 forever
    Tier.PRO: 50,           # Basic ML, most tools, no Anthropic
    Tier.TEAM: 200,         # + Collaboration, + priority
    Tier.ENTERPRISE: 1000,  # + Anthropic, + API access, + support
}


# Feature flags by tier
TIER_FEATURES = {
    Tier.FREE: {
        "basic_dsw": True,       # Basic decision science workbench
        "basic_ml": False,       # No ML model training
        "full_tools": False,     # Limited tool access
        "ai_assistant": False,   # No Anthropic
        "collaboration": False,  # No team features
        "forge_api": False,      # No synthetic data API
        "priority_support": False,
    },
    Tier.FOUNDER: {
        "basic_dsw": True,
        "basic_ml": True,        # Basic ML (same as PRO)
        "full_tools": True,      # Most tools
        "ai_assistant": False,   # No Anthropic
        "collaboration": False,
        "forge_api": True,
        "priority_support": True,  # Founders get priority as thanks
    },
    Tier.PRO: {
        "basic_dsw": True,
        "basic_ml": True,        # Basic ML model training
        "full_tools": True,      # Most tools
        "ai_assistant": False,   # No Anthropic
        "collaboration": False,
        "forge_api": True,
        "priority_support": False,
    },
    Tier.TEAM: {
        "basic_dsw": True,
        "basic_ml": True,
        "full_tools": True,
        "ai_assistant": False,   # Still no Anthropic
        "collaboration": True,   # Team collaboration
        "forge_api": True,
        "priority_support": True,
    },
    Tier.ENTERPRISE: {
        "basic_dsw": True,
        "basic_ml": True,
        "full_tools": True,
        "ai_assistant": True,    # Anthropic access
        "collaboration": True,
        "forge_api": True,
        "priority_support": True,
    },
}


# Stripe price IDs (configure in env or settings)
TIER_STRIPE_PRICES = {
    Tier.FOUNDER: "price_founder_19",  # Replace with actual Stripe price ID
    Tier.PRO: "price_pro_29",
    Tier.TEAM: "price_team_79",
    Tier.ENTERPRISE: "price_enterprise_199",
}


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
    """Check if tier can use Anthropic models (Enterprise only)."""
    return tier == Tier.ENTERPRISE


def can_use_ml(tier: str) -> bool:
    """Check if tier can use ML model training."""
    return has_feature(tier, "basic_ml")


def can_use_tools(tier: str) -> bool:
    """Check if tier has full tool access."""
    return has_feature(tier, "full_tools")

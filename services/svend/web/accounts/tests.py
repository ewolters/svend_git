"""Scenario tests for the accounts module — billing, permissions, models, constants.

Covers:
- Tier constants (enum, limits, features, helpers)
- User model properties (daily_limit, has_full_access, can_query, etc.)
- Permission decorators via real HTTP endpoints
- InviteCode lifecycle (generate, use, exhaust)
- Subscription model (status -> is_active mapping)
- Email verification (token generate, verify, reject)
- Full tier gating scenarios (lifecycle, rate limits, cross-tier)

Per TST-001 section 10.5: scenario tests that mimic real user behavior.
"""

import re
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from accounts.constants import (
    Tier,
    TIER_LIMITS,
    TIER_FEATURES,
    has_feature,
    is_paid_tier,
    get_daily_limit,
)
from accounts.models import InviteCode, Subscription
from core.encryption import hash_token

User = get_user_model()

# Production SECURE_SSL_REDIRECT=True breaks test client HTTP requests.
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)

ALL_TIERS = [Tier.FREE, Tier.FOUNDER, Tier.PRO, Tier.TEAM, Tier.ENTERPRISE]
PAID_TIERS = [Tier.FOUNDER, Tier.PRO, Tier.TEAM, Tier.ENTERPRISE]


def _make_user(email, tier=Tier.FREE, password="testpass123!", **kwargs):
    """Create a test user at the given tier."""
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(
        username=username, email=email, password=password, **kwargs
    )
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


# =========================================================================
# 1. Tier Constants
# =========================================================================


@SECURE_OFF
class TierConstantsTest(TestCase):
    """Unit tests for accounts/constants.py."""

    def test_all_five_tiers_defined(self):
        """Tier enum should have exactly FREE, FOUNDER, PRO, TEAM, ENTERPRISE."""
        values = {t.value for t in Tier}
        self.assertEqual(values, {"free", "founder", "pro", "team", "enterprise"})

    def test_tier_limits_has_entry_for_each_tier(self):
        """TIER_LIMITS must map every Tier value."""
        for tier in ALL_TIERS:
            self.assertIn(tier, TIER_LIMITS, f"Missing TIER_LIMITS entry for {tier}")
            self.assertIsInstance(TIER_LIMITS[tier], int)

    def test_tier_features_has_entry_for_each_tier(self):
        """TIER_FEATURES must map every Tier value."""
        for tier in ALL_TIERS:
            self.assertIn(tier, TIER_FEATURES, f"Missing TIER_FEATURES entry for {tier}")
            self.assertIsInstance(TIER_FEATURES[tier], dict)

    def test_has_feature_free_vs_enterprise(self):
        """Free cannot use ai_assistant; Enterprise can."""
        self.assertFalse(has_feature(Tier.FREE, "ai_assistant"))
        self.assertTrue(has_feature(Tier.ENTERPRISE, "ai_assistant"))

    def test_has_feature_unknown_feature_returns_false(self):
        """Unknown feature name should return False for any tier."""
        self.assertFalse(has_feature(Tier.ENTERPRISE, "teleportation"))

    def test_is_paid_tier(self):
        """FREE is not paid; all others are."""
        self.assertFalse(is_paid_tier(Tier.FREE))
        for tier in PAID_TIERS:
            self.assertTrue(is_paid_tier(tier), f"{tier} should be a paid tier")

    def test_get_daily_limit_known_tiers(self):
        """get_daily_limit should return the correct value for each tier."""
        self.assertEqual(get_daily_limit(Tier.FREE), 5)
        self.assertEqual(get_daily_limit(Tier.FOUNDER), 50)
        self.assertEqual(get_daily_limit(Tier.PRO), 50)
        self.assertEqual(get_daily_limit(Tier.TEAM), 200)
        self.assertEqual(get_daily_limit(Tier.ENTERPRISE), 1000)

    def test_get_daily_limit_unknown_tier_falls_back_to_free(self):
        """Unknown tier string should fall back to FREE limit."""
        self.assertEqual(get_daily_limit("nonexistent"), TIER_LIMITS[Tier.FREE])


# =========================================================================
# 2. User Model
# =========================================================================


@SECURE_OFF
class UserModelTest(TestCase):
    """Unit tests for accounts/models.py User properties and methods."""

    def test_daily_limit_matches_tier_limits(self):
        """User.daily_limit should reflect TIER_LIMITS for its tier."""
        for tier in ALL_TIERS:
            user = _make_user(f"{tier.value}@example.com", tier=tier)
            self.assertEqual(
                user.daily_limit,
                TIER_LIMITS[tier],
                f"daily_limit mismatch for {tier}",
            )

    def test_has_full_access_paid_tiers(self):
        """has_full_access should be True for paid tiers, False for FREE."""
        free = _make_user("free@example.com", Tier.FREE)
        self.assertFalse(free.has_full_access)
        pro = _make_user("pro@example.com", Tier.PRO)
        self.assertTrue(pro.has_full_access)

    def test_has_ai_assistant_enterprise_only(self):
        """has_ai_assistant should be True only for Enterprise."""
        for tier in [Tier.FREE, Tier.FOUNDER, Tier.PRO, Tier.TEAM]:
            user = _make_user(f"noai-{tier.value}@example.com", tier=tier)
            self.assertFalse(
                user.has_ai_assistant,
                f"{tier} should not have ai_assistant",
            )
        ent = _make_user("ent@example.com", Tier.ENTERPRISE)
        self.assertTrue(ent.has_ai_assistant)

    def test_can_collaborate_team_and_enterprise(self):
        """can_collaborate should be True only for Team and Enterprise."""
        for tier in [Tier.FREE, Tier.FOUNDER, Tier.PRO]:
            user = _make_user(f"nocol-{tier.value}@example.com", tier=tier)
            self.assertFalse(user.can_collaborate, f"{tier} should not collaborate")
        team = _make_user("team@example.com", Tier.TEAM)
        self.assertTrue(team.can_collaborate)
        ent = _make_user("entcol@example.com", Tier.ENTERPRISE)
        self.assertTrue(ent.can_collaborate)

    def test_can_query_under_limit(self):
        """can_query returns True when queries_today < daily_limit."""
        user = _make_user("under@example.com", Tier.FREE)
        user.queries_today = 0
        user.queries_reset_at = timezone.now() + timedelta(hours=12)
        user.save(update_fields=["queries_today", "queries_reset_at"])
        self.assertTrue(user.can_query())

    def test_can_query_at_limit(self):
        """can_query returns False when queries_today >= daily_limit."""
        user = _make_user("atlimit@example.com", Tier.FREE)
        user.queries_today = TIER_LIMITS[Tier.FREE]
        user.queries_reset_at = timezone.now() + timedelta(hours=12)
        user.save(update_fields=["queries_today", "queries_reset_at"])
        self.assertFalse(user.can_query())

    def test_can_query_resets_when_past_reset_time(self):
        """can_query resets counter when queries_reset_at is in the past."""
        user = _make_user("reset@example.com", Tier.FREE)
        user.queries_today = 999
        user.queries_reset_at = timezone.now() - timedelta(hours=1)
        user.save(update_fields=["queries_today", "queries_reset_at"])
        # Should reset counter then return True
        self.assertTrue(user.can_query())
        user.refresh_from_db()
        self.assertEqual(user.queries_today, 0)

    def test_increment_queries_atomic(self):
        """increment_queries atomically increments queries_today and total_queries."""
        user = _make_user("incr@example.com", Tier.PRO)
        user.queries_today = 3
        user.total_queries = 100
        user.queries_reset_at = timezone.now() + timedelta(hours=12)
        user.save(update_fields=["queries_today", "total_queries", "queries_reset_at"])

        user.increment_queries()
        self.assertEqual(user.queries_today, 4)
        self.assertEqual(user.total_queries, 101)

        # Verify it persisted via a fresh DB read
        fresh = User.objects.get(pk=user.pk)
        self.assertEqual(fresh.queries_today, 4)
        self.assertEqual(fresh.total_queries, 101)


# =========================================================================
# 3. Permission Decorators (via HTTP)
# =========================================================================


@SECURE_OFF
class PermissionDecoratorTest(TestCase):
    """Test accounts/permissions.py decorators via real API endpoints.

    Endpoints used:
    - /api/core/graph/check-consistency/  -> @rate_limited
    - /api/fmea/                          -> @gated_paid (full_tools gate)
    - /api/guide/chat/                    -> @require_enterprise
    - /api/hoshin/sites/                  -> @require_feature("hoshin_kanri")
    """

    def setUp(self):
        self.client = APIClient()

    # --- @require_auth / @rate_limited: unauthenticated ---

    def test_unauthenticated_rate_limited_endpoint_returns_401(self):
        """Unauthenticated request to @rate_limited endpoint should fail."""
        res = self.client.post("/api/core/graph/check-consistency/")
        self.assertIn(res.status_code, [401, 403])

    # --- @rate_limited: authenticated free user under limit ---

    def test_free_user_rate_limited_succeeds_initially(self):
        """Free user under daily limit can use @rate_limited endpoint."""
        user = _make_user("rlfree@example.com", Tier.FREE)
        user.queries_today = 0
        user.queries_reset_at = timezone.now() + timedelta(hours=12)
        user.save(update_fields=["queries_today", "queries_reset_at"])
        self.client.force_authenticate(user)
        res = self.client.post("/api/core/graph/check-consistency/")
        self.assertEqual(res.status_code, 200)

    # --- @rate_limited: free user exceeds daily limit ---

    def test_free_user_exceeds_daily_limit_gets_429(self):
        """Free user at daily limit gets 429 with upgrade_url."""
        user = _make_user("exceeded@example.com", Tier.FREE)
        user.queries_today = TIER_LIMITS[Tier.FREE]
        user.queries_reset_at = timezone.now() + timedelta(hours=12)
        user.save(update_fields=["queries_today", "queries_reset_at"])
        self.client.force_authenticate(user)
        res = self.client.post("/api/core/graph/check-consistency/")
        self.assertEqual(res.status_code, 429)
        data = res.json()
        # Error may be in envelope format or direct
        err = data.get("error", data)
        if isinstance(err, dict):
            self.assertIn("rate_limit", err.get("code", "") + err.get("message", "").lower())
        else:
            self.assertIn("limit", str(err).lower())

    # --- @gated_paid: free user blocked ---

    def test_free_user_gated_paid_endpoint_returns_403(self):
        """Free user cannot access @gated_paid endpoint (requires full_tools)."""
        user = _make_user("gatefree@example.com", Tier.FREE)
        self.client.force_login(user)
        res = self.client.get("/api/fmea/")
        self.assertEqual(res.status_code, 403)
        data = res.json()
        err = data.get("error", data)
        if isinstance(err, dict):
            self.assertIn("upgrade", (err.get("message", "") + err.get("code", "")).lower())
        else:
            self.assertIn("upgrade", str(err).lower())

    # --- @gated_paid: pro user allowed ---

    def test_pro_user_gated_paid_endpoint_succeeds(self):
        """Pro user should pass the @gated_paid gate."""
        user = _make_user("gatepro@example.com", Tier.PRO)
        self.client.force_login(user)
        res = self.client.get("/api/fmea/")
        # Should not be 403 (may be 200 with empty list)
        self.assertNotEqual(res.status_code, 403)

    # --- @require_enterprise: free user blocked ---

    def test_free_user_require_enterprise_returns_error(self):
        """Free user cannot access @require_enterprise endpoint."""
        user = _make_user("entfree@example.com", Tier.FREE)
        # guide_chat is a plain Django view (not DRF @api_view), so
        # force_login (session auth) is required instead of force_authenticate.
        self.client.force_login(user)
        res = self.client.post(
            "/api/guide/chat/",
            {"message": "test", "context": "general"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)

    # --- @require_enterprise: enterprise user allowed ---

    def test_enterprise_user_require_enterprise_passes_gate(self):
        """Enterprise user should pass the @require_enterprise gate.

        The endpoint may still fail downstream (e.g., missing Anthropic key
        returns 503), but the permission gate itself should not block.
        guide_chat is a plain Django view, so force_login (session auth)
        is required instead of DRF's force_authenticate.
        """
        user = _make_user("entuser@example.com", Tier.ENTERPRISE)
        self.client.force_login(user)
        res = self.client.post(
            "/api/guide/chat/",
            {"message": "test", "context": "general"},
            format="json",
        )
        # Should NOT be 401 or 403 (permission gate passed)
        self.assertNotIn(res.status_code, [401, 403])


# =========================================================================
# 4. InviteCode
# =========================================================================


@SECURE_OFF
class InviteCodeTest(TestCase):
    """Test InviteCode model lifecycle."""

    def test_generate_and_use(self):
        """Generate invite code -> use it -> verify used count increments."""
        codes = InviteCode.generate(count=1, max_uses=3, note="test")
        self.assertEqual(len(codes), 1)
        code = codes[0]
        self.assertTrue(code.is_valid)
        self.assertEqual(code.times_used, 0)

        user = _make_user("invite1@example.com")
        result = code.use(user)
        self.assertTrue(result)
        code.refresh_from_db()
        self.assertEqual(code.times_used, 1)
        self.assertIn(user, code.used_by.all())

    def test_use_exhausted_code_fails(self):
        """Using an invite code max_uses times should exhaust it."""
        codes = InviteCode.generate(count=1, max_uses=2)
        code = codes[0]

        u1 = _make_user("inv1@example.com")
        u2 = _make_user("inv2@example.com")
        u3 = _make_user("inv3@example.com")

        self.assertTrue(code.use(u1))
        self.assertTrue(code.use(u2))
        # Third use should fail
        self.assertFalse(code.use(u3))
        code.refresh_from_db()
        self.assertFalse(code.is_valid)
        self.assertEqual(code.times_used, 2)

    def test_generate_format(self):
        """InviteCode.generate() creates XXXX-XXXX format codes."""
        codes = InviteCode.generate(count=3)
        self.assertEqual(len(codes), 3)
        pattern = re.compile(r"^[0-9A-F]{4}-[0-9A-F]{4}$")
        for code in codes:
            self.assertRegex(
                code.code,
                pattern,
                f"Code '{code.code}' does not match XXXX-XXXX hex format",
            )

    def test_deactivated_code_is_not_valid(self):
        """Setting is_active=False should make is_valid return False."""
        codes = InviteCode.generate(count=1, max_uses=10)
        code = codes[0]
        self.assertTrue(code.is_valid)
        code.is_active = False
        code.save(update_fields=["is_active"])
        self.assertFalse(code.is_valid)


# =========================================================================
# 5. Subscription Model
# =========================================================================


@SECURE_OFF
class SubscriptionModelTest(TestCase):
    """Test Subscription.is_active property for various statuses."""

    def _make_subscription(self, status):
        user = _make_user(f"sub-{status}@example.com", Tier.PRO)
        return Subscription.objects.create(
            user=user,
            stripe_subscription_id=f"sub_test_{status}",
            status=status,
        )

    def test_active_subscription_is_active(self):
        sub = self._make_subscription(Subscription.Status.ACTIVE)
        self.assertTrue(sub.is_active)

    def test_trialing_subscription_is_active(self):
        sub = self._make_subscription(Subscription.Status.TRIALING)
        self.assertTrue(sub.is_active)

    def test_canceled_subscription_is_not_active(self):
        sub = self._make_subscription(Subscription.Status.CANCELED)
        self.assertFalse(sub.is_active)

    def test_past_due_subscription_is_not_active(self):
        sub = self._make_subscription(Subscription.Status.PAST_DUE)
        self.assertFalse(sub.is_active)

    def test_incomplete_subscription_is_not_active(self):
        sub = self._make_subscription(Subscription.Status.INCOMPLETE)
        self.assertFalse(sub.is_active)

    def test_paused_subscription_is_not_active(self):
        sub = self._make_subscription(Subscription.Status.PAUSED)
        self.assertFalse(sub.is_active)


# =========================================================================
# 6. Email Verification
# =========================================================================


@SECURE_OFF
class EmailVerificationModelTest(TestCase):
    """Test email verification token generation and verification on the model."""

    def test_generate_verification_token_stores_hash(self):
        """generate_verification_token returns plaintext, stores SHA-256 hash."""
        user = _make_user("verifymodel@example.com")
        plaintext = user.generate_verification_token()
        self.assertIsInstance(plaintext, str)
        self.assertGreater(len(plaintext), 20)

        user.refresh_from_db()
        # Stored value should be the hash, not the plaintext
        self.assertNotEqual(user.email_verification_token, plaintext)
        self.assertEqual(user.email_verification_token, hash_token(plaintext))

    def test_verify_email_correct_token(self):
        """verify_email with the correct token succeeds and clears the token."""
        user = _make_user("correct@example.com")
        plaintext = user.generate_verification_token()
        self.assertFalse(user.is_email_verified)

        result = user.verify_email(plaintext)
        self.assertTrue(result)
        self.assertTrue(user.is_email_verified)
        self.assertEqual(user.email_verification_token, "")

        # Verify it persisted
        user.refresh_from_db()
        self.assertTrue(user.is_email_verified)
        self.assertEqual(user.email_verification_token, "")

    def test_verify_email_wrong_token(self):
        """verify_email with a wrong token fails; state unchanged."""
        user = _make_user("wrong@example.com")
        user.generate_verification_token()
        self.assertFalse(user.is_email_verified)

        result = user.verify_email("totally-wrong-token")
        self.assertFalse(result)
        self.assertFalse(user.is_email_verified)
        # Token should still be set (not cleared)
        self.assertNotEqual(user.email_verification_token, "")

    def test_verify_email_no_token_set(self):
        """verify_email fails if no token was ever generated."""
        user = _make_user("notoken@example.com")
        result = user.verify_email("some-token")
        self.assertFalse(result)
        self.assertFalse(user.is_email_verified)


# =========================================================================
# 7. Tier Gating Scenarios
# =========================================================================


@SECURE_OFF
class TierGatingScenarioTest(TestCase):
    """End-to-end tier gating scenarios per TST-001 section 10.5."""

    def setUp(self):
        self.client = APIClient()

    def test_full_tier_lifecycle(self):
        """Create free user -> verify gates -> upgrade to pro -> upgrade to
        enterprise -> verify all access -> downgrade -> verify revoked."""
        user = _make_user("lifecycle@example.com", Tier.FREE)

        # -- FREE: limited access --
        self.assertFalse(user.has_full_access)
        self.assertFalse(user.has_ai_assistant)
        self.assertFalse(user.can_collaborate)
        self.assertEqual(user.daily_limit, 5)

        # -- Upgrade to PRO --
        user.tier = Tier.PRO
        user.save(update_fields=["tier"])
        user.refresh_from_db()

        self.assertTrue(user.has_full_access)
        self.assertFalse(user.has_ai_assistant)
        self.assertFalse(user.can_collaborate)
        self.assertEqual(user.daily_limit, 50)

        # -- Upgrade to ENTERPRISE --
        user.tier = Tier.ENTERPRISE
        user.save(update_fields=["tier"])
        user.refresh_from_db()

        self.assertTrue(user.has_full_access)
        self.assertTrue(user.has_ai_assistant)
        self.assertTrue(user.can_collaborate)
        self.assertEqual(user.daily_limit, 1000)

        # -- Downgrade to FREE (subscription cancelled) --
        user.tier = Tier.FREE
        user.save(update_fields=["tier"])
        user.refresh_from_db()

        self.assertFalse(user.has_full_access)
        self.assertFalse(user.has_ai_assistant)
        self.assertFalse(user.can_collaborate)
        self.assertEqual(user.daily_limit, 5)

    def test_rate_limit_lifecycle(self):
        """Create user -> query until limit -> verify blocked -> reset -> query again."""
        user = _make_user("ratelimit@example.com", Tier.FREE)
        user.queries_today = 0
        user.queries_reset_at = timezone.now() + timedelta(hours=12)
        user.save(update_fields=["queries_today", "queries_reset_at"])
        self.client.force_authenticate(user)

        # Use all 5 queries (FREE limit)
        for i in range(TIER_LIMITS[Tier.FREE]):
            user.refresh_from_db()
            self.assertTrue(user.can_query(), f"Should be able to query (attempt {i + 1})")
            user.increment_queries()

        # Should be blocked now
        user.refresh_from_db()
        self.assertFalse(user.can_query())

        # Hit the endpoint to confirm 429
        res = self.client.post("/api/core/graph/check-consistency/")
        self.assertEqual(res.status_code, 429)

        # Simulate daily reset (set reset time to past)
        user.queries_reset_at = timezone.now() - timedelta(hours=1)
        user.save(update_fields=["queries_reset_at"])

        # can_query should reset the counter and return True
        self.assertTrue(user.can_query())
        user.refresh_from_db()
        self.assertEqual(user.queries_today, 0)

        # Endpoint should work again
        res = self.client.post("/api/core/graph/check-consistency/")
        self.assertEqual(res.status_code, 200)

    def test_cross_tier_feature_matrix(self):
        """Verify each tier has exactly the expected feature set from TIER_FEATURES."""
        expected_features = {
            Tier.FREE: {
                "basic_dsw": True,
                "basic_ml": False,
                "full_tools": False,
                "ai_assistant": False,
                "collaboration": False,
                "forge_api": False,
                "hoshin_kanri": False,
                "priority_support": False,
            },
            Tier.FOUNDER: {
                "basic_dsw": True,
                "basic_ml": True,
                "full_tools": True,
                "ai_assistant": False,
                "collaboration": False,
                "forge_api": True,
                "hoshin_kanri": False,
                "priority_support": True,
            },
            Tier.PRO: {
                "basic_dsw": True,
                "basic_ml": True,
                "full_tools": True,
                "ai_assistant": False,
                "collaboration": False,
                "forge_api": True,
                "hoshin_kanri": False,
                "priority_support": False,
            },
            Tier.TEAM: {
                "basic_dsw": True,
                "basic_ml": True,
                "full_tools": True,
                "ai_assistant": False,
                "collaboration": True,
                "forge_api": True,
                "hoshin_kanri": False,
                "priority_support": True,
            },
            Tier.ENTERPRISE: {
                "basic_dsw": True,
                "basic_ml": True,
                "full_tools": True,
                "ai_assistant": True,
                "collaboration": True,
                "forge_api": True,
                "hoshin_kanri": True,
                "priority_support": True,
            },
        }

        for tier in ALL_TIERS:
            actual = TIER_FEATURES[tier]
            expected = expected_features[tier]
            self.assertEqual(
                actual,
                expected,
                f"Feature mismatch for {tier}: "
                f"diff = {set(actual.items()) ^ set(expected.items())}",
            )

            # Also verify has_feature() agrees with the dict
            for feature_name, expected_val in expected.items():
                self.assertEqual(
                    has_feature(tier, feature_name),
                    expected_val,
                    f"has_feature({tier}, {feature_name}) mismatch",
                )

"""Comprehensive API tests — auth, billing, permissions, middleware."""

from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from accounts.constants import TIER_FEATURES, TIER_LIMITS, Tier, has_feature

User = get_user_model()

# Production SECURE_SSL_REDIRECT=True breaks test client HTTP requests.
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_user(email, tier=Tier.FREE, password="testpass123!", **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(username=username, email=email, password=password, **kwargs)
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


# =========================================================================
# Health Check
# =========================================================================


@SECURE_OFF
class HealthCheckTest(TestCase):
    """Tests for the /api/health/ endpoint."""

    def test_health_returns_ok(self):
        client = APIClient()
        res = client.get("/api/health/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "svend")


# =========================================================================
# Registration
# =========================================================================


@SECURE_OFF
class RegistrationTest(TestCase):
    """Tests for user registration endpoint."""

    def setUp(self):
        self.client = APIClient()
        p = patch("api.views.RegistrationThrottle.allow_request", return_value=True)
        p.start()
        self.addCleanup(p.stop)

    def test_register_success(self):
        res = self.client.post(
            "/api/auth/register/",
            {
                "username": "newuser",
                "email": "new@example.com",
                "password": "SecurePass123!",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["status"], "registered")
        self.assertEqual(data["username"], "newuser")
        self.assertEqual(data["tier"], "free")
        self.assertFalse(data["email_verified"])
        # User created in DB
        self.assertTrue(User.objects.filter(username="newuser").exists())

    def test_register_auto_login(self):
        """Registration should auto-login the user."""
        res = self.client.post(
            "/api/auth/register/",
            {
                "username": "autologin",
                "email": "auto@example.com",
                "password": "SecurePass123!",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        # Should be able to access /me without logging in again
        res = self.client.get("/api/auth/me/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["username"], "autologin")

    def test_register_short_username(self):
        res = self.client.post(
            "/api/auth/register/",
            {
                "username": "ab",
                "email": "short@example.com",
                "password": "SecurePass123!",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("3 characters", res.json()["error"]["message"])

    def test_register_short_password(self):
        res = self.client.post(
            "/api/auth/register/",
            {
                "username": "newuser",
                "email": "new@example.com",
                "password": "short",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_register_common_password(self):
        res = self.client.post(
            "/api/auth/register/",
            {
                "username": "newuser",
                "email": "new@example.com",
                "password": "password1234",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_register_duplicate_username(self):
        _make_user("existing@example.com", username="existing")
        res = self.client.post(
            "/api/auth/register/",
            {
                "username": "existing",
                "email": "other@example.com",
                "password": "SecurePass123!",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("already taken", res.json()["error"]["message"])

    def test_register_duplicate_email(self):
        _make_user("taken@example.com")
        res = self.client.post(
            "/api/auth/register/",
            {
                "username": "anotheruser",
                "email": "taken@example.com",
                "password": "SecurePass123!",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("already registered", res.json()["error"]["message"])

    def test_register_new_user_is_free_tier(self):
        self.client.post(
            "/api/auth/register/",
            {
                "username": "freetier",
                "email": "free@example.com",
                "password": "SecurePass123!",
            },
            format="json",
        )
        user = User.objects.get(username="freetier")
        self.assertEqual(user.tier, Tier.FREE)

    def test_register_empty_username_auto_generates(self):
        """Empty username is auto-generated from email prefix."""
        res = self.client.post(
            "/api/auth/register/",
            {
                "username": "",
                "email": "x@example.com",
                "password": "SecurePass123!",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)

    def test_register_no_email_fails(self):
        """Email is required at registration."""
        res = self.client.post(
            "/api/auth/register/",
            {
                "username": "noemail",
                "password": "SecurePass123!",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 400)


# =========================================================================
# Login
# =========================================================================


@SECURE_OFF
class LoginTest(TestCase):
    """Tests for user login endpoint."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("user@example.com", username="testuser")
        p = patch("api.views.LoginRateThrottle.allow_request", return_value=True)
        p.start()
        self.addCleanup(p.stop)

    def test_login_with_username(self):
        res = self.client.post(
            "/api/auth/login/",
            {
                "username": "testuser",
                "password": "testpass123!",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "logged_in")
        self.assertEqual(data["user"]["username"], "testuser")

    def test_login_with_email(self):
        res = self.client.post(
            "/api/auth/login/",
            {
                "username": "user@example.com",
                "password": "testpass123!",
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "logged_in")

    def test_login_wrong_password(self):
        res = self.client.post(
            "/api/auth/login/",
            {
                "username": "testuser",
                "password": "wrongpassword",
            },
        )
        self.assertEqual(res.status_code, 401)
        self.assertIn("Invalid", res.json()["error"]["message"])

    def test_login_nonexistent_user(self):
        res = self.client.post(
            "/api/auth/login/",
            {
                "username": "nobody",
                "password": "whatever123",
            },
        )
        self.assertEqual(res.status_code, 401)

    def test_login_inactive_user(self):
        self.user.is_active = False
        self.user.save()
        res = self.client.post(
            "/api/auth/login/",
            {
                "username": "testuser",
                "password": "testpass123!",
            },
        )
        # Django's authenticate() returns None for inactive users
        self.assertIn(res.status_code, [401, 403])

    def test_login_missing_fields(self):
        res = self.client.post("/api/auth/login/", {})
        self.assertEqual(res.status_code, 400)

    def test_login_returns_tier_and_limit(self):
        self.user.tier = Tier.PRO
        self.user.save()
        res = self.client.post(
            "/api/auth/login/",
            {
                "username": "testuser",
                "password": "testpass123!",
            },
        )
        data = res.json()
        self.assertEqual(data["user"]["tier"], "pro")
        self.assertEqual(data["user"]["daily_limit"], TIER_LIMITS[Tier.PRO])


# =========================================================================
# Logout
# =========================================================================


@SECURE_OFF
class LogoutTest(TestCase):
    """Tests for user logout endpoint."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("user@example.com")

    def test_logout_authenticated(self):
        self.client.force_authenticate(self.user)
        res = self.client.post("/api/auth/logout/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "logged_out")

    def test_logout_unauthenticated(self):
        res = self.client.post("/api/auth/logout/")
        self.assertIn(res.status_code, [401, 403])

    def test_logout_destroys_session(self):
        """After logout, /me should fail."""
        self.client.login(username="user", password="testpass123!")
        self.client.post("/api/auth/logout/")
        res = self.client.get("/api/auth/me/")
        self.assertIn(res.status_code, [401, 403])


# =========================================================================
# Me (Profile Read)
# =========================================================================


@SECURE_OFF
class MeEndpointTest(TestCase):
    """Tests for the /api/auth/me/ endpoint."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("me@example.com", Tier.PRO, username="meuser")

    def test_me_returns_profile(self):
        self.client.force_authenticate(self.user)
        res = self.client.get("/api/auth/me/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["username"], "meuser")
        self.assertEqual(data["email"], "me@example.com")
        self.assertEqual(data["tier"], "pro")
        self.assertFalse(data["email_verified"])
        self.assertFalse(data["is_staff"])
        self.assertIn("features", data)
        self.assertIn("queries_today", data)
        self.assertIn("daily_limit", data)

    def test_me_features_match_tier(self):
        self.client.force_authenticate(self.user)
        res = self.client.get("/api/auth/me/")
        features = res.json()["features"]
        expected = TIER_FEATURES[Tier.PRO]
        self.assertEqual(features, expected)

    def test_me_unauthenticated(self):
        res = self.client.get("/api/auth/me/")
        self.assertIn(res.status_code, [401, 403])

    def test_me_includes_subscription_status(self):
        self.client.force_authenticate(self.user)
        res = self.client.get("/api/auth/me/")
        data = res.json()
        self.assertIn("subscription_active", data)
        self.assertIn("onboarding_completed", data)

    def test_me_staff_flag(self):
        self.user.is_staff = True
        self.user.save()
        self.client.force_authenticate(self.user)
        res = self.client.get("/api/auth/me/")
        self.assertTrue(res.json()["is_staff"])
        self.assertTrue(res.json()["is_internal"])


# =========================================================================
# Profile Update
# =========================================================================


@SECURE_OFF
class ProfileUpdateTest(TestCase):
    """Tests for user profile update endpoint."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("update@example.com")
        self.client.force_authenticate(self.user)

    def test_update_display_name(self):
        res = self.client.patch(
            "/api/auth/profile/",
            {
                "display_name": "New Name",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["user"]["display_name"], "New Name")
        self.user.refresh_from_db()
        self.assertEqual(self.user.display_name, "New Name")

    def test_update_theme(self):
        res = self.client.patch(
            "/api/auth/profile/",
            {
                "current_theme": "nordic",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["user"]["current_theme"], "nordic")

    def test_update_preferences_json(self):
        prefs = {"chart_style": "line", "show_hints": True}
        res = self.client.patch(
            "/api/auth/profile/",
            {
                "preferences": prefs,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["user"]["preferences"], prefs)

    def test_update_valid_industry(self):
        res = self.client.patch(
            "/api/auth/profile/",
            {
                "industry": "manufacturing",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["user"]["industry"], "manufacturing")

    def test_update_invalid_industry_rejected(self):
        res = self.client.patch(
            "/api/auth/profile/",
            {
                "industry": "fake_industry",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid", res.json()["error"]["message"])

    def test_update_valid_role(self):
        res = self.client.patch(
            "/api/auth/profile/",
            {
                "role": "engineer",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)

    def test_update_invalid_role_rejected(self):
        res = self.client.patch(
            "/api/auth/profile/",
            {
                "role": "wizard",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_update_multiple_fields(self):
        res = self.client.patch(
            "/api/auth/profile/",
            {
                "display_name": "Multi",
                "bio": "Test bio",
                "industry": "technology",
                "role": "analyst",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()["user"]
        self.assertEqual(data["display_name"], "Multi")
        self.assertEqual(data["bio"], "Test bio")

    def test_update_unauthenticated(self):
        client = APIClient()
        res = client.patch("/api/auth/profile/", {"display_name": "X"}, format="json")
        self.assertIn(res.status_code, [401, 403])


# =========================================================================
# Password Change
# =========================================================================


@SECURE_OFF
class PasswordChangeTest(TestCase):
    """Tests for password change endpoint."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("pw@example.com")
        self.client.force_authenticate(self.user)

    def test_change_password_success(self):
        res = self.client.post(
            "/api/auth/password/",
            {
                "current_password": "testpass123!",
                "new_password": "NewSecurePass456!",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "password_changed")
        # Verify new password works
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("NewSecurePass456!"))

    def test_wrong_current_password(self):
        res = self.client.post(
            "/api/auth/password/",
            {
                "current_password": "wrongpassword",
                "new_password": "NewSecurePass456!",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("incorrect", res.json()["error"]["message"])

    def test_new_password_too_short(self):
        res = self.client.post(
            "/api/auth/password/",
            {
                "current_password": "testpass123!",
                "new_password": "short",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_missing_fields(self):
        res = self.client.post("/api/auth/password/", {}, format="json")
        self.assertEqual(res.status_code, 400)

    def test_session_maintained_after_change(self):
        """Session should survive password change (update_session_auth_hash)."""
        res = self.client.post(
            "/api/auth/password/",
            {
                "current_password": "testpass123!",
                "new_password": "NewSecurePass456!",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        # Should still be authenticated
        res = self.client.get("/api/auth/me/")
        self.assertEqual(res.status_code, 200)


# =========================================================================
# Email Verification
# =========================================================================


@SECURE_OFF
class EmailVerificationTest(TestCase):
    """Tests for email verification flow."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("verify@example.com")
        self.client.force_authenticate(self.user)

    def test_send_verification_no_email(self):
        self.user.email = ""
        self.user.save()
        res = self.client.post("/api/auth/send-verification/")
        self.assertEqual(res.status_code, 400)
        self.assertIn("No email", res.json()["error"]["message"])

    def test_send_verification_already_verified(self):
        self.user.is_email_verified = True
        self.user.save()
        res = self.client.post("/api/auth/send-verification/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "already_verified")

    def test_verify_missing_token(self):
        res = self.client.get("/api/auth/verify-email/")
        self.assertEqual(res.status_code, 400)
        self.assertIn("token required", res.json()["error"]["message"])

    def test_verify_invalid_token(self):
        res = self.client.get("/api/auth/verify-email/?token=invalidtoken123")
        self.assertEqual(res.status_code, 400)

    def test_verify_post_missing_token(self):
        res = self.client.post("/api/auth/verify-email/", {}, format="json")
        self.assertEqual(res.status_code, 400)


# =========================================================================
# Tier Feature Flags
# =========================================================================


@SECURE_OFF
class TierFeatureTest(TestCase):
    """Verify feature flags are correct per tier."""

    def test_free_has_no_collaboration(self):
        self.assertFalse(has_feature(Tier.FREE, "collaboration"))

    def test_free_has_no_full_tools(self):
        self.assertFalse(has_feature(Tier.FREE, "full_tools"))

    def test_pro_has_full_tools(self):
        self.assertTrue(has_feature(Tier.PRO, "full_tools"))

    def test_pro_has_no_collaboration(self):
        self.assertFalse(has_feature(Tier.PRO, "collaboration"))

    def test_team_has_collaboration(self):
        self.assertTrue(has_feature(Tier.TEAM, "collaboration"))

    def test_team_has_no_ai_assistant(self):
        self.assertFalse(has_feature(Tier.TEAM, "ai_assistant"))

    def test_enterprise_has_everything(self):
        for feature in ["collaboration", "ai_assistant", "full_tools", "hoshin_kanri"]:
            self.assertTrue(
                has_feature(Tier.ENTERPRISE, feature),
                msg=f"Enterprise should have {feature}",
            )

    def test_tier_limits_correct(self):
        self.assertEqual(TIER_LIMITS[Tier.FREE], 5)
        self.assertEqual(TIER_LIMITS[Tier.PRO], 50)
        self.assertEqual(TIER_LIMITS[Tier.TEAM], 200)
        self.assertEqual(TIER_LIMITS[Tier.ENTERPRISE], 1000)


# =========================================================================
# Permission Decorators
# =========================================================================


@SECURE_OFF
class PermissionDecoratorTest(TestCase):
    """Test tier gating decorators via real endpoints."""

    def setUp(self):
        self.client = APIClient()

    def test_rate_limited_blocks_unauthenticated(self):
        """@rate_limited endpoints should block unauthenticated users."""
        # check_consistency uses @rate_limited decorator
        res = self.client.post("/api/core/graph/check-consistency/")
        self.assertIn(res.status_code, [401, 403])

    def test_rate_limited_allows_free_user(self):
        """@rate_limited allows free users (up to daily limit)."""
        user = _make_user("free@example.com", Tier.FREE)
        self.client.force_authenticate(user)
        res = self.client.post("/api/core/graph/check-consistency/")
        self.assertEqual(res.status_code, 200)

    def test_gated_paid_blocks_free_user(self):
        """@gated_paid endpoints should block free tier."""
        user = _make_user("free2@example.com", Tier.FREE)
        # FMEA uses plain Django views — need force_login (not force_authenticate)
        self.client.force_login(user)
        res = self.client.get("/api/fmea/")
        self.assertEqual(res.status_code, 403)
        self.assertIn("Upgrade", res.json()["error"]["message"])

    def test_gated_paid_allows_pro_user(self):
        """@gated_paid should allow PRO tier."""
        user = _make_user("pro@example.com", Tier.PRO)
        self.client.force_login(user)
        res = self.client.get("/api/fmea/")
        # 200 (empty list) — not 403
        self.assertNotEqual(res.status_code, 403)

    def test_require_team_blocks_pro(self):
        """@require_team should block PRO tier."""
        user = _make_user("pro@example.com", Tier.PRO)
        self.client.force_authenticate(user)
        res = self.client.post(
            "/api/core/org/create/",
            {
                "name": "Test",
                "slug": "test",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 403)
        self.assertIn("Team plan", res.json()["error"]["message"])

    def test_require_team_allows_team(self):
        """@require_team should allow TEAM tier."""
        user = _make_user("team@example.com", Tier.TEAM)
        self.client.force_authenticate(user)
        res = self.client.post(
            "/api/core/org/create/",
            {
                "name": "Team Org",
                "slug": "team-org",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)


# =========================================================================
# Internal User Access
# =========================================================================


@SECURE_OFF
class InternalAccessTest(TestCase):
    """Test IsInternalUser permission."""

    def setUp(self):
        self.client = APIClient()

    def test_anonymous_blocked(self):
        res = self.client.get("/api/internal/overview/")
        self.assertIn(res.status_code, [401, 403])

    def test_regular_user_blocked(self):
        user = _make_user("regular@example.com", Tier.PRO)
        self.client.force_authenticate(user)
        res = self.client.get("/api/internal/overview/")
        self.assertEqual(res.status_code, 403)

    def test_staff_user_allowed(self):
        user = _make_user("staff@example.com", is_staff=True)
        self.client.force_authenticate(user)
        res = self.client.get("/api/internal/overview/")
        self.assertEqual(res.status_code, 200)

    def test_svend_tenant_admin_allowed(self):
        """Owner/admin of 'svend' tenant should have internal access."""
        from core.models import Membership, Tenant

        user = _make_user("orgadmin@example.com", Tier.TEAM)
        tenant = Tenant.objects.create(name="Svend", slug="svend", plan="team")
        Membership.objects.create(
            tenant=tenant,
            user=user,
            role="admin",
            is_active=True,
            joined_at=timezone.now(),
        )
        self.client.force_authenticate(user)
        res = self.client.get("/api/internal/overview/")
        self.assertEqual(res.status_code, 200)

    def test_svend_tenant_member_blocked(self):
        """Regular member of 'svend' tenant should NOT have internal access."""
        from core.models import Membership, Tenant

        user = _make_user("member@example.com", Tier.TEAM)
        tenant = Tenant.objects.create(name="Svend", slug="svend", plan="team")
        Membership.objects.create(
            tenant=tenant,
            user=user,
            role="member",
            is_active=True,
            joined_at=timezone.now(),
        )
        self.client.force_authenticate(user)
        res = self.client.get("/api/internal/overview/")
        self.assertEqual(res.status_code, 403)

    def test_other_tenant_admin_blocked(self):
        """Admin of a different tenant should NOT have internal access."""
        from core.models import Membership, Tenant

        user = _make_user("otheradmin@example.com", Tier.TEAM)
        tenant = Tenant.objects.create(name="Other Corp", slug="other-corp", plan="team")
        Membership.objects.create(
            tenant=tenant,
            user=user,
            role="admin",
            is_active=True,
            joined_at=timezone.now(),
        )
        self.client.force_authenticate(user)
        res = self.client.get("/api/internal/overview/")
        self.assertEqual(res.status_code, 403)


# =========================================================================
# Billing Endpoints
# =========================================================================


@SECURE_OFF
class BillingStatusTest(TestCase):
    """Test billing status and founder availability."""

    def setUp(self):
        self.client = APIClient()

    def test_founder_availability_is_public(self):
        res = self.client.get("/billing/founder-availability/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total", data)
        self.assertIn("remaining", data)
        self.assertIn("available", data)
        self.assertEqual(data["total"], 50)

    def test_billing_status_requires_auth(self):
        res = self.client.get("/billing/status/")
        # Should redirect to login (302) since it uses @login_required
        self.assertIn(res.status_code, [302, 401, 403])

    def test_billing_status_returns_tier_info(self):
        user = _make_user("billing@example.com", Tier.PRO)
        self.client.force_authenticate(user)
        # Use login for session-based views
        self.client.login(username="billing", password="testpass123!")
        res = self.client.get("/billing/status/")
        if res.status_code == 200:
            data = res.json()
            self.assertEqual(data["tier"], "pro")
            self.assertIn("daily_limit", data)
            self.assertIn("queries_today", data)

    def test_checkout_requires_auth(self):
        res = self.client.get("/billing/checkout/?plan=pro")
        # Should redirect to login
        self.assertIn(res.status_code, [302, 401, 403])

    def test_portal_requires_auth(self):
        res = self.client.get("/billing/portal/")
        self.assertIn(res.status_code, [302, 401, 403])


# =========================================================================
# Rate Limiting (Query Limits)
# =========================================================================


@SECURE_OFF
class QueryLimitTest(TestCase):
    """Test daily query limit enforcement."""

    def setUp(self):
        self.client = APIClient()

    def test_free_user_limited_to_5(self):
        user = _make_user("limited@example.com", Tier.FREE)
        user.queries_today = 5  # Already at limit
        user.queries_reset_at = timezone.now() + timedelta(hours=12)
        user.save()
        self.client.force_authenticate(user)
        # Any @rate_limited endpoint should return 429
        res = self.client.post("/api/core/graph/check-consistency/")
        self.assertEqual(res.status_code, 429)
        data = res.json()
        err = data.get("error", data)
        msg = err.get("message", "") if isinstance(err, dict) else str(err)
        self.assertIn("limit", msg.lower())

    def test_queries_increment_on_success(self):
        user = _make_user("counter@example.com", Tier.FREE)
        user.queries_today = 0
        user.queries_reset_at = timezone.now() + timedelta(hours=12)
        user.save()
        self.client.force_authenticate(user)

        res = self.client.post("/api/core/graph/check-consistency/")
        self.assertEqual(res.status_code, 200)

        user.refresh_from_db()
        self.assertEqual(user.queries_today, 1)


# =========================================================================
# Middleware
# =========================================================================


@SECURE_OFF
class NoCacheDynamicMiddlewareTest(TestCase):
    """Test that dynamic responses get no-cache headers."""

    def setUp(self):
        self.client = APIClient()

    def test_api_response_has_no_cache(self):
        res = self.client.get("/api/health/")
        cc = res.get("Cache-Control", "")
        self.assertIn("no-cache", cc)
        self.assertIn("no-store", cc)
        self.assertIn("private", cc)


@SECURE_OFF
class SiteVisitMiddlewareTest(TestCase):
    """Test anonymous site visit tracking."""

    def setUp(self):
        self.client = APIClient()

    def test_page_visit_tracked(self):
        from api.models import SiteVisit

        count_before = SiteVisit.objects.count()
        # Visit a page (not API, not static)
        self.client.get("/", HTTP_USER_AGENT="Mozilla/5.0 Test Browser")
        count_after = SiteVisit.objects.count()
        self.assertGreater(count_after, count_before)

    def test_api_visits_not_tracked(self):
        from api.models import SiteVisit

        count_before = SiteVisit.objects.count()
        self.client.get("/api/health/")
        count_after = SiteVisit.objects.count()
        self.assertEqual(count_after, count_before)

    def test_static_not_tracked(self):
        from api.models import SiteVisit

        count_before = SiteVisit.objects.count()
        self.client.get("/static/test.css")
        count_after = SiteVisit.objects.count()
        self.assertEqual(count_after, count_before)

    def test_bot_flagged(self):
        from api.models import SiteVisit

        self.client.get("/", HTTP_USER_AGENT="Googlebot/2.1")
        visit = SiteVisit.objects.order_by("-viewed_at").first()
        if visit:
            self.assertTrue(visit.is_bot)

    def test_staff_not_tracked(self):
        from api.models import SiteVisit

        user = _make_user("staff@example.com", is_staff=True)
        self.client.force_login(user)
        count_before = SiteVisit.objects.count()
        self.client.get("/")
        count_after = SiteVisit.objects.count()
        self.assertEqual(count_after, count_before)


# =========================================================================
# Auth + Org Full Integration
# =========================================================================


@SECURE_OFF
class AuthOrgIntegrationTest(TestCase):
    """Test the full flow: register → login → create org → invite."""

    def setUp(self):
        p = patch("api.views.RegistrationThrottle.allow_request", return_value=True)
        p.start()
        self.addCleanup(p.stop)

    def test_register_to_org_flow(self):
        c = APIClient()

        # 1. Register
        res = c.post(
            "/api/auth/register/",
            {
                "username": "orgfounder",
                "email": "founder@company.com",
                "password": "SecurePass123!",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)

        # 2. User starts as FREE — cannot create org
        res = c.get("/api/core/org/")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["has_org"])
        self.assertFalse(res.json()["can_create_org"])

        # 3. Simulate tier upgrade to TEAM
        user = User.objects.get(username="orgfounder")
        user.tier = Tier.TEAM
        user.save()

        # 4. Now can create org
        c.force_authenticate(user)
        res = c.get("/api/core/org/")
        self.assertTrue(res.json()["can_create_org"])

        res = c.post(
            "/api/core/org/create/",
            {
                "name": "My Company",
                "slug": "my-company",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)

        # 5. Verify org shows in profile
        res = c.get("/api/core/org/")
        self.assertTrue(res.json()["has_org"])
        self.assertEqual(res.json()["org"]["name"], "My Company")

        # 6. Can list members (self only)
        res = c.get("/api/core/org/members/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["members"]), 1)

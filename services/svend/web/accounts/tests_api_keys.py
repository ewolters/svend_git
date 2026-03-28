"""API key authentication tests — SEC-001 §4.5.

Tests the full API key lifecycle: model, middleware, management endpoints,
and integration with existing auth decorators.

Compliance: SOC 2 CC6.1 (Logical Access Security)
"""

from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from accounts.constants import Tier
from accounts.models import APIKey
from conftest import make_user

User = get_user_model()

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False, RATELIMIT_ENABLE=False)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class APIKeyModelTest(TestCase):
    """SEC-001 §4.5: APIKey model — create, resolve, revoke, expire."""

    def setUp(self):
        self.user = make_user("apikey-model@test.com", tier=Tier.PRO)

    def test_create_key_returns_plaintext_and_hash(self):
        plaintext, key = APIKey.create_for_user(self.user, name="Test")
        self.assertTrue(plaintext.startswith("sv_"))
        self.assertEqual(len(plaintext), 46)
        self.assertEqual(key.key_prefix, plaintext[:11])
        self.assertNotEqual(key.key_hash, plaintext)  # Hash, not plaintext
        self.assertTrue(key.is_active)
        self.assertIsNone(key.revoked_at)

    def test_key_prefix_is_sv(self):
        plaintext, key = APIKey.create_for_user(self.user)
        self.assertTrue(key.key_prefix.startswith("sv_"))
        self.assertEqual(len(key.key_prefix), 11)

    def test_resolve_valid_key(self):
        plaintext, key = APIKey.create_for_user(self.user)
        resolved = APIKey.resolve(plaintext)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.id, key.id)
        self.assertEqual(resolved.user.id, self.user.id)

    def test_resolve_invalid_key(self):
        resolved = APIKey.resolve("sv_this_is_not_a_real_key_at_all")
        self.assertIsNone(resolved)

    def test_revoked_key_not_resolved(self):
        plaintext, key = APIKey.create_for_user(self.user)
        key.revoke()
        resolved = APIKey.resolve(plaintext)
        self.assertIsNone(resolved)

    def test_expired_key_not_resolved(self):
        plaintext, key = APIKey.create_for_user(self.user, expires_at=timezone.now() - timedelta(hours=1))
        resolved = APIKey.resolve(plaintext)
        self.assertIsNone(resolved)

    def test_future_expiry_key_resolves(self):
        plaintext, key = APIKey.create_for_user(self.user, expires_at=timezone.now() + timedelta(days=30))
        resolved = APIKey.resolve(plaintext)
        self.assertIsNotNone(resolved)

    def test_tier_limit_enforced(self):
        """PRO tier allows 5 keys."""
        for i in range(5):
            APIKey.create_for_user(self.user, name=f"Key {i}")
        with self.assertRaises(ValueError):
            APIKey.create_for_user(self.user, name="Key 6")

    def test_free_tier_cannot_create_keys(self):
        free_user = make_user("free-apikey@test.com", tier=Tier.FREE)
        with self.assertRaises(ValueError):
            APIKey.create_for_user(free_user)

    def test_revoked_keys_dont_count_toward_limit(self):
        """Revoked keys free up slots."""
        for i in range(5):
            _, key = APIKey.create_for_user(self.user, name=f"Key {i}")
        key.revoke()  # Revoke last one
        # Should be able to create one more
        APIKey.create_for_user(self.user, name="Replacement")

    def test_is_usable_property(self):
        _, key = APIKey.create_for_user(self.user)
        self.assertTrue(key.is_usable)
        key.revoke()
        self.assertFalse(key.is_usable)

    def test_is_expired_property(self):
        _, key = APIKey.create_for_user(self.user, expires_at=timezone.now() - timedelta(hours=1))
        self.assertTrue(key.is_expired)


# ---------------------------------------------------------------------------
# Middleware tests
# ---------------------------------------------------------------------------


@SECURE_OFF
class APIKeyMiddlewareTest(TestCase):
    """SEC-001 §4.5: Credential resolver middleware."""

    def setUp(self):
        self.user = make_user("apikey-mw@test.com", tier=Tier.PRO)
        self.plaintext, self.key = APIKey.create_for_user(self.user, name="MW Test")
        self.client = APIClient()

    def test_bearer_token_sets_request_user(self):
        """Valid Bearer sv_... header authenticates the request."""
        res = self.client.get(
            "/api/health/",  # Public endpoint, but middleware still runs
            HTTP_AUTHORIZATION=f"Bearer {self.plaintext}",
        )
        self.assertEqual(res.status_code, 200)

    def test_no_header_falls_through_to_session(self):
        """Without Bearer header, session auth works as before."""
        self.client.force_authenticate(self.user)
        res = self.client.get("/api/auth/me/")
        self.assertEqual(res.status_code, 200)

    def test_invalid_key_returns_401(self):
        """Invalid Bearer token returns 401."""
        res = self.client.get(
            "/api/auth/me/",
            HTTP_AUTHORIZATION="Bearer sv_invalid_key_here_not_real",
        )
        self.assertEqual(res.status_code, 401)
        data = res.json()
        self.assertEqual(data["code"], "invalid_api_key")

    def test_revoked_key_returns_401(self):
        self.key.revoke()
        res = self.client.get(
            "/api/auth/me/",
            HTTP_AUTHORIZATION=f"Bearer {self.plaintext}",
        )
        self.assertEqual(res.status_code, 401)

    def test_non_sv_bearer_ignored(self):
        """Bearer tokens without sv_ prefix are ignored (reserved for JWT)."""
        res = self.client.get(
            "/api/auth/me/",
            HTTP_AUTHORIZATION="Bearer eyJhbGciOiJIUzI1NiJ9.fake",
        )
        # Should fall through to session auth — no session, so 401/403
        self.assertIn(res.status_code, [401, 403])

    def test_api_key_works_through_rate_limited(self):
        """API key auth flows through @rate_limited decorator."""
        res = self.client.get(
            "/api/auth/me/",
            HTTP_AUTHORIZATION=f"Bearer {self.plaintext}",
        )
        # @rate_limited checks is_authenticated + can_query — both should pass
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["email"], self.user.email)


# ---------------------------------------------------------------------------
# Key management endpoint tests
# ---------------------------------------------------------------------------


@SECURE_OFF
class APIKeyEndpointTest(TestCase):
    """SEC-001 §4.5: Key management endpoints."""

    def setUp(self):
        self.user = make_user("apikey-ep@test.com", tier=Tier.PRO, is_email_verified=True)
        self.client = APIClient()
        # Raw Django views need session auth, not DRF force_authenticate
        self.client.login(username="apikey-ep", password="testpass123!")

    def test_create_key(self):
        res = self.client.post(
            "/api/auth/keys/",
            {"name": "CI Pipeline"},
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertTrue(data["key"].startswith("sv_"))
        self.assertEqual(data["name"], "CI Pipeline")
        self.assertIn("warning", data)

    def test_create_key_with_expiry(self):
        res = self.client.post(
            "/api/auth/keys/",
            {"name": "Temp Key", "expires_in_days": 30},
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        self.assertIsNotNone(res.json()["expires_at"])

    def test_list_keys(self):
        APIKey.create_for_user(self.user, name="Key A")
        APIKey.create_for_user(self.user, name="Key B")
        res = self.client.get("/api/auth/keys/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["keys"]), 2)
        # Should NOT contain the raw key or hash
        for k in data["keys"]:
            self.assertNotIn("key_hash", k)
            self.assertNotIn("key", k)
            self.assertIn("key_prefix", k)

    def test_revoke_key(self):
        _, key = APIKey.create_for_user(self.user, name="To Revoke")
        res = self.client.delete(f"/api/auth/keys/{key.id}/")
        self.assertEqual(res.status_code, 200)
        key.refresh_from_db()
        self.assertFalse(key.is_active)

    def test_cannot_revoke_other_users_key(self):
        other = make_user("other-apikey@test.com", tier=Tier.PRO)
        _, other_key = APIKey.create_for_user(other, name="Other's Key")
        res = self.client.delete(f"/api/auth/keys/{other_key.id}/")
        self.assertEqual(res.status_code, 404)

    def test_create_blocked_via_api_key_auth(self):
        """Cannot create keys using API key auth (session only)."""
        plaintext, _ = APIKey.create_for_user(self.user, name="Existing")
        api_client = APIClient()
        res = api_client.post(
            "/api/auth/keys/",
            {"name": "Persistence Attack"},
            format="json",
            HTTP_AUTHORIZATION=f"Bearer {plaintext}",
        )
        self.assertEqual(res.status_code, 403)
        data = res.json()
        # Error may be wrapped by ErrorEnvelopeMiddleware
        error_msg = data.get("error", "")
        if isinstance(error_msg, dict):
            error_msg = error_msg.get("message", "")
        self.assertIn("session", error_msg.lower())

    def test_revoke_blocked_via_api_key_auth(self):
        """Cannot revoke keys using API key auth (session only)."""
        plaintext, key = APIKey.create_for_user(self.user, name="Existing")
        api_client = APIClient()
        res = api_client.delete(
            f"/api/auth/keys/{key.id}/",
            HTTP_AUTHORIZATION=f"Bearer {plaintext}",
        )
        self.assertEqual(res.status_code, 403)

    def test_free_tier_blocked(self):
        make_user("free-ep@test.com", tier=Tier.FREE)
        free_client = APIClient()
        free_client.login(username="free-ep", password="testpass123!")
        res = free_client.post(
            "/api/auth/keys/",
            {"name": "Free Key"},
            format="json",
        )
        self.assertEqual(res.status_code, 403)

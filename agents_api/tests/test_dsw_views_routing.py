"""Routing stub tests — verify zero-coverage DSW endpoints reject unauthenticated requests.

These endpoints are thin routing stubs in dsw_views.py that delegate to
dsw/endpoints_data.py. They had ZERO test coverage prior to INIT-015.

Standard: DSW-001
"""

import json

from django.test import TestCase, override_settings

from accounts.constants import Tier
from accounts.models import User

ROUTING_SETTINGS = {"RATELIMIT_ENABLE": False, "SECURE_SSL_REDIRECT": False}


def _make_user(email, tier=Tier.PRO, staff=False):
    """Create a test user with specified tier."""
    username = email.split("@")[0].replace(".", "_")
    u = User.objects.create_user(username=username, email=email, password="testpass123")
    u.tier = tier
    u.is_staff = staff
    u.email_verified = True
    u.save()
    return u


@override_settings(**ROUTING_SETTINGS)
class DSWRoutingStubUnauthTest(TestCase):
    """Verify unauthenticated requests are rejected for all DSW routing stubs."""

    def test_analyst_assistant_unauth(self):
        """POST /api/dsw/analyst/ without auth → 401."""
        resp = self.client.post(
            "/api/dsw/analyst/",
            json.dumps({}),
            content_type="application/json",
        )
        self.assertIn(resp.status_code, (401, 403))

    def test_download_data_unauth(self):
        """POST /api/dsw/download/ without auth → 401."""
        resp = self.client.post(
            "/api/dsw/download/",
            json.dumps({}),
            content_type="application/json",
        )
        self.assertIn(resp.status_code, (401, 403))

    def test_retrieve_data_unauth(self):
        """POST /api/dsw/retrieve-data/ without auth → 401."""
        resp = self.client.post(
            "/api/dsw/retrieve-data/",
            json.dumps({}),
            content_type="application/json",
        )
        self.assertIn(resp.status_code, (401, 403))

    def test_triage_scan_unauth(self):
        """POST /api/dsw/triage/scan/ without auth → 401."""
        resp = self.client.post(
            "/api/dsw/triage/scan/",
            json.dumps({}),
            content_type="application/json",
        )
        self.assertIn(resp.status_code, (401, 403))


@override_settings(**ROUTING_SETTINGS)
class DSWRoutingStubTierTest(TestCase):
    """Verify tier gating on routing stubs."""

    @classmethod
    def setUpTestData(cls):
        cls.pro_user = _make_user("routing-pro@test.com", tier=Tier.PRO)

    def test_analyst_assistant_requires_enterprise(self):
        """POST /api/dsw/analyst/ as PRO user → 403 (enterprise-only)."""
        self.client.force_login(self.pro_user)
        resp = self.client.post(
            "/api/dsw/analyst/",
            json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 403)


@override_settings(**ROUTING_SETTINGS)
class DSWRoutingStubAuthTest(TestCase):
    """Verify authenticated requests to routing stubs don't return 500."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("routing-auth@test.com")

    def setUp(self):
        self.client.force_login(self.user)

    def test_retrieve_data_empty_body(self):
        """POST /api/dsw/retrieve-data/ with empty body → 400 (not 500)."""
        resp = self.client.post(
            "/api/dsw/retrieve-data/",
            json.dumps({}),
            content_type="application/json",
        )
        self.assertIn(resp.status_code, (400, 404))

    def test_download_data_empty_body(self):
        """POST /api/dsw/download/ with empty body → 400 (not 500)."""
        resp = self.client.post(
            "/api/dsw/download/",
            json.dumps({}),
            content_type="application/json",
        )
        self.assertIn(resp.status_code, (400, 404))

    def test_triage_scan_empty_body(self):
        """POST /api/dsw/triage/scan/ with empty body → 400 (not 500)."""
        resp = self.client.post(
            "/api/dsw/triage/scan/",
            json.dumps({}),
            content_type="application/json",
        )
        self.assertNotEqual(resp.status_code, 500)

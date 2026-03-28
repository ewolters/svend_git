"""Webhook system tests — NTF-001 §5.4.

Tests the full webhook lifecycle: model, HMAC signing, delivery,
retry, circuit breaker, management endpoints, event pattern matching.

Compliance: SOC 2 CC6.1 (Logical Access Security)
"""

import hashlib
import hmac
import json
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from accounts.constants import Tier
from conftest import make_membership, make_tenant, make_user
from core.models.tenant import Membership

from .models import (
    WEBHOOK_CIRCUIT_BREAKER_THRESHOLD,
    WebhookDelivery,
    WebhookEndpoint,
)

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False, RATELIMIT_ENABLE=False)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class WebhookEndpointModelTest(TestCase):
    """NTF-001 §5.4: WebhookEndpoint model."""

    def setUp(self):
        self.tenant = make_tenant(name="WH Org", slug="wh-org")
        self.user = make_user("wh-model@test.com", tier=Tier.TEAM)
        make_membership(self.tenant, self.user, role=Membership.Role.ADMIN)

    def test_create_endpoint_generates_secret(self):
        secret, endpoint = WebhookEndpoint.create_for_user(
            self.user,
            url="https://example.com/webhook",
            event_patterns=["fmea.*"],
            tenant=self.tenant,
        )
        self.assertEqual(len(secret), 64)  # 32 bytes hex
        self.assertEqual(endpoint.secret, secret)
        self.assertTrue(endpoint.is_active)

    def test_https_required(self):
        with self.assertRaises(ValueError):
            WebhookEndpoint.create_for_user(
                self.user,
                url="http://example.com/webhook",
                event_patterns=["*"],
            )

    def test_event_pattern_matching_wildcard(self):
        _, endpoint = WebhookEndpoint.create_for_user(
            self.user,
            url="https://example.com/wh",
            event_patterns=["fmea.*"],
            tenant=self.tenant,
        )
        self.assertTrue(endpoint.matches_event("fmea.created"))
        self.assertTrue(endpoint.matches_event("fmea.row_added"))
        self.assertFalse(endpoint.matches_event("capa.created"))

    def test_event_pattern_matching_exact(self):
        _, endpoint = WebhookEndpoint.create_for_user(
            self.user,
            url="https://example.com/wh",
            event_patterns=["capa.status_changed"],
            tenant=self.tenant,
        )
        self.assertTrue(endpoint.matches_event("capa.status_changed"))
        self.assertFalse(endpoint.matches_event("capa.created"))

    def test_event_pattern_matching_star_all(self):
        _, endpoint = WebhookEndpoint.create_for_user(
            self.user,
            url="https://example.com/wh",
            event_patterns=["*"],
            tenant=self.tenant,
        )
        self.assertTrue(endpoint.matches_event("anything.at_all"))

    def test_event_pattern_matching_suffix_wildcard(self):
        _, endpoint = WebhookEndpoint.create_for_user(
            self.user,
            url="https://example.com/wh",
            event_patterns=["*.created"],
            tenant=self.tenant,
        )
        self.assertTrue(endpoint.matches_event("fmea.created"))
        self.assertTrue(endpoint.matches_event("rca.created"))
        self.assertFalse(endpoint.matches_event("fmea.updated"))

    def test_circuit_breaker_disables_after_threshold(self):
        _, endpoint = WebhookEndpoint.create_for_user(
            self.user,
            url="https://example.com/wh",
            event_patterns=["*"],
            tenant=self.tenant,
        )
        for _ in range(WEBHOOK_CIRCUIT_BREAKER_THRESHOLD):
            endpoint.record_failure()

        endpoint.refresh_from_db()
        self.assertFalse(endpoint.is_active)
        self.assertIsNotNone(endpoint.disabled_at)


# ---------------------------------------------------------------------------
# HMAC signing tests
# ---------------------------------------------------------------------------


class WebhookSigningTest(TestCase):
    """NTF-001 §5.4: HMAC-SHA256 payload signing."""

    def setUp(self):
        self.tenant = make_tenant(name="Sign Org", slug="sign-org")
        self.user = make_user("wh-sign@test.com", tier=Tier.TEAM)
        make_membership(self.tenant, self.user, role=Membership.Role.ADMIN)

    def test_hmac_signature_verifiable(self):
        """Customer can verify the signature with their secret."""
        secret, endpoint = WebhookEndpoint.create_for_user(
            self.user,
            url="https://example.com/wh",
            event_patterns=["*"],
            tenant=self.tenant,
        )
        payload = {"event": "fmea.created", "data": {"id": "test-123"}}
        signature = endpoint.sign_payload(payload)

        # Customer-side verification
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        expected = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
        self.assertEqual(signature, f"sha256={expected}")

    def test_wrong_secret_fails_verification(self):
        _, endpoint = WebhookEndpoint.create_for_user(
            self.user,
            url="https://example.com/wh",
            event_patterns=["*"],
            tenant=self.tenant,
        )
        payload = {"event": "test", "data": {}}
        signature = endpoint.sign_payload(payload)

        # Wrong secret
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        wrong = hmac.new(b"wrong-secret", payload_bytes, hashlib.sha256).hexdigest()
        self.assertNotEqual(signature, f"sha256={wrong}")


# ---------------------------------------------------------------------------
# Delivery engine tests
# ---------------------------------------------------------------------------


class WebhookDeliveryTest(TestCase):
    """NTF-001 §5.4: Webhook delivery with retry."""

    def setUp(self):
        self.tenant = make_tenant(name="Del Org", slug="del-org")
        self.user = make_user("wh-deliver@test.com", tier=Tier.TEAM)
        make_membership(self.tenant, self.user, role=Membership.Role.ADMIN)
        _, self.endpoint = WebhookEndpoint.create_for_user(
            self.user,
            url="https://example.com/wh",
            event_patterns=["*"],
            tenant=self.tenant,
        )

    @patch("notifications.webhook_delivery.requests.post")
    def test_successful_delivery(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "OK"
        mock_post.return_value = mock_resp

        delivery = WebhookDelivery.objects.create(
            endpoint=self.endpoint,
            event_name="fmea.created",
            payload={"event": "fmea.created", "data": {"id": "123"}},
        )

        from .webhook_delivery import deliver

        deliver(delivery.id)
        delivery.refresh_from_db()

        self.assertEqual(delivery.status, WebhookDelivery.Status.DELIVERED)
        self.assertEqual(delivery.response_code, 200)
        self.assertIsNotNone(delivery.delivered_at)
        self.assertEqual(delivery.attempt_count, 1)

    @patch("notifications.webhook_delivery.requests.post")
    def test_retry_on_failure(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        delivery = WebhookDelivery.objects.create(
            endpoint=self.endpoint,
            event_name="fmea.created",
            payload={"event": "fmea.created", "data": {}},
        )

        from .webhook_delivery import deliver

        deliver(delivery.id)
        delivery.refresh_from_db()

        self.assertEqual(delivery.status, WebhookDelivery.Status.FAILED)
        self.assertIsNotNone(delivery.next_retry_at)
        self.assertEqual(delivery.attempt_count, 1)

    @patch("notifications.webhook_delivery.requests.post")
    def test_exhausted_after_max_retries(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Error"
        mock_post.return_value = mock_resp

        delivery = WebhookDelivery.objects.create(
            endpoint=self.endpoint,
            event_name="fmea.created",
            payload={"event": "fmea.created", "data": {}},
        )

        from .webhook_delivery import deliver

        # 4 attempts = 1 initial + 3 retries → exhausted
        for _ in range(4):
            deliver(delivery.id)
            delivery.refresh_from_db()

        self.assertEqual(delivery.status, WebhookDelivery.Status.EXHAUSTED)
        self.assertIsNone(delivery.next_retry_at)

    @patch("notifications.webhook_delivery.requests.post")
    def test_hmac_header_sent(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "OK"
        mock_post.return_value = mock_resp

        delivery = WebhookDelivery.objects.create(
            endpoint=self.endpoint,
            event_name="test.event",
            payload={"event": "test.event", "data": {}},
        )

        from .webhook_delivery import deliver

        deliver(delivery.id)

        # Verify HMAC header was sent
        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))
        self.assertIn("X-Svend-Signature", headers)
        self.assertTrue(headers["X-Svend-Signature"].startswith("sha256="))

    @patch("notifications.webhook_delivery.requests.post")
    def test_dispatch_event_creates_deliveries(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "OK"
        mock_post.return_value = mock_resp

        from .webhook_delivery import dispatch_event

        dispatch_event("fmea.created", {"id": "123", "title": "Test FMEA"})

        self.assertEqual(WebhookDelivery.objects.count(), 1)
        d = WebhookDelivery.objects.first()
        self.assertEqual(d.event_name, "fmea.created")
        self.assertEqual(d.status, WebhookDelivery.Status.DELIVERED)


# ---------------------------------------------------------------------------
# Endpoint management tests
# ---------------------------------------------------------------------------


@SECURE_OFF
class WebhookEndpointViewTest(TestCase):
    """NTF-001 §5.4: Webhook management endpoints."""

    def setUp(self):
        self.tenant = make_tenant(name="View Org", slug="view-org")
        self.user = make_user("wh-views@test.com", tier=Tier.TEAM)
        make_membership(self.tenant, self.user, role=Membership.Role.ADMIN)
        self.client = APIClient()
        self.client.login(username="wh-views", password="testpass123!")

    def test_create_endpoint(self):
        res = self.client.post(
            "/api/webhooks/",
            {
                "url": "https://example.com/hook",
                "event_patterns": ["fmea.*", "capa.*"],
                "description": "Test hook",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertIn("secret", data)
        self.assertEqual(len(data["secret"]), 64)
        self.assertIn("warning", data)

    def test_list_endpoints(self):
        WebhookEndpoint.create_for_user(
            self.user,
            url="https://a.com/wh",
            event_patterns=["*"],
            tenant=self.tenant,
        )
        res = self.client.get("/api/webhooks/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["endpoints"]), 1)
        # Secret must NOT be in list response
        self.assertNotIn("secret", data["endpoints"][0])

    def test_delete_endpoint(self):
        _, endpoint = WebhookEndpoint.create_for_user(
            self.user,
            url="https://del.com/wh",
            event_patterns=["*"],
            tenant=self.tenant,
        )
        res = self.client.delete(f"/api/webhooks/{endpoint.id}/")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(WebhookEndpoint.objects.filter(id=endpoint.id).exists())

    def test_update_endpoint(self):
        _, endpoint = WebhookEndpoint.create_for_user(
            self.user,
            url="https://upd.com/wh",
            event_patterns=["*"],
            tenant=self.tenant,
        )
        res = self.client.put(
            f"/api/webhooks/{endpoint.id}/",
            {"event_patterns": ["fmea.*"], "description": "Updated"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        endpoint.refresh_from_db()
        self.assertEqual(endpoint.event_patterns, ["fmea.*"])
        self.assertEqual(endpoint.description, "Updated")

    def test_reenable_clears_circuit_breaker(self):
        _, endpoint = WebhookEndpoint.create_for_user(
            self.user,
            url="https://re.com/wh",
            event_patterns=["*"],
            tenant=self.tenant,
        )
        # Simulate circuit breaker
        endpoint.is_active = False
        endpoint.failure_count = 10
        endpoint.disabled_at = timezone.now()
        endpoint.save()

        res = self.client.put(
            f"/api/webhooks/{endpoint.id}/",
            {"is_active": True},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        endpoint.refresh_from_db()
        self.assertTrue(endpoint.is_active)
        self.assertEqual(endpoint.failure_count, 0)
        self.assertIsNone(endpoint.disabled_at)

    def test_http_url_rejected(self):
        res = self.client.post(
            "/api/webhooks/",
            {
                "url": "http://insecure.com/hook",
                "event_patterns": ["*"],
            },
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_cross_tenant_blocked(self):
        """Cannot see or modify another tenant's endpoints."""
        other_tenant = make_tenant(name="Other Org", slug="other-org")
        other_user = make_user("other-wh@test.com", tier=Tier.TEAM)
        make_membership(other_tenant, other_user, role=Membership.Role.ADMIN)
        _, other_endpoint = WebhookEndpoint.create_for_user(
            other_user,
            url="https://other.com/wh",
            event_patterns=["*"],
            tenant=other_tenant,
        )

        res = self.client.delete(f"/api/webhooks/{other_endpoint.id}/")
        self.assertEqual(res.status_code, 404)

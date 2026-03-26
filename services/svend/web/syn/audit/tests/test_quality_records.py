"""
FEAT-093: Quality record creation on DSW validation rejection.

Verifies that rejected analysis requests create audit trail entries
and that the quality event is properly registered in the catalog.

Standard: QUAL-001 §7.3
"""

import json
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import RequestFactory, SimpleTestCase, TestCase

from syn.audit.events import AUDIT_EVENTS

User = get_user_model()


# ── Event catalog ──────────────────────────────────────────────────────


class QualityEventCatalogTest(SimpleTestCase):
    """QUAL-001 §7.3: quality.analysis_rejected event registered."""

    def test_event_registered(self):
        """quality.analysis_rejected is in AUDIT_EVENTS catalog."""
        self.assertIn("quality.analysis_rejected", AUDIT_EVENTS)

    def test_event_has_payload_schema(self):
        """Event schema requires 'reason' field."""
        schema = AUDIT_EVENTS["quality.analysis_rejected"]["payload_schema"]
        self.assertIn("reason", schema["properties"])
        self.assertIn("reason", schema["required"])


# ── Rejection logging integration ──────────────────────────────────────


class QualityRejectionLoggingTest(TestCase):
    """QUAL-001 §7.3: DSW rejections create SysLogEntry records."""

    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(
            username="qr_test_user",
            email="qr@test.com",
            password="testpass123",
        )

    def _make_request(self, body=None):
        """Create a POST request to the analysis endpoint."""
        factory = RequestFactory()
        content = body if isinstance(body, (str, bytes)) else json.dumps(body or {})
        request = factory.post(
            "/api/dsw/run/",
            data=content,
            content_type="application/json",
        )
        request.user = self.user
        return request

    def test_invalid_json_creates_entry(self):
        """Invalid JSON POST creates a quality.analysis_rejected SysLogEntry."""
        from syn.audit.models import SysLogEntry

        before = SysLogEntry.objects.filter(
            event_name="quality.analysis_rejected"
        ).count()

        from agents_api.dsw.dispatch import run_analysis

        factory = RequestFactory()
        request = factory.post(
            "/api/dsw/run/", data="not json{{{", content_type="application/json"
        )
        request.user = self.user

        response = run_analysis(request)
        self.assertEqual(response.status_code, 400)

        after = SysLogEntry.objects.filter(
            event_name="quality.analysis_rejected"
        ).count()
        self.assertEqual(after, before + 1)

        entry = (
            SysLogEntry.objects.filter(event_name="quality.analysis_rejected")
            .order_by("-id")
            .first()
        )
        self.assertEqual(entry.payload["reason"], "invalid_json")

    def test_no_data_creates_entry(self):
        """POST with no data_id creates a rejection quality record."""
        from syn.audit.models import SysLogEntry

        before = SysLogEntry.objects.filter(
            event_name="quality.analysis_rejected"
        ).count()

        from agents_api.dsw.dispatch import run_analysis

        request = self._make_request({"type": "stats", "analysis": "ttest"})
        response = run_analysis(request)
        self.assertEqual(response.status_code, 400)

        after = SysLogEntry.objects.filter(
            event_name="quality.analysis_rejected"
        ).count()
        self.assertEqual(after, before + 1)

        entry = (
            SysLogEntry.objects.filter(event_name="quality.analysis_rejected")
            .order_by("-id")
            .first()
        )
        self.assertEqual(entry.payload["reason"], "no_data_loaded")

    def test_unknown_type_creates_entry(self):
        """POST with unknown analysis type creates a rejection quality record."""
        from syn.audit.models import SysLogEntry

        before = SysLogEntry.objects.filter(
            event_name="quality.analysis_rejected"
        ).count()

        from agents_api.dsw.dispatch import run_analysis

        # Provide inline data so we get past the no-data check
        request = self._make_request(
            {
                "type": "nonexistent_type_xyz",
                "analysis": "foo",
                "data": {"col1": [1, 2, 3]},
            }
        )
        response = run_analysis(request)
        self.assertEqual(response.status_code, 400)

        after = SysLogEntry.objects.filter(
            event_name="quality.analysis_rejected"
        ).count()
        self.assertEqual(after, before + 1)

        entry = (
            SysLogEntry.objects.filter(event_name="quality.analysis_rejected")
            .order_by("-id")
            .first()
        )
        self.assertIn("unknown_analysis_type", entry.payload["reason"])

    def test_rejection_payload_has_required_fields(self):
        """Quality record payload contains reason, analysis_type, analysis_id."""
        from agents_api.dsw.dispatch import run_analysis
        from syn.audit.models import SysLogEntry

        request = self._make_request({"type": "stats", "analysis": "ttest"})
        run_analysis(request)

        entry = (
            SysLogEntry.objects.filter(event_name="quality.analysis_rejected")
            .order_by("-id")
            .first()
        )
        self.assertIn("reason", entry.payload)
        self.assertIn("analysis_type", entry.payload)
        self.assertIn("analysis_id", entry.payload)

    def test_logging_failure_does_not_block_response(self):
        """If generate_entry raises, the 400 response still returns."""
        from agents_api.dsw.dispatch import run_analysis

        with patch("syn.audit.utils.generate_entry", side_effect=RuntimeError("boom")):
            factory = RequestFactory()
            request = factory.post(
                "/api/dsw/run/", data="bad json", content_type="application/json"
            )
            request.user = self.user
            response = run_analysis(request)
            self.assertEqual(response.status_code, 400)

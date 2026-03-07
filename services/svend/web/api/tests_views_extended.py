"""Extended API view tests — chat, conversations, email tracking, beacons, compliance.

Covers symbols not already tested in api/tests.py (auth, registration, profile, etc.).
"""

import uuid
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from accounts.constants import Tier

User = get_user_model()

# Production SECURE_SSL_REDIRECT=True breaks test client HTTP requests.
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_user(email, tier=Tier.FREE, password="testpass123!", **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(username=username, email=email, password=password, **kwargs)
    user.tier = tier
    user.is_email_verified = True  # most tests need this to use chat
    user.save(update_fields=["tier", "is_email_verified"])
    return user


def _make_admin(email="admin@example.com"):
    user = _make_user(email, Tier.PRO, username="admin")
    user.is_staff = True
    user.is_superuser = True
    user.save(update_fields=["is_staff", "is_superuser"])
    return user


def _mock_process_query_result(**overrides):
    """Build a mock result object imitating process_query return value."""
    defaults = {
        "response": "The answer is 42.",
        "pipeline_type": "cognition",
        "domain": "math",
        "difficulty": 0.5,
        "verified": True,
        "verification_confidence": 0.95,
        "blocked": False,
        "block_reason": "",
        "reasoning_trace": [{"step": 1, "text": "Computed"}],
        "tool_calls": None,
        "inference_time_ms": 120,
        "success": True,
        "selected_mode": "reasoner",
        "mode_scores": {"reasoner": 0.9},
        "confidence": 0.95,
        "formatted_trace": None,
        "code": None,
        "visualizations": None,
        "execution_outputs": None,
        "execution_errors": None,
        "tools_used": None,
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


# =========================================================================
# ChatEndpointTest
# =========================================================================


@SECURE_OFF
class ChatEndpointTest(TestCase):
    """Tests for POST /api/chat/ — mocks the LLM pipeline."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("chatter@example.com")
        self.client.force_authenticate(self.user)

    @patch("api.views.process_query")
    def test_chat_creates_conversation_and_messages(self, mock_pq):
        mock_pq.return_value = _mock_process_query_result()
        res = self.client.post("/api/chat/", {"message": "What is 2+2?"}, format="json")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("conversation_id", data)
        self.assertIn("user_message", data)
        self.assertIn("assistant_message", data)
        self.assertEqual(data["user_message"]["role"], "user")
        self.assertEqual(data["assistant_message"]["role"], "assistant")
        self.assertIn("The answer is 42", data["assistant_message"]["content"])

    @patch("api.views.process_query")
    def test_chat_reuses_existing_conversation(self, mock_pq):
        mock_pq.return_value = _mock_process_query_result()
        # First message — creates conversation
        res1 = self.client.post("/api/chat/", {"message": "Hello"}, format="json")
        cid = res1.json()["conversation_id"]
        # Second message to same conversation
        res2 = self.client.post("/api/chat/", {"message": "Follow-up", "conversation_id": cid}, format="json")
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["conversation_id"], cid)

    @patch("api.views.process_query")
    def test_chat_nonexistent_conversation_404(self, mock_pq):
        fake_id = str(uuid.uuid4())
        res = self.client.post("/api/chat/", {"message": "Hi", "conversation_id": fake_id}, format="json")
        self.assertEqual(res.status_code, 404)

    def test_chat_unauthenticated_rejected(self):
        client = APIClient()
        res = client.post("/api/chat/", {"message": "Hi"}, format="json")
        self.assertIn(res.status_code, [401, 403])

    @patch("api.views.process_query")
    def test_chat_unverified_email_blocked(self, mock_pq):
        self.user.is_email_verified = False
        self.user.save(update_fields=["is_email_verified"])
        res = self.client.post("/api/chat/", {"message": "Hi"}, format="json")
        self.assertEqual(res.status_code, 403)
        error = res.json().get("error", "")
        # ErrorEnvelopeMiddleware may wrap as {"error": {"message": ...}} or plain string
        if isinstance(error, dict):
            error_text = error.get("message", "")
        else:
            error_text = error
        self.assertIn("verify", error_text.lower())

    @patch("api.views.process_query")
    def test_chat_empty_message_rejected(self, mock_pq):
        res = self.client.post("/api/chat/", {"message": ""}, format="json")
        self.assertEqual(res.status_code, 400)

    @patch("api.views.process_query")
    def test_chat_pipeline_error_returns_graceful_response(self, mock_pq):
        mock_pq.side_effect = RuntimeError("LLM down")
        res = self.client.post("/api/chat/", {"message": "Test"}, format="json")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("error", data["assistant_message"]["content"].lower())


# =========================================================================
# ConversationScenarioTest
# =========================================================================


@SECURE_OFF
class ConversationScenarioTest(TestCase):
    """Full scenario: chat -> list -> detail -> share -> delete."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("scenario@example.com")
        self.client.force_authenticate(self.user)

    @patch("api.views.process_query")
    def test_full_conversation_lifecycle(self, mock_pq):
        mock_pq.return_value = _mock_process_query_result()

        # 1. Chat creates conversation
        res = self.client.post("/api/chat/", {"message": "Hello Svend"}, format="json")
        self.assertEqual(res.status_code, 200)
        cid = res.json()["conversation_id"]

        # 2. List conversations — should contain the new one
        res = self.client.get("/api/conversations/")
        self.assertEqual(res.status_code, 200)
        ids = [c["id"] for c in res.json()]
        self.assertIn(cid, ids)

        # 3. Get conversation detail — should include messages
        res = self.client.get(f"/api/conversations/{cid}/")
        self.assertEqual(res.status_code, 200)
        detail = res.json()
        self.assertEqual(detail["id"], cid)
        self.assertIn("messages", detail)
        self.assertGreaterEqual(len(detail["messages"]), 2)  # user + assistant

        # 4. Share conversation
        res = self.client.post("/api/share/", {"conversation_id": cid}, format="json")
        self.assertEqual(res.status_code, 200)
        self.assertIn("share_id", res.json())
        self.assertIn("url", res.json())

        # 5. Delete conversation
        res = self.client.delete(f"/api/conversations/{cid}/")
        self.assertEqual(res.status_code, 204)

        # 6. Verify deletion
        res = self.client.get(f"/api/conversations/{cid}/")
        self.assertEqual(res.status_code, 404)

    def test_list_conversations_empty(self):
        res = self.client.get("/api/conversations/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), [])

    def test_conversation_detail_wrong_user(self):
        """User cannot access another user's conversation."""
        from chat.models import Conversation

        other = _make_user("other@example.com")
        convo = Conversation.objects.create(user=other, title="Secret")
        res = self.client.get(f"/api/conversations/{convo.id}/")
        self.assertEqual(res.status_code, 404)

    def test_share_nonexistent_conversation(self):
        res = self.client.post("/api/share/", {"conversation_id": str(uuid.uuid4())}, format="json")
        self.assertEqual(res.status_code, 404)


# =========================================================================
# FlagMessageTest
# =========================================================================


@SECURE_OFF
class FlagMessageTest(TestCase):
    """Tests for POST /api/messages/<uuid>/flag/."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("flagger@example.com")
        self.client.force_authenticate(self.user)

    @patch("api.views.process_query")
    def test_flag_message_creates_training_candidate(self, mock_pq):
        from chat.models import TrainingCandidate

        mock_pq.return_value = _mock_process_query_result()
        res = self.client.post("/api/chat/", {"message": "Bad answer"}, format="json")
        msg_id = res.json()["assistant_message"]["id"]

        res = self.client.post(f"/api/messages/{msg_id}/flag/", {"reason": "Incorrect"}, format="json")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "flagged")
        self.assertIn("candidate_id", res.json())

        # Verify training candidate exists
        cid = res.json()["candidate_id"]
        candidate = TrainingCandidate.objects.get(id=cid)
        self.assertEqual(candidate.candidate_type, "user_flagged")
        self.assertIn("Incorrect", candidate.reviewer_notes)

    def test_flag_nonexistent_message_404(self):
        res = self.client.post(f"/api/messages/{uuid.uuid4()}/flag/", {"reason": "Bad"}, format="json")
        self.assertEqual(res.status_code, 404)

    @patch("api.views.process_query")
    def test_flag_other_users_message_forbidden(self, mock_pq):
        from chat.models import Conversation, Message

        other = _make_user("victim@example.com")
        convo = Conversation.objects.create(user=other)
        msg = Message.objects.create(conversation=convo, role="assistant", content="Test")
        res = self.client.post(f"/api/messages/{msg.id}/flag/", {"reason": "Nope"}, format="json")
        self.assertEqual(res.status_code, 403)


# =========================================================================
# MonitoringStatsTest
# =========================================================================


@SECURE_OFF
class MonitoringStatsTest(TestCase):
    """Tests for GET /api/stats/traces/ and GET /api/stats/flywheel/."""

    def setUp(self):
        self.client = APIClient()
        self.admin = _make_admin()
        self.user = _make_user("normie@example.com", username="normie")
        self.client.force_authenticate(self.admin)

    def test_trace_stats_returns_expected_shape(self):
        res = self.client.get("/api/stats/traces/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        for key in (
            "period",
            "total_requests",
            "gate_passed",
            "gate_failed",
            "error_count",
            "pass_rate",
            "domains",
            "training_candidates",
        ):
            self.assertIn(key, data, f"Missing key: {key}")

    def test_trace_stats_non_admin_forbidden(self):
        self.client.force_authenticate(self.user)
        res = self.client.get("/api/stats/traces/")
        self.assertIn(res.status_code, [401, 403])

    @patch("api.views.get_flywheel")
    def test_flywheel_stats_returns_data(self, mock_fw):
        mock_fw.return_value.get_stats.return_value = {"total_queries": 100, "escalations": 5}
        res = self.client.get("/api/stats/flywheel/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("total_queries", res.json())

    @patch("api.views.get_flywheel")
    def test_flywheel_stats_non_admin_forbidden(self, mock_fw):
        self.client.force_authenticate(self.user)
        res = self.client.get("/api/stats/flywheel/")
        self.assertIn(res.status_code, [401, 403])


# =========================================================================
# EventTrackingTest
# =========================================================================


@SECURE_OFF
class EventTrackingTest(TestCase):
    """Tests for POST /api/events/."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("tracker@example.com")
        self.client.force_authenticate(self.user)

    def test_track_single_event(self):
        from chat.models import EventLog

        res = self.client.post(
            "/api/events/",
            {"event_type": "page_view", "category": "chat", "page": "/app/chat/"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["tracked"], 1)
        self.assertEqual(EventLog.objects.filter(user=self.user).count(), 1)

    def test_track_batch_events(self):
        events = [
            {"event_type": "page_view", "page": "/app/"},
            {"event_type": "feature_use", "category": "spc", "action": "run_chart"},
        ]
        res = self.client.post("/api/events/", events, format="json")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["tracked"], 2)

    def test_track_invalid_event_type_skipped(self):
        res = self.client.post(
            "/api/events/",
            {"event_type": "hacker_attack"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["tracked"], 0)

    def test_track_unauthenticated_rejected(self):
        client = APIClient()
        res = client.post("/api/events/", {"event_type": "page_view"}, format="json")
        self.assertIn(res.status_code, [401, 403])


# =========================================================================
# ExportPDFTest
# =========================================================================


@SECURE_OFF
class ExportPDFTest(TestCase):
    """Tests for POST /api/export/pdf/."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("exporter@example.com")
        self.client.force_authenticate(self.user)

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_export_pdf_fallback_to_html(self, mock_sub):
        """When wkhtmltopdf and weasyprint are unavailable, returns HTML."""
        with patch.dict("sys.modules", {"weasyprint": None}):
            res = self.client.post(
                "/api/export/pdf/",
                {"content": "# Hello World", "title": "Test Doc"},
                format="json",
            )
        # Should either return PDF or fallback HTML
        self.assertIn(res.status_code, [200])
        content_type = res["Content-Type"]
        self.assertTrue(content_type.startswith("text/html") or content_type == "application/pdf")

    def test_export_pdf_empty_content_rejected(self):
        res = self.client.post("/api/export/pdf/", {"content": ""}, format="json")
        self.assertEqual(res.status_code, 400)

    def test_export_pdf_unauthenticated(self):
        client = APIClient()
        res = client.post("/api/export/pdf/", {"content": "test"}, format="json")
        self.assertIn(res.status_code, [401, 403])


# =========================================================================
# EmailTrackingTest
# =========================================================================


@SECURE_OFF
class EmailTrackingTest(TestCase):
    """Tests for email open/click tracking and unsubscribe flow."""

    def setUp(self):
        self.client = APIClient()

    def _make_recipient(self):
        from api.models import EmailCampaign, EmailRecipient

        campaign = EmailCampaign.objects.create(subject="Test Campaign", body_md="Hello", target="all")
        return EmailRecipient.objects.create(campaign=campaign, email="recipient@example.com")

    def test_email_open_returns_pixel(self):
        rcpt = self._make_recipient()
        res = self.client.get(f"/api/email/open/{rcpt.id}/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res["Content-Type"], "image/gif")
        # Verify opened_at was set
        rcpt.refresh_from_db()
        self.assertIsNotNone(rcpt.opened_at)

    def test_email_open_nonexistent_still_returns_pixel(self):
        fake = uuid.uuid4()
        res = self.client.get(f"/api/email/open/{fake}/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res["Content-Type"], "image/gif")

    def test_email_click_redirects(self):
        rcpt = self._make_recipient()
        res = self.client.get(f"/api/email/click/{rcpt.id}/", {"url": "https://svend.ai/pricing/"})
        self.assertEqual(res.status_code, 302)
        self.assertEqual(res["Location"], "https://svend.ai/pricing/")
        rcpt.refresh_from_db()
        self.assertIsNotNone(rcpt.clicked_at)
        self.assertIsNotNone(rcpt.opened_at)  # click also marks as opened

    def test_email_click_blocks_external_redirect(self):
        rcpt = self._make_recipient()
        res = self.client.get(f"/api/email/click/{rcpt.id}/", {"url": "https://evil.com/phish"})
        self.assertEqual(res.status_code, 302)
        self.assertEqual(res["Location"], "https://svend.ai")

    def test_unsubscribe_with_valid_token(self):
        from api.views import make_unsubscribe_url

        user = _make_user("unsub@example.com")
        url = make_unsubscribe_url(user)
        # Extract token from URL
        token = url.split("token=")[1]

        res = self.client.get("/api/email/unsubscribe/", {"token": token})
        self.assertEqual(res.status_code, 200)
        self.assertIn("Unsubscribed", res.content.decode())

        user.refresh_from_db()
        self.assertTrue(user.is_email_opted_out)

    def test_unsubscribe_invalid_token(self):
        res = self.client.get("/api/email/unsubscribe/", {"token": "garbage"})
        self.assertEqual(res.status_code, 400)

    def test_unsubscribe_missing_token(self):
        res = self.client.get("/api/email/unsubscribe/")
        self.assertEqual(res.status_code, 400)


# =========================================================================
# PublicBeaconTest
# =========================================================================


@SECURE_OFF
class PublicBeaconTest(TestCase):
    """Tests for site_duration and funnel_event — public, no auth."""

    def setUp(self):
        self.client = APIClient()

    def test_site_duration_valid_beacon(self):
        # Create a matching SiteVisit first
        import hashlib

        from api.models import SiteVisit

        ip = "127.0.0.1"
        ip_hash = hashlib.sha256(ip.encode()).hexdigest()
        visit = SiteVisit.objects.create(path="/app/chat/", ip_hash=ip_hash)

        res = self.client.post(
            "/api/site-duration/",
            {"path": "/app/chat/", "duration_ms": 5000},
            format="json",
        )
        self.assertEqual(res.status_code, 204)

        visit.refresh_from_db()
        self.assertEqual(visit.duration_ms, 5000)

    def test_site_duration_missing_fields_returns_204(self):
        res = self.client.post("/api/site-duration/", {}, format="json")
        self.assertEqual(res.status_code, 204)

    def test_site_duration_too_short_ignored(self):
        res = self.client.post(
            "/api/site-duration/",
            {"path": "/app/", "duration_ms": 100},  # < 1000ms
            format="json",
        )
        self.assertEqual(res.status_code, 204)

    def test_site_duration_too_long_ignored(self):
        res = self.client.post(
            "/api/site-duration/",
            {"path": "/app/", "duration_ms": 2_000_000},  # > 30min
            format="json",
        )
        self.assertEqual(res.status_code, 204)

    def test_funnel_event_valid_action(self):
        from api.models import SiteVisit

        res = self.client.post(
            "/api/funnel-event/",
            {"page": "/register/", "action": "email_focus"},
            format="json",
        )
        self.assertEqual(res.status_code, 204)
        # Verify SiteVisit was created with funnel path convention
        self.assertTrue(SiteVisit.objects.filter(path="/register/#_email_focus").exists())

    def test_funnel_event_invalid_action_ignored(self):
        res = self.client.post(
            "/api/funnel-event/",
            {"page": "/register/", "action": "hack_the_planet"},
            format="json",
        )
        self.assertEqual(res.status_code, 204)

    def test_funnel_event_missing_page_ignored(self):
        res = self.client.post(
            "/api/funnel-event/",
            {"action": "email_focus"},
            format="json",
        )
        self.assertEqual(res.status_code, 204)


# =========================================================================
# FeedbackTest
# =========================================================================


@SECURE_OFF
class FeedbackTest(TestCase):
    """Tests for POST /api/feedback/."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("feedbacker@example.com")
        self.client.force_authenticate(self.user)

    def test_submit_feedback(self):
        from api.models import Feedback

        res = self.client.post(
            "/api/feedback/",
            {"message": "Great tool!", "category": "feature", "page": "/app/dsw/"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "submitted")
        self.assertEqual(Feedback.objects.filter(user=self.user).count(), 1)
        fb = Feedback.objects.get(user=self.user)
        self.assertEqual(fb.message, "Great tool!")
        self.assertEqual(fb.category, "feature")

    def test_submit_feedback_empty_message_rejected(self):
        res = self.client.post("/api/feedback/", {"message": ""}, format="json")
        self.assertEqual(res.status_code, 400)

    def test_submit_feedback_unauthenticated(self):
        client = APIClient()
        res = client.post("/api/feedback/", {"message": "Anon feedback"}, format="json")
        self.assertIn(res.status_code, [401, 403])


# =========================================================================
# UserInfoTest
# =========================================================================


@SECURE_OFF
class UserInfoTest(TestCase):
    """Tests for GET /api/user/."""

    def setUp(self):
        self.client = APIClient()
        self.user = _make_user("info@example.com", Tier.PRO)
        self.client.force_authenticate(self.user)

    def test_user_info_returns_profile_data(self):
        res = self.client.get("/api/user/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["email"], "info@example.com")
        self.assertEqual(data["tier"], "pro")
        self.assertIn("queries_today", data)
        self.assertIn("daily_limit", data)
        self.assertIn("subscription_active", data)

    def test_user_info_unauthenticated(self):
        client = APIClient()
        res = client.get("/api/user/")
        self.assertIn(res.status_code, [401, 403])


# =========================================================================
# CompliancePublicTest
# =========================================================================


@SECURE_OFF
class CompliancePublicTest(TestCase):
    """Tests for public compliance page and data endpoint."""

    def setUp(self):
        self.client = APIClient()

    def test_compliance_data_returns_reports(self):
        res = self.client.get("/compliance/data/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("reports", data)
        self.assertIsInstance(data["reports"], list)

    def test_compliance_page_renders(self):
        res = self.client.get("/compliance/")
        self.assertEqual(res.status_code, 200)
        # compliance_page renders HTML via Django template
        self.assertIn("text/html", res["Content-Type"])


# =========================================================================
# EnterpriseModelTest
# =========================================================================


@SECURE_OFF
class EnterpriseModelTest(TestCase):
    """Tests for EnterpriseModelResult and call_enterprise_model."""

    def test_enterprise_model_result_attributes(self):
        from api.views import EnterpriseModelResult

        result = EnterpriseModelResult(response="Hello", inference_time_ms=42)
        self.assertEqual(result.response, "Hello")
        self.assertEqual(result.inference_time_ms, 42)

    def test_enterprise_model_result_defaults(self):
        from api.views import EnterpriseModelResult

        result = EnterpriseModelResult(response="Hi")
        self.assertEqual(result.inference_time_ms, 0)

    @patch("api.views.process_query")
    def test_enterprise_model_non_enterprise_user_gets_default(self, mock_pq):
        """Non-enterprise users requesting a model silently get default pipeline."""
        mock_pq.return_value = _mock_process_query_result()
        user = _make_user("normie@enterprise.com", Tier.PRO, username="normie_ent")
        client = APIClient()
        client.force_authenticate(user)
        res = client.post("/api/chat/", {"message": "Hi", "model": "opus"}, format="json")
        self.assertEqual(res.status_code, 200)
        # process_query should be called (default pipeline), not call_enterprise_model
        mock_pq.assert_called_once()

    @patch("api.views.call_enterprise_model")
    def test_enterprise_user_gets_selected_model(self, mock_cem):
        from api.views import EnterpriseModelResult

        mock_cem.return_value = EnterpriseModelResult(response="Enterprise response", inference_time_ms=200)
        user = _make_user("boss@corp.com", Tier.FREE, username="boss")
        user.tier = "enterprise"
        user.save(update_fields=["tier"])
        client = APIClient()
        client.force_authenticate(user)
        res = client.post("/api/chat/", {"message": "Analyze this", "model": "opus"}, format="json")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["model"], "opus")
        self.assertIn("Enterprise response", data["response"])


# =========================================================================
# ThrottleClassTest
# =========================================================================


@SECURE_OFF
class ThrottleClassTest(TestCase):
    """Tests for LoginRateThrottle and RegistrationThrottle."""

    def test_login_rate_throttle_has_rate(self):
        from api.views import LoginRateThrottle

        throttle = LoginRateThrottle()
        self.assertEqual(throttle.rate, "5/minute")

    def test_registration_throttle_has_rate(self):
        from api.views import RegistrationThrottle

        throttle = RegistrationThrottle()
        self.assertEqual(throttle.rate, "5/hour")


# =========================================================================
# MakeUnsubscribeUrlTest
# =========================================================================


@SECURE_OFF
class MakeUnsubscribeUrlTest(TestCase):
    """Tests for make_unsubscribe_url helper."""

    def test_generates_valid_url(self):
        from api.views import make_unsubscribe_url

        user = _make_user("urltest@example.com")
        url = make_unsubscribe_url(user)
        self.assertIn("https://svend.ai/api/email/unsubscribe/", url)
        self.assertIn("token=", url)

    def test_url_is_verifiable(self):
        """Generated URL token can be verified by the unsubscribe view."""
        from django.core.signing import Signer

        from api.views import make_unsubscribe_url

        user = _make_user("verify_unsub@example.com")
        url = make_unsubscribe_url(user)
        token = url.split("token=")[1]

        signer = Signer(salt="email-unsubscribe")
        user_id = signer.unsign(token)
        self.assertEqual(user_id, str(user.id))

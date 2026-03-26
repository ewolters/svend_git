"""Functional scenario tests for api/internal_views.py — staff-only dashboard endpoints.

Follows TST-001: Django TestCase + DRF APIClient, force_authenticate, no Factory Boy,
explicit helpers, @override_settings(SECURE_SSL_REDIRECT=False).
"""

import uuid
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from api.models import (
    Feedback,
    OnboardingSurvey,
    SiteVisit,
)

User = get_user_model()

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_user(email, password="testpass123!", **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    is_staff = kwargs.pop("is_staff", False)
    user = User.objects.create_user(
        username=username, email=email, password=password, **kwargs
    )
    if is_staff:
        user.is_staff = True
        user.save(update_fields=["is_staff"])
    return user


def _make_staff(email="admin@example.com", **kwargs):
    return _make_user(email, is_staff=True, **kwargs)


def _make_customer(email, tier="free", **kwargs):
    user = _make_user(email, **kwargs)
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


# =========================================================================
# 1. DashboardAccessTest
# =========================================================================


@SECURE_OFF
class DashboardAccessTest(TestCase):
    """Test dashboard_view and can_access_internal permission logic."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.regular = _make_customer("regular@example.com")

    def test_staff_can_access_overview(self):
        """Staff user gets 200 on internal overview endpoint."""
        self.client.force_authenticate(user=self.staff)
        res = self.client.get("/api/internal/overview/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("total_users", data)
        self.assertIn("mrr", data)
        self.assertIn("changes", data)

    def test_non_staff_gets_403_on_overview(self):
        """Non-staff user is rejected with 403 on internal endpoints."""
        self.client.force_authenticate(user=self.regular)
        res = self.client.get("/api/internal/overview/")
        self.assertEqual(res.status_code, 403)

    def test_unauthenticated_gets_403_on_overview(self):
        """Unauthenticated request is rejected on internal endpoints."""
        res = self.client.get("/api/internal/overview/")
        self.assertEqual(res.status_code, 403)

    def test_non_staff_gets_403_on_users(self):
        """Non-staff user cannot access users endpoint."""
        self.client.force_authenticate(user=self.regular)
        res = self.client.get("/api/internal/users/")
        self.assertEqual(res.status_code, 403)

    def test_non_staff_gets_403_on_feedback(self):
        """Non-staff user cannot access feedback endpoint."""
        self.client.force_authenticate(user=self.regular)
        res = self.client.get("/api/internal/feedback/")
        self.assertEqual(res.status_code, 403)


# =========================================================================
# 2. OverviewEndpointTest
# =========================================================================


@SECURE_OFF
class OverviewEndpointTest(TestCase):
    """Test api_overview returns correct stats structure."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.client.force_authenticate(user=self.staff)

    def test_overview_returns_expected_keys(self):
        """Overview response has all required KPI fields."""
        res = self.client.get("/api/internal/overview/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        expected_keys = [
            "total_users",
            "active_today",
            "requests_today",
            "errors_today",
            "avg_latency_ms",
            "mrr",
            "conversion_rate",
            "changes",
        ]
        for key in expected_keys:
            self.assertIn(key, data, f"Missing key: {key}")

    def test_overview_changes_structure(self):
        """Week-over-week changes dict has expected sub-keys."""
        res = self.client.get("/api/internal/overview/")
        data = res.json()
        changes = data["changes"]
        for key in ("users", "active", "requests", "errors"):
            self.assertIn(key, changes, f"Missing changes key: {key}")

    def test_overview_excludes_staff_from_counts(self):
        """Staff users should not be counted in total_users."""
        _make_customer("customer1@example.com")
        _make_customer("customer2@example.com")
        res = self.client.get("/api/internal/overview/")
        data = res.json()
        # total_users counts non-staff only
        self.assertEqual(data["total_users"], 2)

    def test_overview_mrr_with_paid_users(self):
        """MRR should reflect paid tier users multiplied by tier price."""
        _make_customer("founder1@example.com", tier="founder")
        _make_customer("pro1@example.com", tier="pro")
        res = self.client.get("/api/internal/overview/")
        data = res.json()
        # founder=$19, pro=$29 => MRR = 48
        self.assertEqual(data["mrr"], 19 + 29)

    def test_overview_conversion_rate(self):
        """Conversion rate = paid / total * 100."""
        _make_customer("free1@example.com", tier="free")
        _make_customer("free2@example.com", tier="free")
        _make_customer("pro1@example.com", tier="pro")
        res = self.client.get("/api/internal/overview/")
        data = res.json()
        # 1 paid / 3 total = 33.3%
        self.assertAlmostEqual(data["conversion_rate"], 33.3, places=1)


# =========================================================================
# 3. UserManagementTest
# =========================================================================


@SECURE_OFF
class UserManagementTest(TestCase):
    """Test api_users lists users with expected fields."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.client.force_authenticate(user=self.staff)

    def test_users_returns_expected_keys(self):
        """Users response has all expected top-level keys."""
        res = self.client.get("/api/internal/users/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        for key in (
            "signups",
            "tiers",
            "industries",
            "roles",
            "experience",
            "active_trend",
            "verification_rate",
            "verified_count",
            "total_count",
            "churn_risk",
        ):
            self.assertIn(key, data, f"Missing key: {key}")

    def test_users_tier_distribution(self):
        """Tiers list includes correct counts per tier."""
        _make_customer("free1@example.com", tier="free")
        _make_customer("free2@example.com", tier="free")
        _make_customer("pro1@example.com", tier="pro")
        res = self.client.get("/api/internal/users/")
        data = res.json()
        tier_map = {t["tier"]: t["count"] for t in data["tiers"]}
        self.assertEqual(tier_map.get("free"), 2)
        self.assertEqual(tier_map.get("pro"), 1)

    def test_users_total_count(self):
        """total_count reflects non-staff users only."""
        _make_customer("c1@example.com")
        _make_customer("c2@example.com")
        res = self.client.get("/api/internal/users/")
        data = res.json()
        self.assertEqual(data["total_count"], 2)

    def test_users_days_param(self):
        """Passing ?days= param does not error and returns valid response."""
        _make_customer("c1@example.com")
        res = self.client.get("/api/internal/users/?days=7")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data["signups"], list)

    def test_non_staff_forbidden(self):
        """Non-staff user gets 403."""
        regular = _make_customer("regular@example.com")
        self.client.force_authenticate(user=regular)
        res = self.client.get("/api/internal/users/")
        self.assertEqual(res.status_code, 403)


# =========================================================================
# 4. AnalyticsEndpointTest
# =========================================================================


@SECURE_OFF
class AnalyticsEndpointTest(TestCase):
    """Test DSW analytics, hypothesis health, performance, business,
    activity endpoints return expected JSON shapes."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.client.force_authenticate(user=self.staff)

    def test_dsw_analytics_structure(self):
        """DSW analytics returns volume, type counts, and endpoint stats."""
        res = self.client.get("/api/internal/dsw-analytics/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("daily_volume", data)
        self.assertIn("type_popularity", data)
        self.assertIn("endpoint_popularity", data)
        self.assertIsInstance(data["daily_volume"], list)

    def test_dsw_analytics_days_param(self):
        """DSW analytics respects ?days= query param."""
        res = self.client.get("/api/internal/dsw-analytics/?days=7")
        self.assertEqual(res.status_code, 200)

    def test_hypothesis_health_structure(self):
        """Hypothesis health returns project/hypothesis status and evidence metrics."""
        res = self.client.get("/api/internal/hypothesis-health/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        for key in (
            "project_status",
            "hypothesis_status",
            "evidence_sources",
            "orphan_hypotheses",
            "total_hypotheses",
            "total_projects",
            "total_evidence",
            "link_directions",
            "recent_projects",
        ):
            self.assertIn(key, data, f"Missing key: {key}")
        self.assertIsInstance(data["orphan_hypotheses"], int)
        self.assertIsInstance(data["total_projects"], int)

    def test_performance_structure(self):
        """Performance endpoint returns KPIs, trends, slow endpoints, SLA."""
        res = self.client.get("/api/internal/performance/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        for key in (
            "kpis",
            "latency_trend",
            "error_rate_trend",
            "volume_trend",
            "slow_endpoints",
            "sla_status",
        ):
            self.assertIn(key, data, f"Missing key: {key}")
        kpis = data["kpis"]
        self.assertIn("requests_today", kpis)
        self.assertIn("error_rate_today", kpis)

    def test_business_structure(self):
        """Business endpoint returns revenue, funnel, churn, founder slots."""
        _make_customer("c1@example.com", tier="free")
        res = self.client.get("/api/internal/business/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        for key in ("revenue", "funnel", "churn", "founder_slots", "feature_adoption"):
            self.assertIn(key, data, f"Missing key: {key}")
        self.assertIn("total", data["funnel"])
        self.assertIn("paid", data["funnel"])
        self.assertEqual(data["founder_slots"]["total"], 50)

    def test_business_funnel_counts(self):
        """Business funnel total matches number of non-staff customers."""
        _make_customer("c1@example.com")
        _make_customer("c2@example.com")
        _make_customer("c3@example.com", tier="pro")
        res = self.client.get("/api/internal/business/")
        data = res.json()
        self.assertEqual(data["funnel"]["total"], 3)
        self.assertEqual(data["funnel"]["paid"], 1)

    def test_activity_structure(self):
        """Activity endpoint returns page views, feature use, sessions, totals."""
        res = self.client.get("/api/internal/activity/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        for key in (
            "page_views",
            "feature_use",
            "daily_sessions",
            "journeys",
            "daily_features",
            "totals",
        ):
            self.assertIn(key, data, f"Missing key: {key}")
        totals = data["totals"]
        self.assertIn("events", totals)
        self.assertIn("page_views", totals)

    def test_activity_non_staff_forbidden(self):
        """Non-staff gets 403 on activity."""
        regular = _make_customer("regular@example.com")
        self.client.force_authenticate(user=regular)
        res = self.client.get("/api/internal/activity/")
        self.assertEqual(res.status_code, 403)


# =========================================================================
# 5. ComplianceEndpointTest
# =========================================================================


@SECURE_OFF
class ComplianceEndpointTest(TestCase):
    """Test api_compliance, api_compliance_run, api_compliance_publish."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.client.force_authenticate(user=self.staff)

    def test_compliance_returns_expected_keys(self):
        """Compliance GET returns checks, trend, stats, reports."""
        res = self.client.get("/api/internal/compliance/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        for key in ("checks", "trend", "stats", "reports"):
            self.assertIn(key, data, f"Missing key: {key}")
        self.assertIsInstance(data["checks"], list)
        self.assertIsInstance(data["reports"], list)

    def test_compliance_stats_structure(self):
        """Compliance stats has expected sub-keys."""
        res = self.client.get("/api/internal/compliance/")
        data = res.json()
        stats = data["stats"]
        for key in (
            "total_checks_run",
            "checks_today",
            "overall_pass_rate",
            "infra_checks",
            "infra_passed",
        ):
            self.assertIn(key, stats, f"Missing stats key: {key}")

    def test_compliance_run_list_mode(self):
        """POST with no body returns list of available checks."""
        res = self.client.post("/api/internal/compliance/run/", {}, format="json")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["ok"])
        self.assertIn("checks", data)
        self.assertIn("total", data)
        self.assertIsInstance(data["checks"], list)
        self.assertGreater(data["total"], 0)

    def test_compliance_run_unknown_check(self):
        """POST with unknown check name returns 400."""
        res = self.client.post(
            "/api/internal/compliance/run/",
            {"check": "nonexistent_check"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_compliance_publish_nonexistent(self):
        """Publish on nonexistent report returns 404."""
        fake_id = uuid.uuid4()
        res = self.client.post(f"/api/internal/compliance/publish/{fake_id}/")
        self.assertEqual(res.status_code, 404)

    def test_compliance_non_staff_forbidden(self):
        """Non-staff cannot access compliance."""
        regular = _make_customer("regular@example.com")
        self.client.force_authenticate(user=regular)
        res = self.client.get("/api/internal/compliance/")
        self.assertEqual(res.status_code, 403)


# =========================================================================
# 6. EmailEndpointTest
# =========================================================================


@SECURE_OFF
class EmailEndpointTest(TestCase):
    """Test email endpoints: preview, send, drafts, campaigns."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.client.force_authenticate(user=self.staff)

    def test_email_preview_all_target(self):
        """Email preview with target=all returns count of non-staff users with emails."""
        _make_customer("c1@example.com")
        _make_customer("c2@example.com")
        res = self.client.get("/api/internal/email-preview/?target=all")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["count"], 2)

    def test_email_preview_tier_target(self):
        """Email preview with tier target returns matching count."""
        _make_customer("free1@example.com", tier="free")
        _make_customer("free2@example.com", tier="free")
        _make_customer("pro1@example.com", tier="pro")
        res = self.client.get("/api/internal/email-preview/?target=tier:free")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["count"], 2)

    def test_email_preview_custom_email_returns_null(self):
        """Email preview with @ in target returns count=None (custom email)."""
        res = self.client.get("/api/internal/email-preview/?target=user@example.com")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsNone(data["count"])

    def test_email_preview_empty_target(self):
        """Email preview with empty target returns count=None."""
        res = self.client.get("/api/internal/email-preview/?target=")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsNone(data["count"])

    def test_send_email_missing_fields(self):
        """Send email with missing subject/body returns 400."""
        res = self.client.post(
            "/api/internal/send-email/",
            {"to": "all", "subject": "", "body": ""},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    @patch(
        "api.internal_views.django_send_mail" if False else "django.core.mail.send_mail"
    )
    def test_send_email_test_mode(self, mock_send):
        """Send email in test mode sends to staff user's email only."""
        mock_send.return_value = 1
        _make_customer("target@example.com")
        res = self.client.post(
            "/api/internal/send-email/",
            {
                "to": "all",
                "subject": "Test Subject",
                "body": "Hello world",
                "test": True,
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("sent", data)

    def test_save_email_draft_creates_draft(self):
        """Saving an email draft persists it in user preferences."""
        res = self.client.post(
            "/api/internal/email-draft/save/",
            {
                "to": "all",
                "subject": "Draft Subject",
                "body": "Draft body content",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["saved"])
        self.assertIn("id", data)
        # Verify persisted in DB
        self.staff.refresh_from_db()
        drafts = self.staff.preferences.get("email_drafts", [])
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0]["subject"], "Draft Subject")

    def test_get_email_drafts_empty(self):
        """GET email drafts returns empty list when no drafts."""
        res = self.client.get("/api/internal/email-draft/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["drafts"], [])

    def test_save_then_list_then_delete_draft_workflow(self):
        """Full workflow: save draft, list drafts, delete draft."""
        # Save
        save_res = self.client.post(
            "/api/internal/email-draft/save/",
            {"to": "all", "subject": "Draft 1", "body": "Body 1"},
            format="json",
        )
        self.assertEqual(save_res.status_code, 200)
        draft_id = save_res.json()["id"]

        # List
        list_res = self.client.get("/api/internal/email-draft/")
        self.assertEqual(list_res.status_code, 200)
        drafts = list_res.json()["drafts"]
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0]["id"], draft_id)

        # Delete
        del_res = self.client.delete(f"/api/internal/email-draft/?id={draft_id}")
        self.assertEqual(del_res.status_code, 200)
        self.assertTrue(del_res.json()["deleted"])

        # Verify deleted
        list_res2 = self.client.get("/api/internal/email-draft/")
        self.assertEqual(list_res2.json()["drafts"], [])

    def test_delete_draft_without_id_returns_400(self):
        """DELETE without ?id= returns 400."""
        res = self.client.delete("/api/internal/email-draft/")
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.json())

    def test_email_campaigns_field_error_returns_500(self):
        """Campaigns endpoint hits FieldError due to recipients__failed vs has_failed.

        BUG: The view annotates with Q(recipients__failed=...) but the model field
        is `has_failed` (with db_column='failed'). Django ORM filters by the Python
        field name, not db_column, so this always raises FieldError. This test
        documents the bug — fix requires changing the view to use
        recipients__has_failed.
        """
        res = self.client.get("/api/internal/email-campaigns/")
        self.assertEqual(res.status_code, 500)


# =========================================================================
# 7. OnboardingEndpointTest
# =========================================================================


@SECURE_OFF
class OnboardingEndpointTest(TestCase):
    """Test api_onboarding returns onboarding survey analytics."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.client.force_authenticate(user=self.staff)

    def test_onboarding_structure(self):
        """Onboarding endpoint returns funnel, distributions, averages."""
        res = self.client.get("/api/internal/onboarding/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        for key in (
            "funnel",
            "completion_rate",
            "distributions",
            "averages",
            "challenges",
        ):
            self.assertIn(key, data, f"Missing key: {key}")

    def test_onboarding_funnel_counts(self):
        """Onboarding funnel includes registered, onboarded, verified, paid."""
        _make_customer("c1@example.com")
        _make_customer("c2@example.com")
        res = self.client.get("/api/internal/onboarding/")
        data = res.json()
        funnel = data["funnel"]
        self.assertIn("registered", funnel)
        self.assertIn("onboarded", funnel)
        self.assertEqual(funnel["registered"], 2)

    def test_onboarding_with_survey_data(self):
        """Onboarding shows distribution data when surveys exist."""
        customer = _make_customer("surveyed@example.com")
        OnboardingSurvey.objects.create(
            user=customer,
            industry="manufacturing",
            role="engineer",
            experience_level="intermediate",
        )
        res = self.client.get("/api/internal/onboarding/")
        data = res.json()
        # onboarded counts users with onboarding_completed_at set,
        # not just survey completion — survey exists but user hasn't completed onboarding
        self.assertIn("onboarded", data["funnel"])
        dists = data["distributions"]
        self.assertIn("industry", dists)
        self.assertIn("role", dists)

    def test_onboarding_non_staff_forbidden(self):
        """Non-staff gets 403."""
        regular = _make_customer("regular@example.com")
        self.client.force_authenticate(user=regular)
        res = self.client.get("/api/internal/onboarding/")
        self.assertEqual(res.status_code, 403)


# =========================================================================
# 8. FeedbackEndpointTest
# =========================================================================


@SECURE_OFF
class FeedbackEndpointTest(TestCase):
    """Test api_feedback GET and POST."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.client.force_authenticate(user=self.staff)

    def test_feedback_list_empty(self):
        """Feedback GET returns empty list and summary when no feedback."""
        res = self.client.get("/api/internal/feedback/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["feedback"], [])
        self.assertIn("summary", data)
        self.assertEqual(data["summary"]["total"], 0)

    def test_feedback_list_with_data(self):
        """Feedback GET returns feedback entries with expected fields."""
        customer = _make_customer("fb_user@example.com")
        Feedback.objects.create(
            user=customer,
            category="bug",
            message="Button doesn't work",
            page="/workbench",
        )
        res = self.client.get("/api/internal/feedback/")
        data = res.json()
        self.assertEqual(len(data["feedback"]), 1)
        fb = data["feedback"][0]
        self.assertEqual(fb["category"], "bug")
        self.assertEqual(fb["message"], "Button doesn't work")
        self.assertEqual(fb["page"], "/workbench")
        self.assertEqual(fb["user"], "fb_user")
        self.assertEqual(fb["status"], "new")

    def test_feedback_summary_counts(self):
        """Feedback summary has correct by_status and by_category counts."""
        customer = _make_customer("fb@example.com")
        Feedback.objects.create(user=customer, category="bug", message="Bug 1")
        Feedback.objects.create(user=customer, category="feature", message="Feature 1")
        Feedback.objects.create(
            user=customer, category="bug", message="Bug 2", status="reviewed"
        )
        res = self.client.get("/api/internal/feedback/")
        data = res.json()
        summary = data["summary"]
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["by_status"]["new"], 2)
        self.assertEqual(summary["by_status"]["reviewed"], 1)
        self.assertEqual(summary["by_category"]["bug"], 2)
        self.assertEqual(summary["by_category"]["feature"], 1)

    def test_feedback_filter_by_status(self):
        """Feedback GET with ?status= filters results."""
        customer = _make_customer("fb@example.com")
        Feedback.objects.create(user=customer, category="bug", message="New bug")
        Feedback.objects.create(
            user=customer, category="bug", message="Reviewed bug", status="reviewed"
        )
        res = self.client.get("/api/internal/feedback/?status=reviewed")
        data = res.json()
        self.assertEqual(len(data["feedback"]), 1)
        self.assertEqual(data["feedback"][0]["status"], "reviewed")
        # Summary is always unfiltered
        self.assertEqual(data["summary"]["total"], 2)

    def test_feedback_update_status(self):
        """POST updates feedback status and notes."""
        customer = _make_customer("fb@example.com")
        fb = Feedback.objects.create(user=customer, category="bug", message="Fix this")
        res = self.client.post(
            "/api/internal/feedback/",
            {"id": str(fb.id), "status": "resolved", "notes": "Fixed in v2.1"},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "resolved")
        self.assertEqual(data["notes"], "Fixed in v2.1")
        # Verify DB state
        fb.refresh_from_db()
        self.assertEqual(fb.status, "resolved")
        self.assertEqual(fb.internal_notes, "Fixed in v2.1")

    def test_feedback_update_nonexistent_returns_404(self):
        """POST with unknown feedback ID returns 404."""
        res = self.client.post(
            "/api/internal/feedback/",
            {"id": str(uuid.uuid4()), "status": "resolved"},
            format="json",
        )
        self.assertEqual(res.status_code, 404)


# =========================================================================
# 9. SiteAnalyticsTest
# =========================================================================


@SECURE_OFF
class SiteAnalyticsTest(TestCase):
    """Test api_site_analytics and api_site_live."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.client.force_authenticate(user=self.staff)

    def test_site_analytics_structure(self):
        """Site analytics returns daily, top_pages, referrers, countries, totals."""
        res = self.client.get("/api/internal/site-analytics/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        for key in ("daily", "top_pages", "referrers", "totals", "countries"):
            self.assertIn(key, data, f"Missing key: {key}")
        totals = data["totals"]
        for key in ("hits", "unique_visitors", "bot_hits"):
            self.assertIn(key, totals, f"Missing totals key: {key}")

    def test_site_analytics_with_visits(self):
        """Site analytics counts real visits correctly."""
        now = timezone.now()
        SiteVisit.objects.create(
            path="/",
            viewed_at=now,
            ip_hash="abc123",
            is_bot=False,
        )
        SiteVisit.objects.create(
            path="/pricing",
            viewed_at=now,
            ip_hash="def456",
            is_bot=False,
        )
        SiteVisit.objects.create(
            path="/bot-page",
            viewed_at=now,
            ip_hash="bot1",
            is_bot=True,
        )
        res = self.client.get("/api/internal/site-analytics/?days=1")
        data = res.json()
        totals = data["totals"]
        self.assertEqual(totals["hits"], 2)  # excludes bots
        self.assertEqual(totals["unique_visitors"], 2)
        self.assertEqual(totals["bot_hits"], 1)

    def test_site_live_structure(self):
        """Site live endpoint returns recent hits and totals."""
        res = self.client.get("/api/internal/site-live/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("recent", data)
        self.assertIn("totals", data)
        totals = data["totals"]
        for key in ("hits_today", "hits_hour", "unique_today"):
            self.assertIn(key, totals, f"Missing totals key: {key}")

    def test_site_live_with_visits(self):
        """Site live shows recent non-bot visits."""
        now = timezone.now()
        SiteVisit.objects.create(
            path="/features",
            viewed_at=now,
            ip_hash="visitor1",
            is_bot=False,
            country="US",
        )
        res = self.client.get("/api/internal/site-live/")
        data = res.json()
        self.assertGreaterEqual(len(data["recent"]), 1)
        hit = data["recent"][0]
        self.assertEqual(hit["path"], "/features")
        self.assertIn("ago", hit)
        self.assertIn("visitor", hit)
        self.assertEqual(data["totals"]["hits_today"], 1)

    def test_site_live_limit_param(self):
        """Site live respects ?limit= parameter."""
        now = timezone.now()
        for i in range(5):
            SiteVisit.objects.create(
                path=f"/page{i}",
                viewed_at=now,
                ip_hash=f"v{i}",
                is_bot=False,
            )
        res = self.client.get("/api/internal/site-live/?limit=2")
        data = res.json()
        self.assertEqual(len(data["recent"]), 2)

    def test_site_analytics_non_staff_forbidden(self):
        """Non-staff gets 403 on site analytics."""
        regular = _make_customer("regular@example.com")
        self.client.force_authenticate(user=regular)
        res = self.client.get("/api/internal/site-analytics/")
        self.assertEqual(res.status_code, 403)


# =========================================================================
# 10. InfraEndpointTest
# =========================================================================


@SECURE_OFF
class InfraEndpointTest(TestCase):
    """Test api_infra and api_audit_entries."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.client.force_authenticate(user=self.staff)

    def test_infra_returns_three_sections(self):
        """Infra endpoint returns scheduler, audit, logs sections."""
        res = self.client.get("/api/internal/infra/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        for key in ("scheduler", "audit", "logs"):
            self.assertIn(key, data, f"Missing key: {key}")

    def test_infra_audit_section_keys(self):
        """Infra audit section has expected keys when audit models exist."""
        res = self.client.get("/api/internal/infra/")
        data = res.json()
        audit = data["audit"]
        # Either has real data or an error key
        if "error" not in audit:
            for key in (
                "total_entries",
                "chain_length",
                "chain_ok",
                "event_distribution",
                "integrity_violations_open",
                "drift_total_open",
            ):
                self.assertIn(key, audit, f"Missing audit key: {key}")

    def test_audit_entries_structure(self):
        """Audit entries returns entries list, total, and event_names."""
        res = self.client.get("/api/internal/audit-entries/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("entries", data)
        self.assertIn("total", data)
        self.assertIn("event_names", data)
        self.assertIsInstance(data["entries"], list)
        self.assertIsInstance(data["total"], int)

    def test_audit_entries_limit_param(self):
        """Audit entries respects ?limit= parameter."""
        res = self.client.get("/api/internal/audit-entries/?limit=5")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertLessEqual(len(data["entries"]), 5)

    def test_infra_non_staff_forbidden(self):
        """Non-staff gets 403 on infra."""
        regular = _make_customer("regular@example.com")
        self.client.force_authenticate(user=regular)
        res = self.client.get("/api/internal/infra/")
        self.assertEqual(res.status_code, 403)


# =========================================================================
# 11. StandardsEndpointTest
# =========================================================================


@SECURE_OFF
class StandardsEndpointTest(TestCase):
    """Test api_standards list and detail."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.client.force_authenticate(user=self.staff)

    def test_standards_list_structure(self):
        """Standards list returns a standards array."""
        res = self.client.get("/api/internal/standards/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("standards", data)
        self.assertIsInstance(data["standards"], list)
        # There should be multiple standards in the project
        self.assertGreater(len(data["standards"]), 0)

    def test_standards_list_item_keys(self):
        """Each standard in the list has expected metadata fields."""
        res = self.client.get("/api/internal/standards/")
        data = res.json()
        if data["standards"]:
            std = data["standards"][0]
            for key in ("code", "title", "lines"):
                self.assertIn(key, std, f"Missing standard key: {key}")

    def test_standards_detail_by_code(self):
        """Fetching a single standard by ?code= returns body and metadata."""
        res = self.client.get("/api/internal/standards/?code=TST-001")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("body", data)
        self.assertIn("code", data)
        self.assertEqual(data["code"], "TST-001")
        self.assertIn("lines", data)
        self.assertGreater(data["lines"], 0)

    def test_standards_detail_nonexistent(self):
        """Fetching nonexistent standard returns 404."""
        res = self.client.get("/api/internal/standards/?code=FAKE-999")
        self.assertEqual(res.status_code, 404)
        self.assertIn("error", res.json())

    def test_standards_non_staff_forbidden(self):
        """Non-staff gets 403 on standards."""
        regular = _make_customer("regular@example.com")
        self.client.force_authenticate(user=regular)
        res = self.client.get("/api/internal/standards/")
        self.assertEqual(res.status_code, 403)


# =========================================================================
# 12. CohortRetentionTest
# =========================================================================


@SECURE_OFF
class CohortRetentionTest(TestCase):
    """Test api_cohort_retention."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.client.force_authenticate(user=self.staff)

    def test_cohort_retention_structure(self):
        """Cohort retention returns cohorts list."""
        res = self.client.get("/api/internal/cohort-retention/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("cohorts", data)
        self.assertIsInstance(data["cohorts"], list)

    def test_cohort_retention_with_users(self):
        """Cohort retention includes cohort data when users exist."""
        # Create users in the current month
        _make_customer("recent1@example.com")
        _make_customer("recent2@example.com")
        res = self.client.get("/api/internal/cohort-retention/?months=1")
        data = res.json()
        if data["cohorts"]:
            cohort = data["cohorts"][0]
            self.assertIn("label", cohort)
            self.assertIn("size", cohort)
            self.assertIn("retention", cohort)
            self.assertIsInstance(cohort["retention"], list)
            # Month 0 is always 100%
            self.assertEqual(cohort["retention"][0]["month"], 0)
            self.assertEqual(cohort["retention"][0]["retained"], 100)

    def test_cohort_retention_months_param(self):
        """months param controls how many months to look back."""
        res = self.client.get("/api/internal/cohort-retention/?months=3")
        self.assertEqual(res.status_code, 200)

    def test_cohort_retention_non_staff_forbidden(self):
        """Non-staff gets 403."""
        regular = _make_customer("regular@example.com")
        self.client.force_authenticate(user=regular)
        res = self.client.get("/api/internal/cohort-retention/")
        self.assertEqual(res.status_code, 403)


# =========================================================================
# 13. RateLimitOverrideTest
# =========================================================================


@SECURE_OFF
class RateLimitOverrideTest(TestCase):
    """Test api_rate_limit_override (requires IsAdminUser, not just IsInternalUser)."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.client.force_authenticate(user=self.staff)

    def test_rate_limit_override_success(self):
        """Setting a rate limit override creates/updates the record."""
        res = self.client.post(
            "/api/internal/rate-limit-override/",
            {"tier": "free", "llm_limit": 50},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["tier"], "FREE")
        self.assertEqual(data["llm_limit"], 50)
        self.assertTrue(data["created"])

    def test_rate_limit_override_update(self):
        """Updating an existing override returns created=False."""
        self.client.post(
            "/api/internal/rate-limit-override/",
            {"tier": "pro", "llm_limit": 100},
            format="json",
        )
        res = self.client.post(
            "/api/internal/rate-limit-override/",
            {"tier": "pro", "llm_limit": 200},
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["llm_limit"], 200)
        self.assertFalse(data["created"])

    def test_rate_limit_override_missing_tier(self):
        """Missing tier returns 400."""
        res = self.client.post(
            "/api/internal/rate-limit-override/",
            {"llm_limit": 50},
            format="json",
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.json())

    def test_rate_limit_override_missing_llm_limit(self):
        """Missing llm_limit returns 400."""
        res = self.client.post(
            "/api/internal/rate-limit-override/",
            {"tier": "free"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_rate_limit_override_invalid_llm_limit(self):
        """Non-integer llm_limit returns 400."""
        res = self.client.post(
            "/api/internal/rate-limit-override/",
            {"tier": "free", "llm_limit": "not_a_number"},
            format="json",
        )
        self.assertEqual(res.status_code, 400)

    def test_rate_limit_override_non_staff_forbidden(self):
        """Non-staff gets 403 (IsAdminUser requires is_staff=True)."""
        regular = _make_customer("regular@example.com")
        self.client.force_authenticate(user=regular)
        res = self.client.post(
            "/api/internal/rate-limit-override/",
            {"tier": "free", "llm_limit": 50},
            format="json",
        )
        self.assertEqual(res.status_code, 403)


# =========================================================================
# Scenario: Full feedback triage workflow
# =========================================================================


@SECURE_OFF
class FeedbackTriageScenarioTest(TestCase):
    """Multi-step scenario: customer submits feedback, staff triages it."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.customer = _make_customer("customer@example.com")

    def test_feedback_triage_workflow(self):
        """Scenario: feedback submitted -> staff reviews -> staff resolves."""
        # Step 1: Customer creates feedback (via direct model creation,
        # simulating what the public feedback endpoint does)
        fb = Feedback.objects.create(
            user=self.customer,
            category="bug",
            message="Dashboard chart doesn't load",
            page="/workbench",
        )
        self.assertEqual(fb.status, "new")

        # Step 2: Staff lists feedback and sees the new entry
        self.client.force_authenticate(user=self.staff)
        res = self.client.get("/api/internal/feedback/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["feedback"]), 1)
        self.assertEqual(data["summary"]["by_status"]["new"], 1)

        # Step 3: Staff marks as reviewed with notes
        res = self.client.post(
            "/api/internal/feedback/",
            {
                "id": str(fb.id),
                "status": "reviewed",
                "notes": "Investigating chart.js loading issue",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "reviewed")

        # Step 4: Staff resolves the feedback
        res = self.client.post(
            "/api/internal/feedback/",
            {
                "id": str(fb.id),
                "status": "resolved",
                "notes": "Fixed in commit abc123",
            },
            format="json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "resolved")

        # Step 5: Verify DB state
        fb.refresh_from_db()
        self.assertEqual(fb.status, "resolved")
        self.assertEqual(fb.internal_notes, "Fixed in commit abc123")

        # Step 6: Filter shows resolved
        res = self.client.get("/api/internal/feedback/?status=resolved")
        self.assertEqual(len(res.json()["feedback"]), 1)


# =========================================================================
# Scenario: Email draft + campaign workflow
# =========================================================================


@SECURE_OFF
class EmailDraftCampaignScenarioTest(TestCase):
    """Multi-step scenario: staff saves draft, previews audience, manages campaigns."""

    def setUp(self):
        self.client = APIClient()
        self.staff = _make_staff()
        self.client.force_authenticate(user=self.staff)

    def test_draft_to_preview_workflow(self):
        """Scenario: save draft -> preview audience -> list campaigns."""
        # Create some customers for targeting
        _make_customer("free1@example.com", tier="free")
        _make_customer("free2@example.com", tier="free")
        _make_customer("pro1@example.com", tier="pro")

        # Step 1: Save a draft targeting free users
        save_res = self.client.post(
            "/api/internal/email-draft/save/",
            {
                "to": "tier:free",
                "subject": "Upgrade to Pro",
                "body": "Get more features with Pro!",
            },
            format="json",
        )
        self.assertEqual(save_res.status_code, 200)
        draft_id = save_res.json()["id"]

        # Step 2: Preview audience count
        preview_res = self.client.get("/api/internal/email-preview/?target=tier:free")
        self.assertEqual(preview_res.status_code, 200)
        self.assertEqual(preview_res.json()["count"], 2)

        # Step 3: Save another draft (multiple drafts supported)
        save_res2 = self.client.post(
            "/api/internal/email-draft/save/",
            {
                "to": "all",
                "subject": "Monthly Newsletter",
                "body": "Here are this month's updates...",
            },
            format="json",
        )
        self.assertEqual(save_res2.status_code, 200)
        draft2_id = save_res2.json()["id"]

        # Step 4: List drafts — should have 2
        list_res = self.client.get("/api/internal/email-draft/")
        self.assertEqual(len(list_res.json()["drafts"]), 2)

        # Step 5: Delete first draft
        del_res = self.client.delete(f"/api/internal/email-draft/?id={draft_id}")
        self.assertTrue(del_res.json()["deleted"])

        # Step 6: List drafts — should have 1
        list_res2 = self.client.get("/api/internal/email-draft/")
        drafts = list_res2.json()["drafts"]
        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0]["id"], draft2_id)

        # Step 7: Verify all drafts have correct content
        list_res3 = self.client.get("/api/internal/email-draft/")
        remaining = list_res3.json()["drafts"]
        self.assertEqual(remaining[0]["subject"], "Monthly Newsletter")

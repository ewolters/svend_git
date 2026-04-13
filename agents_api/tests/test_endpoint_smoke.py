"""Endpoint smoke tests — verify every endpoint responds correctly.

Tests three properties for each endpoint:
1. Unauthenticated requests are rejected (401/403)
2. Authenticated requests don't return 500
3. Invalid input returns 400, not 500

Standard: CAL-001 §7 (Endpoint Coverage)
Compliance: SOC 2 CC4.1, CC7.2
"""

from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from accounts.models import Tier, User


def _make_user(email, tier=Tier.PRO, staff=False):
    """Create a test user."""
    username = email.split("@")[0].replace(".", "_")
    u = User.objects.create_user(username=username, email=email, password="testpass123")
    u.tier = tier
    u.is_staff = staff
    u.email_verified = True
    u.save()
    return u


# Minimal valid payloads per endpoint type
DUMMY_UUID = "00000000-0000-0000-0000-000000000000"

# All test classes disable SSL redirect + rate limiting
SMOKE_SETTINGS = {"RATELIMIT_ENABLE": False, "SECURE_SSL_REDIRECT": False}


@override_settings(**SMOKE_SETTINGS)
class PublicAPISmokeTest(TestCase):
    """Smoke tests for public API endpoints (api/views.py)."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("smoke-public@test.com")

    def setUp(self):
        self.auth = APIClient()
        self.auth.force_authenticate(self.user)
        self.anon = APIClient()

    # --- Auth endpoints (public by design) ---

    def test_health(self):
        res = self.anon.get("/api/health/")
        self.assertEqual(res.status_code, 200)

    def test_register_invalid(self):
        res = self.anon.post("/api/auth/register/", {}, format="json")
        self.assertIn(res.status_code, [400, 422])

    def test_login_invalid(self):
        res = self.anon.post("/api/auth/login/", {}, format="json")
        self.assertIn(res.status_code, [400, 401])

    def test_logout_unauth(self):
        res = self.anon.post("/api/auth/logout/")
        self.assertIn(res.status_code, [401, 403])

    def test_me_unauth(self):
        res = self.anon.get("/api/auth/me/")
        self.assertIn(res.status_code, [401, 403])

    def test_me_auth(self):
        res = self.auth.get("/api/auth/me/")
        self.assertNotEqual(res.status_code, 500)

    def test_profile_update_unauth(self):
        res = self.anon.put("/api/auth/profile/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_user_info_auth(self):
        res = self.auth.get("/api/user/")
        self.assertNotEqual(res.status_code, 500)

    # --- Conversations ---

    def test_conversations_unauth(self):
        res = self.anon.get("/api/conversations/")
        self.assertIn(res.status_code, [401, 403])

    def test_conversations_auth(self):
        res = self.auth.get("/api/conversations/")
        self.assertNotEqual(res.status_code, 500)

    # --- Chat ---

    def test_chat_unauth(self):
        res = self.anon.post("/api/chat/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    # --- Events ---

    def test_events_unauth(self):
        res = self.anon.post("/api/events/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    # --- Feedback ---

    def test_feedback_unauth(self):
        res = self.anon.post("/api/feedback/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_feedback_auth(self):
        res = self.auth.post("/api/feedback/", {"message": "test"}, format="json")
        self.assertNotEqual(res.status_code, 500)

    # --- Flag message ---

    def test_flag_message_unauth(self):
        res = self.anon.post(f"/api/messages/{DUMMY_UUID}/flag/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    # --- Share ---

    def test_share_unauth(self):
        res = self.anon.post("/api/share/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    # --- Export ---

    def test_export_pdf_unauth(self):
        res = self.anon.post("/api/export/pdf/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])


@override_settings(**SMOKE_SETTINGS)
class DSWSmokeTest(TestCase):
    """Smoke tests for DSW analysis endpoints."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("smoke-dsw@test.com")

    def setUp(self):
        self.auth = APIClient()
        self.auth.force_authenticate(self.user)
        self.anon = APIClient()

    def test_dsw_analysis_unauth(self):
        res = self.anon.post("/api/dsw/analysis/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_dsw_analysis_empty(self):
        res = self.auth.post("/api/dsw/analysis/", {}, format="json")
        # May return 400 (validation) or 401 (auth check on empty payload)
        self.assertIn(res.status_code, [400, 401, 422])

    def test_dsw_analysis_valid(self):
        """Minimal valid DSW analysis request."""
        res = self.auth.post(
            "/api/dsw/analysis/",
            {
                "analysis_type": "stats",
                "analysis_id": "descriptive",
                "data": {"x": [1, 2, 3, 4, 5]},
                "config": {"var1": "x"},
            },
            format="json",
        )
        self.assertNotEqual(res.status_code, 500)

    def test_dsw_models_unauth(self):
        res = self.anon.get("/api/dsw/models/")
        self.assertIn(res.status_code, [401, 403])

    def test_dsw_from_intent_unauth(self):
        res = self.anon.post("/api/dsw/from-intent/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])


@override_settings(**SMOKE_SETTINGS)
class SPCSmokeTest(TestCase):
    """Smoke tests for SPC endpoints."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("smoke-spc@test.com")

    def setUp(self):
        self.auth = APIClient()
        self.auth.force_authenticate(self.user)
        self.anon = APIClient()

    def test_spc_analyze_unauth(self):
        res = self.anon.post("/api/spc/analyze/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_spc_analyze_empty(self):
        res = self.auth.post("/api/spc/analyze/", {}, format="json")
        self.assertIn(res.status_code, [400, 401, 422])

    def test_spc_chart_unauth(self):
        res = self.anon.post("/api/spc/chart/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_spc_capability_unauth(self):
        res = self.anon.post("/api/spc/capability/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])


@override_settings(**SMOKE_SETTINGS)
class QualityToolsSmokeTest(TestCase):
    """Smoke tests for quality tool endpoints (FMEA, RCA, A3, VSM, Hoshin)."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("smoke-quality@test.com")

    def setUp(self):
        self.auth = APIClient()
        self.auth.force_authenticate(self.user)
        self.anon = APIClient()

    # FMEA
    def test_fmea_list_unauth(self):
        res = self.anon.get("/api/fmea/")
        self.assertIn(res.status_code, [401, 403])

    def test_fmea_list_auth(self):
        res = self.auth.get("/api/fmea/")
        self.assertNotEqual(res.status_code, 500)

    def test_fmea_create_unauth(self):
        res = self.anon.post("/api/fmea/create/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    # RCA
    def test_rca_sessions_unauth(self):
        res = self.anon.get("/api/rca/sessions/")
        self.assertIn(res.status_code, [401, 403])

    def test_rca_sessions_auth(self):
        res = self.auth.get("/api/rca/sessions/")
        self.assertNotEqual(res.status_code, 500)

    def test_rca_critique_unauth(self):
        res = self.anon.post("/api/rca/critique/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    # A3
    def test_a3_list_unauth(self):
        res = self.anon.get("/api/a3/")
        self.assertIn(res.status_code, [401, 403])

    def test_a3_list_auth(self):
        res = self.auth.get("/api/a3/")
        self.assertNotEqual(res.status_code, 500)

    def test_a3_create_unauth(self):
        res = self.anon.post("/api/a3/create/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    # VSM
    def test_vsm_list_unauth(self):
        res = self.anon.get("/api/vsm/")
        self.assertIn(res.status_code, [401, 403])

    def test_vsm_list_auth(self):
        res = self.auth.get("/api/vsm/")
        self.assertNotEqual(res.status_code, 500)

    # Hoshin
    def test_hoshin_sites_unauth(self):
        res = self.anon.get("/api/hoshin/sites/")
        self.assertIn(res.status_code, [401, 403])

    def test_hoshin_projects_unauth(self):
        res = self.anon.get("/api/hoshin/projects/")
        self.assertIn(res.status_code, [401, 403])

    def test_hoshin_projects_auth(self):
        res = self.auth.get("/api/hoshin/projects/")
        self.assertNotEqual(res.status_code, 500)

    # Whiteboard
    def test_whiteboard_list_unauth(self):
        res = self.anon.get("/api/whiteboard/boards/")
        self.assertIn(res.status_code, [401, 403])

    def test_whiteboard_list_auth(self):
        res = self.auth.get("/api/whiteboard/boards/")
        self.assertNotEqual(res.status_code, 500)

    # ISO
    def test_iso_dashboard_unauth(self):
        res = self.anon.get("/api/iso/dashboard/")
        self.assertIn(res.status_code, [401, 403])

    def test_iso_dashboard_auth(self):
        res = self.auth.get("/api/iso/dashboard/")
        self.assertNotEqual(res.status_code, 500)

    def test_iso_ncrs_unauth(self):
        res = self.anon.get("/api/iso/ncrs/")
        self.assertIn(res.status_code, [401, 403])

    # Learn
    def test_learn_modules_unauth(self):
        res = self.anon.get("/api/learn/modules/")
        # May return 302 (login redirect) instead of 401/403
        self.assertIn(res.status_code, [302, 401, 403])

    def test_learn_modules_auth(self):
        res = self.auth.get("/api/learn/modules/")
        self.assertNotEqual(res.status_code, 500)

    # Triage
    def test_triage_clean_unauth(self):
        res = self.anon.post("/api/triage/clean/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_triage_datasets_unauth(self):
        res = self.anon.get("/api/triage/datasets/")
        self.assertIn(res.status_code, [401, 403])

    # Forecast
    def test_forecast_unauth(self):
        res = self.anon.post("/api/forecast/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    # Guide
    def test_guide_chat_unauth(self):
        res = self.anon.post("/api/guide/chat/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    # CAPA
    def test_capa_list_unauth(self):
        res = self.anon.get("/api/capa/")
        self.assertIn(res.status_code, [401, 403])

    def test_capa_list_auth(self):
        res = self.auth.get("/api/capa/")
        self.assertNotEqual(res.status_code, 500)

    # Reports
    def test_reports_list_unauth(self):
        res = self.anon.get("/api/reports/")
        self.assertIn(res.status_code, [401, 403])

    def test_reports_list_auth(self):
        res = self.auth.get("/api/reports/")
        self.assertNotEqual(res.status_code, 500)


@override_settings(**SMOKE_SETTINGS)
class CoreViewsSmokeTest(TestCase):
    """Smoke tests for core views (projects, hypotheses, evidence)."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("smoke-core@test.com")

    def setUp(self):
        self.auth = APIClient()
        self.auth.force_authenticate(self.user)
        self.anon = APIClient()

    def test_projects_unauth(self):
        res = self.anon.get("/api/core/projects/")
        self.assertIn(res.status_code, [401, 403])

    def test_projects_auth(self):
        res = self.auth.get("/api/core/projects/")
        self.assertNotEqual(res.status_code, 500)

    def test_project_create(self):
        res = self.auth.post(
            "/api/core/projects/",
            {"name": "Smoke Test Project", "description": "Smoke test"},
            format="json",
        )
        # 200/201 = success, 400 = validation (missing required fields)
        self.assertNotEqual(res.status_code, 500)

    def test_graph_unauth(self):
        res = self.anon.get("/api/core/graph/")
        self.assertIn(res.status_code, [401, 403])

    def test_org_unauth(self):
        res = self.anon.get("/api/core/org/")
        self.assertIn(res.status_code, [401, 403])

    def test_evidence_from_analysis_unauth(self):
        res = self.anon.post("/api/core/evidence/from-analysis/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])


@override_settings(**SMOKE_SETTINGS)
class WorkbenchSmokeTest(TestCase):
    """Smoke tests for workbench endpoints."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("smoke-workbench@test.com")

    def setUp(self):
        self.auth = APIClient()
        self.auth.force_authenticate(self.user)
        self.anon = APIClient()

    def test_workbench_list_unauth(self):
        res = self.anon.get("/api/workbench/")
        # May return 302 (login redirect) instead of 401/403
        self.assertIn(res.status_code, [302, 401, 403])

    def test_workbench_list_auth(self):
        res = self.auth.get("/api/workbench/")
        self.assertNotEqual(res.status_code, 500)

    def test_workbench_projects_unauth(self):
        res = self.anon.get("/api/workbench/projects/")
        # May return 302 (login redirect) instead of 401/403
        self.assertIn(res.status_code, [302, 401, 403])

    def test_workbench_projects_auth(self):
        res = self.auth.get("/api/workbench/projects/")
        self.assertNotEqual(res.status_code, 500)


@override_settings(**SMOKE_SETTINGS)
class InternalAPISmokeTest(TestCase):
    """Smoke tests for internal/staff API endpoints."""

    @classmethod
    def setUpTestData(cls):
        cls.staff = _make_user("smoke-staff@test.com", staff=True)
        cls.user = _make_user("smoke-nostaff@test.com")

    def setUp(self):
        self.staff_client = APIClient()
        self.staff_client.force_authenticate(self.staff)
        self.user_client = APIClient()
        self.user_client.force_authenticate(self.user)
        self.anon = APIClient()

    # Auth enforcement: non-staff users should be rejected
    def test_overview_nostaff(self):
        res = self.user_client.get("/api/internal/overview/")
        self.assertIn(res.status_code, [401, 403])

    def test_overview_staff(self):
        res = self.staff_client.get("/api/internal/overview/")
        self.assertNotEqual(res.status_code, 500)

    def test_users_staff(self):
        res = self.staff_client.get("/api/internal/users/")
        self.assertNotEqual(res.status_code, 500)

    def test_dsw_analytics_staff(self):
        res = self.staff_client.get("/api/internal/dsw-analytics/")
        self.assertNotEqual(res.status_code, 500)

    def test_performance_staff(self):
        res = self.staff_client.get("/api/internal/performance/")
        self.assertNotEqual(res.status_code, 500)

    def test_business_staff(self):
        res = self.staff_client.get("/api/internal/business/")
        self.assertNotEqual(res.status_code, 500)

    def test_activity_staff(self):
        res = self.staff_client.get("/api/internal/activity/")
        self.assertNotEqual(res.status_code, 500)

    def test_changes_staff(self):
        res = self.staff_client.get("/api/internal/changes/")
        self.assertNotEqual(res.status_code, 500)

    def test_compliance_staff(self):
        res = self.staff_client.get("/api/internal/compliance/")
        self.assertNotEqual(res.status_code, 500)

    def test_incidents_staff(self):
        res = self.staff_client.get("/api/internal/incidents/")
        self.assertNotEqual(res.status_code, 500)

    def test_roadmap_staff(self):
        res = self.staff_client.get("/api/internal/roadmap/")
        self.assertNotEqual(res.status_code, 500)

    def test_features_staff(self):
        res = self.staff_client.get("/api/internal/features/")
        self.assertNotEqual(res.status_code, 500)

    def test_plans_staff(self):
        res = self.staff_client.get("/api/internal/plans/")
        self.assertNotEqual(res.status_code, 500)

    def test_experiments_staff(self):
        res = self.staff_client.get("/api/internal/experiments/")
        self.assertNotEqual(res.status_code, 500)

    def test_automation_rules_staff(self):
        res = self.staff_client.get("/api/internal/automation/rules/")
        self.assertNotEqual(res.status_code, 500)

    def test_autopilot_staff(self):
        res = self.staff_client.get("/api/internal/autopilot/")
        self.assertNotEqual(res.status_code, 500)

    def test_feedback_staff(self):
        res = self.staff_client.get("/api/internal/feedback/")
        self.assertNotEqual(res.status_code, 500)

    def test_calibration_staff(self):
        res = self.staff_client.get("/api/internal/calibration/")
        self.assertNotEqual(res.status_code, 500)

    def test_site_analytics_staff(self):
        res = self.staff_client.get("/api/internal/site-analytics/")
        self.assertNotEqual(res.status_code, 500)

    def test_infra_staff(self):
        res = self.staff_client.get("/api/internal/infra/")
        self.assertNotEqual(res.status_code, 500)

    def test_standards_staff(self):
        res = self.staff_client.get("/api/internal/standards/")
        self.assertNotEqual(res.status_code, 500)

    def test_onboarding_staff(self):
        res = self.staff_client.get("/api/internal/onboarding/")
        self.assertNotEqual(res.status_code, 500)

    # Blog management
    def test_blog_list_staff(self):
        res = self.staff_client.get("/api/internal/blog/")
        self.assertNotEqual(res.status_code, 500)

    def test_blog_list_nostaff(self):
        res = self.user_client.get("/api/internal/blog/")
        self.assertIn(res.status_code, [401, 403])

    # CRM
    def test_crm_leads_staff(self):
        res = self.staff_client.get("/api/internal/crm/leads/")
        self.assertNotEqual(res.status_code, 500)

    def test_crm_pipeline_staff(self):
        res = self.staff_client.get("/api/internal/crm/pipeline/")
        self.assertNotEqual(res.status_code, 500)

    def test_crm_sequences_staff(self):
        res = self.staff_client.get("/api/internal/crm/sequences/")
        self.assertNotEqual(res.status_code, 500)


@override_settings(**SMOKE_SETTINGS)
class FilesSmokeTest(TestCase):
    """Smoke tests for file management endpoints."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("smoke-files@test.com")

    def setUp(self):
        self.auth = APIClient()
        self.auth.force_authenticate(self.user)
        self.anon = APIClient()

    def test_files_list_unauth(self):
        res = self.anon.get("/api/files/")
        self.assertIn(res.status_code, [401, 403])

    def test_files_list_auth(self):
        res = self.auth.get("/api/files/")
        self.assertNotEqual(res.status_code, 500)

    def test_files_upload_unauth(self):
        res = self.anon.post("/api/files/upload/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_files_quota_unauth(self):
        res = self.anon.get("/api/files/quota/")
        self.assertIn(res.status_code, [401, 403])


@override_settings(**SMOKE_SETTINGS)
class ForgeSmokeTest(TestCase):
    """Smoke tests for synthetic data generation."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("smoke-forge@test.com")

    def setUp(self):
        self.auth = APIClient()
        self.auth.force_authenticate(self.user)
        self.anon = APIClient()

    def test_forge_generate_unauth(self):
        # Forge URLs have no trailing slash
        res = self.anon.post("/api/forge/generate", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_forge_schemas_unauth(self):
        res = self.anon.get("/api/forge/schemas")
        self.assertIn(res.status_code, [401, 403])

    def test_forge_schemas_auth(self):
        res = self.auth.get("/api/forge/schemas")
        self.assertNotEqual(res.status_code, 500)


@override_settings(**SMOKE_SETTINGS)
class AgentEndpointsSmokeTest(TestCase):
    """Smoke tests for specialized agent endpoints."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _make_user("smoke-agents@test.com")

    def setUp(self):
        self.auth = APIClient()
        self.auth.force_authenticate(self.user)
        self.anon = APIClient()

    # Experimenter / DOE
    def test_experimenter_power_unauth(self):
        res = self.anon.post("/api/experimenter/power/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    def test_experimenter_design_unauth(self):
        res = self.anon.post("/api/experimenter/design/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    # Synara belief engine (requires workbench_id in URL)
    def test_synara_hypotheses_unauth(self):
        res = self.anon.get(f"/api/synara/{DUMMY_UUID}/hypotheses/")
        self.assertIn(res.status_code, [401, 403])

    # Workflow routes removed in CR-0.6a

    # Plant simulation
    def test_plantsim_list_unauth(self):
        res = self.anon.get("/api/plantsim/")
        self.assertIn(res.status_code, [401, 403])

    def test_plantsim_list_auth(self):
        res = self.auth.get("/api/plantsim/")
        self.assertNotEqual(res.status_code, 500)

    # DSW Autopilot
    def test_dsw_autopilot_unauth(self):
        res = self.anon.post("/api/dsw/autopilot/clean-train/", {}, format="json")
        self.assertIn(res.status_code, [401, 403])

    # X-Matrix (under hoshin)
    def test_xmatrix_unauth(self):
        res = self.anon.get("/api/hoshin/x-matrix/")
        self.assertIn(res.status_code, [401, 403])

    # ISO docs
    def test_iso_docs_list_unauth(self):
        res = self.anon.get("/api/iso-docs/")
        self.assertIn(res.status_code, [401, 403])


@override_settings(**SMOKE_SETTINGS)
class EmailTrackingSmokeTest(TestCase):
    """Smoke tests for email tracking (public, no auth)."""

    def test_email_open_tracking(self):
        """Email open pixel should respond (200/301/404)."""
        res = self.client.get(f"/api/email/open/{DUMMY_UUID}/")
        self.assertNotEqual(res.status_code, 500)

    def test_email_click_tracking(self):
        res = self.client.get(f"/api/email/click/{DUMMY_UUID}/")
        self.assertNotEqual(res.status_code, 500)

    def test_email_unsubscribe(self):
        res = self.client.post("/api/email/unsubscribe/", {}, content_type="application/json")
        self.assertNotEqual(res.status_code, 500)


@override_settings(**SMOKE_SETTINGS)
class PublicBeaconSmokeTest(TestCase):
    """Smoke tests for public beacon endpoints (no auth)."""

    def test_site_duration(self):
        res = self.client.post(
            "/api/site-duration/",
            {"duration": 5000, "path": "/"},
            content_type="application/json",
        )
        self.assertNotEqual(res.status_code, 500)

    def test_funnel_event(self):
        res = self.client.post(
            "/api/funnel-event/",
            {"event": "test", "page": "/"},
            content_type="application/json",
        )
        self.assertNotEqual(res.status_code, 500)

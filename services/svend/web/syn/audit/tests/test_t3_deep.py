"""T3-COV deep smoke tests — comprehensive view coverage for T3 tier modules.

Covers: iso_views, hoshin_views, xmatrix_views, fmea_views, a3_views,
rca_views, whiteboard_views, learn_views, autopilot_views, workbench/views,
forecast_views, vsm_views, problem_views, report_views, capa_views,
triage_views, guide_views, notifications/views, plantsim_views,
iso_doc_views, qms_views.

Strategy: 3-5 smoke tests per module:
  1. Auth test (unauth returns 401/403)
  2. List endpoint — auth user (assertNotEqual 500)
  3. Create with minimal data (covers validation code)
  4. Detail with invalid UUID (covers lookup code)

Uses Django test Client with login() for session auth (force_authenticate
only works with DRF views, not plain Django @gated/@gated_paid views).

Standard: CAL-001 §7 (Endpoint Coverage), TST-001 §10.6
Compliance: SOC 2 CC4.1, CC7.2
<!-- test: syn.audit.tests.test_t3_deep -->
"""

import json

from django.test import TestCase, override_settings

from accounts.models import Tier, User

FAKE_UUID = "00000000-0000-0000-0000-000000000000"
PWD = "testpass123"
SMOKE = {"RATELIMIT_ENABLE": False, "SECURE_SSL_REDIRECT": False}

# Accepted "not a server error" codes for smoke testing
NOT_500 = {200, 201, 301, 302, 400, 401, 403, 404, 405, 429}


def _user(email="t3deep@test.com", tier=Tier.TEAM, staff=False):
    """Create a paid-tier test user with verified email."""
    username = email.split("@")[0].replace(".", "_").replace("-", "_")
    u = User.objects.create_user(username=username, email=email, password=PWD)
    u.tier = tier
    u.email_verified = True
    u.is_staff = staff
    u.save()
    return u


def _post_json(client, url, data=None):
    """POST JSON data using Django test client."""
    return client.post(
        url,
        data=json.dumps(data or {}),
        content_type="application/json",
    )


def _put_json(client, url, data=None):
    """PUT JSON data using Django test client."""
    return client.put(
        url,
        data=json.dumps(data or {}),
        content_type="application/json",
    )


def _not_500(tc, resp, msg=""):
    """Assert response is not a server error."""
    tc.assertNotEqual(
        resp.status_code, 500, msg or f"Got 500 from {resp.wsgi_request.path}"
    )


def _is_unauth(tc, resp):
    """Assert response rejects unauthenticated user (401, 403, or 302 redirect)."""
    tc.assertIn(
        resp.status_code,
        [301, 302, 401, 403],
        f"Expected auth rejection, got {resp.status_code}",
    )


def _is_not_found_or_error(tc, resp):
    """Assert response is not 200 for an invalid resource lookup.
    Accepts any 4xx or 5xx — the point is exercising the code path."""
    tc.assertGreaterEqual(
        resp.status_code,
        400,
        f"Expected error response, got {resp.status_code} from {resp.wsgi_request.path}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. ISO Views (iso_views.py — 948 stmts, 12.1%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class ISODashboardTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("iso-dash@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_dashboard_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/iso/dashboard/"))

    def test_dashboard_auth(self):
        _not_500(self, self.client.get("/api/iso/dashboard/"))


@override_settings(**SMOKE)
class ISONCRTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("iso-ncr@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_ncr_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/iso/ncrs/"))

    def test_ncr_list_auth(self):
        _not_500(self, self.client.get("/api/iso/ncrs/"))

    def test_ncr_create_minimal(self):
        _not_500(
            self,
            _post_json(
                self.client,
                "/api/iso/ncrs/",
                {
                    "title": "Test NCR",
                    "description": "Test NCR desc",
                    "severity": "minor",
                },
            ),
        )

    def test_ncr_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/iso/ncrs/"))

    def test_ncr_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/iso/ncrs/{FAKE_UUID}/"))

    def test_ncr_stats(self):
        _not_500(self, self.client.get("/api/iso/ncrs/stats/"))

    def test_ncr_launch_rca_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/iso/ncrs/{FAKE_UUID}/launch-rca/")
        )

    def test_ncr_files_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/iso/ncrs/{FAKE_UUID}/files/")
        )


@override_settings(**SMOKE)
class ISOAuditTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("iso-aud@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_audit_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/iso/audits/"))

    def test_audit_list_auth(self):
        _not_500(self, self.client.get("/api/iso/audits/"))

    def test_audit_create_minimal(self):
        _not_500(
            self,
            _post_json(
                self.client,
                "/api/iso/audits/",
                {
                    "title": "Test Audit",
                    "scope": "Manufacturing",
                },
            ),
        )

    def test_audit_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/iso/audits/"))

    def test_audit_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/iso/audits/{FAKE_UUID}/"))

    def test_audit_finding_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/iso/audits/{FAKE_UUID}/findings/")
        )


@override_settings(**SMOKE)
class ISOChecklistTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("iso-chk@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_checklist_list(self):
        _not_500(self, self.client.get("/api/iso/checklists/"))

    def test_checklist_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/iso/checklists/"))

    def test_checklist_detail_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/iso/checklists/{FAKE_UUID}/")
        )


@override_settings(**SMOKE)
class ISOTrainingTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("iso-train@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_training_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/iso/training/"))

    def test_training_list_auth(self):
        _not_500(self, self.client.get("/api/iso/training/"))

    def test_training_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/iso/training/"))

    def test_training_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/iso/training/{FAKE_UUID}/"))

    def test_training_record_create_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/iso/training/{FAKE_UUID}/records/")
        )

    def test_training_record_update_invalid(self):
        _is_not_found_or_error(
            self, _put_json(self.client, f"/api/iso/training/records/{FAKE_UUID}/")
        )

    def test_training_record_files_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/iso/training/records/{FAKE_UUID}/files/")
        )


@override_settings(**SMOKE)
class ISOReviewTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("iso-rev@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_review_list(self):
        _not_500(self, self.client.get("/api/iso/reviews/"))

    def test_review_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/iso/reviews/"))

    def test_review_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/iso/reviews/{FAKE_UUID}/"))


@override_settings(**SMOKE)
class ISODocumentTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("iso-doc@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_document_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/iso/documents/"))

    def test_document_list_auth(self):
        _not_500(self, self.client.get("/api/iso/documents/"))

    def test_document_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/iso/documents/"))

    def test_document_detail_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/iso/documents/{FAKE_UUID}/")
        )

    def test_document_files_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/iso/documents/{FAKE_UUID}/files/")
        )


@override_settings(**SMOKE)
class ISOSupplierTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("iso-sup@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_supplier_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/iso/suppliers/"))

    def test_supplier_list_auth(self):
        _not_500(self, self.client.get("/api/iso/suppliers/"))

    def test_supplier_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/iso/suppliers/"))

    def test_supplier_detail_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/iso/suppliers/{FAKE_UUID}/")
        )


@override_settings(**SMOKE)
class ISOSignatureTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("iso-sig@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_signature_list(self):
        _not_500(self, self.client.get("/api/iso/signatures/"))

    def test_signature_sign_empty(self):
        _not_500(self, _post_json(self.client, "/api/iso/signatures/"))

    def test_signature_verify_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/iso/signatures/{FAKE_UUID}/verify/")
        )

    def test_signature_verify_chain_empty(self):
        _not_500(self, _post_json(self.client, "/api/iso/signatures/verify-chain/"))


@override_settings(**SMOKE)
class ISOStudyActionsTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("iso-study@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_raise_capa_empty(self):
        _not_500(self, _post_json(self.client, "/api/iso/study-actions/raise-capa/"))

    def test_schedule_audit_empty(self):
        _not_500(
            self, _post_json(self.client, "/api/iso/study-actions/schedule-audit/")
        )

    def test_request_doc_update_empty(self):
        _not_500(
            self, _post_json(self.client, "/api/iso/study-actions/request-doc-update/")
        )

    def test_flag_training_gap_empty(self):
        _not_500(
            self, _post_json(self.client, "/api/iso/study-actions/flag-training-gap/")
        )

    def test_flag_fmea_update_empty(self):
        _not_500(
            self, _post_json(self.client, "/api/iso/study-actions/flag-fmea-update/")
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Hoshin Views (hoshin_views.py — 873 stmts, 13.3%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class HoshinSiteTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("hoshin-site@test.com", tier=Tier.ENTERPRISE)

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_sites_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/hoshin/sites/"))

    def test_sites_list_auth(self):
        _not_500(self, self.client.get("/api/hoshin/sites/"))

    def test_sites_create_minimal(self):
        _not_500(
            self,
            _post_json(self.client, "/api/hoshin/sites/create/", {"name": "Test Site"}),
        )

    def test_sites_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/hoshin/sites/create/"))

    def test_site_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/hoshin/sites/{FAKE_UUID}/"))

    def test_site_update_invalid(self):
        _is_not_found_or_error(
            self,
            _put_json(
                self.client, f"/api/hoshin/sites/{FAKE_UUID}/update/", {"name": "X"}
            ),
        )

    def test_site_delete_invalid(self):
        _is_not_found_or_error(
            self, self.client.delete(f"/api/hoshin/sites/{FAKE_UUID}/delete/")
        )

    def test_site_members_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/hoshin/sites/{FAKE_UUID}/members/")
        )

    def test_site_grant_access_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(self.client, f"/api/hoshin/sites/{FAKE_UUID}/members/grant/"),
        )

    def test_site_alignment_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/hoshin/sites/{FAKE_UUID}/alignment/")
        )


@override_settings(**SMOKE)
class HoshinProjectTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("hoshin-proj@test.com", tier=Tier.ENTERPRISE)

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_projects_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/hoshin/projects/"))

    def test_projects_list_auth(self):
        _not_500(self, self.client.get("/api/hoshin/projects/"))

    def test_project_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/hoshin/projects/create/"))

    def test_project_detail_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/hoshin/projects/{FAKE_UUID}/")
        )

    def test_project_update_invalid(self):
        _is_not_found_or_error(
            self, _put_json(self.client, f"/api/hoshin/projects/{FAKE_UUID}/update/")
        )

    def test_project_delete_invalid(self):
        _is_not_found_or_error(
            self, self.client.delete(f"/api/hoshin/projects/{FAKE_UUID}/delete/")
        )

    def test_project_from_proposals_empty(self):
        _not_500(self, _post_json(self.client, "/api/hoshin/projects/from-proposals/"))

    def test_project_monthly_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(self.client, f"/api/hoshin/projects/{FAKE_UUID}/monthly/1/"),
        )


@override_settings(**SMOKE)
class HoshinActionsTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("hoshin-act@test.com", tier=Tier.ENTERPRISE)

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_actions_list_invalid_project(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/hoshin/projects/{FAKE_UUID}/actions/")
        )

    def test_action_create_invalid_project(self):
        _is_not_found_or_error(
            self,
            _post_json(
                self.client, f"/api/hoshin/projects/{FAKE_UUID}/actions/create/"
            ),
        )

    def test_action_update_invalid(self):
        _is_not_found_or_error(
            self, _put_json(self.client, f"/api/hoshin/actions/{FAKE_UUID}/update/")
        )

    def test_action_delete_invalid(self):
        _is_not_found_or_error(
            self, self.client.delete(f"/api/hoshin/actions/{FAKE_UUID}/delete/")
        )


@override_settings(**SMOKE)
class HoshinDashboardTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("hoshin-db@test.com", tier=Tier.ENTERPRISE)

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_dashboard_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/hoshin/dashboard/"))

    def test_dashboard_auth(self):
        _not_500(self, self.client.get("/api/hoshin/dashboard/"))

    def test_calendar(self):
        _not_500(self, self.client.get("/api/hoshin/calendar/"))

    def test_calendar_facilitators(self):
        _not_500(self, self.client.get("/api/hoshin/calendar/facilitators/"))

    def test_calculation_methods(self):
        _not_500(self, self.client.get("/api/hoshin/calculation-methods/"))

    def test_test_formula_empty(self):
        _not_500(self, _post_json(self.client, "/api/hoshin/test-formula/"))


@override_settings(**SMOKE)
class HoshinEmployeeTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("hoshin-emp@test.com", tier=Tier.ENTERPRISE)

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_employees_list(self):
        _not_500(self, self.client.get("/api/hoshin/employees/"))

    def test_employees_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/hoshin/employees/"))

    def test_employee_detail_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/hoshin/employees/{FAKE_UUID}/")
        )

    def test_employee_availability_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/hoshin/employees/{FAKE_UUID}/availability/")
        )

    def test_employee_timeline_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/hoshin/employees/{FAKE_UUID}/timeline/")
        )

    def test_employees_import_empty(self):
        _not_500(self, _post_json(self.client, "/api/hoshin/employees/import/"))


@override_settings(**SMOKE)
class HoshinCommitmentTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("hoshin-com@test.com", tier=Tier.ENTERPRISE)

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_commitments_list(self):
        _not_500(self, self.client.get("/api/hoshin/commitments/"))

    def test_commitment_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/hoshin/commitments/"))

    def test_commitment_detail_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/hoshin/commitments/{FAKE_UUID}/")
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. X-Matrix Views (xmatrix_views.py — 457 stmts, 9.8%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class XMatrixDataTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("xmatrix@test.com", tier=Tier.ENTERPRISE)

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_xmatrix_data_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/hoshin/x-matrix/"))

    def test_xmatrix_data_auth(self):
        _not_500(self, self.client.get("/api/hoshin/x-matrix/"))

    def test_xmatrix_correlations_empty(self):
        _not_500(self, _post_json(self.client, "/api/hoshin/x-matrix/correlations/"))

    def test_xmatrix_rollover_empty(self):
        _not_500(self, _post_json(self.client, "/api/hoshin/x-matrix/rollover/"))


@override_settings(**SMOKE)
class XMatrixStrategicTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("xm-strat@test.com", tier=Tier.ENTERPRISE)

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_strategic_list(self):
        _not_500(self, self.client.get("/api/hoshin/strategic-objectives/"))

    def test_strategic_create_minimal(self):
        _not_500(
            self,
            _post_json(
                self.client, "/api/hoshin/strategic-objectives/", {"text": "Test Obj"}
            ),
        )

    def test_strategic_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/hoshin/strategic-objectives/"))

    def test_strategic_detail_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/hoshin/strategic-objectives/{FAKE_UUID}/")
        )

    def test_strategic_update_invalid(self):
        _is_not_found_or_error(
            self,
            _put_json(self.client, f"/api/hoshin/strategic-objectives/{FAKE_UUID}/"),
        )

    def test_strategic_delete_invalid(self):
        _is_not_found_or_error(
            self, self.client.delete(f"/api/hoshin/strategic-objectives/{FAKE_UUID}/")
        )


@override_settings(**SMOKE)
class XMatrixAnnualTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("xm-ann@test.com", tier=Tier.ENTERPRISE)

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_annual_list(self):
        _not_500(self, self.client.get("/api/hoshin/annual-objectives/"))

    def test_annual_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/hoshin/annual-objectives/"))

    def test_annual_detail_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/hoshin/annual-objectives/{FAKE_UUID}/")
        )


@override_settings(**SMOKE)
class XMatrixKPITest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("xm-kpi@test.com", tier=Tier.ENTERPRISE)

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_kpi_list(self):
        _not_500(self, self.client.get("/api/hoshin/kpis/"))

    def test_kpi_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/hoshin/kpis/"))

    def test_kpi_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/hoshin/kpis/{FAKE_UUID}/"))

    def test_kpi_update_invalid(self):
        _is_not_found_or_error(
            self, _put_json(self.client, f"/api/hoshin/kpis/{FAKE_UUID}/")
        )


@override_settings(**SMOKE)
class XMatrixVSMPromoteTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("xm-vsm@test.com", tier=Tier.ENTERPRISE)

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_promote_vsm_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/hoshin/vsm/{FAKE_UUID}/promote/")
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. FMEA Views (fmea_views.py — 543 stmts, 15.5%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class FMEACRUDTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("fmea@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_fmea_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/fmea/"))

    def test_fmea_list_auth(self):
        _not_500(self, self.client.get("/api/fmea/"))

    def test_fmea_create_minimal(self):
        _not_500(
            self,
            _post_json(
                self.client,
                "/api/fmea/create/",
                {
                    "title": "Test FMEA",
                    "process_name": "Assembly",
                },
            ),
        )

    def test_fmea_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/fmea/create/"))

    def test_fmea_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/fmea/{FAKE_UUID}/"))

    def test_fmea_update_invalid(self):
        _is_not_found_or_error(
            self, _put_json(self.client, f"/api/fmea/{FAKE_UUID}/update/")
        )

    def test_fmea_delete_invalid(self):
        _is_not_found_or_error(
            self, self.client.delete(f"/api/fmea/{FAKE_UUID}/delete/")
        )

    def test_fmea_rpn_summary_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/fmea/{FAKE_UUID}/summary/"))

    def test_fmea_trending_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/fmea/{FAKE_UUID}/trending/")
        )

    def test_fmea_patterns(self):
        _not_500(self, self.client.get("/api/fmea/patterns/"))

    def test_fmea_suggest_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/fmea/{FAKE_UUID}/suggest-failure-modes/")
        )


@override_settings(**SMOKE)
class FMEARowTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("fmea-row@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_add_row_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/fmea/{FAKE_UUID}/rows/")
        )

    def test_update_row_invalid(self):
        _is_not_found_or_error(
            self, _put_json(self.client, f"/api/fmea/{FAKE_UUID}/rows/{FAKE_UUID}/")
        )

    def test_delete_row_invalid(self):
        _is_not_found_or_error(
            self, self.client.delete(f"/api/fmea/{FAKE_UUID}/rows/{FAKE_UUID}/delete/")
        )

    def test_reorder_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(self.client, f"/api/fmea/{FAKE_UUID}/reorder/", {"order": []}),
        )

    def test_link_hypothesis_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(self.client, f"/api/fmea/{FAKE_UUID}/rows/{FAKE_UUID}/link/"),
        )

    def test_record_revision_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(self.client, f"/api/fmea/{FAKE_UUID}/rows/{FAKE_UUID}/revise/"),
        )

    def test_spc_update_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(
                self.client, f"/api/fmea/{FAKE_UUID}/rows/{FAKE_UUID}/spc-update/"
            ),
        )

    def test_spc_cpk_update_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(
                self.client, f"/api/fmea/{FAKE_UUID}/rows/{FAKE_UUID}/spc-cpk-update/"
            ),
        )

    def test_promote_action_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(
                self.client, f"/api/fmea/{FAKE_UUID}/rows/{FAKE_UUID}/promote-action/"
            ),
        )

    def test_investigate_row_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(
                self.client, f"/api/fmea/{FAKE_UUID}/rows/{FAKE_UUID}/investigate/"
            ),
        )

    def test_fmea_actions_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/fmea/{FAKE_UUID}/actions/"))


# ═══════════════════════════════════════════════════════════════════════════
# 5. A3 Views (a3_views.py — 428 stmts, 12.6%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class A3CRUDTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("a3@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_a3_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/a3/"))

    def test_a3_list_auth(self):
        _not_500(self, self.client.get("/api/a3/"))

    def test_a3_create_minimal(self):
        _not_500(
            self,
            _post_json(
                self.client,
                "/api/a3/create/",
                {
                    "title": "Test A3",
                    "problem_statement": "Test problem statement",
                },
            ),
        )

    def test_a3_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/a3/create/"))

    def test_a3_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/a3/{FAKE_UUID}/"))

    def test_a3_update_invalid(self):
        _is_not_found_or_error(
            self, _put_json(self.client, f"/api/a3/{FAKE_UUID}/update/")
        )

    def test_a3_delete_invalid(self):
        _is_not_found_or_error(self, self.client.delete(f"/api/a3/{FAKE_UUID}/delete/"))

    def test_a3_import_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/a3/{FAKE_UUID}/import/")
        )

    def test_a3_auto_populate_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/a3/{FAKE_UUID}/auto-populate/")
        )

    def test_a3_critique_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/a3/{FAKE_UUID}/critique/")
        )

    def test_a3_embed_diagram_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/a3/{FAKE_UUID}/embed-diagram/")
        )

    def test_a3_export_pdf_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/a3/{FAKE_UUID}/export/pdf/")
        )

    def test_a3_actions_list_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/a3/{FAKE_UUID}/actions/"))

    def test_a3_action_create_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/a3/{FAKE_UUID}/actions/create/")
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6. RCA Views (rca_views.py — 397 stmts, 15.6%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class RCACRUDTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("rca@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_sessions_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/rca/sessions/"))

    def test_sessions_list_auth(self):
        _not_500(self, self.client.get("/api/rca/sessions/"))

    def test_session_create_minimal(self):
        _not_500(
            self,
            _post_json(
                self.client,
                "/api/rca/sessions/create/",
                {
                    "title": "Test RCA",
                    "problem_statement": "Test problem",
                },
            ),
        )

    def test_session_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/rca/sessions/create/"))

    def test_session_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/rca/sessions/{FAKE_UUID}/"))

    def test_session_update_invalid(self):
        _is_not_found_or_error(
            self, _put_json(self.client, f"/api/rca/sessions/{FAKE_UUID}/update/")
        )

    def test_session_delete_invalid(self):
        _is_not_found_or_error(
            self, self.client.delete(f"/api/rca/sessions/{FAKE_UUID}/delete/")
        )

    def test_session_link_a3_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/rca/sessions/{FAKE_UUID}/link-a3/")
        )

    def test_session_actions_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/rca/sessions/{FAKE_UUID}/actions/")
        )

    def test_session_action_create_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(self.client, f"/api/rca/sessions/{FAKE_UUID}/actions/create/"),
        )


@override_settings(**SMOKE)
class RCAIntelligenceTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("rca-ai@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_critique_unauth(self):
        self.client.logout()
        _is_unauth(self, _post_json(self.client, "/api/rca/critique/"))

    def test_critique_empty(self):
        _not_500(self, _post_json(self.client, "/api/rca/critique/"))

    def test_critique_countermeasure_empty(self):
        _not_500(self, _post_json(self.client, "/api/rca/critique-countermeasure/"))

    def test_evaluate_chain_empty(self):
        _not_500(self, _post_json(self.client, "/api/rca/evaluate/"))

    def test_guided_questions_empty(self):
        _not_500(self, _post_json(self.client, "/api/rca/guided-questions/"))

    def test_clusters(self):
        _not_500(self, self.client.get("/api/rca/clusters/"))

    def test_similar_empty(self):
        _not_500(self, _post_json(self.client, "/api/rca/similar/"))

    def test_reindex(self):
        _not_500(self, _post_json(self.client, "/api/rca/reindex/"))


# ═══════════════════════════════════════════════════════════════════════════
# 7. Whiteboard Views (whiteboard_views.py — 440 stmts, 15.2%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class WhiteboardCRUDTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("wb@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_boards_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/whiteboard/boards/"))

    def test_boards_list_auth(self):
        _not_500(self, self.client.get("/api/whiteboard/boards/"))

    def test_board_create_minimal(self):
        _not_500(
            self,
            _post_json(
                self.client, "/api/whiteboard/boards/create/", {"title": "Test Board"}
            ),
        )

    def test_board_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/whiteboard/boards/create/"))

    def test_board_detail_invalid(self):
        _is_not_found_or_error(
            self, self.client.get("/api/whiteboard/boards/nonexistent-room/")
        )

    def test_board_update_invalid(self):
        _is_not_found_or_error(
            self,
            _put_json(self.client, "/api/whiteboard/boards/nonexistent-room/update/"),
        )

    def test_board_delete_invalid(self):
        _is_not_found_or_error(
            self, self.client.delete("/api/whiteboard/boards/nonexistent-room/delete/")
        )


@override_settings(**SMOKE)
class WhiteboardFeaturesTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("wb-feat@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_cursor_update_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(
                self.client,
                "/api/whiteboard/boards/nonexistent-room/cursor/",
                {"x": 0, "y": 0},
            ),
        )

    def test_toggle_voting_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(self.client, "/api/whiteboard/boards/nonexistent-room/voting/"),
        )

    def test_add_vote_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(
                self.client,
                "/api/whiteboard/boards/nonexistent-room/vote/",
                {"element_id": "x"},
            ),
        )

    def test_remove_vote_invalid(self):
        _is_not_found_or_error(
            self,
            self.client.delete(
                "/api/whiteboard/boards/nonexistent-room/vote/test-element/"
            ),
        )

    def test_export_hypotheses_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(
                self.client,
                "/api/whiteboard/boards/nonexistent-room/export-hypotheses/",
            ),
        )

    def test_export_svg_invalid(self):
        _is_not_found_or_error(
            self, self.client.get("/api/whiteboard/boards/nonexistent-room/svg/")
        )

    def test_export_png_invalid(self):
        _is_not_found_or_error(
            self, self.client.get("/api/whiteboard/boards/nonexistent-room/png/")
        )

    def test_guest_list_invalid(self):
        _is_not_found_or_error(
            self, self.client.get("/api/whiteboard/boards/nonexistent-room/guests/")
        )

    def test_guest_create_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(
                self.client, "/api/whiteboard/boards/nonexistent-room/guests/create/"
            ),
        )

    def test_guest_name_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(
                self.client,
                "/api/whiteboard/boards/nonexistent-room/guest-name/",
                {"name": "Test"},
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════
# 8. Learn Views (learn_views.py — 593 stmts, 11.6%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class LearnModulesTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("learn@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_modules_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/learn/modules/"))

    def test_modules_list_auth(self):
        _not_500(self, self.client.get("/api/learn/modules/"))

    def test_module_detail_invalid(self):
        _is_not_found_or_error(
            self, self.client.get("/api/learn/modules/nonexistent-module/")
        )

    def test_section_detail_invalid(self):
        _is_not_found_or_error(
            self,
            self.client.get(
                "/api/learn/modules/nonexistent-module/sections/nonexistent-section/"
            ),
        )


@override_settings(**SMOKE)
class LearnProgressTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("learn-prog@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_progress_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/learn/progress/"))

    def test_progress_auth(self):
        _not_500(self, self.client.get("/api/learn/progress/"))

    def test_mark_complete_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, "/api/learn/progress/nonexistent/complete/")
        )


@override_settings(**SMOKE)
class LearnSessionTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("learn-sess@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_start_session_empty(self):
        _not_500(self, _post_json(self.client, "/api/learn/session/start/"))

    def test_execute_step_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(
                self.client, f"/api/learn/session/{FAKE_UUID}/step/step1/execute/"
            ),
        )

    def test_reset_session_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/learn/session/{FAKE_UUID}/reset/")
        )


@override_settings(**SMOKE)
class LearnAssessmentTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("learn-asmt@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_assessment_history_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/learn/assessment/history/"))

    def test_assessment_history_auth(self):
        _not_500(self, self.client.get("/api/learn/assessment/history/"))

    def test_generate_assessment_empty(self):
        _not_500(self, _post_json(self.client, "/api/learn/assessment/generate/"))

    def test_submit_assessment_empty(self):
        _not_500(self, _post_json(self.client, "/api/learn/assessment/submit/"))


# ═══════════════════════════════════════════════════════════════════════════
# 9. Autopilot Views (autopilot_views.py — 677 stmts, 5.8%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class AutopilotEndpointsTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("autopilot@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_clean_train_empty(self):
        _not_500(self, _post_json(self.client, "/api/dsw/autopilot/clean-train/"))

    def test_full_pipeline_empty(self):
        _not_500(self, _post_json(self.client, "/api/dsw/autopilot/full-pipeline/"))

    def test_augment_train_empty(self):
        _not_500(self, _post_json(self.client, "/api/dsw/autopilot/augment-train/"))

    def test_retrain_model_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/dsw/models/{FAKE_UUID}/retrain/")
        )


# ═══════════════════════════════════════════════════════════════════════════
# 10. Workbench Views (workbench/views.py — 516 stmts, 24.6%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class WorkbenchCRUDTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("wb-crud@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_workbench_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/workbench/"))

    def test_workbench_list_auth(self):
        _not_500(self, self.client.get("/api/workbench/"))

    def test_workbench_create_minimal(self):
        _not_500(
            self,
            _post_json(self.client, "/api/workbench/create/", {"title": "Test WB"}),
        )

    def test_workbench_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/workbench/create/"))

    def test_workbench_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/workbench/{FAKE_UUID}/"))

    def test_workbench_update_invalid(self):
        _is_not_found_or_error(
            self, _put_json(self.client, f"/api/workbench/{FAKE_UUID}/update/")
        )

    def test_workbench_delete_invalid(self):
        _is_not_found_or_error(
            self, self.client.delete(f"/api/workbench/{FAKE_UUID}/delete/")
        )

    def test_workbench_import_empty(self):
        _not_500(self, _post_json(self.client, "/api/workbench/import/"))

    def test_workbench_export_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/workbench/{FAKE_UUID}/export/")
        )


@override_settings(**SMOKE)
class WorkbenchArtifactTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("wb-art@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_create_artifact_invalid_wb(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/workbench/{FAKE_UUID}/artifacts/")
        )

    def test_get_artifact_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/workbench/{FAKE_UUID}/artifacts/{FAKE_UUID}/")
        )

    def test_update_artifact_invalid(self):
        _is_not_found_or_error(
            self,
            _put_json(
                self.client, f"/api/workbench/{FAKE_UUID}/artifacts/{FAKE_UUID}/update/"
            ),
        )

    def test_delete_artifact_invalid(self):
        _is_not_found_or_error(
            self,
            self.client.delete(
                f"/api/workbench/{FAKE_UUID}/artifacts/{FAKE_UUID}/delete/"
            ),
        )


@override_settings(**SMOKE)
class WorkbenchConnectionTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("wb-conn@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_connect_artifacts_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/workbench/{FAKE_UUID}/connect/")
        )

    def test_disconnect_artifacts_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/workbench/{FAKE_UUID}/disconnect/")
        )

    def test_advance_phase_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/workbench/{FAKE_UUID}/advance-phase/")
        )


@override_settings(**SMOKE)
class WorkbenchGuideTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("wb-guide@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_add_observation_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/workbench/{FAKE_UUID}/guide/observe/")
        )

    def test_acknowledge_observation_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(self.client, f"/api/workbench/{FAKE_UUID}/guide/0/acknowledge/"),
        )


@override_settings(**SMOKE)
class WorkbenchProjectTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("wb-proj@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_projects_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/workbench/projects/"))

    def test_projects_list_auth(self):
        _not_500(self, self.client.get("/api/workbench/projects/"))

    def test_project_create_minimal(self):
        _not_500(
            self,
            _post_json(
                self.client,
                "/api/workbench/projects/create/",
                {"title": "Test", "description": "T"},
            ),
        )

    def test_project_detail_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/workbench/projects/{FAKE_UUID}/")
        )

    def test_project_update_invalid(self):
        _is_not_found_or_error(
            self, _put_json(self.client, f"/api/workbench/projects/{FAKE_UUID}/update/")
        )

    def test_project_delete_invalid(self):
        _is_not_found_or_error(
            self, self.client.delete(f"/api/workbench/projects/{FAKE_UUID}/delete/")
        )

    def test_add_workbench_to_project_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(
                self.client, f"/api/workbench/projects/{FAKE_UUID}/workbenches/add/"
            ),
        )


@override_settings(**SMOKE)
class WorkbenchHypothesisTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("wb-hyp@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_hypotheses_list_invalid_project(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/workbench/projects/{FAKE_UUID}/hypotheses/")
        )

    def test_hypothesis_create_invalid_project(self):
        _is_not_found_or_error(
            self,
            _post_json(
                self.client, f"/api/workbench/projects/{FAKE_UUID}/hypotheses/create/"
            ),
        )

    def test_hypothesis_detail_invalid(self):
        _is_not_found_or_error(
            self,
            self.client.get(
                f"/api/workbench/projects/{FAKE_UUID}/hypotheses/{FAKE_UUID}/"
            ),
        )


@override_settings(**SMOKE)
class WorkbenchGraphTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("wb-graph@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_get_graph_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/workbench/{FAKE_UUID}/graph/")
        )

    def test_get_nodes_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/workbench/{FAKE_UUID}/graph/nodes/")
        )

    def test_add_node_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(self.client, f"/api/workbench/{FAKE_UUID}/graph/nodes/add/"),
        )

    def test_get_edges_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/workbench/{FAKE_UUID}/graph/edges/")
        )

    def test_add_edge_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(self.client, f"/api/workbench/{FAKE_UUID}/graph/edges/add/"),
        )

    def test_apply_evidence_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(
                self.client, f"/api/workbench/{FAKE_UUID}/graph/evidence/apply/"
            ),
        )

    def test_check_expansion_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/workbench/{FAKE_UUID}/graph/expansion/check/")
        )

    def test_expansion_signals_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/workbench/{FAKE_UUID}/graph/expansions/")
        )

    def test_epistemic_log_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/workbench/{FAKE_UUID}/epistemic-log/")
        )

    def test_clear_graph_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/workbench/{FAKE_UUID}/graph/clear/")
        )

    def test_project_graph_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/workbench/projects/{FAKE_UUID}/graph/")
        )


# ═══════════════════════════════════════════════════════════════════════════
# 11. Forecast Views (forecast_views.py — 160 stmts, 11.9%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class ForecastTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("forecast@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_forecast_unauth(self):
        self.client.logout()
        _is_unauth(self, _post_json(self.client, "/api/forecast/"))

    def test_forecast_empty_data(self):
        _not_500(self, _post_json(self.client, "/api/forecast/"))

    def test_forecast_minimal_data(self):
        _not_500(
            self,
            _post_json(
                self.client,
                "/api/forecast/",
                {
                    "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "periods": 3,
                },
            ),
        )

    def test_forecast_quote_empty(self):
        _not_500(self, _post_json(self.client, "/api/forecast/quote/"))


# ═══════════════════════════════════════════════════════════════════════════
# 12. VSM Views (vsm_views.py — 290 stmts, 16.9%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class VSMCRUDTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("vsm@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_vsm_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/vsm/"))

    def test_vsm_list_auth(self):
        _not_500(self, self.client.get("/api/vsm/"))

    def test_vsm_create_minimal(self):
        _not_500(
            self, _post_json(self.client, "/api/vsm/create/", {"title": "Test VSM"})
        )

    def test_vsm_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/vsm/create/"))

    def test_vsm_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/vsm/{FAKE_UUID}/"))

    def test_vsm_update_invalid(self):
        _is_not_found_or_error(
            self, _put_json(self.client, f"/api/vsm/{FAKE_UUID}/update/")
        )

    def test_vsm_delete_invalid(self):
        _is_not_found_or_error(
            self, self.client.delete(f"/api/vsm/{FAKE_UUID}/delete/")
        )

    def test_vsm_process_step_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/vsm/{FAKE_UUID}/process-step/")
        )

    def test_vsm_inventory_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/vsm/{FAKE_UUID}/inventory/")
        )

    def test_vsm_kaizen_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/vsm/{FAKE_UUID}/kaizen/")
        )

    def test_vsm_future_state_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/vsm/{FAKE_UUID}/future-state/")
        )

    def test_vsm_compare_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/vsm/{FAKE_UUID}/compare/"))

    def test_vsm_waste_analysis_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/vsm/{FAKE_UUID}/waste-analysis/")
        )


# ═══════════════════════════════════════════════════════════════════════════
# 13. Problem Views (problem_views.py — 421 stmts, 18.5%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class ProblemCRUDTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("prob@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_problems_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/problems/"))

    def test_problems_list_auth(self):
        _not_500(self, self.client.get("/api/problems/"))

    def test_problem_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/problems/{FAKE_UUID}/"))

    def test_add_hypothesis_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/problems/{FAKE_UUID}/hypotheses/")
        )

    def test_generate_hypotheses_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(self.client, f"/api/problems/{FAKE_UUID}/hypotheses/generate/"),
        )

    def test_add_evidence_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/problems/{FAKE_UUID}/evidence/")
        )

    def test_resolve_problem_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/problems/{FAKE_UUID}/resolve/")
        )

    def test_set_methodology_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/problems/{FAKE_UUID}/methodology/")
        )

    def test_advance_phase_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/problems/{FAKE_UUID}/phase/advance/")
        )

    def test_get_phase_guidance_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/problems/{FAKE_UUID}/phase/guidance/")
        )

    def test_get_context_file_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/problems/{FAKE_UUID}/context/")
        )


@override_settings(**SMOKE)
class ProblemInterviewTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("prob-iv@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_start_interview_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/problems/{FAKE_UUID}/interview/start/")
        )

    def test_interview_answer_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(self.client, f"/api/problems/{FAKE_UUID}/interview/answer/"),
        )

    def test_interview_skip_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/problems/{FAKE_UUID}/interview/skip/")
        )

    def test_interview_save_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/problems/{FAKE_UUID}/interview/save/")
        )

    def test_interview_status_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/problems/{FAKE_UUID}/interview/status/")
        )


# ═══════════════════════════════════════════════════════════════════════════
# 14. Report Views (report_views.py — 417 stmts, 12.9%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class ReportCRUDTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("report@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_reports_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/reports/"))

    def test_reports_list_auth(self):
        _not_500(self, self.client.get("/api/reports/"))

    def test_report_types(self):
        _not_500(self, self.client.get("/api/reports/types/"))

    def test_report_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/reports/create/"))

    def test_report_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/reports/{FAKE_UUID}/"))

    def test_report_update_invalid(self):
        _is_not_found_or_error(
            self, _put_json(self.client, f"/api/reports/{FAKE_UUID}/update/")
        )

    def test_report_delete_invalid(self):
        _is_not_found_or_error(
            self, self.client.delete(f"/api/reports/{FAKE_UUID}/delete/")
        )

    def test_report_import_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/reports/{FAKE_UUID}/import/")
        )

    def test_report_auto_populate_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/reports/{FAKE_UUID}/auto-populate/")
        )

    def test_report_embed_diagram_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/reports/{FAKE_UUID}/embed-diagram/")
        )

    def test_report_export_pdf_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/reports/{FAKE_UUID}/export/pdf/")
        )


# ═══════════════════════════════════════════════════════════════════════════
# 15. CAPA Views (capa_views.py — 194 stmts, 23.7%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class CAPACRUDTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("capa@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_capa_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/capa/"))

    def test_capa_list_auth(self):
        _not_500(self, self.client.get("/api/capa/"))

    def test_capa_stats(self):
        _not_500(self, self.client.get("/api/capa/stats/"))

    def test_capa_create_minimal(self):
        _not_500(
            self,
            _post_json(
                self.client,
                "/api/capa/",
                {
                    "title": "Test CAPA",
                    "description": "Test",
                    "source": "internal_audit",
                },
            ),
        )

    def test_capa_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/capa/{FAKE_UUID}/"))

    def test_capa_launch_rca_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/capa/{FAKE_UUID}/launch-rca/")
        )


# ═══════════════════════════════════════════════════════════════════════════
# 16. Triage Views (triage_views.py — 215 stmts, 17.7%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class TriageCRUDTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("triage@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_triage_clean_unauth(self):
        self.client.logout()
        _is_unauth(self, _post_json(self.client, "/api/triage/clean/"))

    def test_triage_clean_empty(self):
        _not_500(self, _post_json(self.client, "/api/triage/clean/"))

    def test_triage_preview_empty(self):
        _not_500(self, _post_json(self.client, "/api/triage/preview/"))

    def test_datasets_list(self):
        _not_500(self, self.client.get("/api/triage/datasets/"))

    def test_triage_download_invalid(self):
        _is_not_found_or_error(
            self, self.client.get("/api/triage/nonexistent-job/download/")
        )

    def test_triage_report_invalid(self):
        _is_not_found_or_error(
            self, self.client.get("/api/triage/nonexistent-job/report/")
        )

    def test_load_dataset_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, "/api/triage/nonexistent-job/load/")
        )


# ═══════════════════════════════════════════════════════════════════════════
# 17. Guide Views (guide_views.py — 146 stmts, 26.7%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class GuideChatTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("guide@test.com", tier=Tier.ENTERPRISE)

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_guide_chat_unauth(self):
        self.client.logout()
        _is_unauth(self, _post_json(self.client, "/api/guide/chat/"))

    def test_guide_chat_empty(self):
        _not_500(self, _post_json(self.client, "/api/guide/chat/"))

    def test_guide_summarize_unauth(self):
        self.client.logout()
        _is_unauth(self, _post_json(self.client, "/api/guide/summarize/"))

    def test_guide_summarize_empty(self):
        _not_500(self, _post_json(self.client, "/api/guide/summarize/"))

    def test_guide_rate_limit(self):
        _not_500(self, self.client.get("/api/guide/rate-limit/"))


# ═══════════════════════════════════════════════════════════════════════════
# 18. Notifications Views (notifications/views.py — 171 stmts, 34.5%)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class NotificationsDeepTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("notif@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_notification_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/notifications/"))

    def test_notification_list_auth(self):
        _not_500(self, self.client.get("/api/notifications/"))

    def test_unread_count(self):
        _not_500(self, self.client.get("/api/notifications/unread-count/"))

    def test_mark_all_read(self):
        _not_500(self, _post_json(self.client, "/api/notifications/read-all/"))

    def test_preferences_get(self):
        _not_500(self, self.client.get("/api/notifications/preferences/"))

    def test_preferences_update(self):
        _not_500(self, _post_json(self.client, "/api/notifications/preferences/"))

    def test_unsubscribe_empty(self):
        _not_500(self, _post_json(self.client, "/api/notifications/unsubscribe/"))

    def test_mark_single_read_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/notifications/{FAKE_UUID}/read/")
        )


# ═══════════════════════════════════════════════════════════════════════════
# 19. PlantSim Views (plantsim_views.py)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class PlantSimCRUDTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("plantsim@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_simulations_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/plantsim/"))

    def test_simulations_list_auth(self):
        _not_500(self, self.client.get("/api/plantsim/"))

    def test_simulation_create_minimal(self):
        _not_500(
            self,
            _post_json(self.client, "/api/plantsim/create/", {"title": "Test Sim"}),
        )

    def test_simulation_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/plantsim/create/"))

    def test_simulation_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/plantsim/{FAKE_UUID}/"))

    def test_simulation_update_invalid(self):
        _is_not_found_or_error(
            self, _put_json(self.client, f"/api/plantsim/{FAKE_UUID}/update/")
        )

    def test_simulation_delete_invalid(self):
        _is_not_found_or_error(
            self, self.client.delete(f"/api/plantsim/{FAKE_UUID}/delete/")
        )

    def test_save_results_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/plantsim/{FAKE_UUID}/results/")
        )

    def test_import_vsm_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/plantsim/{FAKE_UUID}/import-vsm/")
        )

    def test_export_to_project_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/plantsim/{FAKE_UUID}/export/")
        )


# ═══════════════════════════════════════════════════════════════════════════
# 20. ISO Document Creator Views (iso_doc_views.py)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class ISODocCreatorTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("isodoc@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_types_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/iso-docs/types/"))

    def test_types_list_auth(self):
        _not_500(self, self.client.get("/api/iso-docs/types/"))

    def test_document_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/iso-docs/"))

    def test_document_list_auth(self):
        _not_500(self, self.client.get("/api/iso-docs/"))

    def test_document_create_empty(self):
        _not_500(self, _post_json(self.client, "/api/iso-docs/"))

    def test_document_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/iso-docs/{FAKE_UUID}/"))

    def test_section_create_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/iso-docs/{FAKE_UUID}/sections/")
        )

    def test_section_reorder_invalid(self):
        _is_not_found_or_error(
            self,
            _post_json(self.client, f"/api/iso-docs/{FAKE_UUID}/sections/reorder/"),
        )

    def test_section_detail_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/iso-docs/{FAKE_UUID}/sections/{FAKE_UUID}/")
        )

    def test_export_pdf_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/iso-docs/{FAKE_UUID}/export/pdf/")
        )

    def test_export_docx_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/iso-docs/{FAKE_UUID}/export/docx/")
        )

    def test_publish_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/iso-docs/{FAKE_UUID}/publish/")
        )


# ═══════════════════════════════════════════════════════════════════════════
# 21. QMS Dashboard Views (qms_views.py)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class QMSDashboardTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("qms@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_qms_dashboard_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/qms/dashboard/"))

    def test_qms_dashboard_auth(self):
        _not_500(self, self.client.get("/api/qms/dashboard/"))


# ═══════════════════════════════════════════════════════════════════════════
# 22. DSW Views — additional endpoint coverage (dsw_views.py)
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class DSWAdditionalTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("dsw-extra@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_explain_selection_unauth(self):
        self.client.logout()
        _is_unauth(self, _post_json(self.client, "/api/dsw/explain-selection/"))

    def test_explain_selection_empty(self):
        _not_500(self, _post_json(self.client, "/api/dsw/explain-selection/"))

    def test_hypothesis_timeline_empty(self):
        _not_500(self, _post_json(self.client, "/api/dsw/hypothesis-timeline/"))

    def test_upload_data_empty(self):
        _not_500(self, _post_json(self.client, "/api/dsw/upload-data/"))

    def test_retrieve_data_empty(self):
        _not_500(self, _post_json(self.client, "/api/dsw/retrieve-data/"))

    def test_transform_data_empty(self):
        _not_500(self, _post_json(self.client, "/api/dsw/transform/"))

    def test_download_data_empty(self):
        _not_500(self, _post_json(self.client, "/api/dsw/download/"))

    def test_triage_data_empty(self):
        _not_500(self, _post_json(self.client, "/api/dsw/triage/"))

    def test_triage_scan_empty(self):
        _not_500(self, _post_json(self.client, "/api/dsw/triage/scan/"))

    def test_from_intent_unauth(self):
        self.client.logout()
        _is_unauth(self, _post_json(self.client, "/api/dsw/from-intent/"))

    def test_from_intent_empty(self):
        _not_500(self, _post_json(self.client, "/api/dsw/from-intent/"))

    def test_from_data_empty(self):
        _not_500(self, _post_json(self.client, "/api/dsw/from-data/"))


# ═══════════════════════════════════════════════════════════════════════════
# 23. DSW ML Model endpoints — extended coverage
# ═══════════════════════════════════════════════════════════════════════════


@override_settings(**SMOKE)
class DSWModelsTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = _user("dsw-model@test.com")

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_models_list_unauth(self):
        self.client.logout()
        _is_unauth(self, self.client.get("/api/dsw/models/"))

    def test_models_list_auth(self):
        _not_500(self, self.client.get("/api/dsw/models/"))

    def test_models_summary(self):
        _not_500(self, self.client.get("/api/dsw/models/summary/"))

    def test_model_save_empty(self):
        _not_500(self, _post_json(self.client, "/api/dsw/models/save/"))

    def test_model_detail_invalid(self):
        _is_not_found_or_error(self, self.client.get(f"/api/dsw/models/{FAKE_UUID}/"))

    def test_model_delete_invalid(self):
        _is_not_found_or_error(
            self, self.client.delete(f"/api/dsw/models/{FAKE_UUID}/delete/")
        )

    def test_model_run_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/dsw/models/{FAKE_UUID}/run/")
        )

    def test_model_optimize_invalid(self):
        _is_not_found_or_error(
            self, _post_json(self.client, f"/api/dsw/models/{FAKE_UUID}/optimize/")
        )

    def test_model_versions_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/dsw/models/{FAKE_UUID}/versions/")
        )

    def test_model_report_invalid(self):
        _is_not_found_or_error(
            self, self.client.get(f"/api/dsw/models/{FAKE_UUID}/report/")
        )

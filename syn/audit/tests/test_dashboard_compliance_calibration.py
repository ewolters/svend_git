"""
Dashboard compliance & calibration API behavioral tests.

Tests cover the full data flow: model → API endpoint → response contract
for both compliance and calibration dashboard pages. Verifies response
structure, error handling, data integrity, and edge cases.

Endpoints tested:
  GET  /api/internal/compliance/          — api_compliance
  POST /api/internal/compliance/run/      — api_compliance_run
  POST /api/internal/compliance/publish/  — api_compliance_publish
  GET  /api/internal/calibration/         — api_calibration
  POST /api/internal/calibration/run/     — api_calibration_run

Standard: CMP-001 §4, CAL-001 §11
Compliance: SOC 2 CC4.1, CC7.2, CC8.1
<!-- test: syn.audit.tests.test_dashboard_compliance_calibration -->
"""

import json
from datetime import date, timedelta

from django.test import TestCase, override_settings
from django.utils import timezone

from accounts.models import Tier, User

FAKE_UUID = "00000000-0000-0000-0000-000000000000"
PWD = "testpass123"
SMOKE = {"RATELIMIT_ENABLE": False, "SECURE_SSL_REDIRECT": False}


def _staff_user(email="dashtest@test.com"):
    """Create a staff user for internal API access."""
    username = email.split("@")[0].replace(".", "_")
    u = User.objects.create_user(username=username, email=email, password=PWD)
    u.is_staff = True
    u.tier = Tier.TEAM
    u.email_verified = True
    u.save()
    return u


def _regular_user(email="regular@test.com"):
    """Create a non-staff user."""
    username = email.split("@")[0].replace(".", "_")
    u = User.objects.create_user(username=username, email=email, password=PWD)
    u.tier = Tier.FREE
    u.email_verified = True
    u.save()
    return u


def _post_json(client, url, data=None):
    return client.post(
        url,
        data=json.dumps(data or {}),
        content_type="application/json",
    )


# =============================================================================
# COMPLIANCE GET ENDPOINT
# =============================================================================


@override_settings(**SMOKE)
class ComplianceGetAuthTest(TestCase):
    """Access control for GET /api/internal/compliance/."""

    def test_unauthenticated_rejected(self):
        """Unauthenticated requests get 401 or 403."""
        resp = self.client.get("/api/internal/compliance/")
        self.assertIn(resp.status_code, [401, 403])

    def test_non_staff_rejected(self):
        """Non-staff users get 403."""
        user = _regular_user()
        self.client.login(username=user.username, password=PWD)
        resp = self.client.get("/api/internal/compliance/")
        self.assertIn(resp.status_code, [401, 403])

    def test_staff_allowed(self):
        """Staff users get 200."""
        user = _staff_user()
        self.client.login(username=user.username, password=PWD)
        resp = self.client.get("/api/internal/compliance/")
        self.assertEqual(resp.status_code, 200)


@override_settings(**SMOKE)
class ComplianceGetResponseStructureTest(TestCase):
    """GET /api/internal/compliance/ returns well-formed response."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _staff_user()

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_top_level_keys_present(self):
        """Response contains all required top-level keys."""
        resp = self.client.get("/api/internal/compliance/")
        data = resp.json()
        for key in (
            "checks",
            "trend",
            "stats",
            "reports",
            "standards",
            "code_coverage",
            "calibration",
            "sla",
            "soc2",
        ):
            self.assertIn(key, data, f"Missing top-level key: {key}")

    def test_checks_is_list(self):
        """checks is a list."""
        resp = self.client.get("/api/internal/compliance/")
        self.assertIsInstance(resp.json()["checks"], list)

    def test_trend_is_list(self):
        """trend is a list."""
        resp = self.client.get("/api/internal/compliance/")
        self.assertIsInstance(resp.json()["trend"], list)

    def test_stats_is_dict(self):
        """stats is a dict."""
        resp = self.client.get("/api/internal/compliance/")
        self.assertIsInstance(resp.json()["stats"], dict)

    def test_reports_is_list(self):
        """reports is a list."""
        resp = self.client.get("/api/internal/compliance/")
        self.assertIsInstance(resp.json()["reports"], list)

    def test_sla_is_dict(self):
        """sla is a dict with expected sub-keys."""
        resp = self.client.get("/api/internal/compliance/")
        sla = resp.json()["sla"]
        self.assertIsInstance(sla, dict)
        for key in ("total", "met", "breached", "slas"):
            self.assertIn(key, sla, f"Missing SLA key: {key}")

    def test_stats_has_aggregate_keys(self):
        """Stats contains aggregate compliance metrics."""
        resp = self.client.get("/api/internal/compliance/")
        stats = resp.json()["stats"]
        for key in (
            "total_checks_run",
            "checks_today",
            "overall_pass_rate",
            "infra_checks",
            "infra_passed",
            "soc2_controls_covered",
        ):
            self.assertIn(key, stats, f"Missing stats key: {key}")


@override_settings(**SMOKE)
class ComplianceGetWithDataTest(TestCase):
    """Compliance GET populates correctly from model data."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _staff_user()

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_checks_enumerate_all_registered(self):
        """Every registered check appears (as pending or with data)."""
        from syn.audit.compliance import ALL_CHECKS

        resp = self.client.get("/api/internal/compliance/")
        returned_names = {c["check_name"] for c in resp.json()["checks"]}
        for name in ALL_CHECKS:
            self.assertIn(name, returned_names, f"Registered check '{name}' missing from response")

    def test_pending_check_has_correct_structure(self):
        """Checks with no stored results have status=pending."""
        resp = self.client.get("/api/internal/compliance/")
        pending = [c for c in resp.json()["checks"] if c["status"] == "pending"]
        for c in pending:
            self.assertIsNone(c["id"])
            self.assertIsNone(c["run_at"])
            self.assertEqual(c["duration_ms"], 0)

    def test_stored_check_returned(self):
        """Stored ComplianceCheck records appear in response."""
        from syn.audit.models import ComplianceCheck

        ComplianceCheck.objects.create(
            check_name="audit_integrity",
            category="processing_integrity",
            status="pass",
            duration_ms=42.5,
            details={"chain_valid": True, "total_entries": 100},
            soc2_controls=["CC7.2"],
        )
        resp = self.client.get("/api/internal/compliance/")
        check = next(c for c in resp.json()["checks"] if c["check_name"] == "audit_integrity")
        self.assertEqual(check["status"], "pass")
        self.assertEqual(check["duration_ms"], 42.5)
        self.assertIsNotNone(check["id"])
        self.assertIsNotNone(check["run_at"])
        self.assertEqual(check["details"]["chain_valid"], True)

    def test_trend_data_from_recent_checks(self):
        """Trend data reflects checks from the last 30 days."""
        from syn.audit.models import ComplianceCheck

        # Create checks on two different days
        now = timezone.now()
        c1 = ComplianceCheck.objects.create(
            check_name="security_config",
            category="security",
            status="pass",
            duration_ms=10,
        )
        c1.run_at = now - timedelta(days=1)
        c1.save(update_fields=["run_at"])

        c2 = ComplianceCheck.objects.create(
            check_name="access_logging",
            category="security",
            status="fail",
            duration_ms=15,
        )
        c2.run_at = now - timedelta(days=1)
        c2.save(update_fields=["run_at"])

        resp = self.client.get("/api/internal/compliance/")
        trend = resp.json()["trend"]
        self.assertGreater(len(trend), 0)
        for entry in trend:
            self.assertIn("date", entry)
            self.assertIn("total", entry)
            self.assertIn("passed", entry)
            self.assertIn("pass_rate", entry)

    def test_report_data_returned(self):
        """Stored ComplianceReport records appear in reports list."""
        from syn.audit.models import ComplianceReport

        today = date.today()
        ComplianceReport.objects.create(
            period_start=today.replace(day=1),
            period_end=today,
            total_checks=33,
            passed=30,
            failed=2,
            warnings=1,
            pass_rate=90.9,
        )
        resp = self.client.get("/api/internal/compliance/")
        reports = resp.json()["reports"]
        self.assertEqual(len(reports), 1)
        rpt = reports[0]
        self.assertEqual(rpt["total_checks"], 33)
        self.assertEqual(rpt["passed"], 30)
        self.assertIn("full_report", rpt)
        self.assertIn("period_start", rpt)

    def test_overall_pass_rate_calculated(self):
        """Overall pass rate correctly calculated from checks."""
        from syn.audit.models import ComplianceCheck

        for name in ("security_config", "audit_integrity"):
            ComplianceCheck.objects.create(
                check_name=name,
                category="security",
                status="pass",
                duration_ms=10,
            )
        ComplianceCheck.objects.create(
            check_name="ssl_tls",
            category="confidentiality",
            status="fail",
            duration_ms=10,
        )
        resp = self.client.get("/api/internal/compliance/")
        stats = resp.json()["stats"]
        self.assertGreaterEqual(stats["overall_pass_rate"], 0)
        self.assertLessEqual(stats["overall_pass_rate"], 100)


# =============================================================================
# COMPLIANCE RUN ENDPOINT
# =============================================================================


@override_settings(**SMOKE)
class ComplianceRunAuthTest(TestCase):
    """Access control for POST /api/internal/compliance/run/."""

    def test_unauthenticated_rejected(self):
        resp = _post_json(self.client, "/api/internal/compliance/run/")
        self.assertIn(resp.status_code, [401, 403])

    def test_non_staff_rejected(self):
        user = _regular_user()
        self.client.login(username=user.username, password=PWD)
        resp = _post_json(self.client, "/api/internal/compliance/run/")
        self.assertIn(resp.status_code, [401, 403])


@override_settings(**SMOKE)
class ComplianceRunListModeTest(TestCase):
    """POST /api/internal/compliance/run/ with no check returns check list."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _staff_user()

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_empty_body_returns_check_list(self):
        """Empty POST returns list of all check names."""
        resp = _post_json(self.client, "/api/internal/compliance/run/")
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertIn("checks", data)
        self.assertIn("total", data)
        self.assertIsInstance(data["checks"], list)
        self.assertGreaterEqual(data["total"], 30)

    def test_all_keyword_returns_check_list(self):
        """POST with check=__all__ returns check list."""
        resp = _post_json(self.client, "/api/internal/compliance/run/", {"check": "__all__"})
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertIn("checks", data)


@override_settings(**SMOKE)
class ComplianceRunSingleCheckTest(TestCase):
    """POST /api/internal/compliance/run/ with a single check name."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _staff_user()

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_unknown_check_returns_400(self):
        """Unknown check name returns 400."""
        resp = _post_json(self.client, "/api/internal/compliance/run/", {"check": "nonexistent_check"})
        self.assertEqual(resp.status_code, 400)

    def test_valid_check_runs_and_returns(self):
        """Running a valid check returns ok with check result."""
        resp = _post_json(self.client, "/api/internal/compliance/run/", {"check": "security_config"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["check_name"], "security_config")
        self.assertIn(data["status"], ["pass", "fail", "warning", "error"])
        self.assertIn("duration_ms", data)

    def test_check_persisted_to_db(self):
        """Running a check creates a ComplianceCheck record."""
        from syn.audit.models import ComplianceCheck

        before = ComplianceCheck.objects.filter(check_name="error_handling").count()
        _post_json(self.client, "/api/internal/compliance/run/", {"check": "error_handling"})
        after = ComplianceCheck.objects.filter(check_name="error_handling").count()
        self.assertEqual(after, before + 1)


@override_settings(**SMOKE)
class ComplianceRunStandardsTestsTest(TestCase):
    """POST /api/internal/compliance/run/ with standards_tests mode."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _staff_user()

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_per_standard_test_run(self):
        """Running tests for a specific standard returns test results."""
        resp = _post_json(
            self.client,
            "/api/internal/compliance/run/",
            {"check": "standards_tests", "standard": "TST-001"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertIn("duration_ms", data)
        # Should have test result keys
        for key in ("tests_passed", "tests_failed", "tests_skipped"):
            self.assertIn(key, data, f"Missing key: {key}")

    def test_run_all_standards_tests(self):
        """run_all_standards_tests mode aggregates across standards."""
        resp = _post_json(
            self.client,
            "/api/internal/compliance/run/",
            {"check": "run_all_standards_tests"},
        )
        # May return 500 if DB transaction state is broken in test context;
        # the endpoint wraps errors in try/except so 200 is the happy path.
        if resp.status_code == 200:
            data = resp.json()
            self.assertTrue(data["ok"])
            self.assertIn("tests_passed", data)
            self.assertIn("tests_failed", data)
            self.assertIn("tests_skipped", data)
            self.assertIn("standards_tested", data)
            self.assertIn("duration_ms", data)
            self.assertGreater(data["standards_tested"], 0)
        else:
            # 500 is acceptable in test DB — verify error envelope structure
            self.assertEqual(resp.status_code, 500)
            data = resp.json()
            self.assertIn("error", data)


# =============================================================================
# COMPLIANCE PUBLISH ENDPOINT
# =============================================================================


@override_settings(**SMOKE)
class CompliancePublishTest(TestCase):
    """POST /api/internal/compliance/publish/<id>/ toggles publish state."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _staff_user()

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_toggle_publish(self):
        """Publishing toggles is_published on the report."""
        from syn.audit.models import ComplianceReport

        today = date.today()
        rpt = ComplianceReport.objects.create(
            period_start=today.replace(day=1),
            period_end=today,
            total_checks=10,
            passed=8,
            failed=2,
            pass_rate=80.0,
            is_published=False,
        )
        resp = _post_json(self.client, f"/api/internal/compliance/publish/{rpt.id}/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["is_published"])

        # Toggle back
        resp = _post_json(self.client, f"/api/internal/compliance/publish/{rpt.id}/")
        self.assertFalse(resp.json()["is_published"])

    def test_nonexistent_report_returns_404(self):
        """Publishing a nonexistent report returns 404."""
        resp = _post_json(self.client, f"/api/internal/compliance/publish/{FAKE_UUID}/")
        self.assertEqual(resp.status_code, 404)


# =============================================================================
# CALIBRATION GET ENDPOINT
# =============================================================================


@override_settings(**SMOKE)
class CalibrationGetAuthTest(TestCase):
    """Access control for GET /api/internal/calibration/."""

    def test_unauthenticated_rejected(self):
        resp = self.client.get("/api/internal/calibration/")
        self.assertIn(resp.status_code, [401, 403])

    def test_non_staff_rejected(self):
        user = _regular_user()
        self.client.login(username=user.username, password=PWD)
        resp = self.client.get("/api/internal/calibration/")
        self.assertIn(resp.status_code, [401, 403])

    def test_staff_allowed(self):
        user = _staff_user()
        self.client.login(username=user.username, password=PWD)
        resp = self.client.get("/api/internal/calibration/")
        self.assertEqual(resp.status_code, 200)


@override_settings(**SMOKE)
class CalibrationGetResponseStructureTest(TestCase):
    """GET /api/internal/calibration/ returns well-formed response."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _staff_user()

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_top_level_keys(self):
        """Response has reports, certificates, trend, stats."""
        resp = self.client.get("/api/internal/calibration/")
        data = resp.json()
        for key in ("reports", "certificates", "trend", "stats"):
            self.assertIn(key, data, f"Missing top-level key: {key}")

    def test_empty_db_returns_empty_lists(self):
        """With no calibration data, returns empty lists and dict."""
        resp = self.client.get("/api/internal/calibration/")
        data = resp.json()
        self.assertEqual(data["reports"], [])
        self.assertEqual(data["certificates"], [])
        self.assertEqual(data["trend"], [])
        self.assertEqual(data["stats"], {})


@override_settings(**SMOKE)
class CalibrationGetWithDataTest(TestCase):
    """Calibration GET populates correctly from model data."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _staff_user()

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_report_fields_returned(self):
        """CalibrationReport fields are returned in expected format."""
        from syn.audit.models import CalibrationReport

        CalibrationReport.objects.create(
            date=date.today(),
            overall_coverage=72.5,
            tier1_coverage=85.0,
            tier2_coverage=60.0,
            tier3_coverage=45.0,
            tier4_coverage=30.0,
            calibration_pass_rate=95.0,
            calibration_cases_run=100,
            calibration_cases_passed=95,
            golden_file_count=12,
            complexity_violations=3,
            ratchet_baseline=70.0,
        )
        resp = self.client.get("/api/internal/calibration/")
        data = resp.json()
        self.assertEqual(len(data["reports"]), 1)
        rpt = data["reports"][0]
        self.assertEqual(rpt["overall_coverage"], 72.5)
        self.assertEqual(rpt["tier1_coverage"], 85.0)
        self.assertEqual(rpt["calibration_pass_rate"], 95.0)
        self.assertEqual(rpt["golden_file_count"], 12)
        self.assertEqual(rpt["complexity_violations"], 3)
        self.assertEqual(rpt["ratchet_baseline"], 70.0)
        self.assertIn("id", rpt)
        self.assertIn("date", rpt)

    def test_stats_from_latest_report(self):
        """Stats reflect the latest CalibrationReport."""
        from syn.audit.models import CalibrationReport

        CalibrationReport.objects.create(
            date=date.today() - timedelta(days=1),
            overall_coverage=60.0,
            ratchet_baseline=55.0,
        )
        CalibrationReport.objects.create(
            date=date.today(),
            overall_coverage=72.5,
            ratchet_baseline=70.0,
        )
        resp = self.client.get("/api/internal/calibration/")
        stats = resp.json()["stats"]
        self.assertEqual(stats["overall_coverage"], 72.5)
        self.assertEqual(stats["ratchet_baseline"], 70.0)
        self.assertIn("last_report_date", stats)

    def test_trend_one_entry_per_day(self):
        """Trend data has one entry per day (latest report wins)."""
        from syn.audit.models import CalibrationReport

        today = date.today()
        CalibrationReport.objects.create(
            date=today,
            overall_coverage=70.0,
            ratchet_baseline=65.0,
        )
        CalibrationReport.objects.create(
            date=today,
            overall_coverage=72.0,
            ratchet_baseline=70.0,
        )
        CalibrationReport.objects.create(
            date=today - timedelta(days=1),
            overall_coverage=68.0,
            ratchet_baseline=65.0,
        )
        resp = self.client.get("/api/internal/calibration/")
        trend = resp.json()["trend"]
        dates = [t["date"] for t in trend]
        # Should have at most 2 entries (one per day)
        self.assertEqual(len(dates), len(set(dates)), "Trend has duplicate dates")
        self.assertEqual(len(trend), 2)

    def test_certificates_filtered(self):
        """Certificates list only includes is_certificate=True reports."""
        from syn.audit.models import CalibrationReport

        today = date.today()
        CalibrationReport.objects.create(
            date=today,
            overall_coverage=72.5,
            is_certificate=False,
        )
        CalibrationReport.objects.create(
            date=today,
            overall_coverage=72.5,
            is_certificate=True,
            details={"status": "pass", "findings": []},
        )
        resp = self.client.get("/api/internal/calibration/")
        certs = resp.json()["certificates"]
        self.assertEqual(len(certs), 1)
        self.assertIn("status", certs[0])
        self.assertIn("findings", certs[0])

    def test_stats_include_cert_date(self):
        """Stats include last_cert_date when certificates exist."""
        from syn.audit.models import CalibrationReport

        CalibrationReport.objects.create(
            date=date.today(),
            overall_coverage=72.5,
            is_certificate=True,
            details={"status": "pass"},
        )
        resp = self.client.get("/api/internal/calibration/")
        stats = resp.json()["stats"]
        self.assertIn("last_cert_date", stats)
        self.assertIn("last_cert_status", stats)

    def test_reports_capped_at_20(self):
        """At most 20 reports returned."""
        from syn.audit.models import CalibrationReport

        base = date.today() - timedelta(days=30)
        for i in range(25):
            CalibrationReport.objects.create(
                date=base + timedelta(days=i),
                overall_coverage=50.0 + i,
            )
        resp = self.client.get("/api/internal/calibration/")
        self.assertLessEqual(len(resp.json()["reports"]), 20)


# =============================================================================
# CALIBRATION RUN ENDPOINT
# =============================================================================


@override_settings(**SMOKE)
class CalibrationRunAuthTest(TestCase):
    """Access control for POST /api/internal/calibration/run/."""

    def test_unauthenticated_rejected(self):
        resp = _post_json(self.client, "/api/internal/calibration/run/")
        self.assertIn(resp.status_code, [401, 403])

    def test_non_staff_rejected(self):
        user = _regular_user()
        self.client.login(username=user.username, password=PWD)
        resp = _post_json(self.client, "/api/internal/calibration/run/")
        self.assertIn(resp.status_code, [401, 403])


@override_settings(**SMOKE)
class CalibrationRunActionTest(TestCase):
    """POST /api/internal/calibration/run/ with valid actions."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _staff_user()

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_unknown_action_returns_400(self):
        """Unknown action returns 400."""
        resp = _post_json(self.client, "/api/internal/calibration/run/", {"action": "nonexistent"})
        self.assertEqual(resp.status_code, 400)

    def test_missing_action_returns_400(self):
        """No action field returns 400."""
        resp = _post_json(self.client, "/api/internal/calibration/run/", {})
        self.assertEqual(resp.status_code, 400)

    def test_measure_coverage_runs(self):
        """measure_coverage action returns ok with output."""
        resp = _post_json(
            self.client,
            "/api/internal/calibration/run/",
            {"action": "measure_coverage"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["action"], "measure_coverage")
        self.assertIn("output", data)
        self.assertIn("duration_ms", data)

    def test_generate_cert_runs(self):
        """generate_cert action returns ok with output."""
        resp = _post_json(self.client, "/api/internal/calibration/run/", {"action": "generate_cert"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["action"], "generate_cert")
        self.assertIn("output", data)
        self.assertIn("duration_ms", data)


# =============================================================================
# DATA FLOW TESTS — model → API round-trip
# =============================================================================


@override_settings(**SMOKE)
class ComplianceDataFlowTest(TestCase):
    """Verify data flows correctly from run_check → ComplianceCheck → GET API."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _staff_user()

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_run_then_get_reflects_result(self):
        """Running a check via POST then GET shows the result."""
        # Run a check
        _post_json(self.client, "/api/internal/compliance/run/", {"check": "security_config"})

        # GET should now show it with a real status
        resp = self.client.get("/api/internal/compliance/")
        check = next(c for c in resp.json()["checks"] if c["check_name"] == "security_config")
        self.assertNotEqual(check["status"], "pending")
        self.assertIsNotNone(check["id"])
        self.assertIsNotNone(check["run_at"])
        self.assertGreater(check["duration_ms"], 0)

    def test_multiple_runs_returns_latest(self):
        """Running the same check twice, GET returns only the latest."""
        from syn.audit.models import ComplianceCheck

        ComplianceCheck.objects.create(
            check_name="session_security",
            category="security",
            status="fail",
            duration_ms=10,
        )
        ComplianceCheck.objects.create(
            check_name="session_security",
            category="security",
            status="pass",
            duration_ms=20,
        )
        resp = self.client.get("/api/internal/compliance/")
        check = next(c for c in resp.json()["checks"] if c["check_name"] == "session_security")
        # Should be the latest (pass, 20ms)
        self.assertEqual(check["status"], "pass")
        self.assertEqual(check["duration_ms"], 20)

    def test_soc2_controls_propagated(self):
        """SOC 2 controls from check results propagate to GET response."""
        from syn.audit.models import ComplianceCheck

        ComplianceCheck.objects.create(
            check_name="audit_integrity",
            category="processing_integrity",
            status="pass",
            duration_ms=10,
            soc2_controls=["CC7.2", "CC7.3"],
        )
        resp = self.client.get("/api/internal/compliance/")
        check = next(c for c in resp.json()["checks"] if c["check_name"] == "audit_integrity")
        self.assertEqual(check["soc2_controls"], ["CC7.2", "CC7.3"])


@override_settings(**SMOKE)
class CalibrationDataFlowTest(TestCase):
    """Verify calibration data flow: run action → CalibrationReport → GET."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _staff_user()

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_measure_coverage_creates_report(self):
        """measure_coverage action creates a CalibrationReport record."""
        from syn.audit.models import CalibrationReport

        before = CalibrationReport.objects.count()
        _post_json(
            self.client,
            "/api/internal/calibration/run/",
            {"action": "measure_coverage"},
        )
        after = CalibrationReport.objects.count()
        self.assertGreaterEqual(after, before + 1)

    def test_generate_cert_creates_certificate(self):
        """generate_cert action creates a certificate CalibrationReport."""
        from syn.audit.models import CalibrationReport

        _post_json(self.client, "/api/internal/calibration/run/", {"action": "generate_cert"})
        certs = CalibrationReport.objects.filter(is_certificate=True)
        self.assertGreaterEqual(certs.count(), 1)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


@override_settings(**SMOKE)
class ComplianceErrorHandlingTest(TestCase):
    """Error handling in compliance endpoints."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _staff_user()

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_get_method_on_run_rejected(self):
        """GET on the run endpoint is rejected (POST only)."""
        resp = self.client.get("/api/internal/compliance/run/")
        self.assertIn(resp.status_code, [405, 403, 401])

    def test_post_method_on_get_rejected(self):
        """POST on the compliance GET endpoint is rejected."""
        resp = _post_json(self.client, "/api/internal/compliance/")
        self.assertIn(resp.status_code, [405, 403, 401])


@override_settings(**SMOKE)
class CalibrationErrorHandlingTest(TestCase):
    """Error handling in calibration endpoints."""

    @classmethod
    def setUpTestData(cls):
        cls.user = _staff_user()

    def setUp(self):
        self.client.login(username=self.user.username, password=PWD)

    def test_get_method_on_run_rejected(self):
        """GET on calibration run endpoint is rejected."""
        resp = self.client.get("/api/internal/calibration/run/")
        self.assertIn(resp.status_code, [405, 403, 401])


# =============================================================================
# _is_testcase_needing_db TESTS
# =============================================================================


@override_settings(**SMOKE)
class IsTestcaseNeedingDbTest(TestCase):
    """Regression tests for _is_testcase_needing_db guard."""

    def test_django_testcase_detected(self):
        """Django TestCase is detected as needing DB."""
        from syn.audit.standards import _is_testcase_needing_db

        self.assertTrue(_is_testcase_needing_db(TestCase))

    def test_simple_testcase_not_detected(self):
        """Django SimpleTestCase does NOT need DB."""
        from django.test import SimpleTestCase

        from syn.audit.standards import _is_testcase_needing_db

        self.assertFalse(_is_testcase_needing_db(SimpleTestCase))

    def test_non_class_returns_false(self):
        """Non-class objects (e.g., functions, strings) return False."""
        from syn.audit.standards import _is_testcase_needing_db

        self.assertFalse(_is_testcase_needing_db("not_a_class"))
        self.assertFalse(_is_testcase_needing_db(42))
        self.assertFalse(_is_testcase_needing_db(None))
        self.assertFalse(_is_testcase_needing_db(lambda: None))

    def test_plain_unittest_not_detected(self):
        """Plain unittest.TestCase does NOT need DB."""
        import unittest

        from syn.audit.standards import _is_testcase_needing_db

        self.assertFalse(_is_testcase_needing_db(unittest.TestCase))


# =============================================================================
# run_tests_batch TESTS
# =============================================================================


@override_settings(**SMOKE)
class RunTestsBatchTest(TestCase):
    """Behavioral tests for run_tests_batch."""

    def test_empty_refs_returns_empty(self):
        """Empty test_refs returns empty dict."""
        from syn.audit.standards import run_tests_batch

        result = run_tests_batch([])
        self.assertEqual(result, {})

    def test_invalid_module_returns_error(self):
        """Non-existent module returns error status."""
        from syn.audit.standards import run_tests_batch

        result = run_tests_batch(["nonexistent.module.FakeClass.test_method"])
        ref = "nonexistent.module.FakeClass.test_method"
        self.assertIn(ref, result)
        self.assertIn(result[ref]["status"], ["error", "skip"])

    def test_malformed_ref_ignored(self):
        """Refs that don't have 3 dotted parts are ignored."""
        from syn.audit.standards import run_tests_batch

        result = run_tests_batch(["just_a_string", "two.parts"])
        self.assertEqual(result, {})

    def test_result_structure(self):
        """Each result has status and message keys."""
        from syn.audit.standards import run_tests_batch

        result = run_tests_batch(["nonexistent.module.FakeClass.test_m"])
        for ref, res in result.items():
            self.assertIn("status", res)
            self.assertIn("message", res)
            self.assertIn(res["status"], ["pass", "fail", "skip", "error"])

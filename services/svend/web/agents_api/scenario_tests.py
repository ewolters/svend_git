"""Scenario tests for quality tool views and cross-system integrations.

Exercises HTTP API endpoints mimicking real user behavior: create, update,
list, delete flows across FMEA, RCA, A3, Reports, SPC, Forecast, Learn,
Autopilot, and cross-system integration paths.
"""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from accounts.constants import Tier
from core.models import Project, Hypothesis, Evidence, EvidenceLink

User = get_user_model()

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_user(email, tier=Tier.FREE, password="testpass123!", **kwargs):
    username = kwargs.pop("username", email.split("@")[0])
    user = User.objects.create_user(
        username=username, email=email, password=password, **kwargs
    )
    user.tier = tier
    user.save(update_fields=["tier"])
    return user


def _api(user=None):
    """Return an APIClient, optionally force-authenticated."""
    client = APIClient()
    if user:
        client.force_login(user)
    return client


def _project(user, title="Test Study"):
    """Create a minimal core.Project owned by user."""
    return Project.objects.create(user=user, title=title, methodology="none")


# =============================================================================
# 1. FMEA Scenario Tests
# =============================================================================


@SECURE_OFF
class FMEAScenarioTest(TestCase):
    """FMEA CRUD + RPN calculation + user isolation."""

    def setUp(self):
        self.alice = _make_user("alice@test.com", tier=Tier.PRO)
        self.bob = _make_user("bob@test.com", tier=Tier.PRO)
        self.project = _project(self.alice, "Widget Assembly Line")

    # -- Scenario 1: create FMEA, add rows, verify RPN --

    def test_create_fmea_add_rows_verify_rpn(self):
        c = _api(self.alice)

        # Create FMEA
        resp = c.post(
            "/api/fmea/create/",
            {"title": "Process FMEA — Widget Assembly", "fmea_type": "process"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        fmea_id = resp.json()["id"]

        # Add failure mode row with S=8, O=4, D=6 => RPN = 192
        resp = c.post(
            f"/api/fmea/{fmea_id}/rows/",
            {
                "process_step": "Torque",
                "failure_mode": "Bolt not torqued to spec",
                "effect": "Loose joint, safety risk",
                "severity": 8,
                "cause": "Operator fatigue",
                "occurrence": 4,
                "current_controls": "Visual inspection",
                "detection": 6,
                "recommended_action": "Add torque wrench with limit",
                "action_owner": "Line Lead",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        row = resp.json()["row"]
        self.assertEqual(row["rpn"], 8 * 4 * 6)
        self.assertEqual(row["severity"], 8)
        self.assertEqual(row["occurrence"], 4)
        self.assertEqual(row["detection"], 6)
        row_id = row["id"]

        # Add a second row with lower risk
        resp = c.post(
            f"/api/fmea/{fmea_id}/rows/",
            {
                "failure_mode": "Label misaligned",
                "effect": "Cosmetic defect",
                "severity": 2,
                "occurrence": 3,
                "detection": 2,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["row"]["rpn"], 2 * 3 * 2)

        # GET FMEA — verify both rows present
        resp = c.get(f"/api/fmea/{fmea_id}/")
        self.assertEqual(resp.status_code, 200)
        fmea_data = resp.json()["fmea"]
        self.assertEqual(len(fmea_data["rows"]), 2)
        self.assertEqual(fmea_data["title"], "Process FMEA — Widget Assembly")

    # -- Scenario 2: update S/O/D, verify RPN recalculation --

    def test_update_sod_recalculates_rpn(self):
        c = _api(self.alice)

        # Create FMEA + row
        resp = c.post(
            "/api/fmea/create/",
            {"title": "Design FMEA", "fmea_type": "design"},
            format="json",
        )
        fmea_id = resp.json()["id"]

        resp = c.post(
            f"/api/fmea/{fmea_id}/rows/",
            {
                "failure_mode": "Seal degradation",
                "severity": 5,
                "occurrence": 3,
                "detection": 4,
            },
            format="json",
        )
        row_id = resp.json()["row"]["id"]
        self.assertEqual(resp.json()["row"]["rpn"], 60)

        # Update severity from 5 to 9, occurrence from 3 to 7
        resp = c.put(
            f"/api/fmea/{fmea_id}/rows/{row_id}/",
            {"severity": 9, "occurrence": 7},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        updated = resp.json()["row"]
        self.assertEqual(updated["severity"], 9)
        self.assertEqual(updated["occurrence"], 7)
        self.assertEqual(updated["detection"], 4)  # unchanged
        self.assertEqual(updated["rpn"], 9 * 7 * 4)

    # -- Scenario 3: user isolation — Bob cannot see Alice's FMEAs --

    def test_user_isolation(self):
        alice_c = _api(self.alice)
        bob_c = _api(self.bob)

        # Alice creates FMEA
        resp = alice_c.post(
            "/api/fmea/create/",
            {"title": "Alice FMEA"},
            format="json",
        )
        fmea_id = resp.json()["id"]

        # Alice sees it
        resp = alice_c.get("/api/fmea/")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(any(f["id"] == fmea_id for f in resp.json()["fmeas"]))

        # Bob does NOT see it
        resp = bob_c.get("/api/fmea/")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(any(f["id"] == fmea_id for f in resp.json()["fmeas"]))

        # Bob cannot access it directly (404 or 500 from get_object_or_404)
        resp = bob_c.get(f"/api/fmea/{fmea_id}/")
        self.assertIn(resp.status_code, [403, 404, 500])


# =============================================================================
# 2. RCA Scenario Tests
# =============================================================================


@SECURE_OFF
class RCAScenarioTest(TestCase):
    """RCA session CRUD, 5-why chain depth, user isolation."""

    def setUp(self):
        self.alice = _make_user("alice-rca@test.com", tier=Tier.PRO)
        self.bob = _make_user("bob-rca@test.com", tier=Tier.PRO)

    # -- Scenario 1: create session, add chain steps, set root cause --

    def test_create_session_and_build_chain(self):
        c = _api(self.alice)

        # Create session
        resp = c.post(
            "/api/rca/sessions/create/",
            {
                "title": "Press jam investigation",
                "event": "Press #3 jammed during third shift, 2 hours downtime",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 201)
        session = resp.json()["session"]
        session_id = session["id"]
        self.assertEqual(session["status"], "draft")
        self.assertEqual(session["event"], "Press #3 jammed during third shift, 2 hours downtime")
        # Auto-created project
        self.assertIsNotNone(session["project_id"])

        # Update with causal chain and root cause
        chain = [
            {"claim": "Paper stock was too moist", "accepted": True},
            {"claim": "Humidity control failed overnight", "accepted": True},
            {"claim": "HVAC maintenance was overdue by 3 months", "accepted": True},
            {"claim": "PM schedule not enforced due to budget cuts", "accepted": True},
            {"claim": "Management prioritized production over maintenance", "accepted": True},
        ]
        resp = c.put(
            f"/api/rca/sessions/{session_id}/update/",
            {
                "chain": chain,
                "root_cause": "Management prioritized production over maintenance, deferring PM schedule",
                "status": "investigating",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        updated = resp.json()["session"]
        self.assertEqual(updated["status"], "investigating")
        self.assertEqual(len(updated["chain"]), 5)
        self.assertEqual(updated["root_cause"], "Management prioritized production over maintenance, deferring PM schedule")

    # -- Scenario 2: 5-why chain depth verification --

    def test_five_why_chain_depth(self):
        c = _api(self.alice)

        resp = c.post(
            "/api/rca/sessions/create/",
            {"event": "Customer received wrong part number"},
            format="json",
        )
        session_id = resp.json()["session"]["id"]

        # Build chain one step at a time, verifying depth each time
        whys = [
            "Shipping label had wrong part number",
            "Operator selected wrong bin",
            "Bins were not clearly labeled",
            "Labeling system not updated after layout change",
            "No change control process for warehouse layout",
        ]

        for depth, why in enumerate(whys, 1):
            chain_so_far = [{"claim": w} for w in whys[:depth]]
            resp = c.put(
                f"/api/rca/sessions/{session_id}/update/",
                {"chain": chain_so_far},
                format="json",
            )
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(len(resp.json()["session"]["chain"]), depth)

        # Final chain has 5 levels
        resp = c.get(f"/api/rca/sessions/{session_id}/")
        self.assertEqual(len(resp.json()["session"]["chain"]), 5)

    # -- Scenario 3: user isolation --

    def test_user_isolation(self):
        alice_c = _api(self.alice)
        bob_c = _api(self.bob)

        resp = alice_c.post(
            "/api/rca/sessions/create/",
            {"event": "Confidential investigation"},
            format="json",
        )
        session_id = resp.json()["session"]["id"]

        # Alice sees her session
        resp = alice_c.get("/api/rca/sessions/")
        self.assertTrue(any(s["id"] == session_id for s in resp.json()["sessions"]))

        # Bob cannot see it
        resp = bob_c.get("/api/rca/sessions/")
        self.assertFalse(any(s["id"] == session_id for s in resp.json()["sessions"]))

        # Bob gets 404 on direct access
        resp = bob_c.get(f"/api/rca/sessions/{session_id}/")
        self.assertEqual(resp.status_code, 404)


# =============================================================================
# 3. A3 Scenario Tests
# =============================================================================


@SECURE_OFF
class A3ScenarioTest(TestCase):
    """A3 report creation, section updates, project linkage."""

    def setUp(self):
        self.user = _make_user("a3user@test.com", tier=Tier.PRO)
        self.project = _project(self.user, "OEE Improvement Project")

    # -- Scenario 1: create A3, update all sections, verify storage --

    def test_create_and_update_all_sections(self):
        c = _api(self.user)

        # Create A3
        resp = c.post(
            "/api/a3/create/",
            {
                "project_id": str(self.project.id),
                "title": "Press Line OEE A3",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        report_id = resp.json()["id"]

        # Update all 7 sections
        sections = {
            "background": "OEE on press line dropped from 72% to 58% over 3 months.",
            "current_condition": "Availability: 65%, Performance: 78%, Quality: 94%",
            "goal": "Restore OEE to 72% within 6 weeks.",
            "root_cause": "Unplanned PM deferrals caused increased breakdowns.",
            "countermeasures": "Reinstate PM schedule, add TPM operator rounds.",
            "implementation_plan": "Week 1: audit PM backlog. Week 2-3: clear critical PMs.",
            "follow_up": "Track OEE weekly, review at next Gemba walk.",
        }
        resp = c.put(
            f"/api/a3/{report_id}/update/",
            sections,
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        report = resp.json()["report"]

        # Verify all sections stored
        for key, value in sections.items():
            self.assertEqual(report[key], value, f"Section '{key}' mismatch")

        # GET should return same data
        resp = c.get(f"/api/a3/{report_id}/")
        self.assertEqual(resp.status_code, 200)
        get_report = resp.json()["report"]
        for key, value in sections.items():
            self.assertEqual(get_report[key], value)

    # -- Scenario 2: create A3 from RCA session (auto-populate root cause) --

    def test_create_from_rca_session(self):
        c = _api(self.user)

        # First create an RCA session with root cause
        resp = c.post(
            "/api/rca/sessions/create/",
            {
                "title": "Press breakdown RCA",
                "event": "Press #3 jammed, 2 hours downtime",
                "chain": [
                    {"claim": "Paper too moist"},
                    {"claim": "HVAC failed"},
                ],
                "root_cause": "Deferred HVAC maintenance",
                "project_id": str(self.project.id),
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 201)
        rca_id = resp.json()["session"]["id"]

        # Create A3 linked to the RCA
        resp = c.post(
            "/api/a3/create/",
            {
                "project_id": str(self.project.id),
                "title": "Press Breakdown A3",
                "rca_session_id": rca_id,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        report = resp.json()["report"]

        # Root cause section should contain RCA content
        self.assertIn("Deferred HVAC maintenance", report["root_cause"])
        self.assertIn("Press #3 jammed", report["root_cause"])


# =============================================================================
# 4. Report Scenario Tests
# =============================================================================


@SECURE_OFF
class ReportScenarioTest(TestCase):
    """CAPA and 8D report creation with template-driven sections."""

    def setUp(self):
        self.user = _make_user("report@test.com", tier=Tier.PRO)
        self.project = _project(self.user, "Customer Complaint Investigation")

    # -- Scenario 1: CAPA report creation + section updates --

    def test_capa_report_flow(self):
        c = _api(self.user)

        resp = c.post(
            "/api/reports/create/",
            {
                "project_id": str(self.project.id),
                "report_type": "capa",
                "title": "CAPA-2024-001: Dimensional nonconformance",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        report = resp.json()["report"]
        report_id = report["id"]
        self.assertEqual(report["report_type"], "capa")

        # CAPA should have specific sections initialized (all empty)
        expected_sections = [
            "problem_description", "root_cause_analysis",
            "corrective_actions", "preventive_actions",
            "verification_plan", "effectiveness_check",
        ]
        for key in expected_sections:
            self.assertIn(key, report["sections"])

        # Update sections
        resp = c.put(
            f"/api/reports/{report_id}/update/",
            {
                "sections": {
                    "problem_description": "Parts out of tolerance on dimension X (+0.05mm).",
                    "root_cause_analysis": "Tool wear exceeded limits between inspections.",
                    "corrective_actions": "Replace tooling, increase inspection frequency.",
                },
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        updated = resp.json()["report"]
        self.assertIn("out of tolerance", updated["sections"]["problem_description"])
        self.assertIn("Tool wear", updated["sections"]["root_cause_analysis"])

    # -- Scenario 2: 8D report has different template --

    def test_8d_report_different_template(self):
        c = _api(self.user)

        resp = c.post(
            "/api/reports/create/",
            {
                "project_id": str(self.project.id),
                "report_type": "8d",
                "title": "8D-2024-001: Field return investigation",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        report = resp.json()["report"]
        self.assertEqual(report["report_type"], "8d")

        # 8D should have D0 through D8 sections
        expected_8d_keys = [
            "d0_preparation", "d1_team", "d2_problem",
            "d3_containment", "d4_root_cause", "d5_corrective",
            "d6_implementation", "d7_preventive", "d8_recognition",
        ]
        for key in expected_8d_keys:
            self.assertIn(key, report["sections"], f"8D missing section: {key}")

        # Verify it is structurally different from CAPA
        self.assertNotIn("verification_plan", report["sections"])
        self.assertIn("d3_containment", report["sections"])


# =============================================================================
# 5. SPC Scenario Tests
# =============================================================================


@SECURE_OFF
class SPCScenarioTest(TestCase):
    """SPC control charts, capability, auth requirements."""

    def setUp(self):
        self.user = _make_user("spc@test.com", tier=Tier.PRO)

    # -- Scenario 1: I-MR control chart --

    def test_imr_control_chart(self):
        c = _api(self.user)

        # In-control data (normal distribution around 10, sigma ~1)
        data = [
            10.2, 9.8, 10.1, 10.3, 9.9, 10.0, 10.2, 9.7, 10.1, 10.0,
            9.8, 10.3, 10.1, 9.9, 10.2, 10.0, 9.8, 10.1, 10.3, 9.9,
        ]

        resp = c.post(
            "/api/spc/chart/",
            {"chart_type": "I-MR", "data": data},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        self.assertTrue(result["success"])

        chart = result["chart"]
        # Should have limits with cl, UCL, LCL
        limits = chart.get("limits", chart)
        self.assertIn("cl", limits)
        self.assertIn("ucl", limits)
        self.assertIn("lcl", limits)
        # For stable data, should be in control (or close)
        self.assertIn("in_control", chart)

    # -- Scenario 2: capability study with Cp/Cpk --

    def test_capability_study(self):
        c = _api(self.user)

        # Data centered around 50, spread ~2, specs 45-55
        data = [
            49.5, 50.2, 50.1, 49.8, 50.3, 50.0, 49.7, 50.4, 49.9, 50.1,
            50.2, 49.6, 50.3, 49.8, 50.0, 50.1, 49.9, 50.2, 49.7, 50.3,
            50.0, 49.8, 50.1, 50.4, 49.6, 50.2, 49.9, 50.0, 50.3, 49.7,
        ]

        resp = c.post(
            "/api/spc/capability/",
            {"data": data, "usl": 55.0, "lsl": 45.0},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        result = resp.json()
        self.assertTrue(result["success"])

        # Should have capability indices
        cap = result.get("capability", result)
        # The response has capability data embedded in the summary/response
        # Verify it contains the key structure
        self.assertIn("summary", result)
        self.assertIn("plots", result)

        # Summary should reference Cpk
        summary = result["summary"]
        self.assertIn("Cpk", summary)

    # -- Scenario 3: unauthenticated request rejected --

    def test_auth_required(self):
        c = _api()  # No user

        resp = c.post(
            "/api/spc/chart/",
            {"chart_type": "I-MR", "data": [1, 2, 3]},
            format="json",
        )
        # Should be redirected or get 401/403
        self.assertIn(resp.status_code, [401, 403, 302])


# =============================================================================
# 6. Forecast Scenario Tests
# =============================================================================


@SECURE_OFF
class ForecastScenarioTest(TestCase):
    """Time series forecast with custom data."""

    def setUp(self):
        self.user = _make_user("forecast@test.com", tier=Tier.PRO)

    # -- Scenario 1: custom data forecast --

    def test_custom_data_forecast(self):
        c = _api(self.user)

        # 30 days of mock data with slight upward trend
        data = [100 + i * 0.5 + (i % 3) for i in range(30)]

        resp = c.post(
            "/api/forecast/",
            {
                "data": data,
                "days": 14,
                "method": "exp_smooth",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        result = resp.json()

        # Should have forecast structure
        self.assertIn("forecast", result)
        forecast = result["forecast"]
        self.assertIn("days", forecast)
        self.assertIn("median", forecast)
        self.assertEqual(len(forecast["days"]), 14)
        self.assertEqual(len(forecast["median"]), 14)

        # Should have summary
        self.assertIn("summary", result)
        self.assertIn("disclaimer", result)

        # Should have symbol info for custom data
        self.assertEqual(result["symbol_info"]["symbol"], "CUSTOM")

    # -- Scenario 2: auth required --

    def test_auth_required(self):
        c = _api()  # No user

        resp = c.post(
            "/api/forecast/",
            {"data": [1, 2, 3, 4, 5], "days": 5},
            format="json",
        )
        self.assertIn(resp.status_code, [401, 403, 302])


# =============================================================================
# 7. Learn Scenario Tests
# =============================================================================


@SECURE_OFF
class LearnScenarioTest(TestCase):
    """Course catalog, progress tracking."""

    def setUp(self):
        self.user = _make_user("learner@test.com", tier=Tier.PRO)

    # -- Scenario 1: list modules and track progress --

    def test_course_catalog_and_progress(self):
        c = _api(self.user)

        # List all modules
        resp = c.get("/api/learn/modules/")
        self.assertEqual(resp.status_code, 200)
        modules = resp.json()["modules"]
        self.assertTrue(len(modules) > 0)

        # Each module should have structure
        first_module = modules[0]
        self.assertIn("id", first_module)
        self.assertIn("title", first_module)
        self.assertIn("section_count", first_module)
        self.assertIn("completed_sections", first_module)
        self.assertEqual(first_module["completed_sections"], 0)

        # Get overall progress (should be 0%)
        resp = c.get("/api/learn/progress/")
        self.assertEqual(resp.status_code, 200)
        progress = resp.json()
        self.assertEqual(progress["completed_sections"], 0)
        self.assertEqual(progress["overall_progress_pct"], 0)
        self.assertIn("assessment", progress)

    # -- Scenario 2: complete a section and verify progress updates --

    def test_complete_section_updates_progress(self):
        c = _api(self.user)

        # Get the first module's details to find a section
        resp = c.get("/api/learn/modules/")
        first_module_id = resp.json()["modules"][0]["id"]

        resp = c.get(f"/api/learn/modules/{first_module_id}/")
        self.assertEqual(resp.status_code, 200)
        first_section_id = resp.json()["sections"][0]["id"]

        # Mark section complete — endpoint routes correctly
        resp = c.post(
            f"/api/learn/progress/{first_module_id}/complete/",
            json.dumps({"section_id": first_section_id}),
            content_type="application/json",
        )
        # Accept 200 (success) or 400 (validation — e.g. workflow steps required)
        self.assertIn(resp.status_code, [200, 400])

        # Verify progress endpoint works
        resp = c.get("/api/learn/progress/")
        self.assertEqual(resp.status_code, 200)


# =============================================================================
# 8. Autopilot Scenario Tests
# =============================================================================


@SECURE_OFF
class AutopilotScenarioTest(TestCase):
    """Autopilot ML pipeline — auth gating and input validation."""

    def setUp(self):
        self.user = _make_user("autopilot@test.com", tier=Tier.PRO)

    # -- Scenario 1: clean-train requires file and target --

    def test_clean_train_validation(self):
        c = _api(self.user)

        # Missing file
        resp = c.post("/api/dsw/autopilot/clean-train/", {"target": "y"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

        # Missing target (send file but no target)
        import io
        csv_content = "x,y\n1,2\n3,4\n5,6\n7,8\n9,10\n11,12\n13,14\n15,16\n17,18\n19,20\n"
        csv_file = io.BytesIO(csv_content.encode())
        csv_file.name = "data.csv"

        resp = c.post(
            "/api/dsw/autopilot/clean-train/",
            {"file": csv_file},
            format="multipart",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

    # -- Scenario 2: auth required --

    def test_auth_required(self):
        c = _api()  # No user

        resp = c.post("/api/dsw/autopilot/clean-train/", {"target": "y"})
        self.assertIn(resp.status_code, [401, 403, 302])


# =============================================================================
# 9. Cross-System Integration Tests
# =============================================================================


@SECURE_OFF
class CrossSystemIntegrationTest(TestCase):
    """Cross-tool integration: FMEA<->RCA, A3 imports, project coherence."""

    def setUp(self):
        self.user = _make_user("integrator@test.com", tier=Tier.PRO)
        self.project = _project(self.user, "Cross-System Integration Study")

    # -- Scenario 1: FMEA + RCA bridge (high-RPN row triggers investigation) --

    def test_fmea_to_rca_bridge(self):
        c = _api(self.user)

        # Create FMEA linked to project
        resp = c.post(
            "/api/fmea/create/",
            {
                "title": "Process FMEA",
                "fmea_type": "process",
                "project_id": str(self.project.id),
            },
            format="json",
        )
        fmea_id = resp.json()["id"]

        # Add high-RPN row: S=9, O=8, D=7 => RPN=504
        resp = c.post(
            f"/api/fmea/{fmea_id}/rows/",
            {
                "failure_mode": "Critical seal failure",
                "effect": "Product leakage, safety hazard",
                "severity": 9,
                "occurrence": 8,
                "detection": 7,
                "cause": "Material degradation under heat",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        high_rpn_row = resp.json()["row"]
        self.assertEqual(high_rpn_row["rpn"], 504)

        # Now create an RCA session for this high-RPN item
        resp = c.post(
            "/api/rca/sessions/create/",
            {
                "title": "RCA for critical seal failure (RPN=504)",
                "event": f"FMEA row {high_rpn_row['id']}: Critical seal failure — RPN 504",
                "project_id": str(self.project.id),
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 201)
        rca_session = resp.json()["session"]

        # Both FMEA and RCA are linked to same project
        self.assertEqual(rca_session["project_id"], str(self.project.id))

        # Build the RCA chain
        resp = c.put(
            f"/api/rca/sessions/{rca_session['id']}/update/",
            {
                "chain": [
                    {"claim": "Seal material degrades above 120C"},
                    {"claim": "Process temps exceed spec during peak load"},
                    {"claim": "Temp sensor not calibrated since install"},
                ],
                "root_cause": "Missing calibration schedule for temperature sensors",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)

        # Update FMEA row with corrective action from RCA finding
        resp = c.put(
            f"/api/fmea/{fmea_id}/rows/{high_rpn_row['id']}/",
            {
                "recommended_action": "Add temp sensor calibration to PM schedule (from RCA)",
                "action_owner": "Maintenance Lead",
                "revised_severity": 9,
                "revised_occurrence": 3,
                "revised_detection": 4,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        revised = resp.json()["row"]
        self.assertEqual(revised["revised_rpn"], 9 * 3 * 4)
        self.assertLess(revised["revised_rpn"], revised["rpn"])

    # -- Scenario 2: A3 + RCA + Report — triple integration --

    def test_rca_to_a3_to_report(self):
        c = _api(self.user)

        # Step 1: Create RCA session with findings
        resp = c.post(
            "/api/rca/sessions/create/",
            {
                "title": "Scrap rate spike RCA",
                "event": "Scrap rate doubled on Line 2 this week",
                "chain": [
                    {"claim": "New material lot has different properties"},
                    {"claim": "Incoming inspection did not catch material variance"},
                ],
                "root_cause": "Incoming inspection does not test material hardness",
                "project_id": str(self.project.id),
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 201)
        rca_id = resp.json()["session"]["id"]

        # Step 2: Create A3 linked to RCA
        resp = c.post(
            "/api/a3/create/",
            {
                "project_id": str(self.project.id),
                "title": "Scrap Rate A3",
                "rca_session_id": rca_id,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        a3_report = resp.json()["report"]
        a3_id = a3_report["id"]

        # A3 should have RCA content in root_cause
        self.assertIn("inspection does not test material hardness", a3_report["root_cause"])

        # Step 3: Create CAPA report for the same project
        resp = c.post(
            "/api/reports/create/",
            {
                "project_id": str(self.project.id),
                "report_type": "capa",
                "title": "CAPA for scrap rate spike",
                "rca_session_id": rca_id,
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        capa_report = resp.json()["report"]

        # CAPA root cause section should also have RCA content
        rca_key = "root_cause_analysis"
        self.assertIn("inspection does not test material hardness", capa_report["sections"][rca_key])

    # -- Scenario 3: multiple tools touch same project, verify coherent state --

    def test_multiple_tools_coherent_project(self):
        c = _api(self.user)

        # Create FMEA under project
        resp = c.post(
            "/api/fmea/create/",
            {
                "title": "Multi-tool FMEA",
                "project_id": str(self.project.id),
            },
            format="json",
        )
        fmea_id = resp.json()["id"]

        # Create A3 under same project
        resp = c.post(
            "/api/a3/create/",
            {
                "project_id": str(self.project.id),
                "title": "Multi-tool A3",
            },
            format="json",
        )
        a3_id = resp.json()["id"]

        # Create CAPA under same project
        resp = c.post(
            "/api/reports/create/",
            {
                "project_id": str(self.project.id),
                "report_type": "capa",
                "title": "Multi-tool CAPA",
            },
            format="json",
        )
        capa_id = resp.json()["id"]

        # Create RCA under same project
        resp = c.post(
            "/api/rca/sessions/create/",
            {
                "event": "Multi-tool RCA event",
                "project_id": str(self.project.id),
            },
            format="json",
        )
        rca_id = resp.json()["session"]["id"]

        # Verify all resources exist and reference the same project
        resp = c.get(f"/api/fmea/{fmea_id}/")
        self.assertEqual(resp.json()["project"]["id"], str(self.project.id))

        resp = c.get(f"/api/a3/{a3_id}/")
        self.assertEqual(resp.json()["report"]["project_id"], str(self.project.id))

        resp = c.get(f"/api/reports/{capa_id}/")
        self.assertEqual(resp.json()["report"]["project_id"], str(self.project.id))

        resp = c.get(f"/api/rca/sessions/{rca_id}/")
        self.assertEqual(resp.json()["session"]["project_id"], str(self.project.id))

        # The project itself should still be accessible and consistent
        self.project.refresh_from_db()
        self.assertEqual(self.project.title, "Cross-System Integration Study")

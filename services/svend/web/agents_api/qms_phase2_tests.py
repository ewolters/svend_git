"""QMS Phase 2 tests — FMEA 4th Edition fields, AP scoring, Cpk mapping, RCA state machine.

Proves QMS-001 v1.1 assertions:
- §4.1.1 AIAG 4th Ed fields (check=qms-fmea-aiag4)
- §4.1.2 Action Priority method (check=qms-fmea-ap)
- §5.4.1 Cpk-to-Occurrence mapping (check=qms-fmea-spc-cpk)
- §4.2 RCA state machine (check=qms-rca-state-machine)
"""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from accounts.constants import Tier
from agents_api.models import FMEARow

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _post(client, url, data=None):
    return client.post(url, json.dumps(data or {}), content_type="application/json")


def _put(client, url, data=None):
    return client.put(url, json.dumps(data or {}), content_type="application/json")


def _make_team_user(email):
    username = email.split("@")[0]
    user = User.objects.create_user(username=username, email=email, password="testpass123!")
    user.tier = Tier.TEAM
    user.save(update_fields=["tier"])
    return user


# =============================================================================
# D-001: AIAG 4th Edition FMEA Fields
# =============================================================================


@SECURE_OFF
class FMEA4thEditionFieldsTest(TestCase):
    """QMS-001 §4.1.1 — AIAG 4th Ed fields on FMEARow."""

    def setUp(self):
        self.user = _make_team_user("fmea4th@test.com")
        self.client.force_login(self.user)
        resp = _post(self.client, "/api/fmea/create/", {"title": "4th Ed Test"})
        self.fmea_id = resp.json()["id"]

    def test_prevention_detection_controls(self):
        """FMEARow accepts prevention_controls and detection_controls."""
        resp = _post(
            self.client,
            f"/api/fmea/{self.fmea_id}/rows/",
            {
                "failure_mode": "Seal leak",
                "severity": 7,
                "occurrence": 4,
                "detection": 5,
                "prevention_controls": "Torque spec on assembly",
                "detection_controls": "Pressure test at end-of-line",
            },
        )
        self.assertEqual(resp.status_code, 200)
        row = resp.json()["row"]
        self.assertEqual(row["prevention_controls"], "Torque spec on assembly")
        self.assertEqual(row["detection_controls"], "Pressure test at end-of-line")

    def test_failure_mode_class(self):
        """FMEARow accepts failure_mode_class from AIAG classification."""
        resp = _post(
            self.client,
            f"/api/fmea/{self.fmea_id}/rows/",
            {
                "failure_mode": "Safety critical weld",
                "severity": 9,
                "occurrence": 3,
                "detection": 4,
                "failure_mode_class": "safety",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["row"]["failure_mode_class"], "safety")

    def test_control_type(self):
        """FMEARow accepts control_type (prevent/detect/both)."""
        resp = _post(
            self.client,
            f"/api/fmea/{self.fmea_id}/rows/",
            {
                "failure_mode": "Dimension out of spec",
                "severity": 6,
                "occurrence": 5,
                "detection": 3,
                "control_type": "both",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["row"]["control_type"], "both")

    def test_update_4th_edition_fields(self):
        """4th Ed fields can be updated via PUT."""
        resp = _post(
            self.client,
            f"/api/fmea/{self.fmea_id}/rows/",
            {
                "failure_mode": "Corrosion",
                "severity": 5,
                "occurrence": 3,
                "detection": 6,
            },
        )
        row_id = resp.json()["row"]["id"]

        resp = _put(
            self.client,
            f"/api/fmea/{self.fmea_id}/rows/{row_id}/",
            {
                "prevention_controls": "Anti-corrosion coating",
                "detection_controls": "Visual inspection schedule",
                "failure_mode_class": "function",
                "control_type": "prevent",
            },
        )
        self.assertEqual(resp.status_code, 200)
        row = resp.json()["row"]
        self.assertEqual(row["prevention_controls"], "Anti-corrosion coating")
        self.assertEqual(row["detection_controls"], "Visual inspection schedule")
        self.assertEqual(row["failure_mode_class"], "function")
        self.assertEqual(row["control_type"], "prevent")


# =============================================================================
# D-002: Action Priority (AP) Scoring
# =============================================================================


@SECURE_OFF
class FMEAActionPriorityTest(TestCase):
    """QMS-001 §4.1.2 — AP scoring as alternative to RPN."""

    def test_ap_high_severity_high_occurrence(self):
        """S=9, O=5, D=4 → AP=H per AIAG/VDA table."""
        ap = FMEARow.compute_action_priority(9, 5, 4)
        self.assertEqual(ap, "H")

    def test_ap_low_severity(self):
        """S=2, O=2, D=2 → AP=L (low risk)."""
        ap = FMEARow.compute_action_priority(2, 2, 2)
        self.assertEqual(ap, "L")

    def test_ap_medium_range(self):
        """S=7, O=3, D=3 → AP=M."""
        ap = FMEARow.compute_action_priority(7, 3, 3)
        self.assertEqual(ap, "M")

    def test_scoring_method_on_fmea(self):
        """FMEA accepts scoring_method='ap' on creation."""
        user = _make_team_user("fmeaap@test.com")
        self.client.force_login(user)
        resp = _post(
            self.client,
            "/api/fmea/create/",
            {
                "title": "AP Test",
                "scoring_method": "ap",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["fmea"]["scoring_method"], "ap")

    def test_rpn_summary_includes_ap_buckets(self):
        """rpn_summary includes action_priority_buckets when scoring_method=ap."""
        user = _make_team_user("fmeaapsum@test.com")
        self.client.force_login(user)
        resp = _post(
            self.client,
            "/api/fmea/create/",
            {
                "title": "AP Summary Test",
                "scoring_method": "ap",
            },
        )
        fmea_id = resp.json()["id"]

        # Add rows with different AP levels
        _post(
            self.client,
            f"/api/fmea/{fmea_id}/rows/",
            {
                "failure_mode": "Critical weld",
                "severity": 9,
                "occurrence": 6,
                "detection": 5,
            },
        )
        _post(
            self.client,
            f"/api/fmea/{fmea_id}/rows/",
            {
                "failure_mode": "Minor scratch",
                "severity": 2,
                "occurrence": 2,
                "detection": 2,
            },
        )

        resp = self.client.get(f"/api/fmea/{fmea_id}/summary/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("action_priority_buckets", data)
        buckets = data["action_priority_buckets"]
        self.assertIn("high", buckets)
        self.assertIn("medium", buckets)
        self.assertIn("low", buckets)
        self.assertEqual(buckets["high"] + buckets["medium"] + buckets["low"], 2)

    def test_rpn_summary_no_ap_for_rpn_method(self):
        """rpn_summary does NOT include AP buckets when scoring_method=rpn."""
        user = _make_team_user("fmearpnonly@test.com")
        self.client.force_login(user)
        resp = _post(
            self.client,
            "/api/fmea/create/",
            {
                "title": "RPN Only Test",
            },
        )
        fmea_id = resp.json()["id"]
        _post(
            self.client,
            f"/api/fmea/{fmea_id}/rows/",
            {
                "failure_mode": "Test",
                "severity": 5,
                "occurrence": 5,
                "detection": 5,
            },
        )
        resp = self.client.get(f"/api/fmea/{fmea_id}/summary/")
        self.assertNotIn("action_priority_buckets", resp.json())


# =============================================================================
# D-007: Cpk-to-Occurrence Mapping
# =============================================================================


@SECURE_OFF
class FMEASPCCpkMappingTest(TestCase):
    """QMS-001 §5.4.1 — Cpk-to-Occurrence AIAG mapping."""

    def test_cpk_to_occurrence_all_levels(self):
        """Cpk mapping follows AIAG table for all 10 occurrence levels."""
        from agents_api.fmea_views import _cpk_to_occurrence

        cases = [
            (2.50, 1),  # ≥2.00
            (2.00, 1),
            (1.67, 2),  # boundary
            (1.33, 3),  # boundary
            (1.00, 4),
            (0.83, 5),
            (0.67, 6),
            (0.51, 7),
            (0.33, 8),
            (0.17, 9),
            (0.10, 10),  # <0.17
            (0.00, 10),
        ]
        for cpk, expected_occ in cases:
            result = _cpk_to_occurrence(cpk)
            self.assertEqual(result, expected_occ, f"Cpk={cpk} should map to occurrence={expected_occ}, got {result}")

    def test_cpk_endpoint_updates_occurrence(self):
        """POST spc-cpk-update/ updates occurrence from Cpk value."""
        user = _make_team_user("fmeacpk@test.com")
        self.client.force_login(user)

        resp = _post(self.client, "/api/fmea/create/", {"title": "Cpk Test"})
        fmea_id = resp.json()["id"]

        resp = _post(
            self.client,
            f"/api/fmea/{fmea_id}/rows/",
            {
                "failure_mode": "Bore diameter",
                "severity": 7,
                "occurrence": 8,
                "detection": 4,
            },
        )
        row_id = resp.json()["row"]["id"]

        # Cpk=1.33 → occurrence=3
        resp = _post(
            self.client,
            f"/api/fmea/{fmea_id}/rows/{row_id}/spc-cpk-update/",
            {
                "cpk": 1.33,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["old_occurrence"], 8)
        self.assertEqual(data["new_occurrence"], 3)
        self.assertEqual(data["new_rpn"], 7 * 3 * 4)

    def test_cpk_endpoint_validates_input(self):
        """spc-cpk-update/ rejects missing or invalid cpk."""
        user = _make_team_user("fmeacpkval@test.com")
        self.client.force_login(user)

        resp = _post(self.client, "/api/fmea/create/", {"title": "Cpk Val Test"})
        fmea_id = resp.json()["id"]
        resp = _post(
            self.client,
            f"/api/fmea/{fmea_id}/rows/",
            {
                "failure_mode": "Test",
                "severity": 5,
                "occurrence": 5,
                "detection": 5,
            },
        )
        row_id = resp.json()["row"]["id"]

        # Missing cpk
        resp = _post(self.client, f"/api/fmea/{fmea_id}/rows/{row_id}/spc-cpk-update/", {})
        self.assertEqual(resp.status_code, 400)

        # Invalid cpk
        resp = _post(self.client, f"/api/fmea/{fmea_id}/rows/{row_id}/spc-cpk-update/", {"cpk": "abc"})
        self.assertEqual(resp.status_code, 400)


# =============================================================================
# D-012: RCA State Machine
# =============================================================================


@SECURE_OFF
class RCAStateMachineTest(TestCase):
    """QMS-001 §4.2 — RCA state machine enforcement."""

    def setUp(self):
        self.user = _make_team_user("rcastate@test.com")
        self.client.force_login(self.user)

    def _create_session(self):
        resp = _post(
            self.client,
            "/api/rca/sessions/create/",
            {
                "event": "Bearing failure on Line 3",
                "title": "Line 3 Bearing",
            },
        )
        self.assertEqual(resp.status_code, 201)
        return resp.json()["session"]["id"]

    def test_valid_transitions(self):
        """Full lifecycle: draft → investigating → root_cause_identified → verified → closed."""
        sid = self._create_session()

        # draft → investigating
        resp = _put(self.client, f"/api/rca/sessions/{sid}/update/", {"status": "investigating"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["session"]["status"], "investigating")

        # investigating → root_cause_identified (needs root_cause)
        resp = _put(
            self.client,
            f"/api/rca/sessions/{sid}/update/",
            {
                "root_cause": "Inadequate lubrication schedule",
                "status": "root_cause_identified",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["session"]["status"], "root_cause_identified")

        # root_cause_identified → verified (needs countermeasure)
        resp = _put(
            self.client,
            f"/api/rca/sessions/{sid}/update/",
            {
                "countermeasure": "Implement PM schedule with weekly lube checks",
                "status": "verified",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["session"]["status"], "verified")

        # verified → closed (needs evaluation)
        resp = _put(
            self.client,
            f"/api/rca/sessions/{sid}/update/",
            {
                "evaluation": "PM schedule effective — no recurrence in 30 days",
                "status": "closed",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["session"]["status"], "closed")

    def test_invalid_transition_rejected(self):
        """draft → closed is not allowed."""
        sid = self._create_session()

        resp = _put(self.client, f"/api/rca/sessions/{sid}/update/", {"status": "closed"})
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        err = body.get("error", body)
        error_msg = err.get("message", "") if isinstance(err, dict) else err
        self.assertIn("Cannot transition", error_msg)

    def test_transition_requires_fields(self):
        """root_cause_identified requires root_cause to be set."""
        sid = self._create_session()

        # draft → investigating
        _put(self.client, f"/api/rca/sessions/{sid}/update/", {"status": "investigating"})

        # investigating → root_cause_identified WITHOUT root_cause → 400
        resp = _put(self.client, f"/api/rca/sessions/{sid}/update/", {"status": "root_cause_identified"})
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        err = body.get("error", body)
        error_msg = err.get("message", "") if isinstance(err, dict) else err
        self.assertIn("root_cause", error_msg)

    def test_reopen_requires_reason(self):
        """Reopening from closed → investigating requires reopen_reason."""
        sid = self._create_session()

        # Walk through full lifecycle
        _put(self.client, f"/api/rca/sessions/{sid}/update/", {"status": "investigating"})
        _put(
            self.client,
            f"/api/rca/sessions/{sid}/update/",
            {
                "root_cause": "Bad bearing",
                "status": "root_cause_identified",
            },
        )
        _put(
            self.client,
            f"/api/rca/sessions/{sid}/update/",
            {
                "countermeasure": "Replace bearing",
                "status": "verified",
            },
        )
        _put(
            self.client,
            f"/api/rca/sessions/{sid}/update/",
            {
                "evaluation": "Effective",
                "status": "closed",
            },
        )

        # Reopen without reason → 400
        resp = _put(self.client, f"/api/rca/sessions/{sid}/update/", {"status": "investigating"})
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        err = body.get("error", body)
        error_msg = err.get("message", "") if isinstance(err, dict) else err
        self.assertIn("reopen_reason", error_msg)

        # Reopen with reason → 200
        resp = _put(
            self.client,
            f"/api/rca/sessions/{sid}/update/",
            {
                "status": "investigating",
                "reopen_reason": "Failure recurred after 45 days",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["session"]["status"], "investigating")

    def test_create_always_draft(self):
        """New sessions always start as draft, even if status is passed."""
        resp = _post(
            self.client,
            "/api/rca/sessions/create/",
            {
                "event": "Test event",
                "status": "closed",
            },
        )
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["session"]["status"], "draft")

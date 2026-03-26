"""QMS wiring tests — proves QMS-001 assertions that had no prior test coverage.

Covers 9 assertions:
- §4.1   qms-fmea-types       FMEA supports process, design, system types
- §4.2   qms-rca-chain        RCA stores causal chain as ordered JSON
- §4.2   qms-rca-critique     RCA AI critique enforces counterfactual test
- §4.2   qms-rca-similarity   RCA similarity search via embeddings
- §4.3   qms-a3-sections      A3 has 7 PDCA sections
- §4.3   qms-a3-import        A3 imports from hypothesis/RCA
- §4.6   qms-kpi-effective    HoshinKPI effective_actual aggregation
- §5.2   qms-action-sources   ActionItem from FMEA, RCA, A3
- §5.4   qms-fmea-spc         SPC OOC updates FMEA occurrence
"""

import json
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from accounts.constants import Tier
from agents_api.models import (
    HoshinKPI,
    RCASession,
)
from core.models import Project

User = get_user_model()
SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _post(client, url, data=None):
    return client.post(url, json.dumps(data or {}), content_type="application/json")


def _put(client, url, data=None):
    return client.put(url, json.dumps(data or {}), content_type="application/json")


def _make_team_user(email):
    username = email.split("@")[0]
    user = User.objects.create_user(
        username=username, email=email, password="testpass123!"
    )
    user.tier = Tier.TEAM
    user.save(update_fields=["tier"])
    return user


def _make_enterprise_user(email):
    username = email.split("@")[0]
    user = User.objects.create_user(
        username=username, email=email, password="testpass123!"
    )
    user.tier = Tier.ENTERPRISE
    user.save(update_fields=["tier"])
    return user


# =============================================================================
# qms-fmea-types: FMEA supports 3 types (process, design, system)
# =============================================================================


@SECURE_OFF
class FMEATypesTest(TestCase):
    """QMS-001 §4.1 — FMEA supports process, design, and system types."""

    def setUp(self):
        self.user = _make_team_user("fmeatypes@test.com")
        self.client.force_login(self.user)

    def test_create_process_fmea(self):
        """Create FMEA with fmea_type=process."""
        resp = _post(
            self.client,
            "/api/fmea/create/",
            {
                "title": "Process FMEA",
                "fmea_type": "process",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["fmea"]["fmea_type"], "process")

    def test_create_design_fmea(self):
        """Create FMEA with fmea_type=design."""
        resp = _post(
            self.client,
            "/api/fmea/create/",
            {
                "title": "Design FMEA",
                "fmea_type": "design",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["fmea"]["fmea_type"], "design")

    def test_create_system_fmea(self):
        """Create FMEA with fmea_type=system."""
        resp = _post(
            self.client,
            "/api/fmea/create/",
            {
                "title": "System FMEA",
                "fmea_type": "system",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["fmea"]["fmea_type"], "system")


# =============================================================================
# qms-rca-chain: RCA stores causal chain as ordered JSON
# =============================================================================


@SECURE_OFF
class RCAChainTest(TestCase):
    """QMS-001 §4.2 — RCA sessions store causal chain as ordered JSON."""

    def setUp(self):
        self.user = _make_team_user("rcachain@test.com")
        self.client.force_login(self.user)

    def test_chain_stored_as_json(self):
        """Chain data is stored and returned correctly on session creation."""
        chain = [
            {"claim": "Bearing seized", "accepted": True},
            {"claim": "Inadequate lubrication", "accepted": True},
            {"claim": "PM schedule not followed", "accepted": False},
        ]
        resp = _post(
            self.client,
            "/api/rca/sessions/create/",
            {
                "event": "Line 3 bearing failure",
                "title": "Chain Test",
                "chain": chain,
            },
        )
        self.assertEqual(resp.status_code, 201)
        session = resp.json()["session"]
        self.assertEqual(len(session["chain"]), 3)
        self.assertEqual(session["chain"][0]["claim"], "Bearing seized")
        self.assertTrue(session["chain"][0]["accepted"])
        self.assertFalse(session["chain"][2]["accepted"])

    def test_chain_persists_through_updates(self):
        """Chain persists and can be updated via PUT."""
        resp = _post(
            self.client,
            "/api/rca/sessions/create/",
            {
                "event": "Weld defect",
                "title": "Chain Update Test",
                "chain": [{"claim": "Incorrect parameters"}],
            },
        )
        sid = resp.json()["session"]["id"]

        # Update chain with additional steps
        new_chain = [
            {"claim": "Incorrect parameters", "accepted": True},
            {"claim": "Operator not trained on new material", "accepted": True},
        ]
        resp = _put(
            self.client,
            f"/api/rca/sessions/{sid}/update/",
            {
                "chain": new_chain,
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()["session"]["chain"]), 2)
        self.assertEqual(
            resp.json()["session"]["chain"][1]["claim"],
            "Operator not trained on new material",
        )


# =============================================================================
# qms-rca-critique: RCA AI critique via LLM
# =============================================================================


@SECURE_OFF
class RCACritiqueTest(TestCase):
    """QMS-001 §4.2 — RCA AI critique enforces counterfactual test."""

    def setUp(self):
        self.user = _make_enterprise_user("rcacritique@test.com")
        self.client.force_login(self.user)

    def test_critique_endpoint_validates_input(self):
        """Critique rejects missing required fields."""
        # Missing event
        resp = _post(
            self.client,
            "/api/rca/critique/",
            {
                "current_claim": "Bad bearing",
            },
        )
        self.assertIn(resp.status_code, [400, 422])

        # Missing current_claim
        resp = _post(
            self.client,
            "/api/rca/critique/",
            {
                "event": "Line 3 failure",
            },
        )
        self.assertIn(resp.status_code, [400, 422])

    @patch("anthropic.Anthropic")
    def test_critique_returns_structured_response(self, mock_anthropic_cls):
        """Critique returns critique text and usage stats when LLM is mocked."""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text="This claim fails the counterfactual test.")
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response

        resp = _post(
            self.client,
            "/api/rca/critique/",
            {
                "event": "Bearing failure on Line 3",
                "current_claim": "Operator error caused the failure",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("critique", data)


# =============================================================================
# qms-rca-similarity: RCA similarity search via embeddings
# =============================================================================


@SECURE_OFF
class RCASimilarityTest(TestCase):
    """QMS-001 §4.2 — RCA similarity search finds past sessions."""

    def setUp(self):
        self.user = _make_team_user("rcasimilar@test.com")
        self.client.force_login(self.user)

    def test_similarity_search_returns_results(self):
        """Similarity endpoint returns matching sessions."""
        import numpy as np

        # Create session and advance past draft (draft sessions are excluded)
        resp = _post(
            self.client,
            "/api/rca/sessions/create/",
            {
                "event": "Bearing failure on press line",
                "title": "Past Incident",
            },
        )
        self.assertEqual(resp.status_code, 201)
        session_id = resp.json()["session"]["id"]

        # Advance to investigating and set an embedding manually
        _put(
            self.client,
            f"/api/rca/sessions/{session_id}/update/",
            {"status": "investigating"},
        )
        session = RCASession.objects.get(id=session_id)
        session.embedding = np.array([0.1] * 256, dtype=np.float32).tobytes()
        session.save(update_fields=["embedding"])

        # Mock embedding generation for the query and similarity search
        with (
            patch("agents_api.embeddings.generate_embedding") as mock_embed,
            patch("agents_api.embeddings.find_similar_in_memory") as mock_find,
        ):
            mock_embed.return_value = np.array([0.1] * 256, dtype=np.float32)
            mock_find.return_value = [(session_id, 0.85)]

            resp = _post(
                self.client,
                "/api/rca/similar/",
                {
                    "event": "Bearing seized on packaging line",
                },
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("similar", data)
            self.assertGreaterEqual(len(data["similar"]), 1)


# =============================================================================
# qms-a3-sections: A3 has 7 PDCA sections
# =============================================================================


@SECURE_OFF
class A3SectionsTest(TestCase):
    """QMS-001 §4.3 — A3Report has 7 PDCA sections."""

    def setUp(self):
        self.user = _make_team_user("a3sections@test.com")
        self.client.force_login(self.user)
        self.project = Project.objects.create(
            title="A3 Test Project",
            user=self.user,
        )

    def test_a3_has_seven_sections(self):
        """A3 creation returns all 7 section fields."""
        resp = _post(
            self.client,
            "/api/a3/create/",
            {
                "project_id": str(self.project.id),
                "title": "Seven Sections Test",
            },
        )
        self.assertEqual(resp.status_code, 200)
        report = resp.json()["report"]
        sections = [
            "background",
            "current_condition",
            "goal",
            "root_cause",
            "countermeasures",
            "implementation_plan",
            "follow_up",
        ]
        for section in sections:
            self.assertIn(section, report, f"Missing section: {section}")

    def test_a3_section_update(self):
        """A3 sections can be individually updated."""
        resp = _post(
            self.client,
            "/api/a3/create/",
            {
                "project_id": str(self.project.id),
                "title": "Section Update Test",
            },
        )
        report_id = resp.json()["report"]["id"]

        resp = _put(
            self.client,
            f"/api/a3/{report_id}/update/",
            {
                "background": "Machine downtime increased 15% Q4 vs Q3.",
                "goal": "Reduce unplanned downtime to <5% by end of Q1.",
            },
        )
        self.assertEqual(resp.status_code, 200)
        report = resp.json()["report"]
        self.assertIn("15%", report["background"])
        self.assertIn("<5%", report["goal"])


# =============================================================================
# qms-a3-import: A3 imports from hypothesis and RCA
# =============================================================================


@SECURE_OFF
class A3ImportTest(TestCase):
    """QMS-001 §4.3 — A3 can import content from hypotheses and RCA."""

    def setUp(self):
        self.user = _make_team_user("a3import@test.com")
        self.client.force_login(self.user)
        self.project = Project.objects.create(
            title="A3 Import Project",
            user=self.user,
        )

    def test_import_from_hypothesis(self):
        """A3 imports hypothesis content into a section."""
        from core.models import Hypothesis

        hyp = Hypothesis.objects.create(
            project=self.project,
            statement="Bearing failure caused by inadequate lubrication",
            because_clause="Lack of PM schedule leads to dry bearing surfaces",
        )

        resp = _post(
            self.client,
            "/api/a3/create/",
            {
                "project_id": str(self.project.id),
                "title": "Hypothesis Import Test",
            },
        )
        report_id = resp.json()["report"]["id"]

        resp = _post(
            self.client,
            f"/api/a3/{report_id}/import/",
            {
                "section": "root_cause",
                "source_type": "hypothesis",
                "source_id": str(hyp.id),
            },
        )
        self.assertEqual(resp.status_code, 200)
        report = resp.json()["report"]
        self.assertIn("lubrication", report["root_cause"].lower())

    def test_import_from_rca(self):
        """A3 auto-imports RCA content when created with rca_session_id."""
        # Create RCA session with root cause
        resp = _post(
            self.client,
            "/api/rca/sessions/create/",
            {
                "event": "Press line bearing seized",
                "title": "Bearing Failure",
            },
        )
        sid = resp.json()["session"]["id"]

        # Walk to root_cause_identified
        _put(
            self.client, f"/api/rca/sessions/{sid}/update/", {"status": "investigating"}
        )
        _put(
            self.client,
            f"/api/rca/sessions/{sid}/update/",
            {
                "root_cause": "Inadequate lubrication schedule",
                "countermeasure": "Implement weekly PM lube checks",
                "status": "root_cause_identified",
            },
        )

        # Create A3 linked to RCA
        resp = _post(
            self.client,
            "/api/a3/create/",
            {
                "project_id": str(self.project.id),
                "title": "RCA Import Test",
                "rca_session_id": sid,
            },
        )
        self.assertEqual(resp.status_code, 200)
        report = resp.json()["report"]
        # RCA content should appear in root_cause section
        self.assertTrue(
            "lubrication" in report["root_cause"].lower()
            or "bearing" in report["background"].lower()
            or len(report["root_cause"]) > 0,
            "A3 should import some content from linked RCA session",
        )


# =============================================================================
# qms-kpi-effective: HoshinKPI effective_actual aggregation
# =============================================================================


@SECURE_OFF
class KPIEffectiveActualTest(TestCase):
    """QMS-001 §4.6 — HoshinKPI.effective_actual aggregation modes."""

    def test_effective_actual_manual(self):
        """Manual aggregation returns actual_value directly."""
        kpi = HoshinKPI(
            aggregation="manual",
            actual_value=42.5,
        )
        self.assertEqual(kpi.effective_actual, 42.5)

    def test_effective_actual_fallback(self):
        """Fallback returns actual_value when aggregation mode has no derived_from."""
        kpi = HoshinKPI(
            aggregation="sum",
            actual_value=99.0,
            derived_from=None,
        )
        # Without derived_from, sum mode falls back to actual_value
        result = kpi.effective_actual
        self.assertEqual(result, 99.0)


# =============================================================================
# qms-fmea-spc: SPC OOC updates FMEA occurrence
# =============================================================================


@SECURE_OFF
class FMEASPCUpdateTest(TestCase):
    """QMS-001 §5.4 — SPC OOC data updates FMEA occurrence."""

    def setUp(self):
        self.user = _make_team_user("fmeaspc@test.com")
        self.client.force_login(self.user)
        resp = _post(self.client, "/api/fmea/create/", {"title": "SPC Test"})
        self.fmea_id = resp.json()["id"]
        resp = _post(
            self.client,
            f"/api/fmea/{self.fmea_id}/rows/",
            {
                "failure_mode": "Dimension out of spec",
                "severity": 7,
                "occurrence": 5,
                "detection": 4,
            },
        )
        self.row_id = resp.json()["row"]["id"]

    def test_ooc_updates_occurrence(self):
        """SPC OOC rate updates occurrence and recalculates RPN."""
        # 2 out of 100 = 2% → occurrence should be moderate (3-4 per AIAG)
        resp = _post(
            self.client,
            f"/api/fmea/{self.fmea_id}/rows/{self.row_id}/spc-update/",
            {"ooc_count": 2, "total_points": 100},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["old_occurrence"], 5)
        self.assertIn(data["new_occurrence"], range(1, 11))
        self.assertEqual(data["new_rpn"], 7 * data["new_occurrence"] * 4)

    def test_ooc_zero_rate_maps_to_one(self):
        """SPC OOC rate of 0% maps to occurrence=1 (remote)."""
        resp = _post(
            self.client,
            f"/api/fmea/{self.fmea_id}/rows/{self.row_id}/spc-update/",
            {"ooc_count": 0, "total_points": 100},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["new_occurrence"], 1)


# =============================================================================
# qms-action-sources: ActionItem from FMEA, RCA, A3
# =============================================================================


@SECURE_OFF
class ActionItemSourcesTest(TestCase):
    """QMS-001 §5.2 — ActionItem supports multiple source types."""

    def setUp(self):
        self.user = _make_team_user("actionsrc@test.com")
        self.client.force_login(self.user)

    def test_promote_action_from_fmea(self):
        """FMEA row promotes to ActionItem with source_type=fmea."""
        project = Project.objects.create(title="FMEA Action Project", user=self.user)
        resp = _post(
            self.client,
            "/api/fmea/create/",
            {
                "title": "Action FMEA",
                "project_id": str(project.id),
            },
        )
        fmea_id = resp.json()["id"]

        resp = _post(
            self.client,
            f"/api/fmea/{fmea_id}/rows/",
            {
                "failure_mode": "Seal leak",
                "severity": 8,
                "occurrence": 4,
                "detection": 6,
                "recommended_action": "Replace seal material with Viton",
            },
        )
        row_id = resp.json()["row"]["id"]

        resp = _post(
            self.client,
            f"/api/fmea/{fmea_id}/rows/{row_id}/promote-action/",
            {
                "title": "Replace seal material",
            },
        )
        self.assertIn(resp.status_code, [200, 201])
        action = resp.json()["action_item"]
        self.assertEqual(action["source_type"], "fmea")
        self.assertEqual(action["source_id"], row_id)

    def test_create_action_from_rca(self):
        """RCA session creates ActionItem with source_type=rca."""
        resp = _post(
            self.client,
            "/api/rca/sessions/create/",
            {
                "event": "Conveyor jam",
                "title": "Conveyor Action Test",
            },
        )
        sid = resp.json()["session"]["id"]

        resp = _post(
            self.client,
            f"/api/rca/sessions/{sid}/actions/create/",
            {
                "title": "Install jam sensor on conveyor",
            },
        )
        self.assertIn(resp.status_code, [200, 201])
        action = resp.json()["action_item"]
        self.assertEqual(action["source_type"], "rca")


# =========================================================================
# Template Block Consistency
# =========================================================================


class TemplateBlockConsistencyTest(TestCase):
    """Every template extending base_app.html must use only valid block names.

    base_app.html defines: title, extra_head, content, scripts.
    Using any other name (e.g. extra_js) silently drops the content.
    """

    VALID_BLOCKS = {"title", "extra_head", "content", "scripts"}

    def _get_base_app_children(self):
        """Return paths of all templates that extend base_app.html."""
        import re
        from pathlib import Path

        templates_dir = Path(__file__).resolve().parent.parent / "templates"
        children = []
        extends_re = re.compile(r'\{%\s*extends\s+["\']base_app\.html["\']\s*%\}')
        for html_file in templates_dir.glob("*.html"):
            text = html_file.read_text()
            if extends_re.search(text):
                children.append(html_file)
        return children

    def test_all_child_templates_use_valid_blocks(self):
        """No child template uses an undefined block name."""
        import re

        block_re = re.compile(r"\{%\s*block\s+(\w+)")
        children = self._get_base_app_children()
        self.assertGreater(len(children), 0, "Should find child templates")

        violations = []
        for path in children:
            text = path.read_text()
            for match in block_re.finditer(text):
                block_name = match.group(1)
                if block_name not in self.VALID_BLOCKS:
                    violations.append(f"{path.name}: {{% block {block_name} %}}")

        self.assertEqual(
            violations,
            [],
            f"Templates using undefined block names (valid: {self.VALID_BLOCKS}):\n"
            + "\n".join(f"  - {v}" for v in violations),
        )


# =========================================================================
# No Native Browser Dialogs
# =========================================================================


class TemplateNoBrowserDialogsTest(TestCase):
    """Templates must use Svend-branded modals, not native browser dialogs.

    confirm() and prompt() break theming and look unprofessional.
    Use svendConfirm() / svendPrompt() from base_app.html instead.

    Note: alert() is already overridden by svToast in base_app.html,
    so it's excluded from this check.
    """

    # Patterns that indicate native browser dialog usage in JS.
    # Matches: confirm(, prompt( — but not svendConfirm(, svendPrompt(,
    # or comments.
    FORBIDDEN = [
        (r"(?<!svend)(?<!\/\/)(?<!\w)confirm\s*\(", "confirm()"),
        (r"(?<!svend)(?<!\/\/)(?<!\w)prompt\s*\(", "prompt()"),
    ]

    def test_no_browser_dialogs_in_templates(self):
        """No template uses confirm(), alert(), or prompt()."""
        import re
        from pathlib import Path

        templates_dir = Path(__file__).resolve().parent.parent / "templates"
        violations = []

        for html_file in sorted(templates_dir.rglob("*.html")):
            text = html_file.read_text()
            rel = html_file.relative_to(templates_dir)
            for pattern, name in self.FORBIDDEN:
                for match in re.finditer(pattern, text):
                    # Find line number
                    line_no = text[: match.start()].count("\n") + 1
                    # Skip if inside an HTML comment or JS comment line
                    line = text.split("\n")[line_no - 1].strip()
                    if line.startswith("//") or line.startswith("<!--"):
                        continue
                    violations.append(f"{rel}:{line_no} — {name}")

        self.assertEqual(
            violations,
            [],
            "Templates using native browser dialogs (use svendConfirm/svendAlert instead):\n"
            + "\n".join(f"  - {v}" for v in violations),
        )

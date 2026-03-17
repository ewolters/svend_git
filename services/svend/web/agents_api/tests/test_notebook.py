"""Behavioral tests for Notebook API — NB-001 / TST-001 §10.6.

Tests mirror real user workflows: create a charter, open a notebook,
run trials, add pages, conclude with reflection, carry forward learning.

Standard: NB-001 (Notebook & Trial)
Compliance: TST-001 §7.1 (API pattern), §10.6 (no existence-only tests)
<!-- test: agents_api.tests.test_notebook -->
"""

import json

from django.test import Client, TestCase, override_settings

from accounts.models import Tier, User
from core.models import (
    Notebook,
    NotebookPage,
    Project,
    Trial,
    Yokoten,
    YokotenAdoption,
)

SECURE_OFF = override_settings(SECURE_SSL_REDIRECT=False)


def _make_user(email, tier=Tier.PRO, password="testpass123!"):
    username = email.split("@")[0]
    user = User.objects.create_user(username=username, email=email, password=password)
    user.tier = tier
    user.is_email_verified = True
    user.save(update_fields=["tier", "is_email_verified"])
    return user


def _authed_client(user):
    client = Client()
    client.force_login(user)
    return client


def _post_json(client, url, data):
    return client.post(url, json.dumps(data), content_type="application/json")


def _patch_json(client, url, data):
    return client.patch(url, json.dumps(data), content_type="application/json")


def _make_charter(user, title="Line 3 Scrap Reduction", **kwargs):
    defaults = {
        "user": user,
        "title": title,
        "goal_metric": "scrap_rate",
        "goal_baseline": "4.7",
        "goal_target": "2.0",
        "goal_unit": "%",
    }
    defaults.update(kwargs)
    return Project.objects.create(**defaults)


# ===========================================================================
# Notebook CRUD
# ===========================================================================


@SECURE_OFF
class NotebookCreateTest(TestCase):
    """User creates a notebook linked to a charter."""

    def setUp(self):
        self.user = _make_user("nb@test.com")
        self.client = _authed_client(self.user)
        self.charter = _make_charter(self.user)

    def test_create_notebook(self):
        """POST /api/notebooks/ creates a notebook on a charter."""
        res = _post_json(
            self.client,
            "/api/notebooks/",
            {
                "project_id": str(self.charter.id),
                "title": "Scrap Reduction Campaign",
                "baseline_metric": "scrap_rate",
                "baseline_value": 4.7,
                "baseline_unit": "%",
            },
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["title"], "Scrap Reduction Campaign")
        self.assertEqual(data["status"], "open")
        self.assertEqual(data["baseline_value"], 4.7)
        self.assertIsNotNone(data["id"])

        # Verify DB state
        nb = Notebook.objects.get(id=data["id"])
        self.assertEqual(nb.project_id, self.charter.id)
        self.assertEqual(nb.owner, self.user)

    def test_create_requires_title(self):
        res = _post_json(
            self.client,
            "/api/notebooks/",
            {"project_id": str(self.charter.id), "title": ""},
        )
        self.assertEqual(res.status_code, 400)

    def test_create_requires_project(self):
        res = _post_json(
            self.client,
            "/api/notebooks/",
            {"title": "No project"},
        )
        self.assertEqual(res.status_code, 400)

    def test_list_notebooks(self):
        Notebook.objects.create(project=self.charter, title="NB1", owner=self.user)
        Notebook.objects.create(project=self.charter, title="NB2", owner=self.user)
        res = self.client.get("/api/notebooks/")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.json()["notebooks"]), 2)

    def test_unauthenticated_blocked(self):
        self.client = Client()  # unauthenticated
        res = self.client.get("/api/notebooks/")
        self.assertIn(res.status_code, [401, 403])


@SECURE_OFF
class NotebookDetailTest(TestCase):
    """User views and updates a notebook."""

    def setUp(self):
        self.user = _make_user("detail@test.com")
        self.client = _authed_client(self.user)
        self.charter = _make_charter(self.user)
        self.nb = Notebook.objects.create(
            project=self.charter,
            title="Test NB",
            owner=self.user,
            baseline_value=4.7,
            baseline_metric="scrap_rate",
        )

    def test_get_detail_includes_trials_and_pages(self):
        Trial(notebook=self.nb, title="T1", created_by=self.user).save()
        NotebookPage.objects.create(
            notebook=self.nb,
            page_type="note",
            title="A note",
            created_by=self.user,
        )
        res = self.client.get(f"/api/notebooks/{self.nb.id}/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(len(data["trials"]), 1)
        self.assertEqual(len(data["pages"]), 1)
        self.assertIsNone(data["hansei_kai"])

    def test_patch_baseline(self):
        res = _patch_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/",
            {"baseline_value": 5.0, "title": "Updated Title"},
        )
        self.assertEqual(res.status_code, 200)
        self.nb.refresh_from_db()
        self.assertEqual(self.nb.baseline_value, 5.0)
        self.assertEqual(self.nb.title, "Updated Title")

    def test_delete_notebook(self):
        res = self.client.delete(f"/api/notebooks/{self.nb.id}/")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(Notebook.objects.filter(id=self.nb.id).exists())


# ===========================================================================
# Trial lifecycle — the core loop
# ===========================================================================


@SECURE_OFF
class TrialLifecycleTest(TestCase):
    """
    Full trial lifecycle: create → update after → complete with verdict.

    NB-001 §2.2: Before/after is the core primitive. Delta computed, not described.
    <!-- test: agents_api.tests.test_notebook.TrialLifecycleTest -->
    """

    def setUp(self):
        self.user = _make_user("trial@test.com")
        self.client = _authed_client(self.user)
        self.charter = _make_charter(self.user)
        self.nb = Notebook.objects.create(
            project=self.charter,
            title="Trial NB",
            owner=self.user,
            baseline_value=4.7,
            baseline_metric="scrap_rate",
            baseline_unit="%",
        )

    def test_create_trial_auto_activates_notebook(self):
        """Creating first trial transitions notebook from open → active."""
        self.assertEqual(self.nb.status, "open")
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/trials/",
            {"title": "Adjust feed rate", "before_value": 4.7},
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["sequence"], 1)
        self.assertEqual(data["verdict"], "pending")

        self.nb.refresh_from_db()
        self.assertEqual(self.nb.status, "active")
        self.assertEqual(str(self.nb.active_trial_id), data["id"])

    def test_trial_sequence_auto_increments(self):
        """Trials auto-number within a notebook."""
        Trial(notebook=self.nb, title="T1", created_by=self.user).save()
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/trials/",
            {"title": "T2"},
        )
        self.assertEqual(res.json()["sequence"], 2)

    def test_update_after_value_and_complete(self):
        """Set after value via PATCH, then complete with verdict."""
        trial = Trial(
            notebook=self.nb,
            title="Feed rate",
            before_value=4.7,
            created_by=self.user,
        )
        trial.save()

        # Update after value
        res = _patch_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/trials/{trial.id}/",
            {"after_value": 3.1, "after_date": "2026-02-15"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertAlmostEqual(res.json()["delta"], -1.6, places=1)

        # Complete with verdict
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/trials/{trial.id}/complete/",
            {"verdict": "improved", "adopted": True},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["verdict"], "improved")
        self.assertTrue(data["adopted"])
        self.assertIsNotNone(data["completed_at"])

        # Notebook current_value updated from adopted trial
        self.nb.refresh_from_db()
        self.assertAlmostEqual(self.nb.current_value, 3.1, places=1)

    def test_complete_no_effect_does_not_update_current(self):
        """Non-adopted trials don't update notebook current_value."""
        trial = Trial(
            notebook=self.nb,
            title="Supplier change",
            before_value=3.1,
            after_value=3.3,
            created_by=self.user,
        )
        trial.save()
        _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/trials/{trial.id}/complete/",
            {"verdict": "no_effect", "adopted": False},
        )
        self.nb.refresh_from_db()
        self.assertIsNone(self.nb.current_value)

    def test_invalid_verdict_rejected(self):
        trial = Trial(notebook=self.nb, title="T", created_by=self.user)
        trial.save()
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/trials/{trial.id}/complete/",
            {"verdict": "amazing"},
        )
        self.assertEqual(res.status_code, 400)


# ===========================================================================
# Pages — frozen calculator/simulator snapshots
# ===========================================================================


@SECURE_OFF
class NotebookPageTest(TestCase):
    """
    Pages are frozen snapshots of calculator/simulator output.

    NB-001 §2.3: inputs and outputs are point-in-time records.
    <!-- test: agents_api.tests.test_notebook.NotebookPageTest -->
    """

    def setUp(self):
        self.user = _make_user("page@test.com")
        self.client = _authed_client(self.user)
        self.charter = _make_charter(self.user)
        self.nb = Notebook.objects.create(
            project=self.charter,
            title="Page NB",
            owner=self.user,
        )

    def test_create_oee_calculator_page(self):
        """OEE calculator output persisted as frozen page."""
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/pages/",
            {
                "page_type": "calculator",
                "title": "OEE Calculation — Jan 22",
                "source_tool": "oee",
                "inputs": {"availability": 87, "performance": 91, "quality": 95.3},
                "outputs": {"oee": 75.5},
            },
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["source_tool"], "oee")
        self.assertEqual(data["inputs"]["availability"], 87)
        self.assertEqual(data["outputs"]["oee"], 75.5)

        # Page auto-activates notebook
        self.nb.refresh_from_db()
        self.assertEqual(self.nb.status, "active")

    def test_create_kanban_page(self):
        """Kanban card generator output persisted."""
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/pages/",
            {
                "page_type": "calculator",
                "title": "768 SKU Kanban Layout",
                "source_tool": "kanban",
                "inputs": {"sku_count": 768, "demand_rate": "daily"},
                "outputs": {"cards": 768, "bins": 384},
                "rendered_html": "<div>768 cards generated</div>",
            },
        )
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["rendered_html"], "<div>768 cards generated</div>")

    def test_page_linked_to_trial(self):
        """Page can be linked to a trial as before/after/supporting evidence."""
        trial = Trial(notebook=self.nb, title="T1", created_by=self.user)
        trial.save()
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/pages/",
            {
                "page_type": "analysis",
                "title": "Cpk Before",
                "source_tool": "cpk",
                "inputs": {"data": [1, 2, 3], "lsl": 0, "usl": 5},
                "outputs": {"cpk": 0.82},
                "trial_id": str(trial.id),
                "trial_role": "before",
            },
        )
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["trial_role"], "before")

    def test_simulator_page(self):
        """Plant simulator output persisted."""
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/pages/",
            {
                "page_type": "simulator",
                "title": "Line layout v2 — DES",
                "source_tool": "plantsim",
                "inputs": {"stations": 5, "wip_limit": 20},
                "outputs": {"throughput": 142, "utilization": 0.87},
                "narrative": "Predicted throughput +12% with jig station",
            },
        )
        self.assertEqual(res.status_code, 201)

    def test_whiteboard_page(self):
        """Whiteboard snapshot saved as notebook page with SVG rendered_html."""
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/pages/",
            {
                "page_type": "note",
                "title": "Whiteboard — Causal Diagram",
                "source_tool": "whiteboard",
                "inputs": {
                    "room_code": "ABC123",
                    "elements_count": 5,
                    "connections_count": 3,
                    "element_types": {"postit": 3, "text": 2},
                },
                "outputs": {
                    "board_name": "Causal Diagram",
                    "causal_links": 2,
                },
                "rendered_html": '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300"><rect/></svg>',
                "narrative": "Whiteboard snapshot: 5 elements, 3 connections, 2 causal links.",
            },
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["source_tool"], "whiteboard")
        self.assertIn("svg", data["rendered_html"])
        self.assertEqual(data["inputs"]["elements_count"], 5)
        self.assertEqual(data["outputs"]["causal_links"], 2)

    def test_ishikawa_page(self):
        """Ishikawa fishbone diagram saved as notebook page."""
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/pages/",
            {
                "page_type": "note",
                "title": "Ishikawa — High Scrap Rate",
                "source_tool": "ishikawa",
                "inputs": {
                    "effect": "High scrap rate on Line 3",
                    "categories": ["Man", "Machine", "Method", "Material", "Measurement", "Mother Nature"],
                    "total_causes": 12,
                },
                "outputs": {
                    "root_causes": ["Feed rate too high", "No operator refresher training"],
                    "status": "complete",
                    "branch_count": 6,
                },
                "narrative": 'Fishbone analysis for "High scrap rate on Line 3": 6 categories, 12 causes. Root causes: Feed rate too high; No operator refresher training.',
            },
        )
        self.assertEqual(res.status_code, 201)
        data = res.json()
        self.assertEqual(data["source_tool"], "ishikawa")
        self.assertEqual(data["inputs"]["effect"], "High scrap rate on Line 3")
        self.assertEqual(len(data["outputs"]["root_causes"]), 2)

    def test_page_requires_title_and_type(self):
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/pages/",
            {"page_type": "calculator"},
        )
        self.assertEqual(res.status_code, 400)


# ===========================================================================
# Conclude + Hansei Kai + Yokoten
# ===========================================================================


@SECURE_OFF
class ConcludeAndReflectTest(TestCase):
    """
    Full conclusion workflow: conclude notebook → Hansei Kai → Yokoten.

    NB-001 §2.6-2.7: Reflection on conclusion, lateral learning transfer.
    <!-- test: agents_api.tests.test_notebook.ConcludeAndReflectTest -->
    """

    def setUp(self):
        self.user = _make_user("conclude@test.com")
        self.client = _authed_client(self.user)
        self.charter = _make_charter(self.user)
        self.nb = Notebook.objects.create(
            project=self.charter,
            title="Conclude NB",
            owner=self.user,
        )
        # Must be active to conclude
        Trial(notebook=self.nb, title="T1", created_by=self.user).save()
        self.nb.refresh_from_db()
        self.assertEqual(self.nb.status, "active")

    def test_conclude_with_hansei_kai(self):
        """Concluding creates Hansei Kai and transitions to concluded."""
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/conclude/",
            {
                "what_went_well": "Feed rate change worked immediately",
                "what_didnt": "Wasted 3 weeks on supplier change",
                "what_next": "Pilot material changes on one line first",
                "key_learning": "Mechanical alignment fixes outperform material changes",
                "carry_forward": False,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "concluded")
        self.assertIsNotNone(data["concluded_at"])
        self.assertIsNotNone(data["hansei_kai"])
        self.assertEqual(
            data["hansei_kai"]["key_learning"],
            "Mechanical alignment fixes outperform material changes",
        )

        # No yokoten since carry_forward=False
        self.assertFalse(Yokoten.objects.filter(source_notebook=self.nb).exists())

    def test_conclude_with_yokoten(self):
        """carry_forward=True auto-creates Yokoten from key_learning."""
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/conclude/",
            {
                "what_went_well": "Good",
                "what_didnt": "Bad",
                "what_next": "Better",
                "key_learning": "Alignment jigs reduce scrap on high-speed lines",
                "carry_forward": True,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("yokoten", data)
        self.assertEqual(
            data["yokoten"]["learning"],
            "Alignment jigs reduce scrap on high-speed lines",
        )

        # Verify in DB
        yokoten = Yokoten.objects.get(source_notebook=self.nb)
        self.assertEqual(yokoten.learning, "Alignment jigs reduce scrap on high-speed lines")

    def test_conclude_requires_active_status(self):
        """Cannot conclude an open notebook (no trials)."""
        nb2 = Notebook.objects.create(
            project=self.charter,
            title="Empty NB",
            owner=self.user,
        )
        res = _post_json(
            self.client,
            f"/api/notebooks/{nb2.id}/conclude/",
            {
                "what_went_well": "x",
                "what_didnt": "y",
                "what_next": "z",
                "key_learning": "w",
            },
        )
        self.assertEqual(res.status_code, 400)

    def test_conclude_requires_all_reflection_fields(self):
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/conclude/",
            {"what_went_well": "Good", "key_learning": "Something"},
        )
        self.assertEqual(res.status_code, 400)


# ===========================================================================
# Yokoten adoption
# ===========================================================================


@SECURE_OFF
class YokotenAdoptionTest(TestCase):
    """
    Yokoten adoption workflow: see learning → adopt into own notebook.

    NB-001 §2.7: Learning compounds across the organization.
    <!-- test: agents_api.tests.test_notebook.YokotenAdoptionTest -->
    """

    def setUp(self):
        self.user = _make_user("yokoten@test.com")
        self.client = _authed_client(self.user)
        self.charter = _make_charter(self.user)

        # Source notebook with yokoten
        self.source_nb = Notebook.objects.create(
            project=self.charter,
            title="Source NB",
            owner=self.user,
        )
        self.yokoten = Yokoten.objects.create(
            source_notebook=self.source_nb,
            learning="Alignment jigs reduce scrap on high-speed lines",
            context="Heidelberg XL 106, feed rate 10mm/s",
            applicable_to=["scrap_reduction", "high_speed_press"],
            created_by=self.user,
        )

        # Target notebook
        self.target_nb = Notebook.objects.create(
            project=self.charter,
            title="Target NB",
            owner=self.user,
        )

    def test_list_yokoten(self):
        res = self.client.get("/api/notebooks/yokoten/")
        self.assertEqual(res.status_code, 200)
        yokoten_list = res.json()["yokoten"]
        self.assertEqual(len(yokoten_list), 1)
        self.assertEqual(yokoten_list[0]["learning"], self.yokoten.learning)
        self.assertEqual(yokoten_list[0]["adoption_count"], 0)

    def test_adopt_yokoten(self):
        """Adopting yokoten creates a tracked adoption record."""
        res = _post_json(
            self.client,
            f"/api/notebooks/yokoten/{self.yokoten.id}/adopt/",
            {"target_notebook_id": str(self.target_nb.id)},
        )
        self.assertEqual(res.status_code, 201)
        self.assertTrue(res.json()["created"])

        # Verify adoption in DB
        adoption = YokotenAdoption.objects.get(
            yokoten=self.yokoten,
            target_notebook=self.target_nb,
        )
        self.assertEqual(adoption.adopted_by, self.user)

    def test_adopt_yokoten_idempotent(self):
        """Adopting same yokoten twice doesn't duplicate."""
        _post_json(
            self.client,
            f"/api/notebooks/yokoten/{self.yokoten.id}/adopt/",
            {"target_notebook_id": str(self.target_nb.id)},
        )
        res = _post_json(
            self.client,
            f"/api/notebooks/yokoten/{self.yokoten.id}/adopt/",
            {"target_notebook_id": str(self.target_nb.id)},
        )
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["created"])
        self.assertEqual(YokotenAdoption.objects.count(), 1)


# ===========================================================================
# Verdict narrative auto-generation — NB-001 §2.2.2
# ===========================================================================


@SECURE_OFF
class VerdictNarrativeTest(TestCase):
    """
    Verdict narrative auto-generation from trial data and linked page statistics.

    NB-001 §2.2.2: System computes significance and generates narrative.
    User can accept, modify, or override.
    <!-- test: agents_api.tests.test_notebook.VerdictNarrativeTest -->
    """

    def setUp(self):
        self.user = _make_user("narrative@test.com")
        self.client = _authed_client(self.user)
        self.charter = _make_charter(self.user)  # goal: 4.7% → 2.0% (reducing)
        self.nb = Notebook.objects.create(
            project=self.charter,
            title="Narrative NB",
            owner=self.user,
            baseline_value=4.7,
            baseline_metric="scrap_rate",
            baseline_unit="%",
        )

    def test_auto_generates_from_values(self):
        """Narrative auto-generated when before/after values exist and user provides none."""
        trial = Trial(
            notebook=self.nb,
            title="Feed rate change",
            before_value=4.7,
            after_value=3.1,
            created_by=self.user,
        )
        trial.save()

        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/trials/{trial.id}/complete/",
            {"verdict": "improved", "adopted": True},
        )
        self.assertEqual(res.status_code, 200)
        narrative = res.json()["verdict_narrative"]
        self.assertTrue(len(narrative) > 0, "Expected auto-generated narrative")
        # Should mention the delta
        self.assertIn("1.6", narrative)
        # Should mention before/after values
        self.assertIn("4.7", narrative)
        self.assertIn("3.1", narrative)

    def test_user_narrative_takes_priority(self):
        """User-provided narrative is not overwritten by auto-generation."""
        trial = Trial(
            notebook=self.nb,
            title="Manual narrative",
            before_value=4.7,
            after_value=3.1,
            created_by=self.user,
        )
        trial.save()

        user_narrative = "My custom narrative — the change worked great."
        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/trials/{trial.id}/complete/",
            {
                "verdict": "improved",
                "adopted": True,
                "verdict_narrative": user_narrative,
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["verdict_narrative"], user_narrative)

    def test_includes_stats_from_after_page(self):
        """Narrative includes p-value and effect size from linked after page."""
        trial = Trial(
            notebook=self.nb,
            title="With stats",
            before_value=4.7,
            after_value=3.1,
            created_by=self.user,
        )
        trial.save()

        NotebookPage.objects.create(
            notebook=self.nb,
            page_type="analysis",
            title="After analysis",
            source_tool="dsw",
            inputs={"test": "t_test"},
            outputs={
                "p_value": 0.003,
                "statistics": {
                    "cohens_d": 0.84,
                    "confidence_interval": [-2.1, -0.9],
                },
            },
            trial=trial,
            trial_role="after",
            created_by=self.user,
        )

        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/trials/{trial.id}/complete/",
            {"verdict": "improved", "adopted": True},
        )
        self.assertEqual(res.status_code, 200)
        narrative = res.json()["verdict_narrative"]
        self.assertIn("0.003", narrative)
        self.assertIn("0.84", narrative)  # Cohen's d value
        self.assertIn("CI", narrative)

    def test_includes_capability_indices(self):
        """Narrative includes Cpk, Cp, sigma level from SPC capability page."""
        trial = Trial(
            notebook=self.nb,
            title="Capability trial",
            before_value=4.7,
            after_value=3.1,
            created_by=self.user,
        )
        trial.save()

        # Before page with old Cpk
        NotebookPage.objects.create(
            notebook=self.nb,
            page_type="analysis",
            title="Before capability",
            source_tool="spc",
            outputs={"statistics": {"cpk": 0.82, "cp": 0.95}},
            trial=trial,
            trial_role="before",
            created_by=self.user,
        )

        # After page with improved Cpk
        NotebookPage.objects.create(
            notebook=self.nb,
            page_type="analysis",
            title="After capability",
            source_tool="spc",
            outputs={
                "statistics": {
                    "cpk": 1.45,
                    "cp": 1.52,
                    "ppk": 1.38,
                    "pp": 1.50,
                    "sigma_level": 4.35,
                    "yield_pct": 99.9866,
                    "n": 50,
                },
            },
            trial=trial,
            trial_role="after",
            created_by=self.user,
        )

        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/trials/{trial.id}/complete/",
            {"verdict": "improved", "adopted": True},
        )
        self.assertEqual(res.status_code, 200)
        narrative = res.json()["verdict_narrative"]
        self.assertIn("Cpk=1.45", narrative)
        self.assertIn("capable", narrative)
        self.assertIn("Cp=1.52", narrative)
        self.assertIn("Ppk=1.38", narrative)
        self.assertIn("Sigma level: 4.3", narrative)
        self.assertIn("from Cpk=0.82", narrative)  # before comparison
        self.assertIn("n=50", narrative)

    def test_includes_bayesian_evidence(self):
        """Narrative includes Bayes factor and posterior probability."""
        trial = Trial(
            notebook=self.nb,
            title="Bayesian trial",
            before_value=4.7,
            after_value=3.1,
            created_by=self.user,
        )
        trial.save()

        NotebookPage.objects.create(
            notebook=self.nb,
            page_type="analysis",
            title="Bayesian analysis",
            source_tool="dsw",
            outputs={
                "p_value": 0.01,
                "evidence_grade": "B+",
                "bayesian_shadow": {
                    "bf10": 42.5,
                    "posterior_probability": 0.977,
                },
                "statistics": {"cohens_d": 0.65},
            },
            trial=trial,
            trial_role="after",
            created_by=self.user,
        )

        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/trials/{trial.id}/complete/",
            {"verdict": "improved", "adopted": True},
        )
        self.assertEqual(res.status_code, 200)
        narrative = res.json()["verdict_narrative"]
        self.assertIn("BF10=42.5", narrative)
        self.assertIn("very strong", narrative)
        self.assertIn("97.7%", narrative)  # posterior
        self.assertIn("B+", narrative)  # evidence grade

    def test_no_narrative_without_values(self):
        """No narrative generated when before/after values are missing."""
        trial = Trial(
            notebook=self.nb,
            title="No values",
            created_by=self.user,
        )
        trial.save()

        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/trials/{trial.id}/complete/",
            {"verdict": "inconclusive"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["verdict_narrative"], "")

    def test_no_effect_narrative(self):
        """Narrative for no_effect verdict communicates correctly."""
        trial = Trial(
            notebook=self.nb,
            title="No effect trial",
            before_value=3.1,
            after_value=3.3,
            created_by=self.user,
        )
        trial.save()

        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/trials/{trial.id}/complete/",
            {"verdict": "no_effect", "adopted": False},
        )
        self.assertEqual(res.status_code, 200)
        narrative = res.json()["verdict_narrative"]
        self.assertIn("No meaningful change", narrative)

    def test_narrative_respects_goal_direction(self):
        """Narrative correctly identifies improvement direction from charter goal."""
        # Charter goal: 4.7 → 2.0 (decreasing = improving)
        # Delta: 4.7 → 3.1 = -1.6 (decrease = improvement)
        trial = Trial(
            notebook=self.nb,
            title="Goal direction",
            before_value=4.7,
            after_value=3.1,
            created_by=self.user,
        )
        trial.save()

        res = _post_json(
            self.client,
            f"/api/notebooks/{self.nb.id}/trials/{trial.id}/complete/",
            {"verdict": "improved", "adopted": True},
        )
        narrative = res.json()["verdict_narrative"]
        self.assertIn("toward the goal", narrative)

    def test_generate_verdict_narrative_direct(self):
        """Direct unit test of generate_verdict_narrative function."""
        from agents_api.notebook_views import generate_verdict_narrative

        trial = Trial(
            notebook=self.nb,
            title="Direct test",
            before_value=4.7,
            after_value=3.1,
            created_by=self.user,
        )
        trial.save()
        trial.verdict = "improved"
        narrative = generate_verdict_narrative(trial)
        self.assertIn("decreased", narrative)
        self.assertIn("1.6", narrative)
        self.assertIn("toward the goal", narrative)


# ===========================================================================
# Full workflow scenario
# ===========================================================================


@SECURE_OFF
class FullNotebookWorkflowTest(TestCase):
    """
    End-to-end scenario mirroring a real kaizen campaign.

    TST-001 §10.5: Scenario tests verify multi-step workflows.
    NB-001: Charter → Notebook → Baseline → Trials → Pages → Conclude → Yokoten.
    <!-- test: agents_api.tests.test_notebook.FullNotebookWorkflowTest -->
    """

    def setUp(self):
        self.user = _make_user("workflow@test.com")
        self.client = _authed_client(self.user)

    def test_full_kaizen_campaign(self):
        """Simulate a complete improvement campaign."""
        # 1. Create charter
        charter = _make_charter(self.user, title="Line 3 Scrap Reduction")

        # 2. Create notebook
        res = _post_json(
            self.client,
            "/api/notebooks/",
            {
                "project_id": str(charter.id),
                "title": "Scrap Reduction Q1",
                "baseline_metric": "scrap_rate",
                "baseline_value": 4.7,
                "baseline_unit": "%",
            },
        )
        self.assertEqual(res.status_code, 201)
        nb_id = res.json()["id"]

        # 3. Trial 1: Feed rate adjustment
        res = _post_json(
            self.client,
            f"/api/notebooks/{nb_id}/trials/",
            {"title": "Adjust feed rate 12→10 mm/s", "before_value": 4.7},
        )
        self.assertEqual(res.status_code, 201)
        t1_id = res.json()["id"]
        self.assertEqual(res.json()["sequence"], 1)

        # 4. Add OEE calculator page
        res = _post_json(
            self.client,
            f"/api/notebooks/{nb_id}/pages/",
            {
                "page_type": "calculator",
                "title": "OEE Baseline",
                "source_tool": "oee",
                "inputs": {"availability": 87, "performance": 91, "quality": 95.3},
                "outputs": {"oee": 75.5},
            },
        )
        self.assertEqual(res.status_code, 201)

        # 5. Update trial 1 with after value and complete
        _patch_json(
            self.client,
            f"/api/notebooks/{nb_id}/trials/{t1_id}/",
            {"after_value": 3.1},
        )
        res = _post_json(
            self.client,
            f"/api/notebooks/{nb_id}/trials/{t1_id}/complete/",
            {
                "verdict": "improved",
                "adopted": True,
                "verdict_narrative": "Scrap rate decreased 1.6pp (p=0.003)",
            },
        )
        self.assertEqual(res.json()["verdict"], "improved")

        # 6. Trial 2: Supplier change (no effect)
        res = _post_json(
            self.client,
            f"/api/notebooks/{nb_id}/trials/",
            {"title": "Change raw material supplier", "before_value": 3.1},
        )
        t2_id = res.json()["id"]
        _patch_json(
            self.client,
            f"/api/notebooks/{nb_id}/trials/{t2_id}/",
            {"after_value": 3.3},
        )
        _post_json(
            self.client,
            f"/api/notebooks/{nb_id}/trials/{t2_id}/complete/",
            {"verdict": "no_effect", "adopted": False},
        )

        # 7. Check notebook state
        res = self.client.get(f"/api/notebooks/{nb_id}/")
        data = res.json()
        self.assertEqual(data["status"], "active")
        self.assertAlmostEqual(data["current_value"], 3.1, places=1)
        self.assertEqual(len(data["trials"]), 2)
        self.assertEqual(len(data["pages"]), 1)

        # 8. Conclude with Hansei Kai + Yokoten
        res = _post_json(
            self.client,
            f"/api/notebooks/{nb_id}/conclude/",
            {
                "what_went_well": "Feed rate change worked. Data-driven approach built trust.",
                "what_didnt": "Wasted 3 weeks on supplier change with no pilot.",
                "what_next": "Always pilot material changes on one line first.",
                "key_learning": "Mechanical fixes outperform material changes for scrap on high-speed lines",
                "carry_forward": True,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "concluded")
        self.assertIsNotNone(data["yokoten"])

        # 9. Start new notebook, adopt yokoten
        res = _post_json(
            self.client,
            "/api/notebooks/",
            {
                "project_id": str(charter.id),
                "title": "Scrap Reduction Q2",
                "baseline_metric": "scrap_rate",
                "baseline_value": 3.1,
                "baseline_unit": "%",
            },
        )
        nb2_id = res.json()["id"]

        yokoten_id = data["yokoten"]["id"]
        res = _post_json(
            self.client,
            f"/api/notebooks/yokoten/{yokoten_id}/adopt/",
            {"target_notebook_id": nb2_id},
        )
        self.assertEqual(res.status_code, 201)
        self.assertTrue(res.json()["created"])

        # The learning has been carried forward.

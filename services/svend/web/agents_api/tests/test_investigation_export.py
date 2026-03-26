"""
Tests for export_investigation — CANON-002 §9.2.

All tests exercise real behavior per TST-001 §10.6.

<!-- test: agents_api.tests.test_investigation_export.ExportInvestigationTest -->
<!-- test: agents_api.tests.test_investigation_export.ConclusionPackageTest -->
<!-- test: agents_api.tests.test_investigation_export.CausalChainTest -->
"""

from django.test import TestCase

from agents_api.investigation_bridge import (
    HypothesisSpec,
    _build_conclusion_package,
    _trace_causal_chain,
    connect_tool,
    export_investigation,
)
from core.models import Evidence, Investigation, MeasurementSystem, Project


def _make_user(email="test@example.com"):
    from django.contrib.auth import get_user_model

    User = get_user_model()
    return User.objects.create_user(username=email, email=email, password="testpass123")


def _make_investigation(user, **kwargs):
    defaults = {"title": "Export test", "description": "Testing export", "owner": user}
    defaults.update(kwargs)
    return Investigation.objects.create(**defaults)


class ExportInvestigationTest(TestCase):
    """CANON-002 §9.2 — export creates evidence on target project."""

    def setUp(self):
        self.user = _make_user("export@test.com")
        self.project = Project.objects.create(title="Target Project", user=self.user)

    def _build_concluded_investigation(self):
        """Helper: create an investigation with hypotheses, conclude it."""
        inv = _make_investigation(self.user)
        inv.transition_to("active", self.user)

        # Add a hypothesis via bridge
        tool_output = MeasurementSystem.objects.create(
            name="Export Gage", system_type="variable", owner=self.user
        )
        spec = HypothesisSpec(description="Root cause: bearing wear", prior=0.7)
        connect_tool(
            investigation_id=str(inv.id),
            tool_output=tool_output,
            tool_type="rca",
            user=self.user,
            spec=spec,
        )
        inv.refresh_from_db()
        inv.transition_to("concluded", self.user)
        return inv

    def test_export_creates_evidence(self):
        """Export creates an Evidence record on target project."""
        inv = self._build_concluded_investigation()
        initial_count = Evidence.objects.filter(project=self.project).count()

        export_investigation(
            investigation_id=str(inv.id),
            target_project_id=str(self.project.id),
            user=self.user,
        )

        self.assertEqual(
            Evidence.objects.filter(project=self.project).count(), initial_count + 1
        )

    def test_export_transitions_to_exported(self):
        """Export transitions investigation to exported state."""
        inv = self._build_concluded_investigation()
        export_investigation(
            investigation_id=str(inv.id),
            target_project_id=str(self.project.id),
            user=self.user,
        )
        inv.refresh_from_db()
        self.assertEqual(inv.status, "exported")

    def test_export_freezes_package(self):
        """Export freezes conclusion package on investigation."""
        inv = self._build_concluded_investigation()
        export_investigation(
            investigation_id=str(inv.id),
            target_project_id=str(self.project.id),
            user=self.user,
        )
        inv.refresh_from_db()
        self.assertIsNotNone(inv.export_package)
        self.assertIn("top_hypothesis", inv.export_package)

    def test_export_sets_target_project(self):
        """Export sets exported_to_project FK."""
        inv = self._build_concluded_investigation()
        export_investigation(
            investigation_id=str(inv.id),
            target_project_id=str(self.project.id),
            user=self.user,
        )
        inv.refresh_from_db()
        self.assertEqual(inv.exported_to_project, self.project)

    def test_export_rejects_non_concluded(self):
        """Export raises ValueError if investigation is not concluded."""
        inv = _make_investigation(self.user, status="active")
        with self.assertRaises(ValueError) as ctx:
            export_investigation(
                investigation_id=str(inv.id),
                target_project_id=str(self.project.id),
                user=self.user,
            )
        self.assertIn("concluded", str(ctx.exception))

    def test_export_rejects_open(self):
        """Open investigation cannot be exported."""
        inv = _make_investigation(self.user)
        with self.assertRaises(ValueError):
            export_investigation(
                investigation_id=str(inv.id),
                target_project_id=str(self.project.id),
                user=self.user,
            )

    def test_export_returns_package(self):
        """Export returns the conclusion package dict."""
        inv = self._build_concluded_investigation()
        package = export_investigation(
            investigation_id=str(inv.id),
            target_project_id=str(self.project.id),
            user=self.user,
        )
        self.assertIsInstance(package, dict)
        self.assertIn("investigation_id", package)
        self.assertIn("top_hypothesis", package)
        self.assertIn("investigation_metadata", package)


class ConclusionPackageTest(TestCase):
    """CANON-002 §9.1 — conclusion package schema."""

    def setUp(self):
        self.user = _make_user("package@test.com")
        self.inv = _make_investigation(self.user, status="concluded")

    def test_package_schema_keys(self):
        """Package has all required top-level keys."""
        from agents_api.synara.synara import Synara

        synara = Synara()
        synara.create_hypothesis(description="Test H", prior=0.7)

        package = _build_conclusion_package(self.inv, synara, self.user)

        required_keys = [
            "investigation_id",
            "investigation_version",
            "status",
            "top_hypothesis",
            "competing_hypotheses",
            "evidence_summary",
            "unresolved_signals",
            "investigation_metadata",
        ]
        for key in required_keys:
            self.assertIn(key, package, f"Missing key: {key}")

    def test_top_hypothesis_identified(self):
        """Top hypothesis is the one with highest posterior."""
        from agents_api.synara.synara import Synara

        synara = Synara()
        synara.create_hypothesis(description="Low H", prior=0.3)
        synara.create_hypothesis(description="High H", prior=0.8)

        package = _build_conclusion_package(self.inv, synara, self.user)
        self.assertEqual(package["top_hypothesis"]["description"], "High H")

    def test_metadata_fields(self):
        """Investigation metadata includes expected fields."""
        from agents_api.synara.synara import Synara

        synara = Synara()
        synara.create_hypothesis(description="H1", prior=0.5)

        package = _build_conclusion_package(self.inv, synara, self.user)
        meta = package["investigation_metadata"]
        self.assertIn("tools_used", meta)
        self.assertIn("evidence_count", meta)
        self.assertIn("hypothesis_count", meta)
        self.assertIn("duration_days", meta)
        self.assertEqual(meta["hypothesis_count"], 1)

    def test_empty_graph_handles_gracefully(self):
        """Package handles empty Synara graph without error."""
        from agents_api.synara.synara import Synara

        synara = Synara()
        package = _build_conclusion_package(self.inv, synara, self.user)
        self.assertEqual(package["top_hypothesis"]["description"], "No hypotheses")
        self.assertEqual(package["top_hypothesis"]["posterior"], 0.0)


class CausalChainTest(TestCase):
    """CANON-002 §9.2 — causal chain tracing."""

    def test_linear_chain(self):
        """Linear chain A → B → C traces correctly."""
        from agents_api.synara.synara import Synara

        synara = Synara()
        h_a = synara.create_hypothesis(description="Root cause A", prior=0.6)
        h_b = synara.create_hypothesis(description="Intermediate B", prior=0.5)
        h_c = synara.create_hypothesis(description="Effect C", prior=0.4)
        synara.create_link(
            from_id=h_a.id, to_id=h_b.id, strength=0.8, mechanism="A causes B"
        )
        synara.create_link(
            from_id=h_b.id, to_id=h_c.id, strength=0.7, mechanism="B causes C"
        )

        chain = _trace_causal_chain(h_c.id, synara.graph.hypotheses, synara.graph.links)
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0]["description"], "Root cause A")
        self.assertEqual(chain[1]["description"], "Intermediate B")

    def test_no_incoming_links(self):
        """Hypothesis with no incoming links returns empty chain."""
        from agents_api.synara.synara import Synara

        synara = Synara()
        h = synara.create_hypothesis(description="Isolated", prior=0.5)
        chain = _trace_causal_chain(h.id, synara.graph.hypotheses, synara.graph.links)
        self.assertEqual(chain, [])

    def test_cycle_protection(self):
        """Chain walk stops on cycle (does not infinite loop)."""
        from agents_api.synara.synara import Synara

        synara = Synara()
        h_a = synara.create_hypothesis(description="A", prior=0.5)
        h_b = synara.create_hypothesis(description="B", prior=0.5)
        synara.create_link(from_id=h_a.id, to_id=h_b.id, strength=0.7, mechanism="A→B")
        synara.create_link(from_id=h_b.id, to_id=h_a.id, strength=0.7, mechanism="B→A")

        chain = _trace_causal_chain(h_b.id, synara.graph.hypotheses, synara.graph.links)
        # Should not infinite loop — max 20 depth
        self.assertLessEqual(len(chain), 20)

    def test_strongest_link_chosen(self):
        """When multiple incoming links, strongest is followed."""
        from agents_api.synara.synara import Synara

        synara = Synara()
        h_a = synara.create_hypothesis(description="Weak cause", prior=0.3)
        h_b = synara.create_hypothesis(description="Strong cause", prior=0.7)
        h_c = synara.create_hypothesis(description="Effect", prior=0.5)
        synara.create_link(from_id=h_a.id, to_id=h_c.id, strength=0.3, mechanism="weak")
        synara.create_link(
            from_id=h_b.id, to_id=h_c.id, strength=0.9, mechanism="strong"
        )

        chain = _trace_causal_chain(h_c.id, synara.graph.hypotheses, synara.graph.links)
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0]["description"], "Strong cause")

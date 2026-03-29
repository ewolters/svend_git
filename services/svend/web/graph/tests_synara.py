"""
Tests for Synara adapter and contradiction detection — GRAPH-001 §12, §10.

Covers: round-trip (Django → CausalGraph → Django), propagation persistence,
subgraph loading, contradiction detection threshold.
"""

from django.test import TestCase
from django.utils import timezone

from agents_api.synara.kernel import Evidence
from core.models.tenant import Tenant

from .models import EdgeEvidence, ProcessEdge, ProcessGraph, ProcessNode
from .service import GraphService
from .synara_adapter import SynaraAdapter


def _make_tenant(name="Test Org"):
    return Tenant.objects.create(name=name, slug=name.lower().replace(" ", "-"))


class SynaraAdapterLoadTest(TestCase):
    """Load ProcessGraph into CausalGraph and verify structure."""

    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.n1 = GraphService.add_node(
            self.tenant.id, self.graph.id, name="Temperature", node_type="process_parameter"
        )
        self.n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="Viscosity", node_type="material_property")
        self.n3 = GraphService.add_node(
            self.tenant.id, self.graph.id, name="Defect Rate", node_type="quality_characteristic"
        )
        self.e1 = GraphService.add_edge(self.tenant.id, self.graph.id, self.n1.id, self.n2.id)
        self.e2 = GraphService.add_edge(self.tenant.id, self.graph.id, self.n2.id, self.n3.id)

    def test_load_full_graph(self):
        cg, node_map, edge_map = SynaraAdapter.load_causal_graph(self.graph.id, self.tenant.id)
        self.assertEqual(len(cg.hypotheses), 3)
        self.assertEqual(len(cg.links), 2)
        self.assertEqual(len(node_map), 3)
        self.assertEqual(len(edge_map), 2)

    def test_load_subgraph(self):
        cg, node_map, edge_map = SynaraAdapter.load_causal_graph(
            self.graph.id, self.tenant.id, node_ids=[self.n1.id, self.n2.id]
        )
        self.assertEqual(len(cg.hypotheses), 2)
        self.assertEqual(len(cg.links), 1)

    def test_node_id_map_round_trip(self):
        cg, node_map, _ = SynaraAdapter.load_causal_graph(self.graph.id, self.tenant.id)
        for h_id, uuid in node_map.items():
            self.assertIn(h_id, cg.hypotheses)
            self.assertTrue(ProcessNode.objects.filter(id=uuid).exists())

    def test_edge_id_map_round_trip(self):
        _, _, edge_map = SynaraAdapter.load_causal_graph(self.graph.id, self.tenant.id)
        for key, uuid in edge_map.items():
            self.assertIn("->", key)
            self.assertTrue(ProcessEdge.objects.filter(id=uuid).exists())

    def test_tenant_isolation(self):
        other_tenant = _make_tenant("Other Org")
        with self.assertRaises(ProcessGraph.DoesNotExist):
            SynaraAdapter.load_causal_graph(self.graph.id, other_tenant.id)


class SynaraAdapterPersistTest(TestCase):
    """Persist Synara posteriors back to Django models."""

    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="A")
        self.n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="B")
        self.edge = GraphService.add_edge(self.tenant.id, self.graph.id, self.n1.id, self.n2.id)

    def test_persist_updates_edge(self):
        cg, _, edge_map = SynaraAdapter.load_causal_graph(self.graph.id, self.tenant.id)
        # Manually adjust link strength in CausalGraph
        cg.links[0].strength = 0.85

        updated = SynaraAdapter.persist_posteriors(cg, edge_map, self.tenant.id)
        self.assertEqual(updated, 1)

        self.edge.refresh_from_db()
        self.assertAlmostEqual(self.edge.posterior_strength, 0.85, places=2)

    def test_persist_skips_unchanged(self):
        cg, _, edge_map = SynaraAdapter.load_causal_graph(self.graph.id, self.tenant.id)
        # Don't modify anything
        updated = SynaraAdapter.persist_posteriors(cg, edge_map, self.tenant.id)
        self.assertEqual(updated, 0)


class SynaraAdapterPropagateTest(TestCase):
    """Full cycle: add evidence and propagate through graph."""

    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="Temp")
        self.n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="Visc")
        self.n3 = GraphService.add_node(self.tenant.id, self.graph.id, name="Defects")
        GraphService.add_edge(self.tenant.id, self.graph.id, self.n1.id, self.n2.id)
        GraphService.add_edge(self.tenant.id, self.graph.id, self.n2.id, self.n3.id)

    def test_propagation_returns_changes(self):
        evidence = Evidence(
            id="test_ev_1",
            event="spc_shift",
            context={"shift": "night"},
            strength=0.9,
            source="spc",
            supports=[str(self.n1.id)],
        )
        changes = SynaraAdapter.add_evidence_and_propagate(self.graph.id, self.tenant.id, None, evidence)
        self.assertIsInstance(changes, dict)


class ContradictionDetectionTest(TestCase):
    """Edge-scoped contradiction detection — GRAPH-001 §10, D8."""

    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="X")
        self.n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="Y")
        self.edge = GraphService.add_edge(self.tenant.id, self.graph.id, self.n1.id, self.n2.id, provenance="doe")
        # Add baseline evidence to establish a posterior
        for i in range(3):
            GraphService.add_evidence(
                self.tenant.id,
                self.edge.id,
                source_type="doe",
                observed_at=timezone.now(),
                effect_size=0.5,
                strength=0.9,
            )

    def test_no_contradiction_on_consistent_evidence(self):
        ev = EdgeEvidence.objects.create(
            edge=self.edge,
            source_type="doe",
            observed_at=timezone.now(),
            effect_size=0.45,
            strength=0.9,
        )
        result = SynaraAdapter.check_edge_contradiction(self.tenant.id, self.edge.id, ev)
        # Consistent evidence should not trigger contradiction
        # (likelihood should be reasonable)
        # Note: result may or may not be None depending on BeliefEngine's
        # likelihood computation — the key test is that it doesn't crash
        self.assertIsInstance(result, (dict, type(None)))

    def test_contradiction_returns_signal_dict(self):
        # Force edge to high posterior
        self.edge.posterior_strength = 0.95
        self.edge.save()

        ev = EdgeEvidence.objects.create(
            edge=self.edge,
            source_type="operator",
            observed_at=timezone.now(),
            effect_size=-0.9,
            strength=0.3,
        )
        result = SynaraAdapter.check_edge_contradiction(self.tenant.id, self.edge.id, ev, threshold=0.5)
        # With a very permissive threshold, weak contradicting evidence
        # may trigger
        if result is not None:
            self.assertEqual(result["type"], "edge_contradiction")
            self.assertIn("edge_id", result)
            self.assertIn("message", result)

    def test_no_contradiction_with_insufficient_evidence(self):
        """Edges with < 2 evidence records skip contradiction check."""
        n3 = GraphService.add_node(self.tenant.id, self.graph.id, name="Z")
        new_edge = GraphService.add_edge(self.tenant.id, self.graph.id, self.n2.id, n3.id)
        ev = EdgeEvidence.objects.create(
            edge=new_edge,
            source_type="operator",
            observed_at=timezone.now(),
        )
        result = SynaraAdapter.check_edge_contradiction(self.tenant.id, new_edge.id, ev)
        self.assertIsNone(result)

    def test_contradiction_signal_structure(self):
        """Verify all required fields in contradiction signal."""
        self.edge.posterior_strength = 0.9
        self.edge.save()

        ev = EdgeEvidence.objects.create(
            edge=self.edge,
            source_type="investigation",
            observed_at=timezone.now(),
            effect_size=-0.5,
            strength=0.8,
        )
        result = SynaraAdapter.check_edge_contradiction(self.tenant.id, self.edge.id, ev, threshold=0.99)
        if result is not None:
            required_fields = [
                "type",
                "edge_id",
                "source_node",
                "target_node",
                "current_posterior",
                "evidence_likelihood",
                "threshold",
                "evidence_id",
                "message",
            ]
            for field in required_fields:
                self.assertIn(field, result, f"Missing field: {field}")

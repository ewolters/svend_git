"""
Tests for graph/ app — GRAPH-001 Phase 1.

Covers: model creation, GraphService CRUD, evidence stacking,
recency-weighted posterior, gap report, FMIS seeding, tenant isolation,
retraction, upstream/downstream traversal, explain_edge.
"""

from datetime import timedelta
from uuid import uuid4

from django.test import TestCase
from django.utils import timezone

from core.models.tenant import Tenant

from .models import ProcessEdge, ProcessGraph, ProcessNode
from .service import GraphService


def _make_tenant(name="Test Org"):
    return Tenant.objects.create(name=name, slug=name.lower().replace(" ", "-"))


class ProcessGraphModelTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()

    def test_create_graph(self):
        g = ProcessGraph.objects.create(tenant=self.tenant, name="Main Process")
        self.assertEqual(g.node_count, 0)
        self.assertEqual(g.edge_count, 0)
        self.assertEqual(str(g.tenant), self.tenant.name)

    def test_federated_parent_graph(self):
        parent = ProcessGraph.objects.create(tenant=self.tenant, name="Org Graph")
        child = ProcessGraph.objects.create(
            tenant=self.tenant,
            name="Safety",
            process_area="safety",
            parent_graph=parent,
        )
        self.assertEqual(child.parent_graph, parent)
        self.assertEqual(parent.subgraphs.count(), 1)

    def test_str(self):
        g = ProcessGraph.objects.create(tenant=self.tenant, name="Molding", process_area="molding")
        self.assertIn("molding", str(g))


class ProcessNodeModelTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = ProcessGraph.objects.create(tenant=self.tenant, name="Test Graph")

    def test_create_node(self):
        node = ProcessNode.objects.create(
            graph=self.graph,
            name="Zone 3 Temperature",
            node_type="process_parameter",
            unit="C",
            controllability="direct",
        )
        self.assertEqual(node.name, "Zone 3 Temperature")
        self.assertEqual(self.graph.node_count, 1)

    def test_node_types(self):
        for nt in ProcessNode.NodeType.values:
            node = ProcessNode.objects.create(graph=self.graph, name=f"Node {nt}", node_type=nt)
            self.assertEqual(node.node_type, nt)


class ProcessEdgeModelTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = ProcessGraph.objects.create(tenant=self.tenant, name="Test Graph")
        self.n1 = ProcessNode.objects.create(graph=self.graph, name="Temperature", node_type="process_parameter")
        self.n2 = ProcessNode.objects.create(graph=self.graph, name="Viscosity", node_type="material_property")

    def test_create_edge(self):
        edge = ProcessEdge.objects.create(
            graph=self.graph,
            source=self.n1,
            target=self.n2,
            relation_type="causal",
            provenance="fmea_assertion",
        )
        self.assertEqual(edge.posterior_strength, 0.5)
        self.assertFalse(edge.is_calibrated)
        self.assertEqual(self.graph.edge_count, 1)

    def test_is_calibrated(self):
        edge = ProcessEdge.objects.create(
            graph=self.graph,
            source=self.n1,
            target=self.n2,
            provenance="fmea_assertion",
        )
        self.assertFalse(edge.is_calibrated)
        edge.evidence_count = 1
        edge.provenance = "doe"
        self.assertTrue(edge.is_calibrated)


class GraphServiceCRUDTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(tenant_id=self.tenant.id, name="Test Graph")

    def test_create_and_get_graph(self):
        g = GraphService.get_graph(self.tenant.id, self.graph.id)
        self.assertEqual(g.name, "Test Graph")

    def test_get_or_create_org_graph(self):
        g1 = GraphService.get_or_create_org_graph(self.tenant.id)
        g2 = GraphService.get_or_create_org_graph(self.tenant.id)
        self.assertEqual(g1.id, g2.id)

    def test_add_and_get_node(self):
        node = GraphService.add_node(
            self.tenant.id,
            self.graph.id,
            name="Pressure",
            node_type="process_parameter",
            unit="MPa",
        )
        fetched = GraphService.get_node(self.tenant.id, node.id)
        self.assertEqual(fetched.name, "Pressure")
        self.assertEqual(fetched.unit, "MPa")

    def test_update_node(self):
        node = GraphService.add_node(self.tenant.id, self.graph.id, name="Temp")
        updated = GraphService.update_node(self.tenant.id, node.id, name="Temperature", unit="C")
        self.assertEqual(updated.name, "Temperature")
        self.assertEqual(updated.unit, "C")

    def test_remove_node_cascades_edges(self):
        n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="A")
        n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="B")
        GraphService.add_edge(self.tenant.id, self.graph.id, n1.id, n2.id)
        self.assertEqual(self.graph.edge_count, 1)
        GraphService.remove_node(self.tenant.id, n1.id)
        self.assertEqual(self.graph.edge_count, 0)

    def test_add_and_get_edge(self):
        n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="A")
        n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="B")
        edge = GraphService.add_edge(
            self.tenant.id,
            self.graph.id,
            n1.id,
            n2.id,
            relation_type="causal",
            provenance="fmea_assertion",
        )
        fetched = GraphService.get_edge(self.tenant.id, edge.id)
        self.assertEqual(fetched.source.name, "A")
        self.assertEqual(fetched.target.name, "B")

    def test_get_nodes_filtered(self):
        GraphService.add_node(self.tenant.id, self.graph.id, name="Temp", node_type="process_parameter")
        GraphService.add_node(self.tenant.id, self.graph.id, name="Short Shot", node_type="failure_mode")
        params = GraphService.get_nodes(self.tenant.id, self.graph.id, node_type="process_parameter")
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].name, "Temp")

    def test_get_edges_filtered(self):
        n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="A")
        n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="B")
        GraphService.add_edge(self.tenant.id, self.graph.id, n1.id, n2.id, provenance="fmea_assertion")
        edges = GraphService.get_edges(self.tenant.id, self.graph.id, is_stale=False)
        self.assertEqual(len(edges), 1)


class TenantIsolationTest(TestCase):
    """Verify that GraphService enforces tenant boundaries."""

    def setUp(self):
        self.t1 = _make_tenant("Org A")
        self.t2 = _make_tenant("Org B")
        self.g1 = GraphService.create_graph(self.t1.id, "Graph A")
        self.g2 = GraphService.create_graph(self.t2.id, "Graph B")

    def test_cannot_read_other_tenants_graph(self):
        with self.assertRaises(ProcessGraph.DoesNotExist):
            GraphService.get_graph(self.t2.id, self.g1.id)

    def test_cannot_read_other_tenants_node(self):
        node = GraphService.add_node(self.t1.id, self.g1.id, name="Secret")
        with self.assertRaises(ProcessNode.DoesNotExist):
            GraphService.get_node(self.t2.id, node.id)

    def test_cannot_add_evidence_to_other_tenants_edge(self):
        n1 = GraphService.add_node(self.t1.id, self.g1.id, name="A")
        n2 = GraphService.add_node(self.t1.id, self.g1.id, name="B")
        edge = GraphService.add_edge(self.t1.id, self.g1.id, n1.id, n2.id)
        with self.assertRaises(ProcessEdge.DoesNotExist):
            GraphService.add_evidence(self.t2.id, edge.id, "doe", timezone.now())


class EvidenceStackingTest(TestCase):
    """Evidence stacking and recency-weighted posterior — GRAPH-001 §4.4."""

    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="X")
        self.n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="Y")
        self.edge = GraphService.add_edge(
            self.tenant.id,
            self.graph.id,
            self.n1.id,
            self.n2.id,
            provenance="fmea_assertion",
        )

    def test_add_evidence_updates_count(self):
        GraphService.add_evidence(
            self.tenant.id,
            self.edge.id,
            source_type="doe",
            observed_at=timezone.now(),
            effect_size=0.3,
            strength=0.9,
        )
        self.edge.refresh_from_db()
        self.assertEqual(self.edge.evidence_count, 1)

    def test_add_evidence_updates_posterior(self):
        initial = self.edge.posterior_strength
        GraphService.add_evidence(
            self.tenant.id,
            self.edge.id,
            source_type="doe",
            observed_at=timezone.now(),
            effect_size=0.5,
            strength=1.0,
        )
        self.edge.refresh_from_db()
        self.assertNotEqual(self.edge.posterior_strength, initial)

    def test_add_evidence_updates_provenance(self):
        GraphService.add_evidence(
            self.tenant.id,
            self.edge.id,
            source_type="doe",
            observed_at=timezone.now(),
            effect_size=0.3,
        )
        self.edge.refresh_from_db()
        self.assertEqual(self.edge.provenance, "doe")

    def test_multiple_evidence_stacks(self):
        now = timezone.now()
        for i in range(5):
            GraphService.add_evidence(
                self.tenant.id,
                self.edge.id,
                source_type="investigation",
                observed_at=now - timedelta(days=i * 30),
                effect_size=0.2 + i * 0.05,
                strength=0.8,
            )
        self.edge.refresh_from_db()
        self.assertEqual(self.edge.evidence_count, 5)
        self.assertGreater(self.edge.posterior_strength, 0.5)

    def test_recency_weighting(self):
        """Recent evidence should influence posterior more than old evidence."""
        now = timezone.now()
        # Old evidence: large effect
        GraphService.add_evidence(
            self.tenant.id,
            self.edge.id,
            source_type="doe",
            observed_at=now - timedelta(days=365),
            effect_size=0.9,
            strength=1.0,
        )
        posterior_after_old = ProcessEdge.objects.get(id=self.edge.id).posterior_strength

        # Recent evidence: small effect
        GraphService.add_evidence(
            self.tenant.id,
            self.edge.id,
            source_type="doe",
            observed_at=now,
            effect_size=0.1,
            strength=1.0,
        )
        posterior_after_recent = ProcessEdge.objects.get(id=self.edge.id).posterior_strength

        # Posterior should shift toward recent (smaller) evidence
        self.assertLess(posterior_after_recent, posterior_after_old)

    def test_retracted_evidence_excluded(self):
        ev = GraphService.add_evidence(
            self.tenant.id,
            self.edge.id,
            source_type="doe",
            observed_at=timezone.now(),
            effect_size=0.8,
            strength=1.0,
        )
        self.edge.refresh_from_db()

        GraphService.retract_evidence(self.tenant.id, ev.id, "data entry error")
        self.edge.refresh_from_db()
        self.assertEqual(self.edge.evidence_count, 0)
        # Should revert toward uninformative prior
        self.assertAlmostEqual(self.edge.posterior_strength, 0.5, places=1)

    def test_retracted_evidence_visible_in_stack(self):
        ev = GraphService.add_evidence(
            self.tenant.id,
            self.edge.id,
            source_type="operator",
            observed_at=timezone.now(),
        )
        GraphService.retract_evidence(self.tenant.id, ev.id, "mistake")
        all_evidence = list(self.edge.evidence_stack.all())
        self.assertEqual(len(all_evidence), 1)
        self.assertTrue(all_evidence[0].retracted)


class GapReportTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")

    def test_uncalibrated_edges(self):
        n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="A")
        n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="B")
        GraphService.add_edge(self.tenant.id, self.graph.id, n1.id, n2.id, provenance="fmea_assertion")
        report = GraphService.gap_report(self.tenant.id, self.graph.id)
        self.assertEqual(len(report.uncalibrated_edges), 1)

    def test_measurement_gaps(self):
        GraphService.add_node(
            self.tenant.id,
            self.graph.id,
            name="Temp",
            node_type="process_parameter",
        )
        report = GraphService.gap_report(self.tenant.id, self.graph.id)
        self.assertEqual(len(report.measurement_gaps), 1)

    def test_no_gaps_when_calibrated(self):
        n1 = GraphService.add_node(
            self.tenant.id,
            self.graph.id,
            name="Temp",
            node_type="process_parameter",
            linked_equipment=[str(uuid4())],
        )
        n2 = GraphService.add_node(
            self.tenant.id,
            self.graph.id,
            name="Viscosity",
            node_type="material_property",
            linked_equipment=[str(uuid4())],
        )
        edge = GraphService.add_edge(self.tenant.id, self.graph.id, n1.id, n2.id, provenance="doe")
        GraphService.add_evidence(
            self.tenant.id,
            edge.id,
            source_type="doe",
            observed_at=timezone.now(),
            effect_size=0.3,
        )
        report = GraphService.gap_report(self.tenant.id, self.graph.id)
        self.assertEqual(report.total_gaps, 0)

    def test_stale_edges_in_report(self):
        n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="A")
        n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="B")
        edge = GraphService.add_edge(self.tenant.id, self.graph.id, n1.id, n2.id)
        edge.is_stale = True
        edge.staleness_reason = "spc_shift_detected"
        edge.save()
        report = GraphService.gap_report(self.tenant.id, self.graph.id)
        self.assertEqual(len(report.stale_edges), 1)


class FMISSeedingTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")

    def _make_mock_fmis_row(self, cause="Low Pressure", fm="Short Shot", effect="Incomplete Part"):
        class MockRow:
            def __init__(self, cause_text, failure_mode_text, effect_text):
                self.id = uuid4()
                self.cause_text = cause_text
                self.failure_mode_text = failure_mode_text
                self.effect_text = effect_text

        return MockRow(cause, fm, effect)

    def test_seed_generates_proposals(self):
        row = self._make_mock_fmis_row()
        proposals = GraphService.seed_from_fmis(self.tenant.id, self.graph.id, [row])
        node_proposals = [p for p in proposals if p["type"] == "new_node"]
        edge_proposals = [p for p in proposals if p["type"] == "new_edge"]
        self.assertEqual(len(node_proposals), 3)
        self.assertEqual(len(edge_proposals), 2)

    def test_seed_deduplicates_existing_nodes(self):
        GraphService.add_node(self.tenant.id, self.graph.id, name="Low Pressure")
        row = self._make_mock_fmis_row()
        proposals = GraphService.seed_from_fmis(self.tenant.id, self.graph.id, [row])
        node_proposals = [p for p in proposals if p["type"] == "new_node"]
        self.assertEqual(len(node_proposals), 2)  # fm + effect only

    def test_confirm_seed_creates_nodes_and_edges(self):
        row = self._make_mock_fmis_row()
        proposals = GraphService.seed_from_fmis(self.tenant.id, self.graph.id, [row])
        result = GraphService.confirm_seed(self.tenant.id, self.graph.id, proposals)
        self.assertEqual(len(result["created_nodes"]), 3)
        self.assertEqual(len(result["created_edges"]), 2)
        self.assertEqual(self.graph.node_count, 3)
        self.assertEqual(self.graph.edge_count, 2)

    def test_confirm_seed_idempotent(self):
        row = self._make_mock_fmis_row()
        proposals = GraphService.seed_from_fmis(self.tenant.id, self.graph.id, [row])
        GraphService.confirm_seed(self.tenant.id, self.graph.id, proposals)
        # Confirm again — should not duplicate
        result = GraphService.confirm_seed(self.tenant.id, self.graph.id, proposals)
        self.assertEqual(len(result["created_nodes"]), 0)
        self.assertEqual(len(result["created_edges"]), 0)


class TraversalTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        # Build chain: A → B → C → D
        self.nodes = {}
        for name in ("A", "B", "C", "D"):
            self.nodes[name] = GraphService.add_node(self.tenant.id, self.graph.id, name=name)
        for src, tgt in [("A", "B"), ("B", "C"), ("C", "D")]:
            GraphService.add_edge(
                self.tenant.id,
                self.graph.id,
                self.nodes[src].id,
                self.nodes[tgt].id,
            )

    def test_get_upstream(self):
        upstream = GraphService.get_upstream(self.tenant.id, self.nodes["D"].id)
        names = {n.name for n in upstream}
        self.assertEqual(names, {"A", "B", "C"})

    def test_get_downstream(self):
        downstream = GraphService.get_downstream(self.tenant.id, self.nodes["A"].id)
        names = {n.name for n in downstream}
        self.assertEqual(names, {"B", "C", "D"})

    def test_get_upstream_with_depth(self):
        upstream = GraphService.get_upstream(self.tenant.id, self.nodes["D"].id, depth=1)
        names = {n.name for n in upstream}
        self.assertEqual(names, {"C"})

    def test_get_downstream_with_depth(self):
        downstream = GraphService.get_downstream(self.tenant.id, self.nodes["A"].id, depth=2)
        names = {n.name for n in downstream}
        self.assertEqual(names, {"B", "C"})


class ExplainEdgeTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="Temp")
        n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="Viscosity")
        self.edge = GraphService.add_edge(self.tenant.id, self.graph.id, n1.id, n2.id, provenance="fmea_assertion")

    def test_explain_edge_structure(self):
        explanation = GraphService.explain_edge(self.tenant.id, self.edge.id)
        self.assertEqual(explanation["source"]["name"], "Temp")
        self.assertEqual(explanation["target"]["name"], "Viscosity")
        self.assertFalse(explanation["is_calibrated"])
        self.assertEqual(explanation["evidence"], [])

    def test_explain_edge_with_evidence(self):
        GraphService.add_evidence(
            self.tenant.id,
            self.edge.id,
            source_type="doe",
            observed_at=timezone.now(),
            effect_size=0.3,
            source_description="Factorial DOE on Zone 3",
        )
        explanation = GraphService.explain_edge(self.tenant.id, self.edge.id)
        self.assertEqual(len(explanation["evidence"]), 1)
        self.assertTrue(explanation["is_calibrated"])


class StalenessTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.node = GraphService.add_node(self.tenant.id, self.graph.id, name="Temp")
        n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="Visc")
        n3 = GraphService.add_node(self.tenant.id, self.graph.id, name="Output")
        GraphService.add_edge(self.tenant.id, self.graph.id, self.node.id, n2.id)
        GraphService.add_edge(self.tenant.id, self.graph.id, n2.id, n3.id)

    def test_flag_stale_edges(self):
        stale = GraphService.flag_stale_edges(self.tenant.id, self.node.id)
        self.assertEqual(len(stale), 1)  # only edge where node is source or target
        self.assertTrue(stale[0].is_stale)

    def test_flag_stale_idempotent(self):
        GraphService.flag_stale_edges(self.tenant.id, self.node.id)
        stale = GraphService.flag_stale_edges(self.tenant.id, self.node.id)
        self.assertEqual(len(stale), 0)  # already flagged

"""
Tests for graph integrations — Phase 2 wiring.

Covers: DOE→evidence, SPC→distribution+staleness, FFT→evidence,
PC→evidence, investigation scoping, investigation writeback.
All tests verify graceful degradation when graph/edges don't exist.
"""

from uuid import uuid4

from django.test import TestCase

from core.models.tenant import Tenant

from .integrations import (
    doe_to_graph_evidence,
    fft_to_graph_evidence,
    pc_to_graph_evidence,
    scope_investigation_from_graph,
    spc_to_graph,
    write_back_from_investigation,
)
from .service import GraphService


def _make_tenant(name="Test Org"):
    return Tenant.objects.create(name=name, slug=name.lower().replace(" ", "-"))


class DOEIntegrationTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.temp = GraphService.add_node(
            self.tenant.id, self.graph.id, name="Temperature", node_type="process_parameter"
        )
        self.visc = GraphService.add_node(
            self.tenant.id, self.graph.id, name="Viscosity", node_type="material_property"
        )
        self.edge = GraphService.add_edge(
            self.tenant.id, self.graph.id, self.temp.id, self.visc.id, provenance="fmea_assertion"
        )

    def test_significant_coefficient_creates_evidence(self):
        coefficients = [
            {
                "term": "Temperature",
                "effect": 0.45,
                "coefficient": 0.225,
                "se_coef": 0.05,
                "t_value": 4.5,
                "p_value": 0.002,
                "significant": True,
            }
        ]
        created = doe_to_graph_evidence(self.tenant.id, coefficients, sample_size=32, response_name="Viscosity")
        self.assertEqual(len(created), 1)
        self.edge.refresh_from_db()
        self.assertEqual(self.edge.evidence_count, 1)
        self.assertAlmostEqual(self.edge.effect_size, 0.45, places=2)

    def test_non_significant_coefficient_skipped(self):
        coefficients = [
            {
                "term": "Temperature",
                "effect": 0.02,
                "coefficient": 0.01,
                "se_coef": 0.05,
                "p_value": 0.55,
                "significant": False,
            }
        ]
        created = doe_to_graph_evidence(self.tenant.id, coefficients, sample_size=32, response_name="Viscosity")
        self.assertEqual(len(created), 0)

    def test_graceful_no_graph(self):
        other_tenant = _make_tenant("No Graph Org")
        created = doe_to_graph_evidence(
            other_tenant.id,
            [{"term": "X", "effect": 1.0, "significant": True}],
            sample_size=10,
            response_name="Y",
        )
        self.assertEqual(len(created), 0)

    def test_graceful_no_matching_nodes(self):
        coefficients = [{"term": "Nonexistent", "effect": 0.5, "p_value": 0.01, "significant": True}]
        created = doe_to_graph_evidence(self.tenant.id, coefficients, sample_size=32, response_name="Also Nonexistent")
        self.assertEqual(len(created), 0)


class SPCIntegrationTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.chart_id = uuid4()
        self.node = GraphService.add_node(
            self.tenant.id,
            self.graph.id,
            name="Temp",
            node_type="process_parameter",
            linked_spc_chart=self.chart_id,
        )
        self.n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="Visc")
        self.edge = GraphService.add_edge(self.tenant.id, self.graph.id, self.node.id, self.n2.id)

    def test_in_control_updates_distribution(self):
        result = spc_to_graph(
            self.tenant.id,
            chart_id=self.chart_id,
            center_line=220.0,
            ucl=230.0,
            lcl=210.0,
            in_control=True,
            data_points=[218, 220, 222, 219, 221],
            chart_type="I-MR",
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("distribution_updated", result["actions"])
        self.node.refresh_from_db()
        self.assertAlmostEqual(self.node.distribution["mean"], 220.0)

    def test_out_of_control_flags_stale(self):
        result = spc_to_graph(
            self.tenant.id,
            chart_id=self.chart_id,
            center_line=220.0,
            in_control=False,
            out_of_control_count=3,
            data_points=[218, 235, 240, 219, 238],
            chart_type="I-MR",
        )
        self.assertIn("flagged_1_stale_edges", result["actions"])
        self.edge.refresh_from_db()
        self.assertTrue(self.edge.is_stale)

    def test_graceful_no_graph(self):
        other_tenant = _make_tenant("Empty Org")
        result = spc_to_graph(other_tenant.id, node_name="X", center_line=10)
        self.assertEqual(result["status"], "no_graph")

    def test_graceful_no_node(self):
        result = spc_to_graph(self.tenant.id, chart_id=uuid4(), center_line=10)
        self.assertEqual(result["status"], "no_node")

    def test_find_node_by_name(self):
        result = spc_to_graph(
            self.tenant.id,
            node_name="Temp",
            center_line=220.0,
            data_points=[220],
            chart_type="I-MR",
        )
        self.assertEqual(result["status"], "ok")


class FFTIntegrationTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.fmis_row_id = uuid4()
        self.n1 = GraphService.add_node(
            self.tenant.id,
            self.graph.id,
            name="Cause",
            linked_fmis_rows=[str(self.fmis_row_id)],
        )
        self.n2 = GraphService.add_node(
            self.tenant.id,
            self.graph.id,
            name="Failure Mode",
            node_type="failure_mode",
        )
        self.edge = GraphService.add_edge(
            self.tenant.id, self.graph.id, self.n1.id, self.n2.id, provenance="fmea_assertion"
        )

    def test_perfect_detection(self):
        created = fft_to_graph_evidence(
            self.tenant.id,
            fft_id=uuid4(),
            fmis_row_id=self.fmis_row_id,
            detection_count=5,
            injection_count=5,
            result_code="detected",
            control_being_tested="Visual inspection",
        )
        self.assertEqual(len(created), 1)
        self.edge.refresh_from_db()
        self.assertAlmostEqual(self.edge.effect_size, 1.0)

    def test_partial_detection(self):
        created = fft_to_graph_evidence(
            self.tenant.id,
            fft_id=uuid4(),
            fmis_row_id=self.fmis_row_id,
            detection_count=3,
            injection_count=5,
            result_code="partially_detected",
        )
        self.assertEqual(len(created), 1)
        ev = created[0]
        self.assertAlmostEqual(ev.effect_size, 0.6)

    def test_zero_injection_skipped(self):
        created = fft_to_graph_evidence(
            self.tenant.id,
            fft_id=uuid4(),
            fmis_row_id=self.fmis_row_id,
            detection_count=0,
            injection_count=0,
            result_code="",
        )
        self.assertEqual(len(created), 0)

    def test_graceful_no_fmis_link(self):
        created = fft_to_graph_evidence(
            self.tenant.id,
            fft_id=uuid4(),
            fmis_row_id=uuid4(),
            detection_count=5,
            injection_count=5,
            result_code="detected",
        )
        self.assertEqual(len(created), 0)


class PCIntegrationTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="Step A")
        self.n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="Output")
        self.edge = GraphService.add_edge(self.tenant.id, self.graph.id, self.n1.id, self.n2.id)

    def test_system_works_creates_positive_evidence(self):
        created = pc_to_graph_evidence(
            self.tenant.id,
            pc_id=uuid4(),
            diagnosis="system_works",
            pass_rate=1.0,
            observation_count=5,
            linked_node_ids=[self.n1.id],
        )
        self.assertEqual(len(created), 1)
        self.assertAlmostEqual(created[0].effect_size, 1.0)

    def test_process_gap_creates_negative_evidence(self):
        created = pc_to_graph_evidence(
            self.tenant.id,
            pc_id=uuid4(),
            diagnosis="process_gap",
            pass_rate=0.4,
            observation_count=5,
            linked_node_ids=[self.n1.id],
        )
        self.assertEqual(len(created), 1)
        self.assertAlmostEqual(created[0].effect_size, -1.0)

    def test_incomplete_skipped(self):
        created = pc_to_graph_evidence(
            self.tenant.id,
            pc_id=uuid4(),
            diagnosis="incomplete",
            pass_rate=None,
            observation_count=0,
            linked_node_ids=[self.n1.id],
        )
        self.assertEqual(len(created), 0)

    def test_graceful_no_linked_nodes(self):
        created = pc_to_graph_evidence(
            self.tenant.id,
            pc_id=uuid4(),
            diagnosis="system_works",
            pass_rate=1.0,
            observation_count=3,
            linked_node_ids=[],
        )
        self.assertEqual(len(created), 0)


class InvestigationScopingTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.nodes = {}
        for name in ("A", "B", "C", "D"):
            self.nodes[name] = GraphService.add_node(self.tenant.id, self.graph.id, name=name)
        GraphService.add_edge(self.tenant.id, self.graph.id, self.nodes["A"].id, self.nodes["B"].id)
        GraphService.add_edge(self.tenant.id, self.graph.id, self.nodes["B"].id, self.nodes["C"].id)
        GraphService.add_edge(self.tenant.id, self.graph.id, self.nodes["C"].id, self.nodes["D"].id)

    def test_scope_with_neighbors(self):
        snapshot = scope_investigation_from_graph(
            self.tenant.id, self.graph.id, [self.nodes["B"].id], include_neighbors=True
        )
        # B + neighbors A and C = 3 nodes
        self.assertEqual(snapshot["node_count"], 3)
        # A→B and B→C = 2 edges
        self.assertEqual(snapshot["edge_count"], 2)

    def test_scope_without_neighbors(self):
        snapshot = scope_investigation_from_graph(
            self.tenant.id, self.graph.id, [self.nodes["B"].id], include_neighbors=False
        )
        self.assertEqual(snapshot["node_count"], 1)
        self.assertEqual(snapshot["edge_count"], 0)

    def test_scope_multiple_nodes(self):
        snapshot = scope_investigation_from_graph(
            self.tenant.id,
            self.graph.id,
            [self.nodes["A"].id, self.nodes["D"].id],
            include_neighbors=True,
        )
        # A+neighbors(B) + D+neighbors(C) = A,B,C,D = 4
        self.assertEqual(snapshot["node_count"], 4)


class InvestigationWritebackTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="X")
        self.n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="Y")
        self.edge = GraphService.add_edge(self.tenant.id, self.graph.id, self.n1.id, self.n2.id)

    def test_writeback_new_node(self):
        result = write_back_from_investigation(
            self.tenant.id,
            self.graph.id,
            [{"type": "new_node", "name": "Z", "node_type": "environmental_factor"}],
        )
        self.assertEqual(len(result["created_nodes"]), 1)
        self.assertTrue(self.graph.nodes.filter(name="Z").exists())

    def test_writeback_new_edge(self):
        GraphService.add_node(self.tenant.id, self.graph.id, name="Z")
        result = write_back_from_investigation(
            self.tenant.id,
            self.graph.id,
            [{"type": "new_edge", "source_name": "Y", "target_name": "Z"}],
        )
        self.assertEqual(len(result["created_edges"]), 1)

    def test_writeback_new_evidence(self):
        result = write_back_from_investigation(
            self.tenant.id,
            self.graph.id,
            [
                {
                    "type": "new_evidence",
                    "edge_id": str(self.edge.id),
                    "source_type": "investigation",
                    "effect_size": 0.4,
                    "description": "Found causal relationship via DOE",
                }
            ],
        )
        self.assertEqual(len(result["added_evidence"]), 1)
        self.edge.refresh_from_db()
        self.assertEqual(self.edge.evidence_count, 1)

    def test_writeback_deduplicates_nodes(self):
        result = write_back_from_investigation(
            self.tenant.id,
            self.graph.id,
            [
                {"type": "new_node", "name": "X"},  # already exists
                {"type": "new_node", "name": "W"},  # new
            ],
        )
        self.assertEqual(len(result["created_nodes"]), 1)  # only W

    def test_writeback_deduplicates_edges(self):
        result = write_back_from_investigation(
            self.tenant.id,
            self.graph.id,
            [{"type": "new_edge", "source_name": "X", "target_name": "Y"}],  # already exists
        )
        self.assertEqual(len(result["created_edges"]), 0)

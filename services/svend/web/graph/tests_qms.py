"""
Tests for QMS → graph alignment bridges.

Covers: equipment calibration, NCR context, audit finding context,
training gaps from graph, management review input. All verify
graceful degradation when graph doesn't exist.
"""

from uuid import uuid4

from django.test import TestCase

from core.models.tenant import Tenant

from .integrations import (
    audit_finding_to_graph_context,
    equipment_calibration_to_graph,
    management_review_graph_input,
    ncr_to_graph_context,
    training_gaps_from_graph,
)
from .service import GraphService


def _make_tenant(name="Test Org"):
    return Tenant.objects.create(name=name, slug=name.lower().replace(" ", "-"))


class EquipmentCalibrationTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.equip_node = GraphService.add_node(self.tenant.id, self.graph.id, name="CMM #3", node_type="measurement")
        self.param_node = GraphService.add_node(
            self.tenant.id, self.graph.id, name="Dimension A", node_type="quality_characteristic"
        )
        self.edge = GraphService.add_edge(
            self.tenant.id,
            self.graph.id,
            self.equip_node.id,
            self.param_node.id,
            relation_type="measurement",
        )

    def test_calibration_pass_creates_evidence(self):
        from agents_api.models import MeasurementEquipment

        equip = MeasurementEquipment.objects.create(
            name="CMM #3",
            linked_process_node=self.equip_node,
        )
        created = equipment_calibration_to_graph(
            self.tenant.id, equip.id, calibration_result="pass", certificate_number="CAL-001"
        )
        self.assertEqual(len(created), 1)
        self.edge.refresh_from_db()
        self.assertEqual(self.edge.evidence_count, 1)

    def test_calibration_fail_creates_negative_evidence(self):
        from agents_api.models import MeasurementEquipment

        equip = MeasurementEquipment.objects.create(
            name="CMM #3",
            linked_process_node=self.equip_node,
        )
        created = equipment_calibration_to_graph(self.tenant.id, equip.id, calibration_result="fail")
        self.assertEqual(len(created), 1)
        self.assertAlmostEqual(created[0].effect_size, -0.5)

    def test_no_linked_node_skips(self):
        from agents_api.models import MeasurementEquipment

        equip = MeasurementEquipment.objects.create(name="Unlinked Gage")
        created = equipment_calibration_to_graph(self.tenant.id, equip.id)
        self.assertEqual(len(created), 0)

    def test_graceful_no_graph(self):
        other_tenant = _make_tenant("No Graph")
        created = equipment_calibration_to_graph(other_tenant.id, uuid4())
        self.assertEqual(len(created), 0)


class NCRGraphContextTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="Temp")
        self.n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="Defects")
        self.edge = GraphService.add_edge(self.tenant.id, self.graph.id, self.n1.id, self.n2.id)

    def test_returns_affected_edges(self):
        result = ncr_to_graph_context(self.tenant.id, uuid4(), [self.n1.id])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["affected_edges"]), 1)

    def test_suggests_investigation_nodes(self):
        result = ncr_to_graph_context(self.tenant.id, uuid4(), [self.n1.id, self.n2.id])
        self.assertEqual(len(result["suggested_investigation_nodes"]), 2)

    def test_graceful_no_graph(self):
        other_tenant = _make_tenant("Empty")
        result = ncr_to_graph_context(other_tenant.id, uuid4(), [uuid4()])
        self.assertEqual(result["status"], "no_graph")


class AuditFindingContextTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="Process Step")

    def test_major_finding_suggests_investigate(self):
        result = audit_finding_to_graph_context(self.tenant.id, uuid4(), "major", [self.n1.id])
        self.assertEqual(result["suggested_action"], "investigate")

    def test_minor_finding_suggests_monitor(self):
        result = audit_finding_to_graph_context(self.tenant.id, uuid4(), "minor", [self.n1.id])
        self.assertEqual(result["suggested_action"], "monitor")


class TrainingGapsTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")

    def test_uncalibrated_edges_produce_suggestions(self):
        n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="A")
        n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="B")
        GraphService.add_edge(self.tenant.id, self.graph.id, n1.id, n2.id, provenance="fmea_assertion")
        suggestions = training_gaps_from_graph(self.tenant.id)
        self.assertTrue(any(s["type"] == "uncalibrated_edge" for s in suggestions))

    def test_measurement_gaps_produce_suggestions(self):
        GraphService.add_node(self.tenant.id, self.graph.id, name="Temp", node_type="process_parameter")
        suggestions = training_gaps_from_graph(self.tenant.id)
        self.assertTrue(any(s["type"] == "measurement_gap" for s in suggestions))

    def test_empty_graph_no_suggestions(self):
        suggestions = training_gaps_from_graph(self.tenant.id)
        self.assertEqual(len(suggestions), 0)

    def test_graceful_no_graph(self):
        other_tenant = _make_tenant("Empty")
        suggestions = training_gaps_from_graph(other_tenant.id)
        self.assertEqual(len(suggestions), 0)


class ManagementReviewInputTest(TestCase):
    def setUp(self):
        self.tenant = _make_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")

    def test_returns_graph_summary(self):
        n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="X")
        n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="Y")
        GraphService.add_edge(self.tenant.id, self.graph.id, n1.id, n2.id)

        result = management_review_graph_input(self.tenant.id)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["summary"]["total_nodes"], 2)
        self.assertEqual(result["summary"]["total_edges"], 1)
        self.assertEqual(len(result["top_priorities"]), 3)

    def test_calibration_rate(self):
        from django.utils import timezone

        n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="X")
        n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="Y")
        edge = GraphService.add_edge(self.tenant.id, self.graph.id, n1.id, n2.id, provenance="doe")
        GraphService.add_evidence(
            self.tenant.id,
            edge.id,
            source_type="doe",
            observed_at=timezone.now(),
            effect_size=0.3,
        )
        result = management_review_graph_input(self.tenant.id)
        self.assertEqual(result["summary"]["calibration_rate"], 100.0)

    def test_graceful_no_graph(self):
        other_tenant = _make_tenant("Empty")
        result = management_review_graph_input(other_tenant.id)
        self.assertEqual(result["status"], "no_graph")

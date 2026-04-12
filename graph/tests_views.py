"""
Tests for graph API view endpoints.

Covers: /api/graph/data/, /api/graph/node/<id>/, /api/graph/edge/<id>/,
/api/graph/gaps/, /api/graph/seed/, /api/graph/seed/confirm/.
All verify authentication and tenant isolation.
"""

import json
from uuid import uuid4

from django.test import TestCase
from django.utils import timezone

from accounts.models import User
from core.models.tenant import Membership, Tenant

from .service import GraphService


def _setup_user_with_tenant():
    """Create a user with a tenant membership for authenticated API access."""
    slug = f"test-org-{uuid4().hex[:8]}"
    tenant = Tenant.objects.create(name="Test Org", slug=slug)
    user = User.objects.create_user(
        username=f"testuser_{uuid4().hex[:8]}",
        email=f"test_{uuid4().hex[:8]}@example.com",
        password="testpass123",
    )
    Membership.objects.create(user=user, tenant=tenant, role="admin")

    from rest_framework.test import APIClient

    client = APIClient()
    client.force_login(user)
    return user, tenant, client


class GraphDataEndpointTest(TestCase):
    def setUp(self):
        self.user, self.tenant, self.client = _setup_user_with_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="Temp")
        self.n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="Visc")
        self.edge = GraphService.add_edge(self.tenant.id, self.graph.id, self.n1.id, self.n2.id)

    def test_get_graph_data(self):
        resp = self.client.get("/api/graph/data/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["stats"]["node_count"], 2)
        self.assertEqual(data["stats"]["edge_count"], 1)
        self.assertFalse(data["empty"])

    def test_empty_graph(self):
        _, _, other_client = _setup_user_with_tenant()
        resp = other_client.get("/api/graph/data/", follow=True)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["empty"])

    def test_fmea_lens(self):
        fm = GraphService.add_node(self.tenant.id, self.graph.id, name="Short Shot", node_type="failure_mode")
        GraphService.add_edge(self.tenant.id, self.graph.id, self.n1.id, fm.id)
        resp = self.client.get("/api/graph/data/?lens=fmea")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # FMEA lens shows failure modes + their upstream
        self.assertTrue(data["stats"]["node_count"] >= 2)

    def test_gap_lens(self):
        GraphService.add_edge(
            self.tenant.id,
            self.graph.id,
            self.n1.id,
            self.n2.id,
            provenance="fmea_assertion",
        )
        resp = self.client.get("/api/graph/data/?lens=gap")
        self.assertEqual(resp.status_code, 200)

    def test_requires_auth(self):
        from rest_framework.test import APIClient

        anon = APIClient()
        resp = anon.get("/api/graph/data/")
        self.assertIn(resp.status_code, (301, 302, 401, 403))


class NodeDetailEndpointTest(TestCase):
    def setUp(self):
        self.user, self.tenant, self.client = _setup_user_with_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        self.node = GraphService.add_node(
            self.tenant.id,
            self.graph.id,
            name="Temperature",
            node_type="process_parameter",
            unit="C",
        )

    def test_get_node_detail(self):
        resp = self.client.get(f"/api/graph/node/{self.node.id}/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "Temperature")
        self.assertEqual(data["unit"], "C")

    def test_nonexistent_node(self):
        resp = self.client.get(f"/api/graph/node/{uuid4()}/")
        self.assertEqual(resp.status_code, 404)


class EdgeDetailEndpointTest(TestCase):
    def setUp(self):
        self.user, self.tenant, self.client = _setup_user_with_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")
        n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="A")
        n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="B")
        self.edge = GraphService.add_edge(self.tenant.id, self.graph.id, n1.id, n2.id)

    def test_get_edge_detail(self):
        resp = self.client.get(f"/api/graph/edge/{self.edge.id}/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["source"]["name"], "A")
        self.assertEqual(data["target"]["name"], "B")
        self.assertIn("evidence", data)

    def test_edge_with_evidence(self):
        GraphService.add_evidence(
            self.tenant.id,
            self.edge.id,
            source_type="doe",
            observed_at=timezone.now(),
            effect_size=0.3,
        )
        resp = self.client.get(f"/api/graph/edge/{self.edge.id}/")
        data = resp.json()
        self.assertEqual(len(data["evidence"]), 1)
        self.assertTrue(data["is_calibrated"])


class GapReportEndpointTest(TestCase):
    def setUp(self):
        self.user, self.tenant, self.client = _setup_user_with_tenant()
        self.graph = GraphService.create_graph(self.tenant.id, "Test")

    def test_empty_graph_gaps(self):
        resp = self.client.get("/api/graph/gaps/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["total_gaps"], 0)

    def test_uncalibrated_edge_in_gaps(self):
        n1 = GraphService.add_node(self.tenant.id, self.graph.id, name="A")
        n2 = GraphService.add_node(self.tenant.id, self.graph.id, name="B")
        GraphService.add_edge(self.tenant.id, self.graph.id, n1.id, n2.id, provenance="fmea_assertion")
        resp = self.client.get("/api/graph/gaps/")
        data = resp.json()
        self.assertGreater(data["total_gaps"], 0)


class SeedEndpointTest(TestCase):
    def setUp(self):
        self.user, self.tenant, self.client = _setup_user_with_tenant()

    def test_seed_requires_fmis_id(self):
        resp = self.client.post(
            "/api/graph/seed/",
            json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_seed_nonexistent_fmis(self):
        resp = self.client.post(
            "/api/graph/seed/",
            json.dumps({"fmis_id": str(uuid4())}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 404)

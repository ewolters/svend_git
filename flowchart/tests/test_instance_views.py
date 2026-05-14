"""Tests for flowchart instance API endpoints."""

import json

from django.test import TestCase
from pydantic import BaseModel

from conftest import SECURE_OFF, make_user
from flowchart.models import FlowchartInstance, FlowchartTemplate
from syn.plugins.base import Plugin, PluginOutput
from syn.plugins.registry import get_registry


@SECURE_OFF
class TestInstanceCreateEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("api@test.com")
        self.client.login(username="api", password="testpass123!")
        self.tpl = FlowchartTemplate.objects.create(
            name="Quick Cpk",
            definition={
                "devices": [
                    {
                        "id": "ds",
                        "plugin": "data_source",
                        "label": "Data",
                        "ports": {"inputs": [], "outputs": [{"name": "measurements", "type": "data:column"}]},
                    }
                ],
                "connections": [],
                "config": {"ds": {}},
            },
            devices_used=["data_source"],
        )

    def test_create_from_template(self):
        resp = self.client.post(
            "/api/flowchart/instances/",
            data=json.dumps({"template_id": str(self.tpl.id), "name": "My Study"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        assert "id" in body
        assert body["name"] == "My Study"
        assert body["template_id"] == str(self.tpl.id)

    def test_create_blank(self):
        resp = self.client.post(
            "/api/flowchart/instances/",
            data=json.dumps({"name": "Blank"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        assert body["template_id"] is None

    def test_requires_auth(self):
        self.client.logout()
        resp = self.client.post(
            "/api/flowchart/instances/",
            data=json.dumps({"name": "X"}),
            content_type="application/json",
        )
        assert resp.status_code in (401, 403)


@SECURE_OFF
class TestInstanceDetailEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("api@test.com")
        self.client.login(username="api", password="testpass123!")
        self.inst = FlowchartInstance.objects.create(
            name="My Flow",
            user=self.user,
            definition={"devices": [], "connections": [], "config": {}},
        )

    def test_get_instance(self):
        resp = self.client.get(f"/api/flowchart/instances/{self.inst.id}/")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        assert body["name"] == "My Flow"
        assert "definition" in body

    def test_delete_instance(self):
        resp = self.client.delete(f"/api/flowchart/instances/{self.inst.id}/")
        self.assertEqual(resp.status_code, 200)
        self.inst.refresh_from_db()
        assert self.inst.is_deleted is True


@SECURE_OFF
class TestAddDeviceEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("api@test.com")
        self.client.login(username="api", password="testpass123!")
        self.inst = FlowchartInstance.objects.create(
            name="Build Mode",
            user=self.user,
            definition={"devices": [], "connections": [], "config": {}},
        )

    def test_add_device(self):
        resp = self.client.post(
            f"/api/flowchart/instances/{self.inst.id}/devices/",
            data=json.dumps(
                {
                    "device_id": "ds1",
                    "plugin_name": "data_source",
                    "label": "My Data",
                    "position": {"x": 100, "y": 100},
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.inst.refresh_from_db()
        assert len(self.inst.definition["devices"]) == 1


@SECURE_OFF
class TestRemoveDeviceEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("api@test.com")
        self.client.login(username="api", password="testpass123!")
        self.inst = FlowchartInstance.objects.create(
            name="RM",
            user=self.user,
            definition={
                "devices": [{"id": "ds1", "plugin": "data_source", "label": "D"}],
                "connections": [],
                "config": {"ds1": {}},
            },
        )

    def test_remove_device(self):
        resp = self.client.delete(
            f"/api/flowchart/instances/{self.inst.id}/devices/ds1/",
        )
        self.assertEqual(resp.status_code, 200)
        self.inst.refresh_from_db()
        assert len(self.inst.definition["devices"]) == 0


@SECURE_OFF
class TestConnectionEndpoints(TestCase):
    def setUp(self):
        self.user = make_user("api@test.com")
        self.client.login(username="api", password="testpass123!")
        self.inst = FlowchartInstance.objects.create(
            name="Wire",
            user=self.user,
            definition={
                "devices": [
                    {
                        "id": "ds1",
                        "plugin": "data_source",
                        "label": "D",
                        "ports": {"inputs": [], "outputs": [{"name": "measurements", "type": "data:column"}]},
                    },
                    {
                        "id": "cap1",
                        "plugin": "capability_study",
                        "label": "C",
                        "ports": {"inputs": [{"name": "data", "type": "data:column"}], "outputs": []},
                    },
                ],
                "connections": [],
                "config": {},
            },
        )

    def test_add_connection(self):
        resp = self.client.post(
            f"/api/flowchart/instances/{self.inst.id}/connections/",
            data=json.dumps({"source": "ds1.measurements", "target": "cap1.data"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.inst.refresh_from_db()
        assert len(self.inst.definition["connections"]) == 1

    def test_remove_connection(self):
        self.inst.definition["connections"].append(
            {"source": "ds1.measurements", "target": "cap1.data", "type": "data:column"},
        )
        self.inst.save(update_fields=["definition"])

        resp = self.client.post(
            f"/api/flowchart/instances/{self.inst.id}/connections/remove/",
            data=json.dumps({"source": "ds1.measurements", "target": "cap1.data"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.inst.refresh_from_db()
        assert len(self.inst.definition["connections"]) == 0

    def test_validate_connection(self):
        resp = self.client.post(
            f"/api/flowchart/instances/{self.inst.id}/connections/validate/",
            data=json.dumps({"source": "ds1.measurements", "target": "cap1.data"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        assert body["valid"] is True


class _ExecTestInput(BaseModel):
    value: float


class _ExecTestPlugin(Plugin):
    name = "exec_test_plugin"
    version = "1.0.0"
    description = "Test"
    input_schema = _ExecTestInput

    def execute(self, validated_input, context):
        return [PluginOutput("result", "metric", validated_input["value"] * 2)]


@SECURE_OFF
class TestExecuteInstanceEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("exec@test.com")
        self.client.login(username="exec", password="testpass123!")
        reg = get_registry()
        if not reg.has("exec_test_plugin"):
            reg.register(_ExecTestPlugin)
        self.inst = FlowchartInstance.objects.create(
            name="Exec Test",
            user=self.user,
            definition={
                "devices": [{"id": "d1", "plugin": "exec_test_plugin"}],
                "connections": [],
                "config": {"d1": {"value": 5.0}},
            },
        )

    def test_execute_instance(self):
        resp = self.client.post(f"/api/flowchart/instances/{self.inst.id}/run/")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        assert "d1" in body["results"]
        assert body["results"]["d1"]["outputs"]["result"] == 10.0

    def test_execute_stores_job_ids(self):
        resp = self.client.post(f"/api/flowchart/instances/{self.inst.id}/run/")
        body = resp.json()
        assert "job_id" in body["results"]["d1"]

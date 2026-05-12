"""Tests for flowchart API endpoints."""

import json

from django.test import TestCase
from pydantic import BaseModel

from conftest import SECURE_OFF, make_user
from flowchart.models import FlowchartTemplate
from syn.plugins.base import Plugin, PluginOutput
from syn.plugins.registry import get_registry


class _ViewTestInput(BaseModel):
    value: float


class _ViewTestPlugin(Plugin):
    name = "flowchart_view_test"
    version = "1.0.0"
    description = "Test plugin for view tests"
    input_schema = _ViewTestInput

    def execute(self, validated_input, context):
        return [
            PluginOutput("result", "metric", validated_input["value"] * 10),
        ]


@SECURE_OFF
class TestFlowchartRunEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("flow@test.com")
        self.client.login(username="flow", password="testpass123!")
        reg = get_registry()
        if not reg.has("flowchart_view_test"):
            reg.register(_ViewTestPlugin)

    def test_run_inline_definition(self):
        resp = self.client.post(
            "/api/flowchart/run/",
            data=json.dumps(
                {
                    "definition": {
                        "devices": [
                            {"plugin": "flowchart_view_test", "id": "d1"},
                        ],
                        "connections": [],
                        "config": {"d1": {"value": 7.0}},
                    },
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        assert "d1" in body["results"]
        assert body["results"]["d1"]["status"] == "completed"
        assert body["results"]["d1"]["outputs"]["result"] == 70.0

    def test_run_requires_auth(self):
        self.client.logout()
        resp = self.client.post(
            "/api/flowchart/run/",
            data=json.dumps({"definition": {"devices": [], "connections": []}}),
            content_type="application/json",
        )
        assert resp.status_code in (401, 403)


@SECURE_OFF
class TestTemplateListEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("flow@test.com")
        self.client.login(username="flow", password="testpass123!")

    def test_list_templates(self):
        FlowchartTemplate.objects.create(
            name="Quick Cpk",
            definition={"devices": [], "connections": []},
            devices_used=["capability_study"],
        )
        FlowchartTemplate.objects.create(
            name="PPAP",
            definition={"devices": [], "connections": []},
            devices_used=["capability_study", "control_chart"],
        )
        resp = self.client.get("/api/flowchart/templates/")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        assert len(body["templates"]) == 2
        names = [t["name"] for t in body["templates"]]
        assert "Quick Cpk" in names
        assert "PPAP" in names

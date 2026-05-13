"""Tests for flowchart execution engine."""

from typing import List, Optional

from django.test import TestCase
from pydantic import BaseModel

from flowchart.engine import _build_multi_port_lookup, execute_flowchart
from job.models import Job
from syn.plugins.base import Plugin, PluginOutput
from syn.plugins.registry import PluginRegistry


class _SourceInput(BaseModel):
    value: float


class _SourcePlugin(Plugin):
    name = "test_source"
    version = "1.0.0"
    description = "Outputs a metric"
    input_schema = _SourceInput

    def execute(self, validated_input, context):
        return [
            PluginOutput("result", "metric", validated_input["value"]),
        ]


class _SinkInput(BaseModel):
    value: float


class _SinkPlugin(Plugin):
    name = "test_sink"
    version = "1.0.0"
    description = "Receives a metric"
    input_schema = _SinkInput

    def execute(self, validated_input, context):
        doubled = validated_input["value"] * 2
        return [
            PluginOutput("doubled", "metric", doubled),
        ]


class _CollectorInput(BaseModel):
    items: Optional[List] = None


class _CollectorPlugin(Plugin):
    """Multi-port device that receives a list of values."""

    name = "test_collector"
    version = "1.0.0"
    description = "Collects multiple inputs into a list"
    input_schema = _CollectorInput

    def execute(self, validated_input, context):
        items = validated_input.get("items") or []
        return [
            PluginOutput("count", "metric", float(len(items))),
            PluginOutput("collected", "text", {"items": items}),
        ]


class TestExecuteFlowchart(TestCase):
    def setUp(self):
        self.registry = PluginRegistry()
        self.registry.register(_SourcePlugin)
        self.registry.register(_SinkPlugin)

    def test_two_device_chain(self):
        definition = {
            "devices": [
                {"plugin": "test_source", "id": "src"},
                {"plugin": "test_sink", "id": "snk"},
            ],
            "connections": [
                {"source": "src.result", "target": "snk.value"},
            ],
            "config": {
                "src": {"value": 5.0},
            },
        }
        results = execute_flowchart(
            definition=definition,
            actor="test@svend.ai",
            registry=self.registry,
        )
        assert "src" in results
        assert "snk" in results
        assert results["src"]["job"].status == "completed"
        assert results["snk"]["job"].status == "completed"
        snk_outputs = list(results["snk"]["job"].outputs.all())
        doubled = next(o for o in snk_outputs if o.output_key == "doubled")
        assert doubled.value_numeric == 10.0

    def test_topological_order(self):
        definition = {
            "devices": [
                {"plugin": "test_sink", "id": "snk"},
                {"plugin": "test_source", "id": "src"},
            ],
            "connections": [
                {"source": "src.result", "target": "snk.value"},
            ],
            "config": {"src": {"value": 3.0}},
        }
        results = execute_flowchart(
            definition=definition,
            actor="test@svend.ai",
            registry=self.registry,
        )
        snk_outputs = list(results["snk"]["job"].outputs.all())
        doubled = next(o for o in snk_outputs if o.output_key == "doubled")
        assert doubled.value_numeric == 6.0

    def test_unconnected_inputs_use_config(self):
        definition = {
            "devices": [{"plugin": "test_source", "id": "src"}],
            "connections": [],
            "config": {"src": {"value": 42.0}},
        }
        results = execute_flowchart(
            definition=definition,
            actor="test@svend.ai",
            registry=self.registry,
        )
        src_outputs = list(results["src"]["job"].outputs.all())
        result = next(o for o in src_outputs if o.output_key == "result")
        assert result.value_numeric == 42.0

    def test_creates_jobs_per_device(self):
        definition = {
            "devices": [
                {"plugin": "test_source", "id": "src"},
                {"plugin": "test_sink", "id": "snk"},
            ],
            "connections": [{"source": "src.result", "target": "snk.value"}],
            "config": {"src": {"value": 1.0}},
        }
        execute_flowchart(
            definition=definition,
            actor="engine_test@svend.ai",
            registry=self.registry,
        )
        jobs = Job.objects.filter(actor="engine_test@svend.ai").order_by("created_at")
        assert jobs.count() == 2
        assert jobs[0].plugin_name == "test_source"
        assert jobs[1].plugin_name == "test_sink"


class TestBuildMultiPortLookup(TestCase):
    def test_extracts_multi_ports(self):
        devices = [
            {
                "id": "rpt1",
                "plugin": "report_builder",
                "ports": {
                    "inputs": [
                        {"name": "charts", "type": "chart:*", "multi": True},
                        {"name": "title", "type": "config:title"},
                    ],
                    "outputs": [],
                },
            },
        ]
        result = _build_multi_port_lookup(devices)
        assert ("rpt1", "charts") in result
        assert ("rpt1", "title") not in result

    def test_no_ports_key(self):
        """Devices without ports metadata don't break the lookup."""
        devices = [{"id": "src", "plugin": "test_source"}]
        result = _build_multi_port_lookup(devices)
        assert len(result) == 0


class TestMultiPortRouting(TestCase):
    """Engine collects multiple connections into lists for multi-port devices."""

    def setUp(self):
        self.registry = PluginRegistry()
        self.registry.register(_SourcePlugin)
        self.registry.register(_CollectorPlugin)

    def test_multi_port_collects_into_list(self):
        """Two sources wired to the same multi-port arrive as a list."""
        definition = {
            "devices": [
                {"plugin": "test_source", "id": "s1"},
                {"plugin": "test_source", "id": "s2"},
                {
                    "plugin": "test_collector",
                    "id": "col",
                    "ports": {
                        "inputs": [{"name": "items", "type": "metric:*", "multi": True}],
                        "outputs": [],
                    },
                },
            ],
            "connections": [
                {"source": "s1.result", "target": "col.items"},
                {"source": "s2.result", "target": "col.items"},
            ],
            "config": {
                "s1": {"value": 10.0},
                "s2": {"value": 20.0},
            },
        }
        results = execute_flowchart(
            definition=definition,
            actor="test@svend.ai",
            registry=self.registry,
        )
        col_outputs = results["col"]["outputs"]
        assert col_outputs["count"] == 2.0
        assert set(col_outputs["collected"]["items"]) == {10.0, 20.0}

    def test_single_connection_to_multi_port_still_list(self):
        """Even one connection to a multi-port wraps in a list."""
        definition = {
            "devices": [
                {"plugin": "test_source", "id": "s1"},
                {
                    "plugin": "test_collector",
                    "id": "col",
                    "ports": {
                        "inputs": [{"name": "items", "type": "metric:*", "multi": True}],
                        "outputs": [],
                    },
                },
            ],
            "connections": [
                {"source": "s1.result", "target": "col.items"},
            ],
            "config": {"s1": {"value": 7.0}},
        }
        results = execute_flowchart(
            definition=definition,
            actor="test@svend.ai",
            registry=self.registry,
        )
        col_outputs = results["col"]["outputs"]
        assert col_outputs["count"] == 1.0
        assert col_outputs["collected"]["items"] == [7.0]

    def test_non_multi_port_still_overwrites(self):
        """Regular ports still overwrite (backwards compatible)."""
        definition = {
            "devices": [
                {"plugin": "test_source", "id": "src"},
                {"plugin": "test_sink", "id": "snk"},
            ],
            "connections": [
                {"source": "src.result", "target": "snk.value"},
            ],
            "config": {"src": {"value": 5.0}},
        }
        self.registry.register(_SinkPlugin)
        results = execute_flowchart(
            definition=definition,
            actor="test@svend.ai",
            registry=self.registry,
        )
        assert results["snk"]["outputs"]["doubled"] == 10.0

    def test_three_sources_to_multi_port(self):
        """Three sources wired to the same multi-port."""
        definition = {
            "devices": [
                {"plugin": "test_source", "id": "s1"},
                {"plugin": "test_source", "id": "s2"},
                {"plugin": "test_source", "id": "s3"},
                {
                    "plugin": "test_collector",
                    "id": "col",
                    "ports": {
                        "inputs": [{"name": "items", "type": "metric:*", "multi": True}],
                        "outputs": [],
                    },
                },
            ],
            "connections": [
                {"source": "s1.result", "target": "col.items"},
                {"source": "s2.result", "target": "col.items"},
                {"source": "s3.result", "target": "col.items"},
            ],
            "config": {
                "s1": {"value": 1.0},
                "s2": {"value": 2.0},
                "s3": {"value": 3.0},
            },
        }
        results = execute_flowchart(
            definition=definition,
            actor="test@svend.ai",
            registry=self.registry,
        )
        col_outputs = results["col"]["outputs"]
        assert col_outputs["count"] == 3.0
        assert sorted(col_outputs["collected"]["items"]) == [1.0, 2.0, 3.0]

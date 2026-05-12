"""Tests for flowchart execution engine."""

from django.test import TestCase
from pydantic import BaseModel

from flowchart.engine import execute_flowchart
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

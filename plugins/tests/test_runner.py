"""Tests for the plugin runner — full execution lifecycle."""

from unittest.mock import patch

import pytest
from django.test import TestCase
from pydantic import BaseModel, ValidationError

from conftest import SECURE_OFF, make_membership, make_tenant, make_user
from job.models import Job
from syn.plugins.base import Plugin, PluginOutput
from syn.plugins.registry import PluginRegistry
from syn.plugins.runner import run_plugin


class _RunnerInput(BaseModel):
    value: float
    multiplier: float = 2.0


class _RunnerPlugin(Plugin):
    name = "test_runner"
    version = "1.0.0"
    description = "Plugin for runner tests"
    input_schema = _RunnerInput

    def execute(self, validated_input, context):
        result = validated_input["value"] * validated_input["multiplier"]
        return [
            PluginOutput("result", "metric", result, measure_slug="test_measure"),
            PluginOutput("summary", "text", {"text": f"Result: {result}"}),
        ]


class TestRunPlugin(TestCase):
    def setUp(self):
        self.registry = PluginRegistry()
        self.registry.register(_RunnerPlugin)

    def test_creates_job_with_correct_fields(self):
        job = run_plugin(
            "test_runner",
            {"value": 5.0, "multiplier": 3.0},
            actor="test@svend.ai",
            registry=self.registry,
        )
        assert job.plugin_name == "test_runner"
        assert job.status == "completed"
        assert job.actor == "test@svend.ai"
        assert job.inputs == {"value": 5.0, "multiplier": 3.0}
        assert job.duration_ms is not None
        assert job.duration_ms >= 0

    def test_creates_job_outputs(self):
        job = run_plugin(
            "test_runner",
            {"value": 5.0},
            actor="test@svend.ai",
            registry=self.registry,
        )
        outputs = list(job.outputs.all())
        assert len(outputs) == 2

        metric_out = next(o for o in outputs if o.output_key == "result")
        assert metric_out.output_type == "metric"
        assert metric_out.value_numeric == 10.0
        assert metric_out.measure_slug == "test_measure"
        assert metric_out.provenance == "calculated"

        text_out = next(o for o in outputs if o.output_key == "summary")
        assert text_out.output_type == "text"
        assert text_out.value_json == {"text": "Result: 10.0"}

    def test_validates_input(self):
        """Invalid input raises ValidationError before execution."""
        with pytest.raises(ValidationError):
            run_plugin(
                "test_runner",
                {"value": "not_a_number"},
                actor="test@svend.ai",
                registry=self.registry,
            )

    def test_unknown_plugin_raises(self):
        with pytest.raises(KeyError):
            run_plugin(
                "nonexistent",
                {},
                actor="test@svend.ai",
                registry=self.registry,
            )

    def test_emits_bus_event(self):
        with patch("syn.plugins.runner.emit") as mock_emit:
            job = run_plugin(
                "test_runner",
                {"value": 5.0},
                actor="test@svend.ai",
                registry=self.registry,
            )
            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == "plugin.execution.completed"
            assert call_args[0][1]["plugin_name"] == "test_runner"
            assert call_args[0][1]["job_id"] == str(job.id)

    def test_failed_execution_marks_job_failed(self):
        class _FailPlugin(Plugin):
            name = "fail_plugin"
            version = "1.0.0"
            description = "Always fails"
            input_schema = _RunnerInput

            def execute(self, validated_input, context):
                raise ValueError("Something broke")

        fail_registry = PluginRegistry()
        fail_registry.register(_FailPlugin)

        with pytest.raises(ValueError, match="Something broke"):
            run_plugin(
                "fail_plugin",
                {"value": 1.0},
                actor="test@svend.ai",
                registry=fail_registry,
            )

        # Job should exist and be marked failed
        job = Job.objects.filter(plugin_name="fail_plugin").first()
        assert job is not None
        assert job.status == "failed"

    def test_scratch_flag_propagates(self):
        job = run_plugin(
            "test_runner",
            {"value": 1.0},
            actor="test@svend.ai",
            registry=self.registry,
            is_scratch=True,
        )
        assert job.is_scratch is True

    def test_outputs_summary_populated(self):
        job = run_plugin(
            "test_runner",
            {"value": 5.0},
            actor="test@svend.ai",
            registry=self.registry,
        )
        assert job.outputs_summary == {"result": "metric", "summary": "text"}


# --- PCL Write-back Tests ---


class _SimPlugin(Plugin):
    """Plugin that outputs simulated provenance."""

    name = "sim_plugin"
    version = "1.0.0"
    description = "Simulated output for testing"
    input_schema = _RunnerInput

    def execute(self, validated_input, context):
        result = validated_input["value"] * validated_input["multiplier"]
        return [
            PluginOutput("result", "metric", result, provenance="simulated", measure_slug="test_measure"),
        ]


@SECURE_OFF
class TestPCLWriteBack(TestCase):
    """Tests for automatic PCL write-back from plugin runner."""

    def setUp(self):
        self.user = make_user("runner@test.com", tier="team")
        self.tenant = make_tenant("Runner Org", slug="runner-org", plan="team")
        make_membership(self.tenant, self.user)

        from pcl.models import Measure

        self.measure = Measure.objects.create(
            tenant_id=self.tenant.id,
            name="Test Measure",
            slug="test_measure",
            unit="",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )

        self.registry = PluginRegistry()
        self.registry.register(_RunnerPlugin)
        self.registry.register(_SimPlugin)

    def test_metric_output_writes_to_pcl(self):
        """Plugin metric output with measure_slug creates a PCL Datapoint."""
        job = run_plugin(
            "test_runner",
            {"value": 5.0, "multiplier": 3.0},
            actor=self.user.email,
            tenant_id=self.tenant.id,
            registry=self.registry,
        )

        from pcl.models import Datapoint

        dp = Datapoint.objects.filter(measure=self.measure).first()
        assert dp is not None
        assert dp.value == 15.0
        assert dp.provenance == "calculated"
        assert dp.source_job_id == job.id
        assert dp.source_type == "plugin"
        assert dp.source_ref_type == "plugin:test_runner"

    def test_pcl_cache_updated_for_calculated(self):
        """Calculated provenance updates the Measure's cached aggregate."""
        run_plugin(
            "test_runner",
            {"value": 5.0, "multiplier": 2.0},
            actor=self.user.email,
            tenant_id=self.tenant.id,
            registry=self.registry,
        )

        self.measure.refresh_from_db()
        assert self.measure.cached_value == 10.0
        assert self.measure.cached_n == 1

    def test_simulated_skips_cache_update(self):
        """Simulated provenance stores Datapoint but does NOT update cache."""
        run_plugin(
            "sim_plugin",
            {"value": 5.0, "multiplier": 2.0},
            actor=self.user.email,
            tenant_id=self.tenant.id,
            registry=self.registry,
        )

        from pcl.models import Datapoint

        assert Datapoint.objects.filter(measure=self.measure).count() == 1

        self.measure.refresh_from_db()
        assert self.measure.cached_value is None
        assert self.measure.cached_n == 0

    def test_missing_measure_does_not_fail(self):
        """If measure_slug doesn't match a Measure, plugin still succeeds."""

        class _OrphanPlugin(Plugin):
            name = "orphan_plugin"
            version = "1.0.0"
            description = "Outputs to nonexistent measure"
            input_schema = _RunnerInput

            def execute(self, validated_input, context):
                return [
                    PluginOutput("val", "metric", 42.0, measure_slug="nonexistent_slug"),
                ]

        reg = PluginRegistry()
        reg.register(_OrphanPlugin)

        # Should not raise
        job = run_plugin(
            "orphan_plugin",
            {"value": 1.0},
            actor=self.user.email,
            tenant_id=self.tenant.id,
            registry=reg,
        )
        assert job.status == "completed"

    def test_text_output_skips_pcl(self):
        """Non-metric outputs (even with measure_slug) don't write to PCL."""

        class _TextPlugin(Plugin):
            name = "text_plugin"
            version = "1.0.0"
            description = "Text output with measure_slug"
            input_schema = _RunnerInput

            def execute(self, validated_input, context):
                return [
                    PluginOutput("summary", "text", {"text": "hello"}, measure_slug="test_measure"),
                ]

        reg = PluginRegistry()
        reg.register(_TextPlugin)

        run_plugin(
            "text_plugin",
            {"value": 1.0},
            actor=self.user.email,
            tenant_id=self.tenant.id,
            registry=reg,
        )

        from pcl.models import Datapoint

        assert Datapoint.objects.filter(measure=self.measure).count() == 0

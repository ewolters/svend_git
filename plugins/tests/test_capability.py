"""Tests for the Capability Study plugin."""

import numpy as np
from django.test import TestCase

from plugins.capability import CapabilityStudyPlugin
from syn.plugins.registry import PluginRegistry
from syn.plugins.runner import run_plugin


class TestCapabilityStudyPlugin:
    def test_metadata(self):
        meta = CapabilityStudyPlugin.get_metadata()
        assert meta["name"] == "capability_study"
        assert "data" in meta["input_schema"]["properties"]
        assert "usl" in meta["input_schema"]["properties"]
        assert "lsl" in meta["input_schema"]["properties"]

    def test_execute_with_both_specs(self):
        plugin = CapabilityStudyPlugin()
        np.random.seed(42)
        data = np.random.normal(50, 2, 100).tolist()

        outputs = plugin.execute(
            {"data": data, "usl": 56.0, "lsl": 44.0},
            {"job_id": "test", "actor": "test"},
        )

        # Should have metric outputs for cpk, ppk
        keys = [o.key for o in outputs]
        assert "cpk" in keys
        assert "ppk" in keys

        # Cpk should be reasonable for N(50,2) with specs at +/-6
        cpk_out = next(o for o in outputs if o.key == "cpk")
        assert cpk_out.output_type == "metric"
        assert cpk_out.provenance == "calculated"
        assert cpk_out.value > 0.8  # Should be ~1.0 for this data

        # Should have chart outputs
        chart_outputs = [o for o in outputs if o.output_type == "chart"]
        assert len(chart_outputs) >= 1  # At least histogram

        # Should have summary text
        text_outputs = [o for o in outputs if o.output_type == "text"]
        assert len(text_outputs) >= 1

    def test_execute_one_sided_usl_only(self):
        plugin = CapabilityStudyPlugin()
        data = np.random.normal(10, 1, 50).tolist()

        outputs = plugin.execute(
            {"data": data, "usl": 14.0},
            {"job_id": "test", "actor": "test"},
        )

        keys = [o.key for o in outputs]
        assert "cpk" in keys
        # No Cp/Pp for one-sided
        assert "cp" not in keys

    def test_execute_no_specs(self):
        """Without spec limits, no capability indices -- just descriptive stats."""
        plugin = CapabilityStudyPlugin()
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        outputs = plugin.execute(
            {"data": data},
            {"job_id": "test", "actor": "test"},
        )

        keys = [o.key for o in outputs]
        assert "cpk" not in keys
        # Should still get charts and summary
        assert "summary" in keys


class TestCapabilityStudyIntegration(TestCase):
    """Integration test — run through full runner with DB."""

    def test_full_lifecycle(self):
        registry = PluginRegistry()
        registry.register(CapabilityStudyPlugin)

        np.random.seed(42)
        data = np.random.normal(50, 2, 100).tolist()

        job = run_plugin(
            "capability_study",
            {"data": data, "usl": 56.0, "lsl": 44.0, "target": 50.0},
            actor="test@svend.ai",
            registry=registry,
        )

        assert job.status == "completed"
        assert job.plugin_name == "capability_study"

        outputs = list(job.outputs.all())
        assert len(outputs) >= 4  # cpk, ppk, at least 1 chart, summary

        # Check PCL-writable metric
        cpk_output = job.outputs.filter(output_key="cpk").first()
        assert cpk_output is not None
        assert cpk_output.value_numeric is not None
        assert cpk_output.measure_slug == "cpk"

"""Test that Job model has plugin_name field."""

from django.test import TestCase

from job.models import Job


class TestJobPluginName(TestCase):
    def test_job_has_plugin_name_field(self):
        job = Job(
            plugin_name="capability_study",
            actor="test@svend.ai",
            inputs={"data": [1, 2, 3]},
        )
        assert job.plugin_name == "capability_study"

    def test_plugin_name_nullable(self):
        """Jobs created without plugin (legacy, API-driven) have null plugin_name."""
        job = Job(actor="test@svend.ai", inputs={})
        assert job.plugin_name is None

    def test_plugin_name_in_to_dict(self):
        job = Job(plugin_name="control_chart", actor="test@svend.ai", inputs={})
        d = job.to_dict()
        assert d["plugin_name"] == "control_chart"

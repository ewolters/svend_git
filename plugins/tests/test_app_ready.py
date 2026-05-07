"""Test that plugins register on Django startup."""

from django.test import TestCase

from syn.plugins import get_registry


class TestPluginAppReady(TestCase):
    def test_capability_study_registered(self):
        """CapabilityStudyPlugin should be registered after app startup."""
        registry = get_registry()
        assert registry.has("capability_study")

    def test_registry_metadata_accessible(self):
        registry = get_registry()
        meta = registry.get_metadata("capability_study")
        assert meta["name"] == "capability_study"
        assert meta["input_schema"] is not None

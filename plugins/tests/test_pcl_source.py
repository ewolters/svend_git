from django.test import TestCase

from syn.plugins.registry import PluginRegistry


class TestPCLSourcePlugin(TestCase):
    def setUp(self):
        from plugins.pcl_source import PCLSourcePlugin

        self.registry = PluginRegistry()
        self.registry.register(PCLSourcePlugin)
        self.plugin = self.registry.get("pcl_source")

    def test_reads_existing_measure(self):
        from pcl.models import Datapoint, Measure

        m = Measure.objects.create(
            slug="test-cpk",
            name="Test Cpk",
            unit="ratio",
            measure_type="process",
            value_type="continuous",
        )
        Datapoint.objects.create(
            measure=m,
            value=1.45,
            source_type="manual",
            timestamp=m.created_at,
        )

        outputs = self.plugin.execute(
            {"measure_slugs": ["test-cpk"]},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert "test-cpk" in by_key
        assert by_key["test-cpk"].value == 1.45
        assert by_key["test-cpk"].output_type == "metric"
        assert by_key["test-cpk"].provenance == "observed"

    def test_missing_measure_outputs_none(self):
        outputs = self.plugin.execute(
            {"measure_slugs": ["nonexistent-slug"]},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert "nonexistent-slug" in by_key
        assert by_key["nonexistent-slug"].value is None

    def test_multiple_measures(self):
        from pcl.models import Datapoint, Measure

        for slug, val in [("dim-a", 50.1), ("dim-b", 49.8)]:
            m = Measure.objects.create(
                slug=slug,
                name=slug,
                unit="mm",
                measure_type="product",
                value_type="continuous",
            )
            Datapoint.objects.create(
                measure=m,
                value=val,
                source_type="manual",
                timestamp=m.created_at,
            )

        outputs = self.plugin.execute(
            {"measure_slugs": ["dim-a", "dim-b"]},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["dim-a"].value == 50.1
        assert by_key["dim-b"].value == 49.8

    def test_empty_slugs_raises(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            self.plugin.input_schema(measure_slugs=[])

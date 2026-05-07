"""Tests for the Plugin ABC contract."""

import pytest

from syn.plugins.base import Plugin, PluginOutput


class TestPluginOutput:
    def test_metric_output(self):
        out = PluginOutput(key="cpk", output_type="metric", value=1.45)
        assert out.key == "cpk"
        assert out.output_type == "metric"
        assert out.value == 1.45
        assert out.provenance == "calculated"
        assert out.measure_slug is None

    def test_chart_output_with_measure_slug(self):
        out = PluginOutput(
            key="histogram",
            output_type="chart",
            value={"data": [], "layout": {}},
            provenance="calculated",
            measure_slug="bore_cpk",
        )
        assert out.output_type == "chart"
        assert out.measure_slug == "bore_cpk"

    def test_simulated_provenance(self):
        out = PluginOutput(key="projected_cpk", output_type="metric", value=1.8, provenance="simulated")
        assert out.provenance == "simulated"


class TestPluginABC:
    def test_cannot_instantiate_without_name(self):
        class BadPlugin(Plugin):
            description = "Missing name"
            input_schema = None

            def execute(self, validated_input, context):
                return []

        with pytest.raises(TypeError):
            BadPlugin()

    def test_cannot_instantiate_without_execute(self):
        """Plugin is abstract — must implement execute()."""
        with pytest.raises(TypeError):
            Plugin()

    def test_valid_plugin_instantiates(self):
        from pydantic import BaseModel

        class DummyInput(BaseModel):
            x: float

        class DummyPlugin(Plugin):
            name = "dummy"
            version = "1.0.0"
            description = "A test plugin"
            input_schema = DummyInput

            def execute(self, validated_input, context):
                return [PluginOutput("result", "metric", validated_input["x"] * 2)]

        p = DummyPlugin()
        assert p.name == "dummy"
        assert p.version == "1.0.0"

    def test_get_metadata(self):
        from pydantic import BaseModel

        class MetaInput(BaseModel):
            value: float
            label: str = "default"

        class MetaPlugin(Plugin):
            name = "meta_test"
            version = "2.1.0"
            description = "Metadata test"
            input_schema = MetaInput

            def execute(self, validated_input, context):
                return []

        meta = MetaPlugin.get_metadata()
        assert meta["name"] == "meta_test"
        assert meta["version"] == "2.1.0"
        assert meta["description"] == "Metadata test"
        assert "properties" in meta["input_schema"]
        assert "value" in meta["input_schema"]["properties"]

    def test_execute_returns_outputs(self):
        from pydantic import BaseModel

        class AddInput(BaseModel):
            a: float
            b: float

        class AddPlugin(Plugin):
            name = "add"
            version = "1.0.0"
            description = "Add two numbers"
            input_schema = AddInput

            def execute(self, validated_input, context):
                total = validated_input["a"] + validated_input["b"]
                return [PluginOutput("sum", "metric", total)]

        p = AddPlugin()
        outputs = p.execute({"a": 3.0, "b": 4.0}, {})
        assert len(outputs) == 1
        assert outputs[0].value == 7.0

"""Tests for the PluginRegistry."""

import pytest
from pydantic import BaseModel

from syn.plugins.base import Plugin, PluginOutput
from syn.plugins.registry import PluginRegistry


class _TestInput(BaseModel):
    x: float


class _AlphaPlugin(Plugin):
    name = "alpha"
    version = "1.0.0"
    description = "Alpha plugin"
    input_schema = _TestInput

    def execute(self, validated_input, context):
        return [PluginOutput("out", "metric", validated_input["x"])]


class _BetaPlugin(Plugin):
    name = "beta"
    version = "2.0.0"
    description = "Beta plugin"
    input_schema = _TestInput

    def execute(self, validated_input, context):
        return []


class TestPluginRegistry:
    def setup_method(self):
        self.registry = PluginRegistry()

    def test_register_and_get(self):
        self.registry.register(_AlphaPlugin)
        plugin = self.registry.get("alpha")
        assert isinstance(plugin, _AlphaPlugin)
        assert plugin.name == "alpha"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="No plugin registered"):
            self.registry.get("nonexistent")

    def test_list_plugins(self):
        self.registry.register(_AlphaPlugin)
        self.registry.register(_BetaPlugin)
        listing = self.registry.list_plugins()
        assert len(listing) == 2
        names = [p["name"] for p in listing]
        assert "alpha" in names
        assert "beta" in names

    def test_duplicate_registration_raises(self):
        self.registry.register(_AlphaPlugin)
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(_AlphaPlugin)

    def test_has(self):
        self.registry.register(_AlphaPlugin)
        assert self.registry.has("alpha") is True
        assert self.registry.has("nope") is False

    def test_get_metadata(self):
        self.registry.register(_AlphaPlugin)
        meta = self.registry.get_metadata("alpha")
        assert meta["name"] == "alpha"
        assert meta["version"] == "1.0.0"
        assert "properties" in meta["input_schema"]

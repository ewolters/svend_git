"""Regression test: template port schemas match actual plugin input schemas.

Every seed template declares port names in definition JSON. Those port names
become the keys routed by engine.py into plugin input_data dicts. If a template
says "operations" but the plugin expects "steps", execution fails silently or
with a Pydantic validation error.

This test catches that drift automatically.

Run: pytest flowchart/tests/test_template_port_sync.py -v
"""

import pytest

from flowchart.management.commands.seed_templates import TEMPLATES
from syn.plugins.registry import get_registry


@pytest.fixture(scope="module")
def registry():
    return get_registry()


def _get_plugin_input_fields(registry, plugin_name: str) -> set:
    """Return set of field names from the plugin's Pydantic input_schema."""
    plugin_cls = registry.get(plugin_name)
    if not plugin_cls or not plugin_cls.input_schema:
        return set()
    return set(plugin_cls.input_schema.model_fields.keys())


def _get_plugin_required_fields(registry, plugin_name: str) -> set:
    """Return set of REQUIRED field names from the plugin's input_schema."""
    plugin_cls = registry.get(plugin_name)
    if not plugin_cls or not plugin_cls.input_schema:
        return set()
    return {name for name, field in plugin_cls.input_schema.model_fields.items() if field.is_required()}


def _template_ids():
    """Yield (template_name, device) tuples for parametrize."""
    for tpl in TEMPLATES:
        for dev in tpl["definition"]["devices"]:
            yield f"{tpl['name']}::{dev['id']}({dev['plugin']})"


def _template_device_pairs():
    """Yield (template, device_dict) pairs."""
    for tpl in TEMPLATES:
        for dev in tpl["definition"]["devices"]:
            yield tpl, dev


class TestTemplatePortSync:
    """Verify template port names align with plugin input schemas."""

    @pytest.mark.parametrize(
        "tpl,dev",
        list(_template_device_pairs()),
        ids=list(_template_ids()),
    )
    def test_input_port_names_match_plugin_fields(self, registry, tpl, dev):
        """Each input port declared in a template must be a valid field on the plugin's input_schema."""
        plugin_name = dev["plugin"]
        input_fields = _get_plugin_input_fields(registry, plugin_name)
        if not input_fields:
            pytest.skip(f"Plugin {plugin_name} has no input_schema")

        port_names = {p["name"] for p in dev.get("ports", {}).get("inputs", [])}
        bad = port_names - input_fields
        assert not bad, (
            f"Template '{tpl['name']}' device '{dev['id']}' ({plugin_name}): "
            f"input ports {bad} don't match plugin fields {sorted(input_fields)}"
        )

    @pytest.mark.parametrize(
        "tpl,dev",
        list(_template_device_pairs()),
        ids=list(_template_ids()),
    )
    def test_required_fields_have_port_or_config(self, registry, tpl, dev):
        """Every required plugin field must be provided by either a port connection or config.

        Entry-point devices (no input ports declared AND no incoming connections)
        are exempt — they receive data from the user at execution time.
        """
        plugin_name = dev["plugin"]
        required = _get_plugin_required_fields(registry, plugin_name)
        if not required:
            return

        # Fields supplied by input ports (via connections)
        port_names = {p["name"] for p in dev.get("ports", {}).get("inputs", [])}

        # Fields supplied by connections targeting this device
        connected_ports = set()
        for conn in tpl["definition"].get("connections", []):
            tgt_device, tgt_port = conn["target"].split(".", 1)
            if tgt_device == dev["id"]:
                connected_ports.add(tgt_port)

        # Entry-point devices: no input ports and no incoming connections = user supplies data
        if not port_names and not connected_ports:
            return

        # Fields supplied by config
        config_keys = set(tpl["definition"].get("config", {}).get(dev["id"], {}).keys())

        supplied = port_names | connected_ports | config_keys
        missing = required - supplied
        assert not missing, (
            f"Template '{tpl['name']}' device '{dev['id']}' ({plugin_name}): "
            f"required fields {missing} have no port connection or config. "
            f"Supplied: ports={sorted(port_names)}, connected={sorted(connected_ports)}, "
            f"config={sorted(config_keys)}"
        )

    @pytest.mark.parametrize(
        "tpl,dev",
        list(_template_device_pairs()),
        ids=list(_template_ids()),
    )
    def test_connections_target_valid_fields(self, registry, tpl, dev):
        """Every connection targeting this device must route to a real input field."""
        plugin_name = dev["plugin"]
        input_fields = _get_plugin_input_fields(registry, plugin_name)
        if not input_fields:
            pytest.skip(f"Plugin {plugin_name} has no input_schema")

        for conn in tpl["definition"].get("connections", []):
            tgt_device, tgt_port = conn["target"].split(".", 1)
            if tgt_device == dev["id"]:
                assert tgt_port in input_fields, (
                    f"Template '{tpl['name']}': connection {conn['source']} → "
                    f"{conn['target']} routes to '{tgt_port}' which is not in "
                    f"{plugin_name}'s input fields {sorted(input_fields)}"
                )


class TestTemplatePluginsExist:
    """Every plugin referenced by a template must be registered."""

    @pytest.mark.parametrize(
        "tpl,dev",
        list(_template_device_pairs()),
        ids=list(_template_ids()),
    )
    def test_plugin_registered(self, registry, tpl, dev):
        assert registry.has(dev["plugin"]), (
            f"Template '{tpl['name']}' references plugin '{dev['plugin']}' which is not registered"
        )

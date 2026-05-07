"""Test that plugin event schemas are registered."""

import jsonschema
import pytest
from django.test import TestCase

from plugins.apps import PluginsConfig
from syn.events.models import EventSchemaRegistry


class TestPluginEventSchema(TestCase):
    """Verify plugin.execution.completed event schema registration."""

    @classmethod
    def setUpTestData(cls):
        # Replay the same registration that apps.py ready() performs.
        # During tests the ready() call hits the production DB, so the
        # test DB starts empty.  Re-running registration is idempotent.
        config = PluginsConfig("plugins", __import__("plugins"))
        config._register_event_schemas()

    def test_plugin_execution_completed_schema_exists(self):
        schema = EventSchemaRegistry.objects.filter(event_name="plugin.execution.completed").first()
        assert schema is not None

    def test_schema_validates_correct_payload(self):
        schema_obj = EventSchemaRegistry.objects.get(event_name="plugin.execution.completed")
        payload = {
            "job_id": "550e8400-e29b-41d4-a716-446655440000",
            "plugin_name": "capability_study",
            "outputs": ["cpk", "ppk", "histogram"],
        }
        jsonschema.validate(payload, schema_obj.schema)

    def test_schema_rejects_invalid_payload(self):
        schema_obj = EventSchemaRegistry.objects.get(event_name="plugin.execution.completed")
        payload = {"plugin_name": "test"}  # Missing required job_id, outputs
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, schema_obj.schema)

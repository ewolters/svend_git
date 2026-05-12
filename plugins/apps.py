"""SVEND Plugin implementations — register on startup."""

import logging
import uuid

from django.apps import AppConfig

logger = logging.getLogger(__name__)

# Fixed UUID for system-level plugin event schemas (no tenant).
SYSTEM_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


class PluginsConfig(AppConfig):
    name = "plugins"
    verbose_name = "SVEND Plugins"

    def ready(self):
        from syn.plugins import get_registry

        registry = get_registry()

        # Register all device plugins
        from plugins.capability import CapabilityStudyPlugin
        from plugins.control_chart import ControlChartPlugin
        from plugins.fmea_device import FMEAPlugin
        from plugins.queue_device import QueuePlugin
        from plugins.simulation_device import SimulationPlugin
        from plugins.triage_device import TriagePlugin
        from plugins.vsm_device import VSMPlugin

        for plugin_cls in [
            CapabilityStudyPlugin,
            ControlChartPlugin,
            TriagePlugin,
            VSMPlugin,
            FMEAPlugin,
            SimulationPlugin,
            QueuePlugin,
        ]:
            if not registry.has(plugin_cls.name):
                registry.register(plugin_cls)

        # Register event schemas
        self._register_event_schemas()

    def _register_event_schemas(self):
        """Register plugin-related event schemas (EVT-001 compliant)."""
        from syn.events.models import EventSchemaRegistry

        schema_def = {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "format": "uuid"},
                "plugin_name": {"type": "string"},
                "outputs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["job_id", "plugin_name", "outputs"],
        }

        try:
            EventSchemaRegistry.objects.update_or_create(
                event_name="plugin.execution.completed",
                defaults={
                    "version": "1.0.0",
                    "schema": schema_def,
                    "tenant_id": SYSTEM_TENANT_ID,
                    "is_active": True,
                },
            )
        except Exception as exc:
            logger.warning(f"[PLUGINS] Could not register event schema: {exc}")

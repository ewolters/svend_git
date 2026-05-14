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
        from plugins.conditional import ConditionalPlugin
        from plugins.contract_envelope import ContractEnvelopePlugin
        from plugins.contract_router import ContractRouterPlugin
        from plugins.control_chart import ControlChartPlugin
        from plugins.data_source import DataSourcePlugin
        from plugins.doe_design import DOEDesignPlugin
        from plugins.fishbone_device import FishbonePlugin
        from plugins.fmea_device import FMEAPlugin
        from plugins.hypothesis_test import HypothesisTestPlugin
        from plugins.narrative_device import NarrativePlugin
        from plugins.pcl_source import PCLSourcePlugin
        from plugins.process_behavior_device import ProcessBehaviorPlugin
        from plugins.queue_device import QueuePlugin
        from plugins.regression_device import RegressionPlugin
        from plugins.reliability_device import ReliabilityPlugin
        from plugins.report_builder import ReportBuilderPlugin
        from plugins.savings_device import SavingsPlugin
        from plugins.simulation_device import SimulationPlugin
        from plugins.strategic_cascade import StrategicCascadePlugin
        from plugins.text_input import TextInputPlugin
        from plugins.triage_device import TriagePlugin
        from plugins.vsm_device import VSMPlugin
        from plugins.x_matrix import XMatrixPlugin

        for plugin_cls in [
            DataSourcePlugin,
            CapabilityStudyPlugin,
            ConditionalPlugin,
            ContractEnvelopePlugin,
            ContractRouterPlugin,
            ControlChartPlugin,
            DOEDesignPlugin,
            HypothesisTestPlugin,
            NarrativePlugin,
            PCLSourcePlugin,
            ProcessBehaviorPlugin,
            RegressionPlugin,
            ReliabilityPlugin,
            StrategicCascadePlugin,
            TriagePlugin,
            VSMPlugin,
            FMEAPlugin,
            FishbonePlugin,
            SimulationPlugin,
            QueuePlugin,
            ReportBuilderPlugin,
            SavingsPlugin,
            TextInputPlugin,
            XMatrixPlugin,
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

        # Flowchart instance lifecycle events
        instance_schema = {
            "type": "object",
            "properties": {
                "instance_id": {"type": "string", "format": "uuid"},
            },
            "required": ["instance_id"],
        }
        for event_name in [
            "flowchart.instance.created",
            "flowchart.instance.deleted",
            "flowchart.instance.device_added",
            "flowchart.instance.device_removed",
            "flowchart.instance.connection_added",
            "flowchart.instance.connection_removed",
        ]:
            try:
                EventSchemaRegistry.objects.update_or_create(
                    event_name=event_name,
                    defaults={
                        "version": "1.0.0",
                        "schema": instance_schema,
                        "tenant_id": SYSTEM_TENANT_ID,
                        "is_active": True,
                    },
                )
            except Exception as exc:
                logger.warning(f"[PLUGINS] Could not register event schema {event_name}: {exc}")

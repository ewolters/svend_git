import logging

from django.apps import AppConfig

logger = logging.getLogger("svend.graph")


class GraphConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "graph"
    verbose_name = "Graph — Unified Knowledge Graph and Process Model"

    def ready(self):
        logger.info("Graph: GRAPH-001 process knowledge service loaded")

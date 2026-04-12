"""Snapshot knowledge health metrics for all tenants.

Run daily via syn.sched or cron:
    python manage.py snapshot_health

Creates one KnowledgeHealthSnapshot per tenant per day.
Enables historical trending and OLR-001 §13 maturity tracking.
"""

import logging

from django.core.management.base import BaseCommand

from core.models.tenant import Tenant
from graph.models import KnowledgeHealthSnapshot
from graph.service import GraphService

logger = logging.getLogger("svend.graph.health")


class Command(BaseCommand):
    help = "Snapshot knowledge health metrics for all tenants (GRAPH-001 §13)"

    def handle(self, *args, **options):
        tenants = Tenant.objects.all()
        created = 0
        skipped = 0

        for tenant in tenants:
            graph = GraphService.get_org_graph(tenant.id)
            if not graph:
                skipped += 1
                continue

            health = GraphService.compute_knowledge_health(tenant.id, graph.id)
            if not health:
                skipped += 1
                continue

            from django.utils import timezone

            KnowledgeHealthSnapshot.objects.update_or_create(
                graph=graph,
                snapshot_date=timezone.now().date(),
                defaults={
                    "total_nodes": health.get("total_nodes", 0),
                    "total_edges": health.get("total_edges", 0),
                    "calibrated_edges": health.get("calibrated_edges", 0),
                    "calibration_rate": health.get("calibration_rate", 0),
                    "staleness_rate": health.get("staleness_rate", 0),
                    "contradiction_rate": health.get("contradiction_rate", 0),
                    "knowledge_gap_ratio": health.get("knowledge_gap_ratio", 0),
                    "maturity_level": health.get("maturity_level", 1),
                    "detection_distribution": health.get("detection_distribution", {}),
                },
            )
            created += 1

        self.stdout.write(f"Health snapshots: {created} created, {skipped} skipped")
        logger.info("Health snapshot complete: %d created, %d skipped", created, skipped)

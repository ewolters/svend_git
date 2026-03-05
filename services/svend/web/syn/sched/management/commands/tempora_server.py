"""
Tempora Scheduler Server Management Command
=============================================

Starts the cognitive scheduler (syn.sched) as a long-running process.

Usage:
    python manage.py tempora_server --single-node
    python manage.py tempora_server --single-node --workers 4

Standard: SCH-001 §scheduler_lifecycle
"""

import logging
import signal
import sys

from django.core.management.base import BaseCommand

logger = logging.getLogger("tempora.core")


class Command(BaseCommand):
    help = "Start the Tempora cognitive task scheduler"

    def add_arguments(self, parser):
        parser.add_argument(
            "--single-node",
            action="store_true",
            help="Run in single-node mode (no clustering)",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=4,
            help="Number of worker threads (default: 4)",
        )
        parser.add_argument(
            "--node-id",
            type=str,
            default="tempora-1",
            help="Node identifier for clustering",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=7000,
            help="Port for cluster communication",
        )
        parser.add_argument(
            "--peers",
            type=str,
            default="",
            help="Comma-separated peer list (name:host:port)",
        )

    def handle(self, *args, **options):
        from syn.sched.core import CognitiveScheduler

        worker_count = options["workers"]
        node_id = options["node_id"]

        logger.info(
            f"Starting Tempora scheduler (node={node_id}, workers={worker_count})",
        )

        scheduler = CognitiveScheduler(worker_count=worker_count)

        # Handle graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            scheduler.stop(graceful=True, timeout=30)
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Start the scheduler (blocks via schedule loop thread)
        scheduler.start()

        # Keep the main thread alive
        try:
            scheduler._shutdown_event.wait()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt, shutting down...")
            scheduler.stop(graceful=True, timeout=30)

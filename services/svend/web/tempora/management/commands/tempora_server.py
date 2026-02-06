"""
Tempora Server Management Command

Runs the Tempora distributed scheduler server.

Usage:
    python manage.py tempora_server --node-id=tempora-1 --port=7000
    python manage.py tempora_server --config=/etc/tempora/config.json

Standard: TEMPORA-HA-001
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser
from django.utils import timezone

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """
    Run the Tempora distributed scheduler server.

    This command starts a Tempora node that participates in:
    - Leader election (Raft consensus)
    - Log replication
    - Task execution
    - Work distribution
    """

    help = "Run the Tempora distributed scheduler server"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shutdown_requested = False
        self._server = None
        self._scheduler = None

    def add_arguments(self, parser: CommandParser) -> None:
        """Add command arguments."""
        parser.add_argument(
            "--node-id",
            type=str,
            default=os.environ.get("TEMPORA_NODE_ID", "tempora-1"),
            help="Unique node identifier",
        )
        parser.add_argument(
            "--host",
            type=str,
            default=os.environ.get("TEMPORA_HOST", "0.0.0.0"),
            help="Host to bind to",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=int(os.environ.get("TEMPORA_PORT", "7000")),
            help="Port for cluster coordination",
        )
        parser.add_argument(
            "--peers",
            type=str,
            default=os.environ.get("TEMPORA_PEERS", ""),
            help="Comma-separated peer list (node-id:host:port,...)",
        )
        parser.add_argument(
            "--config",
            type=str,
            help="Path to JSON config file",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=int(os.environ.get("TEMPORA_WORKERS", "4")),
            help="Number of worker threads",
        )
        parser.add_argument(
            "--single-node",
            action="store_true",
            help="Run in single-node mode (no clustering)",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        """Handle command execution."""
        # Load config from file if provided
        if options.get("config"):
            config = self._load_config(options["config"])
            options.update(config)

        node_id = options["node_id"]
        host = options["host"]
        port = options["port"]
        workers = options["workers"]
        single_node = options.get("single_node", False)

        # Parse peers
        peers = self._parse_peers(options.get("peers", ""))

        self.stdout.write(self.style.SUCCESS(f"Starting Tempora Server"))
        self.stdout.write(f"  Node ID:     {node_id}")
        self.stdout.write(f"  Bind:        {host}:{port}")
        self.stdout.write(f"  Workers:     {workers}")
        self.stdout.write(f"  Peers:       {len(peers)}")
        self.stdout.write(f"  Single Node: {single_node}")
        self.stdout.write("")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Run the server
        try:
            asyncio.run(self._run_server(
                node_id=node_id,
                host=host,
                port=port,
                peers=peers,
                workers=workers,
                single_node=single_node,
            ))
        except KeyboardInterrupt:
            self.stdout.write("\nShutdown requested...")
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Server error: {e}"))
            sys.exit(1)

        self.stdout.write(self.style.SUCCESS("Tempora server stopped"))

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.stdout.write(f"\nReceived signal {signum}, shutting down...")
        self._shutdown_requested = True

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        with open(path) as f:
            return json.load(f)

    def _parse_peers(self, peers_str: str) -> List[tuple]:
        """Parse peers string into list of tuples."""
        if not peers_str:
            return []

        peers = []
        for peer in peers_str.split(","):
            peer = peer.strip()
            if not peer:
                continue
            parts = peer.split(":")
            if len(parts) == 3:
                node_id, host, port = parts
                peers.append((node_id, host, int(port)))
            else:
                logger.warning(f"Invalid peer format: {peer}")

        return peers

    async def _run_server(
        self,
        node_id: str,
        host: str,
        port: int,
        peers: List[tuple],
        workers: int,
        single_node: bool,
    ) -> None:
        """Run the Tempora server."""
        from tempora.coordination import CoordinationServer
        from tempora.distributed import DistributedCoordinator, DistributedConfig
        from tempora.models import ClusterMember, ClusterMemberRole, ClusterMemberStatus

        # Get cluster secret from settings
        cluster_secret = getattr(
            settings,
            "TEMPORA_CLUSTER_SECRET",
            os.environ.get("TEMPORA_CLUSTER_SECRET", settings.SECRET_KEY)
        )

        # Register this node
        member, created = ClusterMember.objects.update_or_create(
            instance_id=node_id,
            defaults={
                "host": host if host != "0.0.0.0" else "127.0.0.1",
                "port": port,
                "role": ClusterMemberRole.FOLLOWER,
                "status": ClusterMemberStatus.ACTIVE,
                "last_heartbeat": timezone.now(),
            },
        )

        if created:
            self.stdout.write(f"Registered new cluster member: {node_id}")
        else:
            self.stdout.write(f"Updated cluster member: {node_id}")

        # Start coordination server
        self.stdout.write("Starting coordination server...")
        self._server = CoordinationServer(
            instance_id=node_id,
            bind_host=host,
            bind_port=port,
            cluster_secret=cluster_secret,
        )
        await self._server.start()
        self.stdout.write(self.style.SUCCESS(f"Coordination server listening on {host}:{port}"))

        # Connect to peers
        if peers and not single_node:
            self.stdout.write("Connecting to peers...")
            for peer_id, peer_host, peer_port in peers:
                try:
                    await self._server.connect_to_peer(peer_id, peer_host, peer_port)
                    self.stdout.write(f"  Connected to {peer_id} ({peer_host}:{peer_port})")
                except Exception as e:
                    self.stdout.write(self.style.WARNING(
                        f"  Failed to connect to {peer_id}: {e}"
                    ))

        # Start distributed coordinator
        if not single_node and peers:
            self.stdout.write("Starting distributed coordinator...")
            config = DistributedConfig(
                node_id=node_id,
                cluster_secret=cluster_secret,
                election_timeout_min_ms=150,
                election_timeout_max_ms=300,
                heartbeat_interval_ms=50,
            )
            coordinator = DistributedCoordinator(
                config=config,
                coordination_server=self._server,
            )
            await coordinator.start()
            self.stdout.write(self.style.SUCCESS("Distributed coordinator started"))
        elif single_node:
            # In single-node mode, this node is always the leader
            member.role = ClusterMemberRole.LEADER
            member.save()
            self.stdout.write(self.style.SUCCESS("Running in single-node mode (this node is leader)"))

        # Start task scheduler
        self.stdout.write("Starting task scheduler...")
        await self._start_scheduler(workers)
        self.stdout.write(self.style.SUCCESS(f"Task scheduler started with {workers} workers"))

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("=" * 50))
        self.stdout.write(self.style.SUCCESS("Tempora server is running"))
        self.stdout.write(self.style.SUCCESS("=" * 50))
        self.stdout.write("")

        # Main loop - wait for shutdown
        while not self._shutdown_requested:
            await asyncio.sleep(1)

            # Update heartbeat
            member.last_heartbeat = timezone.now()
            member.save(update_fields=["last_heartbeat"])

        # Shutdown
        self.stdout.write("Shutting down...")
        await self._shutdown()

    async def _start_scheduler(self, workers: int) -> None:
        """Start the task scheduler."""
        # Import here to avoid circular imports
        from tempora.core import CognitiveScheduler

        try:
            self._scheduler = CognitiveScheduler(worker_count=workers)
            self._scheduler.start()
        except ImportError as e:
            # Core scheduler may have missing deps, fall back to simple execution
            self.stdout.write(self.style.WARNING(
                f"Full scheduler not available ({e}), using simple mode"
            ))
            self._scheduler = None

    async def _shutdown(self) -> None:
        """Graceful shutdown."""
        if self._scheduler:
            self._scheduler.stop(graceful=True)

        if self._server:
            await self._server.stop()

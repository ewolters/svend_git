"""
Tempora - Native High Availability Distributed Scheduler

A production-ready distributed task scheduler with native clustering,
leader election, state replication, and work distribution.

Features:
- Raft-based leader election with pre-vote optimization
- Distributed log replication with commit tracking
- Multiple work distribution strategies (round-robin, least-loaded, affinity)
- Split-brain prevention with fencing tokens
- Production hardening (rate limiting, TLS, health monitoring)

Standard: TEMPORA-HA-001

Quick Start:
    from tempora import schedule_task

    # One-time task
    schedule_task(
        name="send_email",
        func="myapp.tasks.send_email",
        args={"user_id": 123},
    )

    # Recurring task
    schedule_task(
        name="daily_cleanup",
        func="myapp.tasks.cleanup",
        cron="0 2 * * *"
    )
"""

__version__ = "1.0.0"
__author__ = "Tempora Contributors"
__license__ = "MIT OR Commercial"

# Simple API (recommended for most uses)
from tempora.scheduler import (
    schedule_task,
    get_task_status,
    cancel_task,
    get_cluster_health,
)

# Coordination layer
from tempora.coordination import (
    MessageType,
    Message,
    ProtocolError,
    CoordinationServer,
    CoordinationClient,
    PeerConnection,
    ConnectionPool,
    HeartbeatManager,
    HeartbeatConfig,
    PeerHealth,
    Connection,
    TransportLayer,
)

# Distributed layer
from tempora.distributed import (
    LeaderElector,
    ElectionConfig,
    ElectionState,
    NodeRole,
    VoteRequest,
    VoteResponse,
    StateReplicator,
    ReplicationConfig,
    FollowerProgress,
    LogEntryData,
    DistributedCoordinator,
    DistributedConfig,
    WorkDistributor,
    WorkDistributionConfig,
    DistributionStrategy,
    MemberLoad,
    TaskAssignment,
    NoAvailableMembersError,
    ElectionRateLimiter,
    ConnectionRateLimiter,
    SplitBrainDetector,
    ClusterHealthMonitor,
    TLSConfig,
    RateLimitConfig,
    PartitionState,
    check_production_readiness,
)

__all__ = [
    # Version
    "__version__",
    # Simple API
    "schedule_task",
    "get_task_status",
    "cancel_task",
    "get_cluster_health",
    # Protocol
    "MessageType",
    "Message",
    "ProtocolError",
    # Server/Client
    "CoordinationServer",
    "CoordinationClient",
    "PeerConnection",
    "ConnectionPool",
    # Heartbeat
    "HeartbeatManager",
    "HeartbeatConfig",
    "PeerHealth",
    # Transport
    "Connection",
    "TransportLayer",
    # Election
    "LeaderElector",
    "ElectionConfig",
    "ElectionState",
    "NodeRole",
    "VoteRequest",
    "VoteResponse",
    # Replication
    "StateReplicator",
    "ReplicationConfig",
    "FollowerProgress",
    "LogEntryData",
    # Coordinator
    "DistributedCoordinator",
    "DistributedConfig",
    # Work Distribution
    "WorkDistributor",
    "WorkDistributionConfig",
    "DistributionStrategy",
    "MemberLoad",
    "TaskAssignment",
    "NoAvailableMembersError",
    # Hardening
    "ElectionRateLimiter",
    "ConnectionRateLimiter",
    "SplitBrainDetector",
    "ClusterHealthMonitor",
    "TLSConfig",
    "RateLimitConfig",
    "PartitionState",
    "check_production_readiness",
]

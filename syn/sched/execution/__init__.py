"""
Synara CognitiveScheduler Execution Layer

Standard: SCH-003 (Worker Pool Architecture)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

Architecture:
    Scheduler --> ExecutionQueue --> WorkerPool --> TaskExecutor

Components:
    - ExecutionQueue: In-memory priority queue with batching
    - WorkerPool: Process pool management with auto-scaling
    - TaskExecutor: Isolated task execution with resource monitoring
    - ResourceClass: CPU-bound vs IO-bound worker separation

Features:
    - Concurrency limits (per-queue, per-tenant, global)
    - Retry delays with jitter
    - Isolated worker processes (memory leak containment)
    - Circuit breaker enforcement at dispatch level
    - Resource class separation (CPU/IO)
    - Automatic worker scaling
    - Health monitoring and heartbeats
"""

from .executor import ExecutionContext, ExecutionResult, ExecutionStatus, TaskExecutor
from .queue import ExecutionQueue, QueuedTask, QueueMetrics
from .resource_class import WORKER_CONFIGS, ResourceClass, WorkerConfig
from .worker_pool import WorkerPool, WorkerPoolMetrics, WorkerState

__all__ = [
    # Resource Classes
    "ResourceClass",
    "WorkerConfig",
    "WORKER_CONFIGS",
    # Execution Queue
    "ExecutionQueue",
    "QueuedTask",
    "QueueMetrics",
    # Task Executor
    "TaskExecutor",
    "ExecutionResult",
    "ExecutionContext",
    "ExecutionStatus",
    # Worker Pool
    "WorkerPool",
    "WorkerPoolMetrics",
    "WorkerState",
]

"""
Resource Class Definitions for Worker Pool

Standard: SCH-003 §3 (Resource Class Separation)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

Resource classes enable optimal worker allocation:
- CPU-bound: Matrix operations, encryption, compression
- IO-bound: HTTP calls, database queries, file I/O
- Mixed: Tasks that switch between CPU and IO

Isolation Benefits:
- CPU-bound workers don't starve IO-bound tasks
- IO-bound workers can handle more concurrency
- Memory limits prevent OOM conditions
- Process isolation contains memory leaks
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ResourceClass(Enum):
    """
    Worker resource classification for optimal scheduling.

    Standard: SCH-003 §3.1

    CPU_BOUND: Tasks primarily using CPU (compute-intensive)
        - Lower concurrency, more CPU time per worker
        - Examples: Data processing, encryption, ML inference

    IO_BOUND: Tasks primarily waiting on I/O (network, disk)
        - Higher concurrency, workers mostly idle waiting
        - Examples: API calls, database queries, file operations

    MIXED: Tasks with both CPU and I/O phases
        - Balanced concurrency, adaptive resource allocation
        - Examples: ETL pipelines, report generation

    LIGHTWEIGHT: Quick tasks with minimal resource needs
        - Very high concurrency, fast execution
        - Examples: Event emissions, metric updates, cache ops
    """

    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    MIXED = "mixed"
    LIGHTWEIGHT = "lightweight"


@dataclass
class WorkerConfig:
    """
    Configuration for a worker group by resource class.

    Standard: SCH-003 §3.2

    Attributes:
        resource_class: The resource classification
        min_workers: Minimum workers to maintain (scale-down floor)
        max_workers: Maximum workers allowed (scale-up ceiling)
        default_workers: Initial worker count at startup
        max_concurrent_per_worker: Tasks per worker (IO-bound can handle more)
        memory_limit_mb: Memory limit per worker process
        cpu_affinity: Pin workers to specific CPU cores (optional)
        timeout_multiplier: Multiplier for base task timeout
        process_isolation: Use separate processes (memory leak containment)
        recycle_after_tasks: Recycle worker after N tasks (memory leak prevention)
        health_check_interval_seconds: Heartbeat interval
    """

    resource_class: ResourceClass
    min_workers: int = 1
    max_workers: int = 10
    default_workers: int = 2
    max_concurrent_per_worker: int = 1
    memory_limit_mb: int = 512
    cpu_affinity: list | None = None
    timeout_multiplier: float = 1.0
    process_isolation: bool = True
    recycle_after_tasks: int = 1000
    health_check_interval_seconds: int = 30
    # Scaling thresholds
    scale_up_threshold: float = 0.8  # Scale up at 80% utilization
    scale_down_threshold: float = 0.2  # Scale down at 20% utilization
    scale_cooldown_seconds: int = 60  # Wait between scaling operations


# Default configurations per resource class (SCH-003 §3.3)
WORKER_CONFIGS: dict[ResourceClass, WorkerConfig] = {
    ResourceClass.CPU_BOUND: WorkerConfig(
        resource_class=ResourceClass.CPU_BOUND,
        min_workers=1,
        max_workers=4,  # Limited by CPU cores
        default_workers=2,
        max_concurrent_per_worker=1,  # CPU-bound = 1 task per worker
        memory_limit_mb=1024,  # Higher memory for compute
        timeout_multiplier=2.0,  # Allow longer execution
        process_isolation=True,  # Mandatory for memory containment
        recycle_after_tasks=500,  # Recycle more frequently
        health_check_interval_seconds=30,
        scale_up_threshold=0.7,
        scale_down_threshold=0.3,
        scale_cooldown_seconds=120,  # Slower scaling for CPU workers
    ),
    ResourceClass.IO_BOUND: WorkerConfig(
        resource_class=ResourceClass.IO_BOUND,
        min_workers=2,
        max_workers=20,  # Can scale high for I/O waiting
        default_workers=4,
        max_concurrent_per_worker=10,  # IO-bound can handle more
        memory_limit_mb=256,  # Lower memory per worker
        timeout_multiplier=1.5,  # Network timeouts
        process_isolation=True,
        recycle_after_tasks=2000,  # Recycle less frequently
        health_check_interval_seconds=15,  # Faster health checks
        scale_up_threshold=0.8,
        scale_down_threshold=0.2,
        scale_cooldown_seconds=30,  # Faster scaling for IO workers
    ),
    ResourceClass.MIXED: WorkerConfig(
        resource_class=ResourceClass.MIXED,
        min_workers=1,
        max_workers=8,
        default_workers=2,
        max_concurrent_per_worker=3,  # Balanced concurrency
        memory_limit_mb=512,
        timeout_multiplier=1.5,
        process_isolation=True,
        recycle_after_tasks=1000,
        health_check_interval_seconds=20,
        scale_up_threshold=0.75,
        scale_down_threshold=0.25,
        scale_cooldown_seconds=60,
    ),
    ResourceClass.LIGHTWEIGHT: WorkerConfig(
        resource_class=ResourceClass.LIGHTWEIGHT,
        min_workers=1,
        max_workers=50,  # Very high concurrency
        default_workers=4,
        max_concurrent_per_worker=50,  # Many lightweight tasks per worker
        memory_limit_mb=128,  # Minimal memory
        timeout_multiplier=0.5,  # Should complete quickly
        process_isolation=False,  # Threads OK for lightweight
        recycle_after_tasks=10000,  # Rarely need recycling
        health_check_interval_seconds=60,  # Less frequent checks
        scale_up_threshold=0.9,
        scale_down_threshold=0.1,
        scale_cooldown_seconds=15,  # Fast scaling
    ),
}


def get_worker_config(resource_class: ResourceClass) -> WorkerConfig:
    """Get the worker configuration for a resource class."""
    return WORKER_CONFIGS.get(resource_class, WORKER_CONFIGS[ResourceClass.MIXED])


def infer_resource_class(task_name: str, metadata: dict | None = None) -> ResourceClass:
    """
    Infer resource class from task name and metadata.

    Standard: SCH-003 §3.4

    Heuristics:
    - Tasks with "compute", "process", "transform" -> CPU_BOUND
    - Tasks with "fetch", "http", "api", "db", "query" -> IO_BOUND
    - Tasks with "emit", "metric", "log", "cache" -> LIGHTWEIGHT
    - Default: MIXED

    Can be overridden by explicit metadata["resource_class"].
    """
    # Explicit override
    if metadata and "resource_class" in metadata:
        rc_value = metadata["resource_class"]
        if isinstance(rc_value, ResourceClass):
            return rc_value
        try:
            return ResourceClass(rc_value)
        except ValueError:
            pass

    task_lower = task_name.lower()

    # CPU-bound patterns
    cpu_patterns = [
        "compute",
        "process",
        "transform",
        "encrypt",
        "decrypt",
        "compress",
        "decompress",
        "render",
        "analyze",
        "calculate",
        "ml.",
        "ai.",
        "matrix",
        "algorithm",
    ]
    if any(pattern in task_lower for pattern in cpu_patterns):
        return ResourceClass.CPU_BOUND

    # IO-bound patterns
    io_patterns = [
        "fetch",
        "http",
        "api",
        "request",
        "db.",
        "query",
        "database",
        "file",
        "upload",
        "download",
        "stream",
        "external",
        "webhook",
        "notification",
        "email",
        "sms",
    ]
    if any(pattern in task_lower for pattern in io_patterns):
        return ResourceClass.IO_BOUND

    # Lightweight patterns
    lightweight_patterns = [
        "emit",
        "metric",
        "log",
        "cache",
        "invalidate",
        "ping",
        "heartbeat",
        "health",
        "status",
        "counter",
    ]
    if any(pattern in task_lower for pattern in lightweight_patterns):
        return ResourceClass.LIGHTWEIGHT

    # Default to mixed
    return ResourceClass.MIXED

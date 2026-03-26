"""
Centralized Configuration for CognitiveScheduler (Tempora)

Standard: SCH-001 §4.1 (Configuration Management)
Codename: Tempora
Author: Systems Architect
Version: 1.0
Date: 2025-12-10

This module consolidates all scheduler configuration into a single location.
Each subsystem's config is re-exported here for convenience, with a master
SchedulerConfig that aggregates all settings.

Usage:
    from syn.sched.config import SchedulerConfig, get_default_config

    config = get_default_config()
    scheduler = CognitiveScheduler(config=config)

Configuration Hierarchy:
    SchedulerConfig
    ├── backpressure: BackpressureConfig (SCH-004)
    ├── temporal: TemporalControllerConfig (SCH-006)
    ├── dashboard: DashboardConfig (SCH-005)
    ├── worker: WorkerPoolConfig (SCH-003)
    └── core settings (SCH-001)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Type imports only - avoid circular imports and platform-specific issues
if TYPE_CHECKING:
    from syn.sched.backpressure.controller import BackpressureConfig
    from syn.sched.dashboard.service import DashboardConfig
    from syn.sched.temporal.controller import TemporalControllerConfig

# These are safe to import at module level
from syn.sched.types import CircuitBreakerConfig, RetryConfig, TenantQuota

# =============================================================================
# WORKER POOL CONFIGURATION (SCH-003)
# =============================================================================


@dataclass
class WorkerPoolConfig:
    """
    Configuration for the WorkerPool.

    Standard: SCH-003 §6.1
    """

    # Worker group settings per resource class (dict with string keys for portability)
    worker_configs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Auto-scaling
    auto_scale: bool = True
    scale_interval_seconds: int = 30

    # Health monitoring
    heartbeat_interval_seconds: int = 5
    worker_timeout_seconds: int = 30

    # Recycling
    enable_worker_recycling: bool = True
    recycle_after_tasks: int = 1000

    def __post_init__(self):
        """Initialize default worker configs if not provided."""
        if not self.worker_configs:
            # Set defaults without importing platform-specific modules
            self.worker_configs = {
                "cpu_bound": {
                    "min_workers": 1,
                    "max_workers": 4,
                    "memory_limit_mb": 1024,
                },
                "io_bound": {
                    "min_workers": 2,
                    "max_workers": 16,
                    "memory_limit_mb": 256,
                },
                "mixed": {"min_workers": 1, "max_workers": 8, "memory_limit_mb": 512},
                "lightweight": {
                    "min_workers": 1,
                    "max_workers": 4,
                    "memory_limit_mb": 128,
                },
            }

    def to_dict(self) -> dict[str, Any]:
        return {
            "auto_scale": self.auto_scale,
            "scale_interval_seconds": self.scale_interval_seconds,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "worker_timeout_seconds": self.worker_timeout_seconds,
            "enable_worker_recycling": self.enable_worker_recycling,
            "recycle_after_tasks": self.recycle_after_tasks,
            "worker_configs": self.worker_configs,
        }


# =============================================================================
# SCHEDULER CORE CONFIGURATION (SCH-001)
# =============================================================================


@dataclass
class SchedulerConfig:
    """
    Master configuration for CognitiveScheduler.

    Standard: SCH-001 §4 (Scheduler Architecture)

    This aggregates all subsystem configurations into a single object
    for convenient initialization and serialization.
    """

    # ==========================================================================
    # CORE SETTINGS (SCH-001)
    # ==========================================================================

    # Task queue settings
    default_queue: str = "core"
    max_queue_depth: int = 10000
    max_queue_per_tenant: int = 1000

    # Cascade limits per SCH-001 §5
    max_cascade_depth: int = 5
    cascade_budget: int = 100

    # Correlation
    enable_correlation_tracking: bool = True

    # Schedule processing
    schedule_check_interval_seconds: int = 60
    enable_schedule_processing: bool = True

    # Circuit breaker defaults
    circuit_breaker_config: CircuitBreakerConfig = field(
        default_factory=CircuitBreakerConfig
    )

    # Default retry config
    default_retry_config: RetryConfig = field(default_factory=RetryConfig)

    # ==========================================================================
    # SUBSYSTEM CONFIGURATIONS
    # ==========================================================================
    # Note: These are stored as Optional and lazily initialized via properties
    # to avoid importing platform-specific modules at config load time.

    # Backpressure (SCH-004) - use get_backpressure_config() for typed access
    _backpressure: dict[str, Any] | None = None

    # Temporal (SCH-006) - use get_temporal_config() for typed access
    _temporal: dict[str, Any] | None = None

    # Dashboard (SCH-005) - use get_dashboard_config() for typed access
    _dashboard: dict[str, Any] | None = None

    # Worker Pool (SCH-003)
    worker_pool: WorkerPoolConfig = field(default_factory=WorkerPoolConfig)

    # ==========================================================================
    # FEATURE FLAGS
    # ==========================================================================

    enable_backpressure: bool = True
    enable_temporal_reflexes: bool = True
    enable_dashboard: bool = True
    enable_circuit_breakers: bool = True
    enable_dead_letter_queue: bool = True
    enable_metrics: bool = True

    # ==========================================================================
    # CELERY MIGRATION (SCH-001 §8)
    # ==========================================================================

    # Shadow mode: run both Celery and CognitiveScheduler in parallel
    celery_shadow_mode: bool = False
    celery_shadow_tasks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            # Core settings
            "default_queue": self.default_queue,
            "max_queue_depth": self.max_queue_depth,
            "max_queue_per_tenant": self.max_queue_per_tenant,
            "max_cascade_depth": self.max_cascade_depth,
            "cascade_budget": self.cascade_budget,
            "enable_correlation_tracking": self.enable_correlation_tracking,
            "schedule_check_interval_seconds": self.schedule_check_interval_seconds,
            "enable_schedule_processing": self.enable_schedule_processing,
            "circuit_breaker_config": self.circuit_breaker_config.__dict__,
            "default_retry_config": self.default_retry_config.__dict__,
            # Subsystem configs (as dicts)
            "backpressure": self._backpressure or {},
            "temporal": self._temporal or {},
            "dashboard": self._dashboard or {},
            "worker_pool": self.worker_pool.to_dict(),
            # Feature flags
            "enable_backpressure": self.enable_backpressure,
            "enable_temporal_reflexes": self.enable_temporal_reflexes,
            "enable_dashboard": self.enable_dashboard,
            "enable_circuit_breakers": self.enable_circuit_breakers,
            "enable_dead_letter_queue": self.enable_dead_letter_queue,
            "enable_metrics": self.enable_metrics,
            # Migration
            "celery_shadow_mode": self.celery_shadow_mode,
            "celery_shadow_tasks": self.celery_shadow_tasks,
        }

    def get_backpressure_config(self) -> BackpressureConfig:
        """Get backpressure configuration (lazy import, cached)."""
        from syn.sched.backpressure.controller import BackpressureConfig

        if (
            not hasattr(self, "_backpressure_instance")
            or self._backpressure_instance is None
        ):
            self._backpressure_instance = BackpressureConfig(
                **(self._backpressure or {})
            )
        return self._backpressure_instance

    def get_temporal_config(self) -> TemporalControllerConfig:
        """Get temporal controller configuration (lazy import, cached)."""
        from syn.sched.temporal.controller import TemporalControllerConfig

        if not hasattr(self, "_temporal_instance") or self._temporal_instance is None:
            self._temporal_instance = TemporalControllerConfig(**(self._temporal or {}))
        return self._temporal_instance

    def get_dashboard_config(self) -> DashboardConfig:
        """Get dashboard configuration (lazy import, cached)."""
        from syn.sched.dashboard.service import DashboardConfig

        if not hasattr(self, "_dashboard_instance") or self._dashboard_instance is None:
            self._dashboard_instance = DashboardConfig(**(self._dashboard or {}))
        return self._dashboard_instance

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchedulerConfig:
        """Create configuration from dictionary."""
        config = cls()

        # Core settings
        for key in [
            "default_queue",
            "max_queue_depth",
            "max_queue_per_tenant",
            "max_cascade_depth",
            "cascade_budget",
            "enable_correlation_tracking",
            "schedule_check_interval_seconds",
            "enable_schedule_processing",
        ]:
            if key in data:
                setattr(config, key, data[key])

        # Feature flags
        for key in [
            "enable_backpressure",
            "enable_temporal_reflexes",
            "enable_dashboard",
            "enable_circuit_breakers",
            "enable_dead_letter_queue",
            "enable_metrics",
            "celery_shadow_mode",
        ]:
            if key in data:
                setattr(config, key, data[key])

        if "celery_shadow_tasks" in data:
            config.celery_shadow_tasks = data["celery_shadow_tasks"]

        return config

    @classmethod
    def from_environment(cls) -> SchedulerConfig:
        """Create configuration from environment variables."""
        config = cls()

        # Core settings
        if "SYNARA_SCHED_MAX_QUEUE_DEPTH" in os.environ:
            config.max_queue_depth = int(os.environ["SYNARA_SCHED_MAX_QUEUE_DEPTH"])

        if "SYNARA_SCHED_MAX_CASCADE_DEPTH" in os.environ:
            config.max_cascade_depth = int(os.environ["SYNARA_SCHED_MAX_CASCADE_DEPTH"])

        if "SYNARA_SCHED_CASCADE_BUDGET" in os.environ:
            config.cascade_budget = int(os.environ["SYNARA_SCHED_CASCADE_BUDGET"])

        # Feature flags
        if "SYNARA_SCHED_ENABLE_BACKPRESSURE" in os.environ:
            config.enable_backpressure = (
                os.environ["SYNARA_SCHED_ENABLE_BACKPRESSURE"].lower() == "true"
            )

        if "SYNARA_SCHED_ENABLE_TEMPORAL" in os.environ:
            config.enable_temporal_reflexes = (
                os.environ["SYNARA_SCHED_ENABLE_TEMPORAL"].lower() == "true"
            )

        if "SYNARA_SCHED_CELERY_SHADOW" in os.environ:
            config.celery_shadow_mode = (
                os.environ["SYNARA_SCHED_CELERY_SHADOW"].lower() == "true"
            )

        return config


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================


def get_default_config() -> SchedulerConfig:
    """
    Get the default scheduler configuration.

    Loads from environment variables first, then applies defaults.
    """
    return SchedulerConfig.from_environment()


def get_production_config() -> SchedulerConfig:
    """
    Get production-optimized configuration.

    Tuned for:
    - Higher throughput
    - Aggressive backpressure
    - Full observability
    """
    config = SchedulerConfig()

    # Production queue settings
    config.max_queue_depth = 50000
    config.max_queue_per_tenant = 5000

    # Production backpressure thresholds (stored as dict)
    config._backpressure = {
        "queue_depth_threshold": 5000,
        "dlq_threshold": 500,
        "enable_emergency_shutdown": True,
    }

    # Production temporal settings (stored as dict)
    config._temporal = {
        "evaluation_interval_seconds": 2.0,
    }

    # All features enabled
    config.enable_backpressure = True
    config.enable_temporal_reflexes = True
    config.enable_dashboard = True
    config.enable_circuit_breakers = True
    config.enable_dead_letter_queue = True
    config.enable_metrics = True

    return config


def get_development_config() -> SchedulerConfig:
    """
    Get development configuration.

    Tuned for:
    - Lower resource usage
    - Faster iteration
    - Verbose logging
    """
    config = SchedulerConfig()

    # Lower limits for dev
    config.max_queue_depth = 1000
    config.max_queue_per_tenant = 100

    # Relaxed backpressure (stored as dict)
    config._backpressure = {
        "queue_depth_threshold": 500,
        "enable_emergency_shutdown": False,
    }

    # Faster temporal evaluation for testing (stored as dict)
    config._temporal = {
        "evaluation_interval_seconds": 1.0,
    }

    return config


def get_test_config() -> SchedulerConfig:
    """
    Get test configuration.

    Tuned for:
    - Fast execution
    - Deterministic behavior
    - Minimal side effects
    """
    config = SchedulerConfig()

    # Minimal limits for tests
    config.max_queue_depth = 100
    config.max_queue_per_tenant = 10
    config.max_cascade_depth = 3
    config.cascade_budget = 10

    # Disable background processing
    config.enable_schedule_processing = False

    # Fast intervals for tests (stored as dict)
    config._temporal = {
        "evaluation_interval_seconds": 0.1,
    }

    return config


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Master config
    "SchedulerConfig",
    "WorkerPoolConfig",
    # Types (from syn.sched.types)
    "RetryConfig",
    "CircuitBreakerConfig",
    "TenantQuota",
    # Factory functions
    "get_default_config",
    "get_production_config",
    "get_development_config",
    "get_test_config",
]

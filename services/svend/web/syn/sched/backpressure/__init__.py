"""
Synara CognitiveScheduler Backpressure System

Standard: SCH-004 (Backpressure & Auto-Throttling)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

Architecture:
    SystemHealthMonitor → BackpressureController → ThrottlePolicy → Scheduler

The backpressure system provides:
- Dynamic throttling based on queue depth
- DLQ-based task pausing
- Governance-aware confidence degradation
- Circuit breaker feedback loops
- Cascade prevention via load shedding

This transforms Synara from:
    "I run tasks" → "I regulate systemic load"

Throttle Levels:
    NONE (0.0)     - Normal operation
    LIGHT (0.25)   - Minor slowdown
    MODERATE (0.5) - Significant reduction
    HEAVY (0.75)   - Emergency mode
    CRITICAL (1.0) - Full stop (circuit open)
"""

from .health import SystemHealthMonitor, HealthMetrics, HealthStatus
from .throttle import ThrottlePolicy, ThrottleLevel, ThrottleDecision
from .controller import BackpressureController, BackpressureConfig

__all__ = [
    # Health Monitoring
    "SystemHealthMonitor",
    "HealthMetrics",
    "HealthStatus",
    # Throttle Policy
    "ThrottlePolicy",
    "ThrottleLevel",
    "ThrottleDecision",
    # Controller
    "BackpressureController",
    "BackpressureConfig",
]

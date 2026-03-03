"""
Synara Governance-Linked Temporal Reflexes

Standard: SCH-006 (Governance-Linked Temporal Reflexes)
Author: Systems Architect
Version: 1.0
Date: 2025-12-08

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    ADAPTIVE TEMPORAL CONTROL LOOP                        │
    │                                                                          │
    │   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐               │
    │   │ Governance  │ ── │ Temporal     │ ── │ Scheduler    │               │
    │   │ Layer       │    │ Controller   │    │ Layer        │               │
    │   └─────────────┘    └──────┬───────┘    └──────────────┘               │
    │          ▲                   │                   │                       │
    │          │                   ▼                   ▼                       │
    │   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐               │
    │   │ Cognition   │ ◀─ │ Temporal     │ ── │ Backpressure │               │
    │   │ Layer       │    │ Reflex       │    │ Controller   │               │
    │   └─────────────┘    └──────────────┘    └──────────────┘               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

This package enables governance rules to affect time:
- Pause schedules when risk > threshold
- Accelerate health checks during incidents
- Degrade priority for unstable tenants
- Schedule compensating tasks (self-healing)
- Dynamic TTL adjustments based on load

NO OTHER SYSTEM DOES THIS.

The temporal reflex system creates a feedback loop where:
1. Governance observes system state (risk, health, stability)
2. Temporal policies define timing responses
3. Temporal reflexes execute schedule/priority adjustments
4. The scheduler adapts in real-time
5. Changes feed back into governance for evaluation

This is Synara's "temporal cortex" - adaptive timing intelligence.
"""

from .policy import (
    TemporalPolicy,
    TemporalPolicyRule,
    TemporalAction,
    TemporalTrigger,
    TriggerType,
    ActionType,
)
from .reflex import (
    TemporalReflex,
    ReflexOutcome,
    ReflexState,
    CompensatingTask,
)
from .controller import (
    TemporalController,
    TemporalControllerConfig,
    TemporalMetrics,
)

__all__ = [
    # Policy
    "TemporalPolicy",
    "TemporalPolicyRule",
    "TemporalAction",
    "TemporalTrigger",
    "TriggerType",
    "ActionType",
    # Reflex
    "TemporalReflex",
    "ReflexOutcome",
    "ReflexState",
    "CompensatingTask",
    # Controller
    "TemporalController",
    "TemporalControllerConfig",
    "TemporalMetrics",
]

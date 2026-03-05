"""
Synara Cognitive Scheduler Module (SCH-001/002)
===============================================
Codename: TEMPORA

Cognitive scheduling system implementing priority-based task execution,
circuit breaker patterns, dead letter queues, and cascade management.

Standard:     SCH-001 (Patterns), SCH-002 (Implementation)
Compliance:   ISO 9001:2015 §9.5, SOC 2 CC7.2, NIST SP 800-53 SC-5
Location:     syn/sched/
Version:      1.0.0
Codename:     Tempora (named by the feline CTO)

Features:
---------
- CognitiveTask: Unit of work with cognitive metadata (SCH-001 §5)
- CognitiveScheduler: Main scheduler orchestrator (SCH-001 §5)
- CognitiveWorker: Task execution engine (SCH-001 §6)
- CircuitBreakerState: Per-service circuit breaker (SCH-002 §4)
- DeadLetterEntry: Failed tasks for manual review (SCH-002 §13)
- Schedule: Cron/interval/one-time scheduling (SCH-001 §6)

Celery Transition:
-----------------
This scheduler is designed to replace Celery with a cognitive-aware system.
Sunset date: 2026-01-15 per SCH-002 §celery_transition.

Usage:
------
    from syn.sched import (
        CognitiveScheduler,
        CognitiveTask,
        task,
        TaskPriority,
        QueueType,
    )

    # Register a task handler
    @task("myapp.process_order", queue=QueueType.CORE)
    def process_order(payload: dict, context: TaskContext) -> dict:
        order_id = payload["order_id"]
        # Process the order
        return {"status": "processed", "order_id": order_id}

    # Submit a task
    scheduler = CognitiveScheduler()
    task = scheduler.submit(
        task_name="myapp.process_order",
        payload={"order_id": "12345"},
        tenant_id=tenant_uuid,
        priority=TaskPriority.HIGH,
        urgency=0.8,
    )

    # Start the scheduler
    scheduler.start()
"""

__version__ = "1.0.0"
__standard__ = "SCH-001"
__codename__ = "Tempora"

# Convenience imports - import models and utilities directly from submodules:
#   from syn.sched.models import CognitiveTask, TaskExecution, Schedule
#   from syn.sched.types import TaskState, TaskPriority, QueueType
#   from syn.sched.core import CognitiveScheduler, task
#   from syn.sched.events import emit_scheduler_event, SCHEDULER_EVENTS
#   from syn.sched.config import SchedulerConfig, get_default_config
#   from syn.sched.exceptions import QuotaExceededError, ThrottledError

# Standard-organized exports per SCH-001 §4.1
__all__ = [
    # Version info
    "__version__",
    "__standard__",
    "__codename__",
]

# Scheduler System Audit Report
**Date:** 2025-12-27
**System:** syn/sched/ (Cognitive Scheduler)
**Auditor:** Claude Sonnet 4.5
**Standards:** SCH-001, SCH-002, SCH-003, SCH-004, SCH-006

---

## Executive Summary

This audit examined the Synara Cognitive Scheduler system (syn/sched/) for bugs, errors, invariant violations, and potential vulnerabilities. The system is generally well-architected with strong adherence to standards, but **21 issues** were identified ranging from critical race conditions to minor code quality concerns.

**Critical Issues:** 4
**High Priority:** 6
**Medium Priority:** 7
**Low Priority:** 4

---

## Critical Issues (Fix Immediately)

### C1. Race Condition in fetch_next_task (core.py:588-638)
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\core.py`
**Lines:** 588-638
**Severity:** CRITICAL
**Standard Violation:** SCH-002 §5 (Concurrency Safety)

**Issue:**
The `fetch_next_task` method has a race condition window between reading task state and transitioning to SCHEDULED. Multiple workers could potentially select the same task.

**Code:**
```python
def fetch_next_task(self, queues, worker_id):
    with transaction.atomic():
        task = (
            CognitiveTask.objects.select_for_update(skip_locked=True)
            .filter(...)
            .first()
        )
        if task:
            # Check tenant concurrent task quota
            if not self._check_tenant_quota(task.tenant_id, "concurrent_tasks"):
                return None  # PROBLEM: Task is still locked but not transitioned

            task.transition_to(TaskState.SCHEDULED)
        return task
```

**Problem:** If quota check fails, the task remains in PENDING but was already selected by `select_for_update`. Another worker could pick it up after the lock releases.

**Fix:**
```python
def fetch_next_task(self, queues, worker_id):
    with transaction.atomic():
        task = (
            CognitiveTask.objects.select_for_update(skip_locked=True)
            .filter(...)
            .first()
        )
        if task:
            # Check tenant concurrent task quota BEFORE locking
            current_running = CognitiveTask.objects.filter(
                tenant_id=task.tenant_id,
                state=TaskState.RUNNING.value,
            ).count()

            quota = self._get_tenant_quota(task.tenant_id)
            if current_running >= quota.max_concurrent_tasks:
                # Don't select this task - it stays pending
                # Another task from different tenant can be selected
                return None

            # Now safe to transition
            task.transition_to(TaskState.SCHEDULED)
        return task
```

---

### C2. Missing Django Model Import in core.py
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\core.py`
**Line:** 1302
**Severity:** CRITICAL
**Standard Violation:** SCH-001 §4 (Code Organization)

**Issue:**
Django's `models` module is imported at the END of the file (line 1302), but it's used throughout the file in queries (lines 607-628). This creates a circular dependency risk and violates Python import conventions.

**Code:**
```python
# Line 607-628: Using models.Q but models not imported yet
.filter(
    models.Q(scheduled_at__isnull=True)
    | models.Q(scheduled_at__lte=now)
)

# Line 1302: Import is here (too late!)
from django.db import models
```

**Fix:** Move the import to the top of the file with other Django imports (around line 45).

---

### C3. _check_tenant_quota Returns Inconsistent Types
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\core.py`
**Lines:** 533-576
**Severity:** CRITICAL
**Standard Violation:** SCH-002 §7 (Type Safety)

**Issue:**
The method signature and implementation are inconsistent:
- For "queue_depth": Raises `QuotaExceededError`
- For "concurrent_tasks": Returns `False`
- Implicitly returns `None` on success for queue_depth

**Code:**
```python
def _check_tenant_quota(self, tenant_id, quota_type) -> None:  # Says returns None
    if quota_type == "queue_depth":
        # ...
        if current >= quota.max_queue_depth:
            raise QuotaExceededError(...)  # Raises exception

    elif quota_type == "concurrent_tasks":
        # ...
        if current >= quota.max_concurrent_tasks:
            return False  # Returns boolean

    return True  # Also returns boolean
```

**Fix:**
```python
def _check_tenant_quota(self, tenant_id: uuid.UUID, quota_type: str) -> bool:
    """
    Check and enforce tenant quota.

    Returns:
        True if within quota, False if quota exceeded

    Raises:
        QuotaExceededError: For hard quota limits (queue_depth)
    """
    quota = self._get_tenant_quota(tenant_id)

    if quota_type == "queue_depth":
        current = CognitiveTask.objects.filter(
            tenant_id=tenant_id,
            state__in=[TaskState.PENDING.value, TaskState.SCHEDULED.value],
        ).count()

        if current >= quota.max_queue_depth:
            # Hard limit - reject submission
            emit_scheduler_event(...)
            raise QuotaExceededError(...)

    elif quota_type == "concurrent_tasks":
        current = CognitiveTask.objects.filter(
            tenant_id=tenant_id,
            state=TaskState.RUNNING.value,
        ).count()

        if current >= quota.max_concurrent_tasks:
            # Soft limit - don't fetch this task now
            return False

    return True
```

---

### C4. Unhandled Exception in _execute_task (core.py:1079-1306)
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\core.py`
**Lines:** 1079-1306
**Severity:** CRITICAL
**Standard Violation:** SCH-002 §8 (Error Handling)

**Issue:**
If an exception occurs during handler execution, the circuit breaker state is updated, but the task state is NOT updated in the database. The task remains in RUNNING state forever.

**Code:**
```python
def _execute_task(self, task):
    # ... setup ...

    try:
        handler = TaskRegistry.get_handler(task.task_name)
        if handler is None:
            raise ValueError(f"No handler registered for task: {task.task_name}")

        result = handler(task.payload, context)
        success = True

        if circuit_breaker_service:
            circuit.record_success()

    except Exception as e:
        error_message = str(e)
        error_type = self._classify_error(e)
        error_traceback = traceback.format_exc()

        if circuit_breaker_service:
            circuit.record_failure()

    # ... execution.complete() ...

    if success:
        self._handle_success(task, result, duration_ms)
    else:
        self._handle_failure(task, error_message, error_type, duration_ms)

    # PROBLEM: If exception in handler, task.state is updated via _handle_failure
    # But if exception in setup (before handler), task remains RUNNING
```

**Missing Guard:** No try-catch around the entire execution block to ensure task state is always updated.

**Fix:**
```python
def _execute_task(self, task: CognitiveTask) -> None:
    self._current_task = task
    handler = TaskRegistry.get_handler(task.task_name)
    metadata = TaskRegistry.get_metadata(task.task_name)

    # Circuit breaker check
    circuit_breaker_service = metadata.get("circuit_breaker")
    circuit = None

    try:
        if circuit_breaker_service:
            circuit = CircuitBreakerState.get_or_create_for_service(...)
            can_execute, reason = circuit.can_execute()
            if not can_execute:
                # Reschedule task
                task.next_retry_at = timezone.now() + timedelta(seconds=30)
                task.state = TaskState.PENDING.value
                task.save()
                return

        # Transition to running
        task.transition_to(TaskState.RUNNING)
        task.attempts += 1
        task.save()

        # Create execution record
        execution = TaskExecution.objects.create(...)

        # Execute handler
        start_time = time.time()
        success = False
        result = None
        error_message = None
        error_type = None
        error_traceback = None

        try:
            if handler is None:
                raise ValueError(f"No handler registered for task: {task.task_name}")

            context = task.to_context()
            result = handler(task.payload, context)
            success = True

            if circuit:
                circuit.record_success()

        except Exception as e:
            error_message = str(e)
            error_type = self._classify_error(e)
            error_traceback = traceback.format_exc()

            if circuit:
                circuit.record_failure()

        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Complete execution record
        execution.complete(
            success=success,
            result=result,
            error_message=error_message,
            error_type=error_type,
            error_traceback=error_traceback,
        )

        # Handle result
        if success:
            self._handle_success(task, result, duration_ms)
        else:
            self._handle_failure(task, error_message, error_type, duration_ms)

    except Exception as e:
        # Critical failure - ensure task is marked as failed
        logger.error(f"Critical failure in _execute_task: {e}", exc_info=True)
        try:
            task.state = TaskState.FAILURE.value
            task.error_message = f"Critical execution error: {str(e)}"
            task.error_type = type(e).__name__
            task.save()
        except Exception as save_error:
            logger.error(f"Failed to save task state after critical error: {save_error}")

    finally:
        self._current_task = None
```

---

## High Priority Issues

### H1. Missing Null Check in schedule_retry (models.py:411-434)
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\models.py`
**Lines:** 411-434
**Severity:** HIGH
**Standard Violation:** SCH-002 §12 (Retry Logic)

**Issue:**
The method increments `self.attempts` but doesn't check if it would exceed `self.max_attempts` BEFORE incrementing. This could cause off-by-one errors.

**Code:**
```python
def schedule_retry(self) -> Optional[datetime]:
    self.attempts += 1  # Increment first

    if self.attempts >= self.max_attempts:  # Then check
        self.transition_to(TaskState.DEAD_LETTERED)
        return None
```

**Fix:**
```python
def schedule_retry(self) -> Optional[datetime]:
    """Schedule next retry attempt per SCH-002 §12."""
    # Check BEFORE incrementing
    if self.attempts >= self.max_attempts:
        logger.warning(f"Task {self.id} exhausted retries ({self.attempts}/{self.max_attempts})")
        self.transition_to(TaskState.DEAD_LETTERED)
        return None

    # Now safe to increment
    self.attempts += 1

    config = RetryConfig(...)
    delay = config.get_delay(self.attempts)
    self.next_retry_at = timezone.now() + delay
    self.transition_to(TaskState.RETRYING)

    logger.info(
        f"Task {self.id} scheduled for retry {self.attempts}/{self.max_attempts} "
        f"in {delay.total_seconds()}s"
    )

    return self.next_retry_at
```

---

### H2. DeadLetterEntry.create_from_task Missing Transaction Wrapper
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\models.py`
**Lines:** 1060-1091
**Severity:** HIGH
**Standard Violation:** SCH-002 §13 (DLQ Handling)

**Issue:**
The method creates a DLQ entry but doesn't wrap the operation in a transaction. If the entry creation succeeds but the caller's subsequent operations fail, we have an orphaned DLQ entry.

**Fix:**
```python
@classmethod
@transaction.atomic
def create_from_task(cls, task: CognitiveTask, failure_reason: str) -> "DeadLetterEntry":
    """Create DLQ entry from a failed task."""
    # ... implementation ...
```

---

### H3. process_schedules Missing Deadlock Protection
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\core.py`
**Lines:** 644-735
**Severity:** HIGH
**Standard Violation:** SCH-001 §6 (Concurrency)

**Issue:**
The method uses `select_for_update(skip_locked=True)` but doesn't specify a timeout. On high contention, this could cause indefinite waits.

**Code:**
```python
with transaction.atomic():
    due_schedules = list(Schedule.objects.filter(
        enabled=True,
        next_run_at__lte=now,
    ).select_for_update(skip_locked=True))  # No timeout!
```

**Fix:**
```python
with transaction.atomic():
    # Add timeout to prevent indefinite blocking
    due_schedules = list(Schedule.objects.filter(
        enabled=True,
        next_run_at__lte=now,
    ).select_for_update(skip_locked=True, nowait=False))

    # Also add a limit to prevent processing too many at once
    due_schedules = due_schedules[:50]  # Process max 50 schedules per cycle
```

---

### H4. CircuitBreakerState.can_execute Missing Atomic Transition
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\models.py`
**Lines:** 1269-1296
**Severity:** HIGH
**Standard Violation:** SCH-002 §4 (Circuit Breaker)

**Issue:**
The method checks state and transitions to HALF_OPEN, but doesn't wrap this in a transaction. Multiple threads could transition simultaneously.

**Code:**
```python
def can_execute(self) -> Tuple[bool, str]:
    if self.state == CircuitState.OPEN.value:
        if self.opened_at:
            recovery_time = self.opened_at + timedelta(seconds=self.recovery_timeout_seconds)
            if timezone.now() >= recovery_time:
                self._half_open_circuit()  # Not atomic!
                return True, "Circuit half-open (testing)"
```

**Fix:**
```python
def can_execute(self) -> Tuple[bool, str]:
    """Check if execution is allowed (thread-safe)."""
    with transaction.atomic():
        # Refresh from database to get latest state
        self.refresh_from_db()

        if self.state == CircuitState.CLOSED.value:
            return True, "Circuit closed"

        if self.state == CircuitState.OPEN.value:
            if self.opened_at:
                recovery_time = self.opened_at + timedelta(
                    seconds=self.recovery_timeout_seconds
                )
                if timezone.now() >= recovery_time:
                    # Atomic transition
                    self._half_open_circuit()
                    return True, "Circuit half-open (testing)"
            return False, "Circuit open"

        # ... rest of logic ...
```

---

### H5. Missing Resource Cleanup in TaskExecutor
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\execution\executor.py`
**Lines:** 322-573
**Severity:** HIGH
**Standard Violation:** SCH-003 §5 (Resource Management)

**Issue:**
The `TaskExecutor` creates thread and process pools but doesn't implement `__enter__` and `__exit__` for context manager support. This makes it easy to leak resources.

**Fix:**
```python
class TaskExecutor:
    def __init__(self, ...):
        # ... existing init ...

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.shutdown(wait=True)
        return False

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown executor pools."""
        try:
            self._thread_pool.shutdown(wait=wait)
        except Exception as e:
            logger.error(f"Error shutting down thread pool: {e}")

        if self._process_pool:
            try:
                self._process_pool.shutdown(wait=wait)
            except Exception as e:
                logger.error(f"Error shutting down process pool: {e}")
```

**Usage:**
```python
with TaskExecutor(worker_id="worker-1") as executor:
    result = executor.execute(task, handler)
# Automatically cleaned up
```

---

### H6. BackpressureController Decision Cache Not Thread-Safe
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\backpressure\controller.py`
**Lines:** 227-229
**Severity:** HIGH
**Standard Violation:** SCH-004 §4 (Thread Safety)

**Issue:**
The decision cache is checked and updated without holding the lock consistently.

**Code:**
```python
def should_schedule(self, ...):
    with self._lock:
        # Check cache
        if not bypass_cache and self._is_cache_valid():
            cached = self._cached_decision  # Read without lock
```

**Fix:**
```python
def should_schedule(self, ...):
    with self._lock:
        # Check cache (now all cache access is locked)
        if not bypass_cache and self._is_cache_valid():
            cached = self._cached_decision
            if task_name or priority is not None or is_batch:
                return self._apply_task_restrictions(cached, task_name, priority, is_batch)
            return cached

        # ... rest of method with lock held ...
```

---

## Medium Priority Issues

### M1. Incorrect Query Filter in ExecutionQueue.fetch_batch
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\execution\queue.py`
**Lines:** 362-421
**Severity:** MEDIUM
**Standard Violation:** SCH-003 §4 (Queue Management)

**Issue:**
The retry filter logic uses OR incorrectly - it fetches ALL pending tasks OR ALL retrying tasks, not the intended filtered subset.

**Code:**
```python
# Build query
queryset = CognitiveTask.objects.filter(
    state__in=[TaskState.PENDING.value, TaskState.RETRYING.value],
).filter(
    scheduled_at__lte=now,
).exclude(
    id__in=list(self._task_ids),
).order_by("-priority_score", "created_at")

# Retry filter (WRONG!)
queryset = queryset.filter(
    state=TaskState.PENDING.value,
) | queryset.filter(
    state=TaskState.RETRYING.value,
    next_retry_at__lte=now,
)
```

**Problem:** The `|` operator creates a union of two separate querysets, not a filtered queryset.

**Fix:**
```python
from django.db.models import Q

queryset = CognitiveTask.objects.filter(
    Q(state=TaskState.PENDING.value) |
    Q(state=TaskState.RETRYING.value, next_retry_at__lte=now)
).filter(
    scheduled_at__lte=now,
).exclude(
    id__in=list(self._task_ids),
).order_by("-priority_score", "created_at")
```

---

### M2. WorkerPool._update_task_state Swallows Exceptions
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\execution\worker_pool.py`
**Lines:** 553-596
**Severity:** MEDIUM
**Standard Violation:** SCH-003 §6 (Error Visibility)

**Issue:**
Critical database errors are logged but swallowed, making debugging difficult.

**Fix:**
```python
def _update_task_state(self, result: ExecutionResult) -> None:
    """Update task state in database."""
    try:
        from syn.sched.models import CognitiveTask, TaskExecution, DeadLetterEntry
        from syn.sched.types import TaskState

        task = CognitiveTask.objects.get(id=result.task_id)

        # ... existing logic ...

        task.save()

    except CognitiveTask.DoesNotExist:
        logger.error(f"[WORKER_POOL] Task {result.task_id} not found in database")
        # Re-raise for visibility
        raise

    except Exception as e:
        logger.error(
            f"[WORKER_POOL] Failed to update task state: {e}",
            exc_info=True,
            extra={
                "task_id": str(result.task_id),
                "status": result.status.value,
            }
        )
        # Re-raise critical errors
        raise
```

---

### M3. SystemHealthMonitor._collect_cascade_metrics Uses Wrong Aggregate
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\backpressure\health.py`
**Lines:** 437-459
**Severity:** MEDIUM
**Standard Violation:** SCH-004 §2 (Metrics Accuracy)

**Issue:**
Uses `Count("cascade_depth")` which counts non-null values, not the maximum depth.

**Code:**
```python
max_depth_result = recent_tasks.aggregate(max_depth=Count("cascade_depth"))
```

**Fix:**
```python
from django.db.models import Max

max_depth_result = recent_tasks.aggregate(max_depth=Max("cascade_depth"))
if max_depth_result["max_depth"] is not None:
    metrics.max_cascade_depth_seen = max_depth_result["max_depth"]
```

---

### M4. Missing Validation in CognitiveTask.create_task
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\models.py`
**Lines:** 477-536
**Severity:** MEDIUM
**Standard Violation:** SCH-001 §5 (Input Validation)

**Issue:**
No validation for cognitive scores (urgency, confidence, governance_risk) being in valid ranges (0.0-1.0).

**Fix:**
```python
@classmethod
def create_task(
    cls,
    tenant_id: uuid.UUID,
    task_name: str,
    payload: Dict[str, Any],
    priority: TaskPriority = TaskPriority.NORMAL,
    queue: QueueType = QueueType.CORE,
    correlation_id: Optional[uuid.UUID] = None,
    parent_task: Optional["CognitiveTask"] = None,
    urgency: float = 0.5,
    confidence_score: float = 1.0,
    governance_risk: float = 0.0,
    deadline: Optional[datetime] = None,
    timeout_seconds: int = 60,
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    max_attempts: int = 3,
) -> "CognitiveTask":
    """Factory method to create a new cognitive task."""

    # Validate cognitive scores
    if not 0.0 <= urgency <= 1.0:
        raise ValueError(f"urgency must be in [0.0, 1.0], got {urgency}")
    if not 0.0 <= confidence_score <= 1.0:
        raise ValueError(f"confidence_score must be in [0.0, 1.0], got {confidence_score}")
    if not 0.0 <= governance_risk <= 1.0:
        raise ValueError(f"governance_risk must be in [0.0, 1.0], got {governance_risk}")

    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")

    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    # ... rest of implementation ...
```

---

### M5. TemporalController._collect_context Ignores All Exceptions
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\temporal\controller.py`
**Lines:** 449-535
**Severity:** MEDIUM
**Standard Violation:** SCH-006 §4 (Error Handling)

**Issue:**
All context collection errors are caught and logged at WARNING level, but this masks serious configuration issues.

**Fix:**
```python
def _collect_context(self) -> Dict[str, Any]:
    """Collect context from all sources."""
    context: Dict[str, Any] = {
        "timestamp": timezone.now().isoformat(),
    }

    errors = []

    # Backpressure context
    if self._config.enable_backpressure_context and self._backpressure:
        try:
            # ... collection code ...
        except AttributeError as e:
            # Configuration error - escalate
            logger.error(f"[TEMPORAL CONTROLLER] Backpressure integration error: {e}")
            errors.append(("backpressure", str(e)))
        except Exception as e:
            logger.warning(f"[TEMPORAL CONTROLLER] Failed to collect backpressure context: {e}")

    # Add error tracking to context
    if errors:
        context["_collection_errors"] = errors

    return context
```

---

### M6. Missing Index on CognitiveTask.scheduled_at
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\models.py`
**Lines:** 225-229
**Severity:** MEDIUM
**Standard Violation:** SCH-001 §5 (Performance)

**Issue:**
The `scheduled_at` field has `db_index=True` but is frequently queried with `state` in compound queries. A compound index would be more efficient.

**Fix:**
Add to `Meta.indexes`:
```python
class Meta(SynaraEntity.Meta):
    # ... existing indexes ...
    indexes = [
        # ... existing ...
        models.Index(
            fields=["state", "scheduled_at", "priority_score"],
            name="idx_task_scheduled_fetch",
        ),
    ]
```

---

### M7. RetryConfig.get_delay Missing Random Import Guard
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\types.py`
**Lines:** 142-168
**Severity:** MEDIUM
**Standard Violation:** SCH-002 §12 (Code Quality)

**Issue:**
The `random` module is imported inline within the method, which is inefficient and unconventional.

**Code:**
```python
def get_delay(self, attempt: int) -> timedelta:
    # ... logic ...
    if self.jitter:
        import random  # Import inside method!
        jitter_factor = random.uniform(0.5, 1.5)
```

**Fix:**
Move to top-level imports:
```python
# At top of file
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
```

---

## Low Priority Issues

### L1. Inconsistent Logging Levels
**Severity:** LOW
**Files:** Multiple

**Issue:**
Some error conditions log at DEBUG level when they should be WARNING or ERROR.

**Examples:**
- `execution/queue.py:225` - Duplicate task logged at DEBUG (should be INFO)
- `backpressure/health.py:294` - Metric collection errors at DEBUG (should be WARNING)

---

### L2. Missing Type Hints on Private Methods
**Severity:** LOW
**Files:** Multiple

**Issue:**
Many private methods lack type hints, making IDE support less effective.

**Example:**
```python
def _calculate_trends(self, metrics):  # Missing return type
    """Calculate trend metrics from history."""
```

**Should be:**
```python
def _calculate_trends(self, metrics: HealthMetrics) -> None:
    """Calculate trend metrics from history."""
```

---

### L3. Magic Numbers in Throttle Rules
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\backpressure\throttle.py`
**Lines:** 214-334
**Severity:** LOW
**Standard Violation:** Code Quality

**Issue:**
Hard-coded thresholds should be configurable constants.

**Fix:**
```python
# At module level
DEFAULT_QUEUE_LIGHT_THRESHOLD = 0.6
DEFAULT_QUEUE_MODERATE_THRESHOLD = 0.8
DEFAULT_QUEUE_HEAVY_THRESHOLD = 0.95
DEFAULT_DLQ_GROWTH_THRESHOLD = 5.0

DEFAULT_THROTTLE_RULES: List[ThrottleRule] = [
    ThrottleRule(
        name="queue_light",
        description="Light throttle when queue is 60% full",
        metric="queue_utilization",
        threshold=DEFAULT_QUEUE_LIGHT_THRESHOLD,
        # ...
    ),
    # ...
]
```

---

### L4. Unused Import in events.py
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\events.py`
**Lines:** 1-20
**Severity:** LOW

**Issue:**
No actual unused imports found, but the file could benefit from explicit `__all__` export list.

**Recommendation:**
```python
__all__ = [
    "SCHEDULER_EVENTS",
    "emit_scheduler_event",
    "build_task_created_payload",
    "build_task_started_payload",
    # ... all exported functions
]
```

---

## Invariant Violations

### IV1. TaskState Transition Validation Incomplete
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\models.py`
**Lines:** 353-405
**Severity:** MEDIUM

**Issue:**
The `can_transition_to` method checks valid_transitions(), but `transition_to` logs a warning and **continues anyway** instead of raising an exception.

**Code:**
```python
def transition_to(self, new_state: TaskState, **kwargs) -> bool:
    if not self.can_transition_to(new_state):
        logger.warning(
            f"Invalid state transition: {self.state} -> {new_state.value} "
            f"for task {self.id}"
        )
        return False  # Returns False but doesn't raise

    # ... continues with transition ...
```

**Problem:** Caller might not check return value, leading to silent failures.

**Fix:**
```python
def transition_to(self, new_state: TaskState, **kwargs) -> bool:
    if not self.can_transition_to(new_state):
        error_msg = (
            f"Invalid state transition: {self.state} -> {new_state.value} "
            f"for task {self.id}"
        )
        logger.error(error_msg)
        raise InvalidStateTransitionError(
            task_id=str(self.id),
            current_state=self.state,
            target_state=new_state.value,
        )

    # ... rest of method ...
```

---

### IV2. CASCADE_BUDGET Not Enforced Consistently
**File:** `C:\Users\ewolt\.claude-worktrees\synara_qms\confident-bhaskara\syn\sched\types.py`
**Lines:** 522-540
**Severity:** MEDIUM

**Issue:**
The `max_tasks_per_depth` budget is defined but never actually enforced in the code. Only `max_depth` is checked.

**Location:** `core.py:414-431` checks depth but not task count per depth.

**Fix Required:** Add enforcement in `submit()` method:
```python
def submit(self, task_name, payload, tenant_id, parent_task=None, ...):
    # ... existing code ...

    if parent_task:
        cascade_depth = parent_task.cascade_depth + 1
        limit = get_cascade_limit(cascade_depth)
        if limit == 0:
            raise ValueError(f"Cascade depth {cascade_depth} exceeds maximum")

        # NEW: Check task count at this depth
        tasks_at_depth = CognitiveTask.objects.filter(
            root_correlation_id=parent_task.root_correlation_id,
            cascade_depth=cascade_depth,
            state__in=[TaskState.PENDING.value, TaskState.RUNNING.value],
        ).count()

        max_tasks = CASCADE_BUDGET["max_tasks_per_depth"].get(cascade_depth, 0)
        if tasks_at_depth >= max_tasks:
            emit_scheduler_event("scheduler.cascade.throttled", ...)
            raise CascadeLimitError(
                f"Cascade depth {cascade_depth} has {tasks_at_depth}/{max_tasks} tasks"
            )
```

---

## SCH-002 Compliance Issues

### SC1. Missing DLQ Retention Policy
**Standard:** SCH-002 §13 (Dead Letter Queue)
**Severity:** MEDIUM

**Issue:**
The `DeadLetterEntry` model has no TTL or automatic cleanup mechanism. Old entries will accumulate indefinitely.

**Fix:**
Add to model:
```python
class DeadLetterEntry(SynaraEntity):
    # ... existing fields ...

    retention_days = models.PositiveIntegerField(
        default=90,
        help_text="Days to retain this entry before auto-deletion",
    )
    auto_delete_at = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Automatic deletion timestamp",
    )

    def save(self, *args, **kwargs):
        if not self.auto_delete_at:
            self.auto_delete_at = timezone.now() + timedelta(days=self.retention_days)
        super().save(*args, **kwargs)
```

Add cleanup task:
```python
@task("scheduler.dlq.cleanup", queue=QueueType.BATCH)
def cleanup_expired_dlq_entries(payload, context):
    """Clean up expired DLQ entries."""
    now = timezone.now()
    expired = DeadLetterEntry.objects.filter(
        auto_delete_at__lte=now,
        status__in=["resolved", "discarded"],
    )
    count = expired.count()
    expired.delete()

    logger.info(f"Cleaned up {count} expired DLQ entries")
    return {"deleted_count": count}
```

---

### SC2. Circuit Breaker Persistence Not Transactional
**Standard:** SCH-002 §4 (Circuit Breaker)
**Severity:** HIGH

**Issue:**
Circuit state changes (record_success, record_failure) don't use transactions, risking lost updates under concurrent access.

**Fix:**
```python
@transaction.atomic
def record_success(self) -> None:
    """Record a successful call (thread-safe)."""
    # Refresh to get latest state
    self.refresh_from_db()

    if self.state == CircuitState.HALF_OPEN.value:
        self.success_count += 1
        if self.success_count >= self.success_threshold:
            self._close_circuit()
        else:
            self.save()
    elif self.state == CircuitState.CLOSED.value:
        if self.failure_count > 0:
            self.failure_count = 0
            self.save()

@transaction.atomic
def record_failure(self) -> None:
    """Record a failed call (thread-safe)."""
    self.refresh_from_db()

    self.failure_count += 1
    self.last_failure_at = timezone.now()

    if self.state == CircuitState.HALF_OPEN.value:
        self._open_circuit()
    elif self.state == CircuitState.CLOSED.value:
        if self.failure_count >= self.failure_threshold:
            self._open_circuit()
        else:
            self.save()
```

---

## Recommendations

### Code Quality
1. **Add comprehensive docstrings** to all public methods with examples
2. **Implement property-based testing** for state transitions
3. **Add integration tests** for backpressure scenarios
4. **Create load tests** for queue depth and worker pool scaling

### Performance
1. **Add database query optimization** - use `select_related` and `prefetch_related`
2. **Implement connection pooling** for Django DB
3. **Add Redis cache** for circuit breaker states
4. **Use bulk operations** for batch task creation

### Monitoring
1. **Add Prometheus metrics** for all key operations
2. **Implement distributed tracing** with OpenTelemetry
3. **Create alerting rules** for critical conditions
4. **Add performance benchmarks** to CI/CD

### Security
1. **Add rate limiting** per tenant
2. **Implement payload size limits** to prevent DoS
3. **Add SQL injection protection** (Django ORM should handle this, but verify)
4. **Audit logging** for all state changes

---

## Summary of Fixes Applied

None - this is an audit report only. All fixes should be reviewed and applied systematically.

---

## Next Steps

1. **Immediate (P0):** Fix critical issues C1-C4
2. **Short-term (P1):** Address high-priority issues H1-H6
3. **Medium-term (P2):** Resolve medium-priority issues M1-M7
4. **Long-term (P3):** Clean up low-priority issues L1-L4
5. **Ongoing:** Implement recommendations for code quality and monitoring

---

**End of Audit Report**

"""
Synara Event Bus — Flat Dispatcher
===================================

Replaces the 10-layer cognitive pipeline (syn.engine + syn.kernel) with a
simple validate → govern → dispatch flow.

Architecture:
    emit(event_name, payload, context)
        1. Schema validate (optional — if EventSchema registered)
        2. Governance check (if GovernanceRule matches)
        3. Dispatch to subscribers (pattern-matched, FIFO)

Design decisions:
    - Pull-only: subscribers register interest, bus dispatches (FLOW-1)
    - Events chain, features don't: subscribers can emit new events (FLOW-2)
    - Contracts visible: all subscriptions are inspectable (FLOW-3)
    - No biological metaphor. No layers. No Cortex.

Standard: EVT-001, GOV-001, FLOW-1/2/3
"""

from __future__ import annotations

import fnmatch
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """An emitted event with metadata."""

    name: str
    payload: Dict[str, Any]
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_correlation_id: Optional[str] = None
    tenant_id: Optional[str] = None
    actor: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmitResult:
    """Result of emitting an event."""

    event: Event
    allowed: bool = True
    governance_judgment: Optional[str] = None  # allow / block / escalate
    governance_reason: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)
    dispatched_to: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class Subscription:
    """A registered event subscriber."""

    pattern: str  # fnmatch pattern, e.g. "pcl.characteristic.*"
    handler: Callable[[Event], None]
    subscriber_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    tenant_id: Optional[str] = None  # None = global subscriber


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------


class EventBus:
    """
    Flat event dispatcher: validate → govern → dispatch.

    Usage:
        bus = get_bus()

        # Subscribe
        bus.subscribe("pcl.characteristic.*", my_handler, description="Update dashboard")

        # Emit
        result = bus.emit("pcl.characteristic.updated", {"char_id": "...", "cpk": 1.45})

        # Inspect
        bus.list_subscriptions()
    """

    def __init__(self):
        self._subscriptions: List[Subscription] = []
        self._emit_count: int = 0
        self._block_count: int = 0

    # -- Subscribe / Unsubscribe --

    def subscribe(
        self,
        pattern: str,
        handler: Callable[[Event], None],
        *,
        description: str = "",
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        Register a handler for events matching pattern.

        Args:
            pattern: fnmatch pattern (e.g. "pcl.*", "job.completed")
            handler: Callable receiving Event
            description: Human-readable purpose
            tenant_id: Scope to tenant (None = all tenants)

        Returns:
            Subscription ID (for unsubscribe)
        """
        sub = Subscription(
            pattern=pattern,
            handler=handler,
            description=description,
            tenant_id=tenant_id,
        )
        self._subscriptions.append(sub)
        logger.debug(f"[BUS] Subscribed {sub.subscriber_id[:8]}... to '{pattern}': {description}")
        return sub.subscriber_id

    def unsubscribe(self, subscriber_id: str) -> bool:
        """Remove a subscription by ID. Returns True if found."""
        before = len(self._subscriptions)
        self._subscriptions = [s for s in self._subscriptions if s.subscriber_id != subscriber_id]
        removed = len(self._subscriptions) < before
        if removed:
            logger.debug(f"[BUS] Unsubscribed {subscriber_id[:8]}...")
        return removed

    # -- Emit --

    def emit(
        self,
        event_name: str,
        payload: Dict[str, Any],
        *,
        correlation_id: Optional[str] = None,
        parent_correlation_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        actor: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmitResult:
        """
        Emit an event: validate → govern → dispatch.

        Args:
            event_name: Dotted event name (e.g. "pcl.characteristic.updated")
            payload: Event data
            correlation_id: Tracking ID (auto-generated if None)
            parent_correlation_id: Parent event's correlation ID
            tenant_id: Emitting tenant
            actor: Who/what emitted
            metadata: Additional metadata

        Returns:
            EmitResult with dispatch details
        """
        event = Event(
            name=event_name,
            payload=payload,
            correlation_id=correlation_id or str(uuid.uuid4()),
            parent_correlation_id=parent_correlation_id,
            tenant_id=tenant_id,
            actor=actor,
            metadata=metadata or {},
        )

        result = EmitResult(event=event)
        self._emit_count += 1

        # Step 1: Schema validation (optional)
        validation_errors = self._validate_schema(event)
        if validation_errors:
            result.validation_errors = validation_errors
            result.allowed = False
            logger.warning(f"[BUS] Schema validation failed for '{event_name}': {validation_errors}")
            return result

        # Step 2: Governance check (optional)
        judgment = self._check_governance(event)
        if judgment:
            result.governance_judgment = judgment["judgment"]
            result.governance_reason = judgment.get("reason", "")
            if judgment["judgment"] == "block":
                result.allowed = False
                self._block_count += 1
                logger.info(f"[BUS] Governance BLOCKED '{event_name}': {judgment.get('reason', '')}")
                return result
            if judgment["judgment"] == "escalate":
                logger.info(f"[BUS] Governance ESCALATED '{event_name}': {judgment.get('reason', '')}")
                # Escalated events still dispatch but are flagged

        # Step 3: Dispatch to subscribers
        matched = self._match_subscribers(event)
        for sub in matched:
            try:
                sub.handler(event)
                result.dispatched_to += 1
            except Exception as e:
                error_msg = f"Subscriber {sub.subscriber_id[:8]}... failed: {e}"
                result.errors.append(error_msg)
                logger.error(f"[BUS] {error_msg}")

        logger.debug(
            f"[BUS] '{event_name}' dispatched to {result.dispatched_to} subscriber(s)"
            f" [corr={event.correlation_id[:8]}...]"
        )
        return result

    # -- Introspection (FLOW-3: contracts visible) --

    def list_subscriptions(self) -> List[Dict[str, Any]]:
        """List all active subscriptions. FLOW-3: contracts visible."""
        return [
            {
                "id": s.subscriber_id,
                "pattern": s.pattern,
                "description": s.description,
                "tenant_id": s.tenant_id,
            }
            for s in self._subscriptions
        ]

    @property
    def stats(self) -> Dict[str, int]:
        """Bus statistics."""
        return {
            "subscriptions": len(self._subscriptions),
            "total_emitted": self._emit_count,
            "total_blocked": self._block_count,
        }

    # -- Internal: Schema Validation --

    def _validate_schema(self, event: Event) -> List[str]:
        """
        Validate event payload against registered EventSchema (if any).

        Returns list of validation errors (empty = valid or no schema).
        """
        try:
            from syn.schema.models import EventSchema

            schema_obj = (
                EventSchema.objects.filter(
                    name=event.name,
                    is_active=True,
                )
                .order_by("-version")
                .first()
            )

            if not schema_obj:
                return []  # No schema registered — allow through

            import jsonschema

            try:
                jsonschema.validate(event.payload, schema_obj.schema_json)
                return []
            except jsonschema.ValidationError as e:
                return [str(e.message)]
            except jsonschema.SchemaError as e:
                return [f"Invalid schema definition: {e.message}"]

        except ImportError:
            return []  # jsonschema not installed — skip validation
        except Exception as e:
            logger.error(f"[BUS] Schema validation error: {e}")
            return []  # Fail open — don't block events due to validation infra errors

    # -- Internal: Governance Check --

    def _check_governance(self, event: Event) -> Optional[Dict[str, str]]:
        """
        Check event against active GovernanceRules.

        Returns dict with judgment/reason, or None if no rules match.
        """
        try:
            from syn.governance.models import GovernanceRule

            rules = GovernanceRule.objects.filter(
                trigger_event=event.name,
                status="active",
            ).order_by("-priority")

            if event.tenant_id:
                # Tenant-specific + global rules
                rules = (
                    rules.filter(models_tenant_id__in=[event.tenant_id, None]) if False else rules
                )  # TODO: proper tenant filtering

            rule = rules.first()
            if not rule:
                return None

            # Evaluate conditions (JSONLogic) if present
            if rule.conditions:
                try:
                    from json_logic import jsonLogic

                    match = jsonLogic(rule.conditions, event.payload)
                    if not match:
                        return None  # Conditions not met — rule doesn't apply
                except (ImportError, Exception):
                    pass  # If json_logic unavailable, apply rule unconditionally

            return {
                "judgment": rule.judgment_type,
                "reason": f"Rule '{rule.name}' (priority {rule.priority})",
                "rule_id": str(rule.id),
            }

        except Exception as e:
            logger.error(f"[BUS] Governance check error: {e}")
            return None  # Fail open — governance errors don't block events

    # -- Internal: Subscriber Matching --

    def _match_subscribers(self, event: Event) -> List[Subscription]:
        """Match event name against subscription patterns."""
        matched = []
        for sub in self._subscriptions:
            if fnmatch.fnmatch(event.name, sub.pattern):
                # Tenant filtering: global subscribers see all, scoped see their own
                if sub.tenant_id and event.tenant_id and sub.tenant_id != event.tenant_id:
                    continue
                matched.append(sub)
        return matched


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_bus: Optional[EventBus] = None


def get_bus() -> EventBus:
    """Get the global event bus instance."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def emit(
    event_name: str,
    payload: Dict[str, Any],
    **kwargs,
) -> EmitResult:
    """Convenience: emit on the global bus."""
    return get_bus().emit(event_name, payload, **kwargs)


def subscribe(
    pattern: str,
    handler: Callable[[Event], None],
    **kwargs,
) -> str:
    """Convenience: subscribe on the global bus."""
    return get_bus().subscribe(pattern, handler, **kwargs)

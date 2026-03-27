"""
ToolEventBus handlers for QMS cross-cutting concerns.

Handles evidence creation, project logging, and investigation bridging
in response to tool lifecycle events. Import this module to register
all handlers (done in agents_api.apps.ready()).

Standard:     ARCH-001 §10.2 (ToolEventBus)
Compliance:   AUD-001 §3 (Audit Trail)

CR: 9e7e3fd2
"""

import logging

from .tool_events import tool_events

logger = logging.getLogger(__name__)


# =============================================================================
# A3 Report Handlers
# =============================================================================


@tool_events.on("a3.created")
def on_a3_created(event):
    """Log A3 creation to project timeline."""
    report = event.record
    if report.project:
        report.project.log_event(
            "a3_created",
            f"A3 report: {report.title}",
            user=event.user,
        )


@tool_events.on("a3.projected")
def on_a3_projected(event):
    """Log A3 projection from notebook to project timeline."""
    report = event.record
    notebook_title = event.extra.get("notebook_title", "")
    if report.project:
        report.project.log_event(
            "a3_projected",
            f"A3 projected from notebook: {notebook_title}",
            user=event.user,
        )


@tool_events.on("a3.updated")
def on_a3_updated_evidence(event):
    """Create evidence records when A3 root_cause or follow_up are updated."""
    from .evidence_bridge import create_tool_evidence

    report = event.record
    data = event.extra.get("data", {})

    if not report.project:
        return

    if "root_cause" in data and data["root_cause"]:
        create_tool_evidence(
            project=report.project,
            user=event.user,
            summary=f"A3 root cause: {data['root_cause'][:200]}",
            source_tool="a3",
            source_id=str(report.id),
            source_field="root_cause",
            details=data["root_cause"],
            source_type="analysis",
        )

    if "follow_up" in data and data["follow_up"]:
        create_tool_evidence(
            project=report.project,
            user=event.user,
            summary=f"A3 follow-up: {data['follow_up'][:200]}",
            source_tool="a3",
            source_id=str(report.id),
            source_field="follow_up",
            details=data["follow_up"],
            source_type="experiment",
        )

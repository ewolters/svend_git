"""
ToolEventBus handlers for QMS cross-cutting concerns.

Handles evidence creation, project logging, and investigation bridging
in response to tool lifecycle events. Import this module to register
all handlers (done in agents_api.apps.ready()).

Standard:     ARCH-001 §10.2 (ToolEventBus)
Compliance:   AUD-001 §3 (Audit Trail)
"""

import logging

from .tool_events import tool_events

logger = logging.getLogger(__name__)


# =============================================================================
# Wildcard Handlers — Project Timeline Logging
# =============================================================================


@tool_events.on("*.created")
def on_any_created(event):
    """Log tool creation to project timeline."""
    record = event.record
    project = getattr(record, "project", None)
    if not project:
        return
    tool = event.name.split(".")[0]
    title = getattr(record, "title", None) or getattr(record, "name", str(record.id)[:8])
    project.log_event(
        f"{tool}_created",
        f"{tool.upper()}: {title}",
        user=event.user,
    )


@tool_events.on("*.updated")
def on_any_updated(event):
    """Log tool updates to project timeline."""
    record = event.record
    project = getattr(record, "project", None)
    if not project:
        return
    tool = event.name.split(".")[0]
    title = getattr(record, "title", None) or getattr(record, "name", str(record.id)[:8])
    project.log_event(
        f"{tool}_updated",
        f"{tool.upper()} updated: {title}",
        user=event.user,
    )


# =============================================================================
# A3 Report Handlers
# =============================================================================


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


# =============================================================================
# RCA Handlers
# =============================================================================


@tool_events.on("rca.updated")
def on_rca_updated_evidence(event):
    """Create evidence when RCA root_cause is identified."""
    from .evidence_bridge import create_tool_evidence

    session = event.record
    data = event.extra.get("data", {})
    project = getattr(session, "project", None)

    if not project or "root_cause" not in data or not data["root_cause"]:
        return

    create_tool_evidence(
        project=project,
        user=event.user,
        summary=f"RCA root cause: {data['root_cause'][:200]}",
        source_tool="rca",
        source_id=str(session.id),
        source_field="root_cause",
        details=data["root_cause"],
        source_type="analysis",
    )


# =============================================================================
# FMEA Handlers
# =============================================================================


@tool_events.on("fmea.row_updated")
def on_fmea_row_updated_evidence(event):
    """Create evidence when FMEA row RPN changes significantly."""
    from .evidence_bridge import create_tool_evidence

    row = event.record
    fmea = getattr(row, "fmea", None)
    project = getattr(fmea, "project", None) if fmea else None

    if not project:
        return

    create_tool_evidence(
        project=project,
        user=event.user,
        summary=f"FMEA row updated: {row.failure_mode[:100]} (RPN={row.rpn})",
        source_tool="fmea",
        source_id=str(row.id),
        source_field="rpn",
        details=f"S={row.severity} O={row.occurrence} D={row.detection} RPN={row.rpn}",
        source_type="analysis",
    )


# =============================================================================
# SPC Signal Handler
# =============================================================================


@tool_events.on("spc.signal")
def on_spc_signal(event):
    """Log SPC out-of-control signal. Connects to Verify mode."""
    logger.info(
        "SPC signal: %s chart, %d/%d points OOC (fmea_row=%s, project=%s)",
        event.extra.get("chart_type"),
        event.extra.get("ooc_count", 0),
        event.extra.get("total_points", 0),
        event.extra.get("fmea_row_id"),
        event.extra.get("project_id"),
    )

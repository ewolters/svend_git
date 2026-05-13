"""
Plugin Runner — the full execution lifecycle.

    validate input -> create Job -> execute -> create JobOutputs -> emit event

This is the single entry point for running any plugin. Canvas views,
API endpoints, and CLI all call run_plugin().
"""

import logging
from typing import Any, Dict, Optional

from django.utils import timezone

from syn.bus import emit
from syn.plugins.registry import PluginRegistry, get_registry

logger = logging.getLogger(__name__)


def run_plugin(
    plugin_name: str,
    input_data: Dict[str, Any],
    *,
    actor: str,
    tenant_id=None,
    canvas_id=None,
    is_scratch: bool = False,
    registry: Optional[PluginRegistry] = None,
) -> "Job":
    """Run a plugin through the full lifecycle.

    Args:
        plugin_name: Registered plugin name.
        input_data: Raw input (will be validated against plugin.input_schema).
        actor: Who triggered this (user email or system identifier).
        tenant_id: Tenant UUID (optional for individual users).
        canvas_id: Canvas UUID that triggered this run (optional).
        is_scratch: Mark as scratch/exploratory work.
        registry: PluginRegistry instance (defaults to global singleton).

    Returns:
        Completed Job instance with outputs.

    Raises:
        KeyError: Plugin not found in registry.
        pydantic.ValidationError: Input doesn't match schema.
        Exception: Re-raises plugin execution errors after marking Job failed.
    """
    from job.models import Job, JobOutput

    reg = registry or get_registry()
    plugin = reg.get(plugin_name)

    # Validate input
    validated = plugin.input_schema(**input_data).model_dump()

    # Create Job (status=running)
    job = Job.objects.create(
        plugin_name=plugin_name,
        canvas_id=canvas_id,
        inputs=validated,
        actor=actor,
        tenant_id=tenant_id,
        status="running",
        started_at=timezone.now(),
        is_scratch=is_scratch,
    )

    # Execute
    try:
        context = {
            "job_id": str(job.id),
            "actor": actor,
            "tenant_id": str(tenant_id) if tenant_id else None,
        }
        outputs = plugin.execute(validated, context)
    except Exception:
        # Mark job failed
        job.status = "failed"
        job.completed_at = timezone.now()
        job.duration_ms = int((job.completed_at - job.started_at).total_seconds() * 1000)
        job.save(update_fields=["status", "completed_at", "duration_ms"])
        raise

    # Create JobOutputs
    for out in outputs:
        JobOutput.objects.create(
            job=job,
            output_key=out.key,
            output_type=out.output_type,
            value_numeric=out.value if out.output_type == "metric" else None,
            value_json=out.value if out.output_type != "metric" else {},
            provenance=out.provenance,
            measure_slug=out.measure_slug,
        )

    # PCL write-back: persist metric outputs to Process Characteristics Library
    _write_to_pcl(outputs, job=job, actor=actor, tenant_id=tenant_id, plugin_name=plugin_name)

    # Complete job
    job.status = "completed"
    job.completed_at = timezone.now()
    job.duration_ms = int((job.completed_at - job.started_at).total_seconds() * 1000)
    job.outputs_summary = {out.key: out.output_type for out in outputs}
    job.save(update_fields=["status", "completed_at", "duration_ms", "outputs_summary"])

    # Emit bus event
    try:
        emit(
            "plugin.execution.completed",
            {
                "job_id": str(job.id),
                "plugin_name": plugin_name,
                "outputs": [out.key for out in outputs],
            },
            actor=actor,
            tenant_id=str(tenant_id) if tenant_id else None,
        )
    except Exception as e:
        logger.warning(f"[PLUGINS] Bus emit failed for {plugin_name}: {e}")

    return job


def _write_to_pcl(outputs, *, job, actor, tenant_id, plugin_name):
    """Write metric outputs with measure_slug to PCL.

    Non-fatal — if a Measure doesn't exist, we log and skip.
    Only outputs with measure_slug set and output_type=="metric" are written.
    Provenance gating (observed/calculated vs simulated/projected) is handled
    by pcl.service.write() itself.
    """
    from pcl import service as pcl_service

    for out in outputs:
        if not out.measure_slug or out.output_type != "metric":
            continue
        try:
            pcl_service.write(
                measure_slug=out.measure_slug,
                value=out.value,
                source_type="plugin",
                actor=actor,
                tenant_id=tenant_id,
                provenance=out.provenance,
                source_job_id=job.id,
                source_ref_type=f"plugin:{plugin_name}",
                notes=f"Auto-written by {plugin_name} plugin",
            )
        except ValueError:
            # Measure doesn't exist yet — that's fine, PCL binding is optional
            logger.debug(
                "[PCL] Measure '%s' not found, skipping write from %s",
                out.measure_slug,
                plugin_name,
            )

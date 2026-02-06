"""
Forge Tasks (Tempora-enabled)

Task handlers for data generation.
Tasks are registered with Tempora for async execution.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from django.utils import timezone
from django.conf import settings

# Tempora integration
from tempora.core import task, TaskRegistry
from tempora.types import TaskPriority, QueueType, RetryStrategy

logger = logging.getLogger(__name__)

# Result storage path
RESULTS_DIR = Path(settings.BASE_DIR) / "forge_results"
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Tempora Task Handler Registration
# =============================================================================

@task(
    "forge.tasks.generate_data_task",
    queue=QueueType.BATCH,
    priority=TaskPriority.NORMAL,
    timeout_seconds=600,
    retry_strategy=RetryStrategy.EXPONENTIAL,
    max_attempts=3,
)
def generate_data_task_async(payload: Dict[str, Any], context: Any = None) -> dict:
    """
    Tempora-registered handler for async data generation.

    Called by Tempora workers when processing queued jobs.
    """
    job_id = payload.get("job_id")
    if not job_id:
        return {"success": False, "error": "Missing job_id in payload"}

    return _generate_data(job_id)


def generate_data_task(job_id: str) -> dict:
    """
    Synchronous entry point for small jobs.

    Called directly for fast execution without queueing.
    """
    return _generate_data(job_id)


def _generate_data(job_id: str) -> dict:
    """
    Internal implementation of data generation.

    Called by both sync (generate_data_task) and async
    (generate_data_task_async) entry points.
    """
    from forge.models import Job, JobStatus, DataType, UsageLog

    try:
        job = Job.objects.get(job_id=job_id)
    except Job.DoesNotExist:
        logger.error(f"Job not found: {job_id}")
        return {"success": False, "error": "Job not found"}

    try:
        job.mark_processing()
        logger.info(f"Processing job {job_id}: {job.data_type}, {job.record_count} records")

        # Generate based on data type
        if job.data_type == DataType.TABULAR:
            result = generate_tabular(job)
        elif job.data_type == DataType.TEXT:
            result = generate_text(job)
        else:
            raise ValueError(f"Unknown data type: {job.data_type}")

        if not result["success"]:
            job.mark_failed(result.get("error", "Unknown error"))
            return result

        records = result["records"]

        # Run quality validation
        from forge.quality import validate_records
        quality_report = validate_records(records, job.schema_def, job.quality_level)

        # Format output
        content = format_output(records, job.output_format)

        # Store results
        file_path = RESULTS_DIR / f"{job_id}.{job.output_format}"
        file_path.write_text(content)

        # Update job
        job.mark_completed(
            result_path=str(file_path),
            records=len(records),
            size_bytes=len(content.encode("utf-8")),
        )
        job.quality_score = quality_report.get("overall_score", 1.0)
        job.quality_report = quality_report
        job.save(update_fields=["quality_score", "quality_report"])

        # Log usage
        log_usage(job, len(records))

        logger.info(f"Job {job_id} completed: {len(records)} records")

        return {
            "success": True,
            "records_generated": len(records),
            "data": records[:100] if len(records) <= 1000 else None,  # Include data for small jobs
        }

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        job.mark_failed(str(e))
        return {"success": False, "error": str(e)}


def generate_tabular(job) -> dict:
    """Generate tabular data."""
    from forge.generators.tabular import TabularGenerator

    try:
        generator = TabularGenerator(
            schema=job.schema_def,
            domain=job.domain or "general",
        )
        records = generator.generate(job.record_count)
        return {"success": True, "records": records}
    except Exception as e:
        logger.exception(f"Tabular generation failed: {e}")
        return {"success": False, "error": str(e)}


def generate_text(job) -> dict:
    """Generate text data using Svend's Qwen LLM."""
    from forge.generators.text import TextGenerator

    try:
        text_type = job.schema_def.get("text_type", "review")

        # Get LLM from Svend's shared instance (Qwen-7B)
        llm = None
        try:
            from agents_api.views import get_shared_llm
            llm = get_shared_llm()
            if llm:
                logger.info("Using Svend's Qwen LLM for text generation")
        except Exception as e:
            logger.warning(f"Could not load LLM, using templates: {e}")

        generator = TextGenerator(
            domain=job.domain or "ecommerce",
            text_type=text_type,
            llm=llm,
        )
        records = generator.generate(job.record_count)
        return {"success": True, "records": records}
    except Exception as e:
        logger.exception(f"Text generation failed: {e}")
        return {"success": False, "error": str(e)}


def format_output(records: list, output_format: str) -> str:
    """Format records for output."""
    if output_format == "json":
        return json.dumps(records, indent=2, default=str)
    elif output_format == "jsonl":
        return "\n".join(json.dumps(r, default=str) for r in records)
    elif output_format == "csv":
        import csv
        import io
        if not records:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
        return output.getvalue()
    else:
        return "\n".join(json.dumps(r, default=str) for r in records)


def log_usage(job, record_count: int):
    """Log usage for billing."""
    from forge.models import UsageLog

    now = timezone.now()
    period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    if now.month == 12:
        period_end = period_start.replace(year=now.year + 1, month=1)
    else:
        period_end = period_start.replace(month=now.month + 1)

    UsageLog.objects.create(
        api_key=job.api_key,
        job=job,
        data_type=job.data_type,
        quality_level=job.quality_level,
        record_count=record_count,
        cost_cents=job.cost_cents,
        period_start=period_start,
        period_end=period_end,
    )

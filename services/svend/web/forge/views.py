"""Forge API views."""

import hashlib
import logging
import secrets
from datetime import timedelta

from django.db.models import Sum
from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .models import APIKey, Job, JobStatus, DataType, QualityLevel, UsageLog, SchemaTemplate
from accounts.constants import Tier
from .serializers import (
    GenerateRequestSerializer,
    GenerateResponseSerializer,
    JobSerializer,
    JobResultSerializer,
    SchemaTemplateSerializer,
    UsageSummarySerializer,
    UsageResponseSerializer,
)
from .tasks import generate_data_task

logger = logging.getLogger(__name__)


# =============================================================================
# API Key Authentication
# =============================================================================

def get_api_key(request) -> APIKey | None:
    """Extract and validate API key from request."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        key = auth_header[7:]
    else:
        key = request.headers.get("X-API-Key", "")

    if not key:
        return None

    key_hash = hashlib.sha256(key.encode()).hexdigest()

    try:
        api_key = APIKey.objects.get(key_hash=key_hash, is_active=True)
        api_key.last_used_at = timezone.now()
        api_key.save(update_fields=["last_used_at"])
        return api_key
    except APIKey.DoesNotExist:
        return None


def require_api_key(view_func):
    """Decorator to require valid API key OR authenticated session."""
    def wrapper(request, *args, **kwargs):
        # First try API key
        api_key = get_api_key(request)
        if api_key:
            request.api_key = api_key
            return view_func(request, *args, **kwargs)

        # Fall back to session authentication for web UI users
        if request.user and request.user.is_authenticated:
            request.api_key = None  # No API key, but user is authenticated
            return view_func(request, *args, **kwargs)

        return Response(
            {"error": "Invalid or missing API key"},
            status=status.HTTP_401_UNAUTHORIZED
        )
    return wrapper


# =============================================================================
# Health
# =============================================================================

@api_view(["GET"])
@permission_classes([AllowAny])
def health(request):
    """Health check."""
    return Response({
        "status": "healthy",
        "service": "forge",
        "version": "1.0.0",
    })


# =============================================================================
# Generation
# =============================================================================

# Records per month limits for Forge API by tier
TIER_LIMITS = {
    Tier.FREE: 1000,
    Tier.FOUNDER: 50000,
    Tier.PRO: 50000,
    Tier.TEAM: 200000,
    Tier.ENTERPRISE: 500000,
}

PRICE_TABULAR_PER_1K = 100  # $1 per 1000 records (in cents)
PRICE_TEXT_PER_1K = 500     # $5 per 1000 records
PREMIUM_MULTIPLIER = 2.0


def calculate_cost(data_type: str, record_count: int, quality_level: str) -> int:
    """Calculate cost in cents."""
    base_rate = PRICE_TABULAR_PER_1K if data_type == "tabular" else PRICE_TEXT_PER_1K
    multiplier = PREMIUM_MULTIPLIER if quality_level == "premium" else 1.0

    # Volume discounts
    if record_count >= 100000:
        discount = 0.15
    elif record_count >= 50000:
        discount = 0.10
    elif record_count >= 10000:
        discount = 0.05
    else:
        discount = 0.0

    cost = (record_count / 1000) * base_rate * multiplier * (1 - discount)
    return int(cost)


def get_current_period_usage(api_key: APIKey) -> int:
    """Get total records used in current billing period."""
    now = timezone.now()
    period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    total = Job.objects.filter(
        api_key=api_key,
        created_at__gte=period_start,
        status__in=[JobStatus.COMPLETED, JobStatus.PROCESSING, JobStatus.QUEUED],
    ).aggregate(total=Sum("record_count"))["total"]

    return total or 0


@api_view(["POST"])
@require_api_key
def generate(request):
    """Create a synthetic data generation job."""
    serializer = GenerateRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    api_key = request.api_key

    # Get schema from template or request
    if data.get("template"):
        try:
            template = SchemaTemplate.objects.get(name=data["template"])
            schema_def = template.schema_def
            domain = template.domain
        except SchemaTemplate.DoesNotExist:
            return Response(
                {"error": f"Template not found: {data['template']}"},
                status=status.HTTP_400_BAD_REQUEST
            )
    else:
        schema_def = data["schema"]
        domain = data.get("domain", "")

    # Calculate cost
    cost_cents = calculate_cost(
        data["data_type"],
        data["record_count"],
        data["quality_level"],
    )

    # Check tier limits (skip for session-authenticated users without API key)
    if api_key:
        current_usage = get_current_period_usage(api_key)
        tier_limit = TIER_LIMITS.get(api_key.tier, TIER_LIMITS[Tier.FREE])

        if current_usage + data["record_count"] > tier_limit:
            return Response(
                {
                    "error": "Tier limit exceeded",
                    "current_usage": current_usage,
                    "tier_limit": tier_limit,
                    "tier": api_key.tier,
                },
                status=status.HTTP_402_PAYMENT_REQUIRED
            )

    # Create job
    job = Job.objects.create(
        api_key=api_key,  # Will be None for session auth
        user=request.user if request.user.is_authenticated else None,
        data_type=data["data_type"],
        domain=domain,
        record_count=data["record_count"],
        schema_def=schema_def,
        quality_level=data["quality_level"],
        output_format=data["output_format"],
        cost_cents=cost_cents,
    )

    # For small jobs, execute synchronously
    SYNC_THRESHOLD = 1000
    if data["record_count"] <= SYNC_THRESHOLD and data["data_type"] == "tabular":
        result = generate_data_task(str(job.job_id))
        job.refresh_from_db()

        response_data = {
            "job_id": job.job_id,
            "status": job.status,
            "record_count": job.record_count,
            "estimated_cost_cents": cost_cents,
            "message": "Generation completed.",
        }

        if job.status == JobStatus.COMPLETED and result.get("data"):
            response_data["data"] = result["data"]

        return Response(response_data, status=status.HTTP_201_CREATED)

    # Queue larger jobs with Tempora
    from tempora import schedule_task
    task_id = schedule_task(
        name=f"forge.generate.{job.job_id}",
        func="forge.tasks.generate_data_task",
        args={"job_id": str(job.job_id)},
    )
    job.task_id = task_id
    job.save(update_fields=["task_id"])

    return Response(
        {
            "job_id": job.job_id,
            "status": "queued",
            "record_count": data["record_count"],
            "estimated_cost_cents": cost_cents,
            "message": "Job queued for processing.",
        },
        status=status.HTTP_202_ACCEPTED
    )


@api_view(["GET"])
@require_api_key
def job_status(request, job_id):
    """Get job status."""
    try:
        job = Job.objects.get(job_id=job_id, api_key=request.api_key)
    except Job.DoesNotExist:
        return Response({"error": "Job not found"}, status=status.HTTP_404_NOT_FOUND)

    serializer = JobSerializer(job)
    return Response(serializer.data)


@api_view(["GET"])
@require_api_key
def job_result(request, job_id):
    """Get job result download URL."""
    try:
        job = Job.objects.get(job_id=job_id, api_key=request.api_key)
    except Job.DoesNotExist:
        return Response({"error": "Job not found"}, status=status.HTTP_404_NOT_FOUND)

    if job.status != JobStatus.COMPLETED:
        return Response(
            {"error": f"Job not completed. Status: {job.status}"},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Generate download URL
    expires_at = timezone.now() + timedelta(hours=24)
    download_url = f"/api/forge/download/{job_id}"

    return Response({
        "job_id": job.job_id,
        "download_url": download_url,
        "expires_at": expires_at,
        "size_bytes": job.result_size_bytes or 0,
        "record_count": job.records_generated or job.record_count,
        "output_format": job.output_format,
    })


@api_view(["GET"])
@require_api_key
def download(request, job_id):
    """Download generated data."""
    from django.http import HttpResponse

    try:
        job = Job.objects.get(job_id=job_id, api_key=request.api_key)
    except Job.DoesNotExist:
        return Response({"error": "Job not found"}, status=status.HTTP_404_NOT_FOUND)

    if job.status != JobStatus.COMPLETED:
        return Response(
            {"error": f"Job not completed. Status: {job.status}"},
            status=status.HTTP_400_BAD_REQUEST
        )

    # Get content from storage
    # Alpha: Local storage via job.result_path
    # Beta: Integrate with MinIO/S3 for scalable object storage

    try:
        from forge.storage import get_job_content
        content = get_job_content(job)
    except Exception as e:
        logger.error(f"Failed to retrieve job content: {e}")
        return Response(
            {"error": "Failed to retrieve data"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    content_type = {
        "json": "application/json",
        "jsonl": "application/jsonl",
        "csv": "text/csv",
    }.get(job.output_format, "application/octet-stream")

    response = HttpResponse(content, content_type=content_type)
    response["Content-Disposition"] = f'attachment; filename="forge_{job_id}.{job.output_format}"'
    return response


# =============================================================================
# Schemas
# =============================================================================

@api_view(["GET"])
@require_api_key
def list_schemas(request):
    """List available schema templates."""
    templates = SchemaTemplate.objects.all()
    serializer = SchemaTemplateSerializer(templates, many=True)
    return Response({"schemas": serializer.data})


# =============================================================================
# Usage
# =============================================================================

@api_view(["GET"])
@require_api_key
def usage(request):
    """Get usage statistics."""
    api_key = request.api_key
    now = timezone.now()

    period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if now.month == 12:
        period_end = period_start.replace(year=now.year + 1, month=1)
    else:
        period_end = period_start.replace(month=now.month + 1)

    jobs = Job.objects.filter(
        api_key=api_key,
        created_at__gte=period_start,
        created_at__lt=period_end,
    )

    total_records = jobs.aggregate(total=Sum("record_count"))["total"] or 0
    total_cost = jobs.aggregate(total=Sum("cost_cents"))["total"] or 0
    jobs_completed = jobs.filter(status=JobStatus.COMPLETED).count()
    jobs_failed = jobs.filter(status=JobStatus.FAILED).count()

    records_by_type = {}
    for dtype in DataType:
        count = jobs.filter(data_type=dtype).aggregate(total=Sum("record_count"))["total"]
        if count:
            records_by_type[dtype.value] = count

    tier_limit = TIER_LIMITS.get(api_key.tier, TIER_LIMITS[Tier.FREE])

    return Response({
        "current_period": {
            "period_start": period_start,
            "period_end": period_end,
            "total_records": total_records,
            "total_cost_cents": total_cost,
            "jobs_completed": jobs_completed,
            "jobs_failed": jobs_failed,
            "records_by_type": records_by_type,
        },
        "tier": api_key.tier,
        "tier_limit": tier_limit,
        "records_remaining": max(0, tier_limit - total_records),
    })

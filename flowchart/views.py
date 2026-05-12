"""Flowchart API endpoints.

POST /api/flowchart/run/       — execute a flowchart definition
GET  /api/flowchart/templates/ — list available templates
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from accounts.permissions import require_auth
from flowchart.engine import execute_flowchart
from flowchart.models import FlowchartTemplate

logger = logging.getLogger(__name__)


@csrf_exempt
@require_auth
@require_POST
def flowchart_run(request):
    """Execute a flowchart and return results per device."""
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    definition = body.get("definition")
    if not definition:
        return JsonResponse({"error": "definition is required"}, status=400)

    is_scratch = body.get("is_scratch", False)

    try:
        results = execute_flowchart(
            definition=definition,
            actor=request.user.email,
            tenant_id=getattr(request.user, "tenant_id", None),
            is_scratch=is_scratch,
        )
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception(f"[FLOWCHART] Execution failed: {e}")
        return JsonResponse({"error": str(e)}, status=500)

    response = {"results": {}}
    for device_id, data in results.items():
        job = data["job"]
        response["results"][device_id] = {
            "job_id": str(job.id),
            "status": job.status,
            "duration_ms": job.duration_ms,
            "outputs": data["outputs"],
        }

    return JsonResponse(response)


@csrf_exempt
@require_auth
@require_GET
def flowchart_templates(request):
    """List available flowchart templates."""
    templates = FlowchartTemplate.objects.filter(
        is_deleted=False,
    ).order_by("name")

    result = []
    for tpl in templates:
        result.append(
            {
                "id": str(tpl.id),
                "name": tpl.name,
                "description": tpl.description,
                "devices_used": tpl.devices_used,
                "is_shared": tpl.is_shared,
                "created_at": tpl.created_at.isoformat() if tpl.created_at else None,
            }
        )

    return JsonResponse({"templates": result})

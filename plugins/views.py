"""Demo canvas endpoint — staff/dev only.

POST /api/demo/canvas/run/
    Runs a plugin through the full lifecycle and returns serialized outputs.
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from accounts.permissions import require_auth
from syn.plugins.runner import run_plugin

logger = logging.getLogger(__name__)


@csrf_exempt
@require_auth
@require_POST
def canvas_run(request):
    """Run a plugin and return Job outputs."""
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    plugin_name = body.get("plugin_name")
    input_data = body.get("input_data", {})
    is_scratch = body.get("is_scratch", False)

    if not plugin_name:
        return JsonResponse({"error": "plugin_name is required"}, status=400)

    try:
        job = run_plugin(
            plugin_name,
            input_data,
            actor=request.user.email,
            tenant_id=getattr(request.user, "tenant_id", None),
            is_scratch=is_scratch,
        )
    except KeyError:
        return JsonResponse({"error": f"Unknown plugin: {plugin_name}"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

    outputs = []
    for out in job.outputs.all().order_by("created_at"):
        outputs.append(
            {
                "key": out.output_key,
                "type": out.output_type,
                "value": out.value_numeric if out.output_type == "metric" else out.value_json,
                "provenance": out.provenance,
                "measure_slug": out.measure_slug,
            }
        )

    return JsonResponse(
        {
            "job_id": str(job.id),
            "status": job.status,
            "duration_ms": job.duration_ms,
            "outputs": outputs,
        }
    )

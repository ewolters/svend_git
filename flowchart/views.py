"""Flowchart API endpoints.

POST /api/flowchart/run/             — execute a flowchart definition
GET  /api/flowchart/templates/       — list available templates
GET  /api/flowchart/templates/<id>/  — get template with full definition
GET  /api/flowchart/devices/         — list registered plugin devices + schemas
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from accounts.permissions import require_auth
from flowchart.engine import execute_flowchart
from flowchart.models import FlowchartInstance, FlowchartTemplate
from flowchart.service import (
    add_connection,
    add_device,
    create_instance,
    delete_instance,
    remove_connection,
    remove_device,
    validate_connection_request,
)
from syn.plugins.registry import get_registry

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


@csrf_exempt
@require_auth
@require_GET
def flowchart_template_detail(request, template_id):
    """Get a single template with full definition (for rendering)."""
    try:
        tpl = FlowchartTemplate.objects.get(id=template_id, is_deleted=False)
    except FlowchartTemplate.DoesNotExist:
        return JsonResponse({"error": "Template not found"}, status=404)

    return JsonResponse(
        {
            "id": str(tpl.id),
            "name": tpl.name,
            "description": tpl.description,
            "definition": tpl.definition,
            "devices_used": tpl.devices_used,
            "is_shared": tpl.is_shared,
            "created_at": tpl.created_at.isoformat() if tpl.created_at else None,
        }
    )


@csrf_exempt
@require_auth
@require_GET
def flowchart_devices(request):
    """List all registered plugin devices with their schemas."""
    registry = get_registry()
    return JsonResponse({"devices": registry.list_plugins()})


@csrf_exempt
@require_auth
def instance_create(request):
    """POST /api/flowchart/instances/ — create from template or blank."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    name = body.get("name", "Untitled")
    template_id = body.get("template_id")
    template = None

    if template_id:
        try:
            template = FlowchartTemplate.objects.get(id=template_id, is_deleted=False)
        except FlowchartTemplate.DoesNotExist:
            return JsonResponse({"error": "Template not found"}, status=404)

    inst = create_instance(
        template=template,
        name=name,
        user=request.user,
        actor=request.user.email,
        tenant_id=getattr(request.user, "tenant_id", None),
        is_scratch=body.get("is_scratch", False),
    )

    return JsonResponse(
        {
            "id": str(inst.id),
            "name": inst.name,
            "template_id": str(inst.template_id) if inst.template_id else None,
            "definition": inst.definition,
            "created_at": inst.created_at.isoformat(),
        },
        status=201,
    )


@csrf_exempt
@require_auth
def instance_detail(request, instance_id):
    """GET/DELETE /api/flowchart/instances/<id>/"""
    try:
        inst = FlowchartInstance.objects.get(id=instance_id, is_deleted=False, user=request.user)
    except FlowchartInstance.DoesNotExist:
        return JsonResponse({"error": "Instance not found"}, status=404)

    if request.method == "DELETE":
        delete_instance(inst, actor=request.user.email)
        return JsonResponse({"deleted": True})

    return JsonResponse(
        {
            "id": str(inst.id),
            "name": inst.name,
            "template_id": str(inst.template_id) if inst.template_id else None,
            "definition": inst.definition,
            "created_at": inst.created_at.isoformat(),
        }
    )


@csrf_exempt
@require_auth
def instance_add_device(request, instance_id):
    """POST /api/flowchart/instances/<id>/devices/"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        inst = FlowchartInstance.objects.get(id=instance_id, is_deleted=False, user=request.user)
    except FlowchartInstance.DoesNotExist:
        return JsonResponse({"error": "Instance not found"}, status=404)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        add_device(
            instance=inst,
            device_id=body["device_id"],
            plugin_name=body["plugin_name"],
            label=body.get("label", body["plugin_name"]),
            position=body.get("position", {"x": 100, "y": 100}),
            actor=request.user.email,
            port_schema=body.get("port_schema"),
        )
    except (ValueError, KeyError) as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"ok": True, "definition": inst.definition})


@csrf_exempt
@require_auth
def instance_remove_device(request, instance_id, device_id):
    """DELETE /api/flowchart/instances/<id>/devices/<device_id>/"""
    if request.method != "DELETE":
        return JsonResponse({"error": "DELETE required"}, status=405)

    try:
        inst = FlowchartInstance.objects.get(id=instance_id, is_deleted=False, user=request.user)
    except FlowchartInstance.DoesNotExist:
        return JsonResponse({"error": "Instance not found"}, status=404)

    try:
        remove_device(instance=inst, device_id=device_id, actor=request.user.email)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"ok": True, "definition": inst.definition})


@csrf_exempt
@require_auth
def instance_add_connection(request, instance_id):
    """POST /api/flowchart/instances/<id>/connections/"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        inst = FlowchartInstance.objects.get(id=instance_id, is_deleted=False, user=request.user)
    except FlowchartInstance.DoesNotExist:
        return JsonResponse({"error": "Instance not found"}, status=404)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        add_connection(
            instance=inst,
            source=body["source"],
            target=body["target"],
            actor=request.user.email,
        )
    except (ValueError, KeyError) as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"ok": True, "definition": inst.definition})


@csrf_exempt
@require_auth
def instance_remove_connection(request, instance_id):
    """POST /api/flowchart/instances/<id>/connections/remove/"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        inst = FlowchartInstance.objects.get(id=instance_id, is_deleted=False, user=request.user)
    except FlowchartInstance.DoesNotExist:
        return JsonResponse({"error": "Instance not found"}, status=404)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        remove_connection(
            instance=inst,
            source=body["source"],
            target=body["target"],
            actor=request.user.email,
        )
    except (ValueError, KeyError) as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"ok": True, "definition": inst.definition})


@csrf_exempt
@require_auth
def instance_validate_connection(request, instance_id):
    """POST /api/flowchart/instances/<id>/connections/validate/"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        inst = FlowchartInstance.objects.get(id=instance_id, is_deleted=False, user=request.user)
    except FlowchartInstance.DoesNotExist:
        return JsonResponse({"error": "Instance not found"}, status=404)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    result = validate_connection_request(inst.definition, body["source"], body["target"])
    return JsonResponse(result)


@csrf_exempt
@require_auth
@require_POST
def instance_run(request, instance_id):
    """POST /api/flowchart/instances/<id>/run/ — execute a flowchart instance."""
    try:
        inst = FlowchartInstance.objects.get(id=instance_id, is_deleted=False, user=request.user)
    except FlowchartInstance.DoesNotExist:
        return JsonResponse({"error": "Instance not found"}, status=404)

    try:
        results = execute_flowchart(
            definition=inst.definition,
            actor=request.user.email,
            tenant_id=getattr(request.user, "tenant_id", None),
            is_scratch=inst.is_scratch,
            flowchart_instance_id=inst.id,
        )
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception(f"[FLOWCHART] Instance execution failed: {e}")
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

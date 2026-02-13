"""Shared action item update/delete views.

These work for action items regardless of source (hoshin, a3, rca, fmea).
Auth: user must own the project the action item belongs to.
"""

import json

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid
from .models import ActionItem


@csrf_exempt
@gated_paid
@require_http_methods(["PUT", "PATCH"])
def update_action_item(request, action_id):
    """Update an action item from any source."""
    item = get_object_or_404(ActionItem, id=action_id, project__user=request.user)
    data = json.loads(request.body)

    for field in ["title", "description", "owner_name", "status",
                  "start_date", "end_date", "due_date"]:
        if field in data:
            setattr(item, field, data[field])

    if "progress" in data:
        item.progress = max(0, min(100, int(data["progress"])))
    if "sort_order" in data:
        item.sort_order = int(data["sort_order"])
    if "depends_on_id" in data:
        if data["depends_on_id"]:
            item.depends_on = ActionItem.objects.filter(
                id=data["depends_on_id"], project=item.project,
            ).first()
        else:
            item.depends_on = None

    item.save()
    return JsonResponse({"success": True, "action_item": item.to_dict()})


@csrf_exempt
@gated_paid
@require_http_methods(["DELETE"])
def delete_action_item(request, action_id):
    """Delete an action item from any source."""
    item = get_object_or_404(ActionItem, id=action_id, project__user=request.user)
    item.delete()
    return JsonResponse({"success": True})

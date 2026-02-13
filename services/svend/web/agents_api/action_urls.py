"""Shared action item update/delete routes.

Works for action items from any source (hoshin, a3, rca, fmea).
"""

from django.urls import path
from . import action_views

urlpatterns = [
    path("<uuid:action_id>/update/", action_views.update_action_item, name="action_update"),
    path("<uuid:action_id>/delete/", action_views.delete_action_item, name="action_delete"),
]

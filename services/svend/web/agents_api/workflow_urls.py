"""Workflow API URLs."""

from django.urls import path

from . import workflow_views as views

urlpatterns = [
    path("", views.workflows_list, name="workflows_list"),
    path("<str:workflow_id>/", views.workflow_detail, name="workflow_detail"),
    path("<str:workflow_id>/run/", views.workflow_run, name="workflow_run"),
]

"""Composable QMS URL routes."""

from django.urls import path

from . import pull_views, views

app_name = "qms"

urlpatterns = [
    # Templates
    path("templates/", views.template_list, name="template-list"),
    path("templates/<uuid:template_id>/", views.template_detail, name="template-detail"),
    # Artifacts
    path("artifacts/", views.artifact_list, name="artifact-list"),
    path("artifacts/create/", views.artifact_create, name="artifact-create"),
    path("artifacts/<uuid:artifact_id>/", views.artifact_detail, name="artifact-detail"),
    path("artifacts/<uuid:artifact_id>/update/", views.artifact_update, name="artifact-update"),
    path("artifacts/<uuid:artifact_id>/delete/", views.artifact_delete, name="artifact-delete"),
    # Workflows
    path("workflows/", views.workflow_list, name="workflow-list"),
    path("workflows/<uuid:workflow_id>/", views.workflow_detail, name="workflow-detail"),
    path("signal-types/", views.signal_type_list, name="signal-type-list"),
    # Pull contract
    path("pull/containers/", pull_views.container_list, name="pull-container-list"),
    path("pull/containers/<uuid:pk>/", pull_views.container_detail, name="pull-container-detail"),
    path("pull/containers/<uuid:container_id>/references/", pull_views.list_references, name="pull-list-references"),
    path("pull/artifacts/<uuid:artifact_id>/", pull_views.artifact_detail_view, name="pull-artifact-detail"),
    path(
        "pull/artifacts/<uuid:artifact_id>/references/", pull_views.register_reference, name="pull-register-reference"
    ),
    path("pull/artifacts/<uuid:artifact_id>/<path:key_path>/", pull_views.artifact_sub, name="pull-artifact-sub"),
]

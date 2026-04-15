"""FMEA pull contract URL routes."""

from django.urls import path

from . import pull_views

app_name = "fmea_pull"

urlpatterns = [
    path("containers/", pull_views.container_list, name="container_list"),
    path("containers/<uuid:pk>/", pull_views.container_detail, name="container_detail"),
    path("containers/<uuid:container_id>/references/", pull_views.list_references, name="list_references"),
    path("artifacts/<uuid:artifact_id>/", pull_views.artifact_detail, name="artifact_detail"),
    path("artifacts/<uuid:artifact_id>/references/", pull_views.register_reference, name="register_reference"),
    path("artifacts/<uuid:artifact_id>/<path:key_path>/", pull_views.artifact_sub, name="artifact_sub"),
]

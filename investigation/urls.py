"""URL routes for Investigation API — CANON-002 §13."""

from django.urls import path

from . import views

urlpatterns = [
    path(
        "",
        views.list_create_investigations,
        name="investigation_list_create",
    ),
    path(
        "<uuid:investigation_id>/",
        views.investigation_detail,
        name="investigation_detail",
    ),
    path(
        "<uuid:investigation_id>/transition/",
        views.transition_investigation,
        name="investigation_transition",
    ),
    path(
        "<uuid:investigation_id>/reopen/",
        views.reopen_investigation,
        name="investigation_reopen",
    ),
    path(
        "<uuid:investigation_id>/export/",
        views.export_investigation_view,
        name="investigation_export",
    ),
    path(
        "<uuid:investigation_id>/members/",
        views.manage_members,
        name="investigation_members",
    ),
    path(
        "<uuid:investigation_id>/graph/",
        views.get_graph,
        name="investigation_graph",
    ),
    path(
        "<uuid:investigation_id>/tools/",
        views.list_tools,
        name="investigation_tools",
    ),
]

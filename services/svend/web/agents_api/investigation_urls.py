"""URL routes for Investigation API — CANON-002 §13."""

from django.urls import path

from . import investigation_views

urlpatterns = [
    path(
        "",
        investigation_views.list_create_investigations,
        name="investigation_list_create",
    ),
    path(
        "<uuid:investigation_id>/",
        investigation_views.investigation_detail,
        name="investigation_detail",
    ),
    path(
        "<uuid:investigation_id>/transition/",
        investigation_views.transition_investigation,
        name="investigation_transition",
    ),
    path(
        "<uuid:investigation_id>/reopen/",
        investigation_views.reopen_investigation,
        name="investigation_reopen",
    ),
    path(
        "<uuid:investigation_id>/export/",
        investigation_views.export_investigation_view,
        name="investigation_export",
    ),
    path(
        "<uuid:investigation_id>/members/",
        investigation_views.manage_members,
        name="investigation_members",
    ),
    path(
        "<uuid:investigation_id>/graph/",
        investigation_views.get_graph,
        name="investigation_graph",
    ),
    path(
        "<uuid:investigation_id>/tools/",
        investigation_views.list_tools,
        name="investigation_tools",
    ),
]

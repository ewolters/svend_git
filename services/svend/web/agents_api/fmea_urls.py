"""FMEA API URLs."""

from django.urls import path

from . import fmea_views as views

urlpatterns = [
    # FMEA CRUD
    path("", views.list_fmeas, name="fmea_list"),
    path("create/", views.create_fmea, name="fmea_create"),
    path("<uuid:fmea_id>/", views.get_fmea, name="fmea_get"),
    path("<uuid:fmea_id>/update/", views.update_fmea, name="fmea_update"),
    path("<uuid:fmea_id>/delete/", views.delete_fmea, name="fmea_delete"),
    # Row CRUD
    path("<uuid:fmea_id>/rows/", views.add_row, name="fmea_add_row"),
    path(
        "<uuid:fmea_id>/rows/<uuid:row_id>/", views.update_row, name="fmea_update_row"
    ),
    path(
        "<uuid:fmea_id>/rows/<uuid:row_id>/delete/",
        views.delete_row,
        name="fmea_delete_row",
    ),
    path("<uuid:fmea_id>/reorder/", views.reorder_rows, name="fmea_reorder"),
    # Evidence linking
    path(
        "<uuid:fmea_id>/rows/<uuid:row_id>/link/",
        views.link_to_hypothesis,
        name="fmea_link_hypothesis",
    ),
    path(
        "<uuid:fmea_id>/rows/<uuid:row_id>/revise/",
        views.record_revision,
        name="fmea_record_revision",
    ),
    # Summary
    path("<uuid:fmea_id>/summary/", views.rpn_summary, name="fmea_rpn_summary"),
    # SPC ↔ FMEA closed loop (Phase C)
    path(
        "<uuid:fmea_id>/rows/<uuid:row_id>/spc-update/",
        views.spc_update_occurrence,
        name="fmea_spc_update",
    ),
    path(
        "<uuid:fmea_id>/rows/<uuid:row_id>/spc-cpk-update/",
        views.spc_cpk_update_occurrence,
        name="fmea_spc_cpk_update",
    ),
    # Action items
    path("<uuid:fmea_id>/actions/", views.list_fmea_actions, name="fmea_actions"),
    path(
        "<uuid:fmea_id>/rows/<uuid:row_id>/promote-action/",
        views.promote_fmea_action,
        name="fmea_promote_action",
    ),
    path(
        "<uuid:fmea_id>/rows/<uuid:row_id>/promote-capa/",
        views.promote_fmea_capa,
        name="fmea_promote_capa",
    ),
    path(
        "<uuid:fmea_id>/rows/<uuid:row_id>/promote-risk/",
        views.promote_fmea_risk,
        name="fmea_promote_risk",
    ),
    # FMEA → RCA bridge (QMS-001 §5.1 closed loop)
    path(
        "<uuid:fmea_id>/rows/<uuid:row_id>/investigate/",
        views.investigate_row,
        name="fmea_investigate",
    ),
    # Intelligence Layer (Phase 3)
    path("<uuid:fmea_id>/trending/", views.rpn_trending, name="fmea_trending"),
    path("patterns/", views.cross_fmea_patterns, name="fmea_patterns"),
    path(
        "<uuid:fmea_id>/suggest-failure-modes/",
        views.suggest_failure_modes,
        name="fmea_suggest",
    ),
]

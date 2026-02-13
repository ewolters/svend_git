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
    path("<uuid:fmea_id>/rows/<uuid:row_id>/", views.update_row, name="fmea_update_row"),
    path("<uuid:fmea_id>/rows/<uuid:row_id>/delete/", views.delete_row, name="fmea_delete_row"),
    path("<uuid:fmea_id>/reorder/", views.reorder_rows, name="fmea_reorder"),

    # Evidence linking
    path("<uuid:fmea_id>/rows/<uuid:row_id>/link/", views.link_to_hypothesis, name="fmea_link_hypothesis"),
    path("<uuid:fmea_id>/rows/<uuid:row_id>/revise/", views.record_revision, name="fmea_record_revision"),

    # Summary
    path("<uuid:fmea_id>/summary/", views.rpn_summary, name="fmea_rpn_summary"),
]

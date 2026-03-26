"""QMS cross-module intelligence API URLs."""

from django.urls import path

from . import iso_views, qms_views

urlpatterns = [
    path("dashboard/", qms_views.qms_dashboard, name="qms_dashboard"),
    # QMS Attachments (ISO 9001 §7.5)
    path(
        "attachments/", iso_views.qms_attachment_list_create, name="qms_attachment_list"
    ),
    path(
        "attachments/<uuid:attachment_id>/",
        iso_views.qms_attachment_delete,
        name="qms_attachment_delete",
    ),
]

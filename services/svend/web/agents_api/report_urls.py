"""Report API URLs (CAPA, 8D, etc.)."""

from django.urls import path
from . import report_views

urlpatterns = [
    path("types/", report_views.list_report_types, name="report_types"),
    path("", report_views.list_reports, name="report_list"),
    path("create/", report_views.create_report, name="report_create"),
    path("<uuid:report_id>/", report_views.get_report, name="report_get"),
    path("<uuid:report_id>/update/", report_views.update_report, name="report_update"),
    path("<uuid:report_id>/delete/", report_views.delete_report, name="report_delete"),
    path("<uuid:report_id>/import/", report_views.import_to_report, name="report_import"),
    path("<uuid:report_id>/auto-populate/", report_views.auto_populate_report, name="report_auto_populate"),
    path("<uuid:report_id>/embed-diagram/", report_views.embed_diagram, name="report_embed_diagram"),
    path("<uuid:report_id>/diagram/<str:diagram_id>/", report_views.remove_diagram, name="report_remove_diagram"),
]

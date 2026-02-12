"""A3 Report API URLs."""

from django.urls import path
from . import a3_views

urlpatterns = [
    path("", a3_views.list_a3_reports, name="a3_list"),
    path("create/", a3_views.create_a3_report, name="a3_create"),
    path("<uuid:report_id>/", a3_views.get_a3_report, name="a3_get"),
    path("<uuid:report_id>/update/", a3_views.update_a3_report, name="a3_update"),
    path("<uuid:report_id>/delete/", a3_views.delete_a3_report, name="a3_delete"),
    path("<uuid:report_id>/import/", a3_views.import_to_a3, name="a3_import"),
    path("<uuid:report_id>/auto-populate/", a3_views.auto_populate_a3, name="a3_auto_populate"),
    path("<uuid:report_id>/embed-diagram/", a3_views.embed_diagram, name="a3_embed_diagram"),
    path("<uuid:report_id>/diagram/<str:diagram_id>/", a3_views.remove_diagram, name="a3_remove_diagram"),
]

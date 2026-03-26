"""Privacy API URL configuration (PRIV-001)."""

from django.urls import path

from . import privacy_views

app_name = "privacy"

urlpatterns = [
    path("exports/", privacy_views.exports_collection, name="exports"),
    path(
        "exports/<uuid:export_id>/", privacy_views.export_resource, name="export_detail"
    ),
]

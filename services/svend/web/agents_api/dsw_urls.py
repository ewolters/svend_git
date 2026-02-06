"""DSW (Decision Science Workbench) API URLs."""

from django.urls import path

from . import dsw_views as views

urlpatterns = [
    path("from-intent/", views.dsw_from_intent, name="dsw_from_intent"),
    path("from-data/", views.dsw_from_data, name="dsw_from_data"),
    path("download/<str:result_id>/<str:file_type>/", views.dsw_download, name="dsw_download"),

    # Note: Scrub is exposed via /api/triage/ endpoints (triage_urls.py)

    # Saved models
    path("models/", views.list_models, name="list_models"),
    path("models/save/", views.save_model_from_cache, name="save_model"),
    path("models/<uuid:model_id>/", views.download_model, name="download_model"),
    path("models/<uuid:model_id>/delete/", views.delete_model, name="delete_model"),
    path("models/<uuid:model_id>/run/", views.run_model, name="run_model"),

    # Analysis Workbench
    path("analysis/", views.run_analysis, name="run_analysis"),
    path("execute/", views.execute_code, name="execute_code"),
    path("generate-code/", views.generate_code, name="generate_code"),
    path("upload-data/", views.upload_data, name="upload_data"),
    path("analyst/", views.analyst_assistant, name="analyst_assistant"),
    path("transform/", views.transform_data, name="transform_data"),
    path("download/", views.download_data, name="download_data"),
    path("triage/", views.triage_data, name="triage_data"),
    path("triage/scan/", views.triage_scan, name="triage_scan"),
]

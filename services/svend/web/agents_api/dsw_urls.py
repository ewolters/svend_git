"""DSW (Decision Science Workbench) API URLs."""

from django.urls import path

from . import autopilot_views as views_auto
from . import dsw_views as views

urlpatterns = [
    path("from-intent/", views.dsw_from_intent, name="dsw_from_intent"),
    path("from-data/", views.dsw_from_data, name="dsw_from_data"),
    path("download/<str:result_id>/<str:file_type>/", views.dsw_download, name="dsw_download"),

    # Note: Scrub is exposed via /api/triage/ endpoints (triage_urls.py)

    # Saved models
    path("models/", views.list_models, name="list_models"),
    path("models/summary/", views.models_summary, name="models_summary"),
    path("models/save/", views.save_model_from_cache, name="save_model"),
    path("models/<uuid:model_id>/", views.download_model, name="download_model"),
    path("models/<uuid:model_id>/delete/", views.delete_model, name="delete_model"),
    path("models/<uuid:model_id>/run/", views.run_model, name="run_model"),
    path("models/<uuid:model_id>/optimize/", views.optimize_model, name="optimize_model"),
    path("models/<uuid:model_id>/versions/", views.model_versions, name="model_versions"),
    path("models/<uuid:model_id>/report/", views.model_report, name="model_report"),
    path("models/<uuid:model_id>/retrain/", views_auto.retrain_model, name="retrain_model"),

    # Autopilot pipelines
    path("autopilot/clean-train/", views_auto.autopilot_clean_train, name="autopilot_clean_train"),
    path("autopilot/full-pipeline/", views_auto.autopilot_full_pipeline, name="autopilot_full_pipeline"),
    path("autopilot/augment-train/", views_auto.autopilot_augment_train, name="autopilot_augment_train"),

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

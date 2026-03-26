"""DSW (Decision Science Workbench) API URLs."""

from django.urls import path

from . import autopilot_views as views_auto
from . import dsw_views as views  # Phase 1: run_analysis wrapper only
from .dsw.endpoints_ml import (
    delete_model,
    download_model,
    dsw_download,
    dsw_from_data,
    dsw_from_intent,
    list_models,
    model_report,
    model_versions,
    models_summary,
    optimize_model,
    run_model,
    save_model_from_cache,
)

urlpatterns = [
    # Phase 4: ML endpoints now served from dsw/endpoints_ml.py
    path("from-intent/", dsw_from_intent, name="dsw_from_intent"),
    path("from-data/", dsw_from_data, name="dsw_from_data"),
    path(
        "download/<str:result_id>/<str:file_type>/", dsw_download, name="dsw_download"
    ),
    # Note: Scrub is exposed via /api/triage/ endpoints (triage_urls.py)
    # Saved models
    path("models/", list_models, name="list_models"),
    path("models/summary/", models_summary, name="models_summary"),
    path("models/save/", save_model_from_cache, name="save_model"),
    path("models/<uuid:model_id>/", download_model, name="download_model"),
    path("models/<uuid:model_id>/delete/", delete_model, name="delete_model"),
    path("models/<uuid:model_id>/run/", run_model, name="run_model"),
    path("models/<uuid:model_id>/optimize/", optimize_model, name="optimize_model"),
    path("models/<uuid:model_id>/versions/", model_versions, name="model_versions"),
    path("models/<uuid:model_id>/report/", model_report, name="model_report"),
    path(
        "models/<uuid:model_id>/retrain/",
        views_auto.retrain_model,
        name="retrain_model",
    ),
    # Autopilot pipelines
    path(
        "autopilot/clean-train/",
        views_auto.autopilot_clean_train,
        name="autopilot_clean_train",
    ),
    path(
        "autopilot/full-pipeline/",
        views_auto.autopilot_full_pipeline,
        name="autopilot_full_pipeline",
    ),
    path(
        "autopilot/augment-train/",
        views_auto.autopilot_augment_train,
        name="autopilot_augment_train",
    ),
    # Analysis Workbench
    path("analysis/", views.run_analysis, name="run_analysis"),  # Phase 1 wrapper
    path("explain-selection/", views.explain_selection, name="dsw_explain_selection"),
    path(
        "hypothesis-timeline/",
        views.hypothesis_timeline,
        name="dsw_hypothesis_timeline",
    ),
    # Phase 5 reverted — data endpoints back on monolith for testing
    path("execute/", views.execute_code, name="execute_code"),
    path("generate-code/", views.generate_code, name="generate_code"),
    path("upload-data/", views.upload_data, name="upload_data"),
    path("analyst/", views.analyst_assistant, name="analyst_assistant"),
    path("transform/", views.transform_data, name="transform_data"),
    path("download/", views.download_data, name="download_data"),
    path("retrieve-data/", views.retrieve_data, name="retrieve_data"),
    path("triage/", views.triage_data, name="triage_data"),
    path("triage/scan/", views.triage_scan, name="triage_scan"),
]

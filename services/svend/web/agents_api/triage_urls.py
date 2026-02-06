"""Triage API URL configuration."""

from django.urls import path
from . import triage_views

urlpatterns = [
    path("clean/", triage_views.triage_clean, name="triage_clean"),
    path("preview/", triage_views.triage_preview, name="triage_preview"),
    path("datasets/", triage_views.list_datasets, name="list_datasets"),
    path("<str:job_id>/download/", triage_views.triage_download, name="triage_download"),
    path("<str:job_id>/report/", triage_views.triage_report, name="triage_report"),
    path("<str:job_id>/load/", triage_views.load_dataset, name="load_dataset"),
]

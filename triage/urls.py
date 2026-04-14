"""Triage API URL configuration."""

from django.urls import path

from . import views

urlpatterns = [
    path("clean/", views.triage_clean, name="triage_clean"),
    path("preview/", views.triage_preview, name="triage_preview"),
    path("datasets/", views.list_datasets, name="list_datasets"),
    path("<str:job_id>/download/", views.triage_download, name="triage_download"),
    path("<str:job_id>/report/", views.triage_report, name="triage_report"),
    path("<str:job_id>/load/", views.load_dataset, name="load_dataset"),
]

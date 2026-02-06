"""Forge URL configuration."""

from django.urls import path
from . import views

app_name = "forge"

urlpatterns = [
    # Health
    path("health", views.health, name="health"),

    # Generation
    path("generate", views.generate, name="generate"),
    path("jobs/<uuid:job_id>", views.job_status, name="job_status"),
    path("jobs/<uuid:job_id>/result", views.job_result, name="job_result"),
    path("download/<uuid:job_id>", views.download, name="download"),

    # Schemas
    path("schemas", views.list_schemas, name="list_schemas"),

    # Usage
    path("usage", views.usage, name="usage"),
]

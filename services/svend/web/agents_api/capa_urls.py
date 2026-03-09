"""CAPA URL configuration."""

from django.urls import path

from . import capa_views

urlpatterns = [
    path("", capa_views.capa_list_create, name="capa_list"),
    path("stats/", capa_views.capa_stats, name="capa_stats"),
    path("copq/", capa_views.copq_summary, name="capa_copq_summary"),
    path("recurrence/", capa_views.recurrence_report, name="capa_recurrence_report"),
    path("<uuid:capa_id>/", capa_views.capa_detail, name="capa_detail"),
    path("<uuid:capa_id>/launch-rca/", capa_views.capa_launch_rca, name="capa_launch_rca"),
]

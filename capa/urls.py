"""CAPA URL configuration."""

from django.urls import path

from . import views

urlpatterns = [
    path("", views.capa_list_create, name="capa_list"),
    path("stats/", views.capa_stats, name="capa_stats"),
    path("copq/", views.copq_summary, name="capa_copq_summary"),
    path("recurrence/", views.recurrence_report, name="capa_recurrence_report"),
    path("<uuid:capa_id>/", views.capa_detail, name="capa_detail"),
    path("<uuid:capa_id>/launch-rca/", views.capa_launch_rca, name="capa_launch_rca"),
]

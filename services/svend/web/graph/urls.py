"""Graph API URL configuration — GRAPH-001 §11."""

from django.urls import path

from . import views

app_name = "graph"

urlpatterns = [
    path("data/", views.graph_data, name="graph_data"),
    path("node/<uuid:node_id>/", views.node_detail, name="node_detail"),
    path("edge/<uuid:edge_id>/", views.edge_detail, name="edge_detail"),
    path("gaps/", views.gap_report, name="gap_report"),
    path("seed/", views.seed_from_fmis, name="seed_from_fmis"),
    path("seed/confirm/", views.confirm_seed, name="confirm_seed"),
]

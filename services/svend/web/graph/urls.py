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
    # S1-3: Search (command palette)
    path("search/", views.search, name="search"),
    # S1-4: Knowledge health
    path("health/", views.knowledge_health, name="knowledge_health"),
    # S1-6: Activity feed
    path("activity/<str:entity_type>/<uuid:entity_id>/", views.activity_feed, name="activity_feed"),
    # S1-7: Workflow gates
    path("gates/<str:entity_type>/<uuid:entity_id>/", views.workflow_gates, name="workflow_gates"),
]

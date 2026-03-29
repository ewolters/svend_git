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
    # Configuration system
    path("config/", views.config_list, name="config_list"),
    path("config/set/", views.config_set, name="config_set"),
    path("config/preset/", views.config_apply_preset, name="config_apply_preset"),
    path("config/presets/", views.config_presets, name="config_presets"),
    path("config/domains/", views.config_domains, name="config_domains"),
    # Document rendering (ForgeDoc)
    path("documents/render/", views.document_render, name="document_render"),
]

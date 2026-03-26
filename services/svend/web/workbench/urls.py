"""URL routes for Workbench API."""

from django.urls import path

from . import graph_views, views

app_name = "workbench"

urlpatterns = [
    # ==========================================================================
    # Workbenches
    # ==========================================================================
    path("", views.list_workbenches, name="list"),
    path("create/", views.create_workbench, name="create"),
    path("<uuid:workbench_id>/", views.get_workbench, name="detail"),
    path("<uuid:workbench_id>/update/", views.update_workbench, name="update"),
    path("<uuid:workbench_id>/delete/", views.delete_workbench, name="delete"),
    # Import/Export
    path("import/", views.import_workbench, name="import"),
    path("<uuid:workbench_id>/export/", views.export_workbench, name="export"),
    # Artifacts
    path(
        "<uuid:workbench_id>/artifacts/", views.create_artifact, name="create_artifact"
    ),
    path(
        "<uuid:workbench_id>/artifacts/<uuid:artifact_id>/",
        views.get_artifact,
        name="get_artifact",
    ),
    path(
        "<uuid:workbench_id>/artifacts/<uuid:artifact_id>/update/",
        views.update_artifact,
        name="update_artifact",
    ),
    path(
        "<uuid:workbench_id>/artifacts/<uuid:artifact_id>/delete/",
        views.delete_artifact,
        name="delete_artifact",
    ),
    # Connections
    path("<uuid:workbench_id>/connect/", views.connect_artifacts, name="connect"),
    path(
        "<uuid:workbench_id>/disconnect/", views.disconnect_artifacts, name="disconnect"
    ),
    # Template-specific
    path(
        "<uuid:workbench_id>/advance-phase/", views.advance_phase, name="advance_phase"
    ),
    # Guide
    path(
        "<uuid:workbench_id>/guide/observe/",
        views.add_guide_observation,
        name="add_observation",
    ),
    path(
        "<uuid:workbench_id>/guide/<int:observation_index>/acknowledge/",
        views.acknowledge_observation,
        name="acknowledge_observation",
    ),
    # ==========================================================================
    # Workbench Knowledge Graph
    # ==========================================================================
    # Graph CRUD
    path("<uuid:workbench_id>/graph/", graph_views.get_graph, name="get_graph"),
    path(
        "<uuid:workbench_id>/graph/clear/", graph_views.clear_graph, name="clear_graph"
    ),
    # Nodes
    path("<uuid:workbench_id>/graph/nodes/", graph_views.get_nodes, name="get_nodes"),
    path("<uuid:workbench_id>/graph/nodes/add/", graph_views.add_node, name="add_node"),
    path(
        "<uuid:workbench_id>/graph/nodes/<str:node_id>/delete/",
        graph_views.remove_node,
        name="remove_node",
    ),
    # Edges (causal vectors)
    path("<uuid:workbench_id>/graph/edges/", graph_views.get_edges, name="get_edges"),
    path("<uuid:workbench_id>/graph/edges/add/", graph_views.add_edge, name="add_edge"),
    path(
        "<uuid:workbench_id>/graph/edges/<str:edge_id>/weight/",
        graph_views.update_edge_weight,
        name="update_edge_weight",
    ),
    # Evidence & Bayesian updates
    path(
        "<uuid:workbench_id>/graph/evidence/apply/",
        graph_views.apply_evidence,
        name="apply_evidence",
    ),
    path(
        "<uuid:workbench_id>/graph/expansion/check/",
        graph_views.check_expansion,
        name="check_expansion",
    ),
    # Graph traversal
    path(
        "<uuid:workbench_id>/graph/chain/<str:from_node>/<str:to_node>/",
        graph_views.get_causal_chain,
        name="get_causal_chain",
    ),
    path(
        "<uuid:workbench_id>/graph/upstream/<str:node_id>/",
        graph_views.get_upstream_causes,
        name="get_upstream_causes",
    ),
    # Expansion signals
    path(
        "<uuid:workbench_id>/graph/expansions/",
        graph_views.get_expansion_signals,
        name="get_expansion_signals",
    ),
    path(
        "<uuid:workbench_id>/graph/expansions/<str:signal_id>/resolve/",
        graph_views.resolve_expansion,
        name="resolve_expansion",
    ),
    # Epistemic log
    path(
        "<uuid:workbench_id>/epistemic-log/",
        graph_views.get_epistemic_log,
        name="get_epistemic_log",
    ),
    path(
        "<uuid:workbench_id>/epistemic-log/<uuid:log_id>/outcome/",
        graph_views.mark_log_outcome,
        name="mark_log_outcome",
    ),
]

"""URL routes for Workbench API."""

from django.urls import path
from . import views
from . import graph_views

app_name = "workbench"

urlpatterns = [
    # ==========================================================================
    # Projects
    # ==========================================================================
    path("projects/", views.list_projects, name="list_projects"),
    path("projects/create/", views.create_project, name="create_project"),
    path("projects/<uuid:project_id>/", views.get_project, name="get_project"),
    path("projects/<uuid:project_id>/update/", views.update_project, name="update_project"),
    path("projects/<uuid:project_id>/delete/", views.delete_project, name="delete_project"),
    path("projects/<uuid:project_id>/workbenches/add/", views.add_workbench_to_project, name="add_workbench_to_project"),
    path("projects/<uuid:project_id>/workbenches/<uuid:workbench_id>/remove/", views.remove_workbench_from_project, name="remove_workbench_from_project"),

    # ==========================================================================
    # Hypotheses
    # ==========================================================================
    path("projects/<uuid:project_id>/hypotheses/", views.list_hypotheses, name="list_hypotheses"),
    path("projects/<uuid:project_id>/hypotheses/create/", views.create_hypothesis, name="create_hypothesis"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/", views.get_hypothesis, name="get_hypothesis"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/update/", views.update_hypothesis, name="update_hypothesis"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/delete/", views.delete_hypothesis, name="delete_hypothesis"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/probability/", views.update_hypothesis_probability, name="update_hypothesis_probability"),

    # ==========================================================================
    # Evidence
    # ==========================================================================
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/evidence/", views.list_evidence, name="list_evidence"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/evidence/create/", views.create_evidence, name="create_evidence"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/evidence/<uuid:evidence_id>/", views.get_evidence, name="get_evidence"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/evidence/<uuid:evidence_id>/delete/", views.delete_evidence, name="delete_evidence"),

    # ==========================================================================
    # Conversations
    # ==========================================================================
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/conversations/", views.list_conversations, name="list_conversations"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/conversations/create/", views.create_conversation, name="create_conversation"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/conversations/<uuid:conversation_id>/", views.get_conversation, name="get_conversation"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/conversations/<uuid:conversation_id>/update/", views.update_conversation, name="update_conversation"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/conversations/<uuid:conversation_id>/message/", views.add_message, name="add_message"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/conversations/<uuid:conversation_id>/delete/", views.delete_conversation, name="delete_conversation"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/conversations/<uuid:conversation_id>/refresh-context/", views.refresh_conversation_context, name="refresh_conversation_context"),

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
    path("<uuid:workbench_id>/artifacts/", views.create_artifact, name="create_artifact"),
    path("<uuid:workbench_id>/artifacts/<uuid:artifact_id>/", views.get_artifact, name="get_artifact"),
    path("<uuid:workbench_id>/artifacts/<uuid:artifact_id>/update/", views.update_artifact, name="update_artifact"),
    path("<uuid:workbench_id>/artifacts/<uuid:artifact_id>/delete/", views.delete_artifact, name="delete_artifact"),

    # Connections
    path("<uuid:workbench_id>/connect/", views.connect_artifacts, name="connect"),
    path("<uuid:workbench_id>/disconnect/", views.disconnect_artifacts, name="disconnect"),

    # Template-specific
    path("<uuid:workbench_id>/advance-phase/", views.advance_phase, name="advance_phase"),

    # Guide
    path("<uuid:workbench_id>/guide/observe/", views.add_guide_observation, name="add_observation"),
    path("<uuid:workbench_id>/guide/<int:observation_index>/acknowledge/", views.acknowledge_observation, name="acknowledge_observation"),

    # ==========================================================================
    # Project Knowledge Graph (for connecting hypotheses)
    # ==========================================================================
    path("projects/<uuid:project_id>/graph/", graph_views.get_project_graph, name="get_project_graph"),
    path("projects/<uuid:project_id>/graph/hypotheses/<uuid:hypothesis_id>/add/", graph_views.add_hypothesis_to_graph, name="add_hypothesis_to_graph"),
    path("projects/<uuid:project_id>/graph/hypotheses/<uuid:hypothesis_id>/connections/", graph_views.get_hypothesis_connections, name="get_hypothesis_connections"),
    path("projects/<uuid:project_id>/graph/connect/", graph_views.connect_hypotheses, name="connect_hypotheses"),

    # ==========================================================================
    # Workbench Knowledge Graph
    # ==========================================================================

    # Graph CRUD
    path("<uuid:workbench_id>/graph/", graph_views.get_graph, name="get_graph"),
    path("<uuid:workbench_id>/graph/clear/", graph_views.clear_graph, name="clear_graph"),

    # Nodes
    path("<uuid:workbench_id>/graph/nodes/", graph_views.get_nodes, name="get_nodes"),
    path("<uuid:workbench_id>/graph/nodes/add/", graph_views.add_node, name="add_node"),
    path("<uuid:workbench_id>/graph/nodes/<str:node_id>/delete/", graph_views.remove_node, name="remove_node"),

    # Edges (causal vectors)
    path("<uuid:workbench_id>/graph/edges/", graph_views.get_edges, name="get_edges"),
    path("<uuid:workbench_id>/graph/edges/add/", graph_views.add_edge, name="add_edge"),
    path("<uuid:workbench_id>/graph/edges/<str:edge_id>/weight/", graph_views.update_edge_weight, name="update_edge_weight"),

    # Evidence & Bayesian updates
    path("<uuid:workbench_id>/graph/evidence/apply/", graph_views.apply_evidence, name="apply_evidence"),
    path("<uuid:workbench_id>/graph/expansion/check/", graph_views.check_expansion, name="check_expansion"),

    # Graph traversal
    path("<uuid:workbench_id>/graph/chain/<str:from_node>/<str:to_node>/", graph_views.get_causal_chain, name="get_causal_chain"),
    path("<uuid:workbench_id>/graph/upstream/<str:node_id>/", graph_views.get_upstream_causes, name="get_upstream_causes"),

    # Expansion signals
    path("<uuid:workbench_id>/graph/expansions/", graph_views.get_expansion_signals, name="get_expansion_signals"),
    path("<uuid:workbench_id>/graph/expansions/<str:signal_id>/resolve/", graph_views.resolve_expansion, name="resolve_expansion"),

    # Epistemic log
    path("<uuid:workbench_id>/epistemic-log/", graph_views.get_epistemic_log, name="get_epistemic_log"),
    path("<uuid:workbench_id>/epistemic-log/<uuid:log_id>/outcome/", graph_views.mark_log_outcome, name="mark_log_outcome"),
]

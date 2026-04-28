"""PROVA URL routes."""

from django.urls import path

from . import views

app_name = "prova"

urlpatterns = [
    # Operating graph
    path("graph/", views.operating_graph_detail, name="graph-detail"),
    path("graph/versions/", views.graph_versions, name="graph-versions"),
    path("graph/versions/<uuid:version_id>/rollback/", views.graph_rollback, name="graph-rollback"),
    path("graph/evaluate/", views.evaluate_graph, name="graph-evaluate"),
    # Nodes
    path("nodes/", views.node_list, name="node-list"),
    path("nodes/create/", views.node_create, name="node-create"),
    # Edges
    path("edges/create/", views.edge_create, name="edge-create"),
    # Working graphs
    path("working/", views.working_graph_list, name="working-list"),
    path("working/create/", views.working_graph_create, name="working-create"),
    # Hypotheses (scoped to working graph)
    path("working/<uuid:working_graph_id>/hypotheses/", views.hypothesis_list, name="hypothesis-list"),
    path("working/<uuid:working_graph_id>/hypotheses/create/", views.hypothesis_create, name="hypothesis-create"),
    path("hypotheses/<uuid:hypothesis_id>/", views.hypothesis_detail, name="hypothesis-detail"),
    # Trials
    path("hypotheses/<uuid:hypothesis_id>/trials/create/", views.trial_create, name="trial-create"),
    path("hypotheses/<uuid:hypothesis_id>/trials/proposals/", views.trial_proposals, name="trial-proposals"),
    path("trials/<uuid:trial_id>/data/", views.trial_submit_data, name="trial-data"),
    path("trials/<uuid:trial_id>/evaluate/", views.trial_evaluate, name="trial-evaluate"),
    path("trials/<uuid:trial_id>/promote/", views.trial_promote, name="trial-promote"),
    # Conflicts
    path("conflicts/", views.conflict_list, name="conflict-list"),
    path("conflicts/<uuid:conflict_id>/", views.conflict_update, name="conflict-update"),
    # Propagation signals
    path("signals/", views.signal_list, name="signal-list"),
    path("signals/<uuid:signal_id>/", views.signal_acknowledge, name="signal-acknowledge"),
    # Bridge — universal tool integration
    path("bridge/", views.bridge_integrate, name="bridge-integrate"),
    # SPC → PROVA → RCA investigation flow
    path("investigate/", views.investigate_ooc, name="investigate-ooc"),
]

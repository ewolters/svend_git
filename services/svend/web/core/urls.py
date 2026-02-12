"""Core API URLs."""

from django.urls import path
from . import views

urlpatterns = [
    # Projects
    path("projects/", views.project_list, name="project_list"),
    path("projects/<uuid:project_id>/", views.project_detail, name="project_detail"),
    path("projects/<uuid:project_id>/hub/", views.project_hub, name="project_hub"),
    path("projects/<uuid:project_id>/advance-phase/", views.project_advance_phase, name="project_advance_phase"),
    path("projects/<uuid:project_id>/recalculate/", views.project_recalculate, name="project_recalculate"),

    # Hypotheses
    path("projects/<uuid:project_id>/hypotheses/", views.hypothesis_list, name="hypothesis_list"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/", views.hypothesis_detail, name="hypothesis_detail"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/recalculate/", views.hypothesis_recalculate, name="hypothesis_recalculate"),

    # Evidence
    path("projects/<uuid:project_id>/evidence/", views.evidence_list, name="evidence_list"),
    path("projects/<uuid:project_id>/hypotheses/<uuid:hypothesis_id>/link-evidence/", views.link_evidence, name="link_evidence"),
    path("projects/<uuid:project_id>/suggest-lr/", views.suggest_likelihood_ratio, name="suggest_lr"),

    # Evidence from tools (Coder, DSW)
    path("evidence/from-code/", views.create_evidence_from_code, name="evidence_from_code"),
    path("evidence/from-analysis/", views.create_evidence_from_analysis, name="evidence_from_analysis"),

    # Datasets
    path("projects/<uuid:project_id>/datasets/", views.dataset_list, name="dataset_list"),
    path("projects/<uuid:project_id>/datasets/<uuid:dataset_id>/", views.dataset_detail, name="dataset_detail"),
    path("projects/<uuid:project_id>/datasets/<uuid:dataset_id>/data/", views.dataset_data, name="dataset_data"),

    # Experiment Designs
    path("projects/<uuid:project_id>/designs/", views.experiment_design_list, name="experiment_design_list"),
    path("projects/<uuid:project_id>/designs/<uuid:design_id>/", views.experiment_design_detail, name="experiment_design_detail"),
    path("projects/<uuid:project_id>/designs/<uuid:design_id>/review/", views.review_design_execution, name="review_design_execution"),

    # Knowledge Graph
    path("graph/", views.knowledge_graph, name="knowledge_graph"),
    path("graph/entities/", views.entity_list, name="entity_list"),
    path("graph/entities/<uuid:entity_id>/", views.entity_detail, name="entity_detail"),
    path("graph/relationships/", views.relationship_list, name="relationship_list"),
    path("graph/check-consistency/", views.check_consistency, name="check_consistency"),
]

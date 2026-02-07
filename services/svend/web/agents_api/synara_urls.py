"""Synara (Belief Engine) API URLs."""

from django.urls import path

from . import synara_views as views

urlpatterns = [
    # Hypotheses
    path("<str:workbench_id>/hypotheses/", views.get_hypotheses, name="synara_hypotheses"),
    path("<str:workbench_id>/hypotheses/add/", views.add_hypothesis, name="synara_add_hypothesis"),
    path("<str:workbench_id>/hypotheses/<str:hypothesis_id>/delete/", views.delete_hypothesis, name="synara_delete_hypothesis"),
    path("<str:workbench_id>/hypotheses/<str:hypothesis_id>/explain/", views.explain_hypothesis, name="synara_explain"),
    path("<str:workbench_id>/hypotheses/<str:hypothesis_id>/chains/", views.get_causal_chains, name="synara_chains"),
    path("<str:workbench_id>/hypotheses/<str:hypothesis_id>/evaluate/", views.evaluate_workbench_hypothesis, name="synara_evaluate_hypothesis"),

    # Causal Links
    path("<str:workbench_id>/links/", views.get_links, name="synara_links"),
    path("<str:workbench_id>/links/add/", views.add_link, name="synara_add_link"),

    # Evidence
    path("<str:workbench_id>/evidence/", views.get_evidence, name="synara_evidence"),
    path("<str:workbench_id>/evidence/add/", views.add_evidence, name="synara_add_evidence"),

    # Expansion Signals
    path("<str:workbench_id>/expansions/", views.get_expansions, name="synara_expansions"),
    path("<str:workbench_id>/expansions/<str:signal_id>/resolve/", views.resolve_expansion, name="synara_resolve"),

    # Analysis
    path("<str:workbench_id>/state/", views.get_belief_state, name="synara_state"),

    # LLM Integration (prompt-only — client calls LLM)
    path("<str:workbench_id>/prompts/validation/", views.get_validation_prompt, name="synara_validation_prompt"),
    path("<str:workbench_id>/prompts/hypothesis/<str:signal_id>/", views.get_hypothesis_prompt, name="synara_hypothesis_prompt"),
    path("<str:workbench_id>/validation/apply/", views.apply_validation_result, name="synara_apply_validation"),

    # LLM Integration (server-side — server calls Claude API)
    path("<str:workbench_id>/llm/validate/", views.llm_validate, name="synara_llm_validate"),
    path("<str:workbench_id>/llm/generate-hypotheses/<str:signal_id>/", views.llm_generate_hypotheses, name="synara_llm_generate"),
    path("<str:workbench_id>/llm/interpret/", views.llm_interpret_evidence, name="synara_llm_interpret"),
    path("<str:workbench_id>/llm/document/", views.llm_document, name="synara_llm_document"),

    # DSL: Formal Hypothesis Language
    path("<str:workbench_id>/dsl/parse/", views.parse_hypothesis_dsl, name="synara_dsl_parse"),
    path("<str:workbench_id>/dsl/validate/", views.validate_hypothesis_dsl, name="synara_dsl_validate"),
    path("<str:workbench_id>/dsl/evaluate/", views.evaluate_hypothesis_dsl, name="synara_dsl_evaluate"),
    path("<str:workbench_id>/dsl/add/", views.add_formal_hypothesis, name="synara_dsl_add"),

    # Serialization
    path("<str:workbench_id>/export/", views.export_synara, name="synara_export"),
    path("<str:workbench_id>/import/", views.import_synara, name="synara_import"),
]

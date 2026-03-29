"""URL routes for Loop API — LOOP-001 §3, §5, §6, §7, §16."""

from django.urls import path

from . import views

urlpatterns = [
    # Dashboard data (§16.2)
    path("dashboard/", views.dashboard_data, name="loop_dashboard_data"),
    # CI Readiness Score (§10)
    path("readiness/", views.readiness_score, name="loop_readiness_score"),
    # Signals (§3.1)
    path("signals/", views.signal_list_create, name="loop_signal_list_create"),
    path("signals/<uuid:signal_id>/", views.signal_detail, name="loop_signal_detail"),
    # Commitments (§3.3)
    path("commitments/", views.commitment_list_create, name="loop_commitment_list_create"),
    path("commitments/<uuid:commitment_id>/", views.commitment_detail, name="loop_commitment_detail"),
    # Commitment Resources (QMS-002 §2.2)
    path(
        "commitments/<uuid:commitment_id>/resources/",
        views.commitment_resource_list_create,
        name="loop_commitment_resources",
    ),
    path(
        "commitments/<uuid:commitment_id>/resources/<uuid:resource_id>/",
        views.commitment_resource_detail,
        name="loop_commitment_resource_detail",
    ),
    # Mode Transitions (§3.2) — read-only
    path("transitions/", views.transition_list, name="loop_transition_list"),
    # Investigation → Commitment bridge
    path(
        "investigations/<uuid:investigation_id>/commitments/",
        views.investigation_commitments,
        name="loop_investigation_commitments",
    ),
    # Investigation entries (§16.3)
    path(
        "investigations/<uuid:investigation_id>/entries/",
        views.investigation_entries,
        name="loop_investigation_entries",
    ),
    # Report generation (§5.2)
    path(
        "investigations/<uuid:investigation_id>/report/",
        views.generate_report,
        name="loop_generate_report",
    ),
    # FMIS (§8) — global risk landscape
    path("fmis/global/", views.fmis_global, name="loop_fmis_global"),
    path("fmis/", views.fmis_list_create, name="loop_fmis_list_create"),
    path("fmis/<uuid:fmis_id>/", views.fmis_detail, name="loop_fmis_detail"),
    path("fmis/<uuid:fmis_id>/rows/", views.fmis_add_row, name="loop_fmis_add_row"),
    path(
        "fmis/<uuid:fmis_id>/rows/<uuid:row_id>/posterior/",
        views.fmis_row_update_posterior,
        name="loop_fmis_row_posterior",
    ),
    # Process Confirmations (§7.1)
    path("pcs/", views.pc_list_create, name="loop_pc_list_create"),
    path("pcs/<uuid:pc_id>/", views.pc_detail, name="loop_pc_detail"),
    # Forced Failure Tests (§7.2)
    path("ffts/", views.fft_list_create, name="loop_fft_list_create"),
    path("ffts/<uuid:fft_id>/", views.fft_detail, name="loop_fft_detail"),
    # Training Reflections (§6.2)
    path("reflections/", views.reflection_list_create, name="loop_reflection_list_create"),
    # QMS Policy Management (§4)
    path("policies/registry/", views.policy_registry, name="loop_policy_registry"),
    path("policies/", views.policy_list_create, name="loop_policy_list_create"),
    path("policies/<uuid:policy_id>/", views.policy_detail, name="loop_policy_detail"),
    # Auditor Portal — token management (§11, authenticated)
    path("auditor-tokens/", views.auditor_token_list_create, name="loop_auditor_token_list"),
    path("auditor-tokens/<uuid:token_id>/", views.auditor_token_revoke, name="loop_auditor_token_revoke"),
    # Auditor Portal — data API (§11, token-authenticated)
    path("portal/<str:token>/data/", views.auditor_portal_data, name="loop_auditor_portal_data"),
    # Supplier Claims
    path("claims/", views.claim_list_create, name="loop_claim_list"),
    path("claims/<uuid:claim_id>/", views.claim_detail, name="loop_claim_detail"),
    path("claims/<uuid:claim_id>/respond/", views.claim_respond, name="loop_claim_respond"),
    # Supplier Portal — claim access (token-authenticated, no login)
    path("portal/claim/<str:token>/data/", views.claim_portal_data, name="loop_claim_portal_data"),
    path("portal/claim/<str:token>/respond/", views.claim_portal_respond, name="loop_claim_portal_respond"),
]

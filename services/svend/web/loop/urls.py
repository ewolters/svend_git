"""URL routes for Loop API — LOOP-001 §3, §5, §6, §7, §16."""

from django.urls import path

from . import views

urlpatterns = [
    # Dashboard data (§16.2)
    path("dashboard/", views.dashboard_data, name="loop_dashboard_data"),
    # Signals (§3.1)
    path("signals/", views.signal_list_create, name="loop_signal_list_create"),
    path("signals/<uuid:signal_id>/", views.signal_detail, name="loop_signal_detail"),
    # Commitments (§3.3)
    path("commitments/", views.commitment_list_create, name="loop_commitment_list_create"),
    path("commitments/<uuid:commitment_id>/", views.commitment_detail, name="loop_commitment_detail"),
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
    # Process Confirmations (§7.1)
    path("pcs/", views.pc_list_create, name="loop_pc_list_create"),
    path("pcs/<uuid:pc_id>/", views.pc_detail, name="loop_pc_detail"),
    # Forced Failure Tests (§7.2)
    path("ffts/", views.fft_list_create, name="loop_fft_list_create"),
    path("ffts/<uuid:fft_id>/", views.fft_detail, name="loop_fft_detail"),
    # Training Reflections (§6.2)
    path("reflections/", views.reflection_list_create, name="loop_reflection_list_create"),
]

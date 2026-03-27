"""URL routes for Loop API — LOOP-001 §3."""

from django.urls import path

from . import views

urlpatterns = [
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
]

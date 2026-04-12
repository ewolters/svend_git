from django.urls import path

from . import views

urlpatterns = [
    # CRUD
    path("", views.list_vsm, name="vsm_list"),
    path("create/", views.create_vsm, name="vsm_create"),
    path("<uuid:vsm_id>/", views.get_vsm, name="vsm_detail"),
    path("<uuid:vsm_id>/update/", views.update_vsm, name="vsm_update"),
    path("<uuid:vsm_id>/delete/", views.delete_vsm, name="vsm_delete"),
    # Structured additions
    path("<uuid:vsm_id>/process-step/", views.add_process_step, name="vsm_add_step"),
    path("<uuid:vsm_id>/inventory/", views.add_inventory, name="vsm_add_inventory"),
    path("<uuid:vsm_id>/kaizen/", views.add_kaizen_burst, name="vsm_add_kaizen"),
    # Future state & comparison
    path("<uuid:vsm_id>/future-state/", views.create_future_state, name="vsm_future_state"),
    path("<uuid:vsm_id>/compare/", views.compare_vsm, name="vsm_compare"),
    # Analysis
    path("<uuid:vsm_id>/waste-analysis/", views.waste_analysis, name="vsm_waste"),
    # Hoshin integration (hanging wires)
    path("<uuid:vsm_id>/generate-proposals/", views.generate_proposals, name="vsm_proposals"),
    path("<uuid:vsm_id>/approve-proposal/", views.approve_proposal, name="vsm_approve"),
]

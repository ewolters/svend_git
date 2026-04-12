"""Plant Simulator API URLs."""

from django.urls import path

from . import plantsim_views

urlpatterns = [
    path("", plantsim_views.list_simulations, name="plantsim_list"),
    path("create/", plantsim_views.create_simulation, name="plantsim_create"),
    path("<uuid:sim_id>/", plantsim_views.get_simulation, name="plantsim_get"),
    path(
        "<uuid:sim_id>/update/",
        plantsim_views.update_simulation,
        name="plantsim_update",
    ),
    path(
        "<uuid:sim_id>/delete/",
        plantsim_views.delete_simulation,
        name="plantsim_delete",
    ),
    path(
        "<uuid:sim_id>/results/",
        plantsim_views.save_results,
        name="plantsim_save_results",
    ),
    path(
        "<uuid:sim_id>/import-vsm/",
        plantsim_views.import_from_vsm,
        name="plantsim_import_vsm",
    ),
    path(
        "<uuid:sim_id>/export/",
        plantsim_views.export_to_project,
        name="plantsim_export",
    ),
]

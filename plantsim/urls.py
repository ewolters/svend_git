"""Plant Simulator API URLs."""

from django.urls import path

from . import views

urlpatterns = [
    path("", views.list_simulations, name="plantsim_list"),
    path("create/", views.create_simulation, name="plantsim_create"),
    path("<uuid:sim_id>/", views.get_simulation, name="plantsim_get"),
    path(
        "<uuid:sim_id>/update/",
        views.update_simulation,
        name="plantsim_update",
    ),
    path(
        "<uuid:sim_id>/delete/",
        views.delete_simulation,
        name="plantsim_delete",
    ),
    path(
        "<uuid:sim_id>/results/",
        views.save_results,
        name="plantsim_save_results",
    ),
    path(
        "<uuid:sim_id>/import-vsm/",
        views.import_from_vsm,
        name="plantsim_import_vsm",
    ),
    path(
        "<uuid:sim_id>/export/",
        views.export_to_project,
        name="plantsim_export",
    ),
]

"""Value Stream Map API URLs."""

from django.urls import path
from . import vsm_views

urlpatterns = [
    path("", vsm_views.list_vsm, name="vsm_list"),
    path("create/", vsm_views.create_vsm, name="vsm_create"),
    path("<uuid:vsm_id>/", vsm_views.get_vsm, name="vsm_get"),
    path("<uuid:vsm_id>/update/", vsm_views.update_vsm, name="vsm_update"),
    path("<uuid:vsm_id>/delete/", vsm_views.delete_vsm, name="vsm_delete"),
    path("<uuid:vsm_id>/process-step/", vsm_views.add_process_step, name="vsm_add_process"),
    path("<uuid:vsm_id>/inventory/", vsm_views.add_inventory, name="vsm_add_inventory"),
    path("<uuid:vsm_id>/kaizen/", vsm_views.add_kaizen_burst, name="vsm_add_kaizen"),
    path("<uuid:vsm_id>/future-state/", vsm_views.create_future_state, name="vsm_future_state"),
    path("<uuid:vsm_id>/compare/", vsm_views.compare_vsm, name="vsm_compare"),
]

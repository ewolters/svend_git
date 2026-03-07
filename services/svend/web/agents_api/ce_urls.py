"""URL routes for Cause & Effect (C&E) Matrix API."""

from django.urls import path

from . import ce_views

urlpatterns = [
    path("sessions/", ce_views.list_matrices, name="ce_list"),
    path("sessions/create/", ce_views.create_matrix, name="ce_create"),
    path("sessions/<uuid:matrix_id>/", ce_views.get_matrix, name="ce_get"),
    path("sessions/<uuid:matrix_id>/update/", ce_views.update_matrix, name="ce_update"),
    path("sessions/<uuid:matrix_id>/delete/", ce_views.delete_matrix, name="ce_delete"),
]

"""URL routes for Ishikawa (Fishbone) diagram API."""

from django.urls import path

from . import ishikawa_views

urlpatterns = [
    path("sessions/", ishikawa_views.list_diagrams, name="ishikawa_list"),
    path("sessions/create/", ishikawa_views.create_diagram, name="ishikawa_create"),
    path("sessions/<uuid:diagram_id>/", ishikawa_views.get_diagram, name="ishikawa_get"),
    path("sessions/<uuid:diagram_id>/update/", ishikawa_views.update_diagram, name="ishikawa_update"),
    path("sessions/<uuid:diagram_id>/delete/", ishikawa_views.delete_diagram, name="ishikawa_delete"),
]

"""URL routes for RCA sessions."""

from django.urls import path

from . import views

app_name = "rca"

urlpatterns = [
    path("sessions/", views.list_sessions, name="list_sessions"),
    path("sessions/create/", views.create_session, name="create_session"),
]

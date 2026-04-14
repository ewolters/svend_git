"""QMS Core URL routes."""

from django.urls import path

from . import views

urlpatterns = [
    path("sites/", views.qms_sites, name="qms_sites"),
]

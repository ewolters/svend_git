"""QMS cross-module intelligence API URLs."""

from django.urls import path
from . import qms_views

urlpatterns = [
    path("dashboard/", qms_views.qms_dashboard, name="qms_dashboard"),
]

"""Notification token URL routes — NTF-001 §5.2.4."""

from django.urls import path
from . import token_views

urlpatterns = [
    path("", token_views.notification_token_view, name="notification_token"),
]

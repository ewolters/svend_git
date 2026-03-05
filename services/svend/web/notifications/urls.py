"""Notification URL routes — NTF-001 §6.1."""

from django.urls import path
from . import views

urlpatterns = [
    path("", views.notification_list, name="notification_list"),
    path("stream/", views.notification_stream, name="notification_stream"),
    path("unread-count/", views.notification_unread_count, name="notification_unread_count"),
    path("read-all/", views.notification_mark_all_read, name="notification_mark_all_read"),
    path("preferences/", views.notification_preferences, name="notification_preferences"),
    path("unsubscribe/", views.notification_type_unsubscribe, name="notification_type_unsubscribe"),
    path("<uuid:notification_id>/read/", views.notification_mark_read, name="notification_mark_read"),
]

"""Chat URL configuration."""

from django.urls import path

from . import views

app_name = "chat"

urlpatterns = [
    path("shared/<uuid:share_id>/", views.shared_conversation, name="shared"),
]

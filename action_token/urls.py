"""ActionToken URL — unauthenticated access for non-user participation.

Standard: QMS-002 §3.3
"""

from django.urls import path

from . import views

urlpatterns = [
    path("", views.action_token_view, name="action_token"),
]

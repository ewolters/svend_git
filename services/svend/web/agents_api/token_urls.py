"""ActionToken URL — unauthenticated access for non-user participation.

Standard: QMS-002 §3.3
"""

from django.urls import path

from . import token_views

urlpatterns = [
    path("", token_views.action_token_view, name="action_token"),
]

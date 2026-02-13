"""URL routes for RCA critique engine."""

from django.urls import path
from . import rca_views

urlpatterns = [
    # AI critique endpoints
    path("critique/", rca_views.critique, name="rca_critique"),
    path("critique-countermeasure/", rca_views.critique_countermeasure, name="rca_critique_countermeasure"),
    path("evaluate/", rca_views.evaluate_chain, name="rca_evaluate"),

    # Session CRUD
    path("sessions/", rca_views.list_sessions, name="rca_list"),
    path("sessions/create/", rca_views.create_session, name="rca_create"),
    path("sessions/<uuid:session_id>/", rca_views.get_session, name="rca_get"),
    path("sessions/<uuid:session_id>/update/", rca_views.update_session, name="rca_update"),
    path("sessions/<uuid:session_id>/delete/", rca_views.delete_session, name="rca_delete"),
    path("sessions/<uuid:session_id>/link-a3/", rca_views.link_to_a3, name="rca_link_a3"),

    # Similarity search
    path("similar/", rca_views.find_similar, name="rca_similar"),
    path("reindex/", rca_views.reindex_embeddings, name="rca_reindex"),

    # Action items
    path("sessions/<uuid:session_id>/actions/", rca_views.list_rca_actions, name="rca_actions"),
    path("sessions/<uuid:session_id>/actions/create/", rca_views.create_rca_action, name="rca_create_action"),
]

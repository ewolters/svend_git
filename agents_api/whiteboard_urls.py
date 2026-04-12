"""URL patterns for Whiteboard API."""

from django.urls import path

from . import whiteboard_views

urlpatterns = [
    # Board CRUD
    path("boards/", whiteboard_views.list_boards, name="whiteboard_list"),
    path("boards/create/", whiteboard_views.create_board, name="whiteboard_create"),
    path("boards/<str:room_code>/", whiteboard_views.get_board, name="whiteboard_get"),
    path(
        "boards/<str:room_code>/update/",
        whiteboard_views.update_board,
        name="whiteboard_update",
    ),
    path(
        "boards/<str:room_code>/delete/",
        whiteboard_views.delete_board,
        name="whiteboard_delete",
    ),
    # Presence
    path(
        "boards/<str:room_code>/cursor/",
        whiteboard_views.update_cursor,
        name="whiteboard_cursor",
    ),
    # Voting
    path(
        "boards/<str:room_code>/voting/",
        whiteboard_views.toggle_voting,
        name="whiteboard_voting",
    ),
    path(
        "boards/<str:room_code>/vote/",
        whiteboard_views.add_vote,
        name="whiteboard_vote_add",
    ),
    path(
        "boards/<str:room_code>/vote/<str:element_id>/",
        whiteboard_views.remove_vote,
        name="whiteboard_vote_remove",
    ),
    # Export to Knowledge
    path(
        "boards/<str:room_code>/export-hypotheses/",
        whiteboard_views.export_hypotheses,
        name="whiteboard_export_hypotheses",
    ),
    # SVG/PNG Export for A3/ISO Doc embedding
    path(
        "boards/<str:room_code>/svg/",
        whiteboard_views.export_svg,
        name="whiteboard_svg",
    ),
    path(
        "boards/<str:room_code>/png/",
        whiteboard_views.export_png,
        name="whiteboard_png",
    ),
    # Guest invite management (owner endpoints)
    path(
        "boards/<str:room_code>/guests/",
        whiteboard_views.list_guest_invites,
        name="whiteboard_guest_list",
    ),
    path(
        "boards/<str:room_code>/guests/create/",
        whiteboard_views.create_guest_invite,
        name="whiteboard_guest_create",
    ),
    path(
        "boards/<str:room_code>/guests/<uuid:invite_id>/revoke/",
        whiteboard_views.revoke_guest_invite,
        name="whiteboard_guest_revoke",
    ),
    # Guest self-service
    path(
        "boards/<str:room_code>/guest-name/",
        whiteboard_views.set_guest_name,
        name="whiteboard_guest_name",
    ),
]

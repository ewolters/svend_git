from django.urls import path

from . import views

urlpatterns = [
    # Board CRUD
    path("boards/", views.list_boards, name="whiteboard_list"),
    path("boards/create/", views.create_board, name="whiteboard_create"),
    path("boards/<str:room_code>/", views.get_board, name="whiteboard_get"),
    path("boards/<str:room_code>/update/", views.update_board, name="whiteboard_update"),
    path("boards/<str:room_code>/delete/", views.delete_board, name="whiteboard_delete"),
    # Presence
    path("boards/<str:room_code>/cursor/", views.update_cursor, name="whiteboard_cursor"),
    # Voting
    path("boards/<str:room_code>/voting/", views.toggle_voting, name="whiteboard_voting"),
    path("boards/<str:room_code>/vote/", views.add_vote, name="whiteboard_vote_add"),
    path("boards/<str:room_code>/vote/<str:element_id>/", views.remove_vote, name="whiteboard_vote_remove"),
    # Export
    path("boards/<str:room_code>/svg/", views.export_svg, name="whiteboard_svg"),
    path("boards/<str:room_code>/png/", views.export_png, name="whiteboard_png"),
    # Guest management
    path("boards/<str:room_code>/guests/", views.list_guest_invites, name="whiteboard_guest_list"),
    path("boards/<str:room_code>/guests/create/", views.create_guest_invite, name="whiteboard_guest_create"),
    path(
        "boards/<str:room_code>/guests/<uuid:invite_id>/revoke/",
        views.revoke_guest_invite,
        name="whiteboard_guest_revoke",
    ),
    path("boards/<str:room_code>/guest-name/", views.set_guest_name, name="whiteboard_guest_name"),
]

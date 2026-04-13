"""Whiteboard API tests — CRUD, voting, guest access, SVG export, hypothesis export."""

import json
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import RequestFactory, TestCase, override_settings
from django.utils import timezone

from core.models import Hypothesis, Project
from whiteboard.models import Board, BoardGuestInvite, BoardParticipant, BoardVote

User = get_user_model()


@override_settings(SECURE_SSL_REDIRECT=False)
class WhiteboardTestBase(TestCase):
    """Shared setup for whiteboard tests."""

    def setUp(self):
        self.factory = RequestFactory()
        self.owner = User.objects.create_user(
            username="owner",
            email="owner@test.com",
            password="pass123",
        )
        self.owner.tier = "pro"
        self.owner.save()

        self.other = User.objects.create_user(
            username="other",
            email="other@test.com",
            password="pass123",
        )
        self.other.tier = "pro"
        self.other.save()

        self.free_user = User.objects.create_user(
            username="freeuser",
            email="free@test.com",
            password="pass123",
        )
        # free_user.tier defaults to "free"

    def _auth_post(self, path, data=None, user=None):
        user = user or self.owner
        self.client.force_login(user)
        return self.client.post(
            path,
            json.dumps(data or {}),
            content_type="application/json",
        )

    def _auth_get(self, path, user=None):
        user = user or self.owner
        self.client.force_login(user)
        return self.client.get(path)

    def _auth_put(self, path, data=None, user=None):
        user = user or self.owner
        self.client.force_login(user)
        return self.client.put(
            path,
            json.dumps(data or {}),
            content_type="application/json",
        )

    def _auth_delete(self, path, user=None):
        user = user or self.owner
        self.client.force_login(user)
        return self.client.delete(path)

    def _guest_get(self, path, token):
        return self.client.get(path, HTTP_X_GUEST_TOKEN=token)

    def _guest_post(self, path, data, token):
        return self.client.post(
            path,
            json.dumps(data),
            content_type="application/json",
            HTTP_X_GUEST_TOKEN=token,
        )

    def _guest_put(self, path, data, token):
        return self.client.put(
            path,
            json.dumps(data),
            content_type="application/json",
            HTTP_X_GUEST_TOKEN=token,
        )

    def _guest_delete(self, path, token):
        return self.client.delete(path, HTTP_X_GUEST_TOKEN=token)

    def _create_board(self, name="Test Board", user=None):
        resp = self._auth_post("/api/whiteboard/boards/create/", {"name": name}, user=user)
        self.assertEqual(resp.status_code, 200)
        return resp.json()

    def _create_invite(self, room_code, permission="edit_vote"):
        resp = self._auth_post(
            f"/api/whiteboard/boards/{room_code}/guests/create/",
            {"permission": permission},
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()


# ==========================================================================
# Board CRUD
# ==========================================================================


class BoardCreateTest(WhiteboardTestBase):
    def test_create_board(self):
        data = self._create_board("My Board")
        self.assertEqual(data["name"], "My Board")
        self.assertTrue(len(data["room_code"]) >= 6)
        self.assertIn("/app/whiteboard/", data["url"])

    def test_create_board_default_name(self):
        resp = self._auth_post("/api/whiteboard/boards/create/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["name"], "Untitled Board")

    def test_create_board_with_project(self):
        project = Project.objects.create(user=self.owner, title="Study A")
        resp = self._auth_post(
            "/api/whiteboard/boards/create/",
            {"name": "Linked", "project_id": str(project.id)},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["project_id"], str(project.id))

    def test_create_board_invalid_project(self):
        resp = self._auth_post(
            "/api/whiteboard/boards/create/",
            {"project_id": "00000000-0000-0000-0000-000000000000"},
        )
        self.assertEqual(resp.status_code, 404)

    def test_create_board_other_users_project(self):
        project = Project.objects.create(user=self.other, title="Not mine")
        resp = self._auth_post(
            "/api/whiteboard/boards/create/",
            {"project_id": str(project.id)},
        )
        self.assertEqual(resp.status_code, 404)

    def test_creator_becomes_participant(self):
        data = self._create_board()
        board = Board.objects.get(room_code=data["room_code"])
        self.assertTrue(BoardParticipant.objects.filter(board=board, user=self.owner).exists())

    def test_free_user_blocked(self):
        resp = self._auth_post(
            "/api/whiteboard/boards/create/",
            {"name": "X"},
            user=self.free_user,
        )
        self.assertEqual(resp.status_code, 403)

    def test_unauthenticated_blocked(self):
        resp = self.client.post(
            "/api/whiteboard/boards/create/",
            json.dumps({"name": "X"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 401)


class BoardGetTest(WhiteboardTestBase):
    def test_get_board(self):
        board_data = self._create_board()
        resp = self._auth_get(f"/api/whiteboard/boards/{board_data['room_code']}/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["room_code"], board_data["room_code"])
        self.assertEqual(data["elements"], [])
        self.assertEqual(data["connections"], [])
        self.assertTrue(data["is_owner"])
        self.assertFalse(data["is_guest"])

    def test_get_board_case_insensitive(self):
        board_data = self._create_board()
        code = board_data["room_code"].lower()
        resp = self._auth_get(f"/api/whiteboard/boards/{code}/")
        self.assertEqual(resp.status_code, 200)

    def test_get_board_nonexistent(self):
        resp = self._auth_get("/api/whiteboard/boards/ZZZZZZ/")
        self.assertEqual(resp.status_code, 404)

    def test_other_user_joins_as_participant(self):
        board_data = self._create_board()
        resp = self._auth_get(
            f"/api/whiteboard/boards/{board_data['room_code']}/",
            user=self.other,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["is_owner"])
        board = Board.objects.get(room_code=board_data["room_code"])
        self.assertTrue(BoardParticipant.objects.filter(board=board, user=self.other).exists())

    def test_get_board_includes_vote_counts(self):
        board_data = self._create_board()
        board = Board.objects.get(room_code=board_data["room_code"])
        board.is_voting_active = True
        board.save()
        BoardVote.objects.create(board=board, user=self.owner, element_id="el-1")
        resp = self._auth_get(f"/api/whiteboard/boards/{board_data['room_code']}/")
        data = resp.json()
        self.assertEqual(data["vote_counts"]["el-1"], 1)
        self.assertIn("el-1", data["user_votes"])

    def test_get_board_project_context(self):
        project = Project.objects.create(user=self.owner, title="Study X")
        board_data = self._auth_post(
            "/api/whiteboard/boards/create/",
            {"name": "Linked", "project_id": str(project.id)},
        ).json()
        resp = self._auth_get(f"/api/whiteboard/boards/{board_data['room_code']}/")
        data = resp.json()
        self.assertEqual(data["project"]["title"], "Study X")


class BoardUpdateTest(WhiteboardTestBase):
    def test_update_elements(self):
        board_data = self._create_board()
        elements = [{"id": "el-1", "type": "postit", "x": 100, "y": 200, "text": "Hello"}]
        resp = self._auth_put(
            f"/api/whiteboard/boards/{board_data['room_code']}/update/",
            {"elements": elements},
        )
        self.assertEqual(resp.status_code, 200)
        board = Board.objects.get(room_code=board_data["room_code"])
        self.assertEqual(len(board.elements), 1)
        self.assertEqual(board.elements[0]["text"], "Hello")

    def test_update_connections(self):
        board_data = self._create_board()
        connections = [
            {
                "id": "c-1",
                "from": {"elementId": "el-1"},
                "to": {"elementId": "el-2"},
                "type": "causal",
            },
        ]
        resp = self._auth_put(
            f"/api/whiteboard/boards/{board_data['room_code']}/update/",
            {"connections": connections},
        )
        self.assertEqual(resp.status_code, 200)
        board = Board.objects.get(room_code=board_data["room_code"])
        self.assertEqual(len(board.connections), 1)

    def test_update_name(self):
        board_data = self._create_board()
        resp = self._auth_put(
            f"/api/whiteboard/boards/{board_data['room_code']}/update/",
            {"name": "Renamed"},
        )
        self.assertEqual(resp.status_code, 200)
        board = Board.objects.get(room_code=board_data["room_code"])
        self.assertEqual(board.name, "Renamed")

    def test_update_zoom_pan(self):
        board_data = self._create_board()
        resp = self._auth_put(
            f"/api/whiteboard/boards/{board_data['room_code']}/update/",
            {"zoom": 1.5, "pan_x": 100.0, "pan_y": -50.0},
        )
        self.assertEqual(resp.status_code, 200)
        board = Board.objects.get(room_code=board_data["room_code"])
        self.assertAlmostEqual(board.zoom, 1.5)
        self.assertAlmostEqual(board.pan_x, 100.0)
        self.assertAlmostEqual(board.pan_y, -50.0)

    def test_version_increments(self):
        board_data = self._create_board()
        board = Board.objects.get(room_code=board_data["room_code"])
        v0 = board.version
        self._auth_put(
            f"/api/whiteboard/boards/{board_data['room_code']}/update/",
            {"elements": []},
        )
        board.refresh_from_db()
        self.assertGreater(board.version, v0)

    def test_version_conflict(self):
        board_data = self._create_board()
        board = Board.objects.get(room_code=board_data["room_code"])
        # Save once to bump version
        board.elements = [{"id": "el-1", "type": "postit"}]
        board.save()

        # Send stale version
        resp = self._auth_put(
            f"/api/whiteboard/boards/{board_data['room_code']}/update/",
            {"version": 0, "elements": [{"id": "el-2"}]},
        )
        self.assertEqual(resp.status_code, 409)
        data = resp.json()
        self.assertTrue(data["conflict"])
        self.assertIn("el-1", str(data["elements"]))

    def test_non_participant_cannot_update(self):
        board_data = self._create_board()
        resp = self._auth_put(
            f"/api/whiteboard/boards/{board_data['room_code']}/update/",
            {"name": "Hacked"},
            user=self.other,
        )
        self.assertEqual(resp.status_code, 403)

    def test_participant_can_update(self):
        board_data = self._create_board()
        board = Board.objects.get(room_code=board_data["room_code"])
        BoardParticipant.objects.create(board=board, user=self.other)
        resp = self._auth_put(
            f"/api/whiteboard/boards/{board_data['room_code']}/update/",
            {"elements": [{"id": "el-new"}]},
            user=self.other,
        )
        self.assertEqual(resp.status_code, 200)

    def test_link_project(self):
        board_data = self._create_board()
        project = Project.objects.create(user=self.owner, title="Link me")
        resp = self._auth_put(
            f"/api/whiteboard/boards/{board_data['room_code']}/update/",
            {"project_id": str(project.id)},
        )
        self.assertEqual(resp.status_code, 200)
        board = Board.objects.get(room_code=board_data["room_code"])
        self.assertEqual(board.project_id, project.id)

    def test_unlink_project(self):
        project = Project.objects.create(user=self.owner, title="Unlink me")
        board_data = self._auth_post(
            "/api/whiteboard/boards/create/",
            {"name": "B", "project_id": str(project.id)},
        ).json()
        resp = self._auth_put(
            f"/api/whiteboard/boards/{board_data['room_code']}/update/",
            {"project_id": None},
        )
        self.assertEqual(resp.status_code, 200)
        board = Board.objects.get(room_code=board_data["room_code"])
        self.assertIsNone(board.project)


class BoardDeleteTest(WhiteboardTestBase):
    def test_delete_board(self):
        board_data = self._create_board()
        resp = self._auth_delete(
            f"/api/whiteboard/boards/{board_data['room_code']}/delete/",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(Board.objects.filter(room_code=board_data["room_code"]).exists())

    def test_only_owner_can_delete(self):
        board_data = self._create_board()
        resp = self._auth_delete(
            f"/api/whiteboard/boards/{board_data['room_code']}/delete/",
            user=self.other,
        )
        self.assertEqual(resp.status_code, 403)

    def test_delete_cascades_participants(self):
        board_data = self._create_board()
        board = Board.objects.get(room_code=board_data["room_code"])
        BoardParticipant.objects.create(board=board, user=self.other)
        self._auth_delete(f"/api/whiteboard/boards/{board_data['room_code']}/delete/")
        self.assertEqual(BoardParticipant.objects.filter(board_id=board.id).count(), 0)


class BoardListTest(WhiteboardTestBase):
    def test_list_empty(self):
        resp = self._auth_get("/api/whiteboard/boards/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["owned"], [])
        self.assertEqual(data["participated"], [])

    def test_list_owned(self):
        self._create_board("Board A")
        self._create_board("Board B")
        resp = self._auth_get("/api/whiteboard/boards/")
        data = resp.json()
        self.assertEqual(len(data["owned"]), 2)
        self.assertTrue(all(b["is_owner"] for b in data["owned"]))

    def test_list_participated(self):
        board_data = self._create_board("Owner Board")
        board = Board.objects.get(room_code=board_data["room_code"])
        BoardParticipant.objects.create(board=board, user=self.other)
        resp = self._auth_get("/api/whiteboard/boards/", user=self.other)
        data = resp.json()
        self.assertEqual(len(data["participated"]), 1)
        self.assertFalse(data["participated"][0]["is_owner"])

    def test_filter_by_project(self):
        project = Project.objects.create(user=self.owner, title="P1")
        self._auth_post(
            "/api/whiteboard/boards/create/",
            {"name": "Linked", "project_id": str(project.id)},
        )
        self._create_board("Unlinked")
        resp = self._auth_get(f"/api/whiteboard/boards/?project_id={project.id}")
        data = resp.json()
        self.assertEqual(len(data["owned"]), 1)
        self.assertEqual(data["owned"][0]["name"], "Linked")


# ==========================================================================
# Cursor / Presence
# ==========================================================================


class CursorTest(WhiteboardTestBase):
    def test_update_cursor(self):
        board_data = self._create_board()
        resp = self._auth_post(
            f"/api/whiteboard/boards/{board_data['room_code']}/cursor/",
            {"x": 150.5, "y": 200.3},
        )
        self.assertEqual(resp.status_code, 200)
        p = BoardParticipant.objects.get(
            board__room_code=board_data["room_code"],
            user=self.owner,
        )
        self.assertAlmostEqual(p.cursor_x, 150.5)
        self.assertAlmostEqual(p.cursor_y, 200.3)


# ==========================================================================
# Voting
# ==========================================================================


class VotingTest(WhiteboardTestBase):
    def setUp(self):
        super().setUp()
        bd = self._create_board("Vote Board")
        self.room = bd["room_code"]
        self.board = Board.objects.get(room_code=self.room)

    def test_toggle_voting_on(self):
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/voting/",
            {"active": True},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["voting_active"])
        self.board.refresh_from_db()
        self.assertTrue(self.board.is_voting_active)

    def test_toggle_voting_off(self):
        self.board.is_voting_active = True
        self.board.save()
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/voting/",
            {"active": False},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["voting_active"])

    def test_only_owner_can_toggle(self):
        BoardParticipant.objects.create(board=self.board, user=self.other)
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/voting/",
            {"active": True},
            user=self.other,
        )
        self.assertEqual(resp.status_code, 403)

    def test_set_votes_per_user(self):
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/voting/",
            {"active": True, "votes_per_user": 5},
        )
        self.assertEqual(resp.json()["votes_per_user"], 5)

    def test_clear_votes(self):
        self.board.is_voting_active = True
        self.board.save()
        BoardVote.objects.create(board=self.board, user=self.owner, element_id="el-1")
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/voting/",
            {"clear_votes": True},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(BoardVote.objects.filter(board=self.board).count(), 0)

    def test_add_vote(self):
        self.board.is_voting_active = True
        self.board.save()
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/vote/",
            {"element_id": "el-1"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["vote_count"], 1)
        self.assertEqual(data["votes_remaining"], 2)  # 3 default - 1

    def test_vote_inactive_rejected(self):
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/vote/",
            {"element_id": "el-1"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("not active", resp.json()["error"])

    def test_vote_limit(self):
        self.board.is_voting_active = True
        self.board.votes_per_user = 2
        self.board.save()
        self._auth_post(f"/api/whiteboard/boards/{self.room}/vote/", {"element_id": "el-1"})
        self._auth_post(f"/api/whiteboard/boards/{self.room}/vote/", {"element_id": "el-2"})
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/vote/",
            {"element_id": "el-3"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("limit", resp.json()["error"].lower())

    def test_duplicate_vote_rejected(self):
        self.board.is_voting_active = True
        self.board.save()
        self._auth_post(f"/api/whiteboard/boards/{self.room}/vote/", {"element_id": "el-1"})
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/vote/",
            {"element_id": "el-1"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Already voted", resp.json()["error"])

    def test_remove_vote(self):
        self.board.is_voting_active = True
        self.board.save()
        self._auth_post(f"/api/whiteboard/boards/{self.room}/vote/", {"element_id": "el-1"})
        resp = self._auth_delete(
            f"/api/whiteboard/boards/{self.room}/vote/el-1/",
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["vote_count"], 0)
        self.assertEqual(resp.json()["votes_remaining"], 3)

    def test_remove_nonexistent_vote(self):
        resp = self._auth_delete(
            f"/api/whiteboard/boards/{self.room}/vote/el-999/",
        )
        self.assertEqual(resp.status_code, 404)

    def test_missing_element_id(self):
        self.board.is_voting_active = True
        self.board.save()
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/vote/",
            {},
        )
        self.assertEqual(resp.status_code, 400)


# ==========================================================================
# Guest Access
# ==========================================================================


class GuestInviteTest(WhiteboardTestBase):
    def setUp(self):
        super().setUp()
        bd = self._create_board("Guest Board")
        self.room = bd["room_code"]
        self.board = Board.objects.get(room_code=self.room)

    def test_create_invite(self):
        data = self._create_invite(self.room, "edit")
        self.assertEqual(data["permission"], "edit")
        self.assertTrue(len(data["token"]) >= 32)
        self.assertIn("/app/whiteboard/guest/", data["url"])

    def test_create_invite_default_permission(self):
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/guests/create/",
            {},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["permission"], "view")

    def test_create_invite_invalid_permission(self):
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/guests/create/",
            {"permission": "admin"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_only_owner_can_invite(self):
        BoardParticipant.objects.create(board=self.board, user=self.other)
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/guests/create/",
            {"permission": "view"},
            user=self.other,
        )
        self.assertEqual(resp.status_code, 403)

    def test_list_invites(self):
        self._create_invite(self.room, "view")
        self._create_invite(self.room, "edit")
        resp = self._auth_get(f"/api/whiteboard/boards/{self.room}/guests/")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.json()["invites"]), 2)

    def test_revoke_invite(self):
        invite_data = self._create_invite(self.room, "edit")
        resp = self._auth_delete(
            f"/api/whiteboard/boards/{self.room}/guests/{invite_data['id']}/revoke/",
        )
        self.assertEqual(resp.status_code, 200)
        invite = BoardGuestInvite.objects.get(id=invite_data["id"])
        self.assertFalse(invite.is_active)

    def test_revoked_token_rejected(self):
        invite_data = self._create_invite(self.room, "edit")
        invite = BoardGuestInvite.objects.get(id=invite_data["id"])
        invite.is_active = False
        invite.save()
        resp = self._guest_get(
            f"/api/whiteboard/boards/{self.room}/",
            invite_data["token"],
        )
        self.assertEqual(resp.status_code, 403)

    def test_expired_token_rejected(self):
        invite_data = self._create_invite(self.room, "edit")
        invite = BoardGuestInvite.objects.get(id=invite_data["id"])
        invite.expires_at = timezone.now() - timedelta(days=1)
        invite.save()
        resp = self._guest_get(
            f"/api/whiteboard/boards/{self.room}/",
            invite_data["token"],
        )
        self.assertEqual(resp.status_code, 403)

    def test_invalid_token_rejected(self):
        resp = self._guest_get(
            f"/api/whiteboard/boards/{self.room}/",
            "bogustoken12345",
        )
        self.assertEqual(resp.status_code, 401)


class GuestBoardAccessTest(WhiteboardTestBase):
    def setUp(self):
        super().setUp()
        bd = self._create_board("Collab Board")
        self.room = bd["room_code"]
        self.board = Board.objects.get(room_code=self.room)

    def test_guest_can_read_board(self):
        invite = self._create_invite(self.room, "view")
        resp = self._guest_get(
            f"/api/whiteboard/boards/{self.room}/",
            invite["token"],
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["is_guest"])
        self.assertEqual(data["guest_permission"], "view")
        self.assertIsNone(data["project"])

    def test_view_guest_cannot_edit(self):
        invite = self._create_invite(self.room, "view")
        resp = self._guest_put(
            f"/api/whiteboard/boards/{self.room}/update/",
            {"elements": [{"id": "el-1"}]},
            invite["token"],
        )
        self.assertEqual(resp.status_code, 403)

    def test_edit_guest_can_edit(self):
        invite = self._create_invite(self.room, "edit")
        resp = self._guest_put(
            f"/api/whiteboard/boards/{self.room}/update/",
            {"elements": [{"id": "el-1", "type": "postit"}]},
            invite["token"],
        )
        self.assertEqual(resp.status_code, 200)
        self.board.refresh_from_db()
        self.assertEqual(len(self.board.elements), 1)

    def test_guest_cannot_change_name(self):
        """Guests can update elements/connections but not board name."""
        invite = self._create_invite(self.room, "edit")
        self._guest_put(
            f"/api/whiteboard/boards/{self.room}/update/",
            {"name": "Hacked Name"},
            invite["token"],
        )
        self.board.refresh_from_db()
        self.assertEqual(self.board.name, "Collab Board")

    def test_guest_token_wrong_board(self):
        invite = self._create_invite(self.room, "edit")
        other_board = self._create_board("Other Board")
        resp = self._guest_get(
            f"/api/whiteboard/boards/{other_board['room_code']}/",
            invite["token"],
        )
        self.assertEqual(resp.status_code, 403)

    def test_guest_cursor_update(self):
        invite = self._create_invite(self.room, "edit")
        resp = self._guest_post(
            f"/api/whiteboard/boards/{self.room}/cursor/",
            {"x": 100, "y": 200},
            invite["token"],
        )
        self.assertEqual(resp.status_code, 200)
        inv = BoardGuestInvite.objects.get(id=invite["id"])
        self.assertAlmostEqual(inv.cursor_x, 100)

    def test_set_guest_name(self):
        invite = self._create_invite(self.room, "edit")
        resp = self._guest_post(
            f"/api/whiteboard/boards/{self.room}/guest-name/",
            {"display_name": "Alice"},
            invite["token"],
        )
        self.assertEqual(resp.status_code, 200)
        inv = BoardGuestInvite.objects.get(id=invite["id"])
        self.assertEqual(inv.display_name, "Alice")

    def test_set_guest_name_empty_rejected(self):
        invite = self._create_invite(self.room, "edit")
        resp = self._guest_post(
            f"/api/whiteboard/boards/{self.room}/guest-name/",
            {"display_name": ""},
            invite["token"],
        )
        self.assertEqual(resp.status_code, 400)

    def test_set_guest_name_truncated(self):
        invite = self._create_invite(self.room, "edit")
        long_name = "A" * 200
        resp = self._guest_post(
            f"/api/whiteboard/boards/{self.room}/guest-name/",
            {"display_name": long_name},
            invite["token"],
        )
        self.assertEqual(resp.status_code, 200)
        inv = BoardGuestInvite.objects.get(id=invite["id"])
        self.assertEqual(len(inv.display_name), 100)


class GuestVotingTest(WhiteboardTestBase):
    def setUp(self):
        super().setUp()
        bd = self._create_board("Vote Guest Board")
        self.room = bd["room_code"]
        self.board = Board.objects.get(room_code=self.room)
        self.board.is_voting_active = True
        self.board.save()

    def test_edit_vote_guest_can_vote(self):
        invite = self._create_invite(self.room, "edit_vote")
        resp = self._guest_post(
            f"/api/whiteboard/boards/{self.room}/vote/",
            {"element_id": "el-1"},
            invite["token"],
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["vote_count"], 1)

    def test_edit_guest_cannot_vote(self):
        invite = self._create_invite(self.room, "edit")
        resp = self._guest_post(
            f"/api/whiteboard/boards/{self.room}/vote/",
            {"element_id": "el-1"},
            invite["token"],
        )
        self.assertEqual(resp.status_code, 403)

    def test_view_guest_cannot_vote(self):
        invite = self._create_invite(self.room, "view")
        resp = self._guest_post(
            f"/api/whiteboard/boards/{self.room}/vote/",
            {"element_id": "el-1"},
            invite["token"],
        )
        self.assertEqual(resp.status_code, 403)

    def test_guest_vote_limit(self):
        invite = self._create_invite(self.room, "edit_vote")
        self.board.votes_per_user = 1
        self.board.save()
        self._guest_post(
            f"/api/whiteboard/boards/{self.room}/vote/",
            {"element_id": "el-1"},
            invite["token"],
        )
        resp = self._guest_post(
            f"/api/whiteboard/boards/{self.room}/vote/",
            {"element_id": "el-2"},
            invite["token"],
        )
        self.assertEqual(resp.status_code, 400)

    def test_guest_remove_vote(self):
        invite = self._create_invite(self.room, "edit_vote")
        self._guest_post(
            f"/api/whiteboard/boards/{self.room}/vote/",
            {"element_id": "el-1"},
            invite["token"],
        )
        resp = self._guest_delete(
            f"/api/whiteboard/boards/{self.room}/vote/el-1/",
            invite["token"],
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["vote_count"], 0)


# ==========================================================================
# SVG Export
# ==========================================================================


class SVGExportTest(WhiteboardTestBase):
    def test_empty_board_svg(self):
        board_data = self._create_board()
        resp = self._auth_get(
            f"/api/whiteboard/boards/{board_data['room_code']}/svg/",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("Empty whiteboard", data["svg"])
        self.assertEqual(data["element_count"], 0)

    def test_svg_with_elements(self):
        board_data = self._create_board()
        board = Board.objects.get(room_code=board_data["room_code"])
        board.elements = [
            {
                "id": "el-1",
                "type": "postit",
                "x": 50,
                "y": 50,
                "width": 120,
                "height": 80,
                "text": "Test",
                "color": "yellow",
            },
            {
                "id": "el-2",
                "type": "rectangle",
                "x": 200,
                "y": 50,
                "width": 120,
                "height": 60,
                "text": "Rect",
            },
        ]
        board.connections = [
            {
                "id": "c-1",
                "from": {"elementId": "el-1"},
                "to": {"elementId": "el-2"},
                "type": "arrow",
            },
        ]
        board.save()

        resp = self._auth_get(
            f"/api/whiteboard/boards/{board_data['room_code']}/svg/",
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["element_count"], 2)
        self.assertEqual(data["connection_count"], 1)
        self.assertIn("<svg", data["svg"])
        self.assertIn("Test", data["svg"])

    def test_svg_all_element_types(self):
        board_data = self._create_board()
        board = Board.objects.get(room_code=board_data["room_code"])
        board.elements = [
            {
                "id": "e1",
                "type": "postit",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 60,
                "text": "P",
                "color": "green",
            },
            {
                "id": "e2",
                "type": "rectangle",
                "x": 150,
                "y": 0,
                "width": 100,
                "height": 60,
                "text": "R",
            },
            {
                "id": "e3",
                "type": "oval",
                "x": 300,
                "y": 0,
                "width": 100,
                "height": 50,
                "text": "O",
            },
            {
                "id": "e4",
                "type": "diamond",
                "x": 0,
                "y": 100,
                "width": 80,
                "height": 80,
                "text": "D",
            },
            {"id": "e5", "type": "text", "x": 150, "y": 100, "text": "T"},
            {
                "id": "e6",
                "type": "group",
                "x": 300,
                "y": 100,
                "width": 200,
                "height": 150,
                "title": "Grp",
            },
        ]
        board.save()

        resp = self._auth_get(f"/api/whiteboard/boards/{board_data['room_code']}/svg/")
        data = resp.json()
        self.assertEqual(data["element_count"], 6)
        svg = data["svg"]
        self.assertIn("P", svg)
        self.assertIn("ellipse", svg)
        self.assertIn("polygon", svg)
        self.assertIn("Grp", svg)

    def test_svg_causal_connection(self):
        board_data = self._create_board()
        board = Board.objects.get(room_code=board_data["room_code"])
        board.elements = [
            {
                "id": "a",
                "type": "postit",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 60,
                "text": "Cause",
            },
            {
                "id": "b",
                "type": "postit",
                "x": 200,
                "y": 0,
                "width": 100,
                "height": 60,
                "text": "Effect",
            },
        ]
        board.connections = [
            {
                "id": "c1",
                "from": {"elementId": "a"},
                "to": {"elementId": "b"},
                "type": "causal",
            },
        ]
        board.save()

        resp = self._auth_get(f"/api/whiteboard/boards/{board_data['room_code']}/svg/")
        svg = resp.json()["svg"]
        self.assertIn("IF", svg)
        self.assertIn("THEN", svg)

    def test_svg_xml_escaping(self):
        board_data = self._create_board()
        board = Board.objects.get(room_code=board_data["room_code"])
        board.elements = [
            {
                "id": "e1",
                "type": "postit",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 60,
                "text": 'A < B & C > D "test"',
                "color": "yellow",
            },
        ]
        board.save()

        resp = self._auth_get(f"/api/whiteboard/boards/{board_data['room_code']}/svg/")
        svg = resp.json()["svg"]
        self.assertIn("&lt;", svg)
        self.assertIn("&amp;", svg)
        self.assertIn("&gt;", svg)
        self.assertIn("&quot;", svg)

    def test_guest_can_export_svg(self):
        board_data = self._create_board()
        board = Board.objects.get(room_code=board_data["room_code"])
        board.elements = [
            {
                "id": "e1",
                "type": "postit",
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 60,
                "text": "X",
            }
        ]
        board.save()
        invite = self._create_invite(self.room if hasattr(self, "room") else board_data["room_code"], "view")
        resp = self._guest_get(
            f"/api/whiteboard/boards/{board_data['room_code']}/svg/",
            invite["token"],
        )
        self.assertEqual(resp.status_code, 200)


# ==========================================================================
# Hypothesis Export
# ==========================================================================


class HypothesisExportTest(WhiteboardTestBase):
    def setUp(self):
        super().setUp()
        self.project = Project.objects.create(user=self.owner, title="Research")
        bd = self._auth_post(
            "/api/whiteboard/boards/create/",
            {"name": "Hypo Board", "project_id": str(self.project.id)},
        ).json()
        self.room = bd["room_code"]
        self.board = Board.objects.get(room_code=self.room)

    def test_export_hypothesis(self):
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/export-hypotheses/",
            {
                "causal_relationships": [
                    {
                        "statement": "If temp rises, yield drops",
                        "condition": "Temperature > 100C",
                        "effect": "Yield < 90%",
                        "prior_probability": 0.7,
                    }
                ]
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["created_count"], 1)
        self.assertEqual(data["project_id"], str(self.project.id))
        h = Hypothesis.objects.get(project=self.project)
        self.assertEqual(h.statement, "If temp rises, yield drops")
        self.assertAlmostEqual(h.prior_probability, 0.7)

    def test_export_deduplicates(self):
        Hypothesis.objects.create(
            project=self.project,
            statement="Existing hypothesis",
            prior_probability=0.5,
            current_probability=0.5,
        )
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/export-hypotheses/",
            {
                "causal_relationships": [
                    {"statement": "Existing hypothesis"},
                    {"statement": "New hypothesis"},
                ]
            },
        )
        data = resp.json()
        self.assertEqual(data["existing_count"], 1)
        self.assertEqual(data["created_count"], 1)

    def test_export_no_project_fails(self):
        unlinked = self._create_board("No Project")
        resp = self._auth_post(
            f"/api/whiteboard/boards/{unlinked['room_code']}/export-hypotheses/",
            {"causal_relationships": [{"statement": "X"}]},
        )
        self.assertEqual(resp.status_code, 400)

    def test_export_empty_relationships_fails(self):
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/export-hypotheses/",
            {"causal_relationships": []},
        )
        self.assertEqual(resp.status_code, 400)

    def test_export_non_participant_denied(self):
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/export-hypotheses/",
            {"causal_relationships": [{"statement": "X"}]},
            user=self.other,
        )
        self.assertEqual(resp.status_code, 403)

    def test_participant_can_export(self):
        BoardParticipant.objects.create(board=self.board, user=self.other)
        resp = self._auth_post(
            f"/api/whiteboard/boards/{self.room}/export-hypotheses/",
            {"causal_relationships": [{"statement": "Collab hypothesis"}]},
            user=self.other,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["created_count"], 1)


# ==========================================================================
# Guest Page View (HTML rendering)
# ==========================================================================


class GuestPageViewTest(WhiteboardTestBase):
    def test_valid_guest_page(self):
        bd = self._create_board()
        invite = self._create_invite(bd["room_code"], "edit")
        resp = self.client.get(f"/app/whiteboard/guest/{invite['token']}/")
        self.assertEqual(resp.status_code, 200)

    def test_invalid_token_404(self):
        resp = self.client.get("/app/whiteboard/guest/invalidtoken123/")
        self.assertEqual(resp.status_code, 404)

    def test_revoked_invite_403(self):
        bd = self._create_board()
        invite_data = self._create_invite(bd["room_code"], "edit")
        invite = BoardGuestInvite.objects.get(id=invite_data["id"])
        invite.is_active = False
        invite.save()
        resp = self.client.get(f"/app/whiteboard/guest/{invite_data['token']}/")
        self.assertEqual(resp.status_code, 403)

    def test_expired_invite_403(self):
        bd = self._create_board()
        invite_data = self._create_invite(bd["room_code"], "edit")
        invite = BoardGuestInvite.objects.get(id=invite_data["id"])
        invite.expires_at = timezone.now() - timedelta(days=1)
        invite.save()
        resp = self.client.get(f"/app/whiteboard/guest/{invite_data['token']}/")
        self.assertEqual(resp.status_code, 403)

"""Whiteboard API views for collaborative boards."""

import json
from datetime import timedelta

from django.db import IntegrityError
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from accounts.constants import GUEST_INVITE_EXPIRY_DAYS, GUEST_INVITE_LIMITS
from accounts.permissions import allow_guest, gated_paid

from .models import Board, BoardGuestInvite, BoardParticipant, BoardVote

PARTICIPANT_COLORS = [
    "#4a9f6e",
    "#47a5e8",
    "#e89547",
    "#a29bfe",
    "#ff7eb9",
    "#ffd93d",
    "#6bcb77",
    "#4d96ff",
]


# =============================================================================
# SOCKETS — integration points for other apps to pull from
# =============================================================================


def get_board_svg(board_id):
    """Socket: render a board as SVG. Used by A3, reports, ISO docs."""
    board = Board.objects.get(id=board_id)
    svg, width, height = _generate_svg(board, theme="dark")
    return {"svg": svg, "width": width, "height": height}


def get_board_summary(board_id):
    """Socket: get board metadata for embedding in other tools."""
    board = Board.objects.get(id=board_id)
    return {
        "id": str(board.id),
        "name": board.name,
        "room_code": board.room_code,
        "element_count": len(board.elements or []),
        "connection_count": len(board.connections or []),
        "updated_at": board.updated_at.isoformat(),
    }


# =============================================================================
# HANGING WIRES — integrations to reconnect later
# =============================================================================


def _resolve_project(user, project_id):
    """Hanging wire: resolve a project for board linking. Reconnect to core."""
    if not project_id:
        return None, None
    try:
        from core.models import Project

        project = Project.objects.get(id=project_id, user=user)
        return project, None
    except Exception:
        return None, None


def _emit_event(event_name, board, user):
    """Hanging wire: emit tool event. Reconnect to event system."""
    pass


# =============================================================================
# HELPERS
# =============================================================================


def get_participant_color(board, user):
    """Assign a consistent color to a participant."""
    participants = list(board.wb_participants.values_list("user_id", flat=True))
    if user.id in participants:
        idx = participants.index(user.id)
    else:
        idx = len(participants)
    return PARTICIPANT_COLORS[idx % len(PARTICIPANT_COLORS)]


def _build_participants_list(board, cutoff):
    """Build combined participants list (users + guests) for board responses."""
    active_users = board.wb_participants.filter(last_seen__gte=cutoff).select_related("user")
    active_guests = board.wb_guest_invites.filter(
        is_active=True,
        last_seen__gte=cutoff,
    ).exclude(display_name="")

    participants = [
        {
            "username": p.user.username,
            "color": p.color,
            "cursor_x": p.cursor_x,
            "cursor_y": p.cursor_y,
            "is_owner": p.user_id == board.owner_id,
            "is_guest": False,
        }
        for p in active_users
    ]
    participants.extend(
        [
            {
                "username": g.display_name,
                "color": g.color,
                "cursor_x": g.cursor_x,
                "cursor_y": g.cursor_y,
                "is_owner": False,
                "is_guest": True,
            }
            for g in active_guests
        ]
    )
    return participants


# =============================================================================
# BOARD CRUD
# =============================================================================


@gated_paid
@require_http_methods(["POST"])
def create_board(request):
    """Create a new board, optionally linked to a project."""
    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    name = data.get("name", "Untitled Board")
    project, _ = _resolve_project(request.user, data.get("project_id"))

    board = Board.objects.create(
        owner=request.user,
        name=name,
        project=project,
    )

    BoardParticipant.objects.create(
        board=board,
        user=request.user,
        color=PARTICIPANT_COLORS[0],
    )

    _emit_event("whiteboard.created", board, user=request.user)

    return JsonResponse(
        {
            "id": str(board.id),
            "room_code": board.room_code,
            "name": board.name,
            "project_id": str(project.id) if project else None,
            "url": f"/app/whiteboard/{board.room_code}/",
        }
    )


@allow_guest
@require_http_methods(["GET"])
def get_board(request, room_code):
    """Get board state by room code."""
    board = get_object_or_404(Board, room_code=room_code.upper())
    cutoff = timezone.now() - timedelta(seconds=30)

    vote_counts = {}
    for vote in board.wb_votes.all():
        vote_counts[vote.element_id] = vote_counts.get(vote.element_id, 0) + 1

    if request.is_guest:
        invite = request.guest_invite
        if invite.board_id != board.id:
            return JsonResponse({"error": "Access denied"}, status=403)

        guest_votes = list(board.wb_votes.filter(guest_invite=invite).values_list("element_id", flat=True))

        return JsonResponse(
            {
                "id": str(board.id),
                "room_code": board.room_code,
                "name": board.name,
                "elements": board.elements,
                "connections": board.connections,
                "zoom": board.zoom,
                "pan_x": board.pan_x,
                "pan_y": board.pan_y,
                "version": board.version,
                "voting_active": board.is_voting_active,
                "votes_per_user": board.wb_votes_per_user,
                "vote_counts": vote_counts,
                "user_votes": guest_votes,
                "user_votes_remaining": board.wb_votes_per_user - len(guest_votes),
                "participants": _build_participants_list(board, cutoff),
                "is_owner": False,
                "is_guest": True,
                "guest_permission": invite.permission,
                "guest_display_name": invite.display_name,
                "my_color": invite.color,
                "project": None,
            }
        )

    participant, created = BoardParticipant.objects.update_or_create(
        board=board,
        user=request.user,
        defaults={"color": get_participant_color(board, request.user)},
    )

    user_votes = list(board.wb_votes.filter(user=request.user).values_list("element_id", flat=True))

    return JsonResponse(
        {
            "id": str(board.id),
            "room_code": board.room_code,
            "name": board.name,
            "elements": board.elements,
            "connections": board.connections,
            "zoom": board.zoom,
            "pan_x": board.pan_x,
            "pan_y": board.pan_y,
            "version": board.version,
            "voting_active": board.is_voting_active,
            "votes_per_user": board.wb_votes_per_user,
            "vote_counts": vote_counts,
            "user_votes": user_votes,
            "user_votes_remaining": board.wb_votes_per_user - len(user_votes),
            "participants": _build_participants_list(board, cutoff),
            "is_owner": request.user.id == board.owner_id,
            "is_guest": False,
            "my_color": participant.color,
            "project": (
                {
                    "id": str(board.project.id),
                    "title": board.project.title,
                }
                if board.project
                else None
            ),
        }
    )


@allow_guest
@require_http_methods(["PUT", "PATCH"])
def update_board(request, room_code):
    """Update board state."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    client_version = data.get("version")
    if client_version is not None and client_version < board.version:
        return JsonResponse(
            {
                "conflict": True,
                "server_version": board.version,
                "elements": board.elements,
                "connections": board.connections,
            },
            status=409,
        )

    if request.is_guest:
        invite = request.guest_invite
        if invite.board_id != board.id:
            return JsonResponse({"error": "Access denied"}, status=403)
        if invite.permission == "view":
            return JsonResponse({"error": "View-only guests cannot edit"}, status=403)
        if "elements" in data:
            board.elements = data["elements"]
        if "connections" in data:
            board.connections = data["connections"]
        board.save()
        return JsonResponse({"success": True, "version": board.version})

    is_participant = BoardParticipant.objects.filter(board=board, user=request.user).exists()
    if request.user.id != board.owner_id and not is_participant:
        return JsonResponse({"error": "You must join this board before editing"}, status=403)

    if "elements" in data:
        board.elements = data["elements"]
    if "connections" in data:
        board.connections = data["connections"]
    if "name" in data:
        board.name = data["name"]
    if "zoom" in data:
        board.zoom = data["zoom"]
    if "pan_x" in data:
        board.pan_x = data["pan_x"]
    if "pan_y" in data:
        board.pan_y = data["pan_y"]
    if "project_id" in data:
        if data["project_id"]:
            proj, _ = _resolve_project(request.user, data["project_id"])
            if proj:
                board.project = proj
        else:
            board.project = None

    board.save()

    BoardParticipant.objects.filter(board=board, user=request.user).update(last_seen=timezone.now())

    _emit_event("whiteboard.updated", board, user=request.user)

    return JsonResponse({"success": True, "version": board.version})


@allow_guest
@require_http_methods(["POST"])
def update_cursor(request, room_code):
    """Update cursor position for presence."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if request.is_guest:
        invite = request.guest_invite
        if invite.board_id != board.id:
            return JsonResponse({"error": "Access denied"}, status=403)
        BoardGuestInvite.objects.filter(id=invite.id).update(
            cursor_x=data.get("x"),
            cursor_y=data.get("y"),
            last_seen=timezone.now(),
        )
    else:
        BoardParticipant.objects.filter(board=board, user=request.user).update(
            cursor_x=data.get("x"),
            cursor_y=data.get("y"),
            last_seen=timezone.now(),
        )

    return JsonResponse({"success": True})


# =============================================================================
# VOTING
# =============================================================================


@gated_paid
@require_http_methods(["POST"])
def toggle_voting(request, room_code):
    """Toggle voting mode (owner only)."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    if request.user.id != board.owner_id:
        return JsonResponse({"error": "Only the owner can control voting"}, status=403)

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    board.is_voting_active = data.get("active", not board.is_voting_active)
    if "votes_per_user" in data:
        board.wb_votes_per_user = data["votes_per_user"]

    if data.get("clear_votes"):
        board.wb_votes.all().delete()

    board.save()

    return JsonResponse(
        {
            "voting_active": board.is_voting_active,
            "votes_per_user": board.wb_votes_per_user,
        }
    )


@allow_guest
@require_http_methods(["POST"])
def add_vote(request, room_code):
    """Add a dot vote to an element."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    if not board.is_voting_active:
        return JsonResponse({"error": "Voting is not active"}, status=400)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    element_id = data.get("element_id")
    if not element_id:
        return JsonResponse({"error": "element_id required"}, status=400)

    if request.is_guest:
        invite = request.guest_invite
        if invite.board_id != board.id:
            return JsonResponse({"error": "Access denied"}, status=403)
        if invite.permission != "edit_vote":
            return JsonResponse({"error": "You don't have voting permission"}, status=403)
        user_vote_count = board.wb_votes.filter(guest_invite=invite).count()
        if user_vote_count >= board.wb_votes_per_user:
            return JsonResponse({"error": "Vote limit reached"}, status=400)
        try:
            BoardVote.objects.create(board=board, guest_invite=invite, element_id=element_id)
        except IntegrityError:
            return JsonResponse({"error": "Already voted on this element"}, status=400)
    else:
        user_vote_count = board.wb_votes.filter(user=request.user).count()
        if user_vote_count >= board.wb_votes_per_user:
            return JsonResponse({"error": "Vote limit reached"}, status=400)
        try:
            BoardVote.objects.create(board=board, user=request.user, element_id=element_id)
        except IntegrityError:
            return JsonResponse({"error": "Already voted on this element"}, status=400)

    vote_count = board.wb_votes.filter(element_id=element_id).count()
    votes_remaining = board.wb_votes_per_user - user_vote_count - 1

    return JsonResponse(
        {
            "success": True,
            "element_id": element_id,
            "vote_count": vote_count,
            "votes_remaining": votes_remaining,
        }
    )


@allow_guest
@require_http_methods(["DELETE"])
def remove_vote(request, room_code, element_id):
    """Remove a dot vote from an element."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    if request.is_guest:
        invite = request.guest_invite
        if invite.board_id != board.id:
            return JsonResponse({"error": "Access denied"}, status=403)
        deleted, _ = BoardVote.objects.filter(
            board=board,
            guest_invite=invite,
            element_id=element_id,
        ).delete()
        remaining_count = board.wb_votes.filter(guest_invite=invite).count()
    else:
        deleted, _ = BoardVote.objects.filter(
            board=board,
            user=request.user,
            element_id=element_id,
        ).delete()
        remaining_count = board.wb_votes.filter(user=request.user).count()

    if deleted == 0:
        return JsonResponse({"error": "Vote not found"}, status=404)

    vote_count = board.wb_votes.filter(element_id=element_id).count()

    return JsonResponse(
        {
            "success": True,
            "element_id": element_id,
            "vote_count": vote_count,
            "votes_remaining": board.wb_votes_per_user - remaining_count,
        }
    )


@gated_paid
@require_http_methods(["GET"])
def list_boards(request):
    """List user's boards (owned + participated)."""
    project_id = request.GET.get("project_id")

    owned = Board.objects.filter(owner=request.user).select_related("project")
    participated = (
        Board.objects.filter(participants__user=request.user).exclude(owner=request.user).select_related("project")
    )

    if project_id:
        owned = owned.filter(project_id=project_id)
        participated = participated.filter(project_id=project_id)

    def board_to_dict(b, is_owner=False):
        return {
            "id": str(b.id),
            "room_code": b.room_code,
            "name": b.name,
            "is_owner": is_owner,
            "updated_at": b.updated_at.isoformat(),
            "participant_count": b.participants.count(),
            "project": ({"id": str(b.project.id), "title": b.project.title} if b.project else None),
        }

    return JsonResponse(
        {
            "owned": [board_to_dict(b, True) for b in owned],
            "participated": [board_to_dict(b, False) for b in participated],
        }
    )


@gated_paid
@require_http_methods(["DELETE"])
def delete_board(request, room_code):
    """Delete a board (owner only)."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    if request.user.id != board.owner_id:
        return JsonResponse({"error": "Only the owner can delete the board"}, status=403)

    board.delete()

    return JsonResponse({"success": True})


# =============================================================================
# SVG/PNG EXPORT
# =============================================================================

POSTIT_COLORS = {
    "yellow": "#fef3c7",
    "green": "#d1fae5",
    "pink": "#fce7f3",
    "blue": "#dbeafe",
    "purple": "#ede9fe",
    "orange": "#ffedd5",
}


def _escape_xml(text):
    """Escape text for XML/SVG."""
    if not text:
        return ""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _render_element_svg(el, offset_x=0, offset_y=0):
    """Render a single element to SVG markup."""
    el_type = el.get("type", "rectangle")
    x = el.get("x", 0) - offset_x
    y = el.get("y", 0) - offset_y
    width = el.get("width", 120)
    height = el.get("height", 60)
    text = _escape_xml(el.get("text", ""))
    color = el.get("color", "yellow")

    if el_type == "postit":
        fill = POSTIT_COLORS.get(color, POSTIT_COLORS["yellow"])
        return f"""<g transform="translate({x},{y})">
            <rect width="{width}" height="{height}" fill="{fill}" stroke="#666" stroke-width="1" rx="3"/>
            <text x="{width / 2}" y="{height / 2}" text-anchor="middle" dominant-baseline="middle" font-size="12" fill="#333">{text}</text>
        </g>"""

    elif el_type == "rectangle":
        return f"""<g transform="translate({x},{y})">
            <rect width="{width}" height="{height}" fill="#1e3a2f" stroke="#4a9f6e" stroke-width="2" rx="4"/>
            <text x="{width / 2}" y="{height / 2}" text-anchor="middle" dominant-baseline="middle" font-size="12" fill="#e8efe8">{text}</text>
        </g>"""

    elif el_type == "oval":
        rx = width / 2
        ry = height / 2
        cx = x + rx
        cy = y + ry
        return f"""<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="#1e3a2f" stroke="#4a9f6e" stroke-width="2"/>
        <text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="middle" font-size="12" fill="#e8efe8">{text}</text>"""

    elif el_type == "diamond":
        cx = x + width / 2
        cy = y + height / 2
        points = f"{cx},{y} {x + width},{cy} {cx},{y + height} {x},{cy}"
        return f"""<polygon points="{points}" fill="#1e3a2f" stroke="#4a9f6e" stroke-width="2"/>
        <text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="middle" font-size="11" fill="#e8efe8">{text}</text>"""

    elif el_type == "text":
        return f"""<text x="{x}" y="{y + 16}" font-size="14" fill="#e8efe8">{text}</text>"""

    elif el_type == "group":
        title = _escape_xml(el.get("title", "Group"))
        return f"""<g transform="translate({x},{y})">
            <rect width="{width}" height="{height}" fill="none" stroke="#4a9f6e" stroke-width="2" stroke-dasharray="5,5" rx="8"/>
            <text x="10" y="20" font-size="12" font-weight="bold" fill="#4a9f6e">{title}</text>
        </g>"""

    elif el_type == "fishbone":
        effect = _escape_xml(el.get("effect", "Effect"))
        categories = el.get("categories", [])
        svg_parts = [f'<g transform="translate({x},{y})">']
        spine_len = 500
        svg_parts.append(f'<line x1="50" y1="150" x2="{50 + spine_len}" y2="150" stroke="#4a9f6e" stroke-width="3"/>')
        svg_parts.append(
            f"""<rect x="{50 + spine_len}" y="120" width="100" height="60" fill="#2c5f2d" stroke="#4a9f6e" stroke-width="2" rx="4"/>
            <text x="{100 + spine_len}" y="155" text-anchor="middle" font-size="12" font-weight="bold" fill="#fff">{effect}</text>"""
        )
        for i, cat in enumerate(categories[:6]):
            cat_name = _escape_xml(cat.get("name", f"Category {i + 1}"))
            bone_x = 100 + (i % 3) * 150
            is_top = i < 3
            bone_y = 80 if is_top else 220
            svg_parts.append(
                f"""<line x1="{bone_x}" y1="150" x2="{bone_x}" y2="{bone_y}" stroke="#4a9f6e" stroke-width="2"/>
                <text x="{bone_x}" y="{bone_y - 10 if is_top else bone_y + 15}" text-anchor="middle" font-size="11" fill="#4a9f6e">{cat_name}</text>"""
            )
        svg_parts.append("</g>")
        return "\n".join(svg_parts)

    return ""


def _render_connection_svg(conn, elements, offset_x=0, offset_y=0):
    """Render a connection between elements."""
    from_id = conn.get("from", {}).get("elementId")
    to_id = conn.get("to", {}).get("elementId")

    from_el = next((e for e in elements if e.get("id") == from_id), None)
    to_el = next((e for e in elements if e.get("id") == to_id), None)

    if not from_el or not to_el:
        return ""

    from_x = from_el.get("x", 0) + from_el.get("width", 120) / 2 - offset_x
    from_y = from_el.get("y", 0) + from_el.get("height", 60) / 2 - offset_y
    to_x = to_el.get("x", 0) + to_el.get("width", 120) / 2 - offset_x
    to_y = to_el.get("y", 0) + to_el.get("height", 60) / 2 - offset_y

    conn_type = conn.get("type", "arrow")

    if conn_type == "causal":
        return f"""<g>
            <defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#4a9f6e"/></marker></defs>
            <path d="M{from_x},{from_y} C{from_x + 50},{from_y} {to_x - 50},{to_y} {to_x},{to_y}" fill="none" stroke="#4a9f6e" stroke-width="2" marker-end="url(#arrowhead)"/>
            <text x="{from_x + 20}" y="{from_y - 10}" font-size="10" fill="#6bcb77">IF</text>
            <text x="{to_x - 30}" y="{to_y - 10}" font-size="10" fill="#6bcb77">THEN</text>
        </g>"""
    else:
        return f"""<line x1="{from_x}" y1="{from_y}" x2="{to_x}" y2="{to_y}" stroke="#4a9f6e" stroke-width="1.5" marker-end="url(#arrowhead)"/>"""


def _generate_svg(board, theme="dark"):
    """Generate SVG string from a Board's elements and connections."""
    elements = board.elements or []
    connections = board.connections or []

    if not elements:
        return None, 0, 0

    min_x = min(el.get("x", 0) for el in elements)
    min_y = min(el.get("y", 0) for el in elements)
    max_x = max(el.get("x", 0) + el.get("width", 120) for el in elements)
    max_y = max(el.get("y", 0) + el.get("height", 60) for el in elements)

    padding = 20
    width = max_x - min_x + padding * 2
    height = max_y - min_y + padding * 2
    offset_x = min_x - padding
    offset_y = min_y - padding

    bg_color = "#1a1a1a" if theme == "dark" else "#ffffff"

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background:{bg_color}">',
        '<defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#4a9f6e"/></marker></defs>',
    ]

    for conn in connections:
        svg_parts.append(_render_connection_svg(conn, elements, offset_x, offset_y))

    for el in elements:
        svg_parts.append(_render_element_svg(el, offset_x, offset_y))

    svg_parts.append("</svg>")
    return "\n".join(svg_parts), width, height


@allow_guest
@require_http_methods(["GET"])
def export_svg(request, room_code):
    """Export whiteboard as SVG."""
    board = get_object_or_404(Board, room_code=room_code.upper())
    theme = request.GET.get("theme", "dark")

    svg_content, width, height = _generate_svg(board, theme=theme)

    if svg_content is None:
        return JsonResponse(
            {
                "svg": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200"><text x="200" y="100" text-anchor="middle" fill="#666">Empty whiteboard</text></svg>',
                "width": 400,
                "height": 200,
                "element_count": 0,
            }
        )

    return JsonResponse(
        {
            "svg": svg_content,
            "width": width,
            "height": height,
            "element_count": len(board.elements or []),
            "connection_count": len(board.connections or []),
            "board_name": board.name,
        }
    )


@allow_guest
@require_http_methods(["GET"])
def export_png(request, room_code):
    """Export whiteboard as PNG."""
    import base64

    import cairosvg

    board = get_object_or_404(Board, room_code=room_code.upper())
    theme = request.GET.get("theme", "light")

    svg_content, width, height = _generate_svg(board, theme=theme)

    if svg_content is None:
        return JsonResponse({"error": "Whiteboard is empty"}, status=400)

    output_width = min(int(request.GET.get("width", width)), 2400)
    png_bytes = cairosvg.svg2png(
        bytestring=svg_content.encode("utf-8"),
        output_width=output_width,
    )

    return JsonResponse(
        {
            "png": base64.b64encode(png_bytes).decode("utf-8"),
            "width": output_width,
            "height": height,
            "board_name": board.name,
        }
    )


# =============================================================================
# GUEST INVITE MANAGEMENT
# =============================================================================


@gated_paid
@require_http_methods(["POST"])
def create_guest_invite(request, room_code):
    """Create a guest invite link. Owner only."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    if request.user.id != board.owner_id:
        return JsonResponse({"error": "Only the board owner can create invites"}, status=403)

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    permission = data.get("permission", "view")
    if permission not in ("view", "edit", "edit_vote"):
        return JsonResponse({"error": "Invalid permission. Use: view, edit, edit_vote"}, status=400)

    tier = request.user.tier
    limit = GUEST_INVITE_LIMITS.get(tier, 0)
    active_count = BoardGuestInvite.objects.filter(board=board, is_active=True).count()

    if active_count >= limit:
        return JsonResponse(
            {
                "error": f"Guest invite limit reached ({limit} per board on your plan)",
                "limit": limit,
                "active": active_count,
                "upgrade_url": "/billing/checkout/",
            },
            status=403,
        )

    expiry_days = GUEST_INVITE_EXPIRY_DAYS.get(tier, 7)
    if expiry_days > 0:
        expires_at = timezone.now() + timedelta(days=expiry_days)
    else:
        expires_at = timezone.now() + timedelta(days=36500)

    used_colors = set(board.wb_participants.values_list("color", flat=True)) | set(
        BoardGuestInvite.objects.filter(board=board, is_active=True).values_list("color", flat=True)
    )
    color = "#ff7eb9"
    for c in PARTICIPANT_COLORS:
        if c not in used_colors:
            color = c
            break

    token = BoardGuestInvite.generate_token()

    invite = BoardGuestInvite.objects.create(
        board=board,
        token=token,
        permission=permission,
        expires_at=expires_at,
        color=color,
    )

    return JsonResponse(
        {
            "id": str(invite.id),
            "token": token,
            "url": f"/app/whiteboard/guest/{token}/",
            "permission": permission,
            "expires_at": invite.expires_at.isoformat(),
            "color": color,
        }
    )


@gated_paid
@require_http_methods(["GET"])
def list_guest_invites(request, room_code):
    """List guest invites for a board. Owner only."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    if request.user.id != board.owner_id:
        return JsonResponse({"error": "Only the board owner can view invites"}, status=403)

    invites = BoardGuestInvite.objects.filter(board=board).order_by("-created_at")
    cutoff = timezone.now() - timedelta(seconds=30)

    return JsonResponse(
        {
            "invites": [
                {
                    "id": str(inv.id),
                    "display_name": inv.display_name or "(waiting to join)",
                    "permission": inv.permission,
                    "is_active": inv.is_active,
                    "is_expired": inv.is_expired,
                    "is_online": inv.last_seen is not None and inv.last_seen >= cutoff,
                    "created_at": inv.created_at.isoformat(),
                    "expires_at": inv.expires_at.isoformat(),
                    "url": f"/app/whiteboard/guest/{inv.token}/",
                    "color": inv.color,
                }
                for inv in invites
            ],
        }
    )


@gated_paid
@require_http_methods(["DELETE"])
def revoke_guest_invite(request, room_code, invite_id):
    """Revoke a guest invite. Owner only."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    if request.user.id != board.owner_id:
        return JsonResponse({"error": "Only the board owner can revoke invites"}, status=403)

    try:
        invite = BoardGuestInvite.objects.get(id=invite_id, board=board)
    except BoardGuestInvite.DoesNotExist:
        return JsonResponse({"error": "Invite not found"}, status=404)

    invite.is_active = False
    invite.save(update_fields=["is_active"])

    return JsonResponse({"success": True, "id": str(invite.id)})


@require_http_methods(["POST"])
def set_guest_name(request, room_code):
    """Set guest display name."""
    guest_token = request.headers.get("X-Guest-Token", "").strip()
    if not guest_token:
        return JsonResponse({"error": "Guest token required"}, status=401)

    try:
        invite = BoardGuestInvite.objects.select_related("board").get(token=guest_token)
    except BoardGuestInvite.DoesNotExist:
        return JsonResponse({"error": "Invalid token"}, status=401)

    if not invite.is_valid:
        return JsonResponse({"error": "Invite expired or revoked"}, status=403)

    if invite.board.room_code != room_code.upper():
        return JsonResponse({"error": "Token does not match this board"}, status=403)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    name = (data.get("display_name") or "").strip()[:100]
    if not name:
        return JsonResponse({"error": "Display name is required"}, status=400)

    invite.display_name = name
    invite.save(update_fields=["display_name"])

    return JsonResponse({"success": True, "display_name": name})


# =============================================================================
# GUEST PAGE VIEW
# =============================================================================


def guest_board_view(request, token):
    """Serve the whiteboard template for a guest user."""
    from django.shortcuts import render

    try:
        invite = BoardGuestInvite.objects.select_related("board").get(token=token)
    except BoardGuestInvite.DoesNotExist:
        return render(request, "guest_invalid.html", status=404)

    if not invite.is_active:
        return render(
            request,
            "guest_invalid.html",
            {"reason": "This invite has been revoked by the board owner."},
            status=403,
        )

    if invite.is_expired:
        return render(
            request,
            "guest_invalid.html",
            {"reason": "This invite link has expired."},
            status=403,
        )

    return render(
        request,
        "whiteboard.html",
        {
            "base_template": "base_guest.html",
            "guest_token": token,
            "guest_room_code": invite.board.room_code,
            "guest_display_name": invite.display_name,
            "guest_permission": invite.permission,
        },
    )

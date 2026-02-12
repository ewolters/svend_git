"""Whiteboard API views for collaborative boards."""

import json
from datetime import timedelta

from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404

from accounts.permissions import gated_paid
from .models import Board, BoardParticipant, BoardVote
from core.models import Project, Hypothesis


# Participant colors for easy differentiation
PARTICIPANT_COLORS = [
    "#4a9f6e",  # Green
    "#47a5e8",  # Blue
    "#e89547",  # Orange
    "#a29bfe",  # Purple
    "#ff7eb9",  # Pink
    "#ffd93d",  # Yellow
    "#6bcb77",  # Light green
    "#4d96ff",  # Bright blue
]


def get_participant_color(board, user):
    """Assign a consistent color to a participant."""
    participants = list(board.participants.values_list('user_id', flat=True))
    if user.id in participants:
        idx = participants.index(user.id)
    else:
        idx = len(participants)
    return PARTICIPANT_COLORS[idx % len(PARTICIPANT_COLORS)]


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def create_board(request):
    """Create a new board, optionally linked to a project."""
    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    name = data.get("name", "Untitled Board")
    project_id = data.get("project_id")

    # Link to project if provided
    project = None
    if project_id:
        try:
            project = Project.objects.get(id=project_id, user=request.user)
        except Project.DoesNotExist:
            return JsonResponse({"error": "Study not found"}, status=404)

    board = Board.objects.create(
        owner=request.user,
        name=name,
        project=project,
    )

    # Add creator as first participant
    BoardParticipant.objects.create(
        board=board,
        user=request.user,
        color=PARTICIPANT_COLORS[0],
    )

    return JsonResponse({
        "id": str(board.id),
        "room_code": board.room_code,
        "name": board.name,
        "project_id": str(project.id) if project else None,
        "url": f"/app/whiteboard/{board.room_code}/",
    })


@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def get_board(request, room_code):
    """Get board state by room code."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    # Add/update participant
    participant, created = BoardParticipant.objects.update_or_create(
        board=board,
        user=request.user,
        defaults={"color": get_participant_color(board, request.user)},
    )

    # Get active participants (seen in last 30 seconds)
    cutoff = timezone.now() - timedelta(seconds=30)
    active_participants = board.participants.filter(last_seen__gte=cutoff).select_related('user')

    # Get vote counts per element
    vote_counts = {}
    for vote in board.votes.all():
        vote_counts[vote.element_id] = vote_counts.get(vote.element_id, 0) + 1

    # Get current user's votes
    user_votes = list(board.votes.filter(user=request.user).values_list('element_id', flat=True))

    return JsonResponse({
        "id": str(board.id),
        "room_code": board.room_code,
        "name": board.name,
        "elements": board.elements,
        "connections": board.connections,
        "zoom": board.zoom,
        "pan_x": board.pan_x,
        "pan_y": board.pan_y,
        "version": board.version,
        "voting_active": board.voting_active,
        "votes_per_user": board.votes_per_user,
        "vote_counts": vote_counts,
        "user_votes": user_votes,
        "user_votes_remaining": board.votes_per_user - len(user_votes),
        "participants": [
            {
                "username": p.user.username,
                "color": p.color,
                "cursor_x": p.cursor_x,
                "cursor_y": p.cursor_y,
                "is_owner": p.user_id == board.owner_id,
            }
            for p in active_participants
        ],
        "is_owner": request.user.id == board.owner_id,
        "my_color": participant.color,
        "project": {
            "id": str(board.project.id),
            "title": board.project.title,
        } if board.project else None,
    })


@csrf_exempt
@gated_paid
@require_http_methods(["PUT", "PATCH"])
def update_board(request, room_code):
    """Update board state."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Check version for conflict detection
    client_version = data.get("version")
    if client_version is not None and client_version < board.version:
        # Client is behind - return current state for merge
        return JsonResponse({
            "conflict": True,
            "server_version": board.version,
            "elements": board.elements,
            "connections": board.connections,
        }, status=409)

    # Update fields
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
        project_id = data["project_id"]
        if project_id:
            from core.models import Project
            try:
                project = Project.objects.get(id=project_id, user=request.user)
                board.project = project
            except Project.DoesNotExist:
                pass
        else:
            board.project = None

    board.save()

    # Update participant last_seen
    BoardParticipant.objects.filter(board=board, user=request.user).update(last_seen=timezone.now())

    return JsonResponse({
        "success": True,
        "version": board.version,
    })


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def update_cursor(request, room_code):
    """Update cursor position for presence."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    BoardParticipant.objects.filter(board=board, user=request.user).update(
        cursor_x=data.get("x"),
        cursor_y=data.get("y"),
        last_seen=timezone.now(),
    )

    return JsonResponse({"success": True})


@csrf_exempt
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

    board.voting_active = data.get("active", not board.voting_active)
    if "votes_per_user" in data:
        board.votes_per_user = data["votes_per_user"]

    # Optionally clear votes when starting new voting round
    if data.get("clear_votes"):
        board.votes.all().delete()

    board.save()

    return JsonResponse({
        "voting_active": board.voting_active,
        "votes_per_user": board.votes_per_user,
    })


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def add_vote(request, room_code):
    """Add a dot vote to an element."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    if not board.voting_active:
        return JsonResponse({"error": "Voting is not active"}, status=400)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    element_id = data.get("element_id")
    if not element_id:
        return JsonResponse({"error": "element_id required"}, status=400)

    # Check vote limit
    user_vote_count = board.votes.filter(user=request.user).count()
    if user_vote_count >= board.votes_per_user:
        return JsonResponse({"error": "Vote limit reached"}, status=400)

    # Add vote (unique constraint prevents duplicates)
    try:
        BoardVote.objects.create(board=board, user=request.user, element_id=element_id)
    except Exception:
        return JsonResponse({"error": "Already voted on this element"}, status=400)

    # Return updated counts
    vote_count = board.votes.filter(element_id=element_id).count()
    votes_remaining = board.votes_per_user - user_vote_count - 1

    return JsonResponse({
        "success": True,
        "element_id": element_id,
        "vote_count": vote_count,
        "votes_remaining": votes_remaining,
    })


@csrf_exempt
@gated_paid
@require_http_methods(["DELETE"])
def remove_vote(request, room_code, element_id):
    """Remove a dot vote from an element."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    deleted, _ = BoardVote.objects.filter(
        board=board,
        user=request.user,
        element_id=element_id,
    ).delete()

    if deleted == 0:
        return JsonResponse({"error": "Vote not found"}, status=404)

    # Return updated counts
    vote_count = board.votes.filter(element_id=element_id).count()
    user_vote_count = board.votes.filter(user=request.user).count()

    return JsonResponse({
        "success": True,
        "element_id": element_id,
        "vote_count": vote_count,
        "votes_remaining": board.votes_per_user - user_vote_count,
    })


@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def list_boards(request):
    """List user's boards (owned + participated).

    Query params:
    - project_id: filter by project
    """
    project_id = request.GET.get("project_id")

    owned = Board.objects.filter(owner=request.user).select_related('project')
    participated = Board.objects.filter(participants__user=request.user).exclude(owner=request.user).select_related('project')

    # Filter by project if specified
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
            "project": {
                "id": str(b.project.id),
                "title": b.project.title,
            } if b.project else None,
        }

    return JsonResponse({
        "owned": [board_to_dict(b, True) for b in owned],
        "participated": [board_to_dict(b, False) for b in participated],
    })


@csrf_exempt
@gated_paid
@require_http_methods(["DELETE"])
def delete_board(request, room_code):
    """Delete a board (owner only)."""
    board = get_object_or_404(Board, room_code=room_code.upper())

    if request.user.id != board.owner_id:
        return JsonResponse({"error": "Only the owner can delete the board"}, status=403)

    board.delete()

    return JsonResponse({"success": True})


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def export_hypotheses(request, room_code):
    """Export causal relationships from whiteboard as hypotheses.

    Extracts if-then connections and creates Hypothesis objects
    linked to the board's project.
    """
    board = get_object_or_404(Board, room_code=room_code.upper())

    if not board.project:
        return JsonResponse({
            "error": "Board must be linked to a study to export hypotheses"
        }, status=400)

    try:
        data = json.loads(request.body) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    # Get causal relationships from request body
    # Frontend sends: { causal_relationships: [{statement, condition, effect, prior}, ...] }
    causal_rels = data.get("causal_relationships", [])

    if not causal_rels:
        return JsonResponse({
            "error": "No causal relationships provided"
        }, status=400)

    created_hypotheses = []

    for rel in causal_rels:
        statement = rel.get("statement", "")
        if not statement:
            continue

        # Check for duplicates (same statement in same project)
        existing = Hypothesis.objects.filter(
            project=board.project,
            statement=statement
        ).first()

        if existing:
            # Skip duplicates but note them
            created_hypotheses.append({
                "id": str(existing.id),
                "statement": existing.statement,
                "status": "existing",
                "message": "Already exists"
            })
            continue

        # Create new hypothesis
        hypothesis = Hypothesis.objects.create(
            project=board.project,
            statement=statement,
            mechanism=f"Condition: {rel.get('condition', '')}\nEffect: {rel.get('effect', '')}",
            prior_probability=rel.get("prior_probability", 0.5),
            current_probability=rel.get("prior_probability", 0.5),
        )

        created_hypotheses.append({
            "id": str(hypothesis.id),
            "statement": hypothesis.statement,
            "status": "created",
            "prior_probability": hypothesis.prior_probability,
        })

    return JsonResponse({
        "success": True,
        "project_id": str(board.project.id),
        "project_title": board.project.title,
        "hypotheses": created_hypotheses,
        "created_count": len([h for h in created_hypotheses if h["status"] == "created"]),
        "existing_count": len([h for h in created_hypotheses if h["status"] == "existing"]),
    })


# ============================================================================
# SVG Export for A3 Embedding
# ============================================================================

# Post-it colors mapping
POSTIT_COLORS = {
    'yellow': '#fef3c7',
    'green': '#d1fae5',
    'pink': '#fce7f3',
    'blue': '#dbeafe',
    'purple': '#ede9fe',
    'orange': '#ffedd5',
}


def _escape_xml(text):
    """Escape text for XML/SVG."""
    if not text:
        return ""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _render_element_svg(el, offset_x=0, offset_y=0):
    """Render a single element to SVG markup."""
    el_type = el.get('type', 'rectangle')
    x = el.get('x', 0) - offset_x
    y = el.get('y', 0) - offset_y
    width = el.get('width', 120)
    height = el.get('height', 60)
    text = _escape_xml(el.get('text', ''))
    color = el.get('color', 'yellow')

    if el_type == 'postit':
        fill = POSTIT_COLORS.get(color, POSTIT_COLORS['yellow'])
        return f'''<g transform="translate({x},{y})">
            <rect width="{width}" height="{height}" fill="{fill}" stroke="#666" stroke-width="1" rx="3"/>
            <text x="{width/2}" y="{height/2}" text-anchor="middle" dominant-baseline="middle" font-size="12" fill="#333">{text}</text>
        </g>'''

    elif el_type == 'rectangle':
        return f'''<g transform="translate({x},{y})">
            <rect width="{width}" height="{height}" fill="#1e3a2f" stroke="#4a9f6e" stroke-width="2" rx="4"/>
            <text x="{width/2}" y="{height/2}" text-anchor="middle" dominant-baseline="middle" font-size="12" fill="#e8efe8">{text}</text>
        </g>'''

    elif el_type == 'oval':
        rx = width / 2
        ry = height / 2
        cx = x + rx
        cy = y + ry
        return f'''<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="#1e3a2f" stroke="#4a9f6e" stroke-width="2"/>
        <text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="middle" font-size="12" fill="#e8efe8">{text}</text>'''

    elif el_type == 'diamond':
        cx = x + width / 2
        cy = y + height / 2
        points = f"{cx},{y} {x+width},{cy} {cx},{y+height} {x},{cy}"
        return f'''<polygon points="{points}" fill="#1e3a2f" stroke="#4a9f6e" stroke-width="2"/>
        <text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="middle" font-size="11" fill="#e8efe8">{text}</text>'''

    elif el_type == 'text':
        return f'''<text x="{x}" y="{y+16}" font-size="14" fill="#e8efe8">{text}</text>'''

    elif el_type == 'group':
        title = _escape_xml(el.get('title', 'Group'))
        return f'''<g transform="translate({x},{y})">
            <rect width="{width}" height="{height}" fill="none" stroke="#4a9f6e" stroke-width="2" stroke-dasharray="5,5" rx="8"/>
            <text x="10" y="20" font-size="12" font-weight="bold" fill="#4a9f6e">{title}</text>
        </g>'''

    elif el_type == 'fishbone':
        # Simplified fishbone - just render the effect and categories
        effect = _escape_xml(el.get('effect', 'Effect'))
        categories = el.get('categories', [])
        svg_parts = [f'<g transform="translate({x},{y})">']

        # Spine
        spine_len = 500
        svg_parts.append(f'<line x1="50" y1="150" x2="{50+spine_len}" y2="150" stroke="#4a9f6e" stroke-width="3"/>')

        # Effect head
        svg_parts.append(f'''<rect x="{50+spine_len}" y="120" width="100" height="60" fill="#2c5f2d" stroke="#4a9f6e" stroke-width="2" rx="4"/>
            <text x="{100+spine_len}" y="155" text-anchor="middle" font-size="12" font-weight="bold" fill="#fff">{effect}</text>''')

        # Categories as bones
        for i, cat in enumerate(categories[:6]):
            cat_name = _escape_xml(cat.get('name', f'Category {i+1}'))
            bone_x = 100 + (i % 3) * 150
            is_top = i < 3
            bone_y = 80 if is_top else 220
            svg_parts.append(f'''<line x1="{bone_x}" y1="150" x2="{bone_x}" y2="{bone_y}" stroke="#4a9f6e" stroke-width="2"/>
                <text x="{bone_x}" y="{bone_y - 10 if is_top else bone_y + 15}" text-anchor="middle" font-size="11" fill="#4a9f6e">{cat_name}</text>''')

        svg_parts.append('</g>')
        return '\n'.join(svg_parts)

    return ''


def _render_connection_svg(conn, elements, offset_x=0, offset_y=0):
    """Render a connection between elements."""
    from_id = conn.get('from', {}).get('elementId')
    to_id = conn.get('to', {}).get('elementId')

    from_el = next((e for e in elements if e.get('id') == from_id), None)
    to_el = next((e for e in elements if e.get('id') == to_id), None)

    if not from_el or not to_el:
        return ''

    # Calculate center points
    from_x = from_el.get('x', 0) + from_el.get('width', 120) / 2 - offset_x
    from_y = from_el.get('y', 0) + from_el.get('height', 60) / 2 - offset_y
    to_x = to_el.get('x', 0) + to_el.get('width', 120) / 2 - offset_x
    to_y = to_el.get('y', 0) + to_el.get('height', 60) / 2 - offset_y

    conn_type = conn.get('type', 'arrow')

    if conn_type == 'causal':
        # Causal connection with IF/THEN labels
        return f'''<g>
            <defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#4a9f6e"/></marker></defs>
            <path d="M{from_x},{from_y} C{from_x+50},{from_y} {to_x-50},{to_y} {to_x},{to_y}" fill="none" stroke="#4a9f6e" stroke-width="2" marker-end="url(#arrowhead)"/>
            <text x="{from_x+20}" y="{from_y-10}" font-size="10" fill="#6bcb77">IF</text>
            <text x="{to_x-30}" y="{to_y-10}" font-size="10" fill="#6bcb77">THEN</text>
        </g>'''
    else:
        # Simple arrow
        return f'''<line x1="{from_x}" y1="{from_y}" x2="{to_x}" y2="{to_y}" stroke="#4a9f6e" stroke-width="1.5" marker-end="url(#arrowhead)"/>'''


@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def export_svg(request, room_code):
    """Export whiteboard as SVG for embedding in A3 reports.

    Query params:
    - width: max width (default 600)
    - height: max height (default 400)
    - theme: 'dark' (default) or 'light'
    """
    board = get_object_or_404(Board, room_code=room_code.upper())

    elements = board.elements or []
    connections = board.connections or []

    if not elements:
        return JsonResponse({
            "svg": f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200"><text x="200" y="100" text-anchor="middle" fill="#666">Empty whiteboard</text></svg>',
            "width": 400,
            "height": 200,
            "element_count": 0,
        })

    # Calculate bounding box
    min_x = min(el.get('x', 0) for el in elements)
    min_y = min(el.get('y', 0) for el in elements)
    max_x = max(el.get('x', 0) + el.get('width', 120) for el in elements)
    max_y = max(el.get('y', 0) + el.get('height', 60) for el in elements)

    # Add padding
    padding = 20
    width = max_x - min_x + padding * 2
    height = max_y - min_y + padding * 2

    # Offset to normalize coordinates
    offset_x = min_x - padding
    offset_y = min_y - padding

    # Get theme
    theme = request.GET.get('theme', 'dark')
    bg_color = '#1a1a1a' if theme == 'dark' else '#ffffff'
    text_color = '#e8efe8' if theme == 'dark' else '#333333'

    # Build SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background:{bg_color}">',
        f'<defs><marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#4a9f6e"/></marker></defs>',
    ]

    # Render connections first (behind elements)
    for conn in connections:
        svg_parts.append(_render_connection_svg(conn, elements, offset_x, offset_y))

    # Render elements
    for el in elements:
        svg_parts.append(_render_element_svg(el, offset_x, offset_y))

    svg_parts.append('</svg>')

    svg_content = '\n'.join(svg_parts)

    return JsonResponse({
        "svg": svg_content,
        "width": width,
        "height": height,
        "element_count": len(elements),
        "connection_count": len(connections),
        "board_name": board.name,
    })

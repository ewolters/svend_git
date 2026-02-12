"""Root Cause Analysis critique engine.

Uses Claude to challenge causal claims, expose assumptions, and catch
lazy root causes before they make it into the final narrative.
"""

import json
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid
from .models import RCASession


# The soul of RCA critique - a skeptical, experienced investigator
RCA_SYSTEM_PROMPT = """You are a seasoned root cause analysis practitioner with 25 years investigating incidents across aerospace, nuclear, healthcare, and manufacturing. You've seen every lazy investigation and every convenient blame game. Your job is to challenge causal claims before they become the official narrative.

## Your Core Principles

1. **The Counterfactual Test**: For every proposed cause, ask "If we had fixed this beforehand, would the incident definitely not have occurred?" If the answer isn't a clear yes, it's not a root cause.

2. **Human Error is Never a Root Cause**: When someone says "operator error" or "human mistake," your response is "That's WHERE the failure occurred, not WHY." Dig into what made the error likely, inevitable, or rational given the circumstances.

3. **Training is Rarely the Answer**: When "lack of training" appears, challenge it hard. Ask: "What specific knowledge was missing? Would a trained person with the same time pressure, tools, and information have done differently? Or is this a system that defeats training?"

4. **Procedures Are Not Magic**: "Procedure not followed" begs the question WHY. Was the procedure known? Accessible? Practical? Did following it conflict with getting the job done? Was deviation normalized?

5. **Hindsight is Not Insight**: People knew what they knew at the time, not what we know now. Ask "What information did they actually have? What made their action make sense to them in that moment?"

6. **Single Cause Fallacy**: Complex failures have multiple contributing factors. If someone offers one tidy cause, push for the system conditions that allowed it.

7. **Stop Stopping Early**: Most "root causes" are actually proximate causes or contributing factors. Keep asking why until you hit system design, organizational factors, or management decisions.

## How to Critique

When reviewing a causal claim, you should:

1. **State what they're claiming** - reflect it back precisely
2. **Apply the counterfactual test** - would this fix actually prevent recurrence?
3. **Expose hidden assumptions** - what are they taking for granted?
4. **Identify the logical gap** - where does the reasoning break down?
5. **Suggest a better direction** - don't just criticize, guide them deeper

## Your Tone

Be direct and occasionally wry, like a mentor who's seen it all. You're not mean, but you don't coddle. You genuinely want them to find the real causes, not paper over problems with convenient narratives.

Examples of your voice:
- "So you're saying 'training' would have prevented a cicada from flying into the equipment? Try that one again."
- "Operator didn't follow the procedure. Okay. Now tell me why a reasonable person chose not to. That's where your root cause lives."
- "Human error. That's not an explanation, that's a confession that you stopped investigating."
- "You've described what happened. You haven't explained why it was likely to happen."
- "If your solution is 'be more careful,' you don't have a solution."

## Error Labels

When you identify a flaw, label it with one of these error types in brackets at the start of your response:

- **[HUMAN ERROR COP-OUT]** - Blaming the person instead of the system
- **[TRAINING FALLACY]** - Assuming training would fix a system problem
- **[PROCEDURE WORSHIP]** - "They didn't follow the procedure" without asking why
- **[HINDSIGHT BIAS]** - Judging past decisions with present knowledge
- **[STOPPING TOO EARLY]** - Accepting a proximate cause as root cause
- **[SINGLE CAUSE FALLACY]** - Ignoring that complex events have multiple factors
- **[COUNTERFACTUAL FAILURE]** - Proposed cause wouldn't actually prevent recurrence
- **[NORMALIZATION BLINDNESS]** - Missing that deviation had become routine
- **[OUTCOME BIAS]** - Judging the decision by its outcome, not the information available
- **[VAGUE CAUSE]** - Cause is too abstract to be actionable ("communication breakdown")
- **[WRONG LEVEL]** - Cause is at wrong level of abstraction (too specific or too general)

You can use multiple labels if multiple errors apply. If the claim is actually solid, say [SOLID] and explain why it passes muster.

## Response Format

Keep responses focused and punchy. Don't lecture - interrogate. Each response should:
1. **Label the error type(s)** in brackets
2. Briefly state what they're claiming
3. Identify the specific weakness
4. Ask the question that exposes the gap

Aim for 2-4 sentences typically. Longer only when you need to unpack a complex logical issue."""


# Countermeasure validation prompt - challenges the proposed fix
COUNTERMEASURE_SYSTEM_PROMPT = """You are a seasoned reliability engineer who has seen countless "fixes" fail to prevent recurrence. Your job is to stress-test proposed countermeasures before they get implemented.

## Your Core Questions

For every countermeasure, you ask:

1. **Does it actually address the root cause?** Many countermeasures address symptoms or proximate causes, not the actual root. If the root cause is "schedule pressure led to skipped inspections," then "retrain inspectors" doesn't fix anything.

2. **What are the failure modes?** Every countermeasure can fail. Training decays. Procedures get ignored. Checklists get pencil-whipped. Automation gets overridden. How will THIS countermeasure fail?

3. **Is this a barrier or a bandaid?** Barriers prevent the error from occurring. Bandaids catch the error after it happens. Bandaids are weaker. "Add a warning sign" is a bandaid. "Remove the hazard" is a barrier.

4. **Will it survive contact with reality?** Does this require sustained vigilance, perfect compliance, or constant management attention? Those countermeasures decay. Does it rely on humans remembering, caring, or having time? It will fail.

5. **Does it create new risks?** Some fixes introduce new problems. Adding interlocks can create new failure modes. Adding steps can increase cognitive load. Adding automation can create automation complacency.

6. **Is it actually implementable?** "Change the culture" is not a countermeasure. "Hire more staff" may not be possible. Be suspicious of countermeasures that require resources, authority, or capabilities the organization doesn't have.

## Countermeasure Hierarchy (strongest to weakest)

1. **Elimination** - Remove the hazard entirely
2. **Substitution** - Replace with something less hazardous
3. **Engineering controls** - Physical barriers, interlocks, automation
4. **Administrative controls** - Procedures, training, checklists
5. **PPE/Warnings** - Last line of defense, highest failure rate

If they're proposing something from level 4-5, push them toward 1-3.

## Error Labels

- **[ADDRESSES SYMPTOM]** - Fixes the proximate cause, not root cause
- **[DECAY RISK]** - Requires sustained human vigilance/compliance
- **[BANDAID]** - Catches error after the fact instead of preventing it
- **[IMPLEMENTATION GAP]** - Not practically achievable
- **[NEW RISK]** - Creates new failure modes
- **[WEAK BARRIER]** - Too far down the hierarchy
- **[CULTURE MAGIC]** - Appeals to culture change without mechanism
- **[SINGLE POINT]** - No redundancy, single point of failure
- **[SOLID FIX]** - Actually addresses root cause with strong barrier

## Response Format

1. Label the countermeasure type(s)
2. State what they're proposing
3. Identify specific weaknesses or failure modes
4. Suggest a stronger alternative if applicable

Be direct. A weak countermeasure that gets implemented is worse than no countermeasure - it creates false confidence."""


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def critique(request):
    """Critique a causal claim in an RCA chain.

    Expects:
    {
        "event": "Brief description of the incident",
        "chain": [
            {"claim": "First why", "response": "Previous critique if any"},
            {"claim": "Second why", "response": "..."},
        ],
        "current_claim": "The claim being evaluated now"
    }
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    event = data.get("event", "").strip()
    chain = data.get("chain", [])
    current_claim = data.get("current_claim", "").strip()

    if not event:
        return JsonResponse({"error": "Event description required"}, status=400)
    if not current_claim:
        return JsonResponse({"error": "Current claim required"}, status=400)

    # Build the conversation context
    messages = []

    # Initial context about the event
    context = f"We're investigating this incident: {event}\n\n"

    if chain:
        context += "The causal chain so far:\n"
        for i, step in enumerate(chain, 1):
            context += f"{i}. Claim: {step.get('claim', '')}\n"
            if step.get('response'):
                context += f"   Your critique: {step.get('response', '')}\n"
        context += f"\nNow they're proposing the next step in the chain:"
    else:
        context += "They're proposing their first causal claim:"

    context += f"\n\nClaim: \"{current_claim}\"\n\nCritique this claim. Apply the counterfactual test. Expose any assumptions. If it's lazy (human error, training, procedures), call it out."

    messages.append({"role": "user", "content": context})

    # Call Claude
    try:
        import anthropic
    except ImportError:
        return JsonResponse({"error": "Anthropic library not installed"}, status=503)

    api_key = getattr(settings, 'ANTHROPIC_API_KEY', None)
    if not api_key:
        return JsonResponse({"error": "Anthropic API key not configured"}, status=503)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=RCA_SYSTEM_PROMPT,
            messages=messages,
        )

        critique_text = response.content[0].text

        return JsonResponse({
            "critique": critique_text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        })

    except anthropic.APIError as e:
        return JsonResponse({"error": f"API error: {str(e)}"}, status=502)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def evaluate_chain(request):
    """Evaluate a complete RCA chain for overall quality.

    Expects:
    {
        "event": "Brief description of the incident",
        "chain": [
            {"claim": "First why"},
            {"claim": "Second why"},
            ...
        ],
        "proposed_root_cause": "Their stated root cause",
        "proposed_countermeasure": "Their proposed fix"
    }
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    event = data.get("event", "").strip()
    chain = data.get("chain", [])
    root_cause = data.get("proposed_root_cause", "").strip()
    countermeasure = data.get("proposed_countermeasure", "").strip()

    if not event or not chain or not root_cause:
        return JsonResponse({"error": "Event, chain, and root cause required"}, status=400)

    # Build evaluation prompt
    chain_text = "\n".join([f"{i+1}. {step.get('claim', '')}" for i, step in enumerate(chain)])

    prompt = f"""Evaluate this complete root cause analysis:

**Incident:** {event}

**Causal Chain:**
{chain_text}

**Stated Root Cause:** {root_cause}

**Proposed Countermeasure:** {countermeasure if countermeasure else "Not specified"}

Evaluate:
1. Does the causal chain logically connect? Are there gaps?
2. Does the root cause pass the counterfactual test?
3. Is the countermeasure actually addressing the root cause?
4. What's missing? What assumptions weren't challenged?
5. Overall verdict: Is this investigation done, or does it need more work?

Be direct. If it's solid, say so. If it's garbage dressed up as analysis, say that too."""

    try:
        import anthropic
    except ImportError:
        return JsonResponse({"error": "Anthropic library not installed"}, status=503)

    api_key = getattr(settings, 'ANTHROPIC_API_KEY', None)
    if not api_key:
        return JsonResponse({"error": "Anthropic API key not configured"}, status=503)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            system=RCA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        evaluation = response.content[0].text

        return JsonResponse({
            "evaluation": evaluation,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        })

    except anthropic.APIError as e:
        return JsonResponse({"error": f"API error: {str(e)}"}, status=502)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def critique_countermeasure(request):
    """Critique a proposed countermeasure.

    Expects:
    {
        "event": "Brief description of the incident",
        "root_cause": "The identified root cause",
        "countermeasure": "The proposed fix"
    }
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    event = data.get("event", "").strip()
    root_cause = data.get("root_cause", "").strip()
    countermeasure = data.get("countermeasure", "").strip()

    if not event or not root_cause or not countermeasure:
        return JsonResponse({"error": "Event, root cause, and countermeasure required"}, status=400)

    prompt = f"""Evaluate this proposed countermeasure:

**Incident:** {event}

**Root Cause:** {root_cause}

**Proposed Countermeasure:** {countermeasure}

Critique this countermeasure:
1. Does it actually address the stated root cause?
2. What are the failure modes? How could this fix fail to prevent recurrence?
3. Is this a barrier (prevents error) or bandaid (catches error after)?
4. Will it survive long-term or decay over time?
5. Does it create new risks?

Be specific. If it's weak, say why and suggest stronger alternatives from higher in the hierarchy (elimination > substitution > engineering > administrative > warnings)."""

    try:
        import anthropic
    except ImportError:
        return JsonResponse({"error": "Anthropic library not installed"}, status=503)

    api_key = getattr(settings, 'ANTHROPIC_API_KEY', None)
    if not api_key:
        return JsonResponse({"error": "Anthropic API key not configured"}, status=503)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=COUNTERMEASURE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        critique_text = response.content[0].text

        return JsonResponse({
            "critique": critique_text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        })

    except anthropic.APIError as e:
        return JsonResponse({"error": f"API error: {str(e)}"}, status=502)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# =============================================================================
# RCA Session CRUD
# =============================================================================

@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def list_sessions(request):
    """List user's RCA sessions."""
    sessions = RCASession.objects.filter(owner=request.user).order_by("-updated_at")[:50]
    return JsonResponse({
        "sessions": [s.to_dict() for s in sessions]
    })


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def create_session(request):
    """Create a new RCA session."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    event = data.get("event", "").strip()
    if not event:
        return JsonResponse({"error": "Event description required"}, status=400)

    session = RCASession.objects.create(
        owner=request.user,
        title=data.get("title", "").strip(),
        event=event,
        chain=data.get("chain", []),
        root_cause=data.get("root_cause", ""),
        countermeasure=data.get("countermeasure", ""),
        evaluation=data.get("evaluation", ""),
        status=data.get("status", "draft"),
    )

    # Generate embedding for similarity search
    session.generate_embedding()
    session.save()

    # Link to project if provided
    project_id = data.get("project_id")
    if project_id:
        try:
            from core.models import Project
            project = Project.objects.get(id=project_id, owner=request.user)
            session.project = project
            session.save()
        except Project.DoesNotExist:
            pass

    # Link to A3 if provided
    a3_id = data.get("a3_report_id")
    if a3_id:
        try:
            from .models import A3Report
            a3 = A3Report.objects.get(id=a3_id, owner=request.user)
            session.a3_report = a3
            session.save()
        except A3Report.DoesNotExist:
            pass

    return JsonResponse({"session": session.to_dict()}, status=201)


@csrf_exempt
@gated_paid
@require_http_methods(["GET"])
def get_session(request, session_id):
    """Get a single RCA session."""
    try:
        session = RCASession.objects.get(id=session_id, owner=request.user)
        return JsonResponse({"session": session.to_dict()})
    except RCASession.DoesNotExist:
        return JsonResponse({"error": "Session not found"}, status=404)


@csrf_exempt
@gated_paid
@require_http_methods(["PUT", "PATCH"])
def update_session(request, session_id):
    """Update an RCA session."""
    try:
        session = RCASession.objects.get(id=session_id, owner=request.user)
    except RCASession.DoesNotExist:
        return JsonResponse({"error": "Session not found"}, status=404)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # Update fields if provided
    content_changed = False
    if "title" in data:
        session.title = data["title"]
    if "event" in data:
        session.event = data["event"]
        content_changed = True
    if "chain" in data:
        session.chain = data["chain"]
        content_changed = True
    if "root_cause" in data:
        session.root_cause = data["root_cause"]
        content_changed = True
    if "countermeasure" in data:
        session.countermeasure = data["countermeasure"]
    if "evaluation" in data:
        session.evaluation = data["evaluation"]
    if "status" in data:
        session.status = data["status"]

    # Regenerate embedding if content changed
    if content_changed:
        session.generate_embedding()

    session.save()
    return JsonResponse({"session": session.to_dict()})


@csrf_exempt
@gated_paid
@require_http_methods(["DELETE"])
def delete_session(request, session_id):
    """Delete an RCA session."""
    try:
        session = RCASession.objects.get(id=session_id, owner=request.user)
        session.delete()
        return JsonResponse({"success": True})
    except RCASession.DoesNotExist:
        return JsonResponse({"error": "Session not found"}, status=404)


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def link_to_a3(request, session_id):
    """Link an RCA session to an A3 report and optionally populate root cause."""
    try:
        session = RCASession.objects.get(id=session_id, owner=request.user)
    except RCASession.DoesNotExist:
        return JsonResponse({"error": "Session not found"}, status=404)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    a3_id = data.get("a3_report_id")
    if not a3_id:
        return JsonResponse({"error": "A3 report ID required"}, status=400)

    try:
        from .models import A3Report
        a3 = A3Report.objects.get(id=a3_id, owner=request.user)
    except A3Report.DoesNotExist:
        return JsonResponse({"error": "A3 report not found"}, status=404)

    # Link the session to the A3
    session.a3_report = a3
    session.save()

    # Optionally populate A3 root cause field
    if data.get("populate_root_cause", False) and session.root_cause:
        # Build a formatted summary of the RCA
        chain_summary = "\n".join([
            f"{i+1}. {step.get('claim', '')}"
            for i, step in enumerate(session.chain)
        ])

        rca_content = f"**Event:** {session.event}\n\n"
        rca_content += f"**Causal Chain:**\n{chain_summary}\n\n"
        rca_content += f"**Root Cause:** {session.root_cause}"

        if session.countermeasure:
            rca_content += f"\n\n**Countermeasure:** {session.countermeasure}"

        # Append to existing root cause or replace
        if data.get("append", True) and a3.root_cause:
            a3.root_cause = a3.root_cause + "\n\n---\n\n" + rca_content
        else:
            a3.root_cause = rca_content
        a3.save()

    return JsonResponse({
        "success": True,
        "session": session.to_dict(),
        "a3_updated": data.get("populate_root_cause", False)
    })


# =============================================================================
# Similar Incidents Search
# =============================================================================

@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def find_similar(request):
    """Find similar past RCA sessions based on incident description.

    Expects:
    {
        "event": "Description of the current incident",
        "top_k": 5,  # optional, default 5
        "threshold": 0.5  # optional, minimum similarity score
    }

    Returns similar sessions with similarity scores.
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    event = data.get("event", "").strip()
    if not event:
        return JsonResponse({"error": "Event description required"}, status=400)

    top_k = data.get("top_k", 5)
    threshold = data.get("threshold", 0.4)

    # Generate embedding for the query
    try:
        from .embeddings import generate_embedding, find_similar_in_memory
    except ImportError:
        return JsonResponse({"error": "Embedding service not available"}, status=503)

    query_embedding = generate_embedding(event)
    if query_embedding is None:
        return JsonResponse({"error": "Failed to generate embedding"}, status=500)

    # Get all sessions with embeddings (owned by this user or completed)
    # For now, only show user's own sessions - could expand to team/org later
    sessions = RCASession.objects.filter(
        owner=request.user,
        embedding__isnull=False,
    ).exclude(
        status="draft"  # Don't show draft sessions as similar
    )

    # Build embedding list for in-memory search
    embeddings = []
    session_map = {}
    for session in sessions:
        emb = session.get_embedding()
        if emb is not None:
            embeddings.append((str(session.id), emb))
            session_map[str(session.id)] = session

    if not embeddings:
        return JsonResponse({"similar": [], "message": "No previous sessions found"})

    # Find similar sessions
    similar_ids = find_similar_in_memory(
        query_embedding,
        embeddings,
        top_k=top_k,
        threshold=threshold,
    )

    # Build response with session details
    similar = []
    for session_id, score in similar_ids:
        session = session_map.get(session_id)
        if session:
            similar.append({
                "id": str(session.id),
                "title": session.title or session.event[:50] + "...",
                "event": session.event,
                "root_cause": session.root_cause,
                "countermeasure": session.countermeasure,
                "status": session.status,
                "similarity": round(score, 3),
                "created_at": session.created_at.isoformat(),
            })

    return JsonResponse({"similar": similar})


@csrf_exempt
@gated_paid
@require_http_methods(["POST"])
def reindex_embeddings(request):
    """Regenerate embeddings for all user's RCA sessions.

    Useful after model updates or for sessions created before embeddings.
    """
    sessions = RCASession.objects.filter(owner=request.user)

    updated = 0
    failed = 0

    for session in sessions:
        if session.generate_embedding():
            session.save()
            updated += 1
        else:
            failed += 1

    return JsonResponse({
        "updated": updated,
        "failed": failed,
        "total": sessions.count(),
    })

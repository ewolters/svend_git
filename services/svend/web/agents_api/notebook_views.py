"""
Notebook API views — NB-001.

Endpoints for Notebook lifecycle: create, list, detail, trials,
pages, conclude with Hansei Kai, and yokoten.

<!-- impl: notebook_views:list_create_notebooks -->
<!-- impl: notebook_views:notebook_detail -->
<!-- impl: notebook_views:list_create_trials -->
<!-- impl: notebook_views:trial_detail -->
<!-- impl: notebook_views:complete_trial -->
<!-- impl: notebook_views:list_create_pages -->
<!-- impl: notebook_views:conclude_notebook -->
<!-- impl: notebook_views:list_yokoten -->
<!-- impl: notebook_views:adopt_yokoten -->
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid
from core.models import (
    HanseiKai,
    Notebook,
    NotebookPage,
    Project,
    Trial,
    Yokoten,
    YokotenAdoption,
)

logger = logging.getLogger("svend.notebook")


# ---------------------------------------------------------------------------
# Verdict narrative generation — NB-001 §2.2.2
# ---------------------------------------------------------------------------


def generate_verdict_narrative(trial):
    """
    Auto-generate a verdict narrative from trial data and linked page statistics.

    NB-001 §2.2.2: When before and after data exist, compute statistical
    significance and effect size, then generate a human-readable narrative.
    The user can accept, modify, or override.

    Returns a narrative string or empty string if insufficient data.
    """
    if trial.before_value is None or trial.after_value is None:
        return ""

    nb = trial.notebook
    metric = nb.baseline_metric or "metric"
    unit = nb.baseline_unit or ""
    delta = trial.delta
    delta_pct = trial.delta_pct

    # Determine improvement direction from charter goal
    improving = True  # default: positive delta = improvement
    try:
        goal_target = float(nb.project.goal_target) if nb.project.goal_target else None
        goal_baseline = float(nb.project.goal_baseline) if nb.project.goal_baseline else None
        if goal_target is not None and goal_baseline is not None:
            improving = (goal_target > goal_baseline and delta > 0) or (goal_target < goal_baseline and delta < 0)
    except (ValueError, TypeError, AttributeError):
        # Fall back to: negative delta on a "rate" metric = improvement
        improving = delta < 0 if "rate" in metric.lower() or "defect" in metric.lower() else delta > 0

    # Try to extract p-value and effect size from linked after-analysis page
    p_value = None
    effect_mag = None
    ci_low = None
    ci_high = None

    after_pages = NotebookPage.objects.filter(trial=trial, trial_role="after").order_by("-created_at")
    for page in after_pages[:1]:
        outputs = page.outputs or {}
        # Extract from page outputs using same patterns as standardize.py
        from agents_api.analysis.standardize import _classify_effect, _extract_p_value

        p_value = _extract_p_value(outputs)
        effect_mag = _classify_effect(outputs)
        stats = outputs.get("statistics", {})
        if isinstance(stats, dict):
            ci = stats.get("confidence_interval") or stats.get("ci")
            if isinstance(ci, (list, tuple)) and len(ci) == 2:
                ci_low, ci_high = ci

    # Build narrative parts
    parts = []

    # Delta description
    abs_delta = abs(delta) if delta else 0
    direction = "decreased" if delta < 0 else "increased"
    delta_str = f"{metric.replace('_', ' ')} {direction} {abs_delta:.1f}{unit}"
    if delta_pct is not None:
        delta_str += f" ({abs(delta_pct):.1f}%)"
    before_after = f"{trial.before_value:.1f}{unit} → {trial.after_value:.1f}{unit}"
    parts.append(f"{delta_str} ({before_after}).")

    # Statistical significance
    if p_value is not None:
        if p_value < 0.001:
            sig_label = "Highly significant"
            parts.append(f"{sig_label} (p<0.001).")
        elif p_value < 0.01:
            sig_label = "Significant"
            parts.append(f"{sig_label} (p={p_value:.3f}).")
        elif p_value < 0.05:
            sig_label = "Statistically significant"
            parts.append(f"{sig_label} (p={p_value:.3f}).")
        else:
            parts.append(f"Not statistically significant (p={p_value:.2f}).")

    # Effect size
    if effect_mag:
        parts.append(f"Effect size is {effect_mag}.")

    # Confidence interval
    if ci_low is not None and ci_high is not None:
        parts.append(f"95% CI: [{ci_low:.2f}, {ci_high:.2f}].")

    # Verdict summary
    if trial.verdict == "improved":
        if improving:
            parts.append("Change moved the metric toward the goal — recommend adoption.")
        else:
            parts.append("Change moved the metric, but away from the goal direction.")
    elif trial.verdict == "no_effect":
        parts.append("No meaningful change observed.")
    elif trial.verdict == "degraded":
        parts.append("Metric moved in the wrong direction.")
    elif trial.verdict == "inconclusive":
        parts.append("Insufficient data to determine effect.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def _serialize_notebook(nb):
    return {
        "id": str(nb.id),
        "project_id": str(nb.project_id),
        "title": nb.title,
        "description": nb.description,
        "status": nb.status,
        "baseline_metric": nb.baseline_metric,
        "baseline_value": nb.baseline_value,
        "baseline_unit": nb.baseline_unit,
        "baseline_date": nb.baseline_date.isoformat() if nb.baseline_date else None,
        "current_value": nb.current_value,
        "current_date": nb.current_date.isoformat() if nb.current_date else None,
        "progress_pct": nb.progress_pct,
        "active_trial_id": str(nb.active_trial_id) if nb.active_trial_id else None,
        "created_at": nb.created_at.isoformat() if nb.created_at else None,
        "updated_at": nb.updated_at.isoformat() if nb.updated_at else None,
        "concluded_at": nb.concluded_at.isoformat() if nb.concluded_at else None,
    }


def _serialize_trial(t):
    return {
        "id": str(t.id),
        "notebook_id": str(t.notebook_id),
        "sequence": t.sequence,
        "title": t.title,
        "description": t.description,
        "before_value": t.before_value,
        "before_date": t.before_date.isoformat() if t.before_date else None,
        "after_value": t.after_value,
        "after_date": t.after_date.isoformat() if t.after_date else None,
        "verdict": t.verdict,
        "verdict_narrative": t.verdict_narrative,
        "delta": t.delta,
        "delta_pct": t.delta_pct,
        "adopted": t.adopted,
        "started_at": t.started_at.isoformat() if t.started_at else None,
        "completed_at": t.completed_at.isoformat() if t.completed_at else None,
    }


def _serialize_page(p):
    return {
        "id": str(p.id),
        "notebook_id": str(p.notebook_id),
        "page_type": p.page_type,
        "title": p.title,
        "source_tool": p.source_tool,
        "inputs": p.inputs,
        "outputs": p.outputs,
        "rendered_html": p.rendered_html,
        "narrative": p.narrative,
        "sequence": p.sequence,
        "trial_id": str(p.trial_id) if p.trial_id else None,
        "trial_role": p.trial_role,
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }


def _serialize_yokoten(y):
    return {
        "id": str(y.id),
        "source_notebook_id": str(y.source_notebook_id),
        "source_trial_id": str(y.source_trial_id) if y.source_trial_id else None,
        "learning": y.learning,
        "context": y.context,
        "applicable_to": y.applicable_to,
        "created_at": y.created_at.isoformat() if y.created_at else None,
        "adoption_count": y.adoptions.count(),
    }


# ---------------------------------------------------------------------------
# Notebook CRUD
# ---------------------------------------------------------------------------


@require_http_methods(["GET", "POST"])
@gated_paid
def list_create_notebooks(request):
    """
    GET  — List user's notebooks.
    POST — Create a new notebook.

    POST body: {"project_id": uuid, "title": str, "description": str?,
                "baseline_metric": str?, "baseline_value": float?,
                "baseline_unit": str?}
    """
    if request.method == "GET":
        notebooks = Notebook.objects.filter(owner=request.user).select_related("project")
        return JsonResponse({"notebooks": [_serialize_notebook(nb) for nb in notebooks]})

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "title is required"}, status=400)

    project_id = data.get("project_id")
    if not project_id:
        return JsonResponse({"error": "project_id is required"}, status=400)

    try:
        project = Project.objects.get(id=project_id, user=request.user)
    except Project.DoesNotExist:
        return JsonResponse({"error": "Project not found"}, status=404)

    nb = Notebook.objects.create(
        project=project,
        title=title,
        description=data.get("description", ""),
        owner=request.user,
        tenant=getattr(request.user, "active_tenant", None),
        baseline_metric=data.get("baseline_metric", ""),
        baseline_value=data.get("baseline_value"),
        baseline_unit=data.get("baseline_unit", ""),
    )
    logger.info(f"Notebook created: {nb.id} for project {project.id}")
    return JsonResponse(_serialize_notebook(nb), status=201)


@require_http_methods(["GET", "PATCH", "DELETE"])
@gated_paid
def notebook_detail(request, notebook_id):
    """GET/PATCH/DELETE a notebook."""
    try:
        nb = Notebook.objects.get(id=notebook_id, owner=request.user)
    except Notebook.DoesNotExist:
        return JsonResponse({"error": "Notebook not found"}, status=404)

    if request.method == "GET":
        result = _serialize_notebook(nb)
        result["trials"] = [_serialize_trial(t) for t in nb.trials.all()]
        result["pages"] = [_serialize_page(p) for p in nb.pages.all()]
        # Include hansei kai if concluded
        try:
            hk = nb.hansei_kai
            result["hansei_kai"] = {
                "what_went_well": hk.what_went_well,
                "what_didnt": hk.what_didnt,
                "what_next": hk.what_next,
                "key_learning": hk.key_learning,
                "carry_forward": hk.carry_forward,
            }
        except HanseiKai.DoesNotExist:
            result["hansei_kai"] = None
        return JsonResponse(result)

    if request.method == "PATCH":
        try:
            data = json.loads(request.body)
        except (json.JSONDecodeError, ValueError):
            return JsonResponse({"error": "Invalid JSON body"}, status=400)

        updatable = [
            "title",
            "description",
            "baseline_summary",
            "baseline_metric",
            "baseline_value",
            "baseline_unit",
            "baseline_date",
            "current_value",
            "current_date",
        ]
        for field in updatable:
            if field in data:
                setattr(nb, field, data[field])
        nb.save()
        nb.refresh_from_db()
        return JsonResponse(_serialize_notebook(nb))

    # DELETE
    nb.delete()
    return JsonResponse({"ok": True})


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------


@require_http_methods(["GET", "POST"])
@gated_paid
def list_create_trials(request, notebook_id):
    """
    GET  — List trials for a notebook.
    POST — Create a new trial.

    POST body: {"title": str, "description": str?,
                "before_value": float?, "before_date": str?}
    """
    try:
        nb = Notebook.objects.get(id=notebook_id, owner=request.user)
    except Notebook.DoesNotExist:
        return JsonResponse({"error": "Notebook not found"}, status=404)

    if request.method == "GET":
        return JsonResponse({"trials": [_serialize_trial(t) for t in nb.trials.all()]})

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    title = data.get("title", "").strip()
    if not title:
        return JsonResponse({"error": "title is required"}, status=400)

    trial = Trial(
        notebook=nb,
        title=title,
        description=data.get("description", ""),
        before_value=data.get("before_value"),
        before_date=data.get("before_date"),
        created_by=request.user,
    )
    trial.save()

    # Set as active trial on notebook
    nb.active_trial = trial
    nb.save(update_fields=["active_trial", "updated_at"])

    logger.info(f"Trial {trial.sequence} created in notebook {nb.id}")
    return JsonResponse(_serialize_trial(trial), status=201)


@require_http_methods(["GET", "PATCH"])
@gated_paid
def trial_detail(request, notebook_id, trial_id):
    """GET/PATCH a trial."""
    try:
        trial = Trial.objects.get(id=trial_id, notebook_id=notebook_id, notebook__owner=request.user)
    except Trial.DoesNotExist:
        return JsonResponse({"error": "Trial not found"}, status=404)

    if request.method == "GET":
        result = _serialize_trial(trial)
        result["pages"] = [_serialize_page(p) for p in trial.pages.all()]
        return JsonResponse(result)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    updatable = [
        "title",
        "description",
        "before_value",
        "before_date",
        "after_value",
        "after_date",
    ]
    for field in updatable:
        if field in data:
            setattr(trial, field, data[field])
    trial.save()
    trial.refresh_from_db()
    return JsonResponse(_serialize_trial(trial))


@require_http_methods(["POST"])
@gated_paid
def complete_trial(request, notebook_id, trial_id):
    """
    Complete a trial with a verdict.

    POST body: {"verdict": str, "adopted": bool?, "verdict_narrative": str?}
    """
    try:
        trial = Trial.objects.get(id=trial_id, notebook_id=notebook_id, notebook__owner=request.user)
    except Trial.DoesNotExist:
        return JsonResponse({"error": "Trial not found"}, status=404)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    verdict = data.get("verdict")
    if verdict not in [c[0] for c in Trial.Verdict.choices]:
        return JsonResponse(
            {"error": f"Invalid verdict. Must be one of: {[c[0] for c in Trial.Verdict.choices]}"},
            status=400,
        )

    # User-provided narrative takes priority; otherwise auto-generate (NB-001 §2.2.2)
    if data.get("verdict_narrative"):
        trial.verdict_narrative = data["verdict_narrative"]

    trial.complete(verdict=verdict, adopted=data.get("adopted", False))

    # Auto-generate narrative if user didn't provide one
    if not trial.verdict_narrative:
        narrative = generate_verdict_narrative(trial)
        if narrative:
            trial.verdict_narrative = narrative
            trial.save(update_fields=["verdict_narrative"])

    # Clear active trial if this was it
    nb = trial.notebook
    if nb.active_trial_id == trial.id:
        nb.active_trial = None
        nb.save(update_fields=["active_trial", "updated_at"])

    logger.info(f"Trial {trial.sequence} completed: {verdict}, adopted={trial.adopted}")
    return JsonResponse(_serialize_trial(trial))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@require_http_methods(["GET", "POST"])
@gated_paid
def list_create_pages(request, notebook_id):
    """
    GET  — List pages for a notebook.
    POST — Create a new page (frozen snapshot).

    POST body: {"page_type": str, "title": str, "source_tool": str?,
                "inputs": dict?, "outputs": dict?, "rendered_html": str?,
                "narrative": str?, "trial_id": uuid?, "trial_role": str?}
    """
    try:
        nb = Notebook.objects.get(id=notebook_id, owner=request.user)
    except Notebook.DoesNotExist:
        return JsonResponse({"error": "Notebook not found"}, status=404)

    if request.method == "GET":
        return JsonResponse({"pages": [_serialize_page(p) for p in nb.pages.all()]})

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    title = data.get("title", "").strip()
    page_type = data.get("page_type", "")
    if not title or not page_type:
        return JsonResponse({"error": "title and page_type are required"}, status=400)

    if page_type not in [c[0] for c in NotebookPage.PageType.choices]:
        return JsonResponse({"error": "Invalid page_type"}, status=400)

    trial = None
    trial_id = data.get("trial_id")
    if trial_id:
        try:
            trial = Trial.objects.get(id=trial_id, notebook=nb)
        except Trial.DoesNotExist:
            return JsonResponse({"error": "Trial not found in this notebook"}, status=404)

    page = NotebookPage.objects.create(
        notebook=nb,
        page_type=page_type,
        title=title,
        source_tool=data.get("source_tool", ""),
        inputs=data.get("inputs", {}),
        outputs=data.get("outputs", {}),
        rendered_html=data.get("rendered_html", ""),
        narrative=data.get("narrative", ""),
        trial=trial,
        trial_role=data.get("trial_role", ""),
        created_by=request.user,
    )

    logger.info(f"Page '{title}' ({page_type}) added to notebook {nb.id}")
    return JsonResponse(_serialize_page(page), status=201)


# ---------------------------------------------------------------------------
# Conclude + Hansei Kai
# ---------------------------------------------------------------------------


@require_http_methods(["POST"])
@gated_paid
def conclude_notebook(request, notebook_id):
    """
    Conclude a notebook with Hansei Kai reflection.

    POST body: {"what_went_well": str, "what_didnt": str, "what_next": str,
                "key_learning": str, "carry_forward": bool?}
    """
    try:
        nb = Notebook.objects.get(id=notebook_id, owner=request.user)
    except Notebook.DoesNotExist:
        return JsonResponse({"error": "Notebook not found"}, status=404)

    if nb.status != Notebook.Status.ACTIVE:
        return JsonResponse({"error": "Only active notebooks can be concluded"}, status=400)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    required = ["what_went_well", "what_didnt", "what_next", "key_learning"]
    missing = [f for f in required if not data.get(f, "").strip()]
    if missing:
        return JsonResponse({"error": f"Required fields: {missing}"}, status=400)

    # Transition notebook
    nb.transition_to("concluded")

    # Create Hansei Kai (auto-creates Yokoten if carry_forward)
    hk = HanseiKai.objects.create(
        notebook=nb,
        what_went_well=data["what_went_well"],
        what_didnt=data["what_didnt"],
        what_next=data["what_next"],
        key_learning=data["key_learning"],
        carry_forward=data.get("carry_forward", False),
        created_by=request.user,
    )

    result = _serialize_notebook(nb)
    result["hansei_kai"] = {
        "what_went_well": hk.what_went_well,
        "what_didnt": hk.what_didnt,
        "what_next": hk.what_next,
        "key_learning": hk.key_learning,
        "carry_forward": hk.carry_forward,
    }

    # Include yokoten if created
    yokoten = Yokoten.objects.filter(source_notebook=nb).first()
    if yokoten:
        result["yokoten"] = _serialize_yokoten(yokoten)

    logger.info(f"Notebook {nb.id} concluded with Hansei Kai (carry_forward={hk.carry_forward})")
    return JsonResponse(result)


# ---------------------------------------------------------------------------
# Yokoten
# ---------------------------------------------------------------------------


@require_http_methods(["GET"])
@gated_paid
def list_yokoten(request):
    """List yokoten visible to the current user (NB-001 §2.7.1 tier gating)."""
    # Pro: own notebooks only. Team/Enterprise: broader visibility.
    yokoten = Yokoten.objects.filter(source_notebook__owner=request.user).order_by("-created_at")
    return JsonResponse({"yokoten": [_serialize_yokoten(y) for y in yokoten]})


@require_http_methods(["POST"])
@gated_paid
def adopt_yokoten(request, yokoten_id):
    """
    Adopt a yokoten learning into a target notebook.

    POST body: {"target_notebook_id": uuid}
    """
    try:
        yokoten = Yokoten.objects.get(id=yokoten_id)
    except Yokoten.DoesNotExist:
        return JsonResponse({"error": "Yokoten not found"}, status=404)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    target_id = data.get("target_notebook_id")
    if not target_id:
        return JsonResponse({"error": "target_notebook_id is required"}, status=400)

    try:
        target_nb = Notebook.objects.get(id=target_id, owner=request.user)
    except Notebook.DoesNotExist:
        return JsonResponse({"error": "Target notebook not found"}, status=404)

    adoption, created = YokotenAdoption.objects.get_or_create(
        yokoten=yokoten,
        target_notebook=target_nb,
        defaults={"adopted_by": request.user},
    )

    return JsonResponse(
        {
            "id": str(adoption.id),
            "yokoten_id": str(yokoten.id),
            "target_notebook_id": str(target_nb.id),
            "created": created,
        },
        status=201 if created else 200,
    )

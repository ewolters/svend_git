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

from django.db.models import Q
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
# Digest trigger — submit async Qwen analysis on learning changes
# ---------------------------------------------------------------------------


def _trigger_digest(user):
    """Submit a front page digest generation task via Tempora."""
    import uuid as _uuid

    try:
        from syn.sched.models import CognitiveTask

        SYSTEM_TENANT = _uuid.UUID("00000000-0000-0000-0000-000000000000")

        # Avoid duplicate tasks — check if one is already pending/running
        existing = CognitiveTask.objects.filter(
            task_name="api.generate_front_page_digest",
            state__in=["PENDING", "SCHEDULED", "RUNNING"],
            payload__user_id=user.id,
        ).exists()
        if existing:
            return

        CognitiveTask.objects.create(
            task_name="api.generate_front_page_digest",
            payload={"user_id": user.id},
            state="PENDING",
            queue="batch",
            priority=0,  # LOW
            tenant_id=SYSTEM_TENANT,
        )
        logger.info("Front page digest task submitted for user %s", user.email)
    except Exception:
        logger.warning("Failed to submit digest task for user %s", user.email, exc_info=True)


# ---------------------------------------------------------------------------
# Verdict narrative generation — NB-001 §2.2.2
# ---------------------------------------------------------------------------


def generate_verdict_narrative(trial):
    """
    Auto-generate a verdict narrative from trial data and linked page statistics.

    NB-001 §2.2.2: Builds the full statistical picture — frequentist significance,
    effect magnitude, capability indices, Bayesian evidence, and sample adequacy.
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
    improving = True
    try:
        goal_target = float(nb.project.goal_target) if nb.project.goal_target else None
        goal_baseline = float(nb.project.goal_baseline) if nb.project.goal_baseline else None
        if goal_target is not None and goal_baseline is not None:
            improving = (goal_target > goal_baseline and delta > 0) or (goal_target < goal_baseline and delta < 0)
    except (ValueError, TypeError, AttributeError):
        improving = delta < 0 if "rate" in metric.lower() or "defect" in metric.lower() else delta > 0

    # Collect statistics from ALL linked pages (before + after + supporting)
    all_pages = NotebookPage.objects.filter(trial=trial).order_by("-created_at")
    after_pages = [p for p in all_pages if p.trial_role == "after"]
    before_pages = [p for p in all_pages if p.trial_role == "before"]

    # Extract from after page (primary evidence source)
    p_value = None
    effect_mag = None
    ci_low = ci_high = None
    cpk = cp = ppk = pp = None
    sigma_level = None
    sample_size = None
    yield_pct = None
    bf10 = None
    posterior_prob = None
    cohens_d = None
    evidence_grade = None

    def _extract_stats(page):
        """Extract all available statistics from a page's outputs."""
        outputs = page.outputs or {}
        stats = outputs.get("statistics", {})
        if not isinstance(stats, dict):
            stats = {}
        # Also check top-level (some analyses put stats at root)
        merged = {**stats}
        for k in (
            "p_value",
            "cpk",
            "cp",
            "ppk",
            "pp",
            "sigma_level",
            "n",
            "sample_size",
            "yield_pct",
            "bf10",
            "bayes_factor",
            "posterior_probability",
            "cohens_d",
            "effect_size_d",
            "evidence_grade",
        ):
            if k in outputs and k not in merged:
                merged[k] = outputs[k]
        return merged, outputs

    for page in after_pages[:1]:
        from agents_api.analysis.standardize import _classify_effect, _extract_p_value

        outputs = page.outputs or {}
        p_value = _extract_p_value(outputs)
        effect_mag = _classify_effect(outputs)
        stats, raw = _extract_stats(page)

        ci = stats.get("confidence_interval") or stats.get("ci")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            ci_low, ci_high = ci

        cpk = stats.get("cpk")
        cp = stats.get("cp")
        ppk = stats.get("ppk")
        pp = stats.get("pp")
        sigma_level = stats.get("sigma_level")
        sample_size = stats.get("n") or stats.get("sample_size")
        yield_pct = stats.get("yield_pct")
        bf10 = stats.get("bf10") or stats.get("bayes_factor")
        posterior_prob = stats.get("posterior_probability")
        cohens_d = stats.get("cohens_d") or stats.get("effect_size_d")
        evidence_grade = raw.get("evidence_grade")

        # Check bayesian_shadow for BF10 if not in stats
        bs = raw.get("bayesian_shadow")
        if isinstance(bs, dict):
            if bf10 is None:
                bf10 = bs.get("bf10") or bs.get("bayes_factor")
            if posterior_prob is None:
                posterior_prob = bs.get("posterior_probability") or bs.get("posterior_prob")

    # Also check before pages for comparison capability values
    cpk_before = None
    for page in before_pages[:1]:
        stats_b, _ = _extract_stats(page)
        cpk_before = stats_b.get("cpk")

    # ── Build narrative ──────────────────────────────────────────────────

    parts = []

    # 1. Delta description (always present)
    abs_delta = abs(delta) if delta else 0
    direction = "decreased" if delta < 0 else "increased"
    delta_str = f"{metric.replace('_', ' ')} {direction} {abs_delta:.1f}{unit}"
    if delta_pct is not None:
        delta_str += f" ({abs(delta_pct):.1f}%)"
    before_after = f"{trial.before_value:.1f}{unit} → {trial.after_value:.1f}{unit}"
    parts.append(f"{delta_str} ({before_after}).")

    # 2. Frequentist significance
    if p_value is not None:
        if p_value < 0.001:
            parts.append("Highly significant (p<0.001).")
        elif p_value < 0.01:
            parts.append(f"Significant (p={p_value:.3f}).")
        elif p_value < 0.05:
            parts.append(f"Statistically significant (p={p_value:.3f}).")
        else:
            parts.append(f"Not statistically significant (p={p_value:.2f}).")

    # 3. Effect size (practical significance)
    if cohens_d is not None:
        import math

        if math.isfinite(cohens_d):
            parts.append(f"Cohen's d={abs(cohens_d):.2f} ({effect_mag or 'see below'} effect).")
    elif effect_mag:
        parts.append(f"Effect size: {effect_mag}.")

    # 4. Confidence interval
    if ci_low is not None and ci_high is not None:
        parts.append(f"95% CI: [{ci_low:.2f}, {ci_high:.2f}].")

    # 5. Capability indices (the precision manufacturing story)
    cap_parts = []
    if cpk is not None:
        label = "capable" if cpk >= 1.33 else "marginal" if cpk >= 1.0 else "not capable"
        cap_parts.append(f"Cpk={cpk:.2f} ({label})")
    if cp is not None:
        cap_parts.append(f"Cp={cp:.2f}")
    if ppk is not None:
        cap_parts.append(f"Ppk={ppk:.2f}")
    if pp is not None:
        cap_parts.append(f"Pp={pp:.2f}")
    if cap_parts:
        cap_str = ", ".join(cap_parts)
        if cpk_before is not None and cpk is not None:
            cap_str += f" (from Cpk={cpk_before:.2f})"
        parts.append(f"Process capability: {cap_str}.")

    # 6. Sigma level and yield
    if sigma_level is not None:
        sigma_str = f"Sigma level: {sigma_level:.1f}"
        if yield_pct is not None:
            sigma_str += f" ({yield_pct:.2f}% yield)"
        parts.append(f"{sigma_str}.")

    # 7. Bayesian evidence (the insurance policy)
    if bf10 is not None:
        import math

        if math.isfinite(bf10):
            if bf10 > 100:
                strength = "decisive"
            elif bf10 > 30:
                strength = "very strong"
            elif bf10 > 10:
                strength = "strong"
            elif bf10 > 3:
                strength = "moderate"
            elif bf10 > 1:
                strength = "anecdotal"
            else:
                strength = "favors null"
            parts.append(f"Bayes factor: BF10={bf10:.1f} ({strength} evidence).")
    if posterior_prob is not None:
        parts.append(f"Posterior probability: {posterior_prob:.1%}.")

    # 8. Evidence grade (if available from standardize)
    if evidence_grade:
        parts.append(f"Evidence grade: {evidence_grade}.")

    # 9. Sample size (context for reliability of estimates)
    if sample_size is not None:
        parts.append(f"n={sample_size}.")

    # 10. Verdict summary
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
        "project_title": nb.project.title if nb.project else None,
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
        "adopted": t.is_adopted,
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
        all_pages = list(nb.pages.all())
        result["front_matter"] = [_serialize_page(p) for p in all_pages if p.trial_role == "front_matter"]
        result["pages"] = [_serialize_page(p) for p in all_pages if p.trial_role != "front_matter"]
        # Include hansei kai if concluded
        try:
            hk = nb.hansei_kai
            result["hansei_kai"] = {
                "what_went_well": hk.what_went_well,
                "what_didnt": hk.what_didnt,
                "what_next": hk.what_next,
                "key_learning": hk.key_learning,
                "carry_forward": hk.is_carry_forward,
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

    POST body: {"verdict": str, "adopted": bool? (maps to is_adopted), "verdict_narrative": str?}
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

    trial.complete(verdict=verdict, is_adopted=data.get("adopted", False))

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

    logger.info(f"Trial {trial.sequence} completed: {verdict}, is_adopted={trial.is_adopted}")
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
# Pull Whiteboard
# ---------------------------------------------------------------------------


@require_http_methods(["GET", "POST"])
@gated_paid
def pull_tool(request, notebook_id):
    """
    GET  — List tool outputs available to pull into this notebook.
    POST — Pull a tool output into the notebook as a frozen page.

    GET params: source_type=whiteboard|fmea|rca|dsw (required)
    POST body: {"source_type": str, "source_id": uuid,
                "role": "before"|"after"|"supporting", "trial_id": uuid?}
    """
    try:
        nb = Notebook.objects.get(id=notebook_id, owner=request.user)
    except Notebook.DoesNotExist:
        return JsonResponse({"error": "Notebook not found"}, status=404)

    if request.method == "GET":
        source_type = request.GET.get("source_type", "")
        return _list_pullable(source_type, nb, request.user)

    # POST — pull a tool output into the notebook
    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    source_type = data.get("source_type", "")
    source_id = data.get("source_id")
    role = data.get("role", "supporting")
    if not source_type or not source_id:
        return JsonResponse({"error": "source_type and source_id are required"}, status=400)

    valid_roles = ["before", "after", "supporting"]
    if role not in valid_roles:
        return JsonResponse({"error": f"role must be one of: {valid_roles}"}, status=400)

    # Optional trial linkage
    trial = None
    trial_id = data.get("trial_id")
    if trial_id:
        try:
            trial = Trial.objects.get(id=trial_id, notebook=nb)
        except Trial.DoesNotExist:
            return JsonResponse({"error": "Trial not found in this notebook"}, status=404)

    # Dispatch to source-specific pull logic
    if source_type == "whiteboard":
        page = _pull_whiteboard(nb, source_id, role, trial, request.user)
    elif source_type == "fmea":
        page = _pull_fmea(nb, source_id, role, trial, request.user)
    elif source_type == "rca":
        page = _pull_rca(nb, source_id, role, trial, request.user)
    elif source_type == "dsw":
        page = _pull_dsw(nb, source_id, role, trial, request.user)
    elif source_type == "ishikawa":
        page = _pull_ishikawa(nb, source_id, role, trial, request.user)
    elif source_type == "ce_matrix":
        page = _pull_ce_matrix(nb, source_id, role, trial, request.user)
    elif source_type == "doe":
        return _pull_doe(nb, source_id, data.get("mode", "loose"), request.user)
    else:
        return JsonResponse({"error": f"Unknown source_type: {source_type}"}, status=400)

    if isinstance(page, JsonResponse):
        return page  # Error response

    logger.info(f"{source_type} pulled into notebook {nb.id} as {role}")
    return JsonResponse(_serialize_page(page), status=201)


def _list_pullable(source_type, nb, user):
    """List tool outputs available to pull into a notebook."""
    from core.models import ExperimentDesign

    from .models import FMEA, Board, CEMatrix, DSWResult, IshikawaDiagram, RCASession
    from .permissions import qms_queryset

    if source_type == "whiteboard":
        boards = Board.objects.filter(Q(project=nb.project) | Q(owner=user, project__isnull=True)).order_by(
            "-updated_at"
        )[:20]
        return JsonResponse(
            {
                "items": [
                    {
                        "id": str(b.id),
                        "title": b.name,
                        "subtitle": f"{len(b.elements or [])} elements",
                        "updated_at": b.updated_at.isoformat(),
                    }
                    for b in boards
                ]
            }
        )

    elif source_type == "fmea":
        fmeas = qms_queryset(FMEA, user)[0].order_by("-updated_at")[:20]
        return JsonResponse(
            {
                "items": [
                    {
                        "id": str(f.id),
                        "title": f.title,
                        "subtitle": f"{f.fmea_type} — {f.rows.count()} rows",
                        "updated_at": f.updated_at.isoformat(),
                    }
                    for f in fmeas
                ]
            }
        )

    elif source_type == "rca":
        rcas = qms_queryset(RCASession, user)[0].order_by("-updated_at")[:20]
        return JsonResponse(
            {
                "items": [
                    {
                        "id": str(r.id),
                        "title": r.title or r.event[:60],
                        "subtitle": r.status,
                        "updated_at": r.updated_at.isoformat(),
                    }
                    for r in rcas
                ]
            }
        )

    elif source_type == "dsw":
        results = DSWResult.objects.filter(user=user).order_by("-created_at")[:20]
        return JsonResponse(
            {
                "items": [
                    {
                        "id": str(r.id),
                        "title": r.title,
                        "subtitle": r.result_type,
                        "updated_at": r.created_at.isoformat(),
                    }
                    for r in results
                ]
            }
        )

    elif source_type == "ishikawa":
        diagrams = qms_queryset(IshikawaDiagram, user)[0].order_by("-updated_at")[:20]
        return JsonResponse(
            {
                "items": [
                    {
                        "id": str(d.id),
                        "title": d.title or d.effect[:60],
                        "subtitle": f"{len(d.branches or [])} categories",
                        "updated_at": d.updated_at.isoformat(),
                    }
                    for d in diagrams
                ]
            }
        )

    elif source_type == "ce_matrix":
        matrices = qms_queryset(CEMatrix, user)[0].order_by("-updated_at")[:20]
        return JsonResponse(
            {
                "items": [
                    {
                        "id": str(m.id),
                        "title": m.title or "Untitled",
                        "subtitle": f"{len(m.inputs or [])} inputs × {len(m.outputs or [])} outputs",
                        "updated_at": m.updated_at.isoformat(),
                    }
                    for m in matrices
                ]
            }
        )

    elif source_type == "doe":
        designs = ExperimentDesign.objects.filter(project__user=user).order_by("-created_at")[:20]
        return JsonResponse(
            {
                "items": [
                    {
                        "id": str(d.id),
                        "title": d.name,
                        "subtitle": f"{d.get_design_type_display()} — {d.num_runs} runs",
                        "updated_at": d.updated_at.isoformat(),
                    }
                    for d in designs
                ]
            }
        )

    return JsonResponse({"error": f"Unknown source_type: {source_type}"}, status=400)


def _pull_whiteboard(nb, source_id, role, trial, user):
    """Snapshot a whiteboard into a notebook page."""
    from .models import Board
    from .whiteboard_views import _generate_svg

    try:
        board = Board.objects.get(id=source_id)
    except Board.DoesNotExist:
        return JsonResponse({"error": "Board not found"}, status=404)

    svg_content, width, height = _generate_svg(board)

    # Extract text content
    text_parts = []
    for el in board.elements or []:
        text = el.get("text") or el.get("title") or el.get("effect")
        if text:
            text_parts.append(f"[{el.get('type', 'item')}] {text}")

    # Extract causal connections
    causal_lines = []
    for conn in board.connections or []:
        if conn.get("type") == "causal":
            elements = board.elements or []
            from_el = next((e for e in elements if e.get("id") == conn.get("from", {}).get("elementId")), None)
            to_el = next((e for e in elements if e.get("id") == conn.get("to", {}).get("elementId")), None)
            if from_el and to_el:
                from_text = from_el.get("text") or from_el.get("title") or "?"
                to_text = to_el.get("text") or to_el.get("title") or "?"
                causal_lines.append(f"If {from_text} → Then {to_text}")

    narrative = f"**Whiteboard:** {board.name}\n\n"
    if text_parts:
        narrative += "**Elements:**\n" + "\n".join(f"- {t}" for t in text_parts[:20]) + "\n"
    if causal_lines:
        narrative += "\n**Causal Relationships:**\n" + "\n".join(f"- {c}" for c in causal_lines[:10]) + "\n"

    return NotebookPage.objects.create(
        notebook=nb,
        page_type="analysis",
        title=f"Whiteboard: {board.name}",
        source_tool="whiteboard",
        inputs={"board_id": str(board.id), "room_code": board.room_code},
        outputs={
            "elements": board.elements or [],
            "connections": board.connections or [],
            "svg_width": width,
            "svg_height": height,
        },
        rendered_html=svg_content or "",
        narrative=narrative,
        trial=trial,
        trial_role=role,
        created_by=user,
    )


def _pull_fmea(nb, source_id, role, trial, user):
    """Snapshot an FMEA into a notebook page."""
    from .models import FMEA
    from .permissions import qms_queryset

    try:
        fmea = qms_queryset(FMEA, user)[0].get(id=source_id)
    except FMEA.DoesNotExist:
        return JsonResponse({"error": "FMEA not found"}, status=404)

    rows = list(fmea.rows.order_by("-rpn"))
    narrative = f"**FMEA:** {fmea.title} ({fmea.fmea_type})\n\n"
    if rows:
        narrative += "| Failure Mode | Severity | Occurrence | Detection | RPN |\n"
        narrative += "|---|---|---|---|---|\n"
        for r in rows[:20]:
            narrative += f"| {r.failure_mode} | {r.severity} | {r.occurrence} | {r.detection} | {r.rpn} |\n"
        top_rpn = rows[0]
        narrative += f"\n**Highest risk:** {top_rpn.failure_mode} (RPN={top_rpn.rpn})"
        if top_rpn.cause:
            narrative += f"\n**Cause:** {top_rpn.cause}"
        if top_rpn.recommended_action:
            narrative += f"\n**Recommended action:** {top_rpn.recommended_action}"

    return NotebookPage.objects.create(
        notebook=nb,
        page_type="analysis",
        title=f"FMEA: {fmea.title}",
        source_tool="fmea",
        inputs={"fmea_id": str(fmea.id), "fmea_type": fmea.fmea_type, "row_count": len(rows)},
        outputs={"rows": [r.to_dict() for r in rows[:30]]},
        narrative=narrative,
        trial=trial,
        trial_role=role,
        created_by=user,
    )


def _pull_rca(nb, source_id, role, trial, user):
    """Snapshot an RCA session into a notebook page."""
    from .models import RCASession
    from .permissions import qms_queryset

    try:
        rca = qms_queryset(RCASession, user)[0].get(id=source_id)
    except RCASession.DoesNotExist:
        return JsonResponse({"error": "RCA session not found"}, status=404)

    narrative = f"**Root Cause Analysis:** {rca.title or rca.event[:80]}\n\n"
    narrative += f"**Event:** {rca.event}\n\n"
    if rca.chain:
        narrative += "**Causal Chain:**\n"
        for i, step in enumerate(rca.chain):
            narrative += f"{i + 1}. {step.get('claim', '')}\n"
    if rca.root_cause:
        narrative += f"\n**Root Cause:** {rca.root_cause}\n"
    if rca.countermeasure:
        narrative += f"\n**Countermeasure:** {rca.countermeasure}\n"
    if rca.evaluation:
        narrative += f"\n**Evaluation:** {rca.evaluation}\n"

    return NotebookPage.objects.create(
        notebook=nb,
        page_type="analysis",
        title=f"RCA: {rca.title or rca.event[:60]}",
        source_tool="rca",
        inputs={"rca_id": str(rca.id), "status": rca.status},
        outputs={"chain": rca.chain or [], "root_cause": rca.root_cause, "countermeasure": rca.countermeasure},
        narrative=narrative,
        trial=trial,
        trial_role=role,
        created_by=user,
    )


def _pull_dsw(nb, source_id, role, trial, user):
    """Snapshot a DSW analysis result into a notebook page."""
    from .models import DSWResult

    try:
        result = DSWResult.objects.get(id=source_id, user=user)
    except DSWResult.DoesNotExist:
        return JsonResponse({"error": "DSW result not found"}, status=404)

    import json as json_module

    try:
        result_data = json_module.loads(result.data) if isinstance(result.data, str) else result.data
    except (json_module.JSONDecodeError, TypeError):
        result_data = {}

    narrative = f"**Analysis:** {result.title}\n\n"
    summary = result_data.get("summary") or result_data.get("guide_observation") or ""
    if summary:
        import re

        clean = re.sub(r"<<COLOR:\w+>>|<</COLOR>>", "", summary)
        narrative += f"{clean}\n"

    stats = result_data.get("statistics", {})
    if isinstance(stats, dict) and stats:
        narrative += "\n**Key Statistics:**\n"
        for key, val in list(stats.items())[:10]:
            if isinstance(val, float):
                narrative += f"- {key}: {val:.4f}\n"
            else:
                narrative += f"- {key}: {val}\n"

    return NotebookPage.objects.create(
        notebook=nb,
        page_type="analysis",
        title=f"DSW: {result.title}",
        source_tool="dsw",
        inputs={"dsw_id": str(result.id), "analysis_id": result_data.get("analysis_id", "")},
        outputs={"statistics": stats, "analysis_id": result_data.get("analysis_id", "")},
        narrative=narrative,
        trial=trial,
        trial_role=role,
        created_by=user,
    )


def _pull_ishikawa(nb, source_id, role, trial, user):
    """Snapshot an Ishikawa (fishbone) diagram into a notebook page."""
    from .models import IshikawaDiagram
    from .permissions import qms_queryset

    try:
        diagram = qms_queryset(IshikawaDiagram, user)[0].get(id=source_id)
    except IshikawaDiagram.DoesNotExist:
        return JsonResponse({"error": "Ishikawa diagram not found"}, status=404)

    narrative = f"**Ishikawa Diagram:** {diagram.title or 'Untitled'}\n"
    if diagram.effect:
        narrative += f"**Effect:** {diagram.effect}\n\n"

    for branch in diagram.branches or []:
        category = branch.get("category", "Unknown")
        causes = branch.get("causes", [])
        if causes:
            narrative += f"**{category}:**\n"
            for cause in causes:
                narrative += f"- {cause.get('text', '')}\n"
                for child in cause.get("children", []):
                    narrative += f"  - {child.get('text', '')}\n"

    return NotebookPage.objects.create(
        notebook=nb,
        page_type="analysis",
        title=f"Ishikawa: {diagram.title or diagram.effect[:60]}",
        source_tool="ishikawa",
        inputs={"ishikawa_id": str(diagram.id), "effect": diagram.effect},
        outputs={"branches": diagram.branches or []},
        narrative=narrative,
        trial=trial,
        trial_role=role,
        created_by=user,
    )


def _pull_ce_matrix(nb, source_id, role, trial, user):
    """Snapshot a C&E matrix into a notebook page."""
    from .models import CEMatrix
    from .permissions import qms_queryset

    try:
        matrix = qms_queryset(CEMatrix, user)[0].get(id=source_id)
    except CEMatrix.DoesNotExist:
        return JsonResponse({"error": "C&E matrix not found"}, status=404)

    totals = matrix.compute_totals()

    narrative = f"**C&E Matrix:** {matrix.title or 'Untitled'}\n\n"

    if matrix.outputs:
        narrative += "**Outputs (Y's):**\n"
        for out in matrix.outputs:
            narrative += f"- {out.get('name', '')} (weight: {out.get('weight', 1)})\n"

    if totals:
        narrative += "\n**Input Rankings (weighted total):**\n"
        narrative += "| Rank | Input | Score |\n|---|---|---|\n"
        for rank, t in enumerate(totals[:15], 1):
            narrative += f"| {rank} | {t['input_name']} | {t['total']:.0f} |\n"

    return NotebookPage.objects.create(
        notebook=nb,
        page_type="analysis",
        title=f"C&E Matrix: {matrix.title or 'Untitled'}",
        source_tool="ce_matrix",
        inputs={
            "ce_matrix_id": str(matrix.id),
            "input_count": len(matrix.inputs or []),
            "output_count": len(matrix.outputs or []),
        },
        outputs={"totals": totals, "outputs": matrix.outputs or [], "inputs": matrix.inputs or []},
        narrative=narrative,
        trial=trial,
        trial_role=role,
        created_by=user,
    )


def _pull_doe(nb, source_id, mode, user):
    """Pull a DOE design into a notebook. Returns JsonResponse directly.

    mode="loose" — design as front matter page only.
    mode="strict" — front matter page + auto-generate trials from design matrix.
    """
    from core.models import ExperimentDesign

    try:
        design = ExperimentDesign.objects.get(id=source_id, project__user=user)
    except ExperimentDesign.DoesNotExist:
        return JsonResponse({"error": "Experiment design not found"}, status=404)

    spec = design.design_spec or {}
    runs = spec.get("runs", [])
    factors = design.factors or spec.get("factors", [])

    # Build narrative
    narrative = f"**DOE Design:** {design.name}\n"
    narrative += f"**Type:** {design.get_design_type_display()}\n"
    narrative += f"**Runs:** {design.num_runs}"
    if design.num_replicates > 1:
        narrative += f" ({design.num_replicates} replicates)"
    if design.num_center_points:
        narrative += f" + {design.num_center_points} center points"
    narrative += "\n"
    if design.resolution:
        narrative += f"**Resolution:** {design.resolution}\n"

    if factors:
        narrative += "\n**Factors:**\n"
        for f in factors:
            name = f.get("name", "")
            levels = f.get("levels", [])
            units = f.get("units", "")
            narrative += f"- {name}: {levels}"
            if units:
                narrative += f" {units}"
            narrative += "\n"

    if runs:
        narrative += f"\n**Design Matrix ({len(runs)} runs):**\n"
        # Header
        factor_names = [f.get("name", f"X{i}") for i, f in enumerate(factors)]
        narrative += "| Run | " + " | ".join(factor_names) + " |\n"
        narrative += "|---" * (len(factor_names) + 1) + "|\n"
        for run in sorted(runs, key=lambda r: r.get("run_order", r.get("run_id", 0))):
            levels = run.get("levels", {})
            row = f"| {run.get('run_order', run.get('run_id', '?'))} | "
            row += " | ".join(str(levels.get(fn, "—")) for fn in factor_names)
            row += " |"
            if run.get("center_point"):
                row += " *(center)*"
            narrative += row + "\n"

    if mode == "strict":
        narrative += "\n**Mode:** Strict — trials auto-generated from design matrix.\n"
    else:
        narrative += "\n**Mode:** Loose — trials created manually.\n"

    # Create front matter page (negative sequence = sorts first)
    page = NotebookPage.objects.create(
        notebook=nb,
        page_type="analysis",
        title=f"DOE: {design.name}",
        source_tool="doe",
        inputs={
            "design_id": str(design.id),
            "design_type": design.design_type,
            "mode": mode,
            "num_runs": design.num_runs,
        },
        outputs={"factors": factors, "runs": runs, "properties": spec.get("properties", {})},
        narrative=narrative,
        trial=None,
        trial_role="front_matter",
        sequence=-10,
        created_by=user,
    )

    result = {"page": _serialize_page(page), "mode": mode, "trials_created": 0}

    # Strict mode: auto-generate trials from design matrix
    if mode == "strict" and runs:
        factor_names = [f.get("name", f"X{i}") for i, f in enumerate(factors)]
        trials_created = 0

        for run in sorted(runs, key=lambda r: r.get("run_order", r.get("run_id", 0))):
            levels = run.get("levels", {})
            run_order = run.get("run_order", run.get("run_id", 0))

            # Build trial title from factor settings
            settings_str = ", ".join(f"{fn}={levels.get(fn, '?')}" for fn in factor_names)
            title = f"Run {run_order}: {settings_str}"

            # Build description
            desc_parts = [f"**DOE Run {run_order}** (standard order: {run.get('standard_order', '?')})"]
            desc_parts.append(f"**Factor Settings:** {settings_str}")
            coded = run.get("coded", {})
            if coded:
                coded_str = ", ".join(f"{fn}={coded.get(fn, '?')}" for fn in factor_names)
                desc_parts.append(f"**Coded:** {coded_str}")
            if run.get("center_point"):
                desc_parts.append("**Center point run**")
            if run.get("replicate", 1) > 1:
                desc_parts.append(f"**Replicate:** {run['replicate']}")

            Trial.objects.create(
                notebook=nb,
                sequence=run_order,
                title=title,
                description="\n".join(desc_parts),
                created_by=user,
            )
            trials_created += 1

        result["trials_created"] = trials_created
        logger.info(f"DOE strict mode: created {trials_created} trials for notebook {nb.id}")

    logger.info(f"DOE '{design.name}' pulled into notebook {nb.id} (mode={mode})")
    return JsonResponse(result, status=201)


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

    # DOE conclude gate — warn on incomplete strict DOE runs
    doe_page = NotebookPage.objects.filter(notebook=nb, source_tool="doe", trial_role="front_matter").first()
    if doe_page and (doe_page.inputs or {}).get("mode") == "strict":
        total_runs = (doe_page.inputs or {}).get("num_runs", 0)
        completed_trials = Trial.objects.filter(notebook=nb).exclude(verdict="pending").count()
        pending_trials = Trial.objects.filter(notebook=nb, verdict="pending").count()
        if pending_trials > 0 and not data.get("confirm_incomplete_doe"):
            return JsonResponse(
                {
                    "error": "incomplete_doe",
                    "message": f"This notebook has a strict DOE design with {pending_trials} of {total_runs} runs still pending. "
                    f"Concluding now means the design matrix is incomplete — effects and interactions may not be estimable.",
                    "pending_runs": pending_trials,
                    "completed_runs": completed_trials,
                    "total_runs": total_runs,
                    "requires_confirmation": True,
                },
                status=409,
            )

    # Transition notebook
    nb.transition_to("concluded")

    # Create Hansei Kai (auto-creates Yokoten if carry_forward)
    hk = HanseiKai.objects.create(
        notebook=nb,
        what_went_well=data["what_went_well"],
        what_didnt=data["what_didnt"],
        what_next=data["what_next"],
        key_learning=data["key_learning"],
        is_carry_forward=data.get("carry_forward", False),
        created_by=request.user,
    )

    result = _serialize_notebook(nb)
    result["hansei_kai"] = {
        "what_went_well": hk.what_went_well,
        "what_didnt": hk.what_didnt,
        "what_next": hk.what_next,
        "key_learning": hk.key_learning,
        "carry_forward": hk.is_carry_forward,
    }

    # Include yokoten if created
    yokoten = Yokoten.objects.filter(source_notebook=nb).first()
    if yokoten:
        result["yokoten"] = _serialize_yokoten(yokoten)

    logger.info(f"Notebook {nb.id} concluded with Hansei Kai (is_carry_forward={hk.is_carry_forward})")
    _trigger_digest(request.user)
    return JsonResponse(result)


# ---------------------------------------------------------------------------
# Front Page — aggregated personal knowledge base
# ---------------------------------------------------------------------------


@require_http_methods(["GET"])
@gated_paid
def front_page(request):
    """
    Aggregated view of all front matter, Hansei Kai reflections, and yokoten
    across all of the user's notebooks. Personal knowledge base.
    """
    from core.models import HanseiKai

    user = request.user

    # Front matter pages across all notebooks
    front_matter = (
        NotebookPage.objects.filter(
            notebook__owner=user,
            trial_role="front_matter",
        )
        .select_related("notebook")
        .order_by("-created_at")
    )

    # Hansei Kai reflections across all notebooks
    reflections = (
        HanseiKai.objects.filter(
            notebook__owner=user,
        )
        .select_related("notebook")
        .order_by("-created_at")
    )

    # Yokoten (own + adopted)
    from core.models import YokotenAdoption

    own_yokoten = (
        Yokoten.objects.filter(
            source_notebook__owner=user,
        )
        .select_related("source_notebook")
        .order_by("-created_at")
    )

    adopted = (
        YokotenAdoption.objects.filter(
            adopted_by=user,
        )
        .select_related("yokoten", "yokoten__source_notebook", "target_notebook")
        .order_by("-adopted_at")
    )

    # Cached LLM digest (if available)
    from core.models import FrontPageDigest

    digest_data = None
    try:
        digest = FrontPageDigest.objects.get(user=user)
        digest_data = {
            "themes": digest.themes,
            "contradictions": digest.contradictions,
            "digest": digest.digest,
            "generated_at": digest.generated_at.isoformat() if digest.generated_at else None,
            "source_items": digest.source_items,
        }
    except FrontPageDigest.DoesNotExist:
        pass

    return JsonResponse(
        {
            "digest": digest_data,
            "front_matter": [
                {
                    **_serialize_page(p),
                    "notebook_title": p.notebook.title,
                }
                for p in front_matter
            ],
            "reflections": [
                {
                    "id": str(r.id),
                    "notebook_id": str(r.notebook_id),
                    "notebook_title": r.notebook.title,
                    "what_went_well": r.what_went_well,
                    "what_didnt": r.what_didnt,
                    "what_next": r.what_next,
                    "key_learning": r.key_learning,
                    "is_carry_forward": r.is_carry_forward,
                    "created_at": r.created_at.isoformat(),
                }
                for r in reflections
            ],
            "yokoten": [_serialize_yokoten(y) for y in own_yokoten],
            "adopted": [
                {
                    "id": str(a.id),
                    "yokoten": _serialize_yokoten(a.yokoten),
                    "target_notebook_id": str(a.target_notebook_id),
                    "target_notebook_title": a.target_notebook.title,
                    "adopted_at": a.adopted_at.isoformat(),
                    "outcome": a.outcome,
                    "notes": a.notes,
                }
                for a in adopted
            ],
        }
    )


# ---------------------------------------------------------------------------
# Front Matter — personal notes, anti-patterns, adopted yokoten
# ---------------------------------------------------------------------------


@require_http_methods(["POST"])
@gated_paid
def add_front_matter(request, notebook_id):
    """
    Add a front matter page to a notebook (personal note or anti-pattern).

    POST body: {"title": str, "narrative": str, "source_tool": str?}

    source_tool conventions:
    - "note" — personal observation or prior
    - "anti_pattern" — "never do this again" entry
    - "yokoten" — auto-created on yokoten adoption (not manual)
    """
    try:
        nb = Notebook.objects.get(id=notebook_id, owner=request.user)
    except Notebook.DoesNotExist:
        return JsonResponse({"error": "Notebook not found"}, status=404)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    title = data.get("title", "").strip()
    narrative = data.get("narrative", "").strip()
    if not title:
        return JsonResponse({"error": "title is required"}, status=400)

    page = NotebookPage.objects.create(
        notebook=nb,
        page_type="note",
        title=title,
        source_tool=data.get("source_tool", "note"),
        trial_role="front_matter",
        narrative=narrative,
        inputs=data.get("inputs", {}),
        outputs=data.get("outputs", {}),
        created_by=request.user,
    )

    logger.info(f"Front matter added to notebook {nb.id}: {title}")
    _trigger_digest(request.user)
    return JsonResponse(_serialize_page(page), status=201)


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

    # Create a front matter page so the learning is visible in the notebook timeline
    if created:
        source_nb = yokoten.source_notebook
        source_title = source_nb.title if source_nb else "Unknown"
        NotebookPage.objects.create(
            notebook=target_nb,
            page_type="note",
            title=f"Yokoten — {yokoten.learning[:80]}",
            source_tool="yokoten",
            trial_role="front_matter",
            inputs={
                "source_notebook": source_title,
                "source_notebook_id": str(source_nb.id) if source_nb else None,
                "applicable_to": yokoten.applicable_to,
            },
            outputs={
                "learning": yokoten.learning,
                "context": yokoten.context,
            },
            narrative=yokoten.learning,
            created_by=request.user,
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

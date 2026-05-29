#!/usr/bin/env python3
"""
Sandbox: Job Lifecycle + Synara Event Bus
==========================================

Tests the REAL Django models and event bus — not mocks.
Exercises: run_plugin() → Job → JobOutput → bus event → subscriber.

Run:
  cd ~/kjerne
  set -a && source /etc/svend/env && set +a
  python3 sandbox/job_and_bus.py

What we're testing:
  1. Job lifecycle: pending → running → completed (via run_plugin)
  2. Scratch vs persistent mode
  3. Session = Job identity (no separate session model)
  4. JobOutput creation and addressability
  5. Event bus emission on plugin completion
  6. Event bus subscription + pattern matching
  7. Bus governance (if rules exist)
  8. Job history queries (what Claude would read)
  9. Heuristic pattern: same plugin + similar inputs → suggest config

DECISIONS (confirmed):
  - Jobs are silent — no modals, no naming, invisible audit
  - session == job — no separate session model (feedback_session_equals_job.md)
  - Only observed/calculated update PCL cache
  - Scratch stays scratch until explicitly promoted
  - Event bus is flat: validate → govern → dispatch

GOTCHAS found during testing (added as we go):
  (filled in by test output)
"""

import os
import sys
import time
import traceback

# Django setup — must happen before any model imports
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "svend.settings")

# Add the kjerne root to path so Django can find everything
sys.path.insert(0, os.path.expanduser("~/kjerne"))

import django
django.setup()

# Now we can import Django models and syn/ code
from django.utils import timezone

from syn.bus import get_bus, emit, subscribe, Event
from syn.plugins.base import Plugin, PluginOutput
from syn.plugins.registry import PluginRegistry
from syn.plugins.runner import run_plugin
from job.models import Job, JobOutput


# ---------------------------------------------------------------------------
# Test Runner (same pattern as semantic_types.py)
# ---------------------------------------------------------------------------

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.gotchas = []

    def test(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            print(f"  PASS  {name}")
        else:
            self.failed += 1
            msg = f"  FAIL  {name}"
            if detail:
                msg += f" -- {detail}"
            print(msg)
            self.errors.append(name)

    def section(self, title):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    def gotcha(self, text):
        """Record a gotcha discovered during testing."""
        self.gotchas.append(text)
        print(f"\n  GOTCHA: {text}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"  RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"  FAILURES:")
            for e in self.errors:
                print(f"    - {e}")
        if self.gotchas:
            print(f"\n  GOTCHAS DISCOVERED ({len(self.gotchas)}):")
            for i, g in enumerate(self.gotchas, 1):
                print(f"    {i}. {g}")
        print(f"{'='*60}")
        return self.failed == 0


# ---------------------------------------------------------------------------
# Sandbox Plugin — minimal plugin for testing lifecycle
# ---------------------------------------------------------------------------

from pydantic import BaseModel

class SandboxInput(BaseModel):
    """Minimal input for sandbox testing."""
    values: list[float]
    usl: float = 10.0
    lsl: float = 0.0

class SandboxPlugin(Plugin):
    """Minimal plugin that computes mean + range.

    Not a real analysis — just enough to exercise the full lifecycle:
    input validation → Job creation → execute → JobOutput → bus event.
    """
    name = "sandbox_test"
    version = "0.0.1"
    description = "Sandbox test plugin — mean + range calculation"
    input_schema = SandboxInput

    def execute(self, validated_input, context):
        values = validated_input["values"]
        mean_val = sum(values) / len(values) if values else 0
        range_val = max(values) - min(values) if values else 0
        usl = validated_input["usl"]
        lsl = validated_input["lsl"]

        return [
            PluginOutput(
                key="mean",
                output_type="metric",
                value=mean_val,
                provenance="calculated",
                measure_slug="sandbox-mean",
            ),
            PluginOutput(
                key="range",
                output_type="metric",
                value=range_val,
                provenance="calculated",
            ),
            PluginOutput(
                key="summary",
                output_type="text",
                value={"text": f"Mean={mean_val:.3f}, Range={range_val:.3f}, n={len(values)}"},
                provenance="calculated",
            ),
        ]


class FailingPlugin(Plugin):
    """Plugin that always fails — tests error handling in runner."""
    name = "sandbox_failing"
    version = "0.0.1"
    description = "Always fails — tests Job failure lifecycle"
    input_schema = SandboxInput

    def execute(self, validated_input, context):
        raise ValueError("Intentional failure for sandbox testing")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests():
    t = TestRunner()

    # Fresh registry for sandbox (don't pollute global)
    registry = PluginRegistry()
    registry.register(SandboxPlugin)
    registry.register(FailingPlugin)

    # Fresh bus for sandbox
    bus = get_bus()
    # Clear any existing subscriptions from prior runs
    bus._subscriptions = []
    bus._emit_count = 0
    bus._block_count = 0

    # Track emitted events
    captured_events = []
    def capture_handler(event):
        captured_events.append(event)

    # Subscribe to plugin events
    sub_id = bus.subscribe(
        "plugin.execution.*",
        capture_handler,
        description="Sandbox test: capture all plugin events",
    )

    # ---------------------------------------------------------------
    t.section("1. Job Lifecycle — Happy Path")
    # ---------------------------------------------------------------
    # run_plugin() should: validate → create Job → execute → create JobOutputs → emit event

    input_data = {"values": [5.1, 4.9, 5.0, 5.2, 4.8], "usl": 10.0, "lsl": 0.0}
    job = run_plugin(
        "sandbox_test",
        input_data,
        actor="sandbox@test",
        is_scratch=False,
        registry=registry,
    )

    t.test("Job created", job is not None)
    t.test("Job status is completed", job.status == "completed")
    t.test("Job has plugin_name", job.plugin_name == "sandbox_test")
    t.test("Job has actor", job.actor == "sandbox@test")
    t.test("Job has started_at", job.started_at is not None)
    t.test("Job has completed_at", job.completed_at is not None)
    t.test("Job has duration_ms", job.duration_ms is not None and job.duration_ms >= 0)
    t.test("Job is NOT scratch", not job.is_scratch)
    t.test("Job inputs frozen", job.inputs == {
        "values": [5.1, 4.9, 5.0, 5.2, 4.8], "usl": 10.0, "lsl": 0.0
    })
    t.test("Job outputs_summary populated",
           job.outputs_summary == {"mean": "metric", "range": "metric", "summary": "text"})

    # ---------------------------------------------------------------
    t.section("2. JobOutput — Addressability + Provenance")
    # ---------------------------------------------------------------

    outputs = list(job.outputs.all().order_by("created_at"))
    t.test("3 JobOutputs created", len(outputs) == 3)

    mean_out = next((o for o in outputs if o.output_key == "mean"), None)
    t.test("mean output exists", mean_out is not None)
    t.test("mean is metric type", mean_out.output_type == "metric")
    t.test("mean value_numeric is 5.0", abs(mean_out.value_numeric - 5.0) < 0.001)
    t.test("mean provenance is calculated", mean_out.provenance == "calculated")
    t.test("mean has measure_slug", mean_out.measure_slug == "sandbox-mean")
    t.test("mean is UUID-addressable", mean_out.id is not None)

    range_out = next((o for o in outputs if o.output_key == "range"), None)
    t.test("range output exists", range_out is not None)
    t.test("range value_numeric is 0.4", abs(range_out.value_numeric - 0.4) < 0.001)

    summary_out = next((o for o in outputs if o.output_key == "summary"), None)
    t.test("summary output exists", summary_out is not None)
    t.test("summary is text type", summary_out.output_type == "text")
    t.test("summary value_json has text key", "text" in summary_out.value_json)

    # GOTCHA: text/chart outputs store in value_json, metric outputs in value_numeric.
    # The runner decides based on output_type. If a plugin returns output_type="metric"
    # but value is a dict, value_numeric will be None and value_json will be {}.
    # This is by design — metric outputs MUST be float-coercible.
    t.test("metric outputs use value_numeric", mean_out.value_json == {})
    t.test("text outputs use value_json", summary_out.value_numeric is None)

    # ---------------------------------------------------------------
    t.section("3. Scratch Mode")
    # ---------------------------------------------------------------

    scratch_job = run_plugin(
        "sandbox_test",
        {"values": [1.0, 2.0, 3.0]},
        actor="sandbox@test",
        is_scratch=True,
        registry=registry,
    )

    t.test("Scratch job created", scratch_job is not None)
    t.test("Scratch job is_scratch=True", scratch_job.is_scratch)
    t.test("Scratch job completed", scratch_job.status == "completed")

    # DECISION: Scratch jobs create JobOutputs just like persistent jobs.
    # The difference is semantic — scratch jobs don't write to PCL and
    # aren't included in audit trails. But the outputs exist for the
    # session (so the user can see results).
    scratch_outputs = list(scratch_job.outputs.all())
    t.test("Scratch job still has outputs", len(scratch_outputs) == 3)

    # Promotion: scratch → persistent is a deliberate action.
    # In production this would be a separate endpoint. For sandbox,
    # test that the field can be toggled.
    scratch_job.is_scratch = False
    scratch_job.save(update_fields=["is_scratch"])
    scratch_job.refresh_from_db()
    t.test("Scratch promoted to persistent", not scratch_job.is_scratch)

    t.gotcha(
        "Promotion is just a field toggle now. Production needs: "
        "(1) confirmation modal, (2) PCL write-back trigger, "
        "(3) audit log entry for the promotion event. "
        "The toggle is necessary but not sufficient."
    )

    # ---------------------------------------------------------------
    t.section("4. Job Failure Lifecycle")
    # ---------------------------------------------------------------

    try:
        failed_job = run_plugin(
            "sandbox_failing",
            {"values": [1.0, 2.0]},
            actor="sandbox@test",
            registry=registry,
        )
        t.test("Failing plugin should raise", False, "Expected exception")
    except ValueError:
        # The runner should have created a Job and marked it failed
        failed_jobs = Job.objects.filter(
            plugin_name="sandbox_failing",
            actor="sandbox@test",
            status="failed",
        ).order_by("-created_at")
        t.test("Failed job exists in DB", failed_jobs.exists())
        if failed_jobs.exists():
            fj = failed_jobs.first()
            t.test("Failed job status=failed", fj.status == "failed")
            t.test("Failed job has completed_at", fj.completed_at is not None)
            t.test("Failed job has duration_ms", fj.duration_ms is not None)
            t.test("Failed job has NO outputs", fj.outputs.count() == 0)

    # ---------------------------------------------------------------
    t.section("5. Event Bus — Plugin Completion Events")
    # ---------------------------------------------------------------

    # We subscribed to "plugin.execution.*" before running plugins.
    # Successful runs should have emitted "plugin.execution.completed".
    t.test("Events captured by subscriber", len(captured_events) > 0)

    if captured_events:
        evt = captured_events[0]
        t.test("Event name is plugin.execution.completed",
               evt.name == "plugin.execution.completed")
        t.test("Event payload has job_id", "job_id" in evt.payload)
        t.test("Event payload has plugin_name", "plugin_name" in evt.payload)
        t.test("Event payload has outputs list", "outputs" in evt.payload)
        t.test("Event has correlation_id", evt.correlation_id is not None)

    # Count: we ran sandbox_test twice (persistent + scratch).
    # sandbox_failing failed, so runner doesn't emit completion event.
    completed_events = [e for e in captured_events
                        if e.name == "plugin.execution.completed"]
    t.test("2 completion events (persistent + scratch)", len(completed_events) == 2)

    t.gotcha(
        "Failed plugin runs do NOT emit bus events. This is correct — "
        "you don't want governance/subscribers reacting to failures. "
        "But we might want a 'plugin.execution.failed' event for monitoring. "
        "Not blocking, but worth adding when we have dashboards."
    )

    # ---------------------------------------------------------------
    t.section("6. Event Bus — Pattern Matching + Governance")
    # ---------------------------------------------------------------

    # Test subscription patterns
    narrow_events = []
    bus.subscribe(
        "plugin.execution.completed",
        lambda e: narrow_events.append(e),
        description="Exact match test",
    )

    wrong_events = []
    bus.subscribe(
        "pcl.characteristic.*",
        lambda e: wrong_events.append(e),
        description="Should NOT match plugin events",
    )

    # Emit a test event
    result = emit(
        "plugin.execution.completed",
        {"job_id": "test-123", "plugin_name": "sandbox_test", "outputs": ["mean"]},
        actor="sandbox@test",
    )
    t.test("Bus emit result.allowed", result.allowed)
    t.test("Bus dispatched to subscribers", result.dispatched_to >= 1)
    t.test("Narrow subscriber got event", len(narrow_events) == 1)
    t.test("Wrong-pattern subscriber got nothing", len(wrong_events) == 0)

    # Test introspection (FLOW-3: contracts visible)
    subs = bus.list_subscriptions()
    t.test("Subscriptions are inspectable", len(subs) >= 2)
    t.test("Subscriptions have patterns",
           all("pattern" in s for s in subs))

    # Bus stats
    stats = bus.stats
    t.test("Bus tracks emit count", stats["total_emitted"] > 0)

    # ---------------------------------------------------------------
    t.section("7. Session = Job — Query Patterns for Claude")
    # ---------------------------------------------------------------
    # DECISION: session == job. No separate session model.
    # Claude reads job history to understand context.
    # Test the query patterns Claude would use.

    # "What has this user done recently?"
    recent_jobs = Job.objects.filter(
        actor="sandbox@test",
    ).order_by("-created_at")[:10]
    t.test("Can query recent jobs by actor", recent_jobs.count() >= 2)

    # "What happened with this plugin?"
    plugin_jobs = Job.objects.filter(
        plugin_name="sandbox_test",
        status="completed",
    ).order_by("-created_at")
    t.test("Can query by plugin name", plugin_jobs.count() >= 2)

    # "What were the results of the last run?"
    last_job = plugin_jobs.first()
    last_outputs = {o.output_key: o.value_numeric for o in last_job.outputs.all()
                    if o.output_type == "metric"}
    t.test("Can reconstruct last run's metrics", "mean" in last_outputs)

    # "Show me the history for this characteristic" (Dana's use case)
    # In production, this would filter by measure_slug or canvas_id.
    # For sandbox, query by measure_slug on JobOutput.
    slug_outputs = JobOutput.objects.filter(
        measure_slug="sandbox-mean",
    ).order_by("-created_at")
    t.test("Can query by measure_slug (Dana's 'last 3 for same characteristic')",
           slug_outputs.count() >= 1)

    # Build a history series from JobOutputs
    history = []
    for out in slug_outputs[:10]:
        history.append({
            "job_id": str(out.job_id),
            "value": out.value_numeric,
            "timestamp": out.created_at.isoformat() if out.created_at else None,
            "provenance": out.provenance,
        })
    t.test("History series is buildable", len(history) >= 1)
    if history:
        print(f"    Sample: measure_slug='sandbox-mean', value={history[0]['value']}")

    t.gotcha(
        "Job history queries work but are O(n) table scans without "
        "composite indexes on (measure_slug, created_at) and "
        "(actor, plugin_name, created_at). Add these before production "
        "— the queries are correct but will be slow at scale."
    )

    # ---------------------------------------------------------------
    t.section("8. Heuristic Pattern — Same Plugin, Similar Input")
    # ---------------------------------------------------------------
    # SPEC: "First time slow, second time heuristic."
    # Test: can we detect that a user is running the same analysis
    # on similar data, and suggest config from prior runs?

    # Run the same plugin with slightly different data
    job2 = run_plugin(
        "sandbox_test",
        {"values": [5.05, 4.95, 5.02, 5.18, 4.82], "usl": 10.0, "lsl": 0.0},
        actor="sandbox@test",
        registry=registry,
    )

    # Heuristic: find prior runs of same plugin by same actor.
    # NOTE: exclude scratch jobs — they're exploratory, not a reliable baseline.
    # Also exclude failed jobs — no useful outputs.
    prior_runs = Job.objects.filter(
        plugin_name="sandbox_test",
        actor="sandbox@test",
        status="completed",
        is_scratch=False,
    ).exclude(id=job2.id).order_by("-created_at")

    t.test("Can find prior runs for heuristic", prior_runs.count() >= 1)

    # Heuristic: scan prior runs for best match, not just most recent.
    # LEARNED: most-recent is wrong because promoted scratch jobs with
    # different data sizes pollute the ordering. Match on structure.
    best_match = None
    current_values = job2.inputs.get("values", [])
    for prior in prior_runs:
        prior_values = prior.inputs.get("values", [])
        same_length = len(prior_values) == len(current_values)
        same_specs = (
            prior.inputs.get("usl") == job2.inputs.get("usl") and
            prior.inputs.get("lsl") == job2.inputs.get("lsl")
        )
        if same_length and same_specs:
            best_match = prior
            break

    t.test("Heuristic detects same characteristic", best_match is not None)

    if best_match:
        # Could suggest: "You ran this before. Prior mean was X, new is Y."
        prior_mean = None
        for out in best_match.outputs.filter(output_key="mean"):
            prior_mean = out.value_numeric
        current_mean = None
        for out in job2.outputs.filter(output_key="mean"):
            current_mean = out.value_numeric

        if prior_mean is not None and current_mean is not None:
            delta = current_mean - prior_mean
            print(f"    Heuristic: prior mean={prior_mean:.3f}, "
                  f"current={current_mean:.3f}, delta={delta:+.3f}")
            t.test("Can compute delta between runs", True)

    t.gotcha(
        "Heuristic matching is naive (same length + same specs). "
        "Production needs: (1) measure_slug or part_number FK on Job, "
        "(2) PCL binding to identify 'same characteristic' structurally, "
        "(3) similarity metric beyond exact-match on spec limits. "
        "But the QUERY PATTERN works — Job history is sufficient substrate."
    )

    # ---------------------------------------------------------------
    t.section("9. Claude Context Assembly")
    # ---------------------------------------------------------------
    # What data structure does Claude receive when asked about a flowchart?
    # This tests the SHAPE of the context, not LLM calls.

    # Assemble context for "tell me about this user's recent work"
    claude_context = {
        "user": "sandbox@test",
        "recent_jobs": [],
        "active_measures": [],
    }

    for job_rec in Job.objects.filter(
        actor="sandbox@test",
        status="completed",
    ).order_by("-created_at")[:5]:
        job_ctx = {
            "job_id": str(job_rec.id),
            "plugin": job_rec.plugin_name,
            "when": job_rec.completed_at.isoformat() if job_rec.completed_at else None,
            "scratch": job_rec.is_scratch,
            "duration_ms": job_rec.duration_ms,
            "outputs": {},
        }
        for out in job_rec.outputs.all():
            if out.output_type == "metric":
                job_ctx["outputs"][out.output_key] = {
                    "value": out.value_numeric,
                    "provenance": out.provenance,
                    "measure_slug": out.measure_slug,
                }
            else:
                job_ctx["outputs"][out.output_key] = {
                    "type": out.output_type,
                    "provenance": out.provenance,
                }
        claude_context["recent_jobs"].append(job_ctx)

    t.test("Claude context has recent jobs", len(claude_context["recent_jobs"]) >= 2)
    t.test("Context includes metrics with values",
           any(
               "mean" in j["outputs"] and "value" in j["outputs"]["mean"]
               for j in claude_context["recent_jobs"]
           ))
    t.test("Context includes provenance",
           any(
               any("provenance" in v for v in j["outputs"].values())
               for j in claude_context["recent_jobs"]
           ))

    # Print sample context (what Claude would see)
    import json
    sample = claude_context["recent_jobs"][0] if claude_context["recent_jobs"] else {}
    print(f"\n  Sample Claude context (1 job):")
    print(f"    {json.dumps(sample, indent=4, default=str)[:500]}")

    t.gotcha(
        "Claude context assembly works but is N+1 queries "
        "(1 for jobs + 1 per job for outputs). Production needs: "
        "Job.objects.prefetch_related('outputs') or a denormalized view. "
        "Also: context needs flowchart topology (which devices connected "
        "to which) — not just flat job list. That requires the Connection "
        "model we haven't built yet."
    )

    # ---------------------------------------------------------------
    # Cleanup — remove sandbox data
    # ---------------------------------------------------------------
    t.section("Cleanup")
    sandbox_jobs = Job.objects.filter(actor="sandbox@test")
    output_count = JobOutput.objects.filter(job__actor="sandbox@test").count()
    job_count = sandbox_jobs.count()

    # Delete outputs first (FK constraint), then jobs
    JobOutput.objects.filter(job__actor="sandbox@test").delete()
    sandbox_jobs.delete()
    print(f"  Cleaned up {job_count} jobs and {output_count} outputs")

    bus.unsubscribe(sub_id)

    # ---------------------------------------------------------------
    ok = t.summary()
    return ok


if __name__ == "__main__":
    try:
        ok = run_tests()
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(2)

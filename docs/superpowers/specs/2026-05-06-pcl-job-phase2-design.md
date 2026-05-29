# Phase 2 Design: PCL Provenance + Job App

**Date:** 2026-05-06
**Status:** Approved
**Workstream:** svend-2.0-build
**Depends on:** Phase 0+1 (foundation ports — complete)
**Blocks:** Phase 3 (plugin system), Phase 5 (canvas engine)

---

## Context

PCL (Process Characteristics Library) is the shared data substrate for SVEND 2.0. All tools read from and write to PCL. The existing PCL app (`~/kjerne/pcl/`) has solid foundations — Measure, Datapoint, MeasureTarget models with Bayesian aggregation, confidence hierarchy, safe formula evaluation, and hash-chain immutability. 2,373 LOC, 98 tests.

Phase 2 evolves this foundation with two additions: provenance types on Datapoint (so users can distinguish real measurements from what-if results) and a new Job app (so every canvas run is recorded as an addressable audit trail).

## Decision Log

- **Job gets its own app (`~/kjerne/job/`)** — not inside PCL. A Job is "someone ran something" (platform concept). PCL is "what do we know about this process" (data concept). Different lifecycles, different dependency directions.
- **Provenance types are structural, not cosmetic.** Four values that partition the space: observed, calculated, simulated, projected. Focus-group validated 4/4. Non-negotiable for regulated industries.
- **Writing to PCL is explicit, not automatic.** A Job producing outputs does not automatically create Datapoints. The canvas configuration defines which outputs bind to which measures. The binding logic lives in the canvas layer (Phase 5), not in Job or PCL.

---

## 1. Provenance Enum on Datapoint

### Change

Add `provenance` field to `pcl.Datapoint`:

```python
PROVENANCE_TYPES = [
    ("observed", "Observed"),       # Real measurement from the physical world
    ("calculated", "Calculated"),   # Deterministic output from analysis
    ("simulated", "Simulated"),     # What-if / hypothetical output
    ("projected", "Projected"),     # Forecast / extrapolation
]

provenance = models.CharField(
    max_length=20,
    choices=PROVENANCE_TYPES,
    default="observed",
    db_index=True,
)
```

### Semantics

| Provenance | Meaning | Examples |
|---|---|---|
| observed | Real measurement from the physical world | Manual caliper reading, SPC sensor, time study, stopwatch |
| calculated | Deterministic output from analysis | Cpk from capability study, formula evaluation, DOE result |
| simulated | What-if / hypothetical output | Monte Carlo run, DOE scenario exploration |
| projected | Forecast / extrapolation | Time series forecast, trend projection |

### Rules

- `source_type` (existing field) stays — it tracks tooling origin (manual, workbench, spc, doe, import). Orthogonal to provenance.
- Default is `observed` — most writes are real measurements.
- Calculated/simulated/projected are explicitly set by the creating system.

---

## 2. Job App (`~/kjerne/job/`)

### Purpose

A Job is a record of computation — "someone ran something." Silent creation, no modals, no naming. The atomic audit unit for canvas runs.

### Job Model (extends SynaraEntity)

| Field | Type | Notes |
|---|---|---|
| canvas_id | UUIDField, nullable | Which canvas layout was used. Null for API/CLI runs. |
| status | CharField, choices | pending / running / completed / failed / cancelled |
| inputs | JSONField | Frozen snapshot of everything that went in |
| outputs_summary | JSONField | Summary for listing/search without joining outputs |
| started_at | DateTimeField, nullable | When execution began |
| completed_at | DateTimeField, nullable | When it finished |
| duration_ms | IntegerField, nullable | Wall clock time |
| actor | CharField | Who triggered it |
| is_scratch | BooleanField, default=False | Scratch work stays scratch until promoted |

**What Job is NOT:**
- Not a task. No title, description, assignee, priority, due_date.
- Not a workflow step. No status registry FKs, no templates, no schedules.
- Not plugin-aware. The canvas defines what plugins run and how they're wired. The Job records what happened.

**SynaraEntity provides:** id (UUID), correlation_id, tenant_id, created_at, updated_at, created_by, is_deleted, metadata, soft delete, event emission.

### JobOutput Model (extends SynaraImmutableLog)

| Field | Type | Notes |
|---|---|---|
| job | FK to Job, PROTECT | Parent run |
| output_key | CharField | Named result (e.g. "cpk", "histogram", "violations_list") |
| output_type | CharField, choices | metric / chart / table / text / dataset |
| value_numeric | FloatField, nullable | For PCL-writable values |
| value_json | JSONField | Full output payload |
| provenance | CharField, choices | observed / calculated / simulated / projected |
| measure_slug | SlugField, nullable | If this output wrote to PCL, which measure |

**SynaraImmutableLog provides:** id (UUID), correlation_id, entry_hash, previous_hash, created_at, actor, hash-chain integrity, immutability enforcement (no updates, no deletes).

**Key properties:**
- UUID-addressable — any canvas can reference a specific output from a specific job run.
- Immutable — hash-chained, 21 CFR Part 11 compliant.
- Job uses PROTECT on delete — job outputs persist. CASCADE destroys provenance (spec requirement).

### Session + Heuristic Support

Jobs capture actor + timestamps + canvas_id + inputs + outputs. This enables:
- **Session reconstruction:** query Jobs by actor + time to rebuild full session narratives. PROVA reads this.
- **Behavioral heuristics:** "first time slow, second time heuristic" (facilitator principle #7). Job sequences reveal patterns (e.g., capability → control chart within 10 min when Cpk < 1.33). No additional fields required.

---

## 3. Wiring — Datapoint.source_job

### Change

Add `source_job` FK to `pcl.Datapoint`:

```python
source_job = models.ForeignKey(
    "job.Job",
    on_delete=models.SET_NULL,
    null=True,
    blank=True,
    related_name="datapoints",
)
```

### Flow

1. Canvas runs -> Job created silently
2. Job produces outputs -> JobOutputs recorded (all of them, always)
3. Canvas checks its configured bindings: "output `cpk` writes to measure `bore_diameter_cpk`"
4. Only bound outputs create Datapoints in PCL with `source_job` set
5. No binding configured -> output is in the Job but PCL is untouched

Manual entry: `provenance="observed"`, `source_job=None`.
Canvas run produces Cpk: `provenance="calculated"`, `source_job=<job>`.
What-if scenario: `provenance="simulated"`, `source_job=<job>`.

### Explicit Write Rule

Writing to PCL is a deliberate act configured at canvas design time. The Job model and Datapoint model don't contain binding logic — that lives in the canvas configuration layer (Phase 5).

---

## 4. Provenance-Aware Aggregation

### Change

Gate in `pcl/service.py` `write()` function:

- `observed` or `calculated` -> update Measure cached aggregates (operational truth)
- `simulated` or `projected` -> store Datapoint but skip cache update (exploratory)

The Datapoint is still saved, still immutable, still hash-chained. It just doesn't move the Measure's cached values.

### Rationale

Without this gate, a what-if simulation run contaminates the operational truth. Maria's PPAP Cpk would shift because someone explored a scenario. The provenance types exist precisely to prevent this.

---

## 5. Event Schema Registration

Register schemas in `syn/events/` so the bus can validate payloads and governance can trigger on them.

### PCL Events

- `pcl.datapoint.created` — payload: measure_id, measure_slug, value, provenance, source_type, source_job_id
- `pcl.measure.threshold_crossed` — payload: measure_id, measure_slug, value, threshold_type (min/max), threshold_value

### Job Events

- `job.completed` — payload: job_id, canvas_id, status, duration_ms, output_count
- `job.failed` — payload: job_id, canvas_id, error_summary

### Scope

Schema definitions only. No governance rules, no downstream automation. That's wiring for later phases. The event bus infrastructure (syn/bus.py) and EventSchemaRegistry model already exist.

---

## 6. What's NOT in Scope

- Canvas models or canvas configuration (Phase 5)
- Plugin system or plugin registration (Phase 4)
- Binding logic (which outputs write to which measures) (Phase 5)
- Governance rules for PCL/Job events (later)
- VSM binding to PCL measures (later, uses existing parent_type/parent_id)
- PROVA reading Job history (later)
- cached_shift_detected on Measure (nice-to-have, not blocking)

## 7. Files to Create / Modify

### New files (Job app)

- `~/kjerne/job/__init__.py`
- `~/kjerne/job/apps.py`
- `~/kjerne/job/models.py` — Job, JobOutput
- `~/kjerne/job/migrations/0001_initial.py` — generated
- `~/kjerne/job/admin.py` — minimal registration

### Modified files (PCL)

- `~/kjerne/pcl/models.py` — add provenance to Datapoint, add source_job FK
- `~/kjerne/pcl/service.py` — provenance-aware aggregation gate in write()
- `~/kjerne/pcl/migrations/0002_*.py` — generated
- `~/kjerne/pcl/tests/test_models.py` — provenance tests
- `~/kjerne/pcl/tests/test_service.py` — aggregation gate tests

### Modified files (Infrastructure)

- `~/kjerne/svend/settings.py` — add `job` to INSTALLED_APPS
- Event schema registration (data migration or management command)

### Existing files NOT modified

- `pcl/confidence.py` — no changes needed
- `pcl/aggregation.py` — no changes needed (gate is in service.py, not aggregation)
- `pcl/formulas.py` — no changes needed
- `pcl/views.py` — no changes needed (provenance passes through existing write endpoint)

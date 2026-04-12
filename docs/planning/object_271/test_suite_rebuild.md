# agents_api Extraction — Test Suite Rebuild Plan

**Status:** DRAFT — planning artifact under CR `5bf7354c-3de5-4624-b505-a94a5b6ce0ea`
**Date:** 2026-04-10
**Author:** Claude (Systems Engineer role per Object 271)
**Inputs:**
- `docs/planning/object_271/extraction_sequence.md` — CR-by-CR execution plan (45+ CRs)
- `docs/planning/object_271/extraction_gap_analysis.md` — per-CR test specifications
- `docs/planning/object_271/phase_0_plan.md` — Phase 0 detailed test descriptions
- `docs/standards/TST-001.md` — Testing Patterns standard
- `~/.claude/projects/-home-eric/memory/feedback_build_then_standardize.md` — build first, comprehensive tests from stable state

**Purpose:** Define the testing strategy for the entire agents_api extraction. This plan covers: (1) safety-net tests written per-CR during extraction, (2) route-agnostic dual-path testing during the parallel period, (3) legacy path cleanup after cutover, and (4) the eventual comprehensive test suite rebuild from stable state.

**What this plan is NOT:**
- Not the comprehensive TST-001 test suite rebuild. That comes after the extraction completes and the codebase is stable. Per `feedback_build_then_standardize.md`: "Build the working thing first; write the standard from stable state."
- Not a test-driven development mandate. Tests serve the extraction — they verify that behavior survives relocation and rebuild, not that every edge case is covered.

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [Test categories](#categories)
3. [Per-CR safety-net tests — specifications](#per-cr)
4. [Route-agnostic dual-path pattern](#dual-path)
5. [Phase 3 cutover test plan](#cutover)
6. [Phase 4 Site move test plan](#site-move)
7. [Post-extraction comprehensive rebuild](#post-extraction)
8. [Test infrastructure requirements](#infrastructure)
9. [Existing test inventory](#existing-tests)

---

<a id="philosophy"></a>
## 1. Philosophy

### 1.1 The extraction test contract

Each extraction CR writes **~5 behavior tests** (range: 3-10 depending on complexity). These tests verify:
- The extracted code works at its new location
- Cross-app imports resolve after FK/import path changes
- State machines preserve their transition rules
- The 10-key result contract (where applicable) populates correctly

They do NOT verify:
- Every edge case or error path
- CSS rendering or template correctness
- Performance characteristics
- Exhaustive input validation

### 1.2 Why ~5 per CR, not more

Per architecture §9.A decision #9: "Minimal safety net (~5 behavior tests per extraction). Full TST-001 rebuild is separate later effort."

The reasoning: each CR touches 3-15 files across 2-3 apps. 5 behavior tests that exercise the critical paths catch 90% of relocation breakage (broken imports, wrong FK targets, missing migrations) in ~30 minutes of test writing. Going to 20+ tests per CR would triple CR throughput time for marginal breakage detection gain. The extraction is ~45 CRs — at 5 tests each, that's ~225 new behavior tests. At 20 each, that's 900+ tests written before the codebase is even stable, violating the build-first principle.

### 1.3 TST-001 §10.6 compliance

Existence/sweep tests are PROHIBITED. Every test must exercise real behavior. Tests like "can import X" or "endpoint returns 200" without checking response content are sweep tests and will be flagged by compliance. The ~5 tests per CR are all behavior tests — they assert on response content, model state, or cross-app query results.

---

<a id="categories"></a>
## 2. Test categories

### Category A: Safety-net tests (written per-CR during extraction)

~225 tests across 45 CRs. Written by Claude as part of each CR. Live alongside the extracted code in each app's `tests/` directory.

**Location pattern:** `<app>/tests/test_extraction_safety.py`

Each app gets a single extraction safety test file. As CRs land (1A relocate, then 1B rebuild), tests accumulate in this file. After cutover, the file is the seed for the comprehensive test suite.

### Category B: Dual-path tests (written at Phase 1A/2A, deleted at Phase 3)

Route-agnostic tests that exercise both `/app/<thing>/` (legacy) and `/app/demo/<thing>/` (new) paths during the parallel period. Assert equivalent behavior. These are **temporary** — deleted at Phase 3 cutover when legacy paths die.

**Location pattern:** `<app>/tests/test_dual_path.py`

### Category C: Regression tests (existing, maintained throughout)

Existing behavioral test files that must keep passing throughout the extraction. These are NOT rewritten — they are maintained as-is and updated only when imports or paths change.

Key files:
- `agents_api/tests/test_dsw_views_behavioral.py` — DSW behavior
- `agents_api/tests/iso_tests.py` (351 tests) — ISO clause regression
- `whiteboard_tests.py` — whiteboard behavior (83 tests)
- `loop/tests/` — loop behavior
- `graph/tests/` — graph behavior
- `safety/tests/` — safety behavior

### Category D: Sweep tests (deleted in Phase 0)

Per CR-0.6 in the Phase 0 plan and TST-001 §10.6: delete `test_endpoint_smoke.py` (712 LOC), `test_t1_deep.py` (1,942 LOC), `test_t2_views_smoke.py` (338 LOC), `test_*_coverage.py` family (8 files). These are existence/sweep tests that violate TST-001 §10.6.

---

<a id="per-cr"></a>
## 3. Per-CR safety-net tests — specifications

Every CR from `extraction_sequence.md` has a test count. This section consolidates them into a reference table with test descriptions.

### 3.1 Phase 0 tests

| CR | Tests | Description | File |
|----|-------|-------------|------|
| CR-0.1 | 5 | CSV upload → response shape; chart render → ChartSpec; standardize → evidence_grade; read_csv_safe → graceful error; run_analysis → p_value key | `agents_api/tests/test_phase0_dsw_move.py` |
| CR-0.2 | 7 | One per SPC type: xbar-R, xbar-S, IMR, c, p, capability, gage R&R — each asserts 10-key contract | `agents_api/tests/test_phase0_forgespc.py` |
| CR-0.3 | 10 | One per analysis family: parametric, nonparametric, regression, ANOVA, posthoc, advanced, exploratory, Bayesian, reliability, ML | `agents_api/tests/test_phase0_forgestat.py` |
| CR-0.4 | 5 | BeliefEngine posterior; DSL round-trip; synara API schema; graph adapter propagation; graph evidence creation | `agents_api/tests/test_phase0_forgesia.py` |
| CR-0.5 | 4 | Control chart spec; capability histogram; residuals plot; generic bar/line | `agents_api/tests/test_phase0_forgeviz.py` |
| CR-0.6 | 5 | Commitment from FMEA; commitment from RCA; commitment from A3; get_shared_llm import; /api/agents/ returns 404 | `agents_api/tests/test_phase0_dead_code.py` |
| CR-0.7 | 3 | qms_can_edit permission; qms_queryset tenant filter; get_tenant returns tenant | `qms_core/tests/test_permissions.py` |
| CR-0.8 | 0 | Uses existing `iso_tests.py` (351 tests) as regression — no new tests | — |
| CR-0.9 | 5 | ToolRouter register+urlpatterns; event emit→handler fires; wildcard handler fires; ToolModel FSM; loop/evaluator subscription | `tools/tests/test_extraction_safety.py` |
| **Total** | **44** | | |

### 3.2 Phase 1A tests (leaf relocations)

Each relocation test verifies: model creates at new location, view returns correct response from demo path, cross-app imports resolve.

| CR | App | Tests | Key assertions | File |
|----|-----|-------|----------------|------|
| CR-1A.1 | `triage/` | 3 | TriageResult creates; triage API returns data at demo path; encrypted CSV field roundtrips | `triage/tests/test_extraction_safety.py` |
| CR-1A.2 | `whiteboard/` | 4 | Board creates; BoardParticipant adds; vote records; demo path loads board list | `whiteboard/tests/test_extraction_safety.py` |
| CR-1A.3 | `learn/` | 4 | SectionProgress tracks; AssessmentAttempt scores; LearnSession creates with project FK; learn_content imports resolve | `learn/tests/test_extraction_safety.py` |
| CR-1A.4 | `vsm/` | 3 | ValueStreamMap creates with JSON fields; paired_with self-FK; demo path serves | `vsm/tests/test_extraction_safety.py` |
| CR-1A.5 | `simulation/` | 3 | PlantSimulation creates; source_vsm FK resolves to vsm/; demo path serves | `simulation/tests/test_extraction_safety.py` |
| CR-1A.6 | `qms_measurement/` | 4 | MeasurementEquipment creates; graph FK resolves; loop/evaluator import works; calibration fields populated | `qms_measurement/tests/test_extraction_safety.py` |
| CR-1A.7 | `qms_suppliers/` | 4 | SupplierRecord creates; state machine transitions; loop FK resolves (2 sites); demo path serves | `qms_suppliers/tests/test_extraction_safety.py` |
| **Total** | | **25** | | |

### 3.3 Phase 1B tests (leaf rebuilds)

Rebuild tests verify the pull contract works: reference registration, artifact fetch, delete-friction, tombstone handling.

| CR | App | Tests | Key assertions | File |
|----|-----|-------|----------------|------|
| CR-1B.1 | `workbench/` | 8 | Workbench creates; Artifact saves with 10-key content; ArtifactReference registers; pull API GET by dotted key; delete with references → 409; tombstone renders after source delete; DSWResult conversion creates Artifact row; SavedModel relocates | `workbench/tests/test_extraction_safety.py` |
| CR-1B.1a | `core/` | 4 | Notebook renders analysis result via pull API (not direct ORM); 4 DSWResult ref sites converted; notebook list still works; notebook detail still renders | `core/tests/test_notebook_pull.py` |
| CR-1B.2 | `triage/` | 3 | Reference registration works; artifact fetch returns triage data; delete-friction 409 | `triage/tests/test_extraction_safety.py` (append) |
| CR-1B.3 | `whiteboard/` | 3 | Board as evidence source; reference registration; delete-friction | `whiteboard/tests/test_extraction_safety.py` (append) |
| CR-1B.4 | `learn/` | 3 | sv-* widgets render; course progress persists; assessment grading works | `learn/tests/test_extraction_safety.py` (append) |
| CR-1B.5 | `vsm/` | 5 | Cockpit calculator; takt time via forgesiop; ForgeViz chart renders; future state comparison; demo path serves rebuilt UI | `vsm/tests/test_extraction_safety.py` (append) |
| CR-1B.6 | `simulation/` | 3 | DES engine runs; source_vsm data populates; demo path serves | `simulation/tests/test_extraction_safety.py` (append) |
| CR-1B.7 | `qms_measurement/` | 3 | Pull contract; calibration evidence fetchable; delete-friction | `qms_measurement/tests/test_extraction_safety.py` (append) |
| CR-1B.8 | `qms_suppliers/` | 3 | Pull contract; supplier qualification fetchable; delete-friction | `qms_suppliers/tests/test_extraction_safety.py` (append) |
| **Total** | | **35** | | |

### 3.4 Phase 2A tests (medium-coupling relocations)

Each test verifies: model creates at new location, loop/ FKs resolve, graph/ FKs resolve (where applicable), state machines preserved.

| CR | App | Tests | Key assertions | File |
|----|-----|-------|----------------|------|
| CR-2A.1 | `qms_risk/` | 5 | FMEA creates; FMEARow.hypothesis_link to core.Hypothesis; Risk creates; loop.FMISRow FK resolves; FMEA filters to site | `qms_risk/tests/test_extraction_safety.py` |
| CR-2A.2 | `qms_documents/` | 6 | ControlledDocument creates; graph.ProcessNode M2M resolves; DocumentRevision chains; ISODocument↔ControlledDocument OneToOne; loop/ imports resolve (6 sites); ISOSection embed works | `qms_documents/tests/test_extraction_safety.py` |
| CR-2A.3 | `qms_training/` | 4 | TrainingRequirement creates with ControlledDocument FK; TrainingRecord creates; loop/ imports resolve (4 sites); certification_status property works | `qms_training/tests/test_extraction_safety.py` |
| CR-2A.4 | `qms_nonconformance/` | 5 | NCR creates; state machine transitions; CustomerComplaint creates with NCR FK; loop/ import resolves; graph linkage opt-in works | `qms_nonconformance/tests/test_extraction_safety.py` |
| CR-2A.5 | `qms_audit/` | 4 | InternalAudit creates; AuditFinding.ncr FK resolves to qms_nonconformance; ManagementReview creates; data_snapshot captures | `qms_audit/tests/test_extraction_safety.py` |
| CR-2A.6 | `qms_investigation/` | 5 | RCASession creates; state machine transitions; IshikawaDiagram creates; CEMatrix creates; investigation_bridge resolves | `qms_investigation/tests/test_extraction_safety.py` |
| CR-2A.7 | `qms_a3/` | 4 | A3Report creates; rca_session FK resolves to qms_investigation; Board embed FK resolves; demo path serves | `qms_a3/tests/test_extraction_safety.py` |
| CR-2A.8 | `hoshin/` | 5 | HoshinProject creates; ResourceCommitment state machine; XMatrixCorrelation signal handlers; AFE creates (only via Hoshin per AFE policy); Employee FK resolves | `hoshin/tests/test_extraction_safety.py` |
| CR-2A.9 | `core/` | 3 | notebook_views at new location; lazy imports resolve; demo path serves | `core/tests/test_extraction_safety.py` (append) |
| **Total** | | **41** | | |

### 3.5 Phase 2B tests (medium-coupling rebuilds)

Rebuild tests verify: pull contract consumers work, canonical conversions land, new sinks produce output.

| CR | App | Tests | Key assertions | File |
|----|-----|-------|----------------|------|
| CR-2B.1 | `qms_risk/` | 3 | Pull contract — FMEA fetchable as evidence; individual risk fetchable; delete-friction | `qms_risk/tests/test_extraction_safety.py` (append) |
| CR-2B.2 | `qms_documents/` | 4 | Pull contract; ForgeDoc integration; document pulled into audit context; delete-friction | `qms_documents/tests/test_extraction_safety.py` (append) |
| CR-2B.3 | `qms_training/` | 3 | Pull contract; training record pulled into supplier audit; delete-friction | `qms_training/tests/test_extraction_safety.py` (append) |
| CR-2B.4 | `qms_nonconformance/` | 4 | Pull contract; NCR pulled into investigation; investigation finding pushed back via reference; delete-friction | `qms_nonconformance/tests/test_extraction_safety.py` (append) |
| CR-2B.5 | `qms_audit/` | 4 | Pull contract; auditor independence rules; evidence pull from docs/training/suppliers; clause coverage calculation | `qms_audit/tests/test_extraction_safety.py` (append) |
| CR-2B.6 | `qms_investigation/` | 5 | Pull from workbench + triage + whiteboard; emit investigation container; canonical transition behavior; evidence_bridge creates records; delete-friction | `qms_investigation/tests/test_extraction_safety.py` (append) |
| CR-2B.7 | `qms_a3/` | 5 | **Full rebuild**: new pull from workbench + investigation; A3 container emits; new front-end renders at demo path; import from investigation works; import from whiteboard works | `qms_a3/tests/test_extraction_safety.py` (append) |
| CR-2B.8 | `hoshin/` | 5 | **Canonical HoshinKPI conversion**: linked_artifact FK set; pull API replaces direct ORM; auto-register ArtifactReference; tombstone on source delete; effective_actual returns value via pull | `hoshin/tests/test_extraction_safety.py` (append) |
| CR-2B.9 | `reports/` | 5 | **New sink**: report generates from A3 container; ForgeDoc PDF renders; pull contract consumer (pulls from workbench + investigation); demo path serves; legacy /api/reports/ still works in parallel | `reports/tests/test_extraction_safety.py` |
| CR-2B.10 | `sop/` | 4 | **New sink**: SOP generates from investigation container; ForgeDoc renders; pull contract consumer; demo path serves | `sop/tests/test_extraction_safety.py` |
| CR-2B.11 | CAPA delete | 3 | CAPAReport deletable (no remaining FKs); capa_views.py deleted; investigation + ForgeDoc CAPA generation works | `qms_investigation/tests/test_capa_replacement.py` |
| **Total** | | **45** | | |

### 3.6 Phase 3 + 4 tests

| CR | Tests | Key assertions | File |
|----|-------|----------------|------|
| CR-3.1 (cutover) | 10 | One per extracted app: production URL serves new code; old demo paths 404 or redirect; no broken imports in full `python manage.py check`; ForgeRack unaffected; full test suite passes | `tests/test_cutover.py` |
| CR-4.1 (Site move) | 8 | Site creates in qms_core; every extracted app's Site FK resolves to qms_core.Site; loop/ Site imports work; graph/ Site import works; safety/ Site + Employee imports work; ElectronicSignature hash chain unbroken; permissions shim deleted, direct imports work; agents_api/models.py is empty or deleted | `qms_core/tests/test_site_move.py` |
| **Total** | **18** | | |

### 3.7 Grand total

| Phase | Tests |
|-------|-------|
| Phase 0 | 44 |
| Phase 1A | 25 |
| Phase 1B | 35 |
| Phase 2A | 41 |
| Phase 2B | 45 |
| Phase 3 | 10 |
| Phase 4 | 8 |
| **TOTAL** | **~208** |

Plus existing regression tests maintained throughout (~351 iso_tests + ~83 whiteboard_tests + loop/graph/safety tests).

---

<a id="dual-path"></a>
## 4. Route-agnostic dual-path pattern

Per architecture §13.3: during the parallel period (Phase 1A through Phase 2B), both legacy and demo paths serve the same data. Tests must exercise both and assert equivalent behavior.

### 4.1 Pattern

```python
# <app>/tests/test_dual_path.py

import pytest
from django.test import Client

LEGACY_PREFIX = "/api/triage"
DEMO_PREFIX = "/app/demo/triage"


class TestTriageDualPath:
    """Verify legacy and demo paths return equivalent results.

    These tests are TEMPORARY — deleted at Phase 3 cutover when
    legacy paths are removed.
    """

    @pytest.fixture
    def client(self):
        return Client()

    @pytest.fixture
    def auth_headers(self, user):
        # ... authentication setup
        pass

    def test_list_equivalent(self, client, auth_headers):
        """Both paths return same triage result list."""
        legacy = client.get(f"{LEGACY_PREFIX}/", **auth_headers)
        demo = client.get(f"{DEMO_PREFIX}/", **auth_headers)
        assert legacy.status_code == demo.status_code == 200
        assert legacy.json()["results"] == demo.json()["results"]

    def test_detail_equivalent(self, client, auth_headers, triage_result):
        """Both paths return same detail for a given ID."""
        pk = str(triage_result.id)
        legacy = client.get(f"{LEGACY_PREFIX}/{pk}/", **auth_headers)
        demo = client.get(f"{DEMO_PREFIX}/{pk}/", **auth_headers)
        assert legacy.status_code == demo.status_code == 200
        assert legacy.json() == demo.json()

    def test_create_via_demo_readable_via_legacy(self, client, auth_headers):
        """Item created via demo path is visible via legacy path."""
        resp = client.post(f"{DEMO_PREFIX}/create/", data={...}, **auth_headers)
        assert resp.status_code == 201
        pk = resp.json()["id"]
        legacy = client.get(f"{LEGACY_PREFIX}/{pk}/", **auth_headers)
        assert legacy.status_code == 200
```

### 4.2 Which apps need dual-path tests

Only apps where the demo path serves a different code path than legacy. Apps where the demo path is a simple URL alias (same view function) don't need dual-path tests — their safety-net tests already cover the behavior.

| App | Needs dual-path? | Reason |
|-----|-----------------|--------|
| `triage/` | Yes | New app serves demo, old agents_api serves legacy |
| `whiteboard/` | Yes | Same |
| `learn/` | Yes | Same |
| `vsm/` | Yes (Phase 1B) | Cockpit rebuild is a different UI |
| `simulation/` | Yes | New app |
| `qms_measurement/` | Yes | New app |
| `qms_suppliers/` | Yes | New app |
| All Phase 2A apps | Yes | New apps with loop/ coordination |
| `workbench/` | No | Pull contract is additive, not a path replacement |
| `reports/` | Yes (Phase 2B) | New sink replaces old report_views |
| `sop/` | No | Brand new, no legacy path |

### 4.3 Estimated dual-path test count

~3 tests per app × 15 apps needing dual-path = **~45 dual-path tests**. All deleted at Phase 3 cutover.

### 4.4 Dual-path test lifecycle

1. **Written at:** Phase 1A relocation (for leaves) or Phase 2A relocation (for medium-coupling)
2. **Maintained during:** parallel period (may need updates at Phase 1B/2B rebuild if response shapes change)
3. **Deleted at:** Phase 3 cutover (CR-3.1). The cutover CR includes deletion of all `test_dual_path.py` files and the legacy routes they tested.

---

<a id="cutover"></a>
## 5. Phase 3 cutover test plan

The Phase 3 cutover (CR-3.1) is a single-night operation. The test plan for cutover night:

### 5.1 Pre-cutover checklist (run before starting)

1. Full test suite passes at current state (including all dual-path tests)
2. All demo paths verified by Eric in Gate 2 review
3. Database backup taken
4. Rollback plan ready (revert URL routing, no data migration needed)

### 5.2 Cutover sequence with test gates

```
Step 1: URL swap in svend/urls.py
  → TEST: python manage.py check --deploy (no broken URL references)

Step 2: Delete legacy route files (old *_urls.py files)
  → TEST: python manage.py check --deploy (clean URL namespace)

Step 3: Delete legacy view files (old *_views.py files in agents_api/)
  → TEST: python -c "import agents_api" (no import errors)

Step 4: Delete dual-path test files
  → TEST: python -m pytest --collect-only (no collection errors)

Step 5: Delete agents_api/iso/ shim __init__.py re-exports
  → TEST: python manage.py check --deploy

Step 6: Run full test suite
  → TEST: python -m pytest tests/ -x --tb=short
  → PASS CRITERION: all safety-net + regression tests pass

Step 7: Spot-check production URLs
  → Manual: visit 5 key pages, verify data renders
  → /app/analysis/, /app/rca/, /app/hoshin/, /app/fmea/, /app/learn/

Step 8: 301 redirects for API backcompat
  → TEST: curl old API URLs, verify 301 to new locations
```

### 5.3 Rollback trigger

If any test gate fails after Step 1, rollback by reverting the URL swap commit. No data migration to undo — the cutover is purely routing + code deletion.

---

<a id="site-move"></a>
## 6. Phase 4 Site move test plan

Site (24 incoming FKs) is the highest-risk single migration. Special test plan.

### 6.1 Pre-migration tests

Before the Phase 4 migration runs:

1. **FK inventory verification:** Script that introspects all Django models and reports every FK pointing to `agents_api.Site` or `agents_api.Employee`. Must match the gap analysis §9.2 table exactly. Any new FK not in the table = migration must be updated.

2. **ElectronicSignature hash chain:** Verify chain integrity before migration. The hash chain must not break during the table move.
   ```python
   from syn.audit.utils import verify_chain_integrity
   assert verify_chain_integrity(ElectronicSignature)
   ```

3. **Baseline counts:** Record row counts for Site, SiteAccess, Employee, ActionToken, Checklist, ChecklistExecution, ElectronicSignature, QMSFieldChange, QMSAttachment. Post-migration counts must match.

### 6.2 Migration tests

After the Phase 4 migration runs:

1. Every extracted app's Site FK resolves: create a record in each app with a Site FK, verify it saves and loads correctly.
2. `loop/` Site imports work: `from qms_core.models import Site` succeeds, `loop/models.py` Site references resolve.
3. `graph/` Site import works.
4. `safety/` Site + Employee imports work.
5. ElectronicSignature hash chain still valid post-migration.
6. Row counts match pre-migration baseline.
7. `agents_api/permissions.py` shim is deleted; direct `qms_core.permissions` imports work everywhere.
8. `agents_api/models.py` is empty or contains only ForgeRack (RackSession) — no other models remain.

### 6.3 Rollback plan

Phase 4 involves a Django `RunSQL` migration that renames the `db_table` metadata (or actually moves the table). Rollback = reverse migration. Test the reverse migration on a staging copy before running in production.

---

<a id="post-extraction"></a>
## 7. Post-extraction comprehensive rebuild

After the extraction is complete (Phase 4 landed, codebase stable), the comprehensive test suite rebuild begins. This is a **separate initiative** from the extraction — it has its own CRs, its own timeline, and follows `feedback_build_then_standardize.md`.

### 7.1 Scope

- Convert ~208 safety-net tests into comprehensive behavioral test suites per TST-001
- Add edge case coverage, error path testing, permission boundary testing
- Establish per-app test targets (coverage ratchet per CAL-001)
- Wire test hooks to standards (per DOC-001 §7 `<!-- test: -->` syntax)
- Delete any remaining agents_api test files that were orphaned

### 7.2 Priority order for comprehensive testing

Based on risk tier from extraction and production criticality:

1. **workbench/** — pull contract is load-bearing for entire QMS
2. **qms_core/** — Site + permissions + ElectronicSignature (compliance-critical)
3. **hoshin/** — HoshinKPI canonical conversion must be bulletproof
4. **qms_investigation/** — canonical transition app
5. **qms_documents/** — highest coupling after Site
6. **qms_risk/** — FMEA is the most-used QMS tool
7. **tools/** — event bus reliability
8. Everything else by LOC descending

### 7.3 What the extraction tests become

The `test_extraction_safety.py` files in each app are the **seed**. They contain the critical behavioral assertions. The comprehensive rebuild expands them:
- Split into focused test modules (`test_models.py`, `test_views.py`, `test_pull_contract.py`)
- Add fixtures and factories
- Add error path tests
- Add permission boundary tests
- Wire to standards hooks

---

<a id="infrastructure"></a>
## 8. Test infrastructure requirements

### 8.1 Fixtures needed for extraction tests

Most extraction tests need:
- A user (with authentication)
- A Site (enterprise context)
- A Project (evidence integration)
- A Tenant (multi-tenancy)

**Shared fixture module:** `tests/conftest.py` at the web/ root. Provides `user`, `site`, `project`, `tenant` fixtures that all app test files can import.

```python
# tests/conftest.py
import pytest
from django.contrib.auth import get_user_model

User = get_user_model()


@pytest.fixture
def user(db):
    return User.objects.create_user(
        email="test@example.com",
        password="testpass123",
    )


@pytest.fixture
def site(db, user):
    from agents_api.models import Site  # Phase 0-2: agents_api
    # After Phase 4: from qms_core.models import Site
    return Site.objects.create(
        name="Test Site",
        created_by=user.email,
    )


@pytest.fixture
def project(db, user):
    from core.models import Project
    return Project.objects.create(
        name="Test Project",
        user=user,
    )
```

**Note:** The `site` fixture import path changes at Phase 4. During the parallel period, it imports from `agents_api.models`. After Phase 4, from `qms_core.models`. This is the ONE fixture that needs updating at Phase 4.

### 8.2 Test runner configuration

Ensure `pytest.ini` or `pyproject.toml` includes all new app test directories:

```ini
[tool:pytest]
testpaths =
    tests
    triage/tests
    whiteboard/tests
    learn/tests
    vsm/tests
    simulation/tests
    qms_measurement/tests
    qms_suppliers/tests
    qms_core/tests
    qms_risk/tests
    qms_documents/tests
    qms_training/tests
    qms_nonconformance/tests
    qms_audit/tests
    qms_investigation/tests
    qms_a3/tests
    hoshin/tests
    reports/tests
    sop/tests
    tools/tests
    workbench/tests
    core/tests
```

This grows as apps are created. Each Phase 1A/2A CR adds its app's test directory.

### 8.3 CI considerations

Per `project_enterprise_readiness.md`: CI testing is a known gap. The extraction doesn't set up CI — it writes tests that can run locally via `pytest`. CI integration is a separate enterprise readiness initiative.

---

<a id="existing-tests"></a>
## 9. Existing test inventory

### 9.1 Tests that survive the extraction

| File | Tests | Maintained by | Notes |
|------|-------|---------------|-------|
| `iso_tests.py` | 351 | Phase 0.8 split (regression suite) → Phase 2A extractions split tests with code | Tests split alongside iso_views.py sub-modules |
| `whiteboard_tests.py` | ~83 | Phase 1A whiteboard extraction | Move to `whiteboard/tests/` |
| `test_dsw_views_behavioral.py` | ~10 | Phase 0.1 dsw→analysis move | Update imports post-move |
| `loop/tests/*.py` | varies | Maintained as-is; imports updated per-extraction | loop/ is the primary coordinated-CR partner |
| `graph/tests_synara.py` | varies | Phase 0.4 forgesia wiring | Update imports to forgesia |
| `graph/tests_qms.py` | varies | Phase 1A qms_measurement extraction | Update imports |
| `safety/tests/*.py` | varies | Phase 4 Site move | Update imports at Site move |
| `core/tests/*.py` | varies | Phase 1B.1a notebook conversion | Update DSWResult refs |
| `syn/audit/tests/*.py` | 11 files | Phase 3 cutover + follow-up | Update agents_api paths to new apps |
| `tools/tests/*.py` (new) | 3 files | Phase 0.9 tools/ creation | New from relocated test files |

### 9.2 Tests deleted in Phase 0

Per CR-0.6 and TST-001 §10.6:

| File | LOC | Reason |
|------|-----|--------|
| `test_endpoint_smoke.py` | 712 | Sweep test — existence check only |
| `test_t1_deep.py` | 1,942 | Sweep test — smoke assertions |
| `test_t2_views_smoke.py` | 338 | Sweep test |
| `test_*_coverage.py` (8 files) | ~2,000 | Coverage sweep — no behavioral assertions |
| **Total deleted** | **~5,000 LOC** | |

### 9.3 Actual test landscape (audited 2026-04-10)

| Location | Files | LOC | Notes |
|----------|-------|-----|-------|
| `agents_api/tests/` | 49 | 22,863 | 37 behavioral + 12 sweep/smoke |
| `syn/audit/tests/` | 43 | 26,145 | Standards, compliance, change mgmt |
| `syn/core/tests/` | 5 | 3,226 | Infrastructure |
| `syn/api/tests/` | 1 | 687 | API middleware |
| `syn/err/tests/` | 1 | 1,176 | Error handling |
| `syn/log/tests/` | 1 | 1,261 | Logging |
| `syn/sched/tests/` | 4 | 1,209 | Task scheduler |
| `loop/tests/` | 0 | 0 | **No test directory** |
| `graph/tests/` | 0 | 0 | **No test directory** |
| `safety/tests/` | 0 | 0 | **No test directory** |
| `workbench/tests/` | 0 | 0 | **No test directory** |
| `tools/tests/` | 0 | 0 | **Does not exist yet** |
| **Total** | **104** | **56,567** | |

Note: loop/, graph/, safety/ have test files at module level (e.g. `graph/tests_synara.py`, `graph/tests_qms.py`) but no `tests/` subdirectory. These files are maintained during extraction and eventually moved into proper `tests/` directories.

### 9.4 Test count trajectory

| Milestone | Est. test count | Notes |
|-----------|----------------|-------|
| Pre-extraction (current) | ~500+ (many are sweep) | 104 files / 56,567 LOC includes TST-001 violations |
| Post-Phase 0 | ~455 | -12 sweep files deleted; +44 Phase 0 safety-net |
| Post-Phase 1 | ~560 | +25 relocation + 35 rebuild + ~45 dual-path |
| Post-Phase 2 | ~690 | +41 relocation + 45 rebuild |
| Post-Phase 3 cutover | ~655 | -45 dual-path deleted; +10 cutover |
| Post-Phase 4 | ~665 | +8 Site move |
| Post-comprehensive rebuild | ~1,200+ | Expansion from safety-net seeds |

---

## Appendix — Test naming conventions

Per TST-001:

```
test_<what>_<behavior>_<expected>

Examples:
  test_fmea_create_returns_201_with_id
  test_hoshin_kpi_effective_actual_uses_pull_api
  test_site_fk_resolves_after_phase4_migration
  test_dual_path_triage_list_returns_equivalent_results
```

All test functions must have a docstring explaining what behavior they verify. No bare `assert True` or `assert response.status_code == 200` without checking response content.

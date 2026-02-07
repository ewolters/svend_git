# Change Log

All edits to the kjerne codebase are logged here. Each entry records what changed, why, and how to verify.

## Format

```
### YYYY-MM-DD — Summary
**Debt item:** DEBT.md reference (if applicable)
**Files changed:**
- `path/to/file` — what changed
**Verification:** how to confirm it worked
**Commit:** git hash
```

---

### 2026-02-06 — Lock down repo for public push
**Debt item:** [REPO] svend.db + snapshot tar.gz tracked in git
**Files changed:**
- `.gitignore` — added `*.db`, `*.tar.gz`
- `.kjerne/DEBT.md` — added 22 tracked debt items from full audit
- `services/svend/agents/agents/site/data/svend.db` — removed from git tracking (file kept on disk)
- `.kjerne/snapshots/**/*.tar.gz` (10 files) — removed from git tracking (files kept on disk)
**Verification:** `git status` shows clean, `git ls-files '*.db' '*.tar.gz'` returns empty
**Commit:** 9c9396e

---

### 2026-02-06 — Add project documentation and debt closure process
**Debt item:** N/A (infrastructure)
**Files changed:**
- `CLAUDE.md` (new) — root-level architecture documentation: module map, data model (both current + target), API surface, integration pattern, serving config, working conventions
- `log.md` (new) — change log for all edits
- `DEBT-001.md` (new) — repeatable process for closing technical debt: pick → document → change → test → log → update DEBT.md → commit → push. Includes P1 dependency map.
**Verification:** files exist and are readable
**Commit:** 2a3c2b6

---

### 2026-02-06 — P1: DSW ↔ Evidence integration
**Debt item:** [DSW] No integration with Projects/Evidence
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — added `problem_id` support to `run_analysis()` (line ~1038) and `dsw_from_data()` (line ~399). When `problem_id` is in the request body, analysis results are linked as evidence via `add_finding_to_problem()`. Uses `guide_observation` for summary (falls back to cleaned `summary` text). Maps analysis types to evidence types (stats/ml/bayesian/spc → data_analysis, viz → observation).
- `services/svend/web/agents_api/tests.py` — added `EvidenceIntegrationTest` class with 6 tests: Problem.add_evidence(), add_finding_to_problem() helper, invalid/empty ID handling, DSW with/without problem_id.
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — both files pass
- End-to-end test: created Problem → added evidence via add_finding_to_problem() → verified 2 evidence items → cleaned up. PASSED.
**Commit:** 0eef3fb

---

### 2026-02-06 — P1: Experimenter ↔ Evidence integration
**Debt item:** [EXPERIMENTER] Only 2/9 endpoints create evidence
**Files changed:**
- `services/svend/web/agents_api/experimenter_views.py` — added `problem_id` support to 4 additional endpoints:
  - `power_analysis()` — "Power analysis (test_type): need N=X for effect d=Y"
  - `design_experiment()` — "Generated {type} design: N runs, K factors"
  - `contour_plot()` — "Response surface: optimal at X=val, Y=val (predicted=Z)"
  - `optimize_response()` — "DOE optimization: desirability=X, settings: ..."
  - Skipped `doe_guidance_chat` (chat interface, not analysis results), `design_types` and `available_models` (read-only metadata).
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — passes
- All 4 endpoints follow the exact same pattern as existing `full_experiment` and `analyze_results`
**Commit:** 0eef3fb

---

### 2026-02-06 — P1: Phase 1 model migration (Problem → core.Project dual-write)
**Debt item:** [CORE] agents_api.Problem → core.Project migration
**Files changed:**
- `services/svend/web/agents_api/models.py` — added `core_project` FK field to Problem, 4 sync methods: `ensure_core_project()`, `sync_hypothesis_to_core()`, `sync_evidence_to_core()`, `_find_core_hypothesis()`
- `services/svend/web/agents_api/migrations/0008_add_core_project_fk.py` — migration adding core_project FK column
- `services/svend/web/agents_api/problem_views.py` — added dual-write calls to 6 write paths: `problems_list()` POST, `add_hypothesis()`, `add_evidence()`, `reject_hypothesis()`, `resolve_problem()`, `generate_hypotheses()`
- `services/svend/web/agents_api/tests.py` — added `DualWriteMigrationTest` class with 4 tests: ensure_core_project, sync_hypothesis, sync_evidence_with_links, find_core_hypothesis
**Data migration:**
- Existing "Employee Turnover" Problem (5 hypotheses, 0 evidence) migrated to core.Project with 5 core.Hypothesis records
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — all files pass
- End-to-end test: created Problem → ensure_core_project → sync_hypothesis → sync_evidence → verified EvidenceLink + Bayesian update (0.6 → 0.73) → cleaned up. PASSED.
- Verified all 6 view write paths have dual-write wired in via `inspect.getsource()`
- Employee Turnover: core.Project created, 5 hypotheses synced
**Commit:** f4fb8db

---

### 2026-02-06 — P1: Synara persistence to Django ORM
**Debt item:** [SYNARA] In-memory only — state lost on server restart
**Files changed:**
- `services/svend/web/core/models/project.py` — added `synara_state` JSONField to Project model
- `services/svend/web/core/migrations/0003_add_synara_state.py` — migration adding synara_state column
- `services/svend/web/agents_api/synara_views.py` — replaced in-memory `_synara_instances` dict with DB-backed `_synara_cache` + `save_synara()`. Added `_resolve_project()` to resolve both Project and Problem UUIDs. Added `save_synara()` calls to all 9 mutating endpoints.
- `services/svend/web/agents_api/tests.py` — added `SynaraPersistenceTest` class with 3 tests: save/load round-trip, Problem UUID resolution, evidence-belief persistence.
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — all files pass
- End-to-end test: created Synara → add hypothesis → add evidence → save → clear cache → reload → verified hypothesis/evidence/posterior survived round-trip. PASSED.
- Problem-to-Project resolution: Problem UUID → follow FK → save to core.Project. PASSED.
**Commit:** (pending)

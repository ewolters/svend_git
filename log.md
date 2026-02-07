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
**Commit:** (pending)

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
**Commit:** (pending)

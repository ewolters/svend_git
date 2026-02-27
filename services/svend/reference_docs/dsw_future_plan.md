# DSW Monolith Split Plan

**Status:** Planning (attempted 2026-02-16, reverted due to endpoint dependency failures)
**File:** `agents_api/dsw_views.py` (25,035 lines)
**Target:** `agents_api/dsw/` package (12 files, 24,728 lines already extracted)

---

## 1. Current State

### The Monolith (`dsw_views.py`)

The file has five natural zones:

| Zone | Lines | Content | Extractable? |
|------|-------|---------|-------------|
| **Globals + Shared Helpers** | 1–240 | Imports, `_model_cache`, `cache_model`, `get_cached_model`, `log_agent_action`, `_preload_llm_background`, `save_model_to_disk` | Yes → `common.py` (done) |
| **ML Helpers** | 241–1037 | `_build_ml_diagnostics`, `_diag_classification`, `_diag_regression`, `_claude_generate_schema`, `_generate_data_from_schema`, `_clean_for_ml`, `_stratified_split`, `_stratified_split_3way`, `_classification_reliability`, `_auto_train`, `_claude_interpret_results` | Yes → `common.py` (done) |
| **ML/Model HTTP Endpoints** | 1038–2541 | `dsw_from_intent`, `dsw_from_data`, `dsw_download`, `list_models`, `save_model_from_cache`, `download_model`, `delete_model`, `run_model`, `optimize_model`, `models_summary`, `model_versions`, `model_report`, `scrub_data`, `scrub_analyze`, `_detect_data_biases`, `run_analysis` | Partially → `endpoints_ml.py` (done, needs testing) |
| **Analysis Engines** | 2542–23123 | `_effect_magnitude`, `_practical_block`, `_ml_interpretation`, `_fit_best_distribution`, `run_simulation`, `run_statistical_analysis`, `run_ml_analysis`, `run_bayesian_analysis`, `run_reliability_analysis`, SPC helpers, `run_spc_analysis`, `run_visualization` | Yes → 8 files (done, verified working) |
| **Data/Code/Assistant Endpoints** | 23124–25035 | `upload_data`, `execute_code`, `generate_code`, `analyst_assistant`, 5 LLM response generators, `transform_data`, `download_data`, `triage_data`, `triage_scan` | No → `endpoints_data.py` (broken, needs work) |

### The Package (`dsw/`)

Already extracted and verified working via `python -c "import"`:

| File | Lines | Status | Content |
|------|-------|--------|---------|
| `__init__.py` | 18 | OK | Docstring only |
| `common.py` | 1,252 | OK | Shared helpers (cache, ML utilities, diagnostics) |
| `dispatch.py` | 249 | OK | `run_analysis()` router → sub-modules |
| `stats.py` | 11,495 | OK | `run_statistical_analysis()` — 200+ tests |
| `ml.py` | 3,298 | OK | `run_ml_analysis()` |
| `bayesian.py` | 451 | OK | `run_bayesian_analysis()` |
| `reliability.py` | 851 | OK | `run_reliability_analysis()` |
| `spc.py` | 2,049 | OK | `run_spc_analysis()` (imports `..bayes_doe`) |
| `viz.py` | 1,745 | OK | `run_visualization()` — Bayesian SPC plots |
| `simulation.py` | 327 | OK | `run_simulation()` — Monte Carlo |
| `endpoints_ml.py` | 1,063 | Untested | ML/model HTTP endpoints (imports look complete) |
| `endpoints_data.py` | 1,930 | **BROKEN** | Data/code/assistant endpoints (see below) |

---

## 2. What Failed (2026-02-16)

### Attempted: Big-bang switchover

Changed `dsw_urls.py` to import from `dsw/` package, deleted `dsw_views.py`, and updated all consumers. Broke production because:

1. **`endpoints_data.py` missing `_preload_llm_background`** — `upload_data()` calls it at line 133 but the import was missing. Every data upload returned `NameError`.

2. **`sed` extraction missed decorators** — The `def upload_data(request):` line was at 23129, but the three decorators (`@csrf_exempt`, `@require_http_methods(["POST"])`, `@require_auth`) were at lines 23126–23128. The `sed` range started at the function, not the decorators, so the endpoint had no auth/CSRF handling → 403 on every request.

3. **Missing top-level imports** — The extracted file only had `json` and `logging` at the top. Functions use `uuid`, `tempfile`, `Path`, `settings`, `numpy`, `pandas` — all were imported at the monolith top level but not carried over.

4. **Deep LLM coupling** — `analyst_assistant()` at line 525 does `from .. import views as agent_views` to access `agent_views._shared_llm` and `agent_views._shared_llm_loaded`. This is a runtime cross-module dependency on the agent views singleton.

5. **Cascading failures** — Each fix revealed the next missing dependency. After 4 rounds of fixes, `_preload_llm_background` was still undefined, proving the endpoint zone has too many tentacles to extract with `sed`.

### Lesson

The **analysis engine functions** (zones 3–4) are self-contained: they take `(df, analysis_id, config)` and return dicts. They have no Django imports, no request handling, no LLM coupling. They extracted cleanly.

The **HTTP endpoint functions** (zones 2 and 5) are deeply coupled: they access `request`, Django models, the LLM singleton, file system paths, background threads, and call helpers defined hundreds of lines away. They need careful, manual extraction with full dependency auditing.

---

## 3. External Consumers (Public API Surface)

Four files import from `dsw_views.py`:

### `dsw_urls.py` — 21 view functions
```python
from . import dsw_views as views
# References: views.dsw_from_intent, views.dsw_from_data, views.dsw_download,
#   views.list_models, views.models_summary, views.save_model_from_cache,
#   views.download_model, views.delete_model, views.run_model, views.optimize_model,
#   views.model_versions, views.model_report, views.run_analysis, views.execute_code,
#   views.generate_code, views.upload_data, views.analyst_assistant, views.transform_data,
#   views.download_data, views.triage_data, views.triage_scan
```

### `autopilot_views.py` — 6 ML helpers
```python
from .dsw_views import (
    _auto_train, _build_ml_diagnostics, _clean_for_ml,
    _create_ml_evidence, cache_model, save_model_to_disk,
)
```

### `ml_pipeline.py` — 2 ML helpers (local import at line 184)
```python
from .dsw_views import _clean_for_ml, _auto_train
```

### `analysis/__init__.py` — 5 analysis engines (re-export shim)
```python
from agents_api.dsw_views import (
    run_statistical_analysis, run_ml_analysis,
    run_bayesian_analysis, run_spc_analysis, run_visualization,
)
```

---

## 4. Switchover Plan (Incremental)

**Principle:** Switch one import path at a time. Verify after each. Never delete the monolith until ALL paths are switched.

### Phase 1: Analysis Engine Routing (LOW RISK)

**What:** Route `run_analysis` through `dsw/dispatch.py` instead of the monolith's inline dispatcher.

**How:** In `dsw_views.py`, replace the `run_analysis()` function body (lines 2542–2750, ~200 lines) with a thin wrapper:
```python
@csrf_exempt
@require_http_methods(["POST"])
@gated
def run_analysis(request):
    from .dsw.dispatch import _dispatch_analysis
    return _dispatch_analysis(request)
```

The decorators and request parsing stay in the monolith. Only the analysis type routing delegates to `dispatch.py`, which already imports from the sub-modules.

**External impact:** None. `dsw_urls.py` still imports `views.run_analysis` from the monolith.

**Verification:** Upload data → run each of the 7 analysis types → confirm results match.

**Files changed:** `dsw_views.py` only.

### Phase 2: Analysis Shim for `analysis/__init__.py` (LOW RISK)

**What:** Point `analysis/__init__.py` at the `dsw/` sub-modules instead of the monolith.

**How:**
```python
# analysis/__init__.py
from agents_api.dsw.stats import run_statistical_analysis
from agents_api.dsw.ml import run_ml_analysis
from agents_api.dsw.bayesian import run_bayesian_analysis
from agents_api.dsw.spc import run_spc_analysis
from agents_api.dsw.viz import run_visualization
```

**External impact:** Any code doing `from agents_api.analysis import run_statistical_analysis` now gets the sub-module version. These are identical functions.

**Verification:** `python -c "from agents_api.analysis import run_statistical_analysis; print('OK')"`

**Files changed:** `analysis/__init__.py` only.

### Phase 3: Shared Helpers for `autopilot_views.py` and `ml_pipeline.py` (LOW RISK)

**What:** Point `autopilot_views.py` and `ml_pipeline.py` at `dsw/common.py` instead of the monolith.

**How:**
```python
# autopilot_views.py
from .dsw.common import (
    _auto_train, _build_ml_diagnostics, _clean_for_ml,
    _create_ml_evidence, cache_model, save_model_to_disk,
)

# ml_pipeline.py (line 184, local import)
from .dsw.common import _clean_for_ml, _auto_train
```

**Pre-requisite:** Verify `dsw/common.py` exports ALL 6 symbols with identical signatures. Current common.py has all of them.

**Risk:** `common.py` has its own `_model_cache` dict. The monolith also has one. If both are imported, they're separate caches. This is fine as long as we don't switch partially — either ALL model cache users import from common.py or ALL import from the monolith.

**Verification:**
1. Upload data
2. Train a model via autopilot (clean+train)
3. Verify model appears in model list
4. Run prediction on saved model

**Files changed:** `autopilot_views.py`, `ml_pipeline.py`.

### Phase 4: ML Endpoints (`endpoints_ml.py`) (MEDIUM RISK)

**What:** Move ML/model HTTP endpoints from the monolith to `dsw/endpoints_ml.py`.

**Pre-requisite:** Phase 3 must be complete (shared helpers pointing to `dsw/common.py`).

**Current state of `endpoints_ml.py`:** All imports appear complete. File has 16 functions with proper decorators.

**Dependency audit required before switchover:**

| Function | Dependencies to verify |
|----------|----------------------|
| `dsw_from_intent` | `_claude_generate_schema`, `_generate_data_from_schema`, `_clean_for_ml`, `_auto_train`, `_build_ml_diagnostics`, `cache_model`, `_claude_interpret_results`, `_create_ml_evidence` |
| `dsw_from_data` | `_clean_for_ml`, `_auto_train`, `_build_ml_diagnostics`, `cache_model`, `_claude_interpret_results`, `_create_ml_evidence` |
| `run_model` | `get_cached_model`, `SavedModel`, `_predict_numeric` (defined locally) |
| `optimize_model` | `SavedModel`, `get_cached_model` |
| `save_model_from_cache` | `get_cached_model`, `save_model_to_disk` |
| `scrub_data` | `DataCleaner` (from `scrub` package) |

**How:** In `dsw_urls.py`, switch one endpoint at a time:
```python
from . import dsw_views as views
from .dsw.endpoints_ml import dsw_from_intent  # Switch this one first

urlpatterns = [
    path("from-intent/", dsw_from_intent, ...),  # New
    path("from-data/", views.dsw_from_data, ...),  # Still monolith
    ...
]
```

**Verification per endpoint:**
- `dsw_from_intent`: Generate synthetic data from intent → verify CSV download works
- `dsw_from_data`: Upload data → train model → verify diagnostics and model save
- Model CRUD: List, save, download, delete, run, optimize, versions, report
- `scrub_data` / `scrub_analyze`: Run data scrubbing

**Files changed:** `dsw_urls.py` (gradually), eventually `dsw_views.py` (remove switched functions).

### Phase 5: Data/Code/Assistant Endpoints (`endpoints_data.py`) (HIGH RISK)

**What:** Move data upload, code execution, analyst assistant, and triage from the monolith.

**This is the hardest phase.** The `analyst_assistant()` function has deep coupling:

#### Dependency Map for `endpoints_data.py`

```
upload_data()
├── numpy, pandas (local imports)
├── uuid, tempfile, Path, settings (top-level — CURRENTLY MISSING)
└── _preload_llm_background() ← common.py (IMPORT MISSING)

execute_code()
├── numpy, pandas, scipy, sklearn (sandboxed exec)
├── uuid, Path, settings
└── DSWResult model (IMPORT MISSING)

generate_code()
├── anthropic (local import)
└── log_agent_action ← common.py (imported)

analyst_assistant()
├── numpy, pandas (local imports)
├── agent_views._shared_llm / _shared_llm_loaded ← views.py singleton
├── agent_views.get_shared_llm() ← views.py
├── generate_anthropic_response() ← defined locally
├── generate_llm_response() ← defined locally
├── generate_analyst_response() ← defined locally
├── generate_researcher_response() ← defined locally
└── generate_writer_response() ← defined locally

generate_researcher_response()
└── WebSearchAPI (external, imported inline)

generate_writer_response()
└── No external deps beyond numpy/pandas

transform_data()
├── numpy, pandas
└── uuid, Path, settings

triage_data() / triage_scan()
├── DataCleaner from scrub
└── numpy, pandas
```

#### Required fixes before switchover

1. **Add missing imports to `endpoints_data.py`:**
   ```python
   import numpy as np
   import pandas as pd
   import time
   from ..models import DSWResult
   from .common import _preload_llm_background, log_agent_action
   ```

2. **Resolve LLM singleton coupling.** `analyst_assistant()` does:
   ```python
   from .. import views as agent_views
   if agent_views._shared_llm_loaded:
       llm = agent_views._shared_llm
   ```
   This works with relative imports from `dsw/` → `agents_api/views.py`. Verify the `..` resolves correctly.

3. **Verify all 5 generate_*_response functions** are self-contained in the extracted file. Currently they are — each does its own local imports. No changes needed.

4. **Test each endpoint individually** before wiring into `dsw_urls.py`.

**Files changed:** `endpoints_data.py` (fix imports), `dsw_urls.py` (switch paths), eventually `dsw_views.py` (remove functions).

### Phase 6: Delete the Monolith

**Pre-requisites:** ALL of phases 1–5 complete and verified in production.

**How:**
1. `dsw_urls.py` imports nothing from `dsw_views`
2. `autopilot_views.py` imports from `dsw.common`
3. `ml_pipeline.py` imports from `dsw.common`
4. `analysis/__init__.py` imports from `dsw.*` sub-modules
5. `dsw_views.py` has no remaining code that's referenced anywhere
6. Delete `dsw_views.py`
7. Commit and reload gunicorn

---

## 5. Testing Checklist

Each phase must pass ALL applicable checks before proceeding:

### Smoke Tests
- [ ] `python3 -c "import django; django.setup(); from agents_api import dsw_urls"` — no import errors
- [ ] Gunicorn starts without errors after reload (`kill -HUP <master_pid>`)
- [ ] Upload a CSV file (tests `upload_data`)
- [ ] Run one analysis per type: t-test (stats), random forest (ml), bayes t-test (bayesian), control chart (spc), scatter (viz), monte carlo (simulation)
- [ ] Save a model, list models, run prediction, delete model
- [ ] Ask the analyst assistant a question
- [ ] Execute custom code on loaded data
- [ ] Generate code for an analysis
- [ ] Transform data (add column, filter)
- [ ] Download data as CSV
- [ ] Run triage scan and triage clean
- [ ] From-intent: generate synthetic data
- [ ] From-data: upload and auto-train
- [ ] Autopilot: clean+train pipeline

### Regression
- [ ] Workbench save/load still works (CSRF fix preserved)
- [ ] SPC endpoints (`/api/spc/`) still work (separate from DSW)
- [ ] Experimenter endpoints still work
- [ ] No 500 errors in gunicorn logs for 1 hour after switchover

---

## 6. Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1. Analysis routing | Low | Monolith wrapper delegates to dispatch; fallback is one-line revert |
| 2. Analysis shim | Low | Re-export only; no behavior change |
| 3. Shared helpers | Low–Med | `_model_cache` singleton must be unified; test autopilot end-to-end |
| 4. ML endpoints | Medium | 16 functions with Django deps; test each before switching URL |
| 5. Data endpoints | High | LLM singleton coupling, sandboxed exec, 5 response generators |
| 6. Delete monolith | Low | Only after all phases pass; `git revert` if anything breaks |

---

## 7. Estimated Effort

| Phase | Effort | Description |
|-------|--------|-------------|
| Phase 1 | 15 min | Replace function body with 3-line wrapper |
| Phase 2 | 5 min | Change 5 import paths |
| Phase 3 | 15 min | Change 2 files, verify model cache behavior |
| Phase 4 | 1–2 hr | Audit 16 functions, switch URLs one by one, test each |
| Phase 5 | 2–3 hr | Fix imports, test LLM coupling, test all endpoints |
| Phase 6 | 5 min | Delete file, commit |

**Total: ~4–6 hours of careful, incremental work.**

---

## 8. Architecture After Split

```
agents_api/
├── dsw_views.py          ← DELETED (Phase 6)
├── dsw_urls.py            ← imports from dsw/ package
├── autopilot_views.py     ← imports from dsw.common
├── ml_pipeline.py         ← imports from dsw.common
├── dsw/
│   ├── __init__.py        ← package marker
│   ├── common.py          ← shared helpers (1,252 lines)
│   ├── dispatch.py        ← run_analysis router (249 lines)
│   ├── stats.py           ← 200+ statistical tests (11,495 lines)
│   ├── ml.py              ← ML training & comparison (3,298 lines)
│   ├── bayesian.py        ← Bayesian inference (451 lines)
│   ├── reliability.py     ← Weibull, KM, ALT (851 lines)
│   ├── spc.py             ← control charts, capability (2,049 lines)
│   ├── viz.py             ← Bayesian SPC visualization (1,745 lines)
│   ├── simulation.py      ← Monte Carlo (327 lines)
│   ├── endpoints_ml.py    ← from-intent, from-data, models (1,063 lines)
│   └── endpoints_data.py  ← upload, code, assistant, triage (1,930 lines)
└── analysis/
    └── __init__.py        ← re-exports from dsw/ sub-modules
```

Largest file goes from **25,035 lines** → **11,495 lines** (stats.py). Stats.py is a candidate for further splitting by test family (parametric, non-parametric, regression, etc.) but that's a separate debt item.

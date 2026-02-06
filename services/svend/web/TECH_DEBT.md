# Svend Technical Debt Log

Last updated: 2026-02-02

## Recently Fixed

### [FIXED] Bayesian probability math duplicated across three systems
- Created `core/bayesian.py` with unified `BayesianUpdater` class
- Updated `core.Hypothesis.apply_evidence()` to use BayesianUpdater
- Updated `core.Hypothesis.recalculate_probability()` to use BayesianUpdater
- Updated `agents_api.Problem._update_probabilities()` to use BayesianUpdater
- Updated `workbench.Hypothesis.update_probability()` to use BayesianUpdater
- Updated `workbench.KnowledgeGraph.update_from_evidence()` to use BayesianUpdater
- All probability math now follows: posterior_odds = prior_odds × likelihood_ratio

### [FIXED] No deprecation plan for legacy models
- Added deprecation notices to `agents_api.Problem` class
- Added deprecation notices to `workbench.Project`, `workbench.Hypothesis`, `workbench.Evidence`
- Created `MIGRATION_PLAN.md` with 4-phase consolidation roadmap
- See MIGRATION_PLAN.md for full details

### [FIXED] DSW analysis functions not modularized
- Created `agents_api/analysis/` package structure
- Backwards-compatible imports from `dsw_views.py`
- Ready for gradual extraction of analysis functions

### [FIXED] Tier definition duplicated
- Created `accounts/constants.py` with unified Tier enum
- Updated `accounts/models.py` and `forge/models.py` to import from constants
- Pricing: FREE, FOUNDER ($19), PRO ($29), TEAM ($79), ENTERPRISE ($199)

### [FIXED] Rate limiting in multiple places
- Consolidated into single `@rate_limited` decorator in `accounts/permissions.py`
- Middleware now only handles subscription status, not rate limiting
- Decorator handles auth + rate limit check + query increment

### [FIXED] LLM instantiation scattered
- Created `agents_api/llm_manager.py` with `LLMManager` singleton
- Thread-safe lazy loading
- Updated `agents_api/views.py` to use LLMManager

### [FIXED] In-memory caches lost on restart
- Created `agents_api/cache.py` with `SessionCache` class
- Database-backed with `CacheEntry` model
- Tempora integration for TTL cleanup
- `SynaraCache` and `ModelCache` convenience classes

---

## Critical (Must Fix)

### 1. Three parallel systems for hypothesis-driven investigation
**Location:** `agents_api/models.py`, `workbench/models.py`, `core/models/`
**Problem:** Three different systems for the same functionality:
- `agents_api.Problem` - JSON blob storage (legacy)
- `workbench.Project` - ORM with artifacts (partial)
- `core.Project` - Clean ORM with proper Bayesian tracking (canonical)
**Impact:** Data inconsistency, maintenance burden, confusing API surface
**Status:** Migration in progress
- ✅ BayesianUpdater unified (all three use `core.bayesian`)
- ✅ Deprecation notices added
- ✅ Migration plan documented (`MIGRATION_PLAN.md`)
- ⏳ Phase 1: Add `core_project_id` FK to `agents_api.Problem`
- ⏳ Phase 2: Map `workbench.Project` to `core.Project`
- ⏳ Phase 3: Consolidate Knowledge Graphs
**See:** `MIGRATION_PLAN.md` for full roadmap

### 2. dsw_views.py is 7263 lines
**Location:** `agents_api/dsw_views.py`
**Problem:** Massive file mixing API endpoints, business logic, caching, and ML pipeline
**Impact:** Hard to maintain, test, and navigate
**Status:** Package structure created, extraction pending
- ✅ Created `agents_api/analysis/` package with backwards-compatible imports
- ⏳ Extract `stats.py` - Statistical tests (t-tests, ANOVA, regression)
- ⏳ Extract `ml.py` - ML analysis (classification, clustering, PCA)
- ⏳ Extract `bayesian.py` - Bayesian inference (Bayes t-test, A/B, changepoint)
- ⏳ Extract `spc.py` - SPC/control charts (capability, control charts)
- ⏳ Extract `viz.py` - Visualization (scatter, heatmap, pareto)
- ⏳ Create `dsw_pipeline.py` - from_intent/from_data logic
- ⏳ Create `dsw_analyst.py` - LLM analyst assistant

### 2. Migrate existing caches to SessionCache
**Status:** Infrastructure built, migration pending
**Locations to migrate:**
- `_synara_instances` in `agents_api/synara_views.py` → use `SynaraCache`
- `_model_cache` in `agents_api/dsw_views.py` → use `ModelCache`
**Note:** LLM instances (`_shared_llm`, `_coder_llm`, `_flywheel`) stay in memory (they're model objects, not user data)

### 4. Incomplete feature
**Location:** `agents_api/dsw_views.py` line ~5840
**Problem:** `# TODO: Implement - {prompt}` comment
**Fix:** Implement or remove

---

## High (Should Fix Soon)

### 5. 185 bare `except Exception` blocks
**Problem:** Swallowing all errors makes debugging hard
**Fix:**
- Create custom exception hierarchy (PipelineError, ValidationError, etc.)
- Catch specific exceptions
- Log with context

### 6. SubscriptionMiddleware writes on every request
**Location:** `accounts/middleware.py`
**Problem:** `user.save()` called on every authenticated request to update `last_active_at`
**Impact:** Unnecessary database write overhead
**Fix:** Batch updates via async task, or only update on significant actions

### 7. No service layer
**Problem:** Business logic lives in views
**Impact:** Hard to test, reuse logic, or change presentation layer
**Fix:** Extract services (ChatService, DSWService, WorkbenchService)

### 8. Low test coverage
**Current:** 809 lines across 6 test files
**Missing tests for:** accounts, chat, inference, synara
**Target:** >80% coverage on critical paths

---

## Medium (Should Fix)

### 9. JSONField without schema validation
**Locations:** Problem.hypotheses, Problem.evidence, Workflow.steps, etc.
**Problem:** No validation of JSON structure
**Impact:** Data corruption possible, hard to query
**Fix:** Pydantic models for validation, or JSON Schema

### 10. Mixed serialization patterns
**Problem:**
- Some models use `.to_dict()` methods
- Others use DRF serializers
**Fix:** Standardize on DRF serializers everywhere

### 11. Inconsistent decorator usage
**Problem:** Mix of `@api_view`, `@gated`, `@require_auth`, `@require_enterprise`
**Fix:** Document when to use what, or consolidate

### 12. Global state thread safety
**Problem:** Some module-level caches use locks, others don't
**Fix:** Audit all globals, add locks or use thread-local storage

---

## Low (Nice to Have)

### 13. Missing API documentation
**Problem:** No OpenAPI/Swagger docs
**Fix:** Add drf-spectacular or similar

### 14. sys.path manipulation
**Location:** settings.py
**Problem:** `sys.path.insert(0, KJERNE_PATH)` is non-standard
**Fix:** Proper package structure and imports

### 15. Tight coupling api ↔ inference
**Problem:** `api/views.py` directly imports from inference module
**Fix:** Interface/dependency injection

---

## Contradictions Found

### A. ~~Tier systems don't match~~ FIXED
- Now unified in `accounts/constants.py`

### B. ~~Rate limiting implemented in multiple places~~ FIXED
- Now single `@rate_limited` decorator in `accounts/permissions.py`

### C. ~~LLM instantiation scattered~~ FIXED
- Now centralized in `agents_api/llm_manager.py`

---

## Defects

### D1. Synara instances not persisted
**Severity:** High
**Description:** User hypothesis work stored only in memory
**Reproduction:** Create hypothesis, restart server, hypothesis gone

### D2. Model cache not thread-safe initialization
**Location:** `agents_api/dsw_views.py`
**Description:** Uses lock for cache access but double-check locking pattern may have race condition on initialization

### D3. No validation on user-uploaded data structure
**Location:** `agents_api/dsw_views.py` upload_data
**Description:** User can upload malformed CSV/JSON that may cause downstream errors

---

## Simplification Opportunities

### ~~S1. Merge accounts and forge tiers~~ DONE
Single Tier enum in `accounts/constants.py`

### ~~S2. Extract rate limiting to decorator~~ DONE
Single `@rate_limited` decorator in `accounts/permissions.py`

### ~~S3. Create LLMManager~~ DONE
`LLMManager` in `agents_api/llm_manager.py`

### ~~S4. Use Redis for session caches~~ DONE (using Tempora/DB instead)
`SessionCache` in `agents_api/cache.py` with database backend

### S5. Split dsw_views.py - IN PROGRESS
Package structure created (`agents_api/analysis/`), extraction pending. See Critical #2.

### S6. Consolidate to core.Project - IN PROGRESS
Three parallel systems → one canonical `core.Project`. See Critical #1 and `MIGRATION_PLAN.md`.

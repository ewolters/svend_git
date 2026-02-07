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
**Commit:** 841af3d

---

### 2026-02-06 — P2: SPC evidence integration + re-enable agents
**Debt items:** [SPC] 3/7 endpoints, [AGENTS] Coder/Researcher disabled
**Files changed:**
- `services/svend/web/agents_api/spc_views.py` — added `problem_id` support to `statistical_summary()` and `recommend_chart()`. Updated existing 3 endpoints to use `write_context_file()` and `evidence_type="data_analysis"` for consistency.
- `services/svend/web/agents_api/urls.py` — uncommented researcher and coder agent routes
- `services/svend/web/agents_api/views.py` — added `importlib.util` shim to pre-load agent core modules (`core.intent`, `core.search`, `core.verifier`, etc.) in dependency order, fixing namespace collision with Django's `core` app. All 3 agents (researcher, coder, writer) now import successfully.
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — all files pass
- Agent imports: ResearchAgent ✓, CodingAgent ✓, WriterAgent ✓
- URL resolution: `/api/agents/researcher/` ✓, `/api/agents/coder/` ✓
- Researcher endpoint made actual search API calls (arXiv, Semantic Scholar) confirming full integration
**Commit:** 2888c32

---

### 2026-02-06 — P2: Synara DSL parser and belief engine test coverage
**Debt item:** [SYNARA] No test coverage for DSL parser or belief engine
**Files changed:**
- `services/svend/web/agents_api/tests.py` — added 46 unit tests across 9 test classes:
  - `KernelHypothesisRegionTest` (4 tests): matches_context full/partial/neutral, to_dict/from_dict roundtrip
  - `KernelEvidenceTest` (1 test): to_dict/from_dict roundtrip
  - `KernelCausalGraphTest` (8 tests): roots/terminals, upstream/downstream, ancestors/descendants, paths, link references, diamond graph, to_dict
  - `BeliefEngineComputeLikelihoodTest` (6 tests): explicit support/weaken, neutral, strength scaling, behavior alignment positive/conflicting
  - `BeliefEngineUpdatePosteriorsTest` (4 tests): supporting evidence increases posterior, normalization, clamping, evidence tracking
  - `BeliefEnginePropagationTest` (3 tests): chain propagation, no downstream, nonexistent hypothesis
  - `BeliefEngineExpansionTest` (3 tests): expansion signal generation, no expansion above threshold, empty likelihoods
  - `DSLParserBasicTest` (11 tests): comparison, string comparison, implication, quantifiers (ALWAYS/NEVER), logical AND/OR, WHEN domain, empty input, tautology detection, variable extraction
  - `DSLParserToDictTest` (3 tests): comparison/implication/quantified serialization
  - `DSLFormatTest` (3 tests): natural/formal/code formatting
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — passes
- All 46 tests pass (13 kernel + 16 belief + 17 DSL)
**Commit:** afd60e0

---

### 2026-02-06 — P2: Wire Synara LLM interface to Anthropic API
**Debt item:** [SYNARA] LLM interface stubbed — prompts generated but never call API
**Files changed:**
- `services/svend/web/agents_api/synara/llm_interface.py` — added 6 methods to `SynaraLLMInterface`:
  - `_call_llm(user, prompt)` — calls Claude via `LLMManager.chat()`, tier-aware model selection
  - `_extract_json(text)` — robust JSON extraction from LLM responses (direct parse, ```json blocks, brace matching)
  - `validate_graph_llm(user)` — full round-trip: prompt → Claude → parse → `GraphAnalysis`
  - `generate_hypotheses_llm(user, signal)` — prompt → Claude → parse → `list[HypothesisRegion]` (auto-added to graph)
  - `interpret_evidence_llm(user, evidence, result)` — prompt → Claude → plain text interpretation
  - `document_findings_llm(user, format_type)` — prompt → Claude → formatted document (summary/a3/8d/technical)
- `services/svend/web/agents_api/synara_views.py` — added 4 server-side LLM endpoints:
  - `llm_validate` — validates causal graph via Claude
  - `llm_generate_hypotheses` — generates hypotheses from expansion signal via Claude
  - `llm_interpret_evidence` — interprets evidence update via Claude
  - `llm_document` — documents findings via Claude
  - All return 503 with fallback prompt if API key not set
- `services/svend/web/agents_api/synara_urls.py` — registered 4 new URL routes under `/api/synara/<wb_id>/llm/`
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — all files pass
- URL resolution: all 4 endpoints resolve correctly
- Prompt generation + JSON extraction: tested in Django shell, all pass
- Graceful degradation: returns 503 with fallback_prompt when ANTHROPIC_API_KEY not set
**Commit:** fd16c67

---

### 2026-02-06 — P2: Researcher hallucination detection — fuzzy threshold tuning
**Debt item:** [CORE] Researcher hallucination detection needs fuzzy threshold tuning
**Files changed:**
- `services/svend/agents/agents/researcher/validator.py` — 3 improvements to `_validate_claim()`:
  1. **Windowed fuzzy matching**: `_fuzzy_similarity()` now slides a claim-sized window across source text instead of comparing whole strings. Claim "crispr can edit genes" vs 200-word source: old=0.25, new=0.71.
  2. **Bigram overlap**: new `_extract_bigrams()` adds phrase-level matching (word pairs) alongside single-term coverage. Combined score weights: 40% term coverage, 30% bigram overlap, 30% windowed similarity.
  3. **Smooth confidence curve**: replaced stepwise formula (`count * 0.3 + 0.4`) with `1 - 0.5^n` (0 sources→0.0, 1→0.5, 2→0.75, 3→0.88), blended 70/30 with best match quality.
- `services/svend/agents/researcher/validator.py` — synced duplicate copy
**Verification:**
- `py_compile` — both copies pass
- Windowed similarity: 0.706 for embedded claim (vs ~0.25 with old method)
- Bigram extraction: correct word pairs
- Confidence curve: monotonically increasing, properly scaled
- Claim validation: "CRISPR enables precise gene editing" correctly supported with confidence 0.60
**Commit:** 04fae5c

---

### 2026-02-06 — P3: Synara fallacy detection — implement pattern checks
**Debt item:** [SYNARA] Fallacy detection mostly stubbed
**Files changed:**
- `services/svend/web/agents_api/synara/logic_engine.py` — replaced `_check_fallacy_patterns()` stub (returned `[]`) with 5 structural pattern detectors:
  1. **Affirming the consequent**: shared variables between consequent/antecedent across multiple implications
  2. **Denying the antecedent**: negation of an implication's antecedent found in AST
  3. **False dichotomy**: XOR with exactly 2 options, or overlapping NEVER constraints on same variable
  4. **Hasty generalization**: universal quantifier (ALWAYS/NEVER) without WHEN domain restriction
  5. **Overgeneralization**: nested quantifiers
- Added 3 helper methods: `_collect_nodes()`, `_get_variables()`, `_contains_negation_of()`
- `services/svend/web/agents_api/tests.py` — added `FallacyDetectionTest` class with 13 tests covering all 5 fallacy types, helper methods, and `validate_hypothesis()` convenience function
**Verification:**
- `python3 manage.py check` — 0 issues
- All 13 fallacy detection tests pass
- Django shell verification: hasty generalization, XOR false dichotomy, WHEN clause suppression all correct
**Commit:** 0ba85e8

---

### 2026-02-06 — P3: Extend non-parametric battery — Friedman, Wilcoxon, Spearman
**Debt item:** [DSW] Non-parametric battery limited to Mann-Whitney + Kruskal
**Files changed:**
- `services/svend/web/agents_api/dsw_views.py` — added 3 new analysis types after Kruskal-Wallis:
  1. **Wilcoxon signed-rank** (`wilcoxon`): paired non-parametric test with effect size r, difference histogram
  2. **Friedman test** (`friedman`): repeated measures non-parametric ANOVA with Kendall's W, 3+ column checkbox selection
  3. **Spearman correlation** (`spearman`): rank correlation with p-value, 95% CI (Fisher z-transform), scatter plot
- `services/svend/web/templates/dsw.html` — added 3 options to dropdown, updated needsVar2/labels/config JS
- `services/svend/web/templates/analysis_workbench.html` — added 3 items to analysis catalog, form configs with checkboxes for Friedman
- `services/svend/web/templates/workbench_new.html` — added 3 options to More Tests dropdown
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — passes
- End-to-end: Wilcoxon p=0.0020, Friedman p=0.0003, Spearman rho=0.95 — all correct
**Commit:** bfe3956

---

### 2026-02-06 — P2: Phase 2 model cutover — read paths from core.Project FKs
**Debt item:** [CORE] Phase 2 model cutover
**Files changed:**
- `services/svend/web/agents_api/models.py` — added 6 reader methods to Problem:
  - `get_hypotheses()` → reads from core.Hypothesis FKs, falls back to JSON blob
  - `get_evidence()` → reads from core.Evidence via EvidenceLinks, falls back to JSON blob
  - `get_dead_ends()` → reads from core.Hypothesis status=rejected, falls back to JSON blob
  - `get_probable_causes()` → reads from top core.Hypothesis by probability, falls back to JSON blob
  - `get_hypothesis_count()` → ORM count or JSON len
  - `get_evidence_count()` → ORM count or JSON len
- `services/svend/web/agents_api/problem_views.py` — switched 8 read paths:
  - `problem_to_dict()` — hypotheses, evidence, dead_ends, probable_causes
  - `write_context_file()` — hypotheses, evidence, dead_ends, probable_causes
  - `problems_list()` GET — hypothesis_count, evidence_count, top_cause
  - `add_evidence()` response — updated_hypotheses, probable_causes
  - `reject_hypothesis()` response — dead_ends, probable_causes
  - `generate_hypotheses()` — prompt context + response
- `services/svend/web/agents_api/views.py` — `get_problem_context_for_agent()` switched to `get_hypotheses()`
**Design:** All methods read from core.Project FKs when `core_project` FK exists, falling back to JSON blobs when not. API response shape unchanged — templates require no modifications. Fields without core equivalents (`key_uncertainties`, `recommended_next_steps`, `bias_warnings`) stay on Problem.
**Verification:**
- `python3 manage.py check` — 0 issues
- `py_compile` — all 3 files pass
- problem_to_dict(): 5 hypotheses from core FKs, correct dict shape (id, cause, probability, status, etc.)
- write_context_file(): context JSON has 5 hypotheses + 3 probable causes from core FKs
- get_problem_context_for_agent(): hypothesis text from core.Hypothesis
- Fallback: clearing core_project falls back to JSON blob
**Commit:** 98a1628

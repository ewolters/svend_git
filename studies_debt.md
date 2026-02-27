# Studies System Debt

**Created:** 2026-02-20
**Scope:** `core/models/`, `core/views.py`, `agents_api/problem_views.py`, `templates/projects.html`

## Active

### Bugs

- [ ] **A8** (MED) Status ratcheting locks hypotheses — CONFIRMED/REJECTED can never auto-revert | Fix: document or allow re-evaluation

### Logic & Design

- [ ] **B1** (HIGH) Three parallel hypothesis/evidence systems with no sync — core, Problem JSON, workbench | Tracked in DEBT.md Phase 3
- [ ] **B2** (MED) DMAIC phases don't match non-DMAIC methodologies — Phase choices are DMAIC-only | Fix: methodology-aware phase sets
- [ ] **B4** (LOW) Manual evidence submitted via `from-code` endpoint — dummy `code`/`output` fields | Fix: add dedicated manual evidence endpoint or use `evidence_list` POST
- [ ] **B7** (LOW) No endpoint to delete knowledge graph relationships | Fix: add `relationship_detail` view

### UX/Clarity

- [ ] **C3** (LOW) Probability display with no explanation — no tooltip or help text | Fix: add contextual help
- [ ] **C4** (LOW) "Conversations" section goes nowhere — list never loads, button just redirects | Fix: wire up or remove
- [ ] **C7** (LOW) "Knowledge Graph" section is actually a dashboard summary | Fix: rename or add real graph
- [ ] **C8** (LOW) Charter form overwhelming — Six Sigma terminology alienates non-manufacturing users | Fix: progressive disclosure or domain-aware defaults

### Missing Validation

- [ ] **D6** (LOW) `problem_detail` PATCH — no field validation on enum fields
- [ ] **D7** (LOW) `dataset_list` POST — no file size or MIME type validation

## Resolved

- [x] **A1** (HIGH) Probability 0.0 silently becomes 0.5 — fixed `or 0.5` to `if x is None` in `hypothesis.py` save()
- [x] **A2** (HIGH) IDOR in `link_evidence` and `suggest_likelihood_ratio` — scoped Evidence lookup to user-accessible projects
- [x] **A3** (MED) Double rate-limit charge on `generate_hypotheses` — removed redundant manual `can_query()`/`increment_queries()`
- [x] **A4** (MED) Evidence strength slider — replaced raw LR display with plain-language labels (Negligible→Very Strong), fixed hint text
- [x] **A5** (HIGH) Archive button sends DELETE — changed to PUT with `status: 'abandoned'`, updated confirmation text
- [x] **A7** (MED) `advancePhase()` sends empty body — now auto-computes next DMAIC phase and sends `{ phase: nextPhase }`
- [x] **A9** (LOW) `EvidenceLink.applied_at` never set — added `applied_at = timezone.now()` in `apply_evidence()`
- [x] **B6** (MED) No endpoint to edit/delete individual evidence — added `evidence_detail` view (GET/PUT/DELETE) + URL
- [x] **C2** (MED) Evidence strength slider unintuitive — replaced with plain-language labels
- [x] **C5** (MED) Status terminology mismatch — aligned CSS to backend values (active/confirmed/rejected/uncertain/merged)
- [x] **C6** (MED) Direction value mismatch — changed "weakens" to "opposes" in UI radio, CSS, and JS
- [x] **D1** (MED) `Project.resolution_confidence` — added MinValueValidator(0.0)/MaxValueValidator(1.0)
- [x] **D2** (MED) `Evidence.confidence` — added MinValueValidator(0.0)/MaxValueValidator(1.0)
- [x] **D3** (MED) `EvidenceLink.likelihood_ratio` — added MinValueValidator(0.001)
- [x] **D4** (MED) `confirmation_threshold`/`rejection_threshold` — added validators (0.5-1.0 / 0.0-0.5 respectively)
- [x] **A6** (MED) Edit project stub — implemented edit modal reusing create form, `editProject()` pre-fills and switches to PUT mode
- [x] **C1** (MED) "Workbenches" section confusing — renamed to "Methodology & Phase", removed "+ Add Workbench" button
- [x] **B3** (MED) No rate limiting on core endpoints — added `@rate_limited` to computation-heavy endpoints (recalculate, check_consistency, review_design_execution); DRF `UserRateThrottle` already covers all `@api_view` endpoints
- [x] **B8** (LOW) `project_hub` count/list mismatch — split into unsliced base querysets for `.count()` and sliced for display
- [x] **B5** (MED) Hardcoded UNCERTAIN range — derived from thresholds: `(rejection+0.5)/2` to `(confirmation+0.5)/2`
- [x] **D5** (LOW) `advance_phase()` — added phase-order validation preventing forward skips (backwards allowed)

All resolved items: commit pending
Migration: `core/migrations/0008_alter_evidence_confidence_and_more.py` — applied

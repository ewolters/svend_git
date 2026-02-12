# Technical Debt Tracker

Track technical debt here. Review weekly.

## Format
```
[SERVICE] Description | Added: DATE | Priority: P1/P2/P3
```

## Active Debt

### Integration — Module ↔ Hypothesis Pipeline

[AGENTS] Writer and Editor agents have no project/evidence integration | Added: 2026-02-06 | Priority: P3

### Data Model Migration

[CORE] Phase 3 pending — remove JSON blobs entirely, drop Problem.hypotheses/evidence/dead_ends/probable_causes fields. Blocked until all Problems have core_project FK | Added: 2026-02-06 | Priority: P3

### Git / Repo Hygiene

[REPO] 50+ hardcoded /home/eric/ paths across Python files — works on prod server but breaks portability | Added: 2026-02-06 | Priority: P3
[REPO] Duplicate agents/agents/ directory (85 files duplicating agents/) | Added: 2026-02-06 | Priority: P3

### Existing Debt (carried forward)

[CORE] Writer template validation should support nested sections | Added: 2024-01-27 | Priority: P3

### Competitive Gaps (vs Minitab/JMP)

[DOE] No split-plot or mixture designs | Added: 2026-02-06 | Priority: P3
[DOE] Single response optimization only — no multi-response | Added: 2026-02-06 | Priority: P3
[DOE] No acceptance sampling for variable data (normal distribution plans) | Added: 2026-02-07 | Priority: P3

## Resolved

[DSW] No mixed-effects / multi-level modeling | Added: 2026-02-06 | Resolved: 2026-02-07 | Commit: pending
[FORECAST] Limited to random walk MC, SMA, exponential smoothing — no Prophet, TBATS, or seasonal methods | Added: 2026-02-06 | Resolved: 2026-02-07 | Commit: pending
[SPC] No multivariate control charts | Added: 2026-02-06 | Resolved: 2026-02-07 | Commit: pending
[DSW] No survival analysis (Cox proportional hazards, Kaplan-Meier) | Added: 2026-02-07 | Resolved: 2026-02-07 | Commit: pending
[DSW] No discriminant analysis (LDA/QDA) | Added: 2026-02-07 | Resolved: 2026-02-07 | Commit: pending
[DSW] No acceptance sampling plans | Added: 2026-02-07 | Resolved: 2026-02-07 | Commit: pending

[REPO] svend.db tracked in git with user emails | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: 9c9396e
[REPO] .kjerne/snapshots/*.tar.gz binary files in git (4.3MB) | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: 9c9396e
[DSW] No integration with Projects/Evidence — analysis results never become hypothesis evidence | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: 0eef3fb
[EXPERIMENTER] Only 2/9 endpoints create evidence — extended to 6/9 (power, design, contour, optimize + existing full, analyze) | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: 0eef3fb
[CORE] agents_api.Problem → core.Project Phase 1 dual-write — FK field, sync methods, 6 view write paths, existing data migrated | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: f4fb8db
[SYNARA] In-memory only → persisted to core.Project.synara_state JSONField. Cache + DB backed, 9 endpoints auto-save | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: 841af3d
[SPC] Extended evidence to 5/7 endpoints (summary, recommend added; upload_data/chart_types are read-only, intentionally excluded) | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: 2888c32
[AGENTS] Researcher and Coder agents re-enabled — fixed core.intent namespace collision with importlib shim | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: 2888c32
[SYNARA] DSL parser and belief engine test coverage — 46 tests across kernel, belief engine, and DSL parser | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: afd60e0
[SYNARA] LLM interface wired — 4 server-side endpoints calling Claude via LLMManager, graceful 503 fallback | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: fd16c67
[CORE] Researcher hallucination detection — windowed fuzzy matching, bigram overlap, smooth confidence curve | Added: 2024-01-27 | Resolved: 2026-02-06 | Commit: 04fae5c
[SYNARA] Fallacy detection — 5 pattern checks (affirming consequent, denying antecedent, false dichotomy, hasty generalization, overgeneralization) + 13 tests | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: 0ba85e8
[DSW] Non-parametric battery extended — Wilcoxon signed-rank, Friedman test, Spearman correlation with p-values/CIs | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: bfe3956
[CORE] Phase 2 model cutover — all read paths now use core.Project FKs with JSON fallback. 8 read paths switched, API shape unchanged | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: 98a1628

---
*Last reviewed: 2026-02-06*

# Technical Debt Tracker

Track technical debt here. Review weekly.

## Format
```
[SERVICE] Description | Added: DATE | Priority: P1/P2/P3
```

## Active Debt

### Integration — Module ↔ Hypothesis Pipeline

[AGENTS] Writer and Editor agents have no project/evidence integration | Added: 2026-02-06 | Priority: P3

### Synara Belief Engine

[SYNARA] Fallacy detection mostly stubbed — _check_fallacy_patterns() returns empty list. Affirming consequent, denying antecedent, false dichotomy undetected | Added: 2026-02-06 | Priority: P3

### Data Model Migration

[CORE] Phase 1 dual-write active — new Problems auto-create core.Project + sync hypotheses/evidence. Phase 2 (remove JSON blobs, full cutover) still pending | Added: 2026-02-06 | Priority: P2

### Git / Repo Hygiene

[REPO] 50+ hardcoded /home/eric/ paths across Python files — works on prod server but breaks portability | Added: 2026-02-06 | Priority: P3
[REPO] Duplicate agents/agents/ directory (85 files duplicating agents/) | Added: 2026-02-06 | Priority: P3

### Existing Debt (carried forward)

[CORE] Writer template validation should support nested sections | Added: 2024-01-27 | Priority: P3

### Competitive Gaps (vs Minitab/JMP)

[DSW] No mixed-effects / multi-level modeling | Added: 2026-02-06 | Priority: P3
[DSW] Non-parametric battery limited to Mann-Whitney + Kruskal — missing Friedman, Wilcoxon signed-rank, Spearman | Added: 2026-02-06 | Priority: P3
[DOE] No split-plot or mixture designs | Added: 2026-02-06 | Priority: P3
[DOE] Single response optimization only — no multi-response | Added: 2026-02-06 | Priority: P3
[SPC] No multivariate control charts | Added: 2026-02-06 | Priority: P3
[FORECAST] Limited to random walk MC, SMA, exponential smoothing — no Prophet, TBATS, or seasonal methods | Added: 2026-02-06 | Priority: P3

## Resolved

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

---
*Last reviewed: 2026-02-06*

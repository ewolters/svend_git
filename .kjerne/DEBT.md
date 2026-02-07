# Technical Debt Tracker

Track technical debt here. Review weekly.

## Format
```
[SERVICE] Description | Added: DATE | Priority: P1/P2/P3
```

## Active Debt

### Integration — Module ↔ Hypothesis Pipeline

[SPC] Only 3/7 endpoints (control_chart, capability_study, analyze_uploaded) create evidence — extend to remaining 4 (upload_data, summary, recommend, chart_types) | Added: 2026-02-06 | Priority: P2
[AGENTS] Coder agent disabled at router level (url commented out) — has implementation, needs re-enable + project integration | Added: 2026-02-06 | Priority: P2
[AGENTS] Researcher agent disabled at router level — has full implementation with add_finding_to_problem(), needs re-enable | Added: 2026-02-06 | Priority: P2
[AGENTS] Writer and Editor agents have no project/evidence integration | Added: 2026-02-06 | Priority: P3

### Synara Belief Engine

[SYNARA] In-memory only — state lost on server restart. Needs persistence to Django ORM (target: core.Project FK model, not agents_api.Problem JSON blobs) | Added: 2026-02-06 | Priority: P1
[SYNARA] LLM interface stubbed — prompts generated but never call API. Wire to Anthropic/Qwen | Added: 2026-02-06 | Priority: P2
[SYNARA] Fallacy detection mostly stubbed — _check_fallacy_patterns() returns empty list. Affirming consequent, denying antecedent, false dichotomy undetected | Added: 2026-02-06 | Priority: P3
[SYNARA] No test coverage for DSL parser or belief engine | Added: 2026-02-06 | Priority: P2

### Data Model Migration

[CORE] agents_api.Problem (JSON blobs) marked DEPRECATED — migrate to core.Project with proper FK relationships (Hypothesis, Evidence, EvidenceLink models). Migration planned but not executed | Added: 2026-02-06 | Priority: P1

### Git / Repo Hygiene

[REPO] 50+ hardcoded /home/eric/ paths across Python files — works on prod server but breaks portability | Added: 2026-02-06 | Priority: P3
[REPO] Duplicate agents/agents/ directory (85 files duplicating agents/) | Added: 2026-02-06 | Priority: P3

### Existing Debt (carried forward)

[CORE] Researcher hallucination detection needs fuzzy threshold tuning | Added: 2024-01-27 | Priority: P2
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
[DSW] No integration with Projects/Evidence — analysis results never become hypothesis evidence | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: (pending)
[EXPERIMENTER] Only 2/9 endpoints create evidence — extended to 6/9 (power, design, contour, optimize + existing full, analyze) | Added: 2026-02-06 | Resolved: 2026-02-06 | Commit: (pending)

---
*Last reviewed: 2026-02-06*

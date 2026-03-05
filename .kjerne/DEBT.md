# Technical Debt Tracker

Track technical debt here. Review weekly.

## Format
```
[SERVICE] Description | Added: DATE | Priority: P1/P2/P3
```

## Active Debt

### Code Organization

[DSW] dsw_views.py still 24K lines — HTTP endpoints remain in monolith. Analysis engines extracted to dsw/ package (36K lines across 14 modules: stats 14K, bayesian 3.5K, ml 3.3K, spc 3.2K, common 2.6K, d_type 2.3K, viz 2K, endpoints_data 2K, endpoints_ml 1K, reliability 1K, simulation 0.8K). Next step: extract remaining HTTP handlers from dsw_views.py into dsw/endpoints_*.py | Added: 2026-02-16 | Priority: P2

[DSW] stats.py is 14K lines — largest single module in the dsw/ package. Contains ~100 statistical tests. Should split into stats_parametric.py, stats_nonparametric.py, stats_bayesian.py, stats_multivariate.py | Added: 2026-03-03 | Priority: P3

[DSW] pbs_engine.py is 3.5K lines — Process Belief System engine. Self-contained, not urgent, but could separate into pbs/ package if it grows further | Added: 2026-03-03 | Priority: P3

[TEMPLATE] workbench_new.html is 10.9K lines — single-file monolith with inline CSS (~1100 lines) and JS (~8000 lines). Not practical to split without a build system, but could extract CSS to a standalone .css file served via WhiteNoise | Added: 2026-03-03 | Priority: P3

### Data Model Migration

[CORE] Phase 3 pending — remove JSON blobs entirely, drop Problem.hypotheses/evidence/dead_ends/probable_causes fields. Blocked until all Problems have core_project FK | Added: 2026-02-06 | Priority: P3

[STY] ~~24 BooleanField fields missing `is_` prefix (DAT-001 §7.4)~~ **RESOLVED** — 27 fields renamed across agents_api, accounts, api, chat, core, workbench, syn/sched, tempora with db_column preservation (zero DB migration). All references updated (~200 across ~30 files). Compliance check at 0 violations. CR 2c9dce5c. | Added: 2026-03-03 | Resolved: 2026-03-04

### Git / Repo Hygiene

[REPO] ~~Hardcoded /home/eric/ paths~~ **RESOLVED** — all code files (.py, .sh, .service) now use portable paths ($HOME, %h, Path(__file__), SCRIPT_DIR). Only doc/markdown references remain (historical). | Added: 2026-02-06 | Resolved: 2026-03-03 | Commit: pending

[REPO] ~~agents/agents/ nested directory~~ **RESOLVED** — flattened to agents/. Updated settings.py, learn_views.py imports, __init__.py. Cleaned 20+ dead sys.path.insert calls. | Added: 2026-02-06 | Resolved: 2026-03-03 | Commit: pending

[REPO] ~~Root core/ misplaced~~ **RESOLVED** — moved core/ (intent, search, reasoning, etc.) into agents/agent_core/ (renamed to avoid shadowing Django's web/core/ app). All imports updated from `core.X` to `agent_core.X`. | Added: 2026-03-03 | Resolved: 2026-03-03 | Commit: pending

[REPO] ~~services/scrub/ duplicate~~ **RESOLVED** — removed standalone services/scrub/ (older copy). Canonical version is agents/scrub/. No code imported from services/scrub/. | Added: 2026-03-03 | Resolved: 2026-03-03 | Commit: pending

[REPO] ~~Root-level file clutter~~ **RESOLVED** — planning/debt docs moved to docs/planning/, strategy HTML to docs/reference/, DEBT-001.md to .kjerne/, shell scripts and service files to web/ops/. | Added: 2026-03-03 | Resolved: 2026-03-03 | Commit: pending

[REPO] ~~Dead test imports~~ **RESOLVED** — tests.py had imports from empty agents/core/ and nonexistent agents/editor/. Rewrote to test actual importable modules. | Added: 2026-03-03 | Resolved: 2026-03-03 | Commit: pending

[REPO] ~~Legacy "multi-agent"/"neuro-symbolic" references~~ **RESOLVED** — updated docstrings in views.py and core/reasoning.py. | Added: 2026-03-03 | Resolved: 2026-03-03 | Commit: pending

[REPO] shared_context/ directory contains 4 JSON problem files committed to repo — should be in .gitignore or managed separately | Added: 2026-03-03 | Priority: P3

[REPO] services/forge/ standalone service appears unused — Django forge app at web/forge/ has replaced it. Verify and remove if dead. | Added: 2026-03-03 | Priority: P3

### Integration — Module <> Hypothesis Pipeline

[AGENTS] Writer and Editor agents have no project/evidence integration | Added: 2026-02-06 | Priority: P3

### Infrastructure

[INFRA] Off-site backup sync — encrypted backups sit on same machine as data. Push AES-256 encrypted dumps to Backblaze B2 or similar | Added: 2026-02-13 | Priority: P3

### Security — High (Audit 2026-02-20)

[SEC] Three parallel Hypothesis/Evidence systems (workbench, core, Problem) with no sync — related to Phase 3 migration above | Added: 2026-02-20 | Priority: P2

[SEC] Pickle in CacheEntry.value BinaryField — RCE vector if DB compromised | Added: 2026-02-20 | Priority: P1 | *Mitigated: SessionCache now rejects pickle entries. Full fix: remove BinaryField, migrate to JSONField*

### Security — Medium (Audit 2026-02-20)

[SEC] Workbench conversations unencrypted — chat.Message uses EncryptedTextField but workbench stores plaintext JSON | Added: 2026-02-20 | Priority: P2

[SEC] Tempora node ID hardcoded as svend-1 — all instances would claim same ID | Added: 2026-02-20 | Priority: P3
[SEC] Placeholder tenant_id=UUID(int=0) in replication.py:520 | Added: 2026-02-20 | Priority: P3
[SEC] No TLS on Tempora cluster communication despite hardening config | Added: 2026-02-20 | Priority: P3
[SEC] Error messages leak internals — str(e) still in ~50 lower-risk locations (top 17 fixed) | Added: 2026-02-20 | Priority: P3
[SEC] Mass assignment on status fields — directly settable via PATCH, bypassing workflows | Added: 2026-02-20 | Priority: P3

### Statistical Correctness (Audit 2026-02-20)

[STATS] Synara belief propagation is a heuristic, not proper Bayesian network inference — belief.py:183-250 | Added: 2026-02-20 | Priority: P3

### Competitive Gaps (vs Minitab/JMP)

[DOE] No split-plot or mixture designs | Added: 2026-02-06 | Priority: P3
[DOE] Single response optimization only — no multi-response | Added: 2026-02-06 | Priority: P3
[DOE] No acceptance sampling for variable data (normal distribution plans) | Added: 2026-02-07 | Priority: P3

## Resolved

### Resolved 2026-03-03

[DSW] SPC Chart.js rendering path — I-MR, X-bar R, P, C were on old Chart.js `openSPCDialog` path while all others used Plotly DSW path. Migrated all 4 to `openSPCExtDialog`, removed dead Chart.js code (openSPCDialog, runControlChart, renderControlChartOutput, drawControlChart, renderCapabilityOutput) | Added: 2026-03-03 | Resolved: 2026-03-03 | Commit: 6b6380c

[DSW] SPC charts inconsistent — heights, markers, colors, narratives varied across chart types. Standardized all to X-bar S reference: height 290, green markers size 5, red dashed limits, range sliders, narratives, guide_observation | Added: 2026-03-02 | Resolved: 2026-03-02 | Commit: 6b6380c

[DSW] PBS analyses had no narratives — all 9 PBS analyses (full, belief, evidence, e-detector, predictive, adaptive, cpk, cpk_traj, health) now have result["narrative"] + result["education"] | Added: 2026-03-02 | Resolved: 2026-03-02 | Commit: 6b6380c

[DSW] D-type analyses had no/incomplete narratives — all 6 D-type analyses (d-chart, d-cpk, d-nonnorm, d-equiv, d-sig, d-multi) now have narrative + education | Added: 2026-03-02 | Resolved: 2026-03-02 | Commit: 6b6380c

[DSW] RCA button only on Nelson rule violations — OOC points on charts without Nelson rules (P, U, EWMA, Laney, MA, MEWMA) couldn't trigger RCA. Fixed isOOC detection to also check trace name | Added: 2026-03-02 | Resolved: 2026-03-02 | Commit: 6b6380c

[DSW] No chart expand — added universal expand button (hover-reveal) on all Plotly charts with full-viewport overlay | Added: 2026-03-03 | Resolved: 2026-03-03 | Commit: 6b6380c

[DSW] No mobile responsiveness — added tablet (768px) and phone (480px) breakpoints. Charts single-column on phone, modals full-screen, ribbon horizontal scroll, touch-friendly targets | Added: 2026-03-03 | Resolved: 2026-03-03 | Commit: 6b6380c

[DSW] Expand overlay transparent background — charts rendered with transparent paper/plot bgcolor against semi-transparent overlay. Added opaque container background | Added: 2026-03-03 | Resolved: 2026-03-03 | Commit: 6b6380c

[REPO] .sql backup files not gitignored — 3 db_backup_iso_*.sql files in repo root. Added *.sql to .gitignore | Added: 2026-03-03 | Resolved: 2026-03-03 | Commit: 6b6380c

[STATS] DPMO uses only upper tail — spc.py:~219, underestimates defect rate by ~50% | Added: 2026-02-20 | Resolved: 2026-02-20 | *Reviewed: two-sided case already uses both tails correctly*

### Resolved 2026-02-22

[INFRA] RCA critique views gated wrong — changed from @gated_paid to @require_enterprise | Added: 2026-02-13 | Resolved: 2026-02-22 | Commit: 6b6380c
[SEC] LLM prompt injection — XML-wrapped all user inputs in Claude/Qwen prompts, added boundary instructions and 2000-char limits | Added: 2026-02-20 | Resolved: 2026-02-22 | Commit: 6b6380c
[SEC] CSRF globally disabled — removed @csrf_exempt from 278 views, added auto-injecting fetch wrapper. Only Stripe webhook exempt | Added: 2026-02-20 | Resolved: 2026-02-22 | Commit: 6b6380c

### Resolved 2026-02-20

[SEC] Zombie task reaper — reap_zombie_tasks management command | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] HTML injection in PDF export — sanitize before wkhtmltopdf | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] DB migrations applied — Evidence index, Message index, Hypothesis/FMEA validators | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Board.save() version increment — atomic F() expression | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Gunicorn max_requests=1000 with jitter | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Hypothesis probability validation (0-1) — validators + clamping | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] FMEA S/O/D score validation (1-10) — validators + clamping | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] LLM resource cleanup — unload() classmethod | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Missing DB indexes — Evidence.project + Message compound | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Django LOGGING — RotatingFileHandler | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] wkhtmltopdf SSRF — --disable-local-file-access | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] add_finding_to_problem() dual-write to core.Evidence | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Error message sanitization — top 17 str(e) leaks fixed | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] eval() in simulation — np module removed, AST blocked | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Content-Disposition header injection — filenames sanitized | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Registration rate-limited 5/hour | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Password validators applied | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] TEMPORA_CLUSTER_SECRET via HMAC | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Forge tier limit bypass — session-auth now subject to limits | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Forge job IDOR — filter by user | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] eval() in simulation.py — removed raw np, AST blocked | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Unbounded caches — bounded to 128-256 entries | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] DSW upload 50MB limit | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] File type blocklist extended (+18 extensions) | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] get_context_file path leak removed | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Claude API timeout=120.0 | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Rate limit bypass on DB failure — returns error | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Hardcoded debug/credentials defaults to False/empty | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Conversation history injection — sanitize roles, cap length | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Path traversal in data_id — regex validation | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] RCE via exec() __import__ — removed from sandbox | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] RCE via eval() — replaced with pd.eval(engine='numexpr') | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] RCE via pickle.loads — SessionCache JSON-only | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] IDOR on Synara — _resolve_project() takes user param | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] IDOR on Whiteboard writes — owner/participant check | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Missing auth on problems_list/detail — @gated added | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Open redirect in email_track_click — domain allowlist | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Race conditions in counters — F() expressions | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Broken dual-write — field mappings fixed | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Broken email verification — hash before DB query | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] IDOR in DSW views — user= filter added | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] SPC cache_key IDOR — user_id prefix validation | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] Hoshin update/delete_site — permission checks added | Resolved: 2026-02-20 | Commit: 6b6380c
[SEC] FMEA list/promote wrong field — user= changed to owner= | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] Two-sample t-test CI — Welch-Satterthwaite df | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] SPC Pp/Ppk — MR-bar/d2 within-subgroup sigma | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] Cholesky — ridge fallback for collinear data | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] JZS Bayes Factor — Rouder et al. (2009) integral | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] Correlation BF — Ly et al. (2016) integral | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] Synara forced normalization — independent Bayesian updates | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] Q-Q quantiles — rank-based (i-0.5)/n | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] Kruskal-Wallis epsilon squared — clamped [0,1] | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] Sign Test CI — binom.ppf index corrected | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] Wilcoxon Z-score — direct from statistic | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] JZS BF integrand — r^2 per Rouder (2009) | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] bayes_anova — BIC Bayes Factor added | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] Contour argmax — np.unravel_index | Resolved: 2026-02-20 | Commit: 6b6380c
[DOE] Fractional factorial generators — Montgomery patterns | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] bayes_changepoint — BIC scan | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] Regression SE — pinv fallback | Resolved: 2026-02-20 | Commit: 6b6380c
[STATS] Logistic regression Fisher info — ridge fallback | Resolved: 2026-02-20 | Commit: 6b6380c
[DSW] Output standardization — 93+ sections, 4 Bayesian analyses styled | Resolved: 2026-02-20 | Commit: 6b6380c

### Resolved 2026-02-07

[DSW] No mixed-effects / multi-level modeling | Resolved: 2026-02-07 | Commit: 6b6380c
[FORECAST] Limited methods — no Prophet, TBATS, seasonal | Resolved: 2026-02-07 | Commit: 6b6380c
[SPC] No multivariate control charts | Resolved: 2026-02-07 | Commit: 6b6380c
[DSW] No survival analysis | Resolved: 2026-02-07 | Commit: 6b6380c
[DSW] No discriminant analysis | Resolved: 2026-02-07 | Commit: 6b6380c
[DSW] No acceptance sampling | Resolved: 2026-02-07 | Commit: 6b6380c

### Resolved 2026-02-06

[REPO] svend.db tracked in git with user emails | Resolved: 2026-02-06 | Commit: 9c9396e
[REPO] .kjerne/snapshots/*.tar.gz binaries in git | Resolved: 2026-02-06 | Commit: 9c9396e
[DSW] No integration with Projects/Evidence | Resolved: 2026-02-06 | Commit: 0eef3fb
[EXPERIMENTER] Only 2/9 endpoints create evidence — extended to 6/9 | Resolved: 2026-02-06 | Commit: 0eef3fb
[CORE] Phase 1 dual-write — FK field, sync methods, 6 view write paths | Resolved: 2026-02-06 | Commit: f4fb8db
[SYNARA] In-memory → persisted to core.Project.synara_state | Resolved: 2026-02-06 | Commit: 841af3d
[SPC] Extended evidence to 5/7 endpoints | Resolved: 2026-02-06 | Commit: 2888c32
[AGENTS] Researcher/Coder re-enabled — fixed namespace collision | Resolved: 2026-02-06 | Commit: 2888c32
[SYNARA] DSL parser + belief engine 46 tests | Resolved: 2026-02-06 | Commit: afd60e0
[SYNARA] LLM interface wired — 4 Claude endpoints | Resolved: 2026-02-06 | Commit: fd16c67
[CORE] Researcher hallucination detection | Resolved: 2026-02-06 | Commit: 04fae5c
[SYNARA] Fallacy detection — 5 patterns + 13 tests | Resolved: 2026-02-06 | Commit: 0ba85e8
[DSW] Non-parametric battery extended | Resolved: 2026-02-06 | Commit: bfe3956
[CORE] Phase 2 model cutover — all read paths use core.Project FKs | Resolved: 2026-02-06 | Commit: 98a1628
[CORE] Writer template validation nested sections | Added: 2024-01-27 | Resolved: 2026-02-06

---
*Last reviewed: 2026-03-03*

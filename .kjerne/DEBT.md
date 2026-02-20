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

### Code Organization

[DSW] dsw_views.py monolith (25,035 lines) — split into dsw/ package in progress, analysis engines extracted, HTTP endpoints still in monolith. Plan: [dsw_future_plan.md](../services/svend/reference_docs/dsw_future_plan.md) | Added: 2026-02-16 | Priority: P2

### Infrastructure

[INFRA] Off-site backup sync — encrypted backups sit on same machine as data. Push AES-256 encrypted dumps to Backblaze B2 or similar for fire/theft/disk-failure resilience | Added: 2026-02-13 | Priority: P3
[INFRA] RCA critique views (rca_views.py) use @gated_paid instead of @require_enterprise — Founder/Pro/Team users can hit Anthropic API through critique/evaluate_chain/critique_countermeasure | Added: 2026-02-13 | Priority: P2

### Security — Critical (Audit 2026-02-20)

*All 10 critical items resolved — see Resolved section below.*

### Security — High (Audit 2026-02-20)

[SEC] CSRF globally disabled — CsrfExemptSessionAuthentication on all REST Framework endpoints, mitigated only by SameSite=Lax | Added: 2026-02-20 | Priority: P2
[SEC] LLM prompt injection — user input directly interpolated into Claude/Qwen prompts across multiple views | Added: 2026-02-20 | Priority: P2
[SEC] Three parallel Hypothesis/Evidence systems (workbench, core, Problem) with no sync between them | Added: 2026-02-20 | Priority: P2
[SEC] Pickle in CacheEntry.value BinaryField — RCE vector if DB compromised | Added: 2026-02-20 | Priority: P1 | *Mitigated: SessionCache now rejects pickle entries*

### Security — Medium (Audit 2026-02-20)

[SEC] Workbench conversations unencrypted — chat.Message uses EncryptedTextField but workbench stores plaintext JSON | Added: 2026-02-20 | Priority: P2
[SEC] Tempora node ID hardcoded as svend-1 — all instances would claim same ID | Added: 2026-02-20 | Priority: P3
[SEC] Placeholder tenant_id=UUID(int=0) in replication.py:520 | Added: 2026-02-20 | Priority: P3
[SEC] No TLS on Tempora cluster communication despite hardening config | Added: 2026-02-20 | Priority: P3
[SEC] Error messages leak internals — str(e) still in ~50 lower-risk locations (top 17 fixed) | Added: 2026-02-20 | Priority: P3
[SEC] Mass assignment on status fields — directly settable via PATCH, bypassing workflows | Added: 2026-02-20 | Priority: P3

### Statistical Correctness (Audit 2026-02-20)

[STATS] DPMO uses only upper tail — spc.py:~219, underestimates defect rate by ~50% | Added: 2026-02-20 | Priority: P2 | *Reviewed: two-sided case already uses both tails correctly*
[STATS] Synara belief propagation is a heuristic, not proper Bayesian network inference — belief.py:183-250 | Added: 2026-02-20 | Priority: P3

### Existing Debt (carried forward)

[CORE] Writer template validation should support nested sections | Added: 2024-01-27 | Priority: P3

### Competitive Gaps (vs Minitab/JMP)

[DOE] No split-plot or mixture designs | Added: 2026-02-06 | Priority: P3
[DOE] Single response optimization only — no multi-response | Added: 2026-02-06 | Priority: P3
[DOE] No acceptance sampling for variable data (normal distribution plans) | Added: 2026-02-07 | Priority: P3

## Resolved

[SEC] Zombie task reaper — reap_zombie_tasks management command for stale RUNNING tasks | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] HTML injection in PDF export — sanitize script/iframe/event handlers before wkhtmltopdf | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] DB migrations applied — Evidence index, Message index, Hypothesis validators, FMEA validators | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Board.save() version increment — atomic F() expression in save() | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Gunicorn max_requests — added max_requests=1000 with jitter | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Hypothesis probability validation (0-1) — field validators + save() clamping | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] FMEA S/O/D score validation (1-10) — field validators + save() clamping | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] LLM resource cleanup — added unload() classmethod to LLMManager | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Missing DB indexes — Evidence.project + Message(conversation, created_at) compound indexes | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Django LOGGING — RotatingFileHandler for svend.log + security.log | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] wkhtmltopdf SSRF — --enable-local-file-access removed, replaced with --disable-local-file-access | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] add_finding_to_problem() dual-write — now creates core.Evidence alongside JSON blob | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Error message sanitization — top 17 str(e) leaks replaced with generic messages | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] eval() in simulation — np module removed, attribute access blocked in AST | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Content-Disposition header injection — sanitize filenames in 7 files | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Registration endpoint rate-limited — 5/hour via RegistrationThrottle | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Password validators applied — Django validate_password() on register + change_password | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] TEMPORA_CLUSTER_SECRET — derived via HMAC instead of reusing SECRET_KEY | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Forge tier limit bypass — session-auth users now subject to tier limits | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Forge job IDOR — session-auth queries filter by user instead of api_key=None | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] eval() in simulation.py — removed raw np module, blocked attribute access in AST | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Unbounded in-memory caches — bounded _parsed_data_cache(256), _synara_cache(128), _interview_sessions(128) | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] DSW upload size limit — 50 MB max on upload_data endpoint | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] File type blocklist — extended with 18 additional dangerous extensions | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] get_context_file path leak — removed filesystem path from API response | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Claude API timeout — added timeout=120.0 to client.messages.create() | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Rate limit bypass on DB failure — now returns error instead of continuing | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Hardcoded debug/credentials — changed defaults to False/empty | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Conversation history injection — sanitize roles to user/assistant only, cap content length | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Path traversal in data_id — validate with regex ^data_[a-f0-9]+$ | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] Two-sample t-test CI — use Welch-Satterthwaite df instead of pooled | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] SPC Pp/Ppk — Cp/Cpk now uses MR-bar/d2 within-subgroup sigma for individuals | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] Cholesky decomposition — added ridge fallback for collinear data | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] IDOR in DSW views — added user= filter to Project.objects.get() in dsw_views.py | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] SPC cache_key IDOR — validate user_id prefix in cache key against request.user | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Hoshin update_site/delete_site — added _check_site_write() and _is_site_admin() permission checks | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] FMEA list/promote wrong field name — changed user= to owner= matching FMEA model | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] JZS Bayes Factor — replaced ad-hoc formula with Rouder et al. (2009) numerical integral | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] Correlation Bayes Factor — replaced ad-hoc formula with Ly et al. (2016) integral | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] Synara forced normalization — replaced sum-to-1 normalization with independent Bayesian updates | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] RCE via exec() with __import__ in endpoints_data.py — removed __import__, getattr, setattr, hasattr from sandbox builtins | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] RCE via eval() in endpoints_data.py — replaced eval() calculator with pd.eval(engine='numexpr') | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] RCE via pickle.loads in cache.py — SessionCache now JSON-only, rejects pickle entries | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] IDOR on Synara endpoints — _resolve_project() now accepts user param, all callers pass request.user | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] IDOR on Whiteboard write endpoints — update_board and export_hypotheses now require owner/participant status | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Missing auth on problems_list/problem_detail — added @gated decorator | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Open redirect in email_track_click — added domain allowlist validation (svend.ai only) | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Race conditions in increment_queries/record_usage — replaced with F() expressions | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Broken dual-write ensure_core_project/sync_hypothesis_to_core — fixed field mappings to match core.Project/Hypothesis schema | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[SEC] Broken email verification lookup — hash token before DB query | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending

[STATS] Q-Q plot quantiles — replaced linspace(0.01,0.99) with rank-based (i-0.5)/n | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] Kruskal-Wallis ε² negative — clamped to [0, 1] | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] Sign Test CI off-by-one — binom.ppf index corrected | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] Wilcoxon Z-score — compute from test statistic directly, not back-computed from p-value | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] JZS BF integrand — r² moved from exponential to n_eff term per Rouder (2009) | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] bayes_anova — added BIC-approximated Bayes Factor (was pure frequentist) | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] Contour plot argmax — replaced axis decomposition with np.unravel_index | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[DOE] Fractional factorial generators — replaced arbitrary modular confounding with standard Montgomery patterns | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] bayes_changepoint — replaced CUSUM heuristic with BIC-approximated Bayes Factor scan | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] Regression SE — pinv fallback + collinearity warning on singular X'X | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending
[STATS] Logistic regression Fisher info — ridge fallback + warning on singular information matrix (binary + nominal) | Added: 2026-02-20 | Resolved: 2026-02-20 | Commit: pending

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
*Last reviewed: 2026-02-20*

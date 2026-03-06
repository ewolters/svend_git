# QMS-GAP-PLAN: NASA-Grade QMS Gap Closure

**Initiative:** INIT-012
**Created:** 2026-03-06
**Owner:** Eric
**Window:** 60 days (target: 2026-05-05)
**Target Quarter:** Q2-2026

## Executive Summary

Systematic closure of identified QMS gaps toward SOC 2 observation readiness.
Covers risk registry, verification independence, coverage ratcheting, severity-weighted
compliance scoring, and quality record creation. Calibration system hardening is a
**parallel workstream owned elsewhere** — excluded from this plan.

---

## 1. Initiative & CR Registry

### INIT-012: NASA-Grade QMS Gap Closure

| Feature | Title | Status |
|---------|-------|--------|
| FEAT-090 | RISK-001: Persistent Risk Registry — model, views, compliance check | planned |
| FEAT-091 | IVR-001: Mechanical veto in validate_for_transition() | planned |
| FEAT-092 | CMP: Severity-weighted compliance scoring | planned |
| FEAT-093 | QUAL: Rejected analysis creates quality record | planned |
| FEAT-094 | T1-COV: Statistical core symbol coverage push (35% → 70%) | planned |
| FEAT-095 | T2-COV: Revenue path endpoint smoke tests (8.2% → 50%) | planned |
| FEAT-096 | T3-COV: Feature surface coverage push (18.1% → 30%) | planned |
| FEAT-097 | ARCH-001: Complexity governance | planned |
| FEAT-098 | QMS Gap Closure dashboard tab on internal compliance | planned |

### Change Requests

| CR ID | Title | Type | Feature | Status |
|-------|-------|------|---------|--------|
| `4a8f5c91` | RISK-001: Persistent Risk Registry model + views | feature | FEAT-090 | draft |
| `25129695` | IVR-001: Mechanical veto in validate_for_transition() | feature | FEAT-091 | draft |
| `c59f8370` | CMP: Severity-weighted compliance scoring | enhancement | FEAT-092 | draft |
| `ca60cd0c` | QUAL: Rejected analysis creates quality record | enhancement | FEAT-093 | draft |
| `0247654d` | Close Tier 1 coverage gap (35% → 70%) | enhancement | FEAT-094 | draft |
| `8bea4b75` | Close Tier 2 coverage gap (8.2% → 50%) | enhancement | FEAT-095 | draft |
| `5fe481d8` | Close Tier 3 coverage gap (18.1% → 30%) | enhancement | FEAT-096 | draft |
| `e3d693f0` | ARCH-001: Complexity violations | debt | FEAT-097 | draft |

### Deferred (Owned Elsewhere)

| Item | Reason |
|------|--------|
| CR-C: CAL WARNING→FAIL | Calibration session owns all calibration.py changes |
| Calibration case gap analysis | Out of scope — sync on completion before closing T1-COV |
| agents_api/calibration.py | DO NOT TOUCH — active parallel session |

---

## 2. Coverage Baseline (2026-03-06)

### Overall

| Metric | Value |
|--------|-------|
| Total symbols | 1,591 |
| Covered symbols | 351 (22.1%) |
| Target threshold | 30% |
| Gap to pass | 7.9 pp (≈126 more symbols) |
| Total LOC | 108,403 |
| Covered LOC | 38,459 (35.5%) |
| Risk score | 69,774 ungoverned LOC |

### Per-Module Breakdown

| Module | Coverage | Symbols | LOC | Tier | Target |
|--------|----------|---------|-----|------|--------|
| agents_api | 25.0% | 147/587 | 63,907 | T1+T2 | 70% (core), 50% (views) |
| syn | 33.3% | 152/457 | 24,622 | T4 | 20% ✓ DONE |
| api | 2.2% | 4/183 | 7,088 | T2+T3 | 50% |
| core | 13.1% | 13/99 | 5,595 | T3 | 30% |
| workbench | 10.8% | 8/74 | 2,279 | T3 | 30% |
| forge | 0.0% | 0/71 | 1,528 | T3 | 30% |
| accounts | 26.4% | 14/53 | 1,222 | T2 | 50% |
| notifications | 40.0% | 8/20 | 512 | T4 | 20% ✓ DONE |
| files | 0.0% | 0/14 | 514 | T3 | 30% |
| chat | 27.8% | 5/18 | 468 | T3 | 30% |
| inference | 0.0% | 0/9 | 487 | T3 | 30% |

---

## 3. Per-Tier Work Plan

### Tier 1 — Statistical Core (target 70%, current 25.0%)

**Focus:** Behavioral tests for statistical functions + assert→impl→test chain coverage.
Calibration cases excluded (owned elsewhere).

#### Module Inventory

| Module | LOC | Functions | Tested | Gap |
|--------|-----|-----------|--------|-----|
| dsw/common.py | 3,023 | 39 | 0 | **39 (100%)** |
| dsw/stats.py | 19,370 | 1 (entry) | 1 | 0 |
| dsw/spc.py | 4,743 | 4 | 1 | 3 |
| dsw/bayesian.py | 4,456 | 1 (entry) | 1 | 0 |
| dsw/reliability.py | 1,379 | 1 (entry) | 1 | 0 |
| dsw/ml.py | 4,069 | 1 (entry) | 1 | 0 |
| dsw/dispatch.py | 292 | 2 | 1 | 1 |

#### Top 10 Highest-Risk Uncovered (common.py)

| Rank | Function | LOC | Risk Score | Priority |
|------|----------|-----|------------|----------|
| 1 | `_bayesian_model_beliefs` | 448 | 2,822 | P0 |
| 2 | `_bayesian_shadow` | 291 | 1,950 | P0 |
| 3 | `_diag_classification` | 308 | 801 | P1 |
| 4 | `_diag_regression` | 256 | 461 | P1 |
| 5 | `_data_skepticism` | 102 | 275 | P1 |
| 6 | `_permutation_reality_test` | 95 | 200 | P2 |
| 7 | `_auto_train` | 106 | 191 | P2 |
| 8 | `_evidence_grade` | 77 | 139 | P2 |
| 9 | `_claude_interpret_results` | 32 | 22 | P3 |
| 10 | `_build_ml_diagnostics` | 22 | 7 | P3 |

#### Ordered Test Plan

1. **`_bayesian_model_beliefs`** + **`_bayesian_shadow`** — property-based tests, golden-file comparisons
2. **`_diag_classification`** + **`_diag_regression`** — mock sklearn, verify output schema
3. **`_data_skepticism`** — boundary conditions, edge cases
4. **SPC helpers** — `_spc_nelson_rules`, `_spc_build_point_rules`, `_spc_add_ooc_markers`
5. **`_permutation_reality_test`** + **`_auto_train`** — functional tests
6. Remaining common.py helpers (30+ small functions)
7. **`_read_csv_safe`** in dispatch.py

**Estimated tests to reach 70%:** ~80–100 behavioral tests across common.py + spc helpers
**Estimated sessions:** 4–6

### Tier 2 — Revenue Path (target 50%, current 2.7%)

**73 endpoints across 5 view files, 2 tested (2.7%).**

#### Endpoint Inventory

| File | Endpoints | Tested | Gap |
|------|-----------|--------|-----|
| dsw_views.py | 26 | 1 | 25 |
| synara_views.py | 28 | 0 | 28 |
| experimenter_views.py | 9 | 0 | 9 |
| spc_views.py | 8 | 0 | 8 |
| forecast_views.py | 2 | 1 | 1 |

**Auth decorator coverage: 100%** — all 73 endpoints have decorators.

#### Smoke Test Pattern

Every endpoint gets 3 tests minimum:
- `test_{name}_requires_auth` — 401/403 without token
- `test_{name}_valid_request` — 200/201 with proper auth + data
- `test_{name}_invalid_input` — 400 with bad payload

#### Priority Order (revenue × risk)

1. **SPC endpoints** (8) — control_chart, capability_study, gage_rr drive quality decisions
2. **DOE endpoints** (9) — power_analysis, design_experiment are premium tier ($99+)
3. **DSW core** (5) — dsw_from_intent, dsw_from_data, run_model, optimize_model, run_analysis
4. **Synara core** (10) — add_hypothesis, add_evidence, get_belief_state, causal chains
5. **Remaining DSW** (21) — model management, scrub, download, etc.
6. **Remaining Synara** (18) — DSL, LLM endpoints, import/export
7. **Forecast** (1) — quote endpoint

**Estimated tests to reach 50%:** ~110 smoke tests (73 × 1.5 avg)
**Estimated sessions:** 5–7

### Tier 3 — Feature Surface (target 30%, current ~32% avg but 6 modules below)

#### Modules Below 30%

| Module | Views | Tested | Coverage | Gap×LOC | Priority |
|--------|-------|--------|----------|---------|----------|
| learn_views.py | 11 | 0 | 0% | 2,448 | P1 |
| hoshin_views.py | 32 | 6 | 19% | 1,482 | P1 |
| api/views.py | 29 | 10 | 34%* | 1,333 | P2 |
| whiteboard_views.py | 17 | 0 | 0% | 1,041 | P2 |
| workbench/views.py | 40 | 3 | 8% | 888 | P2 |
| triage_views.py | 6 | 0 | 0% | 533 | P3 |
| forge/views.py | 8 | 0 | 0% | 422 | P3 |
| files/views.py | 9 | 0 | 0% | 420 | P3 |

*api/views.py is at 34% but has high absolute gap

#### Quick Wins (< 2 hours each)

1. **chat/views.py** — 1 view, 34 LOC → 5 minutes
2. **guide_views.py** — 1 untested view → 30 minutes
3. **files/views.py** — 9 views, 420 LOC → 1–2 hours
4. **forge/views.py** — 8 views, 422 LOC → 1–2 hours
5. **triage_views.py** — 6 views, 533 LOC → 2–3 hours

#### Top 5 to Close First (ranked by gap×LOC)

1. **learn_views.py** — 0% coverage, largest module (2,448 LOC)
2. **hoshin_views.py** — 19% coverage, enterprise revenue (1,482 gap×LOC)
3. **whiteboard_views.py** — 0% coverage, core collab feature (1,041 gap×LOC)
4. **workbench/views.py** — 8% coverage, new platform (888 gap×LOC)
5. Quick wins batch (chat + guide + files + forge + triage)

**Estimated sessions:** 3–4

### Tier 4 — Infrastructure (target 20%, current 33.3%)

**Status: EXCEEDS TARGET. No action needed.**

syn module at 33.3% (152/457 symbols). notifications at 40.0%.

---

## 4. Gap CR Implementation Details

### CR-A: RISK-001 — Persistent Risk Registry

**Complexity: SMALL (2–3 hours)**

| What | Where | Line |
|------|-------|------|
| New `RiskRegistryItem` model | syn/audit/models.py | After DriftViolation (~434) |
| Pattern reference | DriftViolation (lines 334–434) | SynaraImmutableLog base |
| New compliance check | syn/audit/compliance.py | `@register("risk_registry", ...)` |
| Registration pattern | compliance.py lines 35–49 | `ALL_CHECKS` decorator |
| Dashboard endpoint | api/internal_views.py | After api_compliance (~3890) |

**Model fields:** severity, likelihood, detectability, rpn_score, mitigation_status, owner,
remediation_change_id (FK to ChangeRequest), category, description, is_resolved.

### CR-B: IVR-001 — Mechanical Veto

**Complexity: SMALL (1–2 hours)**

| What | Where | Line |
|------|-------|------|
| Inject veto check | syn/audit/models.py:validate_for_transition() | After line 1237 |
| Security veto already computed | RiskAssessment.compute_aggregate() | Line 1654 |
| AgentVote model | syn/audit/models.py | Lines 1672–1732 |

**Change:** After the existing `risk_assessments.exists()` check at line 1236, add:
```python
# IVR-001: Mechanical veto — security_analyst rejection blocks transition
if self.change_type in self.MULTI_AGENT_TYPES:
    for ra in self.risk_assessments.all():
        if ra.overall_recommendation == "reject":
            security_rejects = ra.votes.filter(
                agent_role="security_analyst", recommendation="reject")
            if security_rejects.exists():
                errors.append("Security analyst veto — change blocked (IVR-001)")
```

**No model changes needed.** RiskAssessment.overall_recommendation and AgentVote already exist.

### CR-D: CMP — Severity-Weighted Scoring

**Complexity: MEDIUM (4–6 hours)**

| What | Where | Line |
|------|-------|------|
| Add severity field | ComplianceCheck model (syn/audit/models.py) | Line 606–651 |
| Add weighted fields | ComplianceReport model | Line 928–961 |
| Weighted calculation | compliance.py pass rate logic | Lines 2450–2500 |
| Report creation | compliance.py | Line 2534–2546 |
| Dashboard trend | api/internal_views.py | Lines 3790–3798 |

**Weights:** FAIL=3×, WARNING=1×, INFO=0×.
**Migration:** Add `severity` CharField with default="medium" to ComplianceCheck.
Add `severity_weighted_score`, `critical_pass_rate` FloatFields to ComplianceReport.

### CR-E: QUAL — Rejected Analysis Quality Record

**Complexity: MEDIUM (3–4 hours)**

| What | Where | Line |
|------|-------|------|
| New `QualityRecord` model | syn/audit/models.py | After ComplianceCheck |
| Injection points | agents_api/dsw/dispatch.py:run_analysis() | Lines 66, 92, 94, 129, 202 |
| Helper function | dispatch.py | New `_log_quality_rejection()` |
| Existing audit pattern | SysLogEntry (models.py lines 21–230) | Reference |

**5 validation rejection points** in dispatch.py currently return 400 silently.
Each gets a `_log_quality_rejection()` call before the return.

---

## 5. Sequencing

### Week 1: Structural (fast, high impact)
- **CR-B (IVR-001)** — Mechanical veto. 5-line injection. Unblocks verification independence claim.
- **FEAT-098** — Dashboard tab for INIT-012 progress visibility.
- Quick wins: chat/views.py, guide_views.py smoke tests.

### Week 2–3: Statistical Core (T1-COV)
- common.py P0: `_bayesian_model_beliefs`, `_bayesian_shadow`
- common.py P1: `_diag_classification`, `_diag_regression`, `_data_skepticism`
- SPC helpers: nelson rules, point rules, OOC markers
- **Target: 50% T1 coverage by end of week 3**

### Week 3–4: Risk Registry + Quality Records
- **CR-A (RISK-001)** — Model + compliance check + dashboard views
- **CR-E (QUAL)** — QualityRecord model + dispatch injection
- T1-COV P2 functions: `_permutation_reality_test`, `_auto_train`, `_evidence_grade`

### Week 4–6: Revenue Path (T2-COV)
- SPC endpoint smoke tests (8 endpoints)
- DOE endpoint smoke tests (9 endpoints)
- DSW core smoke tests (top 5 endpoints)
- Synara core smoke tests (top 10 endpoints)
- **Target: 30% T2 coverage by end of week 5, 50% by week 6**

### Week 6–8: Feature Surface (T3-COV) + Scoring
- **CR-D (CMP)** — Severity-weighted scoring
- learn_views.py smoke tests
- hoshin_views.py gap closure
- whiteboard_views.py + workbench/views.py
- Quick wins batch: files, forge, triage
- **Target: 30% T3 coverage**

### Ongoing
- CR-D + CR-E refinement as compliance runs surface failures
- Complexity governance (FEAT-097) as opportunities arise
- Dashboard tab updates as features complete

### Calibration Sync Point
Calibration hardening is a parallel workstream. Before closing T1-COV:
- Confirm calibration session has completed CAL-001 changes
- Verify calibration coverage check status
- Sync any shared symbol coverage implications

---

## 6. Testing Strategy Rules

All tests under this initiative MUST follow:

1. **Behavioral tests only** — no existence-only tests (TST-001 §10.6)
2. **Every test linked** to `<!-- test: -->` hook in a standard
3. **Smoke test pattern:** auth(401/403) + valid(200/201) + invalid(400)
4. **Common.py pattern:** property-based + golden-file comparison where applicable
5. **After each session:** run `check_symbol_coverage`, verify ratchet ≥ 22.1% baseline, commit
6. **CHG-001 mandatory** — every code change gets a ChangeRequest

---

## 7. Ratchet Baseline Snapshot

```
Date:            2026-03-06
Overall:         22.1% (351/1,591 symbols)
LOC coverage:    35.5% (38,459/108,403)
Risk score:      69,774
Threshold:       30% (FAIL)
Gap to PASS:     7.9 pp (≈126 symbols)
```

### Per-Tier Baselines

| Tier | Module(s) | Current | Target | Gap (symbols) |
|------|-----------|---------|--------|---------------|
| T1 | agents_api (core) | 25.0% | 70% | ~264 |
| T2 | agents_api (views) + api | 2.2–25% | 50% | ~200 |
| T3 | feature modules | 0–19% | 30% | ~80 |
| T4 | syn + notifications | 33–40% | 20% | ✓ DONE |

### Estimated Sessions to Milestones

| Milestone | Sessions | Symbol Coverage |
|-----------|----------|---------------|
| 30% threshold (PASS) | 3–4 | ~477 symbols |
| T1 at 50% | 6–8 | ~550 symbols |
| T1 at 70% | 10–12 | ~650 symbols |
| T2 at 50% | 16–18 | ~750 symbols |
| Full plan complete | 20–24 | ~850+ symbols |

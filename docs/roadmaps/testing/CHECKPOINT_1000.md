# CHECKPOINT 1000: Standards-Linked Test Target

**Date:** 2026-03-04
**Target:** 1,000 tests linked via `<!-- test: -->` hooks in standards
**Strategy:** TST-001 §10 — standards ARE the test strategy
**CR:** `8eb560e6` (enhancement, completed)
**Status:** ACHIEVED

---

## Result

| Metric | Before | After |
|--------|--------|-------|
| Linked tests (`<!-- test: -->`) | 455 | **1,010** |
| Assertions (`<!-- assert: -->`) | 371 | 365 |
| Total test methods | 944 | **1,031** |

### What was done:
- **Stream 1:** Linked 498 existing tests across 23 standards (zero new test code)
- **Stream 2:** Wrote 87 new DSW-001 tests (12 test classes covering all 11 assertions)
- All tests pass, compliance runner confirms 1,010 linked

---

## Original State (pre-work)
| Total test methods in codebase | 944 |
| **Gap to 1000** | **545** |

---

## Work Streams

### Stream 1: Link Existing Tests (~200 links, no new code)

Tests exist but aren't linked to standards via `<!-- test: -->` hooks. Cheapest wins.

| Standard | Tests Exist | Currently Linked | Unlinkable |
|----------|-------------|-----------------|------------|
| SCH-001 | 34 | 12 | ~22 |
| LLM-001 | 20 | 9 | ~11 |
| ERR-001 | 41 | 24 | ~17 |
| OPS-001 | 19 | 12 | ~7 |
| AUD-001 | 26 | 12 | ~14 |
| CHG-001 | 28 | 16 | ~12 |
| LOG-001 | 31 | 23 | ~8 |
| STY-001 | 18 | 9 | ~9 |
| CMP-001 | 44 | 23 | ~21 |
| ARCH-001 | 30 | 14 | ~16 |
| MAP-001 | 23 | 7 | ~16 |
| DOC-001 | 26 | 22 | ~4 |
| **Subtotal** | | | **~157** |

Also: app-level tests not yet linked to any standard:
- `core/tests.py` (70 methods) → DAT-001, SEC-001
- `api/tests.py` (79 methods) → API-001, SEC-001, BILL-001, TST-001
- `agents_api/tests.py` (86 methods) → QMS-001, API-001
- `agents_api/qms_*.py` (120 methods) → QMS-001, QMS-002, TRN-001
- `forge/tests.py` (23 methods) → DAT-001, API-001

**Estimated yield: ~200 new links (standard hook additions only, no test code)**

---

### Stream 2: Zero-Coverage Standards (~300 new tests + links)

Three standards have assertions but zero tests.

#### DSW-001 — Decision Science Workbench (11 assertions, 0 tests)
**Priority: CRITICAL — this is the product core**

The DSW has 200+ statistical analyses across categories. Test strategy:
- Endpoint smoke tests: each `/api/dsw/` analysis returns 200 + valid JSON
- Category coverage: parametric, nonparametric, regression, DOE, Bayesian, ML, SPC, time series
- Input validation: missing fields, wrong types, edge cases
- Result structure: required keys in response (result, method, interpretation)
- Code generation: Python/R output for each analysis type

**Estimated: 150-200 tests** across ~10 test classes

#### STAT-001 — Statistical Methodology (12 assertions, 0 tests)
**Priority: HIGH — methodology correctness**

Tests for:
- Assumption checking (normality, homogeneity, independence)
- Method selection logic (parametric vs nonparametric routing)
- Effect size computation
- Confidence interval conventions
- Multiple comparison corrections
- Sample size requirements

**Estimated: 40-60 tests**

#### SLA-001 — Service Level Agreements (1 assertion, 0 tests)
**Priority: MEDIUM — operational**

Tests for:
- Availability target definitions
- Response time thresholds
- Rate limit configurations
- Tier-based SLA differentiation

**Estimated: 10-20 tests**

---

### Stream 3: Gap-Fill Existing Standards (~100 new tests + links)

Standards with test files but significant assertion gaps.

| Standard | Assertions | Linked Tests | Missing Coverage |
|----------|-----------|-------------|-----------------|
| API-001 | 25 | 14 | URL patterns, pagination, idempotency, error envelope |
| SEC-001 | 26 | 22 | CSP headers, rate limiting, tenant isolation sweeps |
| DAT-001 | 27 | 22 | Field naming conventions, soft delete cascades, versioning |
| BILL-001 | 13 | 19 | Tier enforcement, Stripe webhook handling, quota checks |
| FE-001 | 13 | 28 | Already well-covered; minor gaps in widget compliance |
| TST-001 | 11 | 13 | Test organization verification, fixture pattern checks |
| RDM-001 | 10 | 6 | Roadmap item lifecycle, milestone tracking |
| TRN-001 | 8 | 14 | Competency matrix, certification expiry |

**Estimated: 80-100 new tests**

---

### Stream 4: Untested App Domains (~50 new tests + links)

Apps with zero test coverage.

| App | Standard | What to Test |
|-----|----------|-------------|
| **accounts/** | SEC-001, BILL-001 | Registration flow, login/logout, permission decorators, subscription model |
| **chat/** | LLM-001 | Conversation creation, message handling, context management |
| **inference/** | LLM-001 | Local model loading, inference pipeline |

**Estimated: 40-50 tests**

---

### Stream 5: Registration Pathway Documentation

Document the standard registration process (currently tribal knowledge):
- How a new standard goes from idea → PLANNED → APPROVED
- DOC-001 namespace registration requirement
- MAP-001 registry entry requirement
- Test file creation and `<!-- test: -->` hook linking
- Compliance runner verification

**Deliverable:** New section in TST-001 or DOC-001 (documentation only, no test count)

---

## Yield Estimate

| Stream | New Links | New Tests | Total Linked |
|--------|-----------|-----------|-------------|
| 1. Link existing | ~200 | 0 | 655 |
| 2. Zero-coverage | ~300 | ~250 | 955 |
| 3. Gap-fill | ~100 | ~100 | 1055 |
| 4. Untested apps | ~50 | ~50 | 1105 |
| **Total** | **~650** | **~400** | **~1105** |

Streams 1-3 should hit the 1000 target. Stream 4 is bonus.

---

## CR Strategy

**One umbrella CR** scoped as `enhancement` to testing infrastructure:
- **Title:** `CHECKPOINT-1000: Standards-linked test expansion to 1000`
- **Type:** `enhancement`
- **Risk:** `low` (tests only — no production code changes)
- **Scope:** All `<!-- test: -->` hook additions in `docs/standards/*.md` + new test files in `syn/audit/tests/` and app `tests.py` files

Per CHG-001 §5.5, a PlanDocument can link to one or more CRs. This checkpoint doc serves as the plan. One CR covers the full sweep because:
1. All changes are test additions (zero production code risk)
2. All changes serve one goal (CHECKPOINT 1000)
3. Splitting into 27 CRs (one per standard) adds overhead without reducing risk

If any stream requires production code changes (e.g., adding test helpers to app code), split that into a separate CR.

---

## Execution Order

1. **Stream 1 first** — pure documentation, immediate progress, validates the linking pattern
2. **Stream 2: DSW-001** — highest impact, most tests, product core
3. **Stream 2: STAT-001** — natural pair with DSW-001
4. **Stream 3** — fill gaps across remaining standards
5. **Stream 4** — untested apps
6. **Stream 5** — registration pathway docs (can happen anytime)
7. **Stream 2: SLA-001** — lowest priority, do last

---

## Verification

```bash
# Count linked tests
grep -rh '<!-- test:' docs/standards/*.md | wc -l

# Run all linked tests
python manage.py run_compliance --standards --run-tests

# Run full test suite
python manage.py test syn.audit.tests -v2
```

Target: `grep` count ≥ 1000, all tests PASS.

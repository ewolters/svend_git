# QUAL-001: Output Quality Assurance

**Version:** 1.1
**Status:** APPROVED
**Date:** 2026-03-05
**Supersedes:** DSW-002
**Author:** Eric + Claude (Systems Architect)

**Compliance:**
- DOC-001 ≥ 1.2 (Documentation Structure — §7 Machine-Readable Hooks)
- XRF-001 ≥ 1.0 (Cross-Reference Syntax)
- MAP-001 ≥ 1.0 (Architecture Mapping)
- STY-001 ≥ 1.0 (Code Style & Conventions)
- SOC 2 CC4.1 (Processing Integrity)
- SOC 2 CC7.2 (System Monitoring)
- SOC 2 CC9.1 (Risk Management)
- ISO 9001:2015 §7.1.5 (Monitoring and Measuring Resources)
- ISO 9001:2015 §8.5.1 (Controlled Conditions for Production)

**Related Standards:**
- STAT-001 ≥ 1.2 (Statistical Methodology — calibration, effect sizes)
- DSW-001 ≥ 2.0 (Decision Science Workbench Architecture)
- TST-001 ≥ 1.0 (Testing Patterns)
- CMP-001 ≥ 1.0 (Compliance Automation)
- QMS-001 ≥ 1.0 (Quality Management System)
- CHG-001 ≥ 1.0 (Change Management)
- FE-001 ≥ 1.0 (Frontend Patterns — theme compatibility)

---

## **1. SCOPE AND PURPOSE**

### 1.1 Purpose

Define what "correct output" means for every analysis, tool, and computation the Svend platform produces. QUAL-001 is the unifying standard that connects mathematical correctness (STAT-001), pipeline architecture (DSW-001), compliance automation (CMP-001), code testing (TST-001), and quality tool workflows (QMS-001) into a single output quality framework.

**Core Principle:**

> An output is correct when three things are true simultaneously: the math is right, the data was clean enough to trust, and the narrative matches the numbers.

### 1.2 Scope

**Applies to:**
- All DSW analysis outputs (230+ registered analyses)
- SPC control chart and capability study results
- QMS tool outputs (FMEA RPN calculations, RCA causal chains, A3 summaries, VSM metrics, Hoshin KPI values)
- Forecast engine predictions
- Forge synthetic data quality
- Calibration system operations

**Does NOT apply to:**
- Code quality and test framework patterns (TST-001)
- Statistical formulas and methodology details (STAT-001)
- DSW architecture and dispatch routing (DSW-001)
- Compliance infrastructure mechanics (CMP-001)
- Frontend rendering fidelity (FE-001)

### 1.3 Relationship to Other Standards

| Standard | What It Owns | QUAL-001 Relationship |
|----------|-------------|----------------------|
| STAT-001 | Mathematical formulas, effect sizes, Bayesian insurance | Defines the algorithm; QUAL-001 verifies the output |
| DSW-001 | Dispatch, standardize.py, registry, evidence bridge | Defines the plumbing; QUAL-001 defines what flows through it |
| TST-001 | Test framework, fixtures, conventions | Defines *how* to test; QUAL-001 mandates *what* to test |
| CMP-001 | Compliance automation, daily checks, drift detection | Runs QUAL checks; QUAL-001 defines them |
| QMS-001 | FMEA, RCA, A3, VSM, Hoshin tool architecture | Defines the tools; QUAL-001 defines output correctness |
| CHG-001 | Change management process | Tracks changes to calibration cases and validation rules |

---

## **2. NORMATIVE REFERENCES**

| Standard | Section | Relevance |
|----------|---------|-----------|
| STAT-001 | 4 | Eleven methodological principles -- mathematical correctness criteria |
| STAT-001 | 13 | Calibration system -- reference pool, rotation, drift |
| DSW-001 | 4 | Stateless dispatch -- result dict flows through standardize.py |
| DSW-001 | 6 | Registry -- ANALYSIS_REGISTRY metadata per analysis |
| CMP-001 | 5.6 | Drift detection lifecycle -- DriftViolation creation and resolution |
| CMP-001 | 6 | Infrastructure checks -- daily rotation registration |
| TST-001 | 10 | Standards-linked testing -- assert/impl/test triad |
| QMS-001 | 4-5 | QMS tool architecture and calculations |

---

## **3. TERMINOLOGY**

| Term | Definition |
|------|------------|
| **Output quality** | The degree to which an analysis result is mathematically correct, internally coherent, and produced from valid input data |
| **Mathematical correctness** | Statistical outputs fall within known bounds and produce expected results for known inputs. Verified by calibration |
| **Data quality** | Input data meets minimum requirements for the requested analysis (sufficient rows, correct types, no NaN-only columns) |
| **Output coherence** | Narrative text, summary, guide_observation, evidence_grade, and statistics are internally consistent and non-contradictory |
| **Calibration case** | A reference dataset with analytically known correct answers, fed through an analysis function to verify output accuracy |
| **Bounds check** | Validation that a statistical output falls within its mathematically possible range |
| **Drift** | When a previously correct analysis begins producing incorrect output, detected by calibration case failure |
| **Result dict** | Dictionary returned by any DSW analysis function, containing summary, plots, statistics, guide_observation, evidence_grade, narrative, diagnostics, bayesian_shadow, education, and what_if |

---

## **4. QUALITY FRAMEWORK**

<!-- assert: Output quality enforces three pillars: mathematical correctness, data quality, output coherence | check=qual-three-pillars -->
<!-- impl: agents_api/dsw/standardize.py:standardize_output -->
<!-- impl: agents_api/calibration.py:run_calibration -->
<!-- test: syn.audit.tests.test_output_quality.ThreePillarsTest.test_standardize_called_in_dispatch -->
<!-- test: syn.audit.tests.test_output_quality.ThreePillarsTest.test_calibration_runner_exists -->

### 4.1 Three Pillars

Every output MUST satisfy all three pillars simultaneously:

| Pillar | What It Checks | Enforcement | Primary Standard |
|--------|---------------|-------------|------------------|
| **Mathematical Correctness** | Known inputs produce known outputs; statistics within valid bounds | Calibration (section 5), bounds checks (section 6) | STAT-001 |
| **Data Quality** | Input data sufficient, correctly typed, complete enough | Input validation (section 7) | QUAL-001 |
| **Output Coherence** | Summary matches statistics; grade matches p-value; guide_observation matches conclusion | Coherence checks (section 6.3) | QUAL-001 |

### 4.2 Quality Gate Integration

Two enforcement levels:
- **Real-time** (per-request): `standardize_output()` validates schema, bounds, and coherence on every analysis response
- **Periodic** (daily calibration): known-answer cases verify mathematical correctness hasn't drifted

---

## **5. CALIBRATION SYSTEM**

Codifies and extends STAT-001 section 13.

### 5.1 Reference Pool

<!-- assert: Calibration reference pool has >=15 cases across >=5 categories | check=qual-pool-size -->
<!-- impl: agents_api/calibration.py:_build_reference_pool -->
<!-- test: syn.audit.tests.test_output_quality.CalibrationPoolTest.test_minimum_pool_size -->
<!-- test: syn.audit.tests.test_output_quality.CalibrationPoolTest.test_minimum_category_count -->

| Requirement | Minimum | Rationale |
|-------------|---------|-----------|
| Total cases | 15 | Sufficient for 8-case daily subset with full coverage in ~3 days |
| Categories | 5 of 6 | inference, bayesian, spc, reliability, ml, simulation |
| Cases per major category | 3 (inference), 2 (bayesian, spc) | Major categories need null-true + null-false scenarios |
| Cases per minor category | 1 (reliability, ml, simulation) | Smoke-test level sufficiency |

When new analyses are added: any analysis type with `has_pvalue=True` or in a calibrated category SHOULD have a calibration case added within 30 days.

### 5.2 Case Structure

<!-- assert: CalibrationCase has required fields: case_id, category, analysis_type, analysis_id, config, data, expectations | check=qual-case-structure -->
<!-- impl: agents_api/calibration.py:CalibrationCase -->
<!-- test: syn.audit.tests.test_output_quality.CaseStructureTest.test_required_fields -->
<!-- test: syn.audit.tests.test_output_quality.CaseStructureTest.test_case_id_format -->

| Field | Type | Requirement |
|-------|------|-------------|
| `case_id` | str | Pattern: `CAL-{CATEGORY}-{NNN}` |
| `category` | str | One of: inference, bayesian, spc, reliability, ml, simulation |
| `analysis_type` | str | Must match a key in ANALYSIS_REGISTRY |
| `analysis_id` | str | Must match a registered analysis_id |
| `config` | dict | Valid configuration for the target analysis |
| `data` | dict | Column-name to list-of-values mapping |
| `expectations` | list | At least one Expectation per case |

### 5.3 Expectation Types

<!-- assert: Expectation supports four comparison types: abs_within, greater_than, less_than, contains | check=qual-expectation-types -->
<!-- impl: agents_api/calibration.py:_check_expectation -->
<!-- test: syn.audit.tests.test_output_quality.ExpectationTypesTest.test_abs_within -->
<!-- test: syn.audit.tests.test_output_quality.ExpectationTypesTest.test_greater_than -->
<!-- test: syn.audit.tests.test_output_quality.ExpectationTypesTest.test_less_than -->
<!-- test: syn.audit.tests.test_output_quality.ExpectationTypesTest.test_contains -->

| Comparison | Semantics | Use Case |
|------------|-----------|----------|
| `abs_within` | `abs(actual - expected) <= tolerance` | Effect sizes, R-squared |
| `greater_than` | `actual > expected` | p-values for null-true cases |
| `less_than` | `actual < expected` | p-values for clear-effect cases |
| `contains` | substring match (case-insensitive) | guide_observation keywords |

Tolerance guidelines: directional comparisons for p-values (stochastic variation expected), abs_within with tolerance 0.1-0.3 for effect sizes. Tolerances for abs_within MUST be less than 50% of the expected value.

### 5.4 Daily Rotation

<!-- assert: Calibration rotation uses date-seeded RNG for reproducible daily selection | check=qual-rotation -->
<!-- impl: agents_api/calibration.py:run_calibration -->
<!-- test: syn.audit.tests.test_output_quality.RotationTest.test_same_seed_same_selection -->

- **Seed**: `date.today().toordinal()` -- deterministic per day, reproducible
- **Subset**: 8 cases per day (configurable via `subset_size`)
- **Full run**: `run_calibration(subset_size=0)` runs all cases (used in tests and manual verification)

### 5.5 Drift Severity

<!-- assert: Calibration failures create DriftViolation with enforcement_check="CAL" | check=qual-drift -->
<!-- impl: syn/audit/compliance.py:check_statistical_calibration -->
<!-- test: syn.audit.tests.test_output_quality.DriftSeverityTest.test_drift_violation_created -->

| Pass Rate | Status | Severity | SLA |
|-----------|--------|----------|-----|
| 100% | pass | -- | -- |
| 80-99% | warning | MEDIUM | 72h remediation |
| <80% | fail | HIGH | 24h remediation |

---

## **6. OUTPUT VALIDATION**

### 6.1 Schema Validation

<!-- assert: Every analysis result contains required fields after standardize_output | check=qual-schema -->
<!-- impl: agents_api/dsw/standardize.py:REQUIRED_FIELDS -->
<!-- impl: agents_api/dsw/standardize.py:standardize_output -->
<!-- test: syn.audit.tests.test_output_quality.SchemaTest.test_required_fields_filled -->

Required fields (from `standardize.py:REQUIRED_FIELDS`):

| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `summary` | str | `""` | Non-empty for analyses with `has_narrative=True` |
| `plots` | list | `[]` | Each element a dict with `type` key |
| `narrative` | dict/None | `None` | If present, must have `verdict` key |
| `guide_observation` | str | `""` | Non-empty for all non-viz analyses |
| `evidence_grade` | str/None | `None` | One of: Strong, Moderate, Weak, Inconclusive |
| `bayesian_shadow` | dict/None | `None` | If present, must contain `bf10` key |
| `diagnostics` | list | `[]` | Each element has `severity` and `message` keys |

### 6.2 Bounds Checking

<!-- assert: Statistical outputs validated against mathematically possible bounds | check=qual-bounds -->
<!-- impl: agents_api/dsw/standardize.py:_validate_statistics_bounds -->
<!-- test: syn.audit.tests.test_output_quality.BoundsCheckTest.test_p_value_bounds -->
<!-- test: syn.audit.tests.test_output_quality.BoundsCheckTest.test_correlation_bounds -->
<!-- test: syn.audit.tests.test_output_quality.BoundsCheckTest.test_r_squared_bounds -->
<!-- test: syn.audit.tests.test_output_quality.BoundsCheckTest.test_nan_detection -->
<!-- test: syn.audit.tests.test_output_quality.BoundsCheckTest.test_effect_size_finite -->

| Metric | Valid Range | Violation Action |
|--------|------------|------------------|
| `p_value` | [0.0, 1.0] | Clamp to boundary, log warning |
| `correlation` / `pearson_r` / `spearman_rho` | [-1.0, 1.0] | Clamp to boundary, log warning |
| `r_squared` / `R2` | [0.0, 1.0] | Clamp to boundary, log warning |
| `eta_squared` / `partial_eta_squared` | [0.0, 1.0] | Clamp to boundary, log warning |
| `cramers_v` | [0.0, 1.0] | Clamp to boundary, log warning |
| `bf10` (Bayes Factor) | Positive and finite | Set to None, log warning |
| `cp` / `cpk` (capability) | Finite | Set to None, log warning |
| Any numeric statistic | Not NaN or Inf | Set to None, log warning |

### 6.3 Coherence Checks

<!-- assert: Output coherence verified: evidence_grade consistent with statistics, guide_observation populated | check=qual-coherence -->
<!-- impl: agents_api/dsw/standardize.py:standardize_output -->
<!-- test: syn.audit.tests.test_output_quality.CoherenceTest.test_grade_pvalue_consistency -->
<!-- test: syn.audit.tests.test_output_quality.CoherenceTest.test_guide_observation_populated -->

| Rule | Check | Severity |
|------|-------|----------|
| Guide observation populated | If summary non-empty, guide_observation MUST be non-empty | Auto-fill from summary |
| Grade-statistics consistency | If evidence_grade = "Strong", p_value MUST be < 0.05 (when present) | Warning (log) |
| Narrative verdict present | If narrative dict exists, verdict key MUST be non-empty | Warning (log) |
| BF-grade agreement | If bf10 < 1/3 and evidence_grade = "Strong", log contradiction | Warning (log) |

---

## **7. DATA QUALITY**

<!-- assert: Analysis functions validate input data for minimum rows, required columns, and type correctness | check=qual-input-validation -->
<!-- impl: agents_api/dsw/dispatch.py:run_analysis -->
<!-- test: syn.audit.tests.test_output_quality.InputValidationTest.test_row_limit_enforced -->

### 7.1 Input Validation

| Requirement | Rule | Enforcement |
|-------------|------|-------------|
| Minimum rows | Analysis MUST reject DataFrames with fewer rows than the statistical minimum (e.g., >=3 per group for t-test) | Analysis function |
| Required columns | Columns specified in config MUST exist in the DataFrame | dispatch.py or analysis function |
| Type correctness | Numeric analyses MUST verify columns are numeric or coercible | Analysis function |
| Row limit | Inline data limited to 10,000 rows | dispatch.py |
| Completeness | If a required column has >50% NaN values, warn the user | Analysis function |

### 7.2 Data Quality Reporting

Data quality issues MUST be reported in the `diagnostics` array, never as silent drops:

```python
{"severity": "warning", "message": "Column 'x' has 23% missing values (46 of 200 rows)"}
```

### 7.3 Rejection Quality Records

<!-- rule: mandatory -->

When DSW input validation rejects an analysis request, a quality record MUST be created in the hash-chained audit trail via `generate_entry()`. This enables nonconformance trending and root cause analysis on user data quality issues.

**Rejection reasons logged:** invalid JSON, inline data too large, invalid inline data format, no data loaded, unknown analysis type.

**Event:** `quality.analysis_rejected` — payload includes `reason`, `analysis_type`, `analysis_id`.

<!-- assert: DSW validation rejections create audit trail quality records | check=qual-rejection-logging -->
<!-- impl: agents_api/dsw/dispatch.py:_log_rejection -->
<!-- test: syn.audit.tests.test_quality_records.QualityRejectionLoggingTest.test_invalid_json_creates_entry -->
<!-- test: syn.audit.tests.test_quality_records.QualityRejectionLoggingTest.test_no_data_creates_entry -->
<!-- test: syn.audit.tests.test_quality_records.QualityRejectionLoggingTest.test_unknown_type_creates_entry -->

<!-- assert: quality.analysis_rejected event is registered in audit event catalog | check=qual-event-registered -->
<!-- impl: syn/audit/events.py:AUDIT_EVENTS -->
<!-- test: syn.audit.tests.test_quality_records.QualityEventCatalogTest.test_event_registered -->
<!-- test: syn.audit.tests.test_quality_records.QualityEventCatalogTest.test_event_has_payload_schema -->

---

## **8. MODULE-SPECIFIC QUALITY**

### 8.1 DSW Statistical Analyses

<!-- assert: Registry analyses with has_pvalue=True produce p_value in statistics dict | check=qual-dsw-pvalue -->
<!-- impl: agents_api/dsw/registry.py:ANALYSIS_REGISTRY -->
<!-- test: syn.audit.tests.test_output_quality.DSWQualityTest.test_pvalue_analyses_produce_pvalue -->

Every analysis with `has_pvalue=True` in the registry MUST produce `statistics.p_value`. Every analysis with a non-null `effect_type` MUST produce the corresponding effect size metric.

**Narrative quality** (supersedes DSW-002 §5):
- `narrative` MUST be a dict after post-processing (string narratives normalized by `_narrative_from_summary`)
- `narrative.verdict` MUST be ≥10 characters with no `<<COLOR:` tags
- `guide_observation` MUST be 10–300 characters with no color tags when summary is non-empty

**Chart output** (supersedes DSW-002 §6):
- All charts use transparent backgrounds (`rgba(0,0,0,0)`) for theme compatibility
- Chart traces use `SVEND_COLORS` palette from `common.py`; no hardcoded hex outside the palette
- Standardized dimensions applied by `chart_defaults.py:apply_chart_defaults`

**Education** (supersedes DSW-002 §7):
- Every registered analysis SHOULD have a non-None `education` entry after `standardize_output()`
- Education entries have `title` (≥15 chars) and `content` (≥200 chars, HTML `<dl>` structure)

**Evidence grade** (supersedes DSW-002 §8):
- Valid values: `"Strong"`, `"Moderate"`, `"Weak"`, `"Inconclusive"` (title-case)
- Every analysis with extractable `p_value` MUST have non-None `evidence_grade` after post-processing
- Effect classification thresholds: Cohen's d (0.2/0.5/0.8), eta² (0.01/0.06/0.14), r (0.1/0.3/0.5), R² (0.02/0.13/0.26)

**What-if** (supersedes DSW-002 §9):
- Unified schema: `{type, parameters[], endpoint, recompute_fields}`
- Parameters satisfy `min < max`, `step > 0`, `value ∈ [min, max]`
- Legacy `power_explorer` and `what_if_data` patterns normalized by `_normalize_what_if`

**Diagnostics** (supersedes DSW-002 §10):
- Each entry has `test`/`detail`/`status` fields; `status ∈ {pass, warn, fail, info}`

### 8.2 SPC

<!-- assert: SPC control limits satisfy UCL > CL > LCL and are all finite | check=qual-spc-limits -->
<!-- impl: agents_api/dsw/spc.py -->
<!-- test: syn.audit.tests.test_output_quality.SPCQualityTest.test_control_limits_ordered -->

- Control limits: UCL > CL > LCL, all finite
- Nelson rule annotations reference only rules 1-8
- Capability indices (Cp, Cpk) non-negative when reported

### 8.3 QMS Tools

<!-- assert: FMEA RPN calculation produces values in [1, 1000] with factors in [1, 10] | check=qual-fmea-rpn -->
<!-- impl: agents_api/fmea_views.py -->
<!-- test: syn.audit.tests.test_output_quality.QMSQualityTest.test_fmea_rpn_bounds -->

- **FMEA**: RPN = S * O * D; each factor in [1, 10]; RPN in [1, 1000]
- **RCA**: Causal chains must have at least one cause linked to the stated problem
- **VSM**: Lead time calculations non-negative; cycle times sum to less than lead time
- **Hoshin**: Aggregated KPI values match the declared aggregation method

### 8.4 Forecast

Predictions MUST include confidence intervals. MAPE/RMSE metrics must be non-negative and finite.

### 8.5 Forge

Generated data MUST match the template schema. Column types preserved. Row count matches request.

---

## **9. ANTI-PATTERNS**

**Prohibited:**

1. **Defensive defaults without logging** -- silently filling missing fields and returning as if the analysis succeeded. When `standardize_output` fills a field the registry says SHOULD be present, it MUST log at WARNING level.

2. **p-value outside [0,1] passed to frontend** -- returning a p-value that is negative, >1, NaN, or Inf without clamping or warning.

3. **Calibration cases with trivially wide tolerances** -- tolerances for abs_within MUST be less than 50% of the expected value for numeric comparisons.

4. **Skipping input validation** -- running a statistical test on data that clearly cannot support it (e.g., t-test on 1 observation).

5. **Guide observation contradicting statistics** -- guide_observation saying "no significant difference" when p < 0.01, or "significant" when p > 0.10. The guide_observation MUST be generated from the actual computed statistics.

6. **Existence-only tests for symbol coverage** -- tests that only verify `assertIsNotNone(sym)` or `assertTrue(callable(sym))` without exercising any behavior do not qualify as meaningful coverage. A functional test that calls the symbol and verifies output schema, side effects, or rejection behavior inherently proves existence (TST-001 §10.6).

---

## **10. ACCEPTANCE CRITERIA**

| # | Criterion | Verified By |
|---|-----------|-------------|
| AC-1 | Calibration pool has >=15 cases across >=5 categories | `qual-pool-size` check |
| AC-2 | CalibrationCase has all required fields | `qual-case-structure` check |
| AC-3 | All 4 expectation comparison types exercised in pool | `qual-expectation-types` check |
| AC-4 | Bounds checking catches NaN, Inf, out-of-range values | `qual-bounds` check |
| AC-5 | guide_observation populated when summary exists | `qual-coherence` check |
| AC-6 | FMEA RPN in [1, 1000] | `qual-fmea-rpn` check |
| AC-7 | Output quality check registered and callable | `qual-check-registered` check |

---

## **11. COMPLIANCE MAPPING**

<!-- assert: Output quality compliance check is registered | check=qual-check-registered -->
<!-- impl: syn/audit/compliance.py -->
<!-- test: syn.audit.tests.test_output_quality.CheckRegistrationTest.test_check_registered -->
<!-- test: syn.audit.tests.test_output_quality.CheckRegistrationTest.test_check_is_callable -->
<!-- test: syn.audit.tests.test_output_quality.CheckRegistrationTest.test_check_returns_valid_structure -->

The `output_quality` compliance check validates:
- `_validate_statistics_bounds` exists in standardize.py (section 6)
- `standardize_output` called in dispatch.py (section 6)
- REQUIRED_FIELDS dict has expected keys (section 6)
- Calibration pool meets size requirements (section 5)

| Control | Mapping |
|---------|---------|
| SOC 2 CC4.1 | Processing Integrity -- calibration, output validation |
| SOC 2 CC7.2 | System Monitoring -- drift detection, bounds checking |
| SOC 2 CC9.1 | Risk Management -- data quality, input validation |
| ISO 9001:2015 7.1.5 | Calibration of measurement resources |
| ISO 9001:2015 8.5.1 | Controlled conditions for production |
| ISO 9001:2015 8.6 | Release of products -- acceptance criteria |

---

## **REVISION HISTORY**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-05 | Eric + Claude | Initial release. Three pillars framework, calibration codification, output validation, data quality, module-specific quality |
| 1.1 | 2026-03-05 | Eric + Claude | Supersedes DSW-002. Absorbed DSW output quality requirements (narrative, charts, education, evidence grade, what-if, diagnostics). Fixed compliance header to XRF-001 format |
| 1.2 | 2026-03-06 | Eric + Claude | Added §7.3 Rejection Quality Records — DSW validation rejections logged to audit trail (FEAT-093) |

# PCL + Synara + VSM Binding — Design Spec

**Date:** 2026-04-30 (revised 2026-05-01)
**Status:** Approved
**Author:** Eric Wolters + Claude

## Problem

SVEND's tools (VSM, calculators, SPC, DOE, QMS, Hoshin, PROVA) each maintain their own copies of process data. There is no single source of truth for "what does this process actually look like right now." This causes:

- Manual copy-paste between tools (VSM → calculator → VSM)
- No provenance on values ("where did this cycle time come from?")
- No confidence tracking ("how reliable is this number?")
- No shared data layer for simulation
- Current vs future state comparison is JSON diffing instead of data layer diffing

## Architecture

Three foundational layers. Everything else is a consumer or producer.

```
┌──────────────────────────────────────────────────┐
│  PROVA (theories)    VSM (topology)    Hoshin    │
│  QMS (standards)     Calculators       SPC/DOE   │
│           ↕ read/write via contracts ↕           │
├──────────────────────────────────────────────────┤
│                  SYNARA (contracts)               │
│         mediation, routing, validation            │
├──────────────────────────────────────────────────┤
│              PCL (process characteristics)         │
│        measures, datapoints, confidence,          │
│        operational layer, working layer           │
└──────────────────────────────────────────────────┘
```

**PCL** = measures only. Numbers with meaning, calculations, timestamps.
**PROVA** = models only. Theories about how measures interact (IF-THEN edges).
**Synara** = nervous system. Contracts between all systems.
**VSM** = topology + optional PCL bindings. Map structure with pointers to data.

Each layer degrades gracefully without the others.

### Synara Integration Strategy

Synara's full Cortex pipeline (10-layer biological architecture) is implemented in the original repo (`~/Desktop/experiments/synara_qms/`) but not yet ported to Kjerne. PCL does not require Cortex to function — the data model and read/write API work as direct Python function calls. When Cortex is ported, PCL events (`pcl.datapoint.created`, `pcl.measure.bound`, etc.) flow through the pipeline naturally as Reflexes. No rework required.

**Phase 1:** PCL models + `pcl.write()`/`pcl.read()` as direct function calls.
**Phase 2:** Port Cortex. Register PCL events in EventSchemaRegistry. Wire Reflexes for cross-system triggers (e.g., `pcl.datapoint.created` → `spc.chart.refresh`).

### Base Classes

PCL models inherit from Synara infrastructure:
- **Measure** extends `SynaraEntity` — gets tenant isolation, correlation tracking, audit timestamps, soft delete, event emission hooks, metadata JSONField.
- **Datapoint** extends `SynaraImmutableLog` — write-once with hash-chain integrity (21 CFR Part 11). Observations are never edited, only superseded by new datapoints.
- **MeasureTarget** extends `SynaraEntity` — targets can be updated as aspirations change.

### Reference Pattern

All polymorphic references use string-key pairs (not Django ContentTypes), consistent with the existing `ArtifactReference` pattern in `qms_core/models.py`:
- `parent_type` + `parent_id` (str + UUID) for what a measure belongs to
- `source_ref_type` + `source_ref_id` (str + UUID) for datapoint provenance

## PCL Data Model

### Measure

A named, typed characteristic of a process, material, product, or resource. Anything that "says something" about a system variable in the operation.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| tenant | FK → Tenant | Multi-tenancy |
| name | str | "Press A Cycle Time" |
| slug | str (unique per tenant) | DSL reference: `[press-a-ct]` |
| definition | text | What this measures and why it matters |
| unit | str | "sec", "units/day", "%", "mm", etc. |
| measure_type | enum | process, material, product, resource (extensible) |
| value_type | enum | continuous, discrete, proportion, integer (extensible) |
| range_min | float (nullable) | Below this → alarm |
| range_max | float (nullable) | Above this → alarm |
| formula | text (nullable) | For calculated measures: `[press-a-avail] * [press-a-perf] * [press-a-qual]` |
| parent_type | str (nullable) | Generic FK content type |
| parent_id | UUID (nullable) | Generic FK object ID — links to process step, equipment, material, etc. |
| created_by | FK → User | |
| created_at | datetime | |

**Two kinds of measures:**

1. **Raw** — `formula` is null. Has stored Datapoints. Only raw measures store values.
2. **Calculated** — `formula` is set. No stored values. Resolved by chain-calling components on read. Formulas evaluated via safe AST parser based on Hoshin's `hoshin_calculations.py` pattern (whitelisted arithmetic ops + measure slug lookups, no eval()). Hoshin uses `{{fieldname}}` syntax; PCL uses `[slug]` syntax. Separate engines, different purposes — Hoshin computes from named variables, PCL resolves from database measures.

### Datapoint

A timestamped observation of a raw measure. Every datapoint carries its provenance.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| measure | FK → Measure | Which measure this observes |
| value | float | The observed value |
| timestamp | datetime | When the observation was made |
| source_type | str | automated, doe, workbench, time_study, manual, estimate (extensible) |
| source_ref_type | str (nullable) | Generic FK content type |
| source_ref_id | UUID (nullable) | Generic FK — DOE experiment, SPC chart, DSWResult, etc. |
| observation_count | int | n — number of observations this value represents |
| notes | text (nullable) | Context, method notes |
| confidence | float | Computed on write from source_type × observation_count. Never user-assigned. |
| created_by | FK → User | |

**Confidence hierarchy (base weights, scaled by n):**

| source_type | Base weight | Rationale |
|---|---|---|
| automated | 0.95 | Continuous, no human error |
| doe | 0.90 | Validated design, controlled conditions |
| workbench | 0.85 | Statistical method, sample-based |
| time_study | 0.70 | Scales significantly with n |
| manual | 0.50 | Single human observation |
| estimate | 0.25 | No measurement, educated guess |

Confidence computation: `min(1.0, base_weight * log2(max(1, n)) / log2(plateau))` — scales with observations, plateaus at a source-appropriate n. Default plateau=30, but configurable per source_type (a DOE with n=8 is more informative than a stopwatch study with n=8 — different plateau curves). Exact formula may evolve; the principle is: method × evidence quantity = confidence.

### MeasureTarget (working layer)

Aspirational values — what we're trying to achieve. The future state.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| measure | FK → Measure | Which measure this targets |
| target_value | float | The aspiration |
| target_date | date (nullable) | Achieve by when |
| source | str | "calculator", "hoshin", "manual", etc. |
| source_ref_type | str (nullable) | Generic FK content type |
| source_ref_id | UUID (nullable) | Generic FK |
| created_by | FK → User | |
| created_at | datetime | |

### Immutability

All Datapoints are immutable (`SynaraImmutableLog`). Observations are never edited — corrections are new datapoints that compete on confidence. This provides:
- Hash-chain integrity (21 CFR Part 11 compliance for manufacturing quality data)
- Full audit trail — every value ever recorded is preserved
- No "who changed what" ambiguity — nothing changes, things only accumulate

**Manual entry friction:** Because manual datapoints are immutable and early entries may be the only ground truth, manual input requires confirmation:
- Value entered twice (confirm field, like a password)
- Warning indicators scaled to consequence: if the measure has no prior data, prominent warnings ("this will be the sole observation"); if high-confidence data already exists, lighter treatment (the entry barely moves the needle)
- Once confirmed, the datapoint is permanent

**Corrections:** To fix a wrong manual entry, add a new datapoint with the correct value. The resolution mechanism (see below) determines which value represents the measure — confidence-weighted, not latest-wins. A corrective entry from the same source at the same confidence doesn't automatically supersede; the user must provide context (note, updated observation count) that earns higher confidence, or the system recognizes identical source_type + recency and treats the newer entry as a supersession within that confidence tier.

### Resolution Rules

**`pcl.read(measure_slug)`** — returns the best-available value for a raw measure, determined by confidence-weighted resolution (not simply "latest"). For calculated measures: evaluate formula by recursively resolving each referenced slug.

**Confidence-weighted aggregation (raw measures):**

PCL does not pick a "winning" datapoint. It reports a **weighted aggregate** — every datapoint contributes, weighted by its confidence and its consistency with the established distribution. This is a Bayesian online estimator, not a selector.

The algorithm:

1. **Effective weight** for each datapoint = `confidence × consistency_factor`. The consistency factor penalizes values that deviate significantly from the running estimate. A datapoint 1.7σ from the mean with low confidence contributes near-zero. A datapoint consistent with the distribution and high confidence contributes heavily.
2. **Aggregate value** = weighted mean across all datapoints: `Σ(value_i × effective_weight_i) / Σ(effective_weight_i)`.
3. **Recency decay (optional, configurable).** For measures on processes known to drift, older datapoints decay in effective weight. Decay rate is per-measure, default off. When enabled: `effective_weight *= decay(age)`.
4. **Shift detection.** When a cluster of new high-confidence datapoints consistently deviates from the prior aggregate, the system recognizes a genuine process shift (not noise). The aggregate moves to the new level. This is analogous to an e-detector or CUSUM — sustained deviation from prior = real change.

This means:
- All data contributes. Nothing is discarded.
- A fat-fingered manual entry (low confidence, high deviation) barely moves the aggregate — its effective weight is crushed by the consistency penalty
- A DOE result (high confidence, consistent with distribution) dominates the aggregate
- A genuine process change (sustained high-confidence deviation) shifts the aggregate to the new level
- When PCL is empty and the first manual entry is the only datapoint, it IS the aggregate — hence the entry friction

**`pcl.read(measure_slug)`** returns the aggregate estimate, not a single datapoint's value.

**`pcl.read(measure_slug, at=datetime)`** — historical: computes the aggregate from datapoints at or before the given time.

**`pcl.read_with_meta(measure_slug)`** — returns the aggregate plus distribution metadata: `{value, confidence, variance, n, effective_n, latest_timestamp, staleness_days, shift_detected}`. Consumers that care about data quality (PROVA, Monte Carlo) use this form. `effective_n` reflects how many datapoints meaningfully contribute after weighting.

**Calculated measure metadata:** Calculated measures have no stored confidence. On read, confidence = `min(component_confidences)` (weakest link). Timestamp = `max(component_timestamps)` (most recent update). This is derived on every read, never stored.

**Composite staleness:** When a calculated measure resolves, each component may have a different timestamp. The system surfaces: each component's value, timestamp, and confidence. Flag displayed when component timestamps diverge beyond a configurable threshold.

**Simultaneous refresh:** Users can trigger "update all components" — this prompts for new values on all raw component measures at the same timestamp. The calculated measure still chain-evaluates; no composite value is stored. This is a UX convenience for timestamp-coherent inputs, not a caching mechanism.

## Synara Contracts

### Contract Schema

Extensible. New systems plug in by declaring their contract, not editing a registry.

```python
# Write contract
pcl.write(
    measure_slug="press-a-ct",
    value=44.7,
    source_type="doe",           # extensible, not hardcoded enum
    source_ref=doe_experiment,   # any model instance, or None
    observation_count=30,
    timestamp=now(),
    actor=user,
)
# → validates range, computes confidence, stores Datapoint

# Read contract
pcl.read("press-a-ct")                    # latest value
pcl.read("press-a-ct", at=some_datetime)  # historical
pcl.read_with_meta("press-a-ct")          # value + timestamp + confidence + staleness

# Target contract
pcl.set_target("press-a-ct", target_value=38.0, source="hoshin", target_date=date(2026, 9, 1))
```

### Write Sources

| Source | source_type | Typical confidence | Notes |
|---|---|---|---|
| SPC signals | automated | Highest | Continuous monitoring |
| DOE (/experimenter/) | doe | High | Validated design |
| Workbench analysis | workbench | High | Statistical method |
| Time study | time_study | Medium-high | Scales with n |
| Calculator output | calculator | Derived | Inherits weakest input |
| VSM inline entry | manual | Medium | User-typed |
| Excel/form upload | manual or time_study | Depends on n | Bulk entry |
| QMS audit | manual | Medium | Human observation |

### Read Consumers

| Consumer | What it reads | Time context |
|---|---|---|
| VSM (bound fields) | Measure current value | Latest datapoint |
| VSM (future state) | MeasureTarget value | Target for that measure |
| Hoshin | Operational vs target diff | Latest vs MeasureTarget |
| Calculators | Pre-fill inputs | Latest datapoint |
| PROVA | Ground truth for graph evaluation | Latest + confidence |
| Monte Carlo sim | Distribution (multiple datapoints) | Time range |
| QMS | Control plan parameters | Latest |

### Calculator Write-Back

When a calculator computes a derived value (e.g., takt from demand + available time), the result's confidence is derived — it cannot be higher than its weakest input. Synara tracks the lineage through the formula resolution chain.

## VSM ↔ PCL Binding (Approach C — Hybrid)

### Current State (unchanged)

VSM steps store inline values in JSON:
```json
{ "id": "step-1", "name": "Press A", "cycle_time": 45, "changeover_time": 1800 }
```
This keeps working exactly as-is. No migration required.

### Optional Binding

Any step field can be bound to a PCL measure:
```json
{
  "id": "step-1",
  "name": "Press A",
  "cycle_time": 45,
  "changeover_time": 1800,
  "pcl_bindings": {
    "cycle_time": "press-a-ct",
    "changeover_time": "press-a-co"
  }
}
```

### Resolution Rule

If `pcl_bindings.<field>` exists → resolve from PCL. Otherwise → use inline value. Binding wins when present.

### UI

Each step property field gets a link icon. Click → search/create PCL measure → bind. Bound fields show:
- PCL value (live)
- Subtle visual indicator (chain icon, different color) to distinguish from static inline values
- Confidence and timestamp on hover/detail

### Editing a Bound Field

User types a new value into a bound field → that becomes a new PCL Datapoint (source_type: `manual`, source_ref: the VSM). The user is editing the measure, not the map. The map is a window into PCL.

### Unbinding

Click link icon → unbind. Current PCL value copied into inline field as static snapshot. Map reverts to standalone for that field.

### Future State VSM

Same binding mechanism. Resolves against MeasureTarget instead of latest Datapoint. Future state map shows where you're trying to get to.

### Current vs Future Diff

With PCL binding, the current→future comparison becomes: for each bound measure, compare latest Datapoint (operational) vs MeasureTarget (working). This is a data-layer diff, not a JSON diff. Hoshin integration improves because the improvement targets are explicit MeasureTargets with dates and sources.

## PCL Event Schemas (for Cortex wiring)

Defined now so payloads are stable when Cortex is ported. These become Synara Reflexes.

| Event | Payload | Trigger |
|-------|---------|---------|
| `pcl.datapoint.created` | `{measure_slug, value, source_type, confidence, timestamp, tenant_id}` | Any `pcl.write()` call |
| `pcl.measure.created` | `{measure_slug, name, measure_type, value_type, tenant_id}` | New measure registered |
| `pcl.measure.bound` | `{measure_slug, consumer_app, consumer_type, consumer_id}` | VSM step binds to a measure |
| `pcl.target.set` | `{measure_slug, target_value, source, target_date, tenant_id}` | `pcl.set_target()` call |
| `pcl.alarm.triggered` | `{measure_slug, value, range_min, range_max, violation_type}` | Datapoint outside measure range |

**Example future Reflex:** `pcl.alarm.triggered` where `violation_type == 'above_max'` → execute primitive `qms.ncr.create` → cascade `qms.ncr.created`.

## Reconciliation with Existing Systems

These systems have measure-like concepts that will eventually bind to PCL. No migration required now — Approach C means PCL stands alone. Noted here for future reference.

| Existing System | What it tracks | PCL bridge |
|----------------|----------------|------------|
| `ProcessNode` (graph/) | Quality characteristics, spec/control limits | `ProcessNode.spec_limits` → Measure `range_min`/`range_max` |
| `HoshinKPI` (hoshin/) | Target vs actual values | `actual_value` → `pcl.read()`, `target_value` → `MeasureTarget` |
| `MeasurementEquipment` (agents_api/) | Equipment accuracy, calibration | Equipment referenced via Datapoint `source_ref` |
| `EdgeEvidence` (graph/) | Source-typed evidence with effect sizes | Overlapping `source_type` enums — align when connecting |
| `PROVA GraphNode` (prova/) | Abstract factors/outcomes | Add `pcl_measure_slug` field to ground nodes in reality |

## What This Unlocks

1. **Single source of truth** — all tools read/write the same measures
2. **Provenance** — every value traceable to its source, method, and confidence
3. **Live VSMs** — bound maps update when process data changes
4. **Better Hoshin** — operational vs target diff from PCL layers, not JSON diffing
5. **Validation** — rules can check PCL values for violations (CT > takt, OEE below threshold)
6. **DSL** — `[slug]` syntax resolves measure values in formulas, reports, templates
7. **PROVA ground truth** — graph nodes point to PCL measures for reality
8. **Monte Carlo** — PCL distributions + PROVA routing + VSM topology = simulation
9. **No breaking changes** — existing VSMs work unchanged, PCL adoption is incremental

## Future Extensions (not in scope now, but the design supports them)

- **Bayesian updating** of measure distributions as new datapoints arrive
- **Decision tree / Markov chain models** in PROVA for process routing
- **Monte Carlo simulation** combining PCL + PROVA + VSM
- **PCL-native calculators** that read/write directly instead of going through VSM
- **DSL for building VSMs** from PCL measures (`define_step("Press A", ct=[press-a-ct])`)
- **Automatic anomaly detection** on measures via SPC rules applied to datapoint series

## Non-Goals

- PCL does not store models or theories — that's PROVA
- PCL does not own map topology — that's VSM
- PCL does not handle workflow or approvals — that's QMS
- Synara contracts are extensible, not exhaustive — new systems declare their own contracts

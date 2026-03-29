**CANON-002: INTEGRATION CONTRACTS — EVIDENCE METHODOLOGY**

**Version:** 1.0
**Status:** DRAFT
**Date:** 2026-03-08
**Author:** Eric + Claude (Systems Architect)
**Compliance:**
- DOC-001 ≥ 1.2 (Documentation Structure — §7 Machine-Readable Hooks)
- XRF-001 ≥ 1.0 (Cross-Reference Syntax)
- CANON-001 ≥ 3.0 (System Architecture — three-layer model, tool functions, investigation engine)
**Related Standards:**
- GRAPH-001 ≥ 0.1 (Unified Knowledge Graph and Process Model — single posterior per edge, evidence stacking, Synara integration)
- LOOP-001 ≥ 0.1 (Closed-Loop Operating Model — FMIS renders from graph edge posteriors)
- STAT-001 ≥ 1.2 (Statistical Methodology — mathematical correctness)
- QUAL-001 ≥ 1.1 (Output Quality Assurance — bounds, coherence)
- QMS-001 ≥ 1.7 (Quality Management System — tool specifications)
- DSW-001 ≥ 1.0 (Decision Science Workbench — analysis architecture)

> **Note on unified posteriors (GRAPH-001 §12, Object 271 C6):** ~~The dual posterior model is deprecated.~~ Evidence (e.g., forced failure test) updates ONE posterior on the process graph edge via the unified Synara engine (GRAPH-001 §12.1). That posterior is used for BOTH:
> 1. **FMIS S/O/D rendering**: graph edge posterior mapped to 1-10 integer for FMEA display.
> 2. **Investigation belief updating**: same posterior participates in causal graph propagation via Synara.
> One engine, one posterior, two visualizations. See GRAPH-001 §4.4 (Edge Evidence Stacking) and §12.2 (What Synara Provides).

---

## **1. SCOPE AND PURPOSE**

### **1.1 Purpose**

CANON-002 defines how tool outputs become evidence in the investigation engine, how that evidence is weighted, and how tools interact with each other in sequence. It is the integration contract layer that CANON-001's architecture depends on.

CANON-001 defines *what* each tool does and *where* it sits. CANON-002 defines *how* each tool's output integrates with the Synara causal graph and *how much* that output should influence belief.

**Core Principle:**

> Not all evidence is equal. A designed experiment with randomization and replication produces stronger evidence than an observational study. An observational study with a calibrated instrument produces stronger evidence than one with a measurement system that can't distinguish parts from noise. The methodology of the source determines the weight of the evidence.

### **1.2 Scope**

This standard covers:
- Epistemological hierarchy of evidence sources
- Tool integration contracts (single tool → investigation graph)
- Evidence weighting methodology (source method, sample size, study design)
- Measurement system validity gate (Gage R&R → discount or invalidate)
- Chaining contracts (tool sequences → graph integration)
- Investigation lifecycle (open → active → concluded → exported)
- Evidence supersession (immutable re-run handling)
- Layer 3 export schema (conclusion package)
- Hypothesis confirmation and rejection thresholds
- Concurrent investigation support

Does NOT cover: tool implementation details (see QMS-001), statistical methodology (see STAT-001), output quality (see QUAL-001), or architecture (see CANON-001).

### **1.3 Terminology**

| Term | Definition |
|------|-----------|
| **Evidence weight** | A scalar [0, 1] representing the epistemological strength of a piece of evidence. Computed from source hierarchy, sample properties, and measurement system validity. Modifies how strongly evidence shifts posteriors. |
| **Source rank** | The base weight assigned by the epistemological hierarchy (§2). Reflects study design quality independent of sample size or measurement system. |
| **Validity gate** | A binary or graduated check that can discount or invalidate evidence. Measurement system quality (§4) is the primary validity gate. |
| **Tool contract** | The specification of what a single tool produces and how that output maps to the investigation graph — as hypotheses, causal links, evidence, or priors. |
| **Chain contract** | The specification of how one tool's output becomes another tool's input when operating in sequence on the same investigation. |

---

## **2. EPISTEMOLOGICAL HIERARCHY**

### **2.1 Source Ranking**

Evidence sources are ranked by the strength of inference they support. This hierarchy reflects the degree to which the source controls for confounding, bias, and measurement error.

| Rank | Source Type | Base Weight | Rationale | Example |
|------|-----------|-------------|-----------|---------|
| 1 | **Designed experiment (DOE)** | 0.95 | Randomization + replication + blocking control confounding. Strongest causal inference. | 2³ factorial with 3 replicates on press speed, ink viscosity, substrate |
| 2 | **Controlled observation (SPC)** | 0.85 | Rational subgrouping + time-ordered data controls for temporal confounding. Strong signal detection. | I-MR chart with 25+ subgroups, Nelson rules applied |
| 3 | **Statistical test (DSW)** | 0.75 | Formal hypothesis test with known error rates. Controls Type I/II error but not confounding. | Two-sample t-test, ANOVA, regression with diagnostics |
| 4 | **Structured analysis (Layer 2 tools)** | 0.60 | Methodological framework applied to domain knowledge. Subject to practitioner bias but structured. | Ishikawa with team consensus, FMEA with cross-functional scoring |
| 5 | **Simulation / Monte Carlo** | 0.50 | Model-dependent. Only as strong as the assumptions. Useful for sensitivity, not confirmation. | Monte Carlo on process parameters with assumed distributions |
| 6 | **Observational study** | 0.45 | No experimental control. Confounding uncontrolled. Correlation ≠ causation. | Scatter plot of temperature vs defect rate from production logs |
| 7 | **Expert judgment** | 0.35 | Domain knowledge without data. Valuable for hypothesis generation, weak for confirmation. | "I think it's the supplier change" from a 20-year process engineer |
| 8 | **Anecdotal / single observation** | 0.20 | One data point. No replication, no control. Starting point only. | "Last Tuesday the press jammed when we used that batch" |

### **2.2 Hierarchy Principles**

**Control over confounding determines rank.** The hierarchy is not about mathematical sophistication — it's about how well the method isolates the causal relationship. A simple DOE with 8 runs outranks a complex regression on observational data because the DOE controlled the assignment of factors.

**The hierarchy is a starting point, not a ceiling.** Base weights are modified by sample properties (§3.2), measurement system validity (§4), and study-specific factors. A poorly designed DOE with n=4 and no replication may end up weighted below a well-executed observational study with n=10,000.

**Expert judgment is not disposable.** Rank 7 does not mean "ignore experts." Expert judgment is the primary source for hypothesis generation (conjecture phase of Box & Hunter). It is weak for confirmation because it cannot be replicated or falsified. The hierarchy ranks evidential weight for posterior updates, not value to the investigation.

**Simulation is ranked below direct observation.** A simulation with correct assumptions produces exact results — but "correct assumptions" is the load-bearing phrase. Simulations are ranked at 0.50 because they test the model, not reality. A simulation result that contradicts direct observation means the model is wrong, not reality.

### **2.3 Source Rank Registry**

Source ranks are registered as constants in `agents_api/evidence_weights.py`:

```python
# agents_api/evidence_weights.py

from enum import Enum


class SourceRank(float, Enum):
    """Epistemological hierarchy — CANON-002 §2.1."""
    DESIGNED_EXPERIMENT = 0.95
    CONTROLLED_OBSERVATION = 0.85
    STATISTICAL_TEST = 0.75
    STRUCTURED_ANALYSIS = 0.60
    SIMULATION = 0.50
    OBSERVATIONAL_STUDY = 0.45
    EXPERT_JUDGMENT = 0.35
    ANECDOTAL = 0.20


# Tool → source rank mapping. Used by compute_evidence_weight().
TOOL_SOURCE_RANKS = {
    # Layer 1
    "spc": SourceRank.CONTROLLED_OBSERVATION,
    "dsw": SourceRank.STATISTICAL_TEST,
    "doe_design": None,  # Design phase produces no evidence
    "doe_results": SourceRank.DESIGNED_EXPERIMENT,
    "ml": SourceRank.SIMULATION,
    "forecast": SourceRank.SIMULATION,
    "triage": None,  # Triage produces no evidence
    # Layer 2
    "rca": SourceRank.STRUCTURED_ANALYSIS,
    "ishikawa": SourceRank.STRUCTURED_ANALYSIS,
    "ce_matrix": SourceRank.STRUCTURED_ANALYSIS,
    "fmea": SourceRank.STRUCTURED_ANALYSIS,
    "a3": None,  # Report sink
    "vsm": None,  # Feeds Layer 3 directly
    "report": None,  # Report sink (8D)
    # Layer 3 (when generating own evidence)
    "ncr": SourceRank.STRUCTURED_ANALYSIS,
    "capa": SourceRank.STRUCTURED_ANALYSIS,
    # User-supplied
    "user": SourceRank.EXPERT_JUDGMENT,
    "observation": SourceRank.ANECDOTAL,
}
```

<!-- impl: agents_api/evidence_weights.py::SourceRank -->
<!-- impl: agents_api/evidence_weights.py::TOOL_SOURCE_RANKS -->

### **2.4 Compound Evidence**

When multiple independent pieces of evidence from different sources support the same hypothesis, the combined weight is stronger than any individual piece. This is Bayesian by nature — multiple updates compound.

However, evidence from the **same source** applied multiple times is not independent. Running the same t-test on the same data 10 times does not produce 10x the evidence. Synara's idempotency (same source_tool + source_id + source_field) prevents duplicate evidence creation, but the principle extends to study design: 10 observational studies with the same confounding structure are not 10 independent observations.

---

## **3. EVIDENCE WEIGHTING METHODOLOGY**

### **3.1 Weight Computation**

The final evidence weight used by Synara for posterior updates is:

```
evidence_weight = source_rank × sample_modifier × measurement_validity × study_quality
```

All factors are in [0, 1]. The product is clamped to [0.05, 0.99] **before** assignment to `evidence.strength` — no evidence is perfectly worthless or perfectly certain. The clamp prevents inversion of the damper formula if any component is miscalibrated.

```python
evidence_weight = source_rank * sample_modifier * measurement_validity * study_quality
evidence.strength = max(0.05, min(0.99, evidence_weight))
```

**Full implementation** in `agents_api/evidence_weights.py`:

```python
def compute_evidence_weight(
    source_tool: str,
    sample_size: int | None = None,
    measurement_system_id: str | None = None,
    study_quality_factors: dict | None = None,
) -> float:
    """
    Compute evidence weight per CANON-002 §3.1.
    Returns clamped float in [0.05, 0.99].
    """
    # 1. Source rank
    rank = TOOL_SOURCE_RANKS.get(source_tool)
    if rank is None:
        return 0.0  # Tool produces no evidence

    source_rank = float(rank)

    # 2. Sample modifier (§3.2)
    sample_modifier = _compute_sample_modifier(sample_size)

    # 3. Measurement validity (§4)
    measurement_validity = _compute_measurement_validity(measurement_system_id)

    # 4. Study quality (§3.3)
    study_quality = _compute_study_quality(source_tool, study_quality_factors)

    # 5. Product + clamp
    weight = source_rank * sample_modifier * measurement_validity * study_quality
    return max(0.05, min(0.99, weight))


def _compute_sample_modifier(n: int | None) -> float:
    """CANON-002 §3.2 — sample size modifier."""
    if n is None:
        return 1.0  # Non-sample tools (information, reports)
    if n < 5:
        return 0.50
    if n < 15:
        return 0.70
    if n < 30:
        return 0.85
    if n < 100:
        return 0.95
    return 1.0


def _compute_measurement_validity(measurement_system_id: str | None) -> float:
    """CANON-002 §4 — measurement system validity gate."""
    if measurement_system_id is None:
        return 0.55  # No MSA linked, assumed unvalidated (§4.3)

    from core.models import MeasurementSystem
    try:
        ms = MeasurementSystem.objects.get(id=measurement_system_id)
        return ms.current_validity
    except MeasurementSystem.DoesNotExist:
        return 0.55


def _compute_study_quality(source_tool: str, factors: dict | None) -> float:
    """CANON-002 §3.3 — study quality modifier (geometric mean of applicable factors)."""
    if factors is None:
        return 1.0

    # Applicable factors per source type
    APPLICABLE = {
        "doe_results": ["randomization", "replication", "blocking", "blinding", "pre_registration"],
        "dsw": ["blinding", "pre_registration"],
        "forecast": ["replication", "pre_registration"],
        "ml": [],  # No applicable quality factors
        "spc": [],
    }

    applicable_keys = APPLICABLE.get(source_tool, [])
    if not applicable_keys:
        return 1.0

    values = []
    for key in applicable_keys:
        if key in factors:
            values.append(factors[key])  # Expected: 1.0, 0.7, or 0.5

    if not values:
        return 1.0

    # Geometric mean
    import math
    product = math.prod(values)
    return product ** (1.0 / len(values))
```

<!-- impl: agents_api/evidence_weights.py::compute_evidence_weight -->
<!-- impl: agents_api/evidence_weights.py::_compute_sample_modifier -->
<!-- impl: agents_api/evidence_weights.py::_compute_measurement_validity -->
<!-- impl: agents_api/evidence_weights.py::_compute_study_quality -->

### **3.2 Sample Modifier**

The sample modifier adjusts for how much data underlies the evidence. More data → more reliable estimates → stronger evidence.

| Factor | Modifier | Rationale |
|--------|----------|-----------|
| n < 5 | 0.50 | Insufficient for meaningful inference |
| 5 ≤ n < 15 | 0.70 | Marginal — wide confidence intervals |
| 15 ≤ n < 30 | 0.85 | Adequate for most parametric tests |
| 30 ≤ n < 100 | 0.95 | Strong — CLT reliable |
| n ≥ 100 | 1.00 | Large sample — no discount |

**For tools that don't produce sample-based evidence** (Ishikawa, FMEA, RCA, A3), the sample modifier is 1.0 — their weight comes entirely from source rank and study quality. These tools structure information; they don't produce statistical inference.

**For SPC**, n = number of subgroups (not individual measurements). 25+ subgroups = 1.0 modifier per standard practice.

### **3.3 Study Quality Modifier**

Study quality captures design-specific factors that the source rank doesn't cover:

| Factor | Full credit (1.0) | Partial (0.7) | Penalty (0.5) |
|--------|------------------|---------------|---------------|
| **Randomization** | Properly randomized | Restricted (split-plot) | No randomization |
| **Replication** | ≥ 3 replicates | 2 replicates | Unreplicated |
| **Blocking** | Nuisance factors blocked | Partial blocking | No blocking, known nuisance factors |
| **Blinding** | Assessor blinded | Partially blinded | Unblinded with subjective measures |
| **Pre-registration** | Hypothesis stated before data | — | Post-hoc hypothesis |

Study quality modifier = geometric mean of applicable factors. Not all factors apply to all study types — SPC doesn't have randomization, FMEA doesn't have replication.

**Applicable factors by source type:**

| Source | Randomization | Replication | Blocking | Blinding | Pre-registration |
|--------|--------------|-------------|----------|----------|-----------------|
| DOE | ✓ | ✓ | ✓ | ✓ | ✓ |
| SPC | — | — | — | — | — |
| DSW test | — | — | — | ✓ | ✓ |
| Layer 2 tools | — | — | — | — | — |
| Simulation | — | ✓ (runs) | — | — | ✓ |
| Observational | — | — | — | ✓ | ✓ |
| Expert | — | — | — | — | — |

SPC and Layer 2 tools have no applicable study quality factors — their modifier is 1.0. Their weight is determined entirely by source rank and measurement validity.

---

## **4. MEASUREMENT SYSTEM VALIDITY GATE**

### **4.1 Principle**

The measurement system is not a weighting factor — it is a validity gate. If the measurement system cannot distinguish between parts, the evidence produced by that system is unreliable regardless of study design or sample size. A DOE with perfect randomization and 100 replicates is worthless if the gage can't tell good parts from bad.

**Gage R&R %GRR determines validity:**

| %GRR | Validity | Effect on evidence |
|------|----------|-------------------|
| ≤ 10% | **Valid** | measurement_validity = 1.0 — system adequate |
| 10-20% | **Marginal** | measurement_validity = 0.80 — acceptable for some applications, discount applied |
| 20-30% | **Poor** | measurement_validity = 0.50 — significant discount, flag to practitioner |
| > 30% | **Invalid** | measurement_validity = 0.10 — evidence near-invalidated, investigation should prioritize fixing measurement system |

### **4.2 Cascade Effect**

The measurement system validity applies to **all evidence produced using that measurement system**, across all tools and all investigations. This is not per-tool — it's per-instrument.

If a single gage measures the CTQ (critical to quality) characteristic, and that gage has %GRR = 35%, then:
- Every SPC chart using that gage gets measurement_validity = 0.10
- Every DOE measuring that response with that gage gets measurement_validity = 0.10
- Every capability study (Cpk) from that gage gets measurement_validity = 0.10

This cascade is automatic when the measurement system is linked to the investigation. The practitioner links a Gage R&R study to the measurement system. All downstream evidence inherits the validity factor.

### **4.3 No Gage R&R Available**

When no Gage R&R study exists for the measurement system:
- measurement_validity defaults to **0.55** (assumed unvalidated)
- Synara surfaces an expansion signal: "Measurement system not validated — evidence from this source is discounted"
- The practitioner can override with an explicit measurement_validity assertion (e.g., "this is a calibrated CMM, I trust it") which sets validity to 1.0 with an audit trail entry

### **4.4 Attribute Measurement Systems**

For attribute data (pass/fail, go/no-go), the validity gate uses **Kappa** or **%Agreement** instead of %GRR:

| Metric | Validity | measurement_validity |
|--------|----------|---------------------|
| Kappa ≥ 0.90 or %Agreement ≥ 95% | Valid | 1.0 |
| Kappa 0.75-0.90 or %Agreement 85-95% | Marginal | 0.80 |
| Kappa 0.50-0.75 or %Agreement 70-85% | Poor | 0.50 |
| Kappa < 0.50 or %Agreement < 70% | Invalid | 0.10 |

### **4.5 Measurement System Independence**

Some evidence sources are **not subject to the measurement validity gate:**

| Source | Why exempt |
|--------|-----------|
| Expert judgment | Not measurement-based |
| Simulation | Uses assumed distributions, not measured data |
| Structured analysis (Ishikawa, RCA) | Organizes knowledge, not measurements |
| Time-based observations (cycle time from MES) | System-generated, not gage-dependent |

The validity gate applies only to evidence that depends on a physical measurement instrument.

---

## **5. TOOL INTEGRATION CONTRACTS**

### **5.1 Contract Structure**

Each tool contract specifies:

| Field | Description |
|-------|-------------|
| **Tool** | Tool name and layer |
| **Function** | Information / Intent / Inference (per CANON-001 §1.3) |
| **Source rank** | Position in epistemological hierarchy (§2.1) |
| **Graph interaction** | What the tool creates in the investigation graph |
| **Output schema** | The structured output that feeds the graph or a downstream chain |
| **Evidence trigger** | When evidence is created (if applicable) |
| **Measurement gate** | Whether measurement system validity applies |

### **5.2 Layer 1 Tool Contracts**

#### **5.2.1 SPC**

| Field | Value |
|-------|-------|
| Tool | SPC (Layer 1) |
| Function | Inference |
| Source rank | 0.85 (controlled observation) |
| Graph interaction | Creates evidence that updates hypotheses about process state |
| Measurement gate | Yes — applies to the measured CTQ |

**Output schema:**
```json
{
  "signal_type": "special" | "common" | "none",
  "signals": [
    {
      "rule": "nelson_1" | "nelson_2" | ... | "nelson_8",
      "points": [{"index": int, "value": float}],
      "description": "string"
    }
  ],
  "capability": {
    "cpk": float | null,
    "cp": float | null,
    "pp": float | null,
    "ppk": float | null
  },
  "control_limits": {
    "ucl": float,
    "cl": float,
    "lcl": float
  },
  "n_subgroups": int,
  "chart_type": "i_mr" | "xbar_r" | "xbar_s" | "p" | "np" | "c" | "u"
}
```

**Evidence trigger:** When linked to an investigation and any signal is detected (signal_type ≠ "none"), or when capability results update a performance measure.

**Graph mapping:**
- Special cause signal → evidence supporting "assignable cause present" hypothesis
- Common cause + incapable → evidence supporting "systemic variation exceeds tolerance" hypothesis
- Capability values → evidence updating performance measure hypotheses

#### **5.2.2 DSW (Statistical Tests)**

| Field | Value |
|-------|-------|
| Tool | DSW (Layer 1) |
| Function | Inference |
| Source rank | 0.75 (statistical test) |
| Graph interaction | Creates evidence that updates hypotheses about relationships between variables |
| Measurement gate | Yes — applies to measured variables |

**Output schema:**
```json
{
  "analysis_type": "string",
  "test_statistic": float,
  "p_value": float | null,
  "effect_size": float | null,
  "confidence_interval": {"low": float, "high": float} | null,
  "sample_size": int,
  "practical_significance": "strong" | "moderate" | "weak" | "inconclusive",
  "evidence_grade": "Strong" | "Moderate" | "Weak" | "Inconclusive",
  "assumptions_met": {"normality": bool, "homoscedasticity": bool, ...}
}
```

**Evidence trigger:** When linked to an investigation and analysis completes with a result.

**Graph mapping:**
- Significant result (p < α, practical significance ≥ moderate) → evidence supporting the tested hypothesis
- Non-significant result → evidence weakening the tested hypothesis (absence of evidence, weighted lower)
- Effect size and CI → stored as evidence metadata, available for compound evidence assessment

#### **5.2.3 DOE**

| Field | Value |
|-------|-------|
| Tool | DOE (Layer 1) |
| Function | Intent |
| Source rank | 0.95 (designed experiment) — applies to results, not design |
| Graph interaction | **Design phase:** prescribes which hypotheses to test and how. **Results phase:** creates highest-ranked evidence. |
| Measurement gate | Yes — applies to response variable measurement |

**Output schema (design):**
```json
{
  "design_type": "full_factorial" | "fractional" | "ccd" | "latin_square" | "taguchi" | "custom",
  "factors": [
    {"name": "string", "levels": [float], "hypothesis_id": "uuid" | null}
  ],
  "responses": [{"name": "string", "target": float | null}],
  "runs": int,
  "replicates": int,
  "blocks": int,
  "power": float,
  "randomization_seed": int
}
```

**Output schema (results):**
```json
{
  "significant_factors": [
    {"name": "string", "effect": float, "p_value": float}
  ],
  "optimal_settings": {"factor_name": float},
  "model_r_squared": float,
  "residual_diagnostics": {"normality_p": float, "independence": bool}
}
```

**Evidence trigger:**
- Design phase: no evidence created — design structures intent, not inference
- Results phase: when experimental results are analyzed, evidence created at source rank 0.95

**Graph mapping:**
- Design → links factors to hypotheses (prescriptive, no belief update)
- Results → significant factors become strong evidence for/against causal hypotheses
- Non-significant factors → evidence weakening those causal links

#### **5.2.4 ML**

| Field | Value |
|-------|-------|
| Tool | ML (Layer 1) |
| Function | Inference |
| Source rank | 0.50 (model-dependent, equivalent to simulation) |
| Graph interaction | Creates evidence about predictive relationships, not causal |
| Measurement gate | Yes — applies to training data measurements |

**Note:** ML is ranked at simulation level (0.50) because models learn correlations, not causal structure. Feature importance ≠ causal importance. A random forest that identifies "supplier" as the top feature is a hypothesis generator (conjecture phase), not a causal confirmation. ML evidence should inform which hypotheses to test with DOE, not confirm them.

**Evidence trigger:** When linked to an investigation and model produces predictions or feature importance.

**Graph mapping:**
- Feature importance → suggests hypotheses (information function, despite being Layer 1)
- Prediction accuracy on held-out data → evidence about model validity, not process causation

#### **5.2.5 Triage**

| Field | Value |
|-------|-------|
| Tool | Triage (Layer 1) |
| Function | Inference |
| Source rank | N/A — Triage does not produce evidence |
| Graph interaction | None — Triage produces clean datasets, not claims |
| Measurement gate | N/A |

Triage is a data preparation tool. It does not produce evidence or interact with the investigation graph. Its output (clean dataset, quality score) is consumed by other tools that produce evidence.

#### **5.2.6 Forecast**

| Field | Value |
|-------|-------|
| Tool | Forecast (Layer 1) |
| Function | Inference |
| Source rank | 0.50 (model-dependent) |
| Graph interaction | Creates evidence about trend direction and anomalies |
| Measurement gate | Yes — applies to time series measurements |

**Evidence trigger:** When linked to an investigation and anomaly detected or trend projection diverges from target.

**Graph mapping:**
- Trend projection → evidence about future process state (weak — projection, not observation)
- Anomaly detection → evidence supporting "process change occurred" hypothesis (stronger — observed deviation)

### **5.3 Layer 2 Tool Contracts**

#### **5.3.1 RCA (Root Cause Analysis)**

| Field | Value |
|-------|-------|
| Tool | RCA (Layer 2) |
| Function | Information |
| Source rank | 0.60 (structured analysis) |
| Graph interaction | Builds a linear causal path (chain of hypotheses linked by causal edges) |
| Measurement gate | No — structures knowledge, not measurements |

**Output schema:**
```json
{
  "effect": "string",
  "chain": [
    {
      "why": "string",
      "depth": int,
      "hypothesis_id": "uuid" | null,
      "accepted": bool
    }
  ],
  "root_cause": "string" | null,
  "countermeasure": "string" | null,
  "status": "in_progress" | "root_cause_identified" | "countermeasure_defined"
}
```

**Graph mapping:**
- Each accepted "why" → hypothesis node in the graph
- Sequential whys → causal links (why₁ → why₂ → ... → root cause)
- Root cause → terminal hypothesis with countermeasure attached
- The chain is a single path, not a tree — RCA investigates one causal thread

#### **5.3.2 Ishikawa (Cause & Effect Diagram)**

| Field | Value |
|-------|-------|
| Tool | Ishikawa (Layer 2) |
| Function | Information |
| Source rank | 0.60 (structured analysis) |
| Graph interaction | Populates competing hypotheses across 6M categories, all pointing to a single effect |
| Measurement gate | No |

**Output schema:**
```json
{
  "effect": "string",
  "categories": {
    "man": [{"cause": "string", "sub_causes": ["string"]}],
    "machine": [...],
    "material": [...],
    "method": [...],
    "measurement": [...],
    "environment": [...]
  }
}
```

**Graph mapping:**
- Effect → the target node (what we're investigating)
- Each top-level cause → hypothesis node linked to effect
- Sub-causes → child hypotheses linked to parent cause
- All hypothesis nodes are competing explanations (disjunctive — any subset could contribute)
- Unlike RCA's linear chain, Ishikawa produces a fan structure

#### **5.3.3 C&E Matrix**

| Field | Value |
|-------|-------|
| Tool | C&E Matrix (Layer 2) |
| Function | Information |
| Source rank | 0.60 (structured analysis) |
| Graph interaction | Assigns prior weights to existing hypotheses based on team scoring |
| Measurement gate | No |

**Output schema:**
```json
{
  "outputs": [{"name": "string", "weight": int}],
  "inputs": [
    {
      "name": "string",
      "scores": [int],
      "total": int,
      "rank": int,
      "hypothesis_id": "uuid" | null
    }
  ]
}
```

**Graph mapping:**
- C&E Matrix does not create new hypotheses — it scores existing ones
- **Prerequisite:** C&E Matrix presupposes a populated hypothesis set. If no hypotheses exist in the investigation (no prior Ishikawa, FMEA, or manual hypothesis entry), the system surfaces a workflow error: "C&E Matrix requires existing hypotheses to score — use Ishikawa or add hypotheses first"
- Total scores → normalized to [0, 1] → set as priors on corresponding hypothesis nodes
- Higher-scored inputs get higher priors, focusing the investigation on the most likely contributors

#### **5.3.4 FMEA**

| Field | Value |
|-------|-------|
| Tool | FMEA (Layer 2) |
| Function | Information |
| Source rank | 0.60 (structured analysis) |
| Graph interaction | Populates hypotheses (failure modes) with risk-weighted priors |
| Measurement gate | No |

**Output schema:**
```json
{
  "rows": [
    {
      "failure_mode": "string",
      "effect": "string",
      "cause": "string",
      "severity": int,
      "occurrence": int,
      "detection": int,
      "rpn": int,
      "recommended_action": "string" | null,
      "hypothesis_id": "uuid" | null
    }
  ]
}
```

**Graph mapping:**
- Each failure mode → hypothesis ("this failure mode occurs")
- Cause → causal link to the failure mode hypothesis
- Effect → downstream consequence (linked to higher-level hypothesis)
- RPN → converted to prior: higher RPN = higher prior probability of occurrence
- Severity ≥ 8 → flags for deeper investigation (CANON-001 §3.3.1)

#### **5.3.5 A3 Report**

| Field | Value |
|-------|-------|
| Tool | A3 (Layer 2) |
| Function | Report |
| Source rank | N/A — A3 synthesizes, does not produce evidence |
| Graph interaction | Reads investigation state, does not modify it |
| Measurement gate | No |

A3 is a report sink (CANON-001 §3.3.1). It consumes the investigation graph state and renders it as a structured report. It does not create hypotheses, evidence, or causal links. Its sections (background, current condition, root cause, countermeasure, implementation plan) are populated from the investigation graph and linked tool outputs.

#### **5.3.6 VSM (Value Stream Mapping)**

| Field | Value |
|-------|-------|
| Tool | VSM (Layer 2) |
| Function | Information |
| Source rank | N/A — VSM feeds Layer 3 directly, not the investigation graph |
| Graph interaction | None — VSM produces kaizen proposals, not hypotheses |
| Measurement gate | No |

VSM bypasses both the investigation engine and the evidence bridge (CANON-001 §3.2). The future state diff is organizational intent, not evidence. VSM → Hoshin is a direct path.

#### **5.3.7 8D Report**

| Field | Value |
|-------|-------|
| Tool | 8D (Layer 2) |
| Function | Report |
| Source rank | N/A — 8D synthesizes, does not produce evidence |
| Graph interaction | Reads investigation state, does not modify it |
| Measurement gate | No |

8D is a report format for customer complaint investigations. Like A3, it consumes investigation output and structures it for external communication. 8D chains are not yet defined (CANON-001 §3.3.4).

---

## **6. CHAINING CONTRACTS**

### **6.1 Principle**

A chain contract defines what happens when Tool A's output becomes Tool B's input in the same investigation. The key distinction from tool contracts (§5):

- **Tool contract:** single tool → graph (what does this tool produce, how does it map)
- **Chain contract:** Tool A output → Tool B input → graph (how does sequential operation affect the graph)

Chaining is always user-initiated (CANON-001 §3.3). The system offers the chain option; the practitioner decides.

### **6.2 Information → Information Chains**

These chains pass structured causal knowledge from one information tool to another.

#### **6.2.1 FMEA → RCA**

**Trigger:** Severity ≥ 8 (org-configurable)
**What passes:** Failure mode becomes RCA effect (investigation target)

```json
{
  "source_tool": "fmea",
  "source_row_id": "uuid",
  "target_tool": "rca",
  "context": {
    "effect": "{failure_mode} — {effect}",
    "severity": int,
    "initial_cause": "{cause}"
  }
}
```

**Graph effect:** RCA begins building a causal chain from the FMEA failure mode hypothesis. The RCA's linear chain extends the FMEA's failure mode node deeper into root cause territory.

#### **6.2.2 FMEA → Ishikawa**

**Trigger:** Severity ≥ 8 (org-configurable)
**What passes:** Failure mode becomes Ishikawa effect (the thing to map causes for)

```json
{
  "source_tool": "fmea",
  "source_row_id": "uuid",
  "target_tool": "ishikawa",
  "context": {
    "effect": "{failure_mode} — {effect}",
    "severity": int
  }
}
```

**Graph effect:** Ishikawa creates a fan of competing hypotheses around the FMEA failure mode node. Where FMEA → RCA goes deep (linear chain), FMEA → Ishikawa goes wide (multiple categories).

#### **6.2.3 FMEA → C&E Matrix**

**Trigger:** Severity ≥ 8 (org-configurable)
**What passes:** Failure mode becomes a C&E Matrix output to score against

```json
{
  "source_tool": "fmea",
  "source_row_id": "uuid",
  "target_tool": "ce_matrix",
  "context": {
    "output": "{failure_mode} — {effect}",
    "weight": int
  }
}
```

**Graph effect:** C&E Matrix scores existing hypotheses against the FMEA failure mode, adjusting priors based on team-weighted assessment.

#### **6.2.4 Ishikawa → C&E Matrix**

**Trigger:** On Ishikawa completion (1:1 mapping)
**What passes:** Top-level causes become C&E Matrix inputs

```json
{
  "source_tool": "ishikawa",
  "source_id": "uuid",
  "target_tool": "ce_matrix",
  "context": {
    "inputs": [
      {"name": "string", "category": "man|machine|material|method|measurement|environment"}
    ],
    "effect": "string"
  }
}
```

**Graph effect:** C&E Matrix operates on the hypotheses that Ishikawa already created. It does not add new hypotheses — it scores and prioritizes the existing ones. This is the canonical Ishikawa → C&E chain: Ishikawa maps the territory (creates the hypothesis set), C&E Matrix ranks it (assigns priors). Using C&E Matrix without a prior hypothesis-generating step is a workflow error (see §5.3.3).

### **6.3 Information → Report Chains**

These chains pass investigation state to report tools.

#### **6.3.1 RCA → A3**

**Trigger:** On RCA completion
**What passes:** Root cause + countermeasure → A3 root cause section

```json
{
  "source_tool": "rca",
  "source_id": "uuid",
  "target_tool": "a3",
  "context": {
    "root_cause": "string",
    "countermeasure": "string",
    "chain_summary": "string",
    "evidence_ids": ["uuid"]
  }
}
```

#### **6.3.2 Ishikawa → A3**

**Trigger:** On Ishikawa completion
**What passes:** Contributor map → A3 root cause section

```json
{
  "source_tool": "ishikawa",
  "source_id": "uuid",
  "target_tool": "a3",
  "context": {
    "top_causes": [{"cause": "string", "category": "string"}],
    "effect": "string"
  }
}
```

#### **6.3.3 C&E Matrix → A3**

**Trigger:** On C&E Matrix completion
**What passes:** Prioritized causes → A3 root cause section

```json
{
  "source_tool": "ce_matrix",
  "source_id": "uuid",
  "target_tool": "a3",
  "context": {
    "ranked_inputs": [{"name": "string", "total": int, "rank": int}],
    "top_n": 3
  }
}
```

**Graph effect for all → A3 chains:** None. A3 reads the graph, it does not modify it. The chain passes context to pre-fill the report sections.

### **6.4 Layer 3 → Layer 1-2 Chains**

These chains initiate investigation from Layer 3 containers.

#### **6.4.1 NCR → Investigation Tools**

**Trigger:** Practitioner initiates investigation from NCR
**What passes:** Nonconformance description becomes investigation target

```json
{
  "source_tool": "ncr",
  "source_id": "uuid",
  "target_tool": "rca" | "ishikawa" | "fmea",
  "context": {
    "nonconformance": "string",
    "severity": "critical" | "major" | "minor",
    "containment_action": "string" | null,
    "investigation_id": "uuid" | null
  }
}
```

**Graph effect:** If an investigation exists on the NCR, the target tool joins it. If not, a new investigation is created with the nonconformance as the root effect node.

#### **6.4.2 CAPA → Investigation Tools**

**Trigger:** Practitioner initiates investigation from CAPA
**What passes:** Corrective action target becomes investigation focus

```json
{
  "source_tool": "capa",
  "source_id": "uuid",
  "target_tool": "rca" | "a3",
  "context": {
    "corrective_action": "string",
    "related_ncr_id": "uuid" | null,
    "investigation_id": "uuid" | null
  }
}
```

**Graph effect:** Same as NCR — joins existing investigation or creates new one.

---

## **7. INVESTIGATION LIFECYCLE**

### **7.1 States**

```
open → active → concluded → exported
```

| State | Meaning | Transitions |
|-------|---------|-------------|
| **open** | Investigation created, no tools connected yet | → active (first tool connects) |
| **active** | Tools connected, hypotheses being built, evidence accumulating | → concluded (practitioner signs off or confirmed hypothesis exists) |
| **concluded** | Investigation complete — practitioner has signed off on the conclusion | → exported (bridge exports to Layer 3) |
| **exported** | Conclusion package delivered to Layer 3 container | Terminal |

### **7.1.1 Investigation Model**

**File:** `core/models/investigation.py`

```python
# core/models/investigation.py

import uuid
from django.db import models
from django.conf import settings
from syn.core.models import SynaraEntity


class Investigation(SynaraEntity):
    """
    A structured problem-solving session (CANON-002 §7).
    The investigation graph lives in synara_state (JSON).
    Tools connect via M2M. Evidence flows through Synara.
    """

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        ACTIVE = "active", "Active"
        CONCLUDED = "concluded", "Concluded"
        EXPORTED = "exported", "Exported"

    class MemberRole(models.TextChoices):
        OWNER = "owner", "Owner"
        CONTRIBUTOR = "contributor", "Contributor"
        VIEWER = "viewer", "Viewer"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True, default="")
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.OPEN
    )

    # Ownership
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="owned_investigations"
    )
    tenant = models.ForeignKey(
        "core.Tenant", null=True, blank=True, on_delete=models.CASCADE,
        related_name="investigations"
    )

    # Synara state — the causal graph serialized as JSON
    synara_state = models.JSONField(default=dict, blank=True)

    # Versioning (§7.3)
    version = models.PositiveIntegerField(default=1)
    parent_version = models.ForeignKey(
        "self", null=True, blank=True, on_delete=models.SET_NULL,
        related_name="child_versions",
        help_text="Previous version this was reopened from"
    )

    # Layer 3 linkage (for export)
    exported_to_project = models.ForeignKey(
        "core.Project", null=True, blank=True, on_delete=models.SET_NULL,
        related_name="source_investigations"
    )
    export_package = models.JSONField(
        null=True, blank=True,
        help_text="Conclusion package JSON (§9) frozen at export time"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    concluded_at = models.DateTimeField(null=True, blank=True)
    exported_at = models.DateTimeField(null=True, blank=True)

    # Members
    members = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        through="InvestigationMembership",
        related_name="investigations"
    )

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.title} (v{self.version}, {self.status})"

    def transition_to(self, target_status, user):
        """
        State machine per CANON-002 §7.2.
        Returns True on success, raises ValueError on invalid transition.
        """
        VALID_TRANSITIONS = {
            self.Status.OPEN: [self.Status.ACTIVE],
            self.Status.ACTIVE: [self.Status.CONCLUDED],
            self.Status.CONCLUDED: [self.Status.EXPORTED],
            self.Status.EXPORTED: [],  # Terminal
        }
        if target_status not in VALID_TRANSITIONS.get(self.status, []):
            raise ValueError(
                f"Cannot transition from {self.status} to {target_status}"
            )
        self.status = target_status
        if target_status == self.Status.CONCLUDED:
            from django.utils import timezone
            self.concluded_at = timezone.now()
        if target_status == self.Status.EXPORTED:
            from django.utils import timezone
            self.exported_at = timezone.now()
        self.save()
        return True

    def reopen(self, user):
        """
        Create a new version from a concluded investigation (§7.3).
        Returns the new Investigation instance.
        """
        if self.status not in (self.Status.CONCLUDED, self.Status.EXPORTED):
            raise ValueError("Can only reopen concluded or exported investigations")

        import copy
        new_inv = Investigation(
            title=self.title,
            description=self.description,
            status=self.Status.ACTIVE,
            owner=user,
            tenant=self.tenant,
            synara_state=copy.deepcopy(self.synara_state),
            version=self.version + 1,
            parent_version=self,
        )
        new_inv.save()

        # Copy membership
        for membership in self.investigationmembership_set.all():
            InvestigationMembership.objects.create(
                investigation=new_inv,
                user=membership.user,
                role=membership.role,
            )

        return new_inv


class InvestigationMembership(models.Model):
    """M2M through table for investigation membership (§7.4)."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    investigation = models.ForeignKey(Investigation, on_delete=models.CASCADE)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    role = models.CharField(
        max_length=20,
        choices=Investigation.MemberRole.choices,
        default=Investigation.MemberRole.CONTRIBUTOR,
    )
    joined_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("investigation", "user")]
```

<!-- impl: core/models/investigation.py::Investigation -->
<!-- impl: core/models/investigation.py::Investigation.transition_to -->
<!-- impl: core/models/investigation.py::Investigation.reopen -->
<!-- impl: core/models/investigation.py::InvestigationMembership -->

### **7.2 Transition Rules**

- **open → active:** Automatic when the first tool connects to the investigation or the first hypothesis is created.
- **active → concluded:** Practitioner-initiated. Does not require a confirmed hypothesis — the practitioner may conclude that the investigation is sufficient based on their judgment. The system surfaces a summary (top hypothesis, posterior, unresolved expansion signals) to inform the decision, but does not gate it.
- **concluded → exported:** Triggered when the practitioner links the concluded investigation to a Layer 3 container and initiates export. The bridge creates the conclusion package (§9).

### **7.3 Reopening**

A concluded investigation can be reopened. Reopening creates a **new version** of the investigation — it does not mutate the concluded state. The original conclusion and its export (if any) remain intact in the audit trail.

```
Investigation v1: open → active → concluded → exported
                                       ↓ (reopen)
Investigation v1.1: active → concluded → exported
```

The new version inherits the full graph state (hypotheses, evidence, causal links) from the concluded version. New evidence and hypotheses are added to the new version. The version chain is navigable — any version can reference its predecessor.

### **7.4 Investigation Membership**

Investigations have a membership model:

| Role | Permissions |
|------|-------------|
| **owner** | Created the investigation. Can conclude, export, reopen, add/remove members. |
| **contributor** | Can connect tools, add hypotheses, add evidence. Cannot conclude or export. |
| **viewer** | Read-only access to investigation state. |

Membership follows the same tenant model as the rest of the platform (CANON-001 §5.4). Individual (Pro) investigations have a single owner. Team/Enterprise investigations can have multiple contributors.

---

## **8. EVIDENCE SUPERSESSION**

### **8.1 Principle**

Evidence is immutable. When a tool re-runs with new data and produces a different result, the original evidence node is not modified. A new evidence node is created with a `supersedes_id` FK pointing to the prior node.

### **8.2 Mechanics**

```
Evidence A (SPC chart, Cpk=0.72, 2026-03-01)
  ↑ superseded by
Evidence B (SPC chart, Cpk=1.10, 2026-03-15, supersedes_id=A)
```

- The graph walks to the **most recent non-superseded node** when computing posteriors
- Superseded nodes are excluded from active belief updates but remain in the audit trail
- The supersession chain is navigable: any evidence node can trace its full history
- `supersedes_id` is nullable — original evidence has no predecessor

### **8.3 Evidence Model Addition**

```python
# Addition to core.Evidence
supersedes = models.ForeignKey(
    'self', null=True, blank=True, on_delete=models.SET_NULL,
    related_name='superseded_by',
    help_text="Prior evidence node this supersedes"
)
```

### **8.4 Supersession Detection**

**File:** `agents_api/investigation_bridge.py`

When a tool re-runs and produces new evidence for the same investigation, the system must detect that the new evidence supersedes a prior node. Detection is based on matching `(source_tool, source_id, investigation_id)` — same tool instance producing a new result for the same investigation.

```python
def _detect_and_apply_supersession(
    investigation: Investigation,
    source_tool: str,
    source_id: str,
    new_evidence_id: str,
):
    """
    Check if prior evidence from the same tool instance exists in this
    investigation. If so, mark it as superseded by the new evidence.
    CANON-002 §8.
    """
    from core.models.hypothesis import Evidence

    prior = (
        Evidence.objects
        .filter(
            source_tool=source_tool,
            source_id=source_id,
        )
        .exclude(id=new_evidence_id)
        .exclude(
            # Already superseded by something else
            id__in=Evidence.objects
            .filter(supersedes__isnull=False)
            .values("supersedes_id")
        )
        .order_by("-created_at")
        .first()
    )

    if prior:
        new_evidence = Evidence.objects.get(id=new_evidence_id)
        new_evidence.supersedes = prior
        new_evidence.save(update_fields=["supersedes"])

        logger.info(
            "evidence.superseded",
            extra={
                "new_id": str(new_evidence_id),
                "superseded_id": str(prior.id),
                "source_tool": source_tool,
                "investigation_id": str(investigation.id),
            }
        )
```

<!-- impl: agents_api/investigation_bridge.py::_detect_and_apply_supersession -->

This function is called inside `connect_tool()` after evidence creation for inference tools:

```python
# Inside connect_tool(), after synara.create_evidence():
if tool_function == "inference":
    # ... existing evidence creation ...
    _detect_and_apply_supersession(
        investigation=investigation,
        source_tool=tool_type,
        source_id=str(tool_output.id),
        new_evidence_id=update_result.evidence_id,
    )
```

### **8.5 Graph Walk Rule**

When Synara computes posteriors, it filters evidence:

```python
active_evidence = Evidence.objects.filter(
    investigation=investigation
).exclude(
    id__in=Evidence.objects.filter(supersedes__isnull=False).values('supersedes_id')
)
```

Only non-superseded evidence participates in belief updates. This is equivalent to "use the latest version of each evidence source."

---

## **9. LAYER 3 EXPORT SCHEMA**

### **9.1 Conclusion Package**

When an investigation is exported to a Layer 3 container, the evidence bridge creates a conclusion package — a structured JSON summary of the investigation state.

```json
{
  "investigation_id": "uuid",
  "investigation_version": int,
  "status": "concluded",
  "concluded_at": "iso8601",
  "concluded_by": "uuid (user)",

  "top_hypothesis": {
    "id": "uuid",
    "description": "string",
    "posterior": float,
    "status": "confirmed" | "uncertain" | "rejected",
    "causal_chain": [
      {
        "hypothesis_id": "uuid",
        "description": "string",
        "posterior": float,
        "link_strength": float,
        "mechanism": "string"
      }
    ]
  },

  "competing_hypotheses": [
    {
      "id": "uuid",
      "description": "string",
      "posterior": float,
      "status": "string"
    }
  ],

  "evidence_summary": [
    {
      "id": "uuid",
      "summary": "string",
      "source_tool": "string",
      "evidence_weight": float,
      "supports": ["uuid"],
      "weakens": ["uuid"]
    }
  ],

  "unresolved_signals": [
    {
      "signal_type": "expansion",
      "description": "string",
      "created_at": "iso8601"
    }
  ],

  "investigation_metadata": {
    "tools_used": ["string"],
    "evidence_count": int,
    "hypothesis_count": int,
    "duration_days": int
  }
}
```

### **9.2 Export Implementation**

**File:** `agents_api/investigation_bridge.py`

```python
def export_investigation(
    investigation_id: str,
    target_project_id: str,
    user,
) -> dict:
    """
    Export a concluded investigation to a Layer 3 container (CANON-002 §9).

    Creates:
    1. Conclusion package JSON (frozen snapshot)
    2. core.Evidence record on the target project
    3. State transition: concluded → exported

    Returns the conclusion package dict.
    Raises ValueError if investigation is not in 'concluded' state.
    """
    investigation = get_investigation(investigation_id, user)
    if investigation.status != Investigation.Status.CONCLUDED:
        raise ValueError(
            f"Cannot export investigation in '{investigation.status}' state — "
            "must be 'concluded'"
        )

    from core.models import Project, Evidence
    from django.utils import timezone

    target = Project.objects.get(id=target_project_id, user=user)
    synara = load_synara(investigation)

    # Build conclusion package
    package = _build_conclusion_package(investigation, synara, user)

    # Determine top hypothesis
    top_h = package["top_hypothesis"]

    # Create Evidence record on target project
    Evidence.objects.create(
        project=target,
        source_type="investigation",
        result_type="qualitative",
        summary=(
            f"{top_h['description']} "
            f"(posterior: {top_h['posterior']:.2f})"
        ),
        confidence=top_h["posterior"],
        details=_build_export_details(package),
        raw_output=package,
        source_tool="investigation",
        source_id=str(investigation.id),
    )

    # Freeze package on investigation
    investigation.export_package = package
    investigation.exported_to_project = target
    investigation.save(update_fields=[
        "export_package", "exported_to_project", "updated_at"
    ])

    # State transition
    investigation.transition_to(Investigation.Status.EXPORTED, user)

    logger.info(
        "investigation.exported",
        extra={
            "investigation_id": str(investigation.id),
            "target_project_id": str(target.id),
            "top_posterior": top_h["posterior"],
            "evidence_count": package["investigation_metadata"]["evidence_count"],
        }
    )

    return package


def _build_conclusion_package(
    investigation: Investigation,
    synara: Synara,
    user,
) -> dict:
    """Build the §9.1 conclusion package from current Synara state."""
    from django.utils import timezone

    hypotheses = synara.graph.hypotheses
    evidence_list = synara.graph.evidence
    links = synara.graph.links

    # Sort hypotheses by posterior descending
    sorted_h = sorted(
        hypotheses.values(),
        key=lambda h: h.posterior,
        reverse=True,
    )

    top = sorted_h[0] if sorted_h else None

    # Build causal chain for top hypothesis (walk incoming links)
    causal_chain = []
    if top:
        causal_chain = _trace_causal_chain(top.id, hypotheses, links)

    # Classify hypothesis status
    def h_status(posterior: float) -> str:
        if posterior >= 0.85:
            return "confirmed"
        if posterior <= 0.15:
            return "rejected"
        return "uncertain"

    # Expansion signals
    expansion_signals = [
        {
            "signal_type": "expansion",
            "description": sig.description,
            "created_at": sig.created_at.isoformat()
            if hasattr(sig, "created_at") else None,
        }
        for sig in getattr(synara, "expansion_signals", [])
    ]

    # Tools used (from InvestigationToolLink)
    from core.models.investigation import InvestigationToolLink
    tool_types = list(
        InvestigationToolLink.objects
        .filter(investigation=investigation)
        .values_list("tool_type", flat=True)
        .distinct()
    )

    # Duration
    duration_days = 0
    if investigation.concluded_at and investigation.created_at:
        duration_days = (
            investigation.concluded_at - investigation.created_at
        ).days

    return {
        "investigation_id": str(investigation.id),
        "investigation_version": investigation.version,
        "status": "concluded",
        "concluded_at": (
            investigation.concluded_at.isoformat()
            if investigation.concluded_at else None
        ),
        "concluded_by": str(user.id),

        "top_hypothesis": {
            "id": str(top.id) if top else None,
            "description": top.description if top else "No hypotheses",
            "posterior": round(top.posterior, 4) if top else 0.0,
            "status": h_status(top.posterior) if top else "uncertain",
            "causal_chain": causal_chain,
        },

        "competing_hypotheses": [
            {
                "id": str(h.id),
                "description": h.description,
                "posterior": round(h.posterior, 4),
                "status": h_status(h.posterior),
            }
            for h in sorted_h[1:]  # All except top
            if h.posterior > 0.15  # Exclude rejected
        ],

        "evidence_summary": [
            {
                "id": str(e.id),
                "summary": e.event,
                "source_tool": getattr(e, "source", "unknown"),
                "evidence_weight": round(e.strength, 4),
                "supports": [str(s) for s in getattr(e, "supports", [])],
                "weakens": [str(w) for w in getattr(e, "weakens", [])],
            }
            for e in evidence_list
        ],

        "unresolved_signals": expansion_signals,

        "investigation_metadata": {
            "tools_used": tool_types,
            "evidence_count": len(evidence_list),
            "hypothesis_count": len(hypotheses),
            "duration_days": duration_days,
        },
    }


def _trace_causal_chain(
    hypothesis_id: str,
    hypotheses: dict,
    links: list,
) -> list[dict]:
    """
    Walk causal links backward from the top hypothesis to build a chain.
    Returns list of dicts in causal order (root cause → ... → top hypothesis).
    Max depth 20 to prevent cycles.
    """
    chain = []
    visited = set()
    current = hypothesis_id
    max_depth = 20

    while current and len(chain) < max_depth:
        if current in visited:
            break  # Cycle detected
        visited.add(current)

        # Find links pointing TO current hypothesis
        incoming = [
            link for link in links
            if link.to_id == current
        ]

        if not incoming:
            break

        # Take strongest incoming link
        strongest = max(incoming, key=lambda l: l.strength)
        source_h = hypotheses.get(strongest.from_id)
        if not source_h:
            break

        chain.append({
            "hypothesis_id": str(source_h.id),
            "description": source_h.description,
            "posterior": round(source_h.posterior, 4),
            "link_strength": round(strongest.strength, 4),
            "mechanism": getattr(strongest, "mechanism", ""),
        })

        current = strongest.from_id

    chain.reverse()  # Root cause first
    return chain


def _build_export_details(package: dict) -> str:
    """Build human-readable details string for the Evidence record."""
    n_signals = len(package.get("unresolved_signals", []))
    details = (
        f"Investigation conclusion — "
        f"{package['investigation_metadata']['evidence_count']} evidence nodes, "
        f"{package['investigation_metadata']['hypothesis_count']} hypotheses, "
        f"{package['investigation_metadata']['duration_days']} days"
    )
    if n_signals > 0:
        details += (
            f". WARNING: Investigation has {n_signals} unresolved expansion "
            f"signals — causal surface may be incomplete"
        )
    return details
```

<!-- impl: agents_api/investigation_bridge.py::export_investigation -->
<!-- impl: agents_api/investigation_bridge.py::_build_conclusion_package -->
<!-- impl: agents_api/investigation_bridge.py::_trace_causal_chain -->

---

## **10. HYPOTHESIS CONFIRMATION**

Hypothesis status is determined by posterior probability:

| Posterior | Status | Effect |
|-----------|--------|--------|
| ≥ 0.85 | **Confirmed** | Expansion signals suppressed for this hypothesis. Hypothesis treated as established in causal chain. |
| 0.15 – 0.85 | **Uncertain** | Active investigation target. Evidence continues to accumulate. |
| ≤ 0.15 | **Rejected** | Removed from active causal paths. Causal links from this hypothesis are deactivated (strength set to 0). Node remains in graph for audit. |

Confirmation and rejection are automatic based on posterior updates — the practitioner does not manually set status. The practitioner can override by adding contradictory evidence or manually adjusting the hypothesis, which triggers a re-evaluation.

### **10.1 Implementation**

**File:** `agents_api/investigation_bridge.py`

Confirmation checks run after every posterior update inside `connect_tool()`:

```python
# Thresholds — CANON-002 §10
CONFIRMED_THRESHOLD = 0.85
REJECTED_THRESHOLD = 0.15


def _apply_confirmation_thresholds(
    investigation: Investigation,
    synara: Synara,
    posteriors: dict[str, float],
) -> list[dict]:
    """
    Check all updated hypotheses against confirmation/rejection thresholds.
    Returns list of status change events for logging.

    On confirmation:
      - Suppress expansion signals for that hypothesis
      - Mark hypothesis as confirmed in graph metadata

    On rejection:
      - Deactivate outgoing causal links (strength → 0)
      - Mark hypothesis as rejected in graph metadata
      - Node remains in graph for audit trail
    """
    events = []

    for h_id, posterior in posteriors.items():
        h = synara.graph.hypotheses.get(h_id)
        if not h:
            continue

        prev_status = getattr(h, "confirmation_status", "uncertain")

        if posterior >= CONFIRMED_THRESHOLD and prev_status != "confirmed":
            h.confirmation_status = "confirmed"
            # Suppress expansion signals for this hypothesis
            synara.suppress_expansion(h_id)
            events.append({
                "hypothesis_id": h_id,
                "transition": f"{prev_status} → confirmed",
                "posterior": round(posterior, 4),
            })

        elif posterior <= REJECTED_THRESHOLD and prev_status != "rejected":
            h.confirmation_status = "rejected"
            # Deactivate outgoing causal links
            for link in synara.graph.links:
                if link.from_id == h_id:
                    link.strength = 0.0
            events.append({
                "hypothesis_id": h_id,
                "transition": f"{prev_status} → rejected",
                "posterior": round(posterior, 4),
            })

        elif (
            REJECTED_THRESHOLD < posterior < CONFIRMED_THRESHOLD
            and prev_status != "uncertain"
        ):
            # Re-entered uncertain zone (e.g. new contradictory evidence)
            h.confirmation_status = "uncertain"
            # Re-enable expansion signals
            synara.unsuppress_expansion(h_id)
            # Re-enable links (restore to default strength)
            for link in synara.graph.links:
                if link.from_id == h_id and link.strength == 0.0:
                    link.strength = 0.7  # Default causal link strength
            events.append({
                "hypothesis_id": h_id,
                "transition": f"{prev_status} → uncertain",
                "posterior": round(posterior, 4),
            })

    if events:
        logger.info(
            "investigation.confirmation_changes",
            extra={
                "investigation_id": str(investigation.id),
                "changes": events,
            }
        )

    return events
```

<!-- impl: agents_api/investigation_bridge.py::_apply_confirmation_thresholds -->
<!-- impl: agents_api/investigation_bridge.py::CONFIRMED_THRESHOLD -->
<!-- impl: agents_api/investigation_bridge.py::REJECTED_THRESHOLD -->

This is called inside `connect_tool()` after inference produces posteriors:

```python
# Inside connect_tool(), after posterior update:
if tool_function == "inference" and result.get("posteriors"):
    confirmation_events = _apply_confirmation_thresholds(
        investigation, synara, result["posteriors"]
    )
    result["confirmation_changes"] = confirmation_events
```

**Reversal behavior:** If new evidence moves a confirmed hypothesis below 0.85 or a rejected hypothesis above 0.15, the status reverts to "uncertain" and suppressed signals / deactivated links are restored. This prevents premature lock-in — the practitioner always has the option to introduce contradictory evidence.

---

## **11. CONCURRENT INVESTIGATIONS**

### **11.1 Tool Membership**

A tool output can belong to multiple investigations via a generic linkage model:

```python
# core/models/investigation.py (addition)

class InvestigationToolLink(models.Model):
    """
    Links any tool output to an investigation (§11.1).
    Generic FK avoids adding M2M to every tool model.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    investigation = models.ForeignKey(
        Investigation, on_delete=models.CASCADE,
        related_name="tool_links"
    )
    # Generic FK to any tool model
    content_type = models.ForeignKey(
        "contenttypes.ContentType", on_delete=models.CASCADE
    )
    object_id = models.UUIDField()
    tool_output = GenericForeignKey("content_type", "object_id")

    # Metadata
    tool_type = models.CharField(max_length=30)  # "spc", "rca", "ishikawa", etc.
    tool_function = models.CharField(
        max_length=20,
        choices=[
            ("information", "Information"),
            ("inference", "Inference"),
            ("intent", "Intent"),
            ("report", "Report"),
        ]
    )
    linked_at = models.DateTimeField(auto_now_add=True)
    linked_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE
    )

    class Meta:
        unique_together = [("investigation", "content_type", "object_id")]
        # Prevents linking same tool output to same investigation twice
```

<!-- impl: core/models/investigation.py::InvestigationToolLink -->

**Tool function registry** (used to auto-populate `tool_function` on link creation):

```python
# agents_api/evidence_weights.py (addition)

TOOL_FUNCTIONS = {
    "spc": "inference",
    "dsw": "inference",
    "doe_design": "intent",
    "doe_results": "inference",
    "ml": "inference",
    "forecast": "inference",
    "triage": None,  # Cannot be linked — no graph interaction
    "rca": "information",
    "ishikawa": "information",
    "ce_matrix": "information",
    "fmea": "information",
    "a3": "report",
    "vsm": None,  # Feeds Layer 3 directly, not investigations
    "report": "report",
}
```

<!-- impl: agents_api/evidence_weights.py::TOOL_FUNCTIONS -->

When a tool produces output, it creates evidence in **each** connected investigation independently. The same SPC chart can be evidence in an NCR investigation and a Kaizen investigation simultaneously — the evidence weight and graph mapping are computed per-investigation.

### **11.2 Investigation Sharing**

Investigation membership (§7.4) controls access. Multiple practitioners can contribute to the same investigation. Tool outputs from any contributor are visible to all members.

### **11.3 Cross-Investigation Evidence**

Evidence from one investigation can be referenced by another, but it is not automatically shared. The practitioner must explicitly link evidence across investigations. This prevents contamination — an NCR investigation's conclusion should not automatically influence an unrelated Kaizen investigation.

---

## **12. INTEGRATION ARCHITECTURE**

The following specifications complete the integration layer between CANON-002's evidence methodology and Synara's belief engine:

### **12.1 Evidence Weight → Bayesian Update Mechanics — RESOLVED**

**Decision:** `evidence_weight` maps directly to Synara's `evidence.strength` field.

Synara's BeliefEngine already applies strength as a damper on likelihood:

```
likelihood = 0.5 + (base_likelihood - 0.5) × strength
```

This formula has the correct behavior for epistemological weighting:
- `strength = 1.0` → full likelihood (evidence fully trusted)
- `strength = 0.5` → likelihood pulled halfway to neutral (evidence partially trusted)
- `strength ≈ 0.0` → likelihood ≈ 0.5 (evidence not trusted, no belief update)

**Integration:**
1. Tool produces output
2. System computes `evidence_weight = source_rank × sample_modifier × measurement_validity × study_quality`
3. Clamp: `evidence.strength = max(0.05, min(0.99, evidence_weight))`
4. Evidence enters Synara with computed strength — no manual assignment
5. BeliefEngine's existing damper formula applies the weight during likelihood computation
6. Posterior update and propagation proceed unchanged

**Explicit supports/weakens are not double-counted.** When a tool declares `supports=[h_id]`, the base_likelihood is set to 0.8. The strength damper then scales it: a DOE supporting h gets `0.5 + (0.8 - 0.5) × 0.90 = 0.77`. An expert opinion supporting h gets `0.5 + (0.8 - 0.5) × 0.35 = 0.605`. Same declaration, different evidential weight. One signal path — no double-counting.

**What changes in Synara:**
- `evidence.strength` is no longer a user-supplied confidence knob
- `evidence.strength` is computed from the CANON-002 §3.1 formula
- Tools set this automatically based on their source rank and evidence properties
- The practitioner does not manually set strength — the methodology determines it

**What does NOT change in Synara:**
- The likelihood damper formula
- The posterior update formula (standard Bayes)
- Causal link propagation
- Expansion signal detection
- The explicit supports/weakens override mechanism

### **12.2 Measurement System Linkage Model**

**Data model chain:** Instrument → GageStudy → tool output linkage.

```python
class MeasurementSystem(SynaraEntity):
    """A physical measurement instrument or system."""
    id = UUIDField(primary_key=True)
    name = CharField(max_length=200)        # "Keyence IM-8000 #3"
    system_type = CharField(choices=[        # variable vs attribute
        ("variable", "Variable"),
        ("attribute", "Attribute"),
    ])
    owner = ForeignKey(User, on_delete=CASCADE)
    tenant = ForeignKey(Tenant, null=True, on_delete=CASCADE)
    calibration_due = DateField(null=True)
    status = CharField(choices=[
        ("active", "Active"),
        ("inactive", "Inactive"),
        ("quarantined", "Quarantined"),       # Failed GRR, pending resolution
    ])

    @property
    def current_validity(self) -> float:
        """Returns measurement_validity from most recent GageStudy, or 0.55 default."""
        study = self.gage_studies.order_by('-completed_at').first()
        if not study or not study.completed_at:
            return 0.55  # No GRR available default (CANON-002 §4.3)
        return study.measurement_validity


class GageStudy(SynaraEntity):
    """A Gage R&R or attribute agreement study linked to an instrument."""
    id = UUIDField(primary_key=True)
    measurement_system = ForeignKey(MeasurementSystem, related_name='gage_studies', on_delete=CASCADE)
    study_type = CharField(choices=[
        ("grr_crossed", "GRR Crossed"),
        ("grr_nested", "GRR Nested"),
        ("attribute_agreement", "Attribute Agreement"),
    ])
    completed_at = DateTimeField(null=True)

    # Results
    grr_percent = FloatField(null=True)      # %GRR for variable studies
    kappa = FloatField(null=True)            # Kappa for attribute studies
    percent_agreement = FloatField(null=True) # %Agreement for attribute studies
    ndc = IntegerField(null=True)            # Number of distinct categories

    @property
    def measurement_validity(self) -> float:
        """Compute validity per CANON-002 §4.1 / §4.4."""
        if self.study_type == "attribute_agreement":
            k = self.kappa
            if k is None:
                return 0.55
            if k >= 0.90:
                return 1.0
            if k >= 0.75:
                return 0.80
            if k >= 0.50:
                return 0.50
            return 0.10
        else:
            grr = self.grr_percent
            if grr is None:
                return 0.55
            if grr <= 10:
                return 1.0
            if grr <= 20:
                return 0.80
            if grr <= 30:
                return 0.50
            return 0.10
```

**Tool output linkage:** Each tool model that produces measurement-based evidence gets an optional FK:

```python
# On DSWResult, SPC inline results, ExperimentDesign, etc.
measurement_system = ForeignKey(
    MeasurementSystem, null=True, blank=True,
    on_delete=SET_NULL,
    help_text="Instrument used to measure the response/CTQ"
)
```

**Cascade:** When `compute_evidence_weight()` runs, it looks up `tool_output.measurement_system.current_validity`. If no measurement system is linked and the tool is not exempt (per §4.5), it defaults to 0.55.

**Quarantine:** When a GageStudy completes with %GRR > 30%, the system auto-sets `measurement_system.status = "quarantined"` and surfaces a notification to the owner. Quarantined instruments still produce evidence (at 0.10 validity), but the status is visible throughout the platform.

### **12.3 Tool → Investigation API Contract**

**File:** `agents_api/investigation_bridge.py`

This is the universal integration point between tool views and the investigation engine. Every tool view that supports investigation linkage calls this module.

```python
# agents_api/investigation_bridge.py

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from django.contrib.contenttypes.models import ContentType

from core.models.investigation import (
    Investigation, InvestigationToolLink,
)
from agents_api.evidence_weights import (
    compute_evidence_weight, TOOL_FUNCTIONS, TOOL_SOURCE_RANKS,
)
from agents_api.synara.synara import Synara

logger = logging.getLogger("svend.investigation")


@dataclass
class HypothesisSpec:
    """What an information tool wants to add to the graph."""
    description: str
    behavior_class: str = ""
    domain_conditions: dict = field(default_factory=dict)
    prior: float = 0.5
    # If set, creates a causal link FROM this hypothesis TO target
    causes: str | None = None  # hypothesis_id this causes


@dataclass
class InferenceSpec:
    """What an inference tool wants to add as evidence."""
    event_description: str
    context: dict = field(default_factory=dict)
    supports: list[str] = field(default_factory=list)
    weakens: list[str] = field(default_factory=list)
    raw_output: dict = field(default_factory=dict)
    sample_size: int | None = None
    measurement_system_id: str | None = None
    study_quality_factors: dict | None = None


@dataclass
class IntentSpec:
    """What an intent tool wants to annotate on the graph."""
    target_hypothesis_ids: list[str] = field(default_factory=list)
    design_metadata: dict = field(default_factory=dict)


def get_investigation(investigation_id: str, user) -> Investigation:
    """Load investigation, verify membership."""
    inv = Investigation.objects.get(id=investigation_id)
    if inv.owner == user:
        return inv
    if inv.members.filter(id=user.id).exists():
        return inv
    raise PermissionError("User is not a member of this investigation")


def load_synara(investigation: Investigation) -> Synara:
    """Load Synara engine from investigation state."""
    synara = Synara()
    if investigation.synara_state:
        synara.from_dict(investigation.synara_state)
    return synara


def save_synara(investigation: Investigation, synara: Synara):
    """Persist Synara state back to investigation."""
    investigation.synara_state = synara.to_dict()
    investigation.save(update_fields=["synara_state", "updated_at"])


def connect_tool(
    investigation_id: str,
    tool_output,  # Any tool model instance (RCASession, IshikawaDiagram, etc.)
    tool_type: str,  # "spc", "rca", "ishikawa", etc.
    user,
    spec: HypothesisSpec | InferenceSpec | IntentSpec | list[HypothesisSpec],
) -> dict:
    """
    Universal integration point: tool output → investigation graph.
    Called by tool views when investigation linkage exists.

    Returns dict with:
      - "linked": bool
      - "graph_updated": bool
      - "posteriors": dict (if inference)
      - "expansion_signal": dict | None
      - "hypotheses_added": int (if information)
    """
    investigation = get_investigation(investigation_id, user)
    result = {"linked": True, "graph_updated": False}

    # Auto-transition open → active on first tool connection
    if investigation.status == Investigation.Status.OPEN:
        investigation.transition_to(Investigation.Status.ACTIVE, user)

    # Create tool link (idempotent via unique_together)
    ct = ContentType.objects.get_for_model(tool_output)
    InvestigationToolLink.objects.get_or_create(
        investigation=investigation,
        content_type=ct,
        object_id=tool_output.id,
        defaults={
            "tool_type": tool_type,
            "tool_function": TOOL_FUNCTIONS.get(tool_type, "information"),
            "linked_by": user,
        }
    )

    # Load Synara
    synara = load_synara(investigation)
    tool_function = TOOL_FUNCTIONS.get(tool_type)

    if tool_function == "information":
        # Build graph: create hypotheses and/or causal links
        specs = spec if isinstance(spec, list) else [spec]
        added = 0
        for hs in specs:
            h_id = synara.create_hypothesis(
                description=hs.description,
                behavior_class=hs.behavior_class,
                domain_conditions=hs.domain_conditions,
                prior=hs.prior,
            )
            if hs.causes:
                synara.create_link(
                    from_id=h_id,
                    to_id=hs.causes,
                    strength=0.7,
                    mechanism=f"Causal link from {tool_type}",
                )
            added += 1
        result["graph_updated"] = True
        result["hypotheses_added"] = added

    elif tool_function == "inference":
        assert isinstance(spec, InferenceSpec)
        # Compute evidence weight
        weight = compute_evidence_weight(
            source_tool=tool_type,
            sample_size=spec.sample_size,
            measurement_system_id=spec.measurement_system_id,
            study_quality_factors=spec.study_quality_factors,
        )
        # Create evidence with computed weight
        update_result = synara.create_evidence(
            event=spec.event_description,
            context=spec.context,
            supports=spec.supports,
            weakens=spec.weakens,
            strength=weight,
            source=tool_type,
            data=spec.raw_output,
        )
        result["graph_updated"] = True
        result["posteriors"] = {
            h_id: round(p, 4) for h_id, p in update_result.posteriors.items()
        }
        result["expansion_signal"] = (
            update_result.expansion_signal.to_dict()
            if update_result.expansion_signal else None
        )
        result["evidence_weight"] = round(weight, 4)

    elif tool_function == "intent":
        assert isinstance(spec, IntentSpec)
        # Annotate hypotheses with design linkage
        for h_id in spec.target_hypothesis_ids:
            synara.annotate_hypothesis(
                h_id,
                design_id=str(tool_output.id),
                metadata=spec.design_metadata,
            )
        result["graph_updated"] = True

    elif tool_function == "report":
        # Reports read the graph, they don't modify it
        result["graph_updated"] = False

    # Persist
    save_synara(investigation, synara)

    logger.info(
        "investigation.tool_connected",
        extra={
            "investigation_id": str(investigation.id),
            "tool_type": tool_type,
            "tool_id": str(tool_output.id),
            "function": tool_function,
            "graph_updated": result["graph_updated"],
        }
    )

    return result
```

<!-- impl: agents_api/investigation_bridge.py::connect_tool -->
<!-- impl: agents_api/investigation_bridge.py::HypothesisSpec -->
<!-- impl: agents_api/investigation_bridge.py::InferenceSpec -->
<!-- impl: agents_api/investigation_bridge.py::IntentSpec -->
<!-- impl: agents_api/investigation_bridge.py::get_investigation -->

#### **12.3.1 View Integration Pattern**

Every tool view that supports investigation linkage follows this pattern:

```python
# Example: rca_views.py — after root cause is set

from agents_api.investigation_bridge import connect_tool, HypothesisSpec

def set_root_cause(request, session_id):
    # ... existing RCA logic ...
    session = RCASession.objects.get(id=session_id, owner=request.user)
    session.root_cause = data["root_cause"]
    session.save()

    # Investigation integration (opt-in)
    investigation_id = data.get("investigation_id")
    if investigation_id:
        # RCA builds a linear causal chain in the graph
        specs = []
        for step in session.chain:
            if step["accepted"]:
                specs.append(HypothesisSpec(
                    description=step["why"],
                    behavior_class="causal_factor",
                    causes=specs[-1].description if specs else None,
                ))
        connect_tool(
            investigation_id=investigation_id,
            tool_output=session,
            tool_type="rca",
            user=request.user,
            spec=specs,
        )

    return JsonResponse({"status": "ok"})
```

**Key conventions:**
- `investigation_id` is an optional field in the request JSON — never required
- If absent, tool operates standalone (CANON-001 two-mode model)
- If present, `connect_tool()` handles everything — the view doesn't need to know about Synara
- The view constructs tool-specific specs (`HypothesisSpec`, `InferenceSpec`, `IntentSpec`) from its own output
- `connect_tool()` returns a result dict that the view can include in its response if useful

#### **12.3.2 Tool-Specific Spec Construction**

Each tool constructs its specs differently. Reference implementations:

**SPC (inference):**
```python
spec = InferenceSpec(
    event_description=f"SPC {chart_type}: {signal_description}",
    context={"chart_type": chart_type, "subgroups": n_subgroups},
    supports=hypothesis_ids_if_special_cause,
    weakens=hypothesis_ids_if_in_control,
    raw_output={"cpk": cpk, "signals": signals, "control_limits": limits},
    sample_size=n_subgroups,
    measurement_system_id=request_data.get("measurement_system_id"),
)
```

**Ishikawa (information):**
```python
specs = []
for category, causes in diagram.categories.items():
    for cause in causes:
        specs.append(HypothesisSpec(
            description=cause["cause"],
            behavior_class=f"ishikawa_{category}",
            domain_conditions={"category": category},
            prior=0.5,  # Equal prior — C&E Matrix adjusts later
            causes=effect_hypothesis_id,  # All causes point to the effect
        ))
```

**DOE design (intent):**
```python
spec = IntentSpec(
    target_hypothesis_ids=[f["hypothesis_id"] for f in factors if f.get("hypothesis_id")],
    design_metadata={
        "design_type": design_type,
        "factors": factors,
        "runs": n_runs,
        "replicates": n_replicates,
    },
)
```

**DOE results (inference):**
```python
spec = InferenceSpec(
    event_description=f"DOE {design_type}: {n_significant} significant factors",
    context={"design_type": design_type, "factors": factor_names},
    supports=[f["hypothesis_id"] for f in significant_factors if f.get("hypothesis_id")],
    weakens=[f["hypothesis_id"] for f in non_significant_factors if f.get("hypothesis_id")],
    raw_output=results,
    sample_size=n_runs * n_replicates,
    measurement_system_id=request_data.get("measurement_system_id"),
    study_quality_factors={
        "randomization": 1.0 if randomized else 0.5,
        "replication": 1.0 if n_replicates >= 3 else (0.7 if n_replicates == 2 else 0.5),
        "blocking": 1.0 if n_blocks > 1 else 0.5,
    },
)
```

**FMEA (information):**
```python
specs = []
for row in fmea.rows.all():
    # RPN → prior: normalize RPN [1-1000] to [0.1, 0.9]
    prior = 0.1 + (row.rpn / 1000) * 0.8
    specs.append(HypothesisSpec(
        description=f"Failure mode: {row.failure_mode} → {row.effect}",
        behavior_class="failure_mode",
        domain_conditions={"cause": row.cause, "severity": row.severity},
        prior=prior,
    ))
```

**C&E Matrix (information — prior adjustment only):**
```python
# C&E Matrix does not create hypotheses — it adjusts priors on existing ones
synara = load_synara(investigation)
total = sum(inp["total"] for inp in matrix.inputs)
for inp in matrix.inputs:
    if inp.get("hypothesis_id") and total > 0:
        normalized = inp["total"] / total  # [0, 1]
        prior = 0.1 + normalized * 0.8  # Map to [0.1, 0.9]
        synara.graph.hypotheses[inp["hypothesis_id"]].prior = prior
        synara.graph.hypotheses[inp["hypothesis_id"]].posterior = prior
save_synara(investigation, synara)
```

Note: C&E Matrix bypasses `connect_tool()` for the prior adjustment step because it modifies existing hypotheses rather than creating new ones. It still creates an `InvestigationToolLink` for tracking.

---

## **13. INVESTIGATION API**

### **13.1 URL Routes**

**File:** `agents_api/urls.py` (additions)

```python
# Investigation endpoints — CANON-002 §13
path("api/investigations/", investigation_views.list_create, name="investigation_list_create"),
path("api/investigations/<uuid:investigation_id>/", investigation_views.detail, name="investigation_detail"),
path("api/investigations/<uuid:investigation_id>/transition/", investigation_views.transition, name="investigation_transition"),
path("api/investigations/<uuid:investigation_id>/reopen/", investigation_views.reopen, name="investigation_reopen"),
path("api/investigations/<uuid:investigation_id>/export/", investigation_views.export, name="investigation_export"),
path("api/investigations/<uuid:investigation_id>/members/", investigation_views.members, name="investigation_members"),
path("api/investigations/<uuid:investigation_id>/graph/", investigation_views.graph, name="investigation_graph"),
path("api/investigations/<uuid:investigation_id>/tools/", investigation_views.tools, name="investigation_tools"),
```

### **13.2 View Implementations**

**File:** `agents_api/investigation_views.py`

```python
# agents_api/investigation_views.py

import json
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from accounts.permissions import require_auth, gated
from core.models.investigation import (
    Investigation, InvestigationMembership, InvestigationToolLink,
)
from agents_api.investigation_bridge import (
    get_investigation, load_synara, export_investigation,
)

logger = logging.getLogger("svend.investigation")


@csrf_exempt
@require_auth
@gated("investigations")
def list_create(request):
    """
    GET  — list user's investigations (owned + member of)
    POST — create new investigation
    """
    if request.method == "GET":
        owned = Investigation.objects.filter(owner=request.user)
        member_of = Investigation.objects.filter(members=request.user)
        investigations = (owned | member_of).distinct().order_by("-updated_at")

        # Optional filters
        status = request.GET.get("status")
        if status:
            investigations = investigations.filter(status=status)

        return JsonResponse({
            "investigations": [
                {
                    "id": str(inv.id),
                    "title": inv.title,
                    "status": inv.status,
                    "version": inv.version,
                    "owner_id": str(inv.owner_id),
                    "created_at": inv.created_at.isoformat(),
                    "updated_at": inv.updated_at.isoformat(),
                    "tool_count": inv.tool_links.count(),
                }
                for inv in investigations[:50]
            ]
        })

    elif request.method == "POST":
        data = json.loads(request.body)
        title = data.get("title", "").strip()
        if len(title) < 5:
            return JsonResponse(
                {"error": "Title must be at least 5 characters"}, status=400
            )

        inv = Investigation.objects.create(
            title=title,
            description=data.get("description", ""),
            owner=request.user,
            tenant=getattr(request.user, "active_tenant", None),
        )

        # Owner is automatically a member
        InvestigationMembership.objects.create(
            investigation=inv,
            user=request.user,
            role=Investigation.MemberRole.OWNER,
        )

        return JsonResponse({
            "id": str(inv.id),
            "title": inv.title,
            "status": inv.status,
        }, status=201)

    return JsonResponse({"error": "Method not allowed"}, status=405)


@csrf_exempt
@require_auth
@gated("investigations")
def detail(request, investigation_id):
    """
    GET    — full investigation detail with graph summary
    PATCH  — update title/description
    DELETE — delete (only if open, owner only)
    """
    try:
        inv = get_investigation(str(investigation_id), request.user)
    except (Investigation.DoesNotExist, PermissionError) as e:
        return JsonResponse({"error": str(e)}, status=404)

    if request.method == "GET":
        synara = load_synara(inv)
        hypotheses = synara.graph.hypotheses
        evidence = synara.graph.evidence

        return JsonResponse({
            "id": str(inv.id),
            "title": inv.title,
            "description": inv.description,
            "status": inv.status,
            "version": inv.version,
            "owner_id": str(inv.owner_id),
            "created_at": inv.created_at.isoformat(),
            "updated_at": inv.updated_at.isoformat(),
            "concluded_at": (
                inv.concluded_at.isoformat() if inv.concluded_at else None
            ),
            "exported_at": (
                inv.exported_at.isoformat() if inv.exported_at else None
            ),
            "graph_summary": {
                "hypothesis_count": len(hypotheses),
                "evidence_count": len(evidence),
                "top_hypothesis": _top_hypothesis_summary(hypotheses),
            },
            "parent_version_id": (
                str(inv.parent_version_id) if inv.parent_version_id else None
            ),
        })

    elif request.method == "PATCH":
        if inv.owner != request.user:
            return JsonResponse(
                {"error": "Only owner can edit"}, status=403
            )
        data = json.loads(request.body)
        if "title" in data:
            inv.title = data["title"]
        if "description" in data:
            inv.description = data["description"]
        inv.save()
        return JsonResponse({"status": "updated"})

    elif request.method == "DELETE":
        if inv.owner != request.user:
            return JsonResponse(
                {"error": "Only owner can delete"}, status=403
            )
        if inv.status != Investigation.Status.OPEN:
            return JsonResponse(
                {"error": "Can only delete open investigations"}, status=400
            )
        inv.delete()
        return JsonResponse({"status": "deleted"})

    return JsonResponse({"error": "Method not allowed"}, status=405)


@csrf_exempt
@require_auth
@gated("investigations")
def transition(request, investigation_id):
    """
    POST — transition investigation state.
    Body: {"target_status": "active"|"concluded"|"exported"}
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        inv = get_investigation(str(investigation_id), request.user)
    except (Investigation.DoesNotExist, PermissionError) as e:
        return JsonResponse({"error": str(e)}, status=404)

    # Only owner can transition
    if inv.owner != request.user:
        return JsonResponse(
            {"error": "Only owner can transition state"}, status=403
        )

    data = json.loads(request.body)
    target = data.get("target_status")

    try:
        inv.transition_to(target, request.user)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({
        "status": inv.status,
        "transitioned_at": inv.updated_at.isoformat(),
    })


@csrf_exempt
@require_auth
@gated("investigations")
def reopen(request, investigation_id):
    """
    POST — reopen a concluded/exported investigation as a new version.
    Returns the new investigation ID.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        inv = get_investigation(str(investigation_id), request.user)
    except (Investigation.DoesNotExist, PermissionError) as e:
        return JsonResponse({"error": str(e)}, status=404)

    try:
        new_inv = inv.reopen(request.user)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({
        "id": str(new_inv.id),
        "version": new_inv.version,
        "parent_version_id": str(inv.id),
        "status": new_inv.status,
    }, status=201)


@csrf_exempt
@require_auth
@gated("investigations")
def export(request, investigation_id):
    """
    POST — export investigation to a Layer 3 project.
    Body: {"project_id": "uuid"}
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    data = json.loads(request.body)
    project_id = data.get("project_id")
    if not project_id:
        return JsonResponse(
            {"error": "project_id required"}, status=400
        )

    try:
        package = export_investigation(
            investigation_id=str(investigation_id),
            target_project_id=project_id,
            user=request.user,
        )
    except (ValueError, PermissionError) as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({
        "status": "exported",
        "package": package,
    })


@csrf_exempt
@require_auth
@gated("investigations")
def members(request, investigation_id):
    """
    GET  — list members
    POST — add member (owner only)
           Body: {"user_id": "uuid", "role": "contributor"|"viewer"}
    DELETE — remove member (owner only)
             Body: {"user_id": "uuid"}
    """
    try:
        inv = get_investigation(str(investigation_id), request.user)
    except (Investigation.DoesNotExist, PermissionError) as e:
        return JsonResponse({"error": str(e)}, status=404)

    if request.method == "GET":
        memberships = InvestigationMembership.objects.filter(
            investigation=inv
        ).select_related("user")
        return JsonResponse({
            "members": [
                {
                    "user_id": str(m.user_id),
                    "username": m.user.username,
                    "role": m.role,
                    "joined_at": m.joined_at.isoformat(),
                }
                for m in memberships
            ]
        })

    if inv.owner != request.user:
        return JsonResponse(
            {"error": "Only owner can manage members"}, status=403
        )

    if request.method == "POST":
        data = json.loads(request.body)
        from django.contrib.auth import get_user_model
        User = get_user_model()

        try:
            target_user = User.objects.get(id=data["user_id"])
        except User.DoesNotExist:
            return JsonResponse({"error": "User not found"}, status=404)

        role = data.get("role", "contributor")
        if role not in ("contributor", "viewer"):
            return JsonResponse(
                {"error": "Role must be contributor or viewer"}, status=400
            )

        membership, created = InvestigationMembership.objects.get_or_create(
            investigation=inv,
            user=target_user,
            defaults={"role": role},
        )
        if not created:
            membership.role = role
            membership.save()

        return JsonResponse({
            "status": "added" if created else "updated",
            "user_id": str(target_user.id),
            "role": role,
        })

    elif request.method == "DELETE":
        data = json.loads(request.body)
        deleted, _ = InvestigationMembership.objects.filter(
            investigation=inv,
            user_id=data["user_id"],
        ).exclude(
            role="owner"  # Cannot remove owner
        ).delete()

        return JsonResponse({
            "status": "removed" if deleted else "not_found",
        })

    return JsonResponse({"error": "Method not allowed"}, status=405)


@csrf_exempt
@require_auth
@gated("investigations")
def graph(request, investigation_id):
    """
    GET — return the full Synara graph state for visualization.
    Returns hypotheses, evidence, causal links, and expansion signals.
    """
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        inv = get_investigation(str(investigation_id), request.user)
    except (Investigation.DoesNotExist, PermissionError) as e:
        return JsonResponse({"error": str(e)}, status=404)

    synara = load_synara(inv)

    return JsonResponse({
        "hypotheses": [
            {
                "id": str(h.id),
                "description": h.description,
                "prior": round(h.prior, 4),
                "posterior": round(h.posterior, 4),
                "status": getattr(h, "confirmation_status", "uncertain"),
                "behavior_class": getattr(h, "behavior_class", ""),
            }
            for h in synara.graph.hypotheses.values()
        ],
        "evidence": [
            {
                "id": str(e.id),
                "event": e.event,
                "strength": round(e.strength, 4),
                "source": getattr(e, "source", "unknown"),
                "supports": [str(s) for s in getattr(e, "supports", [])],
                "weakens": [str(w) for w in getattr(e, "weakens", [])],
            }
            for e in synara.graph.evidence
        ],
        "links": [
            {
                "from_id": str(link.from_id),
                "to_id": str(link.to_id),
                "strength": round(link.strength, 4),
                "mechanism": getattr(link, "mechanism", ""),
            }
            for link in synara.graph.links
        ],
    })


@csrf_exempt
@require_auth
@gated("investigations")
def tools(request, investigation_id):
    """
    GET — list all tool outputs linked to this investigation.
    """
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        inv = get_investigation(str(investigation_id), request.user)
    except (Investigation.DoesNotExist, PermissionError) as e:
        return JsonResponse({"error": str(e)}, status=404)

    links = InvestigationToolLink.objects.filter(
        investigation=inv
    ).select_related("linked_by").order_by("-linked_at")

    return JsonResponse({
        "tools": [
            {
                "id": str(link.id),
                "tool_type": link.tool_type,
                "tool_function": link.tool_function,
                "tool_output_id": str(link.object_id),
                "linked_at": link.linked_at.isoformat(),
                "linked_by": link.linked_by.username,
            }
            for link in links
        ]
    })


def _top_hypothesis_summary(hypotheses: dict) -> dict | None:
    """Extract top hypothesis for detail view."""
    if not hypotheses:
        return None
    top = max(hypotheses.values(), key=lambda h: h.posterior)
    return {
        "id": str(top.id),
        "description": top.description,
        "posterior": round(top.posterior, 4),
        "status": getattr(top, "confirmation_status", "uncertain"),
    }
```

<!-- impl: agents_api/investigation_views.py::list_create -->
<!-- impl: agents_api/investigation_views.py::detail -->
<!-- impl: agents_api/investigation_views.py::transition -->
<!-- impl: agents_api/investigation_views.py::reopen -->
<!-- impl: agents_api/investigation_views.py::export -->
<!-- impl: agents_api/investigation_views.py::members -->
<!-- impl: agents_api/investigation_views.py::graph -->
<!-- impl: agents_api/investigation_views.py::tools -->

### **13.3 API Summary**

| Endpoint | Method | Purpose | Auth |
|----------|--------|---------|------|
| `/api/investigations/` | GET | List user's investigations | Member |
| `/api/investigations/` | POST | Create investigation | Auth |
| `/api/investigations/<id>/` | GET | Full detail + graph summary | Member |
| `/api/investigations/<id>/` | PATCH | Update title/description | Owner |
| `/api/investigations/<id>/` | DELETE | Delete (open only) | Owner |
| `/api/investigations/<id>/transition/` | POST | State transition | Owner |
| `/api/investigations/<id>/reopen/` | POST | Create new version from concluded | Member |
| `/api/investigations/<id>/export/` | POST | Export to Layer 3 project | Owner |
| `/api/investigations/<id>/members/` | GET | List members | Member |
| `/api/investigations/<id>/members/` | POST | Add/update member | Owner |
| `/api/investigations/<id>/members/` | DELETE | Remove member | Owner |
| `/api/investigations/<id>/graph/` | GET | Full Synara graph state | Member |
| `/api/investigations/<id>/tools/` | GET | List linked tool outputs | Member |

---

## **14. FILE MANIFEST**

All new files and model changes required to implement CANON-002:

### **14.1 New Files**

| File | Purpose | Section |
|------|---------|---------|
| `agents_api/evidence_weights.py` | SourceRank enum, TOOL_SOURCE_RANKS, TOOL_FUNCTIONS, compute_evidence_weight() | §2.3, §3.1, §11.1 |
| `agents_api/investigation_bridge.py` | connect_tool(), export_investigation(), supersession, confirmation thresholds | §8.4, §9.2, §10.1, §12.3 |
| `agents_api/investigation_views.py` | Investigation CRUD, state transitions, membership, graph, tools | §13.2 |
| `core/models/investigation.py` | Investigation, InvestigationMembership, InvestigationToolLink | §7.1.1, §11.1 |

### **14.2 Model Changes (Existing Files)**

| File | Change | Section |
|------|--------|---------|
| `core/models/hypothesis.py` | Add `supersedes` self-FK to Evidence model | §8.3 |
| `core/models/__init__.py` | Import Investigation, MeasurementSystem, GageStudy | — |
| `agents_api/models.py` | Add optional `measurement_system` FK to DSWResult | §12.2 |
| `agents_api/spc_views.py` | Accept `measurement_system_id`, call `connect_tool()` if `investigation_id` present | §12.3.2 |
| `agents_api/rca_views.py` | Call `connect_tool()` if `investigation_id` present | §12.3.1 |
| `agents_api/experimenter_views.py` | Call `connect_tool()` for DOE design + results | §12.3.2 |
| `agents_api/fmea_views.py` | Call `connect_tool()` if `investigation_id` present | §12.3.2 |
| `agents_api/dsw_views.py` | Accept `measurement_system_id`, call `connect_tool()` | §12.3.2 |

### **14.3 New Model (New File)**

| File | Models | Section |
|------|--------|---------|
| `core/models/measurement.py` | MeasurementSystem, GageStudy | §12.2 |

### **14.4 URL Registration**

| File | Change | Section |
|------|--------|---------|
| `svend/urls.py` or `agents_api/urls.py` | Add 8 investigation URL patterns | §13.1 |

### **14.5 Migrations Required**

| Migration | Models Affected | Dependencies |
|-----------|----------------|--------------|
| `core/migrations/XXXX_add_investigation.py` | Investigation, InvestigationMembership, InvestigationToolLink | core.Tenant, auth.User |
| `core/migrations/XXXX_add_measurement_system.py` | MeasurementSystem, GageStudy | core.Tenant, auth.User |
| `core/migrations/XXXX_evidence_supersession.py` | Evidence (add `supersedes` FK) | core.Evidence |
| `agents_api/migrations/XXXX_measurement_fk.py` | DSWResult (add `measurement_system` FK) | core.MeasurementSystem |

### **14.6 Feature Gate**

Investigation features are gated behind `"investigations"` feature flag (see `@gated("investigations")` in §13.2). This allows phased rollout:

| Tier | Access |
|------|--------|
| Free | No investigation features |
| Professional | Investigation CRUD, tool linkage, graph view |
| Team | + Membership, concurrent investigations |
| Enterprise | + Export to Layer 3, cross-investigation evidence |

---

## **15. ASSERTIONS**

<!-- assert: CANON-002-HIERARCHY — Evidence sources ranked by epistemological strength per §2.1 -->
<!-- assert: CANON-002-WEIGHT — Evidence weight = source_rank × sample_modifier × measurement_validity × study_quality -->
<!-- assert: CANON-002-GATE — Measurement system validity is a gate, not a weight — %GRR > 30% near-invalidates evidence -->
<!-- assert: CANON-002-CASCADE-MSA — Measurement validity cascades to all evidence from that instrument -->
<!-- assert: CANON-002-CONTRACTS — Each tool has a defined integration contract specifying graph interaction -->
<!-- assert: CANON-002-CHAINS — Chain contracts define output→input schemas for sequential tool operation -->
<!-- assert: CANON-002-IDEMPOTENT — Same source evidence is not counted multiple times -->
<!-- assert: CANON-002-LIFECYCLE — Investigations follow open→active→concluded→exported lifecycle -->
<!-- assert: CANON-002-SUPERSESSION — Re-run evidence creates new node with supersedes_id, original immutable -->
<!-- assert: CANON-002-EXPORT — Layer 3 receives conclusion package with top hypothesis, chain, evidence, signals -->
<!-- assert: CANON-002-CONFIRMATION — Posterior ≥ 0.85 = confirmed, ≤ 0.15 = rejected, automatic -->
<!-- assert: CANON-002-CONCURRENT — Tools belong to investigations via M2M, evidence computed per-investigation -->
<!-- assert: CANON-002-CE-PREREQ — C&E Matrix requires populated hypothesis set, surfaces workflow error if empty -->
<!-- assert: CANON-002-STRENGTH-MAP — evidence_weight maps to evidence.strength, clamped [0.05, 0.99] before assignment -->
<!-- assert: CANON-002-NO-DOUBLE-COUNT — Explicit supports/weakens set base_likelihood; strength damper scales it; one signal path -->
<!-- assert: CANON-002-MSA-MODEL — MeasurementSystem → GageStudy → tool output FK chain exists -->
<!-- assert: CANON-002-QUARANTINE — %GRR > 30% auto-quarantines instrument, visible platform-wide -->
<!-- assert: CANON-002-API-CONTRACT — Tools use connect_tool() with function-specific graph interaction -->
<!-- assert: CANON-002-SUPERSESSION-DETECT — Re-run detection matches (source_tool, source_id) and creates supersedes FK -->
<!-- assert: CANON-002-CONFIRMATION-AUTO — Posterior updates trigger automatic confirmed/rejected/uncertain transitions -->
<!-- assert: CANON-002-CONFIRMATION-REVERSAL — New contradictory evidence can reverse confirmation/rejection status -->
<!-- assert: CANON-002-EXPORT-IMPL — export_investigation() creates Evidence on target project + freezes package -->
<!-- assert: CANON-002-API-CRUD — Investigation CRUD endpoints at /api/investigations/ with member/owner auth -->
<!-- assert: CANON-002-FEATURE-GATE — Investigation features gated behind "investigations" feature flag per tier -->

---

## **CHANGELOG**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-08 | Initial release — epistemological hierarchy, evidence weighting, measurement system validity gate (no GRR default 0.55), tool integration contracts (all Layer 1-2 tools), chaining contracts (all CANON-001 §3.3 chains), investigation lifecycle (open→active→concluded→exported with versioned reopen), evidence supersession (immutable with supersedes_id FK + detection logic), Layer 3 export schema (conclusion package + full export_investigation() implementation), hypothesis confirmation thresholds (0.85/0.15 with automatic status transitions and reversal), concurrent investigation support (M2M tool membership), C&E Matrix prerequisite enforcement. Resolved: evidence_weight → evidence.strength mapping (damper formula, no double-counting), measurement system linkage model (MeasurementSystem → GageStudy → tool FK, auto-quarantine), tool → investigation API contract (connect_tool() with function dispatch). Investigation API: full CRUD + state transitions + membership management + graph view + tool listing (§13). File manifest: 4 new files, 8 modified files, 4 migrations, tiered feature gating (§14). |

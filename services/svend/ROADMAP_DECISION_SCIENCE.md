# SVEND: Computable Epistemology Platform

## Vision

**SVEND is an inquiry engine where problems, questions, and curiosity share the same shape.**

The core insight: Human reasoning is discrete inference over continuous reality. We don't observe f(x) at points—we reason over behavioral regions.

```
x ∈ S ⇒ f(x) ~ B    (not f(x) = y)
```

The causal surface is disjunctive normal form:

```
effect ⇐ ⋁ᵢ (⋀ⱼ aᵢⱼ)
```

Evidence reshapes belief mass—it never proves or disproves. When evidence contradicts all hypotheses, the model is **INCOMPLETE** (missing disjunct or conjunct), not wrong.

---

## Architecture: The Four Pages

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SVEND ARCHITECTURE                                   │
│                                                                                   │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│   │   FORGE     │     │   TRIAGE    │     │  WORKBENCH  │     │  KNOWLEDGE  │   │
│   │             │     │             │     │             │     │    GRAPH    │   │
│   │  Generate   │────▶│   Clean     │────▶│  Structure  │────▶│   Relate    │   │
│   │    Data     │     │    Data     │     │  Evidence   │     │  Evidence   │   │
│   │             │     │             │     │             │     │             │   │
│   │ • Synthetic │     │ • Quality   │     │ • Analysis  │     │ • Causal    │   │
│   │ • Simulate  │     │ • Transform │     │ • Findings  │     │   vectors   │   │
│   │ • Code      │     │ • Normalize │     │ • Inference │     │ • Weights   │   │
│   └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘   │
│                                                   │                   ▲          │
│                                                   │                   │          │
│                                                   ▼                   │          │
│                                           ┌─────────────┐             │          │
│                                           │    GUIDE    │─────────────┘          │
│                                           │             │                        │
│                                           │ LLM with    │                        │
│                                           │ full context│                        │
│                                           │ ties back   │                        │
│                                           │ to hypotheses│                       │
│                                           └─────────────┘                        │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Forge — Generate Data

**Purpose:** Create data for analysis and testing.

| Capability | Description |
|------------|-------------|
| **Synthetic Data** | Generate realistic datasets from schemas |
| **Simulations** | Run Monte Carlo, process simulations |
| **Code Generation** | Custom data generation scripts |
| **Scenario Modeling** | "What if" data generation |

### 2. Triage — Clean Data

**Purpose:** Ensure data quality before analysis.

| Capability | Description |
|------------|-------------|
| **Quality Checks** | Missing values, outliers, duplicates |
| **Transformation** | Normalization, encoding, aggregation |
| **Validation** | Schema conformance, business rules |
| **Bias Detection** | Sampling bias, measurement bias |

### 3. Workbench — Structure Evidence

**Purpose:** Transform raw data and observations into structured evidence.

The Workbench is the **universal evidence factory**. Input takes many forms; output is always structured evidence that can attach to the Knowledge Graph.

#### Workbench Modes

| Mode | Tools | Output |
|------|-------|--------|
| **Data Science Bench** | Minitab-style stats, SPC, DOE, capability, ML, regression, ANOVA | Statistical findings, model results |
| **Canvas/Whiteboard** | Affinity diagrams, interrelationship diagraphs, post-its, brainstorming | Clustered themes, causal hypotheses |
| **Code Bench** | Notebooks, custom analysis, simulations | Computed results, validated claims |
| **Visual Bench** | Charts, distributions, comparisons, dashboards | Pattern observations |

#### The Guide

The Guide is an LLM loaded with full context:
- Current hypotheses and their weights
- Knowledge Graph state
- Available evidence
- Domain context

**Key capability:** Real-time inference connection.

```
Example:
User runs EDA → mean = 45.2, std = 6.7

Guide: "This makes the hypothesized mean of 22.0 highly unlikely.
        The observed mean is 3.5 standard deviations away from
        the hypothesis. Evidence strongly weighs against H₃."
```

### 4. Knowledge Graph — Relate Evidence

**Purpose:** Represent the current worldview as a weighted causal graph.

#### Structure

- **Nodes:** Hypotheses, causes, effects, observations
- **Edges:** Causal vectors with weights
- **Weights on edges, not nodes:** Because one cause may have multiple effects, and one can study in either direction (cause→effect or effect→cause)

```
       ┌─────────────┐
       │  Cause A    │
       └──────┬──────┘
              │ w=0.7 (P(effect|cause))
              ▼
       ┌─────────────┐         ┌─────────────┐
       │   Effect    │◀────────│  Cause B    │
       └─────────────┘  w=0.3  └─────────────┘
```

#### Evidence Attachment

Evidence can be:
1. **Attached to a vector within Workbench** — temporary, part of active analysis
2. **Loaded into Knowledge Graph** — permanent, part of worldview

---

## Synara: The Belief Engine

### Two Synaras

#### 1. Synara-Engine (Bayesian Inference)

The computational core for belief updates.

```python
# Core primitives
HypothesisRegion   # Behavioral domain, not point claim
Evidence           # Observations that reshape belief
CausalLink         # Weighted edge in causal graph
ExpansionSignal    # When evidence doesn't fit any hypothesis
```

**Key operations:**
- `P(h|e) ∝ P(e|h) × P(h)` — Bayesian update
- Likelihood computation from evidence
- Belief propagation through causal DAG
- Expansion detection when model is incomplete

#### 2. Synara-Meta (Epistemic Learning)

Logs epistemic processes to learn reasoning patterns.

**What gets logged:** Not chat data—epistemic motion:
- How hypotheses were formulated
- What evidence shifted beliefs
- When expansions occurred
- Which reasoning paths led to insight
- Which led to dead ends

**Purpose:** Learn to reason better over time. A second intelligence that observes the inquiry process itself.

---

## The Bridge: Frequentist → Bayesian

### The Core Insight

SVEND's Workbench produces frequentist outputs (p-values, coefficients, R²). Synara needs Bayesian likelihoods. The power is not in pretending p-values are probabilities—it's in using them as **evidence weights** inside the causal engine.

### What a p-value really is

A p-value is:

```
p = P(data | H₀)
```

It is *not*:

```
P(H | data)
```

You cannot invert it directly. But you **can** transform it into a likelihood ratio, which *is* Bayesian fuel.

### From p-value → Likelihood Ratio

There is a known bound (Sellke–Bayarri–Berger):

```
BF₀₁ ≤ 1 / (-e × p × ln(p))
```

This gives a **minimum Bayes factor** against the null. So from a p-value you can compute:

```
BF₁₀ ≥ -e × p × ln(p)
```

This is now a **likelihood multiplier**—exactly what Synara needs.

### How SVEND Uses It

Each regression coefficient becomes a causal hypothesis:

```
cᵢ ⇒ effect
```

With prior P(cᵢ).

From regression:
1. Take p-value for coefficient
2. Convert to Bayes factor
3. Update belief:

```
P(cᵢ | evidence) ∝ BF₁₀ × P(cᵢ)
```

You're no longer trusting the model. You're **harvesting its epistemic signal**.

### When R² is Low

Low R² means:

```
Σᵢ P(cᵢ | evidence) ≪ 1
```

This is exactly the "missing causes" detector. The frequentist model becomes:
- A Bayesian evidence source
- A trigger for causal expansion
- A feeder into the Knowledge Graph

Not a truth oracle.

### What This Really Is

SVEND turns:
- Regressions
- SPC control charts
- Experiments (DOE)
- Audits and observations

...into **belief updates on causal edges**.

That's not statistics. That's **operational epistemology**.

This is mathematically sound when framed as likelihood weighting, not inversion. This is the bridge between Workbench and Synara.

---

## Formal Hypothesis Language (DSL)

Users formulate **deductive hypotheses** using structured operators:

### Operators

| Type | Operators |
|------|-----------|
| **Quantifiers** | ALWAYS, NEVER, SOMETIMES, ALL, NONE, SOME |
| **Logical** | AND, OR, XOR, NOT |
| **Comparison** | >, <, >=, <=, =, != |
| **Conditional** | IF...THEN, WHEN |

### Examples

```
# Simple claim
if [num_holidays] > 3 then [monthly_sales] < 100000

# Universal quantifier with domain restriction
ALWAYS [temperature] > 20 AND [temperature] < 30 WHEN [machine] = "Line A"

# Existential claim
SOMETIMES [defect_rate] > 0.05

# Negation
NEVER [yield] < 80 WHEN [shift] = "day"
```

### System Validation

The deterministic logic engine parses hypotheses for:
- **Falsifiability** — Can this be tested?
- **Logical fallacies** — Circular reasoning, affirming consequent
- **Completeness** — Are domain conditions specified?
- **Variable references** — Do referenced variables exist in context?

When ambiguity detected → escalate to LLM for clarification.

---

## Evidence Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            EVIDENCE SOURCES                                    │
│                                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   CSV    │  │  XLSX    │  │   ML     │  │ Simulate │  │ Research │       │
│  │  Upload  │  │  Upload  │  │  Model   │  │  Code    │  │ Findings │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │             │             │             │              │
│       └──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘              │
│              │             │             │             │                      │
│              ▼             ▼             ▼             ▼                      │
│       ┌─────────────────────────────────────────────────────┐                │
│       │                    WORKBENCH                         │                │
│       │                                                      │                │
│       │  Raw Data/Observations → Structured Evidence         │                │
│       │                                                      │                │
│       │  Guide: "This finding supports H₂, weakens H₁"       │                │
│       └──────────────────────┬──────────────────────────────┘                │
│                              │                                                │
│                              ▼                                                │
│       ┌─────────────────────────────────────────────────────┐                │
│       │              STRUCTURED EVIDENCE                     │                │
│       │                                                      │                │
│       │  {                                                   │                │
│       │    finding: "Mean = 45.2, outside hypothesis range", │                │
│       │    confidence: 0.95,                                 │                │
│       │    source: "EDA",                                    │                │
│       │    supports: [],                                     │                │
│       │    weakens: ["H₃"],                                  │                │
│       │    strength: 0.8                                     │                │
│       │  }                                                   │                │
│       └──────────────────────┬──────────────────────────────┘                │
│                              │                                                │
│                              ▼                                                │
│       ┌─────────────────────────────────────────────────────┐                │
│       │              KNOWLEDGE GRAPH                         │                │
│       │                                                      │                │
│       │  Edge weights updated via Bayes:                     │                │
│       │  P(H|E) ∝ P(E|H) × P(H)                              │                │
│       │                                                      │                │
│       │  Expansion signal if evidence fits no hypothesis     │                │
│       └─────────────────────────────────────────────────────┘                │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Workbench Capabilities (Full Scope)

The Workbench will handle ALL evidence creation needs:

### Statistical Analysis (Minitab-equivalent)

| Category | Tools |
|----------|-------|
| **SPC** | I-MR, X-bar R, X-bar S, P, NP, C, U charts |
| **Capability** | Cp, Cpk, Pp, Ppk, capability sixpack |
| **DOE** | Full factorial, fractional, Plackett-Burman, response surface, Taguchi |
| **Hypothesis Tests** | t-test, ANOVA, chi-square, F-test |
| **Regression** | Linear, multiple, logistic, polynomial |
| **Reliability** | Weibull, life data analysis |
| **Measurement** | Gage R&R, MSA |

### Data Science

| Category | Tools |
|----------|-------|
| **ML Models** | Classification, regression, clustering |
| **Diagnostics** | Confusion matrix, ROC, feature importance |
| **Time Series** | Forecasting, decomposition, ARIMA |
| **Dimensionality** | PCA, t-SNE, factor analysis |

### ML Expansion: Causal Lens Toolkit

The ML toolkit is not a model zoo—it's a **causal lens toolkit**. Each model is a different way to slice the manifold. When models disagree, we don't choose a winner—we trigger **causal expansion**.

#### Phase 1: Synara Integration (Priority)

| Model | Purpose | Synara Integration |
|-------|---------|-------------------|
| **Bayesian Regression** | Native uncertainty, posterior over coefficients | Credible intervals → edge weights directly |
| **GAM (Generalized Additive Models)** | Human-readable spline curves per feature | "How does X bend Y" - interpretable causation |
| **Isolation Forest** | Anomaly detection = "missing cause" signal | Expansion trigger when points don't fit |

#### Phase 2: Uncertainty & Structure

| Model | Purpose | Visual Output |
|-------|---------|---------------|
| **Gaussian Process Regression** | Behavior over subsets + uncertainty | Mean curve + confidence bands |
| **Partial Least Squares (PLS)** | Collinearity handling for process data | Latent score plots |

#### Phase 3: Causal Crown Jewel

| Model | Purpose |
|-------|---------|
| **Structural Equation Modeling (SEM)** | Explicit causal path modeling with edge weights. Tests mediation. This IS the Knowledge Graph in statistical form. |

#### Standard Visual Primitives

Every model generates:
- **Feature effect curves** (PDP / ICE / GAM splines)
- **Uncertainty bands** (Bayesian / GPR)
- **Residual manifolds** (colored scatter by regime)
- **Shapley flow** (for tree models → causal edges)
- **Regime clustering** (Isolation Forest overlays)

### Facilitation Tools

| Category | Tools |
|----------|-------|
| **Brainstorming** | Affinity diagrams, post-its, freeform canvas |
| **Causal Analysis** | Fishbone/Ishikawa, 5 Whys, fault tree |
| **Relationship** | Interrelationship diagraphs, C&E matrix |
| **Process** | Value stream maps, SIPOC, process flow |

### Visualization

| Category | Tools |
|----------|-------|
| **Distributions** | Histograms, box plots, probability plots |
| **Relationships** | Scatter, matrix plots, correlation heatmaps |
| **Comparison** | Bar charts, Pareto, multi-vari |
| **Trends** | Time series, run charts, trend analysis |

---

## Implementation Status

### Completed

- [x] Synara-Engine core (kernel.py, belief.py, synara.py)
- [x] LLM Interface for hypothesis/validation prompts
- [x] DSL Parser for formal hypotheses
- [x] Logic Engine for hypothesis evaluation
- [x] API endpoints for Synara integration
- [x] Basic SPC (control charts, capability)
- [x] Experimenter (DOE, power analysis)
- [x] Triage (data quality)
- [x] Forge (synthetic data)
- [x] Problem/hypothesis tracking
- [x] Evidence linking

### In Progress

- [ ] Knowledge Graph UI
- [ ] Workbench multi-bench structure
- [ ] Guide integration with real-time inference
- [ ] Synara-Meta logging
- [x] **ML Phase 1: Bayesian Regression, GAM, Isolation Forest** ✓
- [x] **ML Phase 2: Gaussian Process Regression, PLS** ✓
- [x] **ML Phase 3: Structural Equation Modeling (SEM)** ✓
- [x] **Stats Phase 1: Causal Time Series** (Granger Causality, Change Point Detection) ✓
- [x] **Stats Phase 2: Non-parametric Suite** (Mann-Whitney, Kruskal-Wallis, Chi-square) ✓
- [x] **Stats Phase 3: DOE/Factorial** (Main effects, Interaction plots, Multi-vari) ✓
- [x] **Stats Phase 4: Time Series** (ARIMA, Decomposition, ACF/PACF) ✓
- [x] **Stats Phase 5: Reliability** (Weibull, Kaplan-Meier survival) ✓
- [x] **Stats Phase 6: Advanced Stats** (Logistic regression, F-test, Equivalence/TOST, Runs test) ✓
- [x] **SPC Phase 2: Advanced Charts** (NP, C, U, CUSUM, EWMA) ✓
- [x] **MSA: Gage R&R** (Repeatability & Reproducibility) ✓

### Planned

- [x] **Whiteboard/Canvas** (Affinity, Interrelationship, Process Maps, VSM, Fishbone) ✓
- [ ] Bootstrap Confidence Intervals
- [ ] Plackett-Burman / Taguchi DOE designs
- [ ] Documentation export (8D, A3)

---

## The Dual Intelligence Model

```
┌─────────────────────────────────────────────────────────────────┐
│                      DUAL INTELLIGENCE                           │
│                                                                  │
│  ┌────────────────────────┐    ┌────────────────────────┐       │
│  │   LLM LAYER            │    │   LOGIC ENGINE          │       │
│  │                        │    │                         │       │
│  │ • Research prior art   │    │ • Parse formal claims   │       │
│  │ • Generate hypotheses  │    │ • Detect fallacies      │       │
│  │ • Write documentation  │    │ • Evaluate against data │       │
│  │ • Interpret patterns   │    │ • Bayesian updates      │       │
│  │ • Validate causal      │    │ • Propagate belief      │       │
│  │   graph for errors     │    │ • Signal expansion      │       │
│  │                        │    │                         │       │
│  │ Handles: ambiguity,    │    │ Handles: deterministic  │       │
│  │ creativity, context    │    │ logic, computation      │       │
│  └───────────┬────────────┘    └───────────┬─────────────┘       │
│              │                             │                      │
│              └──────────┬──────────────────┘                      │
│                         │                                         │
│                         ▼                                         │
│              ┌─────────────────────┐                             │
│              │   ESCALATION        │                             │
│              │                     │                             │
│              │ Logic Engine →  LLM │                             │
│              │ when ambiguity      │                             │
│              │ detected            │                             │
│              │                     │                             │
│              │ LLM → Logic Engine  │                             │
│              │ for formal          │                             │
│              │ validation          │                             │
│              └─────────────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Principles

### 1. Evidence Never Proves

Evidence reshapes belief mass. It never proves or disproves. High probability ≠ certainty.

### 2. Incomplete, Not Wrong

When evidence contradicts all hypotheses, the causal surface is incomplete:
- Missing disjunct (new cause not considered)
- Missing conjunct (existing hypothesis needs additional premise)

The Expansion Signal tells us to expand the model, not that it's wrong.

### 3. Weights on Edges

Causal relationships have strength, not just existence. The same cause may strongly produce one effect and weakly produce another.

### 4. Discrete Minds, Continuous Reality

We reason in categories over continuous phenomena. Hypotheses are behavioral regions, not point claims.

### 5. Deductive Rigor

Formal hypotheses enable formal evaluation. The DSL forces clarity; the logic engine enforces validity.

---

## Success Criteria

### For Users

- Problems correctly framed before analysis
- Hypotheses tested, not just confirmed
- Evidence systematically connected to beliefs
- Clear understanding of uncertainty
- Documented reasoning trail

### For the System

- Synara-Meta learns effective reasoning patterns
- Expansion signals identify model gaps
- Guide provides actionable inference connections
- Workbench handles all evidence types seamlessly

---

## Summary

SVEND is not a chat interface or an answer machine. It's a **computable epistemology**—a system for structured inquiry where:

1. **Forge** generates data
2. **Triage** cleans data
3. **Workbench** creates structured evidence
4. **Knowledge Graph** relates evidence via weighted causal vectors
5. **Guide** connects findings to hypotheses in real-time
6. **Synara-Engine** computes belief updates
7. **Synara-Meta** learns to reason better

The output is not answers—it's understanding. And understanding, properly structured, makes the right direction obvious.

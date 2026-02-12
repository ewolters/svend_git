# SVEND Architecture

## Document History

- **2026-01-29**: Complete product architecture revision. Unified workbench model. Previous Synara technical details moved to SYNARA_WHITEPAPER.md.
- **2026-01-21**: Synara Bayesian network architecture (superseded for product, retained for engine).

---

## Executive Summary

**SVEND** is a professional-grade decision science workbench. It competes with Minitab on statistical rigor while adding structured reasoning and AI assistance at 96% lower cost.

**One-liner:** Minitab + reasoning + AI at $5/month.

**Core insight:** Engineers don't just need statistics. They need to arrive at structured responses to inquiries - whether diagnosing a problem, planning an experiment, or validating a process.

```
Old Model (Minitab, JMP):
  Data → Statistical Tool → Numbers → User figures out what it means

SVEND Model:
  Inquiry → Artifacts (data, analysis, hypotheses, evidence) → Structured Response
  AI assists thinking throughout, challenges assumptions, drafts conclusions
```

---

## Core Ontology

### The Fundamental Unit: Inquiry → Structured Response

Users don't come to SVEND to "run SPC." They come because they have:

| Trigger | Example | Structured Response |
|---------|---------|---------------------|
| Problem | "Yield dropped 3%" | Causal explanation + action plan |
| Question | "What drives churn?" | Factors + model + recommendations |
| Curiosity | "Is there a pattern here?" | Findings + interpretation |
| Task | "Design an experiment" | DOE plan + methodology |

**What unifies them:** The user arrives at a **structured response**, not just a number.

Running SPC and getting "Cpk = 1.34" is not a structured response.
Running SPC and concluding "Process is capable, variation is within spec, no action needed" - that's a structured response.

### Artifacts

Everything in SVEND produces an **artifact**:

| Category | Artifact Types |
|----------|----------------|
| **Data** | Dataset, Cleaned Dataset, Generated Dataset |
| **Analysis** | SPC Chart, Capability Study, Forecast, Correlation, EDA, ANOVA |
| **Experiment** | DOE Design, DOE Results, Confirmation Run |
| **ML** | Trained Model, Model Metrics, Feature Importance, Predictions |
| **Thinking** | Note, Hypothesis, Evidence, Conclusion |
| **Documents** | Report Draft, Summary, 8D, A3, Control Plan |
| **Code** | Script, Visualization |
| **Research** | Findings, Citations, Base Rates |

The workspace is a collection of artifacts tied to an inquiry. Structure emerges from what the user creates, not from enforced modes.

### No Modes, No Phases

Previous designs attempted to impose structure:
- Mode-based (Diagnostic, Exploratory, etc.) - too many, not clean
- Phase-based (Observe → Hypothesize → Test) - too rigid
- Workflow-based (predefined sequences) - too constraining

**SVEND approach:** Tools are always available. Artifacts accumulate on a canvas. Structure emerges from the work itself. Templates (DMAIC, Kaizen) provide optional scaffolding.

---

## Product Architecture

### Three Functional Areas

```
┌─────────────────────────────────────────────────────────┐
│  SVEND                                                  │
├─────────────────────────────────────────────────────────┤
│   Workbench        Forge        Triage        Settings  │
└─────────────────────────────────────────────────────────┘
```

| Area | Purpose |
|------|---------|
| **Workbench** | Where all inquiry/analysis work happens. Unified home. |
| **Forge** | Data generation service. Workbench calls it, receives dataset artifacts. |
| **Triage** | Data cleaning service. Workbench sends raw data, receives clean data + quality report. |
| **Settings** | User preferences, billing, account. |

### What Workbench Absorbs

Previously separate pages now unified in Workbench:

| Previous | Status |
|----------|--------|
| DSW | Merged into Workbench |
| Problems | Merged into Workbench (inquiries) |
| SPC | Merged into Workbench (tool) |
| Experimenter | Merged into Workbench (tool) |
| Forecast | Merged into Workbench (tool) |
| Models | Merged into Workbench (artifacts) |
| Workflows | Eliminated (emergent from artifact sequences) |

---

## Workbench UI Architecture

### Layout

```
┌────────────────────────────────────────────────────────────────────────────┐
│ ≡ SVEND   [Inquiry Title ▾]                         [Save] [Export]   ⚙️  │
├────────────────────────────────────────────────────────────────────────────┤
│ ┌──────────┬───────────┬────────────┬─────────┬──────────┬──────────────┐ │
│ │Statistics│ DOE       │ Control    │ ML      │ Research │ Report       │ │
│ │          │           │ Charts     │         │          │              │ │
│ ├──────────┴───────────┴────────────┴─────────┴──────────┴──────────────┤ │
│ │ [Tool buttons for active ribbon tab]                                   │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
├───────────────────────┬─────────────────────────────┬──────────────────────┤
│      DATA PANE        │         CANVAS              │    VISUALS PANE      │
│                       │                             │                      │
│ ┌───────────────────┐ │  ┌──────────┐ ┌──────────┐ │ ┌──────────────────┐ │
│ │ active_dataset    │ │  │ Artifact │ │ Artifact │ │ │ [Active Plot]    │ │
│ │ ─────────────────│ │  │          │→│          │ │ │                  │ │
│ │ Col1  Col2  Col3 │ │  └──────────┘ └──────────┘ │ │                  │ │
│ │ ...              │ │                             │ │                  │ │
│ │                  │ │       ┌──────────┐          │ ├──────────────────┤ │
│ ├───────────────────┤ │       │ Artifact │          │ │ Plot History     │ │
│ │ Variables:       │ │       │          │          │ │ • Chart 1        │ │
│ │ • Yield (cont)   │ │       └──────────┘          │ │ • Chart 2        │ │
│ │ • Temp (cont)    │ │                             │ │ • Chart 3        │ │
│ │ • Supplier (cat) │ │                             │ │                  │ │
│ └───────────────────┘ │                             │ └──────────────────┘ │
├───────────────────────┴─────────────────────────────┴──────────────────────┤
│ GUIDE: [AI observations, challenges, suggestions]                    [Ask] │
└────────────────────────────────────────────────────────────────────────────┘
```

### Panes

| Pane | Purpose |
|------|---------|
| **Ribbon** | Tool groups (Minitab-style). Click tab to see tool buttons. |
| **Data** | Worksheet view + variable list. Import, view, manage datasets. Shows types. |
| **Canvas** | Free-form artifact workspace. Drag, connect, group, annotate. |
| **Visuals** | Active plot + history. Click history to swap. Export options. |
| **Guide** | AI observations. Not intrusive but always watching. Collapsible. |

### Ribbon Tool Groups

| Tab | Tools |
|-----|-------|
| **Statistics** | Basic Stats, Descriptive, ANOVA (1-way, 2-way, GLM), Regression (Simple, Multiple, Stepwise, Best Subsets), Hypothesis Tests (t-tests, proportion, equivalence), Correlation, Multivariate, Time Series, Normality Tests |
| **DOE** | Factorial (Full, Fractional), Screening (Plackett-Burman), Response Surface (CCD, Box-Behnken), Optimal Designs (D-optimal, I-optimal), Taguchi, Mixture, Sample Size/Power Analysis |
| **Control Charts** | Variables (I-MR, X̄-R, X̄-S), Attributes (p, np, c, u, Laney p', Laney u'), Capability (Cp, Cpk, Pp, Ppk, Cpm), Box-Cox Transform, Western Electric Rules, Nelson Rules |
| **ML** | Classification, Regression, Clustering, Feature Selection, Model Comparison, Cross-Validation, Hyperparameter Tuning |
| **Research** | Literature Search, Base Rates, Domain Knowledge, Prior Art, Data Sources |
| **Report** | Summary, 8D Report, A3, Control Plan, PPAP, Export (PDF, Excel, Word) |

### MSA Tools (under Statistics or separate)

- Gage R&R (Crossed, Nested)
- Gage Linearity & Bias
- Attribute Agreement Analysis

---

## Templates

Users select a template when creating a new workbench. Templates provide structure without forcing it.

### Available Templates

| Template | Structure | Use Case |
|----------|-----------|----------|
| **Blank** | No structure, free-form | Ad-hoc analysis, exploration |
| **DMAIC** | 5 phases with required artifacts and gates | Six Sigma projects |
| **Kaizen** | Event-based with current/future state | Rapid improvement events |
| **8D** | 8 disciplines, report-oriented | Problem solving, customer complaints |
| **A3** | Single-page constraint, visual thinking | Toyota-style problem solving |
| **Quick Analysis** | No save, ephemeral | One-off calculations |

### DMAIC Template

```
Phases: Define → Measure → Analyze → Improve → Control

Each phase has:
- Required artifacts (e.g., Define needs: Problem Statement, Charter, SIPOC, CTQ, Scope)
- Suggested tools
- Gate criteria (can't advance until artifacts complete)
```

| Phase | Required Artifacts | Key Tools |
|-------|-------------------|-----------|
| **Define** | Problem Statement, Project Charter, SIPOC, CTQ, Scope | Templates, Guide |
| **Measure** | Data Collection Plan, MSA/Gage R&R, Baseline Metrics | MSA, SPC, Basic Stats |
| **Analyze** | Fishbone, 5 Whys, Hypothesis Tests, Root Cause | Stats, Regression, Screening DOE |
| **Improve** | Solutions, DOE, Pilot Results, Before/After | DOE, Confirmation runs |
| **Control** | Control Plan, SPC Charts, SOPs, Handoff | Control Charts, Documentation |

### Kaizen Template

```
Structure: Current State → Waste ID → Future State → Actions → Results
Timeline: Day 1-5 event tracking
```

| Section | Contents |
|---------|----------|
| **Current State** | Process map, cycle time, baseline metrics |
| **Waste Identification** | 8 wastes checklist with time/cost impact |
| **Future State** | Target process map, target metrics |
| **Action Items** | Who, what, when - trackable |
| **Results** | Before/after measurements, % improvement |

---

## Tools → Methods → Knowledge Architecture

### Core Insight

Tools generate knowledge. Methods orchestrate and structure it. This separation enables:

1. **Tools are reusable** across any method
2. **Methods are composable** - users pick the structure that fits
3. **Knowledge persists** independently of how it was created

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              PROJECT                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                         TOOLS (Generate Knowledge)                │   │
│  ├──────────────────────────────┬───────────────────────────────────┤   │
│  │  DSW                         │  Whiteboard                        │   │
│  │  (Quantitative)              │  (Qualitative)                     │   │
│  │  ───────────────             │  ─────────────                     │   │
│  │  • Statistical analysis      │  • Brainstorming                   │   │
│  │  • SPC / Capability          │  • Fishbone diagrams               │   │
│  │  • DOE                       │  • Affinity mapping                │   │
│  │  • Hypothesis testing        │  • Process mapping                 │   │
│  │  • Regression                │  • If-then relationships           │   │
│  │  • ML models                 │  • Causal chains                   │   │
│  └──────────────────────────────┴───────────────────────────────────┘   │
│                                  │                                       │
│                                  ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    KNOWLEDGE (Persistent Artifacts)               │   │
│  ├──────────────────────────────────────────────────────────────────┤   │
│  │  Hypotheses    Evidence      Conclusions    Summaries             │   │
│  │  (with P())    (with LR)     (structured)   (LLM-generated)       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                  │                                       │
│                                  ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    METHODS (Orchestrate & Structure)              │   │
│  ├─────────────┬─────────────┬─────────────┬───────────┬────────────┤   │
│  │    A3       │   DMAIC     │   5-Why     │    8D     │   Kaizen   │   │
│  │  (1-page)   │  (phases)   │ (drill down)│ (report)  │  (event)   │   │
│  └─────────────┴─────────────┴─────────────┴───────────┴────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tools

Tools are knowledge generators. They produce structured output that becomes persistent knowledge.

| Tool | Type | Outputs |
|------|------|---------|
| **DSW** | Quantitative | Statistical results, capability metrics, DOE effects, model predictions, p-values, effect sizes |
| **Whiteboard** | Qualitative | Diagrams, relationships, groupings, causal chains, if-then statements, process flows |

Tools don't care what method (if any) will consume their output. A capability study is the same whether it's part of DMAIC Measure phase or a standalone analysis.

### Knowledge Artifacts

Knowledge persists at the project level. Artifacts are typed and structured:

| Artifact | Source | Contains |
|----------|--------|----------|
| **Hypothesis** | User, Whiteboard, DSW | Statement, prior probability, current probability, evidence links |
| **Evidence** | DSW, Whiteboard, Manual | Source type, result, confidence, likelihood ratio per hypothesis |
| **Conclusion** | LLM Summary, User | Structured finding with supporting evidence |
| **Summary** | LLM (Qwen) | Natural language synthesis of tool session |

### Methods

Methods provide scaffolding and structure. They don't generate knowledge - they organize it.

| Method | Structure | Pulls From |
|--------|-----------|------------|
| **A3** | Single-page sections (Background, Current State, Target, Analysis, Countermeasures, Plan) | DSW summaries → Analysis, Whiteboard fishbone → Root cause |
| **DMAIC** | 5 phases with gates | Phase-specific tool outputs |
| **5-Why** | Drill-down chain | Whiteboard relationships, DSW root cause analysis |
| **8D** | 8 disciplines | All knowledge artifacts |
| **Kaizen** | Current/Future state | Whiteboard process maps, DSW metrics |

### Import/Export Flows

Knowledge flows between tools and methods through structured import/export:

```
WHITEBOARD EXPORTS:
├── If-then relationship  →  Hypothesis (with statement + implied prior)
├── Fishbone branch       →  Hypothesis set (multiple causes)
├── Affinity group        →  Summary (LLM synthesizes cluster)
├── Process map           →  A3 Current State section
└── Session               →  Summary (LLM synthesizes key findings)

DSW EXPORTS:
├── Statistical test      →  Evidence (p-value, effect size, direction)
├── Capability study      →  Evidence + Summary
├── DOE results           →  Evidence + Hypothesis updates (main effects)
├── Regression model      →  Evidence + Conclusions (significant factors)
└── Session               →  Summary (LLM synthesizes analysis)

METHOD IMPORTS:
├── A3.Analysis           ←  DSW summary, Whiteboard fishbone summary
├── A3.Countermeasures    ←  Whiteboard brainstorm summary
├── DMAIC.Measure         ←  DSW capability, MSA results
├── DMAIC.Analyze         ←  DSW hypothesis tests, Whiteboard fishbone
├── 5-Why.Chain           ←  Whiteboard causal chain
└── 8D.Root_Cause         ←  All evidence supporting hypothesis
```

### LLM Summarization Layer

Import/export requires an LLM to translate raw tool output into method-consumable summaries:

```
Tool Session (raw)
       │
       ▼
┌──────────────────┐
│  Qwen Summary    │  ← "Summarize this DSW session for A3 Analysis section"
│  (context-aware) │  ← Knows target format, length constraints
└──────────────────┘
       │
       ▼
Structured Summary
(ready for method import)
```

The summarizer understands:
- **Source context**: What tool, what type of analysis
- **Target context**: Which method, which section, what format expected
- **Constraints**: Length limits (A3 is one page), required elements

### Implementation Phases

**Phase 1: Knowledge Persistence (Current)**
- Hypotheses, Evidence, EvidenceLink models exist
- Basic probability tracking works
- No import/export yet

**Phase 2: Tool → Knowledge Export**
- DSW results create Evidence automatically
- Whiteboard elements exportable to Hypotheses
- LLM summarization for sessions

**Phase 3: Method Import**
- A3, DMAIC templates can pull knowledge artifacts
- Smart suggestions ("You have 3 hypotheses that could go in Analysis")
- One-click import with LLM formatting

**Phase 4: Bi-directional Flow**
- Methods can request specific tool analysis
- "A3 Analysis needs capability data" → opens DSW with context
- Cross-tool references preserved

---

## AI Philosophy

### What AI Does

- **Infers assumptions**: Detects what the user is assuming without stating
- **Detects patterns**: Finds correlations, anomalies in data automatically
- **Challenges reasoning**: "You have supporting evidence but no disconfirming tests"
- **Drafts conclusions**: When enough evidence exists, proposes structured response
- **Pre-writes reports**: Generates 8D, A3, summaries from artifacts

### What AI Does NOT Do

- Suggest root causes (user must reason to those)
- Replace structured thinking with "vibes"
- Give answers without showing work
- Skip rigor for convenience

### Agent Roles

| Agent | Function |
|-------|----------|
| **Decision Guide** | Observes workspace, challenges assumptions, suggests what's missing |
| **Researcher** | Finds literature, base rates, prior art, domain knowledge |
| **Analyst** | Runs statistical analysis, interprets results |
| **Writer** | Drafts reports, summaries, documentation |
| **Coder** | Generates scripts, custom visualizations, automation |

Agents create artifacts. Guide observes and advises.

---

## Session Persistence

### Save Format

Workbench saves as JSON. This JSON becomes the system prompt for agents on load.

```json
{
  "inquiry": "Why did yield drop 3% on Line 3?",
  "template": "DMAIC",
  "phase": "Analyze",
  "artifacts": [
    {
      "id": "a1",
      "type": "hypothesis",
      "content": "Supplier change caused yield drop",
      "probability": 0.6,
      "created": "2026-01-28T10:00:00Z"
    },
    {
      "id": "a2",
      "type": "spc_chart",
      "chart_type": "I-MR",
      "data_ref": "yield_data.csv",
      "result": {"in_control": true, "cpk": 1.34},
      "created": "2026-01-28T10:30:00Z"
    },
    {
      "id": "a3",
      "type": "model",
      "model_type": "random_forest",
      "path": "models/yield_predictor.pt",
      "metrics": {"r2": 0.87, "rmse": 0.4},
      "created": "2026-01-28T14:00:00Z"
    }
  ],
  "connections": [
    {"from": "a1", "to": "a2", "label": "tested by"}
  ],
  "datasets": [
    {"name": "yield_data.csv", "rows": 500, "cols": 12}
  ],
  "created": "2026-01-28T09:00:00Z",
  "updated": "2026-01-29T16:00:00Z"
}
```

### Load Behavior

1. JSON loaded from file
2. JSON injected as system prompt context for agents
3. Agents have full continuity - no "catching up"
4. Models loaded from referenced paths
5. User resumes exactly where they left off

---

## Technical Stack

### Models

| Purpose | Model |
|---------|-------|
| **Reasoning** | Qwen-7B Instruct |
| **Code Generation** | Qwen-Coder |
| **Parsing/Formatting** | Qwen-3B or smaller |

### Backend

| Component | Technology |
|-----------|------------|
| **Framework** | Django 5.0 + Django REST Framework |
| **Database** | PostgreSQL |
| **Server** | Gunicorn |
| **Task Queue** | Tempora (custom) |

### Execution

- **Local Python**: Scripts execute on user's machine
- **Models**: Train and run locally (GPU if available, CPU fallback)
- **Sandboxing**: Tool execution isolated, 30s timeout

### Reasoning Engine

SVEND uses **Synara**, a Bayesian belief network for tool selection and execution. See `SYNARA_WHITEPAPER.md` for technical details.

Key properties:
- Executes tools deterministically (no hallucinated results)
- Explicit uncertainty (knows when it doesn't know)
- Learns from success/failure via Bayesian updates
- No gradient descent required

---

## Pricing & Growth

### Pricing

| Tier | Price | Features |
|------|-------|----------|
| **Individual** | $5/month | Full workbench, all tools, local models |
| **Team** (future) | $15/month | Shared workbenches, comments, dashboards |
| **Enterprise** (future) | Custom | SSO, audit trails, on-prem, API access |

### Market Position

```
Minitab:  $1,500/year  →  Legacy UI, no AI, license hell
JMP:      $1,800/year  →  Powerful but expensive
SVEND:    $60/year     →  Modern UI, AI-assisted, runs anywhere

"Everything you use Minitab for, at 96% less, with AI that helps you think."
```

### Target

6,000 users × $5/month = **$30,000 MRR** = $360K/year

### Growth Path

```
NOW                         +6 MONTHS                    +12 MONTHS
────────────────────────────────────────────────────────────────────
Individual workbench        Team features                Enterprise
Local execution             Email updates                SSO/SAML
DMAIC/Kaizen templates      Shared workbenches           Audit trails
Full statistical suite      Comments/mentions            On-prem option
$5/month                    $15/month team               Custom pricing
```

### Channels

| Channel | Expected Reach |
|---------|---------------|
| LinkedIn (Quality, Six Sigma groups) | 500-1,000 users |
| Reddit (r/sixsigma, r/manufacturing) | 200-500 users |
| ASQ community/conferences | 500-1,000 users |
| SEO ("Minitab alternative", "free SPC") | 1,000-2,000 users |
| Word of mouth | 1,000-2,000 users |
| YouTube tutorials | 500-1,000 users |

---

## Services Architecture

### Forge (Data Generation)

**Purpose:** Generate synthetic datasets from schema or intent.

**Interface:**
```
Workbench sends:
  - Schema (column definitions) OR
  - Intent ("customer churn dataset with 1000 rows")
  - Parameters (row count, distributions)

Forge returns:
  - Dataset artifact
  - Generation report artifact
```

**Standalone access:** Available at `/forge/` for users who just need data generation.

### Triage (Data Cleaning)

**Purpose:** Clean, validate, and profile raw data.

**Interface:**
```
Workbench sends:
  - Raw dataset
  - Cleaning preferences (handle nulls, outliers, etc.)

Triage returns:
  - Cleaned dataset artifact
  - Quality report artifact (issues found, actions taken)
  - Data profile artifact (distributions, correlations)
```

**Standalone access:** Available at `/triage/` for users who just need data cleaning.

---

## Appendix: Statistical Tool Specifications

### Minitab Parity Checklist

#### Basic Statistics
- [x] Descriptive statistics (mean, median, StdDev, CI)
- [x] Normality tests (Anderson-Darling, Shapiro-Wilk, Ryan-Joiner)
- [x] Outlier detection (Grubbs, Dixon)
- [ ] Distribution fitting

#### ANOVA
- [x] One-way ANOVA
- [x] Two-way ANOVA
- [x] General Linear Model
- [x] Post-hoc tests (Tukey, Dunnett, Fisher, Bonferroni)
- [ ] Balanced ANOVA
- [ ] Nested ANOVA

#### Regression
- [x] Simple linear regression
- [x] Multiple regression
- [x] Stepwise regression
- [x] Best subsets
- [x] Diagnostics (VIF, residuals, leverage, Cook's D)
- [ ] Logistic regression
- [ ] Poisson regression

#### Hypothesis Tests
- [x] 1-sample t-test
- [x] 2-sample t-test
- [x] Paired t-test
- [x] 1-proportion test
- [x] 2-proportion test
- [ ] Equivalence tests (TOST)
- [ ] Non-parametric alternatives (Mann-Whitney, Wilcoxon)

#### DOE
- [x] Full factorial (2^k, 3^k, mixed)
- [x] Fractional factorial (Resolution III-V)
- [x] Plackett-Burman screening
- [x] Response Surface (CCD, Box-Behnken)
- [x] Sample size/power analysis
- [ ] Optimal designs (D-optimal, I-optimal)
- [ ] Taguchi designs
- [ ] Mixture designs

#### Control Charts
- [x] I-MR (Individuals and Moving Range)
- [x] X̄-R (Subgroup mean and range)
- [x] X̄-S (Subgroup mean and StdDev)
- [x] p-chart (proportion defective)
- [x] np-chart (count defective)
- [x] c-chart (defects per unit)
- [x] u-chart (defects per unit, variable sample)
- [ ] Laney p' and u' (for overdispersion)
- [x] Western Electric rules
- [ ] Nelson rules

#### Process Capability
- [x] Cp, Cpk
- [x] Pp, Ppk
- [x] Cpm
- [x] Sigma level, DPMO, Yield
- [x] Confidence intervals
- [ ] Non-normal capability (Box-Cox, Johnson)

#### MSA
- [ ] Gage R&R (Crossed)
- [ ] Gage R&R (Nested)
- [ ] Gage Linearity & Bias
- [ ] Attribute Agreement Analysis

---

## References

- `SYNARA_WHITEPAPER.md` - Technical details of Synara reasoning engine
- `ROADMAP_DECISION_SCIENCE.md` - Development roadmap (legacy, superseded by this doc)
- `CLAUDE.md` - Agent context documentation

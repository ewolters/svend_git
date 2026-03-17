**CANON-001: SYSTEM ARCHITECTURE — THREE-LAYER MODEL**

**Version:** 3.0
**Status:** APPROVED
**Date:** 2026-03-08
**Author:** Eric + Claude (Systems Architect)
**Compliance:**
- DOC-001 ≥ 1.2 (Documentation Structure — §7 Machine-Readable Hooks)
- XRF-001 ≥ 1.0 (Cross-Reference Syntax)
- ARCH-001 ≥ 1.0 (Architecture & Structure — layer boundaries)
**Related Standards:**
- QMS-001 ≥ 1.4 (Quality Management System — tooling layer)
- DSW-001 ≥ 1.0 (Decision Science Workbench — analysis layer)
- MAP-001 ≥ 1.0 (Architecture Map — module registry)
- CANON-002 (Integration Contracts — tool chaining schemas, evidence weighting methodology, source-method epistemology)

---

## **1. SCOPE AND PURPOSE**

### **1.1 Purpose**

CANON-001 codifies the three-layer architecture that governs how Svend's systems interact. It is not a code standard — it is an architectural intent document that defines what each layer does, how signals flow between them, and why certain tools exist in certain layers.

**Core Principle:**

> Management cascades down. Problem-solving cascades up. Analysis provides the signals.
> The layers are not a hierarchy of importance — they are a hierarchy of abstraction.
> Each layer consumes the output of the one below it and feeds the one above it.

### **1.2 Scope**

This standard covers:
- Classification of all Svend modules into three layers
- Three tool functions (structure information / intent / inference)
- The investigation engine (Synara) as opt-in Layer 2 infrastructure
- The evidence bridge as connective tissue between layers
- Signal routing rules (which signals trigger which tools)
- Evidence flow patterns (opt-in, bridge fires on linkage + output)
- Layer 3 containers: Project (PMBOK) and Kaizen (Hoshin)
- Tool chaining rules and SPC signal routing UX
- Metric cascade (Hoshin → X-Matrix → Sites → Kaizens)
- Tool registry (canonical list of modules per layer)

Does NOT cover: implementation details of individual tools (see QMS-001, DSW-001), UI/UX patterns (see FE-001), or deployment (see OPS-001).

### **1.3 Terminology**

| Term | Definition |
|------|-----------|
| **Signal** | A statistical output from Layer 1 that indicates something about the process state. Signals do not prescribe action — they inform. |
| **Evidence** | An observation, measurement, or inference that updates the probability of a hypothesis. The common currency of the investigation engine and upward flow to Layer 3. |
| **Decision** | The output of Layer 3 — strategic objectives, corrective actions, kaizen events, task assignments. Decisions cascade downward. |
| **Investigation** | An optional structured problem-solving session within Layer 2. When active, Synara's causal graph connects tools and tracks evolving belief about the problem. When inactive, tools operate as standalone calculators. |

**Three Tool Functions:**

Every Layer 1-2 tool serves one of three functions in the problem-solving process. This classification determines how the tool interacts with an investigation when one is active:

| Function | What it does | Relationship to investigation graph |
|----------|-------------|-------------------------------------|
| **Structure information** | Organizes causal knowledge — maps causes, failure modes, contributor categories | **Builds** the graph — populates hypotheses and causal links |
| **Structure intent** | Designs experiments to test claims — prescribes what to measure and how | **Prescribes** against the graph — targets specific hypotheses for testing |
| **Structure inference** | Produces statistical observations — tests, control charts, simulations | **Updates** the graph — reshapes posteriors via Bayesian evidence |

These functions are not exclusive to a layer. Layer 1 tools primarily structure inference but DOE structures intent. Layer 2 tools primarily structure information but can produce inference (e.g., FMEA risk scores as priors). The function describes the tool's role in the investigation cycle, not its layer membership.

---

## **2. THREE-LAYER ARCHITECTURE**

### **2.1 Overview**

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: SYSTEMS (strategic + tactical)                    │
│  Hoshin (Projects + Kaizens), NCR, CAPA, Action Items       │
│  Produces: Decisions — cascades DOWN                        │
│  Consumes: Investigation conclusions + direct signals       │
├─────────────────────────────────────────────────────────────┤
│  ═══════════ EVIDENCE BRIDGE (connective tissue) ══════════ │
│  Exports investigation conclusions or direct tool output    │
│  to Layer 3 containers when practitioner links them         │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: TOOLING (problem-solving methodology)             │
│  Tools: RCA, Ishikawa, C&E Matrix, FMEA, A3, VSM, 8D       │
│  Investigation engine: Synara causal graph (opt-in)         │
│  Structure information → Build graph (hypotheses, links)    │
│  Structure intent → Design tests (DOE targets claims)       │
│  Structure inference → Update graph (evidence, posteriors)  │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: ANALYSIS (data-driven insight)                    │
│  DSW, SPC, DOE, ML, Triage, Forecast                       │
│  Produces: Signals — feed Layer 2 investigations or         │
│  operate standalone as calculators                          │
└─────────────────────────────────────────────────────────────┘
```

**Two modes of operation:**

All Layer 1-2 tools operate in one of two modes. The mode is determined by whether the practitioner has started or joined an investigation:

- **Standalone (default):** Tool operates as a pure calculator. Produces output, displays results, done. No graph, no evidence propagation. This is the experience for Free/Pro users doing exploratory analysis.
- **Investigation (opt-in):** Tool is connected to a Synara causal graph. Its output is interpreted according to its function — building structure, prescribing intent, or updating inference. The investigation tracks the evolving understanding of the problem.

A practitioner can start an investigation at any time, and any standalone tool output can be linked into an active investigation retroactively. The investigation does not change what the tool does — it changes what happens to the output.

### **2.2 Layer 1 — Analysis**

Data-driven insight generation. These modules consume raw data and produce **signals** — statistical outputs that inform the user about the state of the process.

| Module | Purpose | Outputs | Tool Function |
|--------|---------|---------|---------------|
| DSW | 200+ statistical analyses | Test results, p-values, effect sizes | Inference |
| SPC | Control charts, capability | Signal type (special/common cause), Cpk | Inference |
| DOE | Experimental design | Optimal factor settings, power analysis | Intent |
| ML | Machine learning models | Predictions, feature importance | Inference |
| Triage | Data cleaning | Clean datasets, quality scores | Inference |
| Forecast | Time series | Trend projections, anomaly flags | Inference |

**Key property:** Analysis modules do not prescribe action. They produce signals that inform the user (and the system) about what kind of problem exists. The user or the system then selects the appropriate tool.

**DOE is a Layer 1 module that structures intent.** It lives in Layer 1 because it consumes data (power analysis, prior results) and produces a statistical artifact (experimental design). But its function is prescriptive — it tells you *what to test and how*. When connected to an investigation, DOE targets specific hypotheses in the causal graph for experimental verification.

**Standalone behavior:** Without an investigation, every Layer 1 tool is a pure calculator. DSW runs a t-test. SPC draws a chart. DOE generates a design. No graph, no evidence propagation.

**Investigation behavior:** When connected to an active investigation, Layer 1 outputs become evidence that updates posteriors in the causal graph. SPC signals trigger hypothesis evaluation. DOE designs target specific graph nodes. The tool output is the same — the interpretation changes.

### **2.3 Layer 2 — Tooling**

Problem-solving methodology. Layer 2 contains **tools** that structure knowledge about a problem, and an optional **investigation engine** (Synara) that connects them into a coherent problem-solving workflow.

#### **2.3.1 Tools by Function**

| Tool | Function | What it produces | Investigation role |
|------|----------|-----------------|-------------------|
| RCA | Information | Causal chain (5-Why) | Builds a linear hypothesis path in the graph |
| Ishikawa | Information | Cause-and-effect map (6M) | Populates competing hypotheses across categories |
| C&E Matrix | Information | Prioritized cause ranking | Assigns prior weights to hypotheses |
| FMEA | Information | Failure modes with risk scores | Populates hypotheses with risk-weighted priors |
| A3 | Report | Structured problem-solving report | Synthesizes investigation state into a document |
| VSM | Information | Current/future state process map | Produces kaizen proposals (feeds Layer 3 directly) |
| 8D | Report | Customer complaint investigation | Synthesizes investigation into complaint response |

**Key property:** Layer 2 tools structure information about the problem. When used standalone, they are methodological instruments — an Ishikawa diagram is a diagram, an FMEA is a spreadsheet. When connected to an investigation, their output has semantic meaning: an Ishikawa's branches become hypotheses, an FMEA's failure modes become risk-weighted claims, an RCA's causal chain becomes a directed path in the graph.

#### **2.3.2 Investigation Engine (Synara)**

The investigation engine is optional Layer 2 infrastructure that connects tools into a structured problem-solving workflow following the Box & Hunter deductive cycle:

```
Conjecture → Design → Experiment → Analysis → (revised) Conjecture → ...
```

| Cycle Phase | Maps to | Tool Functions Involved |
|-------------|---------|------------------------|
| Conjecture | Hypotheses in causal graph | Information tools build the graph |
| Design | Experimental plan | Intent tools (DOE) target specific hypotheses |
| Experiment | Data collection | (External — practitioner runs the experiment) |
| Analysis | Belief update | Inference tools (SPC, DSW) produce evidence |
| Revised conjecture | Expansion signals | Synara detects incomplete causal surface |

**Synara's causal graph** represents the practitioner's evolving understanding of the problem:
- **Hypotheses** are behavioral region claims (not point predictions)
- **Causal links** are directed edges with strength and mechanism
- **Evidence** reshapes belief via Bayesian updating
- **Expansion signals** fire when all hypotheses have low likelihood — the model detects its own incompleteness

The investigation is the graph. There is no separate container. The practitioner's understanding *is* the DAG of hypotheses, causal links, and accumulated evidence.

#### **2.3.3 Special vs Common Cause**

| Type | Definition | Tool | Action Pattern |
|------|-----------|------|----------------|
| Special cause | Unique, identifiable event | RCA (5-Why causal chain) | Investigate → find root cause → countermeasure |
| Common cause | Systemic, inherent variation | Ishikawa + C&E Matrix | Map contributors → prioritize → Kaizen |

These are mutually exclusive for a given problem. A plane crash (unique event) gets RCA. Car crashes (process-level effect) get Ishikawa. You do not flow from one into the other — you choose based on the nature of the problem.

### **2.4 Layer 3 — Systems**

Strategic and tactical management. Layer 3 has **two management mechanisms** that serve different purposes, plus standalone reactive containers.

**Hoshin manages PERFORMANCE.** It is proactive — strategy deployment, performance measures, continuous improvement. Hoshin drives Projects (PMBOK, milestones) and Kaizens (Shape/Execute/Consolidate, performance targets). Hoshin is never triggered by reactive events. It sets direction.

**QMS manages the SYSTEM.** It is the quality system lifecycle — NCRs track nonconformances, CAPAs correct and prevent recurrence. QMS is reactive by nature: something went wrong, contain it, fix it, prevent it. QMS does not feed into Hoshin. Turning a proactive management system into a reactive metric-chasing tool defeats the purpose of both.

Quality CAN be improved via Hoshin (a kaizen targeting defect reduction is valid). But NCRs and CAPAs do not trigger Kaizens or Hoshin Projects. The improvement intent comes from Hoshin downward, not from QMS upward.

**Visibility path:** QMS trend data (NCR frequency, categories, root cause patterns, CAPA effectiveness rates) is surfaced to Hoshin planners as an input to strategic planning. This is reporting, not a trigger — the data informs, the planner decides. The mechanism is defined in QMS-001.

| System | Mechanism | Scope | Containers |
|--------|-----------|-------|------------|
| Hoshin Kanri | Performance management (proactive) | Strategy deployment, X-Matrix, performance measures | Projects + Kaizens |
| Projects | Hoshin or standalone | Defined start/end, milestones, deliverables (PMBOK) | — |
| Kaizens | Hoshin only | Performance measure improvement, Shape/Execute/Consolidate | — |
| QMS | Quality management (reactive) | Quality system lifecycle | NCR + CAPA |
| NCR | QMS | Nonconformance tracking, containment, disposition | — |
| CAPA | QMS | Corrective/preventive action, effectiveness verification | — |
| Action Items | Cross-cutting | Task tracking across all containers | — |

**Key property:** Systems consume the output of Layer 2 investigations (or direct Layer 1 signals) and produce decisions that cascade downward (objectives, tasks, corrective actions). However, the flow is not exclusively upward — NCRs and CAPAs can **generate evidence** during their own investigation workflows. An NCR root cause determination is evidence. A CAPA effectiveness check is evidence.

**Bridge behavior:** The evidence bridge exports investigation conclusions or standalone tool outputs to Layer 3 containers when the practitioner links them. Layer 3 does not reach into the investigation — the practitioner decides what to surface. The bridge fires when the practitioner has linked a tool or investigation to a Layer 3 container AND the tool has produced output. Without a linkage, tools and investigations operate independently. Layer 3 can also generate its own evidence (NCR root cause determination, CAPA effectiveness check) which feeds back into the system.

---

## **3. SIGNAL ROUTING**

### **3.1 SPC Signal Routing**

When SPC detects a signal, the type determines which Layer 2 tool is **recommended**:

| SPC Signal | Cause Type | Recommended Tool | Rationale |
|------------|-----------|-----------------|-----------|
| Point beyond control limits | Special | RCA | Unique event — investigate the assignable cause |
| Run of 7+ | Special | RCA | Assignable cause present — investigate |
| 2 of 3 beyond 2σ | Special | RCA | Assignable cause present — investigate |
| Stable but off-target (Cpk < 1) | Common | Ishikawa | Systemic — map contributors, improve process |
| Excessive variation (Cp < 1) | Common | Ishikawa | Systemic — map contributors, reduce variation |
| No signals detected, capable | — | None | Process in control — no action needed |

**Override policy:** Both options are always available. The system recommends based on signal type, but the user decides based on process knowledge. The user may always override. Signal routing is presented as part of standard DSW output for SPC analyses — it is a recommendation, not a gate.

### **3.2 Cross-Layer Signal Flow**

```
Analysis (Layer 1)
  │
  ├── Special cause signal ──→ RCA (Layer 2) ──→ Investigation ──→ NCR/CAPA/Project (Layer 3)
  │
  ├── Common cause signal ──→ Ishikawa (Layer 2) ──→ Investigation ──→ Kaizen/NCR/CAPA (Layer 3)
  │
  └── Direct signal ──────────────────────────via bridge──→ Project/Kaizen (Layer 3)
```

Analysis can feed Layer 3 directly when the signal is self-contained (e.g., an SPC capability result linked to a Kaizen performance measure). No intermediate investigation is needed when the data speaks for itself.

**VSM → Hoshin (direct, no evidence bridge):** VSM's future state diff produces kaizen proposals, not evidence. The diff between current and future state is itself a hypothesis — "if we change the process this way, we'll hit this performance target." This is organizational intent codified in the Hoshin, not a finding. Evidence comes later when the kaizen is executed and results are measured. VSM bypassing the evidence bridge is correct by design.

### **3.3 Tool Chaining**

Tools can chain to other tools. Chaining passes context (the finding, failure mode, cause, etc.) from one tool to the next. All chaining is user-initiated — the tool offers the option, the user decides.

**Context schemas and integration methods for each chain are defined in CANON-002.** This section defines *which* chains exist and *when* they are offered. CANON-002 defines *what* data passes and *how* it integrates with the investigation graph.

#### **3.3.1 Layer 2 → Layer 2 Chains**

| From | To | What passes | When offered |
|------|-----|-------------|-------------|
| FMEA | RCA | Failure mode → investigation target | Severity ≥ 8 (default, org-configurable) |
| FMEA | Ishikawa | Failure mode → effect to map | Severity ≥ 8 (default, org-configurable) |
| FMEA | C&E Matrix | Failure mode → output to score against | Severity ≥ 8 (default, org-configurable) |
| Ishikawa | C&E Matrix | Top-level causes → matrix inputs | On completion (1:1 mapping) |
| RCA | A3 | Root cause + countermeasure → root cause section | On completion |
| Ishikawa | A3 | Contributor map → root cause section | On completion |
| C&E Matrix | A3 | Prioritized causes → root cause section | On completion |

**FMEA threshold:** Default is Severity ≥ 8. Team/Enterprise orgs can override this threshold at the org level. The threshold highlights the recommendation — all chain options are always available regardless of threshold.

**A3 is a report sink.** Multiple tools can feed into A3's root cause section. A3 does not chain to other tools — it produces a report.

#### **3.3.2 Layer 3 → Layer 1-2 Chains (QMS)**

NCR and CAPA are Layer 3 containers that pull from any layer below them.

| From | To | What passes |
|------|-----|-------------|
| NCR | RCA | Nonconformance → investigation target |
| NCR | Ishikawa | Nonconformance → effect (if systemic) |
| NCR | FMEA | Nonconformance → failure mode for risk assessment |
| NCR | SPC | Link control chart data as evidence |
| NCR | DSW | Link analysis results as evidence |
| CAPA | RCA | Corrective action needs root cause investigation |
| CAPA | A3 | Investigation → structured report |

#### **3.3.3 Layer 2 → Layer 3 Chains (investigation output feeds up)**

| From | To | What passes |
|------|-----|-------------|
| RCA | NCR/CAPA | Root cause finding |
| Ishikawa | NCR/CAPA | Contributor findings |
| C&E Matrix | NCR/CAPA | Prioritized causes |
| VSM | Hoshin | Kaizen proposals (via diff, not evidence bridge) |

#### **3.3.4 Chains NOT defined**

The following chains are intentionally excluded:
- **SPC → FMEA** (auto-update occurrence): considered, deferred — unclear value
- **8D chains**: 8D is not sufficiently developed to define chains
- **VSM → Layer 2 tools**: VSM feeds calculators and Hoshin, not other investigation tools

<!-- assert: CANON-001-CHAINING — Tool chains follow §3.3 definitions, all user-initiated -->

### **3.4 SPC Signal Routing UX**

After control chart results, display a **signal summary panel** with classification and routing:

**Special cause detected:**
> "**Special cause detected** — 3 points beyond control limits"
> Buttons: **Investigate with RCA** (highlighted) | Investigate with Ishikawa

**Common cause (stable but incapable):**
> "**Process stable but incapable** — Cpk = 0.72"
> Buttons: Investigate with RCA | **Analyze with Ishikawa** (highlighted)

**No signals, capable:**
> "**Process in control** — Cpk = 1.45"
> No action panel. Process is performing as expected.

**Interaction:**
- Clicking a button opens the target tool with context pre-filled
- Effect text = signal description (e.g., "SPC: 3 OOC points on I-MR chart, Cpk = 0.72")
- If an investigation is active, the downstream tool joins the investigation
- If a Layer 3 container is linked, the downstream tool inherits the linkage
- Both options always available — highlighted button is a recommendation, not a constraint

**Backend:**
- `classify_signal_type()` in spc_views.py returns `"special"`, `"common"`, or `"none"` based on §3.1 table
- Classification included in control chart API response so the frontend can render the panel

<!-- Note: classify_signal_type and signal_routing_panel are planned but not yet implemented -->

---

## **4. EVIDENCE FLOW**

### **4.1 Evidence Bridge**

The evidence bridge is the connective tissue between layers. It is not a layer — it is the exchange mechanism that exports investigation conclusions and standalone tool outputs to Layer 3 containers as `core.Evidence` records.

Implementation: `create_tool_evidence()` in `agents_api/evidence_bridge.py`.

<!-- impl: agents_api/evidence_bridge.py:create_tool_evidence -->

**Contract:**
- Idempotent: same `(source_tool, source_id, source_field)` never duplicates
- Neutral confidence: all tool-generated evidence starts at 0.5 — Synara's challenge process elevates confidence, not the source tool
- Feature-flagged: controlled by `settings.EVIDENCE_INTEGRATION_ENABLED`
- Evidence persists even if the source tool output is re-run, unlinked, or deleted — the audit trail is immutable
- **Evidence weighting by source method** (how a control chart's evidence compares to a simulation, how a DOE compares to an observational study, how measurement system quality discounts downstream evidence) **is defined in CANON-002**

### **4.2 Opt-In Integration Model**

**All integration is opt-in.** No tool automatically pushes evidence anywhere. The practitioner controls what flows where.

The evidence bridge fires when TWO conditions are met:
1. The practitioner has **linked** a tool or investigation to a Layer 3 container (Project, Kaizen, NCR, CAPA)
2. The tool has **produced output** (analysis result, completed investigation, status change)

Without condition 1, tools and investigations operate independently. An SPC chart is just a chart. An Ishikawa diagram is just a diagram. An investigation tracks beliefs without surfacing them to management. They become Layer 3 evidence only when the practitioner has connected them to a container.

Tools do NOT auto-create containers. See §4.4.

### **4.3 Source Tool Registry**

All tools create evidence via `create_tool_evidence()` only when linked to a Layer 3 container.

| Tool | Layer | source_tool value | Evidence created when |
|------|-------|------------------|---------------------|
| SPC | 1 | `"spc"` | Chart run with project linkage, OOC detected |
| RCA | 2 | `"rca"` | Chain step accepted, root cause set |
| Ishikawa | 2 | `"ishikawa"` | Diagram completed — top-level causes per category |
| C&E Matrix | 2 | `"ce_matrix"` | Matrix completed — top-scored inputs |
| FMEA | 2 | `"fmea"` | Row saved with severity/occurrence/detection |
| A3 | 2 | `"a3"` | Section completed |
| 8D/Report | 2 | `"report"` | Report section completed |
| NCR | 3 | `"ncr"` | Root cause, containment, or disposition set |
| CAPA | 3 | `"capa"` | Effectiveness verified, corrective action recorded |

### **4.4 No Auto-Container Creation**

**Decision:** Tools do NOT auto-create Layer 3 containers. If a tool has no linked Project or Kaizen, it operates standalone and no evidence is created. The practitioner must explicitly link a tool to an existing container for evidence to flow.

The `_ensure_<tool>_project()` pattern (RCA, Ishikawa, C&E Matrix) is **deprecated and should be removed**. It created unnecessary records.

**Hoshin-created containers MUST set `project_class` appropriately:**
- Kaizen events: `project_class="kaizen"`
- Capital/milestone projects: `project_class="project"`

<!-- impl: agents_api/hoshin_views.py:create_hoshin_project -->

---

## **5. LAYER 3 CONTAINERS**

### **5.1 Two Containers**

Layer 3 has two container types, both managed through the `Project` model distinguished by `project_class`.

| Attribute | Project | Kaizen |
|-----------|---------|--------|
| `project_class` | `"project"` | `"kaizen"` |
| Parent | Hoshin or standalone | Always Hoshin (Policy Deployment) |
| Lifecycle | PMBOK (initiate → plan → execute → close) | Shape → Execute → Consolidate → Shape |
| End condition | Milestones complete | Performance target sustained |
| Tracking | Milestones, deliverables, Gantt | Performance measure trend, events |
| Children | Tasks with due dates | Events (planned) + contingencies (emergent) |
| Example | "Install new press by Q3" | "Reduce changeover time to < 10 min" |

**Tier access:**
- **Pro**: Standalone Projects only (individual, no collaboration)
- **Team**: Collaborative Projects (shared, assignment, viewer roles)
- **Enterprise**: Projects + Kaizens (full Hoshin, X-matrix, cascade)

**Implementation:** `core.Project.project_class` field with `TextChoices` — choices: `"project"`, `"kaizen"`. This is the single source of truth. `HoshinProject.project_class` is deprecated and should be dropped — the core model owns this classification.

<!-- impl: core/models/project.py:Project.ProjectClass -->

### **5.2 Projects (PMBOK)**

Defined-scope work with milestones, deliverables, and a timeline. Projects have a start and an end — you accomplish A, B, C and you're done. Success is measured by milestones hit, not performance targets.

Projects follow PMBOK lifecycle: Initiate → Plan → Execute → Monitor & Control → Close.

Characteristics:
- Defined start/end, bounded deliverables
- Gantt charts, resource allocation, budget tracking
- May be Hoshin-linked (strategic) or standalone
- Use cases: capital expenditure, HR initiatives, facility projects, system implementations
- Success = milestones achieved, deliverables accepted

### **5.3 Kaizens (Hoshin)**

Continuous improvement events linked to Hoshin performance measures. Kaizens have no defined end — they impact a performance measure through iterative cycles.

Hoshin specifies intent via performance measures. The kaizen charter is the tactical intent. Multiple events fork off a kaizen, including contingencies (pure upside, often unplanned or secondary).

**Lifecycle: Shape → Execute → Consolidate → Shape**
- **Shape:** 5S, standard work, prework, preparation
- **Execute:** Trial of current vs future state
- **Consolidate:** Auditing, standardization, training
- Consolidation bleeds back into shaping — the cycle repeats

Characteristics:
- No defined end — follows performance measure
- Always linked to Hoshin (Policy Deployment)
- Success = performance target sustained, not a milestone checklist
- Children: planned events + contingency events (emergent)
- Requires Team/Enterprise tier with Hoshin feature enabled
- Use case: "Create a SMED charter, link tools and analyses, create a report-out, manage secondary events with contingencies"

**Note:** Shape/Execute/Consolidate lifecycle is not yet implemented. Current Hoshin kaizen events use a simplified status model. Full lifecycle is a future feature.

### **5.4 Resource Management & Notifications**

Projects and Kaizens assign people to tasks, milestones, and charters.

**User notifications:** Users choose email or in-app notification for assignments, approvals, and status changes. Full specification in NTF-001.

**Viewer role:** Team and Enterprise tiers can add viewer-role users who can see dashboards, approve tasks, and respond to assignments but cannot run tools. This replaces the non-user digest concept — all participants have accounts, gated by role permissions. Viewer pricing defined in BILL-001.

**Decision:** Non-user digest (token-authenticated, no-account participation) was considered and rejected in favor of viewer accounts. Same Django auth, simpler architecture, no parallel token system needed. The `viewer` role already exists in the tenant model (owner/admin/member/viewer).

### **5.5 Decision: Studies Eliminated**

**Decision date:** 2026-03-07, revised 2026-03-08

Studies (lightweight investigation containers) were considered and rejected. Rationale:
- The investigation engine (§2.3.2) fulfills the structured problem-solving need that Studies attempted to address — connecting tools, tracking hypotheses, accumulating evidence
- Studies overlapped with Kaizen for enterprise users
- A Study was either a poor Kaizen (without Hoshin) or a hypothesis container (which is Synara's job)
- Adding a third Layer 3 container type created confusion about what to use when

**Individual experience:** Free users have Layer 1-2 tools as standalone calculators. Pro users can start investigations (Synara causal graph) to connect their tools into structured problem-solving workflows. Investigations are Layer 2 — they do not require a Layer 3 container. Team/Enterprise users get collaborative Projects and Kaizens via Hoshin for management. The investigation serves problem-solving; Layer 3 containers serve management. These are different concerns.

<!-- assert: CANON-001-PROJECT-PMBOK — Projects follow PMBOK lifecycle with milestones -->
<!-- assert: CANON-001-KAIZEN-HOSHIN — Kaizens are always Hoshin-linked with performance measure tracking -->

### **5.6 Metric Cascade (Hoshin → X-Matrix → Sites → Kaizens)**

#### **5.6.1 Cascade Model**

The X-matrix defines org-level objectives. Each objective has a **cascade method** — how the target splits to sites. Sites respond via catchball by selecting **calculation methods** for their kaizens. The metric and the dollar value are two views of the same reality — the user picks which view to cascade and display by.

```
X-Matrix (Org)
  │  Objective: "Reduce material spend"
  │  Cascade method: volume
  │  Target: $2M (or 500K lbs waste reduction)
  │  Display: $ or native unit (user choice)
  │
  ├── Site A (allocated 60% by volume)
  │     Target: $1.2M (or 300K lbs)
  │     └── Kaizen 1: waste_pct calculation → $800K impact
  │     └── Kaizen 2: layout calculation → $400K impact
  │     └── Total: $1.2M ✓ meets allocation
  │
  └── Site B (allocated 40% by volume)
        Target: $800K (or 200K lbs)
        └── Kaizen 3: freight calculation → $500K impact
        └── Kaizen 4: direct calculation → $200K impact
        └── Total: $700K ✗ $100K gap visible in X-matrix
```

#### **5.6.2 Three User Choices**

1. **Calculation method** — what are we measuring and how (waste_pct, time_reduction, headcount, etc.). Chosen at the kaizen/site level. Different kaizens under the same objective can use different calculation methods.

2. **Display unit** — native metric (seconds, FTEs, lbs, defects) or dollars. Two views of the same math. `(baseline - actual) × volume × rate` works both directions.

3. **Cascade allocation method** — how the org target splits to sites. Chosen at the X-matrix objective level. One per objective, not mixed.

#### **5.6.3 Cascade Allocation Methods**

| Method | Allocates by | Right for | Example |
|--------|-------------|-----------|---------|
| **Volume** | Proportional to site production volume | Material, throughput | Site producing 60% of org volume gets 60% of target |
| **Mix** | Proportional to activity count (setups, changeovers, etc.) | SMED, changeover | Site with 200 setups/day gets larger share than site with 50 |
| **Dollar** | Flat dollar allocation per site | Cost reduction, energy, freight | "Site A: $500K, Site B: $300K" |
| **Headcount** | Proportional to site headcount or FTE | Labor efficiency | Site with 200 operators gets proportional FTE target |
| **Defect** | DPMO target per site, volume converts to defect count | Quality, Cpk improvement | Cpk → DPMO → defects/unit × volume = defect target |

#### **5.6.4 Catchball**

The cascade is not top-down dictation. It is catchball:

1. **Org** sets objective + cascade method + target (X-matrix)
2. **Sites** receive their allocation and **respond** with planned kaizens using calculation methods that normalize to (or are equivalent to) the org intent
3. **Validation**: do site responses, when normalized to the cascade unit, sum to the org target? Gaps are visible in the X-matrix.
4. Sites cannot game the cascade — a material savings kaizen does not count against a labor target. The calculation method's category must be compatible with the objective's intent.

#### **5.6.5 Calculation Method Categories**

Each calculation method has a category. The cascade objective has an intent category. Kaizen calculation methods must be compatible with the objective's category.

| Calculation Method | Category | Compatible Cascade Intents |
|---|---|---|
| waste_pct | material | Material, cost reduction |
| layout | material | Material, cost reduction |
| time_reduction | labor | Labor, throughput, cost reduction |
| headcount | labor | Labor, cost reduction |
| claims | quality | Quality, cost reduction |
| freight | other | Logistics, cost reduction |
| energy | other | Energy, cost reduction |
| direct | other | Logistics, energy, cost reduction |
| custom | custom | User-defined (requires justification) |

**Note:** Cost reduction (dollar cascade) is universally compatible because all methods normalize to dollars. Category validation only applies when cascading by native unit.

**Constraint on `direct` and `custom`:** These methods bypass category-specific math and could be used to game cascade validation. To prevent this:
- `direct` and `custom` kaizens require a **justification field** explaining why a category-specific method is not applicable
- `direct` and `custom` are capped at **20% of a site's total kaizen portfolio** (by $ impact) per objective. Exceeding this cap requires owner/champion approval.
- This is a soft cap — the system flags it, the owner decides. Not a hard block.

#### **5.6.6 Implementation Status**

The calculation engine exists (`hoshin_calculations.py`) with 8 methods + custom formula support, Monte Carlo simulation, TTM baseline, and monthly aggregation. What is NOT yet implemented:

- Org-level objective model with cascade method
- Site allocation model
- X-matrix ↔ cascade linkage (currently blank dropdown)
- Catchball workflow (site response + gap visibility)
- Category validation (kaizen calc method vs objective intent)

<!-- assert: CANON-001-CASCADE — X-matrix objectives cascade to sites via defined allocation methods -->
<!-- assert: CANON-001-CATCHBALL — Sites respond to cascade allocations with compatible calculation methods -->

---

## **6. TOOL REGISTRY**

### **6.1 Canonical Tool List**

| Tool | Layer | Function | Model | Views | Template | API Prefix |
|------|-------|----------|-------|-------|----------|------------|
| DSW | 1 | Inference | DSWResult | dsw_views.py | workbench_new.html | /api/dsw/ |
| SPC | 1 | Inference | (inline) | spc_views.py | spc.html | /api/spc/ |
| DOE | 1 | Intent | ExperimentDesign | experimenter_views.py | experimenter.html | /api/experimenter/ |
| ML | 1 | Inference | (via DSW registry) | dsw_views.py | workbench_new.html | /api/dsw/ |
| Triage | 1 | Inference | TriageResult | triage_views.py | triage.html | /api/triage/ |
| Forecast | 1 | Inference | (inline) | forecast_views.py | forecast.html | /api/forecast/ |
| RCA | 2 | Information | RCASession | rca_views.py | rca.html | /api/rca/ |
| Ishikawa | 2 | Information | IshikawaDiagram | ishikawa_views.py | ishikawa.html | /api/ishikawa/ |
| C&E Matrix | 2 | Information | CEMatrix | ce_views.py | ce_matrix.html | /api/ce/ |
| FMEA | 2 | Information | FMEA, FMEARow | fmea_views.py | fmea.html | /api/fmea/ |
| A3 | 2 | Report | A3Report | a3_views.py | a3.html | /api/a3/ |
| VSM | 2 | Information | ValueStreamMap | vsm_views.py | vsm.html | /api/vsm/ |
| 8D | 2 | Report | Report | report_views.py | report.html | /api/reports/ |
| Hoshin | 3 | — | HoshinProject | hoshin_views.py | hoshin.html | /api/hoshin/ |
| NCR | 3 | — | NonconformanceRecord | iso_views.py | iso.html | /api/iso/ |
| CAPA | 3 | — | CAPAReport | capa_views.py | — | /api/capa/ |
| Action Items | 3 | — | ActionItem | action_views.py | — | /api/actions/ |

### **6.2 Adding New Tools**

When adding a new tool to any layer:
1. Model in `agents_api/models.py` (UUID PK, owner FK, project FK, status choices, timestamps)
2. Views in `agents_api/<tool>_views.py` (CRUD + evidence hooks)
3. URLs in `agents_api/<tool>_urls.py`
4. Template in `templates/<tool>.html` (extends `base_app.html`)
5. Wired in `svend/urls.py` (API + template routes)
6. Navbar entry in `templates/base_app.html`
7. Tests in `agents_api/<tool>_tests.py`
8. Standard section in QMS-001 with `<!-- impl: -->` and `<!-- test: -->` hooks
9. Update this registry (§6.1)
10. **Register in §4.3** — add `source_tool` value and evidence creation conditions

---

## **7. ASSERTIONS**

<!-- assert: CANON-001-LAYERS — Three layers exist: Analysis (signals), Tooling (methodology + investigation), Systems (decisions) -->
<!-- assert: CANON-001-FUNCTIONS — Tools classified as structure information, structure intent, or structure inference -->
<!-- assert: CANON-001-INVESTIGATION — Investigation engine (Synara) is opt-in Layer 2 infrastructure, graph is the investigation -->
<!-- assert: CANON-001-BRIDGE — Evidence bridge exports investigation conclusions or direct tool output to Layer 3, all integration opt-in -->
<!-- assert: CANON-001-OPT-IN — No tool auto-pushes evidence; bridge fires on practitioner linkage + tool output -->
<!-- assert: CANON-001-ROUTING — SPC signals route to appropriate Layer 2 tool with user override -->
<!-- assert: CANON-001-EVIDENCE — All Layer 1-2 tools use create_tool_evidence() via evidence bridge -->
<!-- assert: CANON-001-PROJECT-CLASS — Project.project_class distinguishes project and kaizen -->
<!-- assert: CANON-001-HOSHIN-KAIZEN — Hoshin-created kaizens set project_class="kaizen" -->
<!-- assert: CANON-001-REGISTRY — All tools in §6.1 have model, views, urls, template, tests -->

---

## **CHANGELOG**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-07 | Initial release — three-layer model, tool registry, evidence flow |
| 2.0 | 2026-03-07 | Major revision: signals/findings/evidence taxonomy, corrected tool registry, added §1.3 terminology, rewrote §5 for Layer 3 containers |
| 2.1 | 2026-03-07 | Corrected evidence flow: all integration opt-in, Layer 3 pulls from below |
| 2.2 | 2026-03-07 | Layer 3 containers: Project (PMBOK) + Kaizen (Hoshin). Studies eliminated. Hoshin = performance (proactive), QMS = quality (reactive), no crossover. VSM→Hoshin = intent not evidence. Tool chaining map (§3.3). SPC routing UX (§3.4). Metric cascade system (§5.6). FMEA Severity ≥ 8 threshold with org override. Auto-project creation removed. HoshinProject.project_class deprecated in favor of core.Project. Non-user digest replaced by viewer role (§5.4). Standalone Projects for Pro tier. |
| 3.0 | 2026-03-08 | Major revision: Three tool functions (structure information / structure intent / structure inference) replace flat tool list. Investigation engine (Synara causal graph) formalized as opt-in Layer 2 infrastructure. Box & Hunter deductive cycle mapped to tool functions. Two modes of operation (standalone calculator vs investigation-connected). "Finding" removed from terminology — tools build graph structure, not findings. Evidence bridge exports investigation conclusions, not individual tool outputs. Layer 2 reframed as problem-solving methodology, not just instruments. "No signals → Ishikawa" routing removed — stable capable process gets no action. "Pull" language replaced with event-driven bridge semantics. CANON-002 forward references added for integration contracts, chaining schemas, and evidence weighting methodology. QMS → Hoshin visibility path added (reporting, not trigger). |

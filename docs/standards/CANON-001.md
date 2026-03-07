**CANON-001: SYSTEM ARCHITECTURE — THREE-LAYER MODEL**

**Version:** 1.0
**Status:** APPROVED
**Date:** 2026-03-07
**Author:** Eric + Claude (Systems Architect)
**Compliance:**
- DOC-001 ≥ 1.2 (Documentation Structure — §7 Machine-Readable Hooks)
- XRF-001 ≥ 1.0 (Cross-Reference Syntax)
- ARCH-001 ≥ 1.0 (Architecture & Structure — layer boundaries)
**Related Standards:**
- QMS-001 ≥ 1.4 (Quality Management System — tooling layer)
- DSW-001 ≥ 1.0 (Decision Science Workbench — analysis layer)
- MAP-001 ≥ 1.0 (Architecture Map — module registry)

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
- Signal routing rules (which signals trigger which tools)
- Evidence flow patterns (how findings propagate upward)
- Project classification (strategic vs tactical)
- Tool registry (canonical list of modules per layer)

Does NOT cover: implementation details of individual tools (see QMS-001, DSW-001), UI/UX patterns (see FE-001), or deployment (see OPS-001).

---

## **2. THREE-LAYER ARCHITECTURE**

### **2.1 Overview**

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: SYSTEMS (strategic + tactical)                │
│  QMS, Hoshin/Kaizen, Projects, Action Items, NCR, CAPA  │
│  Management cascades DOWN                               │
├─────────────────────────────────────────────────────────┤
│  Layer 2: TOOLING (problem-solving instruments)         │
│  RCA, Ishikawa, C&E Matrix, FMEA, A3, VSM, 8D          │
│  Findings cascade UP via evidence bridge                │
├─────────────────────────────────────────────────────────┤
│  Layer 1: ANALYSIS (data-driven insight)                │
│  DSW, SPC, DOE, ML, Triage, Forecast                   │
│  Signals trigger tooling or feed directly to systems    │
└─────────────────────────────────────────────────────────┘
```

### **2.2 Layer 1 — Analysis**

Data-driven insight generation. These modules consume raw data and produce signals, statistics, and visualizations.

| Module | Purpose | Outputs |
|--------|---------|---------|
| DSW | 200+ statistical analyses | Test results, p-values, effect sizes |
| SPC | Control charts, capability | Signal type (special/common cause), Cpk |
| DOE | Experimental design | Optimal factor settings, power analysis |
| ML | Machine learning models | Predictions, feature importance |
| Triage | Data cleaning | Clean datasets, quality scores |
| Forecast | Time series | Trend projections, anomaly flags |

**Key property:** Analysis modules do not prescribe action. They produce signals that inform the user (and the system) about what kind of problem exists. The user or the system then selects the appropriate tool.

### **2.3 Layer 2 — Tooling**

Problem-solving instruments. Each tool is designed for a specific type of investigation or analysis.

| Tool | Signal Type | Purpose |
|------|-------------|---------|
| RCA | Special cause | Investigate unique causal chain for one event |
| Ishikawa | Common cause | Map systemic contributors to process-level effect |
| C&E Matrix | Common cause | Prioritize causes by weighted output scoring |
| FMEA | Proactive | Quantify risk before failures occur |
| A3 | Any | Structured problem-solving report (Toyota) |
| VSM | Process | Visualize flow, identify waste |
| 8D | Reactive | Customer complaint investigation |

**Key property:** Tools produce findings that cascade upward to the systems layer via the evidence bridge. Tools can be triggered by analysis signals (SPC → RCA/Ishikawa) or by system events (NCR → RCA, CAPA → A3).

**Special vs Common Cause:**

| Type | Definition | Tool | Action Pattern |
|------|-----------|------|----------------|
| Special cause | Unique, identifiable event | RCA (5-Why causal chain) | Investigate → find root cause → countermeasure |
| Common cause | Systemic, inherent variation | Ishikawa + C&E Matrix | Map contributors → prioritize → Kaizen |

These are mutually exclusive for a given problem. A plane crash (unique event) gets RCA. Car crashes (process-level effect) get Ishikawa. You do not flow from one into the other — you choose based on the nature of the problem.

### **2.4 Layer 3 — Systems**

Strategic and tactical management. These modules track execution, enforce compliance, and drive improvement.

| System | Scope | Classification |
|--------|-------|---------------|
| Hoshin Kanri | Strategy deployment, X-Matrix, KPI rollup | Strategic |
| Projects | Investigation hub (5W2H, goals, scope, team) | Strategic or Tactical |
| QMS | Quality management lifecycle | Strategic |
| NCR | Nonconformance tracking | Tactical (reactive) |
| CAPA | Corrective/preventive action | Tactical (reactive) |
| Action Items | Cross-module task tracking | Both |

**Key property:** Systems consume findings from Layer 2 tools. Hoshin cascades objectives downward into projects. NCRs and CAPAs trigger Layer 2 tools. Action items bridge execution between layers.

---

## **3. SIGNAL ROUTING**

### **3.1 SPC Signal Routing**

When SPC detects a signal, the type determines which Layer 2 tool is recommended:

| SPC Signal | Cause Type | Recommended Tool | Rationale |
|------------|-----------|-----------------|-----------|
| Point beyond control limits | Special | RCA | Unique event — investigate |
| Run of 7+ | Special | RCA | Assignable cause present |
| 2 of 3 beyond 2σ | Special | RCA | Assignable cause present |
| Stable but off-target (Cpk < 1) | Common | Ishikawa | Systemic — improve process |
| Excessive variation (Cp < 1) | Common | Ishikawa | Systemic — reduce variation |
| No signals detected | Common | Ishikawa | Process improvement opportunity |

**Override policy:** Both options are always available. The system recommends based on signal type, but the user decides based on process knowledge. The user may always override.

### **3.2 Cross-Layer Signal Flow**

```
Analysis (Layer 1)
  │
  ├── Special cause signal ──→ RCA (Layer 2) ──→ Evidence ──→ Project/NCR (Layer 3)
  │
  ├── Common cause signal ──→ Ishikawa (Layer 2) ──→ Evidence ──→ Project/Kaizen (Layer 3)
  │
  └── Direct insight ──────────────────────────────→ Project/Hypothesis (Layer 3)
```

Analysis can bypass Layer 2 when the insight is self-contained (e.g., a DOE result that directly informs a hypothesis).

---

## **4. EVIDENCE FLOW**

### **4.1 Evidence Bridge**

All Layer 2 tools push findings to the core evidence system via `create_tool_evidence()` in `agents_api/evidence_bridge.py`.

<!-- impl: agents_api/evidence_bridge.py::create_tool_evidence -->

**Contract:**
- Idempotent: same `(source_tool, source_id, source_field)` never duplicates
- Neutral confidence: all tool-generated evidence starts at 0.5
- Feature-flagged: controlled by `settings.EVIDENCE_INTEGRATION_ENABLED`

**Source tools:**

| Tool | source_tool value | When evidence is created |
|------|------------------|------------------------|
| RCA | `"rca"` | Chain step accepted, root cause set |
| Ishikawa | `"ishikawa"` | Diagram completed — top-level causes per category |
| C&E Matrix | `"ce_matrix"` | Matrix completed — top-scored inputs |
| FMEA | `"fmea"` | Row saved with severity/occurrence/detection |
| A3 | `"a3"` | Section completed |
| NCR | `"ncr"` | Root cause, containment, or disposition set |

### **4.2 Auto-Project Creation**

Each Layer 2 tool auto-creates a Project (Layer 3) if none is linked. Pattern: `_ensure_<tool>_project()`.

<!-- impl: agents_api/rca_views.py::_ensure_rca_project -->

**Rules:**
- Project `title` derived from tool title or first content field
- Project `project_class` set to `"investigation"` (tactical)
- Project `tags` include `["auto-created", "<tool-name>"]`
- Silent — no notification, no redirect

---

## **5. PROJECT CLASSIFICATION**

### **5.1 Project Classes**

| Class | Label | Trigger | Example |
|-------|-------|---------|---------|
| `investigation` | Investigation / Study | Signal from analysis, NCR, user-initiated | "Why is Line 3 yield dropping?" |
| `strategic` | Strategic / Hoshin | Cascaded from Hoshin objectives | "Reduce scrap 15% by Q4" |

**Implementation:** `Project.project_class` field with `TextChoices`.

<!-- impl: core/models/project.py::Project.ProjectClass -->

### **5.2 Investigation (Study) Projects**

Tactical, triggered by signals. Characteristics:
- Short-lived (days to weeks)
- Tied to specific observations or events
- May reference NCRs, CAPAs, or SPC signals
- Structured data pipeline: tool findings → evidence → hypothesis → decision

### **5.3 Strategic Projects**

Cascaded from Hoshin Kanri. Characteristics:
- Long-lived (months to years)
- Tied to strategic objectives and KPIs
- Tracked via X-Matrix correlation
- Progress reported via Hoshin review cycles

---

## **6. TOOL REGISTRY**

### **6.1 Canonical Tool List**

| Tool | Layer | Model | Views | Template | API Prefix |
|------|-------|-------|-------|----------|------------|
| DSW | 1 | DSWResult | dsw_views.py | dsw_*.html | /api/dsw/ |
| SPC | 1 | (inline) | spc_views.py | spc.html | /api/spc/ |
| DOE | 1 | ExperimentDesign | experimenter_views.py | doe.html | /api/experimenter/ |
| RCA | 2 | RCASession | rca_views.py | rca.html | /api/rca/ |
| Ishikawa | 2 | IshikawaDiagram | ishikawa_views.py | ishikawa.html | /api/ishikawa/ |
| C&E Matrix | 2 | CEMatrix | ce_views.py | ce_matrix.html | /api/ce/ |
| FMEA | 2 | FMEA, FMEARow | fmea_views.py | fmea.html | /api/fmea/ |
| A3 | 2 | A3Report | a3_views.py | a3.html | /api/a3/ |
| VSM | 2 | ValueStreamMap | vsm_views.py | vsm.html | /api/vsm/ |
| Hoshin | 3 | HoshinProject | hoshin_views.py | hoshin.html | /api/hoshin/ |
| NCR | 3 | NonconformanceRecord | ncr_views.py | — | /api/ncr/ |
| CAPA | 3 | CAPAReport | report_views.py | — | /api/reports/ |

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

---

## **7. ASSERTIONS**

<!-- assert: CANON-001-LAYERS — Three layers exist: Analysis, Tooling, Systems -->
<!-- assert: CANON-001-ROUTING — SPC signals route to appropriate Layer 2 tool -->
<!-- assert: CANON-001-EVIDENCE — All Layer 2 tools use create_tool_evidence() -->
<!-- assert: CANON-001-PROJECT-CLASS — Project.project_class distinguishes investigation from strategic -->
<!-- assert: CANON-001-REGISTRY — All tools in §6.1 have model, views, urls, template, tests -->

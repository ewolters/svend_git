# Analysis Workbench — Full Rebuild Specification

**Object 271 | Status: PLANNING | Date: 2026-04-14**

This is NOT a demo. This is NOT an MVP. This is the richest feature in the application — the center of SVEND. Every other system feeds into or pulls from the workbench.

---

## 1. WHAT EXISTS (Backend — Ready to Wire)

### 1.1 Workbench Models (`workbench/models.py`)

| Model | Purpose | Key Fields |
|-------|---------|------------|
| **Workbench** | Session container | `title, template (blank/DMAIC/Kaizen/8D/A3), status, datasets (JSON list), connections, layout, guide_observations, conclusion` |
| **Artifact** | Unit of work (30+ types) | `artifact_type, title, content (JSON), source, source_artifact_id, probability, supports_hypotheses, weakens_hypotheses, tags` |
| **KnowledgeGraph** | Causal belief network | `nodes, edges (with Bayesian weights), expansion_signals` |
| **EpistemicLog** | Reasoning audit trail | `event_type (25 types), event_data, context, led_to_insight, led_to_dead_end` |

### 1.2 Workbench API (`/api/workbench/` — endpoints used by the workbench)

**Workbench CRUD:**
- `GET /` — List workbenches
- `POST /create/` — Create (blank)
- `GET /<id>/` — Full workbench with all artifacts (JSON serialization)
- `PATCH /<id>/update/` — Update title, description, status, datasets
- `DELETE /<id>/delete/` — Archive or permanent delete
- `POST /import/` — Import from JSON
- `POST /<id>/export/` — Export as JSON file

**Artifact CRUD:**
- `POST /<id>/artifacts/` — Create artifact
- `GET /<id>/artifacts/<aid>/` — Get artifact
- `PATCH /<id>/artifacts/<aid>/update/` — Update content, tags
- `DELETE /<id>/artifacts/<aid>/delete/` — Delete + cleanup connections

**Connections:**
- `POST /<id>/connect/` — Link two artifacts (derivation chain)
- `DELETE /<id>/disconnect/` — Remove link

Note: Knowledge Graph, Epistemic Log, and Guide endpoints exist on the
Workbench model but are consumed by OTHER systems (graph, Loop, etc.),
not by the workbench UI itself. The workbench is a source.

### 1.3 Artifact Types (30+)

**Data:** dataset, cleaned_dataset, generated_dataset
**Analysis:** spc_chart, capability_study, anova, regression, correlation, descriptive_stats, hypothesis_test, forecast
**Experiment:** doe_design, doe_results
**ML:** ml_model, model_metrics, feature_importance
**Thinking:** note, hypothesis, evidence, conclusion
**Documents:** report, summary, control_plan
**Code:** script, visualization
**Research:** research_findings, citation
**Process:** process_map, fishbone, five_whys, action_item

### 1.4 Forge Ecosystem (14 packages, all installed)

| Package | Purpose | Key Entry Points |
|---------|---------|-----------------|
| **forgestat** | 200+ statistical analyses | `parametric`, `nonparametric`, `regression`, `bayesian`, `posthoc`, `quality`, `reliability`, `timeseries`, `msa`, `exploratory`, `power`, `intelligence` |
| **forgespc** | SPC engine | `individuals_moving_range_chart`, `xbar_r_chart`, `calculate_capability`, `gage_rr_crossed`, `bayesian_capability`, `conformal_control` |
| **forgeviz** | 19 chart types, zero-dep SVG | `render`, `compose`, `trellis`, `slider`, `addToolbar`, `enableAnnotation`, `enableThresholdDrag`, `addFilterChips`, `linkCharts` |
| **forgedoe** | DOE engine | `full_factorial`, `fractional_factorial`, `central_composite_design`, `box_behnken`, `fit_model`, `optimize_responses`, `AdaptiveExperiment` |
| **forgepad** | Command-driven analysis | `parse`, `execute`, `Session`, `Registry` — 72 commands across 17 categories |
| **forgesia** | Causal reasoning | `CausalGraph`, `apply_evidence`, `bayes_update`, `loopy_belief_propagation`, `differential_diagnosis`, `score_risk` |
| **forgedoc** | Document generation | `Document`, `A3Sheet`, `InvestigationReport`, `ControlPlanDoc`, `render(format=pdf/docx/html)` |
| **forgequeue** | Queueing theory | `single`, `multi`, `priority`, `network`, `staffing` |
| **forgepbs** | Bayesian process monitoring | Normal-Gamma posteriors, BOCPD, e-detector, capability trajectory, cognitive load analysis |
| **forgecal** | Calibration service | `run_calibration`, `detect_drift`, `CalibrationCase`, golden references |
| **forgegov** | Ecosystem governance | `scan`, `check` (contracts), CI pipeline, audit, compat matrix |
| **forgerack** | Modular instrument UI | 32 units, patch jacks, front/back panel, server compute |
| **forgesiop** | Supply chain/inventory | Demand forecasting, EOQ, DDMRP, MRP, capacity planning, Monte Carlo |
| **forge** (core) | Synthetic data generation | Schema → synthetic data with quality tiers |

### 1.5 Analysis Engine (`agents_api/analysis/`)

**Forge Handlers (new, direct):**
- `forge_stats.py` — 80+ statistical tests via forgestat
- `forge_spc.py` — 25 SPC chart types via forgespc
- `forge_bayesian.py` — Bayesian inference suite
- `forge_ml.py` — Classification, regression, clustering, boosting, SHAP
- `forge_misc.py` — Simulation, causal, drift, anytime, quality_econ, pbs, ishap, msa, reliability

**Supporting Modules:**
- `standardize.py` — Canonical output schema (plots, narrative, statistics, evidence_grade, bayesian_shadow, what_if, education)
- `registry.py` — 230+ analysis metadata entries
- `viz/` — ForgeViz chart rendering (engine, specs, hooks, custom, bayesian_spc)
- `education/` — Educational content per analysis
- `excel_export.py` — XLSX result export

### 1.6 Connected Systems (All produce/consume workbench data)

| System | What it produces | Connection point |
|--------|-----------------|-----------------|
| **A3 Reports** (`/api/a3/`) | Problem background, root cause, countermeasures | Import DSW results, link to Project |
| **RCA Sessions** (`/api/rca/`) | Causal chains, critiques, clustered causes | Link to A3, embed in workbench |
| **FMEA** (`/api/fmea/`) | Failure modes, RPN scores, occurrence from SPC | SPC→FMEA occurrence update, link to hypothesis |
| **VSM** (`vsm/`) | Process maps, cycle time, simulation | Process understanding for analysis context |
| **Whiteboard** (`whiteboard/`) | Brainstorm themes, visual reasoning | Extract hypotheses, link evidence |
| **Hoshin** (`/api/hoshin/`) | Strategic goals, X-matrix, projects | Enterprise context for workbench scope |
| **Triage** (`/api/triage/`) | Cleaned datasets, quality reports | Dataset input for workbench |
| **Forge** (`/api/forge/`) | Synthetic datasets | Generated_dataset artifact |
| **Synara** (`/api/synara/`) | Belief engine, causal links, posteriors | Parallel to KnowledgeGraph (needs sync) |
| **Guide** (`/api/guide/`) | LLM observations, project summaries | Guide observations on workbench |
| **Learn** (`/api/learn/`) | Courses, assessments, competency | Contextual education in results |
| **Core** (`/api/core/`) | Projects, Hypotheses, Evidence, EvidenceLinks | Anchor for all workbench data |
| **Notebooks** (core.Notebook) | Before/after trial pages | Frozen analysis snapshots |
| **Files** (`/api/files/`) | Uploaded documents, images | Attachment artifacts |
| **Chat** (`/api/chat/`) | Conversation history | Contextual analysis assistance |
| **Tools Router** (`tools/`) | Unified tool registry | Tool discovery for workbench |

---

## 2. WHAT THE OLD SYSTEM COULD DO

### 2.1 DSW Endpoints (all at `/api/dsw/`)

| Capability | Endpoint | Status in New |
|-----------|----------|---------------|
| Run analysis (200+ types) | `POST /api/dsw/analysis/` | ✅ `/api/analysis/run/` |
| Upload data | `POST /api/dsw/upload-data/` | ✅ `/api/analysis/upload-data/` |
| Retrieve saved dataset | `POST /api/dsw/retrieve-data/` | ❌ Missing |
| Download data (CSV/XLSX) | `POST /api/dsw/download/` | ❌ Missing |
| Transform data (pivot/merge/derive/filter/bin) | `POST /api/dsw/transform/` | ❌ Missing |
| Triage scan (quality preview) | `POST /api/dsw/triage/scan/` | ❌ Missing |
| Triage clean (handle missing/outliers) | `POST /api/dsw/triage/` | ❌ Missing |
| Explain selection (LLM insight) | `POST /api/dsw/explain-selection/` | ❌ Missing |
| Hypothesis timeline (Bayesian tracking) | `POST /api/dsw/hypothesis-timeline/` | ❌ Missing |
| Generate code (Claude/Qwen) | `POST /api/dsw/generate-code/` | ❌ Missing |
| Execute code (sandboxed) | `POST /api/dsw/execute/` | ❌ (was disabled in old too) |
| Analyst assistant (multi-agent) | `POST /api/dsw/analyst/` | ❌ Missing |
| From-intent (NL → model) | `POST /api/dsw/from-intent/` | ❌ Missing |
| From-data (auto-train) | `POST /api/dsw/from-data/` | ❌ Missing |
| Model CRUD (9 endpoints) | `GET/POST /api/dsw/models/*` | ❌ Missing |
| Model download (pkl/ONNX) | `GET /api/dsw/download/<id>/<type>/` | ❌ Missing |
| Autopilot clean-train | `POST /api/dsw/autopilot/clean-train/` | ❌ Missing |
| Autopilot full-pipeline | `POST /api/dsw/autopilot/full-pipeline/` | ❌ Missing |
| Autopilot augment-train | `POST /api/dsw/autopilot/augment-train/` | ❌ Missing |
| XLSX export | `POST /api/analysis/export/xlsx/` | ✅ Exists |
| ForgePad commands | `POST /api/analysis/forgepad/` | ✅ NEW (not in old) |

### 2.2 SPC Endpoints (`/api/spc/`)

| Capability | Endpoint | Status in New |
|-----------|----------|---------------|
| Control chart (all types) | `POST /api/spc/control-chart/` | ✅ Via forge_spc handlers |
| Capability study | `POST /api/spc/capability/` | ✅ Via forge_spc |
| Recommend chart | `POST /api/spc/recommend/` | ❌ Missing |

### 2.3 DOE Endpoints (`/api/experimenter/`)

| Capability | Endpoint | Status in New |
|-----------|----------|---------------|
| Design experiment | `POST /api/experimenter/design/` | ❌ Not wired to new workbench |
| Power analysis | `POST /api/experimenter/power/` | ❌ Not wired |
| Run DOE analysis | `POST /api/experimenter/analyze/` | ❌ Not wired |

### 2.4 Forecast Endpoints (`/api/forecast/`)

| Capability | Endpoint | Status in New |
|-----------|----------|---------------|
| Time series forecast | `POST /api/forecast/` | ❌ Not wired |

---

## 3. WHAT THE NEW SYSTEM MUST BE

### 3.1 The R Workspace Model

The workbench is an **accumulating workspace**, not a stateless form.

**Objects you accumulate:**
- **Datasets** — Multiple, named, browsable. Upload, derive from transforms, load from triage, generate with Forge.
- **Results** — Each analysis run produces a result artifact. Flip through them like pages. Full standardized output attached.
- **Charts** — Accumulated as you work, not replaced. Each tied to the analysis that created it. ForgeViz rendered.
- **Notes** — Free-text observations, hypotheses, conclusions.
- **Models** — Trained ML models with metrics, versions, predictions.

**Two interaction paths:**
1. **Guided** — Sidebar catalog → configure → run → artifact created
2. **Command** — ForgePad console → type commands → artifacts created

Both paths produce the same artifact objects in the same workspace.

### 3.2 Session Model

- **Client-side by default** — fast, no DB overhead. All state in JS.
- **Save to server** — creates/updates a `Workbench` record with all artifacts. Named, resumable.
- **Load from server** — `GET /api/workbench/<id>/` returns full JSON → hydrate client state.
- **Auto-save** — optional, debounced writes to server every N minutes.

### 3.3 Architecture Principle: SOURCE System

The workbench is a **SOURCE** in the source/switch/sink architecture:
- **Sources** create signals: Workbench, SPC monitors, triage
- **Switches** consume and produce: investigation engine, evidence linking
- **Sinks** consume and organize: A3, reports, dashboards, knowledge graph

The workbench does NOT consume from other systems. No DMAIC templates. No knowledge graph UI. No FMEA import. No guide panel. Those systems pull from workbench artifacts — the workbench doesn't need to know they exist.

### 3.4 Layout

NOT the 4-pane layout. The workbench is a panel-based workspace:

- **Object Browser** (left) — tree/list of all workspace objects: datasets, results, charts. Grouped by type. Click to view.
- **Main Viewport** (center) — shows the selected object. Adapts to type:
  - Dataset → spreadsheet view
  - Result → narrative + statistics + evidence strip + assumptions + charts
  - Chart → ForgeViz rendered with full toolbar
- **Console** (bottom) — ForgePad command line + output. Always visible. History, replay.

### 3.4 Multi-Dataset Support

- Upload creates a `dataset` artifact on the workspace
- Transform commands create derived `cleaned_dataset` artifacts (with `source_artifact_id` pointing to parent)
- Analysis runs reference which dataset they used
- Object browser shows all datasets; click to switch active view
- No "one worksheet" — flip between datasets

### 3.5 Analysis → Artifact Pipeline

When an analysis runs:
1. Create Artifact with `artifact_type` matching analysis category (hypothesis_test, regression, spc_chart, etc.)
2. Store standardized result dict as `content` (plots, statistics, narrative, evidence_grade, bayesian_shadow, what_if, education)
3. Set `source = "forge"` or `"forgepad"`
4. Set `source_artifact_id` to the dataset artifact used
5. Add to workspace object list
6. Auto-select the new artifact in the object browser
7. Main viewport shows the result

### 3.6 What-If Integration

Results with `what_if` data render the `FV.slider()` explorer in the inspector panel. Client-side for regression (intercept + coefficients). Server-recompute for power analysis.

### 3.7 Compose/Trellis for Multi-Chart Results

Results with `_layout` metadata use `FV.compose()` (paired control charts) or `FV.trellis()` (sixpacks, gage R&R). Already implemented in forge_spc handlers.

### 3.8 Output Schema (Pullable by Other Systems)

Every artifact's `content` field follows the standardized schema from `standardize.py`:

```python
{
    "summary": str,                     # Plain text summary
    "plots": list[dict],                # ForgeViz specs [{traces, title, x_axis, y_axis, ...}]
    "statistics": dict,                 # All numeric results
    "narrative": {                      # Structured interpretation
        "verdict": str,
        "body": str,
        "next_steps": str,
        "chart_guidance": str,
    },
    "education": {                      # Educational content
        "title": str,
        "content": str,                 # HTML
    },
    "diagnostics": list[dict],          # Assumption checks, warnings
    "evidence_grade": str,              # "Strong" / "Moderate" / "Weak" / "Inconclusive"
    "bayesian_shadow": {                # Bayesian parallel result
        "bf10": float,
        "bf_label": str,
        "credible_interval": dict,
        "interpretation": str,
    },
    "what_if": {                        # Interactive exploration
        "type": str,                    # "slider" or "sensitivity"
        "parameters": list[dict],
        "endpoint": str,
        "client_model": dict,
    },
    "assumptions": dict,                # Pass/fail per assumption
    "guide_observation": str,           # AI guide summary
    "_analysis_type": str,              # e.g. "stats", "spc"
    "_analysis_id": str,                # e.g. "ttest", "capability"
    "_layout": dict,                    # Chart layout hint (compose/trellis)
}
```

This schema is what other systems pull:
- **Loop** pulls `evidence_grade` + `narrative.verdict` for inbox items
- **Graph** pulls `statistics` + `bayesian_shadow` for edge weight updates
- **Reports** pull `narrative` + `plots` for document generation (via forgedoc)
- **FMEA** pulls `statistics.cpk` for occurrence score updates
- **Notebooks** pull full result for before/after trial pages

---

## 4. BUILD SEQUENCE

### Phase 1: Foundation (Workbench-backed session)
- [ ] Template rewrite: object browser + main viewport + console + inspector
- [ ] On load: create Workbench (or resume from `?workbench=<id>`)
- [ ] Upload → dataset artifact
- [ ] Analysis run → result artifact (correct artifact_type)
- [ ] Object browser renders artifact list by type
- [ ] Click artifact → main viewport shows it
- [ ] ForgePad session maps to Workbench

### Phase 2: Multi-dataset + transforms
- [ ] Multiple dataset artifacts, switch active
- [ ] Transform endpoints (filter, derive, pivot, bin, merge)
- [ ] Derived datasets as artifacts with `source_artifact_id`
- [ ] Triage scan + clean endpoints
- [ ] Dataset inspector (column types, summary stats)

### Phase 3: Missing old capabilities
- [ ] Retrieve saved dataset
- [ ] Data download (CSV/XLSX)
- [ ] Explain selection (LLM)
- [ ] Code generation
- [ ] Hypothesis timeline
- [ ] Chart recommendation

### Phase 4: ML platform
- [ ] From-intent, from-data endpoints
- [ ] Model CRUD (save, load, list, delete, run, optimize)
- [ ] Model artifact type in workspace
- [ ] Autopilot pipelines

### Phase 5: Persistence + Export
- [ ] Save/load workbench (server persistence)
- [ ] ForgeDoc export (PDF/DOCX from workspace artifacts)
- [ ] DOE + Forecast wired as analysis types in the workspace

Note: Other systems (Graph, A3, FMEA, Loop, Reports) pull FROM workbench
artifacts via the standardized schema. That wiring lives in THOSE systems,
not here. The workbench is a source — it doesn't need to know about its consumers.

---

## 5. FILES TO MODIFY/CREATE

| File | Action | Purpose |
|------|--------|---------|
| `templates/demo/analysis_workbench.html` | **REWRITE** | New workspace layout (not 4-pane) |
| `agents_api/analysis_views.py` | **EXTEND** | Add transform, triage, retrieve, download, explain endpoints |
| `agents_api/analysis_urls.py` | **EXTEND** | New URL routes |
| `static/js/workbench-session.js` | **CREATE** | Client-side session manager (datasets, results, charts) |
| `static/js/workbench-objects.js` | **CREATE** | Object browser renderer |
| `static/js/workbench-viewport.js` | **CREATE** | Main viewport (adapts to artifact type) |
| `static/css/sv-workbench.css` | **CREATE** | Workspace styles (extracted from template) |

---

## 6. NON-NEGOTIABLES

1. Every analysis creates an Artifact. No stateless forms.
2. Multiple datasets. Not one worksheet.
3. Object browser shows everything you've done. Not replaced on each run.
4. ForgePad and guided path produce identical artifacts.
5. Output schema is structured and pullable. Not just display HTML.
6. Save/load workspaces. Not throwaway sessions.
7. This uses the Workbench/Artifact/KnowledgeGraph models that already exist.

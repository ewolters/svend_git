# Analysis Workbench — /app/ vs /demo/ Gap Audit

**Last updated:** 2026-05-01
**Canonical:** `templates/analysis_workbench.html` (11,487 lines)
**New build:** `templates/demo/analysis_workbench.html` (3,332 lines)

## Summary

| Status | Count | 
|--------|-------|
| CLOSED | 14 |
| DEFERRED | 1 |
| TOTAL  | 15 |

Progress: **93% complete** (was 0% when gap list identified 2026-04-09).

---

## Gap Status

### CLOSED (10)

| # | Feature | Evidence |
|---|---------|----------|
| 1 | Data triage/cleaning | `triageScan()` → `/api/dsw/triage/scan/`, `triageClean()` → `/api/dsw/triage/` |
| 2 | Data transformation pipeline | `applyTransform()` → `/api/dsw/transform/`, supports log/sqrt/standardize/filter |
| 3 | Multiple data tabs | `.aw-dataset-tabs` container, `switchDataset()`, handles multiple datasets with row counts |
| 5 | XLSX export | `/api/analysis/export/xlsx/` with request headers and body |
| 6 | Evidence linking to projects | `/api/core/evidence/from-analysis/` passing analysis results + project context |
| 7 | RCA workflow entry point | `/api/rca/sessions/create/` creates RCA session from analysis |
| 9 | Session save/restore | localStorage with `aw_session_id`, `saveSession()`, `_loadSessionState()` |
| 11 | AI guide integration | `/api/guide/` call, guide content rendering, `getGuideSuggestion()` |
| 12 | AI analyst chat | `/api/dsw/analyst/` call, analyst messages array, full chat UI |
| 13 | DOE matrix editor | `/api/experimenter/` endpoints: design/analyze/contour/optimize/power, run-card UI |

### CLOSED in this session (4)

| # | Feature | Implementation |
|---|---------|----------------|
| 4 | Data retrieval | `retrieveDataDialog()` → GET `/api/triage/datasets/`, `retrieveData()` → POST `/api/dsw/retrieve-data/` |
| 8 | Saved models | `openModelsDialog()` → GET `/api/dsw/models/`, run/delete actions, modal list UI |
| 10 | Computational notebooks | Toolbar dropdowns for notebook + trial selection, auto-select pending trial, wired into analysis POST payload |
| 15 | Chart interactivity | `attachChartInspect()` — click inspect panels, OOC point details, Nelson rule display, RCA drill-through |

### DEFERRED (1)

| # | Feature | Reason |
|---|---------|--------|
| 14 | Measurement system studies | No `/api/spc/measurement-systems/` endpoint exists in either template. Needs new endpoint design. Gage R&R is available as a standard analysis type — the "builder" UI is a net-new feature, not a port. |

---

## Notes

- **Gap 7 (RCA)** — demo has it, canonical does NOT. Demo is ahead here.
- **Gap 14 (MSA)** — neither template has `/api/spc/measurement-systems/` wired. Gage R&R is available as an analysis type in the dropdown but has no dedicated study-builder UI.
- **Gap 15** — ForgeViz charts render but lack interactive controls. The canonical file uses Plotly which has built-in zoom/pan. ForgeViz needs its own implementation.
- **Demo grew from 1,622 → 3,332 lines** since initial creation — roughly doubled as gaps were closed.

## Remaining work

Only **Gap #14 (MSA builder)** remains. This is a net-new feature — requires:
1. Design `/api/spc/measurement-systems/` endpoint (or extend existing DSW MSA save endpoint)
2. Build study-builder UI: operator/part/trial matrix entry, study type selector
3. Wire to existing gage_rr analysis type for computation

This is low-priority — gage R&R already works via the normal analysis dropdown with manual data entry.

---

## Thread A (backend) status — not covered here

See `project_analysis_workbench_migration.md` memory for:
- Viz port (21 handlers, no forge intercept yet)
- Misc native port (10 groups still legacy-wrapped)
- ML native port (21 handlers, biggest scope)

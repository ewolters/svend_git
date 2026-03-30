# Avionics UI Audit — Template Architecture for Forge Ecosystem

**Date:** 2026-03-29
**Agent:** S2

## The Problem

77 templates. 61 have their own inline `<style>` blocks. Every template invents its own CSS class prefix and widget patterns. dashboard.html uses `stat-card`, safety uses `kpi-card`, hoshin uses `summary-card`, analysis workbench uses `aw-*` — all doing the same thing with different names.

This is 77 bespoke cockpits. What we need is one cockpit with standardized instrument slots.

## The Avionics Analogy (Precisely)

In real avionics:
- **ARINC 600** defines the physical form factor — rack sizes, connector pinouts, cooling
- **ARINC 429/664** defines the data bus — how instruments exchange information
- A flight management computer, a weather radar display, and an engine monitor all fit the same rack, use the same connectors, and speak the same data protocol

For SVEND:
- **The slot** = a standardized template component with fixed mounting points (CSS classes, JS hooks, data contract)
- **The data bus** = the forge package API contract (JSON in, structured result out)
- **The instrument** = a self-contained UI module that renders one forge package's output in one slot

## What Exists Today

### 77 templates fall into 5 patterns (whether they know it or not):

**Pattern 1: Workspace** (editor-style, full-height, panels)
- analysis_workbench.html (11,464 lines — the biggest)
- workbench.html / workbench_new.html
- graph_map.html
- whiteboard.html
- simulator.html

**Pattern 2: Dashboard** (KPI strip + card grid + tables)
- dashboard.html
- safety_app.html
- hoshin.html
- internal_dashboard.html

**Pattern 3: CRUD List** (filterable list + detail view)
- projects.html
- fmea.html
- a3.html
- report.html
- investigations.html
- notebooks.html

**Pattern 4: Tool** (input form → run → output)
- rca.html
- experimenter.html
- forecast.html
- forge.html
- triage.html
- ce_matrix.html
- ishikawa.html

**Pattern 5: Content** (read-only, informational)
- learn.html
- calculators.html
- playbooks (5)
- landing pages
- compliance.html

### The problem: each pattern is reimplemented from scratch in every template.

Dashboard pattern alone has 4 implementations:
- `dashboard.html`: `.stat-card`, `.tool-grid`, `.recent-list`
- `safety_app.html`: `.kpi-card`, `.kpi-row`, `.totals-row`
- `hoshin.html`: `.summary-card`, `.summary-row`, `.sites-grid`
- `internal_dashboard.html`: its own everything

## The Avionics Architecture

### Three layers:

```
Layer 1: Rack (base templates)
    base_app.html       — the airframe (nav, theme, auth, toast, command palette)
    base_workspace.html — full-height workspace rack (sidebar | main | detail)
    base_dashboard.html — dashboard rack (KPI strip + panel grid)
    base_crud.html      — list+detail rack (filterable list | detail panel)

Layer 2: Instruments (reusable components)
    Defined ONCE in base_app.html or a shared include:
    - sv-kpi-strip       — row of metric cells
    - sv-data-table      — sortable, filterable table
    - sv-detail-panel    — entity detail with sections
    - sv-chart-panel     — Plotly chart with export actions
    - sv-activity-feed   — chronological event list
    - sv-modal           — already exists
    - sv-command-palette  — already exists
    - sv-filter-bar      — filter buttons/dropdowns
    - sv-status-badge    — semantic status indicator
    - sv-context-banner  — sticky breadcrumb

Layer 3: Pages (thin, configuration-only)
    Each page template:
    1. Extends a rack (workspace/dashboard/crud)
    2. Declares which instruments to mount
    3. Passes configuration (API endpoint, field mapping, actions)
    4. Adds ZERO custom CSS (or minimal overrides)
```

### What a page template looks like in this model:

```html
{% extends "base_crud.html" %}

{% block page_config %}
<script>
PAGE_CONFIG = {
    title: "Signals",
    api_endpoint: "/api/loop/signals/",
    list_fields: [
        {key: "title", label: "Signal", primary: true},
        {key: "severity", label: "Severity", type: "badge"},
        {key: "source_type", label: "Source"},
        {key: "created_at", label: "Created", type: "timeago"},
    ],
    detail_sections: [
        {key: "description", label: "Description", type: "text"},
        {key: "severity", label: "Severity", type: "badge"},
        {key: "source_type", label: "Source"},
        {key: "triage_state", label: "Status", type: "badge"},
    ],
    filters: [
        {key: "triage_state", label: "Status", options: ["untriaged", "acknowledged", "investigating", "resolved"]},
    ],
    actions: [
        {key: "acknowledge", label: "Acknowledge", condition: "triage_state == 'untriaged'"},
        {key: "investigate", label: "Investigate", condition: "triage_state != 'resolved'", primary: true},
        {key: "dismiss", label: "Dismiss", condition: "triage_state != 'resolved'", danger: true},
    ],
    keyboard: {
        a: "acknowledge",
        i: "investigate",
        d: "dismiss",
    },
};
</script>
{% endblock %}
```

That's the ENTIRE template for the Signals page. No CSS. No custom JS. The `base_crud.html` rack handles rendering, filtering, detail display, keyboard navigation, action buttons, activity feed — all from the config.

### What the racks provide:

**base_workspace.html** (for: analysis workbench, graph map, investigation)
```
┌──────────┬──────────────────────────────────┬─────────────┐
│ Sidebar  │ Main Canvas                      │ Tool Panel  │
│ (nav)    │ (editor/graph/chart area)        │ (props/ctx) │
│          │                                  │             │
│ Slots:   │ Slots:                           │ Slots:      │
│ - nav    │ - toolbar                        │ - properties│
│ - tree   │ - canvas                         │ - actions   │
│ - status │ - status bar                     │ - history   │
└──────────┴──────────────────────────────────┴─────────────┘
```

**base_dashboard.html** (for: main dashboard, safety, hoshin, QMS overview)
```
┌─────────────────────────────────────────────────────────┐
│ KPI Strip (sv-kpi-strip)                                │
├─────────────────┬───────────────────┬───────────────────┤
│ Panel 1         │ Panel 2           │ Panel 3           │
│ (configurable)  │ (configurable)    │ (configurable)    │
├─────────────────┴───────────────────┴───────────────────┤
│ Full-width Panel (table, chart, or activity feed)       │
└─────────────────────────────────────────────────────────┘
```

**base_crud.html** (for: signals, commitments, suppliers, NCRs, CAPAs, etc.)
```
┌──────────────────────────────────────┬──────────────────┐
│ Detail Panel                         │ List Rail        │
│ (context banner)                     │ (filters)        │
│ (sections from config)               │ (items from API) │
│ (actions from config)                │ (keyboard nav)   │
│ (activity feed)                      │ (create form)    │
└──────────────────────────────────────┴──────────────────┘
```

## The Shared Instrument Library

Every instrument is defined ONCE. CSS classes are prefixed `sv-` (SVEND standard).

```css
/* Already exist */
.sv-modal           — modal overlay
.sv-toast            — toast notification

/* Need to be created (consolidate from existing) */
.sv-kpi-strip        — replaces: .kpi-strip, .kpi-row, .summary-row, .stats-row
.sv-kpi-cell         — replaces: .kpi-cell, .kpi-card, .stat-card, .summary-card
.sv-data-table       — replaces: .aw-table + 43 other table implementations
.sv-detail-panel     — replaces: .loop-detail-panel + per-template detail rendering
.sv-list-item        — replaces: .loop-list-item + per-template list items
.sv-filter-bar       — replaces: .signal-filter-btn, .cmt-filter-btn, .sup-filter-btn
.sv-status-badge     — replaces: .loop-tag, .cmt-status, .sup-status, .sec-status, .gm-badge-*
.sv-chart-panel      — replaces: .aw-output-plot + per-template chart containers
.sv-activity-feed    — replaces: per-template timeline implementations
.sv-context-banner   — replaces: .loop-context-banner
.sv-section          — replaces: .detail-section, .sup-section, .gm-detail-section
.sv-section-title    — replaces: .detail-label, .sup-section-title, .gm-detail-section-title
```

## What This Enables

### For the QMS rebuild:
The blank `qms.html` doesn't need to be one monolith. It extends `base_dashboard.html` or `base_workspace.html` and declares its instruments via config. Each QMS section (signals, NCRs, audits, etc.) is a config object, not a template.

### For forge package integration:
When forgespc returns a chart result, `sv-chart-panel` knows how to render it. When forgecal returns a calibration report, `sv-data-table` knows how to render it. The instruments are the standardized connectors.

### For configurability:
TenantConfig says "this org doesn't use FMEA" → the FMEA instrument doesn't mount. No template changes. The config drives which instruments appear.

### For the 77 templates:
Most collapse into config objects on one of the three rack templates. The analysis workbench stays custom (it's genuinely unique). The graph map stays custom (Cytoscape is special). Everything else is a configured rack.

## Migration Path

Don't rewrite 77 templates at once. That's what caused the last disaster.

1. **Build the three rack templates** with the instrument library
2. **Migrate ONE template** to prove the pattern (signals → base_crud.html)
3. **Migrate the QMS rebuild** using the racks (this is the forcing function)
4. **Gradually migrate other templates** as they get touched for other reasons
5. **Old templates keep working** — no forced migration, no big bang

## What I Need From Eric

1. Does this rack/instrument/config model match the avionics vision?
2. Should the instrument library live in base_app.html (one file) or in separate includes ({% include "instruments/kpi_strip.html" %})?
3. Priority: build the racks first, or build the instruments first?

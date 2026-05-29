# Phase 5: Canvas Demo — Dave's Test

**Date:** 2026-05-07
**Status:** Approved design
**Scope:** Demo only (`/app/demo/canvas/`). Staff-only. No user-facing models or CRUD. Validates the canvas container concept before full build.

---

## Goal

Prove the canvas concept works end-to-end: paste data → plugin runs → Job created silently → outputs render (chart + metrics + summary). This is Dave's test from the focus group: "paste data → chart → Cpk in 20 min."

## What Gets Built

Three files:

### 1. POST endpoint — `/api/demo/canvas/run/`

Staff-only. Accepts plugin_name + input_data, calls `run_plugin()`, returns serialized JobOutputs.

**Request:**
```json
{
    "plugin_name": "capability_study",
    "input_data": {
        "data": [1.2, 1.3, 1.1, ...],
        "usl": 56.0,
        "lsl": 44.0,
        "target": 50.0
    },
    "is_scratch": true
}
```

**Response:**
```json
{
    "job_id": "uuid",
    "status": "completed",
    "duration_ms": 142,
    "outputs": [
        {"key": "cpk", "type": "metric", "value": 1.45},
        {"key": "ppk", "type": "metric", "value": 1.38},
        {"key": "sigma_level", "type": "metric", "value": 4.35},
        {"key": "yield_pct", "type": "metric", "value": 99.87},
        {"key": "histogram", "type": "chart", "value": {"title": "...", "traces": [...]}},
        {"key": "summary", "type": "text", "value": {"text": "..."}}
    ]
}
```

Error case: plugin raises → Job marked failed → return `{"job_id": "uuid", "status": "failed", "error": "message"}`.

Implementation: add to existing demo URL wiring in `svend/urls.py`. View function in a new `demo_views.py` or added to existing view helpers. Uses `@require_auth` + `@staff_member_required`.

### 2. Template — `templates/demo/canvas.html`

Extends `base_app.html`. Single page with three zones:

**Input zone:**
- Textarea for data (comma, newline, or tab separated — parse all three)
- USL / LSL / Target number fields
- "Run" button
- Hardcoded HTML — mimics what dynamic form generator would produce from `CapabilityInput.model_json_schema()`

**Output zone:**
- Chart area: `ForgeViz.render()` for histogram/capability charts
- Metric cards: Cpk, Ppk, sigma level, yield — styled number blocks
- Summary text block

**Status indicators:**
- Loading spinner during run
- Error display if plugin fails
- Job ID shown small at bottom (proof of silent audit)

Inline JS:
- Parse textarea → float array
- POST to `/api/demo/canvas/run/`
- On success: render charts via ForgeViz, populate metrics and summary from outputs array
- On error: show error message

No drag-and-drop, no gear icons, no layout configurability, no hotbar.

### 3. Tests

- Endpoint returns 200 with valid outputs for good data
- Endpoint returns error for invalid data (validation from plugin input_schema)
- Job is created in DB after successful run
- JobOutputs exist with correct keys (cpk, ppk, charts, summary)
- is_scratch flag propagates

## What Does NOT Get Built

- Canvas model / CanvasRun model (full build)
- Canvas CRUD endpoints (full build)
- Plugin discovery API (full build)
- Dynamic form generation from JSON Schema (full build — hardcoded form for now)
- Layout configurability / gear icons (full build)
- Composite canvases (full build)
- PCL write-back from outputs (tracked debt)
- Hotbar, keyboard shortcuts (full build)

## Dependencies

All exist and are tested:
- `syn/plugins/runner.py` — `run_plugin()`
- `plugins/capability.py` — `CapabilityStudyPlugin`
- `job/models.py` — Job, JobOutput
- `syn/bus.py` — event emission
- `forgeviz.js` — chart rendering (loaded by base_app.html)

## Success Criteria

Eric pastes a dataset, sets spec limits, hits run, and sees:
1. A histogram with spec limit lines
2. Cpk/Ppk numbers
3. Summary interpretation text
4. Job ID visible (proof the audit trail works silently)

If this works, the container concept is validated and we design the full Canvas model + composite system.

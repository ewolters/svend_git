# Canvas Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a demo page at `/app/demo/canvas/` where you paste data, set spec limits, and see a capability study rendered via ForgeViz — proving the plugin→Job→output→render pipeline works end-to-end.

**Architecture:** One POST endpoint calls `run_plugin("capability_study", ...)` and serializes the JobOutputs back as JSON. One template renders the input form and output zones. No new models — Job/JobOutput handle everything.

**Tech Stack:** Django views, `syn.plugins.runner.run_plugin()`, `job.models.Job/JobOutput`, `ForgeViz.render()` (client-side SVG), `accounts.permissions.require_auth`.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `~/kjerne/demo_views.py` | POST endpoint for `/api/demo/canvas/run/` |
| Create | `~/kjerne/templates/demo/canvas.html` | Demo page: input form + output rendering |
| Modify | `~/kjerne/svend/urls.py:271-272` | Wire new URL routes |
| Create | `~/kjerne/plugins/tests/test_canvas_demo.py` | Endpoint + integration tests |

---

### Task 1: POST Endpoint + Tests

**Files:**
- Create: `~/kjerne/demo_views.py`
- Create: `~/kjerne/plugins/tests/test_canvas_demo.py`
- Modify: `~/kjerne/svend/urls.py:271-272`

- [ ] **Step 1: Write the failing test**

Create `~/kjerne/plugins/tests/test_canvas_demo.py`:

```python
"""Tests for canvas demo endpoint."""

import json
import numpy as np
from django.test import TestCase, Client
from django.contrib.auth import get_user_model

from job.models import Job, JobOutput
from syn.plugins.registry import PluginRegistry, get_registry
from plugins.capability import CapabilityStudyPlugin


User = get_user_model()


class TestCanvasDemoEndpoint(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            email="test@svend.ai", password="testpass123"
        )
        self.client.login(email="test@svend.ai", password="testpass123")
        # Ensure capability plugin is registered
        reg = get_registry()
        if not reg.has("capability_study"):
            reg.register(CapabilityStudyPlugin)

        np.random.seed(42)
        self.good_data = np.random.normal(50, 2, 100).tolist()

    def test_successful_run(self):
        resp = self.client.post(
            "/api/demo/canvas/run/",
            data=json.dumps({
                "plugin_name": "capability_study",
                "input_data": {
                    "data": self.good_data,
                    "usl": 56.0,
                    "lsl": 44.0,
                },
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "completed")
        self.assertIn("job_id", body)
        self.assertIsInstance(body["duration_ms"], int)

        # Check outputs contain expected keys
        output_keys = [o["key"] for o in body["outputs"]]
        self.assertIn("cpk", output_keys)
        self.assertIn("ppk", output_keys)
        self.assertIn("summary", output_keys)

        # Check chart outputs exist
        chart_outputs = [o for o in body["outputs"] if o["type"] == "chart"]
        self.assertGreaterEqual(len(chart_outputs), 1)

    def test_job_created_in_db(self):
        initial_count = Job.objects.count()
        self.client.post(
            "/api/demo/canvas/run/",
            data=json.dumps({
                "plugin_name": "capability_study",
                "input_data": {
                    "data": self.good_data,
                    "usl": 56.0,
                    "lsl": 44.0,
                },
            }),
            content_type="application/json",
        )
        self.assertEqual(Job.objects.count(), initial_count + 1)
        job = Job.objects.order_by("-created_at").first()
        self.assertEqual(job.plugin_name, "capability_study")
        self.assertEqual(job.status, "completed")
        self.assertTrue(job.outputs.filter(output_key="cpk").exists())

    def test_scratch_flag(self):
        self.client.post(
            "/api/demo/canvas/run/",
            data=json.dumps({
                "plugin_name": "capability_study",
                "input_data": {
                    "data": self.good_data,
                    "usl": 56.0,
                    "lsl": 44.0,
                },
                "is_scratch": True,
            }),
            content_type="application/json",
        )
        job = Job.objects.order_by("-created_at").first()
        self.assertTrue(job.is_scratch)

    def test_validation_error(self):
        resp = self.client.post(
            "/api/demo/canvas/run/",
            data=json.dumps({
                "plugin_name": "capability_study",
                "input_data": {"data": [1.0]},  # Too few points
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        body = resp.json()
        self.assertIn("error", body)

    def test_unknown_plugin(self):
        resp = self.client.post(
            "/api/demo/canvas/run/",
            data=json.dumps({
                "plugin_name": "nonexistent",
                "input_data": {},
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)

    def test_requires_auth(self):
        self.client.logout()
        resp = self.client.post(
            "/api/demo/canvas/run/",
            data=json.dumps({
                "plugin_name": "capability_study",
                "input_data": {"data": self.good_data, "usl": 56.0, "lsl": 44.0},
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 401)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_canvas_demo.py -v`

Expected: ImportError or 404s — endpoint doesn't exist yet.

- [ ] **Step 3: Write the endpoint**

Create `~/kjerne/demo_views.py`:

```python
"""Demo canvas endpoint — staff/dev only.

POST /api/demo/canvas/run/
    Runs a plugin through the full lifecycle and returns serialized outputs.
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from accounts.permissions import require_auth
from syn.plugins.runner import run_plugin

logger = logging.getLogger(__name__)


@csrf_exempt
@require_auth
@require_POST
def canvas_run(request):
    """Run a plugin and return Job outputs."""
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    plugin_name = body.get("plugin_name")
    input_data = body.get("input_data", {})
    is_scratch = body.get("is_scratch", False)

    if not plugin_name:
        return JsonResponse({"error": "plugin_name is required"}, status=400)

    try:
        job = run_plugin(
            plugin_name,
            input_data,
            actor=request.user.email,
            tenant_id=getattr(request.user, "tenant_id", None),
            is_scratch=is_scratch,
        )
    except KeyError:
        return JsonResponse(
            {"error": f"Unknown plugin: {plugin_name}"}, status=400
        )
    except Exception as e:
        # ValidationError from pydantic, or plugin execution failure
        # run_plugin marks Job as failed internally
        return JsonResponse({"error": str(e)}, status=400)

    # Serialize outputs
    outputs = []
    for out in job.outputs.all().order_by("created_at"):
        outputs.append({
            "key": out.output_key,
            "type": out.output_type,
            "value": out.value_numeric if out.output_type == "metric" else out.value_json,
            "provenance": out.provenance,
            "measure_slug": out.measure_slug,
        })

    return JsonResponse({
        "job_id": str(job.id),
        "status": job.status,
        "duration_ms": job.duration_ms,
        "outputs": outputs,
    })
```

- [ ] **Step 4: Wire the URL**

In `~/kjerne/svend/urls.py`, add after the existing demo routes (around line 272):

```python
# Canvas demo
path("app/demo/canvas/", _app_view("demo/canvas.html"), name="demo_canvas"),
path("api/demo/canvas/run/", demo_canvas_run, name="demo_canvas_run"),
```

Add the import at the top of the file with the other view imports:

```python
from demo_views import canvas_run as demo_canvas_run
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_canvas_demo.py -v`

Expected: All 6 tests pass.

- [ ] **Step 6: Commit**

```bash
cd ~/kjerne && git add demo_views.py plugins/tests/test_canvas_demo.py svend/urls.py
git commit -m "feat: canvas demo endpoint — POST /api/demo/canvas/run/

Thin wrapper around run_plugin() that returns serialized JobOutputs.
Staff-only demo surface for validating canvas concept."
```

---

### Task 2: Template

**Files:**
- Create: `~/kjerne/templates/demo/canvas.html`

- [ ] **Step 1: Create the template**

Create `~/kjerne/templates/demo/canvas.html`:

```html
{% extends "base_app.html" %}

{% block title %}Canvas Demo — SVEND{% endblock %}

{% block extra_head %}
<style>
    .canvas-demo {
        display: grid;
        grid-template-columns: 360px 1fr;
        gap: 24px;
        padding: 24px;
        height: calc(100vh - 80px);
        overflow: hidden;
    }

    /* ── Input Zone ── */
    .canvas-input {
        display: flex;
        flex-direction: column;
        gap: 16px;
        overflow-y: auto;
    }
    .canvas-input h2 {
        margin: 0;
        font-size: 15px;
        color: var(--text-primary);
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .canvas-input label {
        display: block;
        font-size: 12px;
        color: var(--text-dim);
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .canvas-input textarea {
        width: 100%;
        height: 200px;
        background: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 10px;
        font-family: monospace;
        font-size: 13px;
        resize: vertical;
    }
    .canvas-input textarea:focus {
        outline: none;
        border-color: var(--accent-primary);
    }
    .spec-row {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 10px;
    }
    .spec-row input {
        width: 100%;
        background: var(--bg-secondary);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 14px;
    }
    .spec-row input:focus {
        outline: none;
        border-color: var(--accent-primary);
    }
    .btn-run {
        padding: 10px 20px;
        background: var(--accent-primary);
        color: #fff;
        border: none;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        letter-spacing: 0.3px;
    }
    .btn-run:hover { opacity: 0.9; }
    .btn-run:disabled {
        opacity: 0.4;
        cursor: not-allowed;
    }

    /* ── Output Zone ── */
    .canvas-output {
        display: flex;
        flex-direction: column;
        gap: 20px;
        overflow-y: auto;
    }
    .output-empty {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: var(--text-dim);
        font-size: 14px;
    }

    /* Metric cards */
    .metric-cards {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
    }
    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .metric-card .metric-label {
        font-size: 11px;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-card .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
        font-variant-numeric: tabular-nums;
    }

    /* Charts */
    .chart-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        min-height: 300px;
    }

    /* Summary */
    .summary-block {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        font-size: 14px;
        line-height: 1.6;
        color: var(--text-primary);
        white-space: pre-wrap;
    }

    /* Job ID footer */
    .job-footer {
        font-size: 11px;
        color: var(--text-dim);
        padding-top: 8px;
        border-top: 1px solid var(--border);
    }

    /* Status */
    .status-spinner {
        display: none;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        color: var(--text-dim);
    }
    .status-spinner.visible { display: flex; }
    .status-spinner::before {
        content: "";
        width: 14px; height: 14px;
        border: 2px solid var(--border);
        border-top-color: var(--accent-primary);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .error-msg {
        display: none;
        background: rgba(217, 74, 74, 0.1);
        border: 1px solid rgba(217, 74, 74, 0.3);
        border-radius: 6px;
        padding: 10px 14px;
        color: #d94a4a;
        font-size: 13px;
    }
    .error-msg.visible { display: block; }
</style>
{% endblock %}

{% block content %}
<div class="canvas-demo">

    <!-- Input Zone -->
    <div class="canvas-input">
        <h2>Capability Study</h2>

        <div>
            <label for="data-input">Data (paste values — comma, tab, or newline separated)</label>
            <textarea id="data-input" placeholder="1.23, 1.45, 1.31, 1.28, ..."></textarea>
        </div>

        <div class="spec-row">
            <div>
                <label for="lsl-input">LSL</label>
                <input type="number" id="lsl-input" step="any" placeholder="Lower">
            </div>
            <div>
                <label for="target-input">Target</label>
                <input type="number" id="target-input" step="any" placeholder="Target">
            </div>
            <div>
                <label for="usl-input">USL</label>
                <input type="number" id="usl-input" step="any" placeholder="Upper">
            </div>
        </div>

        <button class="btn-run" id="btn-run" onclick="runCanvas()">Run</button>
        <div class="status-spinner" id="spinner">Running analysis...</div>
        <div class="error-msg" id="error-msg"></div>
    </div>

    <!-- Output Zone -->
    <div class="canvas-output" id="output-zone">
        <div class="output-empty" id="output-empty">
            Paste data and set spec limits, then hit Run.
        </div>
    </div>

</div>
{% endblock %}

{% block scripts %}
<script>
function parseData(raw) {
    // Split on commas, tabs, newlines, or whitespace — filter empties
    return raw.split(/[,\t\n\r\s]+/)
        .map(function(s) { return s.trim(); })
        .filter(function(s) { return s.length > 0; })
        .map(Number)
        .filter(function(n) { return !isNaN(n); });
}

function optFloat(id) {
    var v = document.getElementById(id).value.trim();
    return v === '' ? null : parseFloat(v);
}

function escapeText(str) {
    var d = document.createElement('div');
    d.textContent = str;
    return d.textContent;
}

async function runCanvas() {
    var btn = document.getElementById('btn-run');
    var spinner = document.getElementById('spinner');
    var errorEl = document.getElementById('error-msg');
    var outputZone = document.getElementById('output-zone');

    // Parse input
    var raw = document.getElementById('data-input').value;
    var data = parseData(raw);
    if (data.length < 2) {
        errorEl.textContent = 'Need at least 2 numeric data points.';
        errorEl.classList.add('visible');
        return;
    }

    var lsl = optFloat('lsl-input');
    var usl = optFloat('usl-input');
    var target = optFloat('target-input');

    // Validate USL > LSL
    if (usl !== null && lsl !== null && usl <= lsl) {
        errorEl.textContent = 'USL must be greater than LSL.';
        errorEl.classList.add('visible');
        return;
    }

    // UI: loading state
    btn.disabled = true;
    spinner.classList.add('visible');
    errorEl.classList.remove('visible');

    var body = {
        plugin_name: 'capability_study',
        input_data: { data: data },
        is_scratch: true,
    };
    if (lsl !== null) body.input_data.lsl = lsl;
    if (usl !== null) body.input_data.usl = usl;
    if (target !== null) body.input_data.target = target;

    try {
        var resp = await fetch('/api/demo/canvas/run/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]')?.value
                    || document.cookie.match(/csrftoken=([^;]+)/)?.[1] || '',
            },
            body: JSON.stringify(body),
        });

        var result = await resp.json();

        if (!resp.ok || result.status === 'failed') {
            throw new Error(result.error || 'Analysis failed');
        }

        renderOutputs(result);
    } catch (e) {
        errorEl.textContent = e.message;
        errorEl.classList.add('visible');
    } finally {
        btn.disabled = false;
        spinner.classList.remove('visible');
    }
}

function renderOutputs(result) {
    var zone = document.getElementById('output-zone');
    // Clear previous outputs safely
    while (zone.firstChild) { zone.removeChild(zone.firstChild); }

    var outputs = result.outputs || [];
    var LABELS = {
        cp: 'Cp', cpk: 'Cpk', pp: 'Pp', ppk: 'Ppk',
        sigma_level: 'Sigma', yield_pct: 'Yield %', ppm_total: 'PPM',
    };

    // 1. Metric cards
    var metrics = outputs.filter(function(o) { return o.type === 'metric'; });
    if (metrics.length) {
        var grid = document.createElement('div');
        grid.className = 'metric-cards';
        metrics.forEach(function(m) {
            var card = document.createElement('div');
            card.className = 'metric-card';

            var labelEl = document.createElement('div');
            labelEl.className = 'metric-label';
            labelEl.textContent = LABELS[m.key] || m.key;
            card.appendChild(labelEl);

            var valEl = document.createElement('div');
            valEl.className = 'metric-value';
            var formatted = typeof m.value === 'number'
                ? (m.key === 'ppm_total' ? m.value.toFixed(0) : m.value.toFixed(3))
                : String(m.value);
            valEl.textContent = formatted;
            card.appendChild(valEl);

            grid.appendChild(card);
        });
        zone.appendChild(grid);
    }

    // 2. Charts via ForgeViz
    var charts = outputs.filter(function(o) { return o.type === 'chart'; });
    charts.forEach(function(c) {
        var container = document.createElement('div');
        container.className = 'chart-container';
        zone.appendChild(container);

        if (typeof ForgeViz !== 'undefined' && ForgeViz.render) {
            try {
                ForgeViz.renderResponsive(container, c.value, { theme: 'svend_dark' });
            } catch (e) {
                container.textContent = 'Chart render error: ' + e.message;
            }
        } else {
            container.textContent = 'ForgeViz not loaded';
        }
    });

    // 3. Summary text
    var texts = outputs.filter(function(o) { return o.type === 'text'; });
    texts.forEach(function(t) {
        var block = document.createElement('div');
        block.className = 'summary-block';
        var content = typeof t.value === 'object' ? (t.value.text || JSON.stringify(t.value)) : t.value;
        block.textContent = content;
        zone.appendChild(block);
    });

    // 4. Job ID footer (proof of silent audit)
    if (result.job_id) {
        var footer = document.createElement('div');
        footer.className = 'job-footer';
        footer.textContent = 'Job ' + result.job_id + ' \u00b7 ' + result.duration_ms + 'ms';
        zone.appendChild(footer);
    }
}
</script>
{% endblock %}
```

- [ ] **Step 2: Verify the page loads**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 manage.py runserver 0.0.0.0:8000`

Navigate to `/app/demo/canvas/`. Confirm the page renders with the input form and empty output zone.

- [ ] **Step 3: Manual smoke test**

Paste sample data, set USL=56 LSL=44, hit Run. Confirm:
- Metric cards appear (Cpk, Ppk, etc.)
- ForgeViz chart renders
- Summary text shows
- Job ID visible at bottom
- No console errors

- [ ] **Step 4: Commit**

```bash
cd ~/kjerne && git add templates/demo/canvas.html
git commit -m "feat: canvas demo template — paste data, get capability study

ForgeViz charts, metric cards, summary text. Hardcoded input form
for capability study. Validates canvas container concept."
```

---

### Task 3: Final Validation

- [ ] **Step 1: Run full test suite**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/ -v`

Expected: All existing 35 tests + 6 new tests pass (41 total).

- [ ] **Step 2: Run job tests to make sure nothing broke**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest job/tests/ -v`

Expected: All pass.

- [ ] **Step 3: Visual verification**

Open `/app/demo/canvas/` in browser. Paste this data:

```
49.8, 50.2, 51.1, 48.9, 50.5, 49.3, 50.8, 49.1, 50.0, 51.3,
48.7, 50.6, 49.5, 50.4, 49.9, 51.0, 48.8, 50.3, 49.7, 50.1
```

Set LSL=44, USL=56. Run. Confirm:
- Cpk shows ~3.0+ (data is tight relative to specs)
- Histogram renders with spec limit lines
- Summary text explains the result
- Job ID at footer

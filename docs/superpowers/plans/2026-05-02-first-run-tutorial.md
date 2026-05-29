# First-Run Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get new users from registration to their first Bayesian capability analysis in under 2 minutes, then collect survey data and drive email verification.

**Architecture:** The analysis workbench template (`analysis_workbench.html`) serves double duty — a `tutorial=true` context flag pre-loads sample data and initializes `spotlight.js`. A reusable spotlight engine (JSON-defined step sequences) drives the walkthrough. Two new User fields track tutorial completion/skip. Post-tutorial, an inline 1-step survey collects industry/role/goal, then a verify-email CTA gates further analysis runs.

**Tech Stack:** Django views, vanilla JS (spotlight.js), existing sv-* widget patterns, Django migrations

**Spec:** `docs/superpowers/specs/2026-05-02-first-run-tutorial-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `accounts/models.py` | Modify | Add `tutorial_completed_at`, `tutorial_skipped_at` fields to User |
| `accounts/migrations/NNNN_add_tutorial_fields.py` | Create | Migration for new fields |
| `api/views.py` | Modify | Add `tutorial_complete` endpoint, modify verification gate |
| `api/urls.py` | Modify | Add tutorial endpoint URL |
| `svend/urls.py` | Modify | Add `/app/tutorial/` route, custom views for dashboard + analysis |
| `static/js/spotlight.js` | Create | Reusable spotlight/tooltip engine |
| `static/css/svend-widgets.css` | Modify | Add spotlight CSS |
| `templates/register.html` | Modify | Change default redirect from `/app/` to `/app/tutorial/` |
| `templates/analysis_workbench.html` | Modify | Add tutorial mode initialization block + verify gate |
| `templates/onboarding.html` | Modify | Trim to 1-step survey |
| `templates/dashboard.html` | Modify | Add tutorial banner for returning users |
| `tests/test_tutorial.py` | Create | All tutorial tests |

---

### Task 1: User Model — Tutorial Fields + Migration

**Files:**
- Modify: `accounts/models.py:292` (after `onboarding_completed_at`)
- Create: `accounts/migrations/NNNN_add_tutorial_fields.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tutorial.py`:

```python
"""Tests for first-run tutorial flow."""
import json
from django.test import TestCase, override_settings
from django.utils import timezone
from accounts.models import User


class TutorialFieldsTest(TestCase):
    """Test tutorial tracking fields on User model."""

    def setUp(self):
        self.user = User.objects.create_user(
            username="tutorialtest",
            email="tutorial@test.com",
            password="testpass123",
        )

    def test_tutorial_fields_default_null(self):
        """New users have no tutorial timestamps."""
        self.assertIsNone(self.user.tutorial_completed_at)
        self.assertIsNone(self.user.tutorial_skipped_at)

    def test_set_tutorial_completed(self):
        """Can mark tutorial as completed."""
        now = timezone.now()
        self.user.tutorial_completed_at = now
        self.user.save(update_fields=["tutorial_completed_at"])
        self.user.refresh_from_db()
        self.assertEqual(self.user.tutorial_completed_at, now)

    def test_set_tutorial_skipped(self):
        """Can mark tutorial as skipped."""
        now = timezone.now()
        self.user.tutorial_skipped_at = now
        self.user.save(update_fields=["tutorial_skipped_at"])
        self.user.refresh_from_db()
        self.assertEqual(self.user.tutorial_skipped_at, now)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a && source /etc/svend/env && set +a && cd ~/kjerne && python3 manage.py test tests.test_tutorial.TutorialFieldsTest -v2`
Expected: FAIL — `tutorial_completed_at` not a field on User

- [ ] **Step 3: Add fields to User model**

In `accounts/models.py`, after `onboarding_completed_at` (line 292), add:

```python
    tutorial_completed_at = models.DateTimeField(null=True, blank=True)
    tutorial_skipped_at = models.DateTimeField(null=True, blank=True)
```

- [ ] **Step 4: Generate and run migration**

Run:
```bash
set -a && source /etc/svend/env && set +a && cd ~/kjerne
python3 manage.py makemigrations accounts -n add_tutorial_fields
python3 manage.py migrate
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 manage.py test tests.test_tutorial.TutorialFieldsTest -v2`
Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add accounts/models.py accounts/migrations/*add_tutorial_fields* tests/test_tutorial.py
git commit -m "feat: add tutorial_completed_at and tutorial_skipped_at to User model"
```

---

### Task 2: Tutorial Complete API Endpoint

**Files:**
- Modify: `api/views.py` (add `tutorial_complete` view after `onboarding_complete` ~line 1835)
- Modify: `api/urls.py` (add URL pattern after onboarding routes ~line 50)
- Test: `tests/test_tutorial.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tutorial.py`:

```python
from rest_framework.test import APIClient


class TutorialCompleteEndpointTest(TestCase):
    """Test POST /api/auth/tutorial/complete/ endpoint."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="tutorialapi",
            email="tutorialapi@test.com",
            password="testpass123",
        )
        self.client.force_authenticate(user=self.user)

    def test_complete_tutorial(self):
        """POST with action=complete sets tutorial_completed_at."""
        resp = self.client.post(
            "/api/auth/tutorial/complete/",
            {"action": "complete"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.user.refresh_from_db()
        self.assertIsNotNone(self.user.tutorial_completed_at)
        self.assertIsNone(self.user.tutorial_skipped_at)

    def test_skip_tutorial(self):
        """POST with action=skip sets tutorial_skipped_at."""
        resp = self.client.post(
            "/api/auth/tutorial/complete/",
            {"action": "skip"},
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.user.refresh_from_db()
        self.assertIsNone(self.user.tutorial_completed_at)
        self.assertIsNotNone(self.user.tutorial_skipped_at)

    def test_requires_auth(self):
        """Unauthenticated requests get 401/403."""
        anon = APIClient()
        resp = anon.post(
            "/api/auth/tutorial/complete/",
            {"action": "complete"},
            format="json",
        )
        self.assertIn(resp.status_code, [401, 403])

    def test_invalid_action(self):
        """Invalid action returns 400."""
        resp = self.client.post(
            "/api/auth/tutorial/complete/",
            {"action": "invalid"},
            format="json",
        )
        self.assertEqual(resp.status_code, 400)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 manage.py test tests.test_tutorial.TutorialCompleteEndpointTest -v2`
Expected: FAIL — 404 (URL not found)

- [ ] **Step 3: Implement the endpoint**

In `api/views.py`, after the `onboarding_complete` function (~line 1835), add:

```python
@api_view(["POST"])
@permission_classes([IsAuthenticated])
def tutorial_complete(request):
    """Mark the first-run tutorial as completed or skipped."""
    action = request.data.get("action")
    if action not in ("complete", "skip"):
        return Response(
            {"error": "action must be 'complete' or 'skip'"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    user = request.user
    if action == "complete":
        user.tutorial_completed_at = timezone.now()
        user.save(update_fields=["tutorial_completed_at"])
    else:
        user.tutorial_skipped_at = timezone.now()
        user.save(update_fields=["tutorial_skipped_at"])
    return Response({"status": action + "d"})
```

- [ ] **Step 4: Add URL pattern**

In `api/urls.py`, after the onboarding routes (~line 50), add:

```python
    path("auth/tutorial/complete/", views.tutorial_complete, name="tutorial_complete"),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 manage.py test tests.test_tutorial.TutorialCompleteEndpointTest -v2`
Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add api/views.py api/urls.py tests/test_tutorial.py
git commit -m "feat: add POST /api/auth/tutorial/complete/ endpoint"
```

---

### Task 3: Spotlight Engine (static/js/spotlight.js)

**Files:**
- Create: `static/js/spotlight.js`
- Modify: `static/css/svend-widgets.css` (add spotlight CSS)

- [ ] **Step 1: Create spotlight.js**

Create `static/js/spotlight.js`. All DOM construction uses `createElement` + `textContent` (no innerHTML):

```javascript
/**
 * Spotlight — reusable guided-tour engine.
 *
 * Usage:
 *   const tour = new Spotlight([
 *     { selector: '#data-table', title: 'Your Data', text: '...', position: 'bottom' },
 *     { selector: '#run-btn', title: 'Run', text: '...', position: 'right', action: 'click' },
 *   ], { onComplete, onSkip });
 *   tour.start();
 */
(function () {
  'use strict';

  function el(tag, className, textContent) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (textContent) node.textContent = textContent;
    return node;
  }

  class Spotlight {
    constructor(steps, options) {
      options = options || {};
      this.steps = steps;
      this.currentStep = 0;
      this.onComplete = options.onComplete || function () {};
      this.onSkip = options.onSkip || function () {};
      this.overlay = null;
      this.tooltip = null;
      this._actionHandler = null;
    }

    start() {
      this._createOverlay();
      this._createTooltip();
      this._showStep(0);
    }

    _createOverlay() {
      this.overlay = document.createElement('div');
      this.overlay.className = 'spotlight-overlay';

      var svgNS = 'http://www.w3.org/2000/svg';
      var svg = document.createElementNS(svgNS, 'svg');
      svg.setAttribute('width', '100%');
      svg.setAttribute('height', '100%');

      var defs = document.createElementNS(svgNS, 'defs');
      var mask = document.createElementNS(svgNS, 'mask');
      mask.setAttribute('id', 'spotlight-mask');

      var maskWhite = document.createElementNS(svgNS, 'rect');
      maskWhite.setAttribute('width', '100%');
      maskWhite.setAttribute('height', '100%');
      maskWhite.setAttribute('fill', 'white');

      var cutout = document.createElementNS(svgNS, 'rect');
      cutout.setAttribute('id', 'spotlight-cutout');
      cutout.setAttribute('rx', '8');
      cutout.setAttribute('ry', '8');
      cutout.setAttribute('fill', 'black');

      mask.appendChild(maskWhite);
      mask.appendChild(cutout);
      defs.appendChild(mask);
      svg.appendChild(defs);

      var bg = document.createElementNS(svgNS, 'rect');
      bg.setAttribute('width', '100%');
      bg.setAttribute('height', '100%');
      bg.setAttribute('fill', 'rgba(0,0,0,0.6)');
      bg.setAttribute('mask', 'url(#spotlight-mask)');
      svg.appendChild(bg);

      this.overlay.appendChild(svg);
      document.body.appendChild(this.overlay);
    }

    _createTooltip() {
      var tt = el('div', 'spotlight-tooltip');
      var progress = el('div', 'spotlight-progress');
      var title = el('div', 'spotlight-title');
      var text = el('div', 'spotlight-text');
      var actions = el('div', 'spotlight-actions');
      var skipBtn = el('button', 'spotlight-skip', 'Skip tutorial');
      var nav = el('div', 'spotlight-nav');
      var backBtn = el('button', 'spotlight-back', 'Back');
      var nextBtn = el('button', 'spotlight-next', 'Next');

      nav.appendChild(backBtn);
      nav.appendChild(nextBtn);
      actions.appendChild(skipBtn);
      actions.appendChild(nav);

      tt.appendChild(progress);
      tt.appendChild(title);
      tt.appendChild(text);
      tt.appendChild(actions);

      var self = this;
      skipBtn.addEventListener('click', function () { self._skip(); });
      backBtn.addEventListener('click', function () { self._prev(); });
      nextBtn.addEventListener('click', function () { self._next(); });

      this.tooltip = tt;
      document.body.appendChild(tt);
    }

    _showStep(index) {
      this._clearActionHandler();
      this.currentStep = index;
      var step = this.steps[index];
      var target = document.querySelector(step.selector);

      if (!target) {
        console.warn('Spotlight: element not found for selector', step.selector);
        this._next();
        return;
      }

      // Position cutout over target element
      var rect = target.getBoundingClientRect();
      var pad = 8;
      var cutout = document.getElementById('spotlight-cutout');
      cutout.setAttribute('x', rect.left - pad + window.scrollX);
      cutout.setAttribute('y', rect.top - pad + window.scrollY);
      cutout.setAttribute('width', rect.width + pad * 2);
      cutout.setAttribute('height', rect.height + pad * 2);

      // Update tooltip content via textContent (safe, no XSS)
      this.tooltip.querySelector('.spotlight-progress').textContent =
        'Step ' + (index + 1) + ' of ' + this.steps.length;
      this.tooltip.querySelector('.spotlight-title').textContent = step.title;
      this.tooltip.querySelector('.spotlight-text').textContent = step.text;

      // Navigation visibility
      this.tooltip.querySelector('.spotlight-back').style.display = index === 0 ? 'none' : '';
      var nextBtn = this.tooltip.querySelector('.spotlight-next');

      if (step.action === 'click') {
        nextBtn.style.display = 'none';
        var self = this;
        this._actionHandler = function () {
          var waitFor = step.waitForSelector;
          if (waitFor) {
            self._waitForElement(waitFor, function () { self._next(); });
          } else {
            setTimeout(function () { self._next(); }, 500);
          }
        };
        target.addEventListener('click', this._actionHandler, { once: true });
      } else {
        nextBtn.style.display = '';
        nextBtn.textContent = index === this.steps.length - 1 ? 'Finish' : 'Next';
      }

      // Position tooltip relative to target
      this._positionTooltip(rect, step.position || 'bottom');
      target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    _positionTooltip(targetRect, position) {
      var tt = this.tooltip;
      var gap = 16;
      tt.style.top = '';
      tt.style.left = '';
      tt.style.transform = '';

      switch (position) {
        case 'bottom':
          tt.style.top = (targetRect.bottom + gap + window.scrollY) + 'px';
          tt.style.left = Math.max(16, targetRect.left + window.scrollX) + 'px';
          break;
        case 'top':
          tt.style.top = (targetRect.top - gap + window.scrollY) + 'px';
          tt.style.left = Math.max(16, targetRect.left + window.scrollX) + 'px';
          tt.style.transform = 'translateY(-100%)';
          break;
        case 'right':
          tt.style.top = (targetRect.top + window.scrollY) + 'px';
          tt.style.left = (targetRect.right + gap + window.scrollX) + 'px';
          break;
        case 'left':
          tt.style.top = (targetRect.top + window.scrollY) + 'px';
          tt.style.left = (targetRect.left - gap + window.scrollX) + 'px';
          tt.style.transform = 'translateX(-100%)';
          break;
      }
    }

    _waitForElement(selector, callback, timeout) {
      timeout = timeout || 10000;
      var start = Date.now();
      var check = function () {
        if (document.querySelector(selector)) {
          callback();
        } else if (Date.now() - start < timeout) {
          requestAnimationFrame(check);
        } else {
          console.warn('Spotlight: timed out waiting for', selector);
          callback();
        }
      };
      check();
    }

    _next() {
      if (this.currentStep < this.steps.length - 1) {
        this._showStep(this.currentStep + 1);
      } else {
        this._finish();
      }
    }

    _prev() {
      if (this.currentStep > 0) {
        this._showStep(this.currentStep - 1);
      }
    }

    _skip() {
      this._cleanup();
      this.onSkip();
    }

    _finish() {
      this._cleanup();
      this.onComplete();
    }

    _clearActionHandler() {
      if (this._actionHandler && this.currentStep < this.steps.length) {
        var step = this.steps[this.currentStep];
        var target = document.querySelector(step.selector);
        if (target) target.removeEventListener('click', this._actionHandler);
      }
      this._actionHandler = null;
    }

    _cleanup() {
      this._clearActionHandler();
      if (this.overlay) this.overlay.remove();
      if (this.tooltip) this.tooltip.remove();
    }
  }

  window.Spotlight = Spotlight;
})();
```

- [ ] **Step 2: Add spotlight CSS**

Append to `static/css/svend-widgets.css`:

```css
/* Spotlight tutorial overlay */
.spotlight-overlay {
  position: fixed;
  inset: 0;
  z-index: 9998;
  pointer-events: none;
}
.spotlight-overlay svg {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}
.spotlight-tooltip {
  position: absolute;
  z-index: 9999;
  background: var(--sv-surface, #1e1e2e);
  border: 1px solid var(--sv-border, #333);
  border-radius: 8px;
  padding: 16px 20px;
  max-width: 360px;
  color: var(--sv-text, #e0e0e0);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.spotlight-progress {
  font-size: 0.75rem;
  color: var(--sv-text-muted, #888);
  margin-bottom: 4px;
}
.spotlight-title {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 8px;
}
.spotlight-text {
  font-size: 0.875rem;
  line-height: 1.5;
  margin-bottom: 16px;
}
.spotlight-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.spotlight-skip {
  background: none;
  border: none;
  color: var(--sv-text-muted, #888);
  cursor: pointer;
  font-size: 0.8rem;
  padding: 0;
}
.spotlight-skip:hover {
  color: var(--sv-text, #e0e0e0);
}
.spotlight-nav {
  display: flex;
  gap: 8px;
}
.spotlight-back,
.spotlight-next {
  padding: 6px 16px;
  border-radius: 4px;
  border: 1px solid var(--sv-border, #333);
  background: var(--sv-surface, #1e1e2e);
  color: var(--sv-text, #e0e0e0);
  cursor: pointer;
  font-size: 0.85rem;
}
.spotlight-next {
  background: var(--sv-primary, #4a7dff);
  border-color: var(--sv-primary, #4a7dff);
  color: #fff;
}
```

- [ ] **Step 3: Run collectstatic**

Run:
```bash
cd ~/kjerne && python3 manage.py collectstatic --noinput 2>&1 | tail -5
```
Expected: files collected successfully

- [ ] **Step 4: Commit**

```bash
git add static/js/spotlight.js static/css/svend-widgets.css
git commit -m "feat: add reusable spotlight/tooltip tutorial engine"
```

---

### Task 4: Tutorial View + URL Route

**Files:**
- Modify: `svend/urls.py:208` (add `/app/tutorial/` route)
- Test: `tests/test_tutorial.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tutorial.py`:

```python
class TutorialViewTest(TestCase):
    """Test /app/tutorial/ route."""

    def setUp(self):
        self.user = User.objects.create_user(
            username="tutorialview",
            email="tutorialview@test.com",
            password="testpass123",
        )
        self.client.login(username="tutorialview", password="testpass123")

    def test_tutorial_page_loads(self):
        """GET /app/tutorial/ returns 200 for authenticated users."""
        resp = self.client.get("/app/tutorial/")
        self.assertEqual(resp.status_code, 200)

    def test_tutorial_page_has_tutorial_flag(self):
        """Template context includes tutorial=True."""
        resp = self.client.get("/app/tutorial/")
        self.assertTrue(resp.context.get("tutorial"))

    def test_tutorial_page_redirects_anon(self):
        """Unauthenticated users redirect to login."""
        self.client.logout()
        resp = self.client.get("/app/tutorial/")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login/", resp.url)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 manage.py test tests.test_tutorial.TutorialViewTest -v2`
Expected: FAIL — 404

- [ ] **Step 3: Add tutorial view and URL**

In `svend/urls.py`, add after the `_app_view` helper (~line 14). Add `render` import if not present:

```python
from django.shortcuts import render

@login_required
def _tutorial_view(request):
    """Serve analysis workbench in tutorial mode."""
    return render(request, "analysis_workbench.html", {
        "tutorial": True,
        "show_verify_gate": False,
    })
```

Add URL pattern before the `/app/analysis/` route (~line 208):

```python
    path("app/tutorial/", _tutorial_view, name="tutorial"),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 manage.py test tests.test_tutorial.TutorialViewTest -v2`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add svend/urls.py tests/test_tutorial.py
git commit -m "feat: add /app/tutorial/ route serving workbench in tutorial mode"
```

---

### Task 5: Tutorial Mode in Analysis Workbench Template

**Files:**
- Modify: `templates/analysis_workbench.html` (add tutorial init block)
- Test: Manual browser verification

- [ ] **Step 1: Read the analysis workbench template**

Read `templates/analysis_workbench.html` to find:
1. Where `<script>` tags are included (bottom of template)
2. The selectors for: data table/input, LSL/USL fields, run button, results area, narrative section
3. How the existing config form JS (`aw-config-forms.js`) initializes

Document the exact selectors — these go into the tutorial step definitions. Update every selector in step 2 below to match what you find.

- [ ] **Step 2: Add tutorial initialization block**

At the bottom of `templates/analysis_workbench.html`, before the closing `{% endblock %}`, add a `{% if tutorial %}` block. This block:

1. Loads `spotlight.js` via `{% static 'js/spotlight.js' %}`
2. Defines `TUTORIAL_DATA` — 50 shaft diameter measurements, LSL=24.95, USL=25.05
3. Defines `preloadTutorialData()` — populates the workbench form fields with sample data
4. Defines `TUTORIAL_STEPS` — 7-step JSON array (selectors MUST match actual template elements)
5. Defines `onTutorialComplete()` — POSTs action=complete, redirects to `/app/onboarding/`
6. Defines `onTutorialSkip()` — POSTs action=skip, redirects to `/app/onboarding/`
7. On DOMContentLoaded: pre-loads data, then starts spotlight tour

Sample data to embed:

```javascript
var TUTORIAL_DATA = {
  name: "Shaft Diameter — Tutorial Sample",
  values: [
    25.02, 25.01, 24.98, 25.03, 25.00, 24.99, 25.01, 25.04, 24.97, 25.02,
    25.00, 24.98, 25.03, 25.01, 24.99, 25.02, 25.00, 25.01, 24.97, 25.03,
    25.02, 24.99, 25.00, 25.01, 24.98, 25.04, 25.01, 25.00, 24.99, 25.02,
    25.03, 25.01, 24.98, 25.00, 25.02, 24.99, 25.01, 25.03, 25.00, 24.97,
    25.02, 25.01, 24.99, 25.00, 25.03, 24.98, 25.01, 25.02, 25.00, 24.99
  ],
  lsl: 24.95,
  usl: 25.05
};
```

Tutorial step definitions (update selectors to match real template):

```javascript
var TUTORIAL_STEPS = [
  {
    selector: '<DATA_TABLE_SELECTOR>',
    title: 'This is your data',
    text: "We've loaded 50 diameter measurements from a turning process. Each value is a shaft diameter in millimeters.",
    position: 'right'
  },
  {
    selector: '<SPEC_LIMITS_SELECTOR>',
    title: 'Set your specs',
    text: "These are the customer requirements — Lower Spec Limit (24.95mm) and Upper Spec Limit (25.05mm). They're pre-filled but you can edit them.",
    position: 'bottom'
  },
  {
    selector: '<RUN_BUTTON_SELECTOR>',
    title: 'Run the analysis',
    text: 'Click this button to run your Bayesian capability analysis.',
    position: 'right',
    action: 'click',
    waitForSelector: '<RESULTS_SELECTOR>'
  },
  {
    selector: '<CPK_RESULT_SELECTOR>',
    title: 'Your capability indices',
    text: 'Cpk tells you whether your process can consistently meet spec. Values above 1.33 mean your process is capable with margin.',
    position: 'bottom'
  },
  {
    selector: '<BAYESIAN_RESULT_SELECTOR>',
    title: 'The Bayesian difference',
    text: "Unlike traditional Cpk, this gives you a confidence range — not just a point estimate. You can see how certain you should be about the capability.",
    position: 'bottom'
  },
  {
    selector: '<NARRATIVE_SELECTOR>',
    title: 'What this means',
    text: 'Svend tells you what to do with these numbers, not just what they are. Actionable interpretation, not just statistics.',
    position: 'top'
  },
  {
    selector: 'body',
    title: "You're ready",
    text: "That was a real Bayesian capability analysis. Ready to analyze your own data? Let's set up your account.",
    position: 'bottom'
  }
];
```

**CRITICAL: The `<SELECTOR>` placeholders above MUST be replaced with actual selectors from the template.** The implementer must read the template and `aw-config-forms.js` to find real element IDs/classes.

CSRF token for fetch calls: read from cookie via `document.cookie.match(/csrftoken=([^;]+)/)[1]` or from a hidden input `document.querySelector('[name=csrfmiddlewaretoken]').value`.

- [ ] **Step 3: Verify template loads without errors**

Run:
```bash
systemctl --user restart gunicorn
```

Visit `https://svend.ai/app/tutorial/` while logged in. Verify:
- Page loads without JS errors (browser console)
- Spotlight overlay appears
- Sample data is pre-loaded in the form

- [ ] **Step 4: Commit**

```bash
git add templates/analysis_workbench.html
git commit -m "feat: add tutorial mode to analysis workbench template"
```

---

### Task 6: Registration Redirect

**Files:**
- Modify: `templates/register.html:225` (change default redirect)
- Test: `tests/test_tutorial.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tutorial.py`:

```python
class RegistrationRedirectTest(TestCase):
    """Test that registration redirects to tutorial."""

    def test_register_template_has_tutorial_redirect(self):
        """register.html JS defaults to /app/tutorial/ not /app/."""
        resp = self.client.get("/register/")
        content = resp.content.decode()
        self.assertIn("/app/tutorial/", content)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 manage.py test tests.test_tutorial.RegistrationRedirectTest -v2`
Expected: FAIL — `/app/tutorial/` not in template

- [ ] **Step 3: Change the redirect in register.html**

In `templates/register.html` (~line 225), change:

```javascript
: (nextUrl ? decodeURIComponent(nextUrl) : '/app/');
```

To:

```javascript
: (nextUrl ? decodeURIComponent(nextUrl) : '/app/tutorial/');
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 manage.py test tests.test_tutorial.RegistrationRedirectTest -v2`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add templates/register.html tests/test_tutorial.py
git commit -m "feat: redirect new free registrations to /app/tutorial/"
```

---

### Task 7: Trim Onboarding Survey to 1 Step

**Files:**
- Modify: `templates/onboarding.html` (remove steps 1-2, trim step 0)
- Modify: `api/views.py` (make dropped fields optional in `onboarding_complete`)
- Test: `tests/test_tutorial.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tutorial.py`:

```python
class TrimmedSurveyTest(TestCase):
    """Test that onboarding survey works with only industry, role, primary_goal."""

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="surveytest",
            email="surveytest@test.com",
            password="testpass123",
        )
        self.client.force_authenticate(user=self.user)

    def test_survey_completes_with_minimal_fields(self):
        """POST with just industry, role, primary_goal succeeds."""
        resp = self.client.post(
            "/api/auth/onboarding/complete/",
            {
                "industry": "manufacturing",
                "role": "quality_engineer",
                "primary_goal": "process_improvement",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
        self.user.refresh_from_db()
        self.assertIsNotNone(self.user.onboarding_completed_at)

    def test_survey_does_not_require_dropped_fields(self):
        """Dropped fields are optional — no 400 when omitted."""
        resp = self.client.post(
            "/api/auth/onboarding/complete/",
            {
                "industry": "manufacturing",
                "role": "quality_engineer",
                "primary_goal": "process_improvement",
            },
            format="json",
        )
        self.assertEqual(resp.status_code, 200)
```

- [ ] **Step 2: Run tests to check current behavior**

Run: `python3 manage.py test tests.test_tutorial.TrimmedSurveyTest -v2`

Check whether `experience_level` is currently required. If the test fails with 400, we need to update the view. If it passes, only the template needs trimming.

- [ ] **Step 3: Update onboarding_complete view (if needed)**

In `api/views.py` `onboarding_complete` (~line 1746), find the required fields check. If `experience_level` is required, change it so only `industry`, `role`, and `primary_goal` are required. Make all other fields (`organization_size`, `experience_level`, `tools_used`, `confidence_stats`, `urgency`, `biggest_challenge`) optional with sensible defaults (empty string or None).

- [ ] **Step 4: Trim onboarding.html template**

In `templates/onboarding.html`:

1. **Step 0** — keep industry + role selects, move `primary_goal` (from step 1) here as a select or chip picker, remove `organization_size` and `experience_level`
2. **Remove step 1** entirely (tools_used chips — no longer collected)
3. **Remove step 2** entirely (confidence/urgency sliders, challenge textarea)
4. **Step 3** (completion) becomes step 1 — renumber
5. Update progress dots: 4 → 2
6. Update `nextStep()` / `prevStep()` / step validation JS for 2 steps
7. Update submission JS: only send `industry`, `role`, `primary_goal`

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 manage.py test tests.test_tutorial.TrimmedSurveyTest -v2`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add templates/onboarding.html api/views.py tests/test_tutorial.py
git commit -m "feat: trim onboarding survey to 1 step (industry, role, goal)"
```

---

### Task 8: Verification Gate in Analysis Workbench

**Files:**
- Modify: `templates/analysis_workbench.html` (add verify CTA block)
- Modify: `svend/urls.py` (custom analysis view with context)
- Test: `tests/test_tutorial.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tutorial.py`:

```python
class VerificationGateTest(TestCase):
    """Test that unverified users see verify CTA in workbench."""

    def setUp(self):
        self.user = User.objects.create_user(
            username="verifygate",
            email="verifygate@test.com",
            password="testpass123",
        )
        self.client.login(username="verifygate", password="testpass123")

    def test_unverified_user_with_tutorial_done_sees_gate(self):
        """Unverified users who completed tutorial see verify CTA."""
        self.user.tutorial_completed_at = timezone.now()
        self.user.save(update_fields=["tutorial_completed_at"])
        resp = self.client.get("/app/analysis/")
        content = resp.content.decode()
        self.assertIn("verify-gate", content)

    def test_verified_user_no_gate(self):
        """Verified users don't see the gate."""
        self.user.is_email_verified = True
        self.user.tutorial_completed_at = timezone.now()
        self.user.save(update_fields=["is_email_verified", "tutorial_completed_at"])
        resp = self.client.get("/app/analysis/")
        content = resp.content.decode()
        self.assertNotIn("verify-gate", content)

    def test_tutorial_mode_no_gate(self):
        """Tutorial route doesn't show verify gate."""
        resp = self.client.get("/app/tutorial/")
        content = resp.content.decode()
        self.assertNotIn("verify-gate", content)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 manage.py test tests.test_tutorial.VerificationGateTest -v2`
Expected: FAIL — no `verify-gate` in template

- [ ] **Step 3: Replace analysis URL with custom view**

In `svend/urls.py`, replace `_app_view("analysis_workbench.html")` for `/app/analysis/`:

```python
@login_required
def _analysis_view(request):
    """Serve analysis workbench with verification context."""
    show_verify_gate = (
        not request.user.is_email_verified
        and request.user.tutorial_completed_at is not None
    )
    return render(request, "analysis_workbench.html", {
        "show_verify_gate": show_verify_gate,
    })
```

Update URL:
```python
    path("app/analysis/", _analysis_view, name="analysis"),
```

- [ ] **Step 4: Add verify gate to analysis_workbench.html**

Near the top of the main content area, add:

```html
{% if show_verify_gate %}
<div id="verify-gate" class="sv-card" style="border-color: var(--sv-warning); margin-bottom: 1rem; padding: 1rem;">
  <div style="display: flex; align-items: center; justify-content: space-between;">
    <div>
      <strong>Verify your email to run analyses</strong>
      <p style="margin: 4px 0 0; color: var(--sv-text-muted); font-size: 0.875rem;">
        Check your inbox for a verification link, or request a new one.
      </p>
    </div>
    <button onclick="resendVerification()" class="sv-btn sv-btn-sm">Resend email</button>
  </div>
</div>
<script>
function resendVerification() {
  var csrfToken = document.querySelector('[name=csrfmiddlewaretoken]');
  var token = csrfToken ? csrfToken.value : document.cookie.match(/csrftoken=([^;]+)/)[1];
  fetch('/api/auth/send-verification/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-CSRFToken': token }
  }).then(function(r) {
    if (r.ok) sv.toast('Verification email sent!', 'success');
    else sv.toast('Could not send email. Try again later.', 'error');
  });
}
document.addEventListener('DOMContentLoaded', function() {
  var runBtn = document.querySelector('.aw-run-btn, #run-analysis, button[data-action="run"]');
  if (runBtn) {
    runBtn.disabled = true;
    runBtn.title = 'Verify your email to run analyses';
  }
});
</script>
{% endif %}
```

**NOTE:** The Run button selector must match the actual element. Verify against template.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 manage.py test tests.test_tutorial.VerificationGateTest -v2`
Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add svend/urls.py templates/analysis_workbench.html tests/test_tutorial.py
git commit -m "feat: add email verification gate in analysis workbench"
```

---

### Task 9: Returning User Banner on Dashboard

**Files:**
- Modify: `templates/dashboard.html` (add tutorial banner)
- Modify: `svend/urls.py` (custom dashboard view with context)
- Test: `tests/test_tutorial.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tutorial.py`:

```python
class ReturningUserBannerTest(TestCase):
    """Test tutorial banner on dashboard for users who never did tutorial."""

    def setUp(self):
        self.user = User.objects.create_user(
            username="returning",
            email="returning@test.com",
            password="testpass123",
        )
        self.client.login(username="returning", password="testpass123")

    def test_banner_shown_for_new_user(self):
        """Users without tutorial completion see banner."""
        resp = self.client.get("/app/")
        content = resp.content.decode()
        self.assertIn("tutorial-banner", content)

    def test_banner_hidden_after_tutorial(self):
        """Users who completed tutorial don't see banner."""
        self.user.tutorial_completed_at = timezone.now()
        self.user.save(update_fields=["tutorial_completed_at"])
        resp = self.client.get("/app/")
        content = resp.content.decode()
        self.assertNotIn("tutorial-banner", content)

    def test_banner_hidden_after_skip(self):
        """Users who skipped tutorial don't see banner."""
        self.user.tutorial_skipped_at = timezone.now()
        self.user.save(update_fields=["tutorial_skipped_at"])
        resp = self.client.get("/app/")
        content = resp.content.decode()
        self.assertNotIn("tutorial-banner", content)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 manage.py test tests.test_tutorial.ReturningUserBannerTest -v2`
Expected: FAIL — no `tutorial-banner` in dashboard

- [ ] **Step 3: Replace dashboard route with custom view**

In `svend/urls.py`, replace:

```python
    path("app/", _app_view("dashboard.html"), name="app"),
```

With:

```python
@login_required
def _dashboard_view(request):
    """Serve dashboard with tutorial banner context."""
    show_tutorial_banner = (
        request.user.tutorial_completed_at is None
        and request.user.tutorial_skipped_at is None
    )
    return render(request, "dashboard.html", {
        "show_tutorial_banner": show_tutorial_banner,
    })
```

And:
```python
    path("app/", _dashboard_view, name="app"),
```

- [ ] **Step 4: Add banner to dashboard.html**

At the top of the dashboard content area:

```html
{% if show_tutorial_banner %}
<div id="tutorial-banner" class="sv-card" style="border-color: var(--sv-primary); margin-bottom: 1rem; padding: 1rem; display: flex; align-items: center; justify-content: space-between;">
  <div>
    <strong>New to Svend?</strong>
    <span style="color: var(--sv-text-muted); margin-left: 8px;">Take the 2-minute guided tour and run your first analysis.</span>
  </div>
  <div style="display: flex; gap: 8px; align-items: center;">
    <a href="/app/tutorial/" class="sv-btn sv-btn-primary sv-btn-sm">Start tutorial</a>
    <button onclick="this.closest('#tutorial-banner').remove()" style="background:none;border:none;color:var(--sv-text-muted);cursor:pointer;font-size:1.2rem;">&times;</button>
  </div>
</div>
{% endif %}
```

Note: Dismissing the banner only hides it for the current session. It will reappear on next visit until the user starts or skips the tutorial. This is intentional — gentle persistence without server-side tracking of banner dismissals.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 manage.py test tests.test_tutorial.ReturningUserBannerTest -v2`
Expected: 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add svend/urls.py templates/dashboard.html tests/test_tutorial.py
git commit -m "feat: add tutorial banner on dashboard for returning users"
```

---

### Task 10: Update Verification Email Subject

**Files:**
- Modify: `accounts/models.py` (`send_verification_email` method)
- Test: `tests/test_tutorial.py`

- [ ] **Step 1: Read send_verification_email method**

Read `accounts/models.py` and find the `send_verification_email()` method. Note:
- How it sends email (Django `send_mail`, or `syn.sched`, or other)
- The current subject line
- Whether it uses Django's test email backend or an external service

This determines how to test it.

- [ ] **Step 2: Write the test**

Append to `tests/test_tutorial.py`. If it uses Django `send_mail`:

```python
from django.core import mail


class VerificationEmailSubjectTest(TestCase):
    """Test verification email has improved subject line."""

    def test_verification_email_subject(self):
        """Verification email subject is action-oriented."""
        user = User.objects.create_user(
            username="emailsubject",
            email="emailsubject@test.com",
            password="testpass123",
        )
        user.send_verification_email()
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("start analyzing", mail.outbox[0].subject.lower())
```

If it uses a different sending mechanism, adapt the test to check the subject string directly in the method or mock the sender.

- [ ] **Step 3: Run test to check current subject**

Run: `python3 manage.py test tests.test_tutorial.VerificationEmailSubjectTest -v2`
Expected: FAIL — current subject doesn't match

- [ ] **Step 4: Update the subject**

In `accounts/models.py` `send_verification_email()`, change the subject to:

```python
subject = "One click to start analyzing your own data"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 manage.py test tests.test_tutorial.VerificationEmailSubjectTest -v2`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add accounts/models.py tests/test_tutorial.py
git commit -m "feat: update verification email subject to action-oriented copy"
```

---

### Task 11: Integration Test — Full Flow

**Files:**
- Test: `tests/test_tutorial.py`

- [ ] **Step 1: Write the integration test**

Append to `tests/test_tutorial.py`:

```python
class TutorialFullFlowTest(TestCase):
    """Integration test: register -> tutorial -> survey -> verify gate."""

    def test_full_flow(self):
        """New user registration through tutorial completion."""
        # 1. Register
        resp = self.client.post(
            "/api/auth/register/",
            {"email": "fullflow@test.com", "password": "securepass123!"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)

        # 2. Tutorial page loads with tutorial flag
        resp = self.client.get("/app/tutorial/")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.context.get("tutorial"))

        # 3. Complete tutorial
        resp = self.client.post(
            "/api/auth/tutorial/complete/",
            {"action": "complete"},
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

        # 4. Complete survey
        resp = self.client.post(
            "/api/auth/onboarding/complete/",
            {
                "industry": "manufacturing",
                "role": "quality_engineer",
                "primary_goal": "process_improvement",
            },
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)

        # 5. Analysis workbench shows verify gate
        resp = self.client.get("/app/analysis/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("verify-gate", resp.content.decode())

        # 6. Dashboard does NOT show tutorial banner
        resp = self.client.get("/app/")
        self.assertNotIn("tutorial-banner", resp.content.decode())

        # 7. Verify user state
        user = User.objects.get(email="fullflow@test.com")
        self.assertIsNotNone(user.tutorial_completed_at)
        self.assertIsNotNone(user.onboarding_completed_at)
        self.assertFalse(user.is_email_verified)
```

- [ ] **Step 2: Run the integration test**

Run: `python3 manage.py test tests.test_tutorial.TutorialFullFlowTest -v2`
Expected: PASS

- [ ] **Step 3: Run ALL tutorial tests**

Run: `python3 manage.py test tests.test_tutorial -v2`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_tutorial.py
git commit -m "test: add integration test for full tutorial flow"
```

---

### Task 12: Deploy + Manual Verification

**Files:** None (deployment task)

- [ ] **Step 1: Collect static files**

```bash
cd ~/kjerne && python3 manage.py collectstatic --noinput 2>&1 | tail -5
```

- [ ] **Step 2: Restart gunicorn**

```bash
systemctl --user restart gunicorn
```

- [ ] **Step 3: Manual browser verification**

In an incognito browser, verify the full flow:

1. `https://svend.ai/register/` — create test account with real email
2. Confirm redirect to `/app/tutorial/`
3. Spotlight overlay appears, sample data loaded
4. Click through all 7 steps (step 3 waits for Run click)
5. Survey appears post-tutorial (1 step: industry, role, goal)
6. Submit → verify CTA appears
7. `/app/analysis/` — Run button disabled, verify message shown
8. `/app/` — no tutorial banner (already completed)
9. Check inbox — verification email with new subject line
10. Click verify link — go back to `/app/analysis/`, Run button now enabled

- [ ] **Step 4: Clean up test account**

```bash
set -a && source /etc/svend/env && set +a && cd ~/kjerne
python3 manage.py shell -c "from accounts.models import User; User.objects.filter(email='<test-email>').delete()"
```

- [ ] **Step 5: Commit any fixes from manual testing**

```bash
git add -A && git status
# Only commit if there are fixes:
git commit -m "fix: adjustments from manual tutorial flow verification"
```

# Object 271 — UX Deep Audit + Proposal

**Date:** 2026-03-29
**Agent:** S2 (Frontend / UX)
**Scope:** Every QMS-facing surface audited against best-in-class paradigms from ATC/SCADA, Bloomberg, IDE, Linear, EHR, and aviation maintenance systems.
**Status:** PROPOSAL — awaiting S1 backend response + Eric arbitration

---

## Part 1: What We Have vs What Best-in-Class Does

### What We Got Right

1. **The Loop sidebar IS the right structure.** Detect → Investigate → Standardize → Verify as persistent navigation matches the IDE sidebar pattern. Icons, counts, active state — all correct. This is our equivalent of VS Code's activity bar.

2. **The list-detail pattern (detail left, rail right) works.** Signals and Commitments prove the pattern. The detail panel has room to grow. The rail is a scannable queue. This is closer to Linear than to traditional QMS (which uses modals-on-tables).

3. **Modals follow the Svend pattern.** `.sv-modal` is consistent, doesn't use browser dialogs, supports escape/overlay-click-to-close. This is baseline professional.

4. **Supplier workbench with panels (scorecard, claims, CoA, history) is the right depth model.** Each panel is a collapsible section with its own data and actions. This is the EHR Storyboard pattern adapted for supplier management.

5. **Color is mostly semantic.** Status badges use consistent colors (green=good, amber=warning, red=bad, blue=info, purple=transition). Not decorative.

### What We're Missing (Ranked by Impact)

---

### CRITICAL: No Command Palette

**Every professional tool has one.** Bloomberg has `<GO>`, VS Code has `Cmd+K`, Linear has `Cmd+K`. We have nothing. Users navigate by clicking through the sidebar, which is fine for browsing but terrible for recall.

A quality engineer who knows they want `NCR-042` should type 5 characters and be there, not click Detect → Signals → scroll → find it.

**Proposal:** Add `Cmd+K` / `Ctrl+K` command palette to `base_app.html` (not just the Loop shell — globally). Accepts:
- Record IDs: `NCR-042`, `CAPA-017`, `INV-003`
- Entity + filter: `overdue commitments`, `open signals`
- Navigation: `suppliers`, `process map`, `settings`
- Actions: `new signal`, `new commitment`, `new claim`
- Fuzzy search across all entities

This is the single highest-impact UX feature we can add. It makes power users 3x faster and makes the platform feel like a tool, not a website.

---

### CRITICAL: No Keyboard Triage

The signals and commitments lists are mouse-only. You cannot arrow through items, set status with a key, or bulk-select. This is the biggest gap vs Linear.

**Proposal:** Add keyboard navigation to every list-detail view:
- `↑/↓` to move through list items
- `Enter` to select/expand
- `Escape` to deselect
- Single-key actions on selected item (configurable per list):
  - Signals: `A` = acknowledge, `I` = open investigation, `D` = dismiss
  - Commitments: `S` = start, `F` = fulfill, `B` = mark broken
  - Claims: `R` = review, `E` = escalate
- `Shift+↑/↓` for multi-select, then batch action

---

### HIGH: No Context Banner

When you're deep in a claim detail or a CoA measurement table, you lose context. What supplier is this for? What's the claim status? You have to scroll up to the header.

EHR systems solve this with a persistent patient banner. We need a persistent context banner.

**Proposal:** When viewing any detail (signal, commitment, supplier, claim, CoA), show a thin context bar at the top of the detail panel:

```
┌─────────────────────────────────────────────────────────────┐
│ SUPPLIER: Acme Fasteners  │  CLAIM: SC-042  │  STATUS: Under Review  │  DUE: Apr 5  │
└─────────────────────────────────────────────────────────────┘
```

Sticky. Always visible. Shows the chain: what entity, what record, what status, what's urgent.

---

### HIGH: No Activity Feed Pattern

Commitments have notes. Suppliers have status history. But there's no unified activity feed pattern. Each section renders its timeline differently.

**Proposal:** Create a shared `.activity-feed` component in `base_loop.html`:
- Chronological entries with timestamp, author, action type icon, description
- Comment input at the bottom (same as commitment notes)
- System events (status changes, assignments) rendered differently from user comments
- Collapsible by default, expanded on click
- Same component used everywhere: signals, commitments, claims, CoAs, investigations

---

### HIGH: No Text Expansion / Templates

When a quality engineer writes a root cause description, they start from blank. Every time. EHR SmartPhrases solve this — type `.5why` and get a template.

**Proposal:** Add slash commands to all textarea fields in the Loop shell:
- `/5why` → 5-Why template
- `/fishbone` → Ishikawa categories (Man, Machine, Material, Method, Measurement, Environment)
- `/containment` → Containment action checklist
- `/8d` → 8D report skeleton
- `/checklist` → Empty checklist with add-item
- `/table` → Simple data table

Implementation: listen for `/` in textareas, show a dropdown of available templates, insert structured markdown on selection.

---

### MEDIUM: No Prerequisite Enforcement (Workflow Gates)

Our claim lifecycle has `VALID_TRANSITIONS` on the model but the UI doesn't show what's BLOCKING a transition. If a claim can't move to "verified" because there's no verification record, the user just gets an error.

Aviation maintenance and EHR both solve this with visible gates.

**Proposal:** When rendering action buttons, check prerequisites and show them:

```
[Verify ✓ response accepted] [Close ✗ no verification record — Add Verification first]
```

Disabled buttons with tooltip explaining WHY they're disabled. The user sees the path without trial-and-error.

---

### MEDIUM: No Saved Filter Views

Users can filter signals by status, suppliers by rating. But these filters are ephemeral — refresh the page and they're gone. Linear solves this with saved views as sidebar items.

**Proposal:** Allow users to save filter configurations as named views that appear in the Loop sidebar under the relevant section:

```
DETECT
  Signals
  → My Untriaged
  → Critical This Week
  → Supplier Issues Only
```

Storage: user preferences JSON on the User model, or a lightweight `SavedView` model.

---

### MEDIUM: No Diff/Comparison View

When a controlled document is revised, there's no way to see what changed. When a supplier responds to a rejected claim with a "revised" response, there's no diff between revision 1 and revision 2.

**Proposal:** For any entity with revision history (documents, supplier responses, FMIS rows), add a "Compare" action that shows the two versions side-by-side with changes highlighted. Use a simple word-level diff algorithm (no library needed for short text).

---

### LOW: No Minimap / Graph Overview in Detail Context

When working deep in a supplier claim or investigation, the user has no sense of where this fits in the bigger picture. The process map exists but it's a separate page.

**Proposal:** Add an optional "Graph Context" mini-panel to the detail view. Shows the 1-hop neighborhood of relevant process nodes. Clicking opens the full process map centered on those nodes. This is the VS Code minimap adapted for process knowledge.

---

### LOW: No Bulk Operations

You can triage signals one at a time. You can't select 5 signals from the same root cause and batch-link them to one investigation.

**Proposal:** Add multi-select to list views (checkbox or Shift+click), then show a bulk action bar at the top: "5 selected → [Assign to...] [Link to Investigation...] [Dismiss All]"

---

## Part 2: Specific Section Proposals

### Signals → Triage Workstation

Current: list + detail panel. Click signal, see description, click Acknowledge/Investigate/Dismiss.

**Proposed:** Same layout but optimized for throughput:
- Keyboard navigation (↑↓ to scan, A/I/D for actions)
- Auto-advance to next signal after action (triage mode)
- Severity color strip on left edge of each list item (not just a badge)
- Context banner when a signal is selected
- "Related signals" section in detail (other signals from same source, same time period)
- Quick link: "Create Claim from Signal" (pre-populates claim from signal data)

### Commitments → Accountability Board

Current: list + detail with notes, artifacts, resources.

**Proposed additions:**
- **Timeline view** as alternative to list (Gantt-style: commitments as bars on a time axis, colored by status). This is the aviation planning board pattern.
- **Owner grouping** — group commitments by owner to see workload distribution
- **Precondition dependency view** — show which commitments block which (tree/DAG)
- **Due date heatmap** on the list: items due today are brighter, items due next week are dimmer

### Suppliers → Supplier Intelligence

Current: workbench with scorecard, claims, CoA, history.

**Proposed additions:**
- **Radar chart** for evaluation scores (quality/delivery/price/communication as a 4-axis spider chart)
- **Trend sparklines** next to each score dimension (are they improving or declining?)
- **Risk indicator** combining: low quality score + open claims + overdue evaluations = elevated risk badge
- **"Create Signal from Supplier" button** — quick path from supplier issue to Loop detection

### Investigation → Already Good (the 3-pane layout works)

**Proposed additions:**
- Context banner showing investigation status + signal source
- Graph scope indicator (which nodes are in this investigation's subgraph)
- Keyboard shortcuts for tool launching (N = note, R = RCA, D = data)

### Policies → Cloudflare-Style Config Panel

This hasn't been built yet. The existing `loop_policy.html` is a card-based rule builder. The enterprise_configuration_spec.md defines a much richer system.

**Proposal:** Build this as its own section with:
- Domain tabs (Organization, Quality, Process, Safety, Compliance, Notifications)
- Each domain: list of settings with current value, default, and preset indicator
- Toggle/input per setting (inline editing, no modal)
- Preset selector at top ("ISO 9001" / "IATF 16949" / "AS9100D" / "Custom")
- Diff view: "What would change if I switch from ISO 9001 to IATF 16949?"
- Each setting shows which OLR-001 section it implements

---

## Part 3: Structural Changes to base_loop.html

### Add to Shell CSS

```css
/* Context banner */
.loop-context-banner { ... sticky top, thin bar, breadcrumb-style }

/* Activity feed (shared component) */
.activity-feed { ... }
.activity-entry { ... }
.activity-entry.system { ... muted, italic }
.activity-entry.user { ... normal weight }

/* Keyboard selection indicator */
.loop-list-item.kb-focused { ... outline, not background change }

/* Bulk action bar */
.loop-bulk-bar { ... sticky bottom of list, appears when items selected }

/* Severity strip (left edge color bar) */
.loop-list-item .severity-strip { ... 3px left border }
```

### Add to Shell JS

```javascript
// Command palette (global, not loop-specific)
// Goes in base_app.html, triggered by Cmd+K / Ctrl+K

// Keyboard navigation (per list-detail page)
// Arrow keys, enter, escape, single-key actions

// Auto-advance after triage action
```

---

## Part 4: Implementation Priority

| # | Feature | Effort | Impact | Where |
|---|---------|--------|--------|-------|
| 1 | Command palette | 1 session | CRITICAL | base_app.html |
| 2 | Keyboard triage (signals) | 0.5 session | CRITICAL | loop_detect_signals.html |
| 3 | Context banner | 0.5 session | HIGH | base_loop.html + all detail templates |
| 4 | Activity feed component | 0.5 session | HIGH | base_loop.html (shared) |
| 5 | Slash command templates | 1 session | HIGH | base_loop.html (shared) |
| 6 | Workflow gate indicators | 0.5 session | MEDIUM | Per detail template |
| 7 | Configuration panel (Policies) | 2 sessions | MEDIUM | New template |
| 8 | Saved filter views | 1 session | MEDIUM | base_loop.html + sidebar |
| 9 | Keyboard triage (all lists) | 0.5 session | MEDIUM | All list-detail templates |
| 10 | Bulk operations | 1 session | LOW | base_loop.html (shared) |
| 11 | Diff/comparison view | 1 session | LOW | Supplier responses, documents |
| 12 | Graph context mini-panel | 1 session | LOW | Detail templates |

**Recommended sequence:** 1 → 2 → 3 → 4 → 5 → 6 → 7

The command palette alone changes the feel of the entire platform from "website" to "tool."

---

## Part 5: What We Should REJECT

1. **Drag-and-drop everything.** Drag-and-drop is satisfying but slow. Keyboard is faster. Use drag only where spatial arrangement matters (graph map, Gantt timeline). Not for reordering lists.

2. **Wizard-style multi-step forms.** Quality engineers hate wizards. They know what they want to enter. Give them a single form with all fields visible, not 5 steps of 3 fields each. Collapse optional sections, but don't hide them behind a "Next" button.

3. **Toast notifications for every action.** Toast on error/warning only. Success is visible in the UI itself (the status badge changed, the item moved in the list). Don't congratulate the user for doing their job.

4. **Dashboard-first landing.** The dashboard with KPI cards is the wrong starting point. The RIGHT starting point is the action queue — what needs attention NOW. KPIs are for managers reviewing weekly, not for engineers starting their day.

5. **Infinite scroll.** Paginate with explicit counts. "Showing 1-50 of 347 signals." Users need to know the magnitude. Infinite scroll hides it.

6. **Auto-save.** For regulated records, auto-save is dangerous. The user must explicitly submit. Show "unsaved changes" indicator, but never auto-commit. This is a hard stop from the EHR paradigm.

7. **Animated transitions between views.** Slide, fade, zoom — they all add latency. Quality engineers use this tool 8 hours a day. Every 200ms animation costs productivity. Instant render, no transition.

---

## Questions for S1 (Backend)

1. Can the command palette search across all entities (signals, commitments, claims, CoAs, suppliers, investigations, documents) with one query? What endpoint shape would that need?
2. Does the `TenantConfig` model support the full enterprise_configuration_spec.md settings list, or does it need extension?
3. What's the event sourcing model for the activity feed? Do we have timestamped events on every entity, or do some entities lack history?
4. Can saved filter views be stored on the User model's preferences JSON, or do they need their own model?

## Questions for Eric

1. Command palette — is this the right #1 priority? It's invisible to demo viewers but transformative for daily users.
2. The "reject" list — anything you disagree with? Especially #4 (dashboard-first) and #6 (auto-save).
3. How aggressive should keyboard shortcuts be? Single-key (Linear-style: `A` to acknowledge) or modified (Ctrl+A)?
4. The configuration panel is the policy surface that replaces the current Cloudflare-style rule builder. Is it time to build that, or should we wait for OLR-001 to stabilize?

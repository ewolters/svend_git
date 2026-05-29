# Focus Group: Workbench Visualization — "What Do You See?"

**Date:** 2026-05-06
**Technique:** Research-Grounded Focus Group (existing personas)
**Prompt:** Scenario-based. Each persona given their real work situation + system capabilities (no design shown). Asked: "Describe what you SEE on your screen at each step."
**Personas:**
- Maria Kowalczyk — QE Manager, Tier-2 automotive stamping, 340 employees (Champion)
- Dave Przybylski — Quality Manager, aerospace job shop, 85 employees (Skeptic)
- Kenji Morales — CI Lead, food/bev packaging, 500 employees (Adjacent)
- Destiny Washington — Floor Inspector, plastics injection, 220 employees (Constraint)

---

## Persona Responses

### Maria Kowalczyk — Tuesday Morning, 6:45 AM

**Scenario:** Check overnight Cpk across 6 lines, investigate drift on one, run counterfactual, flow results into PPAP.

**Overview screen:** Six cards, one per line. Each card: line name (14pt bold), Cpk number (48pt, color-coded green/yellow/red), sparkline (30-day trend), timestamp. Two rows of three. "I glance at six numbers in two seconds."

**Drill-in:** Card expands to fill monitor, other cards shrink to thin strip (like browser tabs, still showing Cpk). Detail view in thirds:
- Top: Control chart (X-bar/R, 90 days, drift zone highlighted, red dots for WE violations with tooltip)
- Middle: Cpk/Pp/Ppk summary table (left) + histogram with spec limits as red dashed lines (right)
- Bottom: Event timeline (horizontal, like Git log — fixture adjustments, material lots, shift changes)

**Counterfactual:** Right-click fixture event → "What if?" panel slides in from right (35% width). Chart compresses but stays visible. Pre-populated parameters, slider, preview histogram updates real-time. Projected Cpk + confidence. Historical note: "Similar adjustment in Nov improved Cpk from 1.15 to 1.44." Dismiss / Save to Workbench buttons.

**PPAP flow:** Save → toast notification "linked to PPAP." Second monitor: 18-element PPAP checklist (vertical kanban, green/yellow/red). Counterfactual auto-linked. Summary: "12/18 complete. SQE visit May 13."

**Claude:** Toolbar icon, invisible. "If this opens to a chatbot saying 'How can I help you today?' I'm closing it."

**Key quote:** "The whole thing should feel like a Bloomberg Terminal had a baby with Minitab and raised it in a stamping plant."

---

### Dave Przybylski — 7:15 AM, Customer Complaint Response

**Scenario:** Pull data for dimensional issue on titanium part, run control chart, what-if with last 10 subgroups, write audit-ready response letter.

**Home screen:** Search bar at top (prominent, autocomplete). Below: last 5 items (plain text — date, part number, what was run). "Like recent files in Excel." That's it.

**Data view:** Table on left (40%), actual numbers in grid. White background, black text, light gridlines. Spec limits visible at top without clicking: "USL: 1.255 / LSL: 1.245 / Nominal: 1.250."

**Control chart:** Keyboard shortcut ("C"). Chart appears right (60%). Data table compresses to narrow column. Limits labeled ON the lines ("UCL = 1.2538"). Spec limits: dashed red. Control limits: blue. Violations: uniform red dots. No animation. R chart below X-bar, same axis. Math one click deep (X-double-bar, R-bar, A2, D3, D4).

**What-if:** Click-drag on chart to select last 10 subgroups OR type "last 10." Chart redraws in place. Old chart stays — compare mode (full-30 top, last-10 bottom). Three states, one key: full / filtered / compare.

**Response letter:** Key "R" → panel slides right (50% width). Simple text editor, no ribbon. Header auto-filled. "Attach current chart" button drops in chart image. Capability table drops in. System reminds of prior incidents but does NOT auto-generate corrective action.

**Audit trail:** "Trace" icon at bottom. Every action timestamped. "Click trace, print, hand to auditor. Audit readiness as byproduct."

**Key quote:** "No sidebar navigation with 20 icons. No tabs for 8 different modules. One screen. Everything is on it or one keystroke away."

---

### Kenji Morales — Monday Morning, New DMAIC Project

**Scenario:** OEE data → Pareto → current state VSM → SMED simulation → future state → A3 → Director walkthrough.

**Canvas:** Empty, mostly white, light grid. "Think opening Excel — the sheet IS the work."

**Data entry:** Three clicks (Line 3, date range, OEE). Summary card appears on canvas "like a Post-it." Right-click → Pareto appears NEXT TO IT. "Like I taped a printout next to my Post-it." Muted blue bars, biggest on top.

**VSM transition:** Click top Pareto bar → VSM opens to the RIGHT. Pareto slides left but stays visible. "If the Pareto disappears, I'm in browser-navigation hell." VSM looks exactly like Post-its on butcher paper — rectangular blocks, triangles for WIP, timeline along bottom (green = value-add, red = waste). Data PRE-POPULATED. Missing data = dashed border with question mark.

**Simulation:** Lasso-select changeover steps → "simulate" → panel opens right (25% width). VSM DUPLICATES: current top, future bottom. Changeover blocks compressed/lighter. Delta between them: "OEE: 71.2% → 78.8%" in 48pt. "That's the number my Director cares about."

**A3 composition:** "Build A3" → A3 populated from canvas artifacts. Pareto in Current Condition, VSMs in Analysis, OEE delta in Target. Items LINKED not copied — click to zoom to source. "If data updates next week, the A3 updates."

**Presentation mode:** One click → full-screen walkthrough. Four stops: A3 → Pareto → VSM comparison → action plan. Arrow keys. Same artifacts, full-screened. Exit → back to canvas.

**Key quote:** "Make the wall digital. Keep it spatial. Make the links live. I have rebuilt that conference room wall four times this year."

---

### Destiny Washington — 6 AM, First Shift Inspection Round

**Scenario:** Morning inspection on 4 machines (12 measurements), out-of-spec on Machine 2, log NCR, track what happens.

**Entry screen:** Four boxes (Machine 1-4). Big. Tap → measuring. Two taps to start.

**Measurement input:** Calculator-style keypad (bottom half, 3/4 inch keys minimum). Part PICTURE with arrow pointing to dimension. Nominal + tolerances big, not in tooltip. Three input boxes across top (Sample 1, 2, 3), auto-advance on fill. Green border = in spec. Red border (gets bigger) = out of spec.

**Navigation:** One dimension per screen, swipe right. Big arrows (gloves). Header always visible: Machine, Part, "Dimension 2 of 5." Auto-save, no save button.

**Out-of-spec:** Yellow bar slides down: "Out of spec detected — Dimension 3, Machine 2. Finish your checks, then review." Doesn't yank away. Bar follows through remaining measurements. Patient.

**NCR screen — ONE screen:** Top third auto-filled (machine, part, dimension, measurement, tolerance, date, shift, operator). Middle: defect type as BIG BUTTONS (Flash, Short shot, Sink mark — top 8). Severity: 3 buttons (Cosmetic, Functional, Safety). Disposition: 3 buttons. Notes: big text box with voice-to-text. Submit: ONE green button.

**After submit — package-tracking timeline:**
✓ 6:14 AM — NCR logged by Destiny W.
✓ 6:14 AM — Lot flagged for quarantine
✓ 6:14 AM — QE notification sent to Mark Thompson
◯ Pending — QE review
Green checks pulse briefly. Returnable: red badge on Machine 2, tap to see updated timeline.

**History (before submit):** Cards showing prior incidents on same mold/defect. Date in plain language ("3 weeks ago"), disposition, whether fix worked. Frequency alert: "Flash 4 times in 6 months — increasing."

**Key quote:** "If I can do my morning round in the same time as the paper form, I'll use it. If five minutes longer, I'm going back to paper."

---

## Synthesis

### Convergent Design Patterns (independently described by multiple personas)

**1. Compress, don't navigate.** When a new tool opens, the previous context shrinks but stays visible. Maria: cards become browser-tab strip. Dave: data table becomes narrow column. Kenji: Pareto slides left. Destiny: yellow bar follows. Nobody described clicking away and coming back.

**2. Right-side slide-in panels.** Maria: What-if at 35%. Dave: response letter at 50%. Kenji: simulation at 25%. Three of four described secondary tasks as right-side panels at specific width fractions.

**3. Color is three-state status, not decoration.** Green/yellow/red across all four. Always functional. Never branding. Red = stop/out-of-spec. Green = good/in-spec. Yellow = attention/in-progress.

**4. Home screen is the work, not a portal.** Maria: six line cards. Dave: search + recent. Kenji: blank canvas. Destiny: four machine buttons. Nobody described dashboards, sidebars, or module pickers.

**5. History as forward-reading narrative.** Maria: event timeline like Git log. Dave: "see October 2025 response." Kenji: linked artifacts that update. Destiny: package-tracking timeline. History is a story, not a table.

**6. Audit trail as exhaust.** All four described working normally and the log writing itself. Nobody described filling out an audit log as a separate step.

**7. AI invisible until summoned.** Maria described hiding it. Three others didn't mention it at all.

### The Surface Model

Three layers, consistent across all personas:

```
ROSTER          → Your scope (6 lines / search bar / canvas / 4 machines)
WORKSPACE       → Where tools live (adapts to role)
TRACE           → What happened (timeline/links, always present, minimal until inspected)
```

Workspace adapts to role density:
- **Destiny (floor):** Single-focus. One thing per screen. Swipe to advance.
- **Dave (solo QM):** Split-pane. Data + chart + occasional panel. Max 3 things.
- **Maria (QE manager):** Multi-pane. 6 cards, charts, histograms, timelines, panels. 5-6 things.
- **Kenji (CI lead):** Infinite canvas. Spatial arrangement. Unlimited artifacts.

The interaction grammar is the same at every density: compress, slide, link, trace.

### Natural Metaphors (what they reached for unprompted)

| Persona | Metaphor | Implication |
|---|---|---|
| Maria | Bloomberg Terminal + Minitab | Dense, real-time, portfolio view, drill-down |
| Dave | File system / recent files | Search, open, work, close. Commands, not places. |
| Kenji | Conference room wall | Spatial, persistent, point-at-able, live-linked |
| Destiny | Package tracker | Linear, status-driven, "where is my thing now" |

None described: cockpit, game, rack, social feed, chat, control room.

### What Nobody Wanted

- Sidebars with icon grids (0/4)
- Dashboards as home screens (0/4)
- Chat interfaces (0/4, Maria explicitly hostile)
- Dark mode / visual theming (0/4)
- Module switching / app launcher (0/4)
- AI-generated text without asking (0/4)
- Animations beyond status pulses (0/4)
- Confirmation dialogs (Destiny: "I've been doing this for nine years")
- Mandatory fields the system already knows (Destiny: "re-enter data the system already has? never worked on a factory floor")

### The Unstated Agreement

All four independently described a workbench that:
- Opens to their scope, not to "the system"
- Shows data first, tools second
- Keeps previous context visible when new context opens
- Uses color for exactly one purpose (status)
- Records what happened without asking them to record it
- Never puts navigation between them and the work
- Treats AI as infrastructure, not interface

The only disagreement is density — how many things on screen. That's a function of role, not preference.

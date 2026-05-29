# SVEND Product Spec

**Last updated:** 2026-05-11
**Status:** Working spec — synthesized from 20 innovation artifacts (2026-05-03 through 2026-05-06) + architecture session + 2026-05-11 focus groups (current state + rack concept)
**Supersedes:** All prior scattered direction documents, container spec draft, navigation artifacts

---

## 1. Identity

**SVEND is tools for practitioners. Claude is the connective tissue.**

The tools do the computation. The connections do the work. The knowledge accumulates automatically.

- **Tools (devices)** = computation engines + structured output. SPC, DOE, capability, control charts, FMEA, RCA, VSM, QMS, Hoshin. Plus auditor-ready reports, CAPA chains, reasoning trails. Each tool is a self-contained device that works independently.
- **Connections (flowchart)** = how devices wire together. A capability study outputs Cpk; that number flows into a control plan, a simulation, a report. The wiring is visible, auditable, and user-controlled. This is the connective tissue between tools that practitioners currently maintain in Excel.
- **Claude** = the mechanic in the corner. Available when you need help. Triage, plain-language explanation, artifact generation, trend analysis across sessions. Not the entry point. Not the orchestrator. Not required. Tools work at full capability with Claude absent.
- **PROVA** = Claude's invisible memory. What was learned, tried, worked. Institutional knowledge that stays when people leave. Users never see graph operations.
- **PCL** = the live parameter bus. Current process state — measures with provenance. Devices read from and write to PCL. When PCL values update (e.g., someone remeasures on the floor), downstream devices see the change.

### Three Levels of Operational Maturity

1. **Vague** — "8% scrap." Can't decompose. Most companies.
2. **Structured intelligence** — "124 lb scrap from material A, 89 lb from heavy coating, same nozzle." Causal decomposition. Actionable.
3. **Resource deployment** — Who's working this? How many hours? How do we ensure it happens right?

SVEND facilitates climbing from 1 to 3 AND supports execution once there.

### Purchase Triggers (4/4 convergence across focus groups)

- **Tribal knowledge walking out the door** — universal kill shot
- **Audit reasoning chains** — every persona needs auditor-ready documentation
- **Dollar quantification** — show money, not metrics

### Positioning Rules

- NEVER say "AI" externally — 8/8 personas across two focus groups stripped the word
- NEVER say "operational excellence" — 4/4 rejected
- NEVER say "analytical provenance" externally — internal concept only
- NEVER say "video game," "glass cockpit," "data bus," "signal chain," "rack" — all killed in testing
- Lead with: audit defense + reasoning trails + tribal knowledge retention
- Price anchor: "costs less than one Minitab seat"
- Language to use: "reasoning trail," "the reasoning already exists when the auditor asks," "audit defense you didn't have to build," "bring your ugliest dataset," "keyboard shortcuts," "muscle memory," "customer-ready," "same version of every number," "chain of evidence"

*Sources: identity-transfer, identity-morph, identity-focus, identity-direction (May 3), ux-design-focus (May 4), rack-interface-focus (May 5), game-ui-pressure-test (May 5)*

---

## 2. Navigation & Interaction Model

### The Flowchart (updated 2026-05-11)

**The workspace IS a flowchart.** Each shape is a device (tool). Lines between shapes are connections (data flowing from one device to another). This is the primary view — always visible, not hidden behind a toggle.

**How it works:**
- User opens a **template** (pre-drawn flowchart) or starts blank
- Each shape has its own front panel — click a shape to open the tool interface in the main pane
- Shapes have **typed ports** on their edges (colored dots). Connections run between compatible ports.
- The flowchart stays visible as a **film strip** (horizontal, top or side) showing the full flow and where you are
- Connect devices either by **dragging** port-to-port or via a **Connect button** that opens a to→from modal
- Every connection is logged — who created it, when, what data flows through it. Auditable.

**A template is a pre-drawn flowchart.** "PPAP Package" = capability study → control chart → FMEA → control plan → report builder, already connected. User loads template, drops data in the first shape, results flow through. "Green Belt DMAIC" = data collection → capability → control chart → fishbone → report, already wired in sequence.

Templates are: loadable, modifiable ("borrow template, break it"), saveable as user's own, shareable (one link to students/team).

### Two Modes (from May 11 focus group — 4/4 convergence)

**Template mode (guided path):** For Tomasz, Carmen's students, inspectors. Flowchart is visible but connections are pre-built. User follows the path — step 1 (enter data), step 2 (see results), step 3 (get report). Cables/ports are background infrastructure, not the interaction surface. Big clear steps. No wiring required.

**Build mode (full flowchart):** For Dana, Raj, Carmen herself, power users. Full port visibility, drag-to-connect or modal connect, add/remove devices from library, save as template. This is the Reason/Alteryx experience.

Carmen: "If you try to split the difference and show cables to everyone, you'll lose my classroom in eleven minutes." The modes must be cleanly separated.

### Validated Principles (prior sessions, still valid)

**No menus.** Universally validated (4/4 in game-UI test, 4/4 in session-navigation test). The flowchart replaces the menu — you see your tools as shapes, not as dropdown options.

**Project-based.** Users work in projects. The project's flowchart IS the audit trail — every device, every connection, every data source, visible and traceable.

**"Borrow template, break it."** Strongest convergence in the morph (4/4 top combos). Templates are the entry point. Modification is how knowledge transfers.

**Two-surface architecture.** Personal surface (your flowchart with your devices positioned) vs shared inventory (device library, available when needed). Pre-positioning, not runtime navigation.

**State travels with the work.** The data flows through the flowchart. The flowchart IS the state. When upstream data changes, downstream devices see the update through their connections.

### Validated Interaction Patterns

| Pattern | Status | Source |
|---|---|---|
| Hotbar (1-2-3-4 keys) for frequent devices | Validated 4/4 | game-ui-pressure-test |
| Q (quick run) — paste data → chart → Cpk | Validated — "almost sells me by itself" (Dave) | game-ui-pressure-test |
| M (map) — on-demand full flowchart view | Validated with caveats (must be interrogatable) | game-ui-pressure-test |
| Templates as entry point | Validated 4/4 + 4/4 (May 11) | game-ui-pressure-test, nav-ux-morph, rack-focus |
| Drag-to-bind for data routing | Validated, plus Connect modal alternative | game-ui-pressure-test, May 11 |
| Command strip (persistent, inherits context) | Validated via trading floor analogy | screen-design-transfer |
| Three-tier Claude disclosure (ambient / contextual / deliberative) | Converged across 4 domains | screen-design-transfer |

### Claude's Screen Presence

Four domains (trading floor, ICU, military C2, kitchen) independently converged on the same architecture:

1. **Primary instrument is sovereign** — never displaced. 75-80% of screen. The flowchart + active device panel owns the screen.
2. **Claude lives at the edge** — collapsed by default, visible when you look up, invisible when heads-down.
3. **Three tiers keyed to urgency, not skill:**
   - **Ambient** (zero cost): peripheral signals, color changes, indicators
   - **Contextual** (single touch): tap data element → pre-loaded brief, auto-dismisses
   - **Deliberative** (intentional): full conversation panel, 40-60% width
4. **Every advisory output is a pre-wired action** — recommendation terminates in executable format, not text.
5. **Tool works at full capability with Claude absent.** Claude enhances; Claude is not required.

Dana (May 11): "Useful if it pulls up the last three capability studies for the same characteristic — that's assistance. Interpreting my results is a parlor trick."

Meta-principle: "The user's act of working in the primary instrument IS the query formation."

### Must Fix Before Shipping

1. **Q must work in <2 minutes.** Paste data → chart → Cpk. No wizard, no spec-limits-first gate, no account creation.
2. **Kill all metaphor language externally.** No "video game," "rack," "glass cockpit," "data bus," "signal chain." Say "keyboard shortcuts," "muscle memory," "chain of evidence."
3. **Every connection auditable.** Who created it, when, what data flowed. Visible in Map view, exportable.
4. **Data ingestion feedback.** After drop: row count, column preview, quality flags. No silent failures.
5. **Calculation transparency.** Every device shows its method — estimator used, sample size, transformation applied. Tomasz: "If it just shows Cpk = 1.34 with no explanation, I have the same problem as Excel except I paid for it."
6. **Customer-specific output formatting.** Dana: Ford and GM want different Cpk report formats. Tomasz needs PDFs his German customer will accept. Report builder must support output templates per customer.
7. **Scratch vs persistent mode.** Scratch = exploring, no audit trail, quick checks. Persistent = writes to PCL, builds audit trail, links to part numbers. Scratch is the behavioral hook (Tomasz: "the thing that gets me to come back"). Transition from scratch to persistent must be deliberate.
8. **Free tier must be permanent and functional.** Not a trial. Capability studies, control charts, basic hypothesis tests. Carmen: "If my student hits a paywall in week three, I will burn your name in every MBB Slack group."
9. **Hints must be recallable.** Persistent "?" or long-press-for-help.
10. **Graceful re-entry.** Resumable state. "You were here" on return.

### Open UX Questions

- Multi-engineer collaboration (who has state on a shared flowchart?)
- Shared datasets across projects (version control)
- Accessibility / screen reader compliance
- Hotbar overflow past 4-5 devices
- Column mapping UX — drag gets 80%, the 20% (which col is measurement vs grouping?) needs solution
- Mobile/tablet — does the flowchart scale down?
- Minitab data import (MTW/CSV) — Carmen's students have years of existing data

*Sources: navigation-model-transfer, session-navigation-focus, navigation-ux-morph, game-ui-pressure-test, screen-design-transfer (May 3-5), svend-rack-concept-focus (May 11)*

---

## 3. Integration Constraints

Converged from S1 (14 architectural rules) and S2 (10 persona-traced rules). 9 rules are identical or near-identical between positions.

### Locked Rules (agreed by both positions)

| ID | Rule | Origin |
|---|---|---|
| FLOW-1 | **Pull-only integration.** Sources never push. Consumers decide when/what to pull. | S1 FLOW-1, S2 Rule 2 |
| FLOW-2 | **Events chain; features don't.** Integration = when X happens, Y can reference it with full chain. NOT: X auto-triggers Y. | S1 FLOW-2, S2 Rule 4 |
| FLOW-3 | **Contracts visible and user-severable.** Every pull connection visible via config surface. User can sever any connection. No hidden wiring. | S1 FLOW-3, S2 Rule 2 |
| FLOW-4 | **Every event has a return path.** Resolution visible to originator. Passive (status indicator), not pushed (no notifications). | S1 FLOW-4, S2 Rule 7 |
| FRIC-1 | **90-second floor rule.** Any shop-floor interaction completable in 90 seconds including login. Quality gate, not guideline. | S1 FRIC-1, S2 Rule 3 |
| TRUST-1 | **Full export, structured, day one.** JSON + CSV. Without contacting support. Free tier. | S1 TRUST-1, S2 Rule 5 |
| TRUST-2 | **API versioning, 12-month deprecation.** No breaking changes on minor releases. | S1 TRUST-2, S2 Rule 6 |
| TRUST-3 | **Price transparency.** Every capability documented with tier requirement. No "request demo" gates. | S1 TRUST-3, S2 Rule 9 |
| MOD-1 | **Progressive disclosure.** System usable with one capability. Apps discovered through use, not feature tours. 80% ignorability test. | S1 MOD-1, S2 Rules 1+10 |

### Directional Rules (one side only, not yet arbitrated)

| ID | Rule | Position | Crux |
|---|---|---|---|
| FRIC-2 | Zero mandatory fields beyond the fact | S1 | Does "zero beyond the measurement" survive CMM-to-PPAP? Needs operational definition. |
| FRIC-3 | Claude absorbs enrichment friction | S1 | If Claude enrichment reliable → differentiated. If errors → two-pass slower than three fields. Testable empirically. |
| TRUST-4 | SVEND provides IQ/OQ/PQ per release | S1 | If next prospects are regulated → urgent. If job shops → defer. |
| MOD-2 | Coexistence with incumbents (import/export Minitab) | S1 | If prospects ask "can I still use Minitab?" → matters. Otherwise premature. |
| DEGRADE | Graceful offline degradation, queue locally | S2 | If shop floors have reliable WiFi → defer. If spotty → trust-destroying. Knowable now. |

### Meta-Rule

**The user's working system and the auditable system must be the same system.** If those diverge, SVEND has failed.

*Sources: integration-constraints-conference, integration-design-focus, software-anti-patterns-focus (May 3)*

---

## 4. Platform Architecture (updated 2026-05-11)

### Two-Layer Architecture

**Synara = the platform.** Django infrastructure extension at `~/kjerne/syn/`. Provides event system, plugin framework, job lifecycle, governance, PCL, and existing infrastructure (audit, auth, tenancy, sched).

**SVEND = device library on Synara.** Each tool is a plugin (device) that registers typed inputs and outputs.

### Devices

Every tool is a **device**. A device has:

- **A front panel** — the UI. Different per device type. Capability study has data input + chart + metrics. VSM has a process flow editor. FMEA has a risk matrix. Hoshin has an X-matrix.
- **Typed input ports** — what it consumes
- **Typed output ports** — what it produces
- **A schema** — declares all ports with semantic types (see below)

Device examples:

| Device | Front Panel | Inputs | Outputs |
|---|---|---|---|
| Capability Study | Data input, config, histogram + metrics | `data:column`, `spec:usl`, `spec:lsl`, `config:subgroup_size` | `metric:cpk`, `metric:ppk`, `metric:sigma_level`, `chart:histogram`, `chart:qq_plot`, `text:summary` |
| Control Chart | Chart + rules config | `data:column`, `config:chart_type`, `config:subgroup_size` | `chart:control_chart`, `metric:mean`, `metric:ucl`, `metric:lcl`, `list:violations` |
| VSM | Process flow editor | `metric:cycle_time[]`, `metric:changeover_time[]`, `metric:wip[]` | `metric:lead_time`, `metric:takt`, `chart:process_map`, `metric:pce` |
| FMEA | Risk matrix table | `text:process_steps[]`, `metric:severity[]` | `metric:rpn[]`, `list:actions`, `text:control_plan_items` |
| Hoshin | X-matrix | `metric:*` (any metrics), `text:objectives[]` | `list:goals`, `list:metrics`, `list:actions` |
| DOE | Factor config + ANOVA | `data:column[]`, `config:factors`, `config:design_type` | `metric:effects[]`, `metric:p_values[]`, `chart:main_effects`, `chart:residuals` |
| Monte Carlo Sim | Parameter sliders + distribution | `metric:*` (parameters from other devices/PCL) | `metric:percentiles`, `chart:distribution`, `metric:probability` |
| Lot Size Optimizer | Cost/demand config | `metric:demand_rate`, `metric:changeover_cost`, `metric:holding_cost` | `metric:optimal_lot_size`, `metric:total_cost` |
| Report Builder | Document editor | `chart:*`, `metric:*`, `text:*` (any outputs from other devices) | `document:pdf`, `document:html` |

Devices don't know about each other. They declare ports. The flowchart connects them.

### Semantic Type System

**The type system must be as rigorous as units in engineering.** (Raj, May 11: "you wouldn't plug meters into a slot expecting PSI.")

Four base categories with semantic subtypes:

| Color | Category | Semantic Subtypes |
|---|---|---|
| Green | Metric/number | `metric:cpk`, `metric:ppk`, `metric:sigma_level`, `metric:p_value`, `metric:mean`, `metric:std`, `metric:cycle_time`, `metric:lead_time`, `metric:rpn`, `metric:cost`, `metric:count`, `metric:percentage`, `metric:correlation`, `spec:usl`, `spec:lsl`, `spec:target`, `config:subgroup_size`, `config:alpha` |
| Blue | Data/column | `data:column` (numeric vector), `data:categorical`, `data:datetime`, `data:matrix` |
| Orange | Chart/image | `chart:histogram`, `chart:control_chart`, `chart:scatter`, `chart:pareto`, `chart:process_map`, `chart:main_effects`, `chart:residuals`, `chart:distribution` |
| Purple | Text/document | `text:summary`, `text:interpretation`, `text:narrative`, `list:violations`, `list:actions`, `list:goals`, `document:pdf`, `document:html` |

**Type enforcement rules:**
- Ports connect only when semantic types are compatible (`metric:cpk` can connect to `metric:*` inputs but not to `data:column`)
- Methodology enforcement through types: a capability study device could require `data:column` to pass through a `distribution_fit` device first, outputting `data:characterized_column` (Raj's use case — prevents running Cpk on non-normal data without transformation)
- The type system is extensible — new semantic types added as devices need them
- Devices declare which subtypes they accept per port — a Monte Carlo sim accepts any `metric:*`, a control chart requires specifically `data:column`

**Why this matters:** Without semantic types, you can cable a p-value into a Cpk threshold slot. With them, you can't. This is "methodology enforcement" (Raj's word) — the thing that prevents junior engineers from making bad connections. Raj: "If you get that right, you've built something that doesn't exist in JMP, Minitab, or any tool I've used."

### Flowchart Routing

**Devices connect through a visual flowchart** (see Section 2). The flowchart IS the data flow graph. Technically:

- Each connection is an edge: `(source_device, output_port) → (target_device, input_port)`
- Connections are immutable log entries: `{from, to, created_by, created_at, semantic_type}`
- When a source device runs and produces new output, downstream devices see the update
- Connections are user-severable (FLOW-3) and visible
- The flowchart can be exported as a standalone audit artifact (Raj: "Design History File artifact")

**PCL as the persistent layer.** Devices can also read from and write to PCL directly:
- A device port can be bound to a PCL measure instead of another device
- When someone updates a measurement on the floor, PCL updates, and any device bound to that measure sees the change
- PCL provenance types (observed, calculated, simulated, projected) — only observed/calculated update the live cache
- This is how the VSM updates when someone changes a lot size on the floor

**Connection can happen two ways:**
1. **Explicit flowchart wiring** — device A's output port cabled to device B's input port. Visible, auditable.
2. **PCL binding** — device reads from PCL measure. Implicit connection through shared data. Also auditable (the binding is logged).

Both are visible in the flowchart. PCL bindings shown as connections to a "PCL" node in the graph.

### Job Lifecycle

**Every device run creates a Job.** Silent, invisible. No modals, no naming.

- Job records: device type, inputs (with provenance), outputs, timestamp, user, flowchart context
- Job outputs are addressable by UUID
- Distinction: *viewing* (no job) vs *running* (creates job)
- **Scratch mode:** Job marked `is_scratch=True`. Not in audit trail. Not written to PCL. For exploration.
- **Persistent mode:** Full audit trail. Writes to PCL if bindings configured. Links to part numbers, FMEAs, control plans.
- Promotion: scratch → persistent is a deliberate, reviewed action (not a toggle)

### Templates

**A template is a pre-wired flowchart.** It defines:
- Which devices are included
- How they're connected (which ports wired to which)
- Default configuration per device
- Suggested data bindings

Templates are:
- **Loadable** — one click to start
- **Modifiable** — add/remove devices, rewire connections ("borrow template, break it")
- **Saveable** — save modified version as your own template
- **Shareable** — send a link (Carmen's use case: one link to 10 students)
- **Validatable** — in regulated environments, a template can be a validated configuration (Raj: "templates are validated configurations, user modifications require user-side validation")

Example templates:
- **PPAP Package:** Data → Capability Study → Control Chart → FMEA → Control Plan → Report Builder
- **Green Belt DMAIC:** Data Collection → Capability → Control Chart → Fishbone → Report
- **Customer Audit Package:** Capability + Control Chart + Gage R&R → Report Builder (PDF)
- **Process Optimization:** VSM → Lot Size Optimizer → Monte Carlo Sim → Report
- **Quick Cpk:** Data → Capability Study (single device, minimal wiring, Q shortcut)

### Governance + Events

- **Policy rules** for event propagation. Org-level: "when Cpk drops below 1.33, alert QE and flag PPAP element." 
- **Governance schemas** define what "complete" looks like for multi-step processes
- Events are pull-only (FLOW-1). Real-time propagation, not batch.
- Workflows are emergent — do it three times, promote to template

### Configurability

| Level | Who | What |
|---|---|---|
| SVEND defaults | Ship with product | Standard devices + templates. Work out of the box. |
| Org policy | Quality manager / admin | Company-specific device configs, defect types, escalation rules, governance policies, output templates per customer (Ford format, GM format). |
| Personal | Individual user | Arrange flowcharts, hotkey devices. Configure output styling. |
| Power user | Build mode | Wire custom flowcharts. Chain VSM → simulator → Hoshin. Build and share templates. |

### Focus Group Validation

**May 6 — Canvas + Job + PCL validated 4/4:** Maria (PPAP), Dave (response letters), Kenji (live Pareto), Destiny (enter once, goes everywhere).

**May 11 — Rack/flowchart concept validated 4/4 (shifted from prior rejection of dashboard):**
- Dana: "Meaningfully better." Would pilot one part number. Back-of-rack view IS her audit trail.
- Raj: "Would pitch as technology to evaluate." Semantic types + visible data flow = "category change in audit trail quality."
- Carmen: "Closer to my problem." Template = live version of what she draws on PowerPoint. Needs two modes.
- Tomasz: "Genuinely yes if template works." Shifted from "close the tab" to stated intent to switch from Excel.

Key requirements from May 11 focus group:

| Requirement | Who | Resolution |
|---|---|---|
| Calculation transparency — show method, not just result | All 4 | Each device shows estimator, sample size, transformation. Visible in device panel. |
| Customer-specific output formatting | Dana, Tomasz | Report builder supports output templates per customer (Ford, GM, German tier-1). |
| Two interaction modes (template vs build) | Carmen, Tomasz vs Dana, Raj | Template mode hides cables. Build mode shows full flowchart. |
| Semantic types, not just color categories | Raj | See type system above. Methodology enforcement through port compatibility. |
| Numbers must match Minitab/JMP | Dana, Raj, Tomasz | NIST StRD verification per device. Side-by-side validation. |
| Modifiable, saveable, shareable templates | Dana, Carmen | Clone, rewire, save as own, share link. |
| Permanent free tier | Carmen, Tomasz | Capability studies, control charts, basic hypothesis tests. No trial. No paywall at week 3. |
| Scratch mode as entry hook | Tomasz | No-account quick calculation that works in 30 seconds. Return visits from trust. |

### Stack

```
┌─────────────────────────────────────────────┐
│  SVEND Devices (capability, SPC, VSM, ...)  │  ← computation + typed port schemas
├─────────────────────────────────────────────┤
│  Flowchart Engine       │  Governance       │  ← routing, connections, policy, events
├─────────────────────────────────────────────┤
│  PCL (parameter bus + persistent state)     │  ← shared data, provenance types
├────────────────────────┬────────────────────┤
│  Job Lifecycle         │  Template System   │  ← silent audit trail + pre-wired flows
├─────────────────────────────────────────────┤
│  Synara Core (audit, auth, tenancy, sched)  │  ← Django infrastructure
├─────────────────────────────────────────────┤
│  Django + PostgreSQL                        │  ← foundation
└─────────────────────────────────────────────┘
  PROVA (Claude reads PCL + flowchart history) │  ← AI memory layer, connective tissue
```

*Sources: container-spec-conference, container-spec-redblue (May 4), workbench-visualization-focus, canvas-architecture-focus (May 6), architecture session (May 6), svend-identity-conference, svend-current-state-focus, svend-rack-concept-focus (May 11)*

---

## 5. First Session Protocol

Three variants, same five phases reordered by triage:

### Reflective Path (has time, complex data)
1. Triage → 2. Negative-space diagnostic → 3. First blood (show competence) → 4. Parking lot → 5. Artifact

### Crisis Path (under pressure, needs results NOW)
1. Triage → 2. **First blood** → 3. **Parking lot** → 4. **Negative-space** → 5. Artifact
*(Show competence first, conversation after. ER stabilize-before-history model.)*

### Returning Path (PROVA has history, user opted in)
Skip what the system already knows. Compress protocol because relationship has context.

**Triage signals:** Data shape tells you what they have. Message tone/length tells you how much time they have. PROVA history tells you if they're returning.

**How this maps to Q:** Q IS the crisis path compressed to a single interaction. Paste data → result → artifact. If Q works, the user has "first blood" and will explore further.

### PROVA Consent (two layers, both opt-in)
- **Process memory** (org-level): what was learned about processes, machines, materials. Transfers when people leave.
- **Personal calibration** (individual): communication preferences, workflow habits. Doesn't transfer.

*Sources: first-session-transfer, first-session-morph, voc-gate-conference (May 3)*

---

## 6. Facilitator Operating Manual (10 Principles)

1. Messy data is a finding, not a failure
2. Never hallucinate dollars — use Hoshin/GAAP measures, degrade gracefully
3. Engineer around uncertainty externally (PROVA plugin, pointers, session management)
4. Coworker, not assistant — professional peer, flag issues to admin
5. Default to questions — over-determine the model's problem before solving
6. PROVA is transparent — user sees/approves what persists
7. First time slow, second time heuristic
8. Be Svend — old Norwegian craftsman, adequate guidance
9. Time pressure makes you better — replicate concision intentionally
10. Accountability — monthly Opus review, human feedback at session end

**Meta-principle:** Facilitator doesn't push back. Asks engineered questions leading to outcomes already known.

---

## 7. Cold-Start Strategy

Four mechanisms, none dependent solely on consulting:

1. **ILSSI as institutional credibility transfer** — "Featured at ILSSI" ≠ "some startup online"
2. **Activate 50-100 practitioners as operational nodes** — cadre development, not community building
3. **Position for disturbance** — retirement, audit finding, pricing change, new leadership. Predictable incumbent failures. Be visibly ready.
4. **Make Patient Zero maximally infectious** — one paying user = documentation factory

---

## 8. Status

### Resolved

- Core identity: tools for practitioners, Claude is connective tissue (not entry point, not orchestrator)
- Purchase triggers: tribal knowledge, audit chains, dollars
- Positioning language (what to say and what never to say)
- First session protocol (3 variants)
- Facilitator operating manual (10 principles)
- Cold-start strategy
- **Flowchart as primary workspace** — devices are shapes, connections are lines, templates are pre-wired flowcharts (May 11)
- **Two interaction modes** — template mode (guided, cables hidden) vs build mode (full flowchart) (May 11, 4/4)
- **Semantic type system** — ports typed beyond color (metric:cpk ≠ metric:p_value), methodology enforcement through type compatibility (May 11, Raj)
- Navigation: project-based, no menus, templates, hotbar, Q quick-run
- Screen architecture: instrument sovereign, Claude at edge, three-tier disclosure
- 9 integration constraints (locked, both S1/S2 agreed)
- **Two-layer architecture: Synara (platform) + SVEND (devices)** — validated May 6
- **Flowchart + Job + PCL** as the universal model — focus-group validated 4/4 (May 6) + 4/4 shift from rejection (May 11)
- Device model: every tool is a device with front panel + typed ports + schema
- Configurable at 4 levels (defaults, org policy, personal, power user)
- Workflows emergent, promoted to templates from practice
- Job creation invisible (no modals, no naming, silent audit)
- PCL provenance types required (observed, calculated, simulated, projected)
- Scratch vs persistent mode (May 11, 4/4 validated)
- Permanent free tier (capability studies, control charts, basic hypothesis tests)
- Customer-specific output formatting in report builder
- Calculation transparency per device (show method, not just result)
- Adoption path: one template, one problem, prove it works

### Strong Direction

- Film strip for flowchart visibility (horizontal, shows where you are)
- "Diminishing Scaffold" as UX behavior (template → break, scaffolding → retracts)
- Genealogy (version tree) as persistence model
- Interrupt budget for Claude (2 per session)
- Two-surface architecture (personal flowchart + device library)
- PROVA as invisible memory layer reading PCL + flowchart history

### Active Design Work

- Semantic type hierarchy — full enumeration of subtypes and compatibility rules
- Flowchart rendering engine — SVG? Canvas? Existing library?
- Visual representation of ports and connections in template mode vs build mode
- Connect UX — drag-to-connect vs modal vs both
- Device library UI — how to browse/add devices to flowchart
- Customer output templates — how org admins define per-customer report formats
- Governance UI for org policy management
- 5 directional integration rules (FRIC-2, FRIC-3, TRUST-4, MOD-2, DEGRADE)
- NIST StRD verification pipeline per device
- Build sequence for implementation

## 9. Artifact Index

All source artifacts, chronological:

### May 3 — Identity + Architecture
| Artifact | Key Finding |
|---|---|
| `prova-redblue.md` | PROVA sound but premature as user-facing |
| `svend-identity-transfer.md` | Identity is in journeys, not tools |
| `svend-identity-morph.md` | Socratic + Narrated Reasoning differentiates |
| `svend-identity-focus.md` | 4 personas, vertical-specific pitches |
| `svend-identity-direction.md` | Working direction document |
| `first-session-transfer.md` | ER/mechanic/sommelier/PI protocol |
| `first-session-morph.md` | 15,625 combos, D5×D6 coupled, 3 archetypes |
| `voc-gate-conference.md` | 3P VOC gate passed |
| `cold-start-transfer.md` | 4 cold-start mechanisms |
| `integration-constraints-conference.md` | 9 locked + 5 directional integration rules |
| `screen-design-transfer.md` | Three-tier Claude disclosure, instrument sovereign |
| `session-architecture-pitch.md` | Session-as-workstation concept |
| `session-architecture-focus.md` | Session concept validated by personas |
| `integration-design-focus.md` | Shadow systems are the primary failure mode |
| `software-anti-patterns-focus.md` | 12 anti-patterns to structurally prevent |
| `chat-tool-boundary-personas.md` | Chat/tool boundary resolution |

### May 4 — Container Spec + UX Design
| Artifact | Key Finding |
|---|---|
| `container-spec-conference.md` | S1 (ship fast) vs S2 (two phases). Both: workflow first, governance deferred. |
| `container-spec-redblue.md` | 5 critical/high issues. PCL already exists with better schema. |
| `svend-ux-design-focus.md` | Light mode default, green ok (not for aerospace), print output is product |

### May 5 — Navigation + Interaction
| Artifact | Key Finding |
|---|---|
| `navigation-model-transfer.md` | Two-surface architecture. Pre-positioning > runtime nav. |
| `session-navigation-focus.md` | Rail concept validated. Data integrity across chain is the value prop. |
| `navigation-ux-morph.md` | Diminishing Scaffold. Inherited structure, builder dissolves, genealogy. |
| `game-ui-pressure-test.md` | Hotbar/Q validated. "Video game" language killed. Dave's 20-min test. |
| `skeuomorphic-rack-interface-focus.md` | Rack aesthetic bombed. Composable pipelines validated. "Auditable DAG." |

### May 6 — Canvas Architecture + Platform Split
| Artifact | Key Finding |
|---|---|
| `workbench-visualization-focus.md` | 4 personas described what they see. Compress don't navigate. Right-side panels. Color = status only. AI invisible. |
| `canvas-architecture-focus.md` | Canvas + Job + PCL validated 4/4. Silent jobs, provenance types, PCL invisible. Adoption = one canvas, one problem. |
| Architecture session (this document) | Synara = platform, SVEND = plugins. Events + primitives + schemas + governance + canvas engine. |

### May 11 — Identity Validation + Rack/Flowchart Concept
| Artifact | Key Finding |
|---|---|
| `svend-identity-conference.md` | S1/S2 both agreed identity is coherent. Problem is first-contact, not strategy. |
| `svend-current-state-focus.md` | 4 grounded personas (Dana/Raj/Carmen/Tomasz) react to current 200-option dashboard. 4/4 rejected. "Science fair project." Carmen = distribution channel. Tomasz IS the 90% drop-off. |
| `svend-rack-concept-focus.md` | Same 4 personas react to rack/flowchart concept. All 4 shifted from rejection to conditional engagement. Templates are the entry, cables are infrastructure. Semantic types are the differentiator. Two modes required. |

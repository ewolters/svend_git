# SVEND Integration Architecture Specification

**Status:** DRAFT — decided constraints + open questions
**Date:** 2026-05-03
**Evidence:** Two focus groups (8 personas), SVEND identity sessions, S1/S2 conference
**Artifacts:** `docs/superpowers/innovate/2026-05-03-integration-design-focus.md`, `docs/superpowers/innovate/2026-05-03-software-anti-patterns-focus.md`, `docs/superpowers/innovate/2026-05-03-integration-constraints-conference.md`

---

## Meta-Rules

**MR-1: The working system and the auditable system must be the same system.**
If users maintain a shadow system (paper, Excel, personal scripts) alongside SVEND, SVEND has failed. Integration design is judged by whether it prevents this divergence.

**MR-2: Works WITHOUT Claude, BETTER with Claude.**
Every constraint, workflow, and integration must function when Claude is unavailable, slow, or wrong. Claude is the accelerant, not the load-bearing wall.

---

## Decided Constraints

### Flow Architecture

**FLOW-1: Pull-Only Integration**
Switches and sinks pull from upstream sources. Sources never push. No automatic escalations, no mandatory downstream propagation. The consumer decides when and what to bring in.

*Anti-patterns blocked: Alarm Fatigue Theater, mandatory fields nobody fills*

**FLOW-2: Event Traceability, Not Feature Integration**
Integration = when X happens in one context, Y can reference it with the full chain intact. It does NOT mean when X happens, Y automatically fires. Events are immutable facts with typed references. Any downstream tool can pull them. Triggering is human or Claude-facilitated, never automatic.

Contracts define: (a) triggering event, (b) data payload, (c) destination action. Contracts are event-to-action mappings, not data-sharing agreements.

*Anti-patterns blocked: Monolith Misfit, Shadow System Inevitability*

**FLOW-3: Visible, User-Severable Contracts**
Every active pull connection is visible in a config surface the user controls. Activation requires explicit user action. Deactivation equally easy. The user must be able to answer "what is feeding this view?" in under 5 seconds.

*Anti-patterns blocked: Data Roach Motel, Compliance-As-Retention*

**FLOW-4: Return Path to Originator**
When a source emits an event that a downstream consumer acts on, the resolution is visible to the originator. Passive, not pushed — status indicator on prior observations, seen on next interaction. No notifications, no emails, no popups.

Minimum status vocabulary: received, in progress, resolved.

*Anti-patterns blocked: Alarm Fatigue Theater (NCR black hole)*

### Friction Budget

**FRIC-1: 90-Second Floor Rule**
Any shop-floor-facing interaction must complete in under 90 seconds, including authentication, navigation, data entry, and confirmation. This is a quality gate — if exceeded, the workflow ships broken. Analytical workflows (DOE config, FMEA population) have different latency budgets.

*Anti-patterns blocked: Office-Floor Gap, Buyer-User Split*

**FRIC-2: Zero Mandatory Fields Beyond the Fact** *(design rule)*
At point of capture, only the fact itself is required. Everything else (root cause codes, categories, severity, disposition) is optional at capture and enrichable later.

"The fact" is defined per workflow, not globally:
- SPC measurement: value + part identifier
- NCR: what's wrong + where
- CAPA: nonconformance reference
- Complaint: lot + description

Auto-populated context (timestamp, operator, machine) does not count against the field budget.

*Anti-patterns blocked: Office-Floor Gap, Shadow System Inevitability*

**FRIC-3: Enrichment Is Async, Never at Capture** *(design rule)*
Categorization, severity estimation, routing, and linking to prior events happen after capture, not during. Claude can perform this enrichment when available. When Claude is unavailable, the quality engineer enriches manually — the workflow still works, just slower.

This is a data model constraint: capture tables have minimal required fields. Enrichment lives in separate fields or linked records, tagged with source (claude / user / system).

*Anti-patterns blocked: Power-User Ceiling (forces simple capture, enables rich analysis)*

### Trust Architecture

**TRUST-1: Full Structured Export, Any Time, Any Tier**
Every model with user-created data exposes bulk export in JSON + CSV. Export includes relationships (contract links, event chains), not just flat records. Available from day one on the free tier. No support ticket, no contractor. Tested in CI.

*Anti-patterns blocked: Data Roach Motel, Compliance-As-Retention*

**TRUST-2: API Versioning with 12-Month Deprecation**
Versioned endpoints in URL path. No breaking changes on minor releases. 12-month minimum deprecation notice. Schema published and versioned alongside API docs. No "minor update" may alter the shape of exported data. Release notes state whether re-validation is required.

*Anti-patterns blocked: Validation Tax*

**TRUST-3: Price Transparency, No Feature-Tier Hiding**
Every capability documented with tier requirement on a public page. No "request a demo" gates on pricing. Feature gating (`@gated` decorator system) auditable against published pricing. Exit cost documented alongside pricing.

*Anti-patterns blocked: Invisible Cost Accounting, Renewal Trap, Seat-Count Mismatch*

### Modularity

**MOD-1: Single-Problem Entry (the "SAP Firewall")**
No deployment path requires configuring more than one app. Every app functions with zero contracts configured. Contracts are additive enrichment, never prerequisites.

Progressive disclosure at the navigation level — unconfigured apps are completely invisible. No menu items for unused modules. No onboarding wizard walking through all capabilities. Claude may suggest adjacent tools when work demands it ("your Cpk dropped below 1.33 — want to open an investigation?") but the app chrome never displays unused tools.

*Anti-patterns blocked: Monolith Misfit, Buyer-User Split*

**MOD-2: Coexistence, Not Replacement** *(design rule)*
SVEND runs alongside existing tools indefinitely. Accept data from incumbents (import), produce data incumbents can read (export). Never require the user to stop using their current tool. Import/export is a first-class concern in every module, not an afterthought.

This is a design rule governing how modules are built — not a commitment to build specific adapters for specific tools. Adapters are built when customer demand justifies them.

*Anti-patterns blocked: Compliance-As-Retention*

---

## Deferred — Trigger Conditions Defined

### INFRA-1: Offline / Graceful Degradation (the "7 PM Rule")

**What:** Shop-floor workflows define degradation behavior when connectivity fails. Queue locally, sync when available, timestamp accurately. System never loses data because network was down.

**Deferred because:** Requires actual infrastructure — local queuing, sync resolution, conflict handling. Not a design rule.

**Trigger:** First deployment on a shop floor with unreliable connectivity, OR first customer on 2nd/3rd shift without IT support. Knowable from sales conversations.

**Risk of deferral:** If early deployments include spotty-WiFi shop floors, operators revert to paper immediately and SVEND becomes the compliance layer. Silent failure — Marcus Wade won't complain, he'll just stop using it.

### INFRA-2: Validation Documentation (IQ/OQ/PQ per Release)

**What:** SVEND provides IQ/OQ/PQ documentation with each release so regulated customers don't bear full re-validation burden. Generated from test suite and CHG-001 change management system.

**Deferred because:** Requires real process and possibly tooling. The change management system (CHG-001) exists and could generate this, but the documentation templates and per-release workflow don't exist yet.

**Trigger:** First prospect in FDA/ISO 13485/IATF 16949 environment who asks "where's your validation package?" during evaluation.

**Risk of deferral:** David Kwon's evaluation timeline becomes 6-12 months instead of 60-90 days because his team must build validation documentation from scratch. First regulated customer triggers a fire drill.

---

## Open Questions

### Q1: The Chat/Tool Boundary — RESOLVED (2026-05-03)

**Resolution: Direct tool for doing, conversation for thinking.**

The app is the primary interface. Claude is the expert down the hall. 4/4 personas drew this line independently when asked directly.

**Pitched to:** Marcus Wade (floor), Greg Linden (quality director), Priya Chakraborty (solo QM), James Okafor (power user/Black Belt). All drew the same boundary without seeing each other's responses.

**The rule:**

| Mode | Owns | Examples |
|---|---|---|
| **Direct tool** (always-on, visual, forms) | Doing, recording, viewing, approving | Live SPC charts, data entry forms, NCR logging, PPAP assembly, document approval, dashboards, batch API calls |
| **Conversation** (on-demand, when user initiates) | Thinking, investigating, correlating, preparing | "Why did Cpk drop?", cross-system investigation, audit prep (night before, not during), historical pattern retrieval, report narrative drafting, "what changed?" |

**What this means architecturally:**
- Claude is NOT a Switch in Source/Switch/Sink. Claude is an enrichment/investigation layer.
- The app routes through its own UI. Contracts route through app logic, not through Claude.
- Claude is available when the user has a question that charts and forms can't answer.
- Every workflow must have a direct-tool path. Claude makes it better, never required.
- This is fully consistent with MR-2 (works WITHOUT Claude, BETTER with Claude).

**Specific findings by persona:**

- **Marcus Wade (floor):** "Give me a screen that's as fast as paper — always on, zero interaction for the normal case. Then give me the AI for the three or four times a shift when I actually have a question that a chart can't answer." Voice barely realistic (85-90 dB, coolant, gloves). Push-to-talk with short commands possible. Chat window = no.
- **Greg Linden (quality director):** "Build me the tool my people can see and click. Make the output dead-on accurate and traceable. Then put the AI behind a door my engineers can open when they need to dig into something." 58-year-old inspector will not type to an AI. 7am scrap meeting needs a pre-built dashboard, not a conversation. Does NOT trust AI-generated PPAP without seeing data, spec limits, and source — needs a validation period.
- **Priya Chakraborty (solo QM):** "Build the QMS first. Make the AI the layer that makes me faster." During SQF audit: search bar + document tree, NOT a chat window ("looks like I don't know where my own records are"). Weekend trial evaluates forms, not conversation. Line leads need checkboxes, not AI.
- **James Okafor (Black Belt):** Primary interface = programmatic API. 200-parameter batch run is a script call, not a conversation. Conversation for investigation ("why did this drop?"), API for execution, GUI for training Green Belts. Needs computation trace (method, assumptions, diagnostics) — black box "Cpk = 1.45" is worthless.

**The "mechanic" metaphor reframed:** "Users talk to the mechanic, not to the torque wrench" is still true — but the mechanic doesn't stand between you and the dashboard. The mechanic is in the shop when you have a question. The gauges are always visible. You read the gauges yourself. You go to the mechanic when the gauges show something you don't understand.

**Audit trail implication (Greg, explicit):** "IATF 16949 auditors want to see who did what, when, on what screen, with what data. 'I asked the AI and it did it' is going to get me a major nonconformance finding." This means: AI-assisted actions must produce the same audit trail as direct actions. The record shows what was done, not that the AI was asked to do it.

**Evidence artifact:** `docs/superpowers/innovate/2026-05-03-chat-tool-boundary-personas.md`

### Q2: How Current Architecture Maps to These Constraints

**The problem:** SVEND already has ~10 apps with existing integration patterns. Some may already satisfy these constraints. Some may violate them. The gap between current state and this spec is unknown.

**What needs resolving:** An audit of existing apps against these constraints. Which are already compliant? Which need work? How much work? This determines whether the spec is "write it down and keep building" or "retrofit required."

### Q3: PROVA and PCL's Relationship to These Constraints

**The problem:** PROVA (Claude's invisible memory) and PCL (process characteristics substrate) are designed but paused. These constraints reference them implicitly (FRIC-3's enrichment reads PCL context; FLOW-2's event traceability is what PROVA would accumulate). But PROVA's current status is "valid math, changed interaction model, paused."

**What needs resolving:** Whether PROVA/PCL are prerequisites for these constraints or independent systems that benefit from them. If independent, the constraints stand alone and PROVA plugs in later. If prerequisite, the constraints have a hidden dependency on unbuilt infrastructure.

**Current read:** Independent. The constraints describe how apps talk to each other. PROVA describes how accumulated knowledge is structured. They're orthogonal — but this should be confirmed, not assumed.

### Q4: The "Analytical Provenance" Emergence Question

**The problem:** S2's steel-man identified this: conservative pull-only constraints may produce ten excellent but disconnected apps that never produce the emergent reasoning chain. Analytical provenance — the chain connecting a process signal to a business decision — is inherently cross-app. It may require something beyond pull-only event traceability to emerge.

**S1's position:** The chain forms as a byproduct of event traceability (FLOW-2). If every event references its predecessors, the chain exists.

**S2's position:** The chain forms from usage, but only if something accumulates it. Pull-only means nothing pulls unless someone asks.

**What needs resolving:** Whether FLOW-2 (events as immutable facts with typed references) is sufficient for the reasoning chain to be reconstructable on demand, or whether an active accumulator (PROVA, or something simpler) is needed. This may not be answerable until the first few event chains exist in production.

---

## Anti-Pattern Coverage Matrix

| Anti-Pattern | Blocked By | Status |
|---|---|---|
| 1. Buyer-User Split | FRIC-1, MOD-1 | Decided |
| 2. Alarm Fatigue Theater | FLOW-1, FLOW-4 | Decided |
| 3. Compliance-As-Retention | FLOW-3, TRUST-1, MOD-2 | Decided |
| 4. Shadow System Inevitability | MR-1, FRIC-2, FLOW-2 | Decided |
| 5. Validation Tax | TRUST-2, INFRA-2 | Partial — INFRA-2 deferred |
| 6. Seat-Count Mismatch | TRUST-3 | Decided |
| 7. Data Roach Motel | TRUST-1, FLOW-3 | Decided |
| 8. Monolith Misfit | MOD-1, FLOW-2 | Decided |
| 9. Power-User Ceiling | FRIC-3 (async enrichment enables API access) | Decided (design rule) |
| 10. Invisible Cost Accounting | TRUST-3 | Decided |
| 11. Renewal Trap | TRUST-3 | Decided |
| 12. Office-Floor Gap | FRIC-1, FRIC-2 | Decided |

---

## Evidence Chain

Every constraint traces to named personas from VOC:

| Constraint | Primary Persona Evidence |
|---|---|
| FLOW-1 | Greg (mandatory fields), Dana (400 alarms) |
| FLOW-2 | David ("traceability gap BETWEEN systems"), Priya ("pull the full chain") |
| FLOW-3 | Rachel ("what does it cost to leave on day one?"), David ("can I see your schema?") |
| FLOW-4 | Marcus Wade ("NCRs go into a folder"), Dana ("400 alarms, 10 responses") |
| FRIC-1 | Marcus Wade ("90 seconds"), Dana (4-7 min vs 30 sec), Greg (58-year-old inspector) |
| FRIC-2 | Dana (batch entry, wrong timestamps), Greg ("mandatory fields nobody fills") |
| FRIC-3 | James ("running analysis produces the auditable record"), SVEND identity (facilitator model) |
| TRUST-1 | Rachel ("four sentences nobody says"), Marcus Chen ("2,000 hours to re-index") |
| TRUST-2 | David (Epicor broke lot traceability, 40 hrs to fix) |
| TRUST-3 | Priya ("request a demo = already annoyed"), Rachel (invisible cost accounting) |
| MOD-1 | 4/4 integration group ("solve one problem, not ten"), Greg ("it was called SAP") |
| MOD-2 | Greg ("side-by-side with Minitab"), James (dual R/Minitab workflow) |

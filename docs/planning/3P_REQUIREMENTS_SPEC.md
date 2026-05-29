# SVEND 3P Requirements Specification

> **3P Stage:** VOC Complete → Requirements → Design (next)
> **Date:** 2026-05-04
> **Sources:** 20 innovation artifacts (2026-05-03), 12+ personas across 3 focus groups, 6 innovation techniques, existing design specs, founder decisions (2026-05-04)

---

## 1. System Identity

SVEND is an AI-guided operational excellence implementation system. The tools do the computation. The AI provides the judgment. The knowledge accumulates automatically as a byproduct of work.

**Not:** a tools platform, a chatbot bolted onto analytics, an AI wrapper, or a knowledge management system.

---

## 2. Four Pillars + Facilitator

The system has four structural components and one facilitation layer.

### 2.1 Workflows

Reusable pipeline definitions that capture action sequences. Every interaction — from a complex DOE to "pull CSV, route column 3 to capability analysis" — gets logged as a workflow. Workflows are retained and become heuristics.

**Requirements:**
- REQ-WF-1: Every action sequence is captured as a workflow, including data source, column routing, analysis selection, parameters, and output format.
- REQ-WF-2: Retained workflows are searchable and proposable by Claude. "Last time you pulled this CSV and routed column 3 to capability with USL 0.05 — same setup?"
- REQ-WF-3: Workflows that lead to good outcomes get reinforced (Synara confidence increases). Workflows that get overridden or produce poor outcomes decay.
- REQ-WF-4: Generalized workflows become marketplace items. User-specific workflows remain private.
- REQ-WF-5: Pre-built workflow templates ship with the system. Cpk-to-PPAP, incoming inspection, CAPA chain, scrap Pareto, check weigher analysis. **Non-negotiable** (4/4 session architecture personas).
- REQ-WF-6: Users customize templates, never build from scratch as default path. Building from scratch is the power-user path.
- REQ-WF-7: Workflow definitions are JSON, diffable, Git-storable, code-reviewable (David Kwon).
- REQ-WF-8: Byte-for-byte reproducibility of results given the same input data (David Kwon).

**Source:** Session architecture focus group (Greg, Priya, David, Marcus), founder decision 2026-05-04.

### 2.2 Instances (Name TBD)

Running instances of workflows. The actual work happening now.

**Requirements:**
- REQ-IN-1: An instance is a workflow execution with bound data, parameters, and results.
- REQ-IN-2: Every instance produces a log — the provenance trail of what happened and why.
- REQ-IN-3: Instances can be saved, resumed, shared, and versioned.
- REQ-IN-4: Validated state flag — lock an instance version for re-qualification (David Kwon).
- REQ-IN-5: Side-by-side old/new during re-qualification (David Kwon).
- REQ-IN-6: Branching chains — one NCR to two CAPAs, one complaint to three lots (Priya Chakraborty).

**Naming:** Focus groups rejected "session" (4/4). Candidates from personas:
| Name | Source | Context |
|------|--------|---------|
| Setup | Greg's Dave, Marcus Wade | "Pull up the setup for that part" |
| Runsheet | Marcus Wade | "Pull up the runsheet for the F-150 bracket" |
| Job | Greg Linden | "The F-150 bracket job" |
| Workbook | Greg Linden | Familiar from Excel |
| Case / Case file | Priya Chakraborty | Auditor-friendly |
| Workflow | David Kwon | Technical, accurate |

**Decision needed:** Pick one or support contextual aliases.

### 2.3 Synara (Governance)

Rules engine with Bayesian confidence from outcomes. Mediates all contracts, access, and routing.

**Requirements:**
- REQ-SY-1: Rules that fire and lead to good outcomes get reinforced. Rules that get overridden decay. Bayesian confidence from actual usage.
- REQ-SY-2: Heuristics emerge from retained workflows. The system learns what to suggest and what to skip.
- REQ-SY-3: Synara mediates all reads/writes to PCL. Source declares what it's writing and provenance. Consumer declares what it needs and time context.
- REQ-SY-4: All contracts are visible and user-severable (FLOW-3, integration constraints conference).
- REQ-SY-5: Pull-only integration. Sources never push (FLOW-1).
- REQ-SY-6: Event traceability, not feature integration. Events stored as immutable facts with typed references (FLOW-2).
- REQ-SY-7: Every event has a return path. Resolution visible to originator (FLOW-4).
- REQ-SY-8: CHG-001 governance for heuristic changes — create ChangeRequest for rule modifications.
- REQ-SY-9: Confidence scores on every suggestion come from Synara's rule tracking, not Claude's self-assessment.

**Existing infrastructure:** `~/kjerne/syn/` — core models, audit, logging, API middleware, error hierarchy, scheduler, active defense, belief engine. Redesign scope: strip to governance engine, add workflow-confidence tracking.

### 2.4 PCL (Process Characteristics Library)

The data substrate. Measures with meaning. What Claude reads to understand current process state.

**Requirements:**
- REQ-PCL-1: Two measure kinds — Raw (stored datapoints with timestamps) and Calculated (formula referencing other measures, no stored values).
- REQ-PCL-2: Confidence auto-computed from source_type x observation_count. Never user-assigned.
- REQ-PCL-3: Confidence hierarchy: automated > workbench analysis > time study > manual > estimate.
- REQ-PCL-4: Two layers — Operational (current reality) and Working (future state targets).
- REQ-PCL-5: Measure fields: name, definition, unit, measure_type (process/material/product/resource), value_type (continuous/discrete/proportion/integer), realistic_range, formula (optional).
- REQ-PCL-6: Datapoint fields: value, timestamp, source_type, source_ref (FK to origin), observation_count, notes.
- REQ-PCL-7: Synara mediates all reads/writes (REQ-SY-3).
- REQ-PCL-8: Writers: Workbench, SPC, DOE, Calculators, Manual entry, QMS, VSM.
- REQ-PCL-9: Readers: VSM, Hoshin, Claude (via PROVA/workflows), Calculators, QMS, Monte Carlo.
- REQ-PCL-10: VSM Approach C — steps default to inline values, optional PCL binding. Zero migration cost.

**Existing implementation:** Deployed at `~/kjerne/pcl/` with migration 0001_initial (2026-05-01). Three models: Measure (SynaraEntity, slug-based, formula `[slug]` references, decay, cached aggregates), Datapoint (SynaraImmutableLog, hash-chained, auto-confidence from log2 curves), MeasureTarget (aspirational/working layer). Plus aggregation engine, formula evaluator, service layer, confidence computation, API endpoints. **Spec work is diff/merge — extend what's deployed, not rebuild.**

### 2.5 Claude (Facilitator)

Claude is hooked into the system via plugins. Reads PCL + workflow history + Synara rules. Modifies the UI directly — toasts, suggestions, pre-filled options, flagged anomalies, workflow proposals.

**Requirements:**
- REQ-CL-1: Claude searches heuristics in background and proposes when user is reinventing the wheel.
- REQ-CL-2: Claude affects the view the user sees — toasts, option panels, guided steps, pre-fills. Not chat-first.
- REQ-CL-3: Chat is optional, available when user wants to talk through something. Default interaction is: you work, the system guides.
- REQ-CL-4: Claude reads work product (instrument state), not user intent. Reasons from what's on screen (screen design transfer: "advisor reads output, not intent").
- REQ-CL-5: Pre-staged intelligence — Claude computes speculatively in background. User query = cache retrieval, not computation trigger (military C2 transfer).
- REQ-CL-6: Claude absorbs enrichment friction — categorization, severity, routing redirected from user to Claude. Claude reads raw observation and enriches (FRIC-3).
- REQ-CL-7: Per-characteristic explanation density. First out-of-control signal gets full explanation. Familiar measures get minimal narration (TRIZ P3).
- REQ-CL-8: Session-end knowledge distillation — discard conversation, recover delta: new theories, updated confidences, invalidated assumptions. Compress into workflow/heuristic updates (TRIZ P34).
- REQ-CL-9: Claude can find, organize, draft, and route. Cannot decide, approve, or certify (Priya's line).
- REQ-CL-10: Human confirms every dollar figure and audit claim (VOC gate, steel-man response).
- REQ-CL-11: The facilitator plugin is a JSON that tells Claude how to be a facilitator, chunked and hooked. Same architecture as innovate/workstream plugins.
- REQ-CL-12: "Gemba walk as onboarding" — user teaches Claude about their process, Claude silently maps to OpEx tools, populates PCL from casual language (TRIZ P13).

---

## 3. Screen Architecture

Single view, divided top and bottom. Instrument surface + command strip. Depth via marketplace add-ins and plugins (rack concept).

### 3.1 Layout (Validated 2026-05-04 — 4/4 focus group approved single-pane)

- REQ-SC-1: Single view with top/bottom split. Instrument surface (work) on top, command strip (controls) on bottom. **Validated:** all four personas accepted or preferred single-pane. Multi-monitor is compensation for dumb tools, not an analytical requirement.
- REQ-SC-2: Depth managed through marketplace add-ins and plugins. Simple analysis = pull data + tell Claude or use DSL or use interface widgets that instantiate in the workpane.
- REQ-SC-3: Execution surface sovereignty — never obscured, only compressed (4/4 screen design transfer).
- REQ-SC-3a: **Home base.** Each user context has a default view the system returns to. Floor: SPC chart for loaded job. Office: last working tool. The system has a gravity well — nothing displaces it without explicit user action.
- REQ-SC-3b: **Pin panel.** The advisor panel slot (same as Claude's slide-in) can pin reference data (CMM output, spec sheet, linked record) alongside the working tool. Not a second workspace — one fixed reference alongside one working area. Required for QE adoption (Greg: without this, QEs open separate browser windows and copy-paste).
- REQ-SC-3c: **State preservation.** Switching tools preserves full state (scroll position, zoom, annotations, form data, filters). Returning after interruption = exactly where you left off. Universal requirement (3/4 independently named it).
- REQ-SC-3d: **Snapshot/freeze.** Save a view state for later comparison. Power users compare ten outputs asynchronously, not two outputs side-by-side.
- REQ-SC-3e: **Sub-second pane switching.** Pane transitions must feel instant, not like page loads. Deal-breaker for power users (James: "> 2 seconds and I'm back in RStudio by lunch").
- REQ-SC-3f: **Thin command strip on floor.** Floor tablets (10"): command strip minimal — thin bar, expands on tap. 20% screen consumption = 20% less chart = unacceptable.

### 3.2 Three-Tier Progressive Disclosure

- REQ-SC-4: Ambient tier (~0 sec) — peripheral signals, color changes, stale-data indicators, status badges. Marcus's orange border for stale data. NCR-247 resolved badge.
- REQ-SC-5: Directed tier (~15 sec) — tap data element, get pre-loaded brief, auto-dismiss. Marcus's one-tap flag: info appears, read in 10 seconds, dismiss. 5-line brief, no statistics lecture.
- REQ-SC-6: Deliberative tier (~60+ sec) — full conversation or deep analysis. Optional chat. Multi-turn investigation.
- REQ-SC-7: Gravity-based disclosure — panels spring-loaded, require force to hold open, naturally collapse. Auto-dismiss (trading floor transfer).

### 3.3 Claude in the UI

- REQ-SC-8: Claude's output appears as instrument-native format — pre-populated fields, suggested options, toast notifications, flagged values. Not prose in a sidebar.
- REQ-SC-9: Context tunneling — working in the instrument IS the query. Touching a data element generates the query implicitly (clinical transfer).
- REQ-SC-10: Ambient evidence of activity — collapsed advisor shows signs of life (annotations appearing, indicators updating). Builds trust Claude is engaged (military C2 transfer).
- REQ-SC-11: Compressed protocol vocabulary — 90% of interactions use standardized micro-actions (tap, dismiss, accept suggestion), not natural language.

### 3.4 Floor vs. Office

- REQ-SC-12: Floor mode — green/yellow/red glanceable. No browser, no login, no dropdown. Badge tap or machine login. Auto-capture from CMM (Marcus Wade, Dana Kowalski).
- REQ-SC-13: Office mode — full workpane with analysis tools, chain view, workflow builder.
- REQ-SC-14: Line leads see forms that look like their paper forms on a tablet. Never see chain view or the word "session" (Priya).
- REQ-SC-15: Show linear view by default, graph view on demand. "They think in steps, not wires" (Greg).

---

## 4. Hard Constraints (Non-Negotiable)

These are veto criteria. Violating any one kills adoption.

### 4.1 Friction Budget

| ID | Constraint | Source |
|----|-----------|--------|
| HC-1 | 90-second floor rule. Any shop-floor interaction completable in 90 seconds including login, navigation, data entry, confirmation. | Integration constraints (FRIC-1), Dana, Marcus Wade |
| HC-2 | Zero mandatory fields beyond the fact. Part number, timestamp, operator auto-populated. Everything else optional at capture, enriched later by Claude or QE. | Integration constraints (FRIC-2) |
| HC-3 | First results in under 30 minutes or user abandons. 30-45 minutes to first useful output is the hard threshold. | Tameka Jackson, pricing validation |
| HC-3a | Access first, demo call never. Upload CSV → see result → then optional human call. No "request a demo" gate. | Pricing/onboarding validation |
| HC-4 | Learning curve: 2 hours max for inspectors. 58-year-old inspector must learn in a morning. | Ray Nguyen, Greg Linden |

### 4.2 Trust Architecture

| ID | Constraint | Source |
|----|-----------|--------|
| HC-5 | Full structured export at any time, any tier. JSON + CSV with relationships. Export button from day one on free tier. No data hostage. | Integration constraints (TRUST-1), Ray, Rachel, Priya |
| HC-6 | API versioning with 12-month deprecation guarantee. No breaking changes on minor releases. | Integration constraints (TRUST-2), David Kwon |
| HC-7 | Price transparency on public page. No feature-tier hiding. No "request a demo" gates on pricing. | Integration constraints (TRUST-3), all personas |
| HC-8 | Validation burden stays with SVEND. IQ/OQ/PQ documentation provided per release, generated from test suite and CHG-001. | Integration constraints (TRUST-4), Marcus Chen, James Okafor |
| HC-9 | Month-to-month with kill clause. Price lock option. No renewal trap. | Rachel Torres, Ray Nguyen |
| HC-10 | Cancel demo showing full data export. | Ray Nguyen |

### 4.3 Modularity

| ID | Constraint | Source |
|----|-----------|--------|
| HC-11 | Single-problem entry (SAP Firewall). No deployment requires configuring more than one app. Contracts are additive, never prerequisites. | Integration constraints (MOD-1), all personas |
| HC-12 | Coexistence, not replacement. Run alongside Minitab indefinitely. Accept and produce data Minitab can read. | Integration constraints (MOD-2), Dave Kowalski, James Okafor |
| HC-13 | 80% ignorability test. Unconfigured apps completely invisible. No menu items, no empty dashboards, no onboarding wizards for things the user hasn't asked for. | Integration constraints |
| HC-14 | Must replace at least one system, not add system #8. | David Kwon, Rachel Torres |

### 4.4 Audit & Provenance

| ID | Constraint | Source |
|----|-----------|--------|
| HC-15 | Audit trail immutable or append-only. | Marcus Chen |
| HC-16 | Reproducible analytical chain — "the model suggested" is NOT acceptable. | Marcus Chen |
| HC-17 | Numbers must match Minitab exactly. Different answer without clear explanation = trust goes to zero permanently. | Dave Kowalski, James Okafor, Greg Linden |
| HC-18 | Analytical provenance: "why this test, what assumptions, what uncertainty" answerable without reconstruction. | James Okafor, Marcus Chen, Rachel Torres |
| HC-19 | Human confirms every dollar figure and audit claim. Claude guides, doesn't decide. | VOC gate (steel-man), Priya |
| HC-20 | Confirmation before filing. Show AI-generated NCR for 3 seconds before submitting. | Marcus Wade |

### 4.5 Infrastructure

| ID | Constraint | Source |
|----|-----------|--------|
| HC-21 | Graceful degradation (7 PM rule). Queue locally, sync when available, timestamp accurately. Never lose data because network was down. | Integration constraints (Rule 8), Dana |
| HC-22 | Data must still be there tomorrow. | Dana Kowalski |
| HC-23 | No-password entry for floor users. Badge tap, machine login, shop network auth. | Marcus Wade, Dana |
| HC-24 | NCR feedback loop. Every signal produces visible status update: received, in progress, resolved. | Integration constraints (Rule 7), Marcus Wade |

---

## 5. Meta-Rules

| ID | Rule | Source |
|----|------|--------|
| MR-1 | The working system and the auditable system must be the same system. Shadow systems are the primary failure mode. | Integration constraints, anti-patterns focus |
| MR-2 | Works WITHOUT Claude, BETTER with Claude. Every workflow has a direct-tool path. | Integration spec, identity direction |
| MR-3 | AI-assisted actions produce the same audit trail as direct actions. | Integration spec |
| MR-4 | Every constraint must work without Claude and be better with Claude. | Integration constraints |
| MR-5 | "Rules are free, judgment is Claude." Computation and governance are the free tier. AI facilitation is the paid tier. | Founder, VOC gate |
| MR-6 | Lead with the chain/traceability, not the nodes. Lead with templates, not blank canvas. Never lead with AI. | Session architecture (4/4) |

---

## 6. Anti-Patterns (Must Not Replicate)

12 named anti-patterns from the software anti-patterns focus group. Each is a checklist item for design review.

| # | Anti-Pattern | What It Means | Our Mitigation |
|---|-------------|--------------|----------------|
| 1 | Buyer-User Split | Designed for PO signer, not 8-hour user | Floor mode. Dana is the design target, not Rachel. |
| 2 | Alarm Fatigue Theater | Alerts that satisfy audits but drive no action | Synara confidence decay. Only surface high-confidence suggestions. |
| 3 | Compliance-As-Retention | Audit trail dependencies as switching cost | Full export from day one (HC-5). |
| 4 | Shadow System Inevitability | Official system is compliance layer, not tool | MR-1. If people build workarounds, we failed. |
| 5 | Validation Tax | Vendor updates trigger re-validation | HC-8. Validation docs per release. Version stability. |
| 6 | Seat-Count Mismatch | Named-user pricing for broad/shallow usage | Floor users don't need seats. Site license model. |
| 7 | Data Roach Motel | Easy in, structurally difficult out | HC-5, HC-10. |
| 8 | Monolith Misfit | One surface for four different jobs | Rack concept. Marketplace add-ins. Floor mode vs office mode. |
| 9 | Power-User Ceiling | No scripting, API, batch | API non-negotiable (James). DSL. Programmatic access. |
| 10 | Invisible Cost Accounting | License is 40% of real cost | HC-7. Document TCO including validation, training, integration. |
| 11 | Renewal Trap | Auto-renewal, narrow windows, escalation | HC-9. Month-to-month. Price lock. |
| 12 | Office-Floor Gap | Designed for desktop, deployed with gloved hands | Floor mode (REQ-SC-12). 90-second rule (HC-1). |

---

## 7. Purchase Triggers (Validated)

Convergence across all focus groups. These drive positioning, onboarding, and first-session design.

### Universal (4/4 convergence)
1. **Tribal knowledge walking out the door.** Every persona lived through it. "When someone leaves, the next person inherits everything" = universal kill shot.
2. **Audit reasoning chains.** Every persona needs auditor-ready documentation. "Why this test, what assumptions, what uncertainty — without reconstruction."
3. **Dollar quantification.** Show money, not metrics. $180K overfill, $6-8K/mo scrap, 120 hrs/quarter reconstructing traceability.

### Persona-Specific Triggers
| Persona | Trigger | Timeline |
|---------|---------|----------|
| Ray Nguyen (CNC job shop) | Price lock + data export + demo with his data | Immediate |
| Maria Gutierrez (auto QM) | Trial solves audit finding | 1-3 months |
| Rachel Torres (VP Ops) | Aerospace reference + pilot success + displaces existing cost | With reference customer |
| Marcus Wade (machinist) | Live auto-updating chart faster than paper | Immediate if CMM link works |
| Greg Linden (quality director) | Dave (58yo) drops CMM file, gets correct Cpk first try | With working demo |
| Priya Chakraborty (solo QM) | CAPA chain from complaint to training record in one view | 30-minute Saturday eval |

---

## 8. First-Session Protocol

The first session IS the product. Demo = onboarding = product.

### Five Phases
1. **Triage** (90 sec) — resource prediction from data shape. Look at DATA before asking questions.
2. **Negative-space diagnostic** — "What did the last three attempts do and why did it fail?" Novel. No software onboarding does this.
3. **First blood** — one concrete result on their material while they watch. Artifact with standalone value.
4. **Parking lot** — plain language explanation before deeper commitment.
5. **Discharge artifact** — standalone valuable document. Auditor-ready. SVEND-branded. Activate to keep.

### Three Protocol Variants
| Path | When | Sequence |
|------|------|----------|
| Reflective | Has time, complex data (Dave, Maria) | Triage → Negative-space → First blood → Parking lot → Artifact |
| Crisis | Under pressure, needs results NOW (Ray, Tameka) | Triage → First blood → Parking lot → Negative-space → Artifact |
| Returning | Has PROVA history, opted in | Skip what system already knows. "Last three times it was setup drift after tool change. Check that first?" |

### Core Hypothesis
"Quality professionals will complete a guided first session using their own data and produce an artifact valuable enough to pay for continued access." — Diana (VOC gate)

---

## 9. Positioning Rules

Validated through 3 rounds with 12+ personas. These are hard rules, not suggestions.

| Rule | Source |
|------|--------|
| NEVER say "AI" externally | 8/8 personas across two focus groups stripped the word |
| NEVER say "operational excellence" | 4/4 identity focus group rejected it |
| NEVER say "analytical provenance" externally | Tameka: "say that on your sales page and I'll never click past the landing" |
| Lead with: audit defense + reasoning trails + tribal knowledge retention | 4/4 convergence |
| Price anchor: "costs less than one Minitab seat" | Replacement framing, not new-budget-line |
| Language: "reasoning trail," "audit defense you didn't have to build," "bring your ugliest dataset" | Mined from focus groups |
| Internal concept: analytical provenance (the deliberation chain, not just conclusions) | Our differentiator, never external-facing word |

---

## 10. GTM Strategy (Validated)

From cold-start transfer (4 domains: epidemiology, theater, ecology, military) + Eric's information warfare layer.

1. **Make Patient Zero maximally infectious.** Optimize first customer for documentation density, not revenue.
2. **Saturate one cluster until R0 > 1.** 5-8 adoptions in one vertical before expanding.
3. **ILSSI = credibility transfer mechanism.** Not distribution channel. Institution vouches, not vendor.
4. **Activate 50-100 practitioners as operational nodes.** Cadre development, not community building.
5. **Position for disturbance.** Monitor for: retirement, audit finding, pricing change, new leadership. Predictable incumbent failures.
6. **Information warfare.** "Can Minitab explain why it chose that test?" The question the incumbent's own users ask that the incumbent's tool can't answer. Doctrinal rigidity prevents response. Outcome predetermined before competitive engagement.

---

## 11. Existing Infrastructure to Reconcile

### Active and Aligned
| Component | Location | Status | Alignment |
|-----------|----------|--------|-----------|
| Synara infra | `~/kjerne/syn/` | Active | Redesign as governance engine |
| PCL design | Memory only | Confirmed, not built | Aligned — build as designed |
| Contract architecture | `INTEGRATION_SPEC.md` | Active | Aligned — pull-only, visible contracts |
| Composable QMS | `~/kjerne/qms/` | Phase 1+2 done | QMS templates may become workflow templates |
| Analysis Workbench | `~/kjerne/analysis/` | Active | Becomes primary Source |
| Source/Switch/Sink map | Memory | Active | Aligned |
| SPC plugin | `~/agent_dev/` | Active | Proof of plugin architecture |

### Replaced
| Component | Location | Status | Decision |
|-----------|----------|--------|----------|
| PROVA backend | `~/kjerne/prova/` | **Replaced.** | Workflow retention + Synara governance + Claude facilitation replace PROVA. Claude absorbs the facilitator role PROVA was designed to fill. Backend retired. |
| PROVA forgesia | Package | v0.1.0 | Causal graph + belief propagation available if Synara governance needs it internally. Not user-facing. |

### Contradictions to Resolve in Design Phase

1. **One workflow engine.** The unified workflow engine (WorkflowDef/Instance/Step) is canonical. QMS phase gates, approval gates, CFR Part 11 signatures — all extensible via marketplace add-ins on the one engine. The existing QMS WorkflowTemplate/Phase/Transition is absorbed or retired. **Decision: resolved 2026-05-04.**

2. **QMS tools become tiered.** A3, Ishikawa, FMEA, RCA exist as both standalone apps and QMS ToolTemplates. **Resolution (2026-05-04):** Lighter versions ship with the product (Tier 1, free). Full-featured versions are cheap/free marketplace add-ins (Tier 2). Heavy Claude-facilitated versions (guided FMEA, AI-assisted RCA) are Tier 3 (subscription + token). Standalone apps retire as marketplace plugins mature.

3. **PROVA replaced.** PROVA was designed as Claude's invisible memory (Switch: consumes + sends). Resolution (2026-05-04): workflow retention captures what was done, Synara governance tracks what worked, Claude facilitates using both. PROVA backend retired. No separate knowledge graph needed — the workflow history + PCL state + governance confidence IS the institutional knowledge.

---

## 12. Pricing Architecture (Tool Tiers)

Three tiers aligned to meta-rule MR-5: "Rules are free, judgment is Claude."

### Tier 1: Ships With It (Free)

Lighter versions of core quality tools. Works without Claude, without subscription. The trust builder.

- Basic SPC (X-bar/R, I-MR, basic capability)
- **Gage R&R** (inseparable from capability — 3/4 personas volunteered it was missing unprompted)
- Pareto
- Basic Ishikawa (template-driven)
- Simple data import + column routing
- Workflow retention (local heuristics)
- Export (HC-5)
- Audit trail (all tiers)

**Purpose:** Ray's "show me in 60 days" answered. Coexistence with Minitab (HC-12). The tool works on its own. No AI dependency. No subscription gate on core functionality.

### Tier 2: Marketplace Add-ins (Cheap/Free)

Extends capability. Community can contribute. Minimal or no Claude involvement.

- Advanced control charts (CUSUM, EWMA, short-run)
- FMEA (full)
- DOE templates
- Advanced capability (non-normal, Bayesian)
- Industry-specific workflow templates
- VSM, Hoshin (advanced)
- Forge packages (synthetic data, simulation)

**Purpose:** Depth via rack concept. User goes as deep as they want. Marketplace items are workflows generalized from real usage (REQ-WF-4).

### Tier 3: Claude Facilitation (Subscription + Token)

The judgment layer. AI facilitation costs money because API calls cost money. Visible, honest.

- Negative-space diagnostic
- Heuristic search and workflow proposals
- UI modification (toasts, suggestions, enrichment)
- "Knowledge recovered" moment (returning path protocol)
- Dollar quantification assistance
- Report narrative drafting
- Gemba walk onboarding (TRIZ P13)
- Pre-staged intelligence (background analysis)

**Pricing (validated 2026-05-04):**

| Tier | Price | Buyer | Approval |
|------|-------|-------|----------|
| Small | ~$200/mo | Job shop owner, solo QM | Self-approve |
| Mid | ~$400/mo | Quality manager, CI director | QM budget |
| Enterprise | ~$1,500/mo | Site license, VP-approved | Operations budget |

**Flat site license. NEVER per-user.** Anti-pattern #6 (Seat-Count Mismatch) — floor users don't need seats. Per-user pricing kills adoption in plants where 20 people touch the system for 5 minutes each.

**Support tiers by response time and human expertise, NOT by features.** The premium isn't more tools — it's a named contact who knows your regulatory framework.

**Anti-pattern killed:** #10 (Invisible Cost Accounting). The cost structure is transparent. "Rules are free, judgment is Claude" is the literal pricing architecture.

### Build vs. Repackage Assessment

**Critical realization (2026-05-04):** Existing SVEND already has 200+ analyses, SPC, DOE, capability, FMEA, RCA, A3, VSM, Hoshin. That inventory exceeds Tier 2. The tools are built. The work is repackaging into the new container, not rebuilding.

| What | Status | 30-Day Scope |
|------|--------|-------------|
| Analysis tools (200+) | Built | Repackage as marketplace items |
| SPC engine | Built | Lighter version = Tier 1, full = Tier 2 |
| DOE / experimenter | Built | Tier 2 marketplace add-in |
| FMEA, RCA, A3, Ishikawa | Built | Light = Tier 1, full = Tier 2, Claude-guided = Tier 3 |
| VSM, Hoshin | Built | Tier 2 marketplace add-ins |
| Forge packages (15) | Built | Tier 2 marketplace add-ins |
| **Workflow engine** | **Not built** | **Build from scratch** |
| **Synara governance redesign** | **Partially built** | **Redesign existing syn/ as governance engine** |
| **PCL** | **Deployed (hybrid update needed)** | **Diff/merge — add missing fields to existing schema** |
| **Claude facilitation layer** | **Not built** | **Build plugin architecture** |

**What needs building:** Container (workflow engine + instances + logs), Synara governance redesign, PCL models, Claude plugin architecture. Four components. Everything else is repackaging existing tools into the new tiered structure.

### Marketplace Phasing Strategy

Add-ins are phased releases, not a big bang. Each is a release event that re-engages the practitioner network.

1. **Each add-in targets a specific persona's ceiling.** When Tier 1 users hit a limit, the marketplace add-in that solves it drops. "Short-run SPC just dropped" hits Ray when he's ready.
2. **Usage data writes the backlog.** Workflow retention data shows what users reach for and hit a ceiling on. Build what's pulled, not what's pushed.
3. **Community contribution path.** Practitioner cadre (50-100 from cold-start strategy) generalize their workflows into marketplace items. Users become contributors. Contributors become advocates.
4. **Anticipation as strategy.** Phased releases buoy interest over months. The marketplace grows visibly. Each release is proof of life and momentum. Beats one launch that gets one shot.

---

## 13. Unmet Needs Discovered (Not in Original Scope)

These surfaced during VOC but are not in the four-pillar architecture. Track for future.

| Need | Source | Priority |
|------|--------|----------|
| Short-run SPC (DNOM, Q-charts for 25-piece lots) | Ray Nguyen | High — concrete underserved niche |
| CA management from unstructured sources (email, notebooks) | Tameka Jackson | Medium — document assembly |
| NCR voice dictation | Marcus Wade | Medium — "accountability without paperwork" |
| CMM auto-capture integration | Marcus Wade, Greg | High — "if the CMM link is flaky, nothing else matters" |
| Offline/graceful degradation | Dana, integration constraints | Deferred — trigger: first shop floor with unreliable connectivity |
| 21 CFR Part 11 validation package | Marcus Chen, Dave Kowalski, James Okafor | Deferred — trigger: first regulated prospect |
| Auditor-as-user (auditor gets own session to interrogate analyses) | ForgeClaude architecture morph | Future — "why would I use anything else" |
| **Gage R&R** (must be Tier 1, not Tier 2) | Dave + 2 others unprompted | **High — market says inseparable from capability** |
| FAI (First Article Inspection) workflow | Focus group validation | High — natural workflow template |
| Supplier nonconformance log | Focus group validation | Medium — QMS workflow |
| Auditor-ready report generator | Focus group validation | High — discharge artifact |

---

## 14. Language Reference

Words the market uses. Use these in UI, docs, and positioning.

| Their Word | Meaning | Use In |
|------------|---------|--------|
| Reasoning chain | Why this test, what was considered | Audit features |
| Reasoning trail | Same as above, more natural | Marketing |
| Tribal knowledge | What leaves when people leave | Positioning |
| Ugliest dataset | The real proof — not clean demo data | Sales, onboarding |
| Auditor-ready | Output standard | Feature descriptions |
| Email hell | Where CAs actually live | Problem description |
| Pencil speed | Real competitor is paper, not software | Floor design target |
| Data hostage | SaaS exit fear | Trust messaging |
| Hype fatigue | Market's emotional state toward AI claims | Positioning restraint |
| Shadow IT | What tools become without validation | Risk messaging |
| Do the work once, both needs met | Production task + audit artifact | Value prop |
| Costs less than one Minitab seat | Price anchor | Pricing page |
| **App store for quality tools** | How the market categorizes the container + marketplace | **External positioning** (Maria, Tameka — unprompted) |
| Keeps a paper trail for auditors | How AI framing lands positively (3/4 lean in) | AI positioning — USE THIS |
| AI learns your workflows | How AI framing triggers anxiety (3/4 nervous) | AI positioning — AVOID THIS |

---

## 15. Checklist: What Design Must Address

### Resolved in Design Spec (2026-05-04)

- [x] Workflow data model — WorkflowDef/Instance/Step/Heuristic. One engine. Marketplace extends capabilities.
- [x] Synara governance redesign — GovernanceRule/Decision/Outcome/Contract in new `governance/` app.
- [x] PCL Django models — Deployed. Hybrid update: add classification fields to existing schema.
- [x] Claude plugin architecture — svend@eric-tools. 6 skills, 3 hooks. Dog-foods facilitation layer.
- [x] Screen layout — Single-pane validated (4/4 focus group). Home base, pin panel, state preservation.
- [x] Template system — 7 pre-built system templates (Cpk-to-PPAP, incoming inspection, etc.)
- [x] Export architecture — JSON + CSV from day one, any tier (HC-5, HC-10).
- [x] Audit trail architecture — WorkflowStep = provenance trail. SynaraImmutableLog for datapoints (HC-15, HC-16, HC-18).
- [x] One workflow engine — QMS gates/approvals/CFR Part 11 are marketplace extensions, not a second engine.
- [x] PROVA — Replaced by workflow retention + governance heuristics. Claude absorbs PROVA's facilitator role.
- [x] Dollar quantification — Existing GAAP-compliant savings mechanisms (deployed, 89 plants) + PCL measures.

### Sequenced — Designed During Build

- [ ] Heuristic trigger schema — Synara governs this. Schema designed when governance engine is built.
- [ ] DSL design — Describes the system. Designed after system exists.
- [ ] Three-tier disclosure implementation — UI implementation, data models support it.
- [ ] Floor mode vs office mode switching — UI implementation.
- [ ] Stale data indicators (REQ-SC-4) — UI implementation, PCL staleness check exists.
- [ ] NCR feedback loop (HC-24) — Workflow step status + contract return path.
- [ ] API design for power users — Endpoints defined in design spec. DSL wraps them.
- [ ] Floor mode authentication (HC-23) — Infrastructure, not container build.

### Named But Deferred — Post-11-Day Build

- [ ] Instance naming decision — see architecture first, name before UI work.
- [ ] Marketplace/add-in architecture — rack concept defined, implementation after container stable.
- [ ] First-session protocol — product experience layer, not infrastructure.
- [ ] QMS tool migration path — standalone apps → marketplace plugins, after marketplace exists.
- [ ] Source provenance through external systems (David Kwon) — contract architecture supports it.
- [ ] Node contract versioning with deprecation policy (David Kwon) — Contract model has `supersedes` FK.

### Resolved — Separation of Concerns (2026-05-04, morph + conference)

Architecture: **Option A — "The Trust Architecture."** S1/S2 conference unanimous.

**System responsibilities:**
- **PCL** = single system of record (what we know). Stores process characteristics, Bayesian confidence, provenance. Also hosts a rules layer for relational knowledge (multi-factor interactions that survive context windows). Rules layer is part heuristic (where to go, what to fetch, temporal context), part governance (interaction rules), part knowledge compression.
- **Synara** = governance only (what's allowed). Contracts, audit trail, reversibility gate, approvals. Does NOT store knowledge. Does NOT accumulate rules.
- **Claude** = facilitator. Assembles own context from PCL directly (no intermediary). Recommendations ungoverned. Actions governed via Synara write contracts.
- **Workflows** = deterministic programs (hard layer, carries the product). User authors explicitly: tools, data bindings, triggers (schedule/PCL update/threshold), conditionals (staleness check: "only if updated since last run"), governance (Synara approval gates). Named, shareable, schedulable. Session save strips chat/exploration, keeps configs + bindings + layout, leaves input ports open. No drag-and-drop designer.
- **Sessions** = workspaces (the atomic unit). One session = one task. Tool + Data Bindings (PCL) + Layout + Context = Workspace. User-defined open/close boundary. Every session logged immutably by Synara. Ships in 11-day build, not deferred.
- **Heuristics** = soft convenience layer (10% of value, not load-bearing). Pattern matching on session logs for suggestions only. Bayesian reinforcement from outcomes. If the heuristic engine broke, every explicit workflow still runs.

**Access pattern:** Reads are free (direct to PCL). Writes pass through Synara contracts, gated by reversibility (binary: reversible or irreversible, no gray area).

**Audit trail:** `context_cited` on every write from Day 1 — snapshot of PCL entries Claude referenced. Uses existing SynaraImmutableLog with UUID hash chains. Not a new feature — already enforced by infrastructure.

**Learning loop:** Outcomes feed back into PCL via Bayesian update. System learns without a separate learning engine.

**Governance:** Outcomes-only. Don't govern Claude's thinking, govern the acting. Practitioner knowledge captured at point of use — expert judgment (e.g., QE rejects a tool on methodological grounds) enters PCL heuristic layer and propagates to other sites. No approval meetings. Knowledge built in the trenches. Enterprise governance toggle (governed recommendations) deferred until regulated customer demands it.

**Navigation (wargame validated):** Workspace-first, not tool-first. Retained users live in favorites + search. Sidebar is onboarding infrastructure, not daily navigation. Three persona modes: operators get pinned destinations (zero navigation), engineers get personal workbench (favorites + data bus), new users get guided paths (role-based onboarding + contextual suggestions). Two-mode UI: power-user desktop + operator kiosk.

---

*This document is the output of 3P VOC. Next step: system design spec mapping each requirement to concrete architecture.*

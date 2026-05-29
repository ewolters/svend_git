# Conference: SVEND Integration Design Constraints

**Date:** 2026-05-03
**Technique:** S1/S2 Dialectical Debate
**Input:** Two focus groups (8 personas), SVEND identity/facilitator model, Source/Switch/Sink architecture

---

## S1 (Innovator) Position

### Framing

The question is not "how should SVEND's apps talk to each other." The question is: **will SVEND be the real system, or will it become one more layer in someone's shadow system?**

Every persona across both focus groups — all eight — maintains a shadow system. The conventional approach to integration design (clean interfaces, versioned APIs, respect boundaries) is necessary but nowhere near sufficient. If SVEND follows those rules and nothing else, it becomes MasterControl — architecturally sound, universally resented, survived only because exit costs exceed tolerance.

The 12 named anti-patterns are not bugs in bad products. They are **emergent properties of conventional integration design**. The constraints must make those anti-patterns structurally impossible, not merely discouraged.

### Meta-Rule

**The user's working system and the auditable system must be the same system.** If those diverge, SVEND has failed regardless of how clean the API is.

### 14 Constraints in Four Categories

#### Flow Architecture

**FLOW-1: Pull-Only Integration.** Switches and sinks pull from upstream sources. Sources never push. Every "close the loop" integration failure Greg described was a push model — mandatory fields nobody fills, automatic escalations nobody reads. Pull means the consumer decides when and what to bring in.

**FLOW-2: Event Traceability, Not Feature Integration.** Integration means: when X happens in one context, Y can reference it from another context with the full chain intact. It does NOT mean: when X happens, Y automatically fires. Events stored as immutable facts with typed references; any downstream tool can pull them. Triggering is always human or Claude-facilitated.

**FLOW-3: Contracts Are Visible and User-Controlled.** Every pull connection visible via a config surface (modal, panel, or equivalent). The user sees what is being pulled, from where, and can sever any connection. No hidden wiring. No implicit dependencies.

**FLOW-4: Every Event Has a Return Path.** When a source emits an event that a downstream consumer acts on, the resolution must be visible to the originator. The return path is passive — status indicator on prior observations, seen when the operator next interacts. No notifications, no emails, no popups. Present, not pushed.

#### Friction Budget

**FRIC-1: 90-Second Floor Rule.** Any interaction performed by a shop floor operator must be completable in 90 seconds or less, including login, navigation, data entry, and confirmation. Hard veto criterion. If exceeded, the workflow ships broken.

**FRIC-2: Zero Mandatory Fields Beyond the Fact.** When recording an observation, the only mandatory data is the fact itself. Part number, timestamp, operator — auto-populated or inferred from context. Everything else (root cause codes, categories, severity ratings, disposition) is optional at point of capture and enriched later.

**FRIC-3: Claude Absorbs Enrichment Friction.** The friction that mandatory fields were designed to capture (categorization, severity, routing) is redirected to Claude. Claude reads the raw observation and enriches it: suggests disposition, links to prior similar events, estimates severity from PCL context. The operator observes and records. Claude structures, classifies, and routes.

#### Trust Architecture

**TRUST-1: Full Export at Any Time, Structured and Machine-Readable.** Every piece of data exportable in CSV/JSON at any time, without contacting support, without a contractor, without a special license tier. Export button exists from day one on the free tier.

**TRUST-2: API Versioning with 12-Month Deprecation.** Any published API endpoint carries a 12-month deprecation guarantee. No breaking changes on minor releases. Versioned endpoints.

**TRUST-3: Price Transparency and No Feature-Tier Hiding.** Every capability documented with its tier requirement on a public page. No "request a demo" gates on pricing.

**TRUST-4: Validation Burden Stays with SVEND.** When SVEND ships an update, the customer's validation status is not invalidated. For regulated customers, SVEND provides IQ/OQ/PQ documentation per release. Generated from the test suite and change management system (CHG-001).

#### Modularity Contract

**MOD-1: Progressive Disclosure — Enter Through One Problem.** System usable with exactly one capability activated. Capabilities discovered through use, not feature tours. Claude may suggest adjacent tools when work demands it but never forces navigation to an unused capability.

**MOD-2: Coexistence, Not Replacement.** SVEND must run alongside existing tools indefinitely. Accept data from Minitab, produce data Minitab can read, never require the user to stop using Minitab.

### Risks Acknowledged

- **FRIC-2 garbage data:** Mitigated by Claude enrichment (FRIC-3) running asynchronously. Raw data with AI enrichment beats mandatory fields filled with "N/A."
- **FLOW-4 alarm fatigue:** Mitigated by passive return path (present, not pushed).
- **TRUST-4 is expensive:** Mitigated by generating from test suite and CHG-001. Competitive moat, not just a cost.
- **MOD-1 hides value:** Mitigated by Claude as discovery mechanism — context-driven suggestions, not feature tours.
- **FRIC-1 constrains complexity:** 90-second rule applies to operator interaction; system processing time is unlimited.

### What Gets Left on the Table If Conservative Wins

- The shadow system kill (without FRIC-1/FRIC-2, SVEND becomes a compliance layer)
- The facilitator advantage (without FRIC-3, Claude is just a chatbot on top of tools — every QMS vendor will bolt one on in 18 months)
- The trust moat (incumbents cannot adopt TRUST-1 through TRUST-4 without destroying their retention business model)
- The validation disruption (first vendor to solve TRUST-4 owns the regulated manufacturing segment)

### Steel-Manned Counter

"These constraints optimize for a user who does not yet exist. SVEND has one paying user. Greg wants CMM-to-PPAP, not an integration architecture. The constraints add engineering complexity that delays delivering what Greg actually asked for."

S1's response: "The constraints are design rules, not features. The CMM-to-PPAP pipeline built under these constraints looks identical to the user. It just stores events as immutable facts, auto-populates everything except the measurement, and exports to CSV. The incremental cost per feature is small. The cost of retrofitting is enormous — ask anyone who has tried to remove mandatory fields from a deployed system."

---

## S2 (Conservative) Position

### Framing

Same core question: **will SVEND become the real system or the compliance layer?** The answer depends on whether anyone uses it. The constraints exist to ensure usage.

The secondary stake: Greg's SAP pattern-match. Our exact ICP hears a comprehensive platform pitch and his brain fires "SAP" and he disengages permanently. The constraints must prevent this at the architecture level.

### Governing Principle

**The simplest integration that eliminates re-keying between two events.** The user's word for what they want is "pull the full chain" (Priya) and "one click instead of four hours" (David).

### 10 Rules, Each Traced to a Named Persona

**Rule 1: Single-Problem Entry (the "SAP Firewall").** No deployment path may require configuring more than one app. Every app functions with zero contracts configured. Contracts are additive enrichment, never prerequisites.

**Rule 2: Pull-Only, User-Visible Contracts.** Sources never push. Every active contract visible in config panel user controls. Contract activation requires explicit user action. Active contracts displayed on every relevant screen. User must answer "what is feeding this view?" in under 5 seconds.

**Rule 3: Sub-90-Second Task Completion for Primary Actions.** Time every primary action path. If it exceeds 90 seconds, it ships broken. Quality gate, not guideline. Applies to shop-floor workflows; analytical workflows have different latency budgets.

**Rule 4: Events Chain; Features Don't.** Every contract defines: (a) the triggering event, (b) the data payload that transfers, (c) the destination action. Contracts are event-to-action mappings, not data-sharing agreements.

**Rule 5: Structured Export from Day One.** Every model with user-created data exposes bulk export (JSON + CSV minimum). Export includes relationships. Tested in CI.

**Rule 6: API Versioning and Stability Guarantee.** Versioned endpoints in URL path. 12-month deprecation. Schema published and versioned. No "minor update" may alter exported data shape. Release notes state whether re-validation is required.

**Rule 7: NCR Feedback Loop.** Every signal sent into the system produces visible status update to originator. Minimum: received, in progress, resolved. Every contract crossing a role boundary includes a reverse-status channel.

**Rule 8: Graceful Degradation (the "7 PM" Rule).** Every shop-floor workflow has a defined degradation path. Queue locally, sync when available, timestamp accurately. System never loses data because network was down.

**Rule 9: Price and Cost Transparency as Architecture.** Feature gating auditable against published pricing. No feature gated to a tier unless tier and price publicly documented. Exit cost documented alongside pricing.

**Rule 10: The 80% Ignorability Test.** Unconfigured apps completely invisible. No menu items, no empty dashboards, no onboarding wizard walking through all capabilities. App visibility tied to activation state.

### Risks Acknowledged

- Over-constraining to today's personas delays tomorrow's differentiation
- Pull-only may cripple Claude's facilitation model (tension between "no background wiring" and "Claude connects the dots")
- Export-from-day-one is expensive pre-customers
- Ten rules may create bureaucratic overhead

### What Goes Wrong If Innovator's Constraints Are Too Ambitious

- **Beautiful architecture, empty system:** Constraints like "all integration must flow through the knowledge graph" create implementation dependencies that delay shipping. Greg is copying CMM data into Excel 15x/week.
- **Facilitator becomes bottleneck:** If constraints assume Claude is always present and capable, system breaks when Claude is slow, wrong, or unavailable. Every constraint must work WITHOUT Claude and be BETTER with Claude.
- **Constraints that require full deployment to validate:** Abstract constraints can't be tested on single-app deployment. If validation requires full platform running, constraints won't be validated until too late.

### Steel-Manned Counter

"SVEND's differentiation is analytical provenance — the chain of reasoning connecting a process signal to a business decision. That chain is inherently cross-app. It requires the graph, PROVA, Claude connecting information the user didn't explicitly link. Conservative pull-only single-app constraints may produce ten individually excellent apps that never produce the emergent chain."

S2's response: "Analytical provenance is the destination. These constraints are the road. You cannot skip the road because the destination is compelling."

---

## Conference Synthesis

### Agreements

S1 and S2 converge on 7 core rules with near-identical language:

| Principle | S1 | S2 | Status |
|---|---|---|---|
| Pull-only integration | FLOW-1 | Rule 2 | **Identical** |
| Visible, user-severable contracts | FLOW-3 | Rule 2 | **Identical** |
| Sub-90-second shop-floor interactions | FRIC-1 | Rule 3 | **Identical** |
| Events chain, features don't | FLOW-2 | Rule 4 | **Identical** |
| Structured export from day one | TRUST-1 | Rule 5 | **Near-identical** |
| API versioning with 12-month deprecation | TRUST-2 | Rule 6 | **Identical** |
| NCR feedback loop / return path | FLOW-4 | Rule 7 | **Identical** |
| Price transparency | TRUST-3 | Rule 9 | **Identical** |
| Progressive disclosure | MOD-1 | Rule 10 | **Near-identical** |

Both agree on the meta-principle: the working system and the auditable system must be the same system. Both agree shadow-systems are the primary failure mode. Both agree constraints are cheap now and expensive later.

**This is not a debate about whether these constraints should exist. It is a debate about scope, enforcement mechanism, and what gets left out.**

### Disagreements

**A. Claude's Role in Friction Absorption**
- S1 (FRIC-3): Claude absorbs enrichment friction — categorization, severity, routing. Named constraint shaping every data model.
- S2: No equivalent rule. Claude's role should emerge from usage, not be mandated.

**B. Offline / Graceful Degradation**
- S2 (Rule 8): Offline behavior defined for shop-floor workflows. Queue locally, sync later. Hard constraint.
- S1: Not addressed.

**C. Validation Burden (IQ/OQ/PQ)**
- S1 (TRUST-4): SVEND provides IQ/OQ/PQ documentation per release.
- S2: Not addressed as constraint.

**D. Coexistence with Incumbents**
- S1 (MOD-2): Explicit coexistence rule — run alongside Minitab, import/export with incumbents.
- S2: Covered implicitly by structured export; not elevated to named constraint.

**E. Minimum Required Fields**
- S1 (FRIC-2): Zero mandatory fields beyond the fact. Everything enriched later by Claude.
- S2: No equivalent. 90-second rule implicitly constrains form complexity.

**F. Derivation Philosophy**
- S1: 14 rules from architectural reasoning. Some anticipate personas that don't exist yet.
- S2: 10 rules, each traced to a named persona. Excludes anything without persona evidence.

### Crux of Each Disagreement

**A. Claude enrichment:**
If Claude enrichment works reliably and users trust it → S1 wins (differentiated capture experience). If Claude produces errors requiring review → S2 wins (two-pass workflow slower than three fields).

**B. Offline:**
If early deployments have reliable connectivity → S1's omission costs nothing. If shop floors have spotty WiFi → S2's rule prevents trust-destroying failures.

**C. Validation:**
If next 3-5 customers include FDA/ISO-regulated manufacturers → S1 wins. If next 3-5 are job shops caring about speed and price → S2 wins by omission.

**D. Coexistence:**
If prospects consistently ask "can I still use Minitab?" → S1's explicit rule matters. If they ask "what does it do?" → coexistence framing is premature.

**E. Fields:**
If "zero mandatory fields beyond the fact" can be operationalized without ambiguous records → S1 wins. If records captured with only the observation frequently require follow-up → S2's implicit approach is more practical.

**F. Philosophy:**
Value judgment, not empirical claim. Fear of under-specification vs. fear of over-specification.

### Open Questions for the Arbiter

1. **What is Greg's deployment environment?** Reliable WiFi → Rule 8 can wait. Spotty → it's urgent. Knowable now.

2. **How reliable is Claude enrichment today?** Run 50 sample observations through current pipeline. Measure accuracy of auto-categorization vs. time to fill fields manually. Resolves Disagreement A empirically.

3. **Are the next 3 prospects in regulated industries?** If yes, TRUST-4 earns its cost. If no, it's deferred.

4. **Does "zero mandatory fields" survive contact with CMM-to-PPAP?** A CMM reading without a part number is useless. Does FRIC-2 mean "zero beyond the measurement and its identifier" or literally "one field"? Needs operational definition.

5. **How many of these constraints already describe how the codebase works?** If pull-only, event-based, and structured-export are already implemented, adopting them is free documentation. If they require refactoring, cost calculation changes.

6. **What is your tolerance for carrying unused rules?** S1's 14 include at least 2 (TRUST-4, MOD-2) serving no current user. Cheap if they're review-time checks. Expensive if they require infrastructure.

### Risk of Each Path

**Following S1 (14 constraints, architecturally derived):**
- Goes right: Every feature is integration-ready, export-ready, trust-ready from birth. When regulated customer arrives, validation docs exist. Claude-as-enrichment is genuinely differentiated.
- Goes wrong: FRIC-3 is highest-risk bet — if unreliable, every capture has a hidden review step. TRUST-4 consumes time for a segment that may not materialize. 14 constraints create compliance surface for dev process itself. Specific failure: spend two weeks on validation infrastructure while Greg waits for his PPAP export.

**Following S2 (10 constraints, persona-derived):**
- Goes right: No engineering on speculative requirements. 10-rule set fits in working memory. First five customers get what they asked for, fast.
- Goes wrong: No FRIC-2/FRIC-3 means capture forms designed ad hoc — some minimal, some over-fielded, inconsistency accumulates. No MOD-2 means import/export implemented per-request, not by principle. No TRUST-4 means first regulated customer triggers fire drill. Most critically: ten excellent disconnected apps that never produce the analytical provenance chain. Without explicit cross-app intelligence accumulation, pull-only architecture produces isolation by default. Specific failure: customer six is a med device manufacturer, and retrofitting traceability + validation across ten independent apps costs more than building it in.

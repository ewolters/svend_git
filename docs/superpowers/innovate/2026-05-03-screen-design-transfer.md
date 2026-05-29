# Cross-Domain Transfer: Instrument-Advisor Screen Design

**Date:** 2026-05-03
**Technique:** Analogical Transfer
**Domains:** Trading Floor, ICU/Clinical Informatics, Military C2, Restaurant Kitchen

## Problem

How do you design a screen where a primary instrument/tool interface and an expert advisor coexist, forming a do-think-act loop without the transition between modes becoming friction? Users range from floor operators (90 seconds, gloved hands) to power users (deep analysis, programmatic access). The tool is primary, the AI is available on demand.

---

## Domain Solutions

### Trading Floor — "The Dealer Turret Problem"

**Reframing:** Execution surface (blotter) + research analyst, colocated at the same desk.

**Solution:** Three-layer architecture:
1. **Blotter** (100% of screen) — execution surface, never displaced
2. **Command Strip** (persistent 32px bar) — inherits instrument context, two-line response overlay, auto-fades 8 seconds
3. **Analyst Panel** (docked sidebar, 30% width, pushes blotter left) — full conversation, tethered to blotter selection, every recommendation = pre-populated order ticket
4. **Squawk Ticker** (ambient strip) — rate-limited unsolicited alerts (max 3/5min), clickable to open analyst panel pre-loaded

Round-trip: Squawk → click → analyst pre-loaded → recommendation as ticket → one click execute → auto-minimize. ~15 seconds.

**Structural principle:** "Collocated Contextual Escalation with Pre-Wired Action Completion." (1) Execution surface sovereign, (2) context flows downward automatically, (3) disclosure is gravity-based / spring-loaded — takes force to hold open, naturally collapses, (4) every insight terminates in a pre-wired action — advisory output type IS execution input type.

**Key line:** "They are not two applications stitched together. They are two views of a single state machine."

### ICU / Clinical Informatics — "Bedside Monitor + Consulting Physician"

**Reframing:** Patient monitor + clinical decision support, nurse never leaves bedside.

**Solution:** Primary waveforms (75-80%) + right-side Advisor Rail:
- Collapsed: thin status strip with passive clinical alerts (color flags)
- Three tiers keyed to URGENCY, not skill:
  - **Passive/Ambient** (zero cost): peripheral signals — color changes, indicators
  - **Contextual/Anchored** (single touch): tap any vital tile → pre-loaded brief, auto-dismisses 15 seconds
  - **Deliberative/Expansive** (intentional): full consultation, 40-60% width, multi-turn

Context via HL7 Infobutton: touching a data element IS the query. SBAR auto-composed from state. Suggestions = actionable checkboxes pre-populating order forms (staging, not execution).

**Structural principle:** (1) Spatial hierarchy with guaranteed primary persistence, (2) context tunneling — working in the instrument IS query formation, (3) three-tier disclosure keyed to urgency not skill, (4) suggestion staging not direct execution, (5) loop closes through instrument not advisor.

**Key line:** "The user's act of working in the primary instrument IS the query formation."

### Military C2 — "Commander's Workstation"

**Reframing:** Common Operating Picture (COP) + embedded S2/S3 staff officer.

**Solution:** Split-panel TOC (75-80% COP + 20-25% Staff Channel):
- Collapsed: narrow strip with BLUF assessment
- Three engagement tiers matching battle rhythm:
  - **Passive COP Annotation** (zero-click): staff posts anomaly indicators directly onto COP as native overlays
  - **Directed Query** (90-second ceiling): select COP element → pre-staged intelligence product
  - **Extended Session** (no limit): 40-60% split, COP stays live

Staff outputs = executable action payloads rendered as proposed COP overlays. Context shared bidirectionally. Pre-staged intelligence: advisor already prepared the brief because monitoring same feeds.

**Structural principle:** (1) Transition cost is a saccade not a navigation, (2) three-tier matching operator tempo, (3) advisory output in instrument-native format, (4) context shared not transferred, (5) ambient presence — collapsed state shows evidence of activity, building trust.

**Key line:** "An advisor that speaks the instrument's language and lives in the instrument's space is not a mode — it is a capability of the instrument itself."

### Restaurant Kitchen — "The Pass"

**Reframing:** Cook's station + expediter at the pass. Cook never stops cooking.

**Solution:** Station (cook's sacred workspace) + Pass (boundary where expo lives):
- Three communication tiers:
  - **Ticket Rail** (ambient, zero interruption): expo annotates at cook's eye line. Cook glances, never stops.
  - **Call and Response** (3 seconds): compressed protocol. "Fire table 9!" / "Heard!" Hands stay on station.
  - **Walk-Up** (30 seconds, cook initiates): cook steps to pass with a real question.

Key insight: expo reads the plates (output), not the cook's mind. The plate IS the explanation. Expo reasons from work product, not narration.

**Structural principle:** (1) Shared surface, asymmetric access, (2) tiered communication with escalating cost — 90% at cheapest tier, (3) advisor reads output not intent, (4) edge-mounted not center-mounted — visible when you look up, invisible when heads-down, (5) station works at full capability with advisor absent.

**Key line:** "The line runs. The expo reads the line. The expo speaks when the line needs it. The line never stops for the expo."

---

## Synthesis

### Convergent Principles (4/4 or 3/4 independent agreement)

**Execution surface sovereignty.** All four: blotter never displaced, waveforms always live, COP sovereign, station sacred. The doing-surface is never obscured, only compressed. Least surprising but most strongly validated.

**Three-tier progressive disclosure.** All four arrived at exactly three tiers: ambient/zero-cost (~0s), directed/single-action (~15s), deliberative/expansive (~60s+). Tiers map to available attention budget, NOT user skill. The same user moves between tiers in a single session.

**Advisory output in instrument-native format.** Trading (pre-populated ticket), clinical (order entry checkbox), military (COP overlay), kitchen (verbal call using station vocabulary). The advisor's output IS the instrument's input. Not prose in a sidebar — an actionable object on the primary surface.

**Shared state, not transferred context.** Trading ("two views of a single state machine"), military ("context shared not transferred"), kitchen ("expo reads the plates"). Advisor and instrument share one source of truth. The advisor never needs to be told what the user is looking at.

### Novel Principles (uncommon in software UI for manufacturing)

**Gravity-based disclosure (trading).** Advisory panels are spring-loaded — require force to hold open, naturally collapse. Time-in-advisory decays. This is not binary open/closed but a physical interaction model. Implies: auto-dismiss, progressive resistance, intentionality scales with dwell time.

**Context tunneling / working IS querying (clinical).** The act of using the instrument generates the advisor query implicitly. User never switches to "asking mode." Touching a data element IS the query. Eliminates the single largest friction: formulating a question.

**Advisor reads output, not intent (kitchen).** The AI reasons from the work product visible on screen — the control chart, the data, the measurement — not from user narration. Combined with context tunneling: query formation eliminated for the common case.

**Pre-staged intelligence (military).** Advisor computes speculatively in background, watching same data stream. User query = cache retrieval, not computation trigger. Interaction feels instant because it IS instant — analysis was done in advance.

**Compressed protocol vocabulary (kitchen).** 90% of interactions use standardized micro-phrases, not natural language. Chat is the rare, expensive tier. Challenges assumption that AI advisor's primary interface should be a chat box.

**Ambient evidence of activity (military).** Collapsed advisor shows signs of life — annotations appearing, indicators updating. Not notifications (demand attention) but ambient evidence (reward peripheral vision). Builds trust that advisor is engaged even when not consulted.

### Problem Reframings

**Kitchen reframes the advisor's input channel.** Original problem assumes user must invoke/address the advisor. Kitchen eliminates this: expo watches the plates. Translate: AI's primary input = instrument state observation, not user queries. Reframes from "how does user access advisor" to "how does advisor read the instrument."

**Military reframes temporal coupling.** Original assumes synchronous loop (user acts → consults → advisor responds → user acts). C2 breaks this: advisor asynchronously prepares intelligence in parallel. Consultation = cache hit, not computation. Reframes from "minimize round-trip latency" to "maximize pre-computation coverage."

**Clinical reframes who tiers serve.** Original frames user spectrum as population segmentation (floor ops vs power users). Clinical reframes as temporal segmentation: same user moves between urgency levels during a single session. Tiers are moments, not personas.

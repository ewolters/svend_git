# Morphological Analysis: Navigation UX Execution

**Date:** 2026-05-05
**Technique:** Morphological Analysis (Zwicky Box)
**Evaluation lens:** Japanese design aesthetics (kanso, ma, shizen, fukinsei, shibui, datsuzoku)

## Context

Project-based navigation validated. Flowchart-as-project-structure validated. Remaining problem: UX execution across one-off use, complex projects, exploration, floor operators, and power users.

## Morphological Box

| Dimension | Opt 1 | Opt 2 | Opt 3 | Opt 4 | Opt 5 |
|---|---|---|---|---|---|
| **Entry Mode** | Blank Canvas w/ Verb Prompt | Recency Rail | Mandatory Situation Declaration | Physical Locker | Broadcast Start (QR/URL) |
| **Flowchart Visibility** | Breadcrumb Only | Slide-In Map (drawer) | Peripheral Minimap (20% opacity) | Fully Hidden, Completion-Triggered | Film Strip (horizontal) |
| **Claude's Contract** | Paged-in Advisor (modal) | Ambient Listener, Silent | Interrupt Budget (2/session) | Project Architect Only (absent during execution) | Physical Clipboard (hand it context) |
| **Exploration Model** | Symptom Triage | Free Node Placement | Concept Map w/ Costs | Socratic Narrowing (3 Qs) | Borrow Template, Break It |
| **Builder/Runner** | Single Surface, Mode Toggle | Strict Role Separation (diff URLs) | Scaffolding (retracts on Run) | Runner-Native, Builder = Backdoor (long-press) | Sheet Music vs Performance |
| **Persistence** | Explicit Save Points (Close Loop) | Continuous Append Log | Session as Shift (logbook) | Volatility by Default | Genealogy (version tree) |

**Total design space:** 15,625 combinations

## Random Combinations Evaluated

### Combination 1
Blank Canvas / Breadcrumb Only / Interrupt Budget (2) / Free Node Placement / Strict Role Separation / Continuous Append Log

- Feasible but schismatic — role separation fractures the mental model
- Free Node Placement rewards patience; continuous log accumulates silently
- Interrupt Budget introduces scarcity of attention (Claude speaks rarely, so when it does, it lands)
- **Missing:** floor operator. This is a thoughtful analyst's setup.

### Combination 2 ★
Blank Canvas / Film Strip / Paged-in Advisor / Borrow Template Break It / Runner-Native Builder Backdoor / Explicit Save Points

- **Strong candidate.** Most ship-ready.
- Film Strip keeps linearity visible but not intrusive
- Builder exists only for those who know to look (long-press) — profound Ma
- Template-breaking mirrors actual manufacturing knowledge transfer
- Save-as-loop-closure is meaningful, not mechanical
- Shizen: "borrow and adapt" is how manufacturing actually works

### Combination 3
Blank Canvas / Breadcrumb Only / Ambient Listener Silent / Free Node Placement / Sheet Music vs Performance / Genealogy

- Most conceptually rich, least immediately implementable
- Sheet Music reframes what a flowchart IS — scored intention, not lockstep procedure
- Silent Claude + genealogy = system that accumulates interpretations over time
- Describes a mastery system: your performance history annotates the score
- Needs the Sheet Music concept rendered without explanation

### Combination 4
Blank Canvas / Film Strip / Ambient Listener Silent / Borrow Template Break It / Runner-Native Builder Backdoor / Continuous Append Log

- Combination 2's quieter sibling — same philosophy, less interaction
- No save dialogs, no AI prompts → every interruption removed
- Floor operator ideal (90 seconds, zero friction)
- But no reflection points → log becomes archaeology, not decision record
- **Dimension interaction:** Silent Claude + auto-persist = flow state by design

### Combination 5
Physical Locker / Film Strip / Interrupt Budget (2) / Symptom Triage / Strict Role Separation / Volatility by Default

- Symptom Triage + Volatility = troubleshooting mindset (most diagnoses don't need saving)
- Physical Locker (badge scan / station QR) natural for floor operators but needs UX invention
- Volatility is a philosophical stance: most work is disposable exploration
- Most aligned with *investigation* use case, least aligned with *project* use case

### Combination 6
Mandatory Situation Declaration / Peripheral Minimap / Ambient Listener Silent / Free Node Placement / Scaffolding Retracts / Explicit Save Points

- **Context membrane:** situation declared at the door + minimap whispers at edge of perception
- Scaffolding retraction = the surprise (builder dissolves, you're inside the work)
- Most coherent for lean-trained practitioners (A3 thinking: frame situation, then work)
- Free Node Placement within declared frame = freedom inside constraint

### Combination 7
Blank Canvas / Fully Hidden Map / Paged-in Advisor / Concept Map with Costs / Scaffolding Retracts / Genealogy

- Completion-triggered map revelation = strongest Datsuzoku (map appears as reward for progress)
- Each completed node draws more of the map — discovery mechanic
- Concept Map with Costs creates exploration anxiety (wrong pairing)
- Swap Costs for Socratic Narrowing → combination becomes remarkable
- Cinematic: you finish work and suddenly see where you are

### Combination 8
Mandatory Situation Declaration / Breadcrumb Only / Project Architect Only (absent) / Borrow Template Break It / Single Surface Mode Toggle / Volatility by Default

- Claude's absence during execution + Volatility = strong philosophical position
- "AI sets you up, then leaves. What you do isn't necessarily worth keeping."
- Expert users who resent AI intrusion: extremely appealing
- New users: could feel abandoned
- Declaration + Template + Volatile = state situation, get scaffold, work, discard

### Combination 9 ★
Blank Canvas / Film Strip / Interrupt Budget (2) / Borrow Template Break It / Scaffolding Retracts / Genealogy

- **Most internally coherent.** Every element has diminishing presence logic.
- Template → break it down. Scaffolding → retracts. Claude → speaks twice then quiet. Genealogy → records without demanding attention.
- Embodies a learning arc: supported → independent. Never labels itself as pedagogy.
- Film Strip + Genealogy = two-axis history (horizontal current run, branching version tree)
- Scaffolding retraction + interrupt budget = diminishing AI presence arc
- **Strongest shizen in the set.**

### Combination 10
Recency Rail / Breadcrumb Only / Paged-in Advisor / Free Node Placement / Scaffolding Retracts / Explicit Save Points

- The null hypothesis. The control group. Conventional SaaS.
- Entirely feasible, entirely defensible, entirely uninteresting by aesthetic criteria.
- No surprises. No point of view. Furniture.
- Diagnostic value: measures how much the others have to say.

---

## Top 3 Most Promising

### 1. Combination 9 — "The Diminishing Scaffold"
Blank Canvas / Film Strip / Interrupt Budget (2) / Borrow Template Break It / Scaffolding Retracts / Genealogy

**Why:** Every element has a diminishing presence logic — structure that dissolves as you gain confidence. Template gives structure to break, scaffolding gives guidance that retracts, Claude speaks twice then goes quiet, genealogy records without demanding attention. Embodies a full learning arc without being labeled as pedagogical. Strongest shizen (naturalness) in the set.

### 2. Combination 2 — "The Runner's World"
Blank Canvas / Film Strip / Paged-in Advisor / Borrow Template Break It / Runner-Native Builder Backdoor / Explicit Save Points

**Why:** Most ship-ready. Has a genuine philosophical position: running is first-class, building is hidden (long-press). Template-breaking is how manufacturing knowledge actually transfers. Save-as-loop-closure means saves are meaningful. Best Ma — the builder exists in the silence of the interface.

### 3. The Ghost Combination — "The Context Membrane"
Mandatory Situation Declaration / Peripheral Minimap (20% opacity) / Interrupt Budget (2) / Borrow Template Break It / Scaffolding Retracts / Genealogy

**Why:** Never generated by random draw but the logic of the box points toward it. Resolves *context without friction* and *diminishing AI presence with a record*. Situation Declaration routes each user appropriately. Minimap holds structure at perception's edge. Scaffolding retracts. Genealogy branches differentiate needs over time. Most complete answer to the multi-persona problem.

---

## Unexpected Dimension Interactions

### Persistence × Claude's Contract
Silent Claude + auto-persist = every metacognitive interruption removed. No save dialogs, no AI prompts. Describes flow state by design. Ideal for execution, wrong for learning.

### Exploration × Builder/Runner
"Borrow Template, Break It" has an implicit opinion about who the builder is: you're an adaptor, not a designer. Pairs powerfully with Runner-Native (execution first). Breaks under Strict Role Separation (forces adaptors into a different URL for minor mods).

### Entry Mode × Flowchart Visibility
Mandatory Situation Declaration + Peripheral Minimap creates a context membrane. Situation declared once at the door; minimap whispers structure at edge of vision. Neither repeats itself. Together = context without noise. Only Situation Declaration provides semantic content that gives the minimap meaning.

### Interrupt Budget × Runner-Native (never generated)
If Claude speaks only twice AND interface is runner-native, both interrupts occur during execution, not setup. Makes Claude's moments feel like quality checks or flags, not suggestions. Different contract entirely.

---

## The Design Space Narrowing

All three top candidates share:
- **Borrow Template, Break It** as exploration model (4/4 if you count combinations 2, 4, 8, 9)
- **Film Strip** or **Peripheral Minimap** as flowchart visibility (horizontal linearity or ambient whisper — never full presence)
- **Scaffolding Retracts** or **Runner-Native Backdoor** (both treat building as temporary/hidden)
- **Genealogy** as persistence (version trees, not save dialogs)

The design space converges toward: **inherited structure, adapted locally, where the builder dissolves after setup and history accumulates as branching versions.**

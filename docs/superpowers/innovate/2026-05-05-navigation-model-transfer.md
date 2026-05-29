# Cross-Domain Transfer: Navigation Model

**Date:** 2026-05-05
**Technique:** Analogical Transfer
**Domains:** Restaurant kitchen operations, Operating theater (surgery), Music production/DJ performance, Machine shop/tool crib

## Problem Statement

SVEND has ~15 tools (VSM, SPC, FMEA, RCA, A3, DOE, Hoshin, etc.) behind a dropdown navbar. Focus groups rejected "ask Claude to open a tool" (hard no). Wargame showed flat sidebars die at 30+ tools and users build 5-tool micro-products. Constraint: no traditional menus AND no AI-as-launcher. Question: what IS the navigation model?

---

## Domain Solutions

### Restaurant Kitchen Operations

**Reframe:** Shared prep kitchen with 15 specialized equipment pieces. Workers range from pastry cook (one tool, gloves, time pressure) to saucier (orchestrates six simultaneously).

**Solution:** Brigade mise en place with distributed station kits.

**Principles:**
1. **Ownership before execution.** Access decisions made during setup (low pressure) not service (high pressure). The moment of need is the wrong moment to navigate.
2. **Proximity earned by frequency.** Arrangement reflects motion patterns, not categorical logic.
3. **Context travels; material is staged.** Expediter calls propagate context — every station self-activates. Information moves, not workers.
4. **Shared neutral surface for integration.** Multiple workers combine outputs at a shared zone (the pass) owned by no one.

**Key line:** "Navigation disappears because placement decisions happen before the moment of need."

---

### Operating Theater (Surgery)

**Reframe:** OR performing many procedure types. Simple biopsy (one instrument, 90 seconds) to complex reconstruction (fifteen instruments, four hours).

**Solution:** Two-surface instrument architecture (Mayo stand + back table + preference cards).

**Principles:**
1. **Access surface ≠ storage surface.** Collapsing them creates a hard scale ceiling.
2. **Mayo stand = personal, pre-positioned, phase-scoped.** Small by design. Populated by anticipation, not search. Belongs to the practitioner.
3. **Back table = clustered by functional family.** Navigated by spatial reading, not labels.
4. **Preference card = expertise crystallized as pre-positioning instruction.** Converts anticipation into inventory signal. Separable from the person.
5. **Phase-scoping prevents overload.** Fifteen instruments never simultaneously on Mayo. Complexity managed by time-slicing.
6. **Context flows within the field.** Material/specimen continuity stays in working area.

**Key line:** "Separate access from storage, personalize access through pre-positioning based on anticipation, scope access to current phase of work."

---

### Music Production / DJ Performance

**Reframe:** Shared backline with 15 complex outboard units. Vocalist (one effect, 90 seconds) to session engineer (patches 15 units all day).

**Solution:** Session View + Channel Strip + Preset Recall — three layers for three roles.

**Principles:**
1. **Normalled strips** — personal configurations always hot, zero navigation. The tool is already present.
2. **Functional buses, not individual units** — group tools by what they DO TOGETHER. Route to a function ("I need glue"), not to a specific unit.
3. **Scene recall for composition** — professionals switch between pre-built states. Navigation is front-loaded into design time.

**Key line:** "The right number of access models equals the number of distinct roles, not the number of tools." The vocalist wants a RESULT. The engineer wants a STATE. Only the middle player browses, and by FUNCTION, not unit.

---

### Machine Shop / Tool Crib

**Reframe:** Tool crib with 15 complex tools. Setup machinist (one gauge, coolant on gloves) to journeyman (three setups daily, chains tools in sequence).

**Solution:** Tiered point-of-use tooling with standardized interfaces.

**Principles:**
1. **Personal rollaway cart** — daily tools at station, reflects individual work pattern.
2. **Shadow board by job family** — organized by context of use, not tool type. Shape IS the label (silhouette = no reading). Proximity follows workflow, not classification.
3. **Quick-change interface / workpiece carries state** — tools chain because of standardized interfaces. State coherence through shared workpiece, not tool-to-tool communication.
4. **Central storage = replenishment, not access point.** Daily tools never live in the crib.
5. **One-tool user** — invariant dedicated position at their station. Not on shared board.

**Key line:** "The answer to 'where is the boring head?' is always 'which machinist? which cell? which job?' It might be in three locations depending on who's asking, and THAT IS CORRECT."

---

## Synthesis

### Convergent Principles (multiple domains independently arrived at these)

**1. State travels with the work, not through a separate routing layer.**
- Kitchen: context propagates, workers stationary
- Surgery: specimen stays in field
- Machine Shop: workpiece carries state

Three domains independently concluded the work object itself is the state carrier. Routing through intermediaries is wrong. Maps directly to "data bus" concept already validated.

**2. Personal access surface is separate from shared/collective storage.**
- Kitchen: personal station vs. shared pass
- Surgery: Mayo stand vs. back table
- Machine Shop: personal rollaway vs. central crib
- Music: normalled strip vs. shared backline

ALL FOUR separated personal access from shared inventory. Software almost universally collapses these.

**3. Configuration happens before the moment of need, at lower cognitive load.**
- Kitchen: mise en place during setup
- Surgery: preference card before arrival
- Music: states composed at design time
- Machine Shop: rollaway stocked before shift

Every domain found that access decisions under pressure fail. Solution is always temporal separation.

**4. Individual arrangement reflects frequency/workflow, not canonical classification.**
- Kitchen: proximity earned by frequency
- Surgery: preference card encodes personal workflow
- Machine Shop: location is function of who's asking

No domain organized by intrinsic category. All organized by behavioral history or anticipated workflow.

---

### Novel Principles (uncommon in software UI/UX)

| Principle | Why Novel |
|---|---|
| Temporal separation of configuration from use | Software assumes navigation happens at use time. "Arrangement session" as first-class interaction mode is absent. |
| Phase-scoping the access surface | Software shows all tools simultaneously. The idea that the access surface SHRINKS to match current phase is structurally unusual. |
| Preference card as separable configuration artifact | Different from a settings object. Closer to a protocol than a preference — named, shareable, workflow-specific. |
| Same tool, multiple correct locations (determined by requester) | Software assumes one canonical location per tool. Multiple simultaneous correct answers is intended behavior. |
| Spatial reading / shape as label | Software relies on linguistic labels. Spatial reading (position, silhouette, gap as primary info) is absent. |
| Neutral coordination surface that no one owns | No common software analog. Shared clipboard is closest but never first-class. |

---

### Reframing Insights (different understanding of what the problem IS)

**1. The problem is not navigation. The problem is pre-positioning.**
Kitchen and Machine Shop dissolve navigation as a runtime event. If tool selection doesn't happen at the moment of need, the design problem becomes: what is the pre-positioning protocol, who initiates it, and when? Reframes from "how do users find tools" to "how does the system learn what to have ready."

**2. The problem is not tool access. The problem is role-appropriate state management.**
Music Production: the vocalist wants a RESULT, not a tool. The engineer wants a STATE, not tools. Only one role browses. Reframes from "how does a user access 15 tools" to "what are the 2-3 distinct relationships users have with the toolset?"

**3. The problem is not "too many tools." The problem is one access surface serving all phases.**
Surgery: access surface should shrink and change with the phase of work. Reframes from "how do you display 15 tools" to "what phase is the user in, and what 3-5 tools belong on the Mayo stand right now?"

**4. "Where is the tool" has no single correct answer — and that's correct behavior.**
Machine Shop: same tool, multiple correct locations. A single navigational tree is structurally wrong. The access layer is a query evaluated against (user identity, current context, job in progress).

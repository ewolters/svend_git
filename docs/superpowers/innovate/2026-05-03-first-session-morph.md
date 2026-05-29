# Morphological Analysis: First Session Implementation

**Date:** 2026-05-03
**Technique:** Morphological Analysis (Zwicky Box)
**Input:** `2026-05-03-first-session-transfer.md` (ER triage, mechanic, sommelier, PI transfer)

## Problem

How to implement the first-session protocol for SVEND — where Claude is the facilitator, the app is computation + document structure, and PROVA is Claude's invisible memory. The protocol has 5 phases: triage → negative-space diagnostic → first blood → parking lot → discharge artifact. This analysis decomposes the implementation into orthogonal dimensions and mechanically enumerates combinations to find non-obvious pairings.

---

## Morphological Box

| Dimension | Option 1 | Option 2 | Option 3 | Option 4 | Option 5 |
|-----------|----------|----------|----------|----------|----------|
| **D1: Data Ingestion** | Schema-first gatekeeper (user labels columns via drag-drop) | Inferential autopilot (auto-detect, one-click confirm) | Adversarial parse (competing heuristics, Claude breaks ties) | Raw-dump-and-narrate (load as strings, Claude asks questions) | Spec-file sidecar (companion YAML/JSON) |
| **D2: Persona Classification** | Data-shape-only inference | Explicit two-question gate | Behavioral triangulation (data + vocabulary + clicks) | Account-metadata preload (registration as Bayesian prior) | No classification (treat all identically) |
| **D3: Analysis Selection** | Decision-tree dispatch (hardcoded flowchart) | Claude free-pick (LLM reasons over catalog) | Shotgun volley (run 5 cheapest, user picks) | Pain-backward (start from problem, work backward) | Minimum-entropy selector (information gain) |
| **D4: Conversation Orchestration** | Rigid phase-gate (distinct URLs, no going back) | Claude-steered fluid dialogue (phases invisible) | User-driven menu (sidebar checklist) | Timer-boxed segments (wall-clock budget per phase) | Event-sourced state machine (backend drives, Claude renders) |
| **D5: Insight Translation** | Dollar-first framing (cost/savings before stats) | Analogy engine (physical analogies per industry) | Layered progressive disclosure (headline → stats → formula) | Dual-voice panel (technical left, operational right) | User-authored summary (user explains, Claude corrects) |
| **D6: Discharge Artifact** | One-page PDF brief (auditor-ready) | Living dashboard link (persistent URL) | Email-forward package (pre-drafted to boss) | Structured data export (JSON/YAML) | Annotated session transcript (conversation as audit trail) |

**Total design space:** 15,625 combinations

---

## Random Combinations Evaluated

### Combination 1
| Dimension | Selection |
|-----------|-----------|
| D1 | Schema-first gatekeeper |
| D2 | Data-shape-only inference |
| D3 | Shotgun volley |
| D4 | Claude-steered fluid dialogue |
| D5 | Analogy engine |
| D6 | Living dashboard link |

**Feasible:** With modification — schema-first makes shotgun wasteful; replace with filtered volley (3 analyses the schema supports).

**Interesting:** Analogy engine + living dashboard. Analogies are normally ephemeral scaffolding, but a living dashboard makes them persistent. First session: "your Cpk is like a car drifting between lanes." Week 3: the user has internalized it, the dashboard shows real metrics. The analogy was scaffolding, the dashboard is the building.

---

### Combination 2
| Dimension | Selection |
|-----------|-----------|
| D1 | Schema-first gatekeeper |
| D2 | No classification |
| D3 | Decision-tree dispatch |
| D4 | Event-sourced state machine |
| D5 | Dual-voice panel |
| D6 | One-page PDF brief |

**Feasible:** Yes.

**Interesting:** The "enterprise procurement" combination. No classification means the demo works identically for the QE director and the quality engineer watching together. Decision-tree is auditable. Event-sourced logs every transition. Dual-voice speaks to both simultaneously. One-page PDF goes into the approval folder. **This survives a vendor evaluation.** Boring, traceable, transparent. Exactly what Dave needs for budget approval. Risk: Ray bounces in 90 seconds — zero personality.

---

### Combination 3
| Dimension | Selection |
|-----------|-----------|
| D1 | Schema-first gatekeeper |
| D2 | Data-shape-only inference |
| D3 | Claude free-pick |
| D4 | Claude-steered fluid dialogue |
| D5 | User-authored summary |
| D6 | Annotated session transcript |

**Feasible:** Yes. Most conceptually coherent in the batch.

**Interesting:** D5+D6 is the buried treasure. User explains the chart back ("so this is telling me my plating thickness is drifting high?"), Claude corrects. The transcript captures that exchange. **This is tribal knowledge capture happening in real time without the user knowing.** They think they're learning. PROVA is recording their domain interpretation layered on statistical output. Six months later, new QE reads the transcript: they don't just see "Cpk = 0.83" — they see "Maria said this means the plating bath chemistry needs adjustment every 3rd shift because of temperature drop." The knowledge-walking-out-the-door problem solved as a side effect of UX.

---

### Combination 4
| Dimension | Selection |
|-----------|-----------|
| D1 | Schema-first gatekeeper |
| D2 | No classification |
| D3 | Claude free-pick |
| D4 | Event-sourced state machine |
| D5 | Dual-voice panel |
| D6 | Living dashboard link |

**Feasible:** With modification — Claude free-pick and event-sourced state machine fight for control. Fix: Claude picks analysis, state machine governs phase transitions.

**Interesting:** Dual-voice + living dashboard. Technical left panel for QE, operational right panel for manager — the dashboard becomes a shared artifact with two reading modes. Most dashboards are single-audience. This accidentally becomes a communication bridge between floor and office.

---

### Combination 5
| Dimension | Selection |
|-----------|-----------|
| D1 | Raw-dump-and-narrate |
| D2 | No classification |
| D3 | Shotgun volley |
| D4 | Rigid phase-gate |
| D5 | Analogy engine |
| D6 | Structured data export |

**Feasible:** No. Three compounding problems: raw-dump + shotgun = analyzing data whose structure is unconfirmed. Rigid phase-gate = can't backtrack when results reveal misinterpreted columns. JSON output = useless to every persona.

**Interesting despite infeasibility:** Raw-dump + analogy engine. When Claude doesn't know what columns mean, analogies become a data disambiguation tool: "This column — is it more like a temperature reading or a count of defects?" Novel use of analogies nobody would design intentionally.

---

### Combination 6
| Dimension | Selection |
|-----------|-----------|
| D1 | Adversarial parse |
| D2 | Behavioral triangulation |
| D3 | Claude free-pick |
| D4 | Claude-steered fluid dialogue |
| D5 | Layered progressive disclosure |
| D6 | One-page PDF brief |

**Feasible:** Yes. The "power user" combination.

**Interesting:** Maximum intelligence, maximum inference. Competing heuristics on data, tracking vocabulary and clicks on user, Claude reasoning freely, output meeting you at your level. Gets smarter faster than any other combination. Risk: feels like surveillance. The magic must be invisible — which Claude-steered enables, but then the user can't understand how the system knew so much. Trust problem for the auditor-defense use case where transparency is the product.

---

### Combination 7
| Dimension | Selection |
|-----------|-----------|
| D1 | Schema-first gatekeeper |
| D2 | Account-metadata preload |
| D3 | Decision-tree dispatch |
| D4 | User-driven menu |
| D5 | Layered progressive disclosure |
| D6 | Annotated session transcript |

**Feasible:** Yes.

**Interesting:** User-driven menu + annotated transcript = self-documenting process compliance. The user chose which phases to engage with, the transcript records those choices. If they skipped the diagnostic, it shows. Account-metadata + decision-tree means the system pre-selects the analysis path from registration data, user-driven menu lets them override, transcript captures both recommendation and deviation. For Dave: "The system recommended Gage R&R based on your data, but you ran capability instead. Here's what both would have shown." Auditors love this reasoning chain.

---

### Combination 8
| Dimension | Selection |
|-----------|-----------|
| D1 | Adversarial parse |
| D2 | Data-shape-only inference |
| D3 | Pain-backward |
| D4 | Event-sourced state machine |
| D5 | Dollar-first framing |
| D6 | Structured data export |

**Feasible:** With modification — structured data export kills the emotional momentum of dollar-first framing. Replace with one-page PDF or email-forward package.

**Interesting:** Pain-backward + dollar-first is the most commercially lethal pairing in the entire box. "What's your biggest headache?" → work backward to the analysis → present the result in dollars. Three-step path from complaint to purchase justification. Every other strategy starts from data and works toward value. This starts from pain and works toward proof. **This is the focus group findings implemented as a selection algorithm.** The personas already said "tribal knowledge" and "show the dollars" — this random draw surfaced that as architecture.

---

## Three Most Promising

### 1. Combination 3 — The Knowledge Capture Engine
Schema-first + Data-shape inference + Claude free-pick + Claude-steered + **User-authored summary + Annotated transcript**

The tribal-knowledge-capture-as-side-effect property is the single most strategically aligned insight. It solves "knowledge walking out the door" *during* the first session without adding steps. The user thinks they're learning; PROVA records domain interpretation layered on statistical output. The discharge artifact gets more valuable over time as sessions accumulate. This is the only combination where the product's core value proposition is emergent from UX design rather than explicitly engineered.

### 2. Combination 6 — The Intelligence Engine
Adversarial parse + Behavioral triangulation + Claude free-pick + Claude-steered + Layered progressive disclosure + One-page PDF

Fastest time-to-value. System infers everything from every signal. Needs careful trust management — the magic must feel helpful, not surveilling. Best for the Ray persona (job shop owner who wants to skip the sales call and just get results).

### 3. Combination 8 (modified) — The Commercial Kill Shot
Adversarial parse + Data-shape inference + **Pain-backward** + Event-sourced state machine + **Dollar-first framing** + One-page PDF (modified from JSON)

Most likely to convert a trial to paid in one session. Pain-backward + dollar-first is the focus group insight implemented as architecture. The event-sourced state machine logs the entire reasoning chain from pain to proof — which IS the audit trail.

---

## Unexpected Dimension Interactions

### D5 × D6 is the most consequential interaction
- User-authored summary + annotated transcript = accidental knowledge capture (Combo 3)
- Dollar-first framing + structured data export = emotional momentum destroyed by format (Combo 8)
- Analogy engine + living dashboard = ephemeral scaffolding made persistent (Combo 1)
- Dual-voice panel + living dashboard = single artifact, two audiences (Combo 4)

**The discharge artifact doesn't just record the insight translation — it amplifies or kills it.** These two dimensions should be designed as a coupled pair.

### D1 × D3 has a hidden constraint
Schema-first gatekeeper gives enough information to make shotgun volleys wasteful. Raw-dump means you don't know enough for decision-tree dispatch. The ingestion strategy constrains which analysis selection strategies are coherent.

### D2 × D4 interact more than expected
No-classification + Claude-steered means Claude must maintain warmth without persona cues — harder than it sounds. Behavioral triangulation + rigid phase-gate means you gather behavioral signals but can't act on them mid-session.

### Pain-backward (D3) wants dollar-first (D5)
Starting from pain and ending with stats is incoherent. Starting from pain and ending with dollars is a closed loop. These have natural affinity the morphological box treats as independent — suggesting D3 and D5 should be partially coupled in final design.

---

## Design Implications

The morphological box surfaced three implementation archetypes the transfer analysis didn't anticipate:

1. **The Knowledge Capture Engine** — where the first session's primary value isn't the analysis but the domain interpretation captured alongside it (Combo 3)
2. **The Commercial Kill Shot** — where analysis selection is driven by pain, not data shape, and every finding is denominated in dollars (Combo 8 modified)
3. **The Enterprise Evaluator** — where transparency, auditability, and dual-audience communication trump intelligence and warmth (Combo 2)

These aren't mutually exclusive. A mature implementation could use pain-backward + dollar-first for triage, user-authored summary for knowledge capture during first blood, and one-page PDF as artifact — cherry-picking across combinations. But the morph reveals that the *reason* to combine them is the D5×D6 interaction: the insight translation and artifact format must be designed together.

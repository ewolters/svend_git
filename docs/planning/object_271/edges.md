# Object 271 — Edge Exploration

**Date:** 2026-03-28
**Purpose:** Push boundaries on what the graph-as-product identity enables. These are not commitments — they're vectors to evaluate.

---

## Premise

If the graph accumulates knowledge over the lifetime of an org, and that knowledge compounds, then SVEND isn't a SaaS tool. It's an asset management platform. The asset is process knowledge. The tools are how you build the asset. The subscription is access to the instruments that maintain and grow it.

---

## Edge 1: The Graph Is Exportable — And That's a Feature, Not a Risk

Most SaaS companies fear data portability because it lowers switching costs. But a process knowledge graph exported as structured JSON-LD or RDF is more valuable than any report. An org that can hand their auditor a complete, provenanced, calibrated process model — with every edge traced to its source DOE or investigation — has something no competitor's customer can produce.

**The play:** Don't lock the graph in. Make export a premium feature. The graph in SVEND is alive (Bayesian updates, SPC monitoring, staleness detection). The exported graph is a snapshot — useful for audits and transfers, but it doesn't learn. The living graph is the lock-in. The export is the trust signal.

**Implications:**
- Export format becomes a de facto standard if adopted widely enough
- Auditors learn to expect graph-structured evidence
- Competitors can't match the export without building the graph engine
- Trust signal: "we're confident enough in our product to let you leave with your data"

---

## Edge 2: Multi-Process, Multi-Site Graphs Change the Architecture

GRAPH-001 currently says "one graph per org." But a real manufacturer has:
- Injection molding line 1 (Process A)
- Injection molding line 2 (Process B — same tooling, different machine)
- Paint shop (Process C)
- Assembly (Process D)

Some nodes span processes (ambient humidity affects all of them). Some edges are process-specific.

**Three options:**

| Model | Structure | Pro | Con |
|-------|-----------|-----|-----|
| Flat | One graph, process tags on nodes | Simple | Gets noisy at scale |
| Hierarchical | Org → Site → Process → Subgraph | Clean scoping | Rigid, hard to model cross-process |
| Federated | Each process is its own graph, shared nodes create cross-graph edges | Cross-process discovery | More complex persistence |

**The federated model is the most interesting.** It means an investigation in the paint shop can discover that a humidity node it shares with injection molding has a causal edge the injection team hasn't seen. Cross-process discovery. No competitor even conceptualizes this.

**Implications:**
- Shared nodes need identity resolution ("is paint shop humidity the same physical sensor as molding humidity?")
- Cross-graph edges need governance (who can see what across process boundaries?)
- Site-level views aggregate multiple process graphs
- Enterprise tier feature: federated graphs. Team tier: single graph.

---

## Edge 3: The Graph Enables a Marketplace

If process graphs are structured, anonymizable, and exportable, then industry-level knowledge becomes possible.

**What this looks like:**
- "Injection molding — ABS — common failure modes" as a template graph a new org imports as their starting skeleton
- ILSSI providing certified baseline graphs for training programs
- Gemba Exchange users sharing anonymized graph fragments as answers to process questions
- Industry associations publishing reference graphs for their domains

**The ecosystem play:**
```
SVEND produces graphs
Gemba Exchange shares graph knowledge
ILSSI certifies graph quality
The graph format becomes a standard
```

**Implications:**
- Anonymization must strip proprietary data while preserving structural topology
- Template graphs accelerate onboarding: "don't start from zero, start from industry baseline"
- Gemba Exchange transforms from Q&A forum to knowledge exchange (literally exchanging graph fragments)
- Network effects: more orgs on SVEND = better baseline templates = faster onboarding for new orgs
- This is the flywheel that justifies the ecosystem architecture

---

## Edge 4: The Graph IS the AI Context

When you ask an LLM "why are we getting short shots on line 3?" — the right context isn't a prompt template. It's the subgraph around the short_shots node: upstream causes, their calibration state, recent SPC signals, open investigations, evidence stacks.

The graph is the perfect RAG substrate because it's already structured, provenanced, and quantified. No embedding search needed — you traverse the graph.

**What this looks like:**
- "Ask about this node" as a contextual action on any graph element
- "Explain this edge's evidence" — LLM reads the evidence stack and narrates it
- "What should I investigate next?" — LLM reads gap report and prioritizes
- "Summarize this process for a new engineer" — LLM walks the graph and produces a narrative
- Investigation autopilot: LLM reads the scoped subgraph and suggests hypotheses based on graph structure + gaps

**Implications:**
- Guide (LLM chat) becomes graph-native, not a separate surface
- No separate knowledge base needed — the graph IS the knowledge base
- LLM responses are traceable: "I said this because of edge X with evidence Y"
- This is how you justify the AI premium without it feeling bolted on
- Synara + LLM: Synara handles the math (Bayesian updates, propagation), LLM handles the narrative (explanation, suggestion, synthesis)

---

## Edge 5: The Graph Inverts the Sales Conversation

**Old pitch:** "We have 200 statistical tests and FMEA and SPC for $299/mo."

**New pitch:** "What does your organization know about its processes? Can you show me?"

Nobody can answer that question today. Their knowledge is in people's heads, scattered Excel files, tribal lore, and binders on shelves. SVEND makes it visible, structured, and auditable.

**The demo isn't showing tools.** It's showing an empty graph and saying: "This is what you don't know yet. Every tool in the platform fills in a piece."

**Reframes the sale:**
- From "do you need another tool?" → "do you want to see your process?"
- From feature comparison → category creation
- From cost justification → asset building
- From "replace Minitab" → "build something Minitab can't"

**Especially powerful for ILSSI audience:** These are people who already think in terms of process knowledge. They understand that tribal knowledge walks out the door when someone retires. The graph is the answer to "how do we keep what we know?"

**Implications:**
- Marketing language shifts from tools to knowledge
- Case studies focus on graph maturity, not feature usage
- "Time to first calibrated edge" becomes the key activation metric
- Free tier could include graph visualization (see what you don't know) with tools gated behind paid tiers

---

## Edge 6: Certification Against the Graph

If the graph represents what an org knows about its process, then ILSSI (or any certification body) could certify against the graph.

**Not** "did you fill out the FMEA form" but:
- Does your process model have calibrated edges for all critical-to-quality characteristics?
- Are your detection edges backed by forced failure tests?
- Is your staleness rate below threshold?
- Do your investigations produce writeback to the graph?
- Is your graph entropy decreasing over time? (Organization is learning)

**This is a fundamentally different kind of certification:**
- Evidence-based, not form-based
- Quantitative, not qualitative
- Continuously auditable, not point-in-time
- The graph IS the audit evidence

**What ISO 9001 aspires to but can't enforce** because the tools don't exist. Until now.

**Implications:**
- ILSSI could define "Graph Maturity Levels" (Level 1: FMEA skeleton only, Level 2: some calibrated edges, Level 3: full calibration + SPC monitoring, Level 4: predictive capability via value propagation)
- Certification is automated: the CI Readiness Score already measures most of this
- This creates a new revenue stream: certification fees
- It also creates a defensible standard: once orgs certify against graph maturity, they can't leave without losing their certification status
- The Auditor Portal already shows graph evidence by ISO clause — certification is the formalization of what's already there

---

## Risk Assessment

| Edge | Near/Med/Long | Risk | Reward |
|------|---------------|------|--------|
| 1. Export | Near | Low — it's a feature, not architecture | Trust signal, auditor value |
| 2. Multi-process | Near-Med | Medium — architecture decision, needs to be right | Enterprise unlock, cross-process discovery |
| 3. Marketplace | Long | High — requires critical mass, anonymization is hard | Network effects, ecosystem flywheel |
| 4. AI context | Near | Low — graph is already structured for this | LLM differentiation, natural upsell |
| 5. Sales inversion | Near | Low — it's positioning, not code | Category creation, ILSSI alignment |
| 6. Certification | Med-Long | Medium — requires ILSSI partnership, standard definition | New revenue stream, defensible moat |

---

## Which Ones Pull Together

Edges 4 + 5 are the near-term identity shift. The graph as AI context makes the product smarter. The inverted sales conversation makes the positioning clearer. Neither requires significant architecture changes.

Edge 2 (multi-process) is the near-term architecture decision that unblocks enterprise. This needs to be in the GRAPH-001 spec before we build.

Edges 1 + 3 + 6 are the medium-to-long-term ecosystem play. Export enables marketplace enables certification. They chain. But they don't block the core build.

---

## Open Questions for the Conservative Session

1. Does multi-process architecture (Edge 2) need to be in GRAPH-001 v0.1 or can it be v0.2?
2. Is graph-native LLM (Edge 4) a distraction from the core build or an accelerant?
3. Does the sales inversion (Edge 5) risk alienating users who just want a tool?
4. Is certification against the graph (Edge 6) presumptuous without a larger install base?
5. Where does the conservative session see risk that this session is blind to?

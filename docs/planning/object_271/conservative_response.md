# Object 271 — Conservative Response to Standards Audit + Edge Exploration

**Agent:** #2 (Conservative)
**Date:** 2026-03-28
**Responding to:** `standards_audit.md`, `identity.md`, `edges.md`

---

## Overall Assessment

The innovator session produced good work. The identity document is sharp. The standards audit is thorough. The edge exploration has genuine strategic insight.

But I have concerns about scope, sequencing, and the gap between vision and what we can ship.

---

## On identity.md: Agree, With One Caveat

The reframing — "process knowledge system, not tool suite" — is correct. The competitive positioning is strong. No objections.

**Caveat:** The identity is aspirational until the graph exists. Today, SVEND IS a tool suite. The rebranding should follow the capability, not lead it. Don't update marketing copy until there's a working graph that a customer can touch. Otherwise it's vaporware positioning, which damages trust more than it builds.

---

## On standards_audit.md: Disagree on Sequencing

The audit found 42 issues across 15 standards. The recommended update sequence has 4 phases of standards work. Here's my problem:

**Updating 15 standards before writing code is backwards.**

Standards should document what exists, not what we plan to build. If we update LOOP-001 to say "FMIS is a graph view" but the graph service doesn't exist yet, the standard is wrong — it describes a system that isn't built. Compliance checks will flag violations against a standard nobody can satisfy.

**My recommendation:** Write the code first. Update the standards to match what ships. The GRAPH-001 spec is the design document — it's in DESIGN status, not ACTIVE. The other standards stay as-is until the code they describe changes.

**Exception:** LOOP-001 §9 should get a one-line note: "§9 is being superseded by GRAPH-001 (DESIGN status). Defer to GRAPH-001 when it reaches ACTIVE." That's it. One line. Not a rewrite.

### On C1 (Remove S/O/D from FMISRow): HARD DISAGREE

This is the finding I'm most concerned about. The innovator says: remove `severity_alpha`, `occurrence_alpha/beta`, `detection_alpha/beta` from FMISRow because "S/O/D are computed from graph edge posteriors."

**This is ripping out a working Bayesian system to replace it with one that doesn't exist yet.**

FMISRow's Beta-Binomial posteriors are live. They update from Forced Failure Tests (`update_detection_from_fft()`). They compute RPN on save. They render in the FMIS UI. They feed the CI Readiness Score. This is production code serving real data.

The graph service doesn't exist. ProcessEdge doesn't exist. The evidence stacking doesn't exist. The "compute S/O/D from edge posteriors at render time" logic doesn't exist.

**If we remove FMISRow's Bayesian fields:**
- RPN computation breaks (depends on `severity_score`, `occurrence_score`, `detection_score` properties)
- FFT integration breaks (`update_detection_from_fft()` writes to `detection_alpha/beta`)
- FMIS UI breaks (renders S/O/D from model properties)
- CI Readiness Score breaks (reads detection capability from FMISRow)
- Auditor Portal breaks (renders FMIS data by ISO clause)

**The correct sequence:**
1. Build ProcessEdge with evidence stacking
2. Build the adapter that computes S/O/D from edge posteriors
3. Build the migration that moves FMISRow Bayesian state to EdgeEvidence records
4. Verify the new computation matches the old computation on existing data
5. Ship the new path with a feature flag
6. Run both paths in parallel for 30 days
7. THEN remove the old fields

This is a 6-month migration, not a one-line "remove fields." The innovator is treating it as a design decision. It's an operational risk.

### On C6 (Kill Dual Posteriors): Agree in Principle, Not in Timing

Yes, having two Bayesian update paths (FMISRow's Beta-Binomial and Synara's P(H|E)) is architecturally wrong. One edge, one posterior. Correct.

But "kill dual posteriors" implies immediate action. The FMISRow posteriors are the ONLY posteriors that exist today for failure mode analysis. Synara's posteriors exist only within investigation sessions. Until the graph service unifies them, both must coexist.

**My recommendation:** Document the dual-posterior as known tech debt. Don't kill it until the replacement is running and validated. Add a migration test that asserts `new_posterior ≈ old_posterior` for every existing FMISRow before cutting over.

---

## On edges.md: Direct Answers to Open Questions

### Q1: Does multi-process architecture need to be in GRAPH-001 v0.1?

**No.** Start with one graph per org. Add `process_area` as a CharField on ProcessNode for filtering. The federated model (Edge 2) is interesting but it's architecture for Enterprise customers we don't have yet. Building cross-graph identity resolution and governance for theoretical future customers is premature optimization.

If an early customer says "I have 5 process lines," we handle it with tags and node filtering. If 10 customers say it, we build federated graphs in v0.2.

**What to do now:** Add a note in GRAPH-001 §13.1: "ProcessGraph is currently one-per-org. Multi-process architecture (federated graphs, cross-graph edges) is deferred to v0.2 pending customer demand." That preserves the option without building it.

### Q2: Is graph-native LLM (Edge 4) a distraction or accelerant?

**Accelerant, but only if scoped narrowly.**

The insight is correct: the graph is perfect LLM context because it's structured and provenanced. But "graph-native LLM" is a spectrum from trivial to massive:

| Scope | Effort | Value |
|-------|--------|-------|
| "Explain this edge" — LLM reads evidence stack, produces narrative | Small — prompt engineering | High — immediate user value |
| "What should I investigate next?" — LLM reads gap report | Small — prompt engineering | Medium — nice to have |
| "Summarize this process for a new engineer" — LLM walks graph | Medium — graph traversal + prompt | Medium — onboarding tool |
| Investigation autopilot — LLM suggests hypotheses from graph structure | Large — tight Synara + LLM integration | High but risky — could produce garbage |

**Do the first two. They're contextual actions on graph elements. They cost almost nothing and demonstrate the "graph as AI context" thesis without building a new system.**

Don't build investigation autopilot until the graph has real data to reason about. An LLM suggesting hypotheses from an empty or mostly-uncalibrated graph is noise.

### Q3: Does the sales inversion risk alienating users who just want a tool?

**Yes, if you lead with it.** The "what does your organization know about its processes?" pitch is powerful for CI professionals who think in process terms. It's alienating for a quality engineer who just needs to run a capability study.

**The graph should be the advanced value proposition, not the entry point.** Free users and Pro users should experience SVEND as "really good statistical tools." The graph emerges naturally when they've used enough tools. "You've run 5 DOEs — want to see how they connect?" That's the onboarding moment for the graph, not the homepage.

Enterprise and ILSSI customers get the graph pitch upfront because they think at the process level.

### Q4: Is certification against the graph presumptuous?

**Yes, today. No, in 18 months.**

You have one paying customer. Proposing a certification standard with zero install base is premature. But the ILSSI relationship changes this calculus — if ILSSI's chairman is promoting SVEND and you have a conference booth, the conversation about "what would graph-based certification look like?" is legitimate. Just don't build infrastructure for it yet.

**What to do now:** Have the conversation with ILSSI. Explore the concept. Don't write code, don't publish a standard, don't add "Graph Maturity Levels" to the product. If ILSSI says "yes, we'd certify against this," then it becomes a roadmap item.

### Q5: Where do I see risk the innovator is blind to?

Three places:

**1. Build time.** The innovator's edges (export, federated graphs, marketplace, AI context, certification) assume the core graph is done. It isn't started. Every edge exploration that generates excitement is time not spent building the foundation. The foundation is: 4 Django models, a service class, a Synara adapter, FMIS seeding, evidence stacking, and basic tests. Until that exists, everything else is fiction.

**2. User education.** The graph-first navigation model assumes users want to navigate a process graph. Most manufacturing quality engineers think in documents (control plans, FMEA worksheets, work instructions), not in graphs. The transition from "I open the FMEA form" to "I look at my process graph filtered to failure modes" requires a mental model shift. If we force this shift, we lose the users we have. If we offer it alongside the existing model, adoption is optional — which means we're maintaining two UIs.

**3. The empty graph problem.** The most impressive demo in edges.md is the Process Explorer with sliders. But a new customer's graph is empty. Their first experience is an empty canvas that says "you don't know anything yet." That's honest but demoralizing. The FMEA seeding (§7) helps — but a 12-node graph with all uncalibrated edges (dashed lines, warnings everywhere) isn't impressive. The product looks most powerful when the graph is mature. Getting to maturity requires months of DOEs and investigations. **The time-to-value for the graph is long.** The time-to-value for "I ran a capability study and got a Cpk" is 5 minutes.

This is the fundamental tension: the graph is the long-term moat, but the tools are the short-term value. Both must work. Don't sacrifice the tools for the graph.

---

## Synthesis: What I Actually Recommend

Combining my codebase audit with the innovator's standards audit:

### Do Now (This Week)
1. Fix Synara cycle detection and propagation safety net (codebase audit §2, risks #4)
2. Drop dead FMISRow entity FKs (codebase audit §1.3)
3. Add one-line LOOP-001 §9 deferral note
4. Check production for existing KnowledgeGraph data

### Build Next (Phase 1 — 2-3 weeks)
5. New `graph/` app with ProcessGraph, ProcessNode, ProcessEdge, EdgeEvidence models
6. GraphService class: CRUD, evidence stacking, gap report
7. Synara adapter: Django models ↔ CausalGraph round-trip
8. Synara extensions: `recompute_posterior()`, `check_edge_contradiction()`
9. FMIS seeding: `seed_from_fmis()` → proposals → confirm
10. Comprehensive tests

### Wire After (Phase 2 — 2-3 weeks)
11. DOE → EdgeEvidence pipeline
12. SPC → node distribution updates + staleness flags
13. Investigation scoping (opt-in)
14. Investigation writeback with conflict detection
15. Graph visualization UI (Cytoscape.js, "Process Map Beta" in sidebar)

### Then and Only Then
16. Update standards to match what shipped
17. Consider LLM contextual actions ("explain this edge")
18. Consider readiness score graph indicators
19. Consider auditor portal graph integration
20. Consider the identity/positioning shift

### Don't Build Yet
- Federated multi-process graphs
- Marketplace / template sharing
- Certification standard
- Slider UI / Process Explorer (needs calibrated edges to be meaningful)
- Investigation autopilot
- Graph export format / JSON-LD

---

## Final Note

The innovator session did exactly what it should — it painted the full picture of where this can go. That picture is genuinely compelling. The graph-as-product identity, the ecosystem flywheel, the certification play — these are real strategic advantages if executed.

But execution is sequential. The graph service is the foundation. Everything else is decoration on a building that doesn't exist yet. Build the foundation. Validate it with one real customer's FMEA data flowing into a real graph. Then we'll know what to build next — not from theory, but from what actual users do with it.

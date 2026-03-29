# Object 271 — Standards Audit

**Date:** 2026-03-28
**Scope:** All 45 standards in docs/standards/ evaluated against GRAPH-001 and the three-thing model (Graph / Loop / QMS)
**Method:** Systematic review of every domain-relevant standard for contradictions, gaps, and required updates

## The Three-Thing Model

```
Graph  — what the organization knows (persistent process knowledge)
Loop   — how the organization learns (Signal → Investigate → Standardize → Verify)
QMS    — what the organization must demonstrate (compliance lens on the graph)
```

Standards written before GRAPH-001 conflate these concerns. The audit below identifies every point where they need separation.

---

## Summary

| Severity | Count | Standards affected |
|----------|-------|--------------------|
| CRITICAL | 13 | LOOP-001, CANON-001, CANON-002, QMS-001, DAT-001, RISK-001, SAF-001, NB-001 |
| MAJOR | 21 | All of the above + STAT-001, DSW-001, TRN-001, ORG-001, ARCH-001, MAP-001, QUAL-001 |
| MINOR | 8 | Various |
| **Total** | **42** | **15 standards** |

30 standards require no changes (infrastructure, security, billing, frontend, etc. — not domain-relevant).

---

## CRITICAL Findings

### C1. LOOP-001 §8.7 — FMIS stores its own S/O/D state (parallel to graph)

**Current:** FMISRow has `severity_alpha`, `occurrence_alpha/beta`, `detection_alpha/beta` fields — its own Beta-Binomial and Dirichlet posteriors.

**Conflict:** GRAPH-001 §4.5 says "one edge, not two." S/O/D are computed from graph edge posteriors. FMIS is a view, not a data store.

**Required:** Remove S/O/D storage fields from FMISRow. Add FKs to ProcessNode (failure_mode, cause, effect). S/O/D computed at render time from edge posteriors.

### C2. LOOP-001 §9 — Full DPM spec now superseded by GRAPH-001

**Current:** 8 subsections defining schema, gap exposure, investigation subset, simulation.

**Required:** Replace with condensed reference section pointing to GRAPH-001. Keep only LOOP-specific integration points (signal sources, investigation scoping, writeback triggers).

### C3. LOOP-001 §16.5 — FMIS View treated as separate system

**Current:** Describes FMIS as a display with its own S/O/D computation from FMIS-specific posteriors.

**Required:** Reframe as graph lens (GRAPH-001 §15.2). S/O/D rendered from edge posteriors. Editing the view edits the graph.

### C4. CANON-001 §2.3.2 — Investigation graph treated as ephemeral

**Current:** "The investigation is the graph" — implies graph created/destroyed per investigation.

**Required:** Investigations operate on subsets of the persistent process graph (GRAPH-001 §8). The graph predates and outlives any investigation.

### C5. CANON-001 §2.4 — QMS as separate lifecycle with own evidence

**Current:** Layer 3 Systems (NCR/CAPA) "can generate evidence" independently.

**Required:** QMS is a compliance lens ON the graph. All evidence — from investigations, NCRs, CAPAs, observations — feeds the unified graph through GraphService.add_evidence().

### C6. CANON-002 Intro — Dual Posteriors (two Bayesian systems)

**Current:** "Two separate Bayesian updates occur" — one for FMIS (unweighted Beta-Binomial), one for Synara (weighted).

**Required:** ONE posterior per edge via unified Synara. "Dual posterior" language deprecated. One posterior, two visualizations (FMIS shows detection as 1-10 integer; Synara shows full distribution).

### C7. CANON-002 §5.3.4 — FMEA as tool that "maps to" graph

**Current:** FMEA has "graph mapping" — implies FMEA is separate and integrates with graph.

**Required:** FMEA IS a graph view (GRAPH-001 §7.3). No separate FMEA data structure. Editing FMEA edits the graph.

### C8. QMS-001 §1.1 — QMS as knowledge owner

**Current:** "Five integrated modules — FMEA, RCA, A3, VSM, Hoshin — form a closed-loop continuous improvement cycle."

**Required:** QMS enforces compliance structure around graph operations. It doesn't own FMEA or RCA — those are graph views and investigation tools respectively. QMS audits the knowledge, it doesn't create it.

### C9. QMS-001 §1.2 — FMEA listed as "QMS module"

**Current:** FMEA in scope as a QMS module.

**Required:** Separate into modules QMS owns (NCR, CAPA) and modules QMS uses via graph (FMEA = graph view, RCA = investigation tool, A3 = investigation output).

### C10. QMS-001 §4.1 — FMEA as standalone QMS tool

**Current:** Full FMEA spec within QMS context. No reference to graph, investigations, or Bayesian posteriors.

**Required:** Keep AIAG scoring and compliance aspects. Add references to GRAPH-001 §7 for graph integration. Clarify QMS-001 §4.1 covers compliance aspects only.

### C11. QMS-001 §4.2 — RCA as standalone QMS tool

**Current:** RCA treated only as QMS tool (causal chain, AI critique).

**Required:** RCA is both a QMS tool (compliance documentation) AND a graph-building investigation tool. RCA chains become causal edges via GRAPH-001 §8.

### C12. DAT-001 §9.5 — KnowledgeGraph/Entity/Relationship now deprecated

**Current:** Defines KnowledgeGraph as canonical graph model with Entity and Relationship subtypes.

**Required:** Mark deprecated. Add reference to GRAPH-001 §13 for replacement models (ProcessGraph, ProcessNode, ProcessEdge, EdgeEvidence).

### C13. DAT-001 §6.1 — KnowledgeGraph ownership constraint invalid

**Current:** User XOR Tenant constraint on KnowledgeGraph.

**Required:** ProcessGraph is Tenant-only (one per org), not User XOR Tenant. Update constraint documentation.

---

## MAJOR Findings

### M1. LOOP-001 §1.4 — Conflates Graph/Loop/QMS concerns in entropy discussion
Add note separating the three concerns.

### M2. LOOP-001 §3.1 — Signal sources incomplete
Add: graph edge staleness, graph edge contradiction, graph expansion signal.

### M3. LOOP-001 §8.2 — Operational definitions don't reference GRAPH-001 ProcessNode schema
Link FMIS entity FKs to GRAPH-001 ProcessNode records.

### M4. LOOP-001 §10.2 — CI Readiness Score missing graph health metrics
Add: graph calibration coverage, stale edge resolution time, contradiction resolution time, gap prioritization alignment.

### M5. LOOP-001 §16.3 — Investigation workspace "process model subset" vague
Clarify as GRAPH-001 ProcessNode/ProcessEdge references in sidebar.

### M6. LOOP-001 §16.8 — Signal triage missing graph signal types
Add display rows for staleness, contradiction, expansion signals.

### M7. CANON-001 §2.3.2 — Missing reference to GRAPH-001 for graph schema
Add link between investigation graph and persistent process graph.

### M8. CANON-001 §5.6.2 — Metric cascade doesn't reference process model entities
Cascade dependencies should respect graph causal edges.

### M9. CANON-002 §2.3 — Source Rank Registry conflates FMEA assertions with evidence
Uncalibrated FMEA assertions populate structure; calibrated FMEA is evidence. Weight 0.60 applies only post-investigation.

### M10. CANON-002 §5.3 — FMEA vs FMIS conflated as one tool
Create separate contracts: standalone AIAG FMEA vs investigation-native FMIS.

### M11. QMS-001 §4.1.2 — AP scoring unclear with Bayesian posteriors
Clarify how AP lookup table works when S/O/D are continuous posterior means (0-10) vs discrete integers.

### M12. QMS-001 §5.1 — Cross-module data flow uses outdated terminology
Replace "Finding" language with graph operations. "FMEA → Process Graph (node/edge creation)."

### M13. QMS-001 §5.2 — Accountability mechanisms not linked to graph scoping
Signals scope graph subsets, commitments target graph entities, completions write back evidence.

### M14. QMS-001 §5.3 — Evidence hooks bypass CANON-002 weighting methodology
Fixed confidence values (0.5-0.95) don't follow source rank computation. Reference CANON-002 §3.

### M15. STAT-001 §8.2 — DOE evidence flows to "Synara" but not to graph edges
DOE results must create EdgeEvidence records on ProcessEdge.

### M16. DSW-001 §6.1 — Evidence linking "optional"
When project has a graph, analysis must create EdgeEvidence on relevant edges.

### M17. TRN-001 — No mechanism to derive training needs from graph gaps
Add section on training gaps informed by uncalibrated edges, staleness, contradictions.

### M18. DAT-001 §9.1 — Project.graph FK references deprecated KnowledgeGraph
Update to ProcessGraph.

### M19. DAT-001 §10 — Migration strategy missing graph model phase
Add Phase 4: KnowledgeGraph → ProcessGraph migration.

### M20. ORG-001 §2 & §4.4 — References deprecated KnowledgeGraph for org-wide graph
Replace with ProcessGraph (Tenant-only, not User XOR Tenant).

### M21. Various infrastructure standards — Registration and placement

- **MAP-001 §4.2:** Standards registry missing GRAPH-001 entry.
- **MAP-001 §5.2:** Module map missing graph/loop app entry.
- **ARCH-001 §5:** Graph app placement undefined in layer boundaries.
- **QUAL-001 §2:** Missing GRAPH-001 normative reference.
- **QUAL-001 §5:** Calibration evidence not connected to graph edge posteriors.

---

## MINOR Findings

### m1. LOOP-001 §1.2 — Scope bullet for §9 needs "(see GRAPH-001)" suffix
### m2. LOOP-001 preamble — Add GRAPH-001 to Related Standards list
### m3. CANON-001 §6.1 — Tool registry missing graph/loop cross-references
### m4. CANON-002 §7.1 — Investigation lifecycle missing writeback timing note
### m5. QMS-001 §1.2 — "Knowledge graph" mentioned but not referenced to GRAPH-001
### m6. ORG-001 §5.3 — QMS records should note graph integration
### m7. ARCH-001 §10 — ToolEventBus integration with graph events unclear
### m8. QUAL-001 §8 — Missing graph integration module section (add §8.6)

---

## Standards Requiring No Changes

The following 30 standards are infrastructure, security, billing, frontend, or operational — not domain-relevant to GRAPH-001:

API-001, AUD-001, BILL-001, CACHE-001, CAL-001, CHG-001, CMP-001, DOC-001, DSW-002 (superseded), ERR-001, FE-001, FILE-001, INC-001, JS-001, LLM-001, LOG-001, NTF-001, OPS-001, PRIV-001, RDM-001, SCH-001, SEC-001, SLA-001, STY-001, SYS-001, TST-001, VIS-001, XRF-001, QMS-002

---

## The Pattern

Every CRITICAL finding follows one pattern: **standards written before GRAPH-001 treat their tools as independent systems that "integrate" via evidence hooks.**

GRAPH-001 inverts this: the graph IS the system. Tools are lenses, writers, and validators — not owners of their own data.

The old model:
```
FMEA (owns risk data) → evidence hook → Graph (optional integration)
SPC (owns charts) → alert → Investigation (isolated graph)
RCA (owns causal chains) → A3 report
```

The new model:
```
Graph (owns all process knowledge)
  ├── FMEA view → reads failure_mode nodes + upstream edges
  ├── SPC → monitors nodes, flags edge staleness
  ├── Investigation → scopes subgraph, writes back evidence
  ├── RCA → builds causal edges in investigation subgraph
  ├── QMS → audits graph for compliance
  └── DOE → calibrates edges with effect sizes
```

Every CRITICAL fix is the same conceptual change applied to a different standard.

---

## Recommended Update Sequence

**Phase 1 — Authority (do first, unblocks everything):**
1. LOOP-001 §9 → condensed reference to GRAPH-001
2. DAT-001 §9.5, §6.1 → deprecation notices
3. MAP-001 §4.2 → register GRAPH-001

**Phase 2 — Core domain (the conceptual realignment):**
4. CANON-001 §2.3.2, §2.4 → graph is persistent, QMS is lens
5. CANON-002 intro, §5.3.4 → kill dual posteriors, FMEA = graph view
6. QMS-001 §1.1, §1.2, §4.1, §4.2 → QMS doesn't own FMEA/RCA
7. LOOP-001 §8.7, §16.5 → FMIS as graph lens, remove S/O/D storage

**Phase 3 — Integration points (connect the tools):**
8. LOOP-001 §3.1, §10.2, §16.8 → graph signal sources, readiness metrics
9. STAT-001 §8.2, DSW-001 §6.1 → evidence flows to graph edges
10. RISK-001, SAF-001 → risk entries and hazards seed graph
11. NB-001 → trial evidence flows to org-wide graph
12. TRN-001 → training gaps from graph gaps

**Phase 4 — Structural (architecture docs):**
13. ARCH-001, MAP-001, ORG-001, QUAL-001 → module placement, registry, references
14. DAT-001 §10 → migration phase for graph models

# Object 271 — Product Identity

**Date:** 2026-03-28

## What SVEND Is

SVEND is a **process knowledge system**.

The product is not the 200+ statistical tests. It's not the FMEA tool. It's not the SPC engine. It's not the QMS compliance framework. Those are instruments. The product is the accumulated knowledge they produce.

Every time a user runs a DOE, that calibrates an edge in their process model. Every investigation discovers a new causal relationship. Every SPC shift flags stale knowledge. Every process confirmation validates or challenges what the model believes. The tools are how you build and maintain the model. The model is the product.

## Three Concerns

```
Graph  — what the organization knows about its process
         (persistent, calibrated, Bayesian, grows over time)

Loop   — how the organization learns
         (Signal → Investigate → Standardize → Verify)

QMS    — what the organization must demonstrate
         (compliance lens — ISO 9001, IATF 16949, AS9100D)
```

These are peers, not layers. The Graph doesn't sit inside the QMS. The QMS doesn't own the Loop. Each has its own standard:

- **GRAPH-001** — the knowledge graph and process model
- **LOOP-001** — the closed-loop operating model
- **QMS-001** — the quality management compliance framework

## What Each Tool Actually Is

| Tool | Role | Graph relationship |
|------|------|-------------------|
| **FMEA/FMIS** | Structural assertion | Seeds graph nodes + edges (uncalibrated) |
| **DOE** | Calibration instrument | Produces EdgeEvidence with effect sizes + CIs |
| **SPC** | Health monitor | Updates node distributions, flags edge staleness |
| **Investigation** | Learning mechanism | Scopes subgraph, discovers new structure, writes back evidence |
| **RCA** | Causal discovery | Builds causal edge chains in investigation subgraph |
| **Process Confirmation** | Validation | Confirms or challenges edge posteriors |
| **Forced Failure Test** | Detection calibration | Evidence on detection edges |
| **Gage R&R** | Measurement capability | Evidence on measurement edges, trustworthiness of other edges |
| **A3 / CAPA** | Compliance documentation | Reports assembled from graph evidence |
| **Auditor Portal** | External view | Read-only lens on graph filtered by ISO clause |
| **Hoshin Kanri** | Strategic alignment | Priorities informed by graph gap landscape |
| **Training** | Competence building | Gaps informed by graph areas where knowledge is weak |
| **Process Explorer** | Interactive navigation | Slider-based value propagation through calibrated graph |

## Competitive Identity

**Old framing:** "Minitab + ETQ at 1/10th the price"
**New framing:** "The only platform where every analysis you run makes your process model smarter"

Minitab gives you statistical tools — but results die in a report. ETQ gives you compliance forms — but they don't connect to your process knowledge. SVEND's tools write to a shared model that accumulates knowledge over the lifetime of the org.

The moat is not features. It's the integration depth. A competitor can copy any individual tool. They cannot retroactively connect 200+ analysis types to a unified Bayesian process model without rebuilding from scratch.

## What This Changes

### Navigation
The graph becomes home. Users navigate their process, not a tool menu. Click a node to see its SPC chart. Click an edge to see its evidence stack. Select a subgraph to start an investigation. Drag sliders to explore what-if scenarios.

### Onboarding
Day 1: create your first FMEA. The system proposes a graph skeleton. "You've described 12 failure modes with 23 causes. Here's your process model." The user's first experience is seeing their process, not filling out a form.

### Value Proposition
The longer you use SVEND, the more valuable it becomes — because the graph accumulates knowledge. Switching costs are organic, not artificial. You're not locked in by data format. You're locked in because your process model has two years of calibrated evidence that doesn't exist anywhere else.

### Pricing
The graph justifies enterprise pricing. At $299/mo you get tools. The graph is what makes the tools compound. An org with a mature, calibrated process model is getting exponentially more value than one that just signed up.

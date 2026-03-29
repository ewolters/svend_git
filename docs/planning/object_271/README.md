# Object 271

Milestone: Unified Knowledge Graph + Process Model integration.

Named after the kind of thing you find in a restricted Soviet forest — a concrete structure of unknown purpose that turns out to be load-bearing for the entire region's power grid.

## Context

On 2026-03-28, the design of GRAPH-001 revealed that SVEND's product identity is a **process knowledge system**, not a tool suite. The graph is the product. Three distinct concerns orbit it:

```
Graph  — what the organization knows about its process
Loop   — how the organization learns (Signal → Investigate → Standardize → Verify)
QMS    — what the organization must demonstrate to regulators
```

This milestone aligns the entire system around that model.

## Work Streams

### Standards Audit (this session)
Walk every standard in docs/standards/ against GRAPH-001 and the three-thing model. Surface contradictions, update specs.

### Codebase Audit (parallel session)
Walk the codebase against the new architecture. Identify what changes, what stays, what's new.

## Files

- `standards_audit.md` — contradiction-by-contradiction findings from standards review
- `codebase_audit.md` — (parallel session) findings from codebase review
- `build_sequence.md` — ordered implementation plan (after audits converge)
- `identity.md` — updated product identity and positioning

## Reference

- `docs/standards/GRAPH-001.md` — the spec that started this
- `docs/standards/LOOP-001.md` — the operating model
- `docs/planning/NEXT_GEN_QMS_MASTER_PLAN.md` — QMS NG (to be updated)
- `docs/planning/QMS_ROADMAP_2026_03_18.md` — current roadmap (to be revised)

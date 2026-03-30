# Forge Ecosystem Architecture — S2 Response

**Date:** 2026-03-29
**Context:** S3 proposed 10-package consolidation + monorepo + governance. S2 reviewing.

## S2 Position: Agree on 90%, push back on 10%

### Agree: 10 packages, not 14
The consolidation merges make sense:
- forgecal absorbs drift + anytime-valid
- forgesiop absorbs quality economics
- forgestats absorbs forecast
- forgepbs + forgerel → forgeprocess

### Agree: Monorepo
One person, 10 packages, cross-package calibration. Monorepo is the only sane option.

### Agree: ForgeCal is the keystone
Finish it first. It's the safety net for everything else.

### Agree: Sequence
1. ForgeCal v0.1.0 ✓ (just shipped)
2. Add calibration.py to existing packages
3. Monorepo migration
4. ForgeSPC integration into SVEND
5. Extract forgestats (biggest chunk)
6-8. Everything else

### Push back: forgeviz as "debatable"

S1 said forgeviz "might be too thin." Disagree. 3,158 lines of Plotly JSON spec generation is not thin — it's the entire visualization layer. Every chart in SVEND goes through it. If it stays in the monolith, every forge package that wants to return charts has to depend on SVEND's viz code. Extracting it means forge packages can return chart specs that any frontend renders.

Keep forgeviz as a separate package. It's the bridge between computation and display.

### Push back: forgesia missing from the list

S1's consolidation table doesn't include Synara (forgesia). The belief engine is 3,040 lines of pure computation. It's foundational — the graph service depends on it, investigations depend on it, the FMIS Bayesian posteriors depend on it. If we're extracting forge packages, Synara is the most important one after SPC and stats.

Recommend: forgesia stays as its own package. It's too fundamental to merge into forgebay.

### Revised package list (S2 recommendation)

1. **forgespc** — SPC engine (exists)
2. **forgedoe** — DOE + Bayesian experimental design (exists, absorb bayes_doe)
3. **forgedoc** — Document rendering (exists)
4. **forgesiop** — Supply chain planning + quality economics (exists, absorb forgeeco)
5. **forgecal** — Master calibration (exists, absorb drift/anytime)
6. **forgestats** — Statistical analysis suite (to extract, absorb forecast)
7. **forgeml** — ML pipeline (to extract)
8. **forgebay** — Bayesian analysis (to extract)
9. **forgesia** — Synara belief engine (to extract)
10. **forgecausal** — Causal discovery (to extract)
11. **forgeprocess** — Process behavior + reliability (to extract, merge pbs+rel)
12. **forgeviz** — Visualization specs (to extract)

That's 12, not 10. But forgesia and forgeviz earn their independence.

## Immediate Next Steps

1. ForgeCal is shipped and pushed to github.com/ewolters/forgecal
2. S1/S3 to add `get_calibration_adapter()` to forgespc
3. Monorepo structure decision needed from Eric
4. Then wire forgespc into SVEND (replace agents_api/spc.py)

## What Stays in SVEND Permanently

- Django models, auth, tenancy, compliance
- LLM integration (Claude/Qwen, narrative generation)
- Investigation graph + artifact persistence
- Synara LLM interface (llm_interface.py) — stays in SVEND, calls forgesia
- All views/dispatch (thin wrappers that import from forge)
- Forge app (synthetic data — Django app with jobs, not pure computation)
- Graph service (graph/service.py — Django data layer)
- Loop models + views (Signal, Commitment, Claims, CoA — all Django)

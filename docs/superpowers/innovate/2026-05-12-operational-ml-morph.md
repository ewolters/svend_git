# Morphological Analysis: Operational ML for Manufacturing

**Date:** 2026-05-12
**Technique:** Morphological Analysis (Zwicky Box)

## Morphological Box

| Dimension | Option 1 | Option 2 | Option 3 | Option 4 | Option 5 |
|-----------|----------|----------|----------|----------|----------|
| Data Generation | Exhaustive grid enumeration | Boundary adversarial sampling | Importance-weighted from prod logs | Curriculum staged (easy→hard) | Live Oracle co-generation |
| Architecture | Narrow transformer INT8 | KAN (learnable activations) | Mamba SSM (sequential) | Graph NN per process model | Massive MLP + lottery pruning |
| Verification | Spot-check 5% | Confidence-gated fallback | Pre-compiled certification | Continuous shadow mode | Cryptographic attestation |
| Domain Scope | Single-engine specialist | Multi-engine generalist | Compositional pipeline | Simulation-only | Cross-domain transfer |
| Deployment | All resident VRAM | Hot-swap LRU eviction | CPU-only quantized | Edge ONNX per workstation | Batched async queue |
| Lifecycle | Fixed immutable | Periodic retrain | Continuous online learning | Champion-challenger A/B | Decay-triggered retirement |

**Total design space:** 15,625 combinations

## Top 3 Most Promising

### #1: Curriculum + MLP Pruning + Pre-compiled Cert + Single Specialist + Hot-swap LRU + Champion-Challenger A/B
*Combination 5*

The most buildable. Curriculum+pruning interaction is a genuine training efficiency insight: prune after stage 1, continue training sparse network on harder examples. Single specialist keeps scope tractable. Champion-challenger gives safe upgrade path. Every piece reinforces the others. **Could start building tomorrow on existing Stage 1 infrastructure.**

### #2: Exhaustive Grid + Narrow Transformer INT8 + Confidence-gated Fallback + Multi-engine Generalist + Batched Async + Decay-triggered Retirement
*Combination 3*

The **self-diagnosing lifecycle**: confidence fallback rate IS the decay signal. As model drifts, confidence drops, more queries hit Oracle, fallback rate becomes the retirement trigger. Eliminates a monitoring subsystem by making verification and lifecycle the same mechanism. Elegant closed loop.

### #3: Exhaustive Grid + Graph NN + Compositional Pipeline + CPU-only Quantized + Decay-triggered Retirement
*Combination 7 (verification needs upgrade to continuous shadow)*

Most architecturally provocative. GNN over compositional pipelines learns analysis **topology**, not just individual computations. When SPC feeds DOE feeds simulation, the GNN learns cross-engine interactions that sequential execution misses. **Nobody else models the graph of analyses.** Long-term differentiator but highest research risk.

## Unexpected Dimension Interactions

### Verification × Lifecycle creates closed loops
Confidence-gated fallback + decay-triggered retirement means the verification system generates the lifecycle signal for free. Designing them together yields systems simpler than designing them apart.

### Architecture × Deployment has hardware resonance
MLP pruning → sparse networks → ONNX handles well. Mamba → linear recurrence → CPU-friendly. Graph NNs → compositional pipelines structurally. The "right" architecture depends on deployment target, not just accuracy.

### Data Generation × Lifecycle reveals philosophical stance
- Exhaustive grid + fixed immutable = "right first time" (mature manufacturing)
- Importance-weighted + continuous online = "evolve with usage" (new product introduction)
- Boundary adversarial + periodic retrain = "stress test then update"

Each pairing implies a different operational philosophy. The right one depends on process maturity.

### Domain Scope × Verification scales nonlinearly
Single specialist + pre-compiled cert = O(n). Compositional pipeline + any verification = O(n^k). Cross-domain transfer makes verification nearly intractable. Ambitious scope constrains verification choices far more than expected.

## All 10 Random Combinations Evaluated

### Combo 1: Grid + Transformer INT8 + Pre-compiled cert + Generalist + Hot-swap + Periodic retrain
Feasible with modification. Hot-swap LRU + generalist is quietly smart — "app switching" matches how engineers work (one problem type at a time). Pre-compiled cert conflicts with periodic retrain (re-cert every cycle).

### Combo 2: Grid + MLP pruning + Spot-check 5% + Cross-domain transfer + Edge ONNX + Fixed immutable
Not feasible. Cross-domain transfer + 5% spot-check is dangerous. MLP pruning + Edge ONNX is a natural hardware fit — sparse matmuls on ONNX runtimes. A 100M param MLP pruned to 5M effective could run on a plant-floor laptop.

### Combo 3: Grid + Transformer INT8 + Confidence fallback + Generalist + Batched async + Decay retirement
Feasible with modification (partial-batch completion). **Self-diagnosing lifecycle — top 3 pick.**

### Combo 4: Grid + MLP pruning + Confidence fallback + Cross-domain + Edge ONNX + Periodic retrain
Feasible with modification. Edge deployment + periodic retrain = firmware-style model updates. Needs pipeline tooling.

### Combo 5: Curriculum + MLP pruning + Pre-compiled cert + Specialist + Hot-swap + Champion-challenger
**Feasible as-is. Top 3 pick.** Curriculum + pruning training interaction is a research insight.

### Combo 6: Importance-weighted + Mamba + Confidence fallback + Generalist + CPU quantized + Fixed immutable
Feasible with modification. Mamba wrong for tabular data (designed for sequences) — unless scoped to SPC time series / reliability degradation only. Importance-weighted + fixed immutable assumes process stability — domain-appropriate in mature manufacturing.

### Combo 7: Grid + Graph NN + Spot-check 5% + Compositional pipeline + CPU quantized + Decay retirement
Feasible with verification upgrade. **GNN over compositional pipelines is genuinely novel — top 3 pick.** Learns analysis topology, not just individual computations.

### Combo 8: Importance-weighted + Transformer INT8 + Continuous shadow + Cross-domain + All VRAM + Champion-challenger
Feasible with modification. Champion-challenger + continuous shadow creates a **tournament system** — manufacturing's own process validation methodology applied to ML lifecycle.

### Combo 9: Grid + MLP pruning + Pre-compiled cert + Cross-domain + CPU quantized + Decay retirement
Feasible with modification. CPU quantized + decay retirement = "lightbulb model" — cheap, ubiquitous, replaceable.

### Combo 10: Boundary adversarial + Transformer INT8 + Spot-check 5% + Generalist + CPU quantized + Fixed immutable
Not feasible (train on boundaries, verify weakly = contradictory). But **boundary adversarial + fixed immutable = "right first time"** — how safety-critical firmware works. A lean manufacturing principle applied to ML.

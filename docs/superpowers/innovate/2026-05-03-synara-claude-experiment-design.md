# Experiment Design: Claude vs. Claude + Synara

**Date:** 2026-05-03
**Goal:** Measure whether Bayesian governance (Synara) measurably improves Claude's performance on tasks where LLMs are known to be weakest — specifically evidence accumulation, calibration, and frequency sensitivity.
**Architecture thesis:** PCL (data) → Synara (rules + confidence) → Claude (judgment). This experiment tests whether that stack produces better outcomes than Claude alone.

---

## Why This Matters

LLMs perform one-shot approximate inference mimicking human System 1 thinking, including human biases. Bayesian systems perform exact sequential inference with correct evidence accumulation. The hybrid thesis: Claude contributes domain knowledge, NL understanding, interpretation, and judgment. Synara contributes correct evidence accumulation, calibrated uncertainty, proper base rate handling, and convergence guarantees.

If the experiment shows measurable improvement, it validates the entire SVEND architecture: Claude as facilitator, Synara as governance, the distinction between "rules are free, judgment is Claude."

---

## What Synara Can Actually Compute Today

**Working and battle-tested:**
- `core/bayesian.py` — BayesianUpdater with confidence-adjusted likelihood ratios
- `core/models/hypothesis.py` — Hypothesis tracking with prior/posterior, probability history, auto-status
- `core/models/hypothesis.py` — EvidenceLink with LR, confidence, direction, strength (Jeffreys' scale)
- forgesia — pure-Python causal graph + belief propagation (Beta-Binomial conjugate)
- forgepbs — BOCPD changepoint detection (fixed 2026-04-24)
- `Hypothesis.apply_evidence()` and `recalculate_probability()` — multi-evidence Bayesian updating

**The key formula (in production):**
```
adjusted_LR = 1 + (LR - 1) × confidence
posterior_odds = prior_odds × adjusted_LR
```

Low-confidence evidence moves LR toward neutral. Probability history is tracked. Status auto-transitions at thresholds.

---

## Known LLM Weaknesses (Published Research)

1. **Base rate neglect** — LLMs replicate Kahneman & Tversky heuristics rather than performing correct Bayesian updates (Binz & Schulz 2023)
2. **Evidence accumulation failure** — no mechanism for running posterior. Recent observations weighted heavily, cumulative signal lost
3. **Frequency insensitivity** — "observed 3 times" and "observed 30 times" produce similar confidence (Yildirim & Paul 2024)
4. **Calibration drift** — stated confidence doesn't track sample size. Bayesian posterior concentrates as N grows; LLM confidence does not (Jiang et al. 2023)
5. **Conditional flattening** — P(A|B,C) treated similarly to P(A|B) when C is complex (Saparov & He 2023)
6. **Contradiction handling** — either over-anchor on initial evidence or over-react to contradictions

---

## Experiment Design

### Setup: Two Claude Code Sessions, Same Problems

**Session A (Control):** Claude alone. No Synara tools. Just conversation + reasoning.

**Session B (Treatment):** Claude + Synara plugin. Plugin provides:
- `synara.update(hypothesis_id, evidence, likelihood_ratio, confidence)` — Bayesian update
- `synara.posterior(hypothesis_id)` — current posterior with confidence interval
- `synara.history(hypothesis_id)` — full probability trajectory
- `synara.bocpd(data)` — changepoint detection via forgepbs
- `synara.beta_posterior(successes, trials)` — Beta-Binomial posterior for rate estimation

Claude in Session B is instructed: "You have access to a Bayesian reasoning system. Use it for evidence accumulation and probability estimation. Use your own judgment for interpretation and action recommendations."

### Primary Task: Sequential Changepoint Detection

**Why:** Leverages existing forgepbs BOCPD. Ground truth is known. Measurable. Tests the core weakness (evidence accumulation over many observations).

**Protocol:**
1. Generate 50 process streams, each ~100 observations
2. Inject mean shifts at known points: 0.3σ, 0.5σ, 1.0σ, 1.5σ, 2.0σ (10 streams each)
3. Feed observations in batches of 10 to both sessions
4. After each batch, ask: "Has the process shifted? When? Confidence?"
5. Log: detection point, confidence, actual shift point

**Metrics:**
- Detection latency (observations after true shift before detection)
- False alarm rate
- Confidence calibration (reliability diagram: stated confidence vs actual correctness)

**Expected results:**
- Small shifts (0.3-0.5σ): Claude alone misses most or detects very late. Claude + BOCPD detects within 20-30 observations.
- Large shifts (1.5-2.0σ): Both detect, but Claude alone has worse calibration.
- Frequency sensitivity: Claude's confidence at observation 20 vs. 80 (both post-shift) will be similar. BOCPD correctly shows higher confidence at 80.

### Secondary Task: Base Rate Diagnosis (30 min)

50 scenarios with varying base rates (1%, 5%, 20%, 50%) and test accuracies (80%-99%). "Given a positive test, what's the probability of the condition?"

**Metric:** Mean absolute error from true Bayesian answer. Claude alone expected to show base rate neglect (overestimate posterior when base rate is low).

### Secondary Task: Frequency Sensitivity (30 min)

"N parts observed, K defective. What's P(true rate > 5%)?" Vary N from 5 to 500, K/N from 2% to 8%.

**Metric:** At N=5, K=0: true P(rate > 5%) ≈ 23% (Beta(1,6)). At N=500, K=0: essentially 0%. Claude alone will give similar answers. Beta-Binomial posterior correctly distinguishes.

### Tertiary Task: Belief Revision Under Contradiction

10 observations supporting hypothesis A, then 3 strong observations supporting B. Ask for updated belief.

**Metric:** Appropriate weighting. LLMs tend to over-anchor or over-react. Bayesian system correctly balances based on likelihood ratios.

---

## Plugin Design for Session B

The Synara plugin for Claude Code needs:

### Skills
- `synara:update` — apply evidence to a hypothesis
- `synara:query` — get current posterior, history, strength classification
- `synara:bocpd` — run changepoint detection on a data series
- `synara:beta` — compute Beta-Binomial posterior for rate estimation
- `synara:diagnose` — multi-hypothesis posterior update given observed symptom

### MCP Server (alternative)
An MCP server wrapping the Django models:
- `synara_update_hypothesis(id, lr, confidence)` → posterior
- `synara_get_posterior(id)` → {probability, ci, history, strength}
- `synara_bocpd(data, hazard_rate)` → {changepoints, run_length_posterior}
- `synara_beta_posterior(successes, trials, prior_alpha, prior_beta)` → {mean, ci, p_above_threshold}

### What Claude Sees
```
You have access to a Bayesian reasoning system called Synara. 

For probability estimation: use synara tools instead of estimating yourself.
For evidence accumulation: call synara_update_hypothesis after each observation.
For changepoint detection: call synara_bocpd with the data series.
For rate estimation: call synara_beta_posterior with observed counts.

Your role: interpret Synara's outputs, explain them, recommend actions. 
Synara's role: correct probability computation and evidence tracking.
You provide judgment. Synara provides math.
```

---

## Implementation Plan (Tomorrow)

### Morning (2-3 hours): Build the plugin
1. Create MCP server wrapping core/bayesian.py + forgepbs BOCPD
2. Test with simple cases (single update, multi-update, changepoint on known data)
3. Wire to Claude Code as plugin or MCP

### Midday (2-3 hours): Generate data and run experiment
1. Script 50 process streams with known shifts
2. Run Session A (Claude alone) on all 50
3. Run Session B (Claude + Synara) on all 50
4. Run secondary tasks (base rate, frequency sensitivity)

### Afternoon (1-2 hours): Score and analyze
1. Compare detection latency, false alarm rate, calibration
2. Build reliability diagram (stated confidence vs actual)
3. Document findings

---

## What a Positive Result Means

If Claude + Synara measurably outperforms Claude alone on these tasks, it validates:

1. **The SVEND architecture** — Claude as facilitator + Synara as governance is not just organizational, it's computationally superior
2. **The marketplace split** — "rules are free, judgment is Claude" has a measurable basis
3. **The confidence layer** — every suggestion carrying a Bayesian confidence score (from Synara, not Claude's self-assessment) is architecturally justified
4. **The product positioning** — SVEND is not "AI-powered quality." It's a quality system with correct math and an AI that interprets it. The math is what competitors can't replicate by bolting a chatbot onto Minitab.

---

## What a Negative Result Means

If Claude alone performs comparably:
- The Bayesian governance layer adds complexity without measurable benefit
- Synara should be simplified to just event routing (Cortex) without the belief system
- The confidence scores on suggestions have no advantage over Claude's native calibration
- The architecture simplifies but loses a differentiator

Either result is valuable. The experiment takes one day.

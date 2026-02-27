"""Learning content: Probabilistic Bayesian SPC (PBS) Mastery.

The flagship module teaching Svend's unique differentiator — Bayesian
process monitoring that replaces classical SPC's fixed limits and binary
rules with probabilistic beliefs.  All 12 PBS engine components are covered.
"""

from ._datasets import SHARED_DATASET  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════
# Section 1: The Paradigm Shift
# PBS Component: NormalGammaPosterior
# ═══════════════════════════════════════════════════════════════════════════

PBS_PARADIGM_SHIFT = {
    "id": "pbs-paradigm-shift",
    "title": "The Paradigm Shift: From Rules to Beliefs",
    "intro": "Classical SPC uses fixed control limits and binary in/out decisions. That approach was designed in the 1920s. PBS replaces it with a probabilistic belief state that learns from every observation. This section shows you why that matters — and why no other tool can do it.",
    "exercise": {
        "title": "Try It: Classical vs PBS Side-by-Side",
        "steps": [
            "Load the manufacturing dataset (200 rows of bore diameter measurements)",
            "Run a full PBS analysis with spec limits 24.85–25.15mm and target 25.0mm",
            "Read the narrative output — note how it describes shift probability, not just 'out of control'",
            "Compare to a classical I-MR chart on the same data — note the binary flags vs graduated probabilities",
            "Observe how PBS gives you a credible interval on Cpk, not just a point estimate"
        ],
        "dsw_type": "pbs:pbs_full",
        "dsw_config": {
            "column": "diameter_mm",
            "USL": 25.15,
            "LSL": 24.85,
            "target": 25.0
        },
    },
    "content": """
## Why Classical SPC Is Broken

Walter Shewhart invented control charts in 1924. A century later, Minitab and JMP still use essentially the same framework. It has five fatal flaws:

### Flaw 1: Fixed Control Limits

Classical limits are calculated once from historical data and never change. But your process isn't static — your *confidence* in those limits should grow as you collect more data.

**PBS solution:** Adaptive control limits from the posterior predictive distribution. Wide when you have 10 observations, tight when you have 1000.

### Flaw 2: Binary Decisions

A point is either "in control" or "out of control." There's no middle ground.

**PBS solution:** A continuous shift probability from 0% to 100%. A point at P(shift) = 73% tells you something very different from P(shift) = 99%.

### Flaw 3: No Memory

Each subgroup is evaluated independently. The chart doesn't remember that the last five points were all trending upward (unless you manually apply run rules).

**PBS solution:** The Normal-Gamma posterior accumulates *all* information. Every observation updates the belief state.

### Flaw 4: No Uncertainty Quantification

Classical Cpk = 1.4. Is that precise? Could it actually be 0.9? With 20 observations, you have no idea.

**PBS solution:** A full posterior distribution on Cpk. "90% credible interval: [1.1, 1.7]" — now you can make an informed decision.

### Flaw 5: No Learning Across Processes

Start a new product line? Start from scratch. Classical SPC has no mechanism to transfer knowledge.

**PBS solution:** Chart Genealogy. Inherit a discounted prior from a parent process. The new line starts smarter, not from zero.

## The Core: Normal-Gamma Posterior

PBS maintains a **belief state** about your process using a Normal-Gamma conjugate prior:

$$\\mu | \\tau \\sim N(\\mu_0, 1/(\\kappa_0 \\cdot \\tau)), \\quad \\tau \\sim \\text{Gamma}(\\alpha_0, \\beta_0)$$

where $\\tau = 1/\\sigma^2$ (precision).

**Four parameters encode everything you know:**

| Parameter | What It Encodes | Initial Value |
|-----------|----------------|---------------|
| $\\mu$ | Best estimate of process mean | Target or spec midpoint |
| $\\kappa$ | Confidence in the mean (pseudo-observations) | Small (e.g., 1.0) |
| $\\alpha$ | Confidence in variance estimate | Small (e.g., 2.0) |
| $\\beta$ | Scale of variance estimate | Matched to calibration data |

### O(1) Updates

When a new observation $x$ arrives:

$$\\kappa_{new} = \\kappa + 1$$
$$\\mu_{new} = \\frac{\\kappa \\cdot \\mu + x}{\\kappa_{new}}$$
$$\\alpha_{new} = \\alpha + \\frac{1}{2}$$
$$\\beta_{new} = \\beta + \\frac{\\kappa (x - \\mu)^2}{2 \\kappa_{new}}$$

This is **O(1)** — constant time regardless of how much data you've seen. No need to recompute from scratch. No MCMC. No LLM. Pure math.

### The Prior Matters (Then It Doesn't)

With few observations, the prior influences results. This is a feature, not a bug — it lets you encode domain knowledge ("bore diameters on this machine typically run ±0.05mm").

As data accumulates, the likelihood dominates and the prior washes out. After ~50 observations, results are effectively prior-independent.

## What This Means in Practice

| Classical SPC | PBS |
|---------------|-----|
| "Point out of control" | "78% probability of a shift, magnitude ~0.03mm" |
| Cpk = 1.4 | Cpk = 1.4 [1.1, 1.7], P(>1.33) = 62% |
| Fixed ±3σ limits | Limits that narrow as confidence grows |
| Binary alarm | Graduated: watch → alert → alarm |
| No forecast | "32% chance of spec exceedance in next 25 obs" |
""",
    "interactive": {
        "type": "pbs_demo",
        "config": {"analysis": "pbs_full"}
    },
    "key_takeaways": [
        "Classical SPC uses fixed limits and binary decisions — PBS uses adaptive limits and continuous probabilities",
        "The Normal-Gamma posterior encodes everything known about the process in four parameters",
        "Each new observation updates the belief state in O(1) time — no recomputation needed",
        "PBS gives credible intervals on capability, not just point estimates",
        "The prior encodes domain knowledge and washes out as data accumulates",
    ],
    "practice_questions": [
        {
            "question": "You have 15 observations from a new process. Classical Cpk is 1.45. Should you sign off on capability?",
            "answer": "No. With only 15 observations, the uncertainty on Cpk is enormous. A Bayesian credible interval might show [0.7, 2.2]. The point estimate of 1.45 is meaningless without quantifying uncertainty. PBS would tell you P(Cpk > 1.33) — that's the number that matters for a sign-off decision.",
            "hint": "What would the credible interval look like with so few observations?"
        },
        {
            "question": "Why does PBS use a conjugate prior (Normal-Gamma) instead of MCMC sampling?",
            "answer": "Conjugate priors allow exact, closed-form posterior updates in O(1) time. This is critical for streaming data — each new measurement updates the belief instantly. MCMC would require re-running a sampler with every new data point, which is far too slow for real-time process monitoring.",
            "hint": "Think about what happens when a new measurement arrives every few seconds on a production line."
        },
    ]
}


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Detecting Change
# PBS Components: BeliefChart (BOCPD), EDetector
# ═══════════════════════════════════════════════════════════════════════════

PBS_CHANGE_DETECTION = {
    "id": "pbs-change-detection",
    "title": "Detecting Change: BOCPD & E-Detector",
    "intro": "Has the process shifted? Classical SPC gives you a binary answer. PBS gives you a probability — and backs it up with two independent detection methods that corroborate each other.",
    "exercise": {
        "title": "Try It: Detect the Hidden Shift",
        "steps": [
            "Run the Belief Chart analysis on the diameter column",
            "Find where shift probability first exceeds 50% (watch level)",
            "Note where it crosses 95% (alarm level) — that's the confirmed shift",
            "Check the alert cascade: nominal → watch → alert → alarm",
            "Now run the E-Detector on the same data — does it find the same shift?",
            "Compare the two methods: BOCPD uses Bayesian updating, E-Detector uses distribution-free CUSUM"
        ],
        "dsw_type": "pbs:pbs_belief",
        "dsw_config": {
            "column": "diameter_mm"
        },
    },
    "content": """
## The Belief Chart: BOCPD

The Belief Chart uses **Bayesian Online Changepoint Detection** (Adams & MacKay 2007) — one of the most elegant algorithms in sequential analysis.

### Core Idea

At every time step, the model maintains a probability distribution over **run lengths** — how long the current regime has been going. A "changepoint" means the run length resets to zero.

$$P(r_t | x_{1:t})$$

where $r_t$ is the run length at time $t$.

### How It Works

1. **Evaluate:** For each possible run length, compute the predictive probability of the new observation
2. **Grow:** Extend each run length by 1 (no change happened)
3. **Reset:** Compute the probability that a changepoint just occurred (run length → 0)
4. **Normalize:** The result is a valid probability distribution

### Shift Probability

The shift probability at each point is:

$$P(\\text{shift}) = 1 - P(r_t = \\text{max run length})$$

This gives you a **continuous measure** from 0 to 1, not a binary flag.

### Alert Cascade

| Shift Probability | Alert Level | Action |
|-------------------|-------------|--------|
| < 50% | **Nominal** | Continue monitoring |
| 50–80% | **Watch** | Increase attention, check other signals |
| 80–95% | **Alert** | Begin investigation preparation |
| ≥ 95% | **Alarm** | Investigate immediately |

This graduated response prevents both over-reaction (investigating every blip) and under-reaction (ignoring early warning signs).

### The Hazard Function

The hazard rate $\\lambda$ controls sensitivity:

$$P(\\text{changepoint}) = 1/\\lambda$$

- **Small $\\lambda$ (e.g., 20):** Sensitive — detects small shifts quickly but more false alarms
- **Large $\\lambda$ (e.g., 500):** Conservative — requires strong evidence but slower to detect

PBS uses **empirical Bayes** — it runs a grid of $\\lambda$ values and selects the one with the highest marginal likelihood. Changepoints detected across multiple $\\lambda$ values are flagged as **robust**.

### Outlier Robustness: Dm-BOCD

Standard BOCPD can be fooled by outliers. When `beta_robustness > 0`, PBS uses **Density Power Divergence scoring** (Altamirano, Briol & Knoblauch, ICML 2023). Outliers are downweighted rather than discarded:

$$w_i = \\exp(-\\beta \\cdot z^2 / 2)$$

At $\\beta = 0.1$ and $z = 3\\sigma$, the weight is ~64%. The outlier still contributes — just less.

## The E-Detector: Distribution-Free Backup

The E-Detector (Shin, Ramdas & Rinaldo 2024) provides **independent corroboration** using a completely different mathematical framework.

### How It Differs from BOCPD

| | BOCPD | E-Detector |
|-|-------|------------|
| **Approach** | Bayesian (Normal-Gamma model) | Frequentist (sub-Gaussian e-values) |
| **Assumptions** | Parametric (assumes normality) | Distribution-free (any sub-Gaussian) |
| **Output** | Shift probability | Evidence accumulation statistic |
| **Strength** | Precise when model correct | Robust to model misspecification |

### Two-Sided Detection

The E-Detector runs two parallel CUSUM processes:
- **Upper detector:** Tests for upward shift ($\\mu > \\mu_0$)
- **Lower detector:** Tests for downward shift ($\\mu < \\mu_0$)

An alarm fires when either crosses the threshold $\\log(1/\\alpha)$.

### Why Two Methods?

When BOCPD and E-Detector **agree**, you have strong corroboration from two independent methods. When they **disagree**:
- BOCPD alarms but E-Detector doesn't → Shift may be within normal model variation
- E-Detector alarms but BOCPD doesn't → Possible non-Gaussian disturbance
- Both silent → Process is genuinely stable

This dual-detection approach is something no classical SPC system offers.
""",
    "interactive": {
        "type": "pbs_demo",
        "config": {"analysis": "pbs_belief"}
    },
    "key_takeaways": [
        "BOCPD gives you a continuous shift probability (0–100%), not a binary in/out flag",
        "The alert cascade (nominal → watch → alert → alarm) enables graduated response",
        "Empirical Bayes over the hazard rate grid selects optimal sensitivity automatically",
        "The E-Detector provides independent, distribution-free corroboration",
        "Dm-BOCD handles outliers by downweighting instead of discarding them",
        "When both detectors agree, confidence in the shift is high",
    ],
    "practice_questions": [
        {
            "question": "The BOCPD shows P(shift) = 72% at observation 45. Should you stop the line?",
            "answer": "Not yet — 72% is 'watch' level, not 'alarm.' Increase monitoring frequency and check the E-Detector for corroboration. If both methods are signaling, prepare to investigate. If only BOCPD is flagging, the shift may be small or within model variation. Wait for P(shift) to cross 80% (alert) or 95% (alarm) before taking production-disrupting action.",
            "hint": "Check the alert cascade — what level does 72% correspond to?"
        },
        {
            "question": "BOCPD detects a shift at observation 30 with robustness 5/5 (all lambda values). What does this tell you?",
            "answer": "A robustness score of 5/5 means the changepoint was detected regardless of the hazard rate setting. This is a very strong signal — the shift is real and not an artifact of a particular sensitivity tuning. You can have high confidence that the process genuinely changed at that point.",
            "hint": "What does it mean when all lambda values in the grid detect the same changepoint?"
        },
    ]
}


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: Evidence Accumulation
# PBS Component: EvidenceAccumulation
# ═══════════════════════════════════════════════════════════════════════════

PBS_EVIDENCE_ACCUMULATION = {
    "id": "pbs-evidence-accumulation",
    "title": "Accumulating Evidence: Anytime-Valid E-Values",
    "intro": "Every time you peek at a p-value during a running experiment, you inflate your false positive rate. E-values solve this completely — you can check as often as you want, and the evidence is always valid. This is the mathematical foundation for continuous process monitoring.",
    "exercise": {
        "title": "Try It: Watch Evidence Accumulate",
        "steps": [
            "Run the Evidence Accumulation analysis on the diameter column",
            "Watch the e-value grow as observations are processed",
            "Note which observation the evidence first becomes 'notable' (5:1)",
            "Find where it becomes 'strong' (20:1) and 'decisive' (100:1)",
            "Compare the evidence timeline to where BOCPD detected the shift",
            "Key insight: e-values grow proportionally to the strength of the signal"
        ],
        "dsw_type": "pbs:pbs_evidence",
        "dsw_config": {
            "column": "diameter_mm"
        },
    },
    "content": """
## The Peeking Problem

In classical statistics, if you compute a p-value at observation 50, then again at 100, then at 150, your actual false positive rate is much higher than 0.05. This is called the **optional stopping problem.**

Manufacturing processes run continuously. You *need* to check regularly. Classical hypothesis testing can't handle this.

## E-Values: Always Valid

E-values (Grünwald 2024) are **evidence ratios** that remain valid under any stopping rule:

$$e_t = \\frac{P(x_t | H_1)}{P(x_t | H_0)}$$

The key property: accumulated e-values $E = \\prod e_t$ satisfy **Ville's inequality:**

$$P(E \\geq 1/\\alpha \\text{ at any time } t) \\leq \\alpha$$

This means you can peek at the evidence at **every single observation** and your error guarantee still holds.

### How PBS Computes E-Values

PBS uses a **Gaussian mixture alternative** — a robust choice that doesn't require specifying a particular alternative hypothesis:

$$e_t = \\sqrt{\\frac{\\sigma^2}{\\sigma^2 + \\sigma^2_{mix}}} \\cdot \\exp\\left(\\frac{\\sigma^2_{mix} (x_t - \\mu_0)^2}{2\\sigma^2(\\sigma^2 + \\sigma^2_{mix})}\\right)$$

Observations near $\\mu_0$ give $e_t \\approx 1$ (no evidence). Observations far from $\\mu_0$ give $e_t > 1$ (evidence of change).

### Evidence Levels

E-values accumulate in log-space (multiplication becomes addition):

| Accumulated E-Value | Evidence Level | Interpretation |
|---------------------|----------------|----------------|
| < 5 | **None** | Insufficient evidence to conclude anything |
| 5–20 | **Notable** | Worth paying attention to |
| 20–100 | **Strong** | Clear evidence of change |
| ≥ 100 | **Decisive** | Overwhelming evidence (capped at 10,000:1 display) |

### Why This Beats P-Values for SPC

| | P-Values | E-Values |
|-|----------|----------|
| **Peeking** | Inflates error rate | Always valid |
| **Interpretation** | "How surprising if H₀ true" | "How much evidence against H₀" |
| **Accumulation** | Cannot combine sequentially | Multiply naturally |
| **Stopping** | Must pre-specify sample size | Stop anytime |
| **Direction** | Against H₀ only | Supports H₁ over H₀ |

### Evidence Corroborates BOCPD

In the full PBS analysis, the evidence accumulation chart runs alongside the Belief Chart. When BOCPD detects a shift at observation $t$ with probability $P$, and the evidence chart shows e-value $E$ at the same point, you get **independent corroboration:**

- BOCPD says "73% chance of a shift"
- Evidence says "12:1 odds something changed"

Both methods reach the same conclusion through different mathematics. That's powerful.
""",
    "interactive": {
        "type": "pbs_demo",
        "config": {"analysis": "pbs_evidence"}
    },
    "key_takeaways": [
        "P-values are invalidated by peeking — e-values remain valid under any stopping rule",
        "E-values accumulate multiplicatively: each observation adds to the evidence",
        "Evidence levels (none/notable/strong/decisive) provide intuitive interpretation",
        "You can check e-values at every observation without inflating error rates",
        "Evidence accumulation corroborates BOCPD shift detection through independent mathematics",
    ],
    "practice_questions": [
        {
            "question": "Your evidence chart shows e-value = 8 after 50 observations. Is this enough to conclude a shift occurred?",
            "answer": "An e-value of 8 is 'notable' — meaningful evidence but not conclusive. It means the data is 8 times more likely under a shifted process than a stable one. For a process-stopping decision, you'd typically want 'strong' (>20) or 'decisive' (>100). But 8:1 combined with other signals (like BOCPD at 'watch' level) might justify investigation without stopping.",
            "hint": "Check the evidence level thresholds — where does 8 fall?"
        },
        {
            "question": "Why does PBS cap the displayed e-value at 10,000:1?",
            "answer": "After 10,000:1, additional evidence doesn't change the practical decision — you're already certain. Displaying larger values can create a false sense of precision and may cause numerical display issues. The log-space accumulation continues internally, but the display is capped for practical interpretation.",
            "hint": "Think about what additional practical value 100,000:1 evidence gives over 10,000:1."
        },
    ]
}


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Predictive & Adaptive
# PBS Components: PredictiveChart, AdaptiveControlLimits
# ═══════════════════════════════════════════════════════════════════════════

PBS_PREDICTIVE_ADAPTIVE = {
    "id": "pbs-predictive-adaptive",
    "title": "Seeing the Future: Prediction Fans & Adaptive Limits",
    "intro": "Classical control charts only look backward — they tell you what happened. PBS looks forward with calibrated uncertainty. How likely is spec exceedance in the next 25 observations? Are your control limits too wide or too tight for how much data you have?",
    "exercise": {
        "title": "Try It: Predict Where the Process Is Going",
        "steps": [
            "Run the Predictive Chart analysis with spec limits 24.85–25.15mm",
            "Examine the prediction fan — note how uncertainty widens at longer horizons",
            "Check the slope posterior: is the process trending? What's P(slope > 0)?",
            "Note P(spec exceedance at 10 obs) vs P(spec exceedance at 25 obs)",
            "Now run Adaptive Control Limits on the same data",
            "Compare the adaptive limits early (wide, few observations) vs late (narrow, many observations)"
        ],
        "dsw_type": "pbs:pbs_predictive",
        "dsw_config": {
            "column": "diameter_mm",
            "USL": 25.15,
            "LSL": 24.85
        },
    },
    "content": """
## The Predictive Chart

Classical SPC is purely **retrospective** — it tells you if something already happened. The Predictive Chart is **prospective** — it tells you where the process is heading.

### Bayesian Linear Trend

PBS fits a Bayesian linear model on a rolling window of recent observations:

$$y_t = \\beta_0 + \\beta_1 \\cdot t + \\epsilon$$

The posterior on $(\\beta_0, \\beta_1)$ gives a full probability distribution on the slope.

### The Prediction Fan

For each future horizon $h$, PBS computes:
- **Predicted mean:** $\\hat{y}_{t+h} = \\beta_0 + \\beta_1(t+h)$
- **90% credible interval:** Where will 90% of future observations fall?
- **50% credible interval:** The most likely range

The fan **widens with horizon** — the further ahead you look, the more uncertain the prediction. This is honest uncertainty quantification.

### Spec Exceedance Probability

The key output: **P(any observation violates spec in the next N steps)**

$$P(\\text{exceed}) = 1 - \\prod_{h=1}^{N} P(LSL < y_{t+h} < USL)$$

| Metric | Meaning |
|--------|---------|
| P(exceed at 10 obs) | Short-term risk |
| P(exceed at 25 obs) | Medium-term risk |
| Est. obs to exceedance | How long until 50% chance of spec violation |

### Slope Interpretation

| Slope Posterior | Meaning |
|----------------|---------|
| P(slope > 0) > 90% | Process is trending upward with high confidence |
| P(slope > 0) ≈ 50% | No detectable trend |
| P(slope < 0) > 90% | Process is trending downward with high confidence |

## Adaptive Control Limits

Classical control limits are computed once and fixed:

$$UCL = \\bar{x} + 3\\sigma, \\quad LCL = \\bar{x} - 3\\sigma$$

PBS computes limits from the **posterior predictive distribution**, which changes with every observation:

$$x_{new} \\sim t_{2\\alpha}(\\mu_n, \\beta_n(\\kappa_n+1)/(\\alpha_n \\kappa_n))$$

### How They Narrow

| Observations | $\\kappa$ | Limit Width | Interpretation |
|-------------|-----------|-------------|----------------|
| 10 | ~11 | Wide | Still learning — limits are conservative |
| 50 | ~51 | Medium | Reasonable confidence in process parameters |
| 200 | ~201 | Narrow | Very confident — limits approach classical ±3σ |

The limits are **honest**: when you have little data, they're wide (admitting uncertainty). As evidence accumulates, they tighten.

### Consistency Check

When $n \\geq 50$ and Bayesian limits diverge from classical by >10%, PBS flags a warning. This usually means the prior is misspecified — the prior expected different process behavior than the data shows.

### Why This Matters

Classical SPC with 15 observations has the same limit width as with 1500. That's wrong — your confidence at $n=15$ is nowhere near your confidence at $n=1500$. Adaptive limits make this explicit.
""",
    "interactive": {
        "type": "pbs_demo",
        "config": {"analysis": "pbs_predictive"}
    },
    "key_takeaways": [
        "The Predictive Chart looks forward — classical charts only look backward",
        "Prediction fans show calibrated uncertainty that widens with horizon",
        "P(spec exceedance) is the key output: how likely is a future violation?",
        "Adaptive control limits narrow as data accumulates — wide at n=10, tight at n=200",
        "Classical fixed limits falsely imply the same confidence at n=15 as n=1500",
        "The slope posterior tells you if the process is trending with quantified certainty",
    ],
    "practice_questions": [
        {
            "question": "The Predictive Chart shows P(spec exceedance at 25 obs) = 45%. P(slope > 0) = 88%. What should you do?",
            "answer": "The process is trending upward (88% confidence) and has a 45% chance of violating spec within 25 observations. This is a clear warning — investigate the trend source now. Don't wait for an actual spec violation. The predictive chart is giving you lead time to act proactively.",
            "hint": "This is the power of forward-looking analysis — you have time to prevent the problem."
        },
        {
            "question": "Why are adaptive control limits wider at the start of a production run?",
            "answer": "With few observations, the posterior is dominated by prior uncertainty. The kappa parameter is low, so the posterior predictive t-distribution has heavy tails (wide spread). As observations accumulate, kappa grows, the degrees of freedom increase, and the limits narrow toward the classical ±3σ. This correctly reflects that early measurements carry more uncertainty about the true process parameters.",
            "hint": "Think about what kappa represents — effective sample size for the mean."
        },
    ]
}


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Bayesian Capability
# PBS Components: BayesianCpk, CpkTrajectory
# ═══════════════════════════════════════════════════════════════════════════

PBS_BAYESIAN_CAPABILITY = {
    "id": "pbs-bayesian-capability",
    "title": "Capability with Honesty: Bayesian Cpk",
    "intro": "When someone tells you Cpk = 1.4, the right question is: 'How sure are you?' Classical capability analysis can't answer that. Bayesian Cpk gives you a full probability distribution — including the chance that your supposedly capable process actually isn't.",
    "exercise": {
        "title": "Try It: Cpk as a Distribution, Not a Number",
        "steps": [
            "Run Bayesian Cpk with spec limits 24.85–25.15mm",
            "Compare the classical Cpk point estimate to the Bayesian median",
            "Check the 90% credible interval — how wide is the uncertainty?",
            "Note P(Cpk > 1.33): what's the probability of actually meeting the standard?",
            "Now run the Cpk Trajectory to see how capability has evolved over time",
            "Check P(Cpk declining): is capability getting worse?"
        ],
        "dsw_type": "pbs:pbs_cpk",
        "dsw_config": {
            "column": "diameter_mm",
            "USL": 25.15,
            "LSL": 24.85
        },
    },
    "content": """
## The Problem with Point-Estimate Cpk

Classical Cpk:
$$C_{pk} = \\min\\left(\\frac{USL - \\bar{x}}{3s}, \\frac{\\bar{x} - LSL}{3s}\\right)$$

This gives you a single number. But that number has **massive uncertainty** at typical sample sizes:

| n | Classical Cpk | True 90% CI (approximate) |
|---|---------------|---------------------------|
| 20 | 1.40 | [0.85, 1.95] |
| 50 | 1.40 | [1.10, 1.70] |
| 200 | 1.40 | [1.28, 1.52] |

At $n=20$, a "Cpk of 1.4" could easily be 0.85 (not capable) or 1.95 (excellent). The point estimate hides this.

## Bayesian Cpk: The Full Picture

PBS computes a posterior distribution on Cpk using **ancestral sampling** from the Normal-Gamma posterior:

1. **Sample $\\tau$** from Gamma($\\alpha$, $1/\\beta$) — process precision
2. **Sample $\\mu | \\tau$** from Normal($\\mu_n$, $1/(\\kappa_n \\tau)$) — process mean given precision
3. **Compute $\\sigma = 1/\\sqrt{\\tau}$** — process standard deviation
4. **Compute Cpk** for each $(\\mu, \\sigma)$ pair

From 10,000 samples, PBS gives you:

### Key Outputs

| Output | What It Tells You |
|--------|-------------------|
| **Cpk median** | Best estimate of capability |
| **90% credible interval** | Range where Cpk almost certainly falls |
| **P(Cpk > 1.0)** | Probability of basic capability |
| **P(Cpk > 1.33)** | Probability of meeting the common standard |
| **P(Cpk > 1.67)** | Probability of excellent capability |

### Making Decisions with Uncertainty

Instead of "Cpk = 1.4, we're capable," you get:

> Bayesian Cpk: 1.38 [1.05, 1.71]. P(Cpk > 1.33) = 62%.

Now you can make a **risk-informed decision:** there's a 62% chance you meet the 1.33 standard, but a 38% chance you don't. Is that acceptable for this product? For an automotive safety part, probably not. For a cosmetic dimension, maybe.

## Cpk Trajectory: Capability Over Time

A single Cpk snapshot tells you where you are. The **Cpk Trajectory** tells you where you're going.

PBS computes Bayesian Cpk at regular intervals throughout the data and fits a Bayesian linear trend to the trajectory:

### Trajectory Outputs

| Output | Meaning |
|--------|---------|
| **Trajectory plot** | Cpk median and credible intervals at each time step |
| **Trend slope** | Rate of Cpk change per observation |
| **P(Cpk declining)** | Posterior probability that capability is getting worse |
| **Est. obs to threshold** | When Cpk will cross below the target (if declining) |

### Early Warning

If P(Cpk declining) > 80% and the estimated crossing point is within your planning horizon, you have an **actionable early warning.** Address the root cause before capability actually drops below the threshold.

This is something classical capability analysis simply cannot provide — it gives you a snapshot, not a trajectory.
""",
    "interactive": {
        "type": "pbs_demo",
        "config": {"analysis": "pbs_cpk"}
    },
    "key_takeaways": [
        "Classical Cpk is a point estimate that hides enormous uncertainty at small sample sizes",
        "Bayesian Cpk gives a full posterior distribution via ancestral sampling from Normal-Gamma",
        "P(Cpk > 1.33) is the decision-relevant output — not the point estimate itself",
        "Cpk Trajectory tracks capability over time and detects declining trends",
        "P(Cpk declining) provides early warning before capability actually drops below threshold",
    ],
    "practice_questions": [
        {
            "question": "Your Bayesian Cpk shows median 1.42, 90% CI [0.98, 1.86], P(>1.33) = 58%. Your customer requires Cpk > 1.33. What do you tell them?",
            "answer": "Be honest: there is a 58% chance the process meets their 1.33 standard, but a 42% chance it doesn't. The credible interval spans from below 1.0 to above 1.8, indicating substantial uncertainty. Recommend collecting more data to narrow the interval. With more observations, if the process is truly capable, P(>1.33) will increase. If it's not, you'll know sooner.",
            "hint": "Would you bet on a 58% probability for a critical quality characteristic?"
        },
        {
            "question": "The Cpk Trajectory shows P(declining) = 91% with estimated 150 observations to the 1.33 threshold. Current Cpk median is 1.55. What's the priority?",
            "answer": "High priority. Despite the comfortable current Cpk of 1.55, the process is deteriorating with 91% confidence. At the current rate, you have roughly 150 observations before capability drops below the 1.33 threshold. Start investigating root causes now — tool wear, material drift, environmental changes. The trajectory analysis gives you lead time that a single Cpk snapshot never could.",
            "hint": "A current Cpk of 1.55 would normally look great — but what's the trajectory saying?"
        },
    ]
}


# ═══════════════════════════════════════════════════════════════════════════
# Section 6: Health Fusion & Narrative
# PBS Components: MultiStreamHealth, UncertaintyFusion, ProcessNarrative,
#                 InvestigationTimeline, TaguchiLoss
# ═══════════════════════════════════════════════════════════════════════════

PBS_HEALTH_FUSION = {
    "id": "pbs-health-fusion",
    "title": "The Full Picture: Health Fusion & Narrative",
    "intro": "A process can be 'in control' on an SPC chart while failing on capability. It can look capable while the measurement system is degraded. PBS fuses all signals — SPC, capability, gage, trend, material, environment — into one honest health score and a plain-language narrative.",
    "exercise": {
        "title": "Try It: The Multi-Panel PBS Dashboard",
        "steps": [
            "Run the full PBS analysis with spec limits and target",
            "Read the Process Narrative — it describes the process state in plain English",
            "Check the Health score — what's the primary driver?",
            "Look at the Investigation Timeline: how many regimes were detected?",
            "For each regime, check the per-regime Cpk and its credible interval",
            "If Taguchi Loss is shown, identify whether bias or variance dominates the cost"
        ],
        "dsw_type": "pbs:pbs_full",
        "dsw_config": {
            "column": "diameter_mm",
            "USL": 25.15,
            "LSL": 24.85,
            "target": 25.0
        },
    },
    "content": """
## The Problem: Siloed Metrics

Classical quality systems use separate tools that don't talk to each other:
- SPC chart says "in control" ✓
- Capability study says Cpk = 1.1 (marginal) ⚠
- Gage R&R shows 28% measurement variation ⚠
- Process is trending toward USL ⚠

An engineer looking only at the SPC chart would think everything is fine. PBS **fuses all signals** into one coherent view.

## MultiStream Health

PBS combines multiple health streams using a **log-linear opinion pool:**

$$H = \\exp\\left(\\sum_k w_k \\log(h_k)\\right)$$

### Default Weights

| Stream | Weight | What It Measures |
|--------|--------|-----------------|
| **SPC** | 35% | Shift probability (from BOCPD) |
| **Cpk** | 25% | Capability against spec |
| **Gage** | 15% | Measurement system quality (%GRR) |
| **Trend** | 15% | Process trajectory direction |
| **Material** | 5% | Material lot consistency |
| **Environment** | 5% | Environmental factors |

### Interpreting the Health Score

| Health | Meaning |
|--------|---------|
| > 80% | Healthy — all streams nominal |
| 60–80% | Watch — one or more streams degraded |
| 40–60% | Action needed — investigate primary driver |
| < 40% | Critical — immediate intervention required |

The **primary driver** tells you which stream is pulling health down the most. This focuses your investigation.

## Uncertainty Fusion: Honest Measurement

Every measurement has error. PBS accounts for this using the **measurement system variance** (gage uncertainty):

$$\\sigma^2_{observed} = \\sigma^2_{process} + \\sigma^2_{gage}$$

The fused estimate combines the observation with prior process knowledge:

$$\\hat{x}_{fused} = \\frac{x_{obs}/\\sigma^2_{gage} + \\mu_{process}/\\sigma^2_{process}}{1/\\sigma^2_{gage} + 1/\\sigma^2_{process}}$$

### %GRR Health

| %GRR | Status | Meaning |
|------|--------|---------|
| < 10% | Excellent | Measurement system is precise |
| 10–30% | Acceptable | Some measurement noise |
| > 30% | Attention | Measurement system is a significant source of variation |

When %GRR exceeds 30%, any SPC signals might be measurement noise rather than real process changes.

## Investigation Timeline

When shifts are detected, PBS constructs an **Investigation Timeline** linking all findings:

1. **Changepoints:** Where shifts occurred, with confirmation arc and robustness score
2. **Regimes:** Distinct process states, each with its own Cpk and credible interval
3. **Evidence corroboration:** E-value at each changepoint
4. **Known transitions:** Material lot changes, operator changes (if metadata provided)
5. **Per-lot capability:** When material lot column is available

### Regime Analysis

After a shift, PBS computes **per-regime statistics:**

> Regime 1 (obs 1–45, n=45): Cpk 1.52 [1.18, 1.86], P(>1.33) = 72%
> Regime 2 (obs 46–200, n=155): Cpk 1.21 [1.04, 1.38], P(>1.33) = 28%
> Mean displaced +0.031 toward USL.

This tells you exactly how capability changed after the shift.

## Taguchi Loss: The Cost of Imperfection

PBS computes the **expected Taguchi quality loss** per unit:

$$E[L] = k \\cdot [(\\mu - \\text{target})^2 + \\sigma^2_\\mu + E[\\sigma^2]]$$

Decomposed into:
- **Bias loss:** Cost of being off-target ($k \\cdot (\\mu - \\text{target})^2$)
- **Variance loss:** Cost of process spread ($k \\cdot E[\\sigma^2]$)
- **Uncertainty loss:** Cost of not knowing the mean precisely ($k \\cdot \\sigma^2_\\mu$)

If bias dominates, center the process. If variance dominates, reduce variation. The decomposition tells you where to invest improvement effort.

## The Process Narrative

PBS generates a **plain-language summary** using deterministic templates (no LLM, no hallucination risk):

> "Shift first detected at observation 45 (P = 82%), confirmed to P ≥ 95% by obs ~52 (robust — detected at all λ values). Regime 1 (obs 1–45, n=45): Cpk 1.52 [1.18, 1.86]. Regime 2 (obs 46–200, n=155): Cpk 1.21 [1.04, 1.38]. Mean displaced +0.031 toward USL. Evidence strength: 47:1 (strong). Overall health: 68%. Primary factor: cpk."

Every statement is traceable to a specific computation. No interpretation magic, no prompt engineering — just math rendered as text.
""",
    "interactive": {
        "type": "pbs_demo",
        "config": {"analysis": "pbs_full"}
    },
    "key_takeaways": [
        "MultiStream Health fuses SPC, capability, gage, trend, material, and environment into one score",
        "The primary driver tells you which stream is pulling health down — focus investigation there",
        "Uncertainty Fusion accounts for measurement system error in every observation",
        "The Investigation Timeline links changepoints, evidence, and regime statistics into a coherent story",
        "Taguchi Loss decomposes quality cost into bias vs variance — guiding improvement strategy",
        "The Process Narrative is deterministic and traceable — no LLM, no hallucination risk",
    ],
    "practice_questions": [
        {
            "question": "Health score is 55% with primary driver 'gage'. SPC stream is at 92%, Cpk stream is at 74%. What's happening?",
            "answer": "The measurement system is dragging down overall health. Despite good SPC stability (92%) and decent capability (74%), the gage stream is poor — likely >30% GRR. This means some of the observed process variation is actually measurement noise. Fix the measurement system first — both SPC and Cpk assessments may improve once gage noise is removed from the signal.",
            "hint": "The log-linear fusion means a single poor stream pulls down the overall health disproportionately."
        },
        {
            "question": "Taguchi Loss shows bias fraction = 72%, variance fraction = 28%. Total expected loss is $0.15/unit. What's the improvement priority?",
            "answer": "Bias is the dominant cost driver at 72%. The process mean is off-target, and centering it would eliminate most of the loss. Centering (adjusting the mean) is typically cheaper and faster than reducing variance (which often requires process or equipment changes). Centering alone would reduce expected loss from $0.15 to roughly $0.04/unit.",
            "hint": "Which is easier to fix — centering a process or reducing its spread?"
        },
    ]
}


# ═══════════════════════════════════════════════════════════════════════════
# Section 7: Advanced PBS
# PBS Components: ChartGenealogy, ProbabilisticAlarms
# ═══════════════════════════════════════════════════════════════════════════

PBS_ADVANCED = {
    "id": "pbs-advanced",
    "title": "Advanced PBS: Genealogy & Decision-Theoretic Alarms",
    "intro": "When you launch a new product, do you start from zero? When an alarm fires, do you always investigate? PBS answers both with math: prior inheritance from related processes, and cost-optimized alarm thresholds. This is organizational learning encoded in probability.",
    "exercise": {
        "title": "Try It: Sensitivity Tuning & the Full Workflow",
        "steps": [
            "Run a full PBS analysis with hazard_lambda = 50 (more sensitive)",
            "Compare to the auto-selected lambda — does it detect more or fewer changepoints?",
            "Look at the alarm thresholds: at what P(shift) does PBS recommend investigation?",
            "Consider: if a missed shift costs $10 and a false investigation costs $2, what should the threshold be?",
            "Think about how you would transfer this process's posterior to a new, similar product line"
        ],
        "dsw_type": "pbs:pbs_full",
        "dsw_config": {
            "column": "diameter_mm",
            "USL": 25.15,
            "LSL": 24.85,
            "target": 25.0,
            "hazard_lambda": 50
        },
    },
    "content": """
## Chart Genealogy: Learning Across Processes

Starting every new process from an uninformative prior wastes knowledge. If you've run similar processes before, their posteriors contain valuable information.

### Prior Inheritance

Chart Genealogy takes a parent process's posterior and **discounts** it for the child:

$$\\kappa_{child} = \\kappa_{parent} \\cdot f, \\quad \\alpha_{child} = \\max(\\alpha_{parent} \\cdot f, 0.5), \\quad \\beta_{child} = \\beta_{parent} \\cdot f$$

where $f \\in (0, 1]$ is the **transfer factor:**

| Transfer Factor | When to Use |
|----------------|-------------|
| 0.8–1.0 | Same machine, same material, minor product change |
| 0.5–0.7 | Same process type, different specifications |
| 0.2–0.4 | Related but different process |
| < 0.2 | Weak relationship — mostly starting fresh |

### Multi-Parent Priors

When multiple related processes exist, PBS can combine them:

$$\\mu_{child} = \\frac{\\sum w_i \\mu_i}{\\sum w_i}, \\quad \\kappa_{child} = \\frac{\\sum w_i f_i \\kappa_i}{\\sum w_i}$$

where $w_i$ is the relevance weight and $f_i$ is the transfer factor for each parent.

### The Benefit

| Approach | Observations to Reliable Cpk |
|----------|------------------------------|
| Uninformative prior | ~50–100 |
| Inherited prior ($f=0.5$) | ~20–30 |
| Strong inheritance ($f=0.8$) | ~10–15 |

Prior inheritance can cut your qualification time in half.

## Probabilistic Alarms: The Economics of Investigation

Classical SPC alarms are binary: point outside limits → investigate. But investigation has a cost, and missing a shift has a different cost. PBS makes this explicit.

### Decision-Theoretic Framework

Three cost parameters:

| Parameter | Symbol | Meaning | Example |
|-----------|--------|---------|---------|
| **Cost of missed shift** | $c_{miss}$ | Penalty for ignoring a real shift | $10 (scrap, rework, customer complaints) |
| **Cost of false alarm** | $c_{fa}$ | Penalty for investigating when nothing changed | $1 (wasted engineer time) |
| **Cost of investigation** | $c_{inv}$ | Fixed cost to investigate regardless of outcome | $2 (lab time, test materials) |

### The Optimal Threshold

PBS computes the threshold where expected costs are equal:

$$P^*(\\text{shift}) = \\frac{c_{fa} + c_{inv}}{c_{miss} + c_{fa}}$$

With the example costs: $P^* = (1 + 2) / (10 + 1) = 27\\%$

This means: **investigate when P(shift) > 27%.** This is much more sensitive than the classical "outside 3-sigma" rule (~0.3% false alarm rate), because the cost of missing a real shift ($10) far exceeds the cost of a false alarm ($1).

### Expected Cost Comparison

For any shift probability $P$, PBS computes:
- **Expected cost of ignoring:** $P \\cdot c_{miss}$
- **Expected cost of investigating:** $(1-P) \\cdot c_{fa} + c_{inv}$

The action with lower expected cost wins. This is rational, transparent, and auditable.

### Tuning for Different Contexts

| Context | $c_{miss}$ | $c_{fa}$ | $c_{inv}$ | Threshold |
|---------|-----------|---------|----------|-----------|
| Safety-critical (automotive, medical) | 100 | 1 | 5 | 6% |
| High-volume manufacturing | 10 | 1 | 2 | 27% |
| Low-volume, expensive parts | 50 | 5 | 10 | 27% |
| R&D/prototype | 5 | 2 | 3 | 71% |

Safety-critical processes have very low thresholds — investigate at the slightest signal. R&D processes are more tolerant of variation.

## The Complete PBS Workflow

1. **Setup:** Define prior (from genealogy or calibration data), set spec limits, configure costs
2. **Calibrate:** Process first observations to establish the initial belief state
3. **Monitor:** Stream observations through all 12 PBS components in parallel
4. **Detect:** BOCPD and E-Detector flag shifts, evidence accumulates
5. **Predict:** Predictive Chart forecasts future state
6. **Decide:** Probabilistic Alarms recommend action based on costs
7. **Investigate:** If alarm fires, the Investigation Timeline provides context
8. **Update:** After resolution, reset reference or inherit prior for new regime
9. **Learn:** The posterior from this process becomes a potential parent for future processes

This is a **closed loop** — the system learns continuously, and knowledge transfers across processes. No other SPC platform offers this.
""",
    "interactive": {
        "type": "pbs_demo",
        "config": {"analysis": "pbs_full"}
    },
    "key_takeaways": [
        "Chart Genealogy inherits priors from parent processes, cutting qualification time by 50%+",
        "Transfer factors control how much to trust the parent's posterior for the child",
        "Probabilistic Alarms use cost parameters to compute the economically optimal investigation threshold",
        "The threshold depends on the ratio of missed-shift cost to false-alarm cost",
        "Safety-critical contexts need very low thresholds (investigate at slight signal)",
        "PBS forms a closed loop: monitor → detect → decide → investigate → learn → transfer",
    ],
    "practice_questions": [
        {
            "question": "You're launching a new product on the same machine with the same material. The previous product's posterior has kappa=200, alpha=102, beta=0.05. What transfer factor would you use?",
            "answer": "Use a high transfer factor like 0.7–0.8. Same machine and same material means the process variability should be similar. At f=0.8, the child starts with kappa=160, alpha≈82, beta=0.04 — substantial prior information that will narrow credible intervals immediately. The child still adapts as its own data arrives, but it starts with a significant head start.",
            "hint": "Same machine and material suggests high transferability."
        },
        {
            "question": "A missed shift on your process costs $50 (scrap rework). A false investigation costs $5 (engineer time). Investigation itself costs $10. What should the alarm threshold be?",
            "answer": "P* = (c_fa + c_inv) / (c_miss + c_fa) = (5 + 10) / (50 + 5) = 15/55 ≈ 27%. Investigate whenever P(shift) > 27%. This is the threshold where the expected cost of ignoring equals the expected cost of investigating. Below 27%, the expected cost of a false alarm exceeds the expected cost of a missed shift; above 27%, the reverse is true.",
            "hint": "Apply the formula: P* = (c_fa + c_inv) / (c_miss + c_fa)."
        },
    ]
}

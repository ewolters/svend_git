# Advanced Process Control

## Five Innovation Frontiers — Research & Claude Code Exploration Guide

**SVEND / Eric Soper · March 2026**

---

## Frontier 1: Control Charts as Information Channels

**Core insight:** A control chart is a communication system. Shannon (1948) and Shewhart (1931) never met, but the unification is overdue. The chart transmits the true process state through a noisy channel (measurement error + common cause variation) to a receiver (the operator). Information theory gives us tools to analyze and optimize that channel explicitly — not just statistical power.

### The Central Question

What is the channel capacity of a Shewhart X-bar chart? Can we design control charts that maximize mutual information between the true process state and the observable signal, rather than minimizing ARL?

### Key Concepts

| Concept | Definition / Relevance |
|---|---|
| Mutual Information I(X;Y) | How much knowing the chart signal Y reduces uncertainty about the true process state X. Classical ARL optimization is a proxy for this — MI is the direct measure. |
| Channel Capacity C | Maximum MI achievable over all input distributions. Sets a theoretical ceiling on chart detectability regardless of design choices. |
| Signal-to-Noise Ratio | In SPC: delta/sigma. In IT: determines capacity directly. Subgroup size n affects SNR as sqrt(n) — same as power calculations, but now framed informationally. |
| Differential Entropy H(X) | Continuous analog of Shannon entropy. Measures process state uncertainty. A capable process has low H(X). Your i-type Cpk is built on this. |
| Data Processing Inequality | Any transformation of the signal can only lose information. Implication: rational subgrouping (averaging) loses information about within-subgroup dynamics. |

### Research Questions for Claude Code

- Derive the mutual information I(process_state; chart_signal) as a function of n, delta, sigma, and control limit width
- Compute channel capacity C for Xbar-R, EWMA(lambda), and CUSUM(k,h) — which chart transmits the most information per sample?
- How does the Data Processing Inequality apply to subgroup averaging? What within-subgroup information is destroyed?
- Can we design a chart that maximizes I(X;Y) rather than minimizes ARL? Do they converge or diverge?
- What is the relationship between your i-type Cpk (differential entropy) and chart channel capacity?

### Claude Code Exploration Commands

**Setup: Install dependencies**

```bash
pip install numpy scipy matplotlib seaborn scikit-learn
pip install pymc arviz  # for Bayesian components
```

**Experiment 1: Mutual Information vs. ARL**

```python
# Compute MI between true shift delta and detection event for Xbar chart
# Vary: n (subgroup size), delta (shift magnitude), sigma, k (limit width)
# Compare MI-optimal design vs ARL-optimal design

import numpy as np
from scipy.stats import norm
from sklearn.metrics import mutual_info_score

def simulate_xbar_channel(n, delta, sigma, k=3, n_sim=50000):
    """Simulate process state -> chart signal channel"""
    # IC state (delta=0) and OOC state (delta>0)
    ic_samples = np.random.normal(0, sigma/np.sqrt(n), n_sim)
    ooc_samples = np.random.normal(delta, sigma/np.sqrt(n), n_sim)
    ucl = k * sigma / np.sqrt(n)
    lcl = -ucl
    # Binary signal: 0=in-control, 1=signal
    ic_signals = ((ic_samples > ucl) | (ic_samples < lcl)).astype(int)
    ooc_signals = ((ooc_samples > ucl) | (ooc_samples < lcl)).astype(int)
    # Compute MI between state (0/1) and signal (0/1)
    states = np.array([0]*n_sim + [1]*n_sim)
    signals = np.concatenate([ic_signals, ooc_signals])
    return mutual_info_score(states, signals)

# Sweep over n and delta
for n in [1, 5, 10, 25]:
    for delta_sigma in [0.5, 1.0, 1.5, 2.0]:
        mi = simulate_xbar_channel(n, delta_sigma, sigma=1.0)
        print(f"n={n:3d}, delta/sigma={delta_sigma:.1f}, MI={mi:.4f} bits")
```

**Experiment 2: Chart Comparison by Information Transmitted**

```python
# Compare Xbar, EWMA, CUSUM on MI per sample (not ARL)
# Key question: does EWMA's smoothing gain or lose information?

def ewma_channel(lam, delta, sigma, k=3, n_sim=10000, max_steps=200):
    """Compute average MI per observation for EWMA chart"""
    # Track EWMA statistic evolution under IC and OOC
    results_ic, results_ooc = [], []
    for _ in range(n_sim):
        z = 0
        for t in range(max_steps):
            x_ic = np.random.normal(0, sigma)
            x_ooc = np.random.normal(delta, sigma)
            z_ic = lam * x_ic + (1-lam) * 0  # simplified
            z_ooc = lam * x_ooc + (1-lam) * 0
            ucl = k * sigma * np.sqrt(lam/(2-lam))
            results_ic.append(int(abs(z_ic) > ucl))
            results_ooc.append(int(abs(z_ooc) > ucl))
    states = [0]*len(results_ic) + [1]*len(results_ooc)
    signals = results_ic + results_ooc
    return mutual_info_score(states, signals)

for lam in [0.1, 0.2, 0.4, 0.8, 1.0]:  # lam=1.0 is Shewhart
    mi = ewma_channel(lam, delta=1.0, sigma=1.0)
    print(f"lambda={lam:.1f}, MI={mi:.4f} bits")
```

**Connection to SVEND i-type Cpk**

```python
# Differential entropy of process output as capability measure
# i-type Cpk = f(H(X), specification limits)
from scipy.stats import differential_entropy

def itype_cpk_normal(sigma, usl, lsl, mean=0):
    """i-type Cpk via differential entropy for normal distribution"""
    # Differential entropy of N(mu, sigma^2) = 0.5 * log(2*pi*e*sigma^2)
    H = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
    # Spec window entropy
    spec_width = usl - lsl
    H_spec = np.log(spec_width)
    # Capability ratio (information-theoretic)
    return H_spec - H  # positive = capable, negative = not capable

for sigma in [0.1, 0.2, 0.5, 1.0, 2.0]:
    cpk_it = itype_cpk_normal(sigma, usl=3, lsl=-3)
    print(f"sigma={sigma:.1f}, i-type Cpk={cpk_it:.3f} nats")
```

---

## Frontier 2: Reaction Plan Stability — The Closed-Loop SPC System

**The textbook stops at detection.** But the real system is: process → chart → operator → reaction → process. That is a closed-loop feedback control system. No one has applied control theory's stability criteria to this human-in-the-loop system. Clausewitz's friction — the gap between plan and execution — lives entirely in this loop.

### The Central Question

What is the loop gain of a reaction plan? Can an overreactive operator destabilize an otherwise stable process? What is the Nyquist stability criterion for a human-in-the-loop SPC system?

### Key Concepts

| Concept | Definition / Relevance |
|---|---|
| Loop Gain G | In control theory: product of all gains around the feedback loop. G > 1 at the critical frequency → instability. For SPC: operator reaction magnitude × process sensitivity. |
| Tampering (Deming) | Funnel experiment: adjusting a stable process increases variation. This is loop gain > 1 in engineering terms. The instability mechanism is now formalizable. |
| Phase Margin | How close is the system to instability? SPC reaction plans with no time delay have infinite phase margin — adding human reaction time delay changes everything. |
| Dead Time / Time Delay | Operator reaction delay between signal and adjustment. Pure dead time destabilizes feedback loops. Quality engineers ignore this completely. |
| Bode Plot for SPC | Plot loop gain vs. frequency for the SPC feedback system. Where gain > 1 AND phase shift > 180° → instability. Tampered processes have this signature. |
| Clausewitz Friction | Fog of war = measurement noise. Friction = operator lag, misclassification, incomplete reaction plans. Both degrade effective loop gain in unpredictable ways. |

### Research Questions for Claude Code

- Model the SPC feedback loop as a discrete-time control system. What is the transfer function?
- At what reaction magnitude does a stable process become unstable under periodic over-adjustment?
- How does operator reaction delay (dead time) change the stability boundary?
- Simulate Deming's funnel experiment as a control system and compute its loop gain
- Can we derive a 'stability certificate' for a reaction plan before deploying it?

### Claude Code Exploration Commands

**Experiment 3: Deming Funnel as Control System**

```python
import numpy as np
import matplotlib.pyplot as plt

def funnel_simulation(n=500, rule=1, sigma_process=1.0):
    """
    Simulate Deming funnel experiment as feedback control system
    rule=1: Never adjust (open loop)
    rule=2: Adjust by -x from last position (gain = -1, unstable)
    rule=3: Always aim at center (gain = -1 but from fixed reference)
    rule=4: Aim where last marble landed (random walk, gain = 1)
    Returns: positions, variance over time
    """
    positions = np.zeros(n)
    target = 0.0
    aim = 0.0

    for i in range(n):
        marble = aim + np.random.normal(0, sigma_process)
        positions[i] = marble
        if rule == 1:
            aim = 0  # never adjust
        elif rule == 2:
            aim = aim - marble  # adjust by negative of error from target
        elif rule == 3:
            aim = -marble  # always aim at center from current
        elif rule == 4:
            aim = marble  # aim where it landed (random walk)

    return positions

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
rules = [1, 2, 3, 4]
labels = ["Rule 1: No Adjust (Stable)", "Rule 2: Compensate (Unstable)",
          "Rule 3: Reset to Center", "Rule 4: Random Walk"]
for ax, rule, label in zip(axes.flat, rules, labels):
    pos = funnel_simulation(n=200, rule=rule)
    ax.plot(pos, alpha=0.7)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_title(f"{label}\nVar = {np.var(pos):.3f}")
    ax.set_xlabel("Sample"); ax.set_ylabel("Position")
plt.tight_layout()
plt.savefig("funnel_stability.png", dpi=150, bbox_inches='tight')
print("Saved funnel_stability.png")
```

**Experiment 4: Stability Boundary for Reaction Magnitude**

```python
def spc_feedback_system(gain, delay, sigma=1.0, n=1000, shift_at=500):
    """
    Simulate closed-loop SPC system:
    - Process generates output
    - Chart detects signals
    - Operator reacts with 'gain' magnitude after 'delay' samples
    """
    outputs = []
    process_mean = 0.0
    reaction_queue = []  # pending reactions

    for t in range(n):
        if t == shift_at:
            process_mean += 1.0  # introduce 1-sigma shift

        x = np.random.normal(process_mean, sigma)
        outputs.append(x)

        # Apply queued reactions (after delay)
        if len(reaction_queue) > 0 and reaction_queue[0][0] <= t:
            _, adj = reaction_queue.pop(0)
            process_mean += adj  # operator adjusts process

        # Operator detects signal (simplified: |x| > 3*sigma)
        if abs(x) > 3 * sigma:
            # Queue reaction with delay, magnitude = gain * deviation
            reaction_queue.append((t + delay, -gain * x))

    return np.array(outputs)

print("Gain | Delay | Pre-shift Var | Post-shift Var | Stable?")
print("-" * 60)
for gain in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    for delay in [0, 1, 3, 5]:
        out = spc_feedback_system(gain, delay)
        pre_var = np.var(out[:500])
        post_var = np.var(out[500:])
        stable = "YES" if post_var < 10 else "NO "
        print(f"{gain:4.1f} | {delay:5d} | {pre_var:13.3f} | {post_var:14.3f} | {stable}")
```

---

## Frontier 3: Capability as a Distribution Over Distributions

**Classical Cpk is a point estimate.** Bayesian b-type Cpk gives a posterior over a scalar. But the process itself is non-stationary — the distribution drifts. Cpk is not a number; it is a trajectory in distribution space. The innovation is treating capability as a process to be controlled, not merely measured.

### The Central Question

Can we put a control chart on Cpk itself, using the Wasserstein distance or Jensen-Shannon Divergence as the measurement? When does Cpk-drift constitute a process signal, and when is it common cause?

### Key Concepts

| Concept | Definition / Relevance |
|---|---|
| Wasserstein Distance W_p | Earth mover's distance between distributions. W_2 between two Gaussians has closed form. Natural 'distance' for tracking distribution drift. |
| Jensen-Shannon Divergence | Symmetric, bounded [0,1], square root is a metric. Your d-type Cpk foundation. Natural for change detection in distribution space. |
| Functional Data Analysis | Treat the entire distribution (or its parameters) as the observation. Control charts on functional data are an active research area. |
| Bayesian Posterior Predictive | P(Cpk_new \| data) — distributes over future capability given current evidence. Richer than a point estimate or even a credible interval. |
| Frechet Mean | Mean of a distribution over distributions in Wasserstein space. The 'average process capability' when you have multiple machines or shifts. |
| Non-stationarity Detection | PSI (your current tool) detects distributional shift. But PSI doesn't tell you *what kind* of drift — location, scale, shape, or all three. |

### Research Questions for Claude Code

- Build a Cpk control chart: compute Cpk at each subgroup, track it as a time series, apply EWMA/CUSUM to detect Cpk degradation
- Compute W_2 distance between consecutive process windows — does Wasserstein drift predict future Cpk failure before classical SPC triggers?
- Decompose distributional drift into location shift, scale shift, and shape change (higher moments) — each has a different assignable cause
- Implement your d-type Cpk (JSD-based) and compare its sensitivity to Cpk drift vs. classical b-type
- Build a Bayesian hierarchical model: capability varies by machine, shift, operator — what is the posterior over Cpk for each stratum?

### Claude Code Exploration Commands

**Experiment 5: Cpk Control Chart**

```python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def compute_cpk(data, usl, lsl):
    """Compute classical Cpk"""
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    if sigma == 0: return np.inf
    cpu = (usl - mu) / (3 * sigma)
    cpl = (mu - lsl) / (3 * sigma)
    return min(cpu, cpl)

def cpk_control_chart(n_subgroups=100, subgroup_size=5,
                       usl=3.0, lsl=-3.0,
                       drift_start=60, drift_rate=0.02):
    """
    Simulate process with gradual mean drift, track Cpk over time.
    Drift starts at subgroup 60, mean shifts by drift_rate per subgroup.
    """
    cpk_values = []
    true_mean = 0.0

    for i in range(n_subgroups):
        if i >= drift_start:
            true_mean += drift_rate  # gradual drift

        subgroup = np.random.normal(true_mean, 1.0, subgroup_size)
        cpk_values.append(compute_cpk(subgroup, usl, lsl))

    return np.array(cpk_values)

cpk_series = cpk_control_chart()

# Apply EWMA to Cpk series
lam = 0.2
ewma_cpk = np.zeros(len(cpk_series))
ewma_cpk[0] = cpk_series[0]
for i in range(1, len(cpk_series)):
    ewma_cpk[i] = lam * cpk_series[i] + (1-lam) * ewma_cpk[i-1]

print("Cpk Control Chart Statistics:")
print(f"  Pre-drift mean Cpk:  {np.mean(cpk_series[:60]):.4f}")
print(f"  Post-drift mean Cpk: {np.mean(cpk_series[60:]):.4f}")
print(f"  First EWMA < 1.0 at subgroup: {np.argmax(ewma_cpk < 1.0)}")
print(f"  First raw Cpk < 1.0 at subgroup: {np.argmax(cpk_series < 1.0)}")
```

**Experiment 6: Wasserstein Distance as Drift Metric**

```python
from scipy.stats import wasserstein_distance

def wasserstein_drift_chart(n_windows=50, window_size=30,
                             drift_start=30, drift_sigma_rate=0.05):
    """
    Track W1 distance between consecutive process windows.
    Drift introduces scale change (sigma increase) — invisible to mean charts.
    """
    w1_distances = []
    prev_window = np.random.normal(0, 1.0, window_size)

    for i in range(n_windows):
        sigma = 1.0 + (i - drift_start) * drift_sigma_rate if i >= drift_start else 1.0
        curr_window = np.random.normal(0, sigma, window_size)
        w1 = wasserstein_distance(prev_window, curr_window)
        w1_distances.append(w1)
        prev_window = curr_window

    return np.array(w1_distances)

w1 = wasserstein_drift_chart()
# Baseline: mean W1 before drift
baseline_mean = np.mean(w1[:30])
baseline_std = np.std(w1[:30])
ucl = baseline_mean + 3 * baseline_std
first_signal = np.argmax(w1[30:] > ucl) + 30

print(f"W1 Baseline: {baseline_mean:.4f} +/- {baseline_std:.4f}")
print(f"UCL: {ucl:.4f}")
print(f"First signal at window: {first_signal}")
```

---

## Frontier 4: Replacing Rational Subgrouping for Heterogeneous Processes

**Shewhart's rational subgroup requires homogeneous short-term variation.** High-mix/low-volume manufacturing, job shops, continuous processes — none satisfy this assumption. The standard response (ignore it) is intellectually dishonest. The research question: what replaces rational subgrouping when every unit is different?

### The Central Question

Can we build a covariate-adjusted control chart that removes between-unit heterogeneity before computing control statistics? What is the correct 'common cause' baseline when the process is intrinsically heterogeneous?

### Key Concepts

| Concept | Definition / Relevance |
|---|---|
| Covariate-Adjusted Charts | Regress out known sources of heterogeneity, plot residuals. Assumes the covariate relationship is stable — shift in residuals = assignable cause. |
| Mixed Effects SPC | Hierarchical model: observation = fixed process mean + random shift (machine/shift/lot) + error. Control chart on the random effect posterior. |
| Profile Monitoring | When the 'observation' is a curve, not a scalar (e.g., force-displacement trace). Control the profile, not a summary statistic. |
| T-method (Taguchi) | Mahalanobis-Taguchi System: multivariate Mahalanobis distance as a scalar health index. Your Bayesian workbench already touches this. |
| GAM-Residual Charts | Fit a Generalized Additive Model to the process, chart residuals. Handles non-linear covariate relationships without assuming linearity. |

### Research Questions for Claude Code

- Simulate a high-mix process where units differ by a known covariate (material lot, operator, tool wear). Show that naive Xbar-R fails and covariate-adjusted chart succeeds
- Implement a mixed-effects SPC model in Python (statsmodels or PyMC) — what does the random effect control chart look like?
- Profile monitoring: given a force-displacement curve, what is the 'normal' profile and how do you detect a shift?
- Compare T-method (Mahalanobis distance) to your a-type Cpk (Inverse-Wishart streaming) on a multivariate heterogeneous process

### Claude Code Exploration Commands

**Experiment 7: Covariate-Adjusted Control Chart**

```python
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm

def simulate_heterogeneous_process(n=200, shift_at=150):
    """
    High-mix process: output depends on material hardness (covariate).
    Assignable cause: tool wear shifts the residual after lot 150.
    Naive chart: confounds covariate effect with tool wear.
    Adjusted chart: correctly isolates tool wear signal.
    """
    np.random.seed(42)
    hardness = np.random.normal(50, 5, n)  # material covariate
    true_slope = 0.8  # output = slope * hardness + error
    tool_wear_effect = np.where(np.arange(n) >= shift_at, 2.0, 0.0)

    output = true_slope * hardness + tool_wear_effect + np.random.normal(0, 1, n)

    # Naive chart: just plot output
    naive_mean = np.mean(output[:shift_at])
    naive_std = np.std(output[:shift_at])

    # Adjusted chart: regress out hardness, plot residuals
    X = sm.add_constant(hardness[:shift_at])
    model = sm.OLS(output[:shift_at], X).fit()

    X_all = sm.add_constant(hardness)
    residuals = output - model.predict(X_all)
    res_mean = np.mean(residuals[:shift_at])
    res_std = np.std(residuals[:shift_at])

    # Detection: first point beyond 3-sigma
    naive_signal = np.argmax(np.abs(output[shift_at:] - naive_mean) > 3*naive_std)
    adj_signal = np.argmax(np.abs(residuals[shift_at:] - res_mean) > 3*res_std)

    print(f"Tool wear introduced at sample {shift_at}")
    print(f"Naive chart detects at sample:    {shift_at + naive_signal if naive_signal > 0 else 'NEVER'}")
    print(f"Adjusted chart detects at sample: {shift_at + adj_signal if adj_signal > 0 else 'NEVER'}")
    print(f"Naive R-squared: {np.corrcoef(hardness, output)[0,1]**2:.3f} (covariate contamination)")
    return residuals

residuals = simulate_heterogeneous_process()
```

---

## Frontier 5: Detection/Diagnosis Unification — The Real-Time Fault Classifier

**SPC detects. RCA diagnoses. These are treated as sequential steps separated by a war room meeting.** But the signal pattern itself contains diagnostic information. The goal: a chart that simultaneously estimates what went wrong while detecting that something went wrong. Your Synara Belief Engine is architecturally adjacent to this — Bayesian model selection over fault hypotheses, updated in real time.

### The Central Question

Can we build a Bayesian model selection system that maintains posterior probabilities over a library of fault modes, updated with each new observation? When the posterior on any single fault exceeds a threshold — that IS the control chart signal, and it comes with an assignable cause attached.

### Key Concepts

| Concept | Definition / Relevance |
|---|---|
| Bayesian Model Selection | Compute P(fault_k \| data) for each fault in library. Bayes factor: ratio of marginal likelihoods. The chart signal = posterior concentration. |
| Marginal Likelihood | P(data \| fault_k) = integral of P(data \| theta) P(theta \| fault_k) d_theta. Automatically penalizes overcomplex models (Occam's razor built in). |
| Sequential Bayes | Update posteriors online: P(fault \| data_1:t) = P(x_t \| fault) P(fault \| data_1:t-1) / Z. No batch processing needed. |
| Western Electric Patterns | Run rules are primitive pattern classifiers. Each rule corresponds (loosely) to a specific fault mode. Sequential Bayes is the principled generalization. |
| Synara Belief Engine | SVEND's existing Bayesian hypothesis tracker. Extension: fault library as hypothesis set, SPC observations as evidence stream. |
| FMEA Integration | Your FMEA already lists fault modes with occurrence/severity scores. These become prior probabilities P(fault_k) in the Bayesian fault classifier. |

### Research Questions for Claude Code

- Build a minimal fault library: mean shift up, mean shift down, variance increase, trend, oscillation — each with a generative model
- Implement sequential Bayesian model selection: update P(fault_k \| data) with each new point
- Compare detection latency: classical Shewhart vs. Bayesian fault classifier — does the classifier detect faster when it knows which fault to look for?
- Connect to FMEA: use FMEA occurrence ratings as priors, update with SPC data, output ranked fault hypotheses with posterior probabilities
- Implement the Bayes factor control chart: plot log(max_posterior / prior) — signal when this exceeds a threshold

### Claude Code Exploration Commands

**Experiment 8: Sequential Bayesian Fault Classifier**

```python
import numpy as np
from scipy.stats import norm, t as tdist

class BayesianFaultClassifier:
    """
    Real-time fault classifier that unifies detection and diagnosis.
    Maintains posterior P(fault_k | x_1:t) updated sequentially.
    """
    def __init__(self, mu0=0.0, sigma0=1.0):
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.history = []

        # Fault library: each fault is a generative model
        self.faults = {
            "in_control":    {"type": "normal", "mu": 0.0,  "sigma": 1.0},
            "mean_shift_up": {"type": "normal", "mu": 1.5,  "sigma": 1.0},
            "mean_shift_dn": {"type": "normal", "mu": -1.5, "sigma": 1.0},
            "variance_inc":  {"type": "normal", "mu": 0.0,  "sigma": 2.0},
            "trend":         {"type": "trend",  "rate": 0.1},
            "oscillation":   {"type": "oscillation", "amp": 1.5, "freq": 0.2},
        }

        # Equal priors (could be informed by FMEA occurrence ratings)
        n_faults = len(self.faults)
        self.log_posteriors = {k: np.log(1.0/n_faults) for k in self.faults}

    def log_likelihood(self, x, fault_name, t):
        """Compute log P(x | fault_k) for observation at time t"""
        f = self.faults[fault_name]
        if f["type"] == "normal":
            return norm.logpdf(x, f["mu"], f["sigma"])
        elif f["type"] == "trend":
            expected_mu = f["rate"] * t
            return norm.logpdf(x, expected_mu, self.sigma0)
        elif f["type"] == "oscillation":
            expected_mu = f["amp"] * np.sin(2 * np.pi * f["freq"] * t)
            return norm.logpdf(x, expected_mu, self.sigma0)

    def update(self, x):
        """Sequential Bayesian update for new observation x"""
        t = len(self.history)
        self.history.append(x)

        # Update log posteriors
        for fault_name in self.faults:
            self.log_posteriors[fault_name] += self.log_likelihood(x, fault_name, t)

        # Normalize (log-sum-exp for numerical stability)
        log_sum = np.log(sum(np.exp(v) for v in self.log_posteriors.values()))
        for k in self.log_posteriors:
            self.log_posteriors[k] -= log_sum

        return self.posteriors()

    def posteriors(self):
        """Return normalized posterior probabilities"""
        return {k: np.exp(v) for k, v in self.log_posteriors.items()}

    def signal(self, threshold=0.80):
        """Return fault name if any posterior exceeds threshold"""
        p = self.posteriors()
        best = max(p, key=p.get)
        if best != "in_control" and p[best] > threshold:
            return best, p[best]
        return None, None


# Simulate: process shifts at t=50 (mean shift up 1.5 sigma)
np.random.seed(42)
clf = BayesianFaultClassifier()
signal_detected_at = None
fault_identified = None

for t in range(100):
    if t < 50:
        x = np.random.normal(0, 1)
    else:
        x = np.random.normal(1.5, 1)  # true fault: mean_shift_up

    posteriors = clf.update(x)

    if t % 10 == 0 or t >= 48:
        best = max(posteriors, key=posteriors.get)
        print(f"t={t:3d} | x={x:+.2f} | Top fault: {best:<20s} P={posteriors[best]:.3f}")

    fault, prob = clf.signal(threshold=0.80)
    if fault and signal_detected_at is None:
        signal_detected_at = t
        fault_identified = fault
        print(f"\n*** SIGNAL at t={t}: {fault} (P={prob:.3f}) ***\n")
```

**Connecting Fault Classifier to FMEA Priors**

```python
# Use FMEA occurrence ratings to set informative priors
# FMEA occurrence scale: 1-10 (10 = most frequent)
fmea_occurrence = {
    "in_control":    0,   # not a fault
    "mean_shift_up": 6,   # common: tool wear, material variation
    "mean_shift_dn": 4,   # less common
    "variance_inc":  7,   # common: fixturing issues, measurement
    "trend":         5,   # moderate: gradual degradation
    "oscillation":   3,   # uncommon: vibration, thermal cycling
}

# Convert occurrence to prior probability (softmax over occurrences)
occ_vals = np.array([v for v in fmea_occurrence.values() if v > 0], dtype=float)
priors = np.exp(occ_vals) / np.sum(np.exp(occ_vals))

print("FMEA-informed priors:")
fault_names = [k for k, v in fmea_occurrence.items() if v > 0]
for name, prior in zip(fault_names, priors):
    print(f"  {name:<20s}: {prior:.3f}")
```

---

## Synthesis: How These Five Frontiers Connect

These are not five independent research threads. They form a unified architecture:

| Connection | Implication |
|---|---|
| Frontier 1 → Frontier 4 | Information-theoretic chart design (F1) tells us how much information is destroyed by naive subgrouping (F4). The optimal subgroup strategy maximizes channel capacity. |
| Frontier 2 → Frontier 5 | Reaction plan stability (F2) determines loop gain. The Bayesian fault classifier (F5) reduces loop gain by giving operators the right diagnosis immediately — less tampering, more targeted correction. |
| Frontier 3 → Frontier 5 | Distributional drift (F3) is what the fault classifier (F5) needs to track. When Cpk degrades, the fault classifier should identify which distributional parameter is responsible. |
| Frontier 1 → Frontier 5 | Mutual information (F1) provides the theoretical basis for the detection threshold in the Bayesian fault classifier (F5). Signal when MI between observations and fault hypothesis exceeds a threshold. |
| All Frontiers → SVEND | F1 → i-type Cpk extension. F2 → Reaction plan audit tool. F3 → Cpk control chart widget. F4 → Covariate-adjusted charting for heterogeneous processes. F5 → Synara Belief Engine integration with SPC stream. |

---

## Quick Reference: Python Packages

| Package | Use Case in This Research |
|---|---|
| numpy, scipy | Core numerical / statistical primitives |
| statsmodels | OLS regression, mixed effects models (F4) |
| pymc | Full Bayesian inference, hierarchical models (F3, F5) |
| arviz | Bayesian diagnostics, posterior visualization |
| sklearn.metrics | mutual_info_score for F1 experiments |
| matplotlib, seaborn | Visualization throughout |
| scipy.stats.wasserstein_distance | Wasserstein drift metric (F3) |

---

SVEND · Advanced Process Control Research · March 2026

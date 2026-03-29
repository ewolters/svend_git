**GRAPH-001: UNIFIED KNOWLEDGE GRAPH AND PROCESS MODEL**

**Version:** 0.1
**Status:** DESIGN
**Date:** 2026-03-28
**Authors:** Eric + Claude (Systems Architect)
**Lineage:** Box & Hunter (iterative experimentation) → Deming (theory of knowledge) → Synara (Bayesian belief engine) → GRAPH-001
**Compliance:**
- ISO 9001:2015 §4.4 (Quality Management System and its Processes)
- ISO 9001:2015 §6.1 (Actions to Address Risks and Opportunities)
- ISO 9001:2015 §7.1.6 (Organizational Knowledge)
- ISO 9001:2015 §9.1 (Monitoring, Measurement, Analysis, Evaluation)
- IATF 16949:2016 §6.1.2.1 (Risk Analysis)
- IATF 16949:2016 §10.2.3 (Problem Solving)
**Related Standards:**
- LOOP-001 ≥ 0.1 (Closed-Loop Operating Model — §9 is implemented by this standard)
- CANON-002 ≥ 1.0 (Investigation engine — subgraph scoping and writeback)
- QMS-001 ≥ 1.7 (Quality tools — FMEA, SPC, DOE feed the graph)
- RISK-001 ≥ 1.0 (Risk Registry — FMIS rows seed graph structure)
- STAT-001 (Statistical methodology — DOE calibrates edges)

---

## **1. SCOPE AND PURPOSE**

### **1.1 Purpose**

GRAPH-001 defines the unified Knowledge Graph and Process Model service. This is the persistent, org-wide representation of what an organization knows — and does not know — about its processes.

**Core Principle:**

> The graph is the hypothesis. Every edge is a claim about reality. Every DOE, every investigation, every SPC signal is evidence applied to that claim. The graph is Synara operating on structure — Bayesian belief update applied not to isolated hypotheses but to an entire topology of causal relationships. When the model breaks, it did its job: it surfaced that your knowledge is incomplete.

### **1.2 What This Is Not**

This is not a digital twin. It is not a physics simulation. It does not attempt to perfectly model truth. It is a group's representation of truth — structurally consistent until literally contradicted by newer or better knowledge. The model is SUPPOSED to break. Breaking demands investigation. Investigation produces knowledge. Knowledge updates the model. That is the loop.

### **1.3 Scope**

This standard covers:
- Theoretical foundation: graph vs model duality (§2)
- Node schema and type taxonomy (§3)
- Edge schema, evidence stacking, and Bayesian posteriors (§4)
- Interaction terms and manifold folding (§5)
- Gap exposure taxonomy (§6)
- FMIS seeding — how the graph grows from FMEA (§7)
- Investigation subgraph scoping and writeback (§8)
- Staleness detection from SPC (§9)
- Contradiction handling (§10)
- The unified service interface (§11)
- Synara integration — one engine, one truth (§12)
- Persistence model (§13)
- Value propagation and parameter exploration (§14)
- UX: graph-first navigation model (§15)
- Design constraints (§16)

This standard does NOT cover:
- The Synara engine internals (those are implementation; this standard defines the contract)
- Individual tool behavior (see QMS-001, DSW-001)
- Investigation methodology (see CANON-002)
- The closed-loop operating model (see LOOP-001; this standard implements §9)

### **1.4 Relationship to LOOP-001 §9**

LOOP-001 §9 defines the Dynamic Process Model as a component of the closed-loop operating model. GRAPH-001 supersedes §9 as the authoritative specification. Where LOOP-001 §9 describes WHAT the process model does within the loop, GRAPH-001 defines HOW — the schema, the math, the service boundary, and the integration contracts.

---

## **2. THEORETICAL FOUNDATION**

### **2.1 Graph and Model Are Two Sides of One Coin**

The system has two concerns that operate on the same structure:

| Concern | What it represents | What changes it |
|---------|-------------------|-----------------|
| **Graph** (topology) | Structure of relationships: X influences Y | Adding/removing nodes and edges |
| **Model** (calibration) | Quantification of those relationships: how much, how confidently, under what conditions | Evidence from DOE, investigation, SPC, process confirmation |

The graph says "Zone 3 temperature → viscosity → surface finish." The model says "the effect size is 0.3 ± 0.05, measured via DOE on 2026-02-15, with n=48."

Same edge. Two concerns. They are reflections of each other: the model models the graph, and the graph lays out the relations in the model.

### **2.2 The Graph as Hypothesis Chain**

The knowledge graph IS the deductive topology to which inferential outcomes are applied using Bayesian posteriors. This is Box & Hunter's iterative experimentation formalized as persistent infrastructure:

1. The graph encodes the organization's current theory of how its process works
2. Every edge is a hypothesis: "we believe X influences Y with this strength"
3. Evidence (DOE, investigation, SPC, process confirmation) updates the posterior on each edge
4. When posteriors diverge from observation, the theory is wrong somewhere
5. That divergence demands investigation
6. Investigation produces new knowledge
7. New knowledge updates the graph

This is Synara operating at graph scale — the same P(h|e) ∝ P(e|h) × P(h) that drives investigation-level reasoning, applied to the persistent structure of process knowledge.

### **2.3 The Graph Grows Backwards from Problems**

Nobody maps a full process from scratch. The graph grows iteratively, driven by the problems you're trying to solve:

```
Day 1   (FMEA):     injection_pressure → short_shots
Month 3 (DOE):      viscosity → short_shots              (new node)
Month 5 (SPC):      pellet_moisture → viscosity           (new node)
Month 8 (seasonal): ambient_humidity → pellet_moisture    (new node)
                    building_heating → ambient_temp → viscosity  (new nodes)
```

Each step is an investigation or observation that extends the causal frontier one hop further upstream from a failure. The graph converges on the real causal structure of the process over time. Completeness is not the goal — useful knowledge is.

---

## **3. NODE SCHEMA**

### **3.1 What Is a Node**

A node represents a discrete, identifiable entity in the process: something you can measure, control, observe, or specify. A node is NOT a hypothesis — it is the physical (or logical) thing that hypotheses are ABOUT.

### **3.2 Node Types**

| Type | Description | Examples |
|------|-------------|---------|
| `process_parameter` | A measurable quantity in the process | Temperature, pressure, speed, flow rate |
| `quality_characteristic` | A measurable output attribute | Surface finish, dimension, strength, weight |
| `failure_mode` | A known way the process can fail | Short shot, flash, sink marks, delamination |
| `environmental_factor` | External condition affecting the process | Ambient humidity, ambient temperature, seasonal variation |
| `material_property` | Property of input material | Pellet moisture, hardness, viscosity, lot variation |
| `measurement` | A measurement system or method | Gage, CMM, visual inspection, SPC chart |
| `specification` | Acceptable range for an output | USL, LSL, target (linked to the output node it constrains) |
| `equipment` | Machine, tooling, or fixture | Press #3, Mold cavity 2, Nozzle assembly |
| `human_factor` | Operator skill, training, procedure adherence | Shift, operator experience, procedure version |

This taxonomy is extensible. Organizations may define custom types. The type is a property of the node, not the edge — the nature of what something IS belongs to the node, not to how it connects.

### **3.3 Node Fields**

```
Node:
  id:                UUID
  graph:             FK → Graph (the parent graph this node belongs to)
  name:              str                    # "Zone 3 Temperature"
  node_type:         str (from §3.2)        # "process_parameter"
  description:       text                   # operational definition

  # Operating state (nullable — populated as data arrives)
  unit:              str                    # "°C"
  distribution:      JSON                   # {mean, std, shape, n, source, as_of}
  spec_limits:       JSON                   # {usl, lsl, target} — if applicable
  control_limits:    JSON                   # {ucl, lcl, cl} — if under SPC

  # Controllability
  controllability:   enum                   # direct, indirect, noise, fixed
  control_method:    str                    # "PID loop", "operator setpoint", null

  # Linkage (nullable FKs or UUIDs)
  linked_fmis_rows:  [UUID]                # FMIS rows where this node appears
  linked_equipment:  [UUID]                # MeasurementEquipment records
  linked_spc_chart:  UUID                  # Active SPC chart monitoring this node

  # Metadata
  created_at:        datetime
  updated_at:        datetime
  created_by:        FK → User
  provenance:        str                   # "fmea_seed", "investigation", "manual", "spc"
```

### **3.4 Node Identity**

"Zone 3 temperature" and "Zone 1 temperature" are different nodes. They are different physical locations with potentially different behavior. If they behave identically, that's an edge between them (correlational), not a reason to merge them.

Granularity is determined by the organization's ability to distinguish and control. If you can't measure Zone 1 and Zone 3 separately, they're one node. If you can, they're two.

---

## **4. EDGE SCHEMA**

### **4.1 What Is an Edge**

An edge is a claimed relationship between two nodes: "we believe X influences Y." The claim carries a Bayesian posterior — the organization's current degree of belief in this relationship, given all evidence observed to date.

### **4.2 Edge Relationship Types**

| Type | Semantics | Example |
|------|-----------|---------|
| `causal` | X mechanistically influences Y | temperature → viscosity |
| `correlational` | X and Y move together, causation not established | humidity ↔ defect rate |
| `confounded` | X appears to cause Y but known confounder exists | (links to confounder node) |
| `specification` | X constrains Y | USL → surface_finish |
| `measurement` | X is measured by Y | viscosity → viscometer_3 |

### **4.3 Edge Fields**

```
Edge:
  id:                 UUID
  graph:              FK → Graph
  source:             FK → Node
  target:             FK → Node
  relation_type:      str (from §4.2)

  # --- MODEL LAYER (Bayesian belief state) ---

  # Current posterior (the "relevant average")
  effect_size:        float (nullable)       # point estimate
  effect_ci_lower:    float (nullable)       # confidence interval lower
  effect_ci_upper:    float (nullable)       # confidence interval upper
  posterior_strength:  float (0.0-1.0)       # Synara posterior — belief this relationship exists and has this effect

  # Direction and shape
  direction:          enum                   # positive, negative, nonlinear, unknown
  linearity:          enum                   # linear, nonlinear, threshold, unknown

  # Operating region (edge metadata, not structural — see §5)
  operating_region:   JSON                   # conditions under which this edge manifests
                                             # e.g., {"humidity": {">": 60}, "temp_range": [150, 200]}

  # --- EVIDENCE LAYER (what supports this belief) ---

  evidence_stack:     → EdgeEvidence[]       # reverse FK — all evidence records for this edge
  evidence_count:     int                    # denormalized count

  # --- PROVENANCE ---

  provenance:         str                    # "fmea_assertion", "doe", "investigation", "operator", "spc", "literature"
  source_investigation: UUID (nullable)      # investigation that created/last calibrated this edge
  calibration_date:   datetime (nullable)    # when this edge was last calibrated with empirical data

  # --- INTERACTION (see §5) ---

  interaction_terms:  JSON                   # [{node_id, modulation_type, parameters}]

  # --- HEALTH ---

  is_stale:           bool                   # flagged by SPC or time-based policy
  staleness_reason:   str (nullable)         # "spc_shift_detected", "calibration_expired", "contradiction"
  is_contradicted:    bool                   # newer evidence contradicts posterior
  contradiction_id:   UUID (nullable)        # → Signal that was raised

  # --- METADATA ---

  created_at:         datetime
  updated_at:         datetime
  created_by:         FK → User
```

### **4.4 Edge Evidence Stacking**

Each edge maintains a stack of evidence records — timestamped, UUID-linked observations from any source:

```
EdgeEvidence:
  id:                UUID
  edge:              FK → Edge

  # What was observed
  effect_size:       float (nullable)
  confidence_interval: JSON (nullable)      # {lower, upper}
  sample_size:       int (nullable)
  p_value:           float (nullable)

  # Source
  source_type:       str                    # "doe", "investigation", "spc", "process_confirmation",
                                            # "forced_failure_test", "gage_rr", "operator", "literature"
  source_id:         UUID (nullable)        # FK to the source object
  source_description: str                   # human-readable description of what was observed

  # Reliability
  strength:          float (0.0-1.0)        # measurement reliability / study quality

  # Temporal
  observed_at:       datetime               # when the evidence was generated
  created_at:        datetime               # when it was recorded in the graph
  created_by:        FK → User
```

**The posterior is a recency-weighted aggregate of the evidence stack.** More recent evidence weighs more heavily, but old evidence is never discarded — it contributes to the historical understanding. The weighting function is:

```
w(evidence) = strength × recency_decay(age)
posterior = weighted_center(evidence_stack, w)
```

The recency decay function is org-configurable via QMS Policy. Default: exponential decay with half-life of 180 days. This means a DOE from last month dominates, but a DOE from two years ago still contributes — it just matters less.

Synara's `BeliefEngine.update_posteriors()` performs this computation. Each new evidence record triggers a Bayesian update: P(edge|evidence) ∝ P(evidence|edge) × P(edge).

### **4.5 One Edge, Not Two**

When the FMEA says "injection pressure too low causes short shots," that is ONE edge. The relationship is causal. The fact that "short shots" is a failure mode is a property of the TARGET NODE (node_type = failure_mode), not a separate edge type.

The FMEA is a VIEW that filters the graph to show:
- Nodes of type `failure_mode`
- Their upstream causal edges
- The edge posteriors rendered as S/O/D scores

This eliminates the need for a separate "failure dimension." The FMEA does not own its own graph — it reads from the unified graph through a typed filter.

---

## **5. INTERACTION TERMS AND MANIFOLD FOLDING**

### **5.1 The Problem**

Real processes have interactions. Melt temperature and injection pressure both affect fill completeness. When you change one, the effect of the other changes. This is not a new edge — it's the existing edge behaving differently as a function of other node states.

### **5.2 The Graph Folds, It Does Not Restructure**

The graph topology is stable. Nodes and edges do not appear or disappear based on operating conditions. What changes is the MANIFOLD the graph describes — the shape of the response surface as you move through parameter space.

Interactions are **position features**: they allow the graph's behavior to change within certain bounds without changing its structure. Moving one node can affect multiple edges simultaneously. This is the fold.

### **5.3 Interaction Term Schema**

Each edge may carry interaction terms that modulate its effect size as a function of other node states:

```
interaction_terms: [
  {
    "modulating_node":  UUID,           # the node whose state modulates this edge
    "modulation_type":  str,            # "auto", "unknown" — see §5.4
    "fit_result":       JSON (nullable) # auto-fitted model parameters (null until calibrated)
    "fit_metric":       str (nullable)  # selection criterion used: "aic", "bic"
    "fit_score":        float (nullable)# model selection score
    "source":           UUID,           # investigation/DOE that identified this interaction
    "calibrated":       bool            # do we have empirical parameters or just the assertion?
  }
]
```

### **5.4 Interaction Calibration: Assert First, Fit Later**

Interactions follow the same lifecycle as edges: start as an assertion, graduate to calibrated through evidence.

**Phase 1 — Assertion (uncalibrated):**
An operator or investigation asserts "melt temp and injection pressure interact on fill completeness." The interaction term is created with `modulation_type: "unknown"`, `calibrated: false`. The system knows the interaction exists but not its shape. This is a gap (§6.2).

**Phase 2 — Auto-fit (calibrated):**
When DOE data arrives that covers the interaction space, the system fits the modulation function automatically using the same model selection logic as the Analysis Workbench (AIC/BIC over a family of candidate functions). The user does not manually select "linear" or "threshold" — the evidence selects the functional form.

Candidate function family:
- Constant (no real interaction — null hypothesis)
- Linear
- Threshold / step function
- Low-degree polynomial (quadratic, cubic)
- Piecewise linear

The winning model's parameters are stored in `fit_result`. If the best fit is "constant," the interaction term was a false assertion — the system flags this for review rather than silently removing it.

**Phase 3 — Refinement:**
As more evidence accumulates, the fit is periodically re-evaluated. If a simpler model now explains the data equally well (lower AIC), the system proposes a model simplification. If the data now requires a more complex model, the system proposes an upgrade. Both require human confirmation.

**Example:** Edge from injection_pressure → fill_completeness has interaction term:
```json
{
  "modulating_node": "<melt_temp_uuid>",
  "modulation_type": "auto",
  "fit_result": {"form": "linear", "coefficient": -0.15, "centered_at": 220},
  "fit_metric": "aic",
  "fit_score": -42.3,
  "source": "<doe_uuid>",
  "calibrated": true
}
```

This means: the effect of injection pressure on fill completeness decreases by 0.15 per degree of melt temp above 220°C (auto-fitted from DOE data, linear form won AIC selection). You can compensate low pressure with higher temp. The graph structure didn't change — the manifold folded.

### **5.5 Propagation with Folding**

Synara's `propagate_belief()` currently computes:

```python
influence = link.strength * hypothesis.posterior
```

With interaction terms, this becomes:

```python
effective_strength = link.strength
for interaction in link.interaction_terms:
    modulator_state = get_node_state(interaction.modulating_node)
    effective_strength = apply_modulation(effective_strength, modulator_state, interaction)
influence = effective_strength * hypothesis.posterior
```

For uncalibrated interactions (`calibrated: false`), `apply_modulation` returns `strength` unchanged — the interaction is acknowledged but cannot modulate because we don't know its shape. This is a gap, not a failure.

### **5.6 The Slider UI Implication**

If the graph and model are reflections of the same thing, and interactions are position features that fold the manifold, then the user interface is a set of sliders and toggles that navigate the manifold directly. Moving a slider changes a node's state, which modulates edges via interaction terms, which propagates through the graph, which updates downstream node distributions.

This is not simulation bolted on top. It IS the model. The user is navigating the organization's knowledge of its process in real time.

---

## **6. GAP EXPOSURE**

### **6.1 Purpose**

The graph's primary value is exposing what you don't know. A complete, perfectly calibrated graph is neither achievable nor the goal. The goal is to make ignorance visible so the organization can decide where to invest investigation effort.

### **6.2 Gap Taxonomy**

| Gap Type | Detection | Signal? | Description |
|----------|-----------|---------|-------------|
| **Uncalibrated edge** | `edge.evidence_count == 0 AND edge.provenance == "fmea_assertion"` | No (informational) | "We assert this relationship exists but have no measured effect size." |
| **Stale edge** | `edge.is_stale == True` (set by §9) | Yes → Signal | "This relationship was calibrated N months ago. Process has changed since." |
| **Missing entity** | FMIS row references a cause with no corresponding node | No (informational) | "FMEA lists 'material hardness' as a cause but we have no measurement system for it." |
| **Conflicting evidence** | Two evidence records on same edge with contradictory effect sizes beyond CI overlap | Yes → Signal | "Two investigations measured the same relationship and got different results." |
| **Uncalibrated interaction** | `interaction_term.calibrated == False` | No (informational) | "We know these factors interact but haven't measured the interaction effect." |
| **Measurement gap** | Node has no linked measurement system | No (informational) | "We model this parameter but can't actually measure it." |
| **Low-confidence edge** | `edge.posterior_strength < threshold` (org-configurable) | No (informational) | "We have some evidence for this relationship but confidence is low." |

### **6.3 Gap Report**

The service provides a `gap_report(graph)` function that returns all gaps organized by type and priority. Priority is determined by:

1. **Proximity to failure modes** — gaps on edges closer to failure_mode nodes are higher priority
2. **RPN contribution** — if the edge feeds into an FMIS row, the row's RPN influences priority
3. **Staleness age** — older stale edges are higher priority
4. **Evidence conflict severity** — larger contradictions between evidence records are higher priority

### **6.4 Gap Exposure as Synara Expansion Signal**

Synara already has `ExpansionSignal` — "evidence doesn't fit any hypothesis, causal surface is incomplete." In the graph context, expansion signals fire when:

- New evidence has low likelihood under ALL existing edges (§6.2 "conflicting evidence")
- An SPC signal occurs on a node that has no upstream causal edges (missing structure)
- A forced failure test produces unexpected results (detection model is wrong)

These expansion signals become LOOP-001 Signals, entering the closed loop.

---

## **7. FMIS SEEDING**

### **7.1 How the Graph Grows from FMEA**

The FMIS (Failure Modes Investigation System) is the primary structural input to the graph. Each FMIS row asserts a causal chain:

```
cause (mechanism) → failure_mode → effect (on output)
```

When an FMIS row is created, the graph service:

1. **Creates or finds nodes** for cause, failure mode, and effect
2. **Creates edges** from cause → failure_mode and failure_mode → effect
3. **Sets provenance** to `fmea_assertion` (uncalibrated)
4. **Links back** to the FMIS row (bidirectional UUID reference)

The user confirms, modifies, or rejects the proposed structure. The graph does not auto-populate silently.

### **7.2 Progressive Enrichment**

| Activity | What it contributes | Graph effect |
|----------|-------------------|--------------|
| FMIS row creation | Structural assertion | New nodes + uncalibrated edges |
| Investigation with DOE | Calibrated causal evidence | Effect size + CI on edge, new evidence record |
| Investigation without DOE | Observational evidence | Weaker evidence record, wider CI |
| Process Confirmation | Expected behavior check | Confirms or challenges edge posterior |
| Forced Failure Test | Detection probability | Evidence on detection-related edges |
| SPC data | Ongoing variability | Node distribution updates, staleness flags on edges |
| Gage R&R | Measurement capability | Evidence on measurement edges, trustworthiness of other edges |

### **7.3 FMEA as Graph View**

The FMEA is not a separate data structure. It is a filtered, rendered view of the graph:

- Show all nodes of type `failure_mode`
- Show their upstream edges (causes) and downstream edges (effects)
- Render edge posteriors as Severity, Occurrence, Detection scores
- Display gap indicators on uncalibrated edges

This means editing the FMEA edits the graph. There is one source of truth.

---

## **8. INVESTIGATION SUBGRAPH**

### **8.1 Scoping**

An investigation does not operate on the entire graph. It operates on a subset, scoped by:

- **Relevant nodes**: the specific parameters, outputs, and failure modes under investigation
- **Relevant edges**: edges connecting those nodes
- **Time window**: the period under investigation

The investigation's Synara causal graph IS a subgraph of the process model. The `synara_state` JSON on the Investigation model stores this subgraph.

### **8.2 Subgraph Extraction**

```python
service.scope_for_investigation(
    graph_id=<uuid>,
    node_ids=[<uuid>, ...],          # user-selected nodes
    include_neighbors=True,           # optionally include 1-hop neighbors
) → SubgraphSnapshot
```

The snapshot includes copies of nodes, edges, and their current evidence stacks. The investigation works on this snapshot — changes are local until writeback.

### **8.3 Writeback**

When an investigation concludes, its findings are proposed as updates to the parent graph:

1. **New edges**: causal relationships discovered during investigation
2. **Updated edges**: new evidence records (effect sizes from DOE, observational evidence)
3. **New nodes**: entities discovered that weren't in the graph (e.g., pellet moisture)
4. **Contradiction flags**: investigation results that conflict with existing edge posteriors

The investigator reviews and confirms which findings should persist in the graph. The system does NOT silently overwrite. Confirmed findings create new `EdgeEvidence` records and trigger Synara posterior updates.

```python
service.write_back_from_investigation(
    investigation_id=<uuid>,
    proposed_changes=[
        {"type": "new_node", "node": {...}},
        {"type": "new_edge", "edge": {...}},
        {"type": "new_evidence", "edge_id": <uuid>, "evidence": {...}},
    ],
    confirmed_by=<user_id>,
) → WritebackResult
```

### **8.4 Contradiction Detection on Writeback**

If a proposed evidence record contradicts an existing edge's posterior beyond a configurable threshold:

1. The writeback is NOT blocked — the evidence is recorded
2. The edge is flagged: `is_contradicted = True`
3. A LOOP-001 Signal is raised: "Investigation <uuid> produced evidence contradicting edge <uuid>"
4. The org must investigate the contradiction (which may produce further evidence resolving it)

---

## **9. STALENESS DETECTION**

### **9.1 SPC-Driven Staleness**

When SPC detects a process shift on a node:

1. All edges where this node is the source OR target are evaluated
2. If the shift magnitude is large enough to meaningfully change edge behavior (org-configurable threshold), those edges are flagged: `is_stale = True, staleness_reason = "spc_shift_detected"`
3. A LOOP-001 Signal is raised per stale edge

The graph does NOT auto-update edges from SPC signals. It surfaces the staleness and demands investigation.

### **9.2 Time-Based Staleness**

Edges have a `calibration_date`. The org configures a maximum calibration age via QMS Policy (default: 365 days). Edges exceeding this age are flagged as stale.

### **9.3 Staleness Resolution**

A stale edge is resolved by:
- Running a new DOE or investigation that produces fresh evidence
- An explicit operator review confirming "this relationship still holds" (creates evidence record of type "operator" with lower strength)
- QMS Policy override for edges where recalibration is impractical

---

## **10. CONTRADICTION HANDLING**

### **10.1 What Is a Contradiction**

A contradiction occurs when new evidence pushes an edge's posterior in a direction that conflicts with the existing posterior beyond a significance threshold. Specifically:

- Evidence suggests effect size of opposite sign
- Evidence suggests effect size outside the existing CI by more than 2× the CI width
- Evidence from a higher-reliability source (DOE) contradicts a lower-reliability source (operator assertion)

### **10.2 Contradiction Lifecycle**

```
New evidence arrives
    ↓
Synara computes likelihood — low likelihood under current edge posterior
    ↓
Contradiction detected (expansion signal variant)
    ↓
Edge flagged: is_contradicted = True
    ↓
LOOP-001 Signal raised
    ↓
Investigation scoped to the contradicted edge
    ↓
Investigation produces resolution evidence
    ↓
Edge posterior updated — contradiction resolved OR edge fundamentally revised
```

### **10.3 The Graph Does Not Silently Resolve**

Contradictions require human judgment. The system records the evidence, raises the signal, and waits. It does not average conflicting results, does not pick the "better" study, does not auto-resolve. The organization's theory of its process is at stake — that requires investigation, not arithmetic.

---

## **11. SERVICE INTERFACE**

### **11.1 GraphService**

The unified service that all surfaces call. One interface, one truth.

```python
class GraphService:
    """Unified Knowledge Graph + Process Model service.

    Every tool in the platform reads from or writes to the graph
    through this service. There is no other path to graph state.
    """

    # --- Graph Lifecycle ---

    def create_graph(tenant_id, name, description) → Graph
    def get_graph(graph_id) → Graph
    def get_org_graph(tenant_id) → Graph          # the org's primary process graph

    # --- Node Operations ---

    def add_node(graph_id, node_data) → Node
    def update_node(node_id, updates) → Node
    def remove_node(node_id) → None               # cascades to edges
    def get_node(node_id) → Node
    def get_nodes(graph_id, filters) → [Node]      # filter by type, name, etc.

    # --- Edge Operations ---

    def add_edge(graph_id, source_id, target_id, edge_data) → Edge
    def update_edge(edge_id, updates) → Edge
    def remove_edge(edge_id) → None
    def get_edge(edge_id) → Edge
    def get_edges(graph_id, filters) → [Edge]      # filter by type, staleness, etc.

    # --- Evidence (the core operation) ---

    def add_evidence(edge_id, evidence_data) → EdgeEvidence
        """Add evidence to an edge and trigger Synara posterior update.

        This is the primary write operation. Everything else in the platform
        that learns something about a process relationship calls this.
        Returns the new evidence record and the updated edge posterior.
        Raises Signal if contradiction detected.
        """

    # --- FMIS Integration ---

    def seed_from_fmis(graph_id, fmis_id) → [ProposedChange]
        """Generate proposed nodes + edges from FMIS rows.
        Returns proposals for user confirmation, does not auto-commit."""

    def confirm_seed(graph_id, proposals, confirmed_by) → [Node | Edge]

    # --- Investigation Integration ---

    def scope_for_investigation(graph_id, node_ids, include_neighbors) → SubgraphSnapshot
    def write_back_from_investigation(investigation_id, proposed_changes, confirmed_by) → WritebackResult

    # --- SPC Integration ---

    def flag_stale_edges(node_id, spc_signal) → [Edge]
        """Called when SPC detects a shift on a node. Flags affected edges."""

    def update_node_distribution(node_id, distribution_data) → Node
        """Called by SPC to update a node's current distribution."""

    # --- Gap Analysis ---

    def gap_report(graph_id) → GapReport
        """Full gap taxonomy across the graph."""

    def get_uncalibrated_edges(graph_id) → [Edge]
    def get_stale_edges(graph_id) → [Edge]
    def get_contradicted_edges(graph_id) → [Edge]
    def get_measurement_gaps(graph_id) → [Node]

    # --- Queries ---

    def get_upstream(node_id, depth=None) → [Node]     # all ancestors
    def get_downstream(node_id, depth=None) → [Node]   # all descendants
    def get_paths_between(source_id, target_id) → [[Node]]
    def get_causal_chains_to(node_id) → [[Node]]       # all root-to-node paths
    def explain_edge(edge_id) → EdgeExplanation         # full evidence history + posterior reasoning

    # --- Value Propagation (§14) ---

    def propagate_values(graph_id, input_states, n_iterations=10000) → PropagationResult
    def parameter_sweep(graph_id, node_id, value_range, fixed_states) → SweepResult
    def sensitivity_analysis(graph_id, output_node_id) → SensitivityResult
    def counterfactual(graph_id, input_states, actual_outputs) → CounterfactualResult
```

### **11.2 Who Calls What**

| Surface | Service method | Direction |
|---------|---------------|-----------|
| **FMIS** | `seed_from_fmis()`, `confirm_seed()` | Write (structure) |
| **Investigation** | `scope_for_investigation()`, `write_back_from_investigation()` | Read then Write |
| **DOE** | `add_evidence()` on calibrated edges | Write (evidence) |
| **SPC** | `flag_stale_edges()`, `update_node_distribution()` | Write (health) |
| **Process Confirmation** | `add_evidence()` | Write (evidence) |
| **Forced Failure Test** | `add_evidence()` on detection edges | Write (evidence) |
| **Gage R&R** | `add_evidence()` on measurement edges | Write (evidence) |
| **FMEA view** | `get_nodes(type=failure_mode)`, `get_edges()` | Read |
| **Auditor Portal** | `gap_report()`, `get_nodes()`, `get_edges()` | Read |
| **Whiteboard** | `add_node()`, `add_edge()`, `update_edge()` | Read + Write (visual editor) |
| **Process Explorer** | `propagate_values()`, `parameter_sweep()`, `sensitivity_analysis()` | Read (value propagation) |
| **Counterfactual** | `counterfactual()` | Read (validation) |
| **Graph Navigator** | `get_nodes()`, `get_edges()`, `get_upstream()`, `get_downstream()` | Read (exploration) |

---

## **12. SYNARA INTEGRATION**

### **12.1 One Engine, One Truth**

The graph service uses the existing Synara engine (`agents_api/synara/`) for all Bayesian operations. There is ONE Synara — not a graph-specific fork, not a copy, not a wrapper around a copy. The same `BeliefEngine` that updates posteriors in an investigation updates posteriors on graph edges.

### **12.2 What Synara Provides**

| Synara component | Graph service usage |
|-----------------|-------------------|
| `BeliefEngine.update_posteriors()` | Called by `add_evidence()` — Bayesian update on edge posterior |
| `BeliefEngine.propagate_belief()` | Called after evidence update — propagates changes through graph |
| `BeliefEngine.check_expansion()` | Called after evidence update — detects contradictions and gaps |
| `CausalGraph` | The in-memory representation loaded from persistent storage for computation |
| `ExpansionSignal` | Translated to LOOP-001 Signal when detected |

### **12.3 What Synara Needs (Extensions)**

The existing Synara engine needs these extensions to support graph-scale operation:

1. **Recency-weighted posterior aggregation** — Current Synara has NO temporal weighting. Each `update_posteriors()` call performs a single-step Bayesian update where the previous posterior becomes the new prior. This means evidence is implicitly ordered but not time-weighted — a DOE from two years ago has the same structural impact as one from last week, assuming equal strength. This is fine for short-lived investigation sessions but insufficient for the persistent graph where edges accumulate evidence over years. Extension: add a `recompute_posterior(evidence_stack, decay_function)` method that recalculates an edge's posterior from its full evidence stack with configurable recency decay (default: exponential, half-life 180 days, org-configurable via QMS Policy).

2. **Interaction-modulated propagation** — current `propagate_belief()` uses `link.strength` as constant. Needs to accept interaction terms that modulate strength based on sibling node states (§5.5). For uncalibrated interactions, modulation is identity (no change).

3. **Contradiction detection as edge-scoped expansion** — current `check_expansion()` detects "evidence doesn't fit ANY hypothesis" (whole-graph expansion). Contradiction detection is the same math scoped to a single edge: compute P(new_evidence | current_edge_posterior). If this likelihood falls below a configurable threshold (org-configurable via QMS Policy, default: 0.1 — same as expansion threshold), the evidence is "surprising" given current belief. That's the contradiction signal. Same Synara math, different scope. This is NOT about magnitude of disagreement — it's about the likelihood of the observation under the current model. Extension: add `check_edge_contradiction(edge_posterior, new_evidence) → ContradictionSignal | None`.

4. **Persistent state interface** — current Synara serializes to/from JSON dict. Needs adapter that loads from Django models and writes back, without changing the engine's internal representation.

These are extensions to the existing engine, not replacements. Investigation-level Synara continues to work exactly as it does today.

### **12.4 What We Do NOT Change**

- `syn/synara/` (Cortex, middleware, tenant isolation) — OS-level infrastructure, unrelated
- `core/synara.py` (lighter Bayesian engine on core models) — continues serving core views
- Synara's DSL, logic engine, LLM interface — remain available, potentially useful for expressing operating region constraints

---

## **13. PERSISTENCE MODEL**

### **13.1 Django Models**

The graph service persists to Django models in a new `graph` app (or within `loop/`). The models mirror the schemas in §3 and §4:

- `ProcessGraph` — container, one per org (FK → Tenant)
- `ProcessNode` — §3.3 schema
- `ProcessEdge` — §4.3 schema
- `EdgeEvidence` — §4.4 schema

### **13.2 Relationship to Existing Models**

The existing `core.models.graph` (KnowledgeGraph, Entity, Relationship) and `workbench.models.KnowledgeGraph` are **deprecated** by this standard. They are generic, unused in production, and lack the process-specific semantics required.

Migration path:
1. Build new models alongside existing ones
2. If any existing graph data exists in production, migrate it
3. Remove old models in a subsequent migration
4. Update references in tests and compliance checks

### **13.3 Synara State Bridge**

When performing Bayesian operations, the service:
1. Loads relevant `ProcessNode` and `ProcessEdge` records from the database
2. Constructs a `CausalGraph` (Synara kernel) in memory
3. Runs Synara operations (update, propagate, check expansion)
4. Writes results back to the database models

This bridge is unidirectional per operation — load, compute, persist. No long-lived in-memory graph state.

---

## **14. VALUE PROPAGATION AND PARAMETER EXPLORATION**

### **14.1 Two Operations on One Graph**

The graph supports two fundamentally different propagation operations:

| Operation | Question | Math | Trigger | Changes |
|-----------|----------|------|---------|---------|
| **Belief update** | "Given this evidence, how confident am I in this edge?" | P(h\|e) ∝ P(e\|h) × P(h) | New evidence arrives | Edge posteriors |
| **Value propagation** | "If I set input X to this value, what happens to output Y?" | Monte Carlo through calibrated f(x) + interaction terms | User moves a slider | Node value distributions |

Both use the same graph. Both use the same edges. But belief update changes the EDGES (how confident we are in the relationship), while value propagation changes the NODES (what values they take given inputs). Belief update is the learning mechanism. Value propagation is the exploration mechanism.

### **14.2 Value Propagation Engine**

When a user sets an input node to a specific value (or distribution), the engine propagates downstream:

```
For each downstream edge from the changed node:
    1. Look up edge effect_size, direction, CI
    2. Apply interaction terms — modulate effect based on current state of other nodes (§5.5)
    3. Sample from the effect distribution (Monte Carlo)
    4. Compute the resulting distribution on the target node
    5. If the target node has further downstream edges, recurse
```

**Monte Carlo parameters:**
- Default: 10,000 iterations (sufficient for stable distribution estimates)
- Configurable per-run for speed vs precision tradeoff
- Each iteration samples from: input distribution × effect size distribution × interaction uncertainty

**Output per node:** Mean, std, percentiles (5th, 25th, 50th, 75th, 95th), probability of exceeding spec limits if applicable.

### **14.3 What Happens at Uncalibrated Edges**

When propagation hits an uncalibrated edge (no evidence, FMEA assertion only):

1. **Propagation continues** using a wide prior distribution (high uncertainty)
2. **The edge is visually flagged** — "this prediction passes through an uncalibrated relationship"
3. **The output uncertainty widens dramatically** — reflecting that we're speculating, not predicting
4. **The gap is surfaced** — "To narrow this prediction, calibrate edge X→Y with a DOE"

The system never silently produces confident predictions through uncalibrated edges. The uncertainty is honest.

### **14.4 Exploration Modes**

| Mode | What the user does | What they learn |
|------|-------------------|-----------------|
| **Parameter sweep** | Set one input to a range, hold others fixed | Response curve: how does output change as this input moves? |
| **Sensitivity analysis** | Ask "which inputs matter most for this output?" | Ranked influence: partial variance decomposition through the graph |
| **What-if** | Set multiple inputs to specific values | Joint prediction: "if we run at these settings, what do we expect?" |
| **Failure injection** | Force an input out of spec | Impact map: which outputs are affected and by how much? |
| **Counterfactual** | Compare model prediction to actuals post-change | Validation: "did our fix actually produce the effect the model predicted?" |

### **14.5 The Slider Interface**

The graph is the simulator. The user interface is a set of sliders and toggles that navigate the process manifold:

```
┌─────────────────────────────────────────────────────────────────────┐
│  PROCESS EXPLORER                                    [Reset] [Save] │
├─────────────────────┬───────────────────────────────────────────────┤
│                     │                                               │
│  INPUTS             │              GRAPH VIEW                       │
│  ─────────────      │                                               │
│  Zone 3 Temp        │     [temp]──(0.3±0.05)──→[viscosity]         │
│  ├ 180 ════●═ 260   │         │                    │                │
│  │     220°C        │         │                    ↓                │
│  │                  │    [humidity]──(?)──→[pellet moisture]        │
│  Injection Press    │                          │                    │
│  ├ 40 ═●════ 120    │                          ↓                   │
│  │   60 MPa         │              [viscosity]──→[short shots]     │
│  │                  │                   ↑           ⚠ 12.3%        │
│  Cycle Time         │              [interaction]                    │
│  ├ 15 ══●═══ 45     │                                              │
│  │    25s           │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│  │                  │                                               │
│  ENVIRONMENT        │  OUTPUTS              Current    Predicted    │
│  ─────────────      │  ─────────────────    ────────   ──────────  │
│  Humidity           │  Surface Finish       1.2 μm     1.4 ± 0.3  │
│  ├ 20 ════●═ 80     │  Short Shot Rate      8.1%       12.3% ▲    │
│  │     55% RH       │  Flash Rate           2.0%       1.8% ▼     │
│  │  ⚠ uncalibrated  │  Cycle Efficiency     91%        87% ▼      │
│  │                  │                                               │
│  ──────────────     │  ⚠ 2 edges uncalibrated — predictions wide  │
│  [Sweep] [What-If]  │  ⓘ Click any edge to see evidence stack     │
│  [Sensitivity]      │                                               │
└─────────────────────┴───────────────────────────────────────────────┘
```

**Key behaviors:**
- Moving a slider instantly recomputes downstream distributions
- Graph edges glow based on how much influence they're carrying in the current propagation
- Uncalibrated edges are dashed lines with warning indicators
- Stale edges are amber
- Contradicted edges are red
- Clicking any node opens its SPC chart, distribution, linked equipment
- Clicking any edge opens its evidence stack, posterior history, gap status
- The output panel shows current actuals vs model prediction with uncertainty bands

### **14.6 Confidence Propagation**

Every prediction carries a confidence indicator that degrades as it passes through the graph:

- **High confidence** (green): all edges in the path are calibrated with DOE evidence, n > 30, CI tight
- **Medium confidence** (amber): some edges have observational evidence only, or are approaching staleness
- **Low confidence** (red): path passes through uncalibrated or stale edges
- **Speculative** (dashed): path passes through FMEA assertions with zero evidence

The user always knows: "how much should I trust this prediction?" The answer is in the graph itself.

### **14.7 Integrity Requirement**

When simulation results depend on uncalibrated or stale edges, the system MUST surface this: "This prediction relies on edges with no empirical data." Simulations built on assertions are not predictions — they are structured speculation, and the user must know the difference. The system never hides uncertainty.

### **14.8 Service Interface (additions to §11)**

```python
class GraphService:
    # ... (existing methods from §11) ...

    # --- Value Propagation ---

    def propagate_values(graph_id, input_states, n_iterations=10000) → PropagationResult
        """Set input nodes to specified values/distributions and propagate
        downstream through calibrated edges with Monte Carlo sampling.
        Returns predicted distributions on all downstream nodes with
        confidence indicators and uncalibrated edge warnings."""

    def parameter_sweep(graph_id, node_id, value_range, fixed_states) → SweepResult
        """Sweep one input across a range, propagate each, return response curves."""

    def sensitivity_analysis(graph_id, output_node_id) → SensitivityResult
        """Rank all upstream inputs by influence on output (variance decomposition)."""

    def counterfactual(graph_id, input_states, actual_outputs) → CounterfactualResult
        """Compare model prediction to actual post-change observations.
        Returns prediction accuracy and identifies where model was wrong."""
```

---

## **15. UX: GRAPH-FIRST NAVIGATION MODEL**

### **15.1 The Graph Is Home**

The unified knowledge graph replaces the current tool-centric navigation model. Instead of navigating to separate tools (FMEA, SPC, investigations, training), the user navigates their process graph and accesses tools as contextual actions on graph elements.

**Current model (tool-centric):**
```
Dashboard → FMEA tool → (work) → back → SPC tool → (work) → back → Investigation → ...
```

**New model (graph-first):**
```
Process Graph → click node → SPC chart / distribution / linked equipment
             → click edge → evidence stack / posterior history / gap status
             → click failure mode → FMEA view (upstream causes, S/O/D from posteriors)
             → select subgraph → scope investigation
             → drag sliders → value propagation (§14)
             → red indicators → gaps, staleness, contradictions → investigate
```

The graph is the permanent context. Everything else is a lens or action on the graph.

### **15.2 Graph Views (Lenses)**

The same graph, filtered and rendered differently for different purposes:

| View | Filter | Renders | Who uses it |
|------|--------|---------|-------------|
| **Process Map** | All nodes, all edges | Full topology with health indicators | Engineers, quality managers |
| **FMEA View** | Nodes of type `failure_mode` + upstream causes | S/O/D scores from edge posteriors, RPN | FMEA team |
| **Gap View** | Only uncalibrated, stale, contradicted elements | Priority-ranked gaps with investigation links | Quality managers, CI leads |
| **Control View** | Nodes under SPC + their edges | Live control charts inline, alarm status | Operators, SPC analysts |
| **Audit View** | Same as Process Map, read-only, filtered by ISO clause | Evidence stacks, provenance chains | Auditors (via Auditor Portal) |
| **Explorer View** | User-selected subgraph + sliders | Value propagation, what-if, sensitivity | Engineers, process owners |

Switching views does not leave the graph. It re-renders the same structure with different emphasis. All views share the same URL base — the view selector is a toggle, not a navigation event.

### **15.3 Contextual Actions**

Every graph element supports right-click (or long-press) contextual actions:

**On a Node:**
- View distribution / SPC chart
- View linked FMEA rows
- View linked equipment / measurement systems
- Start investigation scoped to this node
- Add to current investigation subgraph
- Edit operational definition
- Flag for review

**On an Edge:**
- View evidence stack (full history with timestamps)
- View posterior trend (how belief has changed over time)
- Add evidence (manual observation)
- View linked DOEs / investigations
- View interaction terms
- Mark as stale (manual override)
- Start investigation scoped to this edge

**On a Gap Indicator:**
- View gap details (what's missing, why it matters)
- Create investigation to resolve
- Create commitment (LOOP-001 §3.3) to calibrate
- Dismiss with justification (creates audit record)

### **15.4 Entry Points**

Users enter the graph from multiple starting points:

| Entry point | How they get there | What they see |
|-------------|-------------------|---------------|
| **Loop Dashboard** | "My Process" link in nav | Full process map with today's signals/gaps highlighted |
| **Signal triage** | Click untriaged signal | Graph zoomed to the affected node/edge with context |
| **Investigation** | From investigation workspace | Scoped subgraph with investigation data overlaid |
| **FMEA** | From quality menu or FMIS | FMEA view lens (§15.2) |
| **SPC alarm** | From SPC notification | Graph zoomed to alarmed node, stale edges highlighted |
| **Auditor Portal** | External token link | Audit view lens (§15.2), read-only |
| **Process Explorer** | From graph toolbar | Explorer view with sliders (§14.5) |

### **15.5 Relationship to Existing Surfaces**

The graph-first model does NOT replace the Analysis Workbench, Plant Simulator, or individual tool UIs. Those remain as specialized work surfaces. The graph is the navigation hub that connects them:

- **Analysis Workbench** — where you run statistical analyses. Results flow back to the graph as evidence via `add_evidence()`.
- **Plant Simulator** — educational DES tool. Separate from the graph (simulates generic processes, not your calibrated process model).
- **Investigation Workspace** — still the three-pane investigation UI. But now scoped from the graph, and writeback goes to the graph.
- **Individual tools** (DOE designer, Gage R&R, etc.) — still standalone entry points. But results are linked to graph edges as evidence.

The graph is the map. The tools are the instruments. You look at the map to decide where to point the instrument. The instrument's readings update the map.

### **15.6 Progressive Disclosure**

A new org's graph starts empty. The system does not overwhelm with an empty canvas. Progressive entry:

1. **Day 1:** Org creates their first FMEA (FMIS). The system proposes a graph skeleton. "You've described 12 failure modes with 23 causes. Here's your process model — confirm or adjust."
2. **Week 1:** Graph has ~30 nodes, mostly uncalibrated edges (dashed lines). Gap view shows where to invest effort. The graph is a map of what you DON'T know.
3. **Month 1:** First DOEs complete. Some edges become solid (calibrated). SPC is running on key nodes. The graph starts to come alive — some areas are well-understood, others are still assertions.
4. **Month 6:** Graph has 50-100 nodes, mix of calibrated and uncalibrated. Process Explorer becomes useful — enough calibrated edges to make predictions meaningful. First contradictions emerge from SPC data, driving investigations.
5. **Year 1:** The graph IS the organization's process knowledge. New engineers onboard by exploring the graph. Auditors review the graph. Process changes are evaluated against the graph before implementation.

### **15.7 QMS NG Alignment**

This graph-first navigation model aligns with the QMS NG Master Plan's strategic thesis:

> "Svend's position: Only platform combining 200+ statistical tests, real-time SPC, DOE, FMEA, RCA, A3, Hoshin, Bayesian evidence, AI critique, knowledge graph — all connected."

The keyword is "all connected." The graph IS the connection layer. Without it, these are 12 separate tools that happen to be in the same product. With it, they are instruments reading from and writing to a shared model of reality. That integration depth is the moat.

The QMS NG Ultimate State pyramid (Strategic → Operational → Intelligence → Practitioner) maps onto the graph:

| QMS NG Layer | Graph Role |
|-------------|-----------|
| **Strategic** (Hoshin, Risk Register, Management Review) | Graph-level gap reports and CI Readiness Score inform strategic priorities |
| **Operational** (SPC→CAPA→RCA closed loop) | The loop operates ON the graph — signals from SPC, investigations scope subgraphs, CAPA reports reference graph evidence |
| **Intelligence** (pattern detection, recurrence, predictive trending) | Cross-graph analytics — which edges keep going stale? Which failure modes share upstream causes? Where are the systemic gaps? |
| **Practitioner** (Harada, CI Readiness, archetypes) | Individual learning targets informed by graph gaps — "your process has 14 uncalibrated edges in Zone 3; your development plan includes DOE training" |

---

## **16. DESIGN CONSTRAINTS**

### **16.1 The Model Is Not Truth**

The graph represents the organization's current understanding. It is deliberately lossy. Missing edges are visible. Uncalibrated edges are flagged. The system's value comes from making ignorance explicit, not from achieving completeness.

### **16.2 Humans Decide**

The system does not auto-resolve contradictions, auto-update edges from SPC, auto-add nodes from investigations, or auto-seed from FMEA without confirmation. Every structural or calibration change to the graph passes through human judgment. The system computes, surfaces, and proposes. Humans confirm.

### **16.3 Evidence Is Never Discarded**

Old evidence records are never deleted. They may carry less weight (recency decay), but they remain in the stack. An auditor can trace any edge's current posterior back to every observation that produced it.

### **16.4 Provenance Is Mandatory**

Every edge must have a provenance. Every evidence record must have a source. "We believe this because..." is answerable for any element of the graph. If it isn't, the element is flagged as a gap.

---

## **APPENDIX A: GLOSSARY**

| Term | Definition |
|------|-----------|
| **Graph** | The topology — nodes and edges representing process entities and their relationships |
| **Model** | The calibration — quantified belief state on the graph's edges, from evidence |
| **Manifold** | The response surface described by the graph under specific operating conditions |
| **Fold** | A change in the manifold's shape caused by interaction terms when node states change |
| **Edge posterior** | The organization's current degree of belief in an edge, given all evidence |
| **Evidence stack** | The ordered collection of evidence records supporting an edge |
| **Gap** | Any element of the graph where knowledge is incomplete — uncalibrated, stale, missing, or contradicted |
| **Expansion signal** | Synara's detection that the causal surface is incomplete — evidence fits no existing edge |
| **Writeback** | The process of propagating investigation findings back to the parent graph |
| **Seed** | The initial creation of graph structure from FMIS rows |

## **APPENDIX B: MIGRATION FROM LOOP-001 §9**

LOOP-001 §9 concepts map to GRAPH-001 as follows:

| LOOP-001 §9 concept | GRAPH-001 location |
|---------------------|-------------------|
| Entity types (controllable_input, etc.) | §3.2 Node Types (expanded taxonomy) |
| Relationship types (causal, correlational, confounded) | §4.2 Edge Relationship Types |
| Building the model from FMEA | §7 FMIS Seeding |
| Progressive enrichment table | §7.2 Progressive Enrichment |
| Gap exposure | §6 Gap Exposure (expanded taxonomy with priority model) |
| Model integrity / no drift tolerated | §9 Staleness Detection + §10 Contradiction Handling |
| Investigation as process model subset | §8 Investigation Subgraph |
| Simulation | §14 Value Propagation and Parameter Exploration |

LOOP-001 §9 should be updated to reference GRAPH-001 as the authoritative specification.

## **APPENDIX C: QMS NG ALIGNMENT**

GRAPH-001 implements the "all connected" thesis from the QMS NG Master Plan. The following QMS NG surfaces are directly affected:

| QMS NG Surface | Change |
|---------------|--------|
| **Loop Dashboard** | Adds "My Process" entry point to graph navigator |
| **Investigation Workspace** | Scoped from graph subgraph selection, writeback to graph on conclusion |
| **FMEA/FMIS** | Becomes a graph view (§15.2), not a separate data structure |
| **SPC** | Node-linked; alarms create staleness flags on graph edges |
| **DOE** | Results flow to graph as edge evidence via `add_evidence()` |
| **Auditor Portal** | Graph audit view lens, read-only, filtered by ISO clause |
| **Process Confirmations** | Evidence on graph edges confirming expected behavior |
| **Training** | Development plans informed by graph gaps (§15.7) |
| **Hoshin / Strategic** | CI Readiness and gap reports inform strategic priority setting |
| **Process Explorer** | NEW surface — slider-based value propagation through calibrated graph (§14.5) |

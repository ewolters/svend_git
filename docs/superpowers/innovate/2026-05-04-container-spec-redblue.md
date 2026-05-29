# Red/Blue Analysis: SVEND Container System Design Spec

**Date:** 2026-05-04
**Technique:** Red Team / Blue Team
**Attack Dimensions:** Data model gaps, Integration fragility, Operational risk

---

## Blue Team Defense

### Problem Solved
Four models solving four identified gaps: orchestration (WorkflowDef/Instance/Step), process state (Measure/Datapoint), learning (GovernanceRule/Decision), contracts (Contract). All inherit battle-tested SynaraEntity infrastructure.

### Robustness Claims
- Idempotency keys on WorkflowStep prevent duplicate execution
- MAX_DEPTH=3 prevents recursive workflow runaway
- GovernanceDecision immutability prevents history rewriting
- Confidence floor at 0.01 prevents division-by-zero
- Nullable WorkflowDef FK enables ad-hoc workflows
- Version lineage on both WorkflowDef (parent FK) and Contract (supersedes FK)
- All patterns align with established codebase conventions (JSON fields, pull-only, Bayesian)

### Key Assumptions
1. 11 days achievable because hard parts exist (SynaraEntity, audit, compliance, 200+ analyses)
2. JSON fields for flexible schemas follow established QMS patterns
3. Pull-only contracts are settled architecture (push broke prod 4 days)
4. Laplace smoothing correct for sparse binary governance data
5. MAX_DEPTH=3 covers realistic manufacturing workflow chains

### Competitive Moat
Compounding data: more usage → better PCL → better suggestions → more corrections → better governance → fewer corrections needed. Data moat, not code moat.

---

## Red Team Attacks

### CRITICAL (High Likelihood × High Impact)

**1. GovernanceDecision Immutability Paradox**
save() prevents updates after creation. But outcome starts 'pending' and needs to transition to 'good'/'bad'. No clean mutation path exists. QuerySet.update() bypasses save() and violates the immutability contract. Logical contradiction in the spec.

**2. PCL Already Exists with Incompatible Schema**
PCL already deployed at ~/kjerne/pcl/ with applied migration 0001_initial. Different field names (slug vs name, range_min vs realistic_min), different confidence formula (log2-based vs sqrt-based), Datapoint extends SynaraImmutableLog not SynaraEntity, has decay support and cached aggregates. Proposal redesigns production models without acknowledging them.

**3. Second Competing Workflow Engine**
QMS has WorkflowTemplate/WorkflowPhase/WorkflowTransition (graph-based with gate conditions). Proposal adds WorkflowDef/Instance/Step (linear sequence-based). Two workflow engines in one codebase.

**4. WorkflowStep CASCADE Destroys Provenance**
on_delete=CASCADE means deleting instance removes all steps. Steps ARE the provenance trail. Soft-deleted instances become invisible to default SynaraEntityManager, making step queries fail.

**5. syn/synara is Middleware, Not Django App**
No apps.py, not in INSTALLED_APPS. Settings.py explicitly says: "syn.synara NOT registered (no models)." Models placed here won't be discovered by Django's migration system.

### HIGH (Medium-High Likelihood × High Impact)

**6. Router Wrapping Breaks Analyses on Exception**
If WorkflowStep logging throws (DB timeout, idempotency constraint violation), previously-working analysis returns 500. The current router handles handler errors gracefully — wrapping adds an unhandled failure mode.

**7. CHG-001 Overhead Makes 4-Day Build ~5.5 Days**
~8 CRs needed. 30-60 min process per CR (create, submit, risk assess, approve, in_progress). 4-8 hours of pure process overhead in a "4 day build."

**8. CASCADE Bypasses SynaraImmutableLog.delete()**
Django CASCADE uses QuerySet.delete() internally, bypassing model delete() methods. Immutable records become deletable via parent FK cascade. Pre-existing bug amplified by new CASCADE FKs.

### MEDIUM

**9. Contract String-Based App References**
No FK validation on source_app/consumer_app. Typos create phantom contracts. PROTECT on supersedes prevents cleanup.

**10. Heuristic.trigger Schemaless JSON**
No schema = unimplementable matching. Either O(n) full scan or deferred forever. Design needs refinement before implementation.

---

## Adjudication

### Covered by Defense
None of the red team findings are addressed by the blue defense. The blue team built a strong case for the overall approach but did not anticipate any of these specific failure modes.

### Genuine Vulnerabilities (ranked by decision relevance)

| # | Finding | Verdict | Action Required |
|---|---------|---------|-----------------|
| **2** | **PCL already exists** | **CRITICAL** | Proposal must work with existing pcl/ schema or explicitly plan migration. The spec's PCL models are incompatible with what's deployed. |
| **1** | **GovernanceDecision immutability paradox** | **HIGH** | Redesign: either create separate GovernanceOutcome record, or don't make GovernanceDecision immutable (make only the initial decision immutable, outcome is a mutable field). |
| **3** | **Second workflow engine** | **HIGH** | Justify the distinction (analysis orchestration vs QMS flow) or extend existing QMS engine. |
| **5** | **syn/synara not a Django app** | **HIGH** | Models need a different home. New `governance/` app or add to `syn.core`. |
| **4** | **CASCADE destroys provenance** | **MEDIUM-HIGH** | Change to PROTECT on Instance→Step FK. Ensure step queries work on soft-deleted instances. |
| **7** | **CHG-001 overhead** | **MEDIUM** | Add 1-1.5 days to timeline for process compliance. |
| **6** | **Router wrapping** | **LOW-MEDIUM** | Wrap logging in try/except. Logging failure never blocks analysis result. |
| **8** | **CASCADE bypasses immutable delete()** | **MEDIUM** | Use PROTECT on all FKs to SynaraImmutableLog subclasses. Pre-existing bug. |
| **9** | **String-based contracts** | **LOW** | Add choices list or validation. Not a blocker. |
| **10** | **Schemaless heuristic trigger** | **LOW** | Flag for design refinement. Heuristic is last thing built. |

### What Red Team Missed

- **Existing PCL confidence formula is MORE sophisticated** than proposed. Deployed version uses source-type-weighted log2 curves with per-type plateau values, decay support, and cached effective N. Proposal downgrades this.
- **Existing PCL has formula-based calculated measures** using `[slug]` reference syntax. Proposal uses `components M2M` — different resolution mechanism.

### Single Most Important Finding

**PCL Already Exists.** The proposal designs a key component (PCL) that is already deployed with applied migrations, a different schema, a more sophisticated confidence formula, and more features. Everything downstream (governance confidence, workflow integration, the data moat thesis) depends on PCL. The spec must be reconciled with the deployed pcl/ app before implementation begins.

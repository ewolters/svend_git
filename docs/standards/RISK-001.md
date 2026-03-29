**RISK-001: RISK REGISTRY STANDARD**

**Version:** 1.0
**Status:** APPROVED
**Date:** 2026-03-06
**Author:** Eric + Claude (Systems Architect)
**Compliance:**
- SOC 2 CC3.2 (Risk Assessment), CC9.1 (Risk Mitigation)
- ISO 9001:2015 §6.1 (Actions to Address Risks and Opportunities)
- AS9100D §6.1 (Actions to Address Risks and Opportunities)
**Related Standards:**
- GRAPH-001 ≥ 0.1 (Unified Knowledge Graph — risk entries seed graph nodes and edges)

---

## **1. SCOPE AND PURPOSE**

<!-- rule: mandatory -->

This standard defines the persistent risk registry for Kjerne/Svend. Risks are tracked beyond individual ChangeRequests, enabling trending, mitigation verification, and compliance reporting.

## **2. TERMINOLOGY**

| Term | Definition |
|------|-----------|
| **RPN** | Risk Priority Number — Likelihood × Severity × Detectability (1-125) |
| **High Risk** | RPN > 60 — requires documented mitigation plan |
| **Medium Risk** | 20 < RPN ≤ 60 — should have mitigation plan |
| **Low Risk** | RPN ≤ 20 — accepted or monitored |

## **3. RISK SCORING**

<!-- rule: mandatory -->

Each risk entry uses FMEA-style scoring across three dimensions (1-5 each):

| Dimension | 1 | 3 | 5 |
|-----------|---|---|---|
| **Likelihood** | Rare | Possible | Almost certain |
| **Severity** | Negligible | Moderate | Catastrophic |
| **Detectability** | Easily detected | Detectable with effort | Undetectable |

**RPN = Likelihood × Severity × Detectability** (max 125).

> **Bayesian compatibility (LOOP-001 §8.6):** When FMIS uses Bayesian scoring (`svend_bayesian` or `svend_full` policy), the risk registry accepts posterior-mean-derived scores. The system maps the Beta posterior mean to the 1-5 integer scale for this registry. Both the point estimate (for deterministic RPN) and the full posterior credible interval are available via API. The RPN displayed is computed from the point estimates; the uncertainty is available for risk-informed decision making.

<!-- assert: RiskEntry model auto-computes RPN on save | check=risk-rpn-auto -->
<!-- impl: syn/audit/models.py:RiskEntry.save -->
<!-- test: syn.audit.tests.test_risk_registry.RiskEntryModelTest.test_rpn_auto_computed -->

<!-- assert: RiskEntry.risk_level property returns correct tier | check=risk-level-tiers -->
<!-- impl: syn/audit/models.py:RiskEntry.risk_level -->
<!-- test: syn.audit.tests.test_risk_registry.RiskEntryModelTest.test_risk_level_high -->
<!-- test: syn.audit.tests.test_risk_registry.RiskEntryModelTest.test_risk_level_medium -->
<!-- test: syn.audit.tests.test_risk_registry.RiskEntryModelTest.test_risk_level_low -->

## **4. LIFECYCLE**

<!-- rule: mandatory -->

```
identified → mitigating → mitigated → closed
     ↓
  accepted (risk acknowledged, no further action)
```

- **identified**: Risk discovered, not yet addressed
- **mitigating**: Active mitigation work in progress
- **accepted**: Risk acknowledged, no mitigation planned (requires justification)
- **mitigated**: Mitigation implemented, awaiting verification
- **closed**: Risk resolved or no longer applicable

## **5. COMPLIANCE CHECK**

<!-- rule: mandatory -->

The `risk_registry` compliance check runs on the Thursday rotation and verifies:
- No open high-risk items (RPN > 60) lack a mitigation plan → **fail**
- No high-risk items remain in "identified" status without action → **warning**
- Empty registry or all mitigated → **pass**

<!-- assert: risk_registry compliance check flags high-RPN items without mitigation | check=risk-check-enforcement -->
<!-- impl: syn/audit/compliance.py:check_risk_registry -->
<!-- test: syn.audit.tests.test_risk_registry.RiskRegistryCheckTest.test_high_risk_no_mitigation_fails -->
<!-- test: syn.audit.tests.test_risk_registry.RiskRegistryCheckTest.test_high_risk_with_mitigation_passes -->
<!-- test: syn.audit.tests.test_risk_registry.RiskRegistryCheckTest.test_empty_registry_passes -->

<!-- assert: risk_registry check is registered in compliance system | check=risk-check-registered -->
<!-- impl: syn/audit/compliance.py:ALL_CHECKS -->
<!-- test: syn.audit.tests.test_risk_registry.RiskRegistryRegistrationTest.test_check_registered -->
<!-- test: syn.audit.tests.test_risk_registry.RiskRegistryRegistrationTest.test_check_in_rotation -->

## **6. CR LINKAGE**

RiskEntry has an optional `source_cr` FK to ChangeRequest. When a CR's RiskAssessment scores ≥ 3 overall, a corresponding RiskEntry should be created to persist the risk beyond the CR lifecycle.

---

## **REVISION HISTORY**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-06 | Eric + Claude | Initial release — RiskEntry model, FMEA scoring, compliance check (FEAT-090) |

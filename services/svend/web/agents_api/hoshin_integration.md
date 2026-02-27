# Hoshin ↔ VSM Integration Analysis

## Date: 2026-02-22

## Reference: github.com/ewolters/neptune-hoshin (the model)

---

## Neptune vs Svend Diff Summary

### Closed (Svend matches or exceeds Neptune):
- Sites model (same fields)
- Site access control (Svend has roles; Neptune has zero auth)
- Projects with savings tracking (HoshinProject + monthly_actuals)
- Calculation engine (Svend has 8 methods vs Neptune's 7 — added energy)
- Custom formula engine (safe AST eval with {{field}} syntax)
- Kaizen charter (JSONField vs separate model — same data)
- Action items + dependencies (Svend adds cross-source: hoshin/a3/rca/fmea)
- Bowler chart, dashboard, calendar view
- Baseline data tracking
- VSM integration (Svend has full pipeline + Monte Carlo — Neptune has none)

### Gaps — Neptune has, Svend missing:

| # | Feature | Priority | Notes |
|---|---------|----------|-------|
| 1 | **AFE (Authorization for Expenditure)** | High | 3-level approval chain (Site→BU→Corp), digital signatures, ROI/payback, cost thresholds. Real enterprise CI need. |
| 2 | **Project Plan (PDCA document)** | Medium | background, current_conditions, objectives, countermeasures, results. Svend has A3 but not linked to Hoshin. |
| 3 | **Gantt chart for action items** | Medium | Drag-drop timeline with dependency arrows. Currently table-only in Svend. |
| 4 | **Project cloning (Carryover to next FY)** | Medium | Clone project structure, reset monthly data. Annual planning workflow. |
| 5 | **Tags** | Low | Color-coded project tags with dashboard filtering. |
| 6 | **Comments with @mentions** | Low | Per-project thread. Svend has chat system. |
| 7 | **Person model** | Low | Svend uses text fields for owners — works fine. |

### Svend-only (no Neptune equivalent):
- VSM→Hoshin with Monte Carlo savings estimation
- Bayesian hypothesis tracking (Synara) on kaizen bursts
- 200+ statistical analyses (DSW) feeding evidence
- Bidirectional calculator ↔ VSM sync
- Multi-tenant enterprise auth

---

## Integration Architecture

### Direction of data flow (intentional):

```
Calculators ──export──▶ VSM ──proposals──▶ Hoshin
     ▲                    ▲                    │
     │                    │                    │
     └── import from ─────┘                    │
                                               ▼
                                          Action Items
                                          Monthly Actuals
                                          Savings Tracking
```

### Design decisions:
- **Hoshin does NOT write back to VSM.** VSMs are team artifacts that must be manually constructed each cycle.
- **VSMs are not auto-populated for following years.** The mapping process IS the value — it forces the team to re-examine the value stream.
- **VSM→Hoshin is one-way:** kaizen bursts generate proposals, proposals become projects. The project tracks source_vsm + source_burst_id for traceability only.

---

## Open: VSM Lifecycle

**Current VSM statuses:** current, future, archived

**Questions to resolve:**
- How do VSM lifecycles align with Hoshin fiscal years?
- When does a "future state" VSM become "current state"?
- What triggers archival?
- Should VSMs be versioned within a fiscal year?
- How does the current/future pairing work across annual planning cycles?

See discussion below.

---

## Open: Remaining integration gaps

| # | Gap | Description | Decision |
|---|-----|-------------|----------|
| A | Calculator → Hoshin monthly actuals | OEE, CT, scrap from calculators could feed Hoshin operational data | TBD |
| B | Hoshin ↔ A3 linking | Neptune's ProjectPlan ≈ A3. Svend A3 exists but isn't linked to HoshinProject | TBD |

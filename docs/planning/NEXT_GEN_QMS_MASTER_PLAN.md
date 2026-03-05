# Next-Generation QMS Master Plan

**ID:** `plan-nextgen-qms-001`
**Version:** 1.0
**Date:** 2026-03-04
**Authors:** Eric + Claude (Systems Architect)
**Status:** DRAFT — Strategic Planning
**Competitive Analysis:** `docs/reference/competitive_landscape_qms_spc.md`
**Requirements Checklist:** Derived from ISO 9001:2015, IATF 16949, AS9100, ISO 13485, FDA 21 CFR Part 11
**Existing Roadmap:** `docs/roadmaps/2026-03-03_QMS_roadmap_.md`

---

## 1. Strategic Thesis

The QMS software market is split into three silos that don't talk to each other:

1. **QMS platforms** (ETQ, MasterControl, Veeva, Arena) — compliance workflows, document control, CAPA. Zero statistical capability. $12K-$200K+/yr.
2. **Statistical platforms** (Minitab, JMP) — deep analytical engines. Zero workflow, zero compliance. $1,300-$2,600/yr.
3. **SPC platforms** (InfinityQS, WinSPC, DataLyzer) — real-time process monitoring. No CAPA, no document control, no audit. $1,000-$20K/yr.

Quality engineers today switch between 3-4 tools to complete a single investigation, manually transferring findings between systems. The "closed loop" from statistical signal to corrective action is universally manual.

**Svend's position:** The only platform that combines a 200+ test statistical engine, real-time SPC, DOE, FMEA, RCA, A3, VSM, Hoshin Kanri, Bayesian evidence integration, AI critique, and a knowledge graph — in one codebase, with one data model, at $299/mo.

**The goal:** Own the world-class standard for integrated QMS + statistical platforms. Not by building everything — by building the right things, connected in ways no one else can.

---

## 2. Competitive Gap Matrix

What Svend has vs. what the market offers, organized by ISO 9001 clause:

| ISO 9001 Clause | Arena | ETQ | MasterControl | Minitab | Svend Today | Svend Target |
|----------------|-------|-----|---------------|---------|-------------|--------------|
| §4 Context (SWOT, stakeholders) | - | - | - | - | - | Phase 6 |
| §5 Leadership (policy, roles) | Partial | Partial | Partial | - | - | Phase 6 |
| §6 Planning (risk, objectives) | Partial | Yes | Partial | - | FMEA + Hoshin | Complete (Phase 3) |
| §7.1 Resources | - | - | - | - | QMS-002 (Employee, ResourceCommitment) | Complete |
| §7.2 Competence (training) | Yes | Yes | Yes | - | Learn module | Phase 5 |
| §7.5 Doc Control | Yes | Yes | Yes | - | - | Phase 4 |
| §8.3 Design & Development | Yes | Partial | Partial | DOE | DOE + ExperimentDesign | Phase 5 |
| §8.4 Supplier Quality | Partial | Yes | Partial | - | SupplierRecord (basic) | Phase 4 |
| §8.5 Production (SPC) | - | - | - | - | SPC module (full) | Complete |
| §8.7 Nonconformity | Yes | Yes | Yes | - | NCR (NonconformanceRecord) | Phase 3 |
| §9.1 Monitoring (stats) | - | - | - | **Yes** | **200+ tests, SPC, DOE** | Complete |
| §9.2 Internal Audit | Yes | Yes | Yes | - | ComplianceReport (basic) | Phase 5 |
| §9.3 Management Review | Yes | Yes | Yes | - | ManagementReview (basic) | Phase 3 |
| §10.2 CAPA | Yes | Yes | Yes | - | Embedded in NCR | Phase 3 |
| AI/ML | Shallow | Shallow | Minimal | AutoML | AI critique, Synara, Bayesian | **Native advantage** |
| Closed-Loop Integration | Partial | Partial | Partial | None | Phase 1 done | **Core differentiator** |
| Knowledge Graph | None | None | None | None | **Yes** | **Unique** |
| Bayesian Evidence | None | None | None | None | **Yes** | **Unique** |
| Price | $89/user/mo | $25K+/yr | $25K+/yr | $1,851/yr | **$299/mo flat** | Same |

**Svend's unfair advantages** (things no competitor has or can easily build):
1. Statistical engine + QMS in one data model (not stitched by API)
2. Bayesian evidence integration (hypothesis-driven quality)
3. Knowledge graph connecting all quality artifacts
4. AI-native architecture (not retrofitted)
5. Transparent pricing at 1/10th the cost
6. Closed-loop: FMEA → RCA → A3 → Hoshin → Evidence → KG

---

## 3. Feature Roadmap — Phased Build

### Phase 3: Enterprise Foundation [March-April 2026]

**Theme:** Notifications, signatures, CAPA lifecycle, management review

| ID | Feature | ISO Clause | Priority |
|----|---------|-----------|----------|
| E3-001 | Notification system (bell icon + SSE) | Cross-cutting | Critical |
| E3-002 | Email notification with ActionToken response | Cross-cutting | Critical |
| E3-003 | ElectronicSignature (CFR Part 11, SynaraImmutableLog) | §7.5.3 | Critical |
| E3-004 | CAPA as standalone model (extract from NCR) | §10.2 | Critical |
| E3-005 | CAPA lifecycle: NCR → containment → RCA → corrective → verify → close | §10.2 | Critical |
| E3-006 | CAPA → RCA module bridge (auto-populate from NCR data) | §10.2 | High |
| E3-007 | Management Review template system (customizable sections) | §9.3 | High |
| E3-008 | Management Review auto-populate (aggregate QMS metrics) | §9.3 | High |
| E3-009 | QMSAttachment (artifact uploads on NCR/CAPA/FMEA/RCA/A3) | §7.5 | High |
| E3-010 | NCR trending + Pareto analysis | §8.7 | Medium |
| E3-011 | Cost of poor quality (CoPQ) tracking per NCR | §9.1 | Medium |
| E3-012 | Recurrence detection (flag repeat root causes across CAPAs) | §10.2 | Medium |

**Standard:** SIG-001 (Electronic Signatures), extend QMS-001 §10-§12
**CR type:** `feature` + `migration` (new models: Notification, ElectronicSignature, CAPA, QMSAttachment, ManagementReviewTemplate)

### Phase 4: Document Control & Supplier Quality [April-May 2026]

**Theme:** The two biggest gaps vs. Arena/ETQ/MasterControl

| ID | Feature | ISO Clause | Priority |
|----|---------|-----------|----------|
| E4-001 | Controlled document register (version control, approval workflow) | §7.5 | Critical |
| E4-002 | Document change notice (DCN) lifecycle | §7.5 | Critical |
| E4-003 | Document review scheduling + overdue alerts | §7.5 | High |
| E4-004 | External document management (standards, customer specs) | §7.5 | High |
| E4-005 | Master list of documents and records | §7.5 | High |
| E4-006 | Supplier scorecard (quality, delivery, cost metrics) | §8.4 | Critical |
| E4-007 | Supplier CAPA (issue + track + response via ActionToken) | §8.4 | Critical |
| E4-008 | Approved supplier list (ASL) with qualification status | §8.4 | High |
| E4-009 | Incoming inspection management | §8.4 | Medium |
| E4-010 | Supplier audit scheduling + tracking | §8.4 | Medium |
| E4-011 | Supplier portal via ActionToken (doc exchange, CAPA response) | §8.4 | Medium |
| E4-012 | Retention schedule management | §7.5 | Medium |

**Standard:** DOC-002 (Document Control Standard — distinct from DOC-001 which covers our internal documentation structure)
**Why this matters:** Document control is the #1 most-used QMS feature across every platform. It's table stakes for ISO 9001 certification. Without it, auditors can't verify §7.5 compliance.

### Phase 5: Training, Audit, & Design Controls [May-June 2026]

**Theme:** Close the remaining ISO 9001 gaps, add IATF/AS9100 basics

| ID | Feature | ISO Clause | Priority |
|----|---------|-----------|----------|
| E5-001 | Training matrix (role × competency) | §7.2 | Critical |
| E5-002 | Training → Learn module integration (assessments = competency proof) | §7.2 | Critical |
| E5-003 | Training effectiveness tracking (link training to defect rates) | §7.2 | High |
| E5-004 | CAPA → training trigger (root cause = training gap → auto-create requirement) | §7.2 | High |
| E5-005 | Internal audit program management (risk-based scheduling) | §9.2 | Critical |
| E5-006 | Audit checklist builder (clause-based, process-based) | §9.2 | High |
| E5-007 | Audit finding → CAPA bridge | §9.2 | High |
| E5-008 | Layered process audit (LPA) support | IATF §9.2 | Medium |
| E5-009 | Calibration management (gage register, intervals, alerts) | §7.1.5 | High |
| E5-010 | Calibration → Gage R&R link (existing SPC module) | §7.1.5 | High |
| E5-011 | Out-of-calibration impact assessment | §7.1.5 | Medium |
| E5-012 | Control plan management (prototype/pre-launch/production) | IATF §8.5 | Medium |
| E5-013 | FMEA → control plan auto-linkage | IATF §8.5 | Medium |
| E5-014 | Requirements traceability matrix | §8.3 | Medium |

**Standards:** TRN-001 (Training Management), AUD-002 (Internal Audit), CAL-001 (Calibration)

### Phase 6: Intelligence Layer [June-July 2026]

**Theme:** AI capabilities no competitor has

| ID | Feature | Priority |
|----|---------|----------|
| E6-001 | SPC alarm → auto-NCR creation (with process data context) | Critical |
| E6-002 | AI-assisted root cause suggestion (historical pattern matching) | Critical |
| E6-003 | Cross-FMEA pattern detection (systemic failure mode identification) | High |
| E6-004 | Predictive risk trending (which failure modes are increasing?) | High |
| E6-005 | Natural language QMS query ("Show me overdue CAPAs for supplier X") | High |
| E6-006 | Automated trending that triggers action (statistical significance detection) | High |
| E6-007 | Management review auto-narrative (AI-generated executive summary) | Medium |
| E6-008 | Complaint → reportability determination assist (FDA MDR, EU Vigilance) | Medium |
| E6-009 | Change impact analysis (auto-detect downstream document/FMEA/control plan effects) | Medium |
| E6-010 | Audit readiness scoring ("How ready are we for audit?") | Medium |
| E6-011 | Supplier risk prediction (based on scorecard trends + industry signals) | Low |
| E6-012 | Cost of quality automation (aggregate prevention/appraisal/failure costs) | Low |

**This is the moat.** Every competitor is bolting shallow AI onto legacy systems. Svend has AI-native architecture with Bayesian evidence, knowledge graph, and a statistical engine that can actually power the predictions. The data model supports it — theirs doesn't.

### Phase 7: Industry-Specific Extensions [Q3 2026]

**Theme:** IATF 16949 (automotive), AS9100 (aerospace), ISO 13485 (medical device)

| ID | Feature | Standard | Priority |
|----|---------|----------|----------|
| E7-001 | APQP phase-gate management | IATF 16949 | High |
| E7-002 | PPAP submission management (18 elements) | IATF 16949 | High |
| E7-003 | AIAG-VDA harmonized FMEA (7-step, AP tables) | IATF 16949 | High (partially done) |
| E7-004 | First article inspection (AS9102 Forms 1-3) | AS9100 | Medium |
| E7-005 | Configuration management | AS9100 | Medium |
| E7-006 | Counterfeit parts prevention workflow | AS9100 | Medium |
| E7-007 | Design controls (DHF/DMR/DHR management) | ISO 13485 | Medium |
| E7-008 | Complaint handling with reportability determination | ISO 13485/FDA | Medium |
| E7-009 | Field corrective action / recall management | ISO 13485/FDA | Medium |
| E7-010 | Customer-specific requirements (CSR) database per OEM | IATF 16949 | Low |

### Phase 8: DSW Statistical Calibration [March 2026]

**Theme:** Treat statistical analysis functions as measurement devices — calibrate against known reference data, rotate daily, detect drift

This is an internal engineering initiative, not a customer-facing feature phase. The DSW engine has 200+ statistical analysis functions producing p-values, effect sizes, Bayes factors, control limits, and capability indices. If any of those functions silently drifts (dependency update, refactor side-effect, numerical edge case), every downstream decision is compromised. The calibration system treats each analysis function like a gage: feed it reference data with analytically known correct answers, verify outputs within tolerance, flag drift.

| ID | Feature | Standard | Priority |
|----|---------|----------|----------|
| E8-001 | Calibration reference pool — 17 cases across 6 categories (inference, Bayesian, SPC, reliability, ML, simulation) | STAT-001 §15 | Critical |
| E8-002 | Calibration runner with date-seeded daily rotation (8 of 17 cases per run) | STAT-001 §15 | Critical |
| E8-003 | `check_statistical_calibration` compliance check — Thursday rotation, DriftViolation on failure | CMP-001 §6 | Critical |
| E8-004 | Dashboard calibration section — 4 KPI cards + per-case result table (green/red) | CMP-001 §6 | High |
| E8-005 | CAL enforcement type in DriftViolation model | CHG-001 §8 | High |
| E8-006 | Symbol-level impl hooks on existing STAT-001 assertions (replace file-level) | STAT-001 §4-§12 | Medium |
| E8-007 | Calibration test suite — 5 tests covering pool, runner, reproducibility, known-null, drift | TST-001 §9 | High |

**Standard:** STAT-001 ≥ 1.1 (§15 Statistical Calibration — already drafted)
**CR type:** `enhancement` (new compliance infrastructure, no production logic changes)
**Dependencies:** None — this is self-contained internal infrastructure
**Why now:** Symbol coverage audit revealed 539 ungoverned symbols (53,868 LOC) in agents_api. Calibration is the highest-trust way to govern statistical functions without wrapping them in bureaucratic tests.

---

### Phase 9: DSW Output Standardization [March-April 2026]

**Theme:** Lock down the output contract for all 211 DSW analyses — education, charts, narratives, evidence grades, what-if interactivity

The DSW engine has 211 statistical analyses but outputs are wildly inconsistent. Education tabs exist on 28/211 (Bayesian only). Bayesian shadows and evidence grades exist on 8/211. Charts use mixed colors, heights, legend placement. Narratives cover ~95/211. What-if interactivity exists on 8/211. The frontend doesn't even render the education blocks that are generated. This initiative standardizes everything.

| ID | Feature | Standard | Priority |
|----|---------|----------|----------|
| E9-001 | Analysis registry — 211 entries with metadata (category, effect_type, shadow_type, chart_types, what_if_tier) | DSW-001 | Critical |
| E9-002 | Post-processing pipeline — standardize_output() called from dispatch.py, enforces schema | DSW-001 | Critical |
| E9-003 | Chart standardization — apply_chart_defaults(), trace builders, enforce SVEND_COLORS/legend/height | DSW-001 | Critical |
| E9-004 | Frontend rendering — education (collapsible), narrative CSS, evidence grade badge, Bayesian panel, diagnostics | DSW-001/FE-001 | Critical |
| E9-005 | Education content — hand-written for all 211 analyses | DSW-001/STAT-001 | Critical |
| E9-006 | Bayesian shadow + evidence grade rollout — stats.py (~46 analyses with p-values) | STAT-001 §4-§8 | High |
| E9-007 | Shadow rollout — spc.py, ml.py, reliability.py, d_type.py | STAT-001 | High |
| E9-008 | What-if interactivity — unified schema, Tier 1 (20) + Tier 2 (15) analyses | DSW-001 | High |
| E9-009 | DSW-001 standard update + dsw_output_format compliance check | DSW-001/CMP-001 | High |

**Standard:** DSW-001 ≥ 2.0 (output format section to be added)
**CR type:** `enhancement` (systematic standardization, no new features)
**Dependencies:** None — self-contained internal infrastructure
**Why now:** Audit reveals the Bayesian module is gold standard (22/22 consistent). The other 189 analyses are a liability — inconsistent output degrades user trust and makes the platform look unfinished.

---

## 4. What Sets Svend Apart — The Integration Story

The real differentiator isn't any single feature. It's the data flow.

**The closed loop today (Phase 1 complete):**
```
FMEA (high RPN) → RCA (investigate) → A3 (countermeasures) → Hoshin (track)
     ↓                    ↓                    ↓                    ↓
  Evidence            Evidence            Evidence            Evidence
     ↓                    ↓                    ↓                    ↓
  ←←←←←←←←←←← Knowledge Graph ←←←←←←←←←←←←←
                         ↓
               Bayesian Updater
                         ↓
              Hypothesis Probability
```

**The closed loop after Phase 6:**
```
SPC (alarm) → NCR (auto-created) → CAPA (lifecycle) → RCA (investigate)
     ↑                                      ↓
 Calibration ←── Gage R&R               A3 (countermeasures)
                                            ↓
Control Plan ←── FMEA ←── Pattern Detection
     ↓                         ↓
  Training ←── Competency Gap Detection
     ↓
  Evidence → Knowledge Graph → Bayesian → Hypothesis
     ↓
  Management Review (auto-populated)
     ↓
  Hoshin (strategic execution) → Supplier CAPA → Supplier Scorecard
```

No competitor can build this. Arena doesn't have statistics. Minitab doesn't have workflows. ETQ doesn't have a knowledge graph. MasterControl doesn't have Bayesian evidence. They'd all need to rebuild from scratch, and they won't because they have legacy architectures and installed bases to protect.

---

## 5. Pricing Strategy

Current market:
| Platform | Annual Cost (50 users) | What You Get |
|----------|----------------------|--------------|
| ETQ Reliance | $50K-$100K+ | QMS workflows, no stats |
| MasterControl | $100K-$150K+ | QMS + training, no stats |
| Veeva Vault Quality | $60K-$120K+ | QMS + supplier portal, no stats |
| Minitab (separate) | $92K+ | Stats only, no QMS |
| Arena + Minitab | $140K+ | PLM-QMS + stats (separate tools) |
| **Svend Enterprise** | **$3,588/yr** | **QMS + stats + SPC + DOE + AI + KG** |

The pricing is a strategic weapon. At $299/mo, Svend is an impulse purchase for a quality manager. At $100K/yr, ETQ is a 6-month procurement cycle. We don't need to compete on features alone — we compete on total cost of ownership and time-to-value.

---

## 6. Standards Required

| Standard | Scope | Phase |
|----------|-------|-------|
| **SIG-001** (NEW) | Electronic signatures, CFR Part 11 compliance | Phase 3 |
| **DOC-002** (NEW) | Document control (controlled documents, DCN, retention) | Phase 4 |
| **TRN-001** (NEW) | Training management (matrix, effectiveness, competency) | Phase 5 |
| **AUD-002** (NEW) | Internal audit management (program, checklist, findings) | Phase 5 |
| **CAL-001** (NEW) | Calibration management (register, intervals, impact assessment) | Phase 5 |
| **STAT-001** (UPDATE) | §15 Statistical Calibration — reference pool, rotation, drift detection | Phase 8 |
| **CMP-001** (UPDATE) | Add statistical_calibration check to registry + Thursday rotation | Phase 8 |
| QMS-001 (UPDATE) | Add CAPA lifecycle, notification triggers, signature requirements | Phase 3 |
| QMS-002 (UPDATE) | Extend ActionToken for supplier/signature use cases | Phase 3 |
| SEC-001 (UPDATE) | SSE endpoint rules, CFR auth requirements | Phase 3 |
| DAT-001 (UPDATE) | New model documentation per phase | Each phase |

---

## 7. Success Metrics

| Metric | Today | Phase 3 | Phase 5 | Phase 7 |
|--------|-------|---------|---------|---------|
| ISO 9001 clause coverage | ~45% | ~60% | ~85% | ~95% |
| Modules connected (closed loop) | 5/5 | 8/8 | 12/12 | 15/15 |
| Statistical procedures | 200+ | 200+ | 200+ | 200+ |
| QMS-001 assertions passing | ~85% | ~90% | ~95% | 100% |
| Competitor feature parity (vs ETQ) | ~40% | ~55% | ~75% | ~90% |
| Competitor feature parity (vs Minitab) | ~80% | ~80% | ~85% | ~90% |
| Unique capabilities (no competitor has) | 5 | 7 | 9 | 12 |

---

## 8. Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| Scope creep — building too many features before depth | High | High | Each phase ships independently. MVP per feature, iterate. |
| CFR Part 11 compliance gap | Critical | Medium | Dedicated standard (SIG-001) + security review per CHG-001 |
| Document control is a massive feature surface | High | High | Start with controlled register + DCN. Skip full-text search and watermarking initially. |
| Industry-specific extensions dilute focus | Medium | Medium | Phase 7 is last. Core QMS + statistics first. |
| AI features over-promise | Medium | Medium | Ship only when the data pipeline supports it. No demos without production backing. |
| Frontend complexity (vanilla JS) | High | Medium | Consider template components. Keep interactions server-rendered where possible. |

---

## 9. Competitive Intelligence Sources

- `docs/reference/competitive_landscape_qms_spc.md` — Full analysis (Arena, ETQ, MasterControl, Greenlight Guru, Qualio, Veeva, Intelex, SAP QM, TrackWise, Minitab, JMP, InfinityQS, WinSPC, DataLyzer, Tulip, ComplianceQuest)
- Gartner Magic Quadrant for QMS Software (2026 inaugural) — ComplianceQuest and Siemens named Leaders
- G2 reviews: consistent complaints about search, UX, customization complexity, pricing opacity across all QMS platforms
- Key user pain points: management review data aggregation is manual everywhere, SPC-to-CAPA is manual everywhere, training effectiveness unmeasured everywhere

---

*This is a living document. Update after each phase. The goal is not feature parity with any single competitor — it's integration depth that none of them can match, at a price point that makes them irrelevant.*

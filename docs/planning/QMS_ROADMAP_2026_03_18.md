# QMS Roadmap — Updated 2026-03-18

**Source:** Product audit (QMS_AUDIT_2026_03_18.md) + existing master plan (NEXT_GEN_QMS_MASTER_PLAN.md)
**Scope:** All QMS modules through ultimate state
**Status:** Active — replaces previous QMS roadmap dates

---

## Current State

The QMS module has solid bones: 7-tab SPA (iso.html, 2430 lines), state machines on all core records, field-level audit trail (QMSFieldChange), FMEA↔hypothesis evidence bridge, RCA AI critique. But usage data shows most modules have 0-1 records — the system is built but barely exercised, and an ISO expert audit found 13 gaps.

**What works well:**
- FMEA with S/O/D, RPN, AIAG-VDA AP, evidence linking to Bayesian hypotheses
- RCA with AI critique (blame detection, circular logic, premature stops)
- NCR/CAPA/Document/Supplier state machines with transition gates
- Management review auto-snapshot
- Full multi-tenant permission model (ORG-001)

**What's missing or broken:** See QMS_AUDIT_2026_03_18.md for full details.

---

## Ultimate State: The Autonomous Quality System

Svend's QMS end state is not "ISO 9001 compliance software." It's a **self-monitoring quality operating system** where:

1. **Every signal has a response path.** SPC alarm → NCR → CAPA → corrective action → verification → effectiveness check. No manual handoffs. No switching tools. No copy-paste between systems.

2. **Every decision has evidence.** FMEA severity scores connect to hypothesis probabilities. RCA conclusions link to statistical analyses. CAPA effectiveness is verified by SPC data, not opinion. The knowledge graph connects every artifact.

3. **The system gets smarter.** Pattern detection across FMEAs identifies systemic failure modes. Recurrence detection flags repeat root causes. Risk predictions use scorecard trends. The AI doesn't just critique — it anticipates.

4. **Compliance is a byproduct, not a burden.** Audit readiness is continuous, not a quarterly scramble. The registrar sees a compliance dashboard with hash-chained evidence, not a binder of PDFs. Every clause is covered by live data, not static documents.

5. **Personal development feeds organizational improvement.** Harada practitioner archetypes correlate with investigation quality. CI Readiness scores predict kaizen sustainment. The platform develops the person AND the process.

**The moat:** No competitor can replicate this because it requires a statistical engine, Bayesian reasoning, knowledge graph, AI critique, and QMS workflows in one data model. ETQ/MasterControl can't add statistics. Minitab/JMP can't add compliance. Svend has both, connected.

---

## Phase A: Quick Wins [This week]

Tiny fixes that close real gaps with minimal code.

| # | Finding | Effort | What to do |
|---|---------|--------|------------|
| A1 | DES-4: NCR verification without corrective action | 1 hour | Add gate in `can_transition('verification')`: require `corrective_action` or `capa_report` |
| A2 | BUG-1: FMEA revised score validation | 1 hour | Add pre-save check in `fmea_views.py` — return 400 if partial revised scores |
| A3 | ISO-4: Management review completion gates | 2 hours | Add gate in review status transition: require `attendees` non-empty, `outputs` non-empty |
| A4 | DES-1: Supplier re-qualification path | 30 min | Add `disqualified → pending` transition requiring `notes` |
| A5 | ISO-5: Audit finding → NCR warning | 2 hours | Dashboard query: major findings without linked NCRs → surface as warning card |

**Exit criteria:** All 5 done, no new compliance failures.

---

## Phase B: Closed Loop [1-2 weeks]

Make the "closed-loop QMS" marketing claim real.

| # | Finding | Effort | What to do |
|---|---------|--------|------------|
| B1 | INT-1: SPC → NCR bridge | 3 days | New endpoint: `POST /api/spc/raise-ncr/` — takes chart_id, rule_violated, measurement. Creates NCR with source=`process`, pre-populated description, linked DSW result. Add "Raise NCR" button to SPC control chart UI when rule violation detected. |
| B2 | INT-2: FMEA → CAPA promote | 1 day | New endpoint: `POST /api/fmea/<id>/rows/<row_id>/promote-capa/` — creates CAPAReport with failure mode, root cause, severity mapping. Similar pattern to existing `promote-action`. |
| B3 | DES-2: Training ↔ document version trigger | 1 day | On ControlledDocument approval (version bump), query TrainingRequirements linked to that document. If version mismatch, create notification + dashboard warning. |

**Exit criteria:** User can go from SPC alarm → NCR → CAPA → RCA → verified corrective action without leaving the platform. Training gets flagged when procedures change.

---

## Phase C: Missing ISO Registers [2-3 weeks]

Close the gaps that registrars will cite.

| # | Finding | Effort | What to do |
|---|---------|--------|------------|
| C1 | ISO-1: Customer complaint register | 3 days | New model: `CustomerComplaint` — source (phone/email/web/field), product, date_received, date_acknowledged, assigned_to, status (open/investigating/resolved/closed), root_cause, resolution, customer_satisfied (bool), linked_ncr, linked_capa. New tab in iso.html or sub-tab under NCR Tracker. Wire as CAPA source. |
| C2 | ISO-2: Risk register | 3 days | New model: `Risk` — description, category (operational/quality/compliance/strategic/safety), likelihood (1-5), impact (1-5), risk_score (computed), risk_level (low/medium/high/critical), owner, mitigation_actions (JSONField), review_date, status (identified/mitigating/accepted/closed). New tab in iso.html. Link to FMEA for product/process risks, standalone for organizational risks. Surface in management review auto-snapshot. |
| C3 | ISO-3: Calibration equipment register | 3 days | New model: `MeasurementEquipment` — name, asset_id, serial_number, type, location, site, calibration_interval_months, last_calibration_date, next_calibration_due, calibration_provider, status (in_service/out_of_service/due/overdue/out_of_calibration), linked_gage_studies (M2M to DSWResult where analysis_type contains 'gage'). New tab or sub-tab under existing structure. Dashboard alerts for overdue calibrations. |

**Exit criteria:** Registrar can ask "Show me your complaint register / risk register / calibration register" and get a live system response, not a blank stare.

---

## Phase D: Wire Electronic Signatures [1 week]

The model exists (ElectronicSignature, hash-chained, 21 CFR Part 11 compliant). Zero records. Wire it in.

| # | What | Where |
|---|------|-------|
| D1 | Document approval | ControlledDocument `draft → approved` transition creates signature record |
| D2 | NCR closure | NonconformanceRecord `verification → closed` requires approver signature |
| D3 | CAPA closure | CAPAReport `verification → closed` requires signature |
| D4 | Management review sign-off | ManagementReview `in_progress → complete` captures attendee signatures |
| D5 | Signature verification UI | "Signed by X at Y" badge on records + verify button |

**Gating:** Optional for Team tier. Required for Enterprise. This is the FDA/aerospace differentiator.

**Exit criteria:** E-signatures appear on at least 4 record types. Verification endpoint confirms integrity.

---

## Phase E: Intelligence Layer [Ongoing]

From master plan Phase 6 — the moat no one can copy.

| # | What | Depends on |
|---|------|------------|
| E1 | SPC alarm → auto-NCR creation with process context | Phase B (B1 manual version first) |
| E2 | Cross-FMEA pattern detection (systemic failure modes) | Existing — endpoint exists, needs data |
| E3 | Recurrence detection across CAPAs | Phase C (C1 customer complaints feed this) |
| E4 | AI audit readiness scoring | Phases C + D (needs complete data) |
| E5 | Hoshin ↔ QMS KPI integration | INT-3 from audit |
| E6 | Change impact analysis (doc revision → downstream FMEA/training/control plan) | Phase B (B3), Phase D |
| E7 | Predictive risk trending | Phase C (C2 risk register) |
| E8 | Management review auto-narrative | Phase C + existing auto-snapshot |

---

## Phase F: Industry Extensions [Q3 2026]

From master plan Phase 7 — IATF 16949, AS9100, ISO 13485.

| Standard | Key additions |
|----------|--------------|
| **IATF 16949** (automotive) | APQP phase gates, PPAP 18-element management, control plan ↔ FMEA linkage, customer-specific requirements DB, layered process audits |
| **AS9100** (aerospace) | First article inspection (AS9102), configuration management, counterfeit parts prevention, key characteristics |
| **ISO 13485** (medical device) | Design controls (DHF/DMR/DHR), complaint ↔ reportability (MDR/EU Vigilance), field corrective action / recall management, risk management per ISO 14971 |

**Gating:** Enterprise tier only. These are the features that justify $299/mo against Arena at $25K+/yr.

---

## Phase G: Harada Integration [Q2-Q3 2026]

The personal development layer — see `project_harada_spec.md` for locked schema.

| # | What | Depends on |
|---|------|------------|
| G1 | Harada 36 questionnaire in notebook sidebar | Notebook UI |
| G2 | CI Readiness instrument (12 dimensions, mixed format) | Question drafting |
| G3 | K-prototypes clustering pipeline | G1 + G2 data collection |
| G4 | Practitioner archetype assignment + retake tracking | G3 |
| G5 | Goal cascade → 64-window → routine tracker → daily diary | G1 |
| G6 | Hansei Kai reflection on goal completion | G5 |
| G7 | Archetype ↔ notebook outcomes correlation | G4 + notebook data |

**Strategic link:** ILSSI students take Harada 36 + CI Readiness on enrollment. Longitudinal data shows archetype migration over training. This becomes a research publication AND a sales tool for training partners.

---

## Ultimate State Diagram

```
                          ┌─────────────────────────────────────────────┐
                          │           STRATEGIC LAYER                    │
                          │  Hoshin Kanri X-Matrix ← QMS KPIs          │
                          │  Risk Register ← FMEA + Organizational     │
                          │  Management Review ← Auto-snapshot + AI     │
                          └────────────────────┬────────────────────────┘
                                               │
                          ┌────────────────────▼────────────────────────┐
                          │           OPERATIONAL LAYER                  │
                          │                                              │
                          │  SPC alarm ──→ NCR ──→ CAPA ──→ RCA        │
                          │       ↑          │        │        │        │
                          │  Calibration     │    E-Signature  │        │
                          │  Register        │        │        │        │
                          │       ↑          ▼        ▼        ▼        │
                          │  Gage R&R    Customer  A3 Report  Evidence  │
                          │              Complaint     │        │       │
                          │                  │         ▼        ▼       │
                          │              Training  Hoshin    Knowledge  │
                          │              Trigger   Project    Graph     │
                          │                  │                  │       │
                          │                  ▼                  ▼       │
                          │           Document Control   Bayesian       │
                          │           (version → retrain) Updater      │
                          │                                    │       │
                          │                                    ▼       │
                          │                             Hypothesis     │
                          │                             Probability    │
                          └────────────────────┬───────────────────────┘
                                               │
                          ┌────────────────────▼────────────────────────┐
                          │           INTELLIGENCE LAYER                 │
                          │                                              │
                          │  Pattern detection (cross-FMEA, cross-RCA)  │
                          │  Recurrence detection (repeat root causes)  │
                          │  Predictive risk trending                    │
                          │  Audit readiness scoring                     │
                          │  Change impact propagation                   │
                          │  AI critique (RCA, FMEA, A3)                │
                          │  Management review auto-narrative            │
                          └────────────────────┬───────────────────────┘
                                               │
                          ┌────────────────────▼────────────────────────┐
                          │           PRACTITIONER LAYER                 │
                          │                                              │
                          │  Harada 36 + CI Readiness → Archetypes     │
                          │  Goal cascade → 64-window → Routines       │
                          │  Daily diary → Hansei Kai → New cycle      │
                          │  Archetype migration ↔ improvement outcomes │
                          └─────────────────────────────────────────────┘
```

---

## Competitive Position at Ultimate State

| Capability | ETQ ($25K+/yr) | MasterControl ($25K+/yr) | Minitab ($2.6K/yr) | Svend ($299/mo) |
|-----------|----------------|--------------------------|--------------------|--------------------|
| NCR/CAPA lifecycle | Yes | Yes | No | Yes |
| Document control | Yes | Yes | No | Yes |
| Training matrix | Yes | Yes | No | Yes |
| Internal audit | Yes | Yes | No | Yes |
| Supplier quality | Partial | Partial | No | Yes |
| Management review | Partial | Partial | No | Yes + auto-snapshot |
| Customer complaints | Yes | Yes | No | Yes |
| Risk register | Yes | Partial | No | Yes |
| Calibration register | Partial | Partial | No | Yes + Gage R&R link |
| E-signatures (CFR 11) | Yes | Yes | No | Yes |
| Statistical engine (200+) | No | No | Yes | Yes |
| SPC control charts | No | No | Yes | Yes + Bayesian |
| DOE | No | No | Yes | Yes |
| Gage R&R | No | No | Yes | Yes + Bayesian |
| FMEA ↔ evidence | No | No | No | **Yes (unique)** |
| RCA AI critique | No | No | No | **Yes (unique)** |
| Bayesian SPC | No | No | No | **Yes (unique)** |
| Knowledge graph | No | No | No | **Yes (unique)** |
| Hypothesis tracking | No | No | No | **Yes (unique)** |
| Practitioner archetypes | No | No | No | **Yes (unique)** |
| Hoshin Kanri + X-matrix | No | No | No | **Yes (unique)** |
| SPC → NCR closed loop | No | No | No | **Yes (unique)** |
| Purchasing power pricing | No | No | No | **Yes (unique)** |
| AI audit readiness | No | No | No | **Yes (unique)** |

At ultimate state, Svend has everything ETQ has + everything Minitab has + 12 capabilities no one has, at 1/7th the cost of ETQ alone.

---

## Cross-References

- Audit findings: `docs/planning/QMS_AUDIT_2026_03_18.md`
- Master plan (Phases 3-9): `docs/planning/NEXT_GEN_QMS_MASTER_PLAN.md`
- Enterprise integration plan: `docs/planning/QMS_ENTERPRISE_INTEGRATION_PLAN.md`
- Harada spec: Memory → `project_harada_spec.md`
- ISO standard mapping: `docs/standards/QMS-001.md`

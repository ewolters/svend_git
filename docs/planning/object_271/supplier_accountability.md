# Object 271 — Supplier Accountability System

**Date:** 2026-03-29
**Status:** DESIGN — spec before code
**Authors:** Eric + S2 (Conservative)
**ISO:** 8.4 (External Providers), 8.6 (Release of Products), 10.2 (Nonconformity & Corrective Action)

---

## What This Is

Three interlocking systems that hold suppliers to the same analytical rigor we apply internally:

1. **Supplier Claim Management** — structured lifecycle from defect detection to verified resolution
2. **Supplier Response Portal** — external-facing, token-authenticated. Forces structured CAPA from supplier. Analyzes response quality against historical patterns.
3. **CoA Portal + SPC Integration** — suppliers upload Certificates of Analysis. System extracts data, maps to incoming inspection, feeds SPC, links to knowledge graph.

These chain: Claim → Response Portal → Verification → CoA confirms fix → SPC validates ongoing compliance → Graph edges update.

---

## 1. Supplier Claim Management

### The Problem Today

NCR exists (NonconformanceRecord with FK to SupplierRecord). But an NCR is our internal record. It doesn't:
- Formally notify the supplier
- Track what we asked them to do
- Track what they responded with
- Verify their fix worked
- Track the commercial impact (credits, replacements, chargebacks)
- Detect when suppliers give the same non-answer repeatedly

### Model: SupplierClaim

```
SupplierClaim
    id: UUID
    tenant: FK → Tenant
    supplier: FK → SupplierRecord
    ncr: FK → NonconformanceRecord (nullable — claims can exist without NCR)

    # What happened
    claim_type: quality_defect | delivery | documentation | specification | packaging | contamination
    title: CharField(300)
    description: TextField

    # Affected product
    part_number: CharField(100)
    lot_number: CharField(100)
    quantity_affected: IntegerField
    quantity_rejected: IntegerField

    # Evidence
    defect_description: TextField  — what specifically was wrong
    inspection_method: TextField   — how was it detected
    evidence_photos: JSON [UserFile UUIDs]

    # Financial
    cost_of_quality: DecimalField  — internal cost (scrap, rework, sort, downtime)
    credit_requested: DecimalField — what we're asking the supplier to pay
    credit_received: DecimalField  — what we actually got (nullable until resolved)
    disposition: returned | scrapped | reworked | use_as_is | sorted

    # Lifecycle
    status: draft → filed → acknowledged → responded → under_review → verified → closed | rejected | escalated
    filed_at: DateTimeField
    response_due_date: DateField  — deadline for supplier to respond

    # Portal
    portal_token: CharField(64, unique) — for supplier access
    portal_expires_at: DateTimeField

    # Graph integration
    linked_process_node_ids: JSON — which ProcessNodes this affects

    # Provenance
    created_by: FK → User
    created_at, updated_at
```

### Status Lifecycle

```
draft → filed → acknowledged → responded → under_review → verified → closed
                                    ↓              ↓
                                rejected      escalated
                                    ↓
                               re-responded
```

- **draft:** Internal preparation. Not yet visible to supplier.
- **filed:** Portal link generated. Supplier notified (email). Clock starts on response_due_date.
- **acknowledged:** Supplier has accessed the portal and confirmed receipt.
- **responded:** Supplier has submitted their CAPA response. Response quality auto-evaluated.
- **under_review:** Our quality team is reviewing the response. May accept, reject, or request revision.
- **verified:** Corrective action implemented. Next shipment/audit confirms fix.
- **closed:** Claim resolved. Credit applied. Supplier score updated.
- **rejected:** Supplier response rejected. They must re-respond.
- **escalated:** Response insufficient after multiple rejections. Triggers supplier status review (suspension consideration).

### Transition Rules

| From | To | Requires |
|------|-----|----------|
| draft | filed | portal_token generated, response_due_date set |
| filed | acknowledged | Supplier accesses portal (auto-transition) |
| acknowledged/filed | responded | SupplierResponse created via portal |
| responded | under_review | Reviewer assigned |
| under_review | verified | Verification evidence |
| under_review | rejected | reviewer_notes explaining why |
| rejected | responded | New SupplierResponse via portal |
| verified | closed | Credit reconciled OR waived |
| under_review | escalated | Multiple rejections OR response_due exceeded |

### Escalation Logic

- If response_due_date passes without response: auto-flag, notify quality manager
- If 2 rejections on same claim: auto-escalate, suggest supplier status review
- If claim involves safety/critical defect: require verification before close (no waive)

---

## 2. Supplier Response Portal

### External-Facing Surface

Same architecture as Auditor Portal:
- Token-authenticated URL: `/supplier-claim/<token>/`
- No login required — token IS the auth
- Standalone HTML (not extends base_app.html — supplier doesn't see SVEND internals)
- Read-only claim details + structured response form

### What the Supplier Sees

```
┌─────────────────────────────────────────────────────────┐
│ [Your Logo]           SUPPLIER QUALITY CLAIM            │
│                       Claim #SC-2026-042                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ CLAIM DETAILS                                           │
│ ─────────────                                           │
│ Type: Quality Defect                                    │
│ Part: M6-HEX-SS-304    Lot: 2026-L0847                │
│ Qty Affected: 500      Qty Rejected: 47                │
│                                                         │
│ Defect: Bolt torque spec 8.0 ± 0.5 Nm.                │
│         47 units measured at 6.2-6.8 Nm (below LSL).   │
│         Detected at incoming inspection via torque      │
│         wrench calibrated 2026-03-01.                   │
│                                                         │
│ [Photos: 3 attached]                                    │
│                                                         │
│ Response Due: 2026-04-05 (8 days remaining)            │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ YOUR RESPONSE                                           │
│ ─────────────                                           │
│                                                         │
│ Root Cause Category:                                    │
│ [dropdown: Method | Material | Machine | Man |          │
│            Measurement | Environment]                   │
│                                                         │
│ Root Cause Description: *                               │
│ [textarea — min 50 chars, no "N/A" or "TBD" allowed]  │
│                                                         │
│ Corrective Action: *                                    │
│ [textarea — what you are doing to fix THIS shipment]   │
│                                                         │
│ Preventive Action: *                                    │
│ [textarea — what you are doing to prevent recurrence]  │
│                                                         │
│ Implementation Date: *                                  │
│ [date picker]                                           │
│                                                         │
│ Evidence:                                               │
│ [file upload — photos, test reports, process changes]  │
│                                                         │
│ [Submit Response]                                       │
│                                                         │
│ ─────────────────────────────────────────────────────── │
│ Previous responses on this claim: (if rejected/revised) │
│ Response #1 (2026-03-22) — REJECTED                    │
│   Root cause: "Operator error"                          │
│   Reviewer note: "Generic — which operator? Which step  │
│   in the process? What measurement confirmed this?"     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Input Validation (Anti-BS)

The form enforces rigor at input time:

- **Root cause description:** minimum 50 characters. Rejected words: "N/A", "TBD", "will investigate", "unknown" (alone — these are placeholders, not root causes)
- **Corrective vs preventive:** must be different (can't paste the same text)
- **Implementation date:** must be in the future (you haven't fixed it yet if the claim was just filed)
- **Evidence:** required if root cause category is "method" or "machine" (show the change)

### Response Quality Analysis (Post-Submit)

When the supplier submits, the system evaluates automatically:

```python
def evaluate_response_quality(claim, response, supplier):
    """Score the response and detect patterns of weak responses."""

    flags = []
    score = 1.0  # Start at perfect

    # 1. Check for repeat root causes from this supplier
    previous_responses = SupplierResponse.objects.filter(
        claim__supplier=supplier
    ).exclude(claim=claim)

    prev_categories = [r.root_cause_category for r in previous_responses]
    category_freq = Counter(prev_categories)

    if category_freq.get(response.root_cause_category, 0) > len(previous_responses) * 0.4:
        flags.append(f"'{response.root_cause_category}' used in {category_freq[response.root_cause_category]} of {len(previous_responses)} previous claims")
        score -= 0.2

    # 2. Check for similar corrective action text
    for prev in previous_responses:
        similarity = text_similarity(response.corrective_action, prev.corrective_action)
        if similarity > 0.7:
            flags.append(f"Corrective action is {int(similarity*100)}% similar to response on claim {prev.claim.title[:40]}")
            score -= 0.3
            break

    # 3. Check if preventive action == corrective action
    if text_similarity(response.corrective_action, response.preventive_action) > 0.8:
        flags.append("Preventive action is essentially the same as corrective action")
        score -= 0.15

    # 4. Check root cause specificity (length, detail)
    if len(response.root_cause_description) < 100:
        flags.append("Root cause description is very brief")
        score -= 0.1

    return max(0, score), flags
```

The reviewer sees:
- Response quality score (0-1, color-coded)
- Flags with specific concerns
- Previous response history from this supplier
- Pattern indicators: "This supplier's last 4 responses all cited 'operator error'"

### Notification Chain

| Event | Who Gets Notified | How |
|-------|-------------------|-----|
| Claim filed | Supplier (email with portal link) | email_service |
| Response due in 48h | Quality manager | notification + email |
| Response overdue | Quality manager + procurement | notification + email |
| Response submitted | Quality reviewer | notification |
| Response rejected | Supplier (email with portal link to revise) | email_service |
| Claim verified | Supplier (email confirmation) | email_service |
| Claim escalated | Quality manager + procurement lead | notification + email |

---

## 3. CoA Portal + SPC Integration

### The Vision

Suppliers upload Certificates of Analysis (CoA/CoC) through the same portal infrastructure. The system:

1. **Receives the document** (PDF or structured data)
2. **Extracts measurements** (OCR for PDFs, or structured JSON/CSV)
3. **Maps to inspection parameters** (part_number + measurement_name → SPC chart)
4. **Feeds SPC** (incoming material data points added to control charts)
5. **Updates the graph** (material property nodes get evidence from CoA data)
6. **Validates compliance** (CoA values vs spec limits → auto-accept or flag)

### Model: SupplierCoA

```
SupplierCoA
    id: UUID
    tenant: FK → Tenant
    supplier: FK → SupplierRecord

    # Document
    document: FK → UserFile (the uploaded PDF/file)
    coa_number: CharField(100)
    lot_number: CharField(100)
    part_number: CharField(100)
    date_issued: DateField

    # Extracted data
    measurements: JSON [
        {
            parameter: "Tensile Strength",
            value: 485.2,
            unit: "MPa",
            spec_min: 450,
            spec_max: 550,
            method: "ASTM A370",
            conforming: true
        },
        ...
    ]
    extraction_method: manual | structured | ocr
    extraction_confidence: float (0-1, for OCR)

    # Compliance
    all_conforming: bool
    nonconforming_parameters: JSON [parameter names that failed]

    # SPC integration
    spc_data_ingested: bool — have measurements been pushed to SPC charts?
    spc_ingestion_date: DateTimeField
    linked_process_node_ids: JSON — which material property nodes these map to

    # Status
    status: uploaded → reviewed → accepted → ingested | rejected
    reviewed_by: FK → User
    reviewed_at: DateTimeField

    created_at, updated_at
```

### SPC Feed

When a CoA is accepted and ingested:

```python
def ingest_coa_to_spc(coa):
    """Push CoA measurements into SPC charts for incoming material monitoring."""
    for measurement in coa.measurements:
        # Find the SPC chart for this parameter + part
        node = find_material_node(coa.part_number, measurement['parameter'])
        if not node or not node.linked_spc_chart:
            continue

        # Add data point to the chart's dataset
        add_spc_datapoint(
            chart_id=node.linked_spc_chart,
            value=measurement['value'],
            timestamp=coa.date_issued,
            source=f"CoA {coa.coa_number} lot {coa.lot_number}",
        )

        # Update node distribution
        GraphService.update_node_distribution(node.id, ...)

        # Add evidence to material → quality characteristic edges
        for edge in node.outgoing_edges.filter(relation_type='causal'):
            GraphService.add_evidence(edge.id, ...)
```

### Graph Integration

CoA data feeds the graph at multiple points:

- **Material property nodes** get updated distributions from CoA measurements
- **Material → quality characteristic edges** get evidence (CoA confirms material meets spec → edge strengthened)
- **Supplier → material edges** (if supplier performance nodes exist) get evidence
- **Nonconforming CoA** → flag material property node → staleness on downstream edges

### Implementation Phasing

| Phase | What | Effort | Depends On |
|-------|------|--------|------------|
| 1 | SupplierClaim model + lifecycle + API | 1 session | SupplierRecord (exists) |
| 2 | Supplier workbench template in Loop shell | 1 session | Phase 1 |
| 3 | Supplier Response Portal (external HTML) | 1 session | Phase 1 |
| 4 | Response quality analysis (Synara) | 0.5 session | Phase 3 |
| 5 | SupplierCoA model + manual entry | 1 session | SupplierRecord (exists) |
| 6 | CoA → SPC ingestion pipeline | 1 session | Phase 5 + SPC node linkage |
| 7 | CoA → Graph evidence pipeline | 0.5 session | Phase 5 + GraphService |

Phases 1-4 are the claim system (this CR).
Phases 5-7 are the CoA system (separate CR, after claims are validated).

---

## Design Decisions (Resolved 2026-03-29)

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| Q1 | Where does SupplierClaim live? | **loop/models.py** | It emerges from production issues — even if caught at incoming inspection, it's a risk that happened to be detected. That's Loop. |
| Q2 | Response quality analysis? | **Rules first, configurable** | Auditable, explainable. Portal behavior (what's required, how rules apply) configurable via QMS Policy. Synara enhancement later. |
| Q3 | CoA data extraction? | **Manual + CSV** | Most supplier systems can export CSV. OCR deferred — accuracy requirements too high for SPC data. |
| Q4 | Supplier email? | **Existing email_service** | NTF-001 compliant. Same infrastructure, external recipient. |
| Q5 | Portal branding? | **Tenant-branded** | Supplier sees their customer's logo. Professional. Uses Tenant.settings.branding from PDF export feature. |
| Q6 | CoA generation (outbound)? | **Deferred** | Most ERPs handle this. Inbound CoA processing is the higher-value problem. Same data model works for both — build inbound first, outbound becomes a PDF template from your own inspection data. Roadmap Phase 7+. |

### Rule Configurability (Q2 Detail)

Response quality rules are QMS Policy-configurable:
- `supplier.response_min_length`: minimum root cause description length (default: 50 chars)
- `supplier.response_blocked_phrases`: phrases that auto-flag (default: ["N/A", "TBD", "will investigate", "unknown"])
- `supplier.response_repeat_threshold`: % of previous responses using same root cause category before flagging (default: 40%)
- `supplier.response_similarity_threshold`: text similarity score above which corrective action is flagged as copy-paste (default: 0.7)
- `supplier.response_evidence_required_categories`: root cause categories that require file evidence (default: ["method", "machine"])
- `supplier.response_max_rejections_before_escalation`: auto-escalate after N rejections (default: 2)

---

## Impact on Existing Systems

| System | Impact |
|--------|--------|
| **SupplierRecord** | No model changes. Claims reference it via FK. Quality rating could auto-update from claim resolution. |
| **NCR** | No model changes. Claims reference it via FK. NCR view gets "Create Claim" button. |
| **CAPA** | No model changes. Claim responses are essentially supplier CAPAs. Could link SupplierResponse → CAPAReport if needed. |
| **Loop Signals** | Supplier claims create Signals when filed. Overdue responses create escalation Signals. |
| **Graph** | CoA data feeds material property nodes. Claim resolution feeds supplier → material edges. |
| **SPC** | CoA measurements feed incoming material SPC charts. |
| **Auditor Portal** | Supplier claim history and resolution rates become ISO 8.4 evidence. |

---

## What Makes This Different

Every QMS tool tracks supplier NCRs. Some let you email a PDF to the supplier. None of them:

1. **Force structured responses** — the portal form won't accept vague root causes
2. **Detect pattern responses** — "you said 'operator error' 4 times this year"
3. **Auto-evaluate response quality** — flags weak CAPAs before the reviewer opens them
4. **Feed CoA data into SPC** — incoming material is monitored with the same statistical rigor as your own process
5. **Update the knowledge graph** — supplier quality becomes process knowledge, not just a score

This is supplier accountability as a system, not a form.

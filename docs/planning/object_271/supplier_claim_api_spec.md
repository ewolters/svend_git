# Supplier Claim API — Session Handoff

**Date:** 2026-03-29
**For:** New session (backend)
**CR:** 391a1825 (in_progress)
**Models:** Already migrated in commit 1466299. See loop/models.py — SupplierClaim, SupplierResponse, ClaimVerification.

## What to Build

### Endpoints (add to loop/urls.py)

```
# Supplier Claims
GET    /api/loop/claims/                          → list claims (filter: supplier_id, status)
POST   /api/loop/claims/                          → create claim (draft)
GET    /api/loop/claims/<uuid>/                    → detail (include responses + verifications)
POST   /api/loop/claims/<uuid>/                    → actions: file, transition, add_verification
DELETE /api/loop/claims/<uuid>/                    → delete (draft only)

# Supplier Responses (via portal — but internal review also needs this)
POST   /api/loop/claims/<uuid>/respond/            → submit response (creates SupplierResponse)
POST   /api/loop/claims/<uuid>/review/             → accept/reject response with reviewer_notes

# Portal (token-authenticated, no login)
GET    /supplier-claim/<token>/                    → portal view (standalone HTML, handled by S2)
GET    /api/loop/portal/claim/<token>/data/        → claim data for portal
POST   /api/loop/portal/claim/<token>/respond/     → supplier submits response via portal
POST   /api/loop/portal/claim/<token>/acknowledge/ → supplier acknowledges receipt
```

### Serializer

```python
def _serialize_claim(claim, include_responses=False):
    data = {
        "id", "title", "description", "claim_type",
        "supplier_id", "supplier_name",
        "ncr_id",
        "part_number", "lot_number", "quantity_affected", "quantity_rejected",
        "defect_description", "inspection_method", "evidence_photos",
        "cost_of_quality", "credit_requested", "credit_received", "disposition",
        "status", "filed_at", "response_due_date",
        "portal_token" (only if user is staff),
        "portal_is_valid", "response_is_overdue",
        "linked_process_node_ids",
        "response_count", "verification_count",
        "created_by_id", "created_at", "updated_at",
    }
    if include_responses:
        data["responses"] = [_serialize_response(r) for r in claim.responses.all()]
        data["verifications"] = [_serialize_verification(v) for v in claim.verifications.all()]
    return data
```

### Lifecycle Actions (POST to /api/loop/claims/<uuid>/)

| action | From Status | To Status | Side Effects |
|--------|-------------|-----------|-------------|
| `file` | draft | filed | Generate portal_token, set filed_at, send email to supplier |
| `acknowledge` | filed | acknowledged | Auto-set when supplier accesses portal |
| `submit_for_review` | responded | under_review | Assign reviewer |
| `accept` | under_review | verified | Set accepted=True on latest response |
| `reject` | under_review | rejected | Set accepted=False, notify supplier to revise |
| `escalate` | under_review | escalated | Create Signal, consider supplier status change |
| `verify` | verified | closed | Require ClaimVerification record |
| `add_verification` | any (verified+) | — | Create ClaimVerification |

### Response Quality Rules (loop/response_quality.py — new file)

```python
def evaluate_response_quality(claim, response, supplier):
    """Return (score: float 0-1, flags: list[str])"""
    # 1. Check repeat root cause category (>40% of previous claims)
    # 2. Check text similarity of corrective_action vs previous responses (>0.7)
    # 3. Check if corrective == preventive (>0.8 similarity)
    # 4. Check root cause description length (<100 chars = flag)
    # 5. Check blocked phrases in root cause ("N/A", "TBD", etc.)
    # Thresholds from QMS Policy if configured, else defaults
```

### Email Notification (on file)

```python
from notifications.email_service import email_service

email_service.send(
    to=supplier.contact_email,
    subject=f"Quality Claim: {claim.title}",
    body_text=f"...\n\nRespond here: https://svend.ai/supplier-claim/{claim.portal_token}/",
    wrap_template=True,
)
```

### What NOT to Touch

- `loop_supplier.html` — S2 is building that
- `agents_api/models.py` (SupplierRecord) — no changes needed
- `agents_api/iso_views.py` — existing supplier CRUD stays, claims are separate

### Testing

- Create claim from NCR, file it, verify portal_token generated
- Submit response via portal endpoint, verify quality score computed
- Reject response, verify supplier can re-respond (revision increments)
- Verify after 2 rejections → escalation suggestion
- Accept response → verify → close lifecycle

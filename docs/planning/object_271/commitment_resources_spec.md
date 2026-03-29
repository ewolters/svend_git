# Commitment Resource Management — S1 Spec

**Date:** 2026-03-29
**Owner:** S1 (Innovator)
**Depends on:** Employee model (agents_api.models.Employee), ResourceCommitment pattern (agents_api.models.ResourceCommitment)

## What to Build

### CommitmentResource Model (loop/models.py)

```python
class CommitmentResource(models.Model):
    """Employee assignment to a Loop commitment with role and availability.

    Parallels Hoshin's ResourceCommitment but for Loop commitments instead
    of Hoshin projects. Same Employee model, same availability checking.
    """
    id = UUIDField(primary_key=True)
    commitment = ForeignKey(Commitment, CASCADE, related_name="resources")
    employee = ForeignKey("agents_api.Employee", CASCADE, related_name="loop_commitments")
    role = CharField(max_length=30, choices=ROLE_CHOICES)  # reuse Hoshin roles
    hours_needed = DecimalField(max_digits=5, decimal_places=1, null=True)
    start_date = DateField(null=True)  # nullable — not all commitments have date ranges
    end_date = DateField(null=True)
    status = CharField(choices=[requested, confirmed, declined], default="requested")
    requested_by = ForeignKey(User, SET_NULL, null=True)
    created_at = DateTimeField(auto_now_add=True)
```

### API Endpoints (loop/views.py, loop/urls.py)

- `GET /api/loop/commitments/<id>/resources/` — list assigned employees
- `POST /api/loop/commitments/<id>/resources/` — assign employee (sends notification)
- `POST /api/loop/commitments/<id>/resources/<resource_id>/` — update status (confirm/decline)
- `DELETE /api/loop/commitments/<id>/resources/<resource_id>/` — remove assignment

### Availability Check

Reuse `ResourceCommitment.check_availability()` pattern — check if employee has overlapping
commitments (both Hoshin and Loop) before confirming.

### Notification

On assignment: notify employee via `notifications.helpers.notify()` with type="assignment".

### What NOT to Touch

- `loop_commitments.html` — S2 is replacing prompt() with modals there
- `_serialize_commitment()` — S2 already added resource_needs. S1 should add a `resources` key
  alongside it (from CommitmentResource, not the JSONField)
- The JSONField `Commitment.resource_needs` stays for now (free-text resource descriptions).
  CommitmentResource is the structured version for employee assignments.

### Migration

New table only. No changes to existing tables. Safe to run alongside S2's work.

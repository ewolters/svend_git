# DEBT-001: Technical Debt Closure Process

Standard process for closing technical debt items tracked in `.kjerne/DEBT.md`.

---

## 1. Pick an Item

- Open `.kjerne/DEBT.md`
- Choose a P1 item (or the highest-priority unblocked item)
- Check for dependencies — some items must be done in order (e.g., model migration before Synara persistence)

## 2. Create a Branch (if needed)

For large changes, create a feature branch:
```bash
git checkout -b debt/short-description
```

For small, isolated fixes, work directly on `main`.

## 3. Document Before Touching Code

Before writing any code, add an entry to `log.md`:
```
### YYYY-MM-DD — [DEBT ITEM] Short description
**Debt item:** [SERVICE] Description from DEBT.md
**Plan:** What you intend to change and why
**Files to change:**
- `path/to/file` — what will change
```

This creates a record even if the change is interrupted or rolled back.

## 4. Make the Change

- Edit only what's needed. Don't refactor adjacent code.
- Follow existing patterns in the codebase (see CLAUDE.md for architecture).
- If touching a view, match the existing decorator pattern (`@csrf_exempt`, `@gated`, `@require_auth`).
- If adding a model field, create a Django migration.

## 5. Test

### Required before closing any item:

**Syntax check** — the server must start:
```bash
cd ~/kjerne/services/svend/web
python manage.py check
```

**Migration check** — if models changed:
```bash
python manage.py makemigrations --check --dry-run
python manage.py migrate
```

**Endpoint smoke test** — if a view changed:
```bash
# Replace URL and payload with the relevant endpoint
curl -s -X POST http://localhost:8000/api/dsw/analysis/ \
  -H "Content-Type: application/json" \
  -d '{"test": true}' | python -m json.tool
```

**Unit test** — if a test exists for the module:
```bash
python manage.py test agents_api.tests -v2
```

### Optional but recommended:

**Full regression** — run the full test suite:
```bash
python manage.py test -v2
```

**Manual verification** — open the browser and exercise the feature.

## 6. Update the Log

Complete the `log.md` entry:
```
**Files changed:**
- `path/to/file` — what actually changed (may differ from plan)
**Verification:** what you ran and the result
**Commit:** git hash
```

## 7. Update DEBT.md

Move the item from **Active Debt** to **Resolved**:
```
## Resolved
[SERVICE] Description | Added: DATE | Resolved: DATE | Commit: hash
```

Update `*Last reviewed: DATE*` at the bottom.

## 8. Commit

Commit message format:
```
Close DEBT [SERVICE]: short description

- Detail 1
- Detail 2

Refs: DEBT.md item text
```

## 9. Push

```bash
git push origin main
```

---

## Priority Guidelines

| Priority | Meaning | Turnaround |
|----------|---------|------------|
| P1 | Blocks the product vision or has data integrity risk | This sprint |
| P2 | Degrades functionality or developer experience | Next sprint |
| P3 | Nice to have, competitive gap, or cleanup | Backlog |

## Dependency Map (P1 Items)

```
                    ┌─────────────────────────┐
                    │ Problem model migration  │
                    │ (JSON blobs → FK)        │
                    └────────────┬────────────┘
                                 │ must complete first
                    ┌────────────┴────────────┐
                    │ Synara persistence       │
                    │ (in-memory → Django ORM) │
                    └─────────────────────────┘

    Independent of above (can be done in parallel):

    ┌──────────────────┐  ┌──────────────────────┐
    │ DSW ↔ Evidence   │  │ Experimenter ↔ Evid.  │
    │ (add problem_id) │  │ (extend problem_id)   │
    └──────────────────┘  └───────────────────────┘
```

DSW and Experimenter integration work against the *existing* `agents_api.Problem` model (which already has `add_evidence()`). Once the migration to `core.Project` is done, these integrations will be updated to use the new FK model. This avoids blocking integration work on the migration.

---

*Process created: 2026-02-06*

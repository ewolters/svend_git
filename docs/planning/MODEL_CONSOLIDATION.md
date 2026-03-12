# Model Consolidation Playbook

How we safely remove deprecated/duplicate models, one layer at a time.

## Pre-Flight

1. **CR first** — create ChangeRequest before touching code (CHG-001)
2. **Count rows** — `Model.objects.count()` in production shell
3. **Map all references** — grep for class name, imports, FKs, related_names
4. **Check FK population** — any model that FKs to the target: count non-null references
5. **Identify test coverage** — which tests reference the model?

## Rules

- **Zero rows only** — never delete a model with production data without a migration plan
- **One model family at a time** — don't batch unrelated removals
- **FK columns drop before model drop** — remove ForeignKey fields on other models first, then the model itself
- **Migration per step** — generate Django migration after each change, don't squash
- **Tests update WITH removal** — update/remove tests in the same commit, not after
- **Hook to standards** — if a standard references the model, update the standard

## Steps

### 1. Create CR
```
title: Remove deprecated <Model> from <app>
change_type: debt
affected_files: <list all files from grep>
```

### 2. Remove FK references on OTHER models
- Change `ForeignKey(DeprecatedModel, ...)` → remove the field
- `makemigrations` — generates ALTER TABLE DROP COLUMN

### 3. Remove the model class
- Delete class from models.py
- Remove from `__init__.py` exports if applicable
- `makemigrations` — generates DROP TABLE

### 4. Update imports and tests
- Fix every file that imported the model
- Update or remove tests that tested the model
- Update symbol governance / compliance references

### 5. Run tests
```bash
python3 manage.py test --parallel
```

### 6. Commit with CR SHA
- Commit all changes
- Update CR with commit SHA
- Close CR

## Consolidation Tracker

| Deprecated Model | Canonical Model | Rows | Status |
|-----------------|----------------|------|--------|
| workbench.Project | core.Project | 0 | IN PROGRESS |
| workbench.Hypothesis | core.Hypothesis | 0 | IN PROGRESS |
| workbench.Evidence | core.Evidence | 0 | IN PROGRESS |
| GageStudy | (none — unused) | 0 | PENDING |
| StudyAction | (none — unused) | 0 | PENDING |
| agents_api.dsw/* | agents_api.analysis/* | active | MIGRATING (CR-3c0d0e53) |

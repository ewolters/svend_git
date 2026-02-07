# Change Log

All edits to the kjerne codebase are logged here. Each entry records what changed, why, and how to verify.

## Format

```
### YYYY-MM-DD — Summary
**Debt item:** DEBT.md reference (if applicable)
**Files changed:**
- `path/to/file` — what changed
**Verification:** how to confirm it worked
**Commit:** git hash
```

---

### 2026-02-06 — Lock down repo for public push
**Debt item:** [REPO] svend.db + snapshot tar.gz tracked in git
**Files changed:**
- `.gitignore` — added `*.db`, `*.tar.gz`
- `.kjerne/DEBT.md` — added 22 tracked debt items from full audit
- `services/svend/agents/agents/site/data/svend.db` — removed from git tracking (file kept on disk)
- `.kjerne/snapshots/**/*.tar.gz` (10 files) — removed from git tracking (files kept on disk)
**Verification:** `git status` shows clean, `git ls-files '*.db' '*.tar.gz'` returns empty
**Commit:** 9c9396e

---

### 2026-02-06 — Add project documentation and debt closure process
**Debt item:** N/A (infrastructure)
**Files changed:**
- `CLAUDE.md` (new) — root-level architecture documentation: module map, data model (both current + target), API surface, integration pattern, serving config, working conventions
- `log.md` (new) — change log for all edits
- `DEBT-001.md` (new) — repeatable process for closing technical debt: pick → document → change → test → log → update DEBT.md → commit → push. Includes P1 dependency map.
**Verification:** files exist and are readable
**Commit:** (pending)

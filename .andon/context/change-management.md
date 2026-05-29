---
area: process
---

# Change Management Patterns

## Required Patterns

- ChangeRequest MUST exist before any code change (CHG-001)
- ChangeLog action for in_progress state is 'in_progress' (not 'implementation_started')
- log_md_ref is a plain kebab-case slug (not a log.md anchor)
- evidence_grade field is NOT NULL in DB — always pass `or ''` when persisting
- Gunicorn reload: `pgrep -f 'gunicorn.*wsgi' -o | xargs kill -HUP`

## Anti-Patterns

- Writing code without an active CR in in_progress state
- Using sudo or systemctl for gunicorn operations (no systemd unit exists)
- Skipping risk assessment for feature/enhancement/bugfix types
- Using .get(key, '') for nullable DB fields (returns None if key exists with None value — use `or ''`)

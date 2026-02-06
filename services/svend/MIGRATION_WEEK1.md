# Week 1 Migration Plan: Problem Sessions

## Principle: Additive Only

Everything we add is **alongside** existing functionality, not replacing it.

```
CURRENT STATE                    AFTER WEEK 1
─────────────────────────────    ─────────────────────────────
/app/          → Chat            /app/          → Chat (unchanged)
/app/dsw/      → DSW Pipeline    /app/dsw/      → DSW Pipeline (unchanged)
/app/triage/   → Data Cleaning   /app/triage/   → Data Cleaning (unchanged)
/app/workflows/→ Workflows       /app/workflows/→ Workflows (unchanged)
                                 /app/problems/ → NEW: Problem Sessions
```

Users can continue using everything as-is. Problems are a new optional path.

---

## Migration Steps

### Step 1: Add Problem Model (no downtime)
- New model in accounts/models.py
- New migration
- Run migration on production
- **Risk: None** - additive table creation

### Step 2: Add Problem API (no downtime)
- New file: agents_api/problem_views.py
- New file: agents_api/problem_urls.py
- Add to urls.py
- **Risk: None** - new endpoints only

### Step 3: Add Problems UI (no downtime)
- New template: templates/problems.html
- Add nav link
- **Risk: None** - new page only

### Step 4: Wire Decision Guide (optional integration)
- Decision Guide can optionally create a Problem
- Existing Decision Guide behavior unchanged if no problem context
- **Risk: Low** - backwards compatible

### Step 5: Add "Use in Problem" buttons (optional integration)
- DSW, Researcher, Analyst get optional "Save to Problem"
- Works without problem context (existing behavior)
- **Risk: Low** - backwards compatible

---

## Deployment Chunks

### Chunk 1: Model + Migration (5 min downtime max)
```bash
# On production
cd /home/eric/kjerne/services/svend/web
python3 manage.py makemigrations accounts
python3 manage.py migrate
sudo systemctl restart svend
```

### Chunk 2: API Endpoints (no downtime)
- Just add files and update urls.py
- Restart picks up new routes

### Chunk 3: UI (no downtime)
- Add template
- Add nav link
- Restart

### Chunk 4-5: Integration (no downtime)
- Gradual wiring
- Feature flags if needed

---

## Rollback Plan

Since everything is additive:
- Model: Leave table, it's empty anyway
- API: Remove from urls.py, restart
- UI: Remove nav link, restart

No data loss possible - we're only adding.

---

## Testing Checklist

Before each deploy:
- [ ] Existing chat works
- [ ] DSW from-intent works
- [ ] DSW from-data works
- [ ] Triage clean works
- [ ] Researcher works
- [ ] Workflows run
- [ ] Login/logout works
- [ ] Billing pages load

After Problem deploy:
- [ ] /app/problems/ loads
- [ ] Can create problem
- [ ] Can list problems
- [ ] Can view problem detail
- [ ] All existing features still work

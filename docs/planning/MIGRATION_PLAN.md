# SVEND Architecture Consolidation Plan

## Executive Summary

SVEND has **three parallel systems** for hypothesis-driven investigation:
1. `agents_api.Problem` - JSON blob storage (legacy)
2. `workbench.models.Project` - ORM with artifacts (partial)
3. `core.Project` - Clean ORM with proper Bayesian tracking (canonical)

**Goal:** Consolidate to `core.Project` as the single source of truth.

## Current State

### Three API Endpoints for Same Functionality

| API | Model | Storage | Probability Math |
|-----|-------|---------|------------------|
| `/api/problems/` | agents_api.Problem | JSON blobs | ✅ Uses BayesianUpdater |
| `/api/workbench/projects/` | workbench.Project | ORM | ✅ Uses BayesianUpdater |
| `/api/core/projects/` | core.Project | ORM | ✅ Uses BayesianUpdater |

### What's Been Done

1. ✅ Created `core.bayesian.BayesianUpdater` as single source of truth for probability math
2. ✅ Updated `core.Hypothesis.apply_evidence()` to use BayesianUpdater
3. ✅ Updated `core.Hypothesis.recalculate_probability()` to use BayesianUpdater
4. ✅ Updated `agents_api.Problem._update_probabilities()` to use BayesianUpdater
5. ✅ Updated `workbench.Hypothesis.update_probability()` to use BayesianUpdater
6. ✅ Updated `workbench.KnowledgeGraph.update_from_evidence()` to use BayesianUpdater

## Migration Plan

### Phase 1: Deprecate agents_api.Problem (NEXT)

**Goal:** Redirect `/api/problems/` to use `core.Project` under the hood.

**Steps:**
1. Add `core_project_id` field to `agents_api.Problem` (nullable FK)
2. On Problem creation, also create core.Project
3. Sync hypotheses/evidence to core models on write
4. Eventually remove JSON blob storage

**Timeline:** 2-3 weeks

### Phase 2: Deprecate workbench.Project

**Goal:** Redirect `/api/workbench/projects/` to use `core.Project`.

**Steps:**
1. Add `core_project_id` field to `workbench.Project`
2. Map workbench.Hypothesis → core.Hypothesis
3. Map workbench.Evidence → core.Evidence via EvidenceLink
4. Keep Workbench/Artifact models for UI state (not hypothesis data)

**Timeline:** 3-4 weeks

### Phase 3: Consolidate Knowledge Graphs

**Two implementations:**
- `core.KnowledgeGraph` - Relational (Entity → Relationship → Entity)
- `workbench.KnowledgeGraph` - JSON with Bayesian expansion signals

**Decision:** Merge best features:
- Keep core.KnowledgeGraph relational structure
- Add expansion signals from workbench.KnowledgeGraph to core

**Timeline:** 2 weeks

### Phase 4: Split dsw_views.py

**Current:** 8000+ lines mixing everything

**Target structure:**
```
agents_api/
  analysis/
    __init__.py       # Imports for backwards compat
    stats.py          # Statistical tests
    ml.py             # ML analysis
    bayesian.py       # Bayesian inference
    spc.py            # SPC/control charts
    viz.py            # Visualization
  dsw_views.py        # Just endpoints, imports from analysis/
  dsw_pipeline.py     # DSW from_intent/from_data logic
  dsw_analyst.py      # LLM analyst assistant
```

**Timeline:** 1-2 weeks

## API Stability Plan

During migration, maintain backwards compatibility:

```python
# Example: agents_api.Problem proxying to core.Project

class Problem(models.Model):
    # ... existing fields ...

    # NEW: Link to canonical core.Project
    core_project = models.ForeignKey(
        'core.Project',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='legacy_problems',
    )

    def save(self, *args, **kwargs):
        # Sync to core.Project on save
        if not self.core_project:
            self.core_project = Project.objects.create(
                user=self.user,
                title=self.title,
                problem_statement=self.effect_description,
                # ...
            )
        super().save(*args, **kwargs)
```

## Files to Deprecate (Eventually Delete)

After migration is complete:

1. `agents_api/models.py` - Remove `Problem` class
2. `workbench/models.py` - Remove `Project`, `Hypothesis`, `Evidence` classes
3. Keep `Workbench`, `Artifact`, `KnowledgeGraph` (UI state)

## Testing Strategy

1. Create integration tests that verify:
   - API responses identical before/after migration
   - Probability calculations consistent
   - Data properly synced between old/new models

2. Shadow-write period:
   - Write to both old and new models
   - Compare results
   - Alert on divergence

## Rollback Plan

Each migration phase has a feature flag:

```python
# settings.py
USE_CORE_PROJECT_FOR_PROBLEMS = False  # Phase 1
USE_CORE_PROJECT_FOR_WORKBENCH = False  # Phase 2
```

If issues arise, flip flag back to False.

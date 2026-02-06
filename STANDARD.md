# Kjerne Lab Standards

**Version:** 1.0
**Philosophy:** 5S + Traceability + Quality Gates

---

## 5S Methodology

The Kjerne lab follows 5S principles adapted for software development:

### 1. SEIRI (Sort)

**Rule:** Every file has a purpose. No dead code, no orphans.

| Keep | Remove |
|------|--------|
| Active source code | Commented-out code blocks |
| Tests with coverage | Tests that don't run |
| Documentation that's read | READMEs nobody updates |
| Config that's used | Legacy config files |

**Enforcement:** `kjerne lint` scans for:
- Unused imports
- Unreachable code
- Empty files
- Orphan test files

### 2. SEITON (Set in Order)

**Rule:** Everything has a place. Find anything in 3 seconds.

```
~/kjerne/
├── core/                    # SHARED - imported by all services
│   ├── llm.py               # LLM loading utilities
│   ├── quality.py           # Quality framework
│   └── ...
├── services/                # PRODUCTS - deployable units
│   ├── svend/               # Agent service ($5/mo)
│   │   ├── agents/          # Individual agents
│   │   ├── tools/           # Deterministic tools
│   │   ├── docs/            # Documentation generators
│   │   └── api/             # FastAPI endpoints
│   ├── forge/               # Synthetic data API
│   │   ├── generators/      # Data generators
│   │   ├── schemas/         # Data schemas
│   │   └── api/             # FastAPI endpoints
│   └── scrub/               # Data cleaning service
│       ├── cleaners/        # Cleaning modules
│       └── api/             # FastAPI endpoints
├── lab/                     # R&D - not deployed
│   └── synara/              # Synara OS development
├── .kjerne/                 # Meta-tooling
│   ├── snapshots/           # Point-in-time snapshots
│   ├── diffs/               # Diff history
│   ├── hooks/               # Pre-deploy hooks
│   └── config.json          # Lab configuration
└── STANDARD.md              # This file
```

**Naming Conventions:**
- Directories: `lowercase_snake`
- Python files: `lowercase_snake.py`
- Classes: `PascalCase`
- Functions: `lowercase_snake`
- Constants: `UPPER_SNAKE`

### 3. SEISO (Shine)

**Rule:** Clean as you go. Technical debt is tracked, not ignored.

**Before committing:**
1. Run `kjerne lint` - must pass
2. Run `kjerne test` - must pass
3. Run `kjerne security` - no CRITICAL/HIGH

**Debt tracking:**
- File: `~/kjerne/.kjerne/DEBT.md`
- Format: `[SERVICE] Description | Added: DATE | Priority: P1/P2/P3`
- Review weekly

### 4. SEIKETSU (Standardize)

**Rule:** Same patterns everywhere. Copy-paste should work.

**Service Template:**
```
service_name/
├── __init__.py              # Exports public API
├── agent.py                 # Main orchestrator (if agent-based)
├── models.py                # Data models (dataclasses/Pydantic)
├── config.py                # Service configuration
├── api/                     # API endpoints (if service has API)
│   ├── __init__.py
│   ├── routes.py
│   └── schemas.py           # Request/response schemas
└── tests/
    ├── __init__.py
    ├── test_agent.py
    └── conftest.py
```

**Quality Report Interface:**
Every service must produce a `QualityReport` (from `core.quality`):
```python
from core.quality import QualityReport, QualityGrade

class ServiceResult:
    output: Any
    quality: QualityReport  # Required
```

**Error Handling:**
```python
# Standard pattern - raise domain-specific exceptions
class ServiceError(Exception):
    """Base for this service."""
    pass

class ValidationError(ServiceError):
    """Input validation failed."""
    pass

class ProcessingError(ServiceError):
    """Processing failed."""
    pass
```

### 5. SHITSUKE (Sustain)

**Rule:** The system enforces itself. Humans just review.

**Automated enforcement:**
- `kjerne lint` runs before snapshot
- `kjerne test` runs before prod deploy
- `kjerne security` runs before prod deploy
- `kjerne diff` shows what changed

**Manual review:**
- Weekly: Review `DEBT.md`, close or defer
- Monthly: Review service boundaries, refactor if leaking
- Quarterly: Review core/ for bloat

---

## Mini-Git Workflow

### Concept

The lab has two states:
- **~/kjerne/** - Development (active work)
- **~/prod/** - Production (stable, deployed)

Mini-git provides:
1. **Snapshots** - Point-in-time captures of services
2. **Diffs** - What changed between snapshots
3. **Validation** - Quality gates before deployment
4. **Sync** - Safe copy from dev to prod

### Commands

```bash
# Snapshot current state
kjerne snapshot "Added user auth to Svend"

# View recent snapshots
kjerne log

# View what changed since last snapshot
kjerne diff

# Validate before deploy (runs all checks)
kjerne validate services/svend

# Deploy to prod (copies validated service)
kjerne deploy services/svend

# Rollback prod to previous snapshot
kjerne rollback services/svend
```

### Snapshot Format

```
.kjerne/snapshots/
├── 2024-01-27_143022_abc123/
│   ├── manifest.json          # What was snapshotted
│   ├── checksums.json         # SHA256 of all files
│   ├── services_svend.tar.gz  # Compressed snapshot
│   └── quality.json           # Quality report at snapshot time
```

**manifest.json:**
```json
{
  "id": "abc123",
  "timestamp": "2024-01-27T14:30:22Z",
  "message": "Added user auth to Svend",
  "services": ["svend"],
  "author": "eric",
  "checksums_sha256": "..."
}
```

### Validation Rules

Before `kjerne deploy`, these must pass:

| Check | Failure Action |
|-------|----------------|
| Syntax (all .py parse) | Block |
| Tests pass | Block |
| No CRITICAL security | Block |
| No HIGH security | Warn (require --force) |
| Quality grade >= C | Warn (require --force) |
| No uncommitted changes | Block |

### Deployment Flow

```
kjerne dev                    # Normal development
    │
    ▼
kjerne snapshot "message"     # Capture state
    │
    ▼
kjerne validate services/X    # Run all checks
    │
    ├── FAIL → Fix issues
    │
    ▼ PASS
kjerne deploy services/X      # Copy to ~/prod/
    │
    ▼
~/prod/services/X is live
```

---

## Service Interfaces

Services must be standalone but connectable.

### Import Rules

```python
# ALLOWED - import from core
from core.quality import QualityReport
from core.llm import load_qwen

# ALLOWED - import within same service
from .models import MyModel
from .config import settings

# FORBIDDEN - cross-service imports
from services.svend.agents import ResearchAgent  # NO!
```

### Service Communication

Services communicate via:
1. **API calls** (preferred for deployed services)
2. **Adapter interfaces** (for local dev/testing)

```python
# Adapter pattern for testability
class ForgeAdapter:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or "http://localhost:8001"

    def generate(self, schema: dict, n: int) -> pd.DataFrame:
        # In prod: HTTP call
        # In test: mock
        pass
```

---

## Quality Gates

### Per-Service Requirements

| Service | Lint | Tests | Security | Quality Grade |
|---------|------|-------|----------|---------------|
| svend | Required | Required | No CRITICAL | >= C |
| forge | Required | Required | No CRITICAL | >= C |
| scrub | Required | Required | No CRITICAL | >= C |
| core | Required | Required | No HIGH | >= B |

### Continuous Checks

The `.kjerne/hooks/pre-snapshot` runs automatically:
```bash
#!/bin/bash
kjerne lint
kjerne test --quick
kjerne security --level HIGH
```

---

## File Ownership

| Path | Owner | Changes Require |
|------|-------|-----------------|
| `core/` | Lab | Review by any service owner |
| `services/svend/` | Svend team | Self-review |
| `services/forge/` | Forge team | Self-review |
| `services/scrub/` | Scrub team | Self-review |
| `.kjerne/` | Lab | Review + STANDARD.md update |
| `STANDARD.md` | Lab | Consensus |

---

## Versioning

### Lab Version

The entire lab has a version (in `.kjerne/config.json`):
```json
{
  "version": "1.0.0",
  "services": {
    "svend": "1.2.0",
    "forge": "0.5.0",
    "scrub": "0.3.0"
  }
}
```

### Service Versions

Each service has its own version in `__init__.py`:
```python
__version__ = "1.2.0"
```

Bump rules:
- **MAJOR** - Breaking API changes
- **MINOR** - New features, backward compatible
- **PATCH** - Bug fixes only

---

## Migration Checklist

When adding a new service to Kjerne:

- [ ] Create directory under `services/`
- [ ] Add `__init__.py` with `__version__`
- [ ] Add `tests/` directory with at least one test
- [ ] Ensure no cross-service imports
- [ ] Add entry to `.kjerne/config.json`
- [ ] Run `kjerne validate services/newservice`
- [ ] Update this STANDARD.md if needed

---

## Emergency Procedures

### Production Issue

```bash
# 1. Rollback immediately
kjerne rollback services/affected_service

# 2. Document
echo "INCIDENT $(date): Description" >> .kjerne/INCIDENTS.md

# 3. Investigate in dev
# ... fix ...

# 4. Re-deploy with extra validation
kjerne validate --strict services/affected_service
kjerne deploy services/affected_service
```

### Lost Snapshot

Snapshots are in `.kjerne/snapshots/`. If deleted:
1. Check if copied to external backup
2. If not, use `kjerne rebuild` to create new baseline

---

*Last updated: 2024-01-27*

# Kjerne

Monorepo for **SVEND** — a process knowledge system for manufacturing and continuous improvement.

## What SVEND Is

SVEND is a platform where every analysis, investigation, and observation writes to a shared model of your process. The tools are how you build the model. The model is the product.

Three concerns:
- **Process Knowledge** (GRAPH-001) — structured, evidence-based, maintained
- **Learning System** (LOOP-001) — detect, investigate, standardize, verify
- **Compliance** (QMS-001) — audit evidence and regulatory mappings

## Architecture

```
~/kjerne/
├── docs/
│   ├── standards/          # 46 machine-readable standards (DOC-001 through OLR-001)
│   └── planning/           # Roadmaps, specs, milestone docs
│       └── object_271/     # Graph architecture milestone
│
├── services/svend/
│   ├── web/                # Django application
│   │   ├── graph/          # Process knowledge graph (GRAPH-001)
│   │   ├── loop/           # Closed-loop operating model (LOOP-001)
│   │   ├── safety/         # HIRARC safety management
│   │   ├── core/           # Tenants, projects, hypotheses
│   │   ├── agents_api/     # QMS tools, FMEA, DOE, SPC, Hoshin
│   │   ├── syn/            # Synara infrastructure layer
│   │   │   ├── core/       # Base models
│   │   │   ├── audit/      # Compliance automation
│   │   │   ├── log/        # Structured logging
│   │   │   ├── sched/      # Task scheduler
│   │   │   └── varta/      # Active defense
│   │   └── templates/      # Django templates (vanilla JS)
│   │
│   └── site/               # Landing page (svend.ai)
│
└── CLAUDE.md               # Full project context
```

## Stack

- **Django** on PostgreSQL
- **Synara** — Bayesian belief engine (custom, 0 trainable parameters)
- **Gunicorn** behind **Cloudflare Tunnel**
- **WhiteNoise** for static files
- Vanilla JavaScript frontend (no SPA framework)
- **Anthropic API** for LLM features

## Key Surfaces

| Surface | URL | What it does |
|---------|-----|-------------|
| Process Map | /app/process-map/ | Cytoscape.js graph navigator with 4 view lenses |
| Loop Dashboard | /app/loop/ | Closed-loop operating model (signals, investigations, commitments) |
| Analysis Workbench | /app/analysis/ | 200+ statistical analyses, DOE, SPC, ML |
| QMS Dashboard | /app/iso/ | NCR, audits, training, documents, equipment, control plans |
| Safety | /app/safety/ | HIRARC frontier cards, zone management |
| Hoshin Kanri | /app/hoshin/ | Strategic planning, X-Matrix, resource management |

## Standards

46 machine-readable standards govern the codebase. Key ones:

- **GRAPH-001** — Unified Knowledge Graph and Process Model
- **LOOP-001** — Closed-Loop Operating Model
- **QMS-001** — Quality Management System
- **CANON-001/002** — System architecture and integration contracts
- **CHG-001** — Change management (mandatory for all code changes)

Run `python manage.py run_compliance --all` to verify.

## Environment

```bash
# Source environment (required before any Django command)
set -a && source /etc/svend/env && set +a

# No venv needed — system-wide Python 3.10
python3 manage.py check
python3 manage.py run_compliance --standards
python3 -m pytest graph/ -x
```

## Change Management

Every code change requires a ChangeRequest per CHG-001. The pre-commit hook enforces this. See `CLAUDE.md` for the full lifecycle.

## Lineage

CCS 1948 -> Shingo/Ohno -> Charlie Protzman -> Eric Wolters -> Claude -> SVEND

Builder's mark: **&#9673;** — the focused circle.

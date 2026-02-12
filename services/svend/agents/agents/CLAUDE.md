# SVEND Agents

Multi-agent service with documented outputs. From $19/month.

**Philosophy:** The LLM executes tasks. The tools provide value. Deterministic outputs you can trust.

## What's Included

### Agents
| Agent | Special Ability | What it does |
|-------|-----------------|--------------|
| `coder/` | Runs & verifies code | Code generation with syntax check, execution, complexity analysis |
| `researcher/` | Real search APIs | Multi-source research (arXiv, Semantic Scholar, Brave) with citations |
| `writer/` | Templates + voice matching | Document generation with your format and your style |
| `reviewer/` | Pattern detection | Document review with readability scoring |
| `guide/` | Structured interviews | Interview-style workflows (business plan builder) |
| `experimenter/` | DOE + power analysis | Experimental design with visualizations |

### ML Services (Standalone)
| Service | What it does |
|---------|--------------|
| `scrub/` | Data cleaning: outlier detection, missing data, type correction, normalization |
| `analyst/` | ML training: auto model selection, training, QA reports with education |
| `dsw/` | Decision Science Workbench: Zero-to-classifier pipeline with Forge integration |

### Decision Science Workbench
The orchestrator that connects everything:

```
Intent → Research → Schema → Forge → Scrub → Analyst → Deployment
```

Start with just an idea (no data) and get: trained model, deployment code, improvement plan.

### Documentation Tools
| Tool | Output |
|------|--------|
| `docs/chemistry` | LaTeX chemical equations, thermodynamics, reaction mechanisms |
| `docs/latex` | Mathematical notation, matrices, document formatting |
| `docs/charts` | Styled visualizations (Svend theme) |
| `docs/export` | Markdown, LaTeX, HTML, PDF, DOCX export |

### Deterministic Tools
| Tool | Metrics |
|------|---------|
| `tools/readability` | Flesch-Kincaid, Gunning Fog, SMOG, ARI |
| `tools/complexity` | Cyclomatic, Cognitive, Maintainability Index |
| `tools/schema` | JSON Schema validation |
| `tools/stats` | Word count, vocabulary, code statistics |
| `tools/grammar` | Spelling, grammar rules, style checking |

### Workflow Engine
Chain agents together. Save as Python scripts. Re-run anytime.

```python
from workflow import Workflow, ResearchStep, CoderStep, ChemistryStep, WriterStep

workflow = Workflow("Na-K Pump Study")
workflow.add(ResearchStep("ATP hydrolysis Na-K pump"))
workflow.add(CoderStep("simulate pump", uses=["research"]))
workflow.add(ChemistryStep(reactions=["ATP + H2O -> ADP + Pi"]))
workflow.add(WriterStep("lab_report", uses=["research", "code", "chemistry"]))
result = workflow.run()
```

---

## Quick Start

```bash
cd /home/eric/Desktop/agents

# === RESEARCH ===
python -c "
from researcher.agent import ResearchAgent, ResearchQuery
agent = ResearchAgent()
result = agent.run(ResearchQuery('CRISPR gene editing', focus='scientific'))
print(result.to_markdown())
"

# === WRITE WITH TEMPLATE + VOICE ===
python -c "
from writer.agent import WriterAgent, DocumentRequest, DocumentType
from writer.voice import VoiceAnalyzer

# Analyze your writing style
analyzer = VoiceAnalyzer()
voice = analyzer.analyze(['Your sample text here...'], 'my_style')

agent = WriterAgent()
request = DocumentRequest(
    topic='AI in Healthcare',
    doc_type=DocumentType.GRANT_PROPOSAL,
    voice=voice,
)
doc = agent.write(request)
print(doc.to_markdown())
"

# === EXPERIMENTAL DESIGN ===
python -c "
from experimenter import quick_power, quick_factorial

# Power analysis
result = quick_power(effect_size=0.5)
print(result.summary())

# Factorial design
design = quick_factorial({
    'Temperature': [100, 150],
    'Pressure': [1, 2],
    'Catalyst': ['A', 'B'],
})
print(design.to_markdown())
"

# === CHEMISTRY DOCUMENTATION ===
python -c "
from docs.chemistry import ChemistryFormatter

formatter = ChemistryFormatter()
print(formatter.reaction_to_latex('ATP + H2O -> ADP + Pi'))
print(formatter.get_biochemical_info('na_k_pump'))
"

# === WORKFLOW ===
python -c "
from workflow import Workflow, ResearchStep, WriterStep

workflow = Workflow('Quick Analysis')
workflow.add(ResearchStep('machine learning interpretability'))
workflow.add(WriterStep('executive_summary', uses=['research']))
result = workflow.run()
print(result.to_markdown())

# Save as reusable Python script
print(workflow.to_python())
"
```

---

## Agent Details

### Coder Agent
Code generation with verification and QA reports.

```python
from coder.agent import CodingAgent, CodingTask

agent = CodingAgent(llm=your_llm)  # or None for mock
task = CodingTask(
    description="Parse CSV with email validation",
    constraints=["Handle errors gracefully", "Return both valid and invalid rows"],
)
result = agent.run(task)

# Get the code
print(result.code)

# Full QA report with human review checklist
print(result.qa_report())

# Save to file
result.save("output.py")
```

**Special abilities:**
- Syntax verification (AST parse)
- Code execution in sandbox with demo output
- Complexity analysis
- Bayesian quality scoring
- Human review checklist generation
- QA report with execution demo

### Researcher Agent
Multi-source research with real APIs.

```python
from researcher.agent import ResearchAgent, ResearchQuery

agent = ResearchAgent()
query = ResearchQuery(
    question="CRISPR applications in cancer treatment",
    focus="scientific",  # or "market", "general"
    depth="thorough",    # or "quick", "standard"
)
result = agent.run(query)
print(f"Sources: {len(result.sources)}")
print(result.to_markdown())
```

**Special abilities:**
- Real APIs: arXiv, Semantic Scholar, Brave Search
- Source diversity tracking (Shannon index)
- Credibility scoring by domain
- Structured citations

### Writer Agent
Document generation with templates and voice matching.

```python
from writer.agent import WriterAgent, DocumentRequest, DocumentType
from writer.templates import BUILTIN_TEMPLATES
from writer.voice import VoiceAnalyzer

# Extract your writing style
analyzer = VoiceAnalyzer()
voice = analyzer.analyze([
    "Your previous writing sample 1...",
    "Your previous writing sample 2...",
], name="my_voice")

# Use built-in or custom template
template = BUILTIN_TEMPLATES["grant_proposal"]

agent = WriterAgent()
request = DocumentRequest(
    topic="Neural Network Interpretability",
    doc_type=DocumentType.GRANT_PROPOSAL,
    template=template,
    voice=voice,
    target_reading_level=12,
)
doc = agent.write(request)
print(doc.quality_report())
```

**Special abilities:**
- Custom templates (user-defined sections, word limits)
- Voice matching (extracts style from samples)
- Quality gates (reading level, section lengths)
- 8 built-in templates: grant_proposal, sales_proposal, technical_spec, executive_summary, blog_post, whitepaper, literature_review, press_release

### Experimenter Agent
Experimental design with power analysis and DOE.

```python
from experimenter import ExperimenterAgent, ExperimentRequest

agent = ExperimenterAgent(seed=42)  # Reproducible

# Power analysis
request = ExperimentRequest(
    goal="Compare 3 drug treatments",
    request_type="power",
    test_type="anova",
    effect_size=0.25,
    groups=3,
)
result = agent.design_experiment(request)
print(f"Sample size needed: {result.power_result.sample_size}")

# Full factorial design
request = ExperimentRequest(
    goal="Optimize reaction yield",
    request_type="design",
    design_type="full_factorial",
    factors=[
        {"name": "Temperature", "levels": [100, 150], "units": "°C"},
        {"name": "Time", "levels": [30, 60], "units": "min"},
        {"name": "Catalyst", "levels": ["A", "B"]},
    ],
)
result = agent.design_experiment(request)
print(result.design.to_markdown())
```

**Special abilities:**
- Power analysis (t-test, ANOVA, chi-square, correlation)
- Full factorial designs (2^k, general)
- Fractional factorial (Resolution III, IV, V)
- Central Composite Design (response surface)
- Latin Square, Randomized Block
- Visualizations (power curves, design matrices)

---

## Documentation Tools

### Chemistry (`docs/chemistry`)

```python
from docs.chemistry import ChemistryFormatter, quick_reaction

formatter = ChemistryFormatter()

# Format reaction as LaTeX
latex = formatter.reaction_to_latex("2H2 + O2 -> 2H2O")
# Output: \ce{2H2 + O2 -> 2H2O}

# Get thermodynamics
thermo = formatter.estimate_thermodynamics("CH4 + 2O2 -> CO2 + 2H2O")
print(thermo.delta_h)  # Enthalpy

# Known biochemical reactions
info = formatter.get_biochemical_info("atp_hydrolysis")
# Returns: equation, ΔG, description, spontaneity
```

**Included reactions:** ATP hydrolysis, glucose oxidation, photosynthesis, Na-K pump

### LaTeX (`docs/latex`)

```python
from docs.latex import LaTeXFormatter, get_equation

formatter = LaTeXFormatter()

# Fractions, integrals, matrices
print(formatter.fraction("dy", "dx"))
print(formatter.integral("0", "\\infty", "e^{-x}", "x"))
print(formatter.matrix([[1, 2], [3, 4]], "bmatrix"))

# Common equations
eq = get_equation("gibbs_free_energy")
print(eq.display())  # $$\Delta G = \Delta H - T\Delta S$$

# Full LaTeX document
doc = formatter.document(
    title="My Report",
    author="Researcher",
    content="..."
)
```

**Included equations:** Ideal gas, Einstein mass-energy, Gibbs, Nernst, Arrhenius, Schrödinger, Michaelis-Menten, Henderson-Hasselbalch

### Charts (`docs/charts`)

```python
from docs.charts import ChartGenerator, quick_chart

generator = ChartGenerator()

# Line chart
chart = generator.create_chart(
    data={'x': [1,2,3,4,5], 'y': [1,4,9,16,25]},
    chart_type='line',
    title='Quadratic Growth',
    x_label='X',
    y_label='Y',
)

# Bar chart
chart = generator.create_chart(
    data={'categories': ['A', 'B', 'C'], 'values': [10, 25, 15]},
    chart_type='bar',
)

# Heatmap, scatter, box, histogram also available
# Multi-panel figures with generator.multi_panel()
```

### Export (`docs/export`)

```python
from docs.export import DocumentExporter

exporter = DocumentExporter()

sections = [
    {"title": "Introduction", "content": "..."},
    {"title": "Methods", "content": "..."},
    {"title": "Results", "content": "..."},
]

# Export to various formats
exporter.export(sections, "report.md", format="markdown")
exporter.export(sections, "report.tex", format="latex")   # Raw LaTeX for editing
exporter.export(sections, "report.html", format="html")
exporter.export(sections, "report.pdf", format="pdf")     # Requires pdflatex
exporter.export(sections, "report.docx", format="docx")   # Microsoft Word
exporter.export(sections, "report.json", format="json")   # Structured data
```

---

## ML Services

### Scrub (Data Cleaning)

Clean messy data automatically:
- **Type correction**: Infer and convert types (strings → numbers/dates)
- **Missing data**: Impute with mean/median/mode/KNN or flag for review
- **Outlier detection**: IQR, Z-score, Isolation Forest, domain rules
- **Factor normalization**: Case normalization, typo correction via fuzzy matching

```python
from scrub import DataCleaner, CleaningConfig

cleaner = DataCleaner()
config = CleaningConfig(
    domain_rules={'age': (0, 120), 'salary': (0, None)},
)
df_clean, result = cleaner.clean(df, config)

print(result.summary())
# Shows: outliers flagged, missing values filled, normalizations applied
```

**Key principle:** Outliers are FLAGGED, not removed. Human decides.

### Analyst (ML Training)

Train interpretable ML models with educational outputs:
- **Models:** Linear/Logistic Regression, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting
- **Auto-selection:** Chooses best model based on data characteristics
- **Outputs:** Trained model, reproducible code, QA report, educational documentation

```python
from analyst import Analyst

analyst = Analyst()
result = analyst.train(
    data=df,
    target='outcome',
    intent='Predict customer churn',
    priority='balanced',  # or 'accuracy', 'interpretability', 'speed'
)

# Get the model
model = result.model

# Get reproducible Python code
print(result.code)

# Get educational report (explains how the model works)
print(result.report.to_markdown())

# Save everything
result.save('output/')  # model.pkl, train.py, report.md
```

**Compare models:**
```python
results = analyst.compare_models(df, target='outcome')
for model_type, report in results:
    acc = next(m.value for m in report.metrics if m.name == 'Accuracy')
    print(f'{model_type.value}: {acc:.3f}')
```

### Decision Science Workbench (DSW)

Complete zero-to-classifier pipeline. Two entry points:

**1. From Intent (no data needed):**
```python
from dsw import DecisionScienceWorkbench

dsw = DecisionScienceWorkbench()
result = dsw.from_intent(
    intent="Predict which customers will churn in the next 30 days",
    domain="churn",  # or "fraud", "lead_scoring", or auto-detect
    n_records=1000,
)

# What you get:
print(result.schema.summary())     # Problem definition with features, edge cases
print(result.synthetic_data.head()) # Generated by Forge
print(result.model)                # Trained model
print(result.deployment_code)      # Ready-to-use Python
print(result.improvement_plan.to_markdown())  # What to collect next
result.save('output/')
```

**2. From Existing Data:**
```python
result = dsw.from_data(
    data=df,
    target='churned',
    intent='Predict customer churn',
)
```

**Pipeline:** `Intent → Schema → Forge → Scrub → Analyst → Deployment`

### Service Interfaces

All services follow a consistent pattern for plug-and-play:

```python
from dsw.interfaces import ScrubAdapter, AnalystAdapter, ForgeRequest

# Each service has: request → run() → output
scrub = ScrubAdapter()
output = scrub.run(ScrubRequest(data=df, domain_rules={'age': (0, 120)}))

# Compose into pipelines
from dsw.interfaces import Pipeline
pipeline = Pipeline()
pipeline.add("scrub", scrub_adapter, lambda _: ScrubRequest(data=df))
pipeline.add("analyst", analyst_adapter, lambda prev: AnalystRequest(data=prev.data, target="y"))
results = pipeline.run()
```

### Forge Integration

Generate data from schema or JSON:

```python
from dsw import ForgeClient, ProblemSchema

client = ForgeClient()

# From schema
df = client.generate_from_schema(problem_schema, n=1000)

# From JSON (infers schema)
sample = '{"age": 25, "income": 50000, "status": ["active", "inactive"]}'
df = client.generate_from_json(sample, n=1000)

# Direct schema
df = client.generate({"age": {"type": "int", "constraints": {"min": 18}}}, n=1000)
```

---

## Workflow Engine

Chain agents together. No AI orchestration - just a for loop.

### Basic Usage

```python
from workflow import Workflow, ResearchStep, CoderStep, WriterStep, ExportStep

workflow = Workflow("My Analysis")

# Add steps - each can use outputs from previous steps
workflow.add(ResearchStep("topic to research", name="research"))
workflow.add(CoderStep("code to generate", uses=["research"], name="code"))
workflow.add(WriterStep("report", uses=["research", "code"], name="doc"))
workflow.add(ExportStep("output.pdf", format="pdf", uses=["doc"]))

# Run
result = workflow.run()
print(result.to_markdown())

# Save as reproducible Python script
with open("my_workflow.py", "w") as f:
    f.write(workflow.to_python())
```

### Available Steps

| Step | Purpose |
|------|---------|
| `ResearchStep(query, focus, depth)` | Research with real APIs |
| `CoderStep(prompt, language, constraints)` | Generate code |
| `WriterStep(topic, template, tone)` | Generate document |
| `ReviewStep(doc_type)` | Review document quality |
| `ExperimentStep(goal, test_type, factors)` | Design experiment |
| `ChemistryStep(reactions, include_thermodynamics)` | Format chemistry |
| `ChartStep(chart_type, title)` | Generate chart |
| `ExportStep(output_path, format)` | Export to file |
| `CustomStep(func)` | Your own function |

### Example: Full Research Pipeline

```python
from workflow import *

workflow = Workflow("Market Analysis")

workflow.add(ResearchStep(
    query="electric vehicle battery technology trends 2024",
    focus="market",
    depth="thorough",
    name="research"
))

workflow.add(ChartStep(
    chart_type="bar",
    title="Market Segments",
    name="chart",
    uses=["research"]
))

workflow.add(WriterStep(
    topic="EV Battery Market Analysis",
    template="executive_summary",
    tone="business",
    name="report",
    uses=["research", "chart"]
))

workflow.add(ReviewStep(
    doc_type="business",
    name="review",
    uses=["report"]
))

workflow.add(ExportStep(
    output_path="ev_analysis.pdf",
    format="pdf",
    name="export",
    uses=["report"]
))

result = workflow.run()
```

---

## Search APIs

| API | Status | Cost | Rate Limit |
|-----|--------|------|------------|
| arXiv | Working | Free | 3s delay |
| Semantic Scholar | Working | Free | 1s delay |
| Brave Search | Working | Free tier (2K/mo) | 1.5s delay |
| DuckDuckGo | Limited | Free | Instant answers only |

Set API keys (environment variables):
```bash
export BRAVE_API_KEY=your_key        # Web search (2K free/month)
export SEMANTIC_SCHOLAR_API_KEY=key  # Higher rate limits (optional)
```

---

## File Structure

```
agents/
├── core/           # Shared components
│   ├── intent.py   # Intent tracking, drift detection
│   ├── executor.py # Code execution sandbox
│   ├── verifier.py # Syntax, lint, execution checks
│   ├── reasoning.py# Bayesian quality assessment
│   ├── sources.py  # Citation management
│   └── search.py   # Real search APIs
├── coder/          # Code generation agent
├── researcher/     # Research agent
├── writer/         # Document generation agent
│   ├── templates.py# Custom templates
│   └── voice.py    # Voice matching
├── reviewer/       # Document review agent
├── guide/          # Interview-style agents
│   └── business_plan.py
├── experimenter/   # Experimental design
│   ├── stats.py    # Power analysis
│   ├── doe.py      # Design of experiments
│   └── plots.py    # Visualizations
├── tools/          # Deterministic tools
│   ├── readability.py
│   ├── complexity.py
│   ├── schema.py
│   ├── stats.py
│   └── grammar.py  # Spelling and style checker
├── docs/           # Documentation tools
│   ├── chemistry.py
│   ├── latex.py
│   ├── charts.py
│   └── export.py
├── workflow/       # Workflow engine
│   ├── engine.py
│   └── steps.py
├── scrub/          # Data cleaning service
│   ├── cleaner.py  # Main orchestrator
│   ├── outliers.py # IQR, Z-score, Isolation Forest
│   ├── missing.py  # Imputation strategies
│   ├── normalize.py# Factor normalization
│   └── types.py    # Type inference/correction
├── analyst/        # ML training service
│   ├── trainer.py  # Training orchestrator
│   ├── models.py   # Model definitions + educational content
│   ├── selector.py # Auto model selection
│   └── reporter.py # QA reports with education
├── dsw/            # Decision Science Workbench
│   ├── workbench.py   # Main orchestrator
│   ├── schema.py      # Problem schema (Research → Forge)
│   ├── forge_client.py# Forge API client
│   └── interfaces.py  # Service interfaces for plug-and-play
└── CLAUDE.md       # This file
```

---

## LLM Setup

Load a local LLM for agent use:

```python
from core.llm import load_qwen

# Load Qwen (0.5B, 1.5B, 7B, 14B, or coder-14B)
llm = load_qwen("7B")  # Uses GPU if available

# Use with agents
from coder.agent import CodingAgent, CodingTask
agent = CodingAgent(llm=llm)
result = agent.run(CodingTask(description="Write a fibonacci function"))
print(result.code)
```

Supported models (cached in `~/.cache/huggingface/hub/`):
- `Qwen/Qwen2.5-0.5B-Instruct` (fast, lightweight)
- `Qwen/Qwen2.5-1.5B-Instruct` (balanced)
- `Qwen/Qwen2.5-7B-Instruct` (recommended)
- `Qwen/Qwen2.5-14B-Instruct` (high quality)
- `Qwen/Qwen2.5-Coder-14B-Instruct` (code-focused)

---

## Requirements

```
scipy
matplotlib
numpy
transformers
torch
pandas
scikit-learn
python-docx
pyspellchecker
```

Optional:
- `pdflatex` for PDF export (`apt install texlive-latex-base`)
- Brave API key for web search
- Semantic Scholar API key for higher rate limits

---

## Production Notes

Working features:
- [x] All 6 agents functional
- [x] Real search APIs (arXiv, Semantic Scholar, Brave)
- [x] Custom templates and voice matching
- [x] Experimental design with power analysis
- [x] Chemistry and LaTeX formatting
- [x] Workflow engine with Python export
- [x] Deterministic quality tools
- [x] ML services: Scrub (cleaning), Analyst (training), Workbench (full pipeline)
- [x] Educational documentation for ML models

Known limitations:
- Rate limiting on search APIs (add delays)
- PDF export requires pdflatex installed
- LLM integration requires separate model loading

---

## License

MIT

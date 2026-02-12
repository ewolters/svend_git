"""
Svend Agents - AI Agent Service

Agents with documented, deterministic outputs. From $19/month.

Philosophy: The LLM executes tasks. The tools provide value.
Deterministic outputs you can trust.

Agents:
    - Coder: Code generation with verification
    - Researcher: Multi-source research with citations
    - Writer: Document generation with templates and voice matching
    - Reviewer: Document review with readability scoring
    - Guide: Interview-style workflows (business plan builder)
    - Experimenter: Experimental design with power analysis
    - Analyst: ML training with educational outputs

Services:
    - DSW: Decision Science Workbench (zero-to-classifier pipeline)
    - Tools: Readability, complexity, grammar, stats
    - Docs: Chemistry, LaTeX, charts, export
    - Workflow: Chain agents together
"""

__version__ = "1.0.0"

# Agent exports
from .agents.coder import CodingAgent, CodingTask, CodingResult
from .agents.researcher import ResearchAgent, ResearchQuery, ResearchResult
from .agents.writer import WriterAgent, DocumentRequest, Document
from .agents.reviewer import ReviewerAgent, ReviewResult
from .agents.guide import BusinessPlanGuide
from .agents.experimenter import ExperimenterAgent, ExperimentRequest

# ML exports
from .agents.analyst import Analyst, AnalystResult, AnalystReport
from .dsw import DecisionScienceWorkbench, DSWResult, ProblemSchema

__all__ = [
    "__version__",
    # Agents
    "CodingAgent", "CodingTask", "CodingResult",
    "ResearchAgent", "ResearchQuery", "ResearchResult",
    "WriterAgent", "DocumentRequest", "Document",
    "ReviewerAgent", "ReviewResult",
    "BusinessPlanGuide",
    "ExperimenterAgent", "ExperimentRequest",
    # ML
    "Analyst", "AnalystResult", "AnalystReport",
    "DecisionScienceWorkbench", "DSWResult", "ProblemSchema",
]

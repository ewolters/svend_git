"""Core models for Svend.

This module contains the foundational models:
- Tenant, Membership: Multi-tenancy for teams/enterprises
- KnowledgeGraph, Entity, Relationship: Global knowledge storage
- Project, Dataset, ExperimentDesign: Container for hypothesis-driven investigation
- Hypothesis, Evidence, EvidenceLink: Bayesian reasoning
"""

from .tenant import Tenant, Membership
from .graph import KnowledgeGraph, Entity, Relationship
from .project import Project, Dataset, ExperimentDesign
from .hypothesis import Hypothesis, Evidence, EvidenceLink

__all__ = [
    # Multi-tenancy
    "Tenant",
    "Membership",
    # Knowledge Graph
    "KnowledgeGraph",
    "Entity",
    "Relationship",
    # Project & Data
    "Project",
    "Dataset",
    "ExperimentDesign",
    # Hypothesis & Evidence
    "Hypothesis",
    "Evidence",
    "EvidenceLink",
]

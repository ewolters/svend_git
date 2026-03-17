"""Core models for Svend.

This module contains the foundational models:
- Tenant, Membership: Multi-tenancy for teams/enterprises
- KnowledgeGraph, Entity, Relationship: Global knowledge storage
- Project, Dataset, ExperimentDesign: Container for hypothesis-driven investigation
- Hypothesis, Evidence, EvidenceLink: Bayesian reasoning
- Notebook, Trial, NotebookPage: Tactical execution layer (NB-001)
- HanseiKai, Yokoten, YokotenAdoption: Reflection and lateral learning transfer
"""

from .graph import Entity, KnowledgeGraph, Relationship
from .hypothesis import Evidence, EvidenceLink, Hypothesis
from .investigation import Investigation, InvestigationMembership, InvestigationToolLink
from .measurement import GageStudy, MeasurementSystem
from .notebook import (
    HanseiKai,
    Notebook,
    NotebookPage,
    Trial,
    TrialAction,
    TrialToolLink,
    Yokoten,
    YokotenAdoption,
)
from .project import Dataset, ExperimentDesign, Project, StudyAction
from .tenant import Membership, OrgInvitation, Tenant
from .training import StudentEnrollment, TrainingCenter, TrainingProgram

__all__ = [
    # Multi-tenancy
    "Tenant",
    "Membership",
    "OrgInvitation",
    # Training (ILSSI partner infrastructure)
    "TrainingCenter",
    "TrainingProgram",
    "StudentEnrollment",
    # Knowledge Graph
    "KnowledgeGraph",
    "Entity",
    "Relationship",
    # Project & Data
    "Project",
    "Dataset",
    "ExperimentDesign",
    "StudyAction",
    # Hypothesis & Evidence
    "Hypothesis",
    "Evidence",
    "EvidenceLink",
    # Measurement Systems (CANON-002 §4, §12.2)
    "MeasurementSystem",
    "GageStudy",
    # Investigation (CANON-002 §7, §11) — deprecated, see NB-001
    "Investigation",
    "InvestigationMembership",
    "InvestigationToolLink",
    # Notebook (NB-001)
    "Notebook",
    "Trial",
    "NotebookPage",
    "TrialToolLink",
    "TrialAction",
    "HanseiKai",
    "Yokoten",
    "YokotenAdoption",
]

"""
Workbench models for the unified inquiry workspace.

MIGRATION NOTE:
- Project, Hypothesis, Evidence models here are being consolidated to core.models
- core.Project, core.Hypothesis, core.Evidence are the canonical models
- All probability updates now use core.bayesian.BayesianUpdater
- See MIGRATION_PLAN.md for consolidation roadmap

KEEP: Workbench, Artifact, KnowledgeGraph (UI state, not hypothesis data)
DEPRECATE: Project, Hypothesis, Evidence (use core.models instead)

Hierarchy:
- Project: A hypothesis being investigated (e.g., "Temperature causes defects")
- Workbench: An individual analysis/experiment within a project
- Artifact: A unit of work (data, analysis, note, evidence)
- KnowledgeGraph: Causal model linking evidence to hypothesis

Workbenches serialize to JSON for agent context on load.
"""

import uuid
from datetime import datetime
from django.conf import settings
from django.db import models


class Project(models.Model):
    """
    A Project represents a hypothesis under investigation.

    Projects contain multiple Workbenches (analyses, experiments) that
    gather evidence supporting or refuting the hypothesis.
    """

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        COMPLETED = "completed", "Completed"
        ARCHIVED = "archived", "Archived"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="projects",
    )

    # The hypothesis being investigated
    title = models.CharField(
        max_length=255,
        help_text="Short title for the project"
    )
    hypothesis = models.TextField(
        help_text="The hypothesis being investigated"
    )
    description = models.TextField(
        blank=True,
        help_text="Additional context, background, or scope"
    )

    # Domain for context
    domain = models.CharField(
        max_length=50,
        blank=True,
        help_text="Domain area (manufacturing, software, healthcare, etc.)"
    )

    # Status tracking
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE,
    )

    # Conclusion when project is completed
    conclusion = models.TextField(
        blank=True,
        help_text="Final conclusion about the hypothesis"
    )
    conclusion_status = models.CharField(
        max_length=20,
        blank=True,
        choices=[
            ("supported", "Hypothesis Supported"),
            ("refuted", "Hypothesis Refuted"),
            ("inconclusive", "Inconclusive"),
            ("modified", "Hypothesis Modified"),
        ],
        help_text="Outcome of the investigation"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "projects"
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["user", "status"]),
            models.Index(fields=["user", "-updated_at"]),
        ]

    def __str__(self):
        return self.title

    def to_dict(self) -> dict:
        """Serialize for API."""
        return {
            "id": str(self.id),
            "title": self.title,
            "hypothesis": self.hypothesis,
            "description": self.description,
            "domain": self.domain,
            "status": self.status,
            "conclusion": self.conclusion,
            "conclusion_status": self.conclusion_status,
            "workbench_count": self.workbenches.count(),
            "hypothesis_count": self.hypotheses.count() if hasattr(self, 'hypotheses') else 0,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Hypothesis(models.Model):
    """
    A testable hypothesis within a project.

    Hypotheses can be linked to each other via the KnowledgeGraph
    and weighted by evidence from analyses.
    """

    class Status(models.TextChoices):
        INVESTIGATING = "investigating", "Investigating"
        SUPPORTED = "supported", "Supported"
        REFUTED = "refuted", "Refuted"
        INCONCLUSIVE = "inconclusive", "Inconclusive"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="hypotheses",
    )

    # The hypothesis statement
    statement = models.TextField(
        help_text="The hypothesis being tested (e.g., 'High temperature causes defects')"
    )
    mechanism = models.TextField(
        blank=True,
        help_text="How this cause produces the effect"
    )

    # Probability tracking (Bayesian)
    prior_probability = models.FloatField(
        default=0.5,
        help_text="Initial probability estimate (0-1)"
    )
    current_probability = models.FloatField(
        default=0.5,
        help_text="Current probability after evidence updates"
    )

    # Status
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.INVESTIGATING,
    )

    # Conclusion notes
    conclusion_notes = models.TextField(
        blank=True,
        help_text="Notes on why this hypothesis was supported/refuted"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "hypotheses"
        ordering = ["-current_probability", "-created_at"]
        verbose_name_plural = "hypotheses"

    def __str__(self):
        return f"{self.statement[:50]}... ({self.current_probability:.0%})"

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "statement": self.statement,
            "mechanism": self.mechanism,
            "prior_probability": self.prior_probability,
            "current_probability": self.current_probability,
            "status": self.status,
            "conclusion_notes": self.conclusion_notes,
            "evidence_count": self.evidence.count() if hasattr(self, 'evidence') else 0,
            "conversation_count": self.conversations.count() if hasattr(self, 'conversations') else 0,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def update_probability(self, likelihood_ratio: float, confidence: float = 1.0):
        """
        Update probability using Bayes' theorem via core.bayesian.BayesianUpdater.
        likelihood_ratio = P(evidence|H) / P(evidence|not H)
        """
        from core.bayesian import get_updater

        updater = get_updater()
        result = updater.update(
            prior=self.current_probability,
            likelihood_ratio=likelihood_ratio,
            confidence=confidence,
        )
        self.current_probability = result.posterior_probability
        self.save(update_fields=["current_probability", "updated_at"])


class Evidence(models.Model):
    """
    Evidence linked to a hypothesis.

    Evidence can support or weaken a hypothesis and comes from
    various sources (DSW analysis, simulation, observation, etc.)
    """

    class EvidenceType(models.TextChoices):
        STATISTICAL = "statistical", "Statistical Analysis"
        SIMULATION = "simulation", "Simulation Result"
        OBSERVATION = "observation", "Observation"
        EXPERIMENT = "experiment", "Experiment"
        RESEARCH = "research", "Research/Literature"
        EXPERT = "expert", "Expert Opinion"

    class Direction(models.TextChoices):
        SUPPORTS = "supports", "Supports"
        WEAKENS = "weakens", "Weakens"
        NEUTRAL = "neutral", "Neutral"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    hypothesis = models.ForeignKey(
        Hypothesis,
        on_delete=models.CASCADE,
        related_name="evidence",
    )

    # Evidence details
    summary = models.TextField(
        help_text="Summary of the evidence"
    )
    evidence_type = models.CharField(
        max_length=20,
        choices=EvidenceType.choices,
        default=EvidenceType.OBSERVATION,
    )
    direction = models.CharField(
        max_length=10,
        choices=Direction.choices,
        default=Direction.SUPPORTS,
    )
    strength = models.FloatField(
        default=0.5,
        help_text="Strength of evidence (0-1)"
    )

    # Source tracking
    source = models.CharField(
        max_length=100,
        blank=True,
        help_text="Where this evidence came from (e.g., 'DSW regression', 'Simulator')"
    )
    source_artifact_id = models.UUIDField(
        null=True,
        blank=True,
        help_text="Link to source artifact if from a workbench"
    )

    # Raw data/details (JSON)
    details = models.JSONField(
        default=dict,
        blank=True,
        help_text="Detailed data (statistics, parameters, etc.)"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "evidence"
        ordering = ["-created_at"]
        verbose_name_plural = "evidence"

    def __str__(self):
        return f"{self.direction}: {self.summary[:50]}..."

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "hypothesis_id": str(self.hypothesis_id),
            "summary": self.summary,
            "evidence_type": self.evidence_type,
            "direction": self.direction,
            "strength": self.strength,
            "source": self.source,
            "details": self.details,
            "created_at": self.created_at.isoformat(),
        }


class Conversation(models.Model):
    """
    A saved conversation thread for a hypothesis.

    Users can have multiple conversations per hypothesis,
    exploring different aspects or approaches.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    hypothesis = models.ForeignKey(
        Hypothesis,
        on_delete=models.CASCADE,
        related_name="conversations",
    )

    # Conversation metadata
    title = models.CharField(
        max_length=255,
        default="New Conversation",
        help_text="Title for this conversation thread"
    )

    # Messages stored as JSON array
    # [{role: "user"|"assistant", content: "...", timestamp: "..."}]
    messages = models.JSONField(
        default=list,
        help_text="Conversation messages"
    )

    # Context pulled into this conversation
    context = models.JSONField(
        default=dict,
        blank=True,
        help_text="Context snapshot (hypothesis state, evidence summary, etc.)"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "hypothesis_conversations"
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.title} ({len(self.messages)} messages)"

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "hypothesis_id": str(self.hypothesis_id),
            "title": self.title,
            "message_count": len(self.messages),
            "messages": self.messages,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self.save(update_fields=["messages", "updated_at"])

    def build_context(self) -> dict:
        """Build context snapshot from hypothesis and evidence."""
        hypothesis = self.hypothesis
        return {
            "hypothesis": {
                "statement": hypothesis.statement,
                "mechanism": hypothesis.mechanism,
                "probability": hypothesis.current_probability,
                "status": hypothesis.status,
            },
            "evidence": [e.to_dict() for e in hypothesis.evidence.all()[:10]],
            "project": {
                "title": hypothesis.project.title,
                "domain": hypothesis.project.domain,
            },
        }


class Workbench(models.Model):
    """
    A unified workspace for inquiry-based work.

    Replaces the separate Problem, DSW, SPC, Experimenter pages
    with a single canvas-based workspace.
    """

    class Template(models.TextChoices):
        BLANK = "blank", "Blank"
        DMAIC = "dmaic", "DMAIC"
        KAIZEN = "kaizen", "Kaizen Event"
        EIGHT_D = "8d", "8D Report"
        A3 = "a3", "A3 Problem Solving"

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        COMPLETED = "completed", "Completed"
        ARCHIVED = "archived", "Archived"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="workbenches",
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="workbenches",
        null=True,
        blank=True,
        help_text="Project this workbench belongs to (optional)"
    )

    # Basic info
    title = models.CharField(
        max_length=255,
        help_text="The inquiry or question being investigated"
    )
    description = models.TextField(
        blank=True,
        help_text="Additional context about the inquiry"
    )
    template = models.CharField(
        max_length=20,
        choices=Template.choices,
        default=Template.BLANK,
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE,
    )

    # Template-specific state
    # For DMAIC: current phase, phase history
    # For Kaizen: current day, event dates
    # For 8D: current discipline
    template_state = models.JSONField(
        default=dict,
        blank=True,
        help_text="Template-specific state (phases, days, disciplines)"
    )

    # Artifact connections: [{from_id, to_id, label}]
    # Stored here rather than on artifacts for easier bulk operations
    connections = models.JSONField(
        default=list,
        blank=True,
        help_text="Connections between artifacts"
    )

    # Canvas layout: {artifact_id: {x, y, width, height, collapsed}}
    # Allows restoring exact visual state
    layout = models.JSONField(
        default=dict,
        blank=True,
        help_text="Canvas layout positions for artifacts"
    )

    # Datasets associated with this workbench
    # [{name, file_path, rows, cols, variables: [{name, dtype, role}]}]
    datasets = models.JSONField(
        default=list,
        blank=True,
        help_text="Datasets loaded in this workbench"
    )

    # AI Guide observations
    # [{timestamp, observation, suggestion, acknowledged}]
    guide_observations = models.JSONField(
        default=list,
        blank=True,
        help_text="AI Guide observations and suggestions"
    )

    # Conclusion / structured response (when complete)
    conclusion = models.TextField(
        blank=True,
        help_text="The structured response / conclusion reached"
    )
    conclusion_confidence = models.CharField(
        max_length=20,
        blank=True,
        help_text="Confidence in the conclusion"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "workbenches"
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["user", "status"]),
            models.Index(fields=["user", "-updated_at"]),
        ]

    def __str__(self):
        return f"{self.title} ({self.template})"

    # =========================================================================
    # Serialization (for agent context)
    # =========================================================================

    def to_json(self) -> dict:
        """
        Serialize workbench to JSON for agent context.

        This JSON becomes the system prompt when loading a workbench,
        giving agents full continuity.
        """
        return {
            "id": str(self.id),
            "inquiry": self.title,
            "description": self.description,
            "template": self.template,
            "status": self.status,
            "template_state": self.template_state,
            "artifacts": [a.to_json() for a in self.artifacts.all()],
            "connections": self.connections,
            "datasets": self.datasets,
            "guide_observations": self.guide_observations,
            "conclusion": self.conclusion,
            "created": self.created_at.isoformat(),
            "updated": self.updated_at.isoformat(),
        }

    @classmethod
    def from_json(cls, data: dict, user) -> "Workbench":
        """Create a workbench from JSON (for import)."""
        workbench = cls.objects.create(
            user=user,
            title=data.get("inquiry", "Imported Workbench"),
            description=data.get("description", ""),
            template=data.get("template", cls.Template.BLANK),
            template_state=data.get("template_state", {}),
            connections=data.get("connections", []),
            datasets=data.get("datasets", []),
            guide_observations=data.get("guide_observations", []),
        )

        # Create artifacts
        for artifact_data in data.get("artifacts", []):
            Artifact.from_json(artifact_data, workbench)

        return workbench

    # =========================================================================
    # Template helpers
    # =========================================================================

    def init_template(self):
        """Initialize template-specific state."""
        if self.template == self.Template.DMAIC:
            self.template_state = {
                "current_phase": "define",
                "phase_history": [{
                    "phase": "define",
                    "entered_at": datetime.now().isoformat(),
                    "notes": "Started DMAIC project"
                }],
                "phase_artifacts": {
                    "define": [],
                    "measure": [],
                    "analyze": [],
                    "improve": [],
                    "control": [],
                }
            }
        elif self.template == self.Template.KAIZEN:
            self.template_state = {
                "current_day": 1,
                "event_start": datetime.now().isoformat(),
                "sections": {
                    "current_state": [],
                    "waste_identification": [],
                    "future_state": [],
                    "action_items": [],
                    "results": [],
                }
            }
        elif self.template == self.Template.EIGHT_D:
            self.template_state = {
                "current_discipline": "d1",
                "disciplines": {
                    "d1": {"title": "Team Formation", "artifacts": [], "complete": False},
                    "d2": {"title": "Problem Description", "artifacts": [], "complete": False},
                    "d3": {"title": "Containment Actions", "artifacts": [], "complete": False},
                    "d4": {"title": "Root Cause Analysis", "artifacts": [], "complete": False},
                    "d5": {"title": "Corrective Actions", "artifacts": [], "complete": False},
                    "d6": {"title": "Implementation", "artifacts": [], "complete": False},
                    "d7": {"title": "Prevention", "artifacts": [], "complete": False},
                    "d8": {"title": "Congratulate Team", "artifacts": [], "complete": False},
                }
            }
        elif self.template == self.Template.A3:
            self.template_state = {
                "sections": {
                    "background": [],
                    "current_condition": [],
                    "goal": [],
                    "root_cause": [],
                    "countermeasures": [],
                    "implementation": [],
                    "follow_up": [],
                }
            }
        self.save(update_fields=["template_state"])

    def advance_dmaic_phase(self, notes: str = ""):
        """Advance to next DMAIC phase."""
        if self.template != self.Template.DMAIC:
            return

        phases = ["define", "measure", "analyze", "improve", "control"]
        current = self.template_state.get("current_phase", "define")

        try:
            current_idx = phases.index(current)
            if current_idx < len(phases) - 1:
                next_phase = phases[current_idx + 1]
                self.template_state["current_phase"] = next_phase
                self.template_state["phase_history"].append({
                    "phase": next_phase,
                    "entered_at": datetime.now().isoformat(),
                    "notes": notes,
                })
                self.save(update_fields=["template_state", "updated_at"])
        except ValueError:
            pass

    # =========================================================================
    # Artifact management
    # =========================================================================

    def add_artifact(self, artifact_type: str, content: dict, **kwargs) -> "Artifact":
        """Add an artifact to this workbench."""
        artifact = Artifact.objects.create(
            workbench=self,
            artifact_type=artifact_type,
            content=content,
            **kwargs
        )

        # Auto-assign to template section if applicable
        self._assign_to_template(artifact)

        return artifact

    def _assign_to_template(self, artifact: "Artifact"):
        """Auto-assign artifact to template section based on type/context."""
        if self.template == self.Template.DMAIC:
            phase = self.template_state.get("current_phase", "define")
            if phase in self.template_state.get("phase_artifacts", {}):
                self.template_state["phase_artifacts"][phase].append(str(artifact.id))
                self.save(update_fields=["template_state"])

    def connect_artifacts(self, from_id: str, to_id: str, label: str = ""):
        """Connect two artifacts."""
        self.connections.append({
            "from": from_id,
            "to": to_id,
            "label": label,
        })
        self.save(update_fields=["connections", "updated_at"])

    def add_guide_observation(self, observation: str, suggestion: str = ""):
        """Add an AI Guide observation."""
        self.guide_observations.append({
            "timestamp": datetime.now().isoformat(),
            "observation": observation,
            "suggestion": suggestion,
            "acknowledged": False,
        })
        self.save(update_fields=["guide_observations", "updated_at"])


class Artifact(models.Model):
    """
    A single artifact in a workbench.

    Artifacts are the units of work: data analyses, hypotheses, notes,
    charts, models, documents, etc.
    """

    class ArtifactType(models.TextChoices):
        # Data
        DATASET = "dataset", "Dataset"
        CLEANED_DATASET = "cleaned_dataset", "Cleaned Dataset"
        GENERATED_DATASET = "generated_dataset", "Generated Dataset"

        # Analysis
        SPC_CHART = "spc_chart", "SPC Chart"
        CAPABILITY_STUDY = "capability_study", "Capability Study"
        ANOVA = "anova", "ANOVA"
        REGRESSION = "regression", "Regression"
        CORRELATION = "correlation", "Correlation"
        DESCRIPTIVE_STATS = "descriptive_stats", "Descriptive Statistics"
        HYPOTHESIS_TEST = "hypothesis_test", "Hypothesis Test"
        FORECAST = "forecast", "Forecast"

        # Experiment
        DOE_DESIGN = "doe_design", "DOE Design"
        DOE_RESULTS = "doe_results", "DOE Results"

        # ML
        ML_MODEL = "ml_model", "ML Model"
        MODEL_METRICS = "model_metrics", "Model Metrics"
        FEATURE_IMPORTANCE = "feature_importance", "Feature Importance"

        # Thinking
        NOTE = "note", "Note"
        HYPOTHESIS = "hypothesis", "Hypothesis"
        EVIDENCE = "evidence", "Evidence"
        CONCLUSION = "conclusion", "Conclusion"

        # Documents
        REPORT = "report", "Report"
        SUMMARY = "summary", "Summary"
        CONTROL_PLAN = "control_plan", "Control Plan"

        # Code
        SCRIPT = "script", "Script"
        VISUALIZATION = "visualization", "Visualization"

        # Research
        RESEARCH_FINDINGS = "research_findings", "Research Findings"
        CITATION = "citation", "Citation"

        # Process (for Kaizen/DMAIC)
        PROCESS_MAP = "process_map", "Process Map"
        FISHBONE = "fishbone", "Fishbone Diagram"
        FIVE_WHYS = "five_whys", "5 Whys Analysis"
        ACTION_ITEM = "action_item", "Action Item"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workbench = models.ForeignKey(
        Workbench,
        on_delete=models.CASCADE,
        related_name="artifacts",
    )

    # Type and content
    artifact_type = models.CharField(
        max_length=30,
        choices=ArtifactType.choices,
    )
    title = models.CharField(max_length=255, blank=True)
    content = models.JSONField(
        default=dict,
        help_text="Type-specific content (data, results, text, etc.)"
    )

    # Source tracking
    source = models.CharField(
        max_length=50,
        blank=True,
        help_text="What created this: user, analyst, researcher, coder, etc."
    )
    source_artifact_id = models.UUIDField(
        null=True,
        blank=True,
        help_text="If derived from another artifact, link to source"
    )

    # For hypothesis artifacts
    probability = models.FloatField(
        null=True,
        blank=True,
        help_text="Probability estimate (for hypotheses)"
    )

    # For evidence artifacts
    supports_hypotheses = models.JSONField(
        default=list,
        blank=True,
        help_text="Hypothesis IDs this evidence supports"
    )
    weakens_hypotheses = models.JSONField(
        default=list,
        blank=True,
        help_text="Hypothesis IDs this evidence weakens"
    )

    # For model artifacts
    model_path = models.CharField(
        max_length=500,
        blank=True,
        help_text="Path to saved model file"
    )

    # Tags for filtering/grouping
    tags = models.JSONField(
        default=list,
        blank=True,
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "artifacts"
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["workbench", "artifact_type"]),
            models.Index(fields=["workbench", "created_at"]),
        ]

    def __str__(self):
        return f"{self.artifact_type}: {self.title or self.id}"

    def to_json(self) -> dict:
        """Serialize artifact for workbench JSON export."""
        data = {
            "id": str(self.id),
            "type": self.artifact_type,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "tags": self.tags,
            "created": self.created_at.isoformat(),
        }

        if self.probability is not None:
            data["probability"] = self.probability
        if self.supports_hypotheses:
            data["supports"] = self.supports_hypotheses
        if self.weakens_hypotheses:
            data["weakens"] = self.weakens_hypotheses
        if self.model_path:
            data["model_path"] = self.model_path
        if self.source_artifact_id:
            data["source_artifact"] = str(self.source_artifact_id)

        return data

    @classmethod
    def from_json(cls, data: dict, workbench: Workbench) -> "Artifact":
        """Create an artifact from JSON (for import)."""
        return cls.objects.create(
            workbench=workbench,
            artifact_type=data.get("type", cls.ArtifactType.NOTE),
            title=data.get("title", ""),
            content=data.get("content", {}),
            source=data.get("source", "import"),
            probability=data.get("probability"),
            supports_hypotheses=data.get("supports", []),
            weakens_hypotheses=data.get("weakens", []),
            model_path=data.get("model_path", ""),
            tags=data.get("tags", []),
        )

    # =========================================================================
    # Type-specific helpers
    # =========================================================================

    def is_thinking_artifact(self) -> bool:
        """Check if this is a thinking artifact (note, hypothesis, evidence, conclusion)."""
        return self.artifact_type in [
            self.ArtifactType.NOTE,
            self.ArtifactType.HYPOTHESIS,
            self.ArtifactType.EVIDENCE,
            self.ArtifactType.CONCLUSION,
        ]

    def is_analysis_artifact(self) -> bool:
        """Check if this is an analysis artifact."""
        return self.artifact_type in [
            self.ArtifactType.SPC_CHART,
            self.ArtifactType.CAPABILITY_STUDY,
            self.ArtifactType.ANOVA,
            self.ArtifactType.REGRESSION,
            self.ArtifactType.CORRELATION,
            self.ArtifactType.DESCRIPTIVE_STATS,
            self.ArtifactType.HYPOTHESIS_TEST,
            self.ArtifactType.FORECAST,
        ]

    def update_hypothesis_probability(self, new_probability: float):
        """Update probability for hypothesis artifacts."""
        if self.artifact_type != self.ArtifactType.HYPOTHESIS:
            return
        self.probability = max(0.0, min(1.0, new_probability))
        self.save(update_fields=["probability", "updated_at"])


class KnowledgeGraph(models.Model):
    """
    The Knowledge Graph represents the current worldview as weighted causal vectors.

    Structure:
    - Nodes are artifacts (hypotheses, causes, effects, observations)
    - Edges are causal vectors with weights (strength of causal relationship)
    - Weights on edges, not nodes: P(effect | cause)

    The graph can be traversed in either direction:
    - cause → effect (forward inference)
    - effect → cause (backward inference / diagnosis)
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="knowledge_graphs",
    )
    project = models.OneToOneField(
        Project,
        on_delete=models.CASCADE,
        related_name="knowledge_graph",
        null=True,
        blank=True,
        help_text="Project this graph belongs to"
    )
    workbench = models.ForeignKey(
        Workbench,
        on_delete=models.SET_NULL,
        related_name="knowledge_graphs",
        null=True,
        blank=True,
        help_text="Workbench that created/modified this (for tracking)"
    )

    title = models.CharField(max_length=255, default="Knowledge Graph")
    description = models.TextField(blank=True)

    # Graph state serialized
    # nodes: [{id, type, label, artifact_id?, metadata}]
    nodes = models.JSONField(
        default=list,
        blank=True,
        help_text="Graph nodes (hypotheses, causes, effects)"
    )

    # edges: [{id, from_node, to_node, weight, mechanism, evidence_ids, metadata}]
    # Weight = P(to | from), strength of causal relationship
    edges = models.JSONField(
        default=list,
        blank=True,
        help_text="Causal edges with weights"
    )

    # Expansion signals: when evidence doesn't fit any node
    # [{id, evidence, likelihoods, status, resolution}]
    expansion_signals = models.JSONField(
        default=list,
        blank=True,
        help_text="Signals indicating incomplete causal surface"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "knowledge_graphs"
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.title} ({len(self.nodes)} nodes, {len(self.edges)} edges)"

    # =========================================================================
    # Node operations
    # =========================================================================

    def add_node(
        self,
        node_type: str,
        label: str,
        artifact_id: str = None,
        metadata: dict = None
    ) -> dict:
        """Add a node to the graph."""
        node = {
            "id": str(uuid.uuid4())[:8],
            "type": node_type,  # hypothesis, cause, effect, observation
            "label": label,
            "artifact_id": artifact_id,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        }
        self.nodes.append(node)
        self.save(update_fields=["nodes", "updated_at"])
        return node

    def get_node(self, node_id: str) -> dict:
        """Get a node by ID."""
        return next((n for n in self.nodes if n["id"] == node_id), None)

    def remove_node(self, node_id: str):
        """Remove a node and all connected edges."""
        self.nodes = [n for n in self.nodes if n["id"] != node_id]
        self.edges = [
            e for e in self.edges
            if e["from_node"] != node_id and e["to_node"] != node_id
        ]
        self.save(update_fields=["nodes", "edges", "updated_at"])

    # =========================================================================
    # Edge operations (causal vectors)
    # =========================================================================

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        weight: float = 0.5,
        mechanism: str = "",
        evidence_ids: list = None,
        metadata: dict = None
    ) -> dict:
        """
        Add a causal edge between nodes.

        weight: P(to | from) - strength of causal relationship (0.0 - 1.0)
        mechanism: Description of how the cause produces the effect
        """
        edge = {
            "id": str(uuid.uuid4())[:8],
            "from_node": from_node,
            "to_node": to_node,
            "weight": max(0.0, min(1.0, weight)),
            "mechanism": mechanism,
            "evidence_ids": evidence_ids or [],
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
        }
        self.edges.append(edge)
        self.save(update_fields=["edges", "updated_at"])
        return edge

    def get_edge(self, edge_id: str) -> dict:
        """Get an edge by ID."""
        return next((e for e in self.edges if e["id"] == edge_id), None)

    def get_edges_from(self, node_id: str) -> list:
        """Get all edges originating from a node (effects of this cause)."""
        return [e for e in self.edges if e["from_node"] == node_id]

    def get_edges_to(self, node_id: str) -> list:
        """Get all edges pointing to a node (causes of this effect)."""
        return [e for e in self.edges if e["to_node"] == node_id]

    def update_edge_weight(
        self,
        edge_id: str,
        new_weight: float,
        evidence_id: str = None
    ):
        """Update edge weight based on new evidence."""
        for edge in self.edges:
            if edge["id"] == edge_id:
                edge["weight"] = max(0.0, min(1.0, new_weight))
                if evidence_id:
                    edge["evidence_ids"].append(evidence_id)
                self.save(update_fields=["edges", "updated_at"])
                return

    # =========================================================================
    # Bayesian inference
    # =========================================================================

    def update_from_evidence(
        self,
        evidence_id: str,
        supports: list[tuple[str, float]],  # [(edge_id, likelihood_ratio), ...]
        weakens: list[tuple[str, float]],
    ):
        """
        Update edge weights based on new evidence using core.bayesian.BayesianUpdater.

        supports: edges this evidence supports with likelihood_ratio (LR > 1)
        weakens: edges this evidence weakens with likelihood_ratio (LR < 1)

        NOTE: For backwards compatibility, if likelihood values are passed as
        0-1 range (old API), they're converted to likelihood ratios.
        """
        from core.bayesian import get_updater

        updater = get_updater()

        for edge_id, lr in supports:
            edge = self.get_edge(edge_id)
            if edge:
                prior = edge["weight"]
                # Convert old likelihood (0-1) to LR if needed
                if 0 < lr <= 1:
                    lr = 1 + lr  # Convert 0.5 -> 1.5, 1.0 -> 2.0
                result = updater.update(prior=prior, likelihood_ratio=lr)
                self.update_edge_weight(edge_id, result.posterior_probability, evidence_id)

        for edge_id, lr in weakens:
            edge = self.get_edge(edge_id)
            if edge:
                prior = edge["weight"]
                # Convert old likelihood (0-1) to LR if needed
                if 0 < lr <= 1:
                    lr = 1 - lr * 0.5  # Convert 0.5 -> 0.75, 1.0 -> 0.5
                result = updater.update(prior=prior, likelihood_ratio=lr)
                self.update_edge_weight(edge_id, result.posterior_probability, evidence_id)

    def check_expansion(self, evidence_likelihoods: dict[str, float]) -> dict:
        """
        Check if evidence fits any existing hypothesis.

        If max likelihood < threshold, signal expansion needed.
        Returns expansion signal if needed, None otherwise.
        """
        max_likelihood = max(evidence_likelihoods.values()) if evidence_likelihoods else 0

        if max_likelihood < 0.2:  # Threshold for expansion
            signal = {
                "id": str(uuid.uuid4())[:8],
                "likelihoods": evidence_likelihoods,
                "max_likelihood": max_likelihood,
                "status": "pending",
                "message": "Evidence doesn't fit existing hypotheses - causal surface may be incomplete",
                "created_at": datetime.now().isoformat(),
            }
            self.expansion_signals.append(signal)
            self.save(update_fields=["expansion_signals", "updated_at"])
            return signal
        return None

    # =========================================================================
    # Graph traversal
    # =========================================================================

    def get_causal_chain(self, from_node: str, to_node: str) -> list[list[dict]]:
        """Find all causal chains from one node to another."""
        chains = []
        self._find_chains(from_node, to_node, [], chains)
        return chains

    def _find_chains(
        self,
        current: str,
        target: str,
        path: list,
        chains: list,
        visited: set = None
    ):
        """Recursive helper for finding chains."""
        if visited is None:
            visited = set()

        if current in visited:
            return
        visited.add(current)

        if current == target:
            chains.append(path.copy())
            return

        for edge in self.get_edges_from(current):
            path.append(edge)
            self._find_chains(edge["to_node"], target, path, chains, visited)
            path.pop()

    def get_upstream_causes(self, node_id: str, depth: int = 3) -> list[dict]:
        """Get all causes upstream of a node within depth."""
        causes = []
        self._get_upstream(node_id, depth, causes, set())
        return causes

    def _get_upstream(
        self,
        node_id: str,
        depth: int,
        causes: list,
        visited: set
    ):
        """Recursive helper for upstream traversal."""
        if depth <= 0 or node_id in visited:
            return
        visited.add(node_id)

        for edge in self.get_edges_to(node_id):
            from_node = self.get_node(edge["from_node"])
            if from_node:
                causes.append({
                    "node": from_node,
                    "edge": edge,
                    "depth": depth,
                })
                self._get_upstream(edge["from_node"], depth - 1, causes, visited)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict:
        """Serialize for API/export."""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "nodes": self.nodes,
            "edges": self.edges,
            "expansion_signals": self.expansion_signals,
            "project_id": str(self.project_id) if self.project_id else None,
            "workbench_id": str(self.workbench_id) if self.workbench_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class EpistemicLog(models.Model):
    """
    Synara-Meta: Log epistemic processes to learn reasoning patterns.

    Logs not chat data, but epistemic motion:
    - How hypotheses were formulated
    - What evidence shifted beliefs
    - When expansions occurred
    - Which reasoning paths led to insight
    - Which led to dead ends

    This is the learning layer for improving inquiry over time.
    """

    class EventType(models.TextChoices):
        # Hypothesis events
        HYPOTHESIS_CREATED = "hypothesis_created", "Hypothesis Created"
        HYPOTHESIS_UPDATED = "hypothesis_updated", "Hypothesis Updated"
        HYPOTHESIS_REJECTED = "hypothesis_rejected", "Hypothesis Rejected"
        HYPOTHESIS_CONFIRMED = "hypothesis_confirmed", "Hypothesis Confirmed"

        # Evidence events
        EVIDENCE_ADDED = "evidence_added", "Evidence Added"
        EVIDENCE_LINKED = "evidence_linked", "Evidence Linked to Hypothesis"

        # Belief events
        BELIEF_UPDATE = "belief_update", "Belief Updated"
        BELIEF_PROPAGATION = "belief_propagation", "Belief Propagated"

        # Expansion events
        EXPANSION_SIGNAL = "expansion_signal", "Expansion Signal Triggered"
        EXPANSION_RESOLVED = "expansion_resolved", "Expansion Resolved"

        # Graph events
        NODE_ADDED = "node_added", "Node Added to Graph"
        EDGE_ADDED = "edge_added", "Edge Added to Graph"
        EDGE_WEIGHT_UPDATED = "edge_weight_updated", "Edge Weight Updated"

        # Reasoning events
        CAUSAL_CHAIN_TRACED = "causal_chain_traced", "Causal Chain Traced"
        DEAD_END_IDENTIFIED = "dead_end_identified", "Dead End Identified"
        INSIGHT_RECORDED = "insight_recorded", "Insight Recorded"

        # Session events
        INQUIRY_STARTED = "inquiry_started", "Inquiry Started"
        INQUIRY_RESOLVED = "inquiry_resolved", "Inquiry Resolved"
        METHODOLOGY_SELECTED = "methodology_selected", "Methodology Selected"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="epistemic_logs",
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="epistemic_logs",
        null=True,
        blank=True,
    )
    workbench = models.ForeignKey(
        Workbench,
        on_delete=models.CASCADE,
        related_name="epistemic_logs",
        null=True,
        blank=True,
    )
    knowledge_graph = models.ForeignKey(
        KnowledgeGraph,
        on_delete=models.CASCADE,
        related_name="epistemic_logs",
        null=True,
        blank=True,
    )

    # Event details
    event_type = models.CharField(
        max_length=30,
        choices=EventType.choices,
    )
    event_data = models.JSONField(
        default=dict,
        help_text="Event-specific data (before/after states, deltas, etc.)"
    )

    # Context at time of event
    context = models.JSONField(
        default=dict,
        blank=True,
        help_text="Snapshot of relevant context when event occurred"
    )

    # Outcome tracking (for learning)
    led_to_insight = models.BooleanField(
        null=True,
        blank=True,
        help_text="Did this event lead to insight? (marked retrospectively)"
    )
    led_to_dead_end = models.BooleanField(
        null=True,
        blank=True,
        help_text="Did this event lead to dead end? (marked retrospectively)"
    )

    # Source
    source = models.CharField(
        max_length=50,
        default="system",
        help_text="What triggered this event: user, guide, analyst, system, etc."
    )

    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "epistemic_logs"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "event_type"]),
            models.Index(fields=["workbench", "-created_at"]),
            models.Index(fields=["knowledge_graph", "-created_at"]),
            models.Index(fields=["event_type", "-created_at"]),
        ]

    def __str__(self):
        return f"{self.event_type} at {self.created_at}"

    @classmethod
    def log(
        cls,
        user,
        event_type: str,
        event_data: dict,
        workbench=None,
        knowledge_graph=None,
        context: dict = None,
        source: str = "system"
    ) -> "EpistemicLog":
        """Convenience method to create a log entry."""
        return cls.objects.create(
            user=user,
            workbench=workbench,
            knowledge_graph=knowledge_graph,
            event_type=event_type,
            event_data=event_data,
            context=context or {},
            source=source,
        )

    @classmethod
    def log_hypothesis_created(cls, user, workbench, hypothesis_data: dict, source: str = "user"):
        """Log hypothesis creation."""
        return cls.log(
            user=user,
            event_type=cls.EventType.HYPOTHESIS_CREATED,
            event_data=hypothesis_data,
            workbench=workbench,
            source=source,
        )

    @classmethod
    def log_evidence_added(cls, user, workbench, evidence_data: dict, source: str = "user"):
        """Log evidence addition."""
        return cls.log(
            user=user,
            event_type=cls.EventType.EVIDENCE_ADDED,
            event_data=evidence_data,
            workbench=workbench,
            source=source,
        )

    @classmethod
    def log_belief_update(
        cls,
        user,
        knowledge_graph,
        edge_id: str,
        old_weight: float,
        new_weight: float,
        evidence_id: str = None
    ):
        """Log belief update on an edge."""
        return cls.log(
            user=user,
            event_type=cls.EventType.BELIEF_UPDATE,
            event_data={
                "edge_id": edge_id,
                "old_weight": old_weight,
                "new_weight": new_weight,
                "delta": new_weight - old_weight,
                "evidence_id": evidence_id,
            },
            knowledge_graph=knowledge_graph,
            source="synara",
        )

    @classmethod
    def log_expansion_signal(cls, user, knowledge_graph, signal_data: dict):
        """Log expansion signal (incomplete causal surface)."""
        return cls.log(
            user=user,
            event_type=cls.EventType.EXPANSION_SIGNAL,
            event_data=signal_data,
            knowledge_graph=knowledge_graph,
            source="synara",
        )

    def mark_outcome(self, led_to_insight: bool = None, led_to_dead_end: bool = None):
        """Mark the outcome of this event retrospectively."""
        if led_to_insight is not None:
            self.led_to_insight = led_to_insight
        if led_to_dead_end is not None:
            self.led_to_dead_end = led_to_dead_end
        self.save(update_fields=["led_to_insight", "led_to_dead_end"])

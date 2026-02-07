"""Agents API models."""

import uuid

from django.conf import settings
from django.db import models


class Workflow(models.Model):
    """User-defined workflow (chain of agents)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="workflows",
    )
    name = models.CharField(max_length=255)
    steps = models.TextField()  # JSON array of step definitions
    created_at = models.DateTimeField(auto_now_add=True)
    last_run = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.user.username})"


class DSWResult(models.Model):
    """Stored result from DSW pipeline run."""

    id = models.CharField(max_length=50, primary_key=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="dsw_results",
    )
    result_type = models.CharField(max_length=50)  # from_intent, from_data
    data = models.TextField()  # JSON serialized result
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class TriageResult(models.Model):
    """Stored result from Triage (data cleaning) pipeline."""

    id = models.CharField(max_length=50, primary_key=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="triage_results",
    )
    original_filename = models.CharField(max_length=255)
    cleaned_csv = models.TextField()  # The cleaned CSV data
    report_markdown = models.TextField()  # Cleaning report
    summary_json = models.TextField()  # JSON summary stats
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class SavedModel(models.Model):
    """User's saved ML models from DSW pipeline."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="saved_models",
    )
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    model_type = models.CharField(max_length=100)  # RandomForest, GradientBoosting, etc.
    model_path = models.CharField(max_length=500)  # Path to .pkl file
    dsw_result_id = models.CharField(max_length=50, blank=True)  # Link to DSW result
    metrics = models.TextField(blank=True)  # JSON: accuracy, f1, etc.
    feature_names = models.TextField(blank=True)  # JSON list of features
    target_name = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.model_type})"


class AgentLog(models.Model):
    """Operational log for agent invocations."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="agent_logs",
    )
    agent = models.CharField(max_length=50)  # researcher, coder, writer, etc.
    action = models.CharField(max_length=50)  # invoke, complete, error
    latency_ms = models.IntegerField(null=True)  # Time taken
    success = models.BooleanField(default=True)
    error_message = models.TextField(blank=True)
    metadata = models.TextField(blank=True)  # JSON for extra context
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["agent", "created_at"]),
            models.Index(fields=["user", "created_at"]),
        ]


class Problem(models.Model):
    """
    A problem session for structured problem-solving.

    DEPRECATED: This model stores hypotheses/evidence as JSON blobs.
    New code should use core.Project with core.Hypothesis and core.Evidence
    which use proper FK relationships and BayesianUpdater for probability math.

    See MIGRATION_PLAN.md for consolidation roadmap.

    Users describe an effect they're observing, generate hypotheses about causes,
    gather evidence, and track their evolving understanding until they identify
    probable causes worth pursuing.

    This is the core of the Decision Science Workbench.
    """

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        RESOLVED = "resolved", "Resolved"
        ABANDONED = "abandoned", "Abandoned"

    class DMAICPhase(models.TextChoices):
        """Six Sigma DMAIC phases."""
        DEFINE = "define", "Define"
        MEASURE = "measure", "Measure"
        ANALYZE = "analyze", "Analyze"
        IMPROVE = "improve", "Improve"
        CONTROL = "control", "Control"

    class Methodology(models.TextChoices):
        """Problem-solving methodology."""
        NONE = "none", "None/General"
        DMAIC = "dmaic", "Six Sigma DMAIC"
        DOE = "doe", "Design of Experiments"
        PDCA = "pdca", "Plan-Do-Check-Act"
        A3 = "a3", "A3 Problem Solving"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="problems",
    )

    # Phase 1 migration: link to canonical core.Project
    core_project = models.ForeignKey(
        'core.Project',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='legacy_problems',
        help_text="Canonical core.Project (Phase 1 migration — dual-write)",
    )

    # Basic info
    title = models.CharField(max_length=255)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE,
    )

    # Methodology tracking
    methodology = models.CharField(
        max_length=20,
        choices=Methodology.choices,
        default=Methodology.NONE,
        help_text="Problem-solving methodology being used"
    )
    dmaic_phase = models.CharField(
        max_length=20,
        choices=DMAICPhase.choices,
        blank=True,
        help_text="Current DMAIC phase (if using Six Sigma)"
    )
    phase_history = models.JSONField(
        default=list,
        blank=True,
        help_text="History of phase transitions [{phase, entered_at, notes}]"
    )

    # The Effect (what user observes)
    effect_description = models.TextField(
        help_text="What are you observing? The symptom, outcome, or situation."
    )
    effect_magnitude = models.CharField(
        max_length=100,
        blank=True,
        help_text="How big is the effect? (e.g., '40% increase', 'severe', '$50k loss')"
    )
    effect_first_observed = models.CharField(
        max_length=100,
        blank=True,
        help_text="When did you first notice this?"
    )
    effect_confidence = models.CharField(
        max_length=20,
        default="medium",
        help_text="How confident are you that this effect is real?"
    )

    # Context
    domain = models.CharField(
        max_length=100,
        blank=True,
        help_text="Domain area (e.g., 'manufacturing', 'SaaS', 'healthcare')"
    )
    stakeholders = models.JSONField(
        default=list,
        blank=True,
        help_text="Who is affected or involved?"
    )
    constraints = models.JSONField(
        default=list,
        blank=True,
        help_text="Constraints on the investigation or solution"
    )
    prior_beliefs = models.JSONField(
        default=list,
        blank=True,
        help_text="Initial beliefs about the cause [{belief, confidence}]"
    )
    can_experiment = models.BooleanField(
        default=True,
        help_text="Can you run controlled experiments?"
    )
    available_data = models.TextField(
        blank=True,
        help_text="What data do you have access to?"
    )

    # Hypotheses: [{id, cause, mechanism, probability, testable, evidence_for, evidence_against, status}]
    hypotheses = models.JSONField(
        default=list,
        blank=True,
        help_text="List of causal hypotheses being investigated"
    )

    # Evidence: [{id, type, summary, supports, weakens, source, timestamp, agent, metadata}]
    evidence = models.JSONField(
        default=list,
        blank=True,
        help_text="Evidence gathered from research, analysis, experiments"
    )

    # Dead ends: [{hypothesis_id, hypothesis_text, why_rejected, timestamp}]
    dead_ends = models.JSONField(
        default=list,
        blank=True,
        help_text="Hypotheses we've ruled out"
    )

    # Current understanding
    probable_causes = models.JSONField(
        default=list,
        blank=True,
        help_text="Most likely causes [{cause, probability, confidence}]"
    )
    key_uncertainties = models.JSONField(
        default=list,
        blank=True,
        help_text="What we still don't know"
    )
    recommended_next_steps = models.JSONField(
        default=list,
        blank=True,
        help_text="Suggested next actions"
    )

    # Bias tracking
    bias_warnings = models.JSONField(
        default=list,
        blank=True,
        help_text="Cognitive biases detected [{type, description, timestamp}]"
    )

    # Interview state (for save/resume)
    interview_state = models.JSONField(
        null=True,
        blank=True,
        help_text="Saved interview progress {current_section, current_question, answers, bias_warnings}"
    )

    # Resolution (when status=resolved)
    resolution_summary = models.TextField(
        blank=True,
        help_text="What did we learn? What was the probable cause?"
    )
    resolution_confidence = models.CharField(
        max_length=20,
        blank=True,
        help_text="Confidence in the resolution"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["user", "status"]),
            models.Index(fields=["user", "-updated_at"]),
        ]

    def __str__(self):
        return f"{self.title} ({self.status})"

    def add_hypothesis(self, cause: str, mechanism: str = "", probability: float = 0.5) -> dict:
        """Add a new hypothesis."""
        import uuid as uuid_lib
        from datetime import datetime
        hypothesis = {
            "id": str(uuid_lib.uuid4())[:8],
            "cause": cause,
            "mechanism": mechanism,
            "probability": probability,
            "testable": True,
            "evidence_for": [],
            "evidence_against": [],
            "status": "investigating",
            "created_at": datetime.now().isoformat(),
        }
        self.hypotheses.append(hypothesis)
        self.save(update_fields=["hypotheses", "updated_at"])
        return hypothesis

    def add_evidence(self, summary: str, evidence_type: str = "observation",
                     source: str = "", supports: list = None, weakens: list = None) -> dict:
        """Add evidence and optionally link to hypotheses."""
        import uuid as uuid_lib
        from datetime import datetime

        evidence = {
            "id": str(uuid_lib.uuid4())[:8],
            "type": evidence_type,  # observation, research, data_analysis, experiment
            "summary": summary,
            "source": source,
            "supports": supports or [],
            "weakens": weakens or [],
            "timestamp": datetime.now().isoformat(),
        }
        self.evidence.append(evidence)

        # Update hypothesis probabilities
        if supports or weakens:
            self._update_probabilities(evidence)

        self.save(update_fields=["evidence", "hypotheses", "updated_at"])
        return evidence

    def _update_probabilities(self, evidence: dict):
        """Update hypothesis probabilities based on new evidence.

        Uses core.bayesian.BayesianUpdater for proper Bayesian math.
        Default likelihood ratios:
          - Supporting evidence: LR = 2.0 (moderate support)
          - Weakening evidence: LR = 0.5 (moderate opposition)
        """
        from core.bayesian import get_updater

        updater = get_updater()

        # Default LRs for simple support/weaken classification
        SUPPORT_LR = 2.0   # Moderate support
        WEAKEN_LR = 0.5    # Moderate opposition

        for hyp in self.hypotheses:
            hyp_id = hyp["id"]
            current = hyp["probability"]

            if hyp_id in evidence.get("supports", []):
                # Evidence supports this hypothesis
                result = updater.update(prior=current, likelihood_ratio=SUPPORT_LR)
                hyp["probability"] = result.posterior_probability
                hyp["evidence_for"].append(evidence["id"])

            elif hyp_id in evidence.get("weakens", []):
                # Evidence weakens this hypothesis
                result = updater.update(prior=current, likelihood_ratio=WEAKEN_LR)
                hyp["probability"] = result.posterior_probability
                hyp["evidence_against"].append(evidence["id"])

    def update_understanding(self):
        """Recalculate probable causes from current hypotheses."""
        # Sort by probability
        active_hypotheses = [h for h in self.hypotheses if h.get("status") == "investigating"]
        sorted_hyps = sorted(active_hypotheses, key=lambda h: h["probability"], reverse=True)

        # Top hypotheses become probable causes
        self.probable_causes = [
            {"cause": h["cause"], "probability": h["probability"]}
            for h in sorted_hyps[:3]
            if h["probability"] > 0.2
        ]

        self.save(update_fields=["probable_causes", "updated_at"])

    def reject_hypothesis(self, hypothesis_id: str, reason: str):
        """Move a hypothesis to dead ends."""
        from datetime import datetime

        for hyp in self.hypotheses:
            if hyp["id"] == hypothesis_id:
                hyp["status"] = "rejected"

                self.dead_ends.append({
                    "hypothesis_id": hypothesis_id,
                    "hypothesis_text": hyp["cause"],
                    "why_rejected": reason,
                    "timestamp": datetime.now().isoformat(),
                })

                self.update_understanding()
                self.save(update_fields=["hypotheses", "dead_ends", "probable_causes", "updated_at"])
                return

    def resolve(self, summary: str, confidence: str = "medium"):
        """Mark problem as resolved."""
        self.status = self.Status.RESOLVED
        self.resolution_summary = summary
        self.resolution_confidence = confidence
        self.save(update_fields=["status", "resolution_summary", "resolution_confidence", "updated_at"])

    def set_methodology(self, methodology: str):
        """Set the problem-solving methodology."""
        self.methodology = methodology
        if methodology == self.Methodology.DMAIC:
            self.advance_phase(self.DMAICPhase.DEFINE, "Started DMAIC process")
        self.save(update_fields=["methodology", "dmaic_phase", "phase_history", "updated_at"])

    def advance_phase(self, new_phase: str, notes: str = ""):
        """Advance to a new DMAIC phase."""
        from datetime import datetime

        if not self.phase_history:
            self.phase_history = []

        self.phase_history.append({
            "phase": new_phase,
            "entered_at": datetime.now().isoformat(),
            "notes": notes,
        })
        self.dmaic_phase = new_phase
        self.save(update_fields=["dmaic_phase", "phase_history", "updated_at"])

    def get_phase_guidance(self) -> dict:
        """Get guidance for the current DMAIC phase."""
        guidance = {
            self.DMAICPhase.DEFINE: {
                "focus": "Define the problem clearly",
                "activities": [
                    "Create problem statement",
                    "Identify stakeholders",
                    "Define scope and constraints",
                    "Set measurable goals",
                ],
                "deliverables": ["Problem statement", "Project charter", "SIPOC diagram"],
                "next_phase": self.DMAICPhase.MEASURE,
            },
            self.DMAICPhase.MEASURE: {
                "focus": "Measure current performance",
                "activities": [
                    "Identify key metrics",
                    "Collect baseline data",
                    "Validate measurement system",
                    "Document current process",
                ],
                "deliverables": ["Baseline metrics", "Data collection plan", "Process map"],
                "next_phase": self.DMAICPhase.ANALYZE,
            },
            self.DMAICPhase.ANALYZE: {
                "focus": "Analyze root causes",
                "activities": [
                    "Generate hypotheses",
                    "Analyze data for patterns",
                    "Identify root causes",
                    "Validate with experiments",
                ],
                "deliverables": ["Root cause analysis", "Statistical analysis", "Validated causes"],
                "next_phase": self.DMAICPhase.IMPROVE,
            },
            self.DMAICPhase.IMPROVE: {
                "focus": "Implement improvements",
                "activities": [
                    "Generate solutions",
                    "Pilot test improvements",
                    "Implement changes",
                    "Verify results",
                ],
                "deliverables": ["Solution design", "Pilot results", "Implementation plan"],
                "next_phase": self.DMAICPhase.CONTROL,
            },
            self.DMAICPhase.CONTROL: {
                "focus": "Sustain the gains",
                "activities": [
                    "Create control plan",
                    "Standardize process",
                    "Train stakeholders",
                    "Monitor performance",
                ],
                "deliverables": ["Control plan", "Standard procedures", "Monitoring dashboard"],
                "next_phase": None,
            },
        }
        return guidance.get(self.dmaic_phase, {})

    # =========================================================================
    # Phase 1 Migration: Dual-write to core.Project
    # =========================================================================

    def ensure_core_project(self):
        """Create or return the linked core.Project.

        Called on Problem creation. Maps Problem fields → Project fields.
        """
        if self.core_project:
            return self.core_project

        from core.models import Project

        # Map methodology (Problem uses "none", Project uses "none" — same values except SCIENTIFIC)
        methodology_map = {
            "none": "none",
            "dmaic": "dmaic",
            "doe": "doe",
            "pdca": "pdca",
            "a3": "a3",
        }
        mapped_methodology = methodology_map.get(self.methodology, "scientific")

        project = Project.objects.create(
            user=self.user,
            title=self.title,
            problem_statement=self.effect_description,
            effect_description=self.effect_description,
            effect_magnitude=self.effect_magnitude,
            domain=self.domain,
            stakeholders=self.stakeholders,
            constraints=self.constraints,
            can_experiment=self.can_experiment,
            available_data=self.available_data,
            methodology=mapped_methodology,
            status="active",
        )

        self.core_project = project
        self.save(update_fields=["core_project"])
        return project

    def sync_hypothesis_to_core(self, hypothesis_dict: dict):
        """Sync a JSON hypothesis to a core.Hypothesis FK record.

        Called when a hypothesis is added to the Problem.
        Returns the core.Hypothesis instance.
        """
        project = self.ensure_core_project()

        from core.models.hypothesis import Hypothesis

        core_hyp = Hypothesis.objects.create(
            project=project,
            statement=hypothesis_dict["cause"],
            mechanism=hypothesis_dict.get("mechanism", ""),
            prior_probability=hypothesis_dict.get("probability", 0.5),
            current_probability=hypothesis_dict.get("probability", 0.5),
            created_by=self.user,
        )
        return core_hyp

    def sync_evidence_to_core(self, evidence_dict: dict):
        """Sync a JSON evidence entry to core.Evidence + EvidenceLinks.

        Called when evidence is added to the Problem.
        Creates the Evidence record and links it to relevant hypotheses.
        """
        project = self.ensure_core_project()

        from core.models.hypothesis import Evidence, EvidenceLink, Hypothesis

        # Map evidence types
        source_map = {
            "observation": "observation",
            "research": "research",
            "data_analysis": "analysis",
            "experiment": "experiment",
            "calculation": "calculation",
        }
        source_type = source_map.get(evidence_dict.get("type", "observation"), "observation")

        core_evidence = Evidence.objects.create(
            summary=evidence_dict["summary"],
            source_type=source_type,
            source_description=evidence_dict.get("source", ""),
            confidence=0.8,  # Default; Problem model doesn't track confidence
            created_by=self.user,
        )

        # Link to hypotheses that this evidence supports/weakens
        for hyp_id in evidence_dict.get("supports", []):
            core_hyp = self._find_core_hypothesis(hyp_id)
            if core_hyp:
                link = EvidenceLink.objects.create(
                    hypothesis=core_hyp,
                    evidence=core_evidence,
                    likelihood_ratio=2.0,  # Moderate support (matches _update_probabilities)
                    reasoning=f"Evidence supports this hypothesis (from Problem sync)",
                    is_manual=False,
                )
                core_hyp.apply_evidence(link)

        for hyp_id in evidence_dict.get("weakens", []):
            core_hyp = self._find_core_hypothesis(hyp_id)
            if core_hyp:
                link = EvidenceLink.objects.create(
                    hypothesis=core_hyp,
                    evidence=core_evidence,
                    likelihood_ratio=0.5,  # Moderate opposition (matches _update_probabilities)
                    reasoning=f"Evidence opposes this hypothesis (from Problem sync)",
                    is_manual=False,
                )
                core_hyp.apply_evidence(link)

        return core_evidence

    def _find_core_hypothesis(self, json_hyp_id: str):
        """Find a core.Hypothesis that matches a JSON hypothesis ID.

        Uses the hypothesis statement to match, since JSON IDs are short UUIDs
        that don't correspond to core.Hypothesis UUIDs.
        """
        if not self.core_project:
            return None

        from core.models.hypothesis import Hypothesis

        # Find the JSON hypothesis to get its statement
        for hyp in self.hypotheses:
            if hyp["id"] == json_hyp_id:
                # Match by statement text
                return Hypothesis.objects.filter(
                    project=self.core_project,
                    statement=hyp["cause"],
                ).first()
        return None


class CacheEntry(models.Model):
    """Database-backed cache entry with TTL support.

    Stores serialized Python objects with expiration times.
    Tempora cleans up expired entries via scheduled task.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    key = models.CharField(max_length=255, unique=True, db_index=True)
    value = models.BinaryField()  # Pickled Python object
    value_type = models.CharField(max_length=50, default="pickle")  # pickle, json

    # TTL support
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)

    # Metadata
    namespace = models.CharField(max_length=50, blank=True, db_index=True)
    user_id = models.IntegerField(null=True, blank=True, db_index=True)
    hit_count = models.IntegerField(default=0)
    last_accessed = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "session_cache"
        indexes = [
            models.Index(fields=["namespace", "key"]),
            models.Index(fields=["expires_at"]),
            models.Index(fields=["user_id", "namespace"]),
        ]

    def __str__(self):
        return f"{self.key} (expires: {self.expires_at})"

    @property
    def is_expired(self) -> bool:
        from django.utils import timezone
        if self.expires_at is None:
            return False
        return timezone.now() > self.expires_at

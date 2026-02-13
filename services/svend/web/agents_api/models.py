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

    # Optional project linking for A3/method import
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="dsw_results",
    )

    # Human-readable title for the analysis
    title = models.CharField(max_length=255, blank=True, help_text="e.g., 'Capability Study - Line A'")

    class Meta:
        ordering = ["-created_at"]

    def get_summary(self):
        """Return a brief summary of the result for import previews."""
        import json
        try:
            data = json.loads(self.data)
            # Try to extract key findings
            summary_parts = []
            if self.title:
                summary_parts.append(self.title)
            if 'analysis' in data:
                summary_parts.append(f"Analysis: {data['analysis']}")
            if 'summary' in data:
                # Direct summary field
                return data['summary']
            if 'findings' in data and isinstance(data['findings'], list):
                summary_parts.extend(data['findings'][:3])
            return " | ".join(summary_parts) if summary_parts else self.result_type
        except (json.JSONDecodeError, KeyError):
            return self.title or self.result_type


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
    # Phase 2 Migration: Read from core.Project FKs
    # =========================================================================

    def get_hypotheses(self) -> list[dict]:
        """Read hypotheses from core.Project FKs, fall back to JSON blob."""
        if self.core_project:
            return [
                {
                    "id": str(h.id)[:8],
                    "cause": h.statement,
                    "mechanism": h.mechanism,
                    "probability": h.current_probability,
                    "testable": h.is_testable,
                    "evidence_for": [str(e.evidence_id)[:8] for e in h.supporting_evidence],
                    "evidence_against": [str(e.evidence_id)[:8] for e in h.opposing_evidence],
                    "status": h.status,
                    "created_at": h.created_at.isoformat(),
                }
                for h in self.core_project.hypotheses.all()
            ]
        return self.hypotheses

    def get_evidence(self) -> list[dict]:
        """Read evidence from core.Evidence via EvidenceLinks, fall back to JSON blob."""
        if self.core_project:
            from core.models.hypothesis import Evidence, EvidenceLink
            # Get distinct evidence linked to any hypothesis in this project
            evidence_ids = EvidenceLink.objects.filter(
                hypothesis__project=self.core_project
            ).values_list("evidence_id", flat=True).distinct()
            evidences = Evidence.objects.filter(id__in=evidence_ids).order_by("-created_at")
            return [
                {
                    "id": str(e.id)[:8],
                    "type": e.source_type,
                    "summary": e.summary,
                    "source": e.source_description,
                    "supports": [
                        str(link.hypothesis_id)[:8]
                        for link in e.hypothesis_links.filter(likelihood_ratio__gt=1.0)
                    ],
                    "weakens": [
                        str(link.hypothesis_id)[:8]
                        for link in e.hypothesis_links.filter(likelihood_ratio__lt=1.0)
                    ],
                    "timestamp": e.created_at.isoformat(),
                }
                for e in evidences
            ]
        return self.evidence

    def get_dead_ends(self) -> list[dict]:
        """Read dead ends from core.Hypothesis with status=rejected, fall back to JSON blob."""
        if self.core_project:
            rejected = self.core_project.hypotheses.filter(status="rejected")
            return [
                {
                    "hypothesis_id": str(h.id)[:8],
                    "hypothesis_text": h.statement,
                    "why_rejected": h.mechanism or "Rejected based on evidence",
                    "timestamp": h.updated_at.isoformat(),
                }
                for h in rejected
            ]
        return self.dead_ends

    def get_probable_causes(self) -> list[dict]:
        """Read probable causes from core.Hypothesis by probability, fall back to JSON blob."""
        if self.core_project:
            active = self.core_project.hypotheses.filter(
                status__in=["active", "uncertain"]
            ).order_by("-current_probability")[:3]
            return [
                {"cause": h.statement, "probability": h.current_probability}
                for h in active
                if h.current_probability > 0.2
            ]
        return self.probable_causes

    def get_hypothesis_count(self) -> int:
        """Count hypotheses from core.Project or JSON blob."""
        if self.core_project:
            return self.core_project.hypotheses.count()
        return len(self.hypotheses)

    def get_evidence_count(self) -> int:
        """Count evidence from core.Project or JSON blob."""
        if self.core_project:
            return self.core_project.evidence_count
        return len(self.evidence)

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


# =============================================================================
# LLM Usage Tracking & Rate Limiting
# =============================================================================

class LLMUsage(models.Model):
    """Track LLM API usage per user for rate limiting and cost control."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="llm_usage",
    )
    date = models.DateField(db_index=True)
    model = models.CharField(max_length=50)  # haiku, sonnet, opus
    request_count = models.IntegerField(default=0)
    input_tokens = models.IntegerField(default=0)
    output_tokens = models.IntegerField(default=0)

    class Meta:
        unique_together = ("user", "date", "model")
        indexes = [
            models.Index(fields=["user", "date"]),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.date} - {self.model}: {self.request_count} requests"

    @classmethod
    def get_daily_usage(cls, user, date=None):
        """Get total requests for a user on a given date."""
        from django.utils import timezone
        if date is None:
            date = timezone.now().date()
        return cls.objects.filter(user=user, date=date).aggregate(
            total_requests=models.Sum('request_count'),
            total_input_tokens=models.Sum('input_tokens'),
            total_output_tokens=models.Sum('output_tokens'),
        )

    @classmethod
    def record_usage(cls, user, model, input_tokens=0, output_tokens=0):
        """Record an LLM request. Returns updated usage."""
        from django.utils import timezone
        today = timezone.now().date()

        usage, created = cls.objects.get_or_create(
            user=user,
            date=today,
            model=model,
            defaults={'request_count': 0, 'input_tokens': 0, 'output_tokens': 0}
        )

        usage.request_count += 1
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.save()

        return usage


# Rate limits by tier (requests per day)
LLM_RATE_LIMITS = {
    "FREE": 10,
    "FOUNDER": 50,
    "PRO": 200,
    "TEAM": 500,
    "ENTERPRISE": 10000,  # Effectively unlimited
}


def check_rate_limit(user):
    """Check if user is within their rate limit.

    Returns:
        (allowed: bool, remaining: int, limit: int)
    """
    from django.utils import timezone

    tier = getattr(user, 'subscription_tier', 'FREE') or 'FREE'
    limit = LLM_RATE_LIMITS.get(tier.upper(), LLM_RATE_LIMITS["FREE"])

    usage = LLMUsage.get_daily_usage(user, timezone.now().date())
    current = usage['total_requests'] or 0

    return (current < limit, limit - current, limit)


# =============================================================================
# Whiteboard - Collaborative Boards
# =============================================================================

def generate_room_code():
    """Generate a 6-character room code."""
    import random
    import string
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))


class Board(models.Model):
    """Collaborative whiteboard for kaizen sessions."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    room_code = models.CharField(max_length=10, unique=True, default=generate_room_code, db_index=True)
    name = models.CharField(max_length=255, default="Untitled Board")

    # Owner/creator
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="owned_boards",
    )

    # Optional link to a project (for kaizen sessions tied to investigations)
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="boards",
    )

    # Board state as JSON
    elements = models.JSONField(default=list)  # List of element objects
    connections = models.JSONField(default=list)  # List of connection objects

    # Canvas state
    zoom = models.FloatField(default=1.0)
    pan_x = models.FloatField(default=0.0)
    pan_y = models.FloatField(default=0.0)

    # Voting state
    voting_active = models.BooleanField(default=False)
    votes_per_user = models.IntegerField(default=3)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Version for conflict detection (incremented on each save)
    version = models.IntegerField(default=0)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.name} ({self.room_code})"

    def save(self, *args, **kwargs):
        # Increment version on save
        self.version += 1
        super().save(*args, **kwargs)


class BoardParticipant(models.Model):
    """Track who's in a board session."""

    board = models.ForeignKey(Board, on_delete=models.CASCADE, related_name="participants")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="board_participations",
    )

    # Participant color for cursors/attribution
    color = models.CharField(max_length=7, default="#4a9f6e")  # Hex color

    # Last activity (for presence detection)
    last_seen = models.DateTimeField(auto_now=True)

    # Cursor position (for showing where others are)
    cursor_x = models.FloatField(null=True, blank=True)
    cursor_y = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ("board", "user")

    def __str__(self):
        return f"{self.user.username} in {self.board.room_code}"


class BoardVote(models.Model):
    """Dot vote on a board element."""

    board = models.ForeignKey(Board, on_delete=models.CASCADE, related_name="votes")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="board_votes",
    )
    element_id = models.CharField(max_length=50)  # The element ID being voted on
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # One vote per user per element
        unique_together = ("board", "user", "element_id")

    def __str__(self):
        return f"{self.user.username} voted on {self.element_id}"


# =============================================================================
# Methods - A3, DMAIC, 8D, etc.
# =============================================================================

class A3Report(models.Model):
    """Toyota-style A3 problem-solving report.

    A3 is a structured single-page format for problem solving:
    - Forces concise thinking (fits on 11x17 paper)
    - Follows PDCA logic flow
    - Encourages visual management
    """

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        IN_PROGRESS = "in_progress", "In Progress"
        REVIEW = "review", "Under Review"
        COMPLETE = "complete", "Complete"
        ARCHIVED = "archived", "Archived"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Ownership
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="a3_reports",
    )

    # Link to project (A3 is always part of a project)
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.CASCADE,
        related_name="a3_reports",
    )

    # Header
    title = models.CharField(max_length=255, help_text="Problem/theme title")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)

    # Left column (Plan)
    background = models.TextField(
        blank=True,
        help_text="Why does this matter? Business context, impact."
    )
    current_condition = models.TextField(
        blank=True,
        help_text="What's happening now? Data, metrics, observations."
    )
    goal = models.TextField(
        blank=True,
        help_text="What are we trying to achieve? Target condition, metrics."
    )
    root_cause = models.TextField(
        blank=True,
        help_text="Why is this happening? 5-why, fishbone findings."
    )

    # Right column (Do/Check/Act)
    countermeasures = models.TextField(
        blank=True,
        help_text="What will we do about it? Actions to address root causes."
    )
    implementation_plan = models.TextField(
        blank=True,
        help_text="Who, what, when? Action items with owners and dates."
    )
    follow_up = models.TextField(
        blank=True,
        help_text="How will we verify? Check dates, success metrics."
    )

    # Imported content references (for traceability)
    imported_from = models.JSONField(
        default=dict,
        blank=True,
        help_text="References to imported content: {section: [{source, id, summary}]}"
    )

    # Embedded diagrams (SVG snapshots from whiteboards)
    embedded_diagrams = models.JSONField(
        default=dict,
        blank=True,
        help_text="Embedded SVG diagrams: {section: [{id, svg, board_name, room_code}]}"
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "a3_reports"
        ordering = ["-updated_at"]
        verbose_name = "A3 Report"
        verbose_name_plural = "A3 Reports"

    def __str__(self):
        return f"A3: {self.title} ({self.status})"

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "title": self.title,
            "status": self.status,
            "background": self.background,
            "current_condition": self.current_condition,
            "goal": self.goal,
            "root_cause": self.root_cause,
            "countermeasures": self.countermeasures,
            "implementation_plan": self.implementation_plan,
            "follow_up": self.follow_up,
            "imported_from": self.imported_from,
            "embedded_diagrams": self.embedded_diagrams,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class Report(models.Model):
    """Flexible report model for CAPA, 8D, and future report types.

    Uses a registry-driven design: report_type maps to REPORT_TYPES in
    report_types.py which defines the section structure. Adding a new
    report type requires only a new registry entry — zero migrations.

    Sections are stored as a JSONField dict: {"section_key": "content"}.
    """

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        IN_PROGRESS = "in_progress", "In Progress"
        REVIEW = "review", "Under Review"
        COMPLETE = "complete", "Complete"
        ARCHIVED = "archived", "Archived"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="reports",
    )

    project = models.ForeignKey(
        "core.Project",
        on_delete=models.CASCADE,
        related_name="reports",
    )

    report_type = models.CharField(
        max_length=30,
        help_text="Key into REPORT_TYPES registry (e.g. 'capa', '8d')",
    )

    title = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)

    sections = models.JSONField(
        default=dict,
        blank=True,
        help_text='Section content: {"problem_description": "...", "root_cause_analysis": "..."}',
    )

    imported_from = models.JSONField(
        default=dict,
        blank=True,
        help_text="References to imported content: {section: [{source, id, summary}]}",
    )

    embedded_diagrams = models.JSONField(
        default=dict,
        blank=True,
        help_text="Embedded SVG diagrams: {section: [{id, svg, board_name, room_code}]}",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "reports"
        ordering = ["-updated_at"]
        verbose_name = "Report"
        verbose_name_plural = "Reports"

    def __str__(self):
        return f"{self.get_type_name()}: {self.title} ({self.status})"

    def get_type_name(self):
        from .report_types import REPORT_TYPES
        rt = REPORT_TYPES.get(self.report_type, {})
        return rt.get("name", self.report_type.upper())

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "report_type": self.report_type,
            "type_name": self.get_type_name(),
            "title": self.title,
            "status": self.status,
            "sections": self.sections,
            "imported_from": self.imported_from,
            "embedded_diagrams": self.embedded_diagrams,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# FMEA (Failure Mode and Effects Analysis)
# =============================================================================

class FMEA(models.Model):
    """Failure Mode and Effects Analysis study.

    Persistent FMEA with S/O/D scoring and optional Bayesian evidence linking.
    Each FMEA has rows (failure modes) that can generate Evidence records
    linked to Hypothesis objects via EvidenceLink.
    """

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        ACTIVE = "active", "Active"
        REVIEW = "review", "Under Review"
        COMPLETE = "complete", "Complete"

    class FMEAType(models.TextChoices):
        PROCESS = "process", "Process FMEA"
        DESIGN = "design", "Design FMEA"
        SYSTEM = "system", "System FMEA"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="fmeas",
    )

    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="fmeas",
    )

    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)
    fmea_type = models.CharField(max_length=20, choices=FMEAType.choices, default=FMEAType.PROCESS)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "fmeas"
        ordering = ["-updated_at"]
        verbose_name = "FMEA"
        verbose_name_plural = "FMEAs"

    def __str__(self):
        return f"FMEA: {self.title} ({self.status})"

    def to_dict(self):
        rows = list(self.rows.order_by("sort_order"))
        return {
            "id": str(self.id),
            "project_id": str(self.project_id) if self.project_id else None,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "fmea_type": self.fmea_type,
            "rows": [r.to_dict() for r in rows],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class FMEARow(models.Model):
    """A single failure mode row in an FMEA study."""

    class ActionStatus(models.TextChoices):
        NOT_STARTED = "not_started", "Not Started"
        IN_PROGRESS = "in_progress", "In Progress"
        COMPLETE = "complete", "Complete"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    fmea = models.ForeignKey(FMEA, on_delete=models.CASCADE, related_name="rows")
    sort_order = models.IntegerField(default=0)

    # Failure mode description
    process_step = models.CharField(max_length=255, blank=True)
    failure_mode = models.CharField(max_length=255)
    effect = models.TextField(blank=True)

    # Original S/O/D scores (1-10)
    severity = models.IntegerField(default=1)
    cause = models.TextField(blank=True)
    occurrence = models.IntegerField(default=1)
    current_controls = models.TextField(blank=True)
    detection = models.IntegerField(default=1)

    # Computed: severity * occurrence * detection
    rpn = models.IntegerField(default=1)

    # Recommended actions
    recommended_action = models.TextField(blank=True)
    action_owner = models.CharField(max_length=255, blank=True)
    action_status = models.CharField(
        max_length=20, choices=ActionStatus.choices, default=ActionStatus.NOT_STARTED,
    )

    # Revised scores (after corrective actions)
    revised_severity = models.IntegerField(null=True, blank=True)
    revised_occurrence = models.IntegerField(null=True, blank=True)
    revised_detection = models.IntegerField(null=True, blank=True)
    revised_rpn = models.IntegerField(null=True, blank=True)

    # Bayesian bridge — optional links to hypothesis/evidence system
    hypothesis_link = models.ForeignKey(
        "core.Hypothesis",
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="fmea_rows",
        help_text="Hypothesis this failure mode relates to",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "fmea_rows"
        ordering = ["sort_order"]

    def __str__(self):
        return f"{self.failure_mode} (RPN={self.rpn})"

    def save(self, *args, **kwargs):
        self.rpn = self.severity * self.occurrence * self.detection
        if self.revised_severity and self.revised_occurrence and self.revised_detection:
            self.revised_rpn = self.revised_severity * self.revised_occurrence * self.revised_detection
        super().save(*args, **kwargs)

    def to_dict(self):
        return {
            "id": str(self.id),
            "fmea_id": str(self.fmea_id),
            "sort_order": self.sort_order,
            "process_step": self.process_step,
            "failure_mode": self.failure_mode,
            "effect": self.effect,
            "severity": self.severity,
            "cause": self.cause,
            "occurrence": self.occurrence,
            "current_controls": self.current_controls,
            "detection": self.detection,
            "rpn": self.rpn,
            "recommended_action": self.recommended_action,
            "action_owner": self.action_owner,
            "action_status": self.action_status,
            "revised_severity": self.revised_severity,
            "revised_occurrence": self.revised_occurrence,
            "revised_detection": self.revised_detection,
            "revised_rpn": self.revised_rpn,
            "hypothesis_id": str(self.hypothesis_link_id) if self.hypothesis_link_id else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# Root Cause Analysis (RCA) - Special Cause Investigation
# =============================================================================

class RCASession(models.Model):
    """Root Cause Analysis session for special cause investigation.

    RCA is for SPECIAL CAUSE problems only - unique events that require
    investigation of the causal chain (like NTSB investigations).

    For COMMON CAUSE problems (systemic issues), use fishbone/C&E matrix
    and Kaizen instead.
    """

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        IN_PROGRESS = "in_progress", "In Progress"
        REVIEW = "review", "Under Review"
        COMPLETE = "complete", "Complete"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Ownership
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="rca_sessions",
    )

    # Optional project link
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="rca_sessions",
    )

    # Optional A3 link (if this RCA is embedded in an A3)
    a3_report = models.ForeignKey(
        A3Report,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="rca_sessions",
    )

    # The incident/event being investigated
    title = models.CharField(max_length=255, blank=True)
    event = models.TextField(help_text="Description of the incident")

    # Causal chain: [{claim, critique, accepted, error_labels}]
    chain = models.JSONField(
        default=list,
        help_text="Causal chain steps with critiques"
    )

    # Conclusions
    root_cause = models.TextField(blank=True, help_text="Stated root cause")
    countermeasure = models.TextField(blank=True, help_text="Proposed countermeasure")
    evaluation = models.TextField(blank=True, help_text="Final AI evaluation of the analysis")

    # Status
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Embedding for similarity search (stored as bytes for portability)
    embedding = models.BinaryField(
        null=True,
        blank=True,
        help_text="Embedding vector for similarity search"
    )

    class Meta:
        db_table = "rca_sessions"
        ordering = ["-updated_at"]
        verbose_name = "RCA Session"
        verbose_name_plural = "RCA Sessions"

    def __str__(self):
        return f"RCA: {self.title or self.event[:50]} ({self.status})"

    def generate_embedding(self):
        """Generate and store embedding for this RCA session."""
        from .embeddings import generate_rca_embedding
        import numpy as np

        embedding = generate_rca_embedding(
            event=self.event,
            chain=self.chain or [],
            root_cause=self.root_cause,
        )

        if embedding is not None:
            self.embedding = embedding.tobytes()
            return True
        return False

    def get_embedding(self):
        """Get embedding as numpy array."""
        import numpy as np

        if self.embedding is None:
            return None

        return np.frombuffer(self.embedding, dtype=np.float32)

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id) if self.project_id else None,
            "a3_report_id": str(self.a3_report_id) if self.a3_report_id else None,
            "title": self.title,
            "event": self.event,
            "chain": self.chain,
            "root_cause": self.root_cause,
            "countermeasure": self.countermeasure,
            "evaluation": self.evaluation,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# Value Stream Map (VSM)
# =============================================================================

class ValueStreamMap(models.Model):
    """Value Stream Map for lean process analysis.

    VSM visualizes material and information flow to identify waste.
    Unlike a general whiteboard, VSM has structured elements:
    - Process steps with cycle time, wait time, etc.
    - Inventory triangles between steps
    - Information flow (customer demand, scheduling)
    - Timeline showing value-add vs non-value-add time
    """

    class Status(models.TextChoices):
        CURRENT = "current", "Current State"
        FUTURE = "future", "Future State"
        ARCHIVED = "archived", "Archived"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="value_stream_maps",
    )

    # Optional link to project
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="value_stream_maps",
    )

    # Header
    name = models.CharField(max_length=255, default="Untitled VSM")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.CURRENT)
    product_family = models.CharField(max_length=255, blank=True, help_text="Product or service being mapped")

    # Customer info (right side of VSM)
    customer_name = models.CharField(max_length=255, blank=True, default="Customer")
    customer_demand = models.CharField(max_length=100, blank=True, help_text="e.g., 460 units/day")
    takt_time = models.FloatField(null=True, blank=True, help_text="Takt time in seconds")

    # Supplier info (left side of VSM)
    supplier_name = models.CharField(max_length=255, blank=True, default="Supplier")
    supply_frequency = models.CharField(max_length=100, blank=True, help_text="e.g., Weekly")

    # Multiple customers/suppliers as JSON arrays
    # Each: {id, name, detail, x, y}
    customers = models.JSONField(default=list, help_text="Customer entities on the map")
    suppliers = models.JSONField(default=list, help_text="Supplier entities on the map")

    # Process steps as JSON
    # Each step: {id, name, x, y, cycle_time, changeover_time, uptime, operators, shifts, batch_size, ...}
    process_steps = models.JSONField(default=list, help_text="Process boxes with metrics")

    # Inventory between steps
    # Each: {id, before_step_id, quantity, days_of_supply, x, y}
    inventory = models.JSONField(default=list, help_text="Inventory triangles")

    # Information flow
    # Each: {id, from_id, to_id, type: 'electronic'|'manual'|'schedule', label}
    information_flow = models.JSONField(default=list, help_text="Information arrows")

    # Material flow connections
    # Each: {id, from_step_id, to_step_id, type: 'push'|'pull'|'fifo'}
    material_flow = models.JSONField(default=list, help_text="Material flow arrows")

    # Timeline summary (calculated)
    total_lead_time = models.FloatField(null=True, blank=True, help_text="Total lead time in days")
    total_process_time = models.FloatField(null=True, blank=True, help_text="Total value-add time in seconds")
    pce = models.FloatField(null=True, blank=True, help_text="Process Cycle Efficiency (%)")

    # Kaizen bursts (improvement opportunities)
    # Each: {id, x, y, text, priority}
    kaizen_bursts = models.JSONField(default=list, help_text="Improvement opportunities")

    # Work centers (grouped parallel machines)
    # Each: {id, name, x, y, width, height}
    # Process steps link via optional work_center_id field
    work_centers = models.JSONField(default=list, help_text="Work center groupings")

    # Canvas state
    zoom = models.FloatField(default=1.0)
    pan_x = models.FloatField(default=0.0)
    pan_y = models.FloatField(default=0.0)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "value_stream_maps"
        ordering = ["-updated_at"]
        verbose_name = "Value Stream Map"
        verbose_name_plural = "Value Stream Maps"

    def __str__(self):
        return f"VSM: {self.name} ({self.status})"

    def calculate_metrics(self):
        """Calculate total lead time, process time, and PCE.

        For steps in work centers (parallel machines), uses effective cycle time:
        effective_CT = 1 / sum(1/CT_i) for all machines in the work center.
        """
        total_ct = 0.0  # Cycle time in seconds
        total_wait = 0.0  # Wait time in days

        # Group steps by work center
        wc_steps = {}  # work_center_id -> [cycle_times]
        standalone_cts = []

        for step in self.process_steps:
            ct = step.get("cycle_time", 0) or 0
            wc_id = step.get("work_center_id")
            if wc_id:
                wc_steps.setdefault(wc_id, []).append(ct)
            else:
                standalone_cts.append(ct)

        # Standalone steps: sum cycle times directly
        total_ct += sum(standalone_cts)

        # Work center steps: effective CT = 1 / sum(1/CT_i)
        for wc_id, cts in wc_steps.items():
            rate_sum = sum(1.0 / ct for ct in cts if ct > 0)
            if rate_sum > 0:
                total_ct += 1.0 / rate_sum

        for inv in self.inventory:
            days = inv.get("days_of_supply", 0) or 0
            total_wait += days

        self.total_process_time = total_ct
        self.total_lead_time = total_wait + (total_ct / 86400)  # Convert CT to days

        if self.total_lead_time > 0:
            self.pce = (total_ct / 86400 / self.total_lead_time) * 100
        else:
            self.pce = 0

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id) if self.project_id else None,
            "name": self.name,
            "status": self.status,
            "product_family": self.product_family,
            "customer_name": self.customer_name,
            "customer_demand": self.customer_demand,
            "takt_time": self.takt_time,
            "supplier_name": self.supplier_name,
            "supply_frequency": self.supply_frequency,
            "customers": self.customers,
            "suppliers": self.suppliers,
            "process_steps": self.process_steps,
            "inventory": self.inventory,
            "information_flow": self.information_flow,
            "material_flow": self.material_flow,
            "total_lead_time": self.total_lead_time,
            "total_process_time": self.total_process_time,
            "pce": self.pce,
            "kaizen_bursts": self.kaizen_bursts,
            "work_centers": self.work_centers,
            "zoom": self.zoom,
            "pan_x": self.pan_x,
            "pan_y": self.pan_y,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# Learning / Certification Models
# =============================================================================


class SectionProgress(models.Model):
    """Tracks a user's completion of individual course sections."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="section_progress",
    )
    module_id = models.CharField(max_length=64)
    section_id = models.CharField(max_length=64)
    completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)
    time_spent_seconds = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "learn_section_progress"
        ordering = ["-updated_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["user", "module_id", "section_id"],
                name="unique_user_section",
            )
        ]
        indexes = [
            models.Index(fields=["user", "module_id"]),
            models.Index(fields=["user", "completed"]),
        ]

    def __str__(self):
        status = "done" if self.completed else "in progress"
        return f"{self.user} — {self.module_id}/{self.section_id} ({status})"


class AssessmentAttempt(models.Model):
    """Records a single certification assessment attempt."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="assessment_attempts",
    )
    questions = models.JSONField(default=list)  # Full question set with answers
    answers = models.JSONField(default=dict)  # User's submitted answers
    score = models.FloatField(null=True, blank=True)
    passed = models.BooleanField(default=False)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "learn_assessment_attempt"
        ordering = ["-started_at"]
        indexes = [
            models.Index(fields=["user", "-started_at"]),
            models.Index(fields=["user", "passed"]),
        ]

    def __str__(self):
        score_str = f"{self.score:.0%}" if self.score is not None else "pending"
        return f"{self.user} — assessment {score_str}"


# =============================================================================
# Hoshin Kanri CI Project Management (Enterprise-only)
# =============================================================================


class Site(models.Model):
    """Manufacturing site within an enterprise tenant.

    One Tenant (billing entity) can have many Sites (physical plants).
    Contacts are text fields — operators/managers often don't have Svend accounts.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tenant = models.ForeignKey(
        "core.Tenant",
        on_delete=models.CASCADE,
        related_name="sites",
    )
    name = models.CharField(max_length=255)
    code = models.CharField(
        max_length=20, blank=True,
        help_text="Short site code, e.g. PLT-01",
    )
    business_unit = models.CharField(max_length=255, blank=True)
    plant_manager = models.CharField(max_length=255, blank=True)
    ci_leader = models.CharField(max_length=255, blank=True)
    controller = models.CharField(max_length=255, blank=True)
    address = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "hoshin_sites"
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.code})" if self.code else self.name

    def to_dict(self):
        return {
            "id": str(self.id),
            "tenant_id": str(self.tenant_id),
            "name": self.name,
            "code": self.code,
            "business_unit": self.business_unit,
            "plant_manager": self.plant_manager,
            "ci_leader": self.ci_leader,
            "controller": self.controller,
            "address": self.address,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class SiteAccess(models.Model):
    """Per-site access control for Hoshin Kanri.

    Org owners/admins bypass this — they see all sites.
    Members/viewers need an explicit SiteAccess entry to see a site's projects.
    """

    class SiteRole(models.TextChoices):
        VIEWER = "viewer", "Viewer"
        MEMBER = "member", "Member"
        ADMIN = "admin", "Admin"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    site = models.ForeignKey(
        Site, on_delete=models.CASCADE, related_name="access_list",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="site_access",
    )
    role = models.CharField(
        max_length=20, choices=SiteRole.choices, default=SiteRole.MEMBER,
    )
    granted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, related_name="+",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "hoshin_site_access"
        unique_together = [["site", "user"]]

    def __str__(self):
        return f"{self.user} → {self.site} ({self.role})"

    def to_dict(self):
        return {
            "id": str(self.id),
            "site_id": str(self.site_id),
            "user_id": self.user_id,
            "username": self.user.username,
            "display_name": getattr(self.user, "display_name", "") or self.user.username,
            "email": self.user.email,
            "role": self.role,
            "granted_by_id": self.granted_by_id,
            "created_at": self.created_at.isoformat(),
        }


def _current_year():
    from datetime import date
    return date.today().year


class HoshinProject(models.Model):
    """CI improvement project extending core.Project with Hoshin Kanri tracking.

    Wraps core.Project (1:1) to add manufacturing CI-specific fields:
    - Project classification (kaizen event vs extended project)
    - Savings categorization and monthly financial tracking
    - Fiscal year budgeting and site assignment
    - Link back to originating VSM kaizen burst

    The core.Project provides: 5W2H, SMART goals, team, methodology,
    phase tracking, milestones, hypothesis/evidence, Bayesian reasoning.
    """

    class ProjectClass(models.TextChoices):
        KAIZEN = "kaizen", "Kaizen Event"
        PROJECT = "project", "Extended Project"

    class ProjectType(models.TextChoices):
        MATERIAL = "material", "Material Savings"
        LABOR = "labor", "Labor Savings"
        QUALITY = "quality", "Quality Improvement"
        THROUGHPUT = "throughput", "Throughput Improvement"
        ENERGY = "energy", "Energy Reduction"
        SAFETY = "safety", "Safety Improvement"
        OTHER = "other", "Other"

    class Opportunity(models.TextChoices):
        CARRYOVER = "carryover", "Carryover from Prior Year"
        BUDGETED_NEW = "budgeted_new", "Budgeted New"
        CONTINGENCY = "contingency", "Contingency"
        UNPLANNED = "unplanned", "Unplanned/Reactive"

    class HoshinStatus(models.TextChoices):
        PROPOSED = "proposed", "Proposed"
        BUDGETED = "budgeted", "Budgeted"
        ACTIVE = "active", "Active"
        DELAYED = "delayed", "Delayed"
        COMPLETED = "completed", "Completed"
        ABORTED = "aborted", "Aborted"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.OneToOneField(
        "core.Project",
        on_delete=models.CASCADE,
        related_name="hoshin",
    )
    site = models.ForeignKey(
        Site,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="hoshin_projects",
    )
    project_class = models.CharField(
        max_length=20, choices=ProjectClass.choices, default=ProjectClass.PROJECT,
    )
    project_type = models.CharField(
        max_length=20, choices=ProjectType.choices, default=ProjectType.MATERIAL,
    )
    opportunity = models.CharField(
        max_length=20, choices=Opportunity.choices, default=Opportunity.BUDGETED_NEW,
    )
    hoshin_status = models.CharField(
        max_length=20, choices=HoshinStatus.choices, default=HoshinStatus.PROPOSED,
    )
    fiscal_year = models.IntegerField(default=_current_year, help_text="Fiscal year")
    annual_savings_target = models.DecimalField(
        max_digits=12, decimal_places=2, default=0,
        help_text="Target annual savings in dollars",
    )
    calculation_method = models.CharField(
        max_length=30, blank=True,
        help_text="waste_pct, time_reduction, headcount, claims, freight, energy, direct, layout, custom",
    )
    custom_formula = models.CharField(
        max_length=500, blank=True, default="",
        help_text="Custom formula e.g. '(baseline - actual) * volume * rate'",
    )
    custom_formula_desc = models.CharField(
        max_length=200, blank=True, default="",
        help_text="Human-readable description of the custom formula",
    )
    kaizen_charter = models.JSONField(
        default=dict, blank=True,
        help_text="Kaizen event logistics: {event_date, end_date, location, schedule, "
                  "event_type, primary_metric, primary_baseline, primary_target, "
                  "secondary_metric, secondary_baseline, secondary_target, "
                  "process_start, process_end}",
    )
    monthly_actuals = models.JSONField(
        default=list, blank=True,
        help_text="Monthly savings: [{month, baseline, actual, volume, cost_per_unit, savings}]",
    )
    baseline_data = models.JSONField(
        default=list, blank=True,
        help_text="Prior year baselines: [{month, metric_value, volume, cost_per_unit}]",
    )
    source_vsm = models.ForeignKey(
        ValueStreamMap,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="hoshin_projects",
    )
    source_burst_id = models.CharField(
        max_length=50, blank=True,
        help_text="ID of originating kaizen burst in source_vsm.kaizen_bursts",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "hoshin_projects"
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["site", "hoshin_status"]),
            models.Index(fields=["fiscal_year", "hoshin_status"]),
        ]

    def __str__(self):
        return f"[{self.get_project_class_display()}] {self.project.title}"

    @property
    def ytd_savings(self):
        return sum(m.get("savings", 0) or 0 for m in (self.monthly_actuals or []))

    @property
    def savings_pct(self):
        if not self.annual_savings_target:
            return 0
        return float(self.ytd_savings / float(self.annual_savings_target) * 100)

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "project_title": self.project.title,
            "project_status": self.project.status,
            "site_id": str(self.site_id) if self.site_id else None,
            "site_name": self.site.name if self.site else None,
            "project_class": self.project_class,
            "project_type": self.project_type,
            "opportunity": self.opportunity,
            "hoshin_status": self.hoshin_status,
            "fiscal_year": self.fiscal_year,
            "annual_savings_target": float(self.annual_savings_target),
            "calculation_method": self.calculation_method,
            "custom_formula": self.custom_formula,
            "custom_formula_desc": self.custom_formula_desc,
            "kaizen_charter": self.kaizen_charter,
            "monthly_actuals": self.monthly_actuals,
            "baseline_data": self.baseline_data,
            "ytd_savings": self.ytd_savings,
            "savings_pct": self.savings_pct,
            "source_vsm_id": str(self.source_vsm_id) if self.source_vsm_id else None,
            "source_burst_id": self.source_burst_id,
            "champion_name": self.project.champion_name,
            "leader_name": self.project.leader_name,
            "team_members": self.project.team_members,
            "methodology": self.project.methodology,
            "current_phase": self.project.current_phase,
            "goal_metric": self.project.goal_metric,
            "goal_baseline": str(self.project.goal_baseline) if self.project.goal_baseline else None,
            "goal_target": str(self.project.goal_target) if self.project.goal_target else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ActionItem(models.Model):
    """Task/action item with Gantt-style tracking for any project type.

    Attached to core.Project so it works for Hoshin CI projects,
    A3 reports, and general investigations alike.
    """

    class Status(models.TextChoices):
        NOT_STARTED = "not_started", "Not Started"
        IN_PROGRESS = "in_progress", "In Progress"
        COMPLETED = "completed", "Completed"
        BLOCKED = "blocked", "Blocked"
        CANCELLED = "cancelled", "Cancelled"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.CASCADE,
        related_name="action_items",
    )
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    owner_name = models.CharField(max_length=255, blank=True)
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.NOT_STARTED,
    )
    start_date = models.DateField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    due_date = models.DateField(null=True, blank=True)
    progress = models.IntegerField(default=0, help_text="0-100%")
    depends_on = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name="dependents",
    )
    sort_order = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "action_items"
        ordering = ["sort_order", "start_date"]

    def __str__(self):
        return self.title

    def to_dict(self):
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "title": self.title,
            "description": self.description,
            "owner_name": self.owner_name,
            "status": self.status,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "progress": self.progress,
            "depends_on_id": str(self.depends_on_id) if self.depends_on_id else None,
            "sort_order": self.sort_order,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }



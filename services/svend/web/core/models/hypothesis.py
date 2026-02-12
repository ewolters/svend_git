"""Hypothesis and Evidence models for Bayesian reasoning.

Hypothesis structure supports:
- Structured If/Then/Because format for clear causal claims
- Variable identification for linking to data
- Test planning and success criteria
- Bayesian probability tracking with likelihood ratios

All probability updates use core.bayesian.BayesianUpdater.
"""

import uuid
from django.conf import settings
from django.db import models

from core.bayesian import BayesianUpdater, get_updater


class Hypothesis(models.Model):
    """A hypothesis about a potential cause or explanation.

    Structured format:
    - If [independent variable/condition]...
    - Then [dependent variable/outcome]...
    - Because [mechanism/rationale]...

    This structure enables:
    - Clear testable predictions
    - Variable mapping to data
    - Logical validation
    - Report generation
    """

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        CONFIRMED = "confirmed", "Confirmed"
        REJECTED = "rejected", "Rejected"
        UNCERTAIN = "uncertain", "Uncertain"
        MERGED = "merged", "Merged"

    class Direction(models.TextChoices):
        INCREASE = "increase", "Increase"
        DECREASE = "decrease", "Decrease"
        CHANGE = "change", "Change"
        NO_CHANGE = "no_change", "No Change"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.CASCADE,
        related_name="hypotheses",
    )

    # =========================================================================
    # HYPOTHESIS STATEMENT - Structured Format
    # =========================================================================
    statement = models.TextField(
        help_text="Full hypothesis statement (can be generated from structured fields)",
    )

    # Structured components
    if_clause = models.TextField(
        blank=True,
        help_text="IF [condition/change]... What is being changed or tested?",
    )
    then_clause = models.TextField(
        blank=True,
        help_text="THEN [expected outcome]... What do we expect to happen?",
    )
    because_clause = models.TextField(
        blank=True,
        help_text="BECAUSE [mechanism]... Why do we think this will happen?",
    )

    # =========================================================================
    # VARIABLES - For linking to data and DSW
    # =========================================================================
    independent_variable = models.CharField(
        max_length=255,
        blank=True,
        help_text="X variable - what we change or observe as cause",
    )
    independent_var_values = models.JSONField(
        default=list,
        blank=True,
        help_text="Possible values or levels of X [list]",
    )
    dependent_variable = models.CharField(
        max_length=255,
        blank=True,
        help_text="Y variable - what we measure as effect",
    )
    dependent_var_unit = models.CharField(
        max_length=50,
        blank=True,
        help_text="Unit of measure for Y",
    )
    predicted_direction = models.CharField(
        max_length=20,
        choices=Direction.choices,
        default=Direction.CHANGE,
        help_text="Expected direction of change in Y",
    )
    predicted_magnitude = models.CharField(
        max_length=100,
        blank=True,
        help_text="Expected size of effect (e.g., '>10%', '~5 units')",
    )

    # =========================================================================
    # RATIONALE & TESTING
    # =========================================================================
    rationale = models.TextField(
        blank=True,
        help_text="Why do we think this hypothesis might be true? Prior knowledge, observations.",
    )
    test_method = models.TextField(
        blank=True,
        help_text="How will we test this? Experiment design, analysis approach.",
    )
    success_criteria = models.TextField(
        blank=True,
        help_text="What evidence would confirm this? What would refute it?",
    )
    data_requirements = models.JSONField(
        default=list,
        blank=True,
        help_text="Data needed to test [{variable, source, available}]",
    )

    # =========================================================================
    # PROBABILITY TRACKING - Bayesian
    # =========================================================================
    prior_probability = models.FloatField(
        default=0.5,
        help_text="Initial probability before evidence (0.0 to 1.0)",
    )
    current_probability = models.FloatField(
        default=0.5,
        help_text="Current probability after all evidence (0.0 to 1.0)",
    )
    probability_history = models.JSONField(
        default=list,
        blank=True,
        help_text="History of probability updates [{probability, evidence_id, timestamp}]",
    )

    # Thresholds for auto-status changes
    confirmation_threshold = models.FloatField(
        default=0.9,
        help_text="Probability above which hypothesis is confirmed",
    )
    rejection_threshold = models.FloatField(
        default=0.1,
        help_text="Probability below which hypothesis is rejected",
    )

    # =========================================================================
    # STATUS & METADATA
    # =========================================================================
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE,
    )

    is_testable = models.BooleanField(
        default=True,
        help_text="Can this hypothesis be tested with available data/experiments?",
    )
    test_suggestions = models.JSONField(
        default=list,
        blank=True,
        help_text="Suggested ways to test this hypothesis",
    )

    # Links to knowledge graph
    related_entities = models.ManyToManyField(
        "core.Entity",
        blank=True,
        related_name="hypotheses",
    )

    # For merged hypotheses
    merged_into = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="merged_from",
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_hypotheses",
    )

    class Meta:
        db_table = "core_hypothesis"
        verbose_name_plural = "hypotheses"
        ordering = ["-current_probability", "-created_at"]
        indexes = [
            models.Index(fields=["project", "status"]),
            models.Index(fields=["project", "-current_probability"]),
        ]

    def __str__(self):
        preview = self.statement[:50] if self.statement else self.if_clause[:50] if self.if_clause else "Untitled"
        return f"{preview}... ({self.current_probability:.0%})"

    def generate_statement(self) -> str:
        """Generate full statement from structured fields."""
        parts = []
        if self.if_clause:
            parts.append(f"If {self.if_clause}")
        if self.then_clause:
            parts.append(f"then {self.then_clause}")
        if self.because_clause:
            parts.append(f"because {self.because_clause}")
        return ", ".join(parts) + "." if parts else ""

    @property
    def odds(self) -> float:
        """Current odds (probability / (1 - probability))."""
        return BayesianUpdater.probability_to_odds(self.current_probability)

    @property
    def log_odds(self) -> float:
        """Log odds (useful for additive updates)."""
        return BayesianUpdater.probability_to_log_odds(self.current_probability)

    @property
    def evidence_count(self) -> int:
        return self.evidence_links.count()

    @property
    def supporting_evidence(self):
        """Evidence with LR > 1 (supports this hypothesis)."""
        return self.evidence_links.filter(likelihood_ratio__gt=1.0)

    @property
    def opposing_evidence(self):
        """Evidence with LR < 1 (opposes this hypothesis)."""
        return self.evidence_links.filter(likelihood_ratio__lt=1.0)

    def apply_evidence(self, evidence_link: "EvidenceLink") -> float:
        """Apply a single piece of evidence using Bayes' rule."""
        from django.utils import timezone

        updater = get_updater()
        lr = evidence_link.likelihood_ratio
        confidence = evidence_link.evidence.confidence

        result = updater.update(
            prior=self.current_probability,
            likelihood_ratio=lr,
            confidence=confidence,
        )

        self.probability_history.append({
            "probability": result.posterior_probability,
            "previous": result.prior_probability,
            "evidence_id": str(evidence_link.evidence.id),
            "likelihood_ratio": lr,
            "adjusted_lr": result.adjusted_likelihood_ratio,
            "strength": result.strength.value,
            "timestamp": timezone.now().isoformat(),
        })

        self.current_probability = result.posterior_probability
        self._check_status_thresholds()
        self.save()

        return result.posterior_probability

    def recalculate_probability(self):
        """Recalculate probability from all evidence links."""
        from django.utils import timezone

        updater = get_updater()

        evidence_items = [
            (link.likelihood_ratio, link.evidence.confidence)
            for link in self.evidence_links.all()
        ]

        result = updater.update_multiple(
            prior=self.prior_probability,
            evidence=evidence_items,
        )

        self.current_probability = result.posterior_probability
        self.probability_history.append({
            "probability": result.posterior_probability,
            "reason": "recalculated",
            "cumulative_lr": result.likelihood_ratio,
            "evidence_count": len(evidence_items),
            "timestamp": timezone.now().isoformat(),
        })
        self._check_status_thresholds()
        self.save()

    def _check_status_thresholds(self):
        """Update status based on probability thresholds."""
        if self.status not in (self.Status.ACTIVE, self.Status.UNCERTAIN):
            return

        if self.current_probability >= self.confirmation_threshold:
            self.status = self.Status.CONFIRMED
        elif self.current_probability <= self.rejection_threshold:
            self.status = self.Status.REJECTED
        elif 0.3 <= self.current_probability <= 0.7:
            self.status = self.Status.UNCERTAIN
        else:
            self.status = self.Status.ACTIVE


class Evidence(models.Model):
    """A piece of evidence that can affect hypothesis probabilities."""

    class SourceType(models.TextChoices):
        OBSERVATION = "observation", "Observation"
        SIMULATION = "simulation", "Simulation"
        ANALYSIS = "analysis", "Data Analysis"
        EXPERIMENT = "experiment", "Experiment"
        RESEARCH = "research", "Research"
        EXPERT = "expert", "Expert Opinion"
        CALCULATION = "calculation", "Calculation"

    class ResultType(models.TextChoices):
        STATISTICAL = "statistical", "Statistical"
        CATEGORICAL = "categorical", "Categorical"
        QUANTITATIVE = "quantitative", "Quantitative"
        QUALITATIVE = "qualitative", "Qualitative"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Link to project for easier querying
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.CASCADE,
        related_name="evidence",
        null=True,
        blank=True,
    )

    # Description
    summary = models.TextField(
        help_text="Brief description of what this evidence shows",
    )
    details = models.TextField(
        blank=True,
        help_text="Detailed description or methodology",
    )

    # Source
    source_type = models.CharField(
        max_length=20,
        choices=SourceType.choices,
        default=SourceType.OBSERVATION,
    )
    source_description = models.CharField(
        max_length=255,
        blank=True,
        help_text="Where this evidence came from",
    )

    # Result type
    result_type = models.CharField(
        max_length=20,
        choices=ResultType.choices,
        default=ResultType.QUALITATIVE,
    )

    # Confidence (0.0 to 1.0)
    confidence = models.FloatField(
        default=0.8,
        help_text="How reliable is this evidence? (0.0 to 1.0)",
    )

    # Statistical fields
    p_value = models.FloatField(null=True, blank=True)
    confidence_interval_low = models.FloatField(null=True, blank=True)
    confidence_interval_high = models.FloatField(null=True, blank=True)
    effect_size = models.FloatField(null=True, blank=True)
    sample_size = models.IntegerField(null=True, blank=True)
    statistical_test = models.CharField(max_length=100, blank=True)

    # Quantitative fields
    measured_value = models.FloatField(null=True, blank=True)
    expected_value = models.FloatField(null=True, blank=True)
    unit = models.CharField(max_length=50, blank=True)

    # Raw output storage
    raw_output = models.JSONField(default=dict, blank=True)

    # Reproducibility
    is_reproducible = models.BooleanField(default=False)
    code_reference = models.TextField(blank=True)
    data_reference = models.CharField(max_length=500, blank=True)

    # Link to knowledge graph
    related_entity = models.ForeignKey(
        "core.Entity",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="evidence",
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_evidence",
    )

    class Meta:
        db_table = "core_evidence"
        verbose_name_plural = "evidence"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.source_type}: {self.summary[:50]}..."

    @property
    def hypothesis_count(self) -> int:
        return self.hypothesis_links.count()


class EvidenceLink(models.Model):
    """Links evidence to a hypothesis with a specific likelihood ratio."""

    class Direction(models.TextChoices):
        SUPPORTS = "supports", "Supports"
        OPPOSES = "opposes", "Opposes"
        NEUTRAL = "neutral", "Neutral"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    hypothesis = models.ForeignKey(
        Hypothesis,
        on_delete=models.CASCADE,
        related_name="evidence_links",
    )
    evidence = models.ForeignKey(
        Evidence,
        on_delete=models.CASCADE,
        related_name="hypothesis_links",
    )

    # The likelihood ratio
    likelihood_ratio = models.FloatField(
        default=1.0,
        help_text="P(evidence|H true) / P(evidence|H false). >1 supports, <1 opposes",
    )

    direction = models.CharField(
        max_length=10,
        choices=Direction.choices,
        default=Direction.NEUTRAL,
    )

    reasoning = models.TextField(
        blank=True,
        help_text="Why does this evidence support/oppose with this strength?",
    )

    is_manual = models.BooleanField(
        default=True,
        help_text="Created by user (True) or inferred by system (False)?",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    applied_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "core_evidence_link"
        unique_together = [["hypothesis", "evidence"]]
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.evidence} → {self.hypothesis} (LR={self.likelihood_ratio})"

    def save(self, *args, **kwargs):
        # Auto-set direction based on LR
        if self.likelihood_ratio > 1.05:
            self.direction = self.Direction.SUPPORTS
        elif self.likelihood_ratio < 0.95:
            self.direction = self.Direction.OPPOSES
        else:
            self.direction = self.Direction.NEUTRAL
        super().save(*args, **kwargs)

    @property
    def strength(self) -> str:
        """Human-readable strength of evidence."""
        lr = self.likelihood_ratio
        if lr >= 10:
            return "very strong support"
        elif lr >= 3:
            return "strong support"
        elif lr >= 1.5:
            return "moderate support"
        elif lr > 1.05:
            return "weak support"
        elif lr <= 0.1:
            return "very strong opposition"
        elif lr <= 0.33:
            return "strong opposition"
        elif lr <= 0.67:
            return "moderate opposition"
        elif lr < 0.95:
            return "weak opposition"
        else:
            return "neutral"

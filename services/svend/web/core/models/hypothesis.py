"""Hypothesis and Evidence models for Bayesian reasoning.

This is the core of Synara's belief tracking:
- Hypotheses have prior and posterior probabilities
- Evidence has likelihood ratios
- EvidenceLink connects evidence to hypotheses with specific LRs
- Bayesian updates: posterior odds = prior odds × likelihood ratio

All probability updates use core.bayesian.BayesianUpdater.
"""

import uuid
from django.conf import settings
from django.db import models

from core.bayesian import BayesianUpdater, get_updater


class Hypothesis(models.Model):
    """A hypothesis about a potential cause or explanation.

    Hypotheses are beliefs that can be updated based on evidence.
    Synara uses Bayesian reasoning to update probabilities.

    The key equation:
        posterior_odds = prior_odds × likelihood_ratio
        P(H|E) = P(E|H) × P(H) / P(E)
    """

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"  # Under investigation
        CONFIRMED = "confirmed", "Confirmed"  # Strong evidence supports
        REJECTED = "rejected", "Rejected"  # Strong evidence against
        UNCERTAIN = "uncertain", "Uncertain"  # Not enough evidence
        MERGED = "merged", "Merged"  # Combined with another hypothesis

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        "core.Project",
        on_delete=models.CASCADE,
        related_name="hypotheses",
    )

    # The hypothesis itself
    statement = models.TextField(
        help_text="The hypothesis statement (e.g., 'The UI change caused the sales drop')",
    )
    mechanism = models.TextField(
        blank=True,
        help_text="How would this cause the effect? The proposed mechanism.",
    )

    # Probabilities (Bayesian)
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

    # Status
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE,
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

    # Testability
    is_testable = models.BooleanField(
        default=True,
        help_text="Can this hypothesis be tested with evidence?",
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
        help_text="Entities from knowledge graph related to this hypothesis",
    )

    # For merged hypotheses
    merged_into = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="merged_from",
    )

    # Metadata
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
        return f"{self.statement[:50]}... ({self.current_probability:.0%})"

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
        """Apply a single piece of evidence using Bayes' rule.

        Uses core.bayesian.BayesianUpdater for the actual math.
        Returns the new probability.
        """
        from django.utils import timezone

        updater = get_updater()

        # Get likelihood ratio and confidence
        lr = evidence_link.likelihood_ratio
        confidence = evidence_link.evidence.confidence

        # Perform Bayesian update
        result = updater.update(
            prior=self.current_probability,
            likelihood_ratio=lr,
            confidence=confidence,
        )

        # Record history
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
        """Recalculate probability from all evidence links.

        Uses core.bayesian.BayesianUpdater for the actual math.
        Useful after evidence is added, removed, or modified.
        """
        from django.utils import timezone

        updater = get_updater()

        # Gather all evidence as (lr, confidence) tuples
        evidence_items = [
            (link.likelihood_ratio, link.evidence.confidence)
            for link in self.evidence_links.all()
        ]

        # Perform cumulative Bayesian update from prior
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
            return  # Don't override manual status changes

        if self.current_probability >= self.confirmation_threshold:
            self.status = self.Status.CONFIRMED
        elif self.current_probability <= self.rejection_threshold:
            self.status = self.Status.REJECTED
        elif 0.3 <= self.current_probability <= 0.7:
            self.status = self.Status.UNCERTAIN
        else:
            self.status = self.Status.ACTIVE


class Evidence(models.Model):
    """A piece of evidence that can affect hypothesis probabilities.

    Evidence has:
    - A source (where it came from)
    - Structured data for Synara to process
    - Confidence in the evidence itself
    - Links to hypotheses via EvidenceLink (with specific LRs)
    """

    class SourceType(models.TextChoices):
        OBSERVATION = "observation", "Observation"
        SIMULATION = "simulation", "Simulation"
        ANALYSIS = "analysis", "Data Analysis"
        EXPERIMENT = "experiment", "Experiment"
        RESEARCH = "research", "Research"
        EXPERT = "expert", "Expert Opinion"
        CALCULATION = "calculation", "Calculation"

    class ResultType(models.TextChoices):
        STATISTICAL = "statistical", "Statistical"  # Has p-value, CI, etc.
        CATEGORICAL = "categorical", "Categorical"  # Yes/no, present/absent
        QUANTITATIVE = "quantitative", "Quantitative"  # Numeric measurement
        QUALITATIVE = "qualitative", "Qualitative"  # Descriptive

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

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
        help_text="Where this evidence came from (e.g., 'Coder simulation', 'DSW analysis')",
    )

    # Result type and structured data
    result_type = models.CharField(
        max_length=20,
        choices=ResultType.choices,
        default=ResultType.QUALITATIVE,
    )

    # Confidence in this evidence (0.0 to 1.0)
    # Lower confidence = LR moved toward 1 (neutral)
    confidence = models.FloatField(
        default=0.8,
        help_text="How reliable is this evidence? (0.0 to 1.0)",
    )

    # Statistical fields (for STATISTICAL result type)
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

    # Raw output (flexible storage for any structured data)
    raw_output = models.JSONField(
        default=dict,
        blank=True,
        help_text="Raw structured output from analysis/simulation",
    )

    # Reproducibility
    is_reproducible = models.BooleanField(default=False)
    code_reference = models.TextField(
        blank=True,
        help_text="Code that generated this evidence (for reproduction)",
    )
    data_reference = models.CharField(
        max_length=500,
        blank=True,
        help_text="Reference to data used (file path, dataset ID)",
    )

    # Link to knowledge graph entity (if this evidence is about a specific entity)
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
    """Links evidence to a hypothesis with a specific likelihood ratio.

    One piece of evidence can affect multiple hypotheses differently.
    For example:
    - "Sales dropped 40% after UI change" might have:
      - LR = 5.0 for "UI caused drop" (supports)
      - LR = 0.3 for "Seasonality caused drop" (opposes)

    The likelihood ratio is:
        LR = P(evidence | hypothesis true) / P(evidence | hypothesis false)

    LR > 1: Evidence supports hypothesis
    LR < 1: Evidence opposes hypothesis
    LR = 1: Evidence is neutral
    """

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

    # The likelihood ratio for this specific hypothesis
    likelihood_ratio = models.FloatField(
        default=1.0,
        help_text="P(evidence|H true) / P(evidence|H false). >1 supports, <1 opposes",
    )

    # Direction (derived from LR, but useful for queries)
    class Direction(models.TextChoices):
        SUPPORTS = "supports", "Supports"
        OPPOSES = "opposes", "Opposes"
        NEUTRAL = "neutral", "Neutral"

    direction = models.CharField(
        max_length=10,
        choices=Direction.choices,
        default=Direction.NEUTRAL,
    )

    # Explanation of why this LR was assigned
    reasoning = models.TextField(
        blank=True,
        help_text="Why does this evidence support/oppose this hypothesis with this strength?",
    )

    # Was this link created automatically or manually?
    is_manual = models.BooleanField(
        default=True,
        help_text="Was this link created by user (True) or inferred by system (False)?",
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    applied_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When this evidence was applied to update hypothesis probability",
    )

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

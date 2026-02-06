"""Unified Bayesian inference module.

This is THE source of truth for all probability updates in SVEND.
All components (DSW, SPC, Experimenter, Synara, Workbench) must use this.

The key equation:
    posterior_odds = prior_odds × likelihood_ratio
    P(H|E) = posterior_odds / (1 + posterior_odds)

Where:
    likelihood_ratio = P(E|H) / P(E|¬H)
    - LR > 1: Evidence supports hypothesis
    - LR < 1: Evidence opposes hypothesis
    - LR = 1: Evidence is neutral
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum


class EvidenceStrength(Enum):
    """Standardized evidence strength categories (Jeffreys' scale)."""
    VERY_STRONG_SUPPORT = "very_strong_support"      # LR >= 10
    STRONG_SUPPORT = "strong_support"                # LR >= 3
    MODERATE_SUPPORT = "moderate_support"            # LR >= 1.5
    WEAK_SUPPORT = "weak_support"                    # LR > 1.05
    NEUTRAL = "neutral"                              # 0.95 <= LR <= 1.05
    WEAK_OPPOSITION = "weak_opposition"              # LR < 0.95
    MODERATE_OPPOSITION = "moderate_opposition"      # LR <= 0.67
    STRONG_OPPOSITION = "strong_opposition"          # LR <= 0.33
    VERY_STRONG_OPPOSITION = "very_strong_opposition"  # LR <= 0.1


@dataclass
class BayesianUpdate:
    """Result of a Bayesian probability update."""
    prior_probability: float
    posterior_probability: float
    likelihood_ratio: float
    adjusted_likelihood_ratio: float  # After confidence adjustment
    confidence: float
    strength: EvidenceStrength
    log_odds_change: float

    @property
    def probability_change(self) -> float:
        return self.posterior_probability - self.prior_probability

    @property
    def direction(self) -> str:
        if self.likelihood_ratio > 1.05:
            return "supports"
        elif self.likelihood_ratio < 0.95:
            return "opposes"
        return "neutral"


class BayesianUpdater:
    """Unified Bayesian inference engine.

    Usage:
        updater = BayesianUpdater()

        # Single update
        result = updater.update(
            prior=0.5,
            likelihood_ratio=3.0,
            confidence=0.9
        )
        print(f"New probability: {result.posterior_probability}")

        # Multiple evidence items
        result = updater.update_multiple(
            prior=0.5,
            evidence=[(3.0, 0.9), (0.5, 0.8), (2.0, 1.0)]  # (LR, confidence) pairs
        )
    """

    # Bounds to prevent numerical issues
    MIN_PROBABILITY = 0.001
    MAX_PROBABILITY = 0.999

    # Default thresholds for status changes
    CONFIRMATION_THRESHOLD = 0.90
    REJECTION_THRESHOLD = 0.10

    def __init__(
        self,
        min_prob: float = 0.001,
        max_prob: float = 0.999,
        confirmation_threshold: float = 0.90,
        rejection_threshold: float = 0.10,
    ):
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.confirmation_threshold = confirmation_threshold
        self.rejection_threshold = rejection_threshold

    @staticmethod
    def probability_to_odds(p: float) -> float:
        """Convert probability to odds ratio."""
        p = max(0.001, min(0.999, p))
        return p / (1 - p)

    @staticmethod
    def odds_to_probability(odds: float) -> float:
        """Convert odds ratio to probability."""
        return odds / (1 + odds)

    @staticmethod
    def probability_to_log_odds(p: float) -> float:
        """Convert probability to log-odds (logit)."""
        p = max(0.001, min(0.999, p))
        return math.log(p / (1 - p))

    @staticmethod
    def log_odds_to_probability(log_odds: float) -> float:
        """Convert log-odds back to probability."""
        return 1 / (1 + math.exp(-log_odds))

    @classmethod
    def classify_strength(cls, lr: float) -> EvidenceStrength:
        """Classify evidence strength based on likelihood ratio (Jeffreys' scale)."""
        if lr >= 10:
            return EvidenceStrength.VERY_STRONG_SUPPORT
        elif lr >= 3:
            return EvidenceStrength.STRONG_SUPPORT
        elif lr >= 1.5:
            return EvidenceStrength.MODERATE_SUPPORT
        elif lr > 1.05:
            return EvidenceStrength.WEAK_SUPPORT
        elif lr <= 0.1:
            return EvidenceStrength.VERY_STRONG_OPPOSITION
        elif lr <= 0.33:
            return EvidenceStrength.STRONG_OPPOSITION
        elif lr <= 0.67:
            return EvidenceStrength.MODERATE_OPPOSITION
        elif lr < 0.95:
            return EvidenceStrength.WEAK_OPPOSITION
        else:
            return EvidenceStrength.NEUTRAL

    def adjust_likelihood_ratio(
        self,
        lr: float,
        confidence: float = 1.0
    ) -> float:
        """Adjust likelihood ratio based on evidence confidence.

        If confidence < 1, the LR is moved toward 1 (neutral).
        This reflects that uncertain evidence should have less impact.

        Formula: adjusted_lr = 1 + (lr - 1) * confidence

        Examples:
            - LR=3, confidence=1.0 → adjusted_lr=3.0
            - LR=3, confidence=0.5 → adjusted_lr=2.0
            - LR=3, confidence=0.0 → adjusted_lr=1.0 (neutral)
        """
        confidence = max(0.0, min(1.0, confidence))
        return 1 + (lr - 1) * confidence

    def update(
        self,
        prior: float,
        likelihood_ratio: float,
        confidence: float = 1.0,
    ) -> BayesianUpdate:
        """Perform a single Bayesian update.

        Args:
            prior: Prior probability (0-1)
            likelihood_ratio: P(E|H) / P(E|¬H)
            confidence: Confidence in the evidence (0-1), reduces LR impact

        Returns:
            BayesianUpdate with all details
        """
        # Clamp inputs
        prior = max(self.min_prob, min(self.max_prob, prior))
        lr = max(0.001, likelihood_ratio)  # Prevent zero/negative LR
        confidence = max(0.0, min(1.0, confidence))

        # Adjust LR for confidence
        adjusted_lr = self.adjust_likelihood_ratio(lr, confidence)

        # Bayesian update: posterior_odds = prior_odds × adjusted_lr
        prior_odds = self.probability_to_odds(prior)
        posterior_odds = prior_odds * adjusted_lr

        # Convert back to probability
        posterior = self.odds_to_probability(posterior_odds)

        # Clamp result
        posterior = max(self.min_prob, min(self.max_prob, posterior))

        # Calculate log-odds change (useful for interpretation)
        log_odds_change = math.log(adjusted_lr) if adjusted_lr > 0 else 0

        return BayesianUpdate(
            prior_probability=prior,
            posterior_probability=posterior,
            likelihood_ratio=lr,
            adjusted_likelihood_ratio=adjusted_lr,
            confidence=confidence,
            strength=self.classify_strength(lr),
            log_odds_change=log_odds_change,
        )

    def update_multiple(
        self,
        prior: float,
        evidence: List[Tuple[float, float]],
    ) -> BayesianUpdate:
        """Apply multiple pieces of evidence sequentially.

        Args:
            prior: Initial prior probability
            evidence: List of (likelihood_ratio, confidence) tuples

        Returns:
            BayesianUpdate representing the cumulative effect
        """
        if not evidence:
            return BayesianUpdate(
                prior_probability=prior,
                posterior_probability=prior,
                likelihood_ratio=1.0,
                adjusted_likelihood_ratio=1.0,
                confidence=1.0,
                strength=EvidenceStrength.NEUTRAL,
                log_odds_change=0.0,
            )

        current_prob = prior
        cumulative_lr = 1.0
        cumulative_adjusted_lr = 1.0

        for lr, conf in evidence:
            adjusted_lr = self.adjust_likelihood_ratio(lr, conf)
            cumulative_lr *= lr
            cumulative_adjusted_lr *= adjusted_lr

            # Update probability
            odds = self.probability_to_odds(current_prob)
            odds *= adjusted_lr
            current_prob = self.odds_to_probability(odds)
            current_prob = max(self.min_prob, min(self.max_prob, current_prob))

        log_odds_change = math.log(cumulative_adjusted_lr) if cumulative_adjusted_lr > 0 else 0

        return BayesianUpdate(
            prior_probability=prior,
            posterior_probability=current_prob,
            likelihood_ratio=cumulative_lr,
            adjusted_likelihood_ratio=cumulative_adjusted_lr,
            confidence=1.0,  # N/A for cumulative
            strength=self.classify_strength(cumulative_lr),
            log_odds_change=log_odds_change,
        )

    def recalculate_from_evidence(
        self,
        prior: float,
        evidence_links: list,
    ) -> float:
        """Recalculate probability from a list of evidence links.

        Args:
            prior: The prior probability
            evidence_links: List with .likelihood_ratio and .evidence.confidence attributes

        Returns:
            New probability
        """
        odds = self.probability_to_odds(prior)

        for link in evidence_links:
            lr = getattr(link, 'likelihood_ratio', 1.0)
            confidence = getattr(link.evidence, 'confidence', 1.0) if hasattr(link, 'evidence') else 1.0
            adjusted_lr = self.adjust_likelihood_ratio(lr, confidence)
            odds *= adjusted_lr

        posterior = self.odds_to_probability(odds)
        return max(self.min_prob, min(self.max_prob, posterior))

    def suggest_status(self, probability: float) -> str:
        """Suggest hypothesis status based on probability.

        Returns: 'confirmed', 'rejected', 'uncertain', or 'active'
        """
        if probability >= self.confirmation_threshold:
            return 'confirmed'
        elif probability <= self.rejection_threshold:
            return 'rejected'
        elif 0.3 <= probability <= 0.7:
            return 'uncertain'
        else:
            return 'active'

    def required_lr_for_confirmation(self, current_prob: float) -> float:
        """Calculate the LR needed to reach confirmation threshold."""
        if current_prob >= self.confirmation_threshold:
            return 1.0  # Already confirmed

        current_odds = self.probability_to_odds(current_prob)
        target_odds = self.probability_to_odds(self.confirmation_threshold)

        return target_odds / current_odds

    def required_lr_for_rejection(self, current_prob: float) -> float:
        """Calculate the LR needed to reach rejection threshold."""
        if current_prob <= self.rejection_threshold:
            return 1.0  # Already rejected

        current_odds = self.probability_to_odds(current_prob)
        target_odds = self.probability_to_odds(self.rejection_threshold)

        return target_odds / current_odds


# Global singleton for convenience
_default_updater: Optional[BayesianUpdater] = None


def get_updater() -> BayesianUpdater:
    """Get the default BayesianUpdater singleton."""
    global _default_updater
    if _default_updater is None:
        _default_updater = BayesianUpdater()
    return _default_updater


def update_probability(
    prior: float,
    likelihood_ratio: float,
    confidence: float = 1.0,
) -> float:
    """Convenience function for simple updates.

    Returns just the posterior probability.
    """
    result = get_updater().update(prior, likelihood_ratio, confidence)
    return result.posterior_probability


def classify_evidence_strength(likelihood_ratio: float) -> str:
    """Get human-readable evidence strength."""
    strength = BayesianUpdater.classify_strength(likelihood_ratio)
    return strength.value.replace('_', ' ')

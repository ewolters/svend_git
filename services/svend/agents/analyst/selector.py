"""
Model Selector

Automatically selects the best model(s) based on:
- Data characteristics (size, dimensionality, feature types)
- Task type (classification vs regression)
- User constraints (speed, interpretability)
"""

from dataclasses import dataclass, field
from enum import Enum

from .models import ModelType, TaskType, get_model_info


class Priority(Enum):
    """What to optimize for."""
    ACCURACY = "accuracy"
    INTERPRETABILITY = "interpretability"
    SPEED = "speed"
    BALANCED = "balanced"


@dataclass
class DataProfile:
    """Profile of the dataset characteristics."""
    n_samples: int
    n_features: int
    n_classes: int = 0  # 0 for regression
    class_balance: float = 1.0  # ratio of minority to majority class
    feature_types: dict = field(default_factory=dict)  # 'numeric', 'categorical'
    has_missing: bool = False
    is_high_dimensional: bool = False  # features > samples

    @classmethod
    def from_dataframe(cls, df, target_column: str) -> "DataProfile":
        """Create profile from pandas DataFrame."""
        import pandas as pd
        import numpy as np

        X = df.drop(columns=[target_column])
        y = df[target_column]

        n_samples = len(df)
        n_features = len(X.columns)

        # Determine if classification or regression
        if pd.api.types.is_numeric_dtype(y):
            unique_ratio = y.nunique() / len(y)
            if unique_ratio < 0.05 or y.nunique() <= 10:
                n_classes = y.nunique()
            else:
                n_classes = 0  # Regression
        else:
            n_classes = y.nunique()

        # Class balance
        if n_classes > 0:
            class_counts = y.value_counts()
            class_balance = class_counts.min() / class_counts.max()
        else:
            class_balance = 1.0

        # Feature types
        feature_types = {
            'numeric': len(X.select_dtypes(include=[np.number]).columns),
            'categorical': len(X.select_dtypes(include=['object', 'category']).columns),
        }

        return cls(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_balance=class_balance,
            feature_types=feature_types,
            has_missing=df.isna().any().any(),
            is_high_dimensional=n_features > n_samples,
        )


@dataclass
class ModelRecommendation:
    """A model recommendation with reasoning."""
    model_type: ModelType
    score: float  # 0-1, higher is better fit
    reasons: list[str]
    warnings: list[str] = field(default_factory=list)


class ModelSelector:
    """
    Select the best model(s) for a dataset.

    Usage:
        selector = ModelSelector()
        profile = DataProfile.from_dataframe(df, 'target')
        recommendations = selector.recommend(profile, priority=Priority.BALANCED)

        # Get top recommendation
        best = recommendations[0]
        print(f"Recommended: {best.model_type.value}")
        print(f"Reasons: {best.reasons}")
    """

    def recommend(
        self,
        profile: DataProfile,
        priority: Priority = Priority.BALANCED,
        exclude: list[ModelType] = None,
    ) -> list[ModelRecommendation]:
        """
        Recommend models based on data profile.

        Returns list of ModelRecommendation sorted by score (best first).
        """
        exclude = exclude or []
        task_type = TaskType.CLASSIFICATION if profile.n_classes > 0 else TaskType.REGRESSION

        recommendations = []

        # Score each model
        for model_type in ModelType:
            if model_type in exclude:
                continue

            # Skip models that don't fit the task
            info = get_model_info(model_type)
            if model_type == ModelType.LINEAR_REGRESSION and task_type == TaskType.CLASSIFICATION:
                continue
            if model_type == ModelType.LOGISTIC_REGRESSION and task_type == TaskType.REGRESSION:
                continue
            if model_type == ModelType.NAIVE_BAYES and task_type == TaskType.REGRESSION:
                continue

            score, reasons, warnings = self._score_model(
                model_type, profile, task_type, priority
            )

            if score > 0:
                recommendations.append(ModelRecommendation(
                    model_type=model_type,
                    score=score,
                    reasons=reasons,
                    warnings=warnings,
                ))

        # Sort by score (descending)
        recommendations.sort(key=lambda r: r.score, reverse=True)

        return recommendations

    def _score_model(
        self,
        model_type: ModelType,
        profile: DataProfile,
        task_type: TaskType,
        priority: Priority,
    ) -> tuple[float, list[str], list[str]]:
        """Score a model for the given profile."""
        score = 0.5  # Base score
        reasons = []
        warnings = []

        info = get_model_info(model_type)
        n = profile.n_samples
        p = profile.n_features

        # === Size-based scoring ===

        # Small datasets (< 1000)
        if n < 1000:
            if model_type == ModelType.NAIVE_BAYES:
                score += 0.15
                reasons.append("Naive Bayes works well with small datasets")
            elif model_type == ModelType.KNN:
                score += 0.1
                reasons.append("KNN is viable for small datasets")
            elif model_type in [ModelType.LOGISTIC_REGRESSION, ModelType.LINEAR_REGRESSION]:
                score += 0.1
                reasons.append("Linear models are stable with small datasets")

        # Medium datasets (1000-10000)
        elif n < 10000:
            if model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING]:
                score += 0.15
                reasons.append("Ensemble methods shine with medium-sized data")
            elif model_type == ModelType.SVM:
                score += 0.1
                reasons.append("SVM is efficient at this scale")

        # Large datasets (> 10000)
        else:
            if model_type == ModelType.SVM:
                score -= 0.2
                warnings.append("SVM can be slow with large datasets")
            elif model_type == ModelType.KNN:
                score -= 0.15
                warnings.append("KNN prediction time increases with dataset size")
            elif model_type in [ModelType.LOGISTIC_REGRESSION, ModelType.LINEAR_REGRESSION]:
                score += 0.1
                reasons.append("Linear models scale well to large datasets")
            elif model_type == ModelType.GRADIENT_BOOSTING:
                score += 0.1
                reasons.append("Gradient boosting handles large datasets well")

        # === High-dimensional data ===
        if profile.is_high_dimensional:
            if model_type in [ModelType.LOGISTIC_REGRESSION, ModelType.LINEAR_REGRESSION]:
                score += 0.15
                reasons.append("Linear models with regularization handle high-D data")
            elif model_type == ModelType.SVM:
                score += 0.1
                reasons.append("SVM is effective in high-dimensional spaces")
            elif model_type == ModelType.KNN:
                score -= 0.2
                warnings.append("KNN suffers from curse of dimensionality")
            elif model_type == ModelType.NAIVE_BAYES:
                score += 0.1
                reasons.append("Naive Bayes handles high-dimensional data well")

        # === Class imbalance ===
        if profile.n_classes > 0 and profile.class_balance < 0.3:
            if model_type == ModelType.LOGISTIC_REGRESSION:
                score += 0.1
                reasons.append("Logistic regression can use class_weight='balanced'")
            elif model_type == ModelType.RANDOM_FOREST:
                score += 0.1
                reasons.append("Random forest can handle class imbalance")
            elif model_type == ModelType.NAIVE_BAYES:
                score -= 0.1
                warnings.append("Naive Bayes may struggle with class imbalance")

        # === Multiclass ===
        if profile.n_classes > 2:
            if model_type == ModelType.NAIVE_BAYES:
                score += 0.1
                reasons.append("Naive Bayes naturally handles multiclass")
            elif model_type == ModelType.RANDOM_FOREST:
                score += 0.1
                reasons.append("Random forest handles multiclass natively")
            elif model_type == ModelType.SVM:
                score -= 0.05
                warnings.append("SVM uses one-vs-one for multiclass (slower)")

        # === Priority adjustments ===
        if priority == Priority.INTERPRETABILITY:
            if info.interpretability == "high":
                score += 0.2
                reasons.append(f"High interpretability (your priority)")
            elif info.interpretability == "low":
                score -= 0.15

        elif priority == Priority.SPEED:
            if model_type in [ModelType.NAIVE_BAYES, ModelType.LINEAR_REGRESSION,
                              ModelType.LOGISTIC_REGRESSION]:
                score += 0.2
                reasons.append("Fast training and prediction")
            elif model_type in [ModelType.SVM] and n > 5000:
                score -= 0.2

        elif priority == Priority.ACCURACY:
            if model_type in [ModelType.GRADIENT_BOOSTING, ModelType.RANDOM_FOREST]:
                score += 0.15
                reasons.append("Typically achieves high accuracy")

        elif priority == Priority.BALANCED:
            # Slight preference for interpretable models
            if info.interpretability == "high":
                score += 0.05

        # Clamp score
        score = max(0.0, min(1.0, score))

        return score, reasons, warnings

    def auto_select(
        self,
        profile: DataProfile,
        priority: Priority = Priority.BALANCED,
    ) -> ModelType:
        """
        Automatically select the single best model.

        Returns the top recommended ModelType.
        """
        recommendations = self.recommend(profile, priority)
        if not recommendations:
            # Fallback to random forest
            return ModelType.RANDOM_FOREST
        return recommendations[0].model_type


def quick_select(df, target: str, priority: str = "balanced") -> ModelRecommendation:
    """
    Quick model selection helper.

    Args:
        df: pandas DataFrame
        target: name of target column
        priority: "accuracy", "interpretability", "speed", or "balanced"

    Returns:
        Top ModelRecommendation
    """
    selector = ModelSelector()
    profile = DataProfile.from_dataframe(df, target)
    priority_enum = Priority(priority)
    recommendations = selector.recommend(profile, priority_enum)
    return recommendations[0] if recommendations else None

"""
Model Definitions and Information

Six interpretable model types - no black boxes.
Each model includes educational information.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Callable


class ModelType(Enum):
    """Supported model types."""
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    KNN = "knn"
    NAIVE_BAYES = "naive_bayes"
    GRADIENT_BOOSTING = "gradient_boosting"


class TaskType(Enum):
    """Type of ML task."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


@dataclass
class ModelInfo:
    """Educational information about a model."""
    name: str
    model_type: ModelType
    task_type: TaskType
    description: str
    how_it_works: str
    strengths: list[str]
    weaknesses: list[str]
    when_to_use: str
    interpretability: str  # "high", "medium", "low"
    hyperparameters: dict[str, str]  # param -> description
    complexity: str  # "O(n)", "O(n^2)", etc.


# Educational content for each model
MODEL_INFO = {
    ModelType.LINEAR_REGRESSION: ModelInfo(
        name="Linear Regression",
        model_type=ModelType.LINEAR_REGRESSION,
        task_type=TaskType.REGRESSION,
        description="Predicts a continuous value as a weighted sum of features.",
        how_it_works="""
Linear regression finds the best-fitting line (or hyperplane) through your data.

The model learns weights (coefficients) for each feature:
    y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

It minimizes the sum of squared errors between predictions and actual values.
The coefficients tell you exactly how much each feature contributes to the prediction.
""",
        strengths=[
            "Highly interpretable - coefficients show feature importance",
            "Fast to train and predict",
            "Works well when relationships are actually linear",
            "Provides confidence intervals for predictions",
            "Doesn't overfit with proper regularization",
        ],
        weaknesses=[
            "Assumes linear relationships (may miss complex patterns)",
            "Sensitive to outliers",
            "Requires features to be independent (multicollinearity issues)",
            "Can't capture interactions without manual feature engineering",
        ],
        when_to_use="Use when you need interpretability, your features have roughly linear relationships with the target, and you want to understand which features matter most.",
        interpretability="high",
        hyperparameters={
            "fit_intercept": "Whether to calculate the intercept (usually True)",
            "alpha (Ridge)": "Regularization strength - higher values = simpler model",
        },
        complexity="O(np²) where n=samples, p=features",
    ),

    ModelType.LOGISTIC_REGRESSION: ModelInfo(
        name="Logistic Regression",
        model_type=ModelType.LOGISTIC_REGRESSION,
        task_type=TaskType.CLASSIFICATION,
        description="Predicts probabilities for classification using the logistic function.",
        how_it_works="""
Logistic regression predicts the probability that an instance belongs to a class.

It uses the logistic (sigmoid) function to squash linear combinations into [0, 1]:
    P(y=1) = 1 / (1 + e^(-(w₀ + w₁x₁ + ... + wₙxₙ)))

The model learns weights that maximize the likelihood of the observed data.
Positive weights increase probability, negative weights decrease it.
""",
        strengths=[
            "Outputs probabilities, not just class labels",
            "Coefficients are interpretable (log-odds)",
            "Fast and efficient",
            "Works well for binary and multiclass problems",
            "Provides feature importance through coefficients",
        ],
        weaknesses=[
            "Assumes linear decision boundary",
            "Can struggle with highly non-linear patterns",
            "Requires feature scaling for regularization",
            "Sensitive to class imbalance",
        ],
        when_to_use="Use for classification when you need probability outputs, interpretability is important, and the classes are reasonably separable with a linear boundary.",
        interpretability="high",
        hyperparameters={
            "C": "Inverse regularization strength - smaller = more regularization",
            "penalty": "Type of regularization: 'l1' (sparse), 'l2' (default), or 'none'",
            "class_weight": "Set to 'balanced' for imbalanced datasets",
        },
        complexity="O(np) per iteration, typically converges quickly",
    ),

    ModelType.RANDOM_FOREST: ModelInfo(
        name="Random Forest",
        model_type=ModelType.RANDOM_FOREST,
        task_type=TaskType.CLASSIFICATION,  # Can do both, set at runtime
        description="Ensemble of decision trees that vote on predictions.",
        how_it_works="""
Random Forest builds many decision trees and combines their predictions.

Each tree is trained on:
1. A random sample of the data (bagging)
2. A random subset of features at each split

For classification: trees vote, majority wins
For regression: trees average their predictions

This randomness reduces overfitting and improves generalization.
""",
        strengths=[
            "Handles non-linear relationships automatically",
            "Robust to outliers and noise",
            "Provides feature importance rankings",
            "Works well out-of-the-box with little tuning",
            "Can handle missing values and mixed feature types",
        ],
        weaknesses=[
            "Less interpretable than linear models",
            "Can be slow with many trees",
            "May overfit on noisy data with too many trees",
            "Biased toward features with more categories",
        ],
        when_to_use="Use when you have complex, non-linear relationships, want good performance without much tuning, and can sacrifice some interpretability for accuracy.",
        interpretability="medium",
        hyperparameters={
            "n_estimators": "Number of trees (more = better but slower, typically 100-500)",
            "max_depth": "Maximum tree depth (None = unlimited, 10-20 often good)",
            "min_samples_split": "Minimum samples to split a node (prevents overfitting)",
            "max_features": "Features to consider at each split ('sqrt' is common)",
        },
        complexity="O(n × m × log(n) × k) where n=samples, m=features, k=trees",
    ),

    ModelType.SVM: ModelInfo(
        name="Support Vector Machine",
        model_type=ModelType.SVM,
        task_type=TaskType.CLASSIFICATION,
        description="Finds the optimal hyperplane that separates classes with maximum margin.",
        how_it_works="""
SVM finds the decision boundary that maximizes the margin between classes.

Key concepts:
- Support vectors: the data points closest to the decision boundary
- Margin: the distance between the boundary and nearest points
- Kernel trick: transforms data to higher dimensions where it's linearly separable

The 'C' parameter controls the trade-off between a smooth boundary and classifying
training points correctly (higher C = more complex boundary, risk of overfitting).
""",
        strengths=[
            "Effective in high-dimensional spaces",
            "Works well with clear margin of separation",
            "Memory efficient (only stores support vectors)",
            "Versatile through different kernel functions",
        ],
        weaknesses=[
            "Slow on large datasets (O(n²) or worse)",
            "Sensitive to feature scaling",
            "Doesn't provide probability estimates by default",
            "Kernel choice and tuning can be tricky",
        ],
        when_to_use="Use for medium-sized datasets with clear class separation, especially when you have more features than samples. Good for text classification and image recognition.",
        interpretability="low",
        hyperparameters={
            "C": "Regularization (higher = less regularization, more complex boundary)",
            "kernel": "Type of kernel: 'rbf' (default), 'linear', 'poly', 'sigmoid'",
            "gamma": "Kernel coefficient ('scale' or 'auto' usually works)",
        },
        complexity="O(n² to n³) for training, O(n_sv × m) for prediction",
    ),

    ModelType.KNN: ModelInfo(
        name="K-Nearest Neighbors",
        model_type=ModelType.KNN,
        task_type=TaskType.CLASSIFICATION,
        description="Classifies based on the majority vote of k nearest training examples.",
        how_it_works="""
KNN is the simplest ML algorithm: store all training data, then for each prediction:

1. Calculate distances from the new point to all training points
2. Find the k closest training points
3. For classification: majority vote among k neighbors
4. For regression: average of k neighbors' values

No actual learning happens during training - it's "lazy learning."
""",
        strengths=[
            "Simple to understand and implement",
            "No training phase (instant 'fitting')",
            "Naturally handles multi-class problems",
            "Non-parametric - makes no assumptions about data distribution",
            "Can capture complex decision boundaries",
        ],
        weaknesses=[
            "Slow prediction (must compute all distances)",
            "Sensitive to irrelevant features",
            "Requires feature scaling",
            "Struggles with high-dimensional data (curse of dimensionality)",
            "Memory intensive (stores entire training set)",
        ],
        when_to_use="Use for small to medium datasets, when you want a simple baseline, or when the decision boundary is expected to be irregular.",
        interpretability="high",
        hyperparameters={
            "n_neighbors": "Number of neighbors to consider (odd numbers avoid ties)",
            "weights": "'uniform' (equal) or 'distance' (closer = more influence)",
            "metric": "Distance measure: 'euclidean', 'manhattan', 'minkowski'",
        },
        complexity="O(1) for training, O(n × m) for prediction",
    ),

    ModelType.NAIVE_BAYES: ModelInfo(
        name="Naive Bayes",
        model_type=ModelType.NAIVE_BAYES,
        task_type=TaskType.CLASSIFICATION,
        description="Probabilistic classifier based on Bayes' theorem with feature independence assumption.",
        how_it_works="""
Naive Bayes uses probability to classify:

P(class|features) ∝ P(class) × P(feature₁|class) × P(feature₂|class) × ...

The "naive" assumption: features are independent given the class.
Despite this often being wrong, it works surprisingly well in practice.

Variants:
- GaussianNB: continuous features (assumes normal distribution)
- MultinomialNB: count data (text classification)
- BernoulliNB: binary features
""",
        strengths=[
            "Extremely fast training and prediction",
            "Works well with small datasets",
            "Handles high-dimensional data well",
            "Robust to irrelevant features",
            "Provides probability estimates",
        ],
        weaknesses=[
            "Independence assumption rarely holds",
            "Cannot learn interactions between features",
            "Probability estimates can be poorly calibrated",
            "Sensitive to feature distribution assumptions",
        ],
        when_to_use="Use for text classification, spam filtering, when features are roughly independent, or when you need a fast baseline.",
        interpretability="high",
        hyperparameters={
            "var_smoothing": "Portion of variance added for stability (GaussianNB)",
            "alpha": "Smoothing parameter for MultinomialNB (prevents zero probabilities)",
        },
        complexity="O(n × m) for training, O(m × k) for prediction (k=classes)",
    ),

    ModelType.GRADIENT_BOOSTING: ModelInfo(
        name="Gradient Boosting",
        model_type=ModelType.GRADIENT_BOOSTING,
        task_type=TaskType.CLASSIFICATION,
        description="Builds trees sequentially, each correcting the errors of the previous ones.",
        how_it_works="""
Gradient Boosting builds an ensemble of weak learners (usually shallow trees):

1. Start with a simple prediction (mean or most common class)
2. Calculate the errors (residuals)
3. Train a tree to predict these residuals
4. Add this tree to the ensemble (with a learning rate)
5. Repeat steps 2-4

Each new tree focuses on the mistakes of the current ensemble.
The learning rate controls how much each tree contributes.
""",
        strengths=[
            "Often achieves best accuracy on tabular data",
            "Handles non-linear relationships",
            "Robust to outliers (with certain loss functions)",
            "Provides feature importance",
            "Handles mixed feature types",
        ],
        weaknesses=[
            "Prone to overfitting without careful tuning",
            "Slower than Random Forest (sequential training)",
            "Sensitive to hyperparameters",
            "Less interpretable than simpler models",
        ],
        when_to_use="Use when accuracy is the priority, you have tabular data, and you're willing to tune hyperparameters carefully.",
        interpretability="medium",
        hyperparameters={
            "n_estimators": "Number of boosting stages (100-1000 typical)",
            "learning_rate": "Shrinkage per tree (0.01-0.3, lower = needs more trees)",
            "max_depth": "Depth of each tree (3-10, shallower = more trees needed)",
            "subsample": "Fraction of samples per tree (0.8-1.0, adds randomness)",
        },
        complexity="O(n × m × d × k) where d=depth, k=trees",
    ),
}


def get_model_info(model_type: ModelType) -> ModelInfo:
    """Get educational information about a model type."""
    return MODEL_INFO.get(model_type)


def get_all_models() -> dict[ModelType, ModelInfo]:
    """Get info for all models."""
    return MODEL_INFO.copy()


def get_models_for_task(task: TaskType) -> list[ModelType]:
    """Get model types suitable for a task."""
    result = []
    for model_type, info in MODEL_INFO.items():
        # Some models work for both tasks
        if model_type in [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING,
                          ModelType.SVM, ModelType.KNN]:
            result.append(model_type)
        elif info.task_type == task:
            result.append(model_type)
    return result


def create_model(model_type: ModelType, task_type: TaskType, **kwargs):
    """
    Create a scikit-learn model instance.

    Args:
        model_type: Type of model to create
        task_type: Whether this is regression or classification
        **kwargs: Hyperparameters to override defaults

    Returns:
        Configured scikit-learn model
    """
    from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB

    if model_type == ModelType.LINEAR_REGRESSION:
        if task_type == TaskType.CLASSIFICATION:
            raise ValueError("Linear regression is for regression tasks only")
        alpha = kwargs.pop('alpha', None)
        if alpha:
            return Ridge(alpha=alpha, **kwargs)
        return LinearRegression(**kwargs)

    elif model_type == ModelType.LOGISTIC_REGRESSION:
        if task_type == TaskType.REGRESSION:
            raise ValueError("Logistic regression is for classification tasks only")
        defaults = {'max_iter': 1000, 'random_state': 42}
        defaults.update(kwargs)
        return LogisticRegression(**defaults)

    elif model_type == ModelType.RANDOM_FOREST:
        defaults = {'n_estimators': 100, 'random_state': 42}
        defaults.update(kwargs)
        if task_type == TaskType.CLASSIFICATION:
            return RandomForestClassifier(**defaults)
        return RandomForestRegressor(**defaults)

    elif model_type == ModelType.SVM:
        defaults = {'random_state': 42}
        defaults.update(kwargs)
        if task_type == TaskType.CLASSIFICATION:
            defaults.setdefault('probability', True)  # For predict_proba
            return SVC(**defaults)
        return SVR(**defaults)

    elif model_type == ModelType.KNN:
        defaults = {'n_neighbors': 5}
        defaults.update(kwargs)
        if task_type == TaskType.CLASSIFICATION:
            return KNeighborsClassifier(**defaults)
        return KNeighborsRegressor(**defaults)

    elif model_type == ModelType.NAIVE_BAYES:
        if task_type == TaskType.REGRESSION:
            raise ValueError("Naive Bayes is for classification tasks only")
        return GaussianNB(**kwargs)

    elif model_type == ModelType.GRADIENT_BOOSTING:
        defaults = {'n_estimators': 100, 'random_state': 42}
        defaults.update(kwargs)
        if task_type == TaskType.CLASSIFICATION:
            return GradientBoostingClassifier(**defaults)
        return GradientBoostingRegressor(**defaults)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

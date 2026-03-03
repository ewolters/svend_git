"""
Analyst - Simple ML Training Service

Train interpretable ML models with educational outputs.
No neural networks - just models you can understand.

Models available:
- Linear/Logistic Regression (GLM)
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Gradient Boosting

Usage:
    from analyst import Analyst, TrainingRequest

    analyst = Analyst()
    result = analyst.train(
        data=df,
        target='outcome',
        intent='predict customer churn',
    )

    # Get the model
    model = result.model

    # Get reproducible code
    print(result.code)

    # Get educational report
    print(result.report())

    # Save everything
    result.save('output/')
"""

from .trainer import (
    Analyst,
    TrainingRequest,
    TrainingResult,
    DataQualityWarning,
    DataQualityAssessment,
    WarningSeverity,
    assess_data_quality,
)
from .models import ModelType, get_model_info
from .selector import ModelSelector
from .reporter import AnalystReporter
from .eda import EDAAnalyzer, EDAReport, quick_eda

__all__ = [
    "Analyst",
    "TrainingRequest",
    "TrainingResult",
    "ModelType",
    "get_model_info",
    "ModelSelector",
    "AnalystReporter",
    # EDA
    "EDAAnalyzer",
    "EDAReport",
    "quick_eda",
    # Data Quality
    "DataQualityWarning",
    "DataQualityAssessment",
    "WarningSeverity",
    "assess_data_quality",
]

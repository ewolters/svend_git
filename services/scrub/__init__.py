"""
Scrub - Data Cleaning Service

Automated data cleaning with human review:
- Outlier detection (flag for review, don't auto-delete)
- Missing data handling (imputation strategies)
- Factor normalization (case, typos, fuzzy matching)
- Type validation and correction
- Full audit trail

Philosophy: Clean data, clear reports, human decisions on edge cases.
"""

__version__ = "1.0.0"

from .cleaner import DataCleaner, CleaningResult, CleaningConfig
from .outliers import OutlierDetector, OutlierResult
from .missing import MissingHandler, ImputationStrategy
from .normalize import FactorNormalizer, NormalizationResult
from .types import TypeInferrer, TypeCorrection

__all__ = [
    "DataCleaner",
    "CleaningResult",
    "CleaningConfig",
    "OutlierDetector",
    "OutlierResult",
    "MissingHandler",
    "ImputationStrategy",
    "FactorNormalizer",
    "NormalizationResult",
    "TypeInferrer",
    "TypeCorrection",
]

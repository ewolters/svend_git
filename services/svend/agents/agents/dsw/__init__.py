"""
Decision Science Workbench

Orchestrate the full ML lifecycle:

    Intent → Research → Schema → Data (Forge) → Clean (Scrub) → Train (Analyst) → Deploy

Each component is standalone. DSW connects them.

## The Zero-to-Classifier Flow

Someone with an idea but NO data can:

1. Describe their intent: "I want to predict customer churn for a B2B SaaS"
2. Research generates a problem schema with:
   - Relevant features and their distributions
   - Domain constraints
   - Edge cases to consider
   - Success metrics
3. Forge generates synthetic training data from the schema
4. Scrub validates and cleans the data
5. Analyst trains and compares models
6. DSW produces:
   - Trained model
   - Deployment code
   - Improvement plan (what data to collect, how to validate)

## Standalone Services

Each can be used independently:

- `scrub/` - Data cleaning (outliers, missing, types, normalization)
- `analyst/` - ML training (model selection, training, QA reports)
- `researcher/` - Multi-source research with citations
- `coder/` - Code generation with verification

## Usage

```python
from dsw import DecisionScienceWorkbench

dsw = DecisionScienceWorkbench()

# Full pipeline from intent
result = dsw.from_intent(
    intent="Predict which customers will churn in the next 30 days",
    domain="B2B SaaS",
)

# Or with existing data
result = dsw.from_data(
    data=df,
    target='churned',
    intent="Predict customer churn",
)

# Get everything
print(result.schema)           # Problem specification
print(result.model)            # Trained model
print(result.code)             # Deployment code
print(result.improvement_plan) # What to do next
result.save('output/')
```
"""

from .workbench import DecisionScienceWorkbench, DSWResult
from .schema import ProblemSchema, FeatureSpec, SchemaGenerator
from .forge_client import ForgeClient
from .validation import (
    DSWPipelineValidator,
    PipelineValidationReport,
    PipelineStageTracker,
    InterfaceValidator,
    StageStatus,
    validate_dsw_result,
)

__all__ = [
    "DecisionScienceWorkbench",
    "DSWResult",
    "ProblemSchema",
    "FeatureSpec",
    "SchemaGenerator",
    "ForgeClient",
    # Validation
    "DSWPipelineValidator",
    "PipelineValidationReport",
    "PipelineStageTracker",
    "InterfaceValidator",
    "StageStatus",
    "validate_dsw_result",
]

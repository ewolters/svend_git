"""
Forge - Synthetic Data Generation API

Generate high-quality synthetic data for ML training:
- Tabular data (structured records with schema definition)
- Text data (conversations, reviews, support tickets)

Quality tiers:
- Standard: Schema validation + basic stats
- Premium: Full pipeline validation + Synara verification

Domains:
- E-commerce (products, orders, reviews, user profiles)
- Customer service (tickets, chat conversations)

Usage:
    from forge import TabularGenerator, TabularSchema

    schema = TabularSchema.from_dict({
        "fields": {
            "product_name": {"type": "string"},
            "price": {"type": "float", "min": 0.01, "max": 10000, "distribution": "normal"},
            "category": {"type": "category", "values": ["electronics", "clothing"]},
        }
    })

    generator = TabularGenerator()
    df = generator.generate(schema, n=1000)

    # Run QA
    from forge import ForgeQA
    qa = ForgeQA()
    report = qa.analyze(df, schema)
    print(report.summary())
"""

__version__ = "0.1.0"

from .schemas import TabularSchema, FieldSpec, FieldType, SchemaValidationError
from .generators import TabularGenerator, Distribution
from .qa import ForgeQA, ForgeQualityReport, FieldStats

__all__ = [
    "__version__",
    "TabularSchema",
    "FieldSpec",
    "FieldType",
    "SchemaValidationError",
    "TabularGenerator",
    "Distribution",
    "ForgeQA",
    "ForgeQualityReport",
    "FieldStats",
]

"""
Problem Schema Generator

Converts intent + research into a structured schema that:
1. Forge can use to generate synthetic data
2. Analyst can use to understand the problem
3. Documents the assumptions and constraints

The schema is the "contract" between research and generation.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union
from datetime import datetime


class FeatureType(Enum):
    """Types of features."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"


class DistributionType(Enum):
    """Statistical distributions for numeric features."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"
    POISSON = "poisson"
    LOGNORMAL = "lognormal"
    BETA = "beta"
    CUSTOM = "custom"


@dataclass
class NumericConstraints:
    """Constraints for numeric features."""
    min_value: float = None
    max_value: float = None
    distribution: DistributionType = DistributionType.NORMAL
    mean: float = None
    std: float = None
    # For custom distributions
    distribution_params: dict = field(default_factory=dict)


@dataclass
class CategoricalConstraints:
    """Constraints for categorical features."""
    categories: list[str] = field(default_factory=list)
    probabilities: list[float] = None  # Must sum to 1 if provided
    allow_missing: bool = False


@dataclass
class FeatureSpec:
    """Specification for a single feature."""
    name: str
    feature_type: FeatureType
    description: str

    # Constraints (type-specific)
    constraints: Union[NumericConstraints, CategoricalConstraints, dict] = None

    # Relationships
    correlated_with: list[str] = field(default_factory=list)
    correlation_strength: float = 0.0  # -1 to 1

    # Metadata
    importance: str = "medium"  # low, medium, high, critical
    source: str = ""  # Where this feature comes from
    collection_difficulty: str = "easy"  # easy, moderate, hard

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "type": self.feature_type.value,
            "description": self.description,
            "importance": self.importance,
        }
        if self.constraints:
            if isinstance(self.constraints, NumericConstraints):
                result["constraints"] = {
                    "min": self.constraints.min_value,
                    "max": self.constraints.max_value,
                    "distribution": self.constraints.distribution.value,
                    "mean": self.constraints.mean,
                    "std": self.constraints.std,
                }
            elif isinstance(self.constraints, CategoricalConstraints):
                result["constraints"] = {
                    "categories": self.constraints.categories,
                    "probabilities": self.constraints.probabilities,
                }
            else:
                result["constraints"] = self.constraints
        if self.correlated_with:
            result["correlations"] = {
                "features": self.correlated_with,
                "strength": self.correlation_strength,
            }
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSpec":
        """Create from dictionary."""
        feature_type = FeatureType(data.get("type", "numeric"))

        constraints = None
        if "constraints" in data:
            c = data["constraints"]
            if feature_type == FeatureType.NUMERIC:
                constraints = NumericConstraints(
                    min_value=c.get("min"),
                    max_value=c.get("max"),
                    distribution=DistributionType(c.get("distribution", "normal")),
                    mean=c.get("mean"),
                    std=c.get("std"),
                )
            elif feature_type == FeatureType.CATEGORICAL:
                constraints = CategoricalConstraints(
                    categories=c.get("categories", []),
                    probabilities=c.get("probabilities"),
                )

        return cls(
            name=data["name"],
            feature_type=feature_type,
            description=data.get("description", ""),
            constraints=constraints,
            importance=data.get("importance", "medium"),
        )


@dataclass
class EdgeCase:
    """An edge case to consider in the model."""
    name: str
    description: str
    frequency: str  # rare, occasional, common
    handling: str  # How to handle this case
    test_scenario: dict = field(default_factory=dict)  # Example data


@dataclass
class SuccessMetric:
    """How to measure success."""
    name: str
    description: str
    target_value: float = None
    minimum_acceptable: float = None
    measurement_method: str = ""


@dataclass
class ProblemSchema:
    """
    Complete specification of an ML problem.

    This is the contract between:
    - Research (generates it)
    - Forge (uses it to create data)
    - Analyst (uses it to understand constraints)
    - Deployment (uses it to validate inputs)
    """
    # Core
    name: str
    intent: str
    domain: str
    task_type: str  # classification, regression

    # Target
    target_name: str
    target_description: str
    target_type: FeatureType = FeatureType.CATEGORICAL
    target_classes: list[str] = field(default_factory=list)  # For classification

    # Features
    features: list[FeatureSpec] = field(default_factory=list)

    # Constraints and edge cases
    domain_rules: dict = field(default_factory=dict)  # feature -> (min, max)
    edge_cases: list[EdgeCase] = field(default_factory=list)

    # Success criteria
    success_metrics: list[SuccessMetric] = field(default_factory=list)
    baseline_performance: float = None  # What random/naive would achieve

    # Data generation hints
    sample_size_recommendation: int = 1000
    class_balance: dict = field(default_factory=dict)  # class -> proportion

    # Improvement plan
    data_to_collect: list[str] = field(default_factory=list)
    validation_strategy: str = ""

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    sources: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "intent": self.intent,
            "domain": self.domain,
            "task_type": self.task_type,
            "target": {
                "name": self.target_name,
                "description": self.target_description,
                "type": self.target_type.value,
                "classes": self.target_classes,
            },
            "features": [f.to_dict() for f in self.features],
            "domain_rules": self.domain_rules,
            "edge_cases": [
                {"name": e.name, "description": e.description, "frequency": e.frequency}
                for e in self.edge_cases
            ],
            "success_metrics": [
                {"name": m.name, "target": m.target_value, "minimum": m.minimum_acceptable}
                for m in self.success_metrics
            ],
            "data_generation": {
                "sample_size": self.sample_size_recommendation,
                "class_balance": self.class_balance,
            },
            "improvement_plan": {
                "data_to_collect": self.data_to_collect,
                "validation_strategy": self.validation_strategy,
            },
            "metadata": {
                "created_at": self.created_at,
                "sources": self.sources,
                "assumptions": self.assumptions,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_forge_request(self) -> dict:
        """Convert to Forge API request format."""
        schema = {}
        for f in self.features:
            if f.feature_type == FeatureType.NUMERIC:
                c = f.constraints or NumericConstraints()
                schema[f.name] = {
                    "type": "float" if c.distribution != DistributionType.POISSON else "int",
                    "constraints": {
                        "min": c.min_value,
                        "max": c.max_value,
                    },
                    "distribution": c.distribution.value if c.distribution else "normal",
                    "params": {
                        "mean": c.mean,
                        "std": c.std,
                    } if c.mean is not None else {},
                }
            elif f.feature_type == FeatureType.CATEGORICAL:
                c = f.constraints or CategoricalConstraints()
                schema[f.name] = {
                    "type": "category",
                    "constraints": {
                        "values": c.categories,
                    },
                    "probabilities": c.probabilities,
                }
            elif f.feature_type == FeatureType.BOOLEAN:
                schema[f.name] = {"type": "bool"}

        return {
            "data_type": "tabular",
            "domain": self.domain,
            "record_count": self.sample_size_recommendation,
            "schema": schema,
            "quality_level": "standard",
            "output_format": "jsonl",
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProblemSchema":
        """Load from dictionary."""
        features = [FeatureSpec.from_dict(f) for f in data.get("features", [])]

        edge_cases = [
            EdgeCase(
                name=e["name"],
                description=e.get("description", ""),
                frequency=e.get("frequency", "rare"),
                handling=e.get("handling", ""),
            )
            for e in data.get("edge_cases", [])
        ]

        success_metrics = [
            SuccessMetric(
                name=m["name"],
                target_value=m.get("target"),
                minimum_acceptable=m.get("minimum"),
            )
            for m in data.get("success_metrics", [])
        ]

        target = data.get("target", {})
        gen = data.get("data_generation", {})
        improve = data.get("improvement_plan", {})
        meta = data.get("metadata", {})

        return cls(
            name=data.get("name", ""),
            intent=data.get("intent", ""),
            domain=data.get("domain", ""),
            task_type=data.get("task_type", "classification"),
            target_name=target.get("name", "target"),
            target_description=target.get("description", ""),
            target_type=FeatureType(target.get("type", "categorical")),
            target_classes=target.get("classes", []),
            features=features,
            domain_rules=data.get("domain_rules", {}),
            edge_cases=edge_cases,
            success_metrics=success_metrics,
            sample_size_recommendation=gen.get("sample_size", 1000),
            class_balance=gen.get("class_balance", {}),
            data_to_collect=improve.get("data_to_collect", []),
            validation_strategy=improve.get("validation_strategy", ""),
            sources=meta.get("sources", []),
            assumptions=meta.get("assumptions", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ProblemSchema":
        return cls.from_dict(json.loads(json_str))

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"# {self.name}",
            "",
            f"**Intent:** {self.intent}",
            f"**Domain:** {self.domain}",
            f"**Task:** {self.task_type}",
            "",
            f"## Target: {self.target_name}",
            f"{self.target_description}",
        ]

        if self.target_classes:
            lines.append(f"Classes: {', '.join(self.target_classes)}")

        lines.extend(["", f"## Features ({len(self.features)})", ""])
        for f in self.features:
            lines.append(f"- **{f.name}** ({f.feature_type.value}): {f.description}")

        if self.edge_cases:
            lines.extend(["", "## Edge Cases", ""])
            for e in self.edge_cases:
                lines.append(f"- **{e.name}** [{e.frequency}]: {e.description}")

        if self.assumptions:
            lines.extend(["", "## Assumptions", ""])
            for a in self.assumptions:
                lines.append(f"- {a}")

        if self.data_to_collect:
            lines.extend(["", "## Data to Collect for Improvement", ""])
            for d in self.data_to_collect:
                lines.append(f"- {d}")

        return "\n".join(lines)


class SchemaGenerator:
    """
    Generate problem schemas from intent and research.

    Can work with:
    1. LLM to generate schema from intent
    2. Research results to extract features
    3. Manual specification
    """

    # Common domain templates
    DOMAIN_TEMPLATES = {
        "churn": {
            "task_type": "classification",
            "common_features": [
                FeatureSpec(
                    name="tenure_months",
                    feature_type=FeatureType.NUMERIC,
                    description="How long the customer has been active",
                    constraints=NumericConstraints(min_value=0, max_value=120, mean=24, std=18),
                    importance="high",
                ),
                FeatureSpec(
                    name="monthly_spend",
                    feature_type=FeatureType.NUMERIC,
                    description="Average monthly spending",
                    constraints=NumericConstraints(min_value=0, distribution=DistributionType.LOGNORMAL),
                    importance="high",
                ),
                FeatureSpec(
                    name="support_tickets",
                    feature_type=FeatureType.NUMERIC,
                    description="Number of support tickets filed",
                    constraints=NumericConstraints(min_value=0, distribution=DistributionType.POISSON, mean=2),
                    importance="high",
                ),
                FeatureSpec(
                    name="login_frequency",
                    feature_type=FeatureType.NUMERIC,
                    description="Logins per month",
                    constraints=NumericConstraints(min_value=0, distribution=DistributionType.POISSON, mean=10),
                    importance="medium",
                ),
                FeatureSpec(
                    name="contract_type",
                    feature_type=FeatureType.CATEGORICAL,
                    description="Type of contract",
                    constraints=CategoricalConstraints(
                        categories=["month-to-month", "annual", "multi-year"],
                        probabilities=[0.5, 0.35, 0.15],
                    ),
                    importance="high",
                ),
            ],
            "edge_cases": [
                EdgeCase("new_customer", "Customer with < 30 days tenure", "common", "May not have enough data"),
                EdgeCase("inactive_customer", "No activity in 60+ days", "occasional", "High churn signal"),
                EdgeCase("enterprise_customer", "Very high spend outlier", "rare", "Different churn dynamics"),
            ],
            "success_metrics": [
                SuccessMetric("recall", "Catch churning customers", target_value=0.8, minimum_acceptable=0.6),
                SuccessMetric("precision", "Avoid false alarms", target_value=0.7, minimum_acceptable=0.5),
            ],
        },

        "fraud": {
            "task_type": "classification",
            "common_features": [
                FeatureSpec(
                    name="transaction_amount",
                    feature_type=FeatureType.NUMERIC,
                    description="Amount of the transaction",
                    constraints=NumericConstraints(min_value=0, distribution=DistributionType.LOGNORMAL),
                    importance="high",
                ),
                FeatureSpec(
                    name="time_since_last_transaction",
                    feature_type=FeatureType.NUMERIC,
                    description="Hours since last transaction",
                    constraints=NumericConstraints(min_value=0, distribution=DistributionType.EXPONENTIAL),
                    importance="medium",
                ),
                FeatureSpec(
                    name="merchant_category",
                    feature_type=FeatureType.CATEGORICAL,
                    description="Type of merchant",
                    constraints=CategoricalConstraints(categories=["retail", "travel", "food", "entertainment", "other"]),
                    importance="medium",
                ),
                FeatureSpec(
                    name="is_international",
                    feature_type=FeatureType.BOOLEAN,
                    description="Whether transaction is international",
                    importance="high",
                ),
            ],
            "edge_cases": [
                EdgeCase("first_transaction", "New card/account", "common", "No history to compare"),
                EdgeCase("high_value_legitimate", "Large but legitimate purchase", "occasional", "May flag incorrectly"),
                EdgeCase("account_takeover", "Stolen credentials", "rare", "Behavior change detection"),
            ],
            "success_metrics": [
                SuccessMetric("recall", "Catch fraud", target_value=0.95, minimum_acceptable=0.9),
                SuccessMetric("precision", "Minimize false positives", target_value=0.5, minimum_acceptable=0.3),
            ],
        },

        "lead_scoring": {
            "task_type": "classification",
            "common_features": [
                FeatureSpec(
                    name="company_size",
                    feature_type=FeatureType.CATEGORICAL,
                    description="Size of prospect company",
                    constraints=CategoricalConstraints(categories=["startup", "smb", "mid-market", "enterprise"]),
                    importance="high",
                ),
                FeatureSpec(
                    name="website_visits",
                    feature_type=FeatureType.NUMERIC,
                    description="Number of website visits",
                    constraints=NumericConstraints(min_value=0, distribution=DistributionType.POISSON, mean=5),
                    importance="high",
                ),
                FeatureSpec(
                    name="email_opens",
                    feature_type=FeatureType.NUMERIC,
                    description="Marketing emails opened",
                    constraints=NumericConstraints(min_value=0, distribution=DistributionType.POISSON, mean=3),
                    importance="medium",
                ),
                FeatureSpec(
                    name="industry",
                    feature_type=FeatureType.CATEGORICAL,
                    description="Industry vertical",
                    constraints=CategoricalConstraints(categories=["tech", "finance", "healthcare", "retail", "other"]),
                    importance="medium",
                ),
            ],
            "edge_cases": [
                EdgeCase("no_engagement", "Lead with zero interactions", "common", "Insufficient data"),
                EdgeCase("competitor", "Competitor doing research", "occasional", "Filter out"),
            ],
            "success_metrics": [
                SuccessMetric("precision_at_k", "Quality of top leads", target_value=0.3, minimum_acceptable=0.2),
            ],
        },

        "mortgage": {
            "task_type": "classification",
            "common_features": [
                FeatureSpec(
                    name="loan_amount",
                    feature_type=FeatureType.NUMERIC,
                    description="Total loan amount in dollars",
                    constraints=NumericConstraints(min_value=50000, max_value=2000000, mean=350000, std=150000),
                    importance="high",
                ),
                FeatureSpec(
                    name="interest_rate",
                    feature_type=FeatureType.NUMERIC,
                    description="Annual interest rate percentage",
                    constraints=NumericConstraints(min_value=2.0, max_value=12.0, mean=5.5, std=1.5),
                    importance="high",
                ),
                FeatureSpec(
                    name="loan_to_value_ratio",
                    feature_type=FeatureType.NUMERIC,
                    description="Loan amount / property value ratio",
                    constraints=NumericConstraints(min_value=0.3, max_value=1.0, mean=0.8, std=0.15),
                    importance="high",
                ),
                FeatureSpec(
                    name="debt_to_income_ratio",
                    feature_type=FeatureType.NUMERIC,
                    description="Monthly debt payments / monthly income",
                    constraints=NumericConstraints(min_value=0.1, max_value=0.65, mean=0.35, std=0.12),
                    importance="high",
                ),
                FeatureSpec(
                    name="credit_score",
                    feature_type=FeatureType.NUMERIC,
                    description="Borrower credit score",
                    constraints=NumericConstraints(min_value=300, max_value=850, mean=680, std=80),
                    importance="high",
                ),
                FeatureSpec(
                    name="months_employed",
                    feature_type=FeatureType.NUMERIC,
                    description="Months at current employer",
                    constraints=NumericConstraints(min_value=0, max_value=480, mean=48, std=36),
                    importance="medium",
                ),
                FeatureSpec(
                    name="property_type",
                    feature_type=FeatureType.CATEGORICAL,
                    description="Type of property",
                    constraints=CategoricalConstraints(
                        categories=["single_family", "condo", "townhouse", "multi_family"],
                        probabilities=[0.6, 0.2, 0.15, 0.05],
                    ),
                    importance="medium",
                ),
                FeatureSpec(
                    name="loan_type",
                    feature_type=FeatureType.CATEGORICAL,
                    description="Type of mortgage loan",
                    constraints=CategoricalConstraints(
                        categories=["conventional", "fha", "va", "jumbo"],
                        probabilities=[0.6, 0.25, 0.1, 0.05],
                    ),
                    importance="medium",
                ),
            ],
            "edge_cases": [
                EdgeCase("new_employment", "Borrower recently changed jobs", "common", "Higher risk factor"),
                EdgeCase("investment_property", "Non-primary residence", "occasional", "Different risk profile"),
                EdgeCase("jumbo_loan", "Loan exceeds conforming limits", "rare", "Stricter requirements"),
            ],
            "success_metrics": [
                SuccessMetric("recall", "Catch defaults/foreclosures", target_value=0.85, minimum_acceptable=0.7),
                SuccessMetric("precision", "Avoid false rejections", target_value=0.6, minimum_acceptable=0.4),
            ],
        },

        "healthcare": {
            "task_type": "classification",
            "common_features": [
                FeatureSpec(
                    name="age",
                    feature_type=FeatureType.NUMERIC,
                    description="Patient age in years",
                    constraints=NumericConstraints(min_value=0, max_value=120, mean=45, std=20),
                    importance="high",
                ),
                FeatureSpec(
                    name="bmi",
                    feature_type=FeatureType.NUMERIC,
                    description="Body mass index",
                    constraints=NumericConstraints(min_value=12, max_value=60, mean=26, std=5),
                    importance="medium",
                ),
                FeatureSpec(
                    name="blood_pressure_systolic",
                    feature_type=FeatureType.NUMERIC,
                    description="Systolic blood pressure",
                    constraints=NumericConstraints(min_value=80, max_value=200, mean=120, std=15),
                    importance="high",
                ),
                FeatureSpec(
                    name="prior_conditions_count",
                    feature_type=FeatureType.NUMERIC,
                    description="Number of prior medical conditions",
                    constraints=NumericConstraints(min_value=0, max_value=10, distribution=DistributionType.POISSON, mean=2),
                    importance="high",
                ),
                FeatureSpec(
                    name="smoker",
                    feature_type=FeatureType.BOOLEAN,
                    description="Whether patient is a smoker",
                    importance="high",
                ),
                FeatureSpec(
                    name="gender",
                    feature_type=FeatureType.CATEGORICAL,
                    description="Patient gender",
                    constraints=CategoricalConstraints(categories=["male", "female", "other"]),
                    importance="medium",
                ),
            ],
            "edge_cases": [
                EdgeCase("pediatric", "Patients under 18", "common", "Different baseline values"),
                EdgeCase("elderly", "Patients over 80", "occasional", "Higher baseline risk"),
            ],
            "success_metrics": [
                SuccessMetric("recall", "Catch positive cases", target_value=0.9, minimum_acceptable=0.8),
                SuccessMetric("precision", "Minimize false positives", target_value=0.7, minimum_acceptable=0.5),
            ],
        },

        "ecommerce": {
            "task_type": "classification",
            "common_features": [
                FeatureSpec(
                    name="total_orders",
                    feature_type=FeatureType.NUMERIC,
                    description="Total number of past orders",
                    constraints=NumericConstraints(min_value=0, distribution=DistributionType.POISSON, mean=5),
                    importance="high",
                ),
                FeatureSpec(
                    name="avg_order_value",
                    feature_type=FeatureType.NUMERIC,
                    description="Average order value",
                    constraints=NumericConstraints(min_value=0, distribution=DistributionType.LOGNORMAL, mean=75, std=50),
                    importance="high",
                ),
                FeatureSpec(
                    name="days_since_last_order",
                    feature_type=FeatureType.NUMERIC,
                    description="Days since last purchase",
                    constraints=NumericConstraints(min_value=0, distribution=DistributionType.EXPONENTIAL, mean=30),
                    importance="high",
                ),
                FeatureSpec(
                    name="cart_abandonment_rate",
                    feature_type=FeatureType.NUMERIC,
                    description="Ratio of abandoned carts",
                    constraints=NumericConstraints(min_value=0, max_value=1, mean=0.7, std=0.2),
                    importance="medium",
                ),
                FeatureSpec(
                    name="product_category_preference",
                    feature_type=FeatureType.CATEGORICAL,
                    description="Most purchased category",
                    constraints=CategoricalConstraints(categories=["electronics", "clothing", "home", "food", "other"]),
                    importance="medium",
                ),
            ],
            "edge_cases": [
                EdgeCase("new_customer", "First-time buyer", "common", "No purchase history"),
                EdgeCase("bulk_buyer", "B2B or reseller", "rare", "Different buying patterns"),
            ],
            "success_metrics": [
                SuccessMetric("auc", "Overall model performance", target_value=0.8, minimum_acceptable=0.7),
            ],
        },
    }

    # Keyword mappings for generating features from unknown domains
    KEYWORD_FEATURES = {
        # Financial keywords
        "loan": [
            FeatureSpec("loan_amount", FeatureType.NUMERIC, "Loan principal amount",
                       constraints=NumericConstraints(min_value=0, distribution=DistributionType.LOGNORMAL), importance="high"),
            FeatureSpec("interest_rate", FeatureType.NUMERIC, "Interest rate percentage",
                       constraints=NumericConstraints(min_value=0, max_value=30, mean=8, std=4), importance="high"),
        ],
        "credit": [
            FeatureSpec("credit_score", FeatureType.NUMERIC, "Credit score",
                       constraints=NumericConstraints(min_value=300, max_value=850, mean=680, std=80), importance="high"),
            FeatureSpec("credit_utilization", FeatureType.NUMERIC, "Credit utilization ratio",
                       constraints=NumericConstraints(min_value=0, max_value=1, mean=0.3, std=0.2), importance="medium"),
        ],
        "income": [
            FeatureSpec("annual_income", FeatureType.NUMERIC, "Annual income in dollars",
                       constraints=NumericConstraints(min_value=0, distribution=DistributionType.LOGNORMAL, mean=60000, std=40000), importance="high"),
            FeatureSpec("income_stability", FeatureType.CATEGORICAL, "Employment stability",
                       constraints=CategoricalConstraints(categories=["stable", "variable", "seasonal"]), importance="medium"),
        ],
        "payment": [
            FeatureSpec("payment_history", FeatureType.NUMERIC, "On-time payment rate",
                       constraints=NumericConstraints(min_value=0, max_value=1, mean=0.9, std=0.15), importance="high"),
            FeatureSpec("missed_payments", FeatureType.NUMERIC, "Number of missed payments",
                       constraints=NumericConstraints(min_value=0, distribution=DistributionType.POISSON, mean=1), importance="high"),
        ],
        "property": [
            FeatureSpec("property_value", FeatureType.NUMERIC, "Property value in dollars",
                       constraints=NumericConstraints(min_value=0, distribution=DistributionType.LOGNORMAL), importance="high"),
            FeatureSpec("property_age", FeatureType.NUMERIC, "Property age in years",
                       constraints=NumericConstraints(min_value=0, max_value=150, mean=30, std=20), importance="medium"),
        ],
        "customer": [
            FeatureSpec("customer_tenure", FeatureType.NUMERIC, "Months as customer",
                       constraints=NumericConstraints(min_value=0, mean=24, std=18), importance="high"),
            FeatureSpec("customer_segment", FeatureType.CATEGORICAL, "Customer segment",
                       constraints=CategoricalConstraints(categories=["new", "regular", "premium", "at_risk"]), importance="medium"),
        ],
        "risk": [
            FeatureSpec("risk_score", FeatureType.NUMERIC, "Calculated risk score",
                       constraints=NumericConstraints(min_value=0, max_value=100, mean=35, std=20), importance="high"),
            FeatureSpec("risk_category", FeatureType.CATEGORICAL, "Risk classification",
                       constraints=CategoricalConstraints(categories=["low", "medium", "high"]), importance="high"),
        ],
        "time": [
            FeatureSpec("days_since_event", FeatureType.NUMERIC, "Days since last event",
                       constraints=NumericConstraints(min_value=0, distribution=DistributionType.EXPONENTIAL, mean=30), importance="medium"),
            FeatureSpec("account_age_months", FeatureType.NUMERIC, "Account age in months",
                       constraints=NumericConstraints(min_value=0, mean=24, std=18), importance="medium"),
        ],
        "amount": [
            FeatureSpec("transaction_amount", FeatureType.NUMERIC, "Transaction amount",
                       constraints=NumericConstraints(min_value=0, distribution=DistributionType.LOGNORMAL), importance="high"),
            FeatureSpec("total_amount", FeatureType.NUMERIC, "Total amount",
                       constraints=NumericConstraints(min_value=0, distribution=DistributionType.LOGNORMAL), importance="high"),
        ],
    }

    # Default features when nothing else matches
    DEFAULT_FEATURES = [
        FeatureSpec("numeric_feature_1", FeatureType.NUMERIC, "Primary numeric predictor",
                   constraints=NumericConstraints(mean=50, std=15), importance="high"),
        FeatureSpec("numeric_feature_2", FeatureType.NUMERIC, "Secondary numeric predictor",
                   constraints=NumericConstraints(mean=100, std=30), importance="medium"),
        FeatureSpec("numeric_feature_3", FeatureType.NUMERIC, "Tertiary numeric predictor",
                   constraints=NumericConstraints(min_value=0, max_value=1, mean=0.5, std=0.2), importance="medium"),
        FeatureSpec("category_feature", FeatureType.CATEGORICAL, "Categorical predictor",
                   constraints=CategoricalConstraints(categories=["type_a", "type_b", "type_c"]), importance="medium"),
        FeatureSpec("flag_feature", FeatureType.BOOLEAN, "Boolean flag", importance="low"),
    ]

    def from_intent(
        self,
        intent: str,
        domain: str = None,
        llm=None,
    ) -> ProblemSchema:
        """
        Generate schema from intent description.

        Args:
            intent: Natural language description of the goal
            domain: Optional domain hint (churn, fraud, lead_scoring, etc.)
            llm: Optional LLM for generating schema

        Returns:
            ProblemSchema
        """
        # Detect domain from intent if not provided
        if domain is None:
            domain = self._detect_domain(intent)

        # If we have a template for this domain, use it
        if domain in self.DOMAIN_TEMPLATES:
            return self._from_template(intent, domain)

        # Otherwise, create a basic schema
        return self._create_basic_schema(intent, domain)

    def _detect_domain(self, intent: str) -> str:
        """Detect domain from intent keywords."""
        intent_lower = intent.lower()

        if any(w in intent_lower for w in ["churn", "retain", "cancel", "subscription"]):
            return "churn"
        elif any(w in intent_lower for w in ["fraud", "suspicious", "anomal"]):
            return "fraud"
        elif any(w in intent_lower for w in ["lead", "prospect", "convert", "sales"]):
            return "lead_scoring"
        elif any(w in intent_lower for w in ["mortgage", "foreclosure", "home loan", "housing", "real estate"]):
            return "mortgage"
        elif any(w in intent_lower for w in ["patient", "diagnosis", "medical", "health", "disease", "treatment"]):
            return "healthcare"
        elif any(w in intent_lower for w in ["purchase", "buy", "shop", "cart", "order", "ecommerce", "retail"]):
            return "ecommerce"
        elif any(w in intent_lower for w in ["price", "cost", "value"]):
            return "pricing"
        else:
            return "general"

    def _from_template(self, intent: str, domain: str) -> ProblemSchema:
        """Create schema from domain template."""
        template = self.DOMAIN_TEMPLATES[domain]

        # Determine target based on domain
        if domain == "churn":
            target_name = "churned"
            target_desc = "Whether the customer churned (1) or retained (0)"
            target_classes = ["retained", "churned"]
            class_balance = {"retained": 0.8, "churned": 0.2}
        elif domain == "fraud":
            target_name = "is_fraud"
            target_desc = "Whether the transaction is fraudulent"
            target_classes = ["legitimate", "fraud"]
            class_balance = {"legitimate": 0.99, "fraud": 0.01}
        elif domain == "lead_scoring":
            target_name = "converted"
            target_desc = "Whether the lead converted to customer"
            target_classes = ["not_converted", "converted"]
            class_balance = {"not_converted": 0.9, "converted": 0.1}
        elif domain == "mortgage":
            target_name = "will_default"
            target_desc = "Whether the mortgage will default/foreclose"
            target_classes = ["no_default", "default"]
            class_balance = {"no_default": 0.92, "default": 0.08}
        elif domain == "healthcare":
            target_name = "condition_present"
            target_desc = "Whether the condition/outcome is present"
            target_classes = ["negative", "positive"]
            class_balance = {"negative": 0.7, "positive": 0.3}
        elif domain == "ecommerce":
            target_name = "will_purchase"
            target_desc = "Whether the customer will make a purchase"
            target_classes = ["no_purchase", "purchase"]
            class_balance = {"no_purchase": 0.85, "purchase": 0.15}
        else:
            target_name = "target"
            target_desc = "Target variable"
            target_classes = ["0", "1"]
            class_balance = {"0": 0.5, "1": 0.5}

        return ProblemSchema(
            name=f"{domain.replace('_', ' ').title()} Predictor",
            intent=intent,
            domain=domain,
            task_type=template["task_type"],
            target_name=target_name,
            target_description=target_desc,
            target_classes=target_classes,
            features=template["common_features"].copy(),
            edge_cases=template["edge_cases"].copy(),
            success_metrics=template["success_metrics"].copy(),
            class_balance=class_balance,
            assumptions=[
                f"Features are representative of {domain} problems",
                "Data distributions may differ from actual production data",
                "Edge cases should be validated with domain experts",
            ],
            data_to_collect=[
                "Actual outcome data from production",
                "Feature values from real customers/transactions",
                "Domain expert feedback on edge cases",
            ],
            validation_strategy=(
                "1. Start with synthetic data to validate pipeline\n"
                "2. Collect real data for validation set\n"
                "3. A/B test model predictions vs baseline\n"
                "4. Monitor for data drift in production"
            ),
        )

    def _create_basic_schema(self, intent: str, domain: str) -> ProblemSchema:
        """Create schema with auto-generated features when no template exists."""
        # Extract features based on keywords in the intent
        features = self._extract_features_from_intent(intent)

        # If we still have no features, use defaults
        if not features:
            features = [f for f in self.DEFAULT_FEATURES]  # Copy default features

        # Try to infer target from intent
        target_name, target_desc, target_classes = self._infer_target_from_intent(intent)

        return ProblemSchema(
            name=f"{domain.replace('_', ' ').title()} Predictor",
            intent=intent,
            domain=domain,
            task_type="classification",
            target_name=target_name,
            target_description=target_desc,
            target_classes=target_classes,
            features=features,
            class_balance={target_classes[0]: 0.7, target_classes[1]: 0.3},
            assumptions=[
                "Features were auto-generated based on intent keywords",
                "Feature distributions are estimates - validate with real data",
                "Consider adding domain-specific features for better accuracy",
            ],
            data_to_collect=[
                "Actual outcome data from production",
                "Real feature values for validation",
                "Domain expert feedback on feature relevance",
            ],
            validation_strategy=(
                "1. Train on synthetic data to validate pipeline\n"
                "2. Collect real samples for validation\n"
                "3. Compare model predictions to actual outcomes"
            ),
        )

    def _extract_features_from_intent(self, intent: str) -> list:
        """Extract relevant features based on keywords in the intent."""
        intent_lower = intent.lower()
        features = []
        used_feature_names = set()

        # Check each keyword category
        for keyword, keyword_features in self.KEYWORD_FEATURES.items():
            if keyword in intent_lower:
                for f in keyword_features:
                    if f.name not in used_feature_names:
                        # Create a copy of the feature
                        features.append(FeatureSpec(
                            name=f.name,
                            feature_type=f.feature_type,
                            description=f.description,
                            constraints=f.constraints,
                            importance=f.importance,
                        ))
                        used_feature_names.add(f.name)

        return features

    def _infer_target_from_intent(self, intent: str) -> tuple:
        """Infer target variable from intent description."""
        intent_lower = intent.lower()

        # Common prediction targets
        if any(w in intent_lower for w in ["foreclosure", "default", "fail"]):
            return "will_default", "Whether the subject will default/fail", ["no_default", "default"]
        elif any(w in intent_lower for w in ["success", "win", "pass"]):
            return "will_succeed", "Whether the subject will succeed", ["failure", "success"]
        elif any(w in intent_lower for w in ["buy", "purchase", "convert"]):
            return "will_convert", "Whether conversion will occur", ["no_conversion", "conversion"]
        elif any(w in intent_lower for w in ["leave", "quit", "exit"]):
            return "will_leave", "Whether the subject will leave", ["stay", "leave"]
        elif any(w in intent_lower for w in ["approve", "accept", "qualify"]):
            return "approved", "Whether approved/accepted", ["rejected", "approved"]
        elif "predict" in intent_lower:
            # Generic prediction - try to extract what's being predicted
            return "target", "Predicted outcome", ["negative", "positive"]
        else:
            return "target", "Target variable to predict", ["negative", "positive"]

    def add_feature(
        self,
        schema: ProblemSchema,
        name: str,
        feature_type: str,
        description: str,
        **kwargs,
    ) -> ProblemSchema:
        """Add a feature to an existing schema."""
        ft = FeatureType(feature_type)

        constraints = None
        if ft == FeatureType.NUMERIC:
            constraints = NumericConstraints(
                min_value=kwargs.get("min"),
                max_value=kwargs.get("max"),
                mean=kwargs.get("mean"),
                std=kwargs.get("std"),
                distribution=DistributionType(kwargs.get("distribution", "normal")),
            )
        elif ft == FeatureType.CATEGORICAL:
            constraints = CategoricalConstraints(
                categories=kwargs.get("categories", []),
                probabilities=kwargs.get("probabilities"),
            )

        schema.features.append(FeatureSpec(
            name=name,
            feature_type=ft,
            description=description,
            constraints=constraints,
            importance=kwargs.get("importance", "medium"),
        ))

        return schema

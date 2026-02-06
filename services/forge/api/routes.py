"""
Forge API Routes

FastAPI endpoints for synthetic data generation.
"""

from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, field_validator
import time
from collections import defaultdict

from ..schemas.schema import TabularSchema, SchemaValidationError
from ..generators.tabular import TabularGenerator
from ..qa import ForgeQA


# =============================================================================
# Request/Response Models
# =============================================================================

class FieldDefinition(BaseModel):
    """Definition for a single field."""
    type: str = Field(..., description="Field type: string, int, float, bool, category, date")
    nullable: bool = Field(False, description="Whether nulls are allowed")
    null_rate: float = Field(0.0, ge=0.0, le=1.0, description="Rate of null values (0-1)")

    # Numeric constraints
    min: Optional[float] = Field(None, description="Minimum value (numeric)")
    max: Optional[float] = Field(None, description="Maximum value (numeric)")

    # Distribution (for numeric types)
    distribution: str = Field("uniform", description="Distribution: uniform, normal, beta, exponential")
    dist_alpha: Optional[float] = Field(None, description="Alpha param for beta distribution")
    dist_beta: Optional[float] = Field(None, description="Beta param for beta distribution")

    # String constraints
    min_length: Optional[int] = Field(None, ge=0, description="Minimum string length")
    max_length: Optional[int] = Field(None, ge=1, description="Maximum string length")
    pattern: Optional[str] = Field(None, description="Regex pattern for strings")

    # Category constraints
    values: Optional[list[str]] = Field(None, description="Allowed values for category type")
    weights: Optional[list[float]] = Field(None, description="Probability weights for categories")

    # Date constraints
    min_date: Optional[str] = Field(None, description="Minimum date (ISO format)")
    max_date: Optional[str] = Field(None, description="Maximum date (ISO format)")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v):
        valid = ["string", "int", "float", "bool", "category", "date"]
        if v not in valid:
            raise ValueError(f"type must be one of {valid}")
        return v

    @field_validator("distribution")
    @classmethod
    def validate_distribution(cls, v):
        valid = ["uniform", "normal", "beta", "exponential"]
        if v not in valid:
            raise ValueError(f"distribution must be one of {valid}")
        return v


class GenerateRequest(BaseModel):
    """Request body for data generation."""
    fields: dict[str, FieldDefinition] = Field(
        ...,
        description="Field definitions keyed by field name",
    )
    n: int = Field(
        ...,
        gt=0,
        le=100_000,
        description="Number of rows to generate (max 100,000)",
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility",
    )
    format: str = Field(
        "json",
        description="Output format: json, csv",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "fields": {
                "product_name": {"type": "string", "min_length": 5, "max_length": 30},
                "price": {"type": "float", "min": 0.01, "max": 10000, "distribution": "normal"},
                "category": {"type": "category", "values": ["electronics", "clothing", "home"]},
                "in_stock": {"type": "bool"},
                "created_date": {"type": "date", "min_date": "2024-01-01"},
            },
            "n": 1000,
            "seed": 42,
            "format": "json",
        }
    }}


class FieldStatsResponse(BaseModel):
    """Statistics for a single field."""
    name: str
    dtype: str
    count: int
    null_count: int
    null_percent: float
    unique_count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    top_values: Optional[dict[str, int]] = None


class QualityResponse(BaseModel):
    """Quality metrics."""
    score: float
    grade: str
    passed: bool
    schema_compliance: float
    issues: list[dict] = []


class GenerateResponse(BaseModel):
    """Response from data generation."""
    success: bool
    row_count: int
    column_count: int
    data: Optional[list[dict]] = None  # For JSON format
    csv: Optional[str] = None  # For CSV format
    stats: list[FieldStatsResponse]
    quality: QualityResponse


# =============================================================================
# API Setup
# =============================================================================

app = FastAPI(
    title="Forge API",
    description="Synthetic data generation service",
    version="0.1.0",
)


# =============================================================================
# Rate Limiting (Simple in-memory for alpha)
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, requests_per_minute: int = 20):
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > minute_ago
        ]

        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False

        self.requests[client_ip].append(now)
        return True


rate_limiter = RateLimiter(requests_per_minute=20)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limit incoming requests."""
    # Skip health checks
    if request.url.path == "/health":
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"

    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 20 requests per minute.",
        )

    return await call_next(request)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "forge", "version": "0.1.0"}


@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate synthetic data from schema.

    Upload a JSON schema defining fields and constraints,
    receive generated data with QA stats.
    """
    # Convert request to TabularSchema
    schema_dict = {
        "fields": {
            name: {
                "type": field.type,
                "nullable": field.nullable,
                "null_rate": field.null_rate,
                **({"min": field.min} if field.min is not None else {}),
                **({"max": field.max} if field.max is not None else {}),
                **({"distribution": field.distribution} if field.distribution != "uniform" else {}),
                **({"dist_alpha": field.dist_alpha} if field.dist_alpha is not None else {}),
                **({"dist_beta": field.dist_beta} if field.dist_beta is not None else {}),
                **({"min_length": field.min_length} if field.min_length is not None else {}),
                **({"max_length": field.max_length} if field.max_length is not None else {}),
                **({"pattern": field.pattern} if field.pattern else {}),
                **({"values": field.values} if field.values else {}),
                **({"weights": field.weights} if field.weights else {}),
                **({"min_date": field.min_date} if field.min_date else {}),
                **({"max_date": field.max_date} if field.max_date else {}),
            }
            for name, field in request.fields.items()
        }
    }

    try:
        schema = TabularSchema.from_dict(schema_dict)
    except SchemaValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate schema
    errors = schema.validate()
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    # Generate data
    generator = TabularGenerator(seed=request.seed)
    try:
        df = generator.generate(schema, n=request.n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Run QA
    qa = ForgeQA()
    report = qa.analyze(df, schema)

    # Format stats
    stats = [
        FieldStatsResponse(
            name=f.name,
            dtype=f.dtype,
            count=f.count,
            null_count=f.null_count,
            null_percent=f.null_percent,
            unique_count=f.unique_count,
            mean=f.mean,
            std=f.std,
            min=f.min_val,
            max=f.max_val,
            median=f.median,
            top_values=f.top_values,
        )
        for f in report.field_stats
    ]

    # Format quality
    quality = QualityResponse(
        score=report.score,
        grade=report.grade.value,
        passed=report.passed,
        schema_compliance=report.schema_compliance,
        issues=[
            {"severity": i.severity.value, "category": i.category, "message": i.message}
            for i in report.issues
        ],
    )

    # Format output
    if request.format == "csv":
        csv_output = df.to_csv(index=False)
        return GenerateResponse(
            success=True,
            row_count=len(df),
            column_count=len(df.columns),
            csv=csv_output,
            stats=stats,
            quality=quality,
        )
    else:
        # JSON format
        data = df.to_dict(orient="records")
        return GenerateResponse(
            success=True,
            row_count=len(df),
            column_count=len(df.columns),
            data=data,
            stats=stats,
            quality=quality,
        )


@app.post("/api/v1/validate")
async def validate_schema(request: GenerateRequest):
    """
    Validate a schema without generating data.

    Returns validation errors if any.
    """
    schema_dict = {
        "fields": {
            name: {
                "type": field.type,
                "nullable": field.nullable,
                **({"min": field.min} if field.min is not None else {}),
                **({"max": field.max} if field.max is not None else {}),
                **({"values": field.values} if field.values else {}),
            }
            for name, field in request.fields.items()
        }
    }

    try:
        schema = TabularSchema.from_dict(schema_dict)
    except SchemaValidationError as e:
        return {"valid": False, "errors": [str(e)]}

    errors = schema.validate()
    if errors:
        return {"valid": False, "errors": errors}

    return {
        "valid": True,
        "field_count": len(schema.fields),
        "fields": [f.name for f in schema.fields],
    }

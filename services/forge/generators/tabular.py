"""
Tabular Data Generator

Generates synthetic tabular data based on schema definitions.
"""

import random
import string
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from ..schemas.schema import TabularSchema, FieldSpec, FieldType


class Distribution(Enum):
    """Supported distributions for numeric fields."""
    UNIFORM = "uniform"      # Rectangular - equal probability
    NORMAL = "normal"        # Gaussian - bell curve
    BETA = "beta"            # Beta distribution - flexible shape
    EXPONENTIAL = "exponential"  # Exponential - decay


class TabularGenerator:
    """
    Generate synthetic tabular data from a schema.

    Usage:
        schema = TabularSchema.from_dict({...})
        generator = TabularGenerator()
        df = generator.generate(schema, n=1000)
    """

    def __init__(self, seed: int = None):
        """Initialize generator with optional random seed for reproducibility."""
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self, schema: TabularSchema, n: int) -> "pd.DataFrame":
        """
        Generate n rows of data following the schema.

        Args:
            schema: TabularSchema defining the data structure
            n: Number of rows to generate

        Returns:
            pandas DataFrame with generated data
        """
        import pandas as pd

        # Validate schema first
        errors = schema.validate()
        if errors:
            raise ValueError(f"Invalid schema: {errors}")

        if n <= 0:
            raise ValueError("n must be positive")

        if n > 100_000:
            raise ValueError("n exceeds maximum of 100,000 rows")

        data = {}
        for field_spec in schema.fields:
            data[field_spec.name] = self._generate_column(field_spec, n)

        return pd.DataFrame(data)

    def _generate_column(self, spec: FieldSpec, n: int) -> list[Any]:
        """Generate a single column of data."""
        generators = {
            FieldType.STRING: self._gen_string,
            FieldType.INT: self._gen_int,
            FieldType.FLOAT: self._gen_float,
            FieldType.BOOL: self._gen_bool,
            FieldType.CATEGORY: self._gen_category,
            FieldType.DATE: self._gen_date,
        }

        generator = generators.get(spec.field_type)
        if not generator:
            raise ValueError(f"Unknown field type: {spec.field_type}")

        values = generator(spec, n)

        # Apply nulls if nullable
        if spec.nullable and spec.null_rate > 0:
            null_mask = np.random.random(n) < spec.null_rate
            values = [None if null_mask[i] else v for i, v in enumerate(values)]

        return values

    def _gen_string(self, spec: FieldSpec, n: int) -> list[str]:
        """Generate string values."""
        min_len = spec.min_length or 5
        max_len = spec.max_length or 20

        values = []
        for _ in range(n):
            length = random.randint(min_len, max_len)
            # Generate readable-ish strings
            s = ''.join(random.choices(string.ascii_lowercase, k=length))
            values.append(s)

        return values

    def _gen_int(self, spec: FieldSpec, n: int) -> list[int]:
        """Generate integer values."""
        min_val = int(spec.min_value) if spec.min_value is not None else 0
        max_val = int(spec.max_value) if spec.max_value is not None else 1000

        dist = getattr(spec, 'distribution', None) or 'uniform'
        values = self._sample_distribution(dist, min_val, max_val, n, spec)
        return [int(round(v)) for v in values]

    def _gen_float(self, spec: FieldSpec, n: int) -> list[float]:
        """Generate float values."""
        min_val = spec.min_value if spec.min_value is not None else 0.0
        max_val = spec.max_value if spec.max_value is not None else 1000.0

        dist = getattr(spec, 'distribution', None) or 'uniform'
        values = self._sample_distribution(dist, min_val, max_val, n, spec)
        return [round(float(v), 2) for v in values]

    def _sample_distribution(
        self,
        dist: str,
        min_val: float,
        max_val: float,
        n: int,
        spec: FieldSpec,
    ) -> np.ndarray:
        """Sample from specified distribution, scaled to [min_val, max_val]."""
        range_val = max_val - min_val

        if dist == "normal":
            # Normal distribution centered in range, with std = range/6 (99.7% within range)
            mean = (min_val + max_val) / 2
            std = range_val / 6
            values = np.random.normal(mean, std, size=n)
            # Clip to range
            values = np.clip(values, min_val, max_val)

        elif dist == "beta":
            # Beta distribution - default alpha=2, beta=5 gives right-skewed
            alpha = getattr(spec, 'dist_alpha', None) or 2.0
            beta = getattr(spec, 'dist_beta', None) or 5.0
            values = np.random.beta(alpha, beta, size=n)
            # Scale to range
            values = min_val + values * range_val

        elif dist == "exponential":
            # Exponential - scale parameter
            scale = range_val / 3
            values = np.random.exponential(scale, size=n)
            # Shift and clip
            values = min_val + values
            values = np.clip(values, min_val, max_val)

        else:  # uniform (default)
            values = np.random.uniform(min_val, max_val, size=n)

        return values

    def _gen_bool(self, spec: FieldSpec, n: int) -> list[bool]:
        """Generate boolean values."""
        return list(np.random.choice([True, False], size=n))

    def _gen_category(self, spec: FieldSpec, n: int) -> list[str]:
        """Generate categorical values."""
        if not spec.values:
            raise ValueError(f"Category field '{spec.name}' requires values list")

        if spec.weights:
            # Normalize weights
            weights = np.array(spec.weights)
            weights = weights / weights.sum()
            return list(np.random.choice(spec.values, size=n, p=weights))
        else:
            return list(np.random.choice(spec.values, size=n))

    def _gen_date(self, spec: FieldSpec, n: int) -> list[str]:
        """Generate date values in ISO format."""
        # Default date range: past year
        if spec.min_date:
            min_date = datetime.fromisoformat(spec.min_date)
        else:
            min_date = datetime.now() - timedelta(days=365)

        if spec.max_date:
            max_date = datetime.fromisoformat(spec.max_date)
        else:
            max_date = datetime.now()

        delta = (max_date - min_date).days
        if delta < 0:
            raise ValueError(f"min_date > max_date for field '{spec.name}'")

        values = []
        for _ in range(n):
            random_days = random.randint(0, max(delta, 1))
            date = min_date + timedelta(days=random_days)
            values.append(date.strftime("%Y-%m-%d"))

        return values

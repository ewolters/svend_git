"""
Forge Client

Interface to Forge synthetic data generation service.
Forge can be:
1. Local (in-process generation for development)
2. Remote API (production Forge service)

Accepts:
- ProblemSchema objects
- Raw JSON schema
- Sample JSON (infers schema)
"""

import json
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path

from .schema import ProblemSchema, FeatureSpec, FeatureType, NumericConstraints, CategoricalConstraints, DistributionType
from .interfaces import ForgeRequest, ForgeOutput


@dataclass
class ForgeConfig:
    """Configuration for Forge client."""
    mode: str = "local"  # local, api
    api_url: str = "http://localhost:8000/api/v1"
    api_key: str = ""
    default_quality: str = "standard"
    timeout_seconds: int = 300


class ForgeClient:
    """
    Client for Forge synthetic data generation.

    Usage:
        # From schema
        client = ForgeClient()
        df = client.generate_from_schema(problem_schema, n=1000)

        # From JSON
        sample = {"age": 25, "income": 50000, "status": ["active", "inactive"]}
        df = client.generate_from_json(json.dumps(sample), n=1000)

        # Direct schema dict
        schema = {
            "age": {"type": "int", "constraints": {"min": 18, "max": 100}},
            "income": {"type": "float", "constraints": {"min": 0}},
        }
        df = client.generate(schema, n=1000)
    """

    def __init__(self, config: ForgeConfig = None):
        self.config = config or ForgeConfig()

    def generate(
        self,
        schema: dict,
        n: int = 1000,
        include_target: bool = True,
        target_name: str = "target",
        target_classes: list = None,
        class_balance: dict = None,
    ):
        """
        Generate data from schema dict.

        Schema format:
        {
            "column_name": {
                "type": "int" | "float" | "category" | "bool" | "string",
                "constraints": {
                    "min": number,
                    "max": number,
                    "values": [...]  # for category
                },
                "distribution": "normal" | "uniform" | "poisson" | ...,
                "params": {"mean": ..., "std": ...}
            }
        }
        """
        if self.config.mode == "local":
            return self._generate_local(
                schema, n, include_target, target_name, target_classes, class_balance
            )
        else:
            return self._generate_api(schema, n)

    def generate_from_schema(
        self,
        problem_schema: ProblemSchema,
        n: int = None,
    ):
        """Generate data from a ProblemSchema object."""
        n = n or problem_schema.sample_size_recommendation

        # Convert to schema dict
        schema_dict = {}
        for feature in problem_schema.features:
            schema_dict[feature.name] = self._feature_to_schema(feature)

        return self.generate(
            schema=schema_dict,
            n=n,
            include_target=True,
            target_name=problem_schema.target_name,
            target_classes=problem_schema.target_classes,
            class_balance=problem_schema.class_balance,
        )

    def generate_from_json(
        self,
        sample_json: str,
        n: int = 1000,
    ):
        """
        Generate data by inferring schema from sample JSON.

        The sample should have values that hint at the data type:
        - Integers: will be generated as int
        - Floats: will be generated as float
        - Strings: will be generated as category
        - Lists of strings: will be treated as category with those values
        - Booleans: will be generated as bool
        """
        sample = json.loads(sample_json)
        schema = self._infer_schema_from_sample(sample)
        return self.generate(schema, n)

    def run(self, request: ForgeRequest) -> ForgeOutput:
        """Run with interface-compatible request."""
        if request.problem_schema:
            data = self.generate_from_schema(
                request.problem_schema,
                n=request.record_count,
            )
        elif request.schema:
            data = self.generate(request.schema, n=request.record_count)
        elif request.sample_json:
            data = self.generate_from_json(request.sample_json, n=request.record_count)
        else:
            raise ValueError("Must provide schema, problem_schema, or sample_json")

        return ForgeOutput(
            data=data,
            record_count=len(data),
            schema_used=request.schema or {},
        )

    def validate(self, request: ForgeRequest) -> list[str]:
        """Validate request."""
        errors = []
        if not (request.schema or request.problem_schema or request.sample_json):
            errors.append("Must provide schema, problem_schema, or sample_json")
        return errors

    def _feature_to_schema(self, feature: FeatureSpec) -> dict:
        """Convert FeatureSpec to schema dict entry."""
        result = {"type": feature.feature_type.value}

        if feature.feature_type == FeatureType.NUMERIC:
            c = feature.constraints or NumericConstraints()
            result["type"] = "float"
            result["constraints"] = {}
            if c.min_value is not None:
                result["constraints"]["min"] = c.min_value
            if c.max_value is not None:
                result["constraints"]["max"] = c.max_value
            if c.distribution:
                result["distribution"] = c.distribution.value
            if c.mean is not None or c.std is not None:
                result["params"] = {}
                if c.mean is not None:
                    result["params"]["mean"] = c.mean
                if c.std is not None:
                    result["params"]["std"] = c.std

        elif feature.feature_type == FeatureType.CATEGORICAL:
            c = feature.constraints or CategoricalConstraints()
            result["type"] = "category"
            result["constraints"] = {"values": c.categories}
            if c.probabilities:
                result["probabilities"] = c.probabilities

        elif feature.feature_type == FeatureType.BOOLEAN:
            result["type"] = "bool"

        return result

    def _infer_schema_from_sample(self, sample: dict) -> dict:
        """Infer schema from sample data."""
        schema = {}
        for key, value in sample.items():
            if isinstance(value, bool):
                schema[key] = {"type": "bool"}
            elif isinstance(value, int):
                schema[key] = {
                    "type": "int",
                    "constraints": {"min": 0, "max": value * 10},
                    "distribution": "uniform",
                }
            elif isinstance(value, float):
                schema[key] = {
                    "type": "float",
                    "constraints": {"min": 0, "max": value * 10},
                    "distribution": "normal",
                    "params": {"mean": value, "std": value * 0.3},
                }
            elif isinstance(value, list):
                # List of values = categorical
                schema[key] = {
                    "type": "category",
                    "constraints": {"values": value},
                }
            elif isinstance(value, str):
                # Single string = we don't know the categories
                schema[key] = {
                    "type": "category",
                    "constraints": {"values": [value, f"{value}_2", f"{value}_3"]},
                }
        return schema

    def _generate_local(
        self,
        schema: dict,
        n: int,
        include_target: bool,
        target_name: str,
        target_classes: list,
        class_balance: dict,
    ):
        """Generate data locally (no API call)."""
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        data = {}

        for col_name, col_spec in schema.items():
            col_type = col_spec.get("type", "float")
            constraints = col_spec.get("constraints", {})
            distribution = col_spec.get("distribution", "uniform")
            params = col_spec.get("params", {})

            if col_type in ["int", "float"]:
                data[col_name] = self._generate_numeric(
                    n, col_type, constraints, distribution, params
                )

            elif col_type == "category":
                values = constraints.get("values", ["A", "B", "C"])
                probs = col_spec.get("probabilities")
                data[col_name] = np.random.choice(values, size=n, p=probs)

            elif col_type == "bool":
                data[col_name] = np.random.choice([True, False], size=n)

            elif col_type == "string":
                # Generate random strings (placeholder)
                data[col_name] = [f"text_{i}" for i in range(n)]

        # Add target if requested
        if include_target and target_name:
            if target_classes:
                if class_balance:
                    probs = [class_balance.get(c, 1/len(target_classes)) for c in target_classes]
                    # Normalize
                    total = sum(probs)
                    probs = [p/total for p in probs]
                else:
                    probs = None
                data[target_name] = np.random.choice(target_classes, size=n, p=probs)
            else:
                # Default binary
                data[target_name] = np.random.choice([0, 1], size=n)

        return pd.DataFrame(data)

    def _generate_numeric(
        self,
        n: int,
        col_type: str,
        constraints: dict,
        distribution: str,
        params: dict,
    ):
        """Generate numeric column."""
        import numpy as np

        min_val = constraints.get("min", 0)
        max_val = constraints.get("max", 100)
        mean = params.get("mean", (min_val + max_val) / 2 if min_val and max_val else 50)
        std = params.get("std", (max_val - min_val) / 6 if min_val and max_val else 10)

        if distribution == "normal":
            values = np.random.normal(mean, std, n)
        elif distribution == "uniform":
            values = np.random.uniform(min_val or 0, max_val or 100, n)
        elif distribution == "exponential":
            scale = params.get("scale", mean if mean else 10)
            values = np.random.exponential(scale, n)
        elif distribution == "poisson":
            lam = params.get("mean", mean if mean else 5)
            values = np.random.poisson(lam, n)
        elif distribution == "lognormal":
            values = np.random.lognormal(np.log(mean) if mean > 0 else 0, std / mean if mean > 0 else 1, n)
        elif distribution == "beta":
            a = params.get("a", 2)
            b = params.get("b", 5)
            values = np.random.beta(a, b, n) * (max_val - min_val) + min_val
        else:
            values = np.random.uniform(min_val or 0, max_val or 100, n)

        # Apply constraints
        if min_val is not None:
            values = np.maximum(values, min_val)
        if max_val is not None:
            values = np.minimum(values, max_val)

        if col_type == "int":
            values = values.astype(int)

        return values

    def _generate_api(self, schema: dict, n: int):
        """Generate via Forge API."""
        import urllib.request
        import urllib.error

        url = f"{self.config.api_url}/generate"
        payload = {
            "data_type": "tabular",
            "schema": schema,
            "record_count": n,
            "quality_level": self.config.default_quality,
            "output_format": "json",
        }

        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                result = json.loads(resp.read().decode())

            # If async, poll for result
            if result.get("status") == "queued":
                job_id = result.get("job_id")
                return self._poll_job(job_id)

            # Direct result
            import pandas as pd
            return pd.DataFrame(result.get("data", []))

        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Forge API error: {e.code} {e.reason}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Forge API connection error: {e.reason}")

    def _poll_job(self, job_id: str, max_polls: int = 60, interval: float = 5.0):
        """Poll for async job completion."""
        import time
        import urllib.request

        for _ in range(max_polls):
            url = f"{self.config.api_url}/jobs/{job_id}"
            req = urllib.request.Request(url)
            if self.config.api_key:
                req.add_header("X-API-Key", self.config.api_key)

            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode())

            if result.get("status") == "completed":
                # Fetch result
                result_url = f"{self.config.api_url}/jobs/{job_id}/result"
                req = urllib.request.Request(result_url)
                if self.config.api_key:
                    req.add_header("X-API-Key", self.config.api_key)

                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode())

                import pandas as pd
                return pd.DataFrame(data)

            elif result.get("status") == "failed":
                raise RuntimeError(f"Forge job failed: {result.get('error')}")

            time.sleep(interval)

        raise TimeoutError("Forge job timed out")

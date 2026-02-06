"""Tests for Forge synthetic data generation."""

import pytest
import json


class TestTabularSchema:
    """Tests for schema validation."""

    def test_schema_from_dict(self):
        from forge.schemas import TabularSchema

        schema = TabularSchema.from_dict({
            "fields": {
                "name": {"type": "string"},
                "age": {"type": "int", "min": 0, "max": 120},
                "category": {"type": "category", "values": ["a", "b", "c"]},
            }
        })

        assert len(schema.fields) == 3
        assert schema.fields[0].name == "name"
        assert schema.fields[1].min_value == 0
        assert schema.fields[2].values == ["a", "b", "c"]

    def test_schema_validation_errors(self):
        from forge.schemas import TabularSchema

        # Category without values
        schema = TabularSchema.from_dict({
            "fields": {
                "bad_cat": {"type": "category"},  # Missing values
            }
        })
        errors = schema.validate()
        assert len(errors) > 0
        assert "values" in errors[0].lower()

    def test_schema_duplicate_names(self):
        from forge.schemas import TabularSchema, FieldSpec, FieldType

        schema = TabularSchema(fields=[
            FieldSpec(name="foo", field_type=FieldType.STRING),
            FieldSpec(name="foo", field_type=FieldType.INT),
        ])
        errors = schema.validate()
        assert any("duplicate" in e.lower() for e in errors)

    def test_invalid_field_type(self):
        from forge.schemas import TabularSchema, SchemaValidationError

        with pytest.raises(SchemaValidationError):
            TabularSchema.from_dict({
                "fields": {"x": {"type": "invalid_type"}}
            })

    def test_distribution_validation(self):
        from forge.schemas import TabularSchema

        schema = TabularSchema.from_dict({
            "fields": {
                "price": {"type": "float", "distribution": "invalid"}
            }
        })
        errors = schema.validate()
        assert any("distribution" in e.lower() for e in errors)


class TestTabularGenerator:
    """Tests for data generation."""

    def test_basic_generation(self):
        from forge import TabularGenerator, TabularSchema

        schema = TabularSchema.from_dict({
            "fields": {
                "id": {"type": "int", "min": 1, "max": 1000},
                "name": {"type": "string"},
                "active": {"type": "bool"},
            }
        })

        gen = TabularGenerator(seed=42)
        df = gen.generate(schema, n=100)

        assert len(df) == 100
        assert list(df.columns) == ["id", "name", "active"]
        assert df["id"].min() >= 1
        assert df["id"].max() <= 1000

    def test_category_generation(self):
        from forge import TabularGenerator, TabularSchema

        schema = TabularSchema.from_dict({
            "fields": {
                "color": {"type": "category", "values": ["red", "green", "blue"]}
            }
        })

        gen = TabularGenerator(seed=42)
        df = gen.generate(schema, n=1000)

        assert set(df["color"].unique()).issubset({"red", "green", "blue"})

    def test_nullable_generation(self):
        from forge import TabularGenerator, TabularSchema

        schema = TabularSchema.from_dict({
            "fields": {
                "value": {"type": "int", "nullable": True, "null_rate": 0.3}
            }
        })

        gen = TabularGenerator(seed=42)
        df = gen.generate(schema, n=1000)

        null_rate = df["value"].isna().sum() / len(df)
        # Should be roughly 30% nulls (allow some variance)
        assert 0.2 < null_rate < 0.4

    def test_distribution_normal(self):
        from forge import TabularGenerator, TabularSchema

        schema = TabularSchema.from_dict({
            "fields": {
                "value": {"type": "float", "min": 0, "max": 100, "distribution": "normal"}
            }
        })

        gen = TabularGenerator(seed=42)
        df = gen.generate(schema, n=10000)

        # Normal distribution should cluster around mean (50)
        mean = df["value"].mean()
        assert 45 < mean < 55

    def test_distribution_beta(self):
        from forge import TabularGenerator, TabularSchema

        schema = TabularSchema.from_dict({
            "fields": {
                "value": {
                    "type": "float",
                    "min": 0,
                    "max": 1,
                    "distribution": "beta",
                    "dist_alpha": 2,
                    "dist_beta": 5,
                }
            }
        })

        gen = TabularGenerator(seed=42)
        df = gen.generate(schema, n=10000)

        # Beta(2,5) is right-skewed, mean should be < 0.5
        mean = df["value"].mean()
        assert mean < 0.4

    def test_seed_reproducibility(self):
        from forge import TabularGenerator, TabularSchema

        schema = TabularSchema.from_dict({
            "fields": {"x": {"type": "float"}}
        })

        df1 = TabularGenerator(seed=123).generate(schema, n=100)
        df2 = TabularGenerator(seed=123).generate(schema, n=100)

        assert df1["x"].tolist() == df2["x"].tolist()

    def test_max_rows_limit(self):
        from forge import TabularGenerator, TabularSchema

        schema = TabularSchema.from_dict({
            "fields": {"x": {"type": "int"}}
        })

        gen = TabularGenerator()
        with pytest.raises(ValueError, match="exceeds maximum"):
            gen.generate(schema, n=200_000)


class TestForgeQA:
    """Tests for QA analysis."""

    def test_basic_qa(self):
        from forge import TabularGenerator, TabularSchema, ForgeQA

        schema = TabularSchema.from_dict({
            "fields": {
                "id": {"type": "int", "min": 1, "max": 100},
                "name": {"type": "string"},
            }
        })

        gen = TabularGenerator(seed=42)
        df = gen.generate(schema, n=100)

        qa = ForgeQA()
        report = qa.analyze(df, schema)

        assert report.row_count == 100
        assert report.column_count == 2
        assert len(report.field_stats) == 2
        assert report.passed

    def test_qa_field_stats(self):
        from forge import TabularGenerator, TabularSchema, ForgeQA

        schema = TabularSchema.from_dict({
            "fields": {
                "value": {"type": "float", "min": 0, "max": 100}
            }
        })

        gen = TabularGenerator(seed=42)
        df = gen.generate(schema, n=1000)

        qa = ForgeQA()
        report = qa.analyze(df)

        stats = report.field_stats[0]
        assert stats.name == "value"
        assert stats.count == 1000
        assert stats.mean is not None
        assert stats.std is not None
        assert stats.min_val >= 0
        assert stats.max_val <= 100

    def test_qa_schema_compliance(self):
        from forge import TabularGenerator, TabularSchema, ForgeQA

        schema = TabularSchema.from_dict({
            "fields": {
                "value": {"type": "int", "min": 10, "max": 20}
            }
        })

        gen = TabularGenerator(seed=42)
        df = gen.generate(schema, n=100)

        qa = ForgeQA()
        report = qa.analyze(df, schema)

        assert report.schema_compliance == 100.0
        assert len(report.constraint_violations) == 0

    def test_qa_detects_violations(self):
        import pandas as pd
        from forge import TabularSchema, ForgeQA

        # Create data that violates constraints
        df = pd.DataFrame({"value": [5, 15, 25]})  # 5 < min(10), 25 > max(20)

        schema = TabularSchema.from_dict({
            "fields": {
                "value": {"type": "int", "min": 10, "max": 20}
            }
        })

        qa = ForgeQA()
        report = qa.analyze(df, schema)

        assert report.schema_compliance < 100
        assert len(report.constraint_violations) > 0

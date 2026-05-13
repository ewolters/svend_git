"""Tests for DataSourcePlugin — flowchart entry point device."""

import pytest
from pydantic import ValidationError

from plugins.data_source import DataSourceInput, DataSourcePlugin


class TestDataSourceInput:
    def test_valid_input(self):
        inp = DataSourceInput(measurements=[1.0, 2.0, 3.0], usl=10.0, lsl=0.0)
        assert inp.measurements == [1.0, 2.0, 3.0]
        assert inp.usl == 10.0
        assert inp.lsl == 0.0

    def test_measurements_required(self):
        with pytest.raises(ValidationError):
            DataSourceInput(measurements=[])

    def test_single_measurement_ok(self):
        inp = DataSourceInput(measurements=[42.0])
        assert len(inp.measurements) == 1

    def test_usl_gt_lsl(self):
        with pytest.raises(ValidationError, match="USL must be greater than LSL"):
            DataSourceInput(measurements=[1.0], usl=5.0, lsl=10.0)

    def test_specs_optional(self):
        inp = DataSourceInput(measurements=[1.0, 2.0])
        assert inp.usl is None
        assert inp.lsl is None

    def test_problem_statement_optional(self):
        inp = DataSourceInput(measurements=[1.0])
        assert inp.problem_statement == ""


class TestDataSourcePlugin:
    def setup_method(self):
        self.plugin = DataSourcePlugin()

    def test_metadata(self):
        assert self.plugin.name == "data_source"
        assert self.plugin.version == "1.0.0"
        meta = DataSourcePlugin.get_metadata()
        assert meta["name"] == "data_source"
        assert meta["input_schema"] is not None

    def test_execute_full(self):
        validated = DataSourceInput(
            measurements=[1.0, 2.0, 3.0], usl=10.0, lsl=0.0, problem_statement="High scrap rate"
        ).model_dump()
        outputs = self.plugin.execute(validated, {"job_id": "test", "actor": "test"})

        keys = {o.key for o in outputs}
        assert "measurements" in keys
        assert "usl" in keys
        assert "lsl" in keys
        assert "problem_statement" in keys
        assert len(outputs) == 4

    def test_execute_data_only(self):
        validated = DataSourceInput(measurements=[5.0, 6.0]).model_dump()
        outputs = self.plugin.execute(validated, {"job_id": "test", "actor": "test"})

        keys = {o.key for o in outputs}
        assert "measurements" in keys
        assert "usl" not in keys
        assert "lsl" not in keys
        assert "problem_statement" not in keys
        assert len(outputs) == 1

    def test_execute_with_specs_no_problem(self):
        validated = DataSourceInput(measurements=[1.0], usl=10.0, lsl=2.0).model_dump()
        outputs = self.plugin.execute(validated, {"job_id": "test", "actor": "test"})
        assert len(outputs) == 3
        keys = {o.key for o in outputs}
        assert keys == {"measurements", "usl", "lsl"}

    def test_measurements_output_type(self):
        validated = DataSourceInput(measurements=[1.0, 2.0]).model_dump()
        outputs = self.plugin.execute(validated, {"job_id": "test", "actor": "test"})
        meas = next(o for o in outputs if o.key == "measurements")
        assert meas.output_type == "dataset"
        assert meas.value == [1.0, 2.0]

    def test_spec_output_type(self):
        validated = DataSourceInput(measurements=[1.0], usl=5.0).model_dump()
        outputs = self.plugin.execute(validated, {"job_id": "test", "actor": "test"})
        usl = next(o for o in outputs if o.key == "usl")
        assert usl.output_type == "metric"
        assert usl.value == 5.0

    def test_problem_statement_output(self):
        validated = DataSourceInput(measurements=[1.0], problem_statement="Why scrap?").model_dump()
        outputs = self.plugin.execute(validated, {"job_id": "test", "actor": "test"})
        ps = next(o for o in outputs if o.key == "problem_statement")
        assert ps.output_type == "text"
        assert ps.value == {"text": "Why scrap?"}

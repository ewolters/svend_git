"""Tests for FishbonePlugin — cause-and-effect analysis device."""

import pytest
from pydantic import ValidationError

from plugins.fishbone_device import CATEGORIES_6M, FishboneInput, FishbonePlugin


class TestFishboneInput:
    def test_valid(self):
        inp = FishboneInput(problem_statement="High scrap rate on Line 3")
        assert inp.problem_statement == "High scrap rate on Line 3"
        assert inp.categories == CATEGORIES_6M

    def test_empty_statement_rejected(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            FishboneInput(problem_statement="   ")

    def test_custom_categories(self):
        inp = FishboneInput(problem_statement="test", categories=["People", "Process", "Technology"])
        assert inp.categories == ["People", "Process", "Technology"]

    def test_dict_input_unwrapped(self):
        """Upstream text ports arrive as {"text": "..."} via engine routing."""
        inp = FishboneInput(problem_statement={"text": "Scrap on Line 3"})
        assert inp.problem_statement == "Scrap on Line 3"

    def test_empty_dict_rejected(self):
        with pytest.raises(ValidationError):
            FishboneInput(problem_statement={"text": "  "})


class TestFishbonePlugin:
    def setup_method(self):
        self.plugin = FishbonePlugin()

    def test_metadata(self):
        assert self.plugin.name == "fishbone"
        meta = FishbonePlugin.get_metadata()
        assert meta["input_schema"] is not None

    def test_execute(self):
        validated = FishboneInput(problem_statement="Bearing failures on Press 7").model_dump()
        outputs = self.plugin.execute(validated, {"job_id": "test", "actor": "test"})
        assert len(outputs) == 2

        causes = next(o for o in outputs if o.key == "causes")
        assert causes.output_type == "text"
        assert "Man" in causes.value["categories"]
        assert causes.value["problem"] == "Bearing failures on Press 7"

        diagram = next(o for o in outputs if o.key == "diagram")
        assert diagram.output_type == "chart"
        assert diagram.value["chart_type"] == "fishbone"
        assert len(diagram.value["categories"]) == 6

    def test_execute_custom_categories(self):
        validated = FishboneInput(
            problem_statement="Late deliveries", categories=["Supplier", "Transport", "Warehouse"]
        ).model_dump()
        outputs = self.plugin.execute(validated, {"job_id": "test", "actor": "test"})
        diagram = next(o for o in outputs if o.key == "diagram")
        assert diagram.value["categories"] == ["Supplier", "Transport", "Warehouse"]

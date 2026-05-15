"""End-to-end execution test for every seed template.

Each template is executed through the flowchart engine with realistic
synthetic data. This catches:
- Pydantic validation errors (field mismatches between template and plugin)
- Missing forge package attributes
- Engine routing bugs (port name → input_data key mapping)
- Multi-port aggregation failures

Process Monitoring is special — it requires PCL data in the DB.
We test it separately with injected synthetic data.

Run: pytest flowchart/tests/test_template_execution.py -v
"""

import copy

import numpy as np
import pytest

from flowchart.engine import execute_flowchart
from flowchart.management.commands.seed_templates import TEMPLATES
from syn.plugins.registry import get_registry

pytestmark = pytest.mark.django_db


@pytest.fixture(scope="module")
def registry():
    return get_registry()


@pytest.fixture(scope="module")
def measurements():
    np.random.seed(42)
    return list(np.random.normal(10.0, 0.5, 50))


# Config overrides per template — simulates user-supplied data at execution time.
# Entry-point devices (data_source, fmea_analysis, vsm_analysis, text_input)
# get their data from users, not from upstream connections.
def _config_overrides(measurements):
    return {
        "Quick Cpk": {
            "ds1": {"measurements": measurements, "usl": 12.0, "lsl": 8.0},
        },
        "PPAP Package": {
            "ds1": {"measurements": measurements, "usl": 12.0, "lsl": 8.0},
        },
        "Green Belt DMAIC": {
            "ds1": {
                "measurements": measurements,
                "usl": 12.0,
                "lsl": 8.0,
                "problem_statement": "Widget dimension out of spec",
            },
        },
        "VSM Improvement Cycle": {
            "vsm_cur": {
                "steps": [
                    {"name": "Cut", "cycle_time": 10, "changeover_time": 5, "uptime": 0.9},
                    {"name": "Weld", "cycle_time": 15, "changeover_time": 8, "uptime": 0.85},
                    {"name": "Paint", "cycle_time": 8, "changeover_time": 3, "uptime": 0.95},
                ]
            },
            "vsm_fut": {
                "steps": [
                    {"name": "Cut", "cycle_time": 7, "changeover_time": 3, "uptime": 0.95},
                    {"name": "Weld", "cycle_time": 10, "changeover_time": 5, "uptime": 0.92},
                    {"name": "Paint", "cycle_time": 6, "changeover_time": 2, "uptime": 0.97},
                ]
            },
        },
        "FMEA Risk Assessment": {
            "fmea": {
                "rows": [
                    {
                        "process_step": "Machining",
                        "failure_mode": "Tool wear",
                        "severity": 8,
                        "occurrence": 5,
                        "detection": 3,
                    },
                    {
                        "process_step": "Assembly",
                        "failure_mode": "Wrong torque",
                        "severity": 9,
                        "occurrence": 3,
                        "detection": 4,
                    },
                ]
            },
        },
        "Data Exploration": {
            "ds": {"measurements": measurements, "usl": 12.0, "lsl": 8.0},
        },
    }


def _executable_template_ids():
    return [t["name"] for t in TEMPLATES if t["name"] != "Process Monitoring"]


def _executable_templates():
    return [t for t in TEMPLATES if t["name"] != "Process Monitoring"]


class TestTemplateExecution:
    """Execute each template end-to-end through the flowchart engine."""

    @pytest.mark.parametrize(
        "tpl",
        _executable_templates(),
        ids=_executable_template_ids(),
    )
    def test_template_executes_all_devices(self, registry, measurements, tpl):
        overrides = _config_overrides(measurements)
        defn = copy.deepcopy(tpl["definition"])

        if tpl["name"] in overrides:
            for dev_id, conf in overrides[tpl["name"]].items():
                defn["config"][dev_id].update(conf)

        results = execute_flowchart(defn, actor="test@svend.ai", is_scratch=True, registry=registry)

        # Every device must execute
        expected_devices = {d["id"] for d in tpl["definition"]["devices"]}
        assert set(results.keys()) == expected_devices, (
            f"Not all devices executed. Expected {expected_devices}, got {set(results.keys())}"
        )

        # Every device must complete
        for dev_id, data in results.items():
            assert data["job"].status == "completed", f"Device {dev_id} status={data['job'].status}, expected completed"

        # Every device must produce at least one output
        for dev_id, data in results.items():
            assert data["outputs"], f"Device {dev_id} produced no outputs"

    def test_quick_cpk_produces_correct_metrics(self, registry, measurements):
        tpl = next(t for t in TEMPLATES if t["name"] == "Quick Cpk")
        defn = copy.deepcopy(tpl["definition"])
        defn["config"]["ds1"] = {"measurements": measurements, "usl": 12.0, "lsl": 8.0}

        results = execute_flowchart(defn, actor="test@svend.ai", is_scratch=True, registry=registry)

        cap = results["cap1"]["outputs"]
        assert "cpk" in cap, "Capability study must output cpk"
        assert "ppk" in cap, "Capability study must output ppk"
        assert isinstance(cap["cpk"], float), "cpk must be a float"
        assert cap["cpk"] > 0, "cpk must be positive"

    def test_ppap_report_has_sections(self, registry, measurements):
        tpl = next(t for t in TEMPLATES if t["name"] == "PPAP Package")
        defn = copy.deepcopy(tpl["definition"])
        defn["config"]["ds1"] = {"measurements": measurements, "usl": 12.0, "lsl": 8.0}

        results = execute_flowchart(defn, actor="test@svend.ai", is_scratch=True, registry=registry)

        report = results["rpt1"]["outputs"].get("report", {})
        assert isinstance(report, dict), "Report must be a dict"
        assert "sections" in report, "Report must have sections"

    def test_vsm_produces_lead_time(self, registry, measurements):
        tpl = next(t for t in TEMPLATES if t["name"] == "VSM Improvement Cycle")
        defn = copy.deepcopy(tpl["definition"])
        overrides = _config_overrides(measurements)["VSM Improvement Cycle"]
        for dev_id, conf in overrides.items():
            defn["config"][dev_id].update(conf)

        results = execute_flowchart(defn, actor="test@svend.ai", is_scratch=True, registry=registry)

        assert "lead_time" in results["vsm_cur"]["outputs"], "VSM must output lead_time"
        assert results["vsm_cur"]["outputs"]["lead_time"] >= 0

    def test_fmea_routes_contracts(self, registry, measurements):
        tpl = next(t for t in TEMPLATES if t["name"] == "FMEA Risk Assessment")
        defn = copy.deepcopy(tpl["definition"])
        overrides = _config_overrides(measurements)["FMEA Risk Assessment"]
        for dev_id, conf in overrides.items():
            defn["config"][dev_id].update(conf)

        results = execute_flowchart(defn, actor="test@svend.ai", is_scratch=True, registry=registry)

        router = results["router"]["outputs"]
        assert "strategic" in router or "tactical" in router or "quick_wins" in router, (
            "Contract router must produce at least one priority bucket"
        )

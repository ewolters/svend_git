"""Tests for ReportBuilderPlugin — aggregation device."""

from plugins.report_builder import ReportBuilderInput, ReportBuilderPlugin


class TestReportBuilderInput:
    def test_defaults(self):
        inp = ReportBuilderInput()
        assert inp.title == "Report"
        assert inp.charts is None
        assert inp.metrics is None

    def test_with_data(self):
        inp = ReportBuilderInput(title="PPAP", charts=[{"type": "histogram"}], metrics=[1.33])
        assert inp.title == "PPAP"
        assert len(inp.charts) == 1


class TestReportBuilderPlugin:
    def setup_method(self):
        self.plugin = ReportBuilderPlugin()

    def test_metadata(self):
        assert self.plugin.name == "report_builder"

    def test_execute_empty(self):
        validated = ReportBuilderInput().model_dump()
        outputs = self.plugin.execute(validated, {"job_id": "test", "actor": "test"})
        assert len(outputs) == 1
        report = outputs[0]
        assert report.key == "report"
        assert report.value["title"] == "Report"
        assert report.value["sections"] == []

    def test_execute_with_metrics(self):
        validated = ReportBuilderInput(title="Cpk Report", metrics=[1.33, 1.45]).model_dump()
        outputs = self.plugin.execute(validated, {"job_id": "test", "actor": "test"})
        report = outputs[0].value
        assert len(report["sections"]) == 1
        assert report["sections"][0]["type"] == "metrics"
        assert report["sections"][0]["items"] == [1.33, 1.45]

    def test_execute_with_all_inputs(self):
        validated = ReportBuilderInput(
            title="PPAP Package",
            charts=[{"type": "histogram"}, {"type": "control_chart"}],
            metrics=[1.33],
            text=[{"text": "Process is capable"}],
            lists=[{"violations": ["Rule 1"]}],
        ).model_dump()
        outputs = self.plugin.execute(validated, {"job_id": "test", "actor": "test"})
        report = outputs[0].value
        assert len(report["sections"]) == 4
        types = [s["type"] for s in report["sections"]]
        assert types == ["metrics", "charts", "text", "lists"]

    def test_scalar_input_normalized_to_list(self):
        validated = ReportBuilderInput(metrics=1.33).model_dump()
        outputs = self.plugin.execute(validated, {"job_id": "test", "actor": "test"})
        report = outputs[0].value
        assert report["sections"][0]["items"] == [1.33]

    def test_execute_title(self):
        validated = ReportBuilderInput(title="Green Belt DMAIC Report").model_dump()
        outputs = self.plugin.execute(validated, {"job_id": "test", "actor": "test"})
        assert outputs[0].value["title"] == "Green Belt DMAIC Report"

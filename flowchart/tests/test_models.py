from django.test import TestCase

from conftest import make_user
from flowchart.models import FlowchartInstance, FlowchartTemplate


class TestFlowchartTemplate(TestCase):
    def test_create_template(self):
        tpl = FlowchartTemplate.objects.create(
            name="Quick Cpk",
            description="Paste data, get Cpk",
            definition={
                "devices": [
                    {"plugin": "data_source", "id": "ds1"},
                    {"plugin": "capability_study", "id": "cap1"},
                ],
                "connections": [
                    {"source": "ds1.measurements", "target": "cap1.data"},
                    {"source": "ds1.usl", "target": "cap1.usl"},
                    {"source": "ds1.lsl", "target": "cap1.lsl"},
                ],
                "config": {"cap1": {"subgroup_size": 1}},
                "positions": {"ds1": {"x": 0, "y": 0}, "cap1": {"x": 300, "y": 0}},
            },
            devices_used=["data_source", "capability_study"],
        )
        assert tpl.id is not None
        assert tpl.name == "Quick Cpk"
        assert len(tpl.definition["devices"]) == 2
        assert len(tpl.definition["connections"]) == 3
        assert tpl.devices_used == ["data_source", "capability_study"]

    def test_clone_template(self):
        parent = FlowchartTemplate.objects.create(
            name="PPAP Package",
            definition={"devices": [], "connections": []},
            devices_used=[],
        )
        child = FlowchartTemplate.objects.create(
            name="My PPAP",
            definition=parent.definition.copy(),
            devices_used=parent.devices_used.copy(),
            parent_template=parent,
        )
        assert child.parent_template_id == parent.id
        assert child.name == "My PPAP"

    def test_query_by_device(self):
        FlowchartTemplate.objects.create(
            name="Has Cap",
            definition={},
            devices_used=["capability_study", "control_chart"],
        )
        FlowchartTemplate.objects.create(
            name="No Cap",
            definition={},
            devices_used=["control_chart"],
        )
        results = FlowchartTemplate.objects.filter(devices_used__contains=["capability_study"])
        assert results.count() == 1
        assert results.first().name == "Has Cap"


class TestFlowchartInstance(TestCase):
    def test_create_from_template(self):
        tpl = FlowchartTemplate.objects.create(
            name="Quick Cpk",
            definition={"devices": [{"plugin": "capability_study", "id": "cap1"}], "connections": []},
            devices_used=["capability_study"],
        )
        inst = FlowchartInstance.objects.create(
            name="My Cpk Run",
            template=tpl,
            definition=tpl.definition.copy(),
        )
        assert inst.template_id == tpl.id
        assert inst.is_scratch is False

    def test_scratch_instance(self):
        inst = FlowchartInstance.objects.create(
            name="Exploring",
            definition={"devices": [], "connections": []},
            is_scratch=True,
        )
        assert inst.is_scratch is True
        assert inst.template is None


class TestFlowchartInstanceModel(TestCase):
    def setUp(self):
        self.user = make_user("inst@test.com")

    def test_create_instance_from_template(self):
        tpl = FlowchartTemplate.objects.create(
            name="Test",
            definition={"devices": [], "connections": [], "config": {}},
            devices_used=["capability_study"],
        )
        inst = FlowchartInstance.objects.create(
            name="My Cpk Study",
            template=tpl,
            definition=tpl.definition.copy(),
            user=self.user,
        )
        assert inst.user == self.user
        assert inst.template == tpl
        assert inst.definition == tpl.definition

    def test_create_blank_instance(self):
        inst = FlowchartInstance.objects.create(
            name="Scratch",
            definition={"devices": [], "connections": [], "config": {}},
            user=self.user,
        )
        assert inst.template is None
        assert inst.user == self.user

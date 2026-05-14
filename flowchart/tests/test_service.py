from django.test import TestCase

from conftest import make_user
from flowchart.models import FlowchartInstance, FlowchartTemplate
from flowchart.service import (
    add_connection,
    add_device,
    create_instance,
    delete_instance,
    remove_connection,
    remove_device,
    validate_connection_request,
)


class TestCreateInstance(TestCase):
    def setUp(self):
        self.user = make_user("svc@test.com")

    def test_create_from_template(self):
        tpl = FlowchartTemplate.objects.create(
            name="Quick Cpk",
            definition={
                "devices": [{"id": "ds", "plugin": "data_source"}],
                "connections": [],
                "config": {"ds": {}},
            },
            devices_used=["data_source"],
        )
        inst = create_instance(
            template=tpl,
            name="My Study",
            user=self.user,
            actor=self.user.email,
        )
        assert inst.template == tpl
        assert inst.definition == tpl.definition
        assert inst.user == self.user

    def test_create_blank(self):
        inst = create_instance(
            template=None,
            name="Blank",
            user=self.user,
            actor=self.user.email,
        )
        assert inst.template is None
        assert inst.definition == {"devices": [], "connections": [], "config": {}}

    def test_create_emits_bus_event(self):
        from unittest.mock import patch

        with patch("flowchart.service.emit") as mock_emit:
            create_instance(
                template=None,
                name="Evt Test",
                user=self.user,
                actor=self.user.email,
            )
            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == "flowchart.instance.created"
            assert "instance_id" in call_args[0][1]


class TestDeleteInstance(TestCase):
    def setUp(self):
        self.user = make_user("svc@test.com")

    def test_soft_delete(self):
        inst = FlowchartInstance.objects.create(
            name="To Delete",
            definition={"devices": [], "connections": [], "config": {}},
            user=self.user,
        )
        delete_instance(inst, actor=self.user.email)
        inst.refresh_from_db()
        assert inst.is_deleted is True

    def test_delete_emits_event(self):
        from unittest.mock import patch

        inst = FlowchartInstance.objects.create(
            name="Del Evt",
            definition={"devices": [], "connections": [], "config": {}},
            user=self.user,
        )
        with patch("flowchart.service.emit") as mock_emit:
            delete_instance(inst, actor=self.user.email)
            mock_emit.assert_called_once()
            assert mock_emit.call_args[0][0] == "flowchart.instance.deleted"


class TestAddDevice(TestCase):
    def setUp(self):
        self.user = make_user("dev@test.com")
        self.inst = FlowchartInstance.objects.create(
            name="Test",
            definition={"devices": [], "connections": [], "config": {}},
            user=self.user,
        )

    def test_add_device(self):
        add_device(
            instance=self.inst,
            device_id="ds1",
            plugin_name="data_source",
            label="My Data",
            position={"x": 100, "y": 100},
            actor=self.user.email,
        )
        self.inst.refresh_from_db()
        devs = self.inst.definition["devices"]
        assert len(devs) == 1
        assert devs[0]["id"] == "ds1"
        assert devs[0]["plugin"] == "data_source"
        assert devs[0]["label"] == "My Data"

    def test_add_device_populates_port_schema(self):
        add_device(
            instance=self.inst,
            device_id="ds1",
            plugin_name="data_source",
            label="Data",
            position={"x": 0, "y": 0},
            actor=self.user.email,
        )
        self.inst.refresh_from_db()
        dev = self.inst.definition["devices"][0]
        assert "ports" in dev

    def test_add_device_emits_event(self):
        from unittest.mock import patch

        with patch("flowchart.service.emit") as mock_emit:
            add_device(
                instance=self.inst,
                device_id="ds1",
                plugin_name="data_source",
                label="Data",
                position={"x": 0, "y": 0},
                actor=self.user.email,
            )
            mock_emit.assert_called_once()
            assert mock_emit.call_args[0][0] == "flowchart.instance.device_added"

    def test_duplicate_device_id_raises(self):
        add_device(
            instance=self.inst,
            device_id="ds1",
            plugin_name="data_source",
            label="D1",
            position={"x": 0, "y": 0},
            actor=self.user.email,
        )
        with self.assertRaises(ValueError):
            add_device(
                instance=self.inst,
                device_id="ds1",
                plugin_name="data_source",
                label="D2",
                position={"x": 100, "y": 0},
                actor=self.user.email,
            )


class TestRemoveDevice(TestCase):
    def setUp(self):
        self.user = make_user("dev@test.com")
        self.inst = FlowchartInstance.objects.create(
            name="Test",
            definition={
                "devices": [
                    {"id": "ds1", "plugin": "data_source", "label": "Data"},
                    {"id": "cap1", "plugin": "capability_study", "label": "Cpk"},
                ],
                "connections": [
                    {"source": "ds1.measurements", "target": "cap1.data", "type": "data:column"},
                ],
                "config": {"ds1": {}, "cap1": {}},
                "positions": {"ds1": {"x": 0, "y": 0}, "cap1": {"x": 200, "y": 0}},
            },
            user=self.user,
        )

    def test_remove_device_and_connections(self):
        remove_device(instance=self.inst, device_id="cap1", actor=self.user.email)
        self.inst.refresh_from_db()
        ids = [d["id"] for d in self.inst.definition["devices"]]
        assert "cap1" not in ids
        assert len(self.inst.definition["connections"]) == 0
        assert "cap1" not in self.inst.definition["config"]

    def test_remove_emits_event(self):
        from unittest.mock import patch

        with patch("flowchart.service.emit") as mock_emit:
            remove_device(instance=self.inst, device_id="ds1", actor=self.user.email)
            mock_emit.assert_called_once()
            assert mock_emit.call_args[0][0] == "flowchart.instance.device_removed"

    def test_remove_missing_raises(self):
        with self.assertRaises(ValueError):
            remove_device(instance=self.inst, device_id="nope", actor=self.user.email)


class TestAddConnection(TestCase):
    def setUp(self):
        self.user = make_user("conn@test.com")
        self.inst = FlowchartInstance.objects.create(
            name="Test",
            definition={
                "devices": [
                    {
                        "id": "ds1",
                        "plugin": "data_source",
                        "label": "Data",
                        "ports": {
                            "inputs": [],
                            "outputs": [
                                {"name": "measurements", "type": "data:column"},
                                {"name": "usl", "type": "spec:usl"},
                                {"name": "lsl", "type": "spec:lsl"},
                            ],
                        },
                    },
                    {
                        "id": "cap1",
                        "plugin": "capability_study",
                        "label": "Cpk",
                        "ports": {
                            "inputs": [
                                {"name": "data", "type": "data:column"},
                                {"name": "usl", "type": "spec:usl"},
                                {"name": "lsl", "type": "spec:lsl"},
                            ],
                            "outputs": [
                                {"name": "cpk", "type": "metric:cpk"},
                            ],
                        },
                    },
                ],
                "connections": [],
                "config": {},
            },
            user=self.user,
        )

    def test_add_valid_connection(self):
        add_connection(
            instance=self.inst,
            source="ds1.measurements",
            target="cap1.data",
            actor=self.user.email,
        )
        self.inst.refresh_from_db()
        conns = self.inst.definition["connections"]
        assert len(conns) == 1
        assert conns[0]["source"] == "ds1.measurements"
        assert conns[0]["target"] == "cap1.data"
        assert conns[0]["type"] == "data:column"

    def test_add_type_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            add_connection(
                instance=self.inst,
                source="ds1.measurements",
                target="cap1.usl",
                actor=self.user.email,
            )
        assert "Type mismatch" in str(ctx.exception)

    def test_add_connection_emits_event(self):
        from unittest.mock import patch

        with patch("flowchart.service.emit") as mock_emit:
            add_connection(
                instance=self.inst,
                source="ds1.measurements",
                target="cap1.data",
                actor=self.user.email,
            )
            mock_emit.assert_called_once()
            assert mock_emit.call_args[0][0] == "flowchart.instance.connection_added"

    def test_add_connection_cycle_detection(self):
        self.inst.definition = {
            "devices": [
                {
                    "id": "a",
                    "plugin": "data_source",
                    "label": "A",
                    "ports": {
                        "inputs": [{"name": "in1", "type": "metric:cpk"}],
                        "outputs": [{"name": "out1", "type": "metric:cpk"}],
                    },
                },
                {
                    "id": "b",
                    "plugin": "data_source",
                    "label": "B",
                    "ports": {
                        "inputs": [{"name": "in1", "type": "metric:cpk"}],
                        "outputs": [{"name": "out1", "type": "metric:cpk"}],
                    },
                },
            ],
            "connections": [
                {"source": "a.out1", "target": "b.in1", "type": "metric:cpk"},
            ],
            "config": {},
        }
        self.inst.save(update_fields=["definition"])
        with self.assertRaises(ValueError) as ctx:
            add_connection(
                instance=self.inst,
                source="b.out1",
                target="a.in1",
                actor=self.user.email,
            )
        assert "cycle" in str(ctx.exception).lower()

    def test_duplicate_connection_raises(self):
        add_connection(
            instance=self.inst,
            source="ds1.measurements",
            target="cap1.data",
            actor=self.user.email,
        )
        with self.assertRaises(ValueError) as ctx:
            add_connection(
                instance=self.inst,
                source="ds1.measurements",
                target="cap1.data",
                actor=self.user.email,
            )
        assert "already exists" in str(ctx.exception).lower()


class TestRemoveConnection(TestCase):
    def setUp(self):
        self.user = make_user("conn@test.com")
        self.inst = FlowchartInstance.objects.create(
            name="Test",
            definition={
                "devices": [
                    {
                        "id": "ds1",
                        "plugin": "data_source",
                        "label": "D",
                        "ports": {"inputs": [], "outputs": [{"name": "measurements", "type": "data:column"}]},
                    },
                    {
                        "id": "cap1",
                        "plugin": "capability_study",
                        "label": "C",
                        "ports": {"inputs": [{"name": "data", "type": "data:column"}], "outputs": []},
                    },
                ],
                "connections": [
                    {"source": "ds1.measurements", "target": "cap1.data", "type": "data:column"},
                ],
                "config": {},
            },
            user=self.user,
        )

    def test_remove_connection(self):
        remove_connection(
            instance=self.inst,
            source="ds1.measurements",
            target="cap1.data",
            actor=self.user.email,
        )
        self.inst.refresh_from_db()
        assert len(self.inst.definition["connections"]) == 0

    def test_remove_emits_event(self):
        from unittest.mock import patch

        with patch("flowchart.service.emit") as mock_emit:
            remove_connection(
                instance=self.inst,
                source="ds1.measurements",
                target="cap1.data",
                actor=self.user.email,
            )
            assert mock_emit.call_args[0][0] == "flowchart.instance.connection_removed"

    def test_remove_missing_raises(self):
        with self.assertRaises(ValueError):
            remove_connection(
                instance=self.inst,
                source="ds1.nope",
                target="cap1.data",
                actor=self.user.email,
            )


class TestValidateConnectionRequest(TestCase):
    def test_valid(self):
        result = validate_connection_request(
            definition={
                "devices": [
                    {"id": "ds1", "ports": {"inputs": [], "outputs": [{"name": "out", "type": "metric:cpk"}]}},
                    {"id": "cap1", "ports": {"inputs": [{"name": "in1", "type": "metric:*"}], "outputs": []}},
                ],
                "connections": [],
            },
            source="ds1.out",
            target="cap1.in1",
        )
        assert result["valid"] is True

    def test_invalid_type(self):
        result = validate_connection_request(
            definition={
                "devices": [
                    {"id": "ds1", "ports": {"inputs": [], "outputs": [{"name": "out", "type": "metric:cpk"}]}},
                    {"id": "cap1", "ports": {"inputs": [{"name": "in1", "type": "chart:control"}], "outputs": []}},
                ],
                "connections": [],
            },
            source="ds1.out",
            target="cap1.in1",
        )
        assert result["valid"] is False

    def test_unknown_port(self):
        result = validate_connection_request(
            definition={
                "devices": [
                    {"id": "ds1", "ports": {"inputs": [], "outputs": []}},
                    {"id": "cap1", "ports": {"inputs": [], "outputs": []}},
                ],
                "connections": [],
            },
            source="ds1.nope",
            target="cap1.in1",
        )
        assert result["valid"] is False

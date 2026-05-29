# Flowchart Backend Complete — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the flowchart backend so that instances can be created, devices can be added/removed, connections can be wired/validated/removed, and all structural mutations flow through Synara (bus events, Job records where appropriate). Add text_input device and conditional (diamond) node. Add PCL-read source mode.

**Architecture:** FlowchartInstance becomes the user's working document. All structural mutations (add device, remove device, add connection, remove connection) go through a service layer that emits bus events and creates audit records. Connection validation uses the existing SemanticType system. A new text_input plugin lets users supply free-text with a user-chosen semantic subtype that becomes the output port label. Conditional nodes evaluate a predicate on an input port and route to one of two output branches.

**Tech Stack:** Django, Pydantic V2, syn.bus (EventBus), syn.plugins (Plugin ABC, PluginRegistry, run_plugin), flowchart.types (SemanticType, validate_connection), Job/JobOutput models.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `flowchart/service.py` | Create | Service layer for all flowchart mutations (create instance, add/remove device, add/remove connection). Emits bus events. All views call this, not raw ORM. |
| `flowchart/views.py` | Modify | Add 6 new endpoints: instance CRUD, add/remove device, add/remove connection, validate connection |
| `flowchart/models.py` | Modify | Add `user` FK on FlowchartInstance, add `FlowchartConnection` model (normalized, not just JSON) |
| `flowchart/types.py` | Modify | Add `validate_connection_in_context()` — checks type compat + cycle detection on a full graph |
| `flowchart/migrations/0002_instance_user_connection.py` | Create | Migration for model changes |
| `plugins/text_input.py` | Create | Text input device — user types prose, picks semantic subtype, port label is dynamic |
| `plugins/conditional.py` | Create | Conditional (diamond) node — evaluates predicate, routes to pass/fail branches |
| `plugins/pcl_source.py` | Create | PCL source device — pulls measures from PCL by slug, outputs typed ports |
| `plugins/apps.py` | Modify | Register 3 new plugins |
| `flowchart/tests/test_service.py` | Create | Tests for service layer (instance lifecycle, device ops, connection ops, events) |
| `flowchart/tests/test_instance_views.py` | Create | Tests for instance + wiring API endpoints |
| `plugins/tests/test_text_input.py` | Create | Tests for text_input plugin |
| `plugins/tests/test_conditional.py` | Create | Tests for conditional plugin |
| `plugins/tests/test_pcl_source.py` | Create | Tests for PCL source plugin |
| `svend/urls.py` | Modify | Wire new API endpoints |

---

## Task 1: FlowchartInstance Model + Migration

**Files:**
- Modify: `flowchart/models.py`
- Create: `flowchart/migrations/0002_instance_user_connection.py`

- [ ] **Step 1: Write the failing test**

```python
# flowchart/tests/test_models.py — add to existing file
from django.test import TestCase
from conftest import make_user
from flowchart.models import FlowchartInstance, FlowchartTemplate


class TestFlowchartInstanceModel(TestCase):
    def setUp(self):
        self.user = make_user("inst@test.com")

    def test_create_instance_from_template(self):
        tpl = FlowchartTemplate.objects.create(
            name="Test", definition={"devices": [], "connections": [], "config": {}},
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 manage.py test flowchart.tests.test_models.TestFlowchartInstanceModel -v2`
Expected: FAIL — `user` field doesn't exist on FlowchartInstance yet.

- [ ] **Step 3: Add user FK to FlowchartInstance**

In `flowchart/models.py`, add to FlowchartInstance:

```python
from django.conf import settings

class FlowchartInstance(SynaraEntity):
    name = models.CharField(max_length=255)
    template = models.ForeignKey(
        FlowchartTemplate, null=True, blank=True, on_delete=models.SET_NULL, related_name="instances"
    )
    definition = models.JSONField(default=dict)
    is_scratch = models.BooleanField(default=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="flowchart_instances",
        null=True, blank=True,
    )

    class Meta:
        db_table = "flowchart_instance"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
        ]

    class SynaraMeta:
        event_domain = "flowchart"
        emit_events = ["created", "updated"]

    def __str__(self):
        return f"Flowchart: {self.name}"
```

- [ ] **Step 4: Make and apply migration**

Run: `python3 manage.py makemigrations flowchart && python3 manage.py migrate flowchart`

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 manage.py test flowchart.tests.test_models.TestFlowchartInstanceModel -v2`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add flowchart/models.py flowchart/migrations/ flowchart/tests/test_models.py
git commit -m "feat(flowchart): add user FK to FlowchartInstance"
```

---

## Task 2: Flowchart Service Layer — Instance Lifecycle

The service layer mediates all flowchart mutations. Views call service functions, not raw ORM. Each mutation emits a bus event so Synara can track it for heuristics.

**Files:**
- Create: `flowchart/service.py`
- Create: `flowchart/tests/test_service.py`

- [ ] **Step 1: Write the failing test for create_instance**

```python
# flowchart/tests/test_service.py
from django.test import TestCase
from conftest import make_user
from flowchart.models import FlowchartInstance, FlowchartTemplate
from flowchart.service import create_instance, delete_instance


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
            template=tpl, name="My Study", user=self.user, actor=self.user.email,
        )
        assert inst.template == tpl
        assert inst.definition == tpl.definition  # snapshot copy
        assert inst.user == self.user

    def test_create_blank(self):
        inst = create_instance(
            template=None, name="Blank", user=self.user, actor=self.user.email,
        )
        assert inst.template is None
        assert inst.definition == {"devices": [], "connections": [], "config": {}}

    def test_create_emits_bus_event(self):
        from unittest.mock import patch
        with patch("flowchart.service.emit") as mock_emit:
            create_instance(
                template=None, name="Evt Test", user=self.user, actor=self.user.email,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 manage.py test flowchart.tests.test_service.TestCreateInstance flowchart.tests.test_service.TestDeleteInstance -v2`
Expected: FAIL — `flowchart.service` doesn't exist.

- [ ] **Step 3: Implement service layer — instance lifecycle**

```python
# flowchart/service.py
"""Flowchart service layer — all mutations go through here.

Every structural change emits a bus event so Synara can:
- Track for heuristics (which templates get used, which connections fail)
- Feed Claude context assembly
- Inform governance rules
"""

import copy
import logging
from typing import Optional

from syn.bus import emit

from flowchart.models import FlowchartInstance, FlowchartTemplate

logger = logging.getLogger(__name__)


def create_instance(
    *,
    template: Optional[FlowchartTemplate],
    name: str,
    user,
    actor: str,
    tenant_id=None,
    is_scratch: bool = False,
) -> FlowchartInstance:
    """Create a FlowchartInstance — from a template (snapshot) or blank."""
    if template:
        definition = copy.deepcopy(template.definition)
    else:
        definition = {"devices": [], "connections": [], "config": {}}

    inst = FlowchartInstance.objects.create(
        name=name,
        template=template,
        definition=definition,
        user=user,
        tenant_id=tenant_id,
        is_scratch=is_scratch,
    )

    emit(
        "flowchart.instance.created",
        {
            "instance_id": str(inst.id),
            "template_id": str(template.id) if template else None,
            "name": name,
        },
        actor=actor,
        tenant_id=str(tenant_id) if tenant_id else None,
    )

    return inst


def delete_instance(instance: FlowchartInstance, *, actor: str) -> None:
    """Soft-delete a FlowchartInstance."""
    instance.is_deleted = True
    instance.save(update_fields=["is_deleted"])

    emit(
        "flowchart.instance.deleted",
        {"instance_id": str(instance.id), "name": instance.name},
        actor=actor,
        tenant_id=str(instance.tenant_id) if instance.tenant_id else None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 manage.py test flowchart.tests.test_service.TestCreateInstance flowchart.tests.test_service.TestDeleteInstance -v2`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add flowchart/service.py flowchart/tests/test_service.py
git commit -m "feat(flowchart): service layer with create/delete instance + bus events"
```

---

## Task 3: Service Layer — Add/Remove Device

**Files:**
- Modify: `flowchart/service.py`
- Modify: `flowchart/tests/test_service.py`

- [ ] **Step 1: Write the failing test**

```python
# flowchart/tests/test_service.py — add these classes

from flowchart.service import add_device, remove_device


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
        """Port schema is copied from plugin registry into definition."""
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
        # data_source has outputs but no inputs
        assert len(dev["ports"]["outputs"]) > 0

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
            instance=self.inst, device_id="ds1", plugin_name="data_source",
            label="D1", position={"x": 0, "y": 0}, actor=self.user.email,
        )
        with self.assertRaises(ValueError):
            add_device(
                instance=self.inst, device_id="ds1", plugin_name="data_source",
                label="D2", position={"x": 100, "y": 0}, actor=self.user.email,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 manage.py test flowchart.tests.test_service.TestAddDevice flowchart.tests.test_service.TestRemoveDevice -v2`
Expected: FAIL — `add_device` and `remove_device` don't exist.

- [ ] **Step 3: Implement add_device and remove_device**

Add to `flowchart/service.py`:

```python
from syn.plugins.registry import get_registry


def _get_port_schema(plugin_name: str) -> dict:
    """Get port schema from plugin registry for embedding in definition."""
    registry = get_registry()
    plugin = registry.get(plugin_name)
    meta = plugin.get_metadata()
    # Port schema is derived from input_schema/output — plugins define this
    # For now, return empty ports. Templates carry the authoritative ports.
    # Build-mode devices get ports from plugin metadata.
    return {"inputs": [], "outputs": []}


def add_device(
    *,
    instance: FlowchartInstance,
    device_id: str,
    plugin_name: str,
    label: str,
    position: dict,
    actor: str,
    port_schema: dict = None,
) -> None:
    """Add a device to a FlowchartInstance definition.

    Validates plugin exists in registry, checks device_id uniqueness,
    adds device entry + config slot + position, saves, emits event.
    """
    # Validate plugin exists
    registry = get_registry()
    if not registry.has(plugin_name):
        raise ValueError(f"Unknown plugin: {plugin_name}")

    defn = instance.definition
    existing_ids = {d["id"] for d in defn.get("devices", [])}
    if device_id in existing_ids:
        raise ValueError(f"Device ID '{device_id}' already exists in this flowchart")

    # Build port schema from plugin metadata if not provided
    if port_schema is None:
        port_schema = _get_port_schema(plugin_name)

    device_entry = {
        "id": device_id,
        "plugin": plugin_name,
        "label": label,
        "ports": port_schema,
    }

    defn.setdefault("devices", []).append(device_entry)
    defn.setdefault("config", {})[device_id] = {}
    defn.setdefault("positions", {})[device_id] = position

    instance.definition = defn
    instance.save(update_fields=["definition", "updated_at"])

    emit(
        "flowchart.instance.device_added",
        {
            "instance_id": str(instance.id),
            "device_id": device_id,
            "plugin_name": plugin_name,
        },
        actor=actor,
        tenant_id=str(instance.tenant_id) if instance.tenant_id else None,
    )


def remove_device(*, instance: FlowchartInstance, device_id: str, actor: str) -> None:
    """Remove a device and all its connections from a FlowchartInstance."""
    defn = instance.definition
    existing_ids = {d["id"] for d in defn.get("devices", [])}
    if device_id not in existing_ids:
        raise ValueError(f"Device '{device_id}' not found in this flowchart")

    # Remove device
    defn["devices"] = [d for d in defn["devices"] if d["id"] != device_id]

    # Remove connections touching this device
    defn["connections"] = [
        c for c in defn.get("connections", [])
        if not c["source"].startswith(device_id + ".") and not c["target"].startswith(device_id + ".")
    ]

    # Remove config and position
    defn.get("config", {}).pop(device_id, None)
    defn.get("positions", {}).pop(device_id, None)

    instance.definition = defn
    instance.save(update_fields=["definition", "updated_at"])

    emit(
        "flowchart.instance.device_removed",
        {"instance_id": str(instance.id), "device_id": device_id},
        actor=actor,
        tenant_id=str(instance.tenant_id) if instance.tenant_id else None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 manage.py test flowchart.tests.test_service.TestAddDevice flowchart.tests.test_service.TestRemoveDevice -v2`
Expected: PASS (note: `test_add_device_populates_port_schema` will pass with empty ports for now — port schema population is enhanced in Task 7 when we add device-declared port schemas)

- [ ] **Step 5: Commit**

```bash
git add flowchart/service.py flowchart/tests/test_service.py
git commit -m "feat(flowchart): add/remove device in service layer with bus events"
```

---

## Task 4: Service Layer — Add/Remove/Validate Connection

**Files:**
- Modify: `flowchart/service.py`
- Modify: `flowchart/types.py`
- Modify: `flowchart/tests/test_service.py`

- [ ] **Step 1: Write the failing test**

```python
# flowchart/tests/test_service.py — add these classes

from flowchart.service import add_connection, remove_connection, validate_connection_request


class TestAddConnection(TestCase):
    def setUp(self):
        self.user = make_user("conn@test.com")
        self.inst = FlowchartInstance.objects.create(
            name="Test",
            definition={
                "devices": [
                    {
                        "id": "ds1", "plugin": "data_source", "label": "Data",
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
                        "id": "cap1", "plugin": "capability_study", "label": "Cpk",
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
                source="ds1.measurements",  # data:column
                target="cap1.usl",           # spec:usl — type mismatch
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
        """Adding a connection that creates a cycle raises ValueError."""
        # Create A->B->A cycle scenario
        self.inst.definition = {
            "devices": [
                {
                    "id": "a", "plugin": "data_source", "label": "A",
                    "ports": {
                        "inputs": [{"name": "in1", "type": "metric:cpk"}],
                        "outputs": [{"name": "out1", "type": "metric:cpk"}],
                    },
                },
                {
                    "id": "b", "plugin": "data_source", "label": "B",
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
            instance=self.inst, source="ds1.measurements", target="cap1.data",
            actor=self.user.email,
        )
        with self.assertRaises(ValueError) as ctx:
            add_connection(
                instance=self.inst, source="ds1.measurements", target="cap1.data",
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
                    {"id": "ds1", "plugin": "data_source", "label": "D",
                     "ports": {"inputs": [], "outputs": [{"name": "measurements", "type": "data:column"}]}},
                    {"id": "cap1", "plugin": "capability_study", "label": "C",
                     "ports": {"inputs": [{"name": "data", "type": "data:column"}], "outputs": []}},
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
                instance=self.inst, source="ds1.measurements", target="cap1.data",
                actor=self.user.email,
            )
            assert mock_emit.call_args[0][0] == "flowchart.instance.connection_removed"

    def test_remove_missing_raises(self):
        with self.assertRaises(ValueError):
            remove_connection(
                instance=self.inst, source="ds1.nope", target="cap1.data",
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 manage.py test flowchart.tests.test_service.TestAddConnection flowchart.tests.test_service.TestRemoveConnection flowchart.tests.test_service.TestValidateConnectionRequest -v2`
Expected: FAIL — functions don't exist.

- [ ] **Step 3: Add validate_connection_in_context to types.py**

Add to `flowchart/types.py`:

```python
def _has_cycle(devices: list, connections: list) -> bool:
    """Check if connections form a cycle using Kahn's algorithm."""
    adj = {d["id"]: [] for d in devices}
    in_deg = {d["id"]: 0 for d in devices}

    for conn in connections:
        src_dev = conn["source"].split(".", 1)[0]
        tgt_dev = conn["target"].split(".", 1)[0]
        if tgt_dev not in adj.get(src_dev, []):
            adj.setdefault(src_dev, []).append(tgt_dev)
            in_deg[tgt_dev] = in_deg.get(tgt_dev, 0) + 1

    queue = [d for d in in_deg if in_deg[d] == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for neighbor in adj.get(node, []):
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)

    return visited != len(in_deg)


def _find_port(devices: list, port_ref: str) -> dict:
    """Find a port definition by 'device_id.port_name' reference.

    Returns {"type": "...", "multi": bool, "direction": "input"|"output"} or None.
    """
    device_id, port_name = port_ref.split(".", 1)
    for dev in devices:
        if dev["id"] != device_id:
            continue
        for p in dev.get("ports", {}).get("inputs", []):
            if p["name"] == port_name:
                return {**p, "direction": "input"}
        for p in dev.get("ports", {}).get("outputs", []):
            if p["name"] == port_name:
                return {**p, "direction": "output"}
    return None


def validate_connection_in_context(
    definition: dict, source: str, target: str
) -> Dict:
    """Validate a proposed connection within a full flowchart definition.

    Checks:
    1. Source port exists and is an output
    2. Target port exists and is an input
    3. Semantic types are compatible
    4. Adding this connection doesn't create a cycle
    5. Connection doesn't already exist
    """
    devices = definition.get("devices", [])
    connections = definition.get("connections", [])

    # Check ports exist
    src_port = _find_port(devices, source)
    if not src_port:
        return {"valid": False, "error": f"Source port not found: {source}"}
    if src_port["direction"] != "output":
        return {"valid": False, "error": f"Source must be an output port: {source}"}

    tgt_port = _find_port(devices, target)
    if not tgt_port:
        return {"valid": False, "error": f"Target port not found: {target}"}
    if tgt_port["direction"] != "input":
        return {"valid": False, "error": f"Target must be an input port: {target}"}

    # Check type compatibility
    type_result = validate_connection(src_port["type"], tgt_port["type"])
    if not type_result["valid"]:
        return type_result

    # Check duplicate
    for conn in connections:
        if conn["source"] == source and conn["target"] == target:
            return {"valid": False, "error": f"Connection already exists: {source} -> {target}"}

    # Check cycle
    test_connections = connections + [{"source": source, "target": target}]
    if _has_cycle(devices, test_connections):
        return {"valid": False, "error": f"Connection would create a cycle: {source} -> {target}"}

    return {"valid": True, "error": "", "type": src_port["type"]}
```

- [ ] **Step 4: Implement add_connection, remove_connection, validate_connection_request in service.py**

Add to `flowchart/service.py`:

```python
from flowchart.types import validate_connection_in_context


def validate_connection_request(definition: dict, source: str, target: str) -> dict:
    """Validate a proposed connection without mutating anything."""
    return validate_connection_in_context(definition, source, target)


def add_connection(
    *, instance: FlowchartInstance, source: str, target: str, actor: str,
) -> None:
    """Add a validated connection to a FlowchartInstance."""
    defn = instance.definition
    result = validate_connection_in_context(defn, source, target)
    if not result["valid"]:
        raise ValueError(result["error"])

    conn_type = result.get("type", "")
    defn.setdefault("connections", []).append({
        "source": source,
        "target": target,
        "type": conn_type,
    })

    instance.definition = defn
    instance.save(update_fields=["definition", "updated_at"])

    emit(
        "flowchart.instance.connection_added",
        {
            "instance_id": str(instance.id),
            "source": source,
            "target": target,
            "type": conn_type,
        },
        actor=actor,
        tenant_id=str(instance.tenant_id) if instance.tenant_id else None,
    )


def remove_connection(
    *, instance: FlowchartInstance, source: str, target: str, actor: str,
) -> None:
    """Remove a connection from a FlowchartInstance."""
    defn = instance.definition
    conns = defn.get("connections", [])
    new_conns = [c for c in conns if not (c["source"] == source and c["target"] == target)]
    if len(new_conns) == len(conns):
        raise ValueError(f"Connection not found: {source} -> {target}")

    defn["connections"] = new_conns
    instance.definition = defn
    instance.save(update_fields=["definition", "updated_at"])

    emit(
        "flowchart.instance.connection_removed",
        {"instance_id": str(instance.id), "source": source, "target": target},
        actor=actor,
        tenant_id=str(instance.tenant_id) if instance.tenant_id else None,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 manage.py test flowchart.tests.test_service.TestAddConnection flowchart.tests.test_service.TestRemoveConnection flowchart.tests.test_service.TestValidateConnectionRequest -v2`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add flowchart/service.py flowchart/types.py flowchart/tests/test_service.py
git commit -m "feat(flowchart): add/remove/validate connection with type checking + cycle detection"
```

---

## Task 5: Instance + Wiring API Endpoints

**Files:**
- Modify: `flowchart/views.py`
- Create: `flowchart/tests/test_instance_views.py`
- Modify: `svend/urls.py`

- [ ] **Step 1: Write the failing test**

```python
# flowchart/tests/test_instance_views.py
import json
from django.test import TestCase
from conftest import SECURE_OFF, make_user
from flowchart.models import FlowchartInstance, FlowchartTemplate


@SECURE_OFF
class TestInstanceCreateEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("api@test.com")
        self.client.login(username="api", password="testpass123!")
        self.tpl = FlowchartTemplate.objects.create(
            name="Quick Cpk",
            definition={
                "devices": [{"id": "ds", "plugin": "data_source", "label": "Data",
                             "ports": {"inputs": [], "outputs": [{"name": "measurements", "type": "data:column"}]}}],
                "connections": [],
                "config": {"ds": {}},
            },
            devices_used=["data_source"],
        )

    def test_create_from_template(self):
        resp = self.client.post(
            "/api/flowchart/instances/",
            data=json.dumps({"template_id": str(self.tpl.id), "name": "My Study"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        assert "id" in body
        assert body["name"] == "My Study"
        assert body["template_id"] == str(self.tpl.id)

    def test_create_blank(self):
        resp = self.client.post(
            "/api/flowchart/instances/",
            data=json.dumps({"name": "Blank"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        assert body["template_id"] is None

    def test_requires_auth(self):
        self.client.logout()
        resp = self.client.post(
            "/api/flowchart/instances/",
            data=json.dumps({"name": "X"}),
            content_type="application/json",
        )
        assert resp.status_code in (401, 403)


@SECURE_OFF
class TestInstanceDetailEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("api@test.com")
        self.client.login(username="api", password="testpass123!")
        self.inst = FlowchartInstance.objects.create(
            name="My Flow", user=self.user,
            definition={"devices": [], "connections": [], "config": {}},
        )

    def test_get_instance(self):
        resp = self.client.get(f"/api/flowchart/instances/{self.inst.id}/")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        assert body["name"] == "My Flow"
        assert "definition" in body

    def test_delete_instance(self):
        resp = self.client.delete(f"/api/flowchart/instances/{self.inst.id}/")
        self.assertEqual(resp.status_code, 200)
        self.inst.refresh_from_db()
        assert self.inst.is_deleted is True


@SECURE_OFF
class TestAddDeviceEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("api@test.com")
        self.client.login(username="api", password="testpass123!")
        self.inst = FlowchartInstance.objects.create(
            name="Build Mode", user=self.user,
            definition={"devices": [], "connections": [], "config": {}},
        )

    def test_add_device(self):
        resp = self.client.post(
            f"/api/flowchart/instances/{self.inst.id}/devices/",
            data=json.dumps({
                "device_id": "ds1",
                "plugin_name": "data_source",
                "label": "My Data",
                "position": {"x": 100, "y": 100},
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.inst.refresh_from_db()
        assert len(self.inst.definition["devices"]) == 1


@SECURE_OFF
class TestRemoveDeviceEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("api@test.com")
        self.client.login(username="api", password="testpass123!")
        self.inst = FlowchartInstance.objects.create(
            name="RM", user=self.user,
            definition={
                "devices": [{"id": "ds1", "plugin": "data_source", "label": "D"}],
                "connections": [], "config": {"ds1": {}},
            },
        )

    def test_remove_device(self):
        resp = self.client.delete(
            f"/api/flowchart/instances/{self.inst.id}/devices/ds1/",
        )
        self.assertEqual(resp.status_code, 200)
        self.inst.refresh_from_db()
        assert len(self.inst.definition["devices"]) == 0


@SECURE_OFF
class TestConnectionEndpoints(TestCase):
    def setUp(self):
        self.user = make_user("api@test.com")
        self.client.login(username="api", password="testpass123!")
        self.inst = FlowchartInstance.objects.create(
            name="Wire", user=self.user,
            definition={
                "devices": [
                    {"id": "ds1", "plugin": "data_source", "label": "D",
                     "ports": {"inputs": [], "outputs": [{"name": "measurements", "type": "data:column"}]}},
                    {"id": "cap1", "plugin": "capability_study", "label": "C",
                     "ports": {"inputs": [{"name": "data", "type": "data:column"}], "outputs": []}},
                ],
                "connections": [], "config": {},
            },
        )

    def test_add_connection(self):
        resp = self.client.post(
            f"/api/flowchart/instances/{self.inst.id}/connections/",
            data=json.dumps({"source": "ds1.measurements", "target": "cap1.data"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.inst.refresh_from_db()
        assert len(self.inst.definition["connections"]) == 1

    def test_remove_connection(self):
        # Add first
        self.inst.definition["connections"].append(
            {"source": "ds1.measurements", "target": "cap1.data", "type": "data:column"},
        )
        self.inst.save(update_fields=["definition"])

        resp = self.client.post(
            f"/api/flowchart/instances/{self.inst.id}/connections/remove/",
            data=json.dumps({"source": "ds1.measurements", "target": "cap1.data"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        self.inst.refresh_from_db()
        assert len(self.inst.definition["connections"]) == 0

    def test_validate_connection(self):
        resp = self.client.post(
            f"/api/flowchart/instances/{self.inst.id}/connections/validate/",
            data=json.dumps({"source": "ds1.measurements", "target": "cap1.data"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        assert body["valid"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 manage.py test flowchart.tests.test_instance_views -v2`
Expected: FAIL — endpoints don't exist.

- [ ] **Step 3: Add view functions to flowchart/views.py**

Append to `flowchart/views.py`:

```python
from flowchart.service import (
    add_connection,
    add_device,
    create_instance,
    delete_instance,
    remove_connection,
    remove_device,
    validate_connection_request,
)


@csrf_exempt
@require_auth
def instance_create(request):
    """POST /api/flowchart/instances/ — create from template or blank."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    name = body.get("name", "Untitled")
    template_id = body.get("template_id")
    template = None

    if template_id:
        try:
            template = FlowchartTemplate.objects.get(id=template_id, is_deleted=False)
        except FlowchartTemplate.DoesNotExist:
            return JsonResponse({"error": "Template not found"}, status=404)

    inst = create_instance(
        template=template,
        name=name,
        user=request.user,
        actor=request.user.email,
        tenant_id=getattr(request.user, "tenant_id", None),
        is_scratch=body.get("is_scratch", False),
    )

    return JsonResponse({
        "id": str(inst.id),
        "name": inst.name,
        "template_id": str(inst.template_id) if inst.template_id else None,
        "definition": inst.definition,
        "created_at": inst.created_at.isoformat(),
    }, status=201)


@csrf_exempt
@require_auth
def instance_detail(request, instance_id):
    """GET/DELETE /api/flowchart/instances/<id>/"""
    try:
        inst = FlowchartInstance.objects.get(id=instance_id, is_deleted=False, user=request.user)
    except FlowchartInstance.DoesNotExist:
        return JsonResponse({"error": "Instance not found"}, status=404)

    if request.method == "DELETE":
        delete_instance(inst, actor=request.user.email)
        return JsonResponse({"deleted": True})

    return JsonResponse({
        "id": str(inst.id),
        "name": inst.name,
        "template_id": str(inst.template_id) if inst.template_id else None,
        "definition": inst.definition,
        "created_at": inst.created_at.isoformat(),
    })


@csrf_exempt
@require_auth
def instance_add_device(request, instance_id):
    """POST /api/flowchart/instances/<id>/devices/"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        inst = FlowchartInstance.objects.get(id=instance_id, is_deleted=False, user=request.user)
    except FlowchartInstance.DoesNotExist:
        return JsonResponse({"error": "Instance not found"}, status=404)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        add_device(
            instance=inst,
            device_id=body["device_id"],
            plugin_name=body["plugin_name"],
            label=body.get("label", body["plugin_name"]),
            position=body.get("position", {"x": 100, "y": 100}),
            actor=request.user.email,
            port_schema=body.get("port_schema"),
        )
    except (ValueError, KeyError) as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"ok": True, "definition": inst.definition})


@csrf_exempt
@require_auth
def instance_remove_device(request, instance_id, device_id):
    """DELETE /api/flowchart/instances/<id>/devices/<device_id>/"""
    if request.method != "DELETE":
        return JsonResponse({"error": "DELETE required"}, status=405)

    try:
        inst = FlowchartInstance.objects.get(id=instance_id, is_deleted=False, user=request.user)
    except FlowchartInstance.DoesNotExist:
        return JsonResponse({"error": "Instance not found"}, status=404)

    try:
        remove_device(instance=inst, device_id=device_id, actor=request.user.email)
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"ok": True, "definition": inst.definition})


@csrf_exempt
@require_auth
def instance_add_connection(request, instance_id):
    """POST /api/flowchart/instances/<id>/connections/"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        inst = FlowchartInstance.objects.get(id=instance_id, is_deleted=False, user=request.user)
    except FlowchartInstance.DoesNotExist:
        return JsonResponse({"error": "Instance not found"}, status=404)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        add_connection(
            instance=inst,
            source=body["source"],
            target=body["target"],
            actor=request.user.email,
        )
    except (ValueError, KeyError) as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"ok": True, "definition": inst.definition})


@csrf_exempt
@require_auth
def instance_remove_connection(request, instance_id):
    """POST /api/flowchart/instances/<id>/connections/remove/"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        inst = FlowchartInstance.objects.get(id=instance_id, is_deleted=False, user=request.user)
    except FlowchartInstance.DoesNotExist:
        return JsonResponse({"error": "Instance not found"}, status=404)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        remove_connection(
            instance=inst,
            source=body["source"],
            target=body["target"],
            actor=request.user.email,
        )
    except (ValueError, KeyError) as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"ok": True, "definition": inst.definition})


@csrf_exempt
@require_auth
def instance_validate_connection(request, instance_id):
    """POST /api/flowchart/instances/<id>/connections/validate/"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        inst = FlowchartInstance.objects.get(id=instance_id, is_deleted=False, user=request.user)
    except FlowchartInstance.DoesNotExist:
        return JsonResponse({"error": "Instance not found"}, status=404)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    result = validate_connection_request(inst.definition, body["source"], body["target"])
    return JsonResponse(result)
```

- [ ] **Step 4: Wire URLs in svend/urls.py**

Add these paths near the existing flowchart URLs:

```python
    # Flowchart instances
    path("api/flowchart/instances/",
         __import__("flowchart.views", fromlist=["instance_create"]).instance_create,
         name="flowchart_instance_create"),
    path("api/flowchart/instances/<uuid:instance_id>/",
         __import__("flowchart.views", fromlist=["instance_detail"]).instance_detail,
         name="flowchart_instance_detail"),
    path("api/flowchart/instances/<uuid:instance_id>/devices/",
         __import__("flowchart.views", fromlist=["instance_add_device"]).instance_add_device,
         name="flowchart_instance_add_device"),
    path("api/flowchart/instances/<uuid:instance_id>/devices/<str:device_id>/",
         __import__("flowchart.views", fromlist=["instance_remove_device"]).instance_remove_device,
         name="flowchart_instance_remove_device"),
    path("api/flowchart/instances/<uuid:instance_id>/connections/",
         __import__("flowchart.views", fromlist=["instance_add_connection"]).instance_add_connection,
         name="flowchart_instance_add_connection"),
    path("api/flowchart/instances/<uuid:instance_id>/connections/remove/",
         __import__("flowchart.views", fromlist=["instance_remove_connection"]).instance_remove_connection,
         name="flowchart_instance_remove_connection"),
    path("api/flowchart/instances/<uuid:instance_id>/connections/validate/",
         __import__("flowchart.views", fromlist=["instance_validate_connection"]).instance_validate_connection,
         name="flowchart_instance_validate_connection"),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 manage.py test flowchart.tests.test_instance_views -v2`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add flowchart/views.py flowchart/tests/test_instance_views.py svend/urls.py
git commit -m "feat(flowchart): instance + wiring API endpoints (7 new routes)"
```

---

## Task 6: Text Input Plugin

User types prose and picks a semantic subtype from a dropdown. The output port label and type are dynamic based on that choice.

**Files:**
- Create: `plugins/text_input.py`
- Create: `plugins/tests/test_text_input.py`
- Modify: `plugins/apps.py`

- [ ] **Step 1: Write the failing test**

```python
# plugins/tests/test_text_input.py
from django.test import TestCase
from syn.plugins.base import PluginOutput
from syn.plugins.registry import PluginRegistry


class TestTextInputPlugin(TestCase):
    def setUp(self):
        from plugins.text_input import TextInputPlugin
        self.registry = PluginRegistry()
        self.registry.register(TextInputPlugin)
        self.plugin = self.registry.get("text_input")

    def test_basic_output(self):
        outputs = self.plugin.execute(
            {"content": "Part surface finish exceeds tolerance", "subtype": "problem_statement"},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        assert len(outputs) == 1
        assert outputs[0].key == "problem_statement"
        assert outputs[0].output_type == "text"
        assert outputs[0].value == {"text": "Part surface finish exceeds tolerance"}

    def test_observation_subtype(self):
        outputs = self.plugin.execute(
            {"content": "Cycle time increasing since Monday", "subtype": "observation"},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        assert outputs[0].key == "observation"
        assert outputs[0].value["text"] == "Cycle time increasing since Monday"

    def test_custom_subtype(self):
        """Users can type any subtype — not limited to a fixed list."""
        outputs = self.plugin.execute(
            {"content": "Goal: reduce lead time 30%", "subtype": "goal"},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        assert outputs[0].key == "goal"

    def test_empty_content_raises(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self.plugin.input_schema(content="", subtype="problem_statement")

    def test_default_subtype(self):
        """Default subtype is 'note' if not specified."""
        outputs = self.plugin.execute(
            {"content": "Some text"},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        assert outputs[0].key == "note"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 manage.py test plugins.tests.test_text_input -v2`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement text_input plugin**

```python
# plugins/text_input.py
"""Text Input Plugin — user-authored prose with dynamic semantic subtype.

The user types free text and picks a subtype (problem_statement, observation,
hypothesis, goal, note, etc.). The output port name and semantic type are
set dynamically based on that choice.

This replaces having problem_statement baked into data_source.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class TextInputInput(BaseModel):
    content: str
    subtype: str = "note"

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()

    @field_validator("subtype")
    @classmethod
    def subtype_valid(cls, v):
        if not v.strip():
            raise ValueError("Subtype cannot be empty")
        return v.strip().lower().replace(" ", "_")


class TextInputPlugin(Plugin):
    """Free text entry — user picks the semantic subtype, port label follows."""

    name = "text_input"
    version = "1.0.0"
    description = "Text entry with dynamic semantic type (problem statement, observation, goal, etc.)"
    input_schema = TextInputInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        subtype = validated_input["subtype"]
        content = validated_input["content"]

        return [
            PluginOutput(
                key=subtype,
                output_type="text",
                value={"text": content},
            ),
        ]
```

- [ ] **Step 4: Register in plugins/apps.py**

Add to the imports and registration list:

```python
from plugins.text_input import TextInputPlugin
```

Add `TextInputPlugin` to the `for plugin_cls in [...]` list.

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 manage.py test plugins.tests.test_text_input -v2`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add plugins/text_input.py plugins/tests/test_text_input.py plugins/apps.py
git commit -m "feat(plugins): text_input device — dynamic semantic subtype"
```

---

## Task 7: Conditional (Diamond) Plugin

Evaluates a predicate on an input value and routes to a `pass` or `fail` output branch. This is the first non-rectangular device — the renderer will draw it as a diamond.

**Files:**
- Create: `plugins/conditional.py`
- Create: `plugins/tests/test_conditional.py`
- Modify: `plugins/apps.py`

- [ ] **Step 1: Write the failing test**

```python
# plugins/tests/test_conditional.py
from django.test import TestCase
from syn.plugins.registry import PluginRegistry


class TestConditionalPlugin(TestCase):
    def setUp(self):
        from plugins.conditional import ConditionalPlugin
        self.registry = PluginRegistry()
        self.registry.register(ConditionalPlugin)
        self.plugin = self.registry.get("conditional")

    def test_pass_when_above_threshold(self):
        outputs = self.plugin.execute(
            {"value": 1.5, "operator": ">=", "threshold": 1.33},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["result"].value is True
        assert by_key["pass_value"].value == 1.5
        assert by_key["fail_value"].value is None

    def test_fail_when_below_threshold(self):
        outputs = self.plugin.execute(
            {"value": 0.9, "operator": ">=", "threshold": 1.33},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["result"].value is False
        assert by_key["pass_value"].value is None
        assert by_key["fail_value"].value == 0.9

    def test_less_than_operator(self):
        outputs = self.plugin.execute(
            {"value": 5.0, "operator": "<", "threshold": 10.0},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["result"].value is True

    def test_equals_operator(self):
        outputs = self.plugin.execute(
            {"value": 3.0, "operator": "==", "threshold": 3.0},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["result"].value is True

    def test_invalid_operator_raises(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self.plugin.input_schema(value=1.0, operator="LIKE", threshold=2.0)

    def test_label_in_output(self):
        """The label is included so downstream devices know the context."""
        outputs = self.plugin.execute(
            {"value": 1.5, "operator": ">=", "threshold": 1.33, "label": "Cpk check"},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["summary"].value["label"] == "Cpk check"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 manage.py test plugins.tests.test_conditional -v2`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement conditional plugin**

```python
# plugins/conditional.py
"""Conditional (Diamond) Plugin — route based on a predicate.

Evaluates: value <operator> threshold
Outputs: result (bool), pass_value (value if true, else None),
         fail_value (value if false, else None), summary (text).

The pass_value and fail_value outputs allow downstream devices to be
wired to only one branch — unconnected ports cost nothing.
"""

import operator as op
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput

OPERATORS = {
    ">": op.gt,
    ">=": op.ge,
    "<": op.lt,
    "<=": op.le,
    "==": op.eq,
    "!=": op.ne,
}


class ConditionalInput(BaseModel):
    value: float
    operator: str
    threshold: float
    label: str = ""

    @field_validator("operator")
    @classmethod
    def operator_valid(cls, v):
        if v not in OPERATORS:
            raise ValueError(f"Invalid operator '{v}'. Must be one of: {', '.join(OPERATORS)}")
        return v


class ConditionalPlugin(Plugin):
    """Decision node — evaluates predicate, routes to pass or fail branch."""

    name = "conditional"
    version = "1.0.0"
    description = "Conditional gate — routes value to pass/fail branch based on threshold"
    input_schema = ConditionalInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        value = validated_input["value"]
        threshold = validated_input["threshold"]
        op_str = validated_input["operator"]
        label = validated_input.get("label", "")
        op_fn = OPERATORS[op_str]

        passed = op_fn(value, threshold)

        return [
            PluginOutput("result", "metric", passed),
            PluginOutput("pass_value", "metric", value if passed else None),
            PluginOutput("fail_value", "metric", value if not passed else None),
            PluginOutput(
                "summary", "text",
                {
                    "text": f"{'PASS' if passed else 'FAIL'}: {value} {op_str} {threshold}",
                    "label": label,
                    "passed": passed,
                    "value": value,
                    "operator": op_str,
                    "threshold": threshold,
                },
            ),
        ]
```

- [ ] **Step 4: Register in plugins/apps.py**

Add import and registration:

```python
from plugins.conditional import ConditionalPlugin
```

Add `ConditionalPlugin` to the registration list.

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 manage.py test plugins.tests.test_conditional -v2`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add plugins/conditional.py plugins/tests/test_conditional.py plugins/apps.py
git commit -m "feat(plugins): conditional (diamond) device — predicate routing"
```

---

## Task 8: PCL Source Plugin

Pulls measures from PCL by slug. Outputs typed ports based on what it reads. This is how VSM and other non-data-entry flowcharts get their input.

**Files:**
- Create: `plugins/pcl_source.py`
- Create: `plugins/tests/test_pcl_source.py`
- Modify: `plugins/apps.py`

- [ ] **Step 1: Write the failing test**

```python
# plugins/tests/test_pcl_source.py
from django.test import TestCase
from syn.plugins.registry import PluginRegistry


class TestPCLSourcePlugin(TestCase):
    def setUp(self):
        from plugins.pcl_source import PCLSourcePlugin
        self.registry = PluginRegistry()
        self.registry.register(PCLSourcePlugin)
        self.plugin = self.registry.get("pcl_source")

    def test_reads_existing_measure(self):
        """When a measure exists, output its current value."""
        from pcl.models import Measure, Datapoint

        m = Measure.objects.create(
            slug="test-cpk",
            name="Test Cpk",
            unit="ratio",
            measure_type="kpi",
            value_type="continuous",
        )
        Datapoint.objects.create(
            measure=m, value=1.45, source_type="manual", actor="test@svend.ai",
        )

        outputs = self.plugin.execute(
            {"measure_slugs": ["test-cpk"]},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert "test-cpk" in by_key
        assert by_key["test-cpk"].value == 1.45
        assert by_key["test-cpk"].output_type == "metric"
        assert by_key["test-cpk"].provenance == "observed"

    def test_missing_measure_outputs_none(self):
        """Missing measures produce a metric output with None value."""
        outputs = self.plugin.execute(
            {"measure_slugs": ["nonexistent-slug"]},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert "nonexistent-slug" in by_key
        assert by_key["nonexistent-slug"].value is None

    def test_multiple_measures(self):
        from pcl.models import Measure, Datapoint

        for slug, val in [("dim-a", 50.1), ("dim-b", 49.8)]:
            m = Measure.objects.create(
                slug=slug, name=slug, unit="mm",
                measure_type="dimension", value_type="continuous",
            )
            Datapoint.objects.create(
                measure=m, value=val, source_type="manual", actor="test@svend.ai",
            )

        outputs = self.plugin.execute(
            {"measure_slugs": ["dim-a", "dim-b"]},
            {"job_id": "test", "actor": "test@svend.ai"},
        )
        by_key = {o.key: o for o in outputs}
        assert by_key["dim-a"].value == 50.1
        assert by_key["dim-b"].value == 49.8

    def test_empty_slugs_raises(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self.plugin.input_schema(measure_slugs=[])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 manage.py test plugins.tests.test_pcl_source -v2`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement PCL source plugin**

```python
# plugins/pcl_source.py
"""PCL Source Plugin — pull measures from Process Characteristics Library.

Reads one or more measures by slug and outputs their current values
as metric ports. This is how non-data-entry flowcharts (VSM, monitoring)
get their input — by reading the process state from PCL.
"""

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput

logger = logging.getLogger(__name__)


class PCLSourceInput(BaseModel):
    measure_slugs: List[str]

    @field_validator("measure_slugs")
    @classmethod
    def slugs_not_empty(cls, v):
        if len(v) < 1:
            raise ValueError("Need at least one measure slug")
        return v


class PCLSourcePlugin(Plugin):
    """Reads measures from PCL — the process state entry point for flowcharts."""

    name = "pcl_source"
    version = "1.0.0"
    description = "PCL source — pull live process measures into a flowchart"
    input_schema = PCLSourceInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        from pcl.models import Datapoint, Measure

        outputs = []
        for slug in validated_input["measure_slugs"]:
            try:
                measure = Measure.objects.get(slug=slug, is_deleted=False)
            except Measure.DoesNotExist:
                logger.debug("[PCL_SOURCE] Measure '%s' not found", slug)
                outputs.append(PluginOutput(slug, "metric", None, provenance="observed"))
                continue

            # Get latest datapoint
            latest = (
                Datapoint.objects.filter(measure=measure)
                .order_by("-created_at")
                .first()
            )

            value = latest.value if latest else None
            provenance = getattr(latest, "provenance", "observed") if latest else "observed"

            outputs.append(PluginOutput(
                key=slug,
                output_type="metric",
                value=value,
                provenance=provenance,
                measure_slug=slug,
            ))

        return outputs
```

- [ ] **Step 4: Register in plugins/apps.py**

Add import and registration:

```python
from plugins.pcl_source import PCLSourcePlugin
```

Add `PCLSourcePlugin` to the registration list.

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 manage.py test plugins.tests.test_pcl_source -v2`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add plugins/pcl_source.py plugins/tests/test_pcl_source.py plugins/apps.py
git commit -m "feat(plugins): pcl_source device — pull live measures into flowcharts"
```

---

## Task 9: Execute Instance Endpoint

Currently `flowchart_run` takes an inline definition. Add an endpoint that executes a FlowchartInstance by ID — loads its definition, runs the engine, stores results back on the instance.

**Files:**
- Modify: `flowchart/views.py`
- Modify: `flowchart/tests/test_instance_views.py`
- Modify: `svend/urls.py`

- [ ] **Step 1: Write the failing test**

```python
# flowchart/tests/test_instance_views.py — add this class

from syn.plugins.base import Plugin, PluginOutput
from syn.plugins.registry import get_registry
from pydantic import BaseModel


class _ExecTestInput(BaseModel):
    value: float


class _ExecTestPlugin(Plugin):
    name = "exec_test_plugin"
    version = "1.0.0"
    description = "Test"
    input_schema = _ExecTestInput

    def execute(self, validated_input, context):
        return [PluginOutput("result", "metric", validated_input["value"] * 2)]


@SECURE_OFF
class TestExecuteInstanceEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("exec@test.com")
        self.client.login(username="exec", password="testpass123!")
        reg = get_registry()
        if not reg.has("exec_test_plugin"):
            reg.register(_ExecTestPlugin)
        self.inst = FlowchartInstance.objects.create(
            name="Exec Test", user=self.user,
            definition={
                "devices": [{"id": "d1", "plugin": "exec_test_plugin"}],
                "connections": [],
                "config": {"d1": {"value": 5.0}},
            },
        )

    def test_execute_instance(self):
        resp = self.client.post(f"/api/flowchart/instances/{self.inst.id}/run/")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        assert "d1" in body["results"]
        assert body["results"]["d1"]["outputs"]["result"] == 10.0

    def test_execute_stores_job_ids(self):
        """After execution, instance can be queried for job history."""
        resp = self.client.post(f"/api/flowchart/instances/{self.inst.id}/run/")
        body = resp.json()
        assert "job_id" in body["results"]["d1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 manage.py test flowchart.tests.test_instance_views.TestExecuteInstanceEndpoint -v2`
Expected: FAIL — endpoint doesn't exist.

- [ ] **Step 3: Implement execute instance endpoint**

Add to `flowchart/views.py`:

```python
@csrf_exempt
@require_auth
@require_POST
def instance_run(request, instance_id):
    """POST /api/flowchart/instances/<id>/run/ — execute a flowchart instance."""
    try:
        inst = FlowchartInstance.objects.get(id=instance_id, is_deleted=False, user=request.user)
    except FlowchartInstance.DoesNotExist:
        return JsonResponse({"error": "Instance not found"}, status=404)

    try:
        results = execute_flowchart(
            definition=inst.definition,
            actor=request.user.email,
            tenant_id=getattr(request.user, "tenant_id", None),
            is_scratch=inst.is_scratch,
            flowchart_instance_id=inst.id,
        )
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception(f"[FLOWCHART] Instance execution failed: {e}")
        return JsonResponse({"error": str(e)}, status=500)

    response = {"results": {}}
    for device_id, data in results.items():
        job = data["job"]
        response["results"][device_id] = {
            "job_id": str(job.id),
            "status": job.status,
            "duration_ms": job.duration_ms,
            "outputs": data["outputs"],
        }

    return JsonResponse(response)
```

Add URL to `svend/urls.py`:

```python
    path("api/flowchart/instances/<uuid:instance_id>/run/",
         __import__("flowchart.views", fromlist=["instance_run"]).instance_run,
         name="flowchart_instance_run"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 manage.py test flowchart.tests.test_instance_views.TestExecuteInstanceEndpoint -v2`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add flowchart/views.py flowchart/tests/test_instance_views.py svend/urls.py
git commit -m "feat(flowchart): execute instance endpoint — POST /api/flowchart/instances/<id>/run/"
```

---

## Task 10: Register Event Schemas + Full Test Suite

Register event schemas for all new flowchart events. Run the complete test suite to verify no regressions.

**Files:**
- Modify: `plugins/apps.py`

- [ ] **Step 1: Add event schemas for flowchart events**

In `plugins/apps.py`, add to `_register_event_schemas`:

```python
        # Flowchart instance events
        instance_events = [
            "flowchart.instance.created",
            "flowchart.instance.deleted",
            "flowchart.instance.device_added",
            "flowchart.instance.device_removed",
            "flowchart.instance.connection_added",
            "flowchart.instance.connection_removed",
        ]
        instance_schema = {
            "type": "object",
            "properties": {
                "instance_id": {"type": "string", "format": "uuid"},
            },
            "required": ["instance_id"],
        }
        for event_name in instance_events:
            try:
                EventSchemaRegistry.objects.update_or_create(
                    event_name=event_name,
                    defaults={
                        "version": "1.0.0",
                        "schema": instance_schema,
                        "tenant_id": SYSTEM_TENANT_ID,
                        "is_active": True,
                    },
                )
            except Exception as exc:
                logger.warning(f"[PLUGINS] Could not register event schema {event_name}: {exc}")
```

- [ ] **Step 2: Run the full flowchart + plugins test suite**

Run: `python3 manage.py test flowchart plugins --no-input -v2`
Expected: All tests pass (previous 62 + new tests from Tasks 1-9).

- [ ] **Step 3: Commit**

```bash
git add plugins/apps.py
git commit -m "feat(flowchart): register event schemas for all instance lifecycle events"
```

---

## Summary

| Task | What | Tests |
|------|------|-------|
| 1 | FlowchartInstance user FK + migration | 2 |
| 2 | Service layer — create/delete instance + bus events | 4 |
| 3 | Service layer — add/remove device + bus events | 7 |
| 4 | Service layer — add/remove/validate connection + type checking + cycle detection | 12 |
| 5 | Instance + wiring API endpoints (7 routes) | 9 |
| 6 | Text input plugin (dynamic semantic subtype) | 5 |
| 7 | Conditional (diamond) plugin | 6 |
| 8 | PCL source plugin | 4 |
| 9 | Execute instance endpoint | 2 |
| 10 | Event schemas + full regression | 0 (regression run) |
| **Total** | | **~51 new tests** |

**New API surface:**

| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/flowchart/instances/` | Create instance (from template or blank) |
| GET | `/api/flowchart/instances/<id>/` | Get instance with definition |
| DELETE | `/api/flowchart/instances/<id>/` | Soft-delete instance |
| POST | `/api/flowchart/instances/<id>/devices/` | Add device |
| DELETE | `/api/flowchart/instances/<id>/devices/<device_id>/` | Remove device + connections |
| POST | `/api/flowchart/instances/<id>/connections/` | Add validated connection |
| POST | `/api/flowchart/instances/<id>/connections/remove/` | Remove connection |
| POST | `/api/flowchart/instances/<id>/connections/validate/` | Validate proposed connection |
| POST | `/api/flowchart/instances/<id>/run/` | Execute instance |

**New plugins:** text_input, conditional, pcl_source (13 devices total after registration).

**Bus events:** 6 new event types for heuristics — all flowchart structural mutations are observable.

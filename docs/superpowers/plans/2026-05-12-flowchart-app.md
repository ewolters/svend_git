# Flowchart App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the `flowchart/` Django app with Connection + Template models and a topological-sort execution engine that routes data between plugin devices through typed ports.

**Architecture:** New Django app at `~/kjerne/flowchart/`. Three models (FlowchartTemplate, FlowchartInstance, FlowchartConnection) extend SynaraEntity. Execution engine does topological sort on the connection graph and calls `run_plugin()` for each device in order, routing outputs to downstream inputs. Two API endpoints: run a flowchart, list templates. Not surfaced in UI — backend only, tested via API.

**Tech Stack:** Django, SynaraEntity base model, existing `syn.plugins.runner.run_plugin()`, existing `job.models.Job/JobOutput`, pytest + Django TestCase

**Reference:** `sandbox/vsm_full_loop.py` lines 130-175 (working execution engine), `sandbox/semantic_types.py` (type validation)

---

### Task 1: Create the flowchart Django app skeleton

**Files:**
- Create: `flowchart/__init__.py`
- Create: `flowchart/apps.py`
- Create: `flowchart/models.py`
- Modify: `svend/settings.py:82` (add to INSTALLED_APPS)

- [ ] **Step 1: Create app directory and __init__.py**

```bash
mkdir -p ~/kjerne/flowchart
touch ~/kjerne/flowchart/__init__.py
```

- [ ] **Step 2: Create apps.py**

```python
from django.apps import AppConfig


class FlowchartConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "flowchart"
    verbose_name = "Flowchart — Device Connections & Templates"
```

- [ ] **Step 3: Create empty models.py**

```python
"""Flowchart models — connections, templates, instances.

Devices are plugins (registered in syn.plugins.registry).
Connections are typed edges between device ports.
Templates are pre-wired flowcharts (JSON blob).
Instances are user's working copies of templates.
"""
```

- [ ] **Step 4: Add to INSTALLED_APPS**

In `svend/settings.py`, after the `"plugins"` line (line 76), add:

```python
    "flowchart",  # Flowchart: device connections, templates, execution engine
```

- [ ] **Step 5: Verify Django finds the app**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 manage.py check --tag models 2>&1 | tail -5
```

Expected: `System check identified no issues.`

---

### Task 2: FlowchartTemplate model

**Files:**
- Modify: `flowchart/models.py`
- Create: `flowchart/tests/__init__.py`
- Create: `flowchart/tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Create `flowchart/tests/__init__.py` (empty) and `flowchart/tests/test_models.py`:

```python
"""Tests for flowchart models."""

from django.test import TestCase

from flowchart.models import FlowchartTemplate


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
                "config": {
                    "cap1": {"subgroup_size": 1},
                },
                "positions": {
                    "ds1": {"x": 0, "y": 0},
                    "cap1": {"x": 300, "y": 0},
                },
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
        results = FlowchartTemplate.objects.filter(
            devices_used__contains=["capability_study"]
        )
        assert results.count() == 1
        assert results.first().name == "Has Cap"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest flowchart/tests/test_models.py -v --no-header 2>&1 | tail -10
```

Expected: ImportError — `FlowchartTemplate` doesn't exist yet.

- [ ] **Step 3: Write the model**

Add to `flowchart/models.py`:

```python
from django.db import models

from syn.core.base_models import SynaraEntity


class FlowchartTemplate(SynaraEntity):
    """A pre-wired flowchart — devices + connections + config + layout.

    definition JSON structure:
    {
        "devices": [{"plugin": "capability_study", "id": "cap1"}, ...],
        "connections": [{"source": "ds1.measurements", "target": "cap1.data"}, ...],
        "config": {"cap1": {"subgroup_size": 5}, ...},
        "positions": {"cap1": {"x": 300, "y": 0}, ...},
    }

    devices_used: denormalized index for queryability.
    Clone = copy the definition blob + set parent_template.
    Share = is_shared flag + shareable link.
    """

    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    definition = models.JSONField(default=dict)
    devices_used = models.JSONField(
        default=list,
        help_text="List of plugin names for query index.",
    )
    is_shared = models.BooleanField(default=False)
    parent_template = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="children",
    )

    class Meta:
        db_table = "flowchart_template"
        ordering = ["-created_at"]

    class SynaraMeta:
        event_domain = "flowchart"
        emit_events = ["created", "updated"]

    def __str__(self):
        return f"Template: {self.name}"

    def clone(self, new_name: str, actor_tenant_id=None):
        """Clone this template. Returns unsaved instance."""
        return FlowchartTemplate(
            name=new_name,
            description=self.description,
            definition=self.definition.copy(),
            devices_used=self.devices_used.copy(),
            parent_template=self,
            tenant_id=actor_tenant_id or self.tenant_id,
        )
```

- [ ] **Step 4: Create and run migration**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 manage.py makemigrations flowchart && python3 manage.py migrate flowchart
```

- [ ] **Step 5: Run tests**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest flowchart/tests/test_models.py -v --no-header 2>&1 | tail -10
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd ~/kjerne && git add flowchart/ svend/settings.py && git commit -m "feat(flowchart): add FlowchartTemplate model"
```

---

### Task 3: FlowchartInstance model

**Files:**
- Modify: `flowchart/models.py`
- Modify: `flowchart/tests/test_models.py`

- [ ] **Step 1: Write the failing test**

Append to `flowchart/tests/test_models.py`:

```python
from flowchart.models import FlowchartInstance


class TestFlowchartInstance(TestCase):
    def test_create_from_template(self):
        tpl = FlowchartTemplate.objects.create(
            name="Quick Cpk",
            definition={
                "devices": [{"plugin": "capability_study", "id": "cap1"}],
                "connections": [],
            },
            devices_used=["capability_study"],
        )
        inst = FlowchartInstance.objects.create(
            name="My Cpk Run",
            template=tpl,
            definition=tpl.definition.copy(),
        )
        assert inst.template_id == tpl.id
        assert inst.is_scratch is False
        assert inst.definition == tpl.definition

    def test_scratch_instance(self):
        inst = FlowchartInstance.objects.create(
            name="Exploring",
            definition={"devices": [], "connections": []},
            is_scratch=True,
        )
        assert inst.is_scratch is True
        assert inst.template is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest flowchart/tests/test_models.py::TestFlowchartInstance -v --no-header 2>&1 | tail -5
```

Expected: ImportError — `FlowchartInstance` doesn't exist yet.

- [ ] **Step 3: Write the model**

Add to `flowchart/models.py`:

```python
class FlowchartInstance(SynaraEntity):
    """A user's working flowchart — possibly loaded from a template.

    The definition may diverge from the template (user added/removed devices,
    rewired connections). is_scratch mirrors Job.is_scratch — exploratory work
    that hasn't been promoted to persistent.
    """

    name = models.CharField(max_length=255)
    template = models.ForeignKey(
        FlowchartTemplate,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="instances",
    )
    definition = models.JSONField(default=dict)
    is_scratch = models.BooleanField(default=False)

    class Meta:
        db_table = "flowchart_instance"
        ordering = ["-created_at"]

    class SynaraMeta:
        event_domain = "flowchart"
        emit_events = ["created", "updated"]

    def __str__(self):
        return f"Flowchart: {self.name}"
```

- [ ] **Step 4: Create and run migration**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 manage.py makemigrations flowchart && python3 manage.py migrate flowchart
```

- [ ] **Step 5: Run tests**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest flowchart/tests/test_models.py -v --no-header 2>&1 | tail -10
```

Expected: 5 tests PASS (3 template + 2 instance).

- [ ] **Step 6: Commit**

```bash
cd ~/kjerne && git add flowchart/ && git commit -m "feat(flowchart): add FlowchartInstance model"
```

---

### Task 4: Type validation + connection validation utilities

**Files:**
- Create: `flowchart/types.py`
- Create: `flowchart/tests/test_types.py`

- [ ] **Step 1: Write the failing tests**

Create `flowchart/tests/test_types.py`:

```python
"""Tests for semantic type system and connection validation."""

from django.test import TestCase

from flowchart.types import SemanticType, validate_connection


class TestSemanticType(TestCase):
    def test_parse(self):
        st = SemanticType.parse("metric:cpk")
        assert st.category == "metric"
        assert st.subtype == "cpk"

    def test_exact_match(self):
        a = SemanticType.parse("metric:cpk")
        b = SemanticType.parse("metric:cpk")
        assert a.accepts(b)

    def test_wildcard_accepts(self):
        wild = SemanticType.parse("metric:*")
        specific = SemanticType.parse("metric:cpk")
        assert wild.accepts(specific)

    def test_raj_test_rejects(self):
        """metric:cpk must reject metric:p_value — methodology enforcement."""
        cpk = SemanticType.parse("metric:cpk")
        pval = SemanticType.parse("metric:p_value")
        assert not cpk.accepts(pval)

    def test_cross_category_rejects(self):
        metric = SemanticType.parse("metric:cpk")
        data = SemanticType.parse("data:column")
        assert not data.accepts(metric)

    def test_spec_not_metric(self):
        wild = SemanticType.parse("metric:*")
        spec = SemanticType.parse("spec:usl")
        assert not wild.accepts(spec)

    def test_invalid_parse_raises(self):
        import pytest
        with pytest.raises(ValueError):
            SemanticType.parse("cpk")
        with pytest.raises(ValueError):
            SemanticType.parse("")


class TestValidateConnection(TestCase):
    def test_valid_connection(self):
        result = validate_connection(
            source_type="metric:cpk",
            target_type="metric:*",
        )
        assert result["valid"] is True

    def test_invalid_connection(self):
        result = validate_connection(
            source_type="metric:cpk",
            target_type="data:column",
        )
        assert result["valid"] is False
        assert "mismatch" in result["error"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest flowchart/tests/test_types.py -v --no-header 2>&1 | tail -5
```

Expected: ImportError.

- [ ] **Step 3: Write the type system**

Create `flowchart/types.py`:

```python
"""Semantic type system for flowchart port connections.

Ported from sandbox/semantic_types.py (54/54 tests).
This is the production version — used by the execution engine
and the connection validation endpoint.

8 categories: metric, spec, config, data, chart, text, list, document.
Domain-specific categories (vsm:*, hoshin:*) extend naturally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

_TYPE_PATTERN = re.compile(r'^([a-z_]+):([a-z_*]+(?:\[\])?)$')


@dataclass(frozen=True)
class SemanticType:
    """A parsed semantic type like 'metric:cpk' or 'data:column[]'."""

    category: str
    subtype: str
    is_array: bool = False

    @classmethod
    def parse(cls, type_str: str) -> SemanticType:
        m = _TYPE_PATTERN.match(type_str)
        if not m:
            raise ValueError(
                f"Invalid semantic type '{type_str}'. "
                f"Expected 'category:subtype' (e.g. 'metric:cpk')"
            )
        category, subtype = m.group(1), m.group(2)
        is_array = subtype.endswith('[]')
        if is_array:
            subtype = subtype[:-2]
        return cls(category=category, subtype=subtype, is_array=is_array)

    @property
    def is_wildcard(self) -> bool:
        return self.subtype == '*'

    def accepts(self, other: SemanticType) -> bool:
        if self.category != other.category:
            return False
        if self.is_array != other.is_array:
            return False
        if self.is_wildcard:
            return True
        return self.subtype == other.subtype

    def __str__(self) -> str:
        arr = '[]' if self.is_array else ''
        return f"{self.category}:{self.subtype}{arr}"


def validate_connection(source_type: str, target_type: str) -> Dict:
    """Validate that a source output type is compatible with a target input type.

    Returns dict with 'valid' (bool) and 'error' (str, empty if valid).
    """
    try:
        source = SemanticType.parse(source_type)
        target = SemanticType.parse(target_type)
    except ValueError as e:
        return {"valid": False, "error": str(e)}

    if not target.accepts(source):
        return {
            "valid": False,
            "error": f"Type mismatch: {source} -> {target}",
        }

    return {"valid": True, "error": ""}
```

- [ ] **Step 4: Run tests**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest flowchart/tests/test_types.py -v --no-header 2>&1 | tail -15
```

Expected: 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/kjerne && git add flowchart/types.py flowchart/tests/test_types.py && git commit -m "feat(flowchart): add semantic type system"
```

---

### Task 5: Flowchart execution engine

**Files:**
- Create: `flowchart/engine.py`
- Create: `flowchart/tests/test_engine.py`

- [ ] **Step 1: Write the failing tests**

Create `flowchart/tests/test_engine.py`:

```python
"""Tests for flowchart execution engine."""

from django.test import TestCase
from pydantic import BaseModel

from syn.plugins.base import Plugin, PluginOutput
from syn.plugins.registry import PluginRegistry
from flowchart.engine import execute_flowchart
from job.models import Job


class _SourceInput(BaseModel):
    value: float

class _SourcePlugin(Plugin):
    name = "test_source"
    version = "1.0.0"
    description = "Outputs a metric"
    input_schema = _SourceInput

    def execute(self, validated_input, context):
        return [
            PluginOutput("result", "metric", validated_input["value"]),
        ]


class _SinkInput(BaseModel):
    value: float

class _SinkPlugin(Plugin):
    name = "test_sink"
    version = "1.0.0"
    description = "Receives a metric"
    input_schema = _SinkInput

    def execute(self, validated_input, context):
        doubled = validated_input["value"] * 2
        return [
            PluginOutput("doubled", "metric", doubled),
        ]


class TestExecuteFlowchart(TestCase):
    def setUp(self):
        self.registry = PluginRegistry()
        self.registry.register(_SourcePlugin)
        self.registry.register(_SinkPlugin)

    def test_two_device_chain(self):
        """Source → Sink: source produces metric, sink consumes it."""
        definition = {
            "devices": [
                {"plugin": "test_source", "id": "src"},
                {"plugin": "test_sink", "id": "snk"},
            ],
            "connections": [
                {"source": "src.result", "target": "snk.value"},
            ],
            "config": {
                "src": {"value": 5.0},
            },
        }

        results = execute_flowchart(
            definition=definition,
            actor="test@svend.ai",
            registry=self.registry,
        )

        assert "src" in results
        assert "snk" in results
        assert results["src"]["job"].status == "completed"
        assert results["snk"]["job"].status == "completed"

        # Source produced 5.0, sink doubled it to 10.0
        snk_outputs = list(results["snk"]["job"].outputs.all())
        doubled = next(o for o in snk_outputs if o.output_key == "doubled")
        assert doubled.value_numeric == 10.0

    def test_topological_order(self):
        """Devices execute in dependency order regardless of definition order."""
        definition = {
            "devices": [
                {"plugin": "test_sink", "id": "snk"},  # Listed first
                {"plugin": "test_source", "id": "src"},  # Listed second
            ],
            "connections": [
                {"source": "src.result", "target": "snk.value"},
            ],
            "config": {
                "src": {"value": 3.0},
            },
        }

        results = execute_flowchart(
            definition=definition,
            actor="test@svend.ai",
            registry=self.registry,
        )

        snk_outputs = list(results["snk"]["job"].outputs.all())
        doubled = next(o for o in snk_outputs if o.output_key == "doubled")
        assert doubled.value_numeric == 6.0

    def test_unconnected_inputs_use_config(self):
        """Device config provides values for unconnected ports."""
        definition = {
            "devices": [
                {"plugin": "test_source", "id": "src"},
            ],
            "connections": [],
            "config": {
                "src": {"value": 42.0},
            },
        }

        results = execute_flowchart(
            definition=definition,
            actor="test@svend.ai",
            registry=self.registry,
        )

        src_outputs = list(results["src"]["job"].outputs.all())
        result = next(o for o in src_outputs if o.output_key == "result")
        assert result.value_numeric == 42.0

    def test_creates_jobs_per_device(self):
        """Each device execution creates a separate Job."""
        definition = {
            "devices": [
                {"plugin": "test_source", "id": "src"},
                {"plugin": "test_sink", "id": "snk"},
            ],
            "connections": [
                {"source": "src.result", "target": "snk.value"},
            ],
            "config": {
                "src": {"value": 1.0},
            },
        }

        execute_flowchart(
            definition=definition,
            actor="test@svend.ai",
            registry=self.registry,
        )

        jobs = Job.objects.filter(actor="test@svend.ai").order_by("created_at")
        assert jobs.count() == 2
        assert jobs[0].plugin_name == "test_source"
        assert jobs[1].plugin_name == "test_sink"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest flowchart/tests/test_engine.py -v --no-header 2>&1 | tail -5
```

Expected: ImportError — `execute_flowchart` doesn't exist.

- [ ] **Step 3: Write the execution engine**

Create `flowchart/engine.py`:

```python
"""Flowchart execution engine — topological sort + port routing.

Ported from sandbox/vsm_full_loop.py run_flowchart().
This is the production version that creates real Jobs via run_plugin().

Usage:
    results = execute_flowchart(definition, actor="user@svend.ai")
    # results["device_id"]["job"] = Job instance
    # results["device_id"]["outputs"] = {port_name: value}
"""

import logging
from typing import Any, Dict, List, Optional

from syn.plugins.registry import PluginRegistry, get_registry
from syn.plugins.runner import run_plugin

logger = logging.getLogger(__name__)


def execute_flowchart(
    definition: Dict,
    actor: str,
    *,
    tenant_id=None,
    is_scratch: bool = False,
    registry: Optional[PluginRegistry] = None,
    flowchart_instance_id=None,
) -> Dict[str, Dict]:
    """Execute a flowchart definition — run all devices in topological order.

    Args:
        definition: {devices: [...], connections: [...], config: {...}}
        actor: Who triggered this execution.
        tenant_id: Optional tenant UUID.
        is_scratch: Mark all Jobs as scratch.
        registry: PluginRegistry (defaults to global singleton).
        flowchart_instance_id: UUID of the FlowchartInstance (for Job.canvas_id).

    Returns:
        Dict mapping device_id -> {"job": Job, "outputs": {port: value}}
    """
    reg = registry or get_registry()
    devices = definition.get("devices", [])
    connections = definition.get("connections", [])
    config = definition.get("config", {})

    # Build device lookup: id -> plugin name
    device_plugins = {}
    for dev in devices:
        device_plugins[dev["id"]] = dev["plugin"]

    # Build adjacency list and in-degree for topological sort
    adj: Dict[str, List[tuple]] = {dev["id"]: [] for dev in devices}
    in_deg: Dict[str, int] = {dev["id"]: 0 for dev in devices}

    # Parse connections: "src.port" -> (device_id, port_name)
    parsed_connections = []
    for conn in connections:
        src_device, src_port = conn["source"].split(".", 1)
        tgt_device, tgt_port = conn["target"].split(".", 1)
        parsed_connections.append((src_device, src_port, tgt_device, tgt_port))

        if tgt_device not in [t for t, _ in adj.get(src_device, [])]:
            adj[src_device].append((tgt_device, None))
            in_deg[tgt_device] = in_deg.get(tgt_device, 0) + 1

    # Kahn's topological sort
    queue = [d for d in device_plugins if in_deg.get(d, 0) == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor, _ in adj.get(node, []):
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(device_plugins):
        cycled = [d for d in device_plugins if in_deg.get(d, 0) > 0]
        raise ValueError(f"Cycle detected in flowchart involving: {cycled}")

    # Execute in topological order
    results: Dict[str, Dict] = {}
    # Track port values for routing: device_id -> {port_name: value}
    port_values: Dict[str, Dict[str, Any]] = {}

    for device_id in order:
        plugin_name = device_plugins[device_id]

        # Build input: start with config, overlay routed values
        input_data = dict(config.get(device_id, {}))

        # Collect routed values from upstream connections
        for src_dev, src_port, tgt_dev, tgt_port in parsed_connections:
            if tgt_dev == device_id and src_dev in port_values:
                routed_value = port_values[src_dev].get(src_port)
                if routed_value is not None:
                    input_data[tgt_port] = routed_value

        # Run the plugin
        job = run_plugin(
            plugin_name,
            input_data,
            actor=actor,
            tenant_id=tenant_id,
            canvas_id=flowchart_instance_id,
            is_scratch=is_scratch,
            registry=reg,
        )

        # Extract output values for downstream routing
        outputs = {}
        for out in job.outputs.all():
            if out.output_type == "metric":
                outputs[out.output_key] = out.value_numeric
            else:
                outputs[out.output_key] = out.value_json

        port_values[device_id] = outputs
        results[device_id] = {"job": job, "outputs": outputs}

    return results
```

- [ ] **Step 4: Run tests**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest flowchart/tests/test_engine.py -v --no-header 2>&1 | tail -10
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/kjerne && git add flowchart/engine.py flowchart/tests/test_engine.py && git commit -m "feat(flowchart): add execution engine with topological sort"
```

---

### Task 6: API endpoints (run flowchart, list templates)

**Files:**
- Create: `flowchart/views.py`
- Modify: `svend/urls.py`
- Create: `flowchart/tests/test_views.py`

- [ ] **Step 1: Write the failing tests**

Create `flowchart/tests/test_views.py`:

```python
"""Tests for flowchart API endpoints."""

import json

from django.test import TestCase
from pydantic import BaseModel

from conftest import SECURE_OFF, make_user
from flowchart.models import FlowchartTemplate, FlowchartInstance
from syn.plugins.base import Plugin, PluginOutput
from syn.plugins.registry import PluginRegistry, get_registry
from job.models import Job


class _TestInput(BaseModel):
    value: float

class _TestPlugin(Plugin):
    name = "flowchart_test_plugin"
    version = "1.0.0"
    description = "Test plugin for flowchart views"
    input_schema = _TestInput

    def execute(self, validated_input, context):
        return [
            PluginOutput("result", "metric", validated_input["value"] * 10),
        ]


@SECURE_OFF
class TestFlowchartRunEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("flow@test.com")
        self.client.login(username="flow", password="testpass123!")
        reg = get_registry()
        if not reg.has("flowchart_test_plugin"):
            reg.register(_TestPlugin)

    def test_run_inline_definition(self):
        """POST with inline definition (no saved instance)."""
        resp = self.client.post(
            "/api/flowchart/run/",
            data=json.dumps({
                "definition": {
                    "devices": [
                        {"plugin": "flowchart_test_plugin", "id": "d1"},
                    ],
                    "connections": [],
                    "config": {"d1": {"value": 7.0}},
                },
            }),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        assert "d1" in body["results"]
        assert body["results"]["d1"]["status"] == "completed"
        assert body["results"]["d1"]["outputs"]["result"] == 70.0

    def test_run_requires_auth(self):
        self.client.logout()
        resp = self.client.post(
            "/api/flowchart/run/",
            data=json.dumps({"definition": {"devices": [], "connections": []}}),
            content_type="application/json",
        )
        assert resp.status_code in (401, 403)


@SECURE_OFF
class TestTemplateListEndpoint(TestCase):
    def setUp(self):
        self.user = make_user("flow@test.com")
        self.client.login(username="flow", password="testpass123!")

    def test_list_templates(self):
        FlowchartTemplate.objects.create(
            name="Quick Cpk",
            definition={"devices": [], "connections": []},
            devices_used=["capability_study"],
        )
        FlowchartTemplate.objects.create(
            name="PPAP",
            definition={"devices": [], "connections": []},
            devices_used=["capability_study", "control_chart"],
        )
        resp = self.client.get("/api/flowchart/templates/")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        assert len(body["templates"]) == 2
        names = [t["name"] for t in body["templates"]]
        assert "Quick Cpk" in names
        assert "PPAP" in names
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest flowchart/tests/test_views.py -v --no-header 2>&1 | tail -5
```

Expected: 404 — URL not wired.

- [ ] **Step 3: Write the views**

Create `flowchart/views.py`:

```python
"""Flowchart API endpoints.

POST /api/flowchart/run/     — execute a flowchart definition
GET  /api/flowchart/templates/ — list available templates
"""

import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET

from accounts.permissions import require_auth
from flowchart.engine import execute_flowchart
from flowchart.models import FlowchartTemplate

logger = logging.getLogger(__name__)


@csrf_exempt
@require_auth
@require_POST
def flowchart_run(request):
    """Execute a flowchart and return results per device."""
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    definition = body.get("definition")
    if not definition:
        return JsonResponse({"error": "definition is required"}, status=400)

    is_scratch = body.get("is_scratch", False)

    try:
        results = execute_flowchart(
            definition=definition,
            actor=request.user.email,
            tenant_id=getattr(request.user, "tenant_id", None),
            is_scratch=is_scratch,
        )
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception(f"[FLOWCHART] Execution failed: {e}")
        return JsonResponse({"error": str(e)}, status=500)

    # Serialize results
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


@csrf_exempt
@require_auth
@require_GET
def flowchart_templates(request):
    """List available flowchart templates."""
    templates = FlowchartTemplate.objects.filter(
        is_deleted=False,
    ).order_by("name")

    # Filter by tenant if applicable
    tenant_id = getattr(request.user, "tenant_id", None)
    if tenant_id:
        templates = templates.filter(
            models.Q(tenant_id=tenant_id) | models.Q(tenant_id__isnull=True)
        )

    result = []
    for tpl in templates:
        result.append({
            "id": str(tpl.id),
            "name": tpl.name,
            "description": tpl.description,
            "devices_used": tpl.devices_used,
            "is_shared": tpl.is_shared,
            "created_at": tpl.created_at.isoformat() if tpl.created_at else None,
        })

    return JsonResponse({"templates": result})
```

- [ ] **Step 4: Wire URLs**

Add to `svend/urls.py`, near the other API paths (after the `demo_canvas_run` line):

```python
    # Flowchart API
    path("api/flowchart/run/", __import__("flowchart.views", fromlist=["flowchart_run"]).flowchart_run, name="flowchart_run"),
    path("api/flowchart/templates/", __import__("flowchart.views", fromlist=["flowchart_templates"]).flowchart_templates, name="flowchart_templates"),
```

- [ ] **Step 5: Fix the tenant filter import**

The `flowchart_templates` view references `models.Q` without importing it. Add at top of `flowchart/views.py`:

```python
from django.db import models
```

- [ ] **Step 6: Run tests**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest flowchart/tests/test_views.py -v --no-header 2>&1 | tail -10
```

Expected: 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
cd ~/kjerne && git add flowchart/views.py flowchart/tests/test_views.py svend/urls.py && git commit -m "feat(flowchart): add run + templates API endpoints"
```

---

### Task 7: Run all flowchart tests together

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite for the flowchart app**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest flowchart/ -v --no-header 2>&1 | tail -20
```

Expected: 18 tests PASS (3 template + 2 instance + 9 types + 4 engine).

NOTE: View tests (3) may need the `SECURE_OFF` decorator and `make_user()` pattern from `conftest.py`. If they fail with 301 redirects, ensure `@SECURE_OFF` is on the test class.

- [ ] **Step 2: Run existing plugin tests to verify no regression**

```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/ job/ -v --no-header 2>&1 | tail -15
```

Expected: All existing tests still pass.

- [ ] **Step 3: Final commit**

```bash
cd ~/kjerne && git add -A && git status
```

Review that only flowchart/ files and the two modified files (settings.py, urls.py) are staged. Then:

```bash
git commit -m "feat(flowchart): complete app — models, types, engine, API endpoints"
```

---

## File Map Summary

| File | Purpose |
|------|---------|
| `flowchart/__init__.py` | Package marker |
| `flowchart/apps.py` | Django AppConfig |
| `flowchart/models.py` | FlowchartTemplate + FlowchartInstance (SynaraEntity) |
| `flowchart/types.py` | SemanticType parser + validate_connection() |
| `flowchart/engine.py` | execute_flowchart() — topological sort + port routing |
| `flowchart/views.py` | POST /api/flowchart/run/ + GET /api/flowchart/templates/ |
| `flowchart/tests/test_models.py` | Model creation, cloning, query tests |
| `flowchart/tests/test_types.py` | Type parsing, compatibility, Raj test |
| `flowchart/tests/test_engine.py` | Execution order, port routing, job creation |
| `flowchart/tests/test_views.py` | API endpoint integration tests |

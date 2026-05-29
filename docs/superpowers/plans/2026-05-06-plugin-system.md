# Plugin System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SVEND 2.0 plugin framework — a uniform execution contract that wraps all analysis engines (200+ DSW, calculators, simulators) with typed I/O schemas, silent Job creation, and bus event emission.

**Architecture:** Plugin ABC in syn/plugins/ (pure Python, part of Synara platform). PluginRegistry singleton for discovery. Runner function handles the full lifecycle: validate input → execute → create Job + JobOutputs → emit bus event. Plugin implementations are Django app `plugins/` at repo root. First plugin: capability study wrapping existing `agents_api/analysis/spc/capability.py`.

**Tech Stack:** Pydantic V2 (via syn.io.InputSchema/OutputSchema), Django ORM (Job/JobOutput models), syn.bus (event emission), existing scipy/numpy computation.

---

### Task 1: Plugin ABC + PluginOutput

**Files:**
- Create: `syn/plugins/__init__.py`
- Create: `syn/plugins/base.py`
- Create: `plugins/__init__.py`
- Create: `plugins/tests/__init__.py`
- Create: `plugins/tests/test_base.py`

- [ ] **Step 1: Write failing tests for Plugin ABC contract**

```python
# plugins/tests/test_base.py
"""Tests for the Plugin ABC contract."""

import pytest
from syn.plugins.base import Plugin, PluginOutput


class TestPluginOutput:
    def test_metric_output(self):
        out = PluginOutput(key="cpk", output_type="metric", value=1.45)
        assert out.key == "cpk"
        assert out.output_type == "metric"
        assert out.value == 1.45
        assert out.provenance == "calculated"
        assert out.measure_slug is None

    def test_chart_output_with_measure_slug(self):
        out = PluginOutput(
            key="histogram",
            output_type="chart",
            value={"data": [], "layout": {}},
            provenance="calculated",
            measure_slug="bore_cpk",
        )
        assert out.output_type == "chart"
        assert out.measure_slug == "bore_cpk"

    def test_simulated_provenance(self):
        out = PluginOutput(key="projected_cpk", output_type="metric", value=1.8, provenance="simulated")
        assert out.provenance == "simulated"


class TestPluginABC:
    def test_cannot_instantiate_without_name(self):
        class BadPlugin(Plugin):
            description = "Missing name"
            input_schema = None

            def execute(self, validated_input, context):
                return []

        with pytest.raises(TypeError):
            BadPlugin()

    def test_cannot_instantiate_without_execute(self):
        """Plugin is abstract — must implement execute()."""
        with pytest.raises(TypeError):
            Plugin()

    def test_valid_plugin_instantiates(self):
        from pydantic import BaseModel

        class DummyInput(BaseModel):
            x: float

        class DummyPlugin(Plugin):
            name = "dummy"
            version = "1.0.0"
            description = "A test plugin"
            input_schema = DummyInput

            def execute(self, validated_input, context):
                return [PluginOutput("result", "metric", validated_input["x"] * 2)]

        p = DummyPlugin()
        assert p.name == "dummy"
        assert p.version == "1.0.0"

    def test_get_metadata(self):
        from pydantic import BaseModel

        class MetaInput(BaseModel):
            value: float
            label: str = "default"

        class MetaPlugin(Plugin):
            name = "meta_test"
            version = "2.1.0"
            description = "Metadata test"
            input_schema = MetaInput

            def execute(self, validated_input, context):
                return []

        meta = MetaPlugin.get_metadata()
        assert meta["name"] == "meta_test"
        assert meta["version"] == "2.1.0"
        assert meta["description"] == "Metadata test"
        assert "properties" in meta["input_schema"]
        assert "value" in meta["input_schema"]["properties"]

    def test_execute_returns_outputs(self):
        from pydantic import BaseModel

        class AddInput(BaseModel):
            a: float
            b: float

        class AddPlugin(Plugin):
            name = "add"
            version = "1.0.0"
            description = "Add two numbers"
            input_schema = AddInput

            def execute(self, validated_input, context):
                total = validated_input["a"] + validated_input["b"]
                return [PluginOutput("sum", "metric", total)]

        p = AddPlugin()
        outputs = p.execute({"a": 3.0, "b": 4.0}, {})
        assert len(outputs) == 1
        assert outputs[0].value == 7.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'syn.plugins'`

- [ ] **Step 3: Implement Plugin ABC and PluginOutput**

```python
# syn/plugins/__init__.py
"""
Synara Plugin Framework
=======================

Uniform execution contract for all SVEND analysis engines.

Usage:
    from syn.plugins import Plugin, PluginOutput, get_registry, run_plugin
"""

from syn.plugins.base import Plugin, PluginOutput

__all__ = ["Plugin", "PluginOutput"]
```

```python
# syn/plugins/base.py
"""
Plugin ABC — the execution contract for all SVEND tools.

Every plugin declares:
- name: unique identifier (used in registry, Job records, canvas bindings)
- input_schema: Pydantic model for what it accepts
- execute(): computation → list of PluginOutput

Design:
- Sync only (Django is sync, plugins are user-triggered)
- No retry logic (user re-runs on failure)
- No telemetry infra (Job IS the telemetry)
- Schema introspection via get_metadata() for canvas UI generation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


@dataclass
class PluginOutput:
    """A named output from a plugin execution.

    Attributes:
        key: Output identifier (e.g. "cpk", "histogram", "summary").
        output_type: One of metric, chart, table, text, dataset.
        value: The payload — float for metrics, dict for everything else.
        provenance: observed | calculated | simulated | projected.
        measure_slug: If this output should write to PCL, which measure.
    """

    key: str
    output_type: str  # metric, chart, table, text, dataset
    value: Any
    provenance: str = "calculated"
    measure_slug: Optional[str] = None


class Plugin(ABC):
    """Abstract base for all SVEND plugins.

    Subclasses MUST define:
        - name (str): unique plugin identifier
        - description (str): human-readable purpose
        - input_schema (Pydantic BaseModel subclass): what the plugin accepts
        - execute(): the computation

    Optional:
        - version (str): defaults to "1.0.0"
        - output_schema: for documentation/introspection only
    """

    name: str = None
    version: str = "1.0.0"
    description: str = None
    input_schema: Type[BaseModel] = None
    output_schema: Optional[Type[BaseModel]] = None

    def __init__(self):
        if not self.name:
            raise TypeError(f"{self.__class__.__name__} must define 'name'")
        if not self.description:
            raise TypeError(f"{self.__class__.__name__} must define 'description'")

    @abstractmethod
    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        """Run the plugin computation.

        Args:
            validated_input: Dict from input_schema.model_dump() — already validated.
            context: Execution context (job_id, actor, tenant_id).

        Returns:
            List of PluginOutput objects.
        """
        ...

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Introspectable metadata for registry, canvas UI, documentation."""
        return {
            "name": cls.name,
            "version": cls.version,
            "description": cls.description,
            "input_schema": cls.input_schema.model_json_schema() if cls.input_schema else None,
            "output_schema": cls.output_schema.model_json_schema() if cls.output_schema else None,
        }

    def __repr__(self):
        return f"<Plugin: {self.name} v{self.version}>"
```

```python
# plugins/__init__.py
"""SVEND Plugin implementations."""
```

```python
# plugins/tests/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_base.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
cd ~/kjerne && git add syn/plugins/__init__.py syn/plugins/base.py plugins/__init__.py plugins/tests/__init__.py plugins/tests/test_base.py
git commit -m "feat(plugins): Plugin ABC and PluginOutput dataclass

Synara plugin framework — uniform execution contract for all SVEND
analysis engines. Plugin declares name, input_schema, execute().
PluginOutput carries typed results with provenance for PCL writes."
```

---

### Task 2: PluginRegistry

**Files:**
- Create: `syn/plugins/registry.py`
- Create: `plugins/tests/test_registry.py`
- Modify: `syn/plugins/__init__.py`

- [ ] **Step 1: Write failing tests for PluginRegistry**

```python
# plugins/tests/test_registry.py
"""Tests for the PluginRegistry."""

import pytest
from pydantic import BaseModel

from syn.plugins.base import Plugin, PluginOutput
from syn.plugins.registry import PluginRegistry


class _TestInput(BaseModel):
    x: float


class _AlphaPlugin(Plugin):
    name = "alpha"
    version = "1.0.0"
    description = "Alpha plugin"
    input_schema = _TestInput

    def execute(self, validated_input, context):
        return [PluginOutput("out", "metric", validated_input["x"])]


class _BetaPlugin(Plugin):
    name = "beta"
    version = "2.0.0"
    description = "Beta plugin"
    input_schema = _TestInput

    def execute(self, validated_input, context):
        return []


class TestPluginRegistry:
    def setup_method(self):
        self.registry = PluginRegistry()

    def test_register_and_get(self):
        self.registry.register(_AlphaPlugin)
        plugin = self.registry.get("alpha")
        assert isinstance(plugin, _AlphaPlugin)
        assert plugin.name == "alpha"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="No plugin registered"):
            self.registry.get("nonexistent")

    def test_list_plugins(self):
        self.registry.register(_AlphaPlugin)
        self.registry.register(_BetaPlugin)
        listing = self.registry.list_plugins()
        assert len(listing) == 2
        names = [p["name"] for p in listing]
        assert "alpha" in names
        assert "beta" in names

    def test_duplicate_registration_raises(self):
        self.registry.register(_AlphaPlugin)
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(_AlphaPlugin)

    def test_has(self):
        self.registry.register(_AlphaPlugin)
        assert self.registry.has("alpha") is True
        assert self.registry.has("nope") is False

    def test_get_metadata(self):
        self.registry.register(_AlphaPlugin)
        meta = self.registry.get_metadata("alpha")
        assert meta["name"] == "alpha"
        assert meta["version"] == "1.0.0"
        assert "properties" in meta["input_schema"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'syn.plugins.registry'`

- [ ] **Step 3: Implement PluginRegistry**

```python
# syn/plugins/registry.py
"""
Plugin Registry — discovery and access for all registered plugins.

Simple dict-based registry. Plugins register at Django startup (apps.py ready()).
Canvas models reference plugins by name string → registry.get(name).
"""

import logging
from typing import Any, Dict, List, Optional, Type

from syn.plugins.base import Plugin

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry of available plugins. One instance per process."""

    def __init__(self):
        self._plugins: Dict[str, Plugin] = {}

    def register(self, plugin_class: Type[Plugin]) -> None:
        """Register a plugin class. Instantiates it and stores by name."""
        instance = plugin_class()
        name = instance.name
        if name in self._plugins:
            raise ValueError(f"Plugin '{name}' already registered")
        self._plugins[name] = instance
        logger.info(f"[PLUGINS] Registered: {name} v{instance.version}")

    def get(self, name: str) -> Plugin:
        """Get a registered plugin instance by name."""
        if name not in self._plugins:
            raise KeyError(f"No plugin registered with name '{name}'")
        return self._plugins[name]

    def has(self, name: str) -> bool:
        """Check if a plugin is registered."""
        return name in self._plugins

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get plugin metadata by name."""
        return self.get(name).get_metadata()

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins with metadata."""
        return [p.get_metadata() for p in self._plugins.values()]

    @property
    def count(self) -> int:
        return len(self._plugins)


# Module-level singleton
_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry
```

Update `syn/plugins/__init__.py`:

```python
# syn/plugins/__init__.py
"""
Synara Plugin Framework
=======================

Uniform execution contract for all SVEND analysis engines.

Usage:
    from syn.plugins import Plugin, PluginOutput, get_registry
"""

from syn.plugins.base import Plugin, PluginOutput
from syn.plugins.registry import PluginRegistry, get_registry

__all__ = ["Plugin", "PluginOutput", "PluginRegistry", "get_registry"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_registry.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
cd ~/kjerne && git add syn/plugins/registry.py syn/plugins/__init__.py plugins/tests/test_registry.py
git commit -m "feat(plugins): PluginRegistry with register/get/list/has

Dict-based singleton registry. Plugins register by class, accessed by
name string. Canvas models will reference plugins via this registry."
```

---

### Task 3: Add plugin_name to Job model

**Files:**
- Modify: `job/models.py`
- Create: `job/migrations/0003_job_plugin_name.py` (auto-generated)
- Create: `plugins/tests/test_job_plugin_name.py`

- [ ] **Step 1: Write failing test for plugin_name field**

```python
# plugins/tests/test_job_plugin_name.py
"""Test that Job model has plugin_name field."""

import pytest
from django.test import TestCase

from job.models import Job


class TestJobPluginName(TestCase):
    def test_job_has_plugin_name_field(self):
        job = Job(
            plugin_name="capability_study",
            actor="test@svend.ai",
            inputs={"data": [1, 2, 3]},
        )
        assert job.plugin_name == "capability_study"

    def test_plugin_name_nullable(self):
        """Jobs created without plugin (legacy, API-driven) have null plugin_name."""
        job = Job(actor="test@svend.ai", inputs={})
        assert job.plugin_name is None

    def test_plugin_name_in_to_dict(self):
        job = Job(plugin_name="control_chart", actor="test@svend.ai", inputs={})
        d = job.to_dict()
        assert d["plugin_name"] == "control_chart"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_job_plugin_name.py -v`
Expected: FAIL — `Job() got an unexpected keyword argument 'plugin_name'`

- [ ] **Step 3: Add plugin_name field to Job model**

In `job/models.py`, add after the `canvas_id` field:

```python
    plugin_name = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        db_index=True,
        help_text="Which plugin produced this job. Null for legacy/API runs.",
    )
```

Update `to_dict()` to include `"plugin_name": self.plugin_name,`.

- [ ] **Step 4: Generate and apply migration**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 manage.py makemigrations job --name job_plugin_name && python3 manage.py migrate job`
Expected: Migration created and applied successfully.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_job_plugin_name.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Run existing Job tests to confirm no regression**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest job/tests/ -v`
Expected: All existing tests still PASS

- [ ] **Step 7: Commit**

```bash
cd ~/kjerne && git add job/models.py job/migrations/0003_job_plugin_name.py plugins/tests/test_job_plugin_name.py
git commit -m "feat(job): add plugin_name field to Job model

Tracks which plugin produced the job. Nullable for legacy/API runs.
Indexed for efficient lookup by plugin type."
```

---

### Task 4: Plugin Runner

**Files:**
- Create: `syn/plugins/runner.py`
- Create: `plugins/tests/test_runner.py`
- Modify: `syn/plugins/__init__.py`

- [ ] **Step 1: Write failing tests for run_plugin()**

```python
# plugins/tests/test_runner.py
"""Tests for the plugin runner — full execution lifecycle."""

import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from django.test import TestCase
from django.utils import timezone

from syn.plugins.base import Plugin, PluginOutput
from syn.plugins.registry import PluginRegistry
from syn.plugins.runner import run_plugin
from job.models import Job, JobOutput


class _RunnerInput(BaseModel):
    value: float
    multiplier: float = 2.0


class _RunnerPlugin(Plugin):
    name = "test_runner"
    version = "1.0.0"
    description = "Plugin for runner tests"
    input_schema = _RunnerInput

    def execute(self, validated_input, context):
        result = validated_input["value"] * validated_input["multiplier"]
        return [
            PluginOutput("result", "metric", result, measure_slug="test_measure"),
            PluginOutput("summary", "text", {"text": f"Result: {result}"}),
        ]


class TestRunPlugin(TestCase):
    def setUp(self):
        self.registry = PluginRegistry()
        self.registry.register(_RunnerPlugin)

    def test_creates_job_with_correct_fields(self):
        job = run_plugin(
            "test_runner",
            {"value": 5.0, "multiplier": 3.0},
            actor="test@svend.ai",
            registry=self.registry,
        )
        assert job.plugin_name == "test_runner"
        assert job.status == "completed"
        assert job.actor == "test@svend.ai"
        assert job.inputs == {"value": 5.0, "multiplier": 3.0}
        assert job.duration_ms is not None
        assert job.duration_ms >= 0

    def test_creates_job_outputs(self):
        job = run_plugin(
            "test_runner",
            {"value": 5.0},
            actor="test@svend.ai",
            registry=self.registry,
        )
        outputs = list(job.outputs.all())
        assert len(outputs) == 2

        metric_out = next(o for o in outputs if o.output_key == "result")
        assert metric_out.output_type == "metric"
        assert metric_out.value_numeric == 10.0
        assert metric_out.measure_slug == "test_measure"
        assert metric_out.provenance == "calculated"

        text_out = next(o for o in outputs if o.output_key == "summary")
        assert text_out.output_type == "text"
        assert text_out.value_json == {"text": "Result: 10.0"}

    def test_validates_input(self):
        """Invalid input raises ValidationError before execution."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            run_plugin(
                "test_runner",
                {"value": "not_a_number"},
                actor="test@svend.ai",
                registry=self.registry,
            )

    def test_unknown_plugin_raises(self):
        with pytest.raises(KeyError):
            run_plugin(
                "nonexistent",
                {},
                actor="test@svend.ai",
                registry=self.registry,
            )

    def test_emits_bus_event(self):
        with patch("syn.plugins.runner.emit") as mock_emit:
            job = run_plugin(
                "test_runner",
                {"value": 5.0},
                actor="test@svend.ai",
                registry=self.registry,
            )
            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == "plugin.execution.completed"
            assert call_args[0][1]["plugin_name"] == "test_runner"
            assert call_args[0][1]["job_id"] == str(job.id)

    def test_failed_execution_marks_job_failed(self):
        class _FailPlugin(Plugin):
            name = "fail_plugin"
            version = "1.0.0"
            description = "Always fails"
            input_schema = _RunnerInput

            def execute(self, validated_input, context):
                raise ValueError("Something broke")

        fail_registry = PluginRegistry()
        fail_registry.register(_FailPlugin)

        with pytest.raises(ValueError, match="Something broke"):
            run_plugin(
                "fail_plugin",
                {"value": 1.0},
                actor="test@svend.ai",
                registry=fail_registry,
            )

        # Job should exist and be marked failed
        job = Job.objects.filter(plugin_name="fail_plugin").first()
        assert job is not None
        assert job.status == "failed"

    def test_scratch_flag_propagates(self):
        job = run_plugin(
            "test_runner",
            {"value": 1.0},
            actor="test@svend.ai",
            registry=self.registry,
            is_scratch=True,
        )
        assert job.is_scratch is True

    def test_outputs_summary_populated(self):
        job = run_plugin(
            "test_runner",
            {"value": 5.0},
            actor="test@svend.ai",
            registry=self.registry,
        )
        assert job.outputs_summary == {"result": "metric", "summary": "text"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'syn.plugins.runner'`

- [ ] **Step 3: Implement run_plugin()**

```python
# syn/plugins/runner.py
"""
Plugin Runner — the full execution lifecycle.

    validate input → create Job → execute → create JobOutputs → emit event

This is the single entry point for running any plugin. Canvas views,
API endpoints, and CLI all call run_plugin().
"""

import logging
from typing import Any, Dict, Optional

from django.utils import timezone

from syn.plugins.base import Plugin, PluginOutput
from syn.plugins.registry import PluginRegistry, get_registry

logger = logging.getLogger(__name__)


def run_plugin(
    plugin_name: str,
    input_data: Dict[str, Any],
    *,
    actor: str,
    tenant_id=None,
    canvas_id=None,
    is_scratch: bool = False,
    registry: Optional[PluginRegistry] = None,
) -> "Job":
    """Run a plugin through the full lifecycle.

    Args:
        plugin_name: Registered plugin name.
        input_data: Raw input (will be validated against plugin.input_schema).
        actor: Who triggered this (user email or system identifier).
        tenant_id: Tenant UUID (optional for individual users).
        canvas_id: Canvas UUID that triggered this run (optional).
        is_scratch: Mark as scratch/exploratory work.
        registry: PluginRegistry instance (defaults to global singleton).

    Returns:
        Completed Job instance with outputs.

    Raises:
        KeyError: Plugin not found in registry.
        pydantic.ValidationError: Input doesn't match schema.
        Exception: Re-raises plugin execution errors after marking Job failed.
    """
    from job.models import Job, JobOutput

    reg = registry or get_registry()
    plugin = reg.get(plugin_name)

    # Validate input
    validated = plugin.input_schema(**input_data).model_dump()

    # Create Job (status=running)
    job = Job.objects.create(
        plugin_name=plugin_name,
        canvas_id=canvas_id,
        inputs=validated,
        actor=actor,
        tenant_id=tenant_id,
        status="running",
        started_at=timezone.now(),
        is_scratch=is_scratch,
    )

    # Execute
    try:
        context = {
            "job_id": str(job.id),
            "actor": actor,
            "tenant_id": str(tenant_id) if tenant_id else None,
        }
        outputs = plugin.execute(validated, context)
    except Exception:
        # Mark job failed
        job.status = "failed"
        job.completed_at = timezone.now()
        job.duration_ms = int((job.completed_at - job.started_at).total_seconds() * 1000)
        job.save(update_fields=["status", "completed_at", "duration_ms"])
        raise

    # Create JobOutputs
    for out in outputs:
        JobOutput.objects.create(
            job=job,
            output_key=out.key,
            output_type=out.output_type,
            value_numeric=out.value if out.output_type == "metric" else None,
            value_json=out.value if out.output_type != "metric" else {},
            provenance=out.provenance,
            measure_slug=out.measure_slug,
        )

    # Complete job
    job.status = "completed"
    job.completed_at = timezone.now()
    job.duration_ms = int((job.completed_at - job.started_at).total_seconds() * 1000)
    job.outputs_summary = {out.key: out.output_type for out in outputs}
    job.save(update_fields=["status", "completed_at", "duration_ms", "outputs_summary"])

    # Emit bus event
    try:
        from syn.bus import emit

        emit(
            "plugin.execution.completed",
            {
                "job_id": str(job.id),
                "plugin_name": plugin_name,
                "outputs": [out.key for out in outputs],
            },
            actor=actor,
            tenant_id=str(tenant_id) if tenant_id else None,
        )
    except Exception as e:
        logger.warning(f"[PLUGINS] Bus emit failed for {plugin_name}: {e}")

    return job
```

Update `syn/plugins/__init__.py`:

```python
# syn/plugins/__init__.py
"""
Synara Plugin Framework
=======================

Uniform execution contract for all SVEND analysis engines.

Usage:
    from syn.plugins import Plugin, PluginOutput, get_registry, run_plugin
"""

from syn.plugins.base import Plugin, PluginOutput
from syn.plugins.registry import PluginRegistry, get_registry
from syn.plugins.runner import run_plugin

__all__ = ["Plugin", "PluginOutput", "PluginRegistry", "get_registry", "run_plugin"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_runner.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
cd ~/kjerne && git add syn/plugins/runner.py syn/plugins/__init__.py plugins/tests/test_runner.py
git commit -m "feat(plugins): run_plugin() — full execution lifecycle

Validate → Job → execute → JobOutputs → bus event. Handles failures
(marks Job failed), scratch flag, and bus emission. Single entry point
for canvas views, API, and CLI."
```

---

### Task 5: Capability Study Plugin

**Files:**
- Create: `plugins/capability.py`
- Create: `plugins/tests/test_capability.py`

- [ ] **Step 1: Write failing tests for CapabilityStudyPlugin**

```python
# plugins/tests/test_capability.py
"""Tests for the Capability Study plugin."""

import pytest
import numpy as np
from django.test import TestCase

from syn.plugins.base import PluginOutput
from syn.plugins.registry import PluginRegistry
from syn.plugins.runner import run_plugin
from plugins.capability import CapabilityStudyPlugin
from job.models import Job, JobOutput


class TestCapabilityStudyPlugin:
    def test_metadata(self):
        meta = CapabilityStudyPlugin.get_metadata()
        assert meta["name"] == "capability_study"
        assert "data" in meta["input_schema"]["properties"]
        assert "usl" in meta["input_schema"]["properties"]
        assert "lsl" in meta["input_schema"]["properties"]

    def test_execute_with_both_specs(self):
        plugin = CapabilityStudyPlugin()
        np.random.seed(42)
        data = np.random.normal(50, 2, 100).tolist()

        outputs = plugin.execute(
            {"data": data, "usl": 56.0, "lsl": 44.0},
            {"job_id": "test", "actor": "test"},
        )

        # Should have metric outputs for cpk, ppk
        keys = [o.key for o in outputs]
        assert "cpk" in keys
        assert "ppk" in keys

        # Cpk should be reasonable for N(50,2) with specs at ±6
        cpk_out = next(o for o in outputs if o.key == "cpk")
        assert cpk_out.output_type == "metric"
        assert cpk_out.provenance == "calculated"
        assert cpk_out.value > 0.8  # Should be ~1.0 for this data

        # Should have chart outputs
        chart_outputs = [o for o in outputs if o.output_type == "chart"]
        assert len(chart_outputs) >= 1  # At least histogram

        # Should have summary text
        text_outputs = [o for o in outputs if o.output_type == "text"]
        assert len(text_outputs) >= 1

    def test_execute_one_sided_usl_only(self):
        plugin = CapabilityStudyPlugin()
        data = np.random.normal(10, 1, 50).tolist()

        outputs = plugin.execute(
            {"data": data, "usl": 14.0},
            {"job_id": "test", "actor": "test"},
        )

        keys = [o.key for o in outputs]
        assert "cpk" in keys
        # No Cp/Pp for one-sided
        assert "cp" not in keys

    def test_execute_no_specs(self):
        """Without spec limits, no capability indices — just descriptive stats."""
        plugin = CapabilityStudyPlugin()
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        outputs = plugin.execute(
            {"data": data},
            {"job_id": "test", "actor": "test"},
        )

        keys = [o.key for o in outputs]
        assert "cpk" not in keys
        # Should still get charts and summary
        assert "summary" in keys


class TestCapabilityStudyIntegration(TestCase):
    """Integration test — run through full runner with DB."""

    def test_full_lifecycle(self):
        registry = PluginRegistry()
        registry.register(CapabilityStudyPlugin)

        np.random.seed(42)
        data = np.random.normal(50, 2, 100).tolist()

        job = run_plugin(
            "capability_study",
            {"data": data, "usl": 56.0, "lsl": 44.0, "target": 50.0},
            actor="test@svend.ai",
            registry=registry,
        )

        assert job.status == "completed"
        assert job.plugin_name == "capability_study"

        outputs = list(job.outputs.all())
        assert len(outputs) >= 4  # cpk, ppk, at least 1 chart, summary

        # Check PCL-writable metric
        cpk_output = job.outputs.filter(output_key="cpk").first()
        assert cpk_output is not None
        assert cpk_output.value_numeric is not None
        assert cpk_output.measure_slug == "cpk"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_capability.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plugins.capability'`

- [ ] **Step 3: Implement CapabilityStudyPlugin**

```python
# plugins/capability.py
"""Capability Study Plugin — wraps existing SPC capability engine.

Inputs: raw data + spec limits.
Outputs: Cpk/Ppk metrics (PCL-writable), charts, summary text.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from syn.plugins.base import Plugin, PluginOutput


class CapabilityInput(BaseModel):
    """Input schema for capability study."""

    data: List[float]
    usl: Optional[float] = None
    lsl: Optional[float] = None
    target: Optional[float] = None
    measurement: str = "measurement"

    @field_validator("data")
    @classmethod
    def data_not_empty(cls, v):
        if len(v) < 2:
            raise ValueError("Need at least 2 data points")
        return v

    @field_validator("usl")
    @classmethod
    def usl_gt_lsl(cls, v, info):
        lsl = info.data.get("lsl")
        if v is not None and lsl is not None and v <= lsl:
            raise ValueError("USL must be greater than LSL")
        return v


class CapabilityStudyPlugin(Plugin):
    """Process capability analysis: Cp, Cpk, Pp, Ppk with histogram and Q-Q plot."""

    name = "capability_study"
    version = "1.0.0"
    description = "Process capability analysis with Cp, Cpk, Pp, Ppk indices"
    input_schema = CapabilityInput

    def execute(self, validated_input: Dict[str, Any], context: Dict[str, Any]) -> List[PluginOutput]:
        import pandas as pd
        from agents_api.analysis.spc.capability import run_capability

        # Build DataFrame in format expected by run_capability
        measurement = validated_input["measurement"]
        df = pd.DataFrame({measurement: validated_input["data"]})

        config = {
            "measurement": measurement,
            "usl": validated_input.get("usl"),
            "lsl": validated_input.get("lsl"),
            "target": validated_input.get("target"),
        }

        result = run_capability(df, config)

        # Convert to PluginOutputs
        outputs: List[PluginOutput] = []
        stats = result.get("statistics", {})

        # Metric outputs (PCL-writable)
        if stats.get("cp") is not None:
            outputs.append(PluginOutput("cp", "metric", stats["cp"], measure_slug="cp"))
        if stats.get("cpk") is not None:
            outputs.append(PluginOutput("cpk", "metric", stats["cpk"], measure_slug="cpk"))
        if stats.get("pp") is not None:
            outputs.append(PluginOutput("pp", "metric", stats["pp"], measure_slug="pp"))
        if stats.get("ppk") is not None:
            outputs.append(PluginOutput("ppk", "metric", stats["ppk"], measure_slug="ppk"))
        if stats.get("sigma_level") is not None:
            outputs.append(PluginOutput("sigma_level", "metric", stats["sigma_level"]))
        if stats.get("yield_pct") is not None:
            outputs.append(PluginOutput("yield_pct", "metric", stats["yield_pct"]))
        if stats.get("ppm_total") is not None:
            outputs.append(PluginOutput("ppm_total", "metric", stats["ppm_total"]))

        # Chart outputs
        for plot in result.get("plots", []):
            key = plot["title"].lower().replace(" ", "_").replace("(", "").replace(")", "")
            outputs.append(PluginOutput(key, "chart", plot))

        # Text summary
        if result.get("summary"):
            outputs.append(PluginOutput("summary", "text", {"text": result["summary"]}))

        # Narrative (if generated)
        if result.get("narrative"):
            outputs.append(PluginOutput("narrative", "text", result["narrative"]))

        return outputs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_capability.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd ~/kjerne && git add plugins/capability.py plugins/tests/test_capability.py
git commit -m "feat(plugins): CapabilityStudyPlugin — first plugin implementation

Wraps existing run_capability() engine. Outputs Cpk/Ppk as PCL-writable
metrics, histogram/Q-Q charts, and summary text. Validates USL>LSL."
```

---

### Task 6: Django App Wiring + Registration on Startup

**Files:**
- Create: `plugins/apps.py`
- Modify: `svend/settings.py` (add `"plugins"` to INSTALLED_APPS)

- [ ] **Step 1: Write test that plugins app loads and registers on startup**

```python
# plugins/tests/test_app_ready.py
"""Test that plugins register on Django startup."""

from django.test import TestCase

from syn.plugins import get_registry


class TestPluginAppReady(TestCase):
    def test_capability_study_registered(self):
        """CapabilityStudyPlugin should be registered after app startup."""
        registry = get_registry()
        assert registry.has("capability_study")

    def test_registry_metadata_accessible(self):
        registry = get_registry()
        meta = registry.get_metadata("capability_study")
        assert meta["name"] == "capability_study"
        assert meta["input_schema"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_app_ready.py -v`
Expected: FAIL — registry doesn't have capability_study (no apps.py registering it)

- [ ] **Step 3: Create plugins/apps.py and wire into settings**

```python
# plugins/apps.py
"""SVEND Plugin implementations — register on startup."""

from django.apps import AppConfig


class PluginsConfig(AppConfig):
    name = "plugins"
    verbose_name = "SVEND Plugins"

    def ready(self):
        from syn.plugins import get_registry
        from plugins.capability import CapabilityStudyPlugin

        registry = get_registry()
        if not registry.has("capability_study"):
            registry.register(CapabilityStudyPlugin)
```

In `svend/settings.py`, add `"plugins",` to INSTALLED_APPS after the `"job"` line:

```python
    "job",  # Job: Silent canvas run records (audit trail, outputs, session history)
    "plugins",  # SVEND Plugins: analysis engines registered as plugins
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_app_ready.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Run ALL plugin tests together**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/ -v`
Expected: All tests PASS (should be ~22 total across 5 test files)

- [ ] **Step 6: Commit**

```bash
cd ~/kjerne && git add plugins/apps.py svend/settings.py plugins/tests/test_app_ready.py
git commit -m "feat(plugins): Django app wiring + auto-registration

CapabilityStudyPlugin registers on app ready(). Future plugins
added here. Registry accessible from anywhere via get_registry()."
```

---

### Task 7: Event Schema Registration for plugin.execution.completed

**Files:**
- Create: `plugins/tests/test_event_schema.py`
- Modify: `plugins/apps.py`

- [ ] **Step 1: Write test for event schema registration**

```python
# plugins/tests/test_event_schema.py
"""Test that plugin event schemas are registered."""

from django.test import TestCase

from syn.events.models import EventSchemaRegistry


class TestPluginEventSchema(TestCase):
    def test_plugin_execution_completed_schema_exists(self):
        schema = EventSchemaRegistry.objects.filter(
            event_name="plugin.execution.completed"
        ).first()
        assert schema is not None
        assert schema.version == "1.0.0"

    def test_schema_validates_correct_payload(self):
        import jsonschema

        schema_obj = EventSchemaRegistry.objects.get(
            event_name="plugin.execution.completed"
        )
        payload = {
            "job_id": "550e8400-e29b-41d4-a716-446655440000",
            "plugin_name": "capability_study",
            "outputs": ["cpk", "ppk", "histogram"],
        }
        # Should not raise
        jsonschema.validate(payload, schema_obj.schema)

    def test_schema_rejects_invalid_payload(self):
        import jsonschema

        schema_obj = EventSchemaRegistry.objects.get(
            event_name="plugin.execution.completed"
        )
        payload = {"plugin_name": "test"}  # Missing required job_id, outputs
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(payload, schema_obj.schema)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_event_schema.py -v`
Expected: FAIL — no EventSchemaRegistry entry for plugin.execution.completed

- [ ] **Step 3: Register event schema in apps.py ready()**

Update `plugins/apps.py`:

```python
# plugins/apps.py
"""SVEND Plugin implementations — register on startup."""

from django.apps import AppConfig


class PluginsConfig(AppConfig):
    name = "plugins"
    verbose_name = "SVEND Plugins"

    def ready(self):
        from syn.plugins import get_registry
        from plugins.capability import CapabilityStudyPlugin

        registry = get_registry()
        if not registry.has("capability_study"):
            registry.register(CapabilityStudyPlugin)

        # Register event schema for plugin execution
        self._register_event_schemas()

    def _register_event_schemas(self):
        """Register plugin-related event schemas (EVT-001 compliant)."""
        from syn.events.models import EventSchemaRegistry

        schema_def = {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "format": "uuid"},
                "plugin_name": {"type": "string"},
                "outputs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["job_id", "plugin_name", "outputs"],
        }

        EventSchemaRegistry.objects.update_or_create(
            event_name="plugin.execution.completed",
            defaults={
                "version": "1.0.0",
                "schema": schema_def,
                "description": "Emitted after a plugin run completes successfully.",
                "is_active": True,
            },
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest plugins/tests/test_event_schema.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd ~/kjerne && git add plugins/apps.py plugins/tests/test_event_schema.py
git commit -m "feat(plugins): register plugin.execution.completed event schema

EVT-001 compliant event schema for bus emission after plugin runs.
Validates job_id, plugin_name, outputs list. Created on app startup."
```

---

## Self-Review

**Spec coverage:**
- ✅ Plugin ABC with typed InputSchema/OutputSchema
- ✅ PluginRegistry for discovery
- ✅ Runner with validate → Job → execute → JobOutputs → bus event
- ✅ First plugin (capability study) wrapping existing engine
- ✅ Event emission on completion
- ✅ Job model extended with plugin_name
- ✅ Scratch/exploratory flag support
- ✅ Provenance propagation to JobOutput
- ✅ measure_slug for PCL write declarations

**Not in scope (Phase 5+):**
- Canvas models and UI (Phase 5)
- PCL write-back from JobOutput (needs canvas binding config)
- Additional plugins beyond capability study
- ForgeViz chart conversion (charts stay as Plotly dicts until canvas renders them)

**Placeholder scan:** None found — all steps have concrete code.

**Type consistency:** Verified — PluginOutput fields match JobOutput model fields (output_key, output_type, value_numeric, value_json, provenance, measure_slug). Plugin.execute() returns List[PluginOutput] everywhere.

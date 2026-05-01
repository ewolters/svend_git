# PCL (Process Characteristics Library) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational data layer for all process measures in SVEND — measures with confidence-weighted Bayesian aggregation, immutable datapoints, formula evaluation, and VSM binding.

**Architecture:** New `pcl/` Django app extending `SynaraEntity` and `SynaraImmutableLog`. Three models (Measure, Datapoint, MeasureTarget) with a service layer (`pcl.service`) providing `read()`/`write()`/`set_target()`. Formula evaluation adapts Hoshin's AST walker for `[slug]` syntax. VSM binding is Approach C — inline values stay, optional `pcl_bindings` dict on step JSON resolves from PCL when present.

**Tech Stack:** Django 4.x, Python 3.10+, `syn.core.base_models.SynaraEntity`/`SynaraImmutableLog`, AST-based formula evaluation, pytest.

**Spec:** `docs/superpowers/specs/2026-04-30-pcl-synara-vsm-design.md`

---

## File Structure

```
pcl/                              # New Django app
├── __init__.py
├── apps.py                       # PclConfig
├── models.py                     # Measure, Datapoint, MeasureTarget, MeasureAggregate
├── service.py                    # pcl.read(), pcl.write(), pcl.set_target(), pcl.read_with_meta()
├── confidence.py                 # compute_confidence(), CONFIDENCE_WEIGHTS, consistency_factor()
├── formulas.py                   # evaluate_pcl_formula() — AST walker for [slug] syntax
├── aggregation.py                # update_aggregate(), compute_aggregate_from_scratch()
├── urls.py                       # API routes
├── views.py                      # CRUD + search + manual entry with confirmation
├── migrations/
│   └── 0001_initial.py           # Auto-generated
└── tests/
    ├── __init__.py
    ├── test_models.py            # Model creation, constraints, immutability
    ├── test_confidence.py        # Confidence computation
    ├── test_aggregation.py       # Bayesian weighted aggregation, outlier suppression, shift detection
    ├── test_formulas.py          # Formula evaluation, [slug] resolution, safety guards
    ├── test_service.py           # pcl.read/write/set_target integration
    └── test_views.py             # API endpoints
```

**Modified files:**
- `svend/settings.py` — add `"pcl"` to INSTALLED_APPS
- `svend/urls.py` — add `path("api/pcl/", include("pcl.urls"))`
- `vsm/views.py` — modify `_parse_step_context()` to resolve PCL bindings
- `static/js/vsm.js` — add PCL bind/unbind UI in step properties
- `templates/vsm.html` — add PCL binding HTML elements in properties panel

---

### Task 1: App scaffold and Measure model

**Files:**
- Create: `pcl/__init__.py`, `pcl/apps.py`, `pcl/models.py`
- Modify: `svend/settings.py`
- Test: `pcl/tests/__init__.py`, `pcl/tests/test_models.py`

- [ ] **Step 1: Write failing test for Measure creation**

```python
# pcl/tests/test_models.py
import pytest
from django.test import TestCase
from conftest import make_user, make_tenant, make_membership, SECURE_OFF

@SECURE_OFF
class MeasureModelTest(TestCase):
    def setUp(self):
        self.user = make_user("pcl@test.com", tier="team")
        self.tenant = make_tenant("PCL Org", slug="pcl-org", plan="team")
        make_membership(self.tenant, self.user)

    def test_create_raw_measure(self):
        from pcl.models import Measure
        m = Measure.objects.create(
            tenant=self.tenant,
            name="Press A Cycle Time",
            slug="press-a-ct",
            definition="Cycle time for press A stamping operation",
            unit="sec",
            measure_type="process",
            value_type="continuous",
            created_by=self.user.email,
        )
        assert m.id is not None
        assert m.slug == "press-a-ct"
        assert m.formula is None  # raw measure
        assert m.is_deleted is False

    def test_create_calculated_measure(self):
        from pcl.models import Measure
        # Create component measures first
        Measure.objects.create(
            tenant=self.tenant, name="Availability", slug="avail",
            unit="%", measure_type="process", value_type="proportion",
            created_by=self.user.email,
        )
        Measure.objects.create(
            tenant=self.tenant, name="Performance", slug="perf",
            unit="%", measure_type="process", value_type="proportion",
            created_by=self.user.email,
        )
        Measure.objects.create(
            tenant=self.tenant, name="Quality", slug="qual",
            unit="%", measure_type="process", value_type="proportion",
            created_by=self.user.email,
        )
        oee = Measure.objects.create(
            tenant=self.tenant, name="OEE", slug="oee",
            unit="%", measure_type="process", value_type="proportion",
            formula="[avail] * [perf] * [qual]",
            created_by=self.user.email,
        )
        assert oee.formula is not None
        assert oee.is_calculated

    def test_slug_unique_per_tenant(self):
        from pcl.models import Measure
        from django.db import IntegrityError
        Measure.objects.create(
            tenant=self.tenant, name="CT", slug="ct",
            unit="sec", measure_type="process", value_type="continuous",
            created_by=self.user.email,
        )
        with self.assertRaises(IntegrityError):
            Measure.objects.create(
                tenant=self.tenant, name="CT Duplicate", slug="ct",
                unit="sec", measure_type="process", value_type="continuous",
                created_by=self.user.email,
            )

    def test_range_alarm_fields(self):
        from pcl.models import Measure
        m = Measure.objects.create(
            tenant=self.tenant, name="Temp", slug="temp",
            unit="C", measure_type="process", value_type="continuous",
            range_min=18.0, range_max=25.0,
            created_by=self.user.email,
        )
        assert m.range_min == 18.0
        assert m.range_max == 25.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest pcl/tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pcl'`

- [ ] **Step 3: Create app scaffold**

```python
# pcl/__init__.py
# PCL — Process Characteristics Library

# pcl/apps.py
from django.apps import AppConfig

class PclConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "pcl"
    verbose_name = "PCL — Process Characteristics Library"
```

- [ ] **Step 4: Create Measure model**

```python
# pcl/models.py
import uuid
from django.conf import settings
from django.db import models
from syn.core.base_models import SynaraEntity


class Measure(SynaraEntity):
    """A named, typed characteristic of a process, material, product, or resource."""

    tenant = models.ForeignKey(
        "core.Tenant", on_delete=models.CASCADE,
        related_name="pcl_measures",
    )
    name = models.CharField(max_length=255)
    slug = models.SlugField(max_length=100, db_index=True)
    definition = models.TextField(blank=True, default="")
    unit = models.CharField(max_length=50, blank=True, default="")

    MEASURE_TYPES = [
        ("process", "Process"),
        ("material", "Material"),
        ("product", "Product"),
        ("resource", "Resource"),
    ]
    measure_type = models.CharField(max_length=20, choices=MEASURE_TYPES)

    VALUE_TYPES = [
        ("continuous", "Continuous"),
        ("discrete", "Discrete"),
        ("proportion", "Proportion"),
        ("integer", "Integer"),
    ]
    value_type = models.CharField(max_length=20, choices=VALUE_TYPES)

    range_min = models.FloatField(null=True, blank=True)
    range_max = models.FloatField(null=True, blank=True)

    # Calculated measures: formula references other slugs via [slug] syntax
    formula = models.TextField(null=True, blank=True)

    # Polymorphic parent (what this measure belongs to)
    parent_type = models.CharField(max_length=100, blank=True, default="")
    parent_id = models.UUIDField(null=True, blank=True)

    # Cached aggregate (updated incrementally on each pcl.write())
    cached_value = models.FloatField(null=True, blank=True)
    cached_variance = models.FloatField(null=True, blank=True)
    cached_confidence = models.FloatField(null=True, blank=True)
    cached_n = models.IntegerField(default=0)
    cached_effective_n = models.FloatField(default=0.0)
    cached_at = models.DateTimeField(null=True, blank=True)

    # Recency decay (optional — for drifting processes)
    decay_enabled = models.BooleanField(default=False)
    decay_halflife_days = models.FloatField(null=True, blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["tenant", "slug"],
                condition=models.Q(is_deleted=False),
                name="unique_active_measure_slug_per_tenant",
            ),
        ]
        ordering = ["name"]

    class SynaraMeta:
        event_domain = "pcl.measure"
        emit_events = ["created", "updated", "deleted"]

    def __str__(self):
        return f"{self.name} [{self.slug}] ({self.unit})"

    @property
    def is_calculated(self):
        return bool(self.formula)

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "slug": self.slug,
            "definition": self.definition,
            "unit": self.unit,
            "measure_type": self.measure_type,
            "value_type": self.value_type,
            "range_min": self.range_min,
            "range_max": self.range_max,
            "formula": self.formula,
            "is_calculated": self.is_calculated,
            "cached_value": self.cached_value,
            "cached_confidence": self.cached_confidence,
            "cached_n": self.cached_n,
            "decay_enabled": self.decay_enabled,
        }
```

- [ ] **Step 5: Register app in settings**

In `svend/settings.py`, add `"pcl",` to INSTALLED_APPS after the other domain apps.

- [ ] **Step 6: Create migration and run tests**

Run:
```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a
python3 manage.py makemigrations pcl
python3 manage.py migrate pcl
python3 -m pytest pcl/tests/test_models.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add pcl/ svend/settings.py
git commit -m "feat(pcl): add Measure model with SynaraEntity base, slug uniqueness, cached aggregate fields"
```

---

### Task 2: Datapoint model (immutable) and MeasureTarget

**Files:**
- Modify: `pcl/models.py`
- Test: `pcl/tests/test_models.py`

- [ ] **Step 1: Write failing tests for Datapoint and MeasureTarget**

```python
# Append to pcl/tests/test_models.py

@SECURE_OFF
class DatapointModelTest(TestCase):
    def setUp(self):
        self.user = make_user("dp@test.com", tier="team")
        self.tenant = make_tenant("DP Org", slug="dp-org", plan="team")
        make_membership(self.tenant, self.user)
        from pcl.models import Measure
        self.measure = Measure.objects.create(
            tenant=self.tenant, name="CT", slug="ct",
            unit="sec", measure_type="process", value_type="continuous",
            created_by=self.user.email,
        )

    def test_create_datapoint(self):
        from pcl.models import Datapoint
        dp = Datapoint.objects.create(
            measure=self.measure,
            value=45.0,
            source_type="manual",
            observation_count=1,
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        assert dp.id is not None
        assert dp.value == 45.0
        assert dp.confidence > 0

    def test_datapoint_immutable(self):
        from pcl.models import Datapoint
        dp = Datapoint.objects.create(
            measure=self.measure,
            value=45.0,
            source_type="manual",
            observation_count=1,
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        dp.value = 99.0
        with self.assertRaises((ValueError, PermissionError)):
            dp.save()

    def test_datapoint_cannot_delete(self):
        from pcl.models import Datapoint
        dp = Datapoint.objects.create(
            measure=self.measure,
            value=45.0,
            source_type="manual",
            observation_count=1,
            actor=self.user.email,
            tenant_id=self.tenant.id,
        )
        with self.assertRaises(PermissionError):
            dp.delete()

    def test_confidence_auto_computed(self):
        from pcl.models import Datapoint
        dp_manual = Datapoint.objects.create(
            measure=self.measure, value=45.0,
            source_type="manual", observation_count=1,
            actor=self.user.email, tenant_id=self.tenant.id,
        )
        dp_doe = Datapoint.objects.create(
            measure=self.measure, value=44.7,
            source_type="doe", observation_count=30,
            actor=self.user.email, tenant_id=self.tenant.id,
        )
        assert dp_doe.confidence > dp_manual.confidence


@SECURE_OFF
class MeasureTargetModelTest(TestCase):
    def setUp(self):
        self.user = make_user("mt@test.com", tier="team")
        self.tenant = make_tenant("MT Org", slug="mt-org", plan="team")
        make_membership(self.tenant, self.user)
        from pcl.models import Measure
        self.measure = Measure.objects.create(
            tenant=self.tenant, name="CT", slug="ct-target",
            unit="sec", measure_type="process", value_type="continuous",
            created_by=self.user.email,
        )

    def test_create_target(self):
        from pcl.models import MeasureTarget
        from datetime import date
        t = MeasureTarget.objects.create(
            tenant=self.tenant,
            measure=self.measure,
            target_value=38.0,
            target_date=date(2026, 9, 1),
            source="hoshin",
            created_by=self.user.email,
        )
        assert t.target_value == 38.0
        assert t.source == "hoshin"

    def test_target_is_mutable(self):
        from pcl.models import MeasureTarget
        t = MeasureTarget.objects.create(
            tenant=self.tenant,
            measure=self.measure,
            target_value=38.0,
            source="manual",
            created_by=self.user.email,
        )
        t.target_value = 35.0
        t.save()  # Should NOT raise
        t.refresh_from_db()
        assert t.target_value == 35.0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest pcl/tests/test_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'Datapoint'`

- [ ] **Step 3: Add Datapoint and MeasureTarget models**

Append to `pcl/models.py`:

```python
from syn.core.base_models import SynaraImmutableLog


class Datapoint(SynaraImmutableLog):
    """An immutable, timestamped observation of a raw measure.

    Write-once. Hash-chained for 21 CFR Part 11 compliance.
    Confidence auto-computed from source_type x observation_count.
    """

    measure = models.ForeignKey(
        Measure, on_delete=models.CASCADE,
        related_name="datapoints",
    )
    value = models.FloatField()
    source_type = models.CharField(max_length=30, db_index=True)
    source_ref_type = models.CharField(max_length=100, blank=True, default="")
    source_ref_id = models.UUIDField(null=True, blank=True)
    observation_count = models.IntegerField(default=1)
    notes = models.TextField(blank=True, default="")
    confidence = models.FloatField(default=0.0)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["measure", "-created_at"]),
            models.Index(fields=["measure", "source_type"]),
        ]

    def save(self, *args, **kwargs):
        if not self._state.adding:
            raise ValueError("Datapoints are immutable. Create a new datapoint instead.")
        from pcl.confidence import compute_confidence
        self.confidence = compute_confidence(self.source_type, self.observation_count)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.measure.slug}={self.value} ({self.source_type}, conf={self.confidence:.2f})"


class MeasureTarget(SynaraEntity):
    """Aspirational value for a measure — the working/future state layer."""

    tenant = models.ForeignKey(
        "core.Tenant", on_delete=models.CASCADE,
        related_name="pcl_targets",
    )
    measure = models.ForeignKey(
        Measure, on_delete=models.CASCADE,
        related_name="targets",
    )
    target_value = models.FloatField()
    target_date = models.DateField(null=True, blank=True)
    source = models.CharField(max_length=50, default="manual")
    source_ref_type = models.CharField(max_length=100, blank=True, default="")
    source_ref_id = models.UUIDField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    class SynaraMeta:
        event_domain = "pcl.target"
        emit_events = ["created", "updated"]

    def __str__(self):
        return f"{self.measure.slug} target={self.target_value} ({self.source})"
```

- [ ] **Step 4: Create confidence module**

```python
# pcl/confidence.py
"""Confidence computation for PCL datapoints.

Confidence = min(1.0, base_weight * log2(max(1, n)) / log2(plateau))
"""
import math

CONFIDENCE_WEIGHTS = {
    "automated": {"base": 0.95, "plateau": 10},
    "doe": {"base": 0.90, "plateau": 8},
    "workbench": {"base": 0.85, "plateau": 15},
    "time_study": {"base": 0.70, "plateau": 30},
    "manual": {"base": 0.50, "plateau": 30},
    "estimate": {"base": 0.25, "plateau": 30},
    "calculator": {"base": 0.60, "plateau": 10},
}

DEFAULT_WEIGHT = {"base": 0.50, "plateau": 30}


def compute_confidence(source_type: str, observation_count: int) -> float:
    """Compute confidence from source type and observation count.

    Returns float in [0.0, 1.0].
    """
    w = CONFIDENCE_WEIGHTS.get(source_type, DEFAULT_WEIGHT)
    n = max(1, observation_count)
    plateau = w["plateau"]
    raw = w["base"] * math.log2(max(1, n)) / math.log2(plateau)
    return min(1.0, max(0.0, raw))
```

- [ ] **Step 5: Migrate and run tests**

Run:
```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a
python3 manage.py makemigrations pcl
python3 manage.py migrate pcl
python3 -m pytest pcl/tests/test_models.py -v
```
Expected: All tests PASS (8 total).

- [ ] **Step 6: Commit**

```bash
git add pcl/
git commit -m "feat(pcl): add immutable Datapoint (SynaraImmutableLog) and MeasureTarget models with auto-computed confidence"
```

---

### Task 3: Confidence computation tests

**Files:**
- Create: `pcl/tests/test_confidence.py`
- Reference: `pcl/confidence.py` (already created in Task 2)

- [ ] **Step 1: Write confidence tests**

```python
# pcl/tests/test_confidence.py
from pcl.confidence import compute_confidence, CONFIDENCE_WEIGHTS


class TestComputeConfidence:
    def test_manual_single_observation(self):
        c = compute_confidence("manual", 1)
        # base=0.50, log2(1)/log2(30) = 0/~4.9 = 0
        assert c == 0.0

    def test_manual_few_observations(self):
        c = compute_confidence("manual", 3)
        assert 0.0 < c < 0.5

    def test_manual_at_plateau(self):
        c = compute_confidence("manual", 30)
        assert abs(c - 0.50) < 0.01

    def test_doe_at_plateau(self):
        c = compute_confidence("doe", 8)
        assert abs(c - 0.90) < 0.01

    def test_automated_high(self):
        c = compute_confidence("automated", 10)
        assert abs(c - 0.95) < 0.01

    def test_estimate_always_low(self):
        c = compute_confidence("estimate", 1)
        assert c == 0.0
        c30 = compute_confidence("estimate", 30)
        assert abs(c30 - 0.25) < 0.01

    def test_doe_beats_manual_at_same_n(self):
        c_doe = compute_confidence("doe", 8)
        c_manual = compute_confidence("manual", 8)
        assert c_doe > c_manual

    def test_caps_at_one(self):
        c = compute_confidence("automated", 10000)
        assert c <= 1.0

    def test_unknown_source_uses_default(self):
        c = compute_confidence("unknown_source", 10)
        assert 0.0 < c <= 1.0

    def test_zero_observation_treated_as_one(self):
        c = compute_confidence("manual", 0)
        assert c == compute_confidence("manual", 1)
```

- [ ] **Step 2: Run tests**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest pcl/tests/test_confidence.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add pcl/tests/test_confidence.py
git commit -m "test(pcl): confidence computation — source weights, plateau scaling, bounds"
```

---

### Task 4: Bayesian weighted aggregation

**Files:**
- Create: `pcl/aggregation.py`, `pcl/tests/test_aggregation.py`

- [ ] **Step 1: Write failing tests for aggregation**

```python
# pcl/tests/test_aggregation.py
import pytest
from pcl.aggregation import compute_aggregate, update_aggregate_incremental


class TestComputeAggregate:
    def test_single_datapoint(self):
        """First datapoint IS the aggregate."""
        result = compute_aggregate(
            values=[45.0],
            confidences=[0.50],
            timestamps_age_days=[0.0],
            decay_enabled=False,
        )
        assert result["value"] == 45.0
        assert result["n"] == 1
        assert result["effective_n"] == pytest.approx(0.50, abs=0.1)

    def test_two_equal_confidence(self):
        """Equal confidence = simple mean."""
        result = compute_aggregate(
            values=[40.0, 50.0],
            confidences=[0.50, 0.50],
            timestamps_age_days=[0.0, 0.0],
        )
        assert result["value"] == pytest.approx(45.0, abs=0.01)

    def test_high_confidence_dominates(self):
        """DOE at 54 should dominate over manual at 45."""
        result = compute_aggregate(
            values=[45.0, 54.0],
            confidences=[0.30, 0.90],
            timestamps_age_days=[0.0, 0.0],
        )
        # 54 has 3x the weight of 45
        assert result["value"] > 50.0

    def test_fat_finger_suppressed(self):
        """Outlier with low confidence barely moves aggregate."""
        # Establish baseline: three consistent observations
        result = compute_aggregate(
            values=[54.0, 53.8, 54.2, 45.0],
            confidences=[0.90, 0.85, 0.85, 0.30],
            timestamps_age_days=[3.0, 2.0, 1.0, 0.0],
        )
        # 45 is ~1.7σ from ~54 mean, low confidence → near-zero weight
        assert result["value"] > 52.0

    def test_recency_decay(self):
        """Old data decays when decay_enabled."""
        result = compute_aggregate(
            values=[50.0, 55.0],
            confidences=[0.90, 0.90],
            timestamps_age_days=[365.0, 0.0],  # first is 1 year old
            decay_enabled=True,
            decay_halflife_days=90.0,
        )
        # Old value should be heavily decayed, newer value dominates
        assert result["value"] > 53.0

    def test_no_decay_by_default(self):
        """Without decay, old data has full weight."""
        result = compute_aggregate(
            values=[50.0, 55.0],
            confidences=[0.90, 0.90],
            timestamps_age_days=[365.0, 0.0],
            decay_enabled=False,
        )
        assert result["value"] == pytest.approx(52.5, abs=0.5)

    def test_shift_detection(self):
        """Sustained high-confidence deviation = shift detected."""
        # 5 old points at ~50, then 5 new points at ~60
        values = [50.0, 50.5, 49.8, 50.2, 50.1, 60.0, 60.3, 59.8, 60.1, 60.2]
        confidences = [0.85] * 10
        ages = [10.0, 9.0, 8.0, 7.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        result = compute_aggregate(
            values=values, confidences=confidences,
            timestamps_age_days=ages,
            decay_enabled=True, decay_halflife_days=30.0,
        )
        assert result["shift_detected"] is True
        assert result["value"] > 55.0  # aggregate moved toward new level

    def test_empty_returns_none(self):
        result = compute_aggregate(values=[], confidences=[], timestamps_age_days=[])
        assert result["value"] is None
        assert result["n"] == 0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest pcl/tests/test_aggregation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pcl.aggregation'`

- [ ] **Step 3: Implement aggregation module**

```python
# pcl/aggregation.py
"""Bayesian weighted aggregation for PCL measures.

Every datapoint contributes. Weight = confidence x consistency_factor.
Outliers (low confidence + high deviation) get near-zero effective weight.
Shift detection via rolling window comparison.
"""
import math
from typing import Optional


def _consistency_factor(value: float, mean: float, std: float) -> float:
    """Penalize values that deviate from the running estimate.

    Returns 1.0 for values at the mean, decays toward 0 for outliers.
    Uses a Gaussian-shaped penalty: exp(-0.5 * z^2) where z = (value - mean) / std.
    """
    if std <= 0:
        return 1.0
    z = abs(value - mean) / std
    return math.exp(-0.5 * z * z)


def _decay_factor(age_days: float, halflife_days: float) -> float:
    """Exponential decay based on age. Returns 1.0 for age=0, 0.5 at halflife."""
    if halflife_days <= 0 or age_days <= 0:
        return 1.0
    return math.pow(0.5, age_days / halflife_days)


def compute_aggregate(
    values: list[float],
    confidences: list[float],
    timestamps_age_days: list[float],
    decay_enabled: bool = False,
    decay_halflife_days: Optional[float] = None,
) -> dict:
    """Compute the Bayesian weighted aggregate of all datapoints.

    Returns dict with: value, confidence, variance, n, effective_n,
    shift_detected.
    """
    n = len(values)
    if n == 0:
        return {
            "value": None, "confidence": 0.0, "variance": 0.0,
            "n": 0, "effective_n": 0.0, "shift_detected": False,
        }

    # First pass: raw weighted mean (confidence only, no consistency yet)
    raw_weights = []
    for i in range(n):
        w = confidences[i]
        if decay_enabled and decay_halflife_days:
            w *= _decay_factor(timestamps_age_days[i], decay_halflife_days)
        raw_weights.append(w)

    total_raw = sum(raw_weights)
    if total_raw <= 0:
        return {
            "value": values[-1] if values else None,
            "confidence": 0.0, "variance": 0.0,
            "n": n, "effective_n": 0.0, "shift_detected": False,
        }

    raw_mean = sum(v * w for v, w in zip(values, raw_weights)) / total_raw

    # Compute standard deviation from raw weighted mean
    if n >= 2:
        var_sum = sum(w * (v - raw_mean) ** 2 for v, w in zip(values, raw_weights))
        raw_std = math.sqrt(var_sum / total_raw) if total_raw > 0 else 0.0
    else:
        raw_std = 0.0

    # Second pass: apply consistency factor
    effective_weights = []
    for i in range(n):
        cf = _consistency_factor(values[i], raw_mean, raw_std) if raw_std > 0 else 1.0
        ew = raw_weights[i] * cf
        effective_weights.append(ew)

    total_ew = sum(effective_weights)
    if total_ew <= 0:
        return {
            "value": raw_mean, "confidence": 0.0, "variance": raw_std ** 2,
            "n": n, "effective_n": 0.0, "shift_detected": False,
        }

    agg_value = sum(v * w for v, w in zip(values, effective_weights)) / total_ew

    # Weighted variance around aggregate
    if n >= 2:
        agg_var = sum(w * (v - agg_value) ** 2 for v, w in zip(values, effective_weights)) / total_ew
    else:
        agg_var = 0.0

    effective_n = sum(effective_weights)
    agg_confidence = min(1.0, effective_n / n) if n > 0 else 0.0

    # Shift detection: compare recent window (last 30%) vs older window
    shift_detected = False
    if n >= 6:
        split = max(3, n * 7 // 10)  # 70/30 split
        old_vals = values[:split]
        new_vals = values[split:]
        if old_vals and new_vals:
            old_mean = sum(old_vals) / len(old_vals)
            new_mean = sum(new_vals) / len(new_vals)
            if raw_std > 0:
                shift_z = abs(new_mean - old_mean) / raw_std
                shift_detected = shift_z > 1.5

    return {
        "value": agg_value,
        "confidence": agg_confidence,
        "variance": agg_var,
        "n": n,
        "effective_n": effective_n,
        "shift_detected": shift_detected,
    }


def update_aggregate_incremental(
    cached_value: Optional[float],
    cached_variance: Optional[float],
    cached_confidence: Optional[float],
    cached_n: int,
    cached_effective_n: float,
    new_value: float,
    new_confidence: float,
    decay_enabled: bool = False,
    decay_halflife_days: Optional[float] = None,
    new_age_days: float = 0.0,
) -> dict:
    """Incrementally update the cached aggregate with a new datapoint.

    This is the fast path for pcl.write() — avoids recomputing from scratch.
    For the first datapoint, cached_value should be None.
    """
    w = new_confidence
    if decay_enabled and decay_halflife_days:
        w *= _decay_factor(new_age_days, decay_halflife_days)

    if cached_value is None or cached_n == 0:
        return {
            "value": new_value,
            "variance": 0.0,
            "confidence": new_confidence,
            "n": 1,
            "effective_n": w,
        }

    # Consistency factor against existing aggregate
    std = math.sqrt(cached_variance) if cached_variance and cached_variance > 0 else 0.0
    cf = _consistency_factor(new_value, cached_value, std) if std > 0 else 1.0
    ew = w * cf

    total_ew = cached_effective_n + ew
    if total_ew <= 0:
        return {
            "value": cached_value,
            "variance": cached_variance or 0.0,
            "confidence": cached_confidence or 0.0,
            "n": cached_n + 1,
            "effective_n": cached_effective_n,
        }

    # Online weighted mean update
    new_agg = (cached_value * cached_effective_n + new_value * ew) / total_ew

    # Online variance update (Welford-like)
    delta = new_value - cached_value
    new_var = (
        (cached_effective_n * ((cached_variance or 0.0) + (cached_value - new_agg) ** 2)
         + ew * (new_value - new_agg) ** 2) / total_ew
    )

    new_n = cached_n + 1

    return {
        "value": new_agg,
        "variance": new_var,
        "confidence": min(1.0, total_ew / new_n) if new_n > 0 else 0.0,
        "n": new_n,
        "effective_n": total_ew,
    }
```

- [ ] **Step 4: Run tests**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest pcl/tests/test_aggregation.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pcl/aggregation.py pcl/tests/test_aggregation.py
git commit -m "feat(pcl): Bayesian weighted aggregation with consistency penalty, recency decay, shift detection"
```

---

### Task 5: Formula evaluation for calculated measures

**Files:**
- Create: `pcl/formulas.py`, `pcl/tests/test_formulas.py`

- [ ] **Step 1: Write failing tests for formula evaluation**

```python
# pcl/tests/test_formulas.py
import pytest
from pcl.formulas import evaluate_pcl_formula, extract_slugs


class TestExtractSlugs:
    def test_single_slug(self):
        assert extract_slugs("[press-a-ct]") == ["press-a-ct"]

    def test_multiple_slugs(self):
        assert extract_slugs("[avail] * [perf] * [qual]") == ["avail", "perf", "qual"]

    def test_no_slugs(self):
        assert extract_slugs("42 + 3") == []

    def test_duplicate_slugs_deduped(self):
        assert extract_slugs("[x] + [x]") == ["x"]


class TestEvaluatePclFormula:
    def test_simple_arithmetic(self):
        result = evaluate_pcl_formula("[a] + [b]", {"a": 10.0, "b": 20.0})
        assert result == 30.0

    def test_oee_formula(self):
        result = evaluate_pcl_formula(
            "[avail] * [perf] * [qual]",
            {"avail": 0.90, "perf": 0.85, "qual": 0.95},
        )
        assert result == pytest.approx(0.72675, abs=0.001)

    def test_takt_formula(self):
        result = evaluate_pcl_formula(
            "[available_time] / [demand]",
            {"available_time": 28800.0, "demand": 480.0},
        )
        assert result == 60.0

    def test_safe_functions(self):
        result = evaluate_pcl_formula("max([a], [b])", {"a": 10.0, "b": 20.0})
        assert result == 20.0

    def test_unknown_slug_raises(self):
        with pytest.raises(ValueError, match="Unknown variable"):
            evaluate_pcl_formula("[missing]", {})

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            evaluate_pcl_formula("[a] / [b]", {"a": 10.0, "b": 0.0})

    def test_rejects_imports(self):
        with pytest.raises(ValueError):
            evaluate_pcl_formula("__import__('os')", {})

    def test_rejects_attribute_access(self):
        with pytest.raises(ValueError):
            evaluate_pcl_formula("[a].__class__", {"a": 1.0})

    def test_max_length_guard(self):
        with pytest.raises(ValueError, match="too long"):
            evaluate_pcl_formula("[a] + " * 200, {"a": 1.0})

    def test_max_depth_guard(self):
        # Deeply nested: ((((((1 + 1) + 1) + 1) ...
        formula = "[a]"
        for _ in range(25):
            formula = f"({formula} + [a])"
        with pytest.raises(ValueError, match="too deep|too complex"):
            evaluate_pcl_formula(formula, {"a": 1.0})
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest pcl/tests/test_formulas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pcl.formulas'`

- [ ] **Step 3: Implement formula evaluator**

```python
# pcl/formulas.py
"""Safe formula evaluation for PCL calculated measures.

Uses [slug] syntax to reference other measures. Evaluated via restricted
AST walking — same pattern as hoshin/hoshin_calculations.py but with
slug resolution instead of {{fieldname}} variables.
"""
import ast
import math
import operator
import re
from typing import Optional

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "pow": pow,
}

_SLUG_PATTERN = re.compile(r"\[([a-zA-Z0-9_-]+)\]")
_MAX_LEN = 500
_MAX_NODES = 100
_MAX_DEPTH = 20


def extract_slugs(formula: str) -> list[str]:
    """Extract [slug] references from a formula. Returns deduplicated list."""
    return list(dict.fromkeys(_SLUG_PATTERN.findall(formula)))


def _normalize(formula: str) -> str:
    """Replace [slug] with bare names for AST parsing.

    Handles hyphens by replacing with underscores (valid Python identifiers).
    """
    def _replace(m):
        return m.group(1).replace("-", "_")
    return _SLUG_PATTERN.sub(_replace, formula)


def evaluate_pcl_formula(formula: str, variables: dict[str, float]) -> float:
    """Safely evaluate a PCL formula with [slug] references.

    Args:
        formula: e.g. "[avail] * [perf] * [qual]"
        variables: dict mapping slug (with hyphens) to float values.
                   e.g. {"avail": 0.90, "perf": 0.85, "qual": 0.95}

    Returns:
        float result

    Raises:
        ValueError: if formula is unsafe, too complex, or references unknown slugs
        ZeroDivisionError: if formula divides by zero
    """
    if len(formula) > _MAX_LEN:
        raise ValueError(f"Formula too long (max {_MAX_LEN} chars)")

    # Normalize: [press-a-ct] -> press_a_ct
    normalized = _normalize(formula)
    # Also normalize variable keys
    norm_vars = {k.replace("-", "_"): v for k, v in variables.items()}

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid formula syntax: {e}")

    def _count(node):
        c = 1
        for child in ast.iter_child_nodes(node):
            c += _count(child)
        return c

    if _count(tree) > _MAX_NODES:
        raise ValueError(f"Formula too complex (max {_MAX_NODES} AST nodes)")

    def _eval(node, depth=0):
        if depth > _MAX_DEPTH:
            raise ValueError(f"Formula nesting too deep (max {_MAX_DEPTH} levels)")
        if isinstance(node, ast.Expression):
            return _eval(node.body, depth + 1)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Non-numeric constant: {node.value!r}")
        elif isinstance(node, ast.Name):
            if node.id in norm_vars:
                return float(norm_vars[node.id])
            raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.BinOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            left = _eval(node.left, depth + 1)
            right = _eval(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and right > 10:
                raise ValueError("Exponent too large (max 10)")
            return op_fn(left, right)
        elif isinstance(node, ast.UnaryOp):
            op_fn = _SAFE_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op_fn(_eval(node.operand, depth + 1))
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            fn = _SAFE_FUNCS.get(node.func.id)
            if fn is None:
                raise ValueError(f"Function not allowed: {node.func.id}")
            args = [_eval(a, depth + 1) for a in node.args]
            return fn(*args)
        else:
            raise ValueError(f"Unsupported expression: {type(node).__name__}")

    return _eval(tree)
```

- [ ] **Step 4: Run tests**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest pcl/tests/test_formulas.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pcl/formulas.py pcl/tests/test_formulas.py
git commit -m "feat(pcl): safe formula evaluator with [slug] syntax, AST walking, security guards"
```

---

### Task 6: Service layer — pcl.read(), pcl.write(), pcl.set_target()

**Files:**
- Create: `pcl/service.py`, `pcl/tests/test_service.py`

- [ ] **Step 1: Write failing tests for service layer**

```python
# pcl/tests/test_service.py
import pytest
from django.test import TestCase
from django.utils import timezone
from conftest import make_user, make_tenant, make_membership, SECURE_OFF


@SECURE_OFF
class PCLServiceTest(TestCase):
    def setUp(self):
        self.user = make_user("svc@test.com", tier="team")
        self.tenant = make_tenant("Svc Org", slug="svc-org", plan="team")
        make_membership(self.tenant, self.user)

        from pcl.models import Measure
        self.ct = Measure.objects.create(
            tenant=self.tenant, name="Cycle Time", slug="ct",
            unit="sec", measure_type="process", value_type="continuous",
            created_by=self.user.email,
        )

    def test_write_creates_datapoint(self):
        from pcl.service import write
        dp = write(
            tenant=self.tenant, measure_slug="ct",
            value=45.0, source_type="manual",
            observation_count=1, actor=self.user.email,
        )
        assert dp.value == 45.0
        assert dp.confidence > 0

    def test_write_updates_cached_aggregate(self):
        from pcl.service import write
        write(tenant=self.tenant, measure_slug="ct",
              value=45.0, source_type="manual",
              observation_count=1, actor=self.user.email)
        self.ct.refresh_from_db()
        assert self.ct.cached_value == 45.0
        assert self.ct.cached_n == 1

    def test_read_returns_aggregate(self):
        from pcl.service import write, read
        write(tenant=self.tenant, measure_slug="ct",
              value=45.0, source_type="doe",
              observation_count=30, actor=self.user.email)
        write(tenant=self.tenant, measure_slug="ct",
              value=46.0, source_type="doe",
              observation_count=30, actor=self.user.email)
        val = read(tenant=self.tenant, measure_slug="ct")
        assert 45.0 <= val <= 46.0

    def test_read_missing_returns_none(self):
        from pcl.service import read
        val = read(tenant=self.tenant, measure_slug="ct")
        assert val is None

    def test_read_nonexistent_slug_raises(self):
        from pcl.service import read
        with self.assertRaises(ValueError):
            read(tenant=self.tenant, measure_slug="nonexistent")

    def test_read_with_meta(self):
        from pcl.service import write, read_with_meta
        write(tenant=self.tenant, measure_slug="ct",
              value=45.0, source_type="manual",
              observation_count=1, actor=self.user.email)
        meta = read_with_meta(tenant=self.tenant, measure_slug="ct")
        assert meta["value"] == 45.0
        assert "confidence" in meta
        assert "n" in meta
        assert "effective_n" in meta

    def test_read_calculated_measure(self):
        from pcl.models import Measure
        from pcl.service import write, read
        a = Measure.objects.create(
            tenant=self.tenant, name="A", slug="a",
            unit="", measure_type="process", value_type="continuous",
            created_by=self.user.email,
        )
        b = Measure.objects.create(
            tenant=self.tenant, name="B", slug="b",
            unit="", measure_type="process", value_type="continuous",
            created_by=self.user.email,
        )
        c = Measure.objects.create(
            tenant=self.tenant, name="A+B", slug="apb",
            unit="", measure_type="process", value_type="continuous",
            formula="[a] + [b]",
            created_by=self.user.email,
        )
        write(tenant=self.tenant, measure_slug="a", value=10.0,
              source_type="manual", observation_count=1, actor=self.user.email)
        write(tenant=self.tenant, measure_slug="b", value=20.0,
              source_type="manual", observation_count=1, actor=self.user.email)
        val = read(tenant=self.tenant, measure_slug="apb")
        assert val == 30.0

    def test_write_checks_range(self):
        from pcl.models import Measure
        from pcl.service import write
        m = Measure.objects.create(
            tenant=self.tenant, name="Temp", slug="temp",
            unit="C", measure_type="process", value_type="continuous",
            range_min=18.0, range_max=25.0,
            created_by=self.user.email,
        )
        # Value outside range still stores but returns alarm flag
        dp = write(tenant=self.tenant, measure_slug="temp",
                    value=30.0, source_type="manual",
                    observation_count=1, actor=self.user.email)
        assert dp.value == 30.0  # stored
        # The alarm is a side effect — tested via event schemas later

    def test_set_target(self):
        from pcl.service import set_target
        from datetime import date
        t = set_target(
            tenant=self.tenant, measure_slug="ct",
            target_value=38.0, source="hoshin",
            target_date=date(2026, 9, 1),
            actor=self.user.email,
        )
        assert t.target_value == 38.0

    def test_fat_finger_suppressed_in_aggregate(self):
        from pcl.service import write, read
        # Build a baseline
        for v in [54.0, 53.8, 54.2, 53.9, 54.1]:
            write(tenant=self.tenant, measure_slug="ct",
                  value=v, source_type="doe",
                  observation_count=30, actor=self.user.email)
        # Fat finger
        write(tenant=self.tenant, measure_slug="ct",
              value=45.0, source_type="manual",
              observation_count=1, actor=self.user.email)
        val = read(tenant=self.tenant, measure_slug="ct")
        assert val > 52.0  # fat finger barely moves aggregate
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest pcl/tests/test_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pcl.service'`

- [ ] **Step 3: Implement service layer**

```python
# pcl/service.py
"""PCL service layer — the public API for reading and writing measures.

Usage:
    from pcl.service import read, write, read_with_meta, set_target

    write(tenant, "press-a-ct", 44.7, source_type="doe", observation_count=30, actor="eric@svend.ai")
    value = read(tenant, "press-a-ct")
    meta = read_with_meta(tenant, "press-a-ct")
    set_target(tenant, "press-a-ct", target_value=38.0, source="hoshin")
"""
import logging
from datetime import date, datetime
from typing import Optional
from uuid import UUID

from django.utils import timezone

from .models import Measure, Datapoint, MeasureTarget
from .aggregation import compute_aggregate, update_aggregate_incremental
from .formulas import evaluate_pcl_formula, extract_slugs

logger = logging.getLogger(__name__)


def _get_measure(tenant, slug: str) -> Measure:
    try:
        return Measure.objects.get(tenant=tenant, slug=slug, is_deleted=False)
    except Measure.DoesNotExist:
        raise ValueError(f"Measure '{slug}' not found for tenant {tenant}")


def write(
    tenant,
    measure_slug: str,
    value: float,
    source_type: str,
    observation_count: int = 1,
    actor: str = "system",
    timestamp: Optional[datetime] = None,
    source_ref_type: str = "",
    source_ref_id: Optional[UUID] = None,
    notes: str = "",
) -> Datapoint:
    """Write a datapoint to a raw measure. Updates cached aggregate."""
    measure = _get_measure(tenant, measure_slug)

    if measure.is_calculated:
        raise ValueError(f"Cannot write datapoints to calculated measure '{measure_slug}'")

    dp = Datapoint.objects.create(
        measure=measure,
        value=value,
        source_type=source_type,
        source_ref_type=source_ref_type,
        source_ref_id=source_ref_id,
        observation_count=observation_count,
        notes=notes,
        actor=actor,
        tenant_id=tenant.id,
    )

    # Update cached aggregate incrementally
    agg = update_aggregate_incremental(
        cached_value=measure.cached_value,
        cached_variance=measure.cached_variance,
        cached_confidence=measure.cached_confidence,
        cached_n=measure.cached_n,
        cached_effective_n=measure.cached_effective_n,
        new_value=value,
        new_confidence=dp.confidence,
        decay_enabled=measure.decay_enabled,
        decay_halflife_days=measure.decay_halflife_days,
    )
    measure.cached_value = agg["value"]
    measure.cached_variance = agg["variance"]
    measure.cached_confidence = agg["confidence"]
    measure.cached_n = agg["n"]
    measure.cached_effective_n = agg["effective_n"]
    measure.cached_at = timezone.now()
    measure.save(update_fields=[
        "cached_value", "cached_variance", "cached_confidence",
        "cached_n", "cached_effective_n", "cached_at",
    ])

    return dp


def read(tenant, measure_slug: str, at: Optional[datetime] = None) -> Optional[float]:
    """Read the current aggregate value of a measure.

    For raw measures: returns cached aggregate (or computes from scratch if stale).
    For calculated measures: evaluates formula by resolving component slugs.
    Returns None if no datapoints exist.
    """
    measure = _get_measure(tenant, measure_slug)

    if measure.is_calculated:
        return _resolve_calculated(tenant, measure, at=at)

    if at is not None:
        return _compute_historical(measure, at)

    if measure.cached_n == 0:
        return None

    return measure.cached_value


def read_with_meta(tenant, measure_slug: str) -> dict:
    """Read aggregate value plus metadata."""
    measure = _get_measure(tenant, measure_slug)

    if measure.is_calculated:
        value = _resolve_calculated(tenant, measure)
        # Collect component metadata
        slugs = extract_slugs(measure.formula)
        component_confidences = []
        component_timestamps = []
        for s in slugs:
            try:
                comp = _get_measure(tenant, s)
                if comp.cached_confidence is not None:
                    component_confidences.append(comp.cached_confidence)
                if comp.cached_at is not None:
                    component_timestamps.append(comp.cached_at)
            except ValueError:
                pass
        return {
            "value": value,
            "confidence": min(component_confidences) if component_confidences else 0.0,
            "variance": None,
            "n": None,
            "effective_n": None,
            "latest_timestamp": max(component_timestamps) if component_timestamps else None,
            "staleness_days": None,
            "shift_detected": False,
            "is_calculated": True,
            "formula": measure.formula,
        }

    latest_dp = measure.datapoints.first()
    staleness = None
    if measure.cached_at:
        staleness = (timezone.now() - measure.cached_at).days

    return {
        "value": measure.cached_value,
        "confidence": measure.cached_confidence,
        "variance": measure.cached_variance,
        "n": measure.cached_n,
        "effective_n": measure.cached_effective_n,
        "latest_timestamp": measure.cached_at,
        "staleness_days": staleness,
        "shift_detected": False,  # TODO: wire shift detection from full recompute
        "is_calculated": False,
    }


def set_target(
    tenant,
    measure_slug: str,
    target_value: float,
    source: str = "manual",
    target_date: Optional[date] = None,
    actor: str = "system",
    source_ref_type: str = "",
    source_ref_id: Optional[UUID] = None,
) -> MeasureTarget:
    """Set or update the target (working layer) for a measure."""
    measure = _get_measure(tenant, measure_slug)
    target, created = MeasureTarget.objects.update_or_create(
        tenant=tenant,
        measure=measure,
        defaults={
            "target_value": target_value,
            "target_date": target_date,
            "source": source,
            "source_ref_type": source_ref_type,
            "source_ref_id": source_ref_id,
            "created_by": actor,
        },
    )
    return target


def _resolve_calculated(
    tenant, measure: Measure, at: Optional[datetime] = None,
    _depth: int = 0,
) -> Optional[float]:
    """Resolve a calculated measure by evaluating its formula."""
    if _depth > 10:
        raise ValueError(f"Circular formula reference detected at '{measure.slug}'")

    slugs = extract_slugs(measure.formula)
    variables = {}
    for slug in slugs:
        comp = _get_measure(tenant, slug)
        if comp.is_calculated:
            val = _resolve_calculated(tenant, comp, at=at, _depth=_depth + 1)
        elif at is not None:
            val = _compute_historical(comp, at)
        else:
            val = comp.cached_value

        if val is None:
            return None
        variables[slug] = val

    return evaluate_pcl_formula(measure.formula, variables)


def _compute_historical(measure: Measure, at: datetime) -> Optional[float]:
    """Compute aggregate from datapoints at or before a given time."""
    dps = measure.datapoints.filter(created_at__lte=at).order_by("created_at")
    if not dps.exists():
        return None

    now = timezone.now()
    values = []
    confidences = []
    ages = []
    for dp in dps:
        values.append(dp.value)
        confidences.append(dp.confidence)
        ages.append((now - dp.created_at).total_seconds() / 86400)

    agg = compute_aggregate(
        values=values,
        confidences=confidences,
        timestamps_age_days=ages,
        decay_enabled=measure.decay_enabled,
        decay_halflife_days=measure.decay_halflife_days,
    )
    return agg["value"]
```

- [ ] **Step 4: Run tests**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest pcl/tests/test_service.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pcl/service.py pcl/tests/test_service.py
git commit -m "feat(pcl): service layer — read/write/set_target with Bayesian aggregation and formula resolution"
```

---

### Task 7: API endpoints (CRUD + search)

**Files:**
- Create: `pcl/urls.py`, `pcl/views.py`, `pcl/tests/test_views.py`
- Modify: `svend/urls.py`

- [ ] **Step 1: Write failing tests for API endpoints**

```python
# pcl/tests/test_views.py
import json
import pytest
from django.test import TestCase, Client
from conftest import make_user, make_tenant, make_membership, SECURE_OFF


@SECURE_OFF
class MeasureAPITest(TestCase):
    def setUp(self):
        self.user = make_user("api@test.com", tier="team")
        self.tenant = make_tenant("API Org", slug="api-org", plan="team")
        make_membership(self.tenant, self.user)
        self.client = Client()
        self.client.force_login(self.user)

    def test_create_measure(self):
        resp = self.client.post(
            "/api/pcl/measures/create/",
            json.dumps({
                "name": "Press CT",
                "slug": "press-ct",
                "unit": "sec",
                "measure_type": "process",
                "value_type": "continuous",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["data"]["slug"] == "press-ct"

    def test_list_measures(self):
        # Create one first
        from pcl.models import Measure
        Measure.objects.create(
            tenant=self.tenant, name="CT", slug="ct",
            unit="sec", measure_type="process", value_type="continuous",
            created_by=self.user.email,
        )
        resp = self.client.get("/api/pcl/measures/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 1

    def test_write_datapoint(self):
        from pcl.models import Measure
        m = Measure.objects.create(
            tenant=self.tenant, name="CT", slug="ct-dp",
            unit="sec", measure_type="process", value_type="continuous",
            created_by=self.user.email,
        )
        resp = self.client.post(
            f"/api/pcl/measures/{m.slug}/write/",
            json.dumps({
                "value": 45.0,
                "value_confirm": 45.0,
                "source_type": "manual",
                "observation_count": 1,
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["value"] == 45.0
        assert data["data"]["confidence"] > 0

    def test_write_rejects_mismatched_confirm(self):
        from pcl.models import Measure
        m = Measure.objects.create(
            tenant=self.tenant, name="CT", slug="ct-confirm",
            unit="sec", measure_type="process", value_type="continuous",
            created_by=self.user.email,
        )
        resp = self.client.post(
            f"/api/pcl/measures/{m.slug}/write/",
            json.dumps({
                "value": 45.0,
                "value_confirm": 46.0,
                "source_type": "manual",
                "observation_count": 1,
            }),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "confirm" in resp.json()["error"].lower()

    def test_read_measure(self):
        from pcl.models import Measure
        from pcl.service import write
        m = Measure.objects.create(
            tenant=self.tenant, name="CT", slug="ct-read",
            unit="sec", measure_type="process", value_type="continuous",
            created_by=self.user.email,
        )
        write(tenant=self.tenant, measure_slug="ct-read",
              value=45.0, source_type="manual",
              observation_count=1, actor=self.user.email)
        resp = self.client.get(f"/api/pcl/measures/{m.slug}/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["cached_value"] == 45.0

    def test_search_measures(self):
        from pcl.models import Measure
        Measure.objects.create(
            tenant=self.tenant, name="Press A Cycle Time", slug="press-a-ct",
            unit="sec", measure_type="process", value_type="continuous",
            created_by=self.user.email,
        )
        Measure.objects.create(
            tenant=self.tenant, name="Weld B Uptime", slug="weld-b-uptime",
            unit="%", measure_type="process", value_type="proportion",
            created_by=self.user.email,
        )
        resp = self.client.get("/api/pcl/measures/search/?q=press")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["slug"] == "press-a-ct"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest pcl/tests/test_views.py -v`
Expected: FAIL — 404s (no URL routing)

- [ ] **Step 3: Create URL routing**

```python
# pcl/urls.py
from django.urls import path
from . import views

app_name = "pcl"

urlpatterns = [
    path("measures/", views.list_measures, name="measure-list"),
    path("measures/create/", views.create_measure, name="measure-create"),
    path("measures/search/", views.search_measures, name="measure-search"),
    path("measures/<slug:slug>/", views.detail_measure, name="measure-detail"),
    path("measures/<slug:slug>/write/", views.write_datapoint, name="measure-write"),
    path("measures/<slug:slug>/history/", views.measure_history, name="measure-history"),
    path("measures/<slug:slug>/target/", views.set_measure_target, name="measure-target"),
]
```

Add to `svend/urls.py`:
```python
path("api/pcl/", include("pcl.urls")),
```

- [ ] **Step 4: Create views**

```python
# pcl/views.py
import json
import logging

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from accounts.permissions import require_auth
from qms_core.permissions import require_tenant

from .models import Measure, Datapoint
from . import service

logger = logging.getLogger(__name__)


@require_http_methods(["GET"])
@require_auth
def list_measures(request):
    tenant, err = require_tenant(request.user)
    if err:
        return err
    measures = Measure.objects.filter(tenant=tenant)
    return JsonResponse({
        "status": "success",
        "data": [m.to_dict() for m in measures],
    })


@require_http_methods(["POST"])
@require_auth
def create_measure(request):
    tenant, err = require_tenant(request.user)
    if err:
        return err
    data = json.loads(request.body)
    required = ["name", "slug", "measure_type", "value_type"]
    for f in required:
        if not data.get(f):
            return JsonResponse({"status": "error", "error": f"Missing required field: {f}"}, status=400)
    try:
        m = Measure.objects.create(
            tenant=tenant,
            name=data["name"],
            slug=data["slug"],
            definition=data.get("definition", ""),
            unit=data.get("unit", ""),
            measure_type=data["measure_type"],
            value_type=data["value_type"],
            range_min=data.get("range_min"),
            range_max=data.get("range_max"),
            formula=data.get("formula"),
            parent_type=data.get("parent_type", ""),
            parent_id=data.get("parent_id"),
            created_by=request.user.email,
        )
        return JsonResponse({"status": "success", "data": m.to_dict()})
    except Exception as e:
        return JsonResponse({"status": "error", "error": str(e)}, status=400)


@require_http_methods(["GET"])
@require_auth
def detail_measure(request, slug):
    tenant, err = require_tenant(request.user)
    if err:
        return err
    try:
        meta = service.read_with_meta(tenant, slug)
        m = Measure.objects.get(tenant=tenant, slug=slug, is_deleted=False)
        result = m.to_dict()
        result.update(meta)
        return JsonResponse({"status": "success", "data": result})
    except ValueError as e:
        return JsonResponse({"status": "error", "error": str(e)}, status=404)


@require_http_methods(["POST"])
@require_auth
def write_datapoint(request, slug):
    tenant, err = require_tenant(request.user)
    if err:
        return err
    data = json.loads(request.body)

    value = data.get("value")
    value_confirm = data.get("value_confirm")
    if value is None:
        return JsonResponse({"status": "error", "error": "Missing 'value'"}, status=400)
    if value_confirm is not None and float(value) != float(value_confirm):
        return JsonResponse({"status": "error", "error": "Value and confirmation do not match. Please re-enter."}, status=400)

    try:
        dp = service.write(
            tenant=tenant,
            measure_slug=slug,
            value=float(value),
            source_type=data.get("source_type", "manual"),
            observation_count=int(data.get("observation_count", 1)),
            actor=request.user.email,
            source_ref_type=data.get("source_ref_type", ""),
            source_ref_id=data.get("source_ref_id"),
            notes=data.get("notes", ""),
        )
        return JsonResponse({
            "status": "success",
            "data": {
                "id": str(dp.id),
                "value": dp.value,
                "confidence": dp.confidence,
                "source_type": dp.source_type,
            },
        })
    except ValueError as e:
        return JsonResponse({"status": "error", "error": str(e)}, status=400)


@require_http_methods(["GET"])
@require_auth
def measure_history(request, slug):
    tenant, err = require_tenant(request.user)
    if err:
        return err
    try:
        m = Measure.objects.get(tenant=tenant, slug=slug, is_deleted=False)
    except Measure.DoesNotExist:
        return JsonResponse({"status": "error", "error": f"Measure '{slug}' not found"}, status=404)
    dps = m.datapoints.all()[:100]
    return JsonResponse({
        "status": "success",
        "data": [{
            "id": str(dp.id),
            "value": dp.value,
            "confidence": dp.confidence,
            "source_type": dp.source_type,
            "observation_count": dp.observation_count,
            "notes": dp.notes,
            "created_at": dp.created_at.isoformat(),
            "actor": dp.actor,
        } for dp in dps],
    })


@require_http_methods(["POST"])
@require_auth
def set_measure_target(request, slug):
    tenant, err = require_tenant(request.user)
    if err:
        return err
    data = json.loads(request.body)
    if "target_value" not in data:
        return JsonResponse({"status": "error", "error": "Missing 'target_value'"}, status=400)
    try:
        from datetime import date as date_type
        target_date = None
        if data.get("target_date"):
            target_date = date_type.fromisoformat(data["target_date"])
        t = service.set_target(
            tenant=tenant,
            measure_slug=slug,
            target_value=float(data["target_value"]),
            source=data.get("source", "manual"),
            target_date=target_date,
            actor=request.user.email,
        )
        return JsonResponse({
            "status": "success",
            "data": {
                "id": str(t.id),
                "target_value": t.target_value,
                "target_date": str(t.target_date) if t.target_date else None,
                "source": t.source,
            },
        })
    except ValueError as e:
        return JsonResponse({"status": "error", "error": str(e)}, status=400)


@require_http_methods(["GET"])
@require_auth
def search_measures(request):
    tenant, err = require_tenant(request.user)
    if err:
        return err
    q = request.GET.get("q", "").strip()
    if not q:
        return JsonResponse({"status": "success", "data": []})
    measures = Measure.objects.filter(
        tenant=tenant,
        is_deleted=False,
    ).filter(
        models.Q(name__icontains=q) | models.Q(slug__icontains=q) | models.Q(definition__icontains=q)
    )[:20]
    return JsonResponse({
        "status": "success",
        "data": [m.to_dict() for m in measures],
    })
```

Add missing import at the top of views.py:
```python
from django.db.models import Q
```

And update the `search_measures` view to use `Q` instead of `models.Q`:
```python
    ).filter(
        Q(name__icontains=q) | Q(slug__icontains=q) | Q(definition__icontains=q)
    )[:20]
```

- [ ] **Step 5: Run tests**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest pcl/tests/test_views.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add pcl/urls.py pcl/views.py pcl/tests/test_views.py svend/urls.py
git commit -m "feat(pcl): API endpoints — CRUD, write with confirm, search, history, target"
```

---

### Task 8: VSM ↔ PCL binding — backend

**Files:**
- Modify: `vsm/views.py` (add bind/unbind endpoints, modify `_parse_step_context`)
- Modify: `vsm/urls.py`
- Test: `pcl/tests/test_vsm_binding.py`

- [ ] **Step 1: Write failing tests for VSM binding**

```python
# pcl/tests/test_vsm_binding.py
import json
import pytest
from django.test import TestCase, Client
from conftest import make_user, make_tenant, make_membership, SECURE_OFF


@SECURE_OFF
class VSMBindingTest(TestCase):
    def setUp(self):
        self.user = make_user("vsm@test.com", tier="team")
        self.tenant = make_tenant("VSM Org", slug="vsm-org", plan="team")
        make_membership(self.tenant, self.user)
        self.client = Client()
        self.client.force_login(self.user)

        # Create a VSM with a process step
        from agents_api.models import ValueStreamMap
        self.vsm = ValueStreamMap.objects.create(
            owner=self.user,
            tenant=self.tenant,
            name="Test VSM",
            process_steps=[{
                "id": "step-1",
                "name": "Press A",
                "cycle_time": 45.0,
                "changeover_time": 1800.0,
                "uptime": 95.0,
                "x": 100, "y": 200,
            }],
        )

        # Create PCL measures
        from pcl.models import Measure
        from pcl.service import write
        self.ct_measure = Measure.objects.create(
            tenant=self.tenant, name="Press A CT", slug="press-a-ct",
            unit="sec", measure_type="process", value_type="continuous",
            created_by=self.user.email,
        )
        write(tenant=self.tenant, measure_slug="press-a-ct",
              value=54.0, source_type="doe",
              observation_count=30, actor=self.user.email)

    def test_bind_step_to_measure(self):
        resp = self.client.post(
            f"/api/vsm/{self.vsm.id}/bind-pcl/step-1/",
            json.dumps({
                "field": "cycle_time",
                "measure_slug": "press-a-ct",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        self.vsm.refresh_from_db()
        step = self.vsm.process_steps[0]
        assert step["pcl_bindings"]["cycle_time"] == "press-a-ct"

    def test_unbind_step(self):
        # Bind first
        self.vsm.process_steps[0]["pcl_bindings"] = {"cycle_time": "press-a-ct"}
        self.vsm.save()

        resp = self.client.post(
            f"/api/vsm/{self.vsm.id}/unbind-pcl/step-1/",
            json.dumps({"field": "cycle_time"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        self.vsm.refresh_from_db()
        step = self.vsm.process_steps[0]
        bindings = step.get("pcl_bindings", {})
        assert "cycle_time" not in bindings
        # Inline value should be snapshot of PCL value
        assert step["cycle_time"] == 54.0

    def test_parse_step_context_resolves_binding(self):
        """When a step has pcl_bindings, _parse_step_context returns PCL value."""
        self.vsm.process_steps[0]["pcl_bindings"] = {"cycle_time": "press-a-ct"}
        self.vsm.save()

        from vsm.views import _parse_step_context
        ctx = _parse_step_context(self.vsm.process_steps[0], self.vsm)
        # PCL value is 54.0 (from DOE), not inline 45.0
        assert ctx["ct"] == 54.0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest pcl/tests/test_vsm_binding.py -v`
Expected: FAIL — 404 or assertion errors

- [ ] **Step 3: Add bind/unbind endpoints to vsm/views.py**

Add at end of `vsm/views.py`:

```python
# =============================================================================
# PCL BINDING
# =============================================================================

@gated_paid
@require_http_methods(["POST"])
def bind_step_to_pcl(request, vsm_id, step_id):
    """Bind a step field to a PCL measure."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)
    data = json.loads(request.body)
    field = data.get("field")
    measure_slug = data.get("measure_slug")
    if not field or not measure_slug:
        return JsonResponse({"error": "field and measure_slug required"}, status=400)

    steps = vsm.process_steps or []
    for step in steps:
        if step.get("id") == step_id:
            if "pcl_bindings" not in step:
                step["pcl_bindings"] = {}
            step["pcl_bindings"][field] = measure_slug
            vsm.save()
            return JsonResponse({"status": "success", "bindings": step["pcl_bindings"]})

    return JsonResponse({"error": "Step not found"}, status=404)


@gated_paid
@require_http_methods(["POST"])
def unbind_step_from_pcl(request, vsm_id, step_id):
    """Unbind a step field from PCL, snapshot current PCL value into inline."""
    vsm = get_object_or_404(ValueStreamMap, id=vsm_id, owner=request.user)
    data = json.loads(request.body)
    field = data.get("field")
    if not field:
        return JsonResponse({"error": "field required"}, status=400)

    steps = vsm.process_steps or []
    for step in steps:
        if step.get("id") == step_id:
            bindings = step.get("pcl_bindings", {})
            slug = bindings.pop(field, None)
            if slug:
                # Snapshot PCL value into inline field
                try:
                    from pcl.service import read
                    from qms_core.permissions import get_tenant
                    tenant = get_tenant(request.user)
                    pcl_value = read(tenant, slug)
                    if pcl_value is not None:
                        step[field] = pcl_value
                except Exception:
                    pass  # Keep existing inline value
            vsm.save()
            return JsonResponse({"status": "success", "bindings": bindings})

    return JsonResponse({"error": "Step not found"}, status=404)
```

- [ ] **Step 4: Add URL routes to vsm/urls.py**

Add before the closing bracket of `urlpatterns`:
```python
    # PCL binding
    path("<uuid:vsm_id>/bind-pcl/<str:step_id>/", views.bind_step_to_pcl, name="vsm_bind_pcl"),
    path("<uuid:vsm_id>/unbind-pcl/<str:step_id>/", views.unbind_step_from_pcl, name="vsm_unbind_pcl"),
```

- [ ] **Step 5: Modify _parse_step_context to resolve PCL bindings**

In `vsm/views.py`, at the top of `_parse_step_context()` (after the `_float` helper), add PCL resolution:

```python
    # Resolve PCL bindings (Approach C: binding wins when present)
    pcl_bindings = step.get("pcl_bindings", {})
    if pcl_bindings:
        try:
            from pcl.service import read as pcl_read
            from qms_core.permissions import get_tenant
            # VSM has tenant FK
            tenant = getattr(vsm, 'tenant', None)
            if tenant:
                for field_name, slug in pcl_bindings.items():
                    try:
                        pcl_val = pcl_read(tenant, slug)
                        if pcl_val is not None:
                            step = {**step, field_name: pcl_val}  # shadow with PCL value
                    except (ValueError, Exception):
                        pass  # fall back to inline value
        except ImportError:
            pass  # PCL not installed yet — use inline values
```

- [ ] **Step 6: Run tests**

Run: `cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 -m pytest pcl/tests/test_vsm_binding.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add vsm/views.py vsm/urls.py pcl/tests/test_vsm_binding.py
git commit -m "feat(pcl): VSM binding — bind/unbind endpoints, _parse_step_context resolves PCL values"
```

---

### Task 9: VSM ↔ PCL binding — frontend

**Files:**
- Modify: `templates/vsm.html` — add PCL bind UI elements
- Modify: `static/js/vsm.js` — add bind/unbind/search logic

This task is frontend-only. No new Python tests — the backend was tested in Task 8.

- [ ] **Step 1: Add PCL binding HTML to templates/vsm.html**

After the shifts field group (around line 1062), add:

```html
<!-- PCL Measure Binding -->
<div id="pcl-binding-panel" style="margin-top:8px; padding:6px 0; border-top:1px solid var(--border); display:none;">
    <div class="smp-section-title">PCL Bindings</div>
    <div id="pcl-binding-list" style="font-size:0.7rem;"></div>
</div>
```

- [ ] **Step 2: Add PCL binding JS to static/js/vsm.js**

After the `showProperties()` function, add:

```javascript
function _showPCLBindings(step) {
    const panel = document.getElementById('pcl-binding-panel');
    if (!panel || !currentVSM) { return; }
    const bindings = step.pcl_bindings || {};
    const bindableFields = ['cycle_time', 'changeover_time', 'uptime', 'demand_rate', 'batch_size'];
    let html = '';
    bindableFields.forEach(field => {
        const slug = bindings[field];
        const label = field.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        if (slug) {
            html += `<div style="display:flex; align-items:center; gap:4px; margin:2px 0; padding:3px 6px; background:rgba(74,159,110,0.08); border-radius:3px;">`;
            html += `<span style="color:var(--success);">\u26d3</span>`;
            html += `<span style="flex:1;">${label} \u2192 <strong>${slug}</strong></span>`;
            html += `<button onclick="_unbindPCL('${step.id}','${field}')" style="font-size:0.6rem; padding:1px 4px; background:var(--bg-tertiary); border:1px solid var(--border); border-radius:2px; cursor:pointer;">\u2715</button>`;
            html += `</div>`;
        } else {
            html += `<div style="display:flex; align-items:center; gap:4px; margin:2px 0;">`;
            html += `<span style="color:var(--text-dim);">${label}</span>`;
            html += `<button onclick="_bindPCLPrompt('${step.id}','${field}')" style="font-size:0.6rem; padding:1px 6px; background:var(--bg-tertiary); border:1px solid var(--border); border-radius:2px; cursor:pointer; margin-left:auto;">\u26d3 Bind</button>`;
            html += `</div>`;
        }
    });
    panel.innerHTML = `<div class="smp-section-title" style="margin-bottom:4px;">PCL Bindings</div>${html}`;
    panel.style.display = 'block';
}

function _bindPCLPrompt(stepId, field) {
    const slug = prompt(`Enter PCL measure slug to bind to ${field.replace(/_/g, ' ')}:`);
    if (!slug) return;
    fetch(`/api/vsm/${currentVSM.id}/bind-pcl/${stepId}/`, {
        method: 'POST',
        credentials: 'include',
        headers: {'Content-Type': 'application/json', 'X-CSRFToken': getCsrfToken()},
        body: JSON.stringify({field: field, measure_slug: slug}),
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'success') {
            const step = (currentVSM.process_steps || []).find(s => s.id === stepId);
            if (step) {
                step.pcl_bindings = data.bindings;
                _showPCLBindings(step);
                saveVSM();
            }
        } else {
            alert(data.error || 'Binding failed');
        }
    });
}

function _unbindPCL(stepId, field) {
    fetch(`/api/vsm/${currentVSM.id}/unbind-pcl/${stepId}/`, {
        method: 'POST',
        credentials: 'include',
        headers: {'Content-Type': 'application/json', 'X-CSRFToken': getCsrfToken()},
        body: JSON.stringify({field: field}),
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'success') {
            const step = (currentVSM.process_steps || []).find(s => s.id === stepId);
            if (step) {
                step.pcl_bindings = data.bindings;
                _showPCLBindings(step);
                renderVSM();
                saveVSM();
            }
        }
    });
}
```

- [ ] **Step 3: Wire _showPCLBindings into showProperties()**

In `showProperties()`, before `selectedElement = element;`, add:

```javascript
    _showPCLBindings(element);
```

- [ ] **Step 4: Verify JS syntax and collectstatic**

Run:
```bash
node -e "const fs=require('fs'); try{new Function(fs.readFileSync('/home/eric/kjerne/static/js/vsm.js','utf8'));console.log('JS OK')}catch(e){console.log('ERROR: '+e.message)}"
cd ~/kjerne && set -a && source /etc/svend/env && set +a && python3 manage.py collectstatic --noinput 2>&1 | tail -1
```
Expected: `JS OK` and `static files copied`.

- [ ] **Step 5: Commit**

```bash
git add static/js/vsm.js templates/vsm.html
git commit -m "feat(pcl): VSM frontend — PCL bind/unbind UI in step properties panel"
```

---

### Task 10: Integration verification

**Files:** None new — this task runs the full test suite and verifies imports.

- [ ] **Step 1: Run full PCL test suite**

Run:
```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a
python3 -m pytest pcl/ -v --tb=short
```
Expected: All tests PASS (models, confidence, aggregation, formulas, service, views, vsm_binding).

- [ ] **Step 2: Verify Django can load**

Run:
```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a
python3 -c "
import os; os.environ.setdefault('DJANGO_SETTINGS_MODULE','svend.settings')
import django; django.setup()
from pcl.models import Measure, Datapoint, MeasureTarget
from pcl.service import read, write, read_with_meta, set_target
from pcl.formulas import evaluate_pcl_formula, extract_slugs
from pcl.confidence import compute_confidence
from pcl.aggregation import compute_aggregate
print('All PCL imports OK')
"
```
Expected: `All PCL imports OK`

- [ ] **Step 3: Verify URL routing**

Run:
```bash
cd ~/kjerne && set -a && source /etc/svend/env && set +a
python3 -c "
import os; os.environ.setdefault('DJANGO_SETTINGS_MODULE','svend.settings')
import django; django.setup()
from django.urls import reverse
print(reverse('pcl:measure-list'))
print(reverse('pcl:measure-create'))
print(reverse('pcl:measure-search'))
print(reverse('pcl:measure-write', kwargs={'slug': 'test'}))
print('URL routing OK')
"
```
Expected: All URLs resolve.

- [ ] **Step 4: Commit final state**

```bash
git add -A
git status
# If clean, no commit needed. If there are uncommitted changes:
git commit -m "chore(pcl): integration verification pass"
```

---

## Spec Coverage Checklist

| Spec Requirement | Task |
|-----------------|------|
| Measure model (name, slug, definition, unit, types, range, formula, parent) | Task 1 |
| Datapoint model (immutable, SynaraImmutableLog, provenance, confidence) | Task 2 |
| MeasureTarget model (working layer) | Task 2 |
| Confidence auto-computation (source_type × n, plateau curves) | Task 2, 3 |
| Bayesian weighted aggregation (consistency penalty, outlier suppression) | Task 4 |
| Recency decay (optional, per-measure halflife) | Task 4 |
| Shift detection | Task 4 |
| Cached aggregate on Measure (incremental update) | Task 1 (fields), Task 6 (logic) |
| Formula evaluation ([slug] syntax, AST walker, safety guards) | Task 5 |
| Calculated measure resolution (chain-call components) | Task 6 |
| pcl.read() / pcl.write() / pcl.read_with_meta() / pcl.set_target() | Task 6 |
| Manual entry confirmation (value_confirm field) | Task 7 |
| API endpoints (CRUD, search, write, history, target) | Task 7 |
| VSM binding — Approach C (pcl_bindings on step JSON) | Task 8 |
| _parse_step_context resolves PCL bindings | Task 8 |
| VSM bind/unbind UI | Task 9 |
| Unbind snapshots PCL value to inline | Task 8 |
| String-key references (not GenericFK) | Task 1, 2 |
| SynaraEntity / SynaraImmutableLog base classes | Task 1, 2 |
| Event schemas (defined in spec, not wired yet — Phase 2) | — (future) |
| Reconciliation with existing systems | — (future) |
| Cortex integration | — (future, Phase 2) |

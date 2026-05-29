# Conference: Semantic Type System for SVEND Flowchart Workspace

**Date:** 2026-05-12
**Technique:** S1/S2 Dialectical Debate

---

## S1 (Innovator) Position

### 1. What's Really at Stake

This isn't a type system decision. It's a **language design** decision.

Every flowchart connection is a sentence: "the Cpk from this capability study flows into this control chart's threshold." The type system is the grammar of that language. Get it wrong and you get one of two failure modes:

- **Too loose:** Users cable anything to anything, get garbage results, blame SVEND. Raj walks. The tool is "decorative."
- **Too rigid:** Users can't express valid workflows because the type system doesn't understand that a Cpk and a Ppk are both capability indices. Tomasz gives up. The tool is hostile.

The real stake: **SVEND's type system IS its moat.** Anyone can wrap a statistics library in a UI. The thing that makes SVEND worth paying for is that it *understands manufacturing methodology* — it knows that a severity score from an FMEA can feed a risk priority matrix but shouldn't feed a control limit calculation. That knowledge, encoded structurally, is what Raj means by "as rigorous as units in engineering."

The conservative approach (flat 4-color + string subtypes) is a trap. It looks simple today. In 6 months with 50 devices, you'll have a ball of `if output_type.startswith("metric:")` conditionals scattered across every device, every template validator, and every renderer. You'll have invented a type system anyway — just a bad, implicit, untestable one.

### 2. Recommended Approach: Structural Typing with Semantic Tags

Not nominal types. Not a rigid hierarchy. **Structural typing** — a port accepts a connection if the output's shape satisfies the input's contract.

The insight: manufacturing data doesn't form a clean tree. A Cpk is simultaneously:
- A scalar metric
- A capability index
- A ratio (dimensionless)
- Something with a confidence interval
- Something derived from a distribution fit

A tree hierarchy forces you to pick ONE of those. Structural typing lets a port say "I need a scalar with a confidence interval" and accept both Cpk and a reliability estimate.

**But** — pure structural typing is too abstract for users. Raj thinks in domain terms, not shape contracts. So we layer **semantic tags** on top of structural shapes. The tags are what the UI shows. The shapes are what the validator checks.

### 3. The Architecture

#### Core Data Structures

```python
class Shape(Enum):
    """Physical shape of data flowing through a cable."""
    SCALAR = "scalar"           # single number
    VECTOR = "vector"           # 1D array of numbers
    MATRIX = "matrix"           # 2D array
    SERIES = "series"           # ordered sequence (time-indexed or not)
    DISTRIBUTION = "distribution"  # parameters of a fitted distribution
    TABLE = "table"             # columnar data (DataFrame-like)
    RECORD = "record"           # key-value struct
    TEXT = "text"               # free text / narrative
    CHART_SPEC = "chart_spec"   # ForgeViz ChartSpec
    BOOLEAN = "boolean"         # pass/fail, yes/no

class Trait(Enum):
    """
    Structural properties. A port output can have many traits.
    A port input declares which traits it REQUIRES.
    Compatibility = output traits >= input required traits.
    """
    # Numeric properties
    HAS_CONFIDENCE_INTERVAL = "has_ci"
    HAS_UNITS = "has_units"
    DIMENSIONLESS = "dimensionless"
    BOUNDED_0_1 = "bounded_0_1"
    NON_NEGATIVE = "non_negative"
    # Statistical properties
    FROM_DISTRIBUTION_FIT = "from_dist_fit"
    FROM_HYPOTHESIS_TEST = "from_hyp_test"
    IS_INDEX = "is_index"
    IS_COUNT = "is_count"
    IS_RATE = "is_rate"
    # Domain properties
    IS_SPECIFICATION = "is_spec"
    IS_MEASUREMENT = "is_measurement"
    IS_PARAMETER = "is_parameter"
    IS_TEMPORAL = "is_temporal"
    IS_CATEGORICAL = "is_categorical"
    # Process/quality domain
    IS_RISK_SCORE = "is_risk_score"
    IS_CAPABILITY = "is_capability"
    IS_CONTROL_LIMIT = "is_control_limit"

@dataclass(frozen=True)
class SemanticTag:
    """Human-readable domain classification. For display and search."""
    category: str   # "metric", "data", "chart", "text", "config", "risk"
    name: str       # "cpk", "histogram", "spec_limits", "severity"
    label: str      # "Cpk", "Histogram", "Spec Limits", "Severity Score"
    color: str      # hex color for UI rendering
    @property
    def slug(self) -> str:
        return f"{self.category}:{self.name}"

@dataclass(frozen=True)
class PortType:
    """
    The complete type declaration for a port.
    Output ports declare what they PROVIDE: exact shape, all traits, semantic tag.
    Input ports declare what they REQUIRE: accepted shapes, minimum traits, accepted tags.
    """
    shape: Shape
    traits: FrozenSet[Trait] = field(default_factory=frozenset)
    tag: Optional[SemanticTag] = None
    accepts_shapes: FrozenSet[Shape] = field(default_factory=frozenset)
    accepts_tags: FrozenSet[str] = None  # INPUT ONLY: whitelist (None = any)
```

#### Compatibility Check (12 lines)

```python
def is_compatible(output: PortType, input_port: PortType) -> bool:
    # Rule 1: Shape
    accepted_shapes = {input_port.shape} | input_port.accepts_shapes
    if output.shape not in accepted_shapes:
        return False
    # Rule 2: Traits (output must have everything input requires)
    if not output.traits.issuperset(input_port.traits):
        return False
    # Rule 3: Tag whitelist (optional)
    if input_port.accepts_tags is not None:
        if output.tag is None or output.tag.slug not in input_port.accepts_tags:
            return False
    return True
```

#### Concrete Device Examples

**Capability Study:**
- Input `measurements`: Shape=SERIES, traits={IS_MEASUREMENT}
- Input `spec_limits`: Shape=RECORD, traits={IS_SPECIFICATION}
- Output `cpk`: Shape=SCALAR, traits={HAS_CI, DIMENSIONLESS, NON_NEGATIVE, IS_INDEX, IS_CAPABILITY, FROM_DIST_FIT}
- Output `ppk`: Shape=SCALAR, traits={HAS_CI, DIMENSIONLESS, NON_NEGATIVE, IS_INDEX, IS_CAPABILITY}
- Output `histogram`: Shape=CHART_SPEC
- Output `distribution_fit`: Shape=DISTRIBUTION, traits={FROM_DIST_FIT}

**Control Chart:**
- Input `measurements`: Shape=SERIES, traits={IS_MEASUREMENT}
- Input `control_limits`: Shape=RECORD, traits={IS_CONTROL_LIMIT}
- Output `chart`: Shape=CHART_SPEC
- Output `computed_limits`: Shape=RECORD, traits={IS_CONTROL_LIMIT}
- Output `mean`: Shape=SCALAR, traits={NON_NEGATIVE, HAS_UNITS}

**FMEA:**
- Input `capability_index`: Shape=SCALAR, traits={IS_CAPABILITY} — accepts Cpk OR Ppk
- Output `severity`: Shape=SCALAR, traits={IS_RISK_SCORE, NON_NEGATIVE, IS_COUNT}
- Output `rpn`: Shape=SCALAR, traits={IS_RISK_SCORE, NON_NEGATIVE}

**Monte Carlo:**
- Input `distribution`: Shape=DISTRIBUTION, traits={FROM_DIST_FIT}
- Input `threshold`: Shape=SCALAR, accepts_tags={"metric:cpk", "metric:ppk", ...} — BLOCKS severity scores via tag whitelist
- Input `spec_limits`: Shape=RECORD, traits={IS_SPECIFICATION}

#### Connection Traces

```
VALID: Capability Study [cpk] -> FMEA [capability_index]
  Shape: SCALAR in {SCALAR}                                          OK
  Traits: {has_ci, dimensionless, non_neg, is_index, is_cap, from_dist_fit}
          superset of {is_capability}                                OK
  Tags: accepts_tags is None                                         OK

INVALID: FMEA [severity] -> Monte Carlo [threshold]
  Shape: SCALAR in {SCALAR}                                          OK
  Traits: {is_risk_score, non_neg, is_count} superset of {}          OK
  Tags: "risk:severity" in {"metric:cpk", "metric:ppk", ...}?        BLOCKED

INVALID: FMEA [severity] -> Control Chart [measurements]
  Shape: SCALAR in {SERIES}?                                          BLOCKED (shape)

VALID: Capability Study [distribution_fit] -> Monte Carlo [distribution]
  Shape: DISTRIBUTION in {DISTRIBUTION}                               OK
  Traits: {from_dist_fit} superset of {from_dist_fit}                 OK

VALID: Control Chart [computed_limits] -> Control Chart [control_limits]
  Shape: RECORD in {RECORD}                                           OK
  Traits: {is_control_limit} superset of {is_control_limit}           OK
```

#### Extension Model

- **New tag, existing shape+traits:** Add to tag registry. Zero compatibility changes.
- **New trait:** Add to enum. Additive. Existing devices unaffected.
- **New shape:** Add to enum. Existing devices unaffected.
- **Type system versioned.** Saved connections store version. Migrations like Django migrations.

### 4. Risks Acknowledged

- **Trait explosion:** Enum could grow to 50+ members. Mitigated by additive independence.
- **Device authors get traits wrong:** Mitigated by DeviceLinter ("your cpk output lacks IS_CAPABILITY — it won't connect to risk tools").
- **`accepts_tags` maintenance:** Keep it rare (exception, not rule).
- **User confusion on rejection:** Mitigated by `compatibility_reason()` returning human-readable explanations.
- **More complex than flat strings:** Complexity in device definitions, not in compatibility check (12 lines) or UI (colors + labels).

### 5. What Gets Left on the Table if Conservative Wins

- **Cross-cutting compatibility.** Traits encode "is capability" once vs N times per device.
- **Principled rejection.** Shape+traits catch what flat types miss (p-value into Cpk slot).
- **Self-documenting ports.** Trait sets ARE the specification.
- **Teaching opportunity.** "Invalid: a risk score isn't a capability index" teaches methodology. "Type mismatch: risk:severity != metric:cpk" teaches nothing.
- **Composable path validation.** Future workflow-level checking via set intersection on trait paths.

### 6. Steel-Man Against This Position

"You're solving a problem you don't have yet, and the solution is load-bearing enough to be dangerous."

Flat string types (`metric:cpk`) are trivially understandable. The 3-rule check has three degrees of freedom per port — exponential bug surface. A device author could set wide-open shapes with no trait requirements, defeating the purpose. Or over-constrain with 8 required traits.

Ship flat types, build 20 devices, observe what connections are incorrectly accepted or rejected, and THEN design the structural system based on real data instead of speculation. You can always tighten a loose system; loosening a tight system breaks existing connections.

**S1's response:** The migration cost argument cuts the other way. Once 20 devices exist with flat strings, migrating to structural types means rewriting every device definition. Flat-to-structural is deferred payment with interest. The trait taxonomy risk is real, but traits are additive — splitting IS_CAPABILITY into two traits is non-breaking.

---

## S2 (Conservative) Position

### 1. What's Really at Stake

This isn't a type system design problem. It's a **commitment surface** problem.

Every type you define becomes a contract. Every connection rule is a semantic claim about manufacturing methodology that you're encoding in software.

The real stakes:

- **Too rigid early:** You encode assumptions about how 200+ analysis types relate to each other before you've built more than one. You'll be wrong. Changing it breaks templates, saved flowcharts, and user muscle memory.
- **Too loose early:** You ship Raj's "decorative" system. Users cable nonsense together.
- **Too complex early:** You build a rich type hierarchy for 200+ devices when you have 1 device. You spend weeks on type algebra that serves no user for months. Meanwhile, the MRR goal recedes.

The constraint that matters most: **you have 1 device today, maybe 5-10 in 3 months, maybe 30 in a year.** The type system for 200 devices is a different system than the type system for 5 devices, and you don't know enough yet to design the former.

### 2. Recommended Approach: Flat Types with Explicit Compatibility

Ship flat string types now. Make them structurally validated. Add semantic depth only when a real connection ambiguity forces it.

- **Phase 1 (now, 1-5 devices):** Flat types with explicit compatibility declarations per device. No hierarchy. No wildcards.
- **Phase 2 (5-15 devices):** Extract common patterns into type groups when repeated compatibility lists appear. Bottom-up.
- **Phase 3 (15+ devices):** If needed, formalize hierarchy from observed patterns.

Why? Because you'd be guessing otherwise. Is Cpk a `metric:capability` or `metric:index` or `metric:ratio`? It's all three. The "right" place in a hierarchy depends on what downstream device is consuming it. You don't know which axis matters until you build those downstream devices.

### 3. The Architecture

#### Core Type Definition

```python
@dataclass(frozen=True)
class PortType:
    """A port type is a simple string tag with a color category."""
    name: str       # e.g., "cpk", "raw_measurements"
    category: str   # one of: "metric", "data", "chart", "text"
    def __str__(self): return f"{self.category}:{self.name}"

class TypeRegistry:
    def __init__(self):
        self._types: dict[str, PortType] = {}
    def register(self, name: str, category: str) -> PortType:
        key = f"{category}:{name}"
        if key in self._types: return self._types[key]
        pt = PortType(name=name, category=category)
        self._types[key] = pt
        return pt
    def is_compatible(self, source: PortType, accepted: FrozenSet[PortType]) -> bool:
        return source in accepted

# Register types as devices need them
CPK = TYPES.register("cpk", "metric")
PPK = TYPES.register("ppk", "metric")
P_VALUE = TYPES.register("p_value", "metric")
RAW_MEASUREMENTS = TYPES.register("raw_measurements", "data")
SPEC_LIMITS = TYPES.register("spec_limits", "data")
HISTOGRAM = TYPES.register("histogram", "chart")
# ... more as devices are built
```

#### Port and Device Declaration

```python
@dataclass(frozen=True)
class OutputPort:
    key: str
    port_type: PortType
    label: str

@dataclass(frozen=True)
class InputPort:
    key: str
    accepted_types: FrozenSet[PortType]  # EXPLICIT set — no wildcards
    label: str
    required: bool = True
```

#### Compatibility Check (1 line)

```python
def is_compatible(source: OutputPort, target: InputPort) -> bool:
    return source.port_type in target.accepted_types
```

#### Concrete Device Examples

**Capability Study:**
```python
input_ports = [
    InputPort("measurements", frozenset([RAW_MEASUREMENTS]), "Measurement Data"),
    InputPort("spec_limits", frozenset([SPEC_LIMITS]), "Specification Limits"),
]
output_ports = [
    OutputPort("cpk", CPK, "Cpk"),
    OutputPort("ppk", PPK, "Ppk"),
    OutputPort("mean", MEAN, "Sample Mean"),
    OutputPort("histogram", HISTOGRAM, "Histogram"),
    OutputPort("summary_stats", SUMMARY_STATS, "Summary Statistics"),
]
```

**Control Plan (accepting capability):**
```python
InputPort("capability", frozenset([CPK, PPK]), "Process Capability")
# p_value NOT in set — cannot connect. Explicit exclusion.
```

**Monte Carlo (accepting metrics):**
```python
InputPort("metric_input", frozenset([CPK, PPK, PP, MEAN, SIGMA_LEVEL]), "Input Metric")
# FMEA severity NOT in set. Each addition is a deliberate decision.
```

#### Key Design Choices

- **Config values are NOT port types.** Subgroup size, alpha, confidence level are device configuration set in the config panel, not cabled in.
- **No wildcards.** If Monte Carlo accepts "any metric," enumerate them. The list grows deliberately.
- **Charts never connect to analytical inputs.** Enforced by absence from accepted_types, not by category rule.
- **New types:** `TYPES.register("gage_rr", "metric")` — no migration, no hierarchy update. Existing devices don't accept it (correct default). Update specific devices if they should.
- **Type renames:** Alias table, stays empty until needed.

#### Connection Traces

```
VALID: capability_study.cpk -> control_plan.capability
  CPK in frozenset([CPK, PPK])                                       OK

INVALID: capability_study.p_value -> control_plan.capability
  P_VALUE in frozenset([CPK, PPK])                                    BLOCKED

VALID: fmea.results -> control_plan.risk_input
  FMEA_RESULTS in frozenset([FMEA_RESULTS, RISK_SCORES])             OK

INVALID: capability_study.summary_stats -> xbar_r_chart.data
  SUMMARY_STATS in frozenset([RAW_MEASUREMENTS, SUBGROUPED_DATA])     BLOCKED
```

#### Evolution Path

Phase 2 (when pain arrives): Named groups replace repeated frozensets.

```python
# Bottom-up: observed pattern across 3+ devices
CAPABILITY_INDICES = frozenset([CPK, PPK, PP, CPM])

# Find-and-replace in device declarations
InputPort("capability", CAPABILITY_INDICES, "Process Capability")
```

No schema migration. No saved flowchart changes. Wire format (`"metric:cpk"`) unchanged.

### 4. Risks Acknowledged

- **Combinatorial explosion of accepted_types.** At 50 types, maintaining lists is tedious. Mitigated by deferred type groups.
- **Missing cross-cutting patterns.** Duplication across devices. Accepted as cost of avoiding premature abstraction.
- **Raj may find early system decorative.** Latent rigor until device count grows.
- **May be wrong about hierarchy timing.** Quality domain IS well-understood. But software representation of domain is not.

### 5. What Goes Wrong If the Ambitious Approach Fails

- **Hierarchy fossilizes prematurely.** Bayesian capability posterior doesn't fit neatly in the tree. You either force-fit it or restructure (breaking connections).
- **Wildcards hide bugs.** `metric:*` silently accepts severity_score in Monte Carlo.
- **Type algebra becomes the tar pit.** Weeks debating whether `data:timeseries:measurements` is a subtype of `data:tabular` instead of building devices.
- **Migration debt compounds.** Two hierarchy changes per quarter = 8 migrations/year touching every saved flowchart. One developer.

### 6. Steel-Man Against This Position

"You're building a type system for 5 devices when you know you'll need one for 200. The explicit-list approach collapses at large scale, and by then you have production data in the flat schema. A well-designed hierarchy encodes known manufacturing taxonomy. Restructuring one hierarchy is cheaper than updating 40 frozensets across 80 devices."

**S2's response:** Flat-to-grouped is a **refactor** (additive, mechanical). Wrong-hierarchy-to-right-hierarchy is a **migration** (destructive, semantic). I'll take the refactor.

---

## Conference Synthesis

### Agreements

These are more substantial than the surface framing suggests:

1. **The type system IS SVEND's moat.** Both agree. S1 says it explicitly. S2 treats every type as a "contract" and a "semantic claim about manufacturing methodology encoded in software."

2. **Both failure modes are real.** Too-loose (decorative/garbage) and too-rigid (hostile/blocking valid workflows). Neither dismisses either risk.

3. **Config values are NOT port types.** Subgroup size, alpha, confidence level are device configuration, not cable types. This boundary is settled.

4. **Output ports produce one type. Input ports declare what they accept.** Identical data modeling instincts.

5. **Compatibility is checked at connection time, not execution time.**

6. **New types must be additive.** No existing device breaks when you register a new type.

7. **Charts and text never connect to analytical inputs.** Both enforce this — S1 via shape mismatch, S2 via absence from accepted_types.

8. **At 200 devices, you need something richer than flat strings.** The disagreement is about when and how, not whether.

### Disagreements

**A. When to encode structure**

- S1: Encode structural typing (shapes, traits, semantic tags) now, before building more devices. Manufacturing taxonomy is known. Deferring creates implicit, untestable type systems.
- S2: Ship flat types now. Extract structure bottom-up when real ambiguities appear. Refactoring from flat to structured is cheaper than migrating from wrong structure to right structure.

**B. Compatibility mechanism**

- S1: Three-rule check (shape match AND traits superset AND optional tag whitelist). Compatibility is computed from structural properties.
- S2: Set membership (`source_type in accepted_types`). Compatibility is declared explicitly per input port. No inference.

**C. Cross-cutting properties (traits)**

- S1: Traits like IS_CAPABILITY should exist now. They encode "I accept anything that is a capability index" once, instead of repeating the list per device.
- S2: Enumerate explicitly. When 3+ devices share 5+ types, introduce named groups bottom-up. Until then, duplication is manageable and explicitness is valuable.

**D. Wildcard/inference behavior**

- S1: Traits provide principled "wildcards" — a future type tagged IS_CAPABILITY is auto-accepted wherever capability is required.
- S2: No auto-acceptance. Each new type's inclusion in each port is a deliberate human decision. Implicit acceptance is a bug vector.

**E. Error surface and debuggability**

- S1: Three rules with clear semantics. `compatibility_reason()` explains which rule failed.
- S2: One rule. "Type not in accepted set" is the only possible failure. Three degrees of freedom = exponentially more bugs.

### Crux of Each Disagreement

**Crux A (When to encode):**
- S1 believes: the manufacturing taxonomy is stable enough to encode now. Deferring creates accidental complexity.
- S2 believes: the mapping from manufacturing taxonomy to *software types* is not known until you've built 15+ devices. Textbook taxonomy != port type taxonomy.
- **If** the first 10 devices reveal that shape+trait model maps cleanly with few surprises, **S1 wins.** If trait assignments are contested, frequently revised, or need special cases, **S2 wins.**

**Crux B (Compatibility mechanism):**
- S1 believes: computed compatibility scales sublinearly — new type with right traits auto-works everywhere.
- S2 believes: computed compatibility hides bugs — a new type might share traits with existing types but be methodologically invalid for those ports.
- **If** most new types should be accepted by most ports sharing their traits, **S1 wins** (enumeration is busywork). **If** new types frequently share traits but shouldn't be accepted by the same ports, **S2 wins** (implicit acceptance is dangerous).

**Crux C (Cross-cutting properties):**
- S1 believes: "Cpk and Ppk are both capability indices" is AIAG, not speculation.
- S2 believes: whether every device that accepts Cpk should also accept Ppk is a device-specific decision, not a type-system decision.
- **If** capability-index-ness is generally sufficient to determine compatibility, **S1 wins.** **If** ports frequently want Cpk but not Ppk, traits are wrong abstraction and **S2 wins.**

**Crux D (Wildcards):**
- S1 believes: auto-acceptance via traits is a feature because trait taxonomy is stable.
- S2 believes: auto-acceptance is a silent correctness risk.
- Partly empirical (how often are auto-accepted connections valid?), partly value judgment (prefer false negatives or false positives?).

**Crux E (Error surface):**
- S1 believes: three rules with clear semantics are tractable. Complexity is in the domain.
- S2 believes: one developer maintaining a 3-rule system across 200 devices will make mistakes harder to diagnose.
- Primarily a judgment about Eric's bandwidth and debugging cost of trait-assignment errors vs. enumeration-maintenance errors.

### Open Questions for Arbiter

**Q1: How many devices before revenue pressure demands you stop touching the type system?**
If 5-10, S2's phased approach fits naturally. If 30+ (Forge ecosystem already defines 15 analysis types and you want them wirable quickly), S1's upfront investment pays off sooner.

**Q2: How often should a new analysis type be "automatically compatible" with existing devices?**
If a new capability metric (Cpm) should immediately work everywhere Cpk works without updating 12 device definitions, traits pay for themselves. If each device has specific opinions about which metrics it accepts, explicit enumeration is safer.

**Q3: Who builds new devices?**
If always Eric + Claude, the complexity tax of traits is lower. If third parties or community contributors ever build devices, is it easier to teach "list the types you accept" or "assign the right traits to your output"?

**Q4: How painful is the flat-to-structured migration S2 assumes will happen later?**
S2 calls it "additive, mechanical." At 30 devices with saved user flowcharts, adding a trait layer requires updating every device definition and testing that no compatibility outcome changed. Is that actually easier than getting traits somewhat wrong now and adjusting?

**Q5: What does "decorative" mean in practice at 5 devices?**
If Raj is evaluating SVEND competitively at device 5, decorative connections could be disqualifying. If device 5 is still pre-revenue validation, it doesn't matter yet.

**Q6: How stable is the Forge package taxonomy?**
The 15 Forge packages already have defined analysis types. Do those map cleanly to traits (grounding S1's domain model)? Or do they have idiosyncratic compatibility requirements that would fight a trait system?

### Risk of Each Path

**If you follow S1 (Structural typing with traits now):**

- **Trait taxonomy gets it wrong.** Define IS_CAPABILITY and discover some devices want "process capability" (Cpk/Ppk) but not "measurement capability" (Gage R&R %), forcing a split. Each split touches every device using the original trait.
- **Trait assignment bugs are silent.** Wrong traits on an output allow invalid connections. Debugging requires understanding the full trait set of both ports.
- **Over-engineering tax.** Weeks on Shape/Trait/Tag infrastructure before device 2 exists. If Ernie or ILSSI creates urgency, type system work competes with demo-ready features.
- **Premature abstraction lock-in.** The 3-rule check becomes load-bearing. If the rule needs to change (e.g., OR instead of AND), the migration touches every device.

**If you follow S2 (Flat types with explicit sets now):**

- **Enumeration maintenance becomes a drag.** At 30 devices with 40 types, adding a new metric type requires auditing every device's accepted_types. Miss one: false rejection. Include one you shouldn't: silent bad connection.
- **Implicit type system emerges anyway.** Without formal structure, device code grows `if type.startswith("metric:")` checks, helper functions that group types. This IS a type system — just untested.
- **Delayed migration is more expensive than claimed.** At 30 devices with user-saved flowcharts, moving to traits requires migrating code and data. A connection that was explicitly listed might not be covered by the new trait set.
- **Competitive disadvantage.** "These connections are valid because both are capability indices" vs "this type is in the accepted list." The latter feels like a dumb router to Raj-type evaluators.

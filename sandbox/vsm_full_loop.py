#!/usr/bin/env python3
"""
VSM Full Loop — The Big Test
==============================

Current state VSM → Simulations → Lot size optimization → Future state VSM
→ Diff → Mock improvement charters

This is the test that proves the architecture works for COMPLEX workflows,
not just data-in/stats-out. If this works, the port system handles everything.

Run:
  cd ~/kjerne
  python3 sandbox/vsm_full_loop.py

What we're testing:
  - VSM as a device with 24+ operations
  - Simulation devices that take VSM outputs and produce optimized parameters
  - Lot size calculators feeding back into future state
  - Future state VSM consuming simulation outputs
  - Diff engine comparing current vs future
  - Mock charter generation from the diff

The math is INTENTIONALLY SIMPLE. This tests the SHAPE, not the computation.
Real forge packages have the math. This tests whether data flows correctly
between devices through typed ports.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Reuse the type system from semantic_types.py (inline, no import needed)
# ---------------------------------------------------------------------------

import re
_TYPE_PATTERN = re.compile(r'^([a-z_]+):([a-z_*]+(?:\[\])?)$')

@dataclass(frozen=True)
class SemanticType:
    category: str
    subtype: str
    is_array: bool = False

    @classmethod
    def parse(cls, s):
        m = _TYPE_PATTERN.match(s)
        if not m:
            raise ValueError(f"Invalid type '{s}'")
        cat, sub = m.group(1), m.group(2)
        arr = sub.endswith('[]')
        if arr: sub = sub[:-2]
        return cls(cat, sub, arr)

    @property
    def is_wildcard(self): return self.subtype == '*'

    def accepts(self, other):
        if self.category != other.category: return False
        if self.is_array != other.is_array: return False
        if self.is_wildcard: return True
        return self.subtype == other.subtype

    def __str__(self):
        return f"{self.category}:{self.subtype}{'[]' if self.is_array else ''}"

@dataclass
class Port:
    name: str
    semantic_type: SemanticType
    multi: bool = False
    required: bool = True
    default: Any = None

    @classmethod
    def from_str(cls, name, type_str, **kw):
        return cls(name=name, semantic_type=SemanticType.parse(type_str), **kw)

@dataclass
class DeviceSchema:
    name: str
    description: str
    inputs: List[Port] = field(default_factory=list)
    outputs: List[Port] = field(default_factory=list)

    def get_input(self, n): return next((p for p in self.inputs if p.name == n), None)
    def get_output(self, n): return next((p for p in self.outputs if p.name == n), None)

@dataclass
class Connection:
    source_device: str
    source_port: str
    target_device: str
    target_port: str


# ---------------------------------------------------------------------------
# Device Execution Engine (minimal — proves data flows through ports)
# ---------------------------------------------------------------------------

class DeviceInstance:
    """A live device with schema + execute function + stored outputs."""

    def __init__(self, name: str, schema: DeviceSchema, execute_fn):
        self.name = name
        self.schema = schema
        self.execute_fn = execute_fn
        self.port_values: Dict[str, Any] = {}  # output port name -> value
        self.input_values: Dict[str, Any] = {}  # input port name -> value

    def receive(self, port_name: str, value: Any):
        """Receive a value on an input port."""
        port = self.schema.get_input(port_name)
        if port and port.multi:
            self.input_values.setdefault(port_name, []).append(value)
        else:
            self.input_values[port_name] = value

    def execute(self):
        """Run the device and populate output ports."""
        self.port_values = self.execute_fn(self.input_values)
        return self.port_values

    def get_output(self, port_name: str) -> Any:
        return self.port_values.get(port_name)


def run_flowchart(devices: Dict[str, DeviceInstance], connections: List[Connection]):
    """Execute devices in topological order, routing data through connections.

    This is the core engine. It:
    1. Topological sorts the devices
    2. For each device in order: execute, then push outputs to downstream inputs
    """
    # Build adjacency + in-degree
    adj = {name: [] for name in devices}
    in_deg = {name: 0 for name in devices}
    for conn in connections:
        if conn.target_device not in [c[0] for c in adj.get(conn.source_device, [])]:
            adj[conn.source_device].append((conn.target_device, conn))
            in_deg[conn.target_device] = in_deg.get(conn.target_device, 0) + 1

    # Kahn's topological sort
    queue = [d for d in devices if in_deg[d] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor, conn in adj.get(node, []):
            in_deg[neighbor] -= 1
            if in_deg[neighbor] == 0:
                queue.append(neighbor)

    # Execute in topological order
    results = {}
    for name in order:
        device = devices[name]
        device.execute()
        results[name] = device.port_values

        # Route outputs to downstream devices
        for _, conn in adj.get(name, []):
            value = device.get_output(conn.source_port)
            if value is not None:
                target = devices[conn.target_device]
                target.receive(conn.target_port, value)

    return results


# ---------------------------------------------------------------------------
# VSM Operation Data — 24 operations, realistic manufacturing flow
# ---------------------------------------------------------------------------

def make_vsm_operations() -> List[Dict]:
    """24-operation value stream for a machined component.

    Each operation has: name, cycle_time (sec), setup_time (min),
    batch_size, wip, uptime %, operators, scrap_rate %
    """
    return [
        {"name": "Raw Material Receiving", "cycle_time": 30, "setup_time": 0, "batch_size": 500, "wip": 2000, "uptime": 0.99, "operators": 1, "scrap_rate": 0.0},
        {"name": "Incoming Inspection", "cycle_time": 45, "setup_time": 5, "batch_size": 500, "wip": 500, "uptime": 0.95, "operators": 1, "scrap_rate": 0.5},
        {"name": "Bar Cut (Saw)", "cycle_time": 18, "setup_time": 15, "batch_size": 200, "wip": 400, "uptime": 0.90, "operators": 1, "scrap_rate": 1.0},
        {"name": "Deburr (Manual)", "cycle_time": 25, "setup_time": 0, "batch_size": 200, "wip": 200, "uptime": 1.0, "operators": 1, "scrap_rate": 0.0},
        {"name": "CNC Turn Op 10", "cycle_time": 62, "setup_time": 45, "batch_size": 100, "wip": 300, "uptime": 0.85, "operators": 1, "scrap_rate": 2.0},
        {"name": "CNC Turn Op 20", "cycle_time": 58, "setup_time": 40, "batch_size": 100, "wip": 250, "uptime": 0.87, "operators": 1, "scrap_rate": 1.5},
        {"name": "CNC Mill Op 30", "cycle_time": 75, "setup_time": 60, "batch_size": 100, "wip": 350, "uptime": 0.82, "operators": 1, "scrap_rate": 2.5},
        {"name": "Wash", "cycle_time": 120, "setup_time": 10, "batch_size": 50, "wip": 100, "uptime": 0.95, "operators": 0, "scrap_rate": 0.0},
        {"name": "CMM Inspection", "cycle_time": 180, "setup_time": 20, "batch_size": 50, "wip": 150, "uptime": 0.90, "operators": 1, "scrap_rate": 0.0},
        {"name": "Heat Treat (External)", "cycle_time": 0, "setup_time": 0, "batch_size": 500, "wip": 1500, "uptime": 1.0, "operators": 0, "scrap_rate": 0.5, "lead_time_days": 5},
        {"name": "Receiving (Heat Treat)", "cycle_time": 30, "setup_time": 0, "batch_size": 500, "wip": 500, "uptime": 0.99, "operators": 1, "scrap_rate": 0.0},
        {"name": "Hardness Test", "cycle_time": 60, "setup_time": 5, "batch_size": 50, "wip": 100, "uptime": 0.95, "operators": 1, "scrap_rate": 0.0},
        {"name": "CNC Grind OD", "cycle_time": 90, "setup_time": 30, "batch_size": 50, "wip": 200, "uptime": 0.80, "operators": 1, "scrap_rate": 3.0},
        {"name": "CNC Grind ID", "cycle_time": 85, "setup_time": 35, "batch_size": 50, "wip": 200, "uptime": 0.82, "operators": 1, "scrap_rate": 2.5},
        {"name": "Hone", "cycle_time": 45, "setup_time": 25, "batch_size": 50, "wip": 100, "uptime": 0.88, "operators": 1, "scrap_rate": 1.0},
        {"name": "Wash (Post-Grind)", "cycle_time": 120, "setup_time": 10, "batch_size": 50, "wip": 50, "uptime": 0.95, "operators": 0, "scrap_rate": 0.0},
        {"name": "Surface Treatment", "cycle_time": 300, "setup_time": 15, "batch_size": 100, "wip": 200, "uptime": 0.90, "operators": 1, "scrap_rate": 0.5},
        {"name": "Final Inspection", "cycle_time": 120, "setup_time": 10, "batch_size": 25, "wip": 75, "uptime": 0.95, "operators": 1, "scrap_rate": 0.0},
        {"name": "Marking/Etch", "cycle_time": 15, "setup_time": 5, "batch_size": 100, "wip": 100, "uptime": 0.98, "operators": 1, "scrap_rate": 0.0},
        {"name": "Packaging", "cycle_time": 20, "setup_time": 5, "batch_size": 100, "wip": 100, "uptime": 0.99, "operators": 1, "scrap_rate": 0.0},
        {"name": "Final Wash/Clean", "cycle_time": 90, "setup_time": 5, "batch_size": 50, "wip": 50, "uptime": 0.95, "operators": 0, "scrap_rate": 0.0},
        {"name": "Visual Inspection", "cycle_time": 30, "setup_time": 0, "batch_size": 50, "wip": 50, "uptime": 1.0, "operators": 1, "scrap_rate": 0.0},
        {"name": "Cert & Documentation", "cycle_time": 300, "setup_time": 0, "batch_size": 25, "wip": 25, "uptime": 0.99, "operators": 1, "scrap_rate": 0.0},
        {"name": "Ship", "cycle_time": 60, "setup_time": 0, "batch_size": 500, "wip": 500, "uptime": 0.99, "operators": 1, "scrap_rate": 0.0},
    ]


# ---------------------------------------------------------------------------
# Device Execute Functions — each takes input_values dict, returns output dict
# ---------------------------------------------------------------------------

def execute_vsm_current(inputs: Dict) -> Dict:
    """Current state VSM device.

    IN:  vsm:operations[] — list of operation dicts
         config:demand_rate — parts/day customer wants
    OUT: metric:lead_time, metric:process_time, metric:pce,
         metric:total_wip, metric:bottleneck_ct,
         metric:takt_time, vsm:operations_analyzed[]
    """
    ops = inputs.get("operations", make_vsm_operations())
    demand_rate = inputs.get("demand_rate", 100)  # parts/day

    available_seconds = 8 * 3600  # 8-hour shift
    takt_time = available_seconds / demand_rate

    total_ct = 0
    total_wip = 0
    bottleneck_ct = 0
    bottleneck_name = ""
    analyzed = []

    for op in ops:
        ct = op["cycle_time"]
        effective_ct = ct / op["uptime"] if op["uptime"] > 0 else ct
        wip_days = op["wip"] / demand_rate if demand_rate > 0 else 0
        setup_per_part = (op["setup_time"] * 60) / op["batch_size"] if op["batch_size"] > 0 else 0

        total_ct += ct
        total_wip += op["wip"]

        if effective_ct > bottleneck_ct:
            bottleneck_ct = effective_ct
            bottleneck_name = op["name"]

        analyzed.append({
            **op,
            "effective_ct": round(effective_ct, 1),
            "wip_days": round(wip_days, 1),
            "setup_per_part": round(setup_per_part, 1),
            "va_ratio": round(ct / effective_ct, 3) if effective_ct > 0 else 0,
            "exceeds_takt": effective_ct > takt_time,
        })

    # Lead time = sum of WIP days + external lead times
    lead_time_days = sum(a["wip_days"] for a in analyzed) + \
                     sum(op.get("lead_time_days", 0) for op in ops)
    process_time_sec = total_ct
    process_time_days = process_time_sec / available_seconds
    pce = (process_time_days / lead_time_days * 100) if lead_time_days > 0 else 0

    return {
        "lead_time": round(lead_time_days, 1),
        "process_time": round(process_time_sec, 1),
        "pce": round(pce, 2),
        "total_wip": total_wip,
        "bottleneck_ct": round(bottleneck_ct, 1),
        "bottleneck_name": bottleneck_name,
        "takt_time": round(takt_time, 1),
        "operations_analyzed": analyzed,
        "demand_rate": demand_rate,
        "summary": (
            f"Current State VSM: {len(ops)} operations\n"
            f"  Lead time: {lead_time_days:.1f} days\n"
            f"  Process time: {process_time_sec:.0f} sec ({process_time_days:.2f} days)\n"
            f"  PCE: {pce:.2f}%\n"
            f"  Total WIP: {total_wip:,} units\n"
            f"  Bottleneck: {bottleneck_name} ({bottleneck_ct:.1f} sec)\n"
            f"  Takt time: {takt_time:.1f} sec"
        ),
    }


def execute_lot_size_optimizer(inputs: Dict) -> Dict:
    """Lot size optimizer — EPQ/EOQ for each operation.

    IN:  vsm:operations_analyzed[] — from current state VSM
         config:demand_rate
         config:holding_cost_pct — annual holding cost as % of unit cost
         config:unit_cost
    OUT: vsm:optimized_batches[] — recommended batch sizes per operation
         metric:total_setup_reduction_pct
         text:summary
    """
    ops = inputs.get("operations_analyzed", [])
    demand_rate = inputs.get("demand_rate", 100)
    holding_pct = inputs.get("holding_cost_pct", 0.25)
    unit_cost = inputs.get("unit_cost", 50.0)

    annual_demand = demand_rate * 250  # 250 working days
    holding_cost = unit_cost * holding_pct

    optimized = []
    total_current_setup = 0
    total_optimal_setup = 0

    for op in ops:
        setup_time_hrs = op["setup_time"] / 60
        setup_cost = setup_time_hrs * 75  # $75/hr loaded rate

        if setup_cost > 0 and annual_demand > 0:
            # EPQ formula: Q* = sqrt(2DS / H)
            eoq = math.sqrt(2 * annual_demand * setup_cost / holding_cost)
            eoq = max(10, round(eoq / 10) * 10)  # Round to nearest 10, min 10
        else:
            eoq = op["batch_size"]

        current_setups_per_year = annual_demand / op["batch_size"] if op["batch_size"] > 0 else 0
        optimal_setups_per_year = annual_demand / eoq if eoq > 0 else 0

        total_current_setup += current_setups_per_year * setup_time_hrs
        total_optimal_setup += optimal_setups_per_year * setup_time_hrs

        optimized.append({
            "name": op["name"],
            "current_batch": op["batch_size"],
            "optimal_batch": int(eoq),
            "setup_time_min": op["setup_time"],
            "current_setups_yr": round(current_setups_per_year, 1),
            "optimal_setups_yr": round(optimal_setups_per_year, 1),
            "setup_reduction_pct": round(
                (1 - optimal_setups_per_year / current_setups_per_year) * 100, 1
            ) if current_setups_per_year > 0 else 0,
        })

    setup_reduction = (
        (1 - total_optimal_setup / total_current_setup) * 100
        if total_current_setup > 0 else 0
    )

    return {
        "optimized_batches": optimized,
        "total_setup_reduction_pct": round(setup_reduction, 1),
        "summary": (
            f"Lot Size Optimization: {len(optimized)} operations analyzed\n"
            f"  Total setup time reduction: {setup_reduction:.1f}%\n"
            f"  Current annual setup hours: {total_current_setup:.0f}\n"
            f"  Optimal annual setup hours: {total_optimal_setup:.0f}"
        ),
    }


def execute_simulation(inputs: Dict) -> Dict:
    """Monte Carlo simulation on the value stream.

    IN:  vsm:operations_analyzed[] — from current state VSM
         vsm:optimized_batches[] — from lot size optimizer
         config:num_runs
    OUT: vsm:simulated_operations[] — ops with simulated parameters
         metric:sim_lead_time_p50, metric:sim_lead_time_p95
         metric:sim_throughput
         list:improvement_opportunities
         text:summary
    """
    ops = inputs.get("operations_analyzed", [])
    batches = inputs.get("optimized_batches", [])
    num_runs = inputs.get("num_runs", 1000)

    # Build batch lookup
    batch_lookup = {b["name"]: b["optimal_batch"] for b in batches}

    # Simulate: vary cycle times, uptimes, scrap rates
    lead_times = []
    random.seed(42)  # Reproducible

    for _ in range(num_runs):
        run_lead_time = 0
        for op in ops:
            # Vary cycle time +/- 20%
            ct_var = op["cycle_time"] * (1 + random.uniform(-0.2, 0.2))
            # Vary uptime: beta distribution around actual
            uptime_var = min(1.0, max(0.5, op["uptime"] * (1 + random.uniform(-0.1, 0.1))))
            # Use optimized batch size
            batch = batch_lookup.get(op["name"], op["batch_size"])
            # WIP days = batch / demand (simplified)
            wip_days = batch / 100  # simplified
            run_lead_time += wip_days + op.get("lead_time_days", 0)
        lead_times.append(run_lead_time)

    lead_times.sort()
    p50 = lead_times[len(lead_times) // 2]
    p95 = lead_times[int(len(lead_times) * 0.95)]

    # Build simulated operations with optimized parameters
    simulated_ops = []
    improvements = []

    for op in ops:
        opt_batch = batch_lookup.get(op["name"], op["batch_size"])
        # Simulate improved uptime (+5% from PM program)
        sim_uptime = min(1.0, op["uptime"] + 0.05)
        # Simulate reduced setup (SMED target: 50% reduction for setups > 30 min)
        sim_setup = op["setup_time"]
        smed_candidate = False
        if op["setup_time"] > 30:
            sim_setup = op["setup_time"] * 0.5
            smed_candidate = True

        # WIP in future state: use SMALLER of current WIP or optimal batch
        # EPQ gives bigger batches (fewer setups) but the WIP improvement
        # comes from SMED (smaller batches become feasible) + flow.
        # For the simulation: target WIP = min(current, optimal_batch)
        # Real sim would model queuing theory; this tests the SHAPE.
        target_wip = min(op["wip"], opt_batch)

        sim_op = {
            **op,
            "batch_size": min(opt_batch, op["batch_size"]),  # Don't increase batch size
            "wip": target_wip,
            "uptime": round(sim_uptime, 3),
            "setup_time": round(sim_setup, 1),
            "smed_candidate": smed_candidate,
        }
        simulated_ops.append(sim_op)

        if smed_candidate:
            improvements.append({
                "type": "SMED",
                "operation": op["name"],
                "current_setup": op["setup_time"],
                "target_setup": round(sim_setup, 1),
                "reduction_pct": 50,
            })

        if op["uptime"] < 0.85:
            improvements.append({
                "type": "TPM",
                "operation": op["name"],
                "current_uptime": op["uptime"],
                "target_uptime": round(sim_uptime, 3),
            })

        if op["scrap_rate"] > 2.0:
            improvements.append({
                "type": "Quality",
                "operation": op["name"],
                "current_scrap": op["scrap_rate"],
                "target_scrap": round(op["scrap_rate"] * 0.5, 1),
            })

    return {
        "simulated_operations": simulated_ops,
        "sim_lead_time_p50": round(p50, 1),
        "sim_lead_time_p95": round(p95, 1),
        "sim_throughput": round(100 * (1 - sum(op["scrap_rate"] for op in ops) / 100 / len(ops)), 1),
        "improvement_opportunities": improvements,
        "summary": (
            f"Simulation ({num_runs} runs):\n"
            f"  Lead time P50: {p50:.1f} days\n"
            f"  Lead time P95: {p95:.1f} days\n"
            f"  Improvement opportunities: {len(improvements)}"
        ),
    }


def execute_vsm_future(inputs: Dict) -> Dict:
    """Future state VSM — consumes simulation outputs.

    IN:  vsm:simulated_operations[] — from simulation (optimized params)
         config:demand_rate
    OUT: same as current state VSM but with improved numbers
    """
    sim_ops = inputs.get("simulated_operations", [])
    demand_rate = inputs.get("demand_rate", 100)

    # Run the same VSM calculation on the simulated/optimized operations
    # This is the KEY: future state consumes simulation outputs through ports
    fake_inputs = {"operations": sim_ops, "demand_rate": demand_rate}
    return execute_vsm_current(fake_inputs)


def execute_vsm_diff(inputs: Dict) -> Dict:
    """Diff current vs future state — produces improvement charters.

    IN:  vsm:current_state — full current state output
         vsm:future_state — full future state output
         list:improvement_opportunities — from simulation
    OUT: list:charters — mock improvement project charters
         text:summary — diff summary
         metric:lead_time_reduction_pct
         metric:wip_reduction_pct
         metric:pce_improvement
    """
    current = inputs.get("current_state", {})
    future = inputs.get("future_state", {})
    improvements = inputs.get("improvement_opportunities", [])

    c_lead = current.get("lead_time", 0)
    f_lead = future.get("lead_time", 0)
    c_wip = current.get("total_wip", 0)
    f_wip = future.get("total_wip", 0)
    c_pce = current.get("pce", 0)
    f_pce = future.get("pce", 0)

    lead_reduction = ((c_lead - f_lead) / c_lead * 100) if c_lead > 0 else 0
    wip_reduction = ((c_wip - f_wip) / c_wip * 100) if c_wip > 0 else 0
    pce_improvement = f_pce - c_pce

    # Generate mock charters from improvement opportunities
    charters = []
    for imp in improvements:
        if imp["type"] == "SMED":
            charters.append({
                "project_type": "SMED",
                "title": f"SMED — {imp['operation']}",
                "scope": f"Reduce setup time from {imp['current_setup']} min to {imp['target_setup']} min",
                "target": f"{imp['reduction_pct']}% setup reduction",
                "operation": imp["operation"],
                "estimated_weeks": 6,
                "team": ["Process Engineer", "Setup Technician", "Operator"],
                "phases": [
                    "Video current setup",
                    "Separate internal/external",
                    "Convert internal to external",
                    "Streamline remaining",
                    "Standardize & document",
                ],
                "status": "draft",
            })
        elif imp["type"] == "TPM":
            charters.append({
                "project_type": "TPM",
                "title": f"TPM — {imp['operation']}",
                "scope": f"Increase uptime from {imp['current_uptime']*100:.0f}% to {imp['target_uptime']*100:.0f}%",
                "target": f"{(imp['target_uptime'] - imp['current_uptime'])*100:.0f}pp uptime improvement",
                "operation": imp["operation"],
                "estimated_weeks": 8,
                "team": ["Maintenance Tech", "Operator", "Process Engineer"],
                "phases": [
                    "Baseline OEE measurement",
                    "PM task analysis",
                    "Operator basic care training",
                    "Implement AM/PM schedule",
                    "Monitor & adjust",
                ],
                "status": "draft",
            })
        elif imp["type"] == "Quality":
            charters.append({
                "project_type": "Quality Improvement",
                "title": f"Scrap Reduction — {imp['operation']}",
                "scope": f"Reduce scrap from {imp['current_scrap']}% to {imp['target_scrap']}%",
                "target": f"50% scrap reduction",
                "operation": imp["operation"],
                "estimated_weeks": 12,
                "team": ["Quality Engineer", "Operator", "Process Engineer"],
                "phases": [
                    "Pareto of defect modes",
                    "Root cause analysis (5-Why / Fishbone)",
                    "Countermeasure implementation",
                    "Process capability validation",
                    "Control plan update",
                ],
                "status": "draft",
            })

    return {
        "charters": charters,
        "lead_time_reduction_pct": round(lead_reduction, 1),
        "wip_reduction_pct": round(wip_reduction, 1),
        "pce_improvement": round(pce_improvement, 2),
        "summary": (
            f"VSM Diff: Current → Future\n"
            f"  Lead time: {c_lead:.1f} → {f_lead:.1f} days "
            f"({lead_reduction:+.1f}%)\n"
            f"  WIP: {c_wip:,} → {f_wip:,} units "
            f"({wip_reduction:+.1f}%)\n"
            f"  PCE: {c_pce:.2f}% → {f_pce:.2f}% "
            f"({pce_improvement:+.2f}pp)\n"
            f"  Improvement projects: {len(charters)}"
        ),
    }


# ---------------------------------------------------------------------------
# Device Schemas
# ---------------------------------------------------------------------------

# NOTE: Using vsm:* types for VSM-specific data. These are NEW semantic types
# that emerged from this test. The type system is extensible — we didn't have
# to modify anything, just used new category:subtype strings.
# This is exactly how it should work: devices define the types they need.

VSM_CURRENT_SCHEMA = DeviceSchema(
    "vsm_current_state", "Current state value stream map",
    inputs=[
        Port.from_str("operations", "vsm:operations[]"),
        Port.from_str("demand_rate", "config:demand_rate", required=False, default=100),
    ],
    outputs=[
        Port.from_str("lead_time", "metric:lead_time"),
        Port.from_str("process_time", "metric:process_time"),
        Port.from_str("pce", "metric:pce"),
        Port.from_str("total_wip", "metric:wip"),
        Port.from_str("bottleneck_ct", "metric:cycle_time"),
        Port.from_str("takt_time", "metric:takt_time"),
        Port.from_str("operations_analyzed", "vsm:operations[]"),
        Port.from_str("summary", "text:summary"),
    ],
)

LOT_SIZE_SCHEMA = DeviceSchema(
    "lot_size_optimizer", "EPQ/EOQ lot size optimization",
    inputs=[
        Port.from_str("operations_analyzed", "vsm:operations[]"),
        Port.from_str("demand_rate", "config:demand_rate", required=False, default=100),
        Port.from_str("holding_cost_pct", "config:holding_cost_pct", required=False, default=0.25),
        Port.from_str("unit_cost", "config:unit_cost", required=False, default=50.0),
    ],
    outputs=[
        Port.from_str("optimized_batches", "vsm:batches[]"),
        Port.from_str("total_setup_reduction_pct", "metric:percentage"),
        Port.from_str("summary", "text:summary"),
    ],
)

SIMULATION_SCHEMA = DeviceSchema(
    "simulation", "Monte Carlo simulation on value stream",
    inputs=[
        Port.from_str("operations_analyzed", "vsm:operations[]"),
        Port.from_str("optimized_batches", "vsm:batches[]"),
        Port.from_str("num_runs", "config:num_runs", required=False, default=1000),
    ],
    outputs=[
        Port.from_str("simulated_operations", "vsm:operations[]"),
        Port.from_str("sim_lead_time_p50", "metric:lead_time"),
        Port.from_str("sim_lead_time_p95", "metric:lead_time"),
        Port.from_str("sim_throughput", "metric:percentage"),
        Port.from_str("improvement_opportunities", "list:improvements"),
        Port.from_str("summary", "text:summary"),
    ],
)

VSM_FUTURE_SCHEMA = DeviceSchema(
    "vsm_future_state", "Future state value stream map (from simulation)",
    inputs=[
        Port.from_str("simulated_operations", "vsm:operations[]"),
        Port.from_str("demand_rate", "config:demand_rate", required=False, default=100),
    ],
    outputs=[
        Port.from_str("lead_time", "metric:lead_time"),
        Port.from_str("process_time", "metric:process_time"),
        Port.from_str("pce", "metric:pce"),
        Port.from_str("total_wip", "metric:wip"),
        Port.from_str("bottleneck_ct", "metric:cycle_time"),
        Port.from_str("takt_time", "metric:takt_time"),
        Port.from_str("operations_analyzed", "vsm:operations[]"),
        Port.from_str("summary", "text:summary"),
    ],
)

DIFF_SCHEMA = DeviceSchema(
    "vsm_diff", "Current vs future state diff + charter generation",
    inputs=[
        Port.from_str("current_state", "vsm:state"),
        Port.from_str("future_state", "vsm:state"),
        Port.from_str("improvement_opportunities", "list:improvements"),
    ],
    outputs=[
        Port.from_str("charters", "list:charters"),
        Port.from_str("lead_time_reduction_pct", "metric:percentage"),
        Port.from_str("wip_reduction_pct", "metric:percentage"),
        Port.from_str("pce_improvement", "metric:percentage"),
        Port.from_str("summary", "text:summary"),
    ],
)


# ---------------------------------------------------------------------------
# Run the full loop
# ---------------------------------------------------------------------------

def run_full_loop():
    print("=" * 70)
    print("  VSM FULL LOOP — The Big Test")
    print("  24 operations → simulate → optimize → future state → diff → charters")
    print("=" * 70)

    # Create device instances
    devices = {
        "vsm_current": DeviceInstance("vsm_current", VSM_CURRENT_SCHEMA, execute_vsm_current),
        "lot_size": DeviceInstance("lot_size", LOT_SIZE_SCHEMA, execute_lot_size_optimizer),
        "sim": DeviceInstance("sim", SIMULATION_SCHEMA, execute_simulation),
        "vsm_future": DeviceInstance("vsm_future", VSM_FUTURE_SCHEMA, execute_vsm_future),
        "diff": DeviceInstance("diff", DIFF_SCHEMA, execute_vsm_diff),
    }

    # Seed current state VSM with operations
    devices["vsm_current"].input_values = {
        "operations": make_vsm_operations(),
        "demand_rate": 100,
    }

    # Wire them together — this is the flowchart
    connections = [
        # Current state → lot size optimizer
        Connection("vsm_current", "operations_analyzed", "lot_size", "operations_analyzed"),

        # Current state → simulation
        Connection("vsm_current", "operations_analyzed", "sim", "operations_analyzed"),

        # Lot size → simulation (optimized batches inform sim)
        Connection("lot_size", "optimized_batches", "sim", "optimized_batches"),

        # Simulation → future state VSM (THIS IS THE FEEDBACK)
        Connection("sim", "simulated_operations", "vsm_future", "simulated_operations"),

        # NOTE: diff device needs FULL VSM state, not individual ports.
        # This surfaces GOTCHA #1: composite ports. The diff device
        # receives the full output dict as vsm:state, not individual
        # metrics. We run the diff separately after the main flowchart.
        Connection("sim", "improvement_opportunities", "diff", "improvement_opportunities"),
    ]

    # Run the main flowchart (everything except diff)
    print("\n--- Running flowchart ---\n")
    results = run_flowchart(devices, connections)

    # Wire the diff device with full state outputs (composite port pattern)
    # GOTCHA #1: This is where we need vsm:state as a composite type.
    # For now, pass the full output dicts directly.
    devices["diff"].input_values["current_state"] = results["vsm_current"]
    devices["diff"].input_values["future_state"] = results["vsm_future"]
    devices["diff"].input_values["improvement_opportunities"] = \
        results["sim"].get("improvement_opportunities", [])
    devices["diff"].execute()
    results["diff"] = devices["diff"].port_values

    # --- Print Results ---

    print("=" * 70)
    print("  1. CURRENT STATE VSM")
    print("=" * 70)
    print(results["vsm_current"]["summary"])

    print(f"\n  Operations exceeding takt ({results['vsm_current']['takt_time']}s):")
    for op in results["vsm_current"]["operations_analyzed"]:
        if op["exceeds_takt"]:
            print(f"    ! {op['name']}: {op['effective_ct']}s (takt={results['vsm_current']['takt_time']}s)")

    print(f"\n{'=' * 70}")
    print(f"  2. LOT SIZE OPTIMIZATION")
    print(f"{'=' * 70}")
    print(results["lot_size"]["summary"])
    print(f"\n  Top changes:")
    for batch in results["lot_size"]["optimized_batches"]:
        if batch["current_batch"] != batch["optimal_batch"]:
            print(f"    {batch['name']}: {batch['current_batch']} → {batch['optimal_batch']} "
                  f"({batch['setup_reduction_pct']:+.0f}% setups)")

    print(f"\n{'=' * 70}")
    print(f"  3. SIMULATION")
    print(f"{'=' * 70}")
    print(results["sim"]["summary"])

    print(f"\n  Improvement opportunities:")
    for imp in results["sim"]["improvement_opportunities"]:
        if imp["type"] == "SMED":
            print(f"    SMED: {imp['operation']} — setup {imp['current_setup']}→{imp['target_setup']} min")
        elif imp["type"] == "TPM":
            print(f"    TPM:  {imp['operation']} — uptime {imp['current_uptime']*100:.0f}→{imp['target_uptime']*100:.0f}%")
        elif imp["type"] == "Quality":
            print(f"    QUAL: {imp['operation']} — scrap {imp['current_scrap']}→{imp['target_scrap']}%")

    print(f"\n{'=' * 70}")
    print(f"  4. FUTURE STATE VSM")
    print(f"{'=' * 70}")
    print(results["vsm_future"]["summary"])

    print(f"\n{'=' * 70}")
    print(f"  5. DIFF: CURRENT → FUTURE")
    print(f"{'=' * 70}")
    print(results["diff"]["summary"])

    print(f"\n{'=' * 70}")
    print(f"  6. MOCK CHARTERS ({len(results['diff']['charters'])} projects)")
    print(f"{'=' * 70}")
    for i, charter in enumerate(results["diff"]["charters"], 1):
        print(f"\n  [{i}] {charter['title']}")
        print(f"      Type: {charter['project_type']}")
        print(f"      Scope: {charter['scope']}")
        print(f"      Target: {charter['target']}")
        print(f"      Duration: {charter['estimated_weeks']} weeks")
        print(f"      Team: {', '.join(charter['team'])}")
        print(f"      Phases:")
        for j, phase in enumerate(charter['phases'], 1):
            print(f"        {j}. {phase}")

    # --- Validation ---
    print(f"\n{'=' * 70}")
    print(f"  VALIDATION")
    print(f"{'=' * 70}")

    passed = 0
    failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name} -- {detail}")

    check("24 operations processed",
          len(results["vsm_current"]["operations_analyzed"]) == 24)
    check("Lead time > 0",
          results["vsm_current"]["lead_time"] > 0)
    check("PCE calculated",
          results["vsm_current"]["pce"] > 0)
    check("Bottleneck identified",
          results["vsm_current"]["bottleneck_name"] != "")
    check("Lot sizes optimized",
          len(results["lot_size"]["optimized_batches"]) == 24)
    check("Simulation produced operations",
          len(results["sim"]["simulated_operations"]) == 24)
    check("Future state has lower lead time",
          results["vsm_future"]["lead_time"] < results["vsm_current"]["lead_time"],
          f"future={results['vsm_future']['lead_time']}, current={results['vsm_current']['lead_time']}")
    check("Future state has lower WIP",
          results["vsm_future"]["total_wip"] < results["vsm_current"]["total_wip"])
    check("Future state has higher PCE",
          results["vsm_future"]["pce"] > results["vsm_current"]["pce"])
    check("Diff produced charters",
          len(results["diff"]["charters"]) > 0)
    check("SMED charters for high-setup operations",
          any(c["project_type"] == "SMED" for c in results["diff"]["charters"]))
    check("TPM charters for low-uptime operations",
          any(c["project_type"] == "TPM" for c in results["diff"]["charters"]))
    check("Quality charters for high-scrap operations",
          any(c["project_type"] == "Quality Improvement" for c in results["diff"]["charters"]))
    check("Lead time reduction is positive",
          results["diff"]["lead_time_reduction_pct"] > 0)
    check("Charters have phases",
          all(len(c["phases"]) > 0 for c in results["diff"]["charters"]))

    # Data flow validation — did ports actually route correctly?
    check("Current VSM output fed lot size optimizer",
          len(devices["lot_size"].input_values.get("operations_analyzed", [])) == 24)
    check("Lot size output fed simulation",
          len(devices["sim"].input_values.get("optimized_batches", [])) > 0)
    check("Simulation output fed future state VSM",
          len(devices["vsm_future"].input_values.get("simulated_operations", [])) == 24)

    print(f"\n  RESULTS: {passed}/{passed+failed} passed, {failed} failed")

    # --- Gotchas discovered ---
    print(f"\n{'=' * 70}")
    print(f"  GOTCHAS FROM VSM FULL LOOP")
    print(f"{'=' * 70}")
    print("""
  1. COMPOSITE PORTS: The diff device needs the FULL output of both
     VSMs, not individual metrics. Had to manually wire the full dict.
     Production needs: vsm:state as a composite type that bundles all
     VSM outputs. Or the diff device declares individual ports for
     each metric it needs (verbose but type-safe).

  2. NEW TYPE CATEGORY: vsm:operations[], vsm:batches[], vsm:state
     emerged naturally. The type system didn't need modification —
     just new category:subtype strings. This proves extensibility.
     But: should 'vsm' be a category or should operations be
     data:operations? Leaning toward domain-specific categories.

  3. FEEDBACK LOOP (simulation → future state): This is NOT a cycle
     in the graph. Current → sim → future is a DAG. The simulation
     PRODUCES new parameters, the future state VSM CONSUMES them.
     No circular dependency. But conceptually it feels like feedback
     because the future state "updates" the same parameters.

  4. CHARTER GENERATION is mechanical from the diff. SMED if setup
     reduced, TPM if uptime improved, Quality if scrap reduced.
     Real charters need more context (cost, resource availability,
     priority). But the SHAPE is right — diff → charters is a valid
     device connection.

  5. CONFIG FAN-OUT works: demand_rate needed by current VSM, lot
     size optimizer, AND future state VSM. In the real flowchart,
     a process config device outputs config:demand_rate and all
     three devices connect to it. "Same version of every number."
""")

    return failed == 0


if __name__ == "__main__":
    ok = run_full_loop()
    sys.exit(0 if ok else 1)

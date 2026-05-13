"""Flowchart execution engine — topological sort + port routing.

Ported from sandbox/vsm_full_loop.py run_flowchart().
Production version that creates real Jobs via run_plugin().
"""

import logging
from typing import Any, Dict, List, Optional

from syn.plugins.registry import PluginRegistry, get_registry
from syn.plugins.runner import run_plugin

logger = logging.getLogger(__name__)


def _build_multi_port_lookup(devices: List[Dict]) -> set:
    """Scan device port schemas for multi=True input ports.

    Returns a set of (device_id, port_name) tuples for ports that should
    collect connections into a list instead of overwriting.
    """
    multi = set()
    for dev in devices:
        ports = dev.get("ports", {})
        for port in ports.get("inputs", []):
            if port.get("multi"):
                multi.add((dev["id"], port["name"]))
    return multi


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

    # Build device lookup
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

    # Build multi-port lookup from device port schemas in definition
    multi_ports = _build_multi_port_lookup(devices)

    # Execute in topological order
    results: Dict[str, Dict] = {}
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
                    if (device_id, tgt_port) in multi_ports:
                        # Multi-port: collect into list
                        if tgt_port not in input_data or not isinstance(input_data[tgt_port], list):
                            input_data[tgt_port] = []
                        input_data[tgt_port].append(routed_value)
                    else:
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

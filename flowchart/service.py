"""Flowchart service layer — all mutations go through here.

Every structural change emits a bus event so Synara can:
- Track for heuristics (which templates get used, which connections fail)
- Feed Claude context assembly
- Inform governance rules

Functions:
  create_instance(template, name, user, actor) → FlowchartInstance
  delete_instance(instance, actor) → None (soft delete)
  add_device(instance, device_id, plugin_name, label, position, actor) → None
  remove_device(instance, device_id, actor) → None (cascades connections)
  add_connection(instance, source, target, actor) → None (type-checked + cycle-detected)
  remove_connection(instance, source, target, actor) → None
  validate_connection_request(definition, source, target) → dict (dry-run, no mutation)

Bus events emitted:
  flowchart.instance.created / deleted / device_added / device_removed /
  connection_added / connection_removed

Adding a new device type:
  1. Create plugins/<name>.py (Plugin subclass + Pydantic input schema)
  2. Register in plugins/apps.py ready()
  3. Optionally add port_schema to seed_templates.py for template-mode
  4. That's it — the device is immediately available via add_device()
"""

import copy
import logging
from typing import Optional

from flowchart.models import FlowchartInstance, FlowchartTemplate
from flowchart.types import validate_connection_in_context
from syn.bus import emit
from syn.plugins.registry import get_registry

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


def _get_port_schema(plugin_name: str) -> dict:
    """Get port schema from plugin registry for embedding in definition."""
    registry = get_registry()
    registry.get(plugin_name)  # validates plugin exists
    # Port schema derived from plugin metadata. For build-mode devices.
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
    """Add a device to a FlowchartInstance definition."""
    registry = get_registry()
    if not registry.has(plugin_name):
        raise ValueError(f"Unknown plugin: {plugin_name}")

    defn = instance.definition
    existing_ids = {d["id"] for d in defn.get("devices", [])}
    if device_id in existing_ids:
        raise ValueError(f"Device ID '{device_id}' already exists in this flowchart")

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

    defn["devices"] = [d for d in defn["devices"] if d["id"] != device_id]
    defn["connections"] = [
        c
        for c in defn.get("connections", [])
        if not c["source"].startswith(device_id + ".") and not c["target"].startswith(device_id + ".")
    ]
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


def validate_connection_request(definition: dict, source: str, target: str) -> dict:
    """Validate a proposed connection without mutating anything."""
    return validate_connection_in_context(definition, source, target)


def add_connection(
    *,
    instance: FlowchartInstance,
    source: str,
    target: str,
    actor: str,
) -> None:
    """Add a validated connection to a FlowchartInstance."""
    defn = instance.definition
    result = validate_connection_in_context(defn, source, target)
    if not result["valid"]:
        raise ValueError(result["error"])

    conn_type = result.get("type", "")
    defn.setdefault("connections", []).append(
        {
            "source": source,
            "target": target,
            "type": conn_type,
        }
    )

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
    *,
    instance: FlowchartInstance,
    source: str,
    target: str,
    actor: str,
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


def promote_to_template(
    *,
    instance: FlowchartInstance,
    name: str,
    description: str = "",
    is_shared: bool = False,
    actor: str,
) -> FlowchartTemplate:
    """Promote a FlowchartInstance to a reusable FlowchartTemplate."""
    devices_used = [d["plugin"] for d in instance.definition.get("devices", [])]

    tpl = FlowchartTemplate.objects.create(
        name=name,
        description=description,
        definition=copy.deepcopy(instance.definition),
        devices_used=devices_used,
        is_shared=is_shared,
        tenant_id=instance.tenant_id,
    )

    emit(
        "flowchart.template.promoted",
        {
            "template_id": str(tpl.id),
            "instance_id": str(instance.id),
            "name": name,
        },
        actor=actor,
        tenant_id=str(instance.tenant_id) if instance.tenant_id else None,
    )

    return tpl

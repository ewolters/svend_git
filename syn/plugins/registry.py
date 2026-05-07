"""
Plugin Registry — discovery and access for all registered plugins.

Simple dict-based registry. Plugins register at Django startup (apps.py ready()).
Canvas models reference plugins by name string -> registry.get(name).
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

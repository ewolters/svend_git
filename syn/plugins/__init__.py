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

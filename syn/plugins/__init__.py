"""
Synara Plugin Framework
=======================

Uniform execution contract for all SVEND analysis engines.

Usage:
    from syn.plugins import Plugin, PluginOutput
"""

from syn.plugins.base import Plugin, PluginOutput

__all__ = ["Plugin", "PluginOutput"]

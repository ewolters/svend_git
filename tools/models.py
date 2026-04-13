"""Tools app models.

Re-exports ToolModel for Django model discovery. ToolModel is abstract,
so no database tables are created by this app.
"""

from .base_model import ToolModel

__all__ = ["ToolModel"]

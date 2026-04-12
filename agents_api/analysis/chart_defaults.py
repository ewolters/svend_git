"""Chart defaults — re-exports from dsw.chart_defaults.

This file existed as a duplicate of dsw/chart_defaults.py. Now it's a thin
re-export to avoid maintaining two copies. The canonical implementation
lives in agents_api/dsw/chart_defaults.py.

Both will be replaced by ForgeViz when chart rendering migrates.
"""

from agents_api.dsw.chart_defaults import *  # noqa: F401, F403
from agents_api.dsw.chart_defaults import _normalize_color, apply_chart_defaults  # noqa: F401

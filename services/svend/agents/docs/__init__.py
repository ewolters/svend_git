"""
Documentation Tools

Rich formatters for technical documentation:
- Chemistry equations and reactions (LaTeX/mhchem)
- Mathematical notation (LaTeX/KaTeX)
- Charts and visualizations (matplotlib)
- Export (Markdown, LaTeX, HTML, PDF)

These are FORMATTERS, not reasoning engines.
They produce structured, publication-ready output.

Usage:
    from docs.chemistry import ChemistryFormatter, quick_reaction
    from docs.latex import LaTeXFormatter, get_equation
    from docs.charts import ChartGenerator, quick_chart
    from docs.export import DocumentExporter

    # Format a reaction
    result = quick_reaction("ATP + H2O -> ADP + Pi")
    print(result['latex'])  # \\ce{ATP + H2O -> ADP + Pi}

    # Create a chart
    chart = quick_chart({'x': [1,2,3], 'y': [1,4,9]}, 'line')

    # Get a common equation
    eq = get_equation('gibbs_free_energy')
    print(eq.display())  # $$\\Delta G = \\Delta H - T\\Delta S$$
"""

from .chemistry import (
    ChemistryFormatter,
    ReactionInfo,
    ThermodynamicsData,
    quick_reaction,
    BIOCHEMICAL_REACTIONS,
)

from .latex import (
    LaTeXFormatter,
    Equation,
    get_equation,
    quick_latex,
    COMMON_EQUATIONS,
)

from .charts import (
    ChartGenerator,
    ChartConfig,
    quick_chart,
    SVEND_COLORS,
    SVEND_PALETTE,
)

from .export import (
    DocumentExporter,
    Section,
    quick_export,
)

__all__ = [
    # Chemistry
    "ChemistryFormatter",
    "ReactionInfo",
    "ThermodynamicsData",
    "quick_reaction",
    "BIOCHEMICAL_REACTIONS",
    # LaTeX
    "LaTeXFormatter",
    "Equation",
    "get_equation",
    "quick_latex",
    "COMMON_EQUATIONS",
    # Charts
    "ChartGenerator",
    "ChartConfig",
    "quick_chart",
    "SVEND_COLORS",
    "SVEND_PALETTE",
    # Export
    "DocumentExporter",
    "Section",
    "quick_export",
]

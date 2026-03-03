"""Server-side Plotly JSON -> SVG conversion for report embedding."""

import logging
import uuid

logger = logging.getLogger(__name__)


def plotly_dict_to_svg(plot_dict, width=700, height=400):
    """Convert a Plotly plot dict {data, layout, title} to an SVG string.

    Returns SVG string or None on failure.
    """
    try:
        import plotly.graph_objects as go

        data = plot_dict.get("data", [])
        layout = plot_dict.get("layout", {})

        if "title" not in layout and plot_dict.get("title"):
            layout["title"] = plot_dict["title"]

        layout.setdefault("paper_bgcolor", "white")
        layout.setdefault("plot_bgcolor", "white")
        layout.setdefault("width", width)
        layout.setdefault("height", height)
        if "font" not in layout:
            layout["font"] = {}
        layout["font"].setdefault("size", 11)

        fig = go.Figure(data=data, layout=layout)
        svg_bytes = fig.to_image(format="svg", engine="kaleido")
        return svg_bytes.decode("utf-8")
    except Exception as e:
        logger.warning(f"Chart SVG render failed: {e}")
        return None


def render_dsw_charts(plots, max_charts=10):
    """Render a list of Plotly plot dicts to SVG strings.

    Returns list of {id, svg, title, source, width, height} dicts,
    matching the embedded_diagrams schema.
    """
    results = []
    for plot_dict in (plots or [])[:max_charts]:
        svg = plotly_dict_to_svg(plot_dict)
        if svg:
            results.append({
                "id": uuid.uuid4().hex[:8],
                "svg": svg,
                "title": plot_dict.get("title", "Chart"),
                "source": "dsw",
                "width": plot_dict.get("layout", {}).get("width", 700),
                "height": plot_dict.get("layout", {}).get("height", 400),
            })
    return results

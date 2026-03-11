"""Custom visualization handlers — complex charts requiring manual trace construction.

Extracted from viz monolith. These charts have unique layout patterns
(subplots, parcoords, shapes) that don't fit the spec-driven engine.

CR: 3c0d0e53
"""

import numpy as np

from ..common import SVEND_COLORS


def run_matrix(df, config):
    result = {"plots": [], "summary": ""}

    vars_list = config.get("vars", [])

    if len(vars_list) < 2:
        result["summary"] = "Please select at least 2 variables for matrix plot."
        return result

    # Create scatter matrix
    n_vars = len(vars_list)
    fig_data = []

    for i, y_var in enumerate(vars_list):
        for j, x_var in enumerate(vars_list):
            row = n_vars - i
            col = j + 1

            if i == j:
                # Diagonal: histogram
                fig_data.append(
                    {
                        "type": "histogram",
                        "x": df[x_var].dropna().tolist(),
                        "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}},
                        "xaxis": f"x{col if col > 1 else ''}",
                        "yaxis": f"y{row if row > 1 else ''}",
                        "showlegend": False,
                    }
                )
            else:
                # Off-diagonal: scatter
                fig_data.append(
                    {
                        "type": "scatter",
                        "x": df[x_var].tolist(),
                        "y": df[y_var].tolist(),
                        "mode": "markers",
                        "marker": {"color": "#4a9f6e", "size": 3},
                        "xaxis": f"x{col if col > 1 else ''}",
                        "yaxis": f"y{row if row > 1 else ''}",
                        "showlegend": False,
                    }
                )

    # Build layout with subplots
    layout = {
        "height": 100 + n_vars * 120,
        "showlegend": False,
    }

    # Create axis layout
    for i in range(n_vars):
        col = i + 1
        row = n_vars - i
        x_key = f"xaxis{col if col > 1 else ''}"
        y_key = f"yaxis{row if row > 1 else ''}"

        layout[x_key] = {
            "domain": [i / n_vars + 0.02, (i + 1) / n_vars - 0.02],
            "title": vars_list[i] if row == 1 else "",
            "showticklabels": row == 1,
        }
        layout[y_key] = {
            "domain": [i / n_vars + 0.02, (i + 1) / n_vars - 0.02],
            "title": vars_list[n_vars - 1 - i] if col == 1 else "",
            "showticklabels": col == 1,
        }

    result["plots"].append({"title": "Matrix Plot", "data": fig_data, "layout": layout})

    return result


def run_parallel_coordinates(df, config):
    result = {"plots": [], "summary": ""}

    dims = config.get("dimensions", [])
    color_col = config.get("color")

    if len(dims) < 2:
        result["summary"] = "Select at least 2 dimensions for parallel coordinates."
        return result

    dimensions = []
    for col in dims:
        if np.issubdtype(df[col].dtype, np.number):
            dimensions.append(
                {
                    "label": col,
                    "values": df[col].fillna(df[col].median()).tolist(),
                    "range": [float(df[col].min()), float(df[col].max())],
                }
            )
        else:
            cats = df[col].dropna().unique().tolist()
            cat_map = {c: i for i, c in enumerate(cats)}
            dimensions.append(
                {
                    "label": col,
                    "values": df[col].map(cat_map).fillna(-1).astype(int).tolist(),
                    "tickvals": list(range(len(cats))),
                    "ticktext": [str(c) for c in cats],
                }
            )

    trace = {"type": "parcoords", "dimensions": dimensions}
    if color_col and color_col not in ("", "None"):
        if np.issubdtype(df[color_col].dtype, np.number):
            trace["line"] = {
                "color": df[color_col].fillna(0).tolist(),
                "colorscale": [[0, "#4a9f6e"], [0.5, "#e8c547"], [1, "#d94a4a"]],
                "showscale": True,
                "colorbar": {"title": color_col},
            }
        else:
            cats = df[color_col].dropna().unique().tolist()
            cat_map = {c: i for i, c in enumerate(cats)}
            trace["line"] = {
                "color": df[color_col].map(cat_map).fillna(0).tolist(),
                "colorscale": [[0, "#4a9f6e"], [0.5, "#4a90d9"], [1, "#d94a4a"]],
                "showscale": True,
            }
    else:
        trace["line"] = {"color": "#4a9f6e"}

    result["plots"].append(
        {
            "title": f"Parallel Coordinates ({len(dims)} dimensions)",
            "data": [trace],
            "layout": {"height": 450},
        }
    )
    result["summary"] = (
        f"Parallel Coordinates Plot\n\n"
        f"Dimensions: {len(dims)}\nObservations: {len(df)}\n\n"
        f"Drag axis ranges to filter. Reorder axes by dragging labels."
    )

    return result


def run_mosaic(df, config):
    import pandas as pd

    result = {"plots": [], "summary": ""}
    theme_colors = SVEND_COLORS

    row_var = config.get("row_var")
    col_var = config.get("col_var")

    # Cap levels
    if df[row_var].nunique() > 15:
        top_rows = df[row_var].value_counts().head(15).index
        df = df[df[row_var].isin(top_rows)]
    if df[col_var].nunique() > 15:
        top_cols = df[col_var].value_counts().head(15).index
        df = df[df[col_var].isin(top_cols)]

    ct = pd.crosstab(df[row_var], df[col_var])
    row_totals = ct.sum(axis=1)
    grand_total = int(ct.values.sum())
    col_names = ct.columns.tolist()
    row_names = ct.index.tolist()

    shapes = []
    annotations = []
    x_cursor = 0.0

    for ri, row_name in enumerate(row_names):
        row_width = float(row_totals[row_name]) / grand_total
        y_cursor = 0.0
        col_total = float(row_totals[row_name])

        for ci, col_name in enumerate(col_names):
            cell_val = float(ct.loc[row_name, col_name])
            cell_height = cell_val / col_total if col_total > 0 else 0

            shapes.append(
                {
                    "type": "rect",
                    "x0": x_cursor,
                    "x1": x_cursor + row_width,
                    "y0": y_cursor,
                    "y1": y_cursor + cell_height,
                    "fillcolor": theme_colors[ci % len(theme_colors)],
                    "opacity": 0.7,
                    "line": {"color": "#1a1a2e", "width": 1},
                }
            )
            if cell_height > 0.06 and row_width > 0.06:
                annotations.append(
                    {
                        "x": x_cursor + row_width / 2,
                        "y": y_cursor + cell_height / 2,
                        "text": str(int(cell_val)),
                        "showarrow": False,
                        "font": {"color": "#fff", "size": 10},
                    }
                )
            y_cursor += cell_height

        annotations.append(
            {
                "x": x_cursor + row_width / 2,
                "y": -0.04,
                "text": str(row_name),
                "showarrow": False,
                "font": {"color": "#b0b0b0", "size": 9},
            }
        )
        x_cursor += row_width

    # Legend traces for column categories
    legend_traces = [
        {
            "type": "scatter",
            "x": [None],
            "y": [None],
            "mode": "markers",
            "marker": {"color": theme_colors[ci % len(theme_colors)], "size": 10},
            "name": str(col_name),
            "showlegend": True,
        }
        for ci, col_name in enumerate(col_names)
    ]

    result["plots"].append(
        {
            "title": f"Mosaic: {row_var} x {col_var}",
            "data": legend_traces,
            "layout": {
                "height": 400,
                "shapes": shapes,
                "annotations": annotations,
                "xaxis": {"range": [0, 1], "title": row_var, "showticklabels": False},
                "yaxis": {"range": [-0.08, 1], "title": col_var, "showticklabels": False},
                "showlegend": True,
            },
        }
    )
    result["summary"] = (
        f"Mosaic Plot\n\nRow: {row_var} ({len(row_names)} levels)\n"
        f"Column: {col_var} ({len(col_names)} levels)\n"
        f"Total: {grand_total} observations\n\n"
        f"Tile widths proportional to row totals. Heights proportional to column distribution within each row."
    )

    return result


_CUSTOM_DISPATCH = {
    "matrix": run_matrix,
    "parallel_coordinates": run_parallel_coordinates,
    "mosaic": run_mosaic,
}

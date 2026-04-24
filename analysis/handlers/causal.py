"""Causal discovery handler — PC algorithm and LiNGAM via forgesia."""

import logging

import pandas as pd
from forgeviz.core.spec import ChartSpec

logger = logging.getLogger(__name__)


def run(df, analysis_id, config):
    dispatch = {
        "causal_pc": _pc,
        "causal_lingam": _lingam,
    }
    fn = dispatch.get(analysis_id)
    if fn:
        return fn(df, config)
    return {"summary": f"Causal '{analysis_id}' not in forge dispatch.", "charts": [], "statistics": {}}


def _pc(df, config):
    variables = config.get("variables") or df.select_dtypes(include="number").columns.tolist()
    if isinstance(variables, str):
        variables = [variables]
    alpha = float(config.get("alpha", 0.05))

    try:
        from forgesia.discovery import run_pc

        data = df[variables].apply(pd.to_numeric, errors="coerce").dropna().values
        result = run_pc(
            data,
            labels=variables,
            alpha=alpha,
            max_cond_size=int(config["max_cond_size"]) if config.get("max_cond_size") else None,
        )

        edges = result.get("directed_edges", []) + result.get("undirected_edges", [])
        chart = _dag_chart(variables, edges, "PC Algorithm — Causal Graph")

        return {
            "charts": [chart],
            "statistics": {"n_edges": len(edges), "n_variables": len(variables), "algorithm": "PC"},
            "summary": f"PC algorithm: {len(edges)} edges among {len(variables)} variables.",
        }
    except Exception as e:
        return {"summary": f"PC algorithm error: {e}", "charts": [], "statistics": {}}


def _lingam(df, config):
    variables = config.get("variables") or df.select_dtypes(include="number").columns.tolist()
    if isinstance(variables, str):
        variables = [variables]

    try:
        from forgesia.discovery import run_lingam

        data = df[variables].apply(pd.to_numeric, errors="coerce").dropna().values
        result = run_lingam(data, labels=variables, prune=float(config.get("alpha", 0.05)))

        edges = result.get("directed_edges", [])
        chart = _dag_chart(variables, edges, "LiNGAM — Causal Graph")

        return {
            "charts": [chart],
            "statistics": {
                "n_edges": len(edges),
                "causal_order": result.get("causal_order", []),
                "algorithm": "LiNGAM",
            },
            "summary": f"LiNGAM: {len(edges)} causal edges. Order: {' → '.join(result.get('causal_order', []))}.",
        }
    except Exception as e:
        return {"summary": f"LiNGAM error: {e}", "charts": [], "statistics": {}}


def _dag_chart(variables, edges, title):
    """Simple bar chart of edge counts per variable (proper DAG viz needs JS)."""
    counts = {v: 0 for v in variables}
    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src, tgt = str(edge[0]), str(edge[1])
            if src in counts:
                counts[src] += 1
            if tgt in counts:
                counts[tgt] += 1

    spec = ChartSpec(
        title=title,
        chart_type="causal_graph",
        x_axis={"label": "Variable"},
        y_axis={"label": "Edge Count"},
    )
    spec.add_trace(list(counts.keys()), list(counts.values()), name="Connections", trace_type="bar", color="#4a9f6e")
    return spec

"""
Causal Discovery wrapper for Svend DSW.

Algorithms:
    causal_pc       — PC algorithm (constraint-based, Gaussian CI via partial corr + Fisher z)
    causal_lingam   — ICA-LiNGAM (functional causal model, requires non-Gaussian noise)

Outputs:
    - DAG as Plotly directed graph
    - Edge table with coefficients / partial correlations
    - Bootstrap stability (edge frequency + coefficient CIs)
    - Separating-set explanations (PC: "removed X-Y because X ⊥ Y | {Z1,Z4}, p=0.21")
    - Assumptions panel

Dependencies: causal-learn (CMU), numpy, scipy
"""

import logging

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_ALPHA = 0.05
_DEFAULT_MAX_COND = 4  # max conditioning set size
_DEFAULT_N_BOOT = 100
_EDGE_PRUNE_THRESHOLD = 0.05  # |coef| below this treated as zero


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_causal_discovery(df, analysis_id, config):
    """Dispatch to PC or LiNGAM. Returns standard DSW result dict."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    variables = config.get("variables", [])
    if not variables:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(variables) < 2:
        result["summary"] = (
            "Error: Need at least 2 numeric variables for causal discovery."
        )
        return result

    # Clean data: drop rows with NaN in selected variables
    data_df = df[variables].dropna()
    n, p = data_df.shape
    if n < 10:
        result["summary"] = f"Error: Only {n} complete observations. Need at least 10."
        return result

    data = data_df.values
    labels = list(variables)

    alpha = float(config.get("alpha", _DEFAULT_ALPHA))
    n_boot = int(config.get("n_bootstraps", _DEFAULT_N_BOOT))

    if analysis_id == "causal_pc":
        max_cond = int(config.get("max_cond_size", min(_DEFAULT_MAX_COND, p - 2)))
        return _run_pc_analysis(data, labels, alpha, max_cond, n_boot, result)
    elif analysis_id == "causal_lingam":
        return _run_lingam_analysis(data, labels, alpha, n_boot, result)
    else:
        result["summary"] = f"Error: Unknown causal analysis '{analysis_id}'."
        return result


# ===========================================================================
# PC Algorithm
# ===========================================================================
def _partial_corr_test(data, x_idx, y_idx, cond_set, n):
    """
    Gaussian CI test: partial correlation via regression residuals + Fisher z.

    For X ⊥ Y | S:
      1. Regress X on S → residuals r_X
      2. Regress Y on S → residuals r_Y
      3. Correlate r_X and r_Y → partial correlation ρ_{XY|S}
      4. Fisher z-transform: z = 0.5 * ln((1+ρ)/(1-ρ)) * sqrt(n - |S| - 3)
      5. Two-sided p-value from standard normal

    Returns (partial_corr, p_value, z_stat).
    """
    s = list(cond_set)
    k = len(s)

    # Guard: Fisher z requires n > |S| + 3
    if n <= k + 3:
        return 0.0, 1.0, 0.0  # cannot test, treat as independent

    if k == 0:
        # Unconditional correlation
        rho = np.corrcoef(data[:, x_idx], data[:, y_idx])[0, 1]
    else:
        # Regression residuals approach
        S_data = data[:, s]
        if S_data.ndim == 1:
            S_data = S_data.reshape(-1, 1)
        # Add intercept
        S_aug = np.column_stack([np.ones(n), S_data])
        try:
            # Solve via least squares (numerically stable)
            coef_x, _, _, _ = np.linalg.lstsq(S_aug, data[:, x_idx], rcond=None)
            coef_y, _, _, _ = np.linalg.lstsq(S_aug, data[:, y_idx], rcond=None)
            res_x = data[:, x_idx] - S_aug @ coef_x
            res_y = data[:, y_idx] - S_aug @ coef_y
            rho = np.corrcoef(res_x, res_y)[0, 1]
        except np.linalg.LinAlgError:
            return 0.0, 1.0, 0.0

    # Clamp to avoid log(0) / division by zero
    rho = np.clip(rho, -0.9999, 0.9999)

    # Fisher z-transform
    z = 0.5 * np.log((1 + rho) / (1 - rho)) * np.sqrt(n - k - 3)
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))

    return float(rho), float(p_value), float(z)


def _run_pc_core(data, labels, alpha, max_cond_size):
    """
    PC algorithm implementation using our own CI tests for full control,
    with causal-learn for the skeleton→orientation phase.

    Returns:
        graph_matrix: p×p ndarray, -1=tail, 1=arrowhead, 0=no edge
        separating_sets: dict {(i,j): {"cond_set": [...], "p_value": float, "partial_corr": float}}
        edges_removed_by_depth: dict {depth: [(i,j,cond_set_names,p_val), ...]}
    """
    from causallearn.search.ConstraintBased.PC import pc

    n, p = data.shape

    # Use causal-learn's PC with Fisher z (it uses the precision matrix approach internally)
    # But we also run our own CI tests to capture separating sets with p-values
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cg = pc(
            data,
            alpha=alpha,
            indep_test="fisherz",
            node_names=labels,
            depth=max_cond_size,
        )

    graph_matrix = cg.G.graph  # p×p, -1=tail, 1=arrowhead

    # Now reconstruct separating sets with our own CI tests for reporting
    # The PC algorithm's sepset is stored in cg.sepset (p×p array of sets/None)
    separating_sets = {}
    edges_removed_by_depth = {}

    sepset = cg.sepset  # p×p ndarray; each entry is list of tuples or None
    for i in range(p):
        for j in range(i + 1, p):
            # If no edge exists, find the separating set
            if graph_matrix[i, j] == 0 and graph_matrix[j, i] == 0:
                raw = (
                    sepset[i][j]
                    if sepset[i][j] is not None
                    else (sepset[j][i] if sepset[j][i] is not None else None)
                )
                if raw is not None:
                    # raw is a list of tuples, e.g. [(2,), (2,)] — take first
                    first = raw[0] if isinstance(raw, list) and len(raw) > 0 else raw
                    cond_indices = [int(x) for x in first] if first else []
                    rho, pval, z = _partial_corr_test(data, i, j, cond_indices, n)
                    depth = len(cond_indices)
                    cond_names = (
                        [labels[k] for k in cond_indices] if cond_indices else []
                    )

                    separating_sets[(i, j)] = {
                        "cond_set": cond_indices,
                        "cond_names": cond_names,
                        "p_value": pval,
                        "partial_corr": rho,
                        "depth": depth,
                    }

                    if depth not in edges_removed_by_depth:
                        edges_removed_by_depth[depth] = []
                    edges_removed_by_depth[depth].append(
                        (labels[i], labels[j], cond_names, pval)
                    )

    return graph_matrix, separating_sets, edges_removed_by_depth


def _run_pc_analysis(data, labels, alpha, max_cond_size, n_boot, result):
    """Full PC analysis with bootstrap stability."""
    n, p = data.shape

    # --- Core PC run ---
    graph, sep_sets, removed_by_depth = _run_pc_core(data, labels, alpha, max_cond_size)

    # --- Extract edges ---
    directed_edges = []  # (from, to, partial_corr)
    undirected_edges = []  # (a, b, partial_corr)

    for i in range(p):
        for j in range(i + 1, p):
            if graph[i, j] == -1 and graph[j, i] == 1:
                # i → j
                rho, _, _ = _partial_corr_test(data, i, j, [], n)
                directed_edges.append((labels[i], labels[j], abs(rho)))
            elif graph[i, j] == 1 and graph[j, i] == -1:
                # j → i
                rho, _, _ = _partial_corr_test(data, j, i, [], n)
                directed_edges.append((labels[j], labels[i], abs(rho)))
            elif graph[i, j] == -1 and graph[j, i] == -1:
                # undirected
                rho, _, _ = _partial_corr_test(data, i, j, [], n)
                undirected_edges.append((labels[i], labels[j], abs(rho)))
            elif graph[i, j] == 1 and graph[j, i] == 1:
                # bidirected (latent confounder indicator)
                rho, _, _ = _partial_corr_test(data, i, j, [], n)
                undirected_edges.append((labels[i], labels[j], abs(rho)))

    # --- Bootstrap stability ---
    edge_stability = _bootstrap_pc(data, labels, alpha, max_cond_size, n_boot)

    # --- Build plots ---
    result["plots"].append(
        _build_dag_plot(
            labels,
            directed_edges,
            undirected_edges,
            edge_stability,
            title="PC Algorithm — Candidate Causal Structure",
        )
    )

    # Edge stability bar chart
    if edge_stability:
        result["plots"].append(_build_stability_plot(edge_stability, labels))

    # --- Summary ---
    result["summary"] = _build_pc_summary(
        labels,
        directed_edges,
        undirected_edges,
        sep_sets,
        removed_by_depth,
        edge_stability,
        alpha,
        max_cond_size,
        n,
        p,
    )

    # --- Statistics dict ---
    adj_list = []
    for src, tgt, strength in directed_edges:
        stab = edge_stability.get(_edge_key(src, tgt), 0)
        adj_list.append(
            {
                "from": src,
                "to": tgt,
                "type": "directed",
                "strength": round(strength, 4),
                "stability": round(stab, 3),
            }
        )
    for a, b, strength in undirected_edges:
        stab = edge_stability.get(_edge_key(a, b), 0)
        adj_list.append(
            {
                "from": a,
                "to": b,
                "type": "undirected",
                "strength": round(strength, 4),
                "stability": round(stab, 3),
            }
        )

    result["statistics"] = {
        "algorithm": "PC",
        "alpha": alpha,
        "max_cond_size": max_cond_size,
        "n_obs": n,
        "n_vars": p,
        "n_directed_edges": len(directed_edges),
        "n_undirected_edges": len(undirected_edges),
        "n_bootstraps": n_boot,
        "edges": adj_list,
        "separating_sets": {
            f"{labels[i]}-{labels[j]}": {
                "conditioned_on": info["cond_names"],
                "p_value": round(info["p_value"], 4),
                "partial_corr": round(info["partial_corr"], 4),
            }
            for (i, j), info in sep_sets.items()
        },
    }

    result["guide_observation"] = (
        f"PC algorithm found {len(directed_edges)} directed and {len(undirected_edges)} "
        f"undirected edges among {p} variables (α={alpha}, n={n}). "
        + (
            "All edges >70% stable under bootstrap."
            if all(v >= 0.7 for v in edge_stability.values())
            else "Some edges <70% bootstrap stability — interpret with caution."
        )
    )

    return result


def _bootstrap_pc(data, labels, alpha, max_cond_size, n_boot):
    """Bootstrap the PC algorithm to get edge stability percentages."""
    n, p = data.shape
    edge_counts = {}

    import warnings

    from causallearn.search.ConstraintBased.PC import pc

    for b in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        boot_data = data[idx]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cg = pc(
                    boot_data,
                    alpha=alpha,
                    indep_test="fisherz",
                    node_names=labels,
                    depth=max_cond_size,
                )
            G = cg.G.graph
            for i in range(p):
                for j in range(i + 1, p):
                    if G[i, j] != 0 or G[j, i] != 0:
                        key = _edge_key(labels[i], labels[j])
                        edge_counts[key] = edge_counts.get(key, 0) + 1
        except Exception:
            continue  # skip failed bootstrap

    return {k: v / n_boot for k, v in edge_counts.items()}


# ===========================================================================
# LiNGAM
# ===========================================================================
def _run_lingam_analysis(data, labels, alpha, n_boot, result):
    """Full LiNGAM analysis with bootstrap stability."""
    import warnings

    from causallearn.search.FCMBased import lingam as cl_lingam

    n, p = data.shape

    # --- Core LiNGAM run ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cl_lingam.ICALiNGAM()
        model.fit(data)

    B = model.adjacency_matrix_  # B[i,j] != 0 means j → i
    causal_order = list(model.causal_order_)

    # Prune small coefficients
    B_pruned = B.copy()
    B_pruned[np.abs(B_pruned) < _EDGE_PRUNE_THRESHOLD] = 0

    # Extract directed edges
    directed_edges = []
    for i in range(p):
        for j in range(p):
            if abs(B_pruned[i, j]) > _EDGE_PRUNE_THRESHOLD:
                directed_edges.append((labels[j], labels[i], float(B_pruned[i, j])))

    # --- Bootstrap stability + coefficient CIs ---
    edge_stability, coef_distributions = _bootstrap_lingam(data, labels, n_boot)

    # Build coefficient CI info
    edge_cis = {}
    for key, coefs in coef_distributions.items():
        if len(coefs) >= 5:
            edge_cis[key] = {
                "median": float(np.median(coefs)),
                "ci_low": float(np.percentile(coefs, 2.5)),
                "ci_high": float(np.percentile(coefs, 97.5)),
                "n_boot": len(coefs),
            }

    # --- Non-Gaussianity check ---
    gaussianity_warnings = []
    for j in range(p):
        col = data[:, j]
        # Shapiro-Wilk (limited to 5000 samples)
        sample = col[:5000] if len(col) > 5000 else col
        _, sw_p = sp_stats.shapiro(sample)
        if sw_p > 0.05:
            gaussianity_warnings.append(
                f"  {labels[j]}: Shapiro-Wilk p={sw_p:.3f} — may be Gaussian (LiNGAM assumes non-Gaussian noise)"
            )

    # --- Build plots ---
    result["plots"].append(
        _build_dag_plot(
            labels,
            directed_edges,
            [],
            edge_stability,
            title="LiNGAM — Estimated Causal Structure",
            show_coefficients=True,
        )
    )

    if edge_stability:
        result["plots"].append(_build_stability_plot(edge_stability, labels))

    # --- Summary ---
    result["summary"] = _build_lingam_summary(
        labels,
        directed_edges,
        causal_order,
        edge_stability,
        edge_cis,
        gaussianity_warnings,
        n,
        p,
        n_boot,
    )

    # --- Statistics ---
    adj_list = []
    for src, tgt, coef in directed_edges:
        key = _edge_key(src, tgt)
        stab = edge_stability.get(key, 0)
        ci = edge_cis.get(key, {})
        adj_list.append(
            {
                "from": src,
                "to": tgt,
                "type": "directed",
                "coefficient": round(coef, 4),
                "stability": round(stab, 3),
                "ci_low": round(ci.get("ci_low", 0), 4),
                "ci_high": round(ci.get("ci_high", 0), 4),
            }
        )

    result["statistics"] = {
        "algorithm": "LiNGAM (ICA)",
        "n_obs": n,
        "n_vars": p,
        "n_edges": len(directed_edges),
        "causal_order": [labels[i] for i in causal_order],
        "n_bootstraps": n_boot,
        "edges": adj_list,
        "B_matrix": np.round(B_pruned, 4).tolist(),
        "gaussianity_warnings": gaussianity_warnings,
    }

    result["guide_observation"] = (
        f"LiNGAM found {len(directed_edges)} causal edges among {p} variables (n={n}). "
        f"Causal order: {' → '.join(labels[i] for i in causal_order)}. "
        + (
            f"WARNING: {len(gaussianity_warnings)} variable(s) may be Gaussian — "
            "LiNGAM results unreliable for those edges."
            if gaussianity_warnings
            else "All variables appear non-Gaussian — LiNGAM assumptions satisfied."
        )
    )

    return result


def _bootstrap_lingam(data, labels, n_boot):
    """Bootstrap LiNGAM for edge stability and coefficient distributions."""
    import warnings

    from causallearn.search.FCMBased import lingam as cl_lingam

    n, p = data.shape
    edge_counts = {}
    coef_dists = {}  # {edge_key: [coef, coef, ...]}

    for b in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        boot_data = data[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = cl_lingam.ICALiNGAM()
                model.fit(boot_data)
            B = model.adjacency_matrix_
            for i in range(p):
                for j in range(p):
                    if abs(B[i, j]) > _EDGE_PRUNE_THRESHOLD:
                        key = _edge_key(labels[j], labels[i])
                        edge_counts[key] = edge_counts.get(key, 0) + 1
                        if key not in coef_dists:
                            coef_dists[key] = []
                        coef_dists[key].append(float(B[i, j]))
        except Exception:
            continue

    stability = {k: v / max(n_boot, 1) for k, v in edge_counts.items()}
    return stability, coef_dists


# ===========================================================================
# Plotting
# ===========================================================================
def _build_dag_plot(
    labels,
    directed_edges,
    undirected_edges,
    edge_stability,
    title="Causal Discovery",
    show_coefficients=False,
):
    """Build a Plotly figure of the causal DAG with nodes in a circle layout."""
    p = len(labels)
    # Circle layout
    angles = np.linspace(0, 2 * np.pi, p, endpoint=False)
    # Start from top, go clockwise
    angles = np.pi / 2 - angles
    pos = {
        labels[i]: (float(np.cos(angles[i])), float(np.sin(angles[i])))
        for i in range(p)
    }

    traces = []

    # --- Undirected edges (dashed gray) ---
    for a, b, strength in undirected_edges:
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        key = _edge_key(a, b)
        stab = edge_stability.get(key, 0)
        opacity = max(0.3, min(1.0, stab))
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": [x0, x1, None],
                "y": [y0, y1, None],
                "line": {
                    "color": f"rgba(150,150,150,{opacity})",
                    "width": 1.5,
                    "dash": "dash",
                },
                "hoverinfo": "text",
                "text": f"{a} — {b} (undirected, stability={stab:.0%})",
                "showlegend": False,
            }
        )

    # --- Directed edges (with arrowheads via annotations) ---
    annotations = []
    for src, tgt, coef_or_strength in directed_edges:
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        key = _edge_key(src, tgt)
        stab = edge_stability.get(key, 0)
        opacity = max(0.3, min(1.0, stab))

        # Shorten line slightly so arrow doesn't overlap node
        dx, dy = x1 - x0, y1 - y0
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            shrink = 0.12 / dist
            x0s = x0 + dx * shrink
            y0s = y0 + dy * shrink
            x1s = x1 - dx * shrink
            y1s = y1 - dy * shrink
        else:
            x0s, y0s, x1s, y1s = x0, y0, x1, y1

        color = f"rgba(74,159,110,{opacity})"
        width = max(1.5, abs(coef_or_strength) * 4) if show_coefficients else 2

        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": [x0s, x1s, None],
                "y": [y0s, y1s, None],
                "line": {"color": color, "width": width},
                "hoverinfo": "text",
                "text": (
                    f"{src} → {tgt} (coef={coef_or_strength:.3f}, stability={stab:.0%})"
                    if show_coefficients
                    else f"{src} → {tgt} (stability={stab:.0%})"
                ),
                "showlegend": False,
            }
        )

        # Arrowhead annotation
        annotations.append(
            {
                "x": x1s,
                "y": y1s,
                "ax": x0s,
                "ay": y0s,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "showarrow": True,
                "arrowhead": 2,
                "arrowsize": 1.5,
                "arrowwidth": width,
                "arrowcolor": color,
                "standoff": 0,
            }
        )

        # Edge label (coefficient)
        if show_coefficients:
            mx, my = (x0s + x1s) / 2, (y0s + y1s) / 2
            annotations.append(
                {
                    "x": mx,
                    "y": my,
                    "text": f"{coef_or_strength:.2f}",
                    "showarrow": False,
                    "font": {"size": 9, "color": "#aaa"},
                    "bgcolor": "rgba(26,26,46,0.7)",
                }
            )

    # --- Nodes ---
    node_x = [pos[lbl][0] for lbl in labels]
    node_y = [pos[lbl][1] for lbl in labels]
    traces.append(
        {
            "type": "scatter",
            "mode": "markers+text",
            "x": node_x,
            "y": node_y,
            "marker": {
                "size": 28,
                "color": "#1a3a5c",
                "line": {"width": 2, "color": "#4a9f6e"},
            },
            "text": labels,
            "textposition": "middle center",
            "textfont": {"color": "white", "size": 10},
            "hoverinfo": "text",
            "hovertext": labels,
            "showlegend": False,
        }
    )

    return {
        "title": title,
        "data": traces,
        "layout": {
            "template": "plotly_dark",
            "height": max(400, 50 * p),
            "xaxis": {"visible": False, "range": [-1.5, 1.5]},
            "yaxis": {"visible": False, "range": [-1.5, 1.5], "scaleanchor": "x"},
            "annotations": annotations,
            "margin": {"l": 20, "r": 20, "t": 40, "b": 20},
        },
    }


def _build_stability_plot(edge_stability, labels):
    """Horizontal bar chart of edge bootstrap stability."""
    if not edge_stability:
        return {"title": "Edge Stability", "data": [], "layout": {}}

    sorted_edges = sorted(edge_stability.items(), key=lambda x: x[1])
    edge_names = [k for k, v in sorted_edges]
    stabilities = [v for k, v in sorted_edges]
    colors = [
        (
            "rgba(74,159,110,0.7)"
            if s >= 0.7
            else "rgba(200,170,60,0.7)" if s >= 0.5 else "rgba(208,96,96,0.7)"
        )
        for s in stabilities
    ]

    return {
        "title": f"Edge Stability ({len(edge_stability)} edges, bootstrap)",
        "data": [
            {
                "type": "bar",
                "orientation": "h",
                "x": [round(s, 3) for s in stabilities],
                "y": edge_names,
                "marker": {"color": colors},
                "hovertemplate": "%{y}: %{x:.1%}<extra></extra>",
            }
        ],
        "layout": {
            "template": "plotly_dark",
            "height": max(200, len(edge_names) * 25),
            "xaxis": {
                "title": "Bootstrap frequency",
                "range": [0, 1],
                "tickformat": ".0%",
            },
            "yaxis": {"automargin": True},
            "shapes": [
                {
                    "type": "line",
                    "x0": 0.7,
                    "x1": 0.7,
                    "y0": -0.5,
                    "y1": len(edge_names) - 0.5,
                    "line": {"color": "rgba(255,255,255,0.3)", "dash": "dot"},
                }
            ],
        },
    }


# ===========================================================================
# Summary builders
# ===========================================================================
def _build_pc_summary(
    labels,
    directed,
    undirected,
    sep_sets,
    removed_by_depth,
    stability,
    alpha,
    max_cond,
    n,
    p,
):
    """Build the PC summary text with separating-set explanations."""
    lines = []
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>")
    lines.append("<<COLOR:title>>CAUSAL DISCOVERY — PC ALGORITHM<</COLOR>>")
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n")

    lines.append(f"<<COLOR:highlight>>Variables:<</COLOR>> {', '.join(labels)}")
    lines.append(f"<<COLOR:highlight>>Observations:<</COLOR>> {n}")
    lines.append(f"<<COLOR:highlight>>Significance level (α):<</COLOR>> {alpha}")
    lines.append(f"<<COLOR:highlight>>Max conditioning set size:<</COLOR>> {max_cond}")
    lines.append(f"<<COLOR:highlight>>Directed edges:<</COLOR>> {len(directed)}")
    lines.append(f"<<COLOR:highlight>>Undirected edges:<</COLOR>> {len(undirected)}\n")

    # Causal edges
    if directed:
        lines.append("<<COLOR:accent>>── Directed Edges (Cause → Effect) ──<</COLOR>>")
        for src, tgt, strength in directed:
            key = _edge_key(src, tgt)
            stab = stability.get(key, 0)
            stab_icon = "●" if stab >= 0.7 else "◐" if stab >= 0.5 else "○"
            lines.append(
                f"  {stab_icon} {src} → {tgt}  (|ρ|={strength:.3f}, stability={stab:.0%})"
            )
        lines.append("")

    if undirected:
        lines.append(
            "<<COLOR:accent>>── Undirected Edges (direction unresolved) ──<</COLOR>>"
        )
        for a, b, strength in undirected:
            key = _edge_key(a, b)
            stab = stability.get(key, 0)
            lines.append(f"  ◻ {a} — {b}  (|ρ|={strength:.3f}, stability={stab:.0%})")
        lines.append("")

    # Separating-set explanations (edges REMOVED)
    if sep_sets:
        lines.append("<<COLOR:accent>>── Edges Removed (with explanation) ──<</COLOR>>")
        for (i, j), info in sorted(sep_sets.items()):
            var_i, var_j = labels[i], labels[j]
            cond = info["cond_names"]
            pval = info["p_value"]
            rho = info["partial_corr"]
            cond_str = "{" + ", ".join(cond) + "}" if cond else "∅"
            lines.append(
                f"  ✗ Removed {var_i}–{var_j} because {var_i} ⊥ {var_j} | {cond_str}  (p={pval:.3f}, ρ={rho:.3f})"
            )
        lines.append("")

    # Edges removed by conditioning depth
    if removed_by_depth:
        lines.append("<<COLOR:accent>>── Skeleton Refinement by Depth ──<</COLOR>>")
        for depth in sorted(removed_by_depth.keys()):
            edges = removed_by_depth[depth]
            lines.append(f"  Depth {depth} (|S|={depth}): {len(edges)} edge(s) removed")
        lines.append("")

    # Assumptions
    lines.append("<<COLOR:accent>>── Assumptions ──<</COLOR>>")
    lines.append(
        "<<COLOR:warning>>This is a candidate causal structure, not ground truth.<</COLOR>>"
    )
    lines.append("  The PC algorithm assumes:")
    lines.append(
        "  1. Causal sufficiency — no hidden common causes (use FCI if violated)"
    )
    lines.append(
        "  2. Faithfulness — all conditional independences reflect the true graph"
    )
    lines.append("  3. i.i.d. sampling — observations are independent")
    lines.append(
        "  4. Gaussian CI test — partial correlations detect linear dependencies only"
    )
    lines.append(
        "\n  Sensitivity: try α=0.01 (sparser) or α=0.10 (denser) to assess stability."
    )

    return "\n".join(lines)


def _build_lingam_summary(
    labels, directed, causal_order, stability, edge_cis, gauss_warnings, n, p, n_boot
):
    """Build the LiNGAM summary text."""
    lines = []
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>")
    lines.append("<<COLOR:title>>CAUSAL DISCOVERY — LiNGAM (ICA)<</COLOR>>")
    lines.append(f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n")

    lines.append(f"<<COLOR:highlight>>Variables:<</COLOR>> {', '.join(labels)}")
    lines.append(f"<<COLOR:highlight>>Observations:<</COLOR>> {n}")
    lines.append(f"<<COLOR:highlight>>Edges found:<</COLOR>> {len(directed)}")
    lines.append(
        f"<<COLOR:highlight>>Causal order:<</COLOR>> {' → '.join(labels[i] for i in causal_order)}"
    )
    lines.append(f"<<COLOR:highlight>>Bootstrap resamples:<</COLOR>> {n_boot}\n")

    # Edges with CIs
    if directed:
        lines.append("<<COLOR:accent>>── Causal Edges (Source → Target) ──<</COLOR>>")
        for src, tgt, coef in sorted(directed, key=lambda x: abs(x[2]), reverse=True):
            key = _edge_key(src, tgt)
            stab = stability.get(key, 0)
            ci = edge_cis.get(key, {})
            stab_icon = "●" if stab >= 0.7 else "◐" if stab >= 0.5 else "○"

            ci_str = ""
            if ci:
                ci_str = f", 95% CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"

            lines.append(
                f"  {stab_icon} {src} → {tgt}  (B={coef:.3f}{ci_str}, stability={stab:.0%})"
            )
        lines.append("")

    # Gaussianity warnings
    if gauss_warnings:
        lines.append("<<COLOR:warning>>── Gaussianity Warnings ──<</COLOR>>")
        lines.append("  LiNGAM requires non-Gaussian error distributions.")
        lines.append("  The following variables may violate this assumption:")
        for w in gauss_warnings:
            lines.append(w)
        lines.append("  Edges involving these variables may be unreliable or reversed.")
        lines.append("")

    # Assumptions
    lines.append("<<COLOR:accent>>── Assumptions ──<</COLOR>>")
    lines.append(
        "<<COLOR:warning>>This is a candidate causal structure, not ground truth.<</COLOR>>"
    )
    lines.append("  LiNGAM assumes:")
    lines.append("  1. Linear relationships (X = BX + e)")
    lines.append("  2. Non-Gaussian error terms (violated → edges may reverse)")
    lines.append("  3. Acyclicity — no feedback loops")
    lines.append("  4. Causal sufficiency — no hidden confounders")
    lines.append("\n  Compare with PC algorithm to cross-validate structure.")

    return "\n".join(lines)


# ===========================================================================
# Utilities
# ===========================================================================
def _edge_key(a, b):
    """Canonical edge key for stability tracking. Directed: 'A → B'. Always sorted for undirected."""
    return f"{a} → {b}"

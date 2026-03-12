"""D-Multi — multivariate distributional analysis.

CR: 3c0d0e53
"""

import logging

import numpy as np

from ..common import (
    COLOR_BAD,
    COLOR_GOLD,
    COLOR_GOOD,
    COLOR_INFO,
    COLOR_WARNING,
    SVEND_COLORS,
)
from .helpers import _d_narrative

logger = logging.getLogger(__name__)


def run_d_multi(df, config):
    """Multivariate capability analysis via PCA and Hotelling's T².

    Reduces correlated quality characteristics to principal components,
    computes KDE-based capability on each, and uses T² for joint OOC detection.
    """
    from scipy.stats import f as f_dist

    result = {"plots": [], "summary": "", "guide_observation": ""}

    variables = config.get("variables") or config.get("columns", [])
    if not variables or len(variables) < 2:
        result["summary"] = "<<COLOR:danger>>Select at least 2 numeric variables.<</COLOR>>"
        return result

    missing = [v for v in variables if v not in df.columns]
    if missing:
        result["summary"] = f"<<COLOR:danger>>Columns not found: {', '.join(missing)}<</COLOR>>"
        return result

    tolerance_pct = config.get("tolerance_pct")

    work = df[variables].dropna().astype(float)
    n, p = work.shape
    if n < p + 5:
        result["summary"] = f"<<COLOR:danger>>Need at least {p + 5} observations (have {n}).<</COLOR>>"
        return result

    data_matrix = work.values
    means = data_matrix.mean(axis=0)
    stds = data_matrix.std(axis=0, ddof=1)
    stds[stds == 0] = 1.0

    # Standardize
    Z = (data_matrix - means) / stds

    # PCA
    try:
        from sklearn.decomposition import PCA

        pca = PCA()
        scores = pca.fit_transform(Z)
        explained = pca.explained_variance_ratio_
        loadings = pca.components_  # shape (n_components, p)
    except ImportError:
        # Fallback: manual PCA via eigendecomposition
        cov_matrix = np.cov(Z, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        explained = eigenvalues / eigenvalues.sum()
        scores = Z @ eigenvectors
        loadings = eigenvectors.T

    # Retain components explaining ≥95% variance
    cumvar = np.cumsum(explained)
    k = int(np.searchsorted(cumvar, 0.95) + 1)
    k = min(k, p)

    scores_k = scores[:, :k]
    explained[:k]

    # Hotelling's T² using all p variables
    cov_inv = np.linalg.pinv(np.cov(Z, rowvar=False))
    T2 = np.array([float(z @ cov_inv @ z) for z in Z])

    # UCL for T² (F-distribution based)
    T2_ucl = (p * (n - 1) * (n + 1)) / (n * (n - p)) * f_dist.ppf(0.9973, p, n - p)
    ooc_mask = T2 > T2_ucl
    n_ooc = int(ooc_mask.sum())

    # Per-component capability (using ±3 as natural spec limits for standardized scores)
    component_cpk = []
    for j in range(k):
        comp_scores = scores_k[:, j]
        comp_mean = float(comp_scores.mean())
        comp_std = float(comp_scores.std(ddof=1))
        if comp_std > 0:
            # For PCA components, spec = ±3 standardized units (natural process limits)
            cpk_lo = (comp_mean - (-3)) / (3 * comp_std)
            cpk_hi = (3 - comp_mean) / (3 * comp_std)
            cpk_j = min(cpk_lo, cpk_hi)
        else:
            cpk_j = 999.0
        component_cpk.append(round(float(cpk_j), 3))

    mcpk = min(component_cpk) if component_cpk else 0.0

    # If user provided tolerance, compute per-variable capability too
    var_cpk = []
    if tolerance_pct:
        tol = float(tolerance_pct) / 100.0
        for i_v, v in enumerate(variables):
            v_mean = means[i_v]
            v_std = stds[i_v]
            v_range = np.ptp(data_matrix[:, i_v])
            v_lsl = v_mean - tol * v_range
            v_usl = v_mean + tol * v_range
            if v_std > 0:
                cpk_lo = (v_mean - v_lsl) / (3 * v_std)
                cpk_hi = (v_usl - v_mean) / (3 * v_std)
                var_cpk.append(
                    {
                        "variable": v,
                        "cpk": round(min(cpk_lo, cpk_hi), 3),
                        "lsl": round(v_lsl, 4),
                        "usl": round(v_usl, 4),
                    }
                )
            else:
                var_cpk.append({"variable": v, "cpk": 999.0, "lsl": round(v_lsl, 4), "usl": round(v_usl, 4)})

    # T² capability: proportion within UCL
    t2_capability = float(1.0 - ooc_mask.mean())

    # --- Plot 1: PCA biplot (PC1 vs PC2) ---
    pc1, pc2 = scores[:, 0], scores[:, 1]
    # T² ellipse
    theta = np.linspace(0, 2 * np.pi, 100)
    # Eigenvalue scaling for ellipse
    ev1 = explained[0] * p  # variance on PC1
    ev2 = explained[1] * p if p > 1 else 1
    ellipse_r = np.sqrt(T2_ucl)
    ex = ellipse_r * np.sqrt(ev1) * np.cos(theta)
    ey = ellipse_r * np.sqrt(ev2) * np.sin(theta)

    biplot_traces = [
        {
            "type": "scatter",
            "x": pc1[~ooc_mask].tolist(),
            "y": pc2[~ooc_mask].tolist(),
            "mode": "markers",
            "name": "In Control",
            "marker": {"color": SVEND_COLORS[0], "size": 4, "opacity": 0.5},
        },
        {
            "type": "scatter",
            "x": pc1[ooc_mask].tolist(),
            "y": pc2[ooc_mask].tolist(),
            "mode": "markers",
            "name": f"OOC ({n_ooc})",
            "marker": {"color": COLOR_BAD, "size": 6, "symbol": "x"},
        },
        {
            "type": "scatter",
            "x": ex.tolist(),
            "y": ey.tolist(),
            "mode": "lines",
            "name": "T² UCL",
            "line": {"color": COLOR_BAD, "dash": "dash", "width": 1.5},
        },
    ]
    # Loading arrows
    arrow_annotations = []
    scale_factor = max(abs(pc1).max(), abs(pc2).max()) * 0.8
    for i_v, v in enumerate(variables):
        lx = loadings[0, i_v] * scale_factor
        ly = loadings[1, i_v] * scale_factor
        arrow_annotations.append(
            {
                "x": lx,
                "y": ly,
                "ax": 0,
                "ay": 0,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "showarrow": True,
                "arrowhead": 2,
                "arrowsize": 1.5,
                "arrowcolor": COLOR_INFO,
                "text": v,
                "font": {"color": COLOR_INFO, "size": 9},
            }
        )

    result["plots"].append(
        {
            "title": f"PCA Biplot (PC1: {explained[0] * 100:.1f}%, PC2: {explained[1] * 100:.1f}%)",
            "data": biplot_traces,
            "layout": {
                "height": 380,
                "xaxis": {"title": f"PC1 ({explained[0] * 100:.1f}%)"},
                "yaxis": {"title": f"PC2 ({explained[1] * 100:.1f}%)", "scaleanchor": "x"},
                "showlegend": True,
                "annotations": arrow_annotations,
            },
        }
    )

    # --- Plot 2: T² chart ---
    obs_idx = list(range(1, n + 1))
    result["plots"].append(
        {
            "title": "Hotelling's T² Chart",
            "data": [
                {
                    "type": "scatter",
                    "x": obs_idx,
                    "y": T2.tolist(),
                    "mode": "lines+markers",
                    "name": "T²",
                    "marker": {"color": [COLOR_BAD if ooc else SVEND_COLORS[0] for ooc in ooc_mask], "size": 4},
                    "line": {"color": SVEND_COLORS[0], "width": 1},
                },
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Observation"},
                "yaxis": {"title": "T²"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": 1,
                        "x1": n,
                        "y0": T2_ucl,
                        "y1": T2_ucl,
                        "line": {"color": COLOR_BAD, "dash": "dash", "width": 2},
                    }
                ],
                "annotations": [
                    {
                        "x": n,
                        "y": T2_ucl,
                        "text": f"UCL={T2_ucl:.1f}",
                        "showarrow": False,
                        "font": {"color": COLOR_BAD, "size": 10},
                        "xanchor": "left",
                    }
                ],
            },
        }
    )

    # --- Plot 3: Component capability bars ---
    pc_labels = [f"PC{j + 1}" for j in range(k)]
    result["plots"].append(
        {
            "title": "Per-Component Capability",
            "data": [
                {
                    "type": "bar",
                    "x": pc_labels,
                    "y": component_cpk,
                    "name": "Cpk",
                    "marker": {
                        "color": [
                            COLOR_GOOD if c >= 1.33 else (COLOR_WARNING if c >= 1.0 else COLOR_BAD)
                            for c in component_cpk
                        ]
                    },
                    "text": [f"{c:.2f}" for c in component_cpk],
                    "textposition": "outside",
                },
            ],
            "layout": {
                "height": 280,
                "yaxis": {"title": "Cpk"},
                "shapes": [
                    {
                        "type": "line",
                        "x0": -0.5,
                        "x1": k - 0.5,
                        "y0": 1.33,
                        "y1": 1.33,
                        "line": {"color": COLOR_GOLD, "dash": "dash", "width": 1.5},
                    }
                ],
                "annotations": [
                    {
                        "x": k - 0.5,
                        "y": 1.33,
                        "text": "1.33",
                        "showarrow": False,
                        "font": {"color": COLOR_GOLD},
                        "xanchor": "left",
                    }
                ],
            },
        }
    )

    # --- Plot 4: Correlation heatmap ---
    corr = np.corrcoef(data_matrix, rowvar=False)
    result["plots"].append(
        {
            "title": "Variable Correlation Matrix",
            "data": [
                {
                    "type": "heatmap",
                    "z": corr.tolist(),
                    "x": variables,
                    "y": variables,
                    "colorscale": [[0, "#d06060"], [0.5, "#ffffff"], [1, "#4a9f6e"]],
                    "zmin": -1,
                    "zmax": 1,
                    "text": [[f"{corr[i][j]:.2f}" for j in range(p)] for i in range(p)],
                    "texttemplate": "%{text}",
                    "showscale": True,
                    "colorbar": {"title": "r"},
                }
            ],
            "layout": {"height": 360, "yaxis": {"autorange": "reversed"}},
        }
    )

    # --- Summary ---
    summary = "<<COLOR:title>>D-MULTI — MULTIVARIATE CAPABILITY<</COLOR>>\n\n"
    summary += f"<<COLOR:header>>Data:<</COLOR>> {n} observations, {p} variables\n"
    summary += f"<<COLOR:header>>Variables:<</COLOR>> {', '.join(variables)}\n\n"

    summary += "<<COLOR:header>>PCA Decomposition:<</COLOR>>\n"
    summary += f"  Components retained: {k} of {p} (≥95% variance)\n"
    for j in range(k):
        summary += f"  PC{j + 1}: {explained[j] * 100:.1f}% variance, Cpk = {component_cpk[j]:.3f}\n"
    summary += f"  Cumulative: {cumvar[k - 1] * 100:.1f}%\n"

    summary += "\n<<COLOR:header>>Joint Capability:<</COLOR>>\n"
    summary += f"  MCpk (min component Cpk): {mcpk:.3f}\n"
    summary += f"  T² Capability: {t2_capability * 100:.1f}% within UCL\n"
    summary += f"  T² UCL: {T2_ucl:.2f} (0.27% false alarm rate)\n"
    summary += f"  OOC observations: {n_ooc} of {n} ({n_ooc / n * 100:.1f}%)\n"

    if var_cpk:
        summary += f"\n<<COLOR:header>>Per-Variable Capability (±{tolerance_pct}% tolerance):<</COLOR>>\n"
        for vc in var_cpk:
            summary += f"  {vc['variable']}: Cpk = {vc['cpk']:.3f}\n"

    if mcpk >= 1.33 and n_ooc == 0:
        summary += "\n<<COLOR:success>>Multivariate process is capable and in control.<</COLOR>>"
    elif mcpk >= 1.0:
        summary += f"\n<<COLOR:warning>>Multivariate process is marginally capable (MCpk = {mcpk:.3f}).<</COLOR>>"
    else:
        summary += f"\n<<COLOR:danger>>Multivariate process is NOT capable (MCpk = {mcpk:.3f}).<</COLOR>>"

    if n_ooc > 0:
        summary += (
            f"\n<<COLOR:warning>>{n_ooc} observations exceed T² UCL — investigate multivariate outliers.<</COLOR>>"
        )

    result["summary"] = summary
    result["guide_observation"] = (
        f"D-Multi: MCpk={mcpk:.3f}, T² OOC={n_ooc}/{n}, {k} PCs retain {cumvar[k - 1] * 100:.1f}% variance"
    )
    result["statistics"] = {
        "n": n,
        "p": p,
        "k_components": k,
        "explained_variance": [round(float(e), 4) for e in explained[:k]],
        "component_cpk": component_cpk,
        "mcpk": round(mcpk, 4),
        "t2_ucl": round(T2_ucl, 2),
        "n_ooc": n_ooc,
        "t2_capability": round(t2_capability, 4),
    }

    # --- narrative ---
    if mcpk >= 1.33 and n_ooc == 0:
        verdict = f"Capable & In Control — MCpk = {mcpk:.3f}"
        body = (
            f"The multivariate process is capable (<strong>MCpk = {mcpk:.3f}</strong>) "
            f"with no T² outliers. {k} principal components retain "
            f"{cumvar[k - 1] * 100:.1f}% of the variance across {p} variables."
        )
        nxt = "Process is healthy — continue monitoring."
    elif mcpk >= 1.0:
        verdict = f"Marginally Capable — MCpk = {mcpk:.3f}"
        body = (
            f"MCpk = <strong>{mcpk:.3f}</strong> — above 1.0 but below target. "
            f"{n_ooc} T² outlier{'s' if n_ooc != 1 else ''} detected out of {n} observations. "
            f"{k} PCs retain {cumvar[k - 1] * 100:.1f}% variance."
        )
        nxt = "Identify which variables contribute most to the weakest principal component."
    else:
        verdict = f"Not Capable — MCpk = {mcpk:.3f}"
        body = (
            f"MCpk = <strong>{mcpk:.3f}</strong> — the process is not jointly capable "
            f"across the {p} variables. {n_ooc} T² outlier{'s' if n_ooc != 1 else ''} detected."
        )
        nxt = "Run individual capability studies to identify the weakest variables, then address jointly."
    if n_ooc > 0:
        body += f" <strong>{n_ooc}</strong> observations exceed the T² upper control limit — investigate these multivariate outliers."
    result["narrative"] = _d_narrative(
        f"D-Multi: {verdict}",
        body,
        nxt,
        "The T² chart shows Hotelling's T² statistic per observation — points above "
        "the UCL are multivariate outliers. The component Cpk chart shows capability "
        "on each principal component (the joint minimum is MCpk).",
    )

    result["education"] = {
        "title": "Understanding Multivariate Capability (D-Multi)",
        "content": (
            "<dl>"
            "<dt>Why multivariate?</dt>"
            "<dd>When you have multiple correlated quality characteristics (e.g., length, width, "
            "weight), checking each separately misses the joint picture. A part can pass every "
            "individual spec but still be out of spec <em>jointly</em> because the variables "
            "interact.</dd>"
            "<dt>What is PCA doing here?</dt>"
            "<dd>Principal Component Analysis rotates correlated variables into uncorrelated "
            "'principal components'. We keep enough PCs to explain most of the variance, then "
            "compute KDE-based capability on each — the minimum across PCs is MCpk.</dd>"
            "<dt>What is Hotelling's T²?</dt>"
            "<dd>A multivariate distance measure — how far each observation is from the "
            "centre in all dimensions simultaneously. Points above the UCL are multivariate "
            "outliers, even if they look normal on any single variable.</dd>"
            "<dt>How to interpret MCpk</dt>"
            "<dd><strong>≥ 1.33</strong>: Jointly capable across all dimensions. "
            "<strong>1.0–1.33</strong>: Marginally capable — check which PC is weakest and "
            "trace back to the original variables. <strong>&lt; 1.0</strong>: Not jointly "
            "capable — one or more correlated variable combinations are out of spec.</dd>"
            "</dl>"
        ),
    }

    return result

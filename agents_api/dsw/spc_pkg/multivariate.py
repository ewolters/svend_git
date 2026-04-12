"""SPC multivariate charts — MEWMA, Generalized Variance."""

import numpy as np
from scipy.stats import chi2

from ..common import _narrative
from .helpers import _spc_add_ooc_markers


def run_mewma(df, config):
    """MEWMA -- Multivariate Exponentially Weighted Moving Average."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    vars_list = config.get("variables", [])
    lambda_param = float(config.get("lambda", 0.1))

    if not vars_list or len(vars_list) < 2:
        # Auto-select first 2-4 numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        vars_list = num_cols[: min(4, len(num_cols))]

    X = df[vars_list].dropna().values
    n, p = X.shape

    if n < 10 or p < 2:
        result["summary"] = "MEWMA requires at least 2 variables and 10 observations."
        return result

    # Mean vector and covariance
    mu = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False, ddof=1)

    # Regularize if near-singular
    if np.linalg.cond(Sigma) > 1e10:
        Sigma += np.eye(p) * 1e-6

    # MEWMA vectors
    Z = np.zeros((n, p))
    Z[0] = lambda_param * X[0] + (1 - lambda_param) * mu
    for i in range(1, n):
        Z[i] = lambda_param * X[i] + (1 - lambda_param) * Z[i - 1]

    # T2 statistic for each MEWMA vector
    t2_values = []
    for i in range(n):
        factor = (lambda_param / (2 - lambda_param)) * (1 - (1 - lambda_param) ** (2 * (i + 1)))
        Sigma_Z = factor * Sigma
        try:
            Sigma_Z_inv = np.linalg.inv(Sigma_Z)
        except np.linalg.LinAlgError:
            Sigma_Z_inv = np.linalg.pinv(Sigma_Z)
        diff = Z[i] - mu
        t2 = float(diff @ Sigma_Z_inv @ diff)
        t2_values.append(max(0, t2))

    # UCL: chi-squared approximation (asymptotic)
    ucl = chi2.ppf(1 - 0.0027, p)  # 3-sigma equivalent ARL

    # OOC
    ooc = [i for i, t2 in enumerate(t2_values) if t2 > ucl]

    mewma_chart_data = [
        {
            "type": "scatter",
            "y": t2_values,
            "mode": "lines+markers",
            "name": "MEWMA T\u00b2",
            "marker": {"size": 5, "color": "#4a9f6e"},
            "line": {"color": "#4a9f6e"},
        },
        {
            "type": "scatter",
            "y": [ucl] * n,
            "mode": "lines",
            "name": f"UCL ({ucl:.2f})",
            "line": {"color": "#d63031", "dash": "dash"},
        },
    ]
    _spc_add_ooc_markers(mewma_chart_data, t2_values, ooc)

    result["plots"].append(
        {
            "title": "MEWMA Chart",
            "data": mewma_chart_data,
            "layout": {
                "height": 290,
                "showlegend": True,
                "yaxis": {"title": "T\u00b2 Statistic"},
                "xaxis": {
                    "title": "Observation",
                    "rangeslider": {"visible": True, "thickness": 0.12},
                },
            },
            "interactive": {"type": "spc_inspect"},
        }
    )

    # Variable contribution at OOC points
    if ooc:
        first_ooc = ooc[0]
        diff = Z[first_ooc] - mu
        contributions = diff**2
        total_contrib = contributions.sum()
        if total_contrib > 0:
            pct_contrib = (contributions / total_contrib * 100).tolist()
        else:
            pct_contrib = [0] * p

        result["plots"].append(
            {
                "title": f"Variable Contribution at First OOC (obs {first_ooc})",
                "data": [
                    {
                        "type": "bar",
                        "x": vars_list,
                        "y": pct_contrib,
                        "marker": {"color": "#4a9f6e"},
                    }
                ],
                "layout": {
                    "height": 250,
                    "yaxis": {"title": "% Contribution"},
                    "xaxis": {"title": "Variable"},
                },
            }
        )

    result["summary"] = (
        f"MEWMA Chart Analysis\n\nVariables: {', '.join(vars_list)} (p={p})\n\u03bb (smoothing): {lambda_param}\nUCL (\u03c7\u00b2): {ucl:.4f}\n\nObservations: {n}\nOut-of-control points: {len(ooc)}\n\nNote: Smaller \u03bb increases sensitivity to small sustained shifts but also increases false alarm rate. Typical range: 0.05\u20130.25."
    )

    return result


def run_generalized_variance(df, config):
    """Generalized Variance (|S|) Chart -- monitors the determinant of the covariance matrix."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    variables_gv = config.get("variables") or config.get("columns", [])
    subgroup_col = config.get("subgroup") or config.get("group")
    subgroup_size_gv = int(config.get("subgroup_size", 5))

    if not variables_gv or len(variables_gv) < 2:
        result["summary"] = "Need at least 2 variables for generalized variance chart."
        return result

    data_gv = df[variables_gv + ([subgroup_col] if subgroup_col else [])].dropna()
    p_gv = len(variables_gv)

    # Create subgroups
    if subgroup_col:
        subgroups = [grp[variables_gv].values for _, grp in data_gv.groupby(subgroup_col) if len(grp) >= 2]
        subgroup_labels = [str(name) for name, grp in data_gv.groupby(subgroup_col) if len(grp) >= 2]
    else:
        n_obs = len(data_gv)
        subgroups = [
            data_gv[variables_gv].values[i : i + subgroup_size_gv]
            for i in range(0, n_obs - subgroup_size_gv + 1, subgroup_size_gv)
        ]
        subgroup_labels = [str(i + 1) for i in range(len(subgroups))]

    if len(subgroups) < 3:
        result["summary"] = "Need at least 3 subgroups for generalized variance chart."
        return result

    # Compute |S_i| for each subgroup
    det_values = []
    ns_gv = []
    for sg in subgroups:
        n_sg = len(sg)
        ns_gv.append(n_sg)
        if n_sg < p_gv:
            det_values.append(0.0)
        else:
            cov_sg = np.cov(sg.T, ddof=1)
            det_values.append(float(np.linalg.det(cov_sg)))
    det_values = np.array(det_values)

    # Pooled covariance determinant
    mean_det = float(np.mean(det_values))

    # Control limits for |S|
    n_avg = int(np.mean(ns_gv))

    # Compute b1 and b2 coefficients
    b1 = 1.0
    for i in range(1, p_gv + 1):
        b1 *= (n_avg - i) / (n_avg - 1)

    # Variance coefficient (simplified)
    b2 = b1**2
    for i in range(1, p_gv + 1):
        b2 *= ((n_avg - i + 2) / (n_avg - i)) if (n_avg - i) > 0 else 1
    b2 -= b1**2

    # |Σ| estimate
    sigma_det = mean_det / b1 if b1 > 0 else mean_det
    se_det = sigma_det * np.sqrt(b2) if b2 > 0 else mean_det * 0.1

    cl_gv = mean_det
    ucl_gv = cl_gv + 3 * se_det
    lcl_gv = max(0, cl_gv - 3 * se_det)

    # OOC detection
    ooc_gv = []
    for i, val in enumerate(det_values):
        if val > ucl_gv or val < lcl_gv:
            ooc_gv.append(i)

    # Chart
    [subgroup_labels[i] for i in range(len(det_values)) if i not in ooc_gv]
    [det_values[i] for i in range(len(det_values)) if i not in ooc_gv]
    ooc_x = [subgroup_labels[i] for i in ooc_gv]
    ooc_y = [det_values[i] for i in ooc_gv]

    chart_traces = [
        {
            "x": subgroup_labels,
            "y": det_values.tolist(),
            "mode": "lines+markers",
            "name": "|S|",
            "marker": {"color": "#4a9f6e", "size": 6},
            "line": {"color": "#4a9f6e", "width": 1},
        },
    ]
    if ooc_x:
        chart_traces.append(
            {
                "x": ooc_x,
                "y": ooc_y,
                "mode": "markers",
                "name": "OOC",
                "marker": {"color": "#d94a4a", "size": 10, "symbol": "x"},
            }
        )
    # Control limit lines
    chart_traces.extend(
        [
            {
                "x": [subgroup_labels[0], subgroup_labels[-1]],
                "y": [ucl_gv, ucl_gv],
                "mode": "lines",
                "name": f"UCL ({ucl_gv:.4f})",
                "line": {"color": "#d94a4a", "dash": "dash"},
            },
            {
                "x": [subgroup_labels[0], subgroup_labels[-1]],
                "y": [cl_gv, cl_gv],
                "mode": "lines",
                "name": f"CL ({cl_gv:.4f})",
                "line": {"color": "#4a90d9", "dash": "dot"},
            },
            {
                "x": [subgroup_labels[0], subgroup_labels[-1]],
                "y": [lcl_gv, lcl_gv],
                "mode": "lines",
                "name": f"LCL ({lcl_gv:.4f})",
                "line": {"color": "#d94a4a", "dash": "dash"},
            },
        ]
    )

    result["plots"].append(
        {
            "title": "Generalized Variance |S| Chart",
            "data": chart_traces,
            "layout": {
                "height": 440,
                "xaxis": {
                    "title": "Subgroup",
                    "rangeslider": {"visible": True, "thickness": 0.12},
                },
                "yaxis": {"title": "|S| (Determinant)"},
            },
        }
    )

    summary_gv = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
    summary_gv += "<<COLOR:title>>GENERALIZED VARIANCE CHART<</COLOR>>\n"
    summary_gv += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
    summary_gv += f"<<COLOR:highlight>>Variables:<</COLOR>> {', '.join(variables_gv)}\n"
    summary_gv += f"<<COLOR:highlight>>Subgroups:<</COLOR>> {len(subgroups)}  (avg size = {n_avg})\n\n"
    summary_gv += "<<COLOR:text>>Control Limits:<</COLOR>>\n"
    summary_gv += f"  UCL = {ucl_gv:.6f}\n"
    summary_gv += f"  CL  = {cl_gv:.6f}\n"
    summary_gv += f"  LCL = {lcl_gv:.6f}\n\n"
    if ooc_gv:
        summary_gv += f"<<COLOR:warning>>Out-of-control points: {len(ooc_gv)}<</COLOR>>\n"
        for idx_ooc in ooc_gv:
            summary_gv += f"  Subgroup {subgroup_labels[idx_ooc]}: |S| = {det_values[idx_ooc]:.6f}\n"
    else:
        summary_gv += "<<COLOR:good>>Process variability in control \u2014 no OOC points<</COLOR>>\n"

    result["summary"] = summary_gv
    result["guide_observation"] = (
        f"Generalized variance chart: {len(ooc_gv)} OOC points out of {len(subgroups)} subgroups."
    )
    result["statistics"] = {
        "cl": cl_gv,
        "ucl": ucl_gv,
        "lcl": lcl_gv,
        "det_values": det_values.tolist(),
        "n_subgroups": len(subgroups),
        "ooc_count": len(ooc_gv),
        "p": p_gv,
    }

    # Narrative
    _gv_n_ooc = len(ooc_gv)
    if _gv_n_ooc == 0:
        _gv_verdict = "Generalized Variance \u2014 multivariate spread in control"
        _gv_body = f"All {len(subgroups)} subgroups have covariance determinants within limits. Joint variability of {p_gv} variables is stable."
    else:
        _gv_verdict = f"Generalized Variance \u2014 {_gv_n_ooc} OOC subgroup{'s' if _gv_n_ooc > 1 else ''}"
        _gv_body = f"{_gv_n_ooc} of {len(subgroups)} subgroups show unusual joint variability across {p_gv} variables. The covariance structure has shifted."
    result["narrative"] = _narrative(
        _gv_verdict,
        _gv_body,
        next_steps=(
            "Pair with a Hotelling T\u00b2 chart to distinguish mean shifts from variability shifts."
            if _gv_n_ooc > 0
            else "Continue monitoring. Process variability is stable across all measured dimensions."
        ),
        chart_guidance="Each point is the determinant |S| of the subgroup covariance matrix. Higher values mean more joint spread.",
    )

    return result

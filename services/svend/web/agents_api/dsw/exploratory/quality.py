"""DSW Exploratory — data quality analyses (profiling, graphical summary, missing, outlier, duplicate)."""

import logging

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..common import _narrative

logger = logging.getLogger(__name__)


def run_data_profile(df, config):
    """Full data profiling — column types, missing data, correlations, distributions."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    try:
        n_rows, n_cols = df.shape
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

        # Summary stats table
        desc = df.describe(include="all").T
        desc["missing"] = df.isnull().sum()
        desc["missing%"] = (df.isnull().sum() / n_rows * 100).round(1)
        desc["unique"] = df.nunique()
        desc["dtype"] = df.dtypes

        tbl_rows = []
        for col in df.columns:
            r = desc.loc[col] if col in desc.index else {}
            tbl_rows.append(
                f"<tr><td>{col}</td><td>{df[col].dtype}</td>"
                f"<td>{int(r.get('missing', 0))}</td><td>{r.get('missing%', 0):.1f}%</td>"
                f"<td>{int(r.get('unique', 0))}</td>"
                f"<td>{r.get('mean', ''):.4g}</td>"
                if col in numeric_cols
                else f"<tr><td>{col}</td><td>{df[col].dtype}</td>"
                f"<td>{int(r.get('missing', 0))}</td><td>{r.get('missing%', 0):.1f}%</td>"
                f"<td>{int(r.get('unique', 0))}</td><td>-</td>"
                f"<td>{r.get('top', '-')}</td></tr>"
            )

        table_html = "<table class='result-table'><tr><th>Column</th><th>Type</th><th>Missing</th><th>Missing%</th><th>Unique</th><th>Mean/Top</th></tr>"
        for col in df.columns:
            miss = int(df[col].isnull().sum())
            miss_pct = miss / n_rows * 100 if n_rows > 0 else 0
            uniq = df[col].nunique()
            if col in numeric_cols:
                val = f"{df[col].mean():.4g}"
            else:
                top = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "-"
                val = str(top)[:20]
            table_html += f"<tr><td>{col}</td><td>{df[col].dtype}</td><td>{miss}</td><td>{miss_pct:.1f}%</td><td>{uniq}</td><td>{val}</td></tr>"
        table_html += "</table>"

        # Top correlations
        corr_text = ""
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    pairs.append(
                        (numeric_cols[i], numeric_cols[j], abs(corr_matrix.iloc[i, j]), corr_matrix.iloc[i, j])
                    )
            pairs.sort(key=lambda x: x[2], reverse=True)
            top_pairs = pairs[:10]
            corr_lines = [f"  {a} <-> {b}: {v:+.3f}" for a, b, _, v in top_pairs]
            corr_text = "\n".join(corr_lines)

        total_missing = int(df.isnull().sum().sum())
        total_cells = n_rows * n_cols
        complete_rows = int((~df.isnull().any(axis=1)).sum())

        summary = f"""DATA PROFILE
{"=" * 50}
Shape: {n_rows} rows x {n_cols} columns
Memory: {mem_mb:.2f} MB

Column Types:
  Numeric:     {len(numeric_cols)}
  Categorical: {len(cat_cols)}
  Datetime:    {len(dt_cols)}

Missing Data:
  Total missing cells: {total_missing} / {total_cells} ({total_missing / total_cells * 100:.1f}%)
  Complete rows: {complete_rows} / {n_rows} ({complete_rows / n_rows * 100:.1f}%)

Top Correlations:
{corr_text if corr_text else "  (need 2+ numeric columns)"}"""

        result["summary"] = summary
        result["tables"] = [{"title": "Column Summary", "html": table_html}]

        # Plot 1: Correlation heatmap
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            result["plots"].append(
                {
                    "data": [
                        {
                            "type": "heatmap",
                            "z": corr_matrix.values.tolist(),
                            "x": numeric_cols,
                            "y": numeric_cols,
                            "colorscale": "RdBu_r",
                            "zmin": -1,
                            "zmax": 1,
                            "text": [[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                            "texttemplate": "%{text}",
                            "hovertemplate": "%{x} vs %{y}: %{z:.3f}<extra></extra>",
                        }
                    ],
                    "layout": {"title": "Correlation Matrix", "height": 400},
                }
            )

        # Plot 2: Missing percentage bar chart
        miss_cols = df.columns[df.isnull().any()].tolist()
        if miss_cols:
            miss_pcts = [(df[c].isnull().sum() / n_rows * 100) for c in miss_cols]
            result["plots"].append(
                {
                    "data": [
                        {
                            "type": "bar",
                            "x": [round(p, 1) for p in miss_pcts],
                            "y": miss_cols,
                            "orientation": "h",
                            "marker": {"color": "rgba(232,71,71,0.6)"},
                        }
                    ],
                    "layout": {
                        "title": "Missing Data by Column (%)",
                        "height": max(250, len(miss_cols) * 25),
                        "xaxis": {"title": "% Missing"},
                        "margin": {"l": 120},
                    },
                }
            )

        # Plot 3: Distribution grid (top 6 numeric)
        for col in numeric_cols[:6]:
            vals = df[col].dropna().tolist()
            if len(vals) > 0:
                result["plots"].append(
                    {
                        "data": [
                            {
                                "type": "histogram",
                                "x": vals,
                                "marker": {"color": "rgba(74,144,217,0.6)"},
                                "nbinsx": 30,
                            }
                        ],
                        "layout": {
                            "title": f"Distribution: {col}",
                            "height": 250,
                            "xaxis": {"title": col},
                            "yaxis": {"title": "Count"},
                        },
                    }
                )

        result["guide_observation"] = (
            f"Data profile: {n_rows} rows, {n_cols} cols, {total_missing} missing cells, {len(numeric_cols)} numeric columns."
        )

        # Narrative
        _dp_miss_note = (
            f" {total_missing} cells are missing ({total_missing / total_cells * 100:.1f}%)."
            if total_missing > 0
            else " No missing data detected."
        )
        _dp_top_corr = ""
        if len(numeric_cols) >= 2:
            _corr = df[numeric_cols].corr()
            _tri = _corr.where(np.triu(np.ones(_corr.shape, dtype=bool), k=1))
            _max_pair = _tri.stack().abs().idxmax() if _tri.stack().abs().max() > 0.5 else None
            if _max_pair:
                _dp_top_corr = f" Strongest correlation: <strong>{_max_pair[0]}</strong> ↔ <strong>{_max_pair[1]}</strong> (r = {_corr.loc[_max_pair[0], _max_pair[1]]:.3f})."
        result["narrative"] = _narrative(
            f"Dataset: {n_rows:,} rows × {n_cols} columns ({len(numeric_cols)} numeric, {len(df.select_dtypes(include=['object', 'category']).columns)} categorical)",
            f"The dataset contains {n_rows:,} observations across {n_cols} variables.{_dp_miss_note}{_dp_top_corr}",
            next_steps="Check distributions for skewness, then run normality tests before parametric analysis.",
            chart_guidance="The correlation heatmap highlights linear relationships. Red/blue extremes indicate strong positive/negative associations.",
        )

    except Exception as e:
        result["summary"] = f"Data profile error: {str(e)}"

    return result


def run_auto_profile(df, config):
    """Auto Profile (lightweight, runs on import) — quick overview of dataset."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    try:
        n_rows, n_cols = df.shape
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
        total_missing = int(df.isnull().sum().sum())
        total_cells = n_rows * n_cols
        miss_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0

        # Per-column one-liner stats
        col_lines = []
        for col in df.columns:
            miss = int(df[col].isnull().sum())
            miss_p = miss / n_rows * 100 if n_rows > 0 else 0
            if col in numeric_cols:
                vals = df[col].dropna()
                if len(vals) > 0:
                    col_lines.append(
                        f"  <<COLOR:accent>>{col}<</COLOR>>  N={len(vals)}  Mean={vals.mean():.4g}  StDev={vals.std():.4g}  Min={vals.min():.4g}  Max={vals.max():.4g}  Missing={miss_p:.1f}%"
                    )
                else:
                    col_lines.append(f"  <<COLOR:warning>>{col}<</COLOR>>  (all missing)")
            else:
                uniq = df[col].nunique()
                top = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "-"
                col_lines.append(
                    f'  <<COLOR:accent>>{col}<</COLOR>>  Unique={uniq}  Top="{str(top)[:20]}"  Missing={miss_p:.1f}%'
                )

        summary = f"""<<COLOR:title>>DATA OVERVIEW<</COLOR>>
{"=" * 50}
<<COLOR:highlight>>{n_rows}<</COLOR>> rows × <<COLOR:highlight>>{n_cols}<</COLOR>> columns
Numeric: {len(numeric_cols)}  |  Categorical: {len(cat_cols)}  |  Datetime: {len(dt_cols)}
Missing: {total_missing} / {total_cells} ({miss_pct:.1f}%)

<<COLOR:title>>Column Summary<</COLOR>>
{chr(10).join(col_lines)}"""

        result["summary"] = summary

        # Correlation heatmap (if 2+ numeric)
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            result["plots"].append(
                {
                    "data": [
                        {
                            "type": "heatmap",
                            "z": corr_matrix.values.tolist(),
                            "x": numeric_cols,
                            "y": numeric_cols,
                            "colorscale": "RdBu_r",
                            "zmin": -1,
                            "zmax": 1,
                            "text": [[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                            "texttemplate": "%{text}",
                            "hovertemplate": "%{x} vs %{y}: %{z:.3f}<extra></extra>",
                        }
                    ],
                    "layout": {"title": "Correlation Matrix", "height": 350},
                }
            )

        # Distribution histograms (up to 12 numeric)
        for col in numeric_cols[:12]:
            vals = df[col].dropna().tolist()
            if len(vals) > 0:
                result["plots"].append(
                    {
                        "data": [
                            {
                                "type": "histogram",
                                "x": vals,
                                "marker": {"color": "rgba(74,144,217,0.6)"},
                                "nbinsx": 30,
                            }
                        ],
                        "layout": {
                            "title": f"{col}",
                            "height": 220,
                            "xaxis": {"title": col},
                            "yaxis": {"title": "Count"},
                            "margin": {"t": 30, "b": 40, "l": 50, "r": 20},
                        },
                    }
                )

        # Missing bar if any
        miss_cols = df.columns[df.isnull().any()].tolist()
        if miss_cols:
            miss_pcts = [(df[c].isnull().sum() / n_rows * 100) for c in miss_cols]
            result["plots"].append(
                {
                    "data": [
                        {
                            "type": "bar",
                            "x": [round(p, 1) for p in miss_pcts],
                            "y": miss_cols,
                            "orientation": "h",
                            "marker": {"color": "rgba(232,71,71,0.6)"},
                        }
                    ],
                    "layout": {
                        "title": "Missing Data (%)",
                        "height": max(200, len(miss_cols) * 22),
                        "xaxis": {"title": "% Missing"},
                        "margin": {"l": 120},
                    },
                }
            )

        result["guide_observation"] = (
            f"Auto-profile: {n_rows} rows, {n_cols} cols, {total_missing} missing, {len(numeric_cols)} numeric."
        )

        # Narrative
        _ap_miss = (
            f" {total_missing} missing cells ({miss_pct:.1f}%) — check the missing data bar chart."
            if total_missing > 0
            else " Dataset is complete (no missing values)."
        )
        result["narrative"] = _narrative(
            f"Data Overview: {n_rows:,} rows × {n_cols} columns",
            f"{len(numeric_cols)} numeric, {len(cat_cols)} categorical, {len(dt_cols)} datetime columns.{_ap_miss}",
            next_steps="Review distributions for normality. Address missing data before hypothesis testing.",
            chart_guidance="Histograms show the shape of each numeric variable. Look for skewness, bimodality, or outliers.",
        )

    except Exception as e:
        result["summary"] = f"Auto-profile error: {str(e)}"

    return result


def run_graphical_summary(df, config):
    """Graphical Summary (Minitab-style) — histogram, boxplot, CIs, normality test."""
    result = {"plots": [], "summary": "", "guide_observation": "", "tables": []}

    try:
        conf_level = config.get("confidence", 0.95)
        selected = config.get("vars", [])
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not selected:
            selected = numeric_cols
        selected = [c for c in selected if c in numeric_cols]

        if not selected:
            result["summary"] = "No numeric columns selected or available for Graphical Summary."
        else:
            all_summaries = []
            for col in selected:
                vals = df[col].dropna().values
                n = len(vals)
                n_star = int(df[col].isnull().sum())

                if n < 3:
                    all_summaries.append(f"<<COLOR:title>>{col}<</COLOR>>: insufficient data (N={n})")
                    continue

                # Descriptive stats
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1))
                var_val = std_val**2
                se = std_val / np.sqrt(n)
                skew_val = float(sp_stats.skew(vals, bias=False))
                kurt_val = float(sp_stats.kurtosis(vals, bias=False))
                q1 = float(np.percentile(vals, 25))
                median_val = float(np.median(vals))
                q3 = float(np.percentile(vals, 75))
                min_val = float(np.min(vals))
                max_val = float(np.max(vals))

                # Anderson-Darling test
                ad_result = sp_stats.anderson(vals, dist="norm")
                ad_stat = ad_result.statistic
                # Use 5% significance level (index 2 in critical_values)
                ad_crit = (
                    ad_result.critical_values[2]
                    if len(ad_result.critical_values) > 2
                    else ad_result.critical_values[-1]
                )
                ad_sig = (
                    ad_result.significance_level[2]
                    if len(ad_result.significance_level) > 2
                    else ad_result.significance_level[-1]
                )
                ad_pass = ad_stat < ad_crit
                ad_verdict = (
                    "<<COLOR:success>>Yes (fail to reject H₀)<</COLOR>>"
                    if ad_pass
                    else "<<COLOR:danger>>No (reject H₀)<</COLOR>>"
                )

                # CI for mean (t-interval)
                ci_mean = sp_stats.t.interval(conf_level, df=n - 1, loc=mean_val, scale=se)

                # CI for median (nonparametric sign-test inversion)
                alpha = 1 - conf_level
                from scipy.stats import binom

                j = 0
                for k_idx in range(n):
                    if binom.cdf(k_idx, n, 0.5) >= alpha / 2:
                        j = k_idx
                        break
                sorted_vals = np.sort(vals)
                ci_median_lo = float(sorted_vals[j]) if j < n else float(sorted_vals[0])
                ci_median_hi = float(sorted_vals[n - 1 - j]) if (n - 1 - j) >= 0 else float(sorted_vals[-1])

                # CI for StDev (chi-square)
                chi2_lo = sp_stats.chi2.ppf((1 + conf_level) / 2, n - 1)
                chi2_hi = sp_stats.chi2.ppf((1 - conf_level) / 2, n - 1)
                ci_std_lo = np.sqrt((n - 1) * var_val / chi2_lo)
                ci_std_hi = np.sqrt((n - 1) * var_val / chi2_hi)

                pct_str = f"{conf_level * 100:.0f}%"

                # Summary text
                summ = f"""<<COLOR:title>>{"═" * 50}
{col}
{"═" * 50}<</COLOR>>

<<COLOR:accent>>Anderson-Darling Normality Test<</COLOR>>
  A² = {ad_stat:.4f}    Critical ({ad_sig:.0f}%) = {ad_crit:.4f}
  Normal? {ad_verdict}

<<COLOR:accent>>Descriptive Statistics<</COLOR>>
  N = {n}    N* = {n_star}
  Mean     = {mean_val:.6g}       StDev    = {std_val:.6g}
  Variance = {var_val:.6g}       Skewness = {skew_val:.4f}
  Kurtosis = {kurt_val:.4f}
  Minimum  = {min_val:.6g}       Q1       = {q1:.6g}
  Median   = {median_val:.6g}       Q3       = {q3:.6g}
  Maximum  = {max_val:.6g}

<<COLOR:accent>>Confidence Intervals ({pct_str})<</COLOR>>
  Mean:   ({ci_mean[0]:.6g}, {ci_mean[1]:.6g})
  Median: ({ci_median_lo:.6g}, {ci_median_hi:.6g})
  StDev:  ({ci_std_lo:.6g}, {ci_std_hi:.6g})"""
                all_summaries.append(summ)

                # ── Plotly figure: 3-row subplot ──
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots

                fig = make_subplots(
                    rows=3,
                    cols=1,
                    row_heights=[0.55, 0.20, 0.25],
                    shared_xaxes=True,
                    vertical_spacing=0.06,
                    subplot_titles=[
                        f"{col} — Histogram + Normal Fit",
                        "Boxplot",
                        f"{pct_str} Confidence Intervals",
                    ],
                )

                # Row 1: Histogram + normal curve
                nbins = min(max(int(np.sqrt(n)), 10), 50)
                counts, bin_edges = np.histogram(vals, bins=nbins)
                bin_width = bin_edges[1] - bin_edges[0]

                fig.add_trace(
                    go.Histogram(
                        x=vals.tolist(),
                        nbinsx=nbins,
                        marker=dict(color="rgba(74,144,217,0.6)", line=dict(color="rgba(74,144,217,1)", width=1)),
                        name="Data",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

                # Normal PDF overlay scaled to histogram
                x_fit = np.linspace(min_val - 0.5 * std_val, max_val + 0.5 * std_val, 200)
                y_fit = sp_stats.norm.pdf(x_fit, mean_val, std_val) * n * bin_width
                fig.add_trace(
                    go.Scatter(
                        x=x_fit.tolist(),
                        y=y_fit.tolist(),
                        mode="lines",
                        line=dict(color="rgba(232,71,71,0.9)", width=2),
                        name="Normal Fit",
                        showlegend=False,
                    ),
                    row=1,
                    col=1,
                )

                # Row 2: Boxplot
                fig.add_trace(
                    go.Box(
                        x=vals.tolist(),
                        orientation="h",
                        marker=dict(color="rgba(74,144,217,0.7)"),
                        line=dict(color="rgba(74,144,217,1)"),
                        fillcolor="rgba(74,144,217,0.3)",
                        name="Box",
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

                # Row 3: CI bars (mean and median)
                fig.add_trace(
                    go.Scatter(
                        x=[ci_mean[0], mean_val, ci_mean[1]],
                        y=["Mean", "Mean", "Mean"],
                        mode="lines+markers",
                        marker=dict(
                            size=[8, 12, 8],
                            color=["rgba(232,71,71,0.8)", "rgba(232,71,71,1)", "rgba(232,71,71,0.8)"],
                            symbol=["line-ns", "diamond", "line-ns"],
                        ),
                        line=dict(color="rgba(232,71,71,0.8)", width=2),
                        name="Mean CI",
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=[ci_median_lo, median_val, ci_median_hi],
                        y=["Median", "Median", "Median"],
                        mode="lines+markers",
                        marker=dict(
                            size=[8, 12, 8],
                            color=["rgba(74,144,217,0.8)", "rgba(74,144,217,1)", "rgba(74,144,217,0.8)"],
                            symbol=["line-ns", "diamond", "line-ns"],
                        ),
                        line=dict(color="rgba(74,144,217,0.8)", width=2),
                        name="Median CI",
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )

                fig.update_layout(height=520, margin=dict(t=40, b=30, l=60, r=20))
                fig.update_yaxes(showticklabels=False, row=2, col=1)

                # Convert to JSON-serializable dict
                fig_dict = fig.to_dict()
                result["plots"].append({"data": [t for t in fig_dict["data"]], "layout": fig_dict["layout"]})

                # HTML table for this variable
                tbl = f"""<table class='result-table'>
<tr><th colspan='4' style='text-align:center;'>{col} — Descriptive Statistics</th></tr>
<tr><td>N</td><td>{n}</td><td>N* (missing)</td><td>{n_star}</td></tr>
<tr><td>Mean</td><td>{mean_val:.6g}</td><td>StDev</td><td>{std_val:.6g}</td></tr>
<tr><td>Variance</td><td>{var_val:.6g}</td><td>Skewness</td><td>{skew_val:.4f}</td></tr>
<tr><td>Kurtosis</td><td>{kurt_val:.4f}</td><td>A² Statistic</td><td>{ad_stat:.4f}</td></tr>
<tr><td>Minimum</td><td>{min_val:.6g}</td><td>Q1</td><td>{q1:.6g}</td></tr>
<tr><td>Median</td><td>{median_val:.6g}</td><td>Q3</td><td>{q3:.6g}</td></tr>
<tr><td>Maximum</td><td>{max_val:.6g}</td><td>IQR</td><td>{q3 - q1:.6g}</td></tr>
<tr><th colspan='4'>Confidence Intervals ({pct_str})</th></tr>
<tr><td>Mean</td><td>({ci_mean[0]:.6g}, {ci_mean[1]:.6g})</td><td>Median</td><td>({ci_median_lo:.6g}, {ci_median_hi:.6g})</td></tr>
<tr><td>StDev</td><td>({ci_std_lo:.6g}, {ci_std_hi:.6g})</td><td></td><td></td></tr>
</table>"""
                result["tables"].append({"title": f"{col} Statistics", "html": tbl})

            result["summary"] = "\n\n".join(all_summaries)
            result["guide_observation"] = (
                f"Graphical summary for {len(selected)} variable(s) at {conf_level * 100:.0f}% confidence."
            )

            # Narrative — use first variable's stats as lead
            if selected and len(all_summaries) > 0:
                result["narrative"] = _narrative(
                    f"Graphical Summary — {len(selected)} variable{'s' if len(selected) > 1 else ''} at {conf_level * 100:.0f}% confidence",
                    "Each panel shows the histogram with normal fit overlay, boxplot, and confidence intervals for mean and median. Review the Anderson-Darling test result to determine if the normal distribution is appropriate.",
                    next_steps="If data departs from normality, consider non-parametric tests or a Box-Cox / Johnson transform.",
                    chart_guidance="Points in the boxplot beyond the whiskers are potential outliers. If the histogram shape is skewed or bimodal, the normal fit (red curve) will visually misfit.",
                )

    except Exception as e:
        import traceback

        result["summary"] = f"Graphical Summary error: {str(e)}\n{traceback.format_exc()}"

    return result


def run_missing_data_analysis(df, config):
    """Missing Data Analysis — patterns, MCAR test, correlations."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    try:
        n_rows, n_cols = df.shape
        miss_count = df.isnull().sum()
        miss_pct = (miss_count / n_rows * 100).round(2)
        cols_with_missing = miss_count[miss_count > 0].sort_values(ascending=False)

        # Row completeness
        row_completeness = (~df.isnull()).sum(axis=1) / n_cols * 100
        complete_rows = int((row_completeness == 100).sum())

        # Missing patterns
        miss_indicator = df.isnull().astype(int)
        pattern_strs = miss_indicator.apply(lambda r: "".join(str(v) for v in r), axis=1)
        pattern_counts = pattern_strs.value_counts()
        n_patterns = len(pattern_counts)

        # Pattern table
        pattern_rows = []
        for pat, cnt in pattern_counts.head(15).items():
            cols_missing = [df.columns[i] for i, v in enumerate(pat) if v == "1"]
            desc = ", ".join(cols_missing) if cols_missing else "(complete)"
            pattern_rows.append(f"  {cnt:>6} rows ({cnt / n_rows * 100:.1f}%): {desc}")
        pattern_text = "\n".join(pattern_rows)

        # Little's MCAR test approximation
        mcar_text = ""
        if len(cols_with_missing) >= 2 and len(cols_with_missing) < n_cols:
            try:
                observed_counts = pattern_counts.values
                # Expected under MCAR: each column missing independently
                col_miss_rates = miss_count / n_rows
                expected_probs = []
                for pat in pattern_counts.index:
                    prob = 1.0
                    for i, v in enumerate(pat):
                        p_miss = col_miss_rates.iloc[i]
                        prob *= p_miss if v == "1" else (1 - p_miss)
                    expected_probs.append(prob)
                expected_counts = np.array(expected_probs) * n_rows
                # Filter out zero-expected
                mask = expected_counts > 0.5
                if mask.sum() >= 2:
                    from scipy.stats import chi2

                    obs = observed_counts[mask]
                    exp = expected_counts[mask]
                    chi2_stat = float(np.sum((obs - exp) ** 2 / exp))
                    dof = int(mask.sum() - 1)
                    p_val = float(1 - chi2.cdf(chi2_stat, dof))
                    conclusion = "Data appears MCAR (p >= 0.05)" if p_val >= 0.05 else "Data may NOT be MCAR (p < 0.05)"
                    mcar_text = f"""
MCAR Test (Chi-squared approximation):
  Chi-squared: {chi2_stat:.2f}  (df={dof})
  p-value: {p_val:.4f}
  Conclusion: {conclusion}"""
            except Exception:
                mcar_text = "\nMCAR Test: Could not compute (insufficient patterns)"

        # Missing correlation
        miss_corr_text = ""
        if len(cols_with_missing) >= 2:
            miss_corr = miss_indicator[cols_with_missing.index].corr()
            pairs = []
            idx_list = cols_with_missing.index.tolist()
            for i in range(len(idx_list)):
                for j in range(i + 1, len(idx_list)):
                    pairs.append((idx_list[i], idx_list[j], miss_corr.loc[idx_list[i], idx_list[j]]))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            if pairs:
                lines = [f"  {a} <-> {b}: {v:+.3f}" for a, b, v in pairs[:5]]
                miss_corr_text = "\nMissing Correlation (top pairs):\n" + "\n".join(lines)

        summary_lines = ["MISSING DATA ANALYSIS", "=" * 50]
        summary_lines.append(f"Dataset: {n_rows} rows x {n_cols} columns")
        summary_lines.append(
            f"Total missing: {int(miss_count.sum())} / {n_rows * n_cols} ({miss_count.sum() / (n_rows * n_cols) * 100:.1f}%)"
        )
        summary_lines.append(f"Complete rows: {complete_rows} / {n_rows} ({complete_rows / n_rows * 100:.1f}%)")
        summary_lines.append(f"\nColumns with missing data ({len(cols_with_missing)}):")
        for col, cnt in cols_with_missing.items():
            summary_lines.append(f"  {col:<30} {cnt:>6} ({miss_pct[col]:.1f}%)")
        summary_lines.append(f"\nMissing Patterns ({n_patterns} unique):")
        summary_lines.append(pattern_text)
        if mcar_text:
            summary_lines.append(mcar_text)
        if miss_corr_text:
            summary_lines.append(miss_corr_text)

        result["summary"] = "\n".join(summary_lines)

        # Plot 1: Missing pattern heatmap
        if len(cols_with_missing) > 0:
            top_patterns = pattern_counts.head(20)
            z_data = []
            y_labels = []
            for pat, cnt in top_patterns.items():
                z_data.append([int(v) for v in pat])
                cols_miss = sum(1 for v in pat if v == "1")
                y_labels.append(f"{cnt} rows ({cols_miss} missing)")
            result["plots"].append(
                {
                    "data": [
                        {
                            "type": "heatmap",
                            "z": z_data,
                            "x": df.columns.tolist(),
                            "y": y_labels,
                            "colorscale": [[0, "rgba(74,144,217,0.15)"], [1, "rgba(232,71,71,0.7)"]],
                            "showscale": False,
                            "hovertemplate": "%{x}: %{z}<extra>0=present, 1=missing</extra>",
                        }
                    ],
                    "layout": {
                        "title": "Missing Data Patterns",
                        "height": max(300, len(z_data) * 25 + 100),
                        "xaxis": {"tickangle": -45},
                        "margin": {"b": 100, "l": 150},
                    },
                }
            )

        # Plot 2: Row completeness histogram
        result["plots"].append(
            {
                "data": [
                    {
                        "type": "histogram",
                        "x": row_completeness.tolist(),
                        "nbinsx": 20,
                        "marker": {"color": "rgba(74,159,110,0.6)"},
                    }
                ],
                "layout": {
                    "title": "Row Completeness Distribution",
                    "height": 280,
                    "xaxis": {"title": "% Complete", "range": [0, 105]},
                    "yaxis": {"title": "Row Count"},
                },
            }
        )

        result["guide_observation"] = (
            f"Missing data: {int(miss_count.sum())} cells across {len(cols_with_missing)} columns. {n_patterns} unique patterns."
        )

        # Narrative
        _md_total = int(miss_count.sum())
        _md_pct = _md_total / (n_rows * n_cols) * 100 if n_rows * n_cols > 0 else 0
        _md_worst = cols_with_missing.index[0] if len(cols_with_missing) > 0 else "N/A"
        _md_worst_pct = float(miss_pct[_md_worst]) if len(cols_with_missing) > 0 else 0
        _md_severity = "minimal" if _md_pct < 5 else ("moderate" if _md_pct < 20 else "substantial")
        result["narrative"] = _narrative(
            f"Missing Data — {_md_severity} ({_md_pct:.1f}% of cells)",
            f"{_md_total:,} missing cells across {len(cols_with_missing)} columns in {n_patterns} unique patterns. "
            + (
                f"Worst column: <strong>{_md_worst}</strong> ({_md_worst_pct:.1f}% missing). "
                if len(cols_with_missing) > 0
                else ""
            )
            + f"{complete_rows} of {n_rows} rows ({complete_rows / n_rows * 100:.1f}%) are complete.",
            next_steps="If missingness is random (MCAR), listwise deletion is safe. Otherwise consider multiple imputation.",
            chart_guidance="The heatmap shows which columns are missing together (correlated patterns suggest non-random missingness).",
        )

    except Exception as e:
        result["summary"] = f"Missing data analysis error: {str(e)}"

    return result


def run_outlier_analysis(df, config):
    """Outlier Analysis — IQR, Z-score, Modified Z, Mahalanobis methods."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    try:
        columns = config.get("columns", [])
        methods = config.get("methods", ["iqr"])
        iqr_mult = float(config.get("iqr_multiplier", 1.5))
        z_thresh = float(config.get("zscore_threshold", 3.0))

        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        if not columns:
            result["summary"] = "No numeric columns found for outlier analysis."
            return result

        all_results = {}
        consensus = np.zeros(len(df), dtype=int)

        for col in columns:
            vals = df[col].dropna()
            col_results = {}

            if "iqr" in methods:
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
                mask = (df[col] < lower) | (df[col] > upper)
                col_results["IQR"] = {
                    "count": int(mask.sum()),
                    "pct": round(mask.sum() / len(df) * 100, 1),
                    "bounds": f"[{lower:.4g}, {upper:.4g}]",
                }
                consensus += mask.astype(int).values

            if "zscore" in methods:
                z = np.abs((df[col] - vals.mean()) / vals.std()) if vals.std() > 0 else pd.Series(0, index=df.index)
                mask = z > z_thresh
                col_results["Z-score"] = {
                    "count": int(mask.sum()),
                    "pct": round(mask.sum() / len(df) * 100, 1),
                    "threshold": z_thresh,
                }
                consensus += mask.astype(int).values

            if "modified_zscore" in methods:
                median = vals.median()
                mad = np.median(np.abs(vals - median))
                if mad > 0:
                    modified_z = 0.6745 * np.abs(df[col] - median) / mad
                    mask = modified_z > 3.5
                else:
                    mask = pd.Series(False, index=df.index)
                col_results["Modified Z"] = {
                    "count": int(mask.sum()),
                    "pct": round(mask.sum() / len(df) * 100, 1),
                    "threshold": 3.5,
                }
                consensus += mask.astype(int).values

            if "mahalanobis" in methods and len(columns) >= 2:
                try:
                    from scipy.spatial.distance import mahalanobis as mah_dist

                    sub = df[columns].dropna()
                    if len(sub) > len(columns):
                        cov = np.cov(sub.T)
                        cov_inv = np.linalg.inv(cov + np.eye(len(columns)) * 1e-6)
                        mean = sub.mean().values
                        dists = sub.apply(lambda r: mah_dist(r.values, mean, cov_inv), axis=1)
                        from scipy.stats import chi2 as chi2_dist

                        threshold = chi2_dist.ppf(0.975, df=len(columns))
                        mask_full = dists > np.sqrt(threshold)
                        col_results["Mahalanobis"] = {
                            "count": int(mask_full.sum()),
                            "pct": round(mask_full.sum() / len(sub) * 100, 1),
                            "threshold": f"chi2(p=0.975, df={len(columns)})",
                        }
                except Exception:
                    col_results["Mahalanobis"] = {"count": 0, "pct": 0, "error": "Could not compute"}

            all_results[col] = col_results

        # Build summary
        summary_lines = ["OUTLIER ANALYSIS", "=" * 50]
        summary_lines.append(f"Columns: {', '.join(columns)}")
        summary_lines.append(f"Methods: {', '.join(methods)}")
        summary_lines.append(f"Rows: {len(df)}\n")

        for col, methods_res in all_results.items():
            summary_lines.append(f"{col}:")
            for method, info in methods_res.items():
                summary_lines.append(f"  {method:<18} {info['count']:>5} outliers ({info['pct']}%)")
            summary_lines.append("")

        # Consensus
        n_methods_used = len([m for m in methods if m != "mahalanobis" or len(columns) >= 2])
        if n_methods_used >= 2:
            int((consensus >= n_methods_used * len(columns)).sum()) if n_methods_used > 0 else 0
            flagged_majority = int((consensus >= max(1, n_methods_used * len(columns) // 2)).sum())
            summary_lines.append("Consensus:")
            summary_lines.append(f"  Flagged by majority of methods: {flagged_majority} rows")

        result["summary"] = "\n".join(summary_lines)

        # Table
        table_html = "<table class='result-table'><tr><th>Column</th><th>Method</th><th>Outliers</th><th>%</th><th>Details</th></tr>"
        for col, methods_res in all_results.items():
            for method, info in methods_res.items():
                detail = info.get("bounds", info.get("threshold", ""))
                table_html += f"<tr><td>{col}</td><td>{method}</td><td>{info['count']}</td><td>{info['pct']}%</td><td>{detail}</td></tr>"
        table_html += "</table>"
        result["tables"] = [{"title": "Outlier Summary", "html": table_html}]

        # Plot: Boxplots
        for col in columns[:6]:
            vals = df[col].dropna().tolist()
            result["plots"].append(
                {
                    "data": [
                        {
                            "type": "box",
                            "y": vals,
                            "name": col,
                            "boxpoints": "outliers",
                            "marker": {"color": "rgba(74,144,217,0.6)", "outliercolor": "rgba(232,71,71,0.8)"},
                            "line": {"color": "rgba(74,144,217,0.8)"},
                        }
                    ],
                    "layout": {"title": f"Outlier Detection: {col}", "height": 300, "yaxis": {"title": col}},
                }
            )

        result["guide_observation"] = f"Outlier analysis on {len(columns)} columns with {len(methods)} methods."

        # Narrative
        _oa_total = sum(info["count"] for col_res in all_results.values() for info in col_res.values())
        _oa_worst_col = (
            max(all_results.keys(), key=lambda c: max(info["count"] for info in all_results[c].values()))
            if all_results
            else ""
        )
        _oa_worst_n = max(info["count"] for info in all_results[_oa_worst_col].values()) if _oa_worst_col else 0
        result["narrative"] = _narrative(
            f"Outlier Analysis — {len(columns)} columns, {len(methods)} method{'s' if len(methods) > 1 else ''}",
            (
                f"Most outliers detected in <strong>{_oa_worst_col}</strong> ({_oa_worst_n} points). "
                if _oa_worst_col
                else ""
            )
            + f"Methods used: {', '.join(methods)}. Review boxplots for visual confirmation before removing any points.",
            next_steps="Investigate outliers for data entry errors or special causes. Do not blindly remove — they may carry process information.",
            chart_guidance="Points beyond whiskers in the boxplots are flagged. Compare across methods for consensus.",
        )

    except Exception as e:
        result["summary"] = f"Outlier analysis error: {str(e)}"

    return result


def run_duplicate_analysis(df, config):
    """Duplicate Analysis — exact and subset duplicate detection."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    try:
        mode = config.get("mode", "exact")
        subset_cols = config.get("subset_columns", [])

        if mode == "subset" and subset_cols:
            check_cols = [c for c in subset_cols if c in df.columns]
        else:
            check_cols = df.columns.tolist()

        duplicated_mask = df.duplicated(subset=check_cols, keep=False)
        n_dup_rows = int(duplicated_mask.sum())
        n_dup_groups = int(df[duplicated_mask].groupby(check_cols).ngroups) if n_dup_rows > 0 else 0
        first_dup_mask = df.duplicated(subset=check_cols, keep="first")
        n_extra = int(first_dup_mask.sum())

        summary_lines = ["DUPLICATE ANALYSIS", "=" * 50]
        summary_lines.append(
            f"Mode: {'Exact (all columns)' if mode == 'exact' else 'Subset (' + ', '.join(check_cols) + ')'}"
        )
        summary_lines.append(f"Total rows: {len(df)}")
        summary_lines.append(f"Unique rows: {len(df) - n_extra}")
        summary_lines.append(f"Duplicate rows: {n_dup_rows} ({n_dup_rows / len(df) * 100:.1f}%)")
        summary_lines.append(f"Duplicate groups: {n_dup_groups}")
        summary_lines.append(f"Extra copies (removable): {n_extra}")

        # Show top duplicate groups
        if n_dup_rows > 0:
            dup_df = df[duplicated_mask].copy()
            group_sizes = dup_df.groupby(check_cols).size().sort_values(ascending=False)
            summary_lines.append("\nLargest duplicate groups:")
            for i, (vals, cnt) in enumerate(group_sizes.head(10).items()):
                if isinstance(vals, tuple):
                    desc = ", ".join(f"{c}={v}" for c, v in zip(check_cols, vals))
                else:
                    desc = f"{check_cols[0]}={vals}"
                summary_lines.append(f"  Group {i + 1}: {cnt} copies — {desc[:80]}")

        result["summary"] = "\n".join(summary_lines)

        # Table of sample duplicates
        if n_dup_rows > 0:
            sample = df[duplicated_mask].head(20)
            table_html = "<table class='result-table'><tr>"
            for c in sample.columns:
                table_html += f"<th>{c}</th>"
            table_html += "<th>Row#</th></tr>"
            for idx, row in sample.iterrows():
                table_html += "<tr>"
                for c in sample.columns:
                    table_html += f"<td>{row[c]}</td>"
                table_html += f"<td>{idx}</td></tr>"
            table_html += "</table>"
            result["tables"] = [{"title": "Sample Duplicate Rows", "html": table_html}]

        # Plot: duplicate group size histogram
        if n_dup_groups > 0:
            group_sizes_list = dup_df.groupby(check_cols).size().tolist()
            result["plots"].append(
                {
                    "data": [{"type": "histogram", "x": group_sizes_list, "marker": {"color": "rgba(232,149,71,0.6)"}}],
                    "layout": {
                        "title": "Duplicate Group Sizes",
                        "height": 280,
                        "xaxis": {"title": "Copies per Group"},
                        "yaxis": {"title": "Number of Groups"},
                    },
                }
            )

        result["guide_observation"] = (
            f"Duplicate analysis ({mode}): {n_dup_rows} duplicate rows in {n_dup_groups} groups."
        )

        # Narrative
        _da_pct = n_dup_rows / len(df) * 100 if len(df) > 0 else 0
        if n_dup_rows == 0:
            _da_verdict = "No duplicates found"
            _da_body = f"All {len(df):,} rows are unique across the checked columns."
        else:
            _da_verdict = f"{n_dup_rows:,} duplicate rows ({_da_pct:.1f}%)"
            _da_body = f"{n_dup_groups} duplicate groups found. {n_extra} rows are extra copies that could be removed, leaving {len(df) - n_extra:,} unique rows."
        result["narrative"] = _narrative(
            f"Duplicate Analysis — {_da_verdict}",
            _da_body,
            next_steps="Verify duplicates are true repeats (not valid repeat measurements) before removing."
            if n_dup_rows > 0
            else None,
            chart_guidance="The histogram shows how many copies exist per duplicate group." if n_dup_rows > 0 else None,
        )

    except Exception as e:
        result["summary"] = f"Duplicate analysis error: {str(e)}"

    return result

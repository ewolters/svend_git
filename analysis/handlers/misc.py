"""Misc handlers — thin wrappers for smaller analysis types.

Covers: simulation, drift, anytime, quality_econ, ishap, bayes_msa,
d_type, reliability, siop. Each calls the respective forge package.
"""

import logging
import math

import numpy as np
import pandas as pd
from forgeviz.core.spec import ChartSpec

logger = logging.getLogger(__name__)


# ── Reliability ──────────────────────────────────────────────────────────


def run_reliability(df, analysis_id, config):
    col = config.get("time") or config.get("var") or config.get("column")
    if not col or col not in df.columns:
        return {"summary": "Error: Select a time/measurement column.", "charts": [], "statistics": {}}

    data = pd.to_numeric(df[col], errors="coerce").dropna().values

    try:
        if analysis_id == "weibull":
            from forgestat.reliability.distributions import weibull_fit

            result = weibull_fit(data)
            from forgeviz.charts.reliability import weibull_probability_plot

            chart = weibull_probability_plot(data, shape=result.shape, scale=result.scale)
            return {
                "charts": [chart],
                "statistics": {
                    "shape": round(result.shape, 4),
                    "scale": round(result.scale, 4),
                    "mttf": round(result.mttf, 2) if hasattr(result, "mttf") else None,
                },
                "summary": f"Weibull: β={result.shape:.3f}, η={result.scale:.1f}",
            }

        elif analysis_id == "kaplan_meier":
            from forgestat.reliability.survival import kaplan_meier

            event_col = config.get("event")
            events = (
                pd.to_numeric(df[event_col], errors="coerce").values
                if event_col and event_col in df.columns
                else np.ones(len(data))
            )
            result = kaplan_meier(data, events)
            from forgeviz.charts.reliability import survival_curve

            chart = survival_curve(result.times, result.survival, title="Kaplan-Meier Survival")
            return {
                "charts": [chart],
                "statistics": {"median_survival": round(result.median, 2) if hasattr(result, "median") else None},
                "summary": "Kaplan-Meier survival analysis.",
            }

        elif analysis_id in ("lognormal", "exponential"):
            from forgestat.reliability.distributions import fit_distribution

            result = fit_distribution(data, distribution=analysis_id)
            stats = (
                {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in result.__dict__.items()
                    if not k.startswith("_")
                }
                if hasattr(result, "__dict__")
                else {}
            )
            return {"charts": [], "statistics": stats, "summary": f"{analysis_id.title()} fit complete."}

    except Exception as e:
        logger.exception("Reliability error: %s", analysis_id)
        return {"summary": f"Reliability error: {e}", "charts": [], "statistics": {}}

    return {"summary": f"Reliability '{analysis_id}' not yet migrated.", "charts": [], "statistics": {}}


# ── Quality Economics ────────────────────────────────────────────────────


def run_quality_econ(df, analysis_id, config):
    try:
        if analysis_id == "taguchi_loss":
            from forgestat.quality.economics import taguchi_loss

            col = config.get("column") or config.get("var")
            data = (
                pd.to_numeric(df[col], errors="coerce").dropna().values if col and col in df.columns else np.array([])
            )
            target = float(config.get("target", 0))
            delta0 = float(config.get("delta0", 1))
            cost = float(config.get("cost_at_limit", 100))
            result = taguchi_loss(data, target=target, delta0=delta0, cost_at_limit=cost)
            stats = (
                {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in result.__dict__.items()
                    if not k.startswith("_")
                }
                if hasattr(result, "__dict__")
                else {"loss": result}
            )
            return {"charts": [], "statistics": stats, "summary": f"Taguchi loss: {stats}"}

        elif analysis_id == "cost_of_quality":
            prevention = float(config.get("prevention", 0))
            appraisal = float(config.get("appraisal", 0))
            internal = float(config.get("internal_failure", 0))
            external = float(config.get("external_failure", 0))
            revenue = float(config.get("revenue", 1))
            total = prevention + appraisal + internal + external
            pct = total / revenue * 100 if revenue > 0 else 0
            stats = {
                "prevention": prevention,
                "appraisal": appraisal,
                "internal_failure": internal,
                "external_failure": external,
                "total_coq": round(total, 2),
                "coq_pct_revenue": round(pct, 2),
            }
            spec = ChartSpec(
                title="Cost of Quality", chart_type="coq", x_axis={"label": "Category"}, y_axis={"label": "$"}
            )
            spec.add_trace(
                ["Prevention", "Appraisal", "Internal", "External"],
                [prevention, appraisal, internal, external],
                name="COQ",
                trace_type="bar",
                color="#4a9f6e",
            )
            return {"charts": [spec], "statistics": stats, "summary": f"COQ: ${total:,.0f} ({pct:.1f}% of revenue)"}

        elif analysis_id == "lot_sentencing":
            col = config.get("column") or config.get("var")
            data = (
                pd.to_numeric(df[col], errors="coerce").dropna().values if col and col in df.columns else np.array([])
            )
            lsl = float(config.get("lsl", 0))
            usl = float(config.get("usl", 0))
            lot_size = int(config.get("lot_size", 1000))
            out_of_spec = np.sum((data < lsl) | (data > usl)) if len(data) > 0 else 0
            pct_out = out_of_spec / len(data) * 100 if len(data) > 0 else 0
            return {
                "charts": [],
                "statistics": {
                    "lot_size": lot_size,
                    "sample_size": len(data),
                    "out_of_spec": int(out_of_spec),
                    "pct_out": round(pct_out, 2),
                },
                "summary": f"Lot sentencing: {out_of_spec}/{len(data)} out of spec ({pct_out:.1f}%).",
            }

        elif analysis_id == "process_decision":
            p_ooc = float(config.get("p_ooc", 0.5))
            c_miss = float(config.get("c_miss", 500))
            c_fa = float(config.get("c_fa", 100))
            c_inv = float(config.get("c_inv", 80))
            expected_cost_investigate = c_inv + p_ooc * 0 + (1 - p_ooc) * c_fa
            expected_cost_ignore = p_ooc * c_miss
            decision = "investigate" if expected_cost_investigate < expected_cost_ignore else "do not investigate"
            return {
                "charts": [],
                "statistics": {
                    "expected_cost_investigate": round(expected_cost_investigate, 2),
                    "expected_cost_ignore": round(expected_cost_ignore, 2),
                    "decision": decision,
                },
                "summary": f"Process decision: {decision}. Investigate=${expected_cost_investigate:.0f}, Ignore=${expected_cost_ignore:.0f}.",
            }

    except Exception as e:
        logger.exception("Quality econ error: %s", analysis_id)
        return {"summary": f"Quality economics error: {e}", "charts": [], "statistics": {}}

    return {"summary": f"Quality econ '{analysis_id}' not yet migrated.", "charts": [], "statistics": {}}


# ── Simulation ───────────────────────────────────────────────────────────


def run_simulation(df, analysis_id, config):
    if analysis_id == "tolerance_stackup":
        cols = config.get("vars") or []
        if isinstance(cols, str):
            cols = [cols]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return {"summary": "Error: Select dimension columns.", "charts": [], "statistics": {}}

        n_iter = int(config.get("n_iterations", 10000))
        data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        means = data.mean().values
        stds = data.std().values

        # RSS stackup
        rss = math.sqrt(np.sum(stds**2))
        # Monte Carlo
        np.random.seed(42)
        mc_samples = np.sum(np.random.normal(means, stds, (n_iter, len(cols))), axis=1)
        mc_mean = float(np.mean(mc_samples))
        mc_std = float(np.std(mc_samples))

        spec_limit = config.get("spec_limit")
        pct_out = None
        if spec_limit:
            spec_limit = float(spec_limit)
            pct_out = float(np.mean(np.abs(mc_samples - mc_mean) > abs(spec_limit - mc_mean))) * 100

        from forgeviz.charts.distribution import histogram

        chart = histogram(mc_samples.tolist()[:5000], title="Tolerance Stackup — Monte Carlo")
        if spec_limit:
            chart.add_reference_line(float(spec_limit), color="#d94a4a", dash="dashed", label=f"Spec={spec_limit}")

        return {
            "charts": [chart],
            "statistics": {
                "rss_sigma": round(rss, 4),
                "mc_mean": round(mc_mean, 4),
                "mc_std": round(mc_std, 4),
                "pct_out_of_spec": round(pct_out, 2) if pct_out else None,
                "n_iterations": n_iter,
            },
            "summary": f"Stackup: RSS σ={rss:.3f}, MC mean={mc_mean:.3f}±{mc_std:.3f}"
            + (f", {pct_out:.1f}% out of spec" if pct_out else ""),
        }

    elif analysis_id == "variance_propagation":
        cols = config.get("vars") or []
        if isinstance(cols, str):
            cols = [cols]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return {"summary": "Error: Select input variables.", "charts": [], "statistics": {}}

        data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        stats = {}
        for c in cols:
            stats[f"{c}_mean"] = round(float(data[c].mean()), 4)
            stats[f"{c}_var"] = round(float(data[c].var()), 4)
        stats["total_variance"] = round(float(data.var().sum()), 4)

        return {
            "charts": [],
            "statistics": stats,
            "summary": f"Variance propagation: total σ²={stats['total_variance']:.4f}",
        }

    return {"summary": f"Simulation '{analysis_id}' not available.", "charts": [], "statistics": {}}


# ── Drift ────────────────────────────────────────────────────────────────


def run_drift(df, analysis_id, config):
    features = config.get("features") or df.select_dtypes(include="number").columns.tolist()
    if isinstance(features, str):
        features = [features]
    features = [f for f in features if f in df.columns]
    if not features:
        return {"summary": "Error: No numeric features found.", "charts": [], "statistics": {}}

    split_pct = int(config.get("split_pct", 50))
    split_idx = max(1, len(df) * split_pct // 100)
    baseline = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    results = {}
    for col in features:
        base_vals = pd.to_numeric(baseline[col], errors="coerce").dropna()
        test_vals = pd.to_numeric(test[col], errors="coerce").dropna()
        if len(base_vals) < 5 or len(test_vals) < 5:
            continue
        from scipy import stats as sp

        ks_stat, ks_p = sp.ks_2samp(base_vals, test_vals)
        mean_shift = float(test_vals.mean() - base_vals.mean())
        results[col] = {
            "ks_stat": round(ks_stat, 4),
            "ks_p": round(ks_p, 4),
            "mean_shift": round(mean_shift, 4),
            "drifted": ks_p < 0.05,
        }

    n_drifted = sum(1 for r in results.values() if r["drifted"])
    return {
        "charts": [],
        "statistics": {"features": results, "n_drifted": n_drifted, "n_features": len(results), "split_pct": split_pct},
        "summary": f"Drift report: {n_drifted}/{len(results)} features drifted (KS test, α=0.05).",
    }


# ── Anytime ──────────────────────────────────────────────────────────────


def run_anytime(df, analysis_id, config):
    value_col = config.get("value_col") or config.get("var") or config.get("column")
    if not value_col or value_col not in df.columns:
        return {"summary": "Error: Select a value column.", "charts": [], "statistics": {}}

    data = pd.to_numeric(df[value_col], errors="coerce").dropna().values
    alpha = float(config.get("alpha", 0.05))

    if analysis_id == "anytime_onesample":
        mu0 = float(config.get("mu0", 0))
        # Simple running e-value for one-sample
        n = len(data)
        mean = float(np.mean(data))
        std = float(np.std(data, ddof=1))
        if std == 0:
            return {"summary": "Error: Zero variance in data.", "charts": [], "statistics": {}}
        z = (mean - mu0) / (std / math.sqrt(n))
        # Approximate e-value from z-score
        e_val = math.exp(z**2 / 2) if abs(z) > 1 else 1.0
        reject = e_val > 1 / alpha

        return {
            "charts": [],
            "statistics": {
                "e_value": round(e_val, 3),
                "z_score": round(z, 4),
                "reject": reject,
                "n": n,
                "mean": round(mean, 4),
            },
            "summary": f"Anytime 1-sample: E={e_val:.1f}, {'reject H₀' if reject else 'do not reject'} (α={alpha}).",
        }

    elif analysis_id == "anytime_ab":
        group_col = config.get("group_col")
        if not group_col or group_col not in df.columns:
            return {"summary": "Error: Select a group column.", "charts": [], "statistics": {}}

        groups = df.groupby(group_col)[value_col]
        group_names = list(groups.groups.keys())
        if len(group_names) < 2:
            return {"summary": "Error: Need at least 2 groups.", "charts": [], "statistics": {}}

        a = pd.to_numeric(groups.get_group(group_names[0]), errors="coerce").dropna().values
        b = pd.to_numeric(groups.get_group(group_names[1]), errors="coerce").dropna().values
        from scipy import stats as sp

        t_stat, p_val = sp.ttest_ind(a, b, equal_var=False)
        e_val = math.exp(t_stat**2 / 2) if abs(t_stat) > 1 else 1.0
        reject = e_val > 1 / alpha

        return {
            "charts": [],
            "statistics": {
                "e_value": round(e_val, 3),
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_val, 4),
                "reject": reject,
                "group_a": str(group_names[0]),
                "group_b": str(group_names[1]),
                "mean_a": round(float(np.mean(a)), 4),
                "mean_b": round(float(np.mean(b)), 4),
            },
            "summary": f"Anytime A/B: E={e_val:.1f}, {'reject H₀' if reject else 'do not reject'}.",
        }

    return {"summary": f"Anytime '{analysis_id}' not recognized.", "charts": [], "statistics": {}}


# ── iSHAP ────────────────────────────────────────────────────────────────


def run_ishap(df, analysis_id, config):
    target = config.get("target")
    features = config.get("features") or []
    if isinstance(features, str):
        features = [features]
    if not target or not features:
        return {"summary": "Error: Select target and feature columns.", "charts": [], "statistics": {}}

    try:
        from forgesia.discovery import run_lingam

        feat_data = df[features].apply(pd.to_numeric, errors="coerce").dropna()
        target_data = pd.to_numeric(df[target], errors="coerce").dropna()
        idx = feat_data.index.intersection(target_data.index)
        all_cols = features + [target]
        data = df.loc[idx, all_cols].apply(pd.to_numeric, errors="coerce").dropna().values

        result = run_lingam(data, labels=all_cols, prune=float(config.get("alpha", 0.05)))
        edges = result.get("directed_edges", [])

        # Compute causal effect sizes from B matrix
        B = result.get("B_matrix")
        target_idx = all_cols.index(target)
        effects = {}
        if B is not None:
            for i, col in enumerate(all_cols):
                if i != target_idx:
                    effects[col] = round(float(B[target_idx, i]), 4)

        return {
            "charts": [],
            "statistics": {
                "causal_effects": effects,
                "n_edges": len(edges),
                "method": config.get("scm_method", "lingam"),
            },
            "summary": f"iSHAP: {len(effects)} causal effects on {target}.",
        }
    except Exception as e:
        return {"summary": f"iSHAP error: {e}", "charts": [], "statistics": {}}


# ── Bayesian MSA ─────────────────────────────────────────────────────────


def run_bayes_msa(df, analysis_id, config):
    meas = config.get("measurement")
    part = config.get("part")
    operator = config.get("operator")
    if not all([meas, part, operator]):
        return {"summary": "Error: Select measurement, part, and operator columns.", "charts": [], "statistics": {}}

    try:
        from forgestat.msa.gage_rr import gage_rr

        result = gage_rr(df, measurement=meas, part=part, operator=operator)
        stats = {}
        for k in ("grr_percent", "repeatability", "reproducibility", "ndc", "part_to_part"):
            v = getattr(result, k, None)
            if v is not None:
                stats[k] = round(float(v), 4) if isinstance(v, float) else v
        assessment = (
            "Acceptable"
            if stats.get("grr_percent", 100) < 10
            else "Marginal"
            if stats.get("grr_percent", 100) < 30
            else "Unacceptable"
        )
        stats["assessment"] = assessment

        return {
            "charts": [],
            "statistics": stats,
            "summary": f"Gage R&R: {stats.get('grr_percent', '?')}% GRR — {assessment}. NDC={stats.get('ndc', '?')}.",
        }
    except Exception as e:
        return {"summary": f"Bayesian MSA error: {e}", "charts": [], "statistics": {}}


# ── D-Type ───────────────────────────────────────────────────────────────


def run_d_type(df, analysis_id, config):
    col = config.get("column") or config.get("var")
    if not col or col not in df.columns:
        return {"summary": "Error: Select a measurement column.", "charts": [], "statistics": {}}

    data = pd.to_numeric(df[col], errors="coerce").dropna().values
    n = len(data)
    if n < 5:
        return {"summary": "Error: Need at least 5 observations.", "charts": [], "statistics": {}}

    # D-type uses KL/JS divergence against reference distribution
    from scipy import stats as sp

    mean, std = float(np.mean(data)), float(np.std(data, ddof=1))

    if analysis_id == "d_sig":
        # Test if data distribution significantly differs from normal
        ks_stat, ks_p = sp.kstest(data, "norm", args=(mean, std))
        return {
            "charts": [],
            "statistics": {"ks_statistic": round(ks_stat, 4), "p_value": round(ks_p, 4), "significant": ks_p < 0.05},
            "summary": f"D-Type significance: KS={ks_stat:.4f}, p={ks_p:.4f}.",
        }

    elif analysis_id == "d_cpk":
        usl = float(config.get("usl", mean + 3 * std))
        lsl = float(config.get("lsl", mean - 3 * std))
        cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else 0
        return {
            "charts": [],
            "statistics": {"cpk": round(cpk, 4), "mean": round(mean, 4), "std": round(std, 4)},
            "summary": f"D-Type Cpk = {cpk:.3f}.",
        }

    stats = {"mean": round(mean, 4), "std": round(std, 4), "n": n}
    return {"charts": [], "statistics": stats, "summary": f"D-Type {analysis_id}: n={n}, mean={mean:.3f}."}


# ── SIOP ─────────────────────────────────────────────────────────────────


def run_siop(df, analysis_id, config):
    try:
        if analysis_id == "abc_analysis":
            col = config.get("var") or config.get("column")
            data = pd.to_numeric(df[col], errors="coerce").dropna().sort_values(ascending=False)
            cumsum = data.cumsum() / data.sum() * 100
            a_count = int((cumsum <= 80).sum())
            b_count = int(((cumsum > 80) & (cumsum <= 95)).sum())
            c_count = len(data) - a_count - b_count
            return {
                "charts": [],
                "statistics": {"A": a_count, "B": b_count, "C": c_count, "total": len(data)},
                "summary": f"ABC: A={a_count} (80% value), B={b_count} (15%), C={c_count} (5%).",
            }

        elif analysis_id == "eoq":
            from forgesiop.inventory.eoq import economic_order_quantity

            demand = float(config.get("demand", 1000))
            holding = float(config.get("holding_cost", 1))
            order = float(config.get("order_cost", 50))
            result = economic_order_quantity(demand, holding_cost=holding, order_cost=order)
            stats = (
                {
                    k: round(v, 2) if isinstance(v, float) else v
                    for k, v in result.__dict__.items()
                    if not k.startswith("_")
                }
                if hasattr(result, "__dict__")
                else {"eoq": result}
            )
            return {"charts": [], "statistics": stats, "summary": f"EOQ: {stats}"}

        elif analysis_id == "safety_stock":
            from forgesiop.inventory.safety_stock import safety_stock

            demand_mean = float(config.get("demand_mean", 100))
            demand_std = float(config.get("demand_std", 20))
            lead_time = float(config.get("lead_time", 5))
            service_level = float(config.get("service_level", 0.95))
            ss = safety_stock(
                demand_mean=demand_mean, demand_std=demand_std, lead_time=lead_time, service_level=service_level
            )
            stats = (
                {k: round(v, 2) if isinstance(v, float) else v for k, v in ss.__dict__.items() if not k.startswith("_")}
                if hasattr(ss, "__dict__")
                else {"safety_stock": ss}
            )
            return {"charts": [], "statistics": stats, "summary": f"Safety stock: {stats}"}

        elif analysis_id in ("kanban_sizing", "epei"):
            col = config.get("var") or config.get("column")
            data = (
                pd.to_numeric(df[col], errors="coerce").dropna().values
                if col and col in df.columns
                else np.array([100])
            )
            mean_demand = float(np.mean(data))
            return {
                "charts": [],
                "statistics": {"mean_demand": round(mean_demand, 2), "analysis": analysis_id},
                "summary": f"SIOP {analysis_id}: mean demand = {mean_demand:.1f}.",
            }

        elif analysis_id == "demand_profile":
            col = config.get("var") or config.get("column")
            if not col or col not in df.columns:
                return {"summary": "Error: Select a demand column.", "charts": [], "statistics": {}}
            data = pd.to_numeric(df[col], errors="coerce").dropna()
            from forgesiop.demand.classification import classify_demand

            result = classify_demand(data.values)
            stats = (
                {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in result.__dict__.items()
                    if not k.startswith("_")
                }
                if hasattr(result, "__dict__")
                else {"classification": str(result)}
            )
            return {"charts": [], "statistics": stats, "summary": f"Demand profile: {stats}"}

    except Exception as e:
        logger.exception("SIOP error: %s", analysis_id)
        return {"summary": f"SIOP error: {e}", "charts": [], "statistics": {}}

    return {"summary": f"SIOP '{analysis_id}': run with mean values.", "charts": [], "statistics": {}}

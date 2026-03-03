"""DSW Monte Carlo Simulation engine."""

import numpy as np
from scipy import stats as sp_stats
import ast
import math

from .common import _fit_best_distribution, _narrative, SVEND_COLORS, COLOR_GOOD, COLOR_BAD, COLOR_WARNING, COLOR_INFO, _rgba


def run_simulation(df, analysis_id, config, user):
    """Simulation engine — Monte Carlo, tolerance stackup, variance propagation."""

    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "tolerance_stackup":
        return _tolerance_stackup(df, config, result)
    elif analysis_id == "variance_propagation":
        return _variance_propagation(df, config, result)
    elif analysis_id != "monte_carlo":
        result["summary"] = f"Unknown simulation: {analysis_id}"
        return result

    variables = config.get("variables", [])
    formula = config.get("transfer_function", "")
    model_id = config.get("model_id")
    n_iter = min(int(config.get("n_iterations", 10000)), 100000)
    thresholds = config.get("thresholds", [])
    seed = config.get("seed")

    if not variables:
        result["summary"] = "Error: No input variables defined."
        return result
    if len(variables) > 20:
        result["summary"] = "Error: Maximum 20 variables supported."
        return result
    if not formula and not model_id:
        result["summary"] = "Error: Provide a transfer function formula or a saved model ID."
        return result

    rng = np.random.default_rng(int(seed) if seed else None)

    # --- Generate input samples ---
    samples = {}
    var_names = []
    for var in variables:
        name = var.get("name", "X")
        dist = var.get("distribution", "normal")
        params = var.get("params", {})
        var_names.append(name)

        if dist == "fit_from_data":
            col = var.get("column")
            if col and col in df.columns:
                col_data = df[col].dropna().values
                _, best_dist, fit_args, _ = _fit_best_distribution(col_data)
                samples[name] = best_dist.rvs(*fit_args, size=n_iter, random_state=rng)
            else:
                samples[name] = rng.normal(0, 1, n_iter)
        elif dist == "normal":
            samples[name] = rng.normal(float(params.get("mean", 0)), max(float(params.get("std", 1)), 1e-9), n_iter)
        elif dist == "uniform":
            lo, hi = float(params.get("low", 0)), float(params.get("high", 1))
            samples[name] = rng.uniform(lo, hi, n_iter)
        elif dist == "lognormal":
            samples[name] = rng.lognormal(float(params.get("mean", 0)), max(float(params.get("sigma", 1)), 1e-9), n_iter)
        elif dist == "weibull":
            samples[name] = sp_stats.weibull_min.rvs(float(params.get("shape", 2)), scale=float(params.get("scale", 1)), size=n_iter, random_state=rng)
        elif dist == "exponential":
            samples[name] = rng.exponential(max(float(params.get("scale", 1)), 1e-9), n_iter)
        elif dist == "gamma":
            samples[name] = rng.gamma(max(float(params.get("shape", 2)), 0.01), max(float(params.get("scale", 1)), 1e-9), n_iter)
        elif dist == "triangular":
            lo = float(params.get("low", 0))
            mode = float(params.get("mode", 0.5))
            hi = float(params.get("high", 1))
            if lo >= hi:
                hi = lo + 1
            mode = max(lo, min(hi, mode))
            samples[name] = rng.triangular(lo, mode, hi, n_iter)
        elif dist == "beta":
            samples[name] = rng.beta(max(float(params.get("a", 2)), 0.01), max(float(params.get("b", 2)), 0.01), n_iter)
        else:
            samples[name] = rng.normal(0, 1, n_iter)

    # --- Evaluate transfer function ---
    outputs = None

    if model_id:
        # Use saved ML model
        try:
            from ..models import SavedModel
            import pickle, pandas as pd
            saved = SavedModel.objects.get(id=model_id, user=user)
            model_obj = pickle.loads(saved.model_data)
            input_df = pd.DataFrame(samples)
            outputs = model_obj.predict(input_df).astype(float)
        except Exception as e:
            result["summary"] = f"Error loading model: {e}"
            return result
    else:
        # Validate formula AST for security
        # Only bare function names allowed — no module attribute access (prevents np.__class__ etc.)
        allowed_names = set(var_names) | {"sqrt", "log", "exp", "abs", "sin", "cos", "tan",
                                          "pi", "max", "min", "pow", "e", "ceil", "floor",
                                          "mean", "std", "sum"}
        try:
            tree = ast.parse(formula, mode="eval")
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id not in allowed_names:
                    result["summary"] = f"Error: Forbidden name '{node.id}' in formula. Allowed: variable names + math functions."
                    return result
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    result["summary"] = "Error: Import statements not allowed in formula."
                    return result
                if isinstance(node, ast.Attribute):
                    result["summary"] = "Error: Attribute access not allowed. Use function names directly (e.g., sqrt(x) not np.sqrt(x))."
                    return result
        except SyntaxError as e:
            result["summary"] = f"Error: Invalid formula syntax — {e}"
            return result

        # Build safe namespace — no raw module references
        safe_ns = {"__builtins__": {}}
        safe_ns.update({name: samples[name] for name in var_names})
        safe_ns.update({
            "sqrt": np.sqrt, "log": np.log, "exp": np.exp, "abs": np.abs,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "pi": np.pi, "e": np.e, "max": np.maximum, "min": np.minimum,
            "pow": np.power, "ceil": np.ceil, "floor": np.floor,
            "mean": np.mean, "std": np.std, "sum": np.sum,
        })

        try:
            outputs = eval(compile(tree, "<formula>", "eval"), safe_ns)
            outputs = np.asarray(outputs, dtype=float).ravel()
        except Exception as e:
            result["summary"] = f"Error evaluating formula: {e}"
            return result

    if outputs is None or len(outputs) == 0:
        result["summary"] = "Error: Simulation produced no output."
        return result

    # --- Compute statistics ---
    outputs = outputs[:n_iter]  # ensure correct length
    out_mean = float(np.mean(outputs))
    out_std = float(np.std(outputs))
    percentiles = {
        "p1": float(np.percentile(outputs, 1)),
        "p5": float(np.percentile(outputs, 5)),
        "p10": float(np.percentile(outputs, 10)),
        "p25": float(np.percentile(outputs, 25)),
        "p50": float(np.percentile(outputs, 50)),
        "p75": float(np.percentile(outputs, 75)),
        "p90": float(np.percentile(outputs, 90)),
        "p95": float(np.percentile(outputs, 95)),
        "p99": float(np.percentile(outputs, 99)),
    }

    # Threshold probabilities
    threshold_results = []
    for t in thresholds:
        val = float(t.get("value", 0))
        direction = t.get("direction", "above")
        if direction == "above":
            prob = float(np.mean(outputs > val))
        else:
            prob = float(np.mean(outputs < val))
        threshold_results.append({"value": val, "direction": direction, "probability": prob})

    # Sensitivity tornado: vary each input +/-1sigma, hold others at mean
    sensitivity = []
    means = {name: float(np.mean(samples[name])) for name in var_names}
    stds = {name: float(np.std(samples[name])) for name in var_names}

    for name in var_names:
        baseline = {n: np.full(1, means[n]) for n in var_names}
        # Low: -1sigma
        low_ns = dict(baseline)
        low_ns[name] = np.full(1, means[name] - stds[name])
        # High: +1sigma
        high_ns = dict(baseline)
        high_ns[name] = np.full(1, means[name] + stds[name])

        if model_id:
            try:
                import pandas as pd
                low_out = float(model_obj.predict(pd.DataFrame(low_ns))[0])
                high_out = float(model_obj.predict(pd.DataFrame(high_ns))[0])
            except Exception:
                low_out = high_out = out_mean
        else:
            try:
                low_safe = dict(safe_ns)
                low_safe.update(low_ns)
                low_out = float(eval(compile(tree, "<formula>", "eval"), low_safe))
                high_safe = dict(safe_ns)
                high_safe.update(high_ns)
                high_out = float(eval(compile(tree, "<formula>", "eval"), high_safe))
            except Exception:
                low_out = high_out = out_mean

        sensitivity.append({
            "variable": name,
            "low": low_out,
            "high": high_out,
            "swing": abs(high_out - low_out),
        })

    sensitivity.sort(key=lambda x: x["swing"], reverse=True)

    # Input-output correlations
    correlations = []
    for name in var_names:
        try:
            r = float(np.corrcoef(samples[name][:len(outputs)], outputs)[0, 1])
        except Exception:
            r = 0.0
        correlations.append({"variable": name, "r": r})
    correlations.sort(key=lambda x: abs(x["r"]), reverse=True)

    # --- Build plots ---

    # 1. Output histogram
    hist_data = outputs.tolist() if len(outputs) <= 10000 else outputs[:10000].tolist()
    threshold_shapes = []
    for t in threshold_results:
        threshold_shapes.append({
            "type": "line", "x0": t["value"], "x1": t["value"],
            "y0": 0, "y1": 1, "yref": "paper",
            "line": {"color": "#dc5050", "width": 2, "dash": "dash"},
        })

    result["plots"].append({
        "title": f"Output Distribution ({n_iter:,} iterations)",
        "data": [{
            "type": "histogram", "x": hist_data,
            "marker": {"color": "rgba(74, 159, 110, 0.5)", "line": {"color": "#4a9f6e", "width": 1}},
            "name": "Output",
        }],
        "layout": {
            "height": 320,
            "xaxis": {"title": "Output Value"}, "yaxis": {"title": "Frequency"},
            "shapes": threshold_shapes,
        },
    })

    # 2. Tornado chart
    if sensitivity:
        labels = [s["variable"] for s in sensitivity[:10]]
        lows = [s["low"] - out_mean for s in sensitivity[:10]]
        highs = [s["high"] - out_mean for s in sensitivity[:10]]

        result["plots"].append({
            "title": "Sensitivity Tornado (+-1sigma impact)",
            "data": [
                {"type": "bar", "y": labels, "x": lows, "orientation": "h",
                 "name": "Low (-1sigma)", "marker": {"color": "#4a90d9"}},
                {"type": "bar", "y": labels, "x": highs, "orientation": "h",
                 "name": "High (+1sigma)", "marker": {"color": "#dc5050"}},
            ],
            "layout": {
                "height": max(200, len(labels) * 30 + 80),
                "barmode": "overlay", "xaxis": {"title": "Change from mean"},
                "yaxis": {"autorange": "reversed"},
            },
        })

    # 3. Correlation bar chart
    if correlations:
        result["plots"].append({
            "title": "Input-Output Correlations",
            "data": [{
                "type": "bar",
                "x": [c["variable"] for c in correlations],
                "y": [c["r"] for c in correlations],
                "marker": {"color": ["#4a9f6e" if c["r"] >= 0 else "#dc5050" for c in correlations]},
            }],
            "layout": {
                "height": 250,
                "yaxis": {"title": "Pearson r", "range": [-1.05, 1.05]},
            },
        })

    # --- Summary ---
    summary = f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n"
    summary += f"<<COLOR:title>>MONTE CARLO SIMULATION<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{'=' * 70}<</COLOR>>\n\n"
    summary += f"<<COLOR:text>>Iterations:<</COLOR>> {n_iter:,}\n"
    summary += f"<<COLOR:text>>Variables:<</COLOR>> {len(var_names)} ({', '.join(var_names)})\n"
    if formula:
        summary += f"<<COLOR:text>>Formula:<</COLOR>> {formula}\n"
    if model_id:
        summary += f"<<COLOR:text>>Model:<</COLOR>> saved model #{model_id}\n"
    summary += f"\n<<COLOR:highlight>>Output Statistics:<</COLOR>>\n"
    summary += f"  Mean: {out_mean:.4f}\n"
    summary += f"  Std Dev: {out_std:.4f}\n"
    summary += f"  Median: {percentiles['p50']:.4f}\n"
    summary += f"  5th percentile: {percentiles['p5']:.4f}\n"
    summary += f"  95th percentile: {percentiles['p95']:.4f}\n"
    summary += f"  Range: [{percentiles['p1']:.4f}, {percentiles['p99']:.4f}]\n"

    if threshold_results:
        summary += f"\n<<COLOR:highlight>>Threshold Analysis:<</COLOR>>\n"
        for t in threshold_results:
            summary += f"  P(output {'>' if t['direction'] == 'above' else '<'} {t['value']:.2f}) = {t['probability']*100:.1f}%\n"

    if sensitivity:
        summary += f"\n<<COLOR:highlight>>Top Drivers (by +-1sigma swing):<</COLOR>>\n"
        for s in sensitivity[:5]:
            summary += f"  {s['variable']}: swing = {s['swing']:.4f}\n"

    result["summary"] = summary
    result["guide_observation"] = f"Monte Carlo simulation ({n_iter:,} iterations): output mean={out_mean:.4f}, std={out_std:.4f}. Top driver: {sensitivity[0]['variable'] if sensitivity else 'N/A'}."

    # Narrative
    _mc_top = sensitivity[0] if sensitivity else None
    _mc_top_pct = (abs(correlations[0]["r"]) ** 2 * 100) if correlations and correlations[0]["r"] else 0
    _mc_spec_note = ""
    if threshold_results:
        _mc_best_t = threshold_results[0]
        _mc_spec_note = f" {_mc_best_t['probability']*100:.1f}% of simulations {'exceed' if _mc_best_t['direction'] == 'above' else 'fall below'} the threshold of {_mc_best_t['value']:.2f}."
    result["narrative"] = _narrative(
        f"Output: {out_mean:.4f} \u00b1 {out_std:.4f} (mean \u00b1 \u03c3)",
        f"After {n_iter:,} simulations, the output ranges from {percentiles['p5']:.4f} (P5) to {percentiles['p95']:.4f} (P95)."
        + (f" <strong>{_mc_top['variable']}</strong> is the top driver (swing = {_mc_top['swing']:.4f})." if _mc_top else "")
        + (f" It contributes ~{_mc_top_pct:.0f}% of output variance." if _mc_top_pct > 1 else "")
        + _mc_spec_note,
        next_steps="Focus improvement on the top driver. Tighening its tolerance will reduce output variation the most.",
        chart_guidance="The tornado chart shows which inputs drive the most output variation (\u00b11\u03c3 impact). The correlation chart shows linear input-output relationships."
    )

    result["statistics"] = {
        "mean": out_mean,
        "std": out_std,
        "percentiles": percentiles,
        "thresholds": threshold_results,
        "sensitivity": sensitivity,
        "correlations": correlations,
    }

    # Ship output values for client-side interactive threshold
    result["simulation_data"] = {
        "output_values": hist_data,
        "output_mean": out_mean,
        "output_std": out_std,
    }

    return result


# =========================================================================
# TOLERANCE STACKUP (RSS + Monte Carlo)
# =========================================================================

def _tolerance_stackup(df, config, result):
    """Multivariate tolerance stackup: worst-case, RSS, and Monte Carlo."""

    dimensions = config.get("dimensions", [])
    n_iter = min(int(config.get("n_iterations", 50000)), 100000)
    spec_limit = config.get("spec_limit")  # optional assembly spec
    seed = config.get("seed")

    if not dimensions or len(dimensions) < 2:
        result["summary"] = "Error: At least 2 dimensions required for tolerance stackup."
        return result
    if len(dimensions) > 30:
        result["summary"] = "Error: Maximum 30 dimensions supported."
        return result

    rng = np.random.default_rng(int(seed) if seed else None)

    names = []
    nominals = []
    tolerances = []
    mc_samples = []

    for dim in dimensions:
        name = dim.get("name", "D")
        nom = float(dim.get("nominal", 0))
        tol = abs(float(dim.get("tolerance", 0.1)))
        dist = dim.get("distribution", "normal")  # normal, uniform, triangular
        sign = float(dim.get("sign", 1))  # +1 or -1 for assembly direction

        names.append(name)
        nominals.append(nom * sign)
        tolerances.append(tol)

        # Generate samples based on distribution assumption
        if dist == "uniform":
            samp = rng.uniform(nom - tol, nom + tol, n_iter) * sign
        elif dist == "triangular":
            samp = rng.triangular(nom - tol, nom, nom + tol, n_iter) * sign
        else:
            # Normal: tolerance = 3sigma (99.73% within tolerance)
            sigma = tol / 3.0
            samp = rng.normal(nom, max(sigma, 1e-12), n_iter) * sign

        mc_samples.append(samp)

    # --- Worst Case ---
    wc_nominal = sum(nominals)
    wc_tolerance = sum(tolerances)  # arithmetic sum of all tolerances

    # --- RSS (Root Sum of Squares) ---
    rss_tolerance = float(np.sqrt(sum(t ** 2 for t in tolerances)))

    # --- Monte Carlo ---
    mc_assembly = np.sum(mc_samples, axis=0)
    mc_mean = float(np.mean(mc_assembly))
    mc_std = float(np.std(mc_assembly))
    mc_p5 = float(np.percentile(mc_assembly, 5))
    mc_p95 = float(np.percentile(mc_assembly, 95))
    mc_min = float(np.min(mc_assembly))
    mc_max = float(np.max(mc_assembly))

    # Contribution percentages (variance-based)
    total_var = float(np.var(mc_assembly))
    contributions = []
    for i, name in enumerate(names):
        var_i = float(np.var(mc_samples[i]))
        pct = (var_i / total_var * 100) if total_var > 0 else 0
        contributions.append({"name": name, "variance": var_i, "pct": pct, "tolerance": tolerances[i]})
    contributions.sort(key=lambda x: x["pct"], reverse=True)

    # Spec limit analysis
    spec_note = ""
    spec_pct = None
    if spec_limit is not None:
        spec_val = float(spec_limit)
        spec_pct = float(np.mean(np.abs(mc_assembly - wc_nominal) > spec_val) * 100)
        spec_note = f" {spec_pct:.2f}% of assemblies exceed the spec limit of {spec_val}."

    # --- Plots ---

    # 1. Assembly distribution histogram
    hist_data = mc_assembly.tolist() if len(mc_assembly) <= 10000 else mc_assembly[:10000].tolist()
    shapes = []
    # WC limits
    shapes.append({"type": "line", "x0": wc_nominal - wc_tolerance, "x1": wc_nominal - wc_tolerance,
                    "y0": 0, "y1": 1, "yref": "paper",
                    "line": {"color": "#dc5050", "width": 2, "dash": "dash"}})
    shapes.append({"type": "line", "x0": wc_nominal + wc_tolerance, "x1": wc_nominal + wc_tolerance,
                    "y0": 0, "y1": 1, "yref": "paper",
                    "line": {"color": "#dc5050", "width": 2, "dash": "dash"}})
    # RSS limits
    shapes.append({"type": "line", "x0": wc_nominal - rss_tolerance, "x1": wc_nominal - rss_tolerance,
                    "y0": 0, "y1": 1, "yref": "paper",
                    "line": {"color": "#4a90d9", "width": 2, "dash": "dot"}})
    shapes.append({"type": "line", "x0": wc_nominal + rss_tolerance, "x1": wc_nominal + rss_tolerance,
                    "y0": 0, "y1": 1, "yref": "paper",
                    "line": {"color": "#4a90d9", "width": 2, "dash": "dot"}})

    result["plots"].append({
        "title": f"Assembly Distribution ({n_iter:,} simulations)",
        "data": [{
            "type": "histogram", "x": hist_data,
            "marker": {"color": "rgba(74, 159, 110, 0.5)", "line": {"color": "#4a9f6e", "width": 1}},
            "name": "Assembly",
        }],
        "layout": {
            "height": 320,
            "xaxis": {"title": "Assembly Value"},
            "yaxis": {"title": "Frequency"},
            "shapes": shapes,
            "annotations": [
                {"x": wc_nominal - wc_tolerance, "y": 1, "yref": "paper", "text": "WC", "showarrow": False,
                 "font": {"color": "#dc5050", "size": 10}, "yanchor": "bottom"},
                {"x": wc_nominal - rss_tolerance, "y": 0.95, "yref": "paper", "text": "RSS", "showarrow": False,
                 "font": {"color": "#4a90d9", "size": 10}, "yanchor": "bottom"},
            ],
        },
    })

    # 2. Contribution tornado
    result["plots"].append({
        "title": "Variance Contribution by Dimension",
        "data": [{
            "type": "bar",
            "y": [c["name"] for c in contributions],
            "x": [c["pct"] for c in contributions],
            "orientation": "h",
            "marker": {"color": "#4a9f6e"},
            "text": [f"{c['pct']:.1f}%" for c in contributions],
            "textposition": "auto",
        }],
        "layout": {
            "height": max(200, len(contributions) * 30 + 80),
            "xaxis": {"title": "% of Total Variance"},
            "yaxis": {"autorange": "reversed"},
        },
    })

    # 3. Method comparison bar chart
    result["plots"].append({
        "title": "Tolerance Band Comparison",
        "data": [{
            "type": "bar",
            "x": ["Worst Case", "RSS", "Monte Carlo (P5-P95)"],
            "y": [wc_tolerance * 2, rss_tolerance * 2, mc_p95 - mc_p5],
            "marker": {"color": ["#dc5050", "#4a90d9", "#4a9f6e"]},
            "text": [f"\u00b1{wc_tolerance:.4f}", f"\u00b1{rss_tolerance:.4f}", f"{mc_p95 - mc_p5:.4f}"],
            "textposition": "auto",
        }],
        "layout": {
            "height": 250,
            "yaxis": {"title": "Total Tolerance Band"},
        },
    })

    # --- Summary ---
    _eq = "=" * 70
    _dash = "-" * 40
    summary = f"<<COLOR:accent>>{_eq}<</COLOR>>\n"
    summary += "<<COLOR:title>>TOLERANCE STACKUP ANALYSIS<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{_eq}<</COLOR>>\n\n"
    summary += f"<<COLOR:text>>Dimensions:<</COLOR>> {len(names)}\n"
    summary += f"<<COLOR:text>>Assembly nominal:<</COLOR>> {wc_nominal:.4f}\n\n"

    summary += f"<<COLOR:highlight>>Method Comparison:<</COLOR>>\n"
    summary += f"  {'Method':<20} {'Tolerance':>12} {'Band':>14}\n"
    summary += f"  {_dash}\n"
    summary += f"  {'Worst Case':<20} {wc_tolerance:>12.4f} [{wc_nominal - wc_tolerance:.4f}, {wc_nominal + wc_tolerance:.4f}]\n"
    summary += f"  {'RSS':<20} {rss_tolerance:>12.4f} [{wc_nominal - rss_tolerance:.4f}, {wc_nominal + rss_tolerance:.4f}]\n"
    summary += f"  {'Monte Carlo (90%)':<20} {(mc_p95 - mc_p5)/2:>12.4f} [{mc_p5:.4f}, {mc_p95:.4f}]\n"

    summary += f"\n<<COLOR:highlight>>Variance Contributors:<</COLOR>>\n"
    for c in contributions:
        bar = "#" * int(c["pct"] / 2)
        summary += f"  {c['name']:<15} {c['pct']:>6.1f}%  {bar}\n"

    result["summary"] = summary
    result["guide_observation"] = f"Tolerance stackup: {len(names)} dims, WC=\u00b1{wc_tolerance:.4f}, RSS=\u00b1{rss_tolerance:.4f}, MC std={mc_std:.4f}. Top contributor: {contributions[0]['name']} ({contributions[0]['pct']:.0f}%)."

    rss_ratio = rss_tolerance / wc_tolerance if wc_tolerance > 0 else 0
    result["narrative"] = _narrative(
        f"Assembly: {wc_nominal:.4f} \u00b1 {rss_tolerance:.4f} (RSS) vs \u00b1 {wc_tolerance:.4f} (worst case)",
        f"RSS tolerance is {rss_ratio:.0%} of worst-case \u2014 the statistical approach yields a {'much ' if rss_ratio < 0.6 else ''}tighter band. "
        f"<strong>{contributions[0]['name']}</strong> drives {contributions[0]['pct']:.0f}% of assembly variation."
        + spec_note,
        next_steps="Tighten tolerances on the top contributor for maximum impact. Consider switching from worst-case to RSS if assembly rejects are rare.",
        chart_guidance="Red dashed = worst-case limits. Blue dotted = RSS limits. The tornado chart shows which dimensions drive the most variation."
    )

    result["statistics"] = {
        "nominal": wc_nominal,
        "wc_tolerance": wc_tolerance,
        "rss_tolerance": rss_tolerance,
        "mc_mean": mc_mean, "mc_std": mc_std,
        "mc_p5": mc_p5, "mc_p95": mc_p95,
        "mc_min": mc_min, "mc_max": mc_max,
        "contributions": contributions,
        "spec_pct_oor": spec_pct,
    }

    return result


# =========================================================================
# VARIANCE PROPAGATION (Analytical Delta Method)
# =========================================================================

def _variance_propagation(df, config, result):
    """Analytical variance propagation via Taylor series (delta method)."""

    variables = config.get("variables", [])
    formula = config.get("transfer_function", "")
    n_verify = min(int(config.get("n_verify", 10000)), 100000)  # MC verification runs
    seed = config.get("seed")

    if not variables:
        result["summary"] = "Error: No input variables defined."
        return result
    if len(variables) > 20:
        result["summary"] = "Error: Maximum 20 variables supported."
        return result
    if not formula:
        result["summary"] = "Error: Transfer function formula required."
        return result

    rng = np.random.default_rng(int(seed) if seed else None)

    # Parse and validate formula (same AST check as monte_carlo)
    var_names = [v.get("name", "X") for v in variables]
    allowed_names = set(var_names) | {"sqrt", "log", "exp", "abs", "sin", "cos", "tan",
                                      "pi", "max", "min", "pow", "e", "ceil", "floor"}
    try:
        tree = ast.parse(formula, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in allowed_names:
                result["summary"] = f"Error: Forbidden name '{node.id}' in formula."
                return result
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                result["summary"] = "Error: Import statements not allowed."
                return result
            if isinstance(node, ast.Attribute):
                result["summary"] = "Error: Attribute access not allowed."
                return result
    except SyntaxError as e:
        result["summary"] = f"Error: Invalid formula syntax \u2014 {e}"
        return result

    safe_funcs = {
        "sqrt": np.sqrt, "log": np.log, "exp": np.exp, "abs": np.abs,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "pi": np.pi, "e": np.e, "max": np.maximum, "min": np.minimum,
        "pow": np.power, "ceil": np.ceil, "floor": np.floor,
    }

    # Extract means and variances
    means = {}
    variances = {}
    for v in variables:
        name = v.get("name", "X")
        dist = v.get("distribution", "normal")
        params = v.get("params", {})

        if dist == "normal":
            means[name] = float(params.get("mean", 0))
            variances[name] = float(params.get("std", 1)) ** 2
        elif dist == "uniform":
            lo, hi = float(params.get("low", 0)), float(params.get("high", 1))
            means[name] = (lo + hi) / 2
            variances[name] = (hi - lo) ** 2 / 12
        elif dist == "triangular":
            lo = float(params.get("low", 0))
            mode = float(params.get("mode", 0.5))
            hi = float(params.get("high", 1))
            means[name] = (lo + mode + hi) / 3
            variances[name] = (lo ** 2 + mode ** 2 + hi ** 2 - lo * mode - lo * hi - mode * hi) / 18
        elif dist == "exponential":
            scale = float(params.get("scale", 1))
            means[name] = scale
            variances[name] = scale ** 2
        elif dist == "lognormal":
            mu = float(params.get("mean", 0))
            sig = float(params.get("sigma", 1))
            means[name] = math.exp(mu + sig ** 2 / 2)
            variances[name] = (math.exp(sig ** 2) - 1) * math.exp(2 * mu + sig ** 2)
        else:
            means[name] = float(params.get("mean", 0))
            variances[name] = float(params.get("std", 1)) ** 2

    # Evaluate f at the mean point
    safe_ns = {"__builtins__": {}}
    safe_ns.update(safe_funcs)
    safe_ns.update(means)
    try:
        compiled = compile(tree, "<formula>", "eval")
        f_mean = float(eval(compiled, dict(safe_ns)))
    except Exception as e:
        result["summary"] = f"Error evaluating formula at mean point: {e}"
        return result

    # Numerical partial derivatives (central difference)
    partials = {}
    h_scale = 1e-5
    for name in var_names:
        h = max(abs(means[name]) * h_scale, 1e-10)
        ns_plus = dict(safe_ns)
        ns_plus[name] = means[name] + h
        ns_minus = dict(safe_ns)
        ns_minus[name] = means[name] - h
        try:
            f_plus = float(eval(compiled, ns_plus))
            f_minus = float(eval(compiled, ns_minus))
            partials[name] = (f_plus - f_minus) / (2 * h)
        except Exception:
            partials[name] = 0.0

    # Propagated variance: Var(Y) ~ sum (df/dxi)^2 * Var(xi)
    var_contributions = {}
    total_prop_var = 0.0
    for name in var_names:
        contrib = partials[name] ** 2 * variances[name]
        var_contributions[name] = contrib
        total_prop_var += contrib

    prop_std = math.sqrt(total_prop_var) if total_prop_var > 0 else 0

    # Contribution percentages
    contrib_list = []
    for name in var_names:
        pct = (var_contributions[name] / total_prop_var * 100) if total_prop_var > 0 else 0
        contrib_list.append({
            "name": name, "partial": partials[name],
            "variance_contrib": var_contributions[name], "pct": pct,
            "input_mean": means[name], "input_std": math.sqrt(variances[name]),
        })
    contrib_list.sort(key=lambda x: x["pct"], reverse=True)

    # Monte Carlo verification
    mc_samples = {}
    for v in variables:
        name = v.get("name", "X")
        mc_samples[name] = rng.normal(means[name], math.sqrt(variances[name]), n_verify)

    mc_ns = dict(safe_ns)
    mc_ns.update(mc_samples)
    try:
        mc_outputs = np.asarray(eval(compiled, mc_ns), dtype=float).ravel()
        mc_mean = float(np.mean(mc_outputs))
        mc_std = float(np.std(mc_outputs))
    except Exception:
        mc_mean = f_mean
        mc_std = prop_std

    # --- Plots ---

    # 1. Contribution tornado
    result["plots"].append({
        "title": "Variance Contribution by Input",
        "data": [{
            "type": "bar",
            "y": [c["name"] for c in contrib_list],
            "x": [c["pct"] for c in contrib_list],
            "orientation": "h",
            "marker": {"color": "#4a9f6e"},
            "text": [f"{c['pct']:.1f}%" for c in contrib_list],
            "textposition": "auto",
        }],
        "layout": {
            "height": max(200, len(contrib_list) * 30 + 80),
            "xaxis": {"title": "% of Output Variance"},
            "yaxis": {"autorange": "reversed"},
        },
    })

    # 2. Sensitivity coefficients (partial derivatives)
    result["plots"].append({
        "title": "Sensitivity Coefficients (\u2202f/\u2202x)",
        "data": [{
            "type": "bar",
            "x": [c["name"] for c in contrib_list],
            "y": [c["partial"] for c in contrib_list],
            "marker": {"color": ["#4a9f6e" if c["partial"] >= 0 else "#dc5050" for c in contrib_list]},
        }],
        "layout": {
            "height": 250,
            "yaxis": {"title": "Partial Derivative"},
        },
    })

    # 3. Analytical vs MC comparison
    result["plots"].append({
        "title": "Analytical vs Monte Carlo Verification",
        "data": [
            {"type": "bar", "x": ["Analytical", "Monte Carlo"], "y": [f_mean, mc_mean],
             "name": "Mean", "marker": {"color": "#4a90d9"}, "width": 0.3, "offset": -0.17},
            {"type": "bar", "x": ["Analytical", "Monte Carlo"], "y": [prop_std, mc_std],
             "name": "Std Dev", "marker": {"color": "#dc5050"}, "width": 0.3, "offset": 0.17},
        ],
        "layout": {
            "height": 280,
            "barmode": "group",
            "yaxis": {"title": "Value"},
        },
    })

    # --- Summary ---
    _eq = "=" * 70
    _dash = "-" * 50
    summary = f"<<COLOR:accent>>{_eq}<</COLOR>>\n"
    summary += "<<COLOR:title>>VARIANCE PROPAGATION (DELTA METHOD)<</COLOR>>\n"
    summary += f"<<COLOR:accent>>{_eq}<</COLOR>>\n\n"
    summary += f"<<COLOR:text>>Formula:<</COLOR>> {formula}\n"
    summary += f"<<COLOR:text>>Variables:<</COLOR>> {len(var_names)}\n\n"

    summary += "<<COLOR:highlight>>Output at Mean Point:<</COLOR>>\n"
    summary += f"  f(mean) = {f_mean:.6f}\n"
    summary += f"  Propagated Std Dev = {prop_std:.6f}\n"
    summary += f"  Approx 95% interval: [{f_mean - 1.96 * prop_std:.6f}, {f_mean + 1.96 * prop_std:.6f}]\n\n"

    summary += f"<<COLOR:highlight>>MC Verification ({n_verify:,} runs):<</COLOR>>\n"
    summary += f"  MC Mean = {mc_mean:.6f}  (analytical: {f_mean:.6f})\n"
    summary += f"  MC Std  = {mc_std:.6f}  (analytical: {prop_std:.6f})\n"
    pct_err = abs(mc_std - prop_std) / mc_std * 100 if mc_std > 0 else 0
    summary += f"  Std Dev error: {pct_err:.1f}%\n\n"

    summary += "<<COLOR:highlight>>Sensitivity & Contribution:<</COLOR>>\n"
    summary += f"  {'Variable':<15} {'df/dx':>10} {'Var Contrib':>12} {'%':>8}\n"
    summary += f"  {_dash}\n"
    for c in contrib_list:
        summary += f"  {c['name']:<15} {c['partial']:>10.4f} {c['variance_contrib']:>12.6f} {c['pct']:>7.1f}%\n"

    result["summary"] = summary
    result["guide_observation"] = f"Variance propagation: output = {f_mean:.4f} \u00b1 {prop_std:.4f}. Top driver: {contrib_list[0]['name']} ({contrib_list[0]['pct']:.0f}%). Delta method vs MC error: {pct_err:.1f}%."

    accuracy = "excellent" if pct_err < 2 else ("good" if pct_err < 10 else "poor (consider full Monte Carlo)")
    result["narrative"] = _narrative(
        f"Output: {f_mean:.4f} \u00b1 {prop_std:.4f} (analytical \u03c3)",
        f"The delta method propagates input uncertainties through the transfer function. "
        f"<strong>{contrib_list[0]['name']}</strong> contributes {contrib_list[0]['pct']:.0f}% of output variance "
        f"(sensitivity = {contrib_list[0]['partial']:.4f}). "
        f"Agreement with Monte Carlo is {accuracy} ({pct_err:.1f}% std dev error).",
        next_steps="If delta method error > 10%, use full Monte Carlo. For linear/near-linear functions, analytical propagation is faster and exact.",
        chart_guidance="The tornado chart shows variance contribution. Sensitivity coefficients show how much the output changes per unit change in each input."
    )

    result["statistics"] = {
        "f_mean": f_mean, "prop_std": prop_std,
        "mc_mean": mc_mean, "mc_std": mc_std,
        "pct_error": pct_err,
        "contributions": contrib_list,
    }

    return result

"""DSW Monte Carlo Simulation engine."""

import numpy as np
from scipy import stats as sp_stats
import ast
import math

from .common import _fit_best_distribution


def run_simulation(df, analysis_id, config, user):
    """Monte Carlo simulation engine."""

    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id != "monte_carlo":
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
            "template": "plotly_dark", "height": 320,
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
                "template": "plotly_dark", "height": max(200, len(labels) * 30 + 80),
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
                "template": "plotly_dark", "height": 250,
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

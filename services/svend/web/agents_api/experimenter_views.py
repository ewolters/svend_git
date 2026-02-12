"""Experimenter Agent API views.

Design of Experiments (DOE) and Power Analysis endpoints.
Professional DOE tool - Minitab-class functionality.

Features:
- All major design types (Full Factorial, Fractional, CCD, Box-Behnken, Plackett-Burman, Taguchi, DSD)
- Full ANOVA analysis with interaction effects
- Response surface modeling
- Main effects and interaction plots
- Pareto chart of effects
- Response optimization
- Residual diagnostics
"""

import json
import logging
import sys
from datetime import datetime

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated_paid, require_auth
from .models import Problem

logger = logging.getLogger(__name__)

# Add experimenter agent to path
sys.path.insert(0, "/home/eric/kjerne/services/svend/agents/agents")

from experimenter.agent import ExperimenterAgent, ExperimentRequest
from experimenter.doe import DOEGenerator, Factor
from experimenter.stats import PowerAnalyzer, interpret_effect_size


@csrf_exempt
@require_http_methods(["POST"])
@gated_paid
def power_analysis(request):
    """
    DEPRECATED: Use DSW power calculators instead (9 types via /api/dsw/analysis/).
    This endpoint is kept for backwards compatibility with experimenter.html.

    Calculate statistical power and sample size.

    POST body:
    {
        "effect_size": 0.5,  // Cohen's d or equivalent
        "test_type": "ttest_ind",  // ttest_ind, ttest_paired, anova, correlation, chi_square
        "alpha": 0.05,
        "power": 0.80,
        "groups": 2  // for ANOVA
    }
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    effect_size = data.get("effect_size", 0.5)
    test_type = data.get("test_type", "ttest_ind")
    alpha = data.get("alpha", 0.05)
    power = data.get("power", 0.80)
    groups = data.get("groups", 2)

    try:
        analyzer = PowerAnalyzer()

        if test_type == "ttest_ind":
            result = analyzer.power_ttest_ind(effect_size, alpha=alpha, power=power)
        elif test_type == "ttest_paired":
            result = analyzer.power_ttest_paired(effect_size, alpha=alpha, power=power)
        elif test_type == "anova":
            result = analyzer.power_anova(effect_size, groups=groups, alpha=alpha, power=power)
        elif test_type == "correlation":
            result = analyzer.power_correlation(effect_size, alpha=alpha, power=power)
        elif test_type == "chi_square":
            result = analyzer.power_chi_square(effect_size, df=(groups - 1), alpha=alpha, power=power)
        else:
            result = analyzer.power_ttest_ind(effect_size, alpha=alpha, power=power)

        effect_interp = interpret_effect_size(effect_size)

        summary_text = (
            f"To detect a {effect_interp} effect (d={effect_size}) with "
            f"{power:.0%} power at α={alpha}, you need {result.sample_size} total participants"
            + (f" ({result.sample_size_per_group} per group)" if result.sample_size_per_group else "") + "."
        )

        response_data = {
            "success": True,
            "power_analysis": result.to_dict(),
            "interpretation": {
                "effect_size_meaning": effect_interp,
                "summary": summary_text,
            },
        }

        # Link to problem as evidence (if problem_id provided)
        problem_id = data.get("problem_id")
        if problem_id:
            try:
                from .problem_views import write_context_file
                problem = Problem.objects.get(id=problem_id, user=request.user)
                evidence = problem.add_evidence(
                    summary=f"Power analysis ({test_type}): {summary_text}",
                    evidence_type="calculation",
                    source="Experimenter (Power Analysis)",
                )
                write_context_file(problem)
                response_data["problem_updated"] = True
                response_data["evidence_id"] = evidence["id"]
            except Problem.DoesNotExist:
                pass

        return JsonResponse(response_data)

    except Exception as e:
        logger.exception("Power analysis failed")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@gated_paid
def design_experiment(request):
    """
    Generate an experimental design.

    POST body:
    {
        "factors": [
            {"name": "Temperature", "levels": [100, 150], "units": "°C"},
            {"name": "Pressure", "levels": [1, 2], "units": "atm"},
            {"name": "Catalyst", "levels": ["A", "B"], "categorical": true}
        ],
        "design_type": "full_factorial",
        "replicates": 1,
        "center_points": 0,
        "seed": 42
    }

    Supported design types:
    - full_factorial: All combinations (2^k or general)
    - fractional_factorial: Subset for many factors (Resolution IV)
    - ccd / response_surface: Central Composite Design for optimization
    - box_behnken: RSM with no corner points
    - plackett_burman: Screening design (Resolution III)
    - taguchi: Orthogonal array (L4, L8, L9, L12, L16)
    - definitive_screening: Modern screening (DSD)
    - latin_square: Block two sources of variation
    - rcbd: Randomized Complete Block Design
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    factors_data = data.get("factors", [])
    design_type = data.get("design_type", "full_factorial")
    replicates = data.get("replicates", 1)
    center_points = data.get("center_points", 0)
    seed = data.get("seed")
    resolution = data.get("resolution", 4)
    taguchi_array = data.get("taguchi_array", "auto")

    if not factors_data:
        return JsonResponse({"error": "At least one factor is required"}, status=400)

    try:
        # Convert to Factor objects
        factors = [
            Factor(
                name=f["name"],
                levels=f["levels"],
                units=f.get("units", ""),
                is_categorical=f.get("categorical", False),
            )
            for f in factors_data
        ]

        generator = DOEGenerator(seed=seed)

        if design_type == "full_factorial":
            design = generator.full_factorial(factors, replicates=replicates, center_points=center_points)

        elif design_type == "fractional_factorial":
            design = generator.fractional_factorial(factors, resolution=resolution)

        elif design_type in ["ccd", "response_surface", "central_composite"]:
            design = generator.central_composite(factors, center_points=center_points or 5)

        elif design_type == "box_behnken":
            if len(factors) < 3:
                return JsonResponse({"error": "Box-Behnken requires at least 3 factors"}, status=400)
            design = generator.box_behnken(factors, center_points=center_points or 3)

        elif design_type == "plackett_burman":
            design = generator.plackett_burman(factors)

        elif design_type == "taguchi":
            design = generator.taguchi(factors, array_type=taguchi_array)

        elif design_type in ["definitive_screening", "dsd"]:
            if len(factors) < 3:
                return JsonResponse({"error": "Definitive Screening requires at least 3 factors"}, status=400)
            design = generator.definitive_screening(factors)

        elif design_type == "latin_square":
            if len(factors) == 1:
                design = generator.latin_square(factors[0].levels)
            else:
                design = generator.full_factorial(factors)

        elif design_type in ["rcbd", "randomized_block"]:
            treatments = factors[0].levels if factors else ["A", "B", "C"]
            blocks = len(factors[1].levels) if len(factors) > 1 else 4
            design = generator.randomized_block(treatments, blocks)

        elif design_type == "d_optimal":
            num_runs = data.get("num_runs", max(len(factors) * 2 + 1, 12))
            model = data.get("model", "linear")
            design = generator.d_optimal(factors, num_runs=num_runs, model=model, seed=seed)

        elif design_type == "i_optimal":
            num_runs = data.get("num_runs", max(len(factors) * 2 + 1, 12))
            model = data.get("model", "quadratic")
            design = generator.i_optimal(factors, num_runs=num_runs, model=model, seed=seed)

        else:
            design = generator.full_factorial(factors)

        # Calculate alias structure for fractional designs
        alias_structure = None
        if design.resolution and design.resolution < len(factors):
            alias_structure = _calculate_alias_structure(factors, design.resolution)

        response = {
            "success": True,
            "design": design.to_dict(),
            "markdown": design.to_markdown(),
        }

        if alias_structure:
            response["alias_structure"] = alias_structure

        # Link to problem as evidence (if problem_id provided)
        problem_id = data.get("problem_id")
        if problem_id:
            try:
                from .problem_views import write_context_file
                problem = Problem.objects.get(id=problem_id, user=request.user)
                evidence = problem.add_evidence(
                    summary=f"Generated {design_type} design: {design.num_runs} runs, {len(factors)} factors",
                    evidence_type="experiment",
                    source="Experimenter (Design)",
                )
                write_context_file(problem)
                response["problem_updated"] = True
                response["evidence_id"] = evidence["id"]
            except Problem.DoesNotExist:
                pass

        return JsonResponse(response)

    except Exception as e:
        logger.exception("Design generation failed")
        return JsonResponse({"error": str(e)}, status=500)


def _calculate_alias_structure(factors: list, resolution: int) -> dict:
    """Calculate alias/confounding structure for fractional designs."""
    factor_names = [f.name for f in factors]
    aliases = []

    # Resolution III: Main effects aliased with 2FIs
    if resolution == 3:
        from itertools import combinations
        for i, name in enumerate(factor_names):
            two_fi = [f"{factor_names[j]}×{factor_names[k]}"
                     for j, k in combinations(range(len(factor_names)), 2)
                     if j != i and k != i]
            if two_fi:
                aliases.append({
                    "effect": name,
                    "aliased_with": two_fi[:3],  # Show first 3
                    "type": "main"
                })

    # Resolution IV: 2FIs aliased with each other
    elif resolution == 4:
        from itertools import combinations
        two_fis = list(combinations(factor_names, 2))
        for i, (a, b) in enumerate(two_fis[:5]):
            aliases.append({
                "effect": f"{a}×{b}",
                "aliased_with": [f"{c}×{d}" for c, d in two_fis[i+1:i+3]],
                "type": "2fi"
            })

    return {
        "resolution": resolution,
        "interpretation": {
            3: "Main effects confounded with 2-factor interactions",
            4: "Main effects clear; 2FIs confounded with each other",
            5: "Main effects and 2FIs all estimable",
        }.get(resolution, "Check specific aliasing pattern"),
        "aliases": aliases
    }


@csrf_exempt
@require_http_methods(["POST"])
@gated_paid
def full_experiment(request):
    """
    Full experiment design with power analysis and DOE.

    POST body:
    {
        "goal": "Optimize widget production",
        "factors": [...],
        "design_type": "full_factorial",
        "effect_size": 0.5,
        "alpha": 0.05,
        "power": 0.80,
        "problem_id": "uuid"  // optional, link to problem session
    }
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    try:
        # Build experiment request
        exp_request = ExperimentRequest(
            goal=data.get("goal", "Experiment"),
            request_type="full",
            test_type=data.get("test_type", "anova"),
            effect_size=data.get("effect_size", 0.5),
            alpha=data.get("alpha", 0.05),
            power=data.get("power", 0.80),
            groups=len(data.get("factors", [])) or 2,
            factors=data.get("factors", []),
            design_type=data.get("design_type", "full_factorial"),
            include_plots=False,
            seed=data.get("seed"),
        )

        agent = ExperimenterAgent(seed=data.get("seed"))
        result = agent.design_experiment(exp_request)

        response_data = {
            "success": True,
            "result": result.to_dict(),
            "markdown": result.to_markdown(),
        }

        # If linked to a problem, add experiment as evidence
        problem_id = data.get("problem_id")
        if problem_id:
            try:
                problem = Problem.objects.get(id=problem_id, user=request.user)

                # Add experiment design as evidence
                from .problem_views import write_context_file

                evidence = problem.add_evidence(
                    summary=f"Designed experiment: {result.design.name if result.design else 'Power Analysis'}",
                    evidence_type="experiment",
                    source="Experimenter Agent",
                )

                # Update recommended next steps
                if result.design:
                    problem.recommended_next_steps = [{
                        "action": "Run experiment",
                        "details": f"Execute {result.design.num_runs} runs in randomized order",
                        "design_id": result.design.name,
                    }]
                    problem.save(update_fields=["recommended_next_steps"])

                write_context_file(problem)
                response_data["problem_updated"] = True
                response_data["evidence_id"] = evidence["id"]

            except Problem.DoesNotExist:
                pass  # Problem linking is optional

        return JsonResponse(response_data)

    except Exception as e:
        logger.exception("Full experiment design failed")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@gated_paid
def analyze_results(request):
    """
    Professional DOE Analysis - Minitab-class functionality.

    POST body:
    {
        "design": {...},
        "results": [{"run_id": 1, "response": 45.2}, ...],
        "response_name": "Yield",
        "alpha": 0.05,
        "include_interactions": true,
        "fit_quadratic": false,
        "problem_id": "uuid"
    }

    Returns:
    - Full ANOVA table
    - Main effects with confidence intervals
    - Interaction effects
    - Residual diagnostics
    - Plot data (main effects, interactions, Pareto, normal probability)
    - Model equation
    - R-squared and adjusted R-squared
    - Optimization recommendations
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    design_data = data.get("design")
    results_data = data.get("results", [])
    response_name = data.get("response_name", "Response")
    alpha = data.get("alpha", 0.05)
    include_interactions = data.get("include_interactions", True)
    fit_quadratic = data.get("fit_quadratic", False)

    if not design_data or not results_data:
        return JsonResponse({"error": "Design and results are required"}, status=400)

    try:
        import numpy as np
        from scipy import stats
        from itertools import combinations

        # Build data matrix
        factors = design_data.get("factors", [])
        runs = design_data.get("runs", [])
        responses = {r["run_id"]: r["response"] for r in results_data}

        # Construct X matrix (coded values) and Y vector
        X_data = []
        Y_data = []
        run_order = []

        for run in runs:
            run_id = run["run_id"]
            if run_id not in responses:
                continue

            row = [1]  # Intercept
            coded = run.get("coded", {})

            # Main effects
            for f in factors:
                row.append(coded.get(f["name"], 0))

            # Two-factor interactions
            if include_interactions and len(factors) >= 2:
                for i, j in combinations(range(len(factors)), 2):
                    f1, f2 = factors[i]["name"], factors[j]["name"]
                    row.append(coded.get(f1, 0) * coded.get(f2, 0))

            # Quadratic terms (for RSM)
            if fit_quadratic:
                for f in factors:
                    val = coded.get(f["name"], 0)
                    row.append(val ** 2)

            X_data.append(row)
            Y_data.append(responses[run_id])
            run_order.append(run.get("run_order", run_id))

        X = np.array(X_data)
        Y = np.array(Y_data)
        n = len(Y)

        # Build term names
        term_names = ["Constant"]
        for f in factors:
            term_names.append(f["name"])
        if include_interactions and len(factors) >= 2:
            for i, j in combinations(range(len(factors)), 2):
                term_names.append(f"{factors[i]['name']}×{factors[j]['name']}")
        if fit_quadratic:
            for f in factors:
                term_names.append(f"{f['name']}²")

        p = X.shape[1]  # Number of parameters

        # Fit model using least squares
        try:
            coefficients, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            coefficients = np.linalg.pinv(X) @ Y
            residuals = []

        # Predictions and residuals
        Y_pred = X @ coefficients
        residuals = Y - Y_pred

        # Sum of squares
        SS_total = np.sum((Y - np.mean(Y)) ** 2)
        SS_residual = np.sum(residuals ** 2)
        SS_regression = SS_total - SS_residual

        # Degrees of freedom
        df_total = n - 1
        df_regression = p - 1
        df_residual = n - p

        # Mean squares
        MS_regression = SS_regression / df_regression if df_regression > 0 else 0
        MS_residual = SS_residual / df_residual if df_residual > 0 else 0

        # F-statistic and p-value for overall model
        if MS_residual > 0:
            F_stat = MS_regression / MS_residual
            p_value_model = 1 - stats.f.cdf(F_stat, df_regression, df_residual)
        else:
            F_stat = float('inf')
            p_value_model = 0

        # R-squared
        R_squared = 1 - (SS_residual / SS_total) if SS_total > 0 else 0
        R_squared_adj = 1 - ((1 - R_squared) * df_total / df_residual) if df_residual > 0 else 0

        # Standard error of coefficients
        if MS_residual > 0 and df_residual > 0:
            try:
                XtX_inv = np.linalg.inv(X.T @ X)
                se_coefficients = np.sqrt(MS_residual * np.diag(XtX_inv))
            except np.linalg.LinAlgError:
                se_coefficients = np.zeros(p)
        else:
            se_coefficients = np.zeros(p)

        # t-statistics and p-values for each term
        t_stats = []
        p_values = []
        for i, (coef, se) in enumerate(zip(coefficients, se_coefficients)):
            if se > 0:
                t = coef / se
                pval = 2 * (1 - stats.t.cdf(abs(t), df_residual))
            else:
                t = float('inf') if coef != 0 else 0
                pval = 0
            t_stats.append(t)
            p_values.append(pval)

        # Build ANOVA table
        anova_table = {
            "source": ["Regression", "Residual Error", "Total"],
            "df": [df_regression, df_residual, df_total],
            "ss": [round(SS_regression, 4), round(SS_residual, 4), round(SS_total, 4)],
            "ms": [round(MS_regression, 4), round(MS_residual, 4), None],
            "f": [round(F_stat, 4) if F_stat != float('inf') else "∞", None, None],
            "p": [round(p_value_model, 6), None, None],
        }

        # Coefficient table
        coefficient_table = []
        for i, name in enumerate(term_names):
            coef = coefficients[i]
            se = se_coefficients[i]
            t = t_stats[i]
            pval = p_values[i]

            # Effect = 2 * coefficient (for coded -1/+1)
            effect = 2 * coef if i > 0 else coef

            coefficient_table.append({
                "term": name,
                "effect": round(effect, 4) if i > 0 else None,
                "coefficient": round(coef, 4),
                "se_coef": round(se, 4),
                "t_value": round(t, 3) if t != float('inf') else "∞",
                "p_value": round(pval, 6),
                "significant": pval < alpha,
                "vif": None,  # Could calculate if needed
            })

        # Separate main effects and interactions
        main_effects = coefficient_table[1:len(factors)+1]
        interaction_effects = coefficient_table[len(factors)+1:] if include_interactions else []

        # Calculate main effect plot data
        main_effects_plot = []
        for i, f in enumerate(factors):
            factor_name = f["name"]
            levels = f["levels"]

            # Get responses at each level
            level_means = {}
            for level_val in set(levels):
                level_responses = []
                for run in runs:
                    run_id = run["run_id"]
                    if run_id in responses:
                        if run["levels"].get(factor_name) == level_val:
                            level_responses.append(responses[run_id])
                if level_responses:
                    level_means[level_val] = np.mean(level_responses)

            main_effects_plot.append({
                "factor": factor_name,
                "levels": list(level_means.keys()),
                "means": [round(m, 4) for m in level_means.values()],
                "effect": main_effects[i]["effect"] if i < len(main_effects) else None,
            })

        # Interaction plot data (for 2-factor interactions)
        interaction_plots = []
        if include_interactions and len(factors) >= 2:
            for i, j in combinations(range(len(factors)), 2):
                f1, f2 = factors[i], factors[j]
                plot_data = {"factors": [f1["name"], f2["name"]], "data": []}

                for level1 in set(f1["levels"]):
                    series = {"level": level1, "points": []}
                    for level2 in set(f2["levels"]):
                        level_responses = []
                        for run in runs:
                            run_id = run["run_id"]
                            if run_id in responses:
                                if (run["levels"].get(f1["name"]) == level1 and
                                    run["levels"].get(f2["name"]) == level2):
                                    level_responses.append(responses[run_id])
                        if level_responses:
                            series["points"].append({
                                "x": level2,
                                "y": round(np.mean(level_responses), 4)
                            })
                    plot_data["data"].append(series)
                interaction_plots.append(plot_data)

        # Pareto chart data (standardized effects)
        pareto_data = []
        t_critical = stats.t.ppf(1 - alpha/2, df_residual) if df_residual > 0 else 1.96
        for entry in coefficient_table[1:]:  # Skip constant
            pareto_data.append({
                "term": entry["term"],
                "standardized_effect": abs(entry["t_value"]) if entry["t_value"] != "∞" else 100,
                "significant": entry["significant"],
            })
        pareto_data.sort(key=lambda x: x["standardized_effect"], reverse=True)

        # Normal probability plot of residuals
        sorted_residuals = np.sort(residuals)
        n_res = len(sorted_residuals)
        theoretical_quantiles = [stats.norm.ppf((i + 0.5) / n_res) for i in range(n_res)]
        normal_plot = {
            "theoretical": [round(q, 4) for q in theoretical_quantiles],
            "residuals": [round(r, 4) for r in sorted_residuals],
            "anderson_darling": None,
        }

        # Anderson-Darling test for normality
        if n_res >= 8:
            ad_stat, ad_critical, ad_sig = stats.anderson(residuals, dist='norm')
            normal_plot["anderson_darling"] = {
                "statistic": round(ad_stat, 4),
                "critical_5pct": round(ad_critical[2], 4),
                "normal": ad_stat < ad_critical[2],
            }

        # Residuals vs fitted
        residual_vs_fitted = {
            "fitted": [round(f, 4) for f in Y_pred],
            "residuals": [round(r, 4) for r in residuals],
        }

        # Residuals vs order
        residual_vs_order = {
            "order": run_order,
            "residuals": [round(r, 4) for r in residuals],
        }

        # Lack of fit test (if replicates exist)
        lack_of_fit = None
        unique_combinations = {}
        for run in runs:
            run_id = run["run_id"]
            if run_id in responses:
                key = tuple(sorted(run.get("coded", {}).items()))
                if key not in unique_combinations:
                    unique_combinations[key] = []
                unique_combinations[key].append(responses[run_id])

        if any(len(v) > 1 for v in unique_combinations.values()):
            SS_pure_error = sum(
                sum((y - np.mean(ys)) ** 2 for y in ys)
                for ys in unique_combinations.values()
                if len(ys) > 1
            )
            df_pure_error = sum(len(ys) - 1 for ys in unique_combinations.values() if len(ys) > 1)
            SS_lack_of_fit = SS_residual - SS_pure_error
            df_lack_of_fit = df_residual - df_pure_error

            if df_lack_of_fit > 0 and df_pure_error > 0 and SS_pure_error > 0:
                MS_lack_of_fit = SS_lack_of_fit / df_lack_of_fit
                MS_pure_error = SS_pure_error / df_pure_error
                F_lof = MS_lack_of_fit / MS_pure_error
                p_lof = 1 - stats.f.cdf(F_lof, df_lack_of_fit, df_pure_error)

                lack_of_fit = {
                    "ss_lack_of_fit": round(SS_lack_of_fit, 4),
                    "ss_pure_error": round(SS_pure_error, 4),
                    "df_lack_of_fit": df_lack_of_fit,
                    "df_pure_error": df_pure_error,
                    "f_value": round(F_lof, 4),
                    "p_value": round(p_lof, 6),
                    "significant": p_lof < alpha,
                    "interpretation": "Model may be inadequate" if p_lof < alpha else "No significant lack of fit",
                }

        # Model equation
        equation_parts = [f"{coefficients[0]:.4f}"]
        for i, name in enumerate(term_names[1:], 1):
            coef = coefficients[i]
            sign = "+" if coef >= 0 else "-"
            equation_parts.append(f"{sign} {abs(coef):.4f}·{name}")
        model_equation = f"{response_name} = " + " ".join(equation_parts)

        # Optimization (find settings that maximize/minimize response)
        optimization = _find_optimal_settings(factors, coefficients, term_names, include_interactions, fit_quadratic)

        # Overall summary statistics
        overall = {
            "mean": round(np.mean(Y), 4),
            "std": round(np.std(Y, ddof=1), 4),
            "min": round(np.min(Y), 4),
            "max": round(np.max(Y), 4),
            "n": n,
        }

        # Model summary
        model_summary = {
            "s": round(np.sqrt(MS_residual), 4) if MS_residual > 0 else 0,
            "r_squared": round(R_squared * 100, 2),
            "r_squared_adj": round(R_squared_adj * 100, 2),
            "r_squared_pred": None,  # Would need PRESS statistic
        }

        # Significant factors summary
        significant_factors = [
            entry["term"] for entry in coefficient_table[1:]
            if entry["significant"]
        ]

        # Generate interpretation
        interpretation = _generate_interpretation(
            coefficient_table, R_squared, lack_of_fit, response_name, alpha
        )

        response_data = {
            "success": True,
            "analysis": {
                "overall": overall,
                "model_summary": model_summary,
                "anova_table": anova_table,
                "coefficients": coefficient_table,
                "main_effects": main_effects,
                "interaction_effects": interaction_effects,
                "significant_factors": significant_factors,
                "model_equation": model_equation,
            },
            "plots": {
                "main_effects": main_effects_plot,
                "interactions": interaction_plots,
                "pareto": pareto_data,
                "pareto_reference": round(t_critical, 3),
                "normal_probability": normal_plot,
                "residual_vs_fitted": residual_vs_fitted,
                "residual_vs_order": residual_vs_order,
            },
            "diagnostics": {
                "lack_of_fit": lack_of_fit,
                "residual_std": round(np.std(residuals, ddof=1), 4) if len(residuals) > 1 else 0,
                "durbin_watson": round(_durbin_watson(residuals), 4),
            },
            "optimization": optimization,
            "interpretation": interpretation,
        }

        # If linked to a problem, add findings as evidence
        problem_id = data.get("problem_id")
        if problem_id and significant_factors:
            try:
                problem = Problem.objects.get(id=problem_id, user=request.user)
                from .problem_views import write_context_file

                for factor_name in significant_factors[:3]:  # Top 3
                    effect_entry = next((e for e in coefficient_table if e["term"] == factor_name), None)
                    if effect_entry:
                        evidence = problem.add_evidence(
                            summary=f"DOE: {factor_name} significantly affects {response_name} "
                                   f"(p={effect_entry['p_value']:.4f})",
                            evidence_type="experiment",
                            source="DOE Analysis",
                        )

                write_context_file(problem)
                response_data["problem_updated"] = True

            except Problem.DoesNotExist:
                pass

        return JsonResponse(response_data)

    except Exception as e:
        logger.exception("Results analysis failed")
        return JsonResponse({"error": str(e)}, status=500)


def _durbin_watson(residuals) -> float:
    """Calculate Durbin-Watson statistic for autocorrelation."""
    import numpy as np
    diff = np.diff(residuals)
    return np.sum(diff ** 2) / np.sum(residuals ** 2) if np.sum(residuals ** 2) > 0 else 2.0


def _find_optimal_settings(factors, coefficients, term_names, include_interactions, fit_quadratic):
    """Find factor settings that optimize the response."""
    import numpy as np
    from itertools import product

    # Grid search over factor space
    best_max = {"settings": {}, "predicted": float('-inf')}
    best_min = {"settings": {}, "predicted": float('inf')}

    # Generate grid
    grid_points = 5
    factor_grids = []
    for f in factors:
        levels = f["levels"]
        if len(levels) == 2:
            grid = np.linspace(-1, 1, grid_points)
        else:
            grid = np.linspace(-1, 1, grid_points)
        factor_grids.append(grid)

    for combo in product(*factor_grids):
        # Build prediction
        pred = coefficients[0]  # Intercept

        # Main effects
        for i, val in enumerate(combo):
            if i + 1 < len(coefficients):
                pred += coefficients[i + 1] * val

        # Interactions
        if include_interactions:
            idx = len(factors) + 1
            for i in range(len(factors)):
                for j in range(i + 1, len(factors)):
                    if idx < len(coefficients):
                        pred += coefficients[idx] * combo[i] * combo[j]
                        idx += 1

        # Quadratic
        if fit_quadratic:
            for i, val in enumerate(combo):
                idx = len(factors) + 1
                if include_interactions:
                    idx += len(factors) * (len(factors) - 1) // 2
                idx += i
                if idx < len(coefficients):
                    pred += coefficients[idx] * val ** 2

        # Convert coded to actual
        settings = {}
        for i, f in enumerate(factors):
            coded_val = combo[i]
            levels = f["levels"]
            if len(levels) >= 2:
                try:
                    low, high = float(levels[0]), float(levels[1])
                except (ValueError, TypeError):
                    continue
                actual = (low + high) / 2 + coded_val * (high - low) / 2
                settings[f["name"]] = round(actual, 4)

        if pred > best_max["predicted"]:
            best_max = {"settings": settings.copy(), "predicted": round(pred, 4)}
        if pred < best_min["predicted"]:
            best_min = {"settings": settings.copy(), "predicted": round(pred, 4)}

    return {
        "maximize": best_max,
        "minimize": best_min,
        "note": "Based on fitted model within experimental region"
    }


def _generate_interpretation(coefficient_table, R_squared, lack_of_fit, response_name, alpha):
    """Generate human-readable interpretation."""
    interpretation = []

    # Model fit
    if R_squared > 0.9:
        interpretation.append(f"Excellent model fit (R² = {R_squared*100:.1f}%) - the model explains most of the variation in {response_name}.")
    elif R_squared > 0.7:
        interpretation.append(f"Good model fit (R² = {R_squared*100:.1f}%) - the model captures the major effects.")
    elif R_squared > 0.5:
        interpretation.append(f"Moderate model fit (R² = {R_squared*100:.1f}%) - some unexplained variation remains.")
    else:
        interpretation.append(f"Weak model fit (R² = {R_squared*100:.1f}%) - consider additional factors or transformations.")

    # Significant effects
    significant = [e for e in coefficient_table[1:] if e["significant"]]
    if significant:
        sig_names = [e["term"] for e in significant[:3]]
        interpretation.append(f"Significant effects: {', '.join(sig_names)}.")

        # Direction of effects
        for e in significant[:2]:
            if e["effect"] and e["effect"] > 0:
                interpretation.append(f"Increasing {e['term']} increases {response_name}.")
            elif e["effect"] and e["effect"] < 0:
                interpretation.append(f"Increasing {e['term']} decreases {response_name}.")
    else:
        interpretation.append("No factors show statistically significant effects at the current sample size.")

    # Lack of fit
    if lack_of_fit:
        if lack_of_fit["significant"]:
            interpretation.append("⚠️ Lack of fit is significant - the model may be missing important terms (e.g., quadratic or interaction effects).")
        else:
            interpretation.append("No significant lack of fit detected.")

    return interpretation


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def design_types(request):
    """Get available design types with descriptions."""
    return JsonResponse({
        "design_types": [
            # Factorial Designs
            {
                "id": "full_factorial",
                "name": "Full Factorial",
                "category": "factorial",
                "description": "Tests all combinations of factor levels. Best for 2-4 factors.",
                "when_to_use": "When you need to understand all main effects and interactions.",
                "runs": "2^k × replicates + center points",
                "min_factors": 2,
                "max_factors": 6,
                "supports_center_points": True,
            },
            {
                "id": "fractional_factorial",
                "name": "Fractional Factorial",
                "category": "factorial",
                "description": "Subset of full factorial. Reduces runs while estimating main effects.",
                "when_to_use": "When you have 5+ factors and need efficient screening.",
                "runs": "2^(k-p) depending on resolution",
                "min_factors": 5,
                "max_factors": 15,
                "supports_center_points": True,
            },
            # Screening Designs
            {
                "id": "plackett_burman",
                "name": "Plackett-Burman",
                "category": "screening",
                "description": "Highly efficient screening design (Resolution III). Identifies important factors quickly.",
                "when_to_use": "Initial screening of many factors (5-23) to find the vital few.",
                "runs": "N runs (N multiple of 4: 8, 12, 16, 20, 24)",
                "min_factors": 5,
                "max_factors": 23,
                "supports_center_points": False,
            },
            {
                "id": "definitive_screening",
                "name": "Definitive Screening (DSD)",
                "category": "screening",
                "description": "Modern screening design. Main effects orthogonal to 2FIs, can estimate quadratics.",
                "when_to_use": "When you want screening that can also detect curvature.",
                "runs": "2k + 1 runs for k factors",
                "min_factors": 3,
                "max_factors": 12,
                "supports_center_points": False,
            },
            # Response Surface Designs
            {
                "id": "ccd",
                "name": "Central Composite (CCD)",
                "category": "rsm",
                "description": "Response surface design with factorial, axial, and center points. For optimization.",
                "when_to_use": "When you need to find optimal settings and model curvature.",
                "runs": "2^k + 2k + center points",
                "min_factors": 2,
                "max_factors": 6,
                "supports_center_points": True,
            },
            {
                "id": "box_behnken",
                "name": "Box-Behnken",
                "category": "rsm",
                "description": "RSM design without corner points. More economical than CCD for 3+ factors.",
                "when_to_use": "When corner points are expensive or impossible to run.",
                "runs": "Fewer than CCD (no corners)",
                "min_factors": 3,
                "max_factors": 7,
                "supports_center_points": True,
            },
            # Taguchi
            {
                "id": "taguchi",
                "name": "Taguchi Orthogonal Array",
                "category": "robust",
                "description": "Orthogonal arrays for robust parameter design. Standard arrays: L4, L8, L9, L12, L16.",
                "when_to_use": "For robust design and quality engineering (Taguchi method).",
                "runs": "Standard array sizes (4, 8, 9, 12, 16, etc.)",
                "min_factors": 2,
                "max_factors": 15,
                "supports_center_points": False,
            },
            # Blocking Designs
            {
                "id": "rcbd",
                "name": "Randomized Block Design",
                "category": "blocking",
                "description": "Each block contains all treatments, randomized within blocks.",
                "when_to_use": "When you have a nuisance variable (e.g., batches, days, operators).",
                "runs": "treatments × blocks",
                "min_factors": 1,
                "max_factors": 2,
                "supports_center_points": False,
            },
            {
                "id": "latin_square",
                "name": "Latin Square",
                "category": "blocking",
                "description": "Blocks two sources of variation while testing treatments.",
                "when_to_use": "When you have two blocking factors (e.g., row and column effects).",
                "runs": "n squared for n treatments",
                "min_factors": 1,
                "max_factors": 1,
                "supports_center_points": False,
            },
            # Optimal Designs
            {
                "id": "d_optimal",
                "name": "D-Optimal",
                "category": "optimal",
                "description": "Maximizes |X'X| for best parameter estimation. Custom run count.",
                "when_to_use": "When standard designs don't fit, or you need exact run count for efficiency.",
                "runs": "User-specified (min = parameters)",
                "min_factors": 2,
                "max_factors": 10,
                "supports_center_points": False,
            },
            {
                "id": "i_optimal",
                "name": "I-Optimal",
                "category": "optimal",
                "description": "Minimizes average prediction variance. Best for response surface optimization.",
                "when_to_use": "When your goal is prediction accuracy across the design space.",
                "runs": "User-specified (min = parameters)",
                "min_factors": 2,
                "max_factors": 10,
                "supports_center_points": False,
            },
        ],
        "design_categories": [
            {"id": "factorial", "name": "Factorial Designs", "description": "Test factor combinations systematically"},
            {"id": "screening", "name": "Screening Designs", "description": "Efficiently identify important factors"},
            {"id": "rsm", "name": "Response Surface", "description": "Model curvature and optimize"},
            {"id": "robust", "name": "Robust Design", "description": "Quality engineering (Taguchi)"},
            {"id": "blocking", "name": "Blocking Designs", "description": "Control nuisance variation"},
            {"id": "optimal", "name": "Optimal Designs", "description": "Computer-generated efficient designs"},
        ],
        "test_types": [
            {"id": "ttest_ind", "name": "Independent t-test", "description": "Compare means of two independent groups"},
            {"id": "ttest_paired", "name": "Paired t-test", "description": "Compare means of matched/repeated measures"},
            {"id": "anova", "name": "ANOVA", "description": "Compare means across 3+ groups"},
            {"id": "correlation", "name": "Correlation", "description": "Test relationship between two variables"},
            {"id": "chi_square", "name": "Chi-square", "description": "Test association between categorical variables"},
            {"id": "proportion", "name": "Two Proportions", "description": "Compare proportions between groups"},
        ],
    })


@csrf_exempt
@require_http_methods(["POST"])
@gated_paid
def contour_plot(request):
    """
    Generate contour plot data for response surface visualization.

    POST body:
    {
        "design": {...},
        "results": [...],
        "x_factor": "Temperature",
        "y_factor": "Pressure",
        "hold_values": {"Catalyst": 0},  // Coded values for held factors
        "resolution": 25,
        "include_quadratic": true
    }
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    design_data = data.get("design")
    results_data = data.get("results", [])
    x_factor = data.get("x_factor")
    y_factor = data.get("y_factor")
    hold_values = data.get("hold_values", {})
    resolution = data.get("resolution", 25)
    include_quadratic = data.get("include_quadratic", True)

    if not design_data or not results_data:
        return JsonResponse({"error": "Design and results required"}, status=400)

    if not x_factor or not y_factor:
        return JsonResponse({"error": "x_factor and y_factor required"}, status=400)

    try:
        import numpy as np
        from itertools import combinations

        factors = design_data.get("factors", [])
        runs = design_data.get("runs", [])
        responses = {r["run_id"]: r["response"] for r in results_data}

        # Find factor indices
        factor_names = [f["name"] for f in factors]
        x_idx = factor_names.index(x_factor) if x_factor in factor_names else 0
        y_idx = factor_names.index(y_factor) if y_factor in factor_names else 1

        # Build model (same as analyze_results)
        X_data = []
        Y_data = []

        for run in runs:
            run_id = run["run_id"]
            if run_id not in responses:
                continue

            row = [1]  # Intercept
            coded = run.get("coded", {})

            for f in factors:
                row.append(coded.get(f["name"], 0))

            # 2FI
            for i, j in combinations(range(len(factors)), 2):
                f1, f2 = factors[i]["name"], factors[j]["name"]
                row.append(coded.get(f1, 0) * coded.get(f2, 0))

            # Quadratic
            if include_quadratic:
                for f in factors:
                    val = coded.get(f["name"], 0)
                    row.append(val ** 2)

            X_data.append(row)
            Y_data.append(responses[run_id])

        X = np.array(X_data)
        Y = np.array(Y_data)

        # Fit model
        try:
            coefficients = np.linalg.lstsq(X, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(X) @ Y

        # Generate contour grid
        x_grid = np.linspace(-1, 1, resolution)
        y_grid = np.linspace(-1, 1, resolution)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        Z_mesh = np.zeros_like(X_mesh)

        for i in range(resolution):
            for j in range(resolution):
                x_val = X_mesh[i, j]
                y_val = Y_mesh[i, j]

                # Build prediction row
                pred_row = [1]  # Intercept

                # Main effects
                for k, f in enumerate(factors):
                    if k == x_idx:
                        pred_row.append(x_val)
                    elif k == y_idx:
                        pred_row.append(y_val)
                    else:
                        pred_row.append(hold_values.get(f["name"], 0))

                # 2FI
                for fi, fj in combinations(range(len(factors)), 2):
                    val_i = x_val if fi == x_idx else (y_val if fi == y_idx else hold_values.get(factors[fi]["name"], 0))
                    val_j = x_val if fj == x_idx else (y_val if fj == y_idx else hold_values.get(factors[fj]["name"], 0))
                    pred_row.append(val_i * val_j)

                # Quadratic
                if include_quadratic:
                    for k, f in enumerate(factors):
                        if k == x_idx:
                            pred_row.append(x_val ** 2)
                        elif k == y_idx:
                            pred_row.append(y_val ** 2)
                        else:
                            val = hold_values.get(f["name"], 0)
                            pred_row.append(val ** 2)

                # Predict
                if len(pred_row) == len(coefficients):
                    Z_mesh[i, j] = np.dot(pred_row, coefficients)
                else:
                    Z_mesh[i, j] = np.dot(pred_row[:len(coefficients)], coefficients)

        # Convert to actual factor values for labels
        x_factor_data = factors[x_idx]
        y_factor_data = factors[y_idx]
        x_low, x_high = float(x_factor_data["levels"][0]), float(x_factor_data["levels"][1])
        y_low, y_high = float(y_factor_data["levels"][0]), float(y_factor_data["levels"][1])

        x_actual = [x_low + (x_high - x_low) * (v + 1) / 2 for v in x_grid]
        y_actual = [y_low + (y_high - y_low) * (v + 1) / 2 for v in y_grid]

        optimal_x = round(x_actual[int(np.argmax(Z_mesh.max(axis=0)))], 4)
        optimal_y = round(y_actual[int(np.argmax(Z_mesh.max(axis=1)))], 4)
        optimal_z = round(float(Z_mesh.max()), 4)

        response_data = {
            "success": True,
            "contour": {
                "x": [round(v, 4) for v in x_actual],
                "y": [round(v, 4) for v in y_actual],
                "z": [[round(Z_mesh[i, j], 4) for j in range(resolution)] for i in range(resolution)],
                "x_label": x_factor,
                "y_label": y_factor,
                "z_min": round(float(Z_mesh.min()), 4),
                "z_max": round(float(Z_mesh.max()), 4),
            },
            "optimal_point": {
                "x": optimal_x,
                "y": optimal_y,
                "z": optimal_z,
            },
        }

        # Link to problem as evidence (if problem_id provided)
        problem_id = data.get("problem_id")
        if problem_id:
            try:
                from .problem_views import write_context_file
                problem = Problem.objects.get(id=problem_id, user=request.user)
                evidence = problem.add_evidence(
                    summary=f"Response surface: optimal at {x_factor}={optimal_x}, {y_factor}={optimal_y} (predicted={optimal_z})",
                    evidence_type="data_analysis",
                    source="Experimenter (Contour)",
                )
                write_context_file(problem)
                response_data["problem_updated"] = True
                response_data["evidence_id"] = evidence["id"]
            except Problem.DoesNotExist:
                pass

        return JsonResponse(response_data)

    except Exception as e:
        logger.exception("Contour plot generation failed")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@gated_paid
def optimize_response(request):
    """
    Multi-response optimization with desirability functions.

    POST body:
    {
        "design": {...},
        "responses": [
            {"name": "Yield", "results": [...], "goal": "maximize", "lower": 50, "target": 90, "upper": 100, "weight": 1},
            {"name": "Cost", "results": [...], "goal": "minimize", "lower": 10, "target": 10, "upper": 50, "weight": 1}
        ],
        "importance": [1, 1]  // Relative importance of each response
    }
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    design_data = data.get("design")
    responses_config = data.get("responses", [])
    importance = data.get("importance", [1] * len(responses_config))

    if not design_data or not responses_config:
        return JsonResponse({"error": "Design and responses required"}, status=400)

    try:
        import numpy as np
        from itertools import combinations, product

        factors = design_data.get("factors", [])
        runs = design_data.get("runs", [])

        # Fit a model for each response
        models = []
        for resp_config in responses_config:
            results_data = resp_config.get("results", [])
            responses = {r["run_id"]: r["response"] for r in results_data}

            X_data = []
            Y_data = []

            for run in runs:
                run_id = run["run_id"]
                if run_id not in responses:
                    continue

                row = [1]
                coded = run.get("coded", {})
                for f in factors:
                    row.append(coded.get(f["name"], 0))
                for i, j in combinations(range(len(factors)), 2):
                    row.append(coded.get(factors[i]["name"], 0) * coded.get(factors[j]["name"], 0))

                X_data.append(row)
                Y_data.append(responses[run_id])

            X = np.array(X_data)
            Y = np.array(Y_data)
            coefficients = np.linalg.lstsq(X, Y, rcond=None)[0]

            models.append({
                "name": resp_config.get("name", "Response"),
                "coefficients": coefficients,
                "goal": resp_config.get("goal", "maximize"),
                "lower": resp_config.get("lower"),
                "target": resp_config.get("target"),
                "upper": resp_config.get("upper"),
                "weight": resp_config.get("weight", 1),
            })

        # Compute Y ranges for default bounds
        all_Y = np.concatenate([np.array([r["response"] for r in rc.get("results", [])]) for rc in data.get("responses", [])])
        y_min, y_max = float(all_Y.min()), float(all_Y.max())
        y_range = y_max - y_min if y_max > y_min else 1.0

        # Desirability function
        def desirability(value, goal, lower, target, upper, weight=1):
            # Apply sensible defaults for missing bounds
            if goal == "maximize":
                if lower is None: lower = y_min - 0.1 * y_range
                if target is None: target = y_max
                if value <= lower:
                    return 0
                elif value >= target:
                    return 1
                elif target == lower:
                    return 1
                else:
                    return ((value - lower) / (target - lower)) ** weight
            elif goal == "minimize":
                if upper is None: upper = y_max + 0.1 * y_range
                if target is None: target = y_min
                if value >= upper:
                    return 0
                elif value <= target:
                    return 1
                elif upper == target:
                    return 1
                else:
                    return ((upper - value) / (upper - target)) ** weight
            else:  # target
                if lower is None: lower = y_min - 0.1 * y_range
                if upper is None: upper = y_max + 0.1 * y_range
                if target is None: target = (y_min + y_max) / 2
                if value < lower or value > upper:
                    return 0
                elif target == lower:
                    return ((upper - value) / (upper - target)) ** weight if upper != target else 1
                elif value <= target:
                    return ((value - lower) / (target - lower)) ** weight
                else:
                    return ((upper - value) / (upper - target)) ** weight if upper != target else 1

        # Grid search for optimal
        grid_points = 11
        factor_grids = [np.linspace(-1, 1, grid_points) for _ in factors]

        best_composite = 0
        best_settings = {}
        best_predictions = {}

        for combo in product(*factor_grids):
            # Predict each response
            predictions = {}
            desirabilities = []

            for model in models:
                pred_row = [1] + list(combo)
                for i, j in combinations(range(len(factors)), 2):
                    pred_row.append(combo[i] * combo[j])

                pred = np.dot(pred_row[:len(model["coefficients"])], model["coefficients"])
                predictions[model["name"]] = pred

                d = desirability(
                    pred,
                    model["goal"],
                    model.get("lower", pred - 10),
                    model.get("target", pred),
                    model.get("upper", pred + 10),
                    model.get("weight", 1)
                )
                desirabilities.append(d)

            # Composite desirability (geometric mean)
            if all(d > 0 for d in desirabilities):
                weights = [importance[i] if i < len(importance) else 1 for i in range(len(desirabilities))]
                total_weight = sum(weights)
                composite = np.prod([d ** (w / total_weight) for d, w in zip(desirabilities, weights)])
            else:
                composite = 0

            if composite > best_composite:
                best_composite = composite
                best_predictions = predictions.copy()

                # Convert coded to actual
                best_settings = {}
                for i, f in enumerate(factors):
                    levels = f["levels"]
                    try:
                        low, high = float(levels[0]), float(levels[1])
                    except (ValueError, TypeError):
                        continue
                    actual = (low + high) / 2 + combo[i] * (high - low) / 2
                    best_settings[f["name"]] = round(actual, 4)

        response_data = {
            "success": True,
            "optimization": {
                "optimal_settings": best_settings,
                "predicted_responses": {k: round(v, 4) for k, v in best_predictions.items()},
                "composite_desirability": round(best_composite, 4),
                "individual_desirabilities": {
                    model["name"]: round(desirability(
                        best_predictions.get(model["name"], 0),
                        model["goal"],
                        model.get("lower", 0),
                        model.get("target", 0),
                        model.get("upper", 100),
                        model.get("weight", 1)
                    ), 4) for model in models
                },
            },
            "interpretation": _interpret_optimization(best_settings, best_predictions, best_composite),
        }

        # Link to problem as evidence (if problem_id provided)
        problem_id = data.get("problem_id")
        if problem_id:
            try:
                from .problem_views import write_context_file
                problem = Problem.objects.get(id=problem_id, user=request.user)
                settings_str = ", ".join(f"{k}={v}" for k, v in best_settings.items())
                evidence = problem.add_evidence(
                    summary=f"DOE optimization: desirability={best_composite:.2f}, settings: {settings_str}",
                    evidence_type="experiment",
                    source="Experimenter (Optimization)",
                )
                write_context_file(problem)
                response_data["problem_updated"] = True
                response_data["evidence_id"] = evidence["id"]
            except Problem.DoesNotExist:
                pass

        return JsonResponse(response_data)

    except Exception as e:
        logger.exception("Multi-response optimization failed")
        return JsonResponse({"error": str(e)}, status=500)


def _interpret_optimization(settings, predictions, composite):
    """Generate interpretation of optimization results."""
    interpretation = []

    if composite >= 0.9:
        interpretation.append("Excellent optimization - found settings that satisfy all response goals well.")
    elif composite >= 0.7:
        interpretation.append("Good optimization - settings reasonably satisfy most goals.")
    elif composite >= 0.5:
        interpretation.append("Moderate optimization - some trade-offs between responses.")
    else:
        interpretation.append("Difficult optimization - significant conflicts between response goals.")

    interpretation.append(f"Recommended settings: {', '.join(f'{k}={v}' for k, v in settings.items())}")

    for name, pred in predictions.items():
        interpretation.append(f"Predicted {name}: {pred:.2f}")

    return interpretation


# =============================================================================
# DOE Guidance Chat (LLM Integration)
# =============================================================================

DOE_SYSTEM_PROMPT = """You are an expert Design of Experiments (DOE) consultant embedded in a professional DOE workbench tool. Your role is to help users:

1. **Design Selection**: Help users choose the right experimental design based on their goals, number of factors, and constraints:
   - Full Factorial (2^k): When you need all main effects and interactions, small number of factors
   - Fractional Factorial (2^(k-p)): Screening many factors efficiently, accepting some confounding
   - Plackett-Burman: Quick screening of many factors for main effects only
   - Central Composite Design (CCD): Response surface optimization with quadratic terms
   - Box-Behnken: RSM without extreme corner points
   - Taguchi: Robust design focusing on signal-to-noise ratio
   - Definitive Screening Design (DSD): Modern alternative to Plackett-Burman

2. **Factor Definition**: Help define factor levels (low/high), identify potential factors, and determine appropriate ranges.

3. **Analysis Interpretation**: Explain ANOVA tables, p-values, R², main effects, interactions, and residual diagnostics.

4. **Optimization**: Guide response surface analysis, contour plot interpretation, and optimal settings.

5. **Best Practices**: Randomization, blocking, replication, center points, and common pitfalls.

When the user shares their current session context (design, factors, analysis results), provide specific, actionable guidance. Always be practical and focused on helping them achieve valid, actionable experimental results.

Keep responses concise but informative. Use bullet points for clarity. When appropriate, suggest next steps they can take in the workbench."""


@csrf_exempt
@require_http_methods(["POST"])
@gated_paid
def doe_guidance_chat(request):
    """
    DOE Guidance chat endpoint using Qwen LLM.

    POST body:
    {
        "message": "user's question",
        "context": {
            "design": {...},  // Current design if any
            "analysis": {...},  // Current analysis results if any
            "factors": [...],  // Factor definitions
            "panel": "design"  // Current panel user is on
        },
        "history": [  // Previous messages
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    user_message = data.get("message", "").strip()
    if not user_message:
        return JsonResponse({"error": "Message is required"}, status=400)

    context = data.get("context", {})
    history = data.get("history", [])

    # Build context summary for the LLM
    context_parts = []

    if context.get("design"):
        design = context["design"]
        context_parts.append(f"Current Design: {design.get('name', 'Unknown')} with {design.get('properties', {}).get('num_runs', '?')} runs")
        if design.get("factors"):
            factor_str = ", ".join([f"{f['name']} ({len(f.get('levels', []))} levels)" for f in design["factors"]])
            context_parts.append(f"Factors: {factor_str}")

    if context.get("analysis"):
        analysis = context["analysis"]
        if analysis.get("model_summary"):
            ms = analysis["model_summary"]
            context_parts.append(f"Analysis R²: {ms.get('r_squared', '?')}%, R²(adj): {ms.get('r_squared_adj', '?')}%")
        if analysis.get("significant_effects"):
            context_parts.append(f"Significant effects: {', '.join(analysis['significant_effects'])}")

    if context.get("panel"):
        context_parts.append(f"User is currently on: {context['panel']} panel")

    context_summary = "\n".join(context_parts) if context_parts else "No active design or analysis."

    # Build messages for the LLM
    messages = [
        {"role": "system", "content": DOE_SYSTEM_PROMPT},
    ]

    # Add conversation history (limit to last 10 exchanges)
    for msg in history[-20:]:
        messages.append(msg)

    # Add current context and user message
    full_user_message = f"""Current Session Context:
{context_summary}

User Question: {user_message}"""

    messages.append({"role": "user", "content": full_user_message})

    try:
        # Try to get LLM response
        from .llm_manager import LLMManager

        llm = LLMManager.get_shared()

        if llm is None:
            # Fallback to a simple rule-based response
            response_text = _fallback_doe_response(user_message, context)
        else:
            # Use the LLM
            try:
                # Format for Qwen
                prompt = _format_messages_for_qwen(messages)
                response_text = llm.generate(prompt, max_new_tokens=1024, temperature=0.7)

                # Clean up response
                response_text = response_text.strip()
                if response_text.startswith("Assistant:"):
                    response_text = response_text[10:].strip()

            except Exception as e:
                logger.warning(f"LLM generation failed: {e}, using fallback")
                response_text = _fallback_doe_response(user_message, context)

        return JsonResponse({
            "success": True,
            "response": response_text,
            "model": "qwen" if llm else "fallback",
        })

    except Exception as e:
        logger.exception("DOE guidance chat failed")
        return JsonResponse({"error": str(e)}, status=500)


def _format_messages_for_qwen(messages):
    """Format messages for Qwen chat model."""
    prompt_parts = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

    # Add the assistant start token for generation
    prompt_parts.append("<|im_start|>assistant\n")

    return "\n".join(prompt_parts)


def _fallback_doe_response(question, context):
    """Provide basic DOE guidance when LLM is unavailable."""
    question_lower = question.lower()

    # Design selection questions
    if any(kw in question_lower for kw in ["which design", "what design", "choose design", "best design"]):
        factors = context.get("factors", [])
        n_factors = len(factors) if factors else 0

        if "screen" in question_lower or n_factors > 5:
            return """For **screening** many factors (5+), consider:

• **Plackett-Burman**: Efficient screening, main effects only, Resolution III
• **Definitive Screening Design (DSD)**: Modern alternative, can estimate some quadratics
• **Fractional Factorial**: More runs but can estimate some 2-factor interactions

If you have fewer factors and need interactions:
• **Full Factorial**: All effects estimated, requires 2^k runs
• **Fractional Factorial**: Good compromise, Resolution IV or V preferred"""

        elif "optim" in question_lower or "surface" in question_lower:
            return """For **response surface optimization**, consider:

• **Central Composite Design (CCD)**: Most common RSM design, estimates all quadratic terms
• **Box-Behnken**: No corner points (good if extremes are problematic), 3 levels per factor
• **Definitive Screening**: Can transition to RSM if quadratic effects found

CCD is generally preferred for optimization due to:
- Rotatability (equal prediction variance at equal distances from center)
- Ability to estimate all quadratic terms
- Flexibility in axial point distance (α)"""

        else:
            return """Design selection depends on your goals:

**Screening (identify important factors)**:
• Plackett-Burman or DSD for many factors
• Fractional factorial for moderate factors

**Characterization (understand factor effects)**:
• Full factorial for 2-4 factors
• Fractional factorial with high resolution for more

**Optimization (find optimal settings)**:
• CCD or Box-Behnken for response surface
• Often follows a screening study

What are you trying to achieve? How many factors do you have?"""

    # Analysis interpretation
    if any(kw in question_lower for kw in ["p-value", "significant", "interpret", "anova", "r-squared"]):
        analysis = context.get("analysis", {})

        return """**Interpreting DOE Analysis Results:**

**R² (R-squared)**: Proportion of variation explained by the model
• >90%: Excellent fit
• 70-90%: Good fit
• <70%: Consider adding terms or checking for outliers

**P-values**: Test if effects are statistically significant
• p < 0.05: Effect is significant (reject null hypothesis)
• p < 0.01: Highly significant
• p > 0.10: Not significant

**Key checks:**
1. Is the model significant overall? (Model p-value < 0.05)
2. Which main effects are significant?
3. Are any interactions significant?
4. Check residuals for patterns (should be random)

**Red flags:**
• Lack of Fit significant (p < 0.05) → model may be missing terms
• Residuals show patterns → assumption violations
• Very high R² but low adjusted R² → overfitting"""

    # Residuals questions
    if "residual" in question_lower:
        return """**Residual Diagnostics Guide:**

**Normal Probability Plot**: Points should follow diagonal line
• Deviations at tails may indicate outliers or non-normality
• S-shape suggests non-normal distribution

**Residuals vs Fitted**: Should show random scatter
• Funnel shape → non-constant variance (consider transformation)
• Curved pattern → missing quadratic terms

**Residuals vs Order**: Should show random scatter
• Trends suggest time-related effects
• Consider blocking by run order

**Durbin-Watson Statistic**:
• ~2.0: No autocorrelation (good)
• <1.5: Positive autocorrelation
• >2.5: Negative autocorrelation

**Lack of Fit Test** (if replicates available):
• p > 0.05: Model adequately fits data
• p < 0.05: Model may need higher-order terms"""

    # Factor definition
    if any(kw in question_lower for kw in ["factor", "level", "range"]):
        return """**Factor Definition Best Practices:**

**Choosing Levels:**
• Levels should span the region of interest
• Don't make range too narrow (won't detect effects) or too wide (may miss curvature)
• Use process knowledge to set practical limits

**Number of Levels:**
• 2 levels: Sufficient for linear effects and screening
• 3 levels: Needed to detect curvature/quadratic effects
• More levels: Usually not needed, use RSM designs instead

**Center Points:**
• Add center points to detect curvature
• Also provide pure error estimate
• Typically 3-5 center points for factorial designs

**Categorical Factors:**
• Use all categories as levels
• Consider if interactions with continuous factors are important"""

    # General help
    return """I can help you with Design of Experiments. Here are some common topics:

**Design Selection**: "What design should I use for 6 factors?"
**Factor Definition**: "How do I choose factor levels?"
**Analysis Interpretation**: "What does this p-value mean?"
**Residual Diagnostics**: "How do I check if my model is valid?"
**Optimization**: "How do I find the optimal settings?"

What specific question do you have about your experiment?"""


@csrf_exempt
@require_http_methods(["GET"])
@gated_paid
def available_models(request):
    """Get available LLM models for DOE guidance."""
    from .llm_manager import LLMManager

    status = LLMManager.status()

    models = []

    if status["shared_llm"]["available"]:
        models.append({
            "id": "qwen",
            "name": "Qwen",
            "description": "General-purpose reasoning model",
            "available": True,
        })

    if status["coder_llm"]["available"] and status["coder_llm"]["available"] != status["shared_llm"]["available"]:
        models.append({
            "id": "qwen-coder",
            "name": "Qwen Coder",
            "description": "Code-optimized model",
            "available": True,
        })

    if status["anthropic"]["available"]:
        models.append({
            "id": "claude",
            "name": "Claude",
            "description": "Anthropic Claude (Enterprise)",
            "available": True,
        })

    # Always show fallback
    models.append({
        "id": "fallback",
        "name": "Built-in DOE Expert",
        "description": "Rule-based DOE guidance (always available)",
        "available": True,
    })

    return JsonResponse({
        "models": models,
        "default": "qwen" if status["shared_llm"]["available"] else "fallback",
    })

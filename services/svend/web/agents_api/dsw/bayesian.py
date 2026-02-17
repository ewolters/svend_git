"""DSW Bayesian Analysis — Bayesian inference methods for hypothesis testing."""

import numpy as np
from scipy import stats


def run_bayesian_analysis(df, analysis_id, config):
    """Run Bayesian inference analyses - feeds Synara hypothesis testing."""
    import numpy as np
    from scipy import stats

    result = {"plots": [], "summary": "", "guide_observation": ""}

    ci_level = float(config.get("ci", 0.95))
    z = stats.norm.ppf((1 + ci_level) / 2)

    if analysis_id == "bayes_regression":
        # Bayesian Linear Regression with credible intervals
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler

        target = config.get("target")
        features = config.get("features", [])

        if not target or not features:
            result["summary"] = "Error: Select target and at least one feature"
            return result

        y = df[target].dropna()
        X = df[features].loc[y.index].dropna()
        y = y.loc[X.index]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = BayesianRidge(compute_score=True)
        model.fit(X_scaled, y)

        y_pred, y_std = model.predict(X_scaled, return_std=True)
        coef_mean = model.coef_
        coef_std = np.sqrt(1.0 / model.lambda_) * np.ones_like(coef_mean)
        r2 = model.score(X_scaled, y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>R²:<</COLOR>> {r2:.4f}\n\n"
        summary += f"<<COLOR:text>>Coefficient Posteriors ({int(ci_level*100)}% Credible Intervals):<</COLOR>>\n\n"

        for i, feat in enumerate(features):
            mean = coef_mean[i]
            std = coef_std[i]
            ci_low = mean - z * std
            ci_high = mean + z * std
            sig = "***" if ci_low > 0 or ci_high < 0 else ""
            summary += f"  {feat:<20} β = {mean:>8.4f}  [{ci_low:>8.4f}, {ci_high:>8.4f}] {sig}\n"

        result["summary"] = summary
        result["plots"].append({
            "title": "Coefficient Posteriors",
            "data": [{
                "type": "scatter",
                "x": coef_mean.tolist(),
                "y": features,
                "mode": "markers",
                "marker": {"color": "#4a9f6e", "size": 10},
                "error_x": {"type": "data", "array": (z * coef_std).tolist(), "color": "#4a9f6e"},
                "name": f"β ± {int(ci_level*100)}% CI"
            }],
            "layout": {"height": max(300, len(features) * 30), "xaxis": {"zeroline": True}, "margin": {"l": 150}}
        })

        result["synara_weights"] = {
            "analysis_type": "bayesian_regression",
            "target": target,
            "coefficients": [
                {"feature": feat, "mean": float(coef_mean[i]), "ci_low": float(coef_mean[i] - z * coef_std[i]), "ci_high": float(coef_mean[i] + z * coef_std[i])}
                for i, feat in enumerate(features)
            ]
        }

    elif analysis_id == "bayes_ttest":
        # Bayesian t-test comparing two groups
        var1 = config.get("var1")
        var2 = config.get("var2")
        prior_scale = config.get("prior_scale", "medium")

        scale_map = {"small": 0.2, "medium": 0.5, "large": 0.8, "ultrawide": 1.0}
        scale = scale_map.get(prior_scale, 0.5)

        x1 = df[var1].dropna().values
        x2 = df[var2].dropna().values

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(x1)-1)*np.var(x1, ddof=1) + (len(x2)-1)*np.var(x2, ddof=1)) / (len(x1)+len(x2)-2))
        cohens_d = (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0

        # Bayes Factor approximation (JZS prior)
        t_stat, p_value = stats.ttest_ind(x1, x2)
        n_eff = 2 / (1/len(x1) + 1/len(x2))
        bf10 = np.exp(0.5 * (np.log(n_eff) - np.log(2*np.pi) - (t_stat**2)/n_eff)) if abs(t_stat) < 10 else 1e6

        # Posterior on effect size (approximate)
        se_d = np.sqrt((len(x1)+len(x2))/(len(x1)*len(x2)) + cohens_d**2/(2*(len(x1)+len(x2))))
        d_ci_low = cohens_d - z * se_d
        d_ci_high = cohens_d + z * se_d

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN T-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>{var1}<</COLOR>> (n={len(x1)}, μ={np.mean(x1):.3f})\n"
        summary += f"<<COLOR:highlight>>{var2}<</COLOR>> (n={len(x2)}, μ={np.mean(x2):.3f})\n\n"
        summary += f"<<COLOR:text>>Effect Size (Cohen's d):<</COLOR>> {cohens_d:.3f} [{d_ci_low:.3f}, {d_ci_high:.3f}]\n"
        summary += f"<<COLOR:text>>Bayes Factor (BF10):<</COLOR>> {bf10:.2f}\n\n"

        if bf10 > 10:
            summary += f"<<COLOR:success>>Strong evidence for difference<</COLOR>>\n"
        elif bf10 > 3:
            summary += f"<<COLOR:warning>>Moderate evidence for difference<</COLOR>>\n"
        elif bf10 > 1:
            summary += f"<<COLOR:text>>Weak evidence for difference<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>Evidence favors no difference<</COLOR>>\n"

        result["summary"] = summary
        result["statistics"] = {"cohens_d": cohens_d, "bf10": bf10, "d_ci_low": d_ci_low, "d_ci_high": d_ci_high}

        # Posterior distribution plot
        d_range = np.linspace(cohens_d - 3*se_d, cohens_d + 3*se_d, 100)
        posterior = stats.norm.pdf(d_range, cohens_d, se_d)

        result["plots"].append({
            "title": "Posterior Distribution of Effect Size",
            "data": [{
                "type": "scatter",
                "x": d_range.tolist(),
                "y": posterior.tolist(),
                "fill": "tozeroy",
                "fillcolor": "rgba(74, 159, 110, 0.3)",
                "line": {"color": "#4a9f6e"},
                "name": "Posterior"
            }, {
                "type": "scatter",
                "x": [0, 0],
                "y": [0, max(posterior)],
                "mode": "lines",
                "line": {"color": "#e85747", "dash": "dash"},
                "name": "No effect"
            }],
            "layout": {"height": 300, "xaxis": {"title": "Cohen's d"}, "yaxis": {"title": "Density"}}
        })

    elif analysis_id == "bayes_ab":
        # Bayesian A/B test for proportions
        group_col = config.get("group")
        success_col = config.get("success")
        prior_type = config.get("prior", "uniform")

        prior_map = {"uniform": (1, 1), "jeffreys": (0.5, 0.5), "informed": (5, 5)}
        a_prior, b_prior = prior_map.get(prior_type, (1, 1))

        groups = df[group_col].dropna().unique()
        if len(groups) < 2:
            result["summary"] = "Error: Need at least 2 groups"
            return result

        g1, g2 = groups[0], groups[1]
        s1 = df[df[group_col] == g1][success_col].sum()
        n1 = len(df[df[group_col] == g1])
        s2 = df[df[group_col] == g2][success_col].sum()
        n2 = len(df[df[group_col] == g2])

        # Posterior Beta distributions
        a1, b1 = a_prior + s1, b_prior + n1 - s1
        a2, b2 = a_prior + s2, b_prior + n2 - s2

        # Monte Carlo estimation of P(p1 > p2)
        samples1 = np.random.beta(a1, b1, 10000)
        samples2 = np.random.beta(a2, b2, 10000)
        prob_better = np.mean(samples1 > samples2)

        rate1, rate2 = s1/n1, s2/n2
        lift = (rate1 - rate2) / rate2 if rate2 > 0 else 0

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN A/B TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Group A ({g1}):<</COLOR>> {s1}/{n1} = {rate1:.1%}\n"
        summary += f"<<COLOR:highlight>>Group B ({g2}):<</COLOR>> {s2}/{n2} = {rate2:.1%}\n\n"
        summary += f"<<COLOR:text>>P({g1} > {g2}):<</COLOR>> {prob_better:.1%}\n"
        summary += f"<<COLOR:text>>Relative Lift:<</COLOR>> {lift:+.1%}\n\n"

        if prob_better > 0.95:
            summary += f"<<COLOR:success>>Strong evidence {g1} is better<</COLOR>>\n"
        elif prob_better > 0.75:
            summary += f"<<COLOR:warning>>Moderate evidence {g1} is better<</COLOR>>\n"
        elif prob_better < 0.05:
            summary += f"<<COLOR:success>>Strong evidence {g2} is better<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>Inconclusive - need more data<</COLOR>>\n"

        result["summary"] = summary
        result["statistics"] = {"prob_better": prob_better, "rate_a": rate1, "rate_b": rate2, "lift": lift}

        # Posterior distributions
        x = np.linspace(0, 1, 200)
        result["plots"].append({
            "title": "Posterior Distributions",
            "data": [{
                "type": "scatter",
                "x": x.tolist(),
                "y": stats.beta.pdf(x, a1, b1).tolist(),
                "fill": "tozeroy",
                "fillcolor": "rgba(74, 159, 110, 0.3)",
                "line": {"color": "#4a9f6e"},
                "name": f"{g1}"
            }, {
                "type": "scatter",
                "x": x.tolist(),
                "y": stats.beta.pdf(x, a2, b2).tolist(),
                "fill": "tozeroy",
                "fillcolor": "rgba(232, 149, 71, 0.3)",
                "line": {"color": "#e89547"},
                "name": f"{g2}"
            }],
            "layout": {"height": 300, "xaxis": {"title": "Conversion Rate"}, "yaxis": {"title": "Density"}}
        })

    elif analysis_id == "bayes_correlation":
        # Bayesian correlation
        var1 = config.get("var1")
        var2 = config.get("var2")

        x = df[var1].dropna()
        y = df[var2].loc[x.index].dropna()
        x = x.loc[y.index]

        r, p = stats.pearsonr(x, y)
        n = len(x)

        # Fisher z-transformation for CI
        z_r = 0.5 * np.log((1 + r) / (1 - r)) if abs(r) < 1 else 0
        se_z = 1 / np.sqrt(n - 3) if n > 3 else 1
        z_low = z_r - z * se_z
        z_high = z_r + z * se_z
        r_low = (np.exp(2*z_low) - 1) / (np.exp(2*z_low) + 1)
        r_high = (np.exp(2*z_high) - 1) / (np.exp(2*z_high) + 1)

        # BF approximation
        bf10 = np.sqrt((n-1)/2) * np.exp(stats.t.logpdf(r * np.sqrt(n-2) / np.sqrt(1-r**2), n-2)) if abs(r) < 0.999 else 100

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN CORRELATION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>{var1}<</COLOR>> vs <<COLOR:highlight>>{var2}<</COLOR>> (n={n})\n\n"
        summary += f"<<COLOR:text>>Correlation (r):<</COLOR>> {r:.3f} [{r_low:.3f}, {r_high:.3f}]\n"
        summary += f"<<COLOR:text>>Bayes Factor:<</COLOR>> {bf10:.2f}\n\n"

        strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
        direction = "positive" if r > 0 else "negative"
        summary += f"<<COLOR:text>>Interpretation: {strength} {direction} correlation<</COLOR>>\n"

        result["summary"] = summary
        result["statistics"] = {"r": r, "r_ci_low": r_low, "r_ci_high": r_high, "bf10": bf10}

        result["plots"].append({
            "title": f"Scatter: {var1} vs {var2}",
            "data": [{
                "type": "scatter",
                "x": x.values.tolist(),
                "y": y.values.tolist(),
                "mode": "markers",
                "marker": {"color": "#4a9f6e", "size": 6, "opacity": 0.6}
            }],
            "layout": {"height": 300, "xaxis": {"title": var1}, "yaxis": {"title": var2}}
        })

    elif analysis_id == "bayes_anova":
        # Bayesian ANOVA
        response = config.get("response")
        factor = config.get("factor")

        groups = df.groupby(factor)[response].apply(list).to_dict()
        group_names = list(groups.keys())
        group_data = [np.array(groups[g]) for g in group_names]

        # F-test for BF approximation
        f_stat, p_value = stats.f_oneway(*group_data)

        # Effect size (eta-squared)
        ss_between = sum(len(g) * (np.mean(g) - df[response].mean())**2 for g in group_data)
        ss_total = sum((df[response] - df[response].mean())**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN ANOVA<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor} ({len(group_names)} levels)\n\n"

        for name in group_names:
            g = groups[name]
            summary += f"  {name}: n={len(g)}, μ={np.mean(g):.3f}, σ={np.std(g):.3f}\n"

        summary += f"\n<<COLOR:text>>F-statistic:<</COLOR>> {f_stat:.3f}\n"
        summary += f"<<COLOR:text>>Effect size (η²):<</COLOR>> {eta_sq:.3f}\n"

        result["summary"] = summary
        result["statistics"] = {"f_stat": f_stat, "eta_squared": eta_sq, "p_value": p_value}

        # Box plot
        result["plots"].append({
            "title": f"{response} by {factor}",
            "data": [{
                "type": "box",
                "y": groups[name],
                "name": str(name),
                "marker": {"color": "#4a9f6e"}
            } for name in group_names],
            "layout": {"height": 350}
        })

    elif analysis_id == "bayes_changepoint":
        # Bayesian change point detection
        var = config.get("var")
        time_col = config.get("time")
        max_cp = int(config.get("max_cp", 2))

        data = df[var].dropna().values
        n = len(data)

        if time_col:
            time_idx = df[time_col].loc[df[var].dropna().index].values
        else:
            time_idx = np.arange(n)

        # Simple Bayesian change point (PELT-like with BIC)
        from scipy.signal import find_peaks

        # Cumulative sum for change detection
        cumsum = np.cumsum(data - np.mean(data))
        diff2 = np.abs(np.diff(cumsum, 2)) if n > 2 else []

        # Find peaks in second derivative (change points)
        if len(diff2) > 0:
            peaks, props = find_peaks(diff2, height=np.std(diff2), distance=max(5, n//10))
            changepoints = peaks[:max_cp] + 1 if len(peaks) > 0 else []
        else:
            changepoints = []

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN CHANGE POINT DETECTION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n\n"

        if len(changepoints) > 0:
            summary += f"<<COLOR:success>>Detected {len(changepoints)} change point(s):<</COLOR>>\n"
            for i, cp in enumerate(changepoints):
                before = data[:cp]
                after = data[cp:]
                summary += f"  Point {i+1}: index {cp}, before μ={np.mean(before):.3f}, after μ={np.mean(after):.3f}\n"
        else:
            summary += f"<<COLOR:text>>No significant change points detected<</COLOR>>\n"

        result["summary"] = summary
        result["statistics"] = {"n_changepoints": len(changepoints), "changepoint_indices": list(changepoints)}

        # Time series plot with change points
        plot_data = [{
            "type": "scatter",
            "x": time_idx.tolist() if hasattr(time_idx, 'tolist') else list(time_idx),
            "y": data.tolist(),
            "mode": "lines+markers",
            "marker": {"size": 4, "color": "#4a9f6e"},
            "line": {"color": "#4a9f6e"},
            "name": var
        }]

        for cp in changepoints:
            plot_data.append({
                "type": "scatter",
                "x": [time_idx[cp], time_idx[cp]],
                "y": [min(data), max(data)],
                "mode": "lines",
                "line": {"color": "#e85747", "dash": "dash", "width": 2},
                "name": f"Change @ {cp}"
            })

        result["plots"].append({
            "title": "Time Series with Change Points",
            "data": plot_data,
            "layout": {"height": 350, "xaxis": {"title": time_col or "Index"}, "yaxis": {"title": var}}
        })

    elif analysis_id == "bayes_proportion":
        # Bayesian proportion estimation
        success_col = config.get("success")
        prior_type = config.get("prior", "uniform")

        prior_map = {"uniform": (1, 1), "jeffreys": (0.5, 0.5), "optimistic": (8, 2), "pessimistic": (2, 8)}
        a_prior, b_prior = prior_map.get(prior_type, (1, 1))

        data = df[success_col].dropna()
        successes = int(data.sum())
        n = len(data)

        # Posterior
        a_post = a_prior + successes
        b_post = b_prior + n - successes

        # Posterior mean and CI
        post_mean = a_post / (a_post + b_post)
        ci_low, ci_high = stats.beta.ppf([(1-ci_level)/2, (1+ci_level)/2], a_post, b_post)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BAYESIAN PROPORTION ESTIMATION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Observed:<</COLOR>> {successes}/{n} = {successes/n:.1%}\n"
        summary += f"<<COLOR:highlight>>Prior:<</COLOR>> Beta({a_prior}, {b_prior})\n\n"
        summary += f"<<COLOR:text>>Posterior Mean:<</COLOR>> {post_mean:.1%}\n"
        summary += f"<<COLOR:text>>{int(ci_level*100)}% Credible Interval:<</COLOR>> [{ci_low:.1%}, {ci_high:.1%}]\n"

        result["summary"] = summary
        result["statistics"] = {"proportion": post_mean, "ci_low": ci_low, "ci_high": ci_high, "n": n, "successes": successes}

        # Posterior distribution
        x = np.linspace(0, 1, 200)
        result["plots"].append({
            "title": "Posterior Distribution",
            "data": [{
                "type": "scatter",
                "x": x.tolist(),
                "y": stats.beta.pdf(x, a_post, b_post).tolist(),
                "fill": "tozeroy",
                "fillcolor": "rgba(74, 159, 110, 0.3)",
                "line": {"color": "#4a9f6e"},
                "name": f"Beta({a_post:.0f}, {b_post:.0f})"
            }, {
                "type": "scatter",
                "x": [ci_low, ci_high],
                "y": [0, 0],
                "mode": "lines",
                "line": {"color": "#e89547", "width": 4},
                "name": f"{int(ci_level*100)}% CI"
            }],
            "layout": {"height": 300, "xaxis": {"title": "Proportion"}, "yaxis": {"title": "Density"}}
        })

    return result

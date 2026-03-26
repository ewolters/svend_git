"""Advanced regression — Bayesian, GAM, Gaussian Process, PLS, regularized.

CR: 3c0d0e53
"""

import logging
import uuid

import numpy as np

from ..common import cache_model

logger = logging.getLogger(__name__)


def _run_advanced_regression(df, analysis_id, config, user):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    target = config.get("target")
    features = config.get("features", [])

    X = df[features].dropna()
    y = df[target].loc[X.index]

    if analysis_id == "bayesian_regression":
        """
        Bayesian Linear Regression - native uncertainty, posterior over coefficients.
        Directly feeds the belief engine with credible intervals.
        """
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import StandardScaler

        # Scale features for better convergence
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit Bayesian Ridge Regression
        model = BayesianRidge(compute_score=True)
        model.fit(X_scaled, y)

        # Get predictions with uncertainty
        y_pred, y_std = model.predict(X_scaled, return_std=True)

        # Coefficient posteriors (mean and std)
        coef_mean = model.coef_
        # Approximate coefficient std from the model's alpha/lambda
        coef_std = np.sqrt(1.0 / model.lambda_) * np.ones_like(coef_mean)

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>BAYESIAN LINEAR REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n\n"

        # Model fit
        r2 = model.score(X_scaled, y)
        summary += "<<COLOR:text>>Model Fit:<</COLOR>>\n"
        summary += f"  R² = {r2:.4f}\n"
        summary += f"  α (noise precision) = {model.alpha_:.4f}\n"
        summary += f"  λ (weight precision) = {model.lambda_:.4f}\n\n"

        # Coefficient posteriors with credible intervals
        summary += (
            "<<COLOR:text>>Coefficient Posteriors (95% Credible Intervals):<</COLOR>>\n"
        )
        summary += "<<COLOR:text>>These intervals feed directly into Synara edge weights<</COLOR>>\n\n"

        for i, feat in enumerate(features):
            mean = coef_mean[i]
            std = coef_std[i]
            ci_low = mean - 1.96 * std
            ci_high = mean + 1.96 * std

            # Determine significance (CI doesn't include 0)
            sig = "***" if ci_low > 0 or ci_high < 0 else ""

            summary += f"  {feat:<20} β = {mean:>8.4f}  [{ci_low:>8.4f}, {ci_high:>8.4f}] {sig}\n"

        summary += f"\n  Intercept: {model.intercept_:.4f}\n"

        # Synara integration note
        summary += "\n<<COLOR:success>>SYNARA INTEGRATION:<</COLOR>>\n"
        summary += "  Coefficients with CIs not crossing zero indicate strong causal evidence.\n"
        summary += "  CI width indicates uncertainty in edge weight.\n"

        result["summary"] = summary

        # Plot 1: Coefficient posteriors with credible intervals
        result["plots"].append(
            {
                "title": "Coefficient Posteriors (95% CI)",
                "data": [
                    {
                        "type": "scatter",
                        "x": coef_mean.tolist(),
                        "y": features,
                        "mode": "markers",
                        "marker": {"color": "#4a9f6e", "size": 10},
                        "error_x": {
                            "type": "data",
                            "array": (1.96 * coef_std).tolist(),
                            "color": "#4a9f6e",
                            "thickness": 2,
                            "width": 6,
                        },
                        "name": "Coefficient ± 95% CI",
                    },
                    {
                        "type": "scatter",
                        "x": [0],
                        "y": [features[len(features) // 2]],
                        "mode": "lines",
                        "line": {"color": "#e85747", "dash": "dash", "width": 1},
                        "showlegend": False,
                    },
                ],
                "layout": {
                    "height": max(300, len(features) * 30),
                    "xaxis": {
                        "title": "Coefficient Value",
                        "zeroline": True,
                        "zerolinecolor": "#e85747",
                    },
                    "margin": {"l": 150},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 0,
                            "x1": 0,
                            "y0": -0.5,
                            "y1": len(features) - 0.5,
                            "line": {"color": "#e85747", "dash": "dash"},
                        }
                    ],
                },
            }
        )

        # Plot 2: Predictions with uncertainty bands
        sorted_idx = np.argsort(y.values)
        result["plots"].append(
            {
                "title": "Predictions with Uncertainty",
                "data": [
                    {
                        "type": "scatter",
                        "x": list(range(len(y))),
                        "y": y.values[sorted_idx].tolist(),
                        "mode": "markers",
                        "marker": {
                            "color": "rgba(74, 159, 110, 0.5)",
                            "size": 5,
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                        "name": "Actual",
                    },
                    {
                        "type": "scatter",
                        "x": list(range(len(y))),
                        "y": y_pred[sorted_idx].tolist(),
                        "mode": "lines",
                        "line": {"color": "#e89547"},
                        "name": "Predicted",
                    },
                    {
                        "type": "scatter",
                        "x": list(range(len(y))) + list(range(len(y)))[::-1],
                        "y": (y_pred[sorted_idx] + 1.96 * y_std[sorted_idx]).tolist()
                        + (y_pred[sorted_idx] - 1.96 * y_std[sorted_idx])[
                            ::-1
                        ].tolist(),
                        "fill": "toself",
                        "fillcolor": "rgba(232, 149, 71, 0.2)",
                        "line": {"color": "transparent"},
                        "name": "95% CI",
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Observation (sorted)"},
                    "yaxis": {"title": target},
                },
            }
        )

        result["guide_observation"] = (
            f"Bayesian regression R²={r2:.3f}. Coefficients with CIs not crossing zero are significant causal candidates."
        )

        # Include coefficient data for Synara integration
        result["synara_weights"] = {
            "analysis_type": "bayesian_regression",
            "target": target,
            "coefficients": [
                {
                    "feature": feat,
                    "mean": float(coef_mean[i]),
                    "ci_low": float(coef_mean[i] - 1.96 * coef_std[i]),
                    "ci_high": float(coef_mean[i] + 1.96 * coef_std[i]),
                }
                for i, feat in enumerate(features)
            ],
        }

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(
                user.id,
                model_key,
                {"model": model, "scaler": scaler},
                {
                    "model_type": "Bayesian Ridge Regression",
                    "features": features,
                    "target": target,
                    "metrics": {"r2": float(r2)},
                },
            )
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "gam":
        """
        Generalized Additive Model - human-readable spline curves per feature.
        Shows HOW each variable bends the response (interpretable nonlinearity).
        """
        try:
            from pygam import LinearGAM, s
        except ImportError:
            result["summary"] = "Error: pygam not installed. Run: pip install pygam"
            return result

        from sklearn.preprocessing import StandardScaler

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Build GAM with spline terms for each feature
        # s(i) creates a spline term for feature i
        terms = s(0)
        for i in range(1, X_scaled.shape[1]):
            terms += s(i)

        gam = LinearGAM(terms)
        gam.fit(X_scaled, y)

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>GENERALIZED ADDITIVE MODEL (GAM)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n"
        summary += "<<COLOR:highlight>>Model:<</COLOR>> y = Σ f_i(x_i) + ε  (smooth splines)\n\n"

        # Model statistics
        summary += "<<COLOR:text>>Model Statistics:<</COLOR>>\n"
        summary += (
            f"  Pseudo R² = {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}\n"
        )
        summary += f"  GCV Score = {gam.statistics_['GCV']:.4f}\n"
        summary += f"  Effective DF = {gam.statistics_['edof']:.1f}\n\n"

        # Feature significance (approximate p-values)
        summary += "<<COLOR:text>>Feature Effects (Spline Significance):<</COLOR>>\n"
        summary += f"<<COLOR:text>>Each curve shows HOW the feature affects {target}<</COLOR>>\n\n"

        p_values = gam.statistics_["p_values"]
        for i, feat in enumerate(features):
            p = p_values[i] if i < len(p_values) else 1.0
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            summary += f"  {feat:<20} p = {p:.4f} {sig}\n"

        summary += "\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += "  The partial dependence plots show the SHAPE of each effect.\n"
        summary += "  Non-linear curves reveal complex causal relationships.\n"

        result["summary"] = summary

        # Generate partial dependence plots for each feature
        for i, feat in enumerate(features):
            try:
                XX = gam.generate_X_grid(term=i, n=100)
                pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

                # Convert back to original scale for x-axis
                x_grid = XX[:, i] * scaler.scale_[i] + scaler.mean_[i]

                result["plots"].append(
                    {
                        "title": f"Effect of {feat}",
                        "data": [
                            {
                                "type": "scatter",
                                "x": x_grid.tolist(),
                                "y": pdep.tolist(),
                                "mode": "lines",
                                "line": {"color": "#4a9f6e", "width": 2},
                                "name": "Effect",
                            },
                            {
                                "type": "scatter",
                                "x": x_grid.tolist() + x_grid[::-1].tolist(),
                                "y": confi[:, 0].tolist() + confi[::-1, 1].tolist(),
                                "fill": "toself",
                                "fillcolor": "rgba(74, 159, 110, 0.2)",
                                "line": {"color": "transparent"},
                                "name": "95% CI",
                            },
                        ],
                        "layout": {
                            "height": 250,
                            "xaxis": {"title": feat},
                            "yaxis": {"title": f"Effect on {target}"},
                            "showlegend": False,
                        },
                    }
                )
            except Exception:
                logger.warning(
                    f"GAM: partial dependence failed for feature '{feat}', skipping plot"
                )

        result["guide_observation"] = (
            "GAM shows smooth effect curves for each feature. Non-linear patterns indicate complex causal relationships."
        )

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(
                user.id,
                model_key,
                {"gam": gam, "scaler": scaler},
                {
                    "model_type": "Generalized Additive Model",
                    "features": features,
                    "target": target,
                    "metrics": {
                        "pseudo_r2": float(
                            gam.statistics_["pseudo_r2"]["explained_deviance"]
                        )
                    },
                },
            )
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "gaussian_process":
        """
        Gaussian Process Regression - uncertainty quantification over subsets.
        Shows mean prediction with confidence bands.
        GP is O(n³) — subsample large datasets to avoid freezing.
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        from sklearn.preprocessing import StandardScaler

        # GP is O(n³) — cap at 500 rows to prevent freezing
        MAX_GP_ROWS = 500
        n_rows = len(X)
        subsampled = False
        if n_rows > MAX_GP_ROWS:
            idx = np.random.RandomState(42).choice(n_rows, MAX_GP_ROWS, replace=False)
            idx.sort()
            X = X.iloc[idx].reset_index(drop=True)
            y = y.iloc[idx].reset_index(drop=True)
            subsampled = True

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Define kernel: constant * RBF + noise
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(
            noise_level=0.1
        )

        # Fit GP — reduce restarts for larger datasets
        n_restarts = 2 if len(X) > 300 else 5
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=n_restarts, random_state=42
        )
        gp.fit(X_scaled, y)

        # Predictions with uncertainty
        y_pred, y_std = gp.predict(X_scaled, return_std=True)

        # For visualization, use first feature
        feature_for_plot = features[0]
        x_plot = X[feature_for_plot].values
        sort_idx = np.argsort(x_plot)

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>GAUSSIAN PROCESS REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n"
        if subsampled:
            summary += f"<<COLOR:warning>>Note: Subsampled to {MAX_GP_ROWS} rows (from {n_rows}) — GP is O(n³)<</COLOR>>\n"
        summary += "\n"

        summary += "<<COLOR:text>>Kernel:<</COLOR>>\n"
        summary += f"  {gp.kernel_}\n\n"

        summary += "<<COLOR:text>>Model Quality:<</COLOR>>\n"
        summary += (
            f"  Log-Marginal-Likelihood: {gp.log_marginal_likelihood_value_:.4f}\n"
        )

        # Residual analysis
        residuals = y.values - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        r2 = 1 - np.sum(residuals**2) / np.sum((y.values - y.mean()) ** 2)
        summary += f"  RMSE: {rmse:.4f}\n"
        summary += f"  R²: {r2:.4f}\n\n"

        summary += "<<COLOR:text>>Uncertainty Statistics:<</COLOR>>\n"
        summary += f"  Mean std: {y_std.mean():.4f}\n"
        summary += f"  Max std: {y_std.max():.4f}\n"
        summary += f"  Min std: {y_std.min():.4f}\n\n"

        summary += "<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += "  GP provides full uncertainty quantification.\n"
        summary += "  Wide bands = less confident predictions.\n"
        summary += "  Useful for detecting extrapolation risk.\n"

        result["summary"] = summary

        # Plot 1: Predictions with uncertainty bands
        result["plots"].append(
            {
                "title": f"GP Fit: {target} vs {feature_for_plot}",
                "data": [
                    {
                        "type": "scatter",
                        "x": x_plot[sort_idx].tolist(),
                        "y": y.values[sort_idx].tolist(),
                        "mode": "markers",
                        "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6},
                        "name": "Observed",
                    },
                    {
                        "type": "scatter",
                        "x": x_plot[sort_idx].tolist(),
                        "y": y_pred[sort_idx].tolist(),
                        "mode": "lines",
                        "line": {"color": "#4a9f6e", "width": 2},
                        "name": "GP Mean",
                    },
                    {
                        "type": "scatter",
                        "x": np.concatenate(
                            [x_plot[sort_idx], x_plot[sort_idx][::-1]]
                        ).tolist(),
                        "y": np.concatenate(
                            [
                                (y_pred[sort_idx] + 1.96 * y_std[sort_idx]),
                                (y_pred[sort_idx][::-1] - 1.96 * y_std[sort_idx][::-1]),
                            ]
                        ).tolist(),
                        "fill": "toself",
                        "fillcolor": "rgba(74, 159, 110, 0.2)",
                        "line": {"color": "rgba(74, 159, 110, 0)"},
                        "name": "95% CI",
                    },
                ],
                "layout": {
                    "height": 350,
                    "xaxis": {"title": feature_for_plot},
                    "yaxis": {"title": target},
                },
            }
        )

        # Plot 2: Uncertainty map
        result["plots"].append(
            {
                "title": "Prediction Uncertainty",
                "data": [
                    {
                        "type": "scatter",
                        "x": x_plot.tolist(),
                        "y": y_std.tolist(),
                        "mode": "markers",
                        "marker": {
                            "color": y_std.tolist(),
                            "colorscale": [
                                [0, "#4a9f6e"],
                                [0.5, "#e8c547"],
                                [1, "#e85747"],
                            ],
                            "size": 8,
                            "showscale": True,
                            "colorbar": {"title": "Std"},
                        },
                        "name": "Uncertainty",
                    }
                ],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": feature_for_plot},
                    "yaxis": {"title": "Prediction Std Dev"},
                },
            }
        )

        result["guide_observation"] = (
            f"Gaussian Process fit with R²={r2:.3f}. Mean uncertainty: {y_std.mean():.3f}. Check wide uncertainty bands for extrapolation risk."
        )

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(
                user.id,
                model_key,
                {"gp": gp, "scaler": scaler},
                {
                    "model_type": "Gaussian Process Regressor",
                    "features": features,
                    "target": target,
                    "metrics": {"r2": float(r2)},
                },
            )
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "pls":
        """
        Partial Least Squares - handles collinearity in process data.
        Projects to latent space before regression.
        """
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.preprocessing import StandardScaler

        n_components = int(config.get("n_components", min(3, len(features))))

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit PLS
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_scaled, y)

        # Get scores and loadings
        X_scores = pls.x_scores_  # T scores
        Y_scores = pls.y_scores_  # U scores
        X_loadings = pls.x_loadings_  # P loadings
        X_weights = pls.x_weights_  # W weights

        # Predictions
        y_pred = pls.predict(X_scaled).flatten()
        residuals = y.values - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        r2 = 1 - np.sum(residuals**2) / np.sum((y.values - y.mean()) ** 2)

        # Explained variance per component
        # PLS doesn't provide this directly, estimate from scores
        total_var_x = np.var(X_scaled, axis=0).sum()
        explained_var = []
        for i in range(n_components):
            comp_var = np.var(X_scores[:, i]) * np.sum(X_loadings[:, i] ** 2)
            explained_var.append(comp_var / total_var_x * 100)

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>PARTIAL LEAST SQUARES (PLS) REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n"
        summary += f"<<COLOR:highlight>>Components:<</COLOR>> {n_components}\n\n"

        summary += "<<COLOR:text>>Model Quality:<</COLOR>>\n"
        summary += f"  R²: {r2:.4f}\n"
        summary += f"  RMSE: {rmse:.4f}\n\n"

        summary += "<<COLOR:text>>Component Details:<</COLOR>>\n"
        for i in range(n_components):
            summary += f"  Component {i + 1}: ~{explained_var[i]:.1f}% X variance\n"
        summary += "\n"

        summary += "<<COLOR:text>>Feature Weights (Importance):<</COLOR>>\n"
        # VIP scores (Variable Importance in Projection)
        vip = np.sqrt(
            len(features)
            * np.sum(X_weights**2 * np.sum(Y_scores**2, axis=0), axis=1)
            / np.sum(Y_scores**2)
        )
        for feat, v in sorted(zip(features, vip), key=lambda x: -x[1]):
            marker = "★" if v > 1.0 else " "
            summary += f"  {marker} {feat}: VIP={v:.3f}\n"
        summary += "\n"

        summary += "<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += "  PLS handles collinear features by projecting to latent space.\n"
        summary += "  VIP > 1.0 indicates important predictors.\n"
        summary += "  Score plots reveal sample groupings.\n"

        result["summary"] = summary

        # Plot 1: Score plot (T1 vs T2)
        if n_components >= 2:
            result["plots"].append(
                {
                    "title": "PLS Score Plot (T1 vs T2)",
                    "data": [
                        {
                            "type": "scatter",
                            "x": X_scores[:, 0].tolist(),
                            "y": X_scores[:, 1].tolist(),
                            "mode": "markers",
                            "marker": {
                                "color": y.values.tolist(),
                                "colorscale": [
                                    [0, "#4a9f6e"],
                                    [0.5, "#e8c547"],
                                    [1, "#e85747"],
                                ],
                                "size": 8,
                                "showscale": True,
                                "colorbar": {"title": target},
                            },
                            "text": [f"{target}={v:.2f}" for v in y.values],
                            "hoverinfo": "text+x+y",
                        }
                    ],
                    "layout": {
                        "height": 350,
                        "xaxis": {"title": "T1 (Component 1)"},
                        "yaxis": {"title": "T2 (Component 2)"},
                    },
                }
            )

        # Plot 2: Loading plot (feature contributions)
        result["plots"].append(
            {
                "title": "PLS Loadings (Component 1)",
                "data": [
                    {
                        "type": "bar",
                        "x": features,
                        "y": X_loadings[:, 0].tolist(),
                        "marker": {
                            "color": [
                                "#4a9f6e" if v > 0 else "#e85747"
                                for v in X_loadings[:, 0]
                            ]
                        },
                    }
                ],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": "Feature"},
                    "yaxis": {"title": "Loading on Component 1"},
                },
            }
        )

        # Plot 3: VIP scores
        vip_sorted = sorted(zip(features, vip), key=lambda x: -x[1])
        result["plots"].append(
            {
                "title": "Variable Importance (VIP)",
                "data": [
                    {
                        "type": "bar",
                        "x": [f[0] for f in vip_sorted],
                        "y": [f[1] for f in vip_sorted],
                        "marker": {
                            "color": [
                                "#4a9f6e" if v > 1.0 else "rgba(74, 159, 110, 0.4)"
                                for _, v in vip_sorted
                            ]
                        },
                    },
                    {
                        "type": "scatter",
                        "x": [f[0] for f in vip_sorted],
                        "y": [1.0] * len(features),
                        "mode": "lines",
                        "line": {"color": "#e85747", "dash": "dash"},
                        "name": "VIP=1 threshold",
                    },
                ],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": "Feature"},
                    "yaxis": {"title": "VIP Score"},
                    "showlegend": False,
                },
            }
        )

        result["vip_scores"] = {feat: float(v) for feat, v in zip(features, vip)}
        result["guide_observation"] = (
            f"PLS with {n_components} components achieved R²={r2:.3f}. Top features by VIP: {', '.join([f[0] for f in vip_sorted[:3]])}."
        )

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(
                user.id,
                model_key,
                {"pls": pls, "scaler": scaler},
                {
                    "model_type": "PLS Regression",
                    "features": features,
                    "target": target,
                    "metrics": {"r2": float(r2), "n_components": n_components},
                },
            )
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "regularized_regression":
        """
        Ridge, LASSO, and Elastic Net regression with cross-validated alpha selection.
        Handles multicollinearity (Ridge), feature selection (LASSO), or both (Elastic Net).
        """
        from sklearn.linear_model import (
            ElasticNet,
            ElasticNetCV,
            Lasso,
            LassoCV,
            Ridge,
            RidgeCV,
        )
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        response = config.get("response")
        predictors = config.get("predictors", [])
        method = config.get("method", "elastic_net")  # ridge, lasso, elastic_net
        data = df[predictors + [response]].dropna()
        X = data[predictors].values
        y = data[response].values
        n, p_vars = X.shape

        # Standardize for regularization
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)

        # Cross-validated alpha selection
        alphas = np.logspace(-4, 4, 100)

        if method == "ridge":
            cv_model = RidgeCV(alphas=alphas, cv=min(5, n))
            cv_model.fit(X_scaled, y)
            best_alpha = cv_model.alpha_
            final_model = Ridge(alpha=best_alpha).fit(X_scaled, y)
            method_name = "Ridge"
        elif method == "lasso":
            cv_model = LassoCV(alphas=alphas, cv=min(5, n), max_iter=10000)
            cv_model.fit(X_scaled, y)
            best_alpha = cv_model.alpha_
            final_model = Lasso(alpha=best_alpha, max_iter=10000).fit(X_scaled, y)
            method_name = "LASSO"
        else:  # elastic_net
            l1_ratio = float(config.get("l1_ratio", 0.5))
            cv_model = ElasticNetCV(
                alphas=alphas, l1_ratio=l1_ratio, cv=min(5, n), max_iter=10000
            )
            cv_model.fit(X_scaled, y)
            best_alpha = cv_model.alpha_
            final_model = ElasticNet(
                alpha=best_alpha, l1_ratio=l1_ratio, max_iter=10000
            ).fit(X_scaled, y)
            method_name = f"Elastic Net (L1 ratio={l1_ratio})"

        y_pred = final_model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        coefs = final_model.coef_
        intercept = final_model.intercept_

        # Cross-validation R²
        cv_scores = cross_val_score(
            final_model, X_scaled, y, cv=min(5, n), scoring="r2"
        )

        # Count non-zero coefficients (for LASSO/elastic net)
        n_nonzero = np.sum(np.abs(coefs) > 1e-10)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>{method_name.upper()} REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {n}\n"
        summary += f"<<COLOR:highlight>>Optimal α (CV):<</COLOR>> {best_alpha:.6f}\n\n"

        summary += "<<COLOR:text>>Model Performance:<</COLOR>>\n"
        summary += f"  R²: {r2:.4f}\n"
        summary += f"  RMSE: {rmse:.4f}\n"
        summary += (
            f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n"
        )
        if method != "ridge":
            summary += f"  Non-zero coefficients: {n_nonzero}/{p_vars}\n"
        summary += f"  Intercept: {intercept:.4f}\n\n"

        summary += "<<COLOR:text>>Standardized Coefficients:<</COLOR>>\n"
        summary += (
            f"{'Predictor':<25} {'Coefficient':>12} {'|Coef|':>10} {'Status':>10}\n"
        )
        summary += f"{'─' * 60}\n"

        # Sort by absolute coefficient
        coef_order = np.argsort(-np.abs(coefs))
        for idx in coef_order:
            status = (
                "<<COLOR:good>>selected<</COLOR>>"
                if abs(coefs[idx]) > 1e-10
                else "<<COLOR:text>>dropped<</COLOR>>"
            )
            summary += f"{predictors[idx]:<25} {coefs[idx]:>12.6f} {abs(coefs[idx]):>10.6f} {status:>10}\n"

        if method != "ridge" and n_nonzero < p_vars:
            dropped = [predictors[i] for i in range(p_vars) if abs(coefs[i]) <= 1e-10]
            summary += f"\n<<COLOR:text>>Dropped features ({len(dropped)}): {', '.join(dropped)}<</COLOR>>\n"

        result["summary"] = summary

        # Coefficient bar plot
        sorted_idx = coef_order
        result["plots"].append(
            {
                "title": f"{method_name} Coefficients (α={best_alpha:.4f})",
                "data": [
                    {
                        "type": "bar",
                        "x": [predictors[i] for i in sorted_idx],
                        "y": [float(coefs[i]) for i in sorted_idx],
                        "marker": {
                            "color": [
                                (
                                    "#4a9f6e"
                                    if abs(coefs[i]) > 1e-10
                                    else "rgba(90,106,90,0.3)"
                                )
                                for i in sorted_idx
                            ]
                        },
                    }
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"tickangle": -45},
                    "yaxis": {"title": "Standardized Coefficient"},
                },
            }
        )

        # Actual vs Predicted
        result["plots"].append(
            {
                "title": "Actual vs Predicted",
                "data": [
                    {
                        "type": "scatter",
                        "x": y.tolist(),
                        "y": y_pred.tolist(),
                        "mode": "markers",
                        "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.6},
                        "name": "Data",
                    },
                    {
                        "type": "scatter",
                        "x": [float(y.min()), float(y.max())],
                        "y": [float(y.min()), float(y.max())],
                        "mode": "lines",
                        "line": {"color": "#e89547", "dash": "dash"},
                        "name": "Perfect Fit",
                    },
                ],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Actual"},
                    "yaxis": {"title": "Predicted"},
                },
            }
        )

        # Residuals vs fitted
        residuals_rr = y - y_pred
        result["plots"].append(
            {
                "title": "Residuals vs Fitted",
                "data": [
                    {
                        "type": "scatter",
                        "x": y_pred.tolist(),
                        "y": residuals_rr.tolist(),
                        "mode": "markers",
                        "marker": {"color": "#4a9f6e", "size": 5, "opacity": 0.6},
                    }
                ],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": "Fitted"},
                    "yaxis": {"title": "Residual"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": float(y_pred.min()),
                            "x1": float(y_pred.max()),
                            "y0": 0,
                            "y1": 0,
                            "line": {"color": "#e89547", "dash": "dash"},
                        }
                    ],
                },
            }
        )

        result["guide_observation"] = (
            f"{method_name}: R²={r2:.3f}, CV R²={cv_scores.mean():.3f}, α={best_alpha:.4f}. {n_nonzero}/{p_vars} features selected."
        )
        result["statistics"] = {
            "r2": float(r2),
            "rmse": float(rmse),
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
            "best_alpha": float(best_alpha),
            "n_nonzero": int(n_nonzero),
            "coefficients": {predictors[i]: float(coefs[i]) for i in range(p_vars)},
            "intercept": float(intercept),
            "method": method,
        }

    return result

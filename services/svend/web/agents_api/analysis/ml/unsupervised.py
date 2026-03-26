"""Unsupervised ML — clustering, PCA, feature importance, anomaly detection, factor analysis.

CR: 3c0d0e53
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _run_unsupervised(df, analysis_id, config, user):
    """Run unsupervised ML analysis."""
    result = {"plots": [], "summary": "", "guide_observation": ""}

    features = config.get("features", [])
    if not features:
        features = df.select_dtypes(include=[np.number]).columns.tolist()

    X = df[features].dropna()

    if analysis_id == "feature":
        target = config.get("target")
        y = df[target].loc[X.index]

    if analysis_id == "clustering":
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = int(config.get("k") or config.get("n_clusters") or 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # Silhouette for selected k
        from sklearn.metrics import silhouette_score as _sil_score

        sil = (
            _sil_score(X_scaled, clusters)
            if n_clusters > 1 and n_clusters < len(X_scaled)
            else 0.0
        )

        # Cluster size distribution
        cluster_sizes = pd.Series(clusters).value_counts().sort_index()

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>K-MEANS CLUSTERING<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Clusters:<</COLOR>> {n_clusters}\n"
        summary += f"<<COLOR:highlight>>Inertia:<</COLOR>> {kmeans.inertia_:.2f}\n"
        summary += f"<<COLOR:highlight>>Silhouette Score:<</COLOR>> {sil:.3f}\n\n"

        summary += "<<COLOR:text>>Cluster Sizes:<</COLOR>>\n"
        for c_id, c_size in cluster_sizes.items():
            summary += f"  Cluster {c_id}: {c_size} observations ({c_size / len(clusters) * 100:.0f}%)\n"

        summary += f"\n<<COLOR:accent>>{'─' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>CLUSTER QUALITY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'─' * 70}<</COLOR>>\n\n"

        if sil >= 0.7:
            summary += f"<<COLOR:good>>Strong cluster structure (silhouette = {sil:.3f}).<</COLOR>>\n"
            summary += "<<COLOR:text>>Clusters are well-separated and internally cohesive. This grouping reflects real structure in the data.<</COLOR>>"
        elif sil >= 0.5:
            summary += f"<<COLOR:good>>Reasonable cluster structure (silhouette = {sil:.3f}).<</COLOR>>\n"
            summary += "<<COLOR:text>>Clusters have meaningful separation. Some overlap exists but the grouping is useful.<</COLOR>>"
        elif sil >= 0.25:
            summary += f"<<COLOR:warn>>Weak cluster structure (silhouette = {sil:.3f}).<</COLOR>>\n"
            summary += "<<COLOR:text>>Clusters overlap substantially. Try different k values or different features. The data may not have clear natural groups.<</COLOR>>"
        else:
            summary += f"<<COLOR:danger>>No meaningful cluster structure (silhouette = {sil:.3f}).<</COLOR>>\n"
            summary += "<<COLOR:text>>Clusters are essentially arbitrary. The data does not separate into distinct groups with these features.<</COLOR>>"

        # Size imbalance warning
        max_size = cluster_sizes.max()
        min_size = cluster_sizes.min()
        if max_size > 5 * min_size:
            summary += f"\n\n<<COLOR:warn>>Imbalanced clusters:<</COLOR>> largest is {max_size / min_size:.0f}x the smallest. One cluster may be catching 'everything else.' Consider increasing k."

        result["summary"] = summary

        # Scatter plot of first two features colored by cluster
        if len(features) >= 2:
            _cluster_cd = [[int(clusters[i]), i] for i in range(len(clusters))]
            result["plots"].append(
                {
                    "title": f"Clusters ({features[0]} vs {features[1]})",
                    "data": [
                        {
                            "type": "scatter",
                            "x": X[features[0]].tolist(),
                            "y": X[features[1]].tolist(),
                            "mode": "markers",
                            "marker": {
                                "color": clusters.tolist(),
                                "colorscale": "Viridis",
                                "size": 6,
                            },
                            "customdata": _cluster_cd,
                            "hovertemplate": f"{features[0]}: %{{x:.3f}}<br>{features[1]}: %{{y:.3f}}<br>Cluster: %{{customdata[0]}}<br>Obs #%{{customdata[1]}}<extra></extra>",
                        }
                    ],
                    "layout": {"height": 300},
                    "interactive": {"type": "cluster_inspect", "features": features},
                }
            )

        # Elbow plot with silhouette scores
        max_k = min(10, len(X_scaled) - 1)
        if max_k >= 2:
            from sklearn.metrics import silhouette_score

            k_range = range(2, max_k + 1)
            inertias = []
            silhouettes = []
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                lab = km.fit_predict(X_scaled)
                inertias.append(float(km.inertia_))
                silhouettes.append(float(silhouette_score(X_scaled, lab)))
            best_k = list(k_range)[np.argmax(silhouettes)]
            result["plots"].append(
                {
                    "title": "Elbow Plot & Silhouette Score",
                    "data": [
                        {
                            "type": "scatter",
                            "x": list(k_range),
                            "y": inertias,
                            "mode": "lines+markers",
                            "marker": {"color": "#4a9f6e", "size": 7},
                            "line": {"color": "#4a9f6e"},
                            "name": "Inertia",
                            "yaxis": "y",
                        },
                        {
                            "type": "scatter",
                            "x": list(k_range),
                            "y": silhouettes,
                            "mode": "lines+markers",
                            "marker": {"color": "#e89547", "size": 7},
                            "line": {"color": "#e89547"},
                            "name": "Silhouette",
                            "yaxis": "y2",
                        },
                        {
                            "type": "scatter",
                            "x": [best_k],
                            "y": [max(silhouettes)],
                            "mode": "markers",
                            "marker": {
                                "color": "#d94a4a",
                                "size": 12,
                                "symbol": "star",
                            },
                            "name": f"Best k={best_k}",
                            "yaxis": "y2",
                        },
                    ],
                    "layout": {
                        "height": 300,
                        "xaxis": {"title": "Number of Clusters (k)"},
                        "yaxis": {"title": "Inertia", "side": "left"},
                        "yaxis2": {
                            "title": "Silhouette Score",
                            "side": "right",
                            "overlaying": "y",
                        },
                        "legend": {"x": 0.5, "y": 1.15, "orientation": "h"},
                    },
                }
            )

        sil_quality = (
            "strong" if sil >= 0.5 else "weak" if sil >= 0.25 else "no meaningful"
        )
        result["guide_observation"] = (
            f"K-Means: {n_clusters} clusters, silhouette={sil:.3f} ({sil_quality} structure). "
            + (f"Optimal k by silhouette: {best_k}." if max_k >= 2 else "")
        )

    elif analysis_id == "pca":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        n_components = int(config.get("n_components", 2))
        color_by = config.get("color")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=min(n_components, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>PRINCIPAL COMPONENT ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n"
        summary += f"<<COLOR:highlight>>Components:<</COLOR>> {pca.n_components_}\n\n"

        summary += "<<COLOR:text>>Explained Variance:<</COLOR>>\n"
        cumulative = 0
        for i, (var, ratio) in enumerate(
            zip(pca.explained_variance_, pca.explained_variance_ratio_)
        ):
            cumulative += ratio * 100
            summary += (
                f"  PC{i + 1}: {ratio * 100:.1f}% (cumulative: {cumulative:.1f}%)\n"
            )

        summary += "\n<<COLOR:text>>Loadings (feature weights):<</COLOR>>\n"
        for i, component in enumerate(pca.components_[:3]):  # Show first 3 PCs max
            summary += f"\n  PC{i + 1}:\n"
            sorted_idx = np.argsort(np.abs(component))[::-1]
            for j in sorted_idx[:5]:  # Top 5 loadings
                summary += f"    {features[j]}: {component[j]:.3f}\n"

        result["summary"] = summary

        # Biplot (first 2 components)
        if pca.n_components_ >= 2:
            color_values = None
            if color_by and color_by in df.columns:
                color_values = df[color_by].loc[X.index].astype(str).tolist()

            scatter_data = {
                "type": "scatter",
                "x": X_pca[:, 0].tolist(),
                "y": X_pca[:, 1].tolist(),
                "mode": "markers",
                "marker": {"size": 6},
                "name": "Observations",
            }
            if color_values:
                scatter_data["text"] = color_values
                scatter_data["marker"]["color"] = [hash(v) % 10 for v in color_values]
                scatter_data["marker"]["colorscale"] = "Viridis"

            result["plots"].append(
                {
                    "title": "PCA Biplot",
                    "data": [scatter_data],
                    "layout": {
                        "height": 400,
                        "xaxis": {
                            "title": f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
                        },
                        "yaxis": {
                            "title": f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
                        },
                    },
                }
            )

        # Scree plot
        result["plots"].append(
            {
                "title": "Scree Plot",
                "data": [
                    {
                        "type": "bar",
                        "x": [
                            f"PC{i + 1}"
                            for i in range(len(pca.explained_variance_ratio_))
                        ],
                        "y": (pca.explained_variance_ratio_ * 100).tolist(),
                        "marker": {
                            "color": "rgba(74, 159, 110, 0.4)",
                            "line": {"color": "#4a9f6e", "width": 1.5},
                        },
                    }
                ],
                "layout": {"height": 250, "yaxis": {"title": "Variance Explained (%)"}},
            }
        )

    elif analysis_id == "feature":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder

        # Determine if classification or regression
        y_unique = y.nunique()
        is_classification = y.dtype == "object" or y_unique < 10

        if is_classification:
            if y.dtype == "object":
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
            else:
                y_encoded = y
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y_encoded)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>FEATURE IMPORTANCE (Random Forest)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Task:<</COLOR>> {'Classification' if is_classification else 'Regression'}\n\n"

        summary += "<<COLOR:text>>Feature Rankings:<</COLOR>>\n"
        for rank, idx in enumerate(indices, 1):
            bar_len = int(importances[idx] * 30)
            bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
            summary += f"  {rank}. {features[idx]:<20} {bar} {importances[idx]:.3f}\n"

        result["summary"] = summary

        # Horizontal bar chart
        sorted_features = [features[i] for i in indices]
        sorted_importances = [importances[i] for i in indices]

        result["plots"].append(
            {
                "title": "Feature Importance",
                "data": [
                    {
                        "type": "bar",
                        "x": sorted_importances[::-1],
                        "y": sorted_features[::-1],
                        "orientation": "h",
                        "marker": {
                            "color": "rgba(74, 159, 110, 0.4)",
                            "line": {"color": "#4a9f6e", "width": 1.5},
                        },
                    }
                ],
                "layout": {
                    "height": max(250, len(features) * 25),
                    "xaxis": {"title": "Importance"},
                    "margin": {"l": 150},
                },
            }
        )

    elif analysis_id == "isolation_forest":
        """
        Isolation Forest - anomaly detection as 'missing cause' signal.
        Points that don't fit trigger causal expansion in Synara.
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        contamination = float(config.get("contamination", 0.05))

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit Isolation Forest
        iso = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100
        )
        predictions = iso.fit_predict(X_scaled)
        scores = iso.decision_function(X_scaled)

        # Identify anomalies
        anomalies = predictions == -1
        n_anomalies = anomalies.sum()
        anomaly_pct = n_anomalies / len(predictions) * 100

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>ISOLATION FOREST (Anomaly Detection)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n"
        summary += (
            f"<<COLOR:highlight>>Contamination:<</COLOR>> {contamination:.1%}\n\n"
        )

        summary += "<<COLOR:text>>Results:<</COLOR>>\n"
        summary += f"  Total observations: {len(predictions)}\n"
        summary += f"  Anomalies detected: {n_anomalies} ({anomaly_pct:.1f}%)\n"
        summary += f"  Normal observations: {len(predictions) - n_anomalies}\n\n"

        # Anomaly score statistics
        summary += "<<COLOR:text>>Anomaly Scores:<</COLOR>>\n"
        summary += f"  Mean score: {scores.mean():.4f}\n"
        summary += f"  Std score: {scores.std():.4f}\n"
        summary += "  Threshold: ~0.0 (negative = anomaly)\n\n"

        # Show most anomalous observations
        if n_anomalies > 0:
            summary += "<<COLOR:warning>>MOST ANOMALOUS OBSERVATIONS:<</COLOR>>\n"
            anomaly_idx = np.where(anomalies)[0]
            sorted_anomalies = anomaly_idx[np.argsort(scores[anomaly_idx])][:10]

            for idx in sorted_anomalies:
                summary += f"  Row {idx}: score={scores[idx]:.4f}\n"
                for feat in features[:3]:
                    summary += f"    {feat}={X[feat].iloc[idx]:.2f}\n"
                summary += "\n"

        summary += "<<COLOR:success>>SYNARA INTEGRATION:<</COLOR>>\n"
        summary += "  Anomalies are observations that don't fit the current model.\n"
        summary += "  This signals MISSING CAUSES - trigger causal expansion.\n"
        summary += "  Investigate what makes these points different.\n"

        result["summary"] = summary

        # Plot 1: Anomaly scores distribution
        result["plots"].append(
            {
                "title": "Anomaly Score Distribution",
                "data": [
                    {
                        "type": "histogram",
                        "x": scores.tolist(),
                        "nbinsx": 50,
                        "marker": {
                            "color": "rgba(74, 159, 110, 0.6)",
                            "line": {"color": "#4a9f6e", "width": 1},
                        },
                        "name": "Scores",
                    },
                    {
                        "type": "scatter",
                        "x": [0, 0],
                        "y": [0, len(scores) / 10],
                        "mode": "lines",
                        "line": {"color": "#e85747", "dash": "dash", "width": 2},
                        "name": "Threshold",
                    },
                ],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": "Anomaly Score (negative = anomaly)"},
                    "yaxis": {"title": "Count"},
                    "shapes": [
                        {
                            "type": "line",
                            "x0": 0,
                            "x1": 0,
                            "y0": 0,
                            "y1": 1,
                            "yref": "paper",
                            "line": {"color": "#e85747", "dash": "dash"},
                        }
                    ],
                },
            }
        )

        # Plot 2: Scatter with anomalies highlighted (first 2 features)
        if len(features) >= 2:
            colors = ["#e85747" if a else "rgba(74, 159, 110, 0.5)" for a in anomalies]
            sizes = [12 if a else 6 for a in anomalies]

            result["plots"].append(
                {
                    "title": f"Anomalies: {features[0]} vs {features[1]}",
                    "data": [
                        {
                            "type": "scatter",
                            "x": X[features[0]].tolist(),
                            "y": X[features[1]].tolist(),
                            "mode": "markers",
                            "marker": {
                                "color": colors,
                                "size": sizes,
                                "line": {
                                    "color": "#e85747",
                                    "width": [1 if a else 0 for a in anomalies],
                                },
                            },
                            "text": [f"Score: {s:.3f}" for s in scores],
                            "hoverinfo": "text+x+y",
                        }
                    ],
                    "layout": {
                        "height": 350,
                        "xaxis": {"title": features[0]},
                        "yaxis": {"title": features[1]},
                    },
                }
            )

        # Store anomaly data for potential export
        result["anomalies"] = {
            "indices": np.where(anomalies)[0].tolist(),
            "scores": scores[anomalies].tolist(),
            "count": int(n_anomalies),
        }

        result["guide_observation"] = (
            f"Isolation Forest detected {n_anomalies} anomalies ({anomaly_pct:.1f}%). These are 'missing cause' signals - investigate what makes them different."
        )

    elif analysis_id == "factor_analysis":
        """
        Exploratory Factor Analysis — identifies latent factors underlying observed variables.
        Supports varimax, promax, and no rotation. Includes scree plot, loading heatmap,
        communalities table.
        """
        variables = config.get("variables", [])
        n_factors = config.get("n_factors", None)
        rotation = config.get("rotation", "varimax")

        try:
            from sklearn.decomposition import FactorAnalysis

            if not variables:
                variables = df.select_dtypes(include=[np.number]).columns.tolist()

            data = df[variables].dropna()
            N = len(data)
            p = len(variables)

            # Standardize
            X = data.values.astype(float)
            X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

            # Determine number of factors via eigenvalues > 1 (Kaiser criterion) if not specified
            corr_matrix = np.corrcoef(X_std.T)
            eigvals = np.sort(np.linalg.eigvalsh(corr_matrix))[::-1]

            if n_factors is None:
                n_factors = max(1, int(np.sum(eigvals > 1)))
            n_factors = min(n_factors, p)

            # Fit factor analysis
            fa = FactorAnalysis(n_components=n_factors, random_state=42)
            scores = fa.fit_transform(X_std)
            loadings = fa.components_.T  # shape: (p, n_factors)

            # Apply rotation
            if rotation == "varimax" and n_factors > 1:
                # Varimax rotation
                rotated = loadings.copy()
                for _ in range(100):
                    old = rotated.copy()
                    for j in range(n_factors):
                        for k in range(j + 1, n_factors):
                            u = rotated[:, j] ** 2 - rotated[:, k] ** 2
                            v = 2 * rotated[:, j] * rotated[:, k]
                            A = np.sum(u)
                            B = np.sum(v)
                            C = np.sum(u**2 - v**2)
                            D = 2 * np.sum(u * v)
                            num = D - 2 * A * B / p
                            den = C - (A**2 - B**2) / p
                            angle = 0.25 * np.arctan2(num, den)
                            cos_a, sin_a = np.cos(angle), np.sin(angle)
                            rotated[:, [j, k]] = rotated[:, [j, k]] @ np.array(
                                [[cos_a, sin_a], [-sin_a, cos_a]]
                            )
                    if np.allclose(rotated, old, atol=1e-6):
                        break
                loadings = rotated

            # Communalities
            communalities = np.sum(loadings**2, axis=1)

            # Variance explained
            var_explained = np.sum(loadings**2, axis=0)
            pct_var = var_explained / p * 100

            summary_text = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_text += "<<COLOR:title>>EXPLORATORY FACTOR ANALYSIS<</COLOR>>\n"
            summary_text += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_text += (
                f"<<COLOR:highlight>>Variables:<</COLOR>> {', '.join(variables)}\n"
            )
            summary_text += f"<<COLOR:highlight>>N:<</COLOR>> {N}\n"
            summary_text += (
                f"<<COLOR:highlight>>Factors extracted:<</COLOR>> {n_factors}\n"
            )
            summary_text += f"<<COLOR:highlight>>Rotation:<</COLOR>> {rotation}\n\n"

            # Factor loadings table
            summary_text += (
                "<<COLOR:text>>Factor Loadings"
                + (f" ({rotation} rotated)" if rotation != "none" else "")
                + ":<</COLOR>>\n"
            )
            header = (
                f"{'Variable':<20}"
                + "".join([f"{'F' + str(i + 1):>10}" for i in range(n_factors)])
                + f"{'Communality':>12}\n"
            )
            summary_text += header
            summary_text += f"{'─' * (20 + 10 * n_factors + 12)}\n"
            for vi, var_name in enumerate(variables):
                row = f"{var_name:<20}"
                for fi in range(n_factors):
                    val = loadings[vi, fi]
                    if abs(val) >= 0.4:
                        row += f"<<COLOR:good>>{val:>10.3f}<</COLOR>>"
                    else:
                        row += f"{val:>10.3f}"
                row += f"{communalities[vi]:>12.3f}\n"
                summary_text += row

            summary_text += "\n<<COLOR:text>>Variance Explained:<</COLOR>>\n"
            for fi in range(n_factors):
                summary_text += (
                    f"  Factor {fi + 1}: {var_explained[fi]:.3f} ({pct_var[fi]:.1f}%)\n"
                )
            summary_text += (
                f"  Total: {np.sum(var_explained):.3f} ({np.sum(pct_var):.1f}%)\n"
            )

            result["summary"] = summary_text

            # Scree plot
            result["plots"].append(
                {
                    "title": "Scree Plot",
                    "data": [
                        {
                            "x": list(range(1, len(eigvals) + 1)),
                            "y": eigvals.tolist(),
                            "mode": "lines+markers",
                            "name": "Eigenvalues",
                            "marker": {"color": "#4a90d9", "size": 8},
                            "line": {"color": "#4a90d9", "width": 2},
                        },
                        {
                            "x": [1, len(eigvals)],
                            "y": [1, 1],
                            "mode": "lines",
                            "name": "Kaiser Criterion",
                            "line": {"color": "#d94a4a", "dash": "dash"},
                        },
                    ],
                    "layout": {
                        "height": 280,
                        "xaxis": {"title": "Factor Number"},
                        "yaxis": {"title": "Eigenvalue"},
                    },
                }
            )

            # Loading heatmap
            result["plots"].append(
                {
                    "title": f"Factor Loadings Heatmap ({rotation})",
                    "data": [
                        {
                            "type": "heatmap",
                            "z": loadings.tolist(),
                            "x": [f"Factor {i + 1}" for i in range(n_factors)],
                            "y": variables,
                            "colorscale": "RdBu",
                            "zmid": 0,
                            "text": [
                                [f"{loadings[vi, fi]:.3f}" for fi in range(n_factors)]
                                for vi in range(p)
                            ],
                            "texttemplate": "%{text}",
                            "showscale": True,
                        }
                    ],
                    "layout": {"height": max(250, p * 25)},
                }
            )

            result["statistics"] = {
                "n_factors": n_factors,
                "rotation": rotation,
                "n": N,
                "eigenvalues": eigvals.tolist(),
                "variance_explained": var_explained.tolist(),
                "pct_variance": pct_var.tolist(),
                "communalities": {
                    v: float(communalities[i]) for i, v in enumerate(variables)
                },
                "total_variance_explained": float(np.sum(pct_var)),
            }
            result["guide_observation"] = (
                f"Factor analysis: {n_factors} factors extracted ({rotation}), explaining {np.sum(pct_var):.1f}% of variance."
            )

        except Exception as e:
            result["summary"] = f"Factor analysis error: {str(e)}"

    return result

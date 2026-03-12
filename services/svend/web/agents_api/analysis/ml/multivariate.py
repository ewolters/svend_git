"""Multivariate methods — SEM, discriminant analysis, correspondence analysis, item analysis.

CR: 3c0d0e53
"""

import logging
import uuid

import numpy as np
import pandas as pd

from ..common import cache_model

logger = logging.getLogger(__name__)


def _run_multivariate(df, analysis_id, config, user):
    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "sem":
        """
        Structural Equation Modeling - causal path analysis.
        Supports path models, mediation, and latent variables.
        """
        try:
            from semopy import Model
            from semopy.stats import calc_stats
        except ImportError:
            result["summary"] = "Error: semopy not installed. Run: pip install semopy"
            return result

        model_type = config.get("model_type", "path")
        outcome = config.get("outcome")
        predictors = config.get("predictors", [])
        mediator = config.get("mediator")

        if not outcome or not predictors:
            result["summary"] = "Error: Please select outcome and predictors."
            return result

        # Build model specification based on type
        if model_type == "mediation" and mediator:
            # Classic mediation: X → M → Y (and X → Y for direct effect)
            predictor = predictors[0] if predictors else None
            if not predictor:
                result["summary"] = "Error: Mediation requires at least one predictor."
                return result

            model_spec = f"""
            # Direct effect
            {outcome} ~ c*{predictor}
            # Path a: predictor to mediator
            {mediator} ~ a*{predictor}
            # Path b: mediator to outcome
            {outcome} ~ b*{mediator}
            # Indirect effect
            indirect := a*b
            # Total effect
            total := c + a*b
            """
        else:
            # Path model: multiple predictors → outcome
            predictor_terms = " + ".join(predictors)
            model_spec = f"{outcome} ~ {predictor_terms}"

        # Prepare data - only keep relevant columns
        relevant_cols = [outcome] + predictors
        if mediator:
            relevant_cols.append(mediator)
        relevant_cols = list(set(relevant_cols))

        model_df = df[relevant_cols].dropna()

        if len(model_df) < 30:
            result["summary"] = (
                f"Warning: Only {len(model_df)} complete cases. SEM typically needs n > 200 for stable estimates."
            )

        # Fit model
        try:
            mod = Model(model_spec)
            mod.fit(model_df)
        except Exception as e:
            result["summary"] = f"Error fitting SEM model: {str(e)}"
            return result

        # Get estimates
        estimates = mod.inspect()

        # Get fit statistics
        try:
            stats = calc_stats(mod)
            chi2 = stats.get("chi2", [None])[0]
            dof = stats.get("dof", [None])[0]
            pvalue = stats.get("chi2 p-value", [None])[0]
            cfi = stats.get("CFI", [None])[0]
            tli = stats.get("TLI", [None])[0]
            rmsea = stats.get("RMSEA", [None])[0]
            srmr = stats.get("SRMR", [None])[0]
        except Exception:
            chi2 = dof = pvalue = cfi = tli = rmsea = srmr = None

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += "<<COLOR:title>>STRUCTURAL EQUATION MODEL<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"

        summary += f"<<COLOR:highlight>>Model Type:<</COLOR>> {model_type.title()}\n"
        summary += f"<<COLOR:highlight>>Outcome:<</COLOR>> {outcome}\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        if mediator:
            summary += f"<<COLOR:highlight>>Mediator:<</COLOR>> {mediator}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {len(model_df)}\n\n"

        # Fit indices
        summary += "<<COLOR:text>>Model Fit:<</COLOR>>\n"
        if chi2 is not None:
            summary += f"  χ² = {chi2:.3f}, df = {dof}, p = {pvalue:.4f}\n"
        if cfi is not None:
            cfi_ok = "✓" if cfi > 0.95 else "⚠" if cfi > 0.90 else "✗"
            summary += f"  CFI = {cfi:.3f} {cfi_ok} (>.95 good)\n"
        if tli is not None:
            tli_ok = "✓" if tli > 0.95 else "⚠" if tli > 0.90 else "✗"
            summary += f"  TLI = {tli:.3f} {tli_ok} (>.95 good)\n"
        if rmsea is not None:
            rmsea_ok = "✓" if rmsea < 0.05 else "⚠" if rmsea < 0.08 else "✗"
            summary += f"  RMSEA = {rmsea:.3f} {rmsea_ok} (<.05 good)\n"
        if srmr is not None:
            srmr_ok = "✓" if srmr < 0.08 else "⚠" if srmr < 0.10 else "✗"
            summary += f"  SRMR = {srmr:.3f} {srmr_ok} (<.08 good)\n"
        summary += "\n"

        # Path estimates
        summary += "<<COLOR:text>>Path Estimates:<</COLOR>>\n"
        for _, row in estimates.iterrows():
            lval = row.get("lval", "")
            op = row.get("op", "")
            rval = row.get("rval", "")
            est = row.get("Estimate", 0)
            se = row.get("Std. Err", 0)
            pval = row.get("p-value", 1)

            if op == "~":  # Regression
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                summary += f"  {lval} ← {rval}: β = {est:.3f} (SE={se:.3f}) {sig}\n"
            elif op == ":=":  # Defined parameter
                summary += f"  <<COLOR:accent>>{lval}<</COLOR>>: {est:.3f} (SE={se:.3f})\n"
        summary += "\n"

        # Interpretation for mediation
        if model_type == "mediation" and mediator:
            indirect_row = estimates[estimates["lval"] == "indirect"]
            if not indirect_row.empty:
                indirect_est = indirect_row["Estimate"].values[0]
                indirect_p = indirect_row.get("p-value", pd.Series([1])).values[0]

                summary += "<<COLOR:success>>MEDIATION ANALYSIS:<</COLOR>>\n"
                if indirect_p < 0.05:
                    summary += f"  Significant indirect effect: {indirect_est:.3f}\n"
                    summary += f"  {mediator} mediates the {predictors[0]} → {outcome} relationship.\n"
                else:
                    summary += f"  No significant mediation (indirect = {indirect_est:.3f}, p > .05)\n"

        result["summary"] = summary

        # Plot 1: Path diagram (simplified as coefficient bar chart)
        path_data = estimates[estimates["op"] == "~"].copy()
        if not path_data.empty:
            labels = [f"{row['rval']} → {row['lval']}" for _, row in path_data.iterrows()]
            coefs = path_data["Estimate"].tolist()
            colors = ["#4a9f6e" if c > 0 else "#e85747" for c in coefs]

            result["plots"].append(
                {
                    "title": "Path Coefficients",
                    "data": [{"type": "bar", "x": coefs, "y": labels, "orientation": "h", "marker": {"color": colors}}],
                    "layout": {
                        "height": max(200, len(labels) * 40),
                        "xaxis": {"title": "Standardized Coefficient"},
                        "yaxis": {"automargin": True},
                        "margin": {"l": 150},
                    },
                }
            )

        # Plot 2: Residuals vs fitted (if possible)
        try:
            fitted = mod.predict(model_df)
            if outcome in fitted.columns:
                y_fitted = fitted[outcome].values
                y_actual = model_df[outcome].values
                residuals = y_actual - y_fitted

                result["plots"].append(
                    {
                        "title": f"Residuals: {outcome}",
                        "data": [
                            {
                                "type": "scatter",
                                "x": y_fitted.tolist(),
                                "y": residuals.tolist(),
                                "mode": "markers",
                                "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6},
                            },
                            {
                                "type": "scatter",
                                "x": [min(y_fitted), max(y_fitted)],
                                "y": [0, 0],
                                "mode": "lines",
                                "line": {"color": "#e85747", "dash": "dash"},
                            },
                        ],
                        "layout": {
                            "height": 250,
                            "xaxis": {"title": "Fitted"},
                            "yaxis": {"title": "Residual"},
                            "showlegend": False,
                        },
                    }
                )
        except Exception:
            pass

        fit_quality = (
            "good"
            if (cfi and cfi > 0.95 and rmsea and rmsea < 0.05)
            else "acceptable"
            if (cfi and cfi > 0.90)
            else "poor"
        )
        result["guide_observation"] = (
            f"SEM {model_type} model with {fit_quality} fit. Check path coefficients for significant relationships."
        )

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(
                user.id,
                model_key,
                mod,
                {
                    "model_type": f"SEM ({model_type.title()})",
                    "features": predictors,
                    "target": outcome,
                    "metrics": {"cfi": float(cfi) if cfi else None, "rmsea": float(rmsea) if rmsea else None},
                },
            )
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "discriminant_analysis":
        """
        Discriminant Analysis — LDA and QDA for classification and dimensionality reduction.
        LDA finds linear boundaries; QDA allows quadratic (class-specific covariances).
        Reports classification accuracy, prior probabilities, discriminant coefficients.
        """
        from sklearn.discriminant_analysis import (
            LinearDiscriminantAnalysis,
            QuadraticDiscriminantAnalysis,
        )
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        response = config.get("response") or config.get("target")
        predictors = config.get("predictors") or config.get("features", [])
        method = config.get("method", "lda")  # lda or qda

        if not response:
            result["summary"] = "Error: Please select a target (group) variable."
            return result
        if not predictors:
            predictors = [c for c in df.select_dtypes(include=[np.number]).columns if c != response]
        if not predictors:
            result["summary"] = "Error: No numeric predictor variables available."
            return result

        data = df[[response] + predictors].dropna()
        le = LabelEncoder()
        y = le.fit_transform(data[response])
        X = data[predictors].values.astype(float)
        classes = le.classes_

        # Split for evaluation
        from sklearn.model_selection import cross_val_score as cvs

        split_val = float(config.get("split", 20))
        test_frac = split_val if split_val < 1 else split_val / 100

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, random_state=42, stratify=y)

        if method == "qda":
            model = QuadraticDiscriminantAnalysis()
            model_name = "Quadratic Discriminant Analysis (QDA)"
        else:
            n_components = min(len(classes) - 1, len(predictors))
            model = LinearDiscriminantAnalysis(n_components=n_components)
            model_name = "Linear Discriminant Analysis (LDA)"

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, y_pred)
        cv_scores = cvs(model, X, y, cv=min(5, len(classes)), scoring="accuracy")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        result["plots"].append(
            {
                "data": [
                    {
                        "z": cm.tolist(),
                        "x": [str(c) for c in classes],
                        "y": [str(c) for c in classes],
                        "type": "heatmap",
                        "colorscale": [[0, "#f0f4f0"], [1, "#2c5f2d"]],
                        "text": [[str(v) for v in row] for row in cm],
                        "texttemplate": "%{text}",
                        "showscale": True,
                    }
                ],
                "layout": {
                    "title": f"{model_name} — Confusion Matrix",
                    "xaxis": {"title": "Predicted"},
                    "yaxis": {"title": "Actual", "autorange": "reversed"},
                },
            }
        )

        # LDA: scatter plot in discriminant space
        if method != "qda" and hasattr(model, "transform") and n_components >= 1:
            X_proj = model.transform(X)
            if X_proj.shape[1] >= 2:
                traces = []
                colors = ["#2c5f2d", "#4a90d9", "#d94a4a", "#d9a04a", "#7d4ad9", "#d94a99"]
                for i, cls in enumerate(classes):
                    mask = y == i
                    traces.append(
                        {
                            "x": X_proj[mask, 0].tolist(),
                            "y": X_proj[mask, 1].tolist(),
                            "mode": "markers",
                            "name": str(cls),
                            "marker": {"color": colors[i % len(colors)], "size": 6, "opacity": 0.7},
                            "type": "scatter",
                        }
                    )
                result["plots"].append(
                    {
                        "data": traces,
                        "layout": {
                            "title": "LDA — Discriminant Space Projection",
                            "xaxis": {"title": "LD1"},
                            "yaxis": {"title": "LD2"},
                        },
                    }
                )
            else:
                # 1D discriminant: histogram
                traces = []
                colors = ["#2c5f2d", "#4a90d9", "#d94a4a", "#d9a04a"]
                for i, cls in enumerate(classes):
                    mask = y == i
                    traces.append(
                        {
                            "x": X_proj[mask, 0].tolist(),
                            "type": "histogram",
                            "name": str(cls),
                            "opacity": 0.6,
                            "marker": {"color": colors[i % len(colors)]},
                        }
                    )
                result["plots"].append(
                    {
                        "data": traces,
                        "layout": {
                            "title": "LDA — Discriminant Score Distribution",
                            "xaxis": {"title": "LD1 Score"},
                            "yaxis": {"title": "Count"},
                            "barmode": "overlay",
                        },
                    }
                )

        # Coefficient importance (LDA only)
        coef_info = ""
        if method != "qda" and hasattr(model, "coef_"):
            coef_magnitudes = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            sorted_idx = np.argsort(coef_magnitudes)[::-1]
            coef_rows = []
            for idx in sorted_idx:
                coef_val = model.coef_[0][idx] if model.coef_.ndim > 1 else model.coef_[idx]
                coef_rows.append(f"| {predictors[idx]} | {coef_val:.4f} |")
            coef_info = (
                "\n\n**Discriminant Coefficients (LD1):**\n| Predictor | Coefficient |\n|---|---|\n"
                + "\n".join(coef_rows)
            )

        # Classification report
        cr = classification_report(y_test, y_pred, target_names=[str(c) for c in classes])

        result["summary"] = (
            f"**{model_name}**\n\nClasses: {', '.join(str(c) for c in classes)} ({len(classes)} groups)\nTraining accuracy: {train_acc:.3f}\nTest accuracy: {test_acc:.3f}\nCV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n\nPrior probabilities: {', '.join(f'{c}: {p:.3f}' for c, p in zip(classes, model.priors_))}{coef_info}\n\n**Classification Report:**\n```\n{cr}\n```"
        )
        result["guide_observation"] = (
            f"{method.upper()}: test accuracy={test_acc:.3f}, CV accuracy={cv_scores.mean():.3f}. {len(classes)} classes, {len(predictors)} predictors."
        )
        result["statistics"] = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "n_classes": len(classes),
            "n_predictors": len(predictors),
            "priors": {str(c): float(p) for c, p in zip(classes, model.priors_)},
            "method": method,
        }

    elif analysis_id == "correspondence_analysis":
        """
        Correspondence Analysis — visualizes associations in a contingency table
        as a biplot. Decomposes chi-squared structure into orthogonal dimensions.
        Shows row and column profiles in shared low-dimensional space.
        """
        row_var_ca = config.get("row_var") or config.get("rows")
        col_var_ca = config.get("col_var") or config.get("columns")

        data_ca = df[[row_var_ca, col_var_ca]].dropna()

        try:
            ct_ca = pd.crosstab(data_ca[row_var_ca], data_ca[col_var_ca])
            if ct_ca.shape[0] < 2 or ct_ca.shape[1] < 2:
                result["summary"] = "Need at least 2 rows and 2 columns for correspondence analysis."
                return result

            # Total, row/col profiles
            N_ca = ct_ca.values.sum()
            P_ca = ct_ca.values / N_ca  # correspondence matrix
            r_ca = P_ca.sum(axis=1)  # row masses
            c_ca = P_ca.sum(axis=0)  # column masses

            # Standardized residuals
            Dr_inv_sqrt = np.diag(1.0 / np.sqrt(r_ca))
            Dc_inv_sqrt = np.diag(1.0 / np.sqrt(c_ca))
            S_ca = Dr_inv_sqrt @ (P_ca - np.outer(r_ca, c_ca)) @ Dc_inv_sqrt

            # SVD
            U_ca, sigma_ca, Vt_ca = np.linalg.svd(S_ca, full_matrices=False)

            # Number of dimensions (min of rows-1, cols-1)
            n_dims = min(ct_ca.shape[0] - 1, ct_ca.shape[1] - 1, 2)
            if n_dims < 1:
                result["summary"] = "Not enough dimensions for correspondence analysis."
                return result

            # Inertia (eigenvalues = sigma^2)
            inertia = sigma_ca**2
            total_inertia = float(np.sum(inertia))
            pct_inertia = inertia / total_inertia * 100 if total_inertia > 0 else inertia * 0

            # Row and column coordinates (principal coordinates)
            row_coords = Dr_inv_sqrt @ U_ca[:, :n_dims] * sigma_ca[:n_dims]
            col_coords = Dc_inv_sqrt @ Vt_ca[:n_dims, :].T * sigma_ca[:n_dims]

            row_labels_ca = [str(x) for x in ct_ca.index]
            col_labels_ca = [str(x) for x in ct_ca.columns]

            # Chi-squared test of independence
            from scipy import stats as ca_stats

            chi2_ca, p_chi2_ca, dof_ca, _ = ca_stats.chi2_contingency(ct_ca.values)

            # Summary
            summary_ca = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_ca += "<<COLOR:title>>CORRESPONDENCE ANALYSIS<</COLOR>>\n"
            summary_ca += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_ca += f"<<COLOR:highlight>>Row variable:<</COLOR>> {row_var_ca} ({ct_ca.shape[0]} levels)\n"
            summary_ca += f"<<COLOR:highlight>>Column variable:<</COLOR>> {col_var_ca} ({ct_ca.shape[1]} levels)\n"
            summary_ca += f"<<COLOR:highlight>>Total N:<</COLOR>> {int(N_ca)}\n\n"

            summary_ca += "<<COLOR:text>>Chi-squared test of independence:<</COLOR>>\n"
            summary_ca += f"  χ² = {chi2_ca:.2f},  df = {dof_ca},  p = {p_chi2_ca:.4f}"
            if p_chi2_ca < 0.05:
                summary_ca += "  <<COLOR:good>>Significant association<</COLOR>>"
            summary_ca += "\n\n"

            summary_ca += "<<COLOR:text>>Inertia (variance explained by dimensions):<</COLOR>>\n"
            for d in range(min(len(inertia), 5)):
                summary_ca += f"  Dim {d + 1}: {inertia[d]:.4f} ({pct_inertia[d]:.1f}%)\n"
            summary_ca += f"  Total: {total_inertia:.4f}\n"

            if n_dims >= 2:
                summary_ca += f"\n<<COLOR:text>>First 2 dimensions explain {pct_inertia[0] + pct_inertia[1]:.1f}% of inertia.<</COLOR>>\n"

            # Row contributions
            summary_ca += "\n<<COLOR:text>>Row Coordinates:<</COLOR>>\n"
            summary_ca += f"{'Level':<20}" + "".join([f"{'Dim ' + str(d + 1):>10}" for d in range(n_dims)]) + "\n"
            summary_ca += f"{'─' * (20 + 10 * n_dims)}\n"
            for ri, rl in enumerate(row_labels_ca):
                row_str = f"{rl:<20}"
                for d in range(n_dims):
                    row_str += f"{row_coords[ri, d]:>10.4f}"
                summary_ca += row_str + "\n"

            summary_ca += "\n<<COLOR:text>>Column Coordinates:<</COLOR>>\n"
            summary_ca += f"{'Level':<20}" + "".join([f"{'Dim ' + str(d + 1):>10}" for d in range(n_dims)]) + "\n"
            summary_ca += f"{'─' * (20 + 10 * n_dims)}\n"
            for ci, cl in enumerate(col_labels_ca):
                col_str = f"{cl:<20}"
                for d in range(n_dims):
                    col_str += f"{col_coords[ci, d]:>10.4f}"
                summary_ca += col_str + "\n"

            result["summary"] = summary_ca

            # Biplot (if 2D)
            if n_dims >= 2:
                traces_ca = [
                    {
                        "type": "scatter",
                        "mode": "markers+text",
                        "x": row_coords[:, 0].tolist(),
                        "y": row_coords[:, 1].tolist(),
                        "text": row_labels_ca,
                        "textposition": "top center",
                        "name": f"{row_var_ca} (rows)",
                        "marker": {"color": "#4a9f6e", "size": 10, "symbol": "circle"},
                    },
                    {
                        "type": "scatter",
                        "mode": "markers+text",
                        "x": col_coords[:, 0].tolist(),
                        "y": col_coords[:, 1].tolist(),
                        "text": col_labels_ca,
                        "textposition": "bottom center",
                        "name": f"{col_var_ca} (columns)",
                        "marker": {"color": "#4a90d9", "size": 10, "symbol": "diamond"},
                    },
                ]
                result["plots"].append(
                    {
                        "title": f"Correspondence Analysis Biplot ({pct_inertia[0]:.1f}% + {pct_inertia[1]:.1f}% = {pct_inertia[0] + pct_inertia[1]:.1f}%)",
                        "data": traces_ca,
                        "layout": {
                            "height": 450,
                            "xaxis": {
                                "title": f"Dimension 1 ({pct_inertia[0]:.1f}%)",
                                "zeroline": True,
                                "zerolinecolor": "#5a6a5a",
                            },
                            "yaxis": {
                                "title": f"Dimension 2 ({pct_inertia[1]:.1f}%)",
                                "zeroline": True,
                                "zerolinecolor": "#5a6a5a",
                            },
                        },
                    }
                )

            # Scree plot of inertia
            result["plots"].append(
                {
                    "title": "Inertia Scree Plot",
                    "data": [
                        {
                            "x": list(range(1, len(inertia) + 1)),
                            "y": pct_inertia[: len(inertia)].tolist(),
                            "mode": "lines+markers",
                            "name": "% Inertia",
                            "marker": {"color": "#4a9f6e", "size": 8},
                            "line": {"color": "#4a9f6e", "width": 2},
                        }
                    ],
                    "layout": {"height": 280, "xaxis": {"title": "Dimension"}, "yaxis": {"title": "% of Inertia"}},
                }
            )

            result["guide_observation"] = (
                f"Correspondence analysis: χ²={chi2_ca:.1f} (p={p_chi2_ca:.4f}), {n_dims} dimensions explain {sum(pct_inertia[:n_dims]):.1f}% of inertia."
            )
            result["statistics"] = {
                "chi2": chi2_ca,
                "p_value": p_chi2_ca,
                "total_inertia": total_inertia,
                "n_dims": n_dims,
                "inertia": inertia[:n_dims].tolist(),
                "pct_inertia": pct_inertia[:n_dims].tolist(),
                "row_coords": {rl: row_coords[ri, :n_dims].tolist() for ri, rl in enumerate(row_labels_ca)},
                "col_coords": {cl: col_coords[ci, :n_dims].tolist() for ci, cl in enumerate(col_labels_ca)},
            }

        except Exception as e:
            result["summary"] = f"Correspondence analysis error: {str(e)}"

    elif analysis_id == "item_analysis":
        """
        Item Analysis — reliability assessment for multi-item scales/questionnaires.
        Computes Cronbach's alpha (overall and if-item-deleted), item-total correlations,
        inter-item correlation matrix. Standard tool for survey/psychometric validation.
        """
        items_ia = config.get("items") or config.get("variables", [])

        if not items_ia:
            items_ia = df.select_dtypes(include=[np.number]).columns.tolist()

        data_ia = df[items_ia].dropna()
        n_ia = len(data_ia)
        k_ia = len(items_ia)

        if k_ia < 2:
            result["summary"] = "Need at least 2 items for reliability analysis."
            return result

        try:
            X_ia = data_ia.values.astype(float)

            # Cronbach's alpha
            item_vars = np.var(X_ia, axis=0, ddof=1)
            total_var = np.var(X_ia.sum(axis=1), ddof=1)
            alpha_overall = (k_ia / (k_ia - 1)) * (1 - np.sum(item_vars) / total_var) if total_var > 0 else 0

            # Item statistics
            total_scores = X_ia.sum(axis=1)
            item_stats = []
            for i, item_name in enumerate(items_ia):
                item_mean = float(np.mean(X_ia[:, i]))
                item_std = float(np.std(X_ia[:, i], ddof=1))
                # Corrected item-total correlation (correlation with total minus this item)
                rest_total = total_scores - X_ia[:, i]
                corr_it = float(np.corrcoef(X_ia[:, i], rest_total)[0, 1]) if item_std > 0 else 0

                # Alpha if item deleted
                if k_ia > 2:
                    remaining = np.delete(X_ia, i, axis=1)
                    rem_item_vars = np.var(remaining, axis=0, ddof=1)
                    rem_total_var = np.var(remaining.sum(axis=1), ddof=1)
                    k_rem = k_ia - 1
                    alpha_deleted = (
                        (k_rem / (k_rem - 1)) * (1 - np.sum(rem_item_vars) / rem_total_var) if rem_total_var > 0 else 0
                    )
                else:
                    alpha_deleted = 0

                item_stats.append(
                    {
                        "item": item_name,
                        "mean": item_mean,
                        "std": item_std,
                        "corrected_item_total": corr_it,
                        "alpha_if_deleted": float(alpha_deleted),
                    }
                )

            # Inter-item correlation matrix
            corr_matrix_ia = np.corrcoef(X_ia.T)
            # Average inter-item correlation (off-diagonal)
            off_diag = corr_matrix_ia[np.triu_indices(k_ia, k=1)]
            avg_inter_item = float(np.mean(off_diag)) if len(off_diag) > 0 else 0

            # Standardized alpha (based on average inter-item correlation)
            std_alpha = (
                (k_ia * avg_inter_item) / (1 + (k_ia - 1) * avg_inter_item)
                if (1 + (k_ia - 1) * avg_inter_item) > 0
                else 0
            )

            # Summary
            summary_ia = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary_ia += "<<COLOR:title>>ITEM ANALYSIS (RELIABILITY)<</COLOR>>\n"
            summary_ia += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary_ia += f"<<COLOR:highlight>>Items:<</COLOR>> {k_ia}\n"
            summary_ia += f"<<COLOR:highlight>>N (complete cases):<</COLOR>> {n_ia}\n\n"

            summary_ia += "<<COLOR:text>>Overall Reliability:<</COLOR>>\n"
            alpha_color = "good" if alpha_overall >= 0.7 else ("warning" if alpha_overall >= 0.5 else "accent")
            summary_ia += f"  <<COLOR:{alpha_color}>>Cronbach's α = {alpha_overall:.4f}<</COLOR>>\n"
            summary_ia += f"  Standardized α = {std_alpha:.4f}\n"
            summary_ia += f"  Average inter-item correlation = {avg_inter_item:.4f}\n\n"

            if alpha_overall >= 0.9:
                summary_ia += "  <<COLOR:good>>Excellent reliability<</COLOR>>\n"
            elif alpha_overall >= 0.8:
                summary_ia += "  <<COLOR:good>>Good reliability<</COLOR>>\n"
            elif alpha_overall >= 0.7:
                summary_ia += "  <<COLOR:good>>Acceptable reliability<</COLOR>>\n"
            elif alpha_overall >= 0.6:
                summary_ia += "  <<COLOR:warning>>Questionable reliability<</COLOR>>\n"
            elif alpha_overall >= 0.5:
                summary_ia += "  <<COLOR:warning>>Poor reliability<</COLOR>>\n"
            else:
                summary_ia += "  <<COLOR:accent>>Unacceptable reliability<</COLOR>>\n"

            summary_ia += "\n<<COLOR:text>>Item Statistics:<</COLOR>>\n"
            summary_ia += f"{'Item':<25} {'Mean':>8} {'SD':>8} {'r(item-total)':>14} {'α if deleted':>12}\n"
            summary_ia += f"{'─' * 72}\n"
            for s in item_stats:
                flag = " <<COLOR:warning>>↑<</COLOR>>" if s["alpha_if_deleted"] > alpha_overall + 0.01 else ""
                summary_ia += f"{s['item']:<25} {s['mean']:>8.3f} {s['std']:>8.3f} {s['corrected_item_total']:>14.4f} {s['alpha_if_deleted']:>12.4f}{flag}\n"

            summary_ia += "\n<<COLOR:text>>↑ = removing this item would improve α<</COLOR>>\n"

            result["summary"] = summary_ia

            # Item-total correlation bar chart
            result["plots"].append(
                {
                    "title": "Corrected Item-Total Correlations",
                    "data": [
                        {
                            "type": "bar",
                            "x": [s["item"] for s in item_stats],
                            "y": [s["corrected_item_total"] for s in item_stats],
                            "marker": {
                                "color": [
                                    "#4a9f6e" if s["corrected_item_total"] >= 0.3 else "#d94a4a" for s in item_stats
                                ]
                            },
                        }
                    ],
                    "layout": {
                        "height": 300,
                        "xaxis": {"tickangle": -45},
                        "yaxis": {"title": "Corrected Item-Total r"},
                        "shapes": [
                            {
                                "type": "line",
                                "x0": -0.5,
                                "x1": k_ia - 0.5,
                                "y0": 0.3,
                                "y1": 0.3,
                                "line": {"color": "#e89547", "dash": "dash"},
                            }
                        ],
                    },
                }
            )

            # Alpha-if-deleted plot
            result["plots"].append(
                {
                    "title": "Cronbach's α if Item Deleted",
                    "data": [
                        {
                            "type": "bar",
                            "x": [s["item"] for s in item_stats],
                            "y": [s["alpha_if_deleted"] for s in item_stats],
                            "marker": {
                                "color": [
                                    "#d94a4a" if s["alpha_if_deleted"] > alpha_overall else "#4a9f6e"
                                    for s in item_stats
                                ]
                            },
                        },
                        {
                            "type": "scatter",
                            "mode": "lines",
                            "name": f"Current α ({alpha_overall:.3f})",
                            "x": [items_ia[0], items_ia[-1]],
                            "y": [alpha_overall, alpha_overall],
                            "line": {"color": "#e89547", "dash": "dash"},
                        },
                    ],
                    "layout": {"height": 300, "xaxis": {"tickangle": -45}, "yaxis": {"title": "Cronbach's α"}},
                }
            )

            # Inter-item correlation heatmap
            result["plots"].append(
                {
                    "title": "Inter-Item Correlation Matrix",
                    "data": [
                        {
                            "type": "heatmap",
                            "z": corr_matrix_ia.tolist(),
                            "x": items_ia,
                            "y": items_ia,
                            "colorscale": "RdBu",
                            "zmid": 0,
                            "text": [[f"{corr_matrix_ia[i, j]:.2f}" for j in range(k_ia)] for i in range(k_ia)],
                            "texttemplate": "%{text}",
                            "showscale": True,
                        }
                    ],
                    "layout": {"height": max(300, k_ia * 25)},
                }
            )

            n_weak = sum(1 for s in item_stats if s["corrected_item_total"] < 0.3)
            result["guide_observation"] = (
                f"Item analysis: α={alpha_overall:.3f} ({k_ia} items). {n_weak} items with weak item-total correlation (<0.3)."
            )
            result["statistics"] = {
                "cronbach_alpha": alpha_overall,
                "standardized_alpha": std_alpha,
                "avg_inter_item_correlation": avg_inter_item,
                "n_items": k_ia,
                "n_cases": n_ia,
                "item_stats": item_stats,
            }

        except Exception as e:
            result["summary"] = f"Item analysis error: {str(e)}"

    return result

"""ML Pipeline Engine — in-memory data operations for ML workflows.

Provides DataFrame-level operations for:
- Triage: clean data in memory (no file round-trip)
- Forge: augment small datasets with synthetic rows
- Training: wrapper around _auto_train with recipe capture
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def triage_clean_df(df, config=None):
    """Clean a DataFrame in memory using the scrub library.

    Args:
        df: Input DataFrame
        config: Optional dict with cleaning config keys:
            detect_outliers, handle_missing, normalize_factors,
            correct_types, case_style, drop_threshold

    Returns:
        (cleaned_df, cleaning_summary) tuple where cleaning_summary is a dict
    """
    from scrub import DataCleaner, CleaningConfig

    config = config or {}
    cleaning_config = CleaningConfig(
        detect_outliers=config.get("detect_outliers", True),
        handle_missing=config.get("handle_missing", True),
        normalize_factors=config.get("normalize_factors", True),
        correct_types=config.get("correct_types", True),
        case_style=config.get("case_style", "title"),
        drop_threshold=config.get("drop_threshold", 0.5),
    )

    if "domain_rules" in config:
        cleaning_config.domain_rules = config["domain_rules"]
    if "imputation_strategies" in config:
        cleaning_config.imputation_strategies = config["imputation_strategies"]

    cleaner = DataCleaner()
    df_clean, result = cleaner.clean(df, cleaning_config)

    summary = {
        "original_shape": list(df.shape),
        "cleaned_shape": list(df_clean.shape),
        "outliers_flagged": getattr(result.outliers, "count", 0) if result.outliers else 0,
        "missing_filled": getattr(result.missing, "total_filled", 0) if result.missing else 0,
        "columns_dropped": getattr(result.missing, "columns_dropped", []) if result.missing else [],
        "rows_dropped": getattr(result.missing, "rows_dropped", 0) if result.missing else 0,
        "warnings": getattr(result, "warnings", []),
    }

    logger.info(
        f"Triage: {df.shape} → {df_clean.shape}, "
        f"outliers={summary['outliers_flagged']}, missing_filled={summary['missing_filled']}"
    )

    return df_clean, summary


def forge_augment_df(df, n_synthetic, schema=None):
    """Augment a DataFrame with synthetic rows using Forge TabularGenerator.

    Auto-infers schema from existing data if not provided.

    Args:
        df: Original DataFrame
        n_synthetic: Number of synthetic rows to generate
        schema: Optional Forge schema dict. If None, inferred from df.

    Returns:
        (augmented_df, forge_report) tuple
    """
    from forge.generators.tabular import TabularGenerator

    if schema is None:
        schema = _infer_forge_schema(df)

    generator = TabularGenerator(schema=schema)
    synthetic_records = generator.generate(n_synthetic)

    synthetic_df = pd.DataFrame(synthetic_records)

    # Align columns to match original DataFrame
    for col in df.columns:
        if col not in synthetic_df.columns:
            synthetic_df[col] = None
    synthetic_df = synthetic_df[[c for c in df.columns if c in synthetic_df.columns]]

    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)

    report = {
        "original_rows": len(df),
        "synthetic_rows": n_synthetic,
        "total_rows": len(augmented_df),
        "schema_inferred": schema is None,
        "columns_generated": list(schema.keys()),
    }

    logger.info(f"Forge: {len(df)} + {n_synthetic} synthetic → {len(augmented_df)} rows")

    return augmented_df, report


def _infer_forge_schema(df):
    """Infer a Forge-compatible schema from a DataFrame.

    Maps DataFrame dtypes to Forge field types with appropriate constraints.
    """
    schema = {}
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            schema[col] = {"type": "string"}
            continue

        dtype = df[col].dtype

        if np.issubdtype(dtype, np.integer):
            schema[col] = {
                "type": "int",
                "constraints": {
                    "min": int(series.min()),
                    "max": int(series.max()),
                },
            }
        elif np.issubdtype(dtype, np.floating):
            schema[col] = {
                "type": "float",
                "constraints": {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "precision": 2,
                },
            }
        elif dtype == bool or (series.nunique() == 2 and set(series.unique()) <= {True, False, 0, 1}):
            schema[col] = {
                "type": "bool",
                "constraints": {
                    "true_probability": float(series.astype(bool).mean()),
                },
            }
        elif series.nunique() <= 20:
            values = series.unique().tolist()
            # Convert numpy types to Python types for JSON serialization
            values = [str(v) if not isinstance(v, (str, int, float, bool)) else v for v in values]
            schema[col] = {
                "type": "category",
                "constraints": {
                    "values": values,
                },
            }
        else:
            schema[col] = {
                "type": "string",
                "constraints": {
                    "min_length": 3,
                    "max_length": min(int(series.str.len().max()) if dtype == object else 50, 100),
                },
            }

    return schema


def train_with_recipe(df, target, config=None):
    """Train a model and capture the full training recipe for reproducibility.

    Wraps _auto_train but returns a recipe dict that can reproduce the result.

    Args:
        df: Full DataFrame including target column
        target: Target column name
        config: Optional dict with task_type override

    Returns:
        (model, metrics, importances, task, X_test, y_test, y_pred, recipe)
    """
    from .dsw_views import _clean_for_ml, _auto_train

    config = config or {}

    X, y, label_map = _clean_for_ml(df, target)

    model, metrics, importances, task, X_test, y_test, y_pred = _auto_train(
        X, y, task=config.get("task_type")
    )

    recipe = {
        "features": list(X.columns),
        "target": target,
        "task_type": task,
        "model_class": type(model).__name__,
        "hyperparams": model.get_params() if hasattr(model, "get_params") else {},
        "test_size": 0.2,
        "random_state": 42,
        "label_map": label_map,
        "input_shape": list(df.shape),
        "training_shape": [X.shape[0], X.shape[1]],
    }

    return model, metrics, importances, task, X_test, y_test, y_pred, recipe

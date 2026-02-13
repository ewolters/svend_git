"""Triage (Data Cleaning) API views."""

import io
import json
import logging
import tempfile
import time
import uuid
from datetime import datetime

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated, require_auth
from .models import TriageResult, AgentLog

logger = logging.getLogger(__name__)


def log_agent_action(user, agent, action, latency_ms=None, success=True, error_message="", metadata=None):
    """Log an agent action to the database."""
    try:
        AgentLog.objects.create(
            user=user if user.is_authenticated else None,
            agent=agent,
            action=action,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
            metadata=json.dumps(metadata) if metadata else "",
        )
    except Exception as e:
        logger.warning(f"Failed to log agent action: {e}")


@csrf_exempt
@require_http_methods(["POST"])
@gated
def triage_clean(request):
    """
    Clean uploaded CSV data.

    POST /api/triage/clean/
    - file: CSV file (multipart)
    - config: JSON config (optional)

    Returns cleaned data + report.
    """
    # Get uploaded file
    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    uploaded_file = request.FILES["file"]
    if not uploaded_file.name.endswith(".csv"):
        return JsonResponse({"error": "Only CSV files are supported"}, status=400)

    # Parse optional config
    config_json = request.POST.get("config", "{}")
    try:
        config_data = json.loads(config_json)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid config JSON"}, status=400)

    try:
        import pandas as pd
        from scrub import DataCleaner, CleaningConfig

        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Build config
        config = CleaningConfig(
            detect_outliers=config_data.get("detect_outliers", True),
            handle_missing=config_data.get("handle_missing", True),
            normalize_factors=config_data.get("normalize_factors", True),
            correct_types=config_data.get("correct_types", True),
            case_style=config_data.get("case_style", "title"),
            drop_threshold=config_data.get("drop_threshold", 0.5),
        )

        # Add domain rules if provided
        if "domain_rules" in config_data:
            config.domain_rules = config_data["domain_rules"]

        # Add imputation strategies if provided
        if "imputation_strategies" in config_data:
            config.imputation_strategies = config_data["imputation_strategies"]

        # Clean data
        cleaner = DataCleaner()
        df_clean, result = cleaner.clean(df, config)

        # Generate job ID for tracking
        job_id = str(uuid.uuid4())[:8]

        # Convert cleaned data to CSV
        csv_buffer = io.StringIO()
        df_clean.to_csv(csv_buffer, index=False)
        cleaned_csv = csv_buffer.getvalue()

        # Store in database for persistence
        TriageResult.objects.create(
            id=job_id,
            user=request.user,
            original_filename=uploaded_file.name,
            cleaned_csv=cleaned_csv,
            report_markdown=result.to_markdown(),
            summary_json=json.dumps(result.to_dict()),
        )

        # Log the action
        log_agent_action(
            user=request.user,
            agent="triage",
            action="clean",
            success=True,
            metadata={"job_id": job_id, "rows": result.original_shape[0]},
        )

        # Track usage
        request.user.increment_queries()

        return JsonResponse({
            "success": True,
            "job_id": job_id,
            "summary": result.to_dict(),
            "report": result.to_markdown(),
            "preview": {
                "original_rows": result.original_shape[0],
                "original_cols": result.original_shape[1],
                "cleaned_rows": result.cleaned_shape[0],
                "cleaned_cols": result.cleaned_shape[1],
            },
            "outliers_flagged": result.outliers.count if result.outliers else 0,
            "missing_filled": result.missing.total_filled if result.missing else 0,
            "warnings": result.warnings,
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def triage_download(request, job_id):
    """
    Download cleaned CSV data.

    GET /api/triage/{job_id}/download/
    """
    try:
        result = TriageResult.objects.get(id=job_id, user=request.user)
    except TriageResult.DoesNotExist:
        return JsonResponse({"error": "Job not found or expired"}, status=404)

    response = HttpResponse(result.cleaned_csv, content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="cleaned_{job_id}.csv"'
    return response


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def triage_report(request, job_id):
    """
    Get cleaning report.

    GET /api/triage/{job_id}/report/
    """
    try:
        result = TriageResult.objects.get(id=job_id, user=request.user)
    except TriageResult.DoesNotExist:
        return JsonResponse({"error": "Job not found or expired"}, status=404)

    return JsonResponse({"report": result.report_markdown})


@csrf_exempt
@require_http_methods(["POST"])
@gated
def triage_preview(request):
    """
    Preview data issues without cleaning.

    POST /api/triage/preview/
    - file: CSV file (multipart)

    Returns detected issues without modifying data.
    """

    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    uploaded_file = request.FILES["file"]

    try:
        import pandas as pd
        from scrub import OutlierDetector, MissingHandler, TypeInferrer
        from scrub.outliers import OutlierMethod

        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Analyze without modifying
        issues = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing": {},
            "outliers": {},
            "type_issues": [],
        }

        # Check missing data
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues["missing"][col] = {
                    "count": int(missing_count),
                    "percent": float(missing_count / len(df) * 100),
                }

        # Check outliers (numeric columns only)
        detector = OutlierDetector()
        outlier_result = detector.detect(df, methods=[OutlierMethod.IQR])
        if outlier_result.count > 0:
            for col_result in outlier_result.by_column:
                if col_result.outlier_count > 0:
                    issues["outliers"][col_result.column] = {
                        "count": col_result.outlier_count,
                        "indices": col_result.indices[:10],  # First 10
                    }

        # Infer types
        inferrer = TypeInferrer()
        for col in df.columns:
            inferred = inferrer._infer_type(df[col])
            actual = str(df[col].dtype)
            if inferred != actual and inferred != "object":
                issues["type_issues"].append({
                    "column": col,
                    "current": actual,
                    "suggested": inferred,
                })

        # Detect potential biases
        bias_warnings = _detect_data_biases(df)

        # Track query usage
        request.user.increment_queries()

        return JsonResponse({
            "success": True,
            "issues": issues,
            "bias_warnings": bias_warnings,
            "recommendations": _generate_recommendations(issues, bias_warnings),
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def _detect_data_biases(df) -> list:
    """
    Detect potential biases in dataset.

    Checks for:
    - Proxy variables for protected classes
    - Class imbalance
    - Missing data patterns that correlate with sensitive features
    """
    warnings = []

    # Protected/sensitive column patterns
    sensitive_patterns = [
        "gender", "sex", "race", "ethnicity", "religion", "age",
        "disability", "marital", "nationality", "origin", "zip", "zipcode",
        "postal", "income", "salary", "neighborhood",
    ]

    # Check for sensitive columns
    for col in df.columns:
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        for pattern in sensitive_patterns:
            if pattern in col_lower:
                warnings.append({
                    "type": "sensitive_feature",
                    "column": col,
                    "message": f"'{col}' may contain sensitive/protected information. "
                               f"Consider whether it should be used in modeling, "
                               f"or if it could serve as a proxy for discrimination.",
                    "severity": "high",
                })
                break

    # Check for class imbalance in categorical columns
    for col in df.columns:
        if df[col].dtype in ['object', 'bool', 'category'] or df[col].nunique() <= 10:
            if df[col].nunique() >= 2:
                value_counts = df[col].value_counts(normalize=True)
                max_ratio = value_counts.iloc[0]
                min_ratio = value_counts.iloc[-1]
                if max_ratio > 0.9:
                    warnings.append({
                        "type": "class_imbalance",
                        "column": col,
                        "message": f"'{col}' has severe class imbalance "
                                   f"({max_ratio:.1%} majority vs {min_ratio:.1%} minority). "
                                   f"Models may be biased toward the majority class.",
                        "severity": "medium",
                    })

    # Check for correlated missing data (systematic collection bias)
    missing_cols = df.columns[df.isnull().any()].tolist()
    if len(missing_cols) > 1:
        for i, col1 in enumerate(missing_cols[:5]):  # Limit checks
            for col2 in missing_cols[i+1:i+4]:
                both_missing = (df[col1].isnull() & df[col2].isnull()).mean()
                col1_missing = df[col1].isnull().mean()
                col2_missing = df[col2].isnull().mean()
                expected = col1_missing * col2_missing

                if col1_missing > 0 and col2_missing > 0:
                    if both_missing > 2 * expected and both_missing > 0.05:
                        warnings.append({
                            "type": "correlated_missing",
                            "columns": [col1, col2],
                            "message": f"Missing data in '{col1}' and '{col2}' are correlated. "
                                       f"This could indicate systematic bias in data collection.",
                            "severity": "medium",
                        })

    return warnings


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def list_datasets(request):
    """
    List user's cleaned datasets from Triage.

    GET /api/triage/datasets/
    """
    datasets = TriageResult.objects.filter(user=request.user).order_by("-created_at")[:20]

    return JsonResponse({
        "datasets": [
            {
                "id": d.id,
                "filename": d.original_filename,
                "created_at": d.created_at.isoformat(),
                "summary": json.loads(d.summary_json) if d.summary_json else {},
            }
            for d in datasets
        ]
    })


@csrf_exempt
@require_http_methods(["GET"])
@require_auth
def load_dataset(request, job_id):
    """
    Load a cleaned dataset for Analysis Workbench.

    GET /api/triage/{job_id}/load/

    Returns columns with dtypes and preview data.
    """
    try:
        result = TriageResult.objects.get(id=job_id, user=request.user)
    except TriageResult.DoesNotExist:
        return JsonResponse({"error": "Dataset not found"}, status=404)

    try:
        import pandas as pd
        import numpy as np
        from io import StringIO

        # Parse the stored CSV
        df = pd.read_csv(StringIO(result.cleaned_csv))

        # Determine column types
        columns = []
        for col in df.columns:
            dtype = df[col].dtype
            if np.issubdtype(dtype, np.number):
                col_type = "numeric"
            elif np.issubdtype(dtype, np.datetime64):
                col_type = "datetime"
            else:
                col_type = "text"

            columns.append({
                "name": col,
                "dtype": col_type,
            })

        # Generate preview (first 100 rows)
        preview = df.head(100).replace({np.nan: None}).to_dict(orient="records")

        # Preload LLM in background for Analysis Workbench assistant
        _preload_llm_background()

        return JsonResponse({
            "id": job_id,
            "filename": result.original_filename,
            "rows": len(df),
            "columns": columns,
            "preview": preview,
        })

    except Exception as e:
        return JsonResponse({"error": f"Failed to load dataset: {str(e)}"}, status=500)


def _preload_llm_background():
    """Start loading the LLM in a background thread if not already loaded."""
    import threading
    try:
        from . import views as agent_views
        if not agent_views._shared_llm_loaded:
            logger.info("Preloading LLM in background (triage data loaded)")
            def load():
                try:
                    agent_views.get_shared_llm()
                    logger.info("LLM preload completed")
                except Exception as e:
                    logger.error(f"LLM preload failed: {e}")
            threading.Thread(target=load, daemon=True).start()
    except Exception as e:
        logger.warning(f"Could not trigger LLM preload: {e}")


def _generate_recommendations(issues: dict, bias_warnings: list = None) -> list[str]:
    """Generate cleaning recommendations based on detected issues."""
    recs = []

    # Bias recommendations (first - most important)
    if bias_warnings:
        sensitive = [w for w in bias_warnings if w["type"] == "sensitive_feature"]
        imbalanced = [w for w in bias_warnings if w["type"] == "class_imbalance"]

        if sensitive:
            cols = [w["column"] for w in sensitive]
            recs.append(f"⚠️ BIAS ALERT: Sensitive features detected ({', '.join(cols[:3])}). "
                       f"Review whether these should be used in modeling.")

        if imbalanced:
            cols = [w["column"] for w in imbalanced]
            recs.append(f"⚠️ Class imbalance in {', '.join(cols[:3])}. "
                       f"Consider SMOTE, class weights, or stratified sampling.")

    # Missing data recommendations
    if issues["missing"]:
        high_missing = [col for col, data in issues["missing"].items()
                       if data["percent"] > 50]
        if high_missing:
            recs.append(f"Consider dropping columns with >50% missing: {', '.join(high_missing)}")

        low_missing = [col for col, data in issues["missing"].items()
                      if data["percent"] <= 50]
        if low_missing:
            recs.append(f"Imputation recommended for: {', '.join(low_missing[:5])}")

    # Outlier recommendations
    if issues["outliers"]:
        outlier_cols = list(issues["outliers"].keys())
        recs.append(f"Review outliers in: {', '.join(outlier_cols[:5])}")

    # Type recommendations
    if issues["type_issues"]:
        type_cols = [t["column"] for t in issues["type_issues"]]
        recs.append(f"Type conversion suggested for: {', '.join(type_cols[:5])}")

    if not recs:
        recs.append("Data looks clean! Minor optimizations may still apply.")

    return recs

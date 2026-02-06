"""DSW (Decision Science Workbench) API views."""

import json
import logging
import time
import uuid
import tempfile
from pathlib import Path
from collections import OrderedDict
from threading import Lock

from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from django.conf import settings
from .models import DSWResult, AgentLog, SavedModel
from accounts.permissions import gated, require_auth, require_enterprise

logger = logging.getLogger(__name__)

# Temporary cache for trained models (max 100 per user, expires after 1 hour)
# Structure: {user_id: {model_key: {'model': model, 'metadata': {...}, 'timestamp': time}}}
_model_cache = {}
_model_cache_lock = Lock()
MODEL_CACHE_MAX_SIZE = 100
MODEL_CACHE_EXPIRY = 3600  # 1 hour


def cache_model(user_id, model_key, model, metadata):
    """Cache a trained model temporarily for later saving."""
    with _model_cache_lock:
        if user_id not in _model_cache:
            _model_cache[user_id] = OrderedDict()

        # Clean expired entries
        now = time.time()
        expired = [k for k, v in _model_cache[user_id].items() if now - v['timestamp'] > MODEL_CACHE_EXPIRY]
        for k in expired:
            del _model_cache[user_id][k]

        # Limit cache size per user
        while len(_model_cache[user_id]) >= MODEL_CACHE_MAX_SIZE:
            _model_cache[user_id].popitem(last=False)

        _model_cache[user_id][model_key] = {
            'model': model,
            'metadata': metadata,
            'timestamp': now
        }


def get_cached_model(user_id, model_key):
    """Retrieve a cached model."""
    with _model_cache_lock:
        if user_id in _model_cache and model_key in _model_cache[user_id]:
            entry = _model_cache[user_id][model_key]
            if time.time() - entry['timestamp'] < MODEL_CACHE_EXPIRY:
                return entry
            else:
                del _model_cache[user_id][model_key]
    return None


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


def _preload_llm_background():
    """Start loading the LLM in a background thread if not already loaded."""
    import threading
    try:
        # Quick CUDA check before attempting load
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available - skipping LLM preload")
                return
        except ImportError:
            logger.warning("PyTorch not available - skipping LLM preload")
            return

        from . import views as agent_views
        if not agent_views._shared_llm_loaded:
            logger.info("Preloading LLM in background (data loaded)")
            def load():
                try:
                    agent_views.get_shared_llm()
                    logger.info("LLM preload completed")
                except Exception as e:
                    logger.error(f"LLM preload failed: {e}")
            threading.Thread(target=load, daemon=True).start()
    except Exception as e:
        logger.warning(f"Could not trigger LLM preload: {e}")


def save_model_to_disk(user, model, model_type, dsw_result_id, name=None, metrics=None, features=None, target=None):
    """Save a trained model to disk and create database record.

    Returns SavedModel instance or None if failed.
    """
    import pickle

    try:
        # Create user's model directory
        models_dir = Path(settings.MEDIA_ROOT) / "models" / str(user.id)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        model_id = uuid.uuid4()
        model_filename = f"{model_id}.pkl"
        model_path = models_dir / model_filename

        # Save model with pickle
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Create database record
        saved_model = SavedModel.objects.create(
            id=model_id,
            user=user,
            name=name or f"{model_type} - {dsw_result_id[:8]}",
            model_type=model_type,
            model_path=str(model_path),
            dsw_result_id=dsw_result_id,
            metrics=json.dumps(metrics) if metrics else "",
            feature_names=json.dumps(features) if features else "",
            target_name=target or "",
        )

        logger.info(f"Saved model {model_id} for user {user.username}")
        return saved_model

    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return None


@csrf_exempt
@require_http_methods(["POST"])
@gated
def dsw_from_intent(request):
    """Run DSW pipeline from intent (zero data)."""
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    intent = data.get("intent", "").strip()
    domain = data.get("domain")
    n_records = int(data.get("n_records", 500))
    priority = data.get("priority", "balanced")

    if not intent:
        return JsonResponse({"error": "Intent is required"}, status=400)

    try:
        import sys
        sys.path.insert(0, "/home/eric/Desktop/agents")

        from dsw import DecisionScienceWorkbench

        dsw = DecisionScienceWorkbench()
        result = dsw.from_intent(
            intent=intent,
            domain=domain if domain else None,
            n_records=n_records,
            priority=priority,
        )

        # Store result in database
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"

        # Save synthetic data to file
        data_dir = Path(settings.MEDIA_ROOT) / "dsw_data" / str(request.user.id)
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / f"{result_id}.csv"
        result.synthetic_data.to_csv(data_path, index=False)

        result_data = {
            "schema": {
                "name": result.schema.name,
                "intent": result.schema.intent,
                "domain": result.schema.domain,
                "features": [f.name for f in result.schema.features],
                "target": result.schema.target_name,
            },
            "data_shape": list(result.synthetic_data.shape),
            "data_path": str(data_path),
            "model_type": result.model_type,
            "metrics": result.metrics,
            "pipeline": result.pipeline_steps,
            "warnings": result.warnings,
            "summary": result.summary(),
        }

        DSWResult.objects.create(
            id=result_id,
            user=request.user,
            result_type="from_intent",
            data=json.dumps(result_data),
        )

        # Save trained model
        saved_model = None
        if result.model:
            saved_model = save_model_to_disk(
                user=request.user,
                model=result.model,
                model_type=result.model_type,
                dsw_result_id=result_id,
                name=f"Model: {intent[:50]}",
                metrics=result.metrics,
                features=[f.name for f in result.schema.features],
                target=result.schema.target_name,
            )

        log_agent_action(
            user=request.user,
            agent="dsw",
            action="from_intent",
            success=True,
            metadata={"result_id": result_id, "domain": domain},
        )

        # Track usage
        request.user.increment_queries()

        response_data = {
            "result_id": result_id,
            **result_data,
        }
        if saved_model:
            response_data["model_id"] = str(saved_model.id)

        return JsonResponse(response_data)

    except ImportError as e:
        # DSW module not available - return mock result for testing
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"

        mock_result = {
            "schema": {
                "name": f"Schema for: {intent[:50]}",
                "intent": intent,
                "domain": domain or "auto-detected",
                "features": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
                "target": "target",
            },
            "data_shape": [n_records, 6],
            "model_type": "RandomForest",
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1": 0.85,
                "auc_roc": 0.91,
            },
            "pipeline": ["schema_generation", "data_synthesis", "data_cleaning", "model_training"],
            "warnings": ["This is a mock result - DSW module not available"],
            "summary": f"Generated {n_records} synthetic records based on intent: {intent}. Model achieved 85% accuracy.",
        }

        DSWResult.objects.create(
            id=result_id,
            user=request.user,
            result_type="from_intent_mock",
            data=json.dumps(mock_result),
        )

        log_agent_action(
            user=request.user,
            agent="dsw",
            action="from_intent_mock",
            success=True,
            metadata={"result_id": result_id, "mock": True},
        )

        # Track usage
        request.user.increment_queries()

        return JsonResponse({"result_id": result_id, **mock_result})

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"DSW from_intent failed: {e}")

        log_agent_action(
            user=request.user,
            agent="dsw",
            action="from_intent",
            success=False,
            error_message=str(e)[:500],
            metadata={"intent": intent[:100], "domain": domain},
        )

        return JsonResponse({
            "error": f"Pipeline failed: {str(e)[:200]}",
            "suggestion": "Try rephrasing your intent or use a different domain.",
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@gated
def dsw_from_data(request):
    """Run DSW pipeline from uploaded data."""
    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    file = request.FILES["file"]
    target = request.POST.get("target", "").strip()
    intent = request.POST.get("intent", "")
    priority = request.POST.get("priority", "balanced")

    if not target:
        return JsonResponse({"error": "Target column is required"}, status=400)

    try:
        import sys
        import pandas as pd
        sys.path.insert(0, "/home/eric/Desktop/agents")

        # Read uploaded CSV
        df = pd.read_csv(file)

        if target not in df.columns:
            return JsonResponse({"error": f'Target column "{target}" not found in data'}, status=400)

        from dsw import DecisionScienceWorkbench

        dsw = DecisionScienceWorkbench()
        result = dsw.from_data(
            data=df,
            target=target,
            intent=intent,
            priority=priority,
        )

        # Store result in database
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"
        result_data = {
            "original_shape": list(result.original_data.shape),
            "cleaned_shape": list(result.cleaned_data.shape),
            "model_type": result.model_type,
            "metrics": result.metrics,
            "feature_importance": result.feature_importance[:10],
            "pipeline": result.pipeline_steps,
            "warnings": result.warnings,
            "summary": result.summary(),
        }

        DSWResult.objects.create(
            id=result_id,
            user=request.user,
            result_type="from_data",
            data=json.dumps(result_data),
        )

        # Save trained model
        saved_model = None
        if result.model:
            features = [c for c in df.columns if c != target]
            saved_model = save_model_to_disk(
                user=request.user,
                model=result.model,
                model_type=result.model_type,
                dsw_result_id=result_id,
                name=f"Model: {target} prediction",
                metrics=result.metrics,
                features=features,
                target=target,
            )

        log_agent_action(
            user=request.user,
            agent="dsw",
            action="from_data",
            success=True,
            metadata={"result_id": result_id, "rows": df.shape[0]},
        )

        # Track usage
        request.user.increment_queries()

        response_data = {"result_id": result_id, **result_data}
        if saved_model:
            response_data["model_id"] = str(saved_model.id)

        return JsonResponse(response_data)

    except ImportError as e:
        # DSW module not available - return mock result
        import pandas as pd

        df = pd.read_csv(file)
        result_id = f"dsw_{uuid.uuid4().hex[:8]}"

        mock_result = {
            "original_shape": list(df.shape),
            "cleaned_shape": [int(df.shape[0] * 0.95), df.shape[1]],
            "model_type": "GradientBoosting",
            "metrics": {
                "accuracy": 0.82,
                "precision": 0.80,
                "recall": 0.84,
                "f1": 0.82,
            },
            "feature_importance": [
                {"feature": col, "importance": 1.0 / len(df.columns)}
                for col in df.columns[:10] if col != target
            ],
            "pipeline": ["data_loading", "data_cleaning", "feature_engineering", "model_training"],
            "warnings": ["This is a mock result - DSW module not available"],
            "summary": f"Processed {df.shape[0]} rows with {df.shape[1]} columns. Target: {target}. Model achieved 82% accuracy.",
        }

        DSWResult.objects.create(
            id=result_id,
            user=request.user,
            result_type="from_data_mock",
            data=json.dumps(mock_result),
        )

        log_agent_action(
            user=request.user,
            agent="dsw",
            action="from_data_mock",
            success=True,
            metadata={"result_id": result_id, "mock": True},
        )

        # Track usage
        request.user.increment_queries()

        return JsonResponse({"result_id": result_id, **mock_result})

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"DSW from_data failed: {e}")

        log_agent_action(
            user=request.user,
            agent="dsw",
            action="from_data",
            success=False,
            error_message=str(e)[:500],
            metadata={"target": target, "intent": intent[:100] if intent else None},
        )

        return JsonResponse({
            "error": f"Pipeline failed: {str(e)[:200]}",
            "suggestion": "Check that your data is properly formatted and the target column exists.",
        }, status=500)


@require_http_methods(["GET"])
@require_auth
def dsw_download(request, result_id, file_type):
    """Download DSW result files."""
    # Try to get from database
    try:
        db_result = DSWResult.objects.get(id=result_id, user=request.user)
        result = json.loads(db_result.data)
    except DSWResult.DoesNotExist:
        return JsonResponse({"error": "Result not found"}, status=404)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            if file_type == "report":
                if hasattr(result, "full_report"):
                    content = result.full_report()
                elif hasattr(result, "report"):
                    content = result.report.to_markdown()
                elif isinstance(result, dict) and "summary" in result:
                    content = f"# DSW Report\n\n{result['summary']}\n\n## Metrics\n\n"
                    if "metrics" in result:
                        for k, v in result["metrics"].items():
                            content += f"- {k}: {v}\n"
                else:
                    return JsonResponse({"error": "No report available"}, status=400)

                path = tmpdir / "report.md"
                path.write_text(content)
                return FileResponse(open(path, "rb"), as_attachment=True, filename="report.md")

            elif file_type == "code":
                if hasattr(result, "deployment_code"):
                    code = result.deployment_code
                elif hasattr(result, "code"):
                    code = result.code
                else:
                    # Generate template code
                    code = """# Generated by SVEND DSW Pipeline
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data.csv')

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy:.3f}')
"""

                path = tmpdir / "train.py"
                path.write_text(code)
                return FileResponse(open(path, "rb"), as_attachment=True, filename="train.py")

            elif file_type == "data":
                # Check for saved data file path
                data_path = result.get("data_path")
                if data_path and Path(data_path).exists():
                    return FileResponse(
                        open(data_path, "rb"),
                        as_attachment=True,
                        filename="data.csv"
                    )
                else:
                    return JsonResponse({"error": "No data available"}, status=400)

            elif file_type == "model":
                # Find saved model for this DSW result
                try:
                    saved_model = SavedModel.objects.get(dsw_result_id=result_id, user=request.user)
                    model_path = Path(saved_model.model_path)
                    if model_path.exists():
                        return FileResponse(
                            open(model_path, "rb"),
                            as_attachment=True,
                            filename=f"{saved_model.name.replace(' ', '_')}.pkl"
                        )
                    else:
                        return JsonResponse({"error": "Model file not found"}, status=404)
                except SavedModel.DoesNotExist:
                    return JsonResponse({"error": "No saved model for this result"}, status=404)

            else:
                return JsonResponse({"error": f"Unknown file type: {file_type}"}, status=400)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
@require_auth
def list_models(request):
    """List user's saved models."""
    models = SavedModel.objects.filter(user=request.user)
    return JsonResponse({
        "models": [
            {
                "id": str(m.id),
                "name": m.name,
                "description": m.description,
                "model_type": m.model_type,
                "dsw_result_id": m.dsw_result_id,
                "metrics": json.loads(m.metrics) if m.metrics else None,
                "features": json.loads(m.feature_names) if m.feature_names else None,
                "target": m.target_name,
                "created_at": m.created_at.isoformat(),
            }
            for m in models
        ]
    })


@csrf_exempt
@require_http_methods(["POST"])
@require_auth
def save_model_from_cache(request):
    """Save a trained model from the cache to permanent storage."""
    try:
        data = json.loads(request.body)
        model_key = data.get("model_key")
        name = data.get("name", "Untitled Model")
        description = data.get("description", "")

        if not model_key:
            return JsonResponse({"error": "model_key is required"}, status=400)

        # Get model from cache
        cached = get_cached_model(request.user.id, model_key)
        if not cached:
            return JsonResponse({"error": "Model not found in cache. It may have expired."}, status=404)

        model = cached['model']
        metadata = cached['metadata']

        # Save to disk
        saved = save_model_to_disk(
            user=request.user,
            model=model,
            model_type=metadata.get('model_type', 'unknown'),
            dsw_result_id=model_key,
            name=name,
            metrics=metadata.get('metrics'),
            features=metadata.get('features'),
            target=metadata.get('target')
        )

        if saved:
            # Update description
            if description:
                saved.description = description
                saved.save()

            return JsonResponse({
                "success": True,
                "model_id": str(saved.id),
                "name": saved.name,
                "message": f"Model '{name}' saved successfully"
            })
        else:
            return JsonResponse({"error": "Failed to save model"}, status=500)

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@require_http_methods(["GET"])
@require_auth
def download_model(request, model_id):
    """Download a saved model by ID."""
    try:
        saved_model = SavedModel.objects.get(id=model_id, user=request.user)
        model_path = Path(saved_model.model_path)

        if not model_path.exists():
            return JsonResponse({"error": "Model file not found"}, status=404)

        return FileResponse(
            open(model_path, "rb"),
            as_attachment=True,
            filename=f"{saved_model.name.replace(' ', '_')}.pkl"
        )

    except SavedModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)


@csrf_exempt
@require_http_methods(["DELETE"])
@require_auth
def delete_model(request, model_id):
    """Delete a saved model."""
    try:
        saved_model = SavedModel.objects.get(id=model_id, user=request.user)
        model_path = Path(saved_model.model_path)

        # Delete file
        if model_path.exists():
            model_path.unlink()

        # Delete record
        saved_model.delete()

        return JsonResponse({"success": True})

    except SavedModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)


@csrf_exempt
@require_http_methods(["POST"])
@gated
def run_model(request, model_id):
    """Run inference with a saved model."""
    try:
        import pickle
        import pandas as pd

        saved_model = SavedModel.objects.get(id=model_id, user=request.user)
        model_path = Path(saved_model.model_path)

        if not model_path.exists():
            return JsonResponse({"error": "Model file not found"}, status=404)

        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Get input data
        if "file" in request.FILES:
            df = pd.read_csv(request.FILES["file"])
        elif request.body:
            data = json.loads(request.body)
            if "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                return JsonResponse({"error": "No input data provided"}, status=400)
        else:
            return JsonResponse({"error": "No input data provided"}, status=400)

        # Run prediction
        features = json.loads(saved_model.feature_names) if saved_model.feature_names else list(df.columns)
        X = df[features] if all(f in df.columns for f in features) else df

        predictions = model.predict(X)

        # Get probabilities if available
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(X).tolist()
            except Exception:
                pass

        # Track query usage
        request.user.increment_queries()

        return JsonResponse({
            "predictions": predictions.tolist(),
            "probabilities": probabilities,
            "count": len(predictions),
        })

    except SavedModel.DoesNotExist:
        return JsonResponse({"error": "Model not found"}, status=404)
    except Exception as e:
        logger.exception("Model inference error")
        return JsonResponse({"error": str(e)}, status=500)


# =============================================================================
# Scrub Endpoints - Standalone Data Cleaning
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@gated
def scrub_data(request):
    """
    Clean uploaded data using Scrub.

    POST with file upload or JSON data.
    Returns cleaned data + detailed report.
    """
    import pandas as pd
    import sys
    sys.path.insert(0, "/home/eric/kjerne/services/svend/agents/agents")

    try:
        from scrub import DataCleaner, CleaningConfig

        # Get data
        if "file" in request.FILES:
            df = pd.read_csv(request.FILES["file"])
        elif request.body:
            data = json.loads(request.body)
            if "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                return JsonResponse({"error": "No data provided"}, status=400)
        else:
            return JsonResponse({"error": "No data provided"}, status=400)

        # Get config from request
        config_data = {}
        if request.body:
            try:
                body = json.loads(request.body)
                config_data = body.get("config", {})
            except:
                pass

        config = CleaningConfig(
            detect_outliers=config_data.get("detect_outliers", True),
            handle_missing=config_data.get("handle_missing", True),
            normalize_factors=config_data.get("normalize_factors", True),
            correct_types=config_data.get("correct_types", True),
            domain_rules=config_data.get("domain_rules", {}),
        )

        # Clean
        cleaner = DataCleaner()
        df_clean, result = cleaner.clean(df, config)

        # Track query usage
        request.user.increment_queries()

        return JsonResponse({
            "success": True,
            "original_shape": list(result.original_shape),
            "cleaned_shape": list(result.cleaned_shape),
            "outliers_flagged": result.outliers.count if result.outliers else 0,
            "missing_filled": result.missing.total_filled if result.missing else 0,
            "normalizations": result.normalization.total_changes if result.normalization else 0,
            "warnings": result.warnings,
            "report_markdown": result.to_markdown(),
            "report_summary": result.summary(),
            "data": df_clean.to_dict(orient="records"),
        })

    except Exception as e:
        logger.exception("Scrub error")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@gated
def scrub_analyze(request):
    """
    Analyze data without modifying it.

    Returns analysis of what would be cleaned + potential biases.
    """
    import pandas as pd
    import sys
    sys.path.insert(0, "/home/eric/kjerne/services/svend/agents/agents")

    try:
        from scrub import DataCleaner

        # Get data
        if "file" in request.FILES:
            df = pd.read_csv(request.FILES["file"])
        elif request.body:
            data = json.loads(request.body)
            if "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                return JsonResponse({"error": "No data provided"}, status=400)
        else:
            return JsonResponse({"error": "No data provided"}, status=400)

        cleaner = DataCleaner()
        analysis = cleaner.analyze(df)

        # Add bias detection
        bias_warnings = _detect_data_biases(df)
        analysis["bias_warnings"] = bias_warnings

        return JsonResponse({
            "success": True,
            "analysis": analysis,
        })

    except Exception as e:
        logger.exception("Scrub analyze error")
        return JsonResponse({"error": str(e)}, status=500)


def _detect_data_biases(df) -> list:
    """
    Detect potential biases in dataset.

    Checks for:
    - Proxy variables for protected classes
    - Class imbalance
    - Missing data patterns that correlate with sensitive features
    """
    import pandas as pd

    warnings = []

    # Protected/sensitive column patterns
    sensitive_patterns = [
        "gender", "sex", "race", "ethnicity", "religion", "age",
        "disability", "marital", "nationality", "origin", "zip", "zipcode",
    ]

    # Check for sensitive columns
    for col in df.columns:
        col_lower = col.lower()
        for pattern in sensitive_patterns:
            if pattern in col_lower:
                warnings.append({
                    "type": "sensitive_feature",
                    "column": col,
                    "message": f"Column '{col}' may contain sensitive/protected information. "
                               f"Consider whether it should be used in modeling.",
                    "severity": "high",
                })
                break

    # Check for class imbalance in likely target columns
    for col in df.columns:
        if df[col].dtype in ['object', 'bool', 'category'] or df[col].nunique() <= 10:
            value_counts = df[col].value_counts(normalize=True)
            if len(value_counts) >= 2:
                max_ratio = value_counts.iloc[0]
                min_ratio = value_counts.iloc[-1]
                if max_ratio > 0.9:
                    warnings.append({
                        "type": "class_imbalance",
                        "column": col,
                        "message": f"Column '{col}' has severe class imbalance "
                                   f"({max_ratio:.1%} vs {min_ratio:.1%}). "
                                   f"Consider resampling strategies.",
                        "severity": "medium",
                    })

    # Check for correlated missing data
    missing_cols = df.columns[df.isnull().any()].tolist()
    if len(missing_cols) > 1:
        for i, col1 in enumerate(missing_cols):
            for col2 in missing_cols[i+1:]:
                # Check if missing in one correlates with missing in other
                both_missing = (df[col1].isnull() & df[col2].isnull()).mean()
                col1_missing = df[col1].isnull().mean()
                col2_missing = df[col2].isnull().mean()
                expected = col1_missing * col2_missing

                if both_missing > 2 * expected and both_missing > 0.05:
                    warnings.append({
                        "type": "correlated_missing",
                        "columns": [col1, col2],
                        "message": f"Missing data in '{col1}' and '{col2}' are correlated. "
                                   f"This could indicate systematic bias in data collection.",
                        "severity": "medium",
                    })

    return warnings


# =============================================================================
# ANALYSIS WORKBENCH ENDPOINTS
# =============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@gated
def run_analysis(request):
    """
    Run a statistical/ML analysis.

    Request body:
    {
        "type": "stats" | "ml" | "spc" | "viz",
        "analysis": "anova" | "regression" | etc,
        "config": {...},
        "data_id": "..."
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    analysis_type = body.get("type")
    analysis_id = body.get("analysis")
    config = body.get("config", {})
    data_id = body.get("data_id")

    if not data_id:
        return JsonResponse({"error": "No data loaded. Please load a dataset first."}, status=400)

    start_time = time.time()

    try:
        import pandas as pd
        import numpy as np
        from io import StringIO

        # Load data - check multiple sources
        df = None

        # Source 1: Uploaded via upload_data endpoint (data_xxx format)
        if data_id and data_id.startswith("data_"):
            try:
                data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path)
            except Exception:
                pass

            # Fallback to temp directory
            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = pd.read_csv(data_path)
                except Exception:
                    pass

        # Source 2: Triage cleaned dataset
        if df is None and data_id:
            try:
                from .models import TriageResult
                triage_result = TriageResult.objects.get(id=data_id, user=request.user)
                df = pd.read_csv(StringIO(triage_result.cleaned_csv))
            except Exception:
                pass

        if df is None:
            return JsonResponse({"error": "Data not found. Please load a dataset first."}, status=404)

        result = {"plots": [], "summary": "", "guide_observation": ""}

        # Route to appropriate analysis
        if analysis_type == "stats":
            result = run_statistical_analysis(df, analysis_id, config)
        elif analysis_type == "ml":
            result = run_ml_analysis(df, analysis_id, config, request.user)
        elif analysis_type == "spc":
            result = run_spc_analysis(df, analysis_id, config)
        elif analysis_type == "viz":
            result = run_visualization(df, analysis_id, config)
        elif analysis_type == "bayesian":
            result = run_bayesian_analysis(df, analysis_id, config)
        else:
            return JsonResponse({"error": f"Unknown analysis type: {analysis_type}"}, status=400)

        latency = int((time.time() - start_time) * 1000)
        log_agent_action(request.user, "analysis", analysis_id, latency_ms=latency)

        logger.info(f"Analysis complete: {analysis_id}, plots: {len(result.get('plots', []))}, summary length: {len(result.get('summary', ''))}")

        return JsonResponse(result)

    except Exception as e:
        logger.exception(f"Analysis error: {e}")
        log_agent_action(request.user, "analysis", analysis_id, success=False, error_message=str(e))
        return JsonResponse({"error": str(e)}, status=500)


def run_statistical_analysis(df, analysis_id, config):
    """Run statistical analysis."""
    import numpy as np
    from scipy import stats

    result = {"plots": [], "summary": "", "guide_observation": ""}

    if analysis_id == "descriptive":
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Get selected vars from config, fall back to all numeric
        vars_from_config = config.get("vars", [])
        if isinstance(vars_from_config, list) and len(vars_from_config) > 0:
            vars_to_analyze = [v for v in vars_from_config if v in df.columns]
        else:
            vars_to_analyze = numeric_cols

        if not vars_to_analyze:
            result["summary"] = "No numeric variables found to analyze."
            return result

        desc = df[vars_to_analyze].describe().to_string()
        result["summary"] = f"Descriptive Statistics:\n\n{desc}"

        # Add explicit statistics for Synara integration
        result["statistics"] = {}
        for var in vars_to_analyze:
            col = df[var].dropna()
            result["statistics"][f"mean({var})"] = float(col.mean())
            result["statistics"][f"std({var})"] = float(col.std())
            result["statistics"][f"min({var})"] = float(col.min())
            result["statistics"][f"max({var})"] = float(col.max())
            result["statistics"][f"median({var})"] = float(col.median())
            result["statistics"][f"n({var})"] = int(len(col))

        # Add histogram for each variable
        for var in vars_to_analyze:
            try:
                data = df[var].dropna().tolist()
                if len(data) > 0:
                    result["plots"].append({
                        "title": f"Distribution of {var}",
                        "data": [{
                            "type": "histogram",
                            "x": data,
                            "name": var,
                            "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}
                        }],
                        "layout": {"height": 200}
                    })
            except Exception as plot_err:
                logger.warning(f"Could not create histogram for {var}: {plot_err}")

    elif analysis_id == "ttest":
        # One-sample t-test
        var1 = config.get("var1")
        mu = float(config.get("mu", 0))
        conf = int(config.get("conf", 95))
        alpha = 1 - conf / 100

        x = df[var1].dropna()
        n = len(x)
        stat, pval = stats.ttest_1samp(x, mu)

        # Confidence interval
        se = x.std() / np.sqrt(n)
        ci = stats.t.interval(conf/100, n-1, loc=x.mean(), scale=se)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ONE-SAMPLE T-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var1} (n = {n})\n"
        summary += f"<<COLOR:highlight>>Hypothesized mean:<</COLOR>> {mu}\n\n"
        summary += f"<<COLOR:text>>Sample Statistics:<</COLOR>>\n"
        summary += f"  Mean: {x.mean():.4f}\n"
        summary += f"  Std Dev: {x.std():.4f}\n"
        summary += f"  SE Mean: {se:.4f}\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  t-statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n"
        summary += f"  {conf}% CI: ({ci[0]:.4f}, {ci[1]:.4f})\n\n"

        if pval < alpha:
            summary += f"<<COLOR:good>>Reject H₀: Mean differs significantly from {mu} (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>Fail to reject H₀ (p >= {alpha})<</COLOR>>"

        result["summary"] = summary
        result["guide_observation"] = f"One-sample t-test with p-value {pval:.4f}. " + ("Significant." if pval < alpha else "Not significant.")

        # Explicit statistics for Synara
        result["statistics"] = {
            f"mean({var1})": float(x.mean()),
            f"std({var1})": float(x.std()),
            f"n({var1})": int(n),
            "t_statistic": float(stat),
            "p_value": float(pval),
            f"ci_lower({var1})": float(ci[0]),
            f"ci_upper({var1})": float(ci[1]),
        }

    elif analysis_id == "ttest2":
        # Two-sample t-test
        var1 = config.get("var1")
        var2 = config.get("var2")
        conf = int(config.get("conf", 95))
        alpha = 1 - conf / 100

        x = df[var1].dropna()
        y = df[var2].dropna()
        stat, pval = stats.ttest_ind(x, y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>TWO-SAMPLE T-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Sample 1:<</COLOR>> {var1} (n = {len(x)}, mean = {x.mean():.4f}, std = {x.std():.4f})\n"
        summary += f"<<COLOR:text>>Sample 2:<</COLOR>> {var2} (n = {len(y)}, mean = {y.mean():.4f}, std = {y.std():.4f})\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  Difference of means: {x.mean() - y.mean():.4f}\n"
        summary += f"  t-statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n\n"

        if pval < alpha:
            summary += f"<<COLOR:good>>Means are significantly different (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference (p >= {alpha})<</COLOR>>"

        result["summary"] = summary
        result["guide_observation"] = f"Two-sample t-test with p-value {pval:.4f}."

    elif analysis_id == "paired_t":
        # Paired t-test
        var1 = config.get("var1")
        var2 = config.get("var2")
        conf = int(config.get("conf", 95))
        alpha = 1 - conf / 100

        x = df[var1].dropna()
        y = df[var2].dropna()
        # Align samples
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]

        diff = x - y
        stat, pval = stats.ttest_rel(x, y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>PAIRED T-TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:text>>Sample 1 (Before):<</COLOR>> {var1}\n"
        summary += f"<<COLOR:text>>Sample 2 (After):<</COLOR>> {var2}\n"
        summary += f"<<COLOR:text>>Pairs:<</COLOR>> {len(x)}\n\n"
        summary += f"<<COLOR:text>>Difference Statistics:<</COLOR>>\n"
        summary += f"  Mean difference: {diff.mean():.4f}\n"
        summary += f"  Std of differences: {diff.std():.4f}\n\n"
        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  t-statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n\n"

        if pval < alpha:
            summary += f"<<COLOR:good>>Significant difference between paired observations (p < {alpha})<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant difference (p >= {alpha})<</COLOR>>"

        result["summary"] = summary
        result["guide_observation"] = f"Paired t-test with p-value {pval:.4f}."

    elif analysis_id == "anova":
        response = config.get("response")
        # Support both 'factor' (single) and 'factors' (list) for backwards compatibility
        factor = config.get("factor")
        factors = config.get("factors", [])
        if factor and not factors:
            factors = [factor]

        if len(factors) >= 1:
            # One-way ANOVA
            factor_col = factors[0]
            groups = [df[df[factor_col] == level][response].dropna() for level in df[factor_col].unique()]
            groups = [g for g in groups if len(g) > 0]  # Remove empty groups

            stat, pval = stats.f_oneway(*groups)

            # Calculate group statistics
            summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary += f"<<COLOR:title>>ONE-WAY ANOVA<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
            summary += f"<<COLOR:highlight>>Factor:<</COLOR>> {factor_col}\n\n"

            summary += f"<<COLOR:text>>Group Statistics:<</COLOR>>\n"
            for level in df[factor_col].unique():
                grp = df[df[factor_col] == level][response].dropna()
                summary += f"  {level}: n={len(grp)}, mean={grp.mean():.4f}, std={grp.std():.4f}\n"

            summary += f"\n<<COLOR:text>>ANOVA Results:<</COLOR>>\n"
            summary += f"  F-statistic: {stat:.4f}\n"
            summary += f"  p-value: {pval:.4f}\n\n"

            if pval < 0.05:
                summary += f"<<COLOR:good>>Significant difference between groups (p < 0.05)<</COLOR>>"
            else:
                summary += f"<<COLOR:text>>No significant difference (p >= 0.05)<</COLOR>>"

            result["summary"] = summary

            # Box plot
            result["plots"].append({
                "title": f"{response} by {factor_col}",
                "data": [{
                    "type": "box",
                    "y": df[response].tolist(),
                    "x": df[factor_col].astype(str).tolist(),
                    "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}
                }],
                "layout": {"template": "plotly_dark", "height": 300}
            })
        else:
            result["summary"] = "Please select a factor column for ANOVA."

    elif analysis_id == "anova2":
        # Two-way ANOVA
        response = config.get("response")
        factor_a = config.get("factor_a")
        factor_b = config.get("factor_b")

        try:
            import statsmodels.api as sm
            from statsmodels.formula.api import ols

            # Build formula
            formula = f'{response} ~ C({factor_a}) + C({factor_b}) + C({factor_a}):C({factor_b})'
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
            summary += f"<<COLOR:title>>TWO-WAY ANOVA<</COLOR>>\n"
            summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
            summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
            summary += f"<<COLOR:highlight>>Factor A:<</COLOR>> {factor_a}\n"
            summary += f"<<COLOR:highlight>>Factor B:<</COLOR>> {factor_b}\n\n"

            summary += f"<<COLOR:text>>ANOVA Table:<</COLOR>>\n"
            summary += anova_table.to_string() + "\n\n"

            # Interpret results
            for idx in anova_table.index:
                if 'PR(>F)' in anova_table.columns:
                    p = anova_table.loc[idx, 'PR(>F)']
                    if not np.isnan(p):
                        sig = "<<COLOR:good>>*<</COLOR>>" if p < 0.05 else ""
                        summary += f"{idx}: p = {p:.4f} {sig}\n"

            result["summary"] = summary

            # Interaction plot
            means = df.groupby([factor_a, factor_b])[response].mean().unstack()
            traces = []
            for col in means.columns:
                traces.append({
                    "type": "scatter",
                    "x": means.index.astype(str).tolist(),
                    "y": means[col].tolist(),
                    "mode": "lines+markers",
                    "name": str(col)
                })

            result["plots"].append({
                "title": "Interaction Plot",
                "data": traces,
                "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": factor_a}}
            })

        except ImportError:
            result["summary"] = "Two-way ANOVA requires statsmodels. Install with: pip install statsmodels"
        except Exception as e:
            result["summary"] = f"Two-way ANOVA error: {str(e)}"

    elif analysis_id == "regression":
        response = config.get("response")
        predictors = config.get("predictors", [])
        degree = int(config.get("degree", 1))
        interactions = config.get("interactions", "none")

        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import r2_score, mean_squared_error

        X_raw = df[predictors].dropna()
        y = df[response].loc[X_raw.index]
        n = len(y)

        # Build feature names and transform data
        feature_names = list(predictors)

        if degree > 1 or interactions == "all":
            # Use PolynomialFeatures for polynomial and/or interaction terms
            include_interaction = interactions == "all"
            poly = PolynomialFeatures(
                degree=degree,
                include_bias=False,
                interaction_only=(degree == 1 and include_interaction)
            )
            X = poly.fit_transform(X_raw)

            # Generate readable feature names
            feature_names = []
            for powers in poly.powers_:
                parts = []
                for i, power in enumerate(powers):
                    if power == 0:
                        continue
                    elif power == 1:
                        parts.append(predictors[i])
                    else:
                        parts.append(f"{predictors[i]}^{power}")
                if parts:
                    feature_names.append("·".join(parts) if len(parts) > 1 else parts[0])
        else:
            X = X_raw.values

        p = X.shape[1]  # Number of features after transformation

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Calculate comprehensive statistics
        residuals = y.values - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y.values - np.mean(y.values))**2)
        r2 = 1 - (ss_res / ss_tot)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mse = ss_res / (n - p - 1) if n > p + 1 else 1e-10
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        # Calculate standard errors and t-statistics
        X_with_const = np.column_stack([np.ones(n), X])
        try:
            var_coef = mse * np.linalg.inv(X_with_const.T @ X_with_const).diagonal()
            se = np.sqrt(var_coef)
            coefs = np.concatenate([[model.intercept_], model.coef_])
            t_stats = coefs / se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
        except:
            se = np.zeros(p + 1)
            t_stats = np.zeros(p + 1)
            p_values = np.ones(p + 1)
            coefs = np.concatenate([[model.intercept_], model.coef_])

        # F-statistic
        f_stat = (ss_tot - ss_res) / p / mse if p > 0 else 0
        f_pvalue = 1 - stats.f.cdf(f_stat, p, n - p - 1) if p > 0 else 1

        # Durbin-Watson statistic
        dw = np.sum(np.diff(residuals)**2) / ss_res

        # Build colored summary output
        model_type = "Linear" if degree == 1 else "Quadratic" if degree == 2 else "Cubic"
        if interactions == "all":
            model_type += " + Interactions"

        summary = "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n"
        summary += f"<<COLOR:accent>>                          {model_type.upper()} REGRESSION RESULTS<</COLOR>>\n"
        summary += "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n\n"

        summary += f"<<COLOR:dim>>Dep. Variable:<</COLOR>>    <<COLOR:text>>{response}<</COLOR>>\n"
        summary += f"<<COLOR:dim>>No. Observations:<</COLOR>> <<COLOR:text>>{n}<</COLOR>>\n"
        summary += f"<<COLOR:dim>>No. Features:<</COLOR>>     <<COLOR:text>>{p}<</COLOR>> (from {len(predictors)} predictors)\n"
        summary += f"<<COLOR:dim>>Model:<</COLOR>>            <<COLOR:text>>OLS - {model_type}<</COLOR>>\n\n"

        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += "<<COLOR:accent>>                               MODEL FIT<</COLOR>>\n"
        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"

        r2_color = "success" if r2 > 0.7 else "warning" if r2 > 0.4 else "danger"
        summary += f"<<COLOR:dim>>Residual std. error:<</COLOR>> <<COLOR:text>>{np.sqrt(mse):.4f}<</COLOR>> on <<COLOR:text>>{n - p - 1}<</COLOR>> degrees of freedom\n"
        summary += f"<<COLOR:dim>>Multiple R-squared:<</COLOR>>  <<COLOR:{r2_color}>>{r2:.4f}<</COLOR>>\n"
        summary += f"<<COLOR:dim>>Adjusted R-squared:<</COLOR>> <<COLOR:{r2_color}>>{adj_r2:.4f}<</COLOR>>\n"

        f_color = "success" if f_pvalue < 0.05 else "warning" if f_pvalue < 0.1 else "danger"
        summary += f"<<COLOR:dim>>F-statistic:<</COLOR>>        <<COLOR:text>>{f_stat:.2f}<</COLOR>> on <<COLOR:text>>{p}<</COLOR>> and <<COLOR:text>>{n - p - 1}<</COLOR>> DF\n"
        summary += f"<<COLOR:dim>>p-value:<</COLOR>>            <<COLOR:{f_color}>>{f_pvalue:.4e}<</COLOR>>\n"
        summary += f"<<COLOR:dim>>Durbin-Watson:<</COLOR>>      <<COLOR:text>>{dw:.3f}<</COLOR>>\n\n"

        summary += "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n"
        summary += "<<COLOR:accent>>                              COEFFICIENTS<</COLOR>>\n"
        summary += "<<COLOR:accent>>════════════════════════════════════════════════════════════════════════════<</COLOR>>\n"
        summary += "<<COLOR:dim>>                            Estimate    Std.Err    t value    Pr(>|t|)     <</COLOR>>\n"

        names = ["(Intercept)"] + feature_names
        non_sig_predictors = []
        for i, name in enumerate(names):
            pv = p_values[i]
            sig = "***" if pv < 0.001 else "** " if pv < 0.01 else "*  " if pv < 0.05 else ".  " if pv < 0.1 else "   "
            p_color = "success" if pv < 0.05 else "warning" if pv < 0.1 else "dim"
            summary += f"<<COLOR:text>>{name:<24}<</COLOR>> {coefs[i]:>10.4f}   {se[i]:>9.4f}   {t_stats[i]:>8.3f}    <<COLOR:{p_color}>>{pv:>9.4f}  {sig}<</COLOR>>\n"
            if i > 0 and pv >= 0.1:  # Track non-significant predictors (excluding intercept)
                non_sig_predictors.append(name)

        summary += "<<COLOR:dim>>---<</COLOR>>\n"
        summary += "<<COLOR:dim>>Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1<</COLOR>>\n\n"

        # Diagnostics interpretation
        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += "<<COLOR:accent>>                            DIAGNOSTICS SUMMARY<</COLOR>>\n"
        summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"

        # Interpret R²
        if r2 > 0.7:
            summary += "<<COLOR:success>>✓ Good fit:<</COLOR>> Model explains {:.1f}% of variance\n".format(r2*100)
        elif r2 > 0.4:
            summary += "<<COLOR:warning>>◐ Moderate fit:<</COLOR>> Model explains {:.1f}% of variance\n".format(r2*100)
        else:
            summary += "<<COLOR:danger>>✗ Poor fit:<</COLOR>> Model explains only {:.1f}% of variance\n".format(r2*100)

        # Interpret Durbin-Watson
        if 1.5 < dw < 2.5:
            summary += "<<COLOR:success>>✓ No autocorrelation:<</COLOR>> Durbin-Watson ≈ 2\n"
        else:
            summary += "<<COLOR:warning>>◐ Possible autocorrelation:<</COLOR>> Durbin-Watson = {:.2f}\n".format(dw)

        # Interpret F-test
        if f_pvalue < 0.05:
            summary += "<<COLOR:success>>✓ Model significant:<</COLOR>> F-test p < 0.05\n"
        else:
            summary += "<<COLOR:danger>>✗ Model not significant:<</COLOR>> F-test p = {:.3f}\n".format(f_pvalue)

        # Calculate diagnostic values first (needed for suggestions and plots)
        std_residuals = residuals / np.std(residuals)

        # Leverage (hat values)
        try:
            H = X_with_const @ np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T
            leverage = np.diag(H)
        except:
            leverage = np.ones(n) / n

        # Cook's distance
        cooks_d = (std_residuals**2 / (p + 1)) * (leverage / (1 - leverage + 1e-10)**2)

        # Square root of standardized residuals for scale-location
        sqrt_std_resid = np.sqrt(np.abs(std_residuals))

        # Model improvement suggestions
        suggestions = []

        # Low R² suggestions
        if r2 < 0.4:
            suggestions.append("Add more predictors or interaction terms (X1*X2)")
            suggestions.append("Try polynomial terms (X², X³) for non-linear relationships")
            suggestions.append("Check for outliers that may be distorting the fit")
        elif r2 < 0.7:
            suggestions.append("Consider adding interaction terms or polynomial features")

        # Non-significant predictors
        if non_sig_predictors:
            if len(non_sig_predictors) <= 3:
                suggestions.append(f"Consider removing non-significant: {', '.join(non_sig_predictors)}")
            else:
                suggestions.append(f"Consider removing {len(non_sig_predictors)} non-significant predictors")

        # Autocorrelation
        if dw < 1.5:
            suggestions.append("Positive autocorrelation detected - consider time series methods or lag terms")
        elif dw > 2.5:
            suggestions.append("Negative autocorrelation detected - check data ordering")

        # Model not significant
        if f_pvalue >= 0.05:
            suggestions.append("Model not significant - try different predictors or check data quality")

        # High leverage points
        high_leverage = int(np.sum(leverage > 2 * (p + 1) / n)) if n > 0 else 0
        if high_leverage > 0:
            suggestions.append(f"{high_leverage} high-leverage points detected - check for influential outliers")

        # Large Cook's distance
        high_cooks = int(np.sum(cooks_d > 4 / n)) if n > 0 else 0
        if high_cooks > 0:
            suggestions.append(f"{high_cooks} influential observations (Cook's D) - consider robust regression")

        if suggestions:
            summary += "\n<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            summary += "<<COLOR:accent>>                          IMPROVEMENT SUGGESTIONS<</COLOR>>\n"
            summary += "<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            for i, sug in enumerate(suggestions[:5], 1):  # Limit to 5 suggestions
                summary += f"<<COLOR:warning>>{i}.<</COLOR>> <<COLOR:text>>{sug}<</COLOR>>\n"

        result["summary"] = summary

        # Create 4-panel diagnostic plots

        # 1. Residuals vs Fitted
        result["plots"].append({
            "title": "1. Residuals vs Fitted",
            "data": [
                {
                    "type": "scatter",
                    "x": y_pred.tolist(),
                    "y": residuals.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "Residuals"
                },
                {
                    "type": "scatter",
                    "x": [min(y_pred), max(y_pred)],
                    "y": [0, 0],
                    "mode": "lines",
                    "line": {"color": "#9f4a4a", "dash": "dash"},
                    "name": "Zero line"
                }
            ],
            "layout": {"height": 250, "xaxis": {"title": "Fitted values"}, "yaxis": {"title": "Residuals"}}
        })

        # 2. Normal Q-Q Plot
        sorted_std_resid = np.sort(std_residuals)
        theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_std_resid)))
        result["plots"].append({
            "title": "2. Normal Q-Q",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical_q.tolist(),
                    "y": sorted_std_resid.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "Residuals"
                },
                {
                    "type": "scatter",
                    "x": [-3, 3],
                    "y": [-3, 3],
                    "mode": "lines",
                    "line": {"color": "#9f4a4a", "dash": "dash"},
                    "name": "Normal line"
                }
            ],
            "layout": {"height": 250, "xaxis": {"title": "Theoretical Quantiles"}, "yaxis": {"title": "Std. Residuals"}}
        })

        # 3. Scale-Location
        result["plots"].append({
            "title": "3. Scale-Location",
            "data": [
                {
                    "type": "scatter",
                    "x": y_pred.tolist(),
                    "y": sqrt_std_resid.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "√|Std. Residuals|"
                }
            ],
            "layout": {"height": 250, "xaxis": {"title": "Fitted values"}, "yaxis": {"title": "√|Standardized residuals|"}}
        })

        # 4. Residuals vs Leverage
        result["plots"].append({
            "title": "4. Residuals vs Leverage",
            "data": [
                {
                    "type": "scatter",
                    "x": leverage.tolist(),
                    "y": std_residuals.tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": cooks_d.tolist(),
                        "colorscale": [[0, "#4a9f6e"], [0.5, "#e89547"], [1, "#9f4a4a"]],
                        "size": 6,
                        "colorbar": {"title": "Cook's D", "len": 0.5}
                    },
                    "name": "Observations"
                },
                {
                    "type": "scatter",
                    "x": [0, max(leverage)],
                    "y": [0, 0],
                    "mode": "lines",
                    "line": {"color": "#9f4a4a", "dash": "dash"},
                    "showlegend": False
                }
            ],
            "layout": {"height": 250, "xaxis": {"title": "Leverage"}, "yaxis": {"title": "Std. Residuals"}}
        })

        # Explicit statistics for Synara integration
        result["statistics"] = {
            "R²": float(r2),
            "Adj_R²": float(adj_r2),
            "RMSE": float(rmse),
            "F_statistic": float(f_stat),
            "F_p_value": float(f_pvalue),
            "n": int(n),
            "predictors": int(p),
        }
        # Add coefficients as statistics
        for i, feat in enumerate(feature_names):
            result["statistics"][f"coef({feat})"] = float(model.coef_[i])
            if i < len(p_values) - 1:
                result["statistics"][f"p_value({feat})"] = float(p_values[i + 1])

    elif analysis_id == "correlation":
        vars_list = config.get("vars", [])
        method = config.get("method", "pearson")

        if vars_list:
            numeric_cols = [v for v in vars_list if v in df.columns]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        corr_matrix = df[numeric_cols].corr(method=method)

        # Build formatted output
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>CORRELATION ANALYSIS ({method.upper()})<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variables:<</COLOR>> {', '.join(numeric_cols)}\n\n"
        summary += "<<COLOR:text>>Correlation Matrix:<</COLOR>>\n"
        summary += corr_matrix.to_string() + "\n"

        result["summary"] = summary

        # Heatmap
        result["plots"].append({
            "title": "Correlation Heatmap",
            "data": [{
                "type": "heatmap",
                "z": corr_matrix.values.tolist(),
                "x": numeric_cols,
                "y": numeric_cols,
                "colorscale": "RdBu",
                "zmid": 0,
                "text": [[f"{v:.3f}" for v in row] for row in corr_matrix.values],
                "texttemplate": "%{text}",
                "textfont": {"size": 10}
            }],
            "layout": {"template": "plotly_dark", "height": 400}
        })

    elif analysis_id == "normality":
        var = config.get("var")
        test_type = config.get("test", "anderson")

        x = df[var].dropna()
        n = len(x)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>NORMALITY TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var} (n = {n})\n\n"

        if test_type == "anderson":
            stat_result = stats.anderson(x)
            summary += f"<<COLOR:text>>Anderson-Darling Test<</COLOR>>\n"
            summary += f"  Statistic: {stat_result.statistic:.4f}\n"
            summary += f"  Critical Values:\n"
            for cv, sl in zip(stat_result.critical_values, stat_result.significance_level):
                marker = "<<COLOR:good>>✓<</COLOR>>" if stat_result.statistic < cv else "<<COLOR:bad>>✗<</COLOR>>"
                summary += f"    {marker} {sl}%: {cv:.4f}\n"
        elif test_type == "shapiro":
            stat, pval = stats.shapiro(x)
            summary += f"<<COLOR:text>>Shapiro-Wilk Test<</COLOR>>\n"
            summary += f"  W-statistic: {stat:.4f}\n"
            summary += f"  p-value: {pval:.4f}\n"
            if pval < 0.05:
                summary += f"\n<<COLOR:bad>>Data is NOT normally distributed (p < 0.05)<</COLOR>>"
            else:
                summary += f"\n<<COLOR:good>>Data appears normally distributed (p >= 0.05)<</COLOR>>"
        elif test_type == "ks":
            stat, pval = stats.kstest(x, 'norm', args=(x.mean(), x.std()))
            summary += f"<<COLOR:text>>Kolmogorov-Smirnov Test<</COLOR>>\n"
            summary += f"  D-statistic: {stat:.4f}\n"
            summary += f"  p-value: {pval:.4f}\n"
            if pval < 0.05:
                summary += f"\n<<COLOR:bad>>Data is NOT normally distributed (p < 0.05)<</COLOR>>"
            else:
                summary += f"\n<<COLOR:good>>Data appears normally distributed (p >= 0.05)<</COLOR>>"

        result["summary"] = summary

        # Q-Q plot
        sorted_data = np.sort(x)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(x)))

        result["plots"].append({
            "title": f"Normal Q-Q Plot: {var}",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical_quantiles.tolist(),
                    "y": sorted_data.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "Data"
                },
                {
                    "type": "scatter",
                    "x": [theoretical_quantiles.min(), theoretical_quantiles.max()],
                    "y": [sorted_data.min(), sorted_data.max()],
                    "mode": "lines",
                    "line": {"color": "#ff7675", "dash": "dash"},
                    "name": "Reference"
                }
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 300,
                "xaxis": {"title": "Theoretical Quantiles"},
                "yaxis": {"title": "Sample Quantiles"}
            }
        })

    elif analysis_id == "chi2":
        row_var = config.get("row_var")
        col_var = config.get("col_var")

        # Create contingency table
        contingency = pd.crosstab(df[row_var], df[col_var])
        chi2, pval, dof, expected = stats.chi2_contingency(contingency)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>CHI-SQUARE TEST FOR INDEPENDENCE<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Row Variable:<</COLOR>> {row_var}\n"
        summary += f"<<COLOR:highlight>>Column Variable:<</COLOR>> {col_var}\n\n"

        summary += f"<<COLOR:text>>Contingency Table (Observed):<</COLOR>>\n"
        summary += contingency.to_string() + "\n\n"

        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  Chi-square statistic: {chi2:.4f}\n"
        summary += f"  Degrees of freedom: {dof}\n"
        summary += f"  p-value: {pval:.4f}\n\n"

        if pval < 0.05:
            summary += f"<<COLOR:good>>Variables are significantly associated (p < 0.05)<</COLOR>>"
        else:
            summary += f"<<COLOR:text>>No significant association found (p >= 0.05)<</COLOR>>"

        result["summary"] = summary

        # Heatmap of observed counts
        result["plots"].append({
            "title": f"Contingency Table: {row_var} × {col_var}",
            "data": [{
                "type": "heatmap",
                "z": contingency.values.tolist(),
                "x": contingency.columns.astype(str).tolist(),
                "y": contingency.index.astype(str).tolist(),
                "colorscale": "Blues",
                "text": contingency.values.tolist(),
                "texttemplate": "%{text}",
                "textfont": {"size": 12}
            }],
            "layout": {"template": "plotly_dark", "height": 300}
        })

    elif analysis_id == "granger":
        """
        Granger Causality Test - does X help predict Y?
        Tests if past values of X improve prediction of Y beyond Y's own past.
        """
        from statsmodels.tsa.stattools import grangercausalitytests

        var_x = config.get("var_x")
        var_y = config.get("var_y")
        max_lag = int(config.get("max_lag", 4))

        # Prepare data - must be stationary ideally
        x = df[var_x].dropna().values
        y = df[var_y].dropna().values

        # Align lengths
        min_len = min(len(x), len(y))
        data_matrix = np.column_stack([y[:min_len], x[:min_len]])

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>GRANGER CAUSALITY TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Question:<</COLOR>> Does {var_x} Granger-cause {var_y}?\n"
        summary += f"<<COLOR:highlight>>Max Lags:<</COLOR>> {max_lag}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {min_len}\n\n"

        summary += f"<<COLOR:text>>Interpretation:<</COLOR>>\n"
        summary += f"  If p < 0.05, past values of {var_x} help predict {var_y}\n"
        summary += f"  beyond what {var_y}'s own past provides.\n\n"

        try:
            # Run Granger test (suppress output)
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            gc_results = grangercausalitytests(data_matrix, maxlag=max_lag, verbose=False)
            sys.stdout = old_stdout

            summary += f"<<COLOR:text>>Results by Lag:<</COLOR>>\n"
            summary += f"{'Lag':<6} {'F-stat':<12} {'p-value':<12} {'Significant':<12}\n"
            summary += f"{'-'*42}\n"

            significant_lags = []
            p_values = []
            f_stats = []

            for lag in range(1, max_lag + 1):
                if lag in gc_results:
                    test_result = gc_results[lag][0]['ssr_ftest']
                    f_stat = test_result[0]
                    p_val = test_result[1]
                    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""

                    f_stats.append(f_stat)
                    p_values.append(p_val)

                    if p_val < 0.05:
                        significant_lags.append(lag)

                    summary += f"{lag:<6} {f_stat:<12.4f} {p_val:<12.4f} {sig}\n"

            summary += f"\n<<COLOR:accent>>────────────────────────────────────────<</COLOR>>\n"
            if significant_lags:
                summary += f"<<COLOR:good>>CONCLUSION: {var_x} Granger-causes {var_y}<</COLOR>>\n"
                summary += f"<<COLOR:text>>Significant at lags: {significant_lags}<</COLOR>>\n"
                summary += f"<<COLOR:text>>This suggests {var_x} has predictive power for {var_y}.<</COLOR>>\n"
            else:
                summary += f"<<COLOR:warning>>CONCLUSION: No Granger causality detected<</COLOR>>\n"
                summary += f"<<COLOR:text>>{var_x} does not significantly improve prediction of {var_y}.<</COLOR>>\n"

            result["summary"] = summary

            # Plot: p-values by lag
            result["plots"].append({
                "title": f"Granger Causality: {var_x} → {var_y}",
                "data": [{
                    "type": "bar",
                    "x": [f"Lag {i+1}" for i in range(len(p_values))],
                    "y": p_values,
                    "marker": {
                        "color": ["#e85747" if p < 0.05 else "rgba(74, 159, 110, 0.4)" for p in p_values],
                        "line": {"color": "#4a9f6e", "width": 1.5}
                    }
                }, {
                    "type": "scatter",
                    "x": [f"Lag {i+1}" for i in range(len(p_values))],
                    "y": [0.05] * len(p_values),
                    "mode": "lines",
                    "line": {"color": "#e85747", "dash": "dash", "width": 2},
                    "name": "α = 0.05"
                }],
                "layout": {
                    "height": 300,
                    "yaxis": {"title": "p-value", "range": [0, max(0.2, max(p_values) * 1.1)]},
                    "xaxis": {"title": "Lag"}
                }
            })

            # Time series plot
            result["plots"].append({
                "title": f"Time Series: {var_x} and {var_y}",
                "data": [{
                    "type": "scatter",
                    "y": x[:min_len].tolist(),
                    "mode": "lines",
                    "name": var_x,
                    "line": {"color": "#4a9f6e"}
                }, {
                    "type": "scatter",
                    "y": y[:min_len].tolist(),
                    "mode": "lines",
                    "name": var_y,
                    "yaxis": "y2",
                    "line": {"color": "#47a5e8"}
                }],
                "layout": {
                    "height": 250,
                    "yaxis": {"title": var_x, "side": "left"},
                    "yaxis2": {"title": var_y, "side": "right", "overlaying": "y"},
                    "xaxis": {"title": "Observation"}
                }
            })

            result["guide_observation"] = f"Granger causality: {var_x} → {var_y} " + (f"significant at lags {significant_lags}." if significant_lags else "not significant.")

            # Statistics for Synara
            result["statistics"] = {
                f"granger_min_pvalue": float(min(p_values)),
                f"granger_significant_lags": len(significant_lags),
                f"granger_causal": 1 if significant_lags else 0
            }

        except Exception as e:
            result["summary"] = f"Granger causality test failed: {str(e)}\n\nEnsure both variables are numeric and have sufficient observations."

    elif analysis_id == "changepoint":
        """
        Change Point Detection - when did the process shift?
        Uses PELT algorithm to find optimal change points.
        """
        var = config.get("var")
        penalty = config.get("penalty", "bic")
        min_size = int(config.get("min_size", 10))

        data = df[var].dropna().values
        n = len(data)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>CHANGE POINT DETECTION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n"
        summary += f"<<COLOR:highlight>>Method:<</COLOR>> PELT with {penalty.upper()} penalty\n\n"

        try:
            import ruptures as rpt

            # Use PELT algorithm (optimal for multiple change points)
            model = rpt.Pelt(model="rbf", min_size=min_size).fit(data)

            # Determine penalty value
            if penalty == "bic":
                pen = np.log(n) * np.var(data)
            elif penalty == "aic":
                pen = 2 * np.var(data)
            else:
                pen = float(penalty) if penalty.replace('.','').isdigit() else np.log(n) * np.var(data)

            change_points = model.predict(pen=pen)

            # Remove the last point (always equals n)
            if change_points and change_points[-1] == n:
                change_points = change_points[:-1]

            summary += f"<<COLOR:text>>Change Points Detected:<</COLOR>> {len(change_points)}\n\n"

            if change_points:
                summary += f"<<COLOR:accent>>{'─' * 50}<</COLOR>>\n"
                summary += f"<<COLOR:text>>{'Index':<10} {'Value at CP':<15} {'Segment Mean':<15}<</COLOR>>\n"
                summary += f"<<COLOR:accent>>{'─' * 50}<</COLOR>>\n"

                segments = [0] + change_points + [n]
                for i, cp in enumerate(change_points):
                    seg_start = segments[i]
                    seg_end = cp
                    seg_mean = np.mean(data[seg_start:seg_end])
                    summary += f"{cp:<10} {data[cp-1]:<15.4f} {seg_mean:<15.4f}\n"

                # Final segment
                final_mean = np.mean(data[change_points[-1]:])
                summary += f"\n<<COLOR:text>>Final segment mean: {final_mean:.4f}<</COLOR>>\n"

                summary += f"\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
                summary += f"  These points mark where the process behavior shifted.\n"
                summary += f"  Investigate what changed at these times.\n"
            else:
                summary += f"<<COLOR:text>>No significant change points detected.<</COLOR>>\n"
                summary += f"<<COLOR:text>>The process appears stable over this period.<</COLOR>>\n"

            result["summary"] = summary

            # Plot with change points
            plot_data = [{
                "type": "scatter",
                "y": data.tolist(),
                "mode": "lines",
                "name": var,
                "line": {"color": "#4a9f6e", "width": 1.5}
            }]

            # Add vertical lines for change points
            shapes = []
            for cp in change_points:
                shapes.append({
                    "type": "line",
                    "x0": cp,
                    "x1": cp,
                    "y0": 0,
                    "y1": 1,
                    "yref": "paper",
                    "line": {"color": "#e85747", "dash": "dash", "width": 2}
                })

            # Add segment means as horizontal lines
            segments = [0] + list(change_points) + [n]
            for i in range(len(segments) - 1):
                seg_mean = np.mean(data[segments[i]:segments[i+1]])
                plot_data.append({
                    "type": "scatter",
                    "x": [segments[i], segments[i+1]-1],
                    "y": [seg_mean, seg_mean],
                    "mode": "lines",
                    "line": {"color": "#e89547", "width": 2},
                    "name": f"Segment {i+1} mean" if i == 0 else None,
                    "showlegend": i == 0
                })

            result["plots"].append({
                "title": f"Change Points: {var}",
                "data": plot_data,
                "layout": {
                    "height": 350,
                    "shapes": shapes,
                    "xaxis": {"title": "Observation"},
                    "yaxis": {"title": var}
                }
            })

            result["guide_observation"] = f"Detected {len(change_points)} change point(s) in {var}." + (" Investigate what caused these shifts." if change_points else " Process appears stable.")

            # Statistics for Synara
            result["statistics"] = {
                "change_points_count": len(change_points),
                "change_point_indices": change_points if change_points else []
            }

        except ImportError:
            result["summary"] = "Change point detection requires the 'ruptures' package.\n\nInstall with: pip install ruptures"
        except Exception as e:
            result["summary"] = f"Change point detection failed: {str(e)}"

    elif analysis_id == "mann_whitney":
        """
        Mann-Whitney U Test - non-parametric alternative to 2-sample t-test.
        Tests if two groups have different distributions.
        """
        var = config.get("var")
        group_var = config.get("group_var")

        groups = df[group_var].dropna().unique()
        if len(groups) != 2:
            result["summary"] = f"Mann-Whitney U requires exactly 2 groups. Found {len(groups)}: {list(groups)}"
            return result

        group1 = df[df[group_var] == groups[0]][var].dropna()
        group2 = df[df[group_var] == groups[1]][var].dropna()

        stat, pval = stats.mannwhitneyu(group1, group2, alternative='two-sided')

        # Effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        effect_size = 1 - (2 * stat) / (n1 * n2)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>MANN-WHITNEY U TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} (n={n1}) vs {groups[1]} (n={n2})\n\n"

        summary += f"<<COLOR:text>>Group Statistics:<</COLOR>>\n"
        summary += f"  {groups[0]}: median = {group1.median():.4f}, mean rank = {stats.rankdata(np.concatenate([group1, group2]))[:n1].mean():.1f}\n"
        summary += f"  {groups[1]}: median = {group2.median():.4f}, mean rank = {stats.rankdata(np.concatenate([group1, group2]))[n1:].mean():.1f}\n\n"

        summary += f"<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  U statistic: {stat:.2f}\n"
        summary += f"  p-value: {pval:.4f}\n"
        summary += f"  Effect size (r): {effect_size:.3f}\n\n"

        if pval < 0.05:
            summary += f"<<COLOR:good>>Groups differ significantly (p < 0.05)<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>No significant difference (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Box plots
        result["plots"].append({
            "title": f"Mann-Whitney: {var} by {group_var}",
            "data": [
                {"type": "box", "y": group1.tolist(), "name": str(groups[0]), "marker": {"color": "#4a9f6e"}, "fillcolor": "rgba(74, 159, 110, 0.3)"},
                {"type": "box", "y": group2.tolist(), "name": str(groups[1]), "marker": {"color": "#47a5e8"}, "fillcolor": "rgba(71, 165, 232, 0.3)"}
            ],
            "layout": {"height": 300, "yaxis": {"title": var}}
        })

        result["guide_observation"] = f"Mann-Whitney U test p = {pval:.4f}. " + ("Groups differ significantly." if pval < 0.05 else "No significant difference.")
        result["statistics"] = {"U_statistic": float(stat), "p_value": float(pval), "effect_size_r": float(effect_size)}

    elif analysis_id == "kruskal":
        """
        Kruskal-Wallis H Test - non-parametric alternative to one-way ANOVA.
        Tests if multiple groups have different distributions.
        """
        var = config.get("var")
        group_var = config.get("group_var")

        groups = df[group_var].dropna().unique()
        group_data = [df[df[group_var] == g][var].dropna().values for g in groups]

        stat, pval = stats.kruskal(*group_data)

        # Effect size (epsilon squared)
        n_total = sum(len(g) for g in group_data)
        epsilon_sq = (stat - len(groups) + 1) / (n_total - len(groups))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>KRUSKAL-WALLIS H TEST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {len(groups)} levels of {group_var}\n"
        summary += f"<<COLOR:highlight>>Total N:<</COLOR>> {n_total}\n\n"

        summary += f"<<COLOR:text>>Group Statistics:<</COLOR>>\n"
        for g, data in zip(groups, group_data):
            summary += f"  {g}: n={len(data)}, median={np.median(data):.4f}\n"

        summary += f"\n<<COLOR:text>>Test Results:<</COLOR>>\n"
        summary += f"  H statistic: {stat:.4f}\n"
        summary += f"  p-value: {pval:.4f}\n"
        summary += f"  ε² (effect size): {epsilon_sq:.4f}\n\n"

        if pval < 0.05:
            summary += f"<<COLOR:good>>At least one group differs significantly (p < 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>Consider post-hoc tests (Dunn's test) to identify which groups differ.<</COLOR>>\n"
        else:
            summary += f"<<COLOR:text>>No significant difference among groups (p >= 0.05)<</COLOR>>\n"

        result["summary"] = summary

        # Box plots for all groups
        result["plots"].append({
            "title": f"Kruskal-Wallis: {var} by {group_var}",
            "data": [
                {"type": "box", "y": data.tolist(), "name": str(g),
                 "marker": {"color": ["#4a9f6e", "#47a5e8", "#e89547", "#9f4a4a", "#6c5ce7"][i % 5]},
                 "fillcolor": ["rgba(74,159,110,0.3)", "rgba(71,165,232,0.3)", "rgba(232,149,71,0.3)", "rgba(159,74,74,0.3)", "rgba(108,92,231,0.3)"][i % 5]}
                for i, (g, data) in enumerate(zip(groups, group_data))
            ],
            "layout": {"height": 300, "yaxis": {"title": var}}
        })

        result["guide_observation"] = f"Kruskal-Wallis H = {stat:.2f}, p = {pval:.4f}. " + ("Groups differ." if pval < 0.05 else "No difference.")
        result["statistics"] = {"H_statistic": float(stat), "p_value": float(pval), "epsilon_squared": float(epsilon_sq)}

    elif analysis_id == "main_effects":
        """
        Main Effects Plot - shows how each factor affects the response.
        Essential for DOE analysis.
        """
        response = config.get("response")
        factors = config.get("factors", [])

        if not factors:
            result["summary"] = "Please select at least one factor."
            return result

        y = df[response].dropna()

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>MAIN EFFECTS PLOT<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(factors)}\n"
        summary += f"<<COLOR:highlight>>Grand Mean:<</COLOR>> {y.mean():.4f}\n\n"

        summary += f"<<COLOR:text>>Effect Sizes (deviation from grand mean):<</COLOR>>\n"

        colors = ['#4a9f6e', '#47a5e8', '#e89547', '#9f4a4a', '#6c5ce7']

        for i, factor in enumerate(factors):
            # Calculate means for each level
            factor_means = df.groupby(factor)[response].mean()

            summary += f"\n  <<COLOR:accent>>{factor}:<</COLOR>>\n"
            for level, mean_val in factor_means.items():
                effect = mean_val - y.mean()
                direction = "+" if effect > 0 else ""
                summary += f"    {level}: {mean_val:.4f} (effect: {direction}{effect:.4f})\n"

            # Create main effects plot
            levels = [str(l) for l in factor_means.index.tolist()]
            means = factor_means.values.tolist()

            result["plots"].append({
                "title": f"Main Effect: {factor}",
                "data": [{
                    "type": "scatter",
                    "x": levels,
                    "y": means,
                    "mode": "lines+markers",
                    "marker": {"color": colors[i % len(colors)], "size": 10},
                    "line": {"color": colors[i % len(colors)], "width": 2}
                }, {
                    "type": "scatter",
                    "x": levels,
                    "y": [y.mean()] * len(levels),
                    "mode": "lines",
                    "line": {"color": "#9aaa9a", "dash": "dash"},
                    "name": "Grand Mean"
                }],
                "layout": {
                    "height": 280,
                    "xaxis": {"title": factor},
                    "yaxis": {"title": f"Mean of {response}"}
                }
            })

        summary += f"\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  Steeper slopes indicate stronger effects.\n"
        summary += f"  Flat lines suggest the factor has little impact.\n"

        result["summary"] = summary
        result["guide_observation"] = f"Main effects plot for {len(factors)} factor(s). Check for steep slopes indicating strong effects."

    elif analysis_id == "interaction":
        """
        Interaction Plot - shows how factors interact.
        Non-parallel lines indicate interactions.
        """
        response = config.get("response")
        factor1 = config.get("factor1")
        factor2 = config.get("factor2")

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>INTERACTION PLOT<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>X-axis factor:<</COLOR>> {factor1}\n"
        summary += f"<<COLOR:highlight>>Trace factor:<</COLOR>> {factor2}\n\n"

        # Calculate interaction means
        interaction_means = df.groupby([factor1, factor2])[response].mean().unstack()

        summary += f"<<COLOR:text>>Cell Means:<</COLOR>>\n"
        summary += interaction_means.to_string() + "\n\n"

        # Check for interaction (compare slopes)
        levels1 = interaction_means.index.tolist()
        levels2 = interaction_means.columns.tolist()

        colors = ['#4a9f6e', '#47a5e8', '#e89547', '#9f4a4a', '#6c5ce7', '#e8c547']

        plot_data = []
        for i, lev2 in enumerate(levels2):
            means = interaction_means[lev2].tolist()
            plot_data.append({
                "type": "scatter",
                "x": [str(l) for l in levels1],
                "y": means,
                "mode": "lines+markers",
                "name": str(lev2),
                "marker": {"color": colors[i % len(colors)], "size": 8},
                "line": {"color": colors[i % len(colors)], "width": 2}
            })

        result["plots"].append({
            "title": f"Interaction: {factor1} × {factor2}",
            "data": plot_data,
            "layout": {
                "height": 350,
                "xaxis": {"title": factor1},
                "yaxis": {"title": f"Mean of {response}"},
                "legend": {"title": {"text": factor2}}
            }
        })

        # Simple interaction detection (check if lines are parallel)
        if len(levels2) >= 2 and len(levels1) >= 2:
            slopes = []
            for lev2 in levels2:
                vals = interaction_means[lev2].values
                slope = (vals[-1] - vals[0]) / (len(vals) - 1) if len(vals) > 1 else 0
                slopes.append(slope)

            slope_diff = max(slopes) - min(slopes)
            has_interaction = slope_diff > 0.1 * abs(np.mean(slopes)) if np.mean(slopes) != 0 else slope_diff > 0.1

            summary += f"<<COLOR:accent>>{'─' * 50}<</COLOR>>\n"
            if has_interaction:
                summary += f"<<COLOR:warning>>POTENTIAL INTERACTION DETECTED<</COLOR>>\n"
                summary += f"<<COLOR:text>>Lines are not parallel, suggesting {factor1} and {factor2} interact.<</COLOR>>\n"
                summary += f"<<COLOR:text>>The effect of {factor1} depends on the level of {factor2}.<</COLOR>>\n"
            else:
                summary += f"<<COLOR:good>>NO STRONG INTERACTION<</COLOR>>\n"
                summary += f"<<COLOR:text>>Lines appear roughly parallel.<</COLOR>>\n"
                summary += f"<<COLOR:text>>Factors may act independently.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"Interaction plot for {factor1} × {factor2}. " + ("Non-parallel lines suggest interaction." if 'has_interaction' in dir() and has_interaction else "Check for parallel lines.")

    elif analysis_id == "logistic":
        """
        Logistic Regression - for binary outcomes.
        Returns odds ratios, confidence intervals, and ROC curve.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

        response = config.get("response")
        predictors = config.get("predictors", [])

        if not predictors:
            result["summary"] = "Please select at least one predictor."
            return result

        # Prepare data
        X = df[predictors].dropna()
        y = df[response].loc[X.index]

        # Encode target if needed
        unique_vals = y.unique()
        if len(unique_vals) != 2:
            result["summary"] = f"Logistic regression requires binary outcome. Found {len(unique_vals)} unique values."
            return result

        # Map to 0/1
        if y.dtype == 'object' or str(y.dtype) == 'category':
            y = (y == unique_vals[1]).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Coefficients and odds ratios
        coefs = model.coef_[0]
        odds_ratios = np.exp(coefs)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>LOGISTIC REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        summary += f"<<COLOR:highlight>>AUC (ROC):<</COLOR>> {roc_auc:.4f}\n\n"

        summary += f"<<COLOR:text>>Coefficients & Odds Ratios:<</COLOR>>\n"
        summary += f"  {'Predictor':<20} {'Coef':>10} {'Odds Ratio':>12}\n"
        summary += f"  {'-'*44}\n"
        for pred, coef, odds in zip(predictors, coefs, odds_ratios):
            summary += f"  {pred:<20} {coef:>10.4f} {odds:>12.4f}\n"

        summary += f"\n<<COLOR:text>>Confusion Matrix:<</COLOR>>\n"
        summary += f"  Predicted:    0      1\n"
        summary += f"  Actual 0:  {cm[0,0]:>4}   {cm[0,1]:>4}\n"
        summary += f"  Actual 1:  {cm[1,0]:>4}   {cm[1,1]:>4}\n"

        # ROC curve plot
        result["plots"].append({
            "title": "ROC Curve",
            "data": [{
                "type": "scatter",
                "x": fpr.tolist(),
                "y": tpr.tolist(),
                "mode": "lines",
                "name": f"AUC = {roc_auc:.3f}",
                "line": {"color": "#4a9f6e", "width": 2}
            }, {
                "type": "scatter",
                "x": [0, 1],
                "y": [0, 1],
                "mode": "lines",
                "name": "Random",
                "line": {"color": "#9aaa9a", "dash": "dash"}
            }],
            "layout": {
                "height": 300,
                "xaxis": {"title": "False Positive Rate"},
                "yaxis": {"title": "True Positive Rate"}
            }
        })

        # Odds ratio forest plot
        result["plots"].append({
            "title": "Odds Ratios (Log Scale)",
            "data": [{
                "type": "bar",
                "x": odds_ratios.tolist(),
                "y": predictors,
                "orientation": "h",
                "marker": {"color": ["#4a9f6e" if o > 1 else "#e85747" for o in odds_ratios]}
            }],
            "layout": {
                "height": 250,
                "xaxis": {"type": "log", "title": "Odds Ratio"},
                "shapes": [{
                    "type": "line",
                    "x0": 1, "x1": 1,
                    "y0": -0.5, "y1": len(predictors) - 0.5,
                    "line": {"color": "#9aaa9a", "dash": "dash"}
                }]
            }
        })

        result["summary"] = summary
        result["guide_observation"] = f"Logistic regression with AUC = {roc_auc:.3f}. Odds ratios > 1 increase probability of outcome."
        result["statistics"] = {"AUC": float(roc_auc), "accuracy": float((y_pred == y_test).mean())}

    elif analysis_id == "f_test":
        """
        F-test for equality of variances between two groups.
        """
        var = config.get("var")
        group_var = config.get("group_var")

        groups = df[group_var].dropna().unique()
        if len(groups) != 2:
            result["summary"] = f"F-test requires exactly 2 groups. Found {len(groups)}."
            return result

        g1 = df[df[group_var] == groups[0]][var].dropna()
        g2 = df[df[group_var] == groups[1]][var].dropna()

        var1, var2 = g1.var(), g2.var()
        n1, n2 = len(g1), len(g2)

        # F statistic (larger variance / smaller variance)
        if var1 >= var2:
            F = var1 / var2
            df1, df2 = n1 - 1, n2 - 1
        else:
            F = var2 / var1
            df1, df2 = n2 - 1, n1 - 1

        from scipy import stats
        p_value = 2 * min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>F-TEST FOR EQUALITY OF VARIANCES<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} vs {groups[1]}\n\n"
        summary += f"<<COLOR:text>>Group Statistics:<</COLOR>>\n"
        summary += f"  {groups[0]}: n={n1}, variance={var1:.4f}, StDev={np.sqrt(var1):.4f}\n"
        summary += f"  {groups[1]}: n={n2}, variance={var2:.4f}, StDev={np.sqrt(var2):.4f}\n\n"
        summary += f"<<COLOR:highlight>>F statistic:<</COLOR>> {F:.4f}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {p_value:.4f}\n\n"

        if p_value < 0.05:
            summary += f"<<COLOR:warning>>Variances are SIGNIFICANTLY DIFFERENT (p < 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>Consider using Welch's t-test or non-parametric alternatives.<</COLOR>>\n"
        else:
            summary += f"<<COLOR:good>>No significant difference in variances (p >= 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>Equal variance assumption is reasonable.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"F = {F:.2f}, p = {p_value:.4f}. " + ("Variances differ significantly." if p_value < 0.05 else "Variances are similar.")
        result["statistics"] = {"F_statistic": float(F), "p_value": float(p_value), "variance_ratio": float(max(var1,var2)/min(var1,var2))}

    elif analysis_id == "equivalence":
        """
        TOST (Two One-Sided Tests) for equivalence testing.
        Tests whether two means are equivalent within a specified margin.
        """
        var = config.get("var")
        group_var = config.get("group_var")
        margin = float(config.get("margin", 0.5))  # Equivalence margin

        groups = df[group_var].dropna().unique()
        if len(groups) != 2:
            result["summary"] = f"Equivalence test requires exactly 2 groups. Found {len(groups)}."
            return result

        g1 = df[df[group_var] == groups[0]][var].dropna()
        g2 = df[df[group_var] == groups[1]][var].dropna()

        from scipy import stats

        mean1, mean2 = g1.mean(), g2.mean()
        std1, std2 = g1.std(), g2.std()
        n1, n2 = len(g1), len(g2)

        diff = mean1 - mean2
        se = np.sqrt(std1**2/n1 + std2**2/n2)
        df_val = n1 + n2 - 2

        # TOST: Two one-sided tests
        t_lower = (diff - (-margin)) / se
        t_upper = (diff - margin) / se

        p_lower = 1 - stats.t.cdf(t_lower, df_val)
        p_upper = stats.t.cdf(t_upper, df_val)
        p_tost = max(p_lower, p_upper)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>EQUIVALENCE TEST (TOST)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Groups:<</COLOR>> {groups[0]} vs {groups[1]}\n"
        summary += f"<<COLOR:highlight>>Equivalence margin:<</COLOR>> ±{margin}\n\n"
        summary += f"<<COLOR:text>>Group Means:<</COLOR>>\n"
        summary += f"  {groups[0]}: {mean1:.4f} (n={n1})\n"
        summary += f"  {groups[1]}: {mean2:.4f} (n={n2})\n"
        summary += f"  Difference: {diff:.4f}\n\n"
        summary += f"<<COLOR:highlight>>TOST p-value:<</COLOR>> {p_tost:.4f}\n\n"

        if p_tost < 0.05:
            summary += f"<<COLOR:good>>EQUIVALENT within ±{margin} (p < 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>The difference is small enough to be considered equivalent.<</COLOR>>\n"
        else:
            summary += f"<<COLOR:warning>>NOT EQUIVALENT (p >= 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>Cannot conclude equivalence within the specified margin.<</COLOR>>\n"

        # Equivalence plot
        result["plots"].append({
            "title": "Equivalence Plot",
            "data": [{
                "type": "scatter",
                "x": [diff],
                "y": [0.5],
                "mode": "markers",
                "marker": {"size": 15, "color": "#4a9f6e" if p_tost < 0.05 else "#e85747"},
                "error_x": {"type": "constant", "value": 1.96 * se, "color": "#4a9f6e"},
                "name": "Mean Difference"
            }],
            "layout": {
                "height": 200,
                "xaxis": {"title": "Difference", "zeroline": True},
                "yaxis": {"visible": False, "range": [0, 1]},
                "shapes": [
                    {"type": "line", "x0": -margin, "x1": -margin, "y0": 0, "y1": 1, "line": {"color": "#e89547", "dash": "dash"}},
                    {"type": "line", "x0": margin, "x1": margin, "y0": 0, "y1": 1, "line": {"color": "#e89547", "dash": "dash"}},
                    {"type": "rect", "x0": -margin, "x1": margin, "y0": 0, "y1": 1, "fillcolor": "rgba(74,159,110,0.1)", "line": {"width": 0}}
                ]
            }
        })

        result["summary"] = summary
        result["guide_observation"] = f"TOST p = {p_tost:.4f}. " + (f"Groups equivalent within ±{margin}." if p_tost < 0.05 else "Cannot confirm equivalence.")
        result["statistics"] = {"TOST_p_value": float(p_tost), "mean_difference": float(diff), "margin": float(margin)}

    elif analysis_id == "runs_test":
        """
        Runs test for randomness in a sequence.
        Tests whether the sequence is random or has patterns.
        """
        var = config.get("var")
        data = df[var].dropna().values

        from scipy import stats

        # Convert to binary (above/below median)
        median = np.median(data)
        binary = (data > median).astype(int)

        # Count runs
        n_runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                n_runs += 1

        n_pos = np.sum(binary)
        n_neg = len(binary) - n_pos
        n = len(binary)

        # Expected runs and variance
        expected_runs = (2 * n_pos * n_neg / n) + 1
        var_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))

        # Z-score
        z = (n_runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>RUNS TEST FOR RANDOMNESS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n\n"
        summary += f"<<COLOR:text>>Run Statistics:<</COLOR>>\n"
        summary += f"  Observed runs: {n_runs}\n"
        summary += f"  Expected runs: {expected_runs:.2f}\n"
        summary += f"  Above median: {n_pos}\n"
        summary += f"  Below median: {n_neg}\n\n"
        summary += f"<<COLOR:highlight>>Z-statistic:<</COLOR>> {z:.4f}\n"
        summary += f"<<COLOR:highlight>>p-value:<</COLOR>> {p_value:.4f}\n\n"

        if p_value < 0.05:
            if n_runs < expected_runs:
                summary += f"<<COLOR:warning>>SEQUENCE IS NOT RANDOM - Too few runs (clustering)<</COLOR>>\n"
                summary += f"<<COLOR:text>>Values tend to cluster together, suggesting trends or patterns.<</COLOR>>\n"
            else:
                summary += f"<<COLOR:warning>>SEQUENCE IS NOT RANDOM - Too many runs (oscillation)<</COLOR>>\n"
                summary += f"<<COLOR:text>>Values alternate too frequently, suggesting negative autocorrelation.<</COLOR>>\n"
        else:
            summary += f"<<COLOR:good>>SEQUENCE APPEARS RANDOM (p >= 0.05)<</COLOR>>\n"
            summary += f"<<COLOR:text>>No evidence of patterns or trends in the data.<</COLOR>>\n"

        # Plot the sequence with run coloring
        colors = []
        current_color = "#4a9f6e"
        for i in range(len(binary)):
            if i > 0 and binary[i] != binary[i-1]:
                current_color = "#47a5e8" if current_color == "#4a9f6e" else "#4a9f6e"
            colors.append(current_color)

        result["plots"].append({
            "title": f"Sequence Plot ({n_runs} runs)",
            "data": [{
                "type": "scatter",
                "y": data.tolist(),
                "mode": "lines+markers",
                "marker": {"color": colors, "size": 6},
                "line": {"color": "rgba(74,159,110,0.3)"}
            }, {
                "type": "scatter",
                "y": [median] * len(data),
                "mode": "lines",
                "name": "Median",
                "line": {"color": "#e89547", "dash": "dash"}
            }],
            "layout": {"height": 250, "xaxis": {"title": "Observation"}, "yaxis": {"title": var}}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Runs test: {n_runs} runs, p = {p_value:.4f}. " + ("Non-random pattern detected." if p_value < 0.05 else "Sequence appears random.")
        result["statistics"] = {"runs": int(n_runs), "expected_runs": float(expected_runs), "Z_statistic": float(z), "p_value": float(p_value)}

    elif analysis_id == "multi_vari":
        """
        Multi-Vari Chart - shows variation across multiple factors.
        Essential for understanding sources of variation.
        """
        response = config.get("response")
        factors = config.get("factors", [])

        if len(factors) < 1:
            result["summary"] = "Please select at least one factor."
            return result

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>MULTI-VARI CHART<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Factors:<</COLOR>> {', '.join(factors)}\n\n"

        colors = ['#4a9f6e', '#47a5e8', '#e89547', '#9f4a4a', '#6c5ce7']

        if len(factors) == 1:
            # Single factor multi-vari
            factor = factors[0]
            groups = df.groupby(factor)[response]

            plot_data = []
            x_positions = []
            x_labels = []

            for i, (level, group_data) in enumerate(groups):
                vals = group_data.dropna().values
                x_pos = i
                x_positions.append(x_pos)
                x_labels.append(str(level))

                # Individual points
                plot_data.append({
                    "type": "scatter",
                    "x": [x_pos + np.random.uniform(-0.1, 0.1) for _ in vals],
                    "y": vals.tolist(),
                    "mode": "markers",
                    "marker": {"color": colors[i % len(colors)], "size": 6, "opacity": 0.6},
                    "showlegend": False
                })

                # Mean marker
                plot_data.append({
                    "type": "scatter",
                    "x": [x_pos],
                    "y": [vals.mean()],
                    "mode": "markers",
                    "marker": {"color": colors[i % len(colors)], "size": 12, "symbol": "diamond"},
                    "showlegend": False
                })

                summary += f"  {level}: n={len(vals)}, mean={vals.mean():.4f}, std={vals.std():.4f}\n"

            # Connect means
            means = [groups.get_group(l).mean() for l in df[factor].dropna().unique()]
            plot_data.append({
                "type": "scatter",
                "x": x_positions,
                "y": means,
                "mode": "lines",
                "line": {"color": "#e89547", "width": 2},
                "showlegend": False
            })

            result["plots"].append({
                "title": f"Multi-Vari: {response} by {factor}",
                "data": plot_data,
                "layout": {
                    "height": 300,
                    "xaxis": {"tickvals": x_positions, "ticktext": x_labels, "title": factor},
                    "yaxis": {"title": response}
                }
            })

        else:
            # Two or more factors - nested multi-vari
            factor1, factor2 = factors[0], factors[1]
            levels1 = df[factor1].dropna().unique()
            levels2 = df[factor2].dropna().unique()

            plot_data = []
            x_positions = []
            x_labels = []
            pos = 0

            for i, lev1 in enumerate(levels1):
                group_means = []
                for j, lev2 in enumerate(levels2):
                    mask = (df[factor1] == lev1) & (df[factor2] == lev2)
                    vals = df.loc[mask, response].dropna().values

                    if len(vals) > 0:
                        x_positions.append(pos)
                        x_labels.append(f"{lev2}")

                        # Individual points
                        plot_data.append({
                            "type": "scatter",
                            "x": [pos + np.random.uniform(-0.15, 0.15) for _ in vals],
                            "y": vals.tolist(),
                            "mode": "markers",
                            "marker": {"color": colors[i % len(colors)], "size": 5, "opacity": 0.5},
                            "showlegend": False
                        })

                        # Mean
                        group_means.append((pos, vals.mean()))
                        pos += 1

                # Connect means within factor1 level
                if group_means:
                    plot_data.append({
                        "type": "scatter",
                        "x": [g[0] for g in group_means],
                        "y": [g[1] for g in group_means],
                        "mode": "lines+markers",
                        "marker": {"color": colors[i % len(colors)], "size": 10, "symbol": "diamond"},
                        "line": {"color": colors[i % len(colors)], "width": 2},
                        "name": str(lev1)
                    })

                pos += 0.5  # Gap between factor1 levels

            result["plots"].append({
                "title": f"Multi-Vari: {response} by {factor1}/{factor2}",
                "data": plot_data,
                "layout": {
                    "height": 350,
                    "xaxis": {"tickvals": x_positions, "ticktext": x_labels, "title": factor2},
                    "yaxis": {"title": response},
                    "showlegend": True
                }
            })

            summary += f"<<COLOR:text>>Nested structure: {factor1} (colors) → {factor2} (x-axis)<</COLOR>>\n\n"

        summary += f"\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  • Vertical spread within groups = within-group variation\n"
        summary += f"  • Differences between group means = between-group variation\n"
        summary += f"  • Compare spreads to identify dominant sources of variation\n"

        result["summary"] = summary
        result["guide_observation"] = f"Multi-vari chart showing variation in {response} across {len(factors)} factor(s)."

    elif analysis_id == "arima":
        """
        ARIMA Time Series Analysis - fit and forecast.
        """
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller

        var = config.get("var")
        p = int(config.get("p", 1))  # AR order
        d = int(config.get("d", 1))  # Differencing
        q = int(config.get("q", 1))  # MA order
        forecast_periods = int(config.get("forecast", 10))

        data = df[var].dropna().values

        # Stationarity test
        adf_result = adfuller(data)
        adf_stat, adf_pval = adf_result[0], adf_result[1]

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ARIMA TIME SERIES ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> ARIMA({p},{d},{q})\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n\n"

        summary += f"<<COLOR:text>>Stationarity Test (ADF):<</COLOR>>\n"
        summary += f"  ADF Statistic: {adf_stat:.4f}\n"
        summary += f"  p-value: {adf_pval:.4f}\n"
        summary += f"  {'Stationary' if adf_pval < 0.05 else 'Non-stationary (differencing recommended)'}\n\n"

        try:
            model = ARIMA(data, order=(p, d, q))
            fitted = model.fit()

            # Model summary
            summary += f"<<COLOR:text>>Model Parameters:<</COLOR>>\n"
            summary += f"  AIC: {fitted.aic:.2f}\n"
            summary += f"  BIC: {fitted.bic:.2f}\n\n"

            # Forecast
            forecast = fitted.get_forecast(steps=forecast_periods)
            fc_mean = forecast.predicted_mean
            fc_ci = forecast.conf_int()

            summary += f"<<COLOR:text>>Forecast ({forecast_periods} periods):<</COLOR>>\n"
            for i in range(min(5, forecast_periods)):
                summary += f"  Period {i+1}: {fc_mean.iloc[i]:.4f} [{fc_ci.iloc[i, 0]:.4f}, {fc_ci.iloc[i, 1]:.4f}]\n"
            if forecast_periods > 5:
                summary += f"  ... ({forecast_periods - 5} more periods)\n"

            # Plot
            x_hist = list(range(len(data)))
            x_fc = list(range(len(data), len(data) + forecast_periods))

            result["plots"].append({
                "title": f"ARIMA({p},{d},{q}) Forecast",
                "data": [
                    {"type": "scatter", "x": x_hist, "y": data.tolist(), "mode": "lines", "name": "Historical", "line": {"color": "#4a9f6e"}},
                    {"type": "scatter", "x": x_fc, "y": fc_mean.tolist(), "mode": "lines", "name": "Forecast", "line": {"color": "#e89547"}},
                    {"type": "scatter", "x": x_fc + x_fc[::-1], "y": fc_ci.iloc[:, 1].tolist() + fc_ci.iloc[::-1, 0].tolist(),
                     "fill": "toself", "fillcolor": "rgba(232,149,71,0.2)", "line": {"color": "transparent"}, "name": "95% CI"}
                ],
                "layout": {"height": 300, "xaxis": {"title": "Time"}, "yaxis": {"title": var}}
            })

            # Residual diagnostics
            residuals = fitted.resid

            result["plots"].append({
                "title": "Residuals",
                "data": [{"type": "scatter", "y": residuals.tolist(), "mode": "lines", "line": {"color": "#4a9f6e"}}],
                "layout": {"height": 200, "xaxis": {"title": "Time"}, "yaxis": {"title": "Residual"}}
            })

            result["statistics"] = {"AIC": float(fitted.aic), "BIC": float(fitted.bic), "ADF_pvalue": float(adf_pval)}

        except Exception as e:
            summary += f"<<COLOR:warning>>Model fitting failed: {str(e)}<</COLOR>>\n"
            summary += f"<<COLOR:text>>Try different p, d, q values or check data for issues.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"ARIMA({p},{d},{q}) model. {'Stationary data.' if adf_pval < 0.05 else 'Consider differencing.'}"

    elif analysis_id == "decomposition":
        """
        Time Series Decomposition - trend, seasonal, residual.
        """
        from statsmodels.tsa.seasonal import seasonal_decompose

        var = config.get("var")
        period = int(config.get("period", 12))
        model_type = config.get("model", "additive")  # additive or multiplicative

        data = df[var].dropna()

        if len(data) < 2 * period:
            result["summary"] = f"Need at least {2 * period} observations for period={period}. Have {len(data)}."
            return result

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>TIME SERIES DECOMPOSITION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> {model_type.capitalize()}\n"
        summary += f"<<COLOR:highlight>>Period:<</COLOR>> {period}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n\n"

        decomp = seasonal_decompose(data, model=model_type, period=period)

        # Trend statistics
        trend_clean = decomp.trend.dropna()
        summary += f"<<COLOR:text>>Trend:<</COLOR>>\n"
        summary += f"  Start: {trend_clean.iloc[0]:.4f}\n"
        summary += f"  End: {trend_clean.iloc[-1]:.4f}\n"
        summary += f"  Change: {trend_clean.iloc[-1] - trend_clean.iloc[0]:.4f}\n\n"

        # Seasonal strength
        seasonal_var = decomp.seasonal.var()
        resid_var = decomp.resid.dropna().var()
        seasonal_strength = 1 - (resid_var / (seasonal_var + resid_var)) if (seasonal_var + resid_var) > 0 else 0

        summary += f"<<COLOR:text>>Seasonal Strength:<</COLOR>> {seasonal_strength:.2%}\n"
        if seasonal_strength > 0.6:
            summary += f"  <<COLOR:accent>>Strong seasonality detected<</COLOR>>\n"
        elif seasonal_strength > 0.3:
            summary += f"  Moderate seasonality\n"
        else:
            summary += f"  Weak or no seasonality\n"

        x_vals = list(range(len(data)))

        # Original
        result["plots"].append({
            "title": "Original Series",
            "data": [{"type": "scatter", "x": x_vals, "y": data.tolist(), "mode": "lines", "line": {"color": "#4a9f6e"}}],
            "layout": {"height": 150}
        })

        # Trend
        result["plots"].append({
            "title": "Trend",
            "data": [{"type": "scatter", "x": x_vals, "y": decomp.trend.tolist(), "mode": "lines", "line": {"color": "#47a5e8"}}],
            "layout": {"height": 150}
        })

        # Seasonal
        result["plots"].append({
            "title": "Seasonal",
            "data": [{"type": "scatter", "x": x_vals, "y": decomp.seasonal.tolist(), "mode": "lines", "line": {"color": "#e89547"}}],
            "layout": {"height": 150}
        })

        # Residual
        result["plots"].append({
            "title": "Residual",
            "data": [{"type": "scatter", "x": x_vals, "y": decomp.resid.tolist(), "mode": "lines", "line": {"color": "#9aaa9a"}}],
            "layout": {"height": 150}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Decomposition with {seasonal_strength:.0%} seasonal strength."
        result["statistics"] = {"seasonal_strength": float(seasonal_strength), "trend_change": float(trend_clean.iloc[-1] - trend_clean.iloc[0])}

    elif analysis_id == "acf_pacf":
        """
        Autocorrelation and Partial Autocorrelation plots.
        Essential for ARIMA model identification.
        """
        from statsmodels.tsa.stattools import acf, pacf

        var = config.get("var")
        max_lags = int(config.get("lags", 20))

        data = df[var].dropna().values

        acf_vals = acf(data, nlags=max_lags)
        pacf_vals = pacf(data, nlags=max_lags)

        # Confidence interval (95%)
        ci = 1.96 / np.sqrt(len(data))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ACF / PACF ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n"
        summary += f"<<COLOR:highlight>>Lags shown:<</COLOR>> {max_lags}\n\n"

        # Find significant lags
        sig_acf = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > ci]
        sig_pacf = [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > ci]

        summary += f"<<COLOR:text>>Significant ACF lags:<</COLOR>> {sig_acf[:5] if sig_acf else 'None'}\n"
        summary += f"<<COLOR:text>>Significant PACF lags:<</COLOR>> {sig_pacf[:5] if sig_pacf else 'None'}\n\n"

        # ARIMA order suggestions
        summary += f"<<COLOR:success>>ARIMA ORDER SUGGESTIONS:<</COLOR>>\n"
        if len(sig_pacf) > 0 and (len(sig_acf) == 0 or sig_acf[0] > sig_pacf[-1]):
            summary += f"  PACF cuts off at lag {max(sig_pacf)}: Try AR({max(sig_pacf)})\n"
        if len(sig_acf) > 0 and (len(sig_pacf) == 0 or sig_pacf[0] > sig_acf[-1]):
            summary += f"  ACF cuts off at lag {max(sig_acf)}: Try MA({max(sig_acf)})\n"
        if len(sig_acf) > 0 and len(sig_pacf) > 0:
            summary += f"  Both taper: Try ARMA({min(3, max(sig_pacf))},{min(3, max(sig_acf))})\n"

        lags = list(range(max_lags + 1))

        # ACF plot
        result["plots"].append({
            "title": "Autocorrelation Function (ACF)",
            "data": [
                {"type": "bar", "x": lags, "y": acf_vals.tolist(), "marker": {"color": "#4a9f6e"}},
                {"type": "scatter", "x": lags, "y": [ci] * len(lags), "mode": "lines", "line": {"color": "#e85747", "dash": "dash"}, "showlegend": False},
                {"type": "scatter", "x": lags, "y": [-ci] * len(lags), "mode": "lines", "line": {"color": "#e85747", "dash": "dash"}, "showlegend": False}
            ],
            "layout": {"height": 250, "xaxis": {"title": "Lag"}, "yaxis": {"title": "ACF", "range": [-1, 1]}}
        })

        # PACF plot
        result["plots"].append({
            "title": "Partial Autocorrelation Function (PACF)",
            "data": [
                {"type": "bar", "x": lags, "y": pacf_vals.tolist(), "marker": {"color": "#47a5e8"}},
                {"type": "scatter", "x": lags, "y": [ci] * len(lags), "mode": "lines", "line": {"color": "#e85747", "dash": "dash"}, "showlegend": False},
                {"type": "scatter", "x": lags, "y": [-ci] * len(lags), "mode": "lines", "line": {"color": "#e85747", "dash": "dash"}, "showlegend": False}
            ],
            "layout": {"height": 250, "xaxis": {"title": "Lag"}, "yaxis": {"title": "PACF", "range": [-1, 1]}}
        })

        result["summary"] = summary
        result["guide_observation"] = f"ACF/PACF analysis. Significant lags help identify ARIMA orders."

    elif analysis_id == "weibull":
        """
        Weibull Reliability Analysis - life data analysis.
        """
        from scipy import stats
        from scipy.optimize import minimize

        var = config.get("var")  # Time to failure
        censored = config.get("censored")  # Optional censoring indicator

        data = df[var].dropna().values
        data = data[data > 0]  # Weibull requires positive values

        if len(data) < 3:
            result["summary"] = "Need at least 3 data points for Weibull analysis."
            return result

        # Fit Weibull distribution
        shape, loc, scale = stats.weibull_min.fit(data, floc=0)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>WEIBULL RELIABILITY ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var} (time to failure)\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n\n"

        summary += f"<<COLOR:text>>Weibull Parameters:<</COLOR>>\n"
        summary += f"  Shape (β): {shape:.4f}\n"
        summary += f"  Scale (η): {scale:.4f}\n\n"

        # Interpret shape parameter
        summary += f"<<COLOR:text>>Shape Interpretation:<</COLOR>>\n"
        if shape < 1:
            summary += f"  β < 1: <<COLOR:warning>>Infant mortality / early failures<</COLOR>>\n"
            summary += f"  Failure rate DECREASES over time (burn-in period)\n"
        elif shape == 1:
            summary += f"  β ≈ 1: Random failures (exponential distribution)\n"
            summary += f"  Failure rate is CONSTANT\n"
        else:
            summary += f"  β > 1: <<COLOR:accent>>Wear-out failures<</COLOR>>\n"
            summary += f"  Failure rate INCREASES over time\n"

        # Reliability metrics
        mean_life = scale * np.exp(np.log(np.exp(1)) / shape) if shape > 0 else scale
        b10 = stats.weibull_min.ppf(0.10, shape, 0, scale)  # 10% failure life
        b50 = stats.weibull_min.ppf(0.50, shape, 0, scale)  # Median life

        summary += f"\n<<COLOR:text>>Reliability Metrics:<</COLOR>>\n"
        summary += f"  Mean Life (MTTF): {mean_life:.2f}\n"
        summary += f"  B10 Life (10% fail): {b10:.2f}\n"
        summary += f"  B50 Life (median): {b50:.2f}\n"

        # Reliability at specific times
        summary += f"\n<<COLOR:text>>Reliability R(t):<</COLOR>>\n"
        for t in [b10, b50, scale, 2*scale]:
            r = 1 - stats.weibull_min.cdf(t, shape, 0, scale)
            summary += f"  R({t:.1f}) = {r:.2%}\n"

        # Weibull probability plot
        sorted_data = np.sort(data)
        n = len(sorted_data)
        rank = np.arange(1, n + 1)
        median_rank = (rank - 0.3) / (n + 0.4)  # Median rank approximation

        # Linearized Weibull
        x_plot = np.log(sorted_data)
        y_plot = np.log(-np.log(1 - median_rank))

        # Fitted line
        x_fit = np.linspace(x_plot.min(), x_plot.max(), 100)
        y_fit = shape * (x_fit - np.log(scale))

        result["plots"].append({
            "title": "Weibull Probability Plot",
            "data": [
                {"type": "scatter", "x": x_plot.tolist(), "y": y_plot.tolist(), "mode": "markers", "name": "Data", "marker": {"color": "#4a9f6e", "size": 8}},
                {"type": "scatter", "x": x_fit.tolist(), "y": y_fit.tolist(), "mode": "lines", "name": "Fit", "line": {"color": "#e89547"}}
            ],
            "layout": {"height": 300, "xaxis": {"title": "ln(Time)"}, "yaxis": {"title": "ln(-ln(1-F))"}}
        })

        # Reliability curve
        t_range = np.linspace(0, 2 * scale, 100)
        reliability = 1 - stats.weibull_min.cdf(t_range, shape, 0, scale)
        hazard = (shape / scale) * (t_range / scale) ** (shape - 1)

        result["plots"].append({
            "title": "Reliability & Hazard Functions",
            "data": [
                {"type": "scatter", "x": t_range.tolist(), "y": reliability.tolist(), "mode": "lines", "name": "R(t)", "line": {"color": "#4a9f6e"}},
                {"type": "scatter", "x": t_range.tolist(), "y": (hazard / hazard.max()).tolist(), "mode": "lines", "name": "h(t) scaled", "line": {"color": "#e85747", "dash": "dash"}, "yaxis": "y2"}
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Reliability", "range": [0, 1]},
                "yaxis2": {"title": "Hazard (scaled)", "overlaying": "y", "side": "right"}
            }
        })

        result["summary"] = summary
        result["guide_observation"] = f"Weibull β={shape:.2f} ({'wear-out' if shape > 1 else 'early failure' if shape < 1 else 'random'}), η={scale:.2f}."
        result["statistics"] = {"shape_beta": float(shape), "scale_eta": float(scale), "MTTF": float(mean_life), "B10": float(b10)}

    elif analysis_id == "kaplan_meier":
        """
        Kaplan-Meier Survival Analysis.
        """
        time_var = config.get("time")
        event_var = config.get("event")  # 1 = event occurred, 0 = censored
        group_var = config.get("group")  # Optional grouping

        times = df[time_var].dropna().values
        events = df[event_var].dropna().values if event_var else np.ones(len(times))

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>KAPLAN-MEIER SURVIVAL ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Time variable:<</COLOR>> {time_var}\n"
        summary += f"<<COLOR:highlight>>Event variable:<</COLOR>> {event_var or 'None (all events)'}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(times)}\n"
        summary += f"<<COLOR:highlight>>Events:<</COLOR>> {int(events.sum())}\n"
        summary += f"<<COLOR:highlight>>Censored:<</COLOR>> {int(len(events) - events.sum())}\n\n"

        def kaplan_meier_estimator(t, e):
            """Calculate KM survival curve."""
            # Sort by time
            idx = np.argsort(t)
            t_sorted = t[idx]
            e_sorted = e[idx]

            # Unique event times
            unique_times = np.unique(t_sorted[e_sorted == 1])

            survival = [1.0]
            times_out = [0]
            at_risk = len(t)

            for ut in unique_times:
                # Number of events and at risk at this time
                d = np.sum((t_sorted == ut) & (e_sorted == 1))
                n = np.sum(t_sorted >= ut)

                if n > 0:
                    survival.append(survival[-1] * (1 - d / n))
                    times_out.append(ut)

            return np.array(times_out), np.array(survival)

        colors = ['#4a9f6e', '#47a5e8', '#e89547', '#9f4a4a']

        if group_var and group_var != "" and group_var != "None":
            # Grouped analysis
            groups = df[group_var].dropna().unique()
            plot_data = []

            for i, grp in enumerate(groups):
                mask = df[group_var] == grp
                t_grp = df.loc[mask, time_var].dropna().values
                e_grp = df.loc[mask, event_var].dropna().values if event_var else np.ones(len(t_grp))

                km_t, km_s = kaplan_meier_estimator(t_grp, e_grp)

                # Median survival
                median_idx = np.where(km_s <= 0.5)[0]
                median_surv = km_t[median_idx[0]] if len(median_idx) > 0 else "> max observed"

                summary += f"<<COLOR:text>>{grp}:<</COLOR>>\n"
                summary += f"  n={len(t_grp)}, events={int(e_grp.sum())}\n"
                summary += f"  Median survival: {median_surv}\n\n"

                # Step function plot
                x_step = []
                y_step = []
                for j in range(len(km_t) - 1):
                    x_step.extend([km_t[j], km_t[j+1]])
                    y_step.extend([km_s[j], km_s[j]])
                x_step.append(km_t[-1])
                y_step.append(km_s[-1])

                plot_data.append({
                    "type": "scatter",
                    "x": x_step,
                    "y": y_step,
                    "mode": "lines",
                    "name": str(grp),
                    "line": {"color": colors[i % len(colors)], "width": 2}
                })

            result["plots"].append({
                "title": "Kaplan-Meier Survival Curves",
                "data": plot_data,
                "layout": {
                    "height": 350,
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Survival Probability", "range": [0, 1.05]},
                    "showlegend": True
                }
            })

        else:
            # Single group
            km_t, km_s = kaplan_meier_estimator(times, events)

            median_idx = np.where(km_s <= 0.5)[0]
            median_surv = km_t[median_idx[0]] if len(median_idx) > 0 else f"> {times.max():.2f}"

            summary += f"<<COLOR:text>>Survival Estimates:<</COLOR>>\n"
            for prob in [0.75, 0.5, 0.25]:
                idx = np.where(km_s <= prob)[0]
                if len(idx) > 0:
                    summary += f"  {int(prob*100)}% survival at t = {km_t[idx[0]]:.2f}\n"

            summary += f"\n<<COLOR:highlight>>Median survival time:<</COLOR>> {median_surv}\n"

            # Step function
            x_step = []
            y_step = []
            for j in range(len(km_t) - 1):
                x_step.extend([km_t[j], km_t[j+1]])
                y_step.extend([km_s[j], km_s[j]])
            x_step.append(km_t[-1])
            y_step.append(km_s[-1])

            result["plots"].append({
                "title": "Kaplan-Meier Survival Curve",
                "data": [{
                    "type": "scatter",
                    "x": x_step,
                    "y": y_step,
                    "mode": "lines",
                    "line": {"color": "#4a9f6e", "width": 2}
                }],
                "layout": {
                    "height": 300,
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Survival Probability", "range": [0, 1.05]},
                    "shapes": [{
                        "type": "line", "x0": 0, "x1": times.max(),
                        "y0": 0.5, "y1": 0.5,
                        "line": {"color": "#e89547", "dash": "dash"}
                    }]
                }
            })

        result["summary"] = summary
        result["guide_observation"] = f"Kaplan-Meier survival analysis with {int(events.sum())} events."

    elif analysis_id == "gage_rr":
        """
        Gage R&R (Repeatability and Reproducibility) Study.
        Measurement System Analysis for continuous data.
        """
        measurement = config.get("measurement")
        part = config.get("part")
        operator = config.get("operator")

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>GAGE R&R STUDY<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Measurement:<</COLOR>> {measurement}\n"
        summary += f"<<COLOR:highlight>>Part:<</COLOR>> {part}\n"
        summary += f"<<COLOR:highlight>>Operator:<</COLOR>> {operator}\n\n"

        # Get data
        data = df[[measurement, part, operator]].dropna()

        n_parts = data[part].nunique()
        n_operators = data[operator].nunique()
        n_replicates = len(data) // (n_parts * n_operators)

        summary += f"<<COLOR:text>>Study Design:<</COLOR>>\n"
        summary += f"  Parts: {n_parts}\n"
        summary += f"  Operators: {n_operators}\n"
        summary += f"  Replicates: {n_replicates}\n\n"

        # Calculate variance components using ANOVA
        from scipy import stats

        grand_mean = data[measurement].mean()
        total_var = data[measurement].var()

        # Part variation
        part_means = data.groupby(part)[measurement].mean()
        part_var = part_means.var() * n_operators * n_replicates

        # Operator variation
        op_means = data.groupby(operator)[measurement].mean()
        op_var = op_means.var() * n_parts * n_replicates

        # Repeatability (within operator-part)
        repeatability_var = data.groupby([part, operator])[measurement].var().mean()

        # Reproducibility (between operators)
        reproducibility_var = max(0, op_var - repeatability_var / (n_parts * n_replicates))

        # Gage R&R
        gage_rr_var = repeatability_var + reproducibility_var
        total_variation = gage_rr_var + part_var

        # Percentages
        pct_repeatability = 100 * np.sqrt(repeatability_var / total_variation) if total_variation > 0 else 0
        pct_reproducibility = 100 * np.sqrt(reproducibility_var / total_variation) if total_variation > 0 else 0
        pct_gage_rr = 100 * np.sqrt(gage_rr_var / total_variation) if total_variation > 0 else 0
        pct_part = 100 * np.sqrt(part_var / total_variation) if total_variation > 0 else 0

        summary += f"<<COLOR:text>>Variance Components:<</COLOR>>\n"
        summary += f"  {'Source':<20} {'Variance':>12} {'%Contribution':>14}\n"
        summary += f"  {'-'*48}\n"
        summary += f"  {'Total Gage R&R':<20} {gage_rr_var:>12.4f} {pct_gage_rr:>13.1f}%\n"
        summary += f"    {'Repeatability':<18} {repeatability_var:>12.4f} {pct_repeatability:>13.1f}%\n"
        summary += f"    {'Reproducibility':<18} {reproducibility_var:>12.4f} {pct_reproducibility:>13.1f}%\n"
        summary += f"  {'Part-to-Part':<20} {part_var:>12.4f} {pct_part:>13.1f}%\n\n"

        # Assessment
        summary += f"<<COLOR:success>>ASSESSMENT:<</COLOR>>\n"
        if pct_gage_rr < 10:
            summary += f"  <<COLOR:good>>EXCELLENT - Gage R&R < 10%<</COLOR>>\n"
            summary += f"  Measurement system is acceptable.\n"
        elif pct_gage_rr < 30:
            summary += f"  <<COLOR:warning>>MARGINAL - Gage R&R 10-30%<</COLOR>>\n"
            summary += f"  May be acceptable depending on application.\n"
        else:
            summary += f"  <<COLOR:bad>>UNACCEPTABLE - Gage R&R > 30%<</COLOR>>\n"
            summary += f"  Measurement system needs improvement.\n"

        # Number of distinct categories
        ndc = int(1.41 * np.sqrt(part_var / gage_rr_var)) if gage_rr_var > 0 else 0
        summary += f"\n<<COLOR:highlight>>Number of Distinct Categories:<</COLOR>> {ndc}\n"
        summary += f"  (Should be >= 5 for adequate discrimination)\n"

        # Plots
        # By Part
        result["plots"].append({
            "title": f"{measurement} by {part}",
            "data": [{
                "type": "box",
                "x": data[part].astype(str).tolist(),
                "y": data[measurement].tolist(),
                "marker": {"color": "#4a9f6e"}
            }],
            "layout": {"height": 250, "xaxis": {"title": part}}
        })

        # By Operator
        result["plots"].append({
            "title": f"{measurement} by {operator}",
            "data": [{
                "type": "box",
                "x": data[operator].astype(str).tolist(),
                "y": data[measurement].tolist(),
                "marker": {"color": "#47a5e8"}
            }],
            "layout": {"height": 250, "xaxis": {"title": operator}}
        })

        # Components of variation bar chart
        result["plots"].append({
            "title": "Components of Variation",
            "data": [{
                "type": "bar",
                "x": ["Gage R&R", "Repeatability", "Reproducibility", "Part-to-Part"],
                "y": [pct_gage_rr, pct_repeatability, pct_reproducibility, pct_part],
                "marker": {"color": ["#e89547", "#4a9f6e", "#47a5e8", "#9aaa9a"]}
            }],
            "layout": {"height": 250, "yaxis": {"title": "% of Total Variation"}}
        })

        result["summary"] = summary
        result["guide_observation"] = f"Gage R&R = {pct_gage_rr:.1f}%. " + ("Acceptable." if pct_gage_rr < 30 else "Needs improvement.")
        result["statistics"] = {"gage_rr_pct": float(pct_gage_rr), "repeatability_pct": float(pct_repeatability), "reproducibility_pct": float(pct_reproducibility), "ndc": int(ndc)}

    elif analysis_id == "stepwise":
        """
        Stepwise Regression - automatic variable selection.
        Uses forward selection, backward elimination, or both.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        import statsmodels.api as sm

        response = config.get("response")
        predictors = config.get("predictors", [])
        method = config.get("method", "both")  # forward, backward, both
        alpha_enter = float(config.get("alpha_enter", 0.05))
        alpha_remove = float(config.get("alpha_remove", 0.10))

        X_full = df[predictors].dropna()
        y = df[response].loc[X_full.index]
        n = len(y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>STEPWISE REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Candidate Predictors:<</COLOR>> {len(predictors)}\n"
        summary += f"<<COLOR:highlight>>Method:<</COLOR>> {method}\n"
        summary += f"<<COLOR:highlight>>α to enter:<</COLOR>> {alpha_enter}\n"
        summary += f"<<COLOR:highlight>>α to remove:<</COLOR>> {alpha_remove}\n\n"

        selected = []
        remaining = list(predictors)
        step_history = []

        def get_pvalue(X_subset, y_data, var_name):
            """Get p-value for adding a variable."""
            X_with_const = sm.add_constant(X_subset)
            try:
                model = sm.OLS(y_data, X_with_const).fit()
                idx = list(X_subset.columns).index(var_name) + 1  # +1 for constant
                return model.pvalues.iloc[idx]
            except:
                return 1.0

        # Stepwise selection
        step = 0
        while True:
            step += 1
            changed = False

            # Forward step: try adding variables
            if method in ["forward", "both"] and remaining:
                best_pval = 1.0
                best_var = None

                for var in remaining:
                    if selected:
                        X_test = X_full[selected + [var]]
                    else:
                        X_test = X_full[[var]]

                    pval = get_pvalue(X_test, y, var)
                    if pval < best_pval:
                        best_pval = pval
                        best_var = var

                if best_var and best_pval < alpha_enter:
                    selected.append(best_var)
                    remaining.remove(best_var)
                    step_history.append(f"Step {step}: ADD {best_var} (p={best_pval:.4f})")
                    changed = True

            # Backward step: try removing variables
            if method in ["backward", "both"] and selected:
                worst_pval = 0.0
                worst_var = None

                X_current = X_full[selected]
                X_with_const = sm.add_constant(X_current)
                try:
                    model = sm.OLS(y, X_with_const).fit()
                    for i, var in enumerate(selected):
                        pval = model.pvalues.iloc[i + 1]  # +1 for constant
                        if pval > worst_pval:
                            worst_pval = pval
                            worst_var = var
                except:
                    pass

                if worst_var and worst_pval > alpha_remove:
                    selected.remove(worst_var)
                    remaining.append(worst_var)
                    step_history.append(f"Step {step}: REMOVE {worst_var} (p={worst_pval:.4f})")
                    changed = True

            if not changed:
                break

            if step > 50:  # Safety limit
                break

        summary += f"<<COLOR:text>>Selection History:<</COLOR>>\n"
        for hist in step_history:
            summary += f"  {hist}\n"
        summary += "\n"

        # Final model
        if selected:
            X_final = X_full[selected]
            X_with_const = sm.add_constant(X_final)
            final_model = sm.OLS(y, X_with_const).fit()

            summary += f"<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            summary += f"<<COLOR:accent>>                              FINAL MODEL<</COLOR>>\n"
            summary += f"<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
            summary += f"<<COLOR:highlight>>Selected Variables:<</COLOR>> {', '.join(selected)}\n"
            summary += f"<<COLOR:highlight>>R²:<</COLOR>> {final_model.rsquared:.4f}\n"
            summary += f"<<COLOR:highlight>>Adjusted R²:<</COLOR>> {final_model.rsquared_adj:.4f}\n"
            summary += f"<<COLOR:highlight>>F-statistic:<</COLOR>> {final_model.fvalue:.4f} (p={final_model.f_pvalue:.4e})\n\n"

            summary += f"<<COLOR:text>>Coefficients:<</COLOR>>\n"
            summary += f"  {'Variable':<20} {'Coef':>12} {'Std Err':>12} {'t':>10} {'P>|t|':>10}\n"
            summary += f"  {'-'*66}\n"
            for i, name in enumerate(['const'] + selected):
                summary += f"  {name:<20} {final_model.params.iloc[i]:>12.4f} {final_model.bse.iloc[i]:>12.4f} {final_model.tvalues.iloc[i]:>10.3f} {final_model.pvalues.iloc[i]:>10.4f}\n"

            result["statistics"] = {
                "R²": float(final_model.rsquared),
                "Adj_R²": float(final_model.rsquared_adj),
                "n_selected": len(selected),
                "selected_vars": selected
            }

            # Coefficient plot
            result["plots"].append({
                "title": "Stepwise Selected Coefficients",
                "data": [{
                    "type": "bar",
                    "x": selected,
                    "y": [float(final_model.params.iloc[i+1]) for i in range(len(selected))],
                    "marker": {"color": "#4a9f6e"}
                }],
                "layout": {"height": 250, "yaxis": {"title": "Coefficient"}}
            })
        else:
            summary += f"<<COLOR:warning>>No variables met the selection criteria.<</COLOR>>\n"
            result["statistics"] = {"n_selected": 0, "selected_vars": []}

        result["summary"] = summary
        result["guide_observation"] = f"Stepwise regression selected {len(selected)} of {len(predictors)} predictors."

    elif analysis_id == "best_subsets":
        """
        Best Subsets Regression - evaluate all possible models.
        Compares models by R², Adjusted R², Cp, BIC.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        from itertools import combinations

        response = config.get("response")
        predictors = config.get("predictors", [])
        max_predictors = min(int(config.get("max_predictors", 8)), len(predictors), 8)

        X_full = df[predictors].dropna()
        y = df[response].loc[X_full.index]
        n = len(y)
        p_full = len(predictors)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BEST SUBSETS REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Candidate Predictors:<</COLOR>> {p_full}\n"
        summary += f"<<COLOR:highlight>>Max subset size:<</COLOR>> {max_predictors}\n\n"

        # Calculate full model MSE for Cp
        model_full = LinearRegression().fit(X_full, y)
        mse_full = np.mean((y - model_full.predict(X_full))**2)

        results_list = []

        # Evaluate all subsets up to max_predictors
        for k in range(1, max_predictors + 1):
            for combo in combinations(predictors, k):
                X_sub = X_full[list(combo)]
                model = LinearRegression().fit(X_sub, y)
                y_pred = model.predict(X_sub)

                sse = np.sum((y - y_pred)**2)
                r2 = 1 - sse / np.sum((y - y.mean())**2)
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

                # Mallows' Cp
                cp = sse / mse_full - n + 2 * (k + 1)

                # BIC
                bic = n * np.log(sse / n) + k * np.log(n)

                results_list.append({
                    "vars": combo,
                    "k": k,
                    "r2": r2,
                    "adj_r2": adj_r2,
                    "cp": cp,
                    "bic": bic
                })

        # Sort by different criteria
        by_r2 = sorted(results_list, key=lambda x: -x["r2"])
        by_adj_r2 = sorted(results_list, key=lambda x: -x["adj_r2"])
        by_cp = sorted(results_list, key=lambda x: x["cp"])
        by_bic = sorted(results_list, key=lambda x: x["bic"])

        summary += f"<<COLOR:text>>Best Models by Criterion:<</COLOR>>\n\n"

        summary += f"<<COLOR:accent>>By Adjusted R² (top 5):<</COLOR>>\n"
        summary += f"  {'Vars':<8} {'R²':>8} {'Adj R²':>8} {'Cp':>10} {'BIC':>10} {'Predictors'}\n"
        summary += f"  {'-'*70}\n"
        for res in by_adj_r2[:5]:
            vars_str = ', '.join(res['vars'][:3]) + ('...' if len(res['vars']) > 3 else '')
            summary += f"  {res['k']:<8} {res['r2']:>8.4f} {res['adj_r2']:>8.4f} {res['cp']:>10.2f} {res['bic']:>10.2f} {vars_str}\n"

        summary += f"\n<<COLOR:accent>>By Mallows' Cp (top 5):<</COLOR>>\n"
        summary += f"  {'Vars':<8} {'R²':>8} {'Adj R²':>8} {'Cp':>10} {'BIC':>10}\n"
        summary += f"  {'-'*50}\n"
        for res in by_cp[:5]:
            summary += f"  {res['k']:<8} {res['r2']:>8.4f} {res['adj_r2']:>8.4f} {res['cp']:>10.2f} {res['bic']:>10.2f}\n"

        # Best model recommendation
        best = by_adj_r2[0]
        summary += f"\n<<COLOR:success>>RECOMMENDED MODEL:<</COLOR>>\n"
        summary += f"  Variables: {', '.join(best['vars'])}\n"
        summary += f"  R² = {best['r2']:.4f}, Adj R² = {best['adj_r2']:.4f}\n"

        result["summary"] = summary
        result["guide_observation"] = f"Best subsets: recommended {best['k']}-variable model with Adj R² = {best['adj_r2']:.4f}"
        result["statistics"] = {
            "best_r2": float(best["r2"]),
            "best_adj_r2": float(best["adj_r2"]),
            "best_vars": list(best["vars"]),
            "models_evaluated": len(results_list)
        }

        # Plot: R² and Adj R² by number of variables
        k_values = sorted(set(r["k"] for r in results_list))
        best_r2_by_k = [max(r["r2"] for r in results_list if r["k"] == k) for k in k_values]
        best_adj_r2_by_k = [max(r["adj_r2"] for r in results_list if r["k"] == k) for k in k_values]

        result["plots"].append({
            "title": "Best Subsets: R² by Model Size",
            "data": [
                {"type": "scatter", "x": k_values, "y": best_r2_by_k, "mode": "lines+markers", "name": "R²", "line": {"color": "#4a9f6e"}},
                {"type": "scatter", "x": k_values, "y": best_adj_r2_by_k, "mode": "lines+markers", "name": "Adj R²", "line": {"color": "#47a5e8"}}
            ],
            "layout": {"height": 250, "xaxis": {"title": "Number of Predictors"}, "yaxis": {"title": "R²"}}
        })

    elif analysis_id == "bootstrap_ci":
        """
        Bootstrap Confidence Intervals - non-parametric inference.
        Resampling-based confidence intervals for statistics.
        """
        var = config.get("var")
        statistic = config.get("statistic", "mean")  # mean, median, std, correlation
        var2 = config.get("var2")  # For correlation
        n_bootstrap = int(config.get("n_bootstrap", 1000))
        conf_level = float(config.get("conf", 95)) / 100

        data = df[var].dropna().values
        n = len(data)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BOOTSTRAP CONFIDENCE INTERVALS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Statistic:<</COLOR>> {statistic}\n"
        summary += f"<<COLOR:highlight>>Bootstrap samples:<</COLOR>> {n_bootstrap}\n"
        summary += f"<<COLOR:highlight>>Confidence level:<</COLOR>> {conf_level*100:.0f}%\n\n"

        np.random.seed(42)

        # Calculate bootstrap distribution
        boot_stats = []

        if statistic == "correlation" and var2:
            data2 = df[var2].dropna().values
            min_len = min(len(data), len(data2))
            data = data[:min_len]
            data2 = data2[:min_len]
            observed = np.corrcoef(data, data2)[0, 1]

            for _ in range(n_bootstrap):
                idx = np.random.choice(min_len, min_len, replace=True)
                boot_stats.append(np.corrcoef(data[idx], data2[idx])[0, 1])
        else:
            if statistic == "mean":
                observed = np.mean(data)
                stat_func = np.mean
            elif statistic == "median":
                observed = np.median(data)
                stat_func = np.median
            elif statistic == "std":
                observed = np.std(data, ddof=1)
                stat_func = lambda x: np.std(x, ddof=1)
            elif statistic == "trimmed_mean":
                from scipy.stats import trim_mean
                observed = trim_mean(data, 0.1)
                stat_func = lambda x: trim_mean(x, 0.1)
            else:
                observed = np.mean(data)
                stat_func = np.mean

            for _ in range(n_bootstrap):
                boot_sample = np.random.choice(data, n, replace=True)
                boot_stats.append(stat_func(boot_sample))

        boot_stats = np.array(boot_stats)

        # Calculate CI using percentile method
        alpha = 1 - conf_level
        ci_lower = np.percentile(boot_stats, alpha/2 * 100)
        ci_upper = np.percentile(boot_stats, (1 - alpha/2) * 100)

        # BCa (Bias-Corrected and Accelerated) interval
        # Bias correction
        z0 = stats.norm.ppf(np.mean(boot_stats < observed))

        # Acceleration (jackknife)
        jackknife_stats = []
        for i in range(n):
            jack_sample = np.delete(data, i)
            if statistic == "mean":
                jackknife_stats.append(np.mean(jack_sample))
            elif statistic == "median":
                jackknife_stats.append(np.median(jack_sample))
            else:
                jackknife_stats.append(np.mean(jack_sample))

        jackknife_stats = np.array(jackknife_stats)
        jack_mean = np.mean(jackknife_stats)
        a = np.sum((jack_mean - jackknife_stats)**3) / (6 * (np.sum((jack_mean - jackknife_stats)**2))**1.5 + 1e-10)

        # BCa quantiles
        z_alpha_low = stats.norm.ppf(alpha/2)
        z_alpha_high = stats.norm.ppf(1 - alpha/2)

        bca_low_q = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low)))
        bca_high_q = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high)))

        bca_lower = np.percentile(boot_stats, bca_low_q * 100)
        bca_upper = np.percentile(boot_stats, bca_high_q * 100)

        summary += f"<<COLOR:text>>Sample Statistics:<</COLOR>>\n"
        summary += f"  Observed {statistic}: {observed:.4f}\n"
        summary += f"  Bootstrap SE: {np.std(boot_stats):.4f}\n"
        summary += f"  Bootstrap Bias: {np.mean(boot_stats) - observed:.4f}\n\n"

        summary += f"<<COLOR:text>>Confidence Intervals:<</COLOR>>\n"
        summary += f"  Percentile: ({ci_lower:.4f}, {ci_upper:.4f})\n"
        summary += f"  BCa:        ({bca_lower:.4f}, {bca_upper:.4f})\n\n"

        summary += f"<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  We are {conf_level*100:.0f}% confident the true {statistic}\n"
        summary += f"  lies between {bca_lower:.4f} and {bca_upper:.4f}.\n"

        result["summary"] = summary
        result["guide_observation"] = f"Bootstrap {conf_level*100:.0f}% CI for {statistic}: ({bca_lower:.4f}, {bca_upper:.4f})"
        result["statistics"] = {
            f"observed_{statistic}": float(observed),
            "ci_lower": float(bca_lower),
            "ci_upper": float(bca_upper),
            "bootstrap_se": float(np.std(boot_stats))
        }

        # Histogram of bootstrap distribution
        result["plots"].append({
            "title": f"Bootstrap Distribution of {statistic.title()}",
            "data": [
                {"type": "histogram", "x": boot_stats.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}},
                {"type": "scatter", "x": [observed, observed], "y": [0, n_bootstrap/20], "mode": "lines", "line": {"color": "#e89547", "width": 2, "dash": "dash"}, "name": "Observed"},
                {"type": "scatter", "x": [bca_lower, bca_upper], "y": [0, 0], "mode": "markers", "marker": {"color": "#e85747", "size": 12, "symbol": "triangle-up"}, "name": "CI bounds"}
            ],
            "layout": {"height": 250, "xaxis": {"title": statistic.title()}, "yaxis": {"title": "Frequency"}}
        })

    elif analysis_id == "box_cox":
        """
        Box-Cox Transformation - find optimal power transformation.
        Transforms data to approximate normality.
        """
        var = config.get("var")

        data = df[var].dropna().values

        # Box-Cox requires positive data
        if np.any(data <= 0):
            # Shift data to be positive
            shift = -np.min(data) + 1
            data_shifted = data + shift
            shifted = True
        else:
            data_shifted = data
            shift = 0
            shifted = False

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>BOX-COX TRANSFORMATION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {len(data)}\n"
        if shifted:
            summary += f"<<COLOR:warning>>Data shifted by {shift:.4f} (original had non-positive values)<</COLOR>>\n"
        summary += "\n"

        # Find optimal lambda
        transformed, optimal_lambda = stats.boxcox(data_shifted)

        # Test common transformations
        lambdas = [-2, -1, -0.5, 0, 0.5, 1, 2]
        lambda_names = ["1/x²", "1/x", "1/√x", "ln(x)", "√x", "x (none)", "x²"]

        summary += f"<<COLOR:text>>Common Transformations (Log-Likelihood):<</COLOR>>\n"
        for lam, name in zip(lambdas, lambda_names):
            if lam == 0:
                trans = np.log(data_shifted)
            else:
                trans = (data_shifted**lam - 1) / lam
            # Calculate log-likelihood
            ll = -len(data)/2 * np.log(np.var(trans)) + (lam - 1) * np.sum(np.log(data_shifted))
            summary += f"  λ = {lam:>5} ({name:<8}): LL = {ll:.2f}\n"

        summary += f"\n<<COLOR:success>>OPTIMAL TRANSFORMATION:<</COLOR>>\n"
        summary += f"  λ = {optimal_lambda:.4f}\n"

        # Interpret lambda
        if abs(optimal_lambda) < 0.1:
            suggestion = "ln(x) - logarithmic"
        elif abs(optimal_lambda - 0.5) < 0.1:
            suggestion = "√x - square root"
        elif abs(optimal_lambda - 1) < 0.1:
            suggestion = "x - no transformation needed"
        elif abs(optimal_lambda + 1) < 0.1:
            suggestion = "1/x - reciprocal"
        elif optimal_lambda < 0:
            suggestion = f"x^{optimal_lambda:.2f} - inverse power"
        else:
            suggestion = f"x^{optimal_lambda:.2f} - power transformation"

        summary += f"  Suggested: {suggestion}\n\n"

        # Normality tests before and after
        _, p_before = stats.shapiro(data[:min(5000, len(data))])
        _, p_after = stats.shapiro(transformed[:min(5000, len(transformed))])

        summary += f"<<COLOR:text>>Normality Tests (Shapiro-Wilk):<</COLOR>>\n"
        summary += f"  Original: p = {p_before:.4f} {'(normal)' if p_before > 0.05 else '(non-normal)'}\n"
        summary += f"  Transformed: p = {p_after:.4f} {'(normal)' if p_after > 0.05 else '(non-normal)'}\n"

        result["summary"] = summary
        result["guide_observation"] = f"Box-Cox optimal λ = {optimal_lambda:.3f}. {suggestion}."
        result["statistics"] = {
            "optimal_lambda": float(optimal_lambda),
            "p_before": float(p_before),
            "p_after": float(p_after),
            "shift_applied": float(shift)
        }

        # Plot: original vs transformed distributions
        result["plots"].append({
            "title": "Original Distribution",
            "data": [{"type": "histogram", "x": data.tolist(), "marker": {"color": "rgba(232, 87, 71, 0.4)", "line": {"color": "#e85747", "width": 1}}}],
            "layout": {"height": 200, "xaxis": {"title": var}}
        })

        result["plots"].append({
            "title": f"Transformed (λ = {optimal_lambda:.2f})",
            "data": [{"type": "histogram", "x": transformed.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}}],
            "layout": {"height": 200, "xaxis": {"title": f"Box-Cox({var})"}}
        })

    elif analysis_id == "robust_regression":
        """
        Robust Regression - outlier-resistant regression.
        Uses Huber or MM estimators.
        """
        import statsmodels.api as sm
        from statsmodels.robust.robust_linear_model import RLM

        response = config.get("response")
        predictors = config.get("predictors", [])
        method = config.get("method", "huber")  # huber, bisquare, andrews

        X = df[predictors].dropna()
        y = df[response].loc[X.index]
        n = len(y)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ROBUST REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Response:<</COLOR>> {response}\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        summary += f"<<COLOR:highlight>>Method:<</COLOR>> {method.title()} M-estimator\n"
        summary += f"<<COLOR:highlight>>Observations:<</COLOR>> {n}\n\n"

        # Select M-estimator
        if method == "huber":
            M_estimator = sm.robust.norms.HuberT()
            summary += f"<<COLOR:dim>>Huber's T: downweights residuals > 1.345σ<</COLOR>>\n"
        elif method == "bisquare":
            M_estimator = sm.robust.norms.TukeyBiweight()
            summary += f"<<COLOR:dim>>Tukey's Bisquare: zero weight for residuals > 4.685σ<</COLOR>>\n"
        elif method == "andrews":
            M_estimator = sm.robust.norms.AndrewWave()
            summary += f"<<COLOR:dim>>Andrew's Wave: sinusoidal downweighting<</COLOR>>\n"
        else:
            M_estimator = sm.robust.norms.HuberT()

        X_const = sm.add_constant(X)

        # Fit OLS for comparison
        ols_model = sm.OLS(y, X_const).fit()

        # Fit robust model
        robust_model = RLM(y, X_const, M=M_estimator).fit()

        summary += f"\n<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += f"<<COLOR:accent>>                        COMPARISON: OLS vs ROBUST<</COLOR>>\n"
        summary += f"<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += f"  {'Variable':<20} {'OLS Coef':>12} {'Robust Coef':>12} {'Difference':>12}\n"
        summary += f"  {'-'*58}\n"

        var_names = ['const'] + predictors
        for i, name in enumerate(var_names):
            ols_coef = ols_model.params.iloc[i]
            rob_coef = robust_model.params.iloc[i]
            diff = rob_coef - ols_coef
            diff_pct = 100 * diff / abs(ols_coef) if abs(ols_coef) > 1e-10 else 0
            flag = "<<COLOR:warning>>*<</COLOR>>" if abs(diff_pct) > 10 else ""
            summary += f"  {name:<20} {ols_coef:>12.4f} {rob_coef:>12.4f} {diff:>+12.4f} {flag}\n"

        summary += f"\n<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += f"<<COLOR:accent>>                           ROBUST MODEL DETAILS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>────────────────────────────────────────────────────────────────────────────<</COLOR>>\n"
        summary += f"  {'Variable':<20} {'Coef':>12} {'Std Err':>12} {'z':>10} {'P>|z|':>10}\n"
        summary += f"  {'-'*66}\n"

        for i, name in enumerate(var_names):
            summary += f"  {name:<20} {robust_model.params.iloc[i]:>12.4f} {robust_model.bse.iloc[i]:>12.4f} {robust_model.tvalues.iloc[i]:>10.3f} {robust_model.pvalues.iloc[i]:>10.4f}\n"

        # Identify outliers (low weights)
        weights = robust_model.weights
        outlier_threshold = 0.5
        outliers = np.where(weights < outlier_threshold)[0]

        summary += f"\n<<COLOR:text>>Observations with low weight (<{outlier_threshold}):<</COLOR>> {len(outliers)}\n"
        if len(outliers) > 0 and len(outliers) <= 10:
            summary += f"  Indices: {list(outliers)}\n"

        summary += f"\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        coef_diffs = np.abs(robust_model.params.values - ols_model.params.values)
        if np.max(coef_diffs[1:]) > 0.1 * np.max(np.abs(ols_model.params.values[1:])):
            summary += f"  <<COLOR:warning>>Coefficients differ substantially - outliers are influential.<</COLOR>>\n"
            summary += f"  Robust estimates may be more reliable.\n"
        else:
            summary += f"  <<COLOR:good>>Coefficients are similar - outliers have minimal influence.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"Robust regression ({method}) identified {len(outliers)} low-weight observations."
        result["statistics"] = {
            "n_outliers": len(outliers),
            "method": method,
            **{f"coef_{name}": float(robust_model.params.iloc[i]) for i, name in enumerate(var_names)}
        }

        # Plot: OLS residuals vs Robust residuals
        ols_resid = ols_model.resid
        rob_resid = robust_model.resid

        result["plots"].append({
            "title": "Residual Comparison",
            "data": [
                {"type": "scatter", "x": ols_resid.tolist(), "y": rob_resid.tolist(), "mode": "markers",
                 "marker": {"color": weights.tolist(), "colorscale": [[0, "#e85747"], [1, "#4a9f6e"]], "size": 6, "colorbar": {"title": "Weight"}}}
            ],
            "layout": {"height": 300, "xaxis": {"title": "OLS Residuals"}, "yaxis": {"title": "Robust Residuals"}}
        })

        # Weights histogram
        result["plots"].append({
            "title": "Observation Weights",
            "data": [{"type": "histogram", "x": weights.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1}}}],
            "layout": {"height": 200, "xaxis": {"title": "Weight", "range": [0, 1.1]}}
        })

    elif analysis_id == "tolerance_interval":
        """
        Tolerance Intervals - contain a proportion of the population.
        Both normal-based and non-parametric methods.
        """
        var = config.get("var")
        proportion = float(config.get("proportion", 0.95))  # Proportion of population
        confidence = float(config.get("confidence", 0.95))  # Confidence level
        method = config.get("method", "normal")  # normal or nonparametric

        data = df[var].dropna().values
        n = len(data)

        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>TOLERANCE INTERVALS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Variable:<</COLOR>> {var}\n"
        summary += f"<<COLOR:highlight>>Sample size:<</COLOR>> {n}\n"
        summary += f"<<COLOR:highlight>>Coverage:<</COLOR>> {proportion*100:.0f}% of population\n"
        summary += f"<<COLOR:highlight>>Confidence:<</COLOR>> {confidence*100:.0f}%\n\n"

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        summary += f"<<COLOR:text>>Sample Statistics:<</COLOR>>\n"
        summary += f"  Mean: {mean:.4f}\n"
        summary += f"  Std Dev: {std:.4f}\n\n"

        # Normal-based tolerance interval
        # k factor from tolerance interval tables (approximation)
        z_p = stats.norm.ppf((1 + proportion) / 2)
        chi2_val = stats.chi2.ppf(1 - confidence, n - 1)

        # Two-sided tolerance factor
        k_normal = z_p * np.sqrt((n - 1) * (1 + 1/n) / chi2_val)

        tol_lower_normal = mean - k_normal * std
        tol_upper_normal = mean + k_normal * std

        summary += f"<<COLOR:accent>>Normal-Based Tolerance Interval:<</COLOR>>\n"
        summary += f"  k factor: {k_normal:.4f}\n"
        summary += f"  Interval: ({tol_lower_normal:.4f}, {tol_upper_normal:.4f})\n\n"

        # Non-parametric tolerance interval
        # Uses order statistics
        # For 95/95, need approximately n >= 59 for two-sided
        # Coverage probability for (X(r), X(n-r+1)) where r is chosen appropriately

        # Simple approach: use percentiles
        alpha = 1 - confidence
        beta = 1 - proportion

        # Find r such that P(at least proportion*100% between X(r) and X(n-r+1)) >= confidence
        # Using binomial distribution
        from scipy.special import comb

        r_found = None
        for r in range(1, n//2 + 1):
            # Probability that at least proportion of population is between order statistics
            prob = 0
            for j in range(r, n - r + 2):
                prob += comb(n, j, exact=True) * (proportion**(j)) * ((1-proportion)**(n-j))
            if prob >= confidence:
                r_found = r
                break

        if r_found:
            sorted_data = np.sort(data)
            tol_lower_np = sorted_data[r_found - 1]
            tol_upper_np = sorted_data[n - r_found]
            summary += f"<<COLOR:accent>>Non-Parametric Tolerance Interval:<</COLOR>>\n"
            summary += f"  Uses order statistics X({r_found}) and X({n - r_found + 1})\n"
            summary += f"  Interval: ({tol_lower_np:.4f}, {tol_upper_np:.4f})\n\n"
        else:
            tol_lower_np = np.min(data)
            tol_upper_np = np.max(data)
            summary += f"<<COLOR:warning>>Non-Parametric: Sample too small for exact interval.<</COLOR>>\n"
            summary += f"  Using min/max: ({tol_lower_np:.4f}, {tol_upper_np:.4f})\n\n"

        # Comparison with confidence interval
        se = std / np.sqrt(n)
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        ci_lower = mean - t_val * se
        ci_upper = mean + t_val * se

        summary += f"<<COLOR:dim>>For comparison - {confidence*100:.0f}% CI for mean:<</COLOR>>\n"
        summary += f"  ({ci_lower:.4f}, {ci_upper:.4f})\n\n"

        summary += f"<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  We are {confidence*100:.0f}% confident that at least {proportion*100:.0f}%\n"
        summary += f"  of the population falls within the tolerance interval.\n"
        summary += f"\n<<COLOR:dim>>Note: Tolerance intervals are WIDER than confidence intervals<</COLOR>>\n"
        summary += f"<<COLOR:dim>>because they cover the population, not just the mean.<</COLOR>>\n"

        result["summary"] = summary
        result["guide_observation"] = f"Tolerance interval ({proportion*100:.0f}%/{confidence*100:.0f}%): ({tol_lower_normal:.4f}, {tol_upper_normal:.4f})"
        result["statistics"] = {
            "tol_lower_normal": float(tol_lower_normal),
            "tol_upper_normal": float(tol_upper_normal),
            "tol_lower_np": float(tol_lower_np),
            "tol_upper_np": float(tol_upper_np),
            "k_factor": float(k_normal),
            "mean": float(mean),
            "std": float(std)
        }

        # Plot showing intervals
        result["plots"].append({
            "title": "Tolerance vs Confidence Intervals",
            "data": [
                {"type": "histogram", "x": data.tolist(), "marker": {"color": "rgba(74, 159, 110, 0.3)", "line": {"color": "#4a9f6e", "width": 1}}, "name": "Data"},
            ],
            "layout": {
                "height": 300,
                "xaxis": {"title": var},
                "shapes": [
                    # Tolerance interval (normal)
                    {"type": "rect", "x0": tol_lower_normal, "x1": tol_upper_normal, "y0": 0, "y1": 1, "yref": "paper",
                     "fillcolor": "rgba(232, 149, 71, 0.2)", "line": {"color": "#e89547", "width": 2}},
                    # Confidence interval
                    {"type": "rect", "x0": ci_lower, "x1": ci_upper, "y0": 0.4, "y1": 0.6, "yref": "paper",
                     "fillcolor": "rgba(71, 165, 232, 0.4)", "line": {"color": "#47a5e8", "width": 2}},
                    # Mean line
                    {"type": "line", "x0": mean, "x1": mean, "y0": 0, "y1": 1, "yref": "paper",
                     "line": {"color": "#4a9f6e", "width": 2, "dash": "dash"}}
                ],
                "annotations": [
                    {"x": (tol_lower_normal + tol_upper_normal)/2, "y": 0.95, "yref": "paper", "text": "Tolerance", "showarrow": False, "font": {"color": "#e89547"}},
                    {"x": (ci_lower + ci_upper)/2, "y": 0.5, "yref": "paper", "text": "CI", "showarrow": False, "font": {"color": "#47a5e8"}}
                ]
            }
        })

    return result


def run_ml_analysis(df, analysis_id, config, user):
    """Run ML analysis."""
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error

    result = {"plots": [], "summary": "", "guide_observation": ""}

    target = config.get("target")
    features = config.get("features", [])
    algorithm = config.get("algorithm", "rf")
    split_val = float(config.get("split", 20))
    test_split = split_val if split_val < 1 else split_val / 100

    # Analyses that don't require a target variable
    unsupervised_analyses = ["pca", "clustering", "isolation_forest"]

    if analysis_id in unsupervised_analyses:
        # For unsupervised methods, only need features
        if not features:
            # Use all numeric columns if no features specified
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[features].dropna()
        y = None
        X_train = X_test = X
        y_train = y_test = None
    else:
        # Supervised methods need target and features
        if not target:
            result["summary"] = "Error: Please select a target variable."
            return result
        if not features:
            result["summary"] = "Error: Please select at least one feature."
            return result

        X = df[features].dropna()
        y = df[target].loc[X.index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)

    if analysis_id == "classification":
        if algorithm == "rf":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        result["summary"] = f"Classification Results\n\nAlgorithm: {algorithm}\nAccuracy: {accuracy:.4f}\n\n{report}"

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            result["plots"].append({
                "title": "Feature Importance",
                "data": [{
                    "type": "bar",
                    "x": importances.tolist(),
                    "y": features,
                    "orientation": "h",
                    "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}
                }],
                "layout": {"template": "plotly_dark", "height": 300}
            })

        result["guide_observation"] = f"Classification model achieved {accuracy:.1%} accuracy."

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(user.id, model_key, model, {
                'model_type': f"Classification ({algorithm.upper()})",
                'features': features,
                'target': target,
                'metrics': {'accuracy': float(accuracy)}
            })
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "regression_ml":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        result["summary"] = f"Regression Results\n\nR²: {r2:.4f}\nRMSE: {rmse:.4f}"

        # Actual vs Predicted
        result["plots"].append({
            "title": "Actual vs Predicted",
            "data": [
                {"type": "scatter", "x": y_test.tolist(), "y": y_pred.tolist(), "mode": "markers", "marker": {"color": "#6c5ce7", "size": 5}},
                {"type": "scatter", "x": [y_test.min(), y_test.max()], "y": [y_test.min(), y_test.max()], "mode": "lines", "line": {"color": "#ff7675", "dash": "dash"}}
            ],
            "layout": {"template": "plotly_dark", "height": 300, "xaxis": {"title": "Actual"}, "yaxis": {"title": "Predicted"}}
        })

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(user.id, model_key, model, {
                'model_type': "Random Forest Regressor",
                'features': features,
                'target': target,
                'metrics': {'r2': float(r2), 'rmse': float(rmse)}
            })
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "clustering":
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_clusters = int(config.get("n_clusters", 3))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        result["summary"] = f"K-Means Clustering\n\nClusters: {n_clusters}\nInertia: {kmeans.inertia_:.2f}"

        # Scatter plot of first two features colored by cluster
        if len(features) >= 2:
            result["plots"].append({
                "title": f"Clusters ({features[0]} vs {features[1]})",
                "data": [{
                    "type": "scatter",
                    "x": X[features[0]].tolist(),
                    "y": X[features[1]].tolist(),
                    "mode": "markers",
                    "marker": {"color": clusters.tolist(), "colorscale": "Viridis", "size": 6}
                }],
                "layout": {"template": "plotly_dark", "height": 300}
            })

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
        summary += f"<<COLOR:title>>PRINCIPAL COMPONENT ANALYSIS<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n"
        summary += f"<<COLOR:highlight>>Components:<</COLOR>> {pca.n_components_}\n\n"

        summary += f"<<COLOR:text>>Explained Variance:<</COLOR>>\n"
        cumulative = 0
        for i, (var, ratio) in enumerate(zip(pca.explained_variance_, pca.explained_variance_ratio_)):
            cumulative += ratio * 100
            summary += f"  PC{i+1}: {ratio*100:.1f}% (cumulative: {cumulative:.1f}%)\n"

        summary += f"\n<<COLOR:text>>Loadings (feature weights):<</COLOR>>\n"
        for i, component in enumerate(pca.components_[:3]):  # Show first 3 PCs max
            summary += f"\n  PC{i+1}:\n"
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
                "name": "Observations"
            }
            if color_values:
                scatter_data["text"] = color_values
                scatter_data["marker"]["color"] = [hash(v) % 10 for v in color_values]
                scatter_data["marker"]["colorscale"] = "Viridis"

            result["plots"].append({
                "title": "PCA Biplot",
                "data": [scatter_data],
                "layout": {
                    "template": "plotly_dark",
                    "height": 400,
                    "xaxis": {"title": f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"},
                    "yaxis": {"title": f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"}
                }
            })

        # Scree plot
        result["plots"].append({
            "title": "Scree Plot",
            "data": [{
                "type": "bar",
                "x": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
                "y": (pca.explained_variance_ratio_ * 100).tolist(),
                "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}
            }],
            "layout": {
                "template": "plotly_dark",
                "height": 250,
                "yaxis": {"title": "Variance Explained (%)"}
            }
        })

    elif analysis_id == "feature":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder

        # Determine if classification or regression
        y_unique = y.nunique()
        is_classification = y.dtype == 'object' or y_unique < 10

        if is_classification:
            if y.dtype == 'object':
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
        summary += f"<<COLOR:title>>FEATURE IMPORTANCE (Random Forest)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Task:<</COLOR>> {'Classification' if is_classification else 'Regression'}\n\n"

        summary += f"<<COLOR:text>>Feature Rankings:<</COLOR>>\n"
        for rank, idx in enumerate(indices, 1):
            bar_len = int(importances[idx] * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            summary += f"  {rank}. {features[idx]:<20} {bar} {importances[idx]:.3f}\n"

        result["summary"] = summary

        # Horizontal bar chart
        sorted_features = [features[i] for i in indices]
        sorted_importances = [importances[i] for i in indices]

        result["plots"].append({
            "title": "Feature Importance",
            "data": [{
                "type": "bar",
                "x": sorted_importances[::-1],
                "y": sorted_features[::-1],
                "orientation": "h",
                "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}
            }],
            "layout": {
                "template": "plotly_dark",
                "height": max(250, len(features) * 25),
                "xaxis": {"title": "Importance"},
                "margin": {"l": 150}
            }
        })

    # =========================================================================
    # PHASE 1: CAUSAL LENS TOOLKIT
    # =========================================================================

    elif analysis_id == "bayesian_regression":
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
        summary += f"<<COLOR:title>>BAYESIAN LINEAR REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n\n"

        # Model fit
        r2 = model.score(X_scaled, y)
        summary += f"<<COLOR:text>>Model Fit:<</COLOR>>\n"
        summary += f"  R² = {r2:.4f}\n"
        summary += f"  α (noise precision) = {model.alpha_:.4f}\n"
        summary += f"  λ (weight precision) = {model.lambda_:.4f}\n\n"

        # Coefficient posteriors with credible intervals
        summary += f"<<COLOR:text>>Coefficient Posteriors (95% Credible Intervals):<</COLOR>>\n"
        summary += f"<<COLOR:text>>These intervals feed directly into Synara edge weights<</COLOR>>\n\n"

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
        summary += f"\n<<COLOR:success>>SYNARA INTEGRATION:<</COLOR>>\n"
        summary += f"  Coefficients with CIs not crossing zero indicate strong causal evidence.\n"
        summary += f"  CI width indicates uncertainty in edge weight.\n"

        result["summary"] = summary

        # Plot 1: Coefficient posteriors with credible intervals
        result["plots"].append({
            "title": "Coefficient Posteriors (95% CI)",
            "data": [{
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
                    "width": 6
                },
                "name": "Coefficient ± 95% CI"
            }, {
                "type": "scatter",
                "x": [0],
                "y": [features[len(features)//2]],
                "mode": "lines",
                "line": {"color": "#e85747", "dash": "dash", "width": 1},
                "showlegend": False
            }],
            "layout": {
                "height": max(300, len(features) * 30),
                "xaxis": {"title": "Coefficient Value", "zeroline": True, "zerolinecolor": "#e85747"},
                "margin": {"l": 150},
                "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": -0.5, "y1": len(features)-0.5, "line": {"color": "#e85747", "dash": "dash"}}]
            }
        })

        # Plot 2: Predictions with uncertainty bands
        sorted_idx = np.argsort(y.values)
        result["plots"].append({
            "title": "Predictions with Uncertainty",
            "data": [{
                "type": "scatter",
                "x": list(range(len(y))),
                "y": y.values[sorted_idx].tolist(),
                "mode": "markers",
                "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 5, "line": {"color": "#4a9f6e", "width": 1}},
                "name": "Actual"
            }, {
                "type": "scatter",
                "x": list(range(len(y))),
                "y": y_pred[sorted_idx].tolist(),
                "mode": "lines",
                "line": {"color": "#e89547"},
                "name": "Predicted"
            }, {
                "type": "scatter",
                "x": list(range(len(y))) + list(range(len(y)))[::-1],
                "y": (y_pred[sorted_idx] + 1.96*y_std[sorted_idx]).tolist() + (y_pred[sorted_idx] - 1.96*y_std[sorted_idx])[::-1].tolist(),
                "fill": "toself",
                "fillcolor": "rgba(232, 149, 71, 0.2)",
                "line": {"color": "transparent"},
                "name": "95% CI"
            }],
            "layout": {"height": 300, "xaxis": {"title": "Observation (sorted)"}, "yaxis": {"title": target}}
        })

        result["guide_observation"] = f"Bayesian regression R²={r2:.3f}. Coefficients with CIs not crossing zero are significant causal candidates."

        # Include coefficient data for Synara integration
        result["synara_weights"] = {
            "analysis_type": "bayesian_regression",
            "target": target,
            "coefficients": [
                {
                    "feature": feat,
                    "mean": float(coef_mean[i]),
                    "ci_low": float(coef_mean[i] - 1.96 * coef_std[i]),
                    "ci_high": float(coef_mean[i] + 1.96 * coef_std[i])
                }
                for i, feat in enumerate(features)
            ]
        }

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(user.id, model_key, {'model': model, 'scaler': scaler}, {
                'model_type': "Bayesian Ridge Regression",
                'features': features,
                'target': target,
                'metrics': {'r2': float(r2)}
            })
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
        summary += f"<<COLOR:title>>GENERALIZED ADDITIVE MODEL (GAM)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n"
        summary += f"<<COLOR:highlight>>Model:<</COLOR>> y = Σ f_i(x_i) + ε  (smooth splines)\n\n"

        # Model statistics
        summary += f"<<COLOR:text>>Model Statistics:<</COLOR>>\n"
        summary += f"  Pseudo R² = {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}\n"
        summary += f"  GCV Score = {gam.statistics_['GCV']:.4f}\n"
        summary += f"  Effective DF = {gam.statistics_['edof']:.1f}\n\n"

        # Feature significance (approximate p-values)
        summary += f"<<COLOR:text>>Feature Effects (Spline Significance):<</COLOR>>\n"
        summary += f"<<COLOR:text>>Each curve shows HOW the feature affects {target}<</COLOR>>\n\n"

        p_values = gam.statistics_['p_values']
        for i, feat in enumerate(features):
            p = p_values[i] if i < len(p_values) else 1.0
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            summary += f"  {feat:<20} p = {p:.4f} {sig}\n"

        summary += f"\n<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  The partial dependence plots show the SHAPE of each effect.\n"
        summary += f"  Non-linear curves reveal complex causal relationships.\n"

        result["summary"] = summary

        # Generate partial dependence plots for each feature
        for i, feat in enumerate(features[:4]):  # Limit to first 4 features
            # Generate grid for this feature
            XX = gam.generate_X_grid(term=i, n=100)
            pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

            # Convert back to original scale for x-axis
            x_grid = XX[:, i] * scaler.scale_[i] + scaler.mean_[i]

            result["plots"].append({
                "title": f"Effect of {feat}",
                "data": [{
                    "type": "scatter",
                    "x": x_grid.tolist(),
                    "y": pdep.tolist(),
                    "mode": "lines",
                    "line": {"color": "#4a9f6e", "width": 2},
                    "name": "Effect"
                }, {
                    "type": "scatter",
                    "x": x_grid.tolist() + x_grid[::-1].tolist(),
                    "y": confi[:, 0].tolist() + confi[::-1, 1].tolist(),
                    "fill": "toself",
                    "fillcolor": "rgba(74, 159, 110, 0.2)",
                    "line": {"color": "transparent"},
                    "name": "95% CI"
                }],
                "layout": {
                    "height": 250,
                    "xaxis": {"title": feat},
                    "yaxis": {"title": f"Effect on {target}"},
                    "showlegend": False
                }
            })

        result["guide_observation"] = f"GAM shows smooth effect curves for each feature. Non-linear patterns indicate complex causal relationships."

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(user.id, model_key, {'gam': gam, 'scaler': scaler}, {
                'model_type': "Generalized Additive Model",
                'features': features,
                'target': target,
                'metrics': {'pseudo_r2': float(gam.statistics_['pseudo_r2']['explained_deviance'])}
            })
            result["model_key"] = model_key
            result["can_save"] = True

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
        iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        predictions = iso.fit_predict(X_scaled)
        scores = iso.decision_function(X_scaled)

        # Identify anomalies
        anomalies = predictions == -1
        n_anomalies = anomalies.sum()
        anomaly_pct = n_anomalies / len(predictions) * 100

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>ISOLATION FOREST (Anomaly Detection)<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n"
        summary += f"<<COLOR:highlight>>Contamination:<</COLOR>> {contamination:.1%}\n\n"

        summary += f"<<COLOR:text>>Results:<</COLOR>>\n"
        summary += f"  Total observations: {len(predictions)}\n"
        summary += f"  Anomalies detected: {n_anomalies} ({anomaly_pct:.1f}%)\n"
        summary += f"  Normal observations: {len(predictions) - n_anomalies}\n\n"

        # Anomaly score statistics
        summary += f"<<COLOR:text>>Anomaly Scores:<</COLOR>>\n"
        summary += f"  Mean score: {scores.mean():.4f}\n"
        summary += f"  Std score: {scores.std():.4f}\n"
        summary += f"  Threshold: ~0.0 (negative = anomaly)\n\n"

        # Show most anomalous observations
        if n_anomalies > 0:
            summary += f"<<COLOR:warning>>MOST ANOMALOUS OBSERVATIONS:<</COLOR>>\n"
            anomaly_idx = np.where(anomalies)[0]
            sorted_anomalies = anomaly_idx[np.argsort(scores[anomaly_idx])][:10]

            for idx in sorted_anomalies:
                summary += f"  Row {idx}: score={scores[idx]:.4f}\n"
                for feat in features[:3]:
                    summary += f"    {feat}={X[feat].iloc[idx]:.2f}\n"
                summary += "\n"

        summary += f"<<COLOR:success>>SYNARA INTEGRATION:<</COLOR>>\n"
        summary += f"  Anomalies are observations that don't fit the current model.\n"
        summary += f"  This signals MISSING CAUSES - trigger causal expansion.\n"
        summary += f"  Investigate what makes these points different.\n"

        result["summary"] = summary

        # Plot 1: Anomaly scores distribution
        result["plots"].append({
            "title": "Anomaly Score Distribution",
            "data": [{
                "type": "histogram",
                "x": scores.tolist(),
                "nbinsx": 50,
                "marker": {"color": "rgba(74, 159, 110, 0.6)", "line": {"color": "#4a9f6e", "width": 1}},
                "name": "Scores"
            }, {
                "type": "scatter",
                "x": [0, 0],
                "y": [0, len(scores) / 10],
                "mode": "lines",
                "line": {"color": "#e85747", "dash": "dash", "width": 2},
                "name": "Threshold"
            }],
            "layout": {
                "height": 250,
                "xaxis": {"title": "Anomaly Score (negative = anomaly)"},
                "yaxis": {"title": "Count"},
                "shapes": [{"type": "line", "x0": 0, "x1": 0, "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "#e85747", "dash": "dash"}}]
            }
        })

        # Plot 2: Scatter with anomalies highlighted (first 2 features)
        if len(features) >= 2:
            colors = ['#e85747' if a else 'rgba(74, 159, 110, 0.5)' for a in anomalies]
            sizes = [12 if a else 6 for a in anomalies]

            result["plots"].append({
                "title": f"Anomalies: {features[0]} vs {features[1]}",
                "data": [{
                    "type": "scatter",
                    "x": X[features[0]].tolist(),
                    "y": X[features[1]].tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": colors,
                        "size": sizes,
                        "line": {"color": "#e85747", "width": [1 if a else 0 for a in anomalies]}
                    },
                    "text": [f"Score: {s:.3f}" for s in scores],
                    "hoverinfo": "text+x+y"
                }],
                "layout": {
                    "height": 350,
                    "xaxis": {"title": features[0]},
                    "yaxis": {"title": features[1]}
                }
            })

        # Store anomaly data for potential export
        result["anomalies"] = {
            "indices": np.where(anomalies)[0].tolist(),
            "scores": scores[anomalies].tolist(),
            "count": int(n_anomalies)
        }

        result["guide_observation"] = f"Isolation Forest detected {n_anomalies} anomalies ({anomaly_pct:.1f}%). These are 'missing cause' signals - investigate what makes them different."

    elif analysis_id == "gaussian_process":
        """
        Gaussian Process Regression - uncertainty quantification over subsets.
        Shows mean prediction with confidence bands.
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        from sklearn.preprocessing import StandardScaler

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Define kernel: constant * RBF + noise
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

        # Fit GP
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
        gp.fit(X_scaled, y)

        # Predictions with uncertainty
        y_pred, y_std = gp.predict(X_scaled, return_std=True)

        # For visualization, use first feature
        feature_for_plot = features[0]
        x_plot = X[feature_for_plot].values
        sort_idx = np.argsort(x_plot)

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>GAUSSIAN PROCESS REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n\n"

        summary += f"<<COLOR:text>>Kernel:<</COLOR>>\n"
        summary += f"  {gp.kernel_}\n\n"

        summary += f"<<COLOR:text>>Model Quality:<</COLOR>>\n"
        summary += f"  Log-Marginal-Likelihood: {gp.log_marginal_likelihood_value_:.4f}\n"

        # Residual analysis
        residuals = y.values - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        r2 = 1 - np.sum(residuals**2) / np.sum((y.values - y.mean())**2)
        summary += f"  RMSE: {rmse:.4f}\n"
        summary += f"  R²: {r2:.4f}\n\n"

        summary += f"<<COLOR:text>>Uncertainty Statistics:<</COLOR>>\n"
        summary += f"  Mean std: {y_std.mean():.4f}\n"
        summary += f"  Max std: {y_std.max():.4f}\n"
        summary += f"  Min std: {y_std.min():.4f}\n\n"

        summary += f"<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  GP provides full uncertainty quantification.\n"
        summary += f"  Wide bands = less confident predictions.\n"
        summary += f"  Useful for detecting extrapolation risk.\n"

        result["summary"] = summary

        # Plot 1: Predictions with uncertainty bands
        result["plots"].append({
            "title": f"GP Fit: {target} vs {feature_for_plot}",
            "data": [
                {
                    "type": "scatter",
                    "x": x_plot[sort_idx].tolist(),
                    "y": y.values[sort_idx].tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6},
                    "name": "Observed"
                },
                {
                    "type": "scatter",
                    "x": x_plot[sort_idx].tolist(),
                    "y": y_pred[sort_idx].tolist(),
                    "mode": "lines",
                    "line": {"color": "#4a9f6e", "width": 2},
                    "name": "GP Mean"
                },
                {
                    "type": "scatter",
                    "x": np.concatenate([x_plot[sort_idx], x_plot[sort_idx][::-1]]).tolist(),
                    "y": np.concatenate([
                        (y_pred[sort_idx] + 1.96 * y_std[sort_idx]),
                        (y_pred[sort_idx][::-1] - 1.96 * y_std[sort_idx][::-1])
                    ]).tolist(),
                    "fill": "toself",
                    "fillcolor": "rgba(74, 159, 110, 0.2)",
                    "line": {"color": "rgba(74, 159, 110, 0)"},
                    "name": "95% CI"
                }
            ],
            "layout": {
                "height": 350,
                "xaxis": {"title": feature_for_plot},
                "yaxis": {"title": target}
            }
        })

        # Plot 2: Uncertainty map
        result["plots"].append({
            "title": "Prediction Uncertainty",
            "data": [{
                "type": "scatter",
                "x": x_plot.tolist(),
                "y": y_std.tolist(),
                "mode": "markers",
                "marker": {
                    "color": y_std.tolist(),
                    "colorscale": [[0, "#4a9f6e"], [0.5, "#e8c547"], [1, "#e85747"]],
                    "size": 8,
                    "showscale": True,
                    "colorbar": {"title": "Std"}
                },
                "name": "Uncertainty"
            }],
            "layout": {
                "height": 250,
                "xaxis": {"title": feature_for_plot},
                "yaxis": {"title": "Prediction Std Dev"}
            }
        })

        result["guide_observation"] = f"Gaussian Process fit with R²={r2:.3f}. Mean uncertainty: {y_std.mean():.3f}. Check wide uncertainty bands for extrapolation risk."

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(user.id, model_key, {'gp': gp, 'scaler': scaler}, {
                'model_type': "Gaussian Process Regressor",
                'features': features,
                'target': target,
                'metrics': {'r2': float(r2)}
            })
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
        Y_loadings = pls.y_loadings_  # Q loadings
        X_weights = pls.x_weights_  # W weights

        # Predictions
        y_pred = pls.predict(X_scaled).flatten()
        residuals = y.values - y_pred
        rmse = np.sqrt(np.mean(residuals**2))
        r2 = 1 - np.sum(residuals**2) / np.sum((y.values - y.mean())**2)

        # Explained variance per component
        # PLS doesn't provide this directly, estimate from scores
        total_var_x = np.var(X_scaled, axis=0).sum()
        explained_var = []
        for i in range(n_components):
            comp_var = np.var(X_scores[:, i]) * np.sum(X_loadings[:, i]**2)
            explained_var.append(comp_var / total_var_x * 100)

        # Build summary
        summary = f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n"
        summary += f"<<COLOR:title>>PARTIAL LEAST SQUARES (PLS) REGRESSION<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"
        summary += f"<<COLOR:highlight>>Target:<</COLOR>> {target}\n"
        summary += f"<<COLOR:highlight>>Features:<</COLOR>> {', '.join(features)}\n"
        summary += f"<<COLOR:highlight>>Components:<</COLOR>> {n_components}\n\n"

        summary += f"<<COLOR:text>>Model Quality:<</COLOR>>\n"
        summary += f"  R²: {r2:.4f}\n"
        summary += f"  RMSE: {rmse:.4f}\n\n"

        summary += f"<<COLOR:text>>Component Details:<</COLOR>>\n"
        for i in range(n_components):
            summary += f"  Component {i+1}: ~{explained_var[i]:.1f}% X variance\n"
        summary += "\n"

        summary += f"<<COLOR:text>>Feature Weights (Importance):<</COLOR>>\n"
        # VIP scores (Variable Importance in Projection)
        vip = np.sqrt(len(features) * np.sum(X_weights**2 * np.sum(Y_scores**2, axis=0), axis=1) / np.sum(Y_scores**2))
        for feat, v in sorted(zip(features, vip), key=lambda x: -x[1]):
            marker = "★" if v > 1.0 else " "
            summary += f"  {marker} {feat}: VIP={v:.3f}\n"
        summary += "\n"

        summary += f"<<COLOR:success>>INTERPRETATION:<</COLOR>>\n"
        summary += f"  PLS handles collinear features by projecting to latent space.\n"
        summary += f"  VIP > 1.0 indicates important predictors.\n"
        summary += f"  Score plots reveal sample groupings.\n"

        result["summary"] = summary

        # Plot 1: Score plot (T1 vs T2)
        if n_components >= 2:
            result["plots"].append({
                "title": "PLS Score Plot (T1 vs T2)",
                "data": [{
                    "type": "scatter",
                    "x": X_scores[:, 0].tolist(),
                    "y": X_scores[:, 1].tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": y.values.tolist(),
                        "colorscale": [[0, "#4a9f6e"], [0.5, "#e8c547"], [1, "#e85747"]],
                        "size": 8,
                        "showscale": True,
                        "colorbar": {"title": target}
                    },
                    "text": [f"{target}={v:.2f}" for v in y.values],
                    "hoverinfo": "text+x+y"
                }],
                "layout": {
                    "height": 350,
                    "xaxis": {"title": "T1 (Component 1)"},
                    "yaxis": {"title": "T2 (Component 2)"}
                }
            })

        # Plot 2: Loading plot (feature contributions)
        result["plots"].append({
            "title": "PLS Loadings (Component 1)",
            "data": [{
                "type": "bar",
                "x": features,
                "y": X_loadings[:, 0].tolist(),
                "marker": {
                    "color": ["#4a9f6e" if v > 0 else "#e85747" for v in X_loadings[:, 0]]
                }
            }],
            "layout": {
                "height": 250,
                "xaxis": {"title": "Feature"},
                "yaxis": {"title": "Loading on Component 1"}
            }
        })

        # Plot 3: VIP scores
        vip_sorted = sorted(zip(features, vip), key=lambda x: -x[1])
        result["plots"].append({
            "title": "Variable Importance (VIP)",
            "data": [{
                "type": "bar",
                "x": [f[0] for f in vip_sorted],
                "y": [f[1] for f in vip_sorted],
                "marker": {
                    "color": ["#4a9f6e" if v > 1.0 else "rgba(74, 159, 110, 0.4)" for _, v in vip_sorted]
                }
            }, {
                "type": "scatter",
                "x": [f[0] for f in vip_sorted],
                "y": [1.0] * len(features),
                "mode": "lines",
                "line": {"color": "#e85747", "dash": "dash"},
                "name": "VIP=1 threshold"
            }],
            "layout": {
                "height": 250,
                "xaxis": {"title": "Feature"},
                "yaxis": {"title": "VIP Score"},
                "showlegend": False
            }
        })

        result["vip_scores"] = {feat: float(v) for feat, v in zip(features, vip)}
        result["guide_observation"] = f"PLS with {n_components} components achieved R²={r2:.3f}. Top features by VIP: {', '.join([f[0] for f in vip_sorted[:3]])}."

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(user.id, model_key, {'pls': pls, 'scaler': scaler}, {
                'model_type': "PLS Regression",
                'features': features,
                'target': target,
                'metrics': {'r2': float(r2), 'n_components': n_components}
            })
            result["model_key"] = model_key
            result["can_save"] = True

    elif analysis_id == "sem":
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
            result["summary"] = f"Warning: Only {len(model_df)} complete cases. SEM typically needs n > 200 for stable estimates."

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
        summary += f"<<COLOR:title>>STRUCTURAL EQUATION MODEL<</COLOR>>\n"
        summary += f"<<COLOR:accent>>{'═' * 70}<</COLOR>>\n\n"

        summary += f"<<COLOR:highlight>>Model Type:<</COLOR>> {model_type.title()}\n"
        summary += f"<<COLOR:highlight>>Outcome:<</COLOR>> {outcome}\n"
        summary += f"<<COLOR:highlight>>Predictors:<</COLOR>> {', '.join(predictors)}\n"
        if mediator:
            summary += f"<<COLOR:highlight>>Mediator:<</COLOR>> {mediator}\n"
        summary += f"<<COLOR:highlight>>N:<</COLOR>> {len(model_df)}\n\n"

        # Fit indices
        summary += f"<<COLOR:text>>Model Fit:<</COLOR>>\n"
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
        summary += f"<<COLOR:text>>Path Estimates:<</COLOR>>\n"
        for _, row in estimates.iterrows():
            lval = row.get('lval', '')
            op = row.get('op', '')
            rval = row.get('rval', '')
            est = row.get('Estimate', 0)
            se = row.get('Std. Err', 0)
            pval = row.get('p-value', 1)

            if op == '~':  # Regression
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                summary += f"  {lval} ← {rval}: β = {est:.3f} (SE={se:.3f}) {sig}\n"
            elif op == ':=':  # Defined parameter
                summary += f"  <<COLOR:accent>>{lval}<</COLOR>>: {est:.3f} (SE={se:.3f})\n"
        summary += "\n"

        # Interpretation for mediation
        if model_type == "mediation" and mediator:
            indirect_row = estimates[estimates['lval'] == 'indirect']
            if not indirect_row.empty:
                indirect_est = indirect_row['Estimate'].values[0]
                indirect_p = indirect_row.get('p-value', pd.Series([1])).values[0]

                summary += f"<<COLOR:success>>MEDIATION ANALYSIS:<</COLOR>>\n"
                if indirect_p < 0.05:
                    summary += f"  Significant indirect effect: {indirect_est:.3f}\n"
                    summary += f"  {mediator} mediates the {predictors[0]} → {outcome} relationship.\n"
                else:
                    summary += f"  No significant mediation (indirect = {indirect_est:.3f}, p > .05)\n"

        result["summary"] = summary

        # Plot 1: Path diagram (simplified as coefficient bar chart)
        path_data = estimates[estimates['op'] == '~'].copy()
        if not path_data.empty:
            labels = [f"{row['rval']} → {row['lval']}" for _, row in path_data.iterrows()]
            coefs = path_data['Estimate'].tolist()
            colors = ["#4a9f6e" if c > 0 else "#e85747" for c in coefs]

            result["plots"].append({
                "title": "Path Coefficients",
                "data": [{
                    "type": "bar",
                    "x": coefs,
                    "y": labels,
                    "orientation": "h",
                    "marker": {"color": colors}
                }],
                "layout": {
                    "height": max(200, len(labels) * 40),
                    "xaxis": {"title": "Standardized Coefficient"},
                    "yaxis": {"automargin": True},
                    "margin": {"l": 150}
                }
            })

        # Plot 2: Residuals vs fitted (if possible)
        try:
            fitted = mod.predict(model_df)
            if outcome in fitted.columns:
                y_fitted = fitted[outcome].values
                y_actual = model_df[outcome].values
                residuals = y_actual - y_fitted

                result["plots"].append({
                    "title": f"Residuals: {outcome}",
                    "data": [{
                        "type": "scatter",
                        "x": y_fitted.tolist(),
                        "y": residuals.tolist(),
                        "mode": "markers",
                        "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6}
                    }, {
                        "type": "scatter",
                        "x": [min(y_fitted), max(y_fitted)],
                        "y": [0, 0],
                        "mode": "lines",
                        "line": {"color": "#e85747", "dash": "dash"}
                    }],
                    "layout": {
                        "height": 250,
                        "xaxis": {"title": "Fitted"},
                        "yaxis": {"title": "Residual"},
                        "showlegend": False
                    }
                })
        except Exception:
            pass

        fit_quality = "good" if (cfi and cfi > 0.95 and rmsea and rmsea < 0.05) else "acceptable" if (cfi and cfi > 0.90) else "poor"
        result["guide_observation"] = f"SEM {model_type} model with {fit_quality} fit. Check path coefficients for significant relationships."

        # Cache model for saving
        model_key = str(uuid.uuid4())
        if user and user.is_authenticated:
            cache_model(user.id, model_key, mod, {
                'model_type': f"SEM ({model_type.title()})",
                'features': predictors,
                'target': outcome,
                'metrics': {'cfi': float(cfi) if cfi else None, 'rmsea': float(rmsea) if rmsea else None}
            })
            result["model_key"] = model_key
            result["can_save"] = True

    return result


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


def run_spc_analysis(df, analysis_id, config):
    """Run SPC analysis."""
    import numpy as np

    result = {"plots": [], "summary": "", "guide_observation": ""}

    measurement = config.get("measurement")
    data = df[measurement].dropna().values

    if analysis_id == "imr":
        # Individual-Moving Range chart
        n = len(data)
        mr = np.abs(np.diff(data))
        mr_bar = np.mean(mr)

        x_bar = np.mean(data)
        ucl = x_bar + 2.66 * mr_bar
        lcl = x_bar - 2.66 * mr_bar

        mr_ucl = 3.267 * mr_bar

        # I Chart
        result["plots"].append({
            "title": "I Chart (Individuals)",
            "data": [
                {"type": "scatter", "y": data.tolist(), "mode": "lines+markers", "name": "Value", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
                {"type": "scatter", "y": [x_bar]*n, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
                {"type": "scatter", "y": [ucl]*n, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
                {"type": "scatter", "y": [lcl]*n, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 250, "showlegend": True}
        })

        # MR Chart
        result["plots"].append({
            "title": "MR Chart (Moving Range)",
            "data": [
                {"type": "scatter", "y": mr.tolist(), "mode": "lines+markers", "name": "MR", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
                {"type": "scatter", "y": [mr_bar]*(n-1), "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
                {"type": "scatter", "y": [mr_ucl]*(n-1), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 250}
        })

        # Check for out-of-control points
        ooc = np.sum((data > ucl) | (data < lcl))
        result["summary"] = f"I-MR Chart Analysis\n\nMean: {x_bar:.4f}\nUCL: {ucl:.4f}\nLCL: {lcl:.4f}\nMR-bar: {mr_bar:.4f}\n\nOut-of-control points: {ooc}"

        result["guide_observation"] = f"Control chart shows {ooc} out-of-control points." + (" Process appears stable." if ooc == 0 else " Investigation recommended.")

    elif analysis_id == "capability":
        lsl = float(config.get("lsl")) if config.get("lsl") else None
        usl = float(config.get("usl")) if config.get("usl") else None
        target = float(config.get("target")) if config.get("target") else None

        mean = np.mean(data)
        std = np.std(data, ddof=1)

        summary = f"Capability Analysis\n\nMean: {mean:.4f}\nStd Dev: {std:.4f}\n"

        if lsl is not None and usl is not None:
            cp = (usl - lsl) / (6 * std)
            if target:
                cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
            else:
                cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))

            summary += f"\nCp: {cp:.3f}\nCpk: {cpk:.3f}"

            if cpk >= 1.33:
                summary += "\n\nProcess is capable (Cpk >= 1.33)"
            elif cpk >= 1.0:
                summary += "\n\nProcess is marginally capable (1.0 <= Cpk < 1.33)"
            else:
                summary += "\n\nProcess is NOT capable (Cpk < 1.0)"

            result["guide_observation"] = f"Process capability Cpk = {cpk:.2f}. " + ("Capable." if cpk >= 1.33 else "Needs improvement.")

        # Histogram with spec limits - Svend theme (pale green fill, bright green border)
        hist_data = [{
            "type": "histogram",
            "x": data.tolist(),
            "name": "Data",
            "marker": {
                "color": "rgba(74, 159, 110, 0.4)",  # Pale green fill
                "line": {"color": "#4a9f6e", "width": 1.5}  # Bright green border
            }
        }]

        # Add LSL/USL as dashed vertical lines (matching anomaly threshold style)
        shapes = []
        annotations = []

        if lsl:
            shapes.append({
                "type": "line",
                "x0": lsl,
                "x1": lsl,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e85747", "dash": "dash", "width": 2}
            })
            annotations.append({
                "x": lsl,
                "y": 1.05,
                "yref": "paper",
                "text": "LSL",
                "showarrow": False,
                "font": {"color": "#e85747", "size": 11}
            })

        if usl:
            shapes.append({
                "type": "line",
                "x0": usl,
                "x1": usl,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": "#e85747", "dash": "dash", "width": 2}
            })
            annotations.append({
                "x": usl,
                "y": 1.05,
                "yref": "paper",
                "text": "USL",
                "showarrow": False,
                "font": {"color": "#e85747", "size": 11}
            })

        result["plots"].append({
            "title": "Capability Histogram",
            "data": hist_data,
            "layout": {
                "template": "plotly_dark",
                "height": 300,
                "shapes": shapes,
                "annotations": annotations,
                "margin": {"t": 40}
            }
        })

        result["summary"] = summary

    elif analysis_id == "xbar_r":
        # Xbar-R Chart for subgrouped data
        subgroup_col = config.get("subgroup")
        subgroup_size = int(config.get("subgroup_size", 5))

        if subgroup_col:
            # Group by subgroup column
            groups = df.groupby(subgroup_col)[measurement].apply(list).values
        else:
            # Create subgroups from sequential data
            groups = [data[i:i+subgroup_size] for i in range(0, len(data), subgroup_size)]
            groups = [g for g in groups if len(g) == subgroup_size]

        groups = np.array([g for g in groups if len(g) >= 2])
        n_subgroups = len(groups)

        x_bars = np.array([np.mean(g) for g in groups])
        ranges = np.array([np.max(g) - np.min(g) for g in groups])

        x_double_bar = np.mean(x_bars)
        r_bar = np.mean(ranges)

        # Control chart constants (for subgroup size 2-10)
        d2_table = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
        d3_table = {2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848, 7: 0.833, 8: 0.820, 9: 0.808, 10: 0.797}
        A2_table = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
        D3_table = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
        D4_table = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}

        n = min(subgroup_size, 10)
        A2 = A2_table.get(n, 0.577)
        D3 = D3_table.get(n, 0)
        D4 = D4_table.get(n, 2.114)

        # Xbar limits
        xbar_ucl = x_double_bar + A2 * r_bar
        xbar_lcl = x_double_bar - A2 * r_bar

        # R limits
        r_ucl = D4 * r_bar
        r_lcl = D3 * r_bar

        # Xbar Chart
        result["plots"].append({
            "title": "Xbar Chart",
            "data": [
                {"type": "scatter", "y": x_bars.tolist(), "mode": "lines+markers", "name": "X̄", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
                {"type": "scatter", "y": [x_double_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
                {"type": "scatter", "y": [xbar_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
                {"type": "scatter", "y": [xbar_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 250, "showlegend": True, "xaxis": {"title": "Subgroup"}}
        })

        # R Chart
        result["plots"].append({
            "title": "R Chart",
            "data": [
                {"type": "scatter", "y": ranges.tolist(), "mode": "lines+markers", "name": "R", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
                {"type": "scatter", "y": [r_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
                {"type": "scatter", "y": [r_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
                {"type": "scatter", "y": [r_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 250, "xaxis": {"title": "Subgroup"}}
        })

        ooc_xbar = np.sum((x_bars > xbar_ucl) | (x_bars < xbar_lcl))
        ooc_r = np.sum(ranges > r_ucl)

        result["summary"] = f"Xbar-R Chart Analysis\n\nSubgroups: {n_subgroups}\nSubgroup size: {n}\n\nX̄ Chart:\n  X̿: {x_double_bar:.4f}\n  UCL: {xbar_ucl:.4f}\n  LCL: {xbar_lcl:.4f}\n  OOC points: {ooc_xbar}\n\nR Chart:\n  R̄: {r_bar:.4f}\n  UCL: {r_ucl:.4f}\n  LCL: {r_lcl:.4f}\n  OOC points: {ooc_r}"

    elif analysis_id == "xbar_s":
        # Xbar-S Chart (using standard deviation instead of range)
        subgroup_col = config.get("subgroup")
        subgroup_size = int(config.get("subgroup_size", 5))

        if subgroup_col:
            groups = df.groupby(subgroup_col)[measurement].apply(list).values
        else:
            groups = [data[i:i+subgroup_size] for i in range(0, len(data), subgroup_size)]
            groups = [g for g in groups if len(g) == subgroup_size]

        groups = np.array([g for g in groups if len(g) >= 2])
        n_subgroups = len(groups)

        x_bars = np.array([np.mean(g) for g in groups])
        stds = np.array([np.std(g, ddof=1) for g in groups])

        x_double_bar = np.mean(x_bars)
        s_bar = np.mean(stds)

        # Control chart constants for S chart
        c4_table = {2: 0.7979, 3: 0.8862, 4: 0.9213, 5: 0.9400, 6: 0.9515, 7: 0.9594, 8: 0.9650, 9: 0.9693, 10: 0.9727}
        B3_table = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0.030, 7: 0.118, 8: 0.185, 9: 0.239, 10: 0.284}
        B4_table = {2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716}
        A3_table = {2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182, 8: 1.099, 9: 1.032, 10: 0.975}

        n = min(subgroup_size, 10)
        A3 = A3_table.get(n, 1.427)
        B3 = B3_table.get(n, 0)
        B4 = B4_table.get(n, 2.089)

        xbar_ucl = x_double_bar + A3 * s_bar
        xbar_lcl = x_double_bar - A3 * s_bar
        s_ucl = B4 * s_bar
        s_lcl = B3 * s_bar

        # Xbar Chart
        result["plots"].append({
            "title": "Xbar Chart",
            "data": [
                {"type": "scatter", "y": x_bars.tolist(), "mode": "lines+markers", "name": "X̄", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
                {"type": "scatter", "y": [x_double_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
                {"type": "scatter", "y": [xbar_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
                {"type": "scatter", "y": [xbar_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 250, "showlegend": True, "xaxis": {"title": "Subgroup"}}
        })

        # S Chart
        result["plots"].append({
            "title": "S Chart",
            "data": [
                {"type": "scatter", "y": stds.tolist(), "mode": "lines+markers", "name": "S", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
                {"type": "scatter", "y": [s_bar]*n_subgroups, "mode": "lines", "name": "CL", "line": {"color": "#00b894"}},
                {"type": "scatter", "y": [s_ucl]*n_subgroups, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
                {"type": "scatter", "y": [s_lcl]*n_subgroups, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 250, "xaxis": {"title": "Subgroup"}}
        })

        result["summary"] = f"Xbar-S Chart Analysis\n\nSubgroups: {n_subgroups}\nSubgroup size: {n}\n\nX̄ Chart:\n  X̿: {x_double_bar:.4f}\n  UCL: {xbar_ucl:.4f}\n  LCL: {xbar_lcl:.4f}\n\nS Chart:\n  S̄: {s_bar:.4f}\n  UCL: {s_ucl:.4f}\n  LCL: {s_lcl:.4f}"

    elif analysis_id == "p_chart":
        # P Chart for proportion defective
        defectives = config.get("defectives")
        sample_size = config.get("sample_size")

        d = df[defectives].dropna().values
        n = df[sample_size].dropna().values
        p = d / n

        p_bar = np.sum(d) / np.sum(n)
        k = len(p)

        # Control limits (variable since sample size may vary)
        ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / n)
        lcl = np.maximum(0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / n))

        result["plots"].append({
            "title": "P Chart (Proportion Defective)",
            "data": [
                {"type": "scatter", "y": p.tolist(), "mode": "lines+markers", "name": "p", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
                {"type": "scatter", "y": [p_bar]*k, "mode": "lines", "name": "p̄", "line": {"color": "#00b894"}},
                {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
                {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Proportion"}}
        })

        ooc = np.sum((p > ucl) | (p < lcl))
        result["summary"] = f"P Chart Analysis\n\np̄: {p_bar:.4f} ({p_bar*100:.2f}%)\nSamples: {k}\n\nOut-of-control points: {ooc}"

    elif analysis_id == "np_chart":
        """
        NP Chart - Number defective (constant sample size).
        """
        defectives = config.get("defectives")
        sample_size = int(config.get("sample_size", 50))

        d = df[defectives].dropna().values
        n = sample_size
        k = len(d)

        np_bar = np.mean(d)
        p_bar = np_bar / n

        # Control limits
        ucl = np_bar + 3 * np.sqrt(np_bar * (1 - p_bar))
        lcl = max(0, np_bar - 3 * np.sqrt(np_bar * (1 - p_bar)))

        result["plots"].append({
            "title": "NP Chart (Number Defective)",
            "data": [
                {"type": "scatter", "y": d.tolist(), "mode": "lines+markers", "name": "np", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
                {"type": "scatter", "y": [np_bar]*k, "mode": "lines", "name": "n̄p", "line": {"color": "#00b894"}},
                {"type": "scatter", "y": [ucl]*k, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
                {"type": "scatter", "y": [lcl]*k, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Number Defective"}}
        })

        ooc = np.sum((d > ucl) | (d < lcl))
        result["summary"] = f"NP Chart Analysis\n\nn̄p: {np_bar:.2f}\nSample size: {n}\np̄: {p_bar:.4f}\nUCL: {ucl:.2f}\nLCL: {lcl:.2f}\n\nOut-of-control points: {ooc}"

    elif analysis_id == "c_chart":
        """
        C Chart - Count of defects per unit (constant opportunity).
        """
        defects = config.get("defects")

        c = df[defects].dropna().values
        k = len(c)

        c_bar = np.mean(c)

        # Control limits (Poisson-based)
        ucl = c_bar + 3 * np.sqrt(c_bar)
        lcl = max(0, c_bar - 3 * np.sqrt(c_bar))

        result["plots"].append({
            "title": "C Chart (Defects per Unit)",
            "data": [
                {"type": "scatter", "y": c.tolist(), "mode": "lines+markers", "name": "c", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
                {"type": "scatter", "y": [c_bar]*k, "mode": "lines", "name": "c̄", "line": {"color": "#00b894"}},
                {"type": "scatter", "y": [ucl]*k, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
                {"type": "scatter", "y": [lcl]*k, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Defects"}}
        })

        ooc = np.sum((c > ucl) | (c < lcl))
        result["summary"] = f"C Chart Analysis\n\nc̄: {c_bar:.2f}\nSamples: {k}\nUCL: {ucl:.2f}\nLCL: {lcl:.2f}\n\nOut-of-control points: {ooc}"

    elif analysis_id == "u_chart":
        """
        U Chart - Defects per unit (variable sample size).
        """
        defects = config.get("defects")
        units = config.get("units")

        c = df[defects].dropna().values
        n = df[units].dropna().values
        u = c / n
        k = len(u)

        u_bar = np.sum(c) / np.sum(n)

        # Variable control limits
        ucl = u_bar + 3 * np.sqrt(u_bar / n)
        lcl = np.maximum(0, u_bar - 3 * np.sqrt(u_bar / n))

        result["plots"].append({
            "title": "U Chart (Defects per Unit)",
            "data": [
                {"type": "scatter", "y": u.tolist(), "mode": "lines+markers", "name": "u", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
                {"type": "scatter", "y": [u_bar]*k, "mode": "lines", "name": "ū", "line": {"color": "#00b894"}},
                {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
                {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "Defects per Unit"}}
        })

        ooc = np.sum((u > ucl) | (u < lcl))
        result["summary"] = f"U Chart Analysis\n\nū: {u_bar:.4f}\nSamples: {k}\n\nOut-of-control points: {ooc}"

    elif analysis_id == "cusum":
        """
        CUSUM Chart - Cumulative Sum for detecting small shifts.
        """
        measurement = config.get("measurement")
        target = float(config.get("target", 0))  # Target value
        k_param = float(config.get("k", 0.5))  # Slack value (typically 0.5)
        h_param = float(config.get("h", 5))  # Decision interval

        data = df[measurement].dropna().values
        n = len(data)

        if target == 0:
            target = np.mean(data)

        # Estimate standard deviation
        sigma = np.std(data, ddof=1)

        # Standardize
        z = (data - target) / sigma

        # Calculate CUSUM
        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)

        for i in range(n):
            if i == 0:
                cusum_pos[i] = max(0, z[i] - k_param)
                cusum_neg[i] = max(0, -z[i] - k_param)
            else:
                cusum_pos[i] = max(0, cusum_pos[i-1] + z[i] - k_param)
                cusum_neg[i] = max(0, cusum_neg[i-1] - z[i] - k_param)

        result["plots"].append({
            "title": "CUSUM Chart",
            "data": [
                {"type": "scatter", "y": cusum_pos.tolist(), "mode": "lines", "name": "CUSUM+", "line": {"color": "#4a9f6e", "width": 2}},
                {"type": "scatter", "y": (-cusum_neg).tolist(), "mode": "lines", "name": "CUSUM-", "line": {"color": "#47a5e8", "width": 2}},
                {"type": "scatter", "y": [h_param]*n, "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
                {"type": "scatter", "y": [-h_param]*n, "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "CUSUM"}}
        })

        # Detect signals
        signals_pos = np.where(cusum_pos > h_param)[0]
        signals_neg = np.where(cusum_neg > h_param)[0]

        result["summary"] = f"CUSUM Chart Analysis\n\nTarget: {target:.4f}\nσ estimate: {sigma:.4f}\nk (slack): {k_param}\nh (decision): {h_param}\n\nUpward shift signals: {len(signals_pos)} at points {list(signals_pos[:5])}{'...' if len(signals_pos) > 5 else ''}\nDownward shift signals: {len(signals_neg)} at points {list(signals_neg[:5])}{'...' if len(signals_neg) > 5 else ''}"

    elif analysis_id == "ewma":
        """
        EWMA Chart - Exponentially Weighted Moving Average.
        Good for detecting small sustained shifts.
        """
        measurement = config.get("measurement")
        target = float(config.get("target", 0))
        lambda_param = float(config.get("lambda", 0.2))  # Smoothing parameter
        L = float(config.get("L", 3))  # Control limit width

        data = df[measurement].dropna().values
        n = len(data)

        if target == 0:
            target = np.mean(data)

        sigma = np.std(data, ddof=1)

        # Calculate EWMA
        ewma = np.zeros(n)
        ewma[0] = lambda_param * data[0] + (1 - lambda_param) * target

        for i in range(1, n):
            ewma[i] = lambda_param * data[i] + (1 - lambda_param) * ewma[i-1]

        # Control limits (they vary with time, approaching steady state)
        factor = lambda_param / (2 - lambda_param)
        ucl = target + L * sigma * np.sqrt(factor * (1 - (1 - lambda_param)**(2 * np.arange(1, n+1))))
        lcl = target - L * sigma * np.sqrt(factor * (1 - (1 - lambda_param)**(2 * np.arange(1, n+1))))

        # Steady-state limits
        ucl_ss = target + L * sigma * np.sqrt(factor)
        lcl_ss = target - L * sigma * np.sqrt(factor)

        result["plots"].append({
            "title": "EWMA Chart",
            "data": [
                {"type": "scatter", "y": ewma.tolist(), "mode": "lines+markers", "name": "EWMA", "marker": {"size": 5, "color": "#4a9f6e"}, "line": {"color": "#4a9f6e"}},
                {"type": "scatter", "y": [target]*n, "mode": "lines", "name": "Target", "line": {"color": "#00b894"}},
                {"type": "scatter", "y": ucl.tolist(), "mode": "lines", "name": "UCL", "line": {"color": "#d63031", "dash": "dash"}},
                {"type": "scatter", "y": lcl.tolist(), "mode": "lines", "name": "LCL", "line": {"color": "#d63031", "dash": "dash"}},
            ],
            "layout": {"template": "plotly_dark", "height": 300, "showlegend": True, "yaxis": {"title": "EWMA"}}
        })

        ooc = np.sum((ewma > ucl) | (ewma < lcl))
        result["summary"] = f"EWMA Chart Analysis\n\nTarget: {target:.4f}\nλ (smoothing): {lambda_param}\nL (sigma width): {L}\n\nSteady-state limits:\n  UCL: {ucl_ss:.4f}\n  LCL: {lcl_ss:.4f}\n\nOut-of-control points: {ooc}"

    return result


def run_visualization(df, analysis_id, config):
    """Create visualizations."""
    result = {"plots": [], "summary": ""}

    # SVEND theme colors
    theme_colors = ['#4a9f6e', '#4a9f6e', '#e89547', '#9f4a4a', '#e8c547', '#7a6a9a']

    if analysis_id == "histogram":
        var = config.get("var")
        bins = int(config.get("bins", 20))
        groupby = config.get("by")

        if groupby and groupby != "" and groupby != "None":
            traces = []
            for i, group in enumerate(df[groupby].dropna().unique()):
                traces.append({
                    "type": "histogram",
                    "x": df[df[groupby] == group][var].dropna().tolist(),
                    "name": str(group),
                    "opacity": 0.7,
                    "nbinsx": bins,
                    "marker": {"color": theme_colors[i % len(theme_colors)]}
                })
            result["plots"].append({
                "title": f"Histogram of {var} by {groupby}",
                "data": traces,
                "layout": {"height": 300, "barmode": "overlay", "showlegend": True}
            })
        else:
            result["plots"].append({
                "title": f"Histogram of {var}",
                "data": [{"type": "histogram", "x": df[var].dropna().tolist(), "nbinsx": bins, "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}}],
                "layout": {"height": 300}
            })

    elif analysis_id == "boxplot":
        var = config.get("var")
        groupby = config.get("by")

        if groupby and groupby != "" and groupby != "None":
            # Create separate box for each group with different colors
            traces = []
            for i, group in enumerate(df[groupby].dropna().unique()):
                traces.append({
                    "type": "box",
                    "y": df[df[groupby] == group][var].dropna().tolist(),
                    "name": str(group),
                    "marker": {"color": theme_colors[i % len(theme_colors)]},
                    "line": {"color": theme_colors[i % len(theme_colors)]}
                })
            result["plots"].append({
                "title": f"Box Plot of {var} by {groupby}",
                "data": traces,
                "layout": {"height": 300, "showlegend": True}
            })
        else:
            result["plots"].append({
                "title": f"Box Plot of {var}",
                "data": [{"type": "box", "y": df[var].dropna().tolist(), "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}, "line": {"color": "#4a9f6e"}}],
                "layout": {"height": 300}
            })

    elif analysis_id == "scatter":
        x_var = config.get("x")
        y_var = config.get("y")
        color_var = config.get("color")
        trendline = config.get("trendline", False)

        # Distinct colors for groups (fill, border) - semi-transparent fill with solid border
        group_colors = [
            ('rgba(74, 159, 110, 0.5)', '#4a9f6e'),   # Green
            ('rgba(232, 149, 71, 0.5)', '#e89547'),   # Orange
            ('rgba(159, 74, 74, 0.5)', '#9f4a4a'),    # Red
            ('rgba(71, 165, 232, 0.5)', '#47a5e8'),   # Blue
            ('rgba(232, 197, 71, 0.5)', '#e8c547'),   # Yellow
            ('rgba(122, 106, 154, 0.5)', '#7a6a9a'), # Purple
        ]

        data = []

        if color_var and color_var != "" and color_var != "None":
            # Create separate trace for each group
            groups = df[color_var].dropna().unique()
            for i, group in enumerate(groups):
                fill_color, border_color = group_colors[i % len(group_colors)]
                mask = df[color_var] == group
                data.append({
                    "type": "scatter",
                    "x": df.loc[mask, x_var].tolist(),
                    "y": df.loc[mask, y_var].tolist(),
                    "mode": "markers",
                    "marker": {
                        "color": fill_color,
                        "size": 8,
                        "line": {"color": border_color, "width": 1.5}
                    },
                    "name": str(group)
                })
        else:
            data.append({
                "type": "scatter",
                "x": df[x_var].tolist(),
                "y": df[y_var].tolist(),
                "mode": "markers",
                "marker": {
                    "color": "rgba(74, 159, 110, 0.5)",
                    "size": 8,
                    "line": {"color": "#4a9f6e", "width": 1.5}
                },
                "name": y_var
            })

        if trendline:
            import numpy as np
            x = df[x_var].dropna()
            y = df[y_var].loc[x.index].dropna()
            common_idx = x.index.intersection(y.index)
            x = x.loc[common_idx]
            y = y.loc[common_idx]
            if len(x) > 1:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                data.append({
                    "type": "scatter",
                    "x": [float(x.min()), float(x.max())],
                    "y": [float(p(x.min())), float(p(x.max()))],
                    "mode": "lines",
                    "line": {"color": "#e8c547", "dash": "dash"},
                    "name": "Trendline"
                })

        title = f"{y_var} vs {x_var}"
        if color_var and color_var != "" and color_var != "None":
            title += f" (by {color_var})"

        result["plots"].append({
            "title": title,
            "data": data,
            "layout": {
                "height": 300,
                "xaxis": {"title": x_var},
                "yaxis": {"title": y_var},
                "showlegend": len(data) > 1
            }
        })

    elif analysis_id == "heatmap":
        import numpy as np
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[numeric_cols].corr()

        result["plots"].append({
            "title": "Correlation Heatmap",
            "data": [{"type": "heatmap", "z": corr.values.tolist(), "x": numeric_cols, "y": numeric_cols, "colorscale": "RdBu", "zmid": 0}],
            "layout": {"template": "plotly_dark", "height": 400}
        })

    elif analysis_id == "pareto":
        category = config.get("category") or config.get("var")
        value_col = config.get("value")

        if value_col and value_col != "":
            # Sum values by category
            counts = df.groupby(category)[value_col].sum().sort_values(ascending=False)
        else:
            # Count occurrences
            counts = df[category].value_counts()

        cumulative = counts.cumsum() / counts.sum() * 100

        result["plots"].append({
            "title": f"Pareto Chart - {category}",
            "data": [
                {"type": "bar", "x": [str(x) for x in counts.index.tolist()], "y": counts.values.tolist(), "name": "Count", "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}}},
                {"type": "scatter", "x": [str(x) for x in counts.index.tolist()], "y": cumulative.values.tolist(), "name": "Cumulative %", "yaxis": "y2", "line": {"color": "#fdcb6e"}}
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 350,
                "yaxis2": {"overlaying": "y", "side": "right", "range": [0, 100], "title": "Cumulative %"}
            }
        })

        # Find 80% cutoff
        cutoff_idx = (cumulative >= 80).idxmax() if (cumulative >= 80).any() else counts.index[-1]
        vital_few = counts.loc[:cutoff_idx]
        result["summary"] = f"Pareto Analysis\n\nTotal categories: {len(counts)}\nVital few (80%): {len(vital_few)} categories\n\nTop contributors:\n"
        for cat, val in counts.head(5).items():
            pct = val / counts.sum() * 100
            result["summary"] += f"  {cat}: {val} ({pct:.1f}%)\n"

    elif analysis_id == "matrix":
        import numpy as np
        vars_list = config.get("vars", [])
        color_var = config.get("color")

        if len(vars_list) < 2:
            result["summary"] = "Please select at least 2 variables for matrix plot."
            return result

        # Create scatter matrix
        n_vars = len(vars_list)
        fig_data = []

        for i, y_var in enumerate(vars_list):
            for j, x_var in enumerate(vars_list):
                row = n_vars - i
                col = j + 1

                if i == j:
                    # Diagonal: histogram
                    fig_data.append({
                        "type": "histogram",
                        "x": df[x_var].dropna().tolist(),
                        "marker": {"color": "rgba(74, 159, 110, 0.4)", "line": {"color": "#4a9f6e", "width": 1.5}},
                        "xaxis": f"x{col if col > 1 else ''}",
                        "yaxis": f"y{row if row > 1 else ''}",
                        "showlegend": False
                    })
                else:
                    # Off-diagonal: scatter
                    fig_data.append({
                        "type": "scatter",
                        "x": df[x_var].tolist(),
                        "y": df[y_var].tolist(),
                        "mode": "markers",
                        "marker": {"color": "#4a9f6e", "size": 3},
                        "xaxis": f"x{col if col > 1 else ''}",
                        "yaxis": f"y{row if row > 1 else ''}",
                        "showlegend": False
                    })

        # Build layout with subplots
        layout = {
            "template": "plotly_dark",
            "height": 100 + n_vars * 120,
            "showlegend": False,
        }

        # Create axis layout
        for i in range(n_vars):
            col = i + 1
            row = n_vars - i
            x_key = f"xaxis{col if col > 1 else ''}"
            y_key = f"yaxis{row if row > 1 else ''}"

            layout[x_key] = {
                "domain": [i/n_vars + 0.02, (i+1)/n_vars - 0.02],
                "title": vars_list[i] if row == 1 else "",
                "showticklabels": row == 1
            }
            layout[y_key] = {
                "domain": [i/n_vars + 0.02, (i+1)/n_vars - 0.02],
                "title": vars_list[n_vars - 1 - i] if col == 1 else "",
                "showticklabels": col == 1
            }

        result["plots"].append({
            "title": "Matrix Plot",
            "data": fig_data,
            "layout": layout
        })

    elif analysis_id == "timeseries":
        import numpy as np
        x_col = config.get("x")
        y_cols = config.get("y", [])
        show_markers = config.get("markers", False)

        if isinstance(y_cols, str):
            y_cols = [y_cols]

        traces = []
        for i, y_col in enumerate(y_cols):
            trace = {
                "type": "scatter",
                "x": df[x_col].astype(str).tolist(),
                "y": df[y_col].tolist(),
                "mode": "lines+markers" if show_markers else "lines",
                "name": y_col,
                "line": {"color": theme_colors[i % len(theme_colors)]}
            }
            traces.append(trace)

        result["plots"].append({
            "title": f"Time Series: {', '.join(y_cols)}",
            "data": traces,
            "layout": {
                "template": "plotly_dark",
                "height": 350,
                "xaxis": {"title": x_col},
                "yaxis": {"title": "Value"},
                "showlegend": len(y_cols) > 1
            }
        })

    elif analysis_id == "probability":
        import numpy as np
        from scipy import stats

        var = config.get("var")
        dist = config.get("dist", "norm")

        x = df[var].dropna().values
        x_sorted = np.sort(x)
        n = len(x_sorted)

        # Calculate plotting positions (Hazen)
        pp = (np.arange(1, n + 1) - 0.5) / n

        # Get theoretical quantiles for chosen distribution
        if dist == "norm":
            theoretical = stats.norm.ppf(pp)
            dist_name = "Normal"
        elif dist == "lognorm":
            theoretical = stats.lognorm.ppf(pp, s=1)
            dist_name = "Lognormal"
            x_sorted = np.log(x_sorted[x_sorted > 0])
            theoretical = stats.norm.ppf(pp[:len(x_sorted)])
        elif dist == "expon":
            theoretical = stats.expon.ppf(pp)
            dist_name = "Exponential"
        elif dist == "weibull":
            # Weibull with shape=2
            theoretical = stats.weibull_min.ppf(pp, c=2)
            dist_name = "Weibull"
        else:
            theoretical = stats.norm.ppf(pp)
            dist_name = "Normal"

        # Fit line
        slope, intercept = np.polyfit(theoretical, x_sorted, 1)
        fit_line = slope * theoretical + intercept

        result["plots"].append({
            "title": f"Probability Plot ({dist_name}): {var}",
            "data": [
                {
                    "type": "scatter",
                    "x": theoretical.tolist(),
                    "y": x_sorted.tolist(),
                    "mode": "markers",
                    "marker": {"color": "rgba(74, 159, 110, 0.5)", "size": 6, "line": {"color": "#4a9f6e", "width": 1}},
                    "name": "Data"
                },
                {
                    "type": "scatter",
                    "x": theoretical.tolist(),
                    "y": fit_line.tolist(),
                    "mode": "lines",
                    "line": {"color": "#e89547", "dash": "dash"},
                    "name": "Fit Line"
                }
            ],
            "layout": {
                "template": "plotly_dark",
                "height": 350,
                "xaxis": {"title": f"Theoretical Quantiles ({dist_name})"},
                "yaxis": {"title": var}
            }
        })

        # Anderson-Darling test for normality
        if dist == "norm":
            ad_stat = stats.anderson(x)
            result["summary"] = f"Probability Plot ({dist_name})\n\nAnderson-Darling: {ad_stat.statistic:.4f}\nCritical values (15%, 10%, 5%, 2.5%, 1%):\n{ad_stat.critical_values}"

    return result


@csrf_exempt
@require_http_methods(["POST"])
@require_auth
def upload_data(request):
    """
    Upload and parse a data file for the Analysis Workbench.

    Returns columns with dtypes and a preview of the data.
    """
    if "file" not in request.FILES:
        return JsonResponse({"error": "No file provided"}, status=400)

    file = request.FILES["file"]
    filename = file.name.lower()

    try:
        import pandas as pd
        import numpy as np

        # Parse the file - try to detect actual format
        df = None
        parse_errors = []

        # Try based on extension first
        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(file)
            except Exception as e:
                parse_errors.append(f"CSV: {e}")

        elif filename.endswith('.xlsx'):
            try:
                df = pd.read_excel(file, engine='openpyxl')
            except Exception as e:
                parse_errors.append(f"XLSX: {e}")
                # Maybe it's actually a CSV with wrong extension
                file.seek(0)
                try:
                    df = pd.read_csv(file)
                    parse_errors.append("(Parsed as CSV)")
                except:
                    pass

        elif filename.endswith('.xls'):
            try:
                df = pd.read_excel(file, engine='xlrd')
            except Exception as e:
                parse_errors.append(f"XLS: {e}")
                # Maybe it's a CSV or XLSX with wrong extension
                file.seek(0)
                try:
                    df = pd.read_csv(file)
                    parse_errors.append("(Parsed as CSV)")
                except:
                    file.seek(0)
                    try:
                        df = pd.read_excel(file, engine='openpyxl')
                        parse_errors.append("(Parsed as XLSX)")
                    except:
                        pass
        else:
            # Unknown extension - try all formats
            for parser, name in [(lambda f: pd.read_csv(f), 'CSV'),
                                  (lambda f: pd.read_excel(f, engine='openpyxl'), 'XLSX'),
                                  (lambda f: pd.read_excel(f, engine='xlrd'), 'XLS')]:
                try:
                    file.seek(0)
                    df = parser(file)
                    break
                except:
                    continue

        if df is None:
            return JsonResponse({"error": f"Could not parse file. Tried: {'; '.join(parse_errors) or 'all formats'}"}, status=400)

        # Save to temp storage for session use
        data_id = f"data_{uuid.uuid4().hex[:12]}"
        try:
            data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
            data_dir.mkdir(parents=True, exist_ok=True)
            data_path = data_dir / f"{data_id}.csv"
            df.to_csv(data_path, index=False)
        except Exception as save_err:
            # Fall back to temp directory if MEDIA_ROOT not configured
            logger.warning(f"Could not save to MEDIA_ROOT: {save_err}, using temp")
            data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
            data_dir.mkdir(parents=True, exist_ok=True)
            data_path = data_dir / f"{data_id}.csv"
            df.to_csv(data_path, index=False)

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

        logger.info(f"Data uploaded: {request.user.username} - {file.name} ({df.shape[0]} rows, {df.shape[1]} cols)")

        # Preload LLM in background so it's ready when user asks questions
        _preload_llm_background()

        return JsonResponse({
            "id": data_id,
            "filename": file.name,
            "rows": df.shape[0],
            "columns": columns,
            "preview": preview,
        })

    except Exception as e:
        logger.error(f"Data upload error: {e}")
        return JsonResponse({"error": f"Failed to parse file: {str(e)}"}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
@gated
def execute_code(request):
    """
    Execute Python code in a sandboxed environment.

    Request body:
    {
        "code": "...",
        "data_id": "..."
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    code = body.get("code", "")
    data_id = body.get("data_id")

    if not code.strip():
        return JsonResponse({"error": "No code provided"}, status=400)

    try:
        import pandas as pd
        import numpy as np
        from io import StringIO
        import sys

        # Load data if provided
        df = None
        if data_id:
            from files.models import UploadedFile
            try:
                file_record = UploadedFile.objects.get(id=data_id, user=request.user)
                df = pd.read_csv(file_record.file.path) if file_record.file.path.endswith('.csv') else pd.read_excel(file_record.file.path)
            except UploadedFile.DoesNotExist:
                pass

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        # Import additional libraries for the sandbox
        import scipy
        import scipy.stats
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import random
        import math
        import statistics

        # Execute in namespace with common data science libraries
        namespace = {
            "df": df,
            "pd": pd,
            "np": np,
            "scipy": scipy,
            "stats": scipy.stats,
            "plt": plt,
            "random": random,
            "math": math,
            "statistics": statistics,
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "zip": zip,
                "enumerate": enumerate,
                "map": map,
                "filter": filter,
                "any": any,
                "all": all,
                "isinstance": isinstance,
                "type": type,
                "getattr": getattr,
                "setattr": setattr,
                "hasattr": hasattr,
                "__import__": __import__,
            }
        }

        exec(code, namespace)

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        # Check if there are any plot objects to return
        plots = []
        if "fig" in namespace:
            # Assume it's a plotly figure
            try:
                fig = namespace["fig"]
                plots.append({
                    "title": "Output",
                    "data": fig.data,
                    "layout": fig.layout
                })
            except:
                pass

        return JsonResponse({
            "output": output or "Code executed successfully",
            "plots": plots,
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
@gated
def generate_code(request):
    """
    Generate Python code from natural language.

    Uses Qwen Coder by default, or Anthropic models for Enterprise users.

    Request body:
    {
        "prompt": "Run a Monte Carlo simulation...",
        "model": "qwen" | "sonnet" | "opus" | "haiku",
        "context": {
            "hypothesis": "High temperature causes defects",
            "mechanism": "..."
        }
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    prompt = body.get("prompt", "").strip()
    if not prompt:
        return JsonResponse({"error": "Prompt is required"}, status=400)

    model = body.get("model", "qwen")
    context = body.get("context", {})

    # Build context prefix
    context_prefix = ""
    if context.get("hypothesis"):
        context_prefix = f"Context: Testing hypothesis '{context['hypothesis']}'\n"
        if context.get("mechanism"):
            context_prefix += f"Mechanism: {context['mechanism']}\n"

    # Check if using Anthropic models (Enterprise only)
    if model in ("sonnet", "opus", "haiku"):
        # Check enterprise access
        if not hasattr(request.user, 'subscription') or request.user.subscription.plan != 'enterprise':
            return JsonResponse({"error": "Anthropic models require Enterprise subscription"}, status=403)

        try:
            import anthropic

            system_prompt = """You are an expert Python code generator for data science and simulation.
Generate clean, executable Python code based on the user's request.

Rules:
- Only output Python code, no explanations or markdown
- Use numpy, pandas, scipy, matplotlib as needed
- Include print() statements for results
- Keep code concise but complete
- Add brief comments for clarity

Available libraries: numpy (np), pandas (pd), scipy, matplotlib (plt), random, math, statistics"""

            model_map = {
                "opus": "claude-opus-4-20250514",
                "sonnet": "claude-sonnet-4-20250514",
                "haiku": "claude-3-5-haiku-20241022",
            }

            client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=model_map.get(model, "claude-sonnet-4-20250514"),
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": f"{context_prefix}Generate Python code for: {prompt}"}]
            )

            code = response.content[0].text

            # Extract code from markdown if present
            import re
            if "```python" in code:
                match = re.search(r"```python\n?([\s\S]*?)```", code)
                if match:
                    code = match.group(1)
            elif "```" in code:
                match = re.search(r"```\n?([\s\S]*?)```", code)
                if match:
                    code = match.group(1)

            return JsonResponse({"code": code.strip(), "model": model})

        except Exception as e:
            logger.exception(f"Anthropic code generation error: {e}")
            return JsonResponse({"error": str(e)}, status=500)

    # Default: Use Qwen Coder
    code_prompt = f"""{context_prefix}Generate Python code for: {prompt}

Rules:
- Only output Python code, no explanations
- Use numpy, pandas, scipy as needed
- Include print() statements for results
- Keep code concise but complete"""

    try:
        # Get Qwen Coder LLM
        from . import views as agent_views
        llm = agent_views.get_coder_llm()

        if llm is None:
            # Fallback: return a template
            code = f'''import numpy as np
import pandas as pd

# TODO: Implement - {prompt}
# Qwen Coder is loading, please try again in a moment

print("Code generation requires Qwen Coder - loading in background...")
'''
            return JsonResponse({"code": code, "note": "Qwen Coder is loading, try again shortly"})

        # Generate code with Qwen
        code = llm.generate(code_prompt, max_tokens=1024, temperature=0.2)

        # Extract code from markdown if present
        import re
        if "```python" in code:
            match = re.search(r"```python\n?([\s\S]*?)```", code)
            if match:
                code = match.group(1)
        elif "```" in code:
            match = re.search(r"```\n?([\s\S]*?)```", code)
            if match:
                code = match.group(1)

        # Clean up - remove the prompt echo if present
        if code.startswith(code_prompt[:50]):
            code = code[len(code_prompt):].strip()

        return JsonResponse({"code": code.strip(), "model": "qwen"})

    except Exception as e:
        logger.exception(f"Code generation error: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@require_enterprise
def analyst_assistant(request):
    """
    AI assistant for data analysis questions.

    Supports multiple agent types:
    - analyst: Uses Qwen LLM to answer questions about loaded data
    - researcher: Searches the web for domain knowledge and scientific context
    - writer: Generates downloadable documents/reports

    Requires Enterprise tier.
    """
    try:
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        message = body.get("message", "")
        agent_type = body.get("agent_type", "analyst")
        selected_model = body.get("model", "default")
        context = body.get("context", {})
        data_id = context.get("data_id")
        columns = context.get("columns", [])
        data_preview = context.get("data_preview", [])

        # Load data if available
        df = None
        if data_id:
            try:
                import pandas as pd
                from io import StringIO

                if data_id.startswith("data_"):
                    data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = pd.read_csv(data_path)
                    else:
                        data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                        data_path = data_dir / f"{data_id}.csv"
                        if data_path.exists():
                            df = pd.read_csv(data_path)
                else:
                    from .models import TriageResult
                    try:
                        triage_result = TriageResult.objects.get(id=data_id, user=request.user)
                        df = pd.read_csv(StringIO(triage_result.cleaned_csv))
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Could not load data for analyst: {e}")

        # Get session history from context
        session_history = context.get("session_history", [])

        # Get shared LLM (non-blocking check)
        llm = None
        llm_loading = False
        cuda_available = False

        # Quick CUDA check
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if not cuda_available:
                logger.warning("CUDA not available - using keyword fallback")
        except ImportError:
            logger.warning("PyTorch not available")

        if cuda_available:
            try:
                from . import views as agent_views

                # Check if LLM is already loaded - don't block on first load
                if agent_views._shared_llm_loaded:
                    llm = agent_views._shared_llm
                    logger.info(f"LLM already loaded: type={type(llm).__name__ if llm else 'None'}")
                else:
                    # LLM not loaded yet - trigger background loading
                    llm_loading = True
                    logger.info("LLM not yet loaded - triggering background load")
                    import threading
                    def load_llm_background():
                        try:
                            agent_views.get_shared_llm()
                            logger.info("Background LLM load completed")
                        except Exception as e:
                            logger.error(f"Background LLM load failed: {e}")
                    threading.Thread(target=load_llm_background, daemon=True).start()

            except Exception as e:
                logger.error(f"Failed to check LLM status: {e}")
                import traceback
                traceback.print_exc()

        # Handle different agent types
        if agent_type == "researcher":
            # Researcher agent: web search for domain knowledge
            response, sources = generate_researcher_response(message, df, columns, data_preview)
            log_agent_action(request.user, "researcher", "research", success=True)
            return JsonResponse({"response": response, "sources": sources})

        elif agent_type == "writer":
            # Writer agent: generate downloadable documents
            response, document, filename = generate_writer_response(message, df, columns, session_history)
            log_agent_action(request.user, "writer", "write", success=True)
            return JsonResponse({"response": response, "document": document, "filename": filename})

        # Default: Analyst agent
        # Check for enterprise model selection (Opus/Sonnet/Haiku)
        if selected_model in ("opus", "sonnet", "haiku"):
            try:
                response = generate_anthropic_response(selected_model, message, df, columns, session_history)
                log_agent_action(request.user, "analyst", "question", success=True)
                return JsonResponse({"response": response, "model": selected_model})
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                response = f"Model error: {str(e)}"
                return JsonResponse({"response": response})

        if llm_loading:
            # LLM is loading in background - provide helpful response with keyword fallback
            logger.info("LLM loading in background, using enhanced fallback")
            response = generate_analyst_response(message.lower(), df, columns)
            response = "*(Qwen LLM is loading in the background - using quick response mode)*\n\n" + response
        elif llm is None:
            logger.info("Using keyword-based fallback response")
            response = generate_analyst_response(message.lower(), df, columns)
        else:
            logger.info(f"Using LLM for response, message: {message[:50]}...")
            try:
                response = generate_llm_response(llm, message, df, columns, session_history)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                import traceback
                traceback.print_exc()
                response = f"LLM error: {str(e)}"

        log_agent_action(request.user, "analyst", "question", success=True)
        return JsonResponse({"response": response})

    except Exception as e:
        logger.exception(f"Analyst assistant error: {e}")
        return JsonResponse({"response": f"Error: {str(e)}"})


def generate_anthropic_response(model, message, df, columns, session_history=None):
    """Generate analyst response using Anthropic API (Opus/Sonnet/Haiku)."""
    import anthropic
    import numpy as np

    model_map = {
        "opus": "claude-opus-4-20250514",
        "sonnet": "claude-sonnet-4-20250514",
        "haiku": "claude-3-5-haiku-20241022",
    }

    # Build data context
    if df is None:
        data_context = "No dataset loaded."
    else:
        n_rows, n_cols = df.shape
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        data_context = f"""Dataset: {n_rows:,} rows × {n_cols} columns
Numeric columns: {', '.join(numeric_cols[:10])}
Categorical columns: {', '.join(cat_cols[:10])}

Summary:
{df.describe().to_string()}

Sample data (first 5 rows):
{df.head().to_string()}
"""

    system_prompt = f"""You are an expert data analyst assistant in a Decision Science Workbench.
You help users understand their data, suggest analyses, and interpret results.

Current data context:
{data_context}

Be concise but thorough. Use markdown formatting for clarity."""

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = client.messages.create(
        model=model_map.get(model, "claude-sonnet-4-20250514"),
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": message}]
    )

    return response.content[0].text


def generate_llm_response(llm, message, df, columns, session_history=None):
    """Generate response using Qwen LLM as a lab assistant."""
    import numpy as np

    # Build data context
    if df is None:
        return "Please load a dataset first. Click the folder icon in the toolbar to load data from Triage or upload a CSV file."

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build correlation matrix for numeric columns (helps answer relationship questions)
    corr_info = ""
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr()
            # Find top correlations
            corr_pairs = []
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.3:  # Only notable correlations
                        corr_pairs.append((col1, col2, corr_val))
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            if corr_pairs:
                corr_info = "\n\nNOTABLE CORRELATIONS:\n"
                for col1, col2, r in corr_pairs[:10]:
                    strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.5 else "weak"
                    direction = "positive" if r > 0 else "negative"
                    corr_info += f"- {col1} ↔ {col2}: r={r:.3f} ({strength} {direction})\n"
        except Exception:
            pass

    # Build data summary for context
    data_context = f"""Dataset: {n_rows:,} rows × {n_cols} columns

Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:15])}
Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:15])}

Summary statistics:
{df.describe().to_string()}{corr_info}

Sample data (first 3 rows):
{df.head(3).to_string()}
"""

    # Build session history context
    session_context = ""
    if session_history:
        session_context = "\n\nSESSION HISTORY (analyses run this session):\n"
        for item in session_history[-10:]:
            session_context += f"- {item.get('type', 'unknown')}: {item.get('name', '')} - {item.get('summary', '')[:200]}\n"

    # Build prompt - lab assistant persona
    prompt = f"""You are a lab assistant helping a scientist analyze their data. You're knowledgeable, helpful, and speak like a colleague - not a generic chatbot.

Your role:
- Answer questions directly and specifically about THIS data
- Explain patterns, anomalies, or relationships you see
- Suggest appropriate statistical tests with reasoning
- Reference specific column names and values
- Be concise but insightful

Available analysis tools:
- Stat: Descriptive Stats, t-tests, ANOVA, Regression, Correlation, Normality, Chi-Square
- ML: Classification (RF, XGBoost, SVM), Clustering (K-Means, DBSCAN), PCA
- SPC: Control charts (I-MR, Xbar-R), Capability Analysis
- Graph: Histogram, Boxplot, Scatter, Matrix Plot, Time Series, Pareto

DATA:
{data_context}{session_context}

USER: {message}

Respond as a helpful lab assistant. Be specific to this data. If they ask about relationships, look at the correlations. If they ask what to analyze, give concrete suggestions based on the actual columns. Keep response under 250 words."""

    try:
        import concurrent.futures
        # Use a thread with timeout to prevent hanging
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(llm.generate, prompt, max_tokens=500, temperature=0.7)
            try:
                response = future.result(timeout=30)  # 30 second timeout
                return response
            except concurrent.futures.TimeoutError:
                logger.warning("LLM generation timed out after 30s")
                return generate_analyst_response(message.lower(), df, columns) + "\n\n*(Response generated via quick mode due to timeout)*"
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        # Fallback
        return generate_analyst_response(message.lower(), df, columns)


def generate_analyst_response(message, df, columns):
    """Generate helpful response based on the question and data."""
    import numpy as np

    # No data loaded
    if df is None or len(columns) == 0:
        return "Please load a dataset first. Click the folder icon in the toolbar to load data from Triage or upload a CSV file."

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Keywords for different intents
    if any(w in message for w in ['describe', 'summary', 'overview', 'tell me about', 'what is']):
        summary = f"Your dataset has {n_rows:,} rows and {n_cols} columns.\n\n"
        summary += f"**Numeric columns ({len(numeric_cols)}):** {', '.join(numeric_cols[:5])}"
        if len(numeric_cols) > 5:
            summary += f" (+{len(numeric_cols)-5} more)"
        summary += f"\n\n**Categorical columns ({len(cat_cols)}):** {', '.join(cat_cols[:5])}"
        if len(cat_cols) > 5:
            summary += f" (+{len(cat_cols)-5} more)"

        if numeric_cols:
            summary += "\n\n**Quick stats for numeric columns:**\n"
            for col in numeric_cols[:3]:
                summary += f"- {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}\n"

        summary += "\n\nUse **Stat > Descriptive Statistics** for detailed analysis."
        return summary

    if any(w in message for w in ['correlation', 'relationship', 'related', 'correlated']):
        if len(numeric_cols) < 2:
            return "You need at least 2 numeric columns to analyze correlations."

        corr_matrix = df[numeric_cols].corr()
        # Find strongest correlations
        pairs = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                pairs.append((col1, col2, corr_matrix.loc[col1, col2]))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        response = "**Correlation Analysis:**\n\n"
        response += "Strongest relationships:\n"
        for col1, col2, corr in pairs[:5]:
            strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
            direction = "positive" if corr > 0 else "negative"
            response += f"- {col1} & {col2}: r={corr:.3f} ({strength} {direction})\n"

        response += "\n\nTo visualize, use **Graph > Scatterplot** with these variable pairs."
        return response

    if any(w in message for w in ['predict', 'forecast', 'ml', 'machine learning', 'model']):
        response = "**ML Recommendations:**\n\n"

        if cat_cols:
            response += f"For **classification** (predicting categories like '{cat_cols[0]}'), use:\n"
            response += "- **ML > Classification** with Random Forest or XGBoost\n\n"

        if numeric_cols:
            response += f"For **regression** (predicting values like '{numeric_cols[0]}'), use:\n"
            response += "- **ML > Regression** with Random Forest or XGBoost\n\n"

        response += "For **finding patterns/groups**, use:\n"
        response += "- **ML > Clustering** to discover natural groupings\n"
        response += "- **ML > PCA** to reduce dimensions and visualize\n\n"

        response += "**Tip:** Select your target variable and features in the dialog. Start with Random Forest - it works well without tuning."
        return response

    if any(w in message for w in ['outlier', 'anomaly', 'unusual', 'extreme']):
        if not numeric_cols:
            return "No numeric columns found for outlier detection."

        response = "**Outlier Detection:**\n\n"
        for col in numeric_cols[:4]:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)][col]
            if len(outliers) > 0:
                response += f"- **{col}**: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)\n"

        response += "\n\nUse **Graph > Boxplot** to visualize outliers, or use **Triage** to clean them."
        return response

    if any(w in message for w in ['missing', 'null', 'empty', 'na']):
        response = "**Missing Data Analysis:**\n\n"
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) == 0:
            response += "No missing values found in your dataset."
        else:
            response += "Columns with missing values:\n"
            for col, count in missing.head(10).items():
                response += f"- **{col}**: {count} ({count/len(df)*100:.1f}%)\n"

            response += "\n\nUse **Triage** to handle missing values (imputation, removal)."
        return response

    if any(w in message for w in ['compare', 'difference', 'group', 'between']):
        if not cat_cols:
            return "No categorical columns found for group comparisons. Use a categorical variable to split your data."

        response = "**Group Comparisons:**\n\n"
        response += f"You can compare groups using '{cat_cols[0]}' or other categorical variables.\n\n"
        response += "**Recommended analyses:**\n"
        response += "- **Stat > Two-Sample t** - Compare means between 2 groups\n"
        response += "- **Stat > ANOVA** - Compare means across multiple groups\n"
        response += "- **Graph > Boxplot** with 'Group by' - Visualize differences\n"
        response += "- **Graph > Histogram** with 'Group by' - Compare distributions"
        return response

    if any(w in message for w in ['distribution', 'normal', 'spread', 'histogram']):
        if not numeric_cols:
            return "No numeric columns found for distribution analysis."

        response = "**Distribution Analysis:**\n\n"
        for col in numeric_cols[:3]:
            skew = df[col].skew()
            skew_desc = "right-skewed" if skew > 0.5 else "left-skewed" if skew < -0.5 else "approximately symmetric"
            response += f"- **{col}**: {skew_desc} (skewness={skew:.2f})\n"

        response += "\n\n**To visualize:**\n"
        response += "- **Graph > Histogram** for distribution shape\n"
        response += "- **Stat > Normality Test** to test for normality"
        return response

    # Default response
    response = f"I can help you analyze your dataset ({n_rows:,} rows, {n_cols} columns).\n\n"
    response += "**Try asking about:**\n"
    response += "- \"Describe my data\" - Get an overview\n"
    response += "- \"Find correlations\" - Discover relationships\n"
    response += "- \"Check for outliers\" - Find unusual values\n"
    response += "- \"Missing data\" - Analyze gaps\n"
    response += "- \"How to predict X\" - ML recommendations\n"
    response += "- \"Compare groups\" - Statistical comparisons\n\n"
    response += "Or use the **Stat**, **ML**, and **Graph** menus above."
    return response


def generate_researcher_response(message, df, columns, data_preview):
    """
    Researcher agent: searches the web for domain knowledge and scientific context.
    Uses ddgs library for real web search results with intelligent query construction.
    """
    import re

    # Build context about the data for better search
    data_context = ""
    if df is not None:
        data_context = f"Dataset has {len(df)} rows with columns: {', '.join(columns[:10])}"
        if data_preview:
            sample_vals = []
            for col in columns[:5]:
                if col in df.columns:
                    vals = df[col].dropna().head(3).tolist()
                    sample_vals.append(f"{col}: {vals}")
            data_context += f"\nSample values: {'; '.join(sample_vals[:3])}"

    # Extract key technical terms from the question
    # Remove common question words
    query_clean = re.sub(r'\b(can you|could you|please|research|tell me about|what is the|why do|why does|how does|explain|relationship between|correlation between)\b', '', message.lower())
    query_clean = query_clean.strip()

    # Extract potential chemical/technical terms (capitalized words or known patterns)
    technical_terms = re.findall(r'\b[A-Za-z]{4,}\b', message)
    # Filter to likely technical terms (not common words)
    common_words = {'what', 'that', 'this', 'with', 'from', 'have', 'been', 'were', 'they', 'their', 'about', 'which', 'when', 'there', 'would', 'could', 'should', 'between', 'relationship', 'correlation', 'drinking', 'water', 'samples'}
    technical_terms = [t.lower() for t in technical_terms if t.lower() not in common_words]

    sources = []
    search_results = []

    # Try ddgs library
    try:
        from ddgs import DDGS
        ddgs = DDGS()

        # Build multiple targeted searches
        searches = []

        # If we found technical terms, search for their relationship
        if len(technical_terms) >= 2:
            # Search for the specific interaction/relationship
            term_combo = ' '.join(technical_terms[:3])
            searches.append(f'"{technical_terms[0]}" "{technical_terms[1]}" correlation co-occurrence site:epa.gov OR site:pubmed OR site:ncbi.nlm.nih.gov')
            searches.append(f'{term_combo} water contamination research')
            searches.append(f'"{technical_terms[0]}" "{technical_terms[1]}" drinking water study')
        else:
            # Fallback to cleaned query
            searches.append(f'{query_clean} EPA research')
            searches.append(f'{query_clean} scientific study')

        seen_urls = set()
        for search_query in searches:
            try:
                results = list(ddgs.text(search_query, max_results=3))
                for r in results:
                    url = r.get('href', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        search_results.append({
                            'title': r.get('title', 'Result'),
                            'snippet': r.get('body', ''),
                            'url': url
                        })
                        sources.append({'title': r.get('title', 'Source')[:60], 'url': url})
            except Exception as e:
                logger.warning(f"Search query failed: {e}")
                continue

    except ImportError:
        logger.warning("ddgs not installed, trying fallback")
        # Fallback to requests-based search
        try:
            import requests
            import urllib.parse
            from bs4 import BeautifulSoup

            # Use DuckDuckGo HTML search
            search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query_clean + ' scientific')}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            resp = requests.get(search_url, headers=headers, timeout=15)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                for result in soup.select('.result')[:5]:
                    title_el = result.select_one('.result__title')
                    snippet_el = result.select_one('.result__snippet')
                    link_el = result.select_one('.result__url')

                    if title_el and snippet_el:
                        title = title_el.get_text(strip=True)
                        snippet = snippet_el.get_text(strip=True)
                        url = link_el.get('href', '') if link_el else ''

                        search_results.append({
                            'title': title,
                            'snippet': snippet,
                            'url': url
                        })
                        if url:
                            sources.append({'title': title[:60], 'url': url})

        except Exception as e:
            logger.warning(f"Fallback search failed: {e}")

    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")

    # Use LLM to synthesize search results into intelligent response
    if search_results:
        # Try to use shared LLM for synthesis
        llm = None
        try:
            from . import views as agent_views
            if agent_views._shared_llm_loaded:
                llm = agent_views._shared_llm
        except Exception:
            pass

        if llm:
            # Build context for LLM synthesis
            search_context = "\n\n".join([
                f"Source: {r['title']}\n{r['snippet']}"
                for r in search_results[:5] if r.get('snippet')
            ])

            synthesis_prompt = f"""You are a research assistant helping analyze data. The user asked: "{message}"

Here is what web research found:

{search_context}

User's data context: {data_context}

Based on the search results and the user's data, provide a helpful synthesis that:
1. Directly answers their question about the relationship/topic
2. Explains any scientific mechanisms or reasons
3. Relates the findings to their specific data columns if relevant
4. Notes any important caveats or additional considerations

Keep response under 250 words. Be specific and scientific, not generic."""

            try:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(llm.generate, synthesis_prompt, max_tokens=400, temperature=0.7)
                    synthesized = future.result(timeout=30)

                response = f"**Research Analysis:** *{message}*\n\n"
                response += synthesized + "\n\n"

                if sources:
                    response += "**Sources:**\n"
                    for src in sources[:5]:
                        if src.get('url'):
                            response += f"- [{src.get('title', 'Link')[:50]}]({src['url']})\n"

                return response, sources

            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}")
                # Fall through to basic response

        # Fallback: basic response without LLM
        response = f"**Research findings for:** *{message}*\n\n"

        for i, result in enumerate(search_results[:4], 1):
            if result['snippet']:
                response += f"**{result.get('title', f'Finding {i}')}**\n"
                response += f"{result['snippet']}\n\n"

        if data_context:
            response += f"---\n**Your data context:** {data_context}\n\n"

        if sources:
            response += "**Sources:**\n"
            for src in sources[:5]:
                if src.get('url'):
                    response += f"- [{src.get('title', 'Link')[:50]}]({src['url']})\n"
    else:
        response = f"I searched for information about *{message}* but didn't find specific results.\n\n"
        response += "**Suggestions:**\n"
        response += "- Try rephrasing your question with more specific terms\n"
        response += "- Ask about specific chemicals, processes, or scientific concepts\n"
        response += "- Use the Analyst agent for data-specific questions\n"

        if data_context:
            response += f"\n**Your data context:** {data_context}"

    return response, sources


def generate_writer_response(message, df, columns, session_history):
    """
    Writer agent: generates downloadable documents/reports using LLM.
    Creates intelligent markdown documents with analysis and insights.
    """
    import numpy as np
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

    if df is None:
        document = "# Analysis Report\n\nNo data loaded. Please load a dataset first.\n"
        return "Please load a dataset first to generate a document.", document, filename

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build data context for LLM
    stats_summary = df.describe().to_string() if numeric_cols else "No numeric columns"

    # Session history context
    history_text = ""
    if session_history:
        history_text = "\n".join([
            f"- {item.get('name', 'Analysis')}: {item.get('summary', '')[:150]}"
            for item in session_history[-10:]
        ])

    # Try to use LLM for intelligent document generation
    llm = None
    try:
        from . import views as agent_views
        if agent_views._shared_llm_loaded:
            llm = agent_views._shared_llm
    except Exception:
        pass

    if llm:
        writer_prompt = f"""You are a technical writer creating a data analysis report. The user requested: "{message}"

DATA CONTEXT:
- Dataset: {n_rows:,} rows × {n_cols} columns
- Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}
- Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:10])}

SUMMARY STATISTICS:
{stats_summary}

ANALYSES PERFORMED THIS SESSION:
{history_text if history_text else "No analyses run yet"}

Write a professional markdown report that includes:
1. Executive Summary (2-3 sentences answering what the user asked for)
2. Data Overview (brief description of the dataset)
3. Key Findings (based on the statistics and any analyses performed)
4. Recommendations (next steps for analysis)

Use proper markdown formatting with headers (##), bullet points, and tables where appropriate.
Keep it concise but informative (under 500 words)."""

        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(llm.generate, writer_prompt, max_tokens=800, temperature=0.7)
                llm_content = future.result(timeout=45)

            document = f"# Analysis Report\n\n"
            document += f"*Generated: {timestamp}*\n\n"
            document += llm_content + "\n\n"
            document += "---\n\n"
            document += "## Appendix: Data Summary\n\n"
            document += f"**Dataset:** {n_rows:,} rows × {n_cols} columns\n\n"
            document += "| Variable | Type | Missing | Unique |\n"
            document += "|----------|------|---------|--------|\n"
            for col in df.columns[:15]:
                dtype = "Numeric" if col in numeric_cols else "Categorical"
                missing = int(df[col].isna().sum())
                unique = int(df[col].nunique())
                document += f"| {col} | {dtype} | {missing} | {unique} |\n"

            response = "I've written a detailed analysis report based on your data and session history.\n\n"
            response += "The document includes an executive summary, key findings, and recommendations.\n\n"
            response += "Click the download link below to save."

            return response, document, filename

        except Exception as e:
            logger.warning(f"LLM writer failed: {e}")

    # Fallback: static document
    document = f"# Data Analysis Report\n\n"
    document += f"*Generated: {timestamp}*\n\n"
    document += f"*Request: {message}*\n\n"

    document += "## Data Overview\n\n"
    document += f"- **Rows:** {n_rows:,}\n"
    document += f"- **Columns:** {n_cols}\n"
    document += f"- **Numeric variables:** {len(numeric_cols)}\n"
    document += f"- **Categorical variables:** {len(cat_cols)}\n\n"

    document += "### Variables\n\n"
    document += "| Variable | Type | Missing | Unique |\n"
    document += "|----------|------|---------|--------|\n"
    for col in df.columns[:20]:
        dtype = "Numeric" if col in numeric_cols else "Categorical"
        missing = int(df[col].isna().sum())
        unique = int(df[col].nunique())
        document += f"| {col} | {dtype} | {missing} | {unique} |\n"

    if numeric_cols:
        document += "\n## Summary Statistics\n\n"
        document += "| Variable | Mean | Std Dev | Min | Max |\n"
        document += "|----------|------|---------|-----|-----|\n"
        for col in numeric_cols[:10]:
            stats = df[col].describe()
            document += f"| {col} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |\n"

    if session_history:
        document += "\n## Analyses Performed\n\n"
        for item in session_history[-10:]:
            document += f"- **{item.get('name', 'Analysis')}**: {item.get('summary', '')[:100]}\n"

    response = f"I've prepared a report document for your analysis.\n\n"
    response += f"**Document includes:**\n"
    response += f"- Data overview ({n_rows:,} rows × {n_cols} columns)\n"
    response += f"- Variable listing with types and missing values\n"
    if numeric_cols:
        response += f"- Summary statistics for {len(numeric_cols)} numeric variables\n"
    response += f"\nClick the download link below to save."

    return response, document, filename


# ============================================================================
# DATA TRANSFORMATION TOOLS
# ============================================================================

@csrf_exempt
@require_http_methods(["POST"])
@gated
def transform_data(request):
    """
    Apply data transformation tools (subset, sort, calculator, etc.)
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    tool = body.get("tool")
    config = body.get("config", {})
    data_id = body.get("data_id")

    if not data_id:
        return JsonResponse({"error": "No data loaded"}, status=400)

    try:
        import pandas as pd
        import numpy as np
        from io import StringIO

        # Load data
        df = None

        if data_id.startswith("data_"):
            try:
                data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path)
            except Exception:
                pass

            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = pd.read_csv(data_path)
                except Exception:
                    pass

        if df is None:
            try:
                from .models import TriageResult
                triage_result = TriageResult.objects.get(id=data_id, user=request.user)
                df = pd.read_csv(StringIO(triage_result.cleaned_csv))
            except Exception:
                pass

        if df is None:
            return JsonResponse({"error": "Data not found"}, status=404)

        # Apply transformation
        result_df = df.copy()
        message = ""

        if tool == "calculator":
            new_col = config.get("new_col", "").strip()
            expression = config.get("expression", "").strip()

            if not new_col or not expression:
                return JsonResponse({"error": "Column name and expression required"}, status=400)

            # Safe evaluation - only allow column names and basic math
            try:
                # Replace column names with df['column'] syntax
                safe_expr = expression
                for col in df.columns:
                    safe_expr = safe_expr.replace(col, f"df['{col}']")

                # Allow basic numpy functions
                safe_locals = {
                    'df': df,
                    'np': np,
                    'log': np.log,
                    'log10': np.log10,
                    'sqrt': np.sqrt,
                    'abs': np.abs,
                    'exp': np.exp,
                    'sin': np.sin,
                    'cos': np.cos,
                    'round': np.round,
                }

                result_df[new_col] = eval(safe_expr, {"__builtins__": {}}, safe_locals)
                message = f"Created column '{new_col}'"
            except Exception as e:
                return JsonResponse({"error": f"Expression error: {str(e)}"}, status=400)

        elif tool == "subset":
            filter_col = config.get("filter_col")
            condition = config.get("condition")
            filter_value = config.get("filter_value", "")

            if condition == "isna":
                result_df = df[df[filter_col].isna()]
            elif condition == "notna":
                result_df = df[df[filter_col].notna()]
            else:
                # Try to convert value to appropriate type
                try:
                    if df[filter_col].dtype in ['int64', 'float64']:
                        filter_value = float(filter_value)
                except:
                    pass

                if condition == "eq":
                    result_df = df[df[filter_col] == filter_value]
                elif condition == "ne":
                    result_df = df[df[filter_col] != filter_value]
                elif condition == "gt":
                    result_df = df[df[filter_col] > filter_value]
                elif condition == "gte":
                    result_df = df[df[filter_col] >= filter_value]
                elif condition == "lt":
                    result_df = df[df[filter_col] < filter_value]
                elif condition == "lte":
                    result_df = df[df[filter_col] <= filter_value]
                elif condition == "contains":
                    result_df = df[df[filter_col].astype(str).str.contains(str(filter_value), na=False)]

            message = f"Filtered to {len(result_df)} rows where {filter_col} {condition} {filter_value}"

        elif tool == "sort":
            sort_col = config.get("sort_col")
            order = config.get("order", "asc")

            result_df = df.sort_values(by=sort_col, ascending=(order == "asc")).reset_index(drop=True)
            message = f"Sorted by {sort_col} ({order}ending)"

        elif tool == "transpose":
            result_df = df.set_index(df.columns[0]).T.reset_index()
            result_df.columns = ['Variable'] + list(result_df.columns[1:])
            message = f"Transposed: {len(result_df)} rows × {len(result_df.columns)} columns"

        elif tool == "stack":
            operation = config.get("operation", "melt")

            if operation == "melt":
                id_cols = config.get("id_cols", [])
                if id_cols:
                    result_df = df.melt(id_vars=id_cols)
                else:
                    result_df = df.melt()
                message = f"Unpivoted to {len(result_df)} rows"

            elif operation == "pivot":
                index_col = config.get("index")
                pivot_col = config.get("pivot_col")
                values_col = config.get("values")

                result_df = df.pivot(index=index_col, columns=pivot_col, values=values_col).reset_index()
                result_df.columns.name = None
                message = f"Pivoted to {len(result_df)} rows × {len(result_df.columns)} columns"

        else:
            return JsonResponse({"error": f"Unknown tool: {tool}"}, status=400)

        # Save transformed data
        new_data_id = f"data_{uuid.uuid4().hex[:12]}"
        try:
            data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
            data_dir.mkdir(parents=True, exist_ok=True)
            data_path = data_dir / f"{new_data_id}.csv"
            result_df.to_csv(data_path, index=False)
        except Exception:
            data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
            data_dir.mkdir(parents=True, exist_ok=True)
            data_path = data_dir / f"{new_data_id}.csv"
            result_df.to_csv(data_path, index=False)

        # Build column info
        columns = []
        for col in result_df.columns:
            dtype = result_df[col].dtype
            if np.issubdtype(dtype, np.number):
                col_type = "numeric"
            elif np.issubdtype(dtype, np.datetime64):
                col_type = "datetime"
            else:
                col_type = "text"
            columns.append({"name": col, "dtype": col_type})

        return JsonResponse({
            "data_id": new_data_id,
            "filename": f"{tool}_{new_data_id[:8]}",
            "rows": len(result_df),
            "columns": columns,
            "preview": result_df.head(100).to_dict(orient="records"),
            "message": message
        })

    except Exception as e:
        logger.exception(f"Transform error: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@require_auth
def download_data(request):
    """
    Download current data as CSV.
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    data_id = body.get("data_id")

    if not data_id:
        return JsonResponse({"error": "No data_id provided"}, status=400)

    try:
        import pandas as pd
        from io import StringIO
        from django.http import HttpResponse

        df = None

        if data_id.startswith("data_"):
            try:
                data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path)
            except Exception:
                pass

            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = pd.read_csv(data_path)
                except Exception:
                    pass

        if df is None:
            try:
                from .models import TriageResult
                triage_result = TriageResult.objects.get(id=data_id, user=request.user)
                df = pd.read_csv(StringIO(triage_result.cleaned_csv))
            except Exception:
                pass

        if df is None:
            return JsonResponse({"error": "Data not found"}, status=404)

        # Return as CSV
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{data_id}.csv"'
        df.to_csv(response, index=False)
        return response

    except Exception as e:
        logger.exception(f"Download error: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@gated
def triage_data(request):
    """
    Run Triage (data cleaning) on loaded data.

    Uses the scrub module to clean Excel errors, missing values, etc.
    Returns cleaned data as a new dataset.
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    data_id = body.get("data_id")

    if not data_id:
        return JsonResponse({"error": "No data_id provided"}, status=400)

    try:
        import pandas as pd
        import numpy as np
        from io import StringIO
        import sys
        sys.path.insert(0, "/home/eric/kjerne/services")
        from scrub import DataCleaner, CleaningConfig

        # Load the data
        df = None

        if data_id.startswith("data_"):
            try:
                data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path)
            except Exception:
                pass

            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = pd.read_csv(data_path)
                except Exception:
                    pass

        if df is None:
            try:
                from .models import TriageResult
                triage_result = TriageResult.objects.get(id=data_id, user=request.user)
                df = pd.read_csv(StringIO(triage_result.cleaned_csv))
            except Exception:
                pass

        if df is None:
            return JsonResponse({"error": "Data not found"}, status=404)

        # Get user options
        options = body.get("options", {})
        fix_excel = options.get("fix_excel", True)
        fix_missing = options.get("fix_missing", False)
        missing_action = options.get("missing_action", "impute_mean")
        fix_outliers = options.get("fix_outliers", False)
        outlier_action = options.get("outlier_action", "flag")
        fix_types = options.get("fix_types", False)

        df_clean = df.copy()
        changes = {
            "excel_errors": 0,
            "missing_filled": 0,
            "missing_dropped": 0,
            "outliers_flagged": 0,
            "outliers_removed": 0,
            "outliers_clipped": 0,
            "types_converted": 0,
        }
        original_rows = len(df_clean)

        # 1. Fix Excel errors
        if fix_excel:
            excel_errors = ['#NUM!', '#DIV/0!', '#VALUE!', '#REF!', '#NAME?', '#N/A', '#NULL!', '#ERROR!']
            for col in df_clean.columns:
                if df_clean[col].dtype == object:
                    mask = df_clean[col].astype(str).str.upper().isin(excel_errors)
                    count = mask.sum()
                    if count > 0:
                        df_clean.loc[mask, col] = np.nan
                        changes["excel_errors"] += count

        # 2. Fix types (before missing/outlier handling)
        if fix_types:
            for col in df_clean.columns:
                if df_clean[col].dtype == object:
                    # Try to convert to numeric
                    try:
                        converted = pd.to_numeric(df_clean[col].str.replace(',', ''), errors='coerce')
                        # Only convert if >80% success
                        success_rate = converted.notna().sum() / max(df_clean[col].notna().sum(), 1)
                        if success_rate > 0.8:
                            df_clean[col] = converted
                            changes["types_converted"] += 1
                    except Exception:
                        pass

        # 3. Handle missing values
        if fix_missing and missing_action != "leave":
            if missing_action == "drop_rows":
                before = len(df_clean)
                df_clean = df_clean.dropna()
                changes["missing_dropped"] = before - len(df_clean)
            elif missing_action in ["impute_mean", "impute_median"]:
                for col in df_clean.columns:
                    missing_count = df_clean[col].isna().sum()
                    if missing_count > 0:
                        if pd.api.types.is_numeric_dtype(df_clean[col]):
                            if missing_action == "impute_mean":
                                fill_val = df_clean[col].mean()
                            else:
                                fill_val = df_clean[col].median()
                            df_clean[col] = df_clean[col].fillna(fill_val)
                        else:
                            # Mode for categorical
                            mode_val = df_clean[col].mode()
                            if len(mode_val) > 0:
                                df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
                        changes["missing_filled"] += missing_count

        # 4. Handle outliers
        if fix_outliers:
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                q1 = df_clean[col].quantile(0.25)
                q3 = df_clean[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
                    outlier_count = outlier_mask.sum()

                    if outlier_count > 0:
                        if outlier_action == "flag":
                            df_clean[f"{col}_outlier"] = outlier_mask.astype(int)
                            changes["outliers_flagged"] += outlier_count
                        elif outlier_action == "remove":
                            df_clean = df_clean[~outlier_mask]
                            changes["outliers_removed"] += outlier_count
                        elif outlier_action == "clip":
                            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
                            changes["outliers_clipped"] += outlier_count

        # Save cleaned data
        new_data_id = f"data_{uuid.uuid4().hex[:8]}"
        data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / f"{new_data_id}.csv"
        df_clean.to_csv(data_path, index=False)

        # Build column info
        columns = []
        for col in df_clean.columns:
            dtype = df_clean[col].dtype
            if np.issubdtype(dtype, np.number):
                col_type = "numeric"
            elif np.issubdtype(dtype, np.datetime64):
                col_type = "datetime"
            else:
                col_type = "text"
            columns.append({"name": col, "dtype": col_type})

        # Generate preview - convert numpy types to Python native
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj) if not np.isnan(obj) else None
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        preview_df = df_clean.head(100).replace({np.nan: None})
        preview = []
        for _, row in preview_df.iterrows():
            preview.append({k: convert_numpy(v) for k, v in row.items()})

        # Convert changes to Python int
        changes_clean = {k: int(v) for k, v in changes.items()}

        return JsonResponse({
            "success": True,
            "data": {
                "id": new_data_id,
                "rows": len(df_clean),
                "columns": columns,
                "preview": preview,
            },
            "changes": changes_clean,
            "rows_removed": int(original_rows - len(df_clean)),
        })

    except Exception as e:
        logger.exception(f"Triage error: {e}")
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
@gated
def triage_scan(request):
    """
    Scan dataset for issues WITHOUT cleaning.

    Returns detailed report of:
    - Excel error values (#NUM!, #DIV/0!, etc.)
    - Missing values per column
    - Potential outliers
    - Type issues (strings that should be numbers, etc.)
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    data_id = body.get("data_id")

    if not data_id:
        return JsonResponse({"error": "No data_id provided"}, status=400)

    try:
        import pandas as pd
        import numpy as np
        from io import StringIO

        # Load the data
        df = None

        if data_id.startswith("data_"):
            try:
                data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = pd.read_csv(data_path)
            except Exception:
                pass

            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = pd.read_csv(data_path)
                except Exception:
                    pass

        if df is None:
            try:
                from .models import TriageResult
                triage_result = TriageResult.objects.get(id=data_id, user=request.user)
                df = pd.read_csv(StringIO(triage_result.cleaned_csv))
            except Exception:
                pass

        if df is None:
            return JsonResponse({"error": "Data not found"}, status=404)

        # Scan for issues
        issues = {
            "excel_errors": {},
            "missing": {},
            "outliers": {},
            "type_issues": [],
            "totals": {
                "excel_errors": 0,
                "missing": 0,
                "outliers": 0,
                "type_issues": 0,
            }
        }

        excel_errors = ['#NUM!', '#DIV/0!', '#VALUE!', '#REF!', '#NAME?', '#N/A', '#NULL!', '#ERROR!']

        for col in df.columns:
            # Check for Excel errors
            if df[col].dtype == object:
                error_mask = df[col].astype(str).str.upper().isin(excel_errors)
                error_count = error_mask.sum()
                if error_count > 0:
                    issues["excel_errors"][col] = int(error_count)
                    issues["totals"]["excel_errors"] += int(error_count)

            # Check for missing values
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues["missing"][col] = {
                    "count": int(missing_count),
                    "percent": round(float(missing_count / len(df) * 100), 1)
                }
                issues["totals"]["missing"] += int(missing_count)

            # Check for outliers (numeric columns only)
            if pd.api.types.is_numeric_dtype(df[col]):
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
                    if outlier_count > 0:
                        issues["outliers"][col] = {
                            "count": int(outlier_count),
                            "percent": round(float(outlier_count / len(df) * 100), 1),
                            "range": f"{lower:.2f} - {upper:.2f}"
                        }
                        issues["totals"]["outliers"] += int(outlier_count)

            # Check for type issues (strings that look like numbers)
            if df[col].dtype == object:
                sample = df[col].dropna().head(100)
                numeric_count = 0
                for val in sample:
                    try:
                        float(str(val).replace(',', ''))
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass
                if len(sample) > 0 and numeric_count / len(sample) > 0.8:
                    issues["type_issues"].append({
                        "column": col,
                        "current": "text",
                        "suggested": "numeric",
                        "confidence": round(numeric_count / len(sample) * 100, 1)
                    })
                    issues["totals"]["type_issues"] += 1

        # Determine if data has issues
        has_issues = (
            issues["totals"]["excel_errors"] > 0 or
            issues["totals"]["missing"] > 0 or
            issues["totals"]["outliers"] > 0 or
            issues["totals"]["type_issues"] > 0
        )

        return JsonResponse({
            "success": True,
            "has_issues": has_issues,
            "issues": issues,
            "rows": len(df),
            "columns": len(df.columns),
        })

    except Exception as e:
        logger.exception(f"Triage scan error: {e}")
        return JsonResponse({"error": str(e)}, status=500)

"""DSW Data/Code/Assistant HTTP endpoints."""

import json
import logging
import re
import tempfile
import uuid
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

from accounts.permissions import gated, require_auth, require_enterprise

from .common import _preload_llm_background, log_agent_action

# Validate data_id to prevent path traversal (must be data_ + alphanumeric)
_SAFE_DATA_ID = re.compile(r"^data_[a-f0-9]+$")


def _validate_data_id(data_id: str) -> bool:
    """Return True if data_id is safe (no path traversal)."""
    return bool(data_id and _SAFE_DATA_ID.match(data_id))


logger = logging.getLogger(__name__)


def _read_csv_safe(file_or_path):
    """Read CSV with encoding fallback: UTF-8 → latin-1."""
    import io

    import pandas as pd

    if hasattr(file_or_path, "read"):
        raw = file_or_path.read()
        try:
            return pd.read_csv(io.BytesIO(raw), encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    try:
        return pd.read_csv(file_or_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_or_path, encoding="latin-1")


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

    # Limit upload size to 50 MB to prevent OOM
    MAX_UPLOAD_BYTES = 50 * 1024 * 1024
    if file.size and file.size > MAX_UPLOAD_BYTES:
        return JsonResponse(
            {
                "error": f"File too large ({file.size // (1024 * 1024)} MB). Maximum is 50 MB."
            },
            status=413,
        )

    filename = file.name.lower()

    try:
        import numpy as np
        import pandas as pd

        # Parse the file - try to detect actual format
        df = None
        parse_errors = []

        # Try based on extension first
        if filename.endswith(".csv"):
            try:
                df = _read_csv_safe(file)
            except Exception as e:
                parse_errors.append(f"CSV: {e}")

        elif filename.endswith(".xlsx"):
            try:
                df = pd.read_excel(file, engine="openpyxl")
            except Exception as e:
                parse_errors.append(f"XLSX: {e}")
                # Maybe it's actually a CSV with wrong extension
                file.seek(0)
                try:
                    df = _read_csv_safe(file)
                    parse_errors.append("(Parsed as CSV)")
                except Exception:
                    pass

        elif filename.endswith(".xls"):
            try:
                df = pd.read_excel(file, engine="xlrd")
            except Exception as e:
                parse_errors.append(f"XLS: {e}")
                # Maybe it's a CSV or XLSX with wrong extension
                file.seek(0)
                try:
                    df = _read_csv_safe(file)
                    parse_errors.append("(Parsed as CSV)")
                except Exception:
                    file.seek(0)
                    try:
                        df = pd.read_excel(file, engine="openpyxl")
                        parse_errors.append("(Parsed as XLSX)")
                    except Exception:
                        pass
        else:
            # Unknown extension - try all formats
            for parser, name in [
                (_read_csv_safe, "CSV"),
                (lambda f: pd.read_excel(f, engine="openpyxl"), "XLSX"),
                (lambda f: pd.read_excel(f, engine="xlrd"), "XLS"),
            ]:
                try:
                    file.seek(0)
                    df = parser(file)
                    break
                except Exception:
                    continue

        if df is None:
            return JsonResponse(
                {
                    "error": f"Could not parse file. Tried: {'; '.join(parse_errors) or 'all formats'}"
                },
                status=400,
            )

        # Save to temp storage for session use
        data_id = f"data_{uuid.uuid4().hex[:12]}"
        try:
            data_dir = (
                Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
            )
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

            columns.append(
                {
                    "name": col,
                    "dtype": col_type,
                }
            )

        # Generate preview (first 100 rows)
        preview = df.head(100).replace({np.nan: None}).to_dict(orient="records")

        logger.info(
            f"Data uploaded: {request.user.username} - {file.name} ({df.shape[0]} rows, {df.shape[1]} cols)"
        )

        # Preload LLM in background so it's ready when user asks questions
        _preload_llm_background()

        return JsonResponse(
            {
                "id": data_id,
                "filename": file.name,
                "rows": df.shape[0],
                "columns": columns,
                "preview": preview,
            }
        )

    except Exception as e:
        logger.exception(f"Data upload error: {e}")
        return JsonResponse(
            {
                "error": "Failed to parse uploaded file. Please check the file format and try again."
            },
            status=400,
        )


@require_http_methods(["POST"])
@require_auth
def retrieve_data(request):
    """
    Retrieve a previously uploaded dataset by data_id.

    Returns column info and preview in the same format as upload_data,
    so the frontend can restore the data table when loading a saved session.
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    data_id = body.get("data_id")
    if not data_id or not _validate_data_id(data_id):
        return JsonResponse({"error": "Invalid or missing data_id"}, status=400)

    try:
        import numpy as np
        import pandas as pd

        df = None

        # Try MEDIA_ROOT first
        data_dir = Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
        data_path = data_dir / f"{data_id}.csv"
        if data_path.exists():
            df = _read_csv_safe(data_path)

        # Fallback to temp directory
        if df is None:
            data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
            data_path = data_dir / f"{data_id}.csv"
            if data_path.exists():
                df = _read_csv_safe(data_path)

        # Fallback to TriageResult
        if df is None:
            try:
                from io import StringIO

                from ..models import TriageResult

                triage_result = TriageResult.objects.get(id=data_id, user=request.user)
                df = pd.read_csv(StringIO(triage_result.cleaned_csv))
            except Exception:
                pass

        if df is None:
            return JsonResponse({"error": "Dataset not found"}, status=404)

        # Build column info
        columns = []
        for col in df.columns:
            dtype = df[col].dtype
            if np.issubdtype(dtype, np.number):
                col_type = "numeric"
            elif np.issubdtype(dtype, np.datetime64):
                col_type = "datetime"
            else:
                col_type = "text"
            columns.append({"name": col, "dtype": col_type})

        # Preview as dict-of-arrays (matches SPC/displayDataTable format)
        preview_df = df.head(100).replace({np.nan: None})
        preview = {col: list(preview_df[col]) for col in preview_df.columns}

        return JsonResponse(
            {
                "id": data_id,
                "filename": body.get("filename", "dataset"),
                "row_count": df.shape[0],
                "columns": columns,
                "preview": preview,
            }
        )

    except Exception as e:
        logger.exception(f"Data retrieve error: {e}")
        return JsonResponse({"error": "Failed to retrieve dataset"}, status=500)


@require_http_methods(["POST"])
@gated
def execute_code(request):
    """
    Execute Python code in a sandboxed environment.

    DISABLED: The exec()-based sandbox was bypassable via module attribute
    chains (e.g. pd.__builtins__['__import__']('os').system(...)).
    This endpoint is disabled until a proper container-based sandbox is
    implemented. See DEBT.md.
    """
    return JsonResponse(
        {
            "error": "Code execution is temporarily disabled for security hardening. Use the built-in analysis tools instead."
        },
        status=403,
    )

    # --- DEAD CODE BELOW — kept for reference during sandbox reimplementation ---
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    code = body.get("code", "")
    data_id = body.get("data_id")

    if not code.strip():
        return JsonResponse({"error": "No code provided"}, status=400)

    try:
        import sys
        from io import StringIO

        import numpy as np
        import pandas as pd

        # Load data if provided
        df = None
        if data_id:
            from files.models import UploadedFile

            try:
                file_record = UploadedFile.objects.get(id=data_id, user=request.user)
                df = (
                    _read_csv_safe(file_record.file.path)
                    if file_record.file.path.endswith(".csv")
                    else pd.read_excel(file_record.file.path)
                )
            except UploadedFile.DoesNotExist:
                pass

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        # Import additional libraries for the sandbox
        import matplotlib
        import scipy
        import scipy.stats

        matplotlib.use("Agg")  # Non-interactive backend
        import math
        import random
        import statistics

        import matplotlib.pyplot as plt

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
            },
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
                plots.append(
                    {"title": "Output", "data": fig.data, "layout": fig.layout}
                )
            except Exception:
                pass

        return JsonResponse(
            {
                "output": output or "Code executed successfully",
                "plots": plots,
            }
        )

    except Exception as e:
        logger.exception(f"Code execution error: {e}")
        return JsonResponse(
            {"error": "Code execution failed. Please check your inputs."}, status=400
        )


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

    prompt = body.get("prompt", "").strip()[:2000]
    if not prompt:
        return JsonResponse({"error": "Prompt is required"}, status=400)

    model = body.get("model", "qwen")
    context = body.get("context", {})

    # Build context prefix with XML delimiters
    context_prefix = ""
    if context.get("hypothesis"):
        context_prefix = f"<context>Testing hypothesis: {context['hypothesis'][:1000]}"
        if context.get("mechanism"):
            context_prefix += f"\nMechanism: {context['mechanism'][:1000]}"
        context_prefix += "</context>\n"

    # Check if using Anthropic models (Enterprise only)
    if model in ("sonnet", "opus", "haiku"):
        # Check enterprise access
        if (
            not hasattr(request.user, "subscription")
            or request.user.subscription.plan != "enterprise"
        ):
            return JsonResponse(
                {"error": "Anthropic models require Enterprise subscription"},
                status=403,
            )

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
                messages=[
                    {
                        "role": "user",
                        "content": f"{context_prefix}<request>{prompt}</request>",
                    }
                ],
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
            return JsonResponse(
                {"error": "Code generation failed. Please try again."}, status=500
            )

    # Default: Use Qwen Coder
    code_prompt = f"""{context_prefix}<request>{prompt}</request>

Generate Python code for the request above.

Rules:
- Only output Python code, no explanations
- Use numpy, pandas, scipy as needed
- Include print() statements for results
- Keep code concise but complete"""

    try:
        # Get Qwen Coder LLM
        from .. import views as agent_views

        llm = agent_views.get_coder_llm()

        if llm is None:
            # Fallback: return a template
            code = """import numpy as np
import pandas as pd

# Qwen Coder is loading, please try again in a moment
"""
            return JsonResponse(
                {"code": code, "note": "Qwen Coder is loading, try again shortly"}
            )

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
            code = code[len(code_prompt) :].strip()

        return JsonResponse({"code": code.strip(), "model": "qwen"})

    except Exception as e:
        logger.exception(f"Code generation error: {e}")
        return JsonResponse(
            {"error": "Code generation service unavailable. Please try again later."},
            status=500,
        )


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
                from io import StringIO

                import pandas as pd

                if data_id and _validate_data_id(data_id):
                    data_dir = (
                        Path(settings.MEDIA_ROOT)
                        / "analysis_data"
                        / str(request.user.id)
                    )
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = _read_csv_safe(data_path)
                    else:
                        data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                        data_path = data_dir / f"{data_id}.csv"
                        if data_path.exists():
                            df = _read_csv_safe(data_path)
                else:
                    from ..models import TriageResult

                    try:
                        triage_result = TriageResult.objects.get(
                            id=data_id, user=request.user
                        )
                        df = pd.read_csv(StringIO(triage_result.cleaned_csv))
                    except Exception:
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
                from .. import views as agent_views

                # Check if LLM is already loaded - don't block on first load
                if agent_views._shared_llm_loaded:
                    llm = agent_views._shared_llm
                    logger.info(
                        f"LLM already loaded: type={type(llm).__name__ if llm else 'None'}"
                    )
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
            response, sources = generate_researcher_response(
                message, df, columns, data_preview
            )
            log_agent_action(request.user, "researcher", "research", success=True)
            return JsonResponse({"response": response, "sources": sources})

        elif agent_type == "writer":
            # Writer agent: generate downloadable documents
            response, document, filename = generate_writer_response(
                message, df, columns, session_history
            )
            log_agent_action(request.user, "writer", "write", success=True)
            return JsonResponse(
                {"response": response, "document": document, "filename": filename}
            )

        # Default: Analyst agent
        # Check for enterprise model selection (Opus/Sonnet/Haiku)
        if selected_model in ("opus", "sonnet", "haiku"):
            try:
                response = generate_anthropic_response(
                    selected_model, message, df, columns, session_history
                )
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
            response = (
                "*(Qwen LLM is loading in the background - using quick response mode)*\n\n"
                + response
            )
        elif llm is None:
            logger.info("Using keyword-based fallback response")
            response = generate_analyst_response(message.lower(), df, columns)
        else:
            logger.info(f"Using LLM for response, message: {message[:50]}...")
            try:
                response = generate_llm_response(
                    llm, message, df, columns, session_history
                )
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
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        data_context = f"""Dataset: {n_rows:,} rows × {n_cols} columns
Numeric columns: {", ".join(numeric_cols[:10])}
Categorical columns: {", ".join(cat_cols[:10])}

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
        messages=[{"role": "user", "content": message}],
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
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Build correlation matrix for numeric columns (helps answer relationship questions)
    corr_info = ""
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr()
            # Find top correlations
            corr_pairs = []
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1 :]:
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.3:  # Only notable correlations
                        corr_pairs.append((col1, col2, corr_val))
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            if corr_pairs:
                corr_info = "\n\nNOTABLE CORRELATIONS:\n"
                for col1, col2, r in corr_pairs[:10]:
                    strength = (
                        "strong"
                        if abs(r) > 0.7
                        else "moderate" if abs(r) > 0.5 else "weak"
                    )
                    direction = "positive" if r > 0 else "negative"
                    corr_info += (
                        f"- {col1} ↔ {col2}: r={r:.3f} ({strength} {direction})\n"
                    )
        except Exception:
            pass

    # Build data summary for context
    data_context = f"""Dataset: {n_rows:,} rows × {n_cols} columns

Numeric columns ({len(numeric_cols)}): {", ".join(numeric_cols[:15])}
Categorical columns ({len(cat_cols)}): {", ".join(cat_cols[:15])}

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
            future = executor.submit(
                llm.generate, prompt, max_tokens=500, temperature=0.7
            )
            try:
                response = future.result(timeout=30)  # 30 second timeout
                return response
            except concurrent.futures.TimeoutError:
                logger.warning("LLM generation timed out after 30s")
                return (
                    generate_analyst_response(message.lower(), df, columns)
                    + "\n\n*(Response generated via quick mode due to timeout)*"
                )
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
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Keywords for different intents
    if any(
        w in message
        for w in ["describe", "summary", "overview", "tell me about", "what is"]
    ):
        summary = f"Your dataset has {n_rows:,} rows and {n_cols} columns.\n\n"
        summary += (
            f"**Numeric columns ({len(numeric_cols)}):** {', '.join(numeric_cols[:5])}"
        )
        if len(numeric_cols) > 5:
            summary += f" (+{len(numeric_cols) - 5} more)"
        summary += (
            f"\n\n**Categorical columns ({len(cat_cols)}):** {', '.join(cat_cols[:5])}"
        )
        if len(cat_cols) > 5:
            summary += f" (+{len(cat_cols) - 5} more)"

        if numeric_cols:
            summary += "\n\n**Quick stats for numeric columns:**\n"
            for col in numeric_cols[:3]:
                summary += (
                    f"- {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}\n"
                )

        summary += "\n\nUse **Stat > Descriptive Statistics** for detailed analysis."
        return summary

    if any(
        w in message for w in ["correlation", "relationship", "related", "correlated"]
    ):
        if len(numeric_cols) < 2:
            return "You need at least 2 numeric columns to analyze correlations."

        corr_matrix = df[numeric_cols].corr()
        # Find strongest correlations
        pairs = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                pairs.append((col1, col2, corr_matrix.loc[col1, col2]))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        response = "**Correlation Analysis:**\n\n"
        response += "Strongest relationships:\n"
        for col1, col2, corr in pairs[:5]:
            strength = (
                "strong"
                if abs(corr) > 0.7
                else "moderate" if abs(corr) > 0.4 else "weak"
            )
            direction = "positive" if corr > 0 else "negative"
            response += f"- {col1} & {col2}: r={corr:.3f} ({strength} {direction})\n"

        response += (
            "\n\nTo visualize, use **Graph > Scatterplot** with these variable pairs."
        )
        return response

    if any(
        w in message for w in ["predict", "forecast", "ml", "machine learning", "model"]
    ):
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

    if any(w in message for w in ["outlier", "anomaly", "unusual", "extreme"]):
        if not numeric_cols:
            return "No numeric columns found for outlier detection."

        response = "**Outlier Detection:**\n\n"
        for col in numeric_cols[:4]:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)][col]
            if len(outliers) > 0:
                response += f"- **{col}**: {len(outliers)} outliers ({len(outliers) / len(df) * 100:.1f}%)\n"

        response += "\n\nUse **Graph > Boxplot** to visualize outliers, or use **Triage** to clean them."
        return response

    if any(w in message for w in ["missing", "null", "empty", "na"]):
        response = "**Missing Data Analysis:**\n\n"
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) == 0:
            response += "No missing values found in your dataset."
        else:
            response += "Columns with missing values:\n"
            for col, count in missing.head(10).items():
                response += f"- **{col}**: {count} ({count / len(df) * 100:.1f}%)\n"

            response += (
                "\n\nUse **Triage** to handle missing values (imputation, removal)."
            )
        return response

    if any(w in message for w in ["compare", "difference", "group", "between"]):
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

    if any(w in message for w in ["distribution", "normal", "spread", "histogram"]):
        if not numeric_cols:
            return "No numeric columns found for distribution analysis."

        response = "**Distribution Analysis:**\n\n"
        for col in numeric_cols[:3]:
            skew = df[col].skew()
            skew_desc = (
                "right-skewed"
                if skew > 0.5
                else "left-skewed" if skew < -0.5 else "approximately symmetric"
            )
            response += f"- **{col}**: {skew_desc} (skewness={skew:.2f})\n"

        response += "\n\n**To visualize:**\n"
        response += "- **Graph > Histogram** for distribution shape\n"
        response += "- **Stat > Normality Test** to test for normality"
        return response

    # Default response
    response = (
        f"I can help you analyze your dataset ({n_rows:,} rows, {n_cols} columns).\n\n"
    )
    response += "**Try asking about:**\n"
    response += '- "Describe my data" - Get an overview\n'
    response += '- "Find correlations" - Discover relationships\n'
    response += '- "Check for outliers" - Find unusual values\n'
    response += '- "Missing data" - Analyze gaps\n'
    response += '- "How to predict X" - ML recommendations\n'
    response += '- "Compare groups" - Statistical comparisons\n\n'
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
        data_context = (
            f"Dataset has {len(df)} rows with columns: {', '.join(columns[:10])}"
        )
        if data_preview:
            sample_vals = []
            for col in columns[:5]:
                if col in df.columns:
                    vals = df[col].dropna().head(3).tolist()
                    sample_vals.append(f"{col}: {vals}")
            data_context += f"\nSample values: {'; '.join(sample_vals[:3])}"

    # Extract key technical terms from the question
    # Remove common question words
    query_clean = re.sub(
        r"\b(can you|could you|please|research|tell me about|what is the|why do|why does|how does|explain|relationship between|correlation between)\b",
        "",
        message.lower(),
    )
    query_clean = query_clean.strip()

    # Extract potential chemical/technical terms (capitalized words or known patterns)
    technical_terms = re.findall(r"\b[A-Za-z]{4,}\b", message)
    # Filter to likely technical terms (not common words)
    common_words = {
        "what",
        "that",
        "this",
        "with",
        "from",
        "have",
        "been",
        "were",
        "they",
        "their",
        "about",
        "which",
        "when",
        "there",
        "would",
        "could",
        "should",
        "between",
        "relationship",
        "correlation",
        "drinking",
        "water",
        "samples",
    }
    technical_terms = [
        t.lower() for t in technical_terms if t.lower() not in common_words
    ]

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
            term_combo = " ".join(technical_terms[:3])
            searches.append(
                f'"{technical_terms[0]}" "{technical_terms[1]}" correlation co-occurrence site:epa.gov OR site:pubmed OR site:ncbi.nlm.nih.gov'
            )
            searches.append(f"{term_combo} water contamination research")
            searches.append(
                f'"{technical_terms[0]}" "{technical_terms[1]}" drinking water study'
            )
        else:
            # Fallback to cleaned query
            searches.append(f"{query_clean} EPA research")
            searches.append(f"{query_clean} scientific study")

        seen_urls = set()
        for search_query in searches:
            try:
                results = list(ddgs.text(search_query, max_results=3))
                for r in results:
                    url = r.get("href", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        search_results.append(
                            {
                                "title": r.get("title", "Result"),
                                "snippet": r.get("body", ""),
                                "url": url,
                            }
                        )
                        sources.append(
                            {"title": r.get("title", "Source")[:60], "url": url}
                        )
            except Exception as e:
                logger.warning(f"Search query failed: {e}")
                continue

    except ImportError:
        logger.warning("ddgs not installed, trying fallback")
        # Fallback to requests-based search
        try:
            import urllib.parse

            import requests
            from bs4 import BeautifulSoup

            # Use DuckDuckGo HTML search
            search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query_clean + ' scientific')}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            resp = requests.get(search_url, headers=headers, timeout=15)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for result in soup.select(".result")[:5]:
                    title_el = result.select_one(".result__title")
                    snippet_el = result.select_one(".result__snippet")
                    link_el = result.select_one(".result__url")

                    if title_el and snippet_el:
                        title = title_el.get_text(strip=True)
                        snippet = snippet_el.get_text(strip=True)
                        url = link_el.get("href", "") if link_el else ""

                        search_results.append(
                            {"title": title, "snippet": snippet, "url": url}
                        )
                        if url:
                            sources.append({"title": title[:60], "url": url})

        except Exception as e:
            logger.warning(f"Fallback search failed: {e}")

    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")

    # Use LLM to synthesize search results into intelligent response
    if search_results:
        # Try to use shared LLM for synthesis
        llm = None
        try:
            from .. import views as agent_views

            if agent_views._shared_llm_loaded:
                llm = agent_views._shared_llm
        except Exception:
            pass

        if llm:
            # Build context for LLM synthesis
            search_context = "\n\n".join(
                [
                    f"Source: {r['title']}\n{r['snippet']}"
                    for r in search_results[:5]
                    if r.get("snippet")
                ]
            )

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
                    future = executor.submit(
                        llm.generate, synthesis_prompt, max_tokens=400, temperature=0.7
                    )
                    synthesized = future.result(timeout=30)

                response = f"**Research Analysis:** *{message}*\n\n"
                response += synthesized + "\n\n"

                if sources:
                    response += "**Sources:**\n"
                    for src in sources[:5]:
                        if src.get("url"):
                            response += (
                                f"- [{src.get('title', 'Link')[:50]}]({src['url']})\n"
                            )

                return response, sources

            except Exception as e:
                logger.warning(f"LLM synthesis failed: {e}")
                # Fall through to basic response

        # Fallback: basic response without LLM
        response = f"**Research findings for:** *{message}*\n\n"

        for i, result in enumerate(search_results[:4], 1):
            if result["snippet"]:
                response += f"**{result.get('title', f'Finding {i}')}**\n"
                response += f"{result['snippet']}\n\n"

        if data_context:
            response += f"---\n**Your data context:** {data_context}\n\n"

        if sources:
            response += "**Sources:**\n"
            for src in sources[:5]:
                if src.get("url"):
                    response += f"- [{src.get('title', 'Link')[:50]}]({src['url']})\n"
    else:
        response = f"I searched for information about *{message}* but didn't find specific results.\n\n"
        response += "**Suggestions:**\n"
        response += "- Try rephrasing your question with more specific terms\n"
        response += (
            "- Ask about specific chemicals, processes, or scientific concepts\n"
        )
        response += "- Use the Analyst agent for data-specific questions\n"

        if data_context:
            response += f"\n**Your data context:** {data_context}"

    return response, sources


def generate_writer_response(message, df, columns, session_history):
    """
    Writer agent: generates downloadable documents/reports using LLM.
    Creates intelligent markdown documents with analysis and insights.
    """
    from datetime import datetime

    import numpy as np

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

    if df is None:
        document = "# Analysis Report\n\nNo data loaded. Please load a dataset first.\n"
        return "Please load a dataset first to generate a document.", document, filename

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Build data context for LLM
    stats_summary = df.describe().to_string() if numeric_cols else "No numeric columns"

    # Session history context
    history_text = ""
    if session_history:
        history_text = "\n".join(
            [
                f"- {item.get('name', 'Analysis')}: {item.get('summary', '')[:150]}"
                for item in session_history[-10:]
            ]
        )

    # Try to use LLM for intelligent document generation
    llm = None
    try:
        from .. import views as agent_views

        if agent_views._shared_llm_loaded:
            llm = agent_views._shared_llm
    except Exception:
        pass

    if llm:
        writer_prompt = f"""You are a technical writer creating a data analysis report. The user requested: "{message}"

DATA CONTEXT:
- Dataset: {n_rows:,} rows × {n_cols} columns
- Numeric columns ({len(numeric_cols)}): {", ".join(numeric_cols[:10])}
- Categorical columns ({len(cat_cols)}): {", ".join(cat_cols[:10])}

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
                future = executor.submit(
                    llm.generate, writer_prompt, max_tokens=800, temperature=0.7
                )
                llm_content = future.result(timeout=45)

            document = "# Analysis Report\n\n"
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
    document = "# Data Analysis Report\n\n"
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

    response = "I've prepared a report document for your analysis.\n\n"
    response += "**Document includes:**\n"
    response += f"- Data overview ({n_rows:,} rows × {n_cols} columns)\n"
    response += "- Variable listing with types and missing values\n"
    if numeric_cols:
        response += f"- Summary statistics for {len(numeric_cols)} numeric variables\n"
    response += "\nClick the download link below to save."

    return response, document, filename


# ============================================================================
# DATA TRANSFORMATION TOOLS
# ============================================================================


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
        from io import StringIO

        import numpy as np
        import pandas as pd

        # Load data
        df = None

        if data_id and _validate_data_id(data_id):
            try:
                data_dir = (
                    Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                )
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = _read_csv_safe(data_path)
            except Exception:
                pass

            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = _read_csv_safe(data_path)
                except Exception:
                    pass

        if df is None:
            try:
                from ..models import TriageResult

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
                return JsonResponse(
                    {"error": "Column name and expression required"}, status=400
                )

            # Safe evaluation using pandas.eval (no arbitrary code execution)
            try:
                result_df[new_col] = pd.eval(
                    expression, local_dict={"df": df}, engine="numexpr"
                )
                message = f"Created column '{new_col}'"
            except Exception as e:
                return JsonResponse(
                    {"error": f"Expression error: {str(e)}"}, status=400
                )

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
                    if df[filter_col].dtype in ["int64", "float64"]:
                        filter_value = float(filter_value)
                except Exception:
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
                    result_df = df[
                        df[filter_col]
                        .astype(str)
                        .str.contains(str(filter_value), na=False)
                    ]

            message = f"Filtered to {len(result_df)} rows where {filter_col} {condition} {filter_value}"

        elif tool == "sort":
            sort_col = config.get("sort_col")
            order = config.get("order", "asc")

            result_df = df.sort_values(
                by=sort_col, ascending=(order == "asc")
            ).reset_index(drop=True)
            message = f"Sorted by {sort_col} ({order}ending)"

        elif tool == "transpose":
            result_df = df.set_index(df.columns[0]).T.reset_index()
            result_df.columns = ["Variable"] + list(result_df.columns[1:])
            message = (
                f"Transposed: {len(result_df)} rows × {len(result_df.columns)} columns"
            )

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

                result_df = df.pivot(
                    index=index_col, columns=pivot_col, values=values_col
                ).reset_index()
                result_df.columns.name = None
                message = f"Pivoted to {len(result_df)} rows × {len(result_df.columns)} columns"

        elif tool == "encode":
            columns = config.get("columns", [])
            method = config.get("method", "onehot")
            drop_first = config.get("drop_first", False)

            if not columns:
                return JsonResponse({"error": "Select columns to encode"}, status=400)
            columns = [c for c in columns if c in df.columns]

            if method == "onehot":
                result_df = pd.get_dummies(
                    result_df, columns=columns, drop_first=drop_first, dtype=int
                )
                n_new = len(result_df.columns) - len(df.columns)
                message = f"One-hot encoded {len(columns)} column(s) → {n_new} new dummy columns"
            elif method == "label":
                for col in columns:
                    cats = result_df[col].astype(str).unique()
                    cats.sort()
                    mapping = {v: i for i, v in enumerate(cats)}
                    result_df[col] = result_df[col].astype(str).map(mapping)
                message = f"Label encoded {len(columns)} column(s)"
            else:
                return JsonResponse(
                    {"error": f"Unknown encoding method: {method}"}, status=400
                )

        elif tool == "scale":
            columns = config.get("columns", [])
            method = config.get("method", "zscore")

            if not columns:
                return JsonResponse({"error": "Select columns to scale"}, status=400)
            columns = [
                c
                for c in columns
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]

            for col in columns:
                vals = result_df[col]
                if method == "zscore":
                    std = vals.std()
                    result_df[col] = (vals - vals.mean()) / std if std > 0 else 0
                elif method == "minmax":
                    mn, mx = vals.min(), vals.max()
                    result_df[col] = (vals - mn) / (mx - mn) if mx > mn else 0
                elif method == "robust":
                    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                    iqr = q3 - q1
                    result_df[col] = (vals - vals.median()) / iqr if iqr > 0 else 0
                else:
                    return JsonResponse(
                        {"error": f"Unknown scaling method: {method}"}, status=400
                    )

            method_names = {
                "zscore": "Z-score",
                "minmax": "Min-Max [0,1]",
                "robust": "Robust (IQR)",
            }
            message = (
                f"{method_names.get(method, method)} scaled {len(columns)} column(s)"
            )

        elif tool == "bin":
            column = config.get("column", "")
            method = config.get("method", "equal_width")
            n_bins = int(config.get("n_bins", 5))
            custom_bins = config.get("bins", [])
            labels = config.get("labels", [])

            if not column or column not in df.columns:
                return JsonResponse(
                    {"error": "Select a valid column to bin"}, status=400
                )

            new_col = f"{column}_binned"
            try:
                if method == "equal_width":
                    result_df[new_col] = pd.cut(
                        result_df[column], bins=n_bins, labels=labels or False
                    )
                elif method == "equal_frequency":
                    result_df[new_col] = pd.qcut(
                        result_df[column],
                        q=n_bins,
                        labels=labels or False,
                        duplicates="drop",
                    )
                elif method == "custom":
                    if len(custom_bins) < 2:
                        return JsonResponse(
                            {"error": "Provide at least 2 breakpoints"}, status=400
                        )
                    result_df[new_col] = pd.cut(
                        result_df[column],
                        bins=custom_bins,
                        labels=labels or False,
                        include_lowest=True,
                    )
                else:
                    return JsonResponse(
                        {"error": f"Unknown binning method: {method}"}, status=400
                    )

                result_df[new_col] = result_df[new_col].astype(str)
                message = f"Binned {column} into '{new_col}' ({method}, {n_bins if method != 'custom' else len(custom_bins) - 1} bins)"
            except Exception as e:
                return JsonResponse({"error": f"Binning error: {str(e)}"}, status=400)

        else:
            return JsonResponse({"error": f"Unknown tool: {tool}"}, status=400)

        # Save transformed data
        new_data_id = f"data_{uuid.uuid4().hex[:12]}"
        try:
            data_dir = (
                Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
            )
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

        return JsonResponse(
            {
                "data_id": new_data_id,
                "filename": f"{tool}_{new_data_id[:8]}",
                "rows": len(result_df),
                "columns": columns,
                "preview": result_df.head(100).to_dict(orient="records"),
                "message": message,
            }
        )

    except Exception as e:
        logger.exception(f"Transform error: {e}")
        return JsonResponse({"error": str(e)}, status=500)


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
        from io import StringIO

        import pandas as pd
        from django.http import HttpResponse

        df = None

        if data_id and _validate_data_id(data_id):
            try:
                data_dir = (
                    Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                )
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = _read_csv_safe(data_path)
            except Exception:
                pass

            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = _read_csv_safe(data_path)
                except Exception:
                    pass

        if df is None:
            try:
                from ..models import TriageResult

                triage_result = TriageResult.objects.get(id=data_id, user=request.user)
                df = pd.read_csv(StringIO(triage_result.cleaned_csv))
            except Exception:
                pass

        if df is None:
            return JsonResponse({"error": "Data not found"}, status=404)

        # Return as CSV
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="{data_id}.csv"'
        df.to_csv(response, index=False)
        return response

    except Exception as e:
        logger.exception(f"Download error: {e}")
        return JsonResponse({"error": "Download failed. Please try again."}, status=500)


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
        from io import StringIO

        import numpy as np
        import pandas as pd

        # Load the data
        df = None

        if data_id and _validate_data_id(data_id):
            try:
                data_dir = (
                    Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                )
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = _read_csv_safe(data_path)
            except Exception:
                pass

            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = _read_csv_safe(data_path)
                except Exception:
                    pass

        if df is None:
            try:
                from ..models import TriageResult

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
        warnings = []
        original_rows = len(df_clean)
        original_cols = len(df_clean.columns)

        # 1. Fix Excel errors
        if fix_excel:
            excel_errors = [
                "#NUM!",
                "#DIV/0!",
                "#VALUE!",
                "#REF!",
                "#NAME?",
                "#N/A",
                "#NULL!",
                "#ERROR!",
            ]
            for col in df_clean.columns:
                if df_clean[col].dtype == object:
                    mask = df_clean[col].astype(str).str.upper().isin(excel_errors)
                    count = mask.sum()
                    if count > 0:
                        df_clean.loc[mask, col] = np.nan
                        changes["excel_errors"] += int(count)

        # 2. Fix types (before missing/outlier handling)
        if fix_types:
            for col in df_clean.columns:
                if df_clean[col].dtype == object:
                    try:
                        converted = pd.to_numeric(
                            df_clean[col].str.replace(",", ""), errors="coerce"
                        )
                        non_null_original = df_clean[col].notna().sum()
                        non_null_converted = converted.notna().sum()
                        # Only convert if no values are lost (coerce didn't create new NaNs)
                        if (
                            non_null_original > 0
                            and non_null_converted == non_null_original
                        ):
                            df_clean[col] = converted
                            changes["types_converted"] += 1
                        elif (
                            non_null_original > 0
                            and non_null_converted / non_null_original > 0.95
                        ):
                            # >95% converted — apply but warn about lost values
                            lost = int(non_null_original - non_null_converted)
                            df_clean[col] = converted
                            changes["types_converted"] += 1
                            warnings.append(
                                f"Type conversion: '{col}' — {lost} non-numeric values became empty"
                            )
                    except Exception:
                        pass

        # 3. Handle missing values
        if fix_missing and missing_action != "leave":
            if missing_action == "drop_rows":
                # Only drop rows that are mostly empty (>80% missing), NOT rows with a single NaN
                n_cols = len(df_clean.columns)
                if n_cols > 0:
                    row_missing_pct = df_clean.isna().sum(axis=1) / n_cols
                    rows_to_drop = row_missing_pct[row_missing_pct > 0.8].index
                    if len(rows_to_drop) > 0:
                        changes["missing_dropped"] = len(rows_to_drop)
                        warnings.append(
                            f"Dropped {len(rows_to_drop)} rows with >80% missing values"
                        )
                        df_clean = df_clean.drop(rows_to_drop)

                    # Impute remaining missing values instead of dropping their rows
                    for col in df_clean.columns:
                        missing_count = int(df_clean[col].isna().sum())
                        if missing_count > 0:
                            if pd.api.types.is_numeric_dtype(df_clean[col]):
                                fill_val = df_clean[col].median()
                                if pd.notna(fill_val):
                                    if pd.api.types.is_integer_dtype(df_clean[col]):
                                        fill_val = round(fill_val)
                                    df_clean[col] = df_clean[col].fillna(fill_val)
                            else:
                                mode_val = df_clean[col].mode()
                                if len(mode_val) > 0:
                                    df_clean[col] = df_clean[col].fillna(
                                        mode_val.iloc[0]
                                    )
                            changes["missing_filled"] += missing_count
            elif missing_action in ["impute_mean", "impute_median"]:
                for col in df_clean.columns:
                    missing_count = int(df_clean[col].isna().sum())
                    if missing_count > 0:
                        if pd.api.types.is_numeric_dtype(df_clean[col]):
                            if missing_action == "impute_mean":
                                fill_val = df_clean[col].mean()
                            else:
                                fill_val = df_clean[col].median()
                            if pd.notna(fill_val):
                                if pd.api.types.is_integer_dtype(df_clean[col]):
                                    fill_val = round(fill_val)
                                df_clean[col] = df_clean[col].fillna(fill_val)
                        else:
                            mode_val = df_clean[col].mode()
                            if len(mode_val) > 0:
                                df_clean[col] = df_clean[col].fillna(mode_val.iloc[0])
                        changes["missing_filled"] += missing_count

                        # Warn about high-missing columns
                        missing_pct = missing_count / max(original_rows, 1) * 100
                        if missing_pct > 50:
                            warnings.append(
                                f"'{col}' has {missing_pct:.0f}% missing — imputed, but consider whether this column is reliable"
                            )

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
                    outlier_count = int(outlier_mask.sum())

                    if outlier_count > 0:
                        if outlier_action == "flag":
                            df_clean[f"{col}_outlier"] = outlier_mask.astype(int)
                            changes["outliers_flagged"] += outlier_count
                        elif outlier_action == "remove":
                            df_clean = df_clean[~outlier_mask]
                            changes["outliers_removed"] += outlier_count
                            warnings.append(
                                f"Removed {outlier_count} outlier rows from '{col}'"
                            )
                        elif outlier_action == "clip":
                            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
                            changes["outliers_clipped"] += outlier_count

        # Save cleaned data
        new_data_id = f"data_{uuid.uuid4().hex[:8]}"
        data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / f"{new_data_id}.csv"
        df_clean.to_csv(data_path, index=False)

        # Also produce CSV string for frontends that re-upload
        import io

        csv_buf = io.StringIO()
        df_clean.to_csv(csv_buf, index=False)
        cleaned_csv_str = csv_buf.getvalue()

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

        return JsonResponse(
            {
                "success": True,
                "data": {
                    "id": new_data_id,
                    "rows": len(df_clean),
                    "columns": columns,
                    "preview": preview,
                },
                "cleaned_csv": cleaned_csv_str,
                "changes": changes_clean,
                "rows_removed": int(original_rows - len(df_clean)),
                "cols_removed": int(original_cols - len(df_clean.columns)),
                "warnings": warnings,
            }
        )

    except Exception as e:
        logger.exception(f"Triage error: {e}")
        return JsonResponse({"error": str(e)}, status=500)


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
        from io import StringIO

        import pandas as pd

        # Load the data
        df = None

        if data_id and _validate_data_id(data_id):
            try:
                data_dir = (
                    Path(settings.MEDIA_ROOT) / "analysis_data" / str(request.user.id)
                )
                data_path = data_dir / f"{data_id}.csv"
                if data_path.exists():
                    df = _read_csv_safe(data_path)
            except Exception:
                pass

            if df is None:
                try:
                    data_dir = Path(tempfile.gettempdir()) / "svend_analysis"
                    data_path = data_dir / f"{data_id}.csv"
                    if data_path.exists():
                        df = _read_csv_safe(data_path)
                except Exception:
                    pass

        if df is None:
            try:
                from ..models import TriageResult

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
            },
        }

        excel_errors = [
            "#NUM!",
            "#DIV/0!",
            "#VALUE!",
            "#REF!",
            "#NAME?",
            "#N/A",
            "#NULL!",
            "#ERROR!",
        ]

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
                    "percent": round(float(missing_count / len(df) * 100), 1),
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
                            "range": f"{lower:.2f} - {upper:.2f}",
                        }
                        issues["totals"]["outliers"] += int(outlier_count)

            # Check for type issues (strings that look like numbers)
            if df[col].dtype == object:
                sample = df[col].dropna().head(100)
                numeric_count = 0
                for val in sample:
                    try:
                        float(str(val).replace(",", ""))
                        numeric_count += 1
                    except (ValueError, TypeError):
                        pass
                if len(sample) > 0 and numeric_count / len(sample) > 0.8:
                    issues["type_issues"].append(
                        {
                            "column": col,
                            "current": "text",
                            "suggested": "numeric",
                            "confidence": round(numeric_count / len(sample) * 100, 1),
                        }
                    )
                    issues["totals"]["type_issues"] += 1

        # Determine if data has issues
        has_issues = (
            issues["totals"]["excel_errors"] > 0
            or issues["totals"]["missing"] > 0
            or issues["totals"]["outliers"] > 0
            or issues["totals"]["type_issues"] > 0
        )

        return JsonResponse(
            {
                "success": True,
                "has_issues": has_issues,
                "issues": issues,
                "rows": len(df),
                "columns": len(df.columns),
            }
        )

    except Exception as e:
        logger.exception(f"Triage scan error: {e}")
        return JsonResponse({"error": str(e)}, status=500)

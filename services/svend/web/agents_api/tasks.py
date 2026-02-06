"""Tempora task handlers for LLM inference.

These tasks are queued and executed sequentially to avoid GPU contention.
"""

import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def llm_inference(
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    use_coder: bool = False,
    callback_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute LLM inference task.

    This runs in the Tempora worker, ensuring sequential GPU access.

    Args:
        prompt: The prompt to send to the LLM
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        use_coder: If True, use the coder LLM instead of base
        callback_url: Optional webhook to POST results to

    Returns:
        Dict with 'response', 'tokens_used', 'model', 'success'
    """
    from .views import get_shared_llm, get_coder_llm

    try:
        # Get the appropriate LLM
        if use_coder:
            llm = get_coder_llm()
            model_name = "qwen-coder"
        else:
            llm = get_shared_llm()
            model_name = "qwen-base"

        if llm is None:
            return {
                "success": False,
                "error": "LLM not available",
                "response": "",
                "model": model_name,
            }

        # Generate response
        logger.info(f"LLM inference: {len(prompt)} chars, max_tokens={max_tokens}")
        response = llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)

        result = {
            "success": True,
            "response": response,
            "model": model_name,
            "prompt_length": len(prompt),
            "response_length": len(response),
        }

        # Optional callback
        if callback_url:
            try:
                import requests
                requests.post(callback_url, json=result, timeout=5)
            except Exception as e:
                logger.warning(f"Callback failed: {e}")

        return result

    except Exception as e:
        logger.exception(f"LLM inference failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "response": "",
            "model": model_name if 'model_name' in dir() else "unknown",
        }


def analyst_inference(
    message: str,
    data_summary: str,
    session_history: str = "",
    max_tokens: int = 1000,
) -> Dict[str, Any]:
    """
    Analyst assistant inference with reasoning prompt.

    Args:
        message: User's question
        data_summary: Summary of the loaded dataset
        session_history: Previous analyses in this session
        max_tokens: Maximum response tokens

    Returns:
        Dict with response and metadata
    """
    # Build reasoning-enabled prompt
    prompt = f"""You are a data analysis assistant in SVEND's Analysis Workbench.

Think step by step about the user's question before answering.

Available tools in the workbench:
- Stat menu: Descriptive Statistics, t-tests, ANOVA, Regression, Correlation, Normality Test, Chi-Square
- ML menu: Classification (Random Forest, XGBoost, SVM), Clustering (K-Means, DBSCAN), PCA, Feature Importance
- SPC menu: I-MR Chart, Xbar-R Chart, Capability Analysis
- Graph menu: Histogram, Boxplot, Scatterplot, Matrix Plot, Time Series, Pareto Chart

DATA CONTEXT:
{data_summary}
{session_history}

USER QUESTION: {message}

First, consider:
1. What is the user really asking?
2. What does the data tell us about this question?
3. Which analyses would provide the most insight?

Then provide a helpful response with specific recommendations. Reference actual column names from the data."""

    return llm_inference(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        use_coder=False,
    )

"""Inference bridge to Cognition-Typed Qwen pipeline."""

from .pipeline import (
    get_pipeline,
    process_query,
    CognitionPipelineManager,
    InferenceResult,
)

__all__ = [
    "get_pipeline",
    "process_query",
    "CognitionPipelineManager",
    "InferenceResult",
]

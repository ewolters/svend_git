"""Inference bridge to Cognition-Typed Qwen pipeline."""

from .pipeline import (
    CognitionPipelineManager,
    InferenceResult,
    get_pipeline,
    process_query,
)

__all__ = [
    "get_pipeline",
    "process_query",
    "CognitionPipelineManager",
    "InferenceResult",
]

"""Bridge to Cognition-Typed Qwen Pipeline.

Uses cognition modes (not domains) for routing:
- ANALYTICAL: Decomposition, causal chains
- SYNTHETIC: Integration, pattern recognition
- CRITICAL: Contradiction detection, verification
- GENERATIVE: Hypothesis formation, exploration
- EXECUTIVE: Code execution via Qwen-Coder

Thompson sampling selects mode based on query signature.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from functools import lru_cache
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from cognition pipeline inference."""

    # Mode selection
    selected_mode: str = ""
    mode_scores: Dict[str, float] = field(default_factory=dict)

    # Query analysis
    query_type: str = ""

    # Output
    response: str = ""
    output_type: str = ""
    confidence: float = 0.0

    # Execution (EXECUTIVE mode)
    success: bool = False
    code: str = ""
    execution_outputs: List[str] = field(default_factory=list)
    execution_errors: List[str] = field(default_factory=list)
    visualizations: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)

    # Timing
    inference_time_ms: float = 0.0

    # Legacy compatibility
    blocked: bool = False
    block_reason: str = ""
    domain: str = ""
    difficulty: float = 0.0
    reasoning_trace: List = field(default_factory=list)
    tool_calls: List = field(default_factory=list)
    verified: bool = False
    verification_confidence: float = 0.0
    final_answer: Any = None
    synara_success: bool = False
    pipeline_type: str = "cognition"

    def to_dict(self) -> dict:
        return {
            "selected_mode": self.selected_mode,
            "mode_scores": self.mode_scores,
            "query_type": self.query_type,
            "response": self.response,
            "output_type": self.output_type,
            "confidence": self.confidence,
            "success": self.success,
            "code": self.code,
            "has_visualizations": len(self.visualizations) > 0,
            "inference_time_ms": self.inference_time_ms,
            "pipeline_type": self.pipeline_type,
        }


class CognitionPipelineManager:
    """Manages the Cognition-Typed Qwen Pipeline.

    Uses Thompson sampling to select cognition modes:
    - ANALYTICAL, SYNTHETIC, CRITICAL, GENERATIVE, EXECUTIVE
    """

    _instance: Optional["CognitionPipelineManager"] = None
    _pipeline = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):
        """Initialize the Cognition pipeline (lazy loading)."""
        if self._initialized:
            return

        try:
            from cognition_qwen import CognitionPipeline, CognitionMode

            workspace = Path(getattr(settings, 'KJERNE_PATH', '.')) / 'cognition_workspace'

            self._pipeline = CognitionPipeline(
                workspace=workspace,
                base_model="Qwen/Qwen2.5-Coder-14B-Instruct",  # Use Coder as base
                enable_execution=True
            )

            # Pre-load the model
            logger.info("Loading Qwen-Coder model...")
            self._pipeline.lora_manager.load_base_model(quantize=True, model_type='coder')

            self._initialized = True
            logger.info("Cognition Pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Cognition pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise

    @property
    def pipeline(self):
        """Get the pipeline, initializing if needed."""
        if not self._initialized:
            self.initialize()
        return self._pipeline

    def process(self, text: str, mode: str = "auto") -> InferenceResult:
        """Process text through the Cognition pipeline."""
        start_time = time.time()

        try:
            from cognition_qwen import CognitionMode

            # Map mode string to CognitionMode
            force_mode = None
            if mode != "auto":
                mode_map = {
                    "analytical": CognitionMode.ANALYTICAL,
                    "synthetic": CognitionMode.SYNTHETIC,
                    "critical": CognitionMode.CRITICAL,
                    "generative": CognitionMode.GENERATIVE,
                    "executive": CognitionMode.EXECUTIVE,
                    "coder": CognitionMode.EXECUTIVE,  # Alias
                }
                force_mode = mode_map.get(mode.lower())

            # Run through pipeline
            trace = self.pipeline.process(
                query=text,
                force_mode=force_mode,
                use_thompson=(mode == "auto")
            )

            inference_time_ms = (time.time() - start_time) * 1000

            # Extract execution results if EXECUTIVE mode
            exec_outputs = []
            exec_errors = []
            visualizations = []
            tools_used = []
            exec_success = True

            if trace.execution_result:
                exec_outputs = trace.execution_result.outputs
                exec_errors = trace.execution_result.errors
                visualizations = trace.execution_result.visualizations
                tools_used = trace.execution_result.tools_used
                exec_success = trace.execution_result.success

            # Build response
            response = trace.output
            if exec_outputs:
                response = "\n".join(exec_outputs)
            elif exec_errors:
                response = f"Execution error: {'; '.join(exec_errors)}"

            return InferenceResult(
                selected_mode=trace.selected_mode.value,
                mode_scores=trace.mode_scores,
                query_type=trace.signature.query_type if trace.signature else "",
                response=response,
                output_type=trace.output_type,
                confidence=trace.confidence,
                success=exec_success,
                code=trace.output if trace.selected_mode.value == "executive" else "",
                execution_outputs=exec_outputs,
                execution_errors=exec_errors,
                visualizations=visualizations,
                tools_used=tools_used,
                inference_time_ms=inference_time_ms,
                # Legacy compatibility
                verified=exec_success,
                verification_confidence=trace.confidence,
                synara_success=exec_success,
                final_answer=response,
                domain=trace.signature.query_type if trace.signature else "",
                pipeline_type="cognition",
            )

        except Exception as e:
            logger.error(f"Cognition pipeline error: {e}")
            import traceback
            traceback.print_exc()
            inference_time_ms = (time.time() - start_time) * 1000
            return InferenceResult(
                blocked=True,
                block_reason=f"Pipeline error: {str(e)}",
                inference_time_ms=inference_time_ms,
                pipeline_type="cognition",
            )


@lru_cache(maxsize=1)
def get_pipeline():
    """Get the Cognition pipeline manager."""
    logger.info("Using Cognition-Typed Qwen Pipeline")
    return CognitionPipelineManager()


def process_query(text: str, mode: str = "auto") -> InferenceResult:
    """Process a query through the Cognition pipeline.

    Args:
        text: The query to process
        mode: "auto" (Thompson sampling), or force a mode:
              "analytical", "synthetic", "critical", "generative", "executive"

    Returns:
        InferenceResult from the cognition pipeline
    """
    return get_pipeline().process(text, mode=mode)

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ExecutionMode(str, Enum):
    DIRECT_ROUTE = "direct_route"
    COMPARE_OR_ESCALATE = "compare_or_escalate"

class ModelExecutionResult(BaseModel):
    """
    Result of an individual model execution attempt.
    """
    model_id: str = Field(..., description="Target model ID")
    provider_name: str = Field(..., description="Provider name")
    success: bool = Field(..., description="Whether execution succeeded")
    content: Optional[str] = Field(default=None, description="Generated response content if successful")
    error_message: Optional[str] = Field(default=None, description="Error message if execution failed")
    latency_ms: float = Field(..., description="Measured execution duration in milliseconds")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Optional raw provider output")

class MultiModelExecutionResult(BaseModel):
    """
    Combined outcome of the multi-model execution and orchestration layer (Phase 8).
    """
    prompt: str = Field(..., description="Evaluated user prompt")
    execution_mode: ExecutionMode = Field(..., description="Mode executed (DIRECT_ROUTE vs COMPARE_OR_ESCALATE)")
    overall_success: bool = Field(..., description="True if at least one candidate execution succeeded")
    executions: List[ModelExecutionResult] = Field(default_factory=list, description="All model execution results")
    status_message: str = Field(..., description="Human-readable summary of execution outcome")

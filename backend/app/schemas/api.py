from pydantic import BaseModel, Field
from typing import Optional
from app.schemas.profile import PromptProfile
from app.schemas.routing import RoutingResult, ConfidenceGapAnalysis
from app.schemas.execution import MultiModelExecutionResult
from app.schemas.quality import QualityGateResult
from app.schemas.memory import AdaptiveRoutingSignal
from app.schemas.trace import ExplainableDecisionTrace

class RouteRequest(BaseModel):
    prompt: str = Field(..., description="The user prompt to analyze, route, and execute")

class RouteResponse(BaseModel):
    status: str = Field(default="success", description="Response status (e.g., success, error)")
    prompt: str = Field(..., description="The original user prompt")
    profile: Optional[PromptProfile] = None
    routing_result: Optional[RoutingResult] = None
    confidence_gap_analysis: Optional[ConfidenceGapAnalysis] = None
    execution_result: Optional[MultiModelExecutionResult] = None
    quality_result: Optional[QualityGateResult] = None
    adaptive_signal: Optional[AdaptiveRoutingSignal] = None
    trace: Optional[ExplainableDecisionTrace] = None
    final_selected_model: Optional[str] = None
    final_selected_content: Optional[str] = None
    error_message: Optional[str] = None

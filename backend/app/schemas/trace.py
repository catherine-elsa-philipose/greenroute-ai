from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.profile import PromptProfile
from app.schemas.routing import RoutingResult, CandidateScore, ConfidenceGapAnalysis
from app.schemas.execution import MultiModelExecutionResult, ModelExecutionResult
from app.schemas.quality import QualityGateResult, ResponseQualityEvaluation
from app.schemas.memory import AdaptiveRoutingSignal

class ExplainableDecisionTrace(BaseModel):
    """
    Complete, human-readable Explainable Decision Trace (Phase 11).
    Synthesizes facts from profiling, routing, confidence gap analysis, multi-model execution,
    response quality evaluation, and historical outcome memory.
    """
    trace_id: str = Field(..., description="Unique UUID for this decision trace")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO creation timestamp")
    
    # 1. User Prompt & Profiler output
    profile: PromptProfile = Field(..., description="Structured output from Semantic Prompt Intelligence Profiler")
    
    # 2. Router & Confidence Gap output
    routing_result: RoutingResult = Field(..., description="Outcome from Confidence-Aware Router")
    confidence_gap_analysis: Optional[ConfidenceGapAnalysis] = Field(default=None, description="Detailed confidence gap analysis")
    
    # 3. Execution outcome
    execution_result: MultiModelExecutionResult = Field(..., description="Outcome from MultiModelOrchestrator")
    
    # 4. Quality evaluation outcome
    quality_result: QualityGateResult = Field(..., description="Outcome from ResponseQualityGate")
    
    # 5. Optional Adaptive Historical Memory Signal
    adaptive_signal: Optional[AdaptiveRoutingSignal] = Field(default=None, description="Historical performance signal if available")
    
    # 6. Final Decision & Human-Readable Explanation
    final_selected_model_id: Optional[str] = Field(default=None, description="Final model selected for user output")
    final_selected_content: Optional[str] = Field(default=None, description="Final response content returned")
    decision_summary: str = Field(..., description="Structured, transparent, human-readable summary explanation")

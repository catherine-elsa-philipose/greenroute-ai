from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class CandidateScore(BaseModel):
    """
    Detailed scoring breakdown for an individual candidate model evaluated by the router.
    """
    model_id: str = Field(..., description="Unique identifier of the candidate model")
    provider_name: str = Field(..., description="Provider name")
    is_available: bool = Field(..., description="Whether the model was available for selection")
    capability_fit_score: float = Field(..., description="Normalized score (0.0 to 1.0) for prompt requirement fit")
    operational_score: float = Field(..., description="Normalized score (0.0 to 1.0) for speed, reliability, and cost")
    total_score: float = Field(..., description="Final weighted suitability score (0.0 to 1.0)")
    scoring_details: Dict[str, float] = Field(default_factory=dict, description="Detailed sub-scores breakdown")

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RoutingActionRecommendation(str, Enum):
    DIRECT_ROUTE = "direct_route"
    COMPARE_OR_ESCALATE = "compare_or_escalate"
    NO_MODEL_AVAILABLE = "no_model_available"

class ConfidenceGapAnalysis(BaseModel):
    """
    Semantic Confidence Gap Analysis outcome (Phase 7).
    """
    top_model_id: Optional[str] = Field(default=None, description="ID of the highest-scoring candidate")
    top_score: float = Field(default=0.0, description="Score of the highest-scoring candidate")
    second_model_id: Optional[str] = Field(default=None, description="ID of the second highest-scoring candidate, if present")
    second_score: Optional[float] = Field(default=None, description="Score of the second highest-scoring candidate, if present")
    confidence_gap: float = Field(default=0.0, description="Calculated difference: top_score - second_score (0.0 if only 1 candidate)")
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.LOW, description="Categorized confidence level indicator")
    eligible_candidate_count: int = Field(default=0, description="Number of eligible (available) candidate models evaluated")
    recommendation: RoutingActionRecommendation = Field(..., description="Action recommendation (e.g. direct_route vs compare_or_escalate)")
    explanation: str = Field(..., description="Human-readable explanation of the confidence gap analysis")

class RoutingResult(BaseModel):
    """
    Structured outcome returned by the Confidence-Aware Router.
    """
    selected_model_id: Optional[str] = Field(default=None, description="ID of the highest-scoring available candidate model, if any")
    selected_provider_name: Optional[str] = Field(default=None, description="Provider of the selected model")
    top_score: float = Field(default=0.0, description="Suitability score of the selected model")
    candidate_scores: List[CandidateScore] = Field(default_factory=list, description="All evaluated candidate model scores")
    no_available_model: bool = Field(default=False, description="Flag indicating no available models matched")
    confidence_gap_analysis: Optional[ConfidenceGapAnalysis] = Field(default=None, description="Phase 7 Confidence Gap evaluation")
    status_message: str = Field(default="Routing complete", description="Human-readable routing status summary")

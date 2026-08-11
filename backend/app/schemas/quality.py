from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class QualityLabel(str, Enum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    WEAK = "weak"

class ResponseQualityEvaluation(BaseModel):
    """
    Quality evaluation breakdown for an individual model response (Phase 9).
    """
    model_id: str = Field(..., description="ID of the model evaluated")
    provider_name: str = Field(..., description="Provider of the model")
    relevance_score: float = Field(..., description="Relevance score (0.0 to 1.0)")
    completeness_score: float = Field(..., description="Completeness score (0.0 to 1.0)")
    compliance_score: float = Field(..., description="Instruction compliance score (0.0 to 1.0)")
    consistency_score: float = Field(..., description="Consistency score (0.0 to 1.0)")
    overall_quality_score: float = Field(..., description="Weighted total quality score (0.0 to 1.0)")
    quality_label: QualityLabel = Field(..., description="Quality classification label (GOOD, ACCEPTABLE, WEAK)")
    evaluation_notes: str = Field(..., description="Transparent notes explaining the evaluation breakdown")

class QualityGateResult(BaseModel):
    """
    Combined outcome of the Response Quality Gate service.
    """
    evaluations: List[ResponseQualityEvaluation] = Field(default_factory=list, description="Quality evaluations for all successful responses")
    selected_model_id: Optional[str] = Field(default=None, description="ID of the highest-scoring evaluated model")
    selected_content: Optional[str] = Field(default=None, description="Content of the selected response")
    selected_quality_score: float = Field(default=0.0, description="Quality score of the selected response")
    is_weak_response: bool = Field(default=False, description="True if selected response falls below acceptable threshold")
    recommendation: str = Field(default="ACCEPT", description="Action recommendation (ACCEPT vs RE_EVALUATE_OR_FALLBACK)")
    status_message: str = Field(..., description="Summary of quality gate outcome")

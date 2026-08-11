from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from app.schemas.profile import TaskType, Domain
from app.schemas.quality import QualityLabel

class ExecutionOutcomeRecord(BaseModel):
    """
    Structured outcome event recorded in Adaptive Outcome Memory (Phase 10).
    """
    outcome_id: str = Field(..., description="Unique UUID of the outcome record")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO timestamp of execution")
    task_type: TaskType = Field(..., description="Categorized task type")
    domain: Domain = Field(..., description="Categorized domain")
    model_id: str = Field(..., description="ID of the model executed")
    provider_name: str = Field(..., description="Provider name")
    routing_score: float = Field(..., description="Initial routing suitability score (0.0 to 1.0)")
    routing_confidence: str = Field(..., description="Confidence level label (high, medium, low)")
    confidence_gap: float = Field(..., description="Semantic confidence gap score")
    overall_quality_score: float = Field(..., description="Evaluated response quality score (0.0 to 1.0)")
    quality_label: QualityLabel = Field(..., description="Quality label (GOOD, ACCEPTABLE, WEAK)")
    latency_ms: float = Field(..., description="Measured execution duration in ms")
    estimated_cost: float = Field(default=0.0, description="Estimated token cost")
    success: bool = Field(..., description="Whether execution succeeded")
    is_fallback: bool = Field(default=False, description="True if response was flagged weak or required fallback")

class HistoricalModelStats(BaseModel):
    """
    Aggregated performance statistics calculated from recorded outcome records.
    """
    model_id: str = Field(..., description="Model ID")
    total_attempts: int = Field(..., description="Total execution attempts recorded")
    successful_attempts: int = Field(..., description="Count of successful executions")
    success_rate: float = Field(..., description="Ratio of successful executions (0.0 to 1.0)")
    average_quality_score: float = Field(..., description="Mean quality score across all recorded outcomes")
    average_latency_ms: float = Field(..., description="Mean latency in milliseconds")
    average_cost: float = Field(..., description="Mean estimated cost")
    fallback_rate: float = Field(..., description="Ratio of outcomes resulting in fallback/weak label")

class AdaptiveRoutingSignal(BaseModel):
    """
    Historical performance/preference signal exposed to future router iterations.
    """
    model_id: str = Field(..., description="Model ID evaluated")
    task_type: Optional[TaskType] = Field(default=None, description="Filtered task type, if specified")
    domain: Optional[Domain] = Field(default=None, description="Filtered domain, if specified")
    historical_quality_score: Optional[float] = Field(default=None, description="Historical average quality (None if no history)")
    historical_success_rate: Optional[float] = Field(default=None, description="Historical success rate (None if no history)")
    historical_fallback_rate: float = Field(..., description="Historical fallback rate (0.0 if no history)")
    adaptive_weight_multiplier: float = Field(..., description="Calculated preference multiplier (0.5 to 1.5)")
    total_samples: int = Field(..., description="Number of historical samples evaluated")

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ModelCapabilities(BaseModel):
    """
    Configurable capability metadata for an LLM model/provider in GreenRoute AI.
    All capability strength scores are normalized between 0.0 and 1.0.
    
    Note: These initial values represent configured metadata. They can be updated or replaced
    with empirical benchmark metrics or adaptive historical learning in later phases.
    """
    coding_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Coding task capability score (0.0 to 1.0)")
    reasoning_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Logical reasoning capability score (0.0 to 1.0)")
    math_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Mathematical capability score (0.0 to 1.0)")
    multilingual_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Multilingual translation strength (0.0 to 1.0)")
    speed_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Relative speed/latency score (0.0=slow, 1.0=fastest)")
    reliability_score: float = Field(default=0.9, ge=0.0, le=1.0, description="Historical uptime/reliability score (0.0 to 1.0)")
    
    estimated_cost_per_1k_tokens: float = Field(default=0.002, ge=0.0, description="Estimated cost in USD per 1,000 tokens")
    max_tokens: int = Field(default=4096, gt=0, description="Maximum context window or max output tokens")
    supports_streaming: bool = Field(default=False, description="Whether model supports streaming")
    supports_json: bool = Field(default=True, description="Whether model supports structured JSON output")
    extra_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")

class ModelProfile(BaseModel):
    """
    Complete capability registration profile for a model.
    """
    model_id: str = Field(..., description="Unique model identifier (e.g. 'mock-model-a', 'gpt-4o')")
    provider_name: str = Field(..., description="Provider identifier (e.g. 'mock', 'openai', 'anthropic')")
    is_available: bool = Field(default=True, description="Current operational availability flag")
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities, description="Capability scores & operational specs")

class ProviderResponse(BaseModel):
    """Standardized response object returned by any LLM provider adapter."""
    provider_name: str = Field(..., description="Name of the provider (e.g. mock, openai, anthropic)")
    model_name: str = Field(..., description="Specific model ID or name")
    content: str = Field(..., description="Generated text response")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Optional raw provider output")

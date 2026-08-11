import os
from fastapi import Request

from app.profiler.profiler import PromptProfiler
from app.registry.capability_registry import CapabilityRegistry
from app.schemas.provider import ModelProfile, ModelCapabilities
from app.schemas.profile import TaskType, Domain
from app.router.router import ConfidenceAwareRouter
from app.services.orchestrator import MultiModelOrchestrator
from app.quality.quality_gate import ResponseQualityGate
from app.memory.outcome_memory import AdaptiveOutcomeMemory
from app.services.trace_generator import DecisionTraceGenerator

def get_db_path() -> str:
    """Return the path to the persistent SQLite database, creating the directory if needed."""
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "greenroute.db")

# We instantiate singletons for the application lifecycle to avoid rebuilding on every request.
_profiler = PromptProfiler()
_registry = CapabilityRegistry()
_router = ConfidenceAwareRouter()
_orchestrator = MultiModelOrchestrator()
_quality_gate = ResponseQualityGate()
_memory = AdaptiveOutcomeMemory(db_path=get_db_path())
_trace_generator = DecisionTraceGenerator()

from app.registry.mock_provider import MockLLMProvider
from app.registry.openai_provider import OpenAIProvider
from app.registry.gemini_provider import GeminiProvider
from app.registry.base import BaseLLMProvider
from typing import Dict

# Instantiate providers (API keys will be checked at runtime by the provider's is_available())
_providers: Dict[str, BaseLLMProvider] = {
    "coder-v1": MockLLMProvider("coder-v1"),
    "reasoner-v1": MockLLMProvider("reasoner-v1"),
    "multilingual-v1": MockLLMProvider("multilingual-v1"),
    "gpt-4o-mini": OpenAIProvider(model_name="gpt-4o-mini"),
    "gemini-2.5-flash": GeminiProvider(model_name="gemini-2.5-flash")
}

# Seed registry with fictional mock models for the demo
def _seed_registry():
    if not _registry.list_models():
        _registry.register_model(ModelProfile(
            model_id="coder-v1",
            provider_name="mock",
            is_available=_providers["coder-v1"].is_available(),
            capabilities=ModelCapabilities(
                supported_tasks=[TaskType.CODING],
                supported_domains=[Domain.COMPUTER_SCIENCE, Domain.GENERAL],
                max_complexity=0.9,
                reasoning_strength=0.8
            ),
            cost_per_1k_tokens=0.02,
            latency_ms_per_token=15,
            reliability_score=0.95
        ))
        _registry.register_model(ModelProfile(
            model_id="reasoner-v1",
            provider_name="mock",
            is_available=_providers["reasoner-v1"].is_available(),
            capabilities=ModelCapabilities(
                supported_tasks=[TaskType.REASONING, TaskType.GENERAL_QA],
                supported_domains=[Domain.GENERAL, Domain.MATHEMATICS],
                max_complexity=1.0,
                reasoning_strength=1.0
            ),
            cost_per_1k_tokens=0.05,
            latency_ms_per_token=25,
            reliability_score=0.98
        ))
        _registry.register_model(ModelProfile(
            model_id="multilingual-v1",
            provider_name="mock",
            is_available=_providers["multilingual-v1"].is_available(),
            capabilities=ModelCapabilities(
                supported_tasks=[TaskType.TRANSLATION, TaskType.GENERAL_QA],
                supported_domains=[Domain.LANGUAGE, Domain.GENERAL],
                max_complexity=0.7,
                reasoning_strength=0.6
            ),
            cost_per_1k_tokens=0.01,
            latency_ms_per_token=10,
            reliability_score=0.90
        ))
        _registry.register_model(ModelProfile(
            model_id="gpt-4o-mini",
            provider_name="openai",
            is_available=_providers["gpt-4o-mini"].is_available(),
            capabilities=ModelCapabilities(
                supported_tasks=[TaskType.CODING, TaskType.REASONING, TaskType.GENERAL_QA, TaskType.TRANSLATION],
                supported_domains=[Domain.COMPUTER_SCIENCE, Domain.MATHEMATICS, Domain.GENERAL, Domain.LANGUAGE],
                max_complexity=0.95,
                reasoning_strength=0.9
            ),
            cost_per_1k_tokens=0.15,
            latency_ms_per_token=20,
            reliability_score=0.99
        ))
        _registry.register_model(ModelProfile(
            model_id="gemini-2.5-flash",
            provider_name="google",
            is_available=_providers["gemini-2.5-flash"].is_available(),
            capabilities=ModelCapabilities(
                supported_tasks=[TaskType.CODING, TaskType.GENERAL_QA, TaskType.TRANSLATION],
                supported_domains=[Domain.GENERAL, Domain.LANGUAGE, Domain.COMPUTER_SCIENCE],
                max_complexity=0.85,
                reasoning_strength=0.8
            ),
            cost_per_1k_tokens=0.075,
            latency_ms_per_token=12,
            reliability_score=0.98
        ))

_seed_registry()

def get_profiler() -> PromptProfiler:
    return _profiler

def get_registry() -> CapabilityRegistry:
    return _registry

def get_router() -> ConfidenceAwareRouter:
    return _router

def get_orchestrator() -> MultiModelOrchestrator:
    return _orchestrator

# This is now defined above

def get_providers() -> Dict[str, BaseLLMProvider]:
    return _providers

def get_quality_gate() -> ResponseQualityGate:
    return _quality_gate

def get_memory() -> AdaptiveOutcomeMemory:
    return _memory

def get_trace_generator() -> DecisionTraceGenerator:
    return _trace_generator

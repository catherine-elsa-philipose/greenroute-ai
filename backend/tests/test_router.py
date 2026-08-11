import pytest
from app.profiler.profiler import PromptProfiler
from app.registry.capability_registry import CapabilityRegistry
from app.schemas.provider import ModelProfile, ModelCapabilities
from app.router.router import ConfidenceAwareRouter

@pytest.fixture
def populated_registry():
    registry = CapabilityRegistry()
    
    # Model 1: Coding specialist
    m_coding = ModelProfile(
        model_id="code-specialist-v1",
        provider_name="mock_provider",
        is_available=True,
        capabilities=ModelCapabilities(
            coding_strength=0.95,
            reasoning_strength=0.6,
            math_strength=0.5,
            multilingual_strength=0.4,
            speed_score=0.7,
            reliability_score=0.9
        )
    )
    
    # Model 2: Reasoning specialist
    m_reasoning = ModelProfile(
        model_id="reasoning-specialist-v1",
        provider_name="mock_provider",
        is_available=True,
        capabilities=ModelCapabilities(
            coding_strength=0.5,
            reasoning_strength=0.95,
            math_strength=0.7,
            multilingual_strength=0.5,
            speed_score=0.6,
            reliability_score=0.95
        )
    )
    
    # Model 3: Math specialist
    m_math = ModelProfile(
        model_id="math-specialist-v1",
        provider_name="mock_provider",
        is_available=True,
        capabilities=ModelCapabilities(
            coding_strength=0.4,
            reasoning_strength=0.7,
            math_strength=0.98,
            multilingual_strength=0.3,
            speed_score=0.8,
            reliability_score=0.9
        )
    )

    # Model 4: Multilingual specialist
    m_lang = ModelProfile(
        model_id="lang-specialist-v1",
        provider_name="mock_provider",
        is_available=True,
        capabilities=ModelCapabilities(
            coding_strength=0.3,
            reasoning_strength=0.5,
            math_strength=0.4,
            multilingual_strength=0.95,
            speed_score=0.85,
            reliability_score=0.9
        )
    )

    registry.register_model(m_coding)
    registry.register_model(m_reasoning)
    registry.register_model(m_math)
    registry.register_model(m_lang)
    return registry

def test_coding_prompt_selects_coding_specialist(populated_registry):
    profiler = PromptProfiler()
    router = ConfidenceAwareRouter()
    
    profile = profiler.analyze("Debug this Python recursion function.")
    result = router.route(profile, populated_registry)
    
    assert result.no_available_model is False
    assert result.selected_model_id == "code-specialist-v1"

def test_reasoning_prompt_selects_reasoning_specialist(populated_registry):
    profiler = PromptProfiler()
    router = ConfidenceAwareRouter()
    
    profile = profiler.analyze("Compare these two algorithms and explain their tradeoffs.")
    result = router.route(profile, populated_registry)
    
    assert result.no_available_model is False
    assert result.selected_model_id == "reasoning-specialist-v1"

def test_math_prompt_selects_math_specialist(populated_registry):
    profiler = PromptProfiler()
    router = ConfidenceAwareRouter()
    
    profile = profiler.analyze("Solve this math calculus problem.")
    result = router.route(profile, populated_registry)
    
    assert result.no_available_model is False
    assert result.selected_model_id == "math-specialist-v1"

def test_translation_prompt_selects_lang_specialist(populated_registry):
    profiler = PromptProfiler()
    router = ConfidenceAwareRouter()
    
    profile = profiler.analyze("Translate this text into French.")
    result = router.route(profile, populated_registry)
    
    assert result.no_available_model is False
    assert result.selected_model_id == "lang-specialist-v1"

def test_unavailable_model_is_never_selected():
    registry = CapabilityRegistry()
    
    # Fast coding model but unavailable
    m_unavail = ModelProfile(
        model_id="super-coder-offline",
        provider_name="mock",
        is_available=False,
        capabilities=ModelCapabilities(coding_strength=1.0)
    )
    
    # Moderate coding model and available
    m_avail = ModelProfile(
        model_id="moderate-coder-online",
        provider_name="mock",
        is_available=True,
        capabilities=ModelCapabilities(coding_strength=0.6)
    )
    
    registry.register_model(m_unavail)
    registry.register_model(m_avail)
    
    profiler = PromptProfiler()
    router = ConfidenceAwareRouter()
    
    profile = profiler.analyze("Write a Python script.")
    result = router.route(profile, registry)
    
    assert result.selected_model_id == "moderate-coder-online"

def test_no_available_models_handled_gracefully():
    registry = CapabilityRegistry()
    m_offline = ModelProfile(model_id="offline-model", provider_name="mock", is_available=False)
    registry.register_model(m_offline)
    
    profiler = PromptProfiler()
    router = ConfidenceAwareRouter()
    
    profile = profiler.analyze("Hello")
    result = router.route(profile, registry)
    
    assert result.no_available_model is True
    assert result.selected_model_id is None
    assert "No registered models are currently available" in result.status_message

def test_deterministic_scoring_reproducibility(populated_registry):
    profiler = PromptProfiler()
    router = ConfidenceAwareRouter()
    
    profile = profiler.analyze("Debug recursion")
    res1 = router.route(profile, populated_registry)
    res2 = router.route(profile, populated_registry)
    
    assert res1.selected_model_id == res2.selected_model_id
    assert res1.top_score == res2.top_score
